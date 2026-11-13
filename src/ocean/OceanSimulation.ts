import * as THREE from 'three/webgpu';
import { Fn, float, int, ivec2, mix, select, texture, textureLoad, uniform, uv, vec2, vec4 } from 'three/tsl';
import { butterflyPassNode, cMul, createFFTResources, type FFTResources } from './FFT';
import {
  CASCADES,
  createSpectrumTexture,
  generateInitialSpectrum,
  GRAVITY,
  type CascadeConfig,
  type SpectrumParams,
} from './Spectrum';

/**
 * Spectral ocean surface.
 *
 * Per cascade and per frame:
 *   1. evolve   h0(k) -> h(k, t) and derive four packed complex spectra
 *   2. IFFT     log2(N) horizontal + log2(N) vertical butterfly passes, x2 pairs
 *   3. assemble unpack into a displacement texture and a derivative texture
 *
 * Field packing across the two complex slots of two RGBA ping-pong pairs
 * (8 real fields, all used):
 *
 *   A.c0 -> real: Dy (height)          imag: dDx/dz
 *   A.c1 -> real: Dx                   imag: Dz
 *   B.c0 -> real: dDy/dx               imag: dDy/dz
 *   B.c1 -> real: dDx/dx               imag: dDz/dz
 *
 * Packing two real fields into one complex transform halves the pass count: an
 * IFFT of (P + iQ) for real P and Q returns P in the real component and Q in the
 * imaginary one. The cross term dDx/dz is carried so the foam mask can use a
 * true Jacobian rather than the usual diagonal-only approximation, which
 * under-reports folding on obliquely travelling crests.
 */

interface PingPong {
  read: THREE.RenderTarget;
  write: THREE.RenderTarget;
}

/**
 * What remains per cascade once the transform is shared.
 *
 * The spectra, the evolution and the butterfly all live on the simulation now,
 * in one atlas — see `makePingPong`. A cascade keeps only what is genuinely its
 * own: its band configuration, its two output textures, and the two passes that
 * unpack its slice of the atlas into them.
 */
interface Cascade {
  config: CascadeConfig;
  /** RGBA: xyz = displacement, w = foam/folding mask. */
  displacement: THREE.RenderTarget;
  /** RG = surface slope, B = Jacobian, A = |slope|^2 for variance recovery. */
  derivatives: THREE.RenderTarget;
  assembleDisplacement: THREE.NodeMaterial;
  assembleDerivatives: THREE.NodeMaterial;
}

/**
 * Height-field binding slots, fixed so a tier change re-points rather than
 * rebuilds. Matches `MAX_CASCADES` in `OceanMaterial`.
 */
const HEIGHT_SLOTS = 3;

export interface OceanSimulationOptions {
  size: 128 | 256 | 512;
  cascadeCount: 1 | 2 | 3;
  params: SpectrumParams;
}

export class OceanSimulation {
  private readonly renderer: THREE.WebGPURenderer;
  private readonly quad = new THREE.QuadMesh(new THREE.NodeMaterial());
  private readonly cascades: Cascade[] = [];
  private fft: FFTResources;

  /**
   * The whole simulation's spectra, in one atlas, and the passes that step it.
   *
   * Every cascade runs the identical butterfly at the identical size, and the
   * butterfly reads nothing cascade-specific — only the ping-pong texture and
   * the twiddle table. Running it three times therefore paid the per-pass cost
   * three times for one pass of work. One atlas `2 * cascadeCount` slots wide
   * carries all of it: slot `2c` is cascade c's pair A and slot `2c + 1` its
   * pair B.
   */
  private h0!: THREE.DataTexture;
  private pair!: PingPong;
  private evolve!: THREE.NodeMaterial;
  private butterfly: THREE.NodeMaterial[] = [];

  private readonly uTime = uniform(0);
  private readonly uStage = uniform(0);
  private readonly uChoppiness = uniform(1.05);
  /**
   * Travelling-to-standing blend, mirroring `SpectrumParams.standingWaveRatio`.
   *
   * A uniform rather than a rebuild: the spectrum half of this parameter needs
   * `h0` regenerating and the motion half does not, so `updateSpectrum` writes
   * both and the shader never recompiles.
   */
  private readonly uStandingRatio = uniform(0);
  /**
   * Amplitude compensation for the standing blend: `1 / sqrt(1 - r + r^2/2)`.
   *
   * The locked form carries half the time-averaged energy of the travelling one
   * for the same `h0`, so without this the standing control would quietly flatten
   * the sea as it was turned up — a sheltered bay would be sheltered *and*
   * smaller, and the two are separate statements. Energy goes as amplitude
   * squared, hence the square root.
   */
  private readonly uStandingGain = uniform(1);

  private size: OceanSimulationOptions['size'];
  private cascadeCount: OceanSimulationOptions['cascadeCount'];
  private params: SpectrumParams;
  private disposed = false;

  constructor(renderer: THREE.WebGPURenderer, options: OceanSimulationOptions) {
    this.renderer = renderer;
    this.size = options.size;
    this.cascadeCount = options.cascadeCount;
    this.params = { ...options.params };
    // Here as well as in `updateSpectrum`, because a caller can construct with a
    // non-zero standing ratio and then never call the setter — which would leave
    // the spectrum broadened for standing water while the surface still ran.
    this.applyStandingRatio();
    this.fft = createFFTResources(this.size);
    this.build();
  }

  /** Derives both standing-wave uniforms from `params.standingWaveRatio`. */
  private applyStandingRatio(): void {
    const ratio = Math.max(0, Math.min(1, this.params.standingWaveRatio));
    this.uStandingRatio.value = ratio;
    this.uStandingGain.value = 1 / Math.sqrt(1 - ratio + (ratio * ratio) / 2);
  }

  // ---------------------------------------------------------------- public API

  /** Displacement textures, one per active cascade, for the surface material. */
  /**
   * Surface elevation at a world XZ, as a TSL function.
   *
   * The same displacement fields the ocean mesh is built from, summed over the
   * cascades — so a consumer asking "where is the water here" gets the answer the
   * viewer can see rather than a plane at sea level.
   *
   * Deliberately vertical-only. The horizontal (choppy) components displace a
   * vertex sideways as well as up, so the true surface is not a heightfield and
   * inverting it needs iteration. For finding where an eye ray crosses the
   * surface, the vertical term carries essentially all of the answer and the
   * lateral error is a fraction of a wavelength.
   *
   * Safe to bake into a node graph and keep. The bindings are fixed slots that
   * `resize` re-points, exactly as `OceanMaterial.setCascades` does — a consumer
   * that rebuilt its graph after every tier change would be recompiling a shader
   * mid-session, which is the thing the whole cascade-slot arrangement exists to
   * avoid.
   */
  heightNode(): (worldXZ: any) => any {
    this.ensureHeightBindings();
    const nodes = this.heightNodes;
    const tiles = this.heightTiles;
    const weights = this.heightWeights;
    return (worldXZ: any) => {
      let sum: any = float(0);
      for (let i = 0; i < nodes.length; i++) {
        sum = sum.add(nodes[i].sample(vec2(worldXZ).div(tiles[i])).y.mul(weights[i]));
      }
      return sum;
    };
  }

  /** Creates the fixed height-field slots on first use, and points them. */
  private ensureHeightBindings(): void {
    if (this.heightNodes.length === 0) {
      for (let i = 0; i < HEIGHT_SLOTS; i++) {
        const source = Math.min(i, this.displacementTextures.length - 1);
        this.heightNodes.push(texture(this.displacementTextures[source]) as any);
        this.heightTiles.push(uniform(this.tileSizes[source]));
        this.heightWeights.push(uniform(i < this.displacementTextures.length ? 1 : 0));
      }
      return;
    }
    this.repointHeightBindings();
  }

  /** Re-points the height slots at the current targets. Called after `resize`. */
  private repointHeightBindings(): void {
    if (this.heightNodes.length === 0) return;
    const textures = this.displacementTextures;
    const active = Math.min(textures.length, HEIGHT_SLOTS);
    for (let i = 0; i < HEIGHT_SLOTS; i++) {
      const source = Math.min(i, active - 1);
      this.heightNodes[i].value = textures[source];
      this.heightTiles[i].value = this.tileSizes[source];
      this.heightWeights[i].value = i < active ? 1 : 0;
    }
  }

  // Fixed height-field slots, re-pointed rather than rebuilt. See `heightNode`.
  private readonly heightNodes: any[] = [];
  private readonly heightTiles: any[] = [];
  private readonly heightWeights: any[] = [];

  get displacementTextures(): THREE.Texture[] {
    return this.cascades.map((c) => c.displacement.texture);
  }

  get derivativeTextures(): THREE.Texture[] {
    return this.cascades.map((c) => c.derivatives.texture);
  }

  /** Render targets themselves, for CPU readback by the buoyancy sampler. */
  get displacementTargets(): THREE.RenderTarget[] {
    return this.cascades.map((c) => c.displacement);
  }

  get tileSizes(): number[] {
    return this.cascades.map((c) => c.config.tileSize);
  }

  /**
   * Texels per side of every cascade's output.
   *
   * Published because a tile size alone does not say how finely the field is
   * sampled, and the surface needs metres per texel to choose a mip level to
   * displace geometry from. See `ocean/meshSampling`.
   */
  get resolution(): number {
    return this.size;
  }

  /** Longest wavelength each cascade carries, metres. See `cascadeReach`. */
  get maxWavelengths(): number[] {
    return this.cascades.map((c) => c.config.maxWavelength);
  }

  /**
   * Everything the surface needs to know about the cascade set, as one object.
   *
   * The surface takes this both when its graph is built and again after a tier
   * change re-points it. Handing over one object rather than a list of
   * positional arguments is what stops the two sites drifting — the graph was
   * once built from inputs one of which the re-point silently never supplied.
   */
  get fields(): {
    displacementTextures: THREE.Texture[];
    derivativeTextures: THREE.Texture[];
    tileSizes: number[];
    maxWavelengths: number[];
    resolution: number;
  } {
    return {
      displacementTextures: this.displacementTextures,
      derivativeTextures: this.derivativeTextures,
      tileSizes: this.tileSizes,
      maxWavelengths: this.maxWavelengths,
      resolution: this.resolution,
    };
  }

  get activeCascadeCount(): number {
    return this.cascades.length;
  }

  get spectrumParams(): Readonly<SpectrumParams> {
    return this.params;
  }

  setChoppiness(value: number): void {
    this.uChoppiness.value = value;
  }


  /** Re-derives h0 from new wind/wavelength. Cheap enough to call on slider input. */
  updateSpectrum(params: Partial<SpectrumParams>): void {
    this.params = { ...this.params, ...params };
    // The motion half of the standing blend. Kept in step here rather than in a
    // setter of its own so the two halves cannot be written independently — a
    // spectrum broadened for standing water while the surface still travels is
    // the one incoherent state this parameter has.
    this.applyStandingRatio();
    this.writeSpectra();
  }

  /** Rebuilds all GPU resources at a new resolution / cascade count. */
  resize(
    size: OceanSimulationOptions['size'],
    cascadeCount: OceanSimulationOptions['cascadeCount'],
  ): void {
    if (size === this.size && cascadeCount === this.cascadeCount) return;
    this.releaseCascades();
    this.size = size;
    this.cascadeCount = cascadeCount;
    this.fft.butterfly.dispose();
    this.fft = createFFTResources(size);
    this.build();
    this.repointHeightBindings();
  }

  update(elapsed: number): void {
    if (this.disposed) return;
    this.uTime.value = elapsed;

    const previousTarget = this.renderer.getRenderTarget();

    // One evolution and one transform for every cascade at once; only the
    // unpacking is per cascade, because only it writes somewhere different.
    this.runPass(this.evolve, this.pair.read);
    this.runFFT(this.pair, this.butterfly);

    for (const cascade of this.cascades) {
      this.runPass(cascade.assembleDisplacement, cascade.displacement);
      this.runPass(cascade.assembleDerivatives, cascade.derivatives);
    }

    this.renderer.setRenderTarget(previousTarget);
  }

  dispose(): void {
    this.disposed = true;
    this.releaseCascades();
    this.fft.butterfly.dispose();
    this.quad.geometry.dispose();
  }

  // ------------------------------------------------------------------ internals

  private runPass(material: THREE.NodeMaterial, target: THREE.RenderTarget): void {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    this.quad.render(this.renderer);
  }

  /**
   * Runs 2*stages butterfly passes, swapping the ping-pong after each. Because
   * the count is always even the pair ends the frame in the orientation it
   * started in, which is what lets the stage materials bind their source texture
   * once at build time (even stages read `read`, odd stages read `write`) and
   * lets the assemble pass bind `read` permanently.
   */
  private runFFT(pair: PingPong, materials: THREE.NodeMaterial[]): void {
    for (let stage = 0; stage < materials.length; stage++) {
      this.uStage.value = stage % this.fft.stages;
      this.runPass(materials[stage], pair.write);
      const swap = pair.read;
      pair.read = pair.write;
      pair.write = swap;
    }
  }

  private build(): void {
    const configs = CASCADES.slice(0, this.cascadeCount);
    const size = this.size;

    // Shared across every cascade: one spectrum atlas, one ping-pong, one
    // evolution and one transform.
    this.h0 = createSpectrumTexture(size, this.cascadeCount, new Float32Array(
      size * size * 4 * this.cascadeCount,
    ));
    this.writeSpectra();
    this.pair = makePingPong(size, this.cascadeCount);
    this.evolve = this.createEvolveMaterial(configs);
    this.butterfly = this.createButterflyMaterials(this.pair);

    for (let i = 0; i < configs.length; i++) {
      this.cascades.push(this.buildCascade(configs[i], i));
    }
  }

  /**
   * (Re)generates every cascade's initial spectrum into its column of the atlas.
   *
   * The atlas is `cascadeCount * size` wide, so a cascade's rows are not
   * contiguous and this copies row by row. Each cascade keeps its own seed, so
   * the fields stay uncorrelated exactly as they were when they had separate
   * textures.
   */
  private writeSpectra(): void {
    const size = this.size;
    const atlas = this.h0.image.data as Float32Array;
    const configs = CASCADES.slice(0, this.cascadeCount);
    for (let c = 0; c < configs.length; c++) {
      const data = generateInitialSpectrum(size, configs[c], this.params, 1337 + c * 977);
      for (let y = 0; y < size; y++) {
        atlas.set(
          data.subarray(y * size * 4, (y + 1) * size * 4),
          (y * size * this.cascadeCount + c * size) * 4,
        );
      }
    }
    this.h0.needsUpdate = true;
  }

  private buildCascade(config: CascadeConfig, index: number): Cascade {
    const size = this.size;
    return {
      config,
      displacement: makeOutputTarget(size),
      derivatives: makeOutputTarget(size),
      assembleDisplacement: this.createAssembleMaterial(index, config, 'displacement'),
      assembleDerivatives: this.createAssembleMaterial(index, config, 'derivatives'),
    };
  }

  /**
   * One material per butterfly stage, built once — node graph construction is
   * expensive and must never happen inside the frame loop.
   */
  private createButterflyMaterials(pair: PingPong): THREE.NodeMaterial[] {
    const materials: THREE.NodeMaterial[] = [];
    const total = this.fft.stages * 2;

    for (let i = 0; i < total; i++) {
      const direction: 0 | 1 = i < this.fft.stages ? 0 : 1;
      const source = i % 2 === 0 ? pair.read.texture : pair.write.texture;
      const material = new THREE.NodeMaterial();
      material.fragmentNode = butterflyPassNode(
        this.fft,
        source,
        this.uStage,
        direction,
        this.cascadeCount * 2,
      );
      material.depthTest = false;
      material.depthWrite = false;
      materials.push(material);
    }
    return materials;
  }

  /**
   * Evolves `h0(k)` to `h(k, t)` and derives all four packed spectra in one pass
   * across the whole double-width target.
   *
   * It was two passes, one per pair, and everything up to and including `h`
   * itself was computed identically in both — the dispersion, the trig, the
   * mirrored-conjugate combination. This computes `h` once per texel and selects
   * which pair's derived spectra to emit from the texel's half.
   *
   * Both halves' arithmetic is evaluated and one is discarded, which is the
   * honest cost of the fold: a `select` is not a branch. It buys a pass, and a
   * pass is ~53 microseconds against a few dozen ALU operations on 65 000
   * texels, so the trade is not close.
   */
  private createEvolveMaterial(configs: readonly CascadeConfig[]): THREE.NodeMaterial {
    const size = this.size;
    const count = configs.length;
    const time = this.uTime;
    const deltaKs = configs.map((c) => (2 * Math.PI) / c.tileSize);

    const material = new THREE.NodeMaterial();
    material.depthTest = false;
    material.depthWrite = false;

    material.fragmentNode = Fn(() => {
      const coord = ivec2(
        float(size * 2 * count).mul(uv().x).floor().toInt(),
        float(size).mul(uv().y).floor().toInt(),
      ).toVar();

      // Which slot, which cascade, and where inside the slot. Derived in float
      // and floored rather than by integer division, which is spelled
      // differently on the two backends this has to compile to.
      const slot = float(coord.x).div(float(size)).floor().toVar();
      const localX = coord.x.sub(slot.mul(size).toInt()).toVar();
      const cascade = slot.div(2).floor().toVar();
      // Odd slots are pair B.
      const inB = slot.sub(cascade.mul(2)).greaterThan(0.5);

      // The band's wavenumber step. A chain of at most two selects over
      // build-time constants — `cascadeCount` is fixed for the life of the
      // material, so this costs nothing a uniform would not.
      let deltaKNode: any = float(deltaKs[count - 1]);
      for (let c = count - 2; c >= 0; c--) {
        deltaKNode = select(cascade.lessThan(c + 0.5), float(deltaKs[c]), deltaKNode);
      }
      // Loosely typed on purpose: `select` returns a node whose element type
      // TypeScript cannot narrow, and the multiply below then infers vec3.
      const deltaK: any = deltaKNode.toVar();

      const half = float(size * 0.5);
      const kx: any = float(localX).sub(half).mul(deltaK).toVar();
      const kz: any = float(coord.y).sub(half).mul(deltaK).toVar();
      const kLen = vec2(kx, kz).length().max(1e-6).toVar();

      // The spectrum atlas is one slot per cascade, not two, so it is indexed
      // by cascade rather than by slot.
      const spectrum = textureLoad(
        this.h0,
        ivec2(cascade.mul(size).toInt().add(localX), coord.y),
      ).toVar();
      const h0k = vec2(spectrum.x, spectrum.y).toVar();
      // Conjugate of h0(-k); the mirrored half was baked in at generation time.
      const h0MinusKConj = vec2(spectrum.z, spectrum.w.negate()).toVar();

      // Deep-water dispersion. Omega is deliberately not quantised to a common
      // multiple, so the surface never becomes exactly periodic in time.
      const omega = float(GRAVITY).mul(kLen).sqrt().toVar();
      const phase = omega.mul(time).toVar();
      const expIwt = vec2(phase.cos(), phase.sin()).toVar();
      const expMinusIwt = vec2(phase.cos(), phase.sin().negate()).toVar();

      // h(k, t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}
      const travelling = cMul(h0k, expIwt).add(cMul(h0MinusKConj, expMinusIwt)).toVar();

      // The standing limit, and it is the same two components with their phases
      // locked rather than a different model.
      //
      // A travelling wave is `h0(k) e^{iwt}` running one way against
      // `conj(h0(-k)) e^{-iwt}` running the other, each with its own phase. Lock
      // them together — take the Hermitian sum once and let the *whole* mode
      // breathe on `cos(wt)` — and the crests stop advancing and oscillate where
      // they are. That is a standing wave in the strict sense rather than a
      // slowed travelling one: same wavelengths, but with nodes.
      //
      // **It does not carry the same energy, and the amplitude compensation
      // below is why.** For two independent components the time-averaged energy
      // of the locked form is half the travelling form's, so blending at ratio
      // `r` leaves `1 - r + r^2/2` of it — 0.65 at the 0.45 the calmest preset
      // asks for, which is a 19% drop in RMS height. That would make the
      // standing control a sea-state control by the back door, and every other
      // parameter in this file is careful not to be. `uStandingGain` undoes
      // exactly that factor.
      //
      // Blended rather than switched, because the interesting values are in
      // between: real sheltered water is mostly standing with a little run left
      // in it. See `standingWaveRatio` in `Spectrum`, which also broadens the
      // spectrum's headings by the same amount — the two are halves of one
      // statement and moving one without the other gives a sea that oscillates
      // in place while still visibly pointing downwind.
      const standing = h0k.add(h0MinusKConj).mul(phase.cos()).toVar();
      const h = mix(travelling, standing, this.uStandingRatio).mul(this.uStandingGain).toVar();

      const nx = kx.div(kLen).toVar();
      const nz = kz.div(kLen).toVar();

      // Multiplying a complex value by i is a component swap with a sign flip.
      const iH = vec2(h.y.negate(), h.x).toVar();

      // --- pair A ---------------------------------------------------------
      // c0 = h + i * spectrum(dDx/dz), where dDx/dz has spectrum (kx kz / |k|) h
      const aDxdz = h.mul(nx).mul(nz).mul(kLen).toVar();
      const aC0 = vec2(h.x.sub(aDxdz.y), h.y.add(aDxdz.x)).toVar();
      // Dx has spectrum -i (kx/|k|) h, Dz has spectrum -i (kz/|k|) h
      const aDx = iH.mul(nx).negate().toVar();
      const aDz = iH.mul(nz).negate().toVar();
      const aC1 = vec2(aDx.x.sub(aDz.y), aDx.y.add(aDz.x)).toVar();

      // --- pair B ---------------------------------------------------------
      // c0 = dDy/dx + i dDy/dz, both with spectrum i k h
      const slopeX = iH.mul(kx).toVar();
      const slopeZ = iH.mul(kz).toVar();
      const bC0 = vec2(slopeX.x.sub(slopeZ.y), slopeX.y.add(slopeZ.x)).toVar();

      // c1 = dDx/dx + i dDz/dz, with spectra (kx^2/|k|) h and (kz^2/|k|) h
      const ddxdx = h.mul(kx.mul(kx).div(kLen)).toVar();
      const ddzdz = h.mul(kz.mul(kz).div(kLen)).toVar();
      const bC1 = vec2(ddxdx.x.sub(ddzdz.y), ddxdx.y.add(ddzdz.x)).toVar();

      return select(
        inB,
        vec4(bC0.x, bC0.y, bC1.x, bC1.y),
        vec4(aC0.x, aC0.y, aC1.x, aC1.y),
      );
    })();

    return material;
  }

  private createAssembleMaterial(
    cascadeIndex: number,
    config: CascadeConfig,
    output: 'displacement' | 'derivatives',
  ): THREE.NodeMaterial {
    const size = this.size;
    const choppiness = this.uChoppiness;

    const material = new THREE.NodeMaterial();
    material.depthTest = false;
    material.depthWrite = false;

    material.fragmentNode = Fn(() => {
      const coord = ivec2(
        float(size).mul(uv().x).floor().toInt(),
        float(size).mul(uv().y).floor().toInt(),
      ).toVar();

      // The spectrum is stored centred on k = 0, so the inverse transform comes
      // out with an alternating sign across the lattice. Undo it here.
      const parity = float(coord.x.add(coord.y).mod(2)).mul(-2).add(1).toVar();

      // This cascade's two slots in the atlas. Both offsets are build-time
      // constants: slot 2c is pair A and slot 2c + 1 is pair B.
      const baseA = cascadeIndex * 2 * size;
      const a = textureLoad(this.pair.read.texture, ivec2(coord.x.add(int(baseA)), coord.y))
        .mul(parity)
        .toVar();
      const b = textureLoad(
        this.pair.read.texture,
        ivec2(coord.x.add(int(baseA + size)), coord.y),
      )
        .mul(parity)
        .toVar();

      const lambda = choppiness.mul(config.choppiness).toVar();

      // Jacobian of the horizontal displacement map. Where it falls below 1 the
      // surface is folding onto itself — physically where whitecaps form.
      const jxx = float(1).add(b.z.mul(lambda)).toVar();
      const jzz = float(1).add(b.w.mul(lambda)).toVar();
      const jxz = a.y.mul(lambda).toVar();
      const jacobian = jxx.mul(jzz).sub(jxz.mul(jxz)).toVar();

      if (output === 'displacement') {
        // Store the raw Jacobian rather than a pre-thresholded foam value. The
        // surface material combines cascades by taking the most-folded one; a
        // pre-baked per-cascade mask can only be summed, which saturates the
        // whole surface to white as soon as two cascades fold at all.
        return vec4(a.z.mul(lambda), a.x, a.w.mul(lambda), jacobian);
      }

      // Alpha carries the second moment of the slope, and it is the whole reason
      // the distant water stopped boiling.
      //
      // Mipmapping a slope field averages the slope. That is right for the
      // *normal* and wrong for everything computed from it, because specular
      // response is not linear in slope: a footprint holding a hundred wave
      // facets averages to a mean near zero, so the shader sees a mirror, and the
      // mirror flickers as the mean wanders between frames. It is the classic
      // failure that LEAN and Toksvig mapping exist to solve, and on an ocean at
      // a hundred metres it is the single most visible artefact in the frame.
      //
      // Storing |slope|^2 lets the same hardware mip chain deliver E[s^2]
      // alongside E[s], and the variance the filtering destroyed comes back out
      // as E[s^2] - |E[s]|^2 for free. Isotropic rather than per-axis because
      // only one channel is spare, and the covariance term is worth less here
      // than the two diagonal ones. `OceanMaterial` folds it into alpha^2.
      return vec4(b.x, b.y, jacobian, b.x.mul(b.x).add(b.y.mul(b.y)));
    })();

    return material;
  }

  private releaseCascades(): void {
    for (const cascade of this.cascades) {
      cascade.displacement.dispose();
      cascade.derivatives.dispose();
      cascade.assembleDisplacement.dispose();
      cascade.assembleDerivatives.dispose();
    }
    this.cascades.length = 0;

    this.h0?.dispose();
    this.pair?.read.dispose();
    this.pair?.write.dispose();
    this.evolve?.dispose();
    for (const m of this.butterfly) m.dispose();
    this.butterfly = [];
  }
}

function makeTarget(
  size: number,
  cascadeCount: number,
  filter: THREE.MagnificationTextureFilter,
  wrap: THREE.Wrapping,
): THREE.RenderTarget {
  return new THREE.RenderTarget(size * 2 * cascadeCount, size, {
    type: THREE.FloatType,
    format: THREE.RGBAFormat,
    minFilter: filter,
    magFilter: filter,
    wrapS: wrap,
    wrapT: wrap,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  });
}

/**
 * Output targets are sampled at arbitrary world scale by the surface shader, so
 * a 16 m tile can span far less than a pixel at distance. Without a mip chain
 * that undersampling shows up as a shimmering speckle across the whole mid-field
 * — by far the most visible artefact on a moving ocean. Trilinear filtering over
 * generated mips lets the GPU pick the right level per pixel.
 *
 * `anisotropy = 1` is deliberate, and it is the opposite of the usual advice.
 *
 * Measured, on the far-field band under the horizon at High (mean |laplacian|,
 * `tests/gallery-jitter.spec.ts`): anisotropy 16 -> 6.80, 4 -> 6.28, 2 -> 5.01,
 * 1 -> 3.52. Monotonic, and the wrong way round from the usual expectation.
 *
 * The reason is not that anisotropic filtering under-samples. It does not: the
 * reference algorithm takes N = min(ceil(Pmax/Pmin), maxAniso) samples at
 * LOD = log2(Pmax/N), so a tap budget too small for the footprint is compensated
 * by choosing a coarser level, and the footprint is covered either way.
 *
 * The reason is that this is a *slope* field feeding a nonlinear shading model,
 * and filtering does not commute with it. Anisotropic filtering delivers a
 * better estimate of the mean slope over the footprint — and the mean slope is
 * the wrong thing to shade. Specular response is a sharply nonlinear function of
 * slope, so the correct answer is the mean of the shaded facets, not the shading
 * of the mean facet, and a higher tap count buys accuracy in exactly the
 * quantity that is not wanted. What it costs is the four levels of extra
 * sharpness it takes in exchange, which is retained slope variance the shading
 * then turns into noise.
 *
 * Dropping to 1 makes the hardware choose the major-axis level, which carries
 * less of that variance. It is a real trade and not a free win — detail along
 * the well-resolved minor axis goes with it — and it is the right side of the
 * trade for water seen from near its own surface, where the footprint is a
 * fraction of a metre across and tens of metres long. The alpha channel below
 * addresses the same mismatch from the other end, by carrying the variance
 * forward instead of discarding it.
 */
function makeOutputTarget(size: number): THREE.RenderTarget {
  const target = new THREE.RenderTarget(size, size, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearMipmapLinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.RepeatWrapping,
    wrapT: THREE.RepeatWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: true,
  });
  target.texture.anisotropy = 1;
  return target;
}

/**
 * A ping-pong of atlas targets, `2 * cascadeCount` slots of `size` across.
 *
 * Slot `2c` is cascade c's spectra pair A and slot `2c + 1` its pair B, so one
 * set of butterfly passes steps every cascade at once. Total memory is
 * unchanged: this is the same texel count the per-cascade pairs occupied.
 */
function makePingPong(size: number, cascadeCount: number): PingPong {
  return {
    read: makeTarget(size, cascadeCount, THREE.NearestFilter, THREE.ClampToEdgeWrapping),
    write: makeTarget(size, cascadeCount, THREE.NearestFilter, THREE.ClampToEdgeWrapping),
  };
}

export type { Cascade };
