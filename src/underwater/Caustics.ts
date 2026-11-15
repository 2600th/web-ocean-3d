import * as THREE from 'three/webgpu';
import { Fn, exp, float, mix, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

/**
 * Procedural light caustics: one small, periodic, animated tile, read at
 * several scales.
 *
 * The arrangement is abyssal-garden's (MIT, (c) 2026 Maksymilian Dendura; see
 * THIRD_PARTY_NOTICES.md): its `caus` reads a tiling water texture at metre
 * scale for the web a surface shows (`q / 3.3`), modulated by a second read at
 * another scale and orientation (`q.yx / 6.9 + .37`) so the repeat never lines
 * up, and its `W_RAYS` reads the *same* texture eight and three times larger,
 * thresholded, as the beams the shafts are made of (`q / 27`, `q.yx / 11`).
 * The pattern in the tile is our own: two wrapping Voronoi webs with orbiting
 * feature points and a periodic warp, which gives the curved, boiling
 * filaments of a refracting surface.
 *
 * This replaced a 3D-worley field baked into a 320 m world-anchored target.
 * That field had two faults, and both showed in the blind review. Its texel
 * was 0.42 m (High) to 0.62 m, while a filament is a few centimetres wide, so
 * every surface read soft blotches; and its mean was 0.38 with the brightest
 * filament at 1.4, while every consumer — `groundShading`, `LeafField`,
 * `LifeShading`, the seafloor — writes `mix(1, caustics, reach)` and so
 * assumes a field centred on 1. "Caustics" darkened the sand to a third and
 * never brightened it at all. Both lagoon frames were judged to have none.
 *
 * Refraction at the surface is approximated by shearing the sample coordinate
 * along the refracted sun: a point `d` metres below the surface is lit by the
 * patch of surface at `xz + (sunDir.xz / sunDir.y) * d`, so the pattern slides
 * sideways as the sun lowers and the shafts stand at the refracted angle.
 *
 * The bake is one fullscreen pass over the tile per frame and compiles to WGSL
 * and GLSL alike.
 */

export interface CausticsParams {
  /** Contrast of the web, 0..2. 1 is the authored pattern; 0 is uniform light. */
  strength: number;
  /** Cells of the coarser web per metre. Larger = finer, busier caustics. */
  scale: number;
  /** Animation rate of the cell drift. */
  speed: number;
  /** Per metre of depth: how fast the web's contrast relaxes toward uniform light. */
  depthFade: number;
  /** Line sharpness. Higher = thinner, brighter filaments. */
  sharpness: number;
  /** World Y of the water surface the caustics are cast from. */
  surfaceLevel: number;
}

/** Cells of each web across the tile. Integers: the tile must wrap. */
const CELLS_A = 6;
const CELLS_B = 10;

export const DEFAULT_CAUSTICS_PARAMS: CausticsParams = {
  strength: 1.0,
  scale: CELLS_A / 4.2,
  speed: 0.35,
  depthFade: 0.055,
  sharpness: 14,
  surfaceLevel: 0,
};

/** Clock wrap, seconds — long enough to be invisible, short enough for float32. */
const CLOCK_WRAP = 3600;

/**
 * Reciprocal of the tile's mean, measured by reading the bake back at the
 * default sharpness. Dividing by it centres the pattern on 1.
 */
const TILE_NORM = 1 / 0.125;

/**
 * The shafts' two reads, as multiples of the tile: abyssal's `27 / 3.3` and
 * `11 / 3.3`. At these scales the web's cells are metres across and its lines
 * decimetres, so thresholded and extruded along the refracted sun they are the
 * thin sheets of light a real shaft field is, not the blobs a broad ratio made.
 */
const SHAFT_SCALE_A = 8.2;
const SHAFT_SCALE_B = 3.3;

/**
 * How far the web moves a surface's light, below and above the mean; see
 * `applyWeb`. The gaps between filaments keep half the light; the filaments
 * keep most of their excess.
 */
const SURFACE_DARK_SHARE = 0.5;
const SURFACE_BRIGHT_SHARE = 0.8;

/** The surfaces' second, modulating read, abyssal's `6.9 / 3.3`. */
const BROAD_SCALE = 2.1;

/** Sanity bound on the shear; the refracted sun cannot exceed ~1.13. */
const MAX_TILT = 3;

/** Refractive index of sea water, as everywhere else in the project. */
const WATER_IOR = 1.333;

export class Caustics {
  private readonly params: CausticsParams;

  // --- uniforms ------------------------------------------------------------
  private readonly uTime = uniform(0);
  private readonly uStrength = uniform(DEFAULT_CAUSTICS_PARAMS.strength);
  /** World metres one repeat of the tile spans: `CELLS_A / scale`. */
  private readonly uTile = uniform(CELLS_A / DEFAULT_CAUSTICS_PARAMS.scale);
  private readonly uSpeed = uniform(DEFAULT_CAUSTICS_PARAMS.speed);
  private readonly uDepthFade = uniform(DEFAULT_CAUSTICS_PARAMS.depthFade);
  private readonly uSharpness = uniform(DEFAULT_CAUSTICS_PARAMS.sharpness);
  private readonly uSurfaceY = uniform(DEFAULT_CAUSTICS_PARAMS.surfaceLevel);
  /** Metres of horizontal shear per metre of depth, from the sun's elevation. */
  private readonly uSunTilt = uniform(new THREE.Vector2(0, 0));
  /**
   * Red-to-blue separation of the chromatic taps, metres: a constant part plus
   * a part per metre of depth. See `chromaticNode`.
   *
   * About a filament's width at 5 m, so each line carries a red edge and a
   * blue one rather than splitting into three — abyssal's `vec2(.004, .0025)`
   * of its 3.3 m tile, which is 1.3 cm.
   */
  private readonly uDispersion = uniform(new THREE.Vector2(0.012, 0.0028));
  /**
   * Smoothstep edges of the shafts' two reads; see `shaftField`. Set for the
   * heavily blurred levels the march reads (`UnderwaterPass.uShaftBlurLod`),
   * where the web has averaged to within about half of its mean of 1.
   */
  readonly uShaftA = uniform(new THREE.Vector2(0.95, 1.35));
  readonly uShaftB = uniform(new THREE.Vector2(0.85, 1.25));

  /**
   * The node graphs, built exactly once. TSL functions are lazily compiled at
   * shader build time, so constructing them here is safe.
   */
  private readonly fn: any;
  private readonly fnLod: any;
  private readonly fnChroma: any;

  /** The baked tile. Stable reference — safe for materials to bind once. */
  readonly texture: THREE.Texture;
  /** Texels per side. Updated by `setResolution` without replacing the texture. */
  resolution: number;

  private readonly target: THREE.RenderTarget;
  private readonly bakeMaterial: THREE.NodeMaterial;
  private readonly quad = new THREE.QuadMesh(new THREE.NodeMaterial());

  private clock = 0;
  private disposed = false;

  /**
   * @param resolution Texels per side of the tile. 512 over the default 4.2 m
   *   is 8 mm a texel; the filaments are a few centimetres.
   */
  constructor(resolution = 512) {
    this.params = { ...DEFAULT_CAUSTICS_PARAMS };
    this.resolution = normalizeResolution(resolution);

    this.target = new THREE.RenderTarget(this.resolution, this.resolution, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      // Mipmapped, and three consumers depend on it. A surface far away reads
      // a coarse level, which averages the web back to its mean of 1, so
      // distant sand neither shimmers nor changes brightness. The shafts read
      // explicit coarse levels, because the march's neighbouring fragments
      // sample points metres apart and the hardware's inferred derivative is
      // meaningless there. And depth reads a biased level, which is how the
      // filaments soften in deep water.
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      // Repeated: the tile is periodic by construction.
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: true,
    });
    // Anisotropic: the sand is mostly seen at a grazing angle, where an
    // isotropic mip picks its level from the long axis of the footprint and
    // averages the web back to flat grey a few metres out.
    this.target.texture.anisotropy = 8;
    this.texture = this.target.texture;

    this.bakeMaterial = new THREE.NodeMaterial();
    this.bakeMaterial.depthTest = false;
    this.bakeMaterial.depthWrite = false;
    this.bakeMaterial.fragmentNode = Fn(() => {
      const tuv = uv().toVar('cdUv');
      const t = this.uTime.mul(this.uSpeed).toVar('cdT');
      // A gentle periodic warp bends the Voronoi edges into the curved
      // filaments a refracting surface makes; straight edges read as cracked
      // mud. Integer frequencies, so the warp wraps with the tile.
      const tau = 6.2831853;
      const warp = vec2(
        tuv.y.mul(2 * tau).add(t.mul(0.31)).sin().add(tuv.x.mul(3 * tau).sub(t.mul(0.23)).sin().mul(0.6)),
        tuv.x.mul(2 * tau).sub(t.mul(0.27)).sin().add(tuv.y.mul(3 * tau).add(t.mul(0.19)).sin().mul(0.6)),
      ).mul(0.04);
      const w = tuv.add(warp).toVar('cdW');
      const ea = periodicVoronoiEdge(w.mul(CELLS_A), CELLS_A, t.mul(0.9), 0);
      const eb = periodicVoronoiEdge(w.yx.mul(CELLS_B).add(vec2(0.37, 0.61)), CELLS_B, t.mul(-1.2), 17);
      // Edge distance to a line: narrow and bright, with a short glow, which is
      // how a focal curve sits on sand. Squared after summing so a crossing of
      // the two webs flares the way real caustic cusps do.
      const la = exp(ea.mul(this.uSharpness.negate()));
      const lb = exp(eb.mul(this.uSharpness.mul(-1.3))).mul(0.6);
      const web = la.add(lb).pow(2.2).mul(TILE_NORM);
      return vec4(web, 0, 0, 1);
    })();

    this.fn = Fn(([worldPosition]: any) => this.surfaceField(worldPosition));
    this.fnLod = Fn(([worldPosition, lod]: any) => this.shaftField(worldPosition, lod));
    this.fnChroma = Fn(([worldPosition]: any) => this.chromaticField(worldPosition));

    this.setSunDirection(new THREE.Vector3(0.35, 0.62, 0.7));
  }

  /**
   * Resizes the tile in place. The texture object is retained, so every graph
   * that bound it stays valid; the next `bake` repopulates the mip chain.
   */
  setResolution(resolution: number): void {
    if (this.disposed) return;
    const next = normalizeResolution(resolution);
    if (next === this.resolution) return;
    this.target.setSize(next, next);
    this.resolution = next;
  }

  /** World metres one repeat of the tile spans. */
  get tileSize(): number {
    return this.uTile.value;
  }

  // ------------------------------------------------------------------ fields

  /** Depth below the surface and the surface point that lit `p`. */
  private surfacePoint(p: any): { below: any; xz: any } {
    const below = this.uSurfaceY.sub(p.y).max(0.0).toVar();
    const xz = vec2(p.x, p.z).add(this.uSunTilt.mul(below)).toVar();
    return { below, xz };
  }

  /**
   * The web a lit surface shows, centred on 1: bright filaments over dimmer
   * cells, sharp in the shallows and relaxing toward uniform light with depth.
   *
   * Depth does two things to a caustic, and both are here. The focal curves
   * spread, because the surface patch that lights a point grows with the
   * column above it — a mip bias, so the lines soften rather than vanish. And
   * the contrast falls, because the many patches start to overlap — a mix
   * toward 1, *not* toward 0. The old field multiplied by `exp(-k depth)`,
   * which turned depth into darkness on top of the medium's own (already
   * applied by the underwater pass), so a reef at 17 m lost its sunlight twice.
   */
  private surfaceField(worldPosition: any): any {
    const p = vec3(worldPosition).toVar('causticsSP');
    const { below, xz } = this.surfacePoint(p);
    const web = texture(this.texture, xz.div(this.uTile))
      .bias(this.detailBias(below))
      .r.mul(this.broadPattern(xz));
    return applyWeb(web, this.detailContrast(below));
  }

  /**
   * The same field read three times, once per colour channel, a little apart
   * along the refracted sun's horizontal direction.
   *
   * Water is dispersive: blue refracts about 1.3% more than red (n = 1.343 at
   * 450 nm against 1.331 at 650 nm), so the focus each filament forms lands at
   * a slightly different place per wavelength and the bright lines carry a red
   * fringe on one side and a blue one on the other. Adapted from the three-tap
   * `caus` in abyssal-garden, with the separation in metres and growing with
   * depth — the spread is a property of the refracted ray's length.
   */
  private chromaticField(worldPosition: any): any {
    const p = vec3(worldPosition).toVar('causticsCP');
    const { below, xz } = this.surfacePoint(p);
    const tileUv = xz.div(this.uTile).toVar('causticsCUv');

    // Along the tilt where there is one; a fixed diagonal under a vertical sun,
    // where the refracted ray has no horizontal direction to spread along.
    const tiltLen = this.uSunTilt.length().toVar();
    const axis = mix(vec2(0.7071, 0.7071), this.uSunTilt.div(tiltLen.max(1e-4)), tiltLen.smoothstep(0.02, 0.2));
    const spread = this.uDispersion.x.add(this.uDispersion.y.mul(below));
    const offset = axis.mul(spread.div(this.uTile)).toVar('causticsCOff');

    const bias = this.detailBias(below);
    const r = texture(this.texture, tileUv.add(offset)).bias(bias).r;
    const g = texture(this.texture, tileUv).bias(bias).r;
    const b = texture(this.texture, tileUv.sub(offset)).bias(bias).r;
    const web = vec3(r, g, b).mul(this.broadPattern(xz));
    return vec3(
      applyWeb(web.x, this.detailContrast(below)),
      applyWeb(web.y, this.detailContrast(below)),
      applyWeb(web.z, this.detailContrast(below)),
    );
  }

  /**
   * The beams the shafts are made of, 0..1: abyssal-garden's `W_RAYS`
   * pattern, `ss(.3, .7, A) * ss(.15, .6, B)`, with `A` and `B` the tile read
   * eight and three times larger and blurred by `lod`. The thresholds are set
   * for this tile's blurred distribution rather than abyssal's; see `uShaftA`.
   */
  private shaftField(worldPosition: any, lod: any): any {
    const p = vec3(worldPosition).toVar('causticsRP');
    const { xz } = this.surfacePoint(p);
    const a = texture(this.texture, xz.div(this.uTile.mul(SHAFT_SCALE_A))).level(float(lod).add(1)).r;
    const b = texture(this.texture, xz.yx.div(this.uTile.mul(SHAFT_SCALE_B)).add(0.3)).level(float(lod).add(0.5)).r;
    return a.smoothstep(this.uShaftA.x, this.uShaftA.y).mul(b.smoothstep(this.uShaftB.x, this.uShaftB.y));
  }

  /** Mip bias for the tile at `below` metres: sharp at the top, softer down. */
  private detailBias(below: any): any {
    return below.mul(0.08).add(1).log2();
  }

  /** How much of the web survives at `below` metres, times the strength. */
  private detailContrast(below: any): any {
    return exp(below.mul(this.uDepthFade).negate()).mul(this.uStrength).clamp(0, 1.5);
  }

  /**
   * The surfaces' second read: the tile again at another scale and turned a
   * quarter, heavily blurred, as a brightness modulation around 1 — the
   * `.45 + b * 1.3` in abyssal's `caus`. It breaks the repeat and gives the
   * web the larger bright and dim patches real caustics have.
   */
  private broadPattern(xz: any): any {
    const b = texture(this.texture, xz.yx.div(this.uTile.mul(BROAD_SCALE)).add(0.37)).bias(float(3)).r;
    return mix(float(1), b, 0.55).clamp(0.4, 2.2);
  }

  // ------------------------------------------------------------------ bake

  /**
   * Re-bakes the tile. The arguments are the camera's world position, kept so
   * callers need not change; a periodic tile has no centre.
   *
   * Must run outside an active render target; it saves and restores its own.
   */
  bake(renderer: THREE.WebGPURenderer, _centerX = 0, _centerZ = 0): void {
    if (this.disposed) return;
    const previous = renderer.getRenderTarget();
    this.quad.material = this.bakeMaterial;
    renderer.setRenderTarget(this.target);
    this.quad.render(renderer);
    renderer.setRenderTarget(previous);
  }

  /**
   * Caustic light at a world-space point.
   *
   * With `lod` null: the web a lit surface shows, centred on 1 — what every
   * material multiplies its key light by. With an explicit `lod`: the shaft
   * pattern, 0..1, blurred by that many levels — what the volumetric march
   * integrates. Pure: depends only on `worldPosition` and the uniforms.
   */
  intensityNode(worldPosition: unknown, lod: unknown = null): unknown {
    return lod === null ? this.fn(worldPosition) : this.fnLod(worldPosition, lod);
  }

  /** The surface field as RGB with per-channel dispersion; see `chromaticField`. */
  chromaticNode(worldPosition: unknown): unknown {
    return this.fnChroma(worldPosition);
  }

  setParams(p: Partial<CausticsParams>): void {
    if (p.strength !== undefined) this.params.strength = p.strength;
    if (p.scale !== undefined) this.params.scale = p.scale;
    if (p.speed !== undefined) this.params.speed = p.speed;
    if (p.depthFade !== undefined) this.params.depthFade = p.depthFade;
    if (p.sharpness !== undefined) this.params.sharpness = p.sharpness;
    if (p.surfaceLevel !== undefined) this.params.surfaceLevel = p.surfaceLevel;

    this.uStrength.value = Math.max(0, this.params.strength);
    this.uTile.value = CELLS_A / Math.max(1e-3, this.params.scale);
    this.uSpeed.value = this.params.speed;
    this.uDepthFade.value = Math.max(0, this.params.depthFade);
    this.uSharpness.value = Math.max(1, this.params.sharpness);
    this.uSurfaceY.value = this.params.surfaceLevel;
  }

  getParams(): Readonly<CausticsParams> {
    return this.params;
  }

  /**
   * `dir` points *toward* the sun (in air) and need not be normalised.
   *
   * The shear follows the **refracted** ray, not the one in air. Snell's law
   * divides the horizontal component of the unit direction by 1.333, so
   * `skyPro`'s 24-degree sun, which would shear the pattern by 2.2 m per metre
   * of depth in air, arrives under water at 47 degrees and shears by 0.93 —
   * which is what stands the shafts steep, as every reference shows them.
   */
  setSunDirection(dir: THREE.Vector3): void {
    const len = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const hx = dir.x / len / WATER_IOR;
    const hz = dir.z / len / WATER_IOR;
    const y = Math.sqrt(Math.max(1e-4, 1 - hx * hx - hz * hz));
    const tilt = this.uSunTilt.value;
    tilt.set(clampNumber(hx / y, -MAX_TILT, MAX_TILT), clampNumber(hz / y, -MAX_TILT, MAX_TILT));
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  /** Rewinds the animation clock, for reproducible captures. */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.target.dispose();
    this.bakeMaterial.dispose();
    this.quad.geometry.dispose();
  }
}

/**
 * A surface's light under the web `w` (mean 1) at contrast `c`.
 *
 * Not `mix(1, w, c)`. The tile's median is 0.09 against its mean of 1, so at
 * full contrast the brightest tenth carried all the light, the tone curve
 * clipped it, and the rest of the shelf read darker than with no caustics at
 * all: 0.82x mean and 0.58x median on the shore-break shelf, measured with
 * the sea hidden. Here the gaps keep half the light, which is what sand
 * between real caustic lines gets, and the filaments keep 0.8 of their excess.
 * That puts the linear mean somewhat above 1, which the tone curve's
 * compression of the filaments takes back: measured the same way, the shelf
 * now displays at 0.98x of its uncausticked brightness, with the gaps at
 * 0.79x and the filaments at 1.56x (p90).
 */
function applyWeb(w: any, c: any): any {
  const d = float(w).sub(1);
  return d.mul(c).mul(d.greaterThan(0).select(float(SURFACE_BRIGHT_SHARE), float(SURFACE_DARK_SHARE))).add(1);
}

/**
 * Distance to the nearest Voronoi edge (F2 - F1 of the cell distances) on a
 * lattice of `cells` that wraps, with every feature point orbiting its cell so
 * the web boils instead of sliding. `p` is in cell units.
 *
 * Periodic so the tile repeats seamlessly; nine cells because the orbit keeps
 * each point inside its own cell, which is what the 3x3 neighbourhood assumes.
 */
function periodicVoronoiEdge(p: any, cells: number, time: any, seed: number): any {
  const i = p.floor().toVar();
  const f = p.fract().toVar();
  const f1 = float(8).toVar();
  const f2 = float(8).toVar();
  for (let y = -1; y <= 1; y++) {
    for (let x = -1; x <= 1; x++) {
      const o = vec2(x, y);
      const c = i.add(o).add(cells).mod(cells).add(seed);
      const h = vec2(c.dot(vec2(127.1, 311.7)), c.dot(vec2(269.5, 183.3))).sin().mul(43758.5453).fract();
      const orbit = time.mul(h.yx.mul(0.8).add(0.6)).add(h.mul(6.2831853)).sin().mul(0.38);
      const d = o.add(0.5).add(orbit).sub(f);
      const dd = d.dot(d);
      f2.assign(f2.min(f1.max(dd)));
      f1.assign(f1.min(dd));
    }
  }
  return f2.sqrt().sub(f1.sqrt());
}

function normalizeResolution(size: number): number {
  // Capped: past 8 mm a texel the filaments gain nothing, and the bake is the
  // one per-frame cost this module has above water.
  return Number.isFinite(size) ? Math.min(512, Math.max(16, Math.round(size))) : 512;
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
