export type QualityTier = 'low' | 'medium' | 'high' | 'ultra' | 'max';

export interface QualitySettings {
  /** Resolution of each FFT cascade (square). */
  fftSize: 128 | 256 | 512;
  /** Number of spectral cascades composited into the surface. */
  cascades: 1 | 2 | 3;
  /**
   * Concentric rings in the radial ocean grid. The single strongest lever on
   * surface fidelity: vertex spacing scales as radius / meshRings, and a cascade
   * can only displace geometry out to where its wavelength still exceeds that
   * spacing.
   *
   * **It is the worse of the two mesh axes that pulls that lever**, and these
   * pairs were re-proportioned on that basis. Radial spacing is
   * `exp(ln(24000/0.6) / meshRings) - 1` per metre of distance and angular is
   * `2*pi / meshSegments`; a wave survives displacement only if both resolve it,
   * so the coarser axis governs and over-sampling the finer one is vertex budget
   * spent on nothing. Every tier was previously 2.6 to 3.4 times out of square in
   * the same direction — far too many segments, far too few rings.
   *
   * They are now square to within 0.3%, which buys a 1.6x shorter surviving
   * wavelength at every distance and every tier. See `squareGrid` in
   * `ocean/meshSampling`, which derives them, and the test that pins them.
   *
   * **What that does not buy is a quieter image, and it was expected to.**
   * Measured across three trees, squaring recovers about 2% of the near-field
   * detail the vertex band-limit costs — nothing — while raising far-field
   * shimmer 5-7% and temporal change 8-10% on High, Ultra and Max. The reason is
   * that a finer mesh makes the band-limit remove *less*: `cascadeReach` fades a
   * cascade at a distance inversely proportional to vertex spacing, so this moves
   * High's ripple fade from 40-80 m out to 65-131 m. The extra energy is wave
   * geometry genuinely carried further, not noise — which is a distinction
   * `gallery-jitter` cannot draw. Re-measure with `tests/gallery-jitter.spec.ts`,
   * which produced every figure above.
   *
   * It is very slightly *cheaper*, not free: trading segments for rings at a
   * fixed `rings * segments` lowers both the true vertex count `(R+1)*S + 1` and
   * the triangle count `S*(2R+1)` by 0.1% to 0.5%.
   *
   * What it spends is the horizon ring's smoothness. At High the outer ring is a
   * 275-gon rather than a 448-gon, and at 24 km its chord sags 1.57 m — which
   * subtends 6.5e-5 rad, against 9.7e-4 rad for one pixel at 720 lines over a 40
   * degree vertical field. Fifteen times under a pixel, from any camera height.
   */
  meshRings: number;
  /** Segments around the circle. See `meshRings` for how the two are balanced. */
  meshSegments: number;
  /**
   * Sun shadow map resolution per side.
   *
   * Never zero, and that is a deliberate policy change rather than an oversight.
   * Turning a light's `castShadow` off and on at runtime makes three rebuild the
   * light's node graph, and its shadow node is created lazily inside `setup()` —
   * so a transition can leave a cached node holding a null render target that the
   * next thing to draw the scene reads `depthTexture` from. The planar reflector
   * renders the scene from its own `updateBefore`, so it reliably got there
   * first, and the result was a crash on a tier change.
   *
   * The class of bug disappears if the light simply always casts. Low pays one
   * 512-map shadow pass for it, which against 0.33 ms of a 16.7 ms budget is a
   * cost worth taking to make tier changes unable to crash.
   */
  shadowMapSize: 512 | 1024 | 2048 | 4096;
  /**
   * Resolution per side of each of the wake's three RGBA16F targets.
   *
   * The wake is a ping-pong pair plus a resolved output, so the cost is three
   * targets — a constant factor, not a third power — and quadratic in this
   * number: 3 × r² × 8 bytes at RGBA16F, so 24.0 MiB at 1024² and 13.5 at 768²,
   * before driver overhead.
   *
   * Low and Medium use 768², and specifically **not** 512², which the `Wake`
   * constructor comment rules out: the buffer carries whitecaps as well as the
   * wake, whitecap edges are decimetre features, and over a 420 m footprint 512
   * is 0.82 m per texel — enough to turn a crest streak into a smear. Raising
   * that default from 512 to 1024 was a deliberate fix and a tier must not
   * quietly undo it for 7.9 MiB.
   *
   * The texture objects stay bound to the water node graph while a tier change
   * resizes their targets in place.
   */
  wakeResolution: 512 | 768 | 1024;
  /**
   * Resolution per side of the mipmapped RGBA16F caustics field.
   *
   * Caustics are sampled below the surface and then blurred by distance, so Low
   * does not need the 768² source field used by High and above.
   *
   * 256 and 512 are powers of two; 768 is not, and is kept because it is the
   * resolution this field was authored and tuned at. A non-power-of-two side
   * gives a shorter mip chain that stops on an odd level rather than reaching
   * 1×1, which costs nothing here because the blur never reads the tail of the
   * chain. An earlier version of this comment claimed all three were powers of
   * two, which was simply false.
   *
   * The target is resized rather than replaced so the seafloor and underwater
   * node graphs keep their existing texture reference.
   */
  causticsResolution: 256 | 512 | 768;
  /**
   * Face size of the half-float environment cube.
   *
   * **Every tier is 256, and that is a measured result rather than an
   * oversight.** This started at 64 for Low and 128 for Medium on the reasoning
   * that a lighting probe is low-frequency and can be cheap. It is not: the far
   * water is close to a mirror at grazing angles, so it reflects the cube
   * directly, and halving the cube halves that reflection's angular resolution.
   *
   * `gallery-jitter` measured it. Medium's far-field high-frequency energy is
   * 2.352 at 256 and 2.414 at 128 — a 2.6% rise in exactly the quantity that
   * test exists to bound, for 2.25 MiB. Against the 682 MiB this branch takes
   * off the texture budget, that is not a trade worth making, and the number is
   * left here so nobody re-derives the same wrong intuition.
   *
   * The field and its setter are kept rather than folded back into a constant,
   * because the plumbing is what made the measurement possible.
   *
   * Resizing invalidates the next capture and preserves the texture object
   * consumed by `scene.environment`.
   */
  environmentResolution: 64 | 128 | 256;
  /**
   * Maximum device-pixel ratio used for the renderer framebuffer.
   *
   * This cap is applied before the first framebuffer is allocated. Low is
   * deliberately limited to DPR 1 because a four-times larger 2x framebuffer
   * is a memory decision made before the tier can save anything; Medium allows
   * a halfway step and the upper tiers keep the existing DPR 2 ceiling.
   */
  maxPixelRatio: number;
  /**
   * Whether the renderer allocates multisample antialiasing at construction.
   *
   * Low turns MSAA off before `renderer.init()` so the framebuffer never pays
   * for samples it cannot afford. It remains enabled for every richer tier,
   * preserving alpha-to-coverage and the existing edge quality there.
   */
  antialias: boolean;
  /**
   * March steps for the island's own shadow against the heightfield; 0 disables
   * it.
   *
   * A separate lever from `shadowMapSize` because it shadows something the map
   * structurally cannot: the sun's shadow camera is a +/-260 m box following the
   * viewer, and the island is a kilometre across. This is what gives the hill a
   * lit face and a shaded face at any range.
   *
   * Cheap despite being a raymarch, because it is gated on elevation — the sea
   * and the seabed under it never enter the loop, so only the island's own
   * pixels pay, and the step is the ray's clearance rather than a fixed
   * interval, so open sky is crossed in a few strides.
   */
  terrainShadowSteps: number;
  /**
   * Raymarch steps for the volumetric cloud layer; 0 disables volumetrics.
   *
   * High runs 18 rather than 24, and that is a measured trade rather than a
   * rounding. The march integrates its segments energy-conservingly, so the step
   * count refines the *texture* of a cloud and not how much of it there is —
   * which is why the tier can afford to spend the difference elsewhere. Cutting
   * this and `fogSteps` together took High from 10.46 ms to 8.31 and is what puts
   * the suite's own delivered-frame-rate assertion back inside its threshold;
   * the `npm run bench` gate had margin either way.
   */
  cloudSteps: number;
  /** Raymarch steps for underwater god rays; 0 disables them. */
  godRaySteps: number;
  /**
   * Master scale on hull spray. 0 draws none.
   *
   * Zero at Low for the same reason  is: that tier is the
   * WebGL2 floor and everything optional comes off it. The system still
   * builds and still compiles there — only its strength is zero — so
   * raising the tier cannot compile a shader mid-session.
   */
  spray: number;
  underwaterParticles: number;
  /**
   * Gulls in the flock, and fish in the school.
   *
   * Both are single instanced draws with every transform derived on the GPU, so
   * the count moves `instanceCount` and nothing else — no rebuild, no
   * allocation, and an individual keeps its own circuit across a tier change.
   * At 0 the renderer skips the draw entirely, which is why Low can have none
   * without a branch anywhere.
   *
   * The fish figure is a *total across ten schools*, which is what made the
   * previous numbers misleading: 90 at High reads like a shoal and divides into
   * nine fish per school, and nine fish spread over a 30 m station is not a
   * school — it is a scatter of individuals a diver has to go looking for. The
   * whole reef half of the scene was being judged on that. These are set so a
   * reef school is thirty-odd strong, which is the smallest number that still
   * reads as *a school* when one swims past.
   */
  birds: number;
  fish: number;
  /**
   * How much of the water's transmitted colour is the real refracted scene, as
   * opposed to the analytic depth-graded body colour.
   *
   * 0 mixes the refracted colour out; it does **not** skip the backdrop and
   * depth-buffer reads, which are unconditional in the node graph. An earlier
   * version of this comment claimed it did, which was simply false — an
   * independent review traced the uniform to its single use and found no branch.
   * The reads are cheap relative to the surface's fragment cost and both are
   * already paid for by the reflection path, so the honest description is that
   * this is a *visual* policy and not a cost one.
   */
  refraction: number;
  /**
   * How strongly the planar reflection of the scene shows in the water, 0..1.
   * 0 leaves the analytic sky gradient alone.
   *
   * Note what this does *not* do: whether the reflection is rendered at all is a
   * build-time property of the surface's node graph, decided once from the
   * backend, so a WebGPU tier with `reflection: 0` still pays for the mirrored
   * view and simply does not show it. Measured at 0.26 ms on the reference GPU —
   * accepted because the genuine low-end path is WebGL2, which has no reflector
   * in its graph at all.
   */
  reflection: number;
  /**
   * Resolution of the reflection render relative to the canvas. This is the real
   * cost lever — the reflection is sampled through a normal being perturbed by
   * every ripple, so it is never seen sharp.
   */
  reflectionScale: number;
  /**
   * Volumetric fog march steps. 0 is a bit-exact pass-through costing nothing.
   *
   * The march integrates cell weights exactly rather than as Riemann rectangles,
   * so raising this refines the noise detail, not the amount of fog — which is
   * why the tiers can differ this widely without the image changing brightness.
   */
  fogSteps: number;
  /**
   * Whether the bloom pyramid contributes, 0 or 1.
   *
   * An enable, not a strength, and the distinction is the same one `refraction`
   * makes: how strong bloom is belongs to the *look* — it is why
   * `toneMappingExposure` lives on the preset and not here — while what a tier
   * decides is whether a scene can afford five downsample-and-blur passes at
   * all. A tier that also set the strength would make the same preset a
   * different brightness on different hardware.
   *
   * 0 does not remove the node: rebuilding `outputNode` would recompile shaders
   * on a live tier change. It zeroes the contribution and drops the pyramid to
   * an eighth resolution, which together cost a rounding error. See
   * `SceneBloom.setEnabled`.
   */
  bloom: 0 | 1;
  /**
   * Taps in the depth-of-field gather.
   *
   * 0 is a bit-exact pass-through, and genuinely bit-exact rather than merely
   * cheap: the gather sits behind a uniform-coherent branch, so at zero the
   * output is the source texel untouched. The same construction `fogSteps: 0`
   * and `godRaySteps: 0` already use.
   *
   * The taps are laid out on a golden-angle spiral, so raising this refines the
   * bokeh rather than widening it — the radius comes from the aperture and the
   * circle of confusion, never from the count.
   */
  dofSamples: number;
  /**
   * Whether the sun/moon lens flare is drawn, 0 or 1.
   *
   * An enable rather than a strength, for the same reason `bloom` is: how strong
   * a flare reads is a property of the lens the whole project is photographed
   * through, not of how fast the machine is.
   */
  lensFlare: 0 | 1;
  /**
   * Lens-rain droplet lattice count, 1..3. 0 disables the effect entirely.
   *
   * Each level adds a lattice and, above 1, extra texture reads for misting
   * and chromatic dispersion. A clear frame costs one compare at any level.
   */
  lensRainQuality: number;
  /**
   * Strength of the ship wake's surface deformation, 0..1.
   *
   * Three taps of the wake buffer — one in the vertex stage, two in the fragment
   * stage for the gradient — so unlike most of the knobs here 0 genuinely removes
   * work rather than just hiding a result. Low turns it off because it is the
   * tier that also has no refraction and no reflection: a wake it cannot light
   * would read as a grey smear, not as water.
   */
  wakeDisplacement: number;
  /**
   * Fraction of the placed scene dressing that is drawn, 0..1.
   *
   * Instance counts only — the island's planting, the cove's clutter and the
   * reef's rocks thin out, they do not move or disappear as groups. Density is
   * the right lever because the dressing's cost is almost entirely draw-side
   * repetition of small meshes, and because a thinner grove still reads as a
   * grove where a missing one reads as a bug. It never reaches zero: the floor is
   * a quarter of placed capacity, so a low tier changes the island's density and
   * never its shape.
   */
  propsDetail: number;
  /**
   * Blades of kelp and seagrass drawn on the shallow bottom.
   *
   * Instanced, and every transform is derived on the GPU from a seed and the
   * clock, so this moves `instanceCount` and nothing else. Low draws none for
   * the same reason it draws no fish: it has no underwater pass worth dressing.
   */
  kelp: number;
  /**
   * Blades in the island's grass field.
   *
   * Instanced and placed on the GPU from the heightfield, so this is
   * `instanceCount` and nothing else — see `src/scene/Meadow.ts`. The field
   * follows the camera rather than covering the island, so this number is a
   * *density* over a 150 m square rather than a total: halving it halves how
   * thick the sward looks underfoot, wherever the viewer is standing.
   */
  meadow: number;

  /**
   * Canopy billboards standing on the island — see `src/scene/Canopy.ts`.
   *
   * Unlike `meadow` this is a *total*, not a density: the cards are planted on
   * the island rather than tiled around the camera, so the number is how much of
   * the island's forest exists. Low keeps a quarter of it, because the thing it
   * buys — the island having a canopy silhouette at all — is exactly what a low
   * tier must not lose. It is also nearly free: one draw, two triangles a card,
   * no texture and no shadow.
   */
  canopy: number;
}

export const QUALITY_TIERS: Record<QualityTier, QualitySettings> = {
  low: {
    fftSize: 128,
    cascades: 1,
    meshRings: 206,
    meshSegments: 119,
    shadowMapSize: 512,
    wakeResolution: 768,
    causticsResolution: 256,
    environmentResolution: 256,
    maxPixelRatio: 1,
    antialias: false,
    terrainShadowSteps: 0,
    cloudSteps: 0,
    godRaySteps: 0,
    spray: 0,
    underwaterParticles: 400,
    birds: 0,
    fish: 0,
    // No backdrop or depth-buffer read at all. This is the WebGL2 floor, where
    // the analytic body colour has to carry the water on its own.
    refraction: 0,
    reflection: 0,
    reflectionScale: 0.25,
    fogSteps: 0,
    bloom: 0,
    dofSamples: 0,
    lensFlare: 0,
    lensRainQuality: 1,
    wakeDisplacement: 0,
    propsDetail: 0.3,
    kelp: 0,
    meadow: 0,
    canopy: 4000,
  },
  medium: {
    fftSize: 128,
    cascades: 2,
    meshRings: 308,
    meshSegments: 179,
    shadowMapSize: 1024,
    wakeResolution: 768,
    causticsResolution: 512,
    environmentResolution: 256,
    maxPixelRatio: 1.5,
    antialias: true,
    terrainShadowSteps: 12,
    cloudSteps: 12,
    godRaySteps: 12,
    spray: 0.6,
    underwaterParticles: 1200,
    birds: 14,
    fish: 110,
    // Partial: the scene shows through, but the analytic body still carries most
    // of the colour, which hides the coarser depth resolution at this tier.
    refraction: 0.6,
    reflection: 0.6,
    reflectionScale: 0.35,
    fogSteps: 12,
    bloom: 1,
    dofSamples: 8,
    lensFlare: 1,
    lensRainQuality: 2,
    wakeDisplacement: 0.75,
    propsDetail: 0.5,
    kelp: 800,
    meadow: 18000,
    canopy: 9000,
  },
  high: {
    fftSize: 256,
    cascades: 3,
    meshRings: 469,
    meshSegments: 275,
    shadowMapSize: 2048,
    wakeResolution: 1024,
    causticsResolution: 768,
    environmentResolution: 256,
    maxPixelRatio: 2,
    antialias: true,
    terrainShadowSteps: 20,
    cloudSteps: 18,
    godRaySteps: 24,
    spray: 1,
    underwaterParticles: 2400,
    birds: 26,
    fish: 240,
    refraction: 1,
    reflection: 0.85,
    reflectionScale: 0.5,
    fogSteps: 18,
    bloom: 1,
    dofSamples: 16,
    lensFlare: 1,
    lensRainQuality: 3,
    wakeDisplacement: 1,
    propsDetail: 0.75,
    kelp: 1800,
    meadow: 38000,
    canopy: 15000,
  },
  ultra: {
    fftSize: 256,
    cascades: 3,
    meshRings: 613,
    meshSegments: 360,
    shadowMapSize: 2048,
    wakeResolution: 1024,
    causticsResolution: 768,
    environmentResolution: 256,
    maxPixelRatio: 2,
    antialias: true,
    terrainShadowSteps: 28,
    cloudSteps: 34,
    godRaySteps: 40,
    spray: 1,
    underwaterParticles: 4000,
    birds: 40,
    fish: 380,
    refraction: 1,
    reflection: 1,
    reflectionScale: 0.6,
    fogSteps: 34,
    bloom: 1,
    dofSamples: 24,
    lensFlare: 1,
    lensRainQuality: 3,
    wakeDisplacement: 1,
    propsDetail: 1,
    kelp: 2900,
    meadow: 62000,
    canopy: 20000,
  },
  max: {
    fftSize: 512,
    cascades: 3,
    meshRings: 817,
    meshSegments: 481,
    shadowMapSize: 4096,
    wakeResolution: 1024,
    causticsResolution: 768,
    environmentResolution: 256,
    maxPixelRatio: 2,
    antialias: true,
    terrainShadowSteps: 28,
    cloudSteps: 34,
    godRaySteps: 40,
    spray: 1,
    underwaterParticles: 6000,
    birds: 64,
    fish: 560,
    refraction: 1,
    reflection: 1,
    reflectionScale: 0.75,
    fogSteps: 34,
    bloom: 1,
    dofSamples: 32,
    lensFlare: 1,
    lensRainQuality: 3,
    wakeDisplacement: 1,
    propsDetail: 1,
    kelp: 4000,
    meadow: 90000,
    canopy: 24000,
  },
};

export const TIER_ORDER: QualityTier[] = ['low', 'medium', 'high', 'ultra', 'max'];

/**
 * Watches frame health and steps the tier down when the experience is
 * persistently below target. Deliberately one-way during a session: oscillating
 * between tiers is more distracting than running one notch below optimal.
 */
export class AdaptiveQuality {
  private belowTargetFor = 0;
  private aboveTargetFor = 0;
  private lastChange = 0;

  constructor(
    private readonly targetFps = 55,
    private readonly onDowngrade: (tier: QualityTier) => void = () => {},
  ) {}

  enabled = true;

  update(dt: number, fps: number, currentTier: QualityTier, elapsed: number): void {
    if (!this.enabled) return;
    // Ignore the first seconds — shader compilation and asset decode dominate.
    if (elapsed < 4) return;
    // Debounce so a single hitch never triggers a visible quality change.
    if (elapsed - this.lastChange < 6) return;

    if (fps < this.targetFps * 0.75) {
      this.belowTargetFor += dt;
      this.aboveTargetFor = 0;
    } else {
      this.aboveTargetFor += dt;
      this.belowTargetFor = 0;
    }

    if (this.belowTargetFor > 2.5) {
      const index = TIER_ORDER.indexOf(currentTier);
      if (index > 0) {
        this.belowTargetFor = 0;
        this.lastChange = elapsed;
        this.onDowngrade(TIER_ORDER[index - 1]);
      }
    }
  }

  reset(): void {
    this.belowTargetFor = 0;
    this.aboveTargetFor = 0;
  }
}

