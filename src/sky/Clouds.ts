import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  interleavedGradientNoise,
  max,
  min,
  mix,
  mx_fractal_noise_float,
  normalize,
  positionGeometry,
  pow,
  screenCoordinate,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import type { AerialPerspective } from './AerialPerspective';

/**
 * Raymarched volumetric cloud layer.
 *
 * The layer is a **spherical shell** of procedural FBM density between
 * `altitude` and `altitude + thickness`, wrapped on a planet of `PLANET_RADIUS`.
 * That is what makes the deck converge into the horizon instead of stopping in a
 * band, and it is what bounds a grazing ray's crossing without a clamp. It is
 * rendered on a camera-locked dome: the dome only supplies view rays, the march
 * itself happens in world space, so the dome radius is unrelated to the cloud
 * altitude.
 *
 * Density is entirely procedural (`mx_fractal_noise_float`) — there is no 3D
 * noise texture to download and nothing to keep resident in VRAM. Coverage is
 * modulated by a 22 km weather field, so the sky has clear regions and dense
 * ones rather than one threshold over the whole hemisphere.
 *
 * Lighting is a short secondary march toward the sun for optical depth, resolved
 * through three **multiple-scattering** orders after Hillaire — each seeing less
 * extinction, contributing less energy and scattering more isotropically than
 * the last, which is what keeps a thick cloud's interior glowing instead of
 * going flat grey. The ambient is graded between the sea-facing base and the sky
 * the top sees. Compare ref-default.png (scattered cumulus) and ref-storm.png
 * (overcast).
 *
 * **Supported camera range.** Rays at or below the horizon are marched as
 * horizon rays (see `GRAZE_FLOOR`), so a camera *above* the layer cannot look
 * down through it correctly. Nothing in this project flies to 1400 m; if
 * something ever does, that floor is where to start.
 */

export interface CloudParams {
  coverage: number; // 0..1, wired to the demo's Cloud Coverage slider
  density: number;
  altitude: number; // metres
  thickness: number;
  windSpeed: number;
  windDirection: number;
  /**
   * Rate at which billows grow and erode in place, in noise units per second.
   *
   * Distinct from `windSpeed`, which only translates the layer. At the
   * kilometre feature scale this noise runs at, translation is imperceptible on
   * the timescale anyone looks at a demo for — evolution is what makes the sky
   * read as moving.
   */
  evolutionRate: number;
  steps: number; // raymarch steps; 0 disables the layer entirely
  color: THREE.Color;
  shadowColor: THREE.Color;
}

export const DEFAULT_CLOUD_PARAMS: CloudParams = {
  coverage: 0.32,
  density: 1,
  altitude: 1400,
  thickness: 700,
  windSpeed: 9,
  windDirection: 2.6,
  evolutionRate: 0.012,
  steps: 24,
  color: new THREE.Color(1.0, 0.99, 0.96),
  shadowColor: new THREE.Color(0.34, 0.38, 0.47),
};

/** Dome radius in metres — see Atmosphere for why this can be small. */
const DOME_RADIUS = 100;

/** Hard ceiling on the loop the shader is compiled with. */
const MAX_STEPS = 96;
/** Secondary samples taken toward the sun per march step. */
const LIGHT_STEPS = 4;

/**
 * Base octaves the sun march evaluates density at, with the erosion octave
 * skipped entirely — against four octaves plus erosion for the layer itself.
 *
 * This is what `densityAt` has always documented should happen here and did not:
 * "a shadow evaluated once per pixel can afford four; one evaluated inside
 * another raymarch cannot". This one is inside another raymarch — `LIGHT_STEPS`
 * samples at every one of `uSteps` steps — and it was taking the full seven
 * octaves, 72 times a pixel at High.
 *
 * **Worth 39% of the cloud layer**, 5.23 ms to 3.19, on a frame that was 12.31.
 * The layer is the single most expensive thing in the frame, so this is the
 * largest remaining lever that does not restructure anything.
 *
 * **It is a look change and was accepted as one**, on the pictures rather than
 * on the numbers. Nine of the twenty-one visual baselines moved and were
 * regenerated; `sunset` moved most, at mean dE94 0.284 against a 0.167 limit
 * with a peak of 46.9, because a low sun gives the longest path through the deck
 * and so the most integration for a coarser field to disagree about. Two of the
 * nine — `waterline` and `near-water-detail` — were inside their mean and pixel
 * limits and failed on p95 alone.
 *
 * What it costs is the fine structure of a cloud's *self-shadowing*. What it
 * does not cost is the silhouette, which still comes from the full field in the
 * outer march, and which is what the eye actually reads a cloud by. The reason
 * the interior survives coarsening is that this accumulates optical depth toward
 * the sun and then feeds it through `exp` — an integral of the field rather than
 * a sample of it, and integrating a blobbier field lands close to the same
 * number.
 */
const LIGHT_OCTAVES = 2;

/**
 * Radius the cloud layer is wrapped on, metres.
 *
 * The layer used to be a flat slab, and a flat slab is why the sky stopped: the
 * crossing had to be clamped to ten thicknesses or a grazing ray marched to
 * infinity, and then everything below 4 degrees of elevation was faded out to
 * hide the clamp. The result is `clear-day.png` — uniformly sized puffs at
 * uniform spacing all the way out, ending in a band with clear sky underneath
 * it. A real deck converges: puffs shrink and crowd together into the haze
 * because the layer is falling away with the curvature of the earth.
 *
 * Wrapping it on a sphere is eight lines and it bounds the crossing *naturally*,
 * which was the actual reason grazing rays were a problem in the first place.
 * The horizon crossing comes out around 15 km against the slab's clamped 7, and
 * it arrives at a finite distance instead of at infinity.
 *
 * **1500 km, not the earth's 6371.** This is a free parameter and reinder picks
 * the same order of magnitude in `MdGfzh` for the same reason: it exaggerates the
 * convergence, so a deck reads as a deck within the few kilometres this scene
 * actually spans rather than needing a hundred. The horizon distance for a 1.4 km
 * layer falls from 134 km to 65 km, which is where a viewer expects to see the
 * puffs merge.
 */
const PLANET_RADIUS = 1.5e6;

/**
 * Floor on the ray's vertical component for the shell intersection.
 *
 * A sphere has an intersection for a ray pointed *down* as well — on the far
 * side, through the planet. Rather than test for that, rays at or below the
 * horizon are treated as horizon rays, which is what they look like anyway: the
 * layer converged into the haze. The sea draws over them regardless, and the
 * alpha fade below handles the last degree.
 */
const GRAZE_FLOOR = 0.0005;

/**
 * Longest path through the shell that is actually marched, in layer thicknesses.
 *
 * The sphere bounds the crossing where the flat slab could not — but it bounds it
 * at about 15 km near the horizon, and a fixed step count divided into that is
 * 1200 m a step through a 700 m layer at Medium. Adjacent pixels then land at
 * completely different heights *inside* the layer, so the vertical profile term
 * swings between them and the horizon band boils. `gallery-jitter` measured it:
 * far-field high-frequency energy at Medium went to 2.954 against a 2.33 ceiling,
 * while High — the same march at twice the steps — stayed inside its own.
 *
 * Capping the *integration* is not capping the geometry. The shell still curves,
 * the deck still converges, and `tEnter` still walks out to the horizon; what
 * stops is accumulating along a path whose far end the step count cannot resolve.
 * Seven thicknesses is past where the transmittance of anything worth drawing has
 * already closed, and it puts the horizon step at 400 m even at Medium.
 *
 * This is the same quantity the flat slab clamped at ten, and it is worth being
 * clear that its return is not a retreat: there it was load-bearing, hiding a
 * span that ran to infinity, and it needed a four-degree horizon fade on top to
 * cover what it could not. Here it is a sampling bound on a crossing that is
 * finite either way.
 */
const MAX_SPAN_FACTOR = 7;

/** Feature scale of the base noise: 1 noise unit ~= 1/NOISE_SCALE metres. */
const NOISE_SCALE = 0.00055;

/**
 * Feature scale of the weather field, and how far it moves the coverage
 * threshold.
 *
 * `uThreshold` is one scalar over the whole hemisphere, which is exactly why
 * `clear-day.png` and `waves.png` show an even field of near-identical puffs:
 * every part of the sky has the same coverage because the shader was told so.
 * A real sky has regions. `4dSBDt` gates its fine field on a very low-frequency
 * lookup for this, and 22 km is the scale at which a viewer reads "it is
 * clearing over there".
 *
 * Sampled once per main-march step and reused for that step's four light
 * samples, which are only a few hundred metres away — three orders of magnitude
 * inside the field's own feature size, so the reuse is exact to any precision
 * that matters and turns five noise calls a step into one.
 */
const WEATHER_SCALE = 1 / 22000;
const WEATHER_AMOUNT = 0.2;

/**
 * Multiple-scattering approximation: octaves, and how each one differs.
 *
 * A single Beer term drives thick cloud to flat grey, because it says all the
 * light that was not transmitted is gone. It is not — it scattered, and in a
 * medium with an albedo as close to 1 as a water cloud it scatters many times
 * before it leaves. That is why the interior of a cumulus glows instead of going
 * black, and a renderer without it produces exactly the flat lighting the gap
 * analysis describes.
 *
 * The approximation is Hillaire's (Frostbite, 2016): sum a few orders, each
 * seeing less extinction, contributing less energy, and being more isotropic
 * than the last. Three terms, no extra density samples — the optical depth is
 * already in hand and this is three more `exp`s on it.
 *
 * Normalised by the contribution sum so a sample at zero optical depth returns
 * what a single Beer term returned. The extra light therefore lands where it
 * physically belongs — deep in the cloud — rather than as a global brightening.
 */
const MS_OCTAVES = 3;
/** Energy of each successive order. */
const MS_ATTENUATION = 0.5;
/** Extinction each successive order sees. */
const MS_EXTINCTION = 0.42;
/** How much of the phase function each successive order keeps. */
const MS_PHASE = 0.55;
const MS_NORM = (() => {
  let sum = 0;
  for (let n = 0; n < MS_OCTAVES; n++) sum += Math.pow(MS_ATTENUATION, n);
  return sum;
})();

/**
 * Measured quantiles of `mx_fractal_noise_float(p, 4, 2, 0.5, 1) * 0.5 + 0.5`,
 * sampled over a 256^2 patch: the field is near-Gaussian around 0.5 with
 * sigma ~= 0.16, NOT uniform over 0..1.
 *
 * That matters because the naive `threshold = 1 - coverage` remap is then wildly
 * non-linear — coverage 0.32 would put the threshold at 0.68, i.e. above the
 * 90th percentile, leaving the sky essentially clear. Mapping coverage through
 * the measured inverse CDF instead makes the slider behave as "fraction of sky
 * covered", which is what the demo's Cloud Coverage control implies.
 *
 * Pairs are [coveredFraction, noiseThreshold], ascending in fraction.
 */
const COVERAGE_QUANTILES: ReadonlyArray<readonly [number, number]> = [
  [0.0, 1.05],
  [0.05, 0.761],
  [0.25, 0.612],
  [0.5, 0.502],
  [0.75, 0.388],
  [0.95, 0.239],
  [1.0, -0.06],
];

/** Softness of the cloud edge in noise units. Crisper = puffier cumulus. */
const EDGE_WIDTH = 0.045;

/** Strength of the single-scattering term with the sun fully above the horizon. */
const SUN_GAIN = 1.5;

/**
 * Bounds on the combined phase function.
 *
 * The raw Henyey-Greenstein spike at `g = 0.76` is about thirty at zero
 * scattering angle, which blows the disc around the sun to flat white. The
 * ceiling was 3.2, and the gap analysis is right that it was damaging the
 * forward peak: a silver lining is *supposed* to be several times the ambient,
 * and clipping it at 3.2 flattened the one feature that tells a viewer the sun
 * is behind that cloud. 6.5 keeps the lining and still stops the disc.
 */
const PHASE_MIN = 0.3;
const PHASE_MAX = 6.5;

/** Value of the normalised phase function for isotropic scattering. */
const PHASE_ISOTROPIC = 1;

function coverageToThreshold(coverage: number): number {
  const c = Math.min(1, Math.max(0, coverage));
  for (let i = 1; i < COVERAGE_QUANTILES.length; i++) {
    const [f1, t1] = COVERAGE_QUANTILES[i];
    if (c <= f1) {
      const [f0, t0] = COVERAGE_QUANTILES[i - 1];
      const k = f1 === f0 ? 0 : (c - f0) / (f1 - f0);
      return t0 + (t1 - t0) * k;
    }
  }
  return COVERAGE_QUANTILES[COVERAGE_QUANTILES.length - 1][1];
}

export class Clouds {
  readonly mesh: THREE.Mesh;

  private readonly params: CloudParams;
  private readonly geometry: THREE.SphereGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;

  /** Integrated wind displacement, metres. Reused — never reallocated. */
  private readonly windOffset = new THREE.Vector3();
  private readonly windVector = new THREE.Vector3(1, 0, 0);
  /** Accumulated evolution phase; see `update`. */
  private evolution = 0;
  private readonly uEvolution: any = uniform(new THREE.Vector3());

  // --- uniforms -------------------------------------------------------------
  /** Noise threshold derived from `coverage` on the CPU — see COVERAGE_QUANTILES. */
  private readonly uThreshold = uniform(0.66);
  private readonly uDensity = uniform(1);
  private readonly uAltitude = uniform(1400);
  private readonly uThickness = uniform(700);
  // Loosely typed on purpose — see the note in Atmosphere.ts. `uSteps` also has
  // to be usable as a dynamic `Loop` bound, which the typings do not model.
  private readonly uSteps: any = uniform(24, 'int');
  private readonly uInvSteps = uniform(1 / 24);
  private readonly uColor: any = uniform(new THREE.Color(1, 1, 1));
  private readonly uShadowColor: any = uniform(new THREE.Color(0.34, 0.38, 0.47));
  private readonly uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uWindOffset = uniform(new THREE.Vector3());
  /**
   * Extinction per metre of cloud for the ground shadow.
   *
   * Small, because it multiplies the *whole slab thickness*: at 700 m and a
   * density of 1 this gives exp(-1.05), so a solid cloud puts the sea at about a
   * third of full sun. That is roughly what a cumulus shadow measures.
   */
  private readonly uShadowStrength = uniform(0.0015);
  /** Extinction per metre of unit density. */
  private readonly uExtinction = uniform(0.0085);
  private readonly uLightStep = uniform(175);
  private readonly uSunGain = uniform(SUN_GAIN);
  private readonly uAmbientGain = uniform(0.42);
  /**
   * Ambient reaching the *top* of the layer.
   *
   * Written from the atmosphere's zenith colour, so the light a cloud top
   * receives is the sky that is actually over it. The base keeps
   * `shadowColor * ambientGain`, which is the darker, sea-facing end.
   */
  private readonly uAmbientTop: any = uniform(new THREE.Color(0.62, 0.72, 0.86));
  /** How far the weather field is allowed to move the coverage threshold. */
  private readonly uWeatherAmount = uniform(0);

  /** Shared haze, so the deck fades into the same air the island does. */
  private aerial: AerialPerspective | null = null;

  constructor() {
    this.params = {
      ...DEFAULT_CLOUD_PARAMS,
      color: DEFAULT_CLOUD_PARAMS.color.clone(),
      shadowColor: DEFAULT_CLOUD_PARAMS.shadowColor.clone(),
    };

    this.geometry = new THREE.SphereGeometry(1, 40, 24);

    this.material = new THREE.MeshBasicNodeMaterial();
    this.material.side = THREE.BackSide;
    this.material.depthWrite = false;
    this.material.depthTest = false;
    this.material.fog = false;
    // Deliberately NOT `transparent`. Transparent materials are drawn after the
    // whole opaque queue, which would put the clouds on top of the ocean and the
    // ship. `CustomBlending` keeps the mesh in the opaque queue — where
    // renderOrder still orders it right behind the sky — while still alpha
    // blending. Both the WebGPU and the WebGL2 backend honour this.
    this.material.transparent = false;
    this.material.blending = THREE.CustomBlending;
    this.material.blendEquation = THREE.AddEquation;
    this.material.blendSrc = THREE.SrcAlphaFactor;
    this.material.blendDst = THREE.OneMinusSrcAlphaFactor;
    this.material.blendEquationAlpha = THREE.AddEquation;
    this.material.blendSrcAlpha = THREE.OneFactor;
    this.material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;

    this.material.colorNode = this.buildCloudNode();
    this.material.positionNode = positionGeometry.mul(DOME_RADIUS).add(cameraPosition);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'cloud-layer';
    this.mesh.renderOrder = -900;
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;

    this.applyParams();
  }

  setParams(params: Partial<CloudParams>): void {
    if (params.color !== undefined) this.params.color.copy(params.color);
    if (params.shadowColor !== undefined) this.params.shadowColor.copy(params.shadowColor);
    if (params.coverage !== undefined) this.params.coverage = params.coverage;
    if (params.density !== undefined) this.params.density = params.density;
    if (params.altitude !== undefined) this.params.altitude = params.altitude;
    if (params.thickness !== undefined) this.params.thickness = params.thickness;
    if (params.windSpeed !== undefined) this.params.windSpeed = params.windSpeed;
    if (params.windDirection !== undefined) this.params.windDirection = params.windDirection;
    if (params.evolutionRate !== undefined) this.params.evolutionRate = params.evolutionRate;
    if (params.steps !== undefined) this.params.steps = params.steps;
    this.applyParams();
  }

  getParams(): Readonly<CloudParams> {
    return this.params;
  }

  /**
   * The haze the deck dissolves into. Must be set before the node is built to
   * take effect — `buildCloudNode` runs in the constructor — so this is a
   * constructor-time dependency expressed as a setter only because `Clouds` is
   * built before `Atmosphere` is.
   */
  setAerialPerspective(aerial: AerialPerspective): void {
    if (this.aerial === aerial) return;
    this.aerial = aerial;
    this.material.colorNode = this.buildCloudNode();
    this.material.needsUpdate = true;
  }

  /**
   * The sky the layer is lit by, from the same source the water and the dome
   * read. Uniform writes only.
   */
  setSkyColors(zenith: THREE.Color): void {
    // Cloud tops are bright: they see the whole upper hemisphere plus what the
    // deck around them bounces. The zenith radiance alone is the sky *behind*
    // them and reads as a grey top, so it is lifted toward the sunlit cloud
    // colour by a fixed fraction rather than used raw.
    this.uAmbientTop.value.copy(zenith).lerp(this.params.color, 0.45).multiplyScalar(1.35);
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.uSunDir.value.copy(dir).normalize();
    // Once the sun is under the horizon the single-scattering term has to go
    // with it, otherwise the deck stays lit like midday against a night sky.
    // Derived here rather than exposed, so callers only have to push the vector.
    const above = Math.min(1, Math.max(0, (this.uSunDir.value.y + 0.09) / 0.18));
    this.uSunGain.value = SUN_GAIN * above * above * (3 - 2 * above);
  }

  /**
   * Returns the wind advection to its origin.
   *
   * The offset is *accumulated*, not derived from a clock, so unlike the other
   * animated systems it cannot be reproduced by setting a time — it has to be
   * rewound explicitly for a capture to be repeatable.
   */
  resetWind(): void {
    this.windOffset.set(0, 0, 0);
    this.uWindOffset.value.copy(this.windOffset);
    this.evolution = 0;
    this.uEvolution.value.set(0, 0, 0);
  }

  update(dt: number): void {
    if (!this.mesh.visible) return;
    this.windOffset.addScaledVector(this.windVector, dt * this.params.windSpeed);
    // Wrap on the noise period so the offset never grows large enough to eat
    // float precision in a long-running session.
    const period = 1 / NOISE_SCALE;
    this.windOffset.x %= period;
    this.windOffset.z %= period;
    this.uWindOffset.value.copy(this.windOffset);

    // Evolution runs across the wind, so growth is not just more translation
    // wearing a different name.
    this.evolution += dt * this.params.evolutionRate;
    this.evolution %= 1000;
    this.uEvolution.value.set(this.evolution * 0.31, this.evolution, this.evolution * -0.19);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }

  // -------------------------------------------------------------------------

  private applyParams(): void {
    const p = this.params;

    this.uThreshold.value = coverageToThreshold(p.coverage);
    this.uDensity.value = Math.max(0, p.density);
    this.uAltitude.value = p.altitude;
    this.uThickness.value = Math.max(1, p.thickness);
    this.uColor.value.copy(p.color);
    this.uShadowColor.value.copy(p.shadowColor);

    const steps = Math.max(0, Math.min(MAX_STEPS, Math.round(p.steps)));
    // steps === 0 is the Low tier: the layer must cost literally nothing.
    this.mesh.visible = steps > 0;
    this.uSteps.value = Math.max(1, steps);
    this.uInvSteps.value = 1 / Math.max(1, steps);

    this.uLightStep.value = (p.thickness / LIGHT_STEPS) * 1.1;

    // The weather field vanishes at both ends of the coverage range, which is
    // not a fudge: a clear sky and a solid overcast are both *uniform*, and it is
    // only the broken states in between that have regions. Without the taper a
    // storm at 0.95 coverage would develop blue holes and a clear day would grow
    // a bank.
    const c = Math.min(1, Math.max(0, p.coverage));
    this.uWeatherAmount.value = WEATHER_AMOUNT * 4 * c * (1 - c);

    this.windVector.set(Math.cos(p.windDirection), 0, Math.sin(p.windDirection));
  }

  /**
   * Density at a world-space point. Written as a plain helper rather than a
   * `Fn` with a declared layout so it inlines at both call sites (main march and
   * light march) without needing an exact TSL signature.
   */
  /**
   * @param softness 0..1 LOD term. The march step grows with distance, so far
   *   clouds are sampled far below the Nyquist rate of the noise field and
   *   shimmer. Widening the density edge in step with the step size band-limits
   *   the field instead — distant decks go smooth rather than hatched.
   */
  /**
   * How much sunlight reaches a world point on the ground, 0..1.
   *
   * The single largest thing a cloud deck does to a landscape is put moving
   * patches of shade on it, and a sky with no shadow under it reads as a
   * backdrop rather than as weather. It is also nearly free here: walk from the
   * point along the sun direction to the middle of the cloud slab, ask the same
   * density function the march uses, and attenuate.
   *
   * One sample, not a march. A shadow through a kilometre of cloud does not need
   * an integral to look right — what it needs is to be *the same field* the
   * clouds are drawn from, so the shadow lands where the cloud is. Sampling the
   * slab's mid-height gives that for one lookup.
   *
   * Bound against the same uniforms as the cloud node, including the wind offset,
   * so the shade drifts with the deck that casts it.
   */
  /**
   * @param options.samples Points sampled through the layer. Three for a
   *   surface, where the shadow is evaluated once per pixel. One for a consumer
   *   whose own cost is already multiplied — the fog march calls this per cell,
   *   and the dressing compiles it into twenty separate materials.
   *
   * **The loop is a `Loop`, not a JavaScript `for`, and that is a compile-time
   * decision rather than a style one.** `densityAt` expands to a four-octave 3D
   * fractal noise, which is a large amount of code; unrolling three of them into
   * every consumer took the first frame of the whole scene from seconds to
   * minutes on the FXC path this project's test harness forces. A dynamic loop
   * emits the body once. The trade is that the sample count can no longer be a
   * compile-time constant, which costs nothing here because it never varies
   * within a build.
   */
  shadowNode(
    options: { samples?: number; octaves?: number } = {},
  ): (worldPosition: any) => any {
    const shadowSamples = Math.max(1, Math.round(options.samples ?? 3));
    const octaves = Math.max(1, Math.round(options.octaves ?? 4));

    return (worldPosition: any) => {
      const p = vec3(worldPosition).toVar('cloudShadowP');

      // Several samples through the layer, not one, and this is the fix for a
      // defect the previous implementation documented but did not solve: with
      // one sample the walk had to be floored on the sun's elevation, and below
      // about 14 degrees the floored distance stops reaching the slab at all —
      // so the sample lands *outside* the layer and the shadow **vanishes**
      // rather than deepening. Sunset and the tour's squall beat are exactly
      // those cases, and they are the two where a cloud shadow matters most.
      //
      // The march makes the floor unnecessary. Each sample is placed at a fixed
      // fraction of the way up the layer and the walk to it is computed
      // per-sample, so a grazing sun simply travels further between them; there
      // is no distance at which the samples leave the slab, because they are
      // defined by where they are *in* it.
      const acc = float(0).toVar('cloudShadowAcc');
      // Floored only against a division by zero. Rays from a sun on the horizon
      // travel enormously far, which is correct — that is why evening shadows
      // are long — and the sphere is not involved here because the shadow is a
      // local question about one point.
      const invSunY = float(1).div(this.uSunDir.y.max(0.06)).toVar('cloudShadowInvY');

      // Sampled once for the whole walk rather than per sample. The samples span
      // the layer's own thickness, which is a kilometre and a half at most, and
      // the weather field's features are twenty-two — so the three of them are
      // inside the same weather by three orders of magnitude, and asking again
      // would be two more octaves of noise for an identical answer.
      const weather = this.weatherAt(p).toVar('cloudShadowWeather');

      Loop(shadowSamples, ({ i }: any) => {
        const fraction = float(i).add(0.5).mul(1 / shadowSamples);
        const level = this.uAltitude.add(this.uThickness.mul(fraction));
        const t = level.sub(p.y).mul(invSunY).max(0);
        const hit = p.add(this.uSunDir.mul(t));
        // `softness` 1: the shadow is a low-frequency feature by nature and the
        // erosion octave would only alias across the sea surface. At full
        // softness its contribution is already zero, so declining to evaluate it
        // costs nothing and saves a three-octave 3D noise per consumer.
        acc.addAssign(this.densityAt(hit, float(1), weather, octaves));
      });

      // Beer-Lambert along the *sun path* through the layer, not down its
      // vertical thickness. With the sun low the ray crosses far more cloud than
      // the layer is deep, which is why shadows lengthen and deepen toward
      // evening. Bounded at sixteen thicknesses by the floor on `sun.y` above.
      const pathLength = this.uThickness.mul(invSunY).toVar();

      // Never to zero: a shaded sea is darker, not black, because the sky around
      // the cloud still lights it.
      return acc
        .mul(1 / shadowSamples)
        .mul(pathLength)
        .mul(this.uShadowStrength)
        .negate()
        .exp()
        .clamp(0.25, 1);
    };
  }

  /**
   * Height above the wrapped surface, metres, computed without cancellation.
   *
   * `length(p + (0, R, 0)) - R` is the obvious spelling and it is unusable in
   * float32: both terms are 1.5e6, so the subtraction throws away everything
   * below about a quarter of a metre. The algebraically identical
   * `(|p|² - R²) / (|p| + R)` never forms the difference of two large numbers,
   * and `|p|² - R²` expands to terms that are all small except `2 y R`.
   */
  private radialAltitude(p: any): any {
    const lateral = p.x.mul(p.x).add(p.z.mul(p.z));
    const numerator = lateral.add(p.y.mul(p.y)).add(p.y.mul(2 * PLANET_RADIUS));
    const radius = lateral.add(p.y.add(PLANET_RADIUS).mul(p.y.add(PLANET_RADIUS))).sqrt();
    return numerator.div(radius.add(PLANET_RADIUS));
  }

  /** Position in the layer, 0 at its base and 1 at its top. */
  private heightFraction(p: any): any {
    return this.radialAltitude(p).sub(this.uAltitude).div(this.uThickness);
  }

  /**
   * Distance along `rd` from `ro` to the shell at altitude `altitude`.
   *
   * Written as `-c / (b + sqrt(b² - c))` rather than as `-b + sqrt(b² - c)`.
   * They are the same root; the second one subtracts two numbers that agree to
   * five digits when the ray is steep, and loses most of the answer's precision
   * doing it. This form is a division of small by large and keeps all of it.
   */
  private shellDistance(ro: any, rd: any, altitude: any): any {
    // `b = dot(ro + (0,R,0), rd)`, with the large term kept separate so adding
    // R to ro.y cannot round ro.y away first.
    const b = ro.x
      .mul(rd.x)
      .add(ro.z.mul(rd.z))
      .add(ro.y.mul(rd.y))
      .add(rd.y.mul(PLANET_RADIUS))
      .toVar('cloudShellB');
    // `c = |ro + (0,R,0)|² - (R + altitude)²`, expanded so the R² terms cancel
    // symbolically instead of numerically.
    const c = ro.x
      .mul(ro.x)
      .add(ro.z.mul(ro.z))
      .add(ro.y.mul(ro.y))
      .add(ro.y.mul(2 * PLANET_RADIUS))
      .sub(altitude.mul(2 * PLANET_RADIUS))
      .sub(altitude.mul(altitude))
      .toVar('cloudShellC');

    const disc = b.mul(b).sub(c).max(0).sqrt().toVar('cloudShellDisc');
    return c.negate().div(max(b.add(disc), 1e-3));
  }

  /**
   * Coverage at the 20 km scale, so the sky has weather in it.
   *
   * Two octaves is enough: this exists to make one part of the hemisphere
   * clearer than another, and detail below its own feature size would only be a
   * second copy of the field the density function already has.
   */
  private weatherAt(p: any): any {
    const q = vec3(p.x, 0, p.z).sub(this.uWindOffset).mul(WEATHER_SCALE);
    return mx_fractal_noise_float(q, 2, 2.0, 0.5, 1.0).mul(0.5).add(0.5);
  }

  /**
   * @param baseOctaves Non-null selects the *coarse* evaluation: the erosion
   *   octave is skipped entirely and the base field runs at this many octaves
   *   instead of four. Null is the full field the layer is drawn from.
   *
   *   Skipping the erosion octave is free in the image for any caller passing
   *   `softness` 1, because at full softness its contribution is already
   *   multiplied by zero — but the noise call still happens, and a build-time
   *   flag is the only way not to pay for it. Cutting the *base* octaves is not
   *   free: it makes the field blobbier, so it is a choice each caller makes
   *   against its own cost. A shadow evaluated once per pixel can afford four;
   *   one evaluated inside another raymarch cannot.
   */
  private densityAt(p: any, softness: any, weather: any, baseOctaves: number | null = null): any {
    const h = this.heightFraction(p);

    // Flat base, rounded top — the cumulus profile in ref-default.png. Raising
    // the threshold toward the top and bottom of the slab (rather than only
    // scaling density) is what makes the puffs read as rounded volumes instead
    // of as a sheet with soft edges.
    const profile = smoothstep(0.0, 0.12, h).mul(smoothstepDown(h, 0.42, 1.0));

    const q = p.sub(this.uWindOffset).mul(NOISE_SCALE);
    const base = mx_fractal_noise_float(q, baseOctaves ?? 4, 2.0, 0.5, 1.0)
      .mul(0.5)
      .add(0.5);

    // A second, higher-frequency field erodes the billow edges so the silhouette
    // is not a smooth blob. Centred on zero so it breaks edges up without
    // shifting the overall coverage the threshold was calibrated for.
    //
    // It is also *evolved*, on its own clock, at right angles to the wind.
    // Translation alone cannot make a cloud look alive: features here are
    // kilometre-scale, so at any honest wind speed a puff takes minutes to cross
    // its own width and the layer reads as a painted backdrop being slid past.
    // What the eye actually reads as weather is the silhouette changing —
    // billows growing and eroding in place — and that is a second offset through
    // the noise field rather than a faster one along the wind.
    const detail = baseOctaves !== null
      ? null
      : mx_fractal_noise_float(
          q.mul(4.3).add(vec3(7.3, 2.1, 5.7)).add(this.uEvolution),
          3,
          2.0,
          0.5,
          1.0,
        ).mul(0.5);

    // The weather field moves the threshold, so coverage varies across the sky
    // instead of being one number for the whole hemisphere.
    const edge = this.uThreshold
      .add(float(1).sub(profile).mul(0.22))
      .add(weather.sub(0.5).mul(this.uWeatherAmount));
    const width = float(EDGE_WIDTH).add(softness.mul(0.25));

    // The erosion octave fades out as the march coarsens.
    //
    // `softness` already widened the edge with step size, which softens the
    // silhouette but does nothing about the detail field itself — and that field
    // runs at 4.3x the base frequency, so its features are a few hundred metres
    // across. The storm preset marches a 1400 m slab in 24 steps, nearly 60 m a
    // sample, and sampling a 400 m feature at 60 m intervals along a ray whose
    // direction varies smoothly across the screen is exactly the recipe for
    // moire — which is what the banding across the storm cloud deck was. Detail
    // the march cannot resolve is not detail, it is noise, so it is faded rather
    // than sampled. Same reasoning as the wave cascades' geometry fade.
    const detailFade = float(1).sub(softness).clamp(0, 1);
    // Billowed rather than added. A centred detail octave only nibbles the
    // silhouette symmetrically, which reads as fuzz; eroding *more* toward the
    // edge (where the base field is near the threshold) and less in the core is
    // what carves cauliflower lobes into a cumulus instead of fraying it.
    const erosion = detail === null
      ? null
      : detail.abs().mul(-1).add(0.13).mul(detailFade).mul(0.55);
    const field = erosion === null ? base : base.add(erosion);
    const shaped = smoothstep(edge, edge.add(width), field);

    return shaped.mul(profile).mul(this.uDensity);
  }

  private buildCloudNode(): any {
    return Fn(() => {
      const view = normalize(positionGeometry).toVar('cloudView');
      const ro = cameraPosition.toVar('cloudRo');

      // Rays at or below the horizon are marched as horizon rays. On a sphere
      // they would otherwise find the shell on the far side, through the planet;
      // and what they should look like *is* the horizon — the layer converged
      // into the haze — so the floor produces the right image as well as a safe
      // one. The alpha fade at the end removes the last degree, where the sea
      // takes over.
      const rd = normalize(
        vec3(view.x, max(view.y, GRAZE_FLOOR), view.z),
      ).toVar('cloudRd');

      const result = vec4(0, 0, 0, 0).toVar('cloudResult');

      const tEnter = this.shellDistance(ro, rd, this.uAltitude).max(0).toVar('tEnter');
      const tExit = min(
        this.shellDistance(ro, rd, this.uAltitude.add(this.uThickness)).max(0),
        tEnter.add(this.uThickness.mul(MAX_SPAN_FACTOR)),
      ).toVar('tExit');

      If(tExit.greaterThan(tEnter), () => {
        const stepSize = tExit.sub(tEnter).mul(this.uInvSteps).toVar('cloudStep');
        // Offsetting each pixel's first sample turns the raymarch's concentric
        // banding into fine noise, which reads as cloud texture instead of as
        // contour lines. Interleaved gradient noise rather than a plain hash:
        // a 2D hash of the pixel coordinate leaves visible diagonal hatching at
        // these step counts, IGN does not.
        const jitter = interleavedGradientNoise(screenCoordinate);
        const t = tEnter.add(stepSize.mul(jitter)).toVar('cloudT');

        const transmittance = float(1).toVar('cloudTr');
        const scattered = vec3(0, 0, 0).toVar('cloudScatter');

        const cosTheta = dot(rd, this.uSunDir).toVar('cloudCos');
        // Strong forward lobe + a weak backward lobe: the forward term is what
        // makes cloud edges glow when the sun is behind them.
        const phase = clamp(
          hg(cosTheta, 0.76).mul(0.75).add(hg(cosTheta, -0.2).mul(0.25)).mul(12.566),
          PHASE_MIN,
          PHASE_MAX,
        ).toVar('cloudPhase');

        const softness = clamp(stepSize.mul(0.0032), 0.0, 1.0).toVar('cloudLod');

        Loop(this.uSteps, () => {
          const p = ro.add(rd.mul(t)).toVar('cloudP');
          // One weather sample per step, shared with this step's light march —
          // see WEATHER_SCALE for why that is exact rather than approximate.
          const weather = this.weatherAt(p).toVar('cloudWeather');
          const d = this.densityAt(p, softness, weather).toVar('cloudD');

          If(d.greaterThan(0.002), () => {
            // --- optical depth toward the sun -------------------------------
            const lightAcc = float(0).toVar('cloudLightAcc');
            // The coarse field, which is what `densityAt` documents this caller
            // should have been using all along: "a shadow evaluated once per
            // pixel can afford four octaves; one evaluated inside another
            // raymarch cannot". This one is evaluated inside another raymarch —
            // LIGHT_STEPS samples at every one of `uSteps` march steps — and it
            // was taking the full four-octave base plus the three-octave erosion
            // field, seven octaves of noise, 72 times a pixel at High.
            //
            // Two octaves and no erosion is 2 against 7. What it costs is the
            // fine structure of a cloud's *self-shadowing*, and that is the one
            // place the loss does not read: this accumulates optical depth
            // toward the sun and is then fed through `exp`, so it is an integral
            // of the field rather than a sample of it, and integrating a
            // blobbier field gives nearly the same number. The silhouette, which
            // is what the eye actually reads, still comes from the full field in
            // the outer march.
            Loop(LIGHT_STEPS, ({ i }: any) => {
              const lp = p.add(this.uSunDir.mul(this.uLightStep.mul(float(i).add(1.0))));
              lightAcc.addAssign(this.densityAt(lp, softness, weather, LIGHT_OCTAVES));
            });
            const opticalDepth = lightAcc
              .mul(this.uLightStep)
              .mul(this.uExtinction)
              .toVar('cloudOd');

            // Powder term — darkens the optically thin parts of the cloud facing
            // the viewer, which reads as internal structure. First order only:
            // it is a near-surface effect, and the deeper orders are precisely
            // the light that has stopped caring where the surface was.
            const powder = float(1).sub(exp(d.mul(-2.4))).toVar('cloudPowder');

            const sunTerm = float(0).toVar('cloudSun');
            for (let n = 0; n < MS_OCTAVES; n++) {
              const attenuation = Math.pow(MS_ATTENUATION, n);
              const extinction = Math.pow(MS_EXTINCTION, n);
              const phaseWeight = Math.pow(MS_PHASE, n);
              const order = exp(opticalDepth.mul(-extinction))
                .mul(mix(float(PHASE_ISOTROPIC), phase, phaseWeight))
                .mul(attenuation);
              sunTerm.addAssign(n === 0 ? order.mul(powder) : order);
            }
            sunTerm.mulAssign(this.uSunGain.mul(1 / MS_NORM));

            // Ambient graded by depth in the layer.
            //
            // This was one constant — `shadowColor * ambientGain` — identical at
            // the base of a 1400 m column and at its top, and it is most of why
            // the deck read flat. The base of a cumulus sees the sea and its own
            // underside; the top sees the whole sky. Grading between the two is
            // the single substitution that makes a puff read as a volume.
            const height = this.heightFraction(p).clamp(0, 1).toVar('cloudH');
            const ambient = mix(
              this.uShadowColor.mul(this.uAmbientGain),
              this.uAmbientTop,
              height,
            ).toVar('cloudAmbient');

            const lum = ambient.add(this.uColor.mul(sunTerm)).toVar('cloudLum');

            const sampleT = exp(d.mul(stepSize).mul(this.uExtinction).negate());
            // Energy-conserving analytic integration over the step.
            scattered.addAssign(lum.mul(float(1).sub(sampleT)).mul(transmittance));
            transmittance.mulAssign(sampleT);
          });

          t.addAssign(stepSize);

          If(transmittance.lessThan(0.01), () => {
            Break();
          });
        });

        // The deck dissolves into the same haze the sea and the island dissolve
        // into, rather than into an exponential of its own. The layer has fog
        // disabled and is drawn on a camera-locked dome, so `scene.fogNode`
        // never reaches it and the shared function is called by hand.
        const lit =
          this.aerial === null
            ? scattered
            : this.aerial.apply(scattered, tEnter, rd);
        result.assign(vec4(lit, float(1).sub(transmittance)));
      });

      // Full alpha right down to the horizon line, and gone a third of a degree
      // below it where the sea takes over.
      //
      // The fade this replaced ran from 0.9 degrees to 4.3, and it was not a
      // horizon fade at all — it was hiding a flat slab that stretched to
      // infinity along a grazing ray. That is what left clear sky under the
      // cloud field in `clear-day.png`. The sphere bounds the crossing on its
      // own, so what is left here is only the job of not drawing cloud where the
      // water is.
      const alpha = clamp(result.w.mul(smoothstep(-0.006, -0.0005, view.y)), 0.0, 1.0);

      return vec4(result.xyz, alpha);
    })();
  }
}

/** Henyey–Greenstein phase, 1/(4*pi) normalised. */
function hg(cosTheta: any, g: number): any {
  const g2 = g * g;
  const denom = pow(float(1 + g2).sub(cosTheta.mul(2 * g)), 1.5);
  return float(1 - g2).div(max(denom, 1e-4)).mul(0.07957747154594767);
}
