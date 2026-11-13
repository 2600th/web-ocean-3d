import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  dot,
  exp,
  float,
  interleavedGradientNoise,
  max,
  min,
  mrt,
  mix,
  mx_fractal_noise_float,
  mx_noise_float,
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
  /**
   * How ragged the underside of the deck is, 0..1.
   *
   * Fair-weather cumulus have flat bases - the condensation level is one
   * altitude - so this is near zero for every clear preset. A storm deck is the
   * opposite: its base hangs in lobes and pouches (mammatus) and trails scud,
   * and a perfectly flat underside is exactly why the storm read as a grey lid.
   * See `densityAt`.
   */
  baseRoughness: number;
  /**
   * Amount of the high cirrus layer, 0..1. A 2D layer, not a march - see
   * `cirrusAt`. It is what gives a clear day depth above the cumulus, and what
   * stays lit pink after the sun has left the lower deck.
   */
  cirrus: number;
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
  baseRoughness: 0.1,
  cirrus: 0.4,
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

/**
 * How far the wind offset runs before it is wrapped, metres.
 *
 * The offset used to wrap at `1 / NOISE_SCALE` - one noise unit - on the
 * assumption that the field repeats there. It does not: MaterialX Perlin hashes
 * its integer lattice and has no period at all, so every wrap moved the whole
 * deck, its shadow and the weather field to a different part of the noise in
 * one frame. At a storm's wind that was a visible jump every minute or two.
 * There is no wrap distance that is seamless for a non-periodic field, so the
 * only honest choice is one that is never reached in a session: 200 km is two
 * hours at a gale, and still under 500 detail-noise units, which float32 holds
 * to better than 1e-4 of a feature.
 */
const WIND_WRAP = 200_000;

/**
 * Moonlight on the deck, relative to sunlight *at the same sky brightness* —
 * the gain is multiplied by `skyLevel`, so at night this is a multiple of the
 * night sky, enough to silver an edge without whitening the deck.
 */
const MOON_GAIN = 3.5;

/** Ambient gain on the base's shadow colour at a clear noon. */
const AMBIENT_GAIN = 0.42;

/**
 * Luminance of the mean of `Atmosphere.skyRadiance` at the zenith and around
 * the horizon, measured on the default preset at its own noon — the sky the
 * deck's ambient constants were tuned under. `skyLevel` is relative to it.
 */
const DAY_SKY_LUMA = 0.34;
/** Ceiling on `skyLevel`: a hazy bright sky may lift the deck, not bleach it. */
const SKY_LEVEL_MAX = 1.4;
const MOON_CLOUD_COLOR = new THREE.Color(0.62, 0.72, 0.95);

/**
 * Low-sun underside glow.
 *
 * The undersides of a sunset deck are lit *from below*: in the last minutes
 * before and after the sun touches the horizon its rays reach the cloud base
 * under the layer's own shadow. A sun march toward a sun a few degrees up
 * cannot find that light - it only ever walks up and sideways through the
 * layer - so without a term for it every sunset base is the same blue-grey it
 * is at noon, which is the opposite of what anyone photographs a sunset for.
 */
const UNDER_GLOW = 1.1;

/** Altitude of the cirrus sheet, metres above the wrapped surface. */
const CIRRUS_ALTITUDE = 7800;
/** Streak length and width scales for the cirrus noise, per metre. */
const CIRRUS_ALONG = 1 / 7000;
const CIRRUS_ACROSS = 1 / 700;
/** Peak opacity of the cirrus sheet. Ice cloud is thin; it veils, never hides. */
const CIRRUS_OPACITY = 0.5;
/** Cirrus rides the jet, well above the cumulus steering wind. */
const CIRRUS_WIND = 2.2;

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
  /**
   * Colour of the key light the deck is lit by - the sun's extinction colour by
   * day, moonlight by night. The preset's `color` is the cloud's own tint and
   * multiplies this rather than standing in for it, which is why a sunset deck
   * used to be lit by whatever the preset author guessed a sunset was.
   */
  private readonly uLightColor: any = uniform(new THREE.Color(1, 1, 1));
  /** See `UNDER_GLOW`. Colour and strength in one. */
  private readonly uUnderGlow: any = uniform(new THREE.Color(0, 0, 0));
  /** Horizontal unit vector toward the key light, for the underside glow. */
  private readonly uSunFlat = uniform(new THREE.Vector3(1, 0, 0));
  /** See `CloudParams.baseRoughness`. */
  private readonly uBaseRough = uniform(0.1);
  /** See `CloudParams.cirrus`. */
  private readonly uCirrusAmount = uniform(0.4);
  /**
   * Direct light on the cirrus. Separate from `uSunGain` because the sheet is
   * six kilometres higher and keeps the sun for a while after the cumulus
   * below it has lost it - that late pink veil is the point of having it.
   */
  private readonly uCirrusSun = uniform(1);
  /** Wind axis in the horizontal plane, for the cirrus streak direction. */
  private readonly uWindAxis = uniform(new THREE.Vector3(1, 0, 0));
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
  /**
   * Ambient reaching the *base* of the layer: the preset's shadow colour at
   * `AMBIENT_GAIN`, scaled by how bright the sky is. Written by `setSkyColors`.
   */
  private readonly uAmbientBottom: any = uniform(new THREE.Color(0.14, 0.16, 0.2));
  /** Sky brightness relative to a clear noon; see `setSkyColors`. */
  private skyLevel = 1;
  /**
   * The dome's mean radiance just above the horizon, which a distant
   * *continuous* deck fades toward. See the composite in `buildCloudNode`.
   */
  private readonly uHaze: any = uniform(new THREE.Color(0.6, 0.7, 0.8));
  /** 0 for broken cloud, 1 for a continuous deck; from the coverage. */
  private readonly uDeckHold = uniform(0);
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

  /**
   * Frame index for the temporal jitter, 0 when there is no temporal resolve.
   * A CPU counter rather than three's `frameId`, so `resetWind` rewinds it and
   * a deterministic capture sees the same sequence every time.
   */
  private readonly uJitterFrame = uniform(0);
  /** Edge band-limit per metre of march step. See its use in `buildCloudNode`. */
  private readonly uSoftnessPerMetre = uniform(0.0032);
  private jitterFrame = 0;
  private temporalJitter = false;

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
    if (params.baseRoughness !== undefined) this.params.baseRoughness = params.baseRoughness;
    if (params.cirrus !== undefined) this.params.cirrus = params.cirrus;
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
  /**
   * Correct motion vectors for the temporal resolve.
   *
   * Three's stock velocity reprojects `positionPrevious`, which is the *raw
   * geometry attribute* — the unit sphere — while the current position is the
   * `positionNode`, a 100 m dome moved to the camera. The two describe
   * different points, so every cloud pixel reported a velocity of a sizeable
   * fraction of the screen, TRAA read that as motion, and gave the current
   * frame full weight. The layer was never temporally resolved at all, which is
   * why its jittered march showed as diagonal hatching along every edge with
   * the resolve on or off.
   *
   * The layer is at infinity for reprojection purposes, so the right vector is
   * the direction's own: drop translation (w = 0) and reproject through this
   * frame's and last frame's view-projection. Camera translation then moves
   * nothing and rotation moves it exactly as far as the sky moves.
   *
   * Safe as a material MRT override only because the layer is on its own
   * render layer and excluded from the planar mirror: a material `mrtNode`
   * *replaces* the output of a render into a target without MRT, which is what
   * the reflector's target is.
   */
  setPreviousViewProjection(previous: any): void {
    this.material.mrtNode = mrt({
      velocity: Fn(() => {
        const dir = vec4(normalize(positionGeometry), 0);
        const current = (cameraProjectionMatrix as any).mul(cameraViewMatrix).mul(dir);
        const before = previous.mul(dir);
        return current.xy.div(current.w).sub(before.xy.div(before.w));
      })(),
    });
    this.material.needsUpdate = true;
  }

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
  setSkyColors(zenith: THREE.Color, skyRadiance?: THREE.Color, horizonRadiance?: THREE.Color): void {
    if (horizonRadiance !== undefined) (this.uHaze.value as THREE.Color).copy(horizonRadiance);
    // Broken cloud below about 0.6 coverage, a continuous deck from about 0.85.
    const c = this.params.coverage;
    const t = Math.min(1, Math.max(0, (c - 0.6) / 0.25));
    this.uDeckHold.value = t * t * (3 - 2 * t);

    // How bright the sky actually is, relative to a clear noon. See
    // `Atmosphere.skyRadiance`, and `DAY_SKY_LUMA` for the reference.
    //
    // Everything that lights the deck *from the sky* — both ambient terms, and
    // the moon, which lights the sky as well as the cloud — scales by this.
    // Without it the ambient was a fixed fraction of white whatever the hour,
    // so the night deck rendered near-white against a sky a few hundred times
    // darker: pale shapes on black, the inversion of a moonlit sky.
    if (skyRadiance !== undefined) {
      const skyLuma =
        skyRadiance.r * 0.2126 + skyRadiance.g * 0.7152 + skyRadiance.b * 0.0722;
      this.skyLevel = Math.min(SKY_LEVEL_MAX, skyLuma / DAY_SKY_LUMA);
    }

    // Cloud tops are bright: they see the whole upper hemisphere plus what the
    // deck around them bounces. The zenith radiance alone is the sky *behind*
    // them and reads as a grey top, so it is lifted toward the sunlit cloud
    // colour by a fixed fraction rather than used raw.
    //
    // Toward a *neutral* of the zenith's own brightness, not toward the cloud's
    // sunlit tint. The tint is the colour of direct sun on the cloud, and at
    // sunset that is peach: lifting the ambient toward it painted the shaded
    // flanks the same peach as the lit ones and the whole deck went one flat
    // pink. The shadow side of a sunset cumulus is the sky's colour — lavender
    // and blue — and the contrast between that and the gold on the lit side
    // is most of what makes a sunset sky read as glorious rather than tinted.
    const luma = zenith.r * 0.2126 + zenith.g * 0.7152 + zenith.b * 0.0722;
    _neutral.setRGB(1, 1, 1).multiplyScalar(luma * 2.2);
    this.uAmbientTop.value
      .copy(zenith)
      .lerp(_neutral, 0.45)
      .multiplyScalar(1.35 * this.skyLevel);
    (this.uAmbientBottom.value as THREE.Color)
      .copy(this.params.shadowColor)
      .multiplyScalar(AMBIENT_GAIN * this.skyLevel);
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.setLighting(dir, WHITE, dir, 0);
  }

  /**
   * The key light the deck is lit by, and the low-sun glow on its underside.
   *
   * By day that is the sun, in the sun's own colour; once the sun has gone the
   * same single-scattering path is handed to the moon at a fraction of the
   * gain, so a moonlit deck gets silver edges and a lit top instead of going to
   * a flat ambient grey. The hand-over happens where the sun's gain is already
   * zero, so there is no frame where the light visibly jumps.
   *
   * Uniform writes only, no allocation - called every frame.
   */
  setLighting(
    sunDir: THREE.Vector3,
    sunColor: THREE.Color,
    moonDir: THREE.Vector3,
    night: number,
  ): void {
    const sunY = sunDir.y / Math.max(1e-6, sunDir.length());
    // Once the sun is under the horizon the single-scattering term has to go
    // with it, otherwise the deck stays lit like midday against a night sky.
    const above = smooth01((sunY + 0.09) / 0.18);
    const color = this.uLightColor.value as THREE.Color;
    const dir = this.uSunDir.value as THREE.Vector3;
    if (above > 0.001 || night <= 0.001) {
      dir.copy(sunDir).normalize();
      this.uSunGain.value = SUN_GAIN * above;
      // A little white back in: the extinction colour is normalised to its
      // peak channel, and at the horizon that is a pure orange-red which would
      // paint the whole deck one hue. Real sunset cloud keeps a range from gold
      // on the tops to rose on the flanks, and that range is this lerp.
      color.copy(sunColor).lerp(WHITE, 0.18);
    } else {
      dir.copy(moonDir).normalize();
      const moonUp = smooth01((dir.y + 0.02) / 0.17);
      // Scaled with the sky, not absolute: the moon lights the sky as well as
      // the cloud, so a moonlit edge should stand a few times brighter than
      // the night sky beside it — silver, not the noon-white an absolute gain
      // gave against a sky that renders at a few thousandths.
      this.uSunGain.value = SUN_GAIN * MOON_GAIN * moonUp * Math.min(1, night) * this.skyLevel;
      color.copy(MOON_CLOUD_COLOR);
    }

    const flat = this.uSunFlat.value as THREE.Vector3;
    flat.set(dir.x, 0, dir.z);
    if (flat.lengthSq() < 1e-8) flat.set(1, 0, 0);
    flat.normalize();

    // Low-sun underside glow: on from just below the horizon, gone by about
    // fifteen degrees up. And it fades with coverage - under a solid deck
    // nothing gets underneath, which is why a storm sunset is a glow in a slot
    // of clear sky rather than a lit ceiling.
    //
    // Through civil twilight too, when the sun is below the sea but still
    // lighting the deck from underneath — scaled by how bright the sky still
    // is, so the glow fades with the twilight rather than outlasting it.
    const low =
      smooth01((sunY + 0.2) / 0.15) *
      (1 - smooth01((sunY - 0.06) / 0.22)) *
      Math.min(1, this.skyLevel);
    const open = Math.min(1, Math.max(0, (1 - this.params.coverage) * 2.4));
    (this.uUnderGlow.value as THREE.Color)
      .copy(sunColor)
      .multiplyScalar(UNDER_GLOW * low * open);

    // The cirrus sits six kilometres higher and sees the sun a few degrees
    // longer; past that it takes the moon's light like the deck does.
    const cirrusAbove = smooth01((sunY + 0.16) / 0.2);
    this.uCirrusSun.value =
      cirrusAbove > 0.001 || night <= 0.001
        ? cirrusAbove
        : MOON_GAIN * smooth01((moonDir.y + 0.02) / 0.17) * Math.min(1, night) * this.skyLevel;
  }

  /**
   * Returns the wind advection to its origin.
   *
   * The offset is *accumulated*, not derived from a clock, so unlike the other
   * animated systems it cannot be reproduced by setting a time — it has to be
   * rewound explicitly for a capture to be repeatable.
   */
  resetWind(): void {
    this.jitterFrame = 0;
    this.uJitterFrame.value = 0;
    this.windOffset.set(0, 0, 0);
    this.uWindOffset.value.copy(this.windOffset);
    this.evolution = 0;
    this.uEvolution.value.set(0, 0, 0);
  }

  /**
   * Whether a temporal resolve follows, so the march offset may vary per frame.
   * See the jitter note in `buildCloudNode`.
   */
  setTemporalJitter(enabled: boolean): void {
    this.temporalJitter = enabled;
    if (!enabled) this.uJitterFrame.value = 0;
    this.uSoftnessPerMetre.value = enabled ? 0.0024 : 0.0032;
  }

  update(dt: number): void {
    if (!this.mesh.visible) return;
    if (this.temporalJitter) {
      // Sixteen distinct offsets is enough for the resolve to average, and a
      // short cycle keeps the uniform small and exactly reproducible.
      this.jitterFrame = (this.jitterFrame + 1) % 16;
      this.uJitterFrame.value = this.jitterFrame;
    }
    this.windOffset.addScaledVector(this.windVector, dt * this.params.windSpeed);
    // Wrap on the noise period so the offset never grows large enough to eat
    // float precision in a long-running session.
    this.windOffset.x %= WIND_WRAP;
    this.windOffset.z %= WIND_WRAP;
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
    (this.uWindAxis.value as THREE.Vector3).copy(this.windVector);
    this.uBaseRough.value = Math.min(1, Math.max(0, p.baseRoughness));
    this.uCirrusAmount.value = Math.min(1, Math.max(0, p.cirrus));
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

    // How far the underside is lifted here, as a fraction of the layer.
    //
    // Driven by the noise already in hand, so it costs nothing: where the field is
    // strong the column hangs all the way to the condensation level, and where
    // it is weaker the base is drawn up into the layer. Under a fair-weather
    // sky `baseRoughness` is near zero and the bases stay flat, as cumulus
    // bases are. Under a storm deck — coverage near one, every column dense —
    // it is the only thing that gives the underside any relief at all: the
    // strong cells hang down as lobes and pouches between recesses, and the
    // height-graded ambient below then shades each lobe darker than the
    // recess above it. That shading is the mammatus texture; without the lift
    // the base was a single flat plane at one altitude and one grey.
    //
    // Two scales. The base field sets which columns hang at all, over a couple
    // of kilometres; the detail octave, where it is resolved, carves each
    // hanging column into pouches a few hundred metres across — its zero
    // crossings are creases drawn up into the deck and its cell interiors hang
    // as rounded lobes, which is the mammatus shape rather than a lumpy floor.
    // The light march takes the coarse field with no detail, so it sees only
    // the first scale; the self-shadowing of each pouch is a detail it cannot
    // resolve anyway.
    const pouches =
      detail === null
        ? float(0)
        : float(0.16).sub(detail.abs()).clamp(0.0, 0.16).mul(float(1).sub(softness).clamp(0, 1));
    const lift = float(0.62)
      .sub(base)
      .clamp(0.0, 0.3)
      .add(pouches)
      .mul(this.uBaseRough);

    // Flat base, rounded top — the cumulus profile in ref-default.png. Raising
    // the threshold toward the top and bottom of the slab (rather than only
    // scaling density) is what makes the puffs read as rounded volumes instead
    // of as a sheet with soft edges.
    const profile = smoothstep(lift, lift.add(0.12), h).mul(smoothstepDown(h, 0.42, 1.0));

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
        //
        // A *static* IGN still hatches, though — diagonally, along the soft rim
        // of every cloud, where the step offset decides whether a sample lands
        // in the edge or past it. With the temporal resolve on, the pattern is
        // shifted every frame (Jimenez's 5.588 pixel offset per frame) and TRAA
        // integrates it into a smooth edge instead of a ruled one. Off without
        // a resolve, where a moving pattern would only be visible grain.
        const jitter = interleavedGradientNoise(
          screenCoordinate.xy.add(this.uJitterFrame.mul(5.588238)),
        );
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

        // 0.0024 per metre of step, down from 0.0032, when the march offset
        // became temporal. The band-limit exists because a static offset
        // aliased the detail octave into moire; once TRAA is averaging sixteen
        // offsets, the march resolves finer detail than one frame of it can, and
        // the old setting was blurring every mid-distance cumulus into a soft
        // ball. Without the resolve the layer keeps the old softness — see
        // `setTemporalJitter`.
        const softness = clamp(stepSize.mul(this.uSoftnessPerMetre), 0.0, 1.0).toVar('cloudLod');

        // Powder only away from the sun.
        //
        // The powder term darkens the optically thin parts of a cloud, and that
        // is right when the light is behind the viewer: the rim of a cumulus
        // seen with the sun at your back is darker than its core. Looking
        // *toward* the sun it is exactly backwards — the thin rim is where the
        // forward-scattered light gets out, and darkening it erased the silver
        // lining the phase function had just put there.
        const powderView = cosTheta.mul(-0.5).add(0.5).toVar('cloudPowderView');

        // How squarely this ray looks toward the low sun's bearing, for the
        // underside glow. Horizontal only: the glow is the sunset sky's colour
        // on the bases, and it is strongest on the side of the deck the sun is
        // setting behind.
        const flatLength = rd.x.mul(rd.x).add(rd.z.mul(rd.z)).sqrt().max(1e-3);
        const glowBearing = rd.x
          .mul(this.uSunFlat.x)
          .add(rd.z.mul(this.uSunFlat.z))
          .div(flatLength)
          .mul(0.5)
          .add(0.5);
        const glowView = glowBearing.mul(glowBearing).mul(glowBearing).mul(0.7).add(0.3).toVar('cloudGlowView');

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
              sunTerm.addAssign(n === 0 ? order.mul(mix(float(1), powder, powderView)) : order);
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
              this.uAmbientBottom,
              this.uAmbientTop,
              height,
            ).toVar('cloudAmbient');

            // The low sun under the deck. Strongest at the very base and gone
            // by half way up; and attenuated by how much cloud lies between the
            // sample and the sun, which at a grazing sun is a horizontal path
            // — so the glow catches the exposed undersides and flanks rather
            // than the interior of a lobe.
            const below = float(1).sub(height);
            const underGlow = this.uUnderGlow
              .mul(below.mul(below))
              .mul(glowView)
              .mul(exp(opticalDepth.mul(-0.3)).mul(0.65).add(0.35));

            const lum = ambient
              .add(this.uColor.mul(this.uLightColor).mul(sunTerm))
              .add(underGlow)
              .toVar('cloudLum');

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
        //
        // `scattered` is premultiplied — it is radiance already weighted by
        // the coverage it came through — and the blend state multiplies by
        // alpha again. Writing it out raw therefore drew every partly
        // transparent sample at coverage squared: the thin rim of a cloud came
        // out darker than both its core and the sky behind it, which is a
        // dark fringe where a sunlit cumulus should have its brightest edge.
        // Un-premultiplying here (and re-premultiplying for the composite with
        // the cirrus below) restores it; opaque cores are unchanged.
        //
        // The haze takes the deck by *opacity*, not by colour: a distant cloud
        // dissolves into the dome behind it. Mixing it toward the shared
        // haze's inscatter instead painted every far cloud the haze's colour,
        // and that colour is built from the CPU-side sky tints, which at night
        // are a mid blue hundreds of times brighter than the dome — the pale
        // deck on a black sky in the night watch. The dome is the right
        // radiance in every direction at every hour, so fading to it is.
        //
        // Except for a continuous deck. Fading a storm ceiling to transparent
        // uncovered the dome's sun glow behind it — a bright hole where the
        // cloud is thickest — because a deck has no gaps for the dome to show
        // through: the air between the viewer and a distant overcast is lit by
        // the overcast. So, by how continuous the deck is, a far cloud keeps
        // its opacity and its *colour* goes to `uHaze` instead, the dome's own
        // measured radiance near the horizon, which is right at every hour for
        // the same reason the dome is.
        const straight: any = scattered.div(float(1).sub(transmittance).max(1e-4));
        const air: any = this.aerial === null ? float(1) : this.aerial.transmittance(tEnter, rd);
        const hold: any = this.uDeckHold;
        const hazed: any = mix(this.uHaze, straight, air);
        const lit: any = mix(straight, hazed, hold);
        const coverage: any = float(1)
          .sub(transmittance)
          .mul(mix(air, float(1), hold))
          .toVar('cloudCoverage');
        result.assign(vec4(lit.mul(coverage), coverage));
      });

      // The cirrus sheet, behind the cumulus. Premultiplied "over": it shows
      // only through what the deck in front of it lets through, and costs
      // nothing where that is nothing.
      If(this.uCirrusAmount.greaterThan(0.001).and(result.w.lessThan(0.995)), () => {
        const cirrus = this.cirrusAt(ro, rd, view).toVar('cirrus');
        const behind = float(1).sub(result.w);
        result.assign(
          vec4(result.xyz.add(cirrus.xyz.mul(behind)), result.w.add(cirrus.w.mul(behind))),
        );
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

      return vec4(result.xyz.div(result.w.max(1e-4)), alpha);
    })();
  }

  /**
   * The high cirrus veil, premultiplied: `(radiance * a, a)`.
   *
   * One shell intersection and one fractal noise — no march. Ice cloud at eight
   * kilometres is optically thin and has no self-shadowing worth integrating;
   * what it needs is its *shape*, which is long fibrous streaks laid along the
   * upper wind, and its *light*, which is the sun's colour for a few minutes
   * longer than the deck below gets it. Both are a lookup.
   *
   * The streaks come from sampling the noise stretched about five to one along
   * the wind axis, with the across-wind coordinate bent by a slower noise so the
   * filaments curve and hook instead of ruling the sky in straight lines. A
   * large-scale mask breaks the sheet into patches, so it reads as weather on
   * the way rather than as a texture over everything.
   */
  private cirrusAt(ro: any, rd: any, view: any): any {
    const t = this.shellDistance(ro, rd, float(CIRRUS_ALTITUDE)).toVar('cirrusT');
    const p = ro.add(rd.mul(t));
    const drift = vec3(p.x, 0, p.z).sub(this.uWindOffset.mul(CIRRUS_WIND)).toVar('cirrusDrift');
    const axis = this.uWindAxis;
    const along = drift.x.mul(axis.x).add(drift.z.mul(axis.z)).toVar('cirrusAlong');
    const across = drift.z.mul(axis.x).sub(drift.x.mul(axis.z)).toVar('cirrusAcross');

    const patch = mx_noise_float(vec3(along.mul(1 / 30000), across.mul(1 / 30000), 3.7))
      .mul(0.5)
      .add(0.5);
    const mask = smoothstep(float(0.62).sub(this.uCirrusAmount.mul(0.3)), 0.78, patch);

    const bend = mx_noise_float(vec3(along.mul(1 / 9000), across.mul(1 / 9000), 1.3)).mul(3.0);
    const streak = mx_fractal_noise_float(
      vec3(
        along.mul(CIRRUS_ALONG),
        across.mul(CIRRUS_ACROSS).add(bend),
        this.uEvolution.y.mul(0.35),
      ),
      4,
      2.3,
      0.55,
      1.0,
    )
      .mul(0.5)
      .add(0.5);

    // Breaks each filament into tufts — the hooks and wisps of mares' tails —
    // so a streak fades in and out along its length instead of running the
    // width of the sky at one strength.
    const tufts = smoothstep(
      -0.25,
      0.55,
      mx_noise_float(vec3(along.mul(1 / 2600), across.mul(1 / 2600), this.uEvolution.x.mul(0.5))),
    );

    // Thinned toward the horizon, where the sheet is foreshortened below the
    // sampling rate of a screen pixel and the streaks would only shimmer.
    const density = smoothstep(0.5, 0.84, streak)
      .mul(tufts)
      .mul(mask)
      .mul(this.uCirrusAmount)
      .mul(smoothstep(0.012, 0.09, view.y));
    const alpha = density.mul(CIRRUS_OPACITY).toVar('cirrusAlpha');

    // Thin ice is strongly forward-scattering: a bright halo of veil around the
    // sun, plain white away from it.
    const cosSun = dot(rd, this.uSunDir).max(0);
    const forward = pow(cosSun, 10).mul(2.2).add(cosSun.mul(cosSun).mul(0.35));
    const radiance = this.uColor
      .mul(this.uLightColor)
      .mul(this.uCirrusSun.mul(forward.add(0.85)))
      .add(this.uAmbientTop.mul(0.5));
    const faded: any = this.aerial === null ? alpha : alpha.mul(this.aerial.transmittance(t, rd));
    return vec4(radiance.mul(faded), faded);
  }
}

const WHITE = /*@__PURE__*/ new THREE.Color(1, 1, 1);
const _neutral = /*@__PURE__*/ new THREE.Color();

function smooth01(x: number): number {
  const t = x < 0 ? 0 : x > 1 ? 1 : x;
  return t * t * (3 - 2 * t);
}

/** Henyey–Greenstein phase, 1/(4*pi) normalised. */
function hg(cosTheta: any, g: number): any {
  const g2 = g * g;
  const denom = pow(float(1 + g2).sub(cosTheta.mul(2 * g)), 1.5);
  return float(1 - g2).div(max(denom, 1e-4)).mul(0.07957747154594767);
}
