import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  cameraPosition,
  cameraProjectionMatrix,
  clamp,
  dot,
  exp,
  float,
  fract,
  int,
  max,
  min,
  mrt,
  mix,
  mx_noise_float,
  nodeObject,
  normalize,
  positionGeometry,
  pow,
  screenCoordinate,
  screenSize,
  smoothstep,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import type { AerialPerspective } from './AerialPerspective';
import { BLUE_NOISE_SIZE, blueNoiseTexture } from './blueNoise';

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
 * Density is entirely procedural (`cloudFbm`, rotated Perlin) — there is no 3D
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
 *
 * **Where it is drawn.** The march does not run on the dome any more. It runs
 * on a full-screen quad into its own target at the scene pass's resolution,
 * with a blue-noise start offset that moves every frame; a second quad
 * reprojects last frame's result by view direction, clamps it to this frame's
 * 3x3 neighbourhood and blends; and the dome only *composites* that history,
 * adds the haze and the cirrus, and keeps its place in the opaque queue and its
 * motion vectors. This is the structure of every shipped real-time cloud
 * renderer — Schneider's Horizon Zero Dawn clouds ("The Real-time Volumetric
 * Cloudscapes of Horizon Zero Dawn", SIGGRAPH 2015: a sixteen-frame
 * reprojection) and Hillaire's Frostbite clouds ("Physically Based Sky,
 * Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016: a temporal
 * integration of jittered marches) — and the reason is the one this layer
 * found the hard way: a single march of 18 steps cannot sample a cloud edge
 * that is thinner than a step, and the scene's TRAA, which is tuned to keep
 * geometry sharp, does not average a ray-march offset away. See
 * `renderPasses`.
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
 * One extra light sample, at the full field with its erosion, this far toward
 * the sun, and its weight in the optical depth.
 *
 * The coarse light march above sees no billows, so every lobe of a cumulus was
 * lit as if the cloud were a smooth blob: one gradient from the lit side to the
 * shaded one, and from below a flat grey base. That is the "melted" overhead
 * deck in `clouds-1`. A billow is ~330 m across (`DETAIL_FREQUENCY` on
 * `NOISE_SCALE`); a sample a quarter of that toward the sun lands inside the
 * neighbouring lobe when there is one and in clear air when the sample is on a
 * lobe's sunward face, so each lobe gets its own lit cap and shaded underside —
 * Schneider's Nubis does the same with its first, full-detail light sample.
 * 1.5 rather than 1 because one sample stands for the near path it skips.
 * Measured cost under "GPU" in the round-2 report.
 */
const NEAR_LIGHT_DISTANCE = 90;
const NEAR_LIGHT_WEIGHT = 1.5;
/**
 * Extra extinction on the path toward the sun only. Above 1 deepens a cloud's
 * self-shadow — darker flanks and bases under the same lit tops — without
 * making it any more opaque to the eye.
 */
const LIGHT_EXTINCTION_GAIN = 1.0;
/**
 * Extinction toward the eye, relative to `uExtinction`. 4, from 3: a denser
 * surface toward the eye is a harder one, and the billow shading above only
 * reads if the eye stops near the lobe it lights rather than averaging a few
 * hundred metres of them.
 */
const VIEW_EXTINCTION_GAIN = 4.0;

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
// Not domain-warped, and that was tried. web-ocean breaks its cumulus cells up
// with a low-frequency warp (`lowEdge`), aimed here at the "cobblestone" rows
// near the horizon (`clouds-2` in the head-to-head). Measured on its own branch
// state: at web-ocean's 1.4 cells the near cumulus smeared into melted shapes;
// at 0.6 the far rows were only partly broken; and the two extra noise calls
// per march step cost 1.5-2 ms GPU at High 1080p (interleaved A/B, sky-day-b
// and clear-day-wide). Not worth it for that result.
//
// What the rows actually were is in `cloudFbm`: a lattice, not a lack of warp.

/**
 * Lacunarity, and the rotation every octave is turned through, for `cloudFbm`.
 *
 * The field this replaced was `mx_fractal_noise_float(q, 4, 2.0, ...)` on an
 * unrotated domain, and that has a regular grid in it. Perlin noise is zero at
 * every integer lattice point; with a lacunarity of exactly 2 every octave's
 * lattice contains the base octave's, so on the base lattice the *whole* fractal
 * is zero at once. The layer at 1400-2100 m is 0.77-1.16 noise units up, so it
 * straddles the lattice plane y = 1, where the field is pinned to its mean on a
 * 1.8 km square grid. A threshold above the mean then puts a hole on every grid
 * point, and in perspective a square grid of holes is rows of equal puffs
 * marching to the horizon — the "tiled rows" in nearly every head-to-head shot.
 *
 * web-ocean's `cloudFbm` (github.com/Noniv/web-ocean `js/wgsl/sky.js`, MIT) does
 * the standard thing: rotate the domain every octave, scale by 2.03 rather than
 * 2, and offset, so no two octaves share a lattice and none of them is aligned
 * with the layer. This is that, in 3D: the same 0.8 / 0.6 rotation applied
 * about the horizontal x axis and then about the vertical one, which tilts the
 * lattice planes off the horizontal and turns their rows off the axes. It costs
 * a matrix multiply per octave on top of the noise call.
 */
const FBM_LACUNARITY = 2.03;
const FBM_ROT = (() => {
  // Rx(atan2(0.6, 0.8)) then Ry(atan2(0.6, 0.8)): rows of the product.
  const c = 0.8;
  const s = 0.6;
  const rx = [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
  const ry = [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
  const m: number[][] = [];
  for (let i = 0; i < 3; i++) {
    m.push([0, 1, 2].map((j) => ry[i][0] * rx[0][j] + ry[i][1] * rx[1][j] + ry[i][2] * rx[2][j]));
  }
  return m;
})();

function rotateFbm(p: any): any {
  return vec3(
    dot(vec3(FBM_ROT[0][0], FBM_ROT[0][1], FBM_ROT[0][2]), p),
    dot(vec3(FBM_ROT[1][0], FBM_ROT[1][1], FBM_ROT[1][2]), p),
    dot(vec3(FBM_ROT[2][0], FBM_ROT[2][1], FBM_ROT[2][2]), p),
  );
}

/**
 * Rotated fractal Perlin noise, amplitude halving per octave, MaterialX's
 * normalisation: the same noise at the same octave weights as the field the
 * `COVERAGE_QUANTILES` were measured on, which is what the table depends on
 * (rotation and a 2.03 lacunarity do not change a sum's spread). The table was
 * not re-measured. A `Fn` with a
 * layout, so every caller shares one emitted function rather than inlining four
 * noise calls; `shadowNode` is compiled into a dozen materials.
 */
const cloudFbm: any = /*@__PURE__*/ Fn(([pIn, octavesIn]: any) => {
  const p = rotateFbm(vec3(pIn)).toVar();
  const octaves = int(octavesIn).toVar();
  const sum = float(0).toVar();
  const amplitude = float(1).toVar();
  Loop(octaves as any, () => {
    sum.addAssign(amplitude.mul(mx_noise_float(p)));
    amplitude.mulAssign(0.5);
    p.assign(rotateFbm(p).mul(FBM_LACUNARITY).add(vec3(5.3, 1.7, 9.1)));
  });
  return sum;
}).setLayout({
  name: 'cloudFbm',
  type: 'float',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'octaves', type: 'int' },
  ],
});

/**
 * The weather map: where there is cloud, and how tall it grows.
 *
 * Schneider's weather texture, procedurally: a 2D field over the sea, three
 * octaves from 16 km down to 4 km, that moves the coverage threshold and sets
 * each column's cloud type. It replaces a 22 km 3D field that moved the
 * threshold by about 0.05, which was too weak and too large to do what a
 * weather map is for — clusters of cells and the gaps between them, at the
 * scale of a few cells. Without that, every puff is the same size at the same
 * spacing as far as the eye can see, and that is a tiled sky whatever the noise.
 *
 * 2D Perlin, which is half the corners of the 3D noise it replaced, with
 * web-ocean's 2D rotation and 2.03 lacunarity between octaves (`cloudFbm`,
 * MIT). Sampled once per main-march step and shared with that step's light
 * samples, which are a few hundred metres away inside a 4 km feature.
 */
const weatherFbm: any = /*@__PURE__*/ Fn(([pIn]: any) => {
  const p = vec2(pIn).toVar();
  const sum = float(0).toVar();
  let amplitude = 1;
  for (let o = 0; o < 3; o++) {
    sum.addAssign(mx_noise_float(p).mul(amplitude));
    // q = ROT * q * 2.03 + (13.1, 7.7), ROT = (0.8, 0.6; -0.6, 0.8).
    p.assign(vec2(p.x.mul(0.8).sub(p.y.mul(0.6)), p.x.mul(0.6).add(p.y.mul(0.8))).mul(FBM_LACUNARITY).add(vec2(13.1, 7.7)));
    amplitude *= 0.5;
  }
  return sum.div(1.75);
}).setLayout({
  name: 'cloudWeatherFbm',
  type: 'float',
  inputs: [{ name: 'p', type: 'vec2' }],
});

const WEATHER_SCALE = 1 / 16000;
/**
 * How far the weather map moves the coverage threshold at mid coverage. The
 * map's spread is about +-0.15, so this is about +-0.13 of threshold against
 * the base field's 0.16 sigma: clear stretches and packed clusters in one view,
 * which is what web-ocean's own low layer gets from its large-scale term
 * (`large`, 0.7 on its edge). Tapered to nothing at both ends of the coverage
 * range; see `applyParams`.
 */
const WEATHER_AMOUNT = 0.9;
/**
 * Lowest cloud top, as a fraction of the layer, where the weather map is
 * thinnest. Sparse cells stay flat and small, as fair-weather cumulus do; where
 * the map is high they build to the full thickness. Two cloud types in one
 * deck instead of one shape repeated is most of what breaks the rows up.
 */
const TOP_MIN = 0.5;
/** Frequency of the erosion field relative to the base, and its strength. */
const DETAIL_FREQUENCY = 5.5;
/** 1.15, from 0.75 with the switch to billow noise (see `densityFrom`). */
const DETAIL_GAIN = 1.15;
/**
 * The billow erosion is `(min(|detail|, EROSION_CAP) - EROSION_OFFSET) * gain`.
 *
 * The offset is a coverage calibration, measured: with the dome hidden, the
 * fraction of the upper frame that is cloud at the old ridged erosion was
 * 0.637 (`sky-day`) and 0.707 (`clouds-day`); billow erosion at offsets 0.07,
 * 0.10, 0.12 and 0.14 gave 0.767 / 0.819, 0.690 / 0.748, 0.641 / 0.700 and
 * 0.600 / 0.646. 0.12 keeps the coverage the quantile table maps to.
 */
const EROSION_CAP = 0.2;
const EROSION_OFFSET = 0.12;
/**
 * Most the erosion octave can *raise* the field. A sample whose base field is
 * this far under its edge is empty whatever the detail says, so the march
 * skips the detail noise, the profile and the light march for it. Exact, not a
 * heuristic.
 */
const EROSION_LIFT_MAX = (EROSION_CAP - EROSION_OFFSET) * DETAIL_GAIN;

/**
 * Thinning of the far deck, as a rise in the coverage threshold with distance.
 *
 * Past about 20 km a fair-weather deck is seen almost edge-on through tens of
 * kilometres of haze; what reaches the eye is a few of the largest cells, not
 * every one of them stacked. Drawing every cell to 65 km is what filled the
 * horizon with rows and buried a setting sun. Raised, not faded: the far cells
 * that remain stay as crisp as the near ones and the gaps between them open to
 * the sky. Off for a continuous deck (`uDeckHold`), which has no gaps.
 *
 * web-ocean does the same for its low layer ("clear band near the horizon,
 * where the low sun shines under the clouds": density times
 * `1 - smoothstep(18, 36, t)` km, `js/wgsl/sky.js`, MIT). Nearer here because
 * this planet is a quarter the size, which puts its horizon half as far.
 */
/**
 * And a fade of what is left into the dome beyond 12 km, on top of the shared
 * haze. The shared haze is tuned for the sea and the island, which end a few
 * kilometres out; at the 30-60 km of the horizon band it still left every cell
 * at full contrast, with a dark base line under each, and those lines stacked
 * are the rows. Through that much marine boundary layer a cumulus field reads
 * as a pale, low-contrast band — which is what both reference projects draw.
 * From 8 km over 16, since round 2 (was 12 over 30): with the sky brighter
 * the horizon rows stood out again against the paler band behind them.
 */
const FAR_HAZE_START = 8000;
const FAR_HAZE_SCALE = 16000;
const THIN_NEAR = 7000;
const THIN_FAR = 34000;
const THIN_AMOUNT = 0.12;

/**
 * Per-frame weight of the new march in the temporal resolve. About ten frames
 * of history: enough to integrate a 1/10 of the blue-noise offsets' variance,
 * short enough that a preset change settles in a sixth of a second.
 */
const TEMPORAL_BLEND = 0.1;
/** Width of the neighbourhood clamp, in standard deviations. */
const TEMPORAL_GAMMA = 1.25;
/** Golden-ratio step per frame on the blue-noise offset (Peters, "The problem with 3D blue noise", 2017). */
const GOLDEN = 0.6180339887498949;

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

/**
 * Softness of the cloud edge in noise units. Crisper = puffier cumulus.
 * 0.02, from 0.035: the base field changes by ~0.16 over a kilometre or so,
 * so 0.035 was a ramp of about 200 m on every edge — a visible soft fringe on
 * an overhead cloud two kilometres up.
 */
const EDGE_WIDTH = 0.02;

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
 * night sky, enough to silver an edge without whitening the deck. 1.6 since
 * the night dome was lifted from near black to teal-navy (`Atmosphere`'s
 * `NIGHT_HORIZON`): at 3.5 the brighter sky level made the moonlit deck read
 * pale blue-white instead of dark teal with silvered rims.
 */
const MOON_GAIN = 1.6;

/** Ambient gain on the base's shadow colour at a clear noon. */
const AMBIENT_GAIN = 0.34;

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
/**
 * The high sheet's shape is web-ocean's altostratus deck (`highEdge` in
 * github.com/Noniv/web-ocean `js/wgsl/sky.js`, MIT): rotated fBm stretched
 * only 0.09 : 0.20 per km (about 2:1), octaves faded by the pixel footprint.
 *
 * It replaced a 10:1 streak field (7 km along, 700 m across the wind) that the
 * head-to-head read as "radial streak artefacts" across the blue gaps: a field
 * of parallel filaments at 7.8 km seen in perspective converges on the wind's
 * vanishing point, so it drew ruled lines fanning across the sky. Measured on
 * `sky-day-b`: the lines vanish with `cirrus: 0` and are untouched by the
 * volumetric fog (`steps: 0`) — so they were never the fog's crepuscular
 * shafts the AAA plan suspected.
 */
const CIRRUS_SCALE_ALONG = 0.09 / 1000;
const CIRRUS_SCALE_ACROSS = 0.2 / 1000;
/** web-ocean `cloudFbm` octave count for the high layer. Cost-neutral against the old streak field (A/B within noise). */
const CIRRUS_OCTAVES = 6;
/** Peak opacity of the cirrus sheet. Ice cloud is thin; it veils, never hides. */
const CIRRUS_OPACITY = 0.38;
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
  /** See `VIEW_EXTINCTION_GAIN`. A uniform so a probe can vary it. */
  private readonly uViewGain = uniform(VIEW_EXTINCTION_GAIN);
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
   * *continuous* deck fades toward. See `buildCompositeNode`.
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
  /** How much the weather map varies the cloud tops; tapers like the amount. */
  private readonly uTypeAmount = uniform(0);

  /**
   * Frame index for the march offset. A CPU counter advanced once per march,
   * not three's `frameId`, so `resetWind` rewinds it and a deterministic
   * capture sees the same sequence every time.
   */
  private readonly uFrame = uniform(0);
  private frame = 0;
  /** Edge band-limit per metre of march step. See its use in `buildMarchNode`. */
  private readonly uSoftnessPerMetre = uniform(0.0008);

  // --- the march's own passes ----------------------------------------------
  /** Eye position the march starts from; written by `setView`. */
  private readonly uRayOrigin: any = uniform(new THREE.Vector3());
  /** This frame's unjittered view-projection; projects a direction to the targets' UV. */
  private readonly uViewProj: any = uniform(new THREE.Matrix4());
  /** The view-projection the history was marched with. */
  private readonly uPrevViewProj: any = uniform(new THREE.Matrix4());
  /** Inverse of the rotation-only view-projection: target UV to direction. */
  private readonly uInvViewRot: any = uniform(new THREE.Matrix4());
  /** 1 / target size, for the neighbourhood taps. */
  private readonly uTexel: any = uniform(new THREE.Vector2(1, 1));
  /** Weight of this frame's march; 1 discards the history. */
  private readonly uBlend = uniform(1);
  private readonly prevViewProj = new THREE.Matrix4();
  private readonly viewProj = new THREE.Matrix4();
  private historyValid = false;
  /** Target resolution as a fraction of the scene pass's. */
  private resolutionScale = 1;
  private readonly marchTarget: THREE.RenderTarget;
  /** Ping-pong pair: one is this frame's resolve, the other last frame's. */
  private readonly resolveTargets: [THREE.RenderTarget, THREE.RenderTarget];
  private ping = 0;
  private readonly quad = new THREE.QuadMesh();
  private readonly marchMaterial = new THREE.NodeMaterial();
  private readonly resolveMaterial = new THREE.NodeMaterial();
  /** Texture nodes whose `value` follows the ping-pong. */
  private readonly historyNode: any;
  private readonly compositeNode: any;
  private rendererState: any = undefined;

  /** Shared haze, so the deck fades into the same air the island does. */
  private aerial: AerialPerspective | null = null;

  constructor() {
    this.params = {
      ...DEFAULT_CLOUD_PARAMS,
      color: DEFAULT_CLOUD_PARAMS.color.clone(),
      shadowColor: DEFAULT_CLOUD_PARAMS.shadowColor.clone(),
    };

    this.geometry = new THREE.SphereGeometry(1, 40, 24);

    const targetOptions = {
      type: THREE.HalfFloatType,
      depthBuffer: false,
      magFilter: THREE.LinearFilter,
      minFilter: THREE.LinearFilter,
      generateMipmaps: false,
    };
    this.marchTarget = new THREE.RenderTarget(1, 1, targetOptions);
    this.marchTarget.texture.name = 'clouds.march';
    this.resolveTargets = [
      new THREE.RenderTarget(1, 1, targetOptions),
      new THREE.RenderTarget(1, 1, targetOptions),
    ];
    this.resolveTargets[0].texture.name = 'clouds.resolveA';
    this.resolveTargets[1].texture.name = 'clouds.resolveB';
    this.historyNode = texture(this.resolveTargets[1].texture);
    this.compositeNode = texture(this.resolveTargets[0].texture);
    this.marchMaterial.name = 'clouds.march';
    this.resolveMaterial.name = 'clouds.resolve';
    this.marchMaterial.fragmentNode = this.buildMarchNode();
    this.resolveMaterial.fragmentNode = this.buildResolveNode();

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

    this.material.colorNode = this.buildCompositeNode();
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
   * take effect — `buildCompositeNode` runs in the constructor — so this is a
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
   * Both matrices are the unjittered ones the ocean uses (`current`, captured
   * outside the pipeline render). The camera's own projection is jittered for
   * the length of the scene pass, which put up to half a pixel of Halton offset
   * into every cloud pixel's vector and softened the deck's edges.
   *
   * Safe as a material MRT override only because the layer is on its own
   * render layer and excluded from the planar mirror: a material `mrtNode`
   * *replaces* the output of a render into a target without MRT, which is what
   * the reflector's target is.
   */
  setPreviousViewProjection(previous: any, current: any): void {
    this.material.mrtNode = mrt({
      velocity: Fn(() => {
        const dir = vec4(normalize(positionGeometry), 0);
        const now = current.mul(dir);
        const before = previous.mul(dir);
        return now.xy.div(now.w).sub(before.xy.div(before.w));
      })(),
    });
    this.material.needsUpdate = true;
  }

  /**
   * The view the march is drawn from: this frame's **unjittered**
   * view-projection and the eye. Call once a frame before the scene renders.
   *
   * Unjittered on purpose. The march and its history live on a fixed grid, so a
   * still camera reprojects history texel onto history texel and the resolve
   * never resamples — no softening accumulates. The dome then looks the result
   * up through this same matrix from its own jittered fragment, which lands a
   * cloud pixel exactly where jittered geometry would be, and the scene's TRAA
   * sees it move with the jitter like everything else.
   */
  setView(viewProjection: THREE.Matrix4, eye: THREE.Vector3): void {
    this.viewProj.copy(viewProjection);
    (this.uViewProj.value as THREE.Matrix4).copy(viewProjection);
    (this.uRayOrigin.value as THREE.Vector3).copy(eye);
    // Rotation-only VP = VP * T(eye): directions (w = 0) never see the
    // translation, and the inverse never sees kilometre-scale coordinates.
    _rot.makeTranslation(eye.x, eye.y, eye.z);
    (this.uInvViewRot.value as THREE.Matrix4).multiplyMatrices(viewProjection, _rot).invert();
  }

  /**
   * Resolution of the march and its history, as a fraction of the scene pass.
   * 1 at High: the resolve converges a full-resolution march, which is what
   * keeps an eroded edge crisp. Lower tiers may trade it for steps.
   */
  setResolutionScale(scale: number): void {
    this.resolutionScale = Math.min(1, Math.max(0.25, scale));
  }

  /**
   * March, then resolve. Called from the dome's own node before the dome is
   * drawn in the scene pass, once per render of it — so it costs nothing when
   * the layer is hidden, and a capture that renders the scene draws a march.
   *
   * 1. The march, into `marchTarget`, at `uSteps` with a start offset of
   *    `fract(blueNoise(pixel) + frame * golden ratio)`: spatially blue, so one
   *    frame is fine isotropic grain with nothing for the eye to read as lines,
   *    and temporally a low-discrepancy sequence, so ten frames of it at one
   *    pixel are ten well-spread offsets rather than ten random ones.
   * 2. The resolve: last frame's result reprojected by view direction (the
   *    deck is at infinity for this purpose, so camera rotation reprojects
   *    exactly and translation not at all), clamped to the mean +-1.25 sigma
   *    of this frame's 3x3 neighbourhood (Salvi, "An excursion in temporal
   *    supersampling", GDC 2016), and blended at `TEMPORAL_BLEND`.
   *
   * Steady state is a ten-frame average of well-spread offsets — the 96-step
   * march's image at the cost of 18 steps. It is reached in the 90 settle
   * frames of a deterministic capture, and kept while the game is paused,
   * since the renderer keeps drawing and the offset keeps moving.
   */
  /** @internal Driven by `CloudPassNode`. */
  renderPasses(renderer: THREE.Renderer): void {
    const target = renderer.getRenderTarget();
    let width: number;
    let height: number;
    if (target !== null) {
      width = target.width;
      height = target.height;
    } else {
      renderer.getDrawingBufferSize(_size);
      width = _size.x;
      height = _size.y;
    }
    width = Math.max(1, Math.round(width * this.resolutionScale));
    height = Math.max(1, Math.round(height * this.resolutionScale));
    if (this.marchTarget.width !== width || this.marchTarget.height !== height) {
      this.marchTarget.setSize(width, height);
      this.resolveTargets[0].setSize(width, height);
      this.resolveTargets[1].setSize(width, height);
      this.historyValid = false;
    }
    (this.uTexel.value as THREE.Vector2).set(1 / width, 1 / height);

    this.uFrame.value = this.frame % 4096;
    this.frame++;
    this.uBlend.value = this.historyValid ? TEMPORAL_BLEND : 1;
    (this.uPrevViewProj.value as THREE.Matrix4).copy(this.historyValid ? this.prevViewProj : this.viewProj);

    const write = this.resolveTargets[this.ping];
    const read = this.resolveTargets[1 - this.ping];
    this.historyNode.value = read.texture;
    this.compositeNode.value = write.texture;

    this.rendererState = THREE.RendererUtils.resetRendererState(renderer, this.rendererState);
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setRenderTarget(this.marchTarget);
    this.quad.material = this.marchMaterial;
    this.quad.name = 'clouds.march';
    this.quad.render(renderer);
    renderer.setRenderTarget(write);
    this.quad.material = this.resolveMaterial;
    this.quad.name = 'clouds.resolve';
    this.quad.render(renderer);
    THREE.RendererUtils.restoreRendererState(renderer, this.rendererState);

    this.prevViewProj.copy(this.viewProj);
    this.ping = 1 - this.ping;
    this.historyValid = true;
  }

  /** Forget the history: the next frame is resolved from its own march alone. */
  resetHistory(): void {
    this.historyValid = false;
  }

  setAerialPerspective(aerial: AerialPerspective): void {
    if (this.aerial === aerial) return;
    this.aerial = aerial;
    this.material.colorNode = this.buildCompositeNode();
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
      // Most of the direct term goes under a continuous deck. The sun reaches a
      // storm ceiling's top, not the base the eye sees from under it, and the
      // light march's two-octave field cannot know that: wherever it found a
      // short path up through the deck it lit a patch of the base white, which
      // looking toward a sun behind the storm read as glowing holes.
      this.uSunGain.value = SUN_GAIN * above * (1 - 0.75 * this.uDeckHold.value);
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
    // And the march's own temporal state, which a capture must rewind with it:
    // the offset sequence restarts and the next frame resolves from itself.
    this.frame = 0;
    this.historyValid = false;
    this.windOffset.set(0, 0, 0);
    this.uWindOffset.value.copy(this.windOffset);
    this.evolution = 0;
    this.uEvolution.value.set(0, 0, 0);
  }

  update(dt: number): void {
    if (!this.mesh.visible) {
      // Nothing is marched while hidden, so what is in the history is from
      // whenever it was last shown. Start clean when it comes back.
      this.historyValid = false;
      return;
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
    this.marchMaterial.dispose();
    this.resolveMaterial.dispose();
    this.marchTarget.dispose();
    this.resolveTargets[0].dispose();
    this.resolveTargets[1].dispose();
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
    //
    // Squared, so it is near full strength across the broken range (0.83 at
    // 0.3 coverage, 0.92 at 0.6) and gone well before a deck: at the storm's
    // 0.95 the linear taper left enough to thin holes into the ceiling, and
    // those holes lit up white in the forward lobe of a sun behind the deck.
    const c = Math.min(1, Math.max(0, p.coverage));
    const broken = 4 * c * (1 - c);
    this.uWeatherAmount.value = WEATHER_AMOUNT * broken * broken;
    this.uTypeAmount.value = Math.min(1, broken * broken * 1.3);

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

      // Sampled once for the whole walk rather than per sample, at the walk's
      // middle — where the shade actually comes from, which under a low sun is
      // kilometres up-sun of the shaded point. The samples either side of it are
      // a fraction of the weather map's smallest (4 km) octave away.
      const midLevel = this.uAltitude.add(this.uThickness.mul(0.5));
      const mid = p.add(this.uSunDir.mul(midLevel.sub(p.y).mul(invSunY).max(0)));
      const weather = this.weatherAt(mid).toVar('cloudShadowWeather');

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
   * The weather map at a point, about 0.5 +- 0.2. See `weatherFbm`.
   */
  private weatherAt(p: any): any {
    const q = vec2(p.x.sub(this.uWindOffset.x), p.z.sub(this.uWindOffset.z)).mul(WEATHER_SCALE);
    return weatherFbm(q).mul(0.5).add(0.5);
  }

  /**
   * Top of the cloud column as a fraction of the layer, from the weather map:
   * flat small cells where the map is thin, the full thickness where it is
   * dense. See `TOP_MIN`.
   */
  private cloudTop(weather: any): any {
    const sparse = float(1).sub(smoothstep(0.38, 0.62, weather));
    return float(1).sub(sparse.mul(this.uTypeAmount).mul(1 - TOP_MIN));
  }

  /**
   * The vertical profile with no base lift — the largest it can be at this
   * height. Lift only raises the base, so this bounds the real profile from
   * above and the edge from below: what the march's empty-sample test needs.
   */
  private profileMax(h: any, top: any): any {
    return smoothstep(0, 0.12, h).mul(smoothstepDown(h, top.mul(0.42), top));
  }

  /**
   * The coverage threshold for a given profile: raised toward the top and
   * bottom of the column (that is what rounds the tops), and moved by the
   * weather map and any caller bias.
   */
  private edgeFor(profile: any, weather: any, bias: any): any {
    return this.uThreshold
      .add(float(1).sub(profile).mul(0.22))
      .add(weather.sub(0.5).mul(this.uWeatherAmount))
      .add(bias);
  }

  /** Base shape field, 0..1, at noise coordinate `q`. */
  private baseField(q: any, octaves: number): any {
    return cloudFbm(q, int(octaves)).mul(0.5).add(0.5);
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
   * @param bias Added to the coverage threshold; the march's far thinning.
   */
  private densityAt(
    p: any,
    softness: any,
    weather: any,
    baseOctaves: number | null = null,
    bias: any = float(0),
  ): any {
    const h = this.heightFraction(p);
    const q = p.sub(this.uWindOffset).mul(NOISE_SCALE);
    const base = this.baseField(q, baseOctaves ?? 4);
    return this.densityFrom(h, q, base, softness, weather, this.cloudTop(weather), bias, baseOctaves === null);
  }

  /**
   * Density from a base field already in hand — the march evaluates the base
   * first, to decide whether the rest is worth computing.
   */
  private densityFrom(
    h: any,
    q: any,
    base: any,
    softness: any,
    weather: any,
    top: any,
    bias: any,
    withDetail: boolean,
  ): any {
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
    const detail = withDetail
      ? cloudFbm(q.mul(DETAIL_FREQUENCY).add(vec3(7.3, 2.1, 5.7)).add(this.uEvolution), int(3)).mul(0.5)
      : null;

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
    // of as a sheet with soft edges. The top comes from the weather map's cloud
    // type, so a sparse region's cells stay low and flat.
    const profile = smoothstep(lift, lift.add(0.12), h).mul(smoothstepDown(h, top.mul(0.42), top));

    // The weather field moves the threshold, so coverage varies across the sky
    // instead of being one number for the whole hemisphere.
    const edge = this.edgeFor(profile, weather, bias);
    const width = float(EDGE_WIDTH).add(softness.mul(0.25));

    // The erosion octave fades out as the march coarsens.
    //
    // `softness` already widened the edge with step size, which softens the
    // silhouette but does nothing about the detail field itself — and that field
    // runs at 4.3x the base frequency, so its features are a few hundred metres
    // across. Detail the march cannot resolve is not detail, it is noise, so it
    // is faded rather than sampled. Same reasoning as the wave cascades'
    // geometry fade. With the temporal resolve integrating ten offsets per pixel
    // the march resolves a tenth of its step, which is why the band-limit per
    // metre is a third of what a single march needed (`uSoftnessPerMetre`).
    const detailFade = float(1).sub(softness).clamp(0, 1);
    // Billow noise, |n|: rounded lumps around the noise's extrema, with creases
    // along its zero crossings — Perlin's "turbulence", the cauliflower shape.
    //
    // What stood here was the opposite, ridged noise, `0.13 - |n|`: a network of
    // thin raised lines along the zero crossings. On a cloud edge seen from the
    // side that read as fraying; on the overhead deck in `clouds-1`, seen from
    // below at two kilometres, every line was a tendril and the bases read as
    // smoke ("melted"). See `EROSION_OFFSET` for how the coverage was held.
    const erosion = detail === null
      ? null
      : detail.abs().min(EROSION_CAP).sub(EROSION_OFFSET).mul(detailFade).mul(DETAIL_GAIN);
    const field = erosion === null ? base : base.add(erosion);
    const shaped = smoothstep(edge, edge.add(width), field);

    return shaped.mul(profile).mul(this.uDensity);
  }

  /** World direction through a march/resolve target texel, from its UV. */
  private rayFromUv(uvNode: any): any {
    // Target UV is y-down, like every pass texture: NDC y = 1 - 2v.
    const ndc = vec4(uvNode.x.mul(2).sub(1), float(1).sub(uvNode.y.mul(2)), 0.5, 1);
    const p = this.uInvViewRot.mul(ndc);
    return normalize(p.xyz.div(p.w));
  }

  /** Target UV a direction lands on under `viewProj`, and that clip w. */
  private uvFromRay(viewProj: any, dir: any): { uv: any; w: any } {
    const clip = viewProj.mul(vec4(dir, 0)).toVar();
    const w = clip.w.max(1e-6);
    return {
      uv: vec2(clip.x.div(w).mul(0.5).add(0.5), clip.y.div(w).mul(-0.5).add(0.5)),
      w: clip.w,
    };
  }

  /**
   * The march, on a full-screen quad into `marchTarget`: premultiplied
   * radiance and coverage, `(scattered, 1 - transmittance)`, with no haze —
   * the haze needs the scene camera, and the dome applies it.
   */
  private buildMarchNode(): any {
    return Fn(() => {
      const view = this.rayFromUv(uv()).toVar('cloudView');
      const ro = vec3(this.uRayOrigin).toVar('cloudRo');
      const result = vec4(0, 0, 0, 0).toVar('cloudResult');

      // Nothing to march under the horizon: the dome fades the layer out a third
      // of a degree below it and the sea draws over everything beneath. The old
      // dome marched every one of those pixels as a horizon ray anyway — half
      // the cost of a frame pitched down at the sea, for pixels nobody sees.
      If(view.y.greaterThan(-0.012), () => {
        // Rays at or below the horizon are marched as horizon rays. On a sphere
        // they would otherwise find the shell on the far side, through the
        // planet; and what they should look like *is* the horizon — the layer
        // converged into the haze — so the floor produces the right image as
        // well as a safe one.
        const rd = normalize(vec3(view.x, max(view.y, GRAZE_FLOOR), view.z)).toVar('cloudRd');

        const tEnter = this.shellDistance(ro, rd, this.uAltitude).max(0).toVar('tEnter');
        const tExit = min(
          this.shellDistance(ro, rd, this.uAltitude.add(this.uThickness)).max(0),
          tEnter.add(this.uThickness.mul(MAX_SPAN_FACTOR)),
        ).toVar('tExit');

        If(tExit.greaterThan(tEnter), () => {
          const stepSize = tExit.sub(tEnter).mul(this.uInvSteps).toVar('cloudStep');
          // The start offset: blue noise in space, golden ratio in time. See
          // `renderPasses` and `blueNoise.ts`. It replaced interleaved gradient
          // noise shifted 5.588 px a frame, whose diagonal structure was the
          // hatching on every cloud edge in the head-to-head, and whose sixteen
          // offsets per pixel clustered into five or six distinct values.
          const noise = texture(blueNoiseTexture(), screenCoordinate.xy.div(BLUE_NOISE_SIZE)).r;
          const jitter = fract(noise.add(0.5 / 256).add(this.uFrame.mul(GOLDEN))).toVar('cloudJitter');
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

          // The edge band-limit, per metre of step. 0.0011, a third of what a
          // single static march needed (0.0032): the band-limit exists because
          // a fixed offset aliased the detail octave into moire, and a resolve
          // averaging ten well-spread offsets resolves a tenth of a step. At the
          // old setting every mid-distance cumulus was blurred into a soft ball
          // — the "airbrushed" clouds in the head-to-head. 0.0008 since round
          // 2: the billow erosion has rounder, lower-frequency edges than the
          // ridged lines it replaced, and holds at a finer band-limit.
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

          // Far thinning; see `THIN_NEAR`. Per ray, from where it enters.
          const thin = smoothstep(THIN_NEAR, THIN_FAR, tEnter)
            .mul(THIN_AMOUNT)
            .mul(float(1).sub(this.uDeckHold))
            .toVar('cloudThin');

          Loop(this.uSteps, () => {
            const p = ro.add(rd.mul(t)).toVar('cloudP');
            // One weather sample per step, shared with this step's light march.
            const weather = this.weatherAt(p).toVar('cloudWeather');
            const h = this.heightFraction(p).toVar('cloudHf');
            const top = this.cloudTop(weather).toVar('cloudTopF');
            const q = p.sub(this.uWindOffset).mul(NOISE_SCALE).toVar('cloudQ');
            const base = this.baseField(q, 4).toVar('cloudBase');

            // The base field first, and the rest only if it can matter. The
            // erosion octave can raise the field by at most EROSION_LIFT_MAX and
            // the base lift can only raise the edge, so a base this far under
            // the lowest edge the column allows is empty whatever else is
            // computed. Most of a broken sky is that, and it skips the detail
            // noise and the light march.
            const edgeMin = this.edgeFor(this.profileMax(h, top), weather, thin);
            If(base.greaterThan(edgeMin.sub(EROSION_LIFT_MAX)), () => {
              const d = this.densityFrom(h, q, base, softness, weather, top, thin, true).toVar('cloudD');

              If(d.greaterThan(0.002), () => {
                // --- optical depth toward the sun -----------------------------
                const lightAcc = float(0).toVar('cloudLightAcc');
                // The coarse field: two octaves and no erosion, against seven
                // for the layer itself. What it costs is the fine structure of
                // a cloud's *self-shadowing*, and that is the one place the
                // loss does not read: this accumulates optical depth toward the
                // sun and is then fed through `exp`, so it is an integral of the
                // field rather than a sample of it.
                Loop(LIGHT_STEPS, ({ i }: any) => {
                  const lp = p.add(this.uSunDir.mul(this.uLightStep.mul(float(i).add(1.0))));
                  lightAcc.addAssign(this.densityAt(lp, softness, weather, LIGHT_OCTAVES, thin));
                });
                // Plus one sample a billow's radius toward the sun at the full
                // field, erosion included (`NEAR_LIGHT_DISTANCE`).
                const nearAcc = this.densityAt(
                  p.add(this.uSunDir.mul(NEAR_LIGHT_DISTANCE)),
                  softness,
                  weather,
                  null,
                  thin,
                ).mul(NEAR_LIGHT_DISTANCE * NEAR_LIGHT_WEIGHT);
                const opticalDepth = lightAcc
                  .mul(this.uLightStep)
                  .add(nearAcc)
                  .mul(this.uExtinction)
                  .mul(LIGHT_EXTINCTION_GAIN)
                  .toVar('cloudOd');

                // Powder term — darkens the optically thin parts of the cloud
                // facing the viewer, which reads as internal structure. First
                // order only: it is a near-surface effect, and the deeper orders
                // are precisely the light that has stopped caring where the
                // surface was.
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
                // The base of a cumulus sees the sea and its own underside; the
                // top sees the whole sky. Grading between the two is the single
                // substitution that makes a puff read as a volume. Graded over
                // the column's own height (its type's top), not the layer's, so
                // a low flat cell has a lit top too; and on a curve, so the dark
                // holds through the lower third where a cumulus base is.
                const height = h.div(top).clamp(0, 1).toVar('cloudH');
                const ambient = mix(
                  this.uAmbientBottom,
                  this.uAmbientTop,
                  height.mul(height).mul(float(3).sub(height.mul(2))),
                ).toVar('cloudAmbient');

                // The low sun under the deck. Strongest at the very base and gone
                // by half way up; and attenuated by how much cloud lies between
                // the sample and the sun, which at a grazing sun is a horizontal
                // path — so the glow catches the exposed undersides and flanks
                // rather than the interior of a lobe.
                const below = float(1).sub(h.clamp(0, 1));
                const underGlow = this.uUnderGlow
                  .mul(below.mul(below))
                  .mul(glowView)
                  .mul(exp(opticalDepth.mul(-0.3)).mul(0.65).add(0.35));

                const lum = ambient
                  .add(this.uColor.mul(this.uLightColor).mul(sunTerm))
                  .add(underGlow)
                  .toVar('cloudLum');

                const sampleT = exp(d.mul(stepSize).mul(this.uExtinction).mul(this.uViewGain).negate());
                // Energy-conserving analytic integration over the step.
                scattered.addAssign(lum.mul(float(1).sub(sampleT)).mul(transmittance));
                transmittance.mulAssign(sampleT);
              });
            });

            t.addAssign(stepSize);

            If(transmittance.lessThan(0.01), () => {
              Break();
            });
          });

          result.assign(vec4(scattered, float(1).sub(transmittance)));
        });
      });

      return result;
    })();
  }

  /**
   * The temporal resolve; see `renderPasses`. Premultiplied in, premultiplied
   * out, so the blend and the clamp are linear in what the dome composites.
   */
  private buildResolveNode(): any {
    const march = this.marchTarget.texture;
    return Fn(() => {
      const uvN = uv().toVar('crUv');
      const current = texture(march, uvN).toVar('crCur');

      // Mean and spread of this frame's 3x3, for the clamp.
      const m1 = current.toVar('crM1');
      const m2 = current.mul(current).toVar('crM2');
      for (const [x, y] of [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]) {
        const s = texture(march, uvN.add(vec2(x, y).mul(this.uTexel)));
        m1.addAssign(s);
        m2.addAssign(s.mul(s));
      }
      const mean = m1.div(9).toVar('crMean');
      const sigma = m2.div(9).sub(mean.mul(mean)).max(0).sqrt().mul(TEMPORAL_GAMMA).toVar('crSigma');

      // Where this texel's direction was last frame. The deck is at infinity
      // for this purpose: w = 0 drops the translation, and the camera's
      // rotation moves the history exactly as far as it moves the sky.
      const dir = this.rayFromUv(uvN);
      const prev = this.uvFromRay(this.uPrevViewProj, dir);
      const inside = prev.uv.x
        .greaterThanEqual(0)
        .and(prev.uv.x.lessThanEqual(1))
        .and(prev.uv.y.greaterThanEqual(0))
        .and(prev.uv.y.lessThanEqual(1))
        .and(prev.w.greaterThan(0));
      const history = this.historyNode.sample(prev.uv);
      const clamped = clamp(history, mean.sub(sigma), mean.add(sigma));
      const weight = inside.select(this.uBlend, float(1));
      return mix(clamped, current, weight);
    })();
  }

  /**
   * The dome: looks the resolved deck up, dissolves it into the haze, lays the
   * cirrus behind it, and fades it out under the horizon. Drawn in the scene
   * pass, so `cameraPosition` here is the real eye.
   */
  private buildCompositeNode(): any {
    return Fn(() => {
      const view = normalize(positionGeometry).toVar('cloudView');
      const ro = cameraPosition.toVar('cloudRo');
      const rd = normalize(vec3(view.x, max(view.y, GRAZE_FLOOR), view.z)).toVar('cloudRd');

      // Looked up through the *unjittered* projection from this fragment's
      // jittered direction — see `setView`.
      const look = this.uvFromRay(this.uViewProj, view);
      const deck = nodeObject(new CloudPassNode(this, this.compositeNode.sample(look.uv)) as any).toVar('cloudDeck');

      const result = vec4(0, 0, 0, 0).toVar('cloudResult');
      If(deck.w.greaterThan(1e-4), () => {
        const tEnter = this.shellDistance(ro, rd, this.uAltitude).max(0).toVar('tEnter');
        // The deck dissolves into the same haze the sea and the island dissolve
        // into, rather than into an exponential of its own. The layer has fog
        // disabled and is drawn on a camera-locked dome, so `scene.fogNode`
        // never reaches it and the shared function is called by hand.
        //
        // The march's output is premultiplied — radiance already weighted by
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
        const straight: any = deck.xyz.div(deck.w.max(1e-4));
        // Plus the deck's own far fade (`FAR_HAZE_START`).
        const farFade = exp(tEnter.sub(FAR_HAZE_START).max(0).div(-FAR_HAZE_SCALE));
        const air: any = (this.aerial === null ? float(1) : this.aerial.transmittance(tEnter, rd)).mul(farFade);
        const hold: any = this.uDeckHold;
        const hazed: any = mix(this.uHaze, straight, air);
        const lit: any = mix(straight, hazed, hold);
        const coverage: any = deck.w.clamp(0, 1)
          .mul(mix(air, float(1), hold))
          .toVar('cloudCoverage');
        result.assign(vec4(lit.mul(coverage), coverage));
      });

      // The cirrus sheet, behind the cumulus. Premultiplied "over": it shows
      // only through what the deck in front of it lets through, and costs
      // nothing where that is nothing. Drawn here at full resolution rather than
      // in the march: it is a single lookup with its own footprint filtering,
      // and needs neither the steps nor the history.
      If(
        this.uCirrusAmount.greaterThan(0.001).and(result.w.lessThan(0.995)).and(view.y.greaterThan(0.012)),
        () => {
          const cirrus = this.cirrusAt(ro, rd, view).toVar('cirrus');
          const behind = float(1).sub(result.w);
          result.assign(
            vec4(result.xyz.add(cirrus.xyz.mul(behind)), result.w.add(cirrus.w.mul(behind))),
          );
        },
      );

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

    // web-ocean `highEdge` / `cloudFbm`, transcribed. Their noise is a PCG-hashed
    // gradient noise; MaterialX Perlin is the same family and is what this file
    // already uses. `fp` is the pixel footprint in noise units — the slant
    // distance to the sheet times the angle of one pixel, over the sine of the
    // elevation — and each octave fades out before it reaches pixel size,
    // which is what keeps the sheet from shimmering or ruling lines in motion.
    const pixAngle = float(2).div((cameraProjectionMatrix as any).element(1).y.mul(screenSize.y));
    const fp = t
      .mul(pixAngle)
      .div(max(rd.y, 0.03))
      .mul(CIRRUS_SCALE_ACROSS)
      .toVar('cirrusFp');
    const ax = along.mul(CIRRUS_SCALE_ALONG);
    const cy = across.mul(CIRRUS_SCALE_ACROSS);
    // CLOUD_ROT = mat2x2(0.8, 0.6, -0.6, 0.8), column-major.
    const qx = ax.mul(0.8).sub(cy.mul(0.6)).toVar('cirrusQx');
    const qy = ax.mul(0.6).add(cy.mul(0.8)).toVar('cirrusQy');
    const fbm = float(0).toVar('cirrusFbm');
    let amp = 0.5;
    let freq = 1;
    for (let o = 0; o < CIRRUS_OCTAVES; o++) {
      const w = float(1).sub(smoothstep(0.08, 0.28, fp.mul(freq)));
      fbm.addAssign(mx_noise_float(vec3(qx, qy, this.uEvolution.y.mul(0.35).add(o * 3.1))).mul(amp).mul(w));
      // q = ROT * q * 2.03 + (13.1, 7.7)
      const nx = qx.mul(0.8).sub(qy.mul(0.6)).mul(2.03).add(13.1);
      const ny = qx.mul(0.6).add(qy.mul(0.8)).mul(2.03).add(7.7);
      qx.assign(nx);
      qy.assign(ny);
      freq *= 2.03;
      amp *= 0.5;
    }
    const big = mx_noise_float(vec3(along.mul(0.012 / 1000), across.mul(0.012 / 1000), 7.0)).mul(0.6);
    // web-ocean's veil coverage is 0.36 by default; `cirrus` 0.45 (skyPro) maps near it.
    const edge = fbm.mul(1.25).add(0.5).add(big).sub(float(1).sub(this.uCirrusAmount.mul(0.8))).toVar('cirrusEdge');
    const soft = float(0.85).add(min(fp.mul(0.2), 1).mul(0.3));
    // Their analytic silhouette anti-aliasing: opacity ramps over ~1.5 px of edge.
    const aa = smoothstep(0, fp.mul(0.2 * 2.5 * 1.5).add(1e-6), edge);

    // Thinned toward the horizon, where the sheet is foreshortened below the
    // sampling rate of a screen pixel.
    const density = clamp(edge.div(soft), 0, 1)
      .mul(aa)
      .mul(mask)
      .mul(smoothstep(0.012, 0.09, view.y));
    const alpha = density.mul(CIRRUS_OPACITY).toVar('cirrusAlpha');

    // Thin ice is strongly forward-scattering: a brighter veil around the sun,
    // plain white away from it.
    //
    // Gentler than it was (a 2.2 peak at the power of ten, and 0.35 on the
    // square). The head-to-head's "diagonal smear streak beside the sun" in
    // sky-1 and "diagonal light streak" in sky-3 were this sheet: both vanish
    // with `cirrus: 0` and are unchanged with the lens flare off (captured on
    // aaa-pass, `clouds/probe-nocirrus` and `probe-noflare`). A lone patch of
    // veil 25 degrees from the sun was being lit at twice the brightness of the
    // rest of it, over a sky already at its brightest there, and read as a
    // glare artefact rather than as cloud.
    const cosSun = dot(rd, this.uSunDir).max(0);
    const forward = pow(cosSun, 12).mul(0.8).add(cosSun.mul(cosSun).mul(0.2));
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
const _rot = /*@__PURE__*/ new THREE.Matrix4();
const _size = /*@__PURE__*/ new THREE.Vector2();

/**
 * Runs the march and the resolve (`Clouds.renderPasses`) before the dome that
 * reads them is drawn, once per render of it, and stands for the resolved
 * sample in the dome's graph. The same arrangement as three's reflector: a
 * node whose `updateBefore` renders into a target the material then samples.
 */
class CloudPassNode extends THREE.TempNode {
  static get type(): string {
    return 'CloudPassNode';
  }

  constructor(
    private readonly owner: Clouds,
    private readonly sampleNode: any,
  ) {
    super('vec4');
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
  }

  setup(): any {
    return this.sampleNode;
  }

  updateBefore(frame: any): boolean | undefined {
    this.owner.renderPasses(frame.renderer);
    return undefined;
  }
}

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
