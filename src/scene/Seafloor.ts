import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  cameraPosition,
  cameraViewMatrix,
  clamp,
  float,
  min,
  mix,
  normalWorldGeometry,
  normalize,
  positionWorld,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
// Aliased because this module needs a CPU twin of the same ramp under the
// unqualified name; see `smoothstepDown` below.
import { smoothstepDown as smoothstepDownNode } from '../core/tslMath';
import { occludeLight } from '../core/lightOcclusion';
// Shore hook: the swash wetting the sand. Self-contained; see `ocean/Surf`.
import { applySwashToGround, swashRoughness } from '../ocean/Surf';
import { SEEDS, mulberry32 } from '../core/random';

/**
 * Sandy seafloor.
 *
 * The heightfield is the one piece of geometry in this project that must exist
 * simultaneously on the CPU and the GPU: buoyancy and camera collision query it
 * per frame, the water surface reads it to decide how much of the shallow
 * turquoise to let through, and the floor mesh itself is displaced by it. If
 * those three disagree, the ship's shadow lands on water that is a different
 * colour from the sand underneath it.
 *
 * So the noise is deliberately *not* an analytic hash. `sin(dot(p, k)) * 43758`
 * hashes are chaotic by construction: a float32 GPU and a float64 CPU evaluate
 * them to completely different values, and the two representations of the floor
 * drift apart. Instead a single 256² byte texture of random values is generated
 * once and sampled with hand-written bilinear interpolation on both sides.
 * Unsigned-byte texels decode to exactly `n / 255` on every backend, so the two
 * evaluations agree to float32 rounding.
 *
 * Layout the field produces:
 *   - a shallow plateau (~17 m) around the origin, where the play area sits and
 *     the seafloor is meant to read through the water;
 *   - a shelf break falling to ~88 m in open water;
 *   - a second rise around the island, which breaks the surface.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

// ---------------------------------------------------------------- noise source

const NOISE_SIZE = 256;
const NOISE_MASK = NOISE_SIZE - 1;

const NOISE_BYTES = (() => {
  // Deterministic PRNG — the floor must be identical on every run and machine.
  const random = mulberry32(SEEDS.seafloorNoise);
  const bytes = new Uint8Array(NOISE_SIZE * NOISE_SIZE * 4);
  for (let i = 0; i < bytes.length; i++) bytes[i] = (random() * 256) | 0;
  return bytes;
})();

function createNoiseTexture(): THREE.DataTexture {
  const map = new THREE.DataTexture(
    NOISE_BYTES,
    NOISE_SIZE,
    NOISE_SIZE,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  // Nearest + repeat means uv = (i + 0.5) / N lands exactly on texel i for any
  // integer i, positive or negative — which is what makes the hand-rolled
  // bilinear filter below reproducible against the CPU path.
  map.minFilter = THREE.NearestFilter;
  map.magFilter = THREE.NearestFilter;
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.generateMipmaps = false;
  map.colorSpace = THREE.NoColorSpace;
  map.needsUpdate = true;
  return map;
}

function noiseTexel(ix: number, iy: number): number {
  const x = ix & NOISE_MASK;
  const y = iy & NOISE_MASK;
  return NOISE_BYTES[(y * NOISE_SIZE + x) * 4] / 255;
}

function valueNoise(px: number, py: number): number {
  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const fx = px - ix;
  const fy = py - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  const a = noiseTexel(ix, iy);
  const b = noiseTexel(ix + 1, iy);
  const c = noiseTexel(ix, iy + 1);
  const d = noiseTexel(ix + 1, iy + 1);

  return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy;
}

// FBM parameters. Shared verbatim by the CPU and TSL implementations; the
// rotation between octaves is what stops the sum from looking grid-aligned.
const OCTAVES = 4;
const LACUNARITY = 2.13;
const GAIN = 0.5;
const ROT = [0.8, 0.6, -0.6, 0.8] as const;
const OCTAVE_OFFSET = [17.3, 9.1] as const;
const FBM_NORM = (() => {
  let sum = 0;
  for (let o = 0, a = 1; o < OCTAVES; o++, a *= GAIN) sum += a;
  return sum;
})();

function fbm(x: number, y: number): number {
  let qx = x;
  let qy = y;
  let amplitude = 1;
  let sum = 0;
  for (let o = 0; o < OCTAVES; o++) {
    sum += valueNoise(qx, qy) * amplitude;
    const rx = qx * ROT[0] + qy * ROT[1];
    const ry = qx * ROT[2] + qy * ROT[3];
    qx = rx * LACUNARITY + OCTAVE_OFFSET[0];
    qy = ry * LACUNARITY + OCTAVE_OFFSET[1];
    amplitude *= GAIN;
  }
  return sum / FBM_NORM;
}

// ------------------------------------------------- analytic octave derivatives

/**
 * Highest octave index any node in this module evaluates.
 *
 * The heightfield itself stops at `OCTAVES` and **that is not changing** — the
 * mesh, the buoyancy solver, the prop placement and the water's depth term all
 * read the same four octaves they always did. What the octaves past it are for is
 * *shading*: a bump field the mesh has no way to carry, and a cavity term derived
 * from the difference between a detailed and a smoothed evaluation of the same
 * field. Both are things a viewer sees and nothing in the world queries.
 */
const MAX_GRAD_OCTAVES = 9;

/**
 * `dq_o / dp` for each octave, precomputed.
 *
 * The fbm chain is `q_{o+1} = L * R * q_o + c`, so the Jacobian of octave `o`'s
 * sample point with respect to the original point is `(L * R)^o` — a compile-time
 * constant, because both the lacunarity and the inter-octave rotation are. Which
 * means the exact gradient of the sum costs a handful of multiplies and **no
 * extra texture fetches at all**: `noised` already reads the four texels the
 * bilinear value needs, and the derivative of that bilinear patch falls out of
 * the same four numbers.
 *
 * That is the whole reason this is affordable per pixel. Finite-differencing the
 * heightfield instead would have meant four more full evaluations — sixteen more
 * fetches an octave — for a worse answer.
 *
 * Stored row-major as `[m00, m01, m10, m11]`, and applied transposed:
 * `grad_p = M^T * grad_q`.
 */
const OCTAVE_JACOBIAN: readonly (readonly number[])[] = (() => {
  const out: number[][] = [];
  let m = [1, 0, 0, 1];
  for (let o = 0; o < MAX_GRAD_OCTAVES; o++) {
    out.push(m.slice());
    const [r00, r01, r10, r11] = ROT;
    m = [
      LACUNARITY * (r00 * m[0] + r01 * m[2]),
      LACUNARITY * (r00 * m[1] + r01 * m[3]),
      LACUNARITY * (r10 * m[0] + r11 * m[2]),
      LACUNARITY * (r10 * m[1] + r11 * m[3]),
    ];
  }
  return out;
})();

/** Sum of `GAIN^o` over `[from, to)` — the normaliser for a partial octave run. */
function amplitudeSum(from: number, to: number): number {
  let sum = 0;
  for (let o = from; o < to; o++) sum += Math.pow(GAIN, o);
  return sum;
}

/** World metres per cell of octave `o` of the terrain field. */
function octaveFeatureMetres(o: number): number {
  return 1 / (FEATURE_SCALE * Math.pow(LACUNARITY, o));
}

// ------------------------------------------------------------- floor structure

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * CPU twin of `tslMath.smoothstepDown`, argument for argument.
 *
 * Every descending ramp in the heightfield now goes through this on the CPU and
 * through the node version on the GPU, so the two implementations can be read
 * against each other as the same three arguments in the same order. The shape
 * this replaced spelled the same ramp two different ways — `smoothstep(outer,
 * inner, x)` on the CPU against `smoothstepDown(x, inner, outer)` in TSL — and a
 * mirrored pair like that is precisely what survives a careless edit to one side
 * while looking correct in review.
 */
function smoothstepDown(x: number, inner: number, outer: number): number {
  return 1 - smoothstep(inner, outer, x);
}

/**
 * Directions on the island are unit vectors, never angles.
 *
 * Every directional term below is a dot product against one of these, which
 * keeps `atan2` out of the heightfield entirely. That matters twice: TSL has
 * moved its spelling of `atan2` across recent revisions, and the branch cut at
 * +/-pi would draw a seam straight across the island on whichever bearing it
 * happened to land. Harmonics of the bearing come back out of the dot product
 * through the Chebyshev identities (cos 2t = 2c^2 - 1, cos 3t = 4c^3 - 3c),
 * which are polynomials, and so agree between a float64 CPU and a float32 GPU
 * to the last rounding — the same reason the noise is a texture and not a hash.
 */
interface Dir {
  readonly x: number;
  readonly z: number;
}

/** A direction plus the angular reach of the feature that sits on it. */
interface Sector extends Dir {
  /** cos of the half-width: the mask is a smoothstep in cosine, not in angle. */
  readonly edge: number;
}

/** Bearing convention matches the scatter code in `Props`: x = cos, z = sin. */
function dir(bearing: number): Dir {
  return { x: Math.cos(bearing), z: Math.sin(bearing) };
}

function sector(bearing: number, halfWidth: number): Sector {
  return { x: Math.cos(bearing), z: Math.sin(bearing), edge: Math.cos(halfWidth) };
}

/** 1 on the sector's bearing, 0 past its half-width. `u` must be a unit bearing. */
function sectorMask(ux: number, uz: number, s: Sector): number {
  return smoothstep(s.edge, 1, ux * s.x + uz * s.z);
}

/** Rocky island the props dress and the distant silhouette comes from. */
export const ISLAND = {
  x: -1150,
  z: -780,
  /**
   * Mean shoreline radius. The coast itself runs from about 0.70x this at the
   * head of the bay to 1.25x at the tip of the headland, so this is the number
   * to scale placement by and not a number to trust as a coastline — ask
   * `seafloorHeight` where the water is.
   */
  radius: 500,
  /**
   * Height of the summit above mean sea level, before props and before relief.
   *
   * 150 m over a 1 km island, and the number is framing rather than geology.
   * At the 72 m it was, the island had a width-to-height ratio of 14:1 — which
   * is honest for a real Pacific island and *unreadable* at the distance anyone
   * sees this one from. Photographed from 900 m off the beach it was a green
   * bank on the horizon sixty pixels tall: no summit, no profile, nothing for
   * the eye to land on. The whole planting scheme was invisible at the range it
   * was authored for.
   *
   * At 150 the profile is about 7:1, the island is a shape against the sky, and
   * — because everything below plants in absolute metres and the summit rock
   * band is a *fraction* of this constant — the extra height arrives as bare
   * rock above the treeline rather than as more hillside. That silhouette
   * (green flanks, pale crown) is what a tropical high island actually reads
   * as, and it is what the reference frame in `docs/ref/` shows.
   */
  peak: 150,
} as const;

/**
 * Where the ground colour turns from sand to growth, metres above sea level.
 *
 * Not a hard line — the ramp is broken by the patch noise so the edge of the
 * growth is ragged rather than a contour. Low and tight on purpose: the ground
 * carries the grass and scrub that creep onto the back of a beach, and at the
 * 3.5-16 m this used to be, the gently shelving apron came out as a sixty-metre
 * strip of bare sand from the air.
 */
const GROWTH_START_METRES = 2.2;
const GROWTH_FULL_METRES = 7;

/**
 * How completely the growth covers the ground it reaches.
 *
 * Deliberately short of 1. Even closed canopy shows sand and rock through it
 * from above, and leaving a fraction of the substrate visible is what stops the
 * interior reading as painted felt.
 *
 * This is a weight on *albedo*, though, not a percentage of visible ground, and
 * the two are only the same number when the two albedos are close. They are
 * not: the substrate is four times the canopy's luminance, so at the 0.86 this
 * was, a seventh of the mix carried half the brightness and the vegetated
 * slopes came out khaki rather than green — the substrate was showing through
 * as *value* even where it was invisible as texture. 0.93 with a darker
 * `dryInland` gets the same broken-up read without bleaching the hue.
 */
const VEGETATION_COVER = 0.93;

const DEEP_Y = -88;
const PLATEAU_Y = -17;
const PLATEAU_RADIUS = 320;
const SHELF_RADIUS = 1250;
const RELIEF = 11;
/** World metres per noise cell of the coarsest octave. */
const FEATURE_SCALE = 1 / 240;

/**
 * Island shape.
 *
 * What this replaced was `peak * smoothstep(radius, radius * 0.18, dIsland)`:
 * one radially symmetric dome. That is why the shoreline was a circle, why the
 * cliff props could get away with assuming the shore ran tangent to one, and why
 * sailing around the island showed the same silhouette from every bearing. The
 * replacement is a sum of named terms, each doing one thing a reader can point
 * at. Deliberately not another noise call: noise with enough amplitude to move a
 * coastline this far is indistinguishable from static at this scale, and it
 * leaves nothing to aim a set piece at.
 *
 * Everything vertical is expressed against `t` — distance from the island centre
 * over the shoreline radius *on that bearing*. So `t = 1` is the waterline
 * everywhere by construction, and the bay, the headland and the lobes deform the
 * whole island rather than only its outline.
 */

// Shoreline radius, as a multiple of `ISLAND.radius`. Three harmonics of the
// bearing, each about its own axis, so the outline repeats on no obvious period.
/** cos 2t: the island is an ellipse before it is anything else. */
const ELONGATION = 0.12;
const ELONGATION_AXIS = dir(1.95);
/** cos 3t: three broad lobes, which is what stops it reading as an ellipse. */
const LOBES = 0.075;
const LOBE_AXIS = dir(2.6);
/** cos t: sand accretes on the downwind face, the upwind face is cut back. */
const DRIFT = 0.07;
const DRIFT_AXIS = dir(0.9);
/** Floor under the summed fraction; a term deep enough to invert it sends `t` to infinity. */
const SHORE_FLOOR = 0.35;

/**
 * The bay, cut into the windward shore — `Spectrum` blows toward pi/4 by
 * default, so this is the face that takes the swell.
 *
 * It cuts the shore *radius* rather than the height, which is what makes it a
 * bay a ship can enter instead of a dent in a hillside: the whole vertical
 * profile moves inward with the coast, so the bay gets its own beach at its head
 * and deepens toward its mouth for free.
 */
const BAY = sector(2.3, 0.44);
const BAY_CUT = 0.42;

/**
 * The headland: the opposite move. Radius pushed out, and a ridge raised along
 * the same bearing so the point ends in a bluff instead of tapering away to
 * nothing. It forms the far arm of the cove from the spit; between them they are
 * what encloses the lagoon.
 *
 * The ridge is windowed at both ends in `t`. The toe is not optional: a lift
 * that only faded outward is still at full height at the island centre, where it
 * stacks on the crest and puts the summit twenty-six metres above the value
 * `ISLAND.peak` promises.
 */
const HEADLAND = sector(1.45, 0.42);
const HEADLAND_REACH = 0.2;
const HEADLAND_LIFT = 26;
const HEADLAND_TOE = 0.42;
const HEADLAND_CROWN = 0.7;
const HEADLAND_BROW = 0.9;
const HEADLAND_FALL = 1.06;

/**
 * The apron carries the floor from shelf depth up to the waterline and owns the
 * beach gradient. `SHORE_LIFT` is solved, not tuned: it is exactly the lift that
 * puts sea level at `t = 1`, so widening the apron to soften the beach cannot
 * silently drag the coastline in or out.
 */
const APRON_IN = 0.78;
const APRON_OUT = 1.34;
const SHORE_LIFT = -PLATEAU_Y / smoothstepDown(1, APRON_IN, APRON_OUT);

/**
 * The summit sits inland of the island centre, upwind of it. Without that offset
 * every contour is a scaled copy of the shoreline and the island reads as a
 * shape stamped out of a cone however irregular its outline is.
 *
 * `CREST_SPAN` measures the crest against the *local* shore radius, so the crest
 * has died out before the beach on every bearing including the short one at the
 * head of the bay; a crest measured in metres would push land back into it.
 * `CREST_LIFT` is solved like `SHORE_LIFT`: plateau plus apron plus crest is
 * exactly `ISLAND.peak`, so that constant means what its name says.
 */
const CREST_IN = 0;
const CREST_OUT = 0.94;
const CREST_SPAN = 0.9;
const CREST_LIFT = ISLAND.peak - PLATEAU_Y - SHORE_LIFT;
const SUMMIT_DRIFT = 0.16;
const SUMMIT_OFFSET: Dir = {
  x: Math.cos(3.9) * ISLAND.radius * SUMMIT_DRIFT,
  z: Math.sin(3.9) * ISLAND.radius * SUMMIT_DRIFT,
};

/**
 * How far the island keeps its own shelf before the floor is allowed to fall to
 * `DEEP_Y`. The inner edge is past `t = 1` deliberately: the apron's arithmetic
 * assumes the floor under the beach is exactly `PLATEAU_Y`, and a skirt that had
 * already begun to fall there would pull the waterline in by a few metres, by an
 * amount that varied with bearing.
 */
const SKIRT_IN = 1.06;
const SKIRT_OUT = 1.95;

/**
 * The spit: a recurved bar running out from the shore beside the cove.
 *
 * Two masks off one axis. The wide one lifts the *shelf* under the bar; the
 * narrow one puts the bar on top of it. Without the shelf the bar would be a
 * wall standing off fifty metres of water, because the gap between the island's
 * skirt and the origin plateau is the deepest water anywhere near the island.
 *
 * `SPIT_CURVE` hooks the axis toward the lagoon as it runs — the shape longshore
 * drift actually builds, and the cheapest way to stop a straight extrusion from
 * looking like one.
 */
const SPIT = dir(-0.1);
const SPIT_CURVE = 0.095 / ISLAND.radius;
const SPIT_ROOT = 470;
const SPIT_RISE = 90;
const SPIT_TIP = 980;
const SPIT_TAPER = 260;
const SPIT_CORE = 22;
const SPIT_EDGE = 62;
const SPIT_SHOAL_CORE = 70;
const SPIT_SHOAL_EDGE = 180;
const SPIT_LIFT = 21;

/**
 * The lagoon: the water between the beach and the bar, on the lee shore.
 *
 * `Props` puts the pirate cove on this bearing, so the sector has to stay
 * navigable and gently shelving. Hence a *floor* rather than a barrier: applied
 * last and as a maximum, it can only ever raise the seabed toward `LAGOON_Y`.
 * The beach and the spit crest are already above it and pass through untouched,
 * which is also why the order of these last two terms is not free.
 */
const LAGOON = sector(0.68, 0.52);
const LAGOON_IN = 0.98;
const LAGOON_FULL = 1.16;
const LAGOON_EDGE = 1.4;
const LAGOON_OUT = 1.62;
const LAGOON_Y = -5.5;

/**
 * Relief that breaks the dome: radial ridges and gullies, two subsidiary peaks
 * and a stretch of sea cliff.
 *
 * Every term is windowed in `t` so it has died out before the beach, which is
 * what keeps the coastline — and with it the ship's shoal avoidance, the shore
 * break and the prop placement along the waterline — exactly where it was. The
 * cliff is the one term that reaches the shore, and it stops at `CLIFF_FOOT`,
 * short of `t = 1`, so it steepens the last few metres above the water without
 * moving the waterline.
 *
 * The ridges are drawn from the *bearing*, not from position: value noise
 * sampled on a small circle in noise space (`u * RIDGE_RADIUS`) varies smoothly
 * round the island with no seam and no `atan2`, and folding it (`1 - |2n - 1|`)
 * turns its crests into sharp spurs with rounded gullies between — the
 * radial drainage every volcanic high island has. A second sample in world
 * space bends them, so the spurs wander instead of radiating like spokes.
 */
const RIDGE_LIFT = 26;
const RIDGE_RADIUS = 1.45;
const RIDGE_OFFSET = [41.3, 7.9] as const;
const RIDGE_WANDER = 1 / 380;
const RIDGE_IN = 0.08;
const RIDGE_FULL = 0.4;
const RIDGE_FADE = 0.62;
const RIDGE_OUT = 0.9;

interface Peak {
  x: number;
  z: number;
  radius: number;
  lift: number;
}
/** A shoulder peak on the headland side and a lower knoll toward the spit. */
const PEAKS: readonly Peak[] = [
  {
    x: Math.cos(1.25) * ISLAND.radius * 0.46,
    z: Math.sin(1.25) * ISLAND.radius * 0.46,
    radius: ISLAND.radius * 0.42,
    lift: 46,
  },
  {
    x: Math.cos(-0.55) * ISLAND.radius * 0.5,
    z: Math.sin(-0.55) * ISLAND.radius * 0.5,
    radius: ISLAND.radius * 0.3,
    lift: 28,
  },
];
const PEAK_SHORE_IN = 0.7;
const PEAK_SHORE_OUT = 0.9;

/** Sea cliffs on the weather side, away from the cove, the fort and the bay. */
const CLIFF = sector(-1.05, 0.55);
const CLIFF_LIFT = 20;
const CLIFF_RISE_IN = 0.74;
const CLIFF_RISE_OUT = 0.9;
const CLIFF_BROW = 0.935;
const CLIFF_FOOT = 0.975;

/**
 * Shoreline radius on the bearing `(ux, uz)`, as a multiple of `ISLAND.radius`.
 */
function shoreFraction(ux: number, uz: number): number {
  const e = ux * ELONGATION_AXIS.x + uz * ELONGATION_AXIS.z;
  const l = ux * LOBE_AXIS.x + uz * LOBE_AXIS.z;
  const d = ux * DRIFT_AXIS.x + uz * DRIFT_AXIS.z;

  const elongation = ELONGATION * (e * e * 2 - 1);
  const lobes = LOBES * (l * l * l * 4 - l * 3);
  const drift = DRIFT * d;
  const bay = BAY_CUT * sectorMask(ux, uz, BAY);
  const headland = HEADLAND_REACH * sectorMask(ux, uz, HEADLAND);

  return Math.max(1 + elongation + lobes + drift - bay + headland, SHORE_FLOOR);
}

/**
 * Floor elevation in world metres (negative below sea level).
 *
 * Exported so `Props` can seat rocks and cliffs on the same surface the mesh is
 * built from, without either side owning the other. Called per frame from
 * buoyancy and per candidate from the placement loops, so it allocates nothing:
 * every direction it needs is a module constant read component-wise.
 */
export function seafloorHeight(x: number, z: number): number {
  const n = fbm(x * FEATURE_SCALE, z * FEATURE_SCALE);

  const rOrigin = Math.sqrt(x * x + z * z);

  const dx = x - ISLAND.x;
  const dz = z - ISLAND.z;
  const dIsland = Math.sqrt(dx * dx + dz * dz);
  // Guarded so the bearing is finite at the centre. Every directional term reads
  // it, and one NaN there would spread across the whole summit.
  const inv = 1 / Math.max(dIsland, 1);
  const ux = dx * inv;
  const uz = dz * inv;

  const shore = ISLAND.radius * shoreFraction(ux, uz);
  const t = dIsland / shore;

  // Crest measured from the offset summit, in its own normalised frame.
  const sx = dx - SUMMIT_OFFSET.x;
  const sz = dz - SUMMIT_OFFSET.z;
  const tCrest = Math.sqrt(sx * sx + sz * sz) / (shore * CREST_SPAN);

  // Spit, in along/across metres about its own axis; `across` is measured
  // against the hooked centreline rather than a straight one.
  const along = dx * SPIT.x + dz * SPIT.z;
  const across = dz * SPIT.x - dx * SPIT.z;
  const offset = Math.abs(across - along * along * SPIT_CURVE);
  const run =
    smoothstep(SPIT_ROOT, SPIT_ROOT + SPIT_RISE, along) *
    smoothstepDown(along, SPIT_TIP - SPIT_TAPER, SPIT_TIP);
  const shoal = run * smoothstepDown(offset, SPIT_SHOAL_CORE, SPIT_SHOAL_EDGE);
  const crest = run * smoothstepDown(offset, SPIT_CORE, SPIT_EDGE);

  const shallowOrigin = smoothstepDown(rOrigin, PLATEAU_RADIUS, SHELF_RADIUS);
  const shallowIsland = smoothstepDown(t, SKIRT_IN, SKIRT_OUT);
  const shallowness = Math.max(Math.max(shallowOrigin, shallowIsland), shoal);

  let y = DEEP_Y + (PLATEAU_Y - DEEP_Y) * shallowness;
  y += (n - 0.5) * RELIEF * (0.35 + 0.65 * shallowness);
  y += SHORE_LIFT * smoothstepDown(t, APRON_IN, APRON_OUT);
  y += CREST_LIFT * smoothstepDown(tCrest, CREST_IN, CREST_OUT);
  y +=
    HEADLAND_LIFT *
    sectorMask(ux, uz, HEADLAND) *
    smoothstep(HEADLAND_TOE, HEADLAND_CROWN, t) *
    smoothstepDown(t, HEADLAND_BROW, HEADLAND_FALL);
  y += SPIT_LIFT * crest;
  y += islandRelief(dx, dz, ux, uz, t);

  const lagoon =
    sectorMask(ux, uz, LAGOON) *
    smoothstep(LAGOON_IN, LAGOON_FULL, t) *
    smoothstepDown(t, LAGOON_EDGE, LAGOON_OUT);
  // Read `y` into `fill` before touching it. The TSL twin needs the same split
  // so the value the lagoon is filling against is unambiguously the pre-lagoon
  // floor rather than whatever a compound assignment decides to evaluate first.
  const fill = Math.max(LAGOON_Y - y, 0);
  y += lagoon * fill;
  return y;
}

/**
 * The dome-breaking relief, CPU side. `(dx, dz)` is the offset from the island
 * centre, `u` the unit bearing, `t` the shore-normalised radius. Mirrored node
 * for node by `islandReliefNode` in `buildNoiseNodes`.
 */
function islandRelief(dx: number, dz: number, ux: number, uz: number, t: number): number {
  const g = valueNoise(
    ux * RIDGE_RADIUS + RIDGE_OFFSET[0] + dx * RIDGE_WANDER,
    uz * RIDGE_RADIUS + RIDGE_OFFSET[1] + dz * RIDGE_WANDER,
  );
  const ridged = 1 - Math.abs(g * 2 - 1);
  const ridgeMask = smoothstep(RIDGE_IN, RIDGE_FULL, t) * smoothstepDown(t, RIDGE_FADE, RIDGE_OUT);
  let y = RIDGE_LIFT * (ridged - 0.5) * ridgeMask;

  const shoreFade = smoothstepDown(t, PEAK_SHORE_IN, PEAK_SHORE_OUT);
  for (const peak of PEAKS) {
    const px = dx - peak.x;
    const pz = dz - peak.z;
    const d = Math.sqrt(px * px + pz * pz) / peak.radius;
    const bump = smoothstepDown(d, 0, 1);
    y += peak.lift * bump * shoreFade;
  }

  y +=
    CLIFF_LIFT *
    sectorMask(ux, uz, CLIFF) *
    smoothstep(CLIFF_RISE_IN, CLIFF_RISE_OUT, t) *
    smoothstepDown(t, CLIFF_BROW, CLIFF_FOOT);
  return y;
}

/**
 * The floor mesh's own extent and subdivision, as `main` builds it. Exported
 * for `seafloorMeshHeight`, which has to agree with the triangles actually drawn.
 */
export const SEAFLOOR_EXTENT = 4000;
export const SEAFLOOR_SEGMENTS = 256;

/**
 * Height of the *rendered* floor at (x, z): the triangle the mesh actually
 * draws there, interpolated, rather than the analytic field.
 *
 * The two differ by up to a few decimetres, because the mesh samples the field
 * every fifteen metres and lerps between. That is invisible for anything with
 * height to it and fatal for anything lying flat — a strand of weed or a shell
 * seated on the analytic surface is buried under one triangle and floating
 * over the next. Mirrors `PlaneGeometry`'s layout after `rotateX(-pi/2)`: rows
 * run along +z, and each cell is split on the diagonal from (ix, iz+1) to
 * (ix+1, iz).
 */
export function seafloorMeshHeight(
  x: number,
  z: number,
  extent = SEAFLOOR_EXTENT,
  segments = SEAFLOOR_SEGMENTS,
): number {
  const step = extent / segments;
  const gx = (x + extent / 2) / step;
  const gz = (z + extent / 2) / step;
  const ix = Math.floor(gx);
  const iz = Math.floor(gz);
  const u = gx - ix;
  const v = gz - iz;
  const vx = (i: number) => i * step - extent / 2;
  const ha = seafloorHeight(vx(ix), vx(iz));
  const hb = seafloorHeight(vx(ix), vx(iz + 1));
  const hc = seafloorHeight(vx(ix + 1), vx(iz + 1));
  const hd = seafloorHeight(vx(ix + 1), vx(iz));
  if (u + v <= 1) return ha + (hd - ha) * u + (hb - ha) * v;
  return hc + (hb - hc) * (1 - u) + (hd - hc) * (1 - v);
}

/** Positive metres of water above the floor at (x, z); 0 where the floor is dry. */
export function seafloorDepth(x: number, z: number): number {
  return Math.max(0, -seafloorHeight(x, z));
}

// ------------------------------------------------------------------- the reef

/**
 * Where the reef gathers, in world XZ.
 *
 * This lives here, with the heightfield, rather than in `Props` — which is the
 * only module that *builds* anything out of it — because it is a fact about the
 * world and two modules need to agree on it. `Props` grows the coral and the
 * rock on these centres and `Fish` stations its resident schools over them, and
 * the point of the whole exercise is that those are the same places: a diver who
 * finds coral finds fish, and a diver who follows fish arrives at coral. Two
 * independent scatters over the same annulus produce neither.
 *
 * A uniform scatter was the problem this replaces. The plateau's reef band is
 * about 210,000 m² and underwater visibility here runs to a few tens of metres,
 * so any even spread — at any instance count a frame can afford — is invisible
 * on average and the seabed reads as an empty plain. Gathering it into a handful
 * of patches with open sand between them is both what a reef actually looks like
 * and the only arrangement that puts something in front of the camera.
 *
 * Stratified by bearing, one patch per sector, for the same reason the fish
 * radii are: seven free angles leave a third of the compass empty about half the
 * time, and a viewer who swims the wrong way finds the plain again. Radii are
 * square-rooted so the patches spread evenly by area, and inset by the patch
 * radius so none hangs off the plateau into the drop.
 *
 * Memoised: it is a pure function of module constants, and both callers want the
 * same list rather than two draws from the same distribution.
 */
export const REEF_BAND = {
  /** Clear of the spawn point, and inside the shallow plateau. */
  inner: 26,
  outer: 260,
  patches: 7,
  patchRadius: 15,
} as const;

const REEF_PATCH_SEED = 0x8f2c11;

let reefPatchCache: readonly { x: number; z: number }[] | null = null;

export function reefPatches(): readonly { x: number; z: number }[] {
  if (reefPatchCache) return reefPatchCache;

  const random = mulberry32(REEF_PATCH_SEED);
  const inner = REEF_BAND.inner + REEF_BAND.patchRadius;
  const outer = REEF_BAND.outer - REEF_BAND.patchRadius;
  const out: { x: number; z: number }[] = [];

  for (let i = 0; i < REEF_BAND.patches; i++) {
    const angle = ((i + 0.15 + random() * 0.7) / REEF_BAND.patches) * Math.PI * 2;
    const radius = inner + Math.sqrt(random()) * (outer - inner);
    out.push({ x: Math.cos(angle) * radius, z: Math.sin(angle) * radius });
  }

  // The nearest patch is pulled in to the inner edge of the band. Without it the
  // closest reef to the spawn point is wherever the seed happened to put one —
  // 90 m on this seed — and the first thing a viewer does when they dive at the
  // origin is swim across sand looking for the thing the scene is about.
  let nearest = 0;
  for (let i = 1; i < out.length; i++) {
    if (Math.hypot(out[i].x, out[i].z) < Math.hypot(out[nearest].x, out[nearest].z)) nearest = i;
  }
  const bearing = Math.atan2(out[nearest].z, out[nearest].x);
  const close = REEF_BAND.inner + REEF_BAND.patchRadius * 1.2;
  out[nearest] = { x: Math.cos(bearing) * close, z: Math.sin(bearing) * close };

  reefPatchCache = out;
  return out;
}

/**
 * The two world periods the sand normal is sampled at, metres.
 *
 * 13 and 47 are coprime enough that the pair does not visibly repeat inside the
 * few hundred metres a viewer can resolve detail over: the beat is at their
 * least common multiple, 611 m, by which distance the map is a mip average.
 * Neither is a round number for that reason — 12 and 48 would beat at 48.
 */
const SAND_DETAIL_NEAR = 13;
const SAND_DETAIL_FAR = 47;

/** Tangent-space strength of the sand map, matching the scale it used to pass. */
const SAND_NORMAL_SCALE = 0.75;

/**
 * How sharply the triplanar weights favour the dominant axis.
 *
 * Raising it narrows the band where two projections are both contributing. Too
 * low and every surface is a blend of three blurred copies; too high and the
 * transition becomes a visible seam along the 45-degree contour. Four is the
 * usual working value and it looks right on this terrain's gradients.
 */
const TRIPLANAR_SHARPNESS = 4;

// ------------------------------------------------------------ terrain shading

/**
 * Octave range of the shading bump, and how much slope it is worth.
 *
 * The island read as a smooth dome for a measurable reason: the finest octave of
 * the heightfield is ~25 m across and the mesh puts a vertex every 15.6 m, so the
 * surface is sampled at Nyquist and the normal cannot describe anything finer
 * than about a thirty-metre slope. Everything a viewer would call *terrain
 * texture* — the gullies, the scree, the broken ground — lives below that and had
 * nowhere to exist.
 *
 * The published fix is not more triangles. It is to evaluate more octaves at
 * shading rate and add their gradient to the normal, which is what iq's
 * `Rainforest` does with an fbm bump gated on `1 - |n.y|`. Two properties make it
 * the right trade here: the cost is per *pixel* rather than per vertex, so it
 * does not scale with the 4 km quad; and it changes nothing the CPU heightfield
 * has to agree with, so buoyancy, prop seating, the shoreline and the tour's
 * authored surf keys are all untouched.
 *
 * The range starts at octave 2 — inside what the mesh already carries — on
 * purpose. Those two octaves are where the visible faceting lives (53 m and 25 m
 * against a 15.6 m grid), and re-stating them at shading rate is what dissolves
 * the flat triangles into slope. It does mean the island reads as having rather
 * more relief than it geometrically has, which is the intent: the gap analysis's
 * complaint is that the dome is smooth, not that it is inaccurate.
 */
const BUMP_FROM = 2;
const BUMP_TO = 8;
const BUMP_NORM = amplitudeSum(BUMP_FROM, BUMP_TO);
/**
 * RMS slope the bump contributes on a fully-weighted surface.
 *
 * Converted to a gain below rather than used directly, because the gradient the
 * octave sum produces is in noise units per world metre and its magnitude depends
 * on the octave count. Expressing the constant as a slope means changing the
 * range does not silently change how rough the island looks.
 */
const BUMP_SLOPE = 0.34;
/** Measured RMS |grad| of the normalised octave sum over the range above. */
const BUMP_GRADIENT_RMS = 2.55;
const BUMP_GAIN = BUMP_SLOPE / BUMP_GRADIENT_RMS;
/**
 * How much of the bump survives on ground that faces straight up.
 *
 * iq's weight is `1 - |n.y|`, which is zero on the flat. That is right for a
 * cliff-and-scree landscape and wrong for this one, where the flat ground is a
 * beach and a lagoon floor that both have relief of their own. A third keeps the
 * sand from going glassy without competing with the sand normal map that owns
 * the metre scale down there.
 */
const BUMP_FLAT_WEIGHT = 0.33;

/**
 * Distance, in feature widths, over which an octave of the bump fades out.
 *
 * An octave is worth evaluating while its features are still wider than a couple
 * of pixels; past that it is not detail, it is noise, and it will crawl. At this
 * project's 1280 px and 55 degrees the angular size of a pixel is about 0.75
 * mrad, so a feature of `w` metres stops being resolvable somewhere around
 * `w / (2 * 0.00075)` metres away — roughly 660 feature widths. Fading between
 * 500 and 900 puts the transition around that without pretending to be exact.
 *
 * Per octave rather than one fade over the whole bump, which is the difference
 * between an island that keeps its ridges at a kilometre and one that goes smooth
 * as soon as you back off it. At 1 km everything down to a 1.5 m feature is still
 * resolvable and still drawn; only the last octave or two drop out.
 */
const BUMP_FADE_NEAR = 500;
const BUMP_FADE_FAR = 900;

/**
 * Octave range of the cavity term, and how dark it is allowed to get.
 *
 * Ambient occlusion by octave difference: evaluate the field detailed and
 * smoothed, and the residual is signed by whether the point sits in a hollow or
 * on a bump. `MdGfzh` derives its mountain AO exactly this way, and it costs
 * nothing here because the octaves are already being summed for the bump — the
 * value falls out of the same accumulation as the gradient.
 *
 * `Seafloor` previously had no occlusion term at all and said so, substituting a
 * flat `envMapIntensity`. That constant is what made every hollow on the island
 * the same brightness as every ridge.
 */
const AO_FROM = 2;
const AO_TO = 8;
const AO_RESIDUAL_AMP = amplitudeSum(AO_FROM, AO_TO) / FBM_NORM;
/** Ambient reaching the bottom of the deepest hollow the residual describes. */
const AO_FLOOR = 0.52;

/**
 * The terrain's own shadow: parameters of the heightfield march.
 *
 * Nothing on land was shadowed at all, and the reason is structural rather than
 * an oversight — the sun's shadow map is a +/-260 m box that follows the viewer,
 * and the island is a kilometre across sitting 1.4 km from the origin. No
 * cascade arrangement that also keeps the ship's contact shadow sharp will cover
 * it. So the island's own shadow is marched against the heightfield instead,
 * which is free of the box entirely and correct at any range.
 *
 * The accumulator is iq's published soft-shadow form — track `min(k * h / t)`
 * along the ray, so a ray that passes close to the terrain without hitting it
 * comes back partly shadowed and the penumbra widens with distance from the
 * occluder. The step is the current height above the terrain, floored so a ray
 * running parallel above a slope cannot stall, and the floor grows with distance
 * so 24 steps still reach kilometres.
 */
const SHADOW_HARDNESS = 20;
/** Start clear of the surface, or every lit pixel shadows itself. */
const SHADOW_START = 3;
/**
 * How far the caster surface is sunk below the receiver, metres.
 *
 * Without it the summit crown came out with a hard dark band across it and
 * angular blotches below — self-shadowing, from two independent sources that
 * add. The march traces a two-octave field while the pixel being shaded sits on
 * the four-octave one, and the renormalisation means the difference is not a
 * one-sided truncation: it runs to about +/- 2.2 m either way. On top of that the
 * *mesh* is a linear interpolation between vertices 15.6 m apart, so the true
 * field bulges up to a metre above the triangle mid-span. Wherever both errors
 * point the same way the caster is roughly three metres above the receiver, and
 * at this preset's 24-degree sun the ray has only climbed a metre by the time it
 * has travelled `SHADOW_START`.
 *
 * Sinking the caster is the standard depth-bias answer and it costs exactly what
 * it says: terrain features under four metres tall cast nothing. On a 150 m
 * island whose shadow work is the hillside shading its own flank, that is not a
 * feature anyone was going to see.
 */
const SHADOW_CASTER_BIAS = 4;
const SHADOW_MIN_STEP = 3;
const SHADOW_STEP_GROWTH = 0.09;
const SHADOW_MAX_STEP = 220;
const SHADOW_MAX_DIST = 4200;
/**
 * Below this the march is skipped entirely.
 *
 * Almost the whole frame is sea, and the seabed under it is both hidden by the
 * surface and lit through water rather than by the key light directly. Gating on
 * elevation means only the island's own pixels ever pay for the march, which is
 * what keeps a 24-step raymarch inside the budget.
 */
const SHADOW_MIN_Y = -2.5;
/** Octaves the march's height function evaluates. */
const SHADOW_OCTAVES = 2;

// --------------------------------------------------------------- sand detail

/**
 * Tiling normal map for the sand, derived from the same noise field so the
 * micro-detail shares a family resemblance with the macro shape.
 */
function createSandNormalTexture(): THREE.DataTexture {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  const height = new Float32Array(size * size);

  // Two octaves at frequencies that divide `size`, so the result tiles exactly.
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const a = valueNoise((x / size) * 16, (y / size) * 16);
      const b = valueNoise((x / size) * 48 + 31.7, (y / size) * 48 + 11.3);
      // Ripple ridges: sand under swell forms parallel bars, not isotropic bumps.
      //
      // The phase is dragged by the low-frequency octave rather than being a
      // clean multiple of the tile. Six bars per tile with a fixed phase is
      // exactly six bars per tile *everywhere*, and repeated across a seabed it
      // reads as corduroy with a visible seam — which is what it did. Modulating
      // the phase by `a` makes the bars wander, bifurcate and lose count, which
      // is both what real ripple fields do and what stops the eye locking onto
      // the period. The map still tiles exactly, because `a` does.
      const ripple = Math.sin((x / size) * Math.PI * 2 * 6 + a * 14) * 0.5 + 0.5;
      height[y * size + x] = a * 0.55 + b * 0.2 + ripple * 0.25;
    }
  }

  const strength = 2.6;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const xl = height[y * size + ((x - 1 + size) % size)];
      const xr = height[y * size + ((x + 1) % size)];
      const yd = height[((y - 1 + size) % size) * size + x];
      const yu = height[((y + 1) % size) * size + x];

      let nx = (xl - xr) * strength;
      let ny = (yd - yu) * strength;
      let nz = 1;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx *= inv;
      ny *= inv;
      nz *= inv;

      const i = (y * size + x) * 4;
      data[i] = Math.round((nx * 0.5 + 0.5) * 255);
      data[i + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      data[i + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      data[i + 3] = 255;
    }
  }

  const map = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.anisotropy = 8;
  map.colorSpace = THREE.NoColorSpace;
  map.needsUpdate = true;
  return map;
}

// ------------------------------------------------------------------- the class

export interface SeafloorOptions {
  /** Grid subdivisions per side. 256 gives ~15 m spacing over a 4 km extent. */
  segments?: number;
}

export class Seafloor {
  readonly mesh: THREE.Mesh;
  readonly extent: number;

  private readonly geometry: THREE.PlaneGeometry;
  private readonly material: THREE.MeshStandardNodeMaterial;
  private readonly noiseTexture: THREE.DataTexture;
  private readonly sandNormal: THREE.DataTexture;

  /** TSL entry points; built once, reused by every consumer. */
  private readonly nodes: NoiseNodes;

  /**
   * The octave accumulation, as one node object shared by two graphs.
   *
   * `normalNode` wants its gradient and `aoNode` wants its value, and they are
   * separate roots of the same fragment shader. Building the call once and
   * referencing the same node from both is what makes them share the work:
   * three's builder caches a node's generated snippet per build, so the twenty-
   * four fetches happen once. Calling `nodes.detail(...)` twice would create two
   * call nodes and pay for the octaves twice.
   */
  private readonly detailNode: Node;

  private readonly uCausticsStrength = uniform(1);
  /** Direction toward whichever body is currently the key light. */
  private readonly uKeyDir = uniform(new THREE.Vector3(0, 1, 0));
  /** March steps for the terrain's own shadow; 0 leaves the island unshadowed. */
  private readonly uShadowSteps: Node = uniform(24, 'int');
  private causticsNode: Node = null;
  private cloudShadow: ((worldPosition: Node) => Node) | null = null;
  private occlusionAttached = false;
  private disposed = false;

  constructor(extent: number, options: SeafloorOptions = {}) {
    this.extent = extent;
    const segments = options.segments ?? SEAFLOOR_SEGMENTS;

    this.noiseTexture = createNoiseTexture();
    this.sandNormal = createSandNormalTexture();
    // No `repeat` on the sand map: the normal node supplies its own world-space
    // uv at two scales through a triplanar projection, and a repeat set here
    // would be applied on top of them. The `detailTiling` option that used to
    // sit here was read, discarded with a `void`, and had been dead since the
    // two-scale blend replaced the single tile — a knob that documents a
    // behaviour the code no longer has is worse than no knob.

    this.nodes = buildNoiseNodes(this.noiseTexture);
    this.detailNode = this.nodes.detail(
      vec2(positionWorld.x, positionWorld.z),
      positionWorld.distance(cameraPosition),
    );

    this.geometry = new THREE.PlaneGeometry(extent, extent, segments, segments);
    // Bake the flip into the attributes so the position attribute's y really is
    // world up — the displacement loop below depends on that.
    this.geometry.rotateX(-Math.PI / 2);

    const position = this.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < position.count; i++) {
      position.setY(i, seafloorHeight(position.getX(i), position.getZ(i)));
    }
    position.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.geometry.computeBoundingSphere();

    this.material = new THREE.MeshStandardNodeMaterial();
    this.material.name = 'seafloor-sand';
    this.material.roughness = 0.93;
    this.material.metalness = 0;
    // Shore hook: sand the swash just left is glossy. See `ocean/Surf`.
    this.material.roughnessNode = Fn(() => swashRoughness(float(0.93), positionWorld))();
    /**
     * The sky is a source of light on this surface, not a mirror in it.
     *
     * `scene.environment` is the atmosphere's own cube capture, and at full
     * strength on a 0.8 km² dome it does something no amount of albedo work can
     * undo: it lays a flat blue-white fill over the whole island. Measured on the
     * hero frame, the vegetated slopes came out at sRGB (176, 183, 184) — a
     * saturation of 0.04, against 0.49 for the same slope in
     * `docs/ref/tropical-island-sea-level-v2.png`. Turning the environment off
     * entirely moved them to (164, 159, 145): still too bright, but a *colour*
     * again. The blue was the environment.
     *
     * It is not wrong that skylight is blue, and this does not pretend otherwise
     * — it is that the term arrives unoccluded. A canopy shades itself, a slope
     * shades its own hollows, and none of that exists here because the greenery
     * on the mid-slopes lives in this material's colour rather than in geometry
     * that could cast. 0.42 is the fraction of sky an averagely-enclosed patch of
     * ground actually sees, and using it is cheaper and steadier than the ambient
     * occlusion term this surface has no way to compute.
     */
    // 0.78, up from the 0.42 this was, and the two numbers describe the same
    // amount of light. 0.42 was chosen to hold back a blue-white fill from a sky
    // dome that has since been re-calibrated — `SKY_RADIANCE_SCALE` fell from
    // 0.35 to 0.16 when the grade change left it stale — so the term it was
    // fighting is already 2.2x smaller. Leaving it here would have charged the
    // island for the sky's brightness twice and taken the ambient off the
    // shaded slopes with it.
    this.material.envMapIntensity = 0.78;
    /**
     * Sand detail, sampled at two incommensurate world scales and blended.
     *
     * A single tiling map has one period, and that period is visible. This one
     * was laid down at `extent / detailTiling` repeats over a 4 km quad — with
     * the defaults, 571 repeats, which is a **7 metre** tile. Seven metres is
     * inside the range a viewer standing on the beach or swimming over the reef
     * resolves easily, so the seabed came out as a grid of identical patches,
     * and the ripple bars inside each one lined up across the seams into
     * corduroy. It is the single most obvious tell in the underwater frames.
     *
     * The fix is the standard one and it is not "make the tile bigger": that
     * trades a visible grid for a blurry one, because the texel density falls
     * with the same factor. Two samples at scales with no common multiple — 13 m
     * and 47 m here — put the beat frequency of the pair at hundreds of metres,
     * which is past where the detail is resolvable at all. The large scale also
     * costs nothing extra in memory: it is the same 256² map.
     *
     * Combined by the whiteout blend (add the tangent xy, multiply z) rather
     * than by averaging. Averaging two normals flattens both — two independent
     * bump fields average toward flat — and the result was a seabed that tiled
     * correctly and had no relief left. Whiteout keeps the slope of each.
     */
    const sample = (uv: Node, metres: number): Node =>
      texture(this.sandNormal, uv.mul(1 / metres)).xyz.mul(2).sub(1);

    /** Both scales, whiteout-blended, for one projection plane. */
    const plane = (uv: Node): Node => {
      const near = sample(uv, SAND_DETAIL_NEAR).toVar();
      const far = sample(uv, SAND_DETAIL_FAR).toVar();
      const blended = normalize(
        vec3(near.x.add(far.x), near.y.add(far.y), near.z.mul(far.z)),
      ).toVar();
      // The 0.75 the `normalMap` call this replaced passed as its scale — on the
      // seabed, where the ripple bars belong. Dry sand above the swash is
      // wind-smoothed, and at full strength the ripple map read on the beach as
      // dark swirls rather than as sand.
      const scale = mix(float(SAND_NORMAL_SCALE), float(0.08), positionWorld.y.smoothstep(0.2, 1.4)).toVar();
      return vec3(blended.x.mul(scale), blended.y.mul(scale), blended.z);
    };

    /**
     * The terrain normal, built in world space and handed over in view space.
     *
     * Two defects come out of this, and they had one cause between them: the
     * detail map was projected planar on world XZ, which stretches by `1/cos θ`
     * on any slope and degenerates entirely on steep ground — that is the
     * horizontal smearing across the sloped beach — and there was no relief at
     * all below the mesh's 15.6 m vertex spacing, which is the visible
     * triangulation. Neither can be fixed in the tangent frame the geometry
     * supplies, so the whole normal is assembled in world space instead and
     * transformed at the end. That also retires the dependency on the plane's
     * generated tangents.
     *
     * Order matters: the procedural relief goes on first, and the triplanar
     * blend weights are taken from the *relieved* normal. A cliff face that only
     * exists in the bump would otherwise still be sampled as if it were flat
     * ground, which is the same projection error one level down.
     */
    this.material.normalNode = Fn(() => {
      const wp = positionWorld.toVar();
      const geo = normalize(normalWorldGeometry).toVar();

      const d = this.detailNode.toVar();
      // Dry beach, between the swash and the back of the beach, is wind-smoothed:
      // the relief octaves that give the seabed and the hillside their texture
      // drew it as dark, melted swirls metres across. Damped there only.
      const dryBeach = wp.y.smoothstep(-0.3, 0.9).mul(smoothstepDownNode(wp.y, 3.5, 9)).toVar();
      const bumpWeight = mix(BUMP_FLAT_WEIGHT, 1, geo.y.abs().oneMinus())
        .mul(dryBeach.mul(-0.7).add(1))
        .toVar();
      // A heightfield's normal is `(-dh/dx, 1, -dh/dz)`, so added relief
      // subtracts its gradient from the horizontal components.
      const n = normalize(
        vec3(geo.x.sub(d.y.mul(bumpWeight)), geo.y, geo.z.sub(d.z.mul(bumpWeight))),
      ).toVar();

      // Whiteout triplanar blend (Ben Golus, "Normal Mapping for a Triplanar
      // Shader"): perturb each projection's tangent normal by the surface
      // normal's in-plane components, swizzle each into world orientation, and
      // weight by how much the surface faces that axis. Cheaper than building
      // three tangent frames and it keeps the slope of every projection instead
      // of averaging them toward flat.
      const w = n.abs().pow(TRIPLANAR_SHARPNESS).toVar();
      w.divAssign(w.x.add(w.y).add(w.z).max(1e-4));

      const tx = plane(vec2(wp.z, wp.y)).toVar();
      const ty = plane(vec2(wp.x, wp.z)).toVar();
      const tz = plane(vec2(wp.x, wp.y)).toVar();

      const cx = vec3(tx.z.abs().mul(n.x), tx.y.add(n.y), tx.x.add(n.z));
      const cy = vec3(ty.x.add(n.x), ty.z.abs().mul(n.y), ty.y.add(n.z));
      const cz = vec3(tz.x.add(n.x), tz.y.add(n.y), tz.z.abs().mul(n.z));

      const world = normalize(cx.mul(w.x).add(cy.mul(w.y)).add(cz.mul(w.z)));
      return world.transformNormalByViewMatrix(cameraViewMatrix);
    })();

    /**
     * Cavity occlusion, applied where occlusion belongs: to the indirect term.
     *
     * `envMapIntensity` above is a constant standing in for "the fraction of sky
     * an averagely-enclosed patch of ground sees", and its own comment admits it
     * is a substitute for the occlusion this surface had no way to compute. It
     * now can, so the constant describes the average and this describes the
     * variation about it — which is what puts a hollow in shade and a spur in
     * light instead of grading the whole dome evenly.
     */
    this.material.aoNode = Fn(() =>
      mix(AO_FLOOR, 1, this.detailNode.x.smoothstep(-0.7, 0.45)),
    )();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'seafloor';
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = false;
    // The floor is a single 4 km quad centred on the world; culling it against
    // its own bounding sphere is pure overhead and it is never off screen.
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();

    this.rebuildColorNode();
  }

  /** Signed depth below y = 0; positive means that much water above the floor. */
  depthAt(x: number, z: number): number {
    return -seafloorHeight(x, z);
  }

  /**
   * Floor depth as a TSL node, for the water shader's shallow-water tint.
   * `worldPosition` must be a vec3 node; only xz is read.
   */
  depthNode(worldPosition: unknown): unknown {
    const wp = worldPosition as Node;
    return this.nodes.height(vec2(wp.x, wp.z)).negate();
  }

  /** Floor elevation as a TSL node (negative below sea level). */
  heightNode(worldPosition: unknown): unknown {
    const wp = worldPosition as Node;
    return this.nodes.height(vec2(wp.x, wp.z));
  }

  /**
   * Injects the caustics projection. Expected to be centred near 1.0 — it
   * multiplies the sand's lit colour, so 1.0 means "no caustics here".
   * Triggers one shader rebuild; call it at setup, not per frame.
   */
  setCaustics(node: unknown): void {
    this.causticsNode = (node ?? null) as Node;
    this.rebuildColorNode();
  }

  /** Scales the injected caustics without a rebuild. */
  setCausticsStrength(value: number): void {
    this.uCausticsStrength.value = value;
  }

  /**
   * Supplies the cloud deck's shade, which the island receives along with its
   * own. Call before anything reads `keyShadowNode`.
   */
  setCloudShadow(cloudShadow: (worldPosition: Node) => Node): void {
    this.cloudShadow = cloudShadow;
  }

  /**
   * How much key light reaches a world point, 0..1 — the island's own shadow
   * times the cloud deck's.
   *
   * Both are analytic functions of world position rather than rasterised maps,
   * and both are unavailable to the shadow map for structural reasons: the
   * island is four times wider than the sun's +/-260 m shadow box, and the cloud
   * deck is a procedural field with no geometry to render.
   *
   * Public because the things standing *on* the island need the same answer the
   * island gets. A hillside with a shaded face and a fully-lit forest on it is
   * worse than no shadow at all.
   */
  keyShadowNode(worldPosition: unknown): Node {
    const wp = vec3(worldPosition as Node);
    const terrain = this.nodes.sunShadow(wp, this.uKeyDir, this.uShadowSteps);
    if (this.cloudShadow === null) return terrain;
    // Multiplied rather than taken as a minimum: a hillside in its own shadow
    // under a cloud is darker than either alone, because the two occluders are
    // independent.
    return terrain.mul(this.cloudShadow(wp));
  }

  /**
   * The cloud deck's shade alone, without the heightfield march.
   *
   * For consumers where the march is not worth its compile time. The dressing
   * bakes to about twenty distinct materials and a raymarch in each of them
   * takes the first frame from seconds to minutes on a slow shader compiler; the
   * props are also the case where it matters least, because anything close
   * enough for its own cast shadow to read is inside the sun's shadow box
   * already.
   */
  cloudShadowNode(worldPosition: unknown): Node {
    const wp = vec3(worldPosition as Node);
    return this.cloudShadow === null ? float(1) : this.cloudShadow(wp);
  }

  /**
   * Routes `keyShadowNode` into this material's own direct lighting. See
   * `core/lightOcclusion` for why that is not the same as multiplying albedo.
   *
   * Idempotent — a second call wraps nothing twice, so a tier change cannot
   * stack occlusion factors.
   *
   * @param light The key light. `Atmosphere` retargets the same light to the
   *   moon after sunset, which is why the direction is a separate uniform rather
   *   than being read off the light.
   */
  setKeyLight(light: THREE.DirectionalLight): void {
    if (this.occlusionAttached) return;
    this.occlusionAttached = true;
    occludeLight(this.material, light, Fn(() => this.keyShadowNode(positionWorld))());
  }

  /** Direction toward the key light, world space. Cheap; call it per frame. */
  setKeyDirection(dir: THREE.Vector3): void {
    this.uKeyDir.value.copy(dir);
  }

  /**
   * March steps for the terrain shadow. 0 disables it without a rebuild — the
   * loop simply does not run and the factor stays at 1.
   */
  setShadowSteps(steps: number): void {
    this.uShadowSteps.value = Math.max(0, Math.round(steps));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.geometry.dispose();
    this.material.dispose();
    this.noiseTexture.dispose();
    this.sandNormal.dispose();
  }

  // ------------------------------------------------------------------ internals

  private rebuildColorNode(): void {
    const caustics = this.causticsNode;
    const strength = this.uCausticsStrength;
    const valueNoise = this.nodes.valueNoise;

    this.material.colorNode = Fn(() => {
      const wp = positionWorld.toVar();
      const depth = wp.y.negate().toVar();
      const xz = vec2(wp.x, wp.z).toVar();

      // --- palette ------------------------------------------------------------
      //
      // A saturated, stylised-real tropical palette. The albedos are lower than
      // the sRGB targets they stand for (dry sand reads as about #e8d5a8, wet as
      // #b59a6e) because what a viewer sees is albedo times a 3.4-intensity key
      // plus sky, through the tone curve; at the targets' own values the beach
      // clips to white. The *ratios* are the targets': warm pale coral sand, a
      // wet band a little under half its value, and greens that stay green
      // rather than going olive under the sky fill.
      const sandDry: Node = vec3(0.5, 0.41, 0.25);
      const sandWet: Node = vec3(0.21, 0.155, 0.085);
      const shallowSand: Node = vec3(0.46, 0.42, 0.3);
      const deepSilt: Node = vec3(0.1, 0.15, 0.17);
      const soil: Node = vec3(0.13, 0.075, 0.04);
      const grass: Node = vec3(0.08, 0.135, 0.03);
      const jungle: Node = vec3(0.034, 0.088, 0.02);
      const jungleLush: Node = vec3(0.06, 0.13, 0.026);
      const rockLight: Node = vec3(0.23, 0.2, 0.16);
      const rockDark: Node = vec3(0.1, 0.085, 0.07);
      const moss: Node = vec3(0.06, 0.09, 0.03);

      // --- noise --------------------------------------------------------------
      // The same texture-backed value noise the heightfield uses, at three
      // scales: stands (90 m), patches (23 m) and grain (4.7 m).
      const stand: Node = valueNoise(xz.mul(1 / 90)).toVar();
      const patch: Node = valueNoise(xz.mul(1 / 23).add(vec2(17.1, 3.3))).toVar();
      const grain: Node = valueNoise(xz.mul(1 / 4.7).add(vec2(5.7, 11.9))).toVar();

      // --- slope --------------------------------------------------------------
      // From the mesh normal: 0 on the flat, rising with steepness. Rock is
      // where the ground is too steep to hold soil, broken up by the patch
      // noise so the exposure follows spurs and gullies rather than a contour.
      const steep: Node = float(1).sub(normalize(normalWorldGeometry).y).toVar();
      // Thinner soil higher up: the same slope exposes rock sooner near the
      // crest than on the lower flanks.
      const rockExposure: Node = steep
        .add(patch.sub(0.5).mul(0.12))
        .add(wp.y.smoothstep(ISLAND.peak * 0.45, ISLAND.peak * 1.2).mul(0.07))
        .smoothstep(0.1, 0.2)
        .toVar();

      // Strata: banded by elevation, the bands warped by the stand noise so
      // they dip and pinch rather than ruling the cliff with level lines.
      const layer: Node = wp.y.mul(0.42).add(stand.mul(14)).add(patch.mul(3)).sin().mul(0.5).add(0.5);
      const strata: Node = mix(
        rockDark,
        rockLight,
        layer.smoothstep(0.2, 0.8).mul(0.45).add(patch.mul(0.3)).add(grain.mul(0.25)),
      );
      // Moss and lichen on the upper faces of the rock.
      const rock: Node = mix(strata, moss, normalize(normalWorldGeometry).y.smoothstep(0.75, 0.9).mul(0.55)).toVar();

      // --- ground cover -------------------------------------------------------
      // Grass on the low flats behind the beach and in clearings; jungle
      // everywhere else, in stands of two greens. No cap: the forest runs to
      // the summit, and the summit is only rock where it is steep.
      const lowland: Node = smoothstepDownNode(wp.y, 6, 20).mul(smoothstepDownNode(steep, 0.03, 0.08)).mul(0.7);
      const clearing: Node = patch.smoothstep(0.66, 0.82).mul(0.45);
      const openness: Node = lowland.max(clearing).mul(stand.smoothstep(0.25, 0.7)).toVar();
      const forest: Node = mix(jungle, jungleLush, stand.smoothstep(0.3, 0.75)).toVar();
      // Open ground is scrub, not lawn: a flat grass albedo behind the beach
      // read as a mown verge from the sea. Tufts of darker scrub, straw-dry
      // patches and bare sandy soil break it at three scales, and the scrub
      // thickens toward the forest so the treeline grows out of it.
      const tuft: Node = valueNoise(xz.mul(1 / 1.9).add(vec2(41.3, 7.7))).toVar();
      const scrubTone: Node = mix(grass, jungleLush.mul(0.8), tuft.smoothstep(0.45, 0.8).mul(0.75)).toVar();
      const straw: Node = vec3(0.2, 0.17, 0.07);
      const dried: Node = mix(scrubTone, straw, patch.mul(0.6).add(grain.mul(0.4)).smoothstep(0.62, 0.86).mul(0.7));
      const scrub: Node = mix(dried, soil.mul(1.6), grain.mul(tuft).smoothstep(0.42, 0.7).mul(0.4)).toVar();
      const green: Node = mix(forest, scrub, openness).mul(grain.mul(0.3).add(0.85)).toVar();
      // The forest floor shows through as warm soil at the treeline's ragged edge.
      const growthEdge: Node = wp.y.smoothstep(GROWTH_START_METRES, GROWTH_FULL_METRES).toVar();
      const ragged: Node = growthEdge.mul(patch.mul(0.6).add(0.7)).clamp(0, 1).toVar();
      const vegetated: Node = mix(soil, green, ragged.smoothstep(0.35, 0.9).mul(VEGETATION_COVER).add(0.07)).toVar();

      // Beach: dry sand, with the vegetation creeping onto its back edge.
      const beach: Node = sandDry.mul(patch.mul(0.12).add(0.94)).mul(grain.mul(0.1).add(0.95)).toVar();
      const land: Node = mix(beach, vegetated, ragged.smoothstep(0.08, 0.5)).toVar();
      const ground: Node = mix(land, rock, rockExposure.mul(wp.y.smoothstep(1.5, 6))).toVar();

      // --- water's edge -------------------------------------------------------
      // Tight, because this is the beach edge: unambiguously sand above it,
      // seabed below it.
      const aboveWater: Node = wp.y.smoothstep(-0.8, 1.6).toVar();
      const submerged: Node = mix(deepSilt, shallowSand, smoothstepDownNode(depth, 4, 70)).toVar();
      const base: Node = mix(submerged, ground, aboveWater).toVar();

      // The wet band: everything the swash reaches is darker and, through
      // `roughnessNode`, glossy. Its upper edge wanders with the patch noise so
      // the tide line is not a contour.
      const wetTop: Node = patch.mul(0.9).add(1.4);
      const wet: Node = smoothstepDownNode(wp.y, wetTop.mul(0.35), wetTop).mul(wp.y.smoothstep(-1.2, -0.2)).toVar();
      const damp: Node = mix(base, mix(sandWet, base.mul(0.45), rockExposure), wet.mul(aboveWater.max(0.6))).toVar();

      // Seabed mottling: weed and darker sediment under the shallows.
      const seabedMottle: Node = mix(float(1), patch.mul(0.45).add(0.72), float(1).sub(aboveWater));
      const mottled: Node = damp.mul(seabedMottle).toVar();

      // Shore hook: the swash darkens the sand it reaches and it dries over
      // seconds. Scales whatever the palette above produced; see `ocean/Surf`.
      const surfed = applySwashToGround(mottled, wp).toVar();

      if (caustics === null) return vec4(surfed, 1);

      // Caustics only exist under water, and fade out as the floor gets deep
      // enough that the surface pattern has diverged into ambient light.
      const reach = smoothstepDownNode(depth, 2, 48).mul(float(1).sub(aboveWater)).toVar();
      const lit = mix(float(1), caustics, reach.mul(strength)).toVar();
      return vec4(surfed.mul(lit), 1);
    })();

    // Wet sand is a mirror for a moment after each wave; dry sand is chalk.
    this.material.roughnessNode = Fn(() => {
      const y = positionWorld.y;
      const wetTop = valueNoise(vec2(positionWorld.x, positionWorld.z).mul(1 / 23).add(vec2(17.1, 3.3)))
        .mul(0.9)
        .add(1.4);
      const wet = smoothstepDownNode(y, wetTop.mul(0.35), wetTop).mul(y.smoothstep(-1.5, -0.4));
      return mix(float(0.93), float(0.38), wet);
    })();

    this.material.needsUpdate = true;
  }
}

// -------------------------------------------------------------- TSL heightfield

interface NoiseNodes {
  /** vec2 -> float in [0, 1]. */
  valueNoise: (p: Node) => Node;
  /** vec2 -> float in [0, 1]. */
  fbm: (p: Node) => Node;
  /** World xz (vec2) -> floor elevation in metres. */
  height: (p: Node) => Node;
  /**
   * World xz plus view distance -> `vec3(cavity, slopeX, slopeZ)`.
   *
   * One accumulation serving two consumers: `cavity` is the signed octave
   * residual the ambient occlusion is read off, and `slopeXZ` is the shading
   * bump's gradient with each octave already faded by whether this pixel can
   * resolve it. Kept as one call because the two want the same texture fetches
   * and separating them would double the cost of the more expensive half.
   */
  detail: (p: Node, viewDistance: Node) => Node;
  /**
   * World position and key-light direction -> sunlight reaching it, 0..1.
   *
   * A soft march against the heightfield, so it shadows the whole island at any
   * range rather than only what fits inside the sun's shadow box.
   */
  sunShadow: (worldPosition: Node, lightDir: Node, steps: Node) => Node;
}

/**
 * Builds the TSL mirror of the CPU heightfield.
 *
 * Everything is expressed as parameterised `Fn`s rather than closures over
 * caller-scope variables. A no-argument `Fn` that reads a `toVar()` declared by
 * its caller is only correct if TSL happens to inline the body; passing the
 * value in as a parameter is correct either way.
 */
function buildNoiseNodes(map: THREE.Texture): NoiseNodes {
  /**
   * Bilinear value noise matching `valueNoise()` above, texel for texel.
   *
   * Explicit LOD 0 on every fetch: the water material may call `depthNode()`
   * from its vertex stage, where implicit derivatives do not exist and an
   * unqualified sample is a WGSL validation error.
   */
  const valueNoise = Fn(([p]: [Node]) => {
    const i = p.floor().toVar();
    const f = p.sub(i).toVar();
    const u = f.mul(f).mul(f.mul(-2).add(3)).toVar();

    const base = i.add(0.5).div(NOISE_SIZE).toVar();
    const step = float(1 / NOISE_SIZE);

    const a = texture(map, base, 0).r.toVar();
    const b = texture(map, base.add(vec2(step, 0)), 0).r.toVar();
    const c = texture(map, base.add(vec2(0, step)), 0).r.toVar();
    const d = texture(map, base.add(vec2(step, step)), 0).r.toVar();

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  });

  /**
   * The same bilinear patch, returning its analytic gradient alongside its value.
   *
   * `vec3(n, dn/dpx, dn/dpy)`, with the derivative taken with respect to the
   * *noise-space* coordinate. Written out as the expanded bilinear polynomial
   * rather than as nested `mix`es because the derivative is then obvious:
   *
   *   n  = a + k0 u.x + k1 u.y + k2 u.x u.y
   *   du = 6 f (1 - f)                        the Hermite's own derivative
   *
   * and the four corner reads are shared between value and gradient. That
   * sharing is the entire economic argument for doing it this way — the gradient
   * is arithmetic on numbers the sampler has already fetched.
   */
  const noised = Fn(([p]: [Node]) => {
    const i = p.floor().toVar();
    const f = p.sub(i).toVar();
    const u = f.mul(f).mul(f.mul(-2).add(3)).toVar();
    const du = f.mul(f.oneMinus()).mul(6).toVar();

    const base = i.add(0.5).div(NOISE_SIZE).toVar();
    const step = float(1 / NOISE_SIZE);

    // Typed loosely for the same reason the rest of this module is: the chained
    // arithmetic below mixes these with `any`-typed swizzles of `u`, and TSL's
    // declarations resolve that to the widest overload rather than to `float`.
    const a: Node = texture(map, base, 0).r.toVar();
    const b: Node = texture(map, base.add(vec2(step, 0)), 0).r.toVar();
    const c: Node = texture(map, base.add(vec2(0, step)), 0).r.toVar();
    const d: Node = texture(map, base.add(vec2(step, step)), 0).r.toVar();

    const k0 = b.sub(a).toVar();
    const k1 = c.sub(a).toVar();
    const k2 = a.sub(b).sub(c).add(d).toVar();

    return vec3(
      a.add(k0.mul(u.x)).add(k1.mul(u.y)).add(k2.mul(u.x).mul(u.y)),
      du.x.mul(k0.add(k2.mul(u.y))),
      du.y.mul(k1.add(k2.mul(u.x))),
    );
  });

  /**
   * fbm over octaves `[from, to)`, **unnormalised**, value only.
   *
   * The normaliser is the caller's business because the two uses want different
   * ones: the heightfield divides by the full four-octave sum so its relief keeps
   * the amplitude it always had, while a partial run used as a residual has to be
   * measured against its own amplitude to mean anything.
   *
   * Octaves below `from` still advance the chain — the rotation and the offset
   * are what decorrelate the sum — but skip the fetch, so a residual costs only
   * the octaves it actually contains.
   */
  const makeFbm = (from: number, to: number) =>
    Fn(([p]: [Node]) => {
      const q = p.toVar();
      const sum = float(0).toVar();
      let amplitude = 1;
      for (let o = 0; o < to; o++) {
        if (o >= from) sum.addAssign(valueNoise(q).mul(amplitude));
        const rx = q.x.mul(ROT[0]).add(q.y.mul(ROT[1])).toVar();
        const ry = q.x.mul(ROT[2]).add(q.y.mul(ROT[3])).toVar();
        q.assign(
          vec2(
            rx.mul(LACUNARITY).add(OCTAVE_OFFSET[0]),
            ry.mul(LACUNARITY).add(OCTAVE_OFFSET[1]),
          ),
        );
        amplitude *= GAIN;
      }
      return sum;
    });

  const fbmFn = Fn(([p]: [Node]) => makeFbm(0, OCTAVES)(p).mul(1 / FBM_NORM));
  const fbmCoarse = makeFbm(0, SHADOW_OCTAVES);
  const COARSE_NORM = amplitudeSum(0, SHADOW_OCTAVES);

  /**
   * `sectorMask()` above, node for node. A plain arrow rather than an `Fn`
   * because it takes everything it reads as an argument, so it inlines and the
   * scoping hazard the header describes cannot apply.
   */
  const sectorMaskNode = (u: Node, s: Sector): Node =>
    u.dot(vec2(s.x, s.z)).smoothstep(s.edge, 1);

  /** `islandRelief()` above, term for term and in the same order. */
  const islandReliefNode = (dv: Node, u: Node, t: Node): Node => {
    const g = valueNoise(
      vec2(
        u.x.mul(RIDGE_RADIUS).add(RIDGE_OFFSET[0]).add(dv.x.mul(RIDGE_WANDER)),
        u.y.mul(RIDGE_RADIUS).add(RIDGE_OFFSET[1]).add(dv.y.mul(RIDGE_WANDER)),
      ),
    ).toVar();
    const ridged = float(1).sub(g.mul(2).sub(1).abs()).toVar();
    const ridgeMask = t
      .smoothstep(RIDGE_IN, RIDGE_FULL)
      .mul(smoothstepDownNode(t, RIDGE_FADE, RIDGE_OUT))
      .toVar();
    const y = ridged.sub(0.5).mul(RIDGE_LIFT).mul(ridgeMask).toVar();

    const shoreFade = smoothstepDownNode(t, PEAK_SHORE_IN, PEAK_SHORE_OUT).toVar();
    for (const peak of PEAKS) {
      const d = dv.sub(vec2(peak.x, peak.z)).length().div(peak.radius);
      const bump = smoothstepDownNode(d, 0, 1).toVar();
      y.addAssign(bump.mul(peak.lift).mul(shoreFade));
    }

    y.addAssign(
      sectorMaskNode(u, CLIFF)
        .mul(t.smoothstep(CLIFF_RISE_IN, CLIFF_RISE_OUT))
        .mul(smoothstepDownNode(t, CLIFF_BROW, CLIFF_FOOT))
        .mul(CLIFF_LIFT),
    );
    return y;
  };

  /** `shoreFraction()` above, term for term and in the same order. */
  const shoreFractionNode = (u: Node): Node => {
    const e = u.dot(vec2(ELONGATION_AXIS.x, ELONGATION_AXIS.z)).toVar();
    const l = u.dot(vec2(LOBE_AXIS.x, LOBE_AXIS.z)).toVar();
    const d = u.dot(vec2(DRIFT_AXIS.x, DRIFT_AXIS.z)).toVar();

    const elongation = e.mul(e).mul(2).sub(1).mul(ELONGATION).toVar();
    const lobes = l.mul(l).mul(l).mul(4).sub(l.mul(3)).mul(LOBES).toVar();
    const drift = d.mul(DRIFT).toVar();
    const bay = sectorMaskNode(u, BAY).mul(BAY_CUT).toVar();
    const headland = sectorMaskNode(u, HEADLAND).mul(HEADLAND_REACH).toVar();

    return float(1).add(elongation).add(lobes).add(drift).sub(bay).add(headland).max(SHORE_FLOOR);
  };

  /**
   * Mirrors `seafloorHeight()` exactly.
   *
   * Read the two side by side: same locals, same order, same constants, and the
   * same three arguments to every ramp. That correspondence is the only thing
   * keeping the buoyancy solver, the prop placement and the water's depth term
   * on the same surface as this mesh, so a term added to one half without the
   * other is not a cosmetic bug — it is props buried in sand and fish inside
   * rock.
   */
  /**
   * @param noiseFn Unnormalised fbm over whichever octaves this instance wants.
   * @param norm Its amplitude sum, so the relief stays centred on zero whatever
   *   the octave count. Only the four-octave instance is the mirror of
   *   `seafloorHeight`; the coarse one exists solely to be marched against, and
   *   differs from the real surface by the couple of metres its missing octaves
   *   are worth.
   */
  const makeHeight = (noiseFn: Node, norm: number) =>
    Fn(([p]: [Node]) => {
      const xz = p.toVar();
      const n = noiseFn(xz.mul(FEATURE_SCALE)).mul(1 / norm).toVar();

      const rOrigin = xz.length().toVar();

      // `dv` is (dx, dz); its `.y` is the world z offset throughout.
      const dv = xz.sub(vec2(ISLAND.x, ISLAND.z)).toVar();
      const dIsland = dv.length().toVar();
      const inv = float(1).div(dIsland.max(1)).toVar();
      const u = dv.mul(inv).toVar();

      const shore = shoreFractionNode(u).mul(ISLAND.radius).toVar();
      const t = dIsland.div(shore).toVar();

      const sv = dv.sub(vec2(SUMMIT_OFFSET.x, SUMMIT_OFFSET.z)).toVar();
      const tCrest = sv.length().div(shore.mul(CREST_SPAN)).toVar();

      const along = dv.dot(vec2(SPIT.x, SPIT.z)).toVar();
      const across = dv.y.mul(SPIT.x).sub(dv.x.mul(SPIT.z)).toVar();
      const offset = across.sub(along.mul(along).mul(SPIT_CURVE)).abs().toVar();
      const run = along
        .smoothstep(SPIT_ROOT, SPIT_ROOT + SPIT_RISE)
        .mul(smoothstepDownNode(along, SPIT_TIP - SPIT_TAPER, SPIT_TIP))
        .toVar();
      const shoal = run.mul(smoothstepDownNode(offset, SPIT_SHOAL_CORE, SPIT_SHOAL_EDGE)).toVar();
      const crest = run.mul(smoothstepDownNode(offset, SPIT_CORE, SPIT_EDGE)).toVar();

      const shallowOrigin = smoothstepDownNode(rOrigin, PLATEAU_RADIUS, SHELF_RADIUS).toVar();
      const shallowIsland = smoothstepDownNode(t, SKIRT_IN, SKIRT_OUT).toVar();
      const shallowness = shallowOrigin.max(shallowIsland).max(shoal).toVar();

      const y = float(DEEP_Y).add(float(PLATEAU_Y - DEEP_Y).mul(shallowness)).toVar();
      y.addAssign(n.sub(0.5).mul(RELIEF).mul(shallowness.mul(0.65).add(0.35)));
      y.addAssign(smoothstepDownNode(t, APRON_IN, APRON_OUT).mul(SHORE_LIFT));
      y.addAssign(smoothstepDownNode(tCrest, CREST_IN, CREST_OUT).mul(CREST_LIFT));
      y.addAssign(
        sectorMaskNode(u, HEADLAND)
          .mul(t.smoothstep(HEADLAND_TOE, HEADLAND_CROWN))
          .mul(smoothstepDownNode(t, HEADLAND_BROW, HEADLAND_FALL))
          .mul(HEADLAND_LIFT),
      );
      y.addAssign(crest.mul(SPIT_LIFT));
      y.addAssign(islandReliefNode(dv, u, t));

      const lagoon = sectorMaskNode(u, LAGOON)
        .mul(t.smoothstep(LAGOON_IN, LAGOON_FULL))
        .mul(smoothstepDownNode(t, LAGOON_EDGE, LAGOON_OUT))
        .toVar();
      const fill = float(LAGOON_Y).sub(y).max(0).toVar();
      y.addAssign(lagoon.mul(fill));
      return y;
    });

  const height = makeHeight(makeFbm(0, OCTAVES), FBM_NORM);
  /**
   * The same surface at two octaves, for the shadow march to trace against.
   *
   * Four octaves times twenty-four steps is ninety-six fetches a pixel purely to
   * decide whether the sun is blocked, and the answer does not depend on the
   * metre-scale detail: what shadows the island is the island. Two octaves keep
   * the shore, the crest, the headland and the fifty-metre undulation, and cost
   * half as much. The residual disagreement with the real surface is the +/- 2.2 m
   * the missing octaves carry, which `SHADOW_START` already stands clear of.
   */
  const heightCoarse = makeHeight(fbmCoarse, COARSE_NORM);

  /**
   * Shading bump gradient and cavity residual, from one octave accumulation.
   *
   * Each octave is faded by whether this pixel can resolve it — see
   * `BUMP_FADE_NEAR`. The fade is applied to the *gradient* only: the cavity
   * value is a low-frequency quantity that stays meaningful at any range, and
   * fading it would make distant hollows fill in with light.
   */
  const detail = Fn(([p, viewDistance]: [Node, Node]) => {
    const q = p.mul(FEATURE_SCALE).toVar();
    const cavity = float(0).toVar();
    const slope = vec2(0, 0).toVar();
    let amplitude = 1;

    for (let o = 0; o < BUMP_TO; o++) {
      if (o >= Math.min(AO_FROM, BUMP_FROM)) {
        const n = noised(q).toVar();
        if (o >= AO_FROM && o < AO_TO) cavity.addAssign(n.x.mul(amplitude));
        if (o >= BUMP_FROM) {
          const j = OCTAVE_JACOBIAN[o];
          // grad_p = J^T * grad_q, with J constant per octave.
          const fade = smoothstepDownNode(
            viewDistance,
            octaveFeatureMetres(o) * BUMP_FADE_NEAR,
            octaveFeatureMetres(o) * BUMP_FADE_FAR,
          );
          slope.addAssign(
            vec2(
              n.y.mul(j[0]).add(n.z.mul(j[2])),
              n.y.mul(j[1]).add(n.z.mul(j[3])),
            ).mul(amplitude * (1 / BUMP_NORM)).mul(fade),
          );
        }
      }
      const rx = q.x.mul(ROT[0]).add(q.y.mul(ROT[1])).toVar();
      const ry = q.x.mul(ROT[2]).add(q.y.mul(ROT[3])).toVar();
      q.assign(
        vec2(
          rx.mul(LACUNARITY).add(OCTAVE_OFFSET[0]),
          ry.mul(LACUNARITY).add(OCTAVE_OFFSET[1]),
        ),
      );
      amplitude *= GAIN;
    }

    // Signed and normalised to roughly [-1, 1]: negative in a hollow, positive on
    // a rise. The raw sum is a positive quantity centred on half its amplitude.
    const signedCavity = cavity
      .mul(1 / FBM_NORM)
      .div(AO_RESIDUAL_AMP)
      .mul(2)
      .sub(1);

    return vec3(signedCavity, slope.mul(BUMP_GAIN));
  });

  const sunShadow = Fn(([worldPosition, lightDir, steps]: [Node, Node, Node]) => {
    const p = vec3(worldPosition).toVar('terrainShadowP');
    const res = float(1).toVar('terrainShadowRes');

    // Two gates, and between them they are what makes this affordable. Almost
    // every pixel of a typical frame is sea, and the seabed under it is lit
    // through water rather than by the key light — so only the island's own
    // pixels ever enter the loop. The second gate retires the march when the key
    // light is at or below the horizon, where it contributes nothing to shade.
    If(p.y.greaterThan(SHADOW_MIN_Y).and(lightDir.y.greaterThan(0.02)), () => {
      const t = float(SHADOW_START).toVar('terrainShadowT');

      Loop(steps, () => {
        const s = p.add(lightDir.mul(t)).toVar('terrainShadowS');
        const h = s.y
          .sub(heightCoarse(vec2(s.x, s.z)).sub(SHADOW_CASTER_BIAS))
          .toVar('terrainShadowH');

        // iq's soft-shadow accumulator: the closest approach, measured as an
        // angle, is the penumbra. A ray that clears a ridge by little comes back
        // partly shadowed, and the softness grows with distance to the occluder
        // for free because `t` is in the denominator.
        res.assign(min(res, h.mul(SHADOW_HARDNESS).div(t)));

        // Step by the clearance, so open sky is crossed in a few strides and the
        // ray only slows where it is close to the ground. Floored so a ray
        // running parallel just above a slope cannot stall, and the floor grows
        // with distance so the fixed step count still reaches kilometres.
        t.addAssign(
          clamp(h, float(SHADOW_MIN_STEP).add(t.mul(SHADOW_STEP_GROWTH)), SHADOW_MAX_STEP),
        );

        If(res.lessThan(0.02).or(t.greaterThan(SHADOW_MAX_DIST)), () => {
          Break();
        });
      });
    });

    return res.clamp(0, 1);
  });

  return {
    valueNoise: (p: Node) => valueNoise(p),
    fbm: (p: Node) => fbmFn(p),
    height: (p: Node) => height(p),
    detail: (p: Node, viewDistance: Node) => detail(p, viewDistance),
    sunShadow: (worldPosition: Node, lightDir: Node, steps: Node) =>
      sunShadow(worldPosition, lightDir, steps),
  };
}
