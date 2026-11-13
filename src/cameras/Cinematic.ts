import * as THREE from 'three';
import { ISLAND, seafloorHeight } from '../scene/Seafloor';

/**
 * The attract-mode tour: a looping, edited sequence of authored shots that
 * shows the scene off — and sails the ship while it does it.
 *
 * **It is cut, not flown.** The previous tour was one continuous spline, which
 * made it smooth and made it a drone flight: every subject had to be reached by
 * travelling to it, so half the lap was transit and most frames were the island
 * from a kilometre off. A trailer is a sequence of *shots*, each with one idea,
 * joined by edits. Each shot here has its own eye and look splines, its own lens,
 * ease, handheld amount and clearance rule, and says how it is entered: a hard
 * cut, a whip pan, or a blend for the one join that has to be a continuous move
 * (the loop point, which `tests/ocean.spec.ts` holds to having no cut).
 *
 * **Shots are framed on the live hull.** A shot's keys are written in one of
 * three spaces: the world; the hull's own frame (bow +x, starboard +z); or a
 * hull-centred frame turned to face a fixed world bearing or a world point, so
 * "behind the ship, looking at the island" stays true wherever the ship is. The
 * old tour aimed at where the hull *should* have been and photographed empty
 * water whenever the physics disagreed — which it always did in captures, where
 * the harness puts the hull back at its spawn and only then rewinds the tour.
 * Reading the hull's current pose keeps the rig free of per-frame memory, which
 * is the property determinism actually rests on: the pose is a function of the
 * loop clock, the hull's pose *now* and the wave field *now*, all three of which
 * the harness resets. With no hull supplied (a unit test, or the seconds before
 * the ship loads), the closed-form nominal hull stands in.
 *
 * **The hull is driven, not moved.** Every shot carries an engine order, and the
 * rudder is derived so the ship holds one turn radius for the lap and the circuit
 * closes — see `TRACK_RADIUS`. Nothing here writes the ship's transform.
 *
 * **Clearance is enforced, smoothly.** Outside the underwater shot the eye is
 * held above a soft maximum of the live wave height over a small footprint, plus
 * the shot's margin; everywhere it is held above the terrain plus an allowance
 * for the canopy; and it is pushed out of an ellipse around the hull. Each is a
 * softplus rather than a clamp, so a correction arrives as a curve, not as a
 * corner. The authored keys already clear everything; this is what makes the
 * low wave-skimming shot possible at all, because no authored height clears a
 * swell that has not been simulated yet.
 */

// ---------------------------------------------------------------- hull mirrors

/**
 * Terminal speed at full ahead, m/s: `sqrt(MAX_THRUST / DRAG_LONGITUDINAL)` from
 * `ShipController`. Mirrored rather than imported because those constants are
 * private to the controller. A stale mirror makes the circuit close at a slightly
 * different radius; the shots are framed on the live hull and do not care.
 */
const HULL_TOP_SPEED = 9.64;

/**
 * Steady yaw rate per unit of rudder at full authority, rad/s:
 * `RUDDER_TORQUE / YAW_DAMPING` from `ShipController`. The controller's yaw is
 * first order, so a held rudder settles at exactly this rate times the rudder —
 * which is what makes a target turn radius invertible into a rudder order.
 */
const HULL_YAW_RATE_PER_RUDDER = 0.214;

/**
 * Fraction of that rate the hull actually delivers, measured rather than
 * derived: `Buoyancy` applies its own angular damping on top of the
 * controller's, and the rudder spools, so a held order turns the real hull at
 * about half the controller's own steady rate. Measured over two laps of the
 * tour (3.05 rad of heading per lap against 2π asked for). Without it the hull
 * sails a circle twice the authored size, closes it every *second* lap and puts
 * its far side 390 m out, off the plateau.
 */
const HULL_YAW_EFFICIENCY = 0.485;

/** Speed at which the rudder reaches full authority — `RUDDER_REFERENCE_SPEED`. */
const HULL_RUDDER_REFERENCE_SPEED = 5.5;

/**
 * The hull, for clearance: half-length and half-beam of the ellipse the eye is
 * kept out of, metres, and the height above which the rig is clear of the masts.
 * The ship is 27 m long and about 7 m in the beam; the margins allow for the
 * bowsprit, the yards swinging over the side as she rolls, and the live hull
 * being read a frame late.
 */
const HULL_CLEAR_HALF_LENGTH = 21;
const HULL_CLEAR_HALF_BEAM = 9.5;
const HULL_CLEAR_TOP = 36;

// ------------------------------------------------------------------- authoring

type V3 = readonly [number, number, number];

/**
 * What a shot's coordinates are measured from.
 *
 * - `world`: absolute.
 * - `hull`: the live hull's frame. +x is the bow, +z starboard, and y is height
 *   above mean sea level rather than above the hull, so the frame does not heave
 *   — a camera that rose and fell with the hull would put the whole sea in
 *   motion behind it.
 * - `bearing`: hull-centred, with +x along a fixed world bearing. For shots
 *   composed against something in the world rather than against the ship —
 *   the sunrise, the sunset — but which still have to have the ship in them.
 * - `toward`: hull-centred, with +x pointing from the hull at a world point.
 */
type Space =
  | { readonly kind: 'world' }
  | { readonly kind: 'hull' }
  | { readonly kind: 'bearing'; readonly x: number; readonly z: number }
  | { readonly kind: 'toward'; readonly x: number; readonly z: number };

/**
 * How a shot is entered from the one before it.
 *
 * - `cut`: a hard cut on the boundary.
 * - `whip`: a whip pan — the outgoing shot accelerates into a pan for
 *   `WHIP_SECONDS`, the cut lands at the fastest point, and the incoming shot
 *   decelerates out of the same pan. `+1` pans right, `-1` left.
 * - `blend`: both shots evaluated across the window and their poses mixed under
 *   a smootherstep, so the join is a continuous camera move. The loop point uses
 *   this; anything else that wants a "dissolve" without rendering the frame twice
 *   can too.
 */
type Transition =
  | { readonly kind: 'cut' }
  | { readonly kind: 'whip'; readonly direction: 1 | -1 }
  | { readonly kind: 'blend'; readonly seconds: number };

interface Shot {
  /** Stable identifier, surfaced as the beat name. */
  name: string;
  /** Seconds. Also sets how much of the hull's circuit this shot covers. */
  duration: number;
  /**
   * Engine order, 0..1 ahead. Stepped at the boundary on purpose:
   * `ShipController` already spools throttle and rudder over seconds, so a step
   * here arrives at the solver as a smooth telegraph order.
   */
  throttle: number;
  space: Space;
  /** Eye control points, in `space`. Interpolated by a centripetal Catmull–Rom, by arc length. */
  eye: readonly V3[];
  /** Look-at control points, in `space`. Same interpolation, same eased clock. */
  look: readonly V3[];
  /**
   * Ease on the shot's own clock, 0..1. 0 is constant speed along the curve; 1
   * is a full smootherstep — the camera starts and finishes at rest. Between,
   * the move arrives moving and settles a little, which is what a camera
   * operator hitting a mark does. The ease is shared by the eye and the look so
   * the two stay in step.
   */
  ease: number;
  /** Vertical field of view at the start and end of the shot, degrees. */
  fov: readonly [number, number];
  /** Dutch angle, degrees, positive rolls the horizon clockwise. */
  roll?: number;
  /** Slow roll oscillation on top of `roll`, degrees — the storm's heel. */
  sway?: number;
  /** Handheld amount, degrees of drift. 0 for a locked-off or crane shot. */
  shake: number;
  /** Metres the eye is held above the local wave crests. */
  clearance: number;
  /** True for the one shot allowed to go below the surface. */
  underwater?: boolean;
  enter: Transition;
}

/**
 * Seconds each side of a whip-pan cut, and the pan it sweeps through, degrees.
 *
 * The sweep is all of the pan the eye ever sees — 32 degrees into the cut and
 * 32 out of it in a quarter of a second each — and the cut lands at the fastest
 * frame, where the eye cannot tell one blurred sea from another.
 */
const WHIP_SECONDS = 0.28;
const WHIP_DEGREES = 32;

/** Height above sea level the hull-framed look points aim at by default, metres. */
const RIG = 8;

/** World point the island shots aim at: the summit's shoulder rather than its foot. */
const ISLAND_LOOK: V3 = [ISLAND.x, 55, ISLAND.z];

/**
 * A point on a bearing from the island's centre, for keys that follow the coast.
 * Bearings are radians from +x toward +z, as `Props` places the cove on.
 */
function offIsland(bearing: number, radius: number, y: number): V3 {
  return [ISLAND.x + Math.cos(bearing) * radius, y, ISLAND.z + Math.sin(bearing) * radius];
}

/**
 * Dawn and dusk compositions, as horizontal bearings.
 *
 * The sun's azimuth is `(hours - 6) / 24 * 2π + π` and its direction
 * `(sin az, ·, cos az)` — see `main.sunFromClock` and `Atmosphere` — so it rises
 * toward −z and sets toward +z, swinging through −x at noon. `SUNRISE_VIEW` is
 * turned 27° west of the sunrise, which puts the dawn glow on the right of frame
 * and the island, 29° the other side of the axis, on the left. `SUNSET_VIEW` is
 * the sun's bearing at 17:35, when the hull, by the circuit's own geometry, is
 * sailing almost straight into it.
 */
const SUNRISE_VIEW = { kind: 'bearing', x: -0.45, z: -0.893 } as const;
const SUNSET_VIEW = { kind: 'bearing', x: -0.105, z: 0.994 } as const;

/** The reef patch the dive is built around — `Seafloor.reefPatches()[6]`. */
const DIVE_REEF = { x: 147, z: -121 } as const;

/**
 * The shot list.
 *
 * Nine shots, ninety seconds, one day. The order is the day's: dawn, morning on
 * the water, noon under it, the afternoon on the ship and the island, sunset,
 * a squall at dusk, the night watch, and the night pushing back into the dawn
 * the lap opens on. That is also why it loops: the last shot is built to finish
 * exactly where the first begins, in the same hull-centred frame, so the join is
 * a camera move rather than an edit.
 *
 * Every hull-framed key stays outside the clearance ellipse and every world key
 * stays clear of the terrain on its own; the runtime guard is the second line,
 * not the first. Heights in the low shots are deliberately *below* what they
 * will fly at — the wave guard lifts them onto the crests, and authoring them
 * there would leave daylight under the lens in a calm sea.
 */
const SHOTS: readonly Shot[] = [
  {
    // Establishing. Low off the stern quarter in the half-light, the hull dark
    // against the glow, then a long crane up and back until the whole seascape
    // is in frame: ship lower centre, sunrise right, the island left.
    name: 'dawn-crane',
    duration: 10,
    throttle: 0.6,
    space: SUNRISE_VIEW,
    eye: [
      [-30, 3, -4],
      [-42, 16, 26],
      [-120, 50, 48],
    ],
    look: [
      [40, 7, 0],
      [80, 0, 10],
      [160, -20, 22],
      [220, -36, 28],
    ],
    ease: 0.55,
    fov: [38, 52],
    shake: 0.08,
    clearance: 2.5,
    enter: { kind: 'blend', seconds: 1.6 },
  },
  {
    // Wave-skim. A tracking shot from a camera boat on her port side, a metre or
    // two over the crests, moving up the hull from the quarter to the bow and
    // leading her slightly. Long lens to stack the swell, handheld.
    name: 'wave-skim',
    duration: 9,
    throttle: 1.0,
    space: { kind: 'hull' },
    eye: [
      [-30, 1.2, -24],
      [-18, 1.2, -23],
      [-6, 1.2, -22],
      [4, 1.4, -22],
    ],
    look: [
      [-6, 6, 0],
      [0, 6.5, 0],
      [5, 6.5, 0],
      [10, 6, 0],
    ],
    ease: 0.25,
    fov: [42, 40],
    shake: 0.55,
    clearance: 1.3,
    enter: { kind: 'cut' },
  },
  {
    // Down through the surface beside the reef, along it at the fish's depth,
    // and back up through the waterline into the noon light. Slow at both
    // crossings, so the split frame — sky above the line, reef below — holds.
    name: 'reef-dive',
    duration: 12,
    throttle: 0.5,
    space: { kind: 'world' },
    eye: [
      [DIVE_REEF.x + 40, 5, DIVE_REEF.z + 22],
      [DIVE_REEF.x + 26, 0.3, DIVE_REEF.z + 14],
      [DIVE_REEF.x + 14, -10, DIVE_REEF.z + 6],
      [DIVE_REEF.x + 4, -13, DIVE_REEF.z + 1],
      [DIVE_REEF.x - 10, -12, DIVE_REEF.z - 3],
      [DIVE_REEF.x - 26, -5, DIVE_REEF.z - 6],
      [DIVE_REEF.x - 34, -0.3, DIVE_REEF.z - 9],
      [DIVE_REEF.x - 40, 3.5, DIVE_REEF.z - 11],
    ],
    look: [
      [DIVE_REEF.x + 8, -10, DIVE_REEF.z + 4],
      [DIVE_REEF.x - 2, -16, DIVE_REEF.z],
      [DIVE_REEF.x - 10, -16, DIVE_REEF.z - 3],
      [DIVE_REEF.x - 20, -14, DIVE_REEF.z - 6],
      [DIVE_REEF.x - 40, -5, DIVE_REEF.z - 12],
      [DIVE_REEF.x - 70, 2, DIVE_REEF.z - 20],
    ],
    ease: 0.35,
    fov: [50, 50],
    shake: 0.25,
    clearance: 0,
    underwater: true,
    enter: { kind: 'cut' },
  },
  {
    // The hero: a half orbit of the ship under sail, rising, from her starboard
    // bow round the stern to her port quarter, on a lens long enough that the
    // rig fills the frame.
    name: 'hero-orbit',
    duration: 12,
    throttle: 0.9,
    space: { kind: 'hull' },
    eye: orbit(46, 52, 8, 20, 50, 235, 7),
    look: [
      [6, RIG + 2, 0],
      [3, RIG + 3, 0],
      [0, RIG + 3, 0],
      [-2, RIG + 2, 0],
    ],
    ease: 0.4,
    fov: [42, 38],
    shake: 0.1,
    clearance: 3,
    enter: { kind: 'whip', direction: 1 },
  },
  {
    // The island, revealed over the swell. Low astern of the ship and looking
    // down into the wake, the camera cranes up and tilts with it, and the
    // island rises over the crests behind her.
    name: 'island-reveal',
    duration: 8,
    throttle: 0.8,
    space: { kind: 'toward', x: ISLAND_LOOK[0], z: ISLAND_LOOK[2] },
    eye: [
      [-46, 0.8, 12],
      [-56, 4, 10],
      [-70, 13, 7],
    ],
    look: [
      [-14, -1.5, 6],
      [160, 6, 2],
      [1300, 60, 0],
    ],
    ease: 0.6,
    fov: [44, 34],
    shake: 0.3,
    clearance: 1.6,
    enter: { kind: 'cut' },
  },
  {
    // Along the beach. Low over the shallows on the island's north-east face,
    // past the cove and its jetty, looking ahead and inland so the treeline
    // slides by against the hill behind it.
    name: 'beach-flyover',
    duration: 11,
    throttle: 0.6,
    space: { kind: 'world' },
    eye: [
      offIsland(0.3, 615, 44),
      offIsland(0.46, 622, 47),
      offIsland(0.62, 630, 51),
      offIsland(0.78, 640, 56),
    ],
    look: [
      offIsland(0.66, 470, 4),
      offIsland(0.82, 480, 6),
      offIsland(0.98, 480, 9),
      offIsland(1.14, 470, 12),
    ],
    ease: 0.3,
    fov: [46, 44],
    shake: 0.12,
    clearance: 6,
    enter: { kind: 'cut' },
  },
  {
    // Golden hour: on her quarter as she sails into the sunset, sails lit
    // through from behind, the sun just off her bow. A slow dolly-zoom — pulling back while the lens
    // closes in — keeps the hull the same size and grows the sun behind her.
    name: 'golden-hour',
    duration: 9,
    throttle: 0.5,
    space: SUNSET_VIEW,
    eye: [
      [-42, 5, 16],
      [-80, 6, 22],
      [-120, 7, 28],
    ],
    look: [
      [0, RIG + 1, 0],
      [10, RIG + 1.5, 0],
      [20, RIG + 2, 0],
    ],
    ease: 0.7,
    fov: [52, 20],
    shake: 0.06,
    clearance: 2.5,
    enter: { kind: 'whip', direction: -1 },
  },
  {
    // The squall, at dusk. Low on her lee quarter with the rain on the lens,
    // the horizon rolling, pushing in alongside as she drives through it.
    name: 'squall',
    duration: 11,
    throttle: 0.7,
    space: { kind: 'hull' },
    eye: [
      [-40, 2.5, 27],
      [-28, 3, 26],
      [-14, 3.5, 25],
    ],
    look: [
      [4, RIG, 0],
      [10, RIG, 0],
      [18, RIG - 1, 0],
    ],
    ease: 0.3,
    fov: [44, 40],
    roll: -3,
    sway: 6,
    shake: 0.9,
    clearance: 1.8,
    enter: { kind: 'cut' },
  },
  {
    // The night watch: moonlight, the rain clearing, a slow push in on the hull
    // that settles exactly where the dawn crane begins — and the sky pales
    // behind her as it does.
    name: 'night-watch',
    duration: 8,
    throttle: 0.4,
    space: SUNRISE_VIEW,
    eye: [
      [-64, 22, -58],
      [-42, 7, -32],
      [-32, 3.2, -24],
      [-30, 3, -4],
    ],
    look: [
      [0, 2, 0],
      [12, RIG - 2, 0],
      [24, RIG, 0],
      [40, 7, 0],
    ],
    ease: 0.55,
    fov: [44, 38],
    shake: 0.12,
    clearance: 2.5,
    enter: { kind: 'cut' },
  },
];

/**
 * Points on a horizontal arc about the hull, for orbit shots: radius and height
 * each run linearly from start to end, bearing in degrees from the bow toward
 * starboard.
 */
function orbit(
  r0: number,
  r1: number,
  h0: number,
  h1: number,
  fromDeg: number,
  toDeg: number,
  count: number,
): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < count; i++) {
    const f = i / (count - 1);
    const a = THREE.MathUtils.degToRad(fromDeg + (toDeg - fromDeg) * f);
    const r = r0 + (r1 - r0) * f;
    out.push([Math.cos(a) * r, h0 + (h1 - h0) * f, Math.sin(a) * r]);
  }
  return out;
}

// ------------------------------------------------------------ derived timeline

/** Seconds for one lap of the tour and one lap of the hull's circuit. */
export const CINEMATIC_LOOP_SECONDS = SHOTS.reduce((sum, shot) => sum + shot.duration, 0);

/** Absolute loop time each shot starts at. */
const SHOT_START = (() => {
  const starts = new Float64Array(SHOTS.length);
  let t = 0;
  for (let i = 0; i < SHOTS.length; i++) {
    starts[i] = t;
    t += SHOTS[i].duration;
  }
  return starts;
})();

/**
 * Nominal speed through the water for each shot, m/s: terminal speed under
 * quadratic drag goes as the square root of thrust.
 */
const SHOT_SPEED = SHOTS.map((shot) => HULL_TOP_SPEED * Math.sqrt(Math.max(0, shot.throttle)));

/** Distance the hull has run by the start of each shot, metres. */
const SHOT_START_ARC = (() => {
  const arcs = new Float64Array(SHOTS.length);
  let s = 0;
  for (let i = 0; i < SHOTS.length; i++) {
    arcs[i] = s;
    s += SHOT_SPEED[i] * SHOTS[i].duration;
  }
  return arcs;
})();

const TOTAL_ARC =
  SHOT_START_ARC[SHOTS.length - 1] + SHOT_SPEED[SHOTS.length - 1] * SHOTS[SHOTS.length - 1].duration;

/**
 * Radius of the hull's circuit, metres.
 *
 * Not a tuning knob: it is the radius at which one lap of arc is exactly one
 * revolution, which is why the circuit closes whatever the durations and
 * throttles are. It comes out near 112 m, so the lap's far point is ~224 m from
 * the origin — inside the shallow plateau's 320 m edge, over turquoise water.
 */
const TRACK_RADIUS = TOTAL_ARC / (2 * Math.PI);

/**
 * Rudder order for each shot, negative for port: yaw rate is `speed / radius`,
 * the controller delivers yaw rate proportional to rudder, and below the
 * reference speed a rudder buys proportionally less turn.
 */
const SHOT_RUDDER = SHOTS.map((_, i) => {
  const speed = SHOT_SPEED[i];
  const yawRate = speed / TRACK_RADIUS;
  const authority = Math.min(1, speed / HULL_RUDDER_REFERENCE_SPEED);
  const order =
    yawRate / (HULL_YAW_RATE_PER_RUDDER * HULL_YAW_EFFICIENCY * Math.max(1e-3, authority));
  return -Math.min(1, order);
});

/** Index of the shot containing `time`. `time` must already be wrapped. */
function shotAt(time: number): number {
  let i = SHOTS.length - 1;
  while (i > 0 && SHOT_START[i] > time) i--;
  return i;
}

function nominalArc(time: number): number {
  const t = wrapTime(time);
  const shot = shotAt(t);
  return SHOT_START_ARC[shot] + SHOT_SPEED[shot] * (t - SHOT_START[shot]);
}

/**
 * Where the hull is at `time` if it flies the authored orders exactly: a circle
 * of radius `TRACK_RADIUS` centred at `(0, -R)`, entered at the origin heading
 * +x and traversed to port. The rig only falls back on it when there is no live
 * hull; the plateau test reads it to check the circuit stays over the reef.
 */
export function nominalShipXZ(time: number, out: THREE.Vector2): THREE.Vector2 {
  const turned = nominalArc(time) / TRACK_RADIUS;
  return out.set(TRACK_RADIUS * Math.sin(turned), -TRACK_RADIUS * (1 - Math.cos(turned)));
}

/** Heading of the nominal hull, `atan2(forward.z, forward.x)`. */
function nominalHeading(time: number): number {
  return -nominalArc(time) / TRACK_RADIUS;
}

// --------------------------------------------------------------- the day

/**
 * The tour's day: hour of the day against loop time, as knots on a monotone
 * cubic (Fritsch–Carlson), unwrapped so the lap advances exactly one day.
 *
 * Hours are an angle to `sunFromClock` — 24 and 0 are the same sun — so a curve
 * that gains exactly 24 per lap, with matching slopes at the ends, is seamless.
 * Monotone so the sun never runs backwards; knotted per shot so each shot gets
 * the light it was composed for rather than whatever a sine happened to give it:
 * the sun just clearing the horizon behind the dawn crane, noon over the dive,
 * the sun a few degrees up and dead ahead of the ship in `golden-hour`, real dark
 * for the squall and the night watch, and the small hours hurried through while
 * the last shot pushes in, so the sky is already paling as it hands back to dawn.
 *
 * The rates are bounded by what the environment cube can follow: the fastest
 * stretch is the night, which costs least because the sun is below the horizon.
 */
const HOUR_KEYS: readonly (readonly [number, number])[] = [
  [0, 5.75],
  [10, 6.85],
  [19, 9.3],
  [25, 11.6],
  [31, 12.9],
  [43, 14.5],
  [51, 15.4],
  [62, 16.95],
  [66.5, 17.6],
  [71, 18.05],
  [77, 19.9],
  [82, 22.4],
  [86, 26.2],
  [90, 29.75],
];

const HOUR_SLOPES = (() => {
  // Cyclic Fritsch–Carlson: secant slopes with the wrap treated as one more span.
  const n = HOUR_KEYS.length;
  const secant = (i: number): number => {
    const [t0, h0] = HOUR_KEYS[i];
    const [t1, h1] = HOUR_KEYS[i + 1];
    return (h1 - h0) / (t1 - t0);
  };
  const slopes = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    // The first and last knots are the same instant on the loop: both take the
    // slope from the last span into the first.
    const before = i === 0 ? secant(n - 2) : secant(i - 1);
    const after = i === n - 1 ? secant(0) : secant(i);
    slopes[i] = before * after <= 0 ? 0 : (2 * before * after) / (before + after);
  }
  return slopes;
})();

/** Hour of the day at a loop time, 0..24. */
function tourHours(time: number): number {
  const t = wrapTime(time);
  let i = 0;
  while (i < HOUR_KEYS.length - 2 && HOUR_KEYS[i + 1][0] <= t) i++;
  const [t0, h0] = HOUR_KEYS[i];
  const [t1, h1] = HOUR_KEYS[i + 1];
  const span = t1 - t0;
  const s = (t - t0) / span;
  const hours = hermite(h0, HOUR_SLOPES[i] * span, h1, HOUR_SLOPES[i + 1] * span, s) % 24;
  return hours < 0 ? hours + 24 : hours;
}

/**
 * The squall: peak rain rate, centre and half-width of a raised cosine, which
 * reaches zero *with zero derivative* at both ends so the weather makes up and
 * clears without a corner. It starts to spit at the end of `golden-hour`, peaks
 * in `squall` and is clearing through the night watch. Sixteen seconds wet out
 * of ninety: rain has to be a change, so the lap is dry for most of itself.
 */
const RAIN_PEAK = 0.85;
const RAIN_CENTRE_SECONDS = 76;
const RAIN_HALF_WIDTH_SECONDS = 8;

function tourRain(time: number): number {
  let d = Math.abs(wrapTime(time) - RAIN_CENTRE_SECONDS);
  if (d > CINEMATIC_LOOP_SECONDS / 2) d = CINEMATIC_LOOP_SECONDS - d;
  if (d >= RAIN_HALF_WIDTH_SECONDS) return 0;
  return RAIN_PEAK * 0.5 * (1 + Math.cos((Math.PI * d) / RAIN_HALF_WIDTH_SECONDS));
}

// ------------------------------------------------------------------ the curves

/**
 * One shot's eye and look paths, prebuilt.
 *
 * Centripetal Catmull–Rom, sampled by arc length: centripetal so a tight cluster
 * of keys cannot make the curve loop or overshoot, by arc length so the shot's
 * ease is the only thing deciding the camera's speed — a raw spline parameter
 * runs fast on long spans and slow on short ones. The end tangents are kept so a
 * shot can be evaluated a little past either end, which is what a blend needs.
 */
interface ShotCurves {
  eye: THREE.CatmullRomCurve3;
  look: THREE.CatmullRomCurve3;
  eyeLength: number;
  lookLength: number;
  eyeStartTangent: THREE.Vector3;
  eyeEndTangent: THREE.Vector3;
  lookStartTangent: THREE.Vector3;
  lookEndTangent: THREE.Vector3;
}

function buildCurve(points: readonly V3[]): THREE.CatmullRomCurve3 {
  const curve = new THREE.CatmullRomCurve3(
    points.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    false,
    'centripetal',
  );
  curve.arcLengthDivisions = 2000;
  return curve;
}

const CURVES: readonly ShotCurves[] = SHOTS.map((shot) => {
  const eye = buildCurve(shot.eye);
  const look = buildCurve(shot.look);
  return {
    eye,
    look,
    eyeLength: eye.getLength(),
    lookLength: look.getLength(),
    eyeStartTangent: eye.getTangentAt(0),
    eyeEndTangent: eye.getTangentAt(1),
    lookStartTangent: look.getTangentAt(0),
    lookEndTangent: look.getTangentAt(1),
  };
});

/** Cubic Hermite on one value; tangents already scaled to the span. */
function hermite(p0: number, m0: number, p1: number, m1: number, s: number): number {
  const s2 = s * s;
  const s3 = s2 * s;
  return (
    (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1
  );
}

function smootherstep(s: number): number {
  const x = Math.min(1, Math.max(0, s));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

/**
 * The shot's eased clock: `s + a·(smootherstep(s) − s)`, continued linearly past
 * either end at its end slope `1 − a`, so a shot evaluated beyond its own length
 * keeps moving the way it was moving instead of stopping dead.
 */
function easeShot(s: number, a: number): number {
  if (s <= 0) return (1 - a) * s;
  if (s >= 1) return 1 + (1 - a) * (s - 1);
  return s + a * (smootherstep(s) - s);
}

/** A point on an arc-length curve, extrapolated along its end tangent past 0 or 1. */
function curvePoint(
  curve: THREE.CatmullRomCurve3,
  length: number,
  startTangent: THREE.Vector3,
  endTangent: THREE.Vector3,
  u: number,
  out: THREE.Vector3,
): THREE.Vector3 {
  if (length < 1e-6) return out.copy(curve.points[0]);
  if (u < 0) return curve.getPointAt(0, out).addScaledVector(startTangent, u * length);
  if (u > 1) return curve.getPointAt(1, out).addScaledVector(endTangent, (u - 1) * length);
  return curve.getPointAt(u, out);
}

/**
 * Handheld drift: a sum of incommensurate sines, so it never visibly repeats
 * inside a shot, is a pure function of the clock, and has no high-frequency
 * buzz — a real operator's hands drift and breathe, they do not vibrate.
 */
function handheld(t: number, seed: number): number {
  return (
    0.55 * Math.sin(t * 1.13 + seed) +
    0.3 * Math.sin(t * 2.71 + seed * 2.3) +
    0.15 * Math.sin(t * 5.37 + seed * 3.7)
  );
}

// -------------------------------------------------------------------- the mode

/** Camera pose the tour wants this frame. */
export interface CinematicPose {
  position: THREE.Vector3;
  target: THREE.Vector3;
  /** Vertical field of view, degrees. Optional so a caller may ignore the lens. */
  fov?: number;
  /** Roll about the view axis, radians. */
  roll?: number;
}

/** What the tour is asking of the hull this frame. Same units as `setInput`. */
export interface CinematicShipInput {
  /** −1 (full astern) … 1 (full ahead). */
  throttle: number;
  /** −1 (hard to port) … 1 (hard to starboard). */
  rudder: number;
}

/** The live hull, as the chase camera already receives it. */
export interface CinematicHull {
  position: { readonly x: number; readonly z: number };
  /** `atan2(forward.z, forward.x)`. */
  heading: number;
}

/** What the rig asks of the world for clearance. Both optional. */
export interface CinematicWorld {
  /** Live wave height at (x, z). Defaults to flat water at 0. */
  surfaceHeight?: (x: number, z: number) => number;
  /** Terrain height at (x, z). Defaults to `seafloorHeight`. */
  groundHeight?: (x: number, z: number) => number;
}

/** Shot names, start times and durations, in order. */
export const CINEMATIC_BEATS: ReadonlyArray<{
  readonly name: string;
  readonly start: number;
  readonly duration: number;
}> = SHOTS.map((shot, i) => ({ name: shot.name, start: SHOT_START[i], duration: shot.duration }));

/**
 * Everything about the world, other than the camera and the hull, that the tour
 * owns while it is running. Every field is a pure, periodic function of the loop
 * clock, so `resetClock(t)` reproduces lighting and weather exactly.
 */
export interface CinematicEnvironment {
  /** Hour of the day, 0..24. */
  hours: number;
  /** Rain rate, 0..1. */
  rain: number;
  /** Cloud coverage, 0..1. */
  cloudCoverage: number;
  /** Multiplier on the preset's volumetric fog density. */
  fogDensity: number;
  /**
   * Which weather to draw. `Weather` draws nothing while its kind is `'clear'`,
   * and main's `raining` scalar — lens beads, surface stipple, foam, hull wetting
   * — is gated on the same test, so a squall that set only the intensity would
   * produce no weather anywhere.
   */
  weatherKind: 'clear' | 'rain';
}

/** Per-shot pose before transitions, in world space. Scratch, reused. */
interface ShotSample {
  eye: THREE.Vector3;
  look: THREE.Vector3;
  fov: number;
  roll: number;
  shake: number;
  clearance: number;
  underwater: boolean;
}

function makeSample(): ShotSample {
  return {
    eye: new THREE.Vector3(),
    look: new THREE.Vector3(),
    fov: 50,
    roll: 0,
    shake: 0,
    clearance: 0,
    underwater: false,
  };
}

export class CinematicDirector {
  private enabled = false;
  private clock = 0;
  private reducedMotion = false;

  private readonly surfaceHeight: (x: number, z: number) => number;
  private readonly groundHeight: (x: number, z: number) => number;

  private readonly input: CinematicShipInput = { throttle: 0, rudder: 0 };
  private readonly env: CinematicEnvironment = {
    hours: 12,
    rain: 0,
    cloudCoverage: 0.3,
    fogDensity: 1,
    weatherKind: 'clear',
  };

  // Scratch — the frame path must not allocate.
  private readonly sampleA = makeSample();
  private readonly sampleB = makeSample();
  private readonly hullXZ = new THREE.Vector2();
  private readonly local = new THREE.Vector3();
  private readonly dir = new THREE.Vector3();
  private hullX = 0;
  private hullZ = 0;
  private hullHeading = 0;

  constructor(world: CinematicWorld = {}) {
    this.surfaceHeight = world.surfaceHeight ?? (() => 0);
    this.groundHeight = world.groundHeight ?? seafloorHeight;
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  /** Position on the loop, seconds. Always in `[0, CINEMATIC_LOOP_SECONDS)`. */
  get time(): number {
    return this.clock;
  }

  /** Name of the shot playing at the current clock. */
  get beatName(): string {
    return SHOTS[shotAt(this.clock)].name;
  }

  /**
   * Honour `prefers-reduced-motion`: handheld drift drops to a quarter, the
   * storm's roll and the whip pans go, and every lens holds one focal length —
   * no zooms, no dolly-zoom. The cuts and the camera paths are unchanged; a
   * moving camera is the content, a lurching one is not.
   */
  setReducedMotion(on: boolean): void {
    this.reducedMotion = on;
  }

  /**
   * The world the tour wants at `time` (default: now), other than the camera
   * and the hull. Pure function of the loop clock.
   */
  environment(time: number = this.clock): Readonly<CinematicEnvironment> {
    const t = wrapTime(time);
    const rain = tourRain(t);
    const env = this.env;
    env.hours = tourHours(t);
    env.rain = rain;
    // Cloud builds with the rain and does not clear the instant it stops.
    env.cloudCoverage = 0.3 + rain * 0.62;
    env.fogDensity = 1 + rain * 1.7;
    env.weatherKind = rain > 0.001 ? 'rain' : 'clear';
    return env;
  }

  /**
   * Starts or stops the tour. Enabling rewinds to the opening shot; disabling
   * zeroes the ship input on the spot, so a stale order cannot be picked up the
   * next time the hull is put under manual control.
   */
  setEnabled(on: boolean): void {
    if (this.enabled === on) return;
    this.enabled = on;
    this.input.throttle = 0;
    this.input.rudder = 0;
    if (on) this.clock = 0;
  }

  /** Jumps to `time` on the loop. The next `update(0, …)` reports exactly that time's pose. */
  resetClock(time = 0): void {
    this.clock = wrapTime(time);
  }

  /**
   * Advances the clock, then samples it. Advance-then-sample, so `resetClock(t)`
   * followed by `update(0, …)` reports the pose at exactly `t`.
   *
   * @param hull The live hull, or null for the nominal one.
   */
  update(dt: number, out: CinematicPose, hull: CinematicHull | null = null): CinematicShipInput {
    if (!this.enabled) {
      this.input.throttle = 0;
      this.input.rudder = 0;
      return this.input;
    }

    this.clock = wrapTime(this.clock + (Number.isFinite(dt) ? dt : 0));
    const t = this.clock;

    if (hull && Number.isFinite(hull.position.x) && Number.isFinite(hull.heading)) {
      this.hullX = hull.position.x;
      this.hullZ = hull.position.z;
      this.hullHeading = hull.heading;
    } else {
      nominalShipXZ(t, this.hullXZ);
      this.hullX = this.hullXZ.x;
      this.hullZ = this.hullXZ.y;
      this.hullHeading = nominalHeading(t);
    }

    const index = shotAt(t);
    const shot = SHOTS[index];
    const local = t - SHOT_START[index];
    const pose = this.sampleShot(index, local, this.sampleA);

    // Entering by blend: the previous shot runs on past its end, and the two
    // poses are mixed across the window.
    if (shot.enter.kind === 'blend' && local < shot.enter.seconds) {
      const previous = (index - 1 + SHOTS.length) % SHOTS.length;
      const from = this.sampleShot(previous, SHOTS[previous].duration + local, this.sampleB);
      const w = smootherstep(local / shot.enter.seconds);
      pose.eye.lerpVectors(from.eye, pose.eye, w);
      pose.look.lerpVectors(from.look, pose.look, w);
      pose.fov += (from.fov - pose.fov) * (1 - w);
      pose.roll += (from.roll - pose.roll) * (1 - w);
      pose.shake += (from.shake - pose.shake) * (1 - w);
      pose.clearance = Math.max(pose.clearance, from.clearance);
    }

    // Whip pans either side of a whip cut: accelerating into it, decelerating
    // out of it, a yaw of the view about the eye.
    let whip = 0;
    if (!this.reducedMotion) {
      if (shot.enter.kind === 'whip' && local < WHIP_SECONDS) {
        const x = 1 - local / WHIP_SECONDS;
        whip = -shot.enter.direction * WHIP_DEGREES * x * x;
      }
      const next = SHOTS[(index + 1) % SHOTS.length];
      const remaining = shot.duration - local;
      if (next.enter.kind === 'whip' && remaining < WHIP_SECONDS) {
        const x = 1 - remaining / WHIP_SECONDS;
        whip = next.enter.direction * WHIP_DEGREES * x * x;
      }
    }

    // Handheld: yaw and pitch drift of the view, a little roll, a few
    // centimetres of float on the eye.
    const motion = this.reducedMotion ? 0.25 : 1;
    const shake = pose.shake * motion;
    const yaw = THREE.MathUtils.degToRad(whip + shake * handheld(t, 0.7));
    const pitch = THREE.MathUtils.degToRad(shake * 0.6 * handheld(t * 1.07, 2.9));
    let roll = pose.roll + THREE.MathUtils.degToRad(shake * 0.5 * handheld(t * 0.93, 5.1));
    if (this.reducedMotion) roll *= 0.25;
    pose.eye.x += shake * 0.12 * handheld(t * 0.8, 1.3);
    pose.eye.y += shake * 0.1 * handheld(t * 0.9, 4.2);
    pose.eye.z += shake * 0.12 * handheld(t * 0.85, 6.6);

    this.applyClearance(pose);

    // Re-aim with the drift and the whip applied, at the look's own distance.
    const dir = this.dir.subVectors(pose.look, pose.eye);
    const reach = Math.max(1e-3, dir.length());
    let azimuth = Math.atan2(dir.z, dir.x) + yaw;
    const elevation = THREE.MathUtils.clamp(
      Math.asin(THREE.MathUtils.clamp(dir.y / reach, -1, 1)) + pitch,
      -1.45,
      1.45,
    );
    if (!Number.isFinite(azimuth)) azimuth = 0;
    const c = Math.cos(elevation);
    out.position.copy(pose.eye);
    out.target.set(
      pose.eye.x + Math.cos(azimuth) * c * reach,
      pose.eye.y + Math.sin(elevation) * reach,
      pose.eye.z + Math.sin(azimuth) * c * reach,
    );
    out.fov = pose.fov;
    out.roll = roll;

    this.input.throttle = shot.throttle;
    this.input.rudder = SHOT_RUDDER[index];
    return this.input;
  }

  /**
   * One shot's pose at `local` seconds into it, in world space, before
   * transitions, noise and clearance. `local` may run past either end.
   */
  private sampleShot(index: number, local: number, out: ShotSample): ShotSample {
    const shot = SHOTS[index];
    const curves = CURVES[index];
    const s = local / shot.duration;
    const u = easeShot(s, shot.ease);

    curvePoint(curves.eye, curves.eyeLength, curves.eyeStartTangent, curves.eyeEndTangent, u, this.local);
    this.toWorld(shot.space, this.local, out.eye);
    curvePoint(curves.look, curves.lookLength, curves.lookStartTangent, curves.lookEndTangent, u, this.local);
    this.toWorld(shot.space, this.local, out.look);

    const [fov0, fov1] = shot.fov;
    out.fov = this.reducedMotion ? (fov0 + fov1) / 2 : fov0 + (fov1 - fov0) * smootherstep(s);
    const sway = shot.sway ?? 0;
    // The storm's heel: a slow roll on a seven-second period, about the same as
    // the swell's, so the horizon leans with the hull rather than against it.
    out.roll = THREE.MathUtils.degToRad(
      (shot.roll ?? 0) + (this.reducedMotion ? 0 : sway * Math.sin((local / 7) * Math.PI * 2)),
    );
    out.shake = shot.shake;
    out.clearance = shot.clearance;
    out.underwater = shot.underwater === true;
    return out;
  }

  private toWorld(space: Space, p: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    if (space.kind === 'world') return out.copy(p);
    let fx: number;
    let fz: number;
    if (space.kind === 'hull') {
      fx = Math.cos(this.hullHeading);
      fz = Math.sin(this.hullHeading);
    } else if (space.kind === 'bearing') {
      const n = Math.hypot(space.x, space.z) || 1;
      fx = space.x / n;
      fz = space.z / n;
    } else {
      const dx = space.x - this.hullX;
      const dz = space.z - this.hullZ;
      const n = Math.hypot(dx, dz) || 1;
      fx = dx / n;
      fz = dz / n;
    }
    // Starboard is forward × up: (−fz, 0, fx).
    return out.set(this.hullX + p.x * fx - p.z * fz, p.y, this.hullZ + p.x * fz + p.z * fx);
  }

  /**
   * Holds the eye above the crests, the terrain and the canopy, and outside the
   * hull. Every term is a softplus, so a correction is a smooth curve in the
   * camera's motion; well clear of everything it is the identity to within
   * millimetres.
   */
  private applyClearance(pose: ShotSample): void {
    const eye = pose.eye;
    let floor = -Infinity;

    if (!pose.underwater) {
      // Soft maximum of the sea over a few metres around the lens, so a crest
      // about to pass under it lifts the camera before it arrives.
      const r = 3.5;
      const k = 0.35;
      const h0 = this.surfaceHeight(eye.x, eye.z);
      const h1 = this.surfaceHeight(eye.x + r, eye.z);
      const h2 = this.surfaceHeight(eye.x - r, eye.z);
      const h3 = this.surfaceHeight(eye.x, eye.z + r);
      const h4 = this.surfaceHeight(eye.x, eye.z - r);
      const m = Math.max(h0, h1, h2, h3, h4);
      const sum =
        Math.exp((h0 - m) / k) +
        Math.exp((h1 - m) / k) +
        Math.exp((h2 - m) / k) +
        Math.exp((h3 - m) / k) +
        Math.exp((h4 - m) / k);
      floor = m + k * Math.log(sum) + pose.clearance;
    }

    // Terrain, with room for the canopy wherever the ground is high enough to
    // carry it (the trees start a few metres above the beach).
    const ground = this.groundHeight(eye.x, eye.z);
    const canopy = 2.5 + 24 * smoothstepRange(0.5, 6, ground);
    floor = softMax(floor, ground + canopy, 1);

    if (Number.isFinite(floor)) eye.y = floor + softplus(eye.y - floor, 0.5);

    // The hull: an ellipse in the hull's frame, out to the masthead.
    const fx = Math.cos(this.hullHeading);
    const fz = Math.sin(this.hullHeading);
    const dx = eye.x - this.hullX;
    const dz = eye.z - this.hullZ;
    const along = dx * fx + dz * fz;
    const across = -dx * fz + dz * fx;
    const q = Math.hypot(along / HULL_CLEAR_HALF_LENGTH, across / HULL_CLEAR_HALF_BEAM);
    const below = smoothstepRange(HULL_CLEAR_TOP, HULL_CLEAR_TOP - 6, eye.y);
    if (below > 0 && q < 1.4 && q > 1e-4) {
      const pushed = 1 + softplus(q - 1, 0.08);
      const scale = 1 + (pushed / q - 1) * below;
      eye.x = this.hullX + dx * scale;
      eye.z = this.hullZ + dz * scale;
    }
  }
}

/** `k·ln(1 + e^(x/k))`: ≈ x well above 0, ≈ 0 well below, smooth through it. */
function softplus(x: number, k: number): number {
  const z = x / k;
  return z > 30 ? x : k * Math.log1p(Math.exp(z));
}

function softMax(a: number, b: number, k: number): number {
  if (!Number.isFinite(a)) return b;
  return Math.max(a, b) + softplus(-Math.abs(a - b), k);
}

function smoothstepRange(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Folds any time onto the loop, negatives included. */
function wrapTime(time: number): number {
  if (!Number.isFinite(time)) return 0;
  const wrapped = time % CINEMATIC_LOOP_SECONDS;
  return wrapped < 0 ? wrapped + CINEMATIC_LOOP_SECONDS : wrapped;
}
