/**
 * Points of sail, as pure functions.
 *
 * This is a game's model of a square-rigger, not a velocity-prediction program.
 * What it keeps from the real thing is the part a player can feel and learn:
 *
 *  - **There is a no-go zone.** Within about 35 degrees of the wind the canvas
 *    luffs and the drive collapses — slightly *negative*, because a sail taken
 *    aback pushes the ship astern. That is "in irons", and the way out is to
 *    bear away, which needs way on for the rudder to bite: the same foil rule
 *    `ShipController` already enforces.
 *  - **Reaching is fastest.** A square rig is happiest with the wind on the
 *    quarter; beam and broad reaches top the curve.
 *  - **Running is good, not best.** Dead downwind the after sails blanket the
 *    fore and the drive falls back to about three quarters.
 *
 * Two things a player can feel sit on top, both tuned to be potent rather
 * than punishing:
 *
 *  - **Apparent wind.** The canvas feels the wind the ship makes as well as the
 *    true one: under way on a reach it draws forward and freshens, running it
 *    falls light. The drive is read off the polar at the apparent angle.
 *  - **Trim.** The sheets are the player's: hauled in for close-hauled, eased
 *    out for a run. Good trim adds a little, bad trim costs some, and a floor
 *    keeps her moving on every heading outside the no-go zone.
 *
 * Wind strength matters all the way up: past a fresh breeze the canvas is
 * overpressed, heels her hard and spills drive, so in a gale half sail is the
 * faster call. Leeway as a function of keel area is still left out on purpose.
 */

/** Metres per second to knots. */
export const KNOTS_PER_MS = 1.943844;

/**
 * Drive coefficient by *apparent* angle off the wind, degrees → fraction of
 * full drive. Piecewise linear between these marks; symmetric port and
 * starboard. Apparent wind draws forward of the true wind once she is moving,
 * so these marks sit some degrees forward of where a true-wind polar would put
 * them; a beam reach in true wind is about 60 degrees apparent at speed.
 */
const POLAR: readonly (readonly [number, number])[] = [
  [0, -0.06],
  [25, -0.02],
  [32, 0.25],
  [42, 0.62],
  [60, 0.9],
  [80, 1],
  [110, 1],
  [150, 0.9],
  [180, 0.8],
];

/**
 * Below this *true* angle off the wind the ship is in irons: the no-go zone the
 * player is shown and the floor drive fades out across.
 */
export const IRONS_DEG = 35;

/** Fraction of full drive by apparent angle off the wind, degrees (0 = head to wind). */
export function polarDrive(angleOffWindDeg: number): number {
  const a = Math.min(180, Math.max(0, Math.abs(angleOffWindDeg)));
  for (let i = 1; i < POLAR.length; i++) {
    const [a1, d1] = POLAR[i];
    if (a <= a1) {
      const [a0, d0] = POLAR[i - 1];
      const t = (a - a0) / (a1 - a0);
      return d0 + (d1 - d0) * t;
    }
  }
  return POLAR[POLAR.length - 1][1];
}

/**
 * How much the true wind's strength lets the canvas pull, 0.3..1, capped at a
 * fresh breeze. No longer what drives the ship: that is `sailPressure` on the
 * apparent wind, which keeps rising past 9 m/s and is then overpressed. Kept
 * for callers that want a bounded figure.
 */
export function windDriveFactor(windSpeed: number): number {
  if (!(windSpeed > 0)) return 0.3;
  return Math.max(0.3, Math.min(1, Math.pow(windSpeed / 9, 0.8)));
}

/**
 * Apparent wind speed, m/s, at which full canvas is at full pressure. Above it
 * the sails are overpressed; see `overpress`.
 */
export const APPARENT_REF = 10;
/** Calm floor of `sailPressure`: a near-calm preset still ghosts along. */
const CALM_FLOOR = 0.3;
/** How quickly drive falls past full pressure. */
const OVERPRESS_K = 0.8;
/** Least drive an overpressed sail keeps, before the polar and trim. */
const OVERPRESS_FLOOR = 0.4;

/**
 * How hard the canvas is pressed: sail fill times apparent wind over
 * `APPARENT_REF`, with a floor while any true wind blows. Uncapped; this is
 * also what heels her.
 */
export function sailPressure(apparentSpeed: number, trueSpeed: number, sailFill: number): number {
  const raw = Math.max(0, apparentSpeed) / APPARENT_REF;
  return sailFill * Math.max(trueSpeed > 0 ? CALM_FLOOR : 0, raw);
}

/**
 * Drive from canvas pressure. One to one up to full pressure, then a gentle
 * peak (1.08 at 1.3) and a fall: 0.97 at 1.6, the 0.4 floor by about 2.3. An
 * overpressed square rig heels, buries her lee bow and spills wind, so full
 * sail in a gale is slower than half.
 */
export function overpress(pressure: number): number {
  if (pressure <= 1) return pressure;
  const x = pressure - 1;
  return Math.max(OVERPRESS_FLOOR, 1 + 0.5 * x - OVERPRESS_K * x * x);
}

/**
 * The sheet that suits an apparent wind angle, 0 (hauled hard in) to 1 (eased
 * right out): in for close-hauled, out for a run.
 */
export function idealTrim(apparentAngleDeg: number): number {
  return Math.min(1, Math.max(0, (Math.abs(apparentAngleDeg) - 30) / 120));
}

const TRIM_BEST = 1.08;
const TRIM_WORST = 0.72;

/**
 * Drive multiplier for a sheet setting: 1.08 trimmed right, 1.05 a quarter
 * out, 0.96 half out, never below 0.72. Untouched sheets (0.5) cost five to ten
 * percent, so trimming rewards skill without punishing a player who ignores it.
 */
export function trimFactor(trim: number, apparentAngleDeg: number): number {
  const err = trim - idealTrim(apparentAngleDeg);
  return Math.max(TRIM_WORST, TRIM_BEST - 0.5 * err * err);
}

/**
 * Least drive, as a fraction of full, that set canvas gives on any heading
 * outside the no-go zone: about six knots at full sail on flat water. The
 * caller scales it by sail fill and fades it out across the edge of the no-go
 * zone, so head to wind with no way on is still a stall.
 */
export const FLOOR_DRIVE = 0.16;

/** Apparent wind, as the canvas feels it. */
export interface ApparentWind {
  /** Speed, m/s. */
  speed: number;
  /** Angle between the bow and where it comes *from*, degrees, 0..180. */
  angleDeg: number;
  /** +1 coming over the starboard side, -1 over port, 0 dead ahead or astern. */
  side: number;
  /** Direction it blows toward, radians, `atan2(z, x)`. */
  towardRad: number;
}

/**
 * True wind minus the ship's velocity: the wind on the canvas. Pure; the sails
 * and anything else that wants to know which way the canvas is blown can call it.
 *
 * @param windTowardRad Direction the true wind blows toward, `atan2(z, x)`.
 * @param windSpeed     True wind speed, m/s.
 * @param vx            Ship velocity over the ground, world x, m/s.
 * @param vz            Ship velocity over the ground, world z, m/s.
 * @param headingRad    Ship heading, `atan2(forward.z, forward.x)`.
 */
export function apparentWind(
  windTowardRad: number,
  windSpeed: number,
  vx: number,
  vz: number,
  headingRad: number,
  out: ApparentWind = { speed: 0, angleDeg: 0, side: 0, towardRad: 0 },
): ApparentWind {
  const ax = Math.cos(windTowardRad) * windSpeed - vx;
  const az = Math.sin(windTowardRad) * windSpeed - vz;
  const speed = Math.hypot(ax, az);
  out.speed = speed;
  if (speed < 1e-4) {
    out.angleDeg = 0;
    out.side = 0;
    out.towardRad = windTowardRad;
    return out;
  }
  out.towardRad = Math.atan2(az, ax);
  // Where it comes from, relative to the bow. A positive angle is toward
  // increasing heading, which is the starboard side in this convention.
  let d = Math.atan2(-az, -ax) - headingRad;
  d = Math.atan2(Math.sin(d), Math.cos(d));
  out.angleDeg = Math.abs(d) * (180 / Math.PI);
  out.side = out.angleDeg < 0.01 || out.angleDeg > 179.99 ? 0 : Math.sign(d);
  return out;
}

/** Named point of sail, for messages and tests. */
export type PointOfSail =
  | 'in irons'
  | 'close-hauled'
  | 'close reach'
  | 'beam reach'
  | 'broad reach'
  | 'running';

export function pointOfSail(angleOffWindDeg: number): PointOfSail {
  const a = Math.abs(angleOffWindDeg);
  if (a < IRONS_DEG) return 'in irons';
  if (a < 60) return 'close-hauled';
  if (a < 80) return 'close reach';
  if (a < 105) return 'beam reach';
  if (a < 150) return 'broad reach';
  return 'running';
}

/**
 * Angle between the bow and the direction the wind comes *from*, degrees, 0..180.
 *
 * @param headingRad    Ship heading as `atan2(forward.z, forward.x)`.
 * @param windTowardRad Direction the wind blows *toward*, in the same
 *                      convention — `SpectrumParams.windDirection`.
 */
export function angleOffWind(headingRad: number, windTowardRad: number): number {
  // The wind comes from the opposite of where it blows; the bow is head to wind
  // when it points there.
  const from = windTowardRad + Math.PI;
  let d = headingRad - from;
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return Math.abs(d) * (180 / Math.PI);
}

/**
 * Compass bearing in degrees (0 = north = -Z, clockwise, 90 = east = +X) of a
 * horizontal world-space direction.
 */
export function compassFromVector(dx: number, dz: number): number {
  const deg = Math.atan2(dx, -dz) * (180 / Math.PI);
  return (deg + 360) % 360;
}

/** Compass bearing of a heading given as `atan2(forward.z, forward.x)`. */
export function compassFromHeading(headingRad: number): number {
  return compassFromVector(Math.cos(headingRad), Math.sin(headingRad));
}

/** Compass bearing the wind blows *from*, given the direction it blows toward. */
export function windFromCompass(windTowardRad: number): number {
  return compassFromVector(-Math.cos(windTowardRad), -Math.sin(windTowardRad));
}

/** Moves `current` toward `target` by at most `step`. */
export function approach(current: number, target: number, step: number): number {
  const delta = target - current;
  if (Math.abs(delta) <= step) return target;
  return current + Math.sign(delta) * step;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
