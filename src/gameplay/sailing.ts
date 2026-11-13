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
 * What it leaves out — apparent wind, sail trim, leeway as a function of keel
 * area — is left out on purpose: the player sets *how much* canvas and *where to
 * point*, and the ship trims itself.
 */

/** Metres per second to knots. */
export const KNOTS_PER_MS = 1.943844;

/**
 * Drive coefficient by angle off the wind, degrees → fraction of full drive.
 * Piecewise linear between these marks; symmetric port and starboard.
 */
const POLAR: readonly (readonly [number, number])[] = [
  [0, -0.06],
  [30, -0.02],
  [38, 0.2],
  [50, 0.58],
  [70, 0.86],
  [95, 1],
  [125, 1],
  [150, 0.88],
  [180, 0.74],
];

/** Below this angle off the wind the ship is in irons. */
export const IRONS_DEG = 35;

/** Fraction of full drive by angle off the wind, degrees (0 = head to wind). */
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
 * How much the wind's strength lets the canvas pull, 0.3..1.
 *
 * Full drive from about a fresh breeze (9 m/s, the default sea) upward, so a
 * storm does not make the ship faster than the drag was tuned for. A floor,
 * because a near-calm preset that leaves the player becalmed is realistic and
 * not fun: the sunset sea at 2.5 m/s still ghosts along at a third.
 */
export function windDriveFactor(windSpeed: number): number {
  if (!(windSpeed > 0)) return 0.3;
  return Math.max(0.3, Math.min(1, Math.pow(windSpeed / 9, 0.8)));
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
