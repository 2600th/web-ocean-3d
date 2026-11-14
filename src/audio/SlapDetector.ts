/**
 * Turns hull probes into wave slaps.
 *
 * A slap is water arriving at the planking *fast*: the surface climbing past a
 * point on the hull side, relative to the hull, quicker than the hull can
 * yield. So each probe is a point a little above the static waterline, and the
 * signal is its immersion (water height minus probe height) and how fast that
 * is rising. Crossing from dry to wet above `MIN_RATE` is a slap whose strength
 * is that rate; a crest that is already over the probe but still climbing hard
 * slaps again once the cooldown allows. Water falling away is silent — it
 * drains, it does not hit.
 *
 * Pure arithmetic on numbers the caller supplies, so it neither knows about
 * three.js nor allocates; `main.ts` samples the sea at the probe positions.
 */

/** Rising rate, m/s, below which water arriving at the hull is a lap, not a slap. */
const MIN_RATE = 0.55;
/** Rate at which a slap is at full strength. */
const FULL_RATE = 3.2;
/** A crest already over the probe slaps again only if it is climbing this fast. */
const WET_RATE = 1.6;
/** Seconds between slaps at one probe. */
const COOLDOWN = 0.28;

export class SlapDetector {
  private readonly previous: Float64Array;
  private readonly cooldown: Float64Array;
  private readonly primed: Uint8Array;

  constructor(readonly probes: number) {
    this.previous = new Float64Array(probes);
    this.cooldown = new Float64Array(probes);
    this.primed = new Uint8Array(probes);
  }

  /**
   * @param index     Probe index.
   * @param immersion Water surface height minus the probe's height, metres:
   *                  positive when the probe is under water.
   * @returns Slap strength 0..1 this frame, 0 for none.
   */
  update(index: number, immersion: number, dt: number): number {
    if (!(dt > 0) || !Number.isFinite(immersion) || index < 0 || index >= this.probes) return 0;
    const prev = this.previous[index];
    this.previous[index] = immersion;
    this.cooldown[index] = Math.max(0, this.cooldown[index] - dt);
    if (this.primed[index] === 0) {
      this.primed[index] = 1;
      return 0;
    }
    // Clamped so a teleport (reset, camera cut) cannot register as a wall of water.
    const rate = Math.min(8, (immersion - prev) / dt);
    if (this.cooldown[index] > 0 || rate < MIN_RATE) return 0;
    const arriving = prev < 0 && immersion >= 0;
    if (!arriving && !(immersion > 0 && rate > WET_RATE)) return 0;
    this.cooldown[index] = COOLDOWN;
    const strength = (rate - MIN_RATE) / (FULL_RATE - MIN_RATE);
    return Math.min(1, Math.max(0.08, strength));
  }

  reset(): void {
    this.primed.fill(0);
    this.cooldown.fill(0);
  }
}
