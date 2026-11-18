/**
 * Loudness and priority: the mix as a window that follows the loudest sound.
 *
 * The method is the one DICE described for Frostbite as HDR audio ("Adaptive
 * mixing in Frostbite", GDC 2009; linked in `docs/AAA_ROADMAP.md`). Every
 * sound has a *logical* loudness, a number in dB that says how loud it is in
 * the world before any mixing: a close thunder crack is far above a creak, a
 * gale far above a calm. The loudest sound present sets the top of a window;
 * everything is played relative to that top, so when the strike lands the sea,
 * the ship and the band all step back, and when it has rolled away the window
 * falls and they come forward again. Sounds further below the top than the
 * window is tall are not worth a voice (`audible`).
 *
 * Here the logical scale is "LUFS at master volume 1 before this stage",
 * estimated, not measured: each noise bed reports what its level and band
 * imply (`NoiseBed.loudnessDb`), the score a figure per arrangement measured
 * by the bench, and an event its peak and how fast it decays. Estimating is
 * the point: it is deterministic, costs a few multiplies a frame, and knows
 * about a strike 0.25 s before it is heard, which a meter on the output never
 * could.
 *
 * **The map.** A full window would play everything with its top at the same
 * level, which flattens a storm into a calm. So the top is mapped with a slope
 * under 1: `out = TARGET + slope * (top - REF)`. At the default slope a storm
 * eleven dB louder than a calm in the world comes out about three and a half
 * dB louder, and every state lands near -18 LUFS. The reduced range (laptop
 * speakers) flattens it further and lets events rise less over the bed.
 *
 * **Timing.** Attack is immediate and scheduled on the audio clock at the
 * event's own start, so the leading edge of a crack is already ducking the
 * rest; release is a fixed rate in dB per second, so the world comes back up
 * under the tail of the roll rather than snapping.
 *
 * Allocates nothing per frame; events live in a fixed ring.
 */

/** Where an average scene sits on the logical scale, and where it is played. */
const REF_DB = -24;
const TARGET_DB = -18.8;
/** How much of the world's own range survives: 0 none, 1 all of it. */
const SLOPE_FULL = 0.34;
const SLOPE_REDUCED = 0.2;
/** The reduced range: its firmer compressor adds makeup, which this takes back. */
const REDUCED_LIFT_DB = -1.5;
/**
 * How much of an event's rise over the beds is kept: the rest is taken off
 * everything, the event with it. At 0.6 a strike 10 dB over the sea plays 6 dB
 * over where the sea was, and the sea steps back 4 dB under it. The slope above
 * is for the beds alone; applied to events too, a strike was only 2 dB over a
 * storm (bench, recordings run). Less is kept in the reduced range.
 */
const EVENT_KEEP_FULL = 0.6;
const EVENT_KEEP_REDUCED = 0.35;
/** Limits on the gain this stage applies, dB. */
const MAX_BOOST_DB = 10;
const MAX_CUT_DB = -14;
/** The window's floor: silence is not a reason to amplify the noise floor. */
const FLOOR_DB = -42;
/** Release of the window top, dB per second. */
const RELEASE_DB_PER_S = 5;
/** How tall the window is: anything this far under the top is not given a voice. */
const WINDOW_HEIGHT_DB = 34;
/** Gain smoothing, seconds: the attack's time constant on the audio clock. */
const ATTACK_TAU = 0.025;
const FOLLOW_TAU = 0.12;
/** Writes smaller than this, dB, are dropped. */
const EPSILON_DB = 0.15;

/** Event ring: start, peak dB, hold end, decay dB/s. */
const EVENT_SLOTS = 12;
const EVENT_STRIDE = 4;

export class HdrMix {
  /** The stage itself: the world and the score pass through it. */
  readonly node: GainNode;

  private reduced = false;
  /** Top of the window, dB on the logical scale, after release. */
  private top = FLOOR_DB;
  /**
   * This frame's continuous sources, power-summed: the sea, the wind, the rain
   * and the score are one bed together, and a bed of two equal sounds is 3 dB
   * louder than either. Events stand against that sum, loudest first.
   */
  private loudest = FLOOR_DB;
  private bedPower = 0;
  /** How far the loudest event stands over the beds, dB, after release. */
  private excess = 0;
  /** Gain last written, dB; NaN before the first. */
  private written = Number.NaN;
  private readonly events = new Float64Array(EVENT_SLOTS * EVENT_STRIDE);
  private nextSlot = 0;

  constructor(ctx: BaseAudioContext) {
    this.node = ctx.createGain();
    this.node.gain.value = 1;
    for (let i = 0; i < EVENT_SLOTS; i++) this.events[i * EVENT_STRIDE + 1] = -Infinity;
  }

  /** Laptop speakers: less range between a calm and a storm, and between a bed and a crack. */
  setReduced(reduced: boolean): void {
    this.reduced = reduced;
  }

  isReduced(): boolean {
    return this.reduced;
  }

  /** The top of the window, logical dB. */
  get windowTop(): number {
    return this.top;
  }

  /** The gain this stage is applying, dB. */
  get gainDb(): number {
    return Number.isFinite(this.written) ? this.written : 0;
  }

  /** Whether a sound of this logical loudness is inside the window. */
  audible(db: number): boolean {
    return db > this.top - WINDOW_HEIGHT_DB;
  }

  /** Starts a frame's tally of continuous sources. */
  begin(): void {
    this.bedPower = 0;
    this.loudest = FLOOR_DB;
  }

  /** A continuous source's logical loudness this frame. */
  offer(db: number): void {
    if (!(db > FLOOR_DB - 20)) return;
    this.bedPower += Math.pow(10, db / 10);
    const sum = 10 * Math.log10(this.bedPower);
    if (sum > this.loudest) this.loudest = sum;
  }

  /**
   * A transient: heard from `at`, at `peakDb`, held `hold` seconds, then
   * falling `decayDbPerS`. If it raises the window, the gain drop is scheduled
   * at `at` on the audio clock, so it lands with the sound and not a frame late.
   */
  event(at: number, peakDb: number, hold: number, decayDbPerS: number, live: boolean): void {
    if (!Number.isFinite(peakDb) || !Number.isFinite(at)) return;
    const e = this.events;
    const o = this.nextSlot * EVENT_STRIDE;
    this.nextSlot = (this.nextSlot + 1) % EVENT_SLOTS;
    e[o] = at;
    e[o + 1] = peakDb;
    e[o + 2] = at + Math.max(0, hold);
    e[o + 3] = Math.max(1, decayDbPerS);
    const over = peakDb - this.loudest;
    if (over > this.excess && live) this.write(this.gainWith(this.loudest, over), at, ATTACK_TAU, true);
  }

  /** Advances the window and writes the gain. Allocates nothing. */
  update(now: number, dt: number, live: boolean): void {
    const bed = this.loudest;
    let top = bed;
    const e = this.events;
    for (let i = 0; i < EVENT_SLOTS; i++) {
      const o = i * EVENT_STRIDE;
      const peak = e[o + 1];
      if (peak === -Infinity || now < e[o]) continue;
      const level = now <= e[o + 2] ? peak : peak - (now - e[o + 2]) * e[o + 3];
      if (level < FLOOR_DB) {
        e[o + 1] = -Infinity;
        continue;
      }
      if (level > top) top = level;
    }
    // The events' rise over the beds: attack at once, release at a rate.
    const released = this.excess - RELEASE_DB_PER_S * Math.max(0, dt);
    this.excess = Math.max(0, top - bed, released);
    this.top = Math.max(FLOOR_DB, bed + this.excess);
    if (live) this.write(this.gainWith(bed, this.excess), now, FOLLOW_TAU, false);
  }

  /** Jumps to the current gain, for the frame the context resumes on. */
  flush(now: number): void {
    const db = this.gainWith(this.loudest, this.excess);
    this.written = db;
    const g = this.node.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(dbToGain(db), now);
  }

  /** Forgets every event; the window drops to the beds at once. */
  reset(): void {
    for (let i = 0; i < EVENT_SLOTS; i++) this.events[i * EVENT_STRIDE + 1] = -Infinity;
    this.top = FLOOR_DB;
    this.excess = 0;
  }

  /** The gain for beds at `bed` with an event `over` dB above them. */
  private gainWith(bed: number, over: number): number {
    const keep = this.reduced ? EVENT_KEEP_REDUCED : EVENT_KEEP_FULL;
    return Math.max(MAX_CUT_DB, this.gainFor(bed) - (1 - keep) * Math.max(0, over));
  }

  private gainFor(top: number): number {
    const slope = this.reduced ? SLOPE_REDUCED : SLOPE_FULL;
    const out = TARGET_DB + (this.reduced ? REDUCED_LIFT_DB : 0) + slope * (top - REF_DB);
    const db = out - top;
    return db < MAX_CUT_DB ? MAX_CUT_DB : db > MAX_BOOST_DB ? MAX_BOOST_DB : db;
  }

  private write(db: number, at: number, tau: number, force: boolean): void {
    if (!force && Number.isFinite(this.written) && Math.abs(db - this.written) < EPSILON_DB) return;
    this.written = db;
    this.node.gain.setTargetAtTime(dbToGain(db), at, tau);
  }
}

export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

export function gainToDb(gain: number): number {
  return gain > 1e-9 ? 20 * Math.log10(gain) : -180;
}
