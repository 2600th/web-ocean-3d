import { CINEMATIC_LOOP_SECONDS, tourHours } from '../cameras/Cinematic';
import { mulberry32 } from '../core/random';
import type { WeatherKind } from '../presets';

/**
 * A voyage: a beginning, a risk and an end, in three to five minutes.
 *
 * It opens with a line in the ship's log, lays out a few casks adrift, and
 * ends when the player drops anchor off the cove with cargo aboard. The risk
 * is the weather and the light: a squall builds partway through (gusts that
 * overpress full sail, and a cask over the side for a ship that will not take
 * in canvas), and the day runs on while she sails, so the dark closes in.
 * Failure costs time or cargo, never the session.
 *
 * **Pure bookkeeping.** This holds clocks, the seed and the phase, and answers
 * what the sky should be. `Gameplay` owns the casks, the forces and the HUD;
 * `main.ts` applies the sky through the same path the Tour uses.
 *
 * **The day reuses the Tour's own hour curve.** `tourHours` is the Tour's
 * monotone day (dawn, a long afternoon, a short golden hour, the night hurried
 * through); the helm runs it at `DAY_SECONDS` per lap instead of the Tour's
 * ninety, starting early in the afternoon so golden hour arrives about three
 * minutes into a visit.
 *
 * **Deterministic when frozen.** `reset` puts back a fixed seed and freezes
 * every clock, so a capture taken at the helm sees the preset's sky, no squall
 * and the same cask as before voyages existed. Live play unfreezes it with a
 * fresh seed the first time the helm is taken.
 */

export type VoyagePhase = 'gather' | 'homeward' | 'landed';

/** Things worth telling the player, raised by `advance`. */
export type VoyageEvent = 'squallWarning' | 'squallArrived' | 'squallPassed' | 'dusk';

/** What the helm asks of the sky this frame: the Tour's `CinematicEnvironment`, field for field. */
export interface VoyageSky {
  hours: number;
  rain: number;
  cloudCoverage: number;
  fogDensity: number;
  weatherKind: WeatherKind;
}

/** Casks laid out per voyage. */
export const VOYAGE_CASKS = 3;

/** Seconds of helm time per day: ten minutes. */
export const DAY_SECONDS = 600;
/**
 * Where on the Tour's lap the first day starts: 40 s of 90 is about 14:05.
 * Golden hour (the Tour's knot at 66.5 s, 17:36) then arrives
 * (66.5 - 40) / 90 * 600 = 177 s in, and full dark (77 s, 19:54) at 247 s.
 */
const DAY_START_TOUR = 40;
/**
 * Seconds at the helm before the day takes the sky. A harness that selects
 * the helm and then rewinds for a capture spends a few frames live first;
 * without this the sun would jump to the afternoon under it.
 */
const DAY_DELAY = 2;
/** Hour past which the HUD says the sun is going down. */
const DUSK_HOURS = 17.9;

/** Peak rain rate of a voyage's squall, 0..1. The Tour's is 0.85. */
export const SQUALL_PEAK = 0.8;
/** Voyage seconds before the squall starts to build: earliest, plus a seeded spread. */
const SQUALL_EARLIEST = 75;
const SQUALL_SPREAD = 70;
/** Seconds to build to the peak, to hold it, and to clear. */
const SQUALL_BUILD = 20;
const SQUALL_HOLD = 30;
const SQUALL_CLEAR = 25;
/** Seconds of warning the HUD gives before the rain arrives. */
const SQUALL_WARNING = 30;
/** Cloud cover a squall brings at its peak, and the fog multiplier per unit rain (the Tour's 1.7). */
const SQUALL_COVERAGE = 0.92;
const SQUALL_FOG = 1.7;

/** The seed under `reset`: the cask sequence every capture was taken with. */
export const FIXED_VOYAGE_SEED = 0x5ea7e1;

/**
 * Bearing of the cove from the island centre, radians (x = cos, z = sin).
 * `COVE_BEARING` in `scene/Props.ts`, which does not export it; `Seafloor`
 * keeps its lagoon on the same number.
 */
export const COVE_BEARING = 0.7;
/** Water this deep, metres, for this far out, makes an anchorage. */
const ANCHORAGE_DEPTH = 12;
const ANCHORAGE_RUN = 40;
/** How far past the bar the anchorage sits, metres. */
const ANCHORAGE_MARGIN = 30;
/** Metres from the anchorage within which dropping anchor lands the cargo. */
export const ANCHORAGE_RADIUS = 90;

function smootherstep(x: number): number {
  const s = Math.min(1, Math.max(0, x));
  return s * s * s * (s * (s * 6 - 15) + 10);
}

/** Rain from one squall that started building at day-clock `start`, `t` seconds of day. */
function squallRain(start: number, t: number): number {
  const u = t - start;
  if (u <= 0) return 0;
  if (u < SQUALL_BUILD) return SQUALL_PEAK * smootherstep(u / SQUALL_BUILD);
  if (u < SQUALL_BUILD + SQUALL_HOLD) return SQUALL_PEAK;
  const c = u - SQUALL_BUILD - SQUALL_HOLD;
  if (c < SQUALL_CLEAR) return SQUALL_PEAK * (1 - smootherstep(c / SQUALL_CLEAR));
  return 0;
}

/**
 * Where to anchor off the cove: out from the island centre along the cove's
 * bearing, past the beach, the lagoon (5.5 m) and the bar, to the first water
 * deep enough to swing in, and a little beyond. Asked of the heightfield so it
 * follows the coast if the coast moves.
 */
export function findAnchorage(
  floorHeight: (x: number, z: number) => number,
  island: { x: number; z: number; radius: number },
): { x: number; z: number } {
  const ux = Math.cos(COVE_BEARING);
  const uz = Math.sin(COVE_BEARING);
  let run = 0;
  for (let r = island.radius * 0.5; r < island.radius * 4; r += 5) {
    const deep = -floorHeight(island.x + ux * r, island.z + uz * r) >= ANCHORAGE_DEPTH;
    run = deep ? run + 5 : 0;
    if (run >= ANCHORAGE_RUN) {
      const at = r + ANCHORAGE_MARGIN;
      return { x: island.x + ux * at, z: island.z + uz * at };
    }
  }
  const at = island.radius * 2;
  return { x: island.x + ux * at, z: island.z + uz * at };
}

/** A fresh seed for a live voyage. */
export function freshVoyageSeed(): number {
  try {
    const out = new Uint32Array(1);
    globalThis.crypto.getRandomValues(out);
    return out[0] >>> 0;
  } catch {
    return (Math.random() * 0x1_0000_0000) >>> 0;
  }
}

export class Voyage {
  seed = FIXED_VOYAGE_SEED;
  /** 1 for the first voyage this browser has brought home, and so on. */
  number = 1;
  phase: VoyagePhase = 'gather';
  /** Seconds of this voyage at the helm; stops when she lands. */
  clock = 0;
  /** Seconds of day at the helm, carried from one voyage to the next. */
  dayClock = 0;
  /** True under a deterministic capture: no clock runs, no squall, the preset's sky. */
  frozen = true;
  /** False until the opening log line has been given for this voyage. */
  begun = false;
  /** Day-clock second this voyage's squall starts to build. */
  squallStart = 0;
  /** The previous voyage's squall, so one still blowing does not stop dead when a new voyage begins. */
  private lastSquallStart = Number.NEGATIVE_INFINITY;
  /** 0 nothing said, 1 warned, 2 arrived, 3 passed. */
  private squallStage = 0;
  private lastHours = Number.NaN;
  private readonly sky: VoyageSky = { hours: 12, rain: 0, cloudCoverage: 0.3, fogDensity: 1, weatherKind: 'clear' };

  /** Back to the fixed seed with every clock stopped. See the class comment. */
  reset(): void {
    this.frozen = true;
    this.dayClock = 0;
    this.lastSquallStart = Number.NEGATIVE_INFINITY;
    this.begin(FIXED_VOYAGE_SEED, 1);
    this.begun = false;
    this.lastHours = Number.NaN;
  }

  /** Starts voyage `number` from `seed`: the squall's timing is drawn from it here. */
  begin(seed: number, number: number): void {
    this.seed = seed >>> 0;
    this.number = Math.max(1, Math.floor(number));
    this.phase = 'gather';
    this.clock = 0;
    this.begun = true;
    // Its own stream, so the squall never moves the casks the seed lays out.
    const random = mulberry32(this.seed ^ 0x9e3779b9);
    if (this.squallStart > this.lastSquallStart) this.lastSquallStart = this.squallStart;
    this.squallStart = this.dayClock + SQUALL_EARLIEST + random() * SQUALL_SPREAD;
    this.squallStage = 0;
  }

  /** Advances the clocks by `dt` at the helm and appends anything worth saying to `events`. */
  advance(dt: number, events: VoyageEvent[]): void {
    if (this.frozen || !(dt > 0)) return;
    if (this.phase !== 'landed') this.clock += dt;
    this.dayClock += dt;

    if (this.phase !== 'landed') {
      const until = this.squallStart - this.dayClock;
      if (this.squallStage === 0 && until <= SQUALL_WARNING && until > 0) {
        this.squallStage = 1;
        events.push('squallWarning');
      }
      if (this.squallStage <= 1 && this.squallRain() > SQUALL_PEAK * 0.35) {
        this.squallStage = 2;
        events.push('squallArrived');
      }
    }
    if (this.squallStage === 2 && this.dayClock > this.squallStart + SQUALL_BUILD + SQUALL_HOLD + SQUALL_CLEAR * 0.6) {
      this.squallStage = 3;
      events.push('squallPassed');
    }

    const hours = this.hours();
    // Crossing dusk going forward; past midnight the hours wrap and it can come round again.
    if (this.lastHours < DUSK_HOURS && hours >= DUSK_HOURS && this.phase !== 'landed') events.push('dusk');
    this.lastHours = hours;
  }

  /** Rain from the squall now, 0..`SQUALL_PEAK`. Zero while frozen. */
  squallRain(): number {
    if (this.frozen) return 0;
    return Math.max(squallRain(this.squallStart, this.dayClock), squallRain(this.lastSquallStart, this.dayClock));
  }

  /** The squall's strength, 0..1: what the gusts on the canvas scale with. */
  squallStrength(): number {
    return this.squallRain() / SQUALL_PEAK;
  }

  /** Where this voyage's squall stands, as the HUD has been told it. */
  get squallState(): 'none' | 'coming' | 'here' | 'past' {
    return (['none', 'coming', 'here', 'past'] as const)[this.squallStage] ?? 'none';
  }

  /** Seconds until the squall builds, or null once it has. */
  squallIn(): number | null {
    const until = this.squallStart - this.dayClock;
    return until > 0 ? until : null;
  }

  /** Position on the Tour's lap that the day has reached. */
  tourTime(): number {
    const t = DAY_START_TOUR + (this.dayClock * CINEMATIC_LOOP_SECONDS) / DAY_SECONDS;
    return ((t % CINEMATIC_LOOP_SECONDS) + CINEMATIC_LOOP_SECONDS) % CINEMATIC_LOOP_SECONDS;
  }

  /** Hour of the day, 0..24, from the Tour's own curve. */
  hours(): number {
    return tourHours(this.tourTime());
  }

  /**
   * The sky the helm wants, or null to leave the preset's. `base` is what the
   * preset and the player's sliders ask for; the squall is laid over it, and a
   * preset that already rains or snows keeps its kind and only gets heavier.
   * `fixedHour` is the player's own time-of-day setting, which outranks the
   * day: someone who chose an hour keeps it, and still gets the squall.
   *
   * Returns a live object; read it before the next call.
   */
  skyFor(
    base: { coverage: number; kind: WeatherKind; intensity: number },
    fixedHour: number | null,
  ): Readonly<VoyageSky> | null {
    if (this.frozen || this.dayClock < DAY_DELAY) return null;
    const rain = this.squallRain();
    const strength = rain / SQUALL_PEAK;
    const sky = this.sky;
    sky.hours = fixedHour ?? this.hours();
    sky.weatherKind = base.kind !== 'clear' ? base.kind : rain > 0.001 ? 'rain' : 'clear';
    sky.rain = base.kind !== 'clear' ? Math.max(base.intensity, rain) : rain;
    sky.cloudCoverage = base.coverage + Math.max(0, SQUALL_COVERAGE - base.coverage) * strength;
    sky.fogDensity = 1 + rain * SQUALL_FOG;
    return sky;
  }
}
