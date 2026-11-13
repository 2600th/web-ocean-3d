/**
 * The contract between the gameplay layer and whatever draws it.
 *
 * Gameplay never touches the DOM. It reports through a `GameUiSink`, which the
 * HUD implements; until it does, `main.ts` wires a stand-in that forwards to the
 * HUD where a method exists and logs otherwise. Keeping this file free of
 * `three` is deliberate — `src/ui` stays three-free and imports it as-is.
 *
 * **Bearings.** Every angle here is a compass bearing in degrees, 0..360,
 * clockwise from north, where north is world -Z and east is world +X (looking
 * down the Y axis with the default camera, north is "up the screen"). The ship's
 * heading and an objective's bearing use the same convention, so the bearing
 * *relative to the bow* is simply `bearingDeg - headingDeg`.
 */

export type ToastKind = 'info' | 'reward' | 'warning';

/** How much canvas is set. W raises it a step, S takes it in a step. */
export type SailSetting = 'furled' | 'half' | 'full';

export interface GameObjective {
  /** Short imperative line, e.g. "Recover the drifting cargo". */
  title: string;
  /** Secondary line: what the target is. */
  detail?: string;
  /** Compass bearing from the ship to the target, degrees. Null when unknown. */
  bearingDeg: number | null;
  /** Horizontal distance from the ship to the target, metres. */
  distanceM: number | null;
}

export interface ShipTelemetry {
  /** Speed over ground, knots. */
  speedKnots: number;
  /** Compass heading of the bow, degrees. */
  headingDeg: number;
  sail: SailSetting;
  /** True from the moment the anchor is let go until it is fully weighed. */
  anchored: boolean;
  /**
   * Compass bearing the wind blows **from**, degrees — the sailor's convention,
   * so a wind of 90 is an easterly and comes over the starboard beam of a ship
   * heading north.
   */
  windBearingDeg: number;
  /** True wind speed, knots. */
  windKnots: number;
}

/**
 * Implemented by the HUD. Every method may be called every frame while the
 * helm is active (the gameplay layer throttles the continuous ones to ~10 Hz),
 * and must be cheap and idempotent.
 */
export interface GameUiSink {
  /** A transient message. `reward` for a pickup, `warning` for a hazard. */
  toast(text: string, kind?: ToastKind): void;
  /** The current objective, or null to clear it (e.g. on leaving the helm). */
  setObjective(objective: GameObjective | null): void;
  /** A named running total, e.g. ("Cargo", 3) or ("Gold", 420). */
  setScore(label: string, value: number): void;
  /** Ship instruments, or null to hide them (e.g. on leaving the helm). */
  setShipTelemetry(telemetry: ShipTelemetry | null): void;
}

/** The audio the gameplay layer asks for. `AudioSystem` satisfies it. */
export interface GameAudio {
  playSplash(intensity?: number): void;
  playChime(): void;
  playAnchorChain(seconds?: number): void;
  /** The hull striking a wave face or the bottom. Optional so a test double can omit it. */
  hullImpact?(strength: number, grind?: boolean): void;
}

/**
 * What the audio needs from the sails each frame. Written by
 * `Gameplay.audioState` into a caller's object, so polling it allocates nothing.
 */
export interface GameplayAudioState {
  /** At the helm. Everything below is 0 when not. */
  active: boolean;
  /** Canvas set, 0..1. */
  sailFill: number;
  /**
   * How hard the canvas is luffing, 0..1: head to wind, spilled at anchor, or
   * flogging while it is being set or taken in.
   */
  sailLuff: number;
}

/** A sink that drops everything; the default when nothing is attached. */
export const NULL_UI_SINK: GameUiSink = {
  toast: () => undefined,
  setObjective: () => undefined,
  setScore: () => undefined,
  setShipTelemetry: () => undefined,
};
