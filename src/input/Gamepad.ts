/**
 * Gamepad at the helm, and rumble.
 *
 * The standard mapping (W3C Gamepad, `mapping === 'standard'`), which is what
 * Xbox, PlayStation and most other pads report in Chrome, Edge, Firefox and
 * Safari:
 *
 *   left stick, across     rudder
 *   right trigger / d-pad up    more sail
 *   left trigger / d-pad down   less sail
 *   right bumper           ease the sheets out (held)
 *   left bumper            haul the sheets in (held)
 *   A (bottom face)        drop or raise the anchor
 *   right stick            look around the ship
 *
 * Polled once a frame, because the Gamepad API has no events for buttons or
 * axes. Nothing is polled until a pad has connected: browsers only expose one
 * after it has been touched, and `gamepadconnected` says so.
 *
 * Continuous values (rudder, trim, look) are reported only when they change,
 * like the touch stick, so an idle pad never stamps zeroes over the keyboard
 * or the touch controls. Buttons fire on the press, not while held.
 *
 * **Rumble** goes through `vibrationActuator.playEffect('dual-rumble')`, which
 * Chrome, Edge and Safari 16.4+ have and Firefox and iOS Safari do not. It is
 * feature-detected per pad and every call is guarded: a pad without it is
 * still a pad. It is driven by the same trauma value that shakes the camera, so
 * a slam, a grounding or the anchor snubbing up is felt in the hands exactly
 * when it is seen, and `haptics` turns it off.
 *
 * Pure input: no three.js, no DOM beyond `navigator` and `window`.
 */

export interface GamepadCallbacks {
  /** Rudder, -1 (port) .. 1 (starboard). Only on change. */
  onSteer(rudder: number): void;
  /** Sheet rate, -1 (haul in) .. 1 (ease out). Only on change. */
  onTrim(rate: number): void;
  /** One step more (1) or less (-1) sail. */
  onSail(direction: 1 | -1): void;
  /** Drop or raise the anchor. */
  onAnchor(): void;
  /** Look rate, each -1..1: x to the right, y up. Only on change. */
  onLook(x: number, y: number): void;
  /** The pad was used this frame: prompts should show pad glyphs. */
  onActive?(): void;
}

/** Standard-mapping button indices. */
const BUTTON = {
  a: 0,
  lb: 4,
  rb: 5,
  lt: 6,
  rt: 7,
  up: 12,
  down: 13,
} as const;

/** Stick dead zone, rescaled past so the usable range still starts at zero. */
const STICK_DEAD = 0.18;
/** Trigger travel that counts as a press, and the release below which it re-arms. */
const TRIGGER_PRESS = 0.55;
const TRIGGER_RELEASE = 0.3;
/** Trauma rise, per frame, below which nothing is felt. */
const RUMBLE_THRESHOLD = 0.04;
/** Seconds between rumbles, so a run of slams is not one long buzz. */
const RUMBLE_GAP = 0.12;

interface RumbleActuator {
  playEffect(
    type: 'dual-rumble',
    params: { duration: number; startDelay?: number; strongMagnitude: number; weakMagnitude: number },
  ): Promise<unknown>;
}

export class GamepadInput {
  /** Whether rumble plays. The `haptics` state flag. */
  haptics = true;

  private readonly callbacks: GamepadCallbacks;
  private readonly abort = new AbortController();
  private connected = 0;
  private index: number | null = null;

  private rudder = 0;
  private trim = 0;
  private lookX = 0;
  private lookY = 0;
  private readonly held = new Set<number>();
  private enabled = false;

  private lastTrauma = 0;
  private rumbleCooldown = 0;

  constructor(callbacks: GamepadCallbacks) {
    this.callbacks = callbacks;
    if (typeof window === 'undefined' || typeof navigator === 'undefined') return;
    if (typeof navigator.getGamepads !== 'function') return;
    const { signal } = this.abort;
    window.addEventListener(
      'gamepadconnected',
      (event) => {
        this.connected += 1;
        if (this.index === null) this.index = (event as GamepadEvent).gamepad.index;
      },
      { signal },
    );
    window.addEventListener(
      'gamepaddisconnected',
      (event) => {
        this.connected = Math.max(0, this.connected - 1);
        if ((event as GamepadEvent).gamepad.index === this.index) {
          this.index = null;
          this.release();
        }
      },
      { signal },
    );
  }

  /** True while a pad is connected and has been seen. */
  get isConnected(): boolean {
    return this.connected > 0;
  }

  /**
   * Polls the pad. Call once a frame.
   *
   * @param dt      Seconds since the last frame.
   * @param atHelm  Whether the viewer has the helm. Off releases everything.
   * @param trauma  The camera-shake trauma, 0..1; a rise is felt as rumble.
   */
  update(dt: number, atHelm: boolean, trauma: number): void {
    this.rumbleCooldown = Math.max(0, this.rumbleCooldown - Math.max(0, dt));
    const rise = trauma - this.lastTrauma;
    this.lastTrauma = trauma;
    if (this.connected === 0) return;

    const pad = this.pad();
    if (!pad) return;

    if (atHelm && rise > RUMBLE_THRESHOLD) this.rumble(pad, rise);

    if (atHelm !== this.enabled) {
      this.enabled = atHelm;
      if (!atHelm) this.release();
    }
    if (!atHelm) {
      // Keep the edge state current so a button held across the switch does
      // not fire the moment the helm is taken.
      this.held.clear();
      for (const b of Object.values(BUTTON)) if (pressed(pad, b, this.held.has(b))) this.held.add(b);
      return;
    }

    let used = false;

    const rudder = stick(pad.axes[0] ?? 0);
    if (rudder !== this.rudder) {
      this.rudder = rudder;
      this.callbacks.onSteer(rudder);
      used ||= rudder !== 0;
    }

    const trim = (pressed(pad, BUTTON.rb, false) ? 1 : 0) - (pressed(pad, BUTTON.lb, false) ? 1 : 0);
    if (trim !== this.trim) {
      this.trim = trim;
      this.callbacks.onTrim(trim);
      used ||= trim !== 0;
    }

    // Right stick: x to the right, y up (the axis reads down as positive).
    const lx = stick(pad.axes[2] ?? 0);
    const ly = -stick(pad.axes[3] ?? 0);
    if (lx !== this.lookX || ly !== this.lookY) {
      this.lookX = lx;
      this.lookY = ly;
      this.callbacks.onLook(lx, ly);
      used ||= lx !== 0 || ly !== 0;
    }

    // Each edge read on its own: a short-circuit would skip updating the second.
    const more = this.edge(pad, BUTTON.rt);
    const moreUp = this.edge(pad, BUTTON.up);
    const less = this.edge(pad, BUTTON.lt);
    const lessDown = this.edge(pad, BUTTON.down);
    if (more || moreUp) {
      this.callbacks.onSail(1);
      used = true;
    }
    if (less || lessDown) {
      this.callbacks.onSail(-1);
      used = true;
    }
    if (this.edge(pad, BUTTON.a)) {
      this.callbacks.onAnchor();
      used = true;
    }

    if (used) this.callbacks.onActive?.();
  }

  dispose(): void {
    this.abort.abort();
    this.release();
    this.connected = 0;
  }

  // ---------------------------------------------------------------- internals

  private pad(): Gamepad | null {
    let pads: (Gamepad | null)[];
    try {
      pads = navigator.getGamepads();
    } catch {
      return null;
    }
    if (this.index !== null && pads[this.index]) return pads[this.index];
    // The first connected pad, preferring the standard mapping.
    let fallback: Gamepad | null = null;
    for (const pad of pads) {
      if (!pad || !pad.connected) continue;
      if (pad.mapping === 'standard') {
        this.index = pad.index;
        return pad;
      }
      fallback ??= pad;
    }
    if (fallback) this.index = fallback.index;
    return fallback;
  }

  /** True on the frame a button goes down. */
  private edge(pad: Gamepad, button: number): boolean {
    const was = this.held.has(button);
    const now = pressed(pad, button, was);
    if (now) this.held.add(button);
    else this.held.delete(button);
    return now && !was;
  }

  /** Zeroes every continuous value the pad was holding. */
  private release(): void {
    if (this.rudder !== 0) {
      this.rudder = 0;
      this.callbacks.onSteer(0);
    }
    if (this.trim !== 0) {
      this.trim = 0;
      this.callbacks.onTrim(0);
    }
    if (this.lookX !== 0 || this.lookY !== 0) {
      this.lookX = 0;
      this.lookY = 0;
      this.callbacks.onLook(0, 0);
    }
  }

  /**
   * A short dual rumble scaled by how much trauma arrived: the heavy motor for
   * the thump, the light one for the rattle after it.
   */
  private rumble(pad: Gamepad, rise: number): void {
    if (!this.haptics || this.rumbleCooldown > 0) return;
    const actuator = (pad as unknown as { vibrationActuator?: Partial<RumbleActuator> | null })
      .vibrationActuator;
    if (!actuator || typeof actuator.playEffect !== 'function') return;
    const amount = Math.min(1, rise);
    this.rumbleCooldown = RUMBLE_GAP;
    try {
      void actuator
        .playEffect('dual-rumble', {
          duration: Math.round(90 + 260 * amount),
          strongMagnitude: Math.min(1, 0.25 + amount * 1.1),
          weakMagnitude: Math.min(1, 0.35 + amount * 0.8),
        })
        .catch(() => undefined);
    } catch {
      // A pad that advertises the actuator and then refuses it: no rumble.
    }
  }
}

/** A stick axis past the dead zone, rescaled to start at zero. */
function stick(value: number): number {
  const magnitude = Math.abs(value);
  if (!(magnitude > STICK_DEAD)) return 0;
  const scaled = Math.min(1, (magnitude - STICK_DEAD) / (1 - STICK_DEAD));
  return Math.round((value < 0 ? -scaled : scaled) * 100) / 100;
}

/**
 * Whether a button counts as down. Analogue triggers use a press and a lower
 * release threshold, so a trigger resting half way does not chatter.
 */
function pressed(pad: Gamepad, index: number, wasDown: boolean): boolean {
  const button = pad.buttons[index];
  if (!button) return false;
  if (index === BUTTON.lt || index === BUTTON.rt) {
    return button.value >= (wasDown ? TRIGGER_RELEASE : TRIGGER_PRESS);
  }
  return button.pressed;
}
