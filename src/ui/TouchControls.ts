/**
 * On-screen throttle and rudder for touch devices.
 *
 * Boat mode is driven with W/S and A/D, which a phone does not have. This is the
 * same two axes as a thumbstick: push forward for ahead, pull back for astern,
 * left and right for rudder. It writes into `ShipController.setInput`, the same
 * entry point the keyboard resolves to, so there is one place where throttle and
 * rudder become forces and no second control path to keep in step.
 *
 * Shown only when there is a coarse pointer *and* Boat mode is selected. A mouse
 * user already has better controls and a stick over the frame would only be in
 * the way; a touch user in Orbit mode is panning the camera and would find a
 * stick that steers a ship they are not driving actively confusing.
 *
 * Pointer Events rather than Touch Events, so the same code serves a stylus, a
 * touchscreen and a mouse — which also makes it drivable from a test.
 */

import { icon, type IconName } from './icons.ts';

/** Travel of the knob from the centre, as a fraction of the pad's radius. */
const KNOB_TRAVEL = 0.62;

/**
 * Dead zone as a fraction of the radius.
 *
 * A thumb resting on a stick is never exactly centred, and without this the ship
 * creeps and the rudder sits a few degrees over whenever the pad is touched at
 * all. 0.12 is small enough that deliberate input still feels immediate.
 */
const DEAD_ZONE = 0.12;

export interface TouchControlsCallbacks {
  /** Throttle and rudder, each -1..1. Called only when a value changes. */
  onInput(throttle: number, rudder: number): void;
  /** Sail trim one step up (1) or down (-1). Omitted: no sail buttons. */
  onSail?(direction: 1 | -1): void;
  /** Drop or weigh the anchor. Omitted: no anchor button. */
  onAnchor?(): void;
}

export class TouchControls {
  private readonly root: HTMLElement;
  private readonly pad: HTMLElement;
  private readonly knob: HTMLElement;
  /** Sail and anchor buttons; the keyboard's W/S and R for a thumb. */
  private readonly actions: HTMLElement | null;
  private readonly callbacks: TouchControlsCallbacks;
  private readonly controller = new AbortController();

  private activePointer: number | null = null;
  private throttle = 0;
  private rudder = 0;
  private visible = false;
  private disposed = false;

  constructor(root: HTMLElement, callbacks: TouchControlsCallbacks) {
    this.root = root;
    this.callbacks = callbacks;
    const signal = this.controller.signal;

    const pad = document.createElement('div');
    pad.className = 'touchpad';
    pad.setAttribute('role', 'group');
    pad.setAttribute('aria-label', 'Steering stick: up for ahead, down for astern, left and right to steer');
    pad.hidden = true;

    const ring = document.createElement('div');
    ring.className = 'touchpad__ring';

    const knob = document.createElement('div');
    knob.className = 'touchpad__knob';

    const ahead = document.createElement('span');
    ahead.className = 'touchpad__label touchpad__label--ahead';
    ahead.textContent = 'AHEAD';
    const astern = document.createElement('span');
    astern.className = 'touchpad__label touchpad__label--astern';
    astern.textContent = 'ASTERN';

    ring.append(knob);
    pad.append(ring, ahead, astern);

    pad.addEventListener('pointerdown', this.onPointerDown, { signal });
    pad.addEventListener('pointermove', this.onPointerMove, { signal });
    pad.addEventListener('pointerup', this.onPointerUp, { signal });
    pad.addEventListener('pointercancel', this.onPointerUp, { signal });
    // The pad sits over the canvas, which owns its own drag gestures. Without
    // this a stick input also orbits the camera underneath it.
    pad.addEventListener('contextmenu', (e) => e.preventDefault(), { signal });

    this.pad = pad;
    this.knob = knob;
    this.root.append(pad);

    // The stick is rudder and engine; the sails and the anchor are discrete
    // orders, so they are buttons — the same three a keyboard player reaches
    // with W, S and R.
    let actions: HTMLElement | null = null;
    if (callbacks.onSail || callbacks.onAnchor) {
      actions = document.createElement('div');
      actions.className = 'touchhelm';
      actions.setAttribute('role', 'group');
      actions.setAttribute('aria-label', 'Sails and anchor');
      actions.hidden = true;
      const button = (glyph: IconName, label: string, title: string, run: () => void): void => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'touchhelm__btn';
        const caption = document.createElement('span');
        caption.className = 'touchhelm__label';
        caption.textContent = label;
        b.append(icon(glyph, 'touchhelm__icon'), caption);
        b.title = title;
        b.setAttribute('aria-label', title);
        b.addEventListener(
          'pointerdown',
          (e) => {
            e.preventDefault();
            run();
          },
          { signal },
        );
        b.addEventListener('contextmenu', (e) => e.preventDefault(), { signal });
        actions!.append(b);
      };
      if (callbacks.onSail) {
        button('sailUp', 'More', 'More sail', () => callbacks.onSail!(1));
        button('sailDown', 'Less', 'Less sail', () => callbacks.onSail!(-1));
      }
      if (callbacks.onAnchor) {
        button('anchor', 'Anchor', 'Drop or raise the anchor', () => callbacks.onAnchor!());
      }
      this.root.append(actions);
    }
    this.actions = actions;
  }

  /**
   * Whether this device wants on-screen controls at all.
   *
   * `pointer: coarse` is the media query for "the primary input is a finger".
   * `maxTouchPoints` backs it up for the hybrid case — a laptop with a
   * touchscreen reports a fine pointer, and someone who reaches up to touch the
   * screen should still find a stick there.
   */
  static isTouchDevice(): boolean {
    if (typeof window === 'undefined') return false;
    if (window.matchMedia?.('(pointer: coarse)').matches) return true;
    return (navigator.maxTouchPoints ?? 0) > 0;
  }

  /** Boat mode on a touch device shows the pad; anything else hides it. */
  setVisible(visible: boolean): void {
    if (this.disposed || this.visible === visible) return;
    this.visible = visible;
    this.pad.hidden = !visible;
    if (this.actions) this.actions.hidden = !visible;
    if (!visible) this.release();
  }

  get isVisible(): boolean {
    return this.visible;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.controller.abort();
    this.pad.remove();
    this.actions?.remove();
  }

  // ---------------------------------------------------------------- internals

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (this.activePointer !== null) return;
    this.activePointer = event.pointerId;
    this.pad.setPointerCapture(event.pointerId);
    event.preventDefault();
    this.applyFromEvent(event);
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointer) return;
    event.preventDefault();
    this.applyFromEvent(event);
  };

  private readonly onPointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointer) return;
    if (this.pad.hasPointerCapture(event.pointerId)) {
      this.pad.releasePointerCapture(event.pointerId);
    }
    this.release();
  };

  /**
   * Centres the knob and zeroes the input.
   *
   * A stick has to spring back. Leaving the last value latched means lifting a
   * finger mid-turn leaves the rudder hard over and the ship circling, which is
   * exactly the failure a physical spring-centred control exists to prevent.
   */
  private release(): void {
    this.activePointer = null;
    this.knob.style.transform = 'translate(-50%, -50%)';
    this.emit(0, 0);
  }

  private applyFromEvent(event: PointerEvent): void {
    const box = this.pad.getBoundingClientRect();
    const radius = Math.min(box.width, box.height) * 0.5;
    if (radius <= 0) return;

    const dx = (event.clientX - (box.left + box.width * 0.5)) / radius;
    const dy = (event.clientY - (box.top + box.height * 0.5)) / radius;

    // Clamped to the unit disc rather than the square, so the corners cannot
    // produce a magnitude of sqrt(2) and a stick pushed diagonally is not
    // stronger than one pushed straight.
    const length = Math.hypot(dx, dy);
    const scale = length > 1 ? 1 / length : 1;
    const nx = dx * scale;
    const ny = dy * scale;

    this.knob.style.transform =
      `translate(calc(-50% + ${(nx * KNOB_TRAVEL * radius).toFixed(1)}px), ` +
      `calc(-50% + ${(ny * KNOB_TRAVEL * radius).toFixed(1)}px))`;

    // Screen y grows downward; pushing the stick up is ahead.
    this.emit(deadZone(-ny), deadZone(nx));
  }

  private emit(throttle: number, rudder: number): void {
    if (throttle === this.throttle && rudder === this.rudder) return;
    this.throttle = throttle;
    this.rudder = rudder;
    this.callbacks.onInput(throttle, rudder);
  }
}

/**
 * Rescales past the dead zone rather than clipping to it.
 *
 * Clipping leaves a step: the first input past the threshold jumps straight to
 * 0.12 of full throttle. Rescaling means the usable range still starts at zero.
 */
function deadZone(value: number): number {
  const magnitude = Math.abs(value);
  if (magnitude <= DEAD_ZONE) return 0;
  const scaled = (magnitude - DEAD_ZONE) / (1 - DEAD_ZONE);
  return value < 0 ? -scaled : scaled;
}
