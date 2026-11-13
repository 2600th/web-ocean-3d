import * as THREE from 'three/webgpu';
import type { BuoyantBody } from './Buoyancy';

/**
 * Throttle and rudder for the hero ship.
 *
 * The controller never writes a position or a heading. It resolves the driver's
 * intent into a force and a yaw torque and hands those to the buoyancy solver,
 * which integrates them alongside gravity, flotation and the wave response. That
 * is the difference between a hull that climbs a swell and loses way doing it,
 * and one that slides along a wave field as though it were painted on.
 *
 * The forces are the ones that actually decide how a displacement hull feels:
 *
 *   thrust      along the bow, from throttle
 *   drag        opposing motion, quadratic — this is what sets top speed, not a
 *               clamp, so the hull accelerates onto its maximum and settles
 *   lateral     the keel. Water resists sideways motion an order of magnitude
 *               harder than forward motion, and that asymmetry is most of why a
 *               boat feels like a boat: it carves rather than skidding, and it
 *               keeps carrying way through a turn
 *   yaw         rudder torque scaled by speed, because a rudder is a foil and a
 *               stationary rudder does nothing
 *   yaw damping resistance to spinning, so the turn settles instead of winding up
 */

/** Seconds for the throttle to travel its full range. Engine spool, not a key. */
const THROTTLE_RATE = 0.7;
/** Seconds for the rudder to travel its full range. */
const RUDDER_RATE = 2.2;
/** Rudder self-centring when nothing is pressed, in units per second. */
const RUDDER_RETURN = 1.6;

/**
 * Thrust and drag are derived from the behaviour wanted, not picked by feel.
 *
 * For quadratic drag on mass `m`, terminal speed is `sqrt(T / k)` and the time
 * constant of both acceleration and coasting is `m / sqrt(T * k)`. Fixing the
 * two numbers that can actually be judged — about 8 m/s flat out, and roughly
 * eight seconds to feel the hull respond — pins `k` and `T` exactly.
 *
 * The first attempt set them by intuition at 520 kN and 6 400, which gives the
 * same 8 m/s and a time constant of under two seconds: a 90-tonne ship that
 * accelerated like a jet-ski and coasted to a dead stop in fifteen seconds. The
 * terminal speed was right and the mass was completely absent, which is the
 * failure mode this parameterisation exists to avoid.
 */
const MAX_THRUST = 130_000;
/** Astern thrust as a fraction of ahead. A propeller in reverse is inefficient. */
const REVERSE_FRACTION = 0.45;

/** Quadratic drag along the hull, N per (m/s)². With the thrust above: ~8 m/s. */
const DRAG_LONGITUDINAL = 1_400;
/**
 * Quadratic drag across the hull — the keel. An order of magnitude above the
 * longitudinal figure, which is most of why a hull carves instead of skidding.
 */
const DRAG_LATERAL = 16_000;

/** Peak yaw torque from a hard-over rudder at reference speed, N·m. */
const RUDDER_TORQUE = 1_200_000;
/** Speed, m/s, at which the rudder reaches full authority. */
const RUDDER_REFERENCE_SPEED = 5.5;
/** Yaw damping, N·m per (rad/s). Stops a turn winding up. */
const YAW_DAMPING = 5_600_000;

/**
 * Where the hull stops, and why it is a depth rather than a fence.
 *
 * A viewer could sail the ship up the beach and onto the island. The obvious fix
 * is a circle around the island the hull may not enter, and it is the wrong one
 * twice over: it is invisible in a world that otherwise explains itself, and it
 * is a *circle* on a coastline that is deliberately not one — the shore runs from
 * 0.70 to 1.25 of `ISLAND.radius` by bearing, so any single radius is either well
 * out to sea on the bays or inland on the headland. It would also do nothing
 * about the reef, which is the other place a hull has no business being.
 *
 * So the boundary is the seabed. `dutch_ship_medium` floats with about 3.4 m
 * under her, and a hull is in trouble well before the keel touches: 7 m is where
 * a ship this size starts sounding and turning away, which is the behaviour to
 * reproduce. Below that the water pushes back, and it pushes *down the depth
 * gradient* — toward deeper water, whichever way that happens to be — so the same
 * rule works on the beach, on the reef and on the spit without knowing about any
 * of them.
 */
const SHOAL_DEPTH = 7;
/** Depth at which the push is at full strength and thrust is fully cut. */
const SHOAL_HARD_DEPTH = 3.6;
/**
 * Peak force pushing the hull off a shoal, N.
 *
 * Comfortably more than `MAX_THRUST`, and it has to be: a boundary a player can
 * out-power by holding W is not a boundary, it is a suggestion. Applied as a
 * ramp rather than a wall so a hull that drifts into the shallows is turned
 * rather than stopped dead, which is what makes it read as water shoaling
 * instead of as a collision.
 */
const SHOAL_FORCE = 5_600_000;
/** Metres between the samples the depth gradient is taken from. */
const SHOAL_PROBE = 12;

export interface ShipControlState {
  /** -1 (full astern) … 1 (full ahead). */
  throttle: number;
  /** -1 (hard to port) … 1 (hard to starboard). */
  rudder: number;
  /** Speed over ground, m/s. Always positive. */
  speed: number;
  /** Speed along the bow, m/s. Negative when making sternway. */
  forwardSpeed: number;
  /** Heading in radians, `atan2(forward.z, forward.x)`. */
  heading: number;
}

export class ShipController {
  private readonly body: BuoyantBody;

  private throttleInput = 0;
  private rudderInput = 0;
  private throttle = 0;
  private rudder = 0;
  /**
   * Drive from canvas, as a fraction of `MAX_THRUST`, set by the gameplay layer.
   *
   * Summed with the engine rather than replacing it, so `setInput` — the touch
   * stick, the cinematic flight and every test that drives the hull — keeps
   * working whatever the sails are doing. Zero unless something sets it, which is
   * what keeps the physics identical for any caller that has never heard of sails.
   */
  private sailDrive = 0;
  /** Engine plus sails as applied this frame, a fraction of `MAX_THRUST`. */
  private drive = 0;
  /** Whether W/S reach the engine. See `setThrottleKeysEnabled`. */
  private throttleKeys = true;
  private enabled = false;
  /** Whether W/A/S/D reach the hull. See `setKeyboardEnabled`. */
  private keyboard = true;

  private readonly keys = new Set<string>();
  private readonly abort = new AbortController();

  // Scratch. Runs every frame and must not allocate.
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly force = new THREE.Vector3();
  private readonly torque = new THREE.Vector3();

  /**
   * Seabed elevation at a world xz, or `null` to disable the shoal boundary.
   *
   * Injected rather than imported so the physics layer keeps knowing nothing
   * about the scene: this is `Seafloor`'s own CPU heightfield, the same field the
   * floor mesh is displaced by and the props are placed on, so the hull is
   * turned away by exactly the sand a viewer can see.
   */
  private readonly floorHeight: ((x: number, z: number) => number) | null;

  constructor(body: BuoyantBody, floorHeight: ((x: number, z: number) => number) | null = null) {
    this.body = body;
    this.floorHeight = floorHeight;

    const { signal } = this.abort;
    window.addEventListener('keydown', this.onKeyDown, { signal });
    window.addEventListener('keyup', this.onKeyUp, { signal });
    // A hull under power must not keep its throttle open because the tab lost
    // focus mid-key — the keyup would never arrive.
    window.addEventListener('blur', this.releaseKeys, { signal });
  }

  /**
   * Enables or disables ship input.
   *
   * Disabling zeroes the controls rather than freezing them: leaving Boat mode
   * with the throttle open would have the ship continue under power while the
   * viewer is flying somewhere else entirely. It also clears the external force,
   * so the solver goes back to pure buoyancy rather than being handed a stale
   * thrust forever.
   */
  setEnabled(enabled: boolean): void {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    if (!enabled) {
      this.releaseKeys();
      this.throttle = 0;
      this.rudder = 0;
      this.body.externalForce.set(0, 0, 0);
      this.body.externalTorque.set(0, 0, 0);
    }
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  /** Direct control, for tests and for touch input. Values are clamped. */
  /**
   * Whether the viewer's keys reach the hull.
   *
   * Separate from `setEnabled`, because "the ship is under command" and "the
   * *viewer* is commanding it" are different questions and the cinematic tour is
   * the case that separates them: it needs the controller running so the hull
   * sails under its own physics, and it needs the keyboard out of the way so the
   * authored flight is the only thing steering.
   *
   * Releasing held keys on the way out matters. Without it, a key pressed while
   * the tour was running is still in the set when control returns, and the hull
   * takes off on a keystroke the viewer made seconds ago and has long forgotten.
   */
  setKeyboardEnabled(enabled: boolean): void {
    if (this.keyboard === enabled) return;
    this.keyboard = enabled;
    if (!enabled) this.releaseKeys();
  }

  /**
   * Whether W/S open the engine.
   *
   * Off while the gameplay layer owns those keys for the sails: a tap of W must
   * set canvas, not also spool the engine for as long as the key is held.
   * A/D still reach the rudder either way.
   */
  setThrottleKeysEnabled(enabled: boolean): void {
    this.throttleKeys = enabled;
  }

  /**
   * Forward drive from the sails, as a fraction of full engine thrust.
   *
   * Clamped to [-0.2, 1]: a sail taken aback pushes a hull astern, weakly, and
   * nothing a sail does should out-drive the engine the drag was tuned against.
   */
  setSailDrive(drive: number): void {
    this.sailDrive = Number.isFinite(drive) ? clamp(drive, -0.2, 1) : 0;
  }

  setInput(throttle: number, rudder: number): void {
    this.throttleInput = clamp(throttle, -1, 1);
    this.rudderInput = clamp(rudder, -1, 1);
  }

  /**
   * Returns the engine to a full stop, orders *and* spool.
   *
   * `setInput(0, 0)` is not enough and the difference is the whole point of this
   * existing. The throttle and rudder are spooled quantities — they travel
   * toward the ordered value over seconds, because an engine does — so clearing
   * the order leaves the spool wherever it had reached. Every other integrating
   * system here has a `resetClock`; this one did not, and the omission made the
   * visual baselines a function of the order the suite ran in.
   *
   * The mechanism was indirect enough to hide for a long time. A deterministic
   * capture rewinds the clock, returns the hull to its spawn pose and settles for
   * ninety steps with the shot's engine order applied. With the spool already
   * open from the previous shot the hull was under way from the first of those
   * steps instead of accelerating from rest, so it laid a different wake — and
   * the wake displaces and lights the surface across the whole buffer, which is
   * most of the visible water. The captures differed by 13 levels of mean
   * luminance across the frame while every parameter the harness could read back
   * was identical.
   */
  resetInput(): void {
    this.releaseKeys();
    this.throttleInput = 0;
    this.rudderInput = 0;
    this.throttle = 0;
    this.rudder = 0;
    this.sailDrive = 0;
    this.drive = 0;
    this.body.externalForce.set(0, 0, 0);
    this.body.externalTorque.set(0, 0, 0);
  }

  update(dt: number): void {
    if (!this.enabled || !(dt > 0)) return;

    // --- controls ------------------------------------------------------------
    // The keyboard outranks `setInput` so a viewer at the helm always beats the
    // on-screen throttle, and `keyboard` is what stops that rule applying to a
    // driver that is not a viewer. The cinematic tour steers through `setInput`,
    // and with the keys still live, holding S during a full-ahead beat commanded
    // full astern — the tour visibly fighting for its own wheel while the HUD
    // said the flight was holding it.
    const keyThrottle = this.keyboard && this.throttleKeys
      ? (this.keys.has('keyw') ? 1 : 0) - (this.keys.has('keys') ? 1 : 0)
      : 0;
    const keyRudder = this.keyboard
      ? (this.keys.has('keyd') ? 1 : 0) - (this.keys.has('keya') ? 1 : 0)
      : 0;
    const wantThrottle = keyThrottle !== 0 ? keyThrottle : this.throttleInput;
    const wantRudder = keyRudder !== 0 ? keyRudder : this.rudderInput;

    this.throttle = approach(this.throttle, wantThrottle, THROTTLE_RATE * dt);
    // The rudder centres itself when released — a real one is pushed back by the
    // water flowing over it, and holding a turn otherwise requires no input at
    // all, which reads as the ship spiralling on its own.
    this.rudder =
      wantRudder === 0
        ? approach(this.rudder, 0, RUDDER_RETURN * dt)
        : approach(this.rudder, wantRudder, RUDDER_RATE * dt);

    // --- hull frame ----------------------------------------------------------
    // Projected flat. The hull pitches and rolls with the swell, and thrust that
    // followed the bow through that would drive the ship into the water on the
    // face of every wave.
    this.body.forward(this.forward);
    this.forward.y = 0;
    if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
    this.forward.normalize();
    this.right.set(-this.forward.z, 0, this.forward.x);

    const velocity = this.body.velocity;
    const alongSpeed = velocity.dot(this.forward);
    const acrossSpeed = velocity.dot(this.right);

    // --- forces --------------------------------------------------------------
    const ahead = this.throttle >= 0 ? 1 : REVERSE_FRACTION;
    // Engine and canvas together never exceed the engine alone: the drag above
    // was tuned for one full-ahead's worth of thrust, and the top speed with it.
    this.drive = clamp(this.throttle * ahead + this.sailDrive, -REVERSE_FRACTION, 1);
    const thrust = this.drive * MAX_THRUST;

    // Quadratic, and signed by direction of travel rather than by speed, so it
    // always opposes motion instead of adding to it when making sternway.
    const dragAlong = -Math.sign(alongSpeed) * DRAG_LONGITUDINAL * alongSpeed * alongSpeed;
    const dragAcross = -Math.sign(acrossSpeed) * DRAG_LATERAL * acrossSpeed * acrossSpeed;

    this.force.set(0, 0, 0);
    this.force.addScaledVector(this.forward, thrust + dragAlong);
    this.force.addScaledVector(this.right, dragAcross);
    this.body.externalForce.copy(this.force);

    // --- steering ------------------------------------------------------------
    // A rudder is a foil: it generates its moment from water flowing past it, so
    // authority rises with speed and a stopped ship cannot turn on the spot.
    // Saturating rather than growing without bound, because past hull speed a
    // bigger rudder angle mostly just stalls the foil.
    const flow = Math.min(1, Math.abs(alongSpeed) / RUDDER_REFERENCE_SPEED);
    // Reversed when making sternway, which is how a real hull behaves and is the
    // one piece of this that surprises people who have never backed a boat.
    const direction = alongSpeed >= 0 ? 1 : -1;
    const steer = -this.rudder * RUDDER_TORQUE * flow * direction;
    const damping = -this.body.angularVelocity.y * YAW_DAMPING;

    this.torque.set(0, steer + damping, 0);
    this.body.externalTorque.copy(this.torque);

    this.applyShoal();
  }

  /**
   * Turns the hull away from shoaling water, and cuts her thrust in it.
   *
   * The direction is the negated horizontal gradient of the seabed, which points
   * downhill — toward deeper water — by construction. Central differences over
   * `SHOAL_PROBE`, which is about a hull length: a finer probe reads the ripples
   * in the relief rather than the slope of the shore, and steers the ship along
   * the sand ridges instead of off them.
   */
  private applyShoal(): void {
    if (!this.floorHeight) return;

    const { x, z } = this.body.object.position;
    const floor = this.floorHeight(x, z);
    const depth = -floor;
    if (depth >= SHOAL_DEPTH) return;

    const strength = Math.min(
      1,
      Math.max(0, (SHOAL_DEPTH - depth) / (SHOAL_DEPTH - SHOAL_HARD_DEPTH)),
    );

    const dx = this.floorHeight(x + SHOAL_PROBE, z) - this.floorHeight(x - SHOAL_PROBE, z);
    const dz = this.floorHeight(x, z + SHOAL_PROBE) - this.floorHeight(x, z - SHOAL_PROBE);
    const length = Math.hypot(dx, dz);

    // On genuinely flat ground the gradient is degenerate and there is no
    // "deeper" to point at. Falling back to pushing the hull back the way she
    // came is not a guess about the terrain — it is the one direction that is
    // certainly navigable, because she was just there.
    if (length < 1e-4) {
      const speed = this.body.velocity.length();
      if (speed > 1e-3) {
        this.force.copy(this.body.velocity).multiplyScalar(-SHOAL_FORCE * strength / speed);
        this.force.y = 0;
        this.body.externalForce.add(this.force);
      }
    } else {
      this.force.set(-dx / length, 0, -dz / length).multiplyScalar(SHOAL_FORCE * strength);
      this.body.externalForce.add(this.force);
    }

    // And take the engine off her. Without this the boundary is a tug of war the
    // player can feel; with it, running aground simply does not work.
    this.body.externalForce.addScaledVector(
      this.forward,
      -Math.max(0, this.drive) * MAX_THRUST * strength,
    );
  }

  /** Live state, for the HUD and for tests. */
  getState(out: ShipControlState): ShipControlState {
    this.body.forward(this.forward);
    this.forward.y = 0;
    if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
    this.forward.normalize();

    out.throttle = this.throttle;
    out.rudder = this.rudder;
    out.speed = this.body.velocity.length();
    out.forwardSpeed = this.body.velocity.dot(this.forward);
    out.heading = Math.atan2(this.forward.z, this.forward.x);
    return out;
  }

  dispose(): void {
    this.abort.abort();
    this.keys.clear();
  }

  // ------------------------------------------------------------------- input

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (!this.enabled || isTypingTarget(event.target)) return;
    const code = event.code.toLowerCase();
    if (!STEERING_KEYS.has(code)) return;
    // Claimed, so the page does not also scroll on the arrow-adjacent bindings.
    event.preventDefault();
    this.keys.add(code);
  };

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    this.keys.delete(event.code.toLowerCase());
  };

  private readonly releaseKeys = (): void => {
    this.keys.clear();
  };
}

const STEERING_KEYS = new Set(['keyw', 'keya', 'keys', 'keyd']);

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/** Moves `current` toward `target` by at most `step`. */
function approach(current: number, target: number, step: number): number {
  const delta = target - current;
  if (Math.abs(delta) <= step) return target;
  return current + Math.sign(delta) * step;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}
