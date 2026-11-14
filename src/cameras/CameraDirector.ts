import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { CameraMode } from '../ui/types';
import {
  type CinematicEnvironment,
  CinematicDirector,
  type CinematicPose,
  type CinematicShipInput,
} from './Cinematic';

/**
 * Modes the director can be put into.
 *
 * A superset of the UI's `CameraMode`, not a replacement for it. `'cinematic'`
 * is a camera rig that exists whether or not anything has drawn a button for it,
 * and widening the union here rather than in `src/ui/types` means the rig and
 * its control can land independently — while `setMode` still accepts everything
 * the UI is already able to send, because `CameraMode` is assignable to this.
 * When the UI grows the fourth mode the two types simply become identical and
 * this alias can go.
 */
export type DirectorMode = CameraMode | 'cinematic';

export interface CameraTarget {
  /** World position the boat camera chases. */
  position: THREE.Vector3;
  /** Heading in radians, `atan2(forward.z, forward.x)`, for the chase camera to sit behind. */
  heading: number;
}

export interface CameraDirectorOptions {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  /** Queries wave height so the camera never sinks through a crest. */
  surfaceHeight: (x: number, z: number) => number;
}

/**
 * Fly speed, and the range the wheel moves it over.
 *
 * 22 m/s is a reasonable default for looking at the ship and useless for
 * anything else in this world: the island is 1.4 km away, so crossing to it at
 * the default takes over a minute, and once there the same speed is far too
 * quick to look at a tree with. A fixed speed plus a boost key cannot serve both
 * — which is why every editor fly camera in the industry puts the speed on the
 * wheel.
 *
 * Geometric steps, not linear ones. The useful range spans two and a half
 * orders of magnitude, and a linear step that is sensible at 200 m/s is
 * imperceptible at 2. Each notch is a fixed *ratio*, so the control feels the
 * same everywhere in the range.
 */
/**
 * The chase rig's orbit, and why it has one at all.
 *
 * The camera used to be a pure function of the hull: stern bearing, fixed
 * distance, fixed height. That is a good default framing and a bad *only*
 * framing — at the helm the viewer could not look at the island they were
 * sailing toward, could not see what was off the bow, and could not look at
 * their own ship. Every third-person game solves it the same way, by letting the
 * mouse orbit the follow point while the rig keeps following.
 *
 * The offsets recentre when the mouse is released, on a 2.2 s constant. Games
 * split on this and both answers are defensible: leaving the camera where it was
 * put is predictable, and returning it means a viewer who looked over their
 * shoulder in a chase does not then sail for a minute facing sideways. The
 * return is slow enough to read as the camera settling rather than as the game
 * taking the controls back, and it is suspended entirely while a drag is live.
 */
/**
 * Focus distance used where a mode has nothing to focus on, metres.
 *
 * Far enough that a wide lens at the apertures this project uses is effectively
 * focused at infinity — so "no answer" renders as a sharp frame rather than as
 * an arbitrary plane of sharpness somewhere in the middle of the sea.
 */
const DEFAULT_FOCUS_DISTANCE = 400;

const CHASE_DISTANCE = 34;
const CHASE_HEIGHT = 14;
const CHASE_DISTANCE_MIN = 12;
const CHASE_DISTANCE_MAX = 120;
const CHASE_SENSITIVITY = 0.0052;
/** Just short of straight down and straight up the mast. */
const CHASE_PITCH_MIN = -0.5;
const CHASE_PITCH_MAX = 1.15;
const CHASE_RECENTRE_TAU = 2.2;

const FLY_SPEED = 22;
const FLY_SPEED_MIN = 1.5;
const FLY_SPEED_MAX = 600;
const FLY_SPEED_STEP = 1.18;
const FLY_BOOST = 5;
const FLY_DAMPING = 6;
const MOUSE_SENSITIVITY = 0.0022;

/**
 * Owns the camera across four modes and, critically, owns the transitions
 * between them — cutting instantly between an orbit rig and a chase rig is the
 * single most jarring thing a demo like this can do, so every switch is a timed
 * ease from the current pose to the new one.
 */
export class CameraDirector {
  readonly camera: THREE.PerspectiveCamera;
  readonly orbit: OrbitControls;

  private mode: DirectorMode = 'orbit';
  private readonly domElement: HTMLElement;
  private readonly surfaceHeight: (x: number, z: number) => number;

  // Fly state
  private readonly keys = new Set<string>();
  /** Metres per second the fly camera accelerates at. Driven by the wheel. */
  private flySpeed = FLY_SPEED;
  /** Chase-rig orbit offsets from the stern bearing, and its dolly. */
  private chaseYaw = 0;
  private chasePitch = 0;
  private chaseDistance = CHASE_DISTANCE;
  private chaseDragging = false;
  private yaw = 0;
  private pitch = 0;
  private pointerLocked = false;
  private readonly flyVelocity = new THREE.Vector3();

  // Chase state
  private target: CameraTarget | null = null;

  /**
   * Render-only jitter on the chase rig, set by the gameplay layer.
   *
   * Applied on top of the solved pose at the end of `update` and taken off again
   * at the start of the next one, so the follow filter never integrates it — a
   * shake that leaked into the lag would make the rig drift off the hull after
   * every slam. The removal checks that the camera still holds the pose this
   * class left on it, so a `pin` or a mode change in between is not undone.
   */
  private readonly shakeOffset = new THREE.Vector3();
  private readonly shakeEuler = new THREE.Euler(0, 0, 0, 'YXZ');
  private readonly shakeQuat = new THREE.Quaternion();
  private readonly unshakenPosition = new THREE.Vector3();
  private readonly unshakenQuaternion = new THREE.Quaternion();
  private readonly shakenPosition = new THREE.Vector3();
  private readonly shakenQuaternion = new THREE.Quaternion();
  private shakeApplied = false;

  // Cinematic state
  private readonly cinematic: CinematicDirector;
  private readonly cinematicPose: CinematicPose = {
    position: new THREE.Vector3(),
    target: new THREE.Vector3(),
    fov: 50,
    roll: 0,
  };
  /**
   * The lens every mode but Cinematic uses, degrees: whatever the camera was
   * built with. The tour changes focal length shot to shot, and leaving it has
   * to hand this back — eased with the rest of the pose rather than snapped.
   * `setLens` can change it; `defaultFov` is what it started as.
   */
  private baseFov: number;
  private readonly defaultFov: number;
  /**
   * A pose held in the hull's heading frame instead of the chase rig's, or null.
   * See `pinToShip`.
   */
  private shipPin: { position: THREE.Vector3; target: THREE.Vector3 } | null = null;
  private fromFov = 0;
  private readonly rollQuat = new THREE.Quaternion();
  private readonly motionQuery: MediaQueryList | null;
  /**
   * The hull orders the active mode is issuing, republished every frame.
   *
   * Zero in every mode but Cinematic, and zeroed the instant Cinematic is left,
   * so a caller that forwards this unconditionally into
   * `ShipController.setInput` can never leave the hull carrying an order the
   * viewer did not give it.
   */
  private readonly shipOrders: CinematicShipInput = { throttle: 0, rudder: 0 };

  // Transition state
  private transition = 0;
  private readonly fromPosition = new THREE.Vector3();
  private readonly fromQuaternion = new THREE.Quaternion();
  private static readonly TRANSITION_SECONDS = 0.85;

  // Scratch — the update path must not allocate.
  private readonly tmpVec = new THREE.Vector3();
  private readonly tmpVec2 = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
  private readonly tmpEuler = new THREE.Euler(0, 0, 0, 'YXZ');
  private readonly desiredPosition = new THREE.Vector3();
  private readonly desiredQuaternion = new THREE.Quaternion();

  private readonly abort = new AbortController();

  constructor(options: CameraDirectorOptions) {
    this.camera = options.camera;
    this.domElement = options.domElement;
    this.surfaceHeight = options.surfaceHeight;
    this.baseFov = this.camera.fov;
    this.defaultFov = this.camera.fov;
    this.fromFov = this.camera.fov;
    // The tour lifts itself over the crests on the same wave query the chase
    // camera uses, and over the terrain on the heightfield's own function.
    this.cinematic = new CinematicDirector({ surfaceHeight: options.surfaceHeight });
    this.motionQuery =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : null;
    this.cinematic.setReducedMotion(this.motionQuery?.matches ?? false);

    this.orbit = new OrbitControls(this.camera, this.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.06;
    this.orbit.minDistance = 3;
    this.orbit.maxDistance = 1200;
    this.orbit.target.set(0, 2, 0);
    // Allow going below the surface — the underwater view is a headline feature.
    this.orbit.maxPolarAngle = Math.PI;

    const { signal } = this.abort;
    window.addEventListener('keydown', this.onKeyDown, { signal });
    window.addEventListener('keyup', this.onKeyUp, { signal });
    this.domElement.addEventListener('mousedown', this.onMouseDown, { signal });
    document.addEventListener('pointerlockchange', this.onPointerLockChange, { signal });
    document.addEventListener('mousemove', this.onMouseMove, { signal });
    // Not passive: the wheel has to stop the page scrolling under the canvas
    // while the fly camera is using it.
    this.domElement.addEventListener('wheel', this.onWheel, { signal, passive: false });
    window.addEventListener('mouseup', this.onMouseUp, { signal });
    this.motionQuery?.addEventListener(
      'change',
      (event) => this.cinematic.setReducedMotion(event.matches),
      { signal },
    );
  }

  get currentMode(): DirectorMode {
    return this.mode;
  }

  /**
   * Throttle and rudder the active mode wants from the hero ship this frame.
   *
   * Forward it to `ShipController.setInput` unconditionally, every frame, in
   * every mode — the zeroes the other three modes publish are what release the
   * hull when Cinematic ends. Reading it only while Cinematic is active leaves
   * the controller holding the last order the flight issued, which it will act
   * on the moment Boat mode enables it again.
   *
   * Live object; do not retain it.
   */
  get shipInput(): Readonly<CinematicShipInput> {
    return this.shipOrders;
  }

  /** Beat name the cinematic flight is playing, for a HUD label. */
  get cinematicBeat(): string {
    return this.cinematic.beatName;
  }

  /** Position on the cinematic loop in seconds. Meaningless in other modes. */
  get cinematicTime(): number {
    return this.cinematic.time;
  }

  /**
   * The world the cinematic tour wants this frame — hour, rain, cloud, fog.
   * Meaningless in other modes; see `Cinematic.environment`.
   *
   * `time` defaults to the live clock. Passing it explicitly is what lets a
   * deterministic reset ask what the world *will* look like at the moment it is
   * rewinding to, before it seeds the effects that accumulate.
   */
  cinematicEnvironment(time?: number): Readonly<CinematicEnvironment> {
    return this.cinematic.environment(time);
  }

  /**
   * Distance from the eye to whatever this mode is looking at, metres.
   *
   * Per mode, because "what is this shot about" is a different question in each
   * one, and there is no single answer a lens could be focused on. Orbit and
   * Cinematic both carry an explicit look point; the chase rig has the hull;
   * Fly has neither, so it takes where the view ray meets the sea, which is what
   * a viewer flying over water is almost always looking at.
   *
   * Closed-form in every mode, with no filtering against the previous frame.
   * That is required rather than tidy: `DepthOfField` reads this every frame,
   * and a focus that drifted toward its target over time would make a capture
   * depend on how many frames preceded it — the property the whole visual
   * harness rests on.
   */
  focusDistance(): number {
    switch (this.mode) {
      case 'orbit':
        return Math.max(1, this.camera.position.distanceTo(this.orbit.target));
      case 'cinematic':
        return Math.max(1, this.camera.position.distanceTo(this.cinematicPose.target));
      case 'boat':
        return this.target
          ? Math.max(1, this.camera.position.distanceTo(this.target.position))
          : DEFAULT_FOCUS_DISTANCE;
      case 'fly': {
        this.camera.getWorldDirection(this.tmpVec);
        // Looking level or up: nothing to focus on but the sky, which a wide
        // lens at any sane aperture renders sharp anyway.
        if (this.tmpVec.y > -0.02) return DEFAULT_FOCUS_DISTANCE;
        const surface = this.surfaceHeight(this.camera.position.x, this.camera.position.z);
        const drop = this.camera.position.y - surface;
        if (drop <= 0) return DEFAULT_FOCUS_DISTANCE;
        return THREE.MathUtils.clamp(drop / -this.tmpVec.y, 1, 4000);
      }
    }
  }

  /**
   * Sets the chase rig's shake for the next frames: a camera-space offset in
   * metres and pitch/yaw/roll in radians. Zeroes clear it. Ignored outside Boat.
   */
  setShake(offset: THREE.Vector3, pitch: number, yaw: number, roll: number): void {
    this.shakeOffset.copy(offset);
    this.shakeEuler.set(pitch, yaw, roll, 'YXZ');
  }

  setChaseTarget(target: CameraTarget | null): void {
    this.target = target;
    if (target === null) this.shipPin = null;
  }

  setMode(mode: DirectorMode): void {
    if (mode === this.mode) return;
    this.shipPin = null;

    // Capture the current pose so the new rig can be eased into rather than cut to.
    this.fromPosition.copy(this.camera.position);
    this.fromQuaternion.copy(this.camera.quaternion);
    this.fromFov = this.camera.fov;
    this.transition = 1;

    if (this.mode === 'fly') this.exitPointerLock();
    if (this.mode === 'cinematic') {
      // Released here, not on the next update, so the orders are already zero
      // for whatever reads them first after the switch.
      this.cinematic.setEnabled(false);
      this.shipOrders.throttle = 0;
      this.shipOrders.rudder = 0;
    }
    this.mode = mode;

    if (mode === 'orbit') {
      // Re-seat the orbit target ahead of the camera so the first drag does not
      // whip the view around to a stale pivot.
      this.camera.getWorldDirection(this.tmpVec);
      this.orbit.target.copy(this.camera.position).addScaledVector(this.tmpVec, 40);
      this.orbit.update();
    }

    if (mode === 'fly') {
      this.tmpEuler.setFromQuaternion(this.camera.quaternion);
      this.yaw = this.tmpEuler.y;
      this.pitch = this.tmpEuler.x;
      this.flyVelocity.set(0, 0, 0);
    }

    // Rewinds the flight to its opening beat. The pose it opens on is not where
    // the camera currently is, which is exactly what the transition above is
    // for: the ease runs from wherever the viewer left the previous rig onto the
    // flight's first mark, so entering the mode is a move rather than a cut.
    if (mode === 'cinematic') this.cinematic.setEnabled(true);

    this.orbit.enabled = mode === 'orbit';
  }

  /**
   * Places the camera exactly, in whatever mode is active, and synchronises that
   * mode's internal state so the next `update()` does not immediately undo it.
   *
   * A plain `camera.position.set` is not enough for any mode but Orbit: Fly
   * integrates from its own yaw/pitch and would snap back on the next frame, and
   * the chase rig damps toward a pose derived from the ship. Each mode therefore
   * needs its state re-derived from the requested pose, which is what this does.
   *
   * Also cancels any in-flight mode transition — a capture taken mid-ease is not
   * reproducible.
   */
  pin(position: THREE.Vector3, target: THREE.Vector3): void {
    this.transition = 0;
    this.shipPin = null;

    this.camera.position.copy(position);
    this.tmpQuat.setFromRotationMatrix(lookAtMatrix(position, target, UP));
    this.camera.quaternion.copy(this.tmpQuat);
    this.camera.updateMatrixWorld(true);

    this.desiredPosition.copy(position);
    this.desiredQuaternion.copy(this.tmpQuat);
    this.applyFov(this.mode === 'cinematic' ? (this.cinematicPose.fov ?? this.baseFov) : this.baseFov);

    switch (this.mode) {
      case 'orbit':
        this.orbit.target.copy(target);
        this.orbit.update();
        break;
      case 'fly':
        // Re-derive the integrator's yaw/pitch from the pose it must hold.
        this.tmpEuler.setFromQuaternion(this.tmpQuat);
        this.yaw = this.tmpEuler.y;
        this.pitch = this.tmpEuler.x;
        this.flyVelocity.set(0, 0, 0);
        break;
      case 'boat':
        // Nothing to seed: the chase pose is derived from the target each frame.
        // `snapToTarget` is the deterministic entry point for this mode.
        break;
      case 'cinematic':
        // Nothing to seed either, and for a stronger reason: the flight's pose
        // is a function of its clock, and no clock reading corresponds to an
        // arbitrary requested pose. A pin here therefore survives exactly until
        // the next `update` re-samples the curve. `resetCinematic` is the
        // deterministic entry point for this mode — it pins *and* moves the
        // clock, which is the only way the two can agree.
        break;
    }
  }

  /**
   * The vertical field of view, degrees, for every mode but Cinematic; `null`
   * restores the one the camera was built with. For the screenshot harness,
   * which frames each shot with its own lens: nothing in the app calls it, so a
   * viewer's lens never changes under them.
   */
  setLens(fov: number | null): void {
    this.baseFov = fov !== null && Number.isFinite(fov) ? fov : this.defaultFov;
    if (this.mode !== 'cinematic') this.applyFov(this.baseFov);
  }

  /**
   * Holds the camera at a pose fixed to the hull's heading, in place of the
   * chase rig, until the mode changes or another pin replaces it. Boat mode
   * only; false otherwise, or before there is a hull.
   *
   * `position` and `target` are metres ahead of the hull (x), above sea level
   * (y) and to starboard (z). Level whatever the hull's pitch and roll, and with
   * no lag and no shake: a camera boat running alongside, which is how a ship
   * under way is photographed. For the screenshot harness — the chase rig can
   * only ever see the hull from astern, and a pin made after a settle would be
   * one no frame had been updated for: the sky, the fog and the particles all
   * read the camera in `update`.
   */
  pinToShip(position: THREE.Vector3, target: THREE.Vector3): boolean {
    if (this.mode !== 'boat' || !this.target) return false;
    this.shipPin = { position: position.clone(), target: target.clone() };
    this.transition = 0;
    this.updateShipPin();
    return true;
  }

  private updateShipPin(): void {
    const pin = this.shipPin;
    const target = this.target;
    if (!pin || !target) return;
    // Bow along (cos, sin) of the heading; starboard is forward x up.
    const c = Math.cos(target.heading);
    const s = Math.sin(target.heading);
    const origin = target.position;
    const place = (out: THREE.Vector3, p: THREE.Vector3): THREE.Vector3 =>
      out.set(origin.x + c * p.x - s * p.z, p.y, origin.z + s * p.x + c * p.z);
    place(this.desiredPosition, pin.position);
    place(this.tmpVec2, pin.target);
    this.desiredQuaternion.setFromRotationMatrix(lookAtMatrix(this.desiredPosition, this.tmpVec2, UP));
    this.camera.position.copy(this.desiredPosition);
    this.camera.quaternion.copy(this.desiredQuaternion);
    this.camera.updateMatrixWorld(true);
  }

  /**
   * Places the chase camera at its ideal pose for the current target with no
   * damping, so a capture does not depend on how many frames the rig has had to
   * converge.
   */
  snapToTarget(): void {
    if (this.mode !== 'boat' || !this.target) return;
    this.transition = 0;
    this.updateChase(0);
    this.camera.position.copy(this.desiredPosition);
    this.camera.quaternion.copy(this.desiredQuaternion);
    this.camera.updateMatrixWorld(true);
  }

  /**
   * Places the camera at exactly the pose the cinematic flight holds at `time`,
   * with no transition and no damping, and leaves its clock there.
   *
   * The counterpart to `snapToTarget` for this mode, and the reason the flight
   * carries no per-frame state: a capture taken after this call depends on
   * `time` and on nothing else — not on how many frames the mode has run, not on
   * what the previous capture did.
   */
  resetCinematic(time = 0): void {
    if (this.mode !== 'cinematic') return;
    this.transition = 0;
    this.cinematic.resetClock(time);
    this.updateCinematic(0);
    this.camera.position.copy(this.desiredPosition);
    this.camera.quaternion.copy(this.desiredQuaternion);
    this.applyFov(this.cinematicPose.fov ?? this.baseFov);
    this.camera.updateMatrixWorld(true);
  }

  update(dt: number): void {
    this.removeShake();
    switch (this.mode) {
      case 'orbit':
        this.updateOrbit();
        break;
      case 'fly':
        this.updateFly(dt);
        break;
      case 'boat':
        if (this.shipPin) this.updateShipPin();
        else this.updateChase(dt);
        break;
      case 'cinematic':
        this.updateCinematic(dt);
        break;
    }

    const desiredFov =
      this.mode === 'cinematic' ? (this.cinematicPose.fov ?? this.baseFov) : this.baseFov;
    if (this.transition > 0) {
      this.transition = Math.max(0, this.transition - dt / CameraDirector.TRANSITION_SECONDS);
      const t = easeInOutCubic(1 - this.transition);
      this.camera.position.lerpVectors(this.fromPosition, this.desiredPosition, t);
      this.camera.quaternion.slerpQuaternions(this.fromQuaternion, this.desiredQuaternion, t);
      this.applyFov(this.fromFov + (desiredFov - this.fromFov) * t);
    } else {
      this.applyFov(desiredFov);
    }

    this.applyShake();
  }

  private applyShake(): void {
    if (this.mode !== 'boat' || this.shipPin) return;
    const e = this.shakeEuler;
    if (this.shakeOffset.lengthSq() === 0 && e.x === 0 && e.y === 0 && e.z === 0) return;
    this.unshakenPosition.copy(this.camera.position);
    this.unshakenQuaternion.copy(this.camera.quaternion);
    this.shakeQuat.setFromEuler(e);
    this.camera.quaternion.multiply(this.shakeQuat);
    this.camera.position.add(
      this.tmpVec.copy(this.shakeOffset).applyQuaternion(this.unshakenQuaternion),
    );
    this.shakenPosition.copy(this.camera.position);
    this.shakenQuaternion.copy(this.camera.quaternion);
    this.shakeApplied = true;
  }

  private removeShake(): void {
    if (!this.shakeApplied) return;
    this.shakeApplied = false;
    if (
      !this.camera.position.equals(this.shakenPosition) ||
      !this.camera.quaternion.equals(this.shakenQuaternion)
    ) {
      return;
    }
    this.camera.position.copy(this.unshakenPosition);
    this.camera.quaternion.copy(this.unshakenQuaternion);
  }

  private updateOrbit(): void {
    this.orbit.update();
    this.desiredPosition.copy(this.camera.position);
    this.desiredQuaternion.copy(this.camera.quaternion);
  }

  private updateFly(dt: number): void {
    this.tmpEuler.set(this.pitch, this.yaw, 0, 'YXZ');
    this.desiredQuaternion.setFromEuler(this.tmpEuler);

    const forward = this.tmpVec.set(0, 0, -1).applyQuaternion(this.desiredQuaternion);
    const right = this.tmpVec2.set(1, 0, 0).applyQuaternion(this.desiredQuaternion);

    const boost = this.keys.has('shiftleft') || this.keys.has('shiftright') ? FLY_BOOST : 1;
    const speed = this.flySpeed * boost;

    // Accelerate toward the requested direction, then damp — instant velocity
    // changes make a flying camera feel like a cursor rather than a body.
    if (this.keys.has('keyw')) this.flyVelocity.addScaledVector(forward, speed * dt);
    if (this.keys.has('keys')) this.flyVelocity.addScaledVector(forward, -speed * dt);
    if (this.keys.has('keyd')) this.flyVelocity.addScaledVector(right, speed * dt);
    if (this.keys.has('keya')) this.flyVelocity.addScaledVector(right, -speed * dt);
    if (this.keys.has('space')) this.flyVelocity.y += speed * dt;
    if (this.keys.has('controlleft')) this.flyVelocity.y -= speed * dt;

    this.flyVelocity.multiplyScalar(Math.max(0, 1 - FLY_DAMPING * dt));

    this.desiredPosition.copy(this.camera.position).addScaledVector(this.flyVelocity, dt);

    if (this.transition === 0) {
      this.camera.position.copy(this.desiredPosition);
      this.camera.quaternion.copy(this.desiredQuaternion);
    }
  }

  private updateChase(dt: number): void {
    if (!this.target) {
      this.desiredPosition.copy(this.camera.position);
      this.desiredQuaternion.copy(this.camera.quaternion);
      return;
    }

    // Recentre toward the stern whenever the viewer is not driving it. Framed as
    // an exponential rather than a lerp so the rate is the same at any frame
    // rate — the same reason the follow below is one.
    if (!this.chaseDragging) {
      const settle = 1 - Math.exp(-dt / CHASE_RECENTRE_TAU);
      this.chaseYaw -= this.chaseYaw * settle;
      this.chasePitch -= this.chasePitch * settle;
    }

    // Sit behind and above the target, looking slightly down at it — with the
    // viewer's orbit applied on top of the hull's own bearing, so the rig keeps
    // following while the camera can point anywhere.
    //
    // `heading` is `atan2(forward.z, forward.x)` — the hull's and the wake's
    // convention — so the bow points along (cos, sin). This read (sin, cos)
    // from the start, which put the rig astern only on a heading of 45 degrees
    // and off the beam or the bow on every other: at the spawn heading the
    // "chase" camera watched the ship sail across the frame. The yaw offset is
    // subtracted so a drag still orbits the same way it always did.
    const bearing = this.target.heading - this.chaseYaw;
    const lift = CHASE_HEIGHT / CHASE_DISTANCE + this.chasePitch;
    const planar = Math.cos(Math.atan(lift));
    const back = this.tmpVec.set(
      Math.cos(bearing) * planar,
      0,
      Math.sin(bearing) * planar,
    );
    this.desiredPosition
      .copy(this.target.position)
      .addScaledVector(back, -this.chaseDistance)
      .add(this.tmpVec2.set(0, this.chaseDistance * lift * planar, 0));

    // Never below the sea. A rig orbited to a low pitch in a seaway would
    // otherwise dip through the surface every swell, which reads as the camera
    // being swamped rather than as a low angle.
    const surface = this.surfaceHeight(this.desiredPosition.x, this.desiredPosition.z);
    if (this.desiredPosition.y < surface + 1.6) this.desiredPosition.y = surface + 1.6;

    this.tmpVec2.copy(this.target.position).y += 4;
    this.tmpQuat.setFromRotationMatrix(
      lookAtMatrix(this.desiredPosition, this.tmpVec2, UP),
    );
    this.desiredQuaternion.copy(this.tmpQuat);

    if (this.transition === 0) {
      // Critically damped follow so the camera lags the boat a little in chop
      // without ever overshooting into a wobble.
      const lag = 1 - Math.exp(-4 * dt);
      this.camera.position.lerp(this.desiredPosition, lag);
      this.camera.quaternion.slerp(this.desiredQuaternion, lag);
    }
  }

  /**
   * Samples the authored flight and republishes its hull orders.
   *
   * Note what is *not* here: no damping toward the sampled pose. Every other rig
   * in this class lags its target, because every other rig is following
   * something it does not control — a mouse, a key, a hull in a seaway. The
   * flight already contains its own easing, in the shape of the curve, and a lag
   * filter on top would be a second-order term with memory: the pose after a
   * `resetCinematic` would then depend on where the camera happened to be
   * standing beforehand, which is precisely the determinism the mode is built
   * around.
   */
  private updateCinematic(dt: number): void {
    // The live hull, so the shots framed on the ship frame the ship rather than
    // where the flight expected it to be. `target` is republished every frame by
    // main whatever the mode, and is null only until the ship has loaded, when
    // the tour falls back on its nominal circuit.
    const orders = this.cinematic.update(dt, this.cinematicPose, this.target);
    this.shipOrders.throttle = orders.throttle;
    this.shipOrders.rudder = orders.rudder;

    this.desiredPosition.copy(this.cinematicPose.position);
    this.tmpQuat.setFromRotationMatrix(
      lookAtMatrix(this.cinematicPose.position, this.cinematicPose.target, UP),
    );
    // Dutch angle and handheld roll, about the view axis (camera −z).
    const roll = this.cinematicPose.roll ?? 0;
    if (roll !== 0) this.tmpQuat.multiply(this.rollQuat.setFromAxisAngle(VIEW_AXIS, roll));
    this.desiredQuaternion.copy(this.tmpQuat);

    if (this.transition === 0) {
      this.camera.position.copy(this.desiredPosition);
      this.camera.quaternion.copy(this.desiredQuaternion);
    }
  }

  /**
   * How submerged the camera is, 0..1, with a soft band around the surface.
   *
   * This ramp is the *only* thing standing between the camera and a hard cut at
   * the waterline, and that is deliberate.
   *
   * There used to be an `avoidSurfacePenetration` step that teleported the camera
   * out of a 0.35 m band either side of the surface, on the theory that it stopped
   * the underwater state flickering at a crest. What it actually built was a wall:
   * approaching the surface from above put the camera inside the band, which
   * snapped it back to `surface + 0.35`, and since no plausible frame moves the
   * camera 0.7 m at once there was no way through in either direction. Diving and
   * surfacing — the headline feature the fly camera exists for — were impossible
   * except by teleport.
   *
   * The flicker it guarded against is not real either. The state is not a boolean;
   * it is this value, and it cross-fades over the whole 0.7 m band, so a crest
   * passing the lens moves it smoothly rather than toggling anything.
   */
  submersion(): number {
    const surface = this.surfaceHeight(this.camera.position.x, this.camera.position.z);
    const depth = surface - this.camera.position.y;
    return THREE.MathUtils.clamp(depth / 0.7 + 0.5, 0, 1);
  }

  dispose(): void {
    this.abort.abort();
    this.orbit.dispose();
    this.exitPointerLock();
    // Zeroes the published orders on the way out, for the same reason leaving
    // the mode does: a disposed director must not be the last thing that told
    // the hull to open its throttle.
    this.cinematic.setEnabled(false);
    this.shipOrders.throttle = 0;
    this.shipOrders.rudder = 0;
  }

  // ------------------------------------------------------------------ input

  private onKeyDown = (event: KeyboardEvent): void => {
    if (isTypingTarget(event.target)) return;
    this.keys.add(event.code.toLowerCase());
  };

  private onKeyUp = (event: KeyboardEvent): void => {
    this.keys.delete(event.code.toLowerCase());
  };

  private onMouseDown = (): void => {
    if (this.mode === 'fly' && !this.pointerLocked) {
      void this.domElement.requestPointerLock?.();
    }
    // Boat mode drags to orbit. No pointer lock: the helm is a mode a viewer
    // sits in for minutes at a time and locking the cursor there would trap it.
    if (this.mode === 'boat') this.chaseDragging = true;
  };

  private onMouseUp = (): void => {
    this.chaseDragging = false;
  };

  private onPointerLockChange = (): void => {
    this.pointerLocked = document.pointerLockElement === this.domElement;
  };

  private onMouseMove = (event: MouseEvent): void => {
    if (this.mode === 'boat') {
      if (!this.chaseDragging) return;
      this.chaseYaw -= event.movementX * CHASE_SENSITIVITY;
      this.chasePitch += event.movementY * CHASE_SENSITIVITY;
      this.chasePitch = THREE.MathUtils.clamp(this.chasePitch, CHASE_PITCH_MIN, CHASE_PITCH_MAX);
      return;
    }
    if (this.mode !== 'fly' || !this.pointerLocked) return;
    this.yaw -= event.movementX * MOUSE_SENSITIVITY;
    this.pitch -= event.movementY * MOUSE_SENSITIVITY;
    // Stop just short of vertical; passing it would flip the horizon.
    const limit = Math.PI / 2 - 0.01;
    this.pitch = THREE.MathUtils.clamp(this.pitch, -limit, limit);
  };

  /**
   * Wheel changes fly speed; every other mode leaves the event alone.
   *
   * Orbit mode is the one that must not be touched: `OrbitControls` owns the
   * wheel there for dolly, and swallowing it would break zoom. So this returns
   * without calling `preventDefault` unless the fly camera is actually the thing
   * being driven — a guard, not a global capture.
   */
  private onWheel = (event: WheelEvent): void => {
    if (this.mode === 'boat') {
      event.preventDefault();
      const steps = event.deltaY > 0 ? 1 : -1;
      this.chaseDistance = THREE.MathUtils.clamp(
        this.chaseDistance * Math.pow(1.14, steps),
        CHASE_DISTANCE_MIN,
        CHASE_DISTANCE_MAX,
      );
      return;
    }
    if (this.mode !== 'fly') return;
    event.preventDefault();
    // Only the sign is used. Wheel deltas are not comparable between a mouse, a
    // trackpad and a browser's `deltaMode`, so a notch is a notch.
    const steps = event.deltaY > 0 ? -1 : 1;
    this.setFlySpeed(this.flySpeed * Math.pow(FLY_SPEED_STEP, steps));
  };

  /** Fly speed in m/s, clamped to the useful range. */
  setFlySpeed(value: number): void {
    this.flySpeed = THREE.MathUtils.clamp(value, FLY_SPEED_MIN, FLY_SPEED_MAX);
  }

  /** Current fly speed in m/s, for a HUD readout. */
  get flySpeedValue(): number {
    return this.flySpeed;
  }

  /** Sets the lens, touching the projection only when it actually changes. */
  private applyFov(fov: number): void {
    if (!Number.isFinite(fov) || Math.abs(this.camera.fov - fov) < 1e-4) return;
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
  }

  private exitPointerLock(): void {
    if (document.pointerLockElement === this.domElement) document.exitPointerLock();
  }
}

const UP = new THREE.Vector3(0, 1, 0);
const VIEW_AXIS = new THREE.Vector3(0, 0, -1);
const scratchMatrix = new THREE.Matrix4();

function lookAtMatrix(eye: THREE.Vector3, target: THREE.Vector3, up: THREE.Vector3): THREE.Matrix4 {
  return scratchMatrix.lookAt(eye, target, up);
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}
