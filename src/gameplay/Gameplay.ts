import * as THREE from 'three/webgpu';
import { BuoyantBody, createRadialProbes } from '../physics/Buoyancy';
import type { ShipController } from '../physics/ShipController';
import type { OceanSampler } from '../ocean/Sampler';
import { mulberry32 } from '../core/random';
import { BowSpray, Burst, ObjectiveBeacon } from './effects';
import {
  IRONS_DEG,
  KNOTS_PER_MS,
  angleOffWind,
  approach,
  compassFromHeading,
  compassFromVector,
  pointOfSail,
  polarDrive,
  smoothstep,
  windDriveFactor,
  windFromCompass,
  type PointOfSail,
} from './sailing';
import {
  NULL_UI_SINK,
  type GameAudio,
  type GameplayAudioState,
  type GameObjective,
  type GameUiSink,
  type SailSetting,
} from './types';

/**
 * Helm-mode gameplay: sails that answer to the wind, an anchor, and cargo to
 * recover.
 *
 * **Everything here is a force or a report.** Like `ShipController`, this never
 * writes the hull's pose. The sails hand the controller a drive fraction, and
 * heel, leeway and the anchor rode are added to the body's external force and
 * torque *after* the controller has set them and *before* the buoyancy solver
 * integrates them — so they are felt by the same substeps that float the ship,
 * and the hull heels against its own righting moment instead of being tilted.
 *
 * **Only in Helm mode.** Leaving it hides the cargo, clears the HUD, returns the
 * W/S keys to the engine and zeroes every force this class contributes. The
 * cinematic tour shows none of it.
 *
 * **Deterministic.** Every random choice comes from one seeded generator, every
 * clock is the simulation's, and `reset()` returns all of it — sails, anchor,
 * score, the cargo sequence, the shake — to the state at construction. The
 * capture harness calls it from `resetDeterministic`.
 */

/* --------------------------------------------------------------- constants */

/** Canvas fraction per sail setting. */
const SAIL_FILL: Record<SailSetting, number> = { furled: 0, half: 0.55, full: 1 };
const SAIL_ORDER: readonly SailSetting[] = ['furled', 'half', 'full'];
/** How fast canvas is set or taken in, fraction per second. Crew, not a switch. */
const SAIL_RATE = 0.45;
/**
 * Drive at full sail on the best point of sail, as a fraction of the engine.
 *
 * The controller's drag gives about 15 knots for a full engine; a laden
 * square-rigger at her best makes ten to twelve, and this puts the top of the
 * curve there, with waves and leeway adding a knot either way.
 */
const SAIL_POWER = 0.6;
/**
 * Fraction of the drive left while anchored. The crew spills the wind rather
 * than let the canvas tear the anchor out — which also lets her swing.
 */
const ANCHORED_SAIL = 0.25;

/**
 * Peak heeling moment, N·m, at full sail on a beam reach in a fresh breeze.
 *
 * Sized against the hull's own righting stiffness — the probe springs at the
 * ship's beam — for about seven degrees, which is what a laden square-rigger
 * shows and what reads at chase-camera distance without looking like a capsize.
 */
const HEEL_TORQUE = 280_000;
/** Peak sideways push from the canvas, N. The keel turns most of it into a slight crab. */
const LEEWAY_FORCE = 9_000;

/** Anchor rode: stiffness and damping once taut, and the force cap. */
const RODE_STIFFNESS = 260_000;
const RODE_DAMPING = 160_000;
const RODE_MAX_FORCE = 1_400_000;
/** After the rode snubs: how long, and by how much, its turning moment is boosted. */
const SNUB_SECONDS = 2.5;
const SNUB_SWING = 3.5;
/** Seconds to weigh anchor once ordered. */
const WEIGH_SECONDS = 3.5;
/** Extra hull drag while anchored, as a fraction of mass per second. */
const ANCHOR_DRAG = 0.05;
/**
 * Wind on the hull and rigging while at anchor, N per (m/s)². Applied abaft
 * amidships so a ship hanging off her bow swings head to wind, which is what
 * anchored ships do and what makes "the anchor swung her round" visible.
 */
const WINDAGE = 190;

/** Cargo layout: distance from the previous target and the area it stays in. */
const SPAWN_MIN = 150;
const SPAWN_SPAN = 190;
const SPAWN_LIMIT = 900;
/** Water this deep at the target and around it, so it can be sailed to. */
const SPAWN_DEPTH = 14;
const SPAWN_RING_DEPTH = 10;
const SPAWN_RING_RADIUS = 32;
/** Margin, metres, beyond the hull's half-beam within which cargo is taken. */
const PICKUP_MARGIN = 4;

/**
 * Trauma a full-strength impact adds, by kind. A slam is felt through the whole
 * hull; a beam slap mostly heard; green water over the bow is the worst of the
 * three. Strength itself comes from `physics/Spray`, which owns impact
 * detection — this layer only decides how much of it reaches the camera.
 */
const IMPACT_TRAUMA = { slam: 0.42, slap: 0.16, wash: 0.55, float: 0 } as const;
/** Depth under the hull, metres, at which she is touching. Above `SHOAL_HARD_DEPTH` in the controller. */
const GROUND_DEPTH = 6;
/** Trauma lost per second. */
const TRAUMA_DECAY = 1.1;
/** Maximum shake at trauma 1: radians and metres. */
const SHAKE_PITCH = THREE.MathUtils.degToRad(0.9);
const SHAKE_YAW = THREE.MathUtils.degToRad(0.5);
const SHAKE_ROLL = THREE.MathUtils.degToRad(1.3);
const SHAKE_LIFT = 0.12;

/** Seconds between HUD pushes for the continuous readouts. */
const UI_INTERVAL = 0.1;

/**
 * What the cask turns out to be. Six, and it must stay six: the seeded
 * generator picks by index, so a longer list would move every later cargo
 * the deterministic runs depend on.
 */
const CARGO_NAMES = [
  'a cask of dark rum',
  'a cask of cloves',
  "a merchant's lost cask",
  'a cask sealed in wax',
  'a cask of good tea',
  'a cask of salt pork',
] as const;

/**
 * The line a recovered cask gets, chosen by the running count rather than the
 * seeded generator, for the same reason: flavour must not move the cargo.
 */
const HAUL_LINES = [
  'Hauled aboard',
  'Safe in the hold',
  'Fished out and stowed',
  'Another for the hold',
] as const;

export type AnchorState = 'up' | 'down' | 'weighing';

export interface GameplayOptions {
  body: BuoyantBody;
  controller: ShipController;
  hullLength: number;
  hullBeam: number;
  /** The hull's mass, kg, as given to its `BuoyantBody`. */
  mass?: number;
  sampler: OceanSampler;
  floorHeight: (x: number, z: number) => number;
  /** Wind: direction it blows toward (`SpectrumParams.windDirection`) and speed, m/s. */
  wind: () => { towardRad: number; speed: number };
  /** Whether the viewer is at the helm. Polled every update. */
  isHelm: () => boolean;
  /** The eye, for the beacon's range fade. */
  camera: THREE.Camera;
  /**
   * The floating barrel to clone for cargo, already normalised so its origin is
   * at the waterline. Null falls back to a procedural crate.
   */
  lootTemplate: THREE.Object3D | null;
  audio?: GameAudio | null;
  ui?: GameUiSink;
  shake?: { setShake(offset: THREE.Vector3, pitch: number, yaw: number, roll: number): void } | null;
  seed?: number;
  /** Where key events come from. Defaults to `window`. */
  keyTarget?: EventTarget;
}

export interface GameplaySnapshot {
  active: boolean;
  sail: SailSetting;
  sailFill: number;
  drive: number;
  angleOffWindDeg: number;
  pointOfSail: PointOfSail;
  heelDeg: number;
  speedKnots: number;
  headingDeg: number;
  windBearingDeg: number;
  windKnots: number;
  anchor: AnchorState;
  cargo: number;
  gold: number;
  target: { x: number; z: number; name: string; bearingDeg: number; distanceM: number } | null;
  trauma: number;
}

export class Gameplay {
  /** Root of everything this draws. Added to the scene by the caller. */
  readonly object = new THREE.Group();

  private readonly o: GameplayOptions;
  private ui: GameUiSink;

  private active = false;
  /** False until the root has been revealed once; see `update`. */
  private armed = false;

  // --- sails --------------------------------------------------------------
  private sail: SailSetting = 'furled';
  private sailFill = 0;
  private drive = 0;
  private offWind = 0;

  // --- anchor -------------------------------------------------------------
  private anchor: AnchorState = 'up';
  private readonly anchorPoint = new THREE.Vector3();
  private rode = 30;
  private weighTimer = 0;
  private wasTaut = false;
  /** +1 starboard bower, -1 port: whichever was to windward when let go. */
  private anchorSide = 1;
  /** Seconds left of the swing that follows a snub. */
  private snubTimer = 0;

  // --- cargo --------------------------------------------------------------
  private readonly seed: number;
  private random: () => number;
  private readonly loot: THREE.Object3D;
  private readonly lootBody: BuoyantBody;
  private lootName = '';
  private lootValue = 0;
  private cargo = 0;
  private gold = 0;
  private readonly beacon = new ObjectiveBeacon();
  private readonly burst = new Burst();
  private readonly bowSpray = new BowSpray();

  // --- feel ---------------------------------------------------------------
  private trauma = 0;
  private grounded = false;
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private clock = 0;

  // --- messages -----------------------------------------------------------
  private uiTimer = 0;
  private ironsTimer = 0;
  private ironsCooldown = 0;
  private shoalCooldown = 0;
  private greeted = false;

  private readonly abort = new AbortController();

  // Scratch.
  private readonly forward = new THREE.Vector3();
  private readonly lee = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();
  private readonly tmp2 = new THREE.Vector3();
  private readonly bow = new THREE.Vector3();
  private readonly shakeOffset = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly euler = new THREE.Euler();

  constructor(options: GameplayOptions) {
    this.o = options;
    this.ui = options.ui ?? NULL_UI_SINK;
    this.seed = options.seed ?? 0x5ea7e1;
    this.random = mulberry32(this.seed);

    this.object.name = 'gameplay';
    this.loot = makeLoot(options.lootTemplate);
    this.loot.name = 'gameplay-cargo';
    this.lootBody = new BuoyantBody({
      object: this.loot,
      probePoints: createRadialProbes(0.62),
      mass: 40 * 0.62 ** 3,
    });
    this.object.add(this.loot, this.beacon.object, this.burst.mesh, this.bowSpray.mesh);

    this.motionQuery =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : null;
    this.reducedMotion = this.motionQuery?.matches ?? false;
    this.motionQuery?.addEventListener(
      'change',
      (e) => {
        this.reducedMotion = e.matches;
      },
      { signal: this.abort.signal },
    );

    const keys = options.keyTarget ?? (typeof window !== 'undefined' ? window : null);
    keys?.addEventListener('keydown', this.onKeyDown as EventListener, {
      signal: this.abort.signal,
    });

    this.reset();
  }

  /** Points the reports somewhere else, e.g. once the real HUD exists. */
  setUiSink(sink: GameUiSink | null): void {
    this.ui = sink ?? NULL_UI_SINK;
    if (this.active) this.pushAllUi();
  }

  get isActive(): boolean {
    return this.active;
  }

  /**
   * Returns every piece of gameplay state to its seeded start.
   *
   * Call after the hull itself has been returned home — the first cargo is laid
   * out from where the ship is, so it has to be where it will start from.
   */
  reset(): void {
    this.sail = 'furled';
    this.sailFill = 0;
    this.drive = 0;
    this.offWind = 0;
    this.anchor = 'up';
    this.weighTimer = 0;
    this.wasTaut = false;
    this.snubTimer = 0;
    this.cargo = 0;
    this.gold = 0;
    this.trauma = 0;
    this.grounded = false;
    this.clock = 0;
    this.uiTimer = 0;
    this.ironsTimer = 0;
    this.ironsCooldown = 0;
    this.shoalCooldown = 0;
    this.random = mulberry32(this.seed);
    this.o.controller.setSailDrive(0);
    this.burst.clear();
    this.bowSpray.clear();
    this.beacon.resetClock();
    this.clearShake();

    const ship = this.o.body.object.position;
    this.spawnLoot(ship.x, ship.z);
    if (this.active) this.pushAllUi();
  }

  /**
   * Advances the gameplay by `dt`. Call after `ShipController.update` and before
   * the buoyancy solve: forces added here are integrated this frame.
   */
  update(dt: number): void {
    if (!(dt > 0)) return;
    const active = this.o.isHelm();
    if (active !== this.active) this.setActive(active);

    // Visibility is left alone until the root has been revealed once. The scene
    // load adds this hidden and reveals it under a paused loop for the compile
    // pass, which only sees what is visible — hiding the children before then
    // would leave their pipelines to compile mid-voyage.
    if (!this.armed) {
      if (!this.object.visible) return;
      this.armed = true;
    }
    this.loot.visible = active;
    this.beacon.object.visible = active;
    this.burst.mesh.visible = active && this.burst.alive;
    this.bowSpray.mesh.visible = active;
    if (!active) return;

    this.clock += dt;
    const body = this.o.body;
    const wind = this.o.wind();

    // --- sails ------------------------------------------------------------
    this.sailFill = approach(this.sailFill, SAIL_FILL[this.sail], SAIL_RATE * dt);
    this.offWind = angleOffWind(body.heading, wind.towardRad);
    const windFactor = windDriveFactor(wind.speed);
    this.drive =
      this.sailFill *
      polarDrive(this.offWind) *
      windFactor *
      SAIL_POWER *
      (this.anchor === 'up' ? 1 : ANCHORED_SAIL);
    // Read by the controller on its next update — a frame of latency on a
    // quantity that itself takes seconds to change.
    this.o.controller.setSailDrive(this.drive);

    this.forward.set(1, 0, 0).applyQuaternion(body.object.quaternion).setY(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
    this.forward.normalize();

    this.applyHeel(wind.towardRad, windFactor);
    this.applyAnchor(dt, wind);
    this.updateLoot(dt);
    this.updateFeel(dt);
    this.updateMessages(dt);

    this.burst.update(dt, this.o.camera);
    this.bowSpray.update(
      dt,
      body.object,
      body.velocity,
      body.velocity.dot(this.forward),
      this.o.hullLength,
      this.o.hullBeam,
      this.o.camera,
    );
    this.burst.mesh.visible = this.burst.alive;
  }

  /* ------------------------------------------------------------- commands */

  /** Raises (+1) or takes in (-1) one step of sail. */
  stepSail(direction: 1 | -1): void {
    const index = SAIL_ORDER.indexOf(this.sail);
    const next = SAIL_ORDER[Math.min(SAIL_ORDER.length - 1, Math.max(0, index + direction))];
    this.setSail(next);
  }

  setSail(setting: SailSetting): void {
    if (setting === this.sail) return;
    this.sail = setting;
    this.pushTelemetry();
  }

  /**
   * A hull impact detected by `physics/Spray`: a bow slam, a beam slap, or green
   * water over the bow. Feeds the camera shake while the helm is taken and is
   * ignored otherwise — shaking an orbit or cinematic camera would be wrong.
   */
  reportImpact(kind: 'slam' | 'slap' | 'wash' | 'float', strength: number): void {
    if (!this.active) return;
    const s = Math.min(1, Math.max(0, strength));
    this.addTrauma(IMPACT_TRAUMA[kind] * (0.35 + 0.65 * s));
  }

  toggleAnchor(): void {
    if (this.anchor === 'up') this.dropAnchor();
    else if (this.anchor === 'down') this.weighAnchor();
  }

  dropAnchor(): void {
    if (this.anchor !== 'up') return;
    this.bowPoint(this.bow);
    const depth = Math.max(4, -this.o.floorHeight(this.bow.x, this.bow.z));
    // Scope of about three to one, bounded so a deep anchorage still bites in a
    // few hull lengths and a shallow one still lets the ship swing.
    this.rode = THREE.MathUtils.clamp(depth * 2 + 6, 20, 40);
    // The windward bower: the anchor goes from the bow on the side the wind is
    // coming over, a few metres out. When the rode snubs up, that offset is the
    // lever that swings her head round toward the wind — the swing that tells
    // a player the anchor has bitten.
    const wind = this.o.wind();
    this.forward.set(1, 0, 0).applyQuaternion(this.o.body.object.quaternion).setY(0).normalize();
    // Starboard is +Z local, i.e. forward rotated a quarter turn clockwise from above.
    const sx = -this.forward.z;
    const sz = this.forward.x;
    const windward = -(Math.cos(wind.towardRad) * sx + Math.sin(wind.towardRad) * sz) >= 0 ? 1 : -1;
    this.anchorSide = windward;
    const out = this.o.hullBeam * 0.5 + 7;
    this.anchorPoint.set(this.bow.x + sx * out * windward, 0, this.bow.z + sz * out * windward);
    this.anchor = 'down';
    this.wasTaut = false;
    this.o.audio?.playAnchorChain(1.8);
    this.o.audio?.playSplash(0.5);
    this.tmp.set(this.bow.x, this.o.sampler.ready ? this.o.sampler.height(this.bow.x, this.bow.z) : 0, this.bow.z);
    this.burst.fire(this.tmp, 0.45, false);
    this.ui.toast("Anchor down. She'll swing to the wind", 'info');
    this.pushTelemetry();
  }

  weighAnchor(): void {
    if (this.anchor !== 'down') return;
    this.anchor = 'weighing';
    this.weighTimer = WEIGH_SECONDS;
    this.o.audio?.playAnchorChain(WEIGH_SECONDS);
    this.ui.toast('Hauling the anchor up…', 'info');
  }

  /** Moves the current cargo to `distance` metres off the bow. For tests. */
  placeTargetAhead(distance = 40, lateral = 0): void {
    const p = this.o.body.object.position;
    this.forward.set(1, 0, 0).applyQuaternion(this.o.body.object.quaternion).setY(0).normalize();
    const x = p.x + this.forward.x * distance - this.forward.z * lateral;
    const z = p.z + this.forward.z * distance + this.forward.x * lateral;
    this.placeLoot(x, z);
  }

  /** Sail state for the audio, written into `out`. Allocation-free; call every frame. */
  audioState(out: GameplayAudioState): GameplayAudioState {
    out.active = this.active;
    if (!this.active) {
      out.sailFill = 0;
      out.sailLuff = 0;
      return out;
    }
    // The same in-irons ramp `applyHeel` uses for the canvas *not* pulling.
    const irons = 1 - smoothstep(IRONS_DEG - 8, IRONS_DEG + 15, this.offWind);
    const spilled = this.anchor === 'up' ? 0 : 0.7;
    const handling = Math.abs(SAIL_FILL[this.sail] - this.sailFill) > 0.01 ? 0.6 : 0;
    out.sailFill = this.sailFill;
    out.sailLuff = Math.min(1, Math.max(this.sailFill * Math.max(irons, spilled), handling));
    return out;
  }

  snapshot(): GameplaySnapshot {
    const body = this.o.body;
    const wind = this.o.wind();
    const lp = this.loot.position;
    const sp = body.object.position;
    return {
      active: this.active,
      sail: this.sail,
      sailFill: this.sailFill,
      drive: this.drive,
      angleOffWindDeg: angleOffWind(body.heading, wind.towardRad),
      pointOfSail: pointOfSail(angleOffWind(body.heading, wind.towardRad)),
      heelDeg: THREE.MathUtils.radToDeg(this.rollAngle()),
      speedKnots: Math.hypot(body.velocity.x, body.velocity.z) * KNOTS_PER_MS,
      headingDeg: compassFromHeading(body.heading),
      windBearingDeg: windFromCompass(wind.towardRad),
      windKnots: wind.speed * KNOTS_PER_MS,
      anchor: this.anchor,
      cargo: this.cargo,
      gold: this.gold,
      target: {
        x: lp.x,
        z: lp.z,
        name: this.lootName,
        bearingDeg: compassFromVector(lp.x - sp.x, lp.z - sp.z),
        distanceM: Math.hypot(lp.x - sp.x, lp.z - sp.z),
      },
      trauma: this.trauma,
    };
  }

  /** The `__ocean.gameplay` test surface. */
  get testHandle() {
    return {
      isActive: () => this.active,
      state: () => this.snapshot(),
      setSail: (s: SailSetting) => this.setSail(s),
      stepSail: (d: 1 | -1) => this.stepSail(d),
      dropAnchor: () => this.dropAnchor(),
      weighAnchor: () => this.weighAnchor(),
      toggleAnchor: () => this.toggleAnchor(),
      placeTargetAhead: (distance?: number, lateral?: number) =>
        this.placeTargetAhead(distance, lateral),
      reset: () => this.reset(),
    };
  }

  dispose(): void {
    this.abort.abort();
    this.setActive(false);
    this.object.removeFromParent();
    this.beacon.dispose();
    this.burst.dispose();
    this.bowSpray.dispose();
  }

  /* ------------------------------------------------------------ internals */

  private setActive(active: boolean): void {
    this.active = active;
    const controller = this.o.controller;
    controller.setThrottleKeysEnabled(!active);
    if (!active) {
      controller.setSailDrive(0);
      this.clearShake();
      this.ui.setObjective(null);
      this.ui.setShipTelemetry(null);
      return;
    }
    this.pushAllUi();
    if (!this.greeted) {
      this.greeted = true;
      // A welcome, not a manual. The controls are taught one at a time, as
      // they are needed, by the HUD's lessons and the key ribbon, phrased for
      // the hands in use; repeating all of them here would be a wall of keys.
      this.ui.toast('You have the helm. Fair winds', 'info');
    }
  }

  /**
   * Heel and leeway from the canvas.
   *
   * The wind's push on the sails resolves into drive along the bow — which the
   * controller already has — and a sideways component that does two things: it
   * shoves the hull to leeward, which the keel's lateral drag mostly refuses,
   * and it acts high up the masts, which heels her. The heel is a torque about
   * the hull's own fore-and-aft axis and the buoyancy probes supply the righting
   * moment, so the angle she settles at is the solver's answer, and a wave
   * rolling her further is still felt on top of it.
   */
  private applyHeel(windToward: number, windFactor: number): void {
    if (this.sailFill <= 0.001) return;
    // Leeward: the component of the wind's direction across the bow.
    const wx = Math.cos(windToward);
    const wz = Math.sin(windToward);
    const along = wx * this.forward.x + wz * this.forward.z;
    this.lee.set(wx - along * this.forward.x, 0, wz - along * this.forward.z);
    const across = this.lee.length();
    if (across < 1e-3) return;
    this.lee.divideScalar(across);

    // Zero in irons (the canvas is luffing, not pulling), largest abeam, gone
    // dead downwind where the push is all along the bow.
    const pulling = smoothstep(IRONS_DEG - 8, IRONS_DEG + 15, this.offWind);
    const side = across * pulling * this.sailFill * windFactor;

    const body = this.o.body;
    // U x L, with U = world up: the axis that rolls the mastheads to leeward.
    body.externalTorque.x += this.lee.z * HEEL_TORQUE * side;
    body.externalTorque.z += -this.lee.x * HEEL_TORQUE * side;
    body.externalForce.addScaledVector(this.lee, LEEWAY_FORCE * side);
  }

  private applyAnchor(dt: number, wind: { towardRad: number; speed: number }): void {
    if (this.anchor === 'up') return;
    const body = this.o.body;

    if (this.anchor === 'weighing') {
      this.weighTimer -= dt;
      // The rode is held at its length while the capstan works: shortening it
      // here would yank a ship lying at the end of fifty metres of cable toward
      // the anchor at the full force cap.
      if (this.weighTimer <= 0) {
        this.anchor = 'up';
        this.ui.toast("Anchor up. She's free to sail", 'info');
        this.pushTelemetry();
        return;
      }
    }

    const center = body.object.position;
    // The hawse: the rode leads from the bow on the anchor's side, not from the
    // stem, and that beam-wise lever is what turns a snub into a swing.
    this.bowPoint(this.bow);
    const hawse = this.o.hullBeam * 0.45 * this.anchorSide;
    this.bow.x += -this.forward.z * hawse;
    this.bow.z += this.forward.x * hawse;
    const rx = this.bow.x - center.x;
    const rz = this.bow.z - center.z;

    // Velocity of the bow itself: the hull's plus its yaw swinging the bow round.
    const wy = body.angularVelocity.y;
    const vx = body.velocity.x + wy * rz;
    const vz = body.velocity.z - wy * rx;

    const dx = this.bow.x - this.anchorPoint.x;
    const dz = this.bow.z - this.anchorPoint.z;
    const dist = Math.hypot(dx, dz);
    let fx = 0;
    let fz = 0;
    if (dist > this.rode && dist > 1e-3) {
      const ux = dx / dist;
      const uz = dz / dist;
      const outward = vx * ux + vz * uz;
      const stretch = dist - this.rode;
      let pull = RODE_STIFFNESS * stretch + RODE_DAMPING * Math.max(0, outward);
      pull = Math.min(pull, RODE_MAX_FORCE);
      fx = -ux * pull;
      fz = -uz * pull;
      // The moment it snubs up: the jolt is felt through the whole hull.
      if (!this.wasTaut && outward > 0.8) {
        this.addTrauma(Math.min(0.7, 0.35 + outward * 0.15));
        this.snubTimer = SNUB_SECONDS;
        this.o.audio?.playAnchorChain(0.4);
      }
      this.wasTaut = true;
    } else {
      this.wasTaut = false;
    }

    // Drag of the ground tackle, and the wind on the hull and rigging, which is
    // applied abaft amidships so a ship hanging off her bow weathervanes.
    const drag = ANCHOR_DRAG * (this.o.mass ?? 90_000);
    const windForce = WINDAGE * wind.speed * wind.speed;
    const wfx = Math.cos(wind.towardRad) * windForce;
    const wfz = Math.sin(wind.towardRad) * windForce;
    const ax = -this.forward.x * this.o.hullLength * 0.2;
    const az = -this.forward.z * this.o.hullLength * 0.2;

    body.externalForce.x += fx - body.velocity.x * drag + wfx;
    body.externalForce.z += fz - body.velocity.z * drag + wfz;
    // r x F, y component only: both forces are horizontal.
    // For a moment after a snub the rode's turning moment is exaggerated: the
    // controller's yaw damping is tuned to keep a turn from winding up under
    // way, and on its own it smothers the swing that makes anchoring read.
    this.snubTimer = Math.max(0, this.snubTimer - dt);
    const swing = this.snubTimer > 0 ? SNUB_SWING : 1;
    body.externalTorque.y += (rz * fx - rx * fz) * swing + (az * wfx - ax * wfz);
  }

  private updateLoot(dt: number): void {
    this.lootBody.update(dt, this.o.sampler);
    const ship = this.o.body.object.position;
    const lp = this.loot.position;
    this.beacon.update(dt, lp, this.o.camera.position);

    // Distance to the hull's centreline, so a target alongside the quarter is
    // taken as readily as one dead ahead.
    const half = this.o.hullLength * 0.5;
    const relX = lp.x - ship.x;
    const relZ = lp.z - ship.z;
    const along = THREE.MathUtils.clamp(relX * this.forward.x + relZ * this.forward.z, -half, half);
    const cx = relX - this.forward.x * along;
    const cz = relZ - this.forward.z * along;
    const reach = this.o.hullBeam * 0.5 + PICKUP_MARGIN;
    if (cx * cx + cz * cz <= reach * reach) this.collect();
  }

  private collect(): void {
    const lp = this.loot.position;
    this.cargo += 1;
    this.gold += this.lootValue;
    this.burst.fire(this.tmp2.set(lp.x, lp.y + 0.4, lp.z), 1, true);
    this.o.audio?.playSplash(0.7);
    this.o.audio?.playChime();
    this.addTrauma(0.18);
    const haul = HAUL_LINES[(this.cargo - 1) % HAUL_LINES.length];
    this.ui.toast(`${haul}: ${this.lootName} · +${this.lootValue} gold`, 'reward');
    this.ui.setScore('Cargo', this.cargo);
    this.ui.setScore('Gold', this.gold);
    this.spawnLoot(lp.x, lp.z);
    this.pushObjective();
  }

  /** Picks the next cargo position from the seeded sequence, in navigable water. */
  private spawnLoot(fromX: number, fromZ: number): void {
    const random = this.random;
    let x = fromX;
    let z = fromZ;
    let found = false;
    for (let attempt = 0; attempt < 48 && !found; attempt++) {
      const angle = random() * Math.PI * 2;
      // Widen the search if the first rings keep landing on the reef.
      const radius = SPAWN_MIN + random() * SPAWN_SPAN * (1 + attempt / 16);
      const cx = fromX + Math.cos(angle) * radius;
      const cz = fromZ + Math.sin(angle) * radius;
      if (Math.hypot(cx, cz) > SPAWN_LIMIT) continue;
      if (!this.navigable(cx, cz)) continue;
      x = cx;
      z = cz;
      found = true;
    }
    if (!found) {
      // Back toward the middle of the play area, which is open water by design.
      x = fromX * 0.5;
      z = fromZ * 0.5;
    }
    this.lootName = CARGO_NAMES[Math.floor(random() * CARGO_NAMES.length)];
    this.lootValue = 50 + Math.floor(random() * 11) * 10;
    this.placeLoot(x, z, random() * Math.PI * 2);
  }

  private placeLoot(x: number, z: number, yaw = 0): void {
    this.tmp.set(x, 0, z);
    this.euler.set(0.25, yaw, 0.15);
    this.quat.setFromEuler(this.euler);
    this.lootBody.reset(this.tmp, this.quat);
  }

  private navigable(x: number, z: number): boolean {
    const floor = this.o.floorHeight;
    if (-floor(x, z) < SPAWN_DEPTH) return false;
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const px = x + Math.cos(a) * SPAWN_RING_RADIUS;
      const pz = z + Math.sin(a) * SPAWN_RING_RADIUS;
      if (-floor(px, pz) < SPAWN_RING_DEPTH) return false;
    }
    return true;
  }

  /**
   * Slams, groundings and the camera shake they drive.
   *
   * Trauma, not shake, is what accumulates: each event adds to a 0..1 budget
   * that drains linearly, and the shake is its square — so small knocks barely
   * register and a big slam is unmistakable, without a pile of small ones
   * adding up to an earthquake. The motion itself is a sum of incommensurate
   * sines of the simulation clock, so a deterministic run shakes identically.
   */
  private updateFeel(dt: number): void {
    const body = this.o.body;

    // Bow slams and beam slaps arrive through `reportImpact`, from the spray
    // system's probes — one detector for the whole hull rather than a second
    // copy of its bow probe here.

    // Touching the bottom.
    const p = body.object.position;
    const depth = -this.o.floorHeight(p.x, p.z);
    const speed = Math.hypot(body.velocity.x, body.velocity.z);
    const grounded = depth < GROUND_DEPTH && speed > 0.8;
    if (grounded && !this.grounded) {
      this.addTrauma(Math.min(0.7, 0.3 + speed * 0.08));
      this.o.audio?.hullImpact?.(Math.min(1, 0.4 + speed * 0.1), true);
    }
    if (grounded) this.trauma = Math.max(this.trauma, Math.min(0.45, speed * 0.1));
    this.grounded = grounded;
    this.shoalCooldown = Math.max(0, this.shoalCooldown - dt);
    if (depth < GROUND_DEPTH + 3 && speed > 1 && this.shoalCooldown === 0) {
      this.shoalCooldown = 8;
      this.ui.toast('Shallows! Turn her away', 'warning');
    }

    this.trauma = Math.max(0, this.trauma - TRAUMA_DECAY * dt);
    const shake = this.o.shake;
    if (!shake) return;
    const amount = this.reducedMotion ? 0 : this.trauma * this.trauma;
    if (amount <= 1e-5) {
      this.clearShake();
      return;
    }
    const t = this.clock;
    const n1 = Math.sin(t * 13.1 + 0.3) * 0.6 + Math.sin(t * 21.7 + 1.9) * 0.4;
    const n2 = Math.sin(t * 11.3 + 4.1) * 0.6 + Math.sin(t * 17.9 + 2.6) * 0.4;
    const n3 = Math.sin(t * 9.7 + 5.3) * 0.6 + Math.sin(t * 15.1 + 0.8) * 0.4;
    const n4 = Math.sin(t * 14.3 + 3.7) * 0.6 + Math.sin(t * 23.3 + 4.4) * 0.4;
    this.shakeOffset.set(0, n4 * SHAKE_LIFT * amount, 0);
    shake.setShake(this.shakeOffset, n1 * SHAKE_PITCH * amount, n2 * SHAKE_YAW * amount, n3 * SHAKE_ROLL * amount);
  }

  private updateMessages(dt: number): void {
    // In irons: canvas set, head to wind, and stuck there for a couple of seconds.
    this.ironsCooldown = Math.max(0, this.ironsCooldown - dt);
    if (this.sailFill > 0.3 && this.offWind < IRONS_DEG && this.anchor === 'up') {
      this.ironsTimer += dt;
      if (this.ironsTimer > 2 && this.ironsCooldown === 0) {
        this.ironsCooldown = 12;
        this.ui.toast('Stalled head to wind. Turn either way to fill the sails', 'warning');
      }
    } else {
      this.ironsTimer = 0;
    }

    this.uiTimer += dt;
    if (this.uiTimer >= UI_INTERVAL) {
      this.uiTimer = 0;
      this.pushTelemetry();
      this.pushObjective();
    }
  }

  private pushAllUi(): void {
    this.ui.setScore('Cargo', this.cargo);
    this.ui.setScore('Gold', this.gold);
    this.pushTelemetry();
    this.pushObjective();
  }

  private pushTelemetry(): void {
    if (!this.active) return;
    const body = this.o.body;
    const wind = this.o.wind();
    this.ui.setShipTelemetry({
      speedKnots: Math.hypot(body.velocity.x, body.velocity.z) * KNOTS_PER_MS,
      headingDeg: compassFromHeading(body.heading),
      sail: this.sail,
      anchored: this.anchor !== 'up',
      windBearingDeg: windFromCompass(wind.towardRad),
      windKnots: wind.speed * KNOTS_PER_MS,
    });
  }

  private pushObjective(): void {
    if (!this.active) return;
    const ship = this.o.body.object.position;
    const lp = this.loot.position;
    const objective: GameObjective = {
      title: 'Haul in the drifting cargo',
      detail: capitalise(this.lootName) + ` · ${this.lootValue} gold`,
      bearingDeg: compassFromVector(lp.x - ship.x, lp.z - ship.z),
      distanceM: Math.hypot(lp.x - ship.x, lp.z - ship.z),
    };
    this.ui.setObjective(objective);
  }

  private addTrauma(amount: number): void {
    this.trauma = Math.min(1, this.trauma + amount);
  }

  private clearShake(): void {
    this.shakeOffset.set(0, 0, 0);
    this.o.shake?.setShake(this.shakeOffset, 0, 0, 0);
  }

  /** World position of the stem, at the waterline unless `withHeight`. */
  private bowPoint(out: THREE.Vector3, withHeight = false): THREE.Vector3 {
    const object = this.o.body.object;
    out.set(this.o.hullLength * 0.45, 0, 0).applyQuaternion(object.quaternion).add(object.position);
    if (!withHeight) out.y = 0;
    return out;
  }

  /** Roll of the hull, radians, positive with the starboard rail down. */
  private rollAngle(): number {
    const q = this.o.body.object.quaternion;
    // The hull's beam axis (+Z local, to starboard) tilting out of the horizontal.
    this.tmp.set(0, 0, 1).applyQuaternion(q);
    return -Math.asin(THREE.MathUtils.clamp(this.tmp.y, -1, 1));
  }

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (!this.active || event.repeat || isTypingTarget(event.target)) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    switch (event.code) {
      case 'KeyW':
        this.stepSail(1);
        break;
      case 'KeyS':
        this.stepSail(-1);
        break;
      case 'KeyR':
        this.toggleAnchor();
        break;
      default:
        return;
    }
    event.preventDefault();
  };
}

/* ---------------------------------------------------------------- helpers */

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (typeof HTMLElement === 'undefined' || !(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}

/**
 * The cargo's mesh: a clone of the scene's floating barrel when there is one —
 * same geometry and materials, so no new pipeline — or a plain crate.
 */
function makeLoot(template: THREE.Object3D | null): THREE.Object3D {
  if (template) {
    const clone = template.clone(true);
    clone.position.set(0, 0, 0);
    clone.rotation.set(0, 0, 0);
    clone.scale.set(1, 1, 1);
    clone.traverse((node) => {
      node.frustumCulled = false;
    });
    return clone;
  }
  const crate = new THREE.Group();
  const material = new THREE.MeshStandardNodeMaterial({ color: 0x8a5a2b, roughness: 0.85 });
  const box = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.9, 0.9), material);
  box.position.y = 0.09;
  box.castShadow = true;
  box.frustumCulled = false;
  crate.add(box);
  return crate;
}
