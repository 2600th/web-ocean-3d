import * as THREE from 'three/webgpu';
import { BuoyantBody, createRadialProbes } from '../physics/Buoyancy';
import type { ShipController } from '../physics/ShipController';
import type { SprayLanding } from '../physics/Spray';
import type { OceanSampler } from '../ocean/Sampler';
import { mulberry32 } from '../core/random';
import { keybinds } from '../input/keybinds';
import type { WeatherKind } from '../presets';
import { ISLAND } from '../scene/Seafloor';
import { BowSpray, Burst, CargoGlint, ObjectiveBeacon } from './effects';
import {
  FLOOR_DRIVE,
  IRONS_DEG,
  KNOTS_PER_MS,
  angleOffWind,
  apparentWind,
  approach,
  compassFromHeading,
  compassFromVector,
  idealTrim,
  overpress,
  pointOfSail,
  polarDrive,
  sailPressure,
  smoothstep,
  trimFactor,
  windFromCompass,
  type ApparentWind,
  type PointOfSail,
} from './sailing';
import {
  NULL_UI_SINK,
  type GameAudio,
  type GameplayAudioState,
  type GameObjective,
  type GameUiSink,
  type SailSetting,
  type VoyageTally,
} from './types';
import {
  ANCHORAGE_RADIUS,
  VOYAGE_CASKS,
  Voyage,
  findAnchorage,
  freshVoyageSeed,
  type VoyageEvent,
  type VoyagePhase,
  type VoyageSky,
} from './Voyage';
import { readVoyageSave, recordVoyage } from './voyageSave';

/**
 * Helm-mode gameplay: sails that answer to the wind, an anchor, and cargo to
 * recover.
 *
 * **Everything here is a force or a report.** Like `ShipController`, this never
 * writes the hull's pose. The sails hand the controller a force, and
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
 * Force along the bow, N, at full drive: full sail at full pressure on the
 * best point of sail, before trim.
 *
 * Its own figure rather than a fraction of the controller's engine, which the
 * helm never opens. Against the hull's drag this gives about twelve knots on a
 * beam reach in a fresh breeze once apparent wind is counted, which is what a
 * laden square-rigger makes at her best, and some fourteen at the peak of the
 * overpress curve in a strong one.
 */
const SAIL_THRUST = 66_000;
/** Sheet travel per second for a held trim input, full range in two seconds. */
const TRIM_RATE = 0.5;
/** Sheet setting at the start: half out, which suits nothing exactly. */
const TRIM_START = 0.5;
/** Extra rudder authority at speed while the player has the helm. */
const TURN_BOOST = 0.4;
/**
 * Most canvas pressure the heel and leeway see. Pressure past full is what
 * over-heels her in a gale; this cap keeps a hurricane slider from rolling her
 * past what the buoyancy probes can right.
 */
const HEEL_PRESSURE_MAX = 2.2;
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
 * hull; green water over the bow is the worst. A beam slap is heard, not felt:
 * it adds none. Strength itself comes from `physics/Spray`, which owns impact
 * detection — this layer only decides how much of it reaches the camera.
 */
const IMPACT_TRAUMA = { slam: 0.3, slap: 0, wash: 0.55, float: 0 } as const;
/**
 * Strength below which an impact does not reach the camera, by kind; the
 * trauma ramps in from here to full strength.
 *
 * The shake is for the impact that stands out, not for sailing. Every impact
 * used to add trauma, and under way they arrive two or three a second in any
 * sea (sampled over 20 s at full sail: 2.4/s in `skyPro`, 1.9 in
 * `seaOfThieves`, 3.2 in `storm`). Trauma sat near 1 and the camera shook in
 * 47–84% of frames, up to 0.85 degrees of pitch at 2–4 Hz: the whole frame
 * wobbling up and down while simply sailing, on top of the hull's own pitch,
 * which already shows every one of those waves.
 */
const IMPACT_THRESHOLD = { slam: 0.7, slap: 1, wash: 0.5, float: 1 } as const;
// The slam threshold filters little at full sail: `Spray` counts a third of
// the forward speed into a bow strike's strength, so under way nearly every
// one arrives at 1. What keeps slams from becoming a wobble is the cooldown
// below and their smaller trauma (0.3, which squares to a tenth of the full
// shake: under a tenth of a degree, a thud). Green water and groundings keep
// the large ones.
/**
 * Seconds after an impact shakes the camera before another may. Keeps a run of
 * heavy slams in a storm from stacking into a continuous shake; groundings are
 * not limited by it.
 */
const IMPACT_COOLDOWN = 4;
/** Depth under the hull, metres, at which she is touching. Above `SHOAL_HARD_DEPTH` in the controller. */
const GROUND_DEPTH = 6;
/** Trauma lost per second. */
const TRAUMA_DECAY = 1.1;
/** Maximum shake at trauma 1: radians and metres. */
const SHAKE_PITCH = THREE.MathUtils.degToRad(0.9);
const SHAKE_YAW = THREE.MathUtils.degToRad(0.5);
const SHAKE_ROLL = THREE.MathUtils.degToRad(1.3);
const SHAKE_LIFT = 0.12;
/**
 * Thunder shakes the hull a little, by how close it is: the trauma at the
 * nearest strike (`sky/Lightning` puts them 900 m to 6.5 km out) and the range
 * past which it adds none. It lands when the sound does, not with the flash.
 */
/**
 * A grounding dips the view as she stops: pitch in radians at full speed, how
 * fast it eases back, and the metres of drop per radian of it.
 */
const GROUND_KICK = THREE.MathUtils.degToRad(1.6);
const GROUND_KICK_RATE = 9;
const GROUND_KICK_LIFT = 7;
/** Seconds head to wind before the HUD marks it; the toast waits for two. */
const STALL_CUE_SECONDS = 0.5;
const THUNDER_TRAUMA = 0.3;
const THUNDER_RANGE = 6500;
const SPEED_OF_SOUND = 343;
/**
 * A single event this hard also holds the world for a moment: milliseconds of
 * hit-stop as a base plus a share of the trauma, ignored below the threshold.
 * Real time, not simulation time, so it is the same on a 60 and a 120 Hz screen.
 */
const HIT_STOP_TRAUMA = 0.45;
const HIT_STOP_BASE_MS = 40;
const HIT_STOP_PER_TRAUMA_MS = 70;

/** Seconds between HUD pushes for the continuous readouts. */
const UI_INTERVAL = 0.1;

/**
 * What the cask turns out to be. Six, and it must stay six: the seeded
 * generator picks by index, so a longer list would move every later cargo
 * the deterministic runs depend on.
 */
const CARGO_NAMES = [
  'a cask of dark rum',
  'a hogshead of sugar',
  "a merchant's lost cask",
  'a cask sealed in wax',
  'a cask of molasses',
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

/**
 * Hauling a cask aboard: how long it takes to arc from the water to the rail,
 * how high the arc rises (metres), the pause before the next cask appears, and
 * the time that cask and its beacon take to fade in. Simulation time.
 */
const HAUL_SECONDS = 0.55;
const HAUL_ARC = 4.5;
const LOOT_BEAT = 0.8;
const LOOT_FADE_IN = 1;

/**
 * The voyage (`Voyage.ts`). Casks after the first are laid within these bands
 * of the anchorage, metres, so a run makes a loop that ends near home instead
 * of a random walk that can strand her a mile out: the middle casks anywhere
 * reasonable, the last one a short sail from the cove.
 */
const CASK_BAND: readonly [number, number] = [250, 700];
const LAST_CASK_BAND: readonly [number, number] = [200, 420];
/**
 * How much a squall's gusts add to the wind on the canvas at its height: 55%,
 * so the default 11 m/s gusts to 17. Full sail then overpresses (slower, heeled
 * hard), half sail draws about as well as full did before it: taking in canvas
 * is the right call and the faster one. Only the sails feel it; the sea state
 * is the preset's.
 */
const SQUALL_GUST = 0.55;
/**
 * A cask goes over the side when she is driven this hard (canvas pressure,
 * see `sailing.sailPressure`) for this many seconds, with the squall at least
 * this strong. Once per squall, and it floats where it fell, so the cost is the
 * time to turn back for it, or the cask if she sails on.
 */
const OVERBOARD_PRESSURE = 1.6;
const OVERBOARD_SECONDS = 3;
const OVERBOARD_SQUALL = 0.6;
/** Metres astern of the stern where the lost cask is put in the water. */
const OVERBOARD_ASTERN = 12;

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
  /**
   * Something hitting the water at (x, z), the surface at `y`: the anchor let
   * go, cargo hauled out. See `physics/Splash`. Without it the anchor throws
   * the old white burst.
   */
  splash?: ((x: number, y: number, z: number, options: { radius: number; speed: number; mass: number }) => void) | null;
  /** The player's shake intensity, 0..1. Polled each use; 0 also turns hit-stop off. */
  shakeScale?: () => number;
  /** Holds the simulation nearly still for this many real milliseconds. */
  hitStop?: (ms: number) => void;
}

export interface GameplaySnapshot {
  active: boolean;
  sail: SailSetting;
  sailFill: number;
  drive: number;
  angleOffWindDeg: number;
  pointOfSail: PointOfSail;
  /** Sheets, 0 hauled in .. 1 eased out, and the setting that suits the wind now. */
  trim: number;
  idealTrim: number;
  /** Apparent wind on the canvas: angle off the bow it comes from, and speed. */
  apparentAngleDeg: number;
  apparentKnots: number;
  /** Canvas pressure, 1 is full; above about 1.3 she is overpressed. */
  pressure: number;
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
  voyage: {
    phase: VoyagePhase;
    number: number;
    seed: number;
    frozen: boolean;
    seconds: number;
    casks: number;
    squall: 'none' | 'coming' | 'here' | 'past';
    squallRain: number;
    /** Seconds until this voyage's squall builds, or null once it has. */
    squallIn: number | null;
    hours: number;
    anchorage: { x: number; z: number };
    tally: VoyageTally | null;
  };
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
  /** Sheets, 0 hauled hard in .. 1 eased right out. */
  private trim = TRIM_START;
  /** Held sheet rate from the pad or touch stick, -1..1; keys add to it. */
  private trimInput = 0;
  private readonly trimKeys = new Set<string>();
  private pressure = 0;
  private readonly apparent: ApparentWind = { speed: 0, angleDeg: 0, side: 0, towardRad: 0 };

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
  private readonly glint = new CargoGlint();
  /** The cask in flight: a copy of the loot, swung from where it floated to the rail. */
  private readonly haulCask: THREE.Object3D;
  /** Seconds into the flight; negative when none is under way. */
  private haulTime = -1;
  private readonly haulFrom = new THREE.Vector3();
  /** Seconds before the next cask appears, then 0..1 as it and its beacon fade in. */
  private lootWait = 0;
  private lootReveal = 1;
  private readonly burst = new Burst();
  /** Sand churned up where she grounds; its own burst so a pickup does not cut it short. */
  private readonly sandPlume = new Burst();
  private readonly bowSpray = new BowSpray();

  // --- voyage -------------------------------------------------------------
  private readonly voyage = new Voyage();
  /** Where dropping anchor lands the cargo, found from the seabed at construction. */
  private readonly homeAnchorage: { x: number; z: number };
  private readonly anchorage = { x: 0, z: 0 };
  /** Casks of this voyage's plan laid in the water so far. */
  private laid = 1;
  /** Gold each cask aboard was worth, so the one that goes over the side takes its own. */
  private readonly hold: number[] = [];
  /** The current target is a cask that went over the side. */
  private recovering = false;
  /** The planned cask the lost one displaced as the target, put back once it is recovered. */
  private stashed: { x: number; z: number; name: string; value: number } | null = null;
  private overboardTimer = 0;
  private lostThisSquall = false;
  /** The sun has been said to be going down this voyage. */
  private duskSaid = false;
  private tally: VoyageTally | null = null;
  private readonly events: VoyageEvent[] = [];

  // --- feel ---------------------------------------------------------------
  private trauma = 0;
  /** Seconds until an impact may add trauma again; see `IMPACT_COOLDOWN`. */
  private impactCooldown = 0;
  private grounded = false;
  /** Pitch kick from a grounding, radians; negative dips the camera, and it eases back. */
  private kick = 0;
  /** Thunder on its way: seconds until it is heard and the trauma it brings. */
  private readonly thunder: { delay: number; amount: number }[] = [];
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private clock = 0;

  // --- messages -----------------------------------------------------------
  private uiTimer = 0;
  private ironsTimer = 0;
  /** Whether the HUD has been told she is stalled; see `STALL_CUE_SECONDS`. */
  private stalled = false;
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
  private readonly beaconAt = new THREE.Vector3();
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
    this.haulCask = this.loot.clone(true);
    this.haulCask.name = 'gameplay-cargo-haul';
    this.haulCask.visible = false;
    this.object.add(this.loot, this.haulCask, this.glint.mesh, this.beacon.object, this.burst.mesh, this.sandPlume.mesh, this.bowSpray.mesh);
    this.homeAnchorage = findAnchorage(options.floorHeight, ISLAND);

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
    keys?.addEventListener('keyup', this.onKeyUp as EventListener, { signal: this.abort.signal });
    if (typeof window !== 'undefined') {
      window.addEventListener('blur', () => this.trimKeys.clear(), { signal: this.abort.signal });
    }

    this.reset();
    // Live play: the first voyage draws a fresh seed when the helm is taken.
    // `reset` (a capture's rewind) freezes it again on the fixed one.
    this.voyage.frozen = false;
  }

  /** Points the reports somewhere else, e.g. once the real HUD exists. */
  setUiSink(sink: GameUiSink | null): void {
    this.ui = sink ?? NULL_UI_SINK;
    if (this.active) this.pushAllUi();
  }

  get isActive(): boolean {
    return this.active;
  }

  /** True while the bow sheet is drawn at a strength worth counting. */
  get bowSheetUp(): boolean {
    return this.bowSpray.sheetIntensity > 0.15;
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
    this.trim = TRIM_START;
    this.trimInput = 0;
    this.trimKeys.clear();
    this.pressure = 0;
    this.anchor = 'up';
    this.weighTimer = 0;
    this.wasTaut = false;
    this.snubTimer = 0;
    this.cargo = 0;
    this.gold = 0;
    this.trauma = 0;
    this.impactCooldown = 0;
    this.thunder.length = 0;
    this.grounded = false;
    this.clock = 0;
    this.uiTimer = 0;
    this.ironsTimer = 0;
    this.setStalled(false);
    this.ironsCooldown = 0;
    this.shoalCooldown = 0;
    this.random = mulberry32(this.seed);
    this.o.controller.setSailForce(0);
    this.burst.clear();
    this.sandPlume.clear();
    this.kick = 0;
    this.bowSpray.clear();
    this.beacon.resetClock();
    this.haulTime = -1;
    this.haulCask.visible = false;
    this.lootWait = 0;
    this.lootReveal = 1;
    this.clearShake();
    this.voyage.reset();
    this.anchorage.x = this.homeAnchorage.x;
    this.anchorage.z = this.homeAnchorage.z;
    this.clearVoyageCargo();
    if (this.tally !== null) {
      this.tally = null;
      this.ui.setTally?.(null);
    }

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
    this.loot.visible = active && this.lootReveal > 0 && this.voyage.phase === 'gather';
    this.glint.mesh.visible = this.loot.visible;
    if (!active) this.haulCask.visible = false;
    this.beacon.object.visible = active && this.voyage.phase !== 'landed';
    this.burst.mesh.visible = active && this.burst.alive;
    this.sandPlume.mesh.visible = active && this.sandPlume.alive;
    if (!active) return;

    this.clock += dt;
    if (!this.voyage.begun) this.beginVoyage();
    this.events.length = 0;
    this.voyage.advance(dt, this.events);
    for (const event of this.events) this.onVoyageEvent(event);
    const body = this.o.body;
    const wind = this.o.wind();
    // A squall's gusts, on the canvas only.
    const gusting = wind.speed * (1 + SQUALL_GUST * this.voyage.squallStrength());

    // --- sails ------------------------------------------------------------
    this.sailFill = approach(this.sailFill, SAIL_FILL[this.sail], SAIL_RATE * dt);
    this.offWind = angleOffWind(body.heading, wind.towardRad);

    // Sheets: keys and the held stick or pad rate together, clamped.
    const keyTrim =
      (this.trimKeys.has(keybinds.codeOf('trimOut')) ? 1 : 0)
      - (this.trimKeys.has(keybinds.codeOf('trimIn')) ? 1 : 0);
    const trimRate = Math.max(-1, Math.min(1, keyTrim + this.trimInput));
    if (trimRate !== 0) this.trim = Math.min(1, Math.max(0, this.trim + trimRate * TRIM_RATE * dt));

    // The wind on the canvas: true wind less the ship's own way.
    apparentWind(wind.towardRad, gusting, body.velocity.x, body.velocity.z, body.heading, this.apparent);
    this.pressure = sailPressure(this.apparent.speed, gusting, this.sailFill);
    let drive =
      overpress(this.pressure) *
      polarDrive(this.apparent.angleDeg) *
      trimFactor(this.trim, this.apparent.angleDeg);
    // The floor: set canvas keeps her moving on any heading outside the no-go
    // zone, faded out across its edge on the *true* wind so a ship whose own
    // way has drawn the apparent wind forward is not stalled by it.
    const pulling = smoothstep(IRONS_DEG - 8, IRONS_DEG + 15, this.offWind);
    drive = Math.max(drive, FLOOR_DRIVE * this.sailFill * pulling);
    this.drive = drive * (this.anchor === 'up' ? 1 : ANCHORED_SAIL);
    // Read by the controller on its next update — a frame of latency on a
    // quantity that itself takes seconds to change.
    this.o.controller.setSailForce(this.drive * SAIL_THRUST);

    this.forward.set(1, 0, 0).applyQuaternion(body.object.quaternion).setY(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
    this.forward.normalize();

    this.applyHeel();
    this.applyAnchor(dt, wind);
    this.updateOverboard(dt);
    this.updateLoot(dt);
    this.updateFeel(dt);
    this.updateMessages(dt);

    this.burst.update(dt, this.o.camera);
    this.sandPlume.update(dt, this.o.camera);
    this.sandPlume.mesh.visible = this.sandPlume.alive;
    this.burst.mesh.visible = this.burst.alive;
  }

  /**
   * The bow's running spray and sheet. Call after the buoyancy solve, with the
   * hull's world matrix refreshed: from `update`, which runs before the solve,
   * they sat on the previous frame's pose and lagged the stem by a frame.
   *
   * In every mode, not only at the helm: a ship making way throws water off
   * her stem whoever is watching, the Tour included. Hidden while she is hove
   * to and nothing she threw is still in the air.
   */
  updateBowSpray(dt: number): void {
    if (!(dt > 0) || !this.armed) return;
    const body = this.o.body;
    this.forward.set(1, 0, 0).applyQuaternion(body.object.quaternion).setY(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
    this.forward.normalize();
    this.bowSpray.update(
      dt,
      body.object,
      body.velocity,
      body.velocity.dot(this.forward),
      this.o.hullLength,
      this.o.hullBeam,
      this.o.camera,
      (x, z) => this.o.sampler.height(x, z),
    );
    this.bowSpray.mesh.visible = !this.bowSpray.idle;
  }

  /** Where the bow's flung drops leave foam (`physics/Splash`). */
  setSprayLanding(landing: SprayLanding | null): void {
    this.bowSpray.landing = landing;
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
   * Held sheet rate from a stick or a pad, -1 (haul in) .. 1 (ease out). A
   * latch, like the rudder: the caller sets it back to 0 on release. Q and E
   * add to it from the keyboard.
   */
  setTrimInput(rate: number): void {
    this.trimInput = Number.isFinite(rate) ? Math.max(-1, Math.min(1, rate)) : 0;
  }

  /** Sheets, 0 hauled in .. 1 eased out. */
  get sheetTrim(): number {
    return this.trim;
  }

  /**
   * The apparent wind on the canvas as of the last update: what the sails
   * should be drawn filling to. Live object; do not retain or modify it.
   */
  get apparentWind(): Readonly<ApparentWind> {
    return this.apparent;
  }

  /** Camera-shake trauma, 0..1, for rumble. Read only. */
  get shakeTrauma(): number {
    return this.trauma;
  }

  /**
   * A hull impact detected by `physics/Spray`: a bow slam, a beam slap, or green
   * water over the bow. Feeds the camera shake while the helm is taken and is
   * ignored otherwise — shaking an orbit or cinematic camera would be wrong.
   */
  reportImpact(kind: 'slam' | 'slap' | 'wash' | 'float', strength: number): void {
    if (!this.active || this.impactCooldown > 0) return;
    const s = Math.min(1, Math.max(0, strength));
    const threshold = IMPACT_THRESHOLD[kind];
    if (s <= threshold) return;
    this.addTrauma(IMPACT_TRAUMA[kind] * (0.35 + 0.65 * ((s - threshold) / (1 - threshold))));
    this.impactCooldown = IMPACT_COOLDOWN;
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
    if (this.o.splash) {
      // It falls from the cathead, which juts out over the windward bow, so the
      // splash is a metre or so clear of the planking rather than under the stem.
      const cat = this.o.hullBeam * 0.3 + 0.9;
      this.tmp.x += sx * cat * windward - this.forward.x;
      this.tmp.z += sz * cat * windward - this.forward.z;
      this.tmp.y = this.o.sampler.ready ? this.o.sampler.height(this.tmp.x, this.tmp.z) : 0;
      // A period bower: about 300 kg of iron let go from the cathead, some
      // three metres above the water, so it hits at sqrt(2 g h) ≈ 7.7 m/s with
      // the flukes half a metre across. A heavy splash: crown, jet and mist.
      this.o.splash(this.tmp.x, this.tmp.y, this.tmp.z, { radius: 0.5, speed: 7.7, mass: 300 });
    } else {
      this.burst.fire(this.tmp, 0.45, false);
    }
    if (this.atAnchorage() && this.cargo > 0 && this.voyage.phase !== 'landed') {
      this.land();
    } else {
      this.ui.toast("Anchor down. She'll swing to the wind", 'info');
    }
    this.pushTelemetry();
  }

  weighAnchor(): void {
    if (this.anchor !== 'down') return;
    this.anchor = 'weighing';
    this.weighTimer = WEIGH_SECONDS;
    this.o.audio?.playAnchorChain(WEIGH_SECONDS);
    this.ui.toast('Hauling the anchor up…', 'info');
  }

  /**
   * The rode as drawn by `scene/ship/Fittings`: where the anchor lies, which bow
   * it leads from, its length and how much of it the capstan has taken in (0..1).
   * Null while the anchor is up. Allocation-free.
   */
  anchorCable<T extends { x: number; z: number; side: number; rode: number; hauled: number }>(out: T): T | null {
    if (this.anchor === 'up') return null;
    out.x = this.anchorPoint.x;
    out.z = this.anchorPoint.z;
    out.side = this.anchorSide;
    out.rode = this.rode;
    out.hauled = this.anchor === 'weighing' ? 1 - Math.max(0, this.weighTimer) / WEIGH_SECONDS : 0;
    return out;
  }

  /** Moves the current cargo to `distance` metres off the bow. For tests. */
  placeTargetAhead(distance = 40, lateral = 0): void {
    const p = this.o.body.object.position;
    this.forward.set(1, 0, 0).applyQuaternion(this.o.body.object.quaternion).setY(0).normalize();
    const x = p.x + this.forward.x * distance - this.forward.z * lateral;
    const z = p.z + this.forward.z * distance + this.forward.x * lateral;
    this.placeLoot(x, z);
    this.lootWait = 0;
    this.lootReveal = 1;
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
      trim: this.trim,
      idealTrim: idealTrim(this.apparent.angleDeg),
      apparentAngleDeg: this.apparent.angleDeg,
      apparentKnots: this.apparent.speed * KNOTS_PER_MS,
      pressure: this.pressure,
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
      voyage: {
        phase: this.voyage.phase,
        number: this.voyage.number,
        seed: this.voyage.seed,
        frozen: this.voyage.frozen,
        seconds: this.voyage.clock,
        casks: VOYAGE_CASKS,
        squall: this.voyage.squallState,
        squallRain: this.voyage.squallRain(),
        squallIn: this.voyage.squallIn(),
        hours: this.voyage.hours(),
        anchorage: { x: this.anchorage.x, z: this.anchorage.z },
        tally: this.tally,
      },
    };
  }

  /** The `__ocean.gameplay` test surface. */
  get testHandle() {
    return {
      isActive: () => this.active,
      state: () => this.snapshot(),
      setSail: (s: SailSetting) => this.setSail(s),
      stepSail: (d: 1 | -1) => this.stepSail(d),
      setTrim: (trim: number) => {
        this.trim = Math.min(1, Math.max(0, trim));
      },
      setTrimInput: (rate: number) => this.setTrimInput(rate),
      dropAnchor: () => this.dropAnchor(),
      weighAnchor: () => this.weighAnchor(),
      toggleAnchor: () => this.toggleAnchor(),
      placeTargetAhead: (distance?: number, lateral?: number) =>
        this.placeTargetAhead(distance, lateral),
      reset: () => this.reset(),
      /** Test hooks: the events the sea and the bottom would raise. */
      strikeBottom: (speed = 4) => this.onGrounded(speed),
      thunder: (distanceM: number) => this.reportThunder(distanceM),
      /** The voyage: start one live from a seed, skip its clocks, land it. */
      voyage: {
        /** Unfreezes and starts a voyage from `seed`, as a visit would. */
        begin: (seed: number) => {
          this.voyage.frozen = false;
          this.startVoyage(seed);
        },
        /** Runs the voyage and day clocks on by `seconds` without moving the ship, raising what they raise. */
        skip: (seconds: number) => {
          for (let left = seconds; left > 0; left -= 0.5) {
            this.events.length = 0;
            this.voyage.advance(Math.min(0.5, left), this.events);
            for (const event of this.events) this.onVoyageEvent(event);
          }
          this.pushObjective();
        },
        /** Moves the anchorage under the ship, so the next anchor dropped lands her. */
        anchorHere: () => {
          this.anchorage.x = this.o.body.object.position.x;
          this.anchorage.z = this.o.body.object.position.z;
        },
        /** Puts a cask over the side as a squall would. */
        loseCask: () => this.loseCask(),
        sailAgain: () => this.sailAgain(),
      },
    };
  }

  dispose(): void {
    this.abort.abort();
    this.setActive(false);
    this.object.removeFromParent();
    this.beacon.dispose();
    this.glint.dispose();
    this.burst.dispose();
    this.sandPlume.dispose();
    this.bowSpray.dispose();
  }

  /* ------------------------------------------------------------ internals */

  private setActive(active: boolean): void {
    this.active = active;
    const controller = this.o.controller;
    controller.setThrottleKeysEnabled(!active);
    controller.setTurnBoost(active ? TURN_BOOST : 0);
    if (!active) {
      controller.setSailForce(0);
      this.kick = 0;
      this.setStalled(false);
      this.trimKeys.clear();
      this.trimInput = 0;
      this.clearShake();
      this.ui.setObjective(null);
      this.ui.setShipTelemetry(null);
      this.ui.setTally?.(null);
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
  private applyHeel(): void {
    if (this.sailFill <= 0.001) return;
    // Leeward: the component of the apparent wind's direction across the bow,
    // the wind the canvas actually feels.
    const wx = Math.cos(this.apparent.towardRad);
    const wz = Math.sin(this.apparent.towardRad);
    const along = wx * this.forward.x + wz * this.forward.z;
    this.lee.set(wx - along * this.forward.x, 0, wz - along * this.forward.z);
    const across = this.lee.length();
    if (across < 1e-3) return;
    this.lee.divideScalar(across);

    // Zero in irons (the canvas is luffing, not pulling), largest abeam, gone
    // dead downwind where the push is all along the bow. Pressure already
    // carries the sail fill, and past full it is what over-heels her in a gale:
    // the cue that half sail is now the faster call.
    const pulling = smoothstep(IRONS_DEG - 8, IRONS_DEG + 15, this.offWind);
    const side = across * pulling * Math.min(HEEL_PRESSURE_MAX, this.pressure);

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
    this.updateHaul(dt);
    // Homeward, the beacon stands over the anchorage instead of a cask.
    const homeward = this.voyage.phase === 'homeward';
    this.beacon.visibility = homeward ? 1 : this.lootReveal;
    this.beacon.update(
      dt,
      homeward ? this.beaconAt.set(this.anchorage.x, 0, this.anchorage.z) : lp,
      this.o.camera.position,
      this.reducedMotion,
    );
    this.glint.update(dt, lp, this.o.camera, this.reducedMotion, this.lootReveal);
    // The next cask is not there to be taken until it has appeared, and none
    // is out there once the voyage is homeward.
    if (this.lootWait > 0 || this.voyage.phase !== 'gather') return;

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
    // The cask dropped back as it is swung aboard, or the water it leaves: a
    // small splash where it floated.
    const surface = this.o.sampler.ready ? this.o.sampler.height(lp.x, lp.z) : lp.y;
    this.o.splash?.(lp.x, surface, lp.z, { radius: 0.35, speed: 3, mass: 40 });
    this.o.audio?.playSplash(0.7);
    this.o.audio?.playChime();
    this.addTrauma(0.18);
    // The cask is hauled aboard rather than vanishing: a copy of it arcs from
    // where it floated to the rail. The next cask is placed now, so the seeded
    // sequence is unchanged, but stays hidden for a beat and then fades in.
    if (!this.reducedMotion) {
      this.haulFrom.copy(lp);
      this.haulCask.position.copy(lp);
      this.haulCask.quaternion.copy(this.loot.quaternion);
      this.haulCask.scale.setScalar(1);
      this.haulCask.visible = true;
      this.haulTime = 0;
      this.lootWait = LOOT_BEAT;
      this.lootReveal = 0;
      this.loot.visible = false;
      this.glint.mesh.visible = false;
    }
    const haul = HAUL_LINES[(this.cargo - 1) % HAUL_LINES.length];
    this.ui.toast(`${haul}: ${this.lootName} · +${this.lootValue} gold`, 'reward');
    this.ui.setScore('Cargo', this.cargo);
    this.ui.setScore('Gold', this.gold);
    this.hold.push(this.lootValue);
    this.nextTarget(lp.x, lp.z);
    this.pushObjective();
  }

  /**
   * What she sails for after a cask comes aboard: the planned cask a lost one
   * displaced, the next of the voyage's casks, or home.
   */
  private nextTarget(fromX: number, fromZ: number): void {
    if (this.recovering) {
      this.recovering = false;
      const stashed = this.stashed;
      if (stashed !== null) {
        this.stashed = null;
        this.lootName = stashed.name;
        this.lootValue = stashed.value;
        this.placeLoot(stashed.x, stashed.z);
        return;
      }
    }
    if (this.laid < VOYAGE_CASKS) {
      this.laid += 1;
      this.spawnLoot(fromX, fromZ, this.laid === VOYAGE_CASKS ? LAST_CASK_BAND : CASK_BAND);
      return;
    }
    this.voyage.phase = 'homeward';
    this.ui.toast('All cargo aboard', 'info', { detail: 'Make for the cove and drop anchor' });
  }

  /* --------------------------------------------------------------- voyage */

  /**
   * The sky the voyage wants while the helm is taken (`Voyage.skyFor`): the
   * day's hour and the squall over the preset's weather. Null leaves the
   * preset's sky, as under a capture.
   */
  voyageSky(
    base: { coverage: number; kind: WeatherKind; intensity: number },
    fixedHour: number | null,
  ): Readonly<VoyageSky> | null {
    return this.active ? this.voyage.skyFor(base, fixedHour) : null;
  }

  /** The first time the helm is taken: a live voyage on a fresh seed, or the fixed one under a capture. */
  private beginVoyage(): void {
    if (this.voyage.frozen) {
      // The cask `reset` laid is the one captures were taken with: keep it.
      this.voyage.begin(this.voyage.seed, readVoyageSave().voyages + 1);
      this.logVoyage();
    } else {
      this.startVoyage(freshVoyageSeed());
    }
  }

  /** A new voyage from where she lies: a seed, its first cask, the log line. */
  private startVoyage(seed: number): void {
    this.voyage.begin(seed, readVoyageSave().voyages + 1);
    this.random = mulberry32(this.voyage.seed);
    this.cargo = 0;
    this.gold = 0;
    this.clearVoyageCargo();
    if (this.tally !== null) {
      this.tally = null;
      this.ui.setTally?.(null);
    }
    const ship = this.o.body.object.position;
    this.spawnLoot(ship.x, ship.z);
    this.lootWait = 0;
    this.lootReveal = 0;
    this.logVoyage();
    if (this.active) this.pushAllUi();
  }

  private clearVoyageCargo(): void {
    this.laid = 1;
    this.hold.length = 0;
    this.recovering = false;
    this.stashed = null;
    this.overboardTimer = 0;
    this.lostThisSquall = false;
    this.duskSaid = false;
  }

  /** The ship's log: the task, once, as the voyage opens. */
  private logVoyage(): void {
    const casks = COUNT_WORDS[VOYAGE_CASKS] ?? String(VOYAGE_CASKS);
    this.ui.toast(`Ship's log, voyage ${this.voyage.number}`, 'info', {
      detail: `${casks} casks adrift. Haul them aboard, then anchor off the cove before dark`,
      durationMs: 8000,
    });
  }

  /** "Sail again" on the tally. */
  private sailAgain(): void {
    if (this.voyage.phase !== 'landed') return;
    this.startVoyage(this.voyage.frozen ? this.voyage.seed : freshVoyageSeed());
  }

  private readonly onSailAgain = (): void => this.sailAgain();

  private atAnchorage(): boolean {
    const p = this.o.body.object.position;
    return Math.hypot(p.x - this.anchorage.x, p.z - this.anchorage.z) <= ANCHORAGE_RADIUS;
  }

  /** The anchor is down off the cove with cargo aboard: the voyage ends and the tally comes up. */
  private land(): void {
    this.voyage.phase = 'landed';
    const seconds = this.voyage.clock;
    let purse: number;
    let best: number | null;
    let newBest = false;
    let number: number;
    if (this.voyage.frozen) {
      // A capture's run is never kept.
      const save = readVoyageSave();
      purse = save.gold + this.gold;
      best = save.bestTimeS;
      number = this.voyage.number;
    } else {
      const kept = recordVoyage({ gold: this.gold, seconds, fullCargo: this.cargo >= VOYAGE_CASKS });
      purse = kept.save.gold;
      best = kept.save.bestTimeS;
      newBest = kept.newBest;
      number = kept.save.voyages;
    }
    this.tally = {
      voyage: number,
      cargo: this.cargo,
      casks: VOYAGE_CASKS,
      gold: this.gold,
      purse,
      seconds,
      bestSeconds: best,
      newBest,
    };
    this.o.audio?.playChime();
    this.ui.setObjective(null);
    this.ui.setTally?.(this.tally, this.onSailAgain);
  }

  private onVoyageEvent(event: VoyageEvent): void {
    const underway = this.voyage.phase !== 'landed';
    switch (event) {
      case 'squallWarning':
        this.ui.toast('A squall is coming', 'warning', { detail: 'Take in to half sail when the rain arrives' });
        break;
      case 'squallArrived':
        if (this.sail === 'full') {
          this.ui.toast('Squall! Take in to half sail', 'warning', {
            detail: 'Full sail in this wind can put a cask over the side',
          });
        }
        break;
      case 'squallPassed':
        if (underway) this.ui.toast("The squall's passed", 'info', { detail: 'Set full sail again' });
        break;
      case 'dusk':
        this.duskSaid = true;
        this.ui.toast("The sun's going down", 'warning', { detail: 'Make for the cove before dark' });
        break;
    }
    this.pushObjective();
  }

  /** Driven too hard in a squall for too long: a cask goes over the side. */
  private updateOverboard(dt: number): void {
    const strength = this.voyage.squallStrength();
    if (strength <= 0) this.lostThisSquall = false;
    const pressed =
      strength >= OVERBOARD_SQUALL &&
      this.pressure > OVERBOARD_PRESSURE &&
      this.anchor === 'up' &&
      this.cargo > 0 &&
      !this.lostThisSquall &&
      this.voyage.phase !== 'landed';
    this.overboardTimer = pressed ? this.overboardTimer + dt : 0;
    if (this.overboardTimer > OVERBOARD_SECONDS) this.loseCask();
  }

  /**
   * The last cask stowed goes over the side and floats astern, where it
   * becomes the target until it is hauled in again. Losing it costs the time
   * to turn back; sailing home without it costs the cask.
   */
  private loseCask(): void {
    if (this.cargo <= 0 || this.voyage.phase === 'landed') return;
    this.lostThisSquall = true;
    this.overboardTimer = 0;
    const value = this.hold.pop() ?? this.lootValue;
    this.cargo -= 1;
    this.gold = Math.max(0, this.gold - value);
    // The cask already in the water waits its turn.
    if (this.voyage.phase === 'gather' && !this.recovering) {
      const lp = this.loot.position;
      this.stashed = { x: lp.x, z: lp.z, name: this.lootName, value: this.lootValue };
    }
    const ship = this.o.body.object;
    this.forward.set(1, 0, 0).applyQuaternion(ship.quaternion).setY(0).normalize();
    const back = this.o.hullLength * 0.5 + OVERBOARD_ASTERN;
    const x = ship.position.x - this.forward.x * back;
    const z = ship.position.z - this.forward.z * back;
    this.lootName = 'the cask that went over the side';
    this.lootValue = value;
    this.placeLoot(x, z);
    this.recovering = true;
    this.voyage.phase = 'gather';
    this.lootWait = 0;
    this.lootReveal = 1;
    const surface = this.o.sampler.ready ? this.o.sampler.height(x, z) : 0;
    this.o.splash?.(x, surface, z, { radius: 0.35, speed: 4, mass: 40 });
    this.o.audio?.playSplash(0.8);
    this.ui.toast('A cask went over the side', 'danger', {
      detail: "It's floating astern. Take in sail and turn back for it",
    });
    this.ui.setScore('Cargo', this.cargo);
    this.ui.setScore('Gold', this.gold);
    this.pushObjective();
  }

  /** What is coming, for the objective's last line. */
  private voyageWarning(): string | undefined {
    const squall = this.voyage.squallState;
    if (squall === 'coming') return 'Squall coming. Half sail when it hits';
    if (squall === 'here') return this.sail === 'full' ? 'Squall: take in to half sail' : 'Squall: hold her steady, it will pass';
    if (this.duskSaid && this.voyage.phase === 'gather') return "Light's going. Leave time to make the cove";
    return undefined;
  }

  /**
   * Flies the hauled cask to the rail and brings the next one in. The arc is a
   * smoothstep along the line to a point over the deck, which moves with the
   * ship, plus a sine rise; the cask tumbles and shrinks as it goes over the
   * rail.
   */
  private updateHaul(dt: number): void {
    if (this.haulTime >= 0) {
      this.haulTime += dt;
      const u = Math.min(1, this.haulTime / HAUL_SECONDS);
      const e = u * u * (3 - 2 * u);
      const ship = this.o.body.object;
      // Over the waist, just above the rail.
      this.tmp.set(this.o.hullLength * 0.05, this.o.hullBeam * 0.4, 0).applyQuaternion(ship.quaternion).add(ship.position);
      const cask = this.haulCask;
      cask.position.lerpVectors(this.haulFrom, this.tmp, e);
      cask.position.y += Math.sin(u * Math.PI) * HAUL_ARC;
      cask.rotateZ(dt * 5);
      cask.scale.setScalar(1 - 0.75 * smoothstep(0.7, 1, u));
      if (u >= 1) {
        this.haulTime = -1;
        cask.visible = false;
      }
    }
    if (this.lootWait > 0) {
      this.lootWait = Math.max(0, this.lootWait - dt);
    } else if (this.lootReveal < 1) {
      this.lootReveal = Math.min(1, this.lootReveal + dt / LOOT_FADE_IN);
    }
  }

  /**
   * Picks the next cargo position from the seeded sequence, in navigable water.
   * With a `band`, the first 48 tries must also land that many metres from the
   * anchorage; without one (a voyage's first cask) the draws are the ones the
   * cargo sequence always made, so the fixed seed lays the same first cask.
   */
  private spawnLoot(fromX: number, fromZ: number, band: readonly [number, number] | null = null): void {
    const random = this.random;
    let x = fromX;
    let z = fromZ;
    let found = false;
    const tries = band === null ? 48 : 96;
    for (let attempt = 0; attempt < tries && !found; attempt++) {
      const angle = random() * Math.PI * 2;
      // Widen the search if the first rings keep landing on the reef.
      const radius = SPAWN_MIN + random() * SPAWN_SPAN * (1 + (attempt % 48) / 16);
      const cx = fromX + Math.cos(angle) * radius;
      const cz = fromZ + Math.sin(angle) * radius;
      if (Math.hypot(cx, cz) > SPAWN_LIMIT) continue;
      if (band !== null && attempt < 48) {
        const home = Math.hypot(cx - this.anchorage.x, cz - this.anchorage.z);
        if (home < band[0] || home > band[1]) continue;
      }
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
    if (grounded && !this.grounded) this.onGrounded(speed);
    if (grounded) this.trauma = Math.max(this.trauma, Math.min(0.45, speed * 0.1));
    this.grounded = grounded;
    this.shoalCooldown = Math.max(0, this.shoalCooldown - dt);
    if (depth < GROUND_DEPTH + 3 && speed > 1 && this.shoalCooldown === 0) {
      this.shoalCooldown = 8;
      this.ui.toast('Shallows! Turn her away', 'danger');
    }

    for (let i = this.thunder.length - 1; i >= 0; i--) {
      const rumble = this.thunder[i];
      rumble.delay -= dt;
      if (rumble.delay > 0) continue;
      this.thunder.splice(i, 1);
      this.trauma = Math.min(1, this.trauma + rumble.amount);
    }
    this.trauma = Math.max(0, this.trauma - TRAUMA_DECAY * dt);
    this.impactCooldown = Math.max(0, this.impactCooldown - dt);
    const shake = this.o.shake;
    if (!shake) return;
    const scale = this.reducedMotion ? 0 : this.shakeScale();
    const amount = this.trauma * this.trauma * scale;
    // The kick is a single dip of the view, not a shake: it eases back in about
    // a third of a second and is not part of the trauma budget.
    this.kick *= Math.exp(-dt * GROUND_KICK_RATE);
    const kick = this.kick * scale;
    if (amount <= 1e-5 && Math.abs(kick) <= 1e-5) {
      this.clearShake();
      return;
    }
    const t = this.clock;
    const n1 = Math.sin(t * 13.1 + 0.3) * 0.6 + Math.sin(t * 21.7 + 1.9) * 0.4;
    const n2 = Math.sin(t * 11.3 + 4.1) * 0.6 + Math.sin(t * 17.9 + 2.6) * 0.4;
    const n3 = Math.sin(t * 9.7 + 5.3) * 0.6 + Math.sin(t * 15.1 + 0.8) * 0.4;
    const n4 = Math.sin(t * 14.3 + 3.7) * 0.6 + Math.sin(t * 23.3 + 4.4) * 0.4;
    this.shakeOffset.set(0, n4 * SHAKE_LIFT * amount + kick * GROUND_KICK_LIFT, 0);
    shake.setShake(this.shakeOffset, n1 * SHAKE_PITCH * amount + kick, n2 * SHAKE_YAW * amount, n3 * SHAKE_ROLL * amount);
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
    // The HUD hears of a stall well before the toast, so a tack that only
    // brushes the wind does not flash it but a stall that is staying does.
    this.setStalled(this.ironsTimer > STALL_CUE_SECONDS);

    this.uiTimer += dt;
    if (this.uiTimer >= UI_INTERVAL) {
      this.uiTimer = 0;
      this.pushTelemetry();
      this.pushObjective();
    }
  }

  private setStalled(stalled: boolean): void {
    if (stalled === this.stalled) return;
    this.stalled = stalled;
    this.ui.setStalled?.(stalled);
  }

  private pushAllUi(): void {
    this.ui.setScore('Cargo', this.cargo);
    this.ui.setScore('Gold', this.gold);
    this.pushTelemetry();
    this.pushObjective();
    if (this.tally !== null) this.ui.setTally?.(this.tally, this.onSailAgain);
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
    const phase = this.voyage.phase;
    if (phase === 'landed') {
      this.ui.setObjective(null);
      return;
    }
    const ship = this.o.body.object.position;
    const lp = this.loot.position;
    let objective: GameObjective;
    if (phase === 'homeward') {
      const a = this.anchorage;
      objective = {
        title: 'Bring the cargo home',
        detail: `Drop anchor off the cove · ${this.cargo} ${this.cargo === 1 ? 'cask' : 'casks'} aboard`,
        bearingDeg: compassFromVector(a.x - ship.x, a.z - ship.z),
        distanceM: Math.hypot(a.x - ship.x, a.z - ship.z),
      };
    } else {
      // The count leads, so a narrow scroll wraps the name rather than the count.
      const name = this.recovering ? capitalise(this.lootName) : `${this.laid} of ${VOYAGE_CASKS}: ${this.lootName}`;
      objective = {
        title: 'Haul in the drifting cargo',
        detail: `${name} · ${this.lootValue} gold`,
        bearingDeg: compassFromVector(lp.x - ship.x, lp.z - ship.z),
        distanceM: Math.hypot(lp.x - ship.x, lp.z - ship.z),
      };
    }
    const warning = this.voyageWarning();
    if (warning !== undefined) objective.warning = warning;
    this.ui.setObjective(objective);
  }

  private addTrauma(amount: number): void {
    this.trauma = Math.min(1, this.trauma + amount);
    if (amount >= HIT_STOP_TRAUMA && !this.reducedMotion && this.shakeScale() > 0) {
      this.o.hitStop?.(HIT_STOP_BASE_MS + HIT_STOP_PER_TRAUMA_MS * amount);
    }
  }

  private shakeScale(): number {
    return THREE.MathUtils.clamp(this.o.shakeScale?.() ?? 1, 0, 1);
  }

  /**
   * Thunder from a strike `distanceM` away. The shake arrives with the sound
   * (`distance / 343` s late) and fades with range; a strike out of range, or
   * under reduced motion or a zero intensity, adds nothing. Ignored off the helm
   * like every impact.
   */
  reportThunder(distanceM: number): void {
    if (!this.active || this.reducedMotion) return;
    const near = 1 - THREE.MathUtils.clamp(distanceM / THUNDER_RANGE, 0, 1);
    const amount = THUNDER_TRAUMA * near * near;
    if (amount < 0.01) return;
    this.thunder.push({ delay: distanceM / SPEED_OF_SOUND, amount });
  }

  /** The keel has struck: the jolt, the thud, the plume and the news. */
  private onGrounded(speed: number): void {
    this.addTrauma(Math.min(0.7, 0.3 + speed * 0.08));
    this.o.audio?.hullImpact?.(Math.min(1, 0.4 + speed * 0.1), true);
    // The shoal warning has usually fired already; this says she has struck, and
    // holds it back so the two do not stack.
    this.shoalCooldown = 8;
    this.raiseSandPlume(speed);
    this.ui.toast("Aground! Turn her away and she'll slide free", 'danger');
  }

  /**
   * Churned sand where the keel strikes: the plume rises from the water at the
   * stem, with a splash on the surface. Sized by the speed she hit at, and
   * given a short dip of the camera.
   */
  private raiseSandPlume(speed: number): void {
    const strength = THREE.MathUtils.clamp(speed / 5, 0.35, 1);
    this.bowPoint(this.bow);
    const surface = this.o.sampler.ready ? this.o.sampler.height(this.bow.x, this.bow.z) : 0;
    this.sandPlume.fire(this.tmp2.set(this.bow.x, surface + 0.2, this.bow.z), strength, 'sand');
    this.o.splash?.(this.bow.x, surface, this.bow.z, { radius: 0.9, speed: 2 + speed * 0.6, mass: 400 });
    if (!this.reducedMotion) this.kick = -GROUND_KICK * strength;
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
    // The keys are the player's to rebind; see `input/keybinds`.
    switch (keybinds.actionFor(event.code)) {
      case 'sailMore':
        this.stepSail(1);
        break;
      case 'sailLess':
        this.stepSail(-1);
        break;
      case 'anchor':
        this.toggleAnchor();
        break;
      // Sheets: held, so the key's up event ends the haul.
      case 'trimIn':
      case 'trimOut':
        this.trimKeys.add(event.code);
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    this.trimKeys.delete(event.code);
  };
}

/* ---------------------------------------------------------------- helpers */

const COUNT_WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'] as const;

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
