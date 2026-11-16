import * as THREE from 'three/webgpu';
import {
  attribute,
  exp,
  float,
  mix,
  smoothstep,
  uniform,
  uniformArray,
  vec2,
  vec3,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import { CLASS_CLUSTER, CLASS_DROP, CLASS_MIST, CLASS_SHEET, particleVertex, sprayMaterial, sprayUniforms } from './sprayLook';

/**
 * Water thrown into the air by things working in a seaway: the hull slamming
 * into a crest, a beam sea slapping the topsides, green water over the bow
 * pouring back off the rails, and a buoy or a barrel smacking down off a swell.
 *
 * The project already had the two *other* things water does to a moving ship —
 * the wake it leaves (`physics/Wake`) and the wet it puts on the deck
 * (`scene/Wetness`). The steady bow wave a hull peels off at speed belongs to
 * `gameplay/effects.BowSpray`; this is the transient half — impacts.
 *
 * **The whole system is an event table.** Nothing here integrates a particle.
 * Probes ride each source; when one of them meets the surface faster than its
 * threshold, that writes a single record — where, when, how hard, which way,
 * what kind — into a small ring. Every particle is then a closed-form trajectory
 * off one of those records, evaluated in the vertex stage, with linear drag
 * toward the wind:
 *
 *     v(t) = v_inf + (v0 - v_inf) e^(-k t)
 *     p(t) = p0 + v_inf t + (v0 - v_inf)(1 - e^(-k t)) / k,   v_inf = (wind, -g/k)
 *
 * which gives a heavy droplet a near-ballistic arc (small k), a torn sheet a
 * shorter, slower one, and mist the drift-and-hang of fine water blown off a
 * crest (large k). One exponential per particle.
 *
 * **Why closed form rather than an integrator.** `resetClock(t)` has to land on
 * the exact state for time `t`, because the visual harness rests on a capture
 * being a function of the clock rather than of how many frames preceded it. An
 * event table can, because the only state is the table itself.
 *
 * ---
 *
 * **Why this used to draw nothing.** It was parked with the diagnosis that the
 * dynamic uniform-array index did not resolve per instance. It did: an impact
 * injected by hand at strength 1 draws a clear sheet. What the hull actually
 * produced was the problem. The probes sat on the waterline, the strength was
 * `speed / 6` — about 0.2 for a real slam — the launch was `sqrt(speed) * 2.6`,
 * about 3 m/s, so a slam threw a half-metre-high puff of 10–30 cm quads at a
 * fifth of their opacity, unlit and white-on-white over storm foam, from a
 * camera 40 m away. It drew; there was nothing to see. And the test that
 * measured it differenced two captures that, in a storm with rain and TRAA,
 * differ in a third of their pixels anyway, so the number it printed could not
 * distinguish the two cases. What replaced it: launch speeds and sizes from
 * real slam footage (sheets 3–8 m up, droplets to 10 m), strength from the
 * relative vertical speed with the hull's own forward speed added at the bow,
 * spray carrying the hull's velocity, three particle classes, lighting with
 * forward scatter, and probes placed where a hull meets water rather than on
 * the waterline.
 *
 * **Why it then read as cotton, and the look it has now.** Once it drew, the
 * blind review called the result "cotton-ball smoke puffs" and "paper blobs":
 * noise-cut discs with hard quad edges and 10–30 cm round droplets. The
 * particle's look now lives in `sprayLook`, shared with the bow spray: four
 * classes (droplet streaks, droplet clusters, sheets that erode through
 * ligaments into drops, mist), velocity-stretched relative to the camera, soft
 * against the water and hull behind them, and lit with a phase function and
 * the hull's shadow. Floats size their crown from their own radius. The event
 * table and the closed-form trajectories below are unchanged in kind.
 *
 * **No compute, no storage, no per-frame upload of positions.** The CPU writes
 * a few events a second into four small uniform arrays. This compiles on the
 * WebGL2 fallback unchanged.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * How many impacts are in flight at once. A ring; the oldest is overwritten.
 *
 * Four vec4 arrays of this length are the vertex-stage uniform budget, which
 * WebGL2 guarantees at 256 vectors — 160 leaves room for the camera and the
 * lighting.
 */
const EVENTS = 40;

/**
 * Particles per impact at full quality. Lower tiers draw fewer per impact, not
 * fewer impacts: see `setStrength`.
 */
const PER_EVENT = 64;

/** Longest a particle of any class lives, seconds. Mist hangs longest. */
const MAX_LIFE = 2.6;

/** Gravity, m/s^2. */
const GRAVITY = 9.81;

/** World-space distance past which spray is not drawn, metres. */
const DRAW_RANGE = 320;

/** Height of the rail above the waterline, as a fraction of hull length. */
const RAIL_FRACTION = 0.085;

/**
 * How far a wash falls from the rail to the sea, metres. The rail height of the
 * one hull in the scene; the shader needs it as a constant to know where the
 * curtain re-enters the water.
 */
const WASH_DROP = 2.3;

/**
 * Closing speed, m/s, below which a bow or shoulder impact throws no spray of
 * its own while the bow sheet (`gameplay/BowSheet`) is drawn.
 *
 * In a moderate sea a hull under way re-enters at the stem and shoulders every
 * pitch, at 1–2 m/s of closing speed with her forward speed on top, and each
 * re-entry wrote a slam. The bow sheet now answers exactly that: its intensity
 * follows the stem's immersion and plunge, saturating at 1.5 m/s. Both together
 * drew the same water twice — the sheet, and over it a V of slam quads. So
 * while the sheet is up, the bow and shoulders throw only a real slam, the
 * stem driving down well past what the sheet's plunge boost covers. The impact
 * is still reported (audio, shake, the wake's splash): only the particles go.
 */
const SHEET_COVERS_CLOSING = 2.2;

/** Minimum seconds between two emissions from the same probe. */
const PROBE_COOLDOWN = 0.22;

/** Float sources further than this from the viewer are not tested at all. */
const FLOAT_RANGE = 180;

/**
 * Seconds between two float splashes, from any float. Props share the ring
 * with the hull, and in a storm every buoy in range comes down hard off every
 * crest; without a budget they evict the hull's slams, which are the point.
 */
const FLOAT_INTERVAL = 0.35;

/** Seconds between two splashes from the same float. */
const FLOAT_COOLDOWN = 1.4;

/** Event kinds, as stored in the table. */
const KIND_SLAM = 0;
const KIND_SLAP = 1;
const KIND_WASH = 2;
const KIND_FLOAT = 3;

export type SprayImpactKind = 'slam' | 'slap' | 'wash' | 'float';

/** What a probe does when it meets the water. */
export type SprayProbeRole = 'bow' | 'shoulder' | 'side' | 'quarter' | 'float';

export interface SprayProbe {
  /** Attachment point in the object's local frame. */
  offset: THREE.Vector3;
  /**
   * How much water this point throws relative to the others, 0..1. The forefoot
   * throws most of it and the quarters almost none — a hull does not slam evenly
   * along its length.
   */
  weight: number;
  /** Defaults to `bow` for a probe on the centreline and `side` otherwise. */
  role?: SprayProbeRole;
}

/** One impact, reported to whoever drives audio, camera shake and foam. */
export interface SprayImpact {
  kind: SprayImpactKind;
  /** 0..1. */
  strength: number;
  /** World position at the water surface. */
  position: THREE.Vector3;
  /** -1 port, +1 starboard, 0 on the centreline (hull impacts only). */
  side: number;
  /** True for the ship, false for a floating prop. */
  hull: boolean;
  /** The source's size: a float's radius, or the hull's half-beam. Metres. */
  radius: number;
}

export interface SprayHullOptions {
  object: THREE.Object3D;
  probes: SprayProbe[];
  /** The body's world velocity, read every update. */
  velocity: THREE.Vector3;
  hullLength: number;
  hullBeam: number;
}

interface ProbeState {
  previousHeight: number;
  primed: boolean;
  /** Clock of the last emission. */
  last: number;
  /** Below green-water depth on the previous step. */
  deep: boolean;
}

interface Source {
  object: THREE.Object3D;
  probes: SprayProbe[];
  states: ProbeState[];
  velocity: THREE.Vector3 | null;
  hull: boolean;
  /** Particle size multiplier. */
  scale: number;
  railHeight: number;
  hullLength: number;
  hullBeam: number;
}

/** Deterministic hash in [0,1) from two integers. */
function hash2(a: number, b: number): number {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x165667b1, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 15), 0x27d4eb2f);
  return ((h ^ (h >>> 13)) >>> 0) / 4294967296;
}


export class Spray {
  readonly mesh: THREE.Mesh;

  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly depthMaterial: THREE.MeshBasicNodeMaterial;
  private readonly geometry: THREE.InstancedBufferGeometry;

  /** xyz = impact point, w = emission time. */
  private readonly uOrigin = uniformArray(
    Array.from({ length: EVENTS }, () => new THREE.Vector4(0, 0, 0, -1000)),
  );
  /** xyz = velocity carried from the source, w = strength 0..1. */
  private readonly uLaunch = uniformArray(Array.from({ length: EVENTS }, () => new THREE.Vector4()));
  /** xyz = horizontal outward unit vector, w = kind. */
  private readonly uFrame = uniformArray(Array.from({ length: EVENTS }, () => new THREE.Vector4()));
  /** xyz = along-hull axis times its spread (unit unless a wash), w = size scale. */
  private readonly uAlong = uniformArray(Array.from({ length: EVENTS }, () => new THREE.Vector4()));
  private readonly uClock = uniform(0);
  /** Master scale on how much is drawn, from the quality tier. */
  private readonly uStrength = uniform(1);

  private hullSource: Source | null = null;
  private readonly floats: Source[] = [];
  private cursor = 0;
  private clock = 0;
  private readonly previousEye = new THREE.Vector3();
  private eyePrimed = false;
  private lastFloat = -1000;
  private eventSerial = 0;
  /** True while the bow sheet is drawn; see `SHEET_COVERS_CLOSING`. */
  private bowSheet = false;
  private disposed = false;
  private listener: ((impact: SprayImpact) => void) | null = null;
  private readonly impact: SprayImpact = {
    kind: 'slam',
    strength: 0,
    position: new THREE.Vector3(),
    side: 0,
    hull: true,
    radius: 1,
  };

  constructor() {
    this.geometry = buildGeometry();
    this.material = sprayMaterial(() => this.buildVertex(), false);

    // The dense part of every particle, again, into the depth buffer only.
    //
    // This is the second half of why spray was invisible, and the half that
    // only shows in weather. Aerial perspective is applied in post by
    // `VolumetricFog`, from *scene depth*. A particle that writes no depth is
    // fogged as though it were whatever is behind it — the sea a few hundred
    // metres off, or the sky at the far plane — and in the storm preset's fog
    // that is total: a sheet of spray twenty metres from the camera came out
    // exactly the colour of the fog bank. Writing the cores' depth lets the fog
    // see the spray at its own distance. In a pass of its own rather than by
    // depth-writing the colour pass, so the faint mist (which never clears the
    // test) still draws. Ordered after the ocean so the water's refraction
    // never sees it.
    this.depthMaterial = sprayMaterial(() => this.buildVertex(), true);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'spray';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 6;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;

    const depth = new THREE.Mesh(this.geometry, this.depthMaterial);
    depth.name = 'spray-depth';
    depth.frustumCulled = false;
    depth.renderOrder = 5;
    depth.castShadow = false;
    depth.receiveShadow = false;
    this.mesh.add(depth);
  }

  /** Called once per impact, hull or prop. Replaces any previous listener. */
  onImpact(listener: ((impact: SprayImpact) => void) | null): void {
    this.listener = listener;
  }

  /**
   * Attaches the hull. Replaces any previous hull.
   *
   * The object's world matrix is read at update time, so the caller only has to
   * have run the buoyancy solve first.
   */
  setHull(options: SprayHullOptions): void {
    this.hullSource = {
      object: options.object,
      probes: options.probes.map((p) => ({
        offset: p.offset.clone(),
        weight: p.weight,
        role: p.role ?? (Math.abs(p.offset.z) < 1e-3 ? 'bow' : 'side'),
      })),
      states: options.probes.map(() => freshState()),
      velocity: options.velocity,
      hull: true,
      scale: 1,
      railHeight: options.hullLength * RAIL_FRACTION,
      hullLength: options.hullLength,
      hullBeam: options.hullBeam,
    };
  }

  /**
   * Adds a floating prop. One probe at its origin, which for every floater in
   * the scene is the waterline; small splashes, scaled by its radius.
   */
  addFloat(object: THREE.Object3D, radius: number, velocity: THREE.Vector3 | null = null): void {
    this.floats.push({
      object,
      probes: [{ offset: new THREE.Vector3(0, radius * 0.15, 0), weight: 1, role: 'float' }],
      states: [freshState()],
      velocity,
      hull: false,
      // The radius itself: a float's crown is sized from it in the shader.
      scale: radius,
      railHeight: 0,
      hullLength: radius * 2,
      hullBeam: radius * 2,
    });
  }

  /**
   * Master scale. 0 draws nothing; below 1 also draws fewer particles per
   * impact, which is how the quality tiers set the count.
   */
  setStrength(strength: number): void {
    const s = Math.max(0, strength);
    this.uStrength.value = s;
    const perEvent = s <= 0 ? 0 : Math.max(8, Math.round(PER_EVENT * Math.min(1, s)));
    // Instances are laid out event-fastest (see `buildGeometry`), so trimming
    // the count trims particles from every impact evenly.
    this.geometry.instanceCount = Math.max(1, EVENTS * perEvent);
  }

  /**
   * Whether the bow sheet (`gameplay/BowSheet`) is drawn this frame, which
   * takes the routine pitching re-entries at the bow off this system.
   */
  setBowSheet(active: boolean): void {
    this.bowSheet = active;
  }

  /** Near-surface wind, as a world velocity in m/s. */
  setWind(x: number, z: number): void {
    // Shared with the bow spray, which drifts in the same wind.
    (sprayUniforms.wind.value as THREE.Vector3).set(x, 0, z);
  }

  /**
   * Key light and sky. Spray is lit like the foam it came from: by the sun, by
   * the sky, and very strongly from behind — a sheet of droplets between the
   * viewer and the sun is the brightest thing on the sea.
   */
  setLighting(sunDirection: THREE.Vector3, sunColor: THREE.Color, level: number, sky: THREE.Color): void {
    // Shared with the bow spray (`sprayLook.sprayUniforms`), so both are lit by
    // the same sun in the same frame.
    (sprayUniforms.sunDirection.value as THREE.Vector3).copy(sunDirection).normalize();
    (sprayUniforms.sunColor.value as THREE.Color).copy(sunColor).multiplyScalar(Math.max(0, level));
    (sprayUniforms.skyColor.value as THREE.Color).copy(sky);
  }

  /**
   * The lens, the drawing-buffer height and the eye's motion: the first two set
   * the one-pixel floor on droplet width, the last makes the streaks motion
   * blur *relative to the camera*. Shared with the bow spray.
   */
  setView(fovDegrees: number, heightPx: number, eye: THREE.Vector3, dt: number): void {
    sprayUniforms.pixelAngle.value = (2 * Math.tan((fovDegrees * Math.PI) / 360)) / Math.max(1, heightPx);
    const velocity = sprayUniforms.cameraVelocity.value as THREE.Vector3;
    if (this.eyePrimed && dt > 0) {
      velocity.copy(eye).sub(this.previousEye).divideScalar(dt);
      // Faster than any rig moves is a cut, not a pan: no blur across it.
      if (velocity.lengthSq() > 60 * 60) velocity.set(0, 0, 0);
    } else if (!this.eyePrimed) {
      velocity.set(0, 0, 0);
    }
    this.previousEye.copy(eye);
    this.eyePrimed = true;
  }

  /** Impacts with particles still in the air. Exposed for the test. */
  liveEvents(): number {
    let live = 0;
    for (let i = 0; i < EVENTS; i++) {
      const origin = (this.uOrigin.array as THREE.Vector4[])[i];
      const age = this.clock - origin.w;
      if (age >= 0 && age <= MAX_LIFE) live++;
    }
    return live;
  }

  /**
   * Advances the clock and tests every probe for an impact.
   *
   * @param surfaceHeight Wave elevation at a world XZ — the same field the
   *                      buoyancy solve floated the hull on.
   * @param viewer        The eye, so distant props are not tested at all.
   */
  update(
    dt: number,
    elapsed: number,
    surfaceHeight: (x: number, z: number) => number,
    viewer: THREE.Vector3 | null = null,
  ): void {
    if (this.disposed) return;
    this.clock = elapsed;
    this.uClock.value = elapsed;
    this.updateHullShadow();
    if (dt <= 0 || this.uStrength.value <= 0) return;

    if (this.hullSource) this.testSource(this.hullSource, dt, surfaceHeight);
    for (const source of this.floats) {
      if (viewer) {
        const p = source.object.position;
        if (Math.hypot(p.x - viewer.x, p.z - viewer.z) > FLOAT_RANGE) {
          // Unprime rather than keep a stale height, so walking back into range
          // cannot read as a slam.
          source.states[0].primed = false;
          continue;
        }
      }
      if (!source.object.visible) continue;
      this.testSource(source, dt, surfaceHeight);
    }
  }

  /**
   * Clears every event and unprimes every probe.
   *
   * The unpriming matters as much as the clearing: a probe whose remembered
   * height is from before a rewind would compute its crossing speed across the
   * discontinuity, so the first step after a reset observes and emits nothing.
   */
  resetClock(elapsed: number): void {
    this.clock = elapsed;
    this.uClock.value = elapsed;
    this.cursor = 0;
    this.eventSerial = 0;
    this.lastFloat = -1000;
    this.eyePrimed = false;
    for (let i = 0; i < EVENTS; i++) {
      (this.uOrigin.array as THREE.Vector4[])[i].set(0, 0, 0, -1000);
      (this.uLaunch.array as THREE.Vector4[])[i].set(0, 0, 0, 0);
    }
    const sources = this.hullSource ? [this.hullSource, ...this.floats] : this.floats;
    for (const source of sources) {
      for (const state of source.states) Object.assign(state, freshState());
    }
  }

  /**
   * The box the spray's sun-shadow test runs against: the hull below its rail,
   * drawn in from the full length and beam because the stem and the quarters,
   * where most spray is, are where a box overstates a hull most.
   */
  private updateHullShadow(): void {
    const half = sprayUniforms.hullHalf.value as THREE.Vector3;
    const source = this.hullSource;
    if (!source) {
      half.set(0, 0, 0);
      return;
    }
    (sprayUniforms.hullInverse.value as THREE.Matrix4).copy(source.object.matrixWorld).invert();
    const top = source.railHeight + 0.4;
    const bottom = -2.5;
    (sprayUniforms.hullCenter.value as THREE.Vector3).set(0, (top + bottom) / 2, 0);
    half.set(source.hullLength * 0.45, (top - bottom) / 2, source.hullBeam * 0.42);
  }

  dispose(): void {
    this.disposed = true;
    this.geometry.dispose();
    this.material.dispose();
    this.depthMaterial.dispose();
  }

  // ------------------------------------------------------------------ emission

  private testSource(
    source: Source,
    dt: number,
    surfaceHeight: (x: number, z: number) => number,
  ): void {
    const object = source.object;
    _forward.set(1, 0, 0).applyQuaternion(object.quaternion).setY(0);
    if (_forward.lengthSq() < 1e-6) _forward.set(1, 0, 0);
    _forward.normalize();
    _starboard.set(-_forward.z, 0, _forward.x);
    const forwardSpeed = source.velocity ? source.velocity.dot(_forward) : 0;

    for (let i = 0; i < source.probes.length; i++) {
      const probe = source.probes[i];
      const state = source.states[i];
      _probeWorld.copy(probe.offset).applyMatrix4(object.matrixWorld);
      const surface = surfaceHeight(_probeWorld.x, _probeWorld.z);
      const height = _probeWorld.y - surface;

      if (!state.primed) {
        state.previousHeight = height;
        state.primed = true;
        state.deep = false;
        continue;
      }

      // Closing speed between the probe and the water, m/s. Positive when the
      // two are meeting, whichever of them is doing the moving — a hull pitching
      // down into a crest and a crest rising up a stationary hull's side are the
      // same impact to the water.
      const closing = (state.previousHeight - height) / dt;
      const crossed = state.previousHeight > 0 && height <= 0;
      state.previousHeight = height;

      const role = probe.role ?? 'side';

      // Green water: the bow buried past its rail. Detected on the bow probe
      // only, on the way down.
      if (source.hull && role === 'bow') {
        const deep = height < -source.railHeight * 0.55;
        if (deep && !state.deep && closing > 0.6) this.emitWash(source, closing);
        state.deep = deep;
      }

      if (!crossed) continue;
      if (this.clock - state.last < (role === 'float' ? FLOAT_COOLDOWN : PROBE_COOLDOWN)) continue;
      if (role === 'float' && this.clock - this.lastFloat < FLOAT_INTERVAL) continue;

      // The bow meets the water at the hull's own speed as well as its pitch:
      // driving into a head sea at six knots is a harder slam than pitching on
      // the spot, by about a third of the forward speed.
      let effective = closing;
      if (role === 'bow' || role === 'shoulder') effective += 0.3 * Math.max(0, forwardSpeed);
      const threshold = role === 'float' ? 0.75 : role === 'side' ? 1.0 : 1.2;
      const strength = Math.min(1, Math.max(0, (effective - threshold) / 3.6)) * probe.weight;
      if (strength < 0.05) continue;
      state.last = this.clock;

      _at.set(_probeWorld.x, surface + 0.05, _probeWorld.z);
      // The bow sheet's job, unless this is a real slam.
      const sheetCovers =
        source.hull && this.bowSheet && (role === 'bow' || role === 'shoulder') && closing < SHEET_COVERS_CLOSING;
      const sideSign = Math.abs(probe.offset.z) < 1e-3 ? 0 : Math.sign(probe.offset.z);
      const carry = source.velocity ?? _zero;

      if (role === 'float') {
        this.lastFloat = this.clock;
        _out.set(1, 0, 0);
        this.write(_at, carry, _out, _forward, 1, KIND_FLOAT, strength, source.scale);
        this.report('float', strength, _at, 0, false, source.scale);
      } else if (role === 'bow') {
        // A stem throws a V: one sheet off each bow, forward and out.
        for (const s of sheetCovers ? [] : [-1, 1]) {
          _out.copy(_forward).multiplyScalar(0.55).addScaledVector(_starboard, s).normalize();
          this.write(_at, carry, _out, _forward, 1, KIND_SLAM, strength, source.scale);
        }
        this.report('slam', strength, _at, 0, true);
      } else if (role === 'side') {
        // Beam-on: the sheet climbs the topsides and fans along them.
        _out.copy(_starboard).multiplyScalar(sideSign);
        this.write(_at, carry, _out, _forward, 1, KIND_SLAP, strength, source.scale);
        this.report('slap', strength, _at, sideSign, true);
      } else {
        // Shoulder or quarter: out on its own side, a little forward.
        _out.copy(_starboard).multiplyScalar(sideSign).addScaledVector(_forward, 0.35).normalize();
        if (!sheetCovers) this.write(_at, carry, _out, _forward, 1, KIND_SLAM, strength, source.scale);
        this.report('slam', strength, _at, sideSign, true);
      }
    }
  }

  /**
   * Water shipped over the bow running off both rails.
   *
   * A line source along each rail rather than a point: what a viewer sees is a
   * curtain pouring off the side for half the length of the foredeck.
   */
  private emitWash(source: Source, closing: number): void {
    const strength = Math.min(1, 0.35 + closing / 6);
    const object = source.object;
    for (const s of [-1, 1]) {
      _local.set(source.hullLength * 0.18, source.railHeight, s * source.hullBeam * 0.5);
      _at.copy(_local).applyMatrix4(object.matrixWorld);
      _out.copy(_starboard).multiplyScalar(s);
      this.write(
        _at,
        source.velocity ?? _zero,
        _out,
        _forward,
        source.hullLength * 0.45,
        KIND_WASH,
        strength,
        1,
      );
    }
    _at.set(0, source.railHeight, source.hullLength * 0.3).applyMatrix4(object.matrixWorld);
    this.report('wash', strength, _at, 0, true);
  }

  private write(
    at: THREE.Vector3,
    carry: THREE.Vector3,
    out: THREE.Vector3,
    along: THREE.Vector3,
    spread: number,
    kind: number,
    strength: number,
    scale: number,
  ): void {
    const slot = this.cursor % EVENTS;
    this.cursor++;
    this.eventSerial++;
    (this.uOrigin.array as THREE.Vector4[])[slot].set(at.x, at.y, at.z, this.clock);
    // Spray leaves a moving hull at the hull's velocity plus its own; the drag
    // then hands it over to the wind. Vertical carry is dropped — the hull's
    // heave is already in the closing speed that set the strength.
    (this.uLaunch.array as THREE.Vector4[])[slot].set(carry.x * 0.9, 0, carry.z * 0.9, strength);
    (this.uFrame.array as THREE.Vector4[])[slot].set(out.x, 0, out.z, kind);
    // A per-event twist on the along axis, so two impacts in a row do not throw
    // identical fans.
    const twist = (hash2(this.eventSerial, 5) - 0.5) * 0.3;
    _twisted.copy(along).addScaledVector(out, twist).setY(0).normalize().multiplyScalar(spread);
    (this.uAlong.array as THREE.Vector4[])[slot].set(_twisted.x, 0, _twisted.z, scale);
  }

  private report(
    kind: SprayImpactKind,
    strength: number,
    at: THREE.Vector3,
    side: number,
    hull: boolean,
    radius = this.hullSource ? this.hullSource.hullBeam * 0.5 : 1,
  ): void {
    if (!this.listener) return;
    this.impact.radius = radius;
    this.impact.kind = kind;
    this.impact.strength = strength;
    this.impact.position.copy(at);
    this.impact.side = side;
    this.impact.hull = hull;
    this.listener(this.impact);
  }

  // ------------------------------------------------------------------ shading

  /**
   * The trajectory, size and opacity of one particle, in the vertex stage; the
   * rest — camera-facing stretch, lighting, varyings — is `particleVertex`.
   */
  private buildVertex(): any {
    const eventIndex: any = (attribute('sprayEvent', 'float') as any).toInt();
    const seed: any = attribute('spraySeed', 'vec4');
    const cls: any = attribute('sprayClass', 'float');

    const origin: any = (this.uOrigin.element(eventIndex) as any).toVar('sprayOrigin');
    const launch: any = (this.uLaunch.element(eventIndex) as any).toVar('sprayLaunch');
    const frame: any = (this.uFrame.element(eventIndex) as any).toVar('sprayFrame');
    const alongRaw: any = (this.uAlong.element(eventIndex) as any).toVar('sprayAlongRaw');

    const kind: any = frame.w;
    const isSlap: any = smoothstep(0.5, 0.6, kind).mul(smoothstepDown(kind, 1.4, 1.5)).toVar('spraySlap');
    const isWash: any = smoothstep(1.5, 1.6, kind).mul(smoothstepDown(kind, 2.4, 2.5)).toVar('sprayWash');
    const isFloat: any = smoothstep(2.5, 2.6, kind).toVar('sprayFloatK');
    const isHull: any = isFloat.oneMinus().toVar('sprayHullK');

    // Four classes, interleaved through the instance order by `buildGeometry`.
    const isDrop: any = smoothstepDown(cls, 0.4, 0.5).toVar('sprayDrop');
    const isSheet: any = smoothstep(0.5, 0.6, cls).mul(smoothstepDown(cls, 1.4, 1.5)).toVar('spraySheet');
    const isMist: any = smoothstep(1.5, 1.6, cls).mul(smoothstepDown(cls, 2.4, 2.5)).toVar('sprayMist');
    const isCluster: any = smoothstep(2.5, 2.6, cls).toVar('sprayCluster');
    const isBead: any = isDrop.add(isCluster).toVar('sprayBead');

    const strength: any = launch.w.toVar('sprayStrength');
    // Hull events: a size multiplier (1). Floats: the object's radius, metres.
    const scale: any = alongRaw.w.toVar('sprayScale');
    const spreadLen: any = vec2(alongRaw.x, alongRaw.z).length().max(1e-4).toVar('spraySpreadLen');
    const alongDir: any = vec3(alongRaw.x, 0, alongRaw.z).div(spreadLen).toVar('sprayAlongDir');

    // Lifetimes, seconds. Ryerson measured slam spray events on a 115 m cutter
    // averaging 2.73 s from first water to last (Atmos. Res., 1995); the mist
    // here hangs for 1.6–2.5 s after a launch that takes the first few tenths.
    // A floater's splash is a ballistic hop of a few tenths of a second.
    const lifeSpan: any = isDrop
      .mul(seed.w.mul(0.5).add(0.85))
      .add(isCluster.mul(seed.w.mul(0.5).add(0.9)))
      .add(isSheet.mul(seed.w.mul(0.45).add(0.8)))
      .add(isMist.mul(seed.w.mul(0.9).add(1.6)))
      .mul(isWash.mul(0.35).add(1))
      .mul(isFloat.mul(-0.55).add(1))
      .toVar('sprayLifeSpan');
    const age: any = this.uClock.sub(origin.w).toVar('sprayAge');
    const life: any = age.div(lifeSpan).toVar('sprayLife');
    const alive: any = life
      .greaterThanEqual(0)
      .and(life.lessThanEqual(1))
      .and(strength.greaterThan(0))
      .select(float(1), float(0))
      .toVar('sprayAlive');

    // Launch, in the event's own frame: up, outward, and along the hull.
    // Figures are from slam footage of small ships: a sheet stands three to
    // eight metres, the fastest droplets reach ten.
    const speedScale: any = strength.mul(0.9).add(0.55).toVar('spraySpeedScale');
    const lateral: any = seed.x.sub(0.5).mul(2).toVar('sprayLateral');
    const upHull: any = isDrop
      .mul(seed.y.mul(7).add(4.5))
      .add(isCluster.mul(seed.y.mul(5.5).add(4)))
      .add(isSheet.mul(seed.y.mul(4).add(3.2)))
      .add(isMist.mul(seed.y.mul(2).add(1.5)))
      .mul(speedScale)
      // A slap climbs the topsides.
      .mul(isSlap.mul(0.35).add(1));
    // A floater coming down off a swell at 1–4 m/s throws a crown a few tens of
    // centimetres high and beads to about a metre: v²/2g of the launch.
    const upFloat: any = isBead
      .mul(seed.y.mul(2).add(1.6))
      .add(isSheet.mul(seed.y.mul(1.3).add(1)))
      .add(isMist.mul(seed.y.mul(0.5).add(0.5)))
      .mul(strength.mul(0.9).add(0.45));
    const up: any = mix(upHull, upFloat, isFloat).toVar('sprayUp');
    const outHull: any = isDrop
      .mul(seed.z.mul(3.6).add(2))
      .add(isCluster.mul(seed.z.mul(3).add(1.8)))
      .add(isSheet.mul(seed.z.mul(2.2).add(1.4)))
      .add(isMist.mul(seed.z.mul(1.2).add(0.8)))
      .mul(speedScale)
      .mul(isSlap.mul(-0.55).add(1));
    const outFloat: any = isBead
      .mul(seed.z.mul(1.3).add(0.5))
      .add(isSheet.mul(seed.z.mul(0.5).add(0.35)))
      .add(isMist.mul(seed.z.mul(0.4).add(0.3)))
      .mul(strength.mul(0.8).add(0.5));
    const outward: any = mix(outHull, outFloat, isFloat).toVar('sprayOut');
    const alongSpeed: any = lateral
      .mul(isDrop.mul(3.2).add(isCluster.mul(2.8)).add(isSheet.mul(2.4)).add(isMist.mul(2)))
      .mul(isSlap.mul(0.5).add(1))
      .mul(isHull)
      .toVar('sprayAlongSpeed');

    // A float throws a crown: radial, every azimuth.
    const azimuth: any = seed.x.mul(Math.PI * 2).toVar('sprayAzimuth');
    const radial: any = vec3(azimuth.cos(), 0, azimuth.sin());
    const outDir: any = mix(frame.xyz, radial, isFloat).toVar('sprayOutDir');

    // Green water does not leave upward: it pours off the rail.
    const v0Spray: any = launch.xyz
      .add(outDir.mul(outward))
      .add(vec3(0, up, 0) as any)
      .add(alongDir.mul(alongSpeed));
    const v0Wash: any = launch.xyz
      .add(outDir.mul(seed.z.mul(1.4).add(0.6)))
      .add(vec3(0, seed.y.mul(0.9).sub(0.3), 0));
    const v0: any = (mix(v0Spray as any, v0Wash as any, isWash) as any).toVar('sprayV0');

    // Spawn: a wash is a line along the rail, a float a ring at its own radius,
    // everything else a short line along the hull at the impact.
    const p0Hull: any = alongRaw.xyz
      .mul(seed.x.sub(0.5))
      .mul(isWash)
      .add(alongDir.mul(lateral.mul(0.6)).mul(isWash.oneMinus()))
      .add(outDir.mul(seed.z.mul(0.3)));
    const p0Float: any = outDir.mul(scale.mul(seed.z.mul(0.25).add(0.8)));
    const p0: any = origin.xyz.add(mix(p0Hull, p0Float, isFloat)).toVar('sprayP0');

    // Linear drag toward the wind. See the header.
    const k: any = isDrop
      .mul(0.35)
      .add(isCluster.mul(0.5))
      .add(isSheet.mul(0.9))
      .add(isMist.mul(2.2))
      .toVar('sprayDrag');
    const wind: any = sprayUniforms.wind;
    const vInf: any = (vec3(wind.x, float(-GRAVITY).div(k), wind.z) as any).toVar('sprayVInf');
    const decay: any = exp(k.negate().mul(age.max(0))).toVar('sprayDecay');
    const centre: any = p0
      .add(vInf.mul(age))
      .add(v0.sub(vInf).mul(decay.oneMinus().div(k)))
      .toVar('sprayCentre');
    const velocity: any = vInf.add(v0.sub(vInf).mul(decay)).toVar('sprayVelocity');

    // Widths, metres. Droplets are the 2–6 cm class that streaks visibly (see
    // `sprayLook`); a sheet is a metre-scale curtain; mist covers metres. A
    // float's crown is sized to the float.
    const sizeHull: any = isDrop
      .mul(seed.w.mul(0.035).add(0.025))
      .add(isCluster.mul(seed.w.mul(0.35).add(0.35)))
      .add(isSheet.mul(seed.w.mul(0.8).add(0.6)))
      .add(isMist.mul(seed.w.mul(1.6).add(1.6)))
      .mul(strength.mul(0.6).add(0.6));
    const sizeFloat: any = isDrop
      .mul(seed.w.mul(0.012).add(0.012))
      .add(isCluster.mul(seed.w.mul(0.2).add(0.25)).mul(scale))
      .add(isSheet.mul(seed.w.mul(0.25).add(0.3)).mul(scale))
      .add(isMist.mul(seed.w.mul(0.6).add(0.9)).mul(scale))
      .mul(strength.mul(0.5).add(0.7));
    // Sheets and mist spread as they disperse; droplets do not grow.
    const growHull: any = isDrop
      .add(isCluster.mul(life.mul(0.6).add(0.8)))
      .add(isSheet.mul(life.mul(1.5).add(0.55)))
      .add(isMist.mul(life.mul(1.8).add(0.45)));
    const growFloat: any = isDrop
      .add(isCluster.mul(life.mul(0.4).add(0.8)))
      .add(isSheet.mul(life.mul(0.6).add(0.7)))
      .add(isMist.mul(life.mul(0.8).add(0.6)));
    const grow: any = mix(sizeHull.mul(growHull), sizeFloat.mul(growFloat), isFloat)
      .mul(smoothstep(0, 0.08, life).mul(0.6).add(0.4))
      .toVar('sprayGrow');
    const width: any = grow.mul(this.uStrength.min(1).max(0.5)).toVar('sprayWidth');

    // Opacity. A droplet is dense and small; a sheet is thick at first and
    // tears (the erosion does the thinning); mist is barely there.
    const density: any = isDrop
      .mul(0.9)
      .add(isCluster.mul(0.9))
      // A sheet thins as it stretches: it starts dense and ends half as opaque.
      // A float's crown is a thinner wall than a hull's curtain.
      .add(isSheet.mul(life.mul(-0.5).add(0.9)).mul(isFloat.mul(-0.35).add(1)))
      .add(isMist.mul(mix(float(0.14), float(0.05), isFloat)))
      .toVar('sprayDensity');
    const fadeIn: any = smoothstep(0, 0.04, life);
    const fadeOut: any = smoothstepDown(life, isMist.mul(-0.35).add(0.75), 1);
    // Back into the sea: a particle that falls below where it started is gone.
    // A wash starts a rail's height up, so its sea is that much lower.
    const waterLine: any = origin.y.sub(isWash.mul(WASH_DROP)).toVar('sprayWaterLine');
    const submerged: any = smoothstepDown(waterLine.sub(centre.y), 0.05, 0.5);
    const opacity: any = density
      .mul(fadeIn)
      .mul(fadeOut)
      .mul(submerged)
      .mul(strength.sqrt())
      .mul(this.uStrength.min(1))
      .toVar('sprayOpacity');

    return particleVertex({
      centre,
      velocity,
      life,
      alive,
      isDrop,
      isSheet,
      isMist,
      isCluster,
      cls,
      seed: seed.z,
      width,
      // Seconds of motion each streak spans: long for droplets, which is what
      // reads as speed, and none for mist, which has none to show.
      streak: isDrop.mul(0.035).add(isCluster.mul(0.03)).add(isSheet.mul(0.02)).add(isMist.mul(0.04)),
      opacity,
      // Depth over which each class fades into the water or hull behind it.
      soft: isDrop
        .mul(0.08)
        .add(isCluster.mul(0.3))
        .add(isSheet.mul(0.6))
        .add(isMist.mul(1.5))
        .mul(isFloat.mul(-0.6).add(1)),
      range: DRAW_RANGE,
    });
  }
}

function freshState(): ProbeState {
  return { previousHeight: 0, primed: false, last: -1000, deep: false };
}

/** Scratch, module scope so `update` allocates nothing per frame. */
const _probeWorld = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _starboard = new THREE.Vector3();
const _out = new THREE.Vector3();
const _at = new THREE.Vector3();
const _local = new THREE.Vector3();
const _twisted = new THREE.Vector3();
const _zero = new THREE.Vector3();

/** One period of the per-event class interleave. */
const CLASS_ORDER = [
  CLASS_SHEET,
  CLASS_DROP,
  CLASS_CLUSTER,
  CLASS_MIST,
  CLASS_SHEET,
  CLASS_DROP,
  CLASS_CLUSTER,
  CLASS_DROP,
];

function buildGeometry(): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  // A unit quad centred on the origin; the vertex stage supplies every transform.
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
  );
  geometry.setIndex([0, 1, 2, 0, 2, 3]);

  // Event-fastest: instance i belongs to event i % EVENTS and is that event's
  // particle floor(i / EVENTS). Trimming `instanceCount` therefore removes the
  // same number of particles from every event — which is how a lower tier
  // draws thinner spray rather than fewer impacts.
  const count = EVENTS * PER_EVENT;
  const event = new Float32Array(count);
  const seed = new Float32Array(count * 4);
  const cls = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const e = i % EVENTS;
    const sub = Math.floor(i / EVENTS);
    event[i] = e;
    seed[i * 4] = hash2(i, 1);
    seed[i * 4 + 1] = hash2(i, 2);
    seed[i * 4 + 2] = hash2(i, 3);
    seed[i * 4 + 3] = hash2(sub, 4);
    // A quarter sheets, a quarter droplet clusters, an eighth mist and the rest
    // single droplets — interleaved so every prefix of the instance order keeps
    // the same mix, which is what lets a tier trim the count evenly.
    cls[i] = CLASS_ORDER[sub % CLASS_ORDER.length];
  }
  geometry.setAttribute('sprayEvent', new THREE.InstancedBufferAttribute(event, 1));
  geometry.setAttribute('spraySeed', new THREE.InstancedBufferAttribute(seed, 4));
  geometry.setAttribute('sprayClass', new THREE.InstancedBufferAttribute(cls, 1));
  geometry.instanceCount = count;
  // Placed entirely in the vertex stage, so a bound computed from the geometry
  // would describe a unit quad at the origin and cull the whole system.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);
  return geometry;
}
