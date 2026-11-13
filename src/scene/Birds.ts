import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  faceDirection,
  float,
  mix,
  positionGeometry,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { ISLAND } from './Seafloor';

/**
 * Gulls, animated entirely on the GPU: flocks working the ship, a wheel of them
 * over the island, a few soaring high, and a handful perched on the ship's yards
 * and taffrail that lift off when she gets under way.
 *
 * **No downloaded asset.** A rigged glTF gull was not obtainable under a licence
 * this project can verify, and at the distances these are seen instanced
 * procedural geometry is the right technique anyway. Twenty-odd triangles and a
 * vertex shader get the things that actually read: the silhouette — the bent
 * "M" of a gull's wing, arm raised and hand drooped — and the way it moves.
 *
 * The whole system is **one instanced draw**. Geometry is a single gull built in
 * code — a body with volume, a tail, and an arm and a hand panel per wing — and
 * every bird is that geometry deformed and placed by the vertex stage. The CPU
 * writes no transforms: per frame it writes four uniforms, and a float per
 * perched bird when one of them is moving between the ship and the sky.
 *
 * `InstancedBufferGeometry` on a plain `Mesh` rather than `InstancedMesh`, for
 * the same reason `Weather` and `Particles` do it: there is no per-instance
 * matrix to carry, so an `instanceMatrix` would be a buffer of identity matrices
 * multiplied into every vertex for nothing.
 *
 * **Flight is a pure function of one clock.** Each flying bird runs a closed
 * circuit whose position, velocity and acceleration have closed forms in the
 * path parameter, which is what makes `resetClock(t)` land on the exact pose for
 * time `t` — the property the visual regression harness depends on. The perched
 * birds are the one piece of state, and it is a small integrator driven only by
 * the ship's motion and `dt`, both of which the deterministic harness already
 * controls; `resetClock` puts every one of them back on its perch.
 *
 * No compute, no storage textures, nothing WebGPU-only: the same node graph has
 * to compile on the WebGL2 fallback.
 */

/**
 * TSL node objects are structurally dynamic; composing per-component expressions
 * out of `attribute()` values produces `any`, which the overloaded typings then
 * resolve to the wrong constructor. Node-typed locals are therefore `any` by
 * design — the class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

/** `vec3` under a loose signature. See the note above. */
const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => any;
/** `mix` under a loose signature, for the same reason. */
const mixn = mix as unknown as (a: unknown, b: unknown, t: unknown) => any;

/**
 * Seeded, and seeded *here*. The flock must be identical on every load or a sky
 * baseline is comparing two different flocks and every diff is noise.
 */
const SEED = 0x9e11a5;

/**
 * Clock wrap, seconds. **Every frequency in this file is quantised to a whole
 * number of cycles per wrap**, so the flock is exactly periodic with this period
 * and the wrap is invisible — a gull that jumps when the clock reseats is a gull
 * teleporting across the sky, once an hour, with no way to reproduce it.
 */
const CLOCK_WRAP = 3600;

/**
 * Instances the buffers are built for. `setCount` only moves `instanceCount`,
 * so bird `i` keeps the same role and circuit whatever the count is — a tier
 * change re-scales the flock instead of reshuffling it.
 */
const MAX_BIRDS = 128;

/**
 * What each bird is doing. Assigned by index (see `roleOf`) so that every prefix
 * of the flock — every tier's count — has a mix of all of them.
 *
 * - `ship`: working the ship, tight circuits at mast height. These are the birds
 *   that make the ship feel like a ship: gulls follow vessels for what they turn
 *   up, and a hull with nothing over it reads as a model on a table.
 * - `island`: a wheel of birds over the island, high and wide, anchored to the
 *   island itself so they are there when the ship arrives.
 * - `soar`: high, slow, mostly gliding circles that follow the viewer, so open
 *   water always has something in its sky.
 * - `perch`: sitting on the ship; flies with the `ship` group once it lifts off.
 */
type Role = 'ship' | 'island' | 'soar' | 'perch';

/**
 * Where the perched birds sit, in the ship's own frame (+X bow, +Y up, +Z
 * starboard), and which way each faces relative to the bow, radians.
 *
 * Found by casting rays down onto the rigging and hull meshes of
 * `dutch_ship_medium` in the ship's frame and keeping the hits on spars: the
 * main and fore course yards at 10.6 m and 9.4 m, the fore topsail yard at
 * 19.2 m, and the stern rail at 5.8 m. The yards are braced, so their arms do
 * not share an x. Each point is the top of the spar, so the bird's belly sits
 * on it.
 */
const PERCHES: ReadonlyArray<readonly [number, number, number, number]> = [
  [6.4, 9.37, -3.1, 0.3], // fore yard, port arm
  [8.1, 9.38, 3.2, -0.4], // fore yard, starboard arm
  [-12.8, 5.8, 0.0, Math.PI * 0.92], // taffrail, looking aft
  [0.1, 10.57, 3.5, -0.2], // main yard, starboard arm
  [8.1, 19.15, 2.6, 0.5], // fore topsail yard, starboard arm
];

/** Which instance indices perch, in the order of `PERCHES`. */
const PERCH_INDICES = [0, 3, 7, 11, 16];

/** Ship speed that sends perched birds up, m/s over ground. */
const TAKEOFF_SPEED = 1.1;
/** Ship speed below which perched birds start thinking about coming back. */
const SETTLE_SPEED = 0.35;
/** Seconds from leaving the spar to joining the circuit. */
const TAKEOFF_TIME = 1.8;
/** Seconds from leaving the circuit to landing. Longer: landing is careful. */
const LAND_TIME = 4.5;
/** Height of the hop arc between the perch and the circuit, metres. */
const HOP_HEIGHT = 3.5;
/** Height of a perched bird's centre above its perch point, at scale 1. */
const PERCH_LIFT = 0.07;

/**
 * Lag of the anchor the `ship` flock circles, seconds.
 *
 * Gulls do not orbit a moving ship's centre of mass; they hang off her quarter
 * and fall behind on the turns. A lagged anchor gets that for one exponential.
 */
const SHIP_ANCHOR_LAG = 2.5;

/**
 * How far the viewer may roam before the `soar` flock starts to follow, metres.
 *
 * Inside this radius those circuits are anchored to the world origin and the
 * birds parallax as the ship moves, which is most of what sells their distance.
 * Beyond it the anchor slides with the viewer at exactly their speed. The
 * crossover is continuous in position, so nothing teleports.
 */
const HOME_RADIUS = 140;

/** Standard gravity, for the coordinated-turn bank angle. */
const GRAVITY = 9.81;

/**
 * Bank exaggeration. A coordinated turn banks at `atan(a_lat / g)`, which at
 * these circuits is a roll indistinguishable from level flight; birds genuinely
 * over-bank because they use roll to *change* heading. Three reads.
 */
const BANK_GAIN = 3;

/** Ceiling on roll, radians. Beyond ~55 degrees a gull is edge-on and a stick. */
const MAX_ROLL = 0.95;

/** Peak wing rotation about the shoulder, radians. */
const FLAP_AMPLITUDE = 0.7;

/**
 * The gull's wing in a glide: the arm raised and the hand drooped below it.
 *
 * This is the silhouette. A flat cross with a dihedral reads as "a bird"; the
 * kinked M — shoulder up, wrist the high point, primaries angled back down —
 * reads as *a gull*, at any distance where the bird is more than a few pixels.
 * While flapping the kink relaxes toward a straighter wing.
 */
const ARM_DIHEDRAL_GLIDE = 0.26;
const ARM_DIHEDRAL_FLAP = 0.1;
const HAND_DROOP_GLIDE = -0.48;
const HAND_DROOP_FLAP = -0.08;

/**
 * Phase lag from arm to hand, radians. The hand trails the arm by roughly a
 * quarter of a stroke, and that lag is what puts a travelling wave along the
 * span instead of a pair of hinged boards.
 */
const TIP_LAG = 0.9;

/**
 * Wrist flex during the recovery stroke: the hand is drawn inboard and swept
 * back. On the **upstroke** — the downstroke is the power stroke and the wing is
 * held extended through it; folding on it reads as a broken wing.
 */
const FOLD_SHRINK = 0.3;
const FOLD_SWEEP = 0.08;

/** Softness of the flap/glide gate. Wide enough to take about a second. */
const GLIDE_WIDTH = 0.3;

/**
 * Collapse radius, metres. Close enough that a perched bird on a yard is still
 * drawn from the deck, far enough that the camera never ends up inside one.
 * Scaling to zero in the vertex stage rasterises no fragments at all.
 */
const NEAR_HIDE = 1.2;
const NEAR_SHOW = 3.5;

/**
 * Strength of the light coming *through* a backlit wing. A gull's remiges are
 * thin enough to glow against a low sun rather than going black.
 */
const TRANSMISSION = 0.55;

/** Wing half-span of the authored geometry, metres, at instance scale 1. */
const HALF_SPAN = 0.66;
/** Lateral offset of the shoulder joint — the pivot the arm rotates about. */
const SHOULDER_X = 0.045;
/** Lateral offset of the wrist — the pivot the hand rotates about. */
const WRIST_X = 0.3;

export class Birds {
  /** Instances the buffers hold. `setCount` is clamped to this. */
  static readonly MAX_COUNT = MAX_BIRDS;

  readonly object: THREE.Object3D;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;
  /** 1 = sitting on its perch, 0 = flying its circuit. One float per instance. */
  private readonly perchState: THREE.InstancedBufferAttribute;

  private count: number;
  private clock = 0;
  private disposed = false;
  /** What the caller asked for, kept apart from whether there is anything to draw. */
  private wantVisible = true;

  // --- the ship, for the `ship` flock and the perched birds -----------------
  private ship: THREE.Object3D | null = null;
  private readonly lastShipPosition = new THREE.Vector3();
  private shipTracked = false;
  /** Smoothed ground speed of the ship, m/s. */
  private shipSpeed = 0;
  /** Per perched bird: where it is going (1 perch, 0 sky) and its timers. */
  private readonly perchTarget = new Float32Array(PERCHES.length);
  private readonly perchTimer = new Float32Array(PERCHES.length);
  private readonly perchDelay = new Float32Array(PERCHES.length);
  private readonly perchReturn = new Float32Array(PERCHES.length);

  // --- uniforms --------------------------------------------------------------
  private readonly uTime = uniform(0);
  /** Centre of the `soar` circuits, world space. See `HOME_RADIUS`. */
  private readonly uAnchor = uniform(new THREE.Vector3());
  /** Lagged ship position the `ship` circuits are centred on. */
  private readonly uShipAnchor = uniform(new THREE.Vector3());
  /** The ship's world matrix, for the perches. */
  private readonly uShipMatrix = uniform(new THREE.Matrix4());
  /** The viewer, for the near collapse only. */
  private readonly uViewer = uniform(new THREE.Vector3());
  /** Unit vector *toward* the sun. */
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(1.0, 0.95, 0.88));
  /** What an unlit bird settles to: sky-lit, not black. */
  private readonly uShadeColor = uniform(new THREE.Color(0.24, 0.3, 0.38));
  /** Direct-sun weight, faded out as the sun sets. See `setSunDirection`. */
  private readonly uSunGain = uniform(1);

  constructor(count = 24) {
    this.count = clampCount(count);

    this.geometry = buildFlockGeometry();
    this.geometry.instanceCount = this.count;
    this.perchState = this.geometry.getAttribute('perchState') as THREE.InstancedBufferAttribute;

    const random = mulberry32(SEED ^ 0x5eed);
    for (let k = 0; k < PERCHES.length; k++) {
      // Not all at once: a row of gulls lifting off in the same frame is a
      // sprite sheet. Staggered over about a second and a half, the first one
      // up spooks the rest.
      this.perchDelay[k] = 0.15 + random() * 1.4;
      // And back at different times, so the yards fill up one bird at a time.
      this.perchReturn[k] = 16 + random() * 22;
    }

    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'gulls';
    // Placed in world space by the vertex stage, so a bounding sphere is
    // meaningless.
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;

    this.object = new THREE.Object3D();
    this.object.name = 'birds';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
    this.resetPerches();
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Birds actually drawn, 0..`Birds.MAX_COUNT`. Moves an instance count; at 0
   * the renderer skips the draw entirely.
   */
  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.count) return;
    this.count = next;
    this.geometry.instanceCount = next;
    this.applyVisibility();
  }

  setVisible(v: boolean): void {
    this.wantVisible = v;
    this.applyVisibility();
  }

  /**
   * The ship the `ship` flock works and the perched birds sit on. `null` (the
   * default, and what a failed model load leaves) sends the perched birds into
   * the air permanently — there is nothing to sit on — and centres the `ship`
   * circuits on the origin, where the ship would have been.
   */
  setShip(ship: THREE.Object3D | null): void {
    this.ship = ship;
    this.shipTracked = false;
    this.resetPerches();
  }

  /**
   * The sun, for shading. The direct term is faded out as the sun drops below
   * the horizon, so the flock cannot end up lit like midday against a night sky.
   */
  setSunDirection(dir: THREE.Vector3): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(dir);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();

    const above = clamp01((sun.y + 0.09) / 0.18);
    this.uSunGain.value = above * above * (3 - 2 * above);
  }

  /**
   * Sunlight and sky-light colours.
   *
   * `sun` is a hue — `Atmosphere.sunColor`, normalised to a peak of 1. A little
   * white goes back in: at the horizon that colour is a saturated red, and a
   * white bird lit by it alone read as a red dash rather than as a gull in
   * evening light.
   *
   * `shade` is the light an *unlit* bird receives — a third to a half of the
   * sky's radiance, which is the range that reads as a silhouette.
   */
  setLightColors(sun: THREE.Color, shade: THREE.Color): void {
    (this.uSunColor.value as THREE.Color).copy(sun).lerp(WHITE, 0.2);
    (this.uShadeColor.value as THREE.Color).copy(shade);
  }

  /**
   * Advances the clock, re-anchors the flocks and moves the perched birds.
   * No allocation.
   */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    this.anchorTo(cameraPosition);
    this.trackShip(dt);
  }

  /**
   * Jumps the flock to the pose for `time`, and every perched bird back onto
   * its perch.
   *
   * The flying birds are exact — every position, heading, bank and wing angle is
   * a closed-form function of this clock. The perches are reset rather than
   * replayed: what they should be doing depends on the ship's history, which the
   * caller is about to rewind as well.
   */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    this.shipTracked = false;
    this.resetPerches();
  }

  /**
   * Where bird `index` is now, world space, computed on the CPU from the same
   * closed form the vertex stage evaluates (the body only — no wing pose).
   *
   * For the audio, which places a gull's cry on a bird that is actually up
   * there. Allocation-free. False for an index the flock is not drawing.
   */
  positionOf(index: number, out: THREE.Vector3): boolean {
    if (!(index >= 0 && index < this.count)) return false;
    const orbit = this.geometry.getAttribute('orbit').array;
    const harmonics = this.geometry.getAttribute('harmonics').array;
    const bob = this.geometry.getAttribute('bob').array;
    const b = index * 4;
    const theta = orbit[b + 1] * this.clock + orbit[b + 2];
    const second = theta * 2 + harmonics[b + 1];
    const third = theta * 3 + harmonics[b + 3];
    const climb = theta * 2 + bob[b + 1];
    const radius = orbit[b];
    const anchor = this.uAnchor.value as THREE.Vector3;
    out.set(
      anchor.x + radius * Math.cos(theta) + harmonics[b] * Math.cos(second) + harmonics[b + 2] * Math.cos(third),
      anchor.y + orbit[b + 3] + bob[b] * Math.sin(climb),
      anchor.z + radius * Math.sin(theta) + harmonics[b] * Math.sin(second) + harmonics[b + 2] * Math.sin(third),
    );
    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.remove(this.mesh);
    this.object.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }

  // ------------------------------------------------------------------ internals

  /** An empty flock is hidden outright, so it costs not even a culling test. */
  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.count > 0;
  }

  private resetPerches(): void {
    const state = this.perchState.array as Float32Array;
    const seated = this.ship !== null ? 1 : 0;
    for (let k = 0; k < PERCHES.length; k++) {
      this.perchTarget[k] = seated;
      this.perchTimer[k] = 0;
      state[PERCH_INDICES[k]] = seated;
    }
    this.shipSpeed = 0;
    this.perchState.needsUpdate = true;
  }

  /**
   * Follows the ship: the lagged anchor, its world matrix, its speed, and the
   * perched birds' decisions.
   *
   * Speed is measured from the hull's own displacement rather than read from the
   * controller, so it is the speed the birds can *see* — a ship drifting on a
   * swell with the engine off still unsettles them. A jump of more than
   * 30 m/s is a teleport (a deterministic reset returning the hull to its
   * mooring), not motion, and re-seats the tracking instead of launching every
   * bird on the yards.
   */
  private trackShip(dt: number): void {
    const ship = this.ship;
    const anchor = this.uShipAnchor.value as THREE.Vector3;
    if (ship === null) {
      anchor.set(0, 0, 0);
      return;
    }
    (this.uShipMatrix.value as THREE.Matrix4).copy(ship.matrixWorld);
    const x = ship.matrixWorld.elements[12];
    const z = ship.matrixWorld.elements[14];

    if (!this.shipTracked || dt <= 0) {
      this.lastShipPosition.set(x, 0, z);
      anchor.set(x, 0, z);
      this.shipTracked = true;
      return;
    }

    const dx = x - this.lastShipPosition.x;
    const dz = z - this.lastShipPosition.z;
    const step = Math.sqrt(dx * dx + dz * dz);
    this.lastShipPosition.set(x, 0, z);
    if (step > 30 * dt) {
      anchor.set(x, 0, z);
      this.shipSpeed = 0;
      return;
    }

    const follow = 1 - Math.exp(-dt / SHIP_ANCHOR_LAG);
    anchor.x += (x - anchor.x) * follow;
    anchor.z += (z - anchor.z) * follow;
    this.shipSpeed += (step / dt - this.shipSpeed) * (1 - Math.exp(-dt / 0.6));

    const state = this.perchState.array as Float32Array;
    let changed = false;
    for (let k = 0; k < PERCHES.length; k++) {
      const i = PERCH_INDICES[k];
      if (this.perchTarget[k] === 1) {
        // Seated, or coming in. Startled by way on the hull, after this bird's
        // own hesitation.
        if (this.shipSpeed > TAKEOFF_SPEED) {
          this.perchTimer[k] += dt;
          if (this.perchTimer[k] > this.perchDelay[k]) {
            this.perchTarget[k] = 0;
            this.perchTimer[k] = 0;
          }
        } else {
          this.perchTimer[k] = 0;
        }
      } else if (this.shipSpeed < SETTLE_SPEED) {
        // Flying. Comes back once the ship has been still for long enough.
        this.perchTimer[k] += dt;
        if (this.perchTimer[k] > this.perchReturn[k]) {
          this.perchTarget[k] = 1;
          this.perchTimer[k] = 0;
        }
      } else {
        this.perchTimer[k] = 0;
      }

      const current = state[i];
      const target = this.perchTarget[k];
      if (current !== target) {
        const rate = target > current ? dt / LAND_TIME : -dt / TAKEOFF_TIME;
        state[i] = clamp01(current + rate);
        changed = true;
      }
    }
    if (changed) this.perchState.needsUpdate = true;
  }

  /**
   * Slides the `soar` circuit centre so that flock stays reachable. See
   * `HOME_RADIUS`. Horizontal only — the birds fly at an altitude above the sea,
   * not above the camera.
   */
  private anchorTo(cameraPosition: THREE.Vector3): void {
    const anchor = this.uAnchor.value as THREE.Vector3;
    const x = cameraPosition.x;
    const z = cameraPosition.z;
    const distance = Math.sqrt(x * x + z * z);

    if (distance <= HOME_RADIUS) {
      anchor.set(0, 0, 0);
    } else {
      const k = (distance - HOME_RADIUS) / distance;
      anchor.set(x * k, 0, z * k);
    }

    (this.uViewer.value as THREE.Vector3).copy(cameraPosition);
  }

  private buildMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    // Wings and tail are sheets with no thickness, so both faces rasterise.
    material.side = THREE.DoubleSide;
    material.transparent = false;
    material.depthWrite = true;
    // Aerial perspective comes from the shared haze and the volumetric pass on
    // scene depth; vertex fog would double it.
    material.fog = false;

    // Separate graphs rather than shared `.toVar()` locals: a varying opens its
    // own vertex-stage flow, and sharing vars across the two is how you get a
    // shader that references a name it never declared. Rebuilding the frame
    // costs a few dozen vector ops on a few thousand vertices.
    material.positionNode = Fn(() => this.buildVertex().world)() as any;

    const shading: any = varying(
      Fn(() => {
        const vertex = this.buildVertex();
        return vec4(vertex.normal, vertex.outboard);
      })(),
    );
    const bodyShading: any = varying(
      Fn(() => {
        const vertex = this.buildVertex();
        return vec4(vertex.part, vertex.upness, vertex.head, 0);
      })(),
    );

    const normal: any = shading.xyz.normalize();
    /** Distance out the span, 0 at the shoulder and 1 at the tip. */
    const outboard: any = shading.w;
    /** 1 on the body's volume, 0 on the wing and tail sheets. */
    const part: any = bodyShading.x;
    /** The body's local surface normal's up component: back +1, belly -1. */
    const upness: any = bodyShading.y;
    /** 1 toward the head, 0 from the shoulders back. */
    const head: any = bodyShading.z;

    // `faceDirection` is +1 on a sheet's back and -1 on its belly, because the
    // sheets are wound counter-clockwise seen from above; on the closed body it
    // is +1 wherever a face is visible at all.
    const facing: any = faceDirection;
    const seen: any = normal.mul(facing);
    const ndl: any = seen.dot(this.uSunDir);

    // Wrapped diffuse: a hard terminator on a body this small in screen space is
    // not shading, it is a flicker.
    const direct: any = ndl.mul(0.5).add(0.5).mul(this.uSunGain);
    const through: any = ndl.negate().max(0).mul(TRANSMISSION).mul(this.uSunGain);
    const lit: any = direct.add(through).clamp(0, 1);

    // Grey mantle above, white below — the countershading that makes a gull read
    // as a gull the instant it banks and shows the other side. On the body the
    // mantle is the back, and the head stays white.
    const sheetTop: any = facing.mul(0.5).add(0.5);
    const bodyTop: any = upness.smoothstep(0.15, 0.75).mul(head.oneMinus());
    const topness: any = mix(sheetTop, bodyTop, part);
    const plumage: any = mix(vec3(0.95, 0.96, 0.97), vec3(0.6, 0.66, 0.72), topness);
    // Black primaries, on the hand only.
    const tipped: any = mix(plumage, vec3(0.1, 0.11, 0.13), outboard.smoothstep(0.62, 0.95));

    const light: any = mix(this.uShadeColor, this.uSunColor, lit);
    material.colorNode = vec4(tipped.mul(light), 1) as any;

    return material;
  }

  /**
   * The whole animation, per vertex.
   *
   * Read as: solve the circuit for where this bird is and which way it is
   * flying; blend toward its perch if it has one; build a frame from that;
   * deform the wing about the shoulder and the wrist in the bird's own axes;
   * then place the deformed vertex in the frame. Nothing here reads a texture or
   * a previous frame.
   */
  private buildVertex(): {
    world: any;
    normal: any;
    outboard: any;
    part: any;
    upness: any;
    head: any;
  } {
    const orbit: any = attribute('orbit', 'vec4');
    const harmonics: any = attribute('harmonics', 'vec4');
    const bob: any = attribute('bob', 'vec4');
    const style: any = attribute('style', 'vec4');
    const role: any = attribute('role', 'vec4');
    const perch: any = attribute('perch', 'vec4');
    const perched: any = attribute('perchState', 'float');
    const shape: any = attribute('shape', 'vec3');
    const span: any = shape.x;
    const hand: any = shape.y;
    const part: any = shape.z;
    const restNormal: any = attribute('normal', 'vec3');
    const rest: any = positionGeometry;

    const radius = orbit.x;
    const rate = orbit.y;
    const altitude = orbit.w;
    const secondAmp = harmonics.x;
    const thirdAmp = harmonics.z;
    const climbAmp = bob.x;

    // --- the circuit ---------------------------------------------------------
    //
    // A circle plus a second and third harmonic in the same parameter, so the
    // path is closed by construction and there is nowhere for a bird to drift
    // to. `buildInstanceData` holds `2*B + 3*C < A`, so the tangent never
    // vanishes and the bird never stops dead and swings round in one frame.
    const theta = rate.mul(this.uTime).add(orbit.z).toVar();
    const second = theta.mul(2).add(harmonics.y).toVar();
    const third = theta.mul(3).add(harmonics.w).toVar();
    const climb = theta.mul(2).add(bob.y).toVar();

    const s1 = theta.sin().toVar();
    const c1 = theta.cos().toVar();
    const s2 = second.sin().toVar();
    const c2 = second.cos().toVar();
    const s3 = third.sin().toVar();
    const c3 = third.cos().toVar();
    const sc = climb.sin().toVar();
    const cc = climb.cos().toVar();

    const offset = vec3n(
      radius.mul(c1).add(secondAmp.mul(c2)).add(thirdAmp.mul(c3)),
      altitude.add(climbAmp.mul(sc)),
      radius.mul(s1).add(secondAmp.mul(s2)).add(thirdAmp.mul(s3)),
    ).toVar();

    // dP/dtheta and d²P/dtheta², term by term: the heading and the bank.
    const tangent = vec3n(
      radius.mul(s1).negate().sub(secondAmp.mul(s2).mul(2)).sub(thirdAmp.mul(s3).mul(3)),
      climbAmp.mul(cc).mul(2),
      radius.mul(c1).add(secondAmp.mul(c2).mul(2)).add(thirdAmp.mul(c3).mul(3)),
    ).toVar();

    const bend = vec3n(
      radius.mul(c1).negate().sub(secondAmp.mul(c2).mul(4)).sub(thirdAmp.mul(c3).mul(9)),
      climbAmp.mul(sc).mul(-4),
      radius.mul(s1).negate().sub(secondAmp.mul(s2).mul(4)).sub(thirdAmp.mul(s3).mul(9)),
    ).toVar();

    // Chain rule into real units. `rate` carries the sign.
    const velocity = tangent.mul(rate).toVar();
    const accel = bend.mul(rate).mul(rate).toVar();

    // Which anchor this bird's circuit is centred on: one-hot per role.
    const island = vec3(ISLAND.x, 0, ISLAND.z);
    const anchor = this.uShipAnchor
      .mul(role.x)
      .add(island.mul(role.y))
      .add(this.uAnchor.mul(role.z))
      .toVar();
    const flightCentre = anchor.add(offset).toVar();
    const flightForward = velocity.normalize().toVar();

    // --- the perch -----------------------------------------------------------
    //
    // `perched` is 1 sitting, 0 flying, and moves between them over a couple of
    // seconds; every bird without a perch has it at exactly 0, so everything
    // below collapses to the flight pose for them.
    const shipMatrix: any = this.uShipMatrix;
    const perchWorld = shipMatrix.mul(vec4(perch.xyz, 1)).xyz.toVar();
    const shipForward = shipMatrix.mul(vec4(1, 0, 0, 0)).xyz.normalize().toVar();
    const shipSide = shipMatrix.mul(vec4(0, 0, 1, 0)).xyz.normalize().toVar();
    // A sitting gull is not a statue: it turns its head and shuffles round on
    // the spar every few seconds.
    const idle = this.uTime.mul(0.41).add(orbit.z.mul(7)).sin().mul(0.35).add(perch.w).toVar();
    const perchForward = shipForward.mul(idle.cos()).add(shipSide.mul(idle.sin())).toVar();

    const settle = perched.mul(perched).mul(perched.mul(-2).add(3)).toVar();
    const hop = perched.mul(perched.oneMinus()).mul(4).toVar();
    const seat = perchWorld.add(vec3(0, PERCH_LIFT, 0).mul(style.x));
    const centreFlight = mixn(flightCentre, seat, settle).add(vec3(0, HOP_HEIGHT, 0).mul(hop)).toVar();

    // --- the flight frame ----------------------------------------------------
    const forward = mixn(flightForward, perchForward, settle).normalize().toVar();
    const across = forward.cross(vec3(0, 1, 0)).toVar();
    // Guarded: the cross product collapses for a vertically climbing bird.
    const right = across.div(across.length().max(1e-3)).toVar();
    const up = right.cross(forward).toVar();

    // Coordinated turn, exaggerated, and levelled as the bird comes onto a spar.
    const roll = accel
      .dot(right)
      .mul(BANK_GAIN / GRAVITY)
      .atan()
      .clamp(-MAX_ROLL, MAX_ROLL)
      .mul(settle.oneMinus())
      .toVar();
    const rollCos = roll.cos().toVar();
    const rollSin = roll.sin().toVar();
    const bankRight = right.mul(rollCos).sub(up.mul(rollSin)).toVar();
    const bankUp = up.mul(rollCos).add(right.mul(rollSin)).toVar();

    // --- the wing ------------------------------------------------------------
    const outboard = span.abs().toVar();
    // Exactly 0 on the body and tail, so the fuselage is untouched by every
    // rotation below without a branch.
    const side = span.sign().toVar();

    // The glide gate: a slow oscillator per bird, thresholded. The *amplitude*
    // is gated, not the phase, so a bird settles into the glide pose instead of
    // freezing mid-stroke. Taking off and landing are hard work, whatever the
    // gate says — and a bird on its perch does not flap at all.
    const glide = style.y.mul(this.uTime).add(style.z).sin().toVar();
    const gate = glide.smoothstep(style.w.sub(GLIDE_WIDTH), style.w.add(GLIDE_WIDTH));
    const amplitude = gate
      .max(hop.mul(1.25))
      .mul(perched.smoothstep(0.82, 1.0).oneMinus())
      .toVar();

    const stroke = bob.z.mul(this.uTime).add(bob.w).toVar();
    const armRest = mixn(float(ARM_DIHEDRAL_GLIDE), float(ARM_DIHEDRAL_FLAP), amplitude);
    const handRest = mixn(float(HAND_DROOP_GLIDE), float(HAND_DROOP_FLAP), amplitude);
    const shoulder = stroke.sin().mul(amplitude).mul(FLAP_AMPLITUDE).add(armRest).mul(side).toVar();
    // The hand's angle *relative to the arm*: its own lagged stroke minus the
    // arm's, so the tip trails the shoulder through each beat.
    const wrist = stroke
      .sub(TIP_LAG)
      .sin()
      .sub(stroke.sin())
      .mul(amplitude)
      .mul(FLAP_AMPLITUDE * 0.9)
      .add(handRest)
      .mul(side)
      .mul(hand)
      .toVar();

    // The stroke's angular velocity is its cosine: positive through the
    // recovery half, which is when the hand flexes in. See `FOLD_SHRINK`.
    const fold = stroke.cos().max(0).mul(amplitude).mul(hand).toVar();

    // Hand: about the wrist.
    const wristX = side.mul(WRIST_X).toVar();
    const handX = rest.x.sub(wristX).mul(fold.mul(FOLD_SHRINK).oneMinus()).toVar();
    const restZ = rest.z.sub(fold.mul(FOLD_SWEEP)).toVar();
    const wristCos = wrist.cos().toVar();
    const wristSin = wrist.sin().toVar();
    const hx = handX.mul(wristCos).sub(rest.y.mul(wristSin)).add(wristX).toVar();
    const hy = handX.mul(wristSin).add(rest.y.mul(wristCos)).toVar();

    // Arm: about the shoulder, not the centreline, or the body would pump with
    // the wings.
    const pivot = side.mul(SHOULDER_X).toVar();
    const arm = hx.sub(pivot).toVar();
    const shoulderCos = shoulder.cos().toVar();
    const shoulderSin = shoulder.sin().toVar();
    const flying = vec3n(
      arm.mul(shoulderCos).sub(hy.mul(shoulderSin)).add(pivot),
      arm.mul(shoulderSin).add(hy.mul(shoulderCos)),
      restZ,
    ).toVar();

    // Folded along the back for a bird on its perch: the span collapses onto
    // the flank, and the primaries cross over the tail.
    const tuck = perched.smoothstep(0.55, 1.0).mul(side.abs()).toVar();
    const fromShoulder = rest.x.sub(pivot).toVar();
    const folded = vec3n(
      pivot.add(fromShoulder.mul(0.06)),
      fromShoulder.abs().mul(0.05).add(0.03),
      rest.z.sub(fromShoulder.abs().mul(0.42)),
    );
    const shaped = mixn(flying, folded, tuck).toVar();

    // --- placement -----------------------------------------------------------
    // See `NEAR_HIDE`. Scaling to zero costs no fragments; an alpha fade would.
    const size = style.x
      .mul(centreFlight.sub(this.uViewer).length().smoothstep(NEAR_HIDE, NEAR_SHOW))
      .toVar();
    const local = shaped.mul(size);

    const world = centreFlight
      .add(bankRight.mul(local.x))
      .add(bankUp.mul(local.y))
      .add(forward.mul(local.z));

    // Sheets: (0,1,0) rotated about the bird's long axis by the arm angle plus,
    // on the hand, the wrist angle. Body: the authored face normal. A folded
    // wing lies flat on the back.
    const sheetAngle = shoulder.add(wrist).toVar();
    const sheetNormal = vec3n(sheetAngle.sin().negate(), sheetAngle.cos(), 0);
    const localNormal = mixn(mixn(sheetNormal, vec3(0, 1, 0), tuck), restNormal, part).toVar();
    const normal = bankRight
      .mul(localNormal.x)
      .add(bankUp.mul(localNormal.y))
      .add(forward.mul(localNormal.z));

    // Head: the forward third of the body, which stays white.
    const head = rest.z.smoothstep(0.1, 0.18).mul(part);

    return { world, normal, outboard, part, upness: localNormal.y, head };
  }
}

const WHITE = /*@__PURE__*/ new THREE.Color(1, 1, 1);

/**
 * The gull, authored nose-forward: **+X right wing, +Y up, +Z nose**, the frame
 * the vertex stage builds, so placement is three multiply-adds.
 *
 * Non-indexed: every triangle carries its own face normal, because the body is
 * faceted on purpose — at a few dozen pixels a smooth-shaded lozenge reads as a
 * blob, a faceted one reads as a body with a back and a belly.
 *
 * Attributes:
 * - `span`: signed distance out the wing, normalised to the tip. Its sign
 *   mirrors the flap; its magnitude drives the dark primaries. 0 on the body
 *   and tail, which excludes them from every wing rotation.
 * - `hand`: 1 on the vertices outboard of the wrist, which take the wrist
 *   rotation as well as the shoulder's.
 * - `part`: 1 on the body's volume, 0 on the wing and tail sheets.
 *
 * Sheets are wound counter-clockwise seen from above, which is what makes
 * `faceDirection` mean "this is the bird's back"; the body is wound outward.
 * Both are enforced here rather than trusted to the vertex order below.
 */
function buildFlockGeometry(): THREE.InstancedBufferGeometry {
  type V = readonly [number, number, number];
  // Body.
  const nose: V = [0, 0.012, 0.3];
  const crown: V = [0, 0.06, 0.19];
  const back: V = [0, 0.072, 0.02];
  const rump: V = [0, 0.024, -0.17];
  const throat: V = [0, -0.045, 0.17];
  const belly: V = [0, -0.066, 0.0];
  const vent: V = [0, -0.012, -0.17];
  const flankR: V = [0.058, 0.0, 0.05];
  const flankL: V = [-0.058, 0.0, 0.05];
  const bodyCentre: V = [0, 0, 0.04];

  const bodyTris: V[][] = [];
  for (const flank of [flankR, flankL]) {
    bodyTris.push(
      [nose, crown, flank],
      [crown, back, flank],
      [back, rump, flank],
      [nose, throat, flank],
      [throat, belly, flank],
      [belly, vent, flank],
      [rump, vent, flank],
    );
  }

  // Tail: a squared-off fan, a sheet.
  const tailTris: V[][] = [
    [[0.035, 0.02, -0.14], [0.08, 0.02, -0.33], [-0.08, 0.02, -0.33]],
    [[0.035, 0.02, -0.14], [-0.08, 0.02, -0.33], [-0.035, 0.02, -0.14]],
  ];

  // Right wing: arm panel shoulder->wrist, then a long pointed hand swept back.
  // The wrist is the leading edge's most forward point, as on a real gull.
  const shoulderLead: V = [SHOULDER_X, 0, 0.1];
  const shoulderTrail: V = [SHOULDER_X, 0, -0.08];
  const wristLead: V = [WRIST_X, 0, 0.095];
  const wristTrail: V = [WRIST_X, 0, -0.12];
  const handTrail: V = [0.48, 0, -0.15];
  const tip: V = [HALF_SPAN, 0, -0.12];
  const rightWing: V[][] = [
    [shoulderLead, wristLead, wristTrail],
    [shoulderLead, wristTrail, shoulderTrail],
    [wristLead, tip, handTrail],
    [wristLead, handTrail, wristTrail],
  ];
  const mirror = (v: V): V => [-v[0], v[1], v[2]];
  const leftWing: V[][] = rightWing.map((t) => t.map(mirror));

  const positions: number[] = [];
  const normals: number[] = [];
  const spans: number[] = [];
  const hands: number[] = [];
  const parts: number[] = [];

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const n = new THREE.Vector3();
  const out = new THREE.Vector3();

  const push = (tri: V[], kind: 'body' | 'sheet' | 'wing'): void => {
    a.fromArray(tri[0]);
    b.fromArray(tri[1]);
    c.fromArray(tri[2]);
    n.subVectors(b, a).cross(out.subVectors(c, a)).normalize();
    // Sheets face up; the body faces away from its own centre.
    const wantOutward =
      kind === 'body'
        ? out.set(
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3 - bodyCentre[0],
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3 - bodyCentre[1],
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3 - bodyCentre[2],
          )
        : out.set(0, 1, 0);
    const order = n.dot(wantOutward) >= 0 ? [0, 1, 2] : [0, 2, 1];
    if (order[1] === 2) n.negate();
    for (const k of order) {
      const v = tri[k];
      positions.push(v[0], v[1], v[2]);
      normals.push(n.x, n.y, n.z);
      const wing = kind === 'wing';
      spans.push(wing ? v[0] / HALF_SPAN : 0);
      hands.push(wing && Math.abs(v[0]) > WRIST_X + 1e-4 ? 1 : 0);
      parts.push(kind === 'body' ? 1 : 0);
    }
  };

  for (const t of bodyTris) push(t, 'body');
  for (const t of tailTris) push(t, 'sheet');
  for (const t of rightWing) push(t, 'wing');
  for (const t of leftWing) push(t, 'wing');

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  const shape: number[] = [];
  for (let i = 0; i < spans.length; i++) shape.push(spans[i], hands[i], parts[i]);
  geometry.setAttribute('shape', new THREE.Float32BufferAttribute(shape, 3));

  // The six static per-bird vectors share one interleaved buffer. Separate
  // attributes were a vertex buffer each, and with the per-vertex ones and the
  // perch state that came to twelve — over WebGPU's guaranteed limit of eight,
  // which fails pipeline creation outright rather than degrading.
  const data = buildInstanceData();
  const interleaved = new THREE.InstancedInterleavedBuffer(data, INSTANCE_STRIDE, 1);
  INSTANCE_LAYOUT.forEach((name, slot) => {
    geometry.setAttribute(name, new THREE.InterleavedBufferAttribute(interleaved, 4, slot * 4));
  });
  const perchState = new THREE.InstancedBufferAttribute(new Float32Array(MAX_BIRDS), 1);
  perchState.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('perchState', perchState);

  geometry.instanceCount = 0;
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

  return geometry;
}

/**
 * Per-bird vec4s, in their order in the interleaved instance buffer:
 *
 * - `orbit`: radius, angular rate (signed), start phase, altitude.
 * - `harmonics`: second harmonic amplitude and phase, third's amplitude and phase.
 * - `bob`: climb amplitude, climb phase, flap rate, flap phase.
 * - `style`: scale, glide-gate rate, glide-gate phase, glide bias.
 * - `role`: anchor weights — ship, island, free — and one unused.
 * - `perch`: perch point in ship space, and facing relative to the bow.
 */
const INSTANCE_LAYOUT = ['orbit', 'harmonics', 'bob', 'style', 'role', 'perch'] as const;
const INSTANCE_STRIDE = INSTANCE_LAYOUT.length * 4;

/**
 * The role of bird `i`. Perched birds take fixed early indices so every tier
 * that draws birds at all has some on the yards; the rest cycle through the
 * flying roles so every prefix has a mix — half working the ship, a quarter
 * each over the island and soaring.
 */
function roleOf(i: number): Role {
  if (PERCH_INDICES.includes(i)) return 'perch';
  const k = i % 8;
  if (k === 2 || k === 5) return 'island';
  if (k === 4 || k === 1) return 'soar';
  return 'ship';
}

/**
 * Per-role flight envelope: circuit radius, altitude, airspeed, scale, and the
 * glide bias range (higher glides more).
 *
 * Scale is a gull's for the birds that come close — a 1.3 m herring gull — and
 * well past it for the far ones. At the island's 1.4 km a real gull is under a
 * pixel; doubling it is the honest alternative to flying the flock close
 * enough to intersect the rigging, and it is the readable size.
 */
const ENVELOPES: Record<
  Role,
  {
    radius: [number, number];
    altitude: [number, number];
    speed: [number, number];
    scale: [number, number];
    glide: [number, number];
  }
> = {
  ship: { radius: [16, 46], altitude: [11, 34], speed: [8.5, 11.5], scale: [1.0, 1.3], glide: [-0.7, 0.5] },
  perch: { radius: [14, 30], altitude: [12, 26], speed: [8.5, 10.5], scale: [1.15, 1.3], glide: [-0.6, 0.3] },
  island: { radius: [200, 470], altitude: [130, 260], speed: [9, 12.5], scale: [2.0, 2.8], glide: [-0.4, 0.6] },
  soar: { radius: [55, 150], altitude: [60, 130], speed: [8, 10], scale: [1.6, 2.4], glide: [0.35, 0.9] },
};

/**
 * One bird's worth of constants, `MAX_BIRDS` times, from a fixed seed. Every
 * rate is quantised by `perWrap` to a whole number of cycles per `CLOCK_WRAP`.
 */
function buildInstanceData(): Float32Array {
  const random = mulberry32(SEED);

  const data = new Float32Array(MAX_BIRDS * INSTANCE_STRIDE);
  // Views onto one bird's slots, so the body below reads as named fields.
  const field = (name: (typeof INSTANCE_LAYOUT)[number]) => {
    const offset = INSTANCE_LAYOUT.indexOf(name) * 4;
    return {
      set: (bird: number, k: number, v: number) => {
        data[bird * INSTANCE_STRIDE + offset + k] = v;
      },
    };
  };
  const orbit = field('orbit');
  const harmonics = field('harmonics');
  const bob = field('bob');
  const style = field('style');
  const role = field('role');
  const perch = field('perch');

  // Direction and start phase per role, not per bird: mixed directions inside
  // one flock would have half of it flying through the other half. The ship
  // flock is split in two loose skeins going the same way.
  const rolePhase: Record<Role, number> = { ship: 0, perch: 0, island: 0, soar: 0 };
  const roleDirection: Record<Role, number> = { ship: 1, perch: 1, island: 1, soar: 1 };
  for (const r of ['ship', 'perch', 'island', 'soar'] as const) {
    rolePhase[r] = random() * Math.PI * 2;
    roleDirection[r] = random() < 0.4 ? -1 : 1;
  }
  roleDirection.perch = roleDirection.ship;

  const lerp = (range: [number, number], t: number): number => range[0] + (range[1] - range[0]) * t;

  for (let i = 0; i < MAX_BIRDS; i++) {
    const r = roleOf(i);
    const env = ENVELOPES[r];
    const radius = lerp(env.radius, random());

    // Laps from an *airspeed*, so a wide circuit takes longer than a tight one
    // instead of the inner birds crawling. Rounding to whole laps per wrap moves
    // the realised speed by a few percent.
    const speed = lerp(env.speed, random());
    const laps = Math.max(1, Math.round((speed * CLOCK_WRAP) / (2 * Math.PI * radius)));

    // Bounded so that 2*second + 3*third stays below the radius: see the
    // tangent note in `buildVertex`.
    const second = radius * (0.08 + random() * 0.14);
    const third = radius * (0.03 + random() * 0.05);

    orbit.set(i, 0, radius);
    orbit.set(i, 1, (roleDirection[r] * 2 * Math.PI * laps) / CLOCK_WRAP);
    // Two skeins per role, half a circuit apart, strung out rather than evenly
    // spread: a flock evenly spread around its own path reads as traffic.
    orbit.set(i, 2, rolePhase[r] + (i % 2) * Math.PI + (random() - 0.5) * 1.1);
    orbit.set(i, 3, lerp(env.altitude, random()));

    harmonics.set(i, 0, second);
    harmonics.set(i, 1, random() * Math.PI * 2);
    harmonics.set(i, 2, third);
    harmonics.set(i, 3, random() * Math.PI * 2);

    bob.set(i, 0, 1.5 + random() * (r === 'island' ? 12 : 5));
    bob.set(i, 1, random() * Math.PI * 2);
    // 2.3 to 3.6 Hz. A herring gull cruises around 3.
    bob.set(i, 2, perWrap(2.3 + random() * 1.3));
    bob.set(i, 3, random() * Math.PI * 2);

    style.set(i, 0, lerp(env.scale, random()));
    // Glide gate period, 14 to 30 seconds.
    style.set(i, 1, perWrap(1 / (14 + random() * 16)));
    style.set(i, 2, random() * Math.PI * 2);
    // Duty cycle, as a threshold on that oscillator: the higher, the more of
    // the circuit spent gliding.
    style.set(i, 3, lerp(env.glide, random()));

    const onShip = r === 'ship' || r === 'perch';
    role.set(i, 0, onShip ? 1 : 0);
    role.set(i, 1, r === 'island' ? 1 : 0);
    role.set(i, 2, r === 'soar' ? 1 : 0);

    const slot = PERCH_INDICES.indexOf(i);
    if (slot >= 0) {
      const p = PERCHES[slot];
      perch.set(i, 0, p[0]);
      perch.set(i, 1, p[1]);
      perch.set(i, 2, p[2]);
      perch.set(i, 3, p[3]);
    }
  }

  return data;
}

/**
 * Rounds a frequency in Hz to the nearest whole number of cycles per
 * `CLOCK_WRAP` and returns it as radians per second.
 */
function perWrap(hz: number): number {
  const cycles = Math.max(1, Math.round(hz * CLOCK_WRAP));
  return (2 * Math.PI * cycles) / CLOCK_WRAP;
}

function clampCount(count: number): number {
  if (!Number.isFinite(count)) return 0;
  return Math.max(0, Math.min(MAX_BIRDS, Math.floor(count)));
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
