import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  faceDirection,
  float,
  instanceIndex,
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
 * The whole system is **two instanced draws of one animation**: a cheap gull of
 * a few dozen faceted triangles for the flock at distance, and a detailed one —
 * a lofted body with a round head, aerofoil wings with notched primaries, a
 * tapered bill, a wedge tail, legs — for the birds near the camera (see
 * `LOD_DISTANCE`). Each draw collapses, in its vertex stage, the birds the other
 * one is drawing, so every bird is drawn exactly once. Both are deformed and
 * placed by the same vertex code. The CPU writes no transforms: per frame it
 * writes four uniforms, and a float per perched bird when one of them is moving
 * between the ship and the sky.
 *
 * References looked at before building the detailed gull: three.js's GPGPU
 * birds example (`BirdGeometry`, three triangles flapped on y in the vertex
 * shader — the same idea as the cheap gull here) and Sebastian Lague's Boids
 * (a small imported mesh, no procedural body). Neither has a body or wing
 * worth porting at close range, so the detailed gull is built here: a loft of
 * superellipse rings for the body, the NACA four-digit section for the wings.
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
 * 19.2 m, and the stern rail at 5.7 m. The yards are braced, so their arms do
 * not share an x. Each point is the top of the spar, so the bird's feet stand on
 * it (`PERCH_LIFT`).
 *
 * The taffrail bird was on the centreline at 5.8 m, which is the top of the
 * stern lantern's iron bracket (5.77 m, 10 cm wide) rather than the rail: from
 * the deck it stood 3 cm clear of anything, against the rail's crest 10 cm
 * lower. Re-cast on a 4 cm grid, the crest is a 16 cm-wide ridge at
 * 5.66–5.71 m; the bird now stands on it 15 cm to starboard of the bracket.
 */
const PERCHES: ReadonlyArray<readonly [number, number, number, number]> = [
  [6.4, 9.37, -3.1, 0.3], // fore yard, port arm
  [8.1, 9.38, 3.2, -0.4], // fore yard, starboard arm
  [-12.78, 5.685, 0.15, Math.PI * 0.92], // taffrail crest, looking aft
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
/**
 * Where the legs fold to in flight, and how low the feet reach on the perch,
 * in the authored frame (metres at scale 1). The hip is under the body's centre
 * and just inside the belly, and it is also the pivot of `PERCH_PITCH`, so the
 * legs stay vertical and the feet stay on the spar however the body tilts.
 */
const LEG_HIP = [0, -0.044, 0.01] as const;
const LEG_FOOT_Y = -0.12;
/**
 * Height of a perched bird's centre above its perch point, at scale 1: exactly
 * the feet's depth, so the feet stand on the spar.
 */
const PERCH_LIFT = -LEG_FOOT_Y;
/**
 * Nose-up tilt of a perched bird's body, radians (20 degrees). A resting gull
 * stands with its breast up and its folded wingtips angled down over the tail;
 * level, it reads as a bird in flight that has stopped.
 */
const PERCH_PITCH = 0.35;

/**
 * Plumage: a laughing gull (*Leucophaeus atricilla*), the common gull of the
 * Caribbean and the one a ship there would carry on its yards. Summer adult:
 * black hood, dark slate mantle and upper wing, black wingtips, a white
 * trailing edge to the wing, white body and tail, dark red bill, dark legs.
 * Linear albedo, chosen against field-guide plates rather than measured.
 *
 * Painted from one float per vertex (`shape.w`): the wing's chord (0 leading
 * edge, 1 trailing) on the wings — negated on the detailed wing's lower
 * surface, which says which side of the wing a fragment is on, and exactly 0
 * and 1 on the leading and trailing edges the two surfaces share, which are
 * neither (see `upness` in the vertex stage); the cheap gull's sheets, all
 * upper side, use 0.001 and 0.999 for their edges — `PAINT_BILL`
 * and `PAINT_LEG` on those parts, `PAINT_PLUMAGE` everywhere else. No texture
 * and no extra attribute.
 */
const PAINT_PLUMAGE = 0;
const PAINT_BILL = 2;
const PAINT_LEG = 3;
const GULL_WHITE = [0.86, 0.87, 0.88] as const;
/**
 * Slate grey, not black. At 0.15 the upper wing in shade, which is most of the
 * time a bird is seen from above or banked, went to near-black and the flock
 * read as crows.
 */
const GULL_MANTLE = [0.25, 0.265, 0.3] as const;
const GULL_HOOD = [0.03, 0.03, 0.035] as const;
const GULL_TIPS = [0.025, 0.025, 0.03] as const;
const GULL_BILL = [0.36, 0.05, 0.035] as const;
const GULL_LEGS = [0.07, 0.03, 0.03] as const;
/**
 * Soft shading under the bird: how much of the shade light the underwing near
 * the body and the belly lose to the body and wings over them.
 */
const UNDER_OCCLUSION = 0.35;

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
/**
 * How far a backlit wing's upper side pales toward the light coming through it:
 * gain on the transmission term, and the grey (a fraction of the white) it pales
 * to. See `mantleTone`.
 */
const BACKLIT_MANTLE = [3.2, 0.8] as const;

/** Wing half-span of the authored geometry, metres, at instance scale 1. */
const HALF_SPAN = 0.66;
/** Lateral offset of the shoulder joint — the pivot the arm rotates about. */
const SHOULDER_X = 0.045;
/** Lateral offset of the wrist — the pivot the hand rotates about. */
const WRIST_X = 0.3;
/**
 * Height of the wing's chord plane above the body's axis. A gull's wing roots
 * in the upper half of the body; at 0 the wing came out of the middle of the
 * flank and the bird read as low-winged.
 */
const WING_Y = 0.018;

/**
 * The folded wing of a perched bird, in the authored frame at scale 1.
 *
 * The wing is laid along the body as a resting gull carries it. Span runs back
 * along the body from the shoulder (`FOLD_FRONT_Z`, `FOLD_LENGTH`); chord runs
 * round the body's cross-section, from high on the back (`FOLD_PHI_LE`, angle
 * from the top) down to mid-flank (`FOLD_PHI_TE`), hugging it at
 * `FOLD_CLEAR`, so the dark mantle shows between the two wings and the white
 * belly below them. Behind the rump (`FOLD_CROSS_START` of the span) the hand
 * lies down flat over the tail and slides inboard until the two sets of black
 * primaries cross (`FOLD_CROSS`) and project past it. The authored right wing
 * stacks over the left by `FOLD_STACK` where they cross, so they never share a
 * depth.
 *
 * The body's cross-section is an analytic fit to `BODY_STATIONS`
 * (`FOLD_BODY_*`): half-width and half-height an ellipse in z about
 * `FOLD_BODY_MID_Z`, centre rising toward the rump. It is a fixed map from each
 * wing vertex's span and chord fractions, so the cheap and detailed gulls fold
 * the same way, and its frame is orthonormal, so normals map with it.
 */
const FOLD_FRONT_Z = 0.09;
const FOLD_LENGTH = 0.47;
const FOLD_PHI_LE = 0.5;
const FOLD_PHI_TE = 1.5;
// 7% over the body's own half-width and half-height: the body is a
// superellipse (`BODY_ROUNDNESS`), fuller than this ellipse on the diagonals,
// and at the fit's exact size its flank came through the folded wing's
// trailing edge as z-fighting stripes.
const FOLD_BODY_W = 0.062;
const FOLD_BODY_H = 0.064;
const FOLD_BODY_MID_Z = 0.03;
const FOLD_BODY_HALF_LENGTH = 0.22;
const FOLD_CROSS_START = 0.55;
const FOLD_BACK_X = 0.03;
const FOLD_CROSS = 0.042;
/** Height of the hand over the tail where it lies flat, and its width there. */
const FOLD_TAIL_Y = 0.034;
const FOLD_TAIL_WIDTH = 0.03;
const FOLD_STACK = 0.004;
/** Gap between the folded wing's underside and the body, metres. */
const FOLD_CLEAR = 0.004;
/** How much of the wing section's thickness and camber survive folding. */
const FOLD_FLATTEN = 0.35;
/** How much narrower the tail is closed on the perch than spread in flight. */
const TAIL_CLOSE = 0.55;

/**
 * Level of detail. A bird is drawn with the detailed gull while it is closer
 * than this many metres per unit of its scale — 36 m for a 1.0 gull, 47 m for a
 * 1.3 one, where it spans about 45 px of a 1080p frame at the default lens —
 * and with the cheap one beyond. The switch is hard: at that size the two
 * silhouettes are within a pixel of each other, and a crossfade would draw both.
 */
const LOD_DISTANCE = 36;
/**
 * Only the first this-many instances can take the detailed gull: the detailed
 * draw's instance count is capped here, so its vertex cost is fixed whatever the
 * tier's flock size. Every perched bird and most of the ship flock's are below
 * it (`roleOf`); the island and soaring birds below it are never that close and
 * collapse in its vertex stage.
 */
const DETAIL_INSTANCES = 32;

export class Birds {
  /** Instances the buffers hold. `setCount` is clamped to this. */
  static readonly MAX_COUNT = MAX_BIRDS;

  readonly object: THREE.Object3D;

  /** The cheap gull, every bird; and the detailed one, the first `DETAIL_INSTANCES`. */
  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly detailGeometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly detailMaterial: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;
  private readonly detailMesh: THREE.Mesh;
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

    // Both gulls read the same per-bird buffers, so a bird is the same bird
    // whichever of them draws it.
    const instances = buildInstanceAttributes();
    this.perchState = instances.perchState;
    this.geometry = withInstances(buildCheapGull(), instances);
    this.detailGeometry = withInstances(buildDetailGull(), instances);
    this.applyCount();

    const random = mulberry32(SEED ^ 0x5eed);
    for (let k = 0; k < PERCHES.length; k++) {
      // Not all at once: a row of gulls lifting off in the same frame is a
      // sprite sheet. Staggered over about a second and a half, the first one
      // up spooks the rest.
      this.perchDelay[k] = 0.15 + random() * 1.4;
      // And back at different times, so the yards fill up one bird at a time.
      this.perchReturn[k] = 16 + random() * 22;
    }

    this.material = this.buildMaterial(false);
    this.detailMaterial = this.buildMaterial(true);

    const makeMesh = (geometry: THREE.InstancedBufferGeometry, material: THREE.Material, name: string): THREE.Mesh => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = name;
      // Placed in world space by the vertex stage, so a bounding sphere is
      // meaningless.
      mesh.frustumCulled = false;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      return mesh;
    };
    this.mesh = makeMesh(this.geometry, this.material, 'gulls');
    this.detailMesh = makeMesh(this.detailGeometry, this.detailMaterial, 'gulls-near');

    this.object = new THREE.Object3D();
    this.object.name = 'birds';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh, this.detailMesh);
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
    this.applyCount();
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
    this.object.remove(this.mesh, this.detailMesh);
    this.object.removeFromParent();
    this.geometry.dispose();
    this.detailGeometry.dispose();
    this.material.dispose();
    this.detailMaterial.dispose();
  }

  // ------------------------------------------------------------------ internals

  /** The cheap draw takes every bird; the detailed one only those that can be close. */
  private applyCount(): void {
    this.geometry.instanceCount = this.count;
    this.detailGeometry.instanceCount = Math.min(this.count, DETAIL_INSTANCES);
  }

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

  /** `detail`: the detailed gull's draw (near birds) or the cheap one's (the rest). */
  private buildMaterial(detail: boolean): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    // The cheap gull's wings and tail are sheets with no thickness, and both
    // gulls' feet are, so both faces rasterise.
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
    material.positionNode = Fn(() => this.buildVertex(detail).world)() as any;

    // `outboard` (wings) and `zone` (body) share a slot: every triangle is all
    // one part, so `part` unpacks them exactly.
    const shading: any = varying(
      Fn(() => {
        const vertex = this.buildVertex(detail);
        return vec4(vertex.normal, mixn(vertex.outboard, vertex.zone, vertex.part));
      })(),
    );
    const bodyShading: any = varying(
      Fn(() => {
        const vertex = this.buildVertex(detail);
        return vec4(vertex.part, vertex.upness, vertex.head, vertex.paint);
      })(),
    );

    const normal: any = shading.xyz.normalize();
    /** 1 on the body's volume, 0 on the wings and tail. */
    const part: any = bodyShading.x;
    /** Distance out the span, 0 at the shoulder and 1 at the tip. */
    const outboard: any = shading.w.mul(part.oneMinus());
    /** 1 on the body between the neck and the rump, where the mantle is. */
    const zone: any = shading.w.mul(part);
    /**
     * On the body, the surface normal's up component: back +1, belly -1. On the
     * wings and tail, the *rest* normal's: +1 on the upper surface, -1 under.
     */
    const upness: any = bodyShading.y;
    /** 1 on the hood, 0 from the nape back. */
    const head: any = bodyShading.z;

    // `faceDirection` is +1 on a sheet's back and -1 on its belly, because the
    // cheap gull's sheets are wound counter-clockwise seen from above; on a
    // closed surface it is +1 wherever a face is visible at all, and `upness`
    // says which side it is. Their product is the wing's upper side either way.
    const facing: any = faceDirection;
    const wingUp: any = facing.mul(upness);
    const seen: any = normal.mul(facing);
    const ndl: any = seen.dot(this.uSunDir);

    // Wrapped diffuse: a hard terminator on a body this small in screen space is
    // not shading, it is a flicker.
    const direct: any = ndl.mul(0.5).add(0.5).mul(this.uSunGain);
    const through: any = ndl.negate().max(0).mul(TRANSMISSION).mul(this.uSunGain);
    const lit: any = direct.add(through).clamp(0, 1);

    /** `shape.w`: the wing's chord, or `PAINT_BILL` / `PAINT_LEG`. See `PAINT_PLUMAGE`. */
    const paint: any = bodyShading.w;
    /** 1 on the wings; the tail sheet is the one sheet with no span. */
    const wing: any = outboard.step(0.02);
    const white = vec3(...GULL_WHITE);

    // Dark mantle above, white below — the countershading that makes a gull read
    // as a gull the instant it banks and shows the other side. On the body the
    // mantle is the back; the tail is white both sides.
    const sheetTop: any = wingUp.smoothstep(-0.3, 0.3).mul(wing);
    // The body's mantle is the back between the wings: white neck in front of
    // it, white rump behind.
    const bodyTop: any = upness.smoothstep(0.15, 0.75).mul(zone);
    const topness: any = mix(sheetTop, bodyTop, part);
    // Against the light the wing's upper side is the shaded one, and slate in
    // shadow read black against a bright sky. What a backlit gull's wing shows
    // there is light through it, which comes through white underwing feathers:
    // the mantle tends to pale grey with the transmission (`BACKLIT_MANTLE`).
    const mantleTone: any = mix(
      vec3(...GULL_MANTLE),
      white.mul(BACKLIT_MANTLE[1]),
      through.mul(wing).mul(BACKLIT_MANTLE[0]).clamp(0, 1),
    );
    const mantled: any = mix(white, mantleTone, topness);
    // The white trailing edge of the inner wing, on top: the line that shows
    // the wing's shape against the dark mantle.
    const trailing: any = paint
      .abs()
      .smoothstep(0.8, 0.97)
      .mul(wing)
      .mul(part.oneMinus())
      .mul(outboard.smoothstep(0.5, 0.65).oneMinus());
    const edged: any = mix(mantled, white, trailing.mul(sheetTop));
    // The black hood: the whole head, crown to throat.
    const hooded: any = mix(edged, vec3(...GULL_HOOD), head.smoothstep(0.35, 0.8));
    // Black primaries, on the hand only.
    const tipped: any = mix(hooded, vec3(...GULL_TIPS), outboard.smoothstep(0.62, 0.95));
    // Bill and legs: exact per triangle, since every vertex of one carries the
    // same code.
    const bill: any = paint.step(PAINT_BILL - 0.5).mul(paint.step(PAINT_LEG - 0.5).oneMinus());
    const legs: any = paint.step(PAINT_LEG - 0.5);
    const albedo: any = mix(mix(tipped, vec3(...GULL_BILL), bill), vec3(...GULL_LEGS), legs);

    // Soft shading under the bird (`UNDER_OCCLUSION`): the underwing toward the
    // body, which the body and the other wing hide part of the sky from, and the
    // belly, which the wings do. Without it the underside was one flat tone, and
    // a flat white underside is most of what read as a cut-out.
    const underWing: any = wingUp
      .negate()
      .max(0)
      .mul(wing)
      .mul(outboard.smoothstep(0.15, 0.6).oneMinus());
    const underBody: any = upness.negate().smoothstep(0.1, 0.8).mul(part);
    const occlusion: any = underWing.max(underBody).mul(UNDER_OCCLUSION);

    const light: any = mix(this.uShadeColor.mul(occlusion.oneMinus()), this.uSunColor.mul(occlusion.mul(0.5).oneMinus()), lit);
    material.colorNode = vec4(albedo.mul(light), 1) as any;    return material;
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
  private buildVertex(detail: boolean): {
    world: any;
    normal: any;
    outboard: any;
    zone: any;
    part: any;
    upness: any;
    head: any;
    paint: any;
  } {
    const orbit: any = attribute('orbit', 'vec4');
    const harmonics: any = attribute('harmonics', 'vec4');
    const bob: any = attribute('bob', 'vec4');
    const style: any = attribute('style', 'vec4');
    const role: any = attribute('role', 'vec4');
    const perch: any = attribute('perch', 'vec4');
    const perched: any = attribute('perchState', 'float');
    const shape: any = attribute('shape', 'vec4');
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

    // In flight the normal turns with the wing about the bird's long axis: the
    // arm angle, plus the wrist angle on the hand. On the body and tail both are
    // exactly 0 and this is the authored normal.
    const sheetAngle = shoulder.add(wrist).toVar();
    const sheetCos = sheetAngle.cos().toVar();
    const sheetSin = sheetAngle.sin().toVar();
    const flyingNormal = vec3n(
      restNormal.x.mul(sheetCos).sub(restNormal.y.mul(sheetSin)),
      restNormal.x.mul(sheetSin).add(restNormal.y.mul(sheetCos)),
      restNormal.z,
    );

    // Folded along the back for a bird on its perch. See `FOLD_FRONT_Z`: a map
    // from span fraction (back along the body) and chord fraction (down over
    // the flank), with the wing's own thickness laid outward.
    const tuck = perched.smoothstep(0.55, 1.0).mul(side.abs()).toVar();
    const along = rest.x
      .abs()
      .sub(SHOULDER_X)
      .div(HALF_SPAN - SHOULDER_X)
      .clamp(0, 1)
      .toVar();
    // Chord fraction; negative on the detailed wing's lower surface (see
    // `PAINT_PLUMAGE`).
    const chord = shape.w.abs().clamp(0, 1);
    const crossing = along.smoothstep(FOLD_CROSS_START, 1).toVar();
    const foldZ = float(FOLD_FRONT_Z).sub(along.mul(FOLD_LENGTH)).toVar();
    // The body's cross-section at this station (see `FOLD_BODY_W`).
    const bodyFrom = foldZ.sub(FOLD_BODY_MID_Z).div(FOLD_BODY_HALF_LENGTH);
    const bodyExtent = bodyFrom.mul(bodyFrom).oneMinus().max(0).sqrt().toVar();
    const bodyW = bodyExtent.mul(FOLD_BODY_W).toVar();
    const bodyH = bodyExtent.mul(FOLD_BODY_H).toVar();
    const bodyY = foldZ.negate().sub(0.04).max(0).mul(0.1).add(0.006);
    // Thickness off the body: the wing's own section, upper surface outermost.
    // Pressed flatter than the open section (`FOLD_FLATTEN`): a folded wing's
    // feathers lie close, and at full thickness it stood off the flank as a
    // plank.
    const lift = rest.y.sub(WING_Y).mul(FOLD_FLATTEN).add(FOLD_CLEAR).toVar();
    // Round the flank: chord fraction to angle from the top of the body, about
    // the middle of the folded wing's band. The hand's chord is narrower
    // folded, and the band closes in toward the bend of the wing at the front,
    // which tucks into the breast instead of ending square.
    const band = along.smoothstep(0, 0.18).mul(0.65).add(0.35).mul(along.mul(-0.5).add(1));
    const phi = chord
      .sub(0.4)
      .mul(band)
      .mul(FOLD_PHI_TE - FOLD_PHI_LE)
      .add(FOLD_PHI_LE + 0.4 * (FOLD_PHI_TE - FOLD_PHI_LE))
      .toVar();
    const phiSin = phi.sin().toVar();
    const phiCos = phi.cos().toVar();
    const hug = vec3n(
      side.mul(bodyW.add(lift)).mul(phiSin),
      bodyY.add(bodyH.add(lift).mul(phiCos)),
      foldZ,
    );
    // Over the tail: flat, the tips crossing, the trailing edge outboard.
    const flat = vec3n(
      side.mul(float(FOLD_BACK_X).sub(crossing.mul(FOLD_CROSS)).add(chord.mul(FOLD_TAIL_WIDTH))),
      float(FOLD_TAIL_Y).add(lift).add(side.mul(crossing).mul(FOLD_STACK)),
      foldZ,
    );
    const folded = mixn(hug, flat, crossing);
    // The frame the wing's section is laid in: the outward normal of the body's
    // ellipse there, and the direction round it toward the trailing edge; over
    // the tail, straight up and straight out.
    const hugOut = vec3n(side.mul(bodyH.max(1e-3)).mul(phiSin), bodyW.max(1e-3).mul(phiCos), 0).normalize();
    const hugDown = vec3n(side.mul(bodyW.max(1e-3)).mul(phiCos), bodyH.max(1e-3).mul(phiSin).negate(), 0).normalize();
    const outward = mixn(hugOut, vec3(0, 1, 0), crossing).normalize().toVar();
    const down = mixn(hugDown, vec3n(side, 0, 0), crossing).normalize().toVar();
    // The same map applied to the normal: up -> outward, toward the trailing
    // edge -> down, out along the span -> back along the body.
    const foldedNormal = outward
      .mul(restNormal.y)
      .add(down.mul(restNormal.z.negate()))
      .add(vec3(0, 0, -1).mul(restNormal.x.mul(side)));

    // A resting gull closes its tail; spread, it stuck out either side of the
    // body seen head-on. The tail is the one part with neither span nor volume.
    const tailClose = part.oneMinus().mul(side.abs().oneMinus()).mul(settle).mul(TAIL_CLOSE).oneMinus();
    const posed = mixn(flying, folded, tuck).mul(vec3n(tailClose, 1, 1)).toVar();
    const posedNormal = mixn(flyingNormal, foldedNormal, tuck).toVar();

    // Nose up on the perch (`PERCH_PITCH`), about the hip, so the feet stay
    // where they are. The legs themselves stay vertical.
    const legPart = shape.w.step(PAINT_LEG - 0.5).toVar();
    const pitch = settle.mul(PERCH_PITCH).mul(legPart.oneMinus()).toVar();
    const pitchCos = pitch.cos().toVar();
    const pitchSin = pitch.sin().toVar();
    const hip = vec3(...LEG_HIP);
    const fromHip = posed.sub(hip).toVar();
    const pitched = vec3n(
      fromHip.x,
      fromHip.y.mul(pitchCos).add(fromHip.z.mul(pitchSin)),
      fromHip.z.mul(pitchCos).sub(fromHip.y.mul(pitchSin)),
    ).add(hip);
    const localNormal = vec3n(
      posedNormal.x,
      posedNormal.y.mul(pitchCos).add(posedNormal.z.mul(pitchSin)),
      posedNormal.z.mul(pitchCos).sub(posedNormal.y.mul(pitchSin)),
    ).toVar();

    // Legs fold up into the belly in flight; see LEG_HIP.
    const leg = legPart.mul(perched.smoothstep(0.6, 1.0).oneMinus());
    const shaped = mixn(pitched, hip, leg).toVar();

    // --- placement -----------------------------------------------------------
    // See `NEAR_HIDE`. Scaling to zero costs no fragments; an alpha fade would.
    const distance = centreFlight.sub(this.uViewer).length().toVar();
    // Which draw this bird is in: see `LOD_DISTANCE` and `DETAIL_INSTANCES`.
    // The other draw collapses it the same way the near hide does.
    // (`a.step(b)` is 1 where a >= b.)
    const near = float(LOD_DISTANCE)
      .mul(style.x)
      .step(distance)
      .mul(float(DETAIL_INSTANCES - 0.5).step(float(instanceIndex)));
    const drawn = detail ? near : near.oneMinus();
    const size = style.x.mul(distance.smoothstep(NEAR_HIDE, NEAR_SHOW)).mul(drawn).toVar();
    const local = shaped.mul(size);

    // Authored +X goes to `-bankRight`. `bankRight` is forward x up, so
    // `[bankRight, bankUp, forward]` is a reflection, and under a reflection a
    // surface wound outward rasterises as back faces: `faceDirection` was -1 on
    // every visible face of a closed body, and +1 on the cheap gull's sheets
    // seen from *below*, so the mantle was painted on the underside of every
    // bird in flight and the light came from the wrong side of the body
    // (measured: facing rendered as colour). With `-bankRight` the frame is a
    // rotation. The gull is symmetric, so its shape does not change, and the
    // bank still drops the wing on the inside of the turn.
    const world = centreFlight
      .sub(bankRight.mul(local.x))
      .add(bankUp.mul(local.y))
      .add(forward.mul(local.z));

    // The fold is a reflection too — span to back, thickness to outward, chord
    // to down has determinant -1 — so a folded wing's outer, upper surface
    // rasterises as a back face. Flipping the normal and the up-side sign once
    // the wing is mostly folded puts both back.
    const foldSign = tuck.step(0.5).mul(-2).add(1).toVar();
    const normal = bankRight
      .mul(localNormal.x)
      .negate()
      .add(bankUp.mul(localNormal.y))
      .add(forward.mul(localNormal.z))
      .mul(foldSign);

    // The hood: the head, from a line that runs from the nape down and forward
    // to the throat. Measured on the rest pose, so it does not slide when the
    // body tilts.
    const head = rest.z.add(rest.y.mul(0.25)).smoothstep(0.155, 0.215).mul(part);
    // The mantle's extent on the body: from the rump forward to the neck.
    const zone = rest.z
      .smoothstep(-0.15, -0.07)
      .mul(rest.z.smoothstep(0.12, 0.17).oneMinus())
      .mul(part);

    return {
      world,
      normal,
      outboard,
      zone,
      part,
      // Which side of the wing this is: the cheap gull's sheets are all upper
      // side (+1) and `faceDirection` tells the faces apart; the detailed wing
      // marks its lower surface with a negative chord, and the edges the two
      // surfaces share are neither (0). Not the rest normal's sign, which is 0
      // round the leading edge and left a white rim along the top of the folded
      // wing; and not +1 on the shared edges, which painted a mantle rim along
      // the underwing's leading edge.
      upness: mixn(shape.w.sign().mul(shape.w.abs().step(0.9995).oneMinus()).mul(foldSign), localNormal.y, part),
      head,
      paint: shape.w,
    };
  }
}

const WHITE = /*@__PURE__*/ new THREE.Color(1, 1, 1);

/**
 * The gulls are authored nose-forward: **+X right wing, +Y up, +Z nose**, the
 * frame the vertex stage builds, so placement is three multiply-adds. Both carry
 * the same per-vertex attributes:
 *
 * - `shape.x`, span: signed distance out the wing, normalised to the tip. Its
 *   sign mirrors the flap; its magnitude drives the dark primaries. 0 on the
 *   body and tail, which excludes them from every wing rotation.
 * - `shape.y`, hand: 1 on the vertices outboard of the wrist, which take the
 *   wrist rotation as well as the shoulder's.
 * - `shape.z`, part: 1 on the body's volume (with the bill and legs), 0 on the
 *   wings and tail.
 * - `shape.w`, paint: see `PAINT_PLUMAGE`.
 *
 * The cheap gull's sheets are wound counter-clockwise seen from above, which
 * is what makes `faceDirection` mean "this is the bird's back"; every closed
 * surface is wound outward. Both are enforced as the triangles are added rather
 * than trusted to the vertex order.
 */
class GullBuilder {
  readonly positions: number[] = [];
  readonly shape: number[] = [];
  readonly normals: number[] = [];
  readonly index: number[] = [];
  private readonly a = new THREE.Vector3();
  private readonly b = new THREE.Vector3();
  private readonly c = new THREE.Vector3();
  private readonly n = new THREE.Vector3();
  private readonly t = new THREE.Vector3();

  /** Adds a vertex and returns its index. */
  vertex(p: Vec, span: number, hand: number, part: number, paint: number): number {
    this.positions.push(p[0], p[1], p[2]);
    this.shape.push(span, hand, part, paint);
    return this.positions.length / 3 - 1;
  }

  /**
   * An indexed triangle, wound so its face normal agrees with `outward` — a
   * direction, or a point inside the surface (`inside`) the face must face away
   * from.
   */
  tri(i: number, j: number, k: number, outward: Vec, inside = false): void {
    const p = this.positions;
    this.a.fromArray(p, i * 3);
    this.b.fromArray(p, j * 3);
    this.c.fromArray(p, k * 3);
    if (this.facesOut(outward, inside)) this.index.push(i, j, k);
    else this.index.push(i, k, j);
  }

  /** Whether `a`, `b`, `c` as loaded wind toward `outward`. Leaves the unit face normal in `n`. */
  private facesOut(outward: Vec, inside: boolean): boolean {
    this.n.subVectors(this.b, this.a).cross(this.t.subVectors(this.c, this.a)).normalize();
    if (inside) {
      this.t.copy(this.a).add(this.b).add(this.c).divideScalar(3);
      this.t.set(this.t.x - outward[0], this.t.y - outward[1], this.t.z - outward[2]);
    } else {
      this.t.set(outward[0], outward[1], outward[2]);
    }
    return this.n.dot(this.t) >= 0;
  }

  /**
   * A non-indexed, flat-shaded triangle with its own face normal, for the cheap
   * gull. Same winding rule as `tri`.
   */
  facet(tri: readonly Vec[], outward: Vec, inside: boolean, attrs: (v: Vec) => [number, number, number, number]): void {
    this.a.fromArray(tri[0]);
    this.b.fromArray(tri[1]);
    this.c.fromArray(tri[2]);
    const flip = !this.facesOut(outward, inside);
    if (flip) this.n.negate();
    for (const k of flip ? [0, 2, 1] : [0, 1, 2]) {
      this.vertex(tri[k], ...attrs(tri[k]));
      this.normals.push(this.n.x, this.n.y, this.n.z);
    }
  }

  /**
   * A loft: rings of `ring` points per station, joined into a closed tube, with
   * each end either left open or closed on a single point. `centre(s)` is a point
   * inside station `s`'s ring, for the winding. Returns each station's vertex
   * indices.
   */
  loft(
    stations: number,
    ring: number,
    point: (s: number, k: number) => Vec,
    centre: (s: number) => Vec,
    attrs: (v: Vec, s: number, k: number) => [number, number, number, number],
    caps: { start?: Vec; end?: Vec } = {},
    outwardOf?: (s: number, k: number) => Vec,
  ): number[][] {
    const rows: number[][] = [];
    for (let s = 0; s < stations; s++) {
      const row: number[] = [];
      for (let k = 0; k < ring; k++) {
        const v = point(s, k);
        row.push(this.vertex(v, ...attrs(v, s, k)));
      }
      rows.push(row);
    }
    for (let s = 0; s + 1 < stations; s++) {
      const mid = centre(s).map((x, i) => (x + centre(s + 1)[i]) / 2) as unknown as Vec;
      for (let k = 0; k < ring; k++) {
        const k1 = (k + 1) % ring;
        const a = rows[s][k];
        const b = rows[s][k1];
        const c = rows[s + 1][k];
        const d = rows[s + 1][k1];
        const out = outwardOf?.(s, k);
        this.tri(a, b, d, out ?? mid, out === undefined);
        this.tri(a, d, c, out ?? mid, out === undefined);
      }
    }
    const cap = (s: number, tip: Vec, away: number): void => {
      const t = this.vertex(tip, ...attrs(tip, s, -1));
      const dir: Vec = [0, 0, 0];
      const c = centre(s);
      for (let i = 0; i < 3; i++) dir[i] = tip[i] - c[i];
      if (Math.hypot(dir[0], dir[1], dir[2]) < 1e-6) dir[2] = away;
      for (let k = 0; k < ring; k++) this.tri(rows[s][k], rows[s][(k + 1) % ring], t, dir);
    };
    if (caps.start) cap(0, caps.start, -1);
    if (caps.end) cap(stations - 1, caps.end, 1);
    return rows;
  }
}

type Vec = [number, number, number];

/** A superellipse ring point: `k` of `n` round from the top, half-width `w`, half-heights up and down. */
function ringPoint(k: number, n: number, w: number, up: number, down: number, exponent: number): [number, number] {
  const phi = (k / n) * Math.PI * 2;
  const s = Math.sin(phi);
  const c = Math.cos(phi);
  const e = 2 / exponent;
  const x = w * Math.sign(s) * Math.abs(s) ** e;
  const y = (c >= 0 ? up : down) * Math.sign(c) * Math.abs(c) ** e;
  return [x, y];
}

/**
 * The cheap gull: twenty-odd faceted triangles. Non-indexed, every triangle
 * with its own face normal, because at a few dozen pixels a smooth-shaded
 * lozenge reads as a blob, a faceted one as a body with a back and a belly.
 */
function buildCheapGull(): THREE.InstancedBufferGeometry {
  type V = Vec;
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
  const y = WING_Y;
  // Matched to the detailed wing's planform (`WING_STATIONS`), narrow and
  // pointed, so the LOD switch does not change the bird's shape.
  const shoulderLead: V = [SHOULDER_X, y, 0.072];
  const shoulderTrail: V = [SHOULDER_X, y, -0.068];
  const wristLead: V = [WRIST_X, y, 0.096];
  const wristTrail: V = [WRIST_X, y, -0.09];
  const handTrail: V = [0.48, y, -0.096];
  const tip: V = [HALF_SPAN, y, -0.08];
  const rightWing: V[][] = [
    [shoulderLead, wristLead, wristTrail],
    [shoulderLead, wristTrail, shoulderTrail],
    [wristLead, tip, handTrail],
    [wristLead, handTrail, wristTrail],
  ];
  const mirror = (v: V): V => [-v[0], v[1], v[2]];
  const leftWing: V[][] = rightWing.map((t) => t.map(mirror));

  // Bill: a short wedge from the face, in front of the hood.
  const billTop: V = [0, 0.03, 0.27];
  const billLeft: V = [-0.012, 0.012, 0.27];
  const billRight: V = [0.012, 0.012, 0.27];
  const billChin: V = [0, -0.004, 0.27];
  const billTip: V = [0, 0.004, 0.365];
  const billTris: V[][] = [
    [billTop, billRight, billTip],
    [billRight, billChin, billTip],
    [billChin, billLeft, billTip],
    [billLeft, billTop, billTip],
  ];

  // Legs: a thin blade from the hip to the foot, and a webbed foot, each side.
  // Drawn on the perch only; in flight they fold into the belly (the vertex
  // stage collapses them onto `LEG_HIP`).
  const legTris: V[][] = [];
  const footTris: V[][] = [];
  for (const sx of [1, -1]) {
    const hipA: V = [sx * 0.02, LEG_HIP[1], LEG_HIP[2] + 0.008];
    const hipB: V = [sx * 0.02, LEG_HIP[1], LEG_HIP[2] - 0.008];
    const ankle: V = [sx * 0.022, LEG_FOOT_Y + 0.004, LEG_HIP[2] + 0.004];
    const heel: V = [sx * 0.022, LEG_FOOT_Y + 0.004, LEG_HIP[2] - 0.008];
    legTris.push([hipA, ankle, heel], [hipA, heel, hipB]);
    footTris.push([
      [sx * 0.022, LEG_FOOT_Y, LEG_HIP[2] - 0.01],
      [sx * 0.04, LEG_FOOT_Y, LEG_HIP[2] + 0.045],
      [sx * 0.004, LEG_FOOT_Y, LEG_HIP[2] + 0.045],
    ]);
  }

  // Wing chord, 0 on the leading edge and 1 on the trailing (and the tip), for
  // the white trailing edge and the fold. The leading-edge vertices are the
  // only ones forward of the body's centre.
  const chordOf = (v: V): number => (v[2] > 0.05 ? 0.001 : 0.999);

  const g = new GullBuilder();
  const up: Vec = [0, 1, 0];
  const body = (paint: number) => (): [number, number, number, number] => [0, 0, 1, paint];
  for (const t of bodyTris) g.facet(t, bodyCentre, true, body(PAINT_PLUMAGE));
  for (const t of billTris) g.facet(t, bodyCentre, true, body(PAINT_BILL));
  for (const t of legTris) g.facet(t, [t[0][0] > 0 ? 1 : -1, 0, 0], false, body(PAINT_LEG));
  for (const t of footTris) g.facet(t, up, false, body(PAINT_LEG));
  for (const t of tailTris) g.facet(t, up, false, () => [0, 0, 0, PAINT_PLUMAGE]);
  const wingAttrs = (v: V): [number, number, number, number] => [
    v[0] / HALF_SPAN,
    Math.abs(v[0]) > WRIST_X + 1e-4 ? 1 : 0,
    0,
    chordOf(v),
  ];
  for (const t of [...rightWing, ...leftWing]) g.facet(t, up, false, wingAttrs);

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(g.positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(g.normals, 3));
  geometry.setAttribute('shape', new THREE.Float32BufferAttribute(g.shape, 4));
  return geometry;
}

/**
 * Body of the detailed gull: stations along the long axis, nose last —
 * `[z, centre y, half-width, half-height above, half-height below]`, metres at
 * scale 1. A teardrop from the rump to the breast, a short neck at 0.175, a
 * round head, and a face narrowing into the bill. Matched to the cheap gull's
 * extents so the LOD switch does not move the silhouette.
 */
const BODY_STATIONS: ReadonlyArray<readonly [number, number, number, number, number]> = [
  [-0.215, 0.022, 0.01, 0.006, 0.006],
  [-0.17, 0.02, 0.026, 0.022, 0.02],
  [-0.11, 0.012, 0.042, 0.04, 0.042],
  [-0.04, 0.006, 0.054, 0.056, 0.058],
  [0.03, 0.006, 0.058, 0.06, 0.06],
  [0.09, 0.01, 0.054, 0.054, 0.054],
  [0.14, 0.02, 0.042, 0.042, 0.042],
  [0.175, 0.03, 0.03, 0.032, 0.03],
  [0.2, 0.04, 0.031, 0.033, 0.03],
  [0.23, 0.044, 0.034, 0.034, 0.031],
  [0.258, 0.042, 0.031, 0.03, 0.027],
  [0.28, 0.036, 0.023, 0.022, 0.02],
  [0.294, 0.03, 0.012, 0.012, 0.011],
];
/** Rings round the body, and the superellipse exponent: a little fuller than an ellipse. */
const BODY_SEGMENTS = 14;
const BODY_ROUNDNESS = 2.4;
/** The bill: `[z, centre y, half-width, half-height]`. Long, and drooping at the tip. */
const BILL_STATIONS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.284, 0.029, 0.01, 0.011],
  [0.315, 0.026, 0.0078, 0.0085],
  [0.345, 0.022, 0.0055, 0.0065],
  [0.366, 0.016, 0.003, 0.004],
];
const BILL_TIP: Vec = [0, 0.011, 0.378];
/** The tail: `[z, half-width]` of a thin wedge from the rump, widening to a squared end. */
const TAIL_STATIONS: ReadonlyArray<readonly [number, number]> = [
  [-0.15, 0.028],
  [-0.21, 0.05],
  [-0.27, 0.066],
  [-0.318, 0.076],
  [-0.334, 0.066],
];
const TAIL_THICKNESS = 0.005;
/**
 * The wing, spanwise stations `[x, leading-edge z, trailing-edge z]`: a gull's
 * long, narrow, pointed wing. The arm's leading edge sweeps *forward* to the
 * wrist at `WRIST_X`, the most forward point of the wing, and its trailing edge
 * is the straight line of the secondaries; the hand is longer than the arm and
 * tapers to a point, its leading edge swept hard back and its trailing edge
 * closing in to meet it. Aspect ratio about 11.
 *
 * No notches. A first version slotted the primaries the way a crow's or a
 * raptor's are; a flying gull shows a smooth pointed tip, and with a broad
 * rounded hand the notches made the flock read as crows.
 */
const WING_STATIONS: ReadonlyArray<readonly [number, number, number]> = [
  [0.025, 0.07, -0.065],
  [0.1, 0.078, -0.078],
  [0.18, 0.086, -0.086],
  [0.25, 0.092, -0.09],
  [WRIST_X, 0.096, -0.09],
  [0.36, 0.078, -0.094],
  [0.42, 0.054, -0.096],
  [0.48, 0.026, -0.096],
  [0.54, -0.006, -0.094],
  [0.59, -0.034, -0.09],
  [0.625, -0.056, -0.086],
  [0.648, -0.072, -0.083],
  [HALF_SPAN, -0.08, -0.081],
];
/** Chord fractions the section is sampled at, leading edge to trailing. */
const WING_CHORD = [0, 0.04, 0.12, 0.25, 0.42, 0.6, 0.78, 0.92, 1];
/**
 * Section thickness and camber as fractions of chord, root and tip: a thin,
 * well-cambered bird's section, thinning out to the tip, where it closes.
 */
const WING_THICKNESS = [0.1, 0.05] as const;
const WING_CAMBER = [0.06, 0.02] as const;
const WING_CAMBER_AT = 0.4;

/**
 * NACA four-digit section, half-thickness and camber line at chord fraction `c`
 * (Jacobs, Ward and Pinkerton, NACA Report 460, 1933; the closed-trailing-edge
 * coefficient). `t` thickness and `m` camber as fractions of chord, camber
 * peak at `p`.
 */
function nacaSection(c: number, t: number, m: number, p: number): [number, number] {
  const half = 5 * t * (0.2969 * Math.sqrt(c) - 0.126 * c - 0.3516 * c * c + 0.2843 * c ** 3 - 0.1036 * c ** 4);
  const camber = c < p ? (m / (p * p)) * (2 * p * c - c * c) : (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * c - c * c);
  return [half, camber];
}

/**
 * The detailed gull, for the birds near the camera: about 1,400 triangles,
 * indexed, with smooth normals, so no facet reads at 1:1 from the taffrail.
 */
function buildDetailGull(): THREE.InstancedBufferGeometry {
  const g = new GullBuilder();

  // Body: a loft of superellipse rings, closed at the rump and the face.
  const n = BODY_SEGMENTS;
  const bodyPoint = (s: number, k: number): Vec => {
    const [z, cy, w, hu, hd] = BODY_STATIONS[s];
    const [x, y] = ringPoint(k, n, w, hu, hd, BODY_ROUNDNESS);
    return [x, cy + y, z];
  };
  const bodyCentre = (s: number): Vec => [0, BODY_STATIONS[s][1], BODY_STATIONS[s][0]];
  const last = BODY_STATIONS[BODY_STATIONS.length - 1];
  g.loft(BODY_STATIONS.length, n, bodyPoint, bodyCentre, () => [0, 0, 1, PAINT_PLUMAGE], {
    start: [0, BODY_STATIONS[0][1], BODY_STATIONS[0][0] - 0.006],
    end: [0, last[1] - 0.002, last[0] + 0.006],
  });

  // Bill: six-sided, tapering and drooping to a point.
  const billSides = 6;
  g.loft(
    BILL_STATIONS.length,
    billSides,
    (s, k) => {
      const [z, cy, w, h] = BILL_STATIONS[s];
      const [x, y] = ringPoint(k, billSides, w, h, h, 2);
      return [x, cy + y, z];
    },
    (s) => [0, BILL_STATIONS[s][1], BILL_STATIONS[s][0]],
    () => [0, 0, 1, PAINT_BILL],
    { end: BILL_TIP },
  );

  // Tail: a thin wedge, its root inside the rump, closed at the end.
  const tailSides = 10;
  const tailY = (z: number): number => 0.02 + (-0.15 - z) * 0.03;
  g.loft(
    TAIL_STATIONS.length,
    tailSides,
    (s, k) => {
      const [z, w] = TAIL_STATIONS[s];
      const [x, y] = ringPoint(k, tailSides, w, TAIL_THICKNESS, TAIL_THICKNESS, 6);
      return [x, tailY(z) + y, z];
    },
    (s) => [0, tailY(TAIL_STATIONS[s][0]), TAIL_STATIONS[s][0]],
    () => [0, 0, 0, PAINT_PLUMAGE],
    { end: [0, tailY(-0.34), -0.34] },
  );

  // Wings: each station a NACA section, upper surface leading edge to
  // trailing, then the lower surface back; thickness closes to nothing at the
  // tip. The `paint` channel carries the chord fraction (see `PAINT_PLUMAGE`).
  const chordSamples = WING_CHORD.length;
  const ring = 2 * chordSamples - 2;
  const span0 = WING_STATIONS[0][0];
  for (const side of [1, -1]) {
    const sectionAt = (s: number, k: number): { v: Vec; c: number } => {
      const [x, lead, trail] = WING_STATIONS[s];
      const f = (x - span0) / (HALF_SPAN - span0);
      const tip = s === WING_STATIONS.length - 1;
      const t = tip ? 0 : WING_THICKNESS[0] + (WING_THICKNESS[1] - WING_THICKNESS[0]) * f;
      const m = WING_CAMBER[0] + (WING_CAMBER[1] - WING_CAMBER[0]) * f;
      const upper = k < chordSamples;
      const c = WING_CHORD[upper ? k : ring - k];
      const length = lead - trail;
      const [half, camber] = nacaSection(c, t, m, WING_CAMBER_AT);
      const y = WING_Y + (camber + (upper ? half : -half)) * length;
      return { v: [side * x, y, lead - c * length], c };
    };
    g.loft(
      WING_STATIONS.length,
      ring,
      (s, k) => sectionAt(s, k).v,
      () => [0, 0, 0],
      // Paint: the chord fraction, negated on the lower surface.
      (v, s, k) => [
        v[0] / HALF_SPAN,
        Math.abs(v[0]) > WRIST_X + 1e-4 ? 1 : 0,
        0,
        k < chordSamples ? sectionAt(s, k).c : -sectionAt(s, k).c,
      ],
      {},
      // Upper-surface quads face up and lower-surface ones down; the strips
      // either side of the leading and trailing edges take their surface's.
      (_s, k) => [0, k < chordSamples - 1 ? 1 : -1, 0],
    );
  }

  // Legs: a tapering four-sided shank from the hip to the ankle, each side,
  // and a webbed foot of three toes. Folded into the belly in flight.
  for (const sx of [1, -1]) {
    const ends: Array<[Vec, number]> = [
      [[sx * 0.02, LEG_HIP[1], LEG_HIP[2]], 0.0055],
      [[sx * 0.024, LEG_FOOT_Y + 0.006, LEG_HIP[2]], 0.0035],
    ];
    g.loft(
      2,
      4,
      (s, k) => {
        const [c, r] = ends[s];
        const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
        return [c[0] + Math.cos(a) * r, c[1], c[2] + Math.sin(a) * r];
      },
      (s) => ends[s][0],
      () => [0, 0, 1, PAINT_LEG],
    );
    const heel = g.vertex([sx * 0.024, LEG_FOOT_Y, LEG_HIP[2] - 0.012], 0, 0, 1, PAINT_LEG);
    const toes = [
      [sx * 0.046, LEG_FOOT_Y, LEG_HIP[2] + 0.05],
      [sx * 0.026, LEG_FOOT_Y, LEG_HIP[2] + 0.06],
      [sx * 0.004, LEG_FOOT_Y, LEG_HIP[2] + 0.05],
    ].map((v) => g.vertex(v as Vec, 0, 0, 1, PAINT_LEG));
    g.tri(heel, toes[0], toes[1], [0, 1, 0]);
    g.tri(heel, toes[1], toes[2], [0, 1, 0]);
  }

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setIndex(g.index);
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(g.positions, 3));
  geometry.setAttribute('shape', new THREE.Float32BufferAttribute(g.shape, 4));
  geometry.computeVertexNormals();
  return geometry;
}

/** The per-bird buffers both gulls share. */
interface InstanceAttributes {
  interleaved: THREE.InstancedInterleavedBuffer;
  perchState: THREE.InstancedBufferAttribute;
}

function buildInstanceAttributes(): InstanceAttributes {
  // The six static per-bird vectors share one interleaved buffer. Separate
  // attributes were a vertex buffer each, and with the per-vertex ones and the
  // perch state that came to twelve — over WebGPU's guaranteed limit of eight,
  // which fails pipeline creation outright rather than degrading.
  const interleaved = new THREE.InstancedInterleavedBuffer(buildInstanceData(), INSTANCE_STRIDE, 1);
  const perchState = new THREE.InstancedBufferAttribute(new Float32Array(MAX_BIRDS), 1);
  perchState.setUsage(THREE.DynamicDrawUsage);
  return { interleaved, perchState };
}

/** Attaches the shared per-bird buffers to one gull's geometry. */
function withInstances(geometry: THREE.InstancedBufferGeometry, instances: InstanceAttributes): THREE.InstancedBufferGeometry {
  INSTANCE_LAYOUT.forEach((name, slot) => {
    geometry.setAttribute(name, new THREE.InterleavedBufferAttribute(instances.interleaved, 4, slot * 4));
  });
  geometry.setAttribute('perchState', instances.perchState);
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
