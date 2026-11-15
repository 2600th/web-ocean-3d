import * as THREE from 'three/webgpu';
import { attribute, exp, float, normalView, smoothstep, uniform, uv, vec2, vec3 } from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import {
  CLASS_CLUSTER,
  CLASS_DROP,
  CLASS_MIST,
  CLASS_SHEET,
  particleVertex,
  sprayMaterial,
  sprayUniforms,
} from '../physics/sprayLook';

/**
 * The two pieces of gameplay feedback that have to be visible at any range:
 * the column of light that marks the objective, and the burst when it is taken.
 *
 * Both are deliberately simple and CPU-driven. They are one object each, they
 * exist only in Helm mode, and neither has to survive a deterministic rewind
 * beyond being cleared — so the closed-form machinery `Spray` uses to make its
 * particles a function of the clock would be cost without benefit here.
 *
 * Both meshes are `frustumCulled = false`. Not for drawing — the beacon is tall
 * and the burst is tiny — but for compiling: the boot prewarm projects the
 * scene through the camera's frustum, and a mesh outside it at that moment is
 * skipped and compiled on the frame it is first seen, mid-voyage.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Height of the light column, metres. Tall enough to clear a swell at 600 m. */
const BEACON_HEIGHT = 46;
/** Base radius at close range; widened with distance so it never thins to nothing. */
const BEACON_RADIUS = 0.55;
/** Gold, in linear HDR — bright enough for the bloom to pick up. */
const BEACON_COLOR = new THREE.Color(1.0, 0.68, 0.24);

export class ObjectiveBeacon {
  readonly object = new THREE.Group();

  private readonly column: THREE.Mesh;
  private readonly ring: THREE.Mesh;
  private readonly uOpacity = uniform(1);
  private readonly uRing = uniform(0);
  private readonly materials: THREE.Material[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private time = 0;

  constructor() {
    this.object.name = 'objective-beacon';

    const columnGeometry = new THREE.CylinderGeometry(
      BEACON_RADIUS,
      BEACON_RADIUS * 1.6,
      BEACON_HEIGHT,
      16,
      1,
      true,
    );
    columnGeometry.translate(0, BEACON_HEIGHT / 2 - 1, 0);
    const column = new THREE.MeshBasicNodeMaterial();
    column.transparent = true;
    column.depthWrite = false;
    column.blending = THREE.AdditiveBlending;
    column.side = THREE.DoubleSide;
    column.fog = false;
    column.colorNode = vec3(BEACON_COLOR.r, BEACON_COLOR.g, BEACON_COLOR.b).mul(4) as any;
    // Fades up the column and toward the silhouette edges, so it reads as a
    // shaft of light rather than as a solid tube. `uv().y` is 0 at the base.
    const along = float(1).sub(uv().y).pow(1.8);
    const facing = normalView.z.abs().pow(1.2);
    column.opacityNode = along.mul(facing).mul(this.uOpacity).mul(0.85) as any;
    this.column = new THREE.Mesh(columnGeometry, column);
    this.column.frustumCulled = false;
    this.column.renderOrder = 7;
    this.column.castShadow = false;
    this.column.receiveShadow = false;

    // A ring that spreads across the water from the target, every two seconds.
    const ringGeometry = new THREE.RingGeometry(0.82, 1, 48, 1);
    ringGeometry.rotateX(-Math.PI / 2);
    const ring = new THREE.MeshBasicNodeMaterial();
    ring.transparent = true;
    ring.depthWrite = false;
    ring.blending = THREE.AdditiveBlending;
    ring.side = THREE.DoubleSide;
    ring.fog = false;
    ring.colorNode = vec3(BEACON_COLOR.r, BEACON_COLOR.g, BEACON_COLOR.b).mul(1.6) as any;
    ring.opacityNode = float(1).sub(this.uRing).mul(this.uOpacity).mul(0.7) as any;
    this.ring = new THREE.Mesh(ringGeometry, ring);
    this.ring.frustumCulled = false;
    this.ring.renderOrder = 7;

    this.materials.push(column, ring);
    this.geometries.push(columnGeometry, ringGeometry);
    this.object.add(this.column, this.ring);
  }

  /**
   * Places the beacon on the target and fades it with the viewer's distance.
   *
   * Widened with range, because a half-metre column is under a pixel past a few
   * hundred metres; faded out close in, because at arm's length a shaft of light
   * is no longer a marker, it is a wall.
   */
  update(dt: number, target: THREE.Vector3, viewer: THREE.Vector3): void {
    this.time += dt;
    this.object.position.set(target.x, target.y + 0.2, target.z);
    const distance = Math.hypot(viewer.x - target.x, viewer.z - target.z);
    const widen = Math.min(6, Math.max(1, distance / 70));
    this.column.scale.set(widen, 1, widen);
    const pulse = 0.82 + 0.18 * Math.sin(this.time * 2.6);
    const near = Math.min(1, Math.max(0, (distance - 14) / 40));
    this.uOpacity.value = pulse * (0.25 + 0.75 * near);

    const phase = (this.time / 2.2) % 1;
    this.uRing.value = phase;
    const radius = 1.5 + phase * 9;
    this.ring.scale.set(radius, 1, radius);
  }

  resetClock(): void {
    this.time = 0;
  }

  dispose(): void {
    for (const m of this.materials) m.dispose();
    for (const g of this.geometries) g.dispose();
  }
}

/** Sparks per burst. */
const BURST_COUNT = 56;
const BURST_LIFE = 1.8;
const GRAVITY = 9.81;

/**
 * Gold sparks and white water thrown up when cargo is recovered or an anchor
 * hits the water. Integrated on the CPU for the second and a half it is alive;
 * hidden and costing nothing the rest of the time.
 */
export class Burst {
  readonly mesh: THREE.InstancedMesh;

  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly geometry: THREE.BufferGeometry;
  private readonly origin = new THREE.Vector3();
  private readonly velocity: Float32Array;
  private readonly size: Float32Array;
  private readonly uFade = uniform(1);
  private readonly uTint = uniform(new THREE.Color(1, 0.75, 0.35));
  private age = BURST_LIFE;
  private seed = 1;

  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();

  constructor() {
    // Camera-facing soft discs. Solid sparks were tried first and read as
    // white blocks hanging in the air — the same lesson `Spray` records.
    this.geometry = new THREE.PlaneGeometry(1, 1);
    this.material = new THREE.MeshBasicNodeMaterial();
    this.material.transparent = true;
    this.material.depthWrite = false;
    this.material.blending = THREE.AdditiveBlending;
    this.material.side = THREE.DoubleSide;
    this.material.fog = false;
    this.material.colorNode = (this.uTint as any).mul(3);
    const radius = uv().sub(vec2(0.5, 0.5)).length().mul(2);
    const glow = float(1).sub(radius).clamp(0, 1).pow(2.2);
    this.material.opacityNode = glow.mul(this.uFade) as any;
    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, BURST_COUNT);
    this.mesh.name = 'gameplay-burst';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 7;
    this.mesh.castShadow = false;
    this.velocity = new Float32Array(BURST_COUNT * 3);
    this.size = new Float32Array(BURST_COUNT);
    // Collapsed until something fires, so the compile pass sees a real draw and
    // the frame sees nothing.
    this.matrix.makeScale(0, 0, 0);
    for (let i = 0; i < BURST_COUNT; i++) this.mesh.setMatrixAt(i, this.matrix);
  }

  get alive(): boolean {
    return this.age < BURST_LIFE;
  }

  /**
   * @param strength 0..1, scales launch speed and spark size.
   * @param gold     Gold sparks (a reward) or white water (an anchor).
   */
  fire(at: THREE.Vector3, strength = 1, gold = true): void {
    this.origin.copy(at);
    this.age = 0;
    this.uTint.value.setRGB(gold ? 1 : 0.85, gold ? 0.75 : 0.92, gold ? 0.35 : 1);
    // A deterministic spread per burst, so two identical runs throw identical sparks.
    let h = (this.seed++ * 0x9e3779b1) >>> 0;
    const next = () => {
      h ^= h << 13;
      h ^= h >>> 17;
      h ^= h << 5;
      return (h >>> 0) / 4294967296;
    };
    for (let i = 0; i < BURST_COUNT; i++) {
      const angle = next() * Math.PI * 2;
      const out = (1.5 + next() * 5) * strength;
      // High enough to clear the hull and sails from the chase camera: 4 to 13 m.
      const up = (9 + next() * 7) * (0.55 + 0.45 * strength);
      this.velocity[i * 3] = Math.cos(angle) * out;
      this.velocity[i * 3 + 1] = up;
      this.velocity[i * 3 + 2] = Math.sin(angle) * out;
      this.size[i] = (0.8 + 0.8 * next()) * (0.6 + 0.4 * strength);
    }
    this.mesh.visible = true;
  }

  update(dt: number, camera: THREE.Camera): void {
    if (!this.alive) return;
    this.age += dt;
    const t = Math.min(this.age, BURST_LIFE);
    const life = t / BURST_LIFE;
    this.uFade.value = Math.max(0, 1 - life * life);
    const shrink = 1 - life * 0.7;
    for (let i = 0; i < BURST_COUNT; i++) {
      this.position.set(
        this.origin.x + this.velocity[i * 3] * t,
        this.origin.y + this.velocity[i * 3 + 1] * t - 0.5 * GRAVITY * t * t,
        this.origin.z + this.velocity[i * 3 + 2] * t,
      );
      const s = this.size[i] * shrink;
      this.scale.set(s, s, s);
      this.matrix.compose(this.position, camera.quaternion, this.scale);
      this.mesh.setMatrixAt(i, this.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (!this.alive) this.mesh.visible = false;
  }

  clear(): void {
    this.age = BURST_LIFE;
    this.seed = 1;
    this.mesh.visible = false;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/** Bow-spray particles in flight at once: a ring, the oldest overwritten. */
const SPRAY_COUNT = 896;
/**
 * Particles per second at and above `SPRAY_FULL_SPEED`, before the plunge
 * boost. The ring holds 1.28 s of this, and the longest life (mist) is 1.25 s,
 * so a slot is never reused while its particle is in the air until a plunge
 * pushes the rate past 1.0x — and then only the oldest mist is cut short.
 */
const SPRAY_RATE = 700;
/** Hull speed, m/s, below which the bow throws nothing, and at which it throws most. */
const SPRAY_START_SPEED = 2.4;
const SPRAY_FULL_SPEED = 7;
/**
 * How far the stem has to be plunging, m/s, to double the spray. A bow driving
 * down into a sea throws far more than one riding it.
 */
const SPRAY_PLUNGE = 1.5;
/** One period of the class interleave; see `sprayLook` for what each class is. */
const SPRAY_CLASSES = [
  CLASS_DROP,
  CLASS_CLUSTER,
  CLASS_SHEET,
  CLASS_MIST,
  CLASS_CLUSTER,
  CLASS_SHEET,
  CLASS_DROP,
  CLASS_MIST,
];

/**
 * Water thrown off the stem, in proportion to the ship's speed.
 *
 * `physics/Spray` owns slam spray. This is the other half a sailing game needs:
 * the steady sheet a bow peels off at speed, which is the strongest single cue
 * that the ship is *going* somewhere.
 *
 * It was a stream of 0.7–1.6 m lit discs growing to two and a half times that,
 * which the blind review called "cotton-ball smoke puffs" fogging the planks.
 * What a bow at seven or eight knots actually throws is a continuous fan of
 * fine water off the bow wave's crest either side of the stem: streaks and
 * droplet clusters that leave with most of the hull's speed, fall back past
 * the bow within a second, and a thin mist that hangs and blows downwind. So
 * this now emits that, with the slam spray's look (`sprayLook`: velocity
 * stretch, erosion breakup, soft depth fade, sun and sky lighting with forward
 * scatter, the hull's shadow) and the slam spray's closed-form trajectory.
 *
 * **Closed form, like `Spray`.** Each particle is a launch record written once
 * at emission — where, when, how fast — into a ring of instanced attributes;
 * the vertex stage evaluates its position from the clock. The CPU never moves
 * a particle, and a deterministic run throws the same spray: emission is a
 * function of the emission count and the hull's state, and `clear` rewinds it.
 */
export class BowSpray {
  readonly mesh: THREE.Mesh;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly depthMaterial: THREE.MeshBasicNodeMaterial;
  /** xyz = launch point, w = launch time. */
  private readonly origin: THREE.InstancedBufferAttribute;
  /** xyz = launch velocity, w = strength 0..1. */
  private readonly launch: THREE.InstancedBufferAttribute;
  private readonly uClock = uniform(0);
  private time = 0;
  private cursor = 0;
  private serial = 0;
  private pending = 0;
  private dirty = false;
  private readonly previousBow = new THREE.Vector3();
  private bowPrimed = false;

  private readonly forward = new THREE.Vector3();
  private readonly starboard = new THREE.Vector3();
  private readonly bow = new THREE.Vector3();
  private readonly point = new THREE.Vector3();

  constructor() {
    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    this.geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.origin = new THREE.InstancedBufferAttribute(new Float32Array(SPRAY_COUNT * 4), 4);
    this.launch = new THREE.InstancedBufferAttribute(new Float32Array(SPRAY_COUNT * 4), 4);
    this.origin.setUsage(THREE.DynamicDrawUsage);
    this.launch.setUsage(THREE.DynamicDrawUsage);
    const seed = new Float32Array(SPRAY_COUNT * 4);
    const cls = new Float32Array(SPRAY_COUNT);
    for (let i = 0; i < SPRAY_COUNT; i++) {
      for (let c = 0; c < 4; c++) seed[i * 4 + c] = hash(i, 11 + c);
      cls[i] = SPRAY_CLASSES[i % SPRAY_CLASSES.length];
    }
    this.geometry.setAttribute('bowOrigin', this.origin);
    this.geometry.setAttribute('bowLaunch', this.launch);
    this.geometry.setAttribute('bowSeed', new THREE.InstancedBufferAttribute(seed, 4));
    this.geometry.setAttribute('bowClass', new THREE.InstancedBufferAttribute(cls, 1));
    this.geometry.instanceCount = SPRAY_COUNT;
    // Placed in the vertex stage; a bound from the unit quad would cull it all.
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

    this.material = sprayMaterial(() => this.buildVertex(), false);
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'gameplay-bow-spray';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.renderOrder = 6;

    // The dense cores again, into depth only, so the post-process fog sees the
    // spray at its own distance instead of fogging it as whatever is behind it.
    // Same fix and same reasoning as `physics/Spray`.
    this.depthMaterial = sprayMaterial(() => this.buildVertex(), true);
    const depth = new THREE.Mesh(this.geometry, this.depthMaterial);
    depth.name = 'gameplay-bow-spray-depth';
    depth.frustumCulled = false;
    depth.castShadow = false;
    depth.receiveShadow = false;
    depth.renderOrder = 5;
    this.mesh.add(depth);
    this.clear();
  }

  /**
   * @param hull     The ship's root; local +X is the bow and +Z starboard.
   * @param velocity The hull's velocity, m/s.
   * @param speed    Speed along the bow, m/s.
   */
  update(
    dt: number,
    hull: THREE.Object3D,
    velocity: THREE.Vector3,
    speed: number,
    hullLength: number,
    hullBeam: number,
    _camera?: THREE.Camera,
  ): void {
    this.time += dt;
    this.uClock.value = this.time;

    // How fast the stem is driving down, from its own track: a plunge into a
    // sea throws a burst on top of the steady fan.
    this.bow.set(hullLength * 0.5, 0, 0).applyMatrix4(hull.matrixWorld);
    let plunge = 0;
    if (this.bowPrimed && dt > 0) plunge = Math.max(0, (this.previousBow.y - this.bow.y) / dt);
    this.previousBow.copy(this.bow);
    this.bowPrimed = true;
    const boost = Math.min(1, plunge / SPRAY_PLUNGE);

    const strength = smooth01((speed - SPRAY_START_SPEED) / (SPRAY_FULL_SPEED - SPRAY_START_SPEED));
    this.pending += SPRAY_RATE * strength * (0.7 + 0.8 * boost) * dt;
    if (this.pending >= 1) {
      this.forward.set(1, 0, 0).applyQuaternion(hull.quaternion).setY(0);
      if (this.forward.lengthSq() < 1e-6) this.forward.set(1, 0, 0);
      this.forward.normalize();
      this.starboard.set(-this.forward.z, 0, this.forward.x);
    }
    const effective = Math.min(1, strength * (0.8 + 0.4 * boost));
    while (this.pending >= 1) {
      this.pending -= 1;
      this.emit(hull, velocity, effective, hullLength, hullBeam);
    }
    if (this.dirty) {
      this.origin.needsUpdate = true;
      this.launch.needsUpdate = true;
      this.dirty = false;
    }
  }

  clear(): void {
    (this.origin.array as Float32Array).fill(0);
    (this.launch.array as Float32Array).fill(0);
    for (let i = 0; i < SPRAY_COUNT; i++) this.origin.setW(i, -1000);
    this.origin.needsUpdate = true;
    this.launch.needsUpdate = true;
    this.time = 0;
    this.uClock.value = 0;
    this.cursor = 0;
    this.serial = 0;
    this.pending = 0;
    this.dirty = false;
    this.bowPrimed = false;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.depthMaterial.dispose();
  }

  private emit(
    hull: THREE.Object3D,
    velocity: THREE.Vector3,
    strength: number,
    hullLength: number,
    hullBeam: number,
  ): void {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % SPRAY_COUNT;
    const n = this.serial++;
    const r1 = hash(n, 1);
    const r2 = hash(n, 2);
    const r3 = hash(n, 3);
    const r4 = hash(n, 4);
    const cls = SPRAY_CLASSES[i % SPRAY_CLASSES.length];
    // Alternate sides of the stem. Along the forward fifth of the hull, on the
    // bow wave's crest: the hull narrows toward the stem, so the crest's offset
    // from the centreline does too, and the wave climbs the stem as speed rises.
    const sideSign = n % 2 === 0 ? 1 : -1;
    const along = r1;
    this.point
      .set(
        hullLength * (0.38 + 0.1 * along),
        -0.1 + (0.3 + 0.7 * r2) * strength,
        // The crest stands half a metre or so off the planking.
        sideSign * (hullBeam * 0.5 * (1 - 0.8 * along) + 0.35 + 0.35 * r3),
      )
      .applyMatrix4(hull.matrixWorld);

    // Out from the hull and up, at speeds from bow-spray footage of ships of
    // this size at 6–8 knots: the fan clears the rail by a metre or two. Mist
    // leaves slower and hangs; a sheet is the crest itself, lower and heavier.
    const lift = cls === CLASS_MIST ? 0.6 : cls === CLASS_SHEET ? 0.75 : 1;
    const out = (2.5 + 3 * r3) * (0.45 + 0.55 * strength) * lift;
    const up = (2.5 + 3.8 * r4) * (0.4 + 0.6 * strength) * lift;
    const fwd = 0.4 * (r3 - 0.3);
    // Carried with the hull, mostly: spray leaves a moving bow at the bow's
    // speed plus its own, and the drag then hands it to the wind — which is
    // what makes the fan stream aft along the topsides.
    this.origin.setXYZW(i, this.point.x, this.point.y, this.point.z, this.time);
    this.launch.setXYZW(
      i,
      velocity.x * 0.85 + this.starboard.x * sideSign * out + this.forward.x * fwd,
      up,
      velocity.z * 0.85 + this.starboard.z * sideSign * out + this.forward.z * fwd,
      strength,
    );
    this.dirty = true;
  }

  /** Trajectory, size and opacity; `particleVertex` does the rest. */
  private buildVertex(): any {
    const origin: any = attribute('bowOrigin', 'vec4');
    const launch: any = attribute('bowLaunch', 'vec4');
    const seed: any = attribute('bowSeed', 'vec4');
    const cls: any = attribute('bowClass', 'float');

    const isDrop: any = smoothstepDown(cls, 0.4, 0.5).toVar('bowDrop');
    const isSheet: any = smoothstep(0.5, 0.6, cls).mul(smoothstepDown(cls, 1.4, 1.5)).toVar('bowSheet');
    const isMist: any = smoothstep(1.5, 1.6, cls).mul(smoothstepDown(cls, 2.4, 2.5)).toVar('bowMist');
    const isCluster: any = smoothstep(2.5, 2.6, cls).toVar('bowCluster');

    const strength: any = launch.w;
    const lifeSpan: any = isDrop
      .mul(seed.w.mul(0.45).add(0.55))
      .add(isCluster.mul(seed.w.mul(0.4).add(0.6)))
      .add(isSheet.mul(seed.w.mul(0.35).add(0.45)))
      .add(isMist.mul(seed.w.mul(0.35).add(0.9)))
      .toVar('bowLifeSpan');
    const age: any = this.uClock.sub(origin.w).toVar('bowAge');
    const life: any = age.div(lifeSpan).toVar('bowLife');
    const alive: any = life
      .greaterThanEqual(0)
      .and(life.lessThanEqual(1))
      .and(strength.greaterThan(0))
      .select(float(1), float(0))
      .toVar('bowAlive');

    // Same linear drag toward the wind as `Spray`; see its header.
    const k: any = isDrop.mul(0.6).add(isCluster.mul(0.8)).add(isSheet.mul(1.6)).add(isMist.mul(2.6)).toVar('bowK');
    const wind: any = sprayUniforms.wind;
    const vInf: any = vec3(wind.x, float(-GRAVITY).div(k), wind.z).toVar('bowVInf');
    const decay: any = exp(k.negate().mul(age.max(0))).toVar('bowDecay');
    const v0: any = launch.xyz;
    const centre: any = origin.xyz
      .add(vInf.mul(age))
      .add(v0.sub(vInf).mul(decay.oneMinus().div(k)))
      .toVar('bowCentre');
    const velocity: any = vInf.add(v0.sub(vInf).mul(decay)).toVar('bowVelocity');

    const size: any = isDrop
      .mul(seed.y.mul(0.03).add(0.018))
      .add(isCluster.mul(seed.y.mul(0.4).add(0.3)))
      .add(isSheet.mul(seed.y.mul(0.5).add(0.55)))
      .add(isMist.mul(seed.y.mul(0.9).add(0.9)))
      .mul(strength.mul(0.4).add(0.6));
    const grow: any = isDrop
      .add(isCluster.mul(life.mul(0.6).add(0.8)))
      .add(isSheet.mul(life.mul(0.9).add(0.7)))
      .add(isMist.mul(life.mul(1.6).add(0.5)));
    const width: any = size.mul(grow).toVar('bowWidth');

    const density: any = isDrop
      .mul(0.8)
      .add(isCluster.mul(0.85))
      .add(isSheet.mul(life.mul(-0.3).add(0.9)))
      // The white haze over the bow wave, which is what reads from a chase
      // camera when the individual drops are under a pixel. At 0.07 a mist
      // quad was invisible past twenty metres (boat-sea-1: "the bow spray is
      // faint from the chase camera").
      .add(isMist.mul(0.2));
    const fadeIn: any = smoothstep(0, 0.06, life);
    const fadeOut: any = smoothstepDown(life, isMist.mul(-0.35).add(0.7), 1);
    // Back into the sea a little below where it left the crest.
    const submerged: any = smoothstepDown(origin.y.sub(0.3).sub(centre.y), 0.05, 0.4);
    const opacity: any = density.mul(fadeIn).mul(fadeOut).mul(submerged).mul(strength.sqrt());

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
      streak: isDrop.mul(0.035).add(isCluster.mul(0.03)).add(isSheet.mul(0.02)).add(isMist.mul(0.04)),
      opacity,
      soft: isDrop.mul(0.08).add(isCluster.mul(0.3)).add(isSheet.mul(0.5)).add(isMist.mul(1.2)),
      range: 260,
    });
  }
}

function smooth01(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/** Integer hash to [0,1), so emission depends on the emission count alone. */
function hash(a: number, b: number): number {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x165667b1, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 15), 0x27d4eb2f);
  return ((h ^ (h >>> 13)) >>> 0) / 4294967296;
}
