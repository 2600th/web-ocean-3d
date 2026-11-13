import * as THREE from 'three/webgpu';
import { attribute, float, normalView, uniform, uv, vec2, vec3 } from 'three/tsl';

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

/** Bow-spray particles in flight at once. */
const SPRAY_COUNT = 180;
/** Particles per second at and above `SPRAY_FULL_SPEED`. */
const SPRAY_RATE = 140;
/** Hull speed, m/s, below which the bow throws nothing, and at which it throws most. */
const SPRAY_START_SPEED = 2.4;
const SPRAY_FULL_SPEED = 7;
/** Floats of state per particle: position, velocity, age, life, size. */
const STRIDE = 9;

/**
 * Water thrown off the stem, in proportion to the ship's speed.
 *
 * `physics/Spray` owns slam spray — and is parked, drawing nothing, for the
 * reasons `tests/spray.spec.ts` records. This is the other half a sailing game
 * needs and which does not wait on that fix: the steady sheet a bow peels off at
 * speed, which is the strongest single cue that the ship is *going* somewhere.
 * CPU-integrated and billboarded, because there are never more than a couple of
 * hundred quads and each needs a camera-facing matrix anyway.
 *
 * Lit rather than unlit, so it is white water at noon and grey at night instead
 * of a glow. Emission is a function of the emission count alone and `clear`
 * rewinds it, so a deterministic run throws the same spray.
 */
export class BowSpray {
  readonly mesh: THREE.InstancedMesh;

  private readonly geometry: THREE.PlaneGeometry;
  private readonly material: THREE.MeshStandardNodeMaterial;
  private readonly depthMaterial: THREE.MeshBasicNodeMaterial;
  private readonly fade: THREE.InstancedBufferAttribute;
  private readonly state = new Float32Array(SPRAY_COUNT * STRIDE);
  private cursor = 0;
  private serial = 0;
  private pending = 0;

  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly local = new THREE.Vector3();
  private readonly side = new THREE.Vector3();

  constructor() {
    this.geometry = new THREE.PlaneGeometry(1, 1);
    this.fade = new THREE.InstancedBufferAttribute(new Float32Array(SPRAY_COUNT), 1);
    this.fade.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('sprayFade', this.fade);

    this.material = new THREE.MeshStandardNodeMaterial({ color: 0xf0f6ff, roughness: 0.95 });
    this.material.transparent = true;
    this.material.depthWrite = false;
    this.material.side = THREE.DoubleSide;
    this.material.fog = false;
    // A soft disc: no square corners, and a denser core.
    const radius = uv().sub(vec2(0.5, 0.5)).length().mul(2);
    const disc = float(1).sub(radius).clamp(0, 1).pow(1.6);
    this.material.opacityNode = disc.mul(attribute('sprayFade', 'float')).mul(0.6) as any;

    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, SPRAY_COUNT);
    this.mesh.name = 'gameplay-bow-spray';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.renderOrder = 6;

    // The dense cores again, into depth only, so the post-process fog sees the
    // spray at its own distance instead of fogging it as whatever is behind it
    // — in a storm, that was all of it. Same fix and same reasoning as
    // `physics/Spray`; shares this mesh's instance matrices.
    this.depthMaterial = new THREE.MeshBasicNodeMaterial();
    this.depthMaterial.transparent = true;
    this.depthMaterial.depthWrite = true;
    this.depthMaterial.colorWrite = false;
    this.depthMaterial.side = THREE.DoubleSide;
    this.depthMaterial.fog = false;
    this.depthMaterial.alphaTest = 0.3;
    this.depthMaterial.opacityNode = this.material.opacityNode;
    const depth = new THREE.InstancedMesh(this.geometry, this.depthMaterial, SPRAY_COUNT);
    depth.instanceMatrix = this.mesh.instanceMatrix;
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
    camera: THREE.Camera,
  ): void {
    const strength = smooth01((speed - SPRAY_START_SPEED) / (SPRAY_FULL_SPEED - SPRAY_START_SPEED));
    this.pending += SPRAY_RATE * strength * dt;
    while (this.pending >= 1) {
      this.pending -= 1;
      this.emit(hull, velocity, strength, hullLength, hullBeam);
    }

    const s = this.state;
    const q = camera.quaternion;
    for (let i = 0; i < SPRAY_COUNT; i++) {
      const o = i * STRIDE;
      const life = s[o + 7];
      let age = s[o + 6];
      if (age >= life) {
        this.fade.setX(i, 0);
        continue;
      }
      age += dt;
      s[o + 6] = age;
      s[o + 4] -= GRAVITY * dt;
      s[o] += s[o + 3] * dt;
      s[o + 1] += s[o + 4] * dt;
      s[o + 2] += s[o + 5] * dt;
      const t = Math.min(1, age / life);
      // Grows as it disperses; fades in fast and out over the back two thirds.
      const size = s[o + 8] * (0.5 + t * 1.3);
      this.position.set(s[o], s[o + 1], s[o + 2]);
      this.scale.set(size, size, size);
      this.matrix.compose(this.position, q, this.scale);
      this.mesh.setMatrixAt(i, this.matrix);
      this.fade.setX(i, Math.min(1, t * 6) * (1 - smooth01((t - 0.3) / 0.7)));
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.fade.needsUpdate = true;
  }

  clear(): void {
    this.state.fill(0);
    this.cursor = 0;
    this.serial = 0;
    this.pending = 0;
    this.matrix.makeScale(0, 0, 0);
    for (let i = 0; i < SPRAY_COUNT; i++) {
      this.mesh.setMatrixAt(i, this.matrix);
      this.fade.setX(i, 0);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.fade.needsUpdate = true;
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
    // Alternate sides of the stem; each sheet leaves its bow outward and aft.
    const sideSign = n % 2 === 0 ? 1 : -1;
    this.local.set(hullLength * (0.4 + 0.06 * r1), 0.4, sideSign * hullBeam * (0.12 + 0.1 * r2));
    this.position.copy(this.local).applyQuaternion(hull.quaternion).add(hull.position);
    this.side.set(-0.3 - 0.4 * r3, 0, sideSign).applyQuaternion(hull.quaternion);
    this.side.y = 0;
    this.side.normalize();

    const out = (2 + 3 * r3) * (0.5 + 0.5 * strength);
    const up = (2.5 + 3.5 * r4) * (0.5 + 0.5 * strength);
    const s = this.state;
    const o = i * STRIDE;
    s[o] = this.position.x;
    s[o + 1] = this.position.y;
    s[o + 2] = this.position.z;
    // Carried with the hull, mostly: spray leaves a moving bow at the bow's
    // speed plus its own.
    s[o + 3] = velocity.x * 0.85 + this.side.x * out;
    s[o + 4] = up;
    s[o + 5] = velocity.z * 0.85 + this.side.z * out;
    s[o + 6] = 0;
    s[o + 7] = 0.6 + 0.5 * r1;
    s[o + 8] = (0.7 + 0.9 * r2) * (0.6 + 0.4 * strength);
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
