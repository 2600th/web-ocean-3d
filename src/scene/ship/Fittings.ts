import * as THREE from 'three/webgpu';

/**
 * The ship's rigid fittings that answer to gameplay. For now, the anchor cable.
 *
 * The wheel and the rudder are not here because the model cannot carry them:
 * the Poly Haven hull is a single joined mesh (`x-Boats-Dutch-medium-baseB-joined`)
 * with the rudder and the deck furniture baked into it, so there is no node to
 * turn. Doing it properly needs the rudder and the wheel split out of the hull
 * in the asset pipeline.
 *
 * ### The cable
 *
 * A hemp rode from the hawse on the anchor's bow down to where the anchor lies,
 * drawn while the anchor is down and hauled in while the capstan works. Its
 * shape is a parabola through the two ends with the sag the rode's slack calls
 * for — the arc length of a shallow parabola is `c + 8s²/3c`, so the slack
 * `L - c` gives the sag `s = sqrt(3c(L - c)/8)` — and whatever would pass below
 * the seabed lies along it instead, which is what a rode does. A catenary would
 * be the exact curve; at these spans the parabola is within a few centimetres
 * of it and costs nothing to evaluate.
 *
 * It is a 6-sided tube of 40 rings rewritten on the CPU each frame it shows
 * (240 vertices), in world space, one draw, no shadow. While the anchor is up
 * it is collapsed to a point and the CPU does nothing.
 */

/** Per-frame anchor state from `Gameplay.anchorCable`. */
export interface AnchorCableState {
  x: number;
  z: number;
  /** +1 starboard bower, -1 port. */
  side: number;
  rode: number;
  /** 0 while lying to it, rising to 1 as the capstan takes it in. */
  hauled: number;
}

const RINGS = 40;
const SIDES = 6;
/** A 20 cm hemp cable reads as a line at any distance it is seen from. */
const RADIUS = 0.1;

export class ShipFittings {
  readonly object: THREE.Mesh;
  /** Scratch the caller passes to `Gameplay.anchorCable`. */
  readonly anchorState: AnchorCableState = { x: 0, z: 0, side: 1, rode: 0, hauled: 0 };

  private readonly ship: THREE.Object3D;
  private readonly floor: (x: number, z: number) => number;
  /** The hawse in the ship's frame for a starboard anchor; z flips for port. */
  private readonly hawse: THREE.Vector3;
  private readonly positions: THREE.BufferAttribute;
  private readonly normals: THREE.BufferAttribute;
  private readonly centre: THREE.Vector3[] = [];
  private collapsed = false;

  constructor(ship: THREE.Object3D, hullLength: number, hullBeam: number, floor: (x: number, z: number) => number) {
    this.ship = ship;
    this.floor = floor;
    // `Gameplay` pulls from hullLength * 0.45 forward. The bow is fine there, so
    // the hawse holes sit just off the stem, about 2.4 m above the waterline.
    this.hawse = new THREE.Vector3(hullLength * 0.4, 2.4, hullBeam * 0.2);
    for (let i = 0; i < RINGS; i++) this.centre.push(new THREE.Vector3());

    const geometry = new THREE.BufferGeometry();
    this.positions = new THREE.BufferAttribute(new Float32Array(RINGS * SIDES * 3), 3);
    this.normals = new THREE.BufferAttribute(new Float32Array(RINGS * SIDES * 3), 3);
    this.positions.setUsage(THREE.DynamicDrawUsage);
    this.normals.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.positions);
    geometry.setAttribute('normal', this.normals);
    const index: number[] = [];
    for (let r = 0; r < RINGS - 1; r++) {
      for (let s = 0; s < SIDES; s++) {
        const a = r * SIDES + s;
        const b = r * SIDES + ((s + 1) % SIDES);
        index.push(a, b, a + SIDES, b, b + SIDES, a + SIDES);
      }
    }
    geometry.setIndex(index);

    const material = new THREE.MeshStandardNodeMaterial({ color: 0x4a3b2a, roughness: 0.85 });
    material.name = 'ship-anchor-cable';
    this.object = new THREE.Mesh(geometry, material);
    this.object.name = 'ship-anchor-cable';
    this.object.castShadow = false;
    this.object.receiveShadow = true;
    // World-space vertices: the bounds move with the hull, so cull nothing.
    this.object.frustumCulled = false;
    this.collapse();
  }

  /** @param anchor `Gameplay.anchorCable(fittings.anchorState)`; null while the anchor is up. */
  update(anchor: AnchorCableState | null): void {
    if (!anchor || anchor.hauled >= 1) {
      this.collapse();
      return;
    }
    this.collapsed = false;
    this.ship.updateWorldMatrix(true, false);
    const top = _top.set(this.hawse.x, this.hawse.y, this.hawse.z * Math.sign(anchor.side || 1)).applyMatrix4(this.ship.matrixWorld);
    const end = _end.set(anchor.x, this.floor(anchor.x, anchor.z), anchor.z);
    // Hauling: the anchor end comes up to the hawse as the capstan turns.
    end.lerp(top, THREE.MathUtils.clamp(anchor.hauled, 0, 1));
    const chord = top.distanceTo(end);
    const length = Math.max(chord, anchor.rode * (1 - anchor.hauled));
    const sag = chord > 1e-3 ? Math.sqrt((3 * chord * Math.max(0, length - chord)) / 8) : 0;

    for (let i = 0; i < RINGS; i++) {
      const t = i / (RINGS - 1);
      const p = this.centre[i].lerpVectors(top, end, t);
      p.y -= sag * 4 * t * (1 - t);
      // Lies along the bottom rather than through it.
      p.y = Math.max(p.y, this.floor(p.x, p.z) + RADIUS);
    }

    const pos = this.positions.array as Float32Array;
    const nor = this.normals.array as Float32Array;
    for (let i = 0; i < RINGS; i++) {
      const prev = this.centre[Math.max(0, i - 1)];
      const next = this.centre[Math.min(RINGS - 1, i + 1)];
      const tangent = _t.subVectors(next, prev).normalize();
      // A frame around the tangent; the cable is round, so its roll is free.
      const side = _s.crossVectors(tangent, UP);
      if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
      side.normalize();
      const up = _u.crossVectors(side, tangent).normalize();
      for (let s = 0; s < SIDES; s++) {
        const a = (s / SIDES) * Math.PI * 2;
        const nx = side.x * Math.cos(a) + up.x * Math.sin(a);
        const ny = side.y * Math.cos(a) + up.y * Math.sin(a);
        const nz = side.z * Math.cos(a) + up.z * Math.sin(a);
        const k = (i * SIDES + s) * 3;
        const c = this.centre[i];
        pos[k] = c.x + nx * RADIUS;
        pos[k + 1] = c.y + ny * RADIUS;
        pos[k + 2] = c.z + nz * RADIUS;
        nor[k] = nx;
        nor[k + 1] = ny;
        nor[k + 2] = nz;
      }
    }
    this.positions.needsUpdate = true;
    this.normals.needsUpdate = true;
  }

  /**
   * Anchor up: every vertex at one point far below the sea, so the tube draws
   * nothing. Kept drawn rather than hidden so its pipeline is compiled with the
   * ship at boot and letting go the anchor never stalls a frame on a compile.
   */
  private collapse(): void {
    if (this.collapsed) return;
    this.collapsed = true;
    (this.positions.array as Float32Array).fill(0);
    for (let k = 1; k < this.positions.array.length; k += 3) (this.positions.array as Float32Array)[k] = -10000;
    this.positions.needsUpdate = true;
  }

  dispose(): void {
    this.object.removeFromParent();
    this.object.geometry.dispose();
    (this.object.material as THREE.Material).dispose();
  }
}

const UP = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const _top = /*@__PURE__*/ new THREE.Vector3();
const _end = /*@__PURE__*/ new THREE.Vector3();
const _t = /*@__PURE__*/ new THREE.Vector3();
const _s = /*@__PURE__*/ new THREE.Vector3();
const _u = /*@__PURE__*/ new THREE.Vector3();
