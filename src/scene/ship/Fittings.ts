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
 * drawn while the anchor is down and hauled in while the capstan works.
 *
 * A slack rode is a catenary that comes down from the hawse and meets the
 * bottom tangentially, with the rest lying along the bottom to the anchor —
 * the scope. For a catenary of parameter `a` touching down `h` below the
 * hawse, the hanging part spans `a·acosh(1 + h/a)` and is `sqrt(h² + 2ah)`
 * long, so `a` is found (bisection) such that span plus the length lying on
 * the bottom reaches the anchor. In shallow water there is more rode than
 * that can use; the surplus would pile on the bottom, out of sight, so `a` is
 * held at a floor of a fraction of `h`, which keeps the cable leaving the
 * hawse at a slant toward the anchor instead of plumb down like a pole. When
 * the rode is too short to reach the bottom slack (snubbed, or weighing), it
 * is drawn as a parabola through both ends with the sag its slack calls for:
 * the arc length of a shallow parabola is `c + 8s²/3c`, so the slack `L - c`
 * gives the sag `s = sqrt(3c(L - c)/8)`. Whatever would pass below the seabed
 * lies along it.
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
/**
 * A 9 cm hemp cable. The 20 cm one read as a spar: at this size it is a rope
 * from the deck and a thin line from the chase camera.
 */
const RADIUS = 0.045;
/** Rings on the hanging part; the rest lie along the bottom. */
const HANGING_RINGS = 30;
/** Least catenary parameter, as a fraction of the hawse's height above the bottom. */
const MIN_CATENARY = 0.7;

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
    const span = Math.hypot(end.x - top.x, end.z - top.z);
    const drop = top.y - end.y;
    const a = span > 0.5 && drop > 0.5 ? touchdownCatenary(drop, span, length) : null;

    if (a !== null) {
      // Hanging from the hawse to the touchdown, then along the bottom.
      const reach = a * Math.acosh(1 + drop / a);
      for (let i = 0; i < RINGS; i++) {
        const u = i < HANGING_RINGS
          ? reach * (i / (HANGING_RINGS - 1))
          : reach + (span - reach) * ((i - HANGING_RINGS + 1) / (RINGS - HANGING_RINGS));
        const p = this.centre[i].lerpVectors(top, end, u / span);
        p.y = u < reach ? end.y + a * (Math.cosh((reach - u) / a) - 1) : end.y;
        p.y = Math.max(p.y, this.floor(p.x, p.z) + RADIUS);
      }
    } else {
      const sag = chord > 1e-3 ? Math.sqrt((3 * chord * Math.max(0, length - chord)) / 8) : 0;
      for (let i = 0; i < RINGS; i++) {
        const t = i / (RINGS - 1);
        const p = this.centre[i].lerpVectors(top, end, t);
        p.y -= sag * 4 * t * (1 - t);
        // Lies along the bottom rather than through it.
        p.y = Math.max(p.y, this.floor(p.x, p.z) + RADIUS);
      }
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

/**
 * Catenary parameter for a rode of `length` hanging `drop` metres from the
 * hawse to a bottom it meets tangentially, its anchor `span` metres off
 * horizontally; null when the rode cannot reach the bottom slack.
 */
function touchdownCatenary(drop: number, span: number, length: number): number | null {
  const reachOf = (a: number): number => a * Math.acosh(1 + drop / a);
  // Span plus what lies on the bottom, less the distance to the anchor: falls as `a` grows.
  const excess = (a: number): number => reachOf(a) + length - Math.sqrt(drop * drop + 2 * a * drop) - span;
  if (length - drop - span <= 0) return null;
  // At the floor with rode to spare, the true curve is flatter: find it. With
  // none to spare, the surplus piles on the bottom and the floor stands.
  const floor = MIN_CATENARY * drop;
  let a = floor;
  if (excess(floor) > 0) {
    let lo = floor;
    let hi = floor * 2;
    for (let k = 0; k < 30 && excess(hi) > 0; k++) hi *= 2;
    for (let k = 0; k < 40; k++) {
      const mid = 0.5 * (lo + hi);
      if (excess(mid) > 0) lo = mid;
      else hi = mid;
    }
    a = lo;
  }
  // The touchdown must stay short of the anchor; flatten no further than that.
  if (reachOf(a) > 0.9 * span) {
    let lo = 1e-3;
    let hi = a;
    for (let k = 0; k < 40; k++) {
      const mid = 0.5 * (lo + hi);
      if (reachOf(mid) > 0.9 * span) hi = mid;
      else lo = mid;
    }
    a = Math.max(lo, 1e-3);
  }
  return a;
}

const UP = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const _top = /*@__PURE__*/ new THREE.Vector3();
const _end = /*@__PURE__*/ new THREE.Vector3();
const _t = /*@__PURE__*/ new THREE.Vector3();
const _s = /*@__PURE__*/ new THREE.Vector3();
const _u = /*@__PURE__*/ new THREE.Vector3();
