import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import { ISLAND, seafloorHeight } from '../Seafloor';

/**
 * Beachrock: where it lies, as data, for the island's mesh (`IslandDetail`)
 * and for the sea's surf (`OceanMaterial`), which breaks around it.
 *
 * Caribbean beaches commonly have it: sand cemented by carbonate where the
 * water table meets the swash, exposed as flat, seaward-dipping slabs a few
 * metres long, and broken blocks of older beachrock just offshore. It is also
 * what the surf at beach-surf had nothing of to break around: every bore
 * reached the sand as an unbroken parallel line (foam-4, every round).
 *
 * The sea cannot learn where a rock is from the depth buffer alone: that sees
 * only the water in front of a rock's submerged face, a ring a few pixels
 * wide. White water piled on the seaward side, wrapping round, and the gap in
 * the lee are metres across and on every side, so the surf reads the
 * footprints from here (`beachrockFootprints`).
 */

/**
 * Seed of the beachrock's own random stream. Separate from the island
 * detail's, so adding it moved none of the placements drawn after it.
 */
const BEACHROCK_SEED = 0xbeac4;

/**
 * Bearings from the island centre, radians, on the open beach west of the
 * cove. The slabs and heads sit in the left of the beach-surf view (60-85 m
 * out) and outside the beach-swash and shore-break frames.
 */
const SLABS = { from: 0.778, to: 0.788 };
const HEADS = { from: 0.765, to: 0.768 };
/**
 * A reef of older blocks out in the surf, right of centre in both beach views,
 * where the bores break on it. Both reference frames (beach-surf and
 * beach-swash) have their white water working around rocks there.
 */
const REEF = { from: 0.874, to: 0.886, out: { from: 16, to: 24 } };

const UP = new THREE.Vector3(0, 1, 0);

/** Bisects for the shoreline radius on a bearing. */
export function shoreRadius(bearing: number): number {
  const dx = Math.cos(bearing);
  const dz = Math.sin(bearing);
  let inner = ISLAND.radius * 0.4;
  let outer = ISLAND.radius * 1.7;
  for (let i = 0; i < 22; i++) {
    const mid = (inner + outer) / 2;
    if (seafloorHeight(ISLAND.x + dx * mid, ISLAND.z + dz * mid) > 0) inner = mid;
    else outer = mid;
  }
  return (inner + outer) / 2;
}

/** Yaw that turns local +z toward world (dx, dz). */
export function yawToward(dx: number, dz: number): number {
  return Math.atan2(dx, dz);
}

export interface BeachrockPiece {
  kind: 'slab' | 'head';
  /** Which of that kind's shapes. */
  shape: number;
  matrix: THREE.Matrix4;
}

/**
 * A piece's cut through still water, as an ellipse: centre, the unit vector of
 * its first axis, and its two radii, metres.
 */
export interface BeachrockFootprint {
  x: number;
  z: number;
  ax: number;
  az: number;
  ra: number;
  rb: number;
}

let plan: { pieces: BeachrockPiece[]; footprints: BeachrockFootprint[] } | null = null;

/** Shapes are unit spheres scaled by these radii before the piece's own scale (see `IslandDetail.beachrock`). */
export const BEACHROCK_SLAB_SIZE: [number, number, number] = [1, 0.22, 0.55];
export const BEACHROCK_HEAD_SIZE: [number, number, number] = [1, 0.62, 0.8];

function build(): { pieces: BeachrockPiece[]; footprints: BeachrockFootprint[] } {
  const rng = mulberry32(BEACHROCK_SEED);
  const pieces: BeachrockPiece[] = [];
  const footprints: BeachrockFootprint[] = [];
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const seat = (bearing: number, radius: number) => {
    const x = ISLAND.x + Math.cos(bearing) * radius;
    const z = ISLAND.z + Math.sin(bearing) * radius;
    return { x, z, y: seafloorHeight(x, z) };
  };
  const add = (kind: BeachrockPiece['kind'], shape: number, centre: THREE.Vector3, size: [number, number, number]) => {
    const matrix = new THREE.Matrix4().compose(centre, q, s);
    pieces.push({ kind, shape, matrix });
    // Where the ellipsoid meets y = 0 (taken as upright: the slabs' dip is a
    // few degrees). Under water it still shoals the bores, so it keeps a
    // third of its radius there.
    const ry = size[1] * s.y;
    const h = -centre.y / ry;
    const cut = Math.max(0.35, Math.sqrt(Math.max(0, 1 - h * h)));
    const a = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    a.y = 0;
    a.normalize();
    footprints.push({ x: centre.x, z: centre.z, ax: a.x, az: a.z, ra: size[0] * s.x * cut, rb: size[2] * s.z * cut });
  };

  // Slabs: end to end along the shore with gaps, long axis along it, tipped
  // a few degrees seaward, from just above the still water to a few metres
  // out, so the swash runs over some and the bores break on the outer ones.
  const slabs = 6;
  for (let i = 0; i < slabs; i++) {
    const t = (i + 0.2 + rng() * 0.6) / slabs;
    const bearing = SLABS.from + (SLABS.to - SLABS.from) * t;
    const shore = shoreRadius(bearing);
    const p = seat(bearing, shore - 1.5 + rng() * 4);
    const length = 2.4 + rng() * 2.6;
    const tangent = bearing + Math.PI / 2;
    q.setFromAxisAngle(UP, yawToward(Math.cos(tangent), Math.sin(tangent)) - Math.PI / 2 + (rng() - 0.5) * 0.4);
    const seaward = new THREE.Vector3(Math.cos(bearing), 0, Math.sin(bearing));
    const dipAxis = new THREE.Vector3().crossVectors(UP, seaward).normalize();
    q.premultiply(new THREE.Quaternion().setFromAxisAngle(dipAxis, 0.06 + rng() * 0.08));
    s.set(length, length * (0.75 + rng() * 0.5), length * (0.7 + rng() * 0.4));
    add('slab', i % 3, new THREE.Vector3(p.x, p.y - length * 0.05, p.z), BEACHROCK_SLAB_SIZE);
  }
  // Heads: a few broken blocks standing half a metre to a metre clear of the
  // still water in the inner surf.
  const heads = 3;
  for (let i = 0; i < heads; i++) {
    const t = (i + 0.25 + rng() * 0.5) / heads;
    const bearing = HEADS.from + (HEADS.to - HEADS.from) * t;
    const shore = shoreRadius(bearing);
    const p = seat(bearing, shore + 7 + rng() * 2);
    const k = 2 + rng() * 1.3;
    const rise = 0.5 + rng() * 0.5;
    // Tall enough to clear the still water by `rise` from wherever its base sits.
    // The top of the shape is 0.62 of its height above its centre, and the
    // centre is sunk a fifth of the height: 0.42 of the height above the bed.
    const tall = Math.max(k * 0.6, (rise - p.y) / 0.42);
    q.setFromAxisAngle(UP, rng() * Math.PI * 2);
    s.set(k, tall, k * (0.8 + rng() * 0.4));
    add('head', i % 2, new THREE.Vector3(p.x, p.y - tall * 0.2, p.z), BEACHROCK_HEAD_SIZE);
  }
  // Reef: broader blocks further out, standing a metre or so clear of the
  // still water, drawn after the rest so the slabs and heads did not move.
  const reef = 4;
  for (let i = 0; i < reef; i++) {
    const t = (i + 0.15 + rng() * 0.7) / reef;
    const bearing = REEF.from + (REEF.to - REEF.from) * t;
    const shore = shoreRadius(bearing);
    const p = seat(bearing, shore + REEF.out.from + rng() * (REEF.out.to - REEF.out.from));
    const k = 2.4 + rng() * 1.8;
    const rise = 0.6 + rng() * 0.8;
    const tall = Math.max(k * 0.7, (rise - p.y) / 0.42);
    q.setFromAxisAngle(UP, rng() * Math.PI * 2);
    s.set(k, tall, k * (0.75 + rng() * 0.4));
    add('head', i % 2, new THREE.Vector3(p.x, p.y - tall * 0.2, p.z), BEACHROCK_HEAD_SIZE);
  }
  return { pieces, footprints };
}

/** Every beachrock piece and its transform. Pure; computed once. */
export function beachrockPieces(): readonly BeachrockPiece[] {
  plan ??= build();
  return plan.pieces;
}

/** Every piece's waterline ellipse, for the surf. Pure; computed once. */
export function beachrockFootprints(): readonly BeachrockFootprint[] {
  plan ??= build();
  return plan.footprints;
}
