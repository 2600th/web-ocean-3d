import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import { ISLAND, seafloorHeight } from '../Seafloor';

/**
 * Beachrock: where it lies and what shape it is, for the island's mesh
 * (`IslandDetail`) and for the sea's surf (`OceanMaterial`), which breaks
 * around it.
 *
 * Caribbean beaches commonly have it: sand cemented by carbonate where the
 * water table meets the swash, exposed as low, flat-topped, seaward-dipping
 * slabs a few metres long, and broken blocks of older beachrock just offshore.
 * The sea dissolves and batters it into a recognisable shape: flat tops and
 * ledges where the cemented beds part, an undercut notch at the still-water
 * line with an overhanging lip above it, and joints that split it into
 * blocks. So it is built as that, not as a boulder: an irregular outline
 * extruded through a stack of rings, steps where the beds part, a notch at
 * the waterline, and V cracks down its sides (round 11b: the displaced
 * spheres read as "dark, blotchy camouflage blobs with a melted silhouette").
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

/** One ring of a piece's profile: height above its anchor, metres, and the outline's scale there. */
interface Ring {
  y: number;
  s: number;
}

export interface BeachrockPiece {
  kind: 'slab' | 'block';
  /** Where it stands (its bed point) and how it is turned and tipped. */
  anchor: THREE.Vector3;
  rotation: THREE.Quaternion;
  /** Outline radii along local x and z, metres. */
  a: number;
  b: number;
  /** The profile from its buried base to its top, local metres. */
  rings: Ring[];
  seed: number;
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

/**
 * A slab's profile: buried base, the notch at the still-water line with the
 * lip overhanging it, and one step back where an upper bed has broken away if
 * it stands tall enough to have one.
 */
function slabRings(top: number, water: number, rnd: () => number): Ring[] {
  const bottom = -0.3;
  // Above the reach of the still water (a slab up the beach) there is no
  // notch, just an edge broken back a little toward the top.
  if (water < bottom + 0.08) {
    return [
      { y: bottom, s: 0.92 },
      { y: Math.min(0.04, top * 0.3), s: 0.97 },
      { y: top - 0.04, s: 1 },
      { y: top, s: 0.97 },
    ];
  }
  const notch = Math.min(water, top - 0.1);
  const rings: Ring[] = [
    { y: bottom, s: 0.9 },
    { y: notch - 0.05, s: 0.8 },
    { y: notch + 0.03, s: 0.8 },
    { y: Math.min(notch + 0.09, top - 0.03), s: 1 },
  ];
  if (top - notch > 0.32) {
    const ledge = notch + 0.12 + (top - notch - 0.2) * (0.35 + rnd() * 0.3);
    const back = 0.8 + rnd() * 0.1;
    rings.push({ y: ledge, s: 0.99 }, { y: ledge, s: back }, { y: top, s: back * 0.99 });
  } else {
    rings.push({ y: top, s: 0.99 });
  }
  return rings;
}

/**
 * A block's profile: two or three beds stepping back toward a flat top, each
 * slightly undercut at its foot, and the waterline notch cut into whichever
 * bed the still water reaches.
 */
function blockRings(top: number, water: number, rnd: () => number): Ring[] {
  const bottom = -0.45;
  const beds = top - bottom > 2 ? 3 : 2;
  const rings: Ring[] = [{ y: bottom, s: 0.95 }];
  let s = 1;
  const span = top - bottom;
  let y = bottom;
  for (let i = 0; i < beds; i++) {
    const h = i === beds - 1 ? top - y : span * (0.3 + rnd() * 0.15);
    const y1 = y + h;
    // Undercut at the foot of the bed, the bed's face, then its top edge.
    rings.push({ y: y + Math.min(0.12, h * 0.2), s: s * 0.94 }, { y: y1 - Math.min(0.05, h * 0.1), s });
    if (i < beds - 1) {
      const back = s * (0.8 + rnd() * 0.1);
      rings.push({ y: y1, s }, { y: y1, s: back });
      s = back;
    } else {
      rings.push({ y: top, s: s * 0.99 });
    }
    y = y1;
  }
  // The notch: the face drawn in a fifth around the still water.
  if (water > bottom + 0.2 && water < top - 0.2) {
    const out: Ring[] = [];
    for (const r of rings) {
      if (out.length && out[out.length - 1].y <= water - 0.14 && r.y > water - 0.14) {
        const at = scaleAt(rings, water);
        out.push({ y: water - 0.14, s: at }, { y: water - 0.08, s: at * 0.8 }, { y: water + 0.04, s: at * 0.8 }, { y: water + 0.1, s: at });
      }
      if (r.y <= water - 0.14 || r.y >= water + 0.1) out.push(r);
    }
    return out;
  }
  return rings;
}

/** The profile's outline scale at local height `y` (the first match on a ledge). */
function scaleAt(rings: Ring[], y: number): number {
  if (y <= rings[0].y) return rings[0].s;
  for (let i = 1; i < rings.length; i++) {
    const r0 = rings[i - 1];
    const r1 = rings[i];
    if (y <= r1.y) return r1.y === r0.y ? r0.s : r0.s + ((r1.s - r0.s) * (y - r0.y)) / (r1.y - r0.y);
  }
  return rings[rings.length - 1].s;
}

let plan: { pieces: BeachrockPiece[]; footprints: BeachrockFootprint[] } | null = null;

function build(): { pieces: BeachrockPiece[]; footprints: BeachrockFootprint[] } {
  const rng = mulberry32(BEACHROCK_SEED);
  const pieces: BeachrockPiece[] = [];
  const footprints: BeachrockFootprint[] = [];
  const q = new THREE.Quaternion();
  const seat = (bearing: number, radius: number) => {
    const x = ISLAND.x + Math.cos(bearing) * radius;
    const z = ISLAND.z + Math.sin(bearing) * radius;
    return new THREE.Vector3(x, seafloorHeight(x, z), z);
  };
  const add = (kind: BeachrockPiece['kind'], anchor: THREE.Vector3, a: number, b: number, rings: Ring[], seed: number) => {
    pieces.push({ kind, anchor, rotation: q.clone(), a, b, rings, seed });
    const top = anchor.y + rings[rings.length - 1].y;
    // Under water it still shoals the bores, so it keeps a third of its radius there.
    const cut = top < 0 ? 0.35 : scaleAt(rings, -anchor.y);
    const ax = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    ax.y = 0;
    ax.normalize();
    footprints.push({ x: anchor.x, z: anchor.z, ax: ax.x, az: ax.z, ra: a * cut, rb: b * cut });
  };
  // Shape draws come from a stream of their own per piece, so the positions
  // drawn from `rng` are the ones round 11 placed.
  const shapeRng = (i: number) => mulberry32(BEACHROCK_SEED ^ (0x9e37 * (i + 1)));

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
    const thick = 0.75 + rng() * 0.5;
    const wide = 0.7 + rng() * 0.4;
    const r = shapeRng(i);
    const top = 0.12 + 0.06 * length * thick;
    add('slab', p, length * 0.75, length * 0.41 * wide, slabRings(top, -p.y, r), 0xbe0 + i * 131);
  }
  // Heads: a few broken blocks standing half a metre to a metre clear of the
  // still water in the inner surf.
  const heads = 3;
  for (let i = 0; i < heads; i++) {
    const t = (i + 0.25 + rng() * 0.5) / heads;
    const bearing = HEADS.from + (HEADS.to - HEADS.from) * t;
    const shore = shoreRadius(bearing);
    const p = seat(bearing, shore + 6.6 + rng() * 2);
    const k = 2 + rng() * 1.3;
    const rise = 0.5 + rng() * 0.5;
    const top = Math.max(k * 0.25, rise - p.y);
    q.setFromAxisAngle(UP, rng() * Math.PI * 2);
    tip(q, shapeRng(10 + i));
    const w = 0.8 + rng() * 0.4;
    // Four fifths of the reef's size: a fuller block reached into the left edge of beach-swash.
    add('block', p, k * 0.8, k * 0.64 * w, blockRings(top, -p.y, shapeRng(20 + i)), 0xbe7 + i * 59);
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
    const top = Math.max(k * 0.3, rise - p.y);
    q.setFromAxisAngle(UP, rng() * Math.PI * 2);
    tip(q, shapeRng(30 + i));
    const w = 0.75 + rng() * 0.4;
    add('block', p, k, k * 0.8 * w, blockRings(top, -p.y, shapeRng(40 + i)), 0xbf3 + i * 71);
  }
  return { pieces, footprints };
}

/** A broken block lies a few degrees off level, about a random horizontal axis. */
function tip(q: THREE.Quaternion, rnd: () => number): void {
  const t = rnd() * Math.PI * 2;
  const axis = new THREE.Vector3(Math.cos(t), 0, Math.sin(t));
  q.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, 0.03 + rnd() * 0.09));
}

/** Every beachrock piece. Pure; computed once. */
export function beachrockPieces(): readonly BeachrockPiece[] {
  plan ??= build();
  return plan.pieces;
}

/** Every piece's waterline ellipse, for the surf. Pure; computed once. */
export function beachrockFootprints(): readonly BeachrockFootprint[] {
  plan ??= build();
  return plan.footprints;
}

// ------------------------------------------------------------ geometry

const SEGMENTS = 96;

/** A periodic wobble of the outline: a few low harmonics with random phases. */
function harmonics(rnd: () => number, from: number, to: number, fall: number): (t: number) => number {
  const terms: [number, number, number][] = [];
  for (let k = from; k <= to; k++) terms.push([k, rnd() * Math.PI * 2, (rnd() * 0.6 + 0.7) / Math.pow(k, fall)]);
  return (t) => terms.reduce((s, [k, ph, amp]) => s + amp * Math.sin(k * t + ph), 0);
}

/**
 * Builds a piece in its own frame (anchor at the origin, metres): the outline
 * swept through its rings, each band between two rings with its own vertices
 * so steps and the notch keep hard edges, and a flat top.
 *
 * The vertex colour carries large-scale occlusion: under the lip and in the
 * notch dark, cracks dark, ledges and the top open.
 */
export function buildBeachrockGeometry(piece: BeachrockPiece): THREE.BufferGeometry {
  const rnd = mulberry32(piece.seed);
  const coarse = harmonics(rnd, 2, 7, 0.9);
  const fine = harmonics(rnd, 9, 20, 0.6);
  // Joints: V notches down the sides, two to four of them.
  const cracks: [number, number, number][] = [];
  const nCracks = 2 + Math.floor(rnd() * 3);
  for (let i = 0; i < nCracks; i++) cracks.push([rnd() * Math.PI * 2, 0.035 + rnd() * 0.03, 0.08 + rnd() * 0.12]);
  const crackAt = (t: number) => {
    let cut = 0;
    for (const [c, w, d] of cracks) {
      let dt = Math.abs(t - c);
      dt = Math.min(dt, Math.PI * 2 - dt);
      cut = Math.max(cut, d * Math.max(0, 1 - dt / w));
    }
    return cut;
  };
  const ringJitter = piece.rings.map(() => harmonics(rnd, 5, 14, 0.5));
  const outline = (t: number, ring: number, s: number) => {
    const c = Math.cos(t);
    const sn = Math.sin(t);
    const e = 1 / Math.sqrt((c / piece.a) ** 2 + (sn / piece.b) ** 2);
    const r = e * (1 + 0.09 * coarse(t) + 0.025 * fine(t) + 0.018 * ringJitter[ring](t)) * (1 - crackAt(t)) * s;
    return [c * r, sn * r];
  };

  const pos: number[] = [];
  const col: number[] = [];
  const idx: number[] = [];
  const bandStart: number[] = [];
  const rings = piece.rings;
  for (let i = 0; i + 1 < rings.length; i++) {
    const r0 = rings[i];
    const r1 = rings[i + 1];
    const base = pos.length / 3;
    bandStart.push(base);
    for (const [ri, r] of [
      [i, r0],
      [i + 1, r1],
    ] as [number, Ring][]) {
      for (let j = 0; j <= SEGMENTS; j++) {
        const t = (j / SEGMENTS) * Math.PI * 2;
        const [x, z] = outline(t, ri, r.s);
        pos.push(x, r.y, z);
        // Occlusion: faces under an overhang and the buried foot are dark,
        // cracks darker toward their root.
        const under = r1.s > r0.s + 0.01 ? 0.55 : 1;
        const foot = i === 0 ? 0.6 : 1;
        const crack = 1 - Math.min(0.55, crackAt(t) * 4);
        const k = Math.min(under, foot) * crack;
        col.push(k, k, k);
      }
    }
    for (let j = 0; j < SEGMENTS; j++) {
      const a = base + j;
      const b = base + j + 1;
      const c = base + SEGMENTS + 1 + j;
      const d = base + SEGMENTS + 2 + j;
      idx.push(a, c, b, b, c, d);
    }
  }
  // The top: concentric rings to a centre, a few centimetres of relief.
  const topY = rings[rings.length - 1].y;
  const topS = rings[rings.length - 1].s;
  const capBase = pos.length / 3;
  const fractions = [1, 0.82, 0.6, 0.38, 0.18];
  for (let k = 0; k < fractions.length; k++) {
    for (let j = 0; j <= SEGMENTS; j++) {
      const t = (j / SEGMENTS) * Math.PI * 2;
      const [x, z] = outline(t, rings.length - 1, topS * fractions[k]);
      const relief = k === 0 ? 0 : 0.02 * Math.sin(x * 2.3 + z * 1.7 + piece.seed) * Math.cos(z * 2.9 - x * 1.1);
      pos.push(x, topY + relief, z);
      const kc = 1 - Math.min(0.4, crackAt(t) * 2 * (1 - fractions[k]));
      col.push(kc, kc, kc);
    }
  }
  pos.push(0, topY, 0);
  col.push(1, 1, 1);
  const centre = pos.length / 3 - 1;
  for (let k = 0; k + 1 < fractions.length; k++) {
    const r0 = capBase + k * (SEGMENTS + 1);
    const r1 = r0 + SEGMENTS + 1;
    for (let j = 0; j < SEGMENTS; j++) idx.push(r0 + j, r1 + j, r0 + j + 1, r0 + j + 1, r1 + j, r1 + j + 1);
  }
  const last = capBase + (fractions.length - 1) * (SEGMENTS + 1);
  for (let j = 0; j < SEGMENTS; j++) idx.push(last + j, centre, last + j + 1);

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array((pos.length / 3) * 2), 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** A piece's placement: its anchor and rotation, no scale (the geometry is in metres). */
export function beachrockMatrix(piece: BeachrockPiece): THREE.Matrix4 {
  return new THREE.Matrix4().compose(piece.anchor, piece.rotation, new THREE.Vector3(1, 1, 1));
}
