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

const SEGMENTS = 128;

/** A periodic wobble: a few harmonics with random phases. */
function harmonics(rnd: () => number, from: number, to: number, fall: number): (t: number) => number {
  const terms: [number, number, number][] = [];
  for (let k = from; k <= to; k++) terms.push([k, rnd() * Math.PI * 2, (rnd() * 0.6 + 0.7) / Math.pow(k, fall)]);
  return (t) => terms.reduce((s, [k, ph, amp]) => s + amp * Math.sin(k * t + ph), 0);
}

/** Lattice value noise in [-1, 1], smooth, CPU only. */
function valueNoise2(x: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const h = (i: number, j: number) => {
    let v = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(seed, 144665);
    v = Math.imul(v ^ (v >>> 13), 1274126177);
    return (((v ^ (v >>> 16)) >>> 0) / 4294967296) * 2 - 1;
  };
  const p = h(ix, iz) + (h(ix + 1, iz) - h(ix, iz)) * ux;
  const q = h(ix, iz + 1) + (h(ix + 1, iz + 1) - h(ix, iz + 1)) * ux;
  return p + (q - p) * uz;
}

/** Noise at the 0.3-2 m scales the sea bites a rock's plan to: metres of radius. */
function planNoise(x: number, z: number, seed: number): number {
  return (
    valueNoise2(x / 2, z / 2, seed) * 0.2 +
    valueNoise2(x / 0.8 + 17.1, z / 0.8 + 3.3, seed + 1) * 0.09 +
    valueNoise2(x / 0.33 + 5.7, z / 0.33 + 29.2, seed + 2) * 0.04
  );
}

/** Arc distance between two angles, radians. */
function arc(p: number, q: number): number {
  const d = Math.abs(p - q) % (Math.PI * 2);
  return Math.min(d, Math.PI * 2 - d);
}

/** A ring as built: its nominal height and scale, and how it is worn. */
interface BuiltRing {
  y: number;
  s: number;
  /** A bed's outer top edge, worn round: the shoulder below it and the crown drawn in. */
  bevel: 'none' | 'shoulder' | 'crown';
  /** The inner ring of a step back, and the scale it steps back from. */
  stepFrom: number | null;
  /** Start a fresh row of vertices: a hard, concave corner. */
  split: boolean;
  /** Which bed it belongs to, for the bed's own wander. */
  bed: number;
}

/**
 * The profile as built: each bed's outer top edge worn round (a shoulder ring
 * a little below it, the edge itself drawn in, sharing normals), the concave
 * inner corners of the steps kept hard, and the beds numbered.
 */
function wearProfile(rings: Ring[]): BuiltRing[] {
  const out: BuiltRing[] = [];
  let bed = 0;
  for (let i = 0; i < rings.length; i++) {
    const r = rings[i];
    const next = rings[i + 1];
    const prev = rings[i - 1];
    const isTop = i === rings.length - 1;
    const ledgeOuter = next !== undefined && next.y === r.y && next.s < r.s;
    const ledgeInner = prev !== undefined && prev.y === r.y && prev.s > r.s;
    if (isTop || ledgeOuter) {
      const drop = Math.min(0.14, Math.max(0.03, (r.y - (prev?.y ?? r.y - 0.3)) * 0.45));
      out.push({ y: r.y - drop, s: r.s, bevel: 'shoulder', stepFrom: null, split: false, bed });
      out.push({ y: r.y, s: r.s, bevel: 'crown', stepFrom: null, split: false, bed });
    } else if (ledgeInner) {
      out.push({ y: r.y, s: r.s, bevel: 'none', stepFrom: prev.s, split: false, bed });
      out.push({ y: r.y, s: r.s, bevel: 'none', stepFrom: prev.s, split: true, bed });
      bed++;
    } else {
      out.push({ y: r.y, s: r.s, bevel: 'none', stepFrom: null, split: false, bed });
    }
  }
  return out;
}

interface MeshData {
  pos: number[];
  col: number[];
  idx: number[];
}

/**
 * Builds a piece in its own frame (anchor at the origin, metres).
 *
 * Round 11b's blocks read as "stacked concrete or a man-made pier": every bed
 * the same thickness all round, ruled edges, right-angled continuous steps,
 * evenly spaced V cuts and a crisp flat top. Beachrock is a cemented beach
 * the sea is taking apart, so here:
 *
 *  - the plan is noise at 2, 0.8 and 0.33 m over a few broad lobes, shifting
 *    a little with height so the faces are not extrusions;
 *  - the edge is bitten into scallops, most and deepest seaward and at the
 *    still water, where the waves work;
 *  - each bed's top edge is worn round, wider seaward; the concave inner
 *    corners stay sharp;
 *  - each bed's thickness wanders round the rock, and how far each step sets
 *    back does too, down to almost nothing, so ledges come and go;
 *  - the top tips and undulates;
 *  - fissures are few, uneven and branching, opening toward the top;
 *  - broken slabs and chunks lie about the foot.
 *
 * The vertex colour carries large-scale occlusion: under overhangs, at the
 * buried foot and down the fissures.
 */
export function buildBeachrockGeometry(piece: BeachrockPiece): THREE.BufferGeometry {
  const mesh: MeshData = { pos: [], col: [], idx: [] };
  const rnd = mulberry32(piece.seed);
  const inv = piece.rotation.clone().invert();
  const toSea = new THREE.Vector3(piece.anchor.x - ISLAND.x, 0, piece.anchor.z - ISLAND.z).normalize().applyQuaternion(inv);
  const seawardAngle = Math.atan2(toSea.z, toSea.x);
  buildPart(mesh, { a: piece.a, b: piece.b, rings: piece.rings, seed: piece.seed, seawardAngle, water: -piece.anchor.y, full: true }, rnd);
  scatterChunks(mesh, piece, seawardAngle, rnd);

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(mesh.pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(mesh.col, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array((mesh.pos.length / 3) * 2), 2));
  g.setIndex(mesh.idx);
  g.computeVertexNormals();
  return g;
}

interface PartSpec {
  a: number;
  b: number;
  rings: Ring[];
  seed: number;
  seawardAngle: number;
  /** The still water's height in this part's frame: where the bites are deepest. */
  water: number;
  /** A whole rock (noise, bites, fissures, full resolution) or a chunk. */
  full: boolean;
  transform?: THREE.Matrix4;
}

function buildPart(mesh: MeshData, spec: PartSpec, rnd: () => number): void {
  const { a, b, rings, seed, seawardAngle, water, full } = spec;
  const segments = full ? SEGMENTS : 32;
  const lobes = harmonics(rnd, 2, 5, 0.9);
  const seawardness = (t: number) => Math.max(0, Math.cos(t - seawardAngle));

  // Bites: scallops out of the edge, more and deeper seaward.
  const bites: { at: number; width: number; depth: number }[] = [];
  // Fissures: one to three, most with a narrower branch beside them.
  const fissures: { at: number; width: number; depth: number; from: number }[] = [];
  if (full) {
    const nb = 7 + Math.floor(rnd() * 6);
    for (let i = 0; i < nb; i++) {
      const at = rnd() * Math.PI * 2;
      const sw = seawardness(at);
      const keep = rnd() < 0.45 + 0.55 * sw;
      const width = 0.35 + rnd() * 0.8;
      const depth = (0.1 + rnd() * 0.25) * (0.6 + 0.8 * sw);
      if (keep) bites.push({ at, width, depth });
    }
    const nf = 1 + Math.floor(rnd() * 3);
    for (let i = 0; i < nf; i++) {
      const at = rnd() * Math.PI * 2;
      const width = 0.12 + rnd() * 0.2;
      fissures.push({ at, width, depth: 0.25 + rnd() * 0.45, from: rnd() * 0.3 });
      const branch = rnd() < 0.6;
      const off = (rnd() - 0.5) * 0.5;
      const bd = 0.12 + rnd() * 0.2;
      const bf = 0.35 + rnd() * 0.3;
      if (branch) fissures.push({ at: at + off, width: width * 0.5, depth: bd, from: bf });
    }
  }
  const built = wearProfile(rings);
  const bottom = rings[0].y;
  const top = rings[rings.length - 1].y;
  const span = Math.max(0.2, top - bottom);
  const beds = built[built.length - 1].bed + 1;
  const bedWander = Array.from({ length: beds }, () => harmonics(rnd, 2, 9, 0.6));
  const stepWander = Array.from({ length: beds }, () => harmonics(rnd, 1, 6, 0.5));
  const wanderAmp = full ? Math.min(0.09, span * 0.06) : 0;
  const bevelSize = full ? 0.06 + rnd() * 0.04 : 0.035;
  const tiltX = (rnd() - 0.5) * 0.08;
  const tiltZ = (rnd() - 0.5) * 0.08;
  const topRelief = (x: number, z: number) =>
    tiltX * x + tiltZ * z + (full ? valueNoise2(x / 1.4, z / 1.4, seed + 7) * 0.05 + valueNoise2(x / 0.5, z / 0.5, seed + 8) * 0.015 : 0);

  const v = new THREE.Vector3();
  const push = (x: number, y: number, z: number, k: number) => {
    v.set(x, y, z);
    if (spec.transform) v.applyMatrix4(spec.transform);
    mesh.pos.push(v.x, v.y, v.z);
    mesh.col.push(k, k, k);
  };

  /** The worn outline at angle `t` on a ring: radius and height, metres, and occlusion. */
  const at = (t: number, r: BuiltRing): { radius: number; y: number; k: number } => {
    const c = Math.cos(t);
    const sn = Math.sin(t);
    const e = (1 / Math.sqrt((c / a) ** 2 + (sn / b) ** 2)) * (1 + 0.08 * lobes(t));
    const h = (r.y - bottom) / span;
    let radius = e + (full ? planNoise(c * e + r.y * 0.6, sn * e - r.y * 0.4, seed) : 0);
    const atWater = Math.exp(-(((r.y - water) / 0.5) ** 2)) * 0.5 + 0.5;
    for (const bite of bites) {
      const d = (arc(t, bite.at) * e) / bite.width;
      if (d < 1) radius -= bite.depth * atWater * Math.pow(1 - d * d, 1.5);
    }
    let fissure = 0;
    for (const f of fissures) {
      if (h < f.from) continue;
      const open = (h - f.from) / (1 - f.from);
      const w = f.width * (0.5 + open) * (0.8 + 0.4 * valueNoise2(r.y * 6, f.at * 9, seed + 3));
      const d = (arc(t, f.at) * e) / w;
      if (d < 1) fissure = Math.max(fissure, f.depth * open * (1 - d));
    }
    radius -= fissure;
    let s = r.s;
    if (r.stepFrom !== null) {
      const m = THREE.MathUtils.smoothstep(stepWander[r.bed](t), -0.35, 0.45);
      s = r.stepFrom - (r.stepFrom - r.s) * (0.08 + 0.92 * m);
    }
    radius *= s;
    const wear = bevelSize * (1 + seawardness(t));
    if (r.bevel === 'crown') radius -= wear;
    let y = r.y;
    if (r.bevel === 'shoulder') y -= wear * 0.6;
    if (r.y > bottom + 0.01) y += bedWander[r.bed](t) * wanderAmp;
    radius = Math.max(0.05, radius);
    if (r.y >= top - 1e-6) y += topRelief(c * radius, sn * radius);
    let k = r.y <= bottom + 0.01 ? 0.6 : 1;
    k *= 1 - Math.min(0.5, fissure * 2);
    return { radius, y, k };
  };

  // The sides: rows shared across worn edges, a fresh row at each concave corner.
  let prevRow = -1;
  let prevY: number[] = [];
  for (let i = 0; i < built.length; i++) {
    const r = built[i];
    const row = mesh.pos.length / 3;
    const ys: number[] = [];
    const overhang = i > 0 && r.s > built[i - 1].s + 0.01 && r.y > built[i - 1].y;
    for (let j = 0; j <= segments; j++) {
      const t = (j / segments) * Math.PI * 2;
      const p = at(t, r);
      const y = prevY.length ? Math.max(p.y, prevY[j]) : p.y;
      ys.push(y);
      push(Math.cos(t) * p.radius, y, Math.sin(t) * p.radius, p.k * (overhang ? 0.55 : 1));
    }
    if (prevRow >= 0 && !r.split) {
      for (let j = 0; j < segments; j++) {
        const a0 = prevRow + j;
        const c0 = row + j;
        mesh.idx.push(a0, c0, a0 + 1, a0 + 1, c0, c0 + 1);
      }
    }
    prevRow = row;
    prevY = ys;
  }

  // The top: rings in from the crown to a centre, on the top's tilt and relief.
  const crown = built[built.length - 1];
  let last = prevRow;
  for (const f of [0.85, 0.62, 0.4, 0.2]) {
    const row = mesh.pos.length / 3;
    for (let j = 0; j <= segments; j++) {
      const t = (j / segments) * Math.PI * 2;
      const p = at(t, crown);
      const x = Math.cos(t) * p.radius * f;
      const z = Math.sin(t) * p.radius * f;
      push(x, top + bedWander[crown.bed](t) * wanderAmp * f + topRelief(x, z), z, 1);
    }
    for (let j = 0; j < segments; j++) mesh.idx.push(last + j, row + j, last + j + 1, last + j + 1, row + j, row + j + 1);
    last = row;
  }
  const centre = mesh.pos.length / 3;
  push(0, top + topRelief(0, 0), 0, 1);
  for (let j = 0; j < segments; j++) mesh.idx.push(last + j, centre, last + j + 1);
}

/**
 * Broken slabs and chunks about a rock's foot, most on its seaward side where
 * the waves pull them off: two to five round a block, up to two by a slab.
 * Each sits on the bed where it lands, tipped, with its own worn profile.
 */
function scatterChunks(mesh: MeshData, piece: BeachrockPiece, seawardAngle: number, rnd: () => number): void {
  const n = piece.kind === 'block' ? 2 + Math.floor(rnd() * 4) : Math.floor(rnd() * 3);
  const local = new THREE.Vector3();
  const world = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const tipQ = new THREE.Quaternion();
  for (let i = 0; i < n; i++) {
    const t = seawardAngle + (rnd() - 0.5) * 3.2;
    const c = Math.cos(t);
    const sn = Math.sin(t);
    const e = 1 / Math.sqrt((c / piece.a) ** 2 + (sn / piece.b) ** 2);
    const dist = e * (0.95 + rnd() * 0.3) + 0.25;
    const size = 0.3 + rnd() * 0.5;
    const h = 0.12 + rnd() * 0.28;
    local.set(c * dist, 0, sn * dist);
    world.copy(local).applyQuaternion(piece.rotation).add(piece.anchor);
    local.y = seafloorHeight(world.x, world.z) - piece.anchor.y - 0.06;
    q.setFromAxisAngle(UP, rnd() * Math.PI * 2);
    const tipAt = rnd() * Math.PI * 2;
    tipQ.setFromAxisAngle(new THREE.Vector3(Math.cos(tipAt), 0, Math.sin(tipAt)), 0.05 + rnd() * 0.3);
    q.premultiply(tipQ);
    const transform = new THREE.Matrix4().compose(local, q, new THREE.Vector3(1, 1, 1));
    const chunkSeed = piece.seed * 31 + i * 7;
    buildPart(
      mesh,
      {
        a: size,
        b: size * (0.5 + rnd() * 0.4),
        rings: [
          { y: -0.12, s: 0.9 },
          { y: h - 0.04, s: 1 },
          { y: h, s: 0.95 },
        ],
        seed: chunkSeed,
        seawardAngle: 0,
        water: 99,
        full: false,
        transform,
      },
      mulberry32(chunkSeed),
    );
  }
}

/** A piece's placement: its anchor and rotation, no scale (the geometry is in metres). */
export function beachrockMatrix(piece: BeachrockPiece): THREE.Matrix4 {
  return new THREE.Matrix4().compose(piece.anchor, piece.rotation, new THREE.Vector3(1, 1, 1));
}
