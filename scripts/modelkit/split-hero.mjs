/**
 * Splits the hero ship's moving parts out of its joined meshes.
 *
 * Poly Haven's `dutch_ship_medium` is three joined meshes: hull, rigging and
 * sails. The rudder is part of the keel slab in the hull, the helm (a
 * whipstaff, see below) is an island in the hull, and the four square yards are
 * islands in the rigging. None of them can turn. This cuts them out into nodes
 * of their own with their pivots where the real thing pivots, and writes the
 * GLB back with its KTX2 textures untouched and its geometry re-encoded with
 * the same Meshopt settings `scripts/optimize-assets.mjs` uses for heroes.
 *
 * Nodes added (model units, model frame: +X bow, +Y up, +Z starboard):
 *
 *   rudder          pivot on the rudder's leading edge (the line where its
 *                   bevelled fore edge meets the sternpost), `extras.axis` the
 *                   hinge direction, raked aft with the sternpost.
 *   whipstaff       pivot where the staff passes through the quarterdeck,
 *                   `extras.axis` fore and aft: it swings athwartships.
 *   yard_fore, yard_fore_top, yard_main, yard_main_top
 *                   pivot at the middle of each yard, `extras.axis` vertical.
 *                   Each carries its slings and yardarm blocks; the braces,
 *                   lifts and sheets that end at the yardarms stay in the
 *                   rigging mesh.
 *
 * Every pivot node is mesh-less with identity rotation and one child,
 * `<name>_geo`, holding the geometry, so the runtime can rotate the pivot and
 * Meshopt's quantisation transform stays on the child.
 *
 * The helm is a whipstaff, not a wheel. That is what the model carries (a 1.2 m
 * lever rising through the quarterdeck just aft of the binnacle) and it is
 * right for a Dutch ship of about 1700: the wheel appears in the Royal Navy
 * around 1703-1710 and spreads through the century.
 *
 * Usage, from the repository root:
 *
 *   node scripts/modelkit/split-hero.mjs [in.glb] [out.glb]
 *
 * Both default to `public/models/hero/dutch_ship_medium.glb`. The script refuses
 * a file that is already split. `optimize-assets.mjs --force` rebuilds the
 * unsplit file from the source glTF; run this after it.
 *
 * The selection rules are geometric (islands, a hinge line) and measured on
 * this model; each is checked against the triangle count it found, so a
 * different or re-exported source fails loudly rather than cutting the wrong
 * thing.
 */

import { statSync } from 'node:fs';
import { argv, exit } from 'node:process';
import { compactPrimitive, dequantize, meshopt, prune } from '@gltf-transform/functions';
import { MeshoptEncoder } from 'meshoptimizer';
import { bytes, io, weldByPosition } from './lib.mjs';

const DEFAULT = 'public/models/hero/dutch_ship_medium.glb';
const input = argv[2] ?? DEFAULT;
const output = argv[3] ?? input;
/** The hero settings in `optimize-assets.mjs` (HERO_MESHOPT). */
const HERO_MESHOPT = { quantizePosition: 16, quantizeTexcoord: 14, quantizeNormal: 10 };

/**
 * The rudder's leading edge, model frame: the ridge of its bevelled fore edge,
 * which lies along the sternpost's after face. Two points on it.
 */
const RIDGE_A = [-8.074, -1.459, 0];
const RIDGE_B = [-8.746, 1.594, 0];
/** Top of the rudder blade; above it the rudder head runs up into the counter. */
const RUDDER_TOP = 1.7;
/** Whipstaff island bounds. */
const WHIPSTAFF_BOX = { min: [-6.8, 3.0, -0.08], max: [-6.55, 4.05, 0.08] };
/** Islands within this distance of a yard's axis everywhere ride with it. */
const YARD_REACH = 0.5;

const sourceSize = statSync(input).size;
const document = await io.read(input);
const root = document.getRoot();
const scene = root.getDefaultScene() ?? root.listScenes()[0];

if (root.listNodes().some((n) => n.getName() === 'rudder')) {
  console.error(`${input} is already split (has a 'rudder' node). Rebuild it with optimize-assets.mjs --force first.`);
  exit(1);
}

await document.transform(dequantize());

const hullNode = findNode('dutch_ship_medium_hull');
const rigNode = findNode('dutch_ship_medium_rigging');
const hullBoxBefore = nodeBox(hullNode);

// ------------------------------------------------------------------ hull parts
{
  const prim = hullNode.getMesh().listPrimitives()[0];
  const P = worldPositions(hullNode, prim);
  const islands = islandsOf(prim);
  const tris = triangles(prim);

  // The keel slab: the island that runs most of the length and is thin across.
  let keel = -1;
  let keelLength = 0;
  for (const [id, box] of islands.boxes) {
    const len = box.max[0] - box.min[0];
    if (box.max[2] - box.min[2] < 0.3 && len > keelLength) {
      keelLength = len;
      keel = id;
    }
  }
  const ridgeX = (y) => RIDGE_A[0] + ((y - RIDGE_A[1]) / (RIDGE_B[1] - RIDGE_A[1])) * (RIDGE_B[0] - RIDGE_A[0]);
  const rudder = [];
  const whipstaff = [];
  for (let t = 0; t < tris.length; t += 3) {
    const v = [tris[t], tris[t + 1], tris[t + 2]];
    const isl = islands.of[v[0]];
    if (isl === keel) {
      const d = v.map((i) => P[i * 3] - ridgeX(P[i * 3 + 1]));
      const top = Math.max(...v.map((i) => P[i * 3 + 1]));
      // Aft of the ridge (a centimetre's slack), below the head, and not lying
      // flat on the ridge itself: that is the sternpost's after face.
      if (top < RUDDER_TOP && Math.max(...d) < 0.01 && Math.min(...d) < -0.01) rudder.push(t / 3);
    } else if (v.every((i) => inside(P, i, WHIPSTAFF_BOX))) {
      whipstaff.push(t / 3);
    }
  }
  expectCount('rudder', rudder.length, 100, 160);
  expectCount('whipstaff', whipstaff.length, 40, 200);

  const ridgeDir = normalize(sub(RIDGE_B, RIDGE_A));
  // Pivot on the ridge at the model's y = 0.
  const rudderPivot = [ridgeX(0), 0, 0];
  const wsBox = boxOf(P, whipstaff.flatMap((t) => [tris[t * 3], tris[t * 3 + 1], tris[t * 3 + 2]]));
  const whipstaffPivot = [(wsBox.min[0] + wsBox.max[0]) / 2, wsBox.min[1], 0];

  const parts = [
    { name: 'rudder', tris: rudder, pivot: rudderPivot, axis: ridgeDir },
    { name: 'whipstaff', tris: whipstaff, pivot: whipstaffPivot, axis: [1, 0, 0] },
  ];
  splitOut(hullNode, prim, parts);
}

// --------------------------------------------------------------------- yards
{
  const prim = rigNode.getMesh().listPrimitives()[0];
  const P = worldPositions(rigNode, prim);
  const islands = islandsOf(prim);
  const tris = triangles(prim);

  // Square yards: long athwartships, thin vertically, not the spritsail yard
  // under the bowsprit (x > 11), which carries no sail in this model.
  const yards = [];
  for (const [id, box] of islands.boxes) {
    const span = box.max[2] - box.min[2];
    const height = box.max[1] - box.min[1];
    const length = box.max[0] - box.min[0];
    if (span > 2.5 && height < 0.35 && length < 2.2 && box.min[0] < 11) yards.push({ id, box });
  }
  if (yards.length !== 4) {
    console.error(`expected four square yards, found ${yards.length}`);
    exit(1);
  }
  for (const yard of yards) {
    // Axis: the two extreme vertices athwartships.
    let lo = -1;
    let hi = -1;
    for (const i of islands.members.get(yard.id)) {
      if (lo < 0 || P[i * 3 + 2] < P[lo * 3 + 2]) lo = i;
      if (hi < 0 || P[i * 3 + 2] > P[hi * 3 + 2]) hi = i;
    }
    yard.a = [P[lo * 3], P[lo * 3 + 1], P[lo * 3 + 2]];
    yard.b = [P[hi * 3], P[hi * 3 + 1], P[hi * 3 + 2]];
    yard.centre = scale(add(yard.a, yard.b), 0.5);
  }
  // Fore of x = 3 is the foremast; the higher of each pair is the topsail yard.
  const named = (fore, top) => {
    const pair = yards.filter((y) => (y.centre[0] > 3) === fore).sort((p, q) => p.centre[1] - q.centre[1]);
    return pair[top ? 1 : 0];
  };
  const order = [
    ['yard_fore', named(true, false)],
    ['yard_fore_top', named(true, true)],
    ['yard_main', named(false, false)],
    ['yard_main_top', named(false, true)],
  ];

  // Every island whose vertices all lie within YARD_REACH of a yard's axis.
  const owner = new Map();
  for (const [id] of islands.boxes) {
    for (const [name, yard] of order) {
      const all = islands.members.get(id).every((i) => segmentDistance(P, i, yard.a, yard.b) < YARD_REACH);
      if (all) {
        owner.set(id, name);
        break;
      }
    }
  }
  const parts = order.map(([name, yard]) => ({ name, tris: [], pivot: yard.centre, axis: [0, 1, 0] }));
  for (let t = 0; t < tris.length; t += 3) {
    const name = owner.get(islands.of[tris[t]]);
    if (name) parts.find((p) => p.name === name).tris.push(t / 3);
  }
  for (const part of parts) expectCount(part.name, part.tris.length, 150, 1200);
  splitOut(rigNode, prim, parts);
}

// ---------------------------------------------------------------------- write
const hullBoxAfter = nodeBox(hullNode, ['rudder_geo', 'whipstaff_geo']);
const drift = Math.max(...[0, 1, 2].flatMap((d) => [hullBoxBefore.min[d] - hullBoxAfter.min[d], hullBoxBefore.max[d] - hullBoxAfter.max[d]].map(Math.abs)));
console.log(`hull + rudder box drift ${drift.toExponential(2)} model units`);

await document.transform(prune(), meshopt({ encoder: MeshoptEncoder, level: 'high', ...HERO_MESHOPT }));
await io.write(output, document);
console.log(`wrote ${output}  ${bytes(sourceSize)} -> ${bytes(statSync(output).size)}`);

// ================================================================== helpers

function findNode(name) {
  const node = root.listNodes().find((n) => n.getName() === name);
  if (!node) {
    console.error(`no node named ${name}`);
    exit(1);
  }
  return node;
}

/**
 * Moves the listed triangles of `prim` into new meshes, one per part, each on
 * a `<name>_geo` node under a `<name>` pivot node at `part.pivot`.
 */
function splitOut(sourceNode, prim, parts) {
  const indices = prim.getIndices();
  const taken = new Uint8Array(indices.getCount() / 3);
  const material = prim.getMaterial();
  const semantics = prim.listSemantics();
  const srcT = sourceNode.getTranslation();
  const srcS = sourceNode.getScale();
  const srcR = sourceNode.getRotation();
  if (srcR.some((v, i) => Math.abs(v - (i === 3 ? 1 : 0)) > 1e-6)) throw new Error(`${sourceNode.getName()} is rotated`);
  const buffer = root.listBuffers()[0];

  for (const part of parts) {
    const remap = new Map();
    const order = [];
    const newIndex = [];
    for (const t of part.tris) {
      taken[t] = 1;
      for (let k = 0; k < 3; k++) {
        const v = indices.getScalar(t * 3 + k);
        let n = remap.get(v);
        if (n === undefined) {
          n = order.length;
          remap.set(v, n);
          order.push(v);
        }
        newIndex.push(n);
      }
    }
    const newPrim = document.createPrimitive().setMaterial(material).setMode(prim.getMode());
    for (const semantic of semantics) {
      const src = prim.getAttribute(semantic);
      const size = src.getElementSize();
      const out = new Float32Array(order.length * size);
      const el = [];
      order.forEach((v, n) => {
        src.getElement(v, el);
        out.set(el, n * size);
      });
      newPrim.setAttribute(
        semantic,
        document.createAccessor().setType(src.getType()).setArray(out).setNormalized(src.getNormalized()).setBuffer(buffer),
      );
    }
    newPrim.setIndices(document.createAccessor().setType('SCALAR').setArray(new Uint32Array(newIndex)).setBuffer(buffer));
    const mesh = document.createMesh(`${part.name}_geo`).addPrimitive(newPrim);
    const geo = document
      .createNode(`${part.name}_geo`)
      .setMesh(mesh)
      .setTranslation([srcT[0] - part.pivot[0], srcT[1] - part.pivot[1], srcT[2] - part.pivot[2]])
      .setScale(srcS);
    const pivot = document
      .createNode(part.name)
      .setTranslation(part.pivot)
      .setExtras({ axis: part.axis.map((v) => Number(v.toFixed(5))) })
      .addChild(geo);
    scene.addChild(pivot);
    console.log(
      `  ${part.name.padEnd(14)} ${String(part.tris.length).padStart(5)} tris  pivot [${part.pivot.map((v) => v.toFixed(3)).join(', ')}]` +
        `  axis [${part.axis.map((v) => v.toFixed(3)).join(', ')}]`,
    );
  }

  const kept = [];
  for (let t = 0; t < taken.length; t++) {
    if (taken[t]) continue;
    kept.push(indices.getScalar(t * 3), indices.getScalar(t * 3 + 1), indices.getScalar(t * 3 + 2));
  }
  prim.setIndices(document.createAccessor().setType('SCALAR').setArray(new Uint32Array(kept)).setBuffer(buffer));
  compactPrimitive(prim);
}

/** Positions in the scene frame (top-level nodes: world = local). */
function worldPositions(node, prim) {
  const m = node.getWorldMatrix();
  const pos = prim.getAttribute('POSITION');
  const n = pos.getCount();
  const P = new Float64Array(n * 3);
  const e = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    pos.getElement(i, e);
    P[i * 3] = m[0] * e[0] + m[4] * e[1] + m[8] * e[2] + m[12];
    P[i * 3 + 1] = m[1] * e[0] + m[5] * e[1] + m[9] * e[2] + m[13];
    P[i * 3 + 2] = m[2] * e[0] + m[6] * e[1] + m[10] * e[2] + m[14];
  }
  return P;
}

function triangles(prim) {
  const idx = prim.getIndices();
  const out = new Uint32Array(idx.getCount());
  for (let i = 0; i < out.length; i++) out[i] = idx.getScalar(i);
  return out;
}

/** Connected islands, welded by position: `of[vertex]`, bounds and members per island. */
function islandsOf(prim) {
  const n = prim.getAttribute('POSITION').getCount();
  const P = worldPositions(prim.listParents().find((p) => p.propertyType === 'Mesh').listParents().find((p) => p.propertyType === 'Node'), prim);
  const weld = weldByPosition(prim);
  const parent = new Int32Array(n).map((_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    a = find(a);
    b = find(b);
    if (a !== b) parent[a] = b;
  };
  const first = new Map();
  for (let i = 0; i < n; i++) {
    const seen = first.get(weld[i]);
    if (seen === undefined) first.set(weld[i], i);
    else union(i, seen);
  }
  const tris = triangles(prim);
  for (let t = 0; t < tris.length; t += 3) {
    union(tris[t], tris[t + 1]);
    union(tris[t], tris[t + 2]);
  }
  const of = new Int32Array(n);
  const boxes = new Map();
  const members = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    of[i] = r;
    let box = boxes.get(r);
    if (!box) {
      box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
      boxes.set(r, box);
      members.set(r, []);
    }
    members.get(r).push(i);
    for (let d = 0; d < 3; d++) {
      box.min[d] = Math.min(box.min[d], P[i * 3 + d]);
      box.max[d] = Math.max(box.max[d], P[i * 3 + d]);
    }
  }
  return { of, boxes, members };
}

function inside(P, i, box) {
  for (let d = 0; d < 3; d++) if (P[i * 3 + d] < box.min[d] || P[i * 3 + d] > box.max[d]) return false;
  return true;
}

function boxOf(P, ids) {
  const box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const i of ids) {
    for (let d = 0; d < 3; d++) {
      box.min[d] = Math.min(box.min[d], P[i * 3 + d]);
      box.max[d] = Math.max(box.max[d], P[i * 3 + d]);
    }
  }
  return box;
}

/** Scene-frame bounds of a node's mesh, plus any named extra nodes. */
function nodeBox(node, extra = []) {
  const nodes = [node, ...extra.map((name) => root.listNodes().find((n) => n.getName() === name)).filter(Boolean)];
  const box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const n of nodes) {
    for (const prim of n.getMesh().listPrimitives()) {
      const P = worldPositions(n, prim);
      for (let i = 0; i < P.length; i += 3) {
        for (let d = 0; d < 3; d++) {
          box.min[d] = Math.min(box.min[d], P[i + d]);
          box.max[d] = Math.max(box.max[d], P[i + d]);
        }
      }
    }
  }
  return box;
}

function segmentDistance(P, i, a, b) {
  const ab = sub(b, a);
  const ap = [P[i * 3] - a[0], P[i * 3 + 1] - a[1], P[i * 3 + 2] - a[2]];
  const t = Math.max(0, Math.min(1, dot(ap, ab) / dot(ab, ab)));
  return Math.hypot(ap[0] - ab[0] * t, ap[1] - ab[1] * t, ap[2] - ab[2] * t);
}

function expectCount(name, count, lo, hi) {
  if (count < lo || count > hi) {
    console.error(`${name}: ${count} triangles, expected ${lo}-${hi}; the source is not the model these rules were measured on`);
    exit(1);
  }
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function normalize(a) {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
}
