/*
 * Seeded procedural plants: palms, tropical broadleaf trees and understorey.
 *
 * The branch skeleton (`growBranch`, `along`, `deflect`, `growTree`), the
 * parallel-transported bark tube (`addTube`), the leaf card (`addCard`) and the
 * crown-shell shading (`makeCrown`) are adapted from mrdoob/slop-newyork
 * `js/trees.js`:
 *
 *   MIT License — Copyright (c) 2026 Mr.doob
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions: The above copyright
 *   notice and this permission notice shall be included in all copies or
 *   substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
 *   WITHOUT WARRANTY OF ANY KIND.
 *
 * Changes from the original: ported to TypeScript; the per-vertex data channel
 * became an ordinary vertex colour (tint x crown occlusion) so the plants work
 * through the scene's standard material path and its wetting, wind and ground
 * treatments; tropical species parameters; and new generators for palms (a
 * ringed, bent trunk under a crown of folded pinnate fronds), banana, fern,
 * broadleaf and flowering shrubs, grass tufts and hanging vines.
 *
 * Every generator returns one geometry per material per level of detail, built
 * in metres with the base at the origin and y up, so `Props` can bake and
 * instance them exactly as it does a loaded model.
 */
import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import { TILES, tileUv, type TileName } from './leafAtlas';

const V3 = THREE.Vector3;
type Vec3 = THREE.Vector3;
type RGB = readonly [number, number, number];

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const smooth = (e0: number, e1: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};
const UP = new V3(0, 1, 0);

/** One level of one plant: a geometry per material slot, either may be absent. */
export interface PlantLevel {
  leaf: THREE.BufferGeometry | null;
  bark: THREE.BufferGeometry | null;
}

// =============================================================== geometry

export class GeoBuilder {
  readonly pos: number[] = [];
  readonly nrm: number[] = [];
  readonly uv: number[] = [];
  readonly col: number[] = [];
  readonly idx: number[] = [];

  vertex(p: Vec3, n: Vec3, u: number, v: number, c: RGB): number {
    this.pos.push(p.x, p.y, p.z);
    this.nrm.push(n.x, n.y, n.z);
    this.uv.push(u, v);
    this.col.push(c[0], c[1], c[2]);
    return this.pos.length / 3 - 1;
  }

  get empty(): boolean {
    return this.idx.length === 0;
  }

  build(): THREE.BufferGeometry | null {
    if (this.empty) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    const count = this.pos.length / 3;
    g.setIndex(
      count > 65535
        ? new THREE.Uint32BufferAttribute(this.idx, 1)
        : new THREE.Uint16BufferAttribute(this.idx, 1),
    );
    g.computeBoundingSphere();
    g.computeBoundingBox();
    return g;
  }
}

function shadeColour(tint: RGB, ao: number): RGB {
  const k = 0.35 + 0.65 * ao;
  return [tint[0] * k, tint[1] * k, tint[2] * k];
}

// =============================================================== skeleton

interface Bend {
  up: number;
  out: number;
  droop: number;
  jitter: number;
}

interface Branch {
  pts: Vec3[];
  rad: number[];
  ts: number[];
  len: number;
  level: number;
}

/** Unit vector `angle` off `dir`, turned `azimuth` round it. */
function deflect(dir: Vec3, angle: number, azimuth: number): Vec3 {
  const ref = Math.abs(dir.y) < 0.95 ? new V3(0, 1, 0) : new V3(1, 0, 0);
  const u = new V3().crossVectors(dir, ref).normalize();
  const w = new V3().crossVectors(dir, u);
  return dir
    .clone()
    .multiplyScalar(Math.cos(angle))
    .addScaledVector(u, Math.sin(angle) * Math.cos(azimuth))
    .addScaledVector(w, Math.sin(angle) * Math.sin(azimuth))
    .normalize();
}

/**
 * A branch as a polyline from p0 along dir, bending per metre toward the sky
 * (up), away from the trunk axis (out) and downward toward its tip (droop).
 */
function growBranch(
  p0: Vec3,
  dir: Vec3,
  len: number,
  r0: number,
  r1: number,
  segs: number,
  bend: Bend,
  rng: () => number,
  level: number,
  schedule: number[] | null = null,
): Branch {
  const ts = schedule ?? Array.from({ length: segs + 1 }, (_, i) => i / segs);
  const pts = [p0.clone()];
  const rad = [r0];
  const d = dir.clone();
  const p = p0.clone();
  const out = new V3();
  for (let i = 1; i < ts.length; i++) {
    const t = ts[i];
    const step = (t - ts[i - 1]) * len;
    out.set(p.x, 0, p.z);
    if (out.lengthSq() > 1e-6) out.normalize();
    d.x += out.x * bend.out * step + (rng() - 0.5) * bend.jitter;
    d.y += (bend.up - bend.droop * t * 2.0) * step + (rng() - 0.5) * bend.jitter * 0.6;
    d.z += out.z * bend.out * step + (rng() - 0.5) * bend.jitter;
    d.normalize();
    p.addScaledVector(d, step);
    pts.push(p.clone());
    rad.push(r0 + (r1 - r0) * Math.pow(t, 0.8));
  }
  return { pts, rad, len, level, ts };
}

/** Point, direction and radius at length fraction t of a branch. */
function along(b: Branch, t: number): { p: Vec3; d: Vec3; r: number } {
  const ts = b.ts;
  let i = 0;
  while (i < ts.length - 2 && ts[i + 1] < t) i++;
  const k = Math.min(1, Math.max(0, (t - ts[i]) / (ts[i + 1] - ts[i])));
  return {
    p: b.pts[i].clone().lerp(b.pts[i + 1], k),
    d: b.pts[i + 1].clone().sub(b.pts[i]).normalize(),
    r: lerp(b.rad[i], b.rad[i + 1], k),
  };
}

/**
 * Tapered tube along a branch with parallel-transported rings, every
 * `stride`-th skeleton point. `u` wraps `uRepeat` times round; `v` is metres
 * times `vPerMetre`.
 */
function addTube(
  gb: GeoBuilder,
  b: Branch,
  radial: number,
  stride: number,
  colourAt: (p: Vec3, t: number) => RGB,
  vPerMetre = 1,
  uRepeat = 1,
  capped = false,
): void {
  const n = b.pts.length;
  const keep: number[] = [];
  for (let i = 0; i < n; i += stride) keep.push(i);
  if (keep[keep.length - 1] !== n - 1) keep.push(n - 1);
  const tangent = (i: number): Vec3 => {
    const a = b.pts[Math.max(0, i - 1)];
    const c = b.pts[Math.min(n - 1, i + 1)];
    return c.clone().sub(a).normalize();
  };
  let T = tangent(keep[0]);
  const N =
    Math.abs(T.y) < 0.9
      ? new V3(0, 1, 0).cross(T).normalize()
      : new V3(1, 0, 0).cross(T).normalize();
  const q = new THREE.Quaternion();
  let vLen = 0;
  const base = gb.pos.length / 3;
  const dir = new V3();
  const p = new V3();
  keep.forEach((i, ki) => {
    if (ki > 0) {
      const T2 = tangent(i);
      q.setFromUnitVectors(T, T2);
      N.applyQuaternion(q).normalize();
      T = T2;
      vLen += b.pts[i].distanceTo(b.pts[keep[ki - 1]]);
    }
    const Bn = new V3().crossVectors(T, N);
    const c = colourAt(b.pts[i], b.ts[i]);
    for (let j = 0; j <= radial; j++) {
      const ang = (j / radial) * Math.PI * 2;
      dir.copy(N).multiplyScalar(Math.cos(ang)).addScaledVector(Bn, Math.sin(ang));
      p.copy(b.pts[i]).addScaledVector(dir, b.rad[i]);
      gb.vertex(p, dir, (j / radial) * uRepeat, vLen * vPerMetre, c);
    }
  });
  for (let k = 0; k < keep.length - 1; k++) {
    for (let j = 0; j < radial; j++) {
      const a = base + k * (radial + 1) + j;
      const c = a + radial + 1;
      gb.idx.push(a, a + 1, c, a + 1, c + 1, c);
    }
  }
  if (capped) {
    // Close the top end: a fan to the tip point, so a cut stem is not hollow.
    const last = keep[keep.length - 1];
    const tip = gb.vertex(b.pts[last], T, 0.5, vLen * vPerMetre, colourAt(b.pts[last], 1));
    const ring = base + (keep.length - 1) * (radial + 1);
    for (let j = 0; j < radial; j++) gb.idx.push(ring + j, ring + j + 1, tip);
  }
}

/** Crown ellipsoid for shell normals and interior occlusion. */
interface Crown {
  C: Vec3;
  R: Vec3;
  shade(p: Vec3, face: Vec3): { n: Vec3; ao: number };
  barkAO(p: Vec3): number;
}

function makeCrown(points: Vec3[], skyward = 0.22, faceWeight = 0.28): Crown {
  const box = new THREE.Box3().setFromPoints(points);
  const C = box.getCenter(new V3());
  const R = box.getSize(new V3()).multiplyScalar(0.5).max(new V3(0.4, 0.4, 0.4));
  const tmp = new V3();
  const sn = new V3();
  return {
    C,
    R,
    shade(p: Vec3, face: Vec3) {
      tmp.copy(p).sub(C).divide(R);
      const e = tmp.length();
      sn.copy(tmp).normalize();
      sn.y += skyward;
      sn.normalize();
      const f = face.dot(sn) < 0 ? face.clone().negate() : face.clone();
      const n = sn.clone().multiplyScalar(1 - faceWeight).addScaledVector(f, faceWeight).normalize();
      const top = Math.min(1, Math.max(-1, (p.y - C.y) / R.y));
      const ao = (0.25 + 0.75 * smooth(0.1, 1.0, e)) * (0.82 + 0.18 * top);
      return { n, ao };
    },
    barkAO(p: Vec3) {
      if (p.y < box.min.y) return 0.92;
      tmp.copy(p).sub(C).divide(R);
      return lerp(0.35, 0.82, smooth(0.2, 1.0, tmp.length()));
    },
  };
}

/** Leaf card: bottom edge centred on `at`, rising along `up` by `h`, `w` wide, facing `face`. */
function addCard(
  gb: GeoBuilder,
  at: Vec3,
  up: Vec3,
  face: Vec3,
  w: number,
  h: number,
  tile: TileName,
  mirror: boolean,
  crown: Crown,
  tint: RGB,
): void {
  const side = new V3().crossVectors(up, face).normalize();
  const nrm = new V3().crossVectors(side, up).normalize();
  const t = tileUv(tile);
  let u0 = t.u0;
  let u1 = t.u1;
  if (mirror) [u0, u1] = [u1, u0];
  const corners: [number, number, number, number][] = [
    [-0.5, 0, u0, t.vBot],
    [0.5, 0, u1, t.vBot],
    [0.5, 1, u1, t.vTop],
    [-0.5, 1, u0, t.vTop],
  ];
  const base = gb.pos.length / 3;
  const p = new V3();
  for (const [sx, sy, u, v] of corners) {
    p.copy(at).addScaledVector(side, sx * w).addScaledVector(up, sy * h);
    const { n, ao } = crown.shade(p, nrm);
    gb.vertex(p, n, u, v, shadeColour(tint, ao));
  }
  gb.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

/**
 * A bent, folded leaf strip along `spine`.
 *
 * `axis: 'u'` maps the length along the tile's u (the palm frond tiles, base at
 * the left) and the width across v; `axis: 'v'` maps the length up the tile from
 * its bottom edge (fern, banana leaf, grass). Three vertices across, the midrib
 * raised by `fold` times the half-width, which is what makes a frond read as a
 * V of drooping leaflets rather than as a flat ribbon.
 */
function addStrip(
  gb: GeoBuilder,
  spine: Vec3[],
  halfWidth: number,
  tile: TileName,
  axis: 'u' | 'v',
  fold: (t: number) => number,
  crown: Crown,
  tint: RGB,
  aoBias = 1,
  across = 3,
): void {
  const t = tileUv(tile);
  const n = spine.length;
  const base = gb.pos.length / 3;
  const p = new V3();
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const d = spine[Math.min(n - 1, i + 1)].clone().sub(spine[Math.max(0, i - 1)]).normalize();
    let S = new V3().crossVectors(d, UP);
    if (S.lengthSq() < 1e-4) S = new V3(1, 0, 0);
    S.normalize();
    const N = new V3().crossVectors(S, d).normalize();
    // Narrow at the base, so the strip does not show a hard square end at the
    // attachment the texture already tapers into.
    const w = halfWidth * (0.35 + 0.65 * smooth(0, 0.18, s));
    const f = fold(s);
    for (let k = 0; k < across; k++) {
      const a = across === 3 ? k - 1 : k * 2 - 1; // -1..1 across
      p.copy(spine[i]).addScaledVector(S, a * w);
      if (a !== 0) p.addScaledVector(N, -f * w);
      const { n: nn, ao } = crown.shade(p, N);
      const c = shadeColour(tint, Math.min(1, ao * aoBias));
      const acrossT = (a + 1) / 2; // 0 at -S .. 1 at +S
      if (axis === 'u') {
        gb.vertex(p, nn, lerp(t.u0, t.u1, s), lerp(t.vBot, t.vTop, acrossT), c);
      } else {
        gb.vertex(p, nn, lerp(t.u0, t.u1, acrossT), lerp(t.vBot, t.vTop, s), c);
      }
    }
  }
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < across - 1; k++) {
      const a = base + i * across + k;
      const b = a + across;
      gb.idx.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }
}

/** World metres per pixel of a tile, given the length its long axis stands for. */
function tileAspect(tile: TileName): number {
  const r = TILES[tile];
  return (r.h - 12) / (r.w - 12);
}

/** A spine from `p0` along `dir` with gravity bending it down toward the tip. */
function bentSpine(p0: Vec3, dir: Vec3, len: number, segs: number, droop: number, jitter = 0, rng?: () => number): Vec3[] {
  const pts = [p0.clone()];
  const d = dir.clone().normalize();
  const p = p0.clone();
  const step = len / segs;
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    d.y -= droop * step * (0.4 + 1.2 * t);
    if (rng && jitter > 0) {
      d.x += (rng() - 0.5) * jitter;
      d.z += (rng() - 0.5) * jitter;
    }
    d.normalize();
    p.addScaledVector(d, step);
    pts.push(p.clone());
  }
  return pts;
}

/** Resample a dense spine down to `segs` segments, keeping the ends. */
function resample(spine: Vec3[], segs: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i <= segs; i++) {
    const f = (i / segs) * (spine.length - 1);
    const k = Math.min(spine.length - 2, Math.floor(f));
    out.push(spine[k].clone().lerp(spine[k + 1], f - k));
  }
  return out;
}

// ================================================================== palms

export interface PalmParams {
  height: number;
  /** Trunk angle off vertical at the base, radians, toward +x. */
  lean: number;
  /** How much of that lean the trunk recovers toward the top, 0..1. */
  recover: number;
  baseRadius: number;
  topRadius: number;
  fronds: number;
  frondLength: number;
  coconuts: number;
  deadFronds: number;
  tint: RGB;
}

const PALM_BARK: RGB = [1.05, 1.0, 0.92];
const RING_SPACING = 0.24;

/** Palm trunk centreline, from the base up `height` metres of arc length. */
function palmTrunk(p: PalmParams, rng: () => number): Branch {
  const step = RING_SPACING / 2;
  const n = Math.max(4, Math.round(p.height / step));
  const pts: Vec3[] = [new V3(0, 0, 0)];
  const rad: number[] = [];
  const ts: number[] = [0];
  const wobbleA = rng() * Math.PI * 2;
  const pos = new V3();
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const theta = p.lean * (1 - p.recover * Math.pow(t, 0.8));
    const wob = Math.sin(t * 5 + wobbleA) * 0.04;
    const d = new V3(Math.sin(theta), Math.cos(theta), wob).normalize();
    pos.addScaledVector(d, p.height / n);
    pts.push(pos.clone());
    ts.push(t);
  }
  for (let i = 0; i <= n; i++) {
    const s = (i / n) * p.height;
    const t = i / n;
    const flare = 1 + 0.7 * Math.exp(-s / 0.35);
    // Ring scars: a groove every RING_SPACING, a swell between them.
    const ring = i % 2 === 0 ? 0.94 : 1.04;
    rad.push(lerp(p.baseRadius, p.topRadius, Math.pow(t, 0.7)) * flare * ring);
  }
  return { pts, rad, ts, len: p.height, level: 0 };
}

export function buildPalm(seed: number, params: PalmParams): PlantLevel[] {
  const rng = mulberry32(seed);
  const R = (a: number, b: number): number => a + (b - a) * rng();
  const trunk = palmTrunk(params, rng);
  const top = trunk.pts[trunk.pts.length - 1].clone();
  const topDir = top.clone().sub(trunk.pts[trunk.pts.length - 3]).normalize();
  const crownBase = top.clone().addScaledVector(topDir, 0.25);

  // Fronds, oldest (lowest, most drooping) first.
  interface Frond {
    az: number;
    el: number;
    len: number;
    droop: number;
    tile: TileName;
    tint: RGB;
  }
  const fronds: Frond[] = [];
  let az = R(0, Math.PI * 2);
  for (let i = 0; i < params.fronds; i++) {
    const age = 1 - i / Math.max(1, params.fronds - 1); // 1 oldest .. 0 youngest
    az += 2.39996 + R(-0.3, 0.3);
    const tintK = R(0.9, 1.08);
    fronds.push({
      az,
      el: lerp(1.05, -0.12, age) + R(-0.12, 0.12),
      len: params.frondLength * lerp(0.72, 1.04, Math.min(1, age * 1.6)) * R(0.92, 1.06),
      droop: lerp(0.06, 0.2, age) * R(0.85, 1.15),
      tile: age > 0.85 && rng() < 0.45 ? 'frondB' : 'frondA',
      tint: [params.tint[0] * tintK, params.tint[1] * tintK, params.tint[2] * tintK],
    });
  }
  const dead: Frond[] = [];
  for (let i = 0; i < params.deadFronds; i++) {
    dead.push({
      az: R(0, Math.PI * 2),
      el: R(-1.35, -1.05),
      len: params.frondLength * R(0.5, 0.7),
      droop: 0.05,
      tile: 'frondB',
      tint: [1.1, 0.92, 0.62],
    });
  }

  // Dense spines once; levels resample them so every level has the same shape.
  const spines = [...fronds, ...dead].map((f) => {
    const dir = new V3(Math.cos(f.el) * Math.cos(f.az), Math.sin(f.el), Math.cos(f.el) * Math.sin(f.az));
    const start = crownBase.clone().addScaledVector(new V3(Math.cos(f.az), 0, Math.sin(f.az)), 0.12);
    return bentSpine(start, dir, f.len, 16, f.droop * 2.6, 0.04, rng);
  });
  const crownPts: Vec3[] = [];
  for (const s of spines) crownPts.push(...s);
  const crown = makeCrown(crownPts, 0.35, 0.35);

  const coconutCentres: Vec3[] = [];
  for (let i = 0; i < params.coconuts; i++) {
    const a = (i / Math.max(1, params.coconuts)) * Math.PI * 2 + R(-0.3, 0.3);
    coconutCentres.push(
      top
        .clone()
        .add(new V3(Math.cos(a) * R(0.18, 0.26), R(-0.28, 0.02), Math.sin(a) * R(0.18, 0.26))),
    );
  }

  const levels: PlantLevel[] = [];
  for (let lod = 0; lod < 3; lod++) {
    const leaf = new GeoBuilder();
    const bark = new GeoBuilder();
    const radial = [8, 5, 4][lod];
    const stride = [1, 4, Math.max(1, Math.floor(trunk.pts.length / 4))][lod];
    const barkColour = (pp: Vec3, t: number): RGB => {
      const k = lerp(0.82, 1.0, t) * (pp.y < 0.4 ? 0.85 : 1);
      return [PALM_BARK[0] * k, PALM_BARK[1] * k, PALM_BARK[2] * k];
    };
    addTube(bark, trunk, radial, stride, barkColour, 1 / RING_SPACING, 2, true);

    // Crownshaft: the swollen leaf-base knot where the fronds spring from.
    const shaft: Branch = {
      pts: [top.clone().addScaledVector(topDir, -0.35), top.clone(), crownBase.clone()],
      rad: [params.topRadius * 1.05, params.topRadius * 1.5, params.topRadius * 0.9],
      ts: [0, 0.6, 1],
      len: 0.6,
      level: 0,
    };
    addTube(bark, shaft, radial, 1, () => [0.62, 0.6, 0.4], 2, 2, true);

    const segs = [8, 4, 2][lod];
    const all = [...fronds, ...dead];
    all.forEach((f, i) => {
      if (lod === 2 && i >= fronds.length) return;
      if (lod === 2 && i % 2 === 1 && fronds.length > 9) return;
      const spine = resample(spines[i], segs);
      // Wider than the tile's own aspect: real leaflets reach nearly a metre each
      // side, and the stretch fattens them, which is what reads at range.
      const half = ((f.len * tileAspect(f.tile)) / 2) * 1.7 * (lod === 2 ? 1.15 : 1);
      addStrip(
        leaf,
        spine,
        half,
        f.tile,
        'u',
        (s) => lerp(0.35, 0.85, s) * (i >= fronds.length ? 0.3 : 1),
        crown,
        f.tint,
        1.15,
      );
    });

    if (lod === 0) {
      for (const c of coconutCentres) {
        const ico = new THREE.IcosahedronGeometry(0.14, 1);
        const pa = ico.getAttribute('position') as THREE.BufferAttribute;
        const na = ico.getAttribute('normal') as THREE.BufferAttribute;
        const base = bark.pos.length / 3;
        const shade: RGB = rng() < 0.6 ? [0.42, 0.5, 0.18] : [0.5, 0.38, 0.2];
        for (let k = 0; k < pa.count; k++) {
          const p = new V3(pa.getX(k), pa.getY(k) * 1.15, pa.getZ(k)).add(c);
          const nn = new V3(na.getX(k), na.getY(k), na.getZ(k));
          bark.vertex(p, nn, 0.25, 0.5, shade);
        }
        for (let k = 0; k < pa.count; k++) bark.idx.push(base + k);
        ico.dispose();
      }
    }
    levels.push({ leaf: leaf.build(), bark: bark.build() });
  }
  return levels;
}

// ======================================================== broadleaf trees

export interface TreeSpecies {
  trunk: { r: number; top: number; lean: number; segs: number };
  limbs: { n: number; y: [number, number]; tilt: [number, number]; len: [number, number]; r: number } & Bend & { segs: number };
  branches: { n: number; t: [number, number]; angle: [number, number]; len: [number, number] } & Bend & { segs: number };
  twigs: { n: number; t: [number, number]; angle: [number, number]; len: [number, number] } & Bend & { segs: number };
  cards: { perTwig: number; scale: [number, number]; droop: number; size: number };
  /** Fraction of leaf cards drawn from the flower tile instead. */
  flowers: number;
  /** Hanging vines per limb, as a probability. */
  vines: number;
  tints: RGB[];
  bark: RGB;
}

function growTree(sp: TreeSpecies, seed: number): Branch[] {
  const rng = mulberry32(seed);
  const R = (a: number, b: number): number => a + (b - a) * rng();
  const branches: Branch[] = [];
  const still: Bend = { up: 0, out: 0, droop: 0, jitter: 0.025 };

  const T = sp.trunk;
  const leanAz = R(0, Math.PI * 2);
  const trunkDir = new V3(Math.cos(leanAz) * T.lean, 1, Math.sin(leanAz) * T.lean).normalize();
  const schedule = [0, 0.03, 0.08];
  for (let i = 1; i <= T.segs; i++) schedule.push(0.08 + (0.92 * i) / T.segs);
  const trunk = growBranch(new V3(0, 0, 0), trunkDir, T.top, T.r, T.r * 0.35, T.segs, still, rng, 0, schedule);
  // Root flare: tropical trunks buttress hard at the ground.
  trunk.rad = trunk.rad.map((r, i) => r * (1 + 1.1 * Math.exp(-(trunk.ts[i] * T.top) / 0.45)));
  branches.push(trunk);

  const L = sp.limbs;
  const B = sp.branches;
  const W = sp.twigs;
  let az = R(0, Math.PI * 2);
  for (let i = 0; i < L.n; i++) {
    const f = L.n > 1 ? i / (L.n - 1) : 0;
    const y = lerp(L.y[0], L.y[1], f) + R(-0.25, 0.25);
    const base = along(trunk, Math.min(0.98, y / T.top));
    az += 2.39996 + R(-0.35, 0.35);
    const tilt = lerp(L.tilt[0], L.tilt[1], f) + R(-0.08, 0.08);
    const dir = new V3(Math.sin(tilt) * Math.cos(az), Math.cos(tilt), Math.sin(tilt) * Math.sin(az));
    const len = lerp(L.len[0], L.len[1], f) * R(0.88, 1.1);
    const limb = growBranch(base.p, dir, len, base.r * L.r, 0.04, L.segs, L, rng, 1);
    branches.push(limb);

    const nb = B.n + (rng() < 0.5 ? 1 : 0);
    let baz = R(0, Math.PI * 2);
    for (let j = 0; j < nb; j++) {
      const bt = lerp(B.t[0], B.t[1], (j + R(0.15, 0.85)) / nb);
      const a = along(limb, bt);
      baz += 2.39996 + R(-0.4, 0.4);
      const bdir = deflect(a.d, R(B.angle[0], B.angle[1]), baz);
      const outw = new V3(a.p.x, 0, a.p.z).normalize();
      if (bdir.dot(outw) < -0.15) bdir.addScaledVector(outw, 0.7).normalize();
      const blen = len * lerp(B.len[0], B.len[1], bt) * R(0.85, 1.12);
      const br = growBranch(a.p, bdir, blen, Math.max(0.025, a.r * 0.62), 0.015, B.segs, B, rng, 2);
      branches.push(br);

      const nw = W.n + (rng() < 0.4 ? 1 : 0);
      let taz = R(0, Math.PI * 2);
      for (let k = 0; k < nw; k++) {
        const tt = lerp(W.t[0], W.t[1], (k + R(0.2, 0.8)) / nw);
        const ta = along(br, tt);
        taz += 2.39996 + R(-0.5, 0.5);
        const tdir = deflect(ta.d, R(W.angle[0], W.angle[1]), taz);
        const tw = growBranch(
          ta.p,
          tdir,
          lerp(W.len[0], W.len[1], tt) * R(0.8, 1.2),
          Math.max(0.012, ta.r * 0.7),
          0.006,
          W.segs,
          W,
          rng,
          3,
        );
        branches.push(tw);
      }
    }
  }
  return branches;
}

interface CardSpec {
  at: Vec3;
  up: Vec3;
  face: Vec3;
  w: number;
  h: number;
  tile: TileName;
  tint: RGB;
  mirror: boolean;
}

export function buildTree(sp: TreeSpecies, seed: number): PlantLevel[] {
  const branches = growTree(sp, seed);
  const rng = mulberry32(seed ^ 0x5bd1e995);
  const R = (a: number, b: number): number => a + (b - a) * rng();
  const tintFor = (): RGB => {
    const t = sp.tints[Math.floor(rng() * sp.tints.length)];
    const k = R(0.88, 1.1);
    return [t[0] * k, t[1] * k, t[2] * k];
  };

  const tips: Vec3[] = [];
  for (const b of branches) if (b.level === 3) tips.push(b.pts[b.pts.length - 1]);
  const rough = makeCrown(tips);
  const outward = (p: Vec3, c: Crown) =>
    new V3(p.x - c.C.x, (p.y - c.C.y) * 0.6, p.z - c.C.z).normalize();
  const orient = (p: Vec3, d: Vec3, c: Crown) => {
    const o = outward(p, c);
    const up = d
      .clone()
      .multiplyScalar(0.5)
      .addScaledVector(o, 0.5)
      .add(new V3(R(-0.3, 0.3), R(-0.1, 0.4), R(-0.3, 0.3)));
    up.y -= sp.cards.droop;
    up.normalize();
    let face = o.clone().add(new V3(R(-0.9, 0.9), R(-0.4, 1.0), R(-0.9, 0.9))).normalize();
    face.addScaledVector(up, -face.dot(up));
    if (face.lengthSq() < 1e-4) face = new V3(0, 1, 0).cross(up);
    return { up, face: face.normalize() };
  };
  const leafTile = (): TileName => (rng() < sp.flowers ? 'flower' : 'broadleaf');

  // LOD0 cards on every twig and branch end; LOD1 bigger cards on branches.
  const cards0: CardSpec[] = [];
  for (const b of branches) {
    if (b.level !== 3) continue;
    for (let k = 0; k < sp.cards.perTwig; k++) {
      const t = lerp(0.35, 1.0, (k + R(0.3, 0.9)) / sp.cards.perTwig);
      const a = along(b, t);
      const { up, face } = orient(a.p, a.d, rough);
      const s = sp.cards.size * R(sp.cards.scale[0], sp.cards.scale[1]);
      cards0.push({ at: a.p, up, face, w: s, h: s, tile: leafTile(), tint: tintFor(), mirror: rng() < 0.5 });
    }
  }
  for (const b of branches) {
    if (b.level !== 2) continue;
    const a = along(b, 1);
    const { up, face } = orient(a.p, a.d, rough);
    const s = sp.cards.size * R(0.95, 1.15);
    cards0.push({ at: a.p, up, face, w: s, h: s, tile: leafTile(), tint: tintFor(), mirror: rng() < 0.5 });
  }
  const pts: Vec3[] = [];
  for (const c of cards0) {
    pts.push(c.at.clone().addScaledVector(c.up, c.h * 0.5));
    pts.push(c.at.clone().addScaledVector(c.up, c.h));
  }
  const crown = makeCrown(pts);

  const cards1: CardSpec[] = [];
  for (const b of branches) {
    if (b.level !== 2) continue;
    for (const t of [0.45, 0.75, 1.0]) {
      const a = along(b, Math.min(1, t + R(-0.06, 0.04)));
      const { up, face } = orient(a.p, a.d, crown);
      const at = a.p.clone().addScaledVector(outward(a.p, crown), 0.3);
      const s = sp.cards.size * R(1.6, 1.9);
      cards1.push({ at, up, face, w: s, h: s, tile: leafTile(), tint: tintFor(), mirror: rng() < 0.5 });
    }
  }

  // Shell of whole-crown clump cards, kept at every level so the canopy mass
  // never changes shape as the levels hand over.
  const shell: CardSpec[] = [];
  {
    const n = 22;
    const meanR = (crown.R.x + crown.R.y + crown.R.z) / 3;
    const s = Math.max(meanR * 1.15, 2.4);
    for (let i = 0; i < n; i++) {
      const y = 1 - ((i + 0.5) / n) * 1.6;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const ph = i * 2.39996 + R(-0.2, 0.2);
      const dir = new V3(Math.cos(ph) * r, y, Math.sin(ph) * r);
      const centre = crown.C.clone().add(dir.clone().multiply(crown.R).multiplyScalar(0.6));
      const face = dir.clone().add(new V3(R(-0.25, 0.25), R(-0.25, 0.25), R(-0.25, 0.25))).normalize();
      let up = new V3(0, 1, 0).addScaledVector(face, -face.y);
      if (up.lengthSq() < 1e-3) up = new V3(1, 0, 0);
      up.normalize().applyAxisAngle(face, R(-0.5, 0.5));
      const at = centre.clone().addScaledVector(up, -s * 0.5);
      shell.push({ at, up, face, w: s, h: s, tile: 'clump', tint: tintFor(), mirror: rng() < 0.5 });
    }
  }

  // Vines hanging from the limbs.
  const vines: CardSpec[] = [];
  for (const b of branches) {
    if (b.level !== 1 || rng() > sp.vines) continue;
    const a = along(b, R(0.45, 0.9));
    const len = R(2.2, 4.2);
    const face = new V3(R(-1, 1), 0, R(-1, 1)).normalize();
    vines.push({
      at: a.p.clone().add(new V3(0, -len, 0)),
      up: new V3(0, 1, 0),
      face,
      w: len * (128 - 12) / (256 - 12),
      h: len,
      tile: 'vine',
      tint: [0.9, 0.95, 0.85],
      mirror: rng() < 0.5,
    });
  }

  const levels: PlantLevel[] = [];
  for (let lod = 0; lod < 3; lod++) {
    const leaf = new GeoBuilder();
    const bark = new GeoBuilder();
    const barkColour = (p: Vec3): RGB => shadeColour(sp.bark, crown.barkAO(p));
    for (const b of branches) {
      if (lod === 0 && b.level <= 2) addTube(bark, b, [7, 4, 3][b.level], [1, 2, 4][b.level], barkColour, 0.8, 2);
      else if (lod === 1 && b.level <= 1) addTube(bark, b, b.level === 0 ? 5 : 3, b.level === 0 ? 2 : 6, barkColour, 0.8, 2);
      else if (lod === 2 && b.level === 0) addTube(bark, b, 4, Math.max(1, b.pts.length - 1), barkColour, 0.8, 2);
    }
    const cards = lod === 0 ? [...cards0, ...vines] : lod === 1 ? [...cards1, ...vines.slice(0, 2)] : [];
    for (const c of [...cards, ...shell]) {
      addCard(leaf, c.at, c.up, c.face, c.w, c.h, c.tile, c.mirror, crown, c.tint);
    }
    levels.push({ leaf: leaf.build(), bark: bark.build() });
  }
  return levels;
}

// ============================================================ understorey

/**
 * A plant made of bent leaf strips radiating from its base: fern, aroid bush,
 * banana. `stem` adds a pseudo-stem the leaves spring from.
 */
export interface RosetteParams {
  tile: TileName;
  leaves: number;
  length: [number, number];
  /** Elevation of each leaf at its base, radians: young (centre) to old (outer). */
  elevation: [number, number];
  droop: number;
  /** Width as a fraction of what the tile's aspect implies. */
  widthScale: number;
  fold: number;
  stem?: { height: number; radius: number; colour: RGB };
  tints: RGB[];
}

export function buildRosette(p: RosetteParams, seed: number): PlantLevel[] {
  const rng = mulberry32(seed);
  const R = (a: number, b: number): number => a + (b - a) * rng();
  const top = new V3(0, p.stem ? p.stem.height : 0.05, 0);
  interface Leaf {
    spine: Vec3[];
    half: number;
    tint: RGB;
  }
  const leaves: Leaf[] = [];
  let az = R(0, Math.PI * 2);
  for (let i = 0; i < p.leaves; i++) {
    const age = i / Math.max(1, p.leaves - 1);
    az += 2.39996 + R(-0.4, 0.4);
    const el = lerp(p.elevation[0], p.elevation[1], age) + R(-0.1, 0.1);
    const len = R(p.length[0], p.length[1]) * lerp(0.85, 1.05, age);
    const dir = new V3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
    const start = top.clone().addScaledVector(new V3(Math.cos(az), 0, Math.sin(az)), R(0.02, 0.08));
    const t = p.tints[Math.floor(rng() * p.tints.length)];
    const k = R(0.88, 1.1);
    leaves.push({
      spine: bentSpine(start, dir, len, 8, p.droop * lerp(0.6, 1.4, age), 0.03, rng),
      half: ((len * tileAspect(p.tile)) / 2) * p.widthScale,
      tint: [t[0] * k, t[1] * k, t[2] * k],
    });
  }
  const pts: Vec3[] = [new V3(0, 0, 0)];
  for (const l of leaves) pts.push(...l.spine);
  const crown = makeCrown(pts, 0.5, 0.4);

  const levels: PlantLevel[] = [];
  for (let lod = 0; lod < 3; lod++) {
    const leaf = new GeoBuilder();
    const bark = new GeoBuilder();
    const segs = [5, 3, 1][lod];
    leaves.forEach((l, i) => {
      if (lod === 2 && i % 2 === 1) return;
      addStrip(leaf, resample(l.spine, segs), l.half, p.tile, 'v', () => p.fold, crown, l.tint, 1.1, lod === 2 ? 2 : 3);
    });
    if (p.stem) {
      const s = p.stem;
      const stem: Branch = {
        pts: [new V3(0, 0, 0), new V3(0.02, s.height * 0.5, 0), new V3(0, s.height, 0.02)],
        rad: [s.radius * 1.3, s.radius, s.radius * 0.8],
        ts: [0, 0.5, 1],
        len: s.height,
        level: 0,
      };
      addTube(bark, stem, [6, 4, 3][lod], 1, () => s.colour, 0.5, 1, true);
    }
    levels.push({ leaf: leaf.build(), bark: bark.build() });
  }
  return levels;
}

/**
 * A dome of crossed cards: shrubs and grass tufts. `topTile` puts a few cards
 * of another tile (flowers) on the upper half.
 */
export interface ShrubParams {
  tile: TileName;
  cards: number;
  size: [number, number];
  radius: number;
  height: number;
  /** 0 keeps every card upright, 1 lays the outer ones down. */
  splay: number;
  topTile?: TileName;
  topCards?: number;
  tints: RGB[];
}

export function buildShrub(p: ShrubParams, seed: number): PlantLevel[] {
  const rng = mulberry32(seed);
  const R = (a: number, b: number): number => a + (b - a) * rng();
  const specs: CardSpec[] = [];
  const add = (tile: TileName, upper: boolean) => {
    const az = R(0, Math.PI * 2);
    const rr = Math.sqrt(rng()) * p.radius;
    const s = R(p.size[0], p.size[1]);
    const at = new V3(Math.cos(az) * rr, upper ? R(0.35, 0.6) * p.height : R(0, 0.15) * p.height, Math.sin(az) * rr);
    const outDir = new V3(Math.cos(az), 0, Math.sin(az));
    const tilt = p.splay * (rr / Math.max(0.01, p.radius)) * R(0.6, 1.1) + R(0, 0.2);
    const up = new V3(0, 1, 0).multiplyScalar(Math.cos(tilt)).addScaledVector(outDir, Math.sin(tilt)).normalize();
    let face = new V3(R(-1, 1), R(-0.2, 0.3), R(-1, 1)).normalize();
    face.addScaledVector(up, -face.dot(up));
    if (face.lengthSq() < 1e-4) face = new V3(1, 0, 0);
    const t = p.tints[Math.floor(rng() * p.tints.length)];
    const k = R(0.85, 1.12);
    const w = s * ((TILES[tile].w - 12) / (TILES[tile].h - 12));
    specs.push({ at, up, face: face.normalize(), w, h: s, tile, tint: [t[0] * k, t[1] * k, t[2] * k], mirror: rng() < 0.5 });
  };
  for (let i = 0; i < p.cards; i++) add(p.tile, false);
  if (p.topTile) for (let i = 0; i < (p.topCards ?? 0); i++) add(p.topTile, true);
  const pts: Vec3[] = [];
  for (const c of specs) pts.push(c.at, c.at.clone().addScaledVector(c.up, c.h));
  const crown = makeCrown(pts, 0.45, 0.35);
  const levels: PlantLevel[] = [];
  for (let lod = 0; lod < 3; lod++) {
    const leaf = new GeoBuilder();
    const keep = lod === 0 ? 1 : lod === 1 ? 2 : 3;
    specs.forEach((c, i) => {
      if (i % keep !== 0) return;
      const grow = lod === 0 ? 1 : 1 + 0.25 * (keep - 1);
      addCard(leaf, c.at, c.up, c.face, c.w * grow, c.h * grow, c.tile, c.mirror, crown, c.tint);
    });
    levels.push({ leaf: leaf.build(), bark: null });
  }
  return levels;
}

