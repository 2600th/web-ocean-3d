import * as THREE from 'three/webgpu';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

/**
 * Geometry kit for the procedural period props.
 *
 * Every prop is written into one `MeshKit` per material and each kit becomes a
 * single geometry, so a forty-part sea chest is three draws (wood, iron, gold)
 * and the material count — which is what the boot compile pays for — is the
 * same eleven whatever is built. UVs are written in **metres**, with `v` along
 * the grain or the length of a part and `u` across it, so a material can draw
 * wood grain, rope lay or masonry courses at a physical scale without knowing
 * which prop it is on. The vertex colour is a per-part tint, which is where the
 * seeded variety between staves, planks and stones lives.
 *
 * `sweep` and `tube` are ported from Noniv/abyssal-garden's `math.js`
 * (MIT, Copyright (c) 2026 Maksymilian Dendura): the box-section sweep with
 * hard edges and the round tube along a polyline. They are rewritten to emit
 * normals, metric UVs and tints into this kit; the framing logic is theirs.
 */

export type Tint = readonly [number, number, number];
export const WHITE: Tint = [1, 1, 1];

const _v = new THREE.Vector3();
const _n = new THREE.Vector3();
const _nm = new THREE.Matrix3();

export class MeshKit {
  private readonly pos: number[] = [];
  private readonly nrm: number[] = [];
  private readonly uvs: number[] = [];
  private readonly col: number[] = [];
  private readonly idx: number[] = [];
  /** Transform applied to every vertex written through `vertex`; see `withMatrix`. */
  private matrix: THREE.Matrix4 | null = null;

  get vertexCount(): number {
    return this.pos.length / 3;
  }

  get empty(): boolean {
    return this.idx.length === 0;
  }

  /** Runs `body` with every vertex it writes transformed by `m` (composed with any outer one). */
  withMatrix(m: THREE.Matrix4, body: () => void): void {
    const outer = this.matrix;
    this.matrix = outer ? outer.clone().multiply(m) : m;
    try {
      body();
    } finally {
      this.matrix = outer;
    }
  }

  vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, v: number, tint: Tint): number {
    _v.set(x, y, z);
    _n.set(nx, ny, nz);
    if (this.matrix) {
      _v.applyMatrix4(this.matrix);
      _n.applyMatrix3(_nm.getNormalMatrix(this.matrix)).normalize();
    }
    this.pos.push(_v.x, _v.y, _v.z);
    this.nrm.push(_n.x, _n.y, _n.z);
    this.uvs.push(u, v);
    this.col.push(tint[0], tint[1], tint[2]);
    return this.pos.length / 3 - 1;
  }

  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }

  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  /**
   * Appends a three.js geometry transformed by `m`. Its own UVs are kept when
   * `keepUv`; otherwise UVs are projected in metres along the dominant axis of
   * each vertex normal, with `grain` the local axis the grain runs along.
   */
  add(g: THREE.BufferGeometry, m: THREE.Matrix4, tint: Tint, grain: 'x' | 'y' | 'z' | 'keep' = 'y', uvOffset = 0): void {
    const p = g.getAttribute('position') as THREE.BufferAttribute;
    const n = g.getAttribute('normal') as THREE.BufferAttribute | undefined;
    const t = g.getAttribute('uv') as THREE.BufferAttribute | undefined;
    const base = this.vertexCount;
    const full = this.matrix ? this.matrix.clone().multiply(m) : m;
    _nm.getNormalMatrix(full);
    const local = new THREE.Vector3();
    const ln = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      local.fromBufferAttribute(p, i);
      if (n) ln.fromBufferAttribute(n, i);
      else ln.set(0, 1, 0);
      let u = 0;
      let v = 0;
      if (grain === 'keep') {
        u = t ? t.getX(i) : 0;
        v = t ? t.getY(i) : 0;
      } else {
        [u, v] = projectUv(local, ln, grain);
      }
      _v.copy(local).applyMatrix4(full);
      _n.copy(ln).applyMatrix3(_nm).normalize();
      this.pos.push(_v.x, _v.y, _v.z);
      this.nrm.push(_n.x, _n.y, _n.z);
      this.uvs.push(u + uvOffset, v + uvOffset * 0.37);
      this.col.push(tint[0], tint[1], tint[2]);
    }
    if (g.index) for (let i = 0; i < g.index.count; i++) this.idx.push(base + g.index.getX(i));
    else for (let i = 0; i < p.count; i++) this.idx.push(base + i);
  }

  build(): THREE.BufferGeometry | null {
    if (this.empty) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    const count = this.vertexCount;
    g.setIndex(count > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    g.computeBoundingBox();
    return g;
  }
}

/** Metric UV for a vertex from its dominant normal axis; `v` follows `grain` when the face allows it. */
function projectUv(p: THREE.Vector3, n: THREE.Vector3, grain: 'x' | 'y' | 'z'): [number, number] {
  const ax = Math.abs(n.x);
  const ay = Math.abs(n.y);
  const az = Math.abs(n.z);
  // The two in-plane axes of the face, then pick which of them is `v`.
  let a: 'x' | 'y' | 'z';
  let b: 'x' | 'y' | 'z';
  if (ax >= ay && ax >= az) [a, b] = ['z', 'y'];
  else if (ay >= az) [a, b] = ['x', 'z'];
  else [a, b] = ['x', 'y'];
  if (a === grain) [a, b] = [b, a];
  return [p[a], p[b]];
}

/** Material slots every prop draws from. One `MeshKit` per slot, one draw per slot used. */
export type Slot =
  | 'wood'
  | 'iron'
  | 'gunIron'
  | 'masonry'
  | 'lead'
  | 'shingle'
  | 'beaconGlass'
  | 'lanternGlass'
  | 'flame'
  | 'gold'
  | 'gem'
  | 'stoneware'
  | 'bottle'
  | 'rope'
  | 'shell';

/** Accumulates one prop's parts by material slot. */
export class PropBuilder {
  readonly kits = new Map<Slot, MeshKit>();

  kit(slot: Slot): MeshKit {
    let k = this.kits.get(slot);
    if (!k) {
      k = new MeshKit();
      this.kits.set(slot, k);
    }
    return k;
  }

  build(): Map<Slot, THREE.BufferGeometry> {
    const out = new Map<Slot, THREE.BufferGeometry>();
    for (const [slot, kit] of this.kits) {
      const g = kit.build();
      if (g) out.set(slot, g);
    }
    return out;
  }
}

// ------------------------------------------------------------------ helpers

export const M = {
  /** Translation, Euler XYZ rotation (radians), uniform or per-axis scale. */
  at(x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1): THREE.Matrix4 {
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ'));
    const sc = typeof s === 'number' ? new THREE.Vector3(s, s, s) : new THREE.Vector3(...s);
    return new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), q, sc);
  },
  identity(): THREE.Matrix4 {
    return new THREE.Matrix4();
  },
};

/** A box, optionally with rounded edges, grain along `grain`, centred on the origin. */
export function box(
  kit: MeshKit,
  w: number,
  h: number,
  d: number,
  m: THREE.Matrix4,
  tint: Tint,
  grain: 'x' | 'y' | 'z' = 'x',
  bevel = 0,
  uvOffset = 0,
): void {
  const g =
    bevel > 0
      ? new RoundedBoxGeometry(w, h, d, 1, Math.min(bevel, w * 0.49, h * 0.49, d * 0.49))
      : new THREE.BoxGeometry(w, h, d);
  kit.add(g, m, tint, grain, uvOffset);
  g.dispose();
}

/** A cylinder along local y, centred. Metric UV: u around, v along. */
export function cylinder(
  kit: MeshKit,
  rTop: number,
  rBottom: number,
  h: number,
  sides: number,
  m: THREE.Matrix4,
  tint: Tint,
  caps = true,
  uvOffset = 0,
): void {
  const profile: [number, number][] = [];
  if (caps) profile.push([0, -h / 2]);
  profile.push([rBottom, -h / 2], [rTop, h / 2]);
  if (caps) profile.push([0, h / 2]);
  lathe(kit, profile, sides, m, tint, { hard: caps, uvOffset });
}

export interface LatheOptions {
  /** Start angle and angular span, radians. Defaults to a full turn. */
  start?: number;
  span?: number;
  /** Split the normals at every profile vertex (hard rings), for turned mouldings and caps. */
  hard?: boolean;
  /** Per-vertex radius multiplier, for staves and lumps. (angle, profile index) -> factor. */
  radius?: (angle: number, k: number) => number;
  uvOffset?: number;
}

/**
 * Revolves a profile of [radius, y] pairs about local y.
 *
 * Normals are taken from the profile's own slope rather than from the
 * triangles, so a turned shape shades smooth around and stays crisp where the
 * profile turns a corner (with `hard`). UV `u` is metres around at each ring's
 * radius and `v` is metres of profile arc.
 */
export function lathe(kit: MeshKit, profile: readonly [number, number][], segments: number, m: THREE.Matrix4, tint: Tint, options: LatheOptions = {}): void {
  const start = options.start ?? 0;
  const span = options.span ?? Math.PI * 2;
  const radius = options.radius;
  const off = options.uvOffset ?? 0;
  kit.withMatrix(m, () => {
    // Arc length along the profile for v.
    const arc: number[] = [0];
    for (let k = 1; k < profile.length; k++) {
      arc.push(arc[k - 1] + Math.hypot(profile[k][0] - profile[k - 1][0], profile[k][1] - profile[k - 1][1]));
    }
    const segNormal = (k: number): [number, number] => {
      const a = profile[k];
      const b = profile[k + 1];
      const dr = b[0] - a[0];
      const dy = b[1] - a[1];
      const l = Math.hypot(dr, dy) || 1;
      // Outward normal of the segment in the (r, y) plane.
      return [dy / l, -dr / l];
    };
    const rings = (k0: number, k1: number, nAt: (k: number) => [number, number]): void => {
      const rows: number[][] = [];
      for (let k = k0; k <= k1; k++) {
        const [nr, ny] = nAt(k);
        const row: number[] = [];
        const r0 = profile[k][0];
        for (let s = 0; s <= segments; s++) {
          const a = start + (s / segments) * span;
          const f = radius ? radius(a, k) : 1;
          const r = r0 * f;
          const c = Math.cos(a);
          const sn = Math.sin(a);
          row.push(kit.vertex(c * r, profile[k][1], sn * r, c * nr, ny, sn * nr, a * Math.max(r0, 0.02) + off, arc[k] + off * 0.37, tint));
        }
        rows.push(row);
      }
      for (let k = 0; k < rows.length - 1; k++) {
        for (let s = 0; s < segments; s++) kit.quad(rows[k][s], rows[k + 1][s], rows[k + 1][s + 1], rows[k][s + 1]);
      }
    };
    if (options.hard) {
      for (let k = 0; k < profile.length - 1; k++) {
        const n = segNormal(k);
        rings(k, k + 1, () => n);
      }
    } else {
      rings(0, profile.length - 1, (k) => {
        const a = k > 0 ? segNormal(k - 1) : segNormal(k);
        const b = k < profile.length - 1 ? segNormal(k) : segNormal(k - 1);
        const nr = a[0] + b[0];
        const ny = a[1] + b[1];
        const l = Math.hypot(nr, ny) || 1;
        return [nr / l, ny / l];
      });
    }

  });
}

type Vec = readonly [number, number, number];

/**
 * Box-section sweep along a polyline: width `w` across, `h` along `up`.
 * Hard edges (each face has its own vertices). From abyssal-garden `Mesh.sweep`.
 */
export function sweep(
  kit: MeshKit,
  pts: readonly Vec[],
  w: number | ((k: number) => number),
  h: number | ((k: number) => number),
  up: (p: Vec, k: number) => Vec,
  m: THREE.Matrix4,
  tint: Tint,
  caps = true,
): void {
  kit.withMatrix(m, () => {
    const rings: { c: number[][]; n: number[][] }[] = [];
    let along = 0;
    const alongs: number[] = [];
    for (let k = 0; k < pts.length; k++) {
      if (k > 0) along += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2]);
      alongs.push(along);
      const p = pts[k];
      const a = pts[Math.max(k - 1, 0)];
      const b = pts[Math.min(k + 1, pts.length - 1)];
      let tx = b[0] - a[0];
      let ty = b[1] - a[1];
      let tz = b[2] - a[2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl;
      ty /= tl;
      tz /= tl;
      const u0 = up(p, k);
      let sx = u0[1] * tz - u0[2] * ty;
      let sy = u0[2] * tx - u0[0] * tz;
      let sz = u0[0] * ty - u0[1] * tx;
      const sl = Math.hypot(sx, sy, sz) || 1;
      sx /= sl;
      sy /= sl;
      sz /= sl;
      const ux = ty * sz - tz * sy;
      const uy = tz * sx - tx * sz;
      const uz = tx * sy - ty * sx;
      const ww = typeof w === 'function' ? w(k) : w;
      const hh = typeof h === 'function' ? h(k) : h;
      const corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([e, f]) => [p[0] + (sx * e * ww) / 2 + (ux * f * hh) / 2, p[1] + (sy * e * ww) / 2 + (uy * f * hh) / 2, p[2] + (sz * e * ww) / 2 + (uz * f * hh) / 2]);
      // Face normals in corner order: bottom (-up), right (+side), top (+up), left (-side).
      const normals = [
        [-ux, -uy, -uz],
        [sx, sy, sz],
        [ux, uy, uz],
        [-sx, -sy, -sz],
      ];
      rings.push({ c: corners, n: normals });
    }
    for (let k = 0; k < rings.length - 1; k++) {
      const r0 = rings[k];
      const r1 = rings[k + 1];
      for (let s = 0; s < 4; s++) {
        const s2 = (s + 1) % 4;
        const n0 = r0.n[s];
        const n1 = r1.n[s];
        const across = s % 2 === 0 ? 0 : 0.5;
        const a = kit.vertex(r0.c[s][0], r0.c[s][1], r0.c[s][2], n0[0], n0[1], n0[2], across, alongs[k], tint);
        const b = kit.vertex(r0.c[s2][0], r0.c[s2][1], r0.c[s2][2], n0[0], n0[1], n0[2], across + 0.05, alongs[k], tint);
        const c = kit.vertex(r1.c[s2][0], r1.c[s2][1], r1.c[s2][2], n1[0], n1[1], n1[2], across + 0.05, alongs[k + 1], tint);
        const d = kit.vertex(r1.c[s][0], r1.c[s][1], r1.c[s][2], n1[0], n1[1], n1[2], across, alongs[k + 1], tint);
        kit.quad(a, d, c, b);
      }
    }
    if (caps && rings.length > 1) {
      for (const [k, sign] of [
        [0, -1],
        [rings.length - 1, 1],
      ] as const) {
        const r = rings[k];
        const t = [pts[Math.min(k + 1, pts.length - 1)], pts[Math.max(k - 1, 0)]];
        let tx = (t[0][0] - t[1][0]) * -sign;
        let ty = (t[0][1] - t[1][1]) * -sign;
        let tz = (t[0][2] - t[1][2]) * -sign;
        if (k === 0) {
          tx = -(pts[1][0] - pts[0][0]);
          ty = -(pts[1][1] - pts[0][1]);
          tz = -(pts[1][2] - pts[0][2]);
        } else {
          tx = pts[k][0] - pts[k - 1][0];
          ty = pts[k][1] - pts[k - 1][1];
          tz = pts[k][2] - pts[k - 1][2];
        }
        const l = Math.hypot(tx, ty, tz) || 1;
        const q = r.c.map((cc, i) => kit.vertex(cc[0], cc[1], cc[2], tx / l, ty / l, tz / l, i & 1 ? 0.05 : 0, i & 2 ? 0.05 : 0, tint));
        if (sign < 0) kit.quad(q[0], q[1], q[2], q[3]);
        else kit.quad(q[3], q[2], q[1], q[0]);
      }
    }
  });
}

/** Round tube along a polyline with a parallel-transported frame. From abyssal-garden `Mesh.tube`. */
export function tube(
  kit: MeshKit,
  pts: readonly Vec[],
  rad: number | ((k: number) => number),
  sides: number,
  m: THREE.Matrix4,
  tint: Tint,
  capEnds = true,
): void {
  kit.withMatrix(m, () => {
    const rings: number[][] = [];
    let ux = 0.31;
    let uy = 0.9;
    let uz = 0.3;
    let along = 0;
    const centres: number[][] = [];
    for (let k = 0; k < pts.length; k++) {
      const p = pts[k];
      if (k > 0) along += Math.hypot(p[0] - pts[k - 1][0], p[1] - pts[k - 1][1], p[2] - pts[k - 1][2]);
      const a = pts[Math.max(k - 1, 0)];
      const b = pts[Math.min(k + 1, pts.length - 1)];
      let tx = b[0] - a[0];
      let ty = b[1] - a[1];
      let tz = b[2] - a[2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl;
      ty /= tl;
      tz /= tl;
      let sx = uy * tz - uz * ty;
      let sy = uz * tx - ux * tz;
      let sz = ux * ty - uy * tx;
      const sl = Math.hypot(sx, sy, sz) || 1;
      sx /= sl;
      sy /= sl;
      sz /= sl;
      ux = ty * sz - tz * sy;
      uy = tz * sx - tx * sz;
      uz = tx * sy - ty * sx;
      const r = typeof rad === 'function' ? rad(k) : rad;
      const ring: number[] = [];
      for (let s = 0; s <= sides; s++) {
        const an = (s / sides) * Math.PI * 2;
        const nx = sx * Math.cos(an) + ux * Math.sin(an);
        const ny = sy * Math.cos(an) + uy * Math.sin(an);
        const nz = sz * Math.cos(an) + uz * Math.sin(an);
        ring.push(kit.vertex(p[0] + nx * r, p[1] + ny * r, p[2] + nz * r, nx, ny, nz, (s / sides) * Math.PI * 2 * Math.max(r, 0.01), along, tint));
      }
      rings.push(ring);
      centres.push([p[0], p[1], p[2], tx, ty, tz, r]);
    }
    for (let k = 0; k < rings.length - 1; k++) {
      for (let s = 0; s < sides; s++) kit.quad(rings[k][s], rings[k][s + 1], rings[k + 1][s + 1], rings[k + 1][s]);
    }
    if (capEnds) {
      for (const k of [0, pts.length - 1]) {
        const c = centres[k];
        const sign = k === 0 ? -1 : 1;
        const centre = kit.vertex(c[0], c[1], c[2], c[3] * sign, c[4] * sign, c[5] * sign, 0, 0, tint);
        // Shares the tube's rim vertices: the caps here are rope ends and rod
        // ends a few centimetres across, where a split normal is not visible.
        const ring = rings[k];
        for (let s = 0; s < sides; s++) {
          if (sign > 0) kit.tri(ring[s], ring[s + 1], centre);
          else kit.tri(ring[s + 1], ring[s], centre);
        }
      }
    }
  });
}

/** Points on a circle arc in the local XY plane (for bands and hoops built with `sweep`). */
export function arcPoints(radius: number, a0: number, a1: number, n: number, z = 0): Vec[] {
  const out: Vec[] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    out.push([Math.cos(a) * radius, Math.sin(a) * radius, z]);
  }
  return out;
}

/** Seeded jitter of a tint: `amount` of brightness and a little hue drift. */
export function jitter(base: Tint, rng: () => number, amount = 0.12): Tint {
  const k = 1 + (rng() * 2 - 1) * amount;
  const warm = (rng() * 2 - 1) * amount * 0.35;
  return [base[0] * k * (1 + warm), base[1] * k, base[2] * k * (1 - warm)];
}

/**
 * A tint written in display (sRGB) values, returned linear.
 *
 * Every albedo in this library is authored this way because the first pass
 * was not: tints typed as linear numbers that *looked* like wood (0.34, 0.24,
 * 0.15) rendered as sRGB 0.62 — casks the colour of driftwood bleached for a
 * decade. Oak is about 0.4 in sRGB and 0.13 in linear, and the eye judges the
 * former.
 */
export function srgb(r: number, g: number, b: number): Tint {
  const f = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  return [f(r), f(g), f(b)];
}
