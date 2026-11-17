import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  cameraViewMatrix,
  float,
  mix,
  normalWorldGeometry,
  positionWorld,
  texture,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../../core/random';

/**
 * The stone the underwater geology is made of: CPU noise for sculpting at load,
 * a tileable detail texture for shading, the mesh helpers that turn a density
 * field or a displaced sphere into triangles, and the one material every rock,
 * boulder and outcrop is drawn with.
 *
 * The approach is abyssal-garden's — sculpt with a density field, mesh it with
 * naive surface nets, scatter clipped noise-displaced icospheres as the loose
 * rock, and shade with triplanar detail at several scales, strata, encrusting
 * life on the lit faces and darkness in the cavities. The icosphere chunk, the
 * surface-nets mesher and the layered-rock albedo below are adapted from
 * abyssal-garden, MIT, (c) 2026 Maksymilian Dendura (`source/src/math.js`,
 * `world.js` `meshChunk`, `shaders.js` `W_SOLID`); see THIRD_PARTY_NOTICES.md.
 * What differs is where it runs: abyssal-garden sculpts a 512 x 80 x 512 grid
 * in a compute shader, which the WebGL2 fallback of the time could not do, so
 * this sculpts small per-outcrop grids on the CPU instead and keeps the noise
 * cheap enough that the whole shelf costs about a tenth of a second at boot.
 * That constraint no longer applies.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

// ------------------------------------------------------------------ CPU noise

const NOISE_SEED = 0x51ab0c;

/**
 * Permutation-table value noise, not an integer hash per lattice corner.
 *
 * The outcrops evaluate a few hundred thousand noise samples at boot. A
 * multiply-xorshift hash per corner (abyssal-garden's `ihash`) costs about
 * three times what two byte-table lookups do in V8, and the field only has to
 * be deterministic, not cryptographic.
 */
const PERM = (() => {
  const random = mulberry32(NOISE_SEED);
  const base = new Uint8Array(256);
  for (let i = 0; i < 256; i++) base[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const t = base[i];
    base[i] = base[j];
    base[j] = t;
  }
  const p = new Uint8Array(512);
  for (let i = 0; i < 512; i++) p[i] = base[i & 255];
  return p;
})();

const VALUES = (() => {
  const random = mulberry32(NOISE_SEED ^ 0x9e3779b9);
  const v = new Float32Array(256);
  for (let i = 0; i < 256; i++) v[i] = random();
  return v;
})();

function lattice(ix: number, iy: number, iz: number): number {
  return VALUES[PERM[PERM[PERM[ix & 255] + (iy & 255)] + (iz & 255)]];
}

/** 3-D value noise in [0, 1], smoothstep-interpolated. */
export function vnoise3(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  let fx = x - ix;
  let fy = y - iy;
  let fz = z - iz;
  fx = fx * fx * (3 - 2 * fx);
  fy = fy * fy * (3 - 2 * fy);
  fz = fz * fz * (3 - 2 * fz);
  const a = lattice(ix, iy, iz);
  const b = lattice(ix + 1, iy, iz);
  const c = lattice(ix, iy + 1, iz);
  const d = lattice(ix + 1, iy + 1, iz);
  const e = lattice(ix, iy, iz + 1);
  const f = lattice(ix + 1, iy, iz + 1);
  const g = lattice(ix, iy + 1, iz + 1);
  const h = lattice(ix + 1, iy + 1, iz + 1);
  const ab = a + (b - a) * fx;
  const cd = c + (d - c) * fx;
  const ef = e + (f - e) * fx;
  const gh = g + (h - g) * fx;
  const lo = ab + (cd - ab) * fy;
  const hi = ef + (gh - ef) * fy;
  return lo + (hi - lo) * fz;
}

/** Fractal sum of `vnoise3`, normalised to [0, 1]; abyssal-garden's `fbm`. */
export function fbm3(x: number, y: number, z: number, octaves: number): number {
  let a = 0.5;
  let s = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    s += a * vnoise3(x, y, z);
    norm += a;
    x = x * 2.03 + 17.1;
    y = y * 2.03 + 31.7;
    z = z * 2.03 + 5.3;
    a *= 0.5;
  }
  return s / norm;
}

export function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Polynomial smooth max (iq); `k` is the blend width in metres. */
export function smax(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.max(a, b) + h * h * k * 0.25;
}

export function smin(a: number, b: number, k: number): number {
  return -smax(-a, -b, k);
}

// ------------------------------------------------------------- detail texture

const DETAIL_SIZE = 256;

/**
 * Tileable rock detail, 256² RGBA8, shared by every rock and by the seabed's
 * rubble.
 *
 *   rg  tangent-space normal of `b`, the bump the shading adds at each scale
 *   b   height: a fractal sum plus a cellular term, so a close-up reads as
 *       pitted, pebbly stone rather than as smooth noise
 *   a   a second, independent and lower-frequency field, for colour breakup
 *       (encrustation patches, pale and dark beds) that must not correlate with
 *       the bump
 *
 * abyssal-garden packs the same four channels into its `tD` texture and reads
 * it at four scales; this reads it at three. Built once and never disposed:
 * it is 256 KB, two systems hold it, and it lives as long as the page.
 */
let detailTexture: THREE.DataTexture | null = null;

export function rockDetailTexture(): THREE.DataTexture {
  if (detailTexture) return detailTexture;
  const N = DETAIL_SIZE;
  const random = mulberry32(NOISE_SEED ^ 0x1234567);

  // Periodic lattices, one per period, so every octave tiles at 256 px.
  const periodic = (period: number, seed: number) => {
    const r = mulberry32(seed);
    const v = new Float32Array(period * period);
    for (let i = 0; i < v.length; i++) v[i] = r();
    return (x: number, y: number): number => {
      const fx0 = (x / N) * period;
      const fy0 = (y / N) * period;
      const ix = Math.floor(fx0);
      const iy = Math.floor(fy0);
      let fx = fx0 - ix;
      let fy = fy0 - iy;
      fx = fx * fx * (3 - 2 * fx);
      fy = fy * fy * (3 - 2 * fy);
      const x0 = ((ix % period) + period) % period;
      const y0 = ((iy % period) + period) % period;
      const x1 = (x0 + 1) % period;
      const y1 = (y0 + 1) % period;
      const a = v[y0 * period + x0];
      const b = v[y0 * period + x1];
      const c = v[y1 * period + x0];
      const d = v[y1 * period + x1];
      return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
    };
  };
  const octavesH = [periodic(8, 11), periodic(16, 12), periodic(32, 13), periodic(64, 14)];
  const octavesC = [periodic(4, 21), periodic(8, 22), periodic(16, 23)];

  // Cellular term: 24 x 24 jittered points on a torus.
  const CELLS = 24;
  const feature = new Float32Array(CELLS * CELLS * 2);
  for (let i = 0; i < CELLS * CELLS; i++) {
    feature[i * 2] = random();
    feature[i * 2 + 1] = random();
  }
  const cellular = (x: number, y: number): number => {
    const gx = (x / N) * CELLS;
    const gy = (y / N) * CELLS;
    const ix = Math.floor(gx);
    const iy = Math.floor(gy);
    let f1 = 9;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const k = ((((cy % CELLS) + CELLS) % CELLS) * CELLS + (((cx % CELLS) + CELLS) % CELLS)) * 2;
        const px = cx + feature[k] - gx;
        const py = cy + feature[k + 1] - gy;
        f1 = Math.min(f1, px * px + py * py);
      }
    }
    return Math.sqrt(f1);
  };

  const height = new Float32Array(N * N);
  const cavity = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let h = 0;
      let a = 0.5;
      for (const o of octavesH) {
        h += o(x, y) * a;
        a *= 0.5;
      }
      h /= 0.9375;
      // Pebbles: domes in each cell, so the fine scale reads as grains and
      // pits rather than as a smooth undulation.
      const pebble = 1 - Math.min(1, cellular(x, y) * 1.25);
      height[y * N + x] = h * 0.62 + pebble * 0.38;
      let c = 0;
      a = 0.5;
      for (const o of octavesC) {
        c += o(x, y) * a;
        a *= 0.5;
      }
      cavity[y * N + x] = c / 0.875;
    }
  }

  const data = new Uint8Array(N * N * 4);
  const strength = 3.2;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const xl = height[y * N + ((x - 1 + N) % N)];
      const xr = height[y * N + ((x + 1) % N)];
      const yd = height[((y - 1 + N) % N) * N + x];
      const yu = height[((y + 1) % N) * N + x];
      let nx = (xl - xr) * strength;
      let ny = (yd - yu) * strength;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
      nx *= inv;
      ny *= inv;
      const i = (y * N + x) * 4;
      data[i] = Math.round((nx * 0.5 + 0.5) * 255);
      data[i + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      data[i + 2] = Math.round(Math.min(1, Math.max(0, height[y * N + x])) * 255);
      data[i + 3] = Math.round(Math.min(1, Math.max(0, cavity[y * N + x])) * 255);
    }
  }

  const map = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  // [underwater-look-3] Round 10: 4 -> 8. A boulder's flank a few metres off
  // is seen at a grazing angle, and four taps took its fine and grain scales
  // down to a blur the review called soft.
  map.anisotropy = 8;
  map.colorSpace = THREE.NoColorSpace;
  map.needsUpdate = true;
  detailTexture = map;
  return map;
}

// ---------------------------------------------------------------- loose rock

/**
 * Indexed icosphere, `sub` times subdivided, shared vertices. Vertices are unit
 * vectors before `shape` moves them. 20·4^sub triangles: 80, 320, 1280.
 */
function icosphere(sub: number): { pos: number[]; idx: number[] } {
  const t = (1 + Math.sqrt(5)) / 2;
  const pos: number[] = [];
  const push = (x: number, y: number, z: number): number => {
    const l = Math.hypot(x, y, z);
    pos.push(x / l, y / l, z / l);
    return pos.length / 3 - 1;
  };
  for (const [x, y, z] of [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
    [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ]) {
    push(x, y, z);
  }
  let faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let s = 0; s < sub; s++) {
    const cache = new Map<number, number>();
    const mid = (a: number, b: number): number => {
      const key = a < b ? a * 65536 + b : b * 65536 + a;
      let r = cache.get(key);
      if (r === undefined) {
        r = push(pos[a * 3] + pos[b * 3], pos[a * 3 + 1] + pos[b * 3 + 1], pos[a * 3 + 2] + pos[b * 3 + 2]);
        cache.set(key, r);
      }
      return r;
    };
    const next: number[][] = [];
    for (const [a, b, c] of faces) {
      const ab = mid(a, b);
      const bc = mid(b, c);
      const ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  return { pos, idx: faces.flat() };
}

export type ChunkKind = 'chunk' | 'boulder' | 'slab';

/**
 * A loose rock: an icosphere displaced by fractal noise and then cut by a few
 * random planes, which is what turns a lumpy ball into a fractured block with
 * flat faces and edges. abyssal-garden's `meshChunk`.
 *
 * The same seed at every subdivision gives the same rock, so the levels of an
 * LOD chain are one shape at three tessellations rather than three shapes —
 * the noise and the planes are functions of direction, not of the mesh.
 *
 * Unit-ish: the result spans roughly [-1, 1] horizontally and less vertically.
 */
export function buildChunkGeometry(seed: number, sub: number, kind: ChunkKind): THREE.BufferGeometry {
  const random = mulberry32(seed);
  const sd = random() * 50;
  const planeCount = kind === 'boulder' ? 2 : kind === 'slab' ? 5 : 4;
  const planes: number[][] = [];
  for (let i = 0; i < planeCount; i++) {
    let x = random() * 2 - 1;
    let y = random() * 2 - 1;
    let z = random() * 2 - 1;
    if (kind === 'slab' && i < 2) {
      // A slab is bedded: two of its cuts are the bedding planes, near level.
      x *= 0.25;
      z *= 0.25;
      y = i === 0 ? 1 : -1;
    }
    const l = Math.hypot(x, y, z) || 1;
    planes.push([x / l, y / l, z / l, kind === 'boulder' ? 0.86 + random() * 0.1 : 0.74 + random() * 0.22]);
  }
  const amp = kind === 'boulder' ? 0.55 : 0.9;
  const squash = kind === 'slab' ? 0.42 : kind === 'boulder' ? 0.72 : 0.8;

  const { pos, idx } = icosphere(sub);
  const out = new Float32Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i];
    const y = pos[i + 1];
    const z = pos[i + 2];
    const r =
      1 +
      (fbm3(x * 1.6 + sd, y * 1.6, z * 1.6, 2) - 0.5) * amp +
      (fbm3(x * 4 + sd, y * 4, z * 4, 2) - 0.5) * 0.25;
    let px = x * r;
    let py = y * r * squash;
    let pz = z * r;
    for (const p of planes) {
      const d = px * p[0] + py * p[1] + pz * p[2] - p[3] * (kind === 'slab' && Math.abs(p[1]) > 0.9 ? squash : 1);
      if (d > 0) {
        px -= p[0] * d;
        py -= p[1] * d;
        pz -= p[2] * d;
      }
    }
    out[i] = px;
    out[i + 1] = py;
    out[i + 2] = pz;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(out, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  // Loose rock has no baked occlusion; the attribute exists so the one rock
  // material reads the same vertex layout from both kinds of mesh.
  g.setAttribute('rockAo', new THREE.BufferAttribute(new Float32Array(out.length / 3).fill(1), 1));
  g.computeBoundingSphere();
  return g;
}

// -------------------------------------------------------------- surface nets

/**
 * Naive surface nets over a density grid (`D > 0` is solid), after
 * abyssal-garden's `surfaceNets`. One vertex per cell that the surface crosses,
 * placed at the mean of its edge crossings, and one quad per crossed edge.
 * Returns grid-space positions.
 *
 * Chosen over marching cubes for the same reason abyssal-garden chose it: a
 * third of the vertices, no case table, and it shares vertices between
 * neighbouring quads, so smooth normals from the density gradient come out
 * without a weld pass.
 */
export function surfaceNets(
  D: Float32Array,
  nx: number,
  ny: number,
  nz: number,
): { pos: Float32Array; idx: Uint32Array } {
  const pos: number[] = [];
  const idx: number[] = [];
  const sl = nx * ny;
  let cur = new Int32Array(sl);
  let prev = new Int32Array(sl);
  const quad = (s0: boolean, a: number, b: number, c: number, d: number): void => {
    if (s0) idx.push(a, b, c, a, c, d);
    else idx.push(a, c, b, a, d, c);
  };
  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      let i0 = nx * (y + ny * z);
      for (let x = 0; x < nx - 1; x++, i0++) {
        const d0 = D[i0];
        const d1 = D[i0 + 1];
        const d2 = D[i0 + nx];
        const d3 = D[i0 + nx + 1];
        const d4 = D[i0 + sl];
        const d5 = D[i0 + sl + 1];
        const d6 = D[i0 + sl + nx];
        const d7 = D[i0 + sl + nx + 1];
        const b0 = d0 > 0;
        const all = b0 && d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0 && d5 > 0 && d6 > 0 && d7 > 0;
        const none = !b0 && !(d1 > 0) && !(d2 > 0) && !(d3 > 0) && !(d4 > 0) && !(d5 > 0) && !(d6 > 0) && !(d7 > 0);
        if (all || none) continue;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let n = 0;
        const edge = (da: number, db: number, ox: number, oy: number, oz: number, axis: number): void => {
          if (da > 0 === db > 0) return;
          const t = da / (da - db);
          sx += ox + (axis === 0 ? t : 0);
          sy += oy + (axis === 1 ? t : 0);
          sz += oz + (axis === 2 ? t : 0);
          n++;
        };
        edge(d0, d1, 0, 0, 0, 0);
        edge(d2, d3, 0, 1, 0, 0);
        edge(d4, d5, 0, 0, 1, 0);
        edge(d6, d7, 0, 1, 1, 0);
        edge(d0, d2, 0, 0, 0, 1);
        edge(d1, d3, 1, 0, 0, 1);
        edge(d4, d6, 0, 0, 1, 1);
        edge(d5, d7, 1, 0, 1, 1);
        edge(d0, d4, 0, 0, 0, 2);
        edge(d1, d5, 1, 0, 0, 2);
        edge(d2, d6, 0, 1, 0, 2);
        edge(d3, d7, 1, 1, 0, 2);
        const vi = pos.length / 3;
        pos.push(x + sx / n, y + sy / n, z + sz / n);
        const ci = x + nx * y;
        cur[ci] = vi;
        if (d1 > 0 !== b0 && y > 0 && z > 0) quad(b0, vi, cur[ci - nx], prev[ci - nx], prev[ci]);
        if (d2 > 0 !== b0 && x > 0 && z > 0) quad(b0, vi, prev[ci], prev[ci - 1], cur[ci - 1]);
        if (d4 > 0 !== b0 && x > 0 && y > 0) quad(b0, vi, cur[ci - 1], cur[ci - 1 - nx], cur[ci - nx]);
      }
    }
    const t = cur;
    cur = prev;
    prev = t;
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}

/** Trilinear sampler over a density grid, in grid coordinates. */
export function gridSampler(D: Float32Array, nx: number, ny: number, nz: number) {
  return (x: number, y: number, z: number): number => {
    x = Math.min(Math.max(x, 0), nx - 1.001);
    y = Math.min(Math.max(y, 0), ny - 1.001);
    z = Math.min(Math.max(z, 0), nz - 1.001);
    const ix = x | 0;
    const iy = y | 0;
    const iz = z | 0;
    const fx = x - ix;
    const fy = y - iy;
    const fz = z - iz;
    const i = ix + nx * (iy + ny * iz);
    const sl = nx * ny;
    const a = D[i] + (D[i + 1] - D[i]) * fx;
    const b = D[i + nx] + (D[i + nx + 1] - D[i + nx]) * fx;
    const c = D[i + sl] + (D[i + sl + 1] - D[i + sl]) * fx;
    const d = D[i + sl + nx] + (D[i + sl + nx + 1] - D[i + sl + nx]) * fx;
    const ab = a + (b - a) * fy;
    const cd = c + (d - c) * fy;
    return ab + (cd - ab) * fz;
  };
}

// ------------------------------------------------------------------ material

/** World metres per texture repeat at each of the three detail scales. */
const SCALE_BROAD = 11;
const SCALE_MID = 2.6;
const SCALE_FINE = 0.62;
/** The fine scale fades out between these distances; past them it only crawls. */
// [underwater-look-3] Round 7: 6-26 m -> 10-38 m; see `SCALE_GRAIN`.
const FINE_FADE_NEAR = 10;
const FINE_FADE_FAR = 38;
/**
 * [underwater-look-3] Round 7. A fourth, finer scale, abyssal-garden's fourth
 * read of its detail texture: pits and grain about 5 cm across, faded out
 * between `GRAIN_FADE_NEAR` and `GRAIN_FADE_FAR` metres. The round-7 review
 * read the reef mound eight to fifteen metres off as "blurry and smeared":
 * with the fine scale fading from six metres there was nothing under the
 * half-metre features but the 2.6 m and 11 m fields, stretched across a face
 * the size of a house.
 */
const SCALE_GRAIN = 0.19;
const GRAIN_FADE_NEAR = 4;
const GRAIN_FADE_FAR = 22;
/**
 * [underwater-look-3] Round 10. Inside `NEAR_CRISP` metres (full at the near
 * end) the fine and grain scales' relief and the grain's pits gain by
 * `NEAR_CRISP_GAIN`: a boulder at arm's length was all broad bump, the
 * round-10 review's "blurry" foreground rock. No extra reads; the weights only.
 */
const NEAR_CRISP: readonly [number, number] = [3, 14];
const NEAR_CRISP_GAIN = 1.2;
/**
 * [underwater-look-3] Round 7. World metres of the macro brightness field, and
 * its range: whole faces a shade lighter or darker, so a large mass is not
 * one even value across its width (the review's "tiled" read).
 */
const MACRO_METRES = 37;
const MACRO_RANGE: readonly [number, number] = [0.72, 1.22];
/**
 * [underwater-look-3] Round 8. Bedding seams: how many beds the broad height
 * warps the strata by (was 1.6), the seam's half-width as a fraction of a bed,
 * and how dark its floor is (was 0.45, one-sided). See the strata in
 * `createRockMaterial`.
 */
const SEAM_WARP = 0.45;
const SEAM_WIDTH = 0.07;
const SEAM_DARK = 0.3;
/**
 * [underwater-look-3] Round 11. Grain, cracks and crust instead of spots.
 *
 * The detail texture's height carries a cellular term (`rockDetailTexture`'s
 * pebble domes, 24 to a tile), and the albedo was multiplied by that height
 * over a 0.22..1.52 range at the 2.6 m scale and the grain's pits over
 * 0.35..1.65: every boulder in mound-close came out in 11 cm dark dots, the
 * round-11 review's "leopard-spot pit pattern". The texture is shared with
 * the seabed's rubble, so it is left alone and the rock reads it differently:
 *
 * - tone: the broad height, the 2.6 m scale's *colour* field (fractal, no
 *   cells) and the fine height, over `TONE_RANGE`, so the stone has grain and
 *   mottling without a lattice of dots;
 * - the grain scale's colour field rather than its pits, over `GRAIN_RANGE`;
 * - cracks: where the 2.6 m colour field crosses its middle, a line
 *   `CRACK_WIDTH` wide in field units, broken up by the broad colour field
 *   and darkened by `CRACK_DARK`, inside the fine scale's fade;
 * - the mid scale's bump at `MID_BUMP` (was 0.65), and the ambient's cavities
 *   from the broad height and the mid colour field, so the domes no longer
 *   shade as pebbles either;
 * - crust (`CRUST_GAIN`): the coralline and pale crusts spread wider over
 *   the upper faces, the encrusting growth the review asked for.
 */
const TONE_RANGE: readonly [number, number] = [0.3, 1.45];
const GRAIN_RANGE: readonly [number, number] = [0.7, 1.25];
const CRACK_WIDTH = 0.018;
const CRACK_DARK = 0.55;
const MID_BUMP = 0.35;
const CRUST_GAIN = 1.2;

/**
 * The rock material: one for every outcrop, boulder and pebble, so the shelf
 * reads as one geology rather than as a kit of parts.
 *
 * Shading is in **world** space on purpose. The outcrops are many pieces —
 * a sculpted mass, talus at its foot, boulders in front — and world-space
 * strata and colour beds run straight across all of them, which is what makes
 * a pile read as one formation. The pieces never move, so nothing swims.
 *
 * The albedo is abyssal-garden's layered rock (`W_SOLID`, `mat <= 1`): grey
 * to warm-brown beds banded by height and warped by the broad field, darker
 * bedding seams on the vertical faces, coralline encrustation in warm patches,
 * pale sponge-and-barnacle crust, turf algae on whatever faces up — all
 * darkened in the cavities of the height field. Re-tuned paler than
 * abyssal-garden's basalt greys: a Caribbean reef stands on limestone, cream
 * to buff under its crusts (linear ~0.3), and at 0.15 the outcrops read as
 * black holes against the blue at 25 m.
 *
 * Light is the scene's: the caller applies `groundShading`, which puts the
 * caustics and the cloud shade on the key light and the contact darkening on
 * the ambient. `aoNode` here adds what that cannot know — the baked cavity
 * occlusion of a sculpted outcrop, and that a face turned to the seabed sees
 * none of the bright water overhead.
 */
export function createRockMaterial(name: string): THREE.MeshStandardNodeMaterial {
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.88, metalness: 0 });
  material.name = name;
  material.fog = false;
  const detail = rockDetailTexture();

  const wp: Node = positionWorld.toVar('rockP');
  const n0: Node = normalWorldGeometry.toVar('rockN');
  const w: Node = Fn(() => {
    const a = n0.abs().pow(vec3(4)).toVar();
    return a.div(a.x.add(a.y).add(a.z).max(1e-4));
  })().toVar('rockW');

  /** One triplanar read: x = height, y = colour field, zw unused; plus a world-space bump. */
  const tri = (metres: number, label: string): { hc: Node; bump: Node } => {
    const s = 1 / metres;
    const tx: Node = texture(detail, vec2(wp.z, wp.y).mul(s)).toVar(`${label}X`);
    const ty: Node = texture(detail, vec2(wp.x, wp.z).mul(s)).toVar(`${label}Y`);
    const tz: Node = texture(detail, vec2(wp.x, wp.y).mul(s)).toVar(`${label}Z`);
    const hc: Node = vec2(
      tx.b.mul(w.x).add(ty.b.mul(w.y)).add(tz.b.mul(w.z)),
      tx.a.mul(w.x).add(ty.a.mul(w.y)).add(tz.a.mul(w.z)),
    ).toVar(`${label}HC`);
    const a = tx.rg.mul(2).sub(1);
    const b = ty.rg.mul(2).sub(1);
    const c = tz.rg.mul(2).sub(1);
    const bump: Node = vec3(0, a.y, a.x)
      .mul(w.x)
      .add(vec3(b.x, 0, b.y).mul(w.y))
      .add(vec3(c.x, c.y, 0).mul(w.z))
      .toVar(`${label}N`);
    return { hc, bump };
  };

  const A = tri(SCALE_BROAD, 'rockA');
  const B = tri(SCALE_MID, 'rockB');
  const C = tri(SCALE_FINE, 'rockC');
  const D = tri(SCALE_GRAIN, 'rockD');
  const eyeDist: Node = positionWorld.distance(cameraPosition).toVar('rockEyeD');
  const fine: Node = eyeDist.smoothstep(FINE_FADE_NEAR, FINE_FADE_FAR).oneMinus().toVar('rockFine');
  const grain: Node = eyeDist.smoothstep(GRAIN_FADE_NEAR, GRAIN_FADE_FAR).oneMinus().toVar('rockGrain');
  const crisp: Node = eyeDist.smoothstep(NEAR_CRISP[0], NEAR_CRISP[1]).oneMinus().mul(NEAR_CRISP_GAIN).add(1).toVar('rockCrisp');
  const macro: Node = texture(detail, vec2(wp.x, wp.z).mul(1 / MACRO_METRES).add(wp.y.mul(0.011)).add(0.61))
    .a.smoothstep(0.3, 0.7)
    .toVar('rockMacro');
  const height: Node = A.hc.x.mul(0.5).add(B.hc.x.mul(0.35)).add(C.hc.x.mul(0.15)).toVar('rockH');
  // Two planar reads, skewed by height, for colour patches that span metres
  // and do not line up with any projection seam.
  const big: Node = texture(detail, vec2(wp.x, wp.z).mul(0.013).add(wp.y.mul(0.021))).a.toVar('rockBig');
  const big2: Node = texture(detail, vec2(wp.z, wp.x).mul(0.031).add(wp.y.mul(0.043)).add(0.37)).a.toVar('rockBig2');

  // Strata: beds about 1.8 m thick, warped by the broad field so they dip and
  // pinch, and a per-bed tone read from a texel row (deterministic on every
  // backend, which a sine hash of the bed index is not).
  //
  // [underwater-look-3] Round 8: the broad height's warp 1.6 -> `SEAM_WARP`,
  // and the seam a soft, broken groove on both sides of the bed line. The
  // 11 m height varies over a metre or two on a steep face, and at 1.6 beds of
  // warp the seam followed it into tight loops: the round-8 review's "dark
  // scribble lines on the boulder". The seam was also one-sided, a dark ramp
  // ending in a hard edge where `fract` wraps, so each loop was an inked line.
  const strata: Node = wp.y.mul(0.55).add(A.hc.x.mul(SEAM_WARP)).add(big.mul(5)).add(big2.mul(3)).toVar('rockSt');
  const bed: Node = texture(detail, vec2(strata.floor().mul(1 / 61.0), 0.5)).b.toVar('rockBed');
  const seamFract: Node = strata.fract();
  const seam: Node = seamFract
    .min(seamFract.oneMinus())
    .smoothstep(SEAM_WIDTH, 0)
    .mul(B.hc.x.smoothstep(0.3, 0.62))
    .mul(n0.y.abs().oneMinus())
    .toVar('rockSeam');

  material.colorNode = Fn(() => {
    const albedo = mix(vec3(0.13, 0.13, 0.125), vec3(0.34, 0.31, 0.25), A.hc.y.mul(1.5).sub(0.25).clamp(0, 1)).toVar();
    albedo.assign(
      mix(albedo, mix(vec3(0.24, 0.17, 0.1), vec3(0.2, 0.2, 0.15), bed), big.smoothstep(0.42, 0.66).mul(0.7)),
    );
    // Bedding seams cut dark lines into the steep faces only.
    albedo.mulAssign(
      mix(float(0.72), float(1.15), bed).mul(
        float(1).sub(seam.mul(SEAM_DARK)),
      ),
    );
    // Coralline encrustation: crimson to ochre crusts in warm patches.
    // [underwater-look-3] Round 11: wider, on the upper faces; see `CRUST_GAIN`.
    const enc = big2
      .mul(0.6)
      .add(B.hc.y.mul(0.5))
      .smoothstep(0.36, 0.52)
      .mul(height.smoothstep(0.22, 0.5))
      .mul(n0.y.smoothstep(-0.3, 0.5).mul(0.5).add(0.5))
      .mul(CRUST_GAIN)
      .min(1);
    const crust = mix(vec3(0.34, 0.06, 0.08), vec3(0.4, 0.22, 0.06), A.hc.y.smoothstep(0.45, 0.6)).mul(
      B.hc.x.mul(0.8).add(0.5),
    );
    albedo.assign(mix(albedo, crust, enc.mul(0.75)));
    // Pale crust: sponge, barnacle and bleached coralline.
    const pale = vec3(0.42, 0.4, 0.35).mul(C.hc.x.mul(0.7).add(0.5));
    albedo.assign(mix(albedo, pale, B.hc.y.mul(0.6).add(C.hc.y.mul(0.5)).smoothstep(0.6, 0.7).mul(0.5)));
    // Turf algae on whatever faces the light.
    // Patchy, not a carpet: the limestone shows between the turf.
    const turf = n0.y.smoothstep(0.1, 0.7).mul(B.hc.y.mul(0.55).add(big.mul(0.5)).smoothstep(0.45, 0.62));
    albedo.assign(mix(albedo, vec3(0.07, 0.1, 0.04), turf.mul(0.75)));
    // Cavities of the height field, with the fine scale breaking it up close.
    // [underwater-look-3] Round 7: the fine scale at 1.1 from 0.76 (0.45..1.55
    // across its range rather than 0.62..1.38); see `SCALE_GRAIN`.
    // [underwater-look-3] Round 11: tone without the cellular domes; see `TONE_RANGE`.
    const tone = A.hc.x.mul(0.55).add(B.hc.y.mul(0.3)).add(C.hc.x.mul(0.15));
    albedo.mulAssign(mix(float(TONE_RANGE[0]), float(TONE_RANGE[1]), tone).mul(mix(float(1), C.hc.y.mul(0.8).add(0.6), fine)));
    // [underwater-look-3] Round 7: the grain, and the macro field; see
    // `SCALE_GRAIN` and `MACRO_METRES`. Round 11: its colour field, not its
    // pits; see `GRAIN_RANGE`.
    albedo.mulAssign(mix(float(1), mix(float(GRAIN_RANGE[0]), float(GRAIN_RANGE[1]), D.hc.y), grain.mul(crisp).min(1.3)));
    // [underwater-look-3] Round 11: cracks; see `CRACK_WIDTH`.
    const crack = B.hc.y
      .sub(0.5)
      .abs()
      .smoothstep(0, CRACK_WIDTH)
      .oneMinus()
      .mul(A.hc.y.smoothstep(0.42, 0.58))
      .mul(fine)
      .mul(float(1).sub(enc.mul(0.8)));
    albedo.mulAssign(float(1).sub(crack.mul(CRACK_DARK)));
    albedo.mulAssign(mix(float(MACRO_RANGE[0]), float(MACRO_RANGE[1]), macro));
    return vec4(albedo, 1);
  })();

  material.normalNode = Fn(() => {
    // Seam grooves bend the normal down into the bed line on the steep faces.
    const groove = seam.mul(-0.8);
    const n = n0
      .add(A.bump.mul(0.85).add(B.bump.mul(MID_BUMP)).add(C.bump.mul(fine.mul(0.5).mul(crisp))).add(D.bump.mul(grain.mul(0.45).mul(crisp))).mul(1.6))
      .add(vec3(0, groove, 0));
    return cameraViewMatrix.mul(vec4(n.normalize(), 0)).xyz.normalize();
  })();

  // Baked cavity occlusion (outcrops; loose rock carries 1), the height
  // field's cavities, and the sky a face loses by looking at the seabed.
  const baked: Node = attribute('rockAo', 'float');
  material.aoNode = Fn(() => {
    // Under water the ambient is in-scattered light from every side, not a
    // sky dome, so a wall keeps most of it; only a true underside loses it.
    const sky = n0.y.mul(0.5).add(0.5).smoothstep(0, 0.6).mul(0.45).add(0.55);
    // [underwater-look-3] Round 11: cavities from the broad height and the mid
    // colour field, not the cellular domes; see `TONE_RANGE`.
    const cavity = A.hc.x.mul(0.6).add(B.hc.y.mul(0.4));
    return baked.mul(0.7).add(0.3).mul(sky).mul(cavity.smoothstep(0.2, 0.65).mul(0.35).add(0.65));
  })();
  return material;
}
