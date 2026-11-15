import * as THREE from 'three/webgpu';
import { M, PropBuilder, cylinder, jitter, lathe, srgb, tube, type Tint } from './kit';
import { mulberry32 } from '../../core/random';

/**
 * Coopered casks and buckets.
 *
 * The cask is the shipboard container of the period — water, beef, pork, rum,
 * sugar and powder all travelled in them — and the sizes were standard
 * measures, so the presets here are the real ones (heights, roughly):
 * a barrel about 0.75 m, a hogshead about 1.0 m, a puncheon about 1.2 m. The
 * bilge (the belly) is about 1.2-1.25x the head diameter; that bulge is what
 * lets one man roll and turn a full hogshead, and it is the first thing the eye
 * checks on a barrel.
 *
 * What makes it read as coopered rather than turned:
 *  - **Staves**: each is its own strip with its own tint and grain offset, and
 *    the seams between them are cut in as a shallow V — the radius dips at
 *    every stave edge, which the profile-derived normals turn into a dark line
 *    that survives down to the LOD that drops the geometry for a tint.
 *  - **Chimes**: the staves stand proud of the head by a few centimetres, and
 *    the head sits in a groove (the croze) inside them.
 *  - **Hoops**: iron at the chimes and quarters by the early eighteenth century
 *    on rum and spirit casks, split-wood (ash, hazel) bundles on dry goods.
 *    Hoops are raised bands, not paint — at the gameplay distance they are what
 *    carries the shape.
 */

export interface CaskOptions {
  height: number;
  /** Head radius, metres. */
  head: number;
  /** Bilge radius, metres. */
  bilge: number;
  staves: number;
  /** 'iron' all-iron hoops, 'wood' split hoops, 'mixed' iron chimes and wooden quarters. */
  hoops: 'iron' | 'wood' | 'mixed';
  /** Missing staves and one head gone: flotsam, or a cask stove in on the beach. */
  stoveIn?: boolean;
  /** Bung and spile on one stave at the bilge. */
  bung?: boolean;
  /** Rope sling round the bilge, as a cask is hoisted in and out of a hold. */
  sling?: boolean;
  lod: 0 | 1 | 2;
  seed: number;
}

export const CASK_PRESETS = {
  barrel: { height: 0.76, head: 0.235, bilge: 0.29 },
  hogshead: { height: 1.0, head: 0.3, bilge: 0.37 },
  puncheon: { height: 1.2, head: 0.36, bilge: 0.45 },
} as const;

const OAK: Tint = srgb(0.46, 0.31, 0.19);
const WEATHERED: Tint = srgb(0.5, 0.42, 0.33);
const HOOP_WOOD: Tint = srgb(0.55, 0.42, 0.27);
const HEMP: Tint = srgb(0.62, 0.52, 0.36);

/** Radius of a cask's profile at height y: a parabola from head to bilge and back. */
export function caskRadius(o: { height: number; head: number; bilge: number }, y: number): number {
  const t = (2 * y) / o.height - 1;
  return o.head + (o.bilge - o.head) * (1 - t * t);
}

export function buildCask(pb: PropBuilder, m: THREE.Matrix4, o: CaskOptions): void {
  const rng = mulberry32(o.seed);
  const wood = pb.kit('wood');
  const iron = pb.kit('iron');
  const H = o.height;
  const chime = 0.035;
  const wall = 0.022;
  const rows = o.lod === 0 ? 12 : o.lod === 1 ? 8 : 6;
  const staves = o.lod === 2 ? 0 : o.staves;
  const weathered = rng() < 0.5;
  const baseTint = weathered ? WEATHERED : OAK;

  // Missing staves of a stove-in cask: a contiguous run on one side.
  const missingFrom = o.stoveIn ? Math.floor(rng() * staves) : -1;
  const missingCount = o.stoveIn ? 2 + Math.floor(rng() * 2) : 0;
  const missing = (i: number) => missingFrom >= 0 && (i - missingFrom + staves) % staves < missingCount;

  const outer: [number, number][] = [];
  for (let k = 0; k <= rows; k++) {
    const y = (k / rows) * H;
    outer.push([caskRadius(o, y), y]);
  }

  wood.withMatrix(m, () => {
    if (staves === 0) {
      // LOD2: one turned body. The seams go, the silhouette and hoops stay.
      lathe(wood, outer, 12, M.identity(), baseTint);
    } else {
      const span = (Math.PI * 2) / staves;
      for (let i = 0; i < staves; i++) {
        if (missing(i)) continue;
        const a0 = i * span;
        const tint = jitter(baseTint, rng, 0.14);
        const off = rng() * 10;
        // Each stave in its own lathe, the radius pulled in at its edges for the seam.
        const seg = o.lod === 0 ? 4 : 2;
        lathe(wood, outer, seg, M.identity(), tint, {
          start: a0,
          span,
          uvOffset: off,
          radius: (a) => {
            const e = Math.min(a - a0, a0 + span - a) / span;
            return e < 0.08 ? 0.986 + e * 0.17 : 1;
          },
        });
        if (o.stoveIn || o.lod === 0) {
          // Inner face and the end grain at the chimes. A stove-in cask shows
          // its whole inside; a closed one only the few centimetres of stave
          // that stand above each head.
          const inner: [number, number][] = o.stoveIn
            ? (outer.map(([r, y]) => [r - wall, y]).reverse() as [number, number][])
            : [];
          if (!o.stoveIn) {
            for (const [ya, yb] of [
              [H, H - chime - 0.005],
              [chime + 0.005, 0],
            ]) {
              lathe(wood, [[caskRadius(o, ya) - wall, ya], [caskRadius(o, yb) - wall, yb]], seg, M.identity(), [tint[0] * 0.7, tint[1] * 0.7, tint[2] * 0.7], { start: a0, span });
            }
          }
          if (inner.length > 0) lathe(wood, inner, seg, M.identity(), [tint[0] * 0.7, tint[1] * 0.7, tint[2] * 0.7], { start: a0, span, uvOffset: off });
          for (const y of [0, H]) {
            const r = caskRadius(o, y);
            lathe(wood, y === 0 ? [[r - wall, y], [r, y]] : [[r, y], [r - wall, y]], seg, M.identity(), tint, { start: a0, span, hard: true });
          }
        }
      }
    }

    // Heads, sunk below the chime. A stove-in cask has lost its top one.
    for (const top of [false, true]) {
      if (top && o.stoveIn) continue;
      const y = top ? H - chime : chime;
      const r = caskRadius(o, y) - wall * 0.6;
      const boards = o.lod === 2 ? 1 : 3;
      for (let b = 0; b < boards; b++) {
        const x0 = -r + ((2 * r) / boards) * b;
        const x1 = x0 + (2 * r) / boards;
        headBoard(wood, r, x0 + 0.002, x1 - 0.002, y, top, jitter(baseTint, rng, 0.1), o.lod === 0 ? 14 : 8);
      }
    }
  });

  // Hoops: chime and quarter pairs, and a bilge pair on the bigger sizes.
  const hoopYs = [0.06, 0.2, 0.8, 0.94];
  if (o.height >= 0.95) hoopYs.push(0.42, 0.58);
  const sides = o.lod === 0 ? 40 : o.lod === 1 ? 24 : 14;
  for (const f of hoopYs) {
    const chimeHoop = f < 0.1 || f > 0.9;
    const isIron = o.hoops === 'iron' || (o.hoops === 'mixed' && chimeHoop);
    if (o.lod === 2 && !chimeHoop && !isIron) continue;
    const y = f * H;
    const width = isIron ? 0.04 : 0.055;
    const thick = isIron ? 0.006 : 0.012;
    const kit = isIron ? iron : wood;
    const y0 = y - width / 2;
    const y1 = y + width / 2;
    const r0 = caskRadius(o, y0) + thick * 0.4;
    const r1 = caskRadius(o, y1) + thick * 0.4;
    // A short closed profile: inner bottom edge, outer face, inner top edge.
    lathe(
      kit,
      [
        [r0 - thick * 0.4, y0],
        [r0 + thick, y0],
        [r1 + thick, y1],
        [r1 - thick * 0.4, y1],
      ],
      sides,
      m,
      isIron ? [1, 1, 1] : jitter(HOOP_WOOD, rng, 0.1),
      { hard: true },
    );
  }

  if (o.bung && o.lod < 2 && !o.stoveIn) {
    const r = o.bilge;
    const a = rng() * Math.PI * 2;
    // A tapered oak bung driven into the bunghole on the bilge stave.
    cylinder(wood, 0.022, 0.026, 0.03, 8, m.clone().multiply(M.at(Math.cos(a) * (r + 0.006), H / 2, Math.sin(a) * (r + 0.006), 0, -a, Math.PI / 2)), [0.25, 0.17, 0.1]);
  }

  if (o.sling && o.lod < 2) {
    const rope = pb.kit('rope');
    const pts: [number, number, number][] = [];
    const r = o.bilge + 0.012;
    for (let i = 0; i <= 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      pts.push([Math.cos(a) * r, H / 2 + Math.sin(a * 2) * 0.01, Math.sin(a) * r]);
    }
    tube(rope, pts, 0.012, 6, m, HEMP, false);
  }


}

/** One board of a cask head: the strip of the disc between x0 and x1, facing up (top) or down. */
function headBoard(kit: ReturnType<PropBuilder['kit']>, r: number, x0: number, x1: number, y: number, top: boolean, tint: Tint, steps: number): void {
  const ny = top ? 1 : -1;
  const left: number[] = [];
  const right: number[] = [];
  for (let i = 0; i <= steps; i++) {
    const z = -r + ((2 * r) / steps) * i;
    const half = Math.sqrt(Math.max(0, r * r - z * z));
    const a = Math.max(x0, -half);
    const b = Math.min(x1, half);
    if (b <= a) {
      left.push(-1);
      right.push(-1);
      continue;
    }
    left.push(kit.vertex(a, y, z, 0, ny, 0, a, z, tint));
    right.push(kit.vertex(b, y, z, 0, ny, 0, b, z, tint));
  }
  for (let i = 0; i < steps; i++) {
    if (left[i] < 0 || left[i + 1] < 0) continue;
    if (top) kit.quad(left[i], left[i + 1], right[i + 1], right[i]);
    else kit.quad(left[i], right[i], right[i + 1], left[i + 1]);
  }
}

/**
 * A stave bucket: tapered (wider at the mouth), two hoops, a rope bail through
 * two lugs — the fire and swabbing bucket of every ship's waist.
 */
export function buildBucket(pb: PropBuilder, m: THREE.Matrix4, lod: 0 | 1 | 2, seed: number): void {
  const rng = mulberry32(seed);
  const wood = pb.kit('wood');
  const H = 0.36;
  const rb = 0.15;
  const rt = 0.18;
  const staves = lod === 0 ? 14 : lod === 1 ? 10 : 0;
  const prof: [number, number][] = [
    [rb, 0],
    [rt, H],
  ];
  wood.withMatrix(m, () => {
    if (staves === 0) {
      lathe(wood, prof, 10, M.identity(), WEATHERED);
    } else {
      const span = (Math.PI * 2) / staves;
      for (let i = 0; i < staves; i++) {
        const tint = jitter(WEATHERED, rng, 0.15);
        const a0 = i * span;
        const lug = i === 0 || i === staves / 2;
        const top = lug ? H + 0.07 : H;
        const p: [number, number][] = [
          [rb, 0],
          [rt + (top - H) * 0.05, top],
        ];
        const inner: [number, number][] = [
          [rt - 0.016, top],
          [rb - 0.016, 0.03],
        ];
        lathe(wood, p, 2, M.identity(), tint, { start: a0, span, uvOffset: rng() * 10, radius: (a) => (Math.min(a - a0, a0 + span - a) / span < 0.08 ? 0.985 : 1) });
        lathe(wood, inner, 2, M.identity(), [tint[0] * 0.6, tint[1] * 0.6, tint[2] * 0.6], { start: a0, span });
        lathe(wood, [[rt + 0.002, top], [rt - 0.016, top]], 2, M.identity(), tint, { start: a0, span, hard: true });
      }
      // Bottom board inside.
      headBoard(wood, rb - 0.015, -rb, rb, 0.03, true, jitter(WEATHERED, rng, 0.1), 8);
    }
  });
  const iron = pb.kit('iron');
  for (const f of [0.15, 0.82]) {
    const y = f * H;
    const r = rb + (rt - rb) * f + 0.004;
    lathe(iron, [[r - 0.002, y - 0.016], [r + 0.004, y - 0.016], [r + 0.004 + 0.0015, y + 0.016], [r - 0.002, y + 0.016]], lod === 0 ? 28 : 16, m, [1, 1, 1], { hard: true });
  }
  if (lod < 2) {
    // Rope bail between the lugs, sagging to one side as it is left.
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      const a = Math.PI * t;
      pts.push([Math.cos(a) * (rt + 0.01), H + 0.05 + Math.sin(a) * 0.18, Math.sin(a) * 0.09]);
    }
    tube(pb.kit('rope'), pts, 0.009, 5, m, HEMP, true);
  }
}
