import * as THREE from 'three/webgpu';
import { mulberry32 } from '../core/random';
import { ISLAND, REEF_BAND, reefPatches, seafloorHeight } from './Seafloor';
import { FloorCache, LeafField, LeafType, fbm, type Leaf, type LeafShading, type LeafUniforms } from './LeafField';
import type { Clearing, Forest } from './Kelp';

/**
 * Seabed ground cover: Caribbean seagrass meadows, gorgonians and sargassum,
 * drawn as instanced ribbon leaves by the same engine as the kelp columns.
 *
 * abyssal-garden (MIT, (c) 2026 Maksymilian Dendura) carpets its floor in
 * small ribbon tufts — magenta turf, teal grass — placed by the thousand from a
 * density field; the tuft construction (a few leaves fanned from one point at
 * random yaw and lean) is its `cluster()` and ground-cover loops, ported. What
 * grows is not: this is a tropical sea around 1700, so the cover is what a
 * diver on a Caribbean shelf actually swims over:
 *
 *   - **Turtle grass** (*Thalassia*) meadows on the sand: flat green straps,
 *     a fifth of them browned by epiphytes, with **manatee grass**
 *     (*Syringodium*) threads mixed through.
 *   - **Gorgonians** — sea rods — on the hard ground round the reef, the
 *     purple-to-tan accent abyssal-garden gets from magenta turf, beside
 *     **Halimeda**, the segmented calcified green alga that builds Caribbean sand.
 *   - **Sargassum** gold tufts, sparse, under and between the columns.
 *
 * Sea fans proper are the reef's (`ReefLife`), which has a lattice for them.
 *
 * Placement is against the rendered floor (`FloorCache`) so no strap floats or
 * sinks, and against the same clearings the kelp keeps: grass stops at a reef
 * site's edge, the gorgonians gather on its apron.
 */

const COVER_SEED = 0x3ac0e2;

/** Meadow counts, depth bands and radii. Turtle grass wants light: 2-12 m. */
const ISLAND_MEADOWS = 12;
const ISLAND_MEADOW_DEPTH: readonly [number, number] = [1.5, 12];
const PLATEAU_MEADOWS = 6;
const PLATEAU_MEADOW_DEPTH: readonly [number, number] = [12, 22];
const MEADOW_RADIUS: readonly [number, number] = [8, 15];
/** Spacing of grass tufts, metres; stylised, a tuft is a handful of straps. */
const TUFT_SPACING = 0.32;

/** Gorgonian or Halimeda clumps per reef site's apron, and per grove. */
const GORGONIANS_PER_SITE = 16;
/** Clumps among the corals inside a site. */
const GORGONIANS_INSIDE = 22;
const GORGONIANS_PER_FOREST = 6;
/** Understorey tufts per kelp column. */
const UNDERSTOREY_PER_COLUMN = 2.2;

/** Levels for short cover: seven rows near (Halimeda's joints need them), three beyond, half far out. */
const COVER_LEVELS = [
  { rows: 7, thin: 1, until: 0.35 },
  { rows: 3, thin: 1, until: 0.7 },
  { rows: 3, thin: 0.5, until: 1 },
] as const;

export interface GroundCoverOptions {
  /** The kelp's uniforms: one water for both. */
  uniforms: LeafUniforms;
  /** The kelp forests, for their understorey. */
  forests?: readonly Forest[];
  columns?: readonly { x: number; y: number; z: number; height: number }[];
  /** Reef sites: no grass inside, gorgonians on the apron. */
  clearings?: readonly Clearing[];
  shading?: LeafShading;
  range?: number;
  seed?: number;
}

export class GroundCover {
  readonly object: THREE.Object3D;
  private readonly field: LeafField;
  private readonly meadowList: readonly Clearing[];

  constructor(density: number, options: GroundCoverOptions) {
    const random = mulberry32(options.seed ?? COVER_SEED);
    const floor = new FloorCache();
    const reefSites: Clearing[] = [
      ...(options.clearings ?? []),
      // The plateau patches are the same list `ReefLife` hands in; only fall back to them alone.
      ...(options.clearings ? [] : reefPatches().map((p) => ({ x: p.x, z: p.z, radius: REEF_BAND.patchRadius * 1.1 }))),
    ];
    const leaves: Leaf[] = [];
    this.meadowList = plantMeadows(leaves, random, floor, reefSites);
    plantGorgonians(leaves, random, floor, reefSites, options.forests ?? []);
    plantUnderstorey(leaves, random, floor, options.columns ?? [], reefSites);

    this.field = new LeafField(leaves, options.uniforms, {
      name: 'seabed-cover',
      levels: COVER_LEVELS,
      range: options.range ?? 60,
      swayMargin: 0.6,
      shading: options.shading,
    });
    this.object = this.field.object;
    this.setCount(density);
  }

  /** Meadow centres, for a harness to frame one. */
  meadows(): readonly Clearing[] {
    return this.meadowList;
  }

  /** Density 0..1, as the kelp's. */
  setCount(density: number): void {
    this.field.setDensity(density);
  }

  getCount(): number {
    return this.field.getDensity();
  }

  leafCount(): number {
    return this.field.leafCount;
  }

  setRange(metres: number): void {
    this.field.setRange(metres);
  }

  setVisible(v: boolean): void {
    this.field.setVisible(v);
  }

  setCameraPosition(position: THREE.Vector3): void {
    this.field.setEye(position);
  }

  setVelocity(previous: unknown, current: unknown): void {
    this.field.setVelocity(previous, current);
  }

  allMeshes(): THREE.Mesh[] {
    return this.field.allMeshes();
  }

  dispose(): void {
    this.field.dispose();
  }
}

// ------------------------------------------------------------------ placement

function inside(x: number, z: number, sites: readonly Clearing[], margin = 0): boolean {
  for (const s of sites) if (Math.hypot(x - s.x, z - s.z) < s.radius + margin) return true;
  return false;
}

/** A tuft: abyssal's ground-cover loop, `n` leaves fanned from one point. */
function tuft(
  leaves: Leaf[],
  rnd: () => number,
  x: number,
  y: number,
  z: number,
  n: number,
  type: number,
  len: readonly [number, number],
  width: readonly [number, number],
  tilt: readonly [number, number],
  bend: readonly [number, number],
  ao: number,
  spread = 0.08,
): void {
  const rr = (r: readonly [number, number]) => r[0] + (r[1] - r[0]) * rnd();
  const yaw0 = rnd() * Math.PI * 2;
  for (let j = 0; j < n; j++) {
    leaves.push({
      x: x + (rnd() - 0.5) * spread,
      y,
      z: z + (rnd() - 0.5) * spread,
      len: rr(len),
      yaw: yaw0 + (j * Math.PI * 2) / n + (rnd() - 0.5) * 0.8,
      tilt: rr(tilt),
      width: rr(width),
      hA: 0,
      type,
      s1: rnd(),
      s2: rnd(),
      bend: rr(bend),
      ao,
    });
  }
}

/**
 * Seagrass meadows: sites by depth band, then tufts thrown on a jittered grid
 * against an edge falloff and a patchiness field, so a meadow has a ragged
 * margin and sandy holes rather than a disc's edge.
 */
function plantMeadows(leaves: Leaf[], rnd: () => number, floor: FloorCache, reef: readonly Clearing[]): Clearing[] {
  const sites: { x: number; z: number; r: number }[] = [];
  const pick = (count: number, band: readonly [number, number], island: boolean) => {
    for (let i = 0; i < count; i++) {
      for (let attempt = 0; attempt < 300; attempt++) {
        const a = rnd() * Math.PI * 2;
        let x: number;
        let z: number;
        if (island) {
          const r = ISLAND.radius * (0.72 + rnd() * 0.6);
          x = ISLAND.x + Math.cos(a) * r;
          z = ISLAND.z + Math.sin(a) * r;
        } else {
          const r = 24 + Math.sqrt(rnd()) * 270;
          x = Math.cos(a) * r;
          z = Math.sin(a) * r;
        }
        const depth = -seafloorHeight(x, z);
        if (depth < band[0] || depth > band[1]) continue;
        const r = MEADOW_RADIUS[0] + rnd() * (MEADOW_RADIUS[1] - MEADOW_RADIUS[0]);
        if (inside(x, z, reef, 2)) continue;
        if (sites.some((s) => Math.hypot(s.x - x, s.z - z) < s.r + r + 4)) continue;
        sites.push({ x, z, r });
        break;
      }
    }
  };
  pick(ISLAND_MEADOWS, ISLAND_MEADOW_DEPTH, true);
  pick(PLATEAU_MEADOWS, PLATEAU_MEADOW_DEPTH, false);

  const grass = (x: number, z: number, dens: number) => {
    if (rnd() > dens) return;
    if (inside(x, z, reef, 0.5)) return;
    const ground = floor.height(x, z);
    if (ground > -1.2) return;
    const y = ground - 0.04;
    // A third manatee grass, the rest turtle grass; Halimeda clumps through
    // it and the occasional sargassum tuft drifted in.
    const u = rnd();
    if (u < 0.04) {
      halimeda(leaves, rnd, x, y, z, 0.85);
    } else if (u < 0.62) {
      tuft(leaves, rnd, x, y, z, 5 + Math.floor(rnd() * 3), LeafType.TURTLE_GRASS,
        [0.22, 0.46], [0.036, 0.056], [0.04, 0.38], [0.3, 0.5], 0.85);
    } else if (u < 0.96) {
      tuft(leaves, rnd, x, y, z, 3 + Math.floor(rnd() * 3), LeafType.MANATEE_GRASS,
        [0.3, 0.62], [0.018, 0.026], [0.05, 0.5], [0.35, 0.7], 0.85);
    } else {
      tuft(leaves, rnd, x, y, z, 4, LeafType.SARGASSUM,
        [0.18, 0.4], [0.05, 0.08], [0.25, 0.9], [0.4, 0.8], 0.8);
    }
  };

  for (const s of sites) {
    const n = Math.ceil((s.r * 2) / TUFT_SPACING);
    for (let i = 0; i <= n; i++) {
      for (let j = 0; j <= n; j++) {
        const x = s.x - s.r + (i + rnd()) * TUFT_SPACING;
        const z = s.z - s.r + (j + rnd()) * TUFT_SPACING;
        const rr = Math.hypot(x - s.x, z - s.z) / s.r;
        if (rr > 1) continue;
        grass(x, z, (1 - smoothstep(0.55, 1, rr)) * smoothstep(0.3, 0.5, fbm(x * 0.12, z * 0.12, 0x77)));
      }
    }
  }

  // Every reef patch sits in seagrass with a bare sand halo round it: the
  // grazing halo reef fish and urchins crop, a few metres wide, which is what
  // makes a Caribbean patch reef read from a distance. The ring is the meadow;
  // the halo is where the gorgonians stand.
  for (const s of reef) {
    const inner = s.radius + HALO;
    const outer = inner + RING_WIDTH;
    const n = Math.ceil((outer * 2) / TUFT_SPACING);
    for (let i = 0; i <= n; i++) {
      for (let j = 0; j <= n; j++) {
        const x = s.x - outer + (i + rnd()) * TUFT_SPACING;
        const z = s.z - outer + (j + rnd()) * TUFT_SPACING;
        const d = (Math.hypot(x - s.x, z - s.z) - inner) / (outer - inner);
        if (d < 0 || d > 1) continue;
        if (inside(x, z, reef, HALO - 0.5)) continue;
        grass(x, z, smoothstep(0, 0.2, d) * (1 - smoothstep(0.6, 1, d)) * smoothstep(0.28, 0.48, fbm(x * 0.12, z * 0.12, 0x78)));
      }
    }
  }
  return sites.map((m) => ({ x: m.x, z: m.z, radius: m.r }));
}




/** Width of the bare sand halo round a reef patch, metres. */
const HALO = 4;
/** Width of the seagrass ring outside the halo, metres. */
const RING_WIDTH = 7;

/** A Halimeda clump: short stiff segmented chains fanned from a holdfast. */
function halimeda(leaves: Leaf[], rnd: () => number, x: number, y: number, z: number, ao: number): void {
  tuft(leaves, rnd, x, y, z, 6 + Math.floor(rnd() * 6), LeafType.HALIMEDA,
    [0.09, 0.22], [0.045, 0.065], [0.15, 0.8], [0.3, 0.6], ao, 0.1);
}

/**
 * Gorgonians and Halimeda: a clump of sea rods (stiff, nearly straight, purple or tan) or of
 * sea plumes (feathery, lavender, swaying), on the ring just outside a reef site
 * and in the open patches between kelp columns.
 */
function plantGorgonians(
  leaves: Leaf[],
  rnd: () => number,
  floor: FloorCache,
  reef: readonly Clearing[],
  forests: readonly Forest[],
): void {
  const clump = (x: number, z: number, ao: number) => {
    const ground = floor.height(x, z);
    if (ground > -2) return;
    const y = ground - 0.05;
    if (rnd() < 0.6) {
      tuft(leaves, rnd, x, y, z, 5 + Math.floor(rnd() * 5), LeafType.SEA_ROD,
        [0.4, 1.25], [0.03, 0.045], [0.05, 0.42], [0.3, 0.42], ao, 0.12);
    } else {
      halimeda(leaves, rnd, x, y, z, ao);
    }
  };
  for (const s of reef) {
    for (let i = 0; i < GORGONIANS_PER_SITE; i++) {
      const a = rnd() * Math.PI * 2;
      // In the halo: on the reef's hard edge, before the grass starts.
      const r = s.radius * 0.95 + rnd() * (HALO + 1);
      const cx = s.x + Math.cos(a) * r;
      const cz = s.z + Math.sin(a) * r;
      // A clump is a small colony: two or three bases a pace apart.
      const bases = 1 + Math.floor(rnd() * 3);
      for (let b = 0; b < bases; b++) clump(cx + (rnd() - 0.5) * 1.6, cz + (rnd() - 0.5) * 1.6, 0.9);
    }
    // And a few among the corals: on a Caribbean patch reef the rods and the
    // Halimeda grow between the heads, not only round them.
    for (let i = 0; i < GORGONIANS_INSIDE; i++) {
      const a = rnd() * Math.PI * 2;
      const r = s.radius * 0.85 * Math.sqrt(rnd());
      clump(s.x + Math.cos(a) * r, s.z + Math.sin(a) * r, 0.75);
    }
  }
  for (const f of forests) {
    for (let i = 0; i < GORGONIANS_PER_FOREST; i++) {
      const a = rnd() * Math.PI * 2;
      const r = f.radius * Math.sqrt(rnd()) * 1.1;
      clump(f.x + Math.cos(a) * r, f.z + Math.sin(a) * r, 0.7);
    }
  }
}

/** Under the columns: shaded turtle grass and sargassum gold round the holdfasts. */
function plantUnderstorey(
  leaves: Leaf[],
  rnd: () => number,
  floor: FloorCache,
  columns: readonly { x: number; y: number; z: number; height: number }[],
  reef: readonly Clearing[],
): void {
  for (const c of columns) {
    const count = Math.floor(UNDERSTOREY_PER_COLUMN + rnd());
    for (let i = 0; i < count; i++) {
      const a = rnd() * Math.PI * 2;
      const r = 0.6 + rnd() * 1.8;
      const x = c.x + Math.cos(a) * r;
      const z = c.z + Math.sin(a) * r;
      if (inside(x, z, reef)) continue;
      const y = floor.height(x, z) - 0.04;
      if (rnd() < 0.6) {
        tuft(leaves, rnd, x, y, z, 4, LeafType.TURTLE_GRASS, [0.14, 0.3], [0.022, 0.032], [0.05, 0.45], [0.3, 0.5], 0.5);
      } else {
        tuft(leaves, rnd, x, y, z, 4 + Math.floor(rnd() * 3), LeafType.SARGASSUM,
          [0.2, 0.45], [0.05, 0.085], [0.2, 0.85], [0.4, 0.85], 0.55);
      }
    }
  }
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
