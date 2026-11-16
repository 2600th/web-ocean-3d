import * as THREE from 'three/webgpu';
import { mulberry32 } from '../core/random';
import { ISLAND, REEF_BAND, reefPatches, seafloorPlacementHeight } from './Seafloor';
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
// [underwater-look-2] 16/22 -> 24/34, with the reef's own density.
const GORGONIANS_PER_SITE = 24;
/** Clumps among the corals inside a site. */
const GORGONIANS_INSIDE = 34;
const GORGONIANS_PER_FOREST = 6;
/** Understorey tufts per kelp column. */
const UNDERSTOREY_PER_COLUMN = 2.2;
/**
 * [underwater-look-3] Round 10. The floor of a kelp bed, from its own
 * generator so every other tuft keeps its place: `BED_PATCHES_PER_M2` patches
 * of low cover per square metre of forest (turtle grass, Halimeda, a sea rod
 * now and then) over the open floor between the stands, `WRACK_PER_COLUMN`
 * fallen fronds lying on the sand round each holdfast, and
 * `EXTRA_UNDERSTOREY` more shaded tufts at each foot. The review read the bed
 * as kelp over a featureless floor.
 */
const BED_SEED = 0x5eabed;
const BED_PATCHES_PER_M2 = 0.05;
const WRACK_PER_COLUMN = 1.6;
const EXTRA_UNDERSTOREY = 2;

/**
 * [underwater-look-3] Round 12. The island's shallow shelf, which the review
 * found bare sand between twelve small meadows: `SHELF_MEADOWS` more turtle
 * and manatee grass beds between `SHELF_DEPTH` metres, `SHELF_RADIUS` across,
 * from their own generator (`SHELF_SEED`) so every existing tuft keeps its
 * place. Each is cut by sand channels, where a low-frequency field crosses its
 * middle within `SHELF_CHANNEL` (the sinuous bare runs current scours through
 * a Caribbean grass bed), and is holed by the same patchiness field the other
 * meadows use. Tufts at `SHELF_SPACING`, a little sparser than a reef ring.
 */
const SHELF_SEED = 0x5e1f5ea;
const SHELF_MEADOWS = 34;
const SHELF_DEPTH: readonly [number, number] = [1.6, 8];
const SHELF_RADIUS: readonly [number, number] = [12, 24];
const SHELF_CHANNEL: readonly [number, number] = [0.03, 0.065];
const SHELF_SPACING = 0.3;

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
  /** [underwater-look-3] Round 11: rock outcrops among the kelp; sea rods and Halimeda grow round them. */
  outcrops?: readonly { x: number; z: number; radius: number }[];
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
  private readonly shelfList: readonly Clearing[];

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
    plantKelpBed(leaves, mulberry32(BED_SEED), floor, options.forests ?? [], options.columns ?? [], reefSites);
    plantOutcropCover(leaves, mulberry32(BED_SEED ^ 0x0c0ffe), floor, options.outcrops ?? []);
    this.shelfList = plantShelfMeadows(leaves, mulberry32(SHELF_SEED), floor, reefSites, this.meadowList);

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

  /** [underwater-look-3] Round 12: the shelf's beds, for a harness; see `SHELF_MEADOWS`. */
  shelfMeadows(): readonly Clearing[] {
    return this.shelfList;
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
        const depth = -seafloorPlacementHeight(x, z);
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

/** [underwater-look-3] Round 12: the shelf's grass beds; see `SHELF_MEADOWS`. */
function plantShelfMeadows(
  leaves: Leaf[],
  rnd: () => number,
  floor: FloorCache,
  reef: readonly Clearing[],
  meadows: readonly Clearing[],
): Clearing[] {
  const sites: Clearing[] = [];
  for (let i = 0; i < SHELF_MEADOWS; i++) {
    for (let attempt = 0; attempt < 300; attempt++) {
      const a = rnd() * Math.PI * 2;
      const rr = ISLAND.radius * (0.7 + rnd() * 0.75);
      const x = ISLAND.x + Math.cos(a) * rr;
      const z = ISLAND.z + Math.sin(a) * rr;
      const depth = -seafloorPlacementHeight(x, z);
      if (depth < SHELF_DEPTH[0] || depth > SHELF_DEPTH[1]) continue;
      const r = SHELF_RADIUS[0] + rnd() * (SHELF_RADIUS[1] - SHELF_RADIUS[0]);
      if (inside(x, z, reef, r * 0.5)) continue;
      if (sites.concat(meadows).some((m) => Math.hypot(m.x - x, m.z - z) < m.radius + r)) continue;
      sites.push({ x, z, radius: r });
      break;
    }
  }
  for (const m of sites) {
    const n = Math.ceil((m.radius * 2) / SHELF_SPACING);
    for (let i = 0; i <= n; i++) {
      for (let j = 0; j <= n; j++) {
        const x = m.x - m.radius + (i + rnd()) * SHELF_SPACING;
        const z = m.z - m.radius + (j + rnd()) * SHELF_SPACING;
        const rr = Math.hypot(x - m.x, z - m.z) / m.radius;
        if (rr > 1) continue;
        const channel = smoothstep(SHELF_CHANNEL[0], SHELF_CHANNEL[1], Math.abs(fbm(x * 0.03, z * 0.03, 0x5e1f) - 0.5));
        const patch = smoothstep(0.15, 0.32, fbm(x * 0.11, z * 0.11, 0x5e20));
        if (rnd() > (1 - smoothstep(0.6, 1, rr)) * patch * channel) continue;
        if (inside(x, z, reef, 0.5)) continue;
        const ground = floor.height(x, z);
        if (ground > -1.2) continue;
        const y = ground - 0.04;
        const u = rnd();
        if (u < 0.05) {
          halimeda(leaves, rnd, x, y, z, 0.85);
        } else if (u < 0.7) {
          tuft(leaves, rnd, x, y, z, 4 + Math.floor(rnd() * 3), LeafType.TURTLE_GRASS,
            [0.2, 0.44], [0.036, 0.056], [0.04, 0.38], [0.3, 0.5], 0.85);
        } else {
          tuft(leaves, rnd, x, y, z, 3 + Math.floor(rnd() * 2), LeafType.MANATEE_GRASS,
            [0.3, 0.6], [0.018, 0.026], [0.05, 0.5], [0.35, 0.7], 0.85);
        }
      }
    }
  }
  return sites;
}

/**
 * [underwater-look-3] Round 11. Cover at the foot of the kelp beds' outcrops:
 * sea rods, the purple-to-tan accent abyssal-garden puts at the foot of its
 * forest rock, and Halimeda, in a ring round each pile.
 */
function plantOutcropCover(
  leaves: Leaf[],
  rnd: () => number,
  floor: FloorCache,
  outcrops: readonly { x: number; z: number; radius: number }[],
): void {
  for (const o of outcrops) {
    const n = 6 + Math.floor(rnd() * 6);
    for (let i = 0; i < n; i++) {
      const a = rnd() * Math.PI * 2;
      const r = o.radius * (0.9 + rnd() * 0.8);
      const x = o.x + Math.cos(a) * r;
      const z = o.z + Math.sin(a) * r;
      const ground = floor.height(x, z);
      if (ground > -2) continue;
      const y = ground - 0.05;
      if (rnd() < 0.65) {
        tuft(leaves, rnd, x, y, z, 6 + Math.floor(rnd() * 5), LeafType.SEA_ROD,
          [0.5, 1.3], [0.032, 0.05], [0.05, 0.42], [0.3, 0.42], 0.8, 0.16);
      } else {
        halimeda(leaves, rnd, x, y, z, 0.7);
      }
    }
  }
}

/** [underwater-look-3] Round 10: see `BED_PATCHES_PER_M2`. */
function plantKelpBed(
  leaves: Leaf[],
  rnd: () => number,
  floor: FloorCache,
  forests: readonly Forest[],
  columns: readonly { x: number; y: number; z: number; height: number }[],
  reef: readonly Clearing[],
): void {
  for (const f of forests) {
    const n = Math.round(Math.PI * f.radius * f.radius * BED_PATCHES_PER_M2);
    for (let i = 0; i < n; i++) {
      const a = rnd() * Math.PI * 2;
      const r = Math.sqrt(rnd()) * f.radius * 1.1;
      const cx = f.x + Math.cos(a) * r;
      const cz = f.z + Math.sin(a) * r;
      if (inside(cx, cz, reef)) continue;
      // A patch is a few tufts a hand apart: it reads as a patch, not a dot.
      const kind = rnd();
      const tufts = 2 + Math.floor(rnd() * 4);
      for (let j = 0; j < tufts; j++) {
        const x = cx + (rnd() - 0.5) * 0.9;
        const z = cz + (rnd() - 0.5) * 0.9;
        const ground = floor.height(x, z);
        if (ground > -2) continue;
        const y = ground - 0.04;
        if (kind < 0.45) {
          tuft(leaves, rnd, x, y, z, 5, LeafType.TURTLE_GRASS, [0.22, 0.45], [0.026, 0.038], [0.05, 0.5], [0.3, 0.5], 0.6);
        } else if (kind < 0.85) {
          halimeda(leaves, rnd, x, y, z, 0.7);
        } else {
          tuft(leaves, rnd, x, y, z, 5 + Math.floor(rnd() * 4), LeafType.SEA_ROD,
            [0.45, 1.1], [0.032, 0.048], [0.05, 0.4], [0.3, 0.42], 0.75, 0.14);
        }
      }
    }
  }
  for (const c of columns) {
    // Fallen fronds: a blade lying on the sand, curling up off it a little,
    // the yellow-brown of a detached one.
    const wrack = Math.floor(WRACK_PER_COLUMN * rnd() * 2);
    for (let i = 0; i < wrack; i++) {
      const a = rnd() * Math.PI * 2;
      const r = 0.4 + rnd() * 2.2;
      const x = c.x + Math.cos(a) * r;
      const z = c.z + Math.sin(a) * r;
      if (inside(x, z, reef)) continue;
      const len = 0.7 + rnd() * 0.9;
      leaves.push({
        x, y: floor.height(x, z) + 0.03, z, len, yaw: rnd() * Math.PI * 2, tilt: 1.47 + rnd() * 0.08,
        width: len * (0.15 + rnd() * 0.06) + 0.03, hA: 0, type: LeafType.KELP,
        s1: rnd() * 0.35, s2: rnd() < 0.6 ? 0.87 + rnd() * 0.12 : rnd() * 0.8, bend: 0.2 + rnd() * 0.08, ao: 0.55,
      });
    }
    const extra = Math.floor(EXTRA_UNDERSTOREY * rnd() * 2);
    for (let i = 0; i < extra; i++) {
      const a = rnd() * Math.PI * 2;
      const r = 0.7 + rnd() * 2.4;
      const x = c.x + Math.cos(a) * r;
      const z = c.z + Math.sin(a) * r;
      if (inside(x, z, reef)) continue;
      const y = floor.height(x, z) - 0.04;
      if (rnd() < 0.5) halimeda(leaves, rnd, x, y, z, 0.5);
      else tuft(leaves, rnd, x, y, z, 4, LeafType.TURTLE_GRASS, [0.12, 0.28], [0.022, 0.032], [0.05, 0.45], [0.3, 0.5], 0.5);
    }
  }
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
