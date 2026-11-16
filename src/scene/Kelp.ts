import * as THREE from 'three/webgpu';
import { mulberry32 } from '../core/random';
import { ISLAND, REEF_BAND, reefPatches, seafloorPlacementHeight } from './Seafloor';
import {
  FloorCache,
  LeafField,
  LeafType,
  LeafUniforms,
  fbm,
  type Leaf,
  type LeafShading,
} from './LeafField';

/**
 * Forests of towering seaweed columns on the island shelf and the plateau.
 *
 * Adapted from abyssal-garden, MIT, (c) 2026 Maksymilian Dendura: `column()` in
 * its `world.js` — two to four stipes twisting round a column, a strap leaf
 * every 1.7 m up each, and about thirty ribbon blades per metre hung off them
 * with a length, lean and curl that change with height, plus the sparkling
 * specks — is ported here nearly line for line, and the leaves are drawn by
 * `LeafField`, which carries its leaf shader. See `THIRD_PARTY_NOTICES.md`.
 *
 * **What it replaces.** Four thousand flat blades, the tallest five metres,
 * scattered as small tufts: `docs/images/reef.png` shows what that bought — a
 * few weeds on the sand, lost in the haze. A column built out of hundreds of
 * leaves is a different object: it has a silhouette against the Snell window
 * from below, a volume the light comes through, and a sway that reads at fifty
 * metres because the whole column leans together.
 *
 * **Setting.** Giant kelp is a temperate plant and this sea is the Caribbean,
 * so these are stylised columns rather than *Macrocystis* — the look is the
 * point (Sea of Thieves takes the same liberty) — but their palette leans gold,
 * an eighth of them are sargassum-gold throughout, and the floor around them is
 * turtle grass and gorgonians (`GroundCover`), not a cold-water understorey.
 *
 * **Where.** Forests are sited the way the old beds were, against the
 * heightfield rather than coordinates: a light-limited depth band, the island
 * shelf and the origin plateau, and never on a reef site, the spawn or the
 * beach. Inside a forest the columns are thrown against a noise field that
 * leaves clearings, so the reef and the view between the stands stay readable.
 *
 * **Never through the surface.** Every column stops 2 m short of the surface
 * plus its top leaves' reach, so at 10 m/s wind (Hs 2.5 m, troughs ~1.2 m
 * down) nothing breaches. A full gale's troughs can still reach a column top;
 * see the report on the change.
 */

/** Local seed; see the old file's note on why it is a literal. */
const KELP_SEED = 0x6b31f5;

// --------------------------------------------------------------- the forests

/**
 * Grove counts per region. Few, deep and dense: a grove is something to swim
 * into, and in a Caribbean scene it is the stylised hero element, not the
 * ground cover — the meadows (`GroundCover`) are what covers the shelf.
 */
const ISLAND_FORESTS = 5;
const PLATEAU_FORESTS = 2;

/**
 * Water depth at a forest's centre, metres. Under ~8 m there is no room for a
 * column worth the name under the 2 m clearance; past ~24 m the floor is in
 * twilight in this water and the column would be fog.
 */
const ISLAND_DEPTH: readonly [number, number] = [13, 25];
const PLATEAU_DEPTH: readonly [number, number] = [15, 22];
/** Island sampling annulus in island radii: the shelf, from the surf line out. */
const ISLAND_INNER = 0.75;
const ISLAND_OUTER = 1.45;
/** Plateau annulus, metres: clear of the spawn, inside the plateau. */
const PLATEAU_INNER = 52;
const PLATEAU_OUTER = 300;
/** Forest radius range, metres. */
const FOREST_RADIUS: readonly [number, number] = [18, 32];
/** Open water kept between neighbouring forests, metres. */
const FOREST_GAP = 14;
/** Clear water kept around the spawn, where the hull and the anchor are. */
const SPAWN_CLEAR = 46;

// ---------------------------------------------------------------- the columns

/** Minimum spacing between column holdfasts, metres (abyssal's occupancy grid is 2.3). */
const COLUMN_SPACING = 2.9;
/** Leaves per metre of column at full fill (abyssal's 30). */
const LEAVES_PER_METRE = 26;
/** Sparkling specks per metre of column (abyssal's 13-16). */
const GLINTS_PER_METRE = 7;
/**
 * [underwater-look-3] Round 6. A share of the glints actually planted; the
 * rest are drawn and dropped so every other leaf keeps its place. See
 * `LeafField.UNDER_SPARKLE`.
 */
const GLINT_KEEP = 0.25;
/**
 * [underwater-look-3] Round 6. Single columns and small clumps standing out
 * past a forest's edge, to `OUTLIER_REACH` of its radius, at `OUTLIER_SHARE`
 * of the interior's tries. Seen from outside, a forest whose density falls
 * off inside its radius has a clean rim, and the rim read as a rank of
 * columns side by side. Drawn from their own generator after every other
 * column, and not counted in `Forest.columns`, so the forests, their order
 * in `focusPoints` and every existing column's leaves are unchanged.
 */
const OUTLIER_REACH = 1.35;
/**
 * [underwater-look-3] Round 7. Small stands of kelp round every plateau reef:
 * `STANDS_PER_REEF` of them at even bearings (jittered), `STAND_REACH` metres
 * past the reef's edge, each `STAND_RADIUS` across with `STAND_COLUMNS`
 * columns at least `STAND_SPACING` apart. The round-7 review lost the reef
 * frame to abyssal-garden's "depth layering": behind our mound there was only
 * open water, where abyssal's has canyon walls and kelp fading into the
 * blue. A stand thirty to fifty metres off is a silhouette in the water, and
 * at even bearings one stands behind a reef from wherever it is seen. Drawn
 * from each reef's own generator after every other column, not counted in
 * `Forest.columns`, and kept off every clearing (the reefs themselves, the
 * meadows and the spawn).
 */
const STANDS_PER_REEF = 5;
const STAND_REACH: readonly [number, number] = [10, 26];
const STAND_RADIUS: readonly [number, number] = [3, 6.5];
const STAND_COLUMNS: readonly [number, number] = [4, 9];
const STAND_SPACING = 1.8;
const OUTLIER_SHARE = 0.35;
/**
 * [underwater-look-3] Round 6, `clumpColumns`: the noise field's frequency
 * (features of about ten metres), the gain from its slope to a move, the
 * longest move, and the closest two holdfasts may end up.
 */
const CLUMP_FREQ = 0.1;
const CLUMP_GAIN = 14;
const CLUMP_PULL = 1.4;
const CLUMP_MIN_GAP = 1.1;
/** Water kept over a column's top leaves: wave troughs plus the leaf reach. */
const TOP_CLEARANCE = 2.0;
/** Tallest column, metres. */
const MAX_COLUMN = 26;
/** A column too short to be one is not planted. */
const MIN_COLUMN = 3;
/** Holdfast burial, as the old `HOLDFAST_SINK`: the mesh lerps the analytic floor. */
const HOLDFAST_SINK = 0.3;
/** Steepest floor a holdfast takes, as a gradient. */
const MAX_SLOPE = 0.5;

/** Distance bands, as fractions of the range: 10x3 rows, 9, 5, 3 rows at half. */
const KELP_LEVELS = [
  // [underwater-look-3] Three across and ten along in the first tenth of
  // the range, so a blade a metre or two from the lens cups and bends
  // smoothly instead of showing its two-vertex section as one flat polygon.
  // Only there: five across over the whole first band cost 0.8 ms in
  // the forest, in the main and the shadow pass both.
  { rows: 10, cols: 3, thin: 1, until: 0.1 },
  { rows: 9, thin: 1, until: 0.24 },
  { rows: 5, thin: 1, until: 0.52 },
  { rows: 3, thin: 0.5, until: 1 },
] as const;

/** A forest: a centre, a radius, and the floor there. */
export interface Forest {
  x: number;
  z: number;
  y: number;
  radius: number;
  region: 'island' | 'plateau';
  columns: number;
}

/** A disc nothing grows in: a reef site, a meadow. */
export interface Clearing {
  x: number;
  z: number;
  radius: number;
}

export interface KelpForestOptions {
  seed?: number;
  /** Discs no column is planted in — the reef sites. */
  clearings?: readonly Clearing[];
  shading?: LeafShading;
  /** Metres; the tier's `vegetationRange`. */
  range?: number;
  /** [underwater-look-3] Reef sites that get stands of kelp around them; see `STANDS_PER_REEF`. */
  reefs?: readonly Clearing[];
}

export class KelpForest {
  /** `setCount` takes a density 0..1 now: the fraction of thinnable leaves drawn. */
  static readonly MAX_COUNT = 1;

  readonly object: THREE.Object3D;
  /** The water every leaf material reads; `GroundCover` shares it. */
  readonly uniforms = new LeafUniforms();
  readonly forests: readonly Forest[];
  /** Column holdfasts, for the ground cover's understorey and for a harness. */
  readonly columns: readonly { x: number; y: number; z: number; height: number }[];

  private readonly field: LeafField;
  private disposed = false;

  constructor(density: number, options: KelpForestOptions = {}) {
    const random = mulberry32(options.seed ?? KELP_SEED);
    const floor = new FloorCache();
    const clearings = [...(options.clearings ?? []), ...defaultClearings()];
    const forests = placeForests(random, clearings);
    this.forests = forests;
    const { leaves, columns } = growForests(random, floor, forests, clearings, options.reefs ?? []);
    this.columns = columns;

    this.field = new LeafField(leaves, this.uniforms, {
      name: 'kelp-forest',
      levels: KELP_LEVELS,
      range: options.range ?? 95,
      swayMargin: 1.5,
      shading: options.shading,
    });
    this.object = this.field.object;
    this.setCount(density);
  }

  getCount(): number {
    return this.field.getDensity();
  }

  /** Density 0..1. Moves instance counts and a widening uniform; compiles nothing. */
  setCount(density: number): void {
    this.field.setDensity(density);
  }

  /** Leaves built, for a harness. */
  leafCount(): number {
    return this.field.leafCount;
  }

  setRange(metres: number): void {
    this.field.setRange(metres);
  }

  setCastShadow(cast: boolean): void {
    this.field.setCastShadow(cast);
  }

  setVisible(v: boolean): void {
    this.field.setVisible(v);
  }

  /** Kept for callers that only have a direction. */
  setSunDirection(dir: THREE.Vector3): void {
    this.uniforms.setKey(dir, this.uniforms.uKeyColor.value as THREE.Color);
  }

  setKeyLight(direction: THREE.Vector3, color: THREE.Color): void {
    this.uniforms.setKey(direction, color);
  }

  setAmbient(color: THREE.Color): void {
    this.uniforms.setAmbient(color);
  }

  /** Same three numbers the surface takes; see `LeafUniforms.setSwell`. */
  setSwell(bearingRadians: number, windSpeed: number, peakWavelength = 47): void {
    this.uniforms.setSwell(bearingRadians, windSpeed, peakWavelength);
  }

  /** Per frame: which cells are drawn at which level, and the shadow eye. */
  setCameraPosition(position: THREE.Vector3): void {
    this.uniforms.setEye(position);
    this.field.setEye(position);
  }

  setVelocity(previous: unknown, current: unknown): void {
    this.field.setVelocity(previous, current);
  }

  /** Advances the shared clock. Runs whether or not anything is visible. */
  update(dt: number): void {
    if (this.disposed) return;
    this.uniforms.update(dt);
  }

  resetClock(time = 0): void {
    if (this.disposed) return;
    this.uniforms.resetClock(time);
  }

  /** Forest centres, densest first, island forests before plateau ones. */
  focusPoints(): { x: number; y: number; z: number; radius: number; region: string }[] {
    return [...this.forests]
      .sort((a, b) => (a.region === b.region ? b.columns - a.columns : a.region === 'island' ? -1 : 1))
      .map((f) => ({ x: f.x, y: f.y, z: f.z, radius: f.radius, region: f.region }));
  }

  /** Cell meshes attached right now. */
  attachedCells(): number {
    return this.field.attachedCount();
  }

  allMeshes(): THREE.Mesh[] {
    return this.field.allMeshes();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.field.dispose();
  }
}

// ------------------------------------------------------------------ placement

/** The spawn and the plateau reef, which this module can know about itself. */
function defaultClearings(): Clearing[] {
  const out: Clearing[] = [{ x: 0, z: 0, radius: SPAWN_CLEAR }];
  for (const p of reefPatches()) out.push({ x: p.x, z: p.z, radius: REEF_BAND.patchRadius * 1.1 });
  return out;
}

function inClearing(x: number, z: number, clearings: readonly Clearing[], margin: number): boolean {
  for (const c of clearings) {
    if (Math.hypot(x - c.x, z - c.z) < c.radius + margin) return true;
  }
  return false;
}

/**
 * Forest sites: rejection-sampled on depth and slope, kept apart from each
 * other and from every clearing, alternating regions so a short list still
 * has both.
 */
function placeForests(random: () => number, clearings: readonly Clearing[]): Forest[] {
  const forests: Forest[] = [];
  const want: ('island' | 'plateau')[] = [];
  for (let i = 0; i < Math.max(ISLAND_FORESTS, PLATEAU_FORESTS); i++) {
    if (i < ISLAND_FORESTS) want.push('island');
    if (i < PLATEAU_FORESTS) want.push('plateau');
  }

  for (const region of want) {
    const [lo, hi] = region === 'island' ? ISLAND_DEPTH : PLATEAU_DEPTH;
    for (let attempt = 0; attempt < 400; attempt++) {
      const angle = random() * Math.PI * 2;
      const unit = Math.sqrt(random());
      let x: number;
      let z: number;
      if (region === 'island') {
        const r = ISLAND.radius * (ISLAND_INNER + unit * (ISLAND_OUTER - ISLAND_INNER));
        x = ISLAND.x + Math.cos(angle) * r;
        z = ISLAND.z + Math.sin(angle) * r;
      } else {
        const r = PLATEAU_INNER + unit * (PLATEAU_OUTER - PLATEAU_INNER);
        x = Math.cos(angle) * r;
        z = Math.sin(angle) * r;
      }
      const depth = -seafloorPlacementHeight(x, z);
      if (depth < lo || depth > hi) continue;
      const radius = FOREST_RADIUS[0] + random() * (FOREST_RADIUS[1] - FOREST_RADIUS[0]);
      if (inClearing(x, z, clearings, radius * 0.55)) continue;
      let crowded = false;
      for (const f of forests) {
        if (Math.hypot(x - f.x, z - f.z) < f.radius + radius + FOREST_GAP) crowded = true;
      }
      if (crowded) continue;
      forests.push({ x, z, y: -depth, radius, region, columns: 0 });
      break;
    }
  }
  return forests;
}

interface Column {
  x: number;
  y: number;
  z: number;
  height: number;
  kind: number;
  fill: number;
  crowd: number;
}

/**
 * Columns inside every forest, then the leaves on every column.
 *
 * Columns are dart-thrown at `COLUMN_SPACING` against `density`: an edge
 * falloff times a noise field that opens clearings — abyssal's `kelpDens`,
 * which combines hand-placed zones with an fbm threshold the same way.
 */
function growForests(
  random: () => number,
  floor: FloorCache,
  forests: Forest[],
  clearings: readonly Clearing[],
  reefs: readonly Clearing[],
): { leaves: Leaf[]; columns: { x: number; y: number; z: number; height: number }[] } {
  const columns: Column[] = [];
  const grid = new Map<number, Column[]>();
  const gridKey = (x: number, z: number) => (Math.floor(x / 6) + 8192) * 16384 + (Math.floor(z / 6) + 8192);
  const near = (x: number, z: number, r: number): Column[] => {
    const out: Column[] = [];
    const gx = Math.floor(x / 6);
    const gz = Math.floor(z / 6);
    const reach = Math.ceil(r / 6);
    for (let i = -reach; i <= reach; i++) {
      for (let j = -reach; j <= reach; j++) {
        const list = grid.get((gx + i + 8192) * 16384 + (gz + j + 8192));
        if (!list) continue;
        for (const c of list) if (Math.hypot(c.x - x, c.z - z) < r) out.push(c);
      }
    }
    return out;
  };

  for (const forest of forests) {
    const tries = Math.ceil(((Math.PI * forest.radius * forest.radius) / (COLUMN_SPACING * COLUMN_SPACING)) * 2.2);
    for (let i = 0; i < tries; i++) {
      const a = random() * Math.PI * 2;
      const rr = Math.sqrt(random());
      const x = forest.x + Math.cos(a) * rr * forest.radius;
      const z = forest.z + Math.sin(a) * rr * forest.radius;
      const edge = 1 - smoothstep(0.62, 1, rr);
      const patch = smoothstep(0.38, 0.54, fbm(x * 0.05, z * 0.05, 0x51));
      // A core that is always forest, so even an unlucky noise draw leaves the
      // forest a heart to swim into.
      const dens = Math.max(edge * patch, rr < 0.3 ? 0.85 : 0);
      if (random() > dens) continue;
      if (inClearing(x, z, clearings, 3)) continue;
      if (near(x, z, COLUMN_SPACING).length > 0) continue;
      const ground = floor.height(x, z);
      const depth = -ground;
      if (depth < 5 || depth > 28) continue;
      if (floor.slope(x, z) > MAX_SLOPE) continue;
      const maxH = Math.min(MAX_COLUMN, depth - TOP_CLEARANCE);
      if (maxH < MIN_COLUMN) continue;
      const height = Math.max(MIN_COLUMN, maxH * (0.62 + 0.38 * random()));
      // abyssal's kinds: 0 the ordinary column, 2 the golden one; ours is
      // sargassum gold, an eighth of the forest.
      const kind = random() < 0.12 ? 2 : 0;
      const column: Column = {
        x,
        y: ground - HOLDFAST_SINK,
        z,
        height,
        kind,
        fill: 0.55 + 0.75 * random(),
        crowd: 0,
      };
      columns.push(column);
      const key = gridKey(x, z);
      let list = grid.get(key);
      if (!list) grid.set(key, (list = []));
      list.push(column);
      forest.columns++;
    }
  }

  // [underwater-look-3] Round 6: stragglers past the edge. See `OUTLIER_REACH`.
  for (const forest of forests) {
    const rng = mulberry32((hash2(forest.x * 0.37, forest.z * 0.53) * 4294967296) >>> 0);
    const ring = Math.PI * forest.radius * forest.radius * (OUTLIER_REACH * OUTLIER_REACH - 0.8 * 0.8);
    const tries = Math.ceil((ring / (COLUMN_SPACING * COLUMN_SPACING)) * OUTLIER_SHARE);
    for (let i = 0; i < tries; i++) {
      const a = rng() * Math.PI * 2;
      const rr = Math.sqrt(0.64 + rng() * (OUTLIER_REACH * OUTLIER_REACH - 0.64));
      const x = forest.x + Math.cos(a) * rr * forest.radius;
      const z = forest.z + Math.sin(a) * rr * forest.radius;
      const h01 = rng();
      const kind = rng() < 0.12 ? 2 : 0;
      const fill = 0.45 + 0.6 * rng();
      if (inClearing(x, z, clearings, 3)) continue;
      if (near(x, z, COLUMN_SPACING * 1.3).length > 0) continue;
      const depth = -seafloorPlacementHeight(x, z);
      const ground = floor.height(x, z);
      if (depth < 5 || depth > 28 || floor.slope(x, z) > MAX_SLOPE) continue;
      const maxH = Math.min(MAX_COLUMN, Math.min(depth, -ground) - TOP_CLEARANCE);
      if (maxH < MIN_COLUMN) continue;
      const column: Column = {
        x,
        y: ground - HOLDFAST_SINK,
        z,
        height: Math.max(MIN_COLUMN, maxH * (0.5 + 0.45 * h01)),
        kind,
        fill,
        crowd: 0,
      };
      columns.push(column);
      const key = gridKey(x, z);
      let list = grid.get(key);
      if (!list) grid.set(key, (list = []));
      list.push(column);
    }
  }

  // [underwater-look-3] Round 6: clumps and gaps. See `CLUMP_PULL`.
  clumpColumns(columns, floor);
  grid.clear();
  for (const c of columns) {
    const key = gridKey(c.x, c.z);
    let list = grid.get(key);
    if (!list) grid.set(key, (list = []));
    list.push(c);
  }

  // [underwater-look-3] Round 7: stands round the reefs. See `STANDS_PER_REEF`.
  // After the clumping, so every existing column lands where it did.
  for (const reef of reefs) {
    const rng = mulberry32((hash2(reef.x * 0.91, reef.z * 0.29) * 4294967296) >>> 0);
    const base = rng() * Math.PI * 2;
    for (let k = 0; k < STANDS_PER_REEF; k++) {
      const bearing = base + (k * Math.PI * 2) / STANDS_PER_REEF + (rng() - 0.5) * 0.6;
      const reach = reef.radius + STAND_REACH[0] + rng() * (STAND_REACH[1] - STAND_REACH[0]);
      const cx = reef.x + Math.cos(bearing) * reach;
      const cz = reef.z + Math.sin(bearing) * reach;
      const standRadius = STAND_RADIUS[0] + rng() * (STAND_RADIUS[1] - STAND_RADIUS[0]);
      const want = STAND_COLUMNS[0] + Math.floor(rng() * (STAND_COLUMNS[1] - STAND_COLUMNS[0] + 1));
      let planted = 0;
      for (let t = 0; t < want * 4 && planted < want; t++) {
        const a = rng() * Math.PI * 2;
        const r = Math.sqrt(rng()) * standRadius;
        const x = cx + Math.cos(a) * r;
        const z = cz + Math.sin(a) * r;
        const h01 = rng();
        const kind = rng() < 0.12 ? 2 : 0;
        const fill = 0.5 + 0.6 * rng();
        if (inClearing(x, z, clearings, 3)) continue;
        if (near(x, z, STAND_SPACING).length > 0) continue;
        const depth = -seafloorPlacementHeight(x, z);
        const ground = floor.height(x, z);
        if (depth < 5 || depth > 28 || floor.slope(x, z) > MAX_SLOPE) continue;
        const maxH = Math.min(MAX_COLUMN, Math.min(depth, -ground) - TOP_CLEARANCE);
        if (maxH < MIN_COLUMN) continue;
        const column: Column = {
          x,
          y: ground - HOLDFAST_SINK,
          z,
          height: Math.max(MIN_COLUMN, maxH * (0.55 + 0.4 * h01)),
          kind,
          fill,
          crowd: 0,
        };
        columns.push(column);
        const key = gridKey(x, z);
        let list = grid.get(key);
        if (!list) grid.set(key, (list = []));
        list.push(column);
        planted++;
      }
    }
  }

  // Crowding: how much of the sky a column's neighbours take. Drives the baked
  // occlusion, standing in for abyssal's skylight volume.
  for (const c of columns) c.crowd = Math.min(1, (near(c.x, c.z, 6).length - 1) / 6);

  const leaves: Leaf[] = [];
  for (const c of columns) growColumn(leaves, random, c);
  return {
    leaves,
    columns: columns.map((c) => ({ x: c.x, y: c.y, z: c.z, height: c.height })),
  };
}

/**
 * [underwater-look-3] Round 6. Moves every column part of the way up the slope
 * of a smooth noise field, so the forest gathers into clumps with open water
 * between them. Dart-throwing at a fixed spacing, run to saturation, packs the
 * columns almost evenly, and from outside the forest they lined up in ranks:
 * the round-6 review's "monochrome grid of kelp". Giant kelp grows in clumps
 * where the holdfasts found rock. No draws from the layout stream, so every
 * forest and column keeps its identity; a move that would land a column on a
 * neighbour, a slope or out of its depth band is not made.
 */
function clumpColumns(columns: Column[], floor: FloorCache): void {
  const field = (x: number, z: number) => fbm(x * CLUMP_FREQ, z * CLUMP_FREQ, 0x6b1d, 2);
  const e = 0.5;
  const moved = columns.map((c) => {
    const gx = field(c.x + e, c.z) - field(c.x - e, c.z);
    const gz = field(c.x, c.z + e) - field(c.x, c.z - e);
    const g = Math.hypot(gx, gz);
    if (g < 1e-6) return null;
    const step = Math.min(CLUMP_PULL, (g / (2 * e)) * CLUMP_GAIN);
    return { x: c.x + (gx / g) * step, z: c.z + (gz / g) * step };
  });
  for (let i = 0; i < columns.length; i++) {
    const m = moved[i];
    if (!m) continue;
    const c = columns[i];
    let clash = false;
    for (let j = 0; j < columns.length && !clash; j++) {
      if (j === i) continue;
      const o = columns[j];
      if (Math.abs(o.x - m.x) < CLUMP_MIN_GAP && Math.abs(o.z - m.z) < CLUMP_MIN_GAP && Math.hypot(o.x - m.x, o.z - m.z) < CLUMP_MIN_GAP) clash = true;
    }
    if (clash) continue;
    // Accepted on the layout floor (`seafloorPlacementHeight`), seated on the
    // rendered one, like every other placement in the scene.
    const depth = -seafloorPlacementHeight(m.x, m.z);
    const ground = floor.height(m.x, m.z);
    if (depth < 5 || depth > 28 || floor.slope(m.x, m.z) > MAX_SLOPE) continue;
    const maxH = Math.min(MAX_COLUMN, Math.min(depth, -ground) - TOP_CLEARANCE);
    if (maxH < MIN_COLUMN) continue;
    c.x = m.x;
    c.z = m.z;
    c.y = ground - HOLDFAST_SINK;
    c.height = Math.min(c.height, maxH);
  }
}

/**
 * One column: abyssal-garden's `column(x, y, z, H, kind, dens)`.
 *
 * Stipes twist round the axis (`tw` rad/m) and draw in toward the top; the
 * whole column curves to one side (`la`, `lm`). Leaves are distributed up the
 * column with `t = rnd^0.9` — slightly more low down — longer at the base,
 * nearly horizontal low and lifting toward vertical near the top, which is the
 * silhouette in `kelp-forest.jpg`.
 */
function growColumn(leaves: Leaf[], rnd: () => number, c: Column): void {
  const rr = (a: number, b: number) => a + (b - a) * rnd();
  const TAU = Math.PI * 2;
  const H = c.height;
  const strands = 2 + (rnd() < 0.6 ? 1 : 0) + (rnd() < 0.3 ? 1 : 0);
  const ph = rnd() * TAU;
  const tw = rr(0.1, 0.28);
  const sr = rr(0.2, 0.55);
  const sc = rr(0.75, 1.25);
  const la = rnd() * TAU;
  // [underwater-look-3] Round 6: 0.11 -> 0.17, the same draw scaled, so each
  // column's own curve stands out more against the current's common lean.
  const lm = rr(0, 0.17);
  const stem = (s: number, h: number): [number, number, number] => {
    const a = ph + (s * TAU) / strands + h * tw;
    const r = sr * (1 - (0.6 * h) / H);
    const ln = (lm * h * h) / H;
    return [c.x + Math.cos(a) * r + Math.cos(la) * ln, c.z + Math.sin(a) * r + Math.sin(la) * ln, a];
  };
  // Occlusion: crowded columns are dark low down; open ones barely.
  const occlusion = (t: number) => {
    const lit = 0.25 + 0.75 * Math.pow(t, 0.7);
    const k = 0.3 + 0.7 * c.crowd;
    return 1 - k * (1 - lit);
  };

  // [underwater-look-3] The column's own tone, 0..1, from its position rather
  // than from the stream, so every other draw (and the forest's layout) is
  // unchanged. Leaves take most of their palette pick from it, which makes
  // one column golden-brown and its neighbour olive or green instead of every
  // column the same mix of all three.
  const tone = hash2(c.x, c.z);

  // [underwater-look-3] A holdfast: a clump of short haptera spreading from
  // the foot and bending down into the sand. Giant kelp grips the rock with a
  // tangle up to a metre across; without one a column stood on the sand like
  // a pole. Drawn from their own generator for the same reason as `tone`.
  {
    let hs = (hash2(c.z * 1.7, c.x * 0.3) * 4294967296) >>> 0;
    const hr = () => {
      hs = (hs + 0x6d2b79f5) >>> 0;
      let t = hs;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const count = 7 + Math.floor(hr() * 5);
    for (let i = 0; i < count; i++) {
      const a = (i / count) * TAU + hr() * 0.8;
      leaves.push({
        x: c.x + Math.cos(a) * 0.08, y: c.y + HOLDFAST_SINK + 0.05 + hr() * 0.25, z: c.z + Math.sin(a) * 0.08,
        len: 0.45 + hr() * 0.45, yaw: a, tilt: 1.05 + hr() * 0.35, width: 0.07 + hr() * 0.04, hA: 0,
        type: LeafType.STIPE, s1: hr(), s2: hr(), bend: 0.75 + hr() * 0.2, ao: 0.25, keep: true,
      });
    }
  }

  for (let s = 0; s < strands; s++) {
    for (let h = 0; h < H - 1; h += 1.7) {
      const p = stem(s, h);
      leaves.push({
        x: p[0], y: c.y + h, z: p[1], len: 2.1, yaw: p[2] + Math.PI / 2, tilt: 0.1, width: 0.17, hA: h,
        type: LeafType.STIPE, s1: rnd(), s2: rnd(), bend: 0.3, ao: occlusion(h / H), keep: true,
      });
    }
  }

  const n = Math.floor(H * LEAVES_PER_METRE * c.fill);
  for (let i = 0; i < n; i++) {
    const t = Math.pow(rnd(), 0.9);
    const h = t * H;
    const p = stem(Math.floor(rnd() * strands), h);
    const L = ((0.55 + 0.75 * rnd() * rnd()) * (1 - 0.5 * t * t) * sc + 0.25) * 1.15;
    const tilt = mix(rr(0.4, 1.0), rr(0.15, 0.45), t);
    const type = c.kind === 2 ? (rnd() < 0.5 ? LeafType.SARGASSUM : LeafType.KELP) : LeafType.KELP;
    leaves.push({
      x: p[0], y: c.y + h, z: p[1], len: L, yaw: rnd() * TAU, tilt, width: L * rr(0.15, 0.21) + 0.03, hA: h,
      type, s1: mix(rnd(), tone, 0.7), s2: rnd(), bend: rr(0.38, 0.95), ao: occlusion(t),
    });
  }

  // [underwater-look-3] See `GLINT_KEEP`: its own hash, not the stream.
  let glintSeed = (hash2(c.x * 0.71, c.z * 1.37) * 4294967296) >>> 0;
  for (let i = Math.floor(H * GLINTS_PER_METRE); i > 0; i--) {
    const h = rnd() * H;
    const a = rnd() * TAU;
    const r = rr(0.15, 0.8) * (1 - (0.5 * h) / H);
    const ln = (lm * h * h) / H;
    glintSeed = (Math.imul(glintSeed ^ (glintSeed >>> 15), 0x2c1b3c6d) + 0x297a2d39) >>> 0;
    const keep = ((glintSeed ^ (glintSeed >>> 12)) >>> 0) / 4294967296 < GLINT_KEEP;
    const s1 = c.kind === 2 ? rnd() * 0.6 : rnd();
    const s2 = rnd();
    if (!keep) continue;
    leaves.push({
      x: c.x + Math.cos(a) * r + Math.cos(la) * ln, y: c.y + h, z: c.z + Math.sin(a) * r + Math.sin(la) * ln,
      len: rr(0.018, 0.05), yaw: 0, tilt: 0, width: 0, hA: h,
      // s1 picks the palette entry: kelp specks gold-to-cyan, sargassum's gold-green.
      type: LeafType.GLINT, s1, s2, bend: 0, ao: 1,
    });
  }
}

// ------------------------------------------------------------------ utilities

/** A hash of a position to 0..1. */
function hash2(x: number, z: number): number {
  const h = Math.sin(x * 12.9898 + z * 78.233) * 43758.5453;
  return h - Math.floor(h);
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
