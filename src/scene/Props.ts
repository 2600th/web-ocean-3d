import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { dequantiseGeometry, fetchWithDeadline, type AssetLoader } from './AssetLoader';
import { Imposters } from './Imposters';
import { ISLAND, REEF_BAND, reefPatches, seafloorHeight } from './Seafloor';
import { SEEDS, mulberry32 } from '../core/random';

/**
 * Scene dressing: floating props near the play area, the island that sits on the
 * horizon, and the authored set pieces on it — a pirate cove on the leeward
 * shore, a ruined shore fort on the headland above it, and a wreck's worth of
 * cargo on the reef.
 *
 * Placement rules, in the order they matter:
 *
 *  - **Floaters are individual objects.** Each buoy and barrel is driven by its
 *    own `BuoyantBody`, so each needs its own transform. Instancing them would
 *    save a handful of draw calls and cost the independent bobbing that the
 *    reference is quite obviously doing.
 *  - **Everything static and repeated is instanced.** A model kind costs one
 *    draw per *material* no matter how many copies are scattered, because the
 *    loaded sub-meshes are baked and merged per material before instancing.
 *    Authored one-offs — the jetty, the pinnace, the fort, the chest — are plain
 *    meshes; there is exactly one of each and instancing them would only add
 *    indirection.
 *  - **Everything on land is seated on `seafloorHeight`.** The island is a
 *    heightfield, not a plane, and it is the same function the floor mesh is
 *    built from. Placing by radius alone put props two metres in the air on one
 *    bearing and two metres underground on the next.
 *  - **Nothing is placed at an absolute radius.** The scatter bands are
 *    fractions of `ISLAND.radius` and the set pieces are offsets from a
 *    shoreline this file *finds* with `shorelineRadius`. Everything here was
 *    once written in metres against a 260 m island whose shore was a circle at
 *    150 m; when `Seafloor` was reshaped, every one of those numbers became
 *    wrong at once and the whole cove ended up 350 m inland. Fractions and
 *    elevation bands survive that; metres do not.
 *  - **Orientation comes from the terrain, never from the island centre.** See
 *    `contourYaw`.
 *  - **Placement is seeded.** The scene must be identical on every load,
 *    otherwise reference comparison screenshots never match. The dressing draws
 *    from its *own* stream (see `DRESSING_SEED_MIX`) so that adding or removing
 *    a plant kind cannot reshuffle the floaters, the island rocks or the reef.
 *
 * On scale: the island is ~1.4 km from the origin and now covers about 0.8 km²
 * of land rising to 73 m, four times the area the old dome had. Most of it is
 * still a silhouette from the play area, so the budget goes on the things that
 * survive that distance — coastal cliffs, wave-cut shelves, and canopy tall
 * enough to break the skyline — and the understorey exists for the fly camera
 * rather than the horizon, which is why the detail scale thins it hardest.
 */

const BUOY_URL = '/models/ocean_buoy/ocean_buoy_1k.gltf';
const BARREL_URL = '/models/barrel_03/barrel_03_1k.gltf';

/** Scene-dressing library. Every entry is optional: a 404 thins the scene. */
const DRESSING_URLS = {
  // Shore rock. The three `rock_*` kinds are closed solids and the coast kinds
  // are not — see `headlandRock`'s note in `dressIsland` for why that decided
  // which of them survived.
  headlandRock: '/models/dressing/rock_slab_a.glb',
  shoreOutcrop: '/models/dressing/rock_slab_b.glb',
  shoreBoulder: '/models/dressing/rock_boulder.glb',
  coastLineWide: '/models/dressing/coast_line_01.glb',
  coastLineNarrow: '/models/dressing/coast_line_02.glb',
  coastRocksWide: '/models/dressing/coast_rocks_01.glb',
  coastRocksTall: '/models/dressing/coast_rocks_03.glb',
  landRocks: '/models/dressing/coast_land_rocks_03.glb',
  // The island skirt and the reef. These two were loaded raw out of
  // `public/models` until the LOD pass, because they predate the dressing
  // pipeline — which meant the scene's two largest instanced costs were the
  // only kinds in it without a LOD chain. `rock_07` alone stands 154 times.
  islandRock: '/models/dressing/rock_07.glb',
  // The canopy. `tree` and `treeFlame` are botanical models; `treeMid` is the
  // one Poly Haven scan good enough to stand with them.
  tree: '/models/dressing/tree_orchid.glb',
  treeFlame: '/models/dressing/tree_poinciana.glb',
  treeMid: '/models/dressing/island_tree_02.glb',
  jacaranda: '/models/dressing/jacaranda_tree.glb',
  pachira: '/models/dressing/pachira_aquatica_01.glb',
  fern: '/models/dressing/fern_02.glb',
  sorrel: '/models/dressing/shrub_sorrel_01.glb',
  anthurium: '/models/dressing/anthurium_botany_01.glb',
  calathea: '/models/dressing/calathea_orbifolia_01.glb',
  pinnace: '/models/dressing/ship_pinnace.glb',
  pier: '/models/dressing/modular_wooden_pier.glb',
  fort: '/models/dressing/modular_fort_01.glb',
  cannon: '/models/dressing/cannon_01.glb',
  barrels: '/models/dressing/wooden_barrels_01.glb',
  lantern: '/models/dressing/wooden_lantern_01.glb',
  coveCrate: '/models/dressing/wooden_crate_02.glb',
  bucket: '/models/dressing/wooden_bucket_01.glb',
  jug: '/models/dressing/jug_01.glb',
  estoc: '/models/dressing/antique_estoc.glb',
  chest: '/models/dressing/treasure_chest.glb',
  reefCrate: '/models/dressing/wooden_crate_01.glb',
  shell: '/models/dressing/lambis_shell.glb',
  // The reef garden. See `placeReefGarden` for why these are the shapes and
  // ASSET_LICENSES.md for where they come from — these are the only assets in
  // the project not from Poly Haven, because Poly Haven publishes no coral.
  coralSoft: '/models/dressing/soft_coral_set.glb',
  coralFan: '/models/dressing/stylaster_coral.glb',
  coralBirdsnest: '/models/dressing/seriatopora_coral.glb',
  coralBrain: '/models/dressing/goniastrea_coral.glb',
  // Ground cover, replacing the lawn-grass sprig that used to carry the island.
  grassMeadow: '/models/dressing/grass_medium_01.glb',
  grassTussock: '/models/dressing/grass_medium_02.glb',
  // The palms. These replace `Remains.ts`'s procedural one — see the note over
  // their scatters in `dressIsland`.
  palmCoconut: '/models/dressing/palm_coconut.glb',
  palmTall: '/models/dressing/palm_tall.glb',
} as const;

type DressingKey = keyof typeof DRESSING_URLS;

/**
 * One dressing asset: the full-detail model and whatever LOD levels ship with it.
 *
 * `scripts/optimize-assets.mjs` writes `<slug>_lod1.glb` and `<slug>_lod2.glb`
 * beside the base model for the kinds that earn them, so the URLs are derived
 * rather than listed — there is no second table to keep in step with the first.
 * A level that does not exist is simply absent, and a kind with no levels draws
 * exactly as it did before any of this.
 */
interface DressingEntry {
  base: THREE.Group | null;
  lods: THREE.Group[];
}

type Dressing = Record<DressingKey, DressingEntry>;

export type PropsQualityTier = 'low' | 'medium' | 'high' | 'ultra' | 'max';
export type PropsImposterMode = 'off' | 'always' | 'hybrid';

interface ImposterKind {
  species: string;
  meshCullDistance: number;
  minDistance: number;
  maxDistance: number;
}

const IMPOSTER_KINDS: Readonly<Record<string, ImposterKind>> = {
  'island-jacaranda': { species: 'jacaranda_tree', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-trees': { species: 'tree_orchid', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-flame-trees': { species: 'tree_poinciana', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-trees-mid': { species: 'island_tree_02', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-palms': { species: 'palm_coconut', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-palms-tall': { species: 'palm_tall', meshCullDistance: 120, minDistance: 120, maxDistance: 600 },
  'island-anthurium': { species: 'anthurium_botany_01', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
  'island-calathea': { species: 'calathea_orbifolia_01', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
  'island-ferns': { species: 'fern_02', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
  'island-sorrel': { species: 'shrub_sorrel_01', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
  'island-meadow': { species: 'grass_medium_01', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
  'island-tussock': { species: 'grass_medium_02', meshCullDistance: 60, minDistance: 60, maxDistance: 400 },
};

const IMPOSTER_MANAGED_KEYS = new Set<DressingKey>([
  'tree',
  'treeFlame',
  'treeMid',
  'jacaranda',
  'palmCoconut',
  'palmTall',
  'anthurium',
  'calathea',
  'fern',
  'sorrel',
  'grassMeadow',
  'grassTussock',
]);

const IMPOSTER_NO_CULL_DISTANCE = 1_000_000;

const LOD_TEXTURE_SLOTS = [
  'map',
  'roughnessMap',
  'metalnessMap',
  'normalMap',
  'aoMap',
  'emissiveMap',
  'alphaMap',
  'lightMap',
  'bumpMap',
  'displacementMap',
] as const;
/**
 * Distances at which an instance drops to the next level, metres.
 *
 * Set against the two places the camera actually is. From the play area the
 * island is 1.4 km away, so everything on it is past the second switch and wears
 * LOD2 — which is the case that matters, because that is the view the scene
 * spends most of its time in and the island is most of its triangles. From the
 * fly camera over the island, the trees within a hundred metres keep LOD0 and
 * the far side of the island does not.
 *
 * 120 m is roughly where a 5 m tree stops resolving its individual leaves at
 * this field of view, and 420 m is where its canopy is a shape rather than a
 * texture. Both are generous: a LOD that switches too late costs frame time,
 * and one that switches too early is visible as a pop.
 */
const LOD_SWITCH_METRES = [120, 420];

/**
 * How far the camera must move before the levels are re-dealt, metres.
 *
 * The deal is O(instances) and uploads a matrix buffer, so it must not run every
 * frame; it also must not lag far enough behind the camera for a switch to
 * happen visibly late. 25 m is a fifth of the near switch, which bounds the
 * error in where the boundary falls to something well inside the distance at
 * which the two levels are distinguishable.
 */
const LOD_REFRESH_DISTANCE = 25;

/** Shared empty result, so the common case allocates nothing. */
const EMPTY_MESHES: THREE.InstancedMesh[] = [];

const BUOY_COUNT = 5;
const BARREL_COUNT = 6;
const ROCK_COUNT = 34;

/** Reef outcrops on the seafloor. One instanced draw, so this can be generous. */
const REEF_COUNT = 120;
/** Clear of the spawn point, and inside the shallow plateau (radius 320 m). */
const REEF_INNER = REEF_BAND.inner;
const REEF_OUTER = REEF_BAND.outer;
const REEF_PATCH_RADIUS = REEF_BAND.patchRadius;

/**
 * Fraction of the reef rock that gathers into the patches.
 *
 * The rest stays scattered as the rubble between them — a reef has an apron. See
 * `reefPatches` in `Seafloor` for why the patches exist at all and why they are
 * shared with the fish rather than drawn here.
 */
const REEF_PATCH_SHARE = 0.62;

/**
 * The reef garden.
 *
 * Two jobs, and they want different assets. `coralSoft` is the carpet — small
 * varied heads at high density, which is what makes a patch read as *reef*
 * rather than as rocks with decoration on them. The three photogrammetry
 * colonies are the statement pieces: an order of magnitude more expensive per
 * instance, so they are placed an order of magnitude less often, one or two per
 * patch where the eye lands.
 *
 * Counts are per detail scale 1, thinned by `propsDetail` like everything else.
 */
const CORAL_BED_COUNT = 260;
const CORAL_BED_ALT_COUNT = 200;
const CORAL_HEAD_COUNT = 150;
const CORAL_HEAD_ALT_COUNT = 110;
const CORAL_PLATE_COUNT = 60;
const CORAL_FAN_COUNT = 24;
const CORAL_BIRDSNEST_COUNT = 9;
const CORAL_BRAIN_COUNT = 13;

/**
 * Metres a coral stands, before per-instance variation.
 *
 * The source scales differ by an order of magnitude — `stylaster_coral` is a
 * 17 cm museum specimen, the kit pieces are authored against a unit box — so
 * every kind is scaled to a target *height* rather than by a factor. That keeps
 * one comparable table instead of a column of magic numbers whose meaning
 * depends on which file the row came from.
 */
const CORAL_BED_HEIGHT = 0.9;
const CORAL_HEAD_HEIGHT = 1.8;
const CORAL_PLATE_HEIGHT = 1.0;
const CORAL_FAN_HEIGHT = 2.1;
const CORAL_BIRDSNEST_HEIGHT = 1.7;
const CORAL_BRAIN_HEIGHT = 1.3;

/**
 * Radius of a coral colony's own clump, metres, and how many heads it gets.
 *
 * Reef growth is not a Poisson scatter. It colonises a patch of hard substrate
 * and spreads from it, so the spatial signature is clumps of a few heads with
 * bare sand between — and a uniform draw inside the patch, which is what the
 * first version of this did, reads as gravel. Two levels of clustering (patch,
 * then colony) is the cheapest way to get the third scale of structure that
 * makes a reef look grown rather than sprinkled.
 */
const CORAL_COLONY_RADIUS = 2.6;
const CORAL_COLONY_MIN = 3;
const CORAL_COLONY_MAX = 7;

/**
 * Metres every coral is pushed into the sand on top of its proportional sink.
 *
 * The floor *mesh* is a 256-segment grid over a 4 km extent, so it samples
 * `seafloorHeight` every fifteen metres and lerps between; the analytic field
 * these are seated against curves away from that by a decimetre or two in the
 * worst places. A head seated at exactly the analytic height therefore hovers
 * about half the time, and a hovering coral is instantly a bug — `Kelp` carries
 * the same constant for the same reason. Burying the base costs nothing: it is
 * the part of a coral that is cemented to the substrate anyway.
 */
const CORAL_BED_SINK = 0.22;

/**
 * Instance counts for the island dressing at detail 1.
 *
 * These are triangle budgets as much as they are art direction. The photogram-
 * metry assets are decimated LOD0 scans — a canopy tree is ~34k triangles, the
 * long rampart cliff is 92k — so a count here is worth ~30-90k triangles, and
 * the number that makes the island read is a lot smaller than the number that
 * would make it look dense from ten metres away.
 *
 * They went up across the board when the island did. On the old 260 m dome nine
 * trees covered the whole of it; on 0.8 km² of land the same nine read as a bare
 * rock with a few sticks on it, which is the failure these counts exist to fix.
 * Rock is the expensive half and it went up least: a coastline 3.4 km long can
 * never be *covered* by 40 m scans, so the coast kinds are accents chosen for
 * where they land, not for how much of the shore they fill.
 */
const ISLAND_CRAG_COUNT = 22;
const HEADLAND_ROCK_COUNT = 34;
const SHORE_OUTCROP_COUNT = 26;
const SHORE_BOULDER_COUNT = 54;
const COAST_LINE_WIDE_COUNT = 4;
const COAST_LINE_NARROW_COUNT = 4;
const COAST_ROCKS_WIDE_COUNT = 4;
const COAST_ROCKS_TALL_COUNT = 4;
const LAND_ROCKS_COUNT = 4;
/**
 * Planting density, and these numbers are the difference between an island and
 * a sandbank.
 *
 * The first pass at the enlarged island scaled the old counts by about 2.3x on
 * 4x the land area, which is a *thinning* — and an aerial capture showed exactly
 * that: a white dome with objects sprinkled on it. Canopy has to close over the
 * interior for the island to read as vegetated at all, because what a viewer
 * registers from a mile out is the ratio of green to sand and nothing else.
 *
 * These are per-kind capacities, thinned by `propsDetail` — 0.3 at Low — so the
 * high-water mark is what Ultra and Max draw and the lower tiers keep the shape
 * and lose the density. The reef, at 90 instances of a 15k-triangle rock, is
 * still the largest single cost in this file; the entire canopy is less.
 */
const JACARANDA_COUNT = 14;
const TREE_COUNT = 52;
const TREE_MID_COUNT = 44;
/**
 * The flame trees, and why there are so few of them.
 *
 * A royal poinciana in full flower is the loudest thing that grows on an island
 * and the reference frame has four of them in a canopy of hundreds. That ratio
 * is the point: at twenty they are what the eye finds in the green, and at sixty
 * the island is an orange hill.
 */
const TREE_FLAME_COUNT = 20;
const PACHIRA_COUNT = 40;
const ANTHURIUM_COUNT = 70;
const CALATHEA_COUNT = 84;
const FERN_COUNT = 110;
const SORREL_COUNT = 130;
const MEADOW_COUNT = 380;
const TUSSOCK_COUNT = 200;

/**
 * Palms, and why there are two kinds of them.
 *
 * A coconut grove is the one silhouette everybody associates with a tropical
 * island, so it is also the one where a single repeated shape is most obvious.
 * `palmCoconut` is a low double-trunk bearing fruit and `palmTall` is a tall
 * slender single; between them the shoreline reads as a grove rather than as
 * one asset stamped seventy times.
 *
 * Weighted toward the coconut because it is the cheaper mesh and the better
 * shape: 6.6k triangles against 9.9k, and it breaks the skyline at two heights
 * on its own.
 */
const PALM_COCONUT_COUNT = 54;
const PALM_TALL_COUNT = 34;
const SHELL_COUNT = 9;

/**
 * XOR mix that derives the dressing's PRNG seed from the props seed.
 *
 * A separate stream, not a continuation of the same one: `Props` writes the
 * floaters, the island rocks and the reef from `SEEDS.props`, and if the
 * dressing drew from that same generator then adding one fern would move every
 * buoy in the scene. Golden-ratio constant, which is only to say "some fixed
 * number with a well-spread bit pattern".
 */
const DRESSING_SEED_MIX = 0x9e3779b9;

/**
 * Rejected samples per instance before a scatter gives up on that instance.
 *
 * Raised from 24 when the bands became elevation-led. The tightest of them is
 * the waterline strip the `coast_line` shelves want, which is about a tenth of
 * the annulus they are drawn from; at 24 attempts one shelf in five never
 * placed, and a kind with four instances cannot afford to lose one. Placement
 * runs once at load, so the cost of the extra attempts is unmeasurable.
 */
const PLACEMENT_ATTEMPTS = 40;

/**
 * Fraction of an instanced kind that survives however low the detail scale
 * goes. A kind that thins to nothing changes the island's shape rather than its
 * density, which is the one thing the detail scale must not do.
 */
const DETAIL_FLOOR = 0.25;

// ------------------------------------------------------- terrain-derived yaw

/**
 * Gradient below which "downhill" stops meaning anything and `contourYaw` hands
 * back to the random draw. A tenth of a percent of grade is half a metre over
 * the length of the longest cliff slab, which is noise.
 */
const CONTOUR_MIN_GRADE = 1e-3;
/** Peak random yaw laid on top of a contour-derived facing, radians. */
const CONTOUR_JITTER = 0.3;

// ------------------------------------------------------------------ the cove

/**
 * Bearing of the cove from the island centre, radians in the same convention as
 * the scatter code (x = cos, z = sin).
 *
 * `Seafloor` puts its lagoon sector on this bearing and keeps it navigable, so
 * the two files have to agree on the number. The default wind blows toward
 * +x/+z (`Spectrum` defaults to pi/4), so this face is the lee — the only shore
 * of the island where a boat could be left on a mooring. It also happens to be
 * the face that looks back at the play area, which is what makes the cove worth
 * building at all.
 */
const COVE_BEARING = 0.7;

/**
 * Bearing of the headland from the island centre.
 *
 * `Seafloor` pushes the shore out and raises a ridge on this bearing, giving the
 * one genuinely exposed piece of ground on the island — which is why the
 * wind-shorn planting is confined to an arc about it. Written out here rather
 * than exported from `Seafloor` because it is art direction on this side of the
 * fence: `Props` is choosing where the wind gets to matter, not reading a fact
 * about the terrain.
 */
const HEADLAND_BEARING = 1.45;

/**
 * Everything in the cove is placed as metres from the *waterline on the cove's
 * bearing*, which `shorelineRadius` finds by asking the heightfield. Positive is
 * seaward.
 *
 * These were radii from the island centre — `JETTY_RADIUS = 159` and so on —
 * back when the shore on this bearing was a circle at 150 m. It is now at 501 m
 * and it is not a circle, so those numbers put the whole cove a third of the way
 * up the hill. An offset from the water cannot go wrong that way: the shore can
 * move anywhere it likes and the jetty follows it.
 */
const JETTY_OFFSHORE = 5;
const JETTY_SCALE = 1.5;
/** Deck height above mean sea level, metres. */
const JETTY_DECK_Y = 2.1;
/** Deck surface in the pier model's own units, for seating things on it. */
const PIER_DECK_LOCAL = 2.67;

const PINNACE_OFFSHORE = -7;
const PINNACE_ALONGSHORE = -30;
const PINNACE_SCALE = 0.55;
/** Metres the hull is lifted off the sand, so the keel bites rather than floats. */
const PINNACE_KEEL_LIFT = 1.05;
const PINNACE_HEEL = 0.17;
/**
 * Bow-up trim, radians. Negative pitches the bow up in a YXZ rotation.
 *
 * This is the beach gradient and nothing else, so it moved when the beach did:
 * the cove now falls about a tenth of a metre per metre instead of a fifth, and
 * the hull lies mostly *along* the beach rather than up it, which leaves about a
 * twentieth of grade under the keel. At the old -0.09 the boat was correcting
 * for twice the slope it is actually sitting on and stood on its stern.
 */
const PINNACE_TRIM = -0.05;

/**
 * The camp, on the dry beach above the swash. `Seafloor` washes the sand within
 * ~3 m of sea level, and the beach here falls about a metre per ten, so this is
 * the first stretch that is dry ground rather than wet sand.
 */
const CAMP_OFFSHORE = -46;

/**
 * The sword. The scan stands point *up* with its pommel 0.2 m below the origin,
 * so it is turned over before it is planted — hence the pi in the rotation and
 * `ESTOC_POINT`, which is where the tip ends up once it has been.
 *
 * Thrust into the sand rather than laid on it. A sword lying flat on a beach is
 * a 4 cm silhouette from any camera higher than the dune line; standing, it
 * casts a shadow across the sand and reads from the water.
 */
const ESTOC_SCALE = 1.15;
const ESTOC_LEAN = 0.3;
/** Model-space metres from the origin to the point, at unit scale. */
const ESTOC_POINT = 1.29;
/** Metres of blade buried. */
const ESTOC_BURY = 0.35;

/**
 * The three jugs, in the group's own frame, world metres.
 *
 * Two upright and one on its side, which is the whole difference between "a
 * camp" and "three jugs left in a row". The tipped one's `y` rests it on its
 * shoulder: a couple of centimetres of it buried in the sand is invisible, and a
 * couple of centimetres floating is not.
 */
const JUG_PIECES: readonly KitPiece[] = [
  { x: 0, z: 0, yaw: 0.4, scale: 2.2 },
  { x: 0.62, z: 0.38, yaw: 2.1, scale: 2.2 },
  { x: -0.3, y: 0.14, z: 0.66, yaw: 5, tilt: 1.45, scale: 2.2 },
];

// ------------------------------------------------------------- the shore fort

/**
 * The fort sits on the headland's tableland, which `Seafloor` holds at ~35 m out
 * to 550 m before dropping it 22 m into a bluff.
 *
 * Bearing and inset were chosen against the heightfield rather than by eye: over
 * a 44 x 32 m footprint here the ground varies by half a metre, and masonry is
 * built level. Anywhere nearer the brow the same footprint spans five metres of
 * fall, which no amount of sinking makes look like a wall rather than a
 * landslide.
 */
const FORT_BEARING = 1.35;
/** Metres back from the waterline, as a fraction of `ISLAND.radius`. */
const FORT_INSET = 0.34;
/** Metres the whole assembly is pushed into the ground, to bury the half-metre. */
const FORT_SINK = 0.8;
/**
 * How far offshore of the cove's beach the fort is laid.
 *
 * A shore battery is laid on the channel a ship has to cross, not on the open
 * sea, so the fort is turned to face a point out in the lagoon rather than
 * turned to face outward. That works out almost exactly alongshore, which is
 * also what puts its wall face toward the play area instead of its back.
 */
const FORT_AIM_OFFSHORE = 90;

/**
 * The fort as built, in its own frame: +Z toward the water it commands, +X to
 * the right of that, y = 0 at the ground.
 *
 * The asset is a *kit* — twenty-two wall, tower and walkway pieces laid out side
 * by side in the source file, not an assembled fort — so something has to
 * assemble it, and `assemble` is that something.
 *
 * Ruined on purpose, and the ruin is what makes the kit usable: the corner
 * pieces would have to meet their neighbours to within a few centimetres to
 * close a wall, and nothing here knows which quadrant a given corner turns. A
 * broken curtain wall with a breach in it needs no piece to meet any other.
 */
const FORT_PIECES = [
  // The seaward tower, on the flank nearest the cove: 15.8 m across and 13.4 m
  // tall, so it is the piece that carries the fort from the play area. Spans
  // x +7.5..+23.3.
  { node: 'modular_fort_01_tower_round', x: 15.4, z: 6.5 },
  // Curtain wall across the front. The sections are authored running along
  // their own Z, so each is turned a quarter turn to lie along the face. The
  // two of them span x -26.3..+1.3, which leaves 6 m of nothing between the
  // wall and the tower: the breach, and the reason the gun has a field of fire.
  { node: 'modular_fort_01_wall_thick_straight_01', x: -4.4, z: 10, yaw: Math.PI / 2 },
  { node: 'modular_fort_01_wall_thick_straight_02', x: -19, z: 10, yaw: Math.PI / 2 },
  // The broken end, settled and tipped along its own length: where the wall
  // stops rather than where it was built to stop.
  { node: 'modular_fort_01_wall_thick_end_02', x: -28.7, z: 10, y: -0.5, yaw: Math.PI / 2, tilt: 0.08 },
  // Landward flank and its gate, returning inland from that end. Subsiding
  // slightly, which is the cheapest legible difference between a ruin and a
  // building site.
  { node: 'modular_fort_01_wall_thin_straight_02', x: -29, z: 1, tilt: 0.05 },
  { node: 'modular_fort_01_wall_thin_gate_01', x: -29, z: -10.1 },
  // Inside: the stair up to the fighting step, and the step itself behind the
  // curtain. Without them the wall reads as a fence.
  { node: 'modular_fort_01_wall_stairs_straight_01', x: -22, z: 0 },
  { node: 'modular_fort_01_wall_walkway_straight_01', x: -4.4, z: 6.2, yaw: Math.PI / 2 },
] as const;

/**
 * The gun, in the fort's frame: laid in the breach between the end of the
 * curtain wall (which stops at x = +1.3) and the tower (which starts at +7.5).
 *
 * Parented to the fort rather than placed in world space, for the same reason
 * the lantern is parented to the jetty — a gun in a breach is *in* the breach,
 * and moving the fort must not leave it standing in open grass. The local y
 * undoes `FORT_SINK`, so the carriage sits on the ground the walls are dug into.
 */
const CANNON_LOCAL = { x: 4.4, z: 9.5, yaw: 0.18 } as const;
const CANNON_SCALE = 1.7;

/** Centre of the underwater find, world metres. A local high on the plateau. */
const FIND = { x: -38, z: -66 } as const;
/** Radius the shells scatter over, around the chest. */
const FIND_SPREAD = 13;

const UP = new THREE.Vector3(0, 1, 0);
const TAU = Math.PI * 2;

/**
 * Bracket `shorelineRadius` bisects in, as fractions of `ISLAND.radius`.
 *
 * The shore fraction `Seafloor` computes runs from 0.72 at the head of the bay
 * to 1.30 at the tip of the headland, so this brackets the lot with room either
 * side; the inner end has to stay dry land and the outer end has to stay water
 * on every bearing this is asked about.
 */
const SHORE_SEARCH_IN = 0.4;
const SHORE_SEARCH_OUT = 1.7;
/** Halvings. 24 resolves the waterline to well under a millimetre. */
const SHORE_SEARCH_STEPS = 24;

export interface Floater {
  object: THREE.Object3D;
  /** Effective flotation radius in metres, for buoyancy probe layout. */
  radius: number;
}

export interface PropsOptions {
  /** Overrides the seeded layout; useful for A/B-ing a dressing pass. */
  seed?: number;
  /**
   * Density of the scattered dressing, 0..1. Scales the instance count of every
   * repeated kind — planting, shore rocks, island rocks and the reef — without
   * touching the authored set pieces, which are the parts that carry meaning
   * rather than density.
   *
   * Applied by moving `InstancedMesh.count`, never by rebuilding: a tier change
   * can call `setDetailScale` on a live scene for the cost of a few integer
   * writes. Defaults to 1.
   */
  detailScale?: number;
  /** Quality tier used to select the default mesh/imposter policy. */
  qualityTier?: PropsQualityTier;
  /** Explicitly overrides the quality tier policy. */
  imposterMode?: PropsImposterMode;
}

/** An instanced kind the detail scale is allowed to thin. */
interface Thinnable {
  /** One mesh per material, all driven by the same instance matrices. */
  meshes: THREE.InstancedMesh[];
  /** Instances actually written. The ceiling for `count`. */
  capacity: number;
  /** Count below which this kind stops thinning. */
  floor: number;
  /** Present when the kind ships a LOD chain; see `LodKind`. */
  lod?: LodKind;
}

/**
 * A kind drawn at whichever level of detail each instance has earned.
 *
 * `THREE.LOD` is no use here: it switches a whole object, and every kind in this
 * file is one `InstancedMesh` carrying up to six hundred instances spread over a
 * kilometre of island. Half of them can be forty metres away and half of them
 * fourteen hundred, and picking one level for the group means picking it wrong
 * for one of those halves.
 *
 * So each level gets its own mesh, sized for the whole population, and the
 * instances are dealt between them by distance. The matrices are held on the CPU
 * once and re-dealt only when the camera has actually moved — see
 * `LOD_REFRESH_DISTANCE`, which is what keeps this off the per-frame budget.
 */
interface LodKind {
  /** `levels[0]` is LOD0. One entry per level, each one mesh per material. */
  levels: THREE.InstancedMesh[][];
  /**
   * Squared distance at which each level gives way to the next. The last level
   * has no bound — it draws everything past the one before it.
   */
  switchesSq: number[];
  /** Squared distance beyond which this kind is not drawn. */
  cullSq: number;
  /** Every placement, in the seeded order `setDetailScale` truncates. */
  matrices: Float32Array;
  /** World position of each placement, for the distance test. */
  centres: Float32Array;
  /** Camera position the current deal was computed for. */
  dealtAt: THREE.Vector3;
  dealt: boolean;
}

/** A loaded asset baked down to one geometry per material. */
interface BakedPart {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
}

type BakeOrigin =
  /** Keep the asset's own origin. For models that are already assembled. */
  | 'asset'
  /** Recentre the selection as a whole on its footprint, base at y = 0. */
  | 'cluster'
  /**
   * Recentre every node individually, collapsing a row of variants laid out
   * side by side in the source file into one overlapping clump.
   */
  | 'stack';

interface BakeOptions {
  /** Keeps only the nodes whose name passes this test. */
  include?: (name: string) => boolean;
  origin?: BakeOrigin;
}

/** One placed copy of a source node inside an assembled set piece. */
interface KitPiece {
  /** Node to take from the source. Omitted takes the whole model. */
  node?: string;
  /** Position in the assembly's own frame, world metres; y = 0 is the base. */
  x: number;
  y?: number;
  z: number;
  /** Yaw about the assembly's up axis, radians. */
  yaw?: number;
  /**
   * Tip about the piece's *own* X axis, applied before the yaw — so a wall
   * leans along its own length and a jug rolls onto its side. What makes a
   * ruin read as a ruin rather than as a building site.
   */
  tilt?: number;
  scale?: number;
}

/** How one model kind scatters across the island. */
interface ScatterSpec {
  /**
   * Radial band the sampler draws from, as a fraction of `ISLAND.radius`.
   *
   * Fractions, and named `inner`/`outer` rather than `minRadius`/`maxRadius`, so
   * that no metre value from the old island could survive the change unnoticed
   * — every one of them was silently wrong once the shore moved from 150 m to
   * 501 m. The band is only a *sampler* hint: it exists to keep rejection
   * sampling converging, and the elevation band below is what actually decides
   * where a kind lives.
   */
  inner: number;
  outer: number;
  /** Elevation band a sample must land in, metres relative to sea level. */
  minHeight: number;
  maxHeight: number;
  /** Steepest ground the kind will sit on, as a gradient (rise over run). */
  maxSlope?: number;
  /** Confines the kind to an arc; omitted means the whole island. */
  bearing?: number;
  /** Half-width of that arc, radians. */
  spread?: number;
  /** Uniform scale range. */
  minScale: number;
  maxScale: number;
  /** Extra vertical stretch on top of the uniform scale. */
  minStretch?: number;
  maxStretch?: number;
  /** Metres sunk below the ground per unit of instance scale. */
  sink?: number;
  /** How far the model tips toward the ground normal, 0..1. */
  slope?: number;
  /** Half-width of the terrain sample the normal comes from, metres. */
  slopeSpan?: number;
  /** Peak random tilt added on top of the ground normal, radians. */
  lean?: number;
  /** Where the model's long axis points. */
  facing?: 'random' | 'contour';
  /** Number of clumps to gather the instances into; 0 scatters evenly. */
  clusters?: number;
  /** Radius of one clump, metres. */
  clusterRadius?: number;
  /**
   * Whether this kind casts a shadow. Receiving is always on — it is a term in
   * a shader the object already runs — but casting means drawing the kind a
   * second time into the depth map, which is not worth it for ground cover.
   */
  casts?: boolean;
  bake?: BakeOptions;
}

export class Props {
  readonly object: THREE.Object3D;
  readonly floaters: Floater[] = [];
  readonly imposters: Imposters | null;

  private readonly ownedGeometries: THREE.BufferGeometry[] = [];
  private readonly sources: THREE.Group[] = [];
  private readonly thinnable: Thinnable[] = [];
  /** Where the reef gathers. Shared by the rock scatter and the coral garden. */
  private reefPatches: readonly { x: number; z: number }[] = [];
  private detail = 1;
  /**
   * Whether vegetation is meshes, cards, or both — decided once, at load.
   *
   * `readonly` is a real limitation and not an oversight. The mode determines
   * which GLBs are *requested at all*: Low and Medium never fetch the thirteen
   * imposter-replaced species, which is where most of their saving comes from.
   * Changing it mid-session would therefore mean either downloading a set the
   * tier deliberately skipped, or disposing meshes a later upgrade could not
   * get back without a network round trip in the middle of a frame.
   *
   * The consequences, stated plainly: a High-to-Low change keeps the meshes it
   * already downloaded, so it sheds their draw calls but not their VRAM; and a
   * Low-to-High change cannot restore meshes that were never fetched, so it
   * stays on cards. Both are acceptable because the pre-load gate is the
   * primary mechanism and `AdaptiveQuality` is one-way within a session; a
   * visitor who wants the other tier's assets reloads, which the gate makes a
   * deliberate, one-click action.
   */
  private readonly imposterMode: PropsImposterMode;
  private disposed = false;

  private constructor(
    sources: LoadedSources,
    dressing: Dressing,
    seed: number,
    detailScale: number,
    imposterMode: PropsImposterMode,
    imposters: Imposters | null,
  ) {
    this.object = new THREE.Group();
    this.object.name = 'props';
    this.imposterMode = imposterMode;
    this.imposters = imposters;
    this.sources.push(sources.buoy, sources.barrel);
    for (const entry of Object.values(dressing)) {
      if (entry.base) this.sources.push(entry.base);
      for (const lod of entry.lods) this.sources.push(lod);
    }

    const random = mulberry32(seed);

    this.placeFloaters(sources.buoy, sources.barrel, random);
    this.placeIsland(dressing, random);
    this.reefPatches = reefPatches();
    this.placeReef(dressing, random);

    const dressingRandom = mulberry32((seed ^ DRESSING_SEED_MIX) >>> 0);
    this.placeReefGarden(dressing, dressingRandom);
    this.dressIsland(dressing, dressingRandom);
    this.placeCove(dressing);
    this.placeFort(dressing);
    this.placeFind(dressing, dressingRandom);

    if (this.imposters) this.object.add(this.imposters.object);
    this.imposters?.finalize();
    this.setDetailScale(detailScale);
  }

  static async load(loader: AssetLoader, options: PropsOptions = {}): Promise<Props> {
    const dressingKeys = Object.keys(DRESSING_URLS) as DressingKey[];
    const requestedMode = resolveImposterMode(options);

    const [hero, manifest, imposters] = await Promise.all([
      Promise.all([loader.load(BUOY_URL), loader.load(BARREL_URL)]),
      loadDressingManifest().catch((error: unknown) => {
        console.warn('[ocean] dressing manifest unavailable; loading base models only', error);
        return null;
      }),
      requestedMode === 'off'
        ? Promise.resolve(null)
        : Imposters.load().catch((error: unknown) => {
            console.warn('[ocean] vegetation imposters unavailable; falling back to meshes', error);
            return null;
          }),
    ]);

    // If the atlas is unavailable, keep the source models in the request plan
    // even when the caller asked for Low/Medium. A failed optional optimisation
    // must not turn the island into an empty field.
    const imposterMode = imposters ? requestedMode : 'off';
    const requests = dressingKeys.flatMap((key) => {
      if (imposterMode === 'always' && IMPOSTER_MANAGED_KEYS.has(key)) return [];

      const model = manifest?.models[modelSlug(DRESSING_URLS[key])];
      const lods =
        model?.lods
          .slice()
          .sort((a, b) => a.level - b.level) ?? [
          { level: 0, ratio: 1, url: DRESSING_URLS[key] },
        ];
      return [{ key, lods }];
    });

    const loaded = await Promise.all(
      requests.map(async (request) => ({
        request,
        results: await Promise.allSettled(
          request.lods.map((lod) => loader.load(lod.url)),
        ),
      })),
    );
    const resultsByKey = new Map(
      loaded.map(({ request, results }) => [request.key, results]),
    );

    const [buoy, barrel] = hero;
    for (const group of hero) group.updateMatrixWorld(true);

    const dressing = {} as Dressing;
    for (const key of dressingKeys) {
      const entry: DressingEntry = { base: null, lods: [] };
      const request = requests.find((candidate) => candidate.key === key);
      const results = resultsByKey.get(key);
      if (!request || !results) {
        dressing[key] = entry;
        continue;
      }

      const base = results[0];
      if (base?.status === 'fulfilled') {
        base.value.updateMatrixWorld(true);
        entry.base = base.value;
      } else {
        console.warn(
          '[ocean] dressing asset unavailable: ' + request.lods[0].url,
          base?.reason,
        );
      }

      for (let level = 1; level < results.length; level++) {
        if (request.lods[level]?.level !== level) break;
        const result = results[level];
        if (result.status !== 'fulfilled') break;
        result.value.updateMatrixWorld(true);
        entry.lods.push(result.value);
      }

      dressing[key] = entry;
    }

    return new Props(
      { buoy, barrel },
      dressing,
      options.seed ?? SEEDS.props,
      options.detailScale ?? 1,
      imposterMode,
      imposters,
    );
  }
  /** Current scattered-dressing density, 0..1. */
  get detailScale(): number {
    return this.detail;
  }

  /**
   * Thins every scattered kind to `scale` of the instances that were placed.
   *
   * Moves `InstancedMesh.count` only. The matrices stay written and the bounding
   * spheres stay as computed over the full set, so this is reversible, costs no
   * allocation, and can run on a live scene between frames.
   *
   * Instances are written in seeded-random order precisely so that truncating
   * the count thins the scatter evenly instead of clipping off whichever part of
   * the island happened to be filled last.
   */
  setDetailScale(scale: number): void {
    const clamped = Math.min(1, Math.max(0, scale));
    this.detail = clamped;
    for (const entry of this.thinnable) {
      const count = Math.max(entry.floor, Math.round(entry.capacity * clamped));
      for (const mesh of entry.meshes) mesh.count = Math.min(entry.capacity, count);
      // A LOD kind's counts are owned by the deal, not by this: the meshes above
      // are only level 0, and how many instances belong to it depends on where
      // the camera is. Invalidating is enough — `updateLod` redeals next frame.
      if (entry.lod) entry.lod.dealt = false;
    }
    this.imposters?.setDetailScale(clamped);
  }

  /** Updates the atlas wind phase; call this beside FoliageWind.update. */
  update(dt: number): void {
    this.imposters?.update(dt);
  }

  /** Sets the shared wind direction and strength for vegetation cards. */
  setWind(direction: THREE.Vector2, strength: number): void {
    this.imposters?.setWind(direction, strength);
  }

  /** Aligns the atlas phase with a scene clock after a reset or seek. */
  resetClock(time = 0): void {
    this.imposters?.resetClock(time);
  }

  /**
   * Deals every LOD kind's instances to the level each has earned.
   *
   * Cheap to call every frame and normally does nothing: the deal only runs when
   * the camera has moved far enough for a switch to plausibly have changed, and
   * the whole island is a few thousand distance tests when it does. The
   * alternative — sorting per frame — would cost a megabyte of matrix upload
   * every frame to answer a question whose answer changes about once a second.
   */
  /**
   * LOD meshes whose pipeline has been built, and those still waiting.
   *
   * A `Set` rather than a flag on the mesh so nothing has to be cleaned up when
   * a tier change disposes and rebuilds the field — the meshes go, and the sets
   * go with them.
   */
  private readonly compiled = new Set<THREE.InstancedMesh>();
  private readonly uncompiled = new Set<THREE.InstancedMesh>();

  updateLod(cameraPosition: THREE.Vector3): void {
    if (this.disposed) return;

    for (const entry of this.thinnable) {
      const lod = entry.lod;
      if (!lod) continue;
      if (lod.dealt && lod.dealtAt.distanceToSquared(cameraPosition) < LOD_REFRESH_DISTANCE ** 2) {
        continue;
      }

      const drawn = Math.min(entry.capacity, Math.max(entry.floor, Math.round(entry.capacity * this.detail)));
      const counts = new Array<number>(lod.levels.length).fill(0);

      for (let i = 0; i < drawn; i++) {
        const dx = lod.centres[i * 3] - cameraPosition.x;
        const dy = lod.centres[i * 3 + 1] - cameraPosition.y;
        const dz = lod.centres[i * 3 + 2] - cameraPosition.z;
        const distanceSq = dx * dx + dy * dy + dz * dz;

        let level = lod.levels.length;
        if (distanceSq < lod.cullSq) {
          level = lod.levels.length - 1;
          for (let k = 0; k < lod.switchesSq.length; k++) {
            if (distanceSq < lod.switchesSq[k]) {
              level = k;
              break;
            }
          }
        }
        if (level === lod.levels.length) continue;

        const slot = counts[level]++;
        for (const mesh of lod.levels[level]) {
          (mesh.instanceMatrix.array as Float32Array).set(
            lod.matrices.subarray(i * 16, i * 16 + 16),
            slot * 16,
          );
        }
      }

      for (let level = 0; level < lod.levels.length; level++) {
        for (const mesh of lod.levels[level]) {
          mesh.count = counts[level];
          mesh.instanceMatrix.needsUpdate = true;

          // A level holding no instances is hidden rather than left visible with
          // a count of zero, and that is a *boot time* decision rather than a
          // drawing one — an empty instanced draw costs almost nothing.
          //
          // `compileAsync` walks `traverseVisible` and does not look at `count`,
          // so every level of every kind used to have its pipeline built during
          // the boot prewarm, including the two thirds that cannot put a
          // triangle on screen from where the camera starts. Measured: 177
          // pipelines and ~52 s, against 73 pipelines and ~41 s once the
          // undrawable levels are hidden.
          //
          // Why this is worth doing at the level of visibility rather than by
          // compiling a subset: three.js keys a render object's cache on the
          // InstancedMesh's own uuid (`getDynamicCacheKey`, added by
          // mrdoob/three.js#29066 so two meshes of equal count cannot share
          // matrices), so every InstancedMesh gets its own pipeline whatever its
          // material. Sharing material objects therefore buys nothing — measured
          // directly: aliasing the LOD materials took 176 distinct materials to
          // 79 and left the pipeline count at 177. The count of *meshes compiled*
          // is the only quantity that moves.
          const wanted = counts[level] > 0;
          if (wanted && !this.compiled.has(mesh)) {
            // Never drawn before, so its pipeline does not exist yet. Drawing it
            // now is the inline gameplay compile this project rules out, so it
            // stays hidden until `takeUncompiledLevels` has been honoured.
            this.uncompiled.add(mesh);
            mesh.visible = false;
          } else {
            mesh.visible = wanted;
          }
        }
      }

      lod.dealtAt.copy(cameraPosition);
      lod.dealt = true;
    }
  }

  /**
   * Levels that have become drawable but whose pipelines have never been built.
   *
   * Returned once and then forgotten: the caller owns them from that point and
   * is expected to compile them and call `markCompiled`. Empty on almost every
   * frame — it fills only when the camera crosses an LOD switch onto a level
   * that the boot prewarm skipped, which in practice means flying in to the
   * island for the first time.
   */
  takeUncompiledLevels(): THREE.InstancedMesh[] {
    if (this.uncompiled.size === 0) return EMPTY_MESHES;
    const out = [...this.uncompiled];
    this.uncompiled.clear();
    return out;
  }

  /**
   * Records that these meshes now have pipelines, and lets them draw.
   *
   * `markCompiled()` with no argument claims everything currently visible, which
   * is what the boot path wants: the prewarm compiled exactly the visible set,
   * so exactly that set is now safe to draw.
   */
  markCompiled(meshes?: readonly THREE.InstancedMesh[]): void {
    if (meshes === undefined) {
      for (const entry of this.thinnable) {
        if (!entry.lod) continue;
        for (const level of entry.lod.levels) {
          for (const mesh of level) if (mesh.visible) this.compiled.add(mesh);
        }
      }
      return;
    }
    for (const mesh of meshes) {
      this.compiled.add(mesh);
      mesh.visible = mesh.count > 0;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.imposters?.dispose();
    this.object.removeFromParent();
    this.object.clear();
    this.floaters.length = 0;
    this.thinnable.length = 0;
    // Only the geometries this class *created* (baked instancing copies) are
    // ours to free; the loaded originals belong to the AssetLoader.
    for (const geometry of this.ownedGeometries) geometry.dispose();
    this.ownedGeometries.length = 0;
    for (const source of this.sources) source.clear();
    this.sources.length = 0;
  }

  // ------------------------------------------------------------------ internals

  /**
   * Scatters buoys and barrels on an annulus around the origin. The inner
   * radius keeps them clear of the ship at spawn, the outer radius keeps them
   * inside the wake texture's footprint and inside the shallow plateau, where
   * they read against the turquoise instead of vanishing into deep blue.
   */
  private placeFloaters(buoy: THREE.Group, barrel: THREE.Group, random: () => number): void {
    const normaliseBuoy = normaliseFloater(buoy, 2.4);
    const normaliseBarrel = normaliseFloater(barrel, 1.15);

    for (let i = 0; i < BUOY_COUNT; i++) {
      const angle = ((i + random() * 0.6) / BUOY_COUNT) * Math.PI * 2;
      const radius = 55 + random() * 145;
      const object = normaliseBuoy();
      object.name = `buoy-${i}`;
      object.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
      object.rotation.y = random() * Math.PI * 2;
      this.object.add(object);
      this.floaters.push({ object, radius: 1.1 });
    }

    for (let i = 0; i < BARREL_COUNT; i++) {
      // Barrels cluster: flotsam travels together.
      const cluster = i < BARREL_COUNT / 2 ? 0 : 1;
      const baseAngle = cluster === 0 ? 2.1 : 4.9;
      const angle = baseAngle + (random() - 0.5) * 0.5;
      const radius = (cluster === 0 ? 34 : 72) + random() * 22;
      const object = normaliseBarrel();
      object.name = `barrel-${i}`;
      object.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
      object.rotation.set((random() - 0.5) * 0.5, random() * Math.PI * 2, (random() - 0.5) * 0.5);
      this.object.add(object);
      this.floaters.push({ object, radius: 0.62 });
    }
  }

  /**
   * Scatters a reef across the shallow plateau, so there is something down there
   * to look at.
   *
   * The submerged view was empty water over flat sand: the god rays had nothing
   * to fall across, the caustics had nothing to bend over, and there was no
   * parallax to give the depth any scale. Rocks on the bottom fix all three at
   * once, and they are the one thing that can — a fish shoal would move
   * independently of the swell and read as decoration, while a reef is what
   * makes the water *volume* legible.
   *
   * Reuses the island's rock, which is already loaded and instanced, so this
   * costs one more draw call and no download. Scattered on an annulus that
   * clears the spawn point but stays inside the plateau, where the water is
   * shallow enough that light still reaches the bottom.
   */
  private placeReef(dressing: Dressing, random: () => number): void {
    const kind = this.openKind(dressing.islandRock, REEF_COUNT, 'reef-rocks');
    if (!kind) return;
    // Never culled by its own bounds against the surface: the reef is read
    // through refraction from above as well as directly from below.
    for (const level of kind.levels) for (const mesh of level) mesh.castShadow = false;

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const euler = new THREE.Euler();

    for (let i = 0; i < REEF_COUNT; i++) {
      // Most of the rock gathers into the patches and the rest stays scattered
      // between them — see `REEF_PATCHES`. The split is by index rather than by
      // a per-instance draw so that thinning the kind at a lower detail scale
      // keeps the same ratio instead of eating the patches first.
      const inPatch = i < REEF_COUNT * REEF_PATCH_SHARE;
      const site = inPatch ? this.reefSample(random) : reefScatterSample(random);
      const x = site.x;
      const z = site.z;
      const s = inPatch ? 1.8 + random() * 6 : 1.6 + random() * 7;

      // Sunk into the floor by a fraction of their size, so they read as
      // outcrops rather than as boulders resting on a plane.
      position.set(x, seafloorHeight(x, z) - s * 0.28, z);
      euler.set((random() - 0.5) * 0.7, random() * Math.PI * 2, (random() - 0.5) * 0.7);
      quaternion.setFromEuler(euler);
      scale.set(s, s * (0.5 + random() * 0.7), s);
      matrix.compose(position, quaternion, scale);
      for (const mesh of kind.meshes) mesh.setMatrixAt(i, matrix);
      if (kind.matrices && kind.centres) {
        matrix.toArray(kind.matrices, i * 16);
        kind.centres[i * 3] = position.x;
        kind.centres[i * 3 + 1] = position.y;
        kind.centres[i * 3 + 2] = position.z;
      }
    }
    this.closeKind(kind, this.object, REEF_COUNT);
  }

  /** A point inside one of the reef patches, chosen round-robin then jittered. */
  private reefSample(random: () => number): { x: number; z: number } {
    const patch = this.reefPatches[Math.floor(random() * this.reefPatches.length)];
    if (!patch) return reefScatterSample(random);
    const angle = random() * Math.PI * 2;
    // Square root again, for the same reason as the annulus: without it the
    // patch is a ring rather than a mound.
    const radius = Math.sqrt(random()) * REEF_PATCH_RADIUS;
    return { x: patch.x + Math.cos(angle) * radius, z: patch.z + Math.sin(angle) * radius };
  }

  /**
   * Plants the coral.
   *
   * The one thing the underwater view had none of. Everything alive down there
   * was either a fish or a kelp stipe, and both of those are *sparse* by nature,
   * so a viewer who dived found sand, some rocks and a couple of blades. Coral
   * is what fills the space between the rocks, and it is also the only thing
   * here with strong local colour — which matters more than the geometry does,
   * because ten metres of this water has already taken the contrast out of
   * everything else.
   *
   * Every head is seated on the heightfield and sunk slightly, tilted a little
   * off vertical, and given a spin: coral grows toward the light, so the lean is
   * small and the yaw is free. They cast no shadows — a shadow map that reaches
   * the seabed is not something this scene budgets for, and the caustics already
   * carry the light on the bottom.
   *
   * Density comes from `coralSoft`, which is a set of twenty-four distinct forms
   * in one file: `bake.include` takes a different three of them per kind and
   * stacks them into a clump, so two draws' worth of geometry produce a bed that
   * does not read as one shape repeated. The three photogrammetry colonies are
   * placed sparsely on top of that as the pieces the eye actually lands on.
   */
  private placeReefGarden(dressing: Dressing, random: () => number): void {
    const garden = new THREE.Group();
    garden.name = 'reef-garden';
    this.object.add(garden);

    const kinds: {
      source: DressingEntry;
      count: number;
      height: number;
      name: string;
      include?: (name: string) => boolean;
      lean: number;
      sink: number;
    }[] = [
      // The carpet: three low mound forms stacked into one colony, at 800
      // triangles for the clump. This is where the density comes from, and it
      // is only affordable because these forms are cheap — the same count of
      // the tube corals below would be a million triangles on its own.
      {
        source: dressing.coralSoft,
        count: CORAL_BED_COUNT,
        height: CORAL_BED_HEIGHT,
        name: 'reef-coral-bed',
        include: (n) => /^coral(12|16|18)_/.test(n),
        lean: 0.3,
        sink: 0.16,
      },
      {
        source: dressing.coralSoft,
        count: CORAL_BED_ALT_COUNT,
        height: CORAL_BED_HEIGHT * 0.85,
        name: 'reef-coral-bed-alt',
        include: (n) => /^coral(21|27|28)_/.test(n),
        lean: 0.3,
        sink: 0.16,
      },
      // The upright forms — organ-pipe and barrel corals. These are what stop a
      // reef reading as lumps on a floor: they are the only thing down here
      // except the rocks with a vertical silhouette, and a diver's sense of
      // being *inside* something depends on them.
      {
        source: dressing.coralSoft,
        count: CORAL_HEAD_COUNT,
        height: CORAL_HEAD_HEIGHT,
        name: 'reef-coral-head',
        include: (n) => /^coral(10|11)_/.test(n),
        lean: 0.18,
        sink: 0.1,
      },
      {
        source: dressing.coralSoft,
        count: CORAL_HEAD_ALT_COUNT,
        height: CORAL_HEAD_HEIGHT * 0.85,
        name: 'reef-coral-head-alt',
        include: (n) => /^coral(19|25)_/.test(n),
        lean: 0.18,
        sink: 0.1,
      },
      // Table and plate corals: the horizontal counterpoint, and the shape that
      // gives the fish something to shelter under.
      {
        source: dressing.coralSoft,
        count: CORAL_PLATE_COUNT,
        height: CORAL_PLATE_HEIGHT,
        name: 'reef-coral-plate',
        include: (n) => /^coral(13|14)_/.test(n),
        lean: 0.22,
        sink: 0.1,
      },
      {
        source: dressing.coralFan,
        count: CORAL_FAN_COUNT,
        height: CORAL_FAN_HEIGHT,
        name: 'reef-coral-fan',
        lean: 0.3,
        sink: 0.08,
      },
      {
        source: dressing.coralBirdsnest,
        count: CORAL_BIRDSNEST_COUNT,
        height: CORAL_BIRDSNEST_HEIGHT,
        name: 'reef-coral-birdsnest',
        lean: 0.2,
        sink: 0.1,
      },
      {
        source: dressing.coralBrain,
        count: CORAL_BRAIN_COUNT,
        height: CORAL_BRAIN_HEIGHT,
        name: 'reef-coral-brain',
        lean: 0.16,
        sink: 0.22,
      },
    ];

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const euler = new THREE.Euler();
    const box = new THREE.Box3();

    for (const kind of kinds) {
      const bake: BakeOptions = {
        include: kind.include,
        origin: kind.include ? 'stack' : 'cluster',
      };
      const parts = this.bakeParts(kind.source.base, bake);
      const meshes = this.buildScatter(parts, kind.count, kind.name);
      if (!meshes) continue;

      const levels: THREE.InstancedMesh[][] = [meshes];
      for (let i = 0; i < kind.source.lods.length; i++) {
        const level = this.buildScatter(
          this.bakeParts(kind.source.lods[i], bake),
          kind.count,
          `${kind.name}-lod${i + 1}`,
        );
        if (level) levels.push(level);
      }
      const hasLod = levels.length > 1;
      const matrices = hasLod ? new Float32Array(kind.count * 16) : null;
      const centres = hasLod ? new Float32Array(kind.count * 3) : null;

      // The source scales differ by an order of magnitude between a museum
      // specimen and an authored kit piece, so the instance scale is derived
      // from the model's own height rather than written down per asset.
      box.makeEmpty();
      for (const part of parts) {
        part.geometry.computeBoundingBox();
        if (part.geometry.boundingBox) box.union(part.geometry.boundingBox);
      }
      const modelHeight = Math.max(1e-3, box.max.y - box.min.y);

      for (const level of levels) {
        for (const mesh of level) {
          mesh.castShadow = false;
          mesh.receiveShadow = true;
          garden.add(mesh);
        }
      }

      let placed = 0;
      // The colony currently being filled, and how many heads it still owes.
      let colony = { x: 0, z: 0 };
      let colonyLeft = 0;

      for (let i = 0; i < kind.count; i++) {
        if (colonyLeft <= 0) {
          // A tenth of colonies start out on the open sand, so a patch has
          // outliers leading to it rather than an edge.
          colony = random() < 0.9 ? this.reefSample(random) : reefScatterSample(random);
          colonyLeft =
            CORAL_COLONY_MIN + Math.floor(random() * (CORAL_COLONY_MAX - CORAL_COLONY_MIN + 1));
        }
        colonyLeft -= 1;

        const spread = Math.sqrt(random()) * CORAL_COLONY_RADIUS;
        const bearing = random() * Math.PI * 2;
        const site = {
          x: colony.x + Math.cos(bearing) * spread,
          z: colony.z + Math.sin(bearing) * spread,
        };
        const floor = seafloorHeight(site.x, site.z);
        // Nothing is planted where the plateau has fallen away past the light,
        // and nothing above the waterline: this is a reef, not a rock pool.
        if (floor > -3 || floor < -30) continue;

        const s = (kind.height * (0.6 + random() * 0.9)) / modelHeight;
        position.set(site.x, floor - modelHeight * s * kind.sink - CORAL_BED_SINK, site.z);
        euler.set(
          (random() - 0.5) * kind.lean,
          random() * Math.PI * 2,
          (random() - 0.5) * kind.lean,
        );
        quaternion.setFromEuler(euler);
        scale.set(s, s * (0.85 + random() * 0.35), s);
        matrix.compose(position, quaternion, scale);
        for (const mesh of meshes) mesh.setMatrixAt(placed, matrix);
        if (matrices && centres) {
          matrix.toArray(matrices, placed * 16);
          centres[placed * 3] = position.x;
          centres[placed * 3 + 1] = position.y;
          centres[placed * 3 + 2] = position.z;
        }
        placed += 1;
      }

      this.seal(meshes, placed, hasLod ? { levels, matrices: matrices!, centres: centres! } : undefined);
    }
  }

  /**
   * Boulders scattered from the island's shoulders down to the waterline, seated
   * on the seafloor heightfield so nothing floats or buries itself.
   *
   * This used to *be* the island: five `namaqualand_cliff_01` blocks on the
   * summit and thirty-four boulders round them, back when the terrain under them
   * was a 72 m mound and needed the help. It does not any more. The heightfield
   * carries a 150 m dome with its own crest, bays and headland, and five 90 m
   * scans stacked on top of it read as a cairn somebody built there.
   *
   * The cliff kind is gone entirely rather than moved. `shells.mjs` measures it
   * at a depth-over-length of 0.53 — closed, and not a facade — but it is a
   * single 90 m mass with one composed silhouette, and five copies of one
   * silhouette is a repeat the eye finds immediately whatever you do with the
   * rotations. Crags on the upper slopes are now `islandCrag` in `dressIsland`,
   * built from the closed 1-2 m rocks stretched and clumped, which gives the
   * same job a different shape every time.
   */
  private placeIsland(dressing: Dressing, random: () => number): void {
    const island = new THREE.Group();
    island.name = 'island';
    this.object.add(island);

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const euler = new THREE.Euler();

    /** Writes the composed `matrix` into every level and into the deal arrays. */
    const record = (kind: NonNullable<ReturnType<Props['openKind']>>, i: number): void => {
      for (const mesh of kind.meshes) mesh.setMatrixAt(i, matrix);
      if (kind.matrices && kind.centres) {
        matrix.toArray(kind.matrices, i * 16);
        kind.centres[i * 3] = position.x;
        kind.centres[i * 3 + 1] = position.y;
        kind.centres[i * 3 + 2] = position.z;
      }
    };

    const rockKind = this.openKind(dressing.islandRock, ROCK_COUNT, 'island-rocks');
    if (rockKind) {
      for (let i = 0; i < ROCK_COUNT; i++) {
        const angle = random() * Math.PI * 2;
        // Bias toward the shoreline ring so the island gets a broken edge
        // rather than a clean cone.
        const radius = ISLAND.radius * (0.35 + Math.sqrt(random()) * 0.6);
        const x = ISLAND.x + Math.cos(angle) * radius;
        const z = ISLAND.z + Math.sin(angle) * radius;
        const s = 9 + random() * 16;

        position.set(x, seafloorHeight(x, z) - s * 0.02, z);
        euler.set((random() - 0.5) * 0.3, random() * Math.PI * 2, (random() - 0.5) * 0.3);
        quaternion.setFromEuler(euler);
        scale.set(s, s * (0.7 + random() * 0.7), s);
        matrix.compose(position, quaternion, scale);
        record(rockKind, i);
      }
      this.closeKind(rockKind, island, ROCK_COUNT);
    }
  }

  // ---------------------------------------------------------- island dressing

  /**
   * Dresses the island: a broken rocky coast at the waterline, and planting
   * above it that grades from bare sand through scrub to closed canopy inland.
   *
   * The elevation bands do the art direction, and they are the part of this that
   * survived the island being reshaped. Nothing is placed by radius alone
   * because the shoreline is not a circle and never was going to stay where it
   * was — the coastal kinds ask for ground between roughly -5 m and +4 m and
   * land wherever that happens to be, which is what puts rock exactly where the
   * water meets the island instead of on a ring that only approximately follows
   * it.
   *
   * Species vary by elevation and by exposure, which between them are the whole
   * of the planting scheme:
   *
   *  - **0-2 m** nothing. Bare sand and the swash band, which is what makes the
   *    beach read as a beach rather than as lawn that stops at the water.
   *  - **2-14 m** pachira on the fringe, grass and sorrel, the first ferns.
   *  - **5-100 m** the closed canopy: the orchid tree over most of the flanks,
   *    with the jacaranda's broad crown held to sheltered ground below 40 m and
   *    under a fifth grade — a crown that wide on a ridge would be shredded, and
   *    a scan that big on a slope reads as a mushroom sitting on the hill.
   *  - **7-78 m** the flame trees, in three clumps on the sheltered mid-slopes.
   *  - **10-108 m** the second broadleaf carries the upper flanks to the
   *    treeline, which `Seafloor` puts at 0.72 of `ISLAND.peak`.
   *
   * The ceilings are absolute metres rather than fractions of the peak, which is
   * a choice and not an oversight: what decides where a species stops is
   * exposure and soil depth, and both are properties of the slope rather than of
   * how tall the hill happens to be. They do have to be revisited when the peak
   * moves, and when it went from 72 m to 150 m they were — before that pass the
   * entire canopy sat in a ring round the bottom third of the dome and the
   * island read as a bald hill with a hedge.
   *
   * Density falls off toward the shore for free: the radial sample is uniform in
   * *radius*, so on a disc it concentrates inland, and the planting's minimum
   * height then cuts off the last stretch of beach entirely.
   */
  private dressIsland(dressing: Dressing, random: () => number): void {
    const island = new THREE.Group();
    island.name = 'island-dressing';
    this.object.add(island);

    // Broken rock on the headlands, in clusters, and every piece of it a closed
    // solid.
    //
    // What stood here was six copies of `coastal_cliff_02` and three of
    // `coastal_cliff_04`: two beautiful Poly Haven scans that are *facades*. The
    // measurement is in `scripts/modelkit/shells.mjs` — depth over length of
    // 0.21 and 0.28 — and it means what it says. Each is a 41 m and an 87 m
    // cliff *face*, eight and twenty-four metres deep, authored to be set into a
    // hillside and looked at from the front. From behind, or from the water on
    // the wrong side of the headland, they are a wall with nothing on the other
    // side of it, and on an island a viewer can circle that is most approaches.
    //
    // The replacement is not a bigger facade. It is a different idea of what
    // shoreline rock is: three closed solids at 1-2 m, scaled to 4-12 m and
    // stacked in clumps the way rock actually breaks. Boundary-edge fractions of
    // 0.0%, 0.0% and 2.8% against the facades' 0.8% and 0.9% — all five are
    // closed *surfaces*; only these three are closed *shapes*. They can be spun
    // freely, tipped onto any face and seen from below, which is what lets a
    // clump read as a rockfall rather than as a row.
    //
    // Weighted onto the headland arc, which is where the terrain is steepest and
    // where the fort stands. It is also what the reference frame shows: one
    // rocky point holding the ruin, and sand everywhere the swell does not get
    // to. A wide arc rather than a tight one — this is a bias, not a fence.
    // Crags on the upper slopes, where the reference frame shows broken rock
    // pushing through the canopy. Stretched hard and clumped in threes and
    // fours: one scan repeated twenty-two times is a texture the eye learns, and
    // the same scan at a different aspect and a different attitude every time is
    // an outcrop. Held below the treeline's own start so they read as rock
    // *through* the green rather than as a second bald summit.
    this.scatter(island, dressing.headlandRock, ISLAND_CRAG_COUNT, 'island-crags', random, {
      inner: 0.18,
      outer: 0.86,
      minHeight: 26,
      maxHeight: ISLAND.peak * 0.7,
      minScale: 6,
      maxScale: 15,
      minStretch: 0.6,
      maxStretch: 1.6,
      sink: 1.2,
      slope: 0.4,
      slopeSpan: 14,
      clusters: 6,
      clusterRadius: ISLAND.radius * 0.09,
    });

    this.scatter(island, dressing.headlandRock, HEADLAND_ROCK_COUNT, 'island-headland-rock', random, {
      inner: 0.62,
      outer: 1.36,
      minHeight: 0,
      maxHeight: 26,
      bearing: HEADLAND_BEARING,
      spread: 1.7,
      minScale: 3,
      maxScale: 8,
      // Stretched hard and independently on each axis. One scan repeated thirty
      // times is a texture the eye learns; the same scan at a different aspect
      // every time is an outcrop.
      minStretch: 0.7,
      maxStretch: 1.5,
      sink: 0.5,
      slope: 0.55,
      slopeSpan: 10,
      clusters: 5,
      clusterRadius: ISLAND.radius * 0.1,
    });

    this.scatter(island, dressing.shoreOutcrop, SHORE_OUTCROP_COUNT, 'island-shore-outcrop', random, {
      inner: 0.7,
      outer: 1.4,
      minHeight: -2,
      maxHeight: 14,
      minScale: 2.6,
      maxScale: 7,
      minStretch: 0.7,
      maxStretch: 1.45,
      sink: 0.6,
      slope: 0.6,
      slopeSpan: 8,
      clusters: 6,
      clusterRadius: ISLAND.radius * 0.08,
    });

    // The waterline itself. Straddles mean sea level, so half of these stand in
    // the swash and half are awash — which is the band the reference frame shows
    // as dark broken rock between the sand and the turquoise.
    this.scatter(island, dressing.shoreBoulder, SHORE_BOULDER_COUNT, 'island-shore-boulders', random, {
      inner: 0.78,
      outer: 1.42,
      minHeight: -3.5,
      maxHeight: 6,
      minScale: 1.8,
      maxScale: 5.5,
      minStretch: 0.75,
      maxStretch: 1.4,
      sink: 0.35,
      slope: 0.5,
      slopeSpan: 6,
      clusters: 8,
      clusterRadius: ISLAND.radius * 0.07,
    });

    // Shoreline edges. These two are authored as coast rather than as rock: a
    // wave-cut platform with a lip, so they only make sense lying *on* the
    // waterline contour, which is why their elevation band is the tightest here
    // and why they take the contour facing rather than a random spin.
    this.scatter(island, dressing.coastLineWide, COAST_LINE_WIDE_COUNT, 'island-coast-line-wide', random, {
      inner: 0.7,
      outer: 1.36,
      minHeight: -3,
      maxHeight: 2,
      minScale: 0.7,
      maxScale: 1.15,
      sink: 0.6,
      slope: 0.9,
      slopeSpan: 26,
      facing: 'contour',
    });

    this.scatter(island, dressing.coastLineNarrow, COAST_LINE_NARROW_COUNT, 'island-coast-line-narrow', random, {
      inner: 0.7,
      outer: 1.36,
      minHeight: -3,
      maxHeight: 2,
      minScale: 0.8,
      maxScale: 1.3,
      sink: 0.5,
      slope: 0.9,
      slopeSpan: 20,
      facing: 'contour',
    });

    // Wave-cut platforms. Both coast rock scans are wide and flat with their
    // origin buried inside the mass, so they need a wide terrain sample to tilt
    // against: a 60 m shelf levelled from a 6 m sample drives one edge metres
    // into the ground on a slope this gentle.
    this.scatter(island, dressing.coastRocksWide, COAST_ROCKS_WIDE_COUNT, 'island-coast-rocks-wide', random, {
      inner: 0.66,
      outer: 1.42,
      // Floor of the band is set by the model, not by taste: these shelves are
      // only ~1.3 m proud of their own origin, so ground below about -5 m puts
      // a 40k-triangle scan entirely out of sight under the water.
      minHeight: -5,
      maxHeight: 3,
      minScale: 0.7,
      maxScale: 1.15,
      sink: 0.35,
      slope: 0.85,
      slopeSpan: 26,
    });

    this.scatter(island, dressing.coastRocksTall, COAST_ROCKS_TALL_COUNT, 'island-coast-rocks-tall', random, {
      inner: 0.66,
      outer: 1.42,
      minHeight: -5,
      maxHeight: 4,
      minScale: 0.9,
      maxScale: 1.6,
      sink: 0.4,
      slope: 0.85,
      slopeSpan: 13,
    });

    // Boulder fields above the tideline, in clumps. Loose rock does not arrive
    // one stone at a time; it arrives where something above it gave way.
    this.scatter(island, dressing.landRocks, LAND_ROCKS_COUNT, 'island-land-rocks', random, {
      inner: 0.55,
      outer: 1.36,
      minHeight: 2,
      maxHeight: 16,
      minScale: 1.4,
      maxScale: 2.6,
      sink: 0.25,
      slope: 0.9,
      slopeSpan: 8,
      clusters: 3,
      clusterRadius: ISLAND.radius * 0.09,
    });

    // The biggest crowns, in two groves on sheltered ground. Only four of them:
    // this scan is 45k triangles and 19 m across at unit scale, so it is worth
    // having where it can be the thing that gives the canopy a top and worth
    // nothing at all repeated across a hillside.
    this.scatter(island, dressing.jacaranda, JACARANDA_COUNT, 'island-jacaranda', random, {
      inner: 0.25,
      outer: 1.15,
      minHeight: 6,
      maxHeight: 40,
      maxSlope: 0.22,
      minScale: 0.55,
      maxScale: 0.95,
      minStretch: 0.9,
      maxStretch: 1.1,
      sink: 0.2,
      slope: 0.15,
      slopeSpan: 24,
      lean: 0.04,
      clusters: 2,
      clusterRadius: ISLAND.radius * 0.16,
    });

    // Trees in groves rather than an even scatter. Evenly spaced trees read as
    // an orchard from any distance; clumps read as vegetation, and at 1.4 km the
    // clumping is most of what is left of them. The grove radius is a fraction
    // of the island rather than a fixed 45 m, or the same five groves would
    // cover a quarter of the ground they used to.
    //
    // The canopy is a Hong Kong orchid tree now, not `island_tree_01`. That scan
    // is 5 m of half-bare coastal scrub — a beautiful capture of the wrong
    // plant, and on a tropical island it read as a dead stick, which is exactly
    // what it was asked about. This one is an authored botanical model: 5.2 m of
    // closed broadleaf crown, leaves as alpha-tested cards rather than modelled
    // geometry, so it costs a third of the triangles and fills four times the
    // silhouette.
    //
    // Scale 1.5-2.4 puts it at 8-12 m, which is what a mature Bauhinia is.
    this.scatter(island, dressing.tree, TREE_COUNT, 'island-trees', random, {
      inner: 0.1,
      outer: 1.25,
      minHeight: 5,
      maxHeight: 100,
      maxSlope: 0.4,
      minScale: 1.5,
      maxScale: 2.4,
      minStretch: 0.9,
      maxStretch: 1.2,
      sink: 0.1,
      slope: 0.25,
      slopeSpan: 12,
      lean: 0.06,
      clusters: 5,
      clusterRadius: ISLAND.radius * 0.18,
    });

    // The flame trees. Held to the sheltered mid-slopes and clustered tightly:
    // a poinciana is a valley tree, and the reference shows them in two or three
    // clumps rather than spread through the wood. Scale 1.2-1.9 on a 9 m crown
    // gives 11-17 m across, which is the shape — this is a tree that is half
    // again as wide as it is tall.
    this.scatter(island, dressing.treeFlame, TREE_FLAME_COUNT, 'island-flame-trees', random, {
      inner: 0.2,
      outer: 1.1,
      minHeight: 7,
      maxHeight: 78,
      maxSlope: 0.3,
      minScale: 1.2,
      maxScale: 1.9,
      minStretch: 0.9,
      maxStretch: 1.15,
      sink: 0.15,
      slope: 0.12,
      slopeSpan: 16,
      lean: 0.05,
      clusters: 3,
      clusterRadius: ISLAND.radius * 0.12,
    });

    this.scatter(island, dressing.treeMid, TREE_MID_COUNT, 'island-trees-mid', random, {
      inner: 0.05,
      outer: 1.15,
      minHeight: 10,
      maxHeight: 108,
      minScale: 2.2,
      maxScale: 3.6,
      minStretch: 0.9,
      maxStretch: 1.15,
      sink: 0.08,
      slope: 0.25,
      slopeSpan: 10,
      lean: 0.08,
      clusters: 5,
      clusterRadius: ISLAND.radius * 0.15,
    });

    // The pachira ships as four plants laid out in a row; `_d` is the tallest of
    // them, and taking one variant rather than the row is what keeps this to a
    // single instanced clump instead of four plants marching sideways. Held to
    // the low fringe: the species is a swamp tree, and it is what turns the join
    // between beach and canopy into a gradient.
    this.scatter(island, dressing.pachira, PACHIRA_COUNT, 'island-pachira', random, {
      inner: 0.55,
      outer: 1.3,
      minHeight: 2.5,
      maxHeight: 14,
      minScale: 2.2,
      maxScale: 3.8,
      sink: 0.05,
      slope: 0.2,
      slopeSpan: 7,
      lean: 0.08,
      clusters: 4,
      clusterRadius: ISLAND.radius * 0.09,
      bake: { include: (name) => name.endsWith('_d'), origin: 'cluster' },
    });

    // Understorey. Everything below here only *receives* shadow: it lives under
    // the canopy, where the light is already the canopy's shadow, and drawing a
    // few thousand leaf cards a second time into the depth map buys a contact
    // shadow nobody will ever be close enough to see.
    this.scatter(island, dressing.anthurium, ANTHURIUM_COUNT, 'island-anthurium', random, {
      inner: 0.1,
      outer: 1.22,
      minHeight: 4,
      maxHeight: 84,
      maxSlope: 0.3,
      minScale: 1.8,
      maxScale: 3.2,
      slope: 0.3,
      slopeSpan: 8,
      lean: 0.1,
      clusters: 5,
      clusterRadius: ISLAND.radius * 0.07,
      casts: false,
      bake: { include: (name) => name.endsWith('_a'), origin: 'cluster' },
    });

    this.scatter(island, dressing.calathea, CALATHEA_COUNT, 'island-calathea', random, {
      inner: 0.1,
      outer: 1.22,
      minHeight: 4,
      maxHeight: 76,
      maxSlope: 0.3,
      minScale: 2.2,
      maxScale: 3.8,
      slope: 0.3,
      slopeSpan: 8,
      lean: 0.1,
      clusters: 6,
      clusterRadius: ISLAND.radius * 0.07,
      casts: false,
      bake: { include: (name) => name.endsWith('_a'), origin: 'cluster' },
    });

    this.scatter(island, dressing.fern, FERN_COUNT, 'island-ferns', random, {
      inner: 0.08,
      outer: 1.28,
      minHeight: 3,
      maxHeight: 96,
      minScale: 2.4,
      maxScale: 4.2,
      slope: 0.35,
      slopeSpan: 6,
      lean: 0.12,
      clusters: 7,
      clusterRadius: ISLAND.radius * 0.07,
      casts: false,
      bake: { include: (name) => name.endsWith('_b'), origin: 'cluster' },
    });

    this.scatter(island, dressing.sorrel, SORREL_COUNT, 'island-sorrel', random, {
      inner: 0.08,
      outer: 1.28,
      minHeight: 3,
      maxHeight: 98,
      minScale: 7,
      maxScale: 14,
      slope: 0.4,
      slopeSpan: 5,
      lean: 0.12,
      clusters: 7,
      clusterRadius: ISLAND.radius * 0.06,
      casts: false,
      bake: { include: (name) => name.endsWith('_d'), origin: 'cluster' },
    });

    // --- palms -------------------------------------------------------------
    //
    // These replace the procedural palm `Remains.ts` used to build. That palm
    // was a good piece of engineering and a bad tree: its fronds faked
    // transmission through `emissiveNode`, which is neither shadowed nor
    // tone-mapped with the rest of the scene, so on a clear day the whole grove
    // came out chrome blue against a blue sky.
    //
    // A coconut grows to the high-tide line and no further inland than the first
    // rise, which is what the elevation band is: low enough to stand on the
    // beach, high enough to climb the first slope behind it. The bearing weights
    // them toward the cove, because that is the shoreline anyone arrives at.
    this.scatter(island, dressing.palmCoconut, PALM_COCONUT_COUNT, 'island-palms', random, {
      inner: 0.55,
      outer: 1.32,
      minHeight: 1.2,
      maxHeight: 22,
      maxSlope: 0.34,
      bearing: 0.7,
      spread: 1.6,
      minScale: 1.5,
      maxScale: 2.6,
      minStretch: 0.9,
      maxStretch: 1.25,
      sink: 0.12,
      // Palms grow *up*, not out of the slope, so the tilt toward the ground
      // normal is small — but not zero: a shoreline coconut leans seaward, and
      // a grove of perfectly vertical ones reads as telegraph poles.
      slope: 0.18,
      slopeSpan: 12,
      lean: 0.16,
      clusters: 7,
      clusterRadius: ISLAND.radius * 0.11,
    });

    this.scatter(island, dressing.palmTall, PALM_TALL_COUNT, 'island-palms-tall', random, {
      inner: 0.5,
      outer: 1.3,
      minHeight: 1.6,
      maxHeight: 30,
      maxSlope: 0.34,
      bearing: 0.7,
      spread: 2.1,
      minScale: 1.8,
      maxScale: 3.2,
      minStretch: 0.9,
      maxStretch: 1.3,
      sink: 0.1,
      slope: 0.15,
      slopeSpan: 12,
      lean: 0.14,
      clusters: 6,
      clusterRadius: ISLAND.radius * 0.13,
    });

    // --- ground cover ------------------------------------------------------
    //
    // Grass is the one kind with a slope test, and the one kind whose source
    // files are rows of separate tufts rather than single plants: stacking a
    // few variants on a common origin turns a row into a clump, and merging
    // them makes the clump a single geometry and therefore a single draw.
    //
    // `grass_medium_01` and `_02` carry this, with `IslandMeadow`'s GPU field
    // under them for the density no scatter can pay for.
    //
    // `grass_bermuda_01` used to be here and is gone. It is a *lawn* grass — the
    // whole Poly Haven model is 15 cm tall and 8 cm across, twenty-one separate
    // blades of it — so covering a hillside with it needed a scale of four to
    // eight, and a 15 cm plant blown up to a metre does not read as grass. It
    // reads as a black spiked shrub, which is how it looked on the slope. It was
    // demoted to "fine turf close in" rather than removed, and that was the
    // wrong call twice over: the shader meadow now owns close-in turf entirely,
    // so the demotion left six hundred instances of a broken-looking model
    // drawing underneath a field that had already replaced it.
    this.scatter(island, dressing.grassMeadow, MEADOW_COUNT, 'island-meadow', random, {
      inner: 0.05,
      outer: 1.3,
      minHeight: 2,
      maxHeight: 108,
      maxSlope: 0.2,
      minScale: 1.6,
      maxScale: 2.8,
      slope: 0.6,
      slopeSpan: 5,
      lean: 0.06,
      clusters: 9,
      clusterRadius: ISLAND.radius * 0.09,
      casts: false,
      bake: {
        include: (name) => /_(mid_b|small_b|tall_a|tall_c)_/.test(name),
        origin: 'stack',
      },
    });

    this.scatter(island, dressing.grassTussock, TUSSOCK_COUNT, 'island-tussock', random, {
      inner: 0.05,
      outer: 1.28,
      minHeight: 3,
      maxHeight: 100,
      maxSlope: 0.26,
      minScale: 2,
      maxScale: 3.2,
      slope: 0.5,
      slopeSpan: 5,
      lean: 0.08,
      clusters: 8,
      clusterRadius: ISLAND.radius * 0.07,
      casts: false,
      bake: {
        include: (name) => /_(a|c|e)$/.test(name),
        origin: 'stack',
      },
    });

  }

  // -------------------------------------------------------------- pirate cove

  /**
   * The pirate cove: authored placement, not scatter.
   *
   * Every position here is a fixed offset from the waterline rather than a
   * random draw, because the point of the cluster is the relative arrangement —
   * jetty out into the water, boat beached beside it, camp up on the dry sand,
   * a sword dropped between the camp and the sea. Scattered to the same density
   * it would read as debris.
   *
   * The one thing that is *not* seated on the heightfield is the jetty. A jetty's
   * deck is level and its height is set by the water, not by the bank, so it is
   * placed at a fixed height above sea level and the terrain is left to meet it:
   * the landward posts bury themselves in the beach and the seaward ones stand
   * clear in about two metres of water, which is exactly what the model is for.
   *
   * The lagoon behind the jetty is 5.5 m deep out to 750 m and the entrance is
   * left clear, so a ship can still come in — which is the only reason the cove
   * is on this bearing at all.
   */
  private placeCove(dressing: Dressing): void {
    const cove = new THREE.Group();
    cove.name = 'pirate-cove';
    this.object.add(cove);

    const shore = shorelineRadius(COVE_BEARING);
    const point = new THREE.Vector3();

    const pier = this.buildStaticFrom(dressing.pier, 'cove-jetty-deck');
    let jetty: THREE.Group | null = null;
    if (pier) {
      jetty = new THREE.Group();
      jetty.name = 'cove-jetty';
      const bearing = covePoint(shore + JETTY_OFFSHORE, 0, point);
      jetty.position.set(point.x, JETTY_DECK_Y - PIER_DECK_LOCAL * JETTY_SCALE, point.z);
      // The model runs along its own Z with the intact sections at -Z and the
      // collapsed end at +Z, so pointing +Z out to sea leaves the ruined half
      // and the standing mooring posts in the water.
      jetty.rotation.y = yawAlignZ(Math.cos(bearing), Math.sin(bearing));
      jetty.scale.setScalar(JETTY_SCALE);
      jetty.add(pier);
      cove.add(jetty);
    }

    // Parented to the jetty rather than placed in world space: a lantern on a
    // pier is on the pier, and this way moving the jetty cannot leave it hanging
    // over open water. The scale compensates for the jetty's own.
    const lantern = this.buildStaticFrom(dressing.lantern, 'cove-lantern');
    if (lantern && jetty) {
      lantern.position.set(0.86, PIER_DECK_LOCAL, -1.4);
      lantern.rotation.y = 0.7;
      lantern.scale.setScalar(1.4);
      jetty.add(lantern);
    }

    const pinnace = this.buildStaticFrom(dressing.pinnace, 'cove-pinnace');
    if (pinnace) {
      const bearing = covePoint(shore + PINNACE_OFFSHORE, PINNACE_ALONGSHORE, point);
      pinnace.position.set(point.x, point.y + PINNACE_KEEL_LIFT, point.z);
      // Run aground at a shallow angle rather than bow-on: mostly along the
      // beach, angled just enough inshore to look driven there.
      const bowX = -Math.cos(bearing) * 0.42 - Math.sin(bearing) * 0.91;
      const bowZ = -Math.sin(bearing) * 0.42 + Math.cos(bearing) * 0.91;
      // YXZ so the roll is about the model's own keel line and not about world Z
      // — the heel is what says "aground" rather than "moored", and the trim is
      // the beach gradient (see `PINNACE_TRIM`): a level ship on a shelving
      // beach buries its forefoot and floats its rudder.
      pinnace.rotation.set(PINNACE_TRIM, yawAlignZ(bowX, bowZ), PINNACE_HEEL, 'YXZ');
      pinnace.scale.setScalar(PINNACE_SCALE);
      cove.add(pinnace);
    }

    const barrels = this.buildStaticFrom(dressing.barrels, 'cove-barrels');
    if (barrels) {
      covePoint(shore + CAMP_OFFSHORE, -12, point);
      seatOnGround(barrels, point, 2.4, 0.9, 5);
      barrels.scale.setScalar(1.35);
      cove.add(barrels);
    }

    const crate = this.buildStaticFrom(dressing.coveCrate, 'cove-crate');
    if (crate) {
      covePoint(shore + CAMP_OFFSHORE + 6, -4, point);
      seatOnGround(crate, point, 0.55, 0.8, 4);
      crate.scale.setScalar(1.5);
      cove.add(crate);
    }

    const bucket = this.buildStaticFrom(dressing.bucket, 'cove-bucket');
    if (bucket) {
      covePoint(shore + CAMP_OFFSHORE + 9, -18, point);
      seatOnGround(bucket, point, 1.9, 0.9, 3);
      bucket.scale.setScalar(1.8);
      cove.add(bucket);
    }

    // Three jugs, assembled into one geometry rather than placed as three props.
    // The model is a 21 cm pot: instancing it would cost a draw call to save
    // nothing, and three separate meshes would cost three. Baking them together
    // makes the whole group one draw and one bounding sphere, and lets the one
    // on its side be authored as a tilt rather than as a special case.
    const jugLevels = this.assembleLevels(dressing.jug, JUG_PIECES);
    const jugs = this.buildStatic(jugLevels.base, 'cove-jugs', jugLevels.lods);
    if (jugs) {
      covePoint(shore + CAMP_OFFSHORE - 3, -7, point);
      seatOnGround(jugs, point, 1.2, 0.8, 3);
      cove.add(jugs);
    }

    const estoc = this.buildStaticFrom(dressing.estoc, 'cove-estoc');
    if (estoc) {
      const bearing = covePoint(shore + CAMP_OFFSHORE + 14, 3, point);
      // Turned point-down (the pi) and then tilted back out of vertical, so the
      // lean is a lean rather than an overhang. YXZ puts the tilt in the yawed
      // frame, which is what makes the yaw decide *which way* it leans.
      estoc.rotation.set(Math.PI - ESTOC_LEAN, bearing + 1.9, 0, 'YXZ');
      estoc.position.set(
        point.x,
        point.y + ESTOC_POINT * ESTOC_SCALE * Math.cos(ESTOC_LEAN) - ESTOC_BURY,
        point.z,
      );
      estoc.scale.setScalar(ESTOC_SCALE);
      cove.add(estoc);
    }
  }

  // -------------------------------------------------------------- shore fort

  /**
   * The ruined fort on the headland, and the gun laid in its breach.
   *
   * It is the far half of the cove's story and it is placed to be read with it:
   * 300 m along the coast from the beach and 35 m above it, on the tableland
   * that ends in the bluff, facing the water a boat has to cross to reach the
   * jetty. The camp is what someone did last week; the fort is what was already
   * here.
   *
   * Seated level, not tilted to the ground. The site was picked because the
   * ground varies half a metre across the whole footprint, and `FORT_SINK`
   * swallows that — a fort that followed the terrain would read as subsidence
   * on a hillside rather than as masonry on a plateau.
   */
  private placeFort(dressing: Dressing): void {
    const wallLevels = this.assembleLevels(dressing.fort, FORT_PIECES);
    const walls = this.buildStatic(wallLevels.base, 'shore-fort-walls', wallLevels.lods);
    const cannon = this.buildStaticFrom(dressing.cannon, 'shore-fort-cannon');
    if (!walls && !cannon) return;

    const fort = new THREE.Group();
    fort.name = 'shore-fort';
    this.object.add(fort);

    const radius = shorelineRadius(FORT_BEARING) - ISLAND.radius * FORT_INSET;
    const x = ISLAND.x + Math.cos(FORT_BEARING) * radius;
    const z = ISLAND.z + Math.sin(FORT_BEARING) * radius;

    const aim = shorelineRadius(COVE_BEARING) + FORT_AIM_OFFSHORE;
    const aimX = ISLAND.x + Math.cos(COVE_BEARING) * aim;
    const aimZ = ISLAND.z + Math.sin(COVE_BEARING) * aim;

    fort.position.set(x, seafloorHeight(x, z) - FORT_SINK, z);
    fort.rotation.y = yawAlignZ(aimX - x, aimZ - z);

    if (walls) fort.add(walls);
    if (cannon) {
      cannon.position.set(CANNON_LOCAL.x, FORT_SINK, CANNON_LOCAL.z);
      cannon.rotation.y = CANNON_LOCAL.yaw;
      cannon.scale.setScalar(CANNON_SCALE);
      fort.add(cannon);
    }
  }

  // ---------------------------------------------------------- underwater find

  /**
   * The find: a chest, a crate and a scatter of shells on the reef.
   *
   * Sited on a local high of the plateau at about 16 m of water. That depth is
   * chosen against the shaders rather than the art — `Seafloor` fades its
   * caustics out over 2 m to 48 m of depth, so at 16 m the pattern is still at
   * about three quarters strength and the god rays still reach — and it is
   * inside the reef's annulus, so the chest is found among rocks rather than on
   * open sand.
   */
  private placeFind(dressing: Dressing, random: () => number): void {
    const find = new THREE.Group();
    find.name = 'reef-find';
    this.object.add(find);

    const point = new THREE.Vector3();

    const chest = this.buildStaticFrom(dressing.chest, 'reef-chest');
    if (chest) {
      groundPoint(FIND.x, FIND.z, point);
      seatOnGround(chest, point, 0.92, 1, 4);
      chest.position.y -= 0.14;
      chest.scale.setScalar(2.2);
      find.add(chest);
    }

    const crate = this.buildStaticFrom(dressing.reefCrate, 'reef-crate');
    if (crate) {
      groundPoint(FIND.x + 3.6, FIND.z - 2.4, point);
      seatOnGround(crate, point, 2.6, 1, 4);
      crate.position.y -= 0.1;
      crate.scale.setScalar(2);
      find.add(crate);
    }

    const shells = this.buildScatter(this.bakeParts(dressing.shell.base), SHELL_COUNT, 'reef-shells');
    if (!shells) return;

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const normal = new THREE.Vector3();

    for (let i = 0; i < SHELL_COUNT; i++) {
      // Square-root radius so the shells spread over the area around the chest
      // rather than piling up against it.
      const angle = random() * TAU;
      const radius = 2 + Math.sqrt(random()) * FIND_SPREAD;
      const x = FIND.x + Math.cos(angle) * radius;
      const z = FIND.z + Math.sin(angle) * radius;
      const s = 2.2 + random() * 2;

      position.set(x, seafloorHeight(x, z) - s * 0.01, z);
      groundNormal(x, z, 3, normal);
      quaternion.setFromUnitVectors(UP, normal);
      quaternion.multiply(spinAbout(UP, random() * TAU));
      scale.set(s, s, s);
      for (const mesh of shells) mesh.setMatrixAt(i, matrix.compose(position, quaternion, scale));
    }
    this.seal(shells, SHELL_COUNT);
    for (const mesh of shells) {
      // A 14 cm shell 16 m under water casts nothing anyone can resolve.
      mesh.castShadow = false;
      find.add(mesh);
    }
  }

  // ---------------------------------------------------------------- placement

  /**
   * Scatters one model kind across the island according to `spec`.
   *
   * A sample that misses the elevation or slope band is rejected and redrawn,
   * up to `PLACEMENT_ATTEMPTS` times, and an instance that never finds ground is
   * simply not placed — so the count in the constants above is a ceiling, and
   * the instance count is fixed afterwards to what actually landed. Leaving the
   * unplaced instances at their identity matrix would stack the whole kind on
   * the world origin, in the middle of the play area.
   */
  private scatter(
    parent: THREE.Object3D,
    source: DressingEntry,
    count: number,
    name: string,
    random: () => number,
    spec: ScatterSpec,
  ): void {
    const imposterKind = IMPOSTER_KINDS[name];
    const useImposter =
      this.imposters !== null &&
      imposterKind !== undefined &&
      this.imposterMode !== 'off';
    const buildMeshes = this.imposterMode !== 'always' || !useImposter;
    const meshes = buildMeshes
      ? this.buildScatter(this.bakeParts(source.base, spec.bake), count, name)
      : null;
    if (!meshes && !useImposter) return;

    // The same placement loop feeds both representations. In Hybrid, mesh
    // levels cover the near band and the atlas covers the far band; in Always
    // the source groups are absent and only the atlas receives these transforms.
    const lodMeshes = meshes
      ? source.lods.map((group, index) =>
          this.buildScatter(
            this.bakeParts(group, spec.bake),
            count,
            name + '-lod' + (index + 1),
          ),
        )
      : [];
    const levels: THREE.InstancedMesh[][] = meshes ? [meshes] : [];
    for (const level of lodMeshes) if (level) levels.push(level);
    if (meshes) {
      for (const level of lodMeshes) {
        if (level) this.shareLodMaterials(meshes, level);
      }
    }

    const needsDeal =
      levels.length > 1 ||
      (meshes !== null && useImposter && this.imposterMode === 'hybrid');
    const matrices = needsDeal ? new Float32Array(count * 16) : null;
    const centres = needsDeal ? new Float32Array(count * 3) : null;

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const normal = new THREE.Vector3();
    const point = new THREE.Vector3();

    const clusterCount = spec.clusters ?? 0;
    const clusters: number[] = [];
    for (let i = 0; i < clusterCount; i++) {
      if (pickIslandPoint(random, spec, point)) clusters.push(point.x, point.z);
    }
    const clusterRadius = spec.clusterRadius ?? 30;

    const sink = spec.sink ?? 0;
    const slope = spec.slope ?? 0;
    const slopeSpan = spec.slopeSpan ?? 6;
    const lean = spec.lean ?? 0;
    const minStretch = spec.minStretch ?? 1;
    const maxStretch = spec.maxStretch ?? minStretch;
    const contour = spec.facing === 'contour';

    let placed = 0;
    for (let i = 0; i < count; i++) {
      const found =
        clusters.length > 0
          ? pickClusterPoint(
              random,
              spec,
              clusters,
              (i % (clusters.length / 2)) * 2,
              clusterRadius,
              point,
            )
          : pickIslandPoint(random, spec, point);
      if (!found) continue;

      const s = spec.minScale + random() * (spec.maxScale - spec.minScale);
      position.set(point.x, point.y - sink * s, point.z);
      if (slope > 0 || contour) groundNormal(point.x, point.z, slopeSpan, normal);

      let yaw = random() * TAU;
      if (contour) {
        const derived = contourYaw(normal);
        if (derived !== null) {
          yaw =
            derived +
            (random() - 0.5) * CONTOUR_JITTER +
            (random() < 0.5 ? 0 : Math.PI);
        }
      }

      if (slope > 0) {
        if (slope < 1) normal.lerp(UP, 1 - slope).normalize();
        quaternion.setFromUnitVectors(UP, normal);
        quaternion.multiply(spinAbout(UP, yaw));
      } else {
        quaternion.copy(spinAbout(UP, yaw));
      }
      if (lean > 0) quaternion.multiply(randomLean(random, lean));

      const stretch = minStretch + random() * (maxStretch - minStretch);
      scale.set(s, s * stretch, s);
      matrix.compose(position, quaternion, scale);

      if (meshes) {
        for (const mesh of meshes) mesh.setMatrixAt(placed, matrix);
      }
      if (matrices && centres) {
        matrix.toArray(matrices, placed * 16);
        centres[placed * 3] = position.x;
        centres[placed * 3 + 1] = position.y;
        centres[placed * 3 + 2] = position.z;
      }
      if (useImposter) {
        const near =
          this.imposterMode === 'hybrid' && meshes
            ? imposterKind.minDistance
            : 0;
        const far =
          this.imposterMode === 'hybrid' && meshes
            ? imposterKind.maxDistance
            : IMPOSTER_NO_CULL_DISTANCE;
        this.imposters!.addPlacement({
          species: imposterKind.species,
          position: [position.x, position.y, position.z],
          scale: s,
          stretch,
          yaw,
          minDistance: near,
          maxDistance: far,
          detailSlot: placed,
          detailCapacity: count,
        });
      }
      placed++;
    }

    if (placed === 0) {
      for (const level of lodMeshes) {
        if (level) for (const mesh of level) mesh.visible = false;
      }
      return;
    }

    if (meshes) {
      const lod =
        matrices && centres && levels.length > 0
          ? { levels, matrices, centres }
          : undefined;
      const cullDistance =
        useImposter && this.imposterMode === 'hybrid'
          ? imposterKind.meshCullDistance
          : undefined;
      this.seal(meshes, placed, lod, cullDistance);

      for (const mesh of meshes) {
        mesh.castShadow = spec.casts ?? true;
        parent.add(mesh);
      }
      for (let i = 1; i < levels.length; i++) {
        for (const mesh of levels[i]) {
          mesh.castShadow = false;
          parent.add(mesh);
        }
      }
    }
  }
  /**
   * Fixes an instanced kind's count at what was actually placed and takes its
   * bounds.
   *
   * Order matters: `InstancedMesh.computeBoundingSphere` only walks the first
   * `count` instances, so the sphere is taken over the full placed set *before*
   * the detail scale is allowed to trim the count. A sphere that covers
   * instances we are not currently drawing is conservative, and a conservative
   * sphere can never cull something that is on screen.
   */
  private seal(
    meshes: THREE.InstancedMesh[],
    placed: number,
    lod?: { levels: THREE.InstancedMesh[][]; matrices: Float32Array; centres: Float32Array },
    cullDistance?: number,
  ): void {
    for (const mesh of meshes) {
      mesh.count = placed;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
    }

    if (lod) {
      // Every level is given the full population's bounds. A level's own count
      // is whatever the deal last gave it, and `computeBoundingSphere` only
      // walks that prefix — so a sphere taken now, while a level might hold two
      // instances, would cull the kind the moment the camera moved. The sphere
      // has to describe where the instances *can* be, not where they are.
      const sphere = meshes[0].boundingSphere?.clone() ?? null;
      for (let i = 1; i < lod.levels.length; i++) {
        for (const mesh of lod.levels[i]) {
          mesh.count = 0;
          mesh.instanceMatrix.needsUpdate = true;
          if (sphere) mesh.boundingSphere = sphere.clone();
        }
      }
    }

    this.registerThinnable(meshes, placed, lod, cullDistance);
  }

  private registerThinnable(
    meshes: THREE.InstancedMesh[],
    capacity: number,
    lod?: { levels: THREE.InstancedMesh[][]; matrices: Float32Array; centres: Float32Array },
    cullDistance?: number,
  ): void {
    this.thinnable.push({
      meshes,
      capacity,
      floor: Math.max(1, Math.ceil(capacity * DETAIL_FLOOR)),
      lod: lod
        ? {
            levels: lod.levels,
            switchesSq: LOD_SWITCH_METRES.slice(0, lod.levels.length - 1).map((d) => d * d),
            cullSq: cullDistance === undefined ? Number.POSITIVE_INFINITY : cullDistance * cullDistance,
            matrices: lod.matrices,
            centres: lod.centres,
            dealtAt: new THREE.Vector3(Number.POSITIVE_INFINITY, 0, 0),
            dealt: false,
          }
        : undefined,
    });
  }

  // ------------------------------------------------------------------ baking

  /**
   * Bakes a loaded asset down to one geometry per material.
   *
   * Two jobs. First, the glTF node transforms are baked into geometry copies so
   * that an instance matrix is pure placement — otherwise every instance would
   * have to carry the asset's own arbitrary rotation. Second, sub-meshes that
   * share a material are merged, which is what keeps a nineteen-part barrel pile
   * or a twenty-one-blade grass patch down to one draw call per material rather
   * than one per part.
   *
   * Grouping by the three.js `Material` instance rather than by name is load
   * bearing: `GLTFLoader` already forks a material when some primitives carry
   * vertex colours and others do not, so grouping this way separates exactly the
   * meshes whose attribute sets would have made the merge fail.
   */
  private bakeParts(source: THREE.Group | null, options: BakeOptions = {}): BakedPart[] {
    if (!source) return [];
    const origin = options.origin ?? 'asset';

    const collected: THREE.BufferGeometry[] = [];
    const materials: THREE.Material[] = [];
    const collect = (filtered: boolean): void => {
      source.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (!mesh.isMesh) return;
        if (filtered && options.include && !options.include(mesh.name)) return;

        const geometry = mesh.geometry.clone();
        dequantiseGeometry(geometry);
        geometry.applyMatrix4(mesh.matrixWorld);
        // Morph targets cannot survive what this class does with the geometry —
        // it is baked into an `InstancedMesh` that has no morph influences to
        // drive them — and leaving them attached breaks the merge below:
        // `mergeGeometries` requires `morphTargetsRelative` to agree across
        // every input, and `cannon_01` ships exactly one primitive out of ten
        // with a leftover morph from its rig. That single flag was enough to
        // send the whole model down the "ship the parts separately" path, at
        // ten draws instead of one.
        geometry.morphAttributes = {};
        geometry.morphTargetsRelative = false;
        collected.push(geometry);
        materials.push(Array.isArray(mesh.material) ? mesh.material[0] : mesh.material);
      });
    };

    collect(true);
    // A renamed or re-exported asset should degrade to "the whole model, which
    // may look odd" rather than to nothing at all.
    if (collected.length === 0) collect(false);
    if (collected.length === 0) return [];

    if (origin !== 'asset') {
      const box = new THREE.Box3();
      const bounds = new THREE.Box3();
      const shift = new THREE.Vector3();
      // 'cluster' recentres the selection once, preserving how the parts sit
      // relative to each other; 'stack' recentres each part on its own footprint,
      // which is what collapses a row of variants into a single clump.
      if (origin === 'cluster') {
        box.makeEmpty();
        for (const geometry of collected) {
          geometry.computeBoundingBox();
          if (geometry.boundingBox) box.union(geometry.boundingBox);
        }
        recentreShift(box, shift);
        for (const geometry of collected) geometry.translate(shift.x, shift.y, shift.z);
      } else {
        for (const geometry of collected) {
          geometry.computeBoundingBox();
          if (!geometry.boundingBox) continue;
          bounds.copy(geometry.boundingBox);
          recentreShift(bounds, shift);
          geometry.translate(shift.x, shift.y, shift.z);
        }
      }
    }

    const byMaterial = new Map<THREE.Material, THREE.BufferGeometry[]>();
    for (let i = 0; i < collected.length; i++) {
      const material = materials[i];
      const bucket = byMaterial.get(material);
      if (bucket) bucket.push(collected[i]);
      else byMaterial.set(material, [collected[i]]);
    }

    const parts: BakedPart[] = [];
    for (const [material, geometries] of byMaterial) {
      const merged = geometries.length === 1 ? geometries[0] : mergeGeometries(geometries);
      if (merged === null) {
        // `mergeGeometries` refuses mismatched attribute sets and says so on the
        // console. Ship the parts separately rather than dropping the model.
        for (const geometry of geometries) parts.push(this.adopt(geometry, material));
        continue;
      }
      if (merged !== geometries[0]) {
        // Never uploaded, so this is bookkeeping rather than a GPU free.
        for (const geometry of geometries) geometry.dispose();
      }
      parts.push(this.adopt(merged, material));
    }
    return parts;
  }

  /**
   * Bakes an authored arrangement of a kit model's nodes down to one geometry
   * per material.
   *
   * `modular_fort_01` is not a fort. It is twenty-two wall, tower and walkway
   * pieces laid out side by side in the source file for someone to build a fort
   * out of, and `bakeParts` on its own would faithfully reproduce the shop
   * display. This poses the chosen pieces into a temporary group and hands *that*
   * to `bakeParts`, which is why the result is a set piece for the price of one
   * draw per material rather than one per wall.
   *
   * Every piece is re-origined on its own footprint before it is posed, because
   * the offset a piece carries in the kit layout means nothing here; the
   * authored position is then the only position it has.
   */
  private assemble(source: THREE.Group | null, pieces: readonly KitPiece[]): BakedPart[] {
    if (!source) return [];

    const assembly = new THREE.Group();
    const box = new THREE.Box3();
    const shift = new THREE.Vector3();

    for (const piece of pieces) {
      const node = piece.node === undefined ? source : source.getObjectByName(piece.node);
      // A renamed node costs that piece. A ruin is missing pieces by definition,
      // so this degrades to a slightly more ruined fort rather than to nothing.
      if (!node) continue;

      const copy = node.clone(true);
      // Replace the copy's local transform with the original's *world* one:
      // `clone` keeps only the local transform, so every ancestor's contribution
      // would otherwise be silently dropped.
      copy.position.set(0, 0, 0);
      copy.quaternion.identity();
      copy.scale.set(1, 1, 1);
      copy.applyMatrix4(node.matrixWorld);

      const socket = new THREE.Group();
      socket.add(copy);
      socket.updateMatrixWorld(true);
      box.setFromObject(copy, true);
      recentreShift(box, shift);
      copy.position.add(shift);

      socket.position.set(piece.x, piece.y ?? 0, piece.z);
      // YXZ so the tilt happens in the yawed frame — a wall leans out of its own
      // face, not out of the assembly's.
      socket.rotation.set(piece.tilt ?? 0, piece.yaw ?? 0, 0, 'YXZ');
      if (piece.scale !== undefined) socket.scale.setScalar(piece.scale);
      assembly.add(socket);
    }

    assembly.updateMatrixWorld(true);
    return this.bakeParts(assembly);
  }

  /** Takes ownership of a geometry this class created. */
  private adopt(geometry: THREE.BufferGeometry, material: THREE.Material): BakedPart {
    geometry.computeBoundingSphere();
    this.ownedGeometries.push(geometry);
    return { geometry, material };
  }

  /**
   * One `InstancedMesh` per material, sized for `capacity` instances. All of
   * them are driven by the same matrices, so a multi-material kind still places
   * as a single object.
   */
  private buildScatter(
    parts: BakedPart[],
    capacity: number,
    name: string,
  ): THREE.InstancedMesh[] | null {
    if (parts.length === 0 || capacity <= 0) return null;
    return parts.map((part, index) => {
      const mesh = new THREE.InstancedMesh(part.geometry, part.material, capacity);
      mesh.name = parts.length === 1 ? name : `${name}-${index}`;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
      return mesh;
    });
  }

  /** Builds a static prop and all manifest-provided LOD levels. */
  private buildStaticFrom(
    entry: DressingEntry,
    name: string,
    options: BakeOptions = {},
  ): THREE.Object3D | null {
    const base = this.bakeParts(entry.base, options);
    const lods = entry.lods.map((group) => this.bakeParts(group, options));
    return this.buildStatic(base, name, lods);
  }

  /** Bakes the same authored kit arrangement for each source LOD level. */
  private assembleLevels(
    entry: DressingEntry,
    pieces: readonly KitPiece[],
  ): { base: BakedPart[]; lods: BakedPart[][] } {
    return {
      base: this.assemble(entry.base, pieces),
      lods: entry.lods.map((group) => this.assemble(group, pieces)),
    };
  }

  private buildStatic(
    parts: BakedPart[],
    name: string,
    lodParts: BakedPart[][] = [],
  ): THREE.Object3D | null {
    if (parts.length === 0) return null;

    const hasLod = lodParts.some((level) => level.length > 0);
    if (!hasLod) return this.buildStaticGroup(parts, name, true);

    const lod = new THREE.LOD();
    lod.name = name;
    const base = this.buildStaticGroup(parts, name + '-lod0', true);
    if (!base) return null;
    lod.addLevel(base, 0);

    for (let i = 0; i < lodParts.length; i++) {
      const level = this.buildStaticGroup(
        lodParts[i],
        name + '-lod' + (i + 1),
        false,
      );
      if (!level) continue;
      const distance = LOD_SWITCH_METRES[Math.min(i, LOD_SWITCH_METRES.length - 1)];
      lod.addLevel(level, distance);
      this.shareLodPartMaterials(parts, lodParts[i]);
    }
    return lod;
  }

  private buildStaticGroup(
    parts: BakedPart[],
    name: string,
    casts: boolean,
  ): THREE.Group | null {
    if (parts.length === 0) return null;
    const group = new THREE.Group();
    group.name = name;
    for (let i = 0; i < parts.length; i++) {
      const mesh = new THREE.Mesh(parts[i].geometry, parts[i].material);
      mesh.name = name + '-' + i;
      mesh.castShadow = casts;
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
      group.add(mesh);
    }
    return group;
  }

  private shareLodPartMaterials(
    baseParts: BakedPart[],
    lodParts: BakedPart[],
  ): void {
    for (let i = 0; i < lodParts.length; i++) {
      const target = lodParts[i].material;
      const source =
        (target.name &&
          baseParts.find((part) => part.material.name === target.name)?.material) ??
        baseParts[i]?.material;
      if (source) this.shareMaterialMaps(source, target);
    }
  }

  private shareLodMaterials(
    baseMeshes: THREE.InstancedMesh[],
    lodMeshes: THREE.InstancedMesh[],
  ): void {
    for (let i = 0; i < lodMeshes.length; i++) {
      const target = lodMeshes[i].material;
      const source =
        (Array.isArray(target)
          ? undefined
          : target.name &&
            baseMeshes.find((mesh) => {
              const material = mesh.material;
              return !Array.isArray(material) && material.name === target.name;
            })?.material) ??
        baseMeshes[i]?.material;
      if (source && !Array.isArray(source) && !Array.isArray(target)) {
        this.shareMaterialMaps(source, target);
      }
    }
  }

  private shareMaterialMaps(source: THREE.Material, target: THREE.Material): void {
    const sourceRecord = source as unknown as Record<string, unknown>;
    const targetRecord = target as unknown as Record<string, unknown>;
    for (const slot of LOD_TEXTURE_SLOTS) {
      const value = sourceRecord[slot];
      if (
        value &&
        typeof value === 'object' &&
        (value as { isTexture?: boolean }).isTexture
      ) {
        targetRecord[slot] = value;
      }
    }
    target.needsUpdate = true;
  }
  /**
   * Collapses a loaded single-mesh asset into an `InstancedMesh`.
   *
   * The glTF node transform is baked into a copy of the geometry so the
   * instance matrices are pure placement — otherwise every instance would need
   * to carry the asset's own arbitrary rotation.
   */
  /**
   * A LOD-capable instanced kind for the placements that are not `scatter`.
   *
   * `scatter` builds its own levels inline because it also owns the sampling.
   * The island skirt, the crags and the reef place by hand — each has its own
   * idea of where a rock belongs — but they want exactly the same LOD machinery,
   * and open/close is what lets them have it without a second copy of the
   * dealing code. The caller fills `matrices` and `centres` as it places;
   * `closeKind` seals and registers.
   */
  private openKind(
    entry: DressingEntry,
    count: number,
    name: string,
  ): {
    meshes: THREE.InstancedMesh[];
    levels: THREE.InstancedMesh[][];
    matrices: Float32Array | null;
    centres: Float32Array | null;
  } | null {
    const meshes = this.buildScatter(this.bakeParts(entry.base), count, name);
    if (!meshes) return null;

    const levels: THREE.InstancedMesh[][] = [meshes];
    entry.lods.forEach((group, index) => {
      const level = this.buildScatter(this.bakeParts(group), count, name + '-lod' + (index + 1));
      if (level) {
        this.shareLodMaterials(meshes, level);
        levels.push(level);
      }
    });

    const hasLod = levels.length > 1;
    return {
      meshes,
      levels,
      matrices: hasLod ? new Float32Array(count * 16) : null,
      centres: hasLod ? new Float32Array(count * 3) : null,
    };
  }

  /** Seals a kind opened by `openKind` and registers it for thinning and LOD. */
  private closeKind(
    kind: NonNullable<ReturnType<Props['openKind']>>,
    parent: THREE.Object3D,
    placed: number,
  ): void {
    const lod =
      kind.matrices && kind.centres && kind.levels.length > 1
        ? { levels: kind.levels, matrices: kind.matrices, centres: kind.centres }
        : undefined;
    // `seal` registers the kind itself — see its tail. Calling `registerThinnable`
    // here as well put the reef and the island rocks in `this.thinnable` twice,
    // so every detail change processed them twice and every LOD re-deal past the
    // 25 m refresh threshold rebuilt and re-uploaded their instance matrices
    // twice. Invisible in a frame and pure waste on the camera moves that are
    // already the most expensive thing this class does.
    this.seal(kind.meshes, placed, lod);
    for (const level of kind.levels) for (const mesh of level) parent.add(mesh);
  }
}

interface LoadedSources {
  buoy: THREE.Group;
  barrel: THREE.Group;
}

// ------------------------------------------------------------------- helpers

const spinQuaternion = new THREE.Quaternion();
const leanQuaternion = new THREE.Quaternion();
const leanEuler = new THREE.Euler();
const seatNormal = new THREE.Vector3();

interface DressingManifestLod {
  level: number;
  ratio: number;
  url: string;
}

interface DressingManifestModel {
  lods: DressingManifestLod[];
}

interface DressingManifest {
  version: number;
  basePath: string;
  models: Record<string, DressingManifestModel>;
}

async function loadDressingManifest(): Promise<DressingManifest> {
  // Deadlined: the boot awaits this, and a manifest request that never
  // settles is a boot that never finishes. See fetchWithDeadline.
  const response = await fetchWithDeadline('/models/dressing/dressing-manifest.json');
  if (!response.ok) {
    throw new Error(
      'Dressing manifest request failed: ' +
        response.status +
        ' /models/dressing/dressing-manifest.json',
    );
  }
  const manifest = (await response.json()) as DressingManifest;
  if (manifest.version !== 1 || !manifest.models) {
    throw new Error('Unsupported dressing manifest');
  }
  return manifest;
}

function modelSlug(url: string): string {
  const filename = url.slice(url.lastIndexOf('/') + 1);
  return filename.replace(/\.glb$/i, '');
}

function resolveImposterMode(options: PropsOptions): PropsImposterMode {
  if (options.imposterMode) return options.imposterMode;
  if (options.qualityTier === 'low' || options.qualityTier === 'medium') {
    return 'always';
  }
  if (options.qualityTier) return 'hybrid';
  return (options.detailScale ?? 1) <= 0.5 ? 'always' : 'hybrid';
}
/** A point on the open plateau, even in area over the reef annulus. */
function reefScatterSample(random: () => number): { x: number; z: number } {
  const angle = random() * Math.PI * 2;
  // Square-root radius keeps the scatter even in *area* rather than
  // clustering everything at the inner edge.
  const radius = REEF_INNER + Math.sqrt(random()) * (REEF_OUTER - REEF_INNER);
  return { x: Math.cos(angle) * radius, z: Math.sin(angle) * radius };
}

/** Yaw that turns a model's +Z axis toward the world direction (dx, dz). */
function yawAlignZ(dx: number, dz: number): number {
  return Math.atan2(dx, dz);
}

/**
 * A rotation about `axis`, in shared scratch.
 *
 * The result is only valid until the next call. That is safe here and only here:
 * every caller is construction-time, single-threaded, and consumes the value
 * before asking for another.
 */
function spinAbout(axis: THREE.Vector3, angle: number): THREE.Quaternion {
  return spinQuaternion.setFromAxisAngle(axis, angle);
}

function randomLean(random: () => number, amount: number): THREE.Quaternion {
  leanEuler.set((random() - 0.5) * amount, 0, (random() - 0.5) * amount);
  return leanQuaternion.setFromEuler(leanEuler);
}

/**
 * Surface normal of the seafloor heightfield at (x, z), from a central
 * difference `span` metres wide.
 *
 * The width is a parameter because it decides what "the ground" means for the
 * thing being seated: a shell wants the centimetre it sits on, a sixty-metre
 * rock shelf wants the average across its whole footprint, and using one span
 * for both drives one end of the shelf underground. It is also what decides how
 * far `contourYaw` can see, which is the same argument for the same reason.
 */
function groundNormal(x: number, z: number, span: number, out: THREE.Vector3): THREE.Vector3 {
  const dx = (seafloorHeight(x + span, z) - seafloorHeight(x - span, z)) / (2 * span);
  const dz = (seafloorHeight(x, z + span) - seafloorHeight(x, z - span)) / (2 * span);
  return out.set(-dx, 1, -dz).normalize();
}

/**
 * Yaw that lays a model's long (+X) axis along the terrain's contour, or null
 * where the ground is too level for a contour to mean anything.
 *
 * `normal` is a `groundNormal` sample, whose horizontal part is the negated
 * gradient and therefore points *downhill*. Turning the model's +Z down it puts
 * +X across it, along the contour — so the slab lies the way the ground lies and
 * faces the way the ground faces, in a bay, on a headland or along a spit,
 * because the only thing it reads is the shape of the terrain.
 *
 * What this replaced computed the tangent to a *circle* about the island centre.
 * That was defensible while the island was a radially symmetric dome and wrong
 * the moment `Seafloor` stopped being one: on a coast that now runs from 362 m
 * at the head of the bay to 752 m at the tip of the spit, six cliff slabs all
 * turned to the same imaginary circle stood in a line facing the same way and
 * rendered as a straight grey wall across the back of the island. A screenshot
 * of the far side is what caught it — from the play area they were behind the
 * summit, so nothing on the approach ever showed the error.
 */
function contourYaw(normal: THREE.Vector3): number | null {
  if (Math.hypot(normal.x, normal.z) < CONTOUR_MIN_GRADE) return null;
  return yawAlignZ(normal.x, normal.z);
}

/** Fills `out` with (x, ground height, z). */
function groundPoint(x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
  return out.set(x, seafloorHeight(x, z), z);
}

/**
 * Radius at which the heightfield crosses sea level on `bearing`, in metres from
 * the island centre.
 *
 * Bisection rather than arithmetic, because the shoreline is a sum of harmonics,
 * sector masks and a noise field, and `Seafloor` is entitled to change all three
 * without telling anyone. Asking the height function where the water is cannot
 * go stale; `ISLAND.radius` is only the *mean* shore radius and reading it as a
 * coastline is what put the jetty 350 m inland.
 *
 * Assumes a single crossing inside the bracket. That holds on every bearing this
 * is called for; it would not hold on the spit's, which is a bar with water
 * behind it and therefore two crossings.
 */
function shorelineRadius(bearing: number): number {
  const dx = Math.cos(bearing);
  const dz = Math.sin(bearing);
  let inner = ISLAND.radius * SHORE_SEARCH_IN;
  let outer = ISLAND.radius * SHORE_SEARCH_OUT;
  for (let i = 0; i < SHORE_SEARCH_STEPS; i++) {
    const mid = (inner + outer) * 0.5;
    if (seafloorHeight(ISLAND.x + dx * mid, ISLAND.z + dz * mid) > 0) inner = mid;
    else outer = mid;
  }
  return (inner + outer) * 0.5;
}

/**
 * A point on the cove's shore, given a radius from the island centre and a
 * distance along the shore from the cove's bearing. Returns the bearing of the
 * point itself, which is what the props are turned against.
 *
 * Callers pass `shorelineRadius(COVE_BEARING) + offset` rather than a radius,
 * which is the only reason anything in the cove is still where it should be.
 */
function covePoint(radius: number, alongShore: number, out: THREE.Vector3): number {
  const bearing = COVE_BEARING + alongShore / radius;
  groundPoint(ISLAND.x + Math.cos(bearing) * radius, ISLAND.z + Math.sin(bearing) * radius, out);
  return bearing;
}

/** Seats a one-off prop on the ground at `point`, tilted toward the local slope. */
function seatOnGround(
  object: THREE.Object3D,
  point: THREE.Vector3,
  yaw: number,
  slope: number,
  span: number,
): void {
  object.position.copy(point);
  groundNormal(point.x, point.z, span, seatNormal);
  if (slope < 1) seatNormal.lerp(UP, 1 - slope).normalize();
  object.quaternion.setFromUnitVectors(UP, seatNormal);
  object.quaternion.multiply(spinAbout(UP, yaw));
}

/** Translation that moves a box to sit centred on the origin with its base at y = 0. */
function recentreShift(box: THREE.Box3, out: THREE.Vector3): THREE.Vector3 {
  return out.set(-(box.min.x + box.max.x) * 0.5, -box.min.y, -(box.min.z + box.max.z) * 0.5);
}

/** True if a candidate point clears the spec's elevation and slope bands. */
function acceptPoint(x: number, z: number, spec: ScatterSpec, out: THREE.Vector3): boolean {
  const y = seafloorHeight(x, z);
  if (y < spec.minHeight || y > spec.maxHeight) return false;
  if (spec.maxSlope !== undefined) {
    groundNormal(x, z, spec.slopeSpan ?? 6, seatNormal);
    if (Math.hypot(seatNormal.x, seatNormal.z) / seatNormal.y > spec.maxSlope) return false;
  }
  out.set(x, y, z);
  return true;
}

/**
 * Draws a point on the island inside the spec's bands, or returns false.
 *
 * The radius is uniform rather than square-rooted, which on a disc means the
 * density falls off with distance from the centre — the "thins out toward the
 * shore" the dressing wants, for no extra work.
 */
function pickIslandPoint(
  random: () => number,
  spec: ScatterSpec,
  out: THREE.Vector3,
): boolean {
  const spread = spec.spread ?? Math.PI;
  const base = spec.bearing ?? 0;
  for (let attempt = 0; attempt < PLACEMENT_ATTEMPTS; attempt++) {
    const angle = base + (spec.bearing === undefined ? random() * TAU : (random() * 2 - 1) * spread);
    const radius = ISLAND.radius * (spec.inner + random() * (spec.outer - spec.inner));
    if (acceptPoint(ISLAND.x + Math.cos(angle) * radius, ISLAND.z + Math.sin(angle) * radius, spec, out)) {
      return true;
    }
  }
  return false;
}

/** As `pickIslandPoint`, but drawn from a disc around one clump centre. */
function pickClusterPoint(
  random: () => number,
  spec: ScatterSpec,
  clusters: number[],
  index: number,
  clusterRadius: number,
  out: THREE.Vector3,
): boolean {
  const cx = clusters[index];
  const cz = clusters[index + 1];
  for (let attempt = 0; attempt < PLACEMENT_ATTEMPTS; attempt++) {
    const angle = random() * TAU;
    const radius = Math.sqrt(random()) * clusterRadius;
    if (acceptPoint(cx + Math.cos(angle) * radius, cz + Math.sin(angle) * radius, spec, out)) {
      return true;
    }
  }
  return false;
}

/**
 * Returns a factory producing normalised copies of a floating prop: scaled to
 * `targetHeight` metres tall, centred horizontally, and with its resting
 * waterline at y = 0 so buoyancy can drive the object origin directly.
 */
function normaliseFloater(source: THREE.Group, targetHeight: number): () => THREE.Object3D {
  const box = new THREE.Box3().setFromObject(source, true);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = targetHeight / Math.max(1e-4, size.y);

  // Floating debris sits with roughly 40% of its height submerged.
  const waterlineY = box.min.y + size.y * 0.4;

  return () => {
    const inner = source.clone(true);
    inner.position.set(-center.x, -waterlineY, -center.z);

    const holder = new THREE.Object3D();
    const scaled = new THREE.Group();
    scaled.scale.setScalar(scale);
    scaled.add(inner);
    holder.add(scaled);

    holder.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    return holder;
  };
}
