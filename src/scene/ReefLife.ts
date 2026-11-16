import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cos,
  mix,
  positionGeometry,
  positionLocal,
  sin,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { ISLAND, REEF_BAND, reefPatches, seafloorHeight, seafloorPlacementHeight } from './Seafloor';
import { applyGroundShading, type GroundShadingInputs } from './groundShading';
import type { GardenTheme, RockGarden } from './Rocks';
import { InstanceField } from './reef/InstanceField';
import type { MeshBuilder, RGB } from './reef/MeshBuilder';
import { bumpedNormal, lifePattern, worldNormalToView } from './reef/LifeShading';
import {
  buildAnemone,
  buildBarrelSponge,
  buildBrainCoral,
  buildBommie,
  buildBranchingCoral,
  buildElkhornCoral,
  buildEncrustingSponge,
  buildMoundCoral,
  buildRock,
  buildSeaFan,
  buildSeaPlume,
  buildSeaWhip,
  buildSeagrassClump,
  buildShell,
  buildStarfish,
  buildTubeSponge,
  buildUrchin,
  buildVaseSponge,
} from './reef/ReefSpecies';

/**
 * The living reef: corals, sponges, sea fans, anemones, seagrass meadows, rocks
 * and the small things on the sand, all built in code and instanced.
 *
 * **Why procedural.** The reef used to be four downloaded photogrammetry and kit
 * GLBs (about 2.4 MB with their LODs) scattered by `Props`; all four are now
 * gone from the picture, the kit carpet (`soft_coral_set`) last because it was
 * Indo-Pacific soft coral on a Caribbean reef. They are good assets
 * and they are the wrong tool for this job: a reef reads as *alive* through
 * variety — of form, colour and motion — at the scale of hundreds of colonies,
 * and four fixed meshes recoloured by nothing repeat visibly within one patch.
 * Here every species is a template of a few hundred triangles built once from
 * tubes, lathes and blobs (`reef/MeshBuilder`), and every instance carries its
 * own colour from a per-species palette, its own scale, yaw and lean.
 *
 * **Placement is the heightfield's, never a coordinate.** The plateau reef sits
 * on `reefPatches()` — the same centres `Props` grows its rock on and `Fish`
 * stations its residents over, so a diver who finds coral finds fish. The
 * island's reef and meadows are rejection-sampled against `seafloorHeight` in a
 * light-limited depth band, exactly as `Kelp` does, so they follow the island
 * wherever its owner reshapes it. Within a site, growth is clustered into
 * colonies rather than scattered — see `colonySample`.
 *
 * **Motion is one shader term.** Anything that sways — seagrass, fans, whips,
 * anemone tentacles — carries a per-vertex amplitude baked into the template,
 * and the vertex stage displaces it along the swell direction with a phase that
 * travels across the bed (`k · x`) and lags toward the tip, which is the same
 * model `Kelp` derives from linear wave theory. Tentacles add a fast local
 * wriggle on top. Rigid corals carry amplitude zero and pay one multiply.
 *
 * **Lighting is the scene's.** One `MeshStandardNodeMaterial` (two — solid and
 * two-sided) with `groundShading` applied, so the caustics, the cloud shade and
 * the contact darkening that the seafloor and the props receive land on the reef
 * too. Surface detail — brain-coral meanders, polyp pores — is procedural noise
 * of the template-space position with a screen-derivative bump, so it costs no
 * texture and no UVs and sits still on a swaying instance.
 *
 * **Gardens on the rock.** Besides the reef patches and the island sites, every
 * outcrop `Rocks` sculpts is a garden — Caribbean coral, sponge or gorgonian,
 * by depth — and part of each species grows on the rock itself, on the ledges
 * and tops `Rocks` found facing up or out, the way abyssal-garden clusters its
 * sponges and corals on rock surfaces (`world.js`, "sponges and coral").
 *
 * **Budget.** One `InstanceField` per species template, dealt by distance from
 * the camera (`reef/InstanceField`), so only the reef within visibility range
 * is vertex-shaded; the bommie carries a three-level LOD chain. The tier's
 * `reefLife` fraction is a rank threshold, never a rebuild, so thinning removes
 * the same colonies everywhere. The whole group goes invisible when the viewer
 * is far enough above the sea that none of it can resolve.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

const REEF_LIFE_SEED = 0x2ee71f;

/** Deep-water dispersion, for the sway clock. */
const GRAVITY = 9.81;

/** Fast wriggle frequency for tentacles, rad/s. */
const WIGGLE_OMEGA = 2.3;

/** How far the phase lags toward a tip, per metre of sway amplitude. */
const SWAY_LAG = 9;

/** Above this eye height the reef is hidden outright. */
const HIDE_ABOVE = 90;

// ----------------------------------------------------------------- the species

/**
 * How much of a species grows in a rock garden of each theme, per garden at
 * full detail.
 *
 * A garden is one outcrop of `Rocks` dressed as a Caribbean reef zone, and the
 * three themes are the three that zone by depth and exposure there: the
 * shallow coral garden (staghorn thickets, elkhorn, brain and star coral), the
 * sponge garden of the deeper slope (tube, vase and giant barrel sponges over
 * orange encrusting sponge) and the gorgonian forest of walls in the current
 * (sea fans, rods and plumes). Grouping them is what makes a reef readable;
 * everything scattered evenly everywhere reads as noise.
 */
type GardenCounts = Partial<Record<GardenTheme, number>>;

interface SpeciesDef {
  name: string;
  build: (seed: number) => MeshBuilder;
  /** Coarser levels of the same template, for species with an LOD chain. */
  lods?: readonly ((seed: number) => MeshBuilder)[];
  /** Level switches, in distance over instance radius; see `InstanceField`. */
  switches?: readonly number[];
  /** Distinct templates, each its own draw. */
  variants: number;
  /** Two-sided material: ribbons and fans. */
  thin: boolean;
  palette: readonly number[];
  /** Uniform scale range. */
  scale: readonly [number, number];
  /** Extra horizontal-only stretch range, for squat vs tall colonies. */
  squash?: readonly [number, number];
  /** Count on the plateau reef and on the island reef, at full detail. */
  plateau: number;
  island: number;
  /** Count per rock garden of each theme, at full detail. */
  garden?: GardenCounts;
  /** Fraction of the garden count grown on the rock rather than on the sand at its foot. */
  onRock?: number;
  /** Grows in the seagrass meadows rather than on the reef. */
  meadow?: boolean;
  /** Fraction placed on the open sand apron around a reef site. */
  apron?: number;
  /** Metres pushed into the sand. */
  sink: number;
  /** How far the instance up axis leans toward the slope normal, 0..1. */
  followSlope: number;
  /** Random lean, radians. */
  lean: number;
  /** Turn broadside to the current (fans). */
  faceCurrent?: boolean;
  /** Colony size; 1 for solitary. */
  colony: readonly [number, number];
  /** Template radius at scale 1, metres: sets the cull and LOD ranges. */
  size: number;
}

/**
 * The species, Caribbean c. 1700: every form here has a western-Atlantic
 * analogue, and the Indo-Pacific table coral that used to be on this list has
 * gone (elkhorn takes its place as the flat, horizontal form).
 *
 * The plateau counts of the massive corals and the sponges are raised to
 * stand in for `Props`' retired soft-coral carpet, which put 586 colonies and
 * 600k triangles on these same patches: mound forms become brain and star
 * coral, the upright tubes and barrels become tube, vase and barrel sponges,
 * the plates elkhorn. These templates are 250-600 triangles and drawn only
 * inside their cull range, so the stand-in costs a fraction of what it
 * replaces (reef plus rocks, High, reef-dive: +0.3 ms GPU at p10).
 */
const SPECIES: readonly SpeciesDef[] = [
  {
    // The structure the rest of the reef is scattered around — see
    // `buildBommie`. Colours are baked per colony, so the palette is a near
    // white that only shifts the whole head warm or cool. 11-12k triangles at
    // LOD0; LOD1 keeps the rock and the cheap colonies and lumps the rest
    // (~3k), LOD2 is the rock and lumps (~0.9k). Before the chain every
    // bommie drew all 12k at any range, which was the Ultra/Max triangle bill.
    name: 'bommie',
    build: buildBommie,
    lods: [(s) => buildBommie(s, 1), (s) => buildBommie(s, 2)],
    switches: [7, 18],
    variants: 2,
    thin: true,
    palette: [0xffffff, 0xf4f0e8, 0xe8eef4, 0xf8ecec],
    scale: [0.8, 1.5],
    squash: [0.85, 1.25],
    // [underwater-look-2] 30/24 -> 42/32: more heads at mid and far range,
    // which is what gives a reef view its layered silhouettes (view4).
    plateau: 42,
    island: 32,
    garden: { coral: 1 },
    sink: 0.25,
    followSlope: 0.8,
    lean: 0.05,
    colony: [1, 2],
    size: 2,
  },
  {
    // Diploria, Colpophyllia.
    name: 'brain-coral',
    build: buildBrainCoral,
    variants: 2,
    thin: false,
    palette: [0xb8a36a, 0x8fae6b, 0xc98f7a, 0xa5a25a, 0xd2b68c, 0x9fb88a, 0x6fa87a, 0xc8b048],
    scale: [0.6, 1.5],
    squash: [0.85, 1.25],
    plateau: 300,
    island: 70,
    garden: { coral: 7, sponge: 2 },
    onRock: 0.5,
    sink: 0.08,
    followSlope: 0.8,
    lean: 0.08,
    colony: [2, 4],
    size: 0.7,
  },
  {
    // Star coral, Orbicella: the massive boulder colonies.
    name: 'star-coral',
    build: buildMoundCoral,
    variants: 1,
    thin: false,
    palette: [0xa88f63, 0x8c9a6e, 0xb59a7c, 0x8a947e, 0xc4a160, 0x9a8a70],
    scale: [0.9, 2.2],
    squash: [0.8, 1.3],
    plateau: 260,
    island: 50,
    garden: { coral: 6, sponge: 2, fan: 2 },
    onRock: 0.5,
    sink: 0.12,
    followSlope: 0.9,
    lean: 0.06,
    colony: [1, 3],
    size: 0.8,
  },
  {
    // Staghorn, Acropora cervicornis: tan thickets with pale growing tips.
    name: 'staghorn-coral',
    build: buildBranchingCoral,
    variants: 2,
    thin: false,
    palette: [0xc9a46a, 0xb8925a, 0xd6b48b, 0xa98d5f, 0xc7a07a, 0xb59a6a],
    scale: [0.7, 1.4],
    squash: [0.85, 1.3],
    plateau: 140,
    island: 90,
    garden: { coral: 16 },
    onRock: 0.35,
    sink: 0.05,
    followSlope: 0.4,
    lean: 0.12,
    colony: [3, 7],
    size: 0.9,
  },
  {
    // Elkhorn, Acropora palmata: the reef-crest builder.
    name: 'elkhorn-coral',
    build: buildElkhornCoral,
    variants: 2,
    thin: false,
    palette: [0xc9a46a, 0xb8925a, 0xd4b07a, 0xa88a58],
    scale: [0.8, 1.6],
    squash: [0.9, 1.2],
    plateau: 90,
    island: 30,
    garden: { coral: 7 },
    onRock: 0.5,
    sink: 0.05,
    followSlope: 0.3,
    lean: 0.1,
    colony: [1, 3],
    size: 1,
  },
  {
    // Aplysina, Callyspongia: yellow, purple and orange tubes.
    name: 'tube-sponge',
    build: buildTubeSponge,
    variants: 2,
    thin: false,
    palette: [0x9a5bb8, 0xe0a63a, 0xd76a3a, 0x6b8fd6, 0xc4485c, 0xb070c8, 0xf0c83a],
    scale: [0.7, 1.4],
    plateau: 320,
    island: 70,
    garden: { coral: 3, sponge: 14, fan: 4 },
    onRock: 0.5,
    sink: 0.04,
    followSlope: 0.3,
    lean: 0.12,
    colony: [1, 3],
    size: 0.7,
  },
  {
    // Giant barrel sponge, Xestospongia muta.
    name: 'barrel-sponge',
    build: buildBarrelSponge,
    variants: 1,
    thin: false,
    palette: [0x9c5a3e, 0xb8743f, 0x8a4a3a, 0xa86848],
    scale: [0.7, 1.9],
    squash: [0.85, 1.15],
    plateau: 60,
    island: 16,
    garden: { sponge: 6, fan: 1 },
    onRock: 0.3,
    sink: 0.05,
    followSlope: 0.3,
    lean: 0.08,
    colony: [1, 1],
    size: 0.8,
  },
  {
    // Azure and lavender vase sponges, Callyspongia plicifera, Niphates.
    name: 'vase-sponge',
    build: buildVaseSponge,
    variants: 2,
    thin: false,
    palette: [0xb070c8, 0x8a5bc4, 0xd08ab0, 0x6b8fd6, 0x9a6ad0, 0x5aa8d8],
    scale: [0.7, 1.5],
    plateau: 160,
    island: 30,
    garden: { sponge: 9, fan: 3 },
    onRock: 0.4,
    sink: 0.04,
    followSlope: 0.35,
    lean: 0.12,
    colony: [1, 2],
    size: 0.6,
  },
  {
    // Orange icing and red encrusting sponges: Diplastrella, Monanchora.
    name: 'encrusting-sponge',
    build: buildEncrustingSponge,
    variants: 2,
    thin: false,
    palette: [0xe0662e, 0xd8402e, 0xe8a03a, 0xc83a4a, 0xf0b040],
    scale: [0.6, 1.6],
    squash: [0.8, 1.4],
    plateau: 0,
    island: 0,
    garden: { coral: 4, sponge: 12, fan: 7 },
    onRock: 1,
    sink: 0.03,
    followSlope: 1,
    lean: 0.05,
    colony: [1, 3],
    size: 0.5,
  },
  {
    // Gorgonia ventalina, the purple common sea fan.
    name: 'sea-fan',
    build: buildSeaFan,
    variants: 2,
    thin: true,
    // [underwater-look-2] Plus the yellow and rose morphs of G. ventalina.
    palette: [0x8a4ab0, 0xb63c7a, 0x9a5ab8, 0xc8a03a, 0x7a4aa0, 0xe0c048, 0xd06a90],
    scale: [0.7, 1.6],
    plateau: 170,
    island: 60,
    garden: { coral: 3, sponge: 3, fan: 14 },
    onRock: 0.6,
    sink: 0.06,
    followSlope: 0.15,
    lean: 0.1,
    faceCurrent: true,
    colony: [1, 3],
    size: 0.8,
  },
  {
    // Giant Caribbean anemone, Condylactis gigantea.
    name: 'anemone',
    build: buildAnemone,
    variants: 1,
    thin: false,
    palette: [0xd59fd8, 0xb6d68a, 0xe3b7a0, 0x9fd4d0, 0xe0a0b0],
    scale: [0.8, 1.6],
    plateau: 84,
    island: 50,
    garden: { coral: 3 },
    onRock: 0.5,
    sink: 0.03,
    followSlope: 0.7,
    lean: 0.1,
    colony: [1, 3],
    size: 0.4,
  },
  {
    // Sea rods: Plexaura, Eunicea — brown, tan, purple.
    name: 'sea-rod',
    build: buildSeaWhip,
    variants: 1,
    thin: false,
    // [underwater-look-2] Plus Muricea's orange and a purple Eunicea.
    palette: [0x8a6a4a, 0x9a7a5a, 0x7a4a6a, 0xb08a5a, 0x6a4a5a, 0xd0703a, 0x9a4a9a],
    scale: [0.7, 1.4],
    plateau: 210,
    island: 70,
    garden: { coral: 5, sponge: 6, fan: 12 },
    onRock: 0.5,
    sink: 0.04,
    followSlope: 0.2,
    lean: 0.08,
    colony: [1, 4],
    size: 0.7,
  },
  {
    // Sea plumes, Antillogorgia: the soft carpet of a Caribbean reef, and
    // the replacement for `Props`' generic soft-coral kit, which is retired.
    name: 'sea-plume',
    build: buildSeaPlume,
    variants: 2,
    thin: false,
    // [underwater-look-2] Plus the brighter purple and gold plumes.
    palette: [0x7a5a8a, 0x8a6a5a, 0x6a4a7a, 0x9a7a6a, 0xa07a9a, 0xb05ac0, 0xc89a50],
    scale: [0.8, 1.5],
    plateau: 340,
    island: 150,
    garden: { coral: 4, sponge: 3, fan: 10 },
    onRock: 0.35,
    sink: 0.04,
    followSlope: 0.2,
    lean: 0.1,
    colony: [2, 5],
    size: 0.8,
  },
  {
    name: 'reef-rock',
    build: buildRock,
    variants: 2,
    thin: false,
    palette: [0x6f6a60, 0x7c7468, 0x67645d, 0x857a6a, 0x726c62],
    scale: [0.4, 1.6],
    squash: [0.8, 1.5],
    plateau: 170,
    island: 110,
    apron: 0.45,
    sink: 0.1,
    followSlope: 1,
    lean: 0.15,
    colony: [1, 4],
    size: 0.7,
  },
  {
    name: 'shell',
    build: buildShell,
    variants: 1,
    thin: false,
    palette: [0xf0e6d6, 0xe8c9a8, 0xd9b49a, 0xf3d9c0],
    scale: [0.7, 1.4],
    plateau: 110,
    island: 90,
    apron: 0.7,
    sink: 0.02,
    followSlope: 1,
    lean: 0.2,
    colony: [1, 2],
    size: 0.15,
  },
  {
    // Cushion star, Oreaster reticulatus.
    name: 'starfish',
    build: buildStarfish,
    variants: 1,
    thin: false,
    palette: [0xe0662e, 0xc8323a, 0xe8b04a, 0xd0802e, 0xb8502e],
    scale: [0.8, 1.5],
    plateau: 90,
    island: 70,
    apron: 0.6,
    sink: 0.005,
    followSlope: 1,
    lean: 0.05,
    colony: [1, 2],
    size: 0.2,
  },
  {
    // Long-spined urchin, Diadema antillarum.
    name: 'urchin',
    build: buildUrchin,
    variants: 1,
    thin: false,
    palette: [0x2a1f2e, 0x3a2a40, 0x24222c, 0x4a2a30],
    scale: [0.8, 1.4],
    plateau: 70,
    island: 50,
    garden: { coral: 4, sponge: 2 },
    onRock: 0.4,
    sink: 0.02,
    followSlope: 0.8,
    lean: 0.1,
    colony: [2, 5],
    size: 0.25,
  },
  {
    name: 'seagrass',
    build: buildSeagrassClump,
    variants: 2,
    thin: true,
    palette: [0x7fae48, 0x8fba52, 0x6f9e44, 0x9ab858, 0x88a84a],
    scale: [0.8, 1.5],
    plateau: 1600,
    island: 2000,
    meadow: true,
    sink: 0.02,
    followSlope: 0.6,
    lean: 0.08,
    colony: [1, 1],
    size: 0.4,
  },
];

/**
 * How far a reef instance is drawn: `CULL_BASE + radius * CULL_PER_SIZE`,
 * capped. Visibility under water here runs 45-70 m, so a 2 m bommie draws to
 * the edge of it and a 15 cm shell stops at about thirty metres, where it is a
 * pixel or two of something the colour of the sand.
 */
const CULL_BASE = 24;
/** Scales every garden count: the table above is per garden at density 1. */
// [underwater-look-2] 1.5 -> 2.2: view4's rock faces are crowded with life.
const GARDEN_DENSITY = 2.2;
const CULL_PER_SIZE = 40;
const CULL_MAX = 130;

// ----------------------------------------------------------------- the sites

interface Site {
  x: number;
  z: number;
  radius: number;
}

/** Island reef sites: count, depth band and spacing. */
const ISLAND_REEF_SITES = 9;
const ISLAND_REEF_DEPTH: readonly [number, number] = [3.5, 15];
const ISLAND_REEF_RADIUS = 13;
/** Island meadows: shallower, wider. */
const ISLAND_MEADOWS = 12;
const ISLAND_MEADOW_DEPTH: readonly [number, number] = [1.6, 9];
const ISLAND_MEADOW_RADIUS = 7;
/** Plateau meadows: two per reef patch, on the sand beside it. */
const PLATEAU_MEADOWS_PER_PATCH = 1;
const PLATEAU_MEADOW_RADIUS = 7;
const SITE_SPACING = 45;

function findIslandSites(
  random: () => number,
  count: number,
  depth: readonly [number, number],
  radius: number,
  avoid: Site[],
): Site[] {
  const out: Site[] = [];
  for (let tries = 0; tries < count * 400 && out.length < count; tries++) {
    const a = random() * Math.PI * 2;
    const r = ISLAND.radius * (0.7 + random() * 0.75);
    const x = ISLAND.x + Math.cos(a) * r;
    const z = ISLAND.z + Math.sin(a) * r;
    const d = -seafloorPlacementHeight(x, z);
    if (d < depth[0] || d > depth[1]) continue;
    if (slopeAt(x, z) > 0.45) continue;
    let clear = true;
    for (const s of out.concat(avoid)) {
      if (Math.hypot(s.x - x, s.z - z) < SITE_SPACING) {
        clear = false;
        break;
      }
    }
    if (!clear) continue;
    out.push({ x, z, radius });
  }
  return out;
}

function slopeAt(x: number, z: number): number {
  const h = 2;
  const dx = (seafloorPlacementHeight(x + h, z) - seafloorPlacementHeight(x - h, z)) / (2 * h);
  const dz = (seafloorPlacementHeight(x, z + h) - seafloorPlacementHeight(x, z - h)) / (2 * h);
  return Math.hypot(dx, dz);
}

function floorNormal(x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
  const h = 0.6;
  const dx = (seafloorPlacementHeight(x + h, z) - seafloorPlacementHeight(x - h, z)) / (2 * h);
  const dz = (seafloorPlacementHeight(x, z + h) - seafloorPlacementHeight(x, z - h)) / (2 * h);
  return out.set(-dx, 1, -dz).normalize();
}

// ------------------------------------------------------------------ the class

export interface ReefLifeOptions {
  /** Ground shading inputs, for caustics and cloud shade on the reef. */
  shading?: Omit<GroundShadingInputs, 'foliage'>;
  /** Rock gardens from `Rocks`: outcrops to grow on and around. */
  gardens?: readonly RockGarden[];
}

/** One placement: where, and which way is up for it. */
interface Placement {
  x: number;
  y: number;
  z: number;
  /** Surface normal it grows from. */
  nx: number;
  ny: number;
  nz: number;
}

export class ReefLife {
  readonly object = new THREE.Group();

  private readonly fields: InstanceField[] = [];
  private readonly solid: THREE.MeshStandardNodeMaterial;
  private readonly thin: THREE.MeshStandardNodeMaterial;
  private detail = 1;
  private wantVisible = true;
  private disposed = false;

  private swellOmega = Math.sqrt(GRAVITY * ((Math.PI * 2) / 47));
  private clock = 0;

  private readonly uSwayPhase = uniform(0);
  private readonly uWigglePhase = uniform(0);
  private readonly uFlowDir = uniform(new THREE.Vector2(1, 0));
  private readonly uSwellK = uniform((Math.PI * 2) / 47);
  /** Overall current strength, 0.3 calm .. 1.3 storm. */
  private readonly uCurrent = uniform(0.6);
  private readonly uLean = uniform(0.25);

  /** Bearing fans are turned to face, radians. Fixed at placement. */
  private readonly fanBearing: number;

  /** Sites, published for the creatures and the harness. */
  readonly sites: { plateau: Site[]; island: Site[]; meadows: Site[]; gardens: Site[] };

  /** Milliseconds the constructor took, templates and placement. */
  readonly buildMs: number;

  constructor(detail: number, options: ReefLifeOptions = {}) {
    const started = performance.now();
    this.object.name = 'reef-life';
    this.solid = this.buildMaterial(THREE.FrontSide);
    this.thin = this.buildMaterial(THREE.DoubleSide);
    if (options.shading) {
      applyGroundShading(this.solid, options.shading);
      applyGroundShading(this.thin, options.shading);
    }

    // The site draws keep the stream they always had, so the creatures' homes
    // and the fish stations do not move.
    const random = mulberry32(REEF_LIFE_SEED);
    this.fanBearing = random() * Math.PI;

    const plateau: Site[] = reefPatches().map((p) => ({ x: p.x, z: p.z, radius: REEF_BAND.patchRadius * 1.1 }));
    const island = findIslandSites(random, ISLAND_REEF_SITES, ISLAND_REEF_DEPTH, ISLAND_REEF_RADIUS, []);
    const islandMeadows = findIslandSites(random, ISLAND_MEADOWS, ISLAND_MEADOW_DEPTH, ISLAND_MEADOW_RADIUS, island);
    const plateauMeadows: Site[] = [];
    for (const p of plateau) {
      for (let i = 0; i < PLATEAU_MEADOWS_PER_PATCH; i++) {
        const a = random() * Math.PI * 2;
        const r = p.radius + PLATEAU_MEADOW_RADIUS * (0.6 + random() * 0.8);
        plateauMeadows.push({ x: p.x + Math.cos(a) * r, z: p.z + Math.sin(a) * r, radius: PLATEAU_MEADOW_RADIUS });
      }
    }
    const gardens = options.gardens ?? [];
    this.sites = {
      plateau,
      island,
      meadows: plateauMeadows.concat(islandMeadows),
      gardens: gardens.map((g) => ({ x: g.x, z: g.z, radius: g.radius })),
    };

    for (const def of SPECIES) {
      for (let v = 0; v < def.variants; v++) {
        const placements: Placement[] = [];
        for (const region of ['plateau', 'island'] as const) {
          const total = Math.ceil(def[region] / def.variants);
          if (total <= 0) continue;
          const sites = def.meadow
            ? region === 'plateau'
              ? plateauMeadows
              : islandMeadows
            : region === 'plateau'
              ? plateau
              : island;
          if (sites.length === 0) continue;
          this.placeOnSites(def, sites, total, random, placements);
        }
        if (def.garden) this.placeInGardens(def, v, gardens, placements);
        this.buildField(def, v, placements, random);
      }
    }

    this.setDetail(detail);
    this.buildMs = performance.now() - started;
  }

  /** Fraction of placed instances drawn, 0..1. A rank threshold, never a rebuild. */
  setDetail(detail: number): void {
    this.detail = Math.max(0, Math.min(1, detail));
    for (const field of this.fields) field.setDetail(this.detail);
    this.applyVisibility();
  }

  getDetail(): number {
    return this.detail;
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** Hides everything when the eye is high above the sea. */
  setCameraHeight(height: number): void {
    const visible = height < HIDE_ABOVE;
    if (visible !== this.cameraAllows) {
      this.cameraAllows = visible;
      this.applyVisibility();
    }
  }
  private cameraAllows = true;

  /**
   * Height test plus the deal: each species field draws only what is within
   * visibility range of the camera, at the level its apparent size wants.
   * This replaces the per-region switch, which drew the whole of the island's
   * reef whenever the camera was anywhere near the island.
   */
  setCameraPosition(position: THREE.Vector3): void {
    this.setCameraHeight(position.y);
    if (!this.object.visible) return;
    for (const field of this.fields) field.update(position);
  }

  /** Same three numbers the surface and the kelp take. See `Kelp.setSwell`. */
  setSwell(bearingRadians: number, windSpeed: number, peakWavelength = 47): void {
    (this.uFlowDir.value as THREE.Vector2).set(Math.cos(bearingRadians), Math.sin(bearingRadians));
    const lambda = Math.max(8, Math.min(400, peakWavelength));
    const k = (Math.PI * 2) / lambda;
    this.uSwellK.value = k;
    this.swellOmega = Math.sqrt(GRAVITY * k);
    const u = Math.max(0, windSpeed);
    this.uCurrent.value = Math.min(1.3, 0.35 + 0.045 * u);
    this.uLean.value = Math.min(0.5, 0.15 + 0.02 * u);
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock += dt;
    this.pushPhases();
  }

  resetClock(time = 0): void {
    this.clock = time;
    this.pushPhases();
  }

  /** Floor elevation at a point, so a harness can frame a site. */
  floorAt(x: number, z: number): number {
    return seafloorHeight(x, z);
  }

  /** Total triangles currently drawn, for the harness. */
  triangleCount(): number {
    if (!this.object.visible) return 0;
    let n = 0;
    for (const field of this.fields) n += field.triangleCount();
    return n;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const field of this.fields) {
      for (const mesh of field.meshes) mesh.geometry.dispose();
      field.dispose();
    }
    this.solid.dispose();
    this.thin.dispose();
    this.object.removeFromParent();
  }

  // ---------------------------------------------------------------- internals

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.cameraAllows && this.detail > 0;
  }

  private pushPhases(): void {
    // Wrapped in float64 on the CPU so the shader's phase never grows large.
    const tau = Math.PI * 2;
    this.uSwayPhase.value = (this.clock * this.swellOmega) % tau;
    this.uWigglePhase.value = (this.clock * WIGGLE_OMEGA) % tau;
  }

  /** Sand placements on reef sites and meadows, clustered into colonies. */
  private placeOnSites(def: SpeciesDef, sites: Site[], total: number, random: () => number, out: Placement[]): void {
    const normal = new THREE.Vector3();
    let colonyLeft = 0;
    let colony = { x: 0, z: 0 };
    let siteIndex = Math.floor(random() * sites.length);
    for (let i = 0; i < total; i++) {
      if (colonyLeft <= 0) {
        siteIndex = (siteIndex + 1 + Math.floor(random() * 2)) % sites.length;
        const site = sites[siteIndex];
        const apron = def.apron !== undefined && random() < def.apron;
        colony = apron ? apronSample(random, site) : colonySample(random, site);
        colonyLeft = def.colony[0] + Math.floor(random() * (def.colony[1] - def.colony[0] + 1));
      }
      colonyLeft--;
      const spread = def.colony[1] > 1 ? 1.4 + random() * 1.2 : 0;
      const a = random() * Math.PI * 2;
      const r = Math.sqrt(random()) * spread;
      const x = colony.x + Math.cos(a) * r;
      const z = colony.z + Math.sin(a) * r;
      floorNormal(x, z, normal);
      out.push({ x, y: seafloorPlacementHeight(x, z), z, nx: normal.x, ny: normal.y, nz: normal.z });
    }
  }

  /**
   * Garden placements: part on the outcrop itself, on the ledges and tops
   * `Rocks` found facing up or out, the rest on the sand ring at its foot,
   * clustered the same way the reef sites are. Its own random stream per
   * species and variant, so adding a species never moves another.
   */
  private placeInGardens(def: SpeciesDef, variant: number, gardens: readonly RockGarden[], out: Placement[]): void {
    const random = mulberry32(REEF_LIFE_SEED ^ (hashString(def.name) + variant * 7919 + 0x6a5d));
    const normal = new THREE.Vector3();
    for (const g of gardens) {
      const want = def.garden?.[g.theme] ?? 0;
      // The plateau's gardens are the reef patches the play area is built
      // around, so they grow twice as thick as the shelf's.
      const density = GARDEN_DENSITY * (g.region === 'plateau' ? 2 : 1);
      const n = Math.round((want * density) / def.variants + (random() - 0.5));
      const anchors = g.anchors.length / 6;
      let colonyLeft = 0;
      let cx = 0;
      let cz = 0;
      for (let i = 0; i < n; i++) {
        if (anchors > 0 && random() < (def.onRock ?? 0)) {
          const k = Math.floor(random() * anchors) * 6;
          out.push({
            x: g.anchors[k],
            y: g.anchors[k + 1],
            z: g.anchors[k + 2],
            nx: g.anchors[k + 3],
            ny: g.anchors[k + 4],
            nz: g.anchors[k + 5],
          });
          continue;
        }
        if (colonyLeft <= 0) {
          const a = random() * Math.PI * 2;
          const r = g.radius * (0.55 + random() * 0.75);
          cx = g.x + Math.cos(a) * r;
          cz = g.z + Math.sin(a) * r;
          colonyLeft = def.colony[0] + Math.floor(random() * (def.colony[1] - def.colony[0] + 1));
        }
        colonyLeft--;
        const spread = def.colony[1] > 1 ? 1.2 + random() * 1.2 : 0;
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(random()) * spread;
        const x = cx + Math.cos(a) * r;
        const z = cz + Math.sin(a) * r;
        const y = seafloorPlacementHeight(x, z);
        if (y > -1.5) continue;
        floorNormal(x, z, normal);
        out.push({ x, y, z, nx: normal.x, ny: normal.y, nz: normal.z });
      }
    }
  }

  private buildField(def: SpeciesDef, v: number, placements: Placement[], random: () => number): void {
    if (placements.length === 0) return;
    const seed = REEF_LIFE_SEED ^ (hashString(def.name) + v * 977);
    const levels = [def.build(seed).build(), ...(def.lods ?? []).map((lod) => lod(seed).build())];
    const field = new InstanceField({
      name: `reef-${def.name}-${v}`,
      levels,
      material: def.thin ? this.thin : this.solid,
      switches: def.switches ?? [],
      cullBase: CULL_BASE,
      cullPerSize: CULL_PER_SIZE,
      cullMax: CULL_MAX,
      refresh: 3,
      cell: 48,
    });

    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    const normal = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const tilt = new THREE.Quaternion();
    const lean = new THREE.Quaternion();
    const yaw = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const color = new THREE.Color();

    for (const p of placements) {
      const s = def.scale[0] + (def.scale[1] - def.scale[0]) * Math.pow(random(), 1.6);
      const squash = def.squash ? def.squash[0] + (def.squash[1] - def.squash[0]) * random() : 1;
      normal.set(p.nx, p.ny, p.nz);
      // On a rock face nothing grows straight out sideways: life turns toward
      // the light, the more so the less it follows its substrate.
      const toward = normal.clone().lerp(up, 1 - def.followSlope).normalize();
      if (toward.y < 0.35) toward.lerp(up, 0.5).normalize();
      tilt.setFromUnitVectors(up, toward);
      lean.setFromEuler(euler.set((random() - 0.5) * 2 * def.lean, 0, (random() - 0.5) * 2 * def.lean));
      const bearing = def.faceCurrent ? this.fanBearing + (random() - 0.5) * 0.7 : random() * Math.PI * 2;
      yaw.setFromAxisAngle(up, bearing);
      quaternion.copy(tilt).multiply(lean).multiply(yaw);
      position.set(p.x, p.y - def.sink * s, p.z);
      scale.set(s * squash, s, s * (def.squash ? def.squash[0] + (def.squash[1] - def.squash[0]) * random() : 1));
      matrix.compose(position, quaternion, scale);
      color.setHex(def.palette[Math.floor(random() * def.palette.length)]);
      // Per-instance brightness jitter, so a colony is not one swatch.
      color.multiplyScalar(0.82 + random() * 0.36);
      field.add(matrix, color, def.size * s, random());
    }
    field.seal();
    for (const mesh of field.meshes) {
      mesh.receiveShadow = true;
      this.object.add(mesh);
    }
    this.fields.push(field);
  }

  private buildMaterial(side: THREE.Side): THREE.MeshStandardNodeMaterial {
    const material = new THREE.MeshStandardNodeMaterial({
      vertexColors: true,
      roughness: 0.78,
      metalness: 0,
      side,
    });
    material.name = side === THREE.DoubleSide ? 'reef-life-thin' : 'reef-life';
    material.fog = false;

    const swayData: Node = attribute('reefSway', 'vec2');
    const surf: Node = attribute('reefSurf', 'float');

    // --- motion ---------------------------------------------------------------
    // `positionLocal` here is already instanced, and the mesh sits at the world
    // origin, so it *is* the world position — the flow can be a world direction.
    material.positionNode = Fn(() => {
      const p = positionLocal.toVar('reefP');
      const amp = swayData.x.toVar('reefAmp');
      const wig = swayData.y;
      const flow = vec3(this.uFlowDir.x, 0, this.uFlowDir.y);
      const downwave = p.x.mul(this.uFlowDir.x).add(p.z.mul(this.uFlowDir.y));
      const phase = this.uSwayPhase.sub(downwave.mul(this.uSwellK)).sub(amp.mul(SWAY_LAG));
      // Two incommensurate components so the bed never settles into a metronome.
      const bend = sin(phase).add(sin(phase.mul(0.43).add(1.7)).mul(0.5)).mul(this.uCurrent).add(this.uLean);
      const offset = flow.mul(amp.mul(bend)).toVar('reefOffset');
      // Tentacle wriggle: a phase that varies along the tentacle and between
      // tentacles, from the vertex's own world position.
      const w = this.uWigglePhase.add(p.x.mul(7.3)).add(p.z.mul(5.9)).add(p.y.mul(11.0));
      offset.addAssign(
        vec3(sin(w), sin(w.mul(1.31).add(1.1)).mul(0.5), cos(w.mul(0.87))).mul(amp.mul(wig).mul(0.8)),
      );
      // Blades are inextensible: travel sideways costs height.
      offset.y.subAssign(amp.mul(bend.abs()).mul(0.25));
      return p.add(offset);
    })();

    // --- surface detail ---------------------------------------------------------
    // Template-space position, so the pattern sits still on a swaying instance —
    // brain-coral meanders, polyp pores; see `reef/LifeShading`.
    const pattern = lifePattern(surf, positionGeometry);
    material.colorNode = vec4(pattern.y, pattern.y, pattern.y, 1);
    const bumped = bumpedNormal(pattern.x, 0.012);
    if (side === THREE.DoubleSide) {
      // Thin, flexible life (blades, fan lattices, tentacles) stands edge-on
      // to a sun that comes from above, so its true normal leaves it nearly
      // black. Bend the normal of whatever sways toward the sky, the usual
      // foliage trick standing in for light through the leaf. Rigid parts
      // (amplitude 0, e.g. a bommie's rock) keep their own normal.
      const bendUp = swayData.x.smoothstep(0, 0.04).mul(0.6);
      material.normalNode = (mix(bumped, worldNormalToView(vec3(0, 1, 0)), bendUp) as Node).normalize();
    } else {
      material.normalNode = bumped;
    }

    return material;
  }
}

/** A point inside a reef site, centre-weighted. */
function colonySample(random: () => number, site: Site): { x: number; z: number } {
  const a = random() * Math.PI * 2;
  const r = Math.pow(random(), 0.75) * site.radius;
  return { x: site.x + Math.cos(a) * r, z: site.z + Math.sin(a) * r };
}

/** A point on the sand ring just outside a site. */
function apronSample(random: () => number, site: Site): { x: number; z: number } {
  const a = random() * Math.PI * 2;
  const r = site.radius * (0.9 + random() * 1.1);
  return { x: site.x + Math.cos(a) * r, z: site.z + Math.sin(a) * r };
}

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

export type { RGB };
