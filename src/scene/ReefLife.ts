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
import { ISLAND, REEF_BAND, reefPatches, seafloorHeight } from './Seafloor';
import { applyGroundShading, type GroundShadingInputs } from './groundShading';
import type { MeshBuilder, RGB } from './reef/MeshBuilder';
import { bumpedNormal, lifePattern, worldNormalToView } from './reef/LifeShading';
import {
  buildAnemone,
  buildBarrelSponge,
  buildBrainCoral,
  buildBommie,
  buildBranchingCoral,
  buildMoundCoral,
  buildRock,
  buildSeaFan,
  buildSeaWhip,
  buildSeagrassClump,
  buildShell,
  buildStarfish,
  buildTableCoral,
  buildTubeSponge,
  buildUrchin,
} from './reef/ReefSpecies';

/**
 * The living reef: corals, sponges, sea fans, anemones, seagrass meadows, rocks
 * and the small things on the sand, all built in code and instanced.
 *
 * **Why procedural.** The reef used to be four downloaded photogrammetry and kit
 * GLBs (about 2.4 MB with their LODs) scattered by `Props`; the three
 * photogrammetry colonies are now gone and only the kit carpet
 * (`soft_coral_set`) remains there. They are good assets
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
 * **Budget.** One draw per species per region (plateau, island), so the island
 * reef frustum-culls as a unit when it is behind the viewer. Counts scale by the
 * tier's `reefLife` fraction through `InstancedMesh.count`, never a rebuild —
 * instances are shuffled at placement so any prefix is a fair sample of every
 * site. The whole group goes invisible when the viewer is far enough above the
 * sea that none of it can resolve.
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

/**
 * Horizontal distance past a region's edge beyond which it is skipped, metres.
 * Even the clearest preset (70 m visibility) leaves well under 1% of the
 * contrast at this range, and from above the water the reef is out of sight
 * long before it.
 */
const REGION_CULL = 220;

// ----------------------------------------------------------------- the species

type Region = 'plateau' | 'island';

interface SpeciesDef {
  name: string;
  build: (seed: number) => MeshBuilder;
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
}

const SPECIES: readonly SpeciesDef[] = [
  {
    // The structure the rest of the reef is scattered around — see
    // `buildBommie`. Colours are baked per colony, so the palette is a near
    // white that only shifts the whole head warm or cool.
    name: 'bommie',
    build: buildBommie,
    variants: 2,
    thin: true,
    palette: [0xffffff, 0xf4f0e8, 0xe8eef4, 0xf8ecec],
    scale: [0.8, 1.5],
    squash: [0.85, 1.25],
    plateau: 30,
    island: 24,
    sink: 0.25,
    followSlope: 0.8,
    lean: 0.05,
    colony: [1, 2],
  },
  {
    name: 'brain-coral',
    build: buildBrainCoral,
    variants: 2,
    thin: false,
    palette: [0xb8a36a, 0x8fae6b, 0xc98f7a, 0xa5a25a, 0xd2b68c, 0x9fb88a],
    scale: [0.6, 1.5],
    squash: [0.85, 1.25],
    plateau: 130,
    island: 70,
    sink: 0.08,
    followSlope: 0.8,
    lean: 0.08,
    colony: [2, 4],
  },
  {
    name: 'mound-coral',
    build: buildMoundCoral,
    variants: 1,
    thin: false,
    palette: [0xa88f63, 0x8c9a6e, 0xb59a7c, 0x8a947e, 0xc4a160, 0x9a8a70],
    scale: [0.9, 2.2],
    squash: [0.8, 1.3],
    plateau: 90,
    island: 50,
    sink: 0.12,
    followSlope: 0.9,
    lean: 0.06,
    colony: [1, 3],
  },
  {
    name: 'branching-coral',
    build: buildBranchingCoral,
    variants: 2,
    thin: false,
    palette: [0xc58fb5, 0x7aa8c9, 0xd6b48b, 0x9cc07a, 0xe08f6a, 0xb7a0d8, 0x8fc4b8],
    scale: [0.7, 1.4],
    squash: [0.85, 1.3],
    plateau: 140,
    island: 90,
    sink: 0.05,
    followSlope: 0.4,
    lean: 0.12,
    colony: [3, 7],
  },
  {
    name: 'table-coral',
    build: buildTableCoral,
    variants: 1,
    thin: false,
    palette: [0xb9a17c, 0x8fb0b5, 0xc7a3b8, 0xd0b890, 0xa8a8c8],
    scale: [0.55, 1.15],
    squash: [0.9, 1.15],
    plateau: 50,
    island: 30,
    sink: 0.05,
    followSlope: 0.2,
    lean: 0.1,
    colony: [1, 3],
  },
  {
    name: 'tube-sponge',
    build: buildTubeSponge,
    variants: 2,
    thin: false,
    palette: [0x9a5bb8, 0xe0a63a, 0xd76a3a, 0x6b8fd6, 0xc4485c, 0xb070c8],
    scale: [0.7, 1.4],
    plateau: 80,
    island: 40,
    sink: 0.04,
    followSlope: 0.3,
    lean: 0.12,
    colony: [1, 3],
  },
  {
    name: 'barrel-sponge',
    build: buildBarrelSponge,
    variants: 1,
    thin: false,
    palette: [0x9c5a3e, 0xb8743f, 0x8a4a3a, 0xa86848],
    scale: [0.7, 1.6],
    squash: [0.85, 1.15],
    plateau: 36,
    island: 16,
    sink: 0.05,
    followSlope: 0.3,
    lean: 0.08,
    colony: [1, 1],
  },
  {
    name: 'sea-fan',
    build: buildSeaFan,
    variants: 2,
    thin: true,
    palette: [0xb63c7a, 0xd8a03a, 0x8a4ab0, 0xc8542e, 0xc46aa0],
    scale: [0.7, 1.5],
    plateau: 70,
    island: 40,
    sink: 0.06,
    followSlope: 0.15,
    lean: 0.1,
    faceCurrent: true,
    colony: [1, 3],
  },
  {
    name: 'anemone',
    build: buildAnemone,
    variants: 1,
    thin: false,
    palette: [0xd59fd8, 0xb6d68a, 0xe3b7a0, 0x9fd4d0, 0xe0a0b0],
    scale: [0.8, 1.6],
    plateau: 56,
    island: 36,
    sink: 0.03,
    followSlope: 0.7,
    lean: 0.1,
    colony: [1, 3],
  },
  {
    name: 'sea-whip',
    build: buildSeaWhip,
    variants: 1,
    thin: false,
    palette: [0xe0b23a, 0xd0602e, 0xb84a8a, 0xf0d070, 0xc83a3a],
    scale: [0.7, 1.4],
    plateau: 90,
    island: 50,
    sink: 0.04,
    followSlope: 0.2,
    lean: 0.08,
    colony: [1, 4],
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
  },
  {
    name: 'starfish',
    build: buildStarfish,
    variants: 1,
    thin: false,
    palette: [0xe0662e, 0xc8323a, 0x3a6fd0, 0x9a4ab0, 0xe8b04a],
    scale: [0.8, 1.5],
    plateau: 90,
    island: 70,
    apron: 0.6,
    sink: 0.005,
    followSlope: 1,
    lean: 0.05,
    colony: [1, 2],
  },
  {
    name: 'urchin',
    build: buildUrchin,
    variants: 1,
    thin: false,
    palette: [0x2a1f2e, 0x3a2a40, 0x24222c, 0x4a2a30],
    scale: [0.8, 1.4],
    plateau: 70,
    island: 50,
    sink: 0.02,
    followSlope: 0.8,
    lean: 0.1,
    colony: [2, 5],
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
  },
];

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
    const d = -seafloorHeight(x, z);
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
  const dx = (seafloorHeight(x + h, z) - seafloorHeight(x - h, z)) / (2 * h);
  const dz = (seafloorHeight(x, z + h) - seafloorHeight(x, z - h)) / (2 * h);
  return Math.hypot(dx, dz);
}

function floorNormal(x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
  const h = 0.6;
  const dx = (seafloorHeight(x + h, z) - seafloorHeight(x - h, z)) / (2 * h);
  const dz = (seafloorHeight(x, z + h) - seafloorHeight(x, z - h)) / (2 * h);
  return out.set(-dx, 1, -dz).normalize();
}

// ------------------------------------------------------------------ the class

export interface ReefLifeOptions {
  /** Ground shading inputs, for caustics and cloud shade on the reef. */
  shading?: Omit<GroundShadingInputs, 'foliage'>;
}

interface Kind {
  def: SpeciesDef;
  region: Region;
  mesh: THREE.InstancedMesh;
  capacity: number;
}

export class ReefLife {
  readonly object = new THREE.Group();

  private readonly kinds: Kind[] = [];
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
  readonly sites: { plateau: Site[]; island: Site[]; meadows: Site[] };

  constructor(detail: number, options: ReefLifeOptions = {}) {
    this.object.name = 'reef-life';
    this.solid = this.buildMaterial(THREE.FrontSide);
    this.thin = this.buildMaterial(THREE.DoubleSide);
    if (options.shading) {
      applyGroundShading(this.solid, options.shading);
      applyGroundShading(this.thin, options.shading);
    }

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
    this.sites = { plateau, island, meadows: plateauMeadows.concat(islandMeadows) };
    this.regionBounds = {
      plateau: boundingCircle(plateau.concat(plateauMeadows)),
      island: boundingCircle(island.concat(islandMeadows)),
    };

    for (const def of SPECIES) {
      for (const region of ['plateau', 'island'] as const) {
        const total = def[region];
        if (total <= 0) continue;
        const sites = def.meadow
          ? region === 'plateau'
            ? plateauMeadows
            : islandMeadows
          : region === 'plateau'
            ? plateau
            : island;
        if (sites.length === 0) continue;
        this.placeKind(def, region, sites, total, random);
      }
    }

    this.setDetail(detail);
  }

  /** Fraction of placed capacity drawn, 0..1. A uniform-free `count` change. */
  setDetail(detail: number): void {
    this.detail = Math.max(0, Math.min(1, detail));
    this.applyCounts();
  }

  private applyCounts(): void {
    for (const kind of this.kinds) {
      // Small populations keep at least a few, so a low tier thins the reef
      // rather than deleting whole species.
      const n = this.detail <= 0 ? 0 : Math.max(Math.min(kind.capacity, 3), Math.round(kind.capacity * this.detail));
      kind.mesh.count = n;
      kind.mesh.visible = n > 0 && this.regionNear[kind.region];
    }
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
   * Height test plus a per-region distance cull. Each region is one instanced
   * mesh per species spanning every site in it, so the renderer's frustum test
   * sees one huge sphere and never culls it; the plateau and the island reef
   * are about a kilometre apart, and water this murky hides either from the
   * other completely. Skipping the far region is the largest single saving.
   */
  setCameraPosition(position: THREE.Vector3): void {
    this.setCameraHeight(position.y);
    let changed = false;
    for (const region of ['plateau', 'island'] as const) {
      const b = this.regionBounds[region];
      const near = b === null || Math.hypot(position.x - b.x, position.z - b.z) - b.radius < REGION_CULL;
      if (near !== this.regionNear[region]) {
        this.regionNear[region] = near;
        changed = true;
      }
    }
    if (changed) this.applyCounts();
  }
  private readonly regionNear: Record<Region, boolean> = { plateau: true, island: true };
  private readonly regionBounds: Record<Region, Site | null>;

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
    let n = 0;
    for (const kind of this.kinds) {
      if (!kind.mesh.visible) continue;
      n += (kind.mesh.geometry.index!.count / 3) * kind.mesh.count;
    }
    return n;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const kind of this.kinds) {
      kind.mesh.geometry.dispose();
      kind.mesh.dispose();
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

  private placeKind(def: SpeciesDef, region: Region, sites: Site[], total: number, random: () => number): void {
    const perVariant = Math.ceil(total / def.variants);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    const normal = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const tilt = new THREE.Quaternion();
    const yaw = new THREE.Quaternion();
    const color = new THREE.Color();

    for (let v = 0; v < def.variants; v++) {
      const geometry = def.build(REEF_LIFE_SEED ^ (hashString(def.name) + v * 977)).build();
      const mesh = new THREE.InstancedMesh(geometry, def.thin ? this.thin : this.solid, perVariant);
      mesh.name = `reef-${def.name}-${region}-${v}`;
      mesh.castShadow = false;
      mesh.receiveShadow = true;

      // Placements first, then shuffled so a count prefix samples every site.
      const placements: { x: number; z: number }[] = [];
      let colonyLeft = 0;
      let colony = { x: 0, z: 0 };
      let siteIndex = Math.floor(random() * sites.length);
      for (let i = 0; i < perVariant; i++) {
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
        placements.push({ x: colony.x + Math.cos(a) * r, z: colony.z + Math.sin(a) * r });
      }
      for (let i = placements.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const t = placements[i];
        placements[i] = placements[j];
        placements[j] = t;
      }

      for (let i = 0; i < placements.length; i++) {
        const { x, z } = placements[i];
        const s = def.scale[0] + (def.scale[1] - def.scale[0]) * Math.pow(random(), 1.6);
        const squash = def.squash ? def.squash[0] + (def.squash[1] - def.squash[0]) * random() : 1;
        floorNormal(x, z, normal);
        tilt.setFromUnitVectors(up, normal.clone().lerp(up, 1 - def.followSlope).normalize());
        const lean = new THREE.Quaternion().setFromEuler(
          new THREE.Euler((random() - 0.5) * 2 * def.lean, 0, (random() - 0.5) * 2 * def.lean),
        );
        const bearing = def.faceCurrent
          ? this.fanBearing + (random() - 0.5) * 0.7
          : random() * Math.PI * 2;
        yaw.setFromAxisAngle(up, bearing);
        quaternion.copy(tilt).multiply(lean).multiply(yaw);
        position.set(x, seafloorHeight(x, z) - def.sink * s, z);
        scale.set(s * squash, s, s * (def.squash ? def.squash[0] + (def.squash[1] - def.squash[0]) * random() : 1));
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
        const hex = def.palette[Math.floor(random() * def.palette.length)];
        color.setHex(hex);
        // Per-instance brightness jitter, so a colony is not one swatch.
        const k = 0.82 + random() * 0.36;
        color.multiplyScalar(k);
        mesh.setColorAt(i, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingSphere();
      // The sway can carry a tip a few decimetres past the rest pose.
      if (mesh.boundingSphere) mesh.boundingSphere.radius += 1;
      this.object.add(mesh);
      this.kinds.push({ def, region, mesh, capacity: perVariant });
    }
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

/** A circle (centre of the extent, conservative radius) around every site. */
function boundingCircle(sites: Site[]): Site | null {
  if (sites.length === 0) return null;
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const s of sites) {
    minX = Math.min(minX, s.x - s.radius);
    maxX = Math.max(maxX, s.x + s.radius);
    minZ = Math.min(minZ, s.z - s.radius);
    maxZ = Math.max(maxZ, s.z + s.radius);
  }
  const x = (minX + maxX) / 2;
  const z = (minZ + maxZ) / 2;
  return { x, z, radius: Math.hypot(maxX - x, maxZ - z) };
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
