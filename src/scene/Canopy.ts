import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  float,
  mix,
  mx_noise_float,
  normalize,
  positionGeometry,
  rtt,
  texture,
  uniform,
  uniformArray,
  uv,
  vec2,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { ISLAND, seafloorHeight } from './Seafloor';
import { inLighthouseClearing } from './props/site';
import { TRANSLUCENCY_GAIN, TRANSLUCENCY_POWER } from './groundShading';
import { TILES, getLeafAtlas, tileUv } from './flora/leafAtlas';

/**
 * The island's forest canopy: thousands of small, solid trees.
 *
 * **What this replaced, and why.** The forest used to be camera-facing cards, a
 * blob with a ragged edge drawn in the shader from each card's seed. From the
 * play area, 1.4 km off, that is the textbook answer: a ten-metre tree is eight
 * pixels and a card is the cheapest silhouette there is. It stopped being the
 * answer the moment anyone flew in. Inside a few hundred metres a card is a flat
 * sheet that turns to follow you, and a hillside of them read as what a viewer
 * called them: green clipboards. The fade that hid them inside 90 m left the
 * near hillside bald, and no amount of edge noise gives a flat sheet parallax.
 *
 * **What it is now.** Each instance is a tree: a tapered trunk and a crown of
 * five to eight overlapping puffs, the lowest of them understorey, each a subdivided icosahedron pushed in and
 * out by noise, so the crown has lumps and gaps in its outline from every side.
 * It is the construction stylised open-world forests use at mid and far range
 * (the clustered, lumpy crowns of Breath of the Wild's distant woods, the
 * painterly clumps of Sea of Thieves' islands, and the "Ghibli tree" technique
 * of foliage lit as one volume), and it holds up at any distance because it is
 * real geometry.
 *
 * **Lit as a volume, not as lumps.** Shading each puff by its own normal gives
 * every lump its own terminator, and a hillside of those reads as broccoli. So
 * the normal is pulled toward the normal of an ellipsoid around the whole crown,
 * and the lumps survive as silhouette rather than as a dozen small suns. The same trick is why Sea of Thieves' and every
 * "Ghibli" tree reads soft.
 *
 * **A leaf shell over a solid.** Near and in the middle distance each puff is
 * wrapped in cards of the atlas's painted leaf-clump tile, alpha-tested, so the
 * outline is leaves and the surface breaks into clumps; the solid inside fills
 * the gaps as shadowed depth. Far off, the solid alone is the shell's envelope.
 *
 * **Cost.** One draw per level of detail, three for the forest: the six crown
 * shapes are a table the vertex stage reads, not meshes (see `ShapeTable`).
 * Each tree's level is chosen on the CPU by distance and redealt only when the
 * camera has moved. Placement, stand tone and the hill's own shadow are
 * computed on the CPU from `seafloorHeight`, so the vertex stage does no
 * heightfield taps at all, and the cloud deck's shade is one lookup in a small
 * map rather than a march per vertex. Measured with GPU timestamps at 1600x900
 * against the cards: WebGPU High within 0.8 ms in every view, and WebGL2 Low
 * cheaper in most.
 *
 * Nothing accumulates: every visible quantity is a function of the seed, the
 * camera position and one phase uniform, so `resetClock(t)` reproduces a frame
 * exactly, which the visual regression harness requires.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;

const CANOPY_SEED = 0x4f2b91;

/**
 * Candidate trees. A quality tier's `canopy` count is how many of these it
 * considers; the terrain then rejects the ones off the forest band, as it
 * collapsed the cards it rejected. Same semantics, so the tier budgets carry
 * over unchanged.
 */
const MAX_CANDIDATES = 24_000;

/** Nominal tree height before per-instance scale, metres. */
const TREE_HEIGHT = 12;
/** Per-instance size spread, as a fraction either side of nominal. */
const SIZE_VARIATION = 0.32;
/** Horizontal stretch of every crown over its shape's nominal width. */
const CROWN_SPREAD = 1.3;
/** Width against height, varied per instance, area-preserving. */
const ASPECT_VARIATION = 0.22;

/**
 * Distance bands for the three levels of detail, metres from the camera. Where
 * a puff of three to five metres stops showing its facets and its leaves: the
 * near level is a full leaf shell over a 42-vertex solid, the middle one a
 * sparse shell out to 450 m, and past that a puff is a few pixels and a
 * 12-vertex icosahedron is as round as it needs to be.
 */
const LOD_NEAR = 200;
const LOD_FAR = 450;
/**
 * The bands shrink below High, by the tier's own candidate count: a weaker GPU
 * gets its full leaf shells nearer the camera. The far band shrinks less than
 * the near one: a lower tier plants fewer trees, and a bare solid is the one
 * level that reads as a smooth blob rather than as foliage.
 */
function lodBands(candidates: number): [number, number] {
  if (candidates >= 15_000) return [LOD_NEAR, LOD_FAR];
  if (candidates >= 9_000) return [LOD_NEAR * 0.75, LOD_FAR * 0.9];
  return [LOD_NEAR * 0.55, LOD_FAR * 0.8];
}
/** Camera travel that triggers a redeal, metres. */
const REDEAL_DISTANCE = 6;

/**
 * Elevation band, matching `Seafloor`'s vegetation ramp and treeline. Derived
 * from `ISLAND.peak` rather than typed, as `IslandMeadow`'s is.
 */
const CANOPY_MIN_HEIGHT = 6;
const CANOPY_FULL_HEIGHT = 21;
const CANOPY_FADE_HEIGHT = ISLAND.peak + 60;
const CANOPY_MAX_HEIGHT = ISLAND.peak + 90;
/** How far past the shore radius trees may stand, in island radii. */
const ISLAND_REACH = 1.22;
/** Slope sample spacing for the soil test and the sun-exposure bias, metres. */
const SLOPE_SAMPLE_METRES = 4;

/** Canopy colours. The same family as `Seafloor`'s ground biome. */
const CANOPY_SUNLIT = new THREE.Color(0.1, 0.2, 0.04);
const CANOPY_SHADE = new THREE.Color(0.022, 0.062, 0.02);
/** Dry, sun-bleached stands on exposed spurs. A different stand, not a biome. */
const CANOPY_DRY = new THREE.Color(0.11, 0.135, 0.035);
const BARK = new THREE.Color(0.06, 0.045, 0.032);

/** Stand variation: field width, the weights of field and crown, value spread. */
const STAND_FIELD_METRES = 150;
const STAND_FIELD_WEIGHT = 0.62;
const STAND_INSTANCE_WEIGHT = 0.38;
const STAND_SLOPE_WEIGHT = 0.25;
const STAND_VALUE_RANGE = 0.36;

/**
 * How far a crown's shading normal is pulled toward the crown's ellipsoid.
 * Enough that the crown has one terminator; not so much that its lumps go flat.
 */
const VOLUME_NORMAL = 0.62;
/** Wrap on the key light: foliage scatters, so its terminator is soft. */
const CANOPY_WRAP = 0.6;
/** The range the interior's leaf-clump cells fade out over, metres. */
const LEAF_FADE_NEAR = 70;
const LEAF_FADE_FAR = 260;

/** Sway: metres of crown travel at full wind, and the wave that carries it. */
const SWAY_WAVELENGTH = 90;
const SWAY_K = (Math.PI * 2) / SWAY_WAVELENGTH;
const SWAY_SPEED = 7;
const SWAY_OMEGA = SWAY_K * SWAY_SPEED;
// The same as the planted meshes' (`groundShading`), so a tree near the camera
// and the forest behind it move alike.
const SWAY_MAX = 0.9;

const CLOCK_WRAP = 3600;

/**
 * The cloud deck's shade over the island, as a map: texels a side, metres from
 * the island's centre to the map's edge, and the height it is evaluated at.
 *
 * The deck's shadow is a three-sample march through a fractal field, about a
 * dozen noise evaluations; it was per vertex, and a forest of leaf-shelled
 * crowns is millions of vertices. Clouds are hundreds of metres across, so a
 * 12.5 m texel is far finer than the field, and one 128-square pass a frame is
 * sixteen thousand evaluations instead.
 */
const CLOUD_MAP_SIZE = 128;
const CLOUD_MAP_HALF = 800;
const CLOUD_MAP_HEIGHT = 70;
/** Frames between redraws of the cloud map. */
const CLOUD_MAP_INTERVAL = 4;

// ---------------------------------------------------------------- geometry

interface Puff {
  /** Centre in tree space, metres; y up from the foot. */
  c: readonly [number, number, number];
  /** Radii, metres. */
  r: readonly [number, number, number];
}

interface CrownShape {
  puffs: Puff[];
  /** Top of the trunk, metres. */
  trunkTop: number;
  trunkRadius: number;
}

/**
 * Six crown shapes. Every one is reused under a random yaw, a mirror and a
 * non-uniform scale, so the eye meets the same outline rarely and never twice
 * at the same angle.
 */
const CROWNS: CrownShape[] = [
  // Broadleaf dome: a main mass, three shoulders, a cap.
  {
    puffs: [
      { c: [0, 7.8, 0], r: [4.4, 3.2, 4.4] },
      { c: [3.0, 6.6, 0.6], r: [2.6, 2.2, 2.6] },
      { c: [-1.8, 6.9, 2.6], r: [2.5, 2.1, 2.5] },
      { c: [-1.4, 6.8, -2.7], r: [2.6, 2.2, 2.6] },
      { c: [0.6, 10.1, -0.4], r: [2.5, 1.9, 2.5] },
      { c: [1.6, 3.2, -1.2], r: [2.4, 1.9, 2.4] },
    ],
    trunkTop: 6.5,
    trunkRadius: 0.36,
  },
  // Columnar: a tall stack narrowing upward.
  {
    puffs: [
      { c: [0, 6.0, 0], r: [3.2, 2.6, 3.2] },
      { c: [0.4, 8.6, -0.3], r: [2.6, 2.3, 2.6] },
      { c: [-0.3, 10.8, 0.2], r: [1.9, 1.7, 1.9] },
      { c: [1.8, 7.2, 1.2], r: [1.7, 1.6, 1.7] },
      { c: [-1.4, 3.0, 0.8], r: [2.2, 1.8, 2.2] },
    ],
    trunkTop: 5.0,
    trunkRadius: 0.3,
  },
  // Rain tree: a wide, flat-topped umbrella on a short trunk.
  {
    puffs: [
      { c: [0, 8.4, 0], r: [5.6, 2.0, 5.6] },
      { c: [4.2, 8.0, 1.6], r: [2.4, 1.6, 2.4] },
      { c: [-3.6, 8.2, 2.6], r: [2.3, 1.6, 2.3] },
      { c: [-1.2, 8.1, -4.3], r: [2.4, 1.6, 2.4] },
      { c: [2.0, 9.6, -1.6], r: [2.2, 1.3, 2.2] },
      { c: [-1.0, 9.8, 1.2], r: [2.3, 1.3, 2.3] },
      { c: [0.8, 3.4, 1.4], r: [2.6, 2.0, 2.6] },
      { c: [-1.8, 4.6, -1.0], r: [2.0, 1.7, 2.0] },
    ],
    trunkTop: 7.6,
    trunkRadius: 0.44,
  },
  // Two-lobed and leaning: the irregular crown of a tree that lost a limb.
  {
    puffs: [
      { c: [-1.9, 7.4, 0.2], r: [3.2, 2.8, 3.2] },
      { c: [2.1, 8.4, -0.4], r: [2.9, 2.5, 2.9] },
      { c: [0.4, 10.0, 0.6], r: [2.2, 1.8, 2.2] },
      { c: [0.2, 6.6, 2.3], r: [1.9, 1.7, 1.9] },
      { c: [1.2, 3.0, -1.4], r: [2.3, 1.9, 2.3] },
    ],
    trunkTop: 6.4,
    trunkRadius: 0.34,
  },
  // Low, broad and dense: understorey and windward edge.
  {
    puffs: [
      { c: [0, 5.2, 0], r: [4.0, 2.6, 4.0] },
      { c: [2.6, 4.6, -1.8], r: [2.3, 1.9, 2.3] },
      { c: [-2.4, 4.8, -1.4], r: [2.4, 2.0, 2.4] },
      { c: [0.2, 4.4, 2.8], r: [2.3, 1.9, 2.3] },
      { c: [0.4, 7.0, 0.3], r: [2.4, 1.9, 2.4] },
    ],
    trunkTop: 3.8,
    trunkRadius: 0.32,
  },
  // Emergent: a tall bare trunk with its crown held high over the rest.
  {
    puffs: [
      { c: [0, 10.4, 0], r: [3.6, 2.4, 3.6] },
      { c: [2.4, 9.6, 0.8], r: [2.2, 1.8, 2.2] },
      { c: [-2.0, 9.8, -1.2], r: [2.3, 1.8, 2.3] },
      { c: [0.3, 12.0, 0.4], r: [2.0, 1.5, 2.0] },
      { c: [1.4, 3.6, 1.0], r: [2.6, 2.1, 2.6] },
      { c: [-1.6, 5.4, -0.6], r: [2.2, 1.8, 2.2] },
    ],
    trunkTop: 9.6,
    trunkRadius: 0.38,
  },
];

/** Indexed icosphere on the unit sphere; detail 0, 1, 2 give 12, 42, 162 vertices. */
function icosphere(detail: number): { positions: number[]; indices: number[] } {
  const t = (1 + Math.sqrt(5)) / 2;
  const positions: number[] = [];
  const push = (x: number, y: number, z: number): number => {
    const l = Math.hypot(x, y, z);
    positions.push(x / l, y / l, z / l);
    return positions.length / 3 - 1;
  };
  for (const [x, y, z] of [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ]) {
    push(x, y, z);
  }
  let indices = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];
  for (let d = 0; d < detail; d++) {
    const cache = new Map<number, number>();
    const mid = (a: number, b: number): number => {
      const key = a < b ? a * 65536 + b : b * 65536 + a;
      const hit = cache.get(key);
      if (hit !== undefined) return hit;
      const i = push(
        positions[a * 3] + positions[b * 3],
        positions[a * 3 + 1] + positions[b * 3 + 1],
        positions[a * 3 + 2] + positions[b * 3 + 2],
      );
      cache.set(key, i);
      return i;
    };
    const next: number[] = [];
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i];
      const b = indices[i + 1];
      const c = indices[i + 2];
      const ab = mid(a, b);
      const bc = mid(b, c);
      const ca = mid(c, a);
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    indices = next;
  }
  return { positions, indices };
}

/** Lattice hash to [0, 1). Integer arithmetic, so it is the same everywhere. */
function hash3(x: number, y: number, z: number, seed: number): number {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647 + seed * 144665) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Smooth 3D value noise in [0, 1). */
function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const fx = x - xi;
  const fy = y - yi;
  const fz = z - zi;
  const u = fx * fx * (3 - 2 * fx);
  const v = fy * fy * (3 - 2 * fy);
  const w = fz * fz * (3 - 2 * fz);
  const lerp = (a: number, b: number, k: number): number => a + (b - a) * k;
  const c = (i: number, j: number, k: number): number => hash3(xi + i, yi + j, zi + k, seed);
  return lerp(
    lerp(lerp(c(0, 0, 0), c(1, 0, 0), u), lerp(c(0, 1, 0), c(1, 1, 0), u), v),
    lerp(lerp(c(0, 0, 1), c(1, 0, 1), u), lerp(c(0, 1, 1), c(1, 1, 1), u), v),
    w,
  );
}

/** Two octaves, centred on zero, roughly [-1, 1]. */
function lumpNoise(x: number, y: number, z: number, seed: number): number {
  return (
    (valueNoise3(x, y, z, seed) - 0.5) * 2 * 0.7 +
    (valueNoise3(x * 2.3, y * 2.3, z * 2.3, seed + 17) - 0.5) * 2 * 0.3
  );
}

// ------------------------------------------------------------ shape table

/**
 * Puff slots per crown. Every shape has at most this many puffs; a shape with
 * fewer leaves the rest of its slots empty and their vertices collapse.
 */
const SLOTS = 8;

/** Vertex parts: a puff's solid, a leaf card, the trunk. */
const PART_SOLID = 0;
const PART_CARD = 1;
const PART_TRUNK = 2;

/**
 * Every shape's puffs, as data the vertex stage reads.
 *
 * **Why the shapes are data and not meshes.** Six shapes at three levels of
 * detail as their own meshes were eighteen draws, each repeated in every pass
 * that draws the scene, and measured at about a millisecond of CPU a frame over
 * the forest, against one draw for the cards they replaced. So each level of
 * detail is one mesh: every puff slot is a unit sphere, and a tree's vertex
 * stage looks up its own shape's centre and radii for the slot and places the
 * sphere there. Three draws for the whole forest. A shape with fewer puffs than
 * slots collapses the rest to a point, which costs their vertices and nothing
 * else.
 */
interface ShapeTable {
  /** Per shape and slot: centre xyz, and 1 if the slot holds a puff. */
  centres: THREE.Vector4[];
  /** Per shape and slot: radii xyz, and the leaf-card half size for the slot. */
  radii: THREE.Vector4[];
  /** Per shape: the crown envelope's bottom, top and reach, and the trunk's top. */
  envelopes: THREE.Vector4[];
  /** Per shape: the trunk's radius. */
  trunks: THREE.Vector4[];
}

/** Cards per puff in the near shell: the vertices of a once-subdivided icosahedron. */
const CARDS_FULL = 42;

function buildShapeTable(): ShapeTable {
  const centres: THREE.Vector4[] = [];
  const radii: THREE.Vector4[] = [];
  const envelopes: THREE.Vector4[] = [];
  const trunks: THREE.Vector4[] = [];
  for (const shape of CROWNS) {
    if (shape.puffs.length > SLOTS) throw new Error('canopy: a crown has more puffs than slots');
    let minY = Infinity;
    let maxY = -Infinity;
    let maxR = 0;
    for (const p of shape.puffs) {
      minY = Math.min(minY, p.c[1] - p.r[1]);
      maxY = Math.max(maxY, p.c[1] + p.r[1]);
      maxR = Math.max(maxR, Math.hypot(p.c[0], p.c[2]) + Math.max(p.r[0], p.r[2]));
    }
    envelopes.push(new THREE.Vector4(minY, maxY, maxR, shape.trunkTop));
    trunks.push(new THREE.Vector4(shape.trunkRadius, 0, 0, 0));
    for (let k = 0; k < SLOTS; k++) {
      const puff = shape.puffs[k];
      if (!puff) {
        centres.push(new THREE.Vector4(0, 0, 0, 0));
        radii.push(new THREE.Vector4(0, 0, 0, 0));
        continue;
      }
      // A card's half size from the puff's surface area over the cards it gets,
      // so a wide, flat puff gets cards as large as its area needs rather than
      // as large as its mean radius says (Thomsen's ellipsoid approximation).
      const [a, b, c] = puff.r;
      const p = 1.6075;
      const area =
        4 * Math.PI * Math.pow((Math.pow(a * b, p) + Math.pow(a * c, p) + Math.pow(b * c, p)) / 3, 1 / p);
      centres.push(new THREE.Vector4(puff.c[0], puff.c[1], puff.c[2], 1));
      radii.push(new THREE.Vector4(a, b, c, Math.sqrt(area / CARDS_FULL)));
    }
  }
  return { centres, radii, envelopes, trunks };
}

// ------------------------------------------------------------ geometry

/**
 * A slot's surface along unit direction n, on the unit sphere: pushed in and
 * out by lumps about a sixth of the sphere across and a quarter of it deep,
 * and flattened underneath, where a crown is shaded and smooth rather than
 * tufted. Scaled by the puff's radii in the vertex stage, so the lumps are a
 * fixed fraction of whatever puff the slot holds.
 *
 * Every level of detail, and the leaf cards, sample the same function, so the
 * coarser levels are the finer ones sampled more sparsely and a level change
 * moves a facet rather than a lump.
 */
function slotSurface(slot: number, nx: number, ny: number, nz: number): [number, number, number] {
  const lump = lumpNoise(nx * 1.6 + slot * 7.3, ny * 1.6, nz * 1.6, 0x51ed);
  const k = 1 + lump * 0.26 * (ny < -0.3 ? 0.5 : 1);
  return [nx * k, ny * k, nz * k];
}

/** Attribute arrays for one level's geometry, filled part by part. */
class GeometryBuilder {
  readonly positions: number[] = [];
  readonly normals: number[] = [];
  /** Card corner offset xyz and size, or the trunk's radius in w. */
  readonly offsets: number[] = [];
  /** Slot, part, interior flag, spare. */
  readonly parts: number[] = [];
  readonly uvs: number[] = [];
  readonly indices: number[] = [];

  get count(): number {
    return this.positions.length / 3;
  }

  vertex(
    p: readonly number[],
    n: readonly number[],
    offset: readonly number[],
    slot: number,
    part: number,
    interior: number,
    u = 0,
    v = 0,
  ): void {
    this.positions.push(p[0], p[1], p[2]);
    this.normals.push(n[0], n[1], n[2]);
    this.offsets.push(offset[0], offset[1], offset[2], offset[3]);
    this.parts.push(slot, part, interior, 0);
    this.uvs.push(u, v);
  }

  build(): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
    geometry.setAttribute('crownOffset', new THREE.Float32BufferAttribute(this.offsets, 4));
    geometry.setAttribute('crownPart', new THREE.Float32BufferAttribute(this.parts, 4));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2));
    geometry.setIndex(this.indices);
    return geometry;
  }
}

/** Every slot's lumpy unit sphere, with its own smooth normals. */
function addSolids(out: GeometryBuilder, detail: number, interior: number): void {
  const sphere = icosphere(detail);
  for (let slot = 0; slot < SLOTS; slot++) {
    const surface: number[] = [];
    for (let i = 0; i < sphere.positions.length; i += 3) {
      surface.push(...slotSurface(slot, sphere.positions[i], sphere.positions[i + 1], sphere.positions[i + 2]));
    }
    const shell = new THREE.BufferGeometry();
    shell.setAttribute('position', new THREE.Float32BufferAttribute(surface, 3));
    shell.setIndex(sphere.indices);
    shell.computeVertexNormals();
    const normals = shell.getAttribute('normal');
    const base = out.count;
    for (let v = 0; v < surface.length / 3; v++) {
      out.vertex(
        [surface[v * 3], surface[v * 3 + 1], surface[v * 3 + 2]],
        [normals.getX(v), normals.getY(v), normals.getZ(v)],
        [0, 0, 0, 0],
        slot,
        PART_SOLID,
        interior,
      );
    }
    for (const index of sphere.indices) out.indices.push(base + index);
    shell.dispose();
  }
}

/**
 * The leaf shell: a card of the atlas's leaf-clump tile at points over every
 * slot's sphere, standing a little proud of the surface and facing out along
 * it, twisted and tilted at random.
 *
 * This is what stops a crown reading as a smooth solid. The solid gives the
 * mass and fills the gaps; the cards give the outline its leaves and break the
 * surface into clumps. Lit by the crown's volume normal, not the card's own, so
 * a shell of a hundred cards shades as one crown (the "Ghibli tree" technique,
 * and how Sea of Thieves' foliage reads soft).
 *
 * `full`: a card at each of the 42 vertices of a once-subdivided icosahedron
 * per puff, so the shell closes and the solid shows only as shadowed depth
 * between cards. Otherwise at the twelve corners of the plain one, for the
 * middle distance, where a gap is a pixel or two, with fewer underneath.
 *
 * Both faces as two windings rather than a double-sided material, so the solid
 * drawn with them keeps its back faces culled.
 */
function addCards(out: GeometryBuilder, full: boolean): void {
  const ico = icosphere(full ? 1 : 0);
  const uv = tileUv('clump');
  // Half size in units of the slot's card spacing (see `ShapeTable.radii`),
  // which is for the full set: the sparse set covers the same area with fewer.
  const scale = full ? 0.78 : 0.9 * Math.sqrt(CARDS_FULL / 12);
  const n = new THREE.Vector3();
  const t1 = new THREE.Vector3();
  const t2 = new THREE.Vector3();
  for (let slot = 0; slot < SLOTS; slot++) {
    const random = mulberry32(0x1eaf + slot * 977);
    for (let i = 0; i < ico.positions.length; i += 3) {
      const dx = ico.positions[i];
      const dy = ico.positions[i + 1];
      const dz = ico.positions[i + 2];
      if (!full && dy < -0.6 && random() < 0.5) continue;
      const [sx, sy, sz] = slotSurface(slot, dx, dy, dz);
      const centre = [sx * 1.04, sy * 1.04, sz * 1.04];
      // Facing: the surface direction, tilted up to about 35 degrees at random.
      n.set(dx + (random() - 0.5) * 1.2, dy + (random() - 0.5) * 1.2, dz + (random() - 0.5) * 1.2).normalize();
      t1.set(0, 1, 0);
      if (Math.abs(n.y) > 0.9) t1.set(1, 0, 0);
      t1.cross(n).normalize();
      t2.copy(n).cross(t1).normalize();
      const twist = random() * Math.PI * 2;
      const c = Math.cos(twist);
      const s = Math.sin(twist);
      const size = scale * (0.8 + random() * 0.4);
      const base = out.count;
      const corners: Array<[number, number, number, number]> = [
        [-1, -1, uv.u0, uv.vBot],
        [1, -1, uv.u1, uv.vBot],
        [1, 1, uv.u1, uv.vTop],
        [-1, 1, uv.u0, uv.vTop],
      ];
      for (const [a, b, u, v] of corners) {
        const ra = a * c - b * s;
        const rb = a * s + b * c;
        out.vertex(
          centre,
          [n.x, n.y, n.z],
          [t1.x * ra + t2.x * rb, t1.y * ra + t2.y * rb, t1.z * ra + t2.z * rb, size],
          slot,
          PART_CARD,
          0,
          u,
          v,
        );
      }
      out.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      out.indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
    }
  }
}

/**
 * The trunk: a six-sided taper from below the ground into the crown, with two
 * short limbs. Positions are on the axis, in units of the shape's trunk top;
 * the normal is the radial direction and the offset's w the radius, in units
 * of the shape's trunk radius, so one trunk fits every shape. Seen only at the
 * forest's edge from inside a few hundred metres, so six sides and no cap.
 */
function addTrunk(out: GeometryBuilder): void {
  const sides = 6;
  // Axis endpoints in trunk-top units, and radii in trunk-radius units.
  const segments: Array<[number, number, number, number, number, number, number, number]> = [
    [0, -0.1, 0, 1.35, 0, 1, 0, 0.6],
    [0, 0.72, 0, 0.55, 0.25, 1.14, 0.08, 0.3],
    [0, 0.8, 0, 0.5, -0.19, 1.15, -0.17, 0.28],
  ];
  for (const [x0, y0, z0, r0, x1, y1, z1, r1] of segments) {
    const base = out.count;
    for (let ring = 0; ring < 2; ring++) {
      const [cx, cy, cz, r] = ring === 0 ? [x0, y0, z0, r0] : [x1, y1, z1, r1];
      for (let k = 0; k < sides; k++) {
        const a = (k / sides) * Math.PI * 2;
        out.vertex([cx, cy, cz], [Math.cos(a), 0, Math.sin(a)], [0, 0, 0, r], 0, PART_TRUNK, 0);
      }
    }
    for (let k = 0; k < sides; k++) {
      const a = base + k;
      const b = base + ((k + 1) % sides);
      out.indices.push(a, a + sides, b, b, a + sides, b + sides);
    }
  }
}

/**
 * One level of detail's geometry. Near: a 42-vertex solid per slot inside a
 * full leaf shell, and the trunk. Middle: the same solid nearly at full size
 * under a sparse shell. Far: the 12-vertex solid alone, which is the shell's
 * envelope at a few pixels.
 */
function buildLevel(lod: number): THREE.BufferGeometry {
  const out = new GeometryBuilder();
  if (lod === 0) {
    addSolids(out, 1, 1);
    addCards(out, true);
    addTrunk(out);
  } else if (lod === 1) {
    addSolids(out, 1, 0);
    addCards(out, false);
    addTrunk(out);
  } else {
    addSolids(out, 0, 0);
  }
  return out.build();
}

/** How far each level's solid is pulled inside its shell, toward the puff's centre. */
const SOLID_SHRINK = [0.86, 0.95, 1];

/**
 * Mean linear luminance of the leaf-clump tile where it is opaque, so the
 * cards can take the canopy's palette and keep only the tile's light and dark.
 */
function clumpLuminance(atlas: THREE.DataTexture): number {
  const image = atlas.image as { data: ArrayLike<number>; width: number };
  const rect = TILES.clump;
  const toLinear = (v: number): number => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  let sum = 0;
  let count = 0;
  for (let y = rect.y; y < rect.y + rect.h; y += 2) {
    for (let x = rect.x; x < rect.x + rect.w; x += 2) {
      const i = (y * image.width + x) * 4;
      if (image.data[i + 3] < 128) continue;
      sum +=
        0.2126 * toLinear(image.data[i]) +
        0.7152 * toLinear(image.data[i + 1]) +
        0.0722 * toLinear(image.data[i + 2]);
      count++;
    }
  }
  return count > 0 ? sum / count : 0.1;
}

// ---------------------------------------------------------------- placement

interface Forest {
  /** Accepted trees, in candidate order. */
  count: number;
  /** Index of the candidate each tree came from, for `setCount`'s prefix. */
  candidate: Int32Array;
  shape: Uint8Array;
  /** x, ground y, z, yaw. */
  a: Float32Array;
  /** Scale x, y, z, sway phase. */
  b: Float32Array;
  /** Stand tone, hill shadow (filled by `ShadowBaker`), crown shape, tint. */
  c: Float32Array;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * Plants the forest on the CPU, once. The rules are the cards' rules, which were
 * `Seafloor`'s: the vegetation band by elevation, the island's reach, and no
 * trees on ground too steep to hold soil, which is where the terrain paints
 * rock. Presence decides whether a candidate becomes a tree and, at the band's
 * soft edges, how big.
 */
function plantForest(seed: number, sunDir: THREE.Vector3): Forest {
  const random = mulberry32(seed);
  const candidate = new Int32Array(MAX_CANDIDATES);
  const shape = new Uint8Array(MAX_CANDIDATES);
  const a = new Float32Array(MAX_CANDIDATES * 4);
  const b = new Float32Array(MAX_CANDIDATES * 4);
  const c = new Float32Array(MAX_CANDIDATES * 4);
  let count = 0;
  for (let i = 0; i < MAX_CANDIDATES; i++) {
    // Drawn whether or not the candidate is kept, so each candidate's draws are
    // the same at every tier and a tier change adds trees without moving any.
    const angle = random() * Math.PI * 2;
    const radius = Math.sqrt(random()) * ISLAND.radius * ISLAND_REACH;
    const keep = random();
    const size = random();
    const aspectDraw = random();
    const yaw = random() * Math.PI * 2;
    const phase = random();
    const crown = random();
    const shapeDraw = random();
    const mirror = random() < 0.5 ? -1 : 1;

    const x = ISLAND.x + Math.cos(angle) * radius;
    const z = ISLAND.z + Math.sin(angle) * radius;
    const ground = seafloorHeight(x, z);
    const band =
      smoothstep(CANOPY_MIN_HEIGHT, CANOPY_FULL_HEIGHT, ground) *
      (1 - smoothstep(CANOPY_FADE_HEIGHT, CANOPY_MAX_HEIGHT, ground));
    if (band <= 0) continue;
    const fromIsland = Math.hypot(x - ISLAND.x, z - ISLAND.z);
    const onIsland =
      1 - smoothstep(ISLAND.radius * 0.98, ISLAND.radius * ISLAND_REACH, fromIsland);
    const east = seafloorHeight(x + SLOPE_SAMPLE_METRES, z);
    const north = seafloorHeight(x, z + SLOPE_SAMPLE_METRES);
    const n = new THREE.Vector3(ground - east, SLOPE_SAMPLE_METRES, ground - north).normalize();
    const holdsSoil = smoothstep(0.8, 0.88, n.y);
    const presence = band * onIsland * holdsSoil;
    if (presence <= 0 || keep > Math.sqrt(presence)) continue;
    // After every draw, so clearing the site moves no other tree. A crown is
    // up to ~6 m across, hence the margin past the keep-out.
    if (inLighthouseClearing(x, z, 4)) continue;

    const scale = (1 - SIZE_VARIATION + size * SIZE_VARIATION * 2) * (0.7 + 0.3 * presence);
    const aspect = 1 + (aspectDraw - 0.5) * 2 * ASPECT_VARIATION;
    const s = count * 4;
    // Sunk a little, so a tree on a slope has no daylight under its downhill
    // side. The trunk runs 0.6 m below its own origin for the same reason.
    a[s] = x;
    a[s + 1] = ground - 0.4;
    a[s + 2] = z;
    a[s + 3] = yaw;
    // Wider than tall against the shapes' nominal proportions: a tropical
    // canopy closes overhead, and at the shapes' own width the forest read as
    // an orchard, a trunk between every pair of crowns.
    b[s] = scale * aspect * CROWN_SPREAD * mirror;
    b[s + 1] = scale / aspect;
    b[s + 2] = scale * aspect * CROWN_SPREAD;
    b[s + 3] = phase;

    // Stand tone: a broad field across the island plus a per-crown offset, with
    // sun-facing slopes biased dry. Value noise rather than the shader's
    // fractal noise; nothing else has to agree with it.
    const field =
      0.6 * valueNoise3(x / STAND_FIELD_METRES, 0, z / STAND_FIELD_METRES, 7) +
      0.4 * valueNoise3((x * 2) / STAND_FIELD_METRES, 0.5, (z * 2) / STAND_FIELD_METRES, 11);
    const exposure = Math.max(0, n.dot(sunDir));
    const stand =
      (field * STAND_FIELD_WEIGHT + crown * STAND_INSTANCE_WEIGHT) * (1 - STAND_SLOPE_WEIGHT) +
      exposure * STAND_SLOPE_WEIGHT;
    shape[count] = Math.min(CROWNS.length - 1, Math.floor(shapeDraw * CROWNS.length));
    c[s] = Math.min(1, Math.max(0, stand));
    c[s + 1] = 1;
    c[s + 2] = shape[count];
    c[s + 3] = random();
    candidate[count] = i;
    count++;
  }
  return { count, candidate, shape, a, b, c };
}

/**
 * The island's shadow on each tree, for one key-light direction, written into
 * `forest.c[i*4+1]`.
 *
 * The same soft-shadow march `Seafloor` runs per pixel (iq's: the closest
 * approach of the ray to the ground, as an angle, is the penumbra), run once per
 * tree on the CPU over a cached height grid. Per tree because a crown is a few
 * metres and the hill's shadow is hundreds; on the CPU because the vertex stage
 * would otherwise run it for every vertex of every crown.
 */
class ShadowBaker {
  private static readonly CELL = 4;
  private static readonly HALF = 900;
  private static readonly N = Math.round((2 * ShadowBaker.HALF) / ShadowBaker.CELL) + 1;
  private grid: Float32Array | null = null;

  private height(x: number, z: number): number {
    const grid = this.ensureGrid();
    const gx = (x - ISLAND.x + ShadowBaker.HALF) / ShadowBaker.CELL;
    const gz = (z - ISLAND.z + ShadowBaker.HALF) / ShadowBaker.CELL;
    const n = ShadowBaker.N;
    if (gx < 0 || gz < 0 || gx >= n - 1 || gz >= n - 1) return seafloorHeight(x, z);
    const ix = Math.floor(gx);
    const iz = Math.floor(gz);
    const fx = gx - ix;
    const fz = gz - iz;
    const i = iz * n + ix;
    const top = grid[i] + (grid[i + 1] - grid[i]) * fx;
    const bottom = grid[i + n] + (grid[i + n + 1] - grid[i + n]) * fx;
    return top + (bottom - top) * fz;
  }

  private ensureGrid(): Float32Array {
    if (this.grid) return this.grid;
    const n = ShadowBaker.N;
    const grid = new Float32Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        grid[j * n + i] = seafloorHeight(
          ISLAND.x - ShadowBaker.HALF + i * ShadowBaker.CELL,
          ISLAND.z - ShadowBaker.HALF + j * ShadowBaker.CELL,
        );
      }
    }
    this.grid = grid;
    return grid;
  }

  bake(forest: Forest, dir: THREE.Vector3): void {
    for (let t = 0; t < forest.count; t++) {
      const s = t * 4;
      forest.c[s + 1] = this.march(
        forest.a[s],
        forest.a[s + 1] + TREE_HEIGHT * 0.6 * forest.b[s + 1],
        forest.a[s + 2],
        dir,
      );
    }
  }

  /** Constants as `Seafloor`'s march, so a tree and the ground under it agree. */
  private march(px: number, py: number, pz: number, dir: THREE.Vector3): number {
    if (dir.y <= 0.02) return 1;
    let res = 1;
    let t = 3;
    for (let i = 0; i < 32; i++) {
      const sx = px + dir.x * t;
      const sy = py + dir.y * t;
      const sz = pz + dir.z * t;
      const h = sy - (this.height(sx, sz) - 4);
      res = Math.min(res, (h * 20) / t);
      if (res < 0.02 || t > 4200) break;
      t += Math.min(220, Math.max(h, 3 + t * 0.09));
    }
    return Math.min(1, Math.max(0, res));
  }
}

// ---------------------------------------------------------------- the field

interface Level {
  mesh: THREE.Mesh;
  geometry: THREE.InstancedBufferGeometry;
  /** Per tree, interleaved: `treeA`, `treeB`, `treeC` (see `Forest`). One buffer. */
  data: THREE.InstancedInterleavedBuffer;
}

export interface IslandCanopyOptions {
  seed?: number;
  /**
   * How much of the key light the cloud deck lets through at a world point,
   * 0..1. Drawn into a map over the island; see `CLOUD_MAP_SIZE`. The island's
   * own shadow is baked per tree on the CPU and is not this function's job.
   */
  cloudShadow?: (worldPosition: unknown) => unknown;
}

/**
 * The forest. Add `object` to the scene root: the vertex stage emits world
 * coordinates, so the container must carry an identity transform.
 */
export class IslandCanopy {
  static readonly MAX_COUNT = MAX_CANDIDATES;

  readonly object: THREE.Object3D;

  private readonly forest: Forest;
  private readonly baker = new ShadowBaker();
  private readonly levels: Level[] = [];
  private readonly materials: THREE.MeshBasicNodeMaterial[] = [];
  /** The leaf-clump tile's mean luminance; see `clumpLuminance`. */
  private readonly leafLuminance: number;
  /** See `CLOUD_MAP_SIZE`. Null when no cloud shadow was given. */
  private readonly cloudMap: Node | null;
  private cloudFrame = 0;

  private candidates: number;
  private wantVisible = true;
  private disposed = false;
  private phase = 0;
  private readonly dealtAt = new THREE.Vector3(Infinity, Infinity, Infinity);
  private dealtCandidates = -1;
  private readonly bakedFor = new THREE.Vector3(0, 0, 0);
  private readonly sunDirection = new THREE.Vector3(0.35, 0.62, 0.7).normalize();

  private readonly uPhase = uniform(0);
  private readonly uWind = uniform(new THREE.Vector2(1, 0));
  private readonly uWindStrength = uniform(0.5);
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(0xfff2df));
  private readonly uAmbient = uniform(new THREE.Color(0x9dbbe0));
  private readonly uSunlit = uniform(new THREE.Color(CANOPY_SUNLIT));
  private readonly uShade = uniform(new THREE.Color(CANOPY_SHADE));
  private readonly uDry = uniform(new THREE.Color(CANOPY_DRY));
  private readonly uBark = uniform(new THREE.Color(BARK));
  private readonly table: {
    centres: Node;
    radii: Node;
    envelopes: Node;
    trunks: Node;
  };

  constructor(count: number, options: IslandCanopyOptions = {}) {
    this.candidates = clampCount(count);
    this.forest = plantForest(options.seed ?? CANOPY_SEED, this.sunDirection);

    const shapes = buildShapeTable();
    this.table = {
      centres: uniformArray(shapes.centres, 'vec4'),
      radii: uniformArray(shapes.radii, 'vec4'),
      envelopes: uniformArray(shapes.envelopes, 'vec4'),
      trunks: uniformArray(shapes.trunks, 'vec4'),
    };

    const cloudShadow = (options.cloudShadow as ((p: Node) => Node) | undefined) ?? null;
    this.cloudMap = null;
    if (cloudShadow !== null) {
      const field = Fn(() => {
        const at = uv() as Node;
        const world = vec3n(
          at.x.mul(2 * CLOUD_MAP_HALF).add(ISLAND.x - CLOUD_MAP_HALF),
          CLOUD_MAP_HEIGHT,
          at.y.mul(2 * CLOUD_MAP_HALF).add(ISLAND.z - CLOUD_MAP_HALF),
        );
        return vec4(cloudShadow(world), 0, 0, 1);
      })();
      const map = rtt(field, CLOUD_MAP_SIZE, CLOUD_MAP_SIZE, { type: THREE.HalfFloatType }) as Node;
      // Once a frame at most, not once per render: the reflection pass would
      // otherwise redraw a map that cannot have changed. And not every frame
      // either: clouds drift metres a second over 12.5 m texels, so `update`
      // asks for a redraw every few frames.
      map.updateBeforeType = THREE.NodeUpdateType.FRAME;
      map.autoUpdate = false;
      this.cloudMap = map;
    }

    const atlas = getLeafAtlas();
    this.leafLuminance = clumpLuminance(atlas);

    this.object = new THREE.Object3D();
    this.object.name = 'island-canopy-field';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();

    const capacity = Math.max(1, this.forest.count);
    const bounds = new THREE.Sphere(
      new THREE.Vector3(ISLAND.x, ISLAND.peak * 0.4, ISLAND.z),
      ISLAND.radius * ISLAND_REACH + TREE_HEIGHT * 2,
    );
    for (let lod = 0; lod < 3; lod++) {
      const material = this.buildMaterial(lod, atlas);
      this.materials.push(material);
      const base = buildLevel(lod);
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.index = base.index;
      for (const name of Object.keys(base.attributes)) {
        geometry.setAttribute(name, base.getAttribute(name));
      }
      const data = new THREE.InstancedInterleavedBuffer(new Float32Array(capacity * 12), 12);
      data.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('treeA', new THREE.InterleavedBufferAttribute(data, 4, 0));
      geometry.setAttribute('treeB', new THREE.InterleavedBufferAttribute(data, 4, 4));
      geometry.setAttribute('treeC', new THREE.InterleavedBufferAttribute(data, 4, 8));
      geometry.instanceCount = 0;
      geometry.boundingSphere = bounds;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `island-canopy-lod${lod}`;
      // No shadow casting, as the cards before them. Cast into the sun's
      // cascades, the near crowns are drawn again for every cascade they
      // cross, and the shade they would add is already there: the hill's
      // shadow on the trees is baked per tree, and the ground under them is
      // shaded by the terrain's own march.
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      mesh.visible = false;
      this.object.add(mesh);
      this.levels.push({ mesh, geometry, data });
    }
    this.applyVisibility();
  }

  getCount(): number {
    return this.candidates;
  }

  /** Trees planted, at the current count. */
  get plantedCount(): number {
    let n = 0;
    for (let t = 0; t < this.forest.count; t++) if (this.forest.candidate[t] < this.candidates) n++;
    return n;
  }

  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.candidates) return;
    this.candidates = next;
    this.dealtCandidates = -1;
    this.applyVisibility();
  }

  /** Kept for the quality path's call shape; the detail now follows distance. */
  setDetail(_detail: number): void {}

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** `dir` points *toward* the key light and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    this.sunDirection.copy(dir).normalize();
    (this.uSunDir.value as THREE.Vector3).copy(this.sunDirection);
  }

  setSunColor(color: THREE.Color): void {
    (this.uSunColor.value as THREE.Color).copy(color);
  }

  setAmbientColor(color: THREE.Color): void {
    (this.uAmbient.value as THREE.Color).copy(color);
  }

  /** `direction` is a unit vector in world xz; `strength` is 0..1. */
  setWind(direction: THREE.Vector2, strength: number): void {
    const wind = this.uWind.value as THREE.Vector2;
    wind.copy(direction);
    if (wind.lengthSq() < 1e-6) wind.set(1, 0);
    wind.normalize();
    this.uWindStrength.value = Math.max(0, Math.min(1, strength));
  }

  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (this.disposed) return;
    this.phase = (this.phase + dt) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
    if (this.cloudMap !== null && this.cloudFrame++ % CLOUD_MAP_INTERVAL === 0) {
      this.cloudMap.textureNeedsUpdate = true;
    }
    this.refresh(cameraPosition);
  }

  resetClock(time = 0): void {
    this.phase = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
    // Restart the cadence, so a capture's map does not depend on how many
    // frames ran before it.
    this.cloudFrame = 0;
    if (this.cloudMap !== null) this.cloudMap.textureNeedsUpdate = true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const level of this.levels) {
      this.object.remove(level.mesh);
      level.geometry.dispose();
    }
    for (const material of this.materials) material.dispose();
  }

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.candidates > 0;
  }

  /**
   * Rebakes the hill's shadow if the key light has moved, and redeals levels of
   * detail if the camera has. Both synchronous: a capture taken after a clock
   * change must not catch a half-updated forest.
   */
  private refresh(camera: THREE.Vector3): void {
    let dirty = false;
    // Half a degree.
    if (this.sunDirection.dot(this.bakedFor) < 0.99996) {
      this.baker.bake(this.forest, this.sunDirection);
      this.bakedFor.copy(this.sunDirection);
      dirty = true;
    }
    if (
      dirty ||
      this.dealtCandidates !== this.candidates ||
      camera.distanceToSquared(this.dealtAt) > REDEAL_DISTANCE * REDEAL_DISTANCE
    ) {
      this.deal(camera);
    }
  }

  private deal(camera: THREE.Vector3): void {
    this.dealtAt.copy(camera);
    this.dealtCandidates = this.candidates;
    const counts = [0, 0, 0];
    const { a, b, c } = this.forest;
    const [near, far] = lodBands(this.candidates);
    const near2 = near * near;
    const far2 = far * far;
    const out = this.levels.map((level) => level.data.array as Float32Array);
    for (let t = 0; t < this.forest.count; t++) {
      if (this.forest.candidate[t] >= this.candidates) continue;
      const s = t * 4;
      const dx = a[s] - camera.x;
      const dy = a[s + 1] - camera.y;
      const dz = a[s + 2] - camera.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      const lod = d2 < near2 ? 0 : d2 < far2 ? 1 : 2;
      const k = counts[lod]++ * 12;
      const target = out[lod];
      for (let j = 0; j < 4; j++) {
        target[k + j] = a[s + j];
        target[k + 4 + j] = b[s + j];
        target[k + 8 + j] = c[s + j];
      }
    }
    this.levels.forEach((level, lod) => {
      const n = counts[lod];
      level.geometry.instanceCount = n;
      level.mesh.visible = n > 0;
      level.data.clearUpdateRanges();
      level.data.addUpdateRange(0, Math.max(1, n) * 12);
      level.data.needsUpdate = true;
    });
  }

  /**
   * The canopy's material for one level of detail. A crown's solid, its leaf
   * cards and its trunk are one geometry and one draw, told apart per vertex by
   * `crownPart`, under one lighting model, so a crown is one surface whichever
   * part of it a pixel lands on.
   *
   * The near and middle levels carry the leaf shell and so the atlas and the
   * alpha test; the near one also draws the interior's leaf blotches. The far
   * level is the solid alone, opaque and untextured.
   */
  private buildMaterial(lod: number, atlas: THREE.Texture): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    const shell = lod < 2;
    // Tested, not blended, as every masked foliage in the project is: an
    // instanced mesh sorts as one object, and only a depth write lets a shell
    // of cards resolve against itself. Single-sided: the cards carry both
    // windings themselves.
    if (shell) {
      material.alphaTest = 0.5;
      material.alphaToCoverage = true;
    }
    material.side = THREE.FrontSide;

    const treeA = attribute('treeA', 'vec4') as Node;
    const treeB = attribute('treeB', 'vec4') as Node;
    const treeC = attribute('treeC', 'vec4') as Node;
    const part = attribute('crownPart', 'vec4') as Node;
    const offset = attribute('crownOffset', 'vec4') as Node;
    const unit = positionGeometry as Node;
    const unitNormal = attribute('normal', 'vec3') as Node;

    const vNormal = varying(vec3(0, 1, 0), 'canopyNormal') as Node;
    const vProxy = varying(vec3(0, 1, 0), 'canopyProxy') as Node;
    const vWorld = varying(vec3(0, 0, 0), 'canopyWorld') as Node;
    const vLocal = varying(vec3(0, 0, 0), 'canopyLocal') as Node;
    const vKey = varying(float(1), 'canopyKey') as Node;
    /** Crown height 0..1, bark, interior, leaf. */
    const vPart = varying(vec4(0, 0, 0, 0), 'canopyPart') as Node;

    const rotate = (v: Node, c: Node, s: Node): Node =>
      vec3n(v.x.mul(c).sub(v.z.mul(s)), v.y, v.x.mul(s).add(v.z.mul(c)));

    material.positionNode = Fn(() => {
      // --- the tree's own shape, from the table -----------------------------
      const shape = treeC.z.toInt();
      const slot = shape.mul(SLOTS).add(part.x.toInt());
      const centre = this.table.centres.element(slot) as Node;
      const radii = this.table.radii.element(slot) as Node;
      const envelope = this.table.envelopes.element(shape) as Node;
      const trunkRadius = (this.table.trunks.element(shape) as Node).x;

      const isCard = part.y.equal(PART_CARD);
      const isTrunk = part.y.equal(PART_TRUNK);
      const shrink = float(SOLID_SHRINK[lod]);

      // A puff: the slot's unit sphere scaled to the puff's radii (and pulled
      // in for a solid inside a shell), plus a card's own corner offset.
      const onPuff = centre.xyz
        .add(unit.mul(radii.xyz).mul(isCard.select(float(1), shrink)))
        .add(offset.xyz.mul(offset.w).mul(radii.w).mul(isCard.select(float(1), float(0))));
      // An empty slot collapses to a point inside the crown.
      const puff = centre.w.greaterThan(0.5).select(onPuff, vec3n(0, envelope.x, 0));
      const trunk = unit.mul(envelope.w).add(unitNormal.mul(offset.w).mul(trunkRadius));
      const tree = isTrunk.select(trunk, puff);
      // Normals under the puff's non-uniform radii go by the inverse radii.
      const treeNormal = isTrunk.select(unitNormal, normalize(unitNormal.div(radii.xyz.max(0.01))));

      // The crown's envelope: its height ramp, and the volume normal the
      // shading leans toward.
      const height = tree.y.sub(envelope.x).div(envelope.y.sub(envelope.x)).clamp(0, 1);
      const proxy = normalize(
        vec3n(
          tree.x.div(envelope.z),
          tree.y.sub(envelope.x.add(envelope.y).mul(0.5)).div(envelope.y.sub(envelope.x).mul(0.5)),
          tree.z.div(envelope.z),
        ),
      );

      // --- the tree, placed --------------------------------------------------
      const c = treeA.w.cos();
      const s = treeA.w.sin();
      const scale = vec3n(treeB.x, treeB.y, treeB.z);
      const foot = vec3n(treeA.x, treeA.y, treeA.z);

      // Sway, as a shear growing with height. One travelling wave across the
      // island so neighbouring crowns move together: a canopy moves in gusts,
      // and per-tree random phase reads as static.
      const along = foot.xz.dot(this.uWind);
      const gust = along
        .mul(SWAY_K)
        .sub(this.uPhase.mul(SWAY_OMEGA))
        .add(treeB.w.mul(6.28))
        .sin();
      const rise = tree.y.div(TREE_HEIGHT).clamp(0, 1.2);
      const lean = gust.mul(SWAY_MAX).mul(this.uWindStrength).mul(rise.mul(rise));

      const world = foot
        .add(rotate(tree.mul(scale), c, s))
        .add(vec3n(this.uWind.x.mul(lean), 0, this.uWind.y.mul(lean)));

      vNormal.assign(normalize(rotate(treeNormal.div(scale), c, s)));
      vProxy.assign(normalize(rotate(proxy.div(scale), c, s)));
      vWorld.assign(world);
      // Per-tree offset, so neighbouring trees' leaf breakup does not line up.
      vLocal.assign(tree.add(vec3n(treeC.w.mul(37.1), treeB.w.mul(23.7), treeC.w.mul(11.3))));
      vPart.assign(
        vec4(
          height,
          isTrunk.select(float(1), float(0)),
          part.z,
          isCard.select(float(1), float(0)),
        ),
      );

      // The map is the shade at one height. Along the sun's ray the shade does
      // not change, so a crown at any height reads the map where its ray meets
      // that height: exact on a slope, not just at the map's own level.
      let cloud: Node = float(1);
      if (this.cloudMap !== null) {
        const crown = foot.add(vec3(0, TREE_HEIGHT * 0.6, 0));
        const travel = float(CLOUD_MAP_HEIGHT).sub(crown.y).div(this.uSunDir.y.max(0.06));
        const meet = crown.add(this.uSunDir.mul(travel));
        const mapUv = vec2(
          meet.x.sub(ISLAND.x - CLOUD_MAP_HALF).div(2 * CLOUD_MAP_HALF),
          meet.z.sub(ISLAND.z - CLOUD_MAP_HALF).div(2 * CLOUD_MAP_HALF),
        );
        cloud = this.cloudMap.sample(mapUv).x;
      }
      vKey.assign(treeC.y.mul(cloud));

      return world;
    })();

    material.colorNode = Fn(() => {
      const crownHeight = vPart.x;
      const bark = vPart.y;
      const interior = vPart.z;
      const leaf = vPart.w;
      const toCamera = cameraPosition.sub(vWorld);
      const distance = toCamera.length();
      const view = toCamera.div(distance.max(1e-3));

      // Fine detail fades with distance, where it would alias into shimmer and
      // where a crown is a few pixels anyway.
      const leafFade = float(1).sub(distance.smoothstep(LEAF_FADE_NEAR, LEAF_FADE_FAR));
      const geometric = normalize(vNormal) as Node;
      // A card's own facing is random by construction, so the shell leans
      // harder on the volume than the solid does.
      const volume = (normalize as (v: Node) => Node)(
        mix(geometric, normalize(vProxy) as Node, mix(float(VOLUME_NORMAL), float(0.82), leaf)),
      ) as Node;
      const normal = mix(volume, geometric, bark) as Node;

      // Occlusion a crown lit from above has: its underside is in its own shade,
      // and so is anything facing into the crown. A solid inside a leaf shell
      // is the crown's interior, seen only through gaps, and darker again.
      const occlusion = mix(
        crownHeight
          .smoothstep(0, 1)
          .mul(0.55)
          .add(0.45)
          .mul(vProxy.y.mul(0.2).add(0.8))
          .mul(float(1).sub(interior.mul(0.5))),
        float(0.5),
        bark,
      );

      // The leaf tile's own light and dark, around its mean, and its cutout.
      let tile: Node = float(1);
      let alpha: Node = float(1);
      if (shell) {
        const texel = texture(atlas) as Node;
        const luminance = texel.rgb.dot(vec3(0.2126, 0.7152, 0.0722));
        // A card turned edge-on to the eye is a smear, not a leaf: fade it out
        // through the alpha test as it turns, which every card-foliage renderer
        // does. What shows instead is the crown's interior, which is what an
        // edge-on spray of leaves would have revealed anyway.
        const facing = geometric.dot(view).abs();
        alpha = mix(float(1), texel.a.mul(facing.smoothstep(0.08, 0.32)), leaf);
        let solid: Node = float(1);
        if (lod === 0) {
          // The interior, seen through the shell's gaps: clumps of leaves in
          // shade rather than a surface. Blotches of one Perlin octave about
          // half a metre across: a cellular field drew the clumps better at
          // several times the arithmetic, over solids that sit under every
          // card.
          const blotch = mx_noise_float(vLocal.mul(2.2)) as Node;
          solid = (mix as (a: Node, b: Node, t: Node) => Node)(
            float(1),
            blotch.smoothstep(-0.35, 0.45).mul(0.7).add(0.45),
            leafFade.mul(interior),
          );
        }
        tile = mix(solid, luminance.div(this.leafLuminance).clamp(0.35, 1.9), leaf);
      }

      // Albedo: the two greens by occlusion, then the stand's tone.
      const stand = treeC.x;
      const greens = mix(this.uShade, this.uSunlit, occlusion);
      const toned = mix(greens, this.uDry, stand.smoothstep(0.5, 0.95).mul(0.7));
      const value = stand.sub(0.5).mul(STAND_VALUE_RANGE).add(1);
      const albedo = mix(toned.mul(value).mul(tile), this.uBark, bark);

      // Key light: wrapped, because foliage scatters; gone below the horizon.
      const wrapped = normal
        .dot(this.uSunDir)
        .add(CANOPY_WRAP)
        .div(1 + CANOPY_WRAP)
        .clamp(0, 1);
      const elevation = this.uSunDir.y.max(0).smoothstep(0, 0.4);
      const sun = wrapped.mul(elevation).mul(vKey);

      // Sky from above, less from below.
      const sky = this.uAmbient.mul(normal.y.mul(0.25).add(0.6)).mul(occlusion.mul(0.5).add(0.5));

      // Back-scatter through the crown's thin edges with the sun behind it: the
      // same narrow lobe the mesh foliage uses, weighted to the rim.
      const rim = float(1).sub(normal.dot(view).abs()).mul(0.6).add(0.4);
      const backScatter = view
        .dot(this.uSunDir.negate())
        .clamp(0, 1)
        .pow(TRANSLUCENCY_POWER)
        .mul(TRANSLUCENCY_GAIN)
        .mul(rim)
        .mul(float(1).sub(bark));

      const light = sky.add(this.uSunColor.mul(sun)).add(this.uSunColor.mul(backScatter).mul(vKey));
      return vec4(albedo.mul(light), alpha);
    })();

    return material;
  }
}

function clampCount(count: number): number {
  return Math.max(0, Math.min(MAX_CANDIDATES, Math.floor(count)));
}
