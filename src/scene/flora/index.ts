import * as THREE from 'three/webgpu';
import { normalViewGeometry } from 'three/tsl';
import { mulberry32 } from '../../core/random';
import { getLeafAtlas } from './leafAtlas';
import {
  buildPalm,
  buildRosette,
  buildShrub,
  buildTree,
  type PlantLevel,
  type TreeSpecies,
} from './plants';

/**
 * The island's planting, generated instead of downloaded.
 *
 * Every kind here used to be a photogrammetry or botanical glTF — 11 MB of them,
 * plus a 1.8 MB impostor atlas baked from them — and the island still read as a
 * bald dome, because a scan that size can only be planted in the dozens. These
 * are a few kilobytes of code and one canvas-drawn atlas, so the same budget buys
 * a forest.
 *
 * The output is deliberately shaped like a loaded model: a `Group` per level of
 * detail, one mesh per material, base at the origin and y up, in metres. `Props`
 * bakes, instances, LOD-deals and thins them through exactly the path the glTF
 * kinds went through, and `applyGroundShadingTo` gives them the same contact
 * shade, cloud shadow, wind and leaf translucency.
 */

export const FLORA_KEYS = [
  'palmCoconut',
  'palmTall',
  'palmLean',
  'tree',
  'treeFlame',
  'treeMid',
  'jacaranda',
  'pachira',
  'fern',
  'sorrel',
  'anthurium',
  'calathea',
  'grassMeadow',
  'grassTussock',
] as const;
export type FloraKey = (typeof FLORA_KEYS)[number];

export interface FloraEntry {
  base: THREE.Group;
  lods: THREE.Group[];
}

type RGB = readonly [number, number, number];

// Leaf tints multiply the atlas colour; near 1 keeps the drawn palette and the
// spread between them is what separates one stand from the next.
const GREEN: RGB = [1, 1, 1];
const GREEN_DEEP: RGB = [0.78, 0.9, 0.82];
const GREEN_WARM: RGB = [1.12, 1.06, 0.78];
const GREEN_COOL: RGB = [0.82, 0.98, 0.95];

const CANOPY: TreeSpecies = {
  trunk: { r: 0.3, top: 4.6, lean: 0.06, segs: 7 },
  limbs: { n: 6, y: [2.8, 4.4], tilt: [1.0, 0.55], len: [4.6, 3.6], r: 0.62, up: 0.06, out: 0, droop: 0, jitter: 0.08, segs: 6 },
  branches: { n: 4, t: [0.3, 0.98], angle: [0.55, 0.95], len: [0.6, 0.42], up: 0.06, out: 0.02, droop: 0, jitter: 0.1, segs: 4 },
  twigs: { n: 3, t: [0.3, 1.0], angle: [0.6, 1.05], len: [1.0, 0.7], up: 0.03, out: 0, droop: 0.04, jitter: 0.12, segs: 2 },
  cards: { perTwig: 2, scale: [0.85, 1.15], droop: 0.08, size: 1.45 },
  flowers: 0,
  vines: 0.35,
  tints: [GREEN, GREEN_DEEP, GREEN_COOL],
  bark: [0.78, 0.74, 0.66],
};

/** The rain tree: a low trunk under a vast, flat-topped umbrella. */
const UMBRELLA: TreeSpecies = {
  trunk: { r: 0.42, top: 3.6, lean: 0.05, segs: 6 },
  limbs: { n: 7, y: [2.6, 3.5], tilt: [1.2, 0.85], len: [7.2, 5.6], r: 0.62, up: 0.09, out: 0.01, droop: 0, jitter: 0.08, segs: 7 },
  branches: { n: 4, t: [0.3, 0.98], angle: [0.6, 1.0], len: [0.5, 0.36], up: 0.08, out: 0.03, droop: 0, jitter: 0.1, segs: 4 },
  twigs: { n: 3, t: [0.3, 1.0], angle: [0.6, 1.0], len: [1.1, 0.8], up: 0.05, out: 0, droop: 0.02, jitter: 0.12, segs: 2 },
  cards: { perTwig: 2, scale: [0.9, 1.2], droop: 0.02, size: 1.6 },
  flowers: 0,
  vines: 0.5,
  tints: [GREEN, GREEN_WARM, GREEN_DEEP],
  bark: [0.72, 0.68, 0.62],
};

/** A flame tree: the umbrella again, smaller, with a share of flowering sprays. */
const FLAME: TreeSpecies = {
  ...UMBRELLA,
  trunk: { r: 0.32, top: 3.2, lean: 0.07, segs: 6 },
  limbs: { ...UMBRELLA.limbs, n: 6, len: [5.4, 4.2] },
  cards: { ...UMBRELLA.cards, size: 1.4 },
  flowers: 0.3,
  vines: 0.1,
  tints: [GREEN, GREEN_WARM],
};

/** The emergent: a tall clear bole carrying its crown above the canopy. */
const EMERGENT: TreeSpecies = {
  trunk: { r: 0.38, top: 10.5, lean: 0.04, segs: 9 },
  limbs: { n: 6, y: [8.2, 10.6], tilt: [0.95, 0.5], len: [4.4, 3.2], r: 0.6, up: 0.07, out: 0, droop: 0, jitter: 0.08, segs: 6 },
  branches: { n: 4, t: [0.3, 0.98], angle: [0.55, 0.95], len: [0.62, 0.42], up: 0.05, out: 0.02, droop: 0, jitter: 0.1, segs: 4 },
  twigs: { n: 3, t: [0.3, 1.0], angle: [0.6, 1.05], len: [1.0, 0.7], up: 0.02, out: 0, droop: 0.05, jitter: 0.12, segs: 2 },
  cards: { perTwig: 2, scale: [0.85, 1.1], droop: 0.1, size: 1.4 },
  flowers: 0,
  vines: 0.6,
  tints: [GREEN_COOL, GREEN, GREEN_DEEP],
  bark: [0.86, 0.82, 0.76],
};

function canvasTexture(
  name: string,
  size: [number, number],
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number, rng: () => number) => void,
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable for ' + name);
  draw(ctx, size[0], size[1], mulberry32(name.length * 7919 + 17));
  const tex = new THREE.CanvasTexture(canvas);
  tex.name = name;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  return tex;
}

/** One ring of a palm trunk per texture repeat: fibre streaks and a scar groove. */
function palmBarkTexture(): THREE.CanvasTexture {
  return canvasTexture('flora-palm-bark', [128, 128], (c, w, h, rng) => {
    c.fillStyle = 'rgb(150, 134, 112)';
    c.fillRect(0, 0, w, h);
    for (let i = 0; i < 260; i++) {
      const x = rng() * w;
      const y = rng() * h;
      const l = 6 + rng() * 26;
      const k = 0.75 + rng() * 0.5;
      c.strokeStyle = `rgb(${Math.round(132 * k)}, ${Math.round(118 * k)}, ${Math.round(96 * k)})`;
      c.lineWidth = 1 + rng() * 1.5;
      c.beginPath();
      c.moveTo(x, y);
      c.lineTo(x + (rng() - 0.5) * 3, y + l);
      c.stroke();
    }
    // The leaf scar: a dark groove at the ring boundary, lighter lip above it.
    const groove = c.createLinearGradient(0, 0, 0, h);
    groove.addColorStop(0, 'rgba(60, 50, 40, 0.85)');
    groove.addColorStop(0.1, 'rgba(60, 50, 40, 0)');
    groove.addColorStop(0.82, 'rgba(200, 186, 160, 0)');
    groove.addColorStop(0.93, 'rgba(200, 186, 160, 0.35)');
    groove.addColorStop(1, 'rgba(60, 50, 40, 0.85)');
    c.fillStyle = groove;
    c.fillRect(0, 0, w, h);
  });
}

/** Smooth grey-brown tropical bark: lenticels and shallow vertical fissures. */
function treeBarkTexture(): THREE.CanvasTexture {
  return canvasTexture('flora-tree-bark', [128, 128], (c, w, h, rng) => {
    c.fillStyle = 'rgb(138, 126, 110)';
    c.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) {
      const x = rng() * w;
      c.strokeStyle = `rgba(70, 60, 50, ${0.3 + rng() * 0.4})`;
      c.lineWidth = 1 + rng() * 2;
      c.beginPath();
      let y = rng() * h;
      c.moveTo(x, y);
      for (let s = 0; s < 6; s++) {
        y += 6 + rng() * 10;
        c.lineTo(x + (rng() - 0.5) * 4, y);
      }
      c.stroke();
    }
    for (let i = 0; i < 160; i++) {
      const k = rng() < 0.5 ? 'rgba(190, 180, 160, 0.5)' : 'rgba(90, 96, 70, 0.35)';
      c.fillStyle = k;
      c.fillRect(rng() * w, rng() * h, 2 + rng() * 3, 1);
    }
  });
}

interface FloraMaterials {
  leaf: THREE.MeshStandardNodeMaterial;
  palmBark: THREE.MeshStandardNodeMaterial;
  treeBark: THREE.MeshStandardNodeMaterial;
}

function buildMaterials(): FloraMaterials {
  const leaf = new THREE.MeshStandardNodeMaterial({
    map: getLeafAtlas(),
    vertexColors: true,
    side: THREE.DoubleSide,
    alphaTest: 0.5,
    roughness: 0.68,
    metalness: 0,
  });
  leaf.name = 'flora-leaf';
  leaf.alphaToCoverage = true;
  leaf.envMapIntensity = 0.7;
  // Crown-shell normals on both faces. The default flips the normal on a back
  // face, which on a card canopy lights half the leaves from inside the tree.
  leaf.normalNode = normalViewGeometry;

  const bark = (name: string, map: THREE.Texture): THREE.MeshStandardNodeMaterial => {
    const m = new THREE.MeshStandardNodeMaterial({ map, vertexColors: true, roughness: 0.92, metalness: 0 });
    m.name = name;
    m.envMapIntensity = 0.7;
    return m;
  };
  return {
    leaf,
    palmBark: bark('flora-palm-bark', palmBarkTexture()),
    treeBark: bark('flora-tree-bark', treeBarkTexture()),
  };
}

function toGroup(level: PlantLevel, mats: FloraMaterials, bark: 'palmBark' | 'treeBark', name: string): THREE.Group {
  const g = new THREE.Group();
  g.name = name;
  if (level.bark) {
    const m = new THREE.Mesh(level.bark, mats[bark]);
    m.name = name + '-bark';
    g.add(m);
  }
  if (level.leaf) {
    const m = new THREE.Mesh(level.leaf, mats.leaf);
    m.name = name + '-leaf';
    g.add(m);
  }
  g.updateMatrixWorld(true);
  return g;
}

function entry(levels: PlantLevel[], mats: FloraMaterials, bark: 'palmBark' | 'treeBark', name: string): FloraEntry {
  return {
    base: toGroup(levels[0], mats, bark, name),
    lods: levels.slice(1).map((l, i) => toGroup(l, mats, bark, `${name}-lod${i + 1}`)),
  };
}

/** Builds every procedural kind. Seeded, so every load grows the same island. */
export function buildFlora(): Record<FloraKey, FloraEntry> {
  const mats = buildMaterials();
  const palm = (seed: number, p: Parameters<typeof buildPalm>[1]) => buildPalm(seed, p);
  return {
    palmCoconut: entry(
      palm(0x9a1101, {
        height: 9.5,
        lean: 0.34,
        recover: 0.65,
        baseRadius: 0.2,
        topRadius: 0.14,
        fronds: 14,
        frondLength: 4.6,
        coconuts: 7,
        deadFronds: 2,
        tint: [1, 1, 0.95],
      }),
      mats,
      'palmBark',
      'palm-coconut',
    ),
    palmTall: entry(
      palm(0x9a1102, {
        height: 14,
        lean: 0.16,
        recover: 0.4,
        baseRadius: 0.19,
        topRadius: 0.13,
        fronds: 12,
        frondLength: 4.2,
        coconuts: 4,
        deadFronds: 1,
        tint: [0.92, 1, 0.92],
      }),
      mats,
      'palmBark',
      'palm-tall',
    ),
    palmLean: entry(
      palm(0x9a1103, {
        height: 9,
        lean: 0.85,
        recover: 0.8,
        baseRadius: 0.21,
        topRadius: 0.14,
        fronds: 13,
        frondLength: 4.4,
        coconuts: 5,
        deadFronds: 2,
        tint: [1.05, 1.02, 0.88],
      }),
      mats,
      'palmBark',
      'palm-lean',
    ),
    tree: entry(buildTree(CANOPY, 0x7e0001), mats, 'treeBark', 'tree-canopy'),
    treeFlame: entry(buildTree(FLAME, 0x7e0002), mats, 'treeBark', 'tree-flame'),
    treeMid: entry(buildTree(EMERGENT, 0x7e0003), mats, 'treeBark', 'tree-emergent'),
    jacaranda: entry(buildTree(UMBRELLA, 0x7e0004), mats, 'treeBark', 'tree-umbrella'),
    pachira: entry(
      buildRosette(
        {
          tile: 'bigleaf',
          leaves: 8,
          length: [1.9, 2.5],
          elevation: [1.2, 0.35],
          droop: 0.32,
          widthScale: 0.55,
          fold: 0.18,
          stem: { height: 2.0, radius: 0.12, colour: [0.62, 0.7, 0.42] },
          tints: [GREEN, GREEN_WARM],
        },
        0x5e0001,
      ),
      mats,
      'palmBark',
      'banana',
    ),
    fern: entry(
      buildRosette(
        {
          tile: 'fern',
          leaves: 14,
          length: [1.1, 1.7],
          elevation: [1.3, 0.65],
          droop: 0.42,
          widthScale: 1,
          fold: 0.12,
          tints: [GREEN, GREEN_DEEP, GREEN_COOL],
        },
        0x5e0002,
      ),
      mats,
      'treeBark',
      'fern',
    ),
    sorrel: entry(
      buildRosette(
        {
          tile: 'bigleaf',
          leaves: 9,
          length: [1.1, 1.7],
          elevation: [1.25, 0.55],
          droop: 0.45,
          widthScale: 0.9,
          fold: 0.1,
          tints: [GREEN, GREEN_DEEP, GREEN_COOL],
        },
        0x5e0003,
      ),
      mats,
      'treeBark',
      'aroid',
    ),
    anthurium: entry(
      buildShrub(
        {
          tile: 'clump',
          cards: 10,
          size: [1.0, 1.4],
          radius: 0.6,
          height: 0.9,
          splay: 0.5,
          topTile: 'flower',
          topCards: 7,
          tints: [GREEN, GREEN_DEEP],
        },
        0x5e0004,
      ),
      mats,
      'treeBark',
      'hibiscus',
    ),
    calathea: entry(
      buildShrub(
        {
          tile: 'clump',
          cards: 14,
          size: [1.1, 1.7],
          radius: 0.8,
          height: 1.0,
          splay: 0.55,
          topTile: 'broadleaf',
          topCards: 6,
          tints: [GREEN_DEEP, GREEN, GREEN_COOL],
        },
        0x5e0005,
      ),
      mats,
      'treeBark',
      'bush',
    ),
    grassMeadow: entry(
      buildShrub(
        { tile: 'grass', cards: 6, size: [0.7, 1.0], radius: 0.25, height: 0.2, splay: 0.5, tints: [GREEN, GREEN_WARM] },
        0x5e0006,
      ),
      mats,
      'treeBark',
      'grass',
    ),
    grassTussock: entry(
      buildShrub(
        { tile: 'grass', cards: 8, size: [0.9, 1.3], radius: 0.35, height: 0.2, splay: 0.8, tints: [GREEN_WARM, [1.2, 1.1, 0.7]] },
        0x5e0007,
      ),
      mats,
      'treeBark',
      'tussock',
    ),
  };
}
