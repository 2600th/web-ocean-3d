import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import {
  MeshBuilder,
  deflect,
  dirNoise,
  growBranch,
  mixRGB,
  scaleRGB,
  smoothstep,
  type RGB,
} from './MeshBuilder';

/**
 * The reef's plants and sessile animals, each built once in code as a template
 * that an `InstancedMesh` repeats.
 *
 * Every template stands on y = 0 with +Y up, in metres at instance scale 1, and
 * is authored near white wherever its hue is meant to come from the instance —
 * so one brain-coral template is a green, a tan and a rose colony depending on
 * nothing but the instance colour. What the vertex colour does carry is the
 * part of the look that is a property of the *form*: occlusion toward the base
 * and between branches, and the paler growing tips of branching coral and
 * anemone tentacles.
 *
 * Triangle counts are recorded beside each builder because they are the budget:
 * the reef draws every template hundreds of times, and a template that grows by
 * a hundred triangles grows the frame by tens of thousands.
 */


const SURF_BRAIN = 1;
const SURF_PORES = 2;

/** Occlusion ramp up a form: dark where it meets the sand, open at the top. */
function aoRamp(t: number, floor = 0.42, reach = 0.45): number {
  return floor + (1 - floor) * smoothstep(0, reach, t);
}

const grey = (k: number): RGB => [k, k, k];

/** Brain coral: a lumpy, slightly flattened dome. ~520 triangles. */
export function buildBrainCoral(seed: number): MeshBuilder {
  const b = new MeshBuilder();
  b.addBlob(
    new THREE.Vector3(0, -0.05, 0),
    new THREE.Vector3(0.62, 0.46, 0.58),
    11,
    24,
    { color: (t) => grey(aoRamp(t, 0.38, 0.55)), surf: SURF_BRAIN },
    (d) => 0.9 + 0.2 * dirNoise(d.x, d.y, d.z, seed),
    -0.12,
  );
  return b;
}

/**
 * Boulder coral: three or four overlapping lumpy mounds, the massive Porites
 * form that makes up most of a reef's bulk. ~700 triangles.
 */
export function buildMoundCoral(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const lobes = 3 + Math.floor(random() * 2);
  for (let i = 0; i < lobes; i++) {
    const a = (i / lobes) * Math.PI * 2 + random() * 0.8;
    const r = i === 0 ? 0 : 0.28 + random() * 0.2;
    const size = i === 0 ? 0.55 : 0.32 + random() * 0.18;
    b.addBlob(
      new THREE.Vector3(Math.cos(a) * r, -0.04, Math.sin(a) * r),
      new THREE.Vector3(size, size * (0.75 + random() * 0.4), size),
      7,
      14,
      { color: (t) => grey(aoRamp(t, 0.35, 0.6)), surf: SURF_PORES },
      (d) => 0.85 + 0.28 * dirNoise(d.x * 1.6, d.y * 1.6, d.z * 1.6, seed + i * 7),
      -0.1,
    );
  }
  return b;
}

/**
 * Staghorn / branching Acropora: a few stout leaders off a low base, each
 * forking, with pale growing tips. ~900 triangles.
 */
export function buildBranchingCoral(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const leaders = 6 + Math.floor(random() * 3);
  const tipColor = (t: number): RGB => {
    const k = aoRamp(t, 0.5, 0.7);
    return mixRGB(grey(k), [1.35, 1.3, 1.25], smoothstep(0.75, 1, t));
  };
  const branch = (from: THREE.Vector3, dir: THREE.Vector3, len: number, r0: number, depth: number, t0: number): void => {
    const segs = depth === 0 ? 4 : 3;
    const pts = growBranch(from, dir, len, segs, 0.6, 0.35, random);
    const radius = pts.map((_, i) => {
      const u = i / (pts.length - 1);
      return i === pts.length - 1 ? r0 * 0.45 : r0 * (1 - 0.45 * u);
    });
    const span = depth === 0 ? 0.55 : 0.45;
    b.addTube(pts, radius, 5, { color: (t) => tipColor(t0 + t * span), surf: SURF_PORES });
    if (depth >= 1) return;
    const forks = 1 + Math.floor(random() * 2.4);
    for (let f = 0; f < forks; f++) {
      const at = Math.floor(1 + random() * (pts.length - 2));
      const d = deflect(pts[at].clone().sub(pts[at - 1]).normalize(), 0.5 + random() * 0.4, random() * Math.PI * 2);
      branch(pts[at], d, len * (0.45 + random() * 0.25), r0 * 0.75, depth + 1, t0 + (at / (pts.length - 1)) * span);
    }
  };
  for (let i = 0; i < leaders; i++) {
    const a = (i / leaders) * Math.PI * 2 + random() * 0.6;
    const spread = 0.12 + random() * 0.18;
    const from = new THREE.Vector3(Math.cos(a) * spread, -0.05, Math.sin(a) * spread);
    const dir = new THREE.Vector3(Math.cos(a) * (0.5 + random() * 0.4), 1, Math.sin(a) * (0.5 + random() * 0.4));
    branch(from, dir, 0.55 + random() * 0.35, 0.045, 0, 0);
  }
  // A low encrusting base so the leaders grow out of something.
  b.addBlob(
    new THREE.Vector3(0, -0.06, 0),
    new THREE.Vector3(0.32, 0.14, 0.32),
    4,
    10,
    { color: (t) => grey(aoRamp(t, 0.35, 0.9)), surf: SURF_PORES },
    (d) => 0.9 + 0.2 * dirNoise(d.x, d.y, d.z, seed),
    -0.2,
  );
  return b;
}

/**
 * Table coral: a short stalk carrying a broad, thin, slightly dished plate with
 * an irregular rim — the horizontal counterpoint to everything else on the reef
 * and the shape fish shelter under. ~380 triangles.
 */
export function buildTableCoral(seed: number): MeshBuilder {
  const b = new MeshBuilder();
  const stalk = [new THREE.Vector3(0, -0.05, 0), new THREE.Vector3(0.02, 0.22, 0), new THREE.Vector3(0, 0.4, 0.01)];
  b.addTube(stalk, [0.11, 0.09, 0.12], 7, { color: (t) => grey(0.35 + 0.3 * t), surf: SURF_PORES });
  const random = mulberry32(seed);
  // Two plates, the upper one smaller and off-centre: a tiered table rather
  // than a disc on a stick. Bottom face outward to the rim, then the top face
  // back in — the lathe winds both outward. The rim lip turns up, the outline is
  // lobed, and the top carries radial ridges of fused branches.
  const tier = (y0: number, size: number, ox: number, oz: number, lobeSeed: number): void => {
    const mark = b.mark();
    const profile: [number, number][] = [
      [0.1, y0 - 0.02],
      [0.45, y0 + 0.03],
      [0.8, y0 + 0.1],
      [0.95, y0 + 0.16],
      [0.98, y0 + 0.21],
      [0.9, y0 + 0.2],
      [0.55, y0 + 0.12],
      [0.12, y0 + 0.08],
      [0.0, y0 + 0.08],
    ];
    const under = (t: number): number => (t < 0.45 ? 0.32 + 0.3 * t : t < 0.6 ? 1.15 : 0.95);
    b.addLathe(
      profile.map(([r, y]) => [r * size, y] as [number, number]),
      36,
      {
        color: (t, a) => grey(under(t) * (t > 0.5 ? 0.85 + 0.15 * Math.cos(a * Math.PI * 2 * 22) : 1)),
        surf: SURF_PORES,
      },
      (a, t) => {
        const lobes = 0.78 + 0.34 * dirNoise(Math.cos(a * 6.283) * 2.2, 0.3, Math.sin(a * 6.283) * 2.2, lobeSeed);
        // Ridges only on the top face, where the fused branches show.
        const ridge = t > 0.55 ? 1 + 0.05 * Math.cos(a * Math.PI * 2 * 22) : 1;
        return lobes * ridge;
      },
    );
    b.transformFrom(mark, new THREE.Matrix4().makeTranslation(ox, 0, oz));
    // Upturned branchlets near the rim: the plate's growing edge.
    const tips = 16;
    for (let i = 0; i < tips; i++) {
      const a = (i / tips) * Math.PI * 2 + random() * 0.3;
      const r = size * (0.55 + random() * 0.3);
      const base = new THREE.Vector3(ox + Math.cos(a) * r, y0 + 0.12 + r * 0.05, oz + Math.sin(a) * r);
      const top = base.clone().add(new THREE.Vector3(Math.cos(a) * 0.03, 0.06 + random() * 0.05, Math.sin(a) * 0.03));
      b.addTube([base, top], [0.022, 0.008], 4, {
        color: (t) => grey(0.95 + 0.35 * t),
        surf: SURF_PORES,
      });
    }
  };
  tier(0.38, 1, 0, 0, seed);
  const ux = 0.2 + random() * 0.15;
  const uz = -0.1 + random() * 0.2;
  b.addTube(
    [new THREE.Vector3(ux * 0.6, 0.44, uz * 0.6), new THREE.Vector3(ux, 0.68, uz)],
    [0.07, 0.06],
    6,
    { color: () => grey(0.45), surf: SURF_PORES },
  );
  tier(0.62, 0.55, ux, uz, seed + 11);
  return b;
}

/**
 * Tube sponges: a clutch of open-topped tubes with a dark interior. ~600
 * triangles.
 */
export function buildTubeSponge(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const tubes = 3 + Math.floor(random() * 3);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  for (let i = 0; i < tubes; i++) {
    const h = 0.45 + random() * 0.65;
    const r = 0.07 + random() * 0.05;
    const mark = b.mark();
    b.addLathe(
      [
        [r * 0.8, -0.05],
        [r, h * 0.15],
        [r * 1.05, h * 0.6],
        [r * 1.18, h],
        [r * 0.9, h],
        [r * 0.8, h * 0.55],
        [r * 0.6, h * 0.25],
      ],
      10,
      {
        color: (t) => (t > 0.6 ? grey(0.25 + 0.4 * (1 - t)) : grey(aoRamp(t * 1.6, 0.45, 0.4))),
        surf: SURF_PORES,
        sway: (t) => (t > 0.3 ? 0.012 : 0),
      },
    );
    const a = (i / tubes) * Math.PI * 2 + random();
    const off = i === 0 ? 0 : 0.1 + random() * 0.1;
    q.setFromEuler(new THREE.Euler((random() - 0.5) * 0.35, 0, (random() - 0.5) * 0.35));
    m.compose(new THREE.Vector3(Math.cos(a) * off, 0, Math.sin(a) * off), q, new THREE.Vector3(1, 1, 1));
    b.transformFrom(mark, m);
  }
  return b;
}

/** Barrel sponge: a ribbed vase. ~380 triangles. */
export function buildBarrelSponge(_seed: number): MeshBuilder {
  const b = new MeshBuilder();
  b.addLathe(
    [
      [0.22, -0.05],
      [0.36, 0.15],
      [0.47, 0.45],
      [0.52, 0.75],
      [0.54, 0.86],
      [0.47, 0.88],
      [0.42, 0.6],
      [0.3, 0.35],
    ],
    28,
    {
      color: (t, a) => {
        const ridge = 0.85 + 0.15 * Math.cos(a * Math.PI * 2 * 14);
        return t > 0.65 ? grey(0.3) : grey(aoRamp(t * 1.4, 0.45, 0.5) * ridge);
      },
      surf: SURF_PORES,
    },
    (a, t) => (t < 0.7 ? 1 + 0.07 * Math.cos(a * Math.PI * 2 * 14) : 1),
  );
  return b;
}

/**
 * Sea fan (gorgonian): a flat, finely branching lattice in one plane, built as
 * two-sided ribbons. The plane is the template's XY; the reef turns each fan
 * broadside to the current, which is how they grow. ~700 triangles.
 */
export function buildSeaFan(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const planeNormal = new THREE.Vector3(0, 0, 1);
  const height = 1.35;
  const sway = (y: number): number => 0.09 * Math.pow(Math.max(0, y) / height, 2);
  const grow = (from: THREE.Vector3, dir: THREE.Vector3, len: number, width: number, depth: number): void => {
    const segs = depth < 2 ? 4 : 3;
    const pts = growBranch(from, dir, len, segs, 0.9, 0.5, random, planeNormal);
    const across = new THREE.Vector3();
    // Ribbons cross the branch within the fan plane, so the lattice is seen
    // face on when the fan is.
    const d = pts[pts.length - 1].clone().sub(pts[0]).normalize();
    across.crossVectors(d, planeNormal).normalize();
    const widths = pts.map((_, i) => width * (1 - 0.5 * (i / (pts.length - 1))));
    const y0 = from.y;
    b.addRibbon(pts, widths, across, {
      color: (t) => grey(0.7 + 0.3 * Math.min(1, (y0 + t * len) / height)),
      sway: (t) => sway(y0 + t * len * 0.9),
    });
    if (depth >= 4) return;
    const forks = depth < 1 ? 3 : 2;
    for (let f = 0; f < forks; f++) {
      const at = Math.max(1, Math.min(pts.length - 1, Math.round(((f + 0.6 + random() * 0.4) / forks) * (pts.length - 1))));
      const tangent = pts[at].clone().sub(pts[at - 1]).normalize();
      const side = f % 2 === 0 ? 1 : -1;
      const angle = (0.35 + random() * 0.35) * side;
      const nd = tangent.clone().applyAxisAngle(planeNormal, angle);
      grow(pts[at], nd, len * (0.62 + random() * 0.18), width * 0.72, depth + 1);
    }
  };
  grow(new THREE.Vector3(0, -0.05, 0), new THREE.Vector3(0.05, 1, 0), 0.42, 0.045, 0);
  // A short holdfast tube so the fan is anchored to something with volume.
  b.addTube(
    [new THREE.Vector3(0, -0.08, 0), new THREE.Vector3(0, 0.12, 0)],
    [0.05, 0.03],
    5,
    { color: () => grey(0.5) },
  );
  return b;
}

/**
 * Sea anemone: a squat column, an oral disc and a crown of tapering tentacles
 * that wriggle independently. ~900 triangles.
 */
export function buildAnemone(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  // Column, darker than the crown.
  b.addLathe(
    [
      [0.1, -0.04],
      [0.13, 0.05],
      [0.12, 0.13],
      [0.16, 0.18],
      [0.0, 0.2],
    ],
    12,
    { color: (t) => grey(0.4 + 0.3 * t) },
  );
  const tentacles = 34;
  for (let i = 0; i < tentacles; i++) {
    const ring = i % 2 === 0 ? 1 : 0.6;
    const a = (i / tentacles) * Math.PI * 2 + random() * 0.2;
    const out = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const len = 0.18 + random() * 0.1;
    const base = new THREE.Vector3(out.x * 0.15 * ring, 0.18, out.z * 0.15 * ring);
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 4; s++) {
      const t = s / 4;
      pts.push(
        base
          .clone()
          .addScaledVector(out, t * len * (0.55 + 0.4 * ring))
          .add(new THREE.Vector3(0, Math.sin(t * Math.PI * 0.55) * len * 0.85, 0)),
      );
    }
    b.addTube(pts, [0.013, 0.012, 0.01, 0.007, 0.002], 3, {
      color: (t) => mixRGB(grey(0.75), [1.25, 1.25, 1.25], smoothstep(0.5, 1, t)),
      sway: (t) => 0.045 * t * t + 0.005,
      wiggle: 1,
    });
  }
  return b;
}

/** Sea whips / soft coral: a few long, thin, curving rods. ~300 triangles. */
export function buildSeaWhip(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const whips = 4 + Math.floor(random() * 4);
  for (let i = 0; i < whips; i++) {
    const a = random() * Math.PI * 2;
    const len = 0.6 + random() * 0.7;
    const dir = new THREE.Vector3(Math.cos(a) * 0.35, 1, Math.sin(a) * 0.35);
    const pts = growBranch(new THREE.Vector3(Math.cos(a) * 0.05, -0.05, Math.sin(a) * 0.05), dir, len, 6, 0.4, 0.3, random);
    b.addTube(
      pts,
      pts.map((_, k) => (k === pts.length - 1 ? 0.004 : 0.016 * (1 - 0.5 * (k / (pts.length - 1))))),
      3,
      {
        color: (t) => grey(0.55 + 0.5 * t),
        sway: (t) => 0.16 * t * t * len,
        wiggle: 0.15,
        surf: SURF_PORES,
      },
    );
  }
  return b;
}

/** A reef rock: a flattened, lumpy boulder with an algal top. ~250 triangles. */
export function buildRock(seed: number): MeshBuilder {
  const b = new MeshBuilder();
  b.addBlob(
    new THREE.Vector3(0, -0.12, 0),
    new THREE.Vector3(0.62, 0.42, 0.52),
    9,
    14,
    {
      // Top faces carry turf algae, sides are bare rock: the instance colour is
      // the rock, and this tints the cap green-brown.
      color: (t) => mixRGB(grey(aoRamp(t, 0.4, 0.5)), [0.62, 0.78, 0.5], smoothstep(0.55, 0.95, t)),
    },
    (d) => {
      const n = dirNoise(d.x * 1.3, d.y * 1.3, d.z * 1.3, seed);
      const m = dirNoise(d.x * 3.1, d.y * 3.1, d.z * 3.1, seed + 3);
      return 0.72 + 0.38 * n + 0.12 * m;
    },
    -0.35,
  );
  return b;
}

/**
 * A conch-like shell lying on its side: a tapering spiral lathe. ~150
 * triangles.
 */
export function buildShell(seed: number): MeshBuilder {
  const b = new MeshBuilder();
  const mark = b.mark();
  const rows: [number, number][] = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    rows.push([0.075 * Math.sin(Math.min(1, t * 1.25) * Math.PI * 0.85) + 0.004, t * 0.2]);
  }
  b.addLathe(
    rows,
    12,
    {
      color: (t, a) => {
        const band = 0.5 + 0.5 * Math.sin((t * 9 + a * 2) * Math.PI);
        return mixRGB([1, 0.92, 0.82], [0.62, 0.42, 0.3], band * 0.6);
      },
    },
    (a, t) => 1 + 0.18 * Math.sin(a * Math.PI * 2 * 3 + t * 9 + seed) * (1 - t),
  );
  // Lay it down.
  const m = new THREE.Matrix4().makeRotationZ(Math.PI / 2 - 0.15);
  m.setPosition(0.1, 0.03, 0);
  b.transformFrom(mark, m);
  return b;
}

/** Starfish: five tapered arms with a domed centre, lying flat. ~180 triangles. */
export function buildStarfish(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2 + (random() - 0.5) * 0.2;
    const out = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const len = 0.13 + random() * 0.03;
    const pts = [0, 0.33, 0.66, 1].map((t) =>
      out.clone().multiplyScalar(t * len).add(new THREE.Vector3(0, 0.022 - 0.014 * t * t, 0)),
    );
    b.addTube(pts, [0.03, 0.022, 0.013, 0.003], 5, {
      color: (t) => grey(0.8 + 0.2 * (1 - t)),
      surf: SURF_PORES,
    });
  }
  b.addBlob(
    new THREE.Vector3(0, 0.005, 0),
    new THREE.Vector3(0.04, 0.028, 0.04),
    3,
    8,
    { color: () => grey(0.9), surf: SURF_PORES },
    () => 1,
    0,
  );
  return b;
}

/** Long-spined sea urchin: a dark test bristling with spines. ~220 triangles. */
export function buildUrchin(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  b.addBlob(
    new THREE.Vector3(0, 0.06, 0),
    new THREE.Vector3(0.08, 0.065, 0.08),
    5,
    10,
    { color: () => grey(0.5) },
    () => 1,
    -0.4,
  );
  const spines = 26;
  for (let i = 0; i < spines; i++) {
    // Fibonacci sphere over the upper part.
    const y = 1 - (i / (spines - 1)) * 1.25;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * 2.39996 + random() * 0.3;
    const d = new THREE.Vector3(Math.cos(phi) * r, y, Math.sin(phi) * r);
    const from = new THREE.Vector3(0, 0.06, 0).addScaledVector(d, 0.07);
    const to = from.clone().addScaledVector(d, 0.14 + random() * 0.1);
    b.addTube([from, to], [0.006, 0], 3, {
      color: (t) => grey(0.6 - 0.25 * t),
      sway: (t) => 0.01 * t,
      wiggle: 1,
    });
  }
  return b;
}

/**
 * A clump of seagrass blades (Thalassia / Posidonia): flat ribbons leaning out
 * from one rhizome, sway growing with the square of the height. ~70 triangles.
 */
export function buildSeagrassClump(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const blades = 11 + Math.floor(random() * 5);
  for (let i = 0; i < blades; i++) {
    const a = random() * Math.PI * 2;
    const lean = 0.08 + random() * 0.3;
    const len = 0.35 + random() * 0.5;
    const out = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const base = out.clone().multiplyScalar(random() * 0.28).setY(-0.03);
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 3; s++) {
      const t = s / 3;
      // Bends over more toward the tip.
      pts.push(
        base
          .clone()
          .addScaledVector(out, len * (lean * t + 0.25 * lean * t * t))
          .add(new THREE.Vector3(0, len * t * (1 - 0.15 * lean * t), 0)),
      );
    }
    const across = new THREE.Vector3(-out.z, 0, out.x);
    const w = 0.02 + random() * 0.012;
    const yellow = random() < 0.3;
    b.addRibbon(pts, [w * 0.8, w, w * 0.85, w * 0.3], across, {
      color: (t) =>
        mixRGB(grey(0.45 + 0.6 * t), yellow ? [1.15, 1.05, 0.6] : [1.05, 1.1, 0.95], smoothstep(0.7, 1, t)),
      sway: (t) => 0.14 * t * t * len,
      wiggle: 0.08,
    });
  }
  return b;
}

/** sRGB hex to a linear multiplier, for hues baked into composite templates. */
function hexRGB(hex: number): RGB {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}

/** Palettes the bommie paints its colonies from, by kind. */
const BOMMIE_PALETTES: Record<string, readonly number[]> = {
  brain: [0xb8a36a, 0x8fae6b, 0xc98f7a, 0xd2b68c],
  mound: [0xa88f63, 0x8c9a6e, 0xc4a160, 0x9a8a70],
  branching: [0xc58fb5, 0x7aa8c9, 0xd6b48b, 0x9cc07a, 0xe08f6a, 0xb7a0d8],
  table: [0x9eb38a, 0xb9a17c, 0x8fb0b5, 0xc7a3b8],
  tube: [0x9a5bb8, 0xe0a63a, 0xd76a3a, 0x6b8fd6, 0xc4485c],
  fan: [0xb63c7a, 0xd8a03a, 0x8a4ab0, 0xc8542e],
  anemone: [0xd59fd8, 0xb6d68a, 0xe3b7a0, 0x9fd4d0],
  whip: [0xe0b23a, 0xd0602e, 0xb84a8a, 0xf0d070],
};

/**
 * A bommie: a coral head several metres across — a lumpy rock mound grown over
 * with a dozen colonies of mixed species, baked into one template with its
 * colours in the vertex data.
 *
 * This is how the reef gets *mass*. Individual colonies scattered on sand read
 * as a garden however many there are; a reef is mostly a few big structures,
 * densely overgrown, with sand between them. One instanced bommie carries that
 * structure for the price of one draw across every site, and because each
 * colony on it is painted from its own palette the instance colour only has to
 * nudge the whole. 11-12k triangles, the reef's most expensive template; the
 * count is in the 15-19 colonies, not the rock.
 */
export function buildBommie(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const radii = new THREE.Vector3(1.7, 1.05, 1.45);
  const shape = (d: THREE.Vector3): number =>
    0.75 +
    0.35 * dirNoise(d.x * 1.2, d.y * 1.2, d.z * 1.2, seed) +
    0.1 * dirNoise(d.x * 3, d.y * 3, d.z * 3, seed + 5);
  b.addBlob(
    new THREE.Vector3(0, -0.2, 0),
    radii,
    9,
    18,
    {
      color: (t) =>
        mixRGB(scaleRGB(hexRGB(0x7a7266), aoRamp(t, 0.35, 0.5)), hexRGB(0x7f8f5a), smoothstep(0.5, 0.95, t)),
      surf: SURF_PORES,
    },
    shape,
    -0.3,
  );
  const kinds: { name: string; build: (s: number) => MeshBuilder; scale: [number, number]; weight: number }[] = [
    { name: 'brain', build: buildBrainCoral, scale: [0.8, 1.3], weight: 3 },
    { name: 'branching', build: buildBranchingCoral, scale: [0.8, 1.25], weight: 3 },
    { name: 'tube', build: buildTubeSponge, scale: [0.8, 1.3], weight: 2 },
    { name: 'table', build: buildTableCoral, scale: [0.5, 0.85], weight: 1.5 },
    { name: 'fan', build: buildSeaFan, scale: [0.8, 1.2], weight: 1.2 },
    { name: 'anemone', build: buildAnemone, scale: [0.8, 1.2], weight: 1 },
    { name: 'whip', build: buildSeaWhip, scale: [0.6, 1.0], weight: 1.5 },
    { name: 'mound', build: buildMoundCoral, scale: [0.55, 0.9], weight: 2 },
  ];
  const total = kinds.reduce((s, k) => s + k.weight, 0);
  const pick = (): (typeof kinds)[number] => {
    let r = random() * total;
    for (const k of kinds) {
      r -= k.weight;
      if (r <= 0) return k;
    }
    return kinds[0];
  };
  const pieces = 15 + Math.floor(random() * 5);
  const dir = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion();
  const yaw = new THREE.Quaternion();
  const m = new THREE.Matrix4();
  for (let i = 0; i < pieces; i++) {
    // The upper surface, biased toward the crown where the light is.
    const theta = Math.acos(1 - Math.pow(random(), 1.4) * 0.85);
    const phi = random() * Math.PI * 2;
    dir.set(Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi));
    const k = shape(dir) * 0.98;
    const at = new THREE.Vector3(dir.x * radii.x * k, -0.2 + dir.y * radii.y * k, dir.z * radii.z * k);
    const kind = pick();
    const palette = BOMMIE_PALETTES[kind.name];
    const piece = kind.build(seed * 31 + i * 7).tint(hexRGB(palette[Math.floor(random() * palette.length)]));
    const s = kind.scale[0] + random() * (kind.scale[1] - kind.scale[0]);
    // Grows mostly toward the light, partly along the surface normal.
    const normal = new THREE.Vector3(dir.x / radii.x, dir.y / radii.y, dir.z / radii.z).normalize();
    q.setFromUnitVectors(up, normal.lerp(up, 0.6).normalize());
    yaw.setFromAxisAngle(up, random() * Math.PI * 2);
    m.compose(at, q.multiply(yaw), new THREE.Vector3(s, s, s));
    b.append(piece, m);
  }
  return b;
}
