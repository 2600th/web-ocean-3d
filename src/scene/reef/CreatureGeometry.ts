import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import { MeshBuilder, mixRGB, smoothstep, type RGB } from './MeshBuilder';

/**
 * Procedural animal templates for `SeaCreatures`.
 *
 * Every animal is authored nose toward +Z, back toward +Y, in metres at scale 1,
 * and every moving part is tagged in the `reefAux` channel so the vertex stage
 * knows how to move it. No skeleton: a flipper is a set of vertices with a part
 * id and a pivot the shader rotates them about, a wing is a span coordinate the
 * shader bends, a tail is a body coordinate the shader waves.
 */

const grey = (k: number): RGB => [k, k, k];

/** Part ids for the turtle, stored in `reefAux`. */
export const TURTLE_PART = { body: 0, frontLeft: 1, frontRight: 2, rearLeft: 3, rearRight: 4, head: 5 } as const;

/** Shoulder and hip pivots, template space. The shader rotates about these. */
export const TURTLE_PIVOTS = {
  front: new THREE.Vector3(0.27, -0.02, 0.24),
  rear: new THREE.Vector3(0.2, -0.04, -0.34),
} as const;

/**
 * A green sea turtle, about a metre nose to tail. ~1300 triangles.
 *
 * Carapace a flattened dome with scute plates (surface 3), a paler plastron,
 * a blunt head on a short neck, long front flippers swept back like wings and
 * short rear paddles.
 */
export function buildTurtle(): MeshBuilder {
  const b = new MeshBuilder();
  // Green turtle: olive-brown carapace, darker grey-olive skin, cream plastron.
  const shell: RGB = [0.25, 0.2, 0.09];
  const skin: RGB = [0.2, 0.19, 0.12];
  const belly: RGB = [0.6, 0.55, 0.36];

  // Carapace: the upper dome.
  let mark = b.mark();
  b.addBlob(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.36, 0.17, 0.48),
    10,
    22,
    { color: (t) => mixRGB(shell, scaleUp(shell, 1.25), smoothstep(0.5, 1, t)), surf: 3 },
    (d) => 1 - 0.06 * Math.max(0, -d.z) + 0.03 * Math.cos(d.x * 9),
    -0.05,
  );
  b.setAuxFrom(mark, TURTLE_PART.body);
  // Plastron: a flatter lower half.
  mark = b.mark();
  b.addBlob(
    new THREE.Vector3(0, 0.005, 0),
    new THREE.Vector3(0.34, 0.07, 0.45),
    5,
    22,
    { color: () => belly },
    () => 1,
    -1,
  );
  // Flip it so the dome faces down: mirror Y and fix the winding by mirroring X.
  b.transformFrom(mark, new THREE.Matrix4().makeScale(-1, -1, 1));
  b.setAuxFrom(mark, TURTLE_PART.body);

  // Head and neck.
  mark = b.mark();
  b.addBlob(
    new THREE.Vector3(0, 0.0, 0.6),
    new THREE.Vector3(0.075, 0.068, 0.13),
    6,
    12,
    // Pale only on the jaw; a blunt beak narrows the front.
    { color: (t) => mixRGB(skin, belly, smoothstep(0.55, 0.95, 1 - t) * 0.7), surf: 5 },
    (d) => 1 - 0.18 * Math.max(0, d.z) * (1 - Math.abs(d.y)),
  );
  b.addTube(
    [new THREE.Vector3(0, -0.01, 0.38), new THREE.Vector3(0, 0, 0.5)],
    [0.07, 0.062],
    8,
    { color: () => skin, surf: 5 },
  );
  b.setAuxFrom(mark, TURTLE_PART.head);

  // Flippers: flattened blobs, swept back, pivoting at the shoulder.
  const flipper = (side: number, front: boolean): void => {
    const m = b.mark();
    const span = front ? 0.42 : 0.17;
    const chord = front ? 0.11 : 0.08;
    b.addBlob(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(span, 0.022, chord),
      4,
      14,
      { color: (t) => mixRGB(skin, scaleUp(skin, 0.75), t), surf: 5 },
      // Taper toward the tip.
      (d) => 1 - 0.35 * smoothstep(0.2, 1, Math.abs(d.x)) * (Math.abs(d.z) + Math.abs(d.y) * 0.5),
    );
    const pivot = front ? TURTLE_PIVOTS.front : TURTLE_PIVOTS.rear;
    const sweep = front ? 0.55 : 0.9;
    const mat = new THREE.Matrix4()
      .makeTranslation(side * pivot.x, pivot.y, pivot.z)
      .multiply(new THREE.Matrix4().makeRotationY(side * sweep))
      .multiply(new THREE.Matrix4().makeTranslation(side * span * 0.85, 0, 0));
    b.transformFrom(m, mat);
    b.setAuxFrom(
      m,
      front
        ? side < 0
          ? TURTLE_PART.frontLeft
          : TURTLE_PART.frontRight
        : side < 0
          ? TURTLE_PART.rearLeft
          : TURTLE_PART.rearRight,
    );
  };
  flipper(-1, true);
  flipper(1, true);
  flipper(-1, false);
  flipper(1, false);

  // A short tail.
  mark = b.mark();
  b.addTube(
    [new THREE.Vector3(0, -0.01, -0.44), new THREE.Vector3(0, -0.02, -0.56)],
    [0.035, 0.005],
    5,
    { color: () => skin },
  );
  b.setAuxFrom(mark, TURTLE_PART.body);
  return b;
}

function scaleUp(c: RGB, k: number): RGB {
  return [c[0] * k, c[1] * k, c[2] * k];
}

export interface RayStyle {
  /** Half span, metres. */
  halfSpan: number;
  /** Root chord, metres. */
  chord: number;
  top: RGB;
  bottom: RGB;
  /** Surface pattern on the back: 0 plain, 4 spots. */
  topSurf: number;
  /** Tail length as a multiple of the chord. */
  tail: number;
  /** Manta cephalic fins. */
  horns: boolean;
}

export const MANTA: RayStyle = {
  halfSpan: 1.9,
  chord: 1.25,
  top: [0.06, 0.065, 0.075],
  bottom: [0.92, 0.92, 0.9],
  topSurf: 0,
  tail: 0.7,
  horns: true,
};

export const EAGLE_RAY: RayStyle = {
  halfSpan: 0.85,
  chord: 0.75,
  top: [0.13, 0.15, 0.2],
  bottom: [0.95, 0.95, 0.92],
  topSurf: 4,
  tail: 2.4,
  horns: false,
};

/**
 * A ray: a swept, thick-bodied wing built as an upper and a lower surface over
 * a span x chord grid, a tail whip and, for a manta, cephalic fins.
 * `reefAux` carries the signed span coordinate (-1..1) the flap is driven by.
 * ~600 triangles.
 */
export function buildRay(style: RayStyle): MeshBuilder {
  const b = new MeshBuilder();
  const NU = 18;
  const NV = 8;
  const H = style.halfSpan;
  const C = style.chord;
  const planform = (u: number): { lead: number; trail: number } => {
    const a = Math.abs(u);
    // The chord shrinks toward the tip and the whole wing sweeps back.
    const centre = -0.32 * C * Math.pow(a, 1.25);
    const chord = C * (0.06 + 0.94 * (1 - Math.pow(a, 1.7)));
    // A rounded head: the leading edge pushes forward at the root.
    const head = 0.12 * C * Math.max(0, 1 - a * 6);
    return { lead: centre + chord * 0.5 + head, trail: centre - chord * 0.5 };
  };
  const thickness = (u: number, v: number): number => {
    const a = Math.abs(u);
    const body = 0.16 * C * Math.pow(Math.max(0, 1 - a * 1.6), 1.6) + 0.012 * C;
    return body * (1 - Math.pow(2 * v - 1, 2));
  };
  for (const side of [1, -1]) {
    const base = b.vertexCount;
    for (let i = 0; i <= NU; i++) {
      const u = -1 + (2 * i) / NU;
      const { lead, trail } = planform(u);
      for (let j = 0; j <= NV; j++) {
        const v = j / NV;
        const z = trail + (lead - trail) * v;
        const y = side > 0 ? thickness(u, v) : -0.55 * thickness(u, v);
        const edge = smoothstep(0.75, 1, Math.abs(u));
        const color = side > 0 ? mixRGB(style.top, scaleUp(style.top, 1.6), edge * 0.5) : style.bottom;
        b.vertex(u * H, y, z, color, 0, 0, side > 0 ? style.topSurf : 0, u);
      }
    }
    for (let i = 0; i < NU; i++) {
      for (let j = 0; j < NV; j++) {
        const a = base + i * (NV + 1) + j;
        const c = a + NV + 1;
        if (side > 0) b.idx.push(a, a + 1, c, a + 1, c + 1, c);
        else b.idx.push(a, c, a + 1, a + 1, c, c + 1);
      }
    }
  }
  // Tail whip.
  let mark = b.mark();
  const root = planform(0).trail;
  const pts: THREE.Vector3[] = [];
  for (let k = 0; k <= 6; k++) pts.push(new THREE.Vector3(0, 0, root + 0.05 - (k / 6) * C * style.tail));
  b.addTube(
    pts,
    pts.map((_, k) => (k === 6 ? 0.001 : 0.025 * C * (1 - k / 7))),
    4,
    { color: () => style.top },
  );
  b.setAuxFrom(mark, 0);
  if (style.horns) {
    for (const s of [-1, 1]) {
      mark = b.mark();
      const lead = planform(0.12).lead;
      b.addBlob(
        new THREE.Vector3(s * 0.2 * C, 0.01, lead + 0.06 * C),
        new THREE.Vector3(0.035 * C, 0.03 * C, 0.13 * C),
        4,
        8,
        { color: () => style.top },
      );
      b.setAuxFrom(mark, s * 0.1);
    }
  } else {
    // Eagle ray: a duck-billed snout.
    mark = b.mark();
    b.addBlob(
      new THREE.Vector3(0, 0.02 * C, planform(0).lead - 0.02 * C),
      new THREE.Vector3(0.09 * C, 0.06 * C, 0.14 * C),
      5,
      10,
      { color: (t) => mixRGB(style.bottom, style.top, t) },
    );
    b.setAuxFrom(mark, 0);
  }
  return b;
}

/**
 * A small reef fish, unit length along +Z, nose forward. `reefAux` is the body
 * coordinate 0 at the nose to 1 at the tail fork. ~60 triangles.
 */
export function buildReefFish(): MeshBuilder {
  const b = new MeshBuilder();
  // Stations along the body: z, half height, half width.
  const stations: [number, number, number][] = [
    [0.5, 0.0, 0.0],
    [0.42, 0.13, 0.07],
    [0.22, 0.24, 0.1],
    [0.0, 0.25, 0.1],
    [-0.2, 0.18, 0.075],
    [-0.33, 0.08, 0.04],
    [-0.4, 0.05, 0.025],
  ];
  const RING = 6;
  const base = b.vertexCount;
  for (let i = 0; i < stations.length; i++) {
    const [z, hh, hw] = stations[i];
    const s = (0.5 - z) / 0.9;
    for (let j = 0; j <= RING; j++) {
      const a = (j / RING) * Math.PI * 2;
      const y = Math.cos(a) * hh;
      const x = Math.sin(a) * hw;
      // Countershading: dark back, pale belly.
      const k = 0.55 + 0.45 * (0.5 - 0.5 * Math.cos(a));
      b.vertex(x, y, z, grey(1.1 - k * 0.4), 0, 0, 0, s);
    }
  }
  for (let i = 0; i < stations.length - 1; i++) {
    for (let j = 0; j < RING; j++) {
      const a = base + i * (RING + 1) + j;
      const c = a + RING + 1;
      b.idx.push(a, c, a + 1, a + 1, c, c + 1);
    }
  }
  // Forked tail fin: two triangles each side-on.
  const t0 = b.vertex(0, 0, -0.38, grey(0.9), 0, 0, 0, 0.98);
  const t1 = b.vertex(0, 0.2, -0.56, grey(1), 0, 0, 0, 1.15);
  const t2 = b.vertex(0, 0.02, -0.48, grey(0.9), 0, 0, 0, 1.08);
  const t3 = b.vertex(0, -0.2, -0.56, grey(1), 0, 0, 0, 1.15);
  b.tri(t0, t1, t2);
  b.tri(t0, t2, t3);
  // Dorsal and anal fins.
  const d0 = b.vertex(0, 0.24, 0.12, grey(0.9), 0, 0, 0, 0.42);
  const d1 = b.vertex(0, 0.36, -0.08, grey(0.95), 0, 0, 0, 0.64);
  const d2 = b.vertex(0, 0.16, -0.26, grey(0.9), 0, 0, 0, 0.84);
  b.tri(d0, d1, d2);
  const a0 = b.vertex(0, -0.2, -0.05, grey(0.9), 0, 0, 0, 0.6);
  const a1 = b.vertex(0, -0.3, -0.2, grey(0.95), 0, 0, 0, 0.78);
  const a2 = b.vertex(0, -0.12, -0.3, grey(0.9), 0, 0, 0, 0.89);
  b.tri(a0, a1, a2);
  return b;
}

/** Part ids for the crab's legs: 1..8 legs, 9 claws, 0 body. */
export function buildCrab(seed: number): MeshBuilder {
  const random = mulberry32(seed);
  const b = new MeshBuilder();
  const shell: RGB = [1, 1, 1];
  let mark = b.mark();
  b.addBlob(
    new THREE.Vector3(0, 0.045, 0),
    new THREE.Vector3(0.075, 0.03, 0.055),
    5,
    12,
    { color: (t) => grey(0.7 + 0.3 * t), surf: 2 },
    (d) => 1 + 0.08 * Math.cos(d.x * 5),
  );
  // Eyes.
  for (const s of [-1, 1]) {
    b.addTube(
      [new THREE.Vector3(s * 0.02, 0.06, 0.045), new THREE.Vector3(s * 0.025, 0.085, 0.05)],
      [0.006, 0.006],
      3,
      { color: () => grey(0.15) },
    );
  }
  b.setAuxFrom(mark, 0);
  // Legs: four a side, each a two-segment tube up and out then down.
  let leg = 1;
  for (const s of [-1, 1]) {
    for (let i = 0; i < 4; i++) {
      mark = b.mark();
      const z = 0.03 - i * 0.025;
      const hip = new THREE.Vector3(s * 0.06, 0.04, z);
      const knee = new THREE.Vector3(s * (0.11 + random() * 0.01), 0.07, z - 0.005 * i);
      const foot = new THREE.Vector3(s * 0.15, 0.0, z - 0.01 * i);
      b.addTube([hip, knee, foot], [0.008, 0.006, 0.002], 3, { color: () => shell });
      b.setAuxFrom(mark, leg++);
    }
  }
  // Claws.
  for (const s of [-1, 1]) {
    mark = b.mark();
    b.addTube(
      [new THREE.Vector3(s * 0.05, 0.04, 0.05), new THREE.Vector3(s * 0.08, 0.05, 0.09)],
      [0.01, 0.012],
      4,
      { color: () => shell },
    );
    b.addBlob(new THREE.Vector3(s * 0.085, 0.05, 0.11), new THREE.Vector3(0.022, 0.016, 0.03), 3, 8, {
      color: () => grey(1.1),
    });
    b.setAuxFrom(mark, 9);
  }
  return b;
}

/**
 * A moon jelly: a domed bell with a flared rim, four frilled oral arms and a
 * fringe of fine tentacles. `reefSway.x` carries the coordinate along each
 * appendage (0 at the bell, 1 at the tip); `reefAux` the part (0 bell, 1 arm,
 * 2 tentacle). Bell radius 1. ~520 triangles.
 */
export function buildJellyfish(): MeshBuilder {
  const b = new MeshBuilder();
  const rows: [number, number][] = [];
  for (let i = 0; i <= 7; i++) {
    const t = i / 7;
    // From the rim (t 0) up to the crown (t 1): a flattened dome, flared at the edge.
    const ang = (1 - t) * Math.PI * 0.5;
    rows.push([Math.sin(ang) * (1 + 0.08 * (1 - t) * (1 - t)), Math.cos(ang) * 0.62 - 0.05 * (1 - t)]);
  }
  rows.push([0.001, 0.62]);
  let mark = b.mark();
  b.addLathe(rows, 18, {
    color: (t, a) => {
      // Four pale-pink gonad horseshoes near the crown.
      const ring = Math.abs(Math.sin(a * Math.PI * 4));
      const gonad = smoothstep(0.55, 0.85, t) * (1 - smoothstep(0.92, 1, t)) * smoothstep(0.4, 0.9, ring);
      return mixRGB(grey(1), [1.6, 0.8, 1.2], gonad);
    },
  });
  b.setAuxFrom(mark, 0);
  // Oral arms.
  for (let k = 0; k < 4; k++) {
    mark = b.mark();
    const a = (k / 4) * Math.PI * 2 + 0.4;
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 6; s++) {
      const t = s / 6;
      pts.push(new THREE.Vector3(Math.cos(a) * (0.12 + 0.15 * t), -t * 1.3, Math.sin(a) * (0.12 + 0.15 * t)));
    }
    b.addRibbon(pts, pts.map((_, i) => 0.18 * (1 - (i / 6) * 0.6)), new THREE.Vector3(-Math.sin(a), 0, Math.cos(a)), {
      color: () => [1.2, 1.0, 1.1],
      sway: (t) => t,
    });
    b.setAuxFrom(mark, 1);
  }
  // Tentacle fringe.
  const T = 20;
  for (let k = 0; k < T; k++) {
    mark = b.mark();
    const a = (k / T) * Math.PI * 2;
    const len = 1.4 + 0.6 * ((k * 7) % 5) / 5;
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 5; s++) {
      const t = s / 5;
      pts.push(new THREE.Vector3(Math.cos(a) * (1.02 - 0.15 * t), -0.04 - t * len, Math.sin(a) * (1.02 - 0.15 * t)));
    }
    b.addRibbon(pts, pts.map((_, i) => 0.035 * (1 - (i / 5) * 0.7)), new THREE.Vector3(-Math.sin(a), 0, Math.cos(a)), {
      color: () => grey(1.1),
      sway: (t) => t,
    });
    b.setAuxFrom(mark, 2);
  }
  return b;
}
