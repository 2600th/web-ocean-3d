import * as THREE from 'three/webgpu';
import { M, PropBuilder, box, cylinder, jitter, lathe, srgb, sweep, tube, type MeshKit, type Tint } from './kit';
import { mulberry32 } from '../../core/random';

/**
 * Chests.
 *
 * Two period forms, because the domed pirate "treasure chest" is a later trope
 * and the audit asked for what was actually carried:
 *
 *  - **strongbox**: the Spanish iron-bound arca, the box silver and coin
 *    travelled in on the flota. Oak, a shallow barrel-vaulted lid, wide forged
 *    straps over lid and body, angle plates on every corner, a large lock plate,
 *    a hasp, and iron drop handles at the ends for the two men it takes to lift
 *    it full. This is the reef find, lying with its lid sprung and its coin
 *    spilling into the sand.
 *  - **sea**: the sailor's own sea chest. Flat lid with cleats, sides that
 *    batter in toward the top so it will not tip with the roll, rope beckets
 *    for handles, painted. Cheaper, plainer, and what a wreck actually sheds.
 *
 * Built at true size (the strongbox is 0.95 m long); the placement scales it.
 */

export interface ChestOptions {
  style: 'strongbox' | 'sea';
  /** Lid opening about the back hinge, radians. 0 is shut. */
  open: number;
  /** Coin heap inside and a spill in front, with a few stones. */
  treasure: boolean;
  lod: 0 | 1 | 2;
  seed: number;
}

// Oak gone silver-brown at sea before it sank: pale enough that the black
// straps still read against it at 16 m, where the water has taken most of the
// contrast out of everything.
const OAK: Tint = srgb(0.5, 0.39, 0.28);
const PAINT: Tint = srgb(0.24, 0.31, 0.29);
const HEMP: Tint = srgb(0.62, 0.52, 0.36);
const GEMS: Tint[] = [
  [0.55, 0.02, 0.04],
  [0.02, 0.4, 0.12],
  [0.05, 0.12, 0.6],
];

export function buildChest(pb: PropBuilder, m: THREE.Matrix4, o: ChestOptions): void {
  const rng = mulberry32(o.seed);
  const strong = o.style === 'strongbox';
  const L = strong ? 0.95 : 0.9;
  const D = strong ? 0.55 : 0.46;
  const Hb = strong ? 0.44 : 0.42;
  const t = 0.028;
  const bevel = o.lod === 0 ? 0.005 : 0;
  const plankTint = strong ? OAK : PAINT;
  const wood = pb.kit('wood');
  const iron = pb.kit('iron');
  const planks = o.lod === 2 ? 1 : 3;
  // The sea chest's sides batter in: 3 cm narrower at the top on each face.
  const batter = strong ? 0 : Math.atan2(0.03, Hb);

  wood.withMatrix(m, () => {
    // Front and back.
    for (const side of [-1, 1]) {
      for (let i = 0; i < planks; i++) {
        const h = Hb / planks;
        const y = h * (i + 0.5);
        const z = side * (D / 2 - t / 2 - Math.tan(batter) * y);
        box(wood, L, h - 0.004, t, M.at(0, y, z, side * batter, 0, 0), jitter(plankTint, rng, 0.12), 'x', bevel, rng() * 10);
      }
    }
    // Ends.
    for (const side of [-1, 1]) {
      for (let i = 0; i < planks; i++) {
        const h = Hb / planks;
        const y = h * (i + 0.5);
        const depth = D - 2 * t - 2 * Math.tan(batter) * y;
        box(wood, t, h - 0.004, depth, M.at(side * (L / 2 - t / 2 - Math.tan(batter) * y), y, 0, 0, 0, -side * batter), jitter(plankTint, rng, 0.12), 'z', bevel, rng() * 10);
      }
    }
    // Bottom, standing on two skids.
    box(wood, L - 2 * t, t, D - 2 * t, M.at(0, 0.04 + t / 2, 0), jitter(plankTint, rng, 0.1), 'x', 0);
    for (const sx of [-1, 1]) box(wood, 0.06, 0.04, D + 0.02, M.at(sx * (L / 2 - 0.12), 0.02, 0), jitter(plankTint, rng, 0.1), 'z', bevel);
  });

  // ---- lid, hinged on the back top edge
  const hinge = new THREE.Matrix4().makeTranslation(0, Hb, -D / 2 + (strong ? 0 : Math.tan(batter) * Hb));
  const lidM = m.clone().multiply(hinge).multiply(M.at(0, 0, 0, -o.open, 0, 0)).multiply(new THREE.Matrix4().makeTranslation(0, 0, D / 2 - (strong ? 0 : Math.tan(batter) * Hb)));
  const lidD = strong ? D + 0.01 : D - 2 * Math.tan(batter) * Hb + 0.02;
  const skirt = strong ? 0.07 : 0.06;
  const rise = strong ? 0.16 : 0;
  wood.withMatrix(lidM, () => {
    // Skirt.
    for (const side of [-1, 1]) {
      box(wood, L + 0.01, skirt, t, M.at(0, skirt / 2, side * (lidD / 2 - t / 2)), jitter(plankTint, rng, 0.1), 'x', bevel);
      box(wood, t, skirt, lidD - 2 * t, M.at(side * (L / 2 + 0.005 - t / 2), skirt / 2, 0), jitter(plankTint, rng, 0.1), 'z', bevel);
    }
    if (strong) {
      // Barrel vault of narrow boards running the length of the box.
      const R = (lidD * lidD) / 4 / (2 * rise) + rise / 2;
      const a0 = Math.asin(lidD / 2 / R);
      const boards = o.lod === 0 ? 7 : o.lod === 1 ? 5 : 3;
      for (let i = 0; i < boards; i++) {
        const aa = -a0 + ((2 * a0) / boards) * (i + 0.5);
        const w = 2 * R * Math.sin(a0 / boards) - 0.003;
        const cy = skirt + (R * Math.cos(aa) - (R - rise)) - 0.012;
        box(wood, L + 0.01, t * 0.8, w, M.at(0, cy, R * Math.sin(aa), -aa, 0, 0), jitter(plankTint, rng, 0.12), 'x', bevel, rng() * 10);
      }
      // Gable ends: the vault's segment, closed.
      for (const side of [-1, 1]) {
        const steps = o.lod === 0 ? 10 : 6;
        const centre = wood.vertex(side * (L / 2 + 0.003), skirt, 0, side, 0, 0, 0, 0, plankTint);
        const rim: number[] = [];
        for (let i = 0; i <= steps; i++) {
          const aa = -a0 + ((2 * a0) / steps) * i;
          const z = R * Math.sin(aa);
          const y = skirt + R * Math.cos(aa) - (R - rise) - 0.004;
          rim.push(wood.vertex(side * (L / 2 + 0.003), y, z, side, 0, 0, z, y, plankTint));
        }
        for (let i = 0; i < steps; i++) {
          if (side > 0) wood.tri(centre, rim[i], rim[i + 1]);
          else wood.tri(centre, rim[i + 1], rim[i]);
        }
      }
    } else {
      // Flat lid of two boards, with cleats across the ends.
      for (const sz of [-1, 1]) box(wood, L + 0.03, t, lidD / 2 + 0.008, M.at(0, skirt + t / 2, (sz * lidD) / 4), jitter(plankTint, rng, 0.1), 'x', bevel, rng() * 10);
      for (const sx of [-1, 1]) box(wood, 0.05, 0.02, lidD + 0.02, M.at(sx * (L / 2 - 0.04), skirt + t + 0.01, 0), jitter(plankTint, rng, 0.1), 'z', bevel);
    }
  });

  // ---- ironwork
  const strap = 0.05;
  const plate = 0.006;
  if (strong) {
    // Bands round the body at foot and rim.
    for (const y of [0.07, Hb - 0.035]) ring(iron, L + 2 * plate, D + 2 * plate, y, strap, plate, m);
    // Vertical straps on front and back, continuing over the lid.
    const xs = o.lod === 2 ? [-0.3, 0.3] : [-0.33, 0, 0.33];
    for (const x of xs) {
      for (const side of [-1, 1]) {
        if (x === 0 && side === 1) continue; // the lock plate takes the centre of the front
        box(iron, strap, Hb - 0.05, plate, m.clone().multiply(M.at(x, (Hb - 0.05) / 2 + 0.05, side * (D / 2 + plate / 2))), [1, 1, 1], 'y', 0);
      }
      // Over the vault: an arc following the boards.
      const R = (D * D) / 4 / (2 * rise) + rise / 2;
      const a0 = Math.asin(D / 2 / R);
      const pts: [number, number, number][] = [[x, 0, D / 2 + plate + 0.004]];
      const n = o.lod === 0 ? 10 : 5;
      for (let i = 0; i <= n; i++) {
        const aa = a0 - ((2 * a0) / n) * i;
        pts.push([x, skirt + R * Math.cos(aa) - (R - rise) + plate * 0.5 - 0.004, R * Math.sin(aa) * 1.01]);
      }
      pts.push([x, 0, -D / 2 - plate - 0.004]);
      sweep(iron, pts, strap, plate, (p) => [0, p[1] > skirt * 0.6 ? 1 : 0, p[1] > skirt * 0.6 ? 0 : 1], lidM.clone().multiply(new THREE.Matrix4().makeTranslation(0, 0, 0)), [1, 1, 1]);
    }
    // Corner angle plates.
    if (o.lod < 2) {
      for (const sx of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const h = Hb - 0.04;
          box(iron, 0.09, h, plate, m.clone().multiply(M.at(sx * (L / 2 - 0.045 + plate), h / 2 + 0.04, sz * (D / 2 + plate / 2))), [1, 1, 1], 'y', 0);
          box(iron, plate, h, 0.09, m.clone().multiply(M.at(sx * (L / 2 + plate / 2), h / 2 + 0.04, sz * (D / 2 - 0.045 + plate))), [1, 1, 1], 'y', 0);
        }
      }
    }
    // Lock plate, keyhole, and the hasp hanging from the lid.
    const lockM = m.clone().multiply(M.at(0, Hb - 0.13, D / 2 + plate));
    box(iron, 0.17, 0.19, 0.008, lockM, [1, 1, 1], 'y', o.lod === 0 ? 0.003 : 0);
    box(pb.kit('iron'), 0.014, 0.04, 0.004, lockM.clone().multiply(M.at(0, -0.01, 0.006)), [0.25, 0.25, 0.25], 'y', 0);
    if (o.lod < 2) {
      // Hasp: a hinged bar from the lid skirt, sprung outward now the lock has failed.
      const haspM = lidM.clone().multiply(M.at(0, skirt * 0.5, D / 2 + plate + 0.004, 0.5, 0, 0));
      box(iron, 0.05, 0.16, 0.008, haspM.clone().multiply(M.at(0, -0.07, 0)), [1, 1, 1], 'y', o.lod === 0 ? 0.003 : 0);
      tube(iron, ringPoints(0.022, 10).map(([a, b]) => [a, b - 0.15, 0] as [number, number, number]), 0.005, 5, haspM, [1, 1, 1], false);
      // The padlock, shackle open, still hanging from the staple.
      const padM = m.clone().multiply(M.at(0.045, Hb - 0.25, D / 2 + 0.03, 0.15, 0, 0.2));
      lathe(iron, [[0, -0.045], [0.032, -0.04], [0.035, 0.0], [0.03, 0.03], [0, 0.032]], 10, padM.clone().multiply(M.at(0, 0, 0, Math.PI / 2, 0, 0, [1, 1, 0.45])), [1, 1, 1]);
      tube(iron, ringPoints(0.022, 10, Math.PI * 1.15).map(([a, b]) => [a + 0.008, b + 0.05, 0] as [number, number, number]), 0.006, 6, padM, [1, 1, 1], true);
    }
    // Drop handles on the ends.
    for (const sx of [-1, 1]) {
      const hm = m.clone().multiply(M.at(sx * (L / 2 + plate), Hb * 0.62, 0, 0, sx * Math.PI / 2, 0));
      box(iron, 0.14, 0.05, 0.008, hm, [1, 1, 1], 'x', 0);
      if (o.lod < 2) {
        tube(iron, ringPoints(0.07, o.lod === 0 ? 16 : 8, Math.PI).map(([a, b]) => [a, -b - 0.01, 0.02] as [number, number, number]), 0.008, 6, hm, [1, 1, 1], true);
      }
    }
    if (o.lod === 0) rivets(iron, m, L, D, Hb, xs, plate);
  } else {
    // Sea chest: iron only at the hasp and the lid hinges; rope beckets for handles.
    box(iron, 0.06, 0.12, 0.006, m.clone().multiply(M.at(0, Hb - 0.05, D / 2 - Math.tan(batter) * Hb + 0.004, batter, 0, 0)), [1, 1, 1], 'y', 0);
    if (o.lod < 2) {
      const rope = pb.kit('rope');
      for (const sx of [-1, 1]) {
        const x = sx * (L / 2 - Math.tan(batter) * Hb * 0.6 + 0.02);
        const pts: [number, number, number][] = [];
        for (let i = 0; i <= 10; i++) {
          const a = (i / 10) * Math.PI;
          pts.push([x + sx * Math.sin(a) * 0.06, Hb * 0.68 - Math.sin(a) * 0.04, Math.cos(a) * 0.1]);
        }
        tube(rope, pts, 0.012, 6, m, HEMP, true);
        // The cleats the becket is spliced through.
        for (const sz of [-1, 1]) box(wood, 0.035, 0.05, 0.05, m.clone().multiply(M.at(x - sx * 0.008, Hb * 0.68, sz * 0.1)), jitter(PAINT, rng, 0.1), 'y', bevel);
      }
    }
  }

  if (o.treasure) treasure(pb, m, L, D, Hb, t, o, rng);
}

/** A rectangular iron band round the body at height y. */
function ring(kit: MeshKit, L: number, D: number, y: number, w: number, th: number, m: THREE.Matrix4): void {
  box(kit, L, w, th, m.clone().multiply(M.at(0, y, D / 2 - th / 2)), [1, 1, 1], 'x', 0);
  box(kit, L, w, th, m.clone().multiply(M.at(0, y, -D / 2 + th / 2)), [1, 1, 1], 'x', 0);
  box(kit, th, w, D, m.clone().multiply(M.at(L / 2 - th / 2, y, 0)), [1, 1, 1], 'z', 0);
  box(kit, th, w, D, m.clone().multiply(M.at(-L / 2 + th / 2, y, 0)), [1, 1, 1], 'z', 0);
}

function ringPoints(r: number, n: number, span = Math.PI * 2): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = -span / 2 + (span * i) / n - Math.PI / 2;
    out.push([Math.cos(a) * r, Math.sin(a) * r + r]);
  }
  return out;
}

/** Forged rivet heads along the straps: the detail that says hand-made at arm's length. */
function rivets(kit: MeshKit, m: THREE.Matrix4, L: number, D: number, Hb: number, xs: number[], plate: number): void {
  const head = (x: number, y: number, z: number, ry: number) =>
    lathe(kit, [[0.0001, 0.006], [0.006, 0.004], [0.008, 0]], 6, m.clone().multiply(M.at(x, y, z, Math.PI / 2, ry, 0)), [1, 1, 1]);
  for (const x of xs) {
    for (const side of [-1, 1]) {
      if (x === 0 && side === 1) continue;
      for (let y = 0.1; y < Hb - 0.06; y += 0.09) head(x, y, side * (D / 2 + plate), side > 0 ? 0 : Math.PI);
    }
  }
  for (const sz of [-1, 1]) {
    for (let x = -L / 2 + 0.08; x < L / 2 - 0.05; x += 0.12) {
      head(x, 0.07, sz * (D / 2 + plate), sz > 0 ? 0 : Math.PI);
      head(x, Hb - 0.035, sz * (D / 2 + plate), sz > 0 ? 0 : Math.PI);
    }
  }
}

/** Height of the coin heap's lathe profile at normalised radius r (0 centre .. 1 rim). */
function heapHeight(r: number): number {
  const prof: [number, number][] = [
    [0, 0.13],
    [0.5, 0.12],
    [0.8, 0.09],
    [0.96, 0.05],
    [1, 0],
  ];
  for (let i = 0; i < prof.length - 1; i++) {
    const [r0, h0] = prof[i];
    const [r1, h1] = prof[i + 1];
    if (r <= r1) return h0 + ((h1 - h0) * (r - r0)) / (r1 - r0);
  }
  return 0;
}

/** Pieces of eight heaped in the box and spilling over the front into the sand. */
function treasure(pb: PropBuilder, m: THREE.Matrix4, L: number, D: number, Hb: number, t: number, o: ChestOptions, rng: () => number): void {
  const gold = pb.kit('gold');
  // The heap: a lumpy mound filling the box to just under the rim.
  const heapTop = Hb - 0.06;
  lathe(
    gold,
    [
      [0.5, 0],
      [0.48, 0.05],
      [0.4, 0.09],
      [0.25, 0.12],
      [0, 0.13],
    ],
    o.lod === 0 ? 20 : 10,
    m.clone().multiply(M.at(0, heapTop - 0.1, 0.02, 0, 0, 0, [(L - 2 * t) / 1.0, 1, (D - 2 * t) / 1.0])),
    [1, 1, 1],
    { radius: (a, k) => (k === 0 ? 1 : 1 + 0.08 * Math.sin(a * 3 + k) + 0.05 * Math.sin(a * 7)) },
  );
  if (o.lod === 2) return;

  const coin = (x: number, y: number, z: number, rx: number, rz: number, tint: Tint) =>
    cylinder(gold, 0.019, 0.019, 0.0035, o.lod === 0 ? 9 : 6, m.clone().multiply(M.at(x, y, z, rx, rng() * 6, rz)), tint);
  const coinTint = (): Tint => {
    // Silver and gold together: pieces of eight were silver, escudos gold.
    const silver = rng() < 0.45;
    return silver ? [0.5, 0.5, 0.5] : [1, 1, 1];
  };

  const onHeap = o.lod === 0 ? 240 : 50;
  for (let i = 0; i < onHeap; i++) {
    const a = rng() * Math.PI * 2;
    const r = Math.sqrt(rng());
    const x = Math.cos(a) * r * (L / 2 - 0.08);
    const z = Math.sin(a) * r * (D / 2 - 0.08) + 0.02;
    const y = heapTop - 0.1 + heapHeight(r) + 0.006;
    coin(x, y, z, (rng() - 0.5) * 0.7, (rng() - 0.5) * 0.7, coinTint());
  }
  // Over the front rim and down: a trail that widens as it reaches the ground.
  const spill = o.lod === 0 ? 70 : 20;
  for (let i = 0; i < spill; i++) {
    const s = rng();
    const z = D / 2 + 0.02 + s * 0.55;
    const spread = 0.08 + s * 0.35;
    const x = (rng() - 0.5) * 2 * spread + 0.05;
    coin(x, 0.004 + rng() * 0.01, z, (rng() - 0.5) * 0.4, (rng() - 0.5) * 0.4, coinTint());
  }
  // A few standing on edge against the front, where they slid.
  for (let i = 0; i < (o.lod === 0 ? 6 : 2); i++) {
    coin(-0.2 + rng() * 0.4, 0.02, D / 2 + 0.03, Math.PI / 2 - 0.35, 0, coinTint());
  }

  // Cut stones: emerald from New Granada, a ruby, a sapphire. Octahedral, as cut.
  const gem = pb.kit('gem');
  const gems = o.lod === 0 ? 6 : 3;
  for (let i = 0; i < gems; i++) {
    const g = new THREE.OctahedronGeometry(0.022 + rng() * 0.012, 0);
    const onSand = i % 2 === 1;
    const x = (rng() - 0.5) * (onSand ? 0.6 : L - 0.3);
    const z = onSand ? D / 2 + 0.1 + rng() * 0.35 : (rng() - 0.5) * (D - 0.25);
    const y = onSand ? 0.015 : heapTop + 0.02;
    gem.add(g, m.clone().multiply(M.at(x, y, z, rng() * 3, rng() * 3, 0, [1, 0.75, 1])), GEMS[i % GEMS.length], 'keep');
    g.dispose();
  }
}
