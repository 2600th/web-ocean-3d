import * as THREE from 'three/webgpu';
import { M, PropBuilder, box, cylinder, jitter, lathe, srgb, sweep, tube, type Tint } from './kit';
import { buildCask, CASK_PRESETS } from './cask';
import { mulberry32 } from '../../core/random';

/**
 * Ship's stores and camp debris, all built at true size with the origin on the
 * ground (or, for flotsam, on the waterline).
 */

type Lod = 0 | 1 | 2;
type P3 = [number, number, number];

const WHITE: Tint = [1, 1, 1];
const HEMP: Tint = srgb(0.62, 0.52, 0.36);
const TARRED: Tint = srgb(0.17, 0.13, 0.1);
const PINE: Tint = srgb(0.58, 0.45, 0.3);
const GREY: Tint = srgb(0.4, 0.36, 0.31);
const ELM: Tint = srgb(0.4, 0.28, 0.18);
const BRASS: Tint = [0.78, 0.55, 0.3];

/** Rotation taking a lathe's +Y axis to +Z (for barrels lying along Z). */
const Y_TO_Z = M.at(0, 0, 0, Math.PI / 2, 0, 0);

// --------------------------------------------------------------- stoneware

/**
 * A Bartmann ("Bellarmine") jug: Rhenish salt-glazed stoneware, round-bellied,
 * narrow-necked, a strap handle, the bearded mask on the neck and a medallion
 * on the belly. The commonest ceramic on Port Royal and Caribbean wreck sites.
 */
export function buildBartmann(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  const k = pb.kit('stoneware');
  const seg = lod === 0 ? 20 : lod === 1 ? 12 : 8;
  lathe(
    k,
    [
      [0, 0],
      [0.05, 0],
      [0.056, 0.012],
      [0.088, 0.055],
      [0.104, 0.105],
      [0.1, 0.15],
      [0.072, 0.198],
      [0.036, 0.232],
      [0.03, 0.268],
      [0.037, 0.283],
      [0.034, 0.294],
      [0.026, 0.292],
      [0.024, 0.26],
    ],
    seg,
    m,
    WHITE,
  );
  if (lod === 2) return;
  // Strap handle, neck to shoulder.
  const pts: P3[] = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    const a = Math.PI * 0.95 * t;
    pts.push([-0.035 - Math.sin(a) * 0.05 - t * 0.04, 0.262 - (1 - Math.cos(a)) * 0.052, 0]);
  }
  sweep(k, pts, 0.026, 0.011, () => [1, 0, 0], m, WHITE);
  // The mask and the medallion: shallow domes standing proud of the glaze.
  lathe(k, [[0, 0.008], [0.017, 0.005], [0.02, 0]], 8, m.clone().multiply(M.at(0, 0.226, 0.038, Math.PI / 2 - 0.35, 0, 0, [1, 1, 1.25])), WHITE);
  lathe(k, [[0, 0.006], [0.026, 0.004], [0.03, 0]], 10, m.clone().multiply(M.at(0, 0.11, 0.101, Math.PI / 2, 0, 0)), WHITE);
}

/**
 * A square case bottle of dark green glass, the gin and spirit bottle shipped
 * a dozen to a wooden case: wider at the shoulder than the base, short neck,
 * thickened lip.
 */
export function buildCaseBottle(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  lathe(
    pb.kit('bottle'),
    [
      [0, 0.006],
      [0.055, 0],
      [0.062, 0.01],
      [0.074, 0.2],
      [0.07, 0.212],
      [0.034, 0.232],
      [0.016, 0.244],
      [0.015, 0.268],
      [0.02, 0.272],
      [0.019, 0.282],
      [0.012, 0.282],
    ],
    4,
    m,
    WHITE,
    { start: Math.PI / 4 },
  );
  void lod;
}

/** A Spanish olive jar (botija): the egg-shaped coarse-ware jar for oil and wine. */
export function buildOliveJar(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  lathe(
    pb.kit('stoneware'),
    [
      [0, 0],
      [0.06, 0.01],
      [0.13, 0.09],
      [0.155, 0.19],
      [0.14, 0.29],
      [0.09, 0.355],
      [0.045, 0.38],
      [0.05, 0.395],
      [0.04, 0.405],
      [0.03, 0.395],
    ],
    lod === 0 ? 18 : 10,
    m,
    [1.25, 1.35, 1.2],
  );
}

// ----------------------------------------------------------------- lantern

/**
 * A lanthorn: square wooden frame, horn panes (the "-horn" in the name), a
 * pyramid top with a smoke chimney and a ring to hang it by, a candle inside.
 */
export function buildLantern(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  const wood = pb.kit('wood');
  const iron = pb.kit('iron');
  const W = 0.2;
  const H = 0.3;
  box(wood, W + 0.02, 0.03, W + 0.02, m.clone().multiply(M.at(0, 0.015, 0)), ELM, 'x', lod === 0 ? 0.004 : 0);
  box(wood, W + 0.02, 0.025, W + 0.02, m.clone().multiply(M.at(0, H + 0.0125, 0)), ELM, 'x', lod === 0 ? 0.004 : 0);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(wood, 0.018, H, 0.018, m.clone().multiply(M.at((sx * W) / 2, H / 2 + 0.015, (sz * W) / 2)), ELM, 'y', 0);
  const glass = pb.kit('lanternGlass');
  for (let i = 0; i < 4; i++) {
    const a = (i * Math.PI) / 2;
    box(glass, W - 0.02, H - 0.03, 0.004, m.clone().multiply(M.at(Math.sin(a) * (W / 2 - 0.004), H / 2 + 0.015, Math.cos(a) * (W / 2 - 0.004), 0, a, 0)), WHITE, 'y', 0);
    if (lod === 0) box(iron, W - 0.02, 0.008, 0.006, m.clone().multiply(M.at(Math.sin(a) * (W / 2 + 0.001), H * 0.55, Math.cos(a) * (W / 2 + 0.001), 0, a, 0)), WHITE, 'x', 0);
  }
  lathe(iron, [[0.16, H + 0.025], [0.04, H + 0.12], [0.035, H + 0.17], [0, H + 0.17]], 4, m.clone().multiply(M.at(0, 0, 0, 0, Math.PI / 4, 0)), WHITE, { hard: true });
  if (lod < 2) {
    const ring: P3[] = [];
    for (let i = 0; i <= 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      ring.push([0, H + 0.21 + Math.sin(a) * 0.035, Math.cos(a) * 0.035]);
    }
    tube(iron, ring, 0.005, 4, m, WHITE, false);
    cylinder(pb.kit('wood'), 0.013, 0.013, 0.11, 6, m.clone().multiply(M.at(0, 0.085, 0)), [0.85, 0.8, 0.66]);
    lathe(pb.kit('flame'), [[0, 0], [0.01, 0.015], [0.007, 0.035], [0, 0.05]], 5, m.clone().multiply(M.at(0, 0.142, 0)), WHITE);
  }
}

// ------------------------------------------------------------------ cannon

/**
 * A nine-pounder on a four-truck carriage: the gun every ship and every small
 * battery of the period mounted. Cast iron barrel with reinforce rings, a
 * cascabel and a muzzle swell; elm cheeks stepped down toward the breech so
 * the handspikes can bite; four solid trucks; iron capsquares over the
 * trunnions; the quoin under the breech that sets the elevation.
 * +Z is the muzzle; origin on the ground under the trunnions.
 */
export function buildCannon(pb: PropBuilder, m: THREE.Matrix4, lod: Lod, seed: number): void {
  const rng = mulberry32(seed);
  const wood = pb.kit('wood');
  const iron = pb.kit('iron');
  const seg = lod === 0 ? 28 : lod === 1 ? 16 : 10;
  const bevel = lod === 0 ? 0.012 : 0;
  const axisY = 0.74;
  const trunnionZ = 0.15;
  const elevation = 0.05;

  // Barrel, breech at y = 0 in its own lathe frame, trunnions at 1.15 m.
  const L = 2.75;
  const barrelM = m
    .clone()
    .multiply(M.at(0, axisY, trunnionZ, -elevation, 0, 0))
    .multiply(new THREE.Matrix4().makeTranslation(0, 0, -1.15))
    .multiply(Y_TO_Z);
  const prof: [number, number][] = [
    [0, -0.27],
    [0.05, -0.25],
    [0.062, -0.2],
    [0.04, -0.16],
    [0.042, -0.1],
    [0.15, -0.04],
    [0.205, 0],
    [0.21, 0.07],
    [0.195, 0.08],
    [0.192, 0.92],
    [0.2, 0.93],
    [0.2, 0.98],
    [0.18, 0.99],
    [0.177, 1.6],
    [0.185, 1.61],
    [0.185, 1.65],
    [0.165, 1.66],
    [0.135, 2.45],
    [0.15, 2.46],
    [0.15, 2.5],
    [0.128, 2.52],
    [0.128, 2.58],
    [0.162, 2.68],
    [0.158, L],
    [0.07, L],
    [0.07, L - 0.25],
  ];
  lathe(iron, prof, seg, barrelM, WHITE);
  // Trunnions.
  for (const sx of [-1, 1]) {
    cylinder(iron, 0.068, 0.072, 0.16, lod === 0 ? 12 : 8, m.clone().multiply(M.at(sx * 0.27, axisY, trunnionZ, 0, 0, Math.PI / 2)), WHITE);
  }

  // Carriage cheeks, stepped toward the rear.
  const cheekX = 0.27;
  const steps: [number, number, number, number][] = [
    // z0, z1, y0, y1
    [-0.85, 0.75, 0.24, 0.52],
    [-0.45, 0.75, 0.52, 0.64],
    [-0.05, 0.75, 0.64, axisY - 0.04],
  ];
  if (lod === 2) steps.splice(1, 2, [-0.45, 0.75, 0.52, axisY - 0.04]);
  for (const sx of [-1, 1]) {
    for (const [z0, z1, y0, y1] of steps) {
      box(wood, 0.11, y1 - y0, z1 - z0, m.clone().multiply(M.at(sx * cheekX, (y0 + y1) / 2, (z0 + z1) / 2)), jitter(ELM, rng, 0.1), 'z', bevel, rng() * 10);
    }
  }
  // Transom, axletrees, bed.
  box(wood, cheekX * 2 - 0.1, 0.24, 0.12, m.clone().multiply(M.at(0, 0.42, 0.62)), jitter(ELM, rng, 0.1), 'x', bevel);
  for (const [z, r] of [
    [0.55, 0.2],
    [-0.62, 0.17],
  ] as const) {
    box(wood, 0.98, 0.15, 0.17, m.clone().multiply(M.at(0, r + 0.02, z)), jitter(ELM, rng, 0.1), 'x', bevel);
    for (const sx of [-1, 1]) {
      // Trucks: solid elm wheels, an iron linchpin through the axle end.
      const tm = m.clone().multiply(M.at(sx * 0.42, r, z, 0, 0, Math.PI / 2));
      lathe(wood, [[0.03, -0.05], [r, -0.05], [r + 0.004, -0.03], [r + 0.004, 0.03], [r, 0.05], [0.03, 0.05]], lod === 0 ? 22 : 12, tm, jitter(ELM, rng, 0.1), { hard: true });
      if (lod < 2) cylinder(iron, 0.045, 0.045, 0.13, 8, tm, WHITE);
    }
  }
  box(wood, cheekX * 2 - 0.1, 0.05, 0.75, m.clone().multiply(M.at(0, 0.32, -0.35)), jitter(ELM, rng, 0.1), 'z', 0);
  // The quoin: a wedge under the breech.
  if (lod < 2) {
    box(wood, 0.24, 0.1, 0.42, m.clone().multiply(M.at(0, 0.4, -0.75, 0.12, 0, 0)), jitter(ELM, rng, 0.1), 'z', bevel);
    cylinder(wood, 0.02, 0.02, 0.18, 6, m.clone().multiply(M.at(0, 0.43, -1.02, Math.PI / 2, 0, 0)), jitter(ELM, rng, 0.1));
  }
  // Capsquares and the cheek bolts.
  for (const sx of [-1, 1]) {
    const cm = m.clone().multiply(M.at(sx * cheekX, axisY + 0.02, trunnionZ, 0, 0, 0));
    box(iron, 0.12, 0.03, 0.2, cm.clone().multiply(M.at(0, 0.06, 0)), WHITE, 'z', 0);
    if (lod === 0) {
      for (const z of [-0.6, -0.1, 0.5]) cylinder(iron, 0.022, 0.022, 0.14, 6, m.clone().multiply(M.at(sx * cheekX, 0.42, z, 0, 0, Math.PI / 2)), WHITE);
      // Breeching ring bolt on the outside of each cheek.
      const ring: P3[] = [];
      for (let i = 0; i <= 10; i++) {
        const a = (i / 10) * Math.PI * 2;
        ring.push([sx * (cheekX + 0.07), 0.45 + Math.sin(a) * 0.04, 0.05 + Math.cos(a) * 0.04]);
      }
      tube(iron, ring, 0.008, 5, m, WHITE, false);
    }
  }
}

// ----------------------------------------------------------------- cutlass

/**
 * A cutlass: short, broad, slightly curved single-edged blade, an iron shell
 * guard with a knuckle bow, a wooden grip and a brass pommel. The point is at
 * the origin and the hilt up +Y, so it can be planted point-down.
 */
export function buildCutlass(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  const iron = pb.kit('iron');
  const n = lod === 0 ? 14 : 6;
  const BL = 0.66;
  const pts: P3[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    pts.push([0.045 * (1 - t) * (1 - t) - 0.012 * t, t * BL, 0]);
  }
  // Width tapers to the point, the back stays straight-ish.
  sweep(iron, pts, (k) => 0.006 + 0.03 * Math.min(1, (k / n) * 4), (k) => 0.003 + 0.003 * (k / n), () => [0, 0, 1], m, [1.6, 1.6, 1.6]);
  // Shell guard and quillon.
  cylinder(iron, 0.055, 0.06, 0.006, lod === 0 ? 12 : 8, m.clone().multiply(M.at(0.006, BL + 0.004, 0, 0, 0, 0, [1, 1, 0.7])), WHITE);
  if (lod < 2) {
    const bow: P3[] = [];
    for (let i = 0; i <= 8; i++) {
      const a = (i / 8) * Math.PI;
      bow.push([0.045 + Math.sin(a) * 0.022, BL + 0.01 + (1 - Math.cos(a)) * 0.06, 0]);
    }
    tube(iron, bow, 0.0045, 5, m, WHITE, true);
  }
  lathe(pb.kit('wood'), [[0.012, BL + 0.008], [0.015, BL + 0.05], [0.013, BL + 0.11], [0.011, BL + 0.115]], 7, m, [0.12, 0.08, 0.05]);
  lathe(pb.kit('gold'), [[0.01, BL + 0.112], [0.019, BL + 0.122], [0.016, BL + 0.14], [0, BL + 0.145]], 8, m, BRASS);
}

// ------------------------------------------------------------------- conch

/**
 * A queen conch (Aliger gigas): the Caribbean shell, a knobbed spire and a
 * body whorl ending in the broad, flared pink lip. Modelled as a tube swept
 * along a logarithmic helico-spiral (Raup's coiling model), the lip flaring
 * over the last part of the turn. Rests on its lip; origin under it.
 */
export function buildConch(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  const k = pb.kit('shell');
  const turns = 4.5;
  const tMax = turns * Math.PI * 2;
  const b = Math.log(2.6) / (Math.PI * 2);
  const steps = lod === 0 ? 120 : lod === 1 ? 60 : 30;
  const ring = lod === 0 ? 14 : lod === 1 ? 9 : 6;
  const outer: Tint = [0.78, 0.62, 0.45];
  const pink: Tint = [0.95, 0.52, 0.48];
  const rows: number[][] = [];
  let along = 0;
  let prev: P3 | null = null;
  for (let i = 0; i <= steps; i++) {
    const th = (i / steps) * tMax;
    const s = Math.exp(b * (th - tMax));
    const R = 0.085 * s;
    const Y = 0.2 * (1 - s);
    const rr = 0.06 * s;
    const lip = THREE.MathUtils.smoothstep(th, tMax - 1.3, tMax);
    const cx = Math.cos(th) * R;
    const cz = Math.sin(th) * R;
    const centre: P3 = [cx, -Y, cz];
    if (prev) along += Math.hypot(centre[0] - prev[0], centre[1] - prev[1], centre[2] - prev[2]);
    prev = centre;
    const row: number[] = [];
    for (let j = 0; j <= ring; j++) {
      const ph = (j / ring) * Math.PI * 2;
      // Knobs on the shoulder of each whorl.
      const shoulder = Math.pow(Math.max(0, Math.cos(ph - 1.0)), 6) * (0.5 + 0.5 * Math.cos(th * 8)) * 0.35 * (1 - lip);
      let r = rr * (1 + shoulder);
      let ex = Math.cos(ph);
      let ey = Math.sin(ph) * 1.35;
      // The lip: the outer half of the section flares out and up into a wing.
      if (lip > 0 && ex > -0.2) {
        r *= 1 + lip * 1.6 * (ex + 0.2);
        ey += lip * 0.6 * Math.max(0, ex);
      }
      if (lip > 0 && ex < 0) ex *= 1 - lip * 0.4;
      const px = cx + Math.cos(th) * ex * r;
      const pz = cz + Math.sin(th) * ex * r;
      const py = -Y + ey * r;
      const nx = Math.cos(th) * Math.cos(ph);
      const nz = Math.sin(th) * Math.cos(ph);
      const ny = Math.sin(ph);
      const inside = lip > 0.2 && Math.cos(ph) < 0.2;
      const tint: Tint = lip > 0.35 || inside ? pink : outer;
      row.push(k.vertexCount);
      k.withMatrix(m, () => {
        k.vertex(px, py, pz, nx, ny, nz, ph * Math.max(r, 0.01), along, tint);
      });
    }
    rows.push(row);
  }
  for (let i = 0; i < rows.length - 1; i++) {
    for (let j = 0; j < ring; j++) k.quad(rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]);
  }
}

// ------------------------------------------------------------------ anchor

/**
 * An Admiralty-pattern ("old plan") anchor, the only kind before the stockless
 * anchor of the 1820s: a long iron shank with the ring at the top, a wooden
 * stock of two timbers bolted and hooped across it at right angles to the
 * arms, and two curved arms ending in broad triangular flukes. Shank 2.4 m —
 * a bower for a ship of a few hundred tons. Arms in the XY plane, stock along Z.
 */
export function buildAnchor(pb: PropBuilder, m: THREE.Matrix4, lod: Lod): void {
  const iron = pb.kit('iron');
  const wood = pb.kit('wood');
  const S = 2.4;
  // Shank: square section, swelling toward the crown.
  sweep(iron, [[0, 0.1, 0], [0, S * 0.5, 0], [0, S, 0]], (k) => [0.12, 0.1, 0.085][k], (k) => [0.12, 0.1, 0.085][k], () => [0, 0, 1], m, WHITE);
  // Arms: arcs from the crown, round section.
  const n = lod === 0 ? 10 : 5;
  for (const sx of [-1, 1]) {
    const arm: P3[] = [];
    for (let i = 0; i <= n; i++) {
      const a = -Math.PI / 2 + (sx * (Math.PI * 0.32) * i) / n;
      arm.push([Math.cos(a) * 0.95 * sx * sx, 0.95 + Math.sin(a) * 0.95, 0]);
    }
    tube(iron, arm.map(([x, y, z]) => [x, y + 0.02, z] as P3), (k) => 0.07 - 0.025 * (k / n), lod === 0 ? 8 : 5, m, WHITE, true);
    const tip = arm[arm.length - 1];
    // The fluke (palm): a broad spade on the inside face of the arm's end.
    const ang = -Math.PI / 2 + sx * Math.PI * 0.32;
    const fm = m.clone().multiply(M.at(tip[0] - sx * 0.08, tip[1] - 0.16, 0, 0, 0, ang + Math.PI / 2 + (sx > 0 ? 0 : Math.PI)));
    lathe(iron, [[0, -0.22], [0.22, 0.05], [0.12, 0.2], [0, 0.24]], 4, fm.clone().multiply(M.at(0, 0, 0, 0, Math.PI / 4, 0, [1, 1, 0.12])), WHITE, { hard: true });
  }
  // Crown.
  lathe(iron, [[0, -0.02], [0.09, 0.02], [0.08, 0.14], [0, 0.16]], 8, m, WHITE);
  // Ring through the eye of the shank.
  const ring: P3[] = [];
  for (let i = 0; i <= (lod === 0 ? 20 : 10); i++) {
    const a = (i / (lod === 0 ? 20 : 10)) * Math.PI * 2;
    ring.push([0, S + 0.22 + Math.sin(a) * 0.2, Math.cos(a) * 0.2]);
  }
  tube(iron, ring, 0.03, lod === 0 ? 8 : 5, m, WHITE, false);
  // Stock: two oak baulks, tapering to the ends, hooped in iron.
  const stockY = S - 0.12;
  const half = 1.2;
  for (const sz of [-1, 1]) {
    sweep(wood, [[0, stockY, 0], [0, stockY, sz * half]], (k) => (k === 0 ? 0.2 : 0.12), (k) => (k === 0 ? 0.2 : 0.12), () => [0, 1, 0], m, jitter(GREY, mulberry32(sz + 5), 0.1));
    if (lod < 2) for (const d of [0.18, 0.45]) box(iron, 0.21, 0.21, 0.03, m.clone().multiply(M.at(0, stockY, sz * d)), WHITE, 'x', 0);
  }
}

// --------------------------------------------------------------- rope work

/** A flat (Flemish) coil of rope with the end led off: on a jetty or a deck. */
export function buildRopeCoil(pb: PropBuilder, m: THREE.Matrix4, lod: Lod, tarred = false): void {
  const turns = 6;
  const r0 = 0.07;
  const d = 0.04;
  const pts: P3[] = [];
  const per = lod === 0 ? 24 : 12;
  for (let i = 0; i <= turns * per; i++) {
    const a = (i / per) * Math.PI * 2;
    const r = r0 + (d * i) / per;
    pts.push([Math.cos(a) * r, 0.019, Math.sin(a) * r]);
  }
  // The tail, led away across the planking.
  const last = pts[pts.length - 1];
  for (let i = 1; i <= 4; i++) pts.push([last[0] + i * 0.15, 0.019, last[2] + i * 0.05 * Math.sin(i)]);
  tube(pb.kit('rope'), pts, 0.019, lod === 0 ? 6 : 4, m, tarred ? TARRED : HEMP, true);
}

// ---------------------------------------------------------------- flotsam

/** A hatch grating adrift: frame and half-lapped battens. Waterline at y = 0. */
export function buildGrating(pb: PropBuilder, m: THREE.Matrix4, lod: Lod, seed: number): void {
  const rng = mulberry32(seed);
  const wood = pb.kit('wood');
  const S = 1.15;
  for (const s of [-1, 1]) {
    box(wood, S, 0.08, 0.09, m.clone().multiply(M.at(0, -0.07, (s * (S - 0.09)) / 2)), jitter(GREY, rng, 0.1), 'x', lod === 0 ? 0.008 : 0);
    box(wood, 0.09, 0.08, S - 0.18, m.clone().multiply(M.at((s * (S - 0.09)) / 2, -0.07, 0)), jitter(GREY, rng, 0.1), 'z', lod === 0 ? 0.008 : 0);
  }
  const bars = lod === 2 ? 4 : 8;
  for (let i = 0; i < bars; i++) {
    const p = -S / 2 + 0.09 + ((S - 0.18) * (i + 0.5)) / bars;
    box(wood, S - 0.18, 0.03, 0.04, m.clone().multiply(M.at(0, -0.05, p)), jitter(GREY, rng, 0.12), 'x', 0);
    box(wood, 0.04, 0.03, S - 0.18, m.clone().multiply(M.at(p, -0.075, 0)), jitter(GREY, rng, 0.12), 'z', 0);
  }
}

/** Planks and a broken spar lashed together: wreckage that stays together. */
export function buildWreckage(pb: PropBuilder, m: THREE.Matrix4, lod: Lod, seed: number): void {
  const rng = mulberry32(seed);
  const wood = pb.kit('wood');
  const count = 4;
  for (let i = 0; i < count; i++) {
    const len = 1.8 + rng() * 1.4;
    const z = (i - (count - 1) / 2) * 0.27 + (rng() - 0.5) * 0.06;
    box(wood, len, 0.05, 0.24, m.clone().multiply(M.at((rng() - 0.5) * 0.6, -0.1, z, 0, (rng() - 0.5) * 0.18, 0)), jitter(rng() < 0.5 ? GREY : PINE, rng, 0.12), 'x', lod === 0 ? 0.01 : 0, rng() * 10);
  }
  // The spar, its broken end splintered (a cone).
  const sm = m.clone().multiply(M.at(0.1, -0.09, -0.62, 0, 0.08, Math.PI / 2));
  lathe(wood, [[0, -1.9], [0.12, -1.88], [0.12, 1.5], [0.08, 1.75], [0.02, 1.95]], lod === 0 ? 12 : 7, sm, jitter(PINE, rng, 0.1));
  if (lod < 2) {
    for (const x of [-0.7, 0.75]) {
      const loop: P3[] = [];
      for (let i = 0; i <= 12; i++) {
        const a = (i / 12) * Math.PI * 2;
        loop.push([x, -0.06 + Math.sin(a) * 0.16, -0.25 + Math.cos(a) * 0.5]);
      }
      tube(pb.kit('rope'), loop, 0.014, 5, m, HEMP, false);
    }
  }
}

/** A cask adrift with its staves stove in, lying on its side, awash. */
export function buildStoveCask(pb: PropBuilder, m: THREE.Matrix4, lod: Lod, seed: number): void {
  const p = CASK_PRESETS.hogshead;
  buildCask(pb, m.clone().multiply(M.at(0, -p.bilge * 0.55, p.height / 2, -Math.PI / 2, 0, 0)), { ...p, staves: 18, hoops: 'mixed', stoveIn: true, lod, seed });
}
