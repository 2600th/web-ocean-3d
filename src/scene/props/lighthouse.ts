import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  float,
  length,
  normalize,
  positionGeometry,
  cameraViewMatrix,
  vec3,
  vec4,
} from 'three/tsl';
import { M, PropBuilder, box, cylinder, jitter, lathe, srgb, tube, type MeshKit, type Tint } from './kit';
import { beaconLevel } from './materials';
import { buildCask, CASK_PRESETS } from './cask';
import { mulberry32 } from '../../core/random';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;
const v3 = vec3 as unknown as (x: unknown, y?: unknown, z?: unknown) => Node;

/**
 * A light tower as it would have been built about 1700.
 *
 * Period practice decides almost everything here:
 *  - **The light is fixed.** No revolving apparatus existed until Carlsten,
 *    1781; the Argand lamp and parabolic reflector arrived in the 1780s and the
 *    Fresnel lens lit Cordouan in 1823. Lights of this date were coal fires in
 *    iron baskets (Isle of May, 1636) or candles and oil lamps in a glazed
 *    lantern (Winstanley's Eddystone, lit 1698, with candles; Boston Light, 1716,
 *    with oil). This one is the glazed kind: a hoop of tallow candles in an
 *    iron-framed octagonal lantern, which throws a steady, soft glow and no beam.
 *  - **A stout rubble tower.** Coursed rubble, as wide at the foot as it is a
 *    third of its height and battered about one in fifteen — St Agnes (Scilly,
 *    1680) is 22 m over a broad base and North Foreland (1691) was a squat
 *    octagon. Not the slender ashlar column of the nineteenth century. Lime
 *    washed for a day-mark, the wash worn and streaked, the foot dark with
 *    splash and damp (see `masonry` in `materials.ts`).
 *  - **A gallery on stone corbels**, stepped out in three courses under a
 *    flagged walk and railed with wrought iron, so the keeper can clean the
 *    soot off the glass from outside.
 *  - **The lantern is an iron cage**: an iron curb, eight iron posts with
 *    glazing bars holding small panes of crown glass, a domed lead cap and a
 *    vent to draw the candle smoke.
 *  - **A keeper's cottage** at its foot, single storey, shingled, with a stack.
 *
 * Built at true size, origin at ground level at the centre of the tower, the
 * door facing local -Z (landward) and the seaward face +Z.
 */

const SHAFT_BASE = 0.9;
const R_BASE = 3.7;
const R_TOP = 2.8;
const GALLERY_Y = 13.6;
const SLAB = 0.24;
const SHAFT_TOP = GALLERY_Y - SLAB;
const GALLERY_R = 3.55;
const LANTERN_R = 1.75;
const LANTERN_BASE = GALLERY_Y;
const CURB = 0.45;
const GLAZING = 2.0;

/** Height of the lamp hoop above the tower's ground level, metres. */
export const LIGHTHOUSE_LAMP_Y = LANTERN_BASE + CURB + 0.85;
/** Ground radius the lighthouse and its cottage occupy; trees keep clear of it. */
export const LIGHTHOUSE_CLEARING = 15;

const LIME: Tint = [1, 1, 1];
/** Unwashed dressed stone: the plinth, corbels, gallery flags, door and slit dressings. */
const BARE: Tint = [0.85, 0.82, 0.76];
const DOOR: Tint = srgb(0.36, 0.24, 0.15);
const OPENING: Tint = [0.03, 0.028, 0.026];

function shaftRadius(y: number): number {
  const t = Math.min(1, Math.max(0, (y - SHAFT_BASE) / (SHAFT_TOP - SHAFT_BASE)));
  // Slightly concave batter: the foot flares, as masonry towers were built.
  return R_TOP + (R_BASE - R_TOP) * Math.pow(1 - t, 1.25);
}

/**
 * A hand-built wall is not a lathe: a percent of slow bulge round the shaft
 * and up it, so the silhouette and the light across it are not machined.
 * Integer harmonics in angle, so the seam closes.
 */
function shaftBulge(a: number, k: number): number {
  return 1 + 0.009 * Math.sin(3 * a + k * 1.7) + 0.006 * Math.sin(7 * a + k * 0.9 + 1.3);
}

export function buildLighthouse(pb: PropBuilder, m: THREE.Matrix4, lod: 0 | 1 | 2): void {
  const rng = mulberry32(0x11647);
  const stone = pb.kit('masonry');
  const iron = pb.kit('iron');
  const wood = pb.kit('wood');
  const lead = pb.kit('lead');
  const seg = lod === 0 ? 48 : lod === 1 ? 28 : 16;

  // Plinth: two stepped footing courses of unwashed stone, buried below grade.
  lathe(
    stone,
    [
      [R_BASE + 0.5, -2.0],
      [R_BASE + 0.5, 0.42],
      [R_BASE + 0.28, 0.48],
      [R_BASE + 0.28, SHAFT_BASE - 0.05],
      [R_BASE + 0.02, SHAFT_BASE + 0.02],
      [0, SHAFT_BASE + 0.02],
    ],
    seg,
    m,
    BARE,
    { hard: true },
  );

  // Shaft.
  const prof: [number, number][] = [];
  const rows = lod === 0 ? 16 : lod === 1 ? 8 : 4;
  for (let i = 0; i <= rows; i++) {
    const y = SHAFT_BASE + ((SHAFT_TOP - SHAFT_BASE) * i) / rows;
    prof.push([shaftRadius(y), y]);
  }
  lathe(stone, prof, seg, m, LIME, { radius: lod < 2 ? shaftBulge : undefined });

  // Gallery: a ring of flags on corbels stepped out in three courses.
  lathe(
    stone,
    [
      [R_TOP - 0.2, SHAFT_TOP],
      [GALLERY_R, SHAFT_TOP],
      [GALLERY_R + 0.02, SHAFT_TOP + 0.04],
      [GALLERY_R + 0.02, GALLERY_Y],
      [0, GALLERY_Y],
    ],
    seg,
    m,
    BARE,
    { hard: true },
  );
  if (lod < 2) {
    const n = lod === 0 ? 24 : 16;
    // Each corbel: three blocks, each course projecting further than the one
    // below, the top one carrying the flag's edge.
    const tiers: [number, number][] = lod === 0
      ? [
          [0.72, 0.27],
          [0.5, 0.27],
          [0.28, 0.27],
        ]
      : [
          [0.72, 0.36],
          [0.4, 0.36],
        ];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + Math.PI / n;
      let y = SHAFT_TOP;
      for (const [reach, th] of tiers) {
        y -= th;
        const r0 = shaftRadius(y);
        const d = reach + 0.15;
        const r = r0 - 0.15 + d / 2;
        box(stone, 0.38, th - 0.015, d, m.clone().multiply(M.at(Math.cos(a) * r, y + th / 2, Math.sin(a) * r, 0, Math.PI / 2 - a, 0)), jitter(BARE, rng, 0.04), 'z', lod === 0 ? 0.02 : 0);
      }
    }
  } else {
    // Far away the corbel table is a dark flared band.
    lathe(stone, [[shaftRadius(SHAFT_TOP - 0.8), SHAFT_TOP - 0.8], [GALLERY_R - 0.05, SHAFT_TOP]], seg, m, BARE);
  }

  // Door (landward) with a dressed-stone surround, and slit windows up the stair.
  {
    const r = shaftRadius(1.0);
    const dm = m.clone().multiply(M.at(0, SHAFT_BASE, -r + 0.05, 0, Math.PI, 0));
    box(wood, 1.0, 2.0, 0.1, dm.clone().multiply(M.at(0, 1.0, -0.12)), DOOR, 'y', lod === 0 ? 0.01 : 0);
    if (lod === 0) {
      for (let i = 0; i < 5; i++) box(wood, 0.19, 1.98, 0.03, dm.clone().multiply(M.at(-0.4 + i * 0.2, 1.0, -0.06)), jitter(DOOR, rng, 0.15), 'y', 0.006);
      for (const y of [0.35, 1.6]) box(iron, 0.9, 0.05, 0.012, dm.clone().multiply(M.at(0.0, y, -0.04)), [1, 1, 1], 'x', 0);
    }
    // Surround: jambs and a heavy lintel, unwashed.
    for (const sx of [-1, 1]) box(stone, 0.34, 2.3, 0.45, dm.clone().multiply(M.at(sx * 0.67, 1.15, 0.0)), BARE, 'y', lod === 0 ? 0.03 : 0);
    box(stone, 1.7, 0.4, 0.47, dm.clone().multiply(M.at(0, 2.47, 0)), BARE, 'x', lod === 0 ? 0.03 : 0);
    box(stone, 1.7, 0.25, 0.75, dm.clone().multiply(M.at(0, -0.1, 0.25)), BARE, 'x', 0);
  }
  if (lod < 2) {
    const slits = [
      [3.6, 0.5],
      [6.2, 2.3],
      [8.8, 4.1],
      [11.4, 5.6],
    ];
    for (const [y, a] of slits) {
      const r = shaftRadius(y);
      const wm = m.clone().multiply(M.at(Math.sin(a) * r, y, Math.cos(a) * r, 0, a, 0));
      box(wood, 0.24, 0.8, 0.06, wm, OPENING, 'y', 0);
      box(stone, 0.55, 0.16, 0.14, wm.clone().multiply(M.at(0, 0.48, 0.02)), BARE, 'x', 0);
      box(stone, 0.55, 0.12, 0.16, wm.clone().multiply(M.at(0, -0.46, 0.03)), BARE, 'x', 0);
      for (const sx of [-1, 1]) box(stone, 0.14, 0.82, 0.12, wm.clone().multiply(M.at(sx * 0.2, 0, 0.02)), BARE, 'y', 0);
    }
  }

  // Gallery railing: wrought-iron standards, a top rail and a middle rail.
  {
    const posts = lod === 0 ? 32 : lod === 1 ? 16 : 0;
    const rr = GALLERY_R - 0.12;
    for (let i = 0; i < posts; i++) {
      const a = (i / posts) * Math.PI * 2;
      cylinder(iron, 0.022, 0.026, 1.0, 5, m.clone().multiply(M.at(Math.cos(a) * rr, GALLERY_Y + 0.5, Math.sin(a) * rr)), [1, 1, 1], false);
    }
    const ringPts = (y: number, n: number): [number, number, number][] => {
      const out: [number, number, number][] = [];
      for (let i = 0; i <= n; i++) {
        const a = (i / n) * Math.PI * 2;
        out.push([Math.cos(a) * rr, y, Math.sin(a) * rr]);
      }
      return out;
    };
    const ringSeg = lod === 0 ? 56 : lod === 1 ? 28 : 16;
    tube(iron, ringPts(GALLERY_Y + 1.0, ringSeg), 0.03, lod === 2 ? 4 : 6, m, [1, 1, 1], false);
    if (lod < 2) tube(iron, ringPts(GALLERY_Y + 0.5, ringSeg), 0.018, 5, m, [1, 1, 1], false);
  }

  // ---- lantern: an iron cage on an iron curb, glazed, under a lead dome.
  const sides = 8;
  const lm = m.clone().multiply(M.at(0, LANTERN_BASE, 0, 0, Math.PI / 8, 0));
  const g0 = CURB;
  const g1 = CURB + GLAZING;
  // The curb: plated iron, solid to waist height, on a stone step.
  lathe(stone, [[LANTERN_R + 0.3, 0], [LANTERN_R + 0.3, 0.12], [0, 0.12]], sides, lm, BARE, { hard: true });
  lathe(iron, [[LANTERN_R + 0.04, 0.1], [LANTERN_R + 0.04, g0], [LANTERN_R - 0.06, g0], [0, g0]], sides, lm, [1, 1, 1], { hard: true });
  const glass = pb.kit('beaconGlass');
  const apothem = LANTERN_R * Math.cos(Math.PI / sides);
  const faceW = 2 * LANTERN_R * Math.sin(Math.PI / sides);
  for (let i = 0; i < sides; i++) {
    const a0 = (i / sides) * Math.PI * 2;
    const a1 = ((i + 1) / sides) * Math.PI * 2;
    const am = (a0 + a1) / 2;
    const pm = lm.clone().multiply(M.at(Math.cos(am) * (apothem - 0.02), (g0 + g1) / 2, Math.sin(am) * (apothem - 0.02), 0, -am + Math.PI / 2, 0));
    box(glass, faceW - 0.06, GLAZING, 0.02, pm, [1, 1, 1], 'y', 0);
    // Corner post, standing from the curb foot to the dome's ring.
    const cm = lm.clone().multiply(M.at(Math.cos(a0) * LANTERN_R, (0.1 + g1 + 0.1) / 2, Math.sin(a0) * LANTERN_R, 0, -a0, 0));
    box(iron, 0.11, g1, 0.11, cm, [1, 1, 1], 'y', 0);
    if (lod < 2) {
      // Glazing bars: small panes, as crown glass came — four tiers by two.
      const bars = lod === 0 ? 3 : 1;
      for (let b = 1; b <= bars; b++) {
        const y = g0 + (GLAZING * b) / (bars + 1);
        box(iron, faceW, 0.035, 0.04, pm.clone().multiply(M.at(0, y - (g0 + g1) / 2, 0.02)), [1, 1, 1], 'x', 0);
      }
      box(iron, 0.035, GLAZING, 0.04, pm.clone().multiply(M.at(0, 0, 0.02)), [1, 1, 1], 'y', 0);
    }
    if (lod === 0) {
      // Rivet heads down the curb plates either side of each post.
      for (const s of [-1, 1]) {
        for (let r = 0; r < 3; r++) {
          const a = a0 + s * 0.05;
          const y = 0.17 + r * 0.12;
          cylinder(iron, 0.014, 0.014, 0.02, 5, lm.clone().multiply(M.at(Math.cos(a) * (LANTERN_R + 0.045), y, Math.sin(a) * (LANTERN_R + 0.045), 0, -a, Math.PI / 2)), [1, 1, 1]);
        }
      }
    }
  }
  // Sill and head rings.
  lathe(iron, [[LANTERN_R + 0.07, g0 - 0.02], [LANTERN_R + 0.07, g0 + 0.06], [LANTERN_R - 0.05, g0 + 0.06]], sides, lm, [1, 1, 1], { hard: true });
  lathe(iron, [[LANTERN_R - 0.05, g1 - 0.08], [LANTERN_R + 0.12, g1 - 0.08], [LANTERN_R + 0.12, g1 + 0.06], [LANTERN_R - 0.05, g1 + 0.06]], sides, lm, [1, 1, 1], { hard: true });

  // Domed cap: lead over a timber frame, with iron ribs at the posts.
  const domeR = LANTERN_R + 0.1;
  const domeH = 1.15;
  const ventR = 0.2;
  const domeBase = g1 + 0.06;
  const domePts: [number, number][] = [[LANTERN_R + 0.25, domeBase], [LANTERN_R + 0.25, domeBase + 0.08]];
  const steps = lod === 0 ? 9 : lod === 1 ? 5 : 3;
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * Math.PI * 0.5;
    const r = Math.max(ventR, domeR * Math.cos(t));
    domePts.push([r, domeBase + 0.08 + domeH * Math.sin(t)]);
  }
  const domeSeg = lod === 0 ? 32 : lod === 1 ? 16 : 8;
  lathe(lead, domePts, domeSeg, lm, [1, 1, 1]);
  const domeTop = domeBase + 0.08 + domeH;
  if (lod === 0) {
    for (let i = 0; i < sides; i++) {
      const a = (i / sides) * Math.PI * 2;
      const rib: [number, number, number][] = [];
      for (let k = 0; k <= 8; k++) {
        const t = (k / 8) * Math.PI * 0.5;
        const r = Math.max(ventR, domeR * Math.cos(t)) + 0.02;
        rib.push([Math.cos(a) * r, domeBase + 0.08 + domeH * Math.sin(t) + 0.01, Math.sin(a) * r]);
      }
      tube(iron, rib, 0.022, 4, lm, [1, 1, 1], false);
    }
  }
  // The vent: a short stack under a conical hat, raised clear of it on
  // four stays so the smoke gets out and the rain does not get in.
  cylinder(lead, ventR, ventR, 0.42, lod === 0 ? 12 : 8, lm.clone().multiply(M.at(0, domeTop + 0.18, 0)), [1, 1, 1], false);
  const hatY = domeTop + 0.5;
  lathe(lead, [[0, hatY], [0.42, hatY], [0.44, hatY + 0.03], [0, hatY + 0.32]], lod === 0 ? 14 : 8, lm, [1, 1, 1], { hard: true });
  if (lod < 2) {
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
      cylinder(iron, 0.012, 0.012, 0.14, 4, lm.clone().multiply(M.at(Math.cos(a) * 0.3, hatY - 0.07, Math.sin(a) * 0.3)), [1, 1, 1], false);
    }
  }
  // Finial ball and the vane.
  lathe(lead, [[0, hatY + 0.28], [0.08, hatY + 0.33], [0.09, hatY + 0.4], [0.06, hatY + 0.46], [0, hatY + 0.48]], lod === 0 ? 10 : 6, lm, [1, 1, 1]);
  cylinder(iron, 0.022, 0.03, 1.0, 5, lm.clone().multiply(M.at(0, hatY + 0.95, 0)), [1, 1, 1], false);
  if (lod < 2) box(iron, 0.55, 0.28, 0.012, lm.clone().multiply(M.at(0.3, hatY + 1.2, 0, 0, 0.4, 0)), [1, 1, 1], 'x', 0);

  // The lamp: a hoop of candles hung from the roof, and the soot-black hoop itself.
  const flame = pb.kit('flame');
  const candles = lod === 0 ? 16 : lod === 1 ? 8 : 4;
  const hoopY = LIGHTHOUSE_LAMP_Y - LANTERN_BASE - 0.12;
  for (let i = 0; i < candles; i++) {
    const a = (i / candles) * Math.PI * 2;
    const r = 0.55;
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    if (lod < 2) cylinder(pb.kit('wood'), 0.025, 0.025, 0.22, 5, lm.clone().multiply(M.at(x, hoopY + 0.11, z)), [0.85, 0.8, 0.66], true);
    lathe(flame, [[0.0, 0], [0.028, 0.04], [0.02, 0.11], [0, 0.17]], 5, lm.clone().multiply(M.at(x, hoopY + 0.23, z)), [1, 1, 1]);
  }
  if (lod < 2) {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      pts.push([Math.cos(a) * 0.55, hoopY, Math.sin(a) * 0.55]);
    }
    tube(iron, pts, 0.02, 4, lm, [1, 1, 1], false);
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2;
      tube(iron, [[Math.cos(a) * 0.55, hoopY, Math.sin(a) * 0.55], [0, g1 - 0.05, 0]], 0.008, 3, lm, [1, 1, 1], false);
    }
  }
}

/**
 * The keeper's cottage: lime-washed rubble, one room, a shingled gable roof
 * with a stack at the landward end, a plank door and a shuttered window.
 * Origin at the centre of its footprint on the ground; the door faces +Z.
 */
export function buildKeepersHut(pb: PropBuilder, m: THREE.Matrix4, lod: 0 | 1 | 2): void {
  const rng = mulberry32(0x4e11);
  const stone = pb.kit('masonry');
  const wood = pb.kit('wood');
  const W = 5.6;
  const Dp = 4.0;
  const H = 2.6;
  const pitch = 0.75;
  const ridge = H + (Dp / 2) * Math.tan(pitch);
  // Walls, footed below grade.
  for (const sz of [-1, 1]) box(stone, W, H + 1.2, 0.55, m.clone().multiply(M.at(0, (H - 1.2) / 2, sz * (Dp / 2 - 0.275))), LIME, 'y', lod === 0 ? 0.04 : 0);
  for (const sx of [-1, 1]) {
    box(stone, 0.55, H + 1.2, Dp - 1.1, m.clone().multiply(M.at(sx * (W / 2 - 0.275), (H - 1.2) / 2, 0)), LIME, 'y', lod === 0 ? 0.04 : 0);
    // Gable triangle.
    gable(stone, m.clone().multiply(M.at(sx * (W / 2 - 0.275), H, 0)), Dp, ridge - H, sx);
  }
  // Roof: two shingled planes with an overhang, and a ridge board.
  const shingle = pb.kit('shingle');
  const slope = Dp / 2 / Math.cos(pitch) + 0.35;
  for (const sz of [-1, 1]) {
    const rm = m.clone().multiply(M.at(0, ridge + 0.08, 0, sz * pitch, 0, 0)).multiply(M.at(0, 0, sz * (slope / 2 - 0.05)));
    box(shingle, W + 0.6, 0.08, slope, rm, [1, 1, 1], 'z', 0, 0);
  }
  box(wood, W + 0.6, 0.12, 0.18, m.clone().multiply(M.at(0, ridge + 0.12, 0)), [0.28, 0.24, 0.2], 'x', 0);
  // Chimney stack on the landward gable.
  box(stone, 0.8, ridge + 1.0, 0.8, m.clone().multiply(M.at(-W / 2 + 0.6, (ridge + 1.0) / 2, 0)), LIME, 'y', lod === 0 ? 0.03 : 0);
  box(stone, 0.95, 0.15, 0.95, m.clone().multiply(M.at(-W / 2 + 0.6, ridge + 1.05, 0)), BARE, 'y', 0);
  box(wood, 0.5, 0.1, 0.5, m.clone().multiply(M.at(-W / 2 + 0.6, ridge + 1.1, 0)), OPENING, 'y', 0);
  // Door and window on the front.
  const front = Dp / 2;
  box(wood, 0.95, 1.95, 0.08, m.clone().multiply(M.at(0.9, 0.975, front - 0.02)), DOOR, 'y', 0);
  box(stone, 1.25, 0.25, 0.62, m.clone().multiply(M.at(0.9, 2.08, front - 0.27)), BARE, 'x', 0);
  box(wood, 0.8, 0.8, 0.06, m.clone().multiply(M.at(-1.2, 1.5, front + 0.01)), OPENING, 'y', 0);
  if (lod < 2) {
    for (const sx of [-1, 1]) box(wood, 0.42, 0.82, 0.04, m.clone().multiply(M.at(-1.2 + sx * 0.64, 1.5, front + 0.04, 0, sx * 0.5, 0)), jitter(DOOR, rng, 0.2), 'y', 0);
    box(stone, 1.0, 0.12, 0.66, m.clone().multiply(M.at(-1.2, 1.04, front - 0.25)), BARE, 'x', 0);
  }
  // Oil and tallow for the lamp, in casks by the wall.
  if (lod < 2) {
    buildCask(pb, m.clone().multiply(M.at(2.2, 0, front + 0.6)), { ...CASK_PRESETS.barrel, staves: 14, hoops: 'mixed', lod: lod === 0 ? 1 : 2, seed: 71 });
    buildCask(pb, m.clone().multiply(M.at(2.0, 0.24, front + 1.4, Math.PI / 2, 0.3, 0)), { ...CASK_PRESETS.barrel, staves: 14, hoops: 'wood', lod: 2, seed: 72 });
  }
}

/** A triangular gable wall in the YZ plane, `span` wide at its foot, `rise` tall. */
function gable(kit: MeshKit, m: THREE.Matrix4, span: number, rise: number, facing: number): void {
  kit.withMatrix(m, () => {
    for (const side of [-1, 1]) {
      const x = side * 0.275;
      const n = side;
      const a = kit.vertex(x, 0, -span / 2, n, 0, 0, -span / 2, 0, LIME);
      const b = kit.vertex(x, 0, span / 2, n, 0, 0, span / 2, 0, LIME);
      const c = kit.vertex(x, rise, 0, n, 0, 0, 0, rise, LIME);
      if (side > 0) kit.tri(a, c, b);
      else kit.tri(a, b, c);
    }
  });
  void facing;
}

/**
 * The lamp's glow seen from afar: a camera-facing card at the lantern, sized in
 * proportion to its distance so it stays a few pixels across from the play
 * area 1.4 km out, where the lantern itself is under one. Additive and HDR, so
 * the bloom draws its halo; zero by day.
 */
export function buildBeaconHalo(position: THREE.Vector3): THREE.Mesh {
  const g = new THREE.BufferGeometry();
  const pos: number[] = [];
  const corner: number[] = [];
  for (const [cx, cy] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ]) {
    pos.push(position.x, position.y, position.z);
    corner.push(cx, cy);
  }
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('corner', new THREE.Float32BufferAttribute(corner, 2));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.boundingSphere = new THREE.Sphere(position.clone(), 40);

  const mat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  mat.name = 'prop-beacon-halo';
  // Additive, so scene fog must stay off: fog mixes the card's black corners
  // toward the fog colour, and blending adds that, which drew a pale square
  // round the lighthouse at dusk.
  mat.fog = false;
  mat.userData.noGroundShading = true;
  const c = attribute('corner', 'vec2') as Node;
  const centre = positionGeometry as Node;
  const toEye = (cameraPosition as Node).sub(centre);
  const dist = length(toEye);
  mat.positionNode = Fn(() => {
    const view = cameraViewMatrix as Node;
    const right = v3(view.element(0).x, view.element(1).x, view.element(2).x);
    const up = v3(view.element(0).y, view.element(1).y, view.element(2).y);
    // 1.2 % of the distance: about eight pixels across at 1.4 km with the 55-degree lens.
    const size = dist.mul(0.012).max(1.8);
    // Stood off toward the eye, so the lantern's own glass does not clip it.
    const p = centre.add(normalize(toEye).mul(2.2));
    return p.add(right.mul(c.x.mul(size))).add(up.mul(c.y.mul(size)));
  })();
  mat.colorNode = Fn(() => {
    const r = c.length();
    const core = float(1).sub(r.smoothstep(0, 1)).pow(3);
    // Weak up close, where the lit glass carries it; full from a couple of hundred metres.
    const far = dist.smoothstep(40, 260).mul(0.85).add(0.15);
    return vec4(v3(1.0, 0.62, 0.3).mul(core).mul(beaconLevel).mul(far).mul(5), 1);
  })();
  const mesh = new THREE.Mesh(g, mat);
  mesh.name = 'lighthouse-halo';
  mesh.frustumCulled = true;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.renderOrder = 3;
  return mesh;
}
