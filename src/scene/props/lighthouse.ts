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
 *    octagonal lantern, which throws a steady, soft glow and no beam.
 *  - **The tower is rubble masonry, battered and whitewashed.** Thick at the
 *    foot, tapering about one in fifteen — St Agnes (Scilly, 1680) and North
 *    Foreland (1691) are the shape. Lime wash made the tower a day-mark,
 *    which mattered as much as the light did.
 *  - **A gallery on corbels**, railed with wrought iron, so the keeper can
 *    clean the soot off the glass from outside.
 *  - **A keeper's cottage** at its foot, single storey, shingled, with a stack.
 *
 * Built at true size, origin at ground level at the centre of the tower, the
 * door facing local -Z (landward) and the seaward face +Z.
 */

/** Height of the lamp hoop above the tower's ground level, metres. */
export const LIGHTHOUSE_LAMP_Y = 17.85;
/** Ground radius the lighthouse and its cottage occupy; trees keep clear of it. */
export const LIGHTHOUSE_CLEARING = 15;

const SHAFT_BASE = 0.9;
const SHAFT_TOP = 15.7;
const R_BASE = 3.0;
const R_TOP = 2.25;
const GALLERY_Y = 16.2;
const GALLERY_R = 3.0;
const LANTERN_R = 1.55;
const LANTERN_BASE = 16.45;
const DWARF_WALL = 0.55;
const GLAZING = 2.3;

const LIME: Tint = [1, 1, 1];
const BARE: Tint = [0.75, 0.72, 0.66];
const DOOR: Tint = srgb(0.36, 0.24, 0.15);
const OPENING: Tint = [0.03, 0.028, 0.026];

function shaftRadius(y: number): number {
  const t = Math.min(1, Math.max(0, (y - SHAFT_BASE) / (SHAFT_TOP - SHAFT_BASE)));
  // Slightly concave batter: the foot flares, as masonry towers were built.
  return R_TOP + (R_BASE - R_TOP) * Math.pow(1 - t, 1.25);
}

export function buildLighthouse(pb: PropBuilder, m: THREE.Matrix4, lod: 0 | 1 | 2): void {
  const rng = mulberry32(0x11647);
  const stone = pb.kit('masonry');
  const iron = pb.kit('iron');
  const wood = pb.kit('wood');
  const seg = lod === 0 ? 40 : lod === 1 ? 24 : 14;

  // Plinth: an unwashed footing course, buried below grade.
  lathe(stone, [[R_BASE + 0.35, -2.0], [R_BASE + 0.35, 0.55], [R_BASE + 0.05, SHAFT_BASE], [0, SHAFT_BASE]], seg, m, BARE, { hard: true });

  // Shaft.
  const prof: [number, number][] = [];
  const rows = lod === 0 ? 10 : 5;
  for (let i = 0; i <= rows; i++) {
    const y = SHAFT_BASE + ((SHAFT_TOP - SHAFT_BASE) * i) / rows;
    prof.push([shaftRadius(y), y]);
  }
  lathe(stone, prof, seg, m, LIME);
  // String course at mid-height: the one horizontal line on the shaft.
  if (lod < 2) {
    const y = 8.4;
    const r = shaftRadius(y);
    lathe(stone, [[r - 0.02, y - 0.12], [r + 0.08, y - 0.1], [r + 0.08, y + 0.08], [r - 0.02, y + 0.12]], seg, m, LIME, { hard: true });
  }
  // Corbelled cornice carrying the gallery.
  lathe(
    stone,
    [
      [R_TOP, SHAFT_TOP],
      [R_TOP + 0.15, SHAFT_TOP + 0.15],
      [GALLERY_R - 0.25, GALLERY_Y - 0.25],
      [GALLERY_R, GALLERY_Y - 0.05],
      [GALLERY_R, GALLERY_Y + 0.12],
      [0, GALLERY_Y + 0.12],
    ],
    seg,
    m,
    LIME,
    { hard: true },
  );
  if (lod === 0) {
    // Individual corbel blocks under the cornice.
    const n = 20;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const r = R_TOP + 0.22;
      box(stone, 0.32, 0.36, 0.22, m.clone().multiply(M.at(Math.cos(a) * r, SHAFT_TOP - 0.05, Math.sin(a) * r, 0, -a + Math.PI / 2, 0)), jitter(LIME, rng, 0.04), 'y', 0.02);
    }
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
    for (const sx of [-1, 1]) box(stone, 0.3, 2.3, 0.4, dm.clone().multiply(M.at(sx * 0.65, 1.15, 0.0)), BARE, 'y', lod === 0 ? 0.03 : 0);
    box(stone, 1.6, 0.35, 0.42, dm.clone().multiply(M.at(0, 2.45, 0)), BARE, 'x', lod === 0 ? 0.03 : 0);
    // Step.
    box(stone, 1.6, 0.25, 0.7, dm.clone().multiply(M.at(0, -0.1, 0.25)), BARE, 'x', 0);
  }
  if (lod < 2) {
    const slits = [
      [4.2, 0.5],
      [7.4, 2.3],
      [10.6, 4.1],
      [13.4, 5.6],
    ];
    for (const [y, a] of slits) {
      const r = shaftRadius(y);
      const wm = m.clone().multiply(M.at(Math.sin(a) * r, y, Math.cos(a) * r, 0, a, 0));
      box(wood, 0.24, 0.8, 0.06, wm, OPENING, 'y', 0);
      box(stone, 0.5, 0.14, 0.12, wm.clone().multiply(M.at(0, 0.47, 0.02)), BARE, 'x', 0);
      box(stone, 0.5, 0.12, 0.14, wm.clone().multiply(M.at(0, -0.46, 0.03)), BARE, 'x', 0);
    }
  }

  // Gallery railing.
  {
    const posts = lod === 0 ? 28 : lod === 1 ? 14 : 0;
    const rr = GALLERY_R - 0.12;
    for (let i = 0; i < posts; i++) {
      const a = (i / posts) * Math.PI * 2;
      cylinder(iron, 0.022, 0.025, 1.0, 5, m.clone().multiply(M.at(Math.cos(a) * rr, GALLERY_Y + 0.62, Math.sin(a) * rr)), [1, 1, 1], false);
    }
    const ringPts = (y: number, n: number): [number, number, number][] => {
      const out: [number, number, number][] = [];
      for (let i = 0; i <= n; i++) {
        const a = (i / n) * Math.PI * 2;
        out.push([Math.cos(a) * rr, y, Math.sin(a) * rr]);
      }
      return out;
    };
    const ringSeg = lod === 0 ? 48 : lod === 1 ? 24 : 16;
    tube(iron, ringPts(GALLERY_Y + 1.12, ringSeg), 0.03, lod === 2 ? 4 : 6, m, [1, 1, 1], false);
    if (lod < 2) tube(iron, ringPts(GALLERY_Y + 0.62, ringSeg), 0.018, 5, m, [1, 1, 1], false);
  }

  // Lantern: dwarf wall, glazing, roof.
  const sides = 8;
  const lm = m.clone().multiply(M.at(0, LANTERN_BASE, 0, 0, Math.PI / 8, 0));
  lathe(stone, [[LANTERN_R + 0.12, 0], [LANTERN_R + 0.12, DWARF_WALL], [0.0, DWARF_WALL]], sides, lm, LIME, { hard: true });
  const glass = pb.kit('beaconGlass');
  const lead = pb.kit('lead');
  const g0 = DWARF_WALL;
  const g1 = DWARF_WALL + GLAZING;
  for (let i = 0; i < sides; i++) {
    const a0 = (i / sides) * Math.PI * 2;
    const a1 = ((i + 1) / sides) * Math.PI * 2;
    const am = (a0 + a1) / 2;
    const apothem = LANTERN_R * Math.cos(Math.PI / sides);
    const faceW = 2 * LANTERN_R * Math.sin(Math.PI / sides);
    // Pane, set just inside the mullions.
    const pm = lm.clone().multiply(M.at(Math.cos(am) * (apothem - 0.02), (g0 + g1) / 2, Math.sin(am) * (apothem - 0.02), 0, -am + Math.PI / 2, 0));
    box(glass, faceW - 0.04, GLAZING, 0.02, pm, [1, 1, 1], 'y', 0);
    // Mullion at the corner.
    const cm = lm.clone().multiply(M.at(Math.cos(a0) * LANTERN_R, (g0 + g1) / 2, Math.sin(a0) * LANTERN_R, 0, -a0, 0));
    box(iron, 0.08, GLAZING, 0.08, cm, [1, 1, 1], 'y', 0);
    if (lod < 2) {
      // Glazing bars: small panes, as crown glass came.
      const bars = lod === 0 ? 3 : 1;
      for (let b = 1; b <= bars; b++) {
        const y = g0 + (GLAZING * b) / (bars + 1);
        box(iron, faceW, 0.03, 0.035, pm.clone().multiply(M.at(0, y - (g0 + g1) / 2, 0.02)), [1, 1, 1], 'x', 0);
      }
      box(iron, 0.03, GLAZING, 0.035, pm.clone().multiply(M.at(0, 0, 0.02)), [1, 1, 1], 'y', 0);
    }
  }
  // Sill and head rings.
  lathe(iron, [[LANTERN_R + 0.06, g0 - 0.02], [LANTERN_R + 0.06, g0 + 0.06], [LANTERN_R - 0.05, g0 + 0.06]], sides, lm, [1, 1, 1], { hard: true });
  lathe(iron, [[LANTERN_R - 0.05, g1 - 0.08], [LANTERN_R + 0.1, g1 - 0.08], [LANTERN_R + 0.1, g1 + 0.04]], sides, lm, [1, 1, 1], { hard: true });
  // Roof: an ogee cupola in lead, a ventilator ball and a vane.
  lathe(
    lead,
    [
      [LANTERN_R + 0.22, g1 + 0.02],
      [LANTERN_R + 0.1, g1 + 0.22],
      [LANTERN_R * 0.75, g1 + 0.55],
      [LANTERN_R * 0.38, g1 + 0.9],
      [0.16, g1 + 1.12],
      [0, g1 + 1.16],
    ],
    sides,
    lm,
    [1, 1, 1],
    { hard: lod > 0 },
  );
  lathe(lead, [[0.0, g1 + 1.1], [0.22, g1 + 1.22], [0.24, g1 + 1.36], [0.18, g1 + 1.5], [0, g1 + 1.56]], lod === 0 ? 12 : 6, lm, [1, 1, 1]);
  cylinder(iron, 0.025, 0.035, 1.2, 5, lm.clone().multiply(M.at(0, g1 + 2.1, 0)), [1, 1, 1], false);
  if (lod < 2) box(iron, 0.55, 0.28, 0.012, lm.clone().multiply(M.at(0.3, g1 + 2.45, 0, 0, 0.4, 0)), [1, 1, 1], 'x', 0);

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
 * The keeper's cottage: whitewashed rubble, one room, a shingled gable roof
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
