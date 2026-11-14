import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';
import { ISLAND, seafloorHeight, seafloorMeshHeight } from '../Seafloor';
import {
  DetailClock,
  PartBuilder,
  buildCrate,
  buildLog,
  buildSmoke,
  createClothMaterial,
  createDecalMaterial,
  createEmberMaterial,
  createWoodMaterial,
  tileQuadUv,
} from './parts';
import { buildArch, buildRock, buildSkullRock, createRockMaterial } from './rock';

export { applyTideline } from './rock';

/**
 * The island's set dressing: what turns terrain with plants on it into a place.
 *
 *  - **Landmarks** that read from a kilometre out and give the silhouette
 *    something other than a hill: a skull rock in the shallows, a sea arch off
 *    the headland with stacks around it, and crags on the summit and the
 *    shoulder peak.
 *  - **Outcrops** of stratified rock on the steep flanks.
 *  - **The beach**: a line of wrack along the high-tide mark, shell and pebble
 *    litter, and driftwood above it.
 *  - **A camp** behind the cove's jetty: a canvas lean-to, a fire with smoke,
 *    crates, log seats and a ladder — someone was here last week.
 *
 * Every piece is generated, and everything that shares a material is merged
 * into one geometry or one instanced mesh: the whole of this is seven draws.
 * The rock, wood and canvas carry their own colour graphs; `Wetness` wraps
 * them, and `applyGroundShadingTo` gives them contact shade, cloud shadow and
 * caustics like every other prop. The fire and smoke opt out of the latter.
 */

export interface CoveFrame {
  /** Bearing of the cove from the island centre, radians. */
  bearing: number;
  /** Shoreline radius on that bearing, metres. */
  shore: number;
}

const UP = new THREE.Vector3(0, 1, 0);

/**
 * Where the island is mostly seen from: out past the cove, on the bearing the
 * ship's approach and the opening shots look in along. Landmarks turn their
 * faces here.
 */
const VIEWPOINT = new THREE.Vector2(ISLAND.x + Math.cos(0.6) * 1500, ISLAND.z + Math.sin(0.6) * 1500);

/** Bisects for the shoreline radius on a bearing. */
function shoreRadius(bearing: number): number {
  const dx = Math.cos(bearing);
  const dz = Math.sin(bearing);
  let inner = ISLAND.radius * 0.4;
  let outer = ISLAND.radius * 1.7;
  for (let i = 0; i < 22; i++) {
    const mid = (inner + outer) / 2;
    if (seafloorHeight(ISLAND.x + dx * mid, ISLAND.z + dz * mid) > 0) inner = mid;
    else outer = mid;
  }
  return (inner + outer) / 2;
}

/** Radius on a bearing where the ground first falls to `y`, searching inward from the shore. */
function radiusAtHeight(bearing: number, y: number, shore: number): number | null {
  const dx = Math.cos(bearing);
  const dz = Math.sin(bearing);
  const h = (r: number) => seafloorHeight(ISLAND.x + dx * r, ISLAND.z + dz * r);
  let outer = shore;
  let inner = shore - 120;
  if (h(inner) < y) return null;
  for (let i = 0; i < 20; i++) {
    const mid = (inner + outer) / 2;
    if (h(mid) > y) inner = mid;
    else outer = mid;
  }
  return (inner + outer) / 2;
}

function groundNormal(x: number, z: number, span: number, out: THREE.Vector3): THREE.Vector3 {
  const hx = seafloorHeight(x + span, z) - seafloorHeight(x - span, z);
  const hz = seafloorHeight(x, z + span) - seafloorHeight(x, z - span);
  return out.set(-hx, 2 * span, -hz).normalize();
}

function onBearing(bearing: number, radius: number): THREE.Vector3 {
  const x = ISLAND.x + Math.cos(bearing) * radius;
  const z = ISLAND.z + Math.sin(bearing) * radius;
  return new THREE.Vector3(x, seafloorHeight(x, z), z);
}

/** Yaw that turns local +z toward world (dx, dz). */
function yawToward(dx: number, dz: number): number {
  return Math.atan2(dx, dz);
}

export class IslandDetail {
  readonly object = new THREE.Group();
  private readonly clock = new DetailClock();
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly materials: THREE.Material[] = [];

  constructor(cove: CoveFrame, seed = 0x1d7a11) {
    this.object.name = 'island-detail';
    const rng = mulberry32(seed);

    const rock = createRockMaterial();
    const wood = createWoodMaterial();
    const cloth = createClothMaterial();
    const decal = createDecalMaterial();
    const ember = createEmberMaterial(this.clock);
    this.materials.push(rock, wood, cloth, decal, ember);

    const rocks = new PartBuilder();
    this.landmarks(rocks, rng);
    this.outcrops(rocks, rng);
    const woodParts = new PartBuilder();
    const clothParts = new PartBuilder();
    const emberParts = new PartBuilder();
    this.camp(cove, rocks, woodParts, clothParts, emberParts, rng);

    this.mesh(rocks, rock, 'island-landmarks', true);
    this.mesh(woodParts, wood, 'camp-wood', true);
    this.mesh(clothParts, cloth, 'camp-canvas', true);
    this.mesh(emberParts, ember, 'camp-embers', false);

    const decals = new PartBuilder();
    this.beachDecals(decals, rng);
    const decalMesh = this.mesh(decals, decal, 'beach-decals', false);
    if (decalMesh) decalMesh.receiveShadow = true;

    this.driftwood(wood, rng);
  }

  /** Smoke and flicker run on this; call beside `FoliageWind.update`. */
  update(dt: number): void {
    this.clock.update(dt);
  }

  resetClock(time = 0): void {
    this.clock.reset(time);
  }

  setWind(direction: THREE.Vector2, strength: number): void {
    this.clock.setWind(direction, strength);
  }

  /** Where the fire's coals and the torch flames are, world space. Filled by `camp`. */
  readonly lightAnchors: { fire: THREE.Vector3; torches: THREE.Vector3[] } = {
    fire: new THREE.Vector3(),
    torches: [],
  };

  /**
   * How the camp smoke is lit.
   *
   * The plume was unlit — a fixed grey, so at night it stood over a dark beach
   * as a bright white column. `ambient` is the light the sky and the key give a
   * diffuse grey card, normalised so a clear noon is about 1; `fire` is 0..1,
   * the camp fire's glow on the underside of the plume once it is dark enough
   * for the fire to matter.
   */
  setLighting(ambient: THREE.Color, fire: number): void {
    (this.clock.uSmokeLight.value as THREE.Color).copy(ambient);
    this.clock.uFireGlow.value = fire;
  }

  dispose(): void {
    this.object.removeFromParent();
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
  }

  // ------------------------------------------------------------- internals

  private mesh(parts: PartBuilder, material: THREE.Material, name: string, casts: boolean): THREE.Mesh | null {
    if (parts.empty) return null;
    const g = parts.build();
    this.geometries.push(g);
    const mesh = new THREE.Mesh(g, material);
    mesh.name = name;
    mesh.castShadow = casts;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    this.object.add(mesh);
    return mesh;
  }

  /**
   * Landmarks, placed on the faces the play area and the approach see: the
   * cove looks out on a bearing of about 0.6, so the skull sits to its right
   * and the arch on the headland to its left.
   */
  private landmarks(out: PartBuilder, rng: () => number): void {
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3(1, 1, 1);

    // Skull rock, in the shallows off the beach, staring out at the approach.
    {
      const bearing = 0.12;
      const r = shoreRadius(bearing) + 18;
      const p = onBearing(bearing, r);
      const height = 46;
      const g = buildSkullRock(0x5c011, height);
      q.setFromEuler(new THREE.Euler(-0.06, yawToward(VIEWPOINT.x - p.x, VIEWPOINT.y - p.z), 0.04, 'YXZ'));
      // Seated with a fifth of it below the floor, so it stands in the water.
      m.compose(new THREE.Vector3(p.x, p.y + height * 0.3, p.z), q, s);
      out.add(g, m);
      g.dispose();
    }

    // Sea arch off the headland, its opening turned to the approach so the
    // light shows through it, with stacks around it.
    {
      const bearing = 1.78;
      const r = shoreRadius(bearing) + 40;
      const p = onBearing(bearing, r);
      const g = buildArch(0xa2c4, 66, 44, 11);
      q.setFromAxisAngle(UP, yawToward(VIEWPOINT.x - p.x, VIEWPOINT.y - p.z));
      m.compose(new THREE.Vector3(p.x, Math.min(p.y, -1) - 3, p.z), q, s);
      out.add(g, m);
      g.dispose();
      const stacks: [number, number, number][] = [
        [1.6, 26, 40],
        [1.93, 48, 30],
        [2.02, 22, 20],
        [1.48, 58, 22],
        [1.7, 80, 16],
      ];
      for (const [b, off, h] of stacks) {
        const sp = onBearing(b, shoreRadius(b) + off);
        const sg = buildRock({
          detail: 18,
          size: [h * 0.26, h * 0.6, h * 0.24],
          rough: h * 0.05,
          roughScale: h * 0.2,
          terrace: h * 0.07,
          terraceStrength: 0.5,
          base: 0.7,
          taper: 0.35,
          seed: Math.floor(rng() * 1e6),
        });
        q.setFromEuler(new THREE.Euler((rng() - 0.5) * 0.12, rng() * 6.28, (rng() - 0.5) * 0.12));
        m.compose(new THREE.Vector3(sp.x, Math.min(sp.y, 0) + h * 0.3, sp.z), q, s);
        out.add(sg, m);
        sg.dispose();
      }
    }

    // Crags on the summit and on the shoulder peak, so the crest is a broken
    // skyline rather than the top of a dome: a plug of tall spires at the top
    // and a smaller cluster on the shoulder.
    const summit = new THREE.Vector3(ISLAND.x + Math.cos(3.9) * 80, 0, ISLAND.z + Math.sin(3.9) * 80);
    const shoulder = new THREE.Vector3(
      ISLAND.x + Math.cos(1.25) * ISLAND.radius * 0.46,
      0,
      ISLAND.z + Math.sin(1.25) * ISLAND.radius * 0.46,
    );
    const crags: [THREE.Vector3, number, number, number][] = [
      [summit, 6, 66, 34],
      [shoulder, 4, 38, 22],
    ];
    for (const [centre, n, tall, spread] of crags) {
      for (let i = 0; i < n; i++) {
        const a = rng() * 6.28;
        const d = i === 0 ? 0 : spread * (0.45 + rng() * 0.7);
        const x = centre.x + Math.cos(a) * d;
        const z = centre.z + Math.sin(a) * d;
        const h = tall * (i === 0 ? 1 : 0.4 + rng() * 0.4);
        const g = buildRock({
          detail: 18,
          size: [h * 0.22, h * 0.6, h * 0.19],
          rough: h * 0.06,
          roughScale: h * 0.2,
          terrace: h * 0.06,
          terraceStrength: 0.55,
          base: 0.6,
          taper: 0.55,
          seed: Math.floor(rng() * 1e6),
        });
        q.setFromEuler(new THREE.Euler((rng() - 0.5) * 0.22, rng() * 6.28, (rng() - 0.5) * 0.22));
        m.compose(new THREE.Vector3(x, seafloorHeight(x, z) + h * 0.2, z), q, s);
        out.add(g, m);
        g.dispose();
      }
    }
  }

  /** Stratified outcrops where the ground is steep, mostly on the faces seen from the sea. */
  private outcrops(out: PartBuilder, rng: () => number): void {
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3(1, 1, 1);
    const n = new THREE.Vector3();
    const variants = [0, 1, 2].map((i) =>
      buildRock({
        detail: 9,
        size: [1, 0.6, 0.8],
        rough: 0.12,
        roughScale: 0.35,
        terrace: 0.14,
        terraceStrength: 0.65,
        base: 0.4,
        seed: 0x0c0 + i * 97,
      }),
    );
    let placed = 0;
    for (let attempt = 0; attempt < 1400 && placed < 64; attempt++) {
      const bearing = rng() * Math.PI * 2;
      const radius = ISLAND.radius * (0.15 + rng() * 0.75);
      const x = ISLAND.x + Math.cos(bearing) * radius;
      const z = ISLAND.z + Math.sin(bearing) * radius;
      const y = seafloorHeight(x, z);
      if (y < 12) continue;
      groundNormal(x, z, 8, n);
      if (n.y > 0.86) continue;
      const k = 9 + rng() * 14;
      // Bedded with the slope: the outcrop's up tips toward the ground normal.
      q.setFromUnitVectors(UP, n.clone().lerp(UP, 0.5).normalize());
      q.multiply(new THREE.Quaternion().setFromAxisAngle(UP, rng() * 6.28));
      m.compose(new THREE.Vector3(x, y - k * 0.22, z), q, s.set(k, k * (0.7 + rng() * 0.6), k));
      out.add(variants[placed % 3], m);
      placed++;
    }
    for (const v of variants) v.dispose();
  }

  /** Wrack along the high-tide line and shell litter below and above it. */
  private beachDecals(out: PartBuilder, rng: () => number): void {
    const n = new THREE.Vector3();
    const place = (tile: 'wrack' | 'litter', bearing: number, y: number, len: number, wid: number, along: boolean) => {
      const shore = shoreRadius(bearing);
      const r = radiusAtHeight(bearing, y, shore);
      if (r === null) return;
      const cx = ISLAND.x + Math.cos(bearing) * r;
      const cz = ISLAND.z + Math.sin(bearing) * r;
      groundNormal(cx, cz, 4, n);
      // A cliff or a steep bank is not a beach.
      if (n.y < 0.95) return;
      const tangent = new THREE.Vector3(-Math.sin(bearing), 0, Math.cos(bearing));
      const yaw = along ? (rng() - 0.5) * 0.4 : rng() * 6.28;
      tangent.applyAxisAngle(UP, yaw);
      const side = new THREE.Vector3().crossVectors(UP, tangent).normalize();
      const corner = (a: number, b: number) => {
        const x = cx + tangent.x * a * len + side.x * b * wid;
        const z = cz + tangent.z * a * len + side.z * b * wid;
        return new THREE.Vector3(x, seafloorMeshHeight(x, z) + 0.03, z);
      };
      const tint: [number, number, number] = tile === 'wrack' ? [0.9 + rng() * 0.2, 0.9, 0.85] : [1, 1, 1];
      out.quad(
        // Wound so the face points up: `side` is to the tangent's right.
        [corner(-0.5, -0.5), corner(0.5, -0.5), corner(0.5, 0.5), corner(-0.5, 0.5)],
        tileQuadUv(tile, rng() < 0.5),
        tint,
        n.clone(),
      );
    };
    for (let i = 0; i < 2400; i++) {
      const b = rng() * Math.PI * 2;
      place('wrack', b, 1.55 + rng() * 0.5, 3 + rng() * 3, 1 + rng() * 0.8, true);
    }
    for (let i = 0; i < 900; i++) {
      const b = rng() * Math.PI * 2;
      place('litter', b, 0.5 + rng() * 3, 1.4 + rng() * 0.8, 0.5 + rng() * 0.3, false);
    }
  }

  /** Bleached logs above the tide line, one instanced draw. */
  private driftwood(material: THREE.Material, rng: () => number): void {
    const g = buildLog(0xd71f, 1, 0.1, 3);
    this.geometries.push(g);
    const count = 70;
    const mesh = new THREE.InstancedMesh(g, material, count);
    mesh.name = 'beach-driftwood';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const n = new THREE.Vector3();
    let placed = 0;
    const bleached = new THREE.Color();
    for (let attempt = 0; attempt < 400 && placed < count; attempt++) {
      const bearing = rng() * Math.PI * 2;
      const shore = shoreRadius(bearing);
      const r = radiusAtHeight(bearing, 1.7 + rng() * 1.6, shore);
      if (r === null) continue;
      const x = ISLAND.x + Math.cos(bearing) * r;
      const z = ISLAND.z + Math.sin(bearing) * r;
      groundNormal(x, z, 3, n);
      if (n.y < 0.95) continue;
      const len = 3 + rng() * 6;
      const thick = 0.7 + rng() * 0.8;
      // Lying roughly along the shore, as the tide leaves them.
      const yaw = yawToward(-Math.sin(bearing), Math.cos(bearing)) - Math.PI / 2 + (rng() - 0.5) * 1.4;
      q.setFromUnitVectors(UP, n);
      q.multiply(new THREE.Quaternion().setFromAxisAngle(UP, yaw));
      m.compose(
        new THREE.Vector3(x, seafloorMeshHeight(x, z) - 0.04 * thick, z),
        q,
        new THREE.Vector3(len, thick * len * 0.12, thick * len * 0.12).multiplyScalar(1),
      );
      // Keep the cross-section round: scale y and z together, x is length.
      m.compose(
        new THREE.Vector3(x, seafloorMeshHeight(x, z) + 0.02, z),
        q,
        new THREE.Vector3(len, thick * 1.6, thick * 1.6),
      );
      mesh.setMatrixAt(placed, m);
      const k = 0.85 + rng() * 0.35;
      bleached.setRGB(k * 1.12, k * 1.08, k);
      mesh.setColorAt(placed, bleached);
      placed++;
    }
    mesh.count = placed;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    this.object.add(mesh);
  }

  /**
   * The camp behind the jetty. In the cove's own frame: `in` metres inland of
   * the shoreline (negative is inland), `along` metres along it.
   */
  private camp(
    cove: CoveFrame,
    stones: PartBuilder,
    wood: PartBuilder,
    cloth: PartBuilder,
    ember: PartBuilder,
    rng: () => number,
  ): void {
    const at = (inland: number, along: number): THREE.Vector3 => {
      const r = cove.shore + inland;
      const b = cove.bearing + along / r;
      return onBearing(b, r);
    };
    const seaward = new THREE.Vector3(Math.cos(cove.bearing), 0, Math.sin(cove.bearing));
    const alongDir = new THREE.Vector3(-seaward.z, 0, seaward.x);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const one = new THREE.Vector3(1, 1, 1);
    const faceSea = yawToward(seaward.x, seaward.z);

    // --- the fire ---------------------------------------------------------
    const fire = at(-42, 9);
    const stone = buildRock({ detail: 3, size: [0.32, 0.22, 0.28], rough: 0.05, roughScale: 0.2, seed: 0x570 });
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 + rng() * 0.2;
      const p = new THREE.Vector3(fire.x + Math.cos(a) * 0.95, 0, fire.z + Math.sin(a) * 0.95);
      p.y = seafloorMeshHeight(p.x, p.z) + 0.05;
      q.setFromEuler(new THREE.Euler(rng() * 0.4, rng() * 6.28, rng() * 0.4));
      m.compose(p, q, one.clone().multiplyScalar(0.8 + rng() * 0.5));
      stones.add(stone, m);
    }
    stone.dispose();
    const fireY = seafloorMeshHeight(fire.x, fire.z);
    this.lightAnchors.fire.set(fire.x, fireY + 0.55, fire.z);
    const logG = buildLog(0xf12e, 1, 0.07, 0);
    for (let i = 0; i < 6; i++) {
      // A collapsed teepee: logs leaning in to a point above the coals.
      const a = (i / 6) * Math.PI * 2 + 0.3;
      const base = new THREE.Vector3(fire.x + Math.cos(a) * 0.7, fireY + 0.05, fire.z + Math.sin(a) * 0.7);
      const dir = new THREE.Vector3(fire.x, fireY + 0.75, fire.z).sub(base).normalize();
      q.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);
      m.compose(base, q, new THREE.Vector3(1.05, 1.4, 1.4));
      wood.add(logG, m, [0.42, 0.36, 0.3]);
    }
    // Coals and flame tongues: vertex colour red channel is "how hot".
    const coals = new THREE.SphereGeometry(0.62, 10, 5, 0, Math.PI * 2, 0, Math.PI / 2);
    m.compose(new THREE.Vector3(fire.x, fireY - 0.02, fire.z), q.identity(), new THREE.Vector3(1, 0.28, 1));
    ember.add(coals, m, [0.35, 0, 0]);
    coals.dispose();
    for (let i = 0; i < 5; i++) {
      const cone = new THREE.ConeGeometry(0.16 + rng() * 0.08, 0.7 + rng() * 0.5, 6);
      cone.translate(0, 0.35, 0);
      const a = rng() * 6.28;
      m.compose(
        new THREE.Vector3(fire.x + Math.cos(a) * 0.22, fireY + 0.02, fire.z + Math.sin(a) * 0.22),
        q.setFromEuler(new THREE.Euler((rng() - 0.5) * 0.4, 0, (rng() - 0.5) * 0.4)),
        one,
      );
      ember.add(cone, m, [0.95, 0, 0]);
      cone.dispose();
    }
    this.object.add(buildSmoke(this.clock, new THREE.Vector3(fire.x, fireY + 1.0, fire.z), 34, 26));

    // Log seats round the fire.
    for (const [a, len] of [
      [0.6, 3.2],
      [2.5, 2.8],
    ] as const) {
      const p = new THREE.Vector3(fire.x + Math.cos(a + faceSea) * 2.6, 0, fire.z + Math.sin(a + faceSea) * 2.6);
      p.y = seafloorMeshHeight(p.x, p.z) + 0.05;
      q.setFromAxisAngle(UP, -(a + faceSea) + Math.PI / 2);
      m.compose(p.addScaledVector(alongDir, -len / 2), q, new THREE.Vector3(len, 2.8, 2.8));
      wood.add(logG, m, [0.5, 0.42, 0.34]);
    }

    // --- the tent ------------------------------------------------------
    // An A-frame: two uprights and a ridge pole, the canvas pegged to the
    // ground either side and sagging between its supports. The open end faces
    // the fire and the sea, so from the water it reads as a tent, not a wall.
    const shelter = at(-49, 11);
    const sy = seafloorMeshHeight(shelter.x, shelter.z);
    const ridgeH = 2.5;
    const halfW = 2.1;
    const halfLen = 2.3;
    const pole = new THREE.CylinderGeometry(0.06, 0.08, 1, 6);
    pole.translate(0, 0.5, 0);
    const polePlace = (base: THREE.Vector3, top: THREE.Vector3, tint: [number, number, number] = [0.6, 0.48, 0.36]) => {
      const d = top.clone().sub(base);
      q.setFromUnitVectors(UP, d.clone().normalize());
      m.compose(base, q, new THREE.Vector3(1, d.length(), 1));
      wood.add(pole, m, tint);
    };
    const front = shelter.clone().addScaledVector(seaward, halfLen);
    const back = shelter.clone().addScaledVector(seaward, -halfLen);
    front.y = seafloorMeshHeight(front.x, front.z) - 0.2;
    back.y = seafloorMeshHeight(back.x, back.z) - 0.2;
    const frontTop = front.clone().setY(sy + ridgeH);
    const backTop = back.clone().setY(sy + ridgeH);
    polePlace(front, frontTop);
    polePlace(back, backTop);
    polePlace(backTop.clone().addScaledVector(seaward, -0.35), frontTop.clone().addScaledVector(seaward, 0.35));
    // Canvas: rows run along the ridge, columns over it from one side's peg
    // line to the other's.
    const rows = 6;
    const cols = 10;
    const grid: THREE.Vector3[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const row: THREE.Vector3[] = [];
      for (let j = 0; j <= cols; j++) {
        const u = (j / cols) * 2 - 1;
        const p = backTop.clone().lerp(frontTop, t).addScaledVector(alongDir, u * halfW);
        const ground = seafloorMeshHeight(p.x, p.z);
        const a = Math.abs(u);
        const sag = Math.sin(Math.PI * t) * Math.sin(Math.PI * a) * 0.22;
        p.y = THREE.MathUtils.lerp(sy + ridgeH, ground + 0.05, a) - sag;
        row.push(p);
      }
      grid.push(row);
    }
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const patch = (i * 7 + j * 3) % 11 === 0 ? 0.82 : 1;
        cloth.quad(
          [grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]],
          [
            [j / cols, i / rows],
            [(j + 1) / cols, i / rows],
            [(j + 1) / cols, (i + 1) / rows],
            [j / cols, (i + 1) / rows],
          ],
          [patch, patch, patch * 0.96],
        );
      }
    }
    // A bedroll inside, along the ridge.
    const roll = new THREE.CylinderGeometry(0.22, 0.22, 1.8, 8);
    const rollAt = shelter.clone().addScaledVector(alongDir, 0.7);
    rollAt.y = seafloorMeshHeight(rollAt.x, rollAt.z) + 0.2;
    q.setFromUnitVectors(UP, seaward);
    m.compose(rollAt, q, one);
    cloth.add(roll, m, [0.55, 0.36, 0.26]);
    roll.dispose();

    // --- crates, a ladder ----------------------------------------------------
    const crate = buildCrate(0.9);
    for (const [inland, along, k, spin] of [
      [-46, 16.5, 1, 0.2],
      [-46.6, 17.6, 0.85, 0.9],
      [-45.4, 17.3, 0.75, 0.5],
    ] as const) {
      const p = at(inland, along);
      p.y = seafloorMeshHeight(p.x, p.z);
      q.setFromAxisAngle(UP, faceSea + spin);
      m.compose(p, q, one.clone().multiplyScalar(k));
      wood.add(crate, m, [0.62, 0.5, 0.38]);
    }
    // Stack a fourth on top of the first.
    {
      const p = at(-46, 16.5);
      p.y = seafloorMeshHeight(p.x, p.z) + 0.72;
      q.setFromAxisAngle(UP, faceSea + 0.6);
      m.compose(p, q, one.clone().multiplyScalar(0.8));
      wood.add(crate, m, [0.55, 0.46, 0.36]);
    }
    crate.dispose();
    {
      // Ladder against the end of the ridge pole.
      const foot = backTop.clone().addScaledVector(seaward, -1.5).addScaledVector(alongDir, 0.2);
      foot.y = seafloorMeshHeight(foot.x, foot.z);
      const top = backTop.clone().addScaledVector(seaward, -0.3).add(new THREE.Vector3(0, 0.15, 0));
      for (const off of [-0.25, 0.25]) {
        polePlace(foot.clone().addScaledVector(alongDir, off), top.clone().addScaledVector(alongDir, off), [0.55, 0.45, 0.34]);
      }
      for (let i = 1; i < 6; i++) {
        const t = i / 6.2;
        const c = foot.clone().lerp(top, t);
        polePlace(c.clone().addScaledVector(alongDir, -0.27), c.clone().addScaledVector(alongDir, 0.27), [0.5, 0.42, 0.32]);
      }
    }

    // --- torches at the jetty's foot ----------------------------------------
    for (const along of [-3.2, 3.2]) {
      const base = at(-3, along);
      base.y = seafloorMeshHeight(base.x, base.z) - 0.3;
      const top = base.clone().add(new THREE.Vector3(0, 2.6, 0));
      this.lightAnchors.torches.push(top.clone().add(new THREE.Vector3(0, 0.2, 0)));
      polePlace(base, top, [0.45, 0.36, 0.28]);
      const flame = new THREE.ConeGeometry(0.1, 0.34, 6);
      flame.translate(0, 0.17, 0);
      m.compose(top, q.identity(), one);
      ember.add(flame, m, [0.9, 0, 0]);
      flame.dispose();
    }
    pole.dispose();
    logG.dispose();
  }
}
