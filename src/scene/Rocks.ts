import * as THREE from 'three/webgpu';
import { mulberry32 } from '../core/random';
import { ISLAND, reefPatches, REEF_BAND, seafloorPlacementHeight, seafloorPlacementMeshHeight } from './Seafloor';
import { applyGroundShading, type GroundShadingInputs } from './groundShading';
import { InstanceField } from './reef/InstanceField';
import {
  buildChunkGeometry,
  createRockMaterial,
  gridSampler,
  smax,
  smin,
  smoothstep,
  surfaceNets,
  vnoise3,
  type ChunkKind,
} from './reef/RockKit';

/**
 * The seabed's geology: sculpted outcrops, arches and swim-throughs, the talus
 * at their feet, boulder fields, and the rubble strewn across the shelf.
 *
 * **Why.** The heightfield is a smooth function sampled every 15.6 m — it can
 * hold a slope and nothing that stands up from one. Diving the island shelf
 * found sand falling away into blue with nothing on it: no scale, no parallax,
 * no shadow, nothing for the light to fall across. The heightfield is also
 * shared with buoyancy, the shoreline and the water's depth tint, so it is the
 * wrong place to add a cliff. Geometry on top of it is the right one.
 *
 * **Outcrops are sculpted, then instanced.** A library of sixteen outcrops is
 * sculpted at boot — each a small density field: a column or a ridge with a
 * flared base, a waist and an overhanging brow, domain-warped, cut by strata
 * ledges and roughened by noise, sometimes bored through by a tunnel or bent
 * into an arch — and meshed by surface nets at two voxel sizes for two LODs.
 * That is abyssal-garden's terrain recipe (`W_DENS` in `shaders.js`: segment
 * columns with overhang and waist terms, smooth unions, relief and strata)
 * scaled down to one outcrop per grid and moved to the CPU so the WebGL2
 * fallback gets it too. Each sculpt carries a six-metre skirt below its base,
 * so a placement can be seated on the lowest floor under its footprint and the
 * uphill side simply buries more of it — nothing floats on a slope.
 *
 * Sculpting every placement separately was the first version, and it was the
 * wrong trade: ~5 ms of noise per outcrop held the shelf to forty-five of them,
 * one per 200 m, against 45-70 m of visibility — a diver saw none. Instancing a
 * library puts one in sight almost everywhere on the slope for the same boot
 * cost, and the world-space strata and colour beds of the rock material mean
 * two placements of one sculpt do not read as copies.
 *
 * **Loose rock is instanced too.** Four clipped-icosphere shapes (`RockKit`)
 * at four tessellations carry the talus, the boulder fields and the rubble.
 * Everything is dealt by distance over size (`InstanceField`), so a handful of
 * draws covers the whole shelf and only what is in visibility range costs
 * vertices.
 *
 * **Clear of the surface and the ship.** Every placement is scaled down until
 * its top is at least four metres under the surface, nine over the plateau
 * near the ship's water, and nothing tall stands within 40 m of the origin.
 *
 * **What it hands on.** `gardens` — every outcrop as a reef site, with points
 * on its ledges and tops for the reef to grow on — and `influence`, a map the
 * seabed reads to lay scree and darker substrate around each one.
 */

const ROCK_SEED = 0x7a11e5;

/** Above this eye height the geology is hidden outright, as `ReefLife` is. */
const HIDE_ABOVE = 90;

/** Nothing reaches closer to the surface than this, metres. */
const SURFACE_CLEARANCE = 4;
/**
 * Over the plateau the ship sails above, clearance is deeper. Its draft is
 * under four metres and swell troughs about two; nine leaves room for both and
 * for the chase camera. Past `LANE_OUTER` the open plateau takes six.
 */
const LANE_CLEARANCE = 9;
const LANE_OUTER = 130;
const PLATEAU_CLEARANCE = 6;
/** Radius around the origin kept free of outcrops entirely: the spawn and the ship. */
const LANE_RADIUS = 40;

/** Voxel edge for the near and far outcrop meshes, metres. */
const VOXEL_NEAR = 0.85;
const VOXEL_FAR = 2.0;
/** Every sculpt extends this far below its base, for seating on slopes. */
const SKIRT = 6;

export type GardenTheme = 'coral' | 'sponge' | 'fan';

/** An outcrop as a reef site, with places on it for the reef to grow. */
export interface RockGarden {
  x: number;
  z: number;
  /** Horizontal reach of the rock, metres. */
  radius: number;
  /** Water depth at its base. */
  depth: number;
  theme: GardenTheme;
  region: 'plateau' | 'island';
  /** Points on the rock facing up or out: x, y, z, nx, ny, nz per anchor. */
  anchors: Float32Array;
}

/** A map of the geology over the seabed, for the sand's shading. */
export interface RockInfluence {
  texture: THREE.DataTexture;
  minX: number;
  minZ: number;
  size: number;
}

type FormationKind = 'mesa' | 'ridge' | 'spires' | 'arch';

/** One sculpt, in its own frame: base at y = 0, ridges and arches along +x. */
interface Formation {
  kind: FormationKind;
  radius: number;
  /** Ridge half-length, or arch major radius. */
  length: number;
  height: number;
  overhang: number;
  waist: number;
  /** Arch tube radius. */
  minor: number;
  tunnel: number;
  spires: { x: number; z: number; r: number; h: number }[];
  shoulder: { x: number; z: number; r: number; h: number } | null;
  noise: number;
  /** Horizontal reach from the centre. */
  reach: number;
}

interface Sculpt {
  formation: Formation;
  field: InstanceField;
  /** Highest vertex, metres above the base, at scale 1. */
  top: number;
  reach: number;
  /** Local anchors, x y z nx ny nz. */
  anchors: Float32Array;
}

interface Placement {
  sculpt: Sculpt;
  x: number;
  z: number;
  base: number;
  yaw: number;
  scale: number;
  stretch: number;
  region: 'plateau' | 'island';
}

export interface RocksOptions {
  /** Ground shading inputs: caustics, the cloud deck's shade, contact darkening. */
  shading?: Omit<GroundShadingInputs, 'foliage'>;
}

export class Rocks {
  readonly object = new THREE.Group();
  readonly gardens: RockGarden[] = [];
  readonly influence: RockInfluence;

  private readonly material: THREE.MeshStandardNodeMaterial;
  private readonly sculpts: Sculpt[] = [];
  private readonly placements: Placement[] = [];
  /** Loose rock fields; the outcrop fields live on `sculpts`. */
  private readonly fields: InstanceField[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private detail = 1;
  private cameraAllows = true;
  private wantVisible = true;
  private disposed = false;

  /** Milliseconds the constructor took: sculpting, layout and the map. */
  readonly buildMs: number;
  /** [underwater-look-3] Round 10. The loose shapes' LOD chains, for `addKelpBedStones`. */
  private readonly looseLevels: THREE.BufferGeometry[][] = [];
  private kelpBedDone = false;

  constructor(detail: number, options: RocksOptions = {}) {
    const started = performance.now();
    this.object.name = 'seabed-rocks';
    this.material = createRockMaterial('seabed-rock');
    if (options.shading) applyGroundShading(this.material, options.shading);

    const random = mulberry32(ROCK_SEED);
    this.buildLibrary(random);
    this.placeOutcrops(random);
    for (const s of this.sculpts) {
      s.field.seal();
      for (const mesh of s.field.meshes) this.object.add(mesh);
    }
    this.buildLooseRock(random);
    this.influence = buildInfluence(this.placements, this.fieldSites);
    this.setDetail(detail);
    this.buildMs = performance.now() - started;
  }

  /**
   * [underwater-look-3] Round 10. Stones in the kelp beds: a few cobbles and
   * a boulder or two at every holdfast — giant kelp settles on hard ground,
   * and a holdfast on bare sand read as a pole stuck in it — and a sparse
   * scatter over the open floor between the stands, so the bed under a forest
   * is not one flat sheet of sand. Its own fields sharing the loose shapes and
   * the material (the others are sealed by now), seeded from the columns so
   * nothing else moves. Seated as the other loose rock is, on the lowest
   * rendered floor under the footprint.
   */
  addKelpBedStones(
    columns: readonly { x: number; y: number; z: number; height: number }[],
    forests: readonly { x: number; z: number; radius: number }[],
  ): void {
    if (this.kelpBedDone || this.disposed) return;
    this.kelpBedDone = true;
    const random = mulberry32(ROCK_SEED ^ 0x6b31d);
    const fields = this.looseLevels.map(
      (levels, i) =>
        new InstanceField({
          name: `kelp-bed-stones-${i}`,
          levels,
          material: this.material,
          // Small stones: full detail to 10 m, 320 triangles to 28 m.
          switches: [10, 28, 60],
          cullBase: 24,
          cullPerSize: 34,
          cullMax: 110,
          refresh: 4,
        }),
    );
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const euler = new THREE.Euler();
    const color = new THREE.Color();
    const place = (x: number, z: number, size: number, rank: number, bury: number): void => {
      const r = random();
      const kind = r < 0.36 ? 0 : r < 0.7 ? 1 : r < 0.9 ? 2 : 3;
      const field = fields[kind];
      const sy = size * (kind === 3 ? 0.7 + random() * 0.4 : 0.55 + random() * 0.5);
      const sx = size * (0.85 + random() * 0.35);
      const sz = size * (0.85 + random() * 0.35);
      let floor = seafloorPlacementMeshHeight(x, z);
      const reach = Math.max(sx, sz) * 0.7;
      for (let a = 0; a < 4; a++) {
        const ang = (a / 4) * Math.PI * 2 + 0.4;
        floor = Math.min(floor, seafloorPlacementMeshHeight(x + Math.cos(ang) * reach, z + Math.sin(ang) * reach));
      }
      if (floor > -3) return;
      position.set(x, floor - sy * bury, z);
      if (kind === 3) euler.set((random() - 0.5) * 0.5, random() * Math.PI * 2, (random() - 0.5) * 0.5);
      else euler.set(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2);
      quaternion.setFromEuler(euler);
      scale.set(sx, sy, sz);
      matrix.compose(position, quaternion, scale);
      // Paler than the shelf's rubble: limestone cobbles crusted with coralline
      // algae, which the contact shade and the canopy's shadow would otherwise
      // take to black.
      const k = BED_STONE_TINT[0] + random() * (BED_STONE_TINT[1] - BED_STONE_TINT[0]);
      const pink = random() < 0.35 ? 1 : 0;
      color.setRGB(k * (1.0 + 0.06 * pink), k * (0.98 - 0.04 * pink), k * (0.95 + 0.02 * pink));
      field.add(matrix, color, Math.max(sx, sy, sz), rank);
    };
    const size = (lo: number, hi: number, p: number): number => lo + (hi - lo) * Math.pow(random(), p);
    for (const c of columns) {
      // The stone the holdfast grips, half buried beside it, and a few
      // cobbles round it.
      const a0 = random() * Math.PI * 2;
      const big = size(0.45, 1.5, 1.3);
      place(c.x + Math.cos(a0) * (0.35 + big * 0.4), c.z + Math.sin(a0) * (0.35 + big * 0.4), big, random() * 0.3, 0.35);
      const n = 2 + Math.floor(random() * 4);
      for (let i = 0; i < n; i++) {
        const a = random() * Math.PI * 2;
        const r = 0.5 + Math.abs(gauss(random)) * 1.6;
        place(c.x + Math.cos(a) * r, c.z + Math.sin(a) * r, size(0.08, 0.38, 2.2), random(), 0.25);
      }
    }
    for (const f of forests) {
      // Between the stands: gravel spreads and a lone boulder now and then.
      const n = Math.round(f.radius * f.radius * 0.12);
      for (let i = 0; i < n; i++) {
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(random()) * f.radius * 1.15;
        const cx = f.x + Math.cos(a) * r;
        const cz = f.z + Math.sin(a) * r;
        if (random() < 0.3) {
          // A boulder or a small cluster of them: what reads through the haze
          // from across the bed.
          const b = 1 + Math.floor(random() * 3);
          for (let j = 0; j < b; j++) {
            place(cx + gauss(random) * 1.2, cz + gauss(random) * 1.2, size(0.5, 1.8, 1.5), random() * 0.3, 0.3);
          }
          continue;
        }
        const k = 2 + Math.floor(random() * 4);
        for (let j = 0; j < k; j++) {
          place(cx + gauss(random) * 0.9, cz + gauss(random) * 0.9, size(0.07, 0.3, 2.4), random(), 0.25);
        }
      }
    }
    // The bed under a forest is rubble and silt, darker round each stand than
    // between them: the sand's own rubble shading, through the influence map.
    const beds = [
      ...forests.map((f) => ({ x: f.x, z: f.z, r: f.radius * 0.9, k: BED_FOREST })),
      ...columns.map((c) => ({ x: c.x, z: c.z, r: 0.8, k: BED_COLUMN })),
    ];
    const rebuilt = buildInfluence(this.placements, this.fieldSites, beds);
    (this.influence.texture.image.data as Uint8Array).set(rebuilt.texture.image.data as Uint8Array);
    this.influence.texture.needsUpdate = true;
    rebuilt.texture.dispose();

    for (const field of fields) {
      field.seal();
      field.setDetail(this.detail <= 0 ? 0 : 0.25 + 0.75 * this.detail);
      field.setRangeMultiplier(0.7 + 0.3 * this.detail);
      for (const mesh of field.meshes) this.object.add(mesh);
      this.fields.push(field);
    }
  }

  /** Detail fraction 0..1: how much loose rock, and how far the fine meshes reach. */
  setDetail(detail: number): void {
    this.detail = Math.max(0, Math.min(1, detail));
    const range = 0.7 + 0.3 * this.detail;
    // Loose rock thins by rank; structure (big talus) has low ranks and stays.
    for (const field of this.fields) {
      field.setDetail(this.detail <= 0 ? 0 : 0.25 + 0.75 * this.detail);
      field.setRangeMultiplier(range);
    }
    for (const s of this.sculpts) s.field.setRangeMultiplier(range);
  }

  getDetail(): number {
    return this.detail;
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** Deals LODs and culls for this camera. Cheap when the camera has not moved. */
  setCameraPosition(position: THREE.Vector3): void {
    const allows = position.y < HIDE_ABOVE;
    if (allows !== this.cameraAllows) {
      this.cameraAllows = allows;
      this.applyVisibility();
    }
    if (!this.object.visible) return;
    for (const s of this.sculpts) s.field.update(position);
    for (const field of this.fields) field.update(position);
  }

  /** Triangles currently drawn, for the harness. */
  triangleCount(): number {
    if (!this.object.visible) return 0;
    let t = 0;
    for (const s of this.sculpts) t += s.field.triangleCount();
    for (const field of this.fields) t += field.triangleCount();
    return t;
  }

  /** Outcrop placements, for a harness that wants to frame one. */
  outcropCentres(): { x: number; y: number; z: number; kind: string; reach: number; top: number }[] {
    return this.placements.map((p) => ({
      x: p.x,
      y: p.base,
      z: p.z,
      kind: p.sculpt.formation.kind,
      reach: p.sculpt.reach * p.scale,
      top: p.base + p.sculpt.top * p.scale * p.stretch,
    }));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const s of this.sculpts) s.field.dispose();
    for (const field of this.fields) field.dispose();
    for (const g of this.geometries) g.dispose();
    this.material.dispose();
    this.influence.texture.dispose();
    this.object.removeFromParent();
  }

  // ---------------------------------------------------------------- internals

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.cameraAllows;
  }

  /** Sculpts the library: seven mesas (two of them low patch-reef mounds), three ridges, three spire groups, three arches. */
  private buildLibrary(random: () => number): void {
    const kinds: FormationKind[] = [
      'mesa', 'mesa', 'mesa', 'mesa', 'mesa', 'mesa', 'mesa',
      'ridge', 'ridge', 'ridge',
      'spires', 'spires', 'spires',
      'arch', 'arch', 'arch',
    ];
    for (let i = 0; i < kinds.length; i++) {
      const f = makeFormation(kinds[i], random, i);
      const near = sculpt(f, VOXEL_NEAR);
      const far = sculpt(f, VOXEL_FAR);
      if (near === null || far === null) continue;
      this.geometries.push(near.geometry, far.geometry);
      const field = new InstanceField({
        name: `outcrop-${f.kind}-${i}`,
        levels: [near.geometry, far.geometry],
        material: this.material,
        // Distance over reach: a 12 m outcrop keeps its fine mesh to ~40 m.
        switches: [3.4],
        cullBase: 80,
        cullPerSize: 5,
        cullMax: 220,
        refresh: 4,
        cell: 128,
        // Few, large and close to the seabed: the one thing down here whose
        // shadow reads, and a few thousand triangles each in the depth pass.
        castShadow: true,
      });
      this.sculpts.push({ formation: f, field, top: near.top, reach: f.reach, anchors: near.anchors });
    }
  }

  /**
   * Where the outcrops stand.
   *
   * Around the island, on the shelf and down its slope between 6 and 60 m of
   * water, at roughly 50 m spacing so a diver always has the next one at the
   * edge of visibility. Ridges and arches lie along the contour, with a
   * ridge's overhang facing downhill — a ledge wall. Over the plateau, one
   * beside each reef patch on its seaward side (the backdrop the `reef-dive`
   * view has been missing) and a scatter further out, all lower.
   */
  private placeOutcrops(random: () => number): void {
    const byKind = (k: FormationKind): Sculpt[] => this.sculpts.filter((s) => s.formation.kind === k);
    const lib = { mesa: byKind('mesa'), ridge: byKind('ridge'), spires: byKind('spires'), arch: byKind('arch') };
    const near = (x: number, z: number, spacing: number): boolean => {
      for (const p of this.placements) if (Math.hypot(p.x - x, p.z - z) < spacing) return true;
      return false;
    };
    const tryPlace = (kind: FormationKind, region: 'plateau' | 'island', x: number, z: number, ceiling: number, scale: number): boolean => {
      let options = lib[kind];
      // Where the water is shallow for the clearance, only something low fits;
      // a tall sculpt squashed to fit reads as a stub.
      const room = ceiling - seafloorPlacementHeight(x, z);
      if (room < 11) options = lib.mesa.filter((m) => m.top < 8);
      if (options.length === 0) return false;
      const s = options[Math.floor(random() * options.length)];
      const { slope: g, downhill } = slope(x, z);
      const yaw = kind === 'ridge' || (kind === 'arch' && g > 0.08) ? downhill - Math.PI / 2 : random() * Math.PI * 2;
      const stretch = 0.85 + random() * 0.3;
      let k = scale;
      // Seat on the lowest point of the rendered floor under the footprint.
      const seat = (k: number): number => {
        let base = seafloorPlacementMeshHeight(x, z);
        const r = s.reach * k * 0.65;
        for (let a = 0; a < 6; a++) {
          const ang = (a / 6) * Math.PI * 2;
          base = Math.min(base, seafloorPlacementMeshHeight(x + Math.cos(ang) * r, z + Math.sin(ang) * r));
        }
        return base - 0.4;
      };
      let base = seat(k);
      // Twice: a smaller footprint can seat a little higher.
      for (let pass = 0; pass < 2; pass++) {
        const fit = (ceiling - base) / (s.top * stretch);
        if (fit >= k) break;
        k = fit * 0.97;
        base = seat(k);
      }
      if (k < 0.5) return false;
      this.placements.push({ sculpt: s, x, z, base, yaw, scale: k, stretch, region });
      return true;
    };

    // Plateau: beside the reef patches first.
    const patches = reefPatches();
    for (let i = 0; i < patches.length; i++) {
      const p = patches[i];
      const r0 = Math.hypot(p.x, p.z);
      if (r0 < LANE_RADIUS) continue;
      const ux = p.x / r0;
      const uz = p.z / r0;
      // Inside the patch's seaward edge: a patch reef is a rock mound with
      // the reef on and around it, and visibility here is ~35 m, so a
      // backdrop further out than the patch itself is a backdrop nobody sees.
      const off = REEF_BAND.patchRadius * 0.4;
      const x = p.x + ux * off + uz * (random() - 0.5) * 10;
      const z = p.z + uz * off - ux * (random() - 0.5) * 10;
      const kind: FormationKind = i % 3 === 2 && r0 > 100 ? 'arch' : i % 3 === 1 ? 'spires' : 'mesa';
      tryPlace(kind, 'plateau', x, z, -laneClearance(x, z), 0.75 + random() * 0.3);
    }
    for (let tries = 0; tries < 1500 && this.placements.length < patches.length + 22; tries++) {
      const a = random() * Math.PI * 2;
      const r = 90 + random() * 220;
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      if (near(x, z, 55)) continue;
      const roll = random();
      const kind: FormationKind = roll < 0.25 ? 'ridge' : roll < 0.45 ? 'spires' : roll < 0.55 ? 'arch' : 'mesa';
      tryPlace(kind, 'plateau', x, z, -laneClearance(x, z), 0.7 + random() * 0.4);
    }

    // Island shelf and slope.
    const islandTarget = this.placements.length + 340;
    for (let tries = 0; tries < 40000 && this.placements.length < islandTarget; tries++) {
      const a = random() * Math.PI * 2;
      const t = 0.92 + random() * 1.1;
      const x = ISLAND.x + Math.cos(a) * ISLAND.radius * t;
      const z = ISLAND.z + Math.sin(a) * ISLAND.radius * t;
      const depth = -seafloorPlacementHeight(x, z);
      if (depth < 6 || depth > 60) continue;
      if (near(x, z, 32)) continue;
      const { slope: g } = slope(x, z);
      const roll = random();
      let kind: FormationKind;
      if (depth > 12 && roll < 0.1) kind = 'arch';
      else if (g > 0.1 && roll < 0.55) kind = 'ridge';
      else if (roll < 0.75) kind = 'mesa';
      else kind = 'spires';
      tryPlace(kind, 'island', x, z, -SURFACE_CLEARANCE, 0.75 + random() * 0.6);
    }

    // Into the fields, with each placement's anchors in world space.
    const matrix = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    const color = new THREE.Color();
    const v = new THREE.Vector3();
    const n = new THREE.Vector3();
    for (const p of this.placements) {
      q.setFromAxisAngle(up, p.yaw);
      matrix.compose(v.set(p.x, p.base, p.z), q, n.set(p.scale, p.scale * p.stretch, p.scale));
      const k = 0.9 + random() * 0.15;
      color.setRGB(k, k, k);
      p.sculpt.field.add(matrix, color, p.sculpt.reach * p.scale, random() * 0.2);

      const local = p.sculpt.anchors;
      const count = Math.min(local.length / 6, 160);
      const start = Math.floor(random() * Math.max(1, local.length / 6 - count));
      const anchors: number[] = [];
      for (let i = start; i < start + count; i++) {
        v.set(local[i * 6], local[i * 6 + 1], local[i * 6 + 2]).applyMatrix4(matrix);
        if (v.y > -SURFACE_CLEARANCE + 0.5) continue;
        n.set(local[i * 6 + 3], local[i * 6 + 4], local[i * 6 + 5]).applyQuaternion(q);
        anchors.push(v.x, v.y, v.z, n.x, n.y, n.z);
      }
      const depth = -p.base;
      // Caribbean zonation: coral gardens on the shallow terrace (to ~20 m
      // here; Orbicella and staghorn grow deeper than that), sponges and
      // gorgonians below, sponges dominating the deep slope.
      const theme: GardenTheme =
        depth < 20 ? (random() < 0.8 ? 'coral' : 'fan') : depth < 34 ? (random() < 0.5 ? 'sponge' : 'fan') : random() < 0.75 ? 'sponge' : 'fan';
      this.gardens.push({
        x: p.x,
        z: p.z,
        radius: p.sculpt.reach * p.scale + 4,
        depth,
        theme,
        region: p.region,
        anchors: new Float32Array(anchors),
      });
    }
  }

  /** Field centres and radii, for the influence map and the rubble. */
  private readonly fieldSites: { x: number; z: number; r: number }[] = [];

  private buildLooseRock(random: () => number): void {
    const kinds: { kind: ChunkKind; seed: number }[] = [
      { kind: 'chunk', seed: 11 },
      { kind: 'chunk', seed: 23 },
      { kind: 'boulder', seed: 37 },
      { kind: 'slab', seed: 41 },
    ];
    for (const k of kinds) {
      // Four tessellations of one shape: 1280, 320, 80 and 20 triangles.
      const levels = [3, 2, 1, 0].map((sub) => buildChunkGeometry(ROCK_SEED ^ k.seed, sub, k.kind));
      this.geometries.push(...levels);
      this.looseLevels.push(levels);
      this.fields.push(
        new InstanceField({
          name: `rocks-${k.kind}-${k.seed}`,
          levels,
          material: this.material,
          // Distance over radius: a 1 m boulder is at full detail inside 8 m,
          // 320 triangles to 25 m, 80 to 60 m, and a 20-triangle blob past that.
          switches: [8, 25, 60],
          cullBase: 30,
          cullPerSize: 34,
          cullMax: 160,
          refresh: 4,
        }),
      );
    }
    const pick = (): InstanceField => {
      const r = random();
      return this.fields[r < 0.32 ? 0 : r < 0.62 ? 1 : r < 0.85 ? 2 : 3];
    };

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const euler = new THREE.Euler();
    const color = new THREE.Color();

    const place = (x: number, z: number, size: number, rank: number, ceiling: number): void => {
      const field = pick();
      const isSlab = field === this.fields[3];
      const sy = size * (isSlab ? 0.9 + random() * 0.5 : 0.6 + random() * 0.55);
      const sx = size * (0.85 + random() * 0.3);
      const sz = size * (0.85 + random() * 0.3);
      // Seat on the lowest point of the rendered floor under the footprint, so
      // the uphill side is buried rather than the downhill side floating.
      let floor = seafloorPlacementMeshHeight(x, z);
      const reach = Math.max(sx, sz) * 0.7;
      for (let a = 0; a < 4; a++) {
        const ang = (a / 4) * Math.PI * 2 + 0.4;
        floor = Math.min(floor, seafloorPlacementMeshHeight(x + Math.cos(ang) * reach, z + Math.sin(ang) * reach));
      }
      const y = floor + sy * (-0.2 + random() * 0.45);
      if (y + sy > ceiling) return;
      position.set(x, y, z);
      if (isSlab) euler.set((random() - 0.5) * 0.5, random() * Math.PI * 2, (random() - 0.5) * 0.5);
      else euler.set(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2);
      quaternion.setFromEuler(euler);
      scale.set(sx, sy, sz);
      matrix.compose(position, quaternion, scale);
      // Near-white tint jitter: the colour is the material's world-space beds.
      const k = 0.82 + random() * 0.3;
      color.setRGB(k * (0.97 + random() * 0.06), k, k * (0.95 + random() * 0.06));
      field.add(matrix, color, Math.max(sx, sy, sz), rank);
    };
    /** Power-law sizes: many small, few large, as rockfall comes. */
    const size = (lo: number, hi: number, p: number): number => lo + (hi - lo) * Math.pow(random(), p);

    // 1. Talus at the foot of every outcrop: big blocks close in, small ones
    //    further out.
    for (const p of this.placements) {
      const f = p.sculpt.formation;
      const reach = p.sculpt.reach * p.scale;
      const n = Math.round(14 + reach * 1.4);
      const ca = Math.cos(p.yaw);
      const sa = Math.sin(p.yaw);
      for (let i = 0; i < n; i++) {
        const a = random() * Math.PI * 2;
        const out = Math.abs(gauss(random)) * 6;
        const along = f.kind === 'ridge' ? (random() * 2 - 1) * f.length * p.scale : 0;
        const core = (f.kind === 'arch' ? f.minor * 1.8 : f.kind === 'spires' ? 4 : f.radius) * p.scale;
        const r = core + out;
        const lx = along + Math.cos(a) * r;
        const lz = Math.sin(a) * r;
        const x = p.x + lx * ca - lz * sa;
        const z = p.z + lx * sa + lz * ca;
        const s = size(0.3, 2.4, 2.2) * (1.15 - Math.min(1, out / 8) * 0.5);
        place(x, z, s, s > 1 ? random() * 0.25 : random(), p.base + p.sculpt.top * p.scale);
      }
      this.fieldSites.push({ x: p.x, z: p.z, r: reach });
    }

    // 2. Boulder fields on open shelf between the outcrops.
    const centres: { x: number; z: number; r: number; ceiling: number }[] = [];
    for (let tries = 0; tries < 6000 && centres.length < 90; tries++) {
      const a = random() * Math.PI * 2;
      const t = 0.92 + random() * 1.05;
      const x = ISLAND.x + Math.cos(a) * ISLAND.radius * t;
      const z = ISLAND.z + Math.sin(a) * ISLAND.radius * t;
      const depth = -seafloorPlacementHeight(x, z);
      if (depth < 4 || depth > 60) continue;
      centres.push({ x, z, r: 6 + random() * 12, ceiling: -SURFACE_CLEARANCE });
    }
    for (let tries = 0; tries < 3000 && centres.length < 115; tries++) {
      const a = random() * Math.PI * 2;
      const r = 60 + random() * 250;
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      if (-seafloorPlacementHeight(x, z) < 12) continue;
      centres.push({ x, z, r: 5 + random() * 9, ceiling: -laneClearance(x, z) });
    }
    for (const c of centres) {
      const n = Math.round(c.r * 2.2);
      for (let i = 0; i < n; i++) {
        const a = random() * Math.PI * 2;
        const rr = Math.sqrt(random()) * c.r;
        place(c.x + Math.cos(a) * rr, c.z + Math.sin(a) * rr, size(0.25, 2.1, 2.4), random(), c.ceiling);
      }
      this.fieldSites.push({ x: c.x, z: c.z, r: c.r * 0.6 });
    }

    // 3. Rubble over the whole shelf: two thirds gathered around the rock, the
    //    rest anywhere in the band. Low enough everywhere not to matter to
    //    the ship, so only the general clearance applies.
    const anchors = this.fieldSites;
    const scatter = (count: number, sample: () => { x: number; z: number }): void => {
      for (let i = 0; i < count; i++) {
        let p: { x: number; z: number };
        if (random() < 0.66) {
          const s = anchors[Math.floor(random() * anchors.length)];
          const a = random() * Math.PI * 2;
          const r = s.r + Math.abs(gauss(random)) * 14;
          p = { x: s.x + Math.cos(a) * r, z: s.z + Math.sin(a) * r };
        } else {
          p = sample();
        }
        const depth = -seafloorPlacementHeight(p.x, p.z);
        if (depth < 2.5 || depth > 70) continue;
        place(p.x, p.z, size(0.1, 0.5, 2.0), random(), -2);
      }
    };
    scatter(22000, () => {
      const a = random() * Math.PI * 2;
      const t = 0.9 + random() * 1.1;
      return { x: ISLAND.x + Math.cos(a) * ISLAND.radius * t, z: ISLAND.z + Math.sin(a) * ISLAND.radius * t };
    });
    scatter(6000, () => {
      const a = random() * Math.PI * 2;
      const r = 25 + Math.sqrt(random()) * 280;
      return { x: Math.cos(a) * r, z: Math.sin(a) * r };
    });

    for (const field of this.fields) {
      field.seal();
      for (const mesh of field.meshes) this.object.add(mesh);
    }
  }
}

// ------------------------------------------------------------------- layout

function laneClearance(x: number, z: number): number {
  return Math.hypot(x, z) < LANE_OUTER ? LANE_CLEARANCE : PLATEAU_CLEARANCE;
}

function gauss(random: () => number): number {
  return (random() + random() + random() - 1.5) * 1.15;
}

/** Floor gradient magnitude and downhill bearing at a point. */
function slope(x: number, z: number): { slope: number; downhill: number } {
  const h = 3;
  const dx = (seafloorPlacementHeight(x + h, z) - seafloorPlacementHeight(x - h, z)) / (2 * h);
  const dz = (seafloorPlacementHeight(x, z + h) - seafloorPlacementHeight(x, z - h)) / (2 * h);
  return { slope: Math.hypot(dx, dz), downhill: Math.atan2(-dz, -dx) };
}

function makeFormation(kind: FormationKind, random: () => number, index: number): Formation {
  const radius = 5 + random() * 4;
  const f: Formation = {
    kind,
    radius,
    length: 0,
    height: 7 + random() * 9,
    overhang: 1 + random() * 2.5,
    waist: 0.5 + random() * 1.5,
    minor: 0,
    tunnel: 0,
    spires: [],
    shoulder: null,
    noise: index * 37.3 + random() * 11,
    reach: radius,
  };
  switch (kind) {
    case 'mesa': {
      f.length = random() * 4;
      // The last two mesas are low and broad: patch-reef mounds for the
      // plateau, where the lane clearance leaves eight metres of water.
      if (index >= 5) {
        f.height = 3.5 + random() * 2;
        f.radius = 6 + random() * 3;
        f.overhang = 1.2 + random();
      }
      if (random() < 0.75) {
        const a = random() * Math.PI * 2;
        f.shoulder = {
          x: Math.cos(a) * radius * 0.95,
          z: Math.sin(a) * radius * 0.95,
          r: radius * (0.45 + random() * 0.25),
          h: f.height * (0.35 + random() * 0.3),
        };
      }
      // Two of the five mesas are bored through: the swim-throughs.
      if (index === 1 || index === 3) f.tunnel = 1.8 + random() * 0.5;
      f.reach = radius + f.length + f.overhang + (f.shoulder ? f.shoulder.r : 0);
      break;
    }
    case 'ridge': {
      f.radius = 3.2 + random() * 2.4;
      f.length = 8 + random() * 8;
      f.overhang = 2.2 + random() * 2.3;
      f.height = 5 + random() * 7;
      if (index === 8) f.tunnel = 1.7;
      f.reach = f.length + f.radius + f.overhang;
      break;
    }
    case 'spires': {
      const n = 2 + Math.floor(random() * 3);
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + random();
        const d = i === 0 ? 0 : 2.5 + random() * 3.5;
        f.spires.push({ x: Math.cos(a) * d, z: Math.sin(a) * d, r: 1.8 + random() * 1.6, h: (i === 0 ? 10 : 5) + random() * 8 });
      }
      f.height = Math.max(...f.spires.map((p) => p.h));
      f.reach = 9;
      break;
    }
    case 'arch': {
      f.minor = 1.5 + random() * 0.7;
      f.length = 5 + random() * 3;
      f.height = f.length * 0.95 + f.minor;
      f.reach = f.length + f.minor * 2.5;
      break;
    }
  }
  return f;
}

// ------------------------------------------------------------------ sculpting

function seg2(px: number, pz: number, ax: number, az: number, bx: number, bz: number): number {
  const pax = px - ax;
  const paz = pz - az;
  const bax = bx - ax;
  const baz = bz - az;
  const h = Math.min(1, Math.max(0, (pax * bax + paz * baz) / Math.max(bax * bax + baz * baz, 1e-4)));
  return Math.hypot(pax - bax * h, paz - baz * h);
}

/** The outcrop's density at a local point (base at y = 0): positive is rock. */
function density(f: Formation, lx: number, ly: number, lz: number): number {
  const sd = f.noise;
  // Domain warp: low-frequency, so columns lean and bulge rather than wobble.
  const wx = (vnoise3(lx * 0.045 + sd, ly * 0.045, lz * 0.045) - 0.5) * 4.5;
  const wy = (vnoise3(lx * 0.045 + sd + 31.7, ly * 0.045 + 5.1, lz * 0.045) - 0.5) * 2.5;
  const wz = (vnoise3(lx * 0.045 + sd + 57.1, ly * 0.045, lz * 0.045 + 9.3) - 0.5) * 4.5;
  const qx = lx + wx;
  const qy = ly + wy;
  const qz = lz + wz;

  /** A column: abyssal-garden's overhang / base-flare / waist profile. */
  const column = (
    cx: number,
    cz: number,
    len: number,
    r: number,
    h: number,
    over: number,
    waist: number,
    flare: number,
    ledge: boolean,
  ): number => {
    const yr = Math.min(1.3, Math.max(0, qy / h));
    let e = seg2(qx, qz, cx - len, cz, cx + len, cz) - r;
    // A ledge wall overhangs on its downhill (+z) face more than its back.
    const side = ledge ? (qz > cz ? 1 : 0.3) : 1;
    e -=
      over * side * smoothstep(0.3, 0.92, yr) +
      flare * (1 - yr) * (1 - yr) -
      waist * smoothstep(0, 0.5, yr) * (1 - smoothstep(0.5, 1, yr));
    const top = h + (vnoise3(qx * 0.09 + sd, 3.3, qz * 0.09) - 0.5) * 3;
    return smin(-e, top - qy, 2.5);
  };

  let d = -1e3;
  switch (f.kind) {
    case 'mesa':
      d = column(0, 0, f.length, f.radius, f.height, f.overhang, f.waist, 2.5, false);
      if (f.shoulder) {
        const s = f.shoulder;
        d = smax(d, column(s.x, s.z, 0, s.r, s.h, 0.8, 0.3, 1.8, false), 2.5);
      }
      break;
    case 'ridge':
      d = column(0, 0, f.length, f.radius, f.height, f.overhang, f.waist, 2.2, true);
      break;
    case 'spires':
      for (const s of f.spires) {
        const yr = Math.min(1.2, Math.max(0, qy / s.h));
        const e = Math.hypot(qx - s.x, qz - s.z) - s.r * (1 - 0.45 * yr) - 2 * (1 - yr) * (1 - yr);
        const top = s.h + (vnoise3(qx * 0.15 + sd, 7.7, qz * 0.15) - 0.5) * 2;
        d = smax(d, smin(-e, top - qy, 1.5), 2);
      }
      break;
    case 'arch': {
      // A ring standing in the local XY plane, its centre a quarter-radius
      // under the base so both feet are buried, squashed a little upright.
      const R = f.length;
      const cy = -R * 0.25;
      const ry = (qy - cy) / 1.2;
      const ring = Math.hypot(Math.hypot(qx, ry) - R, qz);
      // Thicker at the feet than at the crown, as an eroded arch is.
      const tube = f.minor * (1 + 0.45 * (1 - smoothstep(0, R * 0.9, qy)));
      d = tube - ring;
      for (const side of [-1, 1]) {
        const e = Math.hypot(qx - side * R, qz) - f.minor * 1.7;
        d = smax(d, smin(-e, f.minor * 2.2 - qy, 1.5), 1.8);
      }
      break;
    }
  }

  // Relief and strata ledges, only where they can move the surface.
  if (d > -4 && d < 4) {
    const rk = 0.35 + 0.65 * smoothstep(0.5, 4, ly);
    d +=
      rk *
      ((vnoise3(qx * 0.17 + sd, qy * 0.17, qz * 0.17) - 0.5) * 3.2 +
        (vnoise3(lx * 0.43 + sd + 7, ly * 0.43, lz * 0.43) - 0.5) * 1.3);
    // Creases: ridged noise, for the sharp joints a limestone face breaks along.
    const ridged = 1 - Math.abs(vnoise3(qx * 0.24 + sd + 3, qy * 0.12, qz * 0.24) * 2 - 1);
    d -= rk * (ridged - 0.6) * 1.4;
    // Strata: a triangle wave for the beds plus a groove at each bedding
    // plane, which is what makes a ledge with a lip rather than a ripple.
    const band = ly * 0.21 + wy * 0.9;
    const f = band - Math.floor(band);
    d += rk * (1.3 * (Math.abs(f - 0.5) - 0.25) + 1.1 * (smoothstep(0, 0.12, f) - smoothstep(0.84, 1, f) - 0.6));
  }

  // A swim-through bored across the column at two-fifths height.
  if (f.tunnel > 0) {
    const dist = Math.hypot(qx * 0.35, qy - f.height * 0.4);
    d = smin(d, dist - f.tunnel, 1.2);
  }

  // The skirt: closed off below the seating depth.
  return Math.min(d, ly + SKIRT);
}

/**
 * Meshes one outcrop at one voxel size, in its own frame. Normals from the
 * density gradient, and a baked occlusion from how much rock lies a metre and
 * three metres out along the normal — abyssal-garden bakes the same openness
 * into a volume texture; per vertex is enough for one outcrop.
 */
function sculpt(
  f: Formation,
  voxel: number,
): { geometry: THREE.BufferGeometry; top: number; anchors: Float32Array } | null {
  const pad = 4;
  const hx = f.reach + pad;
  const hz = (f.kind === 'ridge' ? f.radius + f.overhang + 2 : f.kind === 'arch' ? f.minor * 2.5 : f.reach) + pad;
  const nx = Math.ceil((2 * hx) / voxel) + 1;
  const nz = Math.ceil((2 * hz) / voxel) + 1;
  const ox = -hx;
  const oz = -hz;
  const oy = -SKIRT - 1.5;
  const ny = Math.ceil((f.height + 6 - oy) / voxel) + 1;

  const D = new Float32Array(nx * ny * nz);
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const i = ix + nx * (iy + ny * iz);
        if (ix === 0 || iz === 0 || ix === nx - 1 || iz === nz - 1 || iy === 0 || iy === ny - 1) {
          D[i] = -1;
          continue;
        }
        D[i] = density(f, ox + ix * voxel, oy + iy * voxel, oz + iz * voxel);
      }
    }
  }

  const net = surfaceNets(D, nx, ny, nz);
  const count = net.pos.length / 3;
  if (count < 4) return null;
  const sample = gridSampler(D, nx, ny, nz);
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const ao = new Float32Array(count);
  const anchors: number[] = [];
  let top = 0;
  for (let k = 0; k < count; k++) {
    const gx = net.pos[k * 3];
    const gy = net.pos[k * 3 + 1];
    const gz = net.pos[k * 3 + 2];
    const e = 0.8;
    let gnx = sample(gx - e, gy, gz) - sample(gx + e, gy, gz);
    let gny = sample(gx, gy - e, gz) - sample(gx, gy + e, gz);
    let gnz = sample(gx, gy, gz - e) - sample(gx, gy, gz + e);
    const l = Math.hypot(gnx, gny, gnz) || 1;
    gnx /= l;
    gny /= l;
    gnz /= l;
    // How open the space is a metre and three metres out along the normal.
    const s1 = 1.2 / voxel;
    const s2 = 3.2 / voxel;
    const o1 = Math.min(1, Math.max(0, 0.5 - sample(gx + gnx * s1, gy + gny * s1, gz + gnz * s1) / 2.4));
    const o2 = Math.min(1, Math.max(0, 0.5 - sample(gx + gnx * s2, gy + gny * s2, gz + gnz * s2) / 6));
    ao[k] = 0.22 + 0.78 * Math.pow(o1 * 0.55 + o2 * 0.45, 1.4);

    const x = ox + gx * voxel;
    const y = oy + gy * voxel;
    const z = oz + gz * voxel;
    positions[k * 3] = x;
    positions[k * 3 + 1] = y;
    positions[k * 3 + 2] = z;
    normals[k * 3] = gnx;
    normals[k * 3 + 1] = gny;
    normals[k * 3 + 2] = gnz;
    top = Math.max(top, y);
    // Up- and out-facing surface above the base, for the reef to grow on.
    if (gny > 0.15 && y > 0.4 && (k * 7919) % 5 === 0) anchors.push(x, y, z, gnx, gny, gnz);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('rockAo', new THREE.BufferAttribute(ao, 1));
  g.setIndex(new THREE.BufferAttribute(net.idx, 1));
  g.computeBoundingSphere();
  return { geometry: g, top, anchors: shuffleAnchors(new Float32Array(anchors), f.noise) };
}

/** Anchors in a seeded random order, so any run of them samples the whole outcrop. */
function shuffleAnchors(a: Float32Array, seed: number): Float32Array {
  const random = mulberry32(Math.floor(seed * 1000));
  const n = a.length / 6;
  const tmp = new Float32Array(6);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    tmp.set(a.subarray(i * 6, i * 6 + 6));
    a.copyWithin(i * 6, j * 6, j * 6 + 6);
    a.set(tmp, j * 6);
  }
  return a;
}

// --------------------------------------------------------------- influence

const INFLUENCE_SIZE = 512;
/** [underwater-look-3] Round 10. Rubble weight of a kelp forest's floor, and round each holdfast. */
const BED_FOREST = 0.5;
const BED_COLUMN = 0.95;
const BED_STONE_TINT: readonly [number, number] = [1.25, 1.75];
const INFLUENCE_MIN_X = -2250;
const INFLUENCE_MIN_Z = -2000;
const INFLUENCE_EXTENT = 2600;

/**
 * A 5 m-per-texel map of where the rock is, for the sand.
 *
 *   r  scree: rubble and coarse sediment shed by the rock, reaching ~25 m
 *   g  the foot: darker, finer substrate around the rock's base, ~6 m
 *
 * The seabed is one mesh shaded by one material over 16 km², and it has no
 * other way to know that a twelve-metre outcrop is standing on it.
 */
function buildInfluence(
  placements: Placement[],
  fields: { x: number; z: number; r: number }[],
  beds: readonly { x: number; z: number; r: number; k: number }[] = [],
): RockInfluence {
  const N = INFLUENCE_SIZE;
  const texel = INFLUENCE_EXTENT / N;
  const scree = new Float32Array(N * N);
  const foot = new Float32Array(N * N);
  const splat = (x: number, z: number, r: number, reach: number, footReach: number, k: number): void => {
    const outer = r + reach;
    const i0 = Math.max(0, Math.floor((x - outer - INFLUENCE_MIN_X) / texel));
    const i1 = Math.min(N - 1, Math.ceil((x + outer - INFLUENCE_MIN_X) / texel));
    const j0 = Math.max(0, Math.floor((z - outer - INFLUENCE_MIN_Z) / texel));
    const j1 = Math.min(N - 1, Math.ceil((z + outer - INFLUENCE_MIN_Z) / texel));
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const px = INFLUENCE_MIN_X + (i + 0.5) * texel;
        const pz = INFLUENCE_MIN_Z + (j + 0.5) * texel;
        const d = Math.hypot(px - x, pz - z);
        const o = j * N + i;
        scree[o] = Math.max(scree[o], k * (1 - smoothstep(r * 0.6, outer, d)));
        foot[o] = Math.max(foot[o], k * (1 - smoothstep(r * 0.8, r + footReach, d)));
      }
    }
  };
  for (const p of placements) splat(p.x, p.z, p.sculpt.reach * p.scale * 0.8, 24, 6, 1);
  for (const s of fields) splat(s.x, s.z, s.r, 12, 3, 0.7);
  // [underwater-look-3] Round 10: kelp beds, rubble and a silty skirt; see `addKelpBedStones`.
  for (const b of beds) splat(b.x, b.z, b.r, 4, 2, b.k);

  const data = new Uint8Array(N * N * 4);
  for (let i = 0; i < N * N; i++) {
    data[i * 4] = Math.round(Math.min(1, scree[i]) * 255);
    data[i * 4 + 1] = Math.round(Math.min(1, foot[i]) * 255);
    data[i * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return { texture, minX: INFLUENCE_MIN_X, minZ: INFLUENCE_MIN_Z, size: INFLUENCE_EXTENT };
}
