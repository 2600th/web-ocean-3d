import * as THREE from 'three/webgpu';

/**
 * Instances spread over kilometres of seabed, drawn only where a camera can see
 * them, at a level of detail matched to how big they look.
 *
 * `ReefLife` started as one `InstancedMesh` per species per region, with the
 * whole region switched on or off by distance. That works for nine reef sites
 * and stops working the moment the shelf is dressed all the way round the
 * island: the renderer's frustum test sees one sphere a kilometre across and
 * never culls it, so every coral on the far side of the island is vertex-shaded
 * every frame. Underwater visibility here is 45-70 m, so almost all of that work
 * is invisible.
 *
 * So instances are kept on the CPU — matrix, colour, a size and a random rank —
 * bucketed into a coarse grid, and **dealt** into one `InstancedMesh` per LOD
 * level whenever the camera has moved far enough for the answer to change.
 * The same scheme `Props.updateLod` uses for the island dressing, with two
 * differences: the deal walks only the grid cells inside the cull radius, so a
 * field of tens of thousands costs a few thousand distance tests; and the level
 * is chosen by distance *over size* — a 12 m outcrop and a 30 cm pebble switch
 * at the same apparent size, not the same range.
 *
 * Every level mesh stays `visible` with `count` moving; a level with nothing
 * dealt to it has a count of zero, which three skips at draw time
 * (`RenderObject.getDrawParameters` returns null). That is load-bearing: the
 * boot prewarm compiles what is visible, so every level has its pipeline from
 * the first frame and a level switch can never stall on a compile.
 *
 * The tier's detail fraction is a rank threshold rather than a count prefix, so
 * thinning removes the same instances everywhere and never a whole region.
 * Deterministic: the deal is a pure function of the camera position.
 */

export interface InstanceFieldOptions {
  name: string;
  /** Finest first. All levels must share one vertex layout. */
  levels: readonly THREE.BufferGeometry[];
  material: THREE.Material;
  /**
   * Level `k` is used while `distance / size < switches[k]`; the last level
   * past every switch. `switches.length === levels.length - 1`.
   */
  switches: readonly number[];
  /** Instances further than `cullBase + size * cullPerSize` (capped at `cullMax`) are not drawn. */
  cullBase: number;
  cullPerSize: number;
  cullMax: number;
  /** Camera travel, metres, before a new deal. */
  refresh?: number;
  /** Grid cell edge, metres. */
  cell?: number;
  castShadow?: boolean;
  /**
   * [underwater-look-3] Round 12. Cross-fades a level change over this many
   * seconds instead of swapping the mesh in one frame. While an instance is
   * changing level it is dealt to both, each copy carrying `lodFade`: the
   * clock time the change began and +1 on the level fading in, -1 on the one
   * fading out (0 when steady). The material dithers the pair into
   * complementary halves of one screen-space pattern (see `ReefLife`'s
   * `LOD_FADE_SECONDS`). Changes need the distance to clear the switch by
   * `LOD_HYSTERESIS`, so a camera parked on a switch does not flicker between
   * levels. Without the option none of this is built.
   */
  crossFade?: number;
}

/** Fraction of a switch distance a level change has to clear; see `crossFade`. */
const LOD_HYSTERESIS = 0.05;

const _m = new THREE.Matrix4();

/**
 * Instance matrices uploaded only when a deal changes them, and only the dealt
 * prefix.
 *
 * These were `DynamicDrawUsage`, and in r185 that does not mean "changes
 * often", it means "upload all of it on every render call": `Attributes.update`
 * writes a dynamic buffer whenever it is used, version or not, and the update
 * ranges a deal sets are cleared by the first such write, so every later one
 * is the whole capacity. Every level of every field, 10-20 MB a second each
 * of `writeBuffer` at High: about 7 ms of main thread a frame (CPU profile of
 * the production build), which with the rest of the frame put it at 17-20 ms
 * and kept the 120 Hz pacer at a divisor of 3 or 4. Static usage uploads on
 * the version a deal bumps, through the ranges it sets.
 *
 * Static usage alone would break motion vectors. For the velocity pass three
 * clones the matrices into a "previous" attribute (`getPreviousInstance` in
 * `nodes/accessors/Instance.js`), copies the current array into it before each
 * render, and relies on dynamic usage to upload that copy: a static clone is
 * uploaded once and is wrong after the first deal reorders the slots. Nothing
 * here moves between frames, so the previous matrices are the current ones,
 * which is also what the copy-then-upload produced. `clone` returns the
 * attribute itself, so the previous node reads the same GPU buffer. Nothing
 * else clones an instance matrix.
 */
class DealtMatrices extends THREE.InstancedBufferAttribute {
  override clone(): this {
    return this;
  }
}

export class InstanceField {
  readonly meshes: THREE.InstancedMesh[] = [];
  readonly name: string;

  private readonly options: InstanceFieldOptions;
  private capacity = 64;
  private n = 0;
  private matrices = new Float32Array(64 * 16);
  private colors = new Float32Array(64 * 3);
  private centres = new Float32Array(64 * 3);
  private sizes = new Float32Array(64);
  private ranks = new Float32Array(64);
  private sealed = false;

  /** Cells: instance ranges after the sort, and each cell's bounds. */
  private cellStart = new Uint32Array(0);
  private cellEnd = new Uint32Array(0);
  private cellBounds = new Float32Array(0); // minX, minY, minZ, maxX, maxY, maxZ
  private cellCount = 0;

  private detail = 1;
  private multiplier = 1;
  /** Cross-fade state per instance (sealed order): level shown, level left, when. */
  private shown = new Int8Array(0);
  private previous = new Int8Array(0);
  private left = new Int8Array(0);
  private since = new Float32Array(0);
  private readonly fades: THREE.InstancedBufferAttribute[] = [];
  private clock = 0;
  private fading = false;
  private readonly dealtAt = new THREE.Vector3(Infinity, Infinity, Infinity);
  private dirty = true;
  private visible = true;

  constructor(options: InstanceFieldOptions) {
    this.options = options;
    this.name = options.name;
  }

  get count(): number {
    return this.n;
  }

  /**
   * Adds one instance. `size` is its characteristic radius in metres (it sets
   * both the LOD and the cull range); `rank` in [0, 1) decides at which detail
   * fraction it drops out.
   */
  add(matrix: THREE.Matrix4, color: THREE.Color, size: number, rank: number): void {
    if (this.sealed) throw new Error(`InstanceField ${this.name}: add after seal`);
    if (this.n === this.capacity) this.grow();
    const i = this.n++;
    matrix.toArray(this.matrices, i * 16);
    this.colors[i * 3] = color.r;
    this.colors[i * 3 + 1] = color.g;
    this.colors[i * 3 + 2] = color.b;
    const e = matrix.elements;
    this.centres[i * 3] = e[12];
    this.centres[i * 3 + 1] = e[13];
    this.centres[i * 3 + 2] = e[14];
    this.sizes[i] = size;
    this.ranks[i] = rank;
  }

  /**
   * Sorts the instances into cells and builds the level meshes. Every level is
   * filled with the whole population until the first deal, so the boot
   * prewarm, which walks visible objects, compiles each of them.
   */
  seal(): void {
    if (this.sealed) return;
    this.sealed = true;
    const o = this.options;
    const cell = o.cell ?? 64;
    const n = this.n;

    const keys = new Float64Array(n);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const cx = Math.floor(this.centres[i * 3] / cell) + 4096;
      const cz = Math.floor(this.centres[i * 3 + 2] / cell) + 4096;
      keys[i] = cx * 8192 + cz;
      order[i] = i;
    }
    order.sort((a, b) => keys[a] - keys[b] || a - b);

    const matrices = new Float32Array(n * 16);
    const colors = new Float32Array(n * 3);
    const centres = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const ranks = new Float32Array(n);
    const starts: number[] = [];
    const ends: number[] = [];
    const bounds: number[] = [];
    let last = -1;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      matrices.set(this.matrices.subarray(i * 16, i * 16 + 16), k * 16);
      colors.set(this.colors.subarray(i * 3, i * 3 + 3), k * 3);
      centres.set(this.centres.subarray(i * 3, i * 3 + 3), k * 3);
      sizes[k] = this.sizes[i];
      ranks[k] = this.ranks[i];
      const s = sizes[k];
      if (keys[i] !== last) {
        last = keys[i];
        if (starts.length > 0) ends.push(k);
        starts.push(k);
        bounds.push(Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity);
      }
      const b = bounds.length - 6;
      bounds[b] = Math.min(bounds[b], centres[k * 3] - s);
      bounds[b + 1] = Math.min(bounds[b + 1], centres[k * 3 + 1] - s);
      bounds[b + 2] = Math.min(bounds[b + 2], centres[k * 3 + 2] - s);
      bounds[b + 3] = Math.max(bounds[b + 3], centres[k * 3] + s);
      bounds[b + 4] = Math.max(bounds[b + 4], centres[k * 3 + 1] + s);
      bounds[b + 5] = Math.max(bounds[b + 5], centres[k * 3 + 2] + s);
    }
    if (starts.length > 0) ends.push(n);
    this.matrices = matrices;
    this.colors = colors;
    this.centres = centres;
    this.sizes = sizes;
    this.ranks = ranks;
    this.cellStart = Uint32Array.from(starts);
    this.cellEnd = Uint32Array.from(ends);
    this.cellBounds = Float32Array.from(bounds);
    this.cellCount = starts.length;
    this.capacity = n;
    if (o.crossFade !== undefined) {
      this.shown = new Int8Array(n).fill(-1);
      this.previous = new Int8Array(n).fill(-1);
      this.left = new Int8Array(n).fill(-1);
      this.since = new Float32Array(n).fill(-Infinity);
    }

    for (let level = 0; level < o.levels.length; level++) {
      const mesh = new THREE.InstancedMesh(o.levels[level], o.material, Math.max(1, n));
      mesh.name = `${o.name}-lod${level}`;
      mesh.castShadow = o.castShadow ?? false;
      mesh.receiveShadow = true;
      // The field spans kilometres; its own bounds would never cull. The deal
      // is the cull.
      mesh.frustumCulled = false;
      // Static usage throughout; see `DealtMatrices`.
      mesh.instanceMatrix = new DealtMatrices(new Float32Array(Math.max(1, n) * 16), 16);
      (mesh.instanceMatrix.array as Float32Array).set(matrices);
      const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, n) * 3), 3);
      (colorAttr.array as Float32Array).set(colors);
      mesh.instanceColor = colorAttr;
      if (o.crossFade !== undefined) {
        // Instanced attributes live on the geometry, so each level draws a
        // copy of its own; the templates are a few thousand triangles. Zeroed,
        // which is steady: until the first deal every level draws everything.
        const geometry = o.levels[level].clone();
        const fadeAttr = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, n) * 2), 2);
        geometry.setAttribute('lodFade', fadeAttr);
        mesh.geometry = geometry;
        this.fades.push(fadeAttr);
      }
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
      colorAttr.needsUpdate = true;
      mesh.matrixAutoUpdate = false;
      this.meshes.push(mesh);
    }
  }

  /** Detail fraction in [0, 1]; instances whose rank is at or above it drop out. */
  setDetail(detail: number): void {
    const d = Math.max(0, Math.min(1, detail));
    if (d !== this.detail) {
      this.detail = d;
      this.dirty = true;
    }
  }

  /** Scales every cull and switch distance; how a tier trades range for cost. */
  setRangeMultiplier(k: number): void {
    if (k !== this.multiplier) {
      this.multiplier = k;
      this.dirty = true;
    }
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    for (const mesh of this.meshes) mesh.visible = visible;
  }

  /** Total triangles currently dealt, for the harness. */
  triangleCount(): number {
    let t = 0;
    for (const mesh of this.meshes) {
      if (!mesh.visible) continue;
      t += (mesh.geometry.index!.count / 3) * mesh.count;
    }
    return t;
  }

  /**
   * The clock the cross-fades run on, seconds; the material's must match. A
   * clock that went backwards (a harness reset) ends every fade at once.
   */
  setClock(seconds: number): void {
    if (seconds < this.clock) this.endFades();
    this.clock = seconds;
  }

  /** Ends every cross-fade in progress, each instance on the level it was going to. */
  endFades(): void {
    this.since.fill(-Infinity);
    this.dirty = true;
  }

  /** Re-deals if the camera has moved far enough, detail changed, or a cross-fade is running. Allocation-free. */
  update(camera: THREE.Vector3): void {
    if (!this.sealed || !this.visible) return;
    const refresh = this.options.refresh ?? 4;
    if (!this.dirty && !this.fading && this.dealtAt.distanceToSquared(camera) < refresh * refresh) return;
    this.deal(camera);
  }

  /** Forces a deal now, whatever the camera did. */
  deal(camera: THREE.Vector3): void {
    if (!this.sealed) return;
    const o = this.options;
    const levels = this.meshes.length;
    const k = this.multiplier;
    const cullMax = o.cullMax * k;
    const cullMaxSq = cullMax * cullMax;
    const cx = camera.x;
    const cy = camera.y;
    const cz = camera.z;
    const counts = DEAL_COUNTS;
    for (let l = 0; l < levels; l++) counts[l] = 0;
    const detail = this.detail;
    const fade = o.crossFade;
    const now = this.clock;
    let fading = false;
    // Off the deal (culled, thinned, or in a cell out of range) an instance
    // forgets its level, so it comes back without a fade from a stale one.
    if (fade !== undefined) this.shown.fill(-1);

    for (let c = 0; c < this.cellCount; c++) {
      const b = c * 6;
      const dx = Math.max(this.cellBounds[b] - cx, 0, cx - this.cellBounds[b + 3]);
      const dy = Math.max(this.cellBounds[b + 1] - cy, 0, cy - this.cellBounds[b + 4]);
      const dz = Math.max(this.cellBounds[b + 2] - cz, 0, cz - this.cellBounds[b + 5]);
      if (dx * dx + dy * dy + dz * dz > cullMaxSq) continue;
      for (let i = this.cellStart[c]; i < this.cellEnd[c]; i++) {
        if (this.ranks[i] >= detail) continue;
        const ex = this.centres[i * 3] - cx;
        const ey = this.centres[i * 3 + 1] - cy;
        const ez = this.centres[i * 3 + 2] - cz;
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez);
        const size = this.sizes[i];
        const cull = Math.min(cullMax, (o.cullBase + size * o.cullPerSize) * k);
        if (d > cull) continue;
        const r = d / (size * k);
        let level = levels - 1;
        for (let s = 0; s < o.switches.length; s++) {
          if (r < o.switches[s]) {
            level = s;
            break;
          }
        }
        if (fade === undefined) {
          this.put(level, i, counts, 0, 0);
          continue;
        }
        const was = this.previous[i];
        if (was >= 0 && level !== was) {
          // Clear the switch between them by the hysteresis, or stay.
          const at = level > was ? o.switches[was] * (1 + LOD_HYSTERESIS) : o.switches[was - 1] * (1 - LOD_HYSTERESIS);
          if (level > was ? r < at : r > at) level = was;
          else {
            this.left[i] = was;
            this.since[i] = now;
          }
        }
        this.shown[i] = level;
        if (was >= 0 && now - this.since[i] < fade && this.left[i] >= 0 && this.left[i] !== level) {
          this.put(level, i, counts, this.since[i], 1);
          this.put(this.left[i], i, counts, this.since[i], -1);
          fading = true;
        } else {
          this.put(level, i, counts, 0, 0);
        }
      }
    }
    if (fade !== undefined) {
      this.previous.set(this.shown);
      this.fading = fading;
    }

    for (let l = 0; l < levels; l++) {
      const mesh = this.meshes[l];
      const count = counts[l];
      mesh.count = count;
      if (count > 0) {
        // Only the dealt prefix goes to the GPU.
        mesh.instanceMatrix.clearUpdateRanges();
        mesh.instanceMatrix.addUpdateRange(0, count * 16);
        mesh.instanceMatrix.needsUpdate = true;
        const color = mesh.instanceColor!;
        color.clearUpdateRanges();
        color.addUpdateRange(0, count * 3);
        color.needsUpdate = true;
        if (fade !== undefined) {
          const f = this.fades[l];
          f.clearUpdateRanges();
          f.addUpdateRange(0, count * 2);
          f.needsUpdate = true;
        }
      }
    }
    this.dealtAt.set(cx, cy, cz);
    this.dirty = false;
  }

  /** Deals instance `i` into the next slot of `level`, with its `lodFade` (start time, direction). */
  private put(level: number, i: number, counts: Int32Array, start: number, direction: number): void {
    const mesh = this.meshes[level];
    const slot = counts[level]++;
    (mesh.instanceMatrix.array as Float32Array).set(this.matrices.subarray(i * 16, i * 16 + 16), slot * 16);
    const col = mesh.instanceColor!.array as Float32Array;
    col[slot * 3] = this.colors[i * 3];
    col[slot * 3 + 1] = this.colors[i * 3 + 1];
    col[slot * 3 + 2] = this.colors[i * 3 + 2];
    if (this.options.crossFade === undefined) return;
    const f = this.fades[level].array as Float32Array;
    f[slot * 2] = start;
    f[slot * 2 + 1] = direction;
  }

  /** World matrix of instance `i` in sealed order; for placing things on it. */
  matrixAt(i: number, out: THREE.Matrix4 = _m): THREE.Matrix4 {
    return out.fromArray(this.matrices, i * 16);
  }

  dispose(): void {
    for (const mesh of this.meshes) {
      // The cross-fade's per-level geometry copies are this field's own.
      if (this.options.crossFade !== undefined) mesh.geometry.dispose();
      mesh.dispose();
    }
  }

  private grow(): void {
    const cap = this.capacity * 2;
    const grow = (a: Float32Array, k: number) => {
      const b = new Float32Array(cap * k);
      b.set(a);
      return b;
    };
    this.matrices = grow(this.matrices, 16);
    this.colors = grow(this.colors, 3);
    this.centres = grow(this.centres, 3);
    this.sizes = grow(this.sizes, 1);
    this.ranks = grow(this.ranks, 1);
    this.capacity = cap;
  }
}

/** Shared scratch for the per-level counts; no field has more than eight levels. */
const DEAL_COUNTS = new Int32Array(8);
