import * as THREE from 'three/webgpu';
import type { OceanSimulation } from './OceanSimulation';

/**
 * CPU-side query of the wave surface, for buoyancy and camera collision.
 *
 * The displacement field lives on the GPU, so a slice of it is copied back into
 * host memory. The readback is asynchronous and deliberately *not* awaited by the
 * frame loop: stalling on a GPU fence would cost far more than the one-frame
 * staleness it buys, and at 60 Hz a frame of lag on a floating hull is invisible.
 *
 * Sampling is not a plain texture lookup. The surface is choppy, meaning a vertex
 * at grid position p is displayed at p + D(p) — so finding the height *above a
 * given world point* requires inverting that map. A few fixed-point iterations
 * converge quickly for the displacement magnitudes we allow (the map stays
 * invertible as long as the Jacobian stays positive, which is also the condition
 * for not generating foam).
 */

/**
 * Readback grid per cascade, in texels per side.
 *
 * This used to be a constant 64, read as `readRenderTargetPixelsAsync(target,
 * 0, 0, 64, 64)` from a target that is `fftSize` square — 128, 256 or 512. That
 * is not a downsample. It is the **bottom-left corner**, and `sampleBilinear`
 * then spread those 64 texels across the whole tile, so the physics floated the
 * hull on the corner quarter of the wave field stretched over all of it.
 *
 * The two seas had the same amplitude and nothing else: measured against the
 * true field reconstructed from the same targets, correlation was 0.075 and the
 * rms difference 2.30 m, worst 6.26 m. The hull's height relative to the water
 * a viewer can see was therefore very nearly uncorrelated with it, which is what
 * "the ship sometimes goes under" actually was — not a buoyancy failure but the
 * solver and the renderer disagreeing about where the water is. It also explains
 * why it was intermittent: whether the hull sat high or low depended on where it
 * happened to be relative to a phase error, not on the sea state.
 *
 * The slice now covers the whole target. The size is per-cascade because a tier
 * change moves `fftSize`, and it is read back from the target rather than
 * assumed.
 */
interface CascadeSlice {
  tileSize: number;
  /** Texels per side of this slice. Matches the target it was read from. */
  size: number;
  /**
   * RGBA texels, `size^2`, as the target stores them: binary16 bit patterns
   * for a half-float target (`half`), plain floats otherwise. See `adoptInto`.
   */
  data: Float32Array | Uint16Array | null;
  half: boolean;
}

export class OceanSampler {
  private readonly renderer: THREE.WebGPURenderer;
  private readonly simulation: OceanSimulation;
  private readonly slices: CascadeSlice[] = [];
  private pending = false;
  private everResolved = false;

  constructor(renderer: THREE.WebGPURenderer, simulation: OceanSimulation) {
    this.renderer = renderer;
    this.simulation = simulation;
    this.rebuild();
  }

  /** True once at least one readback has landed; before that, height() returns 0. */
  get ready(): boolean {
    return this.everResolved;
  }

  /**
   * Follows the simulation after it was built or rebuilt at another tier.
   *
   * **The last readback is kept across a rebuild of the same bands.** A tier
   * change moves the FFT size, never the tiles, and this used to drop every
   * slice and report height 0 until the next readback landed, a few frames
   * later. Buoyancy had nothing to push against for those frames and the hull
   * fell toward a flat sea, then bounced. The old slices describe the same
   * swell (the spectrum's draws are keyed to the wavevector, not the texel; see
   * `Spectrum.anchorGaussians`), at the old resolution, which `size` records per
   * slice, so they carry the physics until the new targets are read.
   *
   * A readback already in flight was started against the targets the rebuild
   * retired, and may still have copies to issue. The generation stops it before
   * any further copy and discards what it brings back.
   */
  rebuild(): void {
    this.generation++;
    const tiles = this.simulation.tileSizes;
    const sameBands = tiles.length === this.slices.length && tiles.every((t, i) => t === this.slices[i].tileSize);
    if (sameBands) return;
    this.slices.length = 0;
    for (const tileSize of tiles) {
      this.slices.push({ tileSize, size: 0, data: null, half: false });
    }
    this.everResolved = false;
  }

  /** Bumped by `rebuild`; a readback started under another value is stale. */
  private generation = 0;

  /**
   * Kicks off a readback if one is not already in flight. Safe to call every
   * frame; it self-throttles to one outstanding copy.
   */
  update(): void {
    if (this.pending) return;
    this.pending = true;
    void this.readback();
  }

  /**
   * Performs a readback and **waits for it**.
   *
   * The fire-and-forget `update()` is the right trade for interactive play — a
   * frame of staleness on a floating hull is invisible and a pipeline stall is
   * not — but it makes the physics irreproducible: whether a readback lands on
   * step 7 or step 9 changes where the ship ends up. Deterministic stepping calls
   * this instead, accepting the stall in exchange for a repeatable result.
   */
  async readNow(): Promise<void> {
    // Let any in-flight fire-and-forget copy retire first, so this one is not
    // racing it for the same slices. Yielding to the *macrotask* queue, not the
    // microtask queue: the outstanding readback resolves on a GPU fence, and
    // spinning on `Promise.resolve()` would starve the very task that clears it.
    while (this.pending) await new Promise((resolve) => setTimeout(resolve, 0));
    this.pending = true;
    await this.readback();
  }

  private async readback(): Promise<void> {
    try {
      const generation = this.generation;
      const targets = this.simulation.displacementTargets;
      for (let i = 0; i < this.slices.length && i < targets.length; i++) {
        // Rebuilt since this started: these targets are retired. See `rebuild`.
        if (generation !== this.generation) return;
        // The whole target. Half-float RGBA is 8 bytes a texel, so a 256 row is
        // 2048 bytes and a 512 row 4096 — both multiples of the 256-byte
        // alignment the readback needs, so no padding rows appear and the
        // partial-read trap that produced the corner bug cannot recur.
        const size = targets[i].width;
        const raw = await this.renderer.readRenderTargetPixelsAsync(
          targets[i],
          0,
          0,
          size,
          size,
        );
        if (generation !== this.generation) return;
        this.slices[i].size = size;
        adoptInto(raw, this.slices[i]);
      }
      this.everResolved = true;
    } catch {
      // A readback can fail while the device is being reconfigured (resize,
      // backend switch). Dropping the frame is correct; the next one retries.
    } finally {
      this.pending = false;
    }
  }

  /**
   * Raw displacement at an undisplaced grid point, summed over cascades.
   * `out` receives (Dx, Dy, Dz).
   */
  private displacementAt(x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
    out.set(0, 0, 0);
    for (const slice of this.slices) {
      if (!slice.data) continue;
      sampleBilinear(
        slice.data,
        slice.half,
        slice.size,
        x / slice.tileSize,
        z / slice.tileSize,
        this.scratchTexel,
      );
      out.x += this.scratchTexel[0];
      out.y += this.scratchTexel[1];
      out.z += this.scratchTexel[2];
    }
    return out;
  }

  private readonly scratchTexel = new Float32Array(4);
  private readonly scratchDisp = new THREE.Vector3();

  /**
   * Water surface height at world (x, z).
   *
   * Inverts the choppy displacement by fixed-point iteration: start from the
   * assumption that the grid point equals the world point, then repeatedly pull
   * the guess back by the horizontal displacement found there.
   */
  height(x: number, z: number): number {
    if (!this.everResolved) return 0;
    let gx = x;
    let gz = z;
    for (let i = 0; i < 4; i++) {
      const d = this.displacementAt(gx, gz, this.scratchDisp);
      gx = x - d.x;
      gz = z - d.z;
    }
    return this.displacementAt(gx, gz, this.scratchDisp).y;
  }

  /**
   * Surface normal at world (x, z), via central differences on the height field.
   * `epsilon` should be comparable to the finest wavelength you care about.
   */
  normal(x: number, z: number, epsilon = 0.6, out = new THREE.Vector3()): THREE.Vector3 {
    const hL = this.height(x - epsilon, z);
    const hR = this.height(x + epsilon, z);
    const hD = this.height(x, z - epsilon);
    const hU = this.height(x, z + epsilon);
    return out.set(hL - hR, 2 * epsilon, hD - hU).normalize();
  }

  dispose(): void {
    this.slices.length = 0;
  }
}

/**
 * Wraps to [0,1) then samples with bilinear interpolation. Half-float data is
 * decoded here, for the sixteen values a sample reads.
 */
function sampleBilinear(
  data: Float32Array | Uint16Array,
  half: boolean,
  size: number,
  u: number,
  v: number,
  out: Float32Array,
): void {
  const fx = (((u % 1) + 1) % 1) * size - 0.5;
  const fz = (((v % 1) + 1) % 1) * size - 0.5;
  const x0 = Math.floor(fx);
  const z0 = Math.floor(fz);
  const tx = fx - x0;
  const tz = fz - z0;

  const wrap = (n: number) => ((n % size) + size) % size;
  const x0w = wrap(x0);
  const x1w = wrap(x0 + 1);
  const z0w = wrap(z0);
  const z1w = wrap(z0 + 1);

  const lut = half ? halfTable() : null;
  for (let c = 0; c < 4; c++) {
    const ia = (z0w * size + x0w) * 4 + c;
    const ib = (z0w * size + x1w) * 4 + c;
    const id = (z1w * size + x0w) * 4 + c;
    const ie = (z1w * size + x1w) * 4 + c;
    const a = lut ? lut[data[ia]] : data[ia];
    const b = lut ? lut[data[ib]] : data[ib];
    const d = lut ? lut[data[id]] : data[id];
    const e = lut ? lut[data[ie]] : data[ie];
    out[c] = (a + (b - a) * tx) * (1 - tz) + (d + (e - d) * tx) * tz;
  }
}

/**
 * Copies a readback into the slice's own buffer, allocating at most once, and
 * leaves it encoded.
 *
 * Render-target readbacks come back typed to match the attachment. Half-float
 * targets yield a Uint16Array of raw IEEE-754 binary16 bit patterns, which must
 * be decoded — reading them as integers silently produces values in the tens of
 * thousands instead of the tenths of a metre they represent.
 *
 * **Decoded where they are read, not here.** This used to decode the whole
 * field on arrival: at High, three cascades of 256² RGBA, 786k conversions and
 * about 12 ms of main thread per readback, in three chunks that landed whenever
 * each copy came back. A chunk landing just before an animation frame made that
 * frame late, and Chrome then showed it a refresh late: measured from Chrome's
 * own frame reporter at High, 44% of frames were on screen for one or three
 * refreshes instead of two. That was the jitter on the hull and on the island
 * in fly mode, in every camera mode, since the readback runs whatever the
 * camera does. The copies alone measured 2%, the same as no readback at all;
 * the decode alone with no copies, 57%. A sample reads sixteen texels, and the
 * frame takes a few hundred samples, so decoding at the read is a few thousand
 * table lookups instead of most of a million conversions. The values are the
 * same floats either way.
 *
 * Still copied, never kept by reference: the array the renderer hands back is
 * not guaranteed to outlive the next readback. A copy is a memcpy, and the
 * buffer is grown, never shrunk, and reused, so a tier change that alters the
 * slice size reallocates once and then settles.
 */
function adoptInto(raw: ArrayBufferView, slice: CascadeSlice): void {
  if (raw instanceof Uint16Array) {
    let out = slice.data;
    if (!(out instanceof Uint16Array) || out.length !== raw.length) out = new Uint16Array(raw.length);
    out.set(raw);
    slice.data = out;
    slice.half = true;
    return;
  }
  const source =
    raw instanceof Float32Array ? raw : (raw as unknown as ArrayLike<number>);
  let out = slice.data;
  if (!(out instanceof Float32Array) || out.length !== source.length) out = new Float32Array(source.length);
  if (source instanceof Float32Array) out.set(source);
  else for (let i = 0; i < source.length; i++) out[i] = source[i];
  slice.data = out;
  slice.half = false;
}

/**
 * Every binary16 pattern as a float, built on first use: 256 kB, about a
 * millisecond once. Float32 holds every half exactly, so a lookup gives the
 * value `halfToFloat` would.
 */
let _halfTable: Float32Array | null = null;
function halfTable(): Float32Array {
  if (_halfTable === null) {
    _halfTable = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) _halfTable[i] = halfToFloat(i);
  }
  return _halfTable;
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
