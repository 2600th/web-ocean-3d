/**
 * Sound in space: the listener on the camera, sources at places in the world.
 *
 * The Web Audio listener is moved to the camera every frame and turned with
 * it, and a positional source is a `PannerNode` at a world position. Two kinds
 * of panning, by cost:
 *
 *  - **head-related** (`'HRTF'`) for the few hero sources where where-is-it is
 *    the point: thunder from the bolt's bearing, the hull striking something,
 *    the anchor chain, the camp fire. It convolves with measured head
 *    responses, which is what puts a sound behind or above rather than just
 *    left or right, and it costs a convolution per source.
 *  - **equal-power** for everything else (creaks, blocks, canvas, slaps,
 *    splashes): cheap, left and right only, and plenty for a sound that lasts
 *    a fifth of a second.
 *
 * Distance is *not* the panner's job here: its roll-off is off. Each sound
 * already has its distance in its level (the ship bus by how far the ship is,
 * a splash by how far it fell, thunder by how far it struck), worked out where
 * the physics is, so the panner only supplies direction.
 *
 * Without a listener pose (the bench), the listener stands at the origin
 * facing -Z, and the ship is placed from the distance and bearing the caller
 * gives (`shipDistance`, `shipPan`), so every path still has a place to come
 * from. Allocates nothing per frame.
 */

export type PanMode = 'hrtf' | 'equalpower';

/** The listener and ship pose, from `AudioSceneParams`. All optional. */
export interface SpatialInput {
  listenerX?: number;
  listenerY?: number;
  listenerZ?: number;
  listenerForwardX?: number;
  listenerForwardY?: number;
  listenerForwardZ?: number;
  shipDistance?: number;
  shipPan?: number;
  shipX?: number;
  shipY?: number;
  shipZ?: number;
  shipBowX?: number;
  shipBowZ?: number;
  shipLength?: number;
  shipBeam?: number;
}

/** A ship of this size when the caller has not said. Metres. */
const DEFAULT_LENGTH = 30;
const DEFAULT_BEAM = 8;
/** Listener moves smaller than this are not written, metres / unit vector. */
const EPSILON = 0.01;

export class Spatial {
  /** The listener: position and its horizontal forward and right. */
  x = 0;
  y = 2;
  z = 0;
  fx = 0;
  fy = 0;
  fz = -1;
  /** Horizontal right of the listener, unit. */
  rx = 1;
  rz = 0;
  /** The ship: where, which way the bow points (unit, horizontal), how big. */
  shipX = 0;
  shipY = 0;
  shipZ = -20;
  bowX = 1;
  bowZ = 0;
  halfLength = DEFAULT_LENGTH / 2;
  halfBeam = DEFAULT_BEAM / 2;
  hasShip = false;

  private readonly ctx: BaseAudioContext;
  private readonly written = new Float64Array(6).fill(Number.NaN);

  /**
   * One head-related panner made at construction and kept. Chrome loads its
   * head responses when the first such panner is made; made first in the
   * middle of an offline render (the bench) that load never completes and the
   * render hangs. Made up front, it is ready before anything plays.
   */
  private readonly warm: PannerNode | null;

  constructor(ctx: BaseAudioContext) {
    this.ctx = ctx;
    this.warm = this.panner(0, 0, 0, 'hrtf');
  }

  dispose(): void {
    this.warm?.disconnect();
  }

  /** Moves the listener to the camera and the ship to its hull. */
  update(p: Readonly<SpatialInput>, live: boolean): void {
    this.x = fin(p.listenerX, 0);
    this.y = fin(p.listenerY, 2);
    this.z = fin(p.listenerZ, 0);
    let fx = fin(p.listenerForwardX, 0);
    let fy = fin(p.listenerForwardY, 0);
    let fz = fin(p.listenerForwardZ, -1);
    const len = Math.hypot(fx, fy, fz);
    if (len < 1e-6) {
      fx = 0;
      fy = 0;
      fz = -1;
    } else {
      fx /= len;
      fy /= len;
      fz /= len;
    }
    // Looking straight down or up: the horizontal heading is kept from before.
    const h = Math.hypot(fx, fz);
    if (h > 1e-3) {
      this.rx = -fz / h;
      this.rz = fx / h;
    }
    // The up vector is world up, so a camera looking steeply down still has a
    // forward the panner can use: pull the forward toward the horizon.
    if (Math.abs(fy) > 0.95) {
      fy = Math.sign(fy) * 0.95;
      const k = Math.sqrt(1 - fy * fy) / Math.max(1e-3, h);
      fx = h > 1e-3 ? fx * k : this.rz * Math.sqrt(1 - fy * fy);
      fz = h > 1e-3 ? fz * k : -this.rx * Math.sqrt(1 - fy * fy);
    }
    this.fx = fx;
    this.fy = fy;
    this.fz = fz;

    const distance = p.shipDistance;
    this.hasShip = distance !== undefined && Number.isFinite(distance);
    if (p.shipX !== undefined && Number.isFinite(p.shipX)) {
      this.shipX = p.shipX;
      this.shipY = fin(p.shipY, 0);
      this.shipZ = fin(p.shipZ, 0);
    } else if (this.hasShip) {
      // From distance and bearing: in front of the listener, to the side the pan says.
      const d = Math.max(0, distance as number);
      const pan = Math.max(-1, Math.min(1, fin(p.shipPan, 0)));
      const ahead = Math.sqrt(Math.max(0, 1 - pan * pan));
      const hx = this.rz;
      const hz = -this.rx;
      this.shipX = this.x + d * (pan * this.rx + ahead * hx);
      this.shipY = 0;
      this.shipZ = this.z + d * (pan * this.rz + ahead * hz);
    }
    const bx = fin(p.shipBowX, Number.NaN);
    const bz = fin(p.shipBowZ, Number.NaN);
    const bl = Math.hypot(bx, bz);
    if (bl > 1e-3) {
      this.bowX = bx / bl;
      this.bowZ = bz / bl;
    } else {
      // Beam on to the listener.
      this.bowX = this.rx;
      this.bowZ = this.rz;
    }
    this.halfLength = Math.max(2, fin(p.shipLength, DEFAULT_LENGTH)) / 2;
    this.halfBeam = Math.max(1, fin(p.shipBeam, DEFAULT_BEAM)) / 2;

    if (live) this.writeListener();
  }

  /** Writes the pose now, whatever was written before (a resume). */
  flush(): void {
    this.written.fill(Number.NaN);
    this.writeListener();
  }

  /**
   * A point on the ship, in fractions of her half-length (+ toward the bow),
   * metres up from the waterline, and fractions of her half-beam (+ to
   * starboard), as a world position in `out`.
   */
  shipPoint(along: number, up: number, across: number, out: { x: number; y: number; z: number }): void {
    const a = along * this.halfLength;
    const c = across * this.halfBeam;
    // Starboard is the bow turned a right angle toward +Z when the bow is +X.
    const sx = -this.bowZ;
    const sz = this.bowX;
    out.x = this.shipX + this.bowX * a + sx * c;
    out.y = this.shipY + up;
    out.z = this.shipZ + this.bowZ * a + sz * c;
  }

  /**
   * A panner at a world position, distance left to the caller (roll-off off).
   * Null where the platform has none; the caller then plays it unpanned.
   */
  panner(x: number, y: number, z: number, mode: PanMode): PannerNode | null {
    const ctx = this.ctx as BaseAudioContext & { createPanner?: () => PannerNode };
    if (typeof ctx.createPanner !== 'function') return null;
    const node = ctx.createPanner();
    node.panningModel = mode === 'hrtf' ? 'HRTF' : 'equalpower';
    node.distanceModel = 'inverse';
    node.refDistance = 1;
    node.rolloffFactor = 0;
    setPannerPosition(node, x, y, z);
    return node;
  }

  /** Horizontal bearing of a world point from the listener: -1 left .. 1 right. */
  panOf(x: number, z: number): number {
    const dx = x - this.x;
    const dz = z - this.z;
    const d = Math.hypot(dx, dz);
    return d > 1e-3 ? (dx * this.rx + dz * this.rz) / d : 0;
  }

  private writeListener(): void {
    const l = this.ctx.listener as AudioListener | undefined;
    if (l === undefined) return;
    const w = this.written;
    const moved =
      !(Math.abs(w[0] - this.x) < EPSILON) ||
      !(Math.abs(w[1] - this.y) < EPSILON) ||
      !(Math.abs(w[2] - this.z) < EPSILON) ||
      !(Math.abs(w[3] - this.fx) < EPSILON * 0.1) ||
      !(Math.abs(w[4] - this.fy) < EPSILON * 0.1) ||
      !(Math.abs(w[5] - this.fz) < EPSILON * 0.1);
    if (!moved) return;
    w[0] = this.x;
    w[1] = this.y;
    w[2] = this.z;
    w[3] = this.fx;
    w[4] = this.fy;
    w[5] = this.fz;
    if (l.positionX !== undefined) {
      l.positionX.value = this.x;
      l.positionY.value = this.y;
      l.positionZ.value = this.z;
      l.forwardX.value = this.fx;
      l.forwardY.value = this.fy;
      l.forwardZ.value = this.fz;
      l.upX.value = 0;
      l.upY.value = 1;
      l.upZ.value = 0;
    } else {
      const legacy = l as AudioListener & {
        setPosition?: (x: number, y: number, z: number) => void;
        setOrientation?: (x: number, y: number, z: number, ux: number, uy: number, uz: number) => void;
      };
      legacy.setPosition?.(this.x, this.y, this.z);
      legacy.setOrientation?.(this.fx, this.fy, this.fz, 0, 1, 0);
    }
  }
}

/** Moves a panner, on the AudioParams where they exist. */
export function setPannerPosition(node: PannerNode, x: number, y: number, z: number): void {
  if (node.positionX !== undefined) {
    node.positionX.value = x;
    node.positionY.value = y;
    node.positionZ.value = z;
  } else {
    (node as PannerNode & { setPosition?: (x: number, y: number, z: number) => void }).setPosition?.(x, y, z);
  }
}

function fin(v: number | undefined, fallback: number): number {
  return v !== undefined && Number.isFinite(v) ? v : fallback;
}
