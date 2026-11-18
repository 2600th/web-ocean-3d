import * as THREE from 'three/webgpu';
import { attribute, color, float, select, vec4 } from 'three/tsl';
import { uniform } from '../core/uniform';
import { mulberry32 } from '../core/random';

/**
 * Cloud-to-sea lightning for heavy weather.
 *
 * Three things have to land together or a strike reads as a glitch rather than
 * as weather, and the renderer already had one of them: `AudioSystem.playThunder`
 * synthesised a distance-shaped crack and roll that nothing ever called.
 *
 *  1. **The bolt** — a branching channel from the cloud base to the sea, drawn
 *     as an HDR emissive ribbon so the bloom stage gives it its glow for free.
 *     Midpoint displacement with a few forks. Forks are thinner and dimmer than
 *     the channel and fade toward their tips, the way a photograph shows them:
 *     only the path that connected carries the return stroke's current. The
 *     ribbons are written into buffers allocated once, so a strike allocates no
 *     geometry.
 *  2. **The leader** — before the flash, a faint stepped leader works down from
 *     the cloud for a few frames; the return stroke follows when it touches the
 *     sea. Real leaders take tens of milliseconds, which is a handful of frames.
 *  3. **The flash** — the whole deck lights from inside. A real stroke is several
 *     return strokes inside ~200 ms, so the envelope is a short train of
 *     decaying pulses, not one blink. Exposed as a scalar the colour grade adds
 *     to the HDR frame and the clouds can use as ambient.
 *  4. **The thunder** — delayed by the distance over the speed of sound and
 *     shaped by it, so near strikes crack and far ones roll.
 *
 * **Flash safety (WCAG 2.3.1).** No more than three flashes in any one second:
 * every return-stroke pulse is counted against a one-second window across
 * strikes, and a pulse that would be the fourth is not fired. With
 * `setReduceFlashes(true)` a strike is one soft pulse — a slow rise and fall
 * with no return-stroke train and no leader flicker, at about half the peak.
 *
 * Seeded and driven by the caller's clock, so a deterministic capture can switch
 * it off and a session replays the same storm.
 */

/** Strikes are scheduled only when `storminess` is at least this. */
const STORM_THRESHOLD = 0.55;
/** Mean seconds between strikes at full storminess; sparser below it. */
const MEAN_INTERVAL = 7;
const MIN_DISTANCE_M = 900;
const MAX_DISTANCE_M = 6500;
/** Now and then a strike lands close: this share of them, 120–300 m away. */
const CLOSE_CHANCE = 0.1;
const CLOSE_MIN_M = 120;
const CLOSE_MAX_M = 300;
/** Seconds the stepped leader takes to reach the sea. */
const LEADER_SECONDS = 0.07;
/** Return strokes after the leader: start time (s) and relative strength. Three at most. */
const STROKES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0.09, 0.6],
  [0.21, 0.8],
];
const STROKE_DECAY = 0.045;
const FLASH_SECONDS = 0.55;
/** WCAG 2.3.1: flashes allowed in any one-second window. */
const MAX_FLASHES_PER_SECOND = 3;
/** The reduced-flash pulse: one raised-cosine swell, seconds, and its peak. */
const SOFT_SECONDS = 0.9;
const SOFT_PEAK = 0.5;
/** Bolt radiance; high enough that the bloom threshold always catches it. */
const BOLT_RADIANCE = 220;
/** Brightness of the leader against the return stroke. */
const LEADER_GLOW = 0.12;
/** Vertex budget for one bolt, channel and forks together. */
const MAX_VERTICES = 9000;
/** Midpoint-displacement levels on the main channel. */
const LEVELS = 7;

export interface LightningStrike {
  x: number;
  z: number;
  distance: number;
}

export class Lightning {
  readonly object: THREE.Mesh;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly geometry: THREE.BufferGeometry;
  private readonly positions: THREE.BufferAttribute;
  /** Per vertex: x brightness 0..1 (forks dimmer, tapering), y height along the path, 0 cloud .. 1 sea. */
  private readonly glow: THREE.BufferAttribute;
  private random = mulberry32(0x5eed_b017);
  private untilNext = 4;
  private age = Infinity;
  private flash = 0;
  /** Running clock, seconds, for the flash-rate window. */
  private time = 0;
  /** Times the last return strokes fired (pending ones included), oldest first. */
  private readonly pulseTimes: number[] = [];
  /** This strike's return strokes, as [start, strength], after the safety limit. */
  private readonly pulses: Array<[number, number]> = [];
  private reduceFlashes = false;
  /** Forks and leader on; off on the lowest tier. */
  private detailed = true;
  /** Where the current (or last) stroke came down, for the flash light. */
  private readonly strikeAt = { x: 0, z: 0, distance: Infinity };
  private cloudBase = 900;
  /** How far down the leader has reached, 0..1; above 1 the channel is connected. */
  private readonly uReach = uniform(2);
  private readonly uBrightness = uniform(0);
  /** Point buffers per fork depth, reused. */
  private readonly points = [0, 1, 2].map(() => new Float32Array((2 ** LEVELS + 1) * 4));

  constructor() {
    this.geometry = new THREE.BufferGeometry();
    this.positions = new THREE.BufferAttribute(new Float32Array(MAX_VERTICES * 3), 3);
    this.positions.setUsage(THREE.DynamicDrawUsage);
    this.glow = new THREE.BufferAttribute(new Float32Array(MAX_VERTICES * 2), 2);
    this.glow.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('position', this.positions);
    this.geometry.setAttribute('boltGlow', this.glow);
    this.geometry.setDrawRange(0, 0);

    this.material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      // Writes depth so the volumetric fog treats it as the thing kilometres
      // away that it is, rather than as sky behind the whole fog column.
      depthWrite: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      fog: false,
    });
    const g: any = attribute('boltGlow', 'vec2');
    // Below the leader's tip nothing is drawn yet; above it the path glows
    // faintly until the return stroke lights it at full.
    const lit = select(g.y.lessThan(this.uReach), float(1), float(0));
    const strength = select(this.uReach.greaterThan(1), g.x, g.x.mul(LEADER_GLOW));
    const level = lit.mul(strength).mul(this.uBrightness);
    this.material.colorNode = vec4(color(new THREE.Color(0.78, 0.86, 1).multiplyScalar(BOLT_RADIANCE)).mul(level), 1);
    this.object = new THREE.Mesh(this.geometry, this.material);
    this.object.name = 'lightning-bolt';
    this.object.frustumCulled = false;
    this.object.visible = false;
    this.object.renderOrder = 10;
  }

  /** The last strike's ground point and its distance from the eye at the time. */
  get lastStrike(): Readonly<{ x: number; z: number; distance: number }> {
    return this.strikeAt;
  }

  /** Cloud base the bolt starts from, metres. */
  get cloudBaseHeight(): number {
    return this.cloudBase;
  }

  /** 0..~1.4 — how much of the frame the current flash lights. */
  get flashIntensity(): number {
    return this.flash;
  }

  setCloudBase(metres: number): void {
    this.cloudBase = Math.max(300, metres);
  }

  /** One soft pulse per strike instead of a return-stroke train. */
  setReduceFlashes(on: boolean): void {
    this.reduceFlashes = on;
  }

  /** The tier's weather detail, 0..1: below 0.5 there are no forks and no leader. */
  setDetail(detail: number): void {
    this.detailed = detail >= 0.5;
  }

  /** Rewinds the schedule, for deterministic capture. */
  reset(seed = 0x5eed_b017): void {
    this.random = mulberry32(seed);
    this.untilNext = 4;
    this.age = Infinity;
    this.flash = 0;
    this.pulseTimes.length = 0;
    this.pulses.length = 0;
    this.object.visible = false;
  }

  /**
   * Advances the storm. Returns the strike that began this frame, if any, so the
   * caller can schedule its thunder.
   */
  update(
    dt: number,
    camera: THREE.Camera,
    storminess: number,
    enabled: boolean,
  ): LightningStrike | null {
    let started: LightningStrike | null = null;
    this.time += dt;
    if (enabled && storminess >= STORM_THRESHOLD) {
      this.untilNext -= dt;
      if (this.untilNext <= 0) {
        started = this.strike(camera);
        const rate = (storminess - STORM_THRESHOLD) / (1 - STORM_THRESHOLD);
        const mean = MEAN_INTERVAL / Math.max(0.25, rate);
        // Exponential inter-arrival with a floor, so strikes cluster the way
        // they do in a real cell without ever overlapping.
        this.untilNext = 1.2 + -Math.log(1 - this.random() * 0.98) * mean;
      }
    }

    this.age += dt;
    const leader = this.reduceFlashes || !this.detailed ? 0 : LEADER_SECONDS;
    let flash = 0;
    let bolt = 0;
    if (this.reduceFlashes) {
      // One swell: raised cosine, no train of pulses.
      if (this.age < SOFT_SECONDS) {
        flash = SOFT_PEAK * 0.5 * (1 - Math.cos((this.age / SOFT_SECONDS) * Math.PI * 2));
      }
      bolt = flash / SOFT_PEAK;
      this.uReach.value = 2;
    } else {
      for (const [start, strength] of this.pulses) {
        const t = this.age - leader - start;
        if (t >= 0) flash += strength * Math.exp(-t / STROKE_DECAY);
      }
      flash = this.age < leader + FLASH_SECONDS ? Math.min(1.4, flash) : 0;
      if (this.age < leader) {
        this.uReach.value = this.age / leader;
        bolt = 1;
      } else {
        this.uReach.value = 2;
        bolt = Math.min(1, flash);
      }
    }
    this.flash = flash;
    const showing = this.reduceFlashes ? this.age < SOFT_SECONDS : this.age < leader + FLASH_SECONDS;
    this.object.visible = showing && bolt > 0.08;
    this.uBrightness.value = bolt;
    return started;
  }

  /**
   * Starts a strike now. Public for tests that want a strike on demand; the
   * schedule in `update` is the normal way in.
   */
  strike(camera: THREE.Camera): LightningStrike {
    const rand = this.random;
    // In front of the viewer more often than behind, because a strike nobody
    // sees is only thunder — but not always, because a flash from behind that
    // lights the sails is its own good moment.
    camera.getWorldDirection(_forward);
    const facing = Math.atan2(_forward.z, _forward.x);
    const spread = rand() < 0.75 ? 0.42 : Math.PI;
    const bearing = facing + (rand() * 2 - 1) * spread;
    const close = rand() < CLOSE_CHANCE;
    const distance = close
      ? CLOSE_MIN_M + rand() * (CLOSE_MAX_M - CLOSE_MIN_M)
      : MIN_DISTANCE_M + rand() * (MAX_DISTANCE_M - MIN_DISTANCE_M);
    const x = camera.position.x + Math.cos(bearing) * distance;
    const z = camera.position.z + Math.sin(bearing) * distance;

    this.strikeAt.x = x;
    this.strikeAt.z = z;
    this.strikeAt.distance = distance;
    this.buildBolt(x, z, distance, rand);
    this.schedulePulses();
    this.age = 0;
    return { x, z, distance };
  }

  /** The return strokes for a new strike, within three flashes a second. */
  private schedulePulses(): void {
    this.pulses.length = 0;
    const leader = this.reduceFlashes || !this.detailed ? 0 : LEADER_SECONDS;
    while (this.pulseTimes.length > 0 && this.pulseTimes[0] < this.time - 1) this.pulseTimes.shift();
    if (this.reduceFlashes) {
      this.pulseTimes.push(this.time);
      return;
    }
    for (const [start, strength] of STROKES) {
      const at = this.time + leader + start;
      let inWindow = 0;
      for (const t of this.pulseTimes) if (t > at - 1) inWindow++;
      if (inWindow >= MAX_FLASHES_PER_SECOND) continue;
      this.pulseTimes.push(at);
      this.pulses.push([start, strength]);
    }
  }

  private buildBolt(x: number, z: number, distance: number, rand: () => number): void {
    // Width scales with distance so the channel stays a few pixels wide on
    // screen however far away it is — a real channel is centimetres, but what
    // the eye records is the glow around it. A close strike keeps a floor so it
    // is not a hairline.
    const width = Math.max(1.2, distance * 0.0022);
    const pos = this.positions.array as Float32Array;
    const glow = this.glow.array as Float32Array;
    let n = 0;
    const cloudBase = this.cloudBase;

    const put = (px: number, py: number, pz: number, g: number): void => {
      pos[n * 3] = px;
      pos[n * 3 + 1] = py;
      pos[n * 3 + 2] = pz;
      glow[n * 2] = g;
      glow[n * 2 + 1] = 1 - py / cloudBase;
      n++;
    };
    const ribbon = (ax: number, ay: number, az: number, bx: number, by: number, bz: number, wa: number, wb: number, ga: number, gb: number): void => {
      if (n + 12 > MAX_VERTICES) return;
      // Two crossed quads so it reads from any bearing without a per-frame
      // billboard update; each end has its own width, so a fork tapers.
      for (let k = 0; k < 2; k++) {
        const oxa = k === 0 ? wa / 2 : 0;
        const oza = k === 0 ? 0 : wa / 2;
        const oxb = k === 0 ? wb / 2 : 0;
        const ozb = k === 0 ? 0 : wb / 2;
        put(ax - oxa, ay, az - oza, ga);
        put(ax + oxa, ay, az + oza, ga);
        put(bx + oxb, by, bz + ozb, gb);
        put(ax - oxa, ay, az - oza, ga);
        put(bx + oxb, by, bz + ozb, gb);
        put(bx - oxb, by, bz - ozb, gb);
      }
    };

    const channel = (
      fx: number, fy: number, fz: number,
      tx: number, ty: number, tz: number,
      depth: number, w: number, g: number,
    ): void => {
      const levels = LEVELS - depth;
      const pts = this.points[depth];
      const count = displace(pts, fx, fy, fz, tx, ty, tz, levels, rand);
      const forking = this.detailed && depth < 2;
      for (let i = 0; i < count - 1; i++) {
        const a = i * 3;
        const b = a + 3;
        // A fork tapers to nothing and fades along its length; the channel
        // that reached the sea keeps its width and brightness all the way.
        const ta = depth === 0 ? 1 : 1 - i / (count - 1);
        const tb = depth === 0 ? 1 : 1 - (i + 1) / (count - 1);
        ribbon(
          pts[a], pts[a + 1], pts[a + 2], pts[b], pts[b + 1], pts[b + 2],
          w * (0.25 + 0.75 * ta), w * (0.25 + 0.75 * tb), g * ta, g * tb,
        );
        if (forking && i > 1 && rand() < 0.07) {
          const drop = (fy - ty) * (0.15 + rand() * 0.25);
          const ex = pts[a] + (rand() * 2 - 1) * drop * 0.8;
          const ey = Math.max(ty, pts[a + 1] - drop);
          const ez = pts[a + 2] + (rand() * 2 - 1) * drop * 0.8;
          // Copy the start: the recursion reuses the next depth's buffer only.
          channel(pts[a], pts[a + 1], pts[a + 2], ex, ey, ez, depth + 1, w * 0.45, g * (depth === 0 ? 0.32 : 0.5));
        }
      }
    };

    channel(
      x + (rand() * 2 - 1) * 300, cloudBase, z + (rand() * 2 - 1) * 300,
      x, 0, z,
      0, width, 1,
    );

    this.geometry.setDrawRange(0, n);
    this.positions.addUpdateRange(0, n * 3);
    this.positions.needsUpdate = true;
    this.glow.addUpdateRange(0, n * 2);
    this.glow.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

const _forward = new THREE.Vector3();

/**
 * Midpoint displacement between two points, `levels` deep, into `out` as
 * packed xyz. Returns the point count.
 */
function displace(
  out: Float32Array,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  levels: number,
  rand: () => number,
): number {
  out[0] = ax; out[1] = ay; out[2] = az;
  const last = 2 ** levels;
  out[last * 3] = bx; out[last * 3 + 1] = by; out[last * 3 + 2] = bz;
  let amplitude = Math.hypot(bx - ax, by - ay, bz - az) * 0.18;
  // Fill in place, coarse to fine, so the result is the same as subdividing a
  // list: each pass halves the stride and displaces the new midpoints.
  for (let stride = last; stride > 1; stride >>= 1) {
    const half = stride >> 1;
    for (let i = 0; i < last; i += stride) {
      const p = i * 3;
      const q = (i + stride) * 3;
      const m = (i + half) * 3;
      out[m] = (out[p] + out[q]) / 2 + (rand() * 2 - 1) * amplitude;
      out[m + 2] = (out[p + 2] + out[q + 2]) / 2 + (rand() * 2 - 1) * amplitude;
      out[m + 1] = (out[p + 1] + out[q + 1]) / 2 + (rand() * 2 - 1) * amplitude * 0.25;
    }
    amplitude *= 0.55;
  }
  return last + 1;
}
