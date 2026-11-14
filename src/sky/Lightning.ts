import * as THREE from 'three/webgpu';
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
 *     Midpoint displacement with a few forks; rebuilt per strike, which is a few
 *     hundred vertices and costs nothing measurable.
 *  2. **The flash** — the whole deck lights from inside. A real stroke is three
 *     or four return strokes inside ~200 ms, so the envelope is a short train of
 *     decaying pulses, not one blink. Exposed as a scalar the colour grade adds
 *     to the HDR frame and the clouds can use as ambient.
 *  3. **The thunder** — delayed by the distance over the speed of sound and
 *     shaped by it, so near strikes crack and far ones roll.
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
/** Return strokes in one flash: start time (s) and relative strength. */
const STROKES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0.07, 0.55],
  [0.16, 0.8],
  [0.31, 0.35],
];
const STROKE_DECAY = 0.045;
const FLASH_SECONDS = 0.55;
/** Bolt radiance; high enough that the bloom threshold always catches it. */
const BOLT_RADIANCE = 220;

export interface LightningStrike {
  x: number;
  z: number;
  distance: number;
}

export class Lightning {
  readonly object: THREE.Mesh;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private random = mulberry32(0x5eed_b017);
  private untilNext = 4;
  private age = Infinity;
  private flash = 0;
  /** Where the current (or last) stroke came down, for the flash light. */
  private readonly strikeAt = { x: 0, z: 0, distance: Infinity };
  private cloudBase = 900;

  constructor() {
    this.material = new THREE.MeshBasicNodeMaterial({
      color: new THREE.Color(0.78, 0.86, 1).multiplyScalar(BOLT_RADIANCE),
      transparent: true,
      // Writes depth so the volumetric fog treats it as the thing kilometres
      // away that it is, rather than as sky behind the whole fog column.
      depthWrite: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.object = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    this.object.name = 'lightning-bolt';
    this.object.frustumCulled = false;
    this.object.visible = false;
    this.object.renderOrder = 10;
  }

  /** 0..1 — how much of the frame the current flash lights. */
  /** The last strike's ground point and its distance from the eye at the time. */
  get lastStrike(): Readonly<{ x: number; z: number; distance: number }> {
    return this.strikeAt;
  }

  /** Cloud base the bolt starts from, metres. */
  get cloudBaseHeight(): number {
    return this.cloudBase;
  }

  get flashIntensity(): number {
    return this.flash;
  }

  setCloudBase(metres: number): void {
    this.cloudBase = Math.max(300, metres);
  }

  /** Rewinds the schedule, for deterministic capture. */
  reset(seed = 0x5eed_b017): void {
    this.random = mulberry32(seed);
    this.untilNext = 4;
    this.age = Infinity;
    this.flash = 0;
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
    let flash = 0;
    for (const [start, strength] of STROKES) {
      const t = this.age - start;
      if (t >= 0) flash += strength * Math.exp(-t / STROKE_DECAY);
    }
    this.flash = this.age < FLASH_SECONDS ? Math.min(1.4, flash) : 0;
    this.object.visible = this.age < FLASH_SECONDS && this.flash > 0.08;
    this.material.opacity = Math.min(1, this.flash);
    return started;
  }

  private strike(camera: THREE.Camera): LightningStrike {
    const rand = this.random;
    // In front of the viewer more often than behind, because a strike nobody
    // sees is only thunder — but not always, because a flash from behind that
    // lights the sails is its own good moment.
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const facing = Math.atan2(forward.z, forward.x);
    const spread = rand() < 0.75 ? 0.42 : Math.PI;
    const bearing = facing + (rand() * 2 - 1) * spread;
    const distance = MIN_DISTANCE_M + rand() * (MAX_DISTANCE_M - MIN_DISTANCE_M);
    const x = camera.position.x + Math.cos(bearing) * distance;
    const z = camera.position.z + Math.sin(bearing) * distance;

    this.strikeAt.x = x;
    this.strikeAt.z = z;
    this.strikeAt.distance = distance;
    this.buildBolt(x, z, distance, rand);
    this.age = 0;
    return { x, z, distance };
  }

  private buildBolt(x: number, z: number, distance: number, rand: () => number): void {
    // Width scales with distance so the channel stays a few pixels wide on
    // screen however far away it is — a real channel is centimetres, but what
    // the eye records is the glow around it.
    const width = Math.max(2.5, distance * 0.0022);
    const positions: number[] = [];

    const channel = (from: THREE.Vector3, to: THREE.Vector3, depth: number, w: number): void => {
      const points = displace(from, to, 7 - depth, rand);
      for (let i = 0; i < points.length - 1; i++) {
        pushRibbon(positions, points[i], points[i + 1], w);
        // Forks: shorter, thinner, heading down and away.
        if (depth < 2 && i > 1 && rand() < 0.07) {
          const a = points[i];
          const drop = (from.y - to.y) * (0.15 + rand() * 0.25);
          const end = new THREE.Vector3(
            a.x + (rand() * 2 - 1) * drop * 0.8,
            Math.max(to.y, a.y - drop),
            a.z + (rand() * 2 - 1) * drop * 0.8,
          );
          channel(a, end, depth + 1, w * 0.45);
        }
      }
    };

    const top = new THREE.Vector3(
      x + (rand() * 2 - 1) * 300,
      this.cloudBase,
      z + (rand() * 2 - 1) * 300,
    );
    channel(top, new THREE.Vector3(x, 0, z), 0, width);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.object.geometry.dispose();
    this.object.geometry = geometry;
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
  }
}

/** Midpoint displacement between two points, `levels` deep. */
function displace(
  a: THREE.Vector3,
  b: THREE.Vector3,
  levels: number,
  rand: () => number,
): THREE.Vector3[] {
  let points = [a.clone(), b.clone()];
  let amplitude = a.distanceTo(b) * 0.18;
  for (let level = 0; level < levels; level++) {
    const next: THREE.Vector3[] = [points[0]];
    for (let i = 0; i < points.length - 1; i++) {
      const mid = points[i].clone().lerp(points[i + 1], 0.5);
      mid.x += (rand() * 2 - 1) * amplitude;
      mid.z += (rand() * 2 - 1) * amplitude;
      mid.y += (rand() * 2 - 1) * amplitude * 0.25;
      next.push(mid, points[i + 1]);
    }
    points = next;
    amplitude *= 0.55;
  }
  return points;
}

/**
 * A flat ribbon segment, as two crossed quads so it reads from any bearing
 * without a per-frame billboard update.
 */
function pushRibbon(out: number[], a: THREE.Vector3, b: THREE.Vector3, width: number): void {
  const h = width / 2;
  for (const [ox, oz] of [
    [h, 0],
    [0, h],
  ] as const) {
    out.push(
      a.x - ox, a.y, a.z - oz,
      a.x + ox, a.y, a.z + oz,
      b.x + ox, b.y, b.z + oz,
      a.x - ox, a.y, a.z - oz,
      b.x + ox, b.y, b.z + oz,
      b.x - ox, b.y, b.z - oz,
    );
  }
}
