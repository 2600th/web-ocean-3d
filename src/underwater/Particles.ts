import * as THREE from 'three/webgpu';
import {
  attribute,
  cos,
  float,
  max,
  mix,
  mod,
  sin,
  smoothstep,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDown } from '../core/tslMath';
import { SEEDS, fillRandom, mulberry32 } from '../core/random';

/**
 * Suspended matter in the water column.
 *
 * Two populations, because real water has two:
 *
 *   marine snow   fine organic detritus, near-neutrally buoyant, sinking at a
 *                 few centimetres per second with a slow lateral sway. Dim and
 *                 almost colourless — it reads as texture in the fog, not as
 *                 sparkle.
 *   bubble trains rising columns of air, an order of magnitude faster, brighter
 *                 and distinctly cyan, wobbling as they go because a bubble's
 *                 wake is unstable. In the reference capture they arrive as a
 *                 tight cluster rather than an even scatter, so they are grouped
 *                 into a small number of columns.
 *
 * Both are instanced camera-facing quads on `SpriteNodeMaterial`, which gives
 * real world-space sizes on WebGPU, where point primitives are clamped to a
 * single pixel.
 *
 * Motion is entirely in the vertex stage, from per-instance seed attributes and
 * one clock uniform. The CPU never writes a particle position — `update` only
 * pushes the camera position into a uniform so the wrap volume re-centres, which
 * is a single `Vector3.copy` per frame with no allocation.
 *
 * The wrap is done in *world* space around the viewer rather than by translating
 * a container object, so the particles are genuinely stationary in the world and
 * parallax correctly as the camera moves. A container that simply followed the
 * camera would drag the whole field along with it and kill the parallax.
 */

/** Half-extents of the marine snow volume around the viewer, metres. */
const SNOW_RADIUS = 15;
const SNOW_HEIGHT = 12;

/** Half-extents of the bubble volume, metres. Taller — columns need run-up. */
const BUBBLE_RADIUS = 12;
const BUBBLE_HEIGHT = 15;

/** [underwater-look-2] Particles shrink away inside this distance of the eye, metres. */
const NEAR_FADE = 4;

/** Bubbles are grouped into this many rising trains. */
const BUBBLE_COLUMNS = 14;
/** Scatter of a bubble around its column axis, metres. */
const COLUMN_JITTER = 0.55;

/**
 * [underwater-look-3] Round 11. Seeps: bubble streams anchored to the reef, the
 * "few rising bubble streams near the reef" the round-11 review asked for
 * (abyssal-garden's mound has strings of large, clear bubbles rising past
 * it). The trains above wrap round the viewer and belong to no place; these
 * stand at fixed sites on the bed (`setSeeps`), `SEEP_BUBBLES` per stream,
 * rising at `SEEP_RISE` m/s (a few-millimetre bubble's terminal speed) over
 * `SEEP_HEIGHT` metres, `SEEP_SIZE` across at the vent and growing by
 * `SEEP_GROWTH` on the way up as the pressure falls, fading out over the top
 * fifth (by size). Opaque, unlike the trains; see `buildSeepMaterial`.
 */
const SEEP_BUBBLES = 44;
const SEEP_RISE = 0.32;
const SEEP_HEIGHT = 7;
const SEEP_SIZE: readonly [number, number] = [0.035, 0.08];
const SEEP_GROWTH = 0.8;

/** Fraction of the budget spent on marine snow; the rest becomes bubbles. */
const SNOW_SHARE = 0.82;

/** Clock wrap, seconds. Long enough to be invisible, short enough for float32. */
const CLOCK_WRAP = 3600;

/**
 * `vec3` under a loose signature. Composing per-component expressions out of
 * `attribute()` values produces `any`, which the overloaded TSL typings then
 * resolve to the wrong constructor; the runtime behaviour is unaffected.
 */
const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => any;

export class UnderwaterParticles {
  readonly object: THREE.Object3D;

  private snowGeometry: THREE.InstancedBufferGeometry;
  private bubbleGeometry: THREE.InstancedBufferGeometry;

  private readonly snowMaterial: THREE.SpriteNodeMaterial;
  private readonly bubbleMaterial: THREE.SpriteNodeMaterial;

  private readonly snow: THREE.Mesh;
  private readonly bubbles: THREE.Mesh;
  /** [underwater-look-3] Round 11: see `SEEP_BUBBLES`. Built by `setSeeps`. */
  private seeps: THREE.Mesh | null = null;
  private seepMaterial: THREE.SpriteNodeMaterial | null = null;

  private count: number;
  private clock = 0;
  private disposed = false;

  // --- uniforms ------------------------------------------------------------
  private readonly uTime = uniform(0);
  /** Centre of the wrap volume — the viewer. */
  private readonly uCenter = uniform(new THREE.Vector3());
  private readonly uSnowOpacity = uniform(0.75);
  private readonly uBubbleOpacity = uniform(1.0);
  private readonly uSnowColor = uniform(new THREE.Color(0.66, 0.79, 0.86));
  private readonly uBubbleColor = uniform(new THREE.Color(0.55, 0.94, 1.0));
  private readonly uSnowFall = uniform(0.16);
  private readonly uBubbleRise = uniform(1.35);

  constructor(count: number) {
    this.count = Math.max(0, Math.floor(count));

    this.object = new THREE.Object3D();
    this.object.name = 'underwater-particles';
    this.object.frustumCulled = false;
    // The wrap happens in world space; the container itself never moves.
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();

    this.snowGeometry = buildSnowGeometry(this.snowCount());
    this.bubbleGeometry = buildBubbleGeometry(this.bubbleCount());

    this.snowMaterial = this.buildSnowMaterial();
    this.bubbleMaterial = this.buildBubbleMaterial();

    this.snow = new THREE.Mesh(this.snowGeometry, this.snowMaterial);
    this.snow.name = 'marine-snow';
    this.snow.frustumCulled = false;
    this.snow.renderOrder = 140;

    this.bubbles = new THREE.Mesh(this.bubbleGeometry, this.bubbleMaterial);
    this.bubbles.name = 'bubble-trains';
    this.bubbles.frustumCulled = false;
    this.bubbles.renderOrder = 141;

    this.object.add(this.snow, this.bubbles);
    this.object.visible = false;
  }

  getCount(): number {
    return this.count;
  }

  /** Rebuilds both instance buffers. Not cheap — call it on quality changes only. */
  setCount(count: number): void {
    const next = Math.max(0, Math.floor(count));
    if (next === this.count) return;
    this.count = next;

    this.snowGeometry.dispose();
    this.bubbleGeometry.dispose();

    this.snowGeometry = buildSnowGeometry(this.snowCount());
    this.bubbleGeometry = buildBubbleGeometry(this.bubbleCount());

    this.snow.geometry = this.snowGeometry;
    this.bubbles.geometry = this.bubbleGeometry;
  }

  setVisible(v: boolean): void {
    this.object.visible = v && this.count > 0;
  }

  /**
   * [underwater-look-3] Round 11. Bubble streams rising from these points on
   * the bed; see `SEEP_BUBBLES`. Call once, after the seabed is placed.
   */
  setSeeps(sites: readonly { x: number; y: number; z: number }[]): void {
    if (this.seeps) {
      this.object.remove(this.seeps);
      this.seeps.geometry.dispose();
      this.seeps = null;
    }
    if (sites.length === 0) return;
    this.seepMaterial ??= this.buildSeepMaterial();
    const count = sites.length * SEEP_BUBBLES;
    const geometry = buildQuad(count);
    const random = mulberry32(SEEDS.bubbles ^ 0x5ee95);
    const a = new Float32Array(count * 4);
    const b = new Float32Array(count * 4);
    for (let s = 0; s < sites.length; s++) {
      for (let i = 0; i < SEEP_BUBBLES; i++) {
        const k = (s * SEEP_BUBBLES + i) * 4;
        a[k] = sites[s].x;
        a[k + 1] = sites[s].y;
        a[k + 2] = sites[s].z;
        a[k + 3] = random() * 40;
        // Evenly spaced along the stream with some bunching: a string of pearls.
        b[k] = (i + random() * 0.6) / SEEP_BUBBLES;
        b[k + 1] = random();
        b[k + 2] = random();
        b[k + 3] = random();
      }
    }
    geometry.setAttribute('seepA', new THREE.InstancedBufferAttribute(a, 4));
    geometry.setAttribute('seepB', new THREE.InstancedBufferAttribute(b, 4));
    this.seeps = new THREE.Mesh(geometry, this.seepMaterial);
    this.seeps.name = 'reef-seeps';
    this.seeps.frustumCulled = false;
    this.seeps.renderOrder = 142;
    this.seeps.layers.mask = this.bubbles.layers.mask;
    this.object.add(this.seeps);
  }

  /** Opacity of the two populations, so a caller can fade them with submersion. */
  setOpacity(snow: number, bubbles: number): void {
    this.uSnowOpacity.value = Math.max(0, snow);
    this.uBubbleOpacity.value = Math.max(0, bubbles);
  }

  /** Keeps the particle volume centred on the viewer. No allocation. */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (this.disposed || !this.object.visible) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    this.uCenter.value.copy(cameraPosition);
  }

  /** Rewinds the animation clock, for reproducible captures. */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.remove(this.snow, this.bubbles);
    this.snowGeometry.dispose();
    this.bubbleGeometry.dispose();
    this.snowMaterial.dispose();
    this.bubbleMaterial.dispose();
    if (this.seeps) {
      this.object.remove(this.seeps);
      this.seeps.geometry.dispose();
    }
    this.seepMaterial?.dispose();
  }

  // -------------------------------------------------------------------------

  private snowCount(): number {
    return Math.round(this.count * SNOW_SHARE);
  }

  private bubbleCount(): number {
    return this.count - this.snowCount();
  }

  /**
   * Wraps a world-space point into the box of half-extent `half` centred on the
   * viewer. `mod` in both target languages is the floored variant, so this is
   * correct for negative coordinates.
   */
  private wrap(world: any, half: THREE.Vector3): any {
    const size = vec3(half.x * 2, half.y * 2, half.z * 2);
    const offset = vec3(half.x, half.y, half.z);
    return mod(world.sub(this.uCenter).add(offset), size).sub(offset).add(this.uCenter);
  }

  /**
   * [underwater-look-2] Scale toward zero inside `NEAR_FADE` of the eye.
   *
   * The volume wraps around the camera, so now and then a speck lands a few
   * centimetres from the lens, where a 5 cm disc fills sixty pixels: the
   * "ghost double sphere" in `under-hull` was two marine-snow specks 10-20 cm
   * from the camera (found by hiding the scene's children in halves). Real
   * particulate that close is out of focus and gone; scaling rather than
   * fading because the snow is opaque.
   */
  private nearFade(position: any): any {
    return position.sub(this.uCenter).length().smoothstep(NEAR_FADE * 0.4, NEAR_FADE);
  }

  /**
   * Marine snow is *opaque*, masked to a disc, and writes depth.
   *
   * It was additive and depth-less, and that made it invisible exactly where it
   * matters: `UnderwaterPass` fogs each pixel by the depth buffer, so a speck
   * drawn over open water was fogged as though it were the open water fifty
   * metres behind it — transmittance zero, gone. Particulate showed only against
   * the near reef and never in the blue, which is the one place a diver sees it.
   * Writing its own depth gets each speck its own extinction: crisp near the
   * lens, fading into the haze with distance, the way suspended matter gives
   * the water its sense of volume.
   */
  private buildSnowMaterial(): THREE.SpriteNodeMaterial {
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = false;
    material.depthWrite = true;
    material.depthTest = true;
    material.sizeAttenuation = true;
    material.fog = false;

    // `attribute()` is typed as an opaque node; swizzles need the loose view.
    const seed: any = attribute('seed', 'vec4');

    const base = vec3n(
      seed.x.sub(0.5).mul(SNOW_RADIUS * 2),
      seed.y.sub(0.5).mul(SNOW_HEIGHT * 2),
      seed.z.sub(0.5).mul(SNOW_RADIUS * 2),
    );

    // Detritus sinks slowly and swings; two out-of-phase oscillators at
    // different rates keep the swing from looking like a shared metronome.
    const phase = seed.w.mul(61.0);
    const fall: any = this.uSnowFall.mul(seed.w.mul(0.8).add(0.6));
    const drift = vec3n(
      sin(this.uTime.mul(0.21).add(phase)).mul(0.75),
      this.uTime.mul(fall).negate(),
      cos(this.uTime.mul(0.17).add(phase.mul(1.6))).mul(0.75),
    );

    const position = this.wrap(base.add(drift), new THREE.Vector3(SNOW_RADIUS, SNOW_HEIGHT, SNOW_RADIUS));
    material.positionNode = position;

    const size: any = mix(float(0.014), float(0.05), seed.w.mul(seed.w)).mul(this.nearFade(position));
    material.scaleNode = vec2(size, size);

    const coord = uv();
    const d = coord.sub(vec2(0.5, 0.5)).length();
    // A disc, and a dimmer rim: the temporal resolve softens the edge into the
    // soft-cored speck the additive version drew.
    material.maskNode = d.lessThan(0.45);
    const shade = smoothstepDown(d, 0.05, 0.45).mul(0.5).add(0.5);
    const brightness = mix(float(0.45), float(1.1), seed.w).mul(shade).mul(this.uSnowOpacity);
    material.colorNode = vec4((this.uSnowColor as any).mul(brightness), 1);
    return material;
  }

  private buildBubbleMaterial(): THREE.SpriteNodeMaterial {
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = true;
    material.depthWrite = false;
    material.depthTest = true;
    material.blending = THREE.AdditiveBlending;
    material.sizeAttenuation = true;
    material.fog = false;

    const seed: any = attribute('seed', 'vec4');
    // x,z: world position of this bubble's column. y: phase along the column.
    // w: size / speed variation.
    const column: any = attribute('column', 'vec4');

    const rise = this.uBubbleRise.mul(column.w.mul(0.55).add(0.72));
    // Height within the column, wrapped so a train is continuous.
    const h: any = mod(
      column.y.mul(BUBBLE_HEIGHT * 2).add(this.uTime.mul(rise)),
      BUBBLE_HEIGHT * 2,
    ).sub(BUBBLE_HEIGHT);

    // A rising bubble's wake is unstable, so it spirals. Amplitude grows with
    // height because the wobble compounds on the way up.
    const wobblePhase = seed.w.mul(53.0).add(this.uTime.mul(1.7).mul(column.w.add(0.5)));
    const wobbleAmp = float(COLUMN_JITTER).mul(
      smoothstep(-BUBBLE_HEIGHT, BUBBLE_HEIGHT, h).mul(0.7).add(0.3),
    );

    const world = vec3n(
      column.x.add(seed.x.sub(0.5).mul(COLUMN_JITTER * 2)).add(sin(wobblePhase).mul(wobbleAmp)),
      h,
      column.z.add(seed.z.sub(0.5).mul(COLUMN_JITTER * 2)).add(cos(wobblePhase.mul(0.87)).mul(wobbleAmp)),
    );

    const position = this.wrap(world, new THREE.Vector3(BUBBLE_RADIUS, BUBBLE_HEIGHT, BUBBLE_RADIUS));
    material.positionNode = position;

    const size: any = mix(float(0.025), float(0.11), column.w.mul(column.w)).mul(this.nearFade(position));
    material.scaleNode = vec2(size, size);

    const coord = uv();
    const d = coord.sub(vec2(0.5, 0.5)).length();
    // Bright rim plus a dimmer fill — a bubble is a lens, not a dot.
    const rim = smoothstepDown(d, 0.4, 0.5).mul(smoothstep(0.24, 0.36, d));
    const fill = smoothstepDown(d, 0.05, 0.5).mul(0.35);
    const alpha = max(rim, fill)
      .mul(mix(float(0.5), float(1.0), seed.y))
      .mul(this.uBubbleOpacity);

    material.colorNode = vec4(this.uBubbleColor, alpha);
    return material;
  }

  /** [underwater-look-3] Round 11: the seeps' bubbles; see `SEEP_BUBBLES`. */
  private buildSeepMaterial(): THREE.SpriteNodeMaterial {
    // Opaque and depth-writing, as the snow is and for its reason: an additive
    // bead drawn over open water was fogged by the pass as the water behind it
    // and vanished. A bubble is a silvered bead under water anyway, the rim
    // totally reflecting, so a solid disc with a bright rim reads right.
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = false;
    material.depthWrite = true;
    material.depthTest = true;
    material.sizeAttenuation = true;
    material.fog = false;

    const a: any = attribute('seepA', 'vec4');
    const b: any = attribute('seepB', 'vec4');
    const rise = b.w.mul(0.4).add(0.8).mul(SEEP_RISE);
    // 0..1 up the stream, wrapped so the string is continuous.
    const t: any = mod(b.x.add(this.uTime.mul(rise).div(SEEP_HEIGHT)).add(a.w), 1);
    const h = t.mul(SEEP_HEIGHT);
    // The wake's spiral, wider the higher it has come.
    const phase = b.z.mul(41).add(this.uTime.mul(b.w.mul(1.6).add(2.2)));
    const amp = t.mul(0.16).add(0.015);
    const position = vec3n(
      a.x.add(sin(phase).mul(amp)),
      a.y.add(h),
      a.z.add(cos(phase.mul(0.83)).mul(amp)),
    );
    material.positionNode = position;
    const size: any = mix(float(SEEP_SIZE[0]), float(SEEP_SIZE[1]), b.y.mul(b.y))
      .mul(t.mul(SEEP_GROWTH).add(1))
      .mul(smoothstepDown(t, 0.8, 1))
      .mul(smoothstep(0, 0.03, t))
      .mul(this.nearFade(position));
    material.scaleNode = vec2(size, size);

    const d = uv().sub(vec2(0.5, 0.5)).length();
    material.maskNode = d.lessThan(0.48);
    const rim = smoothstep(0.26, 0.44, d);
    // A highlight up toward the light, as a lit sphere of air shows one.
    const spec = smoothstepDown(uv().sub(vec2(0.4, 0.64)).length(), 0.03, 0.11);
    const bright: any = mix(float(0.45), float(1.5), rim).add(spec.mul(1.6)).mul(this.uBubbleOpacity);
    material.colorNode = vec4((this.uBubbleColor as any).mul(bright), 1);
    return material;
  }
}

/**
 * One camera-facing quad, instanced N times. `SpriteNodeMaterial` reads the
 * corner offsets straight from `position`, so the quad is authored in the
 * -0.5..0.5 range it expects.
 *
 * A plain `InstancedBufferGeometry` rather than `InstancedMesh`: the latter
 * would make the node material inject an instance matrix the sprite path does
 * not use.
 */
function buildQuad(count: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
  );
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);

  geometry.instanceCount = count;
  // The volume follows the camera, so a fixed bound is meaningless; culling off.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

  return geometry;
}

/**
 * Seeded, not `Math.random()`.
 *
 * The scatter has to survive a reload, otherwise an underwater baseline capture
 * is comparing two different particle fields and every diff is noise. It also
 * has to survive a `setCount` rebuild at a *different* count: drawing from one
 * generator per call means instance `i` gets the same seed whether the buffer
 * holds 400 particles or 6000, so a tier change re-scales the population instead
 * of reshuffling it.
 */
function buildSnowGeometry(count: number): THREE.InstancedBufferGeometry {
  const geometry = buildQuad(count);
  const seeds = fillRandom(new Float32Array(count * 4), SEEDS.underwaterSnow);
  geometry.setAttribute('seed', new THREE.InstancedBufferAttribute(seeds, 4));
  return geometry;
}

function buildBubbleGeometry(count: number): THREE.InstancedBufferGeometry {
  const geometry = buildQuad(count);
  const random = mulberry32(SEEDS.bubbles);

  geometry.setAttribute(
    'seed',
    new THREE.InstancedBufferAttribute(fillRandom(new Float32Array(count * 4), SEEDS.bubbles), 4),
  );

  // Column axes are chosen once on the CPU: picking them in the shader from a
  // hash would spread the bubbles evenly, and the reference shows them arriving
  // in tight trains.
  const columns: Array<{ x: number; z: number }> = [];
  for (let c = 0; c < BUBBLE_COLUMNS; c++) {
    columns.push({
      x: (random() - 0.5) * BUBBLE_RADIUS * 2,
      z: (random() - 0.5) * BUBBLE_RADIUS * 2,
    });
  }

  const data = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const c = columns[i % BUBBLE_COLUMNS];
    data[i * 4 + 0] = c.x;
    data[i * 4 + 1] = random();
    data[i * 4 + 2] = c.z;
    data[i * 4 + 3] = random();
  }
  geometry.setAttribute('column', new THREE.InstancedBufferAttribute(data, 4));

  return geometry;
}
