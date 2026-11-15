import * as THREE from 'three/webgpu';
import {
  attribute,
  cos,
  float,
  mix,
  mod,
  pow,
  sin,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import { SEEDS, fillRandom } from '../core/random';

/**
 * Camera-following precipitation.
 *
 * Both effects are instanced camera-facing quads driven by `SpriteNodeMaterial`,
 * which gives real world-space particle size on WebGPU (point primitives are
 * clamped to one pixel there) and on WebGL2 alike.
 *
 * Particle motion is computed entirely in the vertex stage from a per-instance
 * seed and a uniform clock, with a modulo wrap — the CPU never touches particle
 * positions. `update` only moves the container object onto the viewer, which is
 * a single transform per frame.
 */

export type WeatherKind = 'clear' | 'rain' | 'snow';

/**
 * Instances allocated once; the visible count is scaled by intensity.
 *
 * Rain was 9000, and with the streak length and brightness it read as a
 * curtain in front of the shot (A/B critic, waves-5: "too long, too bright and
 * too dense"). The storm should be read through the rain, not the rain read
 * instead of the storm.
 */
const RAIN_COUNT = 5000;
const SNOW_COUNT = 6000;

/** Half-extent of the effect volume around the viewer, metres. */
const RAIN_RADIUS = 34;
const RAIN_HEIGHT = 46;
const SNOW_RADIUS = 30;
const SNOW_HEIGHT = 36;

/** Clock wrap, seconds. Long enough to be invisible, short enough for float32. */
const CLOCK_WRAP = 3600;

export class Weather {
  readonly object: THREE.Object3D;

  private readonly rainGeometry: THREE.InstancedBufferGeometry;
  private readonly snowGeometry: THREE.InstancedBufferGeometry;
  private readonly rainMaterial: THREE.SpriteNodeMaterial;
  private readonly snowMaterial: THREE.SpriteNodeMaterial;
  private readonly rain: THREE.Mesh;
  private readonly snow: THREE.Mesh;

  private kind: WeatherKind = 'clear';
  private intensity = 0;
  private clock = 0;

  private readonly uTime = uniform(0);
  private readonly uRainOpacity = uniform(0);
  private readonly uSnowOpacity = uniform(0);
  private readonly uRainSpeed = uniform(24);
  private readonly uSnowSpeed = uniform(1.1);
  /** Horizontal metres of drift per metre of fall — the wind shear. */
  private readonly uShear = uniform(new THREE.Vector2(0.16, 0.05));
  /** Streak length multiplier, rising with rain rate. */
  private readonly uStreakLength = uniform(1);
  /** Streak tilt in radians, derived from the shear so the two cannot disagree. */
  private readonly uStreakTilt = uniform(-Math.atan(0.16));

  constructor() {
    this.object = new THREE.Object3D();
    this.object.name = 'weather';
    this.object.matrixAutoUpdate = true;
    this.object.frustumCulled = false;
    this.object.visible = false;

    this.rainGeometry = buildParticleGeometry(RAIN_COUNT, SEEDS.rain);
    this.snowGeometry = buildParticleGeometry(SNOW_COUNT, SEEDS.snow);

    this.rainMaterial = this.buildRainMaterial();
    this.snowMaterial = this.buildSnowMaterial();

    this.rain = new THREE.Mesh(this.rainGeometry, this.rainMaterial);
    this.rain.name = 'rain';
    this.rain.frustumCulled = false;
    this.rain.renderOrder = 120;
    this.rain.visible = false;

    this.snow = new THREE.Mesh(this.snowGeometry, this.snowMaterial);
    this.snow.name = 'snow';
    this.snow.frustumCulled = false;
    this.snow.renderOrder = 120;
    this.snow.visible = false;

    this.object.add(this.rain, this.snow);
  }

  setKind(kind: WeatherKind): void {
    if (this.kind === kind) return;
    this.kind = kind;
    this.applyVisibility();
  }

  getKind(): WeatherKind {
    return this.kind;
  }

  getIntensity(): number {
    return this.intensity;
  }

  setIntensity(v: number): void {
    const clamped = Math.min(1, Math.max(0, v));
    if (this.intensity === clamped) return;
    this.intensity = clamped;
    this.applyVisibility();
  }

  /** Keeps the effect volume centred on the viewer. */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (!this.object.visible) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    this.object.position.copy(cameraPosition);
  }

  /**
   * Rewinds the animation clock.
   *
   * Particle positions are a pure function of the per-instance seed and this
   * clock, so setting it reproduces an exact frame of the curtain — which is what
   * makes a storm baseline comparable run to run.
   */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  dispose(): void {
    this.object.remove(this.rain, this.snow);
    this.rainGeometry.dispose();
    this.snowGeometry.dispose();
    this.rainMaterial.dispose();
    this.snowMaterial.dispose();
  }

  // -------------------------------------------------------------------------

  private applyVisibility(): void {
    const active = this.kind !== 'clear' && this.intensity > 0.001;
    this.object.visible = active;
    this.rain.visible = active && this.kind === 'rain';
    this.snow.visible = active && this.kind === 'snow';

    // Scaling the instance count rather than the alpha means light rain is
    // genuinely cheaper, not just fainter.
    const ramp = Math.pow(this.intensity, 0.75);
    this.rainGeometry.instanceCount = Math.round(RAIN_COUNT * ramp);
    this.snowGeometry.instanceCount = Math.round(SNOW_COUNT * ramp);

    // Additive and unlit, so this is the streak's whole brightness: at the old
    // 0.6 a storm streak added ~40% of the storm preset's white point. A drop is
    // a small, mostly transparent lens; it should glint, not glow.
    this.uRainOpacity.value = 0.08 + 0.17 * this.intensity;
    this.uSnowOpacity.value = 0.35 + 0.5 * this.intensity;
    // Terminal velocity, measured: about 6.5 m/s for a 2 mm drop and at most
    // ~9 m/s for the largest (Gunn & Kinzer, J. Meteorology 6, 1949). It was
    // 18–30 m/s, three times anything that falls, which is half of why the
    // streaks were so long; the gusts of a storm account for the margin here.
    this.uRainSpeed.value = 7 + 3 * this.intensity;
    // Harder rain falls faster and therefore streaks longer, on top of the
    // per-drop variation the material already applies.
    this.uStreakLength.value = 0.9 + 0.5 * this.intensity;
  }

  /**
   * Sets the wind shear the precipitation falls through.
   *
   * The streak tilt is derived here rather than authored, because a streak that
   * does not lie along its own motion is the single most obvious way to make
   * rain look pasted onto the frame.
   */
  setShear(x: number, z: number): void {
    (this.uShear.value as THREE.Vector2).set(x, z);
    this.uStreakTilt.value = -Math.atan(x);
  }

  private buildRainMaterial(): THREE.SpriteNodeMaterial {
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = true;
    material.depthWrite = false;
    material.blending = THREE.AdditiveBlending;
    material.sizeAttenuation = true;
    material.fog = false;

    // `attribute()` is typed as an opaque node; swizzles need the loose view.
    const seed: any = attribute('seed', 'vec4');

    // Per-drop fall speed varies so the curtain never looks like a rigid sheet.
    const speed = this.uRainSpeed.mul(seed.w.mul(0.45).add(0.8));
    const y = mod(seed.z.mul(RAIN_HEIGHT).sub(this.uTime.mul(speed)), RAIN_HEIGHT).sub(
      RAIN_HEIGHT * 0.35,
    );

    const shear = this.uShear;
    const x = seed.x.sub(0.5).mul(RAIN_RADIUS * 2).add(y.mul(shear.x));
    const z = seed.y.sub(0.5).mul(RAIN_RADIUS * 2).add(y.mul(shear.y));

    material.positionNode = vec3(x, y, z);

    // Streak length follows fall speed, because that is what a streak *is*: the
    // distance a drop covers while the shutter is open. A fixed length makes
    // heavy rain read as more drops of the same size rather than as harder rain,
    // which is most of why the old curtain looked like static.
    const lengthScale: any = speed.div(this.uRainSpeed).mul(this.uStreakLength);
    const width: any = float(0.016).add(seed.z.mul(0.012));
    // Metres. A streak is the fall during the exposure, and at a real terminal
    // velocity a 1/60 s shutter gives ~15 cm; this is about twice that, the
    // eye's own integration time, so a streak still reads as a streak at a few
    // metres. It was 0.7–1.3 m before the speed scale — up to 2.8 m in a storm.
    const length: any = float(0.24).add(seed.w.mul(0.2)).mul(lengthScale);
    material.scaleNode = vec2(width, length);

    // Tilt from the actual shear, not a baked constant, so the streak lies along
    // its own motion when the wind changes.
    material.rotationNode = this.uStreakTilt;

    const coord = uv();
    // Across the streak: a tight core with a soft shoulder. A drop is a lens, so
    // its edge is dim and its centre carries almost all the light.
    const across = float(1).sub(coord.x.sub(0.5).abs().mul(2.0));
    const core = pow(across, 4.0);
    // Along it: bright at the leading end, tapering behind — a real streak is an
    // exposure trail, brightest where the drop is now.
    const along = pow(coord.y, 0.55).mul(float(1).sub(coord.y).mul(2.4).min(1.0));

    // Nearer drops are brighter and better resolved. Depth here is the seed's
    // own scatter, which is cheap and reads correctly because the volume is
    // centred on the viewer.
    const nearness = float(0.55).add(seed.x.sub(0.5).abs().oneMinus().mul(0.45));

    const alpha = core.mul(along).mul(nearness).mul(this.uRainOpacity);
    material.colorNode = vec4(vec3(0.78, 0.85, 0.96), alpha);
    return material;
  }

  private buildSnowMaterial(): THREE.SpriteNodeMaterial {
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = true;
    material.depthWrite = false;
    material.blending = THREE.NormalBlending;
    material.sizeAttenuation = true;
    material.fog = false;

    // `attribute()` is typed as an opaque node; swizzles need the loose view.
    const seed: any = attribute('seed', 'vec4');

    const speed = this.uSnowSpeed.mul(seed.w.mul(0.7).add(0.65));
    const y = mod(seed.z.mul(SNOW_HEIGHT).sub(this.uTime.mul(speed)), SNOW_HEIGHT).sub(
      SNOW_HEIGHT * 0.4,
    );

    // Lateral wobble: two out-of-phase oscillators give a convincing tumble.
    const phase = seed.w.mul(43.0);
    const wobbleX = sin(this.uTime.mul(0.55).add(phase)).mul(0.55);
    const wobbleZ = cos(this.uTime.mul(0.41).add(phase.mul(1.7))).mul(0.55);

    const x = seed.x.sub(0.5).mul(SNOW_RADIUS * 2).add(wobbleX).add(y.mul(0.06));
    const z = seed.y.sub(0.5).mul(SNOW_RADIUS * 2).add(wobbleZ);

    material.positionNode = vec3(x, y, z);
    const flake = mix(float(0.035), float(0.085), seed.w);
    material.scaleNode = vec2(flake, flake);

    const coord = uv();
    const d = coord.sub(vec2(0.5, 0.5)).length();
    const alpha = smoothstepDown(d, 0.06, 0.5).mul(this.uSnowOpacity);

    material.colorNode = vec4(vec3(0.95, 0.97, 1.0), alpha);
    return material;
  }
}

/**
 * One camera-facing quad, instanced N times, plus a vec4 seed per instance.
 * `SpriteNodeMaterial` reads the corner offsets straight from `position`, so the
 * quad is authored in the -0.5..0.5 range it expects.
 *
 * A plain `InstancedBufferGeometry` is used rather than `InstancedMesh` on
 * purpose: `InstancedMesh` would make the node material inject an instance
 * matrix that the sprite path does not use.
 *
 * `seed` is drawn from an explicit seed, not `Math.random()`: the scatter of the
 * rain curtain has to be identical on every load or no storm screenshot can ever
 * be compared against a baseline.
 */
function buildParticleGeometry(count: number, seed: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0],
      3,
    ),
  );
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);

  const seeds = fillRandom(new Float32Array(count * 4), seed);
  geometry.setAttribute('seed', new THREE.InstancedBufferAttribute(seeds, 4));

  geometry.instanceCount = 0;
  // The volume follows the camera, so a fixed bound is meaningless; culling off.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

  return geometry;
}
