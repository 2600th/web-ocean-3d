import * as THREE from 'three/webgpu';
import {
  abs,
  atan,
  attribute,
  cameraViewMatrix,
  cos,
  float,
  floor,
  fract,
  mix,
  mod,
  pow,
  select,
  sin,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDown } from '../core/tslMath';
import { SEEDS, fillRandom } from '../core/random';

/**
 * Precipitation around the viewer.
 *
 * Both effects are instanced camera-facing quads driven by `SpriteNodeMaterial`,
 * which gives real world-space particle size on WebGPU, where point primitives
 * are clamped to one pixel.
 *
 * Particle motion is computed entirely in the vertex stage from a per-instance
 * seed and a uniform clock, with a modulo wrap — the CPU never touches particle
 * positions.
 *
 * **Rain is anchored to the world, not the camera.** It used to live in a
 * container moved onto the camera every frame, so the whole curtain was carried
 * along with the ship and sailing at eight knots never made it stream past.
 * Now each drop has a world position (seed plus fall plus wind drift) wrapped
 * into a box around the eye, as the underwater particles do
 * (`underwater/Particles.ts`): the drops stay where they are and the ship moves
 * through them. Each streak is laid along the drop's motion *relative to the
 * camera*, projected to the screen, and is as long as that motion during a
 * short exposure: a following wind stands the streaks up, a head wind or a
 * fast ship lays them back, and rain seen along the wind is a dot, not a
 * slanted dash. The old tilt came from the world-X shear as a fixed screen
 * rotation, so it leaned the same way whichever way the camera faced.
 *
 * Drops inside the hull and under the courses are dropped (sheltered), and
 * the deck gets its own small set of splash crowns. The colour is the scene's
 * light on the water (`setLighting`), plus the lightning: rain of one constant
 * colour was lifted by the auto-exposure at night into a bright grey haze.
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
/** Splash crowns on the deck, at full rain and full detail. */
const DECK_SPLASH_COUNT = 260;

/** Half-extent of the effect volume around the viewer, metres. */
const RAIN_RADIUS = 34;
const RAIN_HEIGHT = 46;
/** The rain box is centred this far above the eye: more falls above than below. */
const RAIN_LIFT = RAIN_HEIGHT * 0.15;
const SNOW_RADIUS = 30;
const SNOW_HEIGHT = 36;

/**
 * Seconds a streak is exposed for. At 7–10 m/s of fall this is the 0.24–0.44 m
 * the old fixed streaks had, about twice a 1/60 s shutter: the eye's own
 * integration time, so a streak still reads as one at a few metres.
 */
const STREAK_EXPOSURE = 0.034;
/** Longest streak, metres, so a camera whipped round does not draw spears. */
const STREAK_MAX = 1.1;

/** Clock wrap, seconds. Long enough to be invisible, short enough for float32. */
const CLOCK_WRAP = 3600;
/** Camera moves faster than this (m/s) are cuts, not motion. */
const CUT_SPEED = 60;

/**
 * The hero hull's cover, in the ship's frame (+X bow, +Y up from the
 * waterline, +Z starboard), metres. Measured off `dutch_ship_medium.glb` as
 * loaded (hull 27 m x 6.4 m): the waist deck stands about 2.4 m above the
 * waterline, the quarterdeck aft of -7 m and the forecastle forward of 10.5 m
 * about 4.1 m, and the two courses span -2.5 m to 9.3 m with their feet at
 * 5.2 m. Rain under the courses and inside the hull is sheltered.
 */
const HULL_HALF_LENGTH = 13.5;
const HULL_HALF_BEAM = 3.2;
const WAIST_DECK = 2.4;
const QUARTERDECK_FROM = -7;
const FORECASTLE_FROM = 10.5;
const RAISED_DECK = 4.1;
const CANVAS_AFT = -2.5;
const CANVAS_FORE = 9.3;
const CANVAS_FOOT = 5.2;

export class Weather {
  readonly object: THREE.Object3D;

  private readonly rainGeometry: THREE.InstancedBufferGeometry;
  private readonly snowGeometry: THREE.InstancedBufferGeometry;
  private readonly splashGeometry: THREE.InstancedBufferGeometry;
  private readonly rainMaterial: THREE.SpriteNodeMaterial;
  private readonly snowMaterial: THREE.SpriteNodeMaterial;
  private readonly splashMaterial: THREE.SpriteNodeMaterial;
  private readonly rain: THREE.Mesh;
  private readonly snow: THREE.Mesh;
  private readonly splashes: THREE.Mesh;

  private kind: WeatherKind = 'clear';
  private intensity = 0;
  private clock = 0;
  /** 0..1 from the tier: the share of drops drawn, and whether the deck splashes. */
  private detail = 1;
  private ship: THREE.Object3D | null = null;
  private readonly lastEye = new THREE.Vector3();
  private hasEye = false;

  private readonly uTime = uniform(0);
  private readonly uRainOpacity = uniform(0);
  private readonly uSnowOpacity = uniform(0);
  private readonly uRainSpeed = uniform(24);
  private readonly uSnowSpeed = uniform(1.1);
  /** Horizontal metres of drift per metre of fall — the wind shear. */
  private readonly uShear = uniform(new THREE.Vector2(0.16, 0.05));
  /** Streak length multiplier, rising with rain rate. */
  private readonly uStreakLength = uniform(1);
  /** Centre of the rain's wrap volume: the eye. */
  private readonly uCenter = uniform(new THREE.Vector3());
  /** The eye's velocity, m/s, smoothed; the streaks are drawn relative to it. */
  private readonly uEyeVelocity = uniform(new THREE.Vector3());
  /** Light on the rain: the scene's ambient on a drop, plus the lightning. */
  private readonly uRainColor = uniform(new THREE.Color(0.78, 0.85, 0.96));
  /** 1 while a hull is tracked. */
  private readonly uShipOn = uniform(0);
  /** World to ship frame and back, taken at render time so cover never lags the hull. */
  private readonly uShipInverse = uniform(new THREE.Matrix4()).onRenderUpdate((_, node) => {
    const m = node.value as THREE.Matrix4;
    if (this.ship) m.copy(this.ship.matrixWorld).invert();
    return m;
  });
  private readonly uShipMatrix = uniform(new THREE.Matrix4()).onRenderUpdate((_, node) => {
    const m = node.value as THREE.Matrix4;
    if (this.ship) m.copy(this.ship.matrixWorld);
    return m;
  });
  private readonly uSplashOpacity = uniform(0);

  constructor() {
    this.object = new THREE.Object3D();
    this.object.name = 'weather';
    this.object.matrixAutoUpdate = true;
    this.object.frustumCulled = false;
    this.object.visible = false;

    this.rainGeometry = buildParticleGeometry(RAIN_COUNT, SEEDS.rain);
    this.snowGeometry = buildParticleGeometry(SNOW_COUNT, SEEDS.snow);
    this.splashGeometry = buildParticleGeometry(DECK_SPLASH_COUNT, SEEDS.rain ^ 0x5a1a5);

    this.rainMaterial = this.buildRainMaterial();
    this.snowMaterial = this.buildSnowMaterial();
    this.splashMaterial = this.buildSplashMaterial();

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

    this.splashes = new THREE.Mesh(this.splashGeometry, this.splashMaterial);
    this.splashes.name = 'rain-deck-splashes';
    this.splashes.frustumCulled = false;
    this.splashes.renderOrder = 119;
    this.splashes.visible = false;

    this.object.add(this.rain, this.snow, this.splashes);
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

  /**
   * The tier's share, 0..1 (`QualitySettings.weatherDetail`). Moves instance
   * counts only, so a tier change compiles nothing. Below 0.5 the deck has no
   * splashes. Rain at full detail is 5,000 quads and 260 splash quads, all
   * additive and small on screen: a few tenths of a millisecond at most.
   */
  setDetail(detail: number): void {
    this.detail = Math.min(1, Math.max(0, detail));
    this.applyVisibility();
  }

  /** The hull that shelters the rain and takes the deck splashes, or null. */
  setShip(ship: THREE.Object3D | null): void {
    this.ship = ship;
    this.uShipOn.value = ship ? 1 : 0;
    this.applyVisibility();
  }

  /**
   * Light on the rain. `ambient` is the light a grey card gets from the sky and
   * the key, about 1 at a clear noon; `flash` is the lightning's light here,
   * 0..~1.4. A drop is a small lens that shows the sky, so it takes the
   * scene's light rather than glowing on its own, and goes dark with the night.
   */
  setLighting(ambient: THREE.Color, flash: number): void {
    const c = this.uRainColor.value as THREE.Color;
    c.setRGB(0.78, 0.85, 0.96).multiply(ambient).multiplyScalar(RAIN_LIGHT_GAIN);
    const f = Math.max(0, flash) * RAIN_FLASH_GAIN;
    c.r += 0.78 * f;
    c.g += 0.85 * f;
    c.b += 1.0 * f;
  }

  /** Keeps the effect volume centred on the viewer and tracks its motion. */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (!this.object.visible) {
      this.hasEye = false;
      return;
    }
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    (this.uCenter.value as THREE.Vector3).copy(cameraPosition);
    // Snow keeps the old scheme: it falls slowly enough that being carried
    // along costs nothing a viewer can see.
    this.snow.position.copy(cameraPosition);

    const v = this.uEyeVelocity.value as THREE.Vector3;
    if (this.hasEye && dt > 1e-4) {
      const step = _scratch.subVectors(cameraPosition, this.lastEye).divideScalar(dt);
      if (step.length() > CUT_SPEED) v.set(0, 0, 0);
      else v.lerp(step, 1 - Math.exp(-dt / 0.08));
    } else if (!this.hasEye) {
      v.set(0, 0, 0);
    }
    this.lastEye.copy(cameraPosition);
    this.hasEye = true;
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
    // The eye's velocity is measured again from the next frame, so a capture
    // does not depend on where the camera was before the reset.
    this.hasEye = false;
    (this.uEyeVelocity.value as THREE.Vector3).set(0, 0, 0);
  }

  dispose(): void {
    this.object.remove(this.rain, this.snow, this.splashes);
    this.rainGeometry.dispose();
    this.snowGeometry.dispose();
    this.splashGeometry.dispose();
    this.rainMaterial.dispose();
    this.snowMaterial.dispose();
    this.splashMaterial.dispose();
  }

  // -------------------------------------------------------------------------

  private applyVisibility(): void {
    const active = this.kind !== 'clear' && this.intensity > 0.001;
    this.object.visible = active;
    this.rain.visible = active && this.kind === 'rain';
    this.snow.visible = active && this.kind === 'snow';
    const splashing = this.rain.visible && this.ship !== null && this.detail >= 0.5;
    this.splashes.visible = splashing;

    // Scaling the instance count rather than the alpha means light rain is
    // genuinely cheaper, not just fainter.
    const ramp = Math.pow(this.intensity, 0.75);
    this.rainGeometry.instanceCount = Math.round(RAIN_COUNT * ramp * this.detail);
    this.snowGeometry.instanceCount = Math.round(SNOW_COUNT * ramp * this.detail);
    this.splashGeometry.instanceCount = splashing ? Math.round(DECK_SPLASH_COUNT * ramp * this.detail) : 0;

    // Additive, so this is the streak's whole brightness: at the old 0.6 a
    // storm streak added ~40% of the storm preset's white point. A drop is a
    // small, mostly transparent lens; it should glint, not glow. A tier that
    // draws fewer drops draws each a little brighter, so the rain reads as the
    // same rain rate.
    this.uRainOpacity.value = (0.08 + 0.17 * this.intensity) / Math.sqrt(Math.max(0.25, this.detail));
    this.uSnowOpacity.value = 0.35 + 0.5 * this.intensity;
    this.uSplashOpacity.value = 0.35 + 0.4 * this.intensity;
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
   * Sets the wind shear the precipitation falls through: horizontal metres per
   * metre of fall, toward where the wind blows. The streak's slant is not set
   * here; it follows from the drop's motion against the camera's, in the
   * shader.
   */
  setShear(x: number, z: number): void {
    (this.uShear.value as THREE.Vector2).set(x, z);
  }

  /**
   * How high the hull's cover is above the ship-frame point (x, z), and
   * whether the point is under it at all: the deck, raised at the castles, and the
   * courses over the waist. Returns the height, or -1e4 outside the hull.
   */
  private coverHeight(x: any, zSigned: any): any {
    const z: any = abs(zSigned);
    const deck = select(
      x.lessThan(QUARTERDECK_FROM).or(x.greaterThan(FORECASTLE_FROM)),
      float(RAISED_DECK),
      float(WAIST_DECK),
    );
    const underCanvas = x.greaterThan(CANVAS_AFT).and(x.lessThan(CANVAS_FORE)).and(z.lessThan(HULL_HALF_BEAM - 0.2));
    const cover = select(underCanvas, float(CANVAS_FOOT), deck);
    // The hull narrows to the bow and stern; a lens-shaped plan is near enough.
    const halfBeam = float(HULL_HALF_BEAM).mul(
      float(1).sub(x.div(HULL_HALF_LENGTH).mul(x.div(HULL_HALF_LENGTH)).mul(0.55)),
    );
    const inside = abs(x).lessThan(HULL_HALF_LENGTH).and(z.lessThan(halfBeam)).and(this.uShipOn.greaterThan(0.5));
    return select(inside, cover, float(-1e4));
  }

  /** The waist, quarterdeck or forecastle height at ship-frame x. */
  private deckHeight(x: any): any {
    return select(
      x.lessThan(QUARTERDECK_FROM).or(x.greaterThan(FORECASTLE_FROM)),
      float(RAISED_DECK),
      float(WAIST_DECK),
    );
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
    const speed: any = this.uRainSpeed.mul(seed.w.mul(0.45).add(0.8));
    const shear: any = this.uShear;
    const velocity: any = vec3(shear.x.mul(speed), speed.negate(), shear.y.mul(speed));

    // World position: the seed's place, carried by fall and drift, wrapped into
    // the box round the eye. The drop does not move with the camera.
    const base = vec3(
      seed.x.sub(0.5).mul(RAIN_RADIUS * 2),
      seed.z.sub(0.5).mul(RAIN_HEIGHT),
      seed.y.sub(0.5).mul(RAIN_RADIUS * 2),
    );
    const centre: any = (this.uCenter as any).add(vec3(0, RAIN_LIFT, 0));
    const size = vec3(RAIN_RADIUS * 2, RAIN_HEIGHT, RAIN_RADIUS * 2);
    const half = vec3(RAIN_RADIUS, RAIN_HEIGHT / 2, RAIN_RADIUS);
    const world: any = base.add(velocity.mul(this.uTime));
    const position: any = mod(world.sub(centre).add(half), size).sub(half).add(centre).toVar('rainPosition');
    material.positionNode = position;

    // Sheltered: inside the hull, or under the courses.
    const local: any = (this.uShipInverse as any).mul(vec4(position, 1)).xyz;
    const open = local.y.greaterThan(this.coverHeight(local.x, local.z));

    // The streak on screen. In view space a point P moving at V moves across
    // the screen along (P.x V.z - V.x P.z, P.y V.z - V.y P.z); its speed in the
    // plane of the sprite is that over -P.z. Relative to the eye, so a ship
    // making way drives the rain past and lays the streaks back.
    const view: any = cameraViewMatrix;
    const p: any = view.mul(vec4(position, 1)).xyz;
    const v: any = view.mul(vec4(velocity.sub(this.uEyeVelocity), 0)).xyz;
    const dx: any = p.x.mul(v.z).sub(v.x.mul(p.z));
    const dy: any = p.y.mul(v.z).sub(v.y.mul(p.z));
    const depth: any = p.z.negate().max(0.05);
    const planeSpeed: any = vec2(dx, dy).length().div(depth);
    // `rotate` maps the quad's +Y to (-sin, cos); lay it along (dx, dy), with
    // the bright leading end (uv.y = 1) forward.
    material.rotationNode = atan(dx.negate(), dy);

    const width: any = float(0.016).add(seed.z.mul(0.012));
    // Scaled away within a couple of metres of the eye: a drop on the lens is
    // the lens rain pass's job, and a streak a metre from the eye is a spear
    // across the frame. The length is also held to a tenth of the range, a few
    // degrees on screen, for the same reason.
    const range: any = position.sub(this.uCenter).length();
    const length: any = planeSpeed
      .mul(STREAK_EXPOSURE)
      .mul(this.uStreakLength)
      .mul(seed.w.mul(0.3).add(0.85))
      .min(range.mul(0.1))
      .clamp(width.mul(1.5), STREAK_MAX);
    const near = range.smoothstep(0.8, 2.5);
    const keep = select(open, near, float(0));
    material.scaleNode = vec2(width.mul(keep), length.mul(keep));

    const coord = uv();
    // Across the streak: a tight core with a soft shoulder. A drop is a lens, so
    // its edge is dim and its centre carries almost all the light.
    const across = float(1).sub(coord.x.sub(0.5).abs().mul(2.0));
    const core = pow(across, 4.0);
    // Along it: bright at the leading end, tapering behind — a real streak is an
    // exposure trail, brightest where the drop is now.
    const along = pow(coord.y, 0.55).mul(float(1).sub(coord.y).mul(2.4).min(1.0));

    // Nearer drops are brighter and better resolved.
    const nearness = float(1).sub(range.smoothstep(3, RAIN_RADIUS).mul(0.45));

    const alpha = core.mul(along).mul(nearness).mul(this.uRainOpacity);
    material.colorNode = vec4(this.uRainColor, alpha);
    return material;
  }

  /**
   * Crowns on the deck. Not tied to particular drops — a drop that reaches the
   * deck is already out of the rain's box, and following it costs state — but
   * the same rain rate falling on the same planks: each splash slot fires on
   * its own period at a fresh spot on the open deck, and the courses keep the
   * waist under them dry.
   */
  private buildSplashMaterial(): THREE.SpriteNodeMaterial {
    const material = new THREE.SpriteNodeMaterial();
    material.transparent = true;
    material.depthWrite = false;
    material.blending = THREE.AdditiveBlending;
    material.sizeAttenuation = true;
    material.fog = false;

    const seed: any = attribute('seed', 'vec4');
    const period: any = seed.w.mul(0.7).add(0.45);
    const cycle: any = this.uTime.div(period).add(seed.z.mul(17));
    const life: any = fract(cycle);
    const n: any = floor(cycle);
    const h1: any = fract(sin(n.mul(12.9898).add(seed.x.mul(78.233))).mul(43758.5453));
    const h2: any = fract(sin(n.mul(39.3468).add(seed.y.mul(11.135))).mul(24634.6345));

    const x: any = h1.sub(0.5).mul(HULL_HALF_LENGTH * 1.8);
    const taper: any = float(1).sub(x.div(HULL_HALF_LENGTH).mul(x.div(HULL_HALF_LENGTH)).mul(0.55));
    const z: any = h2.sub(0.5).mul(HULL_HALF_BEAM * 1.7).mul(taper);
    const y: any = this.deckHeight(x).add(0.04);
    const local: any = vec3(x, y, z);
    const world: any = (this.uShipMatrix as any).mul(vec4(local, 1)).xyz;
    material.positionNode = world;

    // Dry under the courses, and on the planks only for the first moments.
    const covered = x.greaterThan(CANVAS_AFT).and(x.lessThan(CANVAS_FORE));
    const SPLASH_LIFE = 0.16;
    const s: any = life.div(SPLASH_LIFE).clamp(0, 1);
    const alive: any = select(covered.or(life.greaterThan(SPLASH_LIFE)), float(0), float(1));
    const crown: any = float(0.05).add(s.mul(0.16)).mul(seed.x.mul(0.5).add(0.75)).mul(alive);
    material.scaleNode = vec2(crown, crown.mul(0.55));

    const coord = uv();
    const d = coord.sub(vec2(0.5, 0.5)).length().mul(2);
    // A ring with a faint centre: the crown and the sheet it throws.
    const ring = d.smoothstep(0.35, 0.75).mul(float(1).sub(d.smoothstep(0.8, 1)));
    const body = ring.add(float(1).sub(d.smoothstep(0, 0.6)).mul(0.3));
    const fade = float(1).sub(s).mul(float(1).sub(s));
    material.colorNode = vec4(this.uRainColor, body.mul(fade).mul(this.uSplashOpacity));
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
 * Scene light to rain colour. A grey card at a clear noon is about 1; the old
 * constant colour was what this gives at about 1.6, the brightness the storm
 * shot was tuned at (its ambient is near 0.6).
 */
const RAIN_LIGHT_GAIN = 1.6;
/** The lightning's light on the rain, per unit of flash. */
const RAIN_FLASH_GAIN = 2.2;
const _scratch = new THREE.Vector3();

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
