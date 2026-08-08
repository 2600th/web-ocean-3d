import * as THREE from 'three/webgpu';
import {
  Fn,
  acos,
  cameraPosition,
  clamp,
  cos,
  dot,
  exp,
  float,
  floor,
  fract,
  max,
  mix,
  mx_noise_float,
  normalize,
  positionGeometry,
  pow,
  saturate,
  sin,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';

/**
 * Analytic sky dome.
 *
 * The daylight term is a port of the Preetham model as implemented by the
 * MIT-licensed three.js example `three/addons/objects/Sky.js` (A. Preetham,
 * P. Shirley, B. Smits — "A Practical Analytic Model for Daylight"). The stock
 * example is a raw `ShaderMaterial`, which cannot participate in this project's
 * node pipeline, so the shading is re-authored here as a TSL node graph on a
 * `MeshBasicNodeMaterial`. The night hemisphere (stars, moon disc, halo) and the
 * ground term are original.
 *
 * Everything the caller can tune is a `uniform()`, so `setParams` never rebuilds
 * the node graph — it only writes uniform values.
 */

// ---------------------------------------------------------------------------
// Preetham constants (see Sky.js for their derivation).
// ---------------------------------------------------------------------------

/** Rayleigh scattering coefficient at sea level for the 680/550/450 nm primaries. */
const TOTAL_RAYLEIGH = new THREE.Vector3(
  5.804542996261093e-6,
  1.3562911419845635e-5,
  3.0265902468824876e-5,
);
/** pi * pow( ( 2 * pi ) / lambda, v - 2 ) * K, precomputed per primary. */
const MIE_CONST = new THREE.Vector3(
  1.8399918514433978e14,
  2.7798023919660528e14,
  4.0790479543861094e14,
);

const CUTOFF_ANGLE = 1.6110731556870734; // pi / 1.95 — fakes the earth shadow
const STEEPNESS = 1.5;
const SUN_ENERGY = 1000;
const RAYLEIGH_ZENITH_LENGTH = 8.4e3;
const MIE_ZENITH_LENGTH = 1.25e3;
/** cos of the sun's angular radius (~32 arc minutes). */
const SUN_ANGULAR_DIAMETER_COS = 0.9999566769464485;

/** 3 / ( 16 * pi ) */
const THREE_OVER_SIXTEEN_PI = 0.05968310365946075;
/** 1 / ( 4 * pi ) */
const ONE_OVER_FOUR_PI = 0.07957747154594767;

/**
 * Radius of the dome in metres. Deliberately small: the material disables the
 * depth test and the mesh renders first, so the dome never needs to enclose the
 * scene — it only needs to sit outside the near plane. Keeping it small avoids
 * any dependency on the app's `camera.far`.
 */
const SKY_RADIUS = 100;

/** Distance the directional light is parked at along the sun vector. */
const SUN_LIGHT_DISTANCE = 3000;

/**
 * Environment cube face size.
 *
 * 256, up from 128. The capture feeds every PBR material's specular
 * probe through the PMREM chain, and at 128 the roughest mips were being built
 * from a face barely wider than the blur kernel. The dome is analytic and the
 * capture is throttled (see `updateEnvironment`), so four times the pixels is a
 * cost that does not show up in a frame.
 */
const ENV_SIZE = 256;

/**
 * How much of the sky's own chroma survives a full overcast.
 *
 * Not zero, which is what it effectively was. See the overcast block in
 * `buildSkyNode`.
 */
const OVERCAST_DESATURATION = 0.74;

/** How far a full cloud deck darkens the sky the environment capture sees. */
const CLOUD_ENV_DARKEN = 0.62;

/**
 * How far the key light has to move before the environment is re-captured.
 *
 * About a quarter of a degree. The capture is six render passes and the
 * irradiance it produces is a low-order average, so re-running it for every
 * sub-degree step of the tour's sun is pure cost. Gating on the *value* rather
 * than on a frame counter keeps it deterministic: the same sequence of sun
 * angles from the same reset produces the same sequence of captures, which is
 * what the visual harness rests on.
 */
const ENV_ANGLE_EPSILON = 0.0045;

/**
 * Preetham returns scene-referred radiance in the tens; this maps it onto the
 * renderer's ACES filmic curve.
 *
 * **Re-calibrated when `skyPro`'s exposure went 1 → 0.42.** The old 0.35 was
 * measured against the framebuffer at an exposure of 1 — its own comment said
 * so — and once the grade moved under it the dome was the brightest thing in the
 * scene that had not been re-checked. Measured on the hero island frame against
 * `docs/ref/tropical-island-sea-level-v2.png`: sky 25° up came out at sRGB
 * (90, 160, 203), luminance 148, against the reference's (38, 94, 157) and 87.
 * Seventy percent too bright, and a blue:red ratio of 2.25 against 4.1.
 *
 * The lever is radiance, not chroma, and that is the part worth recording
 * because the obvious alternative was tried and is a dead end. `SKY_CHROMA`
 * below extrapolates colour about luma pre-tonemap, which looks like exactly the
 * right knob; solving the blue/red ratio algebraically asks for 1.57, and at
 * both 1.6 and 1.25 the red channel **clips to zero** and the top of the frame
 * comes out a cyan band. Preetham's linear red at that elevation is already near
 * zero, so chroma extrapolation kills red long before it brings green down — it
 * amplifies the model's hue error instead of correcting it. Travelling *down*
 * the tone curve deepens a blue while keeping its hue, which is the same
 * mechanism that fixed the global grade, and this is where the sky gets on it.
 *
 * It is deliberately not a change to `skyPro.toneMappingExposure` instead: that
 * would take the ship, the island and the sea down with the sky, and the near
 * sea already measures correctly at (30, 93, 123) against (29, 67, 96). Only the
 * dome is hot.
 *
 * This dome is also the source for the environment capture, so cutting it cuts
 * the image-based light on everything — which is why `Seafloor`'s
 * `envMapIntensity` moves in the opposite direction in the same pass. The 0.42
 * there was compensating for a blue-white fill that is now 2.2x smaller in
 * absolute terms, and leaving it would have paid for the sky twice.
 */
const SKY_RADIANCE_SCALE = 0.16;

/**
 * ACES filmic desaturates as it rolls off, which turns a deep zenith blue into
 * pale sky-blue. A small chroma expansion in linear space before tonemapping
 * restores the reference's zenith-to-horizon saturation without touching
 * luminance. This is a grade, not a second tonemap.
 *
 * **1.06 is a ceiling imposed by the model, not a timid setting**, and it is
 * worth recording why so the next reader does not spend the afternoon this cost.
 * Measured 25 degrees above the horizon on `island-approach`, the sky renders
 * sRGB (90, 160, 203) — a blue-to-red ratio of 2.25 — against the art
 * reference's (38, 94, 157) at 4.1, so the obvious move is to push this
 * constant. Solving the ratio algebraically gives k = 1.57.
 *
 * It does not work. Chroma extrapolation about the luma axis amplifies whatever
 * hue the model already has rather than correcting it, and Preetham's linear red
 * at this elevation is already near zero — so red clips to black long before
 * green comes down. Measured: k = 1.25 renders (0, 160, 207) and k = 1.6 renders
 * (0, 161, 214). Both are a *cyan* sky, saturation 1.00, further from the
 * reference than what they replaced even though the saturation number improved.
 * The mid sky does improve at 1.25 — (93, 164, 199) against (115, 163, 195) —
 * but not at the price of a clipped band across the top of every frame.
 *
 * The remaining gap is luminance, not chroma: the reference sky is 87 against
 * this one's 148, and travelling *down* the ACES curve is what deepens a blue
 * while keeping its hue. That correction does not belong here — this dome is the
 * source for the environment capture, so changing its radiance relights every
 * object in the scene — it belongs with the preset's exposure.
 */
const SKY_CHROMA = 1.06;

/**
 * Preetham's `sunIntensity` collapses steeply as the sun approaches the horizon
 * (it fakes the earth's shadow), and the long slant path costs another factor on
 * top. Measured at `exposure: 1`, the sky 10 degrees up drops from RGB ~(176,213,230)
 * at a 26-degree sun to ~(14,31,32) at a 1-degree sun — a 17x fall. A real camera
 * or eye would open up across that range; our renderer's exposure is fixed, so
 * the dome compensates itself. Without this every low-sun preset would have to
 * carry an `exposure` in the teens, and ref-sunset.png would render near black.
 *
 * The curve below is measured, not guessed: for each sun height the framebuffer
 * was sampled across an exposure sweep and the gain that lands the sky 10 degrees
 * up on RGB ~205 green (the value a 20-degree sun produces unaided) was read off.
 * The gain is deliberately capped near the horizon so dusk still reads as dusk
 * rather than flattening into a permanent noon.
 *
 * Pairs are [sin(sunElevation), gain], ascending.
 */
const TWILIGHT_CURVE: ReadonlyArray<readonly [number, number]> = [
  [-1.0, 17],
  [0.0, 17],
  [0.02, 15],
  [0.05, 11.5],
  [0.08, 7.6],
  [0.12, 4.5],
  [0.18, 2.7],
  [0.25, 1.8],
  [0.32, 1.0],
  [1.0, 1.0],
];

function twilightGain(sunY: number): number {
  for (let i = 1; i < TWILIGHT_CURVE.length; i++) {
    const [y1, g1] = TWILIGHT_CURVE[i];
    if (sunY <= y1) {
      const [y0, g0] = TWILIGHT_CURVE[i - 1];
      const k = y1 === y0 ? 0 : (sunY - y0) / (y1 - y0);
      // Smoothstep the blend so the gain has no slope kinks at the knots.
      return g0 + (g1 - g0) * (k * k * (3 - 2 * k));
    }
  }
  return 1;
}

/** Rec. 709 luminance weights. */
const LUMA = /*@__PURE__*/ new THREE.Vector3(0.2126, 0.7152, 0.0722);

export interface AtmosphereParams {
  sunElevation: number; // radians, may be negative (below horizon)
  sunAzimuth: number; // radians
  turbidity: number; // 1..20 haze
  rayleigh: number; // scattering strength
  mieCoefficient: number;
  /**
   * How far the sky is pulled toward flat, achromatic overcast light, 0..1.
   *
   * Not the same thing as cloud coverage, which draws the clouds themselves.
   * This is the light between and behind them, and it is the parameter that
   * makes a storm sky read as weather rather than as dust.
   */
  overcast?: number;
  mieDirectionalG: number; // 0..0.99 forward scattering
  exposure: number;
  groundColor: THREE.Color;
  nightIntensity: number; // 0 = day, 1 = full night (stars + moon)
  moonElevation: number;
  moonAzimuth: number;
}

export const DEFAULT_ATMOSPHERE_PARAMS: AtmosphereParams = {
  sunElevation: 0.42,
  sunAzimuth: 2.6,
  turbidity: 2.6,
  rayleigh: 1.6,
  mieCoefficient: 0.005,
  mieDirectionalG: 0.8,
  /**
   * 1 == the calibrated reference look (see SKY_RADIANCE_SCALE). Treat this as a
   * relative stop, not as three's `toneMappingExposure`.
   */
  exposure: 1,
  groundColor: new THREE.Color(0.05, 0.09, 0.13),
  nightIntensity: 0,
  moonElevation: 0.6,
  moonAzimuth: 1.2,
};

/** Colour of moonlight — a cool, desaturated blue. See ref-moonlit.png. */
const MOON_LIGHT_COLOR = new THREE.Color(0.44, 0.58, 0.9);

// Module-scope scratch — `update`/`setParams` must not allocate.
const _fex = new THREE.Vector3();
const _betaR = new THREE.Vector3();
const _betaM = new THREE.Vector3();

/** Hermite fade, matching GLSL `smoothstep` semantics on the CPU. */
function smoothstepScalar(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Spherical (elevation/azimuth) to the engine's Y-up right-handed basis. */
function directionFromAngles(elevation: number, azimuth: number, out: THREE.Vector3): THREE.Vector3 {
  const c = Math.cos(elevation);
  return out.set(c * Math.sin(azimuth), Math.sin(elevation), c * Math.cos(azimuth));
}

export class Atmosphere {
  readonly mesh: THREE.Mesh;
  readonly sunLight: THREE.DirectionalLight;
  readonly ambientLight: THREE.HemisphereLight;

  private readonly params: AtmosphereParams;

  private readonly geometry: THREE.SphereGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;

  /** Second mesh sharing geometry + material, used only for the env capture. */
  private readonly envMesh: THREE.Mesh;
  private readonly envScene: THREE.Scene;
  private readonly envTarget: THREE.CubeRenderTarget;
  private readonly envCamera: THREE.CubeCamera;
  /** Environment cube face size. */
  environmentResolution: number;
  private envDirty = true;
  /**
   * Set when something other than the light's *direction* changed.
   *
   * The direction throttle below must never swallow a turbidity, overcast or
   * night change: those alter the capture at a fixed sun angle, so a preset
   * switch that happened to leave the sun where it was would otherwise keep the
   * previous preset's image-based light indefinitely.
   */
  private envForce = true;
  private envScene3D: THREE.Scene | null = null;
  private disposed = false;

  private readonly _sunDirection = new THREE.Vector3(0, 1, 0);
  private readonly _moonDirection = new THREE.Vector3(0, 1, 0);
  private readonly _sunColor = new THREE.Color(1, 1, 1);
  private readonly _zenithColor = new THREE.Color(0.32, 0.48, 0.82);
  private readonly _horizonColor = new THREE.Color(0.7, 0.78, 0.9);

  private elapsed = 0;
  /** Clamped copy of `nightIntensity`; also gates the moonlight source. */
  private nightAmount = 0;

  // --- uniforms -------------------------------------------------------------
  private readonly uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uMoonDir = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uBetaR = uniform(new THREE.Vector3());
  private readonly uBetaM = uniform(new THREE.Vector3());
  private readonly uSunE = uniform(0);
  private readonly uMieG = uniform(0.8);
  /** How far toward flat overcast light the sky is pulled, 0..1. */
  private readonly uOvercast = uniform(0);
  /**
   * The colour thick cloud light actually is. Not pure grey: droplet scattering
   * is near-achromatic but the light still arrives having crossed some clear
   * atmosphere, so overcast reads faintly cool rather than neutral.
   */
  /** Set once `setShadowMapSize` has written the shadow's configuration. */
  private shadowConfigured = false;

  /** Half-width of the shadow camera's box, metres. See `setShadowFocus`. */
  private readonly shadowExtent = 260;
  /** Centre of the shadowed region, snapped to texels. See `setShadowFocus`. */
  private readonly shadowFocus = new THREE.Vector3();

  private readonly uOvercastTint: any = uniform(new THREE.Color(0.92, 0.96, 1.04));
  /** Multiplier applied to the flattened sky. Overcast is darker than clear. */
  /**
   * How much darker a fully overcast sky is than the clear one it replaces.
   *
   * 0.78 was far too generous. A thick overcast transmits a small fraction of
   * the light a clear sky at the same sun elevation does — the usual figure is
   * a quarter to a tenth — and at 22% off, the storm preset rendered a bright
   * cream sky, which the ocean then reflected back as a cream sea. 0.34 puts it
   * where a storm belongs: the sun is still a diffuse bright patch through the
   * deck, and everything under it is grey.
   */
  private readonly uOvercastDarken = uniform(0.34);
  /**
   * Day and night are scaled separately, folded on the CPU. They must not share
   * a multiplier: the twilight gain exists to rescue a *sunlit* sky near the
   * horizon, and applying it to the night term as well turns a moonlit sky into
   * a bright blue one.
   */
  private readonly uDayRadiance = uniform(SKY_RADIANCE_SCALE);
  private readonly uNightRadiance = uniform(0);
  // Colour uniforms are held loosely typed: TSL's declaration file narrows a
  // colour uniform to a float-ish node, which blocks legitimate vec3 chaining.
  private readonly uGround: any = uniform(new THREE.Color(0.05, 0.09, 0.13));
  private readonly uSunDiscVisible = uniform(1);
  /**
   * Cloud coverage, applied to the dome **only while the environment is being
   * captured**. Zero in the view, where the marched layer draws the real thing.
   */
  private readonly uEnvCloudAmount = uniform(0);
  private cloudCoverage = 0;
  /** Key direction at the last capture, for the throttle. */
  private readonly envCapturedDir = new THREE.Vector3(0, -2, 0);
  private readonly uMoonDiscCos = uniform(0.9995);
  private readonly uTime = uniform(0);

  constructor(renderer: THREE.WebGPURenderer, environmentResolution = ENV_SIZE) {
    void renderer; // the dome needs no renderer-specific setup; kept for API symmetry

    this.params = { ...DEFAULT_ATMOSPHERE_PARAMS, groundColor: DEFAULT_ATMOSPHERE_PARAMS.groundColor.clone() };
    this.environmentResolution = normalizeResolution(environmentResolution);

    this.geometry = new THREE.SphereGeometry(1, 48, 32);

    this.material = new THREE.MeshBasicNodeMaterial();
    this.material.side = THREE.BackSide;
    this.material.depthWrite = false;
    // The dome is drawn first with the depth test off, so it can be tiny and
    // still sit behind everything — no `camera.far` coupling, no z-fighting.
    this.material.depthTest = false;
    this.material.fog = false;
    this.material.colorNode = this.buildSkyNode();
    // Sprite/points aside, `positionNode` replaces the local position: this
    // recentres the dome on the viewer entirely on the GPU, so there is no
    // per-frame CPU transform work and no need for the camera in `update`.
    this.material.positionNode = positionGeometry.mul(SKY_RADIUS).add(cameraPosition);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'sky-dome';
    this.mesh.renderOrder = -1000;
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;

    // Environment capture: the same dome, but centred on the cube camera at the
    // origin. Geometry and material are shared, so the uniforms stay in sync.
    this.envMesh = new THREE.Mesh(this.geometry, this.material);
    this.envMesh.frustumCulled = false;
    this.envMesh.matrixAutoUpdate = false;
    this.envScene = new THREE.Scene();
    this.envScene.add(this.envMesh);

    this.envTarget = new THREE.CubeRenderTarget(this.environmentResolution, {
      type: THREE.HalfFloatType,
      generateMipmaps: false,
    });
    this.envCamera = new THREE.CubeCamera(1, SKY_RADIUS * 4, this.envTarget);

    this.sunLight = new THREE.DirectionalLight(0xffffff, 3.2);
    this.sunLight.name = 'sun';
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.set(2048, 2048);
    this.sunLight.shadow.camera.near = 1;
    this.sunLight.shadow.camera.far = SUN_LIGHT_DISTANCE * 2;
    this.sunLight.shadow.camera.left = -this.shadowExtent;
    this.sunLight.shadow.camera.right = this.shadowExtent;
    this.sunLight.shadow.camera.top = this.shadowExtent;
    this.sunLight.shadow.camera.bottom = -this.shadowExtent;
    this.sunLight.shadow.bias = -0.0006;
    this.sunLight.shadow.normalBias = 0.6;

    this.ambientLight = new THREE.HemisphereLight(0x88b7ff, 0x0d1a24, 1);
    this.ambientLight.name = 'sky-ambient';

    this.applyParams();
  }

  /**
   * Resizes the environment cube while retaining its CubeTexture object.
   *
   * The cube texture is assigned to `scene.environment` and cached by the
   * material node graph, so replacing `envTarget` would make a tier change
   * sample a disposed resource. `CubeRenderTarget.setSize` releases the old
   * GPU faces in place; call this only after the application's submitted-work
   * fence, then the next environment update rebuilds all six faces and PMREM.
   */
  setEnvironmentResolution(resolution: number): void {
    if (this.disposed) return;
    const next = normalizeResolution(resolution);
    if (next === this.environmentResolution) return;
    this.envTarget.setSize(next, next);
    this.environmentResolution = next;
    this.invalidateEnvironment();
    this.envTarget.texture.needsPMREMUpdate = true;
  }

  /** Alias for callers that apply the same target-resolution API to all systems. */
  setResolution(resolution: number): void {
    this.setEnvironmentResolution(resolution);
  }

  /**
   * Sun shadow map resolution per side.
   *
   * This was honoured once and then frozen, because changing it mid-session
   * crashed the renderer. Three creates a light's shadow node lazily inside
   * `setup()` and caches it on the material, so a runtime change could leave that
   * node holding a null render target, and the planar reflector — which renders
   * the whole scene from its own `updateBefore` — reliably got there first and
   * read `depthTexture` off it. The crash was chased through four triggers
   * (`renderer.shadowMap.enabled`, `shadow.dispose()`, `castShadow`, and the
   * mipmapped reflector resize) and each fix moved it rather than removing it.
   *
   * The reason it kept moving is that none of those was the cause. The cause was
   * destroying a resource still referenced by a command buffer that had been
   * submitted but not retired — WebGPU reports it as "Destroyed texture used in a
   * submit", and *any* of those four could be the thing that happened to trip it.
   * It is now fixed at the source: `App.drainQualityRequests` pauses the loop,
   * awaits the frame already inside `renderAsync`, fences on
   * `queue.onSubmittedWorkDone()`, and only then applies the tier. The GPU is
   * provably idle at this point, so releasing the old map is safe.
   *
   * Releasing it is also required rather than optional: the shadow allocates its
   * target lazily from `mapSize` and then keeps it, so writing a new size without
   * disposing leaves the old target bound and the tier change does nothing at all.
   */
  setShadowMapSize(size: number): void {
    this.sunLight.castShadow = true;
    const side = Math.max(256, Math.round(size));
    if (this.shadowConfigured && this.sunLight.shadow.mapSize.x === side) return;
    this.shadowConfigured = true;
    this.sunLight.shadow.mapSize.set(side, side);
    this.sunLight.shadow.map?.dispose();
    this.sunLight.shadow.map = null;
  }

  /** Current sun shadow map resolution per side. */
  get shadowMapSize(): number {
    return this.sunLight.shadow.mapSize.x;
  }

  /**
   * Moves the shadowed region to follow the viewer.
   *
   * The shadow camera is a +/-260 m box, and it used to be anchored at the world
   * origin because `DirectionalLight.target` defaults to an object there and
   * nothing ever moved it. Everything more than 260 m from the origin was
   * therefore outside it — including the entire island, 1.4 km away, which is
   * why every prop on it had its shadow flags cleared. Those flags were correct
   * given the box; the box was the problem.
   *
   * Following the camera keeps the same resolution over the same area and simply
   * puts it where someone is looking. The cost is that shadows exist only within
   * 260 m of the viewer, which is what was already true near the origin.
   *
   * Snapped to whole shadow texels. Without that the box slides continuously and
   * every shadow edge crawls and fizzes as the camera moves, because each frame
   * rasterises the depth map against a slightly different sub-texel grid — the
   * artefact is far more distracting than the softer edge snapping costs. The
   * snap is done on world XZ rather than in light space, which is exact when the
   * sun is overhead and approximate as it descends; the residual at a low sun is
   * a fraction of a texel and does not read.
   */
  setShadowFocus(x: number, z: number): void {
    const texel = (this.shadowExtent * 2) / Math.max(1, this.sunLight.shadow.mapSize.x);
    this.shadowFocus.set(Math.round(x / texel) * texel, 0, Math.round(z / texel) * texel);
  }

  get sunDirection(): THREE.Vector3 {
    return this._sunDirection;
  }

  get sunColor(): THREE.Color {
    return this._sunColor;
  }

  /**
   * Approximate colour of the sky overhead, and at the horizon.
   *
   * Published so the water can agree with the dome. The surface's reflection and
   * its aerial perspective were driven by preset constants — and, for the sky
   * colour, by the *sun* colour, which is not the sky at all — so wherever the
   * two disagreed the horizon showed a hard step between them. Deriving both
   * from the same state that lights the scene means they cannot drift apart when
   * a preset changes.
   *
   * Approximate on purpose: an exact answer means sampling the dome, and the
   * horizon is precisely where a single sample is least representative.
   */
  get zenithColor(): THREE.Color {
    return this._zenithColor;
  }

  get horizonColor(): THREE.Color {
    return this._horizonColor;
  }

  /** Direction to the moon, normalised, world space. */
  get moonDirection(): THREE.Vector3 {
    return this._moonDirection;
  }

  /**
   * Direction toward whichever body is currently driving `sunLight`.
   *
   * `updateLights` retargets the one shadow-casting light to the moon once the
   * sun is down, so anything computing its own occlusion against that light has
   * to follow the same switch or it will shadow a night scene from a sun that is
   * not there. Published rather than re-derived by each caller, because the
   * threshold is the same `smoothstepScalar(-0.06, 0.1, sunY)` the light itself
   * uses and two copies of it would drift.
   */
  /**
   * The scattering coefficients the dome is currently built from.
   *
   * Published so the shared aerial perspective can take its *hue* from the same
   * atmosphere rather than from a hand-picked triple. Deriving the two
   * independently is how a horizon step gets into a frame.
   */
  get betaRayleigh(): THREE.Vector3 {
    return this.uBetaR.value as THREE.Vector3;
  }

  get betaMie(): THREE.Vector3 {
    return this.uBetaM.value as THREE.Vector3;
  }

  get keyDirection(): THREE.Vector3 {
    return smoothstepScalar(-0.06, 0.1, this._sunDirection.y) > 0.001
      ? this._sunDirection
      : this._moonDirection;
  }

  /**
   * Written out field by field rather than looping `Object.keys` so that an app
   * animating the sun every frame does not allocate a key array per call, and so
   * that the environment map is only invalidated when a value really moved.
   */
  setParams(params: Partial<AtmosphereParams>): void {
    const p = this.params;
    let changed = false;
    /** Anything but the light's own bearing. See `envForce`. */
    let structural = false;

    if (params.sunElevation !== undefined && params.sunElevation !== p.sunElevation) {
      p.sunElevation = params.sunElevation;
      changed = true;
    }
    if (params.sunAzimuth !== undefined && params.sunAzimuth !== p.sunAzimuth) {
      p.sunAzimuth = params.sunAzimuth;
      changed = true;
    }
    if (params.turbidity !== undefined && params.turbidity !== p.turbidity) {
      p.turbidity = params.turbidity;
      changed = true;
      structural = true;
    }
    if (params.rayleigh !== undefined && params.rayleigh !== p.rayleigh) {
      p.rayleigh = params.rayleigh;
      changed = true;
      structural = true;
    }
    if (params.mieCoefficient !== undefined && params.mieCoefficient !== p.mieCoefficient) {
      p.mieCoefficient = params.mieCoefficient;
      changed = true;
      structural = true;
    }
    if (params.mieDirectionalG !== undefined && params.mieDirectionalG !== p.mieDirectionalG) {
      p.mieDirectionalG = params.mieDirectionalG;
      changed = true;
      structural = true;
    }
    if (params.overcast !== undefined && params.overcast !== p.overcast) {
      p.overcast = params.overcast;
      changed = true;
      structural = true;
    }
    if (params.exposure !== undefined && params.exposure !== p.exposure) {
      p.exposure = params.exposure;
      changed = true;
      structural = true;
    }
    if (params.nightIntensity !== undefined && params.nightIntensity !== p.nightIntensity) {
      p.nightIntensity = params.nightIntensity;
      changed = true;
      structural = true;
    }
    if (params.moonElevation !== undefined && params.moonElevation !== p.moonElevation) {
      p.moonElevation = params.moonElevation;
      changed = true;
    }
    if (params.moonAzimuth !== undefined && params.moonAzimuth !== p.moonAzimuth) {
      p.moonAzimuth = params.moonAzimuth;
      changed = true;
    }
    if (params.groundColor !== undefined && !p.groundColor.equals(params.groundColor)) {
      p.groundColor.copy(params.groundColor);
      changed = true;
      structural = true;
    }

    if (!changed) return;
    this.applyParams();
    this.envDirty = true;
    if (structural) this.envForce = true;
  }

  /** Read-only view of the currently applied parameters. */
  getParams(): Readonly<AtmosphereParams> {
    return this.params;
  }

  /**
   * Renders the sky into a small cube target and assigns it as `scene.environment`.
   * Cheap to call every frame: it is a no-op unless a parameter actually changed
   * or the target scene changed.
   */
  /**
   * How cloudy the environment capture should think the sky is, 0..1.
   *
   * Wired from the same coverage the cloud layer is drawn at, so the light an
   * object receives and the deck a viewer sees come from one number.
   */
  setCloudCoverage(coverage: number): void {
    const c = Math.min(1, Math.max(0, coverage));
    if (c === this.cloudCoverage) return;
    this.cloudCoverage = c;
    this.envDirty = true;
    this.envForce = true;
  }

  updateEnvironment(renderer: THREE.WebGPURenderer, scene: THREE.Scene): void {
    if (!this.envDirty && this.envScene3D === scene) return;
    // Throttled on how far the key light has actually moved. `setParams` marks
    // the capture dirty for any change including a sub-degree step of the tour's
    // sun, and six faces for a quarter of a degree is work nothing can see.
    // A scene change is never throttled — that one is not a refinement.
    if (
      !this.envForce &&
      this.envScene3D === scene &&
      this.envCapturedDir.distanceTo(this.keyDirection) < ENV_ANGLE_EPSILON
    ) {
      return;
    }
    this.envScene3D = scene;
    this.envDirty = false;
    this.envForce = false;
    this.envCapturedDir.copy(this.keyDirection);

    // The Preetham sun disc is ~19000x the sky radiance; captured into a cube
    // face it becomes a single blown-out texel that flickers as the sun moves.
    // Sky.js recommends hiding it for environment capture — do the same.
    const disc = this.uSunDiscVisible.value;
    try {
      this.uSunDiscVisible.value = 0;
      // And the deck goes *on* for the capture only: see the note in the sky node.
      this.uEnvCloudAmount.value = this.cloudCoverage;
      this.envCamera.update(renderer, this.envScene);
    } finally {
      // `finally`, because a cube update that threw would otherwise leave the
      // dome permanently sunless and permanently overcast in the *view*, which
      // is a far worse failure than the one that caused it and would look like
      // a shading bug rather than like a lost frame.
      this.uSunDiscVisible.value = disc;
      this.uEnvCloudAmount.value = 0;
    }

    // Force the PMREM chain used by PBR materials to be rebuilt from the new faces.
    this.envTarget.texture.needsPMREMUpdate = true;
    scene.environment = this.envTarget.texture;
  }

  update(dt: number): void {
    this.elapsed += dt;
    // Star twinkle is driven from a CPU-integrated clock rather than the global
    // `time` node so it stays bounded and pauses with the simulation.
    this.uTime.value = this.elapsed % 3600;
  }

  /**
   * Rewinds the animation clock, for reproducible captures.
   *
   * **And invalidates the environment capture**, which is not decoration. The
   * capture is throttled on how far the key light has moved since the last one,
   * and that is a *history*: without clearing it, a reset that happened to land
   * within the threshold of whatever the previous shot left behind would reuse
   * that shot's irradiance. The visual harness runs sixteen shots in one page,
   * so "whatever the previous shot left behind" is exactly the dependency the
   * harness exists to rule out.
   */
  resetClock(time = 0): void {
    this.elapsed = time;
    this.uTime.value = ((time % 3600) + 3600) % 3600;
    this.invalidateEnvironment();
  }

  /**
   * Forces the next `updateEnvironment` to capture, whatever the throttle says.
   */
  invalidateEnvironment(): void {
    this.envDirty = true;
    this.envForce = true;
    // Not a real direction: nothing can be within the threshold of it.
    this.envCapturedDir.set(0, -2, 0);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.geometry.dispose();
    this.material.dispose();
    this.envTarget.dispose();
    this.envScene.remove(this.envMesh);
    this.sunLight.dispose();
    this.sunLight.shadow.dispose();
    this.ambientLight.dispose();
  }

  // -------------------------------------------------------------------------
  // Parameter -> uniform / light derivation (all CPU, all uniform-only inputs)
  // -------------------------------------------------------------------------

  private applyParams(): void {
    const p = this.params;

    directionFromAngles(p.sunElevation, p.sunAzimuth, this._sunDirection).normalize();
    directionFromAngles(p.moonElevation, p.moonAzimuth, this._moonDirection).normalize();

    this.uSunDir.value.copy(this._sunDirection);
    this.uMoonDir.value.copy(this._moonDirection);
    this.uMieG.value = Math.min(0.99, Math.max(0, p.mieDirectionalG));
    const overcast = Math.min(1, Math.max(0, p.overcast ?? 0));
    this.uOvercast.value = overcast;

    // A thick deck hides the sun's disc.
    //
    // The overcast term flattens the sky's *gradient* but the disc is added
    // afterwards at 19000x the sky radiance, so it punched straight through: the
    // storm preset rendered a hard bright patch of sun in the middle of a solid
    // cloud layer, and that one feature was most of why the frame read as a hazy
    // morning rather than as weather. What survives a real overcast is a diffuse
    // brightening around the sun's bearing, which the Mie term already supplies.
    this.uSunDiscVisible.value = 1 - overcast;

    this.nightAmount = Math.min(1, Math.max(0, p.nightIntensity));
    const night = this.nightAmount;
    const base = p.exposure * SKY_RADIANCE_SCALE;
    this.uDayRadiance.value = base * twilightGain(this._sunDirection.y) * (1 - 0.95 * night);
    this.uNightRadiance.value = base * night;
    this.uGround.value.copy(p.groundColor);

    // Preetham's per-frame vertex-stage constants. They depend only on uniforms,
    // so they are cheaper (and more precise) computed once here on the CPU.
    const sunY = this._sunDirection.y;
    const sunfade = 1 - Math.min(1, Math.max(0, 1 - Math.exp(sunY)));
    const rayleighCoefficient = p.rayleigh - (1 - sunfade);

    _betaR.copy(TOTAL_RAYLEIGH).multiplyScalar(rayleighCoefficient);
    const c = 0.2 * p.turbidity * 1e-17;
    _betaM.copy(MIE_CONST).multiplyScalar(0.434 * c * p.mieCoefficient);

    this.uBetaR.value.copy(_betaR);
    this.uBetaM.value.copy(_betaM);

    const zenithCos = Math.min(1, Math.max(-1, sunY));
    this.uSunE.value =
      SUN_ENERGY * Math.max(0, 1 - Math.exp(-((CUTOFF_ANGLE - Math.acos(zenithCos)) / STEEPNESS)));

    this.computeSunColor();
    this.updateLights();
  }

  /**
   * Direct sunlight colour = solar spectrum after atmospheric extinction along
   * the sun's own path. Same `Fex` term the sky shader uses, evaluated once on
   * the CPU so scene lighting cannot drift from the dome.
   */
  private computeSunColor(): void {
    const y = Math.max(0.02, this._sunDirection.y);
    const zenithAngle = Math.acos(y);
    const inverse =
      1 /
      (Math.cos(zenithAngle) +
        0.15 * Math.pow(93.885 - (zenithAngle * 180) / Math.PI, -1.253));
    const sR = RAYLEIGH_ZENITH_LENGTH * inverse;
    const sM = MIE_ZENITH_LENGTH * inverse;

    _fex.set(
      Math.exp(-(_betaR.x * sR + _betaM.x * sM)),
      Math.exp(-(_betaR.y * sR + _betaM.y * sM)),
      Math.exp(-(_betaR.z * sR + _betaM.z * sM)),
    );

    const peak = Math.max(_fex.x, _fex.y, _fex.z) || 1;
    this._sunColor.setRGB(_fex.x / peak, _fex.y / peak, _fex.z / peak);
    // A touch of white keeps a low sun from going fully monochromatic red.
    this._sunColor.lerp(WHITE, 0.08);
  }

  private updateLights(): void {
    const p = this.params;
    const sunUp = smoothstepScalar(-0.06, 0.1, this._sunDirection.y);
    const overcastAmount = Math.min(1, Math.max(0, p.overcast ?? 0));

    if (sunUp > 0.001) {
      this.sunLight.position
        .copy(this._sunDirection)
        .multiplyScalar(SUN_LIGHT_DISTANCE)
        .add(this.shadowFocus);
      this.sunLight.color.copy(this._sunColor);
      // Cloud attenuates the key light, and it was not doing so at all.
      //
      // `overcast` reached the sky dome and the ambient fill's *colour*, but the
      // directional light kept its full clear-day 3.4 — so under a 0.88 deck the
      // storm was lit as if the sun were out: hard specular on every wave, a lit
      // hull, and a scene that read as bright haze rather than as weather. The
      // 0.74 factor matches the dome's own overcast darkening, so the key and the
      // sky it is supposed to come from agree.
      this.sunLight.intensity = 3.4 * sunUp * (1 - 0.74 * overcastAmount);
    } else {
      // Below the horizon the moon becomes the only shadow-casting source, so
      // the same light is retargeted rather than adding a second one. The switch
      // happens where the sun's contribution is already zero.
      const moonUp = smoothstepScalar(-0.02, 0.15, this._moonDirection.y);
      this.sunLight.position
        .copy(this._moonDirection)
        .multiplyScalar(SUN_LIGHT_DISTANCE)
        .add(this.shadowFocus);
      this.sunLight.color.copy(MOON_LIGHT_COLOR);
      this.sunLight.intensity = 0.42 * moonUp * this.nightAmount;
    }

    // The target is what the shadow box is centred on, and it has to be told
    // where that is: three reads `target.matrixWorld`, and the target is not in
    // the scene graph, so nothing else would update it.
    this.sunLight.target.position.copy(this.shadowFocus);
    this.sunLight.target.updateMatrixWorld();

    const day = smoothstepScalar(-0.12, 0.22, this._sunDirection.y);
    const night = this.nightAmount;

    // Hemisphere fill approximates the dome's own irradiance: blue-dominant when
    // the sun is high, sun-tinted at low elevations, near-black at night.
    _skyTint.setRGB(0.32, 0.48, 0.82).lerp(this._sunColor, (1 - day) * 0.6);
    _skyTint.lerp(NIGHT_SKY_TINT, night);
    // Overcast desaturates the fill along with the dome it stands in for, or a
    // grey sky would keep casting blue light into the shadows.
    const overcast = overcastAmount;
    if (overcast > 0) {
      // Partial, for the same reason the dome's is — and this is the second of
      // the three desaturations that were compounding into a monochrome storm.
      // The fill is the light in the *shadows*, and under a deck that light is
      // the whole sky rather than a grey card.
      const grey = _skyTint.r * 0.2126 + _skyTint.g * 0.7152 + _skyTint.b * 0.0722;
      const k = OVERCAST_DESATURATION;
      _overcastFill.setRGB(
        (_skyTint.r + (grey - _skyTint.r) * k) * 0.92,
        (_skyTint.g + (grey - _skyTint.g) * k) * 0.96,
        (_skyTint.b + (grey - _skyTint.b) * k) * 1.04,
      );
      _skyTint.lerp(_overcastFill, overcast);
    }
    this.ambientLight.color.copy(_skyTint);

    // Published for the water surface, which has to agree with the dome at the
    // horizon or the two meet in a visible step. See `zenithColor`.
    this._zenithColor.copy(_skyTint);
    // 0.14, down from 0.42, and this is the fix for the pale far sea.
    //
    // This colour is not decoration: `main` hands it straight to the water as its
    // horizon reflection, and the water's *planar* reflection — the one that
    // contains the actually-rendered sky — is deliberately faded out as the view
    // goes grazing, because a planar reflector cannot be trusted there. So in the
    // far band, where every ray is grazing, the sea is reflecting this analytic
    // stand-in and nothing else. Whitening it by 42% therefore whitened the
    // horizon sea by 42%, which is exactly the defect: the far sea measured a
    // saturation of 0.20 against the reference's 0.66 while the near sea, which
    // reflects at a steeper angle and keeps its own body colour, matched.
    //
    // Two other explanations were tested against that measurement first and both
    // are wrong — weighting the aerial perspective by `1 - fresnel` moved the
    // sample one level, and correcting the haze colour and halving its density
    // moved it three. Neither was ever going to work, because both act on a term
    // that is not what is dominant out there.
    //
    // Some whitening is still right: the sky genuinely does pale toward the
    // horizon through the long slant path, and the dome itself renders that. What
    // 0.42 was doing was applying it a second time, on top of a `_skyTint` that
    // is already the hemisphere average rather than the horizon radiance.
    this._horizonColor.copy(_skyTint).lerp(WHITE, 0.14 * (1 - night));
    this.ambientLight.groundColor.copy(p.groundColor);

    // Night keeps a real fill, not a token one.
    //
    // At 0.03 the hemisphere contributed essentially nothing, and the moon is a
    // single weak directional — so anything facing away from it went to black
    // and the hull was reduced to a flat silhouette with no form. Night does not
    // look like that: the sky is still a hemisphere of sources, moonlight
    // scatters through the whole atmosphere, and the sea throws light back up.
    //
    // 0.16 reads shape without washing the darkness out, and the ground term
    // lifts with it so the underside of the hull picks up the water instead of
    // falling into a void.
    // Overcast moves light out of the key and into the fill rather than removing
    // it: under a deck the whole sky is the source, which is why an overcast day
    // has soft shadows rather than dark ones. Without this the storm simply went
    // dim once the key was attenuated above.
    this.ambientLight.intensity =
      (0.15 + 0.85 * day) * (1 + 0.55 * overcastAmount) * (1 - 0.86 * night) + 0.16 * night;
    if (night > 0.001) {
      // Cool and dim — the colour moonlight actually fills shadows with. Blended
      // rather than assigned, so dusk hands over gradually.
      this.ambientLight.color.lerp(MOON_FILL_COLOR, night * 0.85);
      this.ambientLight.groundColor.lerp(MOON_GROUND_COLOR, night * 0.7);
    }
  }

  // -------------------------------------------------------------------------
  // Node graph
  // -------------------------------------------------------------------------

  /**
   * NOTE: the whole graph is built inside a `Fn` body. TSL only maintains an
   * assignment stack while an `Fn` callback is executing, so `toVar()` and the
   * `*Assign` operators throw if they are reached during plain construction.
   */
  private buildSkyNode(): any {
    return Fn(() => {
      const dir = normalize(positionGeometry).toVar('skyDir');

      // --- Preetham daylight -------------------------------------------------
      const cosZenith = max(dir.y, 0.0);
      const zenithAngle = acos(cosZenith);
      const denom = cos(zenithAngle).add(
        pow(float(93.885).sub(zenithAngle.mul(180 / Math.PI)), -1.253).mul(0.15),
      );
      const opticalInverse = float(1).div(denom);
      const sR = opticalInverse.mul(RAYLEIGH_ZENITH_LENGTH);
      const sM = opticalInverse.mul(MIE_ZENITH_LENGTH);

      const betaR = this.uBetaR;
      const betaM = this.uBetaM;
      const fex = exp(betaR.mul(sR).add(betaM.mul(sM)).negate()).toVar('skyFex');

      const cosTheta = dot(dir, this.uSunDir).toVar('cosTheta');

      // Rayleigh phase, Sky.js convention (argument pre-biased into 0..1).
      const rArg = cosTheta.mul(0.5).add(0.5);
      const rPhase = rArg.mul(rArg).add(1).mul(THREE_OVER_SIXTEEN_PI);
      const mPhase = henyeyGreenstein(cosTheta, this.uMieG);

      const totalBeta = betaR.add(betaM);
      const scatter = betaR.mul(rPhase).add(betaM.mul(mPhase)).div(totalBeta).toVar('scatter');
      const sunE = this.uSunE;

      const lin = pow(
        scatter.mul(sunE).mul(vec3(1, 1, 1).sub(fex)),
        vec3(1.5, 1.5, 1.5),
      ).toVar('lin');
      lin.mulAssign(
        mix(
          vec3(1, 1, 1),
          pow(scatter.mul(sunE).mul(fex), vec3(0.5, 0.5, 0.5)),
          clamp(pow(float(1).sub(this.uSunDir.y), 5.0), 0.0, 1.0),
        ),
      );

      const sunDisc = smoothstep(
        SUN_ANGULAR_DIAMETER_COS,
        SUN_ANGULAR_DIAMETER_COS + 0.00002,
        cosTheta,
      ).mul(this.uSunDiscVisible);
      const l0 = fex.mul(0.1).add(fex.mul(sunE).mul(19000.0).mul(sunDisc));

      const dayColor = lin.add(l0).mul(0.04).add(vec3(0.0, 0.0003, 0.00075)).toVar('dayColor');

      // --- overcast ----------------------------------------------------------
      //
      // Preetham describes a *clear* atmosphere made hazy by aerosol. Asking it
      // for a storm by winding turbidity up to 12 and leaning on the Mie term
      // does not produce an overcast sky: it produces a sandy, brown one, which
      // is what a dusty desert horizon actually looks like and is nothing like
      // weather. The ocean then faithfully reflected that brown, which is why the
      // storm preset had khaki water.
      //
      // Under thick cloud almost none of the light reaching the eye has taken the
      // single-scattering path the model integrates. It has been diffused through
      // a kilometre of water droplets, which are large enough to scatter all
      // visible wavelengths nearly equally — so the sky goes achromatic and
      // loses its vertical gradient. That is what this does: pull toward the
      // sky's own luminance, flatten the gradient, and darken. It is a separate
      // control from cloud coverage because the cloud layer draws the clouds and
      // this describes the light between and behind them.
      // Desaturated toward its own luma, **not replaced by it**, and that is the
      // fix for `storm.png` coming out almost fully monochrome.
      //
      // What stood here was `vec3(skyLuma) * tint`: the chroma was discarded and
      // a tint reapplied to a luma that had already collapsed, so the tint had
      // nothing left to act on. Three desaturations then compounded — this one,
      // the ambient fill's, and the water reflecting the result — and the sea
      // lost its colour entirely. A real storm at sea keeps green-grey in the
      // water; the light under a deck is *diffused*, and diffusion by droplets
      // is near-achromatic but it is not a bleach. Keeping a quarter of the
      // original chroma is what a photograph of one shows.
      const skyLuma = dayColor.dot(vec3(0.2126, 0.7152, 0.0722)).toVar('skyLuma');
      const flat = mix(dayColor, vec3(skyLuma), OVERCAST_DESATURATION)
        .mul(this.uOvercastTint)
        .toVar('skyFlat');
      dayColor.assign(
        mix(dayColor, flat.mul(this.uOvercastDarken), this.uOvercast),
      );

      // The cloud deck's contribution to the environment capture.
      //
      // The capture is the source of every object's image-based light and it saw
      // a *clear* sky: no clouds, no sea, no island. Under a 0.9 overcast that
      // means the hull and the island were being filled with the radiance of a
      // blue sky that is not there — too bright, and much too blue.
      //
      // Deliberately an average rather than the marched layer. Putting the real
      // cloud mesh in `envScene` is the obvious fix and it is what the gap
      // analysis proposes; the arithmetic is against it. The capture is six
      // faces, and the tour moves the sun every frame, so it would be a
      // volumetric raymarch over a third of a million pixels per frame for a
      // term that is by construction a low-order spherical average. Grading the
      // dome by coverage gives that average — the right energy and the right
      // colour — for one `mix`, and the spatial variation it cannot supply is
      // variation an irradiance probe would integrate away regardless.
      const overcastFromDeck = mix(dayColor, vec3(skyLuma).mul(this.uOvercastTint), 0.7);
      dayColor.assign(
        mix(dayColor, overcastFromDeck.mul(CLOUD_ENV_DARKEN), this.uEnvCloudAmount),
      );


      // --- ground half -------------------------------------------------------
      // Below the horizon the dome shows a diffuse ground lit by the horizon sky.
      // It is mostly seen by the environment capture; the ocean hides it in view.
      const groundColor = this.uGround.mul(dayColor.mul(1.4).add(0.004));
      const belowness = smoothstepDown(dir.y, -0.045, 0.0);
      const daySky = mix(dayColor, groundColor, belowness).toVar('daySky');

      // --- night -------------------------------------------------------------
      const nightSky = this.buildNightNode(dir).toVar('nightSky');

      const color = daySky
        .mul(this.uDayRadiance)
        .add(nightSky.mul(this.uNightRadiance))
        .toVar('skyColor');

      const luma = dot(color, vec3(LUMA.x, LUMA.y, LUMA.z));
      const graded = mix(vec3(luma, luma, luma), color, SKY_CHROMA);

      return vec4(graded.max(vec3(0, 0, 0)), 1.0);
    })();
  }

  private buildNightNode(d: any): any {
    // Faintly brighter toward the horizon, as real night skies are.
    const base = mix(
      vec3(0.016, 0.022, 0.044),
      vec3(0.0016, 0.0032, 0.0105),
      saturate(d.y.mul(1.7)),
    );

    // --- star field ----------------------------------------------------------
    // One candidate star per grid cell of direction space. Sparse enough that a
    // single-cell lookup (no 3x3 neighbourhood) is visually indistinguishable
    // and three times cheaper. The cell size is chosen so a star lands at rougly
    // one pixel at a typical field of view — finer than that and the whole field
    // aliases away between frames.
    const cellSpace = d.mul(130.0);
    const cell = floor(cellSpace);
    const local = fract(cellSpace);

    const r0 = hash13(cell);
    const r1 = hash13(cell.add(vec3(11.3, 7.7, 3.1)));
    const r2 = hash13(cell.add(vec3(23.7, 17.3, 5.9)));
    const r3 = hash13(cell.add(vec3(41.1, 29.5, 13.7)));

    const starCentre = vec3(r1, r2, r3);
    const dist = local.sub(starCentre).length();

    const present = smoothstep(0.938, 0.985, r0);
    const core = pow(saturate(float(1).sub(dist.mul(5.0))), 6.0);
    const twinkle = sin(this.uTime.mul(2.6).add(r0.mul(61.0))).mul(0.3).add(0.7);
    const tint = mix(vec3(0.72, 0.81, 1.0), vec3(1.0, 0.88, 0.74), r1);
    const magnitude = r2.mul(0.85).add(0.25);

    const horizonFade = smoothstep(-0.03, 0.2, d.y);
    const stars = tint.mul(core.mul(present).mul(twinkle).mul(magnitude).mul(11.0).mul(horizonFade));

    // --- moon ----------------------------------------------------------------
    const cosMoon = dot(d, this.uMoonDir).toVar('cosMoon');
    const discEdge = this.uMoonDiscCos;
    const disc = smoothstep(discEdge, discEdge.add(0.00025), cosMoon);
    // Cheap maria: low-frequency noise across the disc so it is not a flat blob.
    const maria = mx_noise_float(d.mul(85.0)).mul(0.14).add(0.9);
    const moonBody = vec3(1.0, 0.98, 0.92).mul(disc.mul(maria).mul(9.0));

    const halo = saturate(cosMoon);
    const glow = vec3(0.55, 0.68, 1.0).mul(
      pow(halo, 1400.0).mul(0.5).add(pow(halo, 60.0).mul(0.06)).add(pow(halo, 5.0).mul(0.014)),
    );
    // The halo should not wrap under the horizon.
    const aboveHorizon = smoothstep(-0.08, 0.05, d.y);

    return base.add(stars).add(moonBody.add(glow).mul(aboveHorizon));
  }
}

const WHITE = /*@__PURE__*/ new THREE.Color(1, 1, 1);
const NIGHT_SKY_TINT = /*@__PURE__*/ new THREE.Color(0.12, 0.2, 0.42);
/** Colour moonlight fills shadows with — cool, and never quite neutral. */
const MOON_FILL_COLOR = /*@__PURE__*/ new THREE.Color(0.42, 0.55, 0.86);
/** Light coming back up off the sea at night. */
const MOON_GROUND_COLOR = /*@__PURE__*/ new THREE.Color(0.08, 0.13, 0.22);
const _skyTint = /*@__PURE__*/ new THREE.Color();
const _overcastFill = /*@__PURE__*/ new THREE.Color();

/**
 * Henyey–Greenstein phase function, normalised to 1/(4*pi) like Sky.js.
 * Node-typed values are intentionally `any`: TSL's chained builders are not
 * usefully expressible in TypeScript's type system.
 */
function henyeyGreenstein(cosTheta: any, g: any): any {
  const g2 = g.mul(g);
  const denom = pow(float(1).add(g2).sub(g.mul(cosTheta).mul(2.0)), 1.5);
  return float(1).sub(g2).div(denom).mul(ONE_OVER_FOUR_PI);
}

/** Cheap hash of a lattice cell to [0,1). Deterministic across both backends. */
function hash13(p: any): any {
  const q = p.mul(vec3(127.1, 311.7, 74.7));
  return fract(sin(q.x.add(q.y).add(q.z)).mul(43758.5453123));
}

function normalizeResolution(size: number): number {
  return Number.isFinite(size) ? Math.max(1, Math.round(size)) : ENV_SIZE;
}
