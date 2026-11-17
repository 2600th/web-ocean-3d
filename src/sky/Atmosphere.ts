import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  cameraPosition,
  dot,
  exp,
  float,
  floor,
  fract,
  max,
  mix,
  mx_fractal_noise_float,
  mx_noise_float,
  normalize,
  positionGeometry,
  pow,
  saturate,
  select,
  sin,
  smoothstep,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDown } from '../core/tslMath';
import { PhysicalAtmosphere, SUN_ANG_R } from './PhysicalAtmosphere';

/**
 * Sky dome.
 *
 * The daylight term is a physically based atmosphere — Hillaire's LUT model —
 * **ported from web-ocean** (github.com/Noniv/web-ocean, MIT): see
 * `PhysicalAtmosphere.ts` for what was ported and what changed. It replaced a
 * Preetham port (three's `Sky.js`) because of what a head-to-head against
 * web-ocean showed at sunset: Preetham's Mie term, lifted by the gain needed to
 * keep its collapsing low-sun sky visible, put the sky 6 degrees from a 4-degree
 * sun at 1000x the sky 90 degrees away (measured `skyRadiance`, sunset preset:
 * 253 vs 0.21) — 40% of the frame clipped to white, no sun disc, and the water
 * mirroring it. A physical single+multiple scattering integral bounds that
 * ratio by construction and reddens the sun through the real slant path.
 *
 * The night hemisphere (stars, moon disc, halo), the overcast pull, the twilight
 * glow below the horizon and the ground term are original and unchanged.
 *
 * Everything the caller can tune is a `uniform()`, so `setParams` never rebuilds
 * the node graph — it only writes uniform values.
 */

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
 * Under a continuous deck, the environment capture's horizon band is the
 * deck's own far colour up to `ENV_DECK_HORIZON_FULL` and faded out by
 * `ENV_DECK_HORIZON_TOP` (sines of the elevation: 3 and 17 degrees).
 *
 * The sea mirrors the capture, and at grazing its mirror ray is the horizon,
 * with Fresnel near one. The capture's horizon was the dome's, graded toward a
 * dimmed deck but still brighter than the far deck the frame draws, which
 * converges on the dome's measured radiance a few degrees up
 * (`Clouds.setSkyColors`): under the storm the farthest sea came out a line
 * brighter than both the deck above it and the sea below it. A deck seen edge
 * on is what any mirror at the horizon reflects, so that is what the capture
 * holds there. Off below 0.6 coverage, where the horizon is open sky.
 *
 * Measured in `storm` (sRGB, sky just above the horizon 129): the line was
 * 149; 140 with the band to 9 degrees, which the far sea's blurred lookup
 * reaches past; 137 with this one. It costs the near sea about a sixth of its
 * brightness (mean of the lower third 58 to 48), since it now mirrors the
 * deck the frame shows rather than a dome brighter than it. All-deck (the
 * whole upper capture) took the line to 134 and the sea to 44.
 */
const ENV_DECK_HORIZON_FULL = 0.05;
const ENV_DECK_HORIZON_TOP = 0.3;

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
 * Model radiance (per unit solar illuminance, web-ocean's units) to the dome's
 * pre-`SKY_RADIANCE_SCALE` units.
 *
 * Calibrated so the noon look the presets were graded against is kept: on
 * skyPro (26-degree sun) the sky 25 degrees up and 90 degrees off the sun reads
 * the luma the Preetham dome gave there (0.134 after every scale), measured with
 * `skyRadiance` before and after the swap. Changing this relights everything the
 * environment capture reaches, so it is a calibration, not a look control.
 */
const MODEL_SCALE = 28.5;

/**
 * Drawn radius of the sun's disc, as a multiple of the real one (0.53 degrees
 * across, `SUN_ANG_R`).
 *
 * At the real size the disc is 10 pixels across through a 55-degree lens at
 * 1080p and 19 through a 30-degree one, and at sunset those pixels sat inside
 * a Mie aureole that was itself clipped to white, so nothing separated them
 * (`sky-2`, and with the clouds off). Games that want a readable sun draw it
 * larger than life; 1.8 keeps it a disc rather than a moon-sized ball. Its
 * radiance is divided by the area increase, so the energy the physical disc
 * carries is unchanged before the display cap below applies.
 */
const SUN_DISC_SCALE = 1.8;

/**
 * Display cap on the disc's centre, in tone-mapper input units (scene radiance
 * times `toneMappingExposure`, over three's ACES 0.6), on its brightest channel.
 *
 * The physical disc is ~15000 times the sun's illuminance per steradian, so it
 * is pure white at every exposure and limb darkening, which takes the edge to
 * 35-60 % of the centre, never shows: 60 % of a hundred times white is white.
 * Capped here the centre still clips (ACES reaches 0.96 at 8) while the limb,
 * at 0.35-0.6 of it in the sun's own reddened colour, lands on the shoulder and
 * reads as the warm rim a photographed sunset sun has. Only the drawn disc is
 * capped: the key light, the glint on the water and the environment capture
 * (which hides the disc) never read it.
 */
const SUN_DISC_PEAK = 14;

/**
 * Solar aureole: a forward-scattering halo a few degrees round the disc,
 * drawn after the highlight knee as the disc is, at `AUREOLE_PEAK` in the
 * same tone-mapper input units at the limb and at the reference haze.
 *
 * The round-8 critic called the sunset sun (`sky-4`) "flat with no bloom": a
 * clean disc on a flat glow. The sky-view LUT already holds the Mie lobe, but
 * its texels are a degree or more wide and the knee then compresses what it
 * has, so nothing brightened in the last few degrees round the disc, where a
 * hazy sun's aureole is brightest. Measured aureoles fall steeply over the
 * first degree or two and slowly after (roughly as the inverse square of
 * angle out to ~10 degrees): here a 1-degree and a 4-degree exponential from
 * the limb, 60/40. Single scattering by aerosol scales with the aerosol, so the
 * level goes with the haze over `AUREOLE_REFERENCE_HAZE` (the clear-day
 * presets' 2.5). Swept at 2, 5, 8, 10 and 12: 5 barely reads, 12 starts to
 * wash the clouds next to the sun.
 */
const AUREOLE_PEAK = 10;
const AUREOLE_CORE = (1 * Math.PI) / 180;
const AUREOLE_WIDE = (4 * Math.PI) / 180;
const AUREOLE_WIDE_SHARE = 0.4;
const AUREOLE_REFERENCE_HAZE = 2.5;

/**
 * Soft knee on the sky's own luminance, in the same tone-mapper input units:
 * untouched below `SKY_KNEE_START`, rolled off toward `SKY_KNEE_MAX` above.
 *
 * At the sunset preset the Mie aureole within 4 degrees of the sun measures
 * 7-16 in the dome's units (`skyRadiance`; the sky 90 degrees away is 0.2),
 * which is 6-13 into ACES: flat white in a 20-degree circle around the sun,
 * which is what swallowed the disc. The physical ratio is right — a camera
 * would clip it too — but a clipped circle is not what the art references
 * draw. Compressing only the top of the range keeps the aureole a bright warm
 * glow with its hue (the red channel still clips first, so it goes orange, not
 * grey) and leaves headroom for the disc, while everything under the knee — the
 * whole noon sky except the sun's immediate surround — is unchanged.
 */
const SKY_KNEE_START = 1.0;
const SKY_KNEE_MAX = 2.2;
/** three's ACES divides by this before its curve (`ACESFilmicToneMapping`). */
const ACES_INPUT_SCALE = 0.6;

/** Haze when a caller gives none: web-ocean's sea air is 3, a clear trade-wind day less. */
const DEFAULT_HAZE = 1.5;

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
 * Partial eye adaptation to the low sun, as a gain on the whole dome.
 *
 * The presets carry fixed exposures and the tour and the time slider move the
 * sun through all of them, so as the sun drops the sky away from it falls
 * several stops and nothing would open up for it. This was a measured table
 * for the Preetham dome (up to 17x at the horizon) and that table is what made
 * sunsets clip: it multiplied the Mie halo along with everything else.
 *
 * Now it is derived from the model itself: the luma of the sky 10 degrees up,
 * averaged over the three bearings away from the sun, against the same at the
 * reference 26-degree sun, and only the square root of the ratio is applied —
 * half the stops, the same "correct part of the deviation" rule
 * `post/AutoExposure` uses, so dusk still reads darker than noon. The sun-side
 * sky is excluded so the gain cannot be driven by the halo it would brighten.
 */
const TWILIGHT_MAX_GAIN = 4;
/** sin(26.4 deg), the skyPro sun the dome was calibrated at. */
const REFERENCE_SUN_Y = 0.4439;

/**
 * The twilight glow, as a strength 0..1 for a sun elevation (sine).
 *
 * Preetham's sun energy is cut off a couple of degrees below the horizon —
 * `CUTOFF_ANGLE` fakes the earth's shadow — and past it the dome has nothing
 * but the night term. So the first half-hour after sunset, the most colourful
 * sky of the day, rendered *black*: the tour's dusk squall at a sun 4.7 degrees
 * down had a dome a seventieth of noon's under a deck lit for daylight. A real
 * civil twilight is an orange band on the sun's bearing, a purple belt above
 * it and a deepening blue overhead, fading out by about twelve degrees down.
 * This supplies that, peaking just after sunset and gone well before night.
 */
function twilightGlow(sunY: number): number {
  // Below the horizon only. While the sun is up the physical model already
  // draws the warm horizon, and this authored band — its purple belt in
  // particular — stacked on top of it was part of the pink-and-cyan sunset
  // the head-to-head called "candy".
  return smoothstepScalar(-0.22, -0.03, sunY) * (1 - smoothstepScalar(-0.03, 0.0, sunY)) * 0.9;
}
const TWILIGHT_HORIZON = /*@__PURE__*/ new THREE.Color(1.0, 0.42, 0.15);
const TWILIGHT_BELT = /*@__PURE__*/ new THREE.Color(0.42, 0.3, 0.6);
const TWILIGHT_ZENITH = /*@__PURE__*/ new THREE.Color(0.08, 0.13, 0.32);

/** Rec. 709 luminance weights. */
const LUMA = /*@__PURE__*/ new THREE.Vector3(0.2126, 0.7152, 0.0722);

export interface AtmosphereParams {
  sunElevation: number; // radians, may be negative (below horizon)
  sunAzimuth: number; // radians
  /**
   * Aerosol amount, web-ocean's "Haze" (a multiplier on its marine Mie
   * profile: 1 clear, 3 their default sea air, 6+ thick haze). Drives the
   * dome, the sun's colour and the aerial-perspective hue.
   */
  haze?: number;
  /** Preetham-era parameters, kept for preset compatibility; the physical model ignores them. */
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
  haze: DEFAULT_HAZE,
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
/** Moon key at full night, in the sun's units (3.4 at noon). See `updateLights`. */
const MOON_KEY = 0.6;
/**
 * Hemisphere fill at full night. The environment capture of a night sky is a
 * few thousandths — the dome's night term is starlight, not moonlit air — so at
 * night this fill *is* the sky light, and it scales with the key to keep the
 * day's key-to-fill ratio (about 2.5:1 here) rather than going flat.
 */
const MOON_FILL = 0.24;

// Module-scope scratch — `update`/`setParams` must not allocate.
const _dir = new THREE.Vector3();
const _rad = new THREE.Color();
const _acc = new THREE.Color();
/** Rayleigh and Mie at sea level, per metre (web-ocean `RAY_S`, `MIE_E` x haze, /1000). */
const RAY_PER_M = /*@__PURE__*/ new THREE.Vector3(5.802e-6, 13.558e-6, 33.1e-6);
const MIE_E_PER_M = 4.4e-6;

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
  /** The physical model: LUTs, sky-view pass and CPU queries. */
  private readonly model = new PhysicalAtmosphere();
  private readonly betaR = new THREE.Vector3().copy(RAY_PER_M);
  private readonly betaM = new THREE.Vector3();
  /** Model luma of the reference (26-degree) sky, per haze; see `twilightGain`. */
  private referenceLuma = 0;
  private referenceHaze = -1;
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
  /** Twilight glow strength, radiance units. See `twilightGlow`. */
  private readonly uTwilight = uniform(0);
  /** Horizontal unit vector toward the sun, for the twilight glow's bearing. */
  private readonly uSunFlat = uniform(new THREE.Vector3(1, 0, 0));
  // Colour uniforms are held loosely typed: TSL's declaration file narrows a
  // colour uniform to a float-ish node, which blocks legitimate vec3 chaining.
  private readonly uGround: any = uniform(new THREE.Color(0.05, 0.09, 0.13));
  private readonly uSunDiscVisible = uniform(1);
  /**
   * How much of the stars and the moon an overcast leaves: the disc's own
   * `clear * clear`, kept apart from `uSunDiscVisible` because the environment
   * capture zeroes that one and has no reason to lose the stars. The storm
   * (overcast 0.88, `nightIntensity` 0.1) drew its star field into every gap
   * in the deck.
   */
  private readonly uNightClear = uniform(1);
  /** See `setSunOccluder`. */
  private sunOccluder: ((dir: any) => any) | null = null;
  /** The disc's drawn radiance over `T / max(T)`, scene units; see `SUN_DISC_PEAK`. */
  private readonly uDiscLevel = uniform(0);
  /** The aureole's level at the limb over `T / max(T)`, scene units; see `AUREOLE_PEAK`. */
  private readonly uAureoleLevel = uniform(0);
  /** `SKY_KNEE_START` / `SKY_KNEE_MAX` in scene units at the current display exposure. */
  private readonly uKneeStart = uniform(SKY_KNEE_START * ACES_INPUT_SCALE);
  private readonly uKneeMax = uniform(SKY_KNEE_MAX * ACES_INPUT_SCALE);
  /** The renderer's `toneMappingExposure`, which the knee and the disc cap are set against. */
  private displayExposure = 1;
  /**
   * Cloud coverage, applied to the dome **only while the environment is being
   * captured**. Zero in the view, where the marched layer draws the real thing.
   */
  private readonly uEnvCloudAmount = uniform(0);
  private cloudCoverage = 0;
  /**
   * What a continuous deck looks like at the horizon (`setDeckHorizon`), and how
   * continuous it is: the second only while the environment is captured, like
   * `uEnvCloudAmount`. See `ENV_DECK_HORIZON_TOP`.
   */
  private readonly uEnvDeckHorizon: any = uniform(new THREE.Color(0.5, 0.55, 0.6));
  private readonly uEnvDeckHold = uniform(0);
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
    // A one-shot render: see `syncPipeline` in `main.ts`. A capture dropped for
    // want of a pipeline is not redrawn until the sun next moves, and in the
    // meantime everything lit or reflected from the environment goes dark.
    this.envMesh.userData.syncPipeline = true;
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
  /** How far toward flat overcast light the sky is pulled, 0..1. */
  get overcast(): number {
    return this.uOvercast.value as number;
  }

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
    return this.betaR;
  }

  /**
   * The dome's own radiance in direction `dir`, pre-tonemap — what the sky
   * shader would draw there, less the sun disc, the stars and the moon.
   *
   * A CPU port of `buildSkyNode` for the few directions a caller needs per
   * frame. It exists because the cloud layer's ambient light was a fixed
   * fraction of white plus the hemisphere-fill *tint*, and that tint is a hue,
   * not a brightness: at night it is a mid blue at full scale while the dome
   * itself renders at a few thousandths. The deck was therefore lit hundreds of
   * times brighter than the sky around it and came out near-white against
   * black. Anything that has to sit *in* the sky has to be lit by the sky's
   * actual radiance, and this is that number.
   */
  skyRadiance(dir: THREE.Vector3, out: THREE.Color): THREE.Color {
    this.model.radiance(dir, this._sunDirection, this.params.mieDirectionalG, _rad);
    let dr = _rad.r * MODEL_SCALE;
    let dg = _rad.g * MODEL_SCALE;
    let db = _rad.b * MODEL_SCALE;

    const overcast = this.uOvercast.value as number;
    if (overcast > 0) {
      const luma = dr * LUMA.x + dg * LUMA.y + db * LUMA.z;
      const tint = this.uOvercastTint.value as THREE.Color;
      const darken = (this.uOvercastDarken.value as number) * overcast;
      const k = OVERCAST_DESATURATION;
      dr += ((dr + (luma - dr) * k) * tint.r * darken - dr * overcast);
      dg += ((dg + (luma - dg) * k) * tint.g * darken - dg * overcast);
      db += ((db + (luma - db) * k) * tint.b * darken - db * overcast);
    }

    const up = Math.min(1, Math.max(0, dir.y * 1.7));
    const day = this.uDayRadiance.value as number;
    const night = this.uNightRadiance.value as number;

    const flat = this.uSunFlat.value as THREE.Vector3;
    const flatLength = Math.max(1e-3, Math.sqrt(dir.x * dir.x + dir.z * dir.z));
    // Clamped: a bearing exactly opposite the sun rounds to -1e-17, and `pow` of
    // that is NaN — which `measureSkyRadiance` fed straight into the clouds.
    const bearing = Math.max(0, ((dir.x * flat.x + dir.z * flat.z) / flatLength) * 0.5 + 0.5);
    const h = Math.max(0, dir.y);
    const tw = this.uTwilight.value as number;
    const wHorizon = Math.pow(bearing, 3) * Math.exp(-9 * h) * tw;
    const wBelt = Math.pow(bearing, 1.5) * Math.exp(-3 * h) * 0.5 * tw;
    const wZenith = (h * 0.5 + 0.5) * 0.35 * tw;

    out.setRGB(
      dr * day + (NIGHT_HORIZON.r + (NIGHT_ZENITH.r - NIGHT_HORIZON.r) * up) * night +
        TWILIGHT_HORIZON.r * wHorizon + TWILIGHT_BELT.r * wBelt + TWILIGHT_ZENITH.r * wZenith,
      dg * day + (NIGHT_HORIZON.g + (NIGHT_ZENITH.g - NIGHT_HORIZON.g) * up) * night +
        TWILIGHT_HORIZON.g * wHorizon + TWILIGHT_BELT.g * wBelt + TWILIGHT_ZENITH.g * wZenith,
      db * day + (NIGHT_HORIZON.b + (NIGHT_ZENITH.b - NIGHT_HORIZON.b) * up) * night +
        TWILIGHT_HORIZON.b * wHorizon + TWILIGHT_BELT.b * wBelt + TWILIGHT_ZENITH.b * wZenith,
    );
    // The highlight knee, as the shader applies it.
    const raw = out.r * LUMA.x + out.g * LUMA.y + out.b * LUMA.z;
    const k0 = this.uKneeStart.value as number;
    const k1 = this.uKneeMax.value as number;
    if (raw > k0) {
      const over = raw - k0;
      out.multiplyScalar((k0 + over / (1 + over / (k1 - k0))) / raw);
    }
    const luma = out.r * LUMA.x + out.g * LUMA.y + out.b * LUMA.z;
    return out.setRGB(
      Math.max(0, luma + (out.r - luma) * SKY_CHROMA),
      Math.max(0, luma + (out.g - luma) * SKY_CHROMA),
      Math.max(0, luma + (out.b - luma) * SKY_CHROMA),
    );
  }

  get betaMie(): THREE.Vector3 {
    return this.betaM;
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
    if (params.haze !== undefined && params.haze !== p.haze) {
      p.haze = params.haze;
      changed = true;
      structural = true;
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

  /**
   * The renderer's `toneMappingExposure`, which the sky's highlight knee and the
   * disc's display cap are expressed against (`SKY_KNEE_START`, `SUN_DISC_PEAK`).
   * A uniform write; re-captures the environment, since the knee is in it.
   */
  setDisplayExposure(exposure: number): void {
    const e = Math.max(1e-3, exposure);
    if (e === this.displayExposure) return;
    this.displayExposure = e;
    this.applyParams();
    this.envDirty = true;
    this.envForce = true;
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

  /**
   * The radiance a continuous deck shows at the horizon: `Clouds`' far-deck
   * colour (`setSkyColors`' horizon), which the capture's horizon band takes
   * under a deck. A uniform write; it reaches the cube at the next capture.
   */
  setDeckHorizon(radiance: THREE.Color): void {
    (this.uEnvDeckHorizon.value as THREE.Color).copy(radiance);
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

    // The sun disc is ~15000x the sun's illuminance per steradian (1 / pi R^2); captured into a cube
    // face it becomes a single blown-out texel that flickers as the sun moves.
    // Sky.js recommends hiding it for environment capture — do the same.
    const disc = this.uSunDiscVisible.value;
    try {
      this.uSunDiscVisible.value = 0;
      // And the deck goes *on* for the capture only: see the note in the sky node.
      this.uEnvCloudAmount.value = this.cloudCoverage;
      const t = Math.min(1, Math.max(0, (this.cloudCoverage - 0.6) / 0.25));
      this.uEnvDeckHold.value = t * t * (3 - 2 * t);
      this.envCamera.update(renderer, this.envScene);
    } finally {
      // `finally`, because a cube update that threw would otherwise leave the
      // dome permanently sunless and permanently overcast in the *view*, which
      // is a far worse failure than the one that caused it and would look like
      // a shading bug rather than like a lost frame.
      this.uSunDiscVisible.value = disc;
      this.uEnvCloudAmount.value = 0;
      this.uEnvDeckHold.value = 0;
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

  /** True while a forced capture (see `envForce`) is waiting for `updateEnvironment`. */
  get environmentForced(): boolean {
    return this.envDirty && this.envForce;
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
    this.model.dispose();
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
    this.model.setHaze(p.haze ?? DEFAULT_HAZE);
    this.model.setSun(this._sunDirection, p.mieDirectionalG);
    const overcast = Math.min(1, Math.max(0, p.overcast ?? 0));
    this.uOvercast.value = overcast;

    // A thick deck hides the sun's disc.
    //
    // The overcast term flattens the sky's *gradient* but the disc is added
    // afterwards at ~15000x (19000x under Preetham), so it punched straight through: the
    // storm preset rendered a hard bright patch of sun in the middle of a solid
    // cloud layer, and that one feature was most of why the frame read as a hazy
    // morning rather than as weather. What survives a real overcast is a diffuse
    // brightening around the sun's bearing, which the Mie term already supplies.
    //
    // And it has to go to *zero*, not to 12 %: at that ratio, an eighth of the disc
    // is still two thousand times the sky, and it read as a full moon hanging in
    // a storm. Gone by three-quarters overcast, eased on the way.
    const clear = Math.max(0, 1 - overcast / 0.75);
    this.uSunDiscVisible.value = clear * clear;
    this.uNightClear.value = clear * clear;

    this.nightAmount = Math.min(1, Math.max(0, p.nightIntensity));
    const night = this.nightAmount;
    const base = p.exposure * SKY_RADIANCE_SCALE;
    this.uDayRadiance.value = base * this.twilightGain() * (1 - 0.95 * night);
    this.uNightRadiance.value = base * night;
    this.uTwilight.value =
      base * twilightGlow(this._sunDirection.y) * (1 - 0.75 * overcast) * (1 - 0.6 * night);
    (this.uSunFlat.value as THREE.Vector3)
      .set(this._sunDirection.x, 0, this._sunDirection.z)
      .normalize();
    this.uGround.value.copy(p.groundColor);

    // The knee and the disc cap, from tone-mapper input units to scene units.
    const toScene = ACES_INPUT_SCALE / this.displayExposure;
    this.uKneeStart.value = SKY_KNEE_START * toScene;
    this.uKneeMax.value = SKY_KNEE_MAX * toScene;
    // The disc: physical radiance (energy-conserving at the drawn size) scaled
    // like the dome, capped for display, over its own peak transmittance so
    // the shader's `T(dir) / max(T)` keeps the lower limb redder than the upper.
    this.model.transmittance(this._sunDirection.y, _rad);
    const peakT = Math.max(_rad.r, _rad.g, _rad.b);
    const drawnR = SUN_ANG_R * SUN_DISC_SCALE;
    const physical = (peakT / (Math.PI * drawnR * drawnR)) * MODEL_SCALE * (this.uDayRadiance.value as number);
    const cap = SUN_DISC_PEAK * toScene * (1 - 0.95 * night);
    this.uDiscLevel.value = peakT > 1e-6 ? Math.min(physical, cap) / peakT : 0;
    // The aureole: `AUREOLE_PEAK` at the limb, in proportion to the haze, gone
    // as the sun sets (so no halo rises over the horizon from a sun below it),
    // over the same peak transmittance as the disc so it carries the sun's colour.
    {
      const s = Math.min(1, Math.max(0, (this._sunDirection.y + 0.03) / 0.05));
      const risen = s * s * (3 - 2 * s);
      const hazeScale = this.model.currentHaze / AUREOLE_REFERENCE_HAZE;
      this.uAureoleLevel.value =
        peakT > 1e-6 ? (AUREOLE_PEAK * toScene * (1 - 0.95 * night) * risen * hazeScale) / peakT : 0;
    }

    // Published for the aerial perspective, which takes its *hue* from these.
    // web-ocean's own ocean fog uses exactly this sum (`fogExt`).
    const haze = this.model.currentHaze;
    this.betaM.setScalar(MIE_E_PER_M * haze);

    this.computeSunColor();
    this.updateLights();
  }

  /**
   * Direct sunlight colour = solar spectrum after atmospheric extinction along
   * the sun's own path. Same `Fex` term the sky shader uses, evaluated once on
   * the CPU so scene lighting cannot drift from the dome.
   */
  private computeSunColor(): void {
    // Transmittance along the sun's own slant path, from the same table the
    // dome is integrated with, so the key light cannot drift from the sky.
    // Floored at 1 degree: below that the path ends in the planet and the
    // colour is undefined, while the light itself is already fading out.
    this.model.transmittance(Math.max(0.0175, this._sunDirection.y), this._sunColor);
    const c = this._sunColor;
    const peak = Math.max(c.r, c.g, c.b) || 1;
    c.setRGB(c.r / peak, c.g / peak, c.b / peak);
    // A touch of white keeps a low sun from going fully monochromatic red.
    c.lerp(WHITE, 0.08);
  }

  /** See `TWILIGHT_MAX_GAIN`. Three CPU marches; called from `applyParams` only. */
  private twilightGain(): number {
    const haze = this.model.currentHaze;
    if (haze !== this.referenceHaze) {
      this.referenceHaze = haze;
      _dir.set(Math.sqrt(1 - REFERENCE_SUN_Y * REFERENCE_SUN_Y), REFERENCE_SUN_Y, 0);
      this.referenceLuma = this.awayFromSunLuma(_dir);
    }
    const now = this.awayFromSunLuma(this._sunDirection);
    if (!(now > 1e-9)) return TWILIGHT_MAX_GAIN;
    return Math.min(TWILIGHT_MAX_GAIN, Math.max(1, Math.sqrt(this.referenceLuma / now)));
  }

  /** Mean model luma 10 degrees up at 90, 180 and 270 degrees from the sun's bearing. */
  private awayFromSunLuma(sun: THREE.Vector3): number {
    const az = Math.atan2(sun.x, sun.z);
    const ce = Math.cos(0.1745);
    const se = Math.sin(0.1745);
    let sum = 0;
    for (let k = 1; k <= 3; k++) {
      const a = az + (k * Math.PI) / 2;
      _dir.set(ce * Math.sin(a), se, ce * Math.cos(a));
      this.model.radiance(_dir, sun, this.params.mieDirectionalG, _rad);
      sum += _rad.r * LUMA.x + _rad.g * LUMA.y + _rad.b * LUMA.z;
    }
    return sum / 3;
  }

  /**
   * Hue of the clear sky's hemisphere, as a colour of luma `luma`: the model
   * at the zenith and 15 degrees up on four bearings, weighted 1:4 toward the
   * horizon the way a fill from a hemisphere is.
   */
  private hemisphereHue(luma: number, out: THREE.Color): THREE.Color {
    const sun = this._sunDirection;
    const g = this.params.mieDirectionalG;
    _dir.set(0, 1, 0);
    this.model.radiance(_dir, sun, g, _acc);
    const ce = Math.cos(0.26);
    const se = Math.sin(0.26);
    for (let k = 0; k < 4; k++) {
      const a = (k * Math.PI) / 2 + 0.785;
      _dir.set(ce * Math.sin(a), se, ce * Math.cos(a));
      this.model.radiance(_dir, sun, g, _rad);
      _acc.r += _rad.r;
      _acc.g += _rad.g;
      _acc.b += _rad.b;
    }
    const l = _acc.r * LUMA.x + _acc.g * LUMA.y + _acc.b * LUMA.z;
    if (!(l > 1e-9)) return out.setRGB(0.32, 0.48, 0.82);
    return out.setRGB((_acc.r * luma) / l, (_acc.g * luma) / l, (_acc.b * luma) / l);
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
      // 0.6, up from 0.42. At 0.42 a dark-wood hull under a high moon rendered
      // at a few percent linear — a silhouette with no planking, no rail and no
      // form — while the sea beside it, which reflects the moonlit sky, stayed
      // readable. A game's night is not a photometric night (a full moon is
      // about a millionth of the sun); it is the day's lighting *ratios* at a
      // dark exposure. 0.6 is 18% of the noon sun, which with the fill below
      // keeps the key-to-fill contrast of a clear day and lets the hull read.
      this.sunLight.intensity = MOON_KEY * moonUp * this.nightAmount;
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
    //
    // Sun-tinted only while there is a sun to tint it. Below the horizon the
    // extinction colour goes on reporting a deep red, and mixing that into the
    // blue fill made the whole blue hour purple-pink — the sea at the dusk
    // squall mirrored it as a flat mauve under a sky that is blue everywhere
    // but its sunward horizon. The dome keeps that warm band itself.
    //
    // Not any more by mixing the noon blue with the *sun's* colour: blue plus a
    // red sun is lavender, and at the sunset preset that put the fill at
    // (0.46, 0.40, 0.64) — the magenta the sea mirrored in every sunset pair of
    // the head-to-head. The hue now comes from the physical sky itself (warm
    // horizon, blue zenith, averaged), at the luma the old noon tint had
    // (0.47), so only the colour moves and every consumer tuned to the old
    // brightness keeps it. By day this is within a few percent of the old blue.
    this.hemisphereHue(0.47, _skyTint);
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
    // `MOON_FILL` reads shape without washing the darkness out, and the ground term
    // lifts with it so the underside of the hull picks up the water instead of
    // falling into a void.
    // Overcast moves light out of the key and into the fill rather than removing
    // it: under a deck the whole sky is the source, which is why an overcast day
    // has soft shadows rather than dark ones. Without this the storm simply went
    // dim once the key was attenuated above.
    this.ambientLight.intensity =
      (0.15 + 0.85 * day) * (1 + 0.55 * overcastAmount) * (1 - 0.86 * night) + MOON_FILL * night;
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

      // --- physical daylight (web-ocean's sky-view LUT + sun disc) ------------
      // Above the horizon the LUT; the dome's lower half is replaced by the
      // ground term below, so the LUT is read at the horizon there (as
      // web-ocean clamps `dir.y` to 0.0005).
      const up = normalize(vec3(dir.x, max(dir.y, 0.0005), dir.z));
      const sky = this.model.skyView(up);
      // The disc is added after the highlight knee below, at its own display
      // level (`SUN_DISC_PEAK`), so the knee cannot compress it back into the glow.
      const disc = this.discRadiance(dir);
      const dayColor = sky.mul(MODEL_SCALE).toVar('dayColor');

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
      // Branched rather than multiplied away: the night term is three noise
      // lookups and a star hash per pixel of the dome, and by day it is scaled
      // by exactly zero. A uniform branch is coherent across the whole draw.
      const nightSky = vec3(0, 0, 0).toVar('nightSky');
      If(this.uNightRadiance.greaterThan(0), () => {
        nightSky.assign(this.buildNightNode(dir));
      });

      // --- twilight ----------------------------------------------------------
      // See `twilightGlow`. Zero outside a band of sun elevations around the
      // horizon, and kept off the ground half.
      const flatLength = dir.x.mul(dir.x).add(dir.z.mul(dir.z)).sqrt().max(1e-3);
      const bearing = dir.x
        .mul(this.uSunFlat.x)
        .add(dir.z.mul(this.uSunFlat.z))
        .div(flatLength)
        .mul(0.5)
        .add(0.5)
        .max(0);
      const above = max(dir.y, 0.0);
      const twilight = vec3(TWILIGHT_HORIZON.r, TWILIGHT_HORIZON.g, TWILIGHT_HORIZON.b)
        .mul(pow(bearing, 3.0).mul(exp(above.mul(-9.0))))
        .add(vec3(TWILIGHT_BELT.r, TWILIGHT_BELT.g, TWILIGHT_BELT.b).mul(pow(bearing, 1.5).mul(exp(above.mul(-3.0))).mul(0.5)))
        .add(vec3(TWILIGHT_ZENITH.r, TWILIGHT_ZENITH.g, TWILIGHT_ZENITH.b).mul(above.mul(0.5).add(0.5).mul(0.35)))
        .mul(belowness.oneMinus())
        .mul(this.uTwilight);

      const color = daySky
        .mul(this.uDayRadiance)
        .add(nightSky.mul(this.uNightRadiance))
        .add(twilight)
        .toVar('skyColor');

      // Under a continuous deck, the capture's horizon is the deck's own far
      // colour (`ENV_DECK_HORIZON_TOP`). Zero in the view.
      color.assign(
        mix(color, this.uEnvDeckHorizon, smoothstepDown(dir.y, ENV_DECK_HORIZON_FULL, ENV_DECK_HORIZON_TOP).mul(this.uEnvDeckHold)),
      );

      // Highlight knee (`SKY_KNEE_START`): luminance rolled off, hue kept.
      const raw = dot(color, vec3(LUMA.x, LUMA.y, LUMA.z));
      const over = raw.sub(this.uKneeStart).max(0);
      const kneed = this.uKneeStart.add(over.div(over.div(this.uKneeMax.sub(this.uKneeStart)).add(1)));
      color.mulAssign(select(raw.greaterThan(this.uKneeStart), kneed.div(raw.max(1e-6)), float(1)));
      color.addAssign(disc.add(this.aureoleRadiance(dir)).mul(belowness.oneMinus()));

      const luma = dot(color, vec3(LUMA.x, LUMA.y, LUMA.z));
      const graded = mix(vec3(luma, luma, luma), color, SKY_CHROMA);

      return vec4(graded.max(vec3(0, 0, 0)), 1.0);
    })();
  }

  /** The drawn sun disc toward `dir`, hidden behind the cloud deck. */
  private discRadiance(dir: any): any {
    const disc = this.model
      .sunDisc(dir, SUN_ANG_R * SUN_DISC_SCALE)
      .mul(this.uDiscLevel)
      .mul(this.uSunDiscVisible);
    return this.sunOccluder === null ? disc : disc.mul(this.sunOccluder(dir));
  }

  /**
   * The solar aureole round the disc (`AUREOLE_PEAK`), off with the disc: by
   * overcast (`uSunDiscVisible`, which the environment capture also zeroes) and
   * by the cloud in front of the sun, read once at the sun's own direction, so
   * a cloud over the disc takes the whole halo with it rather than leaving a
   * ring round the cloud's edge.
   */
  private aureoleRadiance(dir: any): any {
    const halo = this.model
      .sunAureole(dir, SUN_ANG_R * SUN_DISC_SCALE, float(AUREOLE_CORE), float(AUREOLE_WIDE), float(AUREOLE_WIDE_SHARE))
      .mul(this.uAureoleLevel)
      .mul(this.uSunDiscVisible);
    return this.sunOccluder === null ? halo : halo.mul(this.sunOccluder(this.model.uSunDir));
  }

  /**
   * What the cloud deck lets through of the sun's disc toward a direction
   * (`Clouds.sunTransmittanceNode`). Rebuilds the dome's graph.
   */
  setSunOccluder(occluder: (dir: any) => any): void {
    if (this.sunOccluder === occluder) return;
    this.sunOccluder = occluder;
    this.material.colorNode = this.buildSkyNode();
    this.material.needsUpdate = true;
  }

  private buildNightNode(d: any): any {
    // Brighter toward the horizon, as real night skies are. See `NIGHT_HORIZON`.
    const base = mix(
      vec3(NIGHT_HORIZON.r, NIGHT_HORIZON.g, NIGHT_HORIZON.b),
      vec3(NIGHT_ZENITH.r, NIGHT_ZENITH.g, NIGHT_ZENITH.b),
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

    // --- the galaxy ---------------------------------------------------------
    //
    // A band of diffuse light around a great circle, tilted against the
    // horizon the way the real one is, with a brighter bulge toward one end
    // and a dark dust rift along its spine. Three things sell it: the band is
    // *clumped* (star clouds, not a gradient), the rift is off-centre and
    // ragged, and the star field itself is denser inside it — the band is
    // made of the same stars as the rest of the sky, just more of them.
    const plane = d.dot(vec3(GALACTIC_POLE.x, GALACTIC_POLE.y, GALACTIC_POLE.z));
    const bandCore = exp(plane.mul(plane).mul(-1 / (0.13 * 0.13)));
    const bulge = pow(saturate(d.dot(vec3(GALACTIC_CENTRE.x, GALACTIC_CENTRE.y, GALACTIC_CENTRE.z)).mul(0.5).add(0.5)), 3.0);
    const clouds = mx_fractal_noise_float(d.mul(9.0), 4, 2.3, 0.6, 1.0).mul(0.5).add(0.5);
    const clumps = smoothstep(0.3, 0.9, clouds);
    const riftOffset = plane.sub(0.025).add(mx_noise_float(d.mul(11.0)).mul(0.035));
    const rift = float(1).sub(
      exp(riftOffset.mul(riftOffset).mul(-1 / (0.035 * 0.035)))
        .mul(smoothstep(0.3, 0.65, mx_noise_float(d.mul(4.0).add(vec3(3.1, 0, 1.7))).mul(0.5).add(0.5)).mul(0.6).add(0.3)),
    );

    // Unresolved stars: the band's grain. A second, much finer star grid with
    // no per-cell centre — each cell is simply lit or not — at a density that
    // follows the band. This is what keeps the band reading as *stars* rather
    // than as a cloud; a smooth glow alone is indistinguishable from the
    // moonlit cumulus drawn over it.
    const fineCell = floor(d.mul(460.0));
    const fine = hash13(fineCell);
    const fineLit = smoothstep(float(1).sub(bandCore.mul(clumps).mul(0.22)), 1.0, fine);
    const band = bandCore.mul(clumps.mul(0.8).add(0.2)).mul(rift).mul(bulge.mul(1.6).add(0.4));
    const galaxyColor = mix(vec3(0.55, 0.64, 0.9), vec3(0.95, 0.86, 0.72), bulge);

    const present = smoothstep(float(0.925).sub(bandCore.mul(0.04)), 0.985, r0);
    const core = pow(saturate(float(1).sub(dist.mul(5.0))), 6.0);
    const twinkle = sin(this.uTime.mul(2.6).add(r0.mul(61.0))).mul(0.3).add(0.7);
    const tint = mix(vec3(0.72, 0.81, 1.0), vec3(1.0, 0.88, 0.74), r1);
    const magnitude = r2.mul(0.85).add(0.25);

    const horizonFade = smoothstep(-0.03, 0.2, d.y);
    const stars = tint.mul(core.mul(present).mul(twinkle).mul(magnitude).mul(11.0).mul(horizonFade));
    const galaxy = galaxyColor.mul(
      band.mul(0.18).add(fineLit.mul(rift).mul(0.9)).mul(horizonFade),
    );

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
    // The 22-degree halo: moonlight refracted through hexagonal ice in high
    // cloud. A faint ring with a reddish inner edge and a bluer outer one —
    // the dispersion is what makes it read as an optical effect rather than
    // a drawn circle. Kept faint: it is a thing you notice, not a thing that
    // frames the moon.
    const ringInner = cosMoon.sub(0.9285).mul(1 / 0.0045);
    const ringOuter = cosMoon.sub(0.9225).mul(1 / 0.006);
    const ring = vec3(1.0, 0.82, 0.7)
      .mul(exp(ringInner.mul(ringInner).negate()))
      .add(vec3(0.6, 0.72, 1.0).mul(exp(ringOuter.mul(ringOuter).negate())))
      .mul(0.012);
    // The halo should not wrap under the horizon.
    const aboveHorizon = smoothstep(-0.08, 0.05, d.y);

    // The galaxy and the faintest stars wash out near a bright moon.
    const moonWash = float(1).sub(pow(halo, 12.0).mul(0.85));
    // Stars, galaxy and moon go behind an overcast the way the sun's disc does
    // (`uNightClear`); only the night sky's own gradient stays.
    return base.add(
      stars
        .add(galaxy.mul(moonWash))
        .add(moonBody.add(glow).add(ring).mul(aboveHorizon))
        .mul(this.uNightClear),
    );
  }
}

const WHITE = /*@__PURE__*/ new THREE.Color(1, 1, 1);
/**
 * Orientation of the galactic plane: its pole, and the direction of the bulge
 * within it. The band rises out of the sea with its bright bulge on the
 * horizon *opposite* the moonlit preset's moon — where the moon's glare does
 * not wash it out — and climbs steeply across the sky toward the zenith, so
 * a viewer who turns their back on the moon is looking straight at it.
 */
const GALACTIC_POLE = /*@__PURE__*/ new THREE.Vector3(-0.432, 0.362, -0.826).normalize();
const GALACTIC_CENTRE = /*@__PURE__*/ new THREE.Vector3(-0.857, 0.12, 0.501)
  .projectOnPlane(GALACTIC_POLE)
  .normalize();
const NIGHT_SKY_TINT = /*@__PURE__*/ new THREE.Color(0.12, 0.2, 0.42);
/**
 * The night dome's own glow, before `uNightRadiance` (0.168 at full night), at
 * the horizon and overhead. Shared by the shader and `skyRadiance`.
 *
 * A deep teal-navy, the colour a night at sea is painted in, rather than
 * black: at a fifth of these values the sky rendered near black while the sea,
 * which reflects the fill tint, sat at mid blue — a dark sky over a bright sea
 * meeting in a hard step. The horizon is about three times the zenith, a
 * soft lighter band and not a line.
 */
const NIGHT_HORIZON = /*@__PURE__*/ new THREE.Color(0.083, 0.19, 0.286);
const NIGHT_ZENITH = /*@__PURE__*/ new THREE.Color(0.024, 0.054, 0.1);
/** Colour moonlight fills shadows with — cool, and never quite neutral. */
const MOON_FILL_COLOR = /*@__PURE__*/ new THREE.Color(0.42, 0.55, 0.86);
/** Light coming back up off the sea at night. */
const MOON_GROUND_COLOR = /*@__PURE__*/ new THREE.Color(0.08, 0.13, 0.22);
const _skyTint = /*@__PURE__*/ new THREE.Color();
const _overcastFill = /*@__PURE__*/ new THREE.Color();

/** Cheap hash of a lattice cell to [0,1). Deterministic across both backends. */
function hash13(p: any): any {
  const q = p.mul(vec3(127.1, 311.7, 74.7));
  return fract(sin(q.x.add(q.y).add(q.z)).mul(43758.5453123));
}

function normalizeResolution(size: number): number {
  return Number.isFinite(size) ? Math.max(1, Math.round(size)) : ENV_SIZE;
}
