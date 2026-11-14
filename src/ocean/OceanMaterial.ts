import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  cameraPosition,
  float,
  linearDepth,
  log2,
  mix,
  normalize,
  positionLocal,
  positionWorld,
  dFdx,
  dFdy,
  screenUV,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
  viewportSafeUV,
  viewportSharedTexture,
  varyingProperty,
} from 'three/tsl';
import { smoothstepDownClamped } from '../core/tslMath';
import { FOOTPRINT_SPACINGS } from './meshSampling';
import {
  breaker,
  foamCells,
  setSurfState,
  surfPhaseGradient,
  surfReachDepth,
  surfUniforms,
  swashSheetHeight,
} from './Surf';

export interface WaterAppearance {
  /** Colour of light that survives deep transmission — the "body" of the water. */
  deepColor: THREE.Color;
  /** Colour in shallow water where the floor is close. */
  shallowColor: THREE.Color;
  /** Colour of light scattered back out of wave crests. */
  scatterColor: THREE.Color;
  foamColor: THREE.Color;
  /** Beer–Lambert extinction per metre, per channel. Higher = murkier. */
  extinction: THREE.Vector3;
  /** Strength of the subsurface glow on backlit crests. */
  scatterStrength: number;
  /** View-independent translucency of thin crest water, 0 disables. */
  crestGlow: number;
  /** Base roughness of the microfacet surface. */
  roughness: number;
  /** Jacobian value at which foam starts to appear. Lower = less foam. */
  foamThreshold: number;
  /** Width of the fold range over which foam ramps from none to full. */
  foamSoftness: number;
}

export const DEFAULT_APPEARANCE: WaterAppearance = {
  deepColor: new THREE.Color(0x03202f),
  shallowColor: new THREE.Color(0x168f92),
  scatterColor: new THREE.Color(0x2fae9c),
  foamColor: new THREE.Color(0xeff8ff),
  extinction: new THREE.Vector3(0.34, 0.11, 0.07),
  scatterStrength: 1.0,
  crestGlow: 0.55,
  roughness: 0.075,
  // Deeper folding required than the naive "J < 1" test: the swell cascade marks
  // broad areas as mildly folded, and treating all of that as whitecap covers a
  // fifth of the sea in white.
  foamThreshold: 0.24,
  foamSoftness: 0.55,
};

/**
 * Everything about the cascade set the surface needs, as one object.
 *
 * Grouped because it is supplied twice — once when the graph is built and again
 * by `setCascades` after a tier change rebuilds the simulation — and the two
 * sites silently disagreeing is a class of bug this file has already had once.
 * `OceanSimulation.fields` produces it, so neither site assembles it by hand.
 */
export interface CascadeFields {
  displacementTextures: THREE.Texture[];
  derivativeTextures: THREE.Texture[];
  tileSizes: number[];
  /** Longest wavelength each cascade carries, metres. See `cascadeReach`. */
  maxWavelengths: number[];
  /**
   * Texels per side of each cascade's output.
   *
   * With `tileSizes` this gives metres per texel, which is what the vertex
   * stage needs to turn a vertex spacing into a mip level.
   */
  resolution: number;
}

export interface OceanMaterialInputs extends CascadeFields {
  /**
   * Optional: a node returning the seafloor's depth below y=0 at a world
   * position. Supplying it switches the transmission term from a view-angle
   * approximation to a real water-column thickness, which is what produces the
   * turquoise-over-sand shallows and the hard read of a drop-off.
   */
  floorDepthNode?: ((worldPosition: any) => any) | null;
  /**
   * World-anchored foam accumulation buffer, from `physics/Wake`.
   *
   * `texture` is the resolved output target, whose reference is stable for the
   * lifetime of the buffer, so it is safe to bind once here at build time. The
   * buffer's *centre* moves every frame and is pushed in through
   * `setFoamCenter`; `extent` is the world size of the square it covers and is
   * fixed, so it is baked in as a constant rather than carried as a uniform.
   */
  foam?: {
    texture: THREE.Texture;
    extent: number;
    /**
     * Texels per side, so the surface can take finite differences of the
     * elevation channel at the buffer's own sampling rate rather than at some
     * guessed world distance. Sampling closer than a texel measures the bilinear
     * filter; sampling much wider measures a smoothed version of the wake and
     * loses the crest lines the whole channel exists for.
     */
    resolution: number;
  } | null;
  /**
   * Planar reflection texture node, from `ocean/Reflections`.
   *
   * Omitted rather than mixed out where it is not wanted. Whether the reflection
   * exists is a *backend* decision, known once at startup and never revisited,
   * so it belongs in the graph's construction — leaving the node in and weighting
   * it to zero would still pay for the second view of the scene every frame.
   * Per-tier strength, which does change at runtime, is a uniform.
   */
  reflectionNode?: unknown | null;
  /**
   * Screen-space reflection, as a function of shading point, normal and the
   * colour to fall back to where the trace misses.
   *
   * Supplied as a callback rather than a node because it has to be *called*
   * inside the fragment function — it emits control flow and reads per-fragment
   * variables, so it cannot be built ahead of time like the planar texture can.
   */
  ssrNode?: ((worldPosition: any, worldNormal: any, fallback: any) => any) | null;
  /**
   * Fraction of sunlight reaching a world point through the cloud deck, 0..1.
   *
   * From `sky/Clouds`, so the shade lands where the cloud that casts it is drawn.
   * Omitted rather than defaulted to 1: a scene with no cloud layer should not
   * pay a noise lookup per fragment to be told the sun is out.
   */
  cloudShadowNode?: ((worldPosition: any) => any) | null;
}

/**
 * Physically motivated water surface.
 *
 * Layers, in the order they combine:
 *   transmission  Beer–Lambert absorption along the view path through water
 *   scattering    a wrapped term that brightens crests when the sun is behind
 *                 them — what gives real water its jade glow
 *   reflection    sky through a Schlick Fresnel term
 *   specular      GGX highlight for the sun disc
 *   foam          Jacobian-driven whitecaps composited on top
 */
/**
 * Cascade slots the shader is always built with.
 *
 * Fixed rather than derived from the active tier, so a tier change re-points
 * texture bindings instead of recompiling the graph. Matches `CASCADES` in
 * `Spectrum`; a tier asking for more than this would silently lose bands, so the
 * two are asserted equal at construction.
 */
const MAX_CASCADES = 3;

/**
 * Slope scale for the grazing-angle reflection fade, dimensionless.
 *
 * Motivated by Cox & Munk's sun-glitter measurements, which give a wind-driven
 * RMS surface slope near 0.28 at 15 m/s — but **this is not a Smith masking
 * term** and should not be read as one. A real Smith function goes to zero at
 * grazing incidence and depends on the chosen NDF; the fade below bottoms out at
 * 0.45, and it multiplies the Fresnel *blend* rather than acting as the geometry
 * factor of the microfacet BRDF. It is an art-directed approximation of what
 * masking would do, with the right shape and the wrong limit.
 *
 * The value is not Cox & Munk's either. At 0.28 the fade removes nearly the
 * whole horizon reflection, which overshoots: the derivative fields still carry
 * real slope at mid distances, so the correction would be applied twice there.
 * 0.16 leaves the far field about half its reflection, which stops it reading as
 * a white sheet, and barely touches the near field where the normals are genuine.
 *
 * Doing this properly means a slope-variance-driven roughness feeding a
 * prefiltered reflection probe, which this renderer does not have.
 */
const GRAZING_SLOPE_SIGMA = 0.16;

/**
 * The undisplaced world XZ a fragment's surface point was generated from.
 *
 * A varying rather than a recomputation: the vertex stage is the only place that
 * knows it, because displacement is applied after.
 */
const surfaceUv = /*@__PURE__*/ varyingProperty('vec2', 'oceanSurfaceUv');

/**
 * Refractive index of water relative to air, for the underside.
 *
 * 1.333 is the sodium-D value at 20 degrees and the number every optics text
 * quotes. What matters here is the critical angle it implies: `asin(1/1.333)` is
 * 48.6 degrees, so the whole sky arrives inside a cone of that half-angle and
 * everything outside it is total internal reflection.
 */
const WATER_IOR = 1.333;

/**
 * cos(48.6 degrees) — the edge of Snell's window.
 *
 * Kept for reference. The window's sky lookup no longer uses it: the air-side
 * angle comes from Snell directly, which is not linear in the incidence cosine.
 */
export const COS_CRITICAL_ANGLE = Math.sqrt(1 - 1 / (WATER_IOR * WATER_IOR));

/**
 * Screen-space variance scale for specular antialiasing.
 *
 * Kaplanyan's paper derives 0.5 for a box filter over the pixel; a real
 * rasteriser's derivatives are finite differences between neighbouring pixels,
 * which already spans that footprint, so the published value is used as-is.
 */
const SPECULAR_AA_SCREEN_SPACE_VARIANCE = 0.5;

/**
 * Ceiling on how much the filter may widen `alpha^2`.
 *
 * The usual published figure is 0.18, and it is wildly wrong for the *geometric*
 * variant on this material. Water's base `alpha^2` is 3.2e-5, so 0.18 is five
 * thousand times it — that does not filter the lobe, it replaces it, and the
 * whole up-sun half of the frame turns into one white sheet.
 *
 * 0.004 caps the filtered roughness near 0.08, about ten times the base: enough
 * to integrate away the single-pixel spikes on a wave face, nowhere near enough
 * to lose the glitter's structure.
 *
 * It is now a per-axis ceiling. The distribution being filtered is genuinely
 * anisotropic — the sea varies about `uSlopeAnisotropy` times as much along the
 * wind as across it — and the filter resolves the screen-space covariance onto
 * the tangent frame before applying this, so each axis is capped against its own
 * measurement rather than both against a shared scalar. See the filtering block
 * in the fragment stage.
 */
const SPECULAR_AA_VARIANCE_CEIL = 0.004;

/**
 * Where the refracted backdrop starts and finishes yielding to the body colour,
 * metres.
 *
 * `waves.png` has a band of khaki-grey scratches across the water 200 to 400 m
 * out, and it is the seafloor. Over the origin plateau the water is 17 m deep, so
 * `absorption` stays high and the refracted sand shows through at close to full
 * contrast — carrying the sand normal map's own aliasing with it, at a distance
 * where that map is far below its Nyquist rate.
 *
 * The argument for fading it is the one the cascades' shading fade already makes:
 * a detail the pixel cannot resolve is not detail. What replaces it is not
 * nothing — it is `inscatter`, the analytic depth-graded body colour, which is
 * what the transmission through that column integrates to anyway and is smooth
 * by construction.
 */
const REFRACTION_FADE_NEAR = 140;
const REFRACTION_FADE_FAR = 460;

/**
 * Baseline over which the seabed gradient is measured for the surf's bearing,
 * metres.
 *
 * Twelve, and both bounds are real. Narrower and the difference is dominated by
 * the heightfield's own metre-scale relief rather than by the beach's slope, so
 * the surf line would wander; wider and it stops resolving a headland, so the
 * sets would run at the same bearing round a corner they should be turning.
 */
/**
 * Where the foam's sub-metre breakup noise fades out, metres.
 *
 * Its octaves run from 0.7 m down to about 8 cm. At 400 m a pixel covers roughly
 * a third of a metre along the view and far more across it, so the whole field
 * is below the sampling rate; what it adds past there is high-frequency energy
 * with no feature behind it.
 */
const FOAM_BREAKUP_NEAR = 180;
const FOAM_BREAKUP_FAR = 520;

/**
 * How far the foam breakup noise is stretched along the wind, as a ratio.
 *
 * Whitecap foam is not an isotropic blob. The wind that breaks a crest keeps
 * blowing over the raft it leaves, so a real whitecap is a streak lying along
 * the wind — the standard aerial photographs of a Beaufort 6 sea are almost all
 * streak and very little patch. Sampling the breakup field in a frame stretched
 * along the wind axis says that for the cost of one dot product, and it is the
 * difference between foam that looks blown and foam that looks sprinkled.
 *
 * 2.6 rather than something larger because past about 3 the streaks start to
 * read as scratches: the noise runs out of features along the stretched axis and
 * what is left is a comb. It also has to stay clear of `uSlopeAnisotropy`, which
 * already stretches the *specular* along the same axis — two anisotropies at the
 * same aspect ratio on the same surface stop reading as wind and start reading
 * as a texture bug.
 *
 * Stretching lowers the along-wind frequency rather than raising anything, so
 * this cannot cost the far-field jitter budget the surrounding comments defend.
 */
const FOAM_WIND_STRETCH = 2.6;

const SHORE_GRADIENT_EPSILON = 12;

/**
 * Ground distance from the eye, metres, inside which the vertex stage lifts the
 * water into the breakers and the swash sheet.
 *
 * Bounded by the mesh, not by cost: the sheet is a few centimetres thick and
 * follows sand with a 1:10 slope, so it needs vertices well under a metre apart
 * to stay on the beach rather than cutting through it, and past a couple of
 * hundred metres the radial grid is metres apart. Past here the ground shader's
 * swash hook carries the effect on its own, and at that range it is a pixel
 * wide anyway.
 */
const SURF_VERTEX_REACH = 220;

/**
 * Fraction of the FFT field's displacement and slope left in the shallowest
 * surf. The open-ocean spectrum knows nothing about the bottom; inside the
 * break the sea is the breakers and a chop on top of them, and leaving the
 * full field on puts metre-high swell crests across the beach.
 */
const SURF_FFT_FLOOR = 0.25;

/**
 * Water thickness a backlit crest and a wave body present to the sun, metres.
 *
 * The crest figure is the one that matters. A wave tip is a thin sheet — light
 * crosses it in centimetres, so almost everything survives and the water glows
 * jade. `THICK_BODY_METRES` is what the face below it presents, far enough that
 * red is gone and most of the green with it, which is what makes the glow stop
 * at the crest instead of washing the whole wave.
 *
 * Both are relative to the local sea state rather than absolute, because a 0.4 m
 * chop and a 6 m swell do not have the same crest. See `uWaveScale`.
 */
const THIN_CREST_METRES = 0.22;
const THICK_BODY_METRES = 4.2;

/**
 * Henyey–Greenstein asymmetry for scattering in sea water.
 *
 * Measured values for ocean water are famously forward-peaked — the Petzold
 * volume-scattering measurements give an average cosine around 0.9 for the
 * particulate component. 0.72 is deliberately below that: the honest figure puts
 * almost all the energy inside a few degrees of the sun, which on a surface this
 * broken produces a glow that snaps on and off as the crest orientation crosses
 * the lobe. Widening it trades a little physics for a term that survives the
 * wave field it is being evaluated on.
 */
const SCATTER_ANISOTROPY = 0.72;

/**
 * Peak value of that phase function, which the term is divided by.
 *
 * `(1 - g^2) / (4pi (1 - g)^3)` = 1.746 at g = 0.72. Normalising by it keeps the
 * lobe peaking at one, which is what the presets' authored `scatterStrength`
 * values were tuned against when the term was a clamped dot product. Without it
 * every preset would have needed retuning to say the same thing.
 */
const SCATTER_NORMALISE = 1 / 1.746;

/**
 * Wrap on the away-from-sun term.
 *
 * The shading normal at a crest is close to horizontal, so a hard `dot > 0` puts
 * the on/off boundary exactly where the normal is noisiest and the glow crawls
 * along the crest line as ripples cross it. Wrapping moves the boundary off the
 * horizon and softens it.
 */
const SCATTER_WRAP = 0.45;

export class OceanMaterial {
  readonly material: THREE.MeshBasicNodeMaterial;

  // Cascade bindings, re-pointed by `setCascades` rather than rebuilt.
  private readonly displacementNodes: any[] = [];
  private readonly derivativeNodes: any[] = [];
  private readonly uTileSizes: any[] = [];
  private readonly uCascadeWeights: any[] = [];
  /** Metres per texel of each cascade's field. See `setCascades`. */
  private readonly uTexelMetres: any[] = [];
  /**
   * Longest wavelength each cascade carries, metres.
   *
   * The band edge past which the low-pass has nothing left to pass, so it is
   * where the cascade stops displacing geometry. See `cascadeReach`.
   */
  private readonly uMaxWavelengths: any[] = [];
  /**
   * Metres of vertex spacing per metre of ground distance, from the mesh.
   *
   * Zero is the safe default rather than a plausible one: it makes the level of
   * detail zero everywhere, so a caller that forgets `setVertexSpacing` gets the
   * behaviour this material had before the band-limit existed instead of a sea
   * silently flattened into a sheet.
   */
  private readonly uVertexSpacing = uniform(0);

  // --- appearance ----------------------------------------------------------
  private readonly uDeepColor = uniform(new THREE.Color(DEFAULT_APPEARANCE.deepColor));
  private readonly uShallowColor = uniform(new THREE.Color(DEFAULT_APPEARANCE.shallowColor));
  private readonly uScatterColor = uniform(new THREE.Color(DEFAULT_APPEARANCE.scatterColor));
  private readonly uFoamColor = uniform(new THREE.Color(DEFAULT_APPEARANCE.foamColor));
  private readonly uExtinction = uniform(new THREE.Vector3().copy(DEFAULT_APPEARANCE.extinction));
  private readonly uScatterStrength = uniform(DEFAULT_APPEARANCE.scatterStrength);
  /** Strength of the view-independent crest translucency. */
  private readonly uCrestGlow = uniform(DEFAULT_APPEARANCE.crestGlow);
  private readonly uRoughness = uniform(DEFAULT_APPEARANCE.roughness);
  /**
   * Unit wind vector in world XZ, and the slope-variance ratio across it.
   *
   * Sun glitter on real water is not a round highlight. Cox & Munk measured the
   * slope distribution from aerial photographs of the glitter itself and found it
   * *anisotropic*: the variance along the wind is roughly twice the variance
   * across it, because the waves are longer-crested than they are steep. A
   * highlight rendered through that distribution stretches perpendicular to the
   * crests — which is the elongated track everyone recognises, and which an
   * isotropic roughness cannot produce at any value.
   */
  private readonly uWindAxis = uniform(new THREE.Vector2(1, 0));
  private readonly uSlopeAnisotropy = uniform(1.9);
  private readonly uFoamThreshold = uniform(DEFAULT_APPEARANCE.foamThreshold);
  private readonly uFoamSoftness = uniform(DEFAULT_APPEARANCE.foamSoftness);

  // --- environment ---------------------------------------------------------
  private readonly uSunDirection = uniform(new THREE.Vector3(0.4, 0.5, 0.3).normalize());
  private readonly uSunColor = uniform(new THREE.Color(1.0, 0.94, 0.84));
  private readonly uSunIntensity = uniform(4.0);
  /**
   * How lit the scene is, 0..1, independent of the key light's colour.
   *
   * Terms that represent transmitted or scattered daylight — as opposed to
   * authored body colour — are scaled by this so they go out after dark.
   */
  private readonly uLightLevel = uniform(1);
  /** Significant wave height, metres. Sets the scale a crest is measured against. */
  private readonly uWaveScale = uniform(3.5);
  private readonly uSkyColor = uniform(new THREE.Color(0x5793d0));
  private readonly uHorizonColor = uniform(new THREE.Color(0xbdd6ec));
  private readonly uFogColor = uniform(new THREE.Color(0xb9d2e8));
  private readonly uFogDensity = uniform(0.00016);
  private readonly uDisplacementScale = uniform(1);

  // The ocean mesh is translated each frame to follow the camera. `positionLocal`
  // is pre-transform, so the vertex stage needs that translation to reconstruct a
  // stable world-space sample coordinate for the wave field.
  private readonly uOffsetX = uniform(0);
  private readonly uOffsetZ = uniform(0);

  /** Metres of screen-space offset applied to the refracted sample, at unit distance. */
  private readonly uRefractionStrength = uniform(0.22);
  /**
   * How much of the transmitted colour comes from the real scene behind the
   * surface, as opposed to the analytic body colour. 0 restores the pre-refraction
   * look exactly, which is what the WebGL2 fallback and Low tier use.
   */
  private readonly uRefractionAmount = uniform(1);
  /**
   * Camera far minus near, in metres.
   *
   * `linearDepth` normalises to the 0..1 range the camera spans, so a depth
   * difference has to be scaled by that span to become a thickness. Pushed as a
   * uniform rather than read from a camera node because this material is also
   * built for a WebGL2 path where the two must agree exactly.
   */
  private readonly uDepthRange = uniform(40000);
  /**
   * Camera forward, world space.
   *
   * Depth-buffer values are distances along this axis, and light travels along
   * the pixel's own ray; converting between the two needs the angle between them.
   */
  private readonly uCameraForward = uniform(new THREE.Vector3(0, 0, -1));

  /** Rain rate, 0..1. Drives how many lattice cells are producing impacts. */
  private readonly uRainIntensity = uniform(0);
  /**
   * Strength of the impact slope perturbation.
   *
   * Lower than it first looks like it should be. With the lattice corrected so
   * that every cell rains at full intensity, 0.32 turned the near surface into a
   * hammered-metal texture — the rings were unmistakably there, which was the
   * point, but they read as a material rather than as weather. Rain dimples a
   * surface; it does not emboss it.
   */
  private readonly uRainSlope = uniform(0.16);
  /** Rain clock, seconds. Separate from the wave clock so it can be frozen. */
  private readonly uRainTime = uniform(0);

  /** How much of the planar reflection reaches the surface, 0..1. */
  /**
   * 1 when the eye is below the surface, 0 above. Decides which side of the
   * interface is being shaded — see the topside normal fix below for why the
   * shading normal must not decide it.
   */
  private readonly uEyeSubmerged = uniform(0);
  private readonly uReflectionAmount = uniform(1);
  /** Screen-space offset applied to the reflection lookup, at unit distance. */
  private readonly uReflectionDistortion = uniform(0.09);

  /** World centre of the foam accumulation buffer. See `setFoamCenter`. */
  private readonly uFoamCenter = uniform(new THREE.Vector2());
  /** How strongly accumulated foam reads against the surface, 0..1. */
  private readonly uFoamStrength = uniform(1);
  /**
   * Multiplier on the wake's elevation channel, 0..1.
   *
   * A tier knob rather than an art knob: the wake displacement costs two extra
   * texture taps in the fragment stage and one in the vertex stage, and the Low
   * tier turns it off. Zero here does genuinely remove the work, because the
   * displacement branch is behind a uniform compare.
   */
  private readonly uWakeDisplacement = uniform(1);

  /**
   * The wake buffer, bound once and shared by both stages.
   *
   * One node rather than two `texture()` calls, because the vertex stage
   * displaces by the elevation channel and the fragment stage reads foam and the
   * elevation *gradient* from the same texels. Two independent bindings of the
   * same texture would compile to two samplers and — worse — make it possible to
   * point one at a new target and forget the other.
   */
  private wakeNode: any = null;

  constructor(inputs: OceanMaterialInputs) {
    this.material = new THREE.MeshBasicNodeMaterial();
    this.material.side = THREE.DoubleSide; // visible from below when submerged
    this.material.name = 'ocean-water';

    // Transparent so the surface is drawn *after* the opaque scene and can read
    // it as a backdrop — that read is the whole basis of refraction, and before
    // the opaque pass there is simply nothing behind the water to sample.
    //
    // Depth is still written, unusually for a transparent material, because
    // everything downstream depends on the water having a position: the
    // underwater pass linearises this depth to bound its shaft march, and without
    // it the shafts would run straight through the surface to the sky.
    this.material.transparent = true;
    this.material.depthWrite = true;

    this.build(inputs);
  }

  // ---------------------------------------------------------------- accessors

  /**
   * The key light: whichever body is actually casting, at its real strength.
   *
   * `lightLevel` is separate from `intensity` because the two do different jobs.
   * `intensity` scales the specular highlight, which is a reflection of the light
   * source and so follows its brightness directly. `lightLevel` scales terms that
   * represent daylight having passed *through* the water, which have to go out
   * after dark but should not track a bright low sun the way a highlight does.
   */
  setSun(direction: THREE.Vector3, color: THREE.Color, intensity: number, lightLevel = 1): void {
    (this.uSunDirection.value as THREE.Vector3).copy(direction).normalize();
    (this.uSunColor.value as THREE.Color).copy(color);
    this.uSunIntensity.value = intensity;
    this.uLightLevel.value = Math.max(0, Math.min(1, lightLevel));
  }

  setSky(sky: THREE.Color, horizon: THREE.Color, fog: THREE.Color, fogDensity: number): void {
    (this.uSkyColor.value as THREE.Color).copy(sky);
    (this.uHorizonColor.value as THREE.Color).copy(horizon);
    (this.uFogColor.value as THREE.Color).copy(fog);
    this.uFogDensity.value = fogDensity;
  }

  setAppearance(a: Partial<WaterAppearance>): void {
    if (a.deepColor) (this.uDeepColor.value as THREE.Color).copy(a.deepColor);
    if (a.shallowColor) (this.uShallowColor.value as THREE.Color).copy(a.shallowColor);
    if (a.scatterColor) (this.uScatterColor.value as THREE.Color).copy(a.scatterColor);
    if (a.foamColor) (this.uFoamColor.value as THREE.Color).copy(a.foamColor);
    if (a.extinction) (this.uExtinction.value as THREE.Vector3).copy(a.extinction);
    if (a.scatterStrength !== undefined) this.uScatterStrength.value = a.scatterStrength;
    if (a.crestGlow !== undefined) this.uCrestGlow.value = a.crestGlow;
    if (a.roughness !== undefined) this.uRoughness.value = a.roughness;
    if (a.foamThreshold !== undefined) this.uFoamThreshold.value = a.foamThreshold;
    if (a.foamSoftness !== undefined) this.uFoamSoftness.value = a.foamSoftness;
  }

  /**
   * Wind bearing in radians and the along/across slope-variance ratio.
   *
   * The ratio is Cox & Munk's: `sigma_u^2 = 0.00316 U` upwind against
   * `sigma_c^2 = 0.003 + 0.00192 U` crosswind, so at 15 m/s the two are 0.047 and
   * 0.032 — a ratio near 1.5 that rises with wind. Passing it here rather than
   * hard-coding it is what lets a glassy dusk have a round highlight and a gale
   * have a long one.
   */
  /**
   * Drives the shore break: a clock for the travelling sets, and a master.
   *
   * Separate from `setRain`, which also carries a clock, because the two are
   * unrelated media and sharing a uniform between them is how a rain slider ends
   * up freezing the surf.
   */
  setSurf(phase: number, strength = 1, peakWavelength = 47, shoreNear = true): void {
    // Swell arriving at a beach is longer-period than the local wind sea's
    // peak: the short components are the ones refraction and the reef strip
    // out first. Deep-water dispersion for the peak, plus fifteen percent.
    const period = Math.sqrt((2 * Math.PI * Math.max(4, peakWavelength)) / 9.81) * 1.15;
    // What breaks on a beach is the swell that reaches it — nearer 1.5 m than
    // the open sea's significant height. A fraction of the sea state rather
    // than the sea state, as the previous shore break had it.
    const height = Math.min(3.2, (this.uWaveScale.value as number) * 0.26 + 0.55);
    setSurfState(phase, period, height, strength, shoreNear);
  }

  setWind(bearingRadians: number, speed: number): void {
    const axis = this.uWindAxis.value as THREE.Vector2;
    axis.set(Math.cos(bearingRadians), Math.sin(bearingRadians));
    const upwind = 0.00316 * Math.max(0, speed);
    const crosswind = 0.003 + 0.00192 * Math.max(0, speed);
    this.uSlopeAnisotropy.value = Math.min(4, Math.max(1, upwind / Math.max(1e-4, crosswind)));

    // Significant wave height for a fully developed sea, Pierson–Moskowitz:
    // Hs = 0.0246 U^2. It is what tells the scattering term where a crest *is* —
    // the term reads a wave's height as a fraction of the sea state, so with a
    // fixed scale a calm day would have no crests to glow and a gale would have
    // nothing but. The floor keeps a glassy preset from dividing by nearly zero.
    this.uWaveScale.value = Math.max(0.35, 0.0246 * Math.max(0, speed) ** 2);
  }

  setDisplacementScale(value: number): void {
    this.uDisplacementScale.value = value;
  }

  /**
   * Per-tier refraction policy.
   *
   * `amount` 0 drops the surface back to the analytic body colour without
   * recompiling anything. Note what that does and does not buy: the backdrop and
   * depth reads are still executed — they are unconditional in the graph — and
   * only their *contribution* is mixed out. So this is a visual policy, not a
   * cost one, and the WebGL2 path takes it to get a coherent simpler image
   * rather than to avoid the reads.
   */
  setRefraction(amount: number, strength = 0.22): void {
    this.uRefractionAmount.value = Math.max(0, Math.min(1, amount));
    this.uRefractionStrength.value = Math.max(0, strength);
  }

  /** Camera far minus near, in metres. See `uDepthRange`. */
  setDepthRange(metres: number): void {
    this.uDepthRange.value = Math.max(1, metres);
  }

  /** Camera forward direction, world space. See `uCameraForward`. */
  setCameraForward(direction: THREE.Vector3): void {
    (this.uCameraForward.value as THREE.Vector3).copy(direction).normalize();
  }

  /**
   * Rain striking the surface.
   *
   * `time` is passed rather than integrated internally so the impacts share the
   * simulation clock — which is what lets a deterministic capture reproduce the
   * same rings, and what makes the rain stop when the world is paused.
   */
  setRain(intensity: number, time: number, slope = 0.32): void {
    this.uRainIntensity.value = Math.max(0, Math.min(1, intensity));
    this.uRainTime.value = time;
    this.uRainSlope.value = Math.max(0, slope);
  }

  /** Per-tier reflection strength. No effect when built without a reflection. */
  /** Whether the eye is under water; selects topside or underside shading. */
  setEyeSubmerged(submerged: boolean): void {
    this.uEyeSubmerged.value = submerged ? 1 : 0;
  }

  setReflection(amount: number, distortion = 0.09): void {
    this.uReflectionAmount.value = Math.max(0, Math.min(1, amount));
    this.uReflectionDistortion.value = Math.max(0, distortion);
  }

  /** Must be called with the ocean mesh's world translation every frame. */
  setWorldOffset(x: number, z: number): void {
    this.uOffsetX.value = x;
    this.uOffsetZ.value = z;
  }

  /**
   * Must be called with the foam buffer's world centre every frame.
   *
   * The buffer is anchored in the world and scrolls under a moving centre, so
   * the surface can only find the right texel if it is told where the buffer
   * currently sits. Getting this wrong does not fail loudly — the wake simply
   * slides around relative to the hull, which is the one thing a wake must
   * never do.
   */
  setFoamCenter(x: number, z: number): void {
    const centre = this.uFoamCenter.value as THREE.Vector2;
    centre.x = x;
    centre.y = z;
  }

  setFoamStrength(value: number): void {
    this.uFoamStrength.value = value;
  }

  /** Tier knob for the wake's surface deformation. 0 skips the taps entirely. */
  setWakeDisplacement(value: number): void {
    this.uWakeDisplacement.value = Math.max(0, Math.min(1, value));
  }

  dispose(): void {
    this.material.dispose();
  }

  // ----------------------------------------------------------------- internals

  /**
   * Rebinds the wave field without rebuilding the shader.
   *
   * A quality change recreates the simulation's render targets, so the surface
   * has to be pointed at the new ones. It used to be reconstructed wholesale for
   * that, which was wrong three ways: the node graph recompiled mid-session,
   * every rebuild had to remember to re-supply every input (and one of them
   * silently didn't — see `floorDepthNode`), and once the surface began sampling
   * the framebuffer for refraction each rebuild leaked the backdrop texture that
   * came with it, measured at forty textures over eight tier changes.
   *
   * The graph is therefore built once for the maximum cascade count and the
   * texture nodes are re-pointed here. Cascades the current tier does not use are
   * bound to a live texture and weighted to zero — binding nothing is not an
   * option, since a sampler with no texture is a validation error even when its
   * result is multiplied away.
   */
  setCascades(fields: CascadeFields): void {
    const { displacementTextures, derivativeTextures, tileSizes, maxWavelengths } = fields;
    const active = Math.min(displacementTextures.length, MAX_CASCADES);
    for (let i = 0; i < MAX_CASCADES; i++) {
      const source = Math.min(i, active - 1);
      this.displacementNodes[i].value = displacementTextures[source];
      this.derivativeNodes[i].value = derivativeTextures[source];
      this.uTileSizes[i].value = tileSizes[source];
      this.uTexelMetres[i].value = tileSizes[source] / fields.resolution;
      this.uMaxWavelengths[i].value = maxWavelengths[source];
      this.uCascadeWeights[i].value = i < active ? 1 : 0;
    }
  }

  /**
   * Tells the surface how finely the mesh it is drawn on samples the world.
   *
   * Must be called whenever that mesh is built or rebuilt — a tier change does
   * both. Getting it wrong does not fail loudly: too small displaces geometry
   * by waves the mesh cannot resolve and the surface facets, too large flattens
   * the sea into a sheet.
   */
  setVertexSpacing(spacingPerMetre: number): void {
    this.uVertexSpacing.value = Math.max(0, spacingPerMetre);
  }

  private build(inputs: OceanMaterialInputs): void {
    const { displacementTextures, derivativeTextures, tileSizes } = inputs;
    const floorDepth = inputs.floorDepthNode ?? null;
    const foam = inputs.foam ?? null;
    const cloudShadow = inputs.cloudShadowNode ?? null;
    if (foam !== null) this.wakeNode = texture(foam.texture) as any;
    const planar = (inputs.reflectionNode ?? null) as any;
    const ssr = inputs.ssrNode ?? null;

    // The graph is always built for the maximum cascade count; `setCascades`
    // decides how many of them contribute.
    const cascadeCount = MAX_CASCADES;
    for (let i = 0; i < MAX_CASCADES; i++) {
      const source = Math.min(i, displacementTextures.length - 1);
      this.displacementNodes.push(texture(displacementTextures[source]) as any);
      this.derivativeNodes.push(texture(derivativeTextures[source]) as any);
      this.uTileSizes.push(uniform(tileSizes[source]));
      this.uTexelMetres.push(uniform(tileSizes[source] / inputs.resolution));
      this.uMaxWavelengths.push(uniform(inputs.maxWavelengths[source]));
      this.uCascadeWeights.push(uniform(i < displacementTextures.length ? 1 : 0));
    }

    // ------------------------------------------------------------- vertex stage
    this.material.positionNode = Fn(() => {
      const local = positionLocal.toVar();
      const worldXZ = vec2(local.x.add(this.uOffsetX), local.z.add(this.uOffsetZ)).toVar();

      // Handed to the fragment stage, and this is the fix for the surface's
      // shimmer rather than a convenience.
      //
      // The fragment used to sample the derivative fields at `positionWorld.xz` —
      // the *displaced* position. That is not where the vertex sampled them. The
      // choppy term moves a vertex horizontally by up to a metre, and everything
      // between vertices is a linear interpolation of displaced corners, so the
      // fragment's lookup lands somewhere the field never generated. The error is
      // whatever the mesh fails to resolve, which is why the measured jitter
      // tracked mesh density almost exactly and barely moved when SSR, the
      // reflection, the fog or the wake were switched off:
      //
      //   mean frame-to-frame |dL| over the water, 1/60 s apart, same camera
      //   medium  3.14   (2 cascades, 192 rings)
      //   high    9.06   (3 cascades, 288 rings)
      //   ultra   7.88   (3 cascades, 384 rings)
      //   max     5.81   (3 cascades, 512 rings)
      //
      // Carrying the undisplaced coordinate through as a varying makes the
      // fragment sample exactly where the vertex did, at every mesh density.
      surfaceUv.assign(worldXZ);

      // The mesh is centred on the camera, so local XZ length is the ground
      // distance from the viewer in metres.
      const groundDistance = vec2(local.x, local.z).length().toVar();

      const displacement = vec3(0).toVar();

      // Displace by exactly as much of the field as this mesh can carry, and
      // not one wave more.
      //
      // A vertex-stage texture read has no derivatives, so it compiles to LOD 0
      // and takes the sharpest mip however far apart the vertices are. That is
      // the whole of the faceting the sea-level shot showed: the ripple cascade
      // runs down to 5 cm wavelengths and was displacing geometry whose vertices
      // are 20 cm apart at five metres and 75 cm apart at twenty, so every
      // triangle landed on an unrelated phase and isolated vertices spiked into
      // pyramids.
      //
      // The mip is chosen so its footprint is two vertex spacings, which puts a
      // box filter's first null exactly on the mesh's Nyquist wavelength. See
      // `FOOTPRINT_SPACINGS`; `geometryLod` there is these lines in testable
      // form and is the specification if the two ever drift.
      //
      // The fixed-metre table this replaces had the right idea and could not act
      // on it, because absolute distances cannot know the tier's mesh density —
      // the same 18-55 m ripple ramp served a 128 ring mesh and a 512 ring one
      // whose spacings differ fourfold. The level knows, because spacing is what
      // it is computed from.
      //
      // **But the table's far end had to come back, and this is the second
      // version of this loop.** The first deleted the fade outright, on the
      // argument that a cascade puts itself out once the level runs past the
      // last mip and the sampler clamps to the 1x1 level — the mean of a field
      // whose DC amplitude is zero by construction. True, and it happens far
      // later than the reasoning assumed: the ripple cascade does not reach its
      // 1x1 level until 213 m, long after its longest 6 m wave stops being
      // resolvable, so in between it kept contributing sidelobe — 38% of that
      // wave at 55 m and -18% at 100 m, sign-inverted, on a mesh with no hope of
      // resolving it. `cascadeReach` cuts each cascade over the interval where
      // the footprint crosses its own longest wavelength, which is where the
      // band's best-surviving component goes from 64% to exactly nulled.
      //
      // The mip chains cost nothing extra. `makeOutputTarget` has always built
      // them — see its header on why the derivative field needs them — and until
      // now the displacement field's chain was generated every frame for a
      // consumer that only existed in the fragment stage.
      const footprint = groundDistance
        .mul(this.uVertexSpacing)
        .mul(FOOTPRINT_SPACINGS)
        .toVar();
      for (let i = 0; i < cascadeCount; i++) {
        const lod = log2(footprint.div(this.uTexelMetres[i]).max(1)).toVar();
        const reach = smoothstepDownClamped(
          footprint.div(this.uMaxWavelengths[i]),
          0.5,
          1,
        ).toVar();
        const sample = this.displacementNodes[i]
          .sample(worldXZ.div(this.uTileSizes[i]))
          .level(lod)
          .toVar();
        displacement.addAssign(sample.xyz.mul(reach).mul(this.uCascadeWeights[i]));
      }

      displacement.mulAssign(this.uDisplacementScale);

      // Wake displacement, in metres, from the world-anchored buffer.
      //
      // Deliberately *outside* `uDisplacementScale`: that scale is the sea-state
      // knob, and a preset that flattens the swell should not also flatten the
      // hole a ship pushes through it.
      //
      // Faded with distance from the camera on the same argument as the wave
      // cascades — the divergent arms are metre-scale, so past a few hundred
      // metres they are subpixel and displacing geometry by them only produces
      // the speckle that the geometry/shading LOD split exists to avoid. The
      // fragment stage keeps contributing them to the normal well past that.
      if (foam !== null) {
        If(this.uWakeDisplacement.greaterThan(0.001), () => {
          const wakeUv = worldXZ.sub(this.uFoamCenter).div(foam.extent).add(0.5).toVar();
          const edge = wakeUv.min(wakeUv.oneMinus()).toVar();
          const inside = edge.x.min(edge.y).smoothstep(0, 0.02).clamp(0, 1).toVar();
          const reach = smoothstepDownClamped(groundDistance, 220, 620);
          displacement.y.addAssign(
            this.wakeNode
              .sample(wakeUv)
              .g.mul(inside)
              .mul(reach)
              .mul(this.uWakeDisplacement),
          );
        });
      }

      // --- surf ----------------------------------------------------------------
      // Inside the surf zone the water is lifted into the breakers and, over the
      // sand, into the swash sheet — real geometry, so the waterline itself runs
      // up and down the beach with each wave instead of a foam decal sliding
      // over a fixed edge. See `ocean/Surf`. Two branches: a uniform one (is a
      // shore anywhere near the eye) and a per-vertex one (is this vertex in the
      // surf), so open water pays two compares and no heightfield.
      if (floorDepth !== null) {
        If(
          surfUniforms.active.greaterThan(0.5).and(groundDistance.lessThan(SURF_VERTEX_REACH)),
          () => {
            const depth = floorDepth(vec3(worldXZ.x, 0, worldXZ.y)).toVar();
            If(depth.lessThan(surfReachDepth()), () => {
              const fft = mix(
                float(SURF_FFT_FLOOR),
                float(1),
                depth.smoothstep(0.3, surfReachDepth().mul(0.7)),
              ).toVar();
              displacement.mulAssign(fft);
              const wave = breaker(depth, worldXZ);
              displacement.y.addAssign(wave.eta);
              // Faded out toward the edge of the vertex reach, where the mesh can
              // no longer hold a sheet this thin on a slope this shallow.
              const sheet = swashSheetHeight(depth, worldXZ).toVar();
              const hold = smoothstepDownClamped(groundDistance, SURF_VERTEX_REACH * 0.7, SURF_VERTEX_REACH);
              displacement.y.assign(mix(displacement.y, displacement.y.max(sheet), hold));
            });
          },
        );
      }

      return vec3(local.x.add(displacement.x), displacement.y, local.z.add(displacement.z));
    })();

    // ----------------------------------------------------------- fragment stage
    this.material.colorNode = Fn(() => {
      const worldPos = positionWorld.toVar();
      // The *undisplaced* sample coordinate — see the vertex stage. Wave fields
      // are read here; anything that genuinely wants the displaced point (the
      // depth column, the foam buffer, the crest bias) still uses `worldPos`.
      const fieldXZ = vec2(surfaceUv).toVar();
      const viewVector = cameraPosition.sub(worldPos).toVar();
      const viewDistance = viewVector.length().toVar();
      const viewDir = viewVector.div(viewDistance).toVar();

      // The seabed under this fragment, evaluated once and shared by the
      // refraction column, the surf and the contact foam. It used to be
      // evaluated three times unconditionally — sixteen texture fetches and the
      // island's structural arithmetic each — on every water pixel in the frame.
      // Signed: negative over sand, where the swash sheet runs.
      const seabedSigned: any = (
        floorDepth === null ? float(1000) : floorDepth(vec3(fieldXZ.x, 0, fieldXZ.y))
      ).toVar();
      const seabedDepth: any = seabedSigned.max(0).toVar();

      // --- surface normal and fold from the derivative fields ----------------
      const slope = vec2(0).toVar();
      // Combine folding across cascades by keeping the most-folded value. Summing
      // would double-count independent bands and wash the whole surface white.
      const fold = float(1).toVar();
      // Slope variance lost inside the texture footprint — see the alpha channel
      // of the derivative pass. Bands are independent, so their variances add.
      const lostSlopeVariance = float(0).toVar();
      for (let i = 0; i < cascadeCount; i++) {
        const d = this.derivativeNodes[i].sample(fieldXZ.div(this.uTileSizes[i])).toVar();
        // Weight folds into the fade, so a cascade the tier does not use
        // contributes no slope and — via `mix` toward 1 — no folding either.
        const fade = cascadeShadingFade(viewDistance, i, cascadeCount).mul(
          this.uCascadeWeights[i],
        );
        slope.addAssign(vec2(d.x, d.y).mul(fade));
        fold.assign(fold.min(mix(float(1), d.z, fade)));
        // E[s^2] - |E[s]|^2, both from the same mip level, so this is exactly the
        // detail this pixel's footprint averaged away — zero where the footprint
        // is a texel and growing as it widens, which is why it needs no distance
        // ramp of its own. `max(0)` guards the half-float subtraction, which can
        // land a few ulps negative when the variance is genuinely nil. Squared
        // fade because variance scales with the square of a linear weight.
        lostSlopeVariance.addAssign(
          d.w.sub(d.x.mul(d.x).add(d.y.mul(d.y))).max(0).mul(fade).mul(fade),
        );
      }

      // Rain impacts, added to the wave slope before the normal is built so they
      // ride the surface rather than sitting on a plane over it.
      //
      // Faded out with distance, hard. The rings are decimetre features; past
      // thirty metres or so they are well under a pixel and all they can
      // contribute is aliasing — the same reason the finest wave cascade fades.
      // Behind a uniform branch, so every clear preset pays one compare.
      If(this.uRainIntensity.greaterThan(0.001), () => {
        const nearness = smoothstepDownClamped(viewDistance, 6, 38);
        slope.addAssign(
          rainSlope(worldPos.xz, this.uRainTime, this.uRainIntensity)
            .mul(this.uRainSlope)
            .mul(nearness),
        );
      });

      // Wake slope, from the gradient of the elevation channel.
      //
      // The vertex stage already displaces geometry by this field, but the ocean
      // takes its normal entirely from the derivative textures and never from the
      // mesh — so without this the hull would push a visible hole through the
      // water that was shaded as if the water were flat. Adding the gradient here
      // is not double-counting; it is the only thing that lights the wake at all.
      //
      // Finite differences over one texel of the buffer, which is 0.41 m at the
      // default 420 m / 1024 configuration. Two extra taps, behind the same
      // uniform compare as the vertex sample.
      if (foam !== null) {
        If(this.uWakeDisplacement.greaterThan(0.001), () => {
          const wakeUv = worldPos.xz.sub(this.uFoamCenter).div(foam.extent).add(0.5).toVar();
          const edge = wakeUv.min(wakeUv.oneMinus()).toVar();
          const inside = edge.x.min(edge.y).smoothstep(0, 0.02).clamp(0, 1).toVar();

          const texelUv = 1 / foam.resolution;
          const texelMetres = foam.extent / foam.resolution;

          const centre = this.wakeNode.sample(wakeUv).g.toVar();
          const gradient = vec2(
            this.wakeNode.sample(wakeUv.add(vec2(texelUv, 0))).g.sub(centre),
            this.wakeNode.sample(wakeUv.add(vec2(0, texelUv))).g.sub(centre),
          ).div(texelMetres);

          // Held further out than the vertex displacement: a normal costs nothing
          // in geometry and the wake stays legible as shading long after
          // displacing vertices by it would only alias.
          const reach = smoothstepDownClamped(viewDistance, 140, 520);
          slope.addAssign(gradient.mul(inside).mul(reach).mul(this.uWakeDisplacement));
        });
      }

      // --- surf: the breakers' shape -------------------------------------------
      // The breaker geometry the vertex stage displaced has to be lit too, and
      // the ocean takes its normal from slope, never from the mesh. Inside the
      // surf zone the FFT slope is cut back by the same factor its displacement
      // was, and the breaker's own slope added along the shoreward direction —
      // which is what makes a wave visibly steepen and stand up as it comes in.
      //
      // Behind a real per-pixel branch. A wavefront is wholly in deep water or
      // wholly in the surf, so the divergence is rare, and every fetch inside
      // uses explicit LOD 0.
      /** Unit vector pointing up the beach. */
      const shoreward = vec2(this.uWindAxis.x, this.uWindAxis.y).toVar();
      const surfPhaseValue = float(0).toVar();
      const surfS = float(0.5).toVar();
      const surfBreak = float(0).toVar();
      const surfBreakLast = float(0).toVar();
      const surfBreakPrevious = float(0).toVar();
      const inSurf = float(0).toVar();
      if (floorDepth !== null) {
        If(seabedSigned.lessThan(surfReachDepth()), () => {
          inSurf.assign(1);
          // The gradient of *depth* points out to sea, so its negation points up
          // the beach. Forward differences over twelve metres: wide enough to
          // average out the heightfield's own metre-scale relief, narrow enough
          // to follow a coastline that turns. Degenerate on a flat bottom, where
          // the wind bearing is the honest answer.
          const e = SHORE_GRADIENT_EPSILON;
          const dDepthX = floorDepth(vec3(fieldXZ.x.add(e), 0, fieldXZ.y)).sub(seabedSigned).toVar();
          const dDepthZ = floorDepth(vec3(fieldXZ.x, 0, fieldXZ.y.add(e))).sub(seabedSigned).toVar();
          const gradient = vec2(dDepthX, dDepthZ).toVar();
          const steepness = gradient.length().toVar();
          shoreward.assign(
            mix(shoreward, gradient.negate().div(steepness.max(1e-4)), steepness.smoothstep(0.02, 0.12)),
          );

          const wave = breaker(seabedDepth, fieldXZ);
          surfPhaseValue.assign(wave.phase);
          surfS.assign(wave.s);
          surfBreak.assign(wave.breakDepth);
          surfBreakLast.assign(wave.breakDepthLast);
          surfBreakPrevious.assign(wave.breakDepthPrevious);

          const fft = mix(
            float(SURF_FFT_FLOOR),
            float(1),
            seabedDepth.smoothstep(0.3, surfReachDepth().mul(0.7)),
          );
          slope.mulAssign(fft);
          // d(eta)/dx along the beach = d(eta)/d(phase) * d(phase)/dx. Faded with
          // distance like the finer cascades: a breaker face is a few metres
          // across, so past a few hundred metres it is shading noise.
          const reach = smoothstepDownClamped(viewDistance, 260, 700);
          slope.addAssign(
            shoreward.mul(wave.etaPerPhase.mul(surfPhaseGradient(seabedDepth))).mul(reach),
          );
        });
      }

      const normal = normalize(vec3(slope.x.negate(), 1, slope.y.negate())).toVar();

      // Flatten the normal toward vertical with distance. Mip filtering removes
      // most of the undersampling shimmer, but past a few kilometres the residual
      // per-pixel slope variation still sparkles, and real water at that range
      // reads as a smooth sheet anyway.
      const flatten = viewDistance.smoothstep(2500, 9000).clamp(0, 1).toVar();
      const n = normalize(mix(normal, vec3(0, 1, 0), flatten)).toVar();

      // A topside normal that faces away from the eye is not a surface the eye
      // can see: at grazing angles the facet that would face away is hidden
      // behind the crest in front of it, and the pixel is actually showing the
      // facets that face the viewer (Noniv/web-ocean, MIT, ocean.js:183-198, does
      // the same). Left alone, `n . v < 0` sent ordinary topside pixels down the
      // *underside* branch below — Snell's window and TIR on water seen from
      // above — which painted flat, hard-edged brown blotches over every
      // gallery frame. Bend such normals back to just facing the viewer.
      If(this.uEyeSubmerged.lessThan(0.5), () => {
        const facing = n.dot(viewDir).toVar();
        If(facing.lessThan(0.06), () => {
          n.assign(normalize(n.add(viewDir.mul(float(0.06).sub(facing)))));
        });
      });

      const nDotV = n.dot(viewDir).clamp(1e-3, 1).toVar();

      // --- Fresnel (Schlick), F0 for an air/water interface ------------------
      const f0 = float(0.02);
      const oneMinus = float(1).sub(nDotV).toVar();
      const fresnelPow = oneMinus.mul(oneMinus).mul(oneMinus).mul(oneMinus).mul(oneMinus).toVar();
      const fresnel = f0.add(float(1).sub(f0).mul(fresnelPow)).clamp(0, 1).toVar();

      // Grazing fade, standing in for the slope the normal no longer carries.
      //
      // Schlick's Fresnel describes one flat interface, and at a grazing angle it
      // is very close to 1 — from 5 m up, water 400 m away is seen at under a
      // degree, so a flat sea reflects 94% of the sky there. That is correct for
      // a flat sea and wrong for this one. The surface has a slope distribution,
      // and the mipmapped derivative fields plus the explicit distance flattening
      // deliberately average it away — so the normal a distant pixel gets is the
      // *mean* normal, which has none of the masking and shadowing the real
      // facets do.
      //
      // The consequence was that everything past the near field went to a sheet
      // of white sky, and it read as foam even though the foam mask there is
      // empty. See `GRAZING_SLOPE_SIGMA` for why this is not the Smith term it
      // resembles.
      const maskedGrazing = nDotV
        .div(nDotV.add(GRAZING_SLOPE_SIGMA))
        .mul(0.55)
        .add(0.45)
        .toVar();
      fresnel.mulAssign(maskedGrazing);

      // --- transmitted colour -------------------------------------------------
      //
      // What the eye sees looking *into* the water: the scene behind the surface,
      // refracted, attenuated by the column of water it crossed, and progressively
      // replaced by light scattered back out of that column.
      //
      // The thickness of the column is measured, not assumed. Reading the depth
      // buffer behind the surface gives the real distance from the surface to
      // whatever is under it, so a hull a metre down stays legible while the
      // seafloor thirty metres down does not — and a drop-off reads as an edge
      // because it genuinely is one. The seafloor heightfield remains the fallback
      // for the WebGL2 path and for anything the depth buffer cannot answer for.

      // Surface normals bend the view ray. Scaled down with distance, because the
      // same lateral offset at the horizon is a whole screen away, and scaled by
      // depth so shallow water distorts less than deep — which is what stops the
      // distortion from tearing at a shoreline.
      const distortion = n.xz
        .mul(this.uRefractionStrength)
        .div(viewDistance.mul(0.06).add(1))
        .toVar();
      const refractedUv: any = (viewportSafeUV(screenUV.add(distortion)) as any).toVar();

      // Depth of the scene behind the surface, and of the surface itself.
      // `linearDepth` is normalised over the near/far range, so the difference is
      // scaled back into metres by `uDepthRange` before it means anything.
      const surfaceZ: any = (linearDepth() as any).toVar();
      const behindZ: any = (linearDepth(viewportDepthTexture(refractedUv)) as any).toVar();

      // A refracted sample can land on something *in front of* the water — the
      // hull's own topsides, most obviously — and pulling that colour underwater
      // smears it down the wave face. Where that happens, fall back to the
      // unrefracted sample, which is behind the surface by construction.
      const straightZ: any = (linearDepth(viewportDepthTexture(screenUV)) as any).toVar();
      const valid: any = behindZ.greaterThan(surfaceZ).toVar();
      const sampleUv: any = valid.select(refractedUv, screenUV).toVar();
      const backdropZ: any = valid.select(behindZ, straightZ).toVar();

      // Column thickness. Two independent estimates, and the smaller wins: the
      // depth buffer knows about the hull and the props, the heightfield knows
      // about seafloor the depth buffer may never have rendered.
      //
      // The depth difference is measured along the camera's *forward axis*, and
      // light travels along the pixel's own ray — which is longer, by exactly the
      // secant of the angle between them. Without that correction the water
      // absorbs less toward the edges of the frame than at its centre, and the
      // apparent density of the whole ocean changes with field of view. An
      // independent review caught this; the volumetric fog had it right and this
      // did not.
      //
      // `nDotV` is the wrong correction for it, incidentally, and was what the
      // seafloor estimate used: that is incidence against the *wave* normal, so
      // it made the water's density flicker with the ripples.
      const axialToRay: any = float(1).div(viewDir.dot(this.uCameraForward).abs().max(0.15));
      const bufferThickness: any = backdropZ
        .sub(surfaceZ)
        .max(0)
        .mul(this.uDepthRange)
        .mul(axialToRay)
        .toVar();
      // Water between the surface and whatever is *directly* behind it, before
      // refraction moved the lookup. Contact foam is about where an object
      // pierces this pixel's surface, so it must not follow the refracted ray.
      const straightThickness: any = straightZ
        .sub(surfaceZ)
        .max(0)
        .mul(this.uDepthRange)
        .mul(axialToRay)
        .toVar();
      const pathLength: any = (
        floorDepth === null
          ? bufferThickness
          : bufferThickness.min(
              // Depth below the surface, stretched along the same ray. The ray
              // bends toward vertical entering water, so this over-estimates a
              // little at grazing angles; the buffer estimate normally wins there
              // anyway.
              seabedDepth.mul(axialToRay),
            )
      ).toVar();

      const absorption: any = this.uExtinction.mul(pathLength).negate().exp().toVar();

      // Beer–Lambert on the refracted scene colour, plus the light scattered out
      // of the column toward the eye. The two are complementary: whatever the
      // water absorbed on the way through is what it has to give back as body
      // colour, which is why `absorption` weights one and its complement the
      // other rather than both being tuned independently.
      const refracted: any = viewportSharedTexture(sampleUv).rgb.toVar();
      // `shallowColor` is the look of thin water *over bright sand* — the bed's
      // own reflectance, tinted. Where the depth buffer finds an object well
      // above the seabed (a hull, a buoy, a rock face), that sand is not there,
      // and the same tint painted the submerged hull a flat turquoise sheet
      // brighter than the open sea around it. Over an object, the column
      // scatters like open water: the deep body colour, weighted by how much
      // the column absorbed.
      const objectBehind: any = (
        floorDepth === null
          ? float(0)
          : floorDepth(worldPos).max(0).sub(bufferThickness).smoothstep(0.35, 1.2)
      ).toVar();
      const inscatter: any = mix(
        mix(this.uDeepColor, this.uShallowColor, absorption),
        this.uDeepColor,
        objectBehind,
      ).toVar();
      // Faded out with distance — see REFRACTION_FADE_NEAR. The backdrop is a
      // screen-space read at whatever resolution the frame happens to be, so
      // past a couple of hundred metres it is delivering the seabed's texture
      // well below its sampling rate and the result is the aliased khaki band
      // across `waves.png`. The body colour it hands over to is the integral of
      // the same column.
      const refractionReach: any = smoothstepDownClamped(
        viewDistance,
        REFRACTION_FADE_NEAR,
        REFRACTION_FADE_FAR,
      ).toVar();
      const bodyColor: any = mix(
        inscatter,
        refracted,
        absorption.mul(this.uRefractionAmount).mul(refractionReach),
      ).toVar();

      // --- subsurface scattering ---------------------------------------------
      //
      // Light that entered the back of a wave, travelled through it, and came
      // out toward the eye. Three things decide how much arrives, and until this
      // rewrite the term used none of them: it was an authored colour times a
      // backlight lobe times the wave's absolute height, which is a tint that
      // happens to appear on tall water. The extinction coefficient, the path
      // length and the absorption were all computed a dozen lines above it for
      // the transmission term and none of them were consulted.
      //
      // **How far the light travelled.** A backlit crest glows because the water
      // there is *thin* — the tip is centimetres of water between the sun and
      // the eye, while the body of the wave below it is metres. That is the
      // opposite of what the old `worldPos.y` ramp said, which brightened the
      // term as the water got taller. Thickness is estimated from the wave's own
      // height relative to the sea state: at the crest it approaches
      // `THIN_CREST_METRES`, and it grows toward `THICK_BODY_METRES` down the
      // face. Normalising by the sea state matters — a 0.4 m chop and a 6 m swell
      // have different crests, and a fixed metre scale makes one of them wrong.
      const seaScale = this.uWaveScale.max(0.35).toVar();
      const heightAboveMean = worldPos.y.div(seaScale).clamp(0, 1).toVar();
      const thickness = mix(
        float(THICK_BODY_METRES),
        float(THIN_CREST_METRES),
        heightAboveMean,
      ).toVar();

      // **What the medium did to it on the way.** Beer–Lambert through that
      // thickness, with the *same* per-channel extinction the transmission term
      // uses. This is where the colour comes from now, and it is why the term no
      // longer needs an authored hue to look like water: red is extinguished in
      // the first metre and blue-green survives, so a thin crest comes out the
      // jade of real backlit water and a thick one goes blue and dies. The
      // authored `scatterColor` stays, demoted to what it physically is — the
      // scattering albedo of the body, which is where turbidity and chlorophyll
      // legitimately live.
      const transmitted: any = this.uExtinction.mul(thickness).negate().exp().toVar();

      // **Which way it went.** Water scatters strongly forward, so the lobe
      // should be sharp and centred on the sun's own direction rather than the
      // cube of a clamped dot, which is a stand-in with no particular shape.
      // Henyey–Greenstein at g = 0.72 is the standard cheap phase function and
      // gives a much tighter forward peak with a real tail off-axis; the tail is
      // what stops the effect switching off the moment the sun leaves the
      // frame's back half.
      const sunDir = normalize(this.uSunDirection).toVar();
      const cosTheta = viewDir.negate().dot(sunDir).toVar();
      const g = float(SCATTER_ANISOTROPY);
      const gg = g.mul(g).toVar();
      const denom = gg.add(1).sub(cosTheta.mul(g).mul(2)).max(1e-4).toVar();
      const phase = gg
        .oneMinus()
        .div(denom.mul(denom.sqrt()).mul(4 * Math.PI))
        .toVar();

      // The surface still has to be facing away from the sun for light to enter
      // its far side. A wrapped term rather than a hard `dot > 0`, because the
      // shading normal at a crest is close to horizontal and a hard cutoff makes
      // the glow flicker along the crest line as the normal crosses it.
      const awayFromSun = n
        .dot(sunDir)
        .negate()
        .add(SCATTER_WRAP)
        .div(1 + SCATTER_WRAP)
        .clamp(0, 1)
        .toVar();

      // Scaled by how much light there actually is.
      //
      // Subsurface scattering is transmitted *sunlight*: it cannot be brighter
      // than what is illuminating the water. Without this the term kept its full
      // daytime value after dark, and the water around the submerged hull glowed
      // green at half past nine at night.
      const scatter: any = (this.uScatterColor as any)
        .mul(transmitted)
        .mul(
          phase
            .mul(awayFromSun)
            .mul(this.uScatterStrength)
            .mul(this.uLightLevel)
            .mul(SCATTER_NORMALISE),
        )
        .toVar();

      // Crest glow: the light a wave's upper third carries *sideways* to the eye
      // whatever the sun is doing. The phase lobe above only fires looking into
      // the sun, so a front-lit or cross-lit sea had no translucency at all and
      // every crest was the same opaque blue as the trough below it — which is
      // what made the open sea read as a flat lagoon. Thin water near the crest
      // scatters skylight and refracted sunlight toward a viewer looking across
      // it, strongest at grazing view (longer path through the thin part) and
      // more so when backlit. This is the term stylised-real game oceans lean on
      // for their emerald crests; here it is driven by the same crest height and
      // thinness the physical term uses, so it scales with the sea state.
      const crest = heightAboveMean.mul(heightAboveMean).toVar();
      const across = float(1).sub(viewDir.y.abs()).clamp(0, 1).toVar();
      scatter.addAssign(
        (this.uScatterColor as any)
          .mul(crest)
          .mul(across.mul(0.6).add(0.4))
          .mul(awayFromSun.mul(0.6).add(0.4))
          .mul(this.uCrestGlow)
          .mul(this.uLightLevel.min(1.2)),
      );

      // --- reflection ----------------------------------------------------------
      // The analytic sky gradient is kept as the base layer, not replaced. It is
      // what fills the horizon, where a planar reflection has nothing to offer
      // and where its texture runs out anyway; the mirrored scene is composited
      // over it wherever it actually contains something.
      const reflectDir = viewDir.negate().reflect(n).toVar();
      const skyBlend = reflectDir.y.clamp(0, 1).sqrt().toVar();
      const reflection = mix(this.uHorizonColor, this.uSkyColor, skyBlend).toVar();

      if (planar !== null) {
        // The mirror is a flat plane; the water is not. Offsetting the lookup by
        // the surface normal is what makes the reflection ripple with the waves
        // rather than sitting on them like a decal. Scaled down with distance for
        // the same reason the refraction offset is: a fixed screen-space offset
        // is metres at the horizon and millimetres underfoot.
        const offset = n.xz
          .mul(this.uReflectionDistortion)
          .div(viewDistance.mul(0.05).add(1))
          .toVar();
        const rawUv = screenUV.add(offset).toVar();

        // Sampled at a roughness-driven mip level, not as a sharp mirror.
        //
        // A mirror is what a *flat* interface does. This surface has a slope
        // distribution the shading normal has already averaged over — a pixel a
        // few hundred metres out covers many wavelengths — so the reflection it
        // should show is the average over that cone of directions, which is a
        // blurred reflection. Taking level 0 everywhere is what makes a rough sea
        // reflect like polished metal, and it is one of the clearest tells that
        // water is being rendered by a mirror plane.
        //
        // Level from the base roughness and from distance, since both widen the
        // cone: a metre of surface at 400 m subtends far less than a pixel.
        // Capped low on purpose. The reflection target is already at half
        // resolution, so level 3 is sampling an 80x45 image — and averaging a
        // bright sky into that produces visible blobs where a blurred reflection
        // should be. Two levels is enough to take the mirror off it.
        const reflectionLod = this.uRoughness
          .mul(12)
          .add(viewDistance.mul(0.004))
          .clamp(0, 1.25)
          .toVar();
        const mirrored = planar.sample(viewportSafeUV(rawUv)).level(reflectionLod).toVar();

        // Faded out where the lookup leaves the reflection, rather than clamped
        // into it. A mirrored camera only covers what is in front of it, so near
        // the horizon the offset walks the lookup off the edge — and clamping
        // there streaks the last row of texels sideways, which showed up as a
        // dark band lying along the horizon.
        const rEdge = rawUv.min(rawUv.oneMinus()).toVar();
        const inFrame = rEdge.x.min(rEdge.y).smoothstep(0, 0.05).clamp(0, 1).toVar();

        // Also faded as the view flattens. At a grazing angle the reflected ray
        // leaves the plane almost immediately and the planar approximation stops
        // describing anything; the analytic sky is the better answer there, and
        // it is also what the eye expects, since distant water reads as sky.
        const facing = nDotV.smoothstep(0.02, 0.22).toVar();

        reflection.assign(
          mix(reflection, mirrored.rgb, inFrame.mul(facing).mul(this.uReflectionAmount)),
        );
      }

      if (ssr !== null) {
        // Layered *over* the planar reflection, not instead of it.
        //
        // The two fail in opposite places, which is exactly why engines composite
        // rather than choose. A screen-space trace has the real displaced surface
        // and gets the hull's reflection attached to the wave under it — the case
        // planar cannot serve, since one mirror plane at y = 0 misplaces
        // everything coming off a wave face. But it can only reflect what is
        // already on screen, so it loses the moment the reflected object leaves
        // the frame, and there the planar view still has it.
        //
        // The distance-flattened `n` is passed rather than the raw normal: the
        // trace is far more sensitive to normal roughness than the shading is,
        // because adjacent pixels fire rays metres apart at grazing incidence.
        reflection.assign(ssr(worldPos, n, reflection));
      }

      // --- sun specular: the whole microfacet BRDF, not just its NDF -----------
      //
      // This was `D * N.L * intensity` and nothing else — no Fresnel, no
      // masking-shadowing, and no `1 / (4 (N.L)(N.V))` denominator. The NDF is
      // normalised over the *microfacet distribution*, not over outgoing
      // radiance, so on its own it is not a reflectance and its peak scales as
      // `1 / (pi * a2)`. At the surface's 0.075 roughness that is a2 = 3.2e-5,
      // so the term peaked near 10,000 before the sun's intensity multiplied it
      // — four to five orders of magnitude of specular energy that does not
      // exist, dumped straight into the tone mapper wherever the half-vector
      // lined up. That is most of why the water read as white foil.
      //
      // The three missing factors are not refinements; each removes a specific
      // error. The visibility term is Heitz's height-correlated Smith form,
      // which already folds in the `4 (N.L)(N.V)` denominator. Fresnel is
      // evaluated on the *half-vector*, not the macro normal, because that is
      // the facet actually doing the reflecting.
      const halfVector = normalize(viewDir.add(sunDir)).toVar();
      const nDotH = n.dot(halfVector).clamp(0, 1).toVar();
      const nDotL = n.dot(sunDir).clamp(0, 1).toVar();
      const vDotH = viewDir.dot(halfVector).clamp(0, 1).toVar();

      // Geometric specular antialiasing.
      //
      // **Not** Kaplanyan et al.'s NDF filtering, which an earlier version of
      // this comment claimed. That method filters the distribution in half-vector
      // slope space and carries the full anisotropic covariance; this takes
      // derivatives of the final shading normal and adds their scalar magnitude
      // to `alpha^2`, which is the cheaper geometric variant that shipped in
      // Frostbite and Unity's HDRP. It is a real technique with a real
      // derivation, it is just a different one, and it throws away the covariance
      // — so it cannot know that a pixel's normal varies more along the wind than
      // across it, which for this surface is precisely the interesting part.
      //
      // Sun glitter on water is the hardest case there is for a narrow specular
      // lobe. The surface carries metre-scale slope detail, a pixel a hundred
      // metres out covers many wavelengths of it, and the shading normal is an
      // average — so wherever a wave face happens to line the half-vector up
      // exactly, one pixel gets the full peak of a lobe that should have been
      // integrated over the whole footprint. That is what the two blown-out
      // hotspots were, and no amount of tuning the roughness fixes it: raising it
      // kills the glitter everywhere, lowering it makes the spikes worse.
      //
      // The fix is to widen the lobe by the *variance of the normal across the
      // pixel*, which the screen-space derivatives measure directly. Where the
      // normal is smooth this adds nothing; where a pixel spans a lot of slope it
      // broadens the lobe to cover what it is actually looking at, which is the
      // definition of correct filtering rather than a hack.
      // Resolved onto the tangent frame further down, once that frame exists —
      // see the filtering block below `aspect`. These two are the raw material.
      const dNdx = dFdx(n).toVar();
      const dNdy = dFdy(n).toVar();

      const alphaBase = this.uRoughness.mul(this.uRoughness).toVar();

      // D — anisotropic Trowbridge-Reitz.
      //
      // The isotropic form gives a round highlight, and sun glitter on water is
      // not round: it is a track drawn *toward the viewer*, perpendicular to the
      // crests. That shape comes straight out of the slope distribution being
      // wider along the wind than across it — see `uSlopeAnisotropy` — so the NDF
      // has to carry two roughnesses rather than one. No amount of tuning a
      // single roughness produces it.
      //
      // Burley's form: with the half-vector resolved onto the tangent frame, the
      // denominator is `(Ht/at)^2 + (Hb/ab)^2 + Hn^2`, and the isotropic GGX falls
      // out when at == ab.
      // The tangent frame has to be orthonormal *to the shading normal*, not to
      // world up.
      //
      // Taking the wind axis and its horizontal perpendicular directly gives a
      // basis that is orthonormal only where the surface happens to be flat. On
      // any tilted wave — which is the entire subject here — `H.T`, `H.B` and
      // `H.N` are then coordinates in a non-orthogonal basis, the sum of squares
      // in the denominator is not the squared length of anything, and the NDF
      // stops being normalised. It over-reports on wave faces, which is where a
      // sun track lives.
      //
      // Gram-Schmidt: project the wind onto the tangent plane and re-normalise.
      // The fallback matters at the poles of that projection — a wave face
      // perpendicular to the wind — where the projection length goes to zero.
      const windWorld = vec3(this.uWindAxis.x, 0, this.uWindAxis.y).toVar();
      const projected = windWorld.sub(n.mul(n.dot(windWorld))).toVar();

      // The fallback has to be an axis the normal cannot be parallel to, and it
      // has to be chosen *before* projecting rather than blended after.
      //
      // Blending two unnormalised projections is not a fallback: near the
      // degenerate direction both are short and they can cancel, which leaves a
      // zero tangent and a NaN frame on a thin ring of wave faces. Picking world
      // X or Z by whichever the normal is less aligned with guarantees a
      // projection of at least 1/sqrt(2) before normalisation.
      const fallbackAxis = n.x
        .abs()
        .lessThan(n.z.abs())
        .select(vec3(1, 0, 0), vec3(0, 0, 1))
        .toVar();
      const fallback = fallbackAxis.sub(n.mul(n.dot(fallbackAxis))).toVar();
      const tangent = normalize(
        projected.length().greaterThan(0.05).select(projected, fallback),
      ).toVar();
      const bitangent = normalize(n.cross(tangent)).toVar();
      // Fourth root, not square root.
      //
      // `uSlopeAnisotropy` is a ratio of slope *variances*, and variance goes as
      // alpha squared — so scaling alpha by sqrt(R) one way and 1/sqrt(R) the
      // other makes the variance ratio R^2, not R. At the measured 1.5 that is
      // 2.25: the track came out half again as stretched as the sea it is
      // supposed to be describing. R^(1/4) gives alphaT/alphaB = sqrt(R) and a
      // variance ratio of exactly R, with the geometric mean of the two
      // roughnesses preserved at the isotropic value.
      const aspect = this.uSlopeAnisotropy.pow(0.25).toVar();

      // Specular antialiasing, resolved on the *anisotropic* frame.
      //
      // This is the fix for the single most visible defect in the gallery: hard
      // white blobs across the mid-field of `clear-day.png`, scratchy stripes in
      // `island.png`, dark speckle holes in `sunset.png`. One cause, three
      // appearances.
      //
      // What stood here added a **scalar** to `alpha^2` and only then split the
      // result into `alphaT` and `alphaB`, which is the geometric variant
      // Frostbite and HDRP ship and is honest about what it discards: the
      // covariance. That matters more on this surface than on almost any other,
      // because a pixel of wind-driven sea does not vary equally in every
      // direction — it varies about `uSlopeAnisotropy` times as much along the
      // wind as across it. Filtering it isotropically and *then* stretching the
      // result by the aspect ratio widens the lobe in proportion to the
      // roughness rather than in proportion to the measured variance, so it
      // over-filters across the wind and under-filters along it. Under-filtering
      // along the wind is what leaves the blobs.
      //
      // Kaplanyan et al. (2016) filter the distribution in slope space and carry
      // the full 2x2 covariance. This carries its diagonal in the tangent frame,
      // which is where the sea's covariance is very nearly diagonal by
      // construction — the frame is built on the wind axis, and the wind axis is
      // the principal axis of the slope distribution. The off-diagonal term the
      // full method would add is what remains after that alignment, and it is
      // small.
      //
      // `dN` is tangential to `n`, so its components on T and B are exactly the
      // per-axis slope changes; `varT + varB` recovers the scalar this replaces.
      const dT = vec2(dNdx.dot(tangent), dNdy.dot(tangent)).toVar();
      const dB = vec2(dNdx.dot(bitangent), dNdy.dot(bitangent)).toVar();
      const varT = dT.dot(dT).mul(SPECULAR_AA_SCREEN_SPACE_VARIANCE).toVar();
      const varB = dB.dot(dB).mul(SPECULAR_AA_SCREEN_SPACE_VARIANCE).toVar();

      // The sub-footprint variance splits along the same axes, because it is the
      // same distribution seen at a smaller scale. `lostSlopeVariance` is the
      // total over both axes and the ratio between them is `uSlopeAnisotropy` by
      // definition, so `R/(1+R)` and `1/(1+R)` divide it exactly. At R = 1 each
      // axis takes half, which reproduces the isotropic behaviour this replaces.
      const ratio = this.uSlopeAnisotropy.toVar();
      const lostShare = ratio.div(ratio.add(1)).toVar();
      const lostT = lostSlopeVariance.mul(lostShare).toVar();
      const lostB = lostSlopeVariance.mul(lostShare.oneMinus()).toVar();

      // Widen alpha^2, not alpha: variance is a second moment. For GGX the slope
      // variance is `alpha^2 / 2`, so a slope variance `v` enters as `2v`.
      //
      // The ceiling applies to the screen-space term only, and that asymmetry is
      // deliberate. It exists because the published screen-space constant is five
      // thousand times water's own alpha^2 and needed reining in; `lostT`/`lostB`
      // are measured slope variances already in the right units, and capping them
      // would throw the correction away exactly where the footprint is widest and
      // it matters most.
      const alphaT = alphaBase
        .mul(aspect)
        .pow(2)
        .add(varT.mul(2).min(SPECULAR_AA_VARIANCE_CEIL))
        .add(lostT.mul(2))
        .min(1)
        .sqrt()
        .toVar();
      const alphaB = alphaBase
        .div(aspect)
        .pow(2)
        .add(varB.mul(2).min(SPECULAR_AA_VARIANCE_CEIL))
        .add(lostB.mul(2))
        .min(1)
        .sqrt()
        .toVar();

      const hT = halfVector.dot(tangent).div(alphaT).toVar();
      const hB = halfVector.dot(bitangent).div(alphaB).toVar();
      const aniDenom = hT.mul(hT).add(hB.mul(hB)).add(nDotH.mul(nDotH)).toVar();
      const distribution = float(1)
        .div(
          alphaT
            .mul(alphaB)
            .mul(aniDenom.mul(aniDenom))
            .mul(Math.PI)
            .max(1e-9),
        )
        .toVar();

      // V — height-correlated Smith visibility, **anisotropic to match D**.
      //
      // Pairing an anisotropic distribution with an isotropic masking term is not
      // a partial implementation, it is an unmatched BRDF: the two describe
      // different microsurfaces, and the mismatch shows up as energy at exactly
      // the grazing angles a stretched glitter track occupies. Heitz's
      // anisotropic form resolves each direction onto the same tangent frame the
      // distribution uses.
      const lengthV = vec3(
        alphaT.mul(viewDir.dot(tangent)),
        alphaB.mul(viewDir.dot(bitangent)),
        nDotV,
      )
        .length()
        .toVar();
      const lengthL = vec3(
        alphaT.mul(sunDir.dot(tangent)),
        alphaB.mul(sunDir.dot(bitangent)),
        nDotL,
      )
        .length()
        .toVar();
      const visibility = float(0.5)
        .div(nDotL.mul(lengthV).add(nDotV.mul(lengthL)).max(1e-6))
        .toVar();

      // F — Schlick on the half-vector, F0 for an air/water interface.
      const oneMinusVH = float(1).sub(vDotH).toVar();
      const vhPow = oneMinusVH
        .mul(oneMinusVH)
        .mul(oneMinusVH)
        .mul(oneMinusVH)
        .mul(oneMinusVH)
        .toVar();
      const fresnelSpecular = f0.add(float(1).sub(f0).mul(vhPow)).toVar();

      const specular = this.uSunColor
        .mul(distribution.mul(visibility).mul(fresnelSpecular).mul(nDotL))
        .mul(this.uSunIntensity)
        .toVar();

      // Cloud shade.
      //
      // A cloud deck's largest effect on what is under it is moving patches of
      // shadow, and a sky drawn over an evenly-lit sea reads as a backdrop rather
      // than as weather. It attenuates the *sun* terms only — the specular and
      // the scattered daylight — and leaves the sky reflection alone, because the
      // sky is still there above the shadow and is what fills it.
      const sunShade = float(1).toVar();
      if (cloudShadow !== null) {
        sunShade.assign(cloudShadow(worldPos));
        specular.mulAssign(sunShade);
      }

      // --- combine ---------------------------------------------------------------
      // The body colour is authored for *daylight*. Its diffuse term bottomed out
      // at 0.45, which is a floor on how dark lit water can get — so at night the
      // shallow-water tint kept glowing at nearly half strength and the water
      // around the submerged hull read as a green lamp. It now falls with the
      // light, to a small floor rather than to zero: a night sea is dark, but it
      // is not black, because the whole sky is still faintly lighting it.
      const ambientFloor = mix(float(0.1), float(1), this.uLightLevel).toVar();
      const litBody = bodyColor
        .mul(nDotL.mul(0.55).add(0.45).mul(sunShade.mul(0.55).add(0.45)))
        .mul(ambientFloor)
        .add(scatter.mul(sunShade))
        .toVar();

      // --- foam --------------------------------------------------------------------
      // The Jacobian is 1 on unstretched water and drops below 0 where the surface
      // folds onto itself; that fold region is physically where whitecaps break.
      const coverage = smoothstepDownClamped(
        fold,
        this.uFoamThreshold.sub(this.uFoamSoftness),
        this.uFoamThreshold,
      ).toVar();

      // Whitecaps break at crests, not in troughs. The Jacobian alone is a
      // low-frequency field and marks broad patches, so bias it by local
      // elevation: the same amount of folding produces foam on a crest and
      // almost none a metre lower down. This is what turns smooth blobs into
      // streaks that follow the wave tops.
      //
      // The bias now runs to zero rather than bottoming out at a quarter. That
      // floor was there to keep the instantaneous mask looking continuous, and
      // it is exactly what put a wash of foam into every trough; with the
      // accumulation buffer carrying persistence, this term no longer has to
      // pretend to be continuous and can be as selective as real whitecaps are.
      const crestBias = worldPos.y.smoothstep(-0.2, 2.2).clamp(0, 1).toVar();
      const biased = coverage.mul(crestBias).toVar();

      // Break the mask up with world-space noise at roughly the scale of real
      // foam clumps (sub-metre), so the edge dissolves into bubbles rather than
      // ending on a clean contour.
      // Sampled in the wind's own frame, stretched along it — see
      // `FOAM_WIND_STRETCH`. The rotation is the wind axis and its perpendicular
      // used as basis vectors, which is a pair of dot products; building a
      // matrix for a 2-D rotation would cost more and say less.
      const windAlong = vec2(this.uWindAxis.x, this.uWindAxis.y).toVar();
      const windAcross = vec2(this.uWindAxis.y.negate(), this.uWindAxis.x).toVar();
      const foamUv = vec2(
        worldPos.xz.dot(windAlong).div(FOAM_WIND_STRETCH),
        worldPos.xz.dot(windAcross),
      )
        .mul(1.4)
        .toVar();
      const breakup = float(0).toVar();
      let amplitude = 0.5;
      let frequency = 1;
      for (let octave = 0; octave < 4; octave++) {
        breakup.addAssign(
          valueNoise(foamUv.mul(frequency).add(vec2(octave * 17.3, octave * 9.1))).mul(amplitude),
        );
        amplitude *= 0.5;
        frequency *= 2.07;
      }
      // Centre the noise on zero so it perturbs the mask both ways instead of
      // only ever eating into it.
      //
      // Faded out with distance, which *is* band-limiting the field rather than
      // the edge. Its finest octave is a sub-metre feature; past a few hundred
      // metres that is well under a pixel, so what it contributes there is not
      // texture, it is noise with no scale — the same argument the wave cascades'
      // shading fade and the rain rings both make. Up close nothing changes.
      const perturb = breakup
        .sub(0.5)
        .mul(smoothstepDownClamped(viewDistance, FOAM_BREAKUP_NEAR, FOAM_BREAKUP_FAR))
        .toVar();

      // A wide smoothstep keeps the boundary soft; the noise decides *where* that
      // boundary falls, which reads as texture rather than as a fading blob.
      //
      // Widened by the mask's own screen-space footprint, and this is the second
      // half of the aliasing fix. The specular filter above was not the whole
      // story: the frames show the *foam* aliasing too, and it is what makes the
      // blobs hard-edged rather than merely bright. At mid distance the Jacobian
      // fold field is undersampled, so a threshold on it flips fully on and fully
      // off between neighbouring pixels — there is no gradient at the boundary
      // because the boundary is narrower than a pixel.
      //
      // `fwidth` of the thresholded quantity is the textbook band-limit for
      // exactly this: it measures how much the argument moves across one pixel,
      // so widening the ramp by it makes the transition take at least a pixel
      // wherever the field stops being resolved, and changes nothing up close
      // where it is. Same argument the specular filter makes, one term along.
      const foamArg = biased.add(perturb.mul(0.45)).toVar();
      // **The ramp is not widened at all, and that is a result rather than an
      // omission.**
      //
      // Widening a threshold by the footprint of its own argument is the textbook
      // band-limit, it is what the specular filter above legitimately does, and it
      // was implemented here twice — once driven by `lostSlopeVariance` and once
      // by `fwidth`. Both were rejected on measurement. `gallery-jitter` puts
      // far-field high-frequency energy at Medium at 2.09 for the mask as it
      // stands, 2.56 with the `fwidth` widening and 2.94 with the variance one,
      // against a 2.33 ceiling.
      //
      // The mechanism is worth recording because it is counter-intuitive and the
      // next reader will have the same idea. A steep threshold on a noisy
      // argument produces a **binary** field: its Laplacian is large, but only on
      // sparse edges, and saturation hides everything either side of them. A ramp
      // wide enough to cover the sub-footprint spread maps that noise
      // *continuously*, so every pixel in the band carries a share of it. The
      // blobs soften and the noise floor rises. Widening band-limits the *edge*;
      // it does not band-limit the *field*, and the field is what the mip chain
      // lost. `fwidth` is no better than the variance here for the same reason it
      // is no help to the specular: it is the derivative of a quantity that is
      // itself undersampled, so in the far field it reports a large number
      // because the wave, not the noise, moved.
      //
      // What *is* correct is fading the field itself where it stops being
      // resolvable, which is what the breakup noise above now does, and what the
      // refracted backdrop and the specular's slope-space filter do in their own
      // terms. The mask's remaining hard edges at mid distance are a real and
      // acknowledged limit, and the honest fix for them is a temporal resolve
      // this renderer deliberately does not have.
      const crestFoam = foamArg.smoothstep(0.12, 0.78).clamp(0, 1).toVar();

      // --- accumulated foam ---------------------------------------------------
      // Wake and any other persistent deposit, read from the world-anchored
      // buffer. Unlike the crest mask above this is *history*: it was laid down
      // on earlier frames and has been decaying since, which is what lets a wake
      // trail behind a moving hull instead of being a decal under it.
      const accumulated = float(0).toVar();
      /** How completely the accumulation buffer covers this fragment, 0..1. */
      const buffered = float(0).toVar();
      if (foam !== null) {
        const foamUvw = worldPos.xz.sub(this.uFoamCenter).div(foam.extent).add(0.5).toVar();

        // The buffer covers a finite square. Outside it there is no information,
        // and clamp-to-edge would smear the border across the whole ocean, so
        // fade out just inside the edge instead.
        const edge = foamUvw.min(foamUvw.oneMinus()).toVar();
        const inside = edge.x.min(edge.y).smoothstep(0, 0.02).clamp(0, 1).toVar();
        buffered.assign(inside);

        accumulated.assign(
          this.wakeNode.sample(foamUvw).r.clamp(0, 1).mul(inside).mul(this.uFoamStrength),
        );

        // Broken up by the same noise as the crest foam so the two read as one
        // material. Without this the wake is a smooth airbrushed streak against
        // bubbly whitecaps and the eye separates them immediately.
        //
        // The breakup is *gated on there being foam here at all*. Added
        // unconditionally it is a signed term, so on empty water the positive
        // half of the noise alone clears the threshold and sprays foam across an
        // ocean that has none — the mask has to be able to return exactly zero.
        // The breakup is deliberately strong. At 0.22 it was a faint dither on a
        // smooth field: the buffer is 0.41 m per texel and bilinearly filtered,
        // so what it hands over is inherently soft, and a weak perturbation
        // leaves it looking airbrushed rather than bubbly. At 0.5 the noise is
        // what decides the foam's edge, and the buffer only decides where the
        // foam *region* is — which is the right division of labour, because the
        // buffer cannot resolve a bubble and the noise is scale-free.
        //
        // The remap is also much wider than it was. `smoothstep(0.16, 0.72)`
        // drove a half-strength deposit to four-fifths white, so foam that should
        // have been thinning as it aged stayed opaque until it vanished.
        const present = accumulated.smoothstep(0.02, 0.2).toVar();
        const density = accumulated
          .add(perturb.mul(0.5).mul(present))
          .smoothstep(0.06, 0.68)
          .clamp(0, 1)
          .mul(present)
          .toVar();
        accumulated.assign(density);

        // Bubble structure, so density reads as foam rather than as paint.
        //
        // The breakup above decides where a patch's edge falls and nothing
        // inside it: a saturated deposit came out as an even, untextured
        // white disc tens of metres across. Real whitecap foam is a network of
        // bubble rafts with water showing through the cells, and only the
        // freshest is solid. So density sets a threshold on a cell field —
        // the same one the surf's lace is drawn from — instead of an opacity.
        // Thin foam is the cell boundaries; denser foam fills the cells from
        // the boundaries inward; a fully saturated texel still shows its
        // centres at half strength.
        //
        // Faded to the cells' average with the breakup noise, for the same
        // reason: past a few hundred metres the cells are under a pixel and
        // would be shimmer, not texture. Evaluated only where there is foam,
        // since nine cells per fragment of open sea buy nothing.
        If(present.greaterThan(0.001), () => {
          // Two scales, the finer one at less than half the size, so the
          // network is not one regular net: a single cell size at this range
          // read as a white fishnet laid on the water.
          const cellUv = foamUv.mul(0.8).add(vec2(breakup.mul(1.1), breakup.mul(-0.8))).toVar();
          const cellsA = foamCells(cellUv).toVar();
          const cellsB = foamCells(cellUv.mul(2.3).add(vec2(5.3, 1.7))).toVar();
          // 1 on the boundaries, 0 at the cell centres.
          const rim = cellsA.y
            .smoothstep(0.02, 0.45)
            .oneMinus()
            .max(cellsB.y.smoothstep(0.02, 0.35).oneMinus().mul(0.75))
            .toVar();
          // Saturated foam still leaves the cell centres open: a 0.85 reach
          // rather than 1, so even a fresh patch reads as a raft, not a sheet.
          const edge0 = density.mul(0.85).oneMinus().sub(0.15).toVar();
          const textured = rim.smoothstep(edge0, edge0.add(0.3)).mul(density.mul(2).min(1));
          // What the cells average to where they are too small to draw, so a
          // patch keeps its brightness as it recedes: about 0.7 of the density
          // once the cells have filled, less while it is only their rims.
          const averaged = density.mul(density.mul(2).min(1)).mul(0.7);
          const near = smoothstepDownClamped(viewDistance, FOAM_BREAKUP_NEAR, FOAM_BREAKUP_FAR);
          accumulated.assign(mix(averaged, textured, near));
        });
      }

      // Where the buffer reaches, the buffer *is* the whitecaps.
      //
      // These two are not two kinds of foam, they are two implementations of one
      // kind, and for a long time both ran everywhere. That is why the near field
      // was a sheet of white: every breaking crest was painted twice, once as a
      // persistent deposit and once as an instantaneous mask over the top of it,
      // and `max` of the two can only ever be the more generous.
      //
      // The buffer is the better answer wherever it has data — it carries
      // persistence, so foam is left behind by a wave instead of travelling with
      // it — but it only covers a 420 m square around the camera. Past that the
      // instantaneous mask is all there is, so it is faded in exactly as the
      // buffer's own edge fades out and the two never overlap.
      // --- surf ----------------------------------------------------------------
      //
      // White water driven by the *seabed*, because every foam term above is
      // driven by the surface and neither knows the bottom is there. A wave
      // does not break because it steepened; it breaks because it ran out of
      // water. The breakers themselves — where each crest is, how big, where it
      // breaks — come from `ocean/Surf`, shared with the vertex stage (which
      // shaped them) and the ground (which the swash wets), so the foam drawn
      // here sits exactly on the crest the geometry raised.
      //
      // Four kinds of white, in the order a viewer on the beach sees them:
      //
      //  - **the lip**: as a crest reaches its break depth it pitches over, and
      //    for a moment there is a bright line along the top of it;
      //  - **the bore**: from there in, the wave is a roller of white water
      //    riding its own front, travelling shoreward with the crest;
      //  - **the lace**: what the roller leaves behind it. Foam on a surf zone
      //    is not a sheet that fades; it tears into a network of bubble lines
      //    that thin and part over several seconds. A ridged noise thresholded
      //    higher as the foam ages does that — fat and continuous right behind
      //    the bore, a net of threads a wave later;
      //  - **the swash edge**: where the water is centimetres thick, at the
      //    waterline and at the front of the sheet running up the sand.
      //
      // The persistent buffer adds the foam the surf leaves standing and
      // drifting between sets. See the shore term in `Wake`.
      const surf = float(0).toVar();
      if (floorDepth !== null) {
        If(inSurf.greaterThan(0.5), () => {
          const d = seabedDepth;
          const s = surfS;
          const age = surfPhaseValue.fract().toVar();

          // Is the nearest crest broken here? Soft over the last few percent of
          // depth so the break line is not a contour.
          const brokenNear = smoothstepDownClamped(d, surfBreak.mul(0.86), surfBreak.mul(1.04)).toVar();

          const bubbles = breakup.mul(0.7).add(0.55).clamp(0, 1).toVar();
          const roller = s
            .smoothstep(-0.07, -0.015)
            .mul(smoothstepDownClamped(s, 0.12, 0.34))
            .mul(brokenNear)
            .mul(bubbles)
            .toVar();

          const lipBand = d.sub(surfBreak).div(surfBreak.mul(0.22).add(0.12)).toVar();
          const lipLine = s.div(0.035).toVar();
          const lip = lipBand
            .mul(lipBand)
            .negate()
            .exp()
            .mul(lipLine.mul(lipLine).negate().exp())
            .toVar();

          // The crest whitening along its top just before it goes.
          const featherLine = s.div(0.02).toVar();
          const feather = smoothstepDownClamped(d, surfBreak.mul(1.05), surfBreak.mul(1.7))
            .mul(brokenNear.oneMinus())
            .mul(featherLine.mul(featherLine).negate().exp())
            .mul(0.45)
            .toVar();

          // Age, in periods, of the youngest broken water at this point: the
          // last crest if it broke here, the one before it otherwise.
          const lastBroke = d.lessThan(surfBreakLast);
          const previousBroke = d.lessThan(surfBreakPrevious);
          const laceAge = lastBroke
            .select(age, previousBroke.select(age.add(1), float(9)))
            .toVar();
          // Drifts shoreward a little as it ages: what is left of a bore is still
          // being carried up the beach by the next.
          const laceUv = worldPos.xz
            .add(shoreward.mul(laceAge.mul(surfUniforms.period).mul(0.35)))
            .mul(0.62)
            .toVar();
          // The network: cell boundaries, whose lines thin as the foam ages.
          // Two scales so it is not a regular honeycomb; a slow patch field so
          // the network tears into rafts and streaks rather than covering the
          // surf zone evenly, and tears further as it ages.
          // Warped, so the cells are stretched and bent like bubble rafts being
          // pulled about by the backwash rather than tiled.
          const warp = vec2(
            valueNoise(laceUv.mul(0.45)),
            valueNoise(laceUv.mul(0.45).add(vec2(5.2, 1.3))),
          )
            .sub(0.5)
            .mul(1.6);
          const warped = laceUv.add(warp).toVar();
          const cellsA = foamCells(warped).toVar();
          const cellsB = foamCells(warped.mul(2.1).add(vec2(7.1, 3.3))).toVar();
          const lineWidth = mix(float(0.3), float(0.05), laceAge.smoothstep(0.03, 1.3)).toVar();
          // Older threads also break: a second noise snips them into segments.
          const snip = valueNoise(warped.mul(1.7).add(vec2(2.2, 9.4)))
            .smoothstep(mix(float(0.0), float(0.45), laceAge.smoothstep(0.3, 1.5)), 0.75)
            .toVar();
          const lines = smoothstepDownClamped(cellsA.y, lineWidth.mul(0.4), lineWidth)
            .max(smoothstepDownClamped(cellsB.y, lineWidth.mul(0.3), lineWidth.mul(0.75)).mul(0.6))
            .mul(snip)
            .toVar();
          const rafts = valueNoise(laceUv.mul(0.22).add(vec2(3.1, 8.7)))
            .smoothstep(mix(float(0.0), float(0.6), laceAge.smoothstep(0.2, 1.6)), 0.85)
            .toVar();
          // Right behind the roller the foam has not torn yet: a sheet, holed by
          // the cells as it starts to part.
          const sheet = smoothstepDownClamped(laceAge, 0.1, 0.42)
            .mul(cellsA.y.smoothstep(0.02, 0.25).mul(0.35).add(0.65))
            .toVar();
          const laceNear = lines
            .mul(rafts)
            .max(sheet)
            .mul(smoothstepDownClamped(laceAge, 1.1, 1.9))
            .mul(bubbles)
            .toVar();
          // Past a few hundred metres the threads are below a pixel; what they
          // average to is a coverage that falls with age.
          const laceFar = smoothstepDownClamped(laceAge, 0.1, 1.4).mul(0.4);
          const lace = mix(laceNear, laceFar, viewDistance.smoothstep(240, 620)).toVar();

          // Centimetres of water: the waterline and the front of the swash. The
          // depth buffer knows how thin the water in front of the sand is,
          // including over the sheet the vertex stage ran up the beach.
          const thin = straightThickness
            .smoothstep(0.004, 0.025)
            .mul(smoothstepDownClamped(straightThickness, 0.03, 0.1))
            .mul(smoothstepDownClamped(seabedSigned, 0.5, 1.4))
            .mul(breakup.mul(0.8).add(0.45).clamp(0, 1))
            .toVar();

          // Seen from far off, a surf zone is a band of white whose individual
          // breakers are a pixel wide. The time-averaged coverage stands in for
          // them there: most of the inner surf zone white most of the time,
          // thinning out to the break line.
          const band = smoothstepDownClamped(d, surfBreak.mul(0.3), surfBreak.mul(1.3))
            .mul(breakup.mul(0.4).add(0.6))
            .mul(0.9)
            .toVar();
          const far = viewDistance.smoothstep(160, 520).toVar();
          // And the line of white at the water's edge, which from more than a
          // few tens of metres is all the swash amounts to: a beach without it
          // reads as a texture boundary. Up close the swash itself is visible
          // and this hands over to it.
          const edgeLine = smoothstepDownClamped(straightThickness, 0.02, 0.3)
            .mul(smoothstepDownClamped(seabedSigned, 0.6, 1.4))
            .mul(viewDistance.smoothstep(25, 110))
            .mul(bubbles)
            .mul(0.75)
            .toVar();
          surf.assign(
            roller
              .max(lip)
              .max(feather)
              .max(lace.mul(0.8))
              .max(thin)
              .max(band.mul(far))
              .max(edgeLine)
              .clamp(0, 1)
              .mul(surfUniforms.strength.min(1)),
          );
        });
      }

      // --- contact foam ------------------------------------------------------
      //
      // Foam where something pierces the surface: the hull's waterline, a buoy,
      // a pier pile, a rock. Water churns and traps air against anything it
      // moves past, so every real object at sea wears a ring of white. Measured
      // from the depth buffer — the water column directly behind this pixel —
      // rather than from any list of objects, so every prop gets it for free.
      //
      // Seabed contact is excluded where the heightfield explains the depth:
      // that is the shore break's job, and treating a whole shallow lagoon as an
      // "object" would foam it edge to edge. Faded with distance because depth
      // precision and the mask's footprint both run out past a couple of
      // hundred metres.
      const contactBand = smoothstepDownClamped(straightThickness, 0.02, 1.3).toVar();
      const notSeabed: any = (
        floorDepth === null
          ? float(1)
          : seabedDepth.sub(straightThickness).smoothstep(0.35, 1.2)
      ).toVar();
      const contactReach = smoothstepDownClamped(viewDistance, 120, 260).toVar();
      const contactFoam = contactBand
        .mul(notSeabed)
        .mul(contactReach)
        // Lace, not an outline: the noise is biased low enough that the ring
        // breaks into patches and gaps, and it is capped short of the crest
        // foam's white. With the bias at +0.25 the ring was solid all the way
        // round and drew the hull's waterline like a cartoon stroke.
        .mul(breakup.mul(1.4).add(perturb.mul(0.9)).sub(0.12).clamp(0, 1))
        .smoothstep(0.24, 0.72)
        .mul(0.85)
        .toVar();

      const foamMask = accumulated
        .max(crestFoam.mul(buffered.oneMinus()))
        .max(surf)
        .max(contactFoam)
        .toVar();

      // Foam is lit, not painted.
      //
      // `mix(surface, foamColor, mask)` toward a constant near-white is what made
      // it read as paint: it has no shading, no structure, and it survives into
      // the night unchanged. Whitecaps are a few centimetres of bubbles — a dense
      // scattering medium, so almost Lambertian, bright but still lit by the same
      // sun as everything else, and shadowed on the side away from it.
      //
      // Lit by the same sun *and sky* as the water around it. The previous form
      // was `foamColor * bubbles * (0.6..1)` with no light term at all, which put
      // a whitecap at roughly 40 % of its albedo — darker than the sky the water
      // beside it was reflecting. At grazing angles, where the surface reflects a
      // bright horizon, every whitecap therefore read as a grey or (at sunset)
      // brown hole in the sea: the hard-edged blotches in every gallery frame.
      // Foam is a near-Lambertian scatterer, so its radiance is albedo times the
      // irradiance arriving on it: a wrapped sun term (bubbles scatter light in
      // from the side) under the cloud shadow, plus the sky dome's ambient.
      const bubbles = breakup.mul(0.22).add(0.84).toVar();
      const foamSun = nDotL.mul(0.55).add(0.3).mul(sunShade.mul(0.7).add(0.3)).toVar();
      const foamSky = (mix(this.uHorizonColor, this.uSkyColor, 0.45) as any).mul(0.42).toVar();
      const foamColor = this.uFoamColor
        .mul(bubbles)
        .mul(
          (this.uSunColor as any)
            .mul(foamSun)
            .mul(this.uLightLevel.min(1.25))
            .add(foamSky),
        )
        .toVar();

      // Foam kills the specular under it. Bubbles are a diffuse rough medium, so a
      // mirror highlight sitting on top of a whitecap is the single most obvious
      // tell that the foam is a decal — and at 15 m/s it happens across half the
      // frame. Not driven fully to zero: the wet film between bubbles does still
      // glint.
      // Where water meets sand it does so at a few centimetres' thickness, and
      // the mesh carrying it cannot follow sand that closely: the edge where the
      // two triangulations cross is a staircase. Fading the reflection and the
      // glint out over the last few centimetres hands that edge to the refracted
      // sand behind it — which is what the water there looks like anyway — so
      // the visible front is the swash foam the ground draws, on an analytic
      // contour, and not the mesh.
      const nearLand = smoothstepDownClamped(seabedSigned, 0.4, 1.2);
      const edgeVisible = mix(float(1), straightThickness.smoothstep(0.0, 0.045), nearLand).toVar();
      const surface = mix(litBody, reflection, fresnel.mul(edgeVisible))
        .add(specular.mul(foamMask.mul(0.88).oneMinus()).mul(edgeVisible))
        .toVar();

      const withFoam = mix(surface, foamColor, foamMask).toVar();

      // --- the surface seen from below: Snell's window and TIR ------------------
      //
      // Looking up from underwater does not show the topside shading. Light
      // arriving from the whole sky hemisphere is refracted into a cone of half
      // angle asin(1/1.333) = 48.6 degrees, so the entire sky — horizon to
      // zenith, in every direction — is compressed into one bright disc directly
      // overhead. Outside that disc nothing from the air can reach the eye at
      // all, and the surface becomes a perfect mirror of the water below it.
      //
      // Until now the material rendered its topside model on the back face, so
      // the underside came out as flat pale facets: no window, no mirror, and a
      // waterline shot from the surface showed the wave field as opaque
      // polygons. This is the one item the claims audit has carried as unbuilt
      // since it was written.
      //
      // The branch is on the *sign* of `n . v`, which is exactly "is this a back
      // face". It is coherent over large regions of the screen, so the cost above
      // water is a compare.
      const facingSign = n.dot(viewDir).toVar();
      If(facingSign.lessThan(0).and(this.uEyeSubmerged.greaterThan(0.5)), () => {
        const cosI = facingSign.abs().clamp(0, 1).toVar();

        // Snell: sin(t) = ior * sin(i). Past sin(t) = 1 there is no transmitted
        // ray and everything is reflected.
        const sinT2 = float(1)
          .sub(cosI.mul(cosI))
          .mul(WATER_IOR * WATER_IOR)
          .toVar();
        const insideWindow = smoothstepDownClamped(sinT2, 0.88, 1.0).toVar();

        // Where in the sky this ray came from — through Snell, not through a
        // linear remap of the incidence cosine.
        //
        // `sin(t) = ior * sin(i)` gives the *air-side* angle, and the sky's
        // elevation is what that angle subtends from vertical. The two are not
        // proportional: the outer third of the window's radius holds more than
        // half the sky, which is exactly why a real Snell window is a bright disc
        // with the whole horizon crammed into a rim around it. A linear remap
        // spreads the sky evenly and loses that.
        const cosT = float(1).sub(sinT2.min(1)).max(0).sqrt().toVar();
        const windowColor = mix(this.uHorizonColor, this.uSkyColor, cosT)
          .mul(this.uLightLevel.mul(0.85).add(0.15))
          .mul(1.9)
          .toVar();

        // Outside the window: a mirror, and it now behaves like one.
        //
        // The first version put an authored colour here — `litBody` scaled — and
        // called it total internal reflection. It is not: a mirror shows *the
        // scene*, and the whole reason a diver notices the ceiling outside the
        // window is that the reef, the hull and the seabed appear upside down in
        // it. So the reflected direction now drives a lookup into the same
        // backdrop the refraction already reads.
        //
        // **A screen-space offset, not a traced reflection.** There is no march
        // and no depth test: the reflected direction is projected to a uv
        // displacement and sampled once, which is the same approximation the
        // refraction above uses and carries the same error — what it returns is
        // whatever happens to be at that screen position, not what the reflected
        // ray would actually hit. It is right for a distant, roughly planar
        // ceiling and wrong for anything with parallax. A real trace is
        // `ScreenSpaceReflection`, and pointing it downward from a submerged
        // camera is the honest upgrade.
        //
        // Off-screen rays fall back to the water's own body colour, which is what
        // an infinite column of water looks like — and is what the previous
        // version showed everywhere.
        const reflectedDir = viewDir.negate().reflect(n).toVar();
        const tirOffset = reflectedDir.xz
          .mul(this.uRefractionStrength.mul(1.6))
          .div(viewDistance.mul(0.05).add(1))
          .toVar();
        // The in-frame test is on the *raw* offset uv, before clamping.
        //
        // `viewportSafeUV` pulls an out-of-range coordinate back inside the
        // frame, so testing its output asks "is this uv on screen" of something
        // that has just been made on screen by construction — the fallback could
        // never engage. The raw coordinate is the one that knows whether the
        // reflected ray left the frame.
        const tirRawUv = screenUV.add(tirOffset).toVar();
        const tirUv: any = (viewportSafeUV(tirRawUv) as any).toVar();
        const tirEdge = tirRawUv.min(tirRawUv.oneMinus()).toVar();
        const tirInFrame = tirEdge.x.min(tirEdge.y).smoothstep(0, 0.06).clamp(0, 1).toVar();
        const tirScene = vec3(viewportSharedTexture(tirUv)).toVar();

        const mirrored = mix(
          litBody.mul(0.88).add(scatter.mul(0.35)),
          tirScene,
          tirInFrame.mul(0.8),
        ).toVar();

        // Foam floats *on* the surface and is visible from beneath as a dark
        // patch against the window, because it scatters the sky away before it
        // can be refracted through.
        const underside = mix(mirrored, windowColor, insideWindow).toVar();
        withFoam.assign(mix(underside, foamColor.mul(0.55), foamMask.mul(0.8)));
      });

      // --- aerial perspective --------------------------------------------------------
      const fogFactor = float(1)
        .sub(this.uFogDensity.mul(viewDistance).negate().exp())
        .clamp(0, 1)
        .toVar();
      const withFog = mix(withFoam, this.uFogColor, fogFactor).toVar();

      return vec4(withFog, 1);
    })();
  }
}

/**
 * Weight for cascade `index` at world distance `distance` (metres).
 *
 * Keyed to real distance, NOT to the mesh's radial parameter: ring radii grow
 * geometrically, so a linear cutoff in radial parameter would kill the finest
 * cascade only a few metres from the camera.
 *
 * Each cascade is carried until its texel footprint approaches a pixel, then
 * cross-faded out. Cascade 0 holds the swell and never fades — it is what forms
 * the horizon silhouette.
 *
 * TSL node objects are structurally dynamic, so node-typed values are `any` here
 * by design — the module's public API stays strongly typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Hash-based 2D value noise, bilinearly interpolated with a quintic fade.
 *
 * Procedural rather than a sampled texture so there is nothing extra to
 * download, and so the foam breakup is scale-free — it stays crisp no matter how
 * close the camera gets.
 */
/**
 * Rain striking the water, as a slope perturbation.
 *
 * Procedural against a lattice rather than a simulated field. Each cell of a
 * world-space grid launches one ring per cycle, at a position and a phase hashed
 * from the cell index, so impacts are scattered in space and time without any
 * state being stored anywhere. That matters more than it sounds: a buffer would
 * have to be camera-relative, and every impact near the edge would pop in and
 * out as the viewer moved. Anchored to the world by construction, a ripple stays
 * where it landed.
 *
 * The spec allows "a shared, camera-relative, low-resolution GPU field or another
 * bounded technique"; this is the second. It is bounded by the 3x3 neighbourhood,
 * costs no memory, needs no readback, and is deterministic — the same second of
 * simulation time produces the same rain on every machine.
 *
 * Slope is accumulated directly rather than height. The normal is what the
 * shading actually wants, and differentiating a height field would mean
 * evaluating this three times per fragment instead of once.
 */

/**
 * Metres per lattice cell, and seconds between a cell's impacts.
 *
 * These two set the impact rate — one strike per cell per period, so at full
 * intensity the surface takes `1 / (RAIN_CELL² × RAIN_PERIOD)` hits per square
 * metre per second. That has to agree with how much rain is visibly falling, or
 * the scene shows a downpour landing as a drizzle. `Weather` renders 9000
 * streaks through a 68 m box; matching the two by eye at full intensity puts the
 * cell near a metre and the period near a second, not the 2.6 m and 1.35 s this
 * started with — which was roughly seven times too few impacts.
 */
const RAIN_CELL = 1.05;
const RAIN_PERIOD = 0.95;
/**
 * Furthest a ring travels before its cell fires again, in cells.
 *
 * `RAIN_SPEED * RAIN_PERIOD` is 1.43 m against a 1.05 m cell, so a ring can
 * reach 1.36 cells from where it landed and the 3x3 neighbourhood the loop walks
 * does not contain every drop that can touch a given fragment. A review caught
 * that; the visible consequence is a discontinuity as a fragment crosses a cell
 * boundary and a ring pops in or out.
 *
 * Rather than widen the loop to 5x5 — which more than doubles the cost for
 * contributions that are almost entirely faded — the ring's envelope is closed
 * off before it can leave the neighbourhood. `exp(-1.6 r)` is already down to 4%
 * of its peak at one cell, so bounding it here changes the image far less than
 * the boundary artefact it removes.
 */
const RAIN_MAX_RADIUS = RAIN_CELL * 0.95;
/** Metres per second the ring expands. */
const RAIN_SPEED = 1.5;
/** Radians per metre across the ring — how tight the wavefront reads. */
const RAIN_FREQUENCY = 13.0;

const rainSlope = /*@__PURE__*/ Fn(([worldXZ, time, intensity]: [any, any, any]) => {
  const cell = worldXZ.div(RAIN_CELL).floor().toVar();
  const slope = vec2(0, 0).toVar();

  for (let dz = -1; dz <= 1; dz++) {
    for (let dx = -1; dx <= 1; dx++) {
      const neighbour = cell.add(vec2(dx, dz)).toVar();
      const h = hash2(neighbour).toVar();

      // Where in the cell the drop landed, and how far through its cycle it is.
      const drop = neighbour.add(h).mul(RAIN_CELL).toVar();
      const age = time.div(RAIN_PERIOD).add(h.x.mul(7.13).add(h.y.mul(3.71))).fract()
        .mul(RAIN_PERIOD).toVar();

      // Rain rate sets how many cells are raining, not how hard each ring hits.
      // Scaling amplitude instead would make light rain look like heavy rain
      // seen through fog; scaling population is what actually changes.
      //
      // Written as an explicit comparison rather than `h.y.step(intensity)`.
      // TSL's method form reverses the arguments — `x.step(e)` is `step(e, x)`,
      // which is `x >= e` — so that spelling activated a cell when its hash
      // *exceeded* the rain rate. Cells switched on as the rain eased and at full
      // intensity none were active at all, the exact inverse of the intent. It
      // survived its own test because that test only asked whether rain changed
      // the surface, and an inverted mask changes it just as much.
      const active = intensity.greaterThan(h.y).select(float(1), float(0)).toVar();

      const delta = worldXZ.sub(drop).toVar();
      const r = delta.length().max(1e-3).toVar();
      const front = age.mul(RAIN_SPEED).toVar();
      const x = r.sub(front).toVar();

      // A wave packet at the expanding front, dying with age and with distance.
      // Closed off before the ring can leave the 3x3 neighbourhood — see
      // `RAIN_MAX_RADIUS`. Without it a ring that outruns the window vanishes
      // abruptly at a cell boundary instead of fading.
      const reach = float(1).sub(r.div(RAIN_MAX_RADIUS)).clamp(0, 1).toVar();
      const envelope = x
        .mul(x)
        .mul(-9.0)
        .exp()
        .mul(float(1).sub(age.div(RAIN_PERIOD)))
        .mul(r.mul(-1.6).exp())
        .mul(reach.mul(reach))
        .toVar();

      // d/dr of sin(x * F) * envelope, keeping the dominant term.
      const dhdr = x.mul(RAIN_FREQUENCY).cos().mul(RAIN_FREQUENCY).mul(envelope).toVar();
      slope.addAssign(delta.div(r).mul(dhdr).mul(active));
    }
  }
  return slope;
});

const hash2 = /*@__PURE__*/ Fn(([p]: [any]) => {
  const h = vec2(p.dot(vec2(127.1, 311.7)), p.dot(vec2(269.5, 183.3))).toVar();
  return h.sin().mul(43758.5453).fract();
});

const valueNoise = /*@__PURE__*/ Fn(([p]: [any]) => {
  const i = p.floor().toVar();
  const f = p.fract().toVar();
  // Quintic fade — a linear blend leaves visible grid creases in the derivative.
  const u = f.mul(f).mul(f.mul(f.mul(6).sub(15)).add(10)).toVar();

  const a = hash2(i).x.toVar();
  const b = hash2(i.add(vec2(1, 0))).x.toVar();
  const c = hash2(i.add(vec2(0, 1))).x.toVar();
  const d = hash2(i.add(vec2(1, 1))).x.toVar();

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
});

/**
 * Geometry and shading need *different* LOD curves, and conflating them is what
 * makes naive ocean meshes sparkle.
 *
 * Geometry can only carry a wave if the local vertex spacing resolves it, and
 * that limit is now enforced where it belongs — in the vertex stage, as a mip
 * level computed from the spacing itself. See the displacement loop in `build`
 * and `ocean/meshSampling`.
 *
 * The fixed-metre table that used to stand here is gone. It had the right idea
 * and could not act on it: expressed in absolute distances, it knew nothing
 * about the tier's mesh density, so the same 18-55 m ripple ramp served a 128
 * ring mesh and a 512 ring one whose spacings differ by four times. The level
 * knows, because spacing is what it is computed from.
 *
 * Shading has no such limit: the derivative textures are mipmapped, so the
 * fragment stage samples them correctly at any distance. Normals therefore carry
 * the detail far beyond where the geometry has flattened out, which is exactly
 * how real distant water reads — a smooth sheet with fine specular structure.
 */
const CASCADE_SHADING_FADE_METRES: [number, number][] = [
  [Infinity, Infinity], // swell: always on
  [1400, 3000],
  [160, 420],
];

function fadeFrom(
  table: [number, number][],
  distance: any,
  index: number,
  count: number,
): any {
  if (count === 1 && index === 0) return float(1);
  const [start, end] = table[Math.min(index, table.length - 1)];
  if (!Number.isFinite(start)) return float(1);
  // smoothstep rises 0 -> 1 with distance, so invert it to get a fade-out.
  return float(1).sub(distance.smoothstep(start, end)).clamp(0, 1);
}

/** Fragment-stage weight: how much this cascade contributes to normals and foam. */
function cascadeShadingFade(distance: any, index: number, count: number): any {
  if (index === 0) return float(1);
  return fadeFrom(CASCADE_SHADING_FADE_METRES, distance, index, count);
}
