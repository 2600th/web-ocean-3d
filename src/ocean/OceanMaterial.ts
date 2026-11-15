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
  pmremTexture,
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
  featherFoam,
  foamPatchiness,
  foamCrestPatches,
  foamPattern,
  foamStreaks,
  foamUniforms,
  setFoamDetail,
  setFoamWind,
} from './FoamPattern';
import {
  breaker,
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
  /**
   * Shading-only slope fields below the ripple cascade's resolution. See
   * `Spectrum.detailCascades`. Optional so a caller without them gets the old
   * three-band surface, weighted rather than rebuilt.
   */
  detailDerivativeTextures?: THREE.Texture[];
  detailTileSizes?: number[];
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

/** Detail slots the shader is built with. Matches `Spectrum.detailCascades`. */
const MAX_DETAIL = 1;

/**
 * Rotation of the detail band's second read, radians.
 *
 * The band's tile is half a metre at High, short enough that its repeat could be
 * seen in the glitter at a few metres. Summing it with a copy of itself rotated
 * by an angle that is not a rational fraction of a turn makes the sum aperiodic:
 * the two lattices never come back into register. The band is near-isotropic
 * (Hasselmann's spreading exponent is below 0.01 at these wavelengths), so the
 * rotated copy is statistically the same sea. Each read is scaled by 1/sqrt(2)
 * so the pair carries the band's slope variance once, not twice.
 */
const DETAIL_ROTATION = 1.1071487; // atan(2)

/**
 * The cascade whose slope the sky reflection treats as statistics at grazing
 * view: the 16 m tile, wavelengths 5 cm to 3 m (`Spectrum.CASCADES[2]`).
 * Together with the detail bands it is the "fine" share of the slope.
 */
const FINE_CASCADE = 2;

/**
 * Smith's `a` for the fine slopes (cot of the view elevation over sqrt(2)
 * sigma) at which the sky reflection starts, and finishes, handing their mean
 * slope back to the normal. Below the first the fine slopes are pure roughness.
 *
 * Measured on `waves-1` and `reflections-2`, where the grey "camo" and the
 * black sunset blotches were traced to this cascade alone: with it zeroed the
 * frames were clean and wave-shaped, with cascades 0 and 1 zeroed instead they
 * were not. A facet slope comparable to the view elevation swings Schlick
 * from 0.3 to 1 and the reflected ray from the horizon to 25 degrees up, so a
 * per-pixel normal turns the short waves' *envelope* (where their slopes
 * happen to be large) into hard patches. The mean over the facets a grazing
 * eye sees is smooth, and that is what a statistical treatment returns.
 */
const FINE_GRAZING_A0 = 1.2;
const FINE_GRAZING_A1 = 3.5;

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

/**
 * How dense *aged* open-sea foam is drawn, as a fraction of its buffer amount.
 *
 * The foam buffer's sea channel is calibrated on its *mean* against Monahan's
 * whitecap coverage (`tests/foam.spec.ts`), and it gets there by saturating:
 * the fold regions of a 47 m swell are tens of metres wide and the deposit
 * fills them to 1 within a few frames. Drawn at face value, every one of those
 * patches was a flat white sheet (the critic's waves-4 and foam-3) or, far off,
 * an opaque oval (foam-2) for as long as the 1.5 s decay held it near 1.
 *
 * Monahan's own distinction is the way out: the actively breaking crest (stage
 * A) is dense, the decaying patch behind it (stage B) covers about ten times the
 * area at a fraction of the density. The buffer's freshness channel
 * (`Wake`, `FRESH_TAU`) says which is which, so fresh foam is drawn at full
 * density and aged foam at this fraction, torn into web-ocean's wind streaks.
 * 0.45 puts a saturated stage-B patch at a little under half white once the
 * streaks have pulled it into lanes: lace with clear water through it, which is
 * what the references show (0.55 still read as sheets in waves-4).
 *
 * The cost is honest and stated: at saturation the *rendered* white fraction is
 * under half the buffer's, and the wash below 0.3 is not drawn as white at all
 * (see `agedCover`), so the frame shows less whitecap than the coverage test
 * measures in the buffer. The test still guards the physics (deposit and decay
 * are unchanged); this is how the coverage is drawn.
 */
const AGED_FOAM_DENSITY = 0.45;

const SHORE_GRADIENT_EPSILON = 12;

/**
 * Which way is up the beach, and how steep the bottom is, at `xz`.
 *
 * The gradient of *depth* points out to sea, so its negation points up the
 * beach. Forward differences over `SHORE_GRADIENT_EPSILON`: wide enough to
 * average out the heightfield's own metre-scale relief, narrow enough to follow
 * a coastline that turns. Degenerate on a flat bottom, where `fallback` (the
 * wind bearing) is the honest answer. Shared by both stages so the vertex
 * stage's lean and the fragment stage's shading of it agree exactly.
 *
 * @returns vec3(shoreward.x, shoreward.z, bottom slope as rise over run)
 */
function shoreBearing(floorDepth: (p: any) => any, xz: any, depthHere: any, fallback: any): any {
  const e = SHORE_GRADIENT_EPSILON;
  const dDepthX = floorDepth(vec3(xz.x.add(e), 0, xz.y)).sub(depthHere);
  const dDepthZ = floorDepth(vec3(xz.x, 0, xz.y.add(e))).sub(depthHere);
  const gradient = vec2(dDepthX, dDepthZ).toVar();
  const steepness = gradient.length().toVar();
  const direction = mix(fallback, gradient.negate().div(steepness.max(1e-4)), steepness.smoothstep(0.02, 0.12));
  return vec3(direction.x, direction.y, steepness.div(e));
}

/**
 * How much of the breakers' pitch the mesh can carry at `groundDistance`.
 *
 * The lean is a bump about two metres wide, so it needs vertices a fraction of
 * that apart; past ~150 m the radial grid stops resolving it and the offset
 * would alias into a sawtooth along the crest. Inside `SURF_VERTEX_REACH` on
 * purpose, and evaluated from the same distance by both stages.
 */
function surfPitchHold(groundDistance: any): any {
  return smoothstepDownClamped(groundDistance, SURF_VERTEX_REACH * 0.45, SURF_VERTEX_REACH * 0.7);
}

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
 * How much of the FFT field survives into the surf, per band, at seabed depth
 * `depth`: `vec2(swell, chop)`, each 0..1.
 *
 * The open-ocean spectrum knows nothing about the bottom. A wave feels it once
 * the depth falls below about half its wavelength, and Crest attenuates each
 * band by exactly that, `saturate(2h / lambda)`. The swell cascade's waves are
 * tens of metres long, so across a surf zone a few metres deep they are the
 * waves that shoal and break — which is what `Surf` draws — and leaving them on
 * as well put the open sea's swell straight through the breakers. At 15 m/s
 * that swell has a significant height of 5.5 m against a 2 m breaker, and with
 * one uniform cut for every band (the previous form, which kept 60-100% of it
 * across most of the surf zone) the breakers simply could not be seen. So the
 * swell is handed over to the surf almost completely, over the surf's own
 * reach, while the short chop — which does ride over a surf zone — keeps a
 * third of itself so the shallows are not glass.
 *
 * Bounded by `surfReachDepth` rather than by each band's wavelength: past the
 * reach there is no surf to replace the swell with, and attenuating it there
 * would flatten the play area's own shallow plateau, which open-sea whitecap
 * coverage is measured on.
 */
function surfFftFade(depth: any): any {
  const swell = mix(float(0.06), float(1), depth.smoothstep(0.5, surfReachDepth()));
  const chop = mix(float(0.35), float(1), depth.smoothstep(0.3, surfReachDepth().mul(0.7)));
  return vec2(swell, chop);
}

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
  // Detail bands, re-pointed by `setCascades` like the cascades above.
  private readonly detailNodes: any[] = [];
  private readonly uDetailTiles: any[] = [];
  private readonly uDetailWeights: any[] = [];
  /**
   * The sky environment, prefiltered (PMREM). See `setEnvironment`.
   *
   * Built on a placeholder: the atmosphere creates its cube after this graph,
   * and a PMREM node re-points by `.value` without recompiling.
   */
  private envNode: any = null;
  private envPlaceholder: THREE.Texture | null = null;
  /** 0 until a real environment is bound; the analytic gradient stands in. */
  private readonly uEnvReady = uniform(0);
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
  /** Drift of floating foam, metres. See `setFoamDrift`. */
  private readonly uFoamDrift = uniform(new THREE.Vector2());
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
    // What breaks on a beach is the swell that reaches it, less than the open
    // sea's significant height — refraction spreads it and the shelf takes
    // energy out before the beach. A fraction of the sea state rather than the
    // sea state: 0.9 m at 6 m/s, 1.2 m at 9, 2.3 m at 15, capped at 3.6 m from
    // about 20 m/s, which is the most this beach's 1:9 face can break without
    // the surf zone running out past the shelf it sits on.
    const height = Math.min(3.6, (this.uWaveScale.value as number) * 0.3 + 0.6);
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
    // The foam pattern's streaks lie along the same axis; it is shared with the
    // island's ground, which draws the swash foam, so it lives in `FoamPattern`.
    setFoamWind(bearingRadians);
  }

  /**
   * Tier knob for how much of the foam's structure is drawn. See
   * `FoamPattern.foamUniforms.detail`; the surf's pitch is gated on it too,
   * because it costs two extra heightfield evaluations per surf vertex.
   */
  setFoamDetail(level: number): void {
    setFoamDetail(level);
  }

  /**
   * How far the sea's surface current has carried floating foam, metres, as a
   * world offset. The foam buffer advects its *amount* by the same current
   * (`Wake.setDrift`); this moves the *pattern* with it, so a patch's bubbles
   * travel with the patch instead of the patch sliding through a pattern
   * nailed to the world. Integrated by the caller from the sea clock, so a
   * deterministic rewind puts it back.
   */
  setFoamDrift(x: number, z: number): void {
    (this.uFoamDrift.value as THREE.Vector2).set(x, z);
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

  /**
   * The sky the water reflects: the atmosphere's environment cube, which
   * `Atmosphere.updateEnvironment` assigns to `scene.environment`.
   *
   * Call whenever that may have changed; it is a no-op for the same texture.
   * The cube is prefiltered by three's PMREM chain (shared with every PBR
   * material through PMREMNode's per-texture cache), so a rough patch of sea
   * reflects a blurred sky and a calm one a sharp sky, as web-ocean's
   * roughness-filtered `envCube` mips do.
   */
  setEnvironment(environment: THREE.Texture | null): void {
    if (environment === null || this.envNode === null) return;
    if (this.envNode.value === environment) return;
    this.envNode.value = environment;
    this.uEnvReady.value = 1;
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
    const details = fields.detailDerivativeTextures ?? [];
    const detailTiles = fields.detailTileSizes ?? [];
    for (let i = 0; i < MAX_DETAIL; i++) {
      const has = i < details.length;
      this.detailNodes[i].value = has ? details[i] : derivativeTextures[derivativeTextures.length - 1];
      this.uDetailTiles[i].value = has ? detailTiles[i] : 1;
      this.uDetailWeights[i].value = has ? 1 : 0;
    }
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
    const details = inputs.detailDerivativeTextures ?? [];
    const detailTiles = inputs.detailTileSizes ?? [];
    for (let i = 0; i < MAX_DETAIL; i++) {
      const has = i < details.length;
      this.detailNodes.push(texture(has ? details[i] : derivativeTextures[derivativeTextures.length - 1]) as any);
      this.uDetailTiles.push(uniform(has ? detailTiles[i] : 1));
      this.uDetailWeights.push(uniform(has ? 1 : 0));
    }
    // A cube that PMREMNode accepts at compile time (six faces present), so the
    // graph compiles with the render-target sampling convention the real
    // environment, also a cube render target, will have. See `setEnvironment`.
    this.envPlaceholder = new THREE.CubeRenderTarget(4).texture;

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
      /** Cascade 0's share of it: the swell, which the surf replaces. */
      const swellDisplacement = vec3(0).toVar();

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
        if (i === 0) swellDisplacement.assign(displacement);
      }

      displacement.mulAssign(this.uDisplacementScale);
      swellDisplacement.mulAssign(this.uDisplacementScale);

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
              const fade = surfFftFade(depth).toVar();
              displacement.assign(
                displacement.sub(swellDisplacement).mul(fade.y).add(swellDisplacement.mul(fade.x)),
              );
              // The pitch needs to know which way is up the beach and how steep
              // the bottom is, which costs two more heightfield evaluations — the
              // same forward differences the fragment stage takes, at the same
              // undisplaced point, so the two agree on every lean exactly. Low
              // skips both and the breakers only rise. See `Surf.breaker`.
              const shoreward = vec2(this.uWindAxis.x, this.uWindAxis.y).toVar();
              const beachSlope = float(0.1).toVar();
              const pitch = float(0).toVar();
              If(foamUniforms.detail.greaterThan(0.5), () => {
                const bearing = shoreBearing(floorDepth, worldXZ, depth, shoreward);
                shoreward.assign(bearing.xy);
                beachSlope.assign(bearing.z);
                pitch.assign(surfPitchHold(groundDistance));
              });
              const wave = breaker(depth, worldXZ, beachSlope, pitch);
              displacement.y.addAssign(wave.eta);
              displacement.x.addAssign(shoreward.x.mul(wave.lean));
              displacement.z.addAssign(shoreward.y.mul(wave.lean));
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
      /**
       * The short-wave share of the slope (the finest cascade and the detail
       * bands), and the part of their tile-wide slope variance this pixel
       * resolves rather than loses. The sky reflection and the Fresnel turn it
       * into roughness at grazing view; see the reflection block.
       */
      const fineSlope = vec2(0).toVar();
      const fineVariance = float(0).toVar();
      /** Cascade 0's share of the slope: the swell, which the surf replaces. */
      const swellSlope = vec2(0).toVar();
      for (let i = 0; i < cascadeCount; i++) {
        const d = this.derivativeNodes[i].sample(fieldXZ.div(this.uTileSizes[i])).toVar();
        // Weight folds into the fade, so a cascade the tier does not use
        // contributes no slope and — via `mix` toward 1 — no folding either.
        const fade = cascadeShadingFade(viewDistance, i, cascadeCount).mul(
          this.uCascadeWeights[i],
        );
        slope.addAssign(vec2(d.x, d.y).mul(fade));
        if (i === 0) swellSlope.assign(slope);
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
        if (i === FINE_CASCADE) {
          // See `fineSlope`. The top of the mip chain is the field's mean |s|^2
          // over the whole tile: its slope variance, the same everywhere.
          fineSlope.addAssign(vec2(d.x, d.y).mul(fade));
          const tileVariance = this.derivativeNodes[i]
            .sample(fieldXZ.div(this.uTileSizes[i]))
            .level(16).w;
          fineVariance.addAssign(
            tileVariance
              .sub(d.w.sub(d.x.mul(d.x).add(d.y.mul(d.y))).max(0))
              .max(0)
              .mul(fade)
              .mul(fade),
          );
        }
      }

      // The detail bands (`Spectrum.detailCascades`): centimetre and millimetre
      // ripples, shading only. Their mean slope adds to the normal and their
      // sub-footprint variance to the glitter's roughness, exactly as the
      // cascades above, so far away they become sheen and up close glints.
      // Kept apart from `lostSlopeVariance` because these short waves are
      // spread almost evenly over every heading (Hasselmann's s falls below
      // 0.01 under a metre), so their variance splits evenly between the axes
      // rather than by the wind's anisotropy.
      const lostDetailVariance = float(0).toVar();
      const cosR = Math.cos(DETAIL_ROTATION);
      const sinR = Math.sin(DETAIL_ROTATION);
      for (let i = 0; i < MAX_DETAIL; i++) {
        const weight = this.uDetailWeights[i].mul(Math.SQRT1_2);
        const uvA = fieldXZ.div(this.uDetailTiles[i]);
        const a = this.detailNodes[i].sample(uvA).toVar();
        // Rotated read: sample at R^T x, and rotate the slope it returns by R.
        const uvB = vec2(
          fieldXZ.x.mul(cosR).add(fieldXZ.y.mul(sinR)),
          fieldXZ.y.mul(cosR).sub(fieldXZ.x.mul(sinR)),
        )
          .div(this.uDetailTiles[i])
          .add(vec2(0.37, 0.71));
        const b = this.detailNodes[i].sample(uvB).toVar();
        const bSlope = vec2(b.x.mul(cosR).sub(b.y.mul(sinR)), b.x.mul(sinR).add(b.y.mul(cosR)));
        slope.addAssign(vec2(a.x, a.y).add(bSlope).mul(weight));
        // Variance scales with the square of the 1/sqrt(2) weight.
        const detailLost = a.w
          .sub(a.x.mul(a.x).add(a.y.mul(a.y)))
          .max(0)
          .add(b.w.sub(b.x.mul(b.x).add(b.y.mul(b.y))).max(0))
          .mul(this.uDetailWeights[i])
          .mul(0.5)
          .toVar();
        lostDetailVariance.addAssign(detailLost);
        // The rotated copy reads the same texture, so the pair's tile variance
        // is the texture's: (var(a) + var(b)) / 2 under the 1/sqrt(2) weights.
        fineSlope.addAssign(vec2(a.x, a.y).add(bSlope).mul(weight));
        fineVariance.addAssign(
          this.detailNodes[i].sample(uvA).level(16).w.mul(this.uDetailWeights[i]).sub(detailLost).max(0),
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
      /** How much of the open-sea swell is left here, 0..1. See `surfFftFade`. */
      const surfSwellLeft = float(1).toVar();
      /** `Surf.BreakerSample.compression`: 1 on still water, toward 0.15 on a pitching face. */
      const surfCompression = float(1).toVar();
      /** How close the nearest crest is to breaking, 0..1. */
      const surfSteep = float(0).toVar();
      /** Surf similarity number of the nearest crest; spilling below ~0.4. */
      const surfXi = float(1).toVar();
      /** Thin, steep, unbroken water near a breaker's crest: it glows. */
      const surfThin = float(0).toVar();
      if (floorDepth !== null) {
        If(seabedSigned.lessThan(surfReachDepth()), () => {
          inSurf.assign(1);
          const bearing = shoreBearing(floorDepth, fieldXZ, seabedSigned, shoreward).toVar();
          shoreward.assign(bearing.xy);

          // The same pitch the vertex stage displaced by: same tier gate, same
          // distance hold (the mesh is centred on the eye, so ground distance is
          // the horizontal distance to it), same slope.
          const groundDistance = fieldXZ.sub(vec2(cameraPosition.x, cameraPosition.z)).length();
          const pitch = foamUniforms.detail
            .greaterThan(0.5)
            .select(surfPitchHold(groundDistance), float(0));
          const wave = breaker(seabedDepth, fieldXZ, bearing.z, pitch);
          surfPhaseValue.assign(wave.phase);
          surfS.assign(wave.s);
          surfBreak.assign(wave.breakDepth);
          surfBreakLast.assign(wave.breakDepthLast);
          surfBreakPrevious.assign(wave.breakDepthPrevious);
          surfCompression.assign(wave.compression);
          surfSteep.assign(wave.steep);
          surfXi.assign(wave.xi);
          // A face about to go is a thin sheet of water stood up against the
          // light: the crest and the upper front face of an unbroken, steep
          // wave. It is what makes a breaker read as water rather than as a
          // white line, and the crest-glow term below already knows what to do
          // with thin water; it only had no way to know a breaker is thin.
          const faceLine = wave.s.add(0.03).div(0.06);
          surfThin.assign(
            faceLine
              .mul(faceLine)
              .negate()
              .exp()
              .mul(wave.steep)
              .mul(seabedDepth.smoothstep(wave.breakDepth.mul(0.85), wave.breakDepth))
              .mul(surfUniforms.strength.min(1)),
          );

          const fade = surfFftFade(seabedDepth).toVar();
          slope.assign(slope.sub(swellSlope).mul(fade.y).add(swellSlope.mul(fade.x)));
          surfSwellLeft.assign(fade.x);
          // d(eta)/dx along the beach = d(eta)/d(phase) * d(phase)/dx. Faded with
          // distance like the finer cascades: a breaker face is a few metres
          // across, so past a few hundred metres it is shading noise.
          //
          // Divided by the pitch's compression: the vertex stage moved the crest
          // shoreward over the face, so a metre of undisplaced surface became
          // `compression` metres of displaced surface and the face rose over
          // that much less run. This is what stands the face up in the shading;
          // the displacement alone only moves where it is.
          const reach = smoothstepDownClamped(viewDistance, 260, 700);
          slope.addAssign(
            shoreward
              .mul(wave.etaPerPhase.mul(surfPhaseGradient(seabedDepth)).div(wave.compression))
              .mul(reach),
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

      // --- slope statistics: LEAN, after Noniv/web-ocean (MIT, ocean.js) -------
      //
      // Everything that reflects (the sun, the sky, the hull) is decided by the
      // distribution of facet slopes inside the pixel, not by one normal. That
      // distribution is a mean (`slope`, from the mipmapped fields) and a 2x2
      // covariance built from three parts, exactly as web-ocean builds `cov`:
      //
      //  - **unresolved**: roughness below the finest band, 0.0024 per axis at
      //    this material's default roughness (web-ocean `waterCol.w`), plus the
      //    0.0035 it ramps in from 30 to 500 m "to reach Cox-Munk-like totals in
      //    the far sheen";
      //  - **lost to filtering**: E[s^2] - E[s]^2 from the mip chains (LEAN). The
      //    geometric cascades split theirs along/across the wind by the Cox-Munk
      //    ratio, because only the isotropic second moment is stored; the
      //    near-isotropic detail bands split theirs evenly;
      //  - **screen-space**: the change of mean slope across the pixel, from its
      //    derivatives (web-ocean's specular anti-aliasing), capped at 0.03 as
      //    web-ocean caps it.
      //
      // Built in the wind frame, where a wind sea's slope distribution is
      // diagonal by symmetry, and rotated to world xz.
      const windU = vec2(this.uWindAxis.x, this.uWindAxis.y).toVar();
      const shareU = this.uSlopeAnisotropy.div(this.uSlopeAnisotropy.add(1)).toVar();
      const roughScale = this.uRoughness.div(0.075).pow(2).toVar();
      const unresolved = float(0.0024)
        .mul(roughScale)
        .add(viewDistance.smoothstep(30, 500).mul(0.0035))
        .toVar();
      const varU = unresolved.add(lostSlopeVariance.mul(shareU)).add(lostDetailVariance.mul(0.5)).toVar();
      const varC = unresolved
        .add(lostSlopeVariance.mul(shareU.oneMinus()))
        .add(lostDetailVariance.mul(0.5))
        .toVar();
      const slopeCov = vec3(
        varU.mul(windU.x.mul(windU.x)).add(varC.mul(windU.y.mul(windU.y))),
        varU.mul(windU.y.mul(windU.y)).add(varC.mul(windU.x.mul(windU.x))),
        varU.sub(varC).mul(windU.x.mul(windU.y)),
      ).toVar();
      const dSlopeX = dFdx(slope).toVar();
      const dSlopeY = dFdy(slope).toVar();
      const screenCov = vec3(
        dSlopeX.x.mul(dSlopeX.x).add(dSlopeY.x.mul(dSlopeY.x)),
        dSlopeX.y.mul(dSlopeX.y).add(dSlopeY.y.mul(dSlopeY.y)),
        dSlopeX.x.mul(dSlopeX.y).add(dSlopeY.x.mul(dSlopeY.y)),
      )
        .mul(0.5)
        .toVar();
      slopeCov.addAssign(screenCov.mul(float(0.03).div(screenCov.x.max(screenCov.y).max(1e-8)).min(1)));

      // --- Fresnel ---------------------------------------------------------------
      //
      // web-ocean's: Schlick (F0 0.02, n = 1.333) on a normal flattened 45% toward
      // vertical, "the long-wave facet plus the grazing-angle statistics of
      // everything finer (visible facets at grazing incidence stay highly
      // reflective)". It replaces the `GRAZING_SLOPE_SIGMA` fade that stood here,
      // which existed because the reflection used to be a fixed horizon colour:
      // at a grazing view that colour is the brightest in the sky, so the far sea
      // went white and had to be held down by hand. The reflection below follows
      // the real reflected ray into the environment, tilted toward the facets a
      // grazing eye actually sees, so the far sea reflects the greyer sky above
      // the horizon by itself and the fade has nothing left to correct.
      //
      // **Evaluated on the sky's normal, `nSky`, not on `n`.** At grazing view
      // the fine slopes (`FINE_CASCADE` and the detail bands) are as steep as
      // the view is low, and both this term and the reflected ray are steep
      // functions there: a per-pixel normal turned the envelope of the short
      // waves into hard grey patches by day ("camo", waves-1/2, foam-2/3) and
      // near-black ones at sunset (reflections-2/4, sky-2/4). So as the view
      // drops below a few sigma of the fine slopes (Smith's `a`, as in
      // `smithLambda` below), their mean slope leaves the normal and their
      // tile-wide variance joins the covariance instead — the step LEAN takes
      // for slopes the footprint averages away, taken here for slopes the
      // grazing eye averages away. The glitter keeps the full slope: point
      // glints are exactly what the fine slopes are for.
      const cotView = viewDir.y
        .max(1e-3)
        .div(float(1).sub(viewDir.y.mul(viewDir.y)).max(1e-6).sqrt())
        .toVar();
      const fineSigma = fineVariance.mul(0.5).add(unresolved).max(1e-6).sqrt().toVar();
      const fineStatistical = float(1)
        .sub(cotView.div(fineSigma.mul(Math.SQRT2)).smoothstep(FINE_GRAZING_A0, FINE_GRAZING_A1))
        .mul(flatten.oneMinus())
        .toVar();
      const skySlope = slope.sub(fineSlope.mul(fineStatistical)).toVar();
      const nSky = normalize(
        mix(normalize(vec3(skySlope.x.negate(), 1, skySlope.y.negate())), vec3(0, 1, 0), flatten),
      ).toVar();
      // The same "just facing the viewer" bend `n` gets above, branchless.
      nSky.assign(
        normalize(
          nSky.add(
            viewDir.mul(float(0.06).sub(nSky.dot(viewDir)).max(0).mul(this.uEyeSubmerged.lessThan(0.5).select(1, 0))),
          ),
        ),
      );
      const skyCov = slopeCov
        .add(vec3(fineVariance, fineVariance, 0).mul(fineStatistical.mul(0.5)))
        .toVar();
      const nFresnel = normalize(mix(nSky, vec3(0, 1, 0), 0.45)).toVar();
      const oneMinus = float(1).sub(nFresnel.dot(viewDir).clamp(0, 1)).toVar();
      const fresnelPow = oneMinus.mul(oneMinus).mul(oneMinus).mul(oneMinus).mul(oneMinus).toVar();
      const fresnel = fresnelPow.mul(0.98).add(0.02).clamp(0, 1).toVar();

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
      const heightAboveMean = worldPos.y.div(seaScale).clamp(0, 1).max(surfThin).toVar();
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
      //
      // A breaker's face counts as crest: it is thin water standing up, and on
      // a 15 m/s sea its metre of height is a fifth of the sea scale, so the
      // height ratio alone never lit it. See `surfThin`.
      const crest = heightAboveMean.mul(heightAboveMean).max(surfThin.mul(0.85)).toVar();
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
      //
      // **The sky comes from the environment, along the true reflected ray.**
      // Ported from web-ocean (ocean.js, "sky reflection"). The surface used to
      // take its sky from the planar mirror: a flat plane at y = 0, sampled at a
      // screen offset of `n.xz * 0.09`. A facet tilted by a slope of 0.2 turns the
      // reflected ray through 23 degrees; that offset moved the lookup by a few
      // dozen pixels. So every ripple reflected nearly the same patch of sky, and
      // the near water read as a mirror with a wobble: the "paint-blob" sky and
      // "jelly-like mottling" of waves-1 and reflections-1.
      //
      // Following web-ocean exactly:
      //  - at grazing angles the *visible* facets are biased toward the viewer
      //    (projected area and masking), so the effective normal tilts toward the
      //    eye by `0.32 sigma_v` as the view drops below about 2.5 sigma_v, and
      //    sees higher, greyer sky;
      //  - a ray that still points into the sea is folded back up, keeping 40%
      //    of its depth, rather than sampling the ground half of the cube;
      //  - the lookup is blurred by the sub-pixel slope spread `sig`: here the
      //    PMREM roughness whose GGX lobe matches a slope RMS of `sig`
      //    (alpha = sqrt(2) sig, roughness = sqrt(alpha)).
      const sig = skyCov.x.add(skyCov.y).mul(0.5).max(1e-6).sqrt().toVar();
      const viewFlat = normalize(vec3(viewDir.x, 0, viewDir.z).add(vec3(1e-5, 0, 0))).toVar();
      const sigV = viewFlat.x
        .mul(viewFlat.x)
        .mul(skyCov.x)
        .add(viewFlat.z.mul(viewFlat.z).mul(skyCov.y))
        .add(viewFlat.x.mul(viewFlat.z).mul(skyCov.z).mul(2))
        .max(1e-6)
        .sqrt()
        .toVar();
      const grazingA = viewDir.y
        .max(1e-3)
        .div(float(1).sub(viewDir.y.mul(viewDir.y)).max(1e-6).sqrt().max(1e-3))
        .div(sigV.mul(1.4142))
        .toVar();
      const tilt = sigV.mul(0.32).mul(float(1).sub(grazingA.smoothstep(0, 2.5))).toVar();
      const nReflect = normalize(nSky.add(viewFlat.mul(tilt))).toVar();
      const rRaw = viewDir.negate().reflect(nReflect).toVar();
      const reflectDir = normalize(
        vec3(rRaw.x, rRaw.y.abs().mul(0.4).add(rRaw.y.max(0).mul(0.6)), rRaw.z),
      ).toVar();
      const envRoughness = sig.mul(1.4142).clamp(0.0, 1).sqrt().toVar();
      this.envNode = pmremTexture(this.envPlaceholder!, reflectDir, envRoughness) as any;
      const skyBlend = reflectDir.y.clamp(0, 1).sqrt().toVar();
      const analyticSky = mix(this.uHorizonColor, this.uSkyColor, skyBlend);
      const reflection = mix(analyticSky, vec3(this.envNode.rgb), this.uEnvReady).toVar();

      if (planar !== null) {
        // **Objects only from the planar mirror.** The hull, the island and the
        // floats are what a planar reflector is for (see `Reflections`): it has
        // the whole scene, including what is above the frame. Its sky would be
        // the flat-mirror sky the environment lookup above replaces, so the dome
        // is kept out of it and its alpha is object coverage, premultiplied
        // into the colour by the transparent clear.
        //
        // Offset by the normal so an object's reflection ripples with the waves.
        // Scaled down with distance because a fixed screen offset is metres at
        // the horizon and millimetres underfoot.
        const offset = nSky.xz
          .mul(this.uReflectionDistortion)
          .div(viewDistance.mul(0.05).add(1))
          .toVar();
        const rawUv = screenUV.add(offset).toVar();
        const safeUv: any = (viewportSafeUV(rawUv) as any).toVar();
        // Softer with sub-pixel roughness and with distance; capped because the
        // target is already at reduced resolution.
        const reflectionLod = sig.mul(6).add(viewDistance.mul(0.004)).clamp(0, 1.25).toVar();
        const mirrored = planar.sample(safeUv).level(reflectionLod).toVar();
        const objectMask: any = mirrored.a.clamp(0, 1).toVar();

        const rEdge = rawUv.min(rawUv.oneMinus()).toVar();
        const inFrame = rEdge.x.min(rEdge.y).smoothstep(0, 0.05).clamp(0, 1).toVar();
        const facing = nDotV.smoothstep(0.02, 0.22).toVar();

        reflection.assign(
          mix(
            reflection,
            mirrored.rgb.div(objectMask.max(1e-3)),
            inFrame.mul(facing).mul(objectMask).mul(this.uReflectionAmount) as any,
          ),
        );
      }

      if (ssr !== null) {
        // Layered over both: a screen-space trace attaches the hull's reflection
        // to the displaced wave it lands on, which one mirror plane at y = 0
        // cannot. The sky's normal is passed, so the hull's reflection breaks
        // up with the same waves the sky's does and not with the fine ones.
        reflection.assign(ssr(worldPos, nSky, reflection));
      }

      // --- sun glitter: anisotropic Gaussian slope distribution ----------------
      //
      // web-ocean's glitter (ocean.js, "sun glitter"), replacing the anisotropic
      // GGX that stood here. Two things change and both matter on water:
      //
      //  - **The distribution is the slope statistics themselves.** A Gaussian in
      //    slope space with mean `slope` and covariance `slopeCov` (the LEAN
      //    form), so the lobe is exactly as wide, as elongated and along the same
      //    axis as the facets inside the pixel. GGX's long tail is right for a
      //    rough solid and wrong for a sea, whose slopes are close to Gaussian
      //    (Cox & Munk); the tail is what spread the glitter into a white sheet.
      //  - **The covariance carries the screen-space term in full**, cross term
      //    included, instead of a per-axis approximation on a tangent frame. That
      //    is the specular anti-aliasing that removes the stair-stepped squares:
      //    a pixel whose slope changes across it gets a lobe covering the change.
      //
      // `D = pdf(h - s) / h_y^4` turns the slope pdf into a normal distribution
      // over the flat macro-surface; Smith masking for a Gaussian slope
      // distribution (Beckmann's rational fit) per direction; Fresnel on the
      // half-vector. Against the macro-surface the cosines are `V.y` and `L.y`,
      // and the sun's `N.L` cancels against the BRDF's.
      const sunDirEarly = normalize(this.uSunDirection).toVar();
      const halfVector = normalize(viewDir.add(sunDirEarly)).toVar();
      const hSlope = vec2(halfVector.x, halfVector.z).negate().div(halfVector.y.max(1e-3)).toVar();
      const dSlope = hSlope.sub(slope).toVar();
      const covDet = slopeCov.x.mul(slopeCov.y).sub(slopeCov.z.mul(slopeCov.z)).max(1e-10).toVar();
      const mahalanobis = dSlope.x
        .mul(dSlope.x)
        .mul(slopeCov.y)
        .add(dSlope.y.mul(dSlope.y).mul(slopeCov.x))
        .sub(dSlope.x.mul(dSlope.y).mul(slopeCov.z).mul(2))
        .div(covDet)
        .toVar();
      const slopePdf = mahalanobis.mul(-0.5).exp().div(covDet.sqrt().mul(2 * Math.PI)).toVar();
      const hy2 = halfVector.y.mul(halfVector.y).toVar();
      const distribution = slopePdf.div(hy2.mul(hy2).max(1e-6)).toVar();
      const smithLambda = (w: any): any => {
        const xz2 = w.x.mul(w.x).add(w.z.mul(w.z)).max(1e-8);
        const il = xz2.inverseSqrt();
        const ax = w.x.mul(il);
        const az = w.z.mul(il);
        const sig2 = ax
          .mul(ax)
          .mul(slopeCov.x)
          .add(az.mul(az).mul(slopeCov.y))
          .add(ax.mul(az).mul(slopeCov.z).mul(2));
        const a = w.y.mul(il).max(1e-3).div(sig2.mul(2).max(1e-7).sqrt()).toVar();
        const lambda = float(1)
          .sub(a.mul(1.259))
          .add(a.mul(a).mul(0.396))
          .div(a.mul(3.535).add(a.mul(a).mul(2.181)));
        return a.greaterThanEqual(1.6).select(float(0), lambda);
      };
      const shadowing = float(1)
        .div(float(1).add(smithLambda(viewDir)).add(smithLambda(sunDirEarly)))
        .toVar();
      const vDotH = viewDir.dot(halfVector).clamp(0, 1).toVar();
      const oneMinusVH = float(1).sub(vDotH).toVar();
      const vhPow = oneMinusVH.mul(oneMinusVH).mul(oneMinusVH).mul(oneMinusVH).mul(oneMinusVH).toVar();
      const fresnelSpecular = vhPow.mul(0.98).add(0.02).toVar();
      const nDotL = n.dot(sunDirEarly).clamp(0, 1).toVar();

      const specular = this.uSunColor
        .mul(
          distribution
            .mul(shadowing)
            .mul(fresnelSpecular)
            .div(viewDir.y.max(0.02).mul(4))
            .mul(sunDirEarly.y.greaterThan(0).select(float(1), float(0))),
        )
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
      //
      // Every source below produces an *amount*, 0..1, and none of them draws
      // anything: the amounts are combined and pushed through one foam pattern
      // and one feather at the end (`ocean/FoamPattern`). That is the whole of
      // the change from drawing each source's contour — which is what made the
      // open sea a field of flat white ovals and the surf a set of paper bands —
      // and it is why the sources below can stay as smooth as the physics that
      // drives them.
      //
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
      // almost none a metre lower down. The bias runs to zero rather than to a
      // floor: with the accumulation buffer carrying persistence this term no
      // longer has to pretend to be continuous, and a floor put a wash of foam
      // into every trough.
      const crestBias = worldPos.y.smoothstep(-0.2, 2.2).clamp(0, 1).toVar();
      const biased = coverage.mul(crestBias).toVar();

      // World-space breakup at the scale of foam clumps (sub-metre), sampled in
      // the wind's own frame and stretched along it (`FOAM_WIND_STRETCH`).
      // It no longer shapes the foam — the pattern does — but it still varies
      // the brightness of the bubbles and perturbs the instantaneous crest mask,
      // and it is faded out with distance because its finest octave is a
      // sub-metre feature that past a few hundred metres is noise with no scale.
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
      const perturb = breakup
        .sub(0.5)
        .mul(smoothstepDownClamped(viewDistance, FOAM_BREAKUP_NEAR, FOAM_BREAKUP_FAR))
        .toVar();

      // Instantaneous crest foam, for the water past the accumulation buffer.
      //
      // **The ramp is not widened by the footprint, and that is a result rather
      // than an omission.** Widening a threshold by the footprint of its own
      // argument (`fwidth`, or the lost slope variance) was implemented twice and
      // rejected on measurement: `gallery-jitter` put far-field high-frequency
      // energy at Medium at 2.09 for the mask as it stands, 2.56 with `fwidth`
      // and 2.94 with the variance, against a 2.33 ceiling. A steep threshold on
      // a noisy argument gives a binary field whose energy sits on sparse edges;
      // a wide ramp maps the sub-footprint noise continuously into every pixel of
      // the band. Widening band-limits the edge, not the field. What does
      // band-limit the field is fading each scale where it stops being resolved,
      // which `perturb` above and the foam pattern's own scales do.
      const crestAmount = biased.add(perturb.mul(0.45)).smoothstep(0.12, 0.78).clamp(0, 1).toVar();

      // --- accumulated foam ---------------------------------------------------
      // Wake, whitecaps and the shore's standing foam, read from the
      // world-anchored buffer. Unlike the crest mask above this is *history*: it
      // was laid down on earlier frames and has been decaying since, which is
      // what lets a wake trail behind a moving hull and a whitecap leave a patch
      // behind the crest that made it.
      const accumulated = float(0).toVar();
      /** How completely the accumulation buffer covers this fragment, 0..1. */
      const buffered = float(0).toVar();
      /** How recently the buffered foam was made, 0..1. See `Wake`'s `FRESH_TAU`. */
      const freshFoam = float(0).toVar();
      if (foam !== null) {
        const foamUvw = worldPos.xz.sub(this.uFoamCenter).div(foam.extent).add(0.5).toVar();

        // The buffer covers a finite square. Outside it there is no information,
        // and clamp-to-edge would smear the border across the whole ocean, so
        // fade out just inside the edge instead.
        const edge = foamUvw.min(foamUvw.oneMinus()).toVar();
        const inside = edge.x.min(edge.y).smoothstep(0, 0.02).clamp(0, 1).toVar();
        buffered.assign(inside);

        const foamTap = this.wakeNode.sample(foamUvw).toVar();
        accumulated.assign(foamTap.r.clamp(0, 1).mul(inside).mul(this.uFoamStrength));
        freshFoam.assign(foamTap.b.clamp(0, 1).mul(inside));
      }

      // Where the buffer reaches, the buffer *is* the whitecaps: it carries the
      // persistence the instantaneous crest mask cannot. Past its 420 m square
      // the crest mask is all there is, faded in exactly as the buffer fades out.
      //
      // **web-ocean's cover model** (`js/wgsl/ocean.js`, `---- foam ----`):
      //
      //   cover  = max(aged, freshNow * (1 - 0.7 w)) + freshHistory * 0.6
      //   cover *= mix(streaks * 1.5, 1, clamp(freshHistory * 2 + freshNow))
      //
      // with `w` its clipmap weight (our `buffered`), `aged` its slow channel
      // (our buffer R, drawn at `AGED_FOAM_DENSITY`), `freshHistory` its fast
      // channel (our buffer B, `Wake`'s freshness) and `freshNow` the crest
      // mask. Fresh foam is dense and stays where it broke; aged foam is torn
      // into the wind's lanes.
      //
      // Kept as two covers rather than web-ocean's one with a mix weight,
      // because ours needs one more step: fresh foam is broken into
      // whitecap-sized patches (`foamCrestPatches`), since our fold regions are
      // tens of metres wide where web-ocean's are a few. Both need the pattern's
      // branch for that, so they are combined there.
      //
      // In the surf the swell that folded to make these whitecaps has been
      // handed over to the breakers (`surfFftFade`), so its whitecaps go with
      // it; the buffer deposited them from the full open-sea field.
      const surfKeep = mix(float(0.25), float(1), surfSwellLeft).toVar();
      const crestNow = crestAmount.mul(buffered.mul(-0.7).add(1)).toVar();
      // The bottom of the buffer's range is not foam on the surface. The
      // windward-face term (`Wake.uWindwardRate`) keeps a wash of 0.2–0.35 over
      // every wind-facing face (measured in the foam-buffer debug view of
      // boat-chase and whitecaps-close), and it is in the buffer because it is
      // in Monahan's total. Drawn as foam, through the pattern, it was the
      // "snow-like speckle" over the whole sea that hid the wake (boat-sea-1).
      // It is aeration, not white water, so it lifts the body below
      // (`aeration`) and the white starts above it.
      const agedCover = accumulated.sub(0.3).div(0.7).clamp(0, 1).mul(AGED_FOAM_DENSITY).mul(surfKeep).toVar();
      const freshCover = crestNow.add(freshFoam.mul(0.6)).mul(surfKeep).toVar();
      const openCover = agedCover.max(freshCover).toVar();

      // --- surf ----------------------------------------------------------------
      //
      // White water driven by the *seabed*, because every foam term above is
      // driven by the surface and neither knows the bottom is there. A wave
      // does not break because it steepened; it breaks because it ran out of
      // water. The breakers themselves — where each crest is, how big, where it
      // breaks, how hard it pitches — come from `ocean/Surf`, shared with the
      // vertex stage (which shaped them) and the ground (which the swash wets),
      // so the foam here sits exactly on the crest the geometry raised.
      //
      // In the order a viewer on the beach sees them:
      //
      //  - **spilling**: on a gentle beach (low Iribarren number) the crest goes
      //    white *before* the break and the foam pours down the face;
      //  - **the lip**: where a plunging face pitches — the compression the
      //    vertex stage's lean causes, Tessendorf's fold criterion applied to
      //    the breaker — and along the crest at the break line;
      //  - **the bore**: from there in, a roller of white water riding its own
      //    front, whose leading edge is ragged rather than ruled;
      //  - **the whitewater it leaves**: dense right behind the roller and
      //    thinning with age — stage A to stage B — which the feather turns into
      //    lace and then threads over about a period;
      //  - **the swash edge**: centimetres of water at the waterline.
      //
      // The persistent buffer adds the foam the surf leaves standing and
      // drifting between sets. See the shore term in `Wake`.
      const surf = float(0).toVar();
      /** Milky, bubble-laden water under and behind the white water, 0..1. */
      const surfAerated = float(0).toVar();
      /** Where the surf's foam has been carried since it was made, metres. */
      const surfAdvect = vec2(0, 0).toVar();
      if (floorDepth !== null) {
        If(inSurf.greaterThan(0.5), () => {
          const d = seabedDepth;
          const s = surfS;
          const age = surfPhaseValue.fract().toVar();

          // Is the nearest crest broken here? Soft over the last few percent of
          // depth so the break line is not a contour.
          const brokenNear = smoothstepDownClamped(d, surfBreak.mul(0.86), surfBreak.mul(1.04)).toVar();

          // The bore's front advances and lags along its length by a few
          // percent of a period: a broken crest is a row of tumbling sections,
          // not a ruled line. Five-metre noise, so the sections are a few
          // metres long.
          const ragged = valueNoise(fieldXZ.mul(0.18).add(vec2(4.1, 2.7))).sub(0.5).mul(0.035);
          const sr = s.add(ragged).toVar();
          const roller = sr
            .smoothstep(-0.055, -0.012)
            .mul(smoothstepDownClamped(sr, 0.03, 0.12))
            .mul(brokenNear)
            .toVar();

          const lipBand = d.sub(surfBreak).div(surfBreak.mul(0.22).add(0.12)).toVar();
          const lipLine = s.div(0.035).toVar();
          const lip = lipBand
            .mul(lipBand)
            .negate()
            .exp()
            .mul(lipLine.mul(lipLine).negate().exp())
            .toVar();
          // Where the pitch is folding the face over.
          const pitchLine = s.div(0.05);
          const pitching = smoothstepDownClamped(surfCompression, 0.3, 0.62)
            .mul(pitchLine.mul(pitchLine).negate().exp())
            .toVar();

          // The crest whitening along its top just before it goes — and, on a
          // spilling beach, the foam pouring down the face ahead of the break.
          const spilling = smoothstepDownClamped(surfXi, 0.35, 0.7).toVar();
          const featherLine = s.add(spilling.mul(0.012)).div(mix(float(0.02), float(0.035), spilling)).toVar();
          const crestWhite = smoothstepDownClamped(d, surfBreak.mul(1.05), surfBreak.mul(1.7))
            .mul(brokenNear.oneMinus())
            .mul(featherLine.mul(featherLine).negate().exp())
            .mul(mix(float(0.45), float(0.85), spilling))
            .toVar();

          // Whether the last two crests broke here, softly: a set's waves break
          // over a band of depths, not on one contour, and a hard test drew the
          // seaward edge of every foam band as a ruled line along the bottom.
          const lastBroke = smoothstepDownClamped(d, surfBreakLast.mul(0.82), surfBreakLast.mul(1.06)).toVar();
          const previousBroke = smoothstepDownClamped(d, surfBreakPrevious.mul(0.82), surfBreakPrevious.mul(1.06)).toVar();
          // Age, in periods, of the youngest broken water at this point: the
          // last crest if it broke here, the one before it otherwise.
          const laceAge = lastBroke
            .greaterThan(0.5)
            .select(age, previousBroke.greaterThan(0.5).select(age.add(1), float(9)))
            .toVar();
          // Not every section of a crest breaks equally hard. Horizon Forbidden
          // West fades its breakers along the wavefront with a strength texture
          // so the break has gaps; this is that in closed form, at about twenty
          // metres, so each wave leaves a run of dense patches and thin ones
          // rather than one even band.
          const sections = valueNoise(fieldXZ.mul(1 / 21).add(vec2(surfPhaseValue.floor().mul(3.7), 1.9)))
            .smoothstep(0.15, 0.85)
            .mul(0.5)
            .add(0.6)
            .toVar();
          // Dense right behind the roller, then thinning: 0.82 at a twentieth
          // of a period, under half by a third of one, a tenth after a period —
          // and gone before the one after.
          const whitewaterAt = (a: any) => a.mul(-2.3).exp().mul(0.92).mul(smoothstepDownClamped(a, 1.3, 1.9));
          const whitewater = whitewaterAt(age)
            .mul(lastBroke)
            .max(whitewaterAt(age.add(1)).mul(previousBroke))
            .mul(sections)
            .min(1)
            .toVar();
          // What is left of a bore is still being carried up the beach by the
          // next, so the pattern drifts shoreward as it ages — and stops doing so
          // at the waterline, where the ground draws the same pattern unmoved.
          surfAdvect.assign(
            shoreward.mul(laceAge.min(2).mul(surfUniforms.period).mul(0.35)).mul(d.smoothstep(0.3, 1.5)),
          );

          // Centimetres of water: the waterline and the front of the swash. The
          // depth buffer knows how thin the water in front of the sand is,
          // including over the sheet the vertex stage ran up the beach.
          const thin = straightThickness
            .smoothstep(0.004, 0.025)
            .mul(smoothstepDownClamped(straightThickness, 0.03, 0.1))
            .mul(smoothstepDownClamped(seabedSigned, 0.5, 1.4))
            .mul(0.75)
            .toVar();

          // Seen from far off, a surf zone is a band of white whose individual
          // breakers are a pixel wide. The time-averaged coverage stands in for
          // them there: most of the inner surf zone white most of the time,
          // thinning out to the break line.
          const band = smoothstepDownClamped(d, surfBreak.mul(0.3), surfBreak.mul(1.3)).mul(0.8).toVar();
          const far = viewDistance.smoothstep(90, 320).toVar();
          // And the line of white at the water's edge, which from more than a
          // few tens of metres is all the swash amounts to: a beach without it
          // reads as a texture boundary.
          const edgeLine = smoothstepDownClamped(straightThickness, 0.02, 0.3)
            .mul(smoothstepDownClamped(seabedSigned, 0.6, 1.4))
            .mul(viewDistance.smoothstep(25, 110))
            .mul(0.7)
            .toVar();

          const strength = surfUniforms.strength.min(1);
          surf.assign(
            roller
              .max(lip)
              .max(pitching)
              .max(crestWhite)
              .max(whitewater)
              .max(thin)
              .max(band.mul(far))
              .max(edgeLine)
              .clamp(0, 1)
              .mul(strength),
          );
          // Behind a bore the water is full of bubbles for seconds after the
          // white has torn: the milky turquoise of every surf photograph, and
          // what the lace sits *on*. Without it the lace sat on glass-clear
          // shallows and read as a decal.
          surfAerated.assign(roller.max(whitewater.mul(1.15)).max(lip).clamp(0, 1).mul(strength));
        });
      }

      // --- contact foam ------------------------------------------------------
      //
      // Foam where something pierces the surface: a cask, a buoy, the hull, a
      // pile. Measured from the depth buffer — the water directly behind this
      // pixel — rather than from any list of objects, so every prop gets it.
      //
      // **A ring at the waterline, not a blob over the object.** The column
      // thickness alone cannot tell "the object comes through the surface just
      // behind this point" from "the object's submerged side is under this
      // point", and it was 1.3 m wide: every floater sat in a white blob and the
      // underwater half of a cask was painted white as a hard band (the
      // coordinator's new-flotsam0 and sbs-barrel captures). The geometry
      // separates them. Along this pixel's ray the object is `t` away; the ray
      // drops `t * V.y` in that distance, so that is how far *below* the
      // surface the object point is. A waterline ring is where both are small:
      // the object is within a few decimetres along the surface and within a
      // few centimetres of the surface itself. Looking down onto a cask the
      // second test leaves a thin collar round it; at a grazing angle the first
      // leaves a band a third of a metre deep in front of it.
      //
      // A quarter of a metre wide and at 0.38 amount, which the threshold below
      // draws as lace over about a third of the ring: it tears, as Water Pro's
      // hull contact foam does, rather than closing into a collar.
      // That cannot scale with the object's size — the depth buffer does not
      // know it — so it is set for the smallest floaters; on the hull the bow
      // and stern foam come from `Wake` and dwarf it.
      //
      // Seabed contact is excluded where the heightfield explains the depth:
      // that is the shore break's job. Faded with distance because depth
      // precision and the ring's footprint both run out.
      const contactAlong = smoothstepDownClamped(straightThickness, 0.03, 0.25);
      const contactBelow = smoothstepDownClamped(straightThickness.mul(viewDir.y.abs()), 0.03, 0.16);
      const notSeabed: any = (
        floorDepth === null
          ? float(1)
          : seabedDepth.sub(straightThickness).smoothstep(0.35, 1.2)
      ).toVar();
      const contactReach = smoothstepDownClamped(viewDistance, 80, 180).toVar();
      const contactAmount = contactAlong
        .mul(contactBelow)
        .mul(notSeabed)
        .mul(contactReach)
        .mul(0.38)
        .toVar();

      // --- the feather ----------------------------------------------------------
      //
      // One pattern, one threshold, for every source. The pattern is read at the
      // *undisplaced* point, as Crest reads its foam texture: the choppy
      // displacement swings the surface back and forth under a fixed world
      // position, and a pattern read there swims across every crest. Moved by
      // the current with the foam it belongs to, and in the surf by the
      // shoreward carry, faded out at the waterline where the ground draws the
      // same pattern in place.
      const foamAmount = openCover.max(surf).max(contactAmount).toVar();
      // The pattern's ramp width. web-ocean uses `fwidth(pattern) + 0.03`;
      // derivatives are undefined inside the non-uniform branch below, so the
      // footprint is taken here and scaled by the pattern's steepest measured
      // gradient, about 2.5 per metre at the 0.4 m sheet scale.
      const foamFootprint = dFdx(fieldXZ).length().max(dFdy(fieldXZ).length()).toVar();
      const foamRamp = foamFootprint.mul(2.5).min(0.25).add(0.03).toVar();
      const foamMask = float(0).toVar();
      If(foamAmount.greaterThan(0.002), () => {
        const carried = mix(this.uFoamDrift, surfAdvect, inSurf).mul(seabedSigned.smoothstep(0.3, 1.5));
        const foamXZ = fieldXZ.sub(carried).toVar();
        const pattern = foamPattern(foamXZ, viewDistance).toVar();

        // Aged open-sea foam into the wind's lanes (web-ocean's streaks near,
        // the 5–26 m patchiness far, where 0.6 m lanes are under a pixel).
        const streakNear = smoothstepDownClamped(viewDistance, 200, 450);
        const lanes = mix(
          foamPatchiness(foamXZ),
          foamStreaks(foamXZ, viewDistance).smoothstep(0.25, 0.75),
          streakNear,
        ).mul(1.5);
        const crestPatch = foamCrestPatches(foamXZ).toVar();
        const cover = agedCover
          .mul(lanes)
          .max(freshCover.mul(mix(float(0.15), float(1.2), crestPatch.mul(crestPatch))))
          .max(contactAmount)
          .toVar();

        // web-ocean's soft threshold: thin foam is a translucent film with a
        // soft margin, dense foam opaque with a crisp edge.
        const soft = cover.clamp(0, 1).oneMinus().mul(0.14).add(0.04);
        const edge = cover.oneMinus();
        const film = pattern.smoothstep(edge.sub(foamRamp).sub(soft), edge.add(foamRamp).add(soft));
        const open = film
          .mul(cover.smoothstep(0, 0.5))
          .clamp(0, 1)
          .mul(cover.smoothstep(0.3, 0.9).mul(0.32).add(0.68))
          // Freshly broken water: a dense, almost opaque core.
          .max(
            freshFoam
              .mul(crestPatch.mul(crestPatch))
              .mul(1.3)
              .smoothstep(0.35, 0.9)
              .mul(pattern.add(0.25).smoothstep(0.25, 0.6))
              .mul(0.95),
          )
          .toVar();

        // As the pattern's scales fade to their means, from ~200 m out, its
        // spread collapses and a threshold on it becomes a step: every patch
        // over about one half clears it everywhere, as a flat plate with a
        // hard rim. So the threshold hands over to plain coverage, the amount
        // itself scaled by what survives of the pattern, with its own soft edge.
        const resolved = smoothstepDownClamped(viewDistance, 200, 420);
        const patternN = pattern.sub(0.29).div(0.53).clamp(0, 1);
        const openFar = cover.clamp(0, 1).mul(mix(float(0.45), float(1.1), patternN)).clamp(0, 1).mul(0.85);
        const openMask = mix(openFar, open, resolved);

        // The surf keeps the feather: its amounts were tuned as white fractions.
        const surfMask = mix(
          surf.clamp(0, 1).mul(mix(float(0.5), float(1.15), patternN)).clamp(0, 1),
          featherFoam(surf, pattern),
          smoothstepDownClamped(viewDistance, 60, 260),
        );
        foamMask.assign(openMask.max(surfMask));
      });

      // Aerated water: the bubble cloud a few centimetres to a metre under the
      // white, which scatters light back up milky and pale. Crest draws it as a
      // second, blurred foam layer under the first; the amount *before* the
      // feather is exactly that layer — smooth, wider than the white, and gone
      // where the foam is.
      const aeration = surfAerated
        .max(openCover.mul(0.3))
        .max(accumulated.mul(0.25))
        .max(contactAmount.mul(0.3))
        .clamp(0, 1)
        .mul(0.5)
        .toVar();

      // The foam's own relief. A raft of bubbles stands a few centimetres proud
      // of the water, so its edges catch the sun on one side and shade on the
      // other; flat-lit, the lace reads as printed on the surface. The gradient
      // of the coverage in world metres, from its screen derivatives — taken
      // here, outside every branch, because derivatives are only defined in
      // uniform control flow — times a nominal raft height. Near only (the
      // derivatives of a sub-pixel pattern are noise) and High and up.
      const relief = foamUniforms.detail
        .greaterThan(1.5)
        .select(smoothstepDownClamped(viewDistance, 25, 70), float(0))
        .toVar();
      const dMaskX = dFdx(foamMask);
      const dMaskY = dFdy(foamMask);
      const dXZx = dFdx(fieldXZ);
      const dXZy = dFdy(fieldXZ);
      const footprintDet = dXZx.x.mul(dXZy.y).sub(dXZx.y.mul(dXZy.x)).toVar();
      const safeDet = footprintDet.abs().max(1e-6).mul(footprintDet.greaterThanEqual(0).select(float(1), float(-1)));
      const maskGradient = vec2(
        dMaskX.mul(dXZy.y).sub(dMaskY.mul(dXZx.y)),
        dXZx.x.mul(dMaskY).sub(dXZy.x.mul(dMaskX)),
      )
        .div(safeDet)
        .toVar();
      // Clamped to a slope of 0.8: at grazing angles the footprint determinant
      // is nearly singular and the solve can return anything.
      const rawRelief = maskGradient.mul(0.035).toVar();
      const reliefSlope = rawRelief.mul(float(0.8).div(rawRelief.length().max(0.8))).mul(relief).toVar();
      const foamNormal = normalize(n.sub(vec3(reliefSlope.x, 0, reliefSlope.y))).toVar();
      const foamNDotL = foamNormal.dot(sunDir).clamp(0, 1).toVar();

      // Foam is lit, not painted.
      //
      // `mix(surface, foamColor, mask)` toward a constant near-white is what made
      // it read as paint: it has no shading, no structure, and it survives into
      // the night unchanged. Whitecaps are a few centimetres of bubbles — a dense
      // scattering medium, so almost Lambertian, bright but still lit by the same
      // sun as everything else, and shadowed on the side away from it.
      //
      // Lit by the same sun *and sky* as the water around it. Without the light
      // term a whitecap sat at roughly 40% of its albedo — darker than the sky
      // the water beside it reflected, so at grazing angles every whitecap read
      // as a grey or (at sunset) brown hole in the sea. Foam is a near-Lambertian
      // scatterer, so its radiance is albedo times the irradiance arriving on
      // it: a wrapped sun term (bubbles scatter light in from the side) under the
      // cloud shadow, plus the sky dome's ambient.
      const bubbles = breakup.mul(0.22).add(0.84).toVar();
      const foamSun = foamNDotL.mul(0.55).add(0.3).mul(sunShade.mul(0.7).add(0.3)).toVar();
      const foamSky = (mix(this.uHorizonColor, this.uSkyColor, 0.45) as any).mul(0.42).toVar();
      const foamLight = (this.uSunColor as any)
        .mul(foamSun)
        .mul(this.uLightLevel.min(1.25))
        .add(foamSky)
        .toVar();
      const foamColor = this.uFoamColor.mul(bubbles).mul(foamLight).toVar();

      // The aerated layer: the same light on a pale turquoise — water seen
      // through bubbles, Crest's bubble colour (0.64, 0.83, 0.82) — at a
      // fraction of the foam's albedo. It replaces part of the transmitted body,
      // never the reflection, so it shows most looking down into the surf.
      //
      // Added rather than mixed in. Replacing the body with a fixed pale colour
      // turned the bubble cloud under every deep-water whitecap into a grey disc:
      // the bubbles brighten the water they are in, they do not repaint it.
      const aeratedLift = (this.uFoamColor as any).mul(vec3(0.6, 0.83, 0.82)).mul(foamLight).mul(0.34);
      litBody.addAssign(aeratedLift.mul(aeration));

      // Foam kills the specular under it. Bubbles are a diffuse rough medium, so a
      // mirror highlight sitting on top of a whitecap is the single most obvious
      // tell that the foam is a decal. Not driven fully to zero: the wet film
      // between bubbles does still glint.
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

      // Never quite opaque: web-ocean composites at `foamMask * 0.95`, so the
      // wave's own shading still shows through the densest foam.
      const withFoam = mix(surface, foamColor, foamMask.mul(0.95)).toVar();

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
