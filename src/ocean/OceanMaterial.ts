import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  refract,
  smoothstep,
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
  foamLace,
  foamPatchiness,
  foamPattern,
  foamStreaks,
  foamUniforms,
  setFoamDetail,
  setFoamWind,
} from './FoamPattern';
import {
  breaker,
  chopBreaker,
  CHOP_PERIOD,
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
 * The cascade that, with the detail bands, makes the "fine" share of the slope:
 * the 16 m tile, wavelengths 5 cm to 3 m (`Spectrum.CASCADES[2]`).
 */
const FINE_CASCADE = 2;

/**
 * The display level, as a fraction of the white point (`1 / exposure`), that
 * the low-sun glint count is scaled against (see `glintGain`). Under ACES
 * that is about 0.9 of display white.
 */
const GLITTER_CEILING = 0.8;

/** Upwelling light under a night sky, as a share of full daylight's. */
const BODY_NIGHT_FLOOR = 0.1;

/**
 * The body's radiance under full daylight, per unit of the authored
 * `deepColor`, in display-referred terms (it multiplies the white point).
 * The body was previously lit in scene units against a 0.42 exposure, so the
 * clear-day sea measured sRGB (10, 31, 54) beside a reference at
 * (91, 145, 169): a near-black navy that the camo then dominated. With the
 * body tied to the white point, `deepColor` reads directly as a display
 * colour, and 1.8 recovers what the diffuse and cloud-shade terms take off
 * the mean (they average about 0.55 over a sunlit frame).
 */
const BODY_GAIN = 1.8;

/**
 * How much of the short-wave slope (`fineSlope`) the body's diffuse term
 * ignores. The body is light scattered back out of the volume, which does not
 * care which way a centimetre ripple faces; lit with the full shading normal
 * it carried the detail maps' envelope as a 2x swing in brightness, the
 * fbm-like mottle on waves-1 and refraction-1. The long waves keep their
 * troughs and faces; a share of the fine slope is kept for texture.
 */
const BODY_FINE_SLOPE_DROP = 0.7;

/** Body brightness swing from trough (1 - x) to crest (1 + x), by height over the sea state. */
const BODY_TROUGH_CONTRAST = 0.3;

/**
 * Facets per square metre times the sun disc's area in slope space: the
 * expected number of glinting facets in a square metre whose slope density at
 * the half-vector is 1. The disc is 0.0047 rad across, so a normal must sit
 * within 0.0023 rad of the half-vector, about 1.7e-5 of slope area; 20 is then
 * about 1.2e6 facets per m^2, millimetre roughness on the capillaries.
 *
 * Tuned on the captures rather than derived, between two failures: at 0.17
 * (centimetre facets) the path all but vanished into a few dozen points, and
 * at 50 the up-sun sea of boat-sea-1 was a carpet of white flecks again.
 */
const GLINT_FACETS_PER_SUN = 20;
/** How far the glitter's ceiling sits above the sky radiance it is reflected beside. */
const GLINT_OVER_SKY = 4;
/**
 * Share of the mean glitter the draw must carry under the ceiling when the sun
 * is low (sunset). On reflections-2, 1 drew a solid column and 0 a few faint
 * points; this keeps a glittering column.
 */
const GLINT_LOW_SUN_FLOOR = 0.5;
/** Terms of the Poisson inverse CDF; above about this count the draw is normal. */
const GLINT_MAX_COUNT = 12;
/** Cap on one cell's draw over its mean, so a lone glint in a dim tail cannot flare. */
const GLINT_MAX_GAIN = 24;
/** How often the draw is re-seeded, per second of simulation time: the twinkle. */
const GLINT_TWINKLE_HZ = 9;

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
 * [underwater-look-2] Share of the wave cascades' deviation the projected
 * refraction follows to the bed; the rest is the flat surface's. See `bent`.
 */
const WAVE_REFRACT_SHARE = 0.35;

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

/**
 * Diffuse attenuation of light to and from the seabed, as a fraction of the
 * preset's beam extinction. NeuralSpecOcean's 0.58 (`Offshore.js`, MIT); clear
 * ocean water sits at about half to two thirds. See the transmission block.
 */
const BED_DIFFUSE_K = 0.58;
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
 * 1.6 rather than something larger because past about 2 the streaks start to
 * read as scratches: the noise runs out of features along the stretched axis and
 * what is left is a comb — at 2.6 the critic still found "hatch lines" in the
 * crest foam at the horizon (waves-3) and "comb-like hatched fringes" on the
 * whitecaps (foam-2). It also has to stay clear of `uSlopeAnisotropy`, which
 * already stretches the *specular* along the same axis — two anisotropies at the
 * same aspect ratio on the same surface stop reading as wind and start reading
 * as a texture bug.
 *
 * Stretching lowers the along-wind frequency rather than raising anything, so
 * this cannot cost the far-field jitter budget the surrounding comments defend.
 */
const FOAM_WIND_STRETCH = 1.6;

/**
 * How much denser the hull's foam is drawn than its buffer amount.
 *
 * The turbulent band behind a transom holds about 0.85 in `Wake`'s hull channel
 * at the transom (see `DEPOSIT_RATE` there) and thins astern. A wake is the
 * most aerated water a ship makes, so the amount is lifted a little before
 * `foamLace` decides how open its holes are: at the transom they are pinholes,
 * a ship length astern a lace.
 */
const HULL_FOAM_GAIN = 1;

/**
 * Most a wake's foam covers the water. Water Pro's wake is milky, never paint:
 * the sea's colour reads through even the densest churn.
 */
const HULL_FOAM_OPACITY = 0.8;

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
   * Scene radiance the tone curve maps to (nearly) white: `1 / exposure`. See
   * `setExposure` and the glitter ceiling.
   */
  private readonly uWhitePoint = uniform(1 / 0.42);
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
  /**
   * [underwater-look] The medium's colour at the top of the column, from
   * `UnderwaterPass.mediumColor`: what total internal reflection shows.
   */
  private readonly uUnderMedium = uniform(new THREE.Vector3(0.03, 0.27, 0.46));
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

  /**
   * The preset's tone-mapping exposure. The glitter is held under the white
   * point it implies, and only the exposure knows where that is (the same
   * reason `SceneBloom.setExposure` exists).
   */
  setExposure(toneMappingExposure: number): void {
    this.uWhitePoint.value = 1 / Math.max(0.05, toneMappingExposure);
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

  /** [underwater-look] See `uUnderMedium`. A uniform write; call per frame. */
  setUnderwaterMedium(color: THREE.Vector3): void {
    (this.uUnderMedium.value as THREE.Vector3).copy(color);
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
              // Band-limited to the mesh: see Surf.breaker's footprint.
              const wave = breaker(depth, worldXZ, beachSlope, pitch, footprint);
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
      // The surface's total Jacobian (Tessendorf 2001, section 4.4), summed over
      // the cascades; only the foam reads it.
      const fold = float(1).toVar();
      // Slope variance lost inside the texture footprint — see the alpha channel
      // of the derivative pass. Bands are independent, so their variances add.
      const lostSlopeVariance = float(0).toVar();
      /**
       * The short-wave share of the slope (the finest cascade and the detail
       * bands), which the body's diffuse term mostly ignores; see BODY_FINE_SLOPE_DROP.
       */
      const fineSlope = vec2(0).toVar();
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
        // The *total* Jacobian, to first order: J - 1 = Σ (J_i - 1). See the
        // breaking-crest term in `Wake`, which uses the same sum, and why.
        fold.addAssign(d.z.sub(1).mul(fade));
        // E[s^2] - |E[s]|^2, both from the same mip level, so this is exactly the
        // detail this pixel's footprint averaged away — zero where the footprint
        // is a texel and growing as it widens, which is why it needs no distance
        // ramp of its own. `max(0)` guards the half-float subtraction, which can
        // land a few ulps negative when the variance is genuinely nil. Squared
        // fade because variance scales with the square of a linear weight.
        lostSlopeVariance.addAssign(
          d.w.sub(d.x.mul(d.x).add(d.y.mul(d.y))).max(0).mul(fade).mul(fade),
        );
        if (i === FINE_CASCADE) fineSlope.addAssign(vec2(d.x, d.y).mul(fade));
      }

      // [underwater-look] The swell and wind-wave cascades only, before the
      // centimetre detail bands are added: what the refraction offset below
      // bends the backdrop with (see the note there), and what the underside's
      // ripple term subtracts to keep only the fine ripples.
      const refractSlope = vec2(slope).toVar('oceanRefractSlope');

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
        fineSlope.addAssign(vec2(a.x, a.y).add(bSlope).mul(weight));
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
      const surfBreakOlder = float(0).toVar();
      /** The wind sea's bores; see `Surf.chopBreaker`. */
      const chopS = float(0.5).toVar();
      const chopPhase = float(0).toVar();
      const chopBreak = float(0).toVar();
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
          // The same band-limit the vertex stage applied, from the same distance,
          // so the shading's crest is the crest the mesh drew.
          const meshFootprint = groundDistance.mul(this.uVertexSpacing).mul(FOOTPRINT_SPACINGS);
          const wave = breaker(seabedDepth, fieldXZ, bearing.z, pitch, meshFootprint);
          surfPhaseValue.assign(wave.phase);
          surfS.assign(wave.s);
          surfBreak.assign(wave.breakDepth);
          surfBreakLast.assign(wave.breakDepthLast);
          surfBreakPrevious.assign(wave.breakDepthPrevious);
          surfBreakOlder.assign(wave.breakDepthOlder);
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
          // The wind sea's bores: slope only, faded sooner because they are
          // smaller.
          const chop = chopBreaker(seabedDepth, fieldXZ);
          chopS.assign(chop.s);
          chopPhase.assign(chop.phase);
          chopBreak.assign(chop.breakDepth);
          slope.addAssign(shoreward.mul(chop.slope).mul(smoothstepDownClamped(viewDistance, 120, 400)));
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
      const nFresnel = normalize(mix(n, vec3(0, 1, 0), 0.45)).toVar();
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

      // **Where the transmitted ray lands, and how much water it crossed.**
      // Ported from NeuralSpecOcean (MIT, `Offshore.js`, the beach optics): the
      // view ray is refracted at the surface (Snell, n = 1.333), followed down
      // to the bed it would reach, and that point is *projected* to find which
      // pixel of the backdrop to read. The offset that used to stand here was
      // the normal's xz times a constant, divided by distance: up to a tenth of
      // the screen a few metres out, with no idea where the bed was, so the near
      // field smeared and clamped at the frame edges (beach-transparency-2).
      //
      // The path length changes for the same reason. The straight continuation
      // of a grazing view ray runs metres through the water to the bed;
      // the refracted ray bends toward vertical and crosses at most
      // `depth / cos(refracted)`, about 1.5 depths at grazing incidence. The
      // old estimate took the smaller of the straight path and the depth
      // stretched along the *unrefracted* ray, which made 2 m of water an
      // opaque turquoise sheet from a boat's deck.
      //
      // The straight sample is still read: the contact foam wants what is
      // directly behind the surface, not where refraction moved the lookup.
      const surfaceZ: any = (linearDepth() as any).toVar();
      const straightZ: any = (linearDepth(viewportDepthTexture(screenUV)) as any).toVar();
      // The depth difference is measured along the camera's *forward axis*, and
      // light travels along the pixel's own ray — which is longer, by exactly the
      // secant of the angle between them. Without that correction the water
      // absorbs less toward the edges of the frame than at its centre, and the
      // apparent density of the whole ocean changes with field of view. An
      // independent review caught this; the volumetric fog had it right and this
      // did not.
      const axialToRay: any = float(1).div(viewDir.dot(this.uCameraForward).abs().max(0.15));
      // Water between the surface and whatever is *directly* behind it, before
      // refraction moved the lookup. Contact foam is about where an object
      // pierces this pixel's surface, so it must not follow the refracted ray.
      const straightThickness: any = straightZ
        .sub(surfaceZ)
        .max(0)
        .mul(this.uDepthRange)
        .mul(axialToRay)
        .toVar();

      // The water column under this point, from the heightfield: what a ray
      // can cross before it meets the bed, wherever the depth buffer has not
      // drawn the bed (or has drawn it too far off to trust).
      const column: any = (
        floorDepth === null ? float(1000) : worldPos.y.add(seabedSigned).max(0)
      ).toVar();
      // [underwater-look] Bent by the wave cascades' normal, not the full one:
      // the centimetre detail bands vary from pixel to pixel at any distance, so
      // refracting by them scrambled the lagoon floor (and its caustic web) into
      // speckle. The detail bands still shade, glint and foam.
      const refractNormal = normalize(vec3(refractSlope.x.negate(), 1, refractSlope.y.negate())).toVar();
      // [underwater-look-2] The flat surface's refraction carries the parallax;
      // only part of the waves' deviation is added to it. Followed metres down
      // to the bed, the full deviation moves the landing point faster than the
      // pixel moves wherever a wave face tilts toward the camera, so the
      // lookup folds back on itself and whole rows of the floor collapse into
      // horizontal streaks (lagoon-caustics at wind 15, after the merge).
      const flatBent = refract(viewDir.negate(), vec3(0, 1, 0), 1 / WATER_IOR);
      const waveBent = refract(viewDir.negate(), refractNormal, 1 / WATER_IOR);
      const bent: any = normalize(mix(flatBent, waveBent, WAVE_REFRACT_SHARE)).toVar();
      // NeuralSpecOcean's travel: to the bed along the bent ray, floored at a
      // 0.28 vertical component so a grazing ray does not run to infinity, and
      // never past what the depth buffer says is straight behind (a hull, a
      // rock) or 8 m, past which the offset is clamped away anyway.
      const travel: any = column
        .div(bent.y.negate().max(0.28))
        .min(straightThickness.min(8))
        .toVar();
      const toUv = (p: any): any => {
        const clip = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(p, 1)).toVar();
        const ndc = clip.xy.div(clip.w.max(1e-4));
        // NDC y runs up, the viewport buffers' uv down; see
        // `ScreenSpaceReflection.clipToScreenUV`.
        return vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(-0.5).add(0.5));
      };
      // `uRefractionStrength` stays the tier's knob: 0.22 is High's, and 1x.
      const projected = toUv(worldPos.add(bent.mul(travel)))
        .sub(toUv(worldPos))
        .mul(this.uRefractionStrength.div(0.22))
        .clamp(-0.045, 0.045)
        .toVar();
      // Faded out near the frame's edge, so a lookup never clamps to the border
      // and smears it (NeuralSpecOcean's `edgeWeight`).
      const edgeDistance = screenUV.min(screenUV.oneMinus()).toVar();
      const edgeBand = projected.abs().mul(2).max(0.004).toVar();
      const edgeWeight = smoothstep(float(0), edgeBand.x, edgeDistance.x)
        .min(smoothstep(float(0), edgeBand.y, edgeDistance.y))
        .toVar();
      const refractedUv: any = screenUV.add(projected.mul(edgeWeight)).toVar();
      const nearerUv: any = screenUV.add(projected.mul(edgeWeight).mul(0.4)).toVar();

      // A refracted sample can land on something *in front of* the water — the
      // hull's own topsides, most obviously — and pulling that colour underwater
      // smears it down the wave face. Where that happens, try a shorter offset,
      // then fall back to the unrefracted sample, which is behind the surface by
      // construction (NeuralSpecOcean steps back the same way).
      const behindZ: any = (linearDepth(viewportDepthTexture(refractedUv)) as any).toVar();
      const nearerZ: any = (linearDepth(viewportDepthTexture(nearerUv)) as any).toVar();
      const valid: any = behindZ.greaterThan(surfaceZ).toVar();
      const nearerValid: any = nearerZ.greaterThan(surfaceZ).toVar();
      const sampleUv: any = valid.select(refractedUv, nearerValid.select(nearerUv, screenUV)).toVar();
      const backdropZ: any = valid.select(behindZ, nearerValid.select(nearerZ, straightZ)).toVar();

      // Straight-line distance from the surface to what the sample shows, and
      // how far below the surface that point is: the vertical drop along the
      // ray. For the seabed that is its depth there; for a rock face, how far
      // under the surface the face is — which is why the water over a rock's
      // shoulder is pale and the water off its foot is not.
      const bufferThickness: any = backdropZ
        .sub(surfaceZ)
        .max(0)
        .mul(this.uDepthRange)
        .mul(axialToRay)
        .toVar();
      const seenDepth: any = bufferThickness.mul(viewDir.y.abs()).min(column).toVar();
      // Snell: cos of the refracted angle, floored at NeuralSpecOcean's 0.66 so
      // the path never exceeds 1.5 depths.
      const cosRefracted: any = float(1)
        .sub(nDotV.mul(nDotV).oneMinus().div(WATER_IOR * WATER_IOR))
        .max(0.44)
        .sqrt()
        .toVar();
      const pathLength: any = seenDepth.div(cosRefracted).toVar();

      // The bed is a diffuse reflector, and the light it sends back is not
      // lost to the eye when it scatters: water scatters almost entirely
      // forward, so what leaves the image ray mostly still arrives, a little
      // blurred. Bed light therefore falls off with the *diffuse* attenuation
      // coefficient K_d on both legs, not with the beam coefficient the
      // preset's `extinction` is — and for clear ocean water K_d is a little
      // over half of it (NeuralSpecOcean takes 0.58, `Offshore.js`, MIT). With
      // the beam coefficient on the way up and K_d plus an extra scattering
      // term on the way down, the green and blue came out 1.6 to 2 times
      // Jerlov type I's (about 0.06 and 0.03 per metre). Over 3 m of sand the
      // shallows were navy rather than aquamarine (shore-break). The same
      // coefficient weights the column's own colour against the bed, so the
      // two stay complementary.
      const diffuseK: any = this.uExtinction.mul(BED_DIFFUSE_K).toVar();
      const absorption: any = diffuseK.mul(pathLength).negate().exp().toVar();
      // Daylight reaching the bed was attenuated on the way *down* too, and the
      // ground shader lights the bed as if it were dry: the same K_d over the
      // depth, divided by the mean cosine of downwelling light (0.82). This is
      // the shallow-water gradient the reference shows — pale over a metre of
      // water, green-blue by four, blue past ten — and it comes from the same
      // coefficients as everything else.
      const downwelling: any = diffuseK
        .mul(seenDepth.div(0.82))
        .negate()
        .exp()
        .toVar();

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
      //
      // Measured vertically now, against the column: the old test compared the
      // bed's depth with the straight distance to the backdrop, which at any
      // grazing angle is many depths long, so it never fired where it was
      // meant to; and it paid a second heightfield evaluation to do it.
      const objectBehind: any = (
        floorDepth === null ? float(0) : column.sub(seenDepth).smoothstep(0.35, 1.2)
      ).toVar();
      // The bed's colour reaches the eye only after the light that shows it has
      // gone *down* to the bed as well as back up the view ray, so it is
      // attenuated over the depth too, not over the view path alone. With the
      // view path alone, thirty metres of clear water (blue extinction
      // 0.06 /m) still passed a sixth of the sand tint, and every open-sea
      // frame near the island came out lagoon cyan (boat-sea-1/2, waves-1).
      // A two-metre lagoon keeps 90% of it.
      const bedLit: any = this.uExtinction.mul(seabedDepth).negate().exp();
      const inscatter: any = mix(
        mix(this.uDeepColor, this.uShallowColor, absorption.mul(bedLit)),
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
      // Split into the two things it is, because they are lit differently: the
      // column's own scattered light, which the water's diffuse term below
      // lights, and the backdrop seen through it, which the scene already lit —
      // multiplying that by the water's `n . l` a second time darkened the bed
      // by up to half and printed the wave normals onto it as fake caustics.
      const sceneWeight: any = absorption.mul(this.uRefractionAmount).mul(refractionReach).toVar();
      const columnColor: any = inscatter.mul(sceneWeight.oneMinus()).toVar();
      const seenThrough: any = refracted.mul(downwelling).mul(sceneWeight).toVar();

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
      const sig = slopeCov.x.add(slopeCov.y).mul(0.5).max(1e-6).sqrt().toVar();
      const viewFlat = normalize(vec3(viewDir.x, 0, viewDir.z).add(vec3(1e-5, 0, 0))).toVar();
      const sigV = viewFlat.x
        .mul(viewFlat.x)
        .mul(slopeCov.x)
        .add(viewFlat.z.mul(viewFlat.z).mul(slopeCov.y))
        .add(viewFlat.x.mul(viewFlat.z).mul(slopeCov.z).mul(2))
        .max(1e-6)
        .sqrt()
        .toVar();
      const grazingA = viewDir.y
        .max(1e-3)
        .div(float(1).sub(viewDir.y.mul(viewDir.y)).max(1e-6).sqrt().max(1e-3))
        .div(sigV.mul(1.4142))
        .toVar();
      const tilt = sigV.mul(0.32).mul(float(1).sub(grazingA.smoothstep(0, 2.5))).toVar();
      const nReflect = normalize(n.add(viewFlat.mul(tilt))).toVar();
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
        const offset = n.xz
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
        // cannot. The distance-flattened normal is passed, as before.
        reflection.assign(ssr(worldPos, n, reflection));
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

      // The brightest a glint can be drawn: GLITTER_CEILING of the white point,
      // or a multiple of the sky this facet reflects where that sky is itself
      // past white (sunset). Only the low-sun count floor below uses it. A soft
      // shoulder that compressed the glitter to it was dropped after the
      // spectrum fix (aaa-pass 0f78f7c): with the slopes right, the sun path is
      // already tight, and the shoulder only dimmed it.
      const skyReflected = reflection.x.max(reflection.y).max(reflection.z).mul(fresnel);
      const whiteCeiling = this.uWhitePoint.mul(GLITTER_CEILING).toVar();
      const glitterCeiling = whiteCeiling.max(skyReflected.mul(GLINT_OVER_SKY)).toVar();
      /** 1 under a low sun, 0 by day: see the count floor in `glintGain`. */
      const lowSun = float(1).sub(sunDirEarly.y.smoothstep(0.1, 0.25)).toVar();

      // **Glints, not a sheet.** What the lobe above returns is the *expected*
      // glitter of the pixel: the mean over every facet in its footprint. A real
      // footprint holds a countable number of facets turned to the sun, a few or
      // none, each as bright as the sun's image in a mirror, and that count is
      // what reads as sparkle. Evaluated as the mean, any pixel whose facets
      // straddle the half-vector lit up together with its neighbours, so the
      // path drew as smeared blotches (reflections-1, waves-6, refraction-1).
      //
      // So the count is drawn, after Zirr & Kaplanyan, "Real-time Rendering of
      // Procedural Multiscale Materials" (I3D 2016): the expected number of
      // glinting facets in a world-anchored cell the size of the pixel is the
      // facet density times the cell's area times the probability a facet faces
      // the sun (the distribution times the sun disc's area in slope space),
      // and a Poisson draw of it, hashed from the cell and the clock, scales the
      // mean. Where the count is large (far water, wide footprints) the draw is
      // its mean and the sheen is untouched; near the eye it is a scatter of
      // points that keeps the path's energy. Two power-of-two cell sizes are
      // blended so the cells do not pop as the footprint changes.
      const glintFootprintX = dFdx(fieldXZ);
      const glintFootprintY = dFdy(fieldXZ);
      const glintArea = glintFootprintX.x
        .mul(glintFootprintY.y)
        .sub(glintFootprintX.y.mul(glintFootprintY.x))
        .abs()
        .max(1e-8)
        .toVar();
      const glintLevel = log2(glintArea).mul(0.5).toVar();
      const glintLevel0 = glintLevel.floor().toVar();
      // `uRainTime` is the simulation clock (see `setRain`), so a deterministic
      // rewind replays the same twinkle.
      const glintSlot = this.uRainTime.mul(GLINT_TWINKLE_HZ).floor().mod(997).toVar();
      const glintMean = specular.x.max(specular.y).max(specular.z).toVar();
      const glintRate = distribution.mul(shadowing).mul(GLINT_FACETS_PER_SUN).toVar();
      const glintGain = (level: any): any => {
        const side = level.exp2();
        const cell = fieldXZ.div(side).floor().mod(4096);
        const u: any = hashNoSine22(cell.add(vec2(glintSlot.mul(41), level.mul(113)))).toVar();
        // Under a low sun, never fewer glints than it takes to carry a share of
        // the mean under the ceiling. The whole path is then seen at grazing
        // incidence beside an aureole already at white, and a sparse draw
        // clips at the display and throws the column's energy away. By day
        // (`lowSun` 0) the draw is left sparse, which is the whole point of it.
        const expected: any = glintRate
          .mul(side.mul(side))
          .max(glintMean.div(glitterCeiling).mul(lowSun.mul(GLINT_LOW_SUN_FLOOR)))
          .toVar();
        // Poisson inverse CDF, unrolled: k = how many of the terms u exceeds.
        const term: any = expected.negate().exp().toVar();
        const cdf: any = term.toVar();
        const count = float(0).toVar();
        for (let k = 1; k <= GLINT_MAX_COUNT; k++) {
          count.addAssign(u.x.greaterThan(cdf).select(float(1), float(0)));
          term.mulAssign(expected.div(k));
          cdf.addAssign(term);
        }
        // Past a dozen the Poisson is its normal approximation (Box-Muller).
        const normal = u.x.max(1e-6).log().mul(-2).sqrt().mul(u.y.mul(2 * Math.PI).cos());
        const gaussian = float(1).add(normal.div(expected.max(1e-3).sqrt())).max(0);
        const drawn = mix(
          count.div(expected.max(1e-6)),
          gaussian,
          expected.smoothstep(GLINT_MAX_COUNT * 0.5, GLINT_MAX_COUNT * 0.8),
        );
        return drawn.min(GLINT_MAX_GAIN);
      };
      const glintDraw = mix(
        glintGain(glintLevel0),
        glintGain(glintLevel0.add(1)),
        glintLevel.sub(glintLevel0),
      ).toVar();
      specular.mulAssign(glintDraw);

      // --- combine ---------------------------------------------------------------
      // The body colour is authored for *daylight*. Its diffuse term bottomed out
      // at 0.45, which is a floor on how dark lit water can get — so at night the
      // shallow-water tint kept glowing at nearly half strength and the water
      // around the submerged hull read as a green lamp. It now falls with the
      // light, to a small floor rather than to zero: a night sea is dark, but it
      // is not black, because the whole sky is still faintly lighting it.
      //
      // The body is display-referred: it scales with the white point
      // (`1 / exposure`), so a preset's exposure, which is set for its sky, no
      // longer decides how bright its water is (see BODY_GAIN). The light
      // reaching the column is the sun's, tinted a third of the way to its
      // colour (the rest is skylight), and falls as the sun drops, toward the
      // night floor.
      const sunLift = sunDirEarly.y.mul(2.2).clamp(0, 1).toVar();
      const upwelling = mix(
        float(BODY_NIGHT_FLOOR),
        float(1),
        this.uLightLevel.mul(sunLift.mul(0.7).add(0.3)),
      ).toVar();
      const bodyLight = mix(vec3(1), this.uSunColor, 0.35)
        .mul(upwelling.mul(this.uWhitePoint).mul(BODY_GAIN))
        .toVar();
      const bodySlope = slope.sub(fineSlope.mul(BODY_FINE_SLOPE_DROP)).toVar();
      const bodyDotL = normalize(vec3(bodySlope.x.negate(), 1, bodySlope.y.negate()))
        .dot(sunDirEarly)
        .clamp(0, 1)
        .toVar();
      // Troughs read darker than faces: the eye looks into a trough through
      // the wave in front of it, and the light that comes back up out of it
      // has crossed more water. Relative to the sea state, as the crest term
      // above is, so a chop and a swell shade alike.
      const troughShade = worldPos.y
        .div(seaScale)
        .clamp(-1, 1)
        .mul(BODY_TROUGH_CONTRAST)
        .add(1)
        .toVar();
      const litBody = columnColor
        .mul(bodyDotL.mul(0.55).add(0.45).mul(sunShade.mul(0.55).add(0.45)))
        .mul(bodyLight.mul(troughShade))
        .add(seenThrough)
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
      /** The hulls' foam — wake, waterline collar, splashes. `Wake`'s alpha. */
      const hullFoam = float(0).toVar();
      if (foam !== null) {
        const foamUvw = worldPos.xz.sub(this.uFoamCenter).div(foam.extent).add(0.5).toVar();

        // The buffer covers a finite square. Outside it there is no information,
        // and clamp-to-edge would smear the border across the whole ocean, so
        // fade out just inside the edge instead.
        const edge = foamUvw.min(foamUvw.oneMinus()).toVar();
        const inside = edge.x.min(edge.y).smoothstep(0, 0.02).clamp(0, 1).toVar();
        buffered.assign(inside);

        const foamTap = this.wakeNode.sample(foamUvw).toVar();
        accumulated.assign(foamTap.r.clamp(0, 1).mul(inside));
        freshFoam.assign(foamTap.b.clamp(0, 1).mul(inside));
        hullFoam.assign(foamTap.a.clamp(0, 1).mul(inside));
      }

      // Where the buffer reaches, the buffer *is* the whitecaps: it carries the
      // persistence the instantaneous crest mask cannot. Past its 420 m square
      // the crest mask is all there is, faded in exactly as the buffer fades out.
      //
      // **web-ocean's cover model** (`js/wgsl/ocean.js`, `---- foam ----`), now
      // whole:
      //
      //   cover  = max(aged, freshNow * (1 - 0.7 w)) + freshHistory * 0.6
      //   cover *= mix(streaks * 1.5, 1, clamp(freshHistory * 2 + freshNow))
      //
      // with `w` its clipmap weight (our `buffered`), `aged` its slow channel
      // (our buffer R), `freshHistory` its fast channel (our buffer B) and
      // `freshNow` the crest mask. The streak factor is applied in the
      // pattern's branch below.
      //
      // What it replaces drew *less* than the buffer held, on purpose: aged foam
      // at 0.45 of its amount above a 0.3 floor, fresh foam cut to 15% outside
      // noise "crest patches". Both were work-arounds for a deposit whose fold
      // regions were the swell's, 10–20 m blobs, so that drawn at face value
      // every one was a flat white sheet (waves-4). With the total Jacobian
      // (`Wake`) the deposit is crest-sized itself, and the work-arounds are
      // what made a 18 m/s sea show no whitecaps from a 3 m eye (foam-1).
      //
      // In the surf the swell that folded to make these whitecaps has been
      // handed over to the breakers (`surfFftFade`), so its whitecaps go with
      // it; the buffer deposited them from the full open-sea field.
      const surfKeep = mix(float(0.25), float(1), surfSwellLeft).toVar();
      const crestNow = crestAmount.mul(buffered.mul(-0.7).add(1)).toVar();
      const seaCover = accumulated
        .max(crestNow)
        .add(freshFoam.mul(0.6))
        .mul(this.uFoamStrength)
        .mul(surfKeep)
        .clamp(0, 1.5)
        .toVar();
      /** How much of the sea's foam is still dense breaking water, 0..1. */
      const seaFresh = freshFoam.mul(2).add(crestNow).clamp(0, 1).toVar();
      // The wake is not torn into wind lanes and not drawn as a whitecap: it is
      // its own cover, dense at the transom and thinning astern (see
      // `HULL_FOAM_GAIN`).
      const hullCover = hullFoam.mul(HULL_FOAM_GAIN).clamp(0, 1.2).toVar();
      const openCover = seaCover.clamp(0, 1).max(hullCover.clamp(0, 1)).toVar();

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
          const olderBroke = smoothstepDownClamped(d, surfBreakOlder.mul(0.82), surfBreakOlder.mul(1.06)).toVar();
          // Age, in periods, of the youngest broken water at this point: the
          // last crest if it broke here, else the one before, else the one
          // before that.
          const laceAge = lastBroke
            .greaterThan(0.5)
            .select(
              age,
              previousBroke.greaterThan(0.5).select(age.add(1), olderBroke.greaterThan(0.5).select(age.add(2), float(9))),
            )
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
          // Two stages, the way a surf zone's foam ages (Monahan's stage A and
          // B whitecaps, and NeuralSpecOcean's long-tail roller wake): the dense
          // white right behind the roller, gone within half a period; and the
          // lace it leaves, which outlives the bore by two more and is what
          // stands between the breakers as a field. The lace used to die with
          // the dense part, after about a period, so a surf zone was the white
          // band of the newest bore on clear water (foam-4).
          //
          // The lace is held to two and a bit periods and then gone by three,
          // exactly when the oldest crest tracked here hands over to one that
          // is not — so nothing pops as the crests renumber.
          const denseAt = (a: any) => a.mul(-3.2).exp().mul(0.92);
          const laceAt = (a: any) => a.mul(-0.6).exp().mul(0.5).mul(smoothstepDownClamped(a, 2.2, 3));

          // **Foam lines.** The white a bore leaves is not a sheet: it is torn
          // into streaks lying *along* the crest that made them, a metre or two
          // apart, with clear green water between (NeuralSpecOcean's surf
          // reference, and every photograph of a spilling beach). Drawn as one
          // smooth amount through the feather it came out as flat white plates
          // with hard rims floating mid-water (beach-transparency-2, foam-4).
          //
          // A crest-parallel coordinate is already to hand: the breaker phase
          // without its clock, `sqrt(depth) * k` plus the along-shore terms,
          // which is constant along every depth contour and so along every
          // crest. Noise read across it at two rates and stretched along it by
          // a slow world term gives lanes ~2.5 m and ~0.9 m apart and tens of
          // metres long, which follow the coast round a headland because the
          // contours do. Static in the world: the lanes are where the foam was
          // left, and `surfAdvect` already carries the bubbles shoreward.
          const contourPhase = surfPhaseValue.sub(surfUniforms.time.div(surfUniforms.period)).toVar();
          const lanes = valueNoise(vec2(contourPhase.mul(13), 0).add(fieldXZ.mul(0.045)).add(vec2(1.7, 3.1)))
            .mul(0.6)
            .add(valueNoise(vec2(contourPhase.mul(37), 0).add(fieldXZ.mul(0.13)).add(vec2(8.3, 0.4))).mul(0.4))
            .smoothstep(0.32, 0.7)
            .toVar();
          // Dense where it is fresh, lanes as it ages: a bore's first second is
          // white all through, and only then tears.
          const tearing = age.smoothstep(0.03, 0.18).toVar();
          // The wind sea's bores: a thinner roller and a short dense wake, where
          // its crests have broken. No lace of their own: their foam joins the
          // swell's field.
          const chopBroken = smoothstepDownClamped(d, chopBreak.mul(0.86), chopBreak.mul(1.04));
          const chopR = chopS.add(ragged.mul(0.7)).toVar();
          const chopRoller = chopR
            .smoothstep(-0.05, -0.012)
            .mul(smoothstepDownClamped(chopR, 0.02, 0.08))
            .mul(chopBroken)
            .mul(d.smoothstep(0.05, 0.25))
            .mul(0.85)
            .toVar();
          const chopDense = denseAt(chopPhase.fract().mul(CHOP_PERIOD))
            .mul(chopBroken)
            .mul(mix(float(0.3), float(1.1), lanes))
            .mul(0.7);
          const dense = denseAt(age).mul(lastBroke).max(denseAt(age.add(1)).mul(previousBroke)).max(chopDense);
          // The lace is the foam lines and nothing else: aged foam has torn all
          // the way into them, with clear water between.
          const lace = laceAt(age)
            .mul(lastBroke)
            .max(laceAt(age.add(1)).mul(previousBroke))
            .max(laceAt(age.add(2)).mul(olderBroke))
            .mul(mix(float(0.06), float(1.3), lanes));
          const whitewater = dense
            .mul(mix(float(1), mix(float(0.18), float(1.15), lanes), tearing))
            .max(lace)
            .mul(sections)
            .min(0.9)
            .toVar();
          // What is left of a bore is still being carried up the beach by the
          // next, so the pattern drifts shoreward as it ages — and stops doing so
          // at the waterline, where the ground draws the same pattern unmoved.
          surfAdvect.assign(
            shoreward.mul(laceAge.min(3).mul(surfUniforms.period).mul(0.35)).mul(d.smoothstep(0.3, 1.5)),
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
          // The roller is the one part that is white all through, but not
          // evenly along its length: its sections and lanes thin it to 60%.
          // The far band keeps the lanes too, which at a few hundred metres are
          // still several pixels apart, so it reads as foam lines rather than
          // the flat plate the time-average alone drew.
          // The residual floor: the old, stable foam of many sets, a sparse
          // scatter of lines over the inner surf zone that is there even in a
          // lull. The field itself is the lace above, which follows the sets;
          // this is only what keeps a lull from going glass-clear. Carried in
          // to the waterline, where the swash's own foam takes over.
          const residual = lanes
            .mul(lanes)
            .mul(smoothstepDownClamped(d, surfBreak.mul(0.45), surfBreak.mul(1.15)))
            .mul(d.smoothstep(0.05, 0.3))
            .mul(0.2)
            .toVar();
          surf.assign(
            roller
              .mul(mix(float(0.6), float(1), lanes.max(sections.sub(0.6).mul(2.5))))
              .max(chopRoller.mul(mix(float(0.55), float(1), lanes)))
              .max(residual)
              .max(lip)
              .max(pitching)
              .max(crestWhite)
              .max(whitewater)
              .max(thin)
              .max(band.mul(far).mul(mix(float(0.35), float(1.1), lanes)))
              .max(edgeLine)
              .clamp(0, 1)
              .mul(strength),
          );
          // Behind a bore the water is full of bubbles for seconds after the
          // white has torn: the milky turquoise of every surf photograph, and
          // what the lace sits *on*. Without it the lace sat on glass-clear
          // shallows and read as a decal.
          // The dense stage is what carries the bubbles; the lace floats on
          // water that has mostly cleared.
          surfAerated.assign(roller.max(dense.mul(1.15)).max(whitewater.mul(0.5)).max(lip).clamp(0, 1).mul(strength));
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
      //
      // **In the surf it is surge, not a collar.** A rock standing in a surf
      // zone has every bore break against it: white water climbs its face,
      // pours round it and is left in a skirt metres wide that the next wave
      // renews (NeuralSpecOcean's rock-contact foam; the critic's "rocks enter
      // the water with no contact foam" in beach-transparency-1 and foam-4).
      // So inside the surf the ring widens and thickens with how hard the water
      // there is breaking, and pulses with each bore's arrival.
      // Outside the break the swell still runs up a rock's face and falls back
      // white, so anything standing in shoal water (under ~6 m: rocks, piles,
      // the shallows' boulders) takes most of it. Floating things are in deep
      // water nearly always — the play area sits on a 17 m plateau — and keep
      // the narrow collar.
      const shoal: any = smoothstepDownClamped(seabedDepth, 2.5, 6.5).mul(0.75);
      const surge: any = inSurf
        .mul(smoothstepDownClamped(seabedDepth, surfBreak.mul(0.4), surfBreak.mul(1.6)))
        .max(shoal)
        .mul(surfUniforms.strength.min(1))
        .toVar();
      const surgePulse: any = surfPhaseValue.fract().mul(-2.2).exp().mul(0.45).add(0.55).toVar();
      const contactAlong = smoothstepDownClamped(straightThickness, 0.03, mix(float(0.25), float(5), surge));
      const contactBelow = smoothstepDownClamped(
        straightThickness.mul(viewDir.y.abs()),
        0.03,
        mix(float(0.16), float(1.2), surge),
      );
      const notSeabed: any = (
        floorDepth === null
          ? float(1)
          : seabedDepth.sub(straightThickness).smoothstep(0.35, 1.2)
      ).toVar();
      const contactReach = smoothstepDownClamped(viewDistance, mix(float(80), float(160), surge), mix(float(180), float(320), surge)).toVar();
      const contactAmount = contactAlong
        .mul(contactBelow)
        .mul(notSeabed)
        .mul(contactReach)
        .mul(mix(float(0.38), float(0.75).mul(surgePulse), surge))
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
        // the 5–26 m patchiness far, where the lanes are under a pixel).
        const streakNear = smoothstepDownClamped(viewDistance, 200, 450);
        const lanes = mix(
          foamPatchiness(foamXZ),
          foamStreaks(foamXZ, viewDistance).smoothstep(0.25, 0.75),
          streakNear,
        ).mul(1.5);
        const cover = seaCover.mul(mix(lanes, float(1), seaFresh)).max(contactAmount).toVar();

        // web-ocean's soft threshold: thin foam is a translucent film with a
        // soft margin, dense foam opaque with a crisp edge.
        const soft = cover.clamp(0, 1).oneMinus().mul(0.14).add(0.04);
        const edge = cover.oneMinus();
        const film = pattern.smoothstep(edge.sub(foamRamp).sub(soft), edge.add(foamRamp).add(soft));
        // Freshly broken water: a dense core. web-ocean multiplies a smooth
        // ramp of the freshness by the pattern, and over a fold region larger
        // than a few metres that ramp *is* the visible edge — the "huge flat
        // slab with a blurred edge" over waves-4's foreground. Thresholding the
        // pattern by it instead (`featherFoam`) puts the core's edge where the
        // pattern runs out, ragged at every scale the pattern has, and leaves
        // the bubbles' holes in it.
        const seaCore = featherFoam(freshFoam.smoothstep(0.3, 0.9).mul(0.92), pattern);
        // Aged foam is a raft that is dying: its bubbles burst and the holes
        // between them widen (`foamLace`) until it is a lace of rims, which is
        // the "foreground foam lace with cellular holes that thins as it ages"
        // of Water Pro's foam-3. Fresh water is not punched: it is still being
        // made.
        const aged = foamLace(foamXZ.add(vec2(37.1, 11.9)), cover.mul(1.3), viewDistance);
        const open = film
          .mul(mix(aged, float(1), seaFresh))
          .mul(cover.smoothstep(0, 0.5))
          .clamp(0, 1)
          .mul(cover.smoothstep(0.3, 0.9).mul(0.32).add(0.68))
          .max(seaCore.mul(0.95))
          .toVar();

        // The wake: not a whitecap and not lanes, but Water Pro's "cellular,
        // Voronoi-like aerated" foam (boat-sea-1) — a milky film punched with
        // round holes that widen as it ages (`foamLace`), its edge torn by the
        // same pattern as every other foam.
        If(hullCover.greaterThan(0.01), () => {
          const hullEdge = pattern.smoothstep(
            hullCover.oneMinus().sub(0.2).sub(foamRamp),
            hullCover.oneMinus().add(0.2).add(foamRamp),
          );
          const hullMask = foamLace(foamXZ, hullCover, viewDistance)
            .mul(hullEdge)
            .mul(hullCover.smoothstep(0, 0.4))
            .mul(HULL_FOAM_OPACITY);
          open.assign(open.max(hullMask));
        });

        // As the pattern's scales fade to their means, from ~200 m out, its
        // spread collapses and a threshold on it becomes a step: every patch
        // over about one half clears it everywhere, as a flat plate with a
        // hard rim. So the threshold hands over to plain coverage, the amount
        // itself scaled by what survives of the pattern, with its own soft edge.
        const resolved = smoothstepDownClamped(viewDistance, 200, 420);
        const patternN = pattern.sub(0.29).div(0.53).clamp(0, 1);
        const openFar = cover
          .max(hullCover.mul(0.6))
          .clamp(0, 1)
          .mul(mix(float(0.45), float(1.1), patternN))
          .clamp(0, 1)
          .mul(0.85);
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
        .max(hullCover.clamp(0, 1).mul(0.45))
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
      // [underwater-look] Rebuilt after abyssal-garden's `W_SURF` (MIT, (c) 2026
      // Maksymilian Dendura); see THIRD_PARTY_NOTICES.md. Three changes, each
      // answering one thing a blind critic saw in `underwater-1` and
      // `refraction-3`:
      //
      // 1. The branch is "the eye is under water", not "this facet faces
      //    away". From below every visible point of the surface is seen from
      //    beneath; a ripple facet that happened to tilt toward the eye was
      //    sent down the *topside* path and came out as a dark ragged hole.
      // 2. The window's edge is read off a normal relaxed toward vertical with
      //    distance (abyssal's `ss(260, 60, d)`), widened to its `ss(0, .22, k)`
      //    band, and faded by the transmission (`1 - fr`), so it is a soft rim
      //    rather than a noisy cut-out that blew to white.
      // 3. Outside the window, total internal reflection shows the *water*:
      //    the medium's own colour looking down (abyssal's `fogIn` along the
      //    reflected ray), not a screen-space lookup that found dark pixels or
      //    the frame edge and drew the stripe along the top of `refraction-3`.
      If(this.uEyeSubmerged.greaterThan(0.5), () => {
        const relax = viewDistance.smoothstep(8, 120).mul(0.12).add(0.85).toVar();
        const nUp = normalize(mix(n, vec3(0, 1, 0), relax)).toVar();
        const cosI = nUp.dot(viewDir).abs().clamp(0, 1).toVar();

        // Snell: sin(t) = ior * sin(i). Past sin(t) = 1 there is no transmitted
        // ray and everything is reflected. `k` is cos^2 of the air-side angle.
        const sinT2 = float(1)
          .sub(cosI.mul(cosI))
          .mul(WATER_IOR * WATER_IOR)
          .toVar();
        const k = float(1).sub(sinT2).toVar();
        const insideWindow = k.smoothstep(0, 0.22).toVar();
        const cosT = k.max(0).sqrt().toVar();
        // Schlick-style transmission falloff toward the critical angle: the
        // window's rim darkens into the mirror instead of ending on a bright edge.
        const fr = float(1).sub(cosT).pow(5).toVar();

        // Where in the sky this ray came from, through Snell: the outer third of
        // the window's radius holds the horizon.
        const skyBase = mix(this.uHorizonColor, this.uSkyColor, cosT).toVar();
        // abyssal's ripple term: the slope, read across the window, brightens and
        // darkens the refracted sky in bands, which is what makes a Snell window
        // read as water rather than as a hole in the ceiling.
        // From the centimetre ripples alone (the full slope less the wave
        // cascades'), as abyssal's is from its short slope texture: it textures
        // the window without deciding where the window is, and without the
        // swell painting metre-wide bright blobs across it.
        const fine = slope.sub(refractSlope).toVar();
        const ripple = float(1)
          .add(fine.x.mul(2.4).sub(fine.y.mul(3.2)))
          .clamp(0.45, 1.9)
          .toVar();
        // The refracted sun, from the air-side ray: a broad glow and a disc.
        const airRay = normalize(vec3(viewDir.x.negate(), cosT, viewDir.z.negate())).toVar();
        const sd = airRay.dot(normalize(this.uSunDirection)).max(0).toVar();
        const sd2 = sd.mul(sd).toVar();
        const sd8 = sd2.mul(sd2).mul(sd2).mul(sd2).toVar();
        const sd64 = sd8.mul(sd8).mul(sd8).mul(sd8).mul(sd8).mul(sd8).mul(sd8).mul(sd8).toVar();
        const sunGlow = sd64.mul(1.1).add(sd64.pow(8).mul(9)).toVar();
        // Seen through the water above it the sky is already the water's
        // colour at its rim (abyssal draws its window cyan, `(.30, .78, 1)`):
        // half way toward the medium's hue at the same brightness.
        const mediumHue = this.uUnderMedium.div(this.uUnderMedium.dot(vec3(0.2126, 0.7152, 0.0722)).max(1e-4));
        const tintedSky = mix(skyBase, mediumHue.mul(skyBase.dot(vec3(0.2126, 0.7152, 0.0722))), 0.6);
        const windowColor = tintedSky
          .mul(ripple)
          .mul(1.1)
          .add((this.uSunColor as any).mul(sunGlow))
          .mul(this.uLightLevel.mul(0.85).add(0.15))
          .mul(fr.oneMinus())
          .toVar();

        // Total internal reflection: the water column below, looking down along
        // the reflected ray. The in-scatter of a ray descending at |R.y|
        // converges on `C * sigma / (sigma + K |R.y|)`, and the pass's `K` is
        // 2.2 sigma, so steeper reflections are darker, as abyssal's are.
        const reflectedDir = viewDir.negate().reflect(nUp).toVar();
        const mirrored = this.uUnderMedium
          .mul(float(1.3).div(reflectedDir.y.abs().mul(2.2).add(1)))
          .mul(mix(float(1), ripple, 0.3))
          .toVar();

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

/**
 * Dave Hoskins' "Hash without Sine" (MIT, https://www.shadertoy.com/view/4djSRW),
 * hash22: two uniforms from a 2D lattice point. Used for the glint draw, whose
 * cells run to thousands, where `sin`-based hashes lose their low bits.
 */
const hashNoSine22 = /*@__PURE__*/ Fn(([p]: [any]) => {
  const p3 = vec3(p.x, p.y, p.x).mul(vec3(0.1031, 0.103, 0.0973)).fract().toVar();
  p3.addAssign(p3.dot(vec3(p3.y, p3.z, p3.x).add(33.33)));
  return vec2(p3.x, p3.x).add(vec2(p3.y, p3.z)).mul(vec2(p3.z, p3.y)).fract();
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
