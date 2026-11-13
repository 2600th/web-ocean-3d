import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  exp,
  float,
  interleavedGradientNoise,
  luminance,
  min,
  mix,
  normalize,
  perspectiveDepthToViewZ,
  screenCoordinate,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';

/**
 * The submerged look, as a post-processing node graph.
 *
 * Three layers, composited in this order:
 *
 *   1. Beer–Lambert extinction along the view ray, per colour channel. Red is
 *      absorbed roughly five times faster than blue in sea water, and that
 *      asymmetry — not the fog colour itself — is what makes an image read as
 *      *underwater* rather than as a scene with blue fog in it.
 *   2. God rays: a screen-space radial accumulation from the projected sun,
 *      masked by the depth buffer so the shafts stop dead against near geometry
 *      (the hull) and only survive where the buffer is far away. The shaft
 *      source is striated with a noise field in polar coordinates around the
 *      sun, which is what gives the fan its individual blades.
 *   3. A blue-shifted grade with a depth-driven loss of saturation and contrast,
 *      standing in for the multiply-scattered light that fills the shadows.
 *
 * Everything cross-fades on `submersion`, and at `submersion === 0` the final
 * `mix` returns the input colour bit-for-bit, so a surface crossing is
 * continuous and the pass is free to stay in the chain permanently.
 *
 * The whole graph is built once. Every tunable is a uniform, including the
 * god-ray tap count (a dynamic loop bound, as `QualityManager.godRaySteps`
 * changes with the tier), so nothing here ever recompiles.
 *
 * Backend note: a post-processing pass is drawn with an internal fullscreen quad
 * and its own orthographic camera, so the built-in `cameraNear` / `cameraFar` /
 * `cameraProjectionMatrix` nodes would resolve to *that* camera, not the scene's.
 * The scene camera is therefore supplied explicitly via `setCamera`, and its
 * near/far and the projected sun position are pushed into uniforms each frame —
 * the same approach `PassNode` itself takes internally.
 */

export interface UnderwaterParams {
  /** 0 = fully above water, 1 = fully submerged. Cross-fade, never a hard cut. */
  submersion: number;
  /** Signed height of the eye above the surface beneath it, metres. */
  eyeHeight: number;
  /** Hue of the medium. */
  waterColor: THREE.Color;
  /** Beer–Lambert extinction per metre, per channel. */
  extinction: THREE.Vector3;
  /** Metres to ~10% contrast. */
  visibility: number;
  godRayStrength: number;
  /** 0 disables. */
  godRaySteps: number;
  sunDirection: THREE.Vector3;
  sunColor: THREE.Color;
  /** Depth of the camera below the surface, metres — drives light falloff. */
  cameraDepth: number;
  causticsStrength: number;
  /** Screen-space refractive warp of the submerged image, in uv. 0 disables. */
  distortion: number;
  /** Spatial frequency of that warp, cycles across the frame. */
  distortionScale: number;
  /** How fast it evolves, radians per second. */
  distortionSpeed: number;
  /**
   * Half-width of the meniscus band, metres of ray distance.
   *
   * Floored away from zero when applied — `crossT` is divided by it — so this is
   * not the control that switches the effect off. `meniscusHighlight` and
   * `meniscusPull` are; set both to zero and the band contributes nothing.
   */
  meniscusThickness: number;
  /** Rim brightness at the meniscus boundary. */
  meniscusHighlight: number;
  /** Rim falloff exponent. */
  meniscusSharpness: number;
  /** How far the meniscus pulls the image across the boundary, in uv. */
  meniscusPull: number;
}

export const DEFAULT_UNDERWATER_PARAMS: UnderwaterParams = {
  submersion: 0,
  eyeHeight: 1,
  waterColor: new THREE.Color(0x1a5e7a),
  extinction: new THREE.Vector3(0.115, 0.031, 0.021),
  visibility: 38,
  godRayStrength: 0.9,
  godRaySteps: 24,
  sunDirection: new THREE.Vector3(0.35, 0.62, 0.7).normalize(),
  sunColor: new THREE.Color(0xd8f0ff),
  cameraDepth: 4,
  causticsStrength: 0.35,
  distortion: 0.012,
  distortionScale: 3,
  distortionSpeed: 0.5,
  meniscusThickness: 0.5,
  meniscusHighlight: 0.8,
  meniscusSharpness: 3,
  meniscusPull: 0.02,
};

/**
 * The caustics field, as a TSL function of world position.
 *
 * Supplied rather than reimplemented so the shafts in the water and the pattern
 * on the seafloor are the same evaluation of the same field — see
 * `underwater/Caustics`.
 */
export type CausticsField = (worldPosition: unknown, lod?: unknown) => unknown;

/**
 * Wave surface elevation at a world XZ, as a TSL function.
 *
 * Supplied by the app from the same displacement cascades the ocean mesh is
 * built from, so the waterline this pass finds is the surface the viewer can see
 * — not a plane at y = 0 that the crests pass through.
 */
export type WaveHeightField = (worldXZ: unknown) => unknown;

/**
 * Fraction of sunlight reaching a submerged world point, 0..1.
 *
 * The caustics field describes light arriving at the *surface* and refracting
 * down; it knows nothing about what is floating on that surface. So a hull
 * directly overhead blocked the sun and still had shafts underneath it, which is
 * the one thing a viewer swimming under a ship is guaranteed to look for.
 */
export type SunOcclusionField = (worldPosition: unknown) => unknown;

/** Hard ceiling on the tap count, for sanity rather than for compilation. */
const MAX_GODRAY_STEPS = 64;

/** ln(10): the extinction that leaves 10% of the contrast at `visibility`. */
const LN10 = 2.302585092994046;

/**
 * Fraction of each preset's authored chromatic absorption that is applied.
 *
 * Applied here rather than by editing nine presets, because it is one
 * observation about the medium rather than nine independent look decisions —
 * see `applyParams`, where the reasoning and the measured ranges are.
 */
const ABSORPTION_REACH = 0.5;

/** Clock wrap, seconds. */
const CLOCK_WRAP = 3600;

/**
 * Fraction of the view extinction applied to sunlight on its way *down* to a
 * submerged surface. See `uDownwell`.
 */
const DOWNWELL_SCALE = 0.55;

/** Fraction of the achromatic (scattering) floor that downwelling still loses. */
const DOWNWELL_SCATTER = 0.25;

/**
 * The water column's brightness by view elevation, as multipliers on the medium
 * colour: looking straight down into the deep, horizontally, and straight up at
 * the lit surface. A diver's whole sense of up and down in open water comes from
 * this gradient — the medium was one flat colour in every direction, which is
 * why the empty water read as fog rather than as depth.
 */
const MEDIUM_BELOW = 0.42;
const MEDIUM_LEVEL = 0.95;
const MEDIUM_ABOVE = 1.75;

/** Forward-scatter lobe around the refracted sun: asymmetry and strength. */
const SUN_LOBE_G = 0.72;
const SUN_LOBE_GAIN = 0.05;

/** Darkening at the frame corners when submerged. */
const VIGNETTE = 0.22;

/** Chroma boost on the in-scattered medium colour. */
const MEDIUM_SATURATION = 1.3;

export class UnderwaterPass {
  private readonly params: UnderwaterParams;

  private camera: THREE.PerspectiveCamera | THREE.OrthographicCamera | null = null;
  private clock = 0;
  private disposed = false;

  // --- medium --------------------------------------------------------------
  /**
   * How submerged the *camera* is, 0..1. Still a global scalar, because the eye
   * is a point — but it no longer decides how much water each pixel looks
   * through. See `uEyeDepth`.
   */
  private readonly uSubmersion = uniform(0);
  /**
   * Signed height of the camera above the wave surface directly beneath it,
   * metres. Negative means submerged.
   *
   * This is what makes the waterline per-pixel. The medium is not a screen-wide
   * tint that fades up as the viewer sinks; it is an integral along each eye ray
   * of however much of that ray lies below the surface. A ray pointing down from
   * an eye 10 cm under a crest crosses metres of water; a ray pointing up crosses
   * ten centimetres and leaves. Those are different pixels in the same frame,
   * and the old whole-frame cross-fade could not express the difference.
   */
  private readonly uEyeHeight = uniform(1);
  /**
   * Band, in metres, over which a pixel hands over from air to water.
   *
   * Not a fudge: the eye has physical extent, and a real camera at the surface
   * shows a meniscus a centimetre or two thick rather than a mathematical line.
   * It also keeps the transition from aliasing along the waterline, which a hard
   * step through a wave-shaped boundary certainly would.
   */
  private readonly uWaterlineBand = uniform(0.06);
  private readonly uWaterColor = uniform(new THREE.Color(DEFAULT_UNDERWATER_PARAMS.waterColor));
  /** Total extinction per metre per channel: absorption + the visibility floor. */
  private readonly uSigma = uniform(new THREE.Vector3(0.2, 0.09, 0.08));
  /**
   * Extinction of the sunlight on its way down to a surface, per metre of that
   * surface's depth below the sea.
   *
   * The scene is lit as though every surface were at the top of the column: the
   * sand at seventeen metres received the same warm sun the beach did, and the
   * only thing that made it read as deep was the view path's fog — so a reef
   * seen from a metre away at depth was the colour of a reef in a rock pool.
   * Applying the column above each pixel's own world position is what makes a
   * deep reef blue and a shallow one gold, independently of where the eye is.
   */
  private readonly uDownwell = uniform(new THREE.Vector3(0.1, 0.05, 0.04));
  /**
   * Brightness of light scattered back out of the medium toward the viewer.
   *
   * 0.36, down from 0.95 and then from 0.55. The inscatter term replaces the
   * scene in proportion to how much the water absorbed, so a bright medium does
   * not just tint the view — it *is* the view at any distance. At 0.95 the
   * submerged scene was a flat teal wash with the seafloor eleven metres away
   * completely invisible; 0.55 still left a hull at ten metres reading as a faint
   * smudge against a milky field, which is what a viewer described as washed out.
   *
   * Real water darkens as it thickens. It does not converge on a bright fog: the
   * limiting colour looking into open water from below is the small fraction of
   * downwelling light that scatters back, and against a sunlit surface directly
   * above, that is a *dark* blue-green, not a pale one. What should be bright is
   * the shafts and the surface itself, both of which are added separately — so
   * lowering this raises the contrast between them and the medium rather than
   * darkening the frame as a whole.
   */
  private readonly uAmbient = uniform(0.36);
  private readonly uCameraDepth = uniform(4);

  // --- volumetric shafts ---------------------------------------------------
  private readonly uGodRayStrength = uniform(0.9);
  // Loosely typed: a uniform used as a dynamic `Loop` bound is not modelled by
  // the TSL typings.
  private readonly uSteps: any = uniform(24, 'int');
  private readonly uInvSteps = uniform(1 / 24);
  private readonly uSunColor = uniform(new THREE.Color(DEFAULT_UNDERWATER_PARAMS.sunColor));
  /** Caustic intensity below which no shaft forms, and the gain on the rest. */
  private readonly uShaftFloor = uniform(0.9);
  private readonly uShaftGain = uniform(3);
  /** Mips coarser than the march's own at which the field's local mean is read. */
  private readonly uShaftMeanLod = uniform(7);
  /** How far along the view ray shafts are integrated, metres. */
  private readonly uShaftRange = uniform(90);
  /**
   * Fraction of the light passing through a metre of water that is scattered
   * toward the viewer — the single-scattering albedo term of the integral.
   *
   * Small, and it has to be. The integral runs over tens of metres and the
   * caustics field averages a few tenths across the cell interiors, not the near
   * zero its thin bright filaments suggest, so the accumulated path radiance
   * lands in the single digits before this term is applied. At 0.55 the whole
   * frame saturated to white. Sea water is a weak forward scatterer and the
   * number should look like one.
   */
  private readonly uShaftDensity = uniform(0.05);
  /** World Y of the mean water surface the shafts descend from. */
  private readonly uSeaLevel = uniform(0);
  private readonly uCaustics = uniform(DEFAULT_UNDERWATER_PARAMS.causticsStrength);
  /**
   * An ellipsoid standing in for the hull, so it can shadow the shafts.
   *
   * Analytic rather than a shadow map, and the choice is deliberate. Three's
   * shadow node lives inside the light's material graph; reaching it from a post
   * pass means either duplicating its matrices or depending on internals that
   * have already crashed a tier change twice in this project. A ray-ellipsoid
   * test is four lines, exact for the one occluder that matters, and cannot go
   * stale.
   *
   * What it does not do is shadow anything else — buoys, barrels, the island. A
   * diver under a barrel gets full shafts. That is a real limitation and a very
   * cheap one to accept: the hull is the object anyone will ever swim under.
   */
  private readonly uHullCenter = uniform(new THREE.Vector3(0, 0, 0));
  private readonly uHullRadius = uniform(new THREE.Vector3(1, 1, 1));
  /** Hull heading as a unit XZ vector, so the long axis follows the bow. */
  private readonly uHullForward = uniform(new THREE.Vector2(1, 0));
  private readonly uHullPresent = uniform(0);
  /**
   * Unit vector toward the sun.
   *
   * The pass did not previously need one — the caustics field carries the sun's
   * refracted direction internally — but the hull occluder traces toward it.
   */
  private readonly uSunUp = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  /**
   * World size of one texel of the caustics field, metres.
   *
   * Pushed in rather than assumed, because it is the caustics module that owns
   * its resolution and extent, and the mip level the march asks for is only
   * correct relative to the real texel size.
   */
  private readonly uCausticsTexel = uniform(320 / 768);

  // --- ray reconstruction ---------------------------------------------------
  // A post pass draws with the post-processor's own orthographic quad camera, so
  // the built-in camera matrix nodes describe that quad and not the scene. Both
  // matrices are therefore pushed explicitly, exactly as `uNear`/`uFar` already
  // are, and for the same reason.
  private readonly uInvProjection = uniform(new THREE.Matrix4());
  private readonly uCameraWorld = uniform(new THREE.Matrix4());
  private readonly uCameraPos = uniform(new THREE.Vector3());

  // --- grade ---------------------------------------------------------------
  private readonly uTint = uniform(new THREE.Vector3(0.84, 0.99, 1.08));
  private readonly uDesaturate = uniform(0.2);
  private readonly uContrastLoss = uniform(0.12);

  // --- refractive distortion -----------------------------------------------
  /**
   * Screen-space warp applied to the submerged image, in uv.
   *
   * Water between the eye and what it is looking at is not optically flat. It
   * carries thermal microstructure and the orbital motion of every wave above
   * it, and the result is that an underwater view is always very slightly
   * swimming. Without it the submerged frames are the stillest images this
   * renderer produces, which is exactly backwards.
   *
   * Weighted by the *pixel's* water path, so it goes to zero on a pixel looking
   * out of the water rather than being switched by the camera's own depth. That
   * matters at the waterline, where half the frame is above and half below.
   */
  private readonly uDistortion = uniform(0.012);
  /** Spatial frequency of the warp, cycles across the frame. */
  private readonly uDistortionScale = uniform(3);
  /** How fast the warp evolves, in radians per second. */
  private readonly uDistortionSpeed = uniform(0.5);

  // --- meniscus ------------------------------------------------------------
  /**
   * Half-width of the meniscus band, metres of ray distance.
   *
   * Where a ray crosses the surface within this distance of the eye, it is
   * passing through the film of water that clings to the lens rather than
   * through open water. That film is a lens in its own right: it bends the view
   * sharply and catches a bright rim along the boundary. It is the difference
   * between a camera coming out of the sea and an image with a horizontal wipe
   * across it, and its absence is the largest single reason a half-submerged
   * frame here reads as a compositing artefact.
   *
   * Expressed against `crossT` — the distance along each ray to the surface —
   * rather than against a screen-space line, because the surface is not a plane
   * and the band has to follow the crests the same way the waterline above it
   * already does.
   */
  private readonly uMeniscusThickness = uniform(0.5);
  /** Rim brightness at the boundary. */
  private readonly uMeniscusHighlight = uniform(0.8);
  /** Rim falloff exponent — higher is a tighter, wetter edge. */
  private readonly uMeniscusSharpness = uniform(3);
  /**
   * How far the band pulls the image across the boundary, in uv.
   *
   * This is the normal tilt, done in the only currency a post pass has. A real
   * meniscus turns the surface normal toward the viewer, which compresses what
   * is behind it into the band; displacing the sample toward the waterline is
   * the same compression arrived at from the other side.
   */
  private readonly uMeniscusPull = uniform(0.02);

  // --- camera --------------------------------------------------------------
  private readonly uNear = uniform(0.5);
  private readonly uFar = uniform(40000);
  private readonly uTime = uniform(0);

  constructor() {
    this.params = {
      ...DEFAULT_UNDERWATER_PARAMS,
      waterColor: DEFAULT_UNDERWATER_PARAMS.waterColor.clone(),
      extinction: DEFAULT_UNDERWATER_PARAMS.extinction.clone(),
      sunDirection: DEFAULT_UNDERWATER_PARAMS.sunDirection.clone(),
      sunColor: DEFAULT_UNDERWATER_PARAMS.sunColor.clone(),
    };
    this.applyParams();
  }

  /**
   * Wraps a scene pass.
   *
   * @param scenePassColor The colour texture node, e.g. `scenePass.getTextureNode()`.
   * @param sceneDepth     The *depth texture* node, e.g. `scenePass.getTextureNode('depth')`.
   *                       It must be a texture node — the pass re-samples it at
   *                       offset coordinates to mask the shafts, which a
   *                       pre-linearised scalar node cannot support.
   * @returns The graded node to hand to `PostProcessing.outputNode`.
   */
  build(
    scenePassColor: unknown,
    sceneDepth: unknown,
    causticsNode: CausticsField,
    waveHeightNode: WaveHeightField | null = null,
    sunOcclusionNode: SunOcclusionField | null = null,
  ): unknown {
    const colorNode: any = scenePassColor;
    const depthNode: any = sceneDepth;
    const caustics = causticsNode as (worldPosition: any, lod?: any) => any;
    const waveHeight = waveHeightNode as ((worldXZ: any) => any) | null;
    const sunOcclusion = sunOcclusionNode as ((worldPosition: any) => any) | null;

    if (colorNode === null || colorNode === undefined) {
      throw new Error('UnderwaterPass.build: scenePassColor is required.');
    }
    if (depthNode === null || depthNode === undefined || typeof depthNode.sample !== 'function') {
      throw new Error(
        'UnderwaterPass.build: sceneDepth must be a texture node, e.g. scenePass.getTextureNode( "depth" ).',
      );
    }
    if (typeof caustics !== 'function') {
      throw new Error(
        'UnderwaterPass.build: causticsNode is required — the volumetric shafts are ' +
          'an integral of the caustics field along the view ray, so there is nothing ' +
          'to integrate without it.',
      );
    }

    // PassTextureNode leaves `uvNode` null and falls back to the quad's uv.
    const baseUv: any = colorNode.uvNode ?? uv();

    /** Distance from the camera to the nearest surface at a screen uv, metres. */
    const viewDistance = (p: any): any =>
      perspectiveDepthToViewZ(depthNode.sample(p).r, this.uNear, this.uFar).negate();

    return Fn(() => {
      const suv = vec2(baseUv).toVar('uwUv');
      const src = colorNode.sample(suv).toVar('uwSrc');
      const outRgb = vec3(src.rgb).toVar('uwOut');

      // Uniform-coherent branch: well clear of the surface the whole effect costs
      // one compare. The gate is `uSubmersion`, which the app ramps over a band
      // around the waterline — so it opens slightly *before* the eye goes under,
      // which is exactly when the per-pixel split starts to matter.
      If(this.uSubmersion.greaterThan(0.001), () => {
        // Sky pixels come back at `far`, which is exactly right here: looking at
        // "nothing" underwater means looking at an infinite column of water.
        const axialDist = viewDistance(suv).toVar('uwAxial');

        // Beer-Lambert wants the distance the light travelled, and the depth
        // buffer measures along the camera's *forward axis*.
        //
        // Off-centre pixels look further to reach the same axial depth — by
        // `1 / cos(theta)`, which at the edge of a 55-degree vertical field with
        // a 16:9 frame is about 1.25. Using the axial value directly therefore
        // under-absorbed toward the edges of the frame and, worse, made the
        // absorption depend on the field of view: the same scene through a wider
        // lens got clearer water. The shaft march below already made this
        // correction for itself, which is what made the inconsistency visible.
        //
        // Reconstructed from the projection rather than from the pixel angle, so
        // it stays correct if the projection is ever changed.
        const ndc0 = vec2(suv.x.mul(2).sub(1), suv.y.mul(-2).add(1)).toVar('uwNdc0');
        const viewH0 = this.uInvProjection.mul(vec4(ndc0.x, ndc0.y, -1, 1)).toVar('uwViewH0');
        const viewDir0 = normalize(viewH0.xyz.div(viewH0.w)).toVar('uwViewDir0');
        const axialCos = viewDir0.z.negate().max(1e-3).toVar('uwAxialCos');
        // Hoisted out of the god-ray branch: the waterline needs it too, and
        // rebuilding the same ray twice in one fragment is a straight waste.
        const worldRay = normalize(
          this.uCameraWorld.mul(vec4(viewDir0, 0)).xyz,
        ).toVar('uwWorldRay');
        const dist = axialDist.div(axialCos).toVar('uwDist');

        // --- 1. how much of this ray is under water ---------------------------
        //
        // The eye is a point, so whether *it* is submerged is a per-frame scalar.
        // What is not scalar is how much water each ray crosses, and that is the
        // quantity Beer-Lambert wants.
        //
        // With the eye at signed height `h` above the surface and the ray's
        // vertical component `dy`, the ray crosses the surface at `t = -h / dy`.
        // From an eye below the surface, an upward ray leaves the water there and
        // a downward ray never does; from an eye above it, a downward ray enters
        // there and an upward ray never does. Four cases, one expression.
        //
        // The surface is taken as the wave field evaluated where the ray meets
        // it, refined once from a first guess at the eye's own column — so the
        // waterline follows the crests rather than sitting on a flat plane at the
        // camera's height. Without the refinement, a camera in a trough draws its
        // waterline through the wave in front of it.
        // Guarded away from zero *with the sign kept*. A ray exactly on the
        // horizon has `dy = 0` and `-h/dy` is an unguarded division — introduced
        // by this pass and a NaN waiting at the horizon line of every frame. The
        // floor is 1e-4, which puts the crossing 10 km away for a 1 m eye height:
        // far beyond anything the march or the extinction can reach, so the guard
        // is never visible.
        const rayYRaw = worldRay.y.toVar('uwRayYRaw');
        const rayY = rayYRaw
          .abs()
          .max(1e-4)
          .mul(rayYRaw.lessThan(0).select(float(-1), float(1)))
          .toVar('uwRayY');
        const eyeH = float(0).toVar('uwEyeH');
        eyeH.assign(this.uEyeHeight);

        if (waveHeight !== null) {
          // First guess: the plane at the eye's own local surface height.
          const guessT = eyeH.negate().div(rayY).toVar('uwGuessT');
          const guessP = this.uCameraPos.add(worldRay.mul(guessT.clamp(0, 400))).toVar('uwGuessP');
          // Refined: the eye's height above the surface *there*.
          eyeH.assign(this.uCameraPos.y.sub(waveHeight(guessP.xz)));
        }

        // Distance at which this ray meets the surface. Negative or huge means
        // it never does in the direction it is travelling.
        const crossT = eyeH.negate().div(rayY).toVar('uwCrossT');
        const submergedStart = float(0).toVar('uwStart');
        const submergedEnd = dist.toVar('uwEnd');

        // Eye under water: submerged from 0 until the ray leaves (upward rays
        // only). Eye above: submerged from where the ray enters (downward rays
        // only) to the scene.
        const eyeUnder = smoothstepDown(eyeH, this.uWaterlineBand.negate(), this.uWaterlineBand)
          .toVar('uwEyeUnder');
        const leaves = rayY.greaterThan(0).select(crossT.max(0), float(1e6)).toVar('uwLeaves');
        const enters = rayY.lessThan(0).select(crossT.max(0), float(1e6)).toVar('uwEnters');

        submergedEnd.assign(mix(dist, dist.min(leaves), eyeUnder));
        submergedStart.assign(mix(dist.min(enters), float(0), eyeUnder));

        const waterPath = submergedEnd.sub(submergedStart).max(0).toVar('uwPath');

        // How much this *pixel* is a view through water, for the grade and the
        // shafts. A ray that crosses centimetres of water is not an underwater
        // image; one that crosses metres is.
        // Deliberately *not* multiplied by `uSubmersion`.
        //
        // That scalar is the gate on the branch, and it has to be, because a
        // branch has to be coherent to be cheap. Using it as the blend weight as
        // well was the per-pixel waterline undoing itself: with the eye exactly at
        // the surface, `submersion` is 0.5, so a ray crossing ten metres of water
        // received half the treatment it had just finished computing the full
        // amount of. The path length is the whole answer; the eye's own depth is
        // already in it, through `eyeH`.
        //
        // The one depth-derived quantity that deliberately stays on the
        // *undisplaced* pixel, now that the transmission and the shaft march both
        // follow the refractive warp below. It has to: it scales the warp, so
        // deriving it from the warped sample would be circular. It also does not
        // need to — `1 - exp(-1.4 d)` has saturated to 1 by three metres of
        // water, so a fish and the fifty metres of sea behind it hand back the
        // same value and this term carries no silhouette edge for the warp to
        // pull out of register. Which is exactly why the two terms that do carry
        // one were the two that broke.
        const pixelSubmersion = float(1)
          .sub(waterPath.mul(-1.4).exp())
          .clamp(0, 1)
          .toVar('uwPixelSub');

        // --- 1b. refraction through the water in front of the lens ------------
        //
        // Two displacements of the same sample, because they are the same kind
        // of thing — water bending the view — and paying for two dependent
        // fetches to keep them apart would buy nothing.
        //
        // The warp is a pair of crossed sines rather than a noise texture. What
        // is wanted is a low-frequency, band-limited, *smooth* field; a hash
        // noise would need several octaves and a smoothstep to get there, and
        // the artefact a plain sine grid would normally show — visible
        // rectilinear structure — is invisible here because the amplitude is a
        // hundredth of the frame and the thing being displaced is already a
        // diffuse underwater image. Driven by `uTime`, which is the simulation
        // clock, so a deterministic rewind puts the warp back where it was.
        const warpPhase = this.uTime.mul(this.uDistortionSpeed).toVar('uwWarpT');
        const warpArg = suv.mul(this.uDistortionScale).toVar('uwWarpArg');
        const warp = vec2(
          warpArg.y.mul(6.283).add(warpPhase).sin().add(
            warpArg.x.mul(3.771).add(warpPhase.mul(1.31)).sin().mul(0.5),
          ),
          warpArg.x.mul(6.283).add(warpPhase.mul(0.87)).sin().add(
            warpArg.y.mul(4.115).add(warpPhase.mul(1.17)).sin().mul(0.5),
          ),
        )
          .mul(this.uDistortion.mul(pixelSubmersion))
          .toVar('uwWarp');

        // The meniscus band. `crossT` is where this ray meets the surface; a
        // crossing within `uMeniscusThickness` of the eye is the film on the
        // glass rather than open water. Rays that never cross carry a negative
        // or enormous `crossT` and fall outside the band on their own, so no
        // separate validity test is needed.
        const meniscus = smoothstepDown(crossT, float(0), this.uMeniscusThickness)
          .mul(crossT.greaterThanEqual(0).select(float(1), float(0)))
          .toVar('uwMeniscus');
        // Pulled along screen Y, toward whichever side of the boundary this ray
        // is on — down for a ray heading into the water, up for one leaving it.
        //
        // Screen Y and not the surface normal, which would be the physical
        // answer and is not available here: a post pass has the depth buffer and
        // the wave field, not the shading normal of a surface it is looking
        // *along*. The approximation is exact while the horizon is level and
        // degrades as the camera rolls, which for a rig whose roll comes from a
        // hull's own motion is a few degrees.
        const pullDir = vec2(0, rayYRaw.lessThan(0).select(float(-1), float(1)))
          .toVar('uwPull');
        warp.addAssign(pullDir.mul(meniscus.mul(this.uMeniscusPull)));

        const warpedUv = suv
          .add(warp)
          .clamp(vec2(0.001, 0.001), vec2(0.999, 0.999))
          .toVar('uwWarpedUv');
        const warped = colorNode.sample(warpedUv).toVar('uwWarped');

        // --- 2. transmission --------------------------------------------------
        //
        // Along the ray the colour was actually fetched from, not along this
        // pixel's own. Fogging a displaced sample with the *undisplaced* pixel's
        // water path dissolves anything narrower than the displacement, and it
        // does it in the most alarming way available: a 0.42 m fish eleven metres
        // out has a short path and therefore almost no fog, so its silhouette was
        // filled with distant background colour and then barely attenuated —
        // arriving as a pale, flat, faintly glowing patch exactly where the fish
        // was. The fish's own colour landed on background pixels whose path is
        // fifty metres, where the transmittance term annihilated it. Kelp blades
        // went the same way, and on a moving camera they shimmered, because the
        // warp evolves.
        //
        // At 0.012 uv and the +-1.5 the crossed sines reach, the displacement is
        // 0.018 uv — 23 px across a 1280-wide frame, against a body the
        // `reef-dive` shot sizes at thirty. Anything wider than the displacement
        // was unharmed, which is why the coral heads and the seafloor looked
        // right and only the two thin things in the scene did not.
        //
        // Only the depth is re-fetched. The ray *geometry* — heading, the
        // crossing distance, the axial cosine, which side of the surface the eye
        // is on — is smooth in uv, and 0.018 uv is about a degree of a 55-degree
        // field, so re-solving it (and the wave-height refinement inside it)
        // would buy a correction far below the term it corrects. Depth is the one
        // quantity here with discontinuities in it, so depth is the one worth
        // sampling twice.
        const warpedDist = viewDistance(warpedUv).div(axialCos).toVar('uwDistW');
        const warpedStart = mix(warpedDist.min(enters), float(0), eyeUnder).toVar('uwStartW');
        const warpedEnd = mix(warpedDist, warpedDist.min(leaves), eyeUnder).toVar('uwEndW');
        const warpedPath = warpedEnd.sub(warpedStart).max(0).toVar('uwPathW');
        const transmit = exp(this.uSigma.mul(warpedPath.negate())).toVar('uwT');

        // How much daylight is left at the camera's own depth. Drives both the
        // inscatter brightness and the shaft brightness, so diving gets darker.
        const daylight = exp(this.uSigma.mul(this.uCameraDepth.mul(-0.55))).toVar('uwDay');

        // The medium by direction. See `MEDIUM_ABOVE`: brighter toward the lit
        // surface, darker into the deep, and a forward-scatter glow around where
        // the sun is after refraction — the bright patch a diver sees overhead
        // and to the sun's side, which is the one thing that tells them which way
        // is up without looking for the surface.
        const elevation = worldRay.y.toVar('uwElev');
        const gradient = mix(
          mix(float(MEDIUM_BELOW), float(MEDIUM_LEVEL), elevation.smoothstep(-0.85, 0)),
          float(MEDIUM_ABOVE),
          elevation.smoothstep(0, 0.95),
        ).toVar('uwGrad');
        // Snell's law on the sun's horizontal component, then renormalised.
        const sunH = this.uSunUp.xz.mul(1 / 1.333).toVar('uwSunH');
        const sunUnder = vec3(sunH.x, float(1).sub(sunH.dot(sunH)).max(0).sqrt(), sunH.y).toVar('uwSunDir');
        const cosSun = worldRay.dot(sunUnder).toVar('uwCosSun');
        const g = SUN_LOBE_G;
        const lobe = float((1 - g * g) / (4 * Math.PI))
          .div(float(1 + g * g).sub(cosSun.mul(2 * g)).max(1e-3).pow(1.5))
          .mul(this.uSunUp.y.smoothstep(-0.02, 0.15))
          .toVar('uwLobe');
        const mediumRaw = this.uWaterColor
          .mul(this.uAmbient)
          .mul(gradient)
          .add((this.uSunColor as any).mul(mix(this.uWaterColor.mul(2.2), vec3(1, 1, 1), 0.2)).mul(lobe.mul(SUN_LOBE_GAIN)))
          .mul(daylight)
          .toVar('uwMediumRaw');
        // A little more chroma than the product gives: the medium is the colour
        // of most of the frame, and the grade downstream desaturates it again.
        const medium = mix(vec3(luminance(mediumRaw)), mediumRaw, MEDIUM_SATURATION).max(0).toVar('uwMedium');

        // Sunlight that reached the surface this pixel shows came down through
        // the column above it — see `uDownwell`. Where the view ray found the sky
        // or anything above the sea, the depth is zero and nothing changes.
        const hitY = this.uCameraPos.y.add(worldRay.y.mul(warpedDist.min(2000))).toVar('uwHitY');
        const downwell = exp(this.uDownwell.mul(this.uSeaLevel.sub(hitY).max(0).negate())).toVar('uwDown');

        const fogged = warped.rgb
          .mul(downwell)
          .mul(transmit)
          .add(medium.mul(transmit.oneMinus()))
          .toVar('uwFog');

        // The rim. Tinted by the medium rather than added as white, because what
        // is bright at a meniscus is the water itself catching the light above
        // it — a white line would read as a graphic overlay, which is precisely
        // the failure this is here to remove. Raised to `uMeniscusSharpness` so
        // the band has a hard inner edge and a soft outer one, which is the way
        // round a real film of water sits.
        const rim = meniscus
          .pow(this.uMeniscusSharpness)
          .mul(this.uMeniscusHighlight)
          .toVar('uwRim');
        fogged.addAssign(medium.add(this.uWaterColor.mul(0.35)).mul(rim));

        // --- 2. volumetric shafts --------------------------------------------
        //
        // Marched along the view ray in world space, not smeared radially out
        // from the projected sun.
        //
        // The radial approach is the standard *atmospheric* sun-shaft effect,
        // and underwater it is wrong in three separate ways. Light refracts
        // entering the water, so the apparent sun sits far closer to vertical
        // than the true one and rays that fan from its unrefracted screen
        // position point the wrong way. Real shafts are not a fan at all: they
        // are columns descending from the bright filaments of the surface caustic
        // pattern, so they stay near-vertical however the viewer turns. And a
        // screen-space effect anchored to the sun has to fade out when the sun
        // leaves the frame, whereas shafts are all around a diver and are most
        // visible looking *across* them.
        //
        // So each sample walks back up to the patch of surface that lit it and
        // asks the caustics field how bright that patch is. Shafts and the
        // pattern they cast on the seafloor are then the same light, evaluated
        // in the same place, rather than two effects tuned to resemble each
        // other.
        const rays = vec3(0, 0, 0).toVar('uwRays');

        If(this.uGodRayStrength.greaterThan(0.0001), () => {
          // Rebuild the world-space view ray for this pixel.
          //
          // **NDC y is flipped, and that is not cosmetic.** Screen uv runs
          // top-down on the WebGPU backend and is explicitly flipped to run
          // top-down on the WebGL one, while NDC y runs bottom-up on both — the
          // same asymmetry `clipToScreenUV` in `ScreenSpaceReflection` exists to
          // absorb. Taking `suv.y * 2 - 1` builds a ray pointing *down* wherever
          // the pixel looks up, which sent every shaft the wrong way and was the
          // reason the god rays never converged on the sun.
          const worldDir = worldRay;

          // Bounded by the **refracted** segment, the same one the transmission
          // term uses, and for the same reason.
          //
          // What ends the march is the geometry the ray runs into, so if the
          // image of that geometry is displaced by the warp then the end of the
          // march has to travel with it. Bounding the march at the undisplaced
          // distance while the image moves leaves the shafts stopping short in a
          // silhouette that is no longer there — which arrives as a dark, sharp
          // copy of the fish offset from the fish, and a set of dark stripes
          // beside the kelp rather than behind it. It is the same defect as the
          // washed-out silhouette the transmission term had, in the one term that
          // subtracts light instead of adding it, and it is more legible than the
          // original because a shadow with nothing casting it reads immediately
          // as wrong.
          //
          // Shafts exist only in the submerged part of the ray, and that segment
        // does not necessarily start at the eye. From an eye above the surface
        // looking down, it starts where the ray enters the water — the march used
        // to begin at t = 0 regardless, so it spent its samples in the air above
        // the surface and stopped before reaching the water it was supposed to be
        // integrating.
        const marchStart = warpedStart.toVar('uwMarchStart');
        const march = min(warpedPath, this.uShaftRange).toVar('uwMarch');
          const stepLength = march.mul(this.uInvSteps).toVar('uwStep');

          // Mip level matched to the march's own sampling rate.
          //
          // The footprint is the step's **horizontal** extent, not its length.
          // The caustics field is a 2D map indexed by world XZ, so a vertical
          // metre of march moves the lookup by nothing at all (bar the sun-shear
          // term, which is far smaller). Using the full 3D step length asked for
          // a coarser level than the sampler needed and over-blurred exactly the
          // near-vertical rays that carry the shafts.
          //
          // `log2(footprint / texel)` is then the level at which one texel spans
          // one step — the Nyquist level for this sampler. Below it the march
          // point-samples a signal it cannot resolve, and because the start
          // offset is an interleaved gradient noise the variance arrives as a
          // stationary screen-space lattice rather than as noise.
          //
          // Backed off to 0.75 of the full level: the exact figure is right for a
          // box filter and this is a trilinear one, which over-blurs slightly.
          const stepHorizontal = stepLength.mul(worldDir.xz.length()).toVar('uwStepH');
          const shaftLod = stepHorizontal
            .div(this.uCausticsTexel)
            .max(1)
            .log2()
            .mul(0.75)
            .max(0)
            .toVar('uwShaftLod');

          // Start the march at a per-pixel fraction of a step. Without it every
          // pixel samples the same set of planes and the shafts show up as
          // concentric bands; the dither trades that for fine noise, which the
          // eye reads as suspended particulate.
          const dither = interleavedGradientNoise(screenCoordinate).toVar('uwDither');
          // `uwMarchT`, not `uwT` — the transmission term above already claims
          // that name, and TSL silently renames the collision rather than
          // failing, which makes the shader harder to read than it needs to be.
          const t = stepLength.mul(dither).toVar('uwMarchT');
          const acc = float(0).toVar('uwAcc');

          Loop(this.uSteps, () => {
            const p = this.uCameraPos.add(worldDir.mul(marchStart.add(t))).toVar('uwP');

            // Above the waterline there is no medium to scatter in. Softened
            // rather than a hard cut so a sample crossing the surface does not
            // flicker as the wave moves under it.
            const submerged = this.uSeaLevel.sub(p.y).smoothstep(-0.4, 0.4).toVar('uwSub');

            // The caustics field already carries its own extinction from the
            // surface down to the sample, so this is the light *arriving* here.
            // What it does not know is how much survives the trip back to the
            // eye, which is the second term.
            //
            // Only the field's *excess over its own local mean* scatters into a
            // shaft. Integrated whole, the mean put a uniform grey veil over
            // every submerged frame — tens of metres of average brightness,
            // against which the streaks were a few percent of modulation — and
            // that veil was most of what made the water read as fog rather than
            // as blue. The local mean is the same field several mips coarser, so
            // the term follows the field's own depth fade with no constant to
            // keep in step with it. The diffuse in-scatter the mean stood for is
            // the medium's job, and the medium now has a direction-dependent one.
            const shaftLocal = caustics(p, shaftLod).toVar('uwShaftC');
            const mean = caustics(p, shaftLod.add(this.uShaftMeanLod)).toVar('uwShaftMean');
            const arriving = shaftLocal
              .sub(mean.mul(this.uShaftFloor))
              .max(0)
              .mul(this.uShaftGain)
              .mul(submerged)
              .toVar('uwArriving');

            // What is between this sample and the sun.
            //
            // The caustics field answers "how bright is the surface patch that lit
            // this point", and stops there. Anything floating on that patch — a
            // hull, most obviously — is invisible to it, so a diver under a ship
            // saw full shafts through the one place there should be none.
            //
            // This is the cloud deck's density field, *not* the sun's shadow map:
            // reaching three's shadow node from a post pass means depending on
            // internals that have already broken a tier change more than once. The
            // hull is handled separately, analytically, below.
            if (sunOcclusion !== null) {
              arriving.mulAssign(sunOcclusion(p));
            }

            // The hull, as an ellipsoid **in the hull's own frame**.
            //
            // Axis-aligned would be wrong the moment the ship turns: a hull is
            // three times longer than it is wide, so an ellipsoid that keeps its
            // long axis on world Z casts a shadow across the beam once the bow
            // comes round. Rotate into the heading frame first — the radius is
            // then (beam, draught, length) as it should be.
            const fwd = this.uHullForward;
            const rel = p.sub(this.uHullCenter).toVar('uwHullRel');
            const local = vec3(
              rel.x.mul(fwd.y).sub(rel.z.mul(fwd.x)),
              rel.y,
              rel.x.mul(fwd.x).add(rel.z.mul(fwd.y)),
            ).toVar('uwHullLocal');
            const sunLocal = vec3(
              this.uSunUp.x.mul(fwd.y).sub(this.uSunUp.z.mul(fwd.x)),
              this.uSunUp.y,
              this.uSunUp.x.mul(fwd.x).add(this.uSunUp.z.mul(fwd.y)),
            ).toVar('uwHullSun');

            const toHull = local.negate().div(this.uHullRadius).toVar('uwHullO');
            const sunUnit = sunLocal.div(this.uHullRadius).toVar('uwHullD');
            const along = toHull.dot(sunUnit).div(sunUnit.dot(sunUnit)).toVar('uwHullT');
            const closest = toHull.sub(sunUnit.mul(along.max(0))).length().toVar('uwHullR');
            // 1 outside, 0 through the middle, with a soft rim so the shadow's
            // edge is penumbral rather than a hard ellipse.
            const hullShade = closest.smoothstep(0.75, 1.25).clamp(0, 1).toVar('uwHullShade');
            arriving.mulAssign(mix(float(1), hullShade, this.uHullPresent));
            // Attenuation back to the eye is over the *whole* path from the eye,
            // including the dry part — which contributes nothing, since `t` is
            // measured from the start of the submerged segment and the air adds no
            // extinction. `marchStart` is therefore deliberately absent here.
            const toEye = exp(this.uSigma.mul(t.negate()));

            acc.addAssign(arriving.mul(toEye.g).mul(stepLength));
            t.addAssign(stepLength);
          });

          // `acc` is a Riemann sum weighted by `stepLength`, so it is already an
          // integral over path length and does not get divided by anything —
          // dividing by the range would turn it back into an average and throw
          // away the very accumulation that makes a shaft brighter the further
          // you look along it. What it is missing is the fraction of that light
          // actually scattered toward the eye per metre, which is what
          // `uShaftDensity` supplies. Because the sum carries `stepLength`,
          // changing the tier's step count changes only the sampling noise, not
          // the brightness.
          rays.assign(
            this.uSunColor
              .mul(acc.mul(this.uShaftDensity))
              .mul(this.uGodRayStrength)
              .mul(daylight),
          );
        });

        const lit = fogged.add(rays).toVar('uwLit');

        // --- 3. grade --------------------------------------------------------
        // Depth robs the image of saturation first and of contrast second; both
        // are driven from `cameraDepth` on the CPU side.
        const grey = vec3(luminance(lit));
        const flat = mix(lit, grey, this.uDesaturate);
        const lifted = mix(flat, medium.mul(1.2), this.uContrastLoss);
        // A mask's worth of vignette, only while the eye itself is under: the
        // frame edge darkening is what a lens behind a port or a mask does, and
        // it pulls the eye to the lit middle of the frame where the reef is.
        const corner = suv.sub(0.5).mul(1.4).toVar('uwCorner');
        const vignette = mix(float(1), float(1 - VIGNETTE).max(float(1).sub(corner.dot(corner).mul(VIGNETTE))), eyeUnder);
        const tinted = lifted.mul(this.uTint).mul(vignette);

        outRgb.assign(mix(src.rgb, tinted, pixelSubmersion));
      });

      return vec4(outRgb, src.a);
    })();
  }

  setParams(params: Partial<UnderwaterParams>): void {
    const p = this.params;
    if (params.waterColor !== undefined) p.waterColor.copy(params.waterColor);
    if (params.extinction !== undefined) p.extinction.copy(params.extinction);
    if (params.sunDirection !== undefined) {
      p.sunDirection.copy(params.sunDirection);
      (this.uSunUp.value as THREE.Vector3).copy(p.sunDirection).normalize();
    }
    if (params.sunColor !== undefined) p.sunColor.copy(params.sunColor);
    if (params.submersion !== undefined) p.submersion = params.submersion;
    if (params.eyeHeight !== undefined) p.eyeHeight = params.eyeHeight;
    if (params.visibility !== undefined) p.visibility = params.visibility;
    if (params.godRayStrength !== undefined) p.godRayStrength = params.godRayStrength;
    if (params.godRaySteps !== undefined) p.godRaySteps = params.godRaySteps;
    if (params.cameraDepth !== undefined) p.cameraDepth = params.cameraDepth;
    if (params.causticsStrength !== undefined) p.causticsStrength = params.causticsStrength;
    if (params.distortion !== undefined) p.distortion = params.distortion;
    if (params.distortionScale !== undefined) p.distortionScale = params.distortionScale;
    if (params.distortionSpeed !== undefined) p.distortionSpeed = params.distortionSpeed;
    if (params.meniscusThickness !== undefined) p.meniscusThickness = params.meniscusThickness;
    if (params.meniscusHighlight !== undefined) p.meniscusHighlight = params.meniscusHighlight;
    if (params.meniscusSharpness !== undefined) p.meniscusSharpness = params.meniscusSharpness;
    if (params.meniscusPull !== undefined) p.meniscusPull = params.meniscusPull;
    this.applyParams();
  }

  getParams(): Readonly<UnderwaterParams> {
    return this.params;
  }

  /**
   * The scene camera. Required for depth linearisation and for projecting the
   * sun; see the backend note at the top of this file.
   */
  /**
   * The hull's occluding volume, in world space.
   *
   * `radius` is the ellipsoid's semi-axes. Pass `present` false when there is no
   * ship, which switches the test off for one compare.
   */
  setHullOccluder(
    center: THREE.Vector3,
    radius: THREE.Vector3,
    headingRadians: number,
    present: boolean,
  ): void {
    (this.uHullCenter.value as THREE.Vector3).copy(center);
    (this.uHullRadius.value as THREE.Vector3).copy(radius);
    const forward = this.uHullForward.value as THREE.Vector2;
    forward.set(Math.cos(headingRadians), Math.sin(headingRadians));
    this.uHullPresent.value = present ? 1 : 0;
  }

  /** World metres per texel of the caustics field. See `uCausticsTexel`. */
  setCausticsTexelSize(metres: number): void {
    this.uCausticsTexel.value = Math.max(1e-3, metres);
  }

  setCamera(camera: THREE.PerspectiveCamera | THREE.OrthographicCamera): void {
    this.camera = camera;
  }

  /** Rewinds the animation clock, for reproducible captures. */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;

    const camera = this.camera;
    if (camera === null) return;

    this.uNear.value = camera.near;
    this.uFar.value = camera.far;

    // Everything the shaft march needs to turn a screen uv back into a world ray.
    // `matrixWorld` is read rather than recomputed: the camera has already been
    // updated for this frame by the director, and the post pass runs after it.
    camera.updateMatrixWorld();
    (this.uInvProjection.value as THREE.Matrix4).copy(camera.projectionMatrixInverse);
    (this.uCameraWorld.value as THREE.Matrix4).copy(camera.matrixWorld);
    (this.uCameraPos.value as THREE.Vector3).setFromMatrixPosition(camera.matrixWorld);
  }

  /**
   * The pass owns no geometry, material or render target — it contributes a node
   * graph to whichever `PostProcessing` instance consumed it, and that owns the
   * compiled material. `dispose` therefore only stops the clock so a stale
   * reference cannot keep driving uniforms.
   */
  dispose(): void {
    this.disposed = true;
    this.camera = null;
  }

  // -------------------------------------------------------------------------

  private applyParams(): void {
    const p = this.params;

    this.uSubmersion.value = clampNumber(p.submersion, 0, 1);
    this.uEyeHeight.value = p.eyeHeight;
    this.uWaterColor.value.copy(p.waterColor);
    this.uSunColor.value.copy(p.sunColor);

    // `extinction` is the per-channel colour of the absorption; `visibility` is
    // the achromatic scattering floor that sets the overall range. Adding them
    // keeps both controls meaningful instead of one overriding the other.
    //
    // The absorption is scaled by `ABSORPTION_REACH`, and that constant is the
    // fix for the reef frame reading as one flat teal value. The ordering was
    // diagnostic: a medium that is too *dense* loses the far field first and
    // keeps its foreground, and ours was losing local contrast on the near sand
    // while still resolving shapes forty metres out. That is the signature of
    // extinction that is too high at short range.
    //
    // The arithmetic bears it out. `skyPro` authors 62 m of visibility and a red
    // absorption of 0.22/m, which with the floor put 92% of the red gone at ten
    // metres — so a coral head an arm's length away was already the same colour
    // as the water behind it, and everything the reef has to offer was being
    // integrated out before it reached the lens. Green was delivering 21 m of
    // usable range against the 62 the preset claims.
    //
    // Halving the chromatic part keeps the hue behaviour that makes water read
    // as water — red still goes first, by the same ratio — while moving red's
    // range from 9 m to 16 m and green's from 21 to 32, which is much closer to
    // what the preset says it wants. The far field is unaffected: past a couple
    // of visibility lengths the achromatic floor was always what closed it.
    //
    // And the floor is now what `visibility` says it is: the range of the
    // *clearest* channel. Only the excess absorption of each channel over the
    // clearest one is added on top, so `skyPro`'s 62 m is 62 m of blue rather
    // than 62 m minus whatever blue's own absorption took — which had been
    // quietly taking a third of it, and is the difference between a reef that
    // reads at fifteen metres and one that is a silhouette at ten. Red still
    // goes first by the same margin, which is all the hue behaviour needs.
    const floorSigma = LN10 / Math.max(1, p.visibility);
    const ex = p.extinction;
    const clearest = Math.max(0, Math.min(ex.x, ex.y, ex.z));
    this.uSigma.value.set(
      (Math.max(0, ex.x) - clearest) * ABSORPTION_REACH + floorSigma,
      (Math.max(0, ex.y) - clearest) * ABSORPTION_REACH + floorSigma,
      (Math.max(0, ex.z) - clearest) * ABSORPTION_REACH + floorSigma,
    );
    // Sunlight reaching a submerged surface has come down through the column
    // above it, so its colour is the colour of that depth. Weaker than the view
    // path's extinction because the light arriving at a reef is mostly diffuse
    // downwelling from a wide cone, not one straight ray. And the achromatic
    // floor is mostly scattering, which turns light aside rather than removing
    // it: what one ray loses the cone around it largely regains, so the floor
    // dims downwelling far less than it dims a line of sight.
    const scatterKept = floorSigma * DOWNWELL_SCATTER;
    (this.uDownwell.value as THREE.Vector3).set(
      (this.uSigma.value.x - floorSigma) * DOWNWELL_SCALE + scatterKept,
      (this.uSigma.value.y - floorSigma) * DOWNWELL_SCALE + scatterKept,
      (this.uSigma.value.z - floorSigma) * DOWNWELL_SCALE + scatterKept,
    );

    const depth = Math.max(0, p.cameraDepth);
    this.uCameraDepth.value = depth;

    // Grade strength ramps over the first ~40 m of descent and then holds.
    //
    // Lighter at the top of the range than it was. The downwelling term now does
    // the depth-driven loss of colour physically, per pixel, so a grade that
    // also desaturated the shallows was applying the same idea twice and taking
    // the reef's colour away exactly where a diver is closest to it.
    const descent = Math.min(1, depth / 40);
    this.uDesaturate.value = 0.05 + 0.3 * descent;
    this.uContrastLoss.value = 0.03 + 0.2 * descent;

    const steps = Math.max(0, Math.min(MAX_GODRAY_STEPS, Math.round(p.godRaySteps)));
    // steps === 0 is the Low tier: the march must cost nothing at all.
    this.uGodRayStrength.value = steps > 0 ? Math.max(0, p.godRayStrength) : 0;
    this.uSteps.value = Math.max(1, steps);
    this.uInvSteps.value = 1 / Math.max(1, steps);

    // How far the shafts are integrated. Tied to visibility because there is no
    // point marching through water the medium has already made opaque: past
    // roughly two visibility lengths the transmittance term has closed the
    // contribution down to nothing and the samples are pure cost.
    this.uShaftRange.value = Math.max(12, Math.min(180, p.visibility * 2.2));

    this.uCaustics.value = Math.max(0, p.causticsStrength);

    this.uDistortion.value = Math.max(0, p.distortion);
    this.uDistortionScale.value = Math.max(0.1, p.distortionScale);
    this.uDistortionSpeed.value = Math.max(0, p.distortionSpeed);

    // Floored away from zero rather than allowed to reach it: `crossT` is
    // divided by this thickness to form the band, and a preset that turned the
    // meniscus off by zeroing its width would take the division with it. The
    // highlight and the pull are the controls that switch it off.
    this.uMeniscusThickness.value = Math.max(0.01, p.meniscusThickness);
    this.uMeniscusHighlight.value = Math.max(0, p.meniscusHighlight);
    this.uMeniscusSharpness.value = Math.max(0.1, p.meniscusSharpness);
    this.uMeniscusPull.value = p.meniscusPull;
  }
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

