import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  exp,
  float,
  interleavedGradientNoise,
  mx_fractal_noise_float,
  normalize,
  perspectiveDepthToViewZ,
  pow,
  screenCoordinate,
  select,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';

/**
 * Volumetric sea fog, as a post-processing node graph.
 *
 * The medium is an exponential height layer: extinction is `density` at
 * `baseHeight` and falls off with an e-folding height of `heightFalloff`, so the
 * fog pools on the water and thins with altitude. That single property is what
 * separates sea fog from a haze — from the deck you are inside it and the
 * horizon is gone; from a mast or an orbit camera you are above it looking down
 * at a bank with a top surface.
 *
 * Three things are integrated along each view ray:
 *
 *   1. **Transmittance, analytically.** For a medium whose density varies only
 *      with height, the optical depth along a straight ray has a closed form
 *      (Wenzel, "Real-Time Atmospheric Effects in Games", SIGGRAPH 2006 — see
 *      `transmittanceAt` below). No march required. So how much of the scene
 *      survives the fog is *exact*, at every quality tier, and identical whether
 *      the march runs 8 steps or 48. This matters more than it sounds: the tier
 *      drops on its own when frames get long, and a fog whose thickness visibly
 *      changed with the step count would turn an invisible quality change into an
 *      obvious one. It also means the effect has no banding in its opacity at
 *      all, which is where a naive volumetric normally shows its steps first.
 *
 *   2. **In-scattering, marched — but weighted analytically.** Each march cell
 *      contributes `L * (T(cellStart) - T(cellEnd))`, which is the *exact*
 *      integral of `sigma * T` across that cell, not a Riemann rectangle. Two
 *      things follow. The sum telescopes to `1 - T(end)` regardless of how many
 *      cells there are, so the total in-scattered energy is step-count
 *      independent in the same way the transmittance is — this is Hillaire's
 *      energy-conserving segment integration (Frostbite, SIGGRAPH 2015) with the
 *      constant-extinction assumption dropped, since here `T` is known in closed
 *      form and does not need one. And the far cells, which this march
 *      deliberately makes long, stop over-contributing: a rectangle rule
 *      overestimates a cell of optical depth ~1 by about 40%.
 *
 *      What the march is actually for is the part that is *not* analytic: how the
 *      light reaching each point varies. Two terms do that. The sun's
 *      transmittance down to a sample is itself a closed-form height integral
 *      (`sigma * H / sun.y`), so fog deep in the layer is lit far less than fog
 *      near its top — the bank glows along its upper surface and goes blue-grey
 *      inside, with no shadow map and no secondary march. And a Henyey–Greenstein
 *      lobe on the angle between the view ray and the sun makes looking toward the
 *      sun through fog bloom, while looking away from it stays flat and cool. The
 *      phase term is constant along the ray for a directional light, so it is
 *      evaluated once per pixel rather than once per step.
 *
 *   3. **A bounded medium.** The layer ends at `maxDistance` for every pixel,
 *      sky included; past it the far-field haze is `AerialPerspective`'s and the
 *      dome's. Because every ray is clamped to the same range there is no shell:
 *      a pixel past the bound gets the same integral as one exactly at it, and
 *      the sea's last row and the sky's first row integrate the same path. This
 *      used to be an analytic tail out to the far plane, and on near-horizontal
 *      sky rays that tail, lit by an unoccluded forward-scattered sun, was the
 *      white seam along the horizon (see `build`). For a uniformly lit medium the
 *      pass still collapses *exactly* to `scene * T + L * (1 - T)` — the classic
 *      fog lerp, which is the correct limit to degrade to.
 *
 * Banks are a two-octave procedural field modulating the *scattering*, not the
 * extinction. That is a deliberate approximation and worth being honest about:
 * modulating extinction would make the banks vary in opacity, which is more
 * correct, but it also destroys the closed-form transmittance that item 1 is
 * built on and hands the banding back. What varies most visibly in real fog at
 * any distance is which patches are catching the light, and that is exactly what
 * modulating the scattered radiance reproduces — at the cost of a silhouette that
 * is smoother than the interior.
 *
 * Nothing here is temporally jittered. The published froxel solutions
 * (Wronski 2014, Hillaire 2015) reproject and jitter across frames because they
 * are backed by a TAA resolve; this project has none, so a temporal offset would
 * be a shimmer with nothing to average it away. The march start is dithered
 * spatially only, with interleaved gradient noise against the pixel coordinate.
 *
 * The whole graph is built once. Every tunable is a uniform, including the march
 * step count (a dynamic `Loop` bound, as the tier changes it), so nothing here
 * ever recompiles. The march is procedural throughout — no 3D noise texture, so
 * nothing to allocate, upload or keep resident, and nothing that would need a
 * storage or compute path on WebGL2.
 *
 * Backend note: a post-processing pass is drawn with an internal fullscreen quad
 * and its own orthographic camera, so the built-in `cameraNear` / `cameraFar` /
 * `cameraProjectionMatrix` nodes would resolve to *that* quad, not the scene's.
 * The scene camera is therefore supplied explicitly via `setCamera` and its
 * near/far, inverse projection and world matrix are pushed into uniforms each
 * frame — the same approach `PassNode` takes internally, and the same one
 * `UnderwaterPass` in this project already takes for its shaft march.
 */

/**
 * TSL node objects are structurally dynamic; the generated types cannot express
 * a uniform whose component type is only known at construction, nor one used as
 * a dynamic loop bound. Node-typed fields are therefore `any` by design — the
 * class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

export interface VolumetricFogParams {
  /**
   * Extinction per metre at `baseHeight`, the densest point of the layer.
   * 0 disables the effect entirely. Roughly, visibility at the base is `1/density`
   * metres, so 0.0075 is a few hundred metres of sea room.
   */
  density: number;
  /** World Y the layer sits on. Mean sea level, normally. */
  baseHeight: number;
  /** e-folding height of the layer in metres. Small = a low bank, large = haze. */
  heightFalloff: number;
  /** Colour of the sky-lit scattering — the fog's own colour away from the sun. */
  color: THREE.Color;
  /** Brightness of that sky-lit term. 1 makes the far field settle on `color`. */
  ambient: number;
  /** Unit vector *toward* the sun. */
  sunDirection: THREE.Vector3;
  sunColor: THREE.Color;
  /** Brightness of the directional in-scattering that produces the sun bloom. */
  sunIntensity: number;
  /**
   * Henyey–Greenstein g. Positive is forward scattering, so the fog brightens
   * toward the sun.
   *
   * Measured fog droplets sit around 0.8–0.87, and using that here would be
   * wrong: what softens the lobe in real fog is multiple scattering, which a
   * single-scattering integral does not have, so a physical g over-concentrates
   * the glow into a hard disc. Engines that expose this generally see 0.1–0.4
   * used for plain haze and 0.6+ only when a visible sunbeam is the point.
   */
  anisotropy: number;
  /**
   * Metres. The extent of the medium along every ray, sky included. Beyond it
   * the haze is `AerialPerspective`'s and the dome's, not this layer's.
   */
  maxDistance: number;
  /** 0..1 strength of the procedural bank variation. 0 skips it per step. */
  detail: number;
  /** Horizontal feature size of the banks, metres. */
  detailScale: number;
  /** Drift speed of the banks, m/s. Slow — fog is not weather in a hurry. */
  windSpeed: number;
  /** Drift bearing, radians. */
  windDirection: number;
  /** March steps. 0 disables the effect, like `density` 0. */
  steps: number;
}

export const DEFAULT_VOLUMETRIC_FOG_PARAMS: VolumetricFogParams = {
  density: 0.0075,
  baseHeight: 0,
  heightFalloff: 18,
  color: new THREE.Color(0xb6c6d4),
  ambient: 0.9,
  sunDirection: new THREE.Vector3(0.35, 0.62, 0.7).normalize(),
  sunColor: new THREE.Color(0xffe9cf),
  sunIntensity: 0.55,
  anisotropy: 0.6,
  maxDistance: 1500,
  detail: 0.55,
  detailScale: 130,
  windSpeed: 2.2,
  windDirection: 2.6,
  steps: 24,
};

/** Hard ceiling on the loop the shader is compiled with. */
const MAX_STEPS = 64;

/** Clock wrap, seconds. Matches the other animated systems. */
const CLOCK_WRAP = 3600;

/**
 * Floor on |k| = |ray.y * t / H| in the closed-form optical depth.
 *
 * The integral carries a removable singularity at k = 0 — a perfectly horizontal
 * ray, where the exact answer is simply `sigma * t`. Clamping the *dimensionless*
 * k rather than the ray slope bounds the error independently of how far the ray
 * runs: below 1e-4 the ratio `(1 - exp(-k)) / k` is within 5e-5 of 1, which is
 * the limit it is heading for anyway. Clamping the slope instead would let the
 * error grow with distance, and near-horizontal rays are exactly the ones that
 * run furthest.
 */
const K_EPSILON = 1e-4;

/**
 * Ceiling on the exponent inside the optical depth, and on the depth itself.
 *
 * A ray pointing down through an exponential layer accumulates density
 * exponentially, and the sky depth this pass sees is the camera's far plane —
 * tens of kilometres. Without a cap the intermediate `exp` overflows to infinity
 * and the fog comes out NaN instead of opaque. exp(-40) is 4e-18; anything past
 * that is opaque by any measure a half-float framebuffer can hold.
 */
const EXP_CEIL = 32;
const TAU_CEIL = 40;

/**
 * Effective sky path length as a multiple of the straight-up path, for the
 * ambient self-shadow.
 *
 * The honest quantity is the average of `1/cos(theta)` over the hemisphere,
 * which diverges — grazing sky directions see an unbounded slab. Any usable
 * number is therefore a choice, not a derivation, and 1.0 is chosen low on
 * purpose: this term exists to give the bank an interior that is darker than its
 * top, not to extinguish it. At 2.0 a merely moderate fog went to slate.
 */
const AMBIENT_PATH = 1.0;

/**
 * Soft ceiling on the Henyey–Greenstein lobe, relative to isotropic.
 *
 * The lobe's peak is `(1 + g) / (1 - g)^2` times isotropic, which is ~10x at
 * g = 0.6 and ~82x at the 0.85 the anisotropy control is limited to. On a cloud
 * that spike is confined to a silver lining along one edge; here it multiplies a
 * *full-screen* integral, and unbounded it turns the quarter of the sky around
 * the sun into a flat white disc.
 *
 * Applied as a Reinhard-style knee, `p * C / (p + C)`, rather than a hard clamp.
 * A hard clamp is what actually produces the flat disc — it truncates the top of
 * the lobe into a plateau with a visible edge. The knee stays monotonic, so the
 * glow keeps falling off from its centre however hard the anisotropy is pushed,
 * and it is near-transparent for the small values behind the viewer.
 *
 * It does not preserve the phase function's unit integral, and there is no
 * published prescription for clamping the evaluated phase — the literature
 * bounds `g` itself, sums several HG lobes, or substitutes Cornette–Shanks. The
 * honest justification for doing it anyway is that this is a single-scattering
 * integral, and single scattering has none of the multiple scattering that
 * softens the forward lobe in real fog; the knee is standing in for that, badly
 * but cheaply.
 */
const PHASE_CEIL = 8;

/**
 * Hard limit on `g`. Not for the spike — the knee handles that — but because the
 * phase function is only defined on the open interval and its denominator
 * collapses at the endpoints.
 */
const MAX_ANISOTROPY = 0.85;

/** Vertical feature size as a fraction of the horizontal one — banks are flat. */
const DETAIL_ASPECT = 0.22;

/**
 * Octaves in the bank field.
 *
 * This one number is the entire performance story of this pass. Measured at
 * 1600x900 on the reference GPU, a march step costs ~0.0025 ms with the bank
 * field off and ~0.013 ms per octave with it on: the noise is roughly five times
 * everything else in the loop body put together, and the rest of the effect —
 * ray setup, two closed-form medium evaluations, the phase lobe, the self-shadow
 * exponentials — rounds to nothing beside it.
 *
 * So one octave, which is also what Wronski's shipped volumetric fog used over
 * the same exponential height profile. Two looked slightly better and cost
 * slightly under twice as much, which is not a trade worth making for an effect
 * that already gets most of its structure from the height profile and the sun
 * self-shadow rather than from the noise.
 */
const DETAIL_OCTAVES = 1;

export class VolumetricFog {
  private readonly params: VolumetricFogParams;

  private camera: THREE.PerspectiveCamera | null = null;
  private clock = 0;
  private cameraHeight = 0;
  private disposed = false;

  /**
   * How much of the key light reaches a world point, 0..1.
   *
   * The march had **no occlusion term at all** — the header above says so — so
   * the in-scattering varied only by height and by the phase lobe, and there
   * were no crepuscular rays: none through the cloud deck, none over the ridge,
   * none through the rigging. The march itself was already the whole recipe;
   * this was the one missing piece.
   *
   * Set before `build`, because the graph is compiled once.
   */
  private sunOcclusion: ((worldPosition: any) => any) | null = null;

  /** Wind as a world vector, metres per second. Reused — never reallocated. */
  private readonly windVector = new THREE.Vector3(1, 0, 0);

  // --- medium ---------------------------------------------------------------
  /**
   * Extinction per metre at the base of the layer. Also the ceiling the marched
   * density is clamped to; see `medium`.
   */
  private readonly uSigmaBase = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.density);
  /**
   * Extinction per metre at the camera's own height.
   *
   * Derived on the CPU each frame because the camera position is known there,
   * which keeps one exponential out of every pixel. It is computed from the
   * camera height *clamped to the base*, so a camera that drops below sea level
   * cannot make the layer denser than its own floor — the profile grows without
   * bound downward and there is no fog under the water anyway.
   */
  private readonly uSigmaOrigin = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.density);
  private readonly uHeight = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.heightFalloff);
  private readonly uInvHeight = uniform(1 / DEFAULT_VOLUMETRIC_FOG_PARAMS.heightFalloff);
  private readonly uMaxDistance = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.maxDistance);

  // --- lighting -------------------------------------------------------------
  private readonly uColor = uniform(new THREE.Color(DEFAULT_VOLUMETRIC_FOG_PARAMS.color));
  private readonly uAmbient = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.ambient);
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(DEFAULT_VOLUMETRIC_FOG_PARAMS.sunColor));
  /** Sun brightness, already faded out by elevation — see `applyParams`. */
  private readonly uSunIntensity = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.sunIntensity);
  private readonly uAnisotropy = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.anisotropy);
  /** Sun path length as a multiple of the vertical one: 1 / max(sun.y, floor). */
  private readonly uSunPath = uniform(1.5);

  // --- banks ----------------------------------------------------------------
  private readonly uDetail = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.detail);
  /** Reciprocal feature size per axis, so the noise lookup is a multiply. */
  private readonly uDetailScale = uniform(new THREE.Vector3());
  /** Reciprocal of the step length at which bank detail is fully faded out. */
  private readonly uDetailLod = uniform(1 / 78);
  /** Accumulated drift, metres. Added before scaling, so it stays a world offset. */
  private readonly uWindOffset = uniform(new THREE.Vector3());

  // --- march ----------------------------------------------------------------
  /**
   * 1 when the effect should run at all, 0 when it must cost a single compare.
   * Uniform across the draw, so the branch is coherent for every wave on the GPU.
   */
  private readonly uEnabled = uniform(1);
  // Loosely typed: a uniform used as a dynamic `Loop` bound is not modelled by
  // the TSL typings.
  private readonly uSteps: any = uniform(DEFAULT_VOLUMETRIC_FOG_PARAMS.steps, 'int');
  private readonly uInvSteps = uniform(1 / DEFAULT_VOLUMETRIC_FOG_PARAMS.steps);

  // --- ray reconstruction ---------------------------------------------------
  // A post pass draws with the post-processor's own orthographic quad camera, so
  // the built-in camera matrix nodes describe that quad and not the scene. Both
  // matrices are pushed explicitly, exactly as `uNear`/`uFar` are, and for the
  // same reason.
  private readonly uInvProjection = uniform(new THREE.Matrix4());
  private readonly uCameraWorld = uniform(new THREE.Matrix4());
  private readonly uCameraPos = uniform(new THREE.Vector3());
  private readonly uNear = uniform(0.1);
  private readonly uFar = uniform(40000);

  constructor() {
    this.params = {
      ...DEFAULT_VOLUMETRIC_FOG_PARAMS,
      color: DEFAULT_VOLUMETRIC_FOG_PARAMS.color.clone(),
      sunDirection: DEFAULT_VOLUMETRIC_FOG_PARAMS.sunDirection.clone(),
      sunColor: DEFAULT_VOLUMETRIC_FOG_PARAMS.sunColor.clone(),
    };
    this.applyParams();
  }

  /**
   * Wraps a scene colour node and its depth texture.
   *
   * @param sceneColor A colour node. Either a texture node — e.g.
   *                   `scenePass.getTextureNode()` — in which case it is sampled
   *                   at this pixel's uv, or any already-composited colour node,
   *                   such as the output of another pass, in which case it is
   *                   used as-is. That second form is what lets this sit anywhere
   *                   in the chain.
   * @param sceneDepth The *depth texture* node, e.g.
   *                   `scenePass.getTextureNode( 'depth' )`. It must be a texture
   *                   node: the march is bounded by where the scene stops, and a
   *                   pre-linearised scalar node cannot be sampled at this pass's
   *                   own uv.
   * @returns The fogged node, to hand to `RenderPipeline.outputNode` or to the
   *          next pass in the chain.
   */
  build(sceneColor: unknown, sceneDepth: unknown): unknown {
    const colorNode: any = sceneColor;
    const depthNode: any = sceneDepth;

    if (colorNode === null || colorNode === undefined) {
      throw new Error('VolumetricFog.build: sceneColor is required.');
    }
    if (depthNode === null || depthNode === undefined || typeof depthNode.sample !== 'function') {
      throw new Error(
        'VolumetricFog.build: sceneDepth must be a texture node, e.g. scenePass.getTextureNode( "depth" ).',
      );
    }

    // `isTextureNode` rather than a duck-typed `.sample`: TSL registers method
    // chaining globally, so *every* node object answers to `.sample` whether or
    // not sampling it means anything.
    const isTexture = colorNode.isTextureNode === true;
    // PassTextureNode leaves `uvNode` null and falls back to the quad's uv.
    const baseUv: any = (isTexture ? colorNode.uvNode : null) ?? uv();

    return Fn(() => {
      const suv = vec2(baseUv).toVar('fogUv');
      const src = (isTexture ? colorNode.sample(suv) : vec4(colorNode)).toVar('fogSrc');
      const outRgb = vec3(src.rgb).toVar('fogOut');

      // Uniform-coherent branch: with the fog off the whole pass is one compare
      // and a copy.
      If(this.uEnabled.greaterThan(0.5), () => {
        // --- the view ray, in world space ------------------------------------
        // NDC from the screen uv. **y is flipped, and that is not cosmetic.**
        //
        // Screen uv runs top-down on the WebGPU backend and is explicitly flipped
        // to run top-down on the WebGL one, while NDC y runs bottom-up on both —
        // the same asymmetry `clipToScreenUV` in `ScreenSpaceReflection` exists to
        // absorb. Taking `suv.y * 2 - 1` therefore builds a ray pointing *down*
        // wherever the pixel looks up.
        //
        // It did not read as a broken ray, which is why it survived. It read as
        // fog: the sky was handed a downward ray into an exponential layer that
        // only thickens downward, so it integrated the whole column and every
        // preset rendered as a white-out, while the water was handed an upward
        // ray, left the layer immediately and got no fog at all. The give-away
        // was that the result did not respond to density — a 175x sweep produced
        // the same white, because both ends of it were saturated.
        const ndc = vec2(suv.x.mul(2).sub(1), suv.y.mul(-2).add(1)).toVar('fogNdc');
        const viewH = this.uInvProjection.mul(vec4(ndc.x, ndc.y, -1, 1)).toVar('fogViewH');
        const viewDir = normalize(viewH.xyz.div(viewH.w)).toVar('fogViewDir');
        const rd = normalize(this.uCameraWorld.mul(vec4(viewDir, 0)).xyz).toVar('fogRd');

        // The depth buffer measures along the camera's forward axis; the ray
        // needs it along *this* pixel's direction, which is longer off-centre.
        // Sky pixels come back at the far plane, which is what we want: looking
        // at nothing means looking through the entire depth of the layer.
        const axial = viewDir.z.negate().max(1e-3).toVar('fogAxial');
        const sceneZ = perspectiveDepthToViewZ(
          depthNode.sample(suv).r,
          this.uNear,
          this.uFar,
        ).negate();
        //
        // **Bounded at `maxDistance`, sky and sea alike.** This is a local sea
        // fog, not the atmosphere: past the march the haze belongs to
        // `AerialPerspective` (geometry) and to the dome (sky), which already
        // carry the low-altitude path toward the horizon. Integrating this layer
        // out to the far plane as well counted that path twice, and on exactly
        // the rays where it is longest — a sky pixel one row above the horizon
        // ran tens of kilometres through the layer while the sea pixel one row
        // below stopped a few kilometres out. The sun's forward lobe on that
        // extra path, unoccluded because the tail took no cloud shadow, was the
        // white seam along every horizon (A/B critic: reflections-1, sky-3,
        // waves-4, waves-5, foam-3). Measured with the sun term zeroed the band
        // went away; with this bound the sea's last row and the sky's first row
        // integrate the same path, so they meet without a step.
        const dist = sceneZ.div(axial).max(0).min(this.uMaxDistance).toVar('fogDist');

        // Rate of change of the height exponent per metre travelled. The whole
        // closed form is a function of this and the origin density.
        const slope = rd.y.mul(this.uInvHeight).toVar('fogSlope');
        const sigmaOrigin = this.uSigmaOrigin;

        // --- transmittance, in closed form -----------------------------------
        const transmittance = this.transmittanceAt(dist, sigmaOrigin, slope).toVar('fogT');
        const tEnd = dist.min(this.uMaxDistance).toVar('fogEnd');

        // --- the phase lobe, once for the whole ray --------------------------
        // A directional sun subtends the same scattering angle at every point on
        // a straight ray, so this is a per-pixel constant and has no business
        // inside the loop.
        //
        // `rd` runs camera to sample and `uSunDir` points at the sun, so
        // `rd . sunDir` is the cosine of the *deflection* angle: both the light's
        // propagation direction and the scattered direction reverse relative to
        // these two vectors, and the two sign flips cancel. Looking toward the
        // sun is therefore forward scattering, which is what a positive `g` has
        // to brighten. Getting this backwards is the classic way to end up with
        // haze that darkens toward the light.
        const phase = softCeiling(
          henyeyGreenstein(rd.dot(this.uSunDir), this.uAnisotropy),
          PHASE_CEIL,
        ).toVar('fogPhase');

        // --- in-scattering ---------------------------------------------------
        //
        // Cells are distributed as t = tEnd * s^2 rather than uniformly. The
        // weight this integral carries is sigma(t) * T(t), and both factors are
        // largest near the camera — one because the viewer is usually inside the
        // layer, the other because transmittance only ever falls. Uniform steps
        // spend most of their samples out where the integrand has already decayed
        // to nothing. Same reasoning as the exponential depth slices of a froxel
        // volume, minus the volume.
        //
        // Each cell's weight is the exact `T(start) - T(end)`, carried forward so
        // the boundary evaluated at the end of one cell is reused as the start of
        // the next: one closed-form evaluation per step, and a sum that telescopes
        // to exactly `1 - T(tEnd)` no matter how the range was divided. The dither
        // then only chooses where *inside* each cell the lighting and the bank
        // field get sampled. Step count changes the texture of the result and
        // never its energy.
        const dither = interleavedGradientNoise(screenCoordinate).toVar('fogDither');
        const cell = tEnd.mul(this.uInvSteps).mul(this.uInvSteps).toVar('fogCell');
        const acc = vec3(0, 0, 0).toVar('fogAcc');
        const boundary = float(1).toVar('fogBoundary');

        Loop(this.uSteps, ({ i }: any) => {
          // Far edge of this cell, and the transmittance there.
          const sFar = float(i).add(1).mul(this.uInvSteps).toVar('fogSFar');
          const tFar = tEnd.mul(sFar).mul(sFar).toVar('fogTFar');
          const reached = this.transmittanceAt(tFar, sigmaOrigin, slope).toVar('fogReached');
          const weight = boundary.sub(reached).max(0).toVar('fogWeight');
          boundary.assign(reached);

          // Stochastic sample inside the cell, for everything that varies.
          const sMid = float(i).add(dither).mul(this.uInvSteps).toVar('fogSMid');
          const t = tEnd.mul(sMid).mul(sMid).toVar('fogT2');
          const sigma = this.sigmaAt(t, sigmaOrigin, slope).toVar('fogSigma');

          // Crepuscular rays. The cell's own sample point is asked how much key
          // light reaches it; where a cloud is in the way, the sun term drops and
          // the surrounding air stays lit, and the difference between the two
          // *is* the shaft. `Nt3XDM`'s Buffer C is this and nothing more:
          // jittered start, a handful of steps, a shadow function at each,
          // weighted by phase. Every part of that was already here.
          const occlusion =
            this.sunOcclusion === null
              ? null
              : this.sunOcclusion(this.uCameraPos.add(rd.mul(t))).toVar('fogOcc');
          const light = this.lightAt(sigma, phase, occlusion).toVar('fogLight');

          If(this.uDetail.greaterThan(0.001), () => {
            const p = this.uCameraPos.add(rd.mul(t)).toVar('fogP');
            const q = p.add(this.uWindOffset).mul(this.uDetailScale).toVar('fogQ');
            // The one place in this loop that costs real time — see
            // DETAIL_OCTAVES, and note that a previous effect in this project
            // evaluated two 3D lattices per march step and took the frame from
            // milliseconds to seconds.
            const n = mx_fractal_noise_float(q, DETAIL_OCTAVES, 2.0, 0.5, 1.0).toVar('fogN');

            // Detail the march cannot resolve is not detail, it is noise. Cells
            // grow quadratically toward the far end of the march, and past
            // roughly the feature size the bank field is being point-sampled far
            // below its Nyquist rate — which shows up as banks crawling as the
            // camera turns. Fade it out with the cell size instead. Same
            // reasoning as the cloud layer's detail fade.
            const dt = cell.mul(float(i).mul(2).add(1)).toVar('fogDt');
            const lod = float(1).sub(dt.mul(this.uDetailLod)).clamp(0, 1).toVar('fogLod');
            light.mulAssign(float(1).add(n.mul(this.uDetail).mul(lod)).clamp(0.05, 2.2));
          });

          acc.addAssign(light.mul(weight));
        });

        // No tail past the march: `dist` is bounded at `maxDistance` above, so
        // `tEnd` is the whole ray and `boundary` already equals `transmittance`.
        // The analytic tail that used to close the segment out to the far plane
        // was the source of the horizon seam; see the note on `dist`.
        outRgb.assign(src.rgb.mul(transmittance).add(acc));
      });

      return vec4(outRgb, src.a);
    })();
  }

  setParams(params: Partial<VolumetricFogParams>): void {
    const p = this.params;
    if (params.color !== undefined) p.color.copy(params.color);
    if (params.sunDirection !== undefined) p.sunDirection.copy(params.sunDirection);
    if (params.sunColor !== undefined) p.sunColor.copy(params.sunColor);
    if (params.density !== undefined) p.density = params.density;
    if (params.baseHeight !== undefined) p.baseHeight = params.baseHeight;
    if (params.heightFalloff !== undefined) p.heightFalloff = params.heightFalloff;
    if (params.ambient !== undefined) p.ambient = params.ambient;
    if (params.sunIntensity !== undefined) p.sunIntensity = params.sunIntensity;
    if (params.anisotropy !== undefined) p.anisotropy = params.anisotropy;
    if (params.maxDistance !== undefined) p.maxDistance = params.maxDistance;
    if (params.detail !== undefined) p.detail = params.detail;
    if (params.detailScale !== undefined) p.detailScale = params.detailScale;
    if (params.windSpeed !== undefined) p.windSpeed = params.windSpeed;
    if (params.windDirection !== undefined) p.windDirection = params.windDirection;
    if (params.steps !== undefined) p.steps = params.steps;
    this.applyParams();
  }

  getParams(): Readonly<VolumetricFogParams> {
    return this.params;
  }

  /**
   * March steps for the current tier. 0 disables the effect entirely.
   *
   * Separate from `setParams` only because the tier changes it on its own
   * schedule, independently of anything an artist touched.
   */
  setSteps(steps: number): void {
    this.params.steps = steps;
    this.applyParams();
  }

  /**
   * The scene camera. Required for depth linearisation and for rebuilding the
   * world-space view ray; see the backend note at the top of this file.
   */
  /**
   * Supplies the occlusion function the march samples. Must be called before
   * `build`, which compiles the graph.
   *
   * Ask for the cheapest variant the source offers: this is evaluated once per
   * march cell, so its cost is multiplied by the step count and Max marches 56.
   */
  setSunOcclusion(occlusion: (worldPosition: any) => any): void {
    this.sunOcclusion = occlusion;
  }

  setCamera(camera: THREE.PerspectiveCamera): void {
    this.camera = camera;
    this.cameraHeight = camera.position.y;
    this.refreshOriginDensity();
  }

  /** Rewinds the drift, for reproducible captures. */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.refreshWind();
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.refreshWind();

    const camera = this.camera;
    if (camera === null) return;

    this.uNear.value = camera.near;
    this.uFar.value = camera.far;

    // `matrixWorld` is read rather than recomputed: the camera has already been
    // updated for this frame by the director, and the post pass runs after it.
    camera.updateMatrixWorld();
    (this.uInvProjection.value as THREE.Matrix4).copy(camera.projectionMatrixInverse);
    (this.uCameraWorld.value as THREE.Matrix4).copy(camera.matrixWorld);
    const origin = this.uCameraPos.value as THREE.Vector3;
    origin.setFromMatrixPosition(camera.matrixWorld);

    if (origin.y !== this.cameraHeight) {
      this.cameraHeight = origin.y;
      this.refreshOriginDensity();
    }
  }

  /**
   * The pass owns no geometry, material, texture or render target — it
   * contributes a node graph to whichever pipeline consumed it, and that owns the
   * compiled material. `dispose` therefore only stops the clock so a stale
   * reference cannot keep driving uniforms.
   */
  dispose(): void {
    this.disposed = true;
    this.camera = null;
  }

  // -------------------------------------------------------------------------

  /**
   * Extinction per metre at distance `t` along the ray.
   *
   * Clamped at the layer's base value, which makes the profile a constant slab
   * below `baseHeight` rather than an unbounded one. There is no fog under the
   * water, and the unbounded form would otherwise dominate any ray that dipped
   * below the surface.
   */
  private sigmaAt(t: any, sigmaOrigin: any, slope: any): any {
    return sigmaOrigin.mul(exp(slope.mul(t).negate().min(EXP_CEIL))).min(this.uSigmaBase);
  }

  /**
   * Transmittance from the camera to distance `t`, in closed form.
   *
   * For density that varies only with height,
   * `sigma(y) = sigma_b * exp(-(y - base) / H)`, the optical depth along a ray
   * leaving `y0` with vertical slope `dy` integrates exactly:
   *
   *   tau(t) = sigma_0 * t * (1 - exp(-k)) / k,   k = t * dy / H
   *
   * with `sigma_0` the extinction at the origin. This is Wenzel's height-fog
   * integral (SIGGRAPH 2006), arranged so the path length `t` stays an explicit
   * factor and the only division is by the *dimensionless* `k`. That arrangement
   * is the point: writing it as `sigma_0 * (1 - exp(-k)) / (dy/H)` is
   * algebraically identical but divides by the ray slope, so the error of any
   * guard on that slope grows with how far the ray runs — and near-horizontal
   * rays, the ones the guard exists for, are exactly the ones that run furthest.
   *
   * The limit at k = 0 is `sigma_0 * t`: a horizontal ray through a column of
   * constant density. Wenzel reaches it with a hard branch, explicitly so no
   * floating-point special can escape into tone mapping; K_EPSILON reaches it
   * with a clamp instead, which is branchless and bounds the error at 5e-5
   * whatever the distance.
   */
  private transmittanceAt(t: any, sigmaOrigin: any, slope: any): any {
    const k = slope.mul(t).toVar();
    const magnitude = k.abs().max(K_EPSILON);
    const kSafe = select(k.lessThan(0), magnitude.negate(), magnitude).toVar();

    const decay = exp(kSafe.negate().min(EXP_CEIL));
    const tau = sigmaOrigin.mul(t).mul(float(1).sub(decay)).div(kSafe).toVar();

    // Cap at the constant-density integral, for the same reason `sigmaAt` caps
    // the density it returns.
    //
    // The medium this pass models does not keep thickening below the layer's
    // base -- `sigmaAt` clamps to `uSigmaBase` there, because a fog bank sitting
    // on the sea does not become arbitrarily dense underneath it. The closed form
    // had no such cap, so for a downward ray the in-scattering the march summed
    // and the transmittance multiplying it described two different media, and the
    // disagreement grew exponentially with how far below the base the ray ran.
    //
    // `sigma_b * t` is the exact optical depth of the capped medium once the ray
    // is fully inside the constant-density region, and a strict upper bound
    // before that, so the minimum is right where it matters and conservative
    // where it does not.
    tau.assign(tau.min(this.uSigmaBase.mul(t)));

    return exp(tau.clamp(0, TAU_CEIL).negate());
  }

  /**
   * Radiance scattered toward the viewer from a point of extinction `sigma`.
   *
   * `sigma * H` is the optical depth from that point straight up to the top of
   * the layer — the same closed form as `medium`, evaluated for a vertical ray
   * of infinite length, where it collapses to a single multiply. Scaling it by
   * `1 / sun.y` gives the depth along the slanted path to the sun, and by a fixed
   * factor gives a stand-in for the sky. Two exponentials, and the bank gets a
   * lit top and a shadowed interior without a shadow map or a secondary march.
   *
   * That `1 / sun.y` is the plane-parallel secant law — the small-angle limit of
   * the Chapman function, and the same approximation atmospheric solvers replace
   * with a transmittance LUT once curvature starts to matter. For a layer whose
   * scale height is tens of metres it never does. What does matter is that the
   * secant diverges at the horizon where the true value stays finite, which is
   * why `uSunPath` carries a floor rather than being a raw reciprocal: without it
   * the entire layer goes black at exactly the elevation fog is most worth
   * looking at.
   */
  private lightAt(sigma: any, phase: any, occlusion: any): any {
    const vertical = sigma.mul(this.uHeight).toVar();
    const sky = exp(vertical.mul(AMBIENT_PATH).negate());
    const sun = exp(vertical.mul(this.uSunPath).negate());
    // Occlusion multiplies the *sun* term only. The ambient term is the sky
    // dome, and a point in a cloud's shadow still sees most of the sky — which
    // is exactly why a shaft reads as a bright wedge in dim air rather than as a
    // hole cut in a lit volume.
    const direct = this.uSunColor.mul(this.uSunIntensity).mul(phase).mul(sun);
    return this.uColor
      .mul(this.uAmbient)
      .mul(sky)
      .add(occlusion === null ? direct : direct.mul(occlusion));
  }

  private applyParams(): void {
    const p = this.params;

    const density = Math.max(0, p.density);
    const falloff = Math.max(0.5, p.heightFalloff);
    this.uSigmaBase.value = density;
    this.uHeight.value = falloff;
    this.uInvHeight.value = 1 / falloff;
    this.uMaxDistance.value = Math.max(1, p.maxDistance);

    this.uColor.value.copy(p.color);
    this.uAmbient.value = Math.max(0, p.ambient);
    this.uSunColor.value.copy(p.sunColor);
    this.uAnisotropy.value = clampNumber(p.anisotropy, -MAX_ANISOTROPY, MAX_ANISOTROPY);

    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(p.sunDirection);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();

    // Once the sun is under the horizon its in-scattering has to go with it, or
    // the fog stays lit like midday against a night sky. Folded into the
    // intensity rather than exposed, so callers only have to push the vector.
    const above = smoothstepScalar(-0.06, 0.12, sun.y);
    this.uSunIntensity.value = Math.max(0, p.sunIntensity) * above;
    // Floored: the slant path to a sun on the horizon is unbounded, and the
    // in-scattering term would go to zero across the whole layer at exactly the
    // elevation where fog is at its most photogenic.
    this.uSunPath.value = 1 / Math.max(0.1, sun.y);

    this.uDetail.value = clampNumber(p.detail, 0, 1);
    const scale = Math.max(4, p.detailScale);
    // Banks are wide and shallow, so the field varies several times faster
    // vertically than horizontally. Without the anisotropy the noise reads as
    // spherical blobs floating in the layer rather than as stratified fog.
    (this.uDetailScale.value as THREE.Vector3).set(
      1 / scale,
      1 / (scale * DETAIL_ASPECT),
      1 / scale,
    );
    // Bank detail is gone by the time a march cell spans much of a feature.
    this.uDetailLod.value = 1 / (scale * 0.6);

    this.windVector.set(Math.cos(p.windDirection), 0, Math.sin(p.windDirection));
    this.windVector.multiplyScalar(Math.max(0, p.windSpeed));
    this.refreshWind();

    const steps = Math.max(0, Math.min(MAX_STEPS, Math.round(p.steps)));
    // steps === 0 is the Low tier, and density === 0 is the UI slider at rest:
    // either has to cost a single compare, not a march that adds nothing.
    this.uEnabled.value = steps > 0 && density > 0 ? 1 : 0;
    this.uSteps.value = Math.max(1, steps);
    this.uInvSteps.value = 1 / Math.max(1, steps);

    this.refreshOriginDensity();
  }

  /** The world offset applied to the bank field. Negated so banks drift *with* the wind. */
  private refreshWind(): void {
    (this.uWindOffset.value as THREE.Vector3)
      .copy(this.windVector)
      .multiplyScalar(-this.clock);
  }

  private refreshOriginDensity(): void {
    const p = this.params;
    const falloff = Math.max(0.5, p.heightFalloff);
    const above = Math.max(0, this.cameraHeight - p.baseHeight);
    this.uSigmaOrigin.value = Math.max(0, p.density) * Math.exp(-above / falloff);
  }
}

/**
 * Henyey–Greenstein phase, normalised so that isotropic scattering is exactly 1.
 *
 * The usual 1/(4*pi) normalisation integrates to one over the sphere, which is
 * correct but leaves every value two orders of magnitude below the radiance it
 * multiplies. Scaling by 4*pi expresses the lobe as a factor *relative to
 * isotropic*, so g = 0 leaves the in-scattering exactly as it was and the
 * anisotropy control only ever redistributes it.
 */
function henyeyGreenstein(cosTheta: any, g: any): any {
  const g2 = g.mul(g);
  const denom = pow(float(1).add(g2).sub(cosTheta.mul(g).mul(2)).max(1e-4), 1.5);
  return float(1).sub(g2).div(denom);
}

/** Reinhard-style knee. Monotonic, asymptotic to `ceiling`. See PHASE_CEIL. */
function softCeiling(x: any, ceiling: number): any {
  return x.mul(ceiling).div(x.add(ceiling));
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstepScalar(edge0: number, edge1: number, x: number): number {
  const t = clampNumber((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
