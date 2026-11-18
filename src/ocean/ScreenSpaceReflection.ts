import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  float,
  interleavedGradientNoise,
  mix,
  normalize,
  perspectiveDepthToViewZ,
  screenCoordinate,
  screenSize,
  screenUV,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
  viewportSharedTexture,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDownClamped } from '../core/tslMath';

import type { QualityTier } from '../core/QualityManager';

/**
 * Screen-space reflection for the ocean surface.
 *
 * ## Why this exists alongside the planar reflector
 *
 * `ocean/Reflections` argues, correctly, that a mirrored camera is exact for a
 * plane and that SSR cannot reflect what is off-screen. Both statements are
 * true. What the argument misses is that *the ocean is not a plane*. The surface
 * is displaced by three cascades of FFT waves, so a reflection mirrored about
 * y = 0 is only right where the water happens to be at mean level and pointing
 * straight up. Everywhere else — which is most of a rough sea — the mirrored
 * image is sampled at the wrong place and merely wobbled toward plausibility by
 * a normal-driven uv offset. That offset is a fudge with no geometry behind it,
 * and it is what reads as "odd": the reflection slides across wave faces instead
 * of being attached to them, and it is systematically wrong in the same
 * direction on every crest.
 *
 * SSR has the opposite error profile. It traces the *actual* reflected ray
 * against the *actual* depth buffer, so a reflection that lands on a wave face
 * lands where the ray says it should, and it stays there as the wave moves under
 * it. What SSR cannot do is invent geometry that is not on screen.
 *
 * The two therefore compose rather than compete, and that is how this module is
 * meant to be used: **pass the planar reflection (or the analytic sky) in as
 * `fallback`.** SSR takes over wherever it is confident and hands back smoothly
 * wherever it is not. That layering — screen-space trace over a coarser but
 * always-available source — is what Frostbite and Unreal both do, and for the
 * same reason: neither source is complete on its own.
 *
 * ## Why this is not a post pass
 *
 * The water is a forward-rendered transparent material. It draws *after* the
 * opaque scene, which means two things, and both of them are gifts:
 *
 *   1. The opaque colour and depth are already available to it, via
 *      `viewportSharedTexture()` and `viewportDepthTexture()` — the same two
 *      buffers `OceanMaterial` already reads for refraction. There is no
 *      G-buffer to build and no extra pass to schedule.
 *   2. **The water is not in the depth buffer it traces against.** Every SSR
 *      implementation that runs as a post pass has to bias the ray origin off
 *      the surface to stop it hitting the pixel it started from; that bias is a
 *      constant tuned against self-intersection at one end and detached contacts
 *      at the other, and it is a permanent source of artefacts. Here the problem
 *      does not exist: at the ray's origin the depth buffer holds the seafloor,
 *      metres further away, so the very first sample cannot be a false hit and
 *      no bias is needed at all.
 *
 * The cost of running in-material is that the trace happens per water fragment
 * at full resolution with no chance of temporal reuse or denoising. That rules
 * out the stochastic/GGX second-generation approach (Stachowiak, "Stochastic
 * Screen-Space Reflections", SIGGRAPH 2015), which is only viable behind a
 * temporal filter. This is a single mirror ray per pixel — first-generation SSR
 * — which is the right shape for a near-mirror surface anyway: water at
 * roughness 0.075 has a reflection lobe narrow enough that one ray is close to
 * the correct answer.
 *
 * ## The march
 *
 * Stepping happens in **screen space with perspective-correct depth**, after
 * Morgan McGuire and Michael Mara, "Efficient GPU Screen-Space Ray Tracing"
 * (JCGT 3(4), 2014). The two endpoints of the ray are projected to clip space;
 * the segment between their screen positions is walked with uniform steps, and
 * the ray's view distance is recovered at each step by interpolating **1/w**
 * linearly along that segment — which is exact, because reciprocal depth is the
 * quantity that is affine in screen space under a perspective projection.
 *
 * The alternative — advancing by a fixed number of metres in view space and
 * projecting each sample — is what most short SSR implementations do, and it is
 * wrong in a way that matters here. Uniform world steps project to wildly
 * non-uniform screen steps: near the camera each step jumps tens of pixels and
 * walks straight over thin geometry, while far away a dozen consecutive samples
 * land in the same texel and are pure waste. On an ocean, where the reflected
 * ray is often near-grazing and covers hundreds of metres for a handful of
 * pixels, that mismatch is severe. Uniform *screen* steps spend the budget where
 * the data actually is: one step, roughly one span of the depth buffer.
 *
 * The screen segment is additionally clipped to the viewport before the budget
 * is divided up, so a ray that leaves the frame after a fifth of its length
 * spends all its steps inside the frame rather than four fifths of them
 * sampling nothing.
 *
 * Deliberately *not* implemented: hierarchical (Hi-Z) tracing, per Uludag,
 * "Hi-Z Screen-Space Cone-Traced Reflections" (GPU Pro 5, 2014). Hi-Z is the
 * right answer when rays are long, numerous and unbounded, because it turns a
 * linear march into a logarithmic one. It needs a min-depth mip pyramid built
 * over the depth buffer every frame, which here would mean an extra pass and an
 * extra full-resolution target to serve a march that is already capped at a few
 * dozen taps on a surface covering part of the screen. The pyramid would cost
 * more than the march it accelerates.
 *
 * ## Fades
 *
 * Everything below is about the same thing: a screen-space trace is an
 * approximation with a sharply defined domain, and every boundary of that domain
 * has to be crossed smoothly or it shows up as a moving edge in the image. The
 * confidence returned by the trace is the product of five terms — screen edge,
 * ray-toward-camera, range, surface facing, and the coarse hit flag — and the
 * result is mixed toward `fallback` by it. A hard cut to the fallback is worse
 * than having no SSR at all, because the eye tracks discontinuities far more
 * readily than it tracks a slightly wrong reflection.
 *
 * ## Disabling
 *
 * Call `setQuality( 0 )` and the whole graph collapses to one uniform compare
 * and the fallback, at no bandwidth cost. The Low tier does this.
 */

/**
 * TSL node objects are structurally dynamic and the generated types cannot
 * express them, so node-typed locals are `any` by design. The module's public
 * API stays typed — nodes cross it as `unknown`.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

/** Hard ceiling on the march, for sanity rather than for compilation. */
const MAX_STEPS = 64;

/**
 * Suggested march budget per quality tier.
 *
 * Shaped like the existing `cloudSteps` / `godRaySteps` ladders so the tiers
 * stay legible next to each other. Two things make SSR cheaper per step than
 * either of those: it only runs on water fragments, not the whole frame, and a
 * step is a single point-sampled depth fetch rather than an evaluation of a
 * procedural field.
 *
 * `low` is 0 deliberately. That tier is the lightest, where the surface mixes
 * out its own depth-buffer refraction — SSR there would lean on a buffer the
 * rest of the material has already chosen not to show.
 */
export const DEFAULT_SSR_STEPS: Record<QualityTier, number> = {
  low: 0,
  medium: 12,
  high: 24,
  ultra: 32,
  max: 48,
};

/**
 * Optional overrides for where the trace reads the scene from.
 *
 * Both default to the viewport singletons, which is what makes the module
 * self-contained. Supplying them is worth it in one specific case: each distinct
 * `viewportSharedTexture()` / `viewportDepthTexture()` node instance in a frame
 * triggers its own `copyFramebufferToTexture`, because the update is keyed on
 * the node, not on the texture the nodes share. `OceanMaterial` already creates
 * one colour instance and three depth instances; leaving these `undefined` adds
 * a fifth and a sixth. At 1600x900 that is on the order of 20 microseconds of
 * copy per frame — small, but free to avoid if the surface ever hoists its own
 * taps into shared handles and passes them in here.
 */
export interface ScreenSpaceReflectionOptions {
  /** vec4 texture node carrying the opaque scene colour. */
  sceneColorNode?: unknown;
  /** Texture node carrying the opaque scene depth, sampled as `.r`. */
  sceneDepthNode?: unknown;
}

export class ScreenSpaceReflection {
  private camera: THREE.PerspectiveCamera | null = null;
  private disposed = false;

  /** Opaque scene colour, sampled once at the resolved hit. */
  private readonly sceneColor: any;
  /** Opaque scene depth, sampled once per march step. */
  private readonly sceneDepth: any;

  // --- march budget ---------------------------------------------------------
  /**
   * 0 or 1. Gates the entire trace behind one uniform-valued compare.
   *
   * A uniform branch is coherent across every fragment in the draw, so the
   * hardware skips the body outright rather than executing it under a mask.
   * That is what makes `setQuality( 0 )` genuinely free instead of merely
   * multiplied away — the same reasoning as the god-ray gate in
   * `underwater/UnderwaterPass`.
   */
  private readonly uEnabled = uniform(0);
  /**
   * Loosely typed: a uniform used as a dynamic `Loop` bound is not modelled by
   * the TSL typings.
   */
  private readonly uSteps: any = uniform(24, 'int');
  private readonly uInvSteps = uniform(1 / 24);
  private readonly uRefineSteps: any = uniform(4, 'int');

  // --- hit acceptance -------------------------------------------------------
  /**
   * Constant part of the thickness window, metres.
   *
   * The depth buffer records a front surface and says nothing about how deep the
   * object behind it goes, so a ray that passes *behind* a thin object looks
   * identical to one that lands *on* a thick one. The thickness window is the
   * assumption that closes that gap: a crossing counts as a hit only if the ray
   * entered the step no more than this far behind the recorded surface.
   *
   * Too small and reflections develop holes — the ray slips behind the hull
   * between two samples and never reports it. Too large and every distant
   * silhouette becomes a solid wall, which smears background colour into
   * reflections that should have shown sky. Sub-metre is right for a scene whose
   * reflective subject is a ship's hull.
   */
  private readonly uThickness = uniform(0.6);
  /**
   * Depth-proportional part of the thickness window, metres per metre.
   *
   * A constant world thickness is measured in the wrong units at distance: one
   * texel of the depth buffer spans centimetres underfoot and metres at two
   * hundred, so a window that is generous close up is narrower than the
   * buffer's own resolution far away, and hits at range are rejected for being
   * inside the quantisation error. Growing the window linearly in scene depth
   * keeps it a roughly constant number of *texels* deep, which is the unit the
   * uncertainty is actually in.
   */
  private readonly uThicknessSlope = uniform(0.015);

  // --- confidence -----------------------------------------------------------
  /**
   * Width of the screen-border fade, as a fraction of screen *height*.
   *
   * The single most visible SSR artefact, and the reason it gets its own
   * uniform. A reflected ray that ends near the frame edge is reflecting
   * something about to leave the frame; without a fade, the moment it does the
   * reflection vanishes in one frame and the eye reads it as a flicker attached
   * to the screen rather than to the water.
   *
   * Height, not uv, because a band measured in uv is nearly twice as wide in
   * pixels on the horizontal edges of a 16:9 frame as on the vertical ones, and
   * the asymmetry is visible as reflections surviving longer out of the sides
   * than out of the top. AMD's FidelityFX SSSR corrects for aspect the same way
   * and settles on 0.05. This is double that: on an ocean the reflected rays
   * travel *upward*, so they pile against the top border and cross it
   * constantly, and that is the one border worth being generous about.
   */
  private readonly uEdgeFade = uniform(0.1);
  /**
   * How far the reflected ray is allowed to travel, metres.
   *
   * Deliberately short by the standards of a general SSR implementation, and
   * long by the standards of a published one — Filament defaults to 3 world
   * units, Unity HDRP to 50 m. The published consensus is that SSR is a
   * near-field effect and that clamping it hard is what keeps it stable; the
   * countervailing fact here is that the subject worth reflecting is a ship,
   * which is tens of metres across and routinely a hundred metres off. 200 m
   * covers every framing the camera director produces and stops well short of
   * the range where aerial perspective has washed the reflection out anyway.
   */
  private readonly uRange = uniform(200);
  /** Per-tier reflection strength, 0..1. */
  private readonly uStrength = uniform(1);
  /**
   * How far the *ray* normal is pulled toward vertical before reflecting, 0..1.
   *
   * The single biggest quality lever, and the least obvious one. A screen-space
   * trace is a point sample of one ray per pixel, and the ocean hands it a
   * normal field with decimetre structure — so adjacent pixels fire at targets
   * metres apart, and a reflection that ought to be a coherent shape breaks into
   * a stipple where some rays find the hull and their neighbours sail past it.
   * It is worst exactly where the reflection is strongest, at grazing incidence,
   * because there a small change in normal swings the ray a long way.
   *
   * No amount of extra marching fixes it — the rays are diverging before the
   * march begins. What fixes it is tracing a calmer surface than the one being
   * shaded: the reflection loses some of its ripple and gains coherence. Every
   * engine that ships SSR on water does some version of this. Unity HDRP forces
   * transparent receivers to "evaluate the reflections as smooth" outright, and
   * Godot's water SSR add-on exposes it as a "Wave Normal Flatness" slider.
   *
   * Only the ray is affected. Fresnel, the specular highlight and the shading
   * normal are the caller's and are untouched, so the surface keeps its detail
   * and only what it reflects is smoothed.
   */
  private readonly uRayFlatten = uniform(0.35);

  // --- camera ---------------------------------------------------------------
  /**
   * Combined view-projection, world to clip.
   *
   * Supplied as a uniform rather than read from the built-in camera nodes, for
   * the reason `underwater/UnderwaterPass` spells out: the built-ins resolve to
   * whichever camera is driving the current render, and this material is also
   * drawn from the shadow camera and — when the planar reflector is active —
   * from a mirrored one. A trace that silently used those matrices while
   * sampling the main view's framebuffer would produce garbage that only shows
   * up in one pass.
   *
   * One matrix rather than separate view and projection: the march only ever
   * needs clip-space coordinates, and a perspective projection puts the positive
   * view distance straight into `clip.w`, so the depth track comes out of the
   * same multiply for free.
   */
  private readonly uViewProjection = uniform(new THREE.Matrix4());
  private readonly uCameraPosition = uniform(new THREE.Vector3());
  private readonly uNear = uniform(0.1);
  private readonly uFar = uniform(40000);

  constructor(options: ScreenSpaceReflectionOptions = {}) {
    this.sceneColor = (options.sceneColorNode ?? viewportSharedTexture()) as any;
    this.sceneDepth = (options.sceneDepthNode ?? viewportDepthTexture()) as any;
  }

  // ---------------------------------------------------------------- accessors

  /**
   * The camera the water is being viewed with.
   *
   * Required. Without it `update` has nothing to push and the trace runs against
   * an identity matrix, which projects every ray onto the same pixel.
   */
  setCamera(camera: THREE.PerspectiveCamera): void {
    this.camera = camera;
  }

  /**
   * Per-tier march cost.
   *
   * `steps` is the coarse march budget; 0 disables the trace entirely and the
   * node returns `fallback` after one uniform compare. See `DEFAULT_SSR_STEPS`
   * for the suggested ladder.
   *
   * Refinement is derived from it rather than exposed separately. Bisection only
   * pays for itself in proportion to how coarse the march was — a 48-step march
   * already lands within a pixel or two of the contact and a fifth halving buys
   * almost nothing, while a 12-step march needs the help but also cannot afford
   * to spend half its budget on it.
   */
  setQuality(steps: number, thickness = 0.6): void {
    const count = Math.max(0, Math.min(MAX_STEPS, Math.round(steps)));

    this.uEnabled.value = count > 0 ? 1 : 0;
    this.uSteps.value = Math.max(1, count);
    this.uInvSteps.value = 1 / Math.max(1, count);
    this.uRefineSteps.value = count >= 32 ? 5 : count >= 20 ? 4 : count >= 10 ? 3 : 2;

    this.uThickness.value = Math.max(0.01, thickness);
  }

  /** How much of the traced reflection reaches the surface, 0..1. */
  setStrength(amount: number): void {
    this.uStrength.value = Math.max(0, Math.min(1, amount));
  }

  /**
   * How far the ray normal is pulled toward vertical, 0..1. See `uRayFlatten`.
   *
   * Raise it in a rough sea and lower it in a calm one — the trade is ripple in
   * the reflection against coherence of it, and which way that trade should fall
   * depends entirely on how much ripple there is to lose.
   */
  setRayFlatten(amount: number): void {
    this.uRayFlatten.value = Math.max(0, Math.min(1, amount));
  }

  /**
   * How far a reflected ray may travel, metres.
   *
   * Not a cost lever — the step count is fixed, so a longer range spreads the
   * same budget over more distance and loses precision rather than gaining time.
   * It is a *reach* control: raise it to catch reflections of things far across
   * the water, lower it to keep contact reflections around the hull sharp.
   */
  setRange(metres: number): void {
    this.uRange.value = Math.max(1, metres);
  }

  /**
   * Width of the screen-border confidence fade, as a fraction of screen height.
   * Applied equally to all four borders in pixel terms. See `uEdgeFade`.
   */
  setEdgeFade(width: number): void {
    this.uEdgeFade.value = Math.max(0, Math.min(0.5, width));
  }

  /**
   * Must be called once per frame, before the water is drawn.
   *
   * `dt` is unused: the graph has no time dependence, and deliberately so. The
   * obvious use for it would be to advance the dither so successive frames
   * sample different sub-steps, but that only helps behind a temporal filter,
   * and this project has none — all it would produce is a march whose banding
   * crawls. The parameter is kept so the module has the same lifecycle shape as
   * every other updatable in the codebase.
   */
  update(_dt: number): void {
    if (this.disposed) return;

    const camera = this.camera;
    if (camera === null) return;

    // `matrixWorld` and its inverse are read rather than recomputed: the
    // director has already placed the camera for this frame and the material is
    // drawn after it. `multiplyMatrices` writes in place, so this allocates
    // nothing per frame.
    camera.updateMatrixWorld();
    (this.uViewProjection.value as THREE.Matrix4).multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );
    (this.uCameraPosition.value as THREE.Vector3).setFromMatrixPosition(camera.matrixWorld);

    this.uNear.value = camera.near;
    this.uFar.value = camera.far;
  }

  /**
   * Owns no render target, material or geometry — it contributes a subgraph to
   * whichever material consumed it, and that material owns the compiled shader.
   * `dispose` therefore only drops the camera reference so a stale one cannot
   * keep driving uniforms after the scene is gone.
   */
  dispose(): void {
    this.disposed = true;
    this.camera = null;
  }

  // ------------------------------------------------------------------- graph

  /**
   * The reflected colour for a shaded point, already blended with `fallback`.
   *
   * **Returns a vec3 node**, not a `{ color, confidence }` pair. The blend is
   * the part that has to be right — a caller compositing it by hand would have
   * to reproduce the same smooth handover on every miss, and every place that
   * got it slightly wrong would show as an edge. `traceNode` is there for the
   * cases that genuinely need the raw confidence.
   *
   * **Must be called from inside the material's fragment `Fn`.** The subgraph
   * contains `If` and `Loop` blocks, which need an enclosing shader function to
   * be emitted into, and `worldPosition` / `worldNormal` are per-fragment
   * variables belonging to that scope. Call it once, at build time; the graph is
   * static from then on and every tunable is a uniform.
   *
   * @param worldPosition vec3 node — the shaded point, in world space.
   * @param worldNormal   vec3 node — the perturbed surface normal, world space.
   *                      Pass the *final* shading normal, the one the Fresnel
   *                      term uses, or the reflection will not agree with the
   *                      surface it is sitting on.
   * @param fallback      vec3 node — the colour used wherever the trace has no
   *                      answer. The analytic sky works; the planar reflection
   *                      works better, because then a miss falls back to
   *                      something that still contains the ship.
   */
  reflectionNode(worldPosition: unknown, worldNormal: unknown, fallback: unknown): unknown {
    if (fallback === null || fallback === undefined) {
      throw new Error(
        'ScreenSpaceReflection.reflectionNode: `fallback` is required — a screen-space ' +
          'trace misses on most pixels of an open ocean, and without somewhere to hand ' +
          'back to those pixels would be black.',
      );
    }

    const sky = fallback as any;
    const traced = this.traceNode(worldPosition, worldNormal) as any;

    return mix(sky, traced.rgb, traced.a);
  }

  /**
   * The raw trace: `vec4( reflected rgb, confidence )`.
   *
   * `confidence` is 0 on a miss and the rgb is undefined there, so it is only
   * ever meaningful weighted by the alpha. Exposed for callers that want to
   * fold the confidence into something else — a roughness-driven blur select,
   * say, or a debug view of where the trace succeeds.
   */
  traceNode(worldPosition: unknown, worldNormal: unknown): unknown {
    if (worldPosition === null || worldPosition === undefined) {
      throw new Error('ScreenSpaceReflection.traceNode: `worldPosition` is required.');
    }
    if (worldNormal === null || worldNormal === undefined) {
      throw new Error('ScreenSpaceReflection.traceNode: `worldNormal` is required.');
    }

    const origin = worldPosition as any;
    const surfaceNormal = worldNormal as any;

    return Fn(() => {
      const hitColor = vec3(0).toVar('ssrColor');
      const confidence = float(0).toVar('ssrConfidence');

      // Uniform-coherent gate. At `setQuality( 0 )` the whole trace costs one
      // compare and nothing else — no texture fetch, no march, no bandwidth.
      If(this.uEnabled.greaterThan(0.5), () => {
        const p0 = vec3(origin).toVar('ssrP0');
        const n = normalize(vec3(surfaceNormal)).toVar('ssrN');

        // View vector points *from* the surface *to* the camera, matching the
        // convention `OceanMaterial` uses, so the two agree about which way is
        // out.
        const toCamera = this.uCameraPosition.sub(p0).toVar('ssrToCam');
        const v = normalize(toCamera).toVar('ssrV');

        // The ray is fired off a calmer surface than the one being shaded. See
        // `uRayFlatten` — this one line is what separates a coherent reflection
        // from a stipple on a rough sea.
        const rayNormal = normalize(mix(n, vec3(0, 1, 0), this.uRayFlatten)).toVar('ssrRayN');
        const dir = normalize(v.negate().reflect(rayNormal)).toVar('ssrDir');

        // --- gates that can be decided before any fetch --------------------

        // Wave slopes routinely tip the shading normal past the point where the
        // surface faces the viewer at all. `reflect` on a back-facing normal
        // returns a ray pointing *into* the water, which would happily trace
        // down onto the seafloor and paint sand onto a crest. The window is
        // narrow on purpose: grazing angles are where reflections matter most
        // on water, so this must kill the degenerate case and nothing else.
        const facingSurface = n.dot(v).smoothstep(0, 0.06).clamp(0, 1).toVar('ssrFacing');

        // Fade as the reflected ray turns back toward the viewer. For a mirror,
        // `dot( R, V )` is `2( N·V )² - 1`: it is -1 at grazing incidence, where
        // the ray shoots away across the scene and SSR is at its best, and +1
        // when looking straight down, where the ray comes straight back at the
        // camera and the only thing it could reflect is the viewer — who is, by
        // construction, not in the frame. There is nothing to trace there, so
        // the fallback is not a compromise but the correct answer.
        //
        // The window is wider than published practice — Killing Floor 2 fades
        // over 0.25 to 0.5, which reaches zero thirty degrees off vertical.
        // Deliberately: a ray that turns back at the viewer crosses the near
        // plane, the near-plane clip then leaves a very short segment, and the
        // march simply fails to find anything and returns the fallback. The
        // failure is already graceful, so the only thing a tighter window buys
        // is the loss of the reflections that band still finds.
        const backAtViewer = dir.dot(v).toVar('ssrBack');
        const notAtViewer = float(1)
          .sub(backAtViewer.smoothstep(0.2, 0.8))
          .clamp(0, 1)
          .toVar('ssrNotAtViewer');

        // Reject rays that descend into the water.
        //
        // Because the water is absent from the depth buffer — the property that
        // makes everything else here simpler — nothing stops a downward ray at
        // the surface. It sails through into the water column and reports a hit
        // on the seafloor, painting sand onto a wave crest. Every published
        // account of SSR on water names this as the failure that has to be
        // handled explicitly.
        //
        // A *slope* test, not a depth-below-the-surface test. The latter is the
        // more obvious formulation and it was tried: carry the ray parameter to
        // the hit, multiply out how far below the shading point it landed, and
        // reject beyond a tolerance. It is worse, and instructively so. Over a
        // long ray the launch point's height stops being a meaningful datum for
        // where the water is — the surface has metres of relief — so the
        // tolerance has to grow with distance, at which point the test has
        // become a slope test with extra steps. What it does in the meantime is
        // stipple grazing reflections, because at grazing incidence the
        // per-pixel wave normal pushes the ray a few thousandths either side of
        // horizontal and the two halves land on opposite sides of a fixed depth
        // tolerance. Grazing is where Fresnel is strongest and where this whole
        // module earns its cost; damaging it to fix an artefact the slope test
        // already covers is a bad trade.
        //
        // The ramp is wide, and that width was measured rather than guessed. A
        // tight window either side of zero looks like the obviously correct
        // choice and produces a hard cross-hatch in exactly the grazing band
        // where the reflection matters: the per-pixel wave normal swings the ray
        // a tenth either side of horizontal there, so adjacent pixels land on
        // opposite sides of the threshold and the reflection breaks into a
        // stipple. Spread over a third of a unit the same modulation becomes a
        // gentle shading of the reflection, which is a fair description of what
        // is physically happening — half those wave faces really are pointing at
        // water rather than at sky.
        //
        // Full weight at level, zero by a slope of a third: at that angle a ray
        // is sixty metres down by the end of its range, which is well past any
        // seafloor this scene has.
        //
        // It also means the trace shuts off almost entirely when the camera is
        // submerged and the surface is drawn from below, leaving the fallback to
        // carry the total-internal-reflection look on its own.
        const notDiving = dir.y.smoothstep(-0.32, 0).clamp(0, 1).toVar('ssrNotDiving');

        const gate = facingSurface.mul(notAtViewer).mul(notDiving).toVar('ssrGate');

        // Skipping the march where it is already worthless is a real saving on
        // a top-down shot, and the term varies smoothly across the surface, so
        // neighbouring fragments overwhelmingly agree and the branch stays
        // coherent within a warp.
        If(gate.greaterThan(0.002), () => {
          // --- project the ray to clip space -----------------------------
          //
          // Clip space is affine in world position, so the segment between the
          // two projected endpoints can be interpolated directly — no need to
          // re-project anything after clipping.
          const clipA = this.uViewProjection.mul(vec4(p0, 1)).toVar('ssrClipA');
          const clipB = this.uViewProjection
            .mul(vec4(p0.add(dir.mul(this.uRange)), 1))
            .toVar('ssrClipB');

          // A THREE perspective projection has (0,0,-1,0) as its last row, so
          // `clip.w` is exactly the positive distance along the camera's
          // forward axis — the same quantity the depth buffer decodes to. That
          // holds under both coordinate systems and under a reversed depth
          // buffer, which is why the depth track rides on `w` rather than on
          // NDC z.
          const wA = clipA.w.toVar('ssrWA');
          const wB = clipB.w.toVar('ssrWB');

          // Clip the far endpoint to just in front of the near plane. Past it
          // `w` goes to zero and then negative, and the perspective divide
          // turns the screen segment inside out.
          const nearLimit = this.uNear.mul(1.02).toVar('ssrNearLimit');
          const tNear = wB
            .lessThan(nearLimit)
            .select(nearLimit.sub(wA).div(wB.sub(wA).min(-1e-4)), float(1))
            .clamp(0.02, 1)
            .toVar('ssrTNear');
          const clipEnd = mix(clipA, clipB, tNear).toVar('ssrClipEnd');

          // --- the screen segment ------------------------------------------
          //
          // The start is `screenUV` rather than the projection of `p0`. They
          // agree to within float error by construction — the rasteriser put
          // this fragment here — and `screenUV` is the coordinate the buffers
          // are actually indexed by, so anchoring to it keeps the first sample
          // exactly on the pixel it belongs to.
          const uvA = vec2(screenUV).toVar('ssrUvA');
          const uvB = clipToScreenUV(clipEnd).toVar('ssrUvB');

          // Reciprocal depth. This is the quantity that varies linearly along a
          // screen-space line under perspective (McGuire & Mara, JCGT 2014);
          // depth itself does not, and interpolating it directly is the classic
          // way to get a march that is dense at one end and sparse at the
          // other.
          const kA = float(1).div(wA.max(1e-4)).toVar('ssrKA');
          const kB = float(1).div(clipEnd.w.max(1e-4)).toVar('ssrKB');

          // Clip the segment to the viewport, using the slab method on the unit
          // square. Everything past the border is unreadable, and spending
          // steps out there is the difference between a budget that samples the
          // frame and one that samples the void.
          const delta = uvB.sub(uvA).toVar('ssrDelta');
          const limitX = delta.x
            .greaterThanEqual(0)
            .select(uvA.x.oneMinus(), uvA.x)
            .div(delta.x.abs().max(1e-5))
            .toVar('ssrLimitX');
          const limitY = delta.y
            .greaterThanEqual(0)
            .select(uvA.y.oneMinus(), uvA.y)
            .div(delta.y.abs().max(1e-5))
            .toVar('ssrLimitY');
          const sMax = limitX.min(limitY).min(1).max(0).toVar('ssrSMax');

          // --- coarse march -------------------------------------------------
          const stepS = sMax.mul(this.uInvSteps).toVar('ssrStepS');

          // Start each pixel at a different fraction of its first step.
          // Interleaved gradient noise (Jimenez, "Next Generation Post
          // Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014) is
          // low-discrepancy over a small neighbourhood, so neighbouring pixels
          // get well-spread offsets rather than merely different ones. Without
          // it every pixel samples the same set of depth planes and a step
          // count this low draws them as visible contour bands across the
          // reflection; with it the same error becomes fine noise, which on a
          // rippled surface is invisible.
          const dither = interleavedGradientNoise(screenCoordinate).toVar('ssrDither');

          const s = stepS.mul(dither).toVar('ssrS');
          const sPrev = float(0).toVar('ssrSPrev');
          const wPrev = wA.toVar('ssrWPrev');
          const found = float(0).toVar('ssrFound');
          const sLo = float(0).toVar('ssrSLo');
          const sHi = float(0).toVar('ssrSHi');
          // Scene depth at `sHi`, carried out of the march and kept up to date
          // through refinement. It is what turns the binary hit test into a
          // graded one at the end, and it costs nothing — the value has already
          // been fetched.
          const sceneAtHi = float(0).toVar('ssrSceneAtHi');

          Loop(this.uSteps, () => {
            const uvS = mix(uvA, uvB, s).toVar('ssrUvS');
            const wS = float(1).div(mix(kA, kB, s).max(1e-8)).toVar('ssrWS');
            const sceneW = this.sceneDistance(uvS).toVar('ssrSceneW');
            const acceptWindow = this.uThickness
              .add(sceneW.mul(this.uThicknessSlope))
              .toVar('ssrWindow');

            // A *segment* test, not a point test. The ray is behind the
            // recorded surface now, and it was not already buried more than a
            // thickness behind it at the previous sample — so the crossing
            // happened inside this step. Testing the interval rather than the
            // endpoint is what stops a fast-moving ray from stepping clean over
            // a surface at a grazing angle, and it is also, for free, the
            // rejection for rays that merely pass behind thin geometry: those
            // arrive at the step already deep behind whatever the buffer holds,
            // and `wPrev` says so.
            If(
              wS.greaterThanEqual(sceneW).and(wPrev.lessThanEqual(sceneW.add(acceptWindow))),
              () => {
                found.assign(1);
                sLo.assign(sPrev);
                sHi.assign(s);
                sceneAtHi.assign(sceneW);
                Break();
              },
            );

            sPrev.assign(s);
            wPrev.assign(wS);
            s.addAssign(stepS);
          });

          // --- bisect the bracketed crossing --------------------------------
          //
          // The coarse march only knows the hit is somewhere in the last step,
          // which at 24 steps across a full-width ray is tens of pixels — enough
          // that a reflection detaches visibly from the object casting it. A few
          // halvings put the contact back where it belongs.
          //
          // Refinement runs *after* the march, not nested inside it. Three's own
          // SSR node found that a loop inside a loop trips shader-compiler bugs
          // on some drivers, and there is no reason to take that risk for a
          // structure that reads better flattened anyway.
          //
          // The bisection tests only "is the ray behind the surface" and drops
          // the thickness clause. The bracket has already been accepted as a
          // real hit; what is being located now is the crossing point, and
          // re-applying an acceptance window here would let a sample inside the
          // bracket fail and push the search the wrong way.
          If(found.greaterThan(0.5), () => {
            Loop(this.uRefineSteps, () => {
              const sMid = sLo.add(sHi).mul(0.5).toVar('ssrSMid');
              const wMid = float(1).div(mix(kA, kB, sMid).max(1e-8)).toVar('ssrWMid');
              const sceneMid = this.sceneDistance(mix(uvA, uvB, sMid)).toVar('ssrSceneMid');

              If(wMid.greaterThanEqual(sceneMid), () => {
                sHi.assign(sMid);
                sceneAtHi.assign(sceneMid);
              }).Else(() => {
                sLo.assign(sMid);
              });
            });

            const hitUv = mix(uvA, uvB, sHi).clamp(0, 1).toVar('ssrHitUv');
            const hitW = float(1).div(mix(kA, kB, sHi).max(1e-8)).toVar('ssrHitW');

            // --- screen-edge fade -------------------------------------------
            //
            // Evaluated on the *hit* uv, not on the shaded pixel: what is about
            // to leave the frame is the thing being reflected, not the water
            // reflecting it. Every engine that documents this does it on the hit
            // — a fade keyed to the shading pixel would dim reflections in a
            // perfectly healthy patch of water merely because it sits near the
            // edge of the screen.
            //
            // The band is measured in fractions of screen *height* on both
            // axes, so it is the same number of pixels top, bottom and sides.
            // Two per-axis ramps multiplied, rather than the `min` idiom the
            // surface uses for the wake buffer, because the product falls off
            // faster into the corners — which is right, since a corner is
            // leaving the frame in two directions at once.
            const fadeSpan = vec2(
              this.uEdgeFade.mul(screenSize.y.div(screenSize.x.max(1))),
              this.uEdgeFade,
            ).toVar('ssrFadeSpan');
            const nearEdge = hitUv.smoothstep(vec2(0), fadeSpan).toVar('ssrNearEdge');
            const farEdge = hitUv
              .smoothstep(vec2(1).sub(fadeSpan), vec2(1))
              .oneMinus()
              .toVar('ssrFarEdge');
            const edgeFade = nearEdge.x
              .mul(nearEdge.y)
              .mul(farEdge.x)
              .mul(farEdge.y)
              .clamp(0, 1)
              .toVar('ssrEdgeFade');

            // --- thickness confidence ---------------------------------------
            //
            // How far behind the recorded surface the refined hit still sits.
            // The march's acceptance test is a predicate — in or out — and a
            // predicate at the end of a graded pipeline is where popping comes
            // back in: a hit drifting across the window boundary as the wave
            // moves switches on and off between frames. Grading the residual
            // instead turns that switch into a ramp.
            //
            // Squared, following AMD's FidelityFX SSSR, which uses the same
            // construction. The square sharpens the middle of the ramp so a hit
            // that genuinely landed on a surface keeps close to full weight and
            // only the marginal ones — the tunnelled ones, where bisection never
            // converged — are pulled down hard.
            const hitWindow = this.uThickness
              .add(sceneAtHi.mul(this.uThicknessSlope))
              .toVar('ssrHitWindow');
            const contact = float(1)
              .sub(hitW.sub(sceneAtHi).max(0).smoothstep(0, hitWindow))
              .clamp(0, 1)
              .toVar('ssrContact');
            const thicknessFade = contact.mul(contact).toVar('ssrThicknessFade');

            // --- range fade -------------------------------------------------
            //
            // How much of the permitted ray length the hit consumed. Both the
            // numerator and the denominator are depth spans, and depth is
            // affine in the world-space ray parameter, so their ratio *is* that
            // parameter — no camera axis needed. Taken as magnitudes so a ray
            // that approaches the camera measures the same way one that recedes
            // does.
            //
            // The fade exists because the range cut is otherwise a hard surface
            // in the middle of the water: geometry a metre inside it reflects,
            // geometry a metre outside does not, and the boundary between them
            // is a visible line that moves with the camera.
            const travelled = hitW
              .sub(wA)
              .abs()
              .div(wB.sub(wA).abs().max(1e-3))
              .clamp(0, 1)
              .toVar('ssrTravelled');
            const rangeFade = smoothstepDownClamped(travelled, 0.72, 1).toVar('ssrRangeFade');

            // --- hits under the water -------------------------------------
            //
            // [island-reflection] The water is not in the depth buffer, so a
            // ray the slope ramp above lets dip slightly finds whatever lies
            // under the surface: in the shallows that is the seabed a metre
            // or two down, and the beach-flyover drew it as brown and grey
            // streaks laid on the swell. Nothing below the water can be seen
            // in a reflection off its top. The hit's height comes from the
            // ray parameter, which `w` gives exactly because `w` is affine in
            // it; where the ray runs across the view axis and `w` barely
            // changes, the height is unknown and the hit is kept. Measured
            // against the lower of the launch point and mean sea level, less
            // a margin, so a hull's waterline seen from a crest survives.
            const dwPerMetre = wB.sub(wA).div(this.uRange).toVar('ssrDwPerMetre');
            const known = dwPerMetre.abs().smoothstep(0.01, 0.03).toVar('ssrKnown');
            const safeDw = dwPerMetre
              .greaterThanEqual(0)
              .select(dwPerMetre.max(0.01), dwPerMetre.min(-0.01));
            const along = hitW.sub(wA).div(safeDw);
            const hitY = p0.y.add(dir.y.mul(along.max(0))).toVar('ssrHitY');
            const floorY = p0.y.min(0).sub(0.3).toVar('ssrFloorY');
            const aboveWater = mix(float(1), hitY.smoothstep(floorY.sub(0.6), floorY), known).toVar('ssrAbove');

            const weight = gate
              .mul(edgeFade)
              .mul(rangeFade)
              .mul(thicknessFade)
              .mul(aboveWater)
              .mul(this.uStrength)
              .clamp(0, 1)
              .toVar('ssrWeight');

            // Snap a very-nearly-opaque result to fully opaque. Five smooth
            // terms multiplied together land just short of 1 over large areas
            // purely from rounding, and the residual few thousandths of the
            // fallback that leaves in the blend is a whole sky's worth of
            // brightness — Godot carries the same guard in its SSR resolve, and
            // names specular leakage as the reason.
            confidence.assign(weight.greaterThan(0.999).select(float(1), weight));

            // The one colour fetch, and it is inside the hit branch on purpose.
            // Most fragments of an open ocean miss — the reflected ray leaves
            // the frame into sky — so paying for the backdrop tap only on hits
            // is a real saving, on top of the two the surface already spends on
            // refraction. Safe in divergent flow because the framebuffer copy is
            // an unfilterable nearest-sampled texture, which compiles to a
            // texel load rather than a derivative-dependent sample.
            hitColor.assign(this.sceneColor.sample(hitUv).rgb);
          });
        });
      });

      return vec4(hitColor, confidence);
    })();
  }

  // ---------------------------------------------------------------- internals

  /**
   * Distance from the camera to the nearest opaque surface at a screen uv, in
   * metres along the camera's forward axis.
   *
   * `perspectiveDepthToViewZ` is used rather than the `linearDepth` /
   * `uDepthRange` pair the surface uses for refraction, for two reasons. It
   * resolves the reversed-depth-buffer case itself, which a hand-rolled
   * normalisation would have to duplicate. And it returns metres directly, in
   * the same units as `clip.w` — so the march compares two quantities that were
   * derived independently and still agree, rather than two that agree only if a
   * separately-pushed range uniform happens to match the camera.
   */
  private sceneDistance(uv: any): any {
    return perspectiveDepthToViewZ(this.sceneDepth.sample(uv).r, this.uNear, this.uFar).negate();
  }
}

/**
 * Clip-space position to the uv the viewport buffers are indexed by.
 *
 * The y flip is not cosmetic. `screenUV` is built from the fragment coordinate,
 * which runs top-down on WebGPU — while NDC y runs bottom-up. Getting this
 * backwards produces a march that walks *down* the screen when the ray goes up,
 * which does not look like a bug so much as a reflection of the wrong thing.
 */
const clipToScreenUV = /*@__PURE__*/ Fn(([clip]: [any]) => {
  const ndc = clip.xy.div(clip.w).toVar();
  return vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(-0.5).add(0.5));
});
