import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { mix, uniform, vec4 } from 'three/tsl';

/**
 * A spatial anti-alias resolve, sitting between the transfer function and the
 * dither.
 *
 * **On most tiers the scene pass is already multisampled, and that is not the
 * problem this solves.** `PassNode.setup` copies `renderer.samples` onto its
 * render target, and the renderer is constructed with `antialias` from the
 * quality tier — true everywhere except Low, which cannot afford it — so on
 * those tiers geometry edges (rigging, masts, the island silhouette, the buoy)
 * are resolved at 4x. What MSAA cannot touch is *shading* aliasing, because it
 * shades once per pixel and only supersamples coverage.
 *
 * Low is the tier where this pass matters most, then: it is the one carrying
 * geometry aliasing as well as shading aliasing, and it is also the one least
 * able to pay for a wide resolve. That is a tension this pass does not
 * currently resolve, and it is recorded rather than hidden. Every remaining stair-step in this scene is of
 * that kind: the specular on a sub-pixel wave, the foam mask's threshold, the
 * rain streaks. `OceanMaterial` already band-limits what it can in slope space
 * and its long comment on the foam mask records why widening the threshold made
 * things worse rather than better; that comment also names the honest fix as a
 * temporal resolve, which this renderer does not have and cannot have while
 * every captured frame must be a function of the clock alone.
 *
 * A spatial resolve is what is left. It works on the finished image, it has no
 * history, and it therefore stays exactly as reproducible as the frame it is
 * given — which is the property the whole visual harness rests on.
 *
 * **Why FXAA and not something better.** three ships exactly two candidates for
 * r185: `FXAANode` and `FSR1Node`. FSR1 is an upscaler whose edge-adaptive pass
 * is designed to run at a different input and output resolution, and using it at
 * 1:1 for its analytic-sharpening side effect is not what it is for. SMAA is not
 * in the addon set. FXAA is the one node in the box that does this job, it is
 * one dependency the project already has, and it is what Water Pro's own
 * post-processing guide names in its recommended order.
 *
 * **Placement is not a matter of taste.** three's node documents that it
 * "requires sRGB input so tone mapping and color space conversion must happen
 * before the anti-aliasing" — its luma threshold and its edge constants are
 * calibrated in display space. That is why `OutputTransform` was split: this
 * runs after `buildDisplay` and before `applyDither`. Both neighbours are load
 * bearing. Before the transfer function it would find no edges in the shadows
 * and treat every highlight as one; after the dither it would blur the dither
 * back into the banding it exists to remove.
 *
 * **The strength is a uniform, not a rebuild.** FXAA is always in the graph and
 * the blend against the unfiltered image is what turns it down, so a quality
 * tier change never recompiles a shader — which is a hard rule here, see the
 * comment on `loadSceneContent` in `main`. The cost of that choice is that the
 * pass runs even at zero strength, and that cost is not nothing: `FXAANode`
 * takes a nine-tap neighbourhood, then up to two six-step directional searches,
 * then a final tap, on top of the fullscreen resolve `convertToTexture` inserts.
 * It is still the cheapest stage in this chain — the depth-of-field gather alone
 * is up to 32 taps — but it is a filter, not a free one.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Default blend against the unfiltered image.
 *
 * 0.3, and the number is a measurement rather than a preference. It started at
 * 0.75 on the reasoning that FXAA's softening is survivable if a quarter of the
 * sharp image is kept; the `boat-chase` capture said otherwise. That shot puts
 * three masts of standing rigging against a bright sky, and at 0.75 the shrouds
 * and ratlines — which are one pixel wide there — were visibly eaten. That is a
 * bad trade in a scene whose geometry is *already* multisampled: MSAA is doing
 * the edges properly, and all this stage is entitled to is the shading aliasing
 * MSAA cannot reach.
 *
 * At 0.3 the specular and foam stair-steps lose their hardest edge and the
 * rigging survives. It is a smaller effect than the machinery suggests, and it
 * should be: the large win was never available, because the honest fix for
 * shading aliasing is a temporal resolve this renderer does not have.
 *
 * 0.42 was then tried from the other end, to pull the far-field shimmer figure
 * under its ceiling — this filter being a low-pass, that is the one lever in the
 * chain that points the right way. It does pull it under, and `gallery-jitter`
 * immediately fails the *other* half of its contract instead: near-field
 * structure drops to 0.375 against a 0.4 floor. That assertion exists precisely
 * to stop the shimmer number being won by blurring the water, and it worked.
 */
const DEFAULT_STRENGTH = 0.3;

export class SpatialAA {
  private readonly uStrength = uniform(DEFAULT_STRENGTH);

  /**
   * @param displayColor The tone-mapped, transfer-encoded frame — *not* linear
   *                     HDR, and not yet dithered.
   */
  build(displayColor: unknown): unknown {
    const src: any = displayColor;
    if (src === null || src === undefined) {
      throw new Error('SpatialAA.build: displayColor is required.');
    }

    // `fxaa` wraps its input with `convertToTexture`, so the resolve it needs is
    // its own doing rather than something this has to arrange.
    const filtered: any = fxaa(src);
    return vec4(mix(vec4(src), vec4(filtered), this.uStrength));
  }

  /** 0 disables the filter exactly; 1 is unblended FXAA. */
  setStrength(strength: number): void {
    this.uStrength.value = Math.min(1, Math.max(0, strength));
  }

  /** Owns no GPU resources; present so the chain's teardown stays uniform. */
  dispose(): void {}
}
