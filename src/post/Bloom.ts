import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { vec4 } from 'three/tsl';
import { uniform } from '../core/uniform';

/**
 * Veiling glare, from three's own bloom node.
 *
 * **Bought, not built — the only stage in `src/post/` that is, and it is worth
 * saying why.** `BloomNode` is a five-level mip pyramid with separable Gaussian
 * blurs, a smoothed luminance high-pass, and per-level tint weights; it resizes
 * itself from the drawing buffer on every render and disposes its own targets.
 * Reimplementing that would be several hundred lines arriving at the same image.
 * The other three stages here are hand-written because each needs something the
 * stock node cannot express — a thin-lens circle of confusion, an anchor on the
 * actual sun, a per-preset CDL. This one needs nothing the stock node does not
 * already do, and "we wrote it ourselves" is not a feature.
 *
 * **The threshold tracks the preset's exposure, and it has to.** This chain runs
 * in linear HDR — `RenderPipeline.outputColorTransform` applies ACES and the
 * sRGB conversion *after* `outputNode` — so it is tempting to say a threshold of
 * 1.0 selects light that exceeded the sensor. It does not, because
 * `toneMappingExposure` is applied at that same later stage: what will read as
 * white in the finished frame is not 1.0 in this chain but `1 / exposure`.
 *
 * The presets run from 0.42 to 1.35, so a fixed threshold of 1.0 means very
 * different things across them — and at Clear Day's 0.42 it sits at 42% of white
 * and admits most of the sky. That was not a subtle error: the first cut of this
 * blew the sunset shot to a white sheet and put a hard glowing blob on open
 * water in the clear-day shot. Deriving it from the exposure instead is what
 * makes "bloom the highlights" mean the same thing in all nine places.
 *
 * **Half resolution, deliberately.** The output is a wide Gaussian; nothing in
 * it is resolvable at full resolution, so the pyramid runs at half and costs a
 * quarter. `updateBefore` re-derives its size from the drawing buffer every
 * frame and multiplies by the scale, so this survives a resize with no hook.
 */
export class SceneBloom {
  /**
   * Strength when enabled. Held separately from the node's uniform so that
   * `setEnabled(false)` can zero the uniform without losing the value to
   * restore, and so a tier change and a look change cannot overwrite each other.
   */
  private strength = 0.03;
  private node: ReturnType<typeof bloom> | null = null;
  /**
   * Linear value that will read as white after exposure and tone mapping.
   *
   * `1 / preset.toneMappingExposure`, pushed in by `applyPreset`. Defaults to
   * the value Clear Day implies so a frame drawn before the first preset lands
   * is not blooming its entire sky.
   */
  private whitePoint = 1 / 0.42;
  /**
   * Where the high-pass sits relative to white.
   *
   * Two stops *above* it: a highlight has to be far past clipping, not merely
   * bright, before it spills. Below 1.0 this stops being veiling glare and
   * becomes a soft-focus filter over the whole image.
   *
   * It was 2.2, and a sea of sun glitter is mostly between two and four times
   * white: on Clear Day's canonical shot (A/B critic, reflections-3, "a bloom
   * veil washes the ship milky grey") the pyramid lifted the backlit hull by 9
   * output levels on average and the whole frame by 1.6. At 4 those are 5 and
   * 0.6, and the sun's own glow at sunset keeps its core — halving `strength`
   * instead took the same off the ship but dimmed the sun's glow everywhere.
   * Measured on corrected captures; the radius barely mattered (0.35 → 0, 9 →
   * 7 levels on the hull).
   */
  private static readonly THRESHOLD_OVER_WHITE = 4;
  /**
   * Ceiling on what the pyramid may see, in multiples of white.
   *
   * Five stops over white. High enough that the sun still spills much harder
   * than a whitecap does, low enough that it spills a finite amount.
   */
  private static readonly CAP_OVER_WHITE = 32;
  /** The cap, as a uniform so a preset change can move it without a rebuild. */
  private readonly uCap = uniform(1 / 0.42 * 32);

  /**
   * @param sceneColor The image to bloom. Should be a texture node: `BloomNode`
   *   evaluates its input inside its own high-pass material, and the caller
   *   evaluates it again for the additive base, so a composited expression here
   *   is computed twice. That is why the chain resolves the depth-of-field
   *   gather to a texture before this point rather than after it.
   * @returns `sceneColor + bloom(sceneColor)`. The node returns only the
   *   contribution — the sum is the caller's job, which is also what makes a
   *   selective bloom possible later by handing it a different input.
   */
  build(sceneColor: unknown): unknown {
    const src = sceneColor as any; // eslint-disable-line @typescript-eslint/no-explicit-any
    if (src === null || src === undefined) {
      throw new Error('SceneBloom.build: sceneColor is required.');
    }
    // Radius 0.35, low rather than high, and this is the difference between a
    // halo and a haze. `radius` weights the pyramid's coarse mips against its
    // fine ones, and the coarsest is a blur of the entire frame — so a wide
    // radius over a sea full of sun glitter lays a milky veil across everything
    // instead of putting a glow around the highlights. Measured on the clear-day
    // shot: at 0.6 the open water went visibly hazy and a single specular hit
    // became a glowing orb.
    // The pyramid sees a *capped* image; the additive base does not.
    //
    // This is the difference between a bloom and a white sheet, and the reason
    // is that the sun disc is not a calibrated radiance. `Atmosphere` notes it
    // runs about 19000x the sky's, and its own environment capture hides the
    // disc for exactly that reason. Feeding that number into an additive bloom
    // gives an additive result 19000 units tall: the first cut of this erased
    // the sunset shot's rigging and sails into a single blown highlight.
    //
    // A real sensor does not do that either — blooming saturates, because the
    // charge that spills from a photosite is bounded by the well that held it.
    // Capping the pyramid's input at a few stops over white is that bound. The
    // base image keeps its full range, so the sun disc itself is still as bright
    // as it was; only its *spill* is finite.
    this.node = bloom(src.min(vec4(this.uCap)), this.strength, 0.35, this.threshold());
    this.node.setResolutionScale(0.5);
    return src.add(this.node);
  }

  private threshold(): number {
    return this.whitePoint * SceneBloom.THRESHOLD_OVER_WHITE;
  }

  /**
   * Tells the pass where white is, from the preset's tone-mapping exposure.
   *
   * Called by `applyPreset`. Without it every preset blooms at a different
   * fraction of its own white point — see the note at the top of this file for
   * what that looked like.
   */
  setExposure(toneMappingExposure: number): void {
    this.whitePoint = 1 / Math.max(1e-3, toneMappingExposure);
    this.uCap.value = this.whitePoint * SceneBloom.CAP_OVER_WHITE;
    if (this.node !== null) this.node.threshold.value = this.threshold();
  }

  /**
   * Turns the contribution off without removing the node.
   *
   * Strength goes to zero and the pyramid drops to an eighth resolution, which
   * between them make it free in the only sense that matters here: the frame is
   * unchanged and the passes cost a rounding error. It cannot simply be removed
   * from the graph, because rebuilding `outputNode` recompiles shaders, and this
   * is reached from a live tier change — which is exactly the in-gameplay shader
   * compile the whole quality system exists to avoid.
   */
  setEnabled(on: boolean): void {
    if (this.node === null) return;
    this.node.strength.value = on ? this.strength : 0;
    this.node.setResolutionScale(on ? 0.5 : 0.125);
  }

  dispose(): void {
    this.node?.dispose();
    this.node = null;
  }
}
