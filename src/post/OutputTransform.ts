import * as THREE from 'three/webgpu';
import { Fn, dot, floor, fract, renderOutput, screenCoordinate, sin, uniform, vec2, vec4 } from 'three/tsl';

/**
 * Tone mapping, the sRGB transfer function, and a dither — done here rather
 * than by the renderer, for the one reason that cannot be worked around.
 *
 * **Why take this over at all.** `RenderPipeline` applies ACES and the sRGB
 * conversion itself, after `outputNode`, whenever `outputColorTransform` is
 * true. That is convenient and it leaves no place to stand: the last thing that
 * happens to a frame is quantisation to 8 bits, and if every node in the chain
 * runs before the tone curve then no node runs late enough to dither. So this
 * sets `outputColorTransform = false` and does the same three steps the
 * renderer would have, plus a fourth.
 *
 * The first three are not reimplemented — `renderOutput` is the very node
 * `RenderPipeline` uses, handed the same tone mapping and colour space. With the
 * dither amplitude at zero this stage is therefore the renderer's own output
 * path, which is exactly the property that makes it safe to adopt: it can be
 * verified by capturing a frame with dithering off and comparing it against one
 * captured before this file existed.
 *
 * **Why dither is worth an architectural change.** Quantising a smooth gradient
 * to 256 levels produces banding — visible steps where the underlying signal
 * changes by less than one level per pixel. This scene is close to a worst case
 * for it: sky and open water are large, smooth, low-contrast gradients, and they
 * occupy most of every frame in the gallery. A dither of the right shape trades
 * that structured artefact for unstructured noise below the quantisation step,
 * which the eye integrates away entirely.
 *
 * **The dither is triangular, not uniform, and that matters.** Adding a single
 * uniform random value decorrelates the quantisation error from the signal only
 * partly: the error's *variance* still depends on where the signal sits between
 * two levels, which is visible as noise that pulses across a gradient. The
 * difference of two independent uniform values — a triangular probability
 * density, ±1 LSB wide — makes both the mean and the variance of the error
 * independent of the signal. That is the standard result from audio dithering
 * and it transfers directly; it costs one extra hash.
 *
 * **The noise is static, and it has to be.** It is a pure function of pixel
 * coordinates with no time term. Animating it would look better in motion, and
 * would make every captured frame a function of how many frames preceded it —
 * which is the one property this project's entire visual harness rests on. A
 * fixed pattern is the price of a reproducible screenshot.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Dither amplitude in output levels.
 *
 * 1.0 is the textbook value: a triangular density spanning exactly ±1 LSB, which
 * is what makes the quantisation error's first two moments independent of the
 * signal. Lower values leave some banding; higher ones are just visible noise.
 */
const DITHER_LEVELS = 1.0;

/** Quantisation step of the output buffer. 8-bit, so 255 steps. */
const OUTPUT_LEVELS = 255;

/**
 * The project's usual sine hash.
 *
 * Safe on the argument `LensRain` gives for it: the input here is a pixel
 * coordinate, so the magnitudes are a few thousand at most and none of the
 * precision loss that afflicts world-space hashing applies.
 */
function hash21(p: any): any {
  return fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));
}

export class OutputTransform {
  private readonly uDither = uniform(DITHER_LEVELS / OUTPUT_LEVELS);

  /**
   * @param sceneColor The finished linear-HDR frame, after every other stage.
   * @param toneMapping Must match what the renderer would have applied.
   * @param colorSpace  Must match the renderer's `outputColorSpace`.
   */
  build(sceneColor: unknown, toneMapping: THREE.ToneMapping, colorSpace: string): unknown {
    return this.applyDither(this.buildDisplay(sceneColor, toneMapping, colorSpace));
  }

  /**
   * Tone mapping and the transfer function, with nothing after them.
   *
   * Split out from `build` so a stage that *requires display-referred input* can
   * stand between the two halves. FXAA is the one that does: it thresholds on
   * luma and its constants are calibrated against sRGB, so three's own node
   * documents that "tone mapping and color space conversion must happen before
   * the anti-aliasing". Running it on the linear HDR frame instead would have it
   * find almost no edges in the shadows and treat every specular highlight as
   * one.
   *
   * The dither still goes last, which is the whole reason this class exists —
   * see the header. Anti-aliasing before quantisation is the correct order
   * anyway: dithering first and then blurring the dither is how you get back the
   * banding it was there to remove.
   */
  buildDisplay(sceneColor: unknown, toneMapping: THREE.ToneMapping, colorSpace: string): unknown {
    const src: any = sceneColor;
    if (src === null || src === undefined) {
      throw new Error('OutputTransform.buildDisplay: sceneColor is required.');
    }

    // The renderer's own output node, with the renderer's own settings. This
    // includes `toneMappingExposure` — the tone-mapping node reads it as a
    // uniform — so the per-preset exposure keeps working untouched.
    return Fn(() => {
      const c = vec4(src);
      const trimmed = OUTPUT_TONE_TRIM === 1 ? c : vec4(c.rgb.mul(OUTPUT_TONE_TRIM), c.a);
      return vec4(renderOutput(trimmed, toneMapping, colorSpace));
    })();
  }

  /** The dither, which must be the last thing that touches the frame. */
  applyDither(displayColor: unknown): unknown {
    const src: any = displayColor;
    if (src === null || src === undefined) {
      throw new Error('OutputTransform.applyDither: displayColor is required.');
    }

    return Fn(() => {
      const display = vec4(src).toVar('outDisplay');

      // Triangular dither, applied *after* the transfer function.
      //
      // After, because the quantisation it is correcting for happens in display
      // space: the levels are evenly spaced there and wildly unevenly spaced in
      // linear. A dither applied before the sRGB curve would be stretched by it
      // — far too coarse in the shadows, far too fine in the highlights, which
      // is precisely backwards, since banding is worst in the darker part of a
      // gradient where the curve is steepest.
      const p = screenCoordinate.xy.toVar('outPixel');
      const r1 = hash21(floor(p)).toVar('outR1');
      const r2 = hash21(floor(p).add(vec2(17.0, 43.0))).toVar('outR2');
      const tpdf = r1.sub(r2).toVar('outTpdf');

      return vec4(display.rgb.add(tpdf.mul(this.uDither)), display.a);
    })();
  }

  /**
   * Dither amplitude, in output levels. 0 disables it exactly.
   *
   * Exposed so a test can prove this stage reproduces the renderer's own output
   * path when the dither is off — which is the evidence that taking the path
   * over was safe.
   */
  setDitherLevels(levels: number): void {
    this.uDither.value = levels / OUTPUT_LEVELS;
  }

  /** Owns no GPU resources; present so the chain's teardown stays uniform. */
  dispose(): void {}
}

/**
 * The tone curve, with `?tonemap=aces|agx|neutral` for A/B.
 *
 * The query exists for the comparison in `docs/AAA_PLAN.md` section N and is
 * read once at boot: the operator is compiled into the output node, so
 * switching it live would be a recompile.
 */
function chooseToneMapping(): THREE.ToneMapping {
  const query =
    typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('tonemap') : null;
  switch (query) {
    case 'agx':
      return THREE.AgXToneMapping;
    case 'neutral':
      return THREE.NeutralToneMapping;
    case 'aces':
    default:
      return THREE.ACESFilmicToneMapping;
  }
}

/** Re-exported so `main` does not have to import three twice to name these. */
export const OUTPUT_TONE_MAPPING: THREE.ToneMapping = chooseToneMapping();
/**
 * Linear gain ahead of the curve that puts scene middle grey (0.18) at the same
 * display value under every operator, so a comparison compares *curves* rather
 * than exposures. Three's ACES divides by 0.6 internally; AgX lands within
 * 0.01 EV of it; Khronos Neutral needs +0.49 EV. Computed by
 * evaluating three's own curve formulas on a grey ramp.
 */
export const OUTPUT_TONE_TRIM: number =
  OUTPUT_TONE_MAPPING === THREE.NeutralToneMapping ? 1.406 : OUTPUT_TONE_MAPPING === THREE.AgXToneMapping ? 1.008 : 1;
export const OUTPUT_COLOR_SPACE = THREE.SRGBColorSpace;
