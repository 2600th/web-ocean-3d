import { expect, test } from '@playwright/test';
import { bootOcean, capture } from './lib/capture';
import { setCamera, setState } from './helpers';

/**
 * Shimmer, measured where genuine motion cannot reach.
 *
 * A plain frame-to-frame difference over the whole water cannot separate shimmer
 * from waves: the ripple cascade is a two-metre band, its phase speed is metres
 * per second, and it legitimately redraws the near field in 1/60 s. Every earlier
 * reading was mostly that.
 *
 * The far field is the discriminator. Beyond a couple of hundred metres a wave
 * moves a fraction of a pixel per frame, so anything that changes there is
 * sampling noise. This measures a horizontal band just under the horizon and
 * reports both the temporal change and the high-spatial-frequency energy, which
 * separates "the image is noisy" from "the noise is moving".
 */

/**
 * Two bands, and the second one is what stops this test rewarding blur.
 *
 * `FAR` sits just under the horizon, where a wave moves a fraction of a pixel
 * per frame and anything changing is sampling noise — so its high-frequency
 * energy is the shimmer figure. On its own that figure is minimised by
 * destroying detail, which would make "render a flat blue plane" the winning
 * strategy. `NEAR` is the bottom of the frame, metres away, where the wave
 * detail is genuinely resolvable and must survive: a change that smooths the
 * far field is only a fix if the near field still has structure in it.
 */
const FAR = { lo: 0.47, hi: 0.6 };
const NEAR = { lo: 0.78, hi: 0.96 };

interface BandStats {
  temporal: number;
  highFreq: number;
  /** Mean |luminance step| between horizontally adjacent pixels. */
  detail: number;
}

function analyse(
  a: Awaited<ReturnType<typeof capture>>,
  b: typeof a,
  band: { lo: number; hi: number },
): BandStats {
  const y0 = Math.floor(a.height * band.lo);
  const y1 = Math.floor(a.height * band.hi);
  let temporal = 0;
  let highFreq = 0;
  let detail = 0;
  let n = 0;
  const luma = (i: number, img: typeof a) =>
    0.2126 * img.data[i] + 0.7152 * img.data[i + 1] + 0.0722 * img.data[i + 2];
  for (let y = y0; y < y1; y++) {
    for (let x = 1; x < a.width - 1; x++) {
      const i = (y * a.width + x) * 4;
      const here = luma(i, a);
      temporal += Math.abs(here - luma(i, b));
      // Laplacian along the row: a smooth gradient gives ~0, per-pixel noise
      // gives its own amplitude.
      highFreq += Math.abs(2 * here - luma(i - 4, a) - luma(i + 4, a));
      detail += Math.abs(here - luma(i - 4, a));
      n++;
    }
  }
  return { temporal: temporal / n, highFreq: highFreq / n, detail: detail / n };
}

/**
 * Ceilings on far-field spatial noise, per tier.
 *
 * Set a few percent above the measured figures — room for driver and scheduling
 * variation, not room for a regression. Medium runs two cascades and the others
 * three, which is the whole reason the tiers differ here.
 *
 * **Recalibrated when `skyPro`'s exposure went from 1 to 0.42**, and the reason
 * matters more than the numbers, because "the gate went red so the gate moved"
 * is exactly what this file exists to prevent.
 *
 * The figure is a Laplacian in *absolute luminance levels*, so it is a function
 * of the grade as well as of the renderer. Measured on the identical frame with
 * nothing changed but the tone-mapping exposure: 0.805 at 1.0 and 0.891 at 0.42,
 * a rise of 10.7%. Under ACES the old exposure sat this band near the shoulder,
 * where per-pixel slope noise was being compressed into a fraction of a level;
 * the new one puts it on the steep part of the curve, where the same noise is
 * visible. Nothing about the water changed.
 *
 * **Re-measured a second time** when the sky was re-calibrated
 * (`SKY_RADIANCE_SCALE` 0.35 -> 0.16) and the haze colour corrected, for the same
 * reason: both change the luminance of the band this figure is computed over.
 * The relative numbers, which is what actually matters, say the far field is
 * *quieter* than the target — high-frequency energy normalised by band
 * luminance is 0.031 here against 0.080 measured on
 * `docs/ref/tropical-island-sea-level-v2.png`. There is no shimmer to chase in
 * the renderer; there is a metric denominated in absolute levels sitting under a
 * grade that moved twice.
 *
 * The honest conclusion is that these ceilings would be better expressed
 * *relative* to the band's own mean luminance, which would make them invariant
 * to exposure. That is left undone deliberately: rewriting the metric would also
 * rewrite what every historical number in this file means, and it is a change
 * worth making on its own rather than folded into an art pass.
 *
 * That is not a licence to move the line without looking, and it was not taken
 * as one. Before recalibrating, the obvious renderer fix was tried and
 * *rejected on measurement*: pulling the distance normal-flatten in from 2500 m
 * to 900 m — which makes every wave beyond that a mirror-flat sheet — moved the
 * medium figure from 1.964 to 1.910, 2.7%. The far-field sparkle in this band is
 * therefore not coming from the surface normal, and flattening the sea to buy
 * 2.7% would have been the exact failure the `DETAIL_FLOOR` assertion below
 * exists to catch. Where it *is* coming from is unresolved and recorded as such.
 */
const SHIMMER_CEILING: Record<string, number> = {
  medium: 2.33,
  high: 3.77,
  ultra: 3.84,
  max: 3.74,
};

/**
 * Ceilings on far-field frame-to-frame change.
 *
 * The spatial figure alone is not enough, and the first version of this test had
 * only that one: a band flashing uniformly from black to white every frame
 * carries no horizontal structure at all, so it scores zero on a Laplacian and
 * would have passed every ceiling while being the worst shimmer imaginable. Both
 * have to be bounded.
 */
const TEMPORAL_CEILING: Record<string, number> = {
  medium: 1.47,
  high: 2.5,
  ultra: 2.61,
  max: 2.29,
};

/**
 * Floor on near-field structure: the mean luminance step between horizontally
 * adjacent pixels.
 *
 * Not a Laplacian, and not a standard deviation. Per-pixel Laplacian energy
 * *rises* with distance — a distant pixel spans many wavelengths and a near one
 * spans a fraction of one — so the near field measured lower on it than the far
 * field, which makes it useless here. Standard deviation over the band is worse:
 * it cannot tell wave detail from a smooth vertical gradient, and this frame has
 * several of those in it (Fresnel, aerial perspective, the reflected sky), so a
 * flat band with an ordinary lighting ramp across it passes comfortably.
 *
 * The mean adjacent-pixel step has neither problem. A linear ramp of forty
 * levels across the frame contributes about 0.03 per pixel; water with waves in
 * it contributes a whole level or more.
 */
const DETAIL_FLOOR = 0.4;

/**
 * **What these two bands cannot tell you, recorded where it will be needed.**
 *
 * Detail is measured in the NEAR band and shimmer in the FAR one, so there is no
 * far-field detail figure — and that is exactly the quantity needed to separate
 * "the far field carries more real wave geometry" from "the far field is
 * noisier". The two look identical to `highFreq`.
 *
 * It came up for real. Squaring the tier mesh proportions raised far-field
 * `highFreq` by 5-7% on High, Ultra and Max while recovering ~2% of near-field
 * detail; the change was kept on the argument that a finer mesh band-limits
 * *less*, so the extra energy is geometry rather than sparkle — but this file
 * cannot confirm that. The figures and the argument are recorded on `meshRings`
 * in `src/core/QualityManager.ts` rather than settled here.
 *
 * The fix is a `detail` figure over the FAR band alongside the existing one. It
 * is deliberately not added here: every ceiling above is denominated in absolute
 * luminance levels and re-cutting the metric would rewrite what all of them
 * historically meant, which is a change worth making on its own.
 */

test('the far field does not shimmer, and the near field keeps its detail', async ({ page }) => {
  test.setTimeout(600_000);
  await bootOcean(page);

  // The output dither is switched off for the whole of this test, and that is a
  // correctness fix rather than a convenience.
  //
  // Both metrics below are per-pixel *spatial* operators — `highFreq` is a
  // Laplacian along the row, `detail` a first difference — and the output stage
  // adds a static, independent, one-LSB triangular dither to every pixel before
  // quantisation. That is per-pixel high-frequency energy by construction, so it
  // lands directly in both, and a Laplacian amplifies it: independent noise of
  // standard deviation s comes through at s*sqrt(6).
  //
  // It confounds this test in *both* directions, which is what makes leaving it
  // on untenable. It inflates `highFreq`, so a renderer whose distant water is
  // perfectly calm reads as boiling — measured here at 2.101 before the dither
  // existed and 2.604 after, against a 2.33 ceiling, with the *temporal* figure
  // unchanged and comfortably passing the whole time. And it inflates `detail`
  // by about the same amount, which is worse: `DETAIL_FLOOR` exists to stop the
  // shimmer figure being won by flattening the water, and a dither would let a
  // genuinely flattened frame clear it on noise alone.
  //
  // The ceilings are measured constants and stay where they were measured. What
  // changes is that the thing being measured is the water again.
  await page.evaluate(() => window.__ocean.setDitherLevels(0));

  for (const quality of ['medium', 'high', 'ultra', 'max'] as const) {
    await setState(page, { quality, preset: 'skyPro', windSpeed: 15 });
    await setCamera(page, [-46, 9, 44], [-90, 6, 8]);

    await page.evaluate(() => window.__ocean.resetDeterministic(23.5, 60));
    const a = await capture(page);
    await page.evaluate(() => window.__ocean.step(1 / 60, 1));
    const b = await capture(page);

    const far = analyse(a, b, FAR);
    const near = analyse(a, b, NEAR);
    console.log(
      `SHIMMER ${quality.padEnd(7)} far temporal ${far.temporal.toFixed(3)} highFreq ` +
        `${far.highFreq.toFixed(3)}  |  near detail ${near.detail.toFixed(3)}`,
    );

    expect(
      far.highFreq,
      `${quality}: far-field high-frequency energy is ${far.highFreq.toFixed(3)}, ` +
        `over the ${SHIMMER_CEILING[quality]} ceiling — the distant water is boiling again`,
    ).toBeLessThan(SHIMMER_CEILING[quality]);

    expect(
      far.temporal,
      `${quality}: the far field changed by ${far.temporal.toFixed(3)} levels in one frame, ` +
        `over the ${TEMPORAL_CEILING[quality]} ceiling — at this distance a wave moves a ` +
        'fraction of a pixel, so this is the image reshuffling rather than the sea moving',
    ).toBeLessThan(TEMPORAL_CEILING[quality]);

    expect(
      near.detail,
      `${quality}: near-field structure collapsed to ${near.detail.toFixed(3)}. ` +
        'The shimmer figure can always be won by flattening the water; this is the ' +
        'assertion that stops that counting as a fix',
    ).toBeGreaterThan(DETAIL_FLOOR);
  }
});
