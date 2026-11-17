import { expect, test } from '@playwright/test';
import { bootOcean } from './lib/capture';
import { setState } from './helpers';

/**
 * Rendered whitecap coverage, against the law that is supposed to generate it.
 *
 * The deposit rate follows Monahan & O'Muircheartaigh's fit to ship
 * observations, `W = 3.84e-6 * U^3.41` — and until this existed, nothing checked
 * what came out. The law was decorating a system whose actual coverage was an
 * order of magnitude higher than it predicts, which is why the reference image
 * rendered a 15 m/s sea as a sheet of white.
 *
 * Measured from the accumulation buffer rather than from the frame. A rendered
 * pixel is white for several reasons — sun glitter, sky reflection, a blown
 * highlight on a crest — and none of them are foam; the buffer's red channel is
 * the foam field itself, so counting it answers the question that was asked.
 */

/** Monahan & O'Muircheartaigh: fractional whitecap coverage at wind speed U. */
const monahan = (u: number) => 3.84e-6 * Math.pow(Math.max(0, u), 3.41);

async function coverage(page: import('@playwright/test').Page, windSpeed: number) {
  await setState(page, { preset: 'skyPro', quality: 'high', windSpeed });
  await page.evaluate(() => window.__ocean.resetDeterministic(30, 240));
  return page.evaluate(async () => {
    const ocean = window.__ocean as any;
    const target = ocean.wake.outputTarget;
    // Half the buffer, from the middle. A 64x64 corner of a world-anchored 1024
    // buffer is a 40 m patch of one particular piece of sea, and it read 0% at
    // 15 m/s and 19% at 9 m/s purely on where the crests happened to be. 512
    // rows of 8 bytes is 4096 per row, still a multiple of the 256-byte
    // alignment the readback needs.
    const size = target.width;
    const half = size >> 1;
    const raw = await ocean.renderer.readRenderTargetPixelsAsync(
      target, size >> 2, size >> 2, half, half,
    );
    const isHalfFloat = (raw as any).BYTES_PER_ELEMENT === 2;
    const toFloat = (bits: number) => {
      const sign = bits & 0x8000 ? -1 : 1;
      const exponent = (bits & 0x7c00) >> 10;
      const mantissa = bits & 0x03ff;
      if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
      if (exponent === 31) return mantissa ? NaN : sign * Infinity;
      return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
    };
    let covered = 0;
    let n = 0;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nonFinite = 0;
    for (let i = 0; i < raw.length; i += 4) {
      const foam = isHalfFloat ? toFloat(raw[i]) : raw[i];
      if (!Number.isFinite(foam)) { nonFinite++; continue; }
      if (foam < min) min = foam;
      if (foam > max) max = foam;
      sum += foam;
      // Not a count above a threshold. The foam field is close to binary — a
      // texel either reaches its equilibrium or sits near zero — so any fixed
      // cut-off turns coverage into a cliff, and the measurement swung between
      // 0% and 8% for a change that moved the mean by a fifth.
      //
      // The surface remaps the buffer through `smoothstep(0.06, 0.68)` and then
      // mixes toward white by the result, so that remap *is* the fraction of a
      // texel's albedo that is foam. Averaging it gives fractional coverage in
      // the same radiometric sense Monahan's W is defined in, and it is
      // continuous in the thing being tuned.
      const t = Math.min(1, Math.max(0, (foam - 0.06) / (0.68 - 0.06)));
      covered += t * t * (3 - 2 * t);
      n++;
    }
    return {
      fraction: covered / n, min, max, mean: sum / n, n, nonFinite,
      kind: (raw as any).constructor.name, len: raw.length,
    };
  });
}

test('whitecap coverage follows Monahan', async ({ page }) => {
  test.setTimeout(600_000);
  await bootOcean(page);

  // 6 and 24 are in the range where `setBreaking`'s clamp binds — the deposit
  // rate stops following the law below about 11.9 m/s and above about 18.9 —
  // and they are here precisely because that is where the claim is most at risk.
  const speeds = [6, 9, 15, 21, 24];
  const measured: number[] = [];
  for (const u of speeds) {
    const r = await coverage(page, u);
    measured.push(r.fraction);
    console.log(
      `[foam] U=${u} rendered ${(r.fraction * 100).toFixed(2)}% Monahan ${(monahan(u) * 100).toFixed(2)}% ` +
        `| min ${r.min.toFixed(3)} max ${r.max.toFixed(3)} mean ${r.mean.toFixed(4)} n ${r.n} ` +
        `nf ${r.nonFinite} ${r.kind} len ${r.len}`,
    );
  }

  // A documented band around the art-directed foam, not a physical match.
  //
  // The renderer was re-tuned by eye on 2026-11-16 (a 15 m/s sea read as bare
  // water when it matched Monahan), so the foam is deliberately denser than the
  // physical law in strong wind. Measured then, rendered against Monahan:
  //
  //   U (m/s)   rendered   Monahan   ratio
  //      6        0.15%      0.17%    0.9
  //      9        1.82%      0.69%    2.6
  //     15       13.36%      3.94%    3.4
  //     21       33.89%     12.4 %    2.7
  //     24       46.10%     19.5 %    2.4
  //
  // The band is 0.5x to 4.5x the law. The top still fails the ten-times-over
  // sheet of white this test was written against, and the bottom still fails a
  // sea that has lost its foam. Re-tuning the look moves the figures above; it
  // should not move them out of this band without the band being re-argued.
  for (let i = 0; i < speeds.length; i++) {
    const predicted = monahan(speeds[i]);
    expect(
      measured[i],
      `at ${speeds[i]} m/s the surface renders ${(measured[i] * 100).toFixed(2)}% foam ` +
        `against ${(predicted * 100).toFixed(2)}% predicted`,
    ).toBeLessThan(predicted * 4.5);
    expect(measured[i]).toBeGreaterThan(predicted * 0.5);
  }

  // And the *shape* of the law, which is the part a single scale factor cannot
  // fake: U^3.41 is very steep, so tripling the wind must raise coverage by far
  // more than three times.
  // Monotonic across the whole range, including where the rate is clamped. A
  // clamp that flattened the response would show up here even if every
  // individual point still sat inside its factor-of-three band.
  for (let i = 1; i < speeds.length; i++) {
    expect(
      measured[i],
      `coverage did not rise from ${speeds[i - 1]} to ${speeds[i]} m/s: ` +
        `${(measured[i - 1] * 100).toFixed(2)}% then ${(measured[i] * 100).toFixed(2)}%`,
    ).toBeGreaterThan(measured[i - 1]);
  }

  // And the *shape* of the law, which is the part a single scale factor cannot
  // fake: U^3.41 is very steep, so more than doubling the wind must raise
  // coverage by far more than twice.
  const first = measured[0];
  const last = measured[measured.length - 1];
  expect(
    last / Math.max(first, 1e-6),
    `coverage went from ${(first * 100).toFixed(2)}% at ${speeds[0]} m/s to ` +
      `${(last * 100).toFixed(2)}% at ${speeds[speeds.length - 1]} m/s, ` +
      'which is far flatter than U^3.41',
  ).toBeGreaterThan(monahan(speeds[speeds.length - 1]) / monahan(speeds[0]) / 3);
});
