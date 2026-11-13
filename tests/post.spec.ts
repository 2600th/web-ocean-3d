import { test, expect } from '@playwright/test';
import { frameHead, frameMean, setCamera, setState } from './helpers';
import { bootOcean } from './lib/capture';

/**
 * The post chain's four image stages: grade, bloom, depth of field, lens flare.
 *
 * These are behavioural rather than pictorial on purpose. What a graded frame
 * *looks* like is the visual suite's job, and it answers it by comparing whole
 * images against a baseline. What this file asks is narrower and cannot be
 * answered that way: is the stage actually in the chain, does its one defining
 * property hold, and does turning it off genuinely turn it off. A baseline
 * comparison passes just as happily against a stage that was never wired in —
 * it only knows the image changed, not which of five things changed it.
 *
 * Every test here pins a camera and calls `resetDeterministic` before measuring,
 * for the reason the shot list's own header gives: without it a measurement
 * depends on whatever the previous test left behind.
 */

test.describe('colour grade', () => {
  test('is in the chain, and a warm slope warms the frame', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'skyPro', quality: 'high', cameraMode: 'orbit' });
    await setCamera(page, [-42, 21, 63], [0, 3, 0]);
    await page.evaluate(() => window.__ocean.resetDeterministic(12, 90));

    const identity = await page.evaluate(async () => {
      window.__ocean.setGrade({
        slope: [1, 1, 1], offset: [0, 0, 0], power: [1, 1, 1], saturation: 1,
      });
      await window.__ocean.step(1 / 60, 2);
      const { data } = await window.__ocean.capturePixels();
      return Array.from(data.slice(0, 4096));
    });

    const warmed = await page.evaluate(async () => {
      window.__ocean.setGrade({
        slope: [1.25, 1, 0.8], offset: [0, 0, 0], power: [1, 1, 1], saturation: 1,
      });
      await window.__ocean.step(1 / 60, 2);
      const { data } = await window.__ocean.capturePixels();
      return Array.from(data.slice(0, 4096));
    });

    // Wired at all.
    expect(warmed).not.toEqual(identity);

    // And wired the right way round. A slope that lifts red and cuts blue must
    // raise the red-to-blue ratio; a stage that was in the chain backwards, or
    // reading the wrong uniform, would still fail the equality above.
    const channel = (a: number[], offset: number) =>
      a.reduce((sum, v, i) => (i % 4 === offset ? sum + v : sum), 0);
    const identityRatio = channel(identity, 0) / channel(identity, 2);
    const warmedRatio = channel(warmed, 0) / channel(warmed, 2);
    expect(warmedRatio).toBeGreaterThan(identityRatio);

    // Leave it as it was found: `setGrade` writes uniforms that outlive the test,
    // and the suite shares a page across tests in some projects.
    await page.evaluate(() =>
      window.__ocean.setGrade({
        slope: [1, 1, 1], offset: [0, 0, 0], power: [1, 1, 1], saturation: 1,
      }),
    );
  });

  test('desaturating to zero leaves a monochrome frame', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'skyPro', quality: 'high', cameraMode: 'orbit' });
    await setCamera(page, [-42, 21, 63], [0, 3, 0]);
    await page.evaluate(() => window.__ocean.resetDeterministic(12, 90));

    const spread = await page.evaluate(async () => {
      window.__ocean.setGrade({
        slope: [1, 1, 1], offset: [0, 0, 0], power: [1, 1, 1], saturation: 0,
      });
      await window.__ocean.step(1 / 60, 2);
      const { data } = await window.__ocean.capturePixels();
      // Largest channel spread anywhere in the frame. A correct desaturation
      // makes this zero up to the 8-bit quantisation of three equal linear
      // values through the tone curve, which is at most one level.
      let worst = 0;
      for (let i = 0; i < data.length; i += 4) {
        const max = Math.max(data[i], data[i + 1], data[i + 2]);
        const min = Math.min(data[i], data[i + 1], data[i + 2]);
        worst = Math.max(worst, max - min);
      }
      return worst;
    });

    expect(spread).toBeLessThanOrEqual(1);

    await page.evaluate(() =>
      window.__ocean.setGrade({
        slope: [1, 1, 1], offset: [0, 0, 0], power: [1, 1, 1], saturation: 1,
      }),
    );
  });
});

/**
 * Rewind, set the stage, then measure — in that order, every time.
 *
 * `frameMean` advances the world by two steps to let the frame converge, so two
 * consecutive calls photograph *different seas* and their difference is mostly
 * the swell moving. That is not a subtle effect: it is larger than the whole
 * contribution of a restrained bloom, and it is what made the first cut of the
 * Low test fail against a stage that was working correctly.
 */
async function measureAt(
  page: import('@playwright/test').Page,
  time: number,
  prepare: () => Promise<void>,
): Promise<number> {
  await page.evaluate((t) => window.__ocean.resetDeterministic(t, 90), time);
  await prepare();
  return frameMean(page);
}

test.describe('bloom', () => {
  const SUNSET_TIME = 31.25;

  test('lifts the sun track and cannot darken the frame', async ({ page }) => {
    await bootOcean(page);
    // Sunset, looking straight down the sun's track: the one composition in the
    // shot list where a large part of the frame is genuinely above 1.0 in
    // linear, which is what the pyramid's threshold selects on.
    await setState(page, { preset: 'sunset', quality: 'high', cameraMode: 'orbit' });
    await setCamera(page, [58, 7, 9], [0, 4, 0]);

    const on = await measureAt(page, SUNSET_TIME, () =>
      page.evaluate(() => window.__ocean.setBloomEnabled(true)),
    );
    const off = await measureAt(page, SUNSET_TIME, () =>
      page.evaluate(() => window.__ocean.setBloomEnabled(false)),
    );

    // Additive by construction: `bloom()` returns a contribution which the chain
    // adds. If this ever came out darker, the stage would be replacing the image
    // rather than adding to it — which is the mistake the node's own usage
    // example exists to prevent, and which a baseline diff would not name.
    expect(on).toBeGreaterThan(off);

    await page.evaluate(() => window.__ocean.setBloomEnabled(true));
  });

  test('the Low tier actually reaches the stage', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'sunset', quality: 'low', cameraMode: 'orbit' });
    await setCamera(page, [58, 7, 9], [0, 4, 0]);

    // Low differs from High for a dozen reasons besides bloom, so comparing the
    // two tiers proves nothing about this stage. What does prove something is
    // asking whether the tier's own value arrived: at Low, turning bloom off
    // must be a no-op, and forcing it on must not be. A `bloom` field that
    // nothing read would pass the second check and fail the first.
    const asTierLeftIt = await measureAt(page, SUNSET_TIME, async () => {});
    const forcedOff = await measureAt(page, SUNSET_TIME, () =>
      page.evaluate(() => window.__ocean.setBloomEnabled(false)),
    );
    expect(Math.abs(forcedOff - asTierLeftIt)).toBeLessThan(0.01);

    const forcedOn = await measureAt(page, SUNSET_TIME, () =>
      page.evaluate(() => window.__ocean.setBloomEnabled(true)),
    );
    expect(forcedOn).toBeGreaterThan(asTierLeftIt);

    await setState(page, { quality: 'high' });
  });
});

test.describe('output transform', () => {
  /**
   * The takeover has to be provably neutral.
   *
   * `RenderPipeline` would have applied ACES and the sRGB transfer itself; this
   * project switches `outputColorTransform` off and does it in `OutputTransform`
   * so that there is a stage running late enough to dither. The evidence that
   * this was safe is that with the dither at zero the stage is the renderer's
   * own output node with the renderer's own settings — and that two captures of
   * an unchanged world are still byte-identical, which a stage that had
   * introduced any frame-dependent state would break.
   */
  test('is deterministic, and its dither is bounded to a level', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'skyPro', quality: 'high', cameraMode: 'orbit' });
    await setCamera(page, [-42, 21, 63], [0, 3, 0]);

    const grab = async (levels: number) => {
      await page.evaluate((n) => window.__ocean.setDitherLevels(n), levels);
      await page.evaluate(() => window.__ocean.resetDeterministic(12, 90));
      return frameHead(page, 262144);
    };

    const plain = await grab(0);
    const plainAgain = await grab(0);
    // Byte-exact across two full re-applications: the dither is a pure function
    // of pixel coordinates with no time term, which is what keeps every
    // cinematic and gallery capture reproducible.
    expect(plain).toEqual(plainAgain);

    const dithered = await grab(1);
    expect(dithered).not.toEqual(plain);

    // A triangular density spanning +/-1 LSB can move a channel by at most one
    // level before rounding, so nothing may move by more than two. A dither that
    // was being applied in linear rather than in display space would blow
    // straight through this in the shadows, where the sRGB curve is steepest —
    // which is the specific mistake this bound exists to catch.
    let worst = 0;
    let moved = 0;
    for (let i = 0; i < plain.length; i++) {
      const d = Math.abs(plain[i] - dithered[i]);
      if (d > worst) worst = d;
      if (d > 0) moved++;
    }
    expect(worst).toBeLessThanOrEqual(2);
    // And it must reach a real share of the frame, or it is not dithering.
    //
    // Deliberately loose. Whether a dithered channel actually *crosses* a
    // rounding boundary depends on where its undithered value sits between two
    // levels: a triangular density spanning +/-1 LSB moves a value across at
    // most about a quarter of the time, and less than that wherever the signal
    // happens to sit near the middle of a level. Measured here at 14% over a
    // patch of sky. The bound above is the assertion with teeth; this one only
    // rules out a dither that is silently doing nothing.
    expect(moved / plain.length).toBeGreaterThan(0.05);

    await page.evaluate(() => window.__ocean.setDitherLevels(1));
  });
});

test.describe('depth of field', () => {
  test('is a pass-through at zero taps and softens off the focal plane', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'skyPro', quality: 'high', cameraMode: 'orbit' });
    // Looking along the water rather than down at it, so the frame carries a
    // real spread of distances: near foreground swell, the hull at the focal
    // plane, and the horizon. A shot with everything at one distance cannot
    // show a depth of field whether or not it works.
    await setCamera(page, [-42, 12, 63], [0, 3, 0]);

    const grab = async (samples: number, fNumber: number) => {
      await page.evaluate(([s, n]) => window.__ocean.setDof(s, n), [samples, fNumber]);
      await page.evaluate(() => window.__ocean.resetDeterministic(12, 90));
      return frameHead(page, 262144);
    };

    // Zero taps must be bit-exact, not merely close: the gather sits behind a
    // uniform-coherent branch, so at zero the source texel is returned
    // untouched rather than summed with itself and divided by one.
    const off = await grab(0, 5.6);
    const offAgain = await grab(0, 1.4);
    expect(off).toEqual(offAgain);

    // A wide aperture must change the frame.
    const wide = await grab(24, 1.4);
    expect(wide).not.toEqual(off);

    // And a narrow one must change it less than a wide one does. This is the
    // assertion that distinguishes a real circle of confusion from any blur
    // that merely responds to a knob: the CoC scales with aperture diameter, so
    // f/16 has to sit between f/1.4 and no lens at all.
    const narrow = await grab(24, 16);
    const spread = (a: number[], b: number[]) => {
      let sum = 0;
      for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
      return sum / a.length;
    };
    expect(spread(off, narrow)).toBeLessThan(spread(off, wide));

    await page.evaluate(() => window.__ocean.setDof(16, 5.6));
  });
});

test.describe('lens flare', () => {
  test('brightens up-sun, is absent under water, and dies when the sun is occluded', async ({
    page,
  }) => {
    await bootOcean(page);
    await setState(page, { preset: 'sunset', quality: 'high', cameraMode: 'orbit' });

    // 1. Looking straight down the sunset sun's track, above water.
    await setCamera(page, [58, 7, 9], [0, 4, 0]);
    const on = await measureAt(page, 31.25, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(true)),
    );
    const off = await measureAt(page, 31.25, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(false)),
    );
    // Purely additive: it cannot darken a frame.
    expect(on).toBeGreaterThan(off);

    // 2. Looking *away* from the sun must be far less affected than looking at
    //    it. This is the angular gate, and it is what separates a flare
    //    anchored to the light from one that fires on any bright pixel — a sea
    //    full of sun glitter looks much the same in both directions.
    await setCamera(page, [-58, 7, -9], [0, 4, 0]);
    const awayOn = await measureAt(page, 31.25, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(true)),
    );
    const awayOff = await measureAt(page, 31.25, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(false)),
    );
    expect(awayOn - awayOff).toBeLessThan(on - off);

    // 3. Submerged, it must contribute nothing at all — the request was for a
    //    flare above water, and there is no lens down there to scatter in.
    await setState(page, { preset: 'skyPro' });
    await setCamera(page, [-16, -7, 16], [-6, 2, 2]);
    // Discarded: the first capture after a camera jump still carries a little
    // of the previous pose (the foam round the hull, amplified by TRAA, AO and
    // exposure), so the pair compared below would otherwise differ by more
    // than the flare. See "capture history" in docs/AAA_PLAN.md.
    await measureAt(page, 72.5, async () => {});
    const underOn = await measureAt(page, 72.5, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(true)),
    );
    const underOff = await measureAt(page, 72.5, () =>
      page.evaluate(() => window.__ocean.setFlareEnabled(false)),
    );
    expect(Math.abs(underOn - underOff)).toBeLessThan(0.01);

    await page.evaluate(() => window.__ocean.setFlareEnabled(true));
  });
});
