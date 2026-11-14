import { test, expect, type Page, type TestInfo } from '@playwright/test';
import {
  ARTIFACT_DIR,
  definitionDifferences,
  hasBaseline,
  readBaseline,
  writeArtifact,
  writeBaseline,
  writeReport,
} from './lib/baselines';
import { applyShot, bootOcean, capture } from './lib/capture';
import {
  compareImages,
  formatResult,
  PER_PIXEL_DELTA_E,
  withinThresholds,
  type ComparisonResult,
  type Thresholds,
} from './lib/compare';
import { decodePng, encodePng, type RgbaImage } from './lib/png';
import {
  SHOTS,
  SHOT_VIEWPORT,
  thresholdsFor,
  type NoiseFloor,
  type Shot,
} from './lib/shots';
import {
  describeStack,
  formatStack,
  isSoftwareRenderer,
  stackDifferences,
  type StackFingerprint,
} from './lib/stack';

/**
 * Deterministic visual regression over the canonical shots.
 *
 * Read `docs/VERIFICATION.md` before touching a threshold. The short version:
 * the limits are derived from a *measured* frame-to-frame noise floor rather
 * than chosen to make the suite green, and the response to a failure is to open
 * the diff artefact — never to raise the limit until it passes.
 *
 * Environment switches:
 *   UPDATE_BASELINES=1   regenerate `tests/baselines/**` from this run
 *   MEASURE_NOISE=1      re-measure the noise floor and print a block ready to
 *                        paste into `MEASURED_NOISE_FLOOR` in `tests/lib/shots.ts`
 *   KEEP_ARTIFACTS=1     write actual/baseline/diff even for shots that pass
 */

const UPDATE_BASELINES = process.env.UPDATE_BASELINES === '1';
const MEASURE_NOISE = process.env.MEASURE_NOISE === '1';
const KEEP_ARTIFACTS = process.env.KEEP_ARTIFACTS === '1';

/** Captures per shot for the noise measurement: five samples, ten pairs. */
const NOISE_SAMPLES = 5;

interface ScoreRecord {
  meanDeltaE: number;
  p95DeltaE: number;
  maxDeltaE: number;
  fractionAbove: number;
  thresholds: Thresholds;
  passed: boolean;
}

function round(value: number, digits = 3): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

test.describe('image pipeline', () => {
  /**
   * The PNG codec is written from the format spec against `node:zlib` rather
   * than taken from a package, and everything downstream — the baselines, the
   * artefacts, the metric — is worthless if it is subtly wrong. A round trip
   * over content that makes different scanlines prefer different filters is
   * cheap and needs no GPU, so it runs even where the shots skip.
   */
  test('PNG round-trips pixel-exactly', () => {
    const width = 137;
    const height = 91;
    const data = new Uint8Array(width * height * 4);
    let seed = 0x02f6e2b1;
    for (let i = 0; i < width * height; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const x = i % width;
      const y = (i / width) | 0;
      // Two smooth ramps and one hash channel, so no single filter type wins
      // every row and the adaptive selection is genuinely exercised.
      data[i * 4] = (x * 2) & 0xff;
      data[i * 4 + 1] = (y * 3) & 0xff;
      data[i * 4 + 2] = (seed >>> 16) & 0xff;
      data[i * 4 + 3] = 255;
    }

    const decoded = decodePng(encodePng({ width, height, data }));
    expect(decoded.width).toBe(width);
    expect(decoded.height).toBe(height);
    expect(Buffer.compare(Buffer.from(decoded.data), Buffer.from(data))).toBe(0);
  });

  test('an image compared with itself scores zero', () => {
    const image = solid(16, 16, 90, 140, 200);
    const result = compareImages(image, image);
    expect(result.meanDeltaE).toBe(0);
    expect(result.p95DeltaE).toBe(0);
    expect(result.fractionAbove).toBe(0);
  });

  /**
   * Guards the property the harness rests on: the metric responds to *where*
   * pixels changed, not to the frame average. Moving a patch from one corner to
   * another leaves the mean colour untouched — which is precisely what the
   * average-colour fingerprint in `ocean.spec.ts` cannot see, and the reason
   * this suite does not reuse it.
   */
  test('a relocated patch scores non-zero despite an identical average', () => {
    const before = solid(64, 64, 40, 60, 80);
    const after = solid(64, 64, 40, 60, 80);
    paint(before, 0, 0, 16, 16, 200, 30, 30);
    paint(after, 48, 48, 16, 16, 200, 30, 30);
    expect(averageChannel(before)).toEqual(averageChannel(after));

    const result = compareImages(before, after);
    expect(result.fractionAbove).toBeGreaterThan(0.1);
    expect(result.p95DeltaE).toBeGreaterThan(PER_PIXEL_DELTA_E);
  });

  /**
   * The software-rasteriser guard is the one skip condition this machine cannot
   * reach on demand — reaching it needs a WebGPU adapter backed by a CPU
   * renderer. The string it has to recognise is not hypothetical though: it is
   * what Playwright's Chromium reports without `--enable-gpu`, so at least the
   * matcher can be held to it.
   */
  test('the software-rasteriser guard recognises the renderers it must', () => {
    expect(
      isSoftwareRenderer(
        'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)',
      ),
    ).toBe(true);
    expect(isSoftwareRenderer('llvmpipe (LLVM 15.0.7, 256 bits)')).toBe(true);
    expect(
      isSoftwareRenderer(
        'ANGLE (NVIDIA, NVIDIA GeForce RTX 5090 (0x00002B85) Direct3D11 vs_5_0 ps_5_0, D3D11)',
      ),
    ).toBe(false);
  });
});

test.describe('canonical shots', () => {
  let page: Page;
  let stack: StackFingerprint;
  /** Non-empty when this stack cannot produce a capture worth comparing. */
  let skipReason = '';
  const measurements: Record<string, NoiseFloor> = {};
  const results: Record<string, ScoreRecord> = {};

  test.beforeAll(async ({ browser }, testInfo) => {
    // One page for the whole describe rather than one per test: booting this
    // scene costs ten to twenty seconds, and the shots are independent by
    // construction because each re-applies its full state and calls
    // `resetDeterministic` before it captures.
    page = await browser.newPage({
      viewport: { ...SHOT_VIEWPORT },
      deviceScaleFactor: 1,
      baseURL: testInfo.project.use.baseURL,
    });
    await bootOcean(page);
    stack = await describeStack(page);
    testInfo.annotations.push({ type: 'stack', description: formatStack(stack) });

    if (stack.backend !== 'webgpu') {
      skipReason =
        `the app is on the ${stack.backend} backend (WebGPU adapter: ${stack.adapter}). ` +
        'Baselines are captured on WebGPU; comparing a fallback render against them ' +
        'would measure the backend, not a regression.';
    } else if (isSoftwareRenderer(stack.gpu)) {
      skipReason =
        `rendering on a software rasteriser (${stack.gpu}). Its texture filtering and ` +
        'rounding differ from any GPU, so a comparison against a GPU baseline says ' +
        'nothing about this project.';
    }
    if (skipReason) console.warn(`\n[visual] SKIPPING every shot: ${skipReason}\n`);
  });

  test.afterAll(async () => {
    await page?.close();
    reportNoiseFloor(stack, measurements);
    // Written on pass as well as on failure: a run that says only "green" gives
    // no way to see a shot creeping toward its limit before it crosses it.
    if (Object.keys(results).length > 0) {
      writeReport('comparison.json', { comparedAt: new Date().toISOString(), stack, results });
    }
  });

  /**
   * Proves the shots contain no UI, rather than assuming it.
   *
   * `capturePixels` reads a render target, and a render target cannot contain
   * DOM — but that is an argument, not evidence, and the harness would look
   * identical if the HUD were somehow compositing into the frame. So: capture,
   * hide the overlay, capture again, and require the two to be byte-identical.
   * The test first asserts the overlay is actually on screen, otherwise it would
   * pass just as happily on a page where the UI failed to build.
   */
  test('captures exclude the HUD and control panel', async () => {
    test.skip(skipReason !== '', skipReason);

    await applyShot(page, SHOTS[0]);
    const overlay = await page.evaluate(() => {
      const selectors = ['.hud', '.panel', '.panel-toggle'];
      return selectors.map((selector) => {
        const element = document.querySelector(selector);
        const box = element?.getBoundingClientRect();
        return { selector, exists: !!element, onScreen: !!box && box.width > 0 && box.height > 0 };
      });
    });
    for (const { selector, exists } of overlay) {
      expect(exists, `${selector} is missing, so this test would prove nothing`).toBe(true);
    }
    // `.panel-toggle` is `display: none` above the mobile breakpoint by design,
    // so at the shot viewport only the HUD and the panel are on screen — but all
    // three still get hidden below, in case that changes.
    expect(
      overlay.filter((o) => o.onScreen).map((o) => o.selector),
      'no overlay is on screen at the shot viewport, so this test would prove nothing',
    ).toEqual(['.hud', '.panel']);

    const withUi = await capture(page);
    await page.evaluate(() => {
      for (const selector of ['.hud', '.panel', '.panel-toggle']) {
        const element = document.querySelector(selector);
        if (element instanceof HTMLElement) element.style.visibility = 'hidden';
      }
    });
    // From the same reset, not straight after the first capture: a capture is a
    // frame of its own, with its own TRAA jitter and history step, so two in a
    // row differ by the resolve however the UI is set.
    await applyShot(page, SHOTS[0]);
    const withoutUi = await capture(page);
    await page.evaluate(() => {
      for (const selector of ['.hud', '.panel', '.panel-toggle']) {
        const element = document.querySelector(selector);
        if (element instanceof HTMLElement) element.style.visibility = '';
      }
    });

    // Identical modulo 8-bit quantisation, rather than byte-exact.
    //
    // Byte-exactness is the wrong bar for a GPU render. Measured on the
    // reference stack, two captures of a genuinely unchanged frame differ in
    // about twenty bytes out of 3.7 million, always by exactly 1 — pixels whose
    // colour lands on a rounding boundary and falls either side of it between
    // renders. Demanding zero makes this test fail for reasons that have nothing
    // to do with the UI.
    //
    // It still detects what it exists to detect. UI bleeding into a capture is
    // a solid region of glass panel and text: tens of thousands of pixels,
    // differing by tens or hundreds of levels. Nothing about the bound below
    // could absorb that.
    let differing = 0;
    let maxDelta = 0;
    for (let i = 0; i < withUi.data.length; i++) {
      const delta = Math.abs(withUi.data[i] - withoutUi.data[i]);
      if (delta > 0) {
        differing++;
        if (delta > maxDelta) maxDelta = delta;
      }
    }
    const differingFraction = differing / withUi.data.length;

    expect(
      maxDelta,
      `hiding the overlay changed a pixel by ${maxDelta} levels, which is more than ` +
        'rounding — the shots contain UI',
    ).toBeLessThanOrEqual(1);
    expect(
      differingFraction,
      `hiding the overlay changed ${differing} of ${withUi.data.length} bytes ` +
        `(${(100 * differingFraction).toFixed(4)}%) — the shots contain UI`,
    ).toBeLessThan(0.0001);
  });

  for (const shot of SHOTS) {
    test(shot.id, async ({}, testInfo) => {
      test.skip(skipReason !== '', skipReason);
      // A reset settles 90 simulation steps, each stalling on a GPU readback, so
      // one shot is seconds rather than milliseconds — and the noise measurement
      // does the whole thing five times.
      test.setTimeout(MEASURE_NOISE ? 600_000 : 180_000);

      await applyShot(page, shot);
      const actual = await capture(page);

      expect(
        `${actual.width}x${actual.height}`,
        'capture resolution does not match the canonical shot size',
      ).toBe(`${SHOT_VIEWPORT.width}x${SHOT_VIEWPORT.height}`);

      if (MEASURE_NOISE) measurements[shot.id] = await measureNoise(page, shot, actual);

      if (UPDATE_BASELINES) {
        writeBaseline(shot, actual, stack);
        testInfo.annotations.push({
          type: 'baseline',
          description: `regenerated on ${formatStack(stack)}`,
        });
        return;
      }

      expect(
        hasBaseline(shot.id),
        `no baseline for "${shot.id}". Capture one with:\n` +
          '  UPDATE_BASELINES=1 npx playwright test --project=visual',
      ).toBe(true);

      const baseline = readBaseline(shot.id);
      const warnings = [
        ...stackDifferences(baseline.meta.stack, stack),
        ...definitionDifferences(shot, baseline.meta),
      ];
      if (warnings.length > 0) {
        console.warn(
          `\n${'='.repeat(78)}\n` +
            `[visual] ${shot.id}: COMPARING ACROSS A CHANGED STACK OR DEFINITION.\n` +
            '[visual] Whatever the numbers below say, they are not evidence of a\n' +
            '[visual] rendering regression until this is resolved.\n' +
            warnings.map((w) => `[visual]   ${w}`).join('\n') +
            `\n${'='.repeat(78)}\n`,
        );
        testInfo.annotations.push({ type: 'stack-mismatch', description: warnings.join('; ') });
      }

      const result = compareImages(baseline.image, actual);
      const thresholds = thresholdsFor(shot.id);
      const passed = withinThresholds(result, thresholds);

      results[shot.id] = {
        meanDeltaE: round(result.meanDeltaE),
        p95DeltaE: round(result.p95DeltaE),
        maxDeltaE: round(result.maxDeltaE),
        fractionAbove: round(result.fractionAbove, 5),
        thresholds,
        passed,
      };
      testInfo.annotations.push({
        type: 'score',
        description:
          `mean ΔE ${result.meanDeltaE.toFixed(3)}/${thresholds.meanDeltaE.toFixed(3)}, ` +
          `p95 ${result.p95DeltaE.toFixed(2)}/${thresholds.p95DeltaE.toFixed(2)}, ` +
          `above ${(100 * result.fractionAbove).toFixed(2)}%/` +
          `${(100 * thresholds.fractionAbove).toFixed(2)}%`,
      });

      if (!passed || KEEP_ARTIFACTS) {
        await attachArtifacts(testInfo, shot.id, actual, baseline.image, result);
      }

      expect(
        passed,
        `${shot.id} — ${shot.title}\n${formatResult(result, thresholds)}\n` +
          (warnings.length > 0 ? `  NOTE: ${warnings.join('; ')}\n` : '') +
          `  artefacts: ${ARTIFACT_DIR}\n` +
          '  Open the diff before deciding this is expected. If it is, regenerate with\n' +
          '  UPDATE_BASELINES=1 npx playwright test --project=visual',
      ).toBe(true);
    });
  }
});

/**
 * Re-applies the shot from scratch until there are `NOISE_SAMPLES` captures of
 * it, and reports the worst metric over every pair.
 *
 * Re-applying the whole shot is the point. Repeating only `capturePixels`
 * measures the readback, which is exactly reproducible and would report a noise
 * floor of zero. What the harness does between one run and the next is set the
 * state, rewind every clock and settle — so that is what has to be measured.
 * Worst pair rather than mean pair, because a threshold set on the average of
 * the noise fails on the noise half the time.
 */
async function measureNoise(page: Page, shot: Shot, first: RgbaImage): Promise<NoiseFloor> {
  const samples: RgbaImage[] = [first];
  for (let i = 1; i < NOISE_SAMPLES; i++) {
    await applyShot(page, shot);
    samples.push(await capture(page));
  }

  const worst: NoiseFloor = { meanDeltaE: 0, p95DeltaE: 0, fractionAbove: 0 };
  for (let i = 0; i < samples.length; i++) {
    for (let j = i + 1; j < samples.length; j++) {
      const result = compareImages(samples[i], samples[j]);
      worst.meanDeltaE = Math.max(worst.meanDeltaE, result.meanDeltaE);
      worst.p95DeltaE = Math.max(worst.p95DeltaE, result.p95DeltaE);
      worst.fractionAbove = Math.max(worst.fractionAbove, result.fractionAbove);
    }
  }
  return worst;
}

function reportNoiseFloor(stack: StackFingerprint, measurements: Record<string, NoiseFloor>): void {
  if (Object.keys(measurements).length === 0) return;
  const path = writeReport('noise-floor.json', {
    measuredAt: new Date().toISOString(),
    stack,
    samplesPerShot: NOISE_SAMPLES,
    perPixelDeltaE: PER_PIXEL_DELTA_E,
    shots: measurements,
  });
  const lines = Object.entries(measurements).map(([id, noise]) => {
    const key = /^[A-Za-z_$][\w$]*$/.test(id) ? id : `'${id}'`;
    return (
      `  ${key}: { meanDeltaE: ${noise.meanDeltaE.toFixed(4)}, ` +
      `p95DeltaE: ${noise.p95DeltaE.toFixed(4)}, ` +
      `fractionAbove: ${noise.fractionAbove.toFixed(5)} },`
    );
  });
  console.log(
    `\n[visual] noise floor written to ${path}\n` +
      '[visual] paste into MEASURED_NOISE_FLOOR in tests/lib/shots.ts:\n\n' +
      `${lines.join('\n')}\n`,
  );
}

async function attachArtifacts(
  testInfo: TestInfo,
  shotId: string,
  actual: RgbaImage,
  baseline: RgbaImage,
  result: ComparisonResult,
): Promise<void> {
  const files: [string, RgbaImage][] = [
    ['actual', actual],
    ['baseline', baseline],
    ['diff', result.diff],
  ];
  for (const [kind, image] of files) {
    const path = writeArtifact(shotId, kind, image);
    await testInfo.attach(`${shotId}-${kind}`, { path, contentType: 'image/png' });
  }
}

// ------------------------------------------------------- synthetic test images

function solid(width: number, height: number, r: number, g: number, b: number): RgbaImage {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

function paint(
  image: RgbaImage,
  x0: number,
  y0: number,
  w: number,
  h: number,
  r: number,
  g: number,
  b: number,
): void {
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      const p = (y * image.width + x) * 4;
      image.data[p] = r;
      image.data[p + 1] = g;
      image.data[p + 2] = b;
    }
  }
}

function averageChannel(image: RgbaImage): [number, number, number] {
  let r = 0;
  let g = 0;
  let b = 0;
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) {
    r += image.data[i * 4];
    g += image.data[i * 4 + 1];
    b += image.data[i * 4 + 2];
  }
  return [r / n, g / n, b / n];
}
