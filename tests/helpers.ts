import type { Page, ConsoleMessage } from '@playwright/test';

export interface OceanHandle {
  backend: 'webgpu';
}

/**
 * Console errors that are environmental rather than defects in this project.
 * Kept deliberately narrow so real errors are never swallowed.
 */
const IGNORABLE = [
  /favicon\.ico/i,
  /powerPreference option is currently ignored/i,
  /Automatic fallback to software WebGL/i,
];

export function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message: ConsoleMessage) => {
    if (message.type() !== 'error') return;
    const text = message.text();
    if (IGNORABLE.some((pattern) => pattern.test(text))) return;
    errors.push(text);
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

/**
 * Waits for the app to boot and to have actually drawn.
 *
 * Deliberately NOT expressed as a number of frames. Automated Chromium throttles
 * requestAnimationFrame to roughly 1 Hz regardless of load, so "wait for 90
 * frames" is a 90-second wait that reliably exceeds the test timeout. The
 * readiness signal is that the loop has recorded a completed frame, plus a
 * wall-clock settle for shader compilation and the first GPU readback.
 */
export async function waitForOcean(page: Page, settleMs = 2500): Promise<void> {
  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 60_000 });
  await page.waitForFunction(
    () => {
      const ocean = (
        window as unknown as { __ocean: { loop: { stats: { frameMs: number } } } }
      ).__ocean;
      return ocean.loop.stats.frameMs > 0;
    },
    undefined,
    // 240 s, and the number tracks a deliberate change in what boot *is*.
    //
    // The loop used to start as soon as the sea and the sky were compiled, and
    // the ship, the island and their sixty materials were compiled afterwards,
    // while frames were already being drawn — so a frame arrived within a couple
    // of seconds and 60 s here was generous. Compiling all of it behind the boot
    // overlay is what removes the hitch that used to land seconds into a session,
    // and the price is that the *first frame* now waits for the whole compile.
    //
    // Measured on this harness: 90 s to first frame on WebGPU, 195 s on the
    // WebGL2 fallback, against about 2 s to parse the models — so this is
    // shader compilation and nothing else. Both figures are inflated by
    // automation throttling rAF and are not what a user waits for.
    { timeout: 240_000 },
  );
  // A first frame is not a ready frame. On the WebGL2 fallback the loop starts
  // drawing while two dozen pipelines are still being built in the background,
  // at seconds a frame, and those frames are black. `isReady` waits for the
  // pipeline backlog, the sampler and the scene content.
  await page.waitForFunction(
    () => (window as unknown as { __ocean: { isReady: () => boolean } }).__ocean.isReady(),
    undefined,
    { timeout: 240_000 },
  );
  // Settle: shader compilation, mip chain construction, first sampler readback,
  // and the boot overlay fade so it never bleeds into a screenshot.
  await page.waitForTimeout(settleMs);
}

/** Places the camera deterministically so screenshots are comparable run to run. */
export async function setCamera(
  page: Page,
  position: [number, number, number],
  target: [number, number, number],
): Promise<void> {
  await page.evaluate(
    ([p, t]) => {
      const ocean = (
        window as unknown as {
          __ocean: {
            setCamera: (
              px: number, py: number, pz: number,
              tx: number, ty: number, tz: number,
            ) => void;
          };
        }
      ).__ocean;
      ocean.setCamera(p[0], p[1], p[2], t[0], t[1], t[2]);
    },
    [position, target] as const,
  );
  await page.waitForTimeout(400);
}

export async function setState(page: Page, partial: Record<string, unknown>): Promise<void> {
  // Awaited, and that matters for `quality`. A tier change waits for the GPU
  // queue to drain before it destroys anything — destroying a texture three has
  // already referenced in a submitted command buffer is a use-after-free, and
  // WebGPU reports it as one. Without the await, a test would assert on the tier
  // it had just left.
  await page.evaluate(async (p) => {
    const ocean = (
      window as unknown as {
        __ocean: { setState: (x: Record<string, unknown>) => void | Promise<void> };
      }
    ).__ocean;
    await ocean.setState(p);
  }, partial);
  await page.waitForTimeout(600);
}

/**
 * True when the browser has a real WebGPU adapter.
 *
 * Playwright's bundled Chromium frequently has none, which forces the app onto
 * the WebGL2 path backed by a *software* rasteriser. That is a correct fallback,
 * but at 1600x900 it is slow enough that the compositor will not deliver a frame
 * inside a test timeout — so any assertion requiring a screenshot has to be
 * skipped rather than made to pass artificially.
 */
export async function hasGpuAdapter(page: Page): Promise<boolean> {
  return page.evaluate(async () => {
    try {
      return navigator.gpu ? (await navigator.gpu.requestAdapter()) !== null : false;
    } catch {
      return false;
    }
  });
}

export interface FrameSample {
  fps: number;
  frameMs: number;
  /**
   * True when the browser is pacing requestAnimationFrame far below the work the
   * renderer is actually doing — i.e. `fps` is a harness artefact, not a
   * measurement of this project.
   */
  rafThrottled: boolean;
}

/**
 * Samples frame cost over a window, discarding an initial settling period.
 *
 * Reports BOTH the rAF-derived rate and the measured per-frame work, because in
 * automated Chromium the two diverge wildly: this project was observed at 1.1
 * "FPS" while spending 0.8 ms per frame, which would be ~1250 FPS of work. The
 * browser throttles rAF in headless/unfocused contexts regardless of load — the
 * same 1.00 fps appears with every effect disabled. Assertions must therefore key
 * on `frameMs`, and treat `fps` as meaningful only when `rafThrottled` is false.
 */
export async function measureFrameRate(page: Page, seconds = 4): Promise<FrameSample> {
  await page.waitForTimeout(1500); // settle: shader compiles, mip builds, GC
  return page.evaluate(async (duration) => {
    const ocean = (
      window as unknown as { __ocean: { loop: { stats: { fps: number; frameMs: number } } } }
    ).__ocean;
    const samples: { fps: number; frameMs: number }[] = [];
    const start = performance.now();
    while (performance.now() - start < duration * 1000) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      samples.push({ ...ocean.loop.stats });
    }
    // Median is robust to the occasional compositor hitch that a mean is not.
    const median = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)];
    };
    const fps = median(samples.map((s) => s.fps));
    const frameMs = median(samples.map((s) => s.frameMs));
    // If the frame budget implied by the delivered rate is more than 4x the work
    // actually performed, the browser is pacing us, not the renderer.
    const rafThrottled = fps > 0 && 1000 / fps > frameMs * 4 && fps < 50;
    return { fps, frameMs, rafThrottled };
  }, seconds);
}

/**
 * Mean of R+G+B across the whole frame.
 *
 * Coarse on purpose. It exists to answer one question — did an *additive* post
 * stage contribute anything — and for that a single scalar over the frame is
 * both sufficient and immune to the specular sparkle that makes per-pixel
 * comparisons of this scene noisy.
 *
 * Two steps before the read, never zero: the frame immediately after a state
 * change is not converged, and `tests/lib/capture.ts` measures that difference
 * at roughly a factor of thirty in mean ΔE.
 */
export async function frameMean(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const ocean = (
      window as unknown as {
        __ocean: {
          step: (dt: number, steps?: number) => Promise<void>;
          capturePixels: () => Promise<{ data: Uint8Array }>;
        };
      }
    ).__ocean;
    await ocean.step(1 / 60, 2);
    const { data } = await ocean.capturePixels();
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) sum += data[i] + data[i + 1] + data[i + 2];
    return sum / (data.length / 4);
  });
}

/**
 * The first `count` bytes of the frame, as plain numbers.
 *
 * For byte-exactness checks, where the question is "is this stage a genuine
 * pass-through" rather than "does this look right". A prefix rather than the
 * whole frame because it crosses the `page.evaluate` boundary as JSON.
 */
export async function frameHead(page: Page, count = 65536): Promise<number[]> {
  return page.evaluate(async (n) => {
    const ocean = (
      window as unknown as {
        __ocean: {
          step: (dt: number, steps?: number) => Promise<void>;
          capturePixels: () => Promise<{ data: Uint8Array }>;
        };
      }
    ).__ocean;
    await ocean.step(1 / 60, 2);
    const { data } = await ocean.capturePixels();
    return Array.from(data.slice(0, n));
  }, count);
}
