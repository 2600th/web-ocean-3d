import { defineConfig, devices } from '@playwright/test';

/**
 * The scene is GPU-heavy and non-deterministic frame to frame, so the suite is
 * built around *behavioural* assertions (no console errors, stable frame budget,
 * interactions take effect) plus screenshot comparison with a generous pixel
 * tolerance rather than exact-match snapshots.
 *
 * Workers are pinned to 1: parallel WebGPU contexts contend for the same device
 * and turn frame-rate assertions into coin flips.
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : [['list']],
  /**
   * Generous, because this is a GPU-bound scene and the runner is not always on
   * a GPU. With the adapter flags below the whole suite is quick; on a machine
   * that genuinely has no GPU, Chromium falls back to a software rasteriser and
   * a single frame of this scene can take tens of seconds. Measured on a
   * software-only runner: 33 minutes, with the failures being timeouts rather
   * than assertion failures.
   *
   * See docs/PERFORMANCE.md — the suite is meaningful on GPU hardware; on a
   * software runner only the non-visual assertions are trustworthy.
   */
  timeout: 240_000,
  expect: { timeout: 30_000 },

  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },

  projects: [
    /**
     * Functional suite.
     *
     * The GPU flags matter more here than they look. Without `--enable-gpu`,
     * Playwright's headless Chromium exposes no WebGPU adapter at all, so the
     * app correctly falls back to WebGL2 — on a *software* rasteriser. Measured
     * on this machine, that turned a 33-minute suite with three skips and two
     * timeouts into the default experience, and the two timeouts were the tier
     * and leak tests, which cycle quality six and eight times respectively and
     * simply cannot finish when a frame takes seconds. The same two sequences
     * complete in 4.2 s and 5.3 s against a real adapter.
     *
     * `--disable-dawn-features=use_dxc` is required alongside it: with
     * `--enable-gpu` alone the adapter appears and `requestDevice()` then fails
     * with `DynamicLib.Open: dxil.dll Windows Error 87`, because Chromium's
     * bundled DXC will not load. Falling back to FXC on the same D3D12 backend
     * brings the device up, and is harmless where DXC does load.
     *
     * `--enable-features=Vulkan` was removed: on Windows the backend is D3D12,
     * and asking for Vulkan here bought nothing.
     */
    {
      name: 'chromium-webgpu',
      // Everything the `visual` project owns. These need its GPU flags and its
      // 1280x720 shot resolution, and running them here as well would compare
      // captures taken at a different viewport — passing or failing for reasons
      // that have nothing to do with what they measure.
      testIgnore: /(visual|gallery|gallery-jitter|isolation|foam)\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1600, height: 900 },
        deviceScaleFactor: 1,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-gpu',
            '--disable-dawn-features=use_dxc',
            // Deterministic frame pacing for the performance assertions.
            '--disable-frame-rate-limit',
            '--disable-gpu-vsync',
          ],
        },
      },
    },
    /**
     * The visual regression suite. Separate from the functional project because
     * it needs three things that project does not:
     *
     * **A real GPU.** Playwright's Chromium runs headless on SwiftShader unless
     * `--enable-gpu` is passed, and without it there is no WebGPU adapter at all
     * — measured on this machine, `requestAdapter()` returns null, the app falls
     * back to WebGL2 on a CPU rasteriser, and every shot skips. That is the
     * honest outcome, but it is not a useful default for a suite whose whole
     * purpose is comparing GPU output.
     *
     * **A working Dawn device.** With `--enable-gpu` alone the adapter appears
     * and then `requestDevice()` fails with `DynamicLib.Open: dxil.dll Windows
     * Error 87` — Chromium's bundled DXC shader compiler will not load here.
     * Turning off the `use_dxc` Dawn feature falls back to FXC on the same D3D12
     * backend and the device comes up. Harmless where DXC does load.
     *
     * **The canonical shot resolution**, which is 1280x720 — see
     * `SHOT_VIEWPORT` in `tests/lib/shots.ts` for why it is not the 1600x900 the
     * performance budget is quoted at.
     */
    {
      name: 'visual',
      // `gallery.spec.ts` lives here too: it needs the same GPU flags and the
      // same shot resolution, and it skips itself unless CAPTURE_GALLERY=1.
      testMatch: /(visual|gallery|gallery-jitter|isolation|foam)\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-gpu',
            '--disable-dawn-features=use_dxc',
            '--disable-frame-rate-limit',
            '--disable-gpu-vsync',
          ],
        },
      },
    },
  ],

  /**
   * Always rebuilt, never reused.
   *
   * `reuseExistingServer: !process.env.CI` is the usual default and it is wrong
   * for this project. A preview server left running from an earlier invocation
   * keeps serving the `dist/` it was started with, so a local run silently tests
   * whatever was built minutes ago. That is bad anywhere; here it is corrosive,
   * because the visual baselines are *generated* by this harness — a stale
   * server bakes yesterday's renderer into the images every later run is
   * measured against. It happened twice while this suite was being written, and
   * the second time it cost a full set of baselines.
   *
   * The build is a couple of seconds. Correctness is worth more.
   */
  webServer: {
    command: 'npm run build && npm run preview',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
