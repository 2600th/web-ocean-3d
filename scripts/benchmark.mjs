#!/usr/bin/env node
/**
 * benchmark.mjs — headed, focused-browser GPU benchmark for web-ocean-3d.
 *
 * This exists because the Playwright functional suite cannot measure this
 * project's performance. Automated Chromium paces `requestAnimationFrame`
 * independently of load — the suite has recorded 1.1 "FPS" while the renderer
 * spent 0.8 ms per frame — and the suite's `frameMs` is wall clock around an
 * asynchronous render call, which is a CPU-side upper bound and not GPU time.
 * Neither number can settle a 16.7 ms GPU budget.
 *
 * So this harness does four things the suite does not:
 *
 *  1. Runs a real, visible, foregrounded Chrome window with vsync and the frame
 *     rate limiter disabled, and refuses to report a pass if the window turns out
 *     not to be focused or the browser turns out to be pacing us anyway.
 *  2. Turns on WebGPU timestamp queries and reports **GPU** frame time. Three.js
 *     only wires the query pool up when the backend's `trackTimestamp` is set,
 *     and this project constructs its renderer without it — but the WebGPU
 *     backend requests a device with every feature the adapter advertises, so
 *     `timestamp-query` is already enabled on the device and the flag can simply
 *     be set at runtime, before the first instrumented frame. See `enableTiming`.
 *  3. Drives frames itself rather than watching the app's loop. `Loop.step`
 *     awaits the render, so the harness knows exactly when a frame's GPU work has
 *     been submitted and can resolve that frame's timestamps and read that
 *     frame's draw-call counters without racing the app.
 *  4. Records what the browser actually ran on — adapter, backend, resolution,
 *     tier — into a JSON file, so a number can be argued with later.
 *
 * Usage:
 *   npm run bench                       # build, serve, run the whole matrix
 *   node scripts/benchmark.mjs          # same, assuming `dist/` is already built
 *   node scripts/benchmark.mjs --only webgpu-high,webgl-low
 *   node scripts/benchmark.mjs --frames 900 --warmup 180
 *   node scripts/benchmark.mjs --url http://127.0.0.1:4173   # use a running server
 *   node scripts/benchmark.mjs --headless                    # diagnostics only:
 *                                                            # every run is UNVERIFIED
 *
 * Exits non-zero unless every gated configuration is a measured PASS. An
 * UNVERIFIED gate is not a pass and does not exit zero.
 */

import { execFile, spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import os from 'node:os';

import { chromium } from '@playwright/test';

const execFileAsync = promisify(execFile);
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RESULTS_DIR = join(ROOT, 'bench-results');

/** Version of the JSON result shape. Documented in docs/PERFORMANCE.md. */
const SCHEMA = 'web-ocean-3d/bench@1';

/**
 * Chrome switches. Every one of these changes what is being measured, so they
 * are recorded into the result file alongside the numbers.
 *
 * `--disable-gpu-vsync` and `--disable-frame-rate-limit` stop the compositor
 * from pacing rAF to the display refresh; without them the delivered frame rate
 * is a property of the monitor rather than of the renderer.
 *
 * `--disable-dawn-features=timestamp_quantization` matters more than it looks.
 * Chrome rounds WebGPU timestamps to 100 us by default so that a page cannot use
 * them as a high-resolution timer. A frame of this scene is 110-ish render
 * passes, and rounding each pass independently before summing them puts several
 * milliseconds of quantisation noise into the frame total — enough to swamp the
 * difference between two quality tiers.
 */
const CHROME_FLAGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--disable-frame-rate-limit',
  '--disable-gpu-vsync',
  '--enable-dawn-features=allow_unsafe_apis',
  '--disable-dawn-features=timestamp_quantization',
  '--window-position=0,0',
  '--autoplay-policy=no-user-gesture-required',
];

/**
 * The canonical benchmark view: the same wide shot the visual harness uses, so a
 * frame cost can be put next to a picture of what produced it. Pinned in every
 * configuration — a benchmark that lets the orbit camera drift is comparing
 * different amounts of ocean.
 */
const CAMERA = { position: [0, 14, 48], target: [0, 2, 0] };

/** The existing visual-regression dive, kept as a benchmark configuration. */
const UNDERWATER_CAMERA = { position: [-16, -7, 16], target: [-6, 2, 2] };

/** Preset held fixed across the matrix; presets differ in cost. */
const PRESET = 'skyPro';

/**
 * Budgets from the project brief: WebGPU High holds 60 FPS / 16.7 ms of GPU time
 * at 1600x900 DPR 1, WebGL2 Low holds 30 FPS / 33.3 ms. Those two are the gates.
 * The remaining tiers are measured against their backend's budget too, but a
 * failure there is informational — `max` is deliberately allowed to cost more
 * than 60 FPS on mid-range hardware.
 */
const BUDGETS = {
  webgpu: { frameMs: 16.7, fps: 60 },
  webgl: { frameMs: 33.3, fps: 30 },
};

/**
 * Boot budgets are deliberately attached to the low-end configuration. The
 * memory value is the renderer's live resource count, not a guessed byte total;
 * it makes the allocation shape auditable on machines where VRAM telemetry is
 * unavailable to the page.
 */
const LOW_BOOT_BUDGET = {
  timeToFirstFrameMs: 15_000,
  peakRendererMemory: { geometries: 120, textures: 100 },
};

/** Minimum samples for a verdict. Percentiles over a handful of frames are noise. */
const MIN_SAMPLES = 200;

const MATRIX = [
  { id: 'webgpu-low', backend: 'webgpu', tier: 'low', gate: false },
  { id: 'webgpu-medium', backend: 'webgpu', tier: 'medium', gate: false },
  { id: 'webgpu-high', backend: 'webgpu', tier: 'high', gate: true },
  { id: 'webgpu-high-underwater', backend: 'webgpu', tier: 'high', gate: false, camera: UNDERWATER_CAMERA },
  { id: 'webgpu-ultra', backend: 'webgpu', tier: 'ultra', gate: false },
  { id: 'webgpu-max', backend: 'webgpu', tier: 'max', gate: false },
  { id: 'webgl-low', backend: 'webgl', tier: 'low', gate: true, bootBudget: LOW_BOOT_BUDGET },
  {
    id: 'webgl-low-cpu',
    backend: 'webgl',
    tier: 'low',
    gate: true,
    dpr: 1,
    cpuThrottlingRate: 4,
    bootBudget: LOW_BOOT_BUDGET,
  },
  { id: 'webgl-high', backend: 'webgl', tier: 'high', gate: false },
];

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const options = {
    width: 1600,
    height: 900,
    dpr: 1,
    warmup: 150,
    frames: 600,
    /** Wall-clock ceiling per configuration, so a slow tier cannot hang the run. */
    maxSeconds: 45,
    /** Length of the rAF-pacing observation that precedes each measurement. */
    pacingSeconds: 3,
    settleSeconds: 3,
    headless: false,
    keepOpen: false,
    url: null,
    only: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case '--width': options.width = Number(next()); break;
      case '--height': options.height = Number(next()); break;
      case '--dpr': options.dpr = Number(next()); break;
      case '--warmup': options.warmup = Number(next()); break;
      case '--frames': options.frames = Number(next()); break;
      case '--max-seconds': options.maxSeconds = Number(next()); break;
      case '--settle-seconds': options.settleSeconds = Number(next()); break;
      case '--headless': options.headless = true; break;
      case '--keep-open': options.keepOpen = true; break;
      case '--url': options.url = next(); break;
      case '--only': options.only = new Set(next().split(',').map((s) => s.trim())); break;
      case '--help':
        process.stdout.write(HELP);
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return options;
}

const HELP = `
web-ocean-3d benchmark

  npm run bench                          build, serve and measure the full matrix
  node scripts/benchmark.mjs [options]

  --only <ids>        comma-separated configuration ids (${MATRIX.map((c) => c.id).join(', ')})
  --width <n>         viewport width (default 1600)
  --height <n>        viewport height (default 900)
  --dpr <n>           device pixel ratio (default 1)
  --frames <n>        samples per configuration after warmup (default 600)
  --warmup <n>        discarded settling frames (default 150)
  --max-seconds <n>   wall-clock ceiling per configuration (default 45)
  --settle-seconds <n> pause after a tier change before warmup (default 3)
  --url <url>         measure an already-running server instead of starting one
  --headless          diagnostics only; headless runs are reported UNVERIFIED
  --keep-open         leave the browser open after the run
`;

// ---------------------------------------------------------------------------
// Preview server
// ---------------------------------------------------------------------------

/** Asks the OS for a free port rather than guessing, so a stale 4173 cannot be measured by mistake. */
function findFreePort() {
  return new Promise((resolvePort, reject) => {
    const probe = createServer();
    probe.unref();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolvePort(port));
    });
  });
}

async function waitForServer(url, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { redirect: 'follow' });
      if (response.ok) return;
    } catch {
      // Not up yet.
    }
    await sleep(250);
  }
  throw new Error(`preview server did not respond at ${url}`);
}

/**
 * Starts `vite preview` on a port nobody else holds.
 *
 * `--strictPort` is deliberate: vite's default behaviour is to walk to the next
 * free port and print it, which during development of this script silently
 * benchmarked a *different* server that another process had left on 4173, with a
 * stale `dist/` whose model assets 404'd. Failing loudly is better.
 */
async function startPreview() {
  const port = await findFreePort();
  const url = `http://127.0.0.1:${port}/`;
  const child = spawn(
    process.execPath,
    [join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js'), 'preview',
      '--port', String(port), '--strictPort', '--host', '127.0.0.1'],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  let log = '';
  child.stdout.on('data', (chunk) => { log += chunk; });
  child.stderr.on('data', (chunk) => { log += chunk; });
  child.on('exit', (code) => {
    if (code !== 0 && code !== null) process.stderr.write(`vite preview exited ${code}\n${log}\n`);
  });
  await waitForServer(url);
  return { url, stop: () => child.kill() };
}

// ---------------------------------------------------------------------------
// Host description
// ---------------------------------------------------------------------------

/**
 * Adapters as the *operating system* sees them.
 *
 * Recorded for provenance only, and labelled as such in the output. This machine
 * has two GPUs; which one Chrome actually bound is a question only the browser
 * can answer, and that answer comes from `collectEnvironment` below.
 */
async function osGpuInfo() {
  if (process.platform !== 'win32') return null;
  try {
    const { stdout } = await execFileAsync('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-Command',
      'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,AdapterRAM | ConvertTo-Json -Compress',
    ], { timeout: 30_000 });
    const parsed = JSON.parse(stdout);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return null;
  }
}

function hostInfo() {
  const cpus = os.cpus();
  return {
    platform: process.platform,
    release: os.release(),
    arch: process.arch,
    cpu: cpus.length > 0 ? `${cpus[0].model} x${cpus.length}` : 'unknown',
    totalMemoryBytes: os.totalmem(),
    node: process.version,
  };
}

// ---------------------------------------------------------------------------
// In-page work
// ---------------------------------------------------------------------------

/**
 * Everything the harness needs to know about the browser it is talking to,
 * gathered from inside the page.
 *
 * The adapter is re-requested here rather than read from the OS: two GPUs are
 * present on the development machine and only the browser knows which one it
 * bound. `WEBGL_debug_renderer_info` is queried as well because Chrome blanks
 * `adapter.info.device` and `.description` for fingerprinting reasons, and the
 * ANGLE renderer string still names the physical part.
 */
async function collectEnvironment(page) {
  return page.evaluate(async () => {
    const result = {
      focused: document.hasFocus(),
      visibility: document.visibilityState,
      devicePixelRatio: window.devicePixelRatio,
      userAgent: navigator.userAgent,
      adapter: null,
      gl: null,
    };

    try {
      const adapter = navigator.gpu
        ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
        : null;
      if (adapter) {
        const info = adapter.info ?? {};
        result.adapter = {
          vendor: info.vendor ?? '',
          architecture: info.architecture ?? '',
          device: info.device ?? '',
          description: info.description ?? '',
          isFallbackAdapter: adapter.isFallbackAdapter === true,
          features: [...adapter.features].sort(),
          hasTimestampQuery: adapter.features.has('timestamp-query'),
        };
      }
    } catch (error) {
      result.adapterError = String(error);
    }

    // A throwaway context, released immediately — the app owns the one that matters.
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        result.gl = {
          version: gl.getParameter(gl.VERSION),
          vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : null,
          renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : null,
          hasDisjointTimerQuery: gl.getExtension('EXT_disjoint_timer_query_webgl2') !== null,
        };
        gl.getExtension('WEBGL_lose_context')?.loseContext();
      }
    } catch (error) {
      result.glError = String(error);
    }

    return result;
  });
}

/**
 * Serves glTF binary payloads out of `dist/` instead of over HTTP.
 *
 * Measured on the development machine, and the reason this exists: a *headed*
 * Chrome on this host receives `HTTP 204, zero bytes` for every `.bin` under the
 * preview server, while `curl` and the same Chrome build in headless mode both
 * receive the full 200 response from the same server on the same port. Something
 * outside the browser is eating `application/octet-stream` on visible sessions.
 * The visible consequence is that `GLTFLoader` reports `Failed to load buffer`,
 * the ship and every prop are absent, and the benchmark quietly measures an
 * empty ocean.
 *
 * A benchmark that silently drops the hero object is worse than no benchmark, so
 * the harness fulfils these requests from the build output itself. Only paths
 * that actually exist in `dist/` are intercepted — a genuine 404 still 404s —
 * and every interception is recorded in the result file.
 */
function serveBinariesFromDisk(page, served) {
  return page.route('**/*.bin', async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const file = join(ROOT, 'dist', decodeURIComponent(pathname).replace(/^\/+/, ''));
    if (!existsSync(file) || !statSync(file).isFile()) return route.continue();
    served.push(pathname);
    return route.fulfill({
      status: 200,
      contentType: 'application/octet-stream',
      body: readFileSync(file),
    });
  });
}

/**
 * The measurement itself, run inside the page.
 *
 * Two measurement phases with the world pinned between them, because they answer
 * different questions.
 *
 * **Phase A** watches the application's own rAF-driven loop and records the
 * interval between delivered frames. That is the only way to find out whether
 * the browser is pacing the page — a number that says nothing about the renderer
 * but everything about whether the renderer's numbers can be trusted. It has to
 * come first, because it is the only phase in which the app owns its own clock.
 *
 * **Phase B** pauses the app loop and drives one frame per rAF tick through
 * `Loop.step`, which runs the identical update and render path and *awaits* the
 * render. Owning the frame boundary is what makes the rest possible: the draw
 * counters can be read before three.js resets them, and the frame's timestamp
 * queries can be resolved knowing that the pool contains that frame's passes and
 * nothing else. The pool holds 2048 queries and this scene issues ~220 per
 * frame, so resolving every frame is required regardless.
 *
 * Two deliberate distortions, both in the direction of a *cleaner* number than
 * the app really achieves, and both recorded in the output:
 *
 *  - Frames are stepped at a fixed 1/60 s rather than at wall clock, so the
 *    simulation advances identically in every configuration.
 *  - `resolveTimestampsAsync` maps a buffer back every frame, which drains the
 *    pipeline between frames. GPU time is measured on the GPU and is unaffected;
 *    CPU frame time loses whatever overlap it would normally get with the
 *    previous frame's GPU work.
 */
async function measureInPage(page, options) {
  return page.evaluate(async (opts) => {
    const ocean = window.__ocean;
    const renderer = ocean.renderer;

    const raf = () => new Promise((done) => requestAnimationFrame(done));

    // --- Phase A: is the browser pacing us? --------------------------------
    const intervals = [];
    const loopWork = [];
    let previous = performance.now();
    const pacingDeadline = previous + opts.pacingSeconds * 1000;
    while (performance.now() < pacingDeadline) {
      await raf();
      const now = performance.now();
      intervals.push(now - previous);
      previous = now;
      loopWork.push(ocean.loop.stats.frameMs);
    }
    const observedFps = ocean.loop.stats.fps;

    // --- Pin the world -----------------------------------------------------
    // Without this the sampled scene is wherever the session happened to drift
    // to. Measured: two runs of the Max tier differed by 505 000 triangles and
    // 2.4 ms of GPU time, because the ship had wandered far enough between them
    // to change what the 4096-square shadow frustum contained. `resetDeterministic`
    // rewinds every clock and accumulation buffer and settles the world at a
    // fixed simulation time, so from here the frame sequence is a function of
    // the step count alone.
    const reset = { applied: false, time: 0, settleSteps: 90, error: null };
    try {
      await ocean.resetDeterministic(reset.time, reset.settleSteps);
      reset.applied = true;
    } catch (error) {
      reset.error = String(error);
    }

    // --- Timestamp queries -------------------------------------------------
    // Enabled after the reset, not before: the settle run would otherwise queue
    // ~10 000 unresolved queries into a 2048-entry pool.
    // `trackTimestamp` lives on the backend, not on the renderer: assigning
    // `renderer.trackTimestamp` creates a stray property and changes nothing.
    // The backend gates it once, at init, behind `hasFeature('timestamp-query')`,
    // and the device was already created with every adapter feature — so setting
    // it now is enough, and no source change is required.
    const backend = renderer.backend;
    const timing = {
      rendererPropertyExists: 'trackTimestamp' in renderer,
      backendPropertyExists: 'trackTimestamp' in backend,
      enabledAtConstruction: backend.trackTimestamp === true,
      deviceHasFeature: (() => {
        try { return renderer.hasFeature('timestamp-query'); } catch { return false; }
      })(),
      enabled: false,
      method: null,
      error: null,
    };
    try {
      backend.trackTimestamp = true;
      timing.enabled = backend.trackTimestamp === true;
    } catch (error) {
      timing.error = String(error);
    }

    // --- Phase B: harness-driven frames ------------------------------------
    // Already paused by `resetDeterministic`, but not if that threw.
    ocean.setPaused(true);

    // AdaptiveQuality reads `loop.stats.fps`, which `Loop.step` never writes, so
    // it would otherwise still be holding whatever the rAF loop last recorded —
    // and on a slow configuration that is low enough to trigger a tier
    // downgrade *during* the sample. Pin the tier by pinning the number it
    // watches; the run asserts afterwards that the tier did not move.
    ocean.loop.stats.fps = 1000;

    // Drain anything queued between enabling tracking and the first sampled
    // frame, so sample zero resolves its own passes and not somebody else's.
    try { await renderer.resolveTimestampsAsync('render'); } catch { /* not available */ }

    const cpuSamples = [];
    const gpuSamples = [];
    let gpuMisses = 0;
    let counters = null;
    let truncated = false;

    const total = opts.warmup + opts.frames;
    const deadline = performance.now() + opts.maxSeconds * 1000;
    for (let i = 0; i < total; i++) {
      if (performance.now() > deadline) { truncated = true; break; }
      await raf();

      const started = performance.now();
      await ocean.loop.step(opts.dt, 1);
      const cpuMs = performance.now() - started;

      // Read synchronously: three.js resets these counters from its own rAF
      // callback, so anything after the next await belongs to the next frame.
      const render = renderer.info.render;
      const frameCounters = {
        drawCalls: render.drawCalls,
        triangles: render.triangles,
        frameCalls: render.frameCalls,
      };

      let gpuMs = null;
      if (timing.enabled) {
        try {
          const value = await renderer.resolveTimestampsAsync('render');
          if (typeof value === 'number' && Number.isFinite(value) && value > 0) gpuMs = value;
        } catch (error) {
          timing.error = timing.error ?? String(error);
        }
      }

      if (i < opts.warmup) continue;
      cpuSamples.push(cpuMs);
      if (gpuMs === null) gpuMisses++; else gpuSamples.push(gpuMs);
      counters = frameCounters;
    }

    if (timing.enabled && gpuSamples.length === 0) {
      timing.enabled = false;
      timing.error = timing.error ?? 'timestamp resolution never returned a usable duration';
    } else if (timing.enabled) {
      timing.method = ocean.backend === 'webgpu'
        ? 'WebGPU timestamp-query, summed over the frame (THREE.TimestampQuery.RENDER)'
        : 'WebGL2 EXT_disjoint_timer_query_webgl2, summed over the frame';
    }

    const memory = { ...renderer.info.memory };
    const state = ocean.getState();

    ocean.setPaused(false);

    return {
      backend: ocean.backend,
      state,
      pacing: { intervals, loopWork, observedFps },
      reset,
      elapsedTime: ocean.elapsedTime(),
      timing,
      cpuSamples,
      gpuSamples,
      gpuMisses,
      counters,
      memory,
      truncated,
      hasShip: ocean.hasShip(),
      focused: document.hasFocus(),
      visibility: document.visibilityState,
      drawingBuffer: {
        width: renderer.domElement.width,
        height: renderer.domElement.height,
        pixelRatio: renderer.getPixelRatio(),
      },
    };
  }, options);
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

function percentile(sorted, fraction) {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(fraction * sorted.length) - 1));
  return sorted[index];
}

/** Median-centred, because a single compositor hitch moves a mean and tells you nothing. */
function summarise(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  return {
    samples: sorted.length,
    min: round(sorted[0]),
    p50: round(percentile(sorted, 0.5)),
    p90: round(percentile(sorted, 0.9)),
    p95: round(percentile(sorted, 0.95)),
    p99: round(percentile(sorted, 0.99)),
    max: round(sorted[sorted.length - 1]),
    mean: round(mean),
  };
}

const round = (value) => (value === null ? null : Math.round(value * 1000) / 1000);

/**
 * Classifies how the browser paced the application's own rAF loop.
 *
 * `workMs` must be the *measured* frame cost from phase B, not the app's own
 * `loop.stats.frameMs`. The app times its render call, and on WebGPU that call
 * returns once the work is submitted — it reads 0.2 ms for a frame that costs
 * 2.4 ms on the GPU. Comparing a delivered interval against that number
 * classifies a perfectly healthy 190 FPS run as throttled.
 *
 * `throttled` is the failure this project has actually been bitten by: the
 * delivered interval bears no relation to the work because the browser decided
 * the page does not deserve frames — 1.1 FPS against 0.8 ms of work.
 * `display-limited` is honest vsync pacing; it caps the delivered rate but says
 * nothing about how long a frame took, so it does not invalidate a GPU-time
 * measurement, only a claim about headroom above the refresh rate.
 */
function classifyPacing(intervals, appLoopWork, workMs) {
  const interval = summarise(intervals);
  const appLoop = summarise(appLoopWork);
  if (!interval) return { classification: 'unknown', deliveredFps: null };
  const deliveredFps = interval.p50 > 0 ? 1000 / interval.p50 : 0;
  const reference = Math.max(workMs ?? 0, 0.1);

  let classification = 'unthrottled';
  if (deliveredFps < 50 && interval.p50 > reference * 4) classification = 'throttled';
  else if (deliveredFps <= 70 && interval.p50 > reference * 2) classification = 'display-limited';

  return {
    classification,
    deliveredFps: round(deliveredFps),
    referenceFrameMs: round(reference),
    intervalMs: interval,
    /** The app's own `Loop.stats.frameMs` — submit cost, kept for contrast. */
    appLoopFrameMs: appLoop,
  };
}

const SOFTWARE_MARKERS = ['swiftshader', 'llvmpipe', 'lavapipe', 'basic render', 'microsoft basic', 'software'];

function looksLikeSoftware(...strings) {
  const haystack = strings.filter(Boolean).join(' ').toLowerCase();
  return SOFTWARE_MARKERS.some((marker) => haystack.includes(marker));
}

/**
 * PASS requires a measurement. Anything that makes the measurement untrustworthy
 * produces UNVERIFIED with the reason attached — never a pass, and never a
 * failure either, because a harness problem is not a regression.
 */
function verdictFor(config, measurement, environment, options) {
  const reasons = [];
  const consoleErrors = measurement.consoleErrors ?? [];
  const budget = BUDGETS[config.backend];

  if (options.headless) reasons.push('headless run — rAF pacing and GPU scheduling are not representative');
  if (!measurement.focused) reasons.push('browser window was not focused');
  if (measurement.visibility !== 'visible') reasons.push(`page visibility was "${measurement.visibility}"`);
  if (measurement.pacing.classification === 'throttled') {
    reasons.push(`browser throttled requestAnimationFrame to ${measurement.pacing.deliveredFps} FPS`);
  }

  if (config.backend === 'webgpu') {
    if (!environment.adapter) reasons.push('no WebGPU adapter');
    else if (environment.adapter.isFallbackAdapter) reasons.push('WebGPU reported a fallback adapter');
    else if (looksLikeSoftware(environment.adapter.vendor, environment.adapter.architecture,
      environment.adapter.device, environment.adapter.description)) {
      reasons.push('WebGPU adapter looks like a software rasteriser');
    }
  } else if (looksLikeSoftware(environment.gl?.renderer, environment.gl?.vendor)) {
    reasons.push('WebGL2 is backed by a software rasteriser');
  }

  if (measurement.backend !== config.backend) {
    reasons.push(`requested the ${config.backend} backend but the app booted ${measurement.backend}`);
  }
  if (measurement.state.quality !== config.tier) {
    reasons.push(`quality tier drifted to "${measurement.state.quality}" during the sample`);
  }
  if (!measurement.timing.enabled) {
    reasons.push(`GPU timestamp queries unavailable${measurement.timing.error ? ` (${measurement.timing.error})` : ''}`);
  }
  if (measurement.cpuSamples.length < MIN_SAMPLES) {
    reasons.push(
      `only ${measurement.cpuSamples.length} CPU samples, need ${MIN_SAMPLES} for percentiles`,
    );
  }
  // The GPU count is the one that matters, and it was never checked.
  //
  // The verdict is computed from `summarise(measurement.gpuSamples)`, but only
  // the CPU sample count was gated -- so a run where timestamp resolution
  // silently failed for all but a handful of frames would report a p50 over five
  // samples and PASS on it. `gpuMisses` was recorded in the artefact and read by
  // nobody.
  if (measurement.gpuSamples.length < MIN_SAMPLES) {
    reasons.push(
      `only ${measurement.gpuSamples.length} GPU samples (${measurement.gpuMisses} frames ` +
        `returned no timestamp), need ${MIN_SAMPLES} for percentiles`,
    );
  }
  // A frame cost measured without the hero object in it is not this project's
  // frame cost, however clean the distribution looks.
  if (!measurement.hasShip) reasons.push('scene content incomplete: the ship did not load');
  if (!measurement.reset.applied) {
    reasons.push(`world state was not pinned (${measurement.reset.error}) — the sample is not reproducible`);
  }
  // Console errors were collected, written into the artefact, and never allowed
  // to affect the verdict. A configuration that threw on every frame could
  // therefore report a clean PASS -- and a frame that fails early is a *fast*
  // frame, so the error made the number look better rather than worse.
  if (consoleErrors.length > 0) {
    const shown = consoleErrors.slice(0, 3).join(' | ');
    const more = consoleErrors.length > 3 ? ` (+${consoleErrors.length - 3} more)` : '';
    reasons.push(`${consoleErrors.length} console error(s) during the sample: ${shown}${more}`);
  }

  if (reasons.length > 0) return { verdict: 'UNVERIFIED', reasons, budget };

  const gpu = summarise(measurement.gpuSamples);
  const exceeded = [];
  if (gpu.p50 > budget.frameMs) {
    exceeded.push(`GPU p50 ${gpu.p50} ms exceeds the ${budget.frameMs} ms budget`);
  }
  if (gpu.p95 > budget.frameMs) {
    exceeded.push(`GPU p95 ${gpu.p95} ms exceeds the ${budget.frameMs} ms budget`);
  }
  if (gpu.p99 > budget.frameMs) {
    exceeded.push(`GPU p99 ${gpu.p99} ms exceeds the ${budget.frameMs} ms budget`);
  }

  if (config.bootBudget) {
    const boot = measurement.boot;
    if (!boot || boot.firstFrameMs === null) {
      exceeded.push('time-to-first-frame was not measured');
    } else if (boot.firstFrameMs > config.bootBudget.timeToFirstFrameMs) {
      exceeded.push(
        `time-to-first-frame ${round(boot.firstFrameMs)} ms exceeds the ` +
          `${config.bootBudget.timeToFirstFrameMs} ms budget`,
      );
    }
    if (!boot || boot.peakRendererMemory.geometries > config.bootBudget.peakRendererMemory.geometries) {
      exceeded.push(
        `peak geometries ${boot?.peakRendererMemory.geometries ?? 'n/a'} exceeds ` +
          `${config.bootBudget.peakRendererMemory.geometries}`,
      );
    }
    if (!boot || boot.peakRendererMemory.textures > config.bootBudget.peakRendererMemory.textures) {
      exceeded.push(
        `peak textures ${boot?.peakRendererMemory.textures ?? 'n/a'} exceeds ` +
          `${config.bootBudget.peakRendererMemory.textures}`,
      );
    }
  }
  if (exceeded.length > 0) return { verdict: 'FAIL', reasons: exceeded, budget };
  return { verdict: 'PASS', reasons, budget };
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

async function runConfiguration(browser, baseUrl, config, options) {
  const context = await browser.newContext({
    viewport: { width: options.width, height: options.height },
    deviceScaleFactor: config.dpr ?? options.dpr,
  });
  const page = await context.newPage();
  const consoleErrors = [];
  const servedFromDisk = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text().slice(0, 400));
  });
  page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.message}`));
  await serveBinariesFromDisk(page, servedFromDisk);

  try {
    if (config.cpuThrottlingRate) {
      const client = await context.newCDPSession(page);
      await client.send('Emulation.setCPUThrottlingRate', { rate: config.cpuThrottlingRate });
    }

    await page.addInitScript(() => {
      window.__benchmarkBoot = {
        firstFrameMs: null,
        peakRendererMemory: { geometries: 0, textures: 0 },
        startedAt: performance.now(),
      };
      const observe = () => {
        const boot = window.__benchmarkBoot;
        const ocean = window.__ocean;
        if (ocean) {
          const memory = ocean.renderer.info.memory;
          boot.peakRendererMemory.geometries = Math.max(
            boot.peakRendererMemory.geometries,
            memory.geometries,
          );
          boot.peakRendererMemory.textures = Math.max(
            boot.peakRendererMemory.textures,
            memory.textures,
          );
          if (boot.firstFrameMs === null && ocean.loop.stats.frameMs > 0) {
            boot.firstFrameMs = performance.now() - boot.startedAt;
          }
        }
        if (!ocean || boot.firstFrameMs === null || !ocean.sceneContentLoaded()) {
          window.setTimeout(observe, 50);
        }
      };
      observe();
    });

    // A fresh page per configuration rather than switching tiers in one session:
    // tier changes rebuild the FFT targets, the surface material and the ocean
    // geometry, and measuring the fifth rebuild of a session is measuring
    // something the user never sees.
    const launchUrl = new URL(baseUrl);
    launchUrl.searchParams.set('gate', '0');
    launchUrl.searchParams.set('quality', config.tier);
    if (config.backend === 'webgl') launchUrl.searchParams.set('webgl', '1');
    const url = launchUrl.toString();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.bringToFront();

    await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 120_000 });
    await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 180_000 });
    await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 180_000 });
    await page.waitForFunction(() => window.__ocean.sceneContentLoaded(), undefined, { timeout: 180_000 });

    await page.waitForFunction(
      () => window.__benchmarkBoot && window.__benchmarkBoot.firstFrameMs !== null,
      undefined,
      { timeout: 180_000 },
    );
    const boot = await page.evaluate(() => ({
      firstFrameMs: window.__benchmarkBoot.firstFrameMs,
      peakRendererMemory: { ...window.__benchmarkBoot.peakRendererMemory },
    }));

    const environment = await collectEnvironment(page);

    const camera = config.camera ?? CAMERA;
    await page.evaluate(async ([preset, camera]) => {
      const ocean = window.__ocean;
      ocean.setState({ preset, pixelRatio: 1, cameraMode: 'orbit' });
      ocean.setCamera(...camera.position, ...camera.target);
      // A tier change rebuilds the water material and the ocean mesh, so the
      // pipelines prewarmed at boot no longer cover the scene. Compile the new
      // ones here instead of letting the first sampled frame pay for them.
      await ocean.renderer.compileAsync(ocean.scene, ocean.camera);
    }, [PRESET, camera]);

    // Settle: mip chains, the first sampler readback after the rebuild, and the
    // buoyancy solver finding the new wave field.
    await page.waitForTimeout(options.settleSeconds * 1000);

    const raw = await measureInPage(page, {
      warmup: options.warmup,
      frames: options.frames,
      maxSeconds: options.maxSeconds,
      pacingSeconds: options.pacingSeconds,
      dt: 1 / 60,
    });

    const cpu = summarise(raw.cpuSamples);
    const gpu = summarise(raw.gpuSamples);
    const pacing = classifyPacing(
      raw.pacing.intervals,
      raw.pacing.loopWork,
      Math.max(gpu?.p50 ?? 0, cpu?.p50 ?? 0),
    );

    const { verdict, reasons, budget } = verdictFor(
      config, { ...raw, pacing, consoleErrors }, environment, options,
    );

    return {
      id: config.id,
      gate: config.gate,
      requestedBackend: config.backend,
      backend: raw.backend,
      tier: config.tier,
      preset: PRESET,
      camera,
      resolution: { width: options.width, height: options.height, dpr: config.dpr ?? options.dpr },
      drawingBuffer: raw.drawingBuffer,
      adapter: environment.adapter,
      gl: environment.gl,
      focus: { hasFocus: raw.focused, visibility: raw.visibility },
      gpuTiming: raw.timing,
      pacing,
      sampling: {
        warmupFrames: options.warmup,
        requestedFrames: options.frames,
        samples: raw.cpuSamples.length,
        gpuSamples: raw.gpuSamples.length,
        gpuMisses: raw.gpuMisses,
        truncated: raw.truncated,
        stepSeconds: 1 / 60,
        deterministicReset: raw.reset,
        simulationTimeAtEnd: round(raw.elapsedTime),
      },
      cpuFrameMs: cpu,
      gpuFrameMs: gpu,
      fps: {
        deliveredP50: pacing.deliveredFps,
        impliedByGpuP50: gpu ? round(1000 / gpu.p50) : null,
        impliedByCpuP50: cpu ? round(1000 / cpu.p50) : null,
      },
      render: raw.counters,
      memory: raw.memory,
      boot,
      cpuThrottlingRate: config.cpuThrottlingRate ?? 1,
      sceneContent: {
        hasShip: raw.hasShip,
        submerged: Boolean(config.camera),
        binariesServedFromDisk: servedFromDisk,
      },
      budget,
      bootBudget: config.bootBudget ?? null,
      verdict,
      reasons,
      consoleErrors,
    };
  } finally {
    if (!options.keepOpen) await context.close();
  }
}

function formatTable(results) {
  const header = ['configuration', 'backend', 'TTFF', 'GPU p50', 'p95', 'p99', 'CPU p50', 'FPS(GPU)', 'delivered', 'draws', 'tris', 'verdict'];
  const rows = results.map((r) => [
    r.id + (r.gate ? ' *' : ''),
    r.backend,
    r.boot?.firstFrameMs !== null && r.boot?.firstFrameMs !== undefined
      ? `${r.boot.firstFrameMs.toFixed(0)} ms`
      : 'n/a',
    r.gpuFrameMs ? `${r.gpuFrameMs.p50.toFixed(2)} ms` : 'n/a',
    r.gpuFrameMs ? `${r.gpuFrameMs.p95.toFixed(2)}` : 'n/a',
    r.gpuFrameMs ? `${r.gpuFrameMs.p99.toFixed(2)}` : 'n/a',
    r.cpuFrameMs ? `${r.cpuFrameMs.p50.toFixed(2)} ms` : 'n/a',
    r.fps.impliedByGpuP50 !== null ? r.fps.impliedByGpuP50.toFixed(0) : 'n/a',
    r.fps.deliveredP50 !== null ? r.fps.deliveredP50.toFixed(0) : 'n/a',
    r.render ? String(r.render.drawCalls) : 'n/a',
    r.render ? r.render.triangles.toLocaleString('en-US') : 'n/a',
    r.verdict,
  ]);
  const widths = header.map((_, column) =>
    Math.max(header[column].length, ...rows.map((row) => row[column].length)));
  const line = (cells) => cells.map((cell, i) => cell.padEnd(widths[i])).join('  ');
  return [line(header), widths.map((w) => '-'.repeat(w)).join('  '), ...rows.map(line)].join('\n');
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const configs = MATRIX.filter((c) => !options.only || options.only.has(c.id));
  if (configs.length === 0) throw new Error('--only matched no configurations');

  if (!existsSync(join(ROOT, 'dist', 'index.html'))) {
    throw new Error('dist/ is missing — run `npm run build` first, or use `npm run bench`');
  }

  const server = options.url ? null : await startPreview();
  const baseUrl = options.url ?? server.url;

  // The real Chrome build is preferred over Playwright's bundled Chromium: the
  // bundled build is frequently shipped without a working GPU sandbox on Windows
  // and falls back to a software adapter, which this harness would correctly
  // refuse to report a pass for.
  let browser;
  let channel = 'chrome';
  try {
    browser = await chromium.launch({ headless: options.headless, channel, args: CHROME_FLAGS });
  } catch {
    channel = 'bundled-chromium';
    browser = await chromium.launch({ headless: options.headless, args: CHROME_FLAGS });
  }
  // Read before the run: `version()` is not available once the browser is closed.
  const browserVersion = browser.version();

  const startedAt = new Date();
  const results = [];
  try {
    for (const config of configs) {
      process.stdout.write(`\n[bench] ${config.id} …\n`);
      const result = await runConfiguration(browser, baseUrl, config, options);
      results.push(result);
      process.stdout.write(
        `[bench] ${config.id}: ${result.verdict}` +
        `${result.gpuFrameMs ? ` — GPU p50 ${result.gpuFrameMs.p50.toFixed(2)} ms` : ''}` +
        `${result.reasons.length ? ` (${result.reasons.join('; ')})` : ''}\n`,
      );
    }
  } finally {
    if (!options.keepOpen) await browser.close();
    server?.stop();
  }

  const finishedAt = new Date();
  const report = {
    schema: SCHEMA,
    startedAt: startedAt.toISOString(),
    finishedAt: finishedAt.toISOString(),
    durationMs: finishedAt - startedAt,
    command: 'npm run bench',
    argv: process.argv.slice(2),
    host: hostInfo(),
    osReportedGpus: await osGpuInfo(),
    browser: { name: 'chromium', channel, version: browserVersion, flags: CHROME_FLAGS, headless: options.headless },
    budgets: BUDGETS,
    bootBudgets: { low: LOW_BOOT_BUDGET },
    minSamples: MIN_SAMPLES,
    configurations: results,
    summary: {
      pass: results.filter((r) => r.verdict === 'PASS').length,
      fail: results.filter((r) => r.verdict === 'FAIL').length,
      unverified: results.filter((r) => r.verdict === 'UNVERIFIED').length,
      gates: results.filter((r) => r.gate).map((r) => ({ id: r.id, verdict: r.verdict, reasons: r.reasons })),
    },
  };

  mkdirSync(RESULTS_DIR, { recursive: true });
  const stamp = startedAt.toISOString().replace(/[:.]/g, '-');
  const file = join(RESULTS_DIR, `bench-${stamp}.json`);
  writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(join(RESULTS_DIR, 'latest.json'), `${JSON.stringify(report, null, 2)}\n`);

  process.stdout.write(`\n${formatTable(results)}\n\n`);
  process.stdout.write(`* gated configuration. GPU p50/p95/p99 budgets: WebGPU ${BUDGETS.webgpu.frameMs} ms, WebGL2 ${BUDGETS.webgl.frameMs} ms. Low boot budget: TTFF ${LOW_BOOT_BUDGET.timeToFirstFrameMs} ms, ${LOW_BOOT_BUDGET.peakRendererMemory.geometries} geometries, ${LOW_BOOT_BUDGET.peakRendererMemory.textures} textures.\n`);
  process.stdout.write(`Results: ${file}\n`);

  const gatesPassed = report.summary.gates.every((gate) => gate.verdict === 'PASS');
  if (!gatesPassed) {
    process.stdout.write('\nNot every gated configuration passed. Gate status:\n');
    for (const gate of report.summary.gates) {
      process.stdout.write(`  ${gate.id}: ${gate.verdict}${gate.reasons.length ? ` — ${gate.reasons.join('; ')}` : ''}\n`);
    }
  }
  process.exit(gatesPassed ? 0 : 1);
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
});
