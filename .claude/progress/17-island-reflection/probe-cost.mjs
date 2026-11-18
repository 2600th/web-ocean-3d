#!/usr/bin/env node
/**
 * GPU frame time in the Tour's beach-flyover view, for an A/B of a shader
 * change: `node probe-cost.mjs <url> [--tier high] [--frames 240]`.
 * The capture script's Chrome flags (Vulkan did not start headless here) plus
 * profile-frame.mjs's timestamp flags, against a running dev server, so it
 * builds nothing. Prints p50 and mean of the summed render-pass GPU time
 * (`--fence` for the fenced wall clock instead).
 */
import { chromium } from '@playwright/test';

const FLAGS = [
  '--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc,timestamp_quantization',
  '--disable-frame-rate-limit', '--disable-gpu-vsync', '--enable-dawn-features=allow_unsafe_apis',
];
const metric = process.argv.includes('--fence') ? 'fence' : 'timestamp';
const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith('--'));
const opt = (name, d) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : d; };
const tier = opt('--tier', 'high');
const frames = Number(opt('--frames', 240));
const shotT = Number(opt('--t', 57.5));

const browser = await chromium.launch({ headless: true, args: FLAGS });
try {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const u = new URL(url); u.searchParams.set('gate', '0'); u.searchParams.set('quality', tier);
  await page.goto(u.toString(), { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__ocean?.isReady() && window.__ocean.shadersReady(), undefined, { timeout: 300000 });
  const res = await page.evaluate(async ({ tier, frames, shotT, metric }) => {
    const o = window.__ocean;
    o.downgrade && (o.downgrade.enabled = false);
    await o.setState({ cameraMode: 'cinematic', windSpeed: 15, peakWavelength: 47, cloudCoverage: 0.32, preset: 'skyPro', quality: tier, pixelRatio: 1 });
    await o.resetDeterministic(30 + shotT, 90, null, 0.5 + shotT);
    const r = o.renderer, device = r.backend.device;
    if (metric === 'timestamp') { r.backend.trackTimestamp = true; try { await r.resolveTimestampsAsync('render'); } catch {} }
    const vals = [];
    // 40 warm-up frames, then the measured ones: about two seconds of the shot.
    for (let i = 0; i < 40 + frames; i++) {
      const t0 = performance.now();
      await o.step(1 / 60, 1);
      let v = null;
      if (metric === 'fence') { await device.queue.onSubmittedWorkDone(); v = performance.now() - t0; }
      else { try { v = await r.resolveTimestampsAsync('render'); } catch {} }
      if (i >= 40 && typeof v === 'number' && v > 0) vals.push(v);
    }
    vals.sort((a, b) => a - b);
    return { n: vals.length, p50: vals[vals.length >> 1], mean: vals.reduce((s, v) => s + v, 0) / vals.length };
  }, { tier, frames, shotT, metric });
  console.log(JSON.stringify({ tier, metric, ...res }));
} finally {
  await browser.close();
}
