#!/usr/bin/env node
/**
 * Stills and short bursts of the Tour's island-reveal and beach-flyover shots,
 * with reflection terms toggled one at a time. Adapted from
 * promo/tools/capture.mjs (stepped simulation, capturePixels, ffmpeg to PNG).
 *
 *   node capture-island.mjs <url> --out <dir> [--tier ultra] [--shots a,b]
 *        [--toggles base,nossr,...] [--burst 0]
 */
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

const FLAGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'];
const STEP = 1 / 60;
const TOUR = { cameraMode: 'cinematic', windSpeed: 15, peakWavelength: 47, cloudCoverage: 0.32, preset: 'skyPro' };

// t is the time into the Tour video (capture.mjs starts the Tour at
// cinematicTime 0.5 after a reset to time 30).
const SHOTS = {
  reveal: { kind: 'tour', t: 46.5 },
  beach: { kind: 'tour', t: 57.5 },
  beach2: { kind: 'tour', t: 51.5 },
  sunset: {
    kind: 'boat',
    state: { cameraMode: 'boat', windSpeed: 6, peakWavelength: 40, cloudCoverage: 0.35, preset: 'sunset' },
    time: 31.25, lens: 30, camera: { position: [70, 3, 44], target: [-4, 11.6, 0] },
  },
  moonlit: {
    kind: 'boat',
    state: { cameraMode: 'boat', windSpeed: 9, peakWavelength: 60, cloudCoverage: 0.25, preset: 'moonlit' },
    time: 18, lens: 36, camera: { position: [-34, 3.2, 40], target: [9.4, 12, 6.7] },
  },
};

// Each toggle runs in the page after the shot is set up.
const TOGGLES = {
  base: '() => {}',
  nossr: '() => window.__ocean.ssr.setQuality(0)',
  nomirror: '() => window.__ocean.water.setReflection(0, 0.09)',
  none: '() => { window.__ocean.ssr.setQuality(0); window.__ocean.water.setReflection(0, 0.09); }',
  nodistort: '() => window.__ocean.water.setReflection(1, 0)',
  norefr: '() => window.__ocean.water.setRefraction(0)',
  nofoam: '() => window.__ocean.setFoamOverride(0)',
  noshade: '() => window.__ocean.setState({ cloudCoverage: 0 })',
};
const UNDO = '() => window.__ocean.setFoamOverride(null)';

function parseArgs(argv) {
  const o = { url: null, out: null, tier: 'ultra', shots: ['reveal', 'beach'], toggles: ['base'], burst: 0, width: 1920, height: 1080 };
  const r = argv.slice(2);
  for (let i = 0; i < r.length; i++) {
    const a = r[i];
    if (a === '--out') o.out = resolve(r[++i]);
    else if (a === '--tier') o.tier = r[++i];
    else if (a === '--shots') o.shots = r[++i].split(',');
    else if (a === '--toggles') o.toggles = r[++i].split(',');
    else if (a === '--burst') o.burst = +r[++i];
    else if (a === '--size') { const m = /^(\d+)x(\d+)$/.exec(r[++i]); o.width = +m[1]; o.height = +m[2]; }
    else if (!a.startsWith('--')) o.url = a;
    else throw new Error(`unknown ${a}`);
  }
  return o;
}

async function boot(page, url, tier) {
  const target = new URL(url);
  target.searchParams.set('gate', '0');
  target.searchParams.set('quality', tier);
  await page.goto(target.toString(), { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 180_000 });
  await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 300_000 });
  await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 300_000 });
}

async function setupShot(page, shot, tier, toggle) {
  await page.evaluate(() => window.__ocean.setState({ cameraMode: 'orbit' }));
  // A tier round trip puts the reflection and SSR settings back after a toggle.
  await page.evaluate((t) => window.__ocean.setState({ quality: t === 'low' ? 'medium' : 'low' }), tier);
  await page.evaluate(`(${UNDO})()`);
  if (shot.kind === 'tour') {
    await page.evaluate((s) => window.__ocean.setState(s), { ...TOUR, quality: tier });
    await page.evaluate(() => window.__ocean.setLens(null));
    await page.evaluate(({ t }) => window.__ocean.resetDeterministic(30 + t, 90, null, 0.5 + t), { t: shot.t });
  } else {
    await page.evaluate((s) => window.__ocean.setState(s), { ...shot.state, quality: tier });
    await page.evaluate((lens) => window.__ocean.setLens(lens), shot.lens);
    await page.evaluate((time) => window.__ocean.resetDeterministic(time, 480, { throttle: 1, rudder: 0 }, null), shot.time);
    await page.evaluate(({ c }) => window.__ocean.setCameraOnShip(...c.position, ...c.target), { c: shot.camera });
  }
  await page.evaluate(`(${TOGGLES[toggle]})()`);
  await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
  await page.evaluate(async () => { for (let i = 0; i < 3; i++) await window.__ocean.capturePixels(); });
}

function grab(page) {
  return page.evaluate(async () => {
    const { width, height, data } = await window.__ocean.capturePixels();
    const url = await new Promise((done, fail) => {
      const reader = new FileReader();
      reader.onload = () => done(reader.result);
      reader.onerror = () => fail(reader.error);
      reader.readAsDataURL(new Blob([data]));
    });
    return { width, height, base64: url.slice(url.indexOf(',') + 1) };
  });
}

function writePng(shot, file) {
  return new Promise((ok, bad) => {
    const child = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba',
      '-s', `${shot.width}x${shot.height}`, '-i', '-', '-frames:v', '1', file], { stdio: ['pipe', 'inherit', 'inherit'] });
    child.on('error', bad);
    child.on('close', (c) => (c === 0 ? ok() : bad(new Error(`ffmpeg ${c}`))));
    child.stdin.end(Buffer.from(shot.base64, 'base64'));
  });
}

const o = parseArgs(process.argv);
mkdirSync(o.out, { recursive: true });
const browser = await chromium.launch({ headless: true, args: FLAGS });
try {
  const page = await browser.newPage({ viewport: { width: o.width, height: o.height }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') console.log('console error:', m.text().slice(0, 300)); });
  page.on('pageerror', (e) => console.log('page error:', String(e).slice(0, 300)));
  try {
    await boot(page, o.url, o.tier);
    // The first shot after boot draws darker water (measured: island-reveal);
    // set one up and throw it away so every saved frame is the steady state.
    await setupShot(page, SHOTS[o.shots[0]], o.tier, "base");
    for (const name of o.shots) {
      for (const tog of o.toggles) {
        await setupShot(page, SHOTS[name], o.tier, tog);
        const file = join(o.out, `${name}-${o.tier}-${tog}.png`);
        await writePng(await grab(page), file);
        console.log(file);
        for (let f = 1; f <= o.burst; f++) {
          await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
          await writePng(await grab(page), join(o.out, `${name}-${o.tier}-${tog}-f${String(f).padStart(2, '0')}.png`));
        }
      }
    }
  } finally {
    await page.close();
  }
} finally {
  await browser.close();
}
