#!/usr/bin/env node
/**
 * Close stills of the sails with shading terms switched one at a time, to find
 * what draws the black patches on the cloth. Needs the dev server (it uses the
 * DEV-only `window.__sails`).
 *
 *   node promo/tools/sail-probe.mjs <url> <outDir> [variantA,variantB]
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { chromium } from '@playwright/test';

const [url, outDir, only] = process.argv.slice(2);
const FLAGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'];
const STEP = 1 / 60;
const W = 1600;
const H = 900;
mkdirSync(outDir, { recursive: true });

const VARIANTS = {
  base: () => {},
  noReceive: () => {
    const m = window.__sails.object.children.find((c) => c.name === 'ship-sails-cloth');
    m.receiveShadow = false;
    m.material.needsUpdate = true;
  },
  noCast: () => {
    const m = window.__sails.object.children.find((c) => c.name === 'ship-sails-cloth');
    m.receiveShadow = true;
    m.castShadow = false;
    m.material.needsUpdate = true;
  },
  noContact: () => window.__ocean.post.chain.setOverride('contactShadows', false),
  contact: () => window.__ocean.post.chain.setOverride('contactShadows', null),
  noAo: () => window.__ocean.post.chain.setOverride('ao', false),
  ao: () => window.__ocean.post.chain.setOverride('ao', null),
  restore: () => {
    const m = window.__sails.object.children.find((c) => c.name === 'ship-sails-cloth');
    m.receiveShadow = true;
    m.castShadow = true;
    m.material.needsUpdate = true;
  },
};

async function grab(page, name) {
  const shot = await page.evaluate(async () => {
    const { width, height, data } = await window.__ocean.capturePixels();
    const dataUrl = await new Promise((done, fail) => {
      const reader = new FileReader();
      reader.onload = () => done(reader.result);
      reader.onerror = () => fail(reader.error);
      reader.readAsDataURL(new Blob([data]));
    });
    return { width, height, base64: dataUrl.slice(dataUrl.indexOf(',') + 1) };
  });
  const file = join(outDir, `${name}.png`);
  const r = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${shot.width}x${shot.height}`, '-i', '-', file],
    { input: Buffer.from(shot.base64, 'base64') },
  );
  if (r.status !== 0) throw new Error(`ffmpeg failed for ${name}`);
  console.log(file);
}

const browser = await chromium.launch({ headless: true, args: FLAGS });
try {
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  page.on('pageerror', (e) => console.log('pageerror', e.message.slice(0, 200)));
  const target = new URL(url);
  target.searchParams.set('gate', '0');
  target.searchParams.set('quality', process.env.TIER ?? 'high');
  await page.goto(target.toString(), { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 180_000 });
  await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 300_000 });
  await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 300_000 });

  const tour = process.env.MODE === 'tour';
  await page.evaluate(() => window.__ocean.setState({ cameraMode: 'orbit' }));
  let pin = async () => {};
  if (tour) {
    // The Tour's wave-skim shot, as the film captured it.
    await page.evaluate(() =>
      window.__ocean.setState({ cameraMode: 'cinematic', windSpeed: 15, peakWavelength: 47, cloudCoverage: 0.32, preset: 'skyPro' }),
    );
    await page.evaluate((t) => window.__ocean.resetDeterministic(30, 90, null, t), Number(process.env.CT ?? 16));
    // Run the shot forward so the cloth has the history it has in the film.
    for (let i = 0; i < Number(process.env.RUN ?? 0) * 2; i++) await page.evaluate(([dt]) => window.__ocean.step(dt, 30), [STEP]);
  } else {
    await page.evaluate(() =>
      window.__ocean.setState({ cameraMode: 'boat', windSpeed: 13, peakWavelength: 40, cloudCoverage: 0.4, preset: 'seaOfThieves' }),
    );
    await page.evaluate(() => window.__ocean.setLens(Number(new URLSearchParams(location.search).get('lens') ?? 24)));
    await page.evaluate(() => window.__ocean.resetDeterministic(55.5, 480, { throttle: 1, rudder: 0 }, null));
    const cam = (process.env.CAM ?? '-30,6,-36,-1.4,15,0').split(',').map(Number);
    pin = () => page.evaluate((c) => window.__ocean.setCameraOnShip(...c), cam);
  }
  await pin();
  await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
  for (let i = 0; i < 3; i++) await page.evaluate(() => window.__ocean.capturePixels());

  const names = only ? only.split(',') : ['base', 'noReceive', 'noCast', 'restore'];
  for (const name of names) {
    if (name.startsWith('t')) {
      // "t6": six frames half a second apart, to watch the patches move.
      const n = Number(name.slice(1));
      for (let i = 0; i < n; i++) {
        await page.evaluate(([dt]) => window.__ocean.step(dt, 30), [STEP]);
        await pin();
        await page.evaluate(([dt]) => window.__ocean.step(dt, 1), [STEP]);
        await grab(page, `time-${i}`);
      }
      continue;
    }
    await page.evaluate(`(${VARIANTS[name].toString()})()`);
    await pin();
    if (!tour) await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
    for (let i = 0; i < 8; i++) await page.evaluate(() => window.__ocean.capturePixels());
    await grab(page, name);
  }
} finally {
  await browser.close();
}
