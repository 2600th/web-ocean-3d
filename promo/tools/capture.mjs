#!/usr/bin/env node
/**
 * Footage for the promo film, captured from the running game.
 *
 * The simulation is stepped, not played (the same path as
 * `scripts/motion-clips.mjs`): each frame is two steps of 1/60 s and one
 * `capturePixels`, so the footage is 30 frames a second of real simulated time
 * whatever the machine's live frame rate. Frames go straight into ffmpeg as
 * raw RGBA; nothing is written to disk but the finished clip.
 *
 *   node promo/tools/capture.mjs <url> [--clips a,b] [--size 1920x1080]
 *        [--tier ultra] [--out promo/assets/footage] [--scout]
 *
 * --scout writes stills (one every `scoutEvery` seconds of each clip) at the
 * given size instead of video, to frame the shots before the long run.
 *
 * Headless only; the page and the browser are closed in `finally`.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from '@playwright/test';

const HERE = dirname(fileURLToPath(import.meta.url));
const PROMO = resolve(HERE, '..');
const FLAGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'];
const STEP = 1 / 60;
const FPS = 30;

const SAILING = { cameraMode: 'boat', windSpeed: 13, peakWavelength: 40, cloudCoverage: 0.4, preset: 'seaOfThieves' };
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t * t * (3 - 2 * t);
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

/**
 * `camera(u)` returns { frame: 'ship' | 'world', position, target } for
 * u = 0..1 through the clip, or is omitted to leave the mode's own camera.
 */
const CLIPS = {
  // The game's own authored flight, dawn to night with a squall: 90 s, uncut.
  tour: {
    state: { cameraMode: 'cinematic', windSpeed: 15, peakWavelength: 47, cloudCoverage: 0.32, preset: 'skyPro' },
    time: 30, cinematicTime: 0.5, settleSteps: 90, seconds: 89, scoutEvery: 3,
  },
  // Off the port bow, low, the ship sailing past: the opening frame.
  pass: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 7, lens: 34, scoutEvery: 3,
    camera: (u) => ({ frame: 'ship', position: mix3([44, 2.2, -30], [20, 3.2, -34], ease(u)), target: [-2, 11, 0] }),
  },
  // On deck behind the helmsman, looking forward under the sails.
  deck: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 6, lens: 50, scoutEvery: 3,
    camera: (u) => ({ frame: 'ship', position: mix3([-13.6, 5.3, 1.9], [-12.4, 5.0, 1.2], ease(u)), target: [6, 5.6, -0.2] }),
  },
  // From the waist looking aft at the hands and the helm.
  hands: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 5, lens: 45, scoutEvery: 2,
    camera: (u) => ({ frame: 'ship', position: mix3([4.6, 4.9, 3.0], [3.4, 4.7, 2.5], ease(u)), target: [-9, 4.3, -0.2] }),
  },
  // The player's own view: the chase camera with the interface, sails set.
  helm: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 8, hud: true, scoutEvery: 3,
    script: [{ at: 2.2, rudder: 0.7 }, { at: 5.2, rudder: 0 }],
  },
  // Off the quarter, the sails broad to the lens: candidates for the opening.
  quarterp: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 8, lens: 36, scoutEvery: 3,
    camera: (u) => ({ frame: 'ship', position: mix3([-34, 2.6, -41], [-25, 3.4, -44], ease(u)), target: [2, 11, 0] }),
  },
  quarters: {
    state: SAILING, time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 7, lens: 36, scoutEvery: 3,
    camera: (u) => ({ frame: 'ship', position: mix3([-36, 2.6, 40], [-26, 3.4, 44], ease(u)), target: [2, 11, 0] }),
  },
  // The hull from below as she passes overhead.
  keel: {
    state: { cameraMode: 'boat', windSpeed: 10, peakWavelength: 40, cloudCoverage: 0.3, preset: 'skyPro' },
    time: 55.5, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 5, lens: 58, scoutEvery: 2,
    camera: (u) => ({ frame: 'ship', position: mix3([20, -7, 9], [6, -8, 8], u), target: [-2, 1, 0] }),
  },
  // The storm from a camera boat, with two strikes.
  storm: {
    state: { cameraMode: 'boat', windSpeed: 21, peakWavelength: 60, cloudCoverage: 0.95, preset: 'storm' },
    time: 41, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 10, lens: 44, scoutEvery: 2,
    strikes: [2.6, 6.4],
    camera: (u) => ({ frame: 'ship', position: mix3([36, 3, 30], [26, 3.4, 34], ease(u)), target: [-6, 11, 0] }),
  },
  // The storm at the helm, the bow going into it.
  stormhelm: {
    state: { cameraMode: 'boat', windSpeed: 21, peakWavelength: 60, cloudCoverage: 0.95, preset: 'storm' },
    time: 41, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 8, scoutEvery: 2,
    strikes: [3.0],
  },
  // Down the moon's track, lanterns lit.
  moonlit: {
    state: { cameraMode: 'boat', windSpeed: 9, peakWavelength: 60, cloudCoverage: 0.25, preset: 'moonlit' },
    time: 18, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 10, lens: 36, scoutEvery: 4,
    camera: (u) => ({ frame: 'ship', position: mix3([-34, 3.2, 40], [-26, 3.6, 46], ease(u)), target: [9.4, 12, 6.7] }),
  },
  // Low sun, the hull on the glitter path.
  sunset: {
    state: { cameraMode: 'boat', windSpeed: 6, peakWavelength: 40, cloudCoverage: 0.35, preset: 'sunset' },
    time: 31.25, settleSteps: 480, shipInput: { throttle: 1, rudder: 0 }, seconds: 7, lens: 30, scoutEvery: 3,
    camera: (u) => ({ frame: 'ship', position: mix3([70, 3, 44], [60, 3.4, 52], ease(u)), target: [-4, 11.6, 0] }),
  },
};

function parseArgs(argv) {
  const o = { url: null, clips: Object.keys(CLIPS), width: 1920, height: 1080, tier: 'ultra', out: join(PROMO, 'assets', 'footage'), scout: false };
  const rest = argv.slice(2);
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === '--clips') o.clips = rest[++i].split(',');
    else if (a === '--size') { const m = /^(\d+)x(\d+)$/.exec(rest[++i]); o.width = +m[1]; o.height = +m[2]; }
    else if (a === '--tier') o.tier = rest[++i];
    else if (a === '--out') o.out = resolve(rest[++i]);
    else if (a === '--scout') o.scout = true;
    else if (!a.startsWith('--')) o.url = a;
    else throw new Error(`unknown option ${a}`);
  }
  if (!o.url) throw new Error('give the URL of a running build');
  for (const c of o.clips) if (!(c in CLIPS)) throw new Error(`unknown clip ${c}`);
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

async function setCamera(page, clip, u) {
  if (!clip.camera) return;
  const c = clip.camera(u);
  await page.evaluate(({ c }) => {
    const [px, py, pz] = c.position;
    const [tx, ty, tz] = c.target;
    if (c.frame === 'ship') window.__ocean.setCameraOnShip(px, py, pz, tx, ty, tz);
    else window.__ocean.setCamera(px, py, pz, tx, ty, tz);
  }, { c });
}

async function apply(page, clip, tier) {
  // A ship pin holds until the mode changes; leave the mode and come back so a
  // clip with no camera of its own gets the mode's camera, not the last clip's.
  await page.evaluate(() => window.__ocean.setState({ cameraMode: 'orbit' }));
  await page.evaluate((state) => window.__ocean.setState(state), { ...clip.state, quality: tier });
  await page.evaluate((lens) => window.__ocean.setLens(lens), clip.lens ?? null);
  await page.evaluate(
    ({ time, settleSteps, shipInput, cinematicTime }) =>
      window.__ocean.resetDeterministic(time, settleSteps, shipInput, cinematicTime),
    { time: clip.time, settleSteps: clip.settleSteps, shipInput: clip.shipInput ?? null, cinematicTime: clip.cinematicTime ?? null },
  );
  if (clip.state.cameraMode === 'boat' && !clip.camera) {
    await page.evaluate(() => window.__ocean.director.snapToTarget());
  }
  if (clip.hud) {
    // The interface as a player has it: sails set, so the readouts agree.
    await page.evaluate(() => window.__ocean.gameplay?.setSail('full'));
    // The frame-rate readout would show the capture's stepping rate, which is
    // not the game's live speed; leave it out of the picture.
    await page.addStyleTag({ content: '.hud-corner { visibility: hidden !important; }' });
    await page.evaluate(([dt, n]) => window.__ocean.step(dt, n), [STEP, 60]);
  }
  await setCamera(page, clip, 0);
  await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
  await page.evaluate(async () => {
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
  });
}

function grabPixels(page) {
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

function encoder(args) {
  const child = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((ok, bad) => {
    child.on('error', bad);
    child.on('close', (code) => (code === 0 ? ok() : bad(new Error(`ffmpeg exited ${code}`))));
  });
  const write = (buf) => new Promise((ok) => (child.stdin.write(buf) ? ok() : child.stdin.once('drain', ok)));
  return { write, end: () => { child.stdin.end(); return done; } };
}

const X264 = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p', '-g', '15', '-movflags', '+faststart'];

async function record(page, name, clip, o) {
  const frames = Math.round(clip.seconds * FPS);
  console.log(`[${name}] settling`);
  await apply(page, clip, o.tier);
  const pending = [...(clip.script ?? [])];
  const strikes = [...(clip.strikes ?? [])];
  mkdirSync(o.out, { recursive: true });
  const scoutStep = Math.round((clip.scoutEvery ?? 2) * FPS);
  let enc = null;
  const started = Date.now();

  for (let f = 0; f < frames; f++) {
    const t = f / FPS;
    if (f > 0) {
      while (pending.length > 0 && pending[0].at <= t) {
        const ev = pending.shift();
        await page.evaluate(([th, r]) => window.__ocean.setShipInput(th, r), [ev.throttle ?? 1, ev.rudder ?? 0]);
      }
      while (strikes.length > 0 && strikes[0] <= t) {
        strikes.shift();
        await page.evaluate(() => window.__ocean.lightning.strike(window.__ocean.camera));
      }
      await setCamera(page, clip, f / Math.max(1, frames - 1));
      await page.evaluate(([dt]) => window.__ocean.step(dt, 2), [STEP]);
    }
    if (o.scout) {
      if (f % scoutStep !== 0) continue;
      const file = join(o.out, `${name}-${String(Math.round(t)).padStart(2, '0')}s.png`);
      if (clip.hud) await page.screenshot({ path: file });
      else {
        const shot = await grabPixels(page);
        const one = encoder(['-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${shot.width}x${shot.height}`, '-i', '-', '-frames:v', '1', file]);
        await one.write(Buffer.from(shot.base64, 'base64'));
        await one.end();
      }
      console.log(`[${name}] scout ${file}`);
      continue;
    }
    if (clip.hud) {
      const png = await page.screenshot({ type: 'png' });
      enc ??= encoder(['-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-', ...X264, join(o.out, `${name}.mp4`)]);
      await enc.write(png);
    } else {
      const shot = await grabPixels(page);
      enc ??= encoder(['-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${shot.width}x${shot.height}`, '-framerate', String(FPS), '-i', '-', ...X264, join(o.out, `${name}.mp4`)]);
      await enc.write(Buffer.from(shot.base64, 'base64'));
    }
    if ((f + 1) % 60 === 0 || f === frames - 1) {
      const per = (Date.now() - started) / (f + 1);
      console.log(`[${name}] ${f + 1}/${frames} frames, ${per.toFixed(0)} ms each`);
    }
  }
  if (enc) await enc.end();
}

const o = parseArgs(process.argv);
const browser = await chromium.launch({ headless: true, args: FLAGS });
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: o.width, height: o.height }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });
  try {
    await boot(page, o.url, o.tier);
    for (const name of o.clips) await record(page, name, CLIPS[name], o);
  } finally {
    await page.close();
  }
} finally {
  await browser.close();
}
writeFileSync(join(o.out, o.scout ? 'scout-errors.json' : 'capture-errors.json'), JSON.stringify([...new Set(errors)], null, 2));
console.log(`done; ${new Set(errors).size} distinct console errors`);
