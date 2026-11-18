#!/usr/bin/env node
/**
 * Motion clips: ten-second frame strips of four situations, so motion can be
 * reviewed. Stills hid the texture-scroll foam fault; this is how the next
 * fault of that kind is caught.
 *
 * Usage
 *   node scripts/motion-clips.mjs <url> [options]
 *
 *   <url>              A running app, e.g. http://127.0.0.1:5173 (npm run dev)
 *                      or http://127.0.0.1:4173 (npm run preview).
 *   --clips a,b        Which clips to record. Default: all four.
 *                      breeze  helm in a breeze, straight run
 *                      turn    a hard turn at speed
 *                      storm   the storm, at the helm
 *                      landfall the tour's beach flyover, the closest it gets to land
 *   --seconds N        Clip length in seconds. Default 10.
 *   --fps N            Frames per second of simulated time. Default 30.
 *   --size WxH         Capture size. Default 960x540. The README gallery and the
 *                      baselines use 1600x900 and 1280x720; larger frames cost
 *                      PNG size, not accuracy.
 *   --tier T           Quality tier: low, medium, high, ultra, max. Default high.
 *   --sheet-every N    One tile in the contact sheet every N frames. Default 15.
 *   --out DIR          Default test-results/motion
 *
 * Output (under --out)
 *   <clip>/0000.png ... numbered frames, top-down RGB
 *   <clip>-sheet.png   contact sheet, tiles left to right, top to bottom
 *   <clip>/clip.json   what was recorded, with the run's settings
 *
 * How it works
 *   The simulation is stepped, not played: `window.__ocean.resetDeterministic`
 *   rewinds every clock and settles the world, then each frame is two steps of
 *   1/60 s (the step the settle uses, so 30 frames a second is real elapsed
 *   simulation time) followed by `capturePixels`, which renders the post chain
 *   offscreen and reads it back. Frame pacing, the display and the browser's
 *   rAF throttling play no part, so the same arguments give the same strip.
 *   This is the path `tests/lib/capture.ts` and `tests/lib/shots.ts` use; the
 *   clips below are built from the canonical shots `boat-chase`, `storm` and
 *   `cinematic-landfall`.
 *
 * Needs a real GPU. Headless Chromium has no WebGPU adapter without the flags
 * below, and a software rasteriser would produce strips of the fallback, so the
 * script stops if the GPU string names one.
 *
 * Browser hygiene: headless only; every page and the browser are closed in
 * `finally`. It starts no server. Point it at one you started.
 */

import { mkdirSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

import { chromium } from '@playwright/test';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');

const CHROME_FLAGS = [
  '--enable-unsafe-webgpu',
  '--enable-gpu',
  '--disable-dawn-features=use_dxc',
];

const TIERS = ['low', 'medium', 'high', 'ultra', 'max'];
const STEP = 1 / 60;

// --------------------------------------------------------------------- clips

/**
 * Each clip names the same things a canonical shot does (see `Shot` in
 * `tests/lib/shots.ts`), plus `script`: ship input changes at a time in seconds
 * from the start of the clip. A boat clip uses the chase rig, which is what a
 * player sees; its camera is snapped to the rig after the settle, exactly as
 * `applyShot` does.
 *
 * `shipInput` is the engine order held through the settle. The ship has a
 * velocity time constant near 6.7 s, so 480 settle steps (8 s) put the hull at
 * about 8 m/s when the clip starts, as in `boat-chase`.
 */
const CLIPS = {
  breeze: {
    title: 'Helm in a breeze',
    state: { cameraMode: 'boat', windSpeed: 9, peakWavelength: 40, cloudCoverage: 0.3, preset: 'seaOfThieves' },
    time: 55.5,
    settleSteps: 480,
    shipInput: { throttle: 1, rudder: 0 },
    script: [],
  },
  turn: {
    title: 'A hard turn',
    state: { cameraMode: 'boat', windSpeed: 11, peakWavelength: 40, cloudCoverage: 0.35, preset: 'seaOfThieves' },
    time: 55.5,
    settleSteps: 480,
    shipInput: { throttle: 1, rudder: 0 },
    // A second of straight running, then the rudder hard over for five seconds,
    // then back to centre so the recovery is in the frame too.
    script: [
      { at: 1, rudder: 1 },
      { at: 6, rudder: 0 },
    ],
  },
  storm: {
    title: 'The storm',
    state: { cameraMode: 'boat', windSpeed: 21, peakWavelength: 60, cloudCoverage: 0.95, preset: 'storm' },
    time: 41,
    settleSteps: 480,
    shipInput: { throttle: 1, rudder: 0 },
    script: [],
  },
  landfall: {
    title: 'Landfall',
    // The tour's `beach-flyover` runs from 51 s to 62 s on its lap; the clip starts half a second in, because the first frame at 51 s is still the cut from the previous shot; the
    // canonical `cinematic-landfall` shot is its middle. The tour carries its
    // own hour and weather, so the clip needs nothing else.
    state: { cameraMode: 'cinematic', windSpeed: 15, peakWavelength: 47, cloudCoverage: 0.32, preset: 'skyPro' },
    time: 30,
    cinematicTime: 51.5,
    settleSteps: 90,
    shipInput: null,
    script: [],
  },
};

// ----------------------------------------------------------------- arguments

function parseArgs(argv) {
  const opts = {
    url: null,
    clips: Object.keys(CLIPS),
    seconds: 10,
    fps: 30,
    width: 960,
    height: 540,
    tier: 'high',
    sheetEvery: 15,
    out: join(ROOT, 'test-results', 'motion'),
  };
  const rest = argv.slice(2);
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    const next = () => {
      const v = rest[++i];
      if (v === undefined) fail(`${a} needs a value`);
      return v;
    };
    switch (a) {
      case '--clips': opts.clips = next().split(',').map((s) => s.trim()).filter(Boolean); break;
      case '--seconds': opts.seconds = Number(next()); break;
      case '--fps': opts.fps = Number(next()); break;
      case '--size': {
        const m = /^(\d+)x(\d+)$/.exec(next());
        if (!m) fail('--size takes WIDTHxHEIGHT, e.g. 960x540');
        opts.width = Number(m[1]);
        opts.height = Number(m[2]);
        break;
      }
      case '--tier': opts.tier = next(); break;
      case '--sheet-every': opts.sheetEvery = Number(next()); break;
      case '--out': opts.out = resolve(next()); break;
      case '-h':
      case '--help':
        console.log('node scripts/motion-clips.mjs <url> [--clips breeze,turn,storm,landfall] [--seconds 10] [--fps 30]\n' +
          '  [--size 960x540] [--tier high] [--sheet-every 15] [--out test-results/motion]');
        process.exit(0);
        break;
      default:
        if (a.startsWith('--')) fail(`unknown option ${a}`);
        if (opts.url !== null) fail(`unexpected argument ${a}`);
        opts.url = a;
    }
  }
  if (opts.url === null) fail('give the URL of a running app, e.g. http://127.0.0.1:5173');
  for (const c of opts.clips) if (!(c in CLIPS)) fail(`unknown clip "${c}"; known: ${Object.keys(CLIPS).join(', ')}`);
  if (!TIERS.includes(opts.tier)) fail(`unknown tier "${opts.tier}"`);
  for (const k of ['seconds', 'fps', 'sheetEvery']) {
    if (!Number.isFinite(opts[k]) || opts[k] < 1) fail(`--${k} must be a positive number`);
  }
  if (!Number.isInteger(opts.fps) || 60 % opts.fps !== 0) {
    fail('--fps must divide 60 (for example 15, 20, 30, 60) so that a frame is a whole number of 1/60 s steps');
  }
  return opts;
}

function fail(message) {
  console.error(`motion-clips: ${message}`);
  process.exit(2);
}

// ----------------------------------------------------------------------- PNG

let crcTable = null;
function crc32(buf) {
  if (crcTable === null) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

/** RGB, 8 bits a channel, from a tightly packed width*height*3 buffer. */
function encodePngRgb(width, height, rgb) {
  const row = width * 3;
  const raw = Buffer.alloc((row + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (row + 1)] = 0; // filter: none
    rgb.copy(raw, y * (row + 1) + 1, y * row, (y + 1) * row);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 3 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** RGBA to packed RGB. The capture's alpha is not a picture, so it is dropped. */
function rgbaToRgb(rgba, width, height) {
  const rgb = Buffer.alloc(width * height * 3);
  for (let i = 0, j = 0; i < width * height * 4; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return rgb;
}

/** Box-filter shrink of a packed RGB image to `tw` x `th`. */
function shrinkRgb(src, sw, sh, tw, th) {
  const dst = Buffer.alloc(tw * th * 3);
  for (let y = 0; y < th; y++) {
    const y0 = Math.floor((y * sh) / th);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * sh) / th));
    for (let x = 0; x < tw; x++) {
      const x0 = Math.floor((x * sw) / tw);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * sw) / tw));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const i = (yy * sw + xx) * 3;
          r += src[i]; g += src[i + 1]; b += src[i + 2]; n++;
        }
      }
      const o = (y * tw + x) * 3;
      dst[o] = r / n; dst[o + 1] = g / n; dst[o + 2] = b / n;
    }
  }
  return dst;
}

/** Tiles in a grid with a thin dark gutter. */
function contactSheet(tiles, tileW, tileH, columns) {
  const gutter = 4;
  const rows = Math.ceil(tiles.length / columns);
  const w = columns * tileW + (columns + 1) * gutter;
  const h = rows * tileH + (rows + 1) * gutter;
  const sheet = Buffer.alloc(w * h * 3, 16);
  tiles.forEach((tile, k) => {
    const ox = gutter + (k % columns) * (tileW + gutter);
    const oy = gutter + Math.floor(k / columns) * (tileH + gutter);
    for (let y = 0; y < tileH; y++) {
      tile.copy(sheet, ((oy + y) * w + ox) * 3, y * tileW * 3, (y + 1) * tileW * 3);
    }
  });
  return { width: w, height: h, rgb: sheet };
}

// ------------------------------------------------------------------- the run

async function bootOcean(page, url, tier) {
  const target = new URL(url);
  target.searchParams.set('gate', '0');
  target.searchParams.set('quality', tier);
  await page.goto(target.toString(), { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 120_000 });
  await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 240_000 });
  await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 240_000 });
}

/** The same order `applyShot` in tests/lib/capture.ts uses. */
async function applyClip(page, clip, tier) {
  await page.evaluate(async (state) => {
    await window.__ocean.setState({
      quality: state.quality,
      cameraMode: state.cameraMode,
      windSpeed: state.windSpeed,
      peakWavelength: state.peakWavelength,
      cloudCoverage: state.cloudCoverage,
      preset: state.preset,
    });
  }, { ...clip.state, quality: tier });

  await page.evaluate(() => window.__ocean.setLens(null));

  await page.evaluate(
    ({ time, settleSteps, shipInput, cinematicTime }) =>
      window.__ocean.resetDeterministic(time, settleSteps, shipInput, cinematicTime),
    {
      time: clip.time,
      settleSteps: clip.settleSteps,
      shipInput: clip.shipInput ?? null,
      cinematicTime: clip.cinematicTime ?? null,
    },
  );

  if (clip.state.cameraMode === 'boat') {
    // The chase rig damps toward its pose; this places it exactly.
    await page.evaluate(() => window.__ocean.director.snapToTarget());
  }

  // Three warm-up captures, discarded; see `applyShot` for the measurement.
  await page.evaluate(async () => {
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
  });
}

/** One frame as base64 RGBA, via a Blob so the browser does the encoding. */
function captureFrame(page) {
  return page.evaluate(async () => {
    const { width, height, data } = await window.__ocean.capturePixels();
    const url = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(new Blob([data]));
    });
    return { width, height, base64: url.slice(url.indexOf(',') + 1) };
  });
}

async function recordClip(page, name, clip, opts) {
  const frames = Math.round(opts.seconds * opts.fps);
  const stepsPerFrame = Math.round(60 / opts.fps);
  const dir = join(opts.out, name);
  if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  console.log(`[${name}] ${clip.title}: settling`);
  await applyClip(page, clip, opts.tier);

  const pending = [...clip.script].sort((a, b) => a.at - b.at);
  const tiles = [];
  let tileSize = null;
  let imgW = 0;
  let imgH = 0;
  const started = Date.now();

  for (let f = 0; f < frames; f++) {
    if (f > 0) {
      const t = f / opts.fps;
      while (pending.length > 0 && pending[0].at <= t) {
        const ev = pending.shift();
        await page.evaluate(
          ([throttle, rudder]) => window.__ocean.setShipInput(throttle, rudder),
          [ev.throttle ?? clip.shipInput?.throttle ?? 1, ev.rudder ?? 0],
        );
      }
      await page.evaluate(([dt, n]) => window.__ocean.step(dt, n), [STEP, stepsPerFrame]);
    }
    const shot = await captureFrame(page);
    imgW = shot.width;
    imgH = shot.height;
    const rgb = rgbaToRgb(Buffer.from(shot.base64, 'base64'), imgW, imgH);
    writeFileSync(join(dir, `${String(f).padStart(4, '0')}.png`), encodePngRgb(imgW, imgH, rgb));

    if (f % opts.sheetEvery === 0) {
      if (tileSize === null) {
        const tw = 320;
        tileSize = [tw, Math.round((tw * imgH) / imgW)];
      }
      tiles.push(shrinkRgb(rgb, imgW, imgH, tileSize[0], tileSize[1]));
    }
    if ((f + 1) % 30 === 0 || f === frames - 1) {
      console.log(`[${name}] ${f + 1}/${frames} frames`);
    }
  }

  const columns = Math.min(5, tiles.length);
  const sheet = contactSheet(tiles, tileSize[0], tileSize[1], columns);
  const sheetPath = join(opts.out, `${name}-sheet.png`);
  writeFileSync(sheetPath, encodePngRgb(sheet.width, sheet.height, sheet.rgb));

  const stateAfter = await page.evaluate(() => window.__ocean.shipState());
  writeFileSync(
    join(dir, 'clip.json'),
    JSON.stringify(
      {
        clip: name,
        title: clip.title,
        url: opts.url,
        tier: opts.tier,
        frames,
        fps: opts.fps,
        seconds: opts.seconds,
        stepSeconds: STEP,
        stepsPerFrame,
        width: imgW,
        height: imgH,
        sheetEvery: opts.sheetEvery,
        shipAtEnd: stateAfter,
        wallSeconds: Math.round((Date.now() - started) / 100) / 10,
      },
      null,
      2,
    ),
  );
  console.log(`[${name}] done in ${Math.round((Date.now() - started) / 1000)} s -> ${dir}\n[${name}] sheet: ${sheetPath}`);
}

async function main() {
  const opts = parseArgs(process.argv);
  mkdirSync(opts.out, { recursive: true });

  const browser = await chromium.launch({ headless: true, args: CHROME_FLAGS });
  let exitCode = 0;
  try {
    for (const name of opts.clips) {
      // A fresh page per clip, as the gallery does, so no clip inherits a sea
      // state from the one before it.
      const context = await browser.newContext({
        viewport: { width: opts.width, height: opts.height },
        deviceScaleFactor: 1,
      });
      const page = await context.newPage();
      const errors = [];
      page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });
      page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
      try {
        await bootOcean(page, opts.url, opts.tier);
        const gpu = await page.evaluate(() => {
          const gl = document.createElement('canvas').getContext('webgl2');
          const info = gl?.getExtension('WEBGL_debug_renderer_info') ?? null;
          return gl && info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : 'unknown';
        });
        if (/swiftshader|llvmpipe|software|basic render/i.test(gpu)) {
          throw new Error(`needs a real GPU; this run is on ${gpu}`);
        }
        await recordClip(page, name, CLIPS[name], opts);
        if (errors.length > 0) console.warn(`[${name}] ${errors.length} console error(s); first: ${errors[0]}`);
      } catch (error) {
        console.error(`[${name}] failed: ${error instanceof Error ? error.message : error}`);
        exitCode = 1;
      } finally {
        await page.close().catch(() => undefined);
        await context.close().catch(() => undefined);
      }
    }
  } finally {
    await browser.close().catch(() => undefined);
  }
  process.exit(exitCode);
}

main();
