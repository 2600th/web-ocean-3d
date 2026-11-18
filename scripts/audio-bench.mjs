#!/usr/bin/env node
/**
 * Audio test bench: renders each game state offline and measures it.
 *
 *   node scripts/audio-bench.mjs                 # every state, 60 s each
 *   node scripts/audio-bench.mjs --states calm,storm --seconds 20
 *   node scripts/audio-bench.mjs --save-baseline # also stores this run as "before"
 *
 * Nothing is listened to here; the numbers catch faults and the WAV files under
 * `test-results/audio/` are kept so a person can.
 *
 * How it works. `OfflineAudioContext` exists only in a browser, so the audio
 * code (`src/audio`) is bundled with Vite in memory and loaded into a headless
 * Chromium page served from an intercepted origin (no dev server, no port). The
 * page builds an `AudioSystem` on an `OfflineAudioContext` (the constructor's
 * `context` option) and drives it the way the game's frame loop does: the
 * render is suspended every `STEP_FRAMES` samples (32 ms at 48 kHz, a whole
 * number of 128-sample render quanta), and at each stop the page calls
 * `update(dt, params)` and fires any scripted events. The score schedules its
 * notes from `update`, 350 ms ahead on the audio clock, so offline it is driven
 * deterministically and the realtime 100 ms timer is not involved. Every random
 * stream in the audio code is seeded, so two runs of the same code render the
 * same samples.
 *
 * What it reports per state:
 *  - integrated loudness, LUFS: ITU-R BS.1770-4 K-weighting (the two biquads,
 *    coefficients derived for any sample rate as in libebur128), 400 ms blocks
 *    on a 100 ms hop, absolute gate -70 LUFS and relative gate -10 LU. Also the
 *    loudest 3 s short-term window.
 *  - true peak, dBTP: 4x oversampled with a 48-tap windowed-sinc interpolator
 *    (BS.1770-4 Annex 2 asks for at least 4x).
 *  - clicks: samples whose second difference stands more than `CLICK_RATIO`
 *    times above the RMS of the second difference in both the 10 ms before and
 *    the 10 ms after (the centre excluded), and above an absolute floor; events
 *    closer than 20 ms are one click; the first 0.5 s is skipped. Requiring
 *    both sides is what separates a click from the onset of something bright.
 *    It cannot see a small step buried under the noise beds, and a bare
 *    sawtooth (a spiky second difference) trips it near 9x, so it is meant
 *    for the full mix. The first few are listed with their times.
 *  - voices: peak one-shot voices (against the tier's cap) and peak music
 *    notes, sampled at every step; peak live audio nodes (created minus
 *    disconnected, counted by wrapping the context's factory methods) and
 *    peak nodes created in any one second; noise beds still rendering at zero
 *    level at the end.
 *  - music repeat: seconds until the score's phrase order comes round to its
 *    start. Measured when it happens inside the render, otherwise projected
 *    from the order length and the tempo the band settled at.
 *
 * The baseline ("before") lives in `docs/AUDIO_BENCH.baseline.json`; the table
 * in `docs/AUDIO_BENCH.md` is rewritten from it and from this run.
 */

import { chromium } from '@playwright/test';
import { build } from 'vite';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'test-results', 'audio');
const BASELINE = join(ROOT, 'docs', 'AUDIO_BENCH.baseline.json');
const REPORT = join(ROOT, 'docs', 'AUDIO_BENCH.md');
const ORIGIN = 'http://audio-bench.localhost';

const SAMPLE_RATE = 48000;
/** 12 render quanta: 32 ms. Suspend points must land on a quantum boundary. */
const STEP_FRAMES = 128 * 12;

// --- the states ------------------------------------------------------------
//
// Scene values are what the game feeds `update` in that situation (see
// `main.ts`, the audio block of the frame loop, and `updateShipAudio`). The
// wave height is the Pierson-Moskowitz figure the rest of the game uses,
// 0.22 U^2 / g. Ship motion is a pair of sinusoids standing in for the hull.
// Events are what the game would fire: slaps from the hull probes, slams,
// thunder from the lightning (delayed by distance at 343 m/s).

const hs = (u) => (0.22 * u * u) / 9.81;
// Each strike on its own bearing (the listener faces -Z), so the placed
// thunder (A4) is exercised; code without placement ignores x and z.
const thunder = (t, km) => ({ t, type: 'thunder', km, delay: (km * 1000) / 343, x: km * 1000 * Math.sin(t * 1.3), z: -km * 1000 * Math.cos(t * 1.3) });

/** Slaps every `every` seconds with a little irregularity, alternating sides. */
function slaps(every, strength, from = 1, to = 60) {
  const out = [];
  let k = 0;
  for (let t = from; t < to; t += every * (0.7 + 0.6 * ((k * 0.618) % 1))) {
    out.push({ t, type: 'slap', strength: strength * (0.6 + 0.4 * ((k * 0.382) % 1)), pan: k % 2 ? 0.6 : -0.6 });
    k++;
  }
  return out;
}

const STATES = {
  calm: {
    music: 'calm', energy: 0,
    params: { windSpeed: 5, waveHeight: hs(5), peakWavelength: 30, rain: 0, submersion: 0, hullSpeed: 0, birdCount: 40, shipDistance: 22, shipPan: 0.2, sailFill: 0, sailLuff: 0 },
    motion: { heel: 0.04, heelPeriod: 6.5, pitch: 0.015, pitchPeriod: 4.7 },
    events: [...slaps(5, 0.2)],
  },
  sailing: {
    music: 'sailing', energy: 0.73,
    params: { windSpeed: 11, waveHeight: hs(11), peakWavelength: 40, rain: 0, submersion: 0, hullSpeed: 5.5, birdCount: 40, shipDistance: 18, shipPan: 0.1, sailFill: 0.9, sailLuff: 0, windToward: 0.5, apparentWindSpeed: 13 },
    motion: { heel: 0.12, heelPeriod: 7, pitch: 0.05, pitchPeriod: 5.2, heelBias: 0.1 },
    events: [...slaps(2.6, 0.45), { t: 17, type: 'impact', strength: 0.3 }, { t: 41, type: 'impact', strength: 0.22 }, { t: 30, type: 'chime' }],
  },
  storm: {
    music: 'storm', energy: 0.8,
    params: { windSpeed: 22, waveHeight: hs(22), peakWavelength: 110, rain: 0.9, submersion: 0, hullSpeed: 6, birdCount: 40, shipDistance: 18, shipPan: -0.1, sailFill: 0.6, sailLuff: 0.15, windToward: 0.6, apparentWindSpeed: 24 },
    motion: { heel: 0.3, heelPeriod: 9, pitch: 0.12, pitchPeriod: 6.3, heelBias: 0.15 },
    events: [
      ...slaps(1.4, 0.8),
      { t: 9, type: 'impact', strength: 0.55 }, { t: 27, type: 'impact', strength: 0.7 }, { t: 49, type: 'impact', strength: 0.45 },
      thunder(4, 3), thunder(16, 0.6), thunder(24, 9), thunder(30, 12), thunder(38, 1.5), thunder(47, 5),
    ],
  },
  underwater: {
    music: 'underwater', energy: 0,
    params: { windSpeed: 11, waveHeight: hs(11), peakWavelength: 40, rain: 0, submersion: 0, hullSpeed: 0, birdCount: 40, shipDistance: 30, shipPan: 0, sailFill: 0, sailLuff: 0 },
    motion: { heel: 0.08, heelPeriod: 7, pitch: 0.03, pitchPeriod: 5.2 },
    /** Under at 1.5 s, and stays there. */
    submersion: (t) => Math.min(1, Math.max(0, (t - 1.5) / 0.4)),
    events: [],
  },
  tour: {
    music: 'tour', energy: 0.6,
    params: { windSpeed: 11, waveHeight: hs(11), peakWavelength: 40, rain: 0, submersion: 0, hullSpeed: 4.5, birdCount: 40, shipDistance: 45, shipPan: -0.3, sailFill: 0.9, sailLuff: 0 },
    motion: { heel: 0.1, heelPeriod: 7, pitch: 0.04, pitchPeriod: 5.2, heelBias: 0.08 },
    events: [...slaps(3.2, 0.35)],
  },
  nightCalm: {
    music: 'calm', energy: 0,
    // The moonlit look's weather. The audio has no time-of-day input, so this
    // differs from `calm` only by what the game feeds it.
    params: { windSpeed: 6, waveHeight: hs(6), peakWavelength: 80, rain: 0, submersion: 0, hullSpeed: 0, birdCount: 40, shipDistance: 22, shipPan: 0.2, sailFill: 0, sailLuff: 0, sunElevation: -0.3 },
    motion: { heel: 0.05, heelPeriod: 9.5, pitch: 0.02, pitchPeriod: 6.1 },
    events: [...slaps(6, 0.2)],
  },
  nearCliffs: {
    // Sailing past a 30 m rock face 40 m off: open water, the cliff, open water.
    music: 'sailing', energy: 0.6,
    params: { windSpeed: 9, waveHeight: hs(9), peakWavelength: 40, rain: 0, submersion: 0, hullSpeed: 4.5, birdCount: 40, shipDistance: 18, shipPan: 0.1, sailFill: 0.9, sailLuff: 0, listenerZ: 0 },
    motion: { heel: 0.1, heelPeriod: 7, pitch: 0.04, pitchPeriod: 5.2, heelBias: 0.08 },
    reflectors: [{ x: 0, z: 60, reach: 20, top: 30 }],
    listenerX: (t) => -270 + 9 * t,
    events: [...slaps(2.8, 0.4), { t: 29, type: 'impact', strength: 0.35 }],
  },
  islandDay: {
    // Running in toward a beach by day (A8): the listener closes from 675 m off
    // the breakers to 15 m, the camp fire 100 m up the beach. A round island
    // 500 m across centred at x = -1000; its ground falls 1 m per metre.
    music: 'calm', energy: 0,
    params: { windSpeed: 7, waveHeight: hs(7), peakWavelength: 50, rain: 0, submersion: 0, hullSpeed: 0, birdCount: 40, shipDistance: 22, shipPan: 0.2, sailFill: 0, sailLuff: 0, listenerZ: 0, sunElevation: 0.6 },
    motion: { heel: 0.05, heelPeriod: 7, pitch: 0.02, pitchPeriod: 5.2 },
    island: { x: -1000, z: 0, radius: 500, fire: { x: -560, y: 3, z: 10 } },
    listenerX: (t) => 200 - 11 * t,
    events: [...slaps(5, 0.2)],
  },
  islandNight: {
    // The same run in at night: insects instead of birds.
    music: 'calm', energy: 0,
    params: { windSpeed: 5, waveHeight: hs(5), peakWavelength: 50, rain: 0, submersion: 0, hullSpeed: 0, birdCount: 0, shipDistance: 22, shipPan: 0.2, sailFill: 0, sailLuff: 0, listenerZ: 0, sunElevation: -0.3 },
    motion: { heel: 0.05, heelPeriod: 7, pitch: 0.02, pitchPeriod: 5.2 },
    island: { x: -1000, z: 0, radius: 500, fire: { x: -560, y: 3, z: 10 } },
    listenerX: (t) => 200 - 11 * t,
    events: [...slaps(5, 0.2)],
  },
  tierChange: {
    // Not a game state: sailing with the adaptive quality dropping to Low at
    // 20 s and back to High at 40 s, which swaps the rooms and the beds.
    music: 'sailing', energy: 0.73,
    params: { windSpeed: 11, waveHeight: hs(11), peakWavelength: 40, rain: 0, submersion: 0, hullSpeed: 5.5, birdCount: 40, shipDistance: 18, shipPan: 0.1, sailFill: 0.9, sailLuff: 0 },
    motion: { heel: 0.12, heelPeriod: 7, pitch: 0.05, pitchPeriod: 5.2, heelBias: 0.1 },
    events: [{ t: 20, type: 'tier', tier: 'low' }, { t: 40, type: 'tier', tier: 'high' }],
  },
};

const DEFAULT_STATES = ['calm', 'sailing', 'storm', 'underwater', 'tour', 'nightCalm', 'nearCliffs', 'islandDay', 'islandNight', 'tierChange'];

// --- arguments ---------------------------------------------------------------

function parseArgs(argv) {
  const args = { states: DEFAULT_STATES, seconds: 60, tier: 'high', volume: 1, music: 0.5, saveBaseline: false, wav: true, src: ROOT, reduced: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--states') args.states = argv[++i].split(',');
    else if (a === '--seconds') args.seconds = Number(argv[++i]);
    else if (a === '--tier') args.tier = argv[++i];
    else if (a === '--volume') args.volume = Number(argv[++i]);
    else if (a === '--music') args.music = Number(argv[++i]);
    else if (a === '--save-baseline') args.saveBaseline = true;
    // A checkout of another revision (it needs src/audio and src/core), e.g. to
    // re-record the baseline: git archive <rev> src/audio src/core | tar -x -C <dir>
    else if (a === '--src') args.src = resolve(argv[++i]);
    else if (a === '--no-wav') args.wav = false;
    // The reduced dynamic range (laptop speakers), as the sound settings offer it.
    else if (a === '--reduced') args.reduced = true;
    else throw new Error(`audio-bench: unknown argument ${a}`);
  }
  for (const s of args.states) if (!STATES[s]) throw new Error(`audio-bench: unknown state ${s}`);
  return args;
}

// --- bundle ------------------------------------------------------------------

async function bundleAudio(src) {
  mkdirSync(OUT_DIR, { recursive: true });
  const entry = join(OUT_DIR, '_bench-entry.ts');
  writeFileSync(
    entry,
    [
      `export { AudioSystem } from ${JSON.stringify(join(src, 'src/audio/AudioSystem.ts').replace(/\\/g, '/'))};`,
      `export { ORDERS, BARS_PER_PHRASE, STEPS_PER_BAR } from ${JSON.stringify(join(src, 'src/audio/music/score.ts').replace(/\\/g, '/'))};`,
      '',
    ].join('\n'),
  );
  const result = await build({
    configFile: false,
    root: ROOT,
    logLevel: 'silent',
    build: {
      write: false,
      minify: false,
      target: 'esnext',
      lib: { entry, formats: ['es'], fileName: 'bench' },
      rollupOptions: { output: { inlineDynamicImports: true } },
    },
  });
  const outputs = Array.isArray(result) ? result : [result];
  const chunk = outputs[0].output.find((o) => o.type === 'chunk' && o.isEntry);
  if (!chunk) throw new Error('audio-bench: no entry chunk from the bundle');
  return chunk.code;
}

// --- the page side -------------------------------------------------------------
//
// Everything below runs in Chromium. It is one function so it can be handed to
// `page.evaluate` whole; it imports the bundle itself.

/* eslint-disable no-undef */
async function renderInPage(cfg) {
  const lib = await import('/bench.js');
  const { AudioSystem, ORDERS, BARS_PER_PHRASE, STEPS_PER_BAR } = lib;
  const sr = cfg.sampleRate;
  const length = Math.round(cfg.seconds * sr);
  const ctx = new OfflineAudioContext({ numberOfChannels: 2, length, sampleRate: sr });

  // --- node accounting: wrap the factories on this context only.
  const created = new WeakSet();
  const retired = new WeakSet();
  let createdCount = 0;
  let retiredCount = 0;
  const perSecond = new Array(Math.ceil(cfg.seconds) + 1).fill(0);
  const proto = Object.getPrototypeOf(ctx);
  const names = new Set();
  for (let p = proto; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
    for (const n of Object.getOwnPropertyNames(p)) {
      if (n.startsWith('create') && n !== 'createBuffer' && n !== 'createPeriodicWave') names.add(n);
    }
  }
  for (const n of names) {
    const original = ctx[n];
    if (typeof original !== 'function') continue;
    ctx[n] = function (...a) {
      const node = original.apply(ctx, a);
      if (!created.has(node)) {
        created.add(node);
        createdCount++;
        perSecond[Math.min(perSecond.length - 1, Math.floor(ctx.currentTime))]++;
      }
      return node;
    };
  }
  const disconnect = AudioNode.prototype.disconnect;
  AudioNode.prototype.disconnect = function (...a) {
    if (created.has(this) && !retired.has(this)) {
      retired.add(this);
      retiredCount++;
    }
    return disconnect.apply(this, a);
  };

  const sys = new AudioSystem({ context: ctx, volume: cfg.volume, musicVolume: cfg.music, quality: cfg.tier });
  if (cfg.reduced && typeof sys.setReducedRange === 'function') sys.setReducedRange(true);
  // The recordings decode before the render starts, so every run hears them.
  if (sys.samplesReady) await sys.samplesReady;
  // Events refused for want of a voice (the deck's, and the gulls' share of it).
  let refused = 0;
  for (const name of ['acquireVoice', 'acquireGullVoice']) {
    const original = sys[name];
    if (typeof original !== 'function') continue;
    sys[name] = (...a) => {
      const ok = original.apply(sys, a);
      if (!ok) refused++;
      return ok;
    };
  }
  // The ship's ambience (creaks, blocks, canvas) asks through its own gate.
  const host = sys.shipSfx?.host;
  if (host && typeof host.acquireAmbient === 'function') {
    const original = host.acquireAmbient;
    host.acquireAmbient = () => {
      const ok = original();
      if (!ok) refused++;
      return ok;
    };
  }
  const st = cfg.state;
  const params = { ...st.params, shipHeel: 0, shipPitch: 0 };
  sys.setMusicState(st.music, st.energy);
  sys.resetClock(0);
  if (st.reflectors && typeof sys.setReflectors === 'function') sys.setReflectors(st.reflectors);
  if (st.island && typeof sys.setIsland === 'function') {
    const isl = st.island;
    sys.setIsland({ ...isl, height: (x, z) => isl.radius - Math.hypot(x - isl.x, z - isl.z) });
  }
  await sys.resume();
  if (cfg.diag === 'wet') {
    // Diagnosis only (BENCH_DIAG=wet): the room's return alone, every dry path
    // and the score cut, to show how wet the world is.
    sys.ambienceBus?.disconnect();
    sys.effectsBus?.disconnect();
    sys.ambienceUnder?.disconnect();
    sys.music.output.disconnect();
  } else if (cfg.diag) {
    // Diagnosis only (BENCH_DIAG=music or a stem name): the score alone.
    sys.airMuffle.disconnect();
    sys.underBus.disconnect();
    if (cfg.diag !== 'music') sys.music.setSolo(cfg.diag);
  }

  const dt = cfg.stepFrames / sr;
  const steps = Math.floor(length / cfg.stepFrames);
  const events = st.events.slice().sort((a, b) => a.t - b.t);
  let nextEvent = 0;
  const submersionAt = st.submersion ? new Function('t', `return (${st.submersion})(t)`) : null;
  const listenerAt = st.listenerX ? new Function('t', `return (${st.listenerX})(t)`) : null;

  let peakFx = 0;
  let peakMusic = 0;
  let peakLive = 0;
  let repeatMeasured = null;
  let lastPhrase = null;
  let lastOrder = null;
  const fxCap = () => sys.settings?.voices ?? null;
  let peakFxCap = fxCap();
  // The loudness window (A3), where the code has one: its gain, dB.
  let hdrMin = Infinity, hdrMax = -Infinity, hdrSum = 0, hdrN = 0;

  const tick = (i) => {
    const t = i * dt;
    const m = st.motion;
    params.shipHeel = (m.heelBias ?? 0) + m.heel * Math.sin((2 * Math.PI * t) / m.heelPeriod);
    params.shipPitch = m.pitch * Math.sin((2 * Math.PI * t) / m.pitchPeriod + 0.7);
    if (submersionAt) params.submersion = submersionAt(t);
    if (listenerAt) params.listenerX = listenerAt(t);
    while (nextEvent < events.length && events[nextEvent].t <= t) {
      const e = events[nextEvent++];
      switch (e.type) {
        case 'thunder': sys.playThunder(e.km, e.delay, e.x, e.z); break;
        case 'slap': sys.waveSlap(e.strength, e.pan); break;
        case 'impact': sys.hullImpact(e.strength, false); break;
        case 'splash': sys.playSplash(e.strength, e.pan ?? 0); break;
        case 'chime': sys.playChime(); break;
        case 'tier': sys.setQuality(e.tier); break;
      }
    }
    sys.setMusicState(st.music, st.energy);
    sys.update(i === 0 ? 0 : dt, params);

    const music = sys.music;
    // Thunder has its own voices once P0.6 landed; before, it held deck slots.
    const fx = (sys.voices?.length ?? 0) - (sys.thunderLive ?? 0);
    const notes = music?.voicesInUse ?? 0;
    if (fx > peakFx) peakFx = fx;
    if (notes > peakMusic) peakMusic = notes;
    const live = createdCount - retiredCount;
    if (live > peakLive) peakLive = live;
    const cap = fxCap();
    if (cap !== null && (peakFxCap === null || cap > peakFxCap)) peakFxCap = cap;
    const hdr = sys.hdrDebug;
    if (hdr && i > 0) {
      hdrMin = Math.min(hdrMin, hdr.gainDb);
      hdrMax = Math.max(hdrMax, hdr.gainDb);
      hdrSum += hdr.gainDb;
      hdrN++;
    }

    const pos = music?.position;
    if (pos) {
      if (lastOrder === pos.order && lastPhrase !== null && pos.phrase < lastPhrase && repeatMeasured === null) {
        repeatMeasured = t;
      }
      lastOrder = pos.order;
      lastPhrase = pos.phrase;
    }
  };

  tick(0);
  for (let i = 1; i < steps; i++) {
    ctx.suspend(i * dt).then(() => {
      tick(i);
      return ctx.resume();
    });
  }
  const rendered = await ctx.startRendering();

  // Beds still rendering at zero level at the end.
  let bedsZero = 0;
  let bedsRendering = 0;
  for (const bed of sys.beds ?? []) {
    const rendering = typeof bed.rendering === 'boolean' ? bed.rendering : bed.active !== false || bed.disconnectAt !== null;
    if (!rendering) continue;
    bedsRendering++;
    if ((bed.level?.target ?? 1) < 1e-4) bedsZero++;
  }

  // Music repeat, projected when it did not happen inside the render.
  const music = sys.music;
  let repeatProjected = null;
  if (music && music.position) {
    const order = ORDERS[music.position.order];
    repeatProjected = order.length * BARS_PER_PHRASE * STEPS_PER_BAR * music.stepDur;
  }

  const L = rendered.getChannelData(0);
  const R = rendered.getChannelData(1);
  const analysis = analyse([L, R], sr, cfg.clickRatio, cfg.clickFloor);

  let wav = null;
  if (cfg.wav) wav = encodeWav([L, R], sr);

  sys.dispose();
  AudioNode.prototype.disconnect = disconnect;

  return {
    ...analysis,
    peakFxVoices: peakFx,
    fxVoiceCap: peakFxCap,
    refused,
    peakMusicNotes: peakMusic,
    peakLiveNodes: peakLive,
    peakNodesPerSecond: Math.max(...perSecond),
    nodesCreated: createdCount,
    bedsRendering,
    bedsAtZero: bedsZero,
    repeatMeasured,
    repeatProjected,
    hdrGainDb: hdrN > 0 ? { min: +hdrMin.toFixed(1), max: +hdrMax.toFixed(1), mean: +(hdrSum / hdrN).toFixed(1) } : null,
    wav,
  };

  // --- analysis ------------------------------------------------------------

  function biquad(x, b0, b1, b2, a1, a2) {
    const y = new Float64Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let n = 0; n < x.length; n++) {
      const v = x[n];
      const o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = v; y2 = y1; y1 = o;
      y[n] = o;
    }
    return y;
  }

  /** BS.1770-4 K-weighting, coefficients as libebur128 derives them for any rate. */
  function kWeight(x, fs) {
    let f0 = 1681.974450955533, G = 3.999843853973347, Q = 0.7071752369554196;
    let K = Math.tan((Math.PI * f0) / fs);
    const Vh = Math.pow(10, G / 20), Vb = Math.pow(Vh, 0.4996667741545416);
    let a0 = 1 + K / Q + K * K;
    const s1 = [(Vh + (Vb * K) / Q + K * K) / a0, (2 * (K * K - Vh)) / a0, (Vh - (Vb * K) / Q + K * K) / a0, (2 * (K * K - 1)) / a0, (1 - K / Q + K * K) / a0];
    f0 = 38.13547087602444; Q = 0.5003270373238773;
    K = Math.tan((Math.PI * f0) / fs);
    a0 = 1 + K / Q + K * K;
    const s2 = [1, -2, 1, (2 * (K * K - 1)) / a0, (1 - K / Q + K * K) / a0];
    const y1 = biquad(x, ...s1);
    return biquad(y1, ...s2);
  }

  function analyse(channels, fs, clickRatio, clickFloor) {
    const n = channels[0].length;
    // Loudness.
    const weighted = channels.map((c) => kWeight(c, fs));
    const block = Math.round(0.4 * fs), hop = Math.round(0.1 * fs);
    const blocks = [];
    for (let s = 0; s + block <= n; s += hop) {
      let z = 0;
      for (const w of weighted) {
        let acc = 0;
        for (let i = s; i < s + block; i++) acc += w[i] * w[i];
        z += acc / block;
      }
      blocks.push(z);
    }
    const lufs = (z) => -0.691 + 10 * Math.log10(z);
    const abs = blocks.filter((z) => lufs(z) > -70);
    let integrated = -Infinity;
    if (abs.length > 0) {
      const rel = lufs(abs.reduce((a, b) => a + b, 0) / abs.length) - 10;
      const gated = abs.filter((z) => lufs(z) > rel);
      if (gated.length > 0) integrated = lufs(gated.reduce((a, b) => a + b, 0) / gated.length);
    }
    // Short-term (3 s) maximum: 30 blocks of 100 ms hop ~ average of momentary z over 3 s.
    let shortMax = -Infinity;
    const st = Math.round(3 * fs);
    for (let s = 0; s + st <= n; s += hop) {
      let z = 0;
      for (const w of weighted) {
        let acc = 0;
        for (let i = s; i < s + st; i++) acc += w[i] * w[i];
        z += acc / st;
      }
      shortMax = Math.max(shortMax, lufs(z));
    }

    // True peak: 4x oversampling, 12 taps per phase, Blackman-windowed sinc.
    const taps = 48, phases = 4, per = taps / phases;
    const h = new Float64Array(taps);
    for (let i = 0; i < taps; i++) {
      const t = (i - (taps - 1) / 2) / phases;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
      h[i] = sinc * w;
    }
    // Normalise each phase to unity DC gain.
    const poly = [];
    for (let p = 0; p < phases; p++) {
      const c = new Float64Array(per);
      let sum = 0;
      for (let k = 0; k < per; k++) { c[k] = h[k * phases + p]; sum += c[k]; }
      for (let k = 0; k < per; k++) c[k] /= sum;
      poly.push(c);
    }
    let samplePeak = 0, truePeak = 0;
    for (const c of channels) {
      for (let i = 0; i < n; i++) {
        const a = Math.abs(c[i]);
        if (a > samplePeak) samplePeak = a;
      }
      for (let i = per; i < n; i++) {
        for (let p = 0; p < phases; p++) {
          const co = poly[p];
          let acc = 0;
          for (let k = 0; k < per; k++) acc += co[k] * c[i - k];
          const a = Math.abs(acc);
          if (a > truePeak) truePeak = a;
        }
      }
    }
    truePeak = Math.max(truePeak, samplePeak);

    // Clicks.
    const W = Math.round(0.01 * fs), EX = 2, MERGE = Math.round(0.02 * fs);
    const clicks = [];
    for (const c of channels) {
      const d2 = new Float64Array(n);
      for (let i = 2; i < n; i++) d2[i] = c[i] - 2 * c[i - 1] + c[i - 2];
      const pre = new Float64Array(n + 1);
      for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + d2[i] * d2[i];
      // The first half second is skipped: offline, the master is not faded in
      // as it is on a real unlock, so the band and the beds start on sample 0.
      for (let i = Math.max(2 + W, Math.round(0.5 * fs)); i < n - W; i++) {
        const a = Math.abs(d2[i]);
        if (a < clickFloor) continue;
        // Louder than both sides: an onset of something bright is louder than
        // what came before it but not than what follows; a click is neither.
        const before = (pre[i - EX] - pre[i - W]) / (W - EX);
        const after = (pre[i + W + 1] - pre[i + EX + 1]) / (W - EX);
        const rms = Math.sqrt(Math.max(before, after));
        if (a > clickRatio * rms) clicks.push({ i, ratio: rms > 0 ? a / rms : Infinity, size: a });
      }
    }
    clicks.sort((a, b) => a.i - b.i);
    const events = [];
    for (const c of clicks) {
      const last = events[events.length - 1];
      if (last && c.i - last.i < MERGE) {
        if (c.ratio > last.ratio) { last.ratio = c.ratio; last.size = c.size; }
        continue;
      }
      events.push({ ...c });
    }
    return {
      integratedLufs: integrated,
      shortTermMaxLufs: shortMax,
      samplePeakDb: 20 * Math.log10(samplePeak || 1e-12),
      truePeakDb: 20 * Math.log10(truePeak || 1e-12),
      clicks: events.length,
      clickTimes: events.slice(0, 12).map((e) => ({ t: +(e.i / fs).toFixed(3), ratio: +e.ratio.toFixed(1), size: +e.size.toFixed(4) })),
    };
  }

  function encodeWav(channels, fs) {
    const n = channels[0].length, ch = channels.length;
    const bytes = 44 + n * ch * 2;
    const buf = new ArrayBuffer(bytes);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, bytes - 8, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, ch, true);
    v.setUint32(24, fs, true); v.setUint32(28, fs * ch * 2, true); v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, n * ch * 2, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < ch; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][i]));
        v.setInt16(o, Math.round(s * 32767), true);
        o += 2;
      }
    }
    const u8 = new Uint8Array(buf);
    let bin = '';
    const CH = 0x8000;
    for (let i = 0; i < u8.length; i += CH) bin += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
    return btoa(bin);
  }
}

// --- the detector's own check ----------------------------------------------------

/** A sine that drops to zero mid-cycle must be one click; noise and a clean sine none. */
async function selfTest(page, clickRatio, clickFloor) {
  return page.evaluate(
    async ({ clickRatio, clickFloor, src }) => {
      const fn = new Function(`return (${src})`)();
      return fn(clickRatio, clickFloor);
    },
    {
      clickRatio,
      clickFloor,
      src: `async function (clickRatio, clickFloor) {
        const sr = 48000, n = sr * 2;
        const sine = new Float32Array(n), cut = new Float32Array(n), noise = new Float32Array(n);
        let seed = 1;
        const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
        for (let i = 0; i < n; i++) {
          sine[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / sr);
          cut[i] = i < sr ? sine[i] : 0;
          noise[i] = 0.2 * rnd();
        }
        // Low-passed noise, like the beds.
        let lp = 0;
        for (let i = 0; i < n; i++) { lp += 0.1 * (noise[i] - lp); noise[i] = lp * 3; }
        const count = (x) => {
          const d2 = new Float64Array(n);
          for (let i = 2; i < n; i++) d2[i] = x[i] - 2 * x[i - 1] + x[i - 2];
          const pre = new Float64Array(n + 1);
          for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + d2[i] * d2[i];
          const W = 480, EX = 2; let hits = 0, last = -1e9;
          for (let i = 2 + W; i < n - W; i++) {
            const a = Math.abs(d2[i]);
            if (a < clickFloor) continue;
            const before = (pre[i - EX] - pre[i - W]) / (W - EX), after = (pre[i + W + 1] - pre[i + EX + 1]) / (W - EX);
            const rms = Math.sqrt(Math.max(before, after));
            if (a > clickRatio * rms) { if (i - last > 960) hits++; last = i; }
          }
          return hits;
        };
        return { sine: count(sine), cut: count(cut), noise: count(noise) };
      }`,
    },
  );
}

// --- report --------------------------------------------------------------------

const fmt = {
  lufs: (v) => (Number.isFinite(v) ? v.toFixed(1) : '-inf'),
  db: (v) => (Number.isFinite(v) ? v.toFixed(1) : '-inf'),
  repeat: (r) => (r.repeatMeasured !== null ? `${r.repeatMeasured.toFixed(0)} s` : r.repeatProjected !== null ? `~${r.repeatProjected.toFixed(0)} s (projected)` : 'n/a'),
};

function row(r) {
  if (!r) return ['', '', '', '', '', '', '', ''];
  return [
    fmt.lufs(r.integratedLufs),
    fmt.lufs(r.shortTermMaxLufs),
    fmt.db(r.truePeakDb),
    String(r.clicks),
    `${r.peakFxVoices}/${r.fxVoiceCap ?? "?"} fx${r.refused !== undefined ? ` (${r.refused} refused)` : ""}, ${r.peakMusicNotes} notes`,
    `${r.peakLiveNodes} live, ${r.peakNodesPerSecond}/s`,
    `${r.bedsAtZero} of ${r.bedsRendering}`,
    fmt.repeat(r),
  ];
}

function writeReport(baseline, current, args) {
  const states = [...new Set([...Object.keys(baseline?.results ?? {}), ...Object.keys(current.results)])];
  const head = ['State', 'Run', 'Integrated LUFS', 'Max short-term LUFS', 'True peak dBTP', 'Clicks', 'Peak voices', 'Peak nodes', 'Beds rendering at zero', 'Music repeats after'];
  const lines = [];
  lines.push('# Audio bench');
  lines.push('');
  lines.push('Generated by `node scripts/audio-bench.mjs` (see `docs/VERIFICATION.md`). Each state is');
  lines.push(`rendered offline for ${args.seconds} s at ${SAMPLE_RATE / 1000} kHz on the ${args.tier} tier, master volume ${args.volume}, music ${args.music}.`);
  lines.push('"before" is the stored baseline (`docs/AUDIO_BENCH.baseline.json`, recorded before the');
  lines.push('P0.6, A1 and A2 changes); "now" is the latest run. The WAV files are in `test-results/audio/`.');
  lines.push('');
  lines.push(`| ${head.join(' | ')} |`);
  lines.push(`|${head.map(() => '---').join('|')}|`);
  for (const s of states) {
    const b = baseline?.results?.[s];
    const c = current.results[s];
    if (b) lines.push(`| ${s} | before | ${row(b).join(' | ')} |`);
    if (c) lines.push(`| ${b ? '' : s} | now | ${row(c).join(' | ')} |`);
  }
  lines.push('');
  lines.push('Click times in the latest run (first twelve per state, seconds, ratio to the local level):');
  lines.push('');
  for (const s of Object.keys(current.results)) {
    const c = current.results[s];
    const list = c.clickTimes.map((k) => `${k.t} (${k.ratio}x)`).join(', ');
    lines.push(`- ${s}: ${c.clicks === 0 ? 'none' : list}`);
  }
  lines.push('');
  lines.push(`Detector self-test: a 440 Hz sine cut to zero mid-cycle gives ${current.selfTest.cut} click(s);`);
  lines.push(`the clean sine ${current.selfTest.sine}, low-passed noise ${current.selfTest.noise}.`);
  lines.push('');
  lines.push('How to read it. Integrated loudness is BS.1770-4 with both gates; true peak is 4x oversampled.');
  lines.push('A click is a sample whose second difference stands more than');
  lines.push(`${current.clickRatio}x above both the 10 ms before and after it and above ${current.clickFloor} absolute (first 0.5 s skipped). Voices are the`);
  lines.push('one-shot effects (against the tier cap) and music notes, sampled every 32 ms. Live nodes are');
  lines.push('nodes created and not yet disconnected; nodes/s is the busiest second. Beds rendering at zero');
  lines.push('are noise beds still connected while their level is zero. The repeat is when the score\'s');
  lines.push('phrase order comes back to its start: measured if it happened in the render, else projected.');
  lines.push('');
  writeFileSync(REPORT, lines.join('\n'));
}

// --- main ----------------------------------------------------------------------

const CLICK_RATIO = Number(process.env.CLICK_RATIO ?? 10);
const CLICK_FLOOR = Number(process.env.CLICK_FLOOR ?? 0.004);

async function main() {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('[audio-bench] bundling src/audio');
  const code = await bundleAudio(args.src);

  const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
  const results = {};
  let selfTestResult = null;
  try {
    const context = await browser.newContext();
    try {
      await context.route(`${ORIGIN}/**`, (route) => {
        const url = new URL(route.request().url());
        if (url.pathname === '/bench.js') {
          return route.fulfill({ status: 200, contentType: 'text/javascript', body: code });
        }
        // The recordings (`public/audio`), as the site serves them.
        if (url.pathname.startsWith('/audio/')) {
          const file = join(args.src, 'public', 'audio', url.pathname.slice('/audio/'.length).replace(/[^a-z0-9._-]/gi, ''));
          if (existsSync(file)) return route.fulfill({ status: 200, contentType: 'audio/ogg', body: readFileSync(file) });
          return route.fulfill({ status: 404, body: '' });
        }
        return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>audio bench</title>' });
      });
      const page = await context.newPage();
      page.on('console', (m) => { if (m.type() === 'error') console.log(`[page] ${m.text()}`); });
      page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
      await page.goto(`${ORIGIN}/`);
      selfTestResult = await selfTest(page, CLICK_RATIO, CLICK_FLOOR);
      console.log(`[audio-bench] click detector self-test: ${JSON.stringify(selfTestResult)}`);

      for (const name of args.states) {
        const st = STATES[name];
        const started = Date.now();
        // BENCH_EVENTS='[{"t":5,"type":"splash","strength":0.4,"pan":0.5}]' adds events to every state.
        const extra = process.env.BENCH_EVENTS ? JSON.parse(process.env.BENCH_EVENTS) : [];
        // BENCH_PARAMS='{"birdCount":0}' overrides scene values in every state (diagnosis).
        const over = process.env.BENCH_PARAMS ? JSON.parse(process.env.BENCH_PARAMS) : {};
        const state = { ...st, params: { ...st.params, ...over }, events: [...st.events, ...extra], submersion: st.submersion ? st.submersion.toString() : null, listenerX: st.listenerX ? st.listenerX.toString() : null };
        const r = await page.evaluate(
          async ({ src, cfg }) => {
            const fn = new Function(`return (${src})`)();
            return fn(cfg);
          },
          {
            src: renderInPage.toString(),
            cfg: {
              state,
              seconds: args.seconds,
              sampleRate: SAMPLE_RATE,
              stepFrames: STEP_FRAMES,
              tier: args.tier,
              volume: args.volume,
              music: args.music,
              wav: args.wav,
              clickRatio: CLICK_RATIO,
              clickFloor: CLICK_FLOOR,
              diag: process.env.BENCH_DIAG ?? null,
              reduced: args.reduced,
            },
          },
        );
        if (r.wav) {
          // The baseline keeps its own files, so before and after can be heard side by side.
          const base = args.saveBaseline ? `${name}-before` : name;
          const file = join(OUT_DIR, `${base}.wav`);
          writeFileSync(file, Buffer.from(r.wav, 'base64'));
          r.file = `test-results/audio/${base}.wav`;
        }
        delete r.wav;
        results[name] = r;
        console.log(
          `[audio-bench] ${name.padEnd(10)} ${fmt.lufs(r.integratedLufs).padStart(6)} LUFS  ` +
            `TP ${fmt.db(r.truePeakDb).padStart(5)} dBTP  clicks ${String(r.clicks).padStart(3)}  ` +
            `fx ${r.peakFxVoices}/${r.fxVoiceCap} refused ${r.refused}  notes ${r.peakMusicNotes}  nodes ${r.peakLiveNodes} (${r.peakNodesPerSecond}/s)  ` +
            `beds@0 ${r.bedsAtZero}/${r.bedsRendering}  repeat ${fmt.repeat(r)}` +
            (r.hdrGainDb ? `  hdr ${r.hdrGainDb.min}..${r.hdrGainDb.max} (mean ${r.hdrGainDb.mean}) dB` : '') +
            `  (${((Date.now() - started) / 1000).toFixed(1)} s)`,
        );
      }
      await page.close();
    } finally {
      await context.close();
    }
  } finally {
    await browser.close();
  }

  const current = { date: new Date().toISOString(), args: { ...args, src: undefined }, clickRatio: CLICK_RATIO, clickFloor: CLICK_FLOOR, selfTest: selfTestResult, results };
  writeFileSync(join(OUT_DIR, 'bench.json'), JSON.stringify(current, null, 2));
  if (args.saveBaseline) writeFileSync(BASELINE, JSON.stringify(current, null, 2) + '\n');
  const full =
    args.seconds === 60 && args.tier === 'high' && !args.reduced && !process.env.BENCH_DIAG && !process.env.BENCH_EVENTS && !process.env.BENCH_PARAMS &&
    DEFAULT_STATES.every((s) => args.states.includes(s));
  if (full) {
    const baseline = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, 'utf8')) : null;
    writeReport(baseline, current, args);
    console.log(`[audio-bench] wrote ${REPORT} and ${OUT_DIR}`);
  } else {
    console.log(`[audio-bench] partial or diagnostic run: ${REPORT} left alone; results in ${OUT_DIR}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
