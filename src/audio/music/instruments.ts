/**
 * The band, synthesised.
 *
 * Each player is a function that builds one note's worth of nodes, schedules
 * it, and hands the source back to the engine for cleanup. Everything shared
 * between notes — a body resonance, a pan, the room — lives on the instrument's
 * channel in `MusicEngine`, built once, so a note costs only what is genuinely
 * per-note: an oscillator or two and an envelope.
 *
 * What makes each one read as the instrument rather than as a synthesiser is, in
 * every case, what the *pitch or the level does over the note*, not the
 * waveform:
 *
 *  - **Fiddle.** A band-limited sawtooth (a bowed string's Helmholtz motion is
 *    one) under a body EQ. The bow is the envelope: a soft attack, a dip as the
 *    bow settles, and a swell on long notes. Vibrato arrives late, after the
 *    note is established, the way a player adds it. Cuts — a flick to the note
 *    above for 30 ms — and slides from below are the idiom's ornaments.
 *  - **Concertina / accordion.** A free reed is a pulse-like wave with a strong
 *    odd series; the bellows make the attack slow and the release short.
 *  - **Plucked strings.** Karplus–Strong: a burst of noise in a tuned, lossy
 *    delay loop. Rendered once per pitch class into a buffer and replayed.
 *  - **Bodhrán.** A frame drum is a membrane whose pitch *falls* as the head
 *    relaxes after the stick lands, plus the stick's own click.
 *  - **Glass and bells.** Inharmonic partials (2.76, 5.4 — a struck bar's
 *    modes) with long tails, for the underwater and attract arrangements.
 */

import { midiToHz } from './theory';

/** Where a note's nodes are registered so the engine can count and clean them up. */
export interface NoteHost {
  readonly ctx: BaseAudioContext;
  track(source: AudioScheduledSourceNode, nodes: readonly AudioNode[]): void;
}

// --- fiddle ------------------------------------------------------------------

export interface FiddleNote {
  /** Grace note to flick from, MIDI, or null. */
  cut: number | null;
  /** Slide up into the note from a quarter-tone flat. */
  slide: boolean;
}

const FIDDLE_ATTACK = 0.045;
const FIDDLE_RELEASE_TAU = 0.035;
const FIDDLE_VIBRATO_DELAY = 0.16;
const FIDDLE_VIBRATO_DEPTH = 0.0055;
/** Notes shorter than this get no vibrato oscillator at all: it would never be heard. */
const FIDDLE_VIBRATO_MIN = 0.32;

export function playFiddle(
  host: NoteHost,
  dest: AudioNode,
  t: number,
  midi: number,
  dur: number,
  vel: number,
  ornament: FiddleNote,
  random: () => number,
): void {
  const ctx = host.ctx;
  const f = midiToHz(midi);

  const osc = ctx.createOscillator();
  osc.type = 'sawtooth';
  if (ornament.cut !== null) {
    osc.frequency.setValueAtTime(midiToHz(ornament.cut), t);
    osc.frequency.setValueAtTime(f, t + 0.032);
  } else if (ornament.slide) {
    osc.frequency.setValueAtTime(f * 0.97, t);
    osc.frequency.exponentialRampToValueAtTime(f, t + 0.07);
  } else {
    osc.frequency.setValueAtTime(f, t);
  }

  const env = ctx.createGain();
  const g = env.gain;
  const release = t + Math.max(0.06, dur * 0.94);
  // Scaled down for quick notes so every event stays ahead of the release.
  const attack = Math.min(FIDDLE_ATTACK, dur * 0.3);
  g.setValueAtTime(0, t);
  g.linearRampToValueAtTime(vel, t + attack);
  g.linearRampToValueAtTime(vel * 0.74, t + attack + Math.min(0.08, dur * 0.3));
  if (dur > 0.5) {
    // The swell of a long bow stroke.
    g.linearRampToValueAtTime(vel * 0.92, t + dur * 0.62);
  }
  g.setValueAtTime(dur > 0.5 ? vel * 0.92 : vel * 0.74, release);
  g.setTargetAtTime(0, release, FIDDLE_RELEASE_TAU);

  osc.connect(env);
  env.connect(dest);
  const end = release + FIDDLE_RELEASE_TAU * 7;
  osc.start(t);
  osc.stop(end);

  if (dur >= FIDDLE_VIBRATO_MIN) {
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 5.2 + random() * 0.7;
    const depth = ctx.createGain();
    depth.gain.setValueAtTime(0, t);
    depth.gain.setValueAtTime(0, t + FIDDLE_VIBRATO_DELAY);
    depth.gain.linearRampToValueAtTime(f * FIDDLE_VIBRATO_DEPTH, t + FIDDLE_VIBRATO_DELAY + 0.3);
    lfo.connect(depth);
    depth.connect(osc.frequency);
    lfo.start(t);
    lfo.stop(end);
    host.track(osc, [env, lfo, depth]);
  } else {
    host.track(osc, [env]);
  }
}

// --- reeds -------------------------------------------------------------------

/**
 * A free reed's spectrum: every harmonic, falling gently, with the odd ones
 * pushed up. That odd emphasis is the difference between a squeezebox and a
 * string — a reed beating against its slot is closer to a pulse than a ramp.
 */
export function buildReedWave(ctx: BaseAudioContext): PeriodicWave {
  const n = 24;
  const real = new Float32Array(n + 1);
  const imag = new Float32Array(n + 1);
  for (let h = 1; h <= n; h++) {
    imag[h] = Math.pow(h, -0.95) * (h % 2 === 1 ? 1.25 : 0.8);
  }
  return ctx.createPeriodicWave(real, imag);
}

export function playReed(
  host: NoteHost,
  dest: AudioNode,
  wave: PeriodicWave,
  t: number,
  midi: number,
  dur: number,
  vel: number,
  attack: number,
  release: number,
  detuneCents: number,
): void {
  const ctx = host.ctx;
  const osc = ctx.createOscillator();
  osc.setPeriodicWave(wave);
  osc.frequency.value = midiToHz(midi);
  osc.detune.value = detuneCents;

  const env = ctx.createGain();
  const g = env.gain;
  const off = t + Math.max(attack + 0.02, dur);
  g.setValueAtTime(0, t);
  g.linearRampToValueAtTime(vel, t + attack);
  g.setValueAtTime(vel, off);
  g.setTargetAtTime(0, off, release / 3);

  osc.connect(env);
  env.connect(dest);
  osc.start(t);
  osc.stop(off + release * 2.5);
  host.track(osc, [env]);
}

// --- plucked strings -----------------------------------------------------------

const PLUCK_SECONDS = 1.6;
/** Seconds for the fundamental to fall 60 dB. */
const PLUCK_T60 = 1.5;

/**
 * One plucked string, rendered by Karplus–Strong.
 *
 * The loop is a delay of one period, averaged with its neighbour each pass (the
 * lowpass that makes the upper partials die first — the whole sound of a
 * string) and scaled for a set T60. A first-order allpass carries the
 * fractional part of the period, without which the integer delay is up to five
 * cents out of tune in this register. The excitation is softened noise with a
 * comb notch at a fifth of the string, i.e. plucked near the bridge.
 */
export function renderPluck(ctx: BaseAudioContext, midi: number, random: () => number): AudioBuffer {
  const sr = ctx.sampleRate;
  const f = midiToHz(midi);
  const delay = sr / f - 0.5;
  let n = Math.floor(delay);
  let frac = delay - n;
  if (frac < 0.1) {
    n -= 1;
    frac += 1;
  }
  const c = (1 - frac) / (1 + frac);
  const loss = Math.pow(10, -3 / (PLUCK_T60 * f));

  const ring = new Float32Array(n);
  let smooth = 0;
  for (let i = 0; i < n; i++) {
    smooth += 0.55 * (random() * 2 - 1 - smooth);
    ring[i] = smooth;
  }
  const notch = Math.max(1, Math.round(n * 0.2));
  for (let i = n - 1; i >= notch; i--) ring[i] -= ring[i - notch];

  const length = Math.round(PLUCK_SECONDS * sr);
  const buffer = ctx.createBuffer(1, length, sr);
  const out = buffer.getChannelData(0);
  let idx = 0;
  let apIn = 0;
  let apOut = 0;
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const cur = ring[idx];
    out[i] = cur;
    const a = Math.abs(cur);
    if (a > peak) peak = a;
    const next = ring[idx + 1 === n ? 0 : idx + 1];
    const avg = loss * 0.5 * (cur + next);
    const y = c * avg + apIn - c * apOut;
    apIn = avg;
    apOut = y;
    ring[idx] = y;
    idx = idx + 1 === n ? 0 : idx + 1;
  }
  const fade = Math.round(0.03 * sr);
  const scale = peak > 1e-6 ? 0.8 / peak : 0;
  for (let i = 0; i < length; i++) {
    const tail = i > length - fade ? (length - i) / fade : 1;
    out[i] *= scale * tail;
  }
  return buffer;
}

export function playPluck(
  host: NoteHost,
  dest: AudioNode,
  buffer: AudioBuffer,
  rate: number,
  t: number,
  vel: number,
  dur: number,
): void {
  const ctx = host.ctx;
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.playbackRate.value = rate;
  const env = ctx.createGain();
  // A damped string: full until the player lets go, then a quick choke.
  const off = t + Math.max(0.08, dur);
  env.gain.setValueAtTime(vel, t);
  env.gain.setValueAtTime(vel, off);
  env.gain.setTargetAtTime(0, off, 0.06);
  src.connect(env);
  env.connect(dest);
  src.start(t);
  src.stop(Math.min(t + buffer.duration / rate, off + 0.4));
  host.track(src, [env]);
}

// --- drums -------------------------------------------------------------------

export type DrumStroke = 'down' | 'up' | 'tom';

interface DrumSpec {
  from: number;
  to: number;
  sweep: number;
  decay: number;
  click: number;
  clickHz: number;
  clickDecay: number;
}

const DRUM: Record<DrumStroke, DrumSpec> = {
  // The downstroke: the full head, a fall from about 120 to 60 Hz.
  down: { from: 124, to: 60, sweep: 0.07, decay: 0.24, click: 0.35, clickHz: 1100, clickDecay: 0.028 },
  // The upstroke is lighter and catches the head higher.
  up: { from: 150, to: 82, sweep: 0.05, decay: 0.12, click: 0.45, clickHz: 1500, clickDecay: 0.02 },
  // A big low tom for the storm.
  tom: { from: 92, to: 44, sweep: 0.14, decay: 0.55, click: 0.25, clickHz: 520, clickDecay: 0.06 },
};

export function playDrum(
  host: NoteHost,
  dest: AudioNode,
  white: AudioBuffer,
  t: number,
  vel: number,
  stroke: DrumStroke,
  random: () => number,
): void {
  const ctx = host.ctx;
  const spec = DRUM[stroke];
  const tune = 0.96 + random() * 0.08;

  const osc = ctx.createOscillator();
  osc.frequency.setValueAtTime(spec.from * tune, t);
  osc.frequency.exponentialRampToValueAtTime(spec.to * tune, t + spec.sweep);
  const body = ctx.createGain();
  body.gain.setValueAtTime(0, t);
  body.gain.linearRampToValueAtTime(vel, t + 0.002);
  body.gain.exponentialRampToValueAtTime(vel * 1e-3, t + spec.decay);

  const noise = ctx.createBufferSource();
  noise.buffer = white;
  noise.loop = true;
  const band = ctx.createBiquadFilter();
  band.type = 'bandpass';
  band.frequency.value = spec.clickHz;
  band.Q.value = 0.9;
  const click = ctx.createGain();
  const clickLevel = Math.max(1e-4, vel * spec.click);
  click.gain.setValueAtTime(0, t);
  click.gain.linearRampToValueAtTime(clickLevel, t + 0.001);
  click.gain.exponentialRampToValueAtTime(clickLevel * 1e-3, t + spec.clickDecay);

  osc.connect(body);
  body.connect(dest);
  noise.connect(band);
  band.connect(click);
  click.connect(dest);

  const end = t + spec.decay + 0.02;
  osc.start(t);
  osc.stop(end);
  noise.start(t, random() * (white.duration - 0.5));
  noise.stop(t + spec.clickDecay + 0.02);
  host.track(osc, [body, noise, band, click]);
}

// --- bass ----------------------------------------------------------------------

export function playBass(
  host: NoteHost,
  dest: AudioNode,
  t: number,
  midi: number,
  dur: number,
  vel: number,
): void {
  const ctx = host.ctx;
  const osc = ctx.createOscillator();
  osc.type = 'sawtooth';
  osc.frequency.value = midiToHz(midi);
  const env = ctx.createGain();
  const off = t + Math.max(0.08, dur);
  env.gain.setValueAtTime(0, t);
  env.gain.linearRampToValueAtTime(vel, t + 0.03);
  env.gain.linearRampToValueAtTime(vel * 0.7, t + 0.12);
  env.gain.setValueAtTime(vel * 0.7, off);
  env.gain.setTargetAtTime(0, off, 0.05);
  osc.connect(env);
  env.connect(dest);
  osc.start(t);
  osc.stop(off + 0.35);
  host.track(osc, [env]);
}

// --- glass ---------------------------------------------------------------------

/** A slow pad tone: triangle plus a quiet octave, long attack and longer release. */
export function playGlass(
  host: NoteHost,
  dest: AudioNode,
  t: number,
  midi: number,
  dur: number,
  vel: number,
): void {
  const ctx = host.ctx;
  const f = midiToHz(midi);
  const osc = ctx.createOscillator();
  osc.type = 'triangle';
  osc.frequency.value = f;
  const air = ctx.createOscillator();
  air.frequency.value = f * 2.003;
  const airLevel = ctx.createGain();
  airLevel.gain.value = 0.22;

  const env = ctx.createGain();
  const attack = Math.min(1.2, dur * 0.4);
  const off = t + dur;
  env.gain.setValueAtTime(0, t);
  env.gain.linearRampToValueAtTime(vel, t + attack);
  env.gain.setValueAtTime(vel, off);
  env.gain.setTargetAtTime(0, off, 0.5);

  osc.connect(env);
  air.connect(airLevel);
  airLevel.connect(env);
  env.connect(dest);
  const end = off + 2.5;
  osc.start(t);
  osc.stop(end);
  air.start(t);
  air.stop(end);
  host.track(osc, [air, airLevel, env]);
}

const BELL_PARTIALS: readonly (readonly [number, number, number])[] = [
  [1, 1, 2.6],
  [2.76, 0.32, 0.9],
  [5.4, 0.1, 0.35],
];

export function playBell(host: NoteHost, dest: AudioNode, t: number, midi: number, vel: number): void {
  const ctx = host.ctx;
  const f = midiToHz(midi);
  const out = ctx.createGain();
  out.gain.value = vel;
  out.connect(dest);
  const nodes: AudioNode[] = [out];
  let first: OscillatorNode | null = null;
  let end = t;
  for (const [ratio, level, decay] of BELL_PARTIALS) {
    const osc = ctx.createOscillator();
    osc.frequency.value = f * ratio;
    const env = ctx.createGain();
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(level, t + 0.004);
    env.gain.exponentialRampToValueAtTime(level * 1e-3, t + decay);
    osc.connect(env);
    env.connect(out);
    osc.start(t);
    osc.stop(t + decay + 0.05);
    end = Math.max(end, t + decay + 0.05);
    if (first === null) first = osc;
    else nodes.push(osc);
    nodes.push(env);
  }
  if (first !== null) {
    first.stop(end);
    host.track(first, nodes);
  }
}

// --- the room ------------------------------------------------------------------

/**
 * A generated stereo impulse response: exponentially decaying noise that
 * darkens as it decays, after a short pre-delay and a few early reflections.
 *
 * A wooden hold rather than a hall — RT60 a little over two seconds, with the
 * top end gone by the second half. No file: the room is forty lines of maths.
 */
export function buildImpulse(
  ctx: BaseAudioContext,
  seconds: number,
  random: () => number,
): AudioBuffer {
  const sr = ctx.sampleRate;
  const length = Math.round(seconds * sr);
  const buffer = ctx.createBuffer(2, length, sr);
  const predelay = Math.round(0.012 * sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    let lp = 0;
    for (let i = predelay; i < length; i++) {
      const x = (i - predelay) / (length - predelay);
      const k = 0.85 - 0.72 * x;
      lp += k * (random() * 2 - 1 - lp);
      data[i] = lp * Math.exp(-6.9 * x);
    }
    // Early reflections off the planking.
    for (let r = 0; r < 6; r++) {
      const at = predelay + Math.round((0.004 + random() * 0.05) * sr);
      if (at < length) data[at] += (random() < 0.5 ? -1 : 1) * (0.5 - r * 0.06);
    }
  }
  return buffer;
}
