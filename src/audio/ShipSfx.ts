/**
 * The ship as an instrument: timber, canvas, blocks and the water against the
 * planking.
 *
 * Every sound here is an event, scheduled from what the hull is doing rather
 * than on a timer, because a creak that is not caused by anything is the
 * clearest sign of a loop:
 *
 *  - **Creaks** come from the *rate* the hull is rolling and pitching, weighted
 *    by how loaded she is (heel, canvas, wind). A ship lying still at anchor in a
 *    calm hardly creaks; one working through a beam sea talks the whole time.
 *    The sound is stick–slip friction: a pulse train whose rate *glides* (the
 *    slip speeding up or catching) through resonant wood modes. The glide is
 *    what makes it a creak and not a buzz.
 *  - **Groans** are the long, low version, fired when the roll reverses at a
 *    good angle — the moment the whole frame takes up the load the other way.
 *  - **Blocks and rigging** knock in the wind, more as she rolls and far more
 *    when the canvas is slatting.
 *  - **Luffing canvas** flogs: broadband slaps at a rate and strength set by how
 *    hard it is luffing, with the odd crack. When a luffing sail fills again it
 *    fills with a *whump*.
 *  - **Wave slaps and impacts** are hooks: the caller knows when water or a
 *    rock hit the hull; this knows what that sounds like.
 *
 * Everything goes to one `out` node, which `AudioSystem` gains by how close the
 * ship is and pans by where it is, so a ship on the far side of the bay is a
 * murmur from the left rather than a creak inside the listener's head.
 */

import { mulberry32 } from '../core/random';

/** What the system lends a one-shot: the context, the voice cap, cleanup. */
export interface SfxHost {
  readonly ctx: BaseAudioContext;
  /** Voice for an event the viewer caused or can see. */
  acquire(): boolean;
  /** Voice for ambience, held short of the cap so it cannot starve an event. */
  acquireAmbient(): boolean;
  track(source: AudioScheduledSourceNode, ...nodes: AudioNode[]): void;
}

/** Hull state, per frame. Angles in radians. */
export interface ShipMotion {
  /** Roll, positive with the starboard rail down. */
  heel: number;
  /** Pitch, positive bow up. */
  pitch: number;
  /** Canvas set, 0..1. */
  sailFill: number;
  /** How hard the canvas is luffing (flogging, not drawing), 0..1. */
  sailLuff: number;
  /** Wind speed, m/s. */
  windSpeed: number;
  /** 0 no ship / too far to hear, 1 close aboard. */
  presence: number;
}

const SEED = 0x5f1e5a;

// --- creaks ---------------------------------------------------------------------
/** Creaks per second per rad/s of hull motion, at full load. */
const CREAK_PER_MOTION = 6;
const CREAK_BASE_RATE = 0.06;
const CREAK_MAX_RATE = 2.5;
const CREAK_MIN_GAP = 0.22;
const CREAK_LEVEL = 0.9;
/** Roll angle at which a reversal groans, radians (about 3 degrees). */
const GROAN_HEEL = 0.05;
const GROAN_COOLDOWN = 2.8;
const GROAN_LEVEL = 1.25;

// --- blocks -------------------------------------------------------------------------
const BLOCK_BASE_RATE = 0.05;
const BLOCK_PER_MOTION = 2;
const BLOCK_PER_LUFF = 1.4;
const BLOCK_MAX_RATE = 1.8;
const BLOCK_MIN_GAP = 0.35;
const BLOCK_LEVEL = 0.22;

// --- canvas -------------------------------------------------------------------------
const FLAP_BASE_RATE = 2.2;
const FLAP_PER_LUFF = 7;
const FLAP_MIN_GAP = 0.07;
const FLAP_LEVEL = 0.34;
const FILL_LEVEL = 0.6;

// --- water on the hull ------------------------------------------------------------
const SLAP_LEVEL = 0.55;
const IMPACT_LEVEL = 0.6;

/** Seconds the motion rates are smoothed over: frame-to-frame jitter is not motion. */
const RATE_TAU = 0.12;

export class ShipSfx {
  private readonly ctx: BaseAudioContext;
  private readonly host: SfxHost;
  private readonly white: AudioBuffer;
  private readonly out: AudioNode;
  private readonly pulse: PeriodicWave;
  private readonly knocks: AudioBuffer[];

  private random = mulberry32(SEED);
  private primed = false;
  private lastHeel = 0;
  private lastPitch = 0;
  private heelRate = 0;
  private pitchRate = 0;
  private creakTimer = 0;
  private blockTimer = 0;
  private flapTimer = 0;
  private groanCooldown = 0;
  private luffMemory = 0;

  constructor(host: SfxHost, white: AudioBuffer, out: AudioNode) {
    this.ctx = host.ctx;
    this.host = host;
    this.white = white;
    this.out = out;
    this.pulse = buildPulseWave(this.ctx);
    const random = mulberry32(SEED ^ 0x77);
    this.knocks = [
      renderKnock(this.ctx, 640, 1730, random),
      renderKnock(this.ctx, 820, 2140, random),
      renderKnock(this.ctx, 1010, 2600, random),
    ];
  }

  /** Smoothed roll rate, rad/s. Exposed because the bow wash and the timbers read the same motion. */
  get rollRate(): number {
    return this.heelRate;
  }

  /** Smoothed pitch rate, rad/s, positive bow rising. */
  get pitchingRate(): number {
    return this.pitchRate;
  }

  update(dt: number, now: number, live: boolean, m: Readonly<ShipMotion>): void {
    if (!(dt > 0)) return;
    if (!this.primed) {
      this.primed = true;
      this.lastHeel = m.heel;
      this.lastPitch = m.pitch;
    }
    const k = 1 - Math.exp(-dt / RATE_TAU);
    const heelRate = (m.heel - this.lastHeel) / dt;
    const pitchRate = (m.pitch - this.lastPitch) / dt;
    const previousHeelRate = this.heelRate;
    // Clamped before smoothing: a teleport (reset, tour cut) is not a roll.
    this.heelRate += (clamp(heelRate, -2, 2) - this.heelRate) * k;
    this.pitchRate += (clamp(pitchRate, -2, 2) - this.pitchRate) * k;
    this.lastHeel = m.heel;
    this.lastPitch = m.pitch;
    this.groanCooldown = Math.max(0, this.groanCooldown - dt);

    const presence = clamp(m.presence, 0, 1);
    if (!live || presence < 0.04) {
      this.creakTimer = Math.max(this.creakTimer, 0.5);
      return;
    }

    const random = this.random;
    const windDrive = clamp(m.windSpeed / 14, 0, 1.3);
    const motion = Math.abs(this.heelRate) + 0.7 * Math.abs(this.pitchRate);
    const load =
      0.35 + 0.4 * clamp(Math.abs(m.heel) / 0.3, 0, 1) + 0.25 * m.sailFill * Math.min(1, windDrive);
    const luff = clamp(m.sailLuff, 0, 1);

    // --- creaks ------------------------------------------------------------
    const creakRate = Math.min(
      CREAK_MAX_RATE,
      CREAK_BASE_RATE + CREAK_PER_MOTION * motion * load,
    );
    this.creakTimer -= dt;
    if (this.creakTimer <= 0) {
      const strength = clamp(0.25 + motion * 2.5 * load, 0, 1) * (0.6 + 0.4 * random());
      this.creak(strength, now + random() * 0.05, false);
      this.creakTimer = Math.max(CREAK_MIN_GAP, -Math.log(1 - random()) / creakRate);
    }

    // --- groans: the roll reversing under load ------------------------------
    if (
      this.groanCooldown === 0 &&
      Math.sign(previousHeelRate) !== Math.sign(this.heelRate) &&
      Math.abs(m.heel) > GROAN_HEEL &&
      random() < 0.55
    ) {
      this.groanCooldown = GROAN_COOLDOWN;
      this.creak(clamp(Math.abs(m.heel) / 0.2, 0.3, 1), now + 0.02, true);
    }

    // --- blocks and rigging --------------------------------------------------
    const blockRate = Math.min(
      BLOCK_MAX_RATE,
      windDrive * (BLOCK_BASE_RATE + BLOCK_PER_MOTION * motion) + BLOCK_PER_LUFF * luff,
    );
    if (blockRate > 0.01) {
      this.blockTimer -= dt;
      if (this.blockTimer <= 0) {
        this.blockRattle(clamp(0.35 + 0.4 * windDrive + 0.3 * luff, 0, 1), now + random() * 0.05);
        this.blockTimer = Math.max(BLOCK_MIN_GAP, -Math.log(1 - random()) / blockRate);
      }
    }

    // --- canvas -----------------------------------------------------------------
    if (luff > 0.04) {
      this.flapTimer -= dt;
      let offset = 0;
      while (this.flapTimer <= 0) {
        const strength = luff * (0.55 + 0.45 * Math.min(1, windDrive)) * (0.6 + 0.4 * random());
        this.flap(strength, now + offset, random() < 0.2 * luff);
        offset += FLAP_MIN_GAP;
        const rate = FLAP_BASE_RATE + FLAP_PER_LUFF * luff;
        this.flapTimer += Math.max(FLAP_MIN_GAP, -Math.log(1 - random()) / rate);
      }
    } else if (this.flapTimer < 0) {
      this.flapTimer = 0;
    }
    // A sail that was flogging and is suddenly drawing has just filled.
    if (luff < 0.15 && this.luffMemory > 0.45 && m.sailFill > 0.3) {
      this.sailFill(clamp(this.luffMemory * windDrive, 0.3, 1), now);
      this.luffMemory = 0;
    }
    this.luffMemory = Math.max(luff, this.luffMemory - dt * 0.6);
  }

  reset(): void {
    this.random = mulberry32(SEED);
    this.primed = false;
    this.heelRate = 0;
    this.pitchRate = 0;
    this.creakTimer = 0.5;
    this.blockTimer = 0;
    this.flapTimer = 0;
    this.groanCooldown = 0;
    this.luffMemory = 0;
  }

  // ------------------------------------------------------------------ voices

  /**
   * A creak (or, with `groan`, a timber groan): a stick–slip pulse train with a
   * gliding rate through two wood resonances.
   */
  creak(strength: number, when: number, groan: boolean): void {
    if (!(groan ? this.host.acquire() : this.host.acquireAmbient())) return;
    const ctx = this.ctx;
    const random = this.random;
    const s = clamp(strength, 0, 1);

    const dur = groan ? 0.8 + 1.1 * random() : 0.22 + 0.6 * random();
    const rate0 = groan ? 9 + 9 * random() : 26 + 46 * random();
    // Either the slip catches (rate falls) or it lets go (rate climbs).
    const rate1 = rate0 * (random() < 0.5 ? 0.55 + 0.3 * random() : 1.25 + 0.5 * random());
    const f1 = groan ? 150 + 110 * random() : 340 + 380 * random();
    const f2 = groan ? 400 + 220 * random() : 880 + 700 * random();
    const level = (groan ? GROAN_LEVEL : CREAK_LEVEL) * (0.35 + 0.65 * s);

    const osc = ctx.createOscillator();
    osc.setPeriodicWave(this.pulse);
    osc.frequency.setValueAtTime(rate0, when);
    osc.frequency.exponentialRampToValueAtTime(rate1, when + dur);

    const bp1 = ctx.createBiquadFilter();
    bp1.type = 'bandpass';
    bp1.frequency.value = f1;
    bp1.Q.value = groan ? 6 : 9;
    const bp2 = ctx.createBiquadFilter();
    bp2.type = 'bandpass';
    bp2.frequency.value = f2;
    bp2.Q.value = groan ? 5 : 7;
    const upper = ctx.createGain();
    upper.gain.value = 0.55;

    const env = ctx.createGain();
    const g = env.gain;
    g.setValueAtTime(0, when);
    g.linearRampToValueAtTime(level, when + dur * 0.35);
    g.linearRampToValueAtTime(level * 0.7, when + dur * 0.7);
    g.linearRampToValueAtTime(0, when + dur);

    osc.connect(bp1);
    osc.connect(bp2);
    bp2.connect(upper);
    bp1.connect(env);
    upper.connect(env);
    const pan = this.panner((random() * 2 - 1) * 0.5);
    env.connect(pan ?? this.out);
    if (pan !== null) pan.connect(this.out);

    osc.start(when);
    osc.stop(when + dur + 0.02);
    if (pan !== null) this.host.track(osc, bp1, bp2, upper, env, pan);
    else this.host.track(osc, bp1, bp2, upper, env);
  }

  /** Two to four wooden knocks in quick succession: a block swinging against the mast. */
  blockRattle(strength: number, when: number): void {
    if (!this.host.acquireAmbient()) return;
    const ctx = this.ctx;
    const random = this.random;
    const out = ctx.createGain();
    out.gain.value = BLOCK_LEVEL * clamp(strength, 0, 1);
    const pan = this.panner((random() * 2 - 1) * 0.6);
    out.connect(pan ?? this.out);
    if (pan !== null) pan.connect(this.out);

    const count = 2 + Math.floor(random() * 3);
    const nodes: AudioNode[] = [out];
    if (pan !== null) nodes.push(pan);
    let t = when;
    let last: AudioBufferSourceNode | null = null;
    for (let i = 0; i < count; i++) {
      const src = ctx.createBufferSource();
      src.buffer = this.knocks[Math.floor(random() * this.knocks.length)];
      src.playbackRate.value = 0.92 + 0.16 * random();
      // Each bounce is softer: the block is coming to rest.
      const g = ctx.createGain();
      g.gain.value = Math.pow(0.62, i) * (0.8 + 0.4 * random());
      src.connect(g);
      g.connect(out);
      src.start(t);
      if (last !== null) nodes.push(last);
      nodes.push(g);
      last = src;
      t += 0.045 + 0.07 * random() * (1 + i * 0.3);
    }
    if (last !== null) this.host.track(last, ...nodes);
  }

  /** One slap of luffing canvas; `crack` is the bright report of a sail snapping taut. */
  flap(strength: number, when: number, crack: boolean): void {
    if (!this.host.acquireAmbient()) return;
    const ctx = this.ctx;
    const random = this.random;
    const s = clamp(strength, 0, 1);
    const src = ctx.createBufferSource();
    src.buffer = this.white;
    src.loop = true;
    src.playbackRate.value = 0.8 + 0.3 * random();

    const band = ctx.createBiquadFilter();
    band.type = crack ? 'highpass' : 'bandpass';
    const dur = crack ? 0.05 + 0.04 * random() : 0.07 + 0.1 * random();
    if (crack) {
      band.frequency.value = 1100;
      band.Q.value = 0.7;
    } else {
      band.Q.value = 0.9;
      band.frequency.setValueAtTime(700 + 500 * random(), when);
      band.frequency.exponentialRampToValueAtTime(260 + 120 * random(), when + dur);
    }
    const env = ctx.createGain();
    const level = Math.max(1e-4, FLAP_LEVEL * (crack ? 1.3 : 1) * (0.25 + 0.75 * s));
    env.gain.setValueAtTime(0, when);
    env.gain.linearRampToValueAtTime(level, when + (crack ? 0.002 : 0.006));
    env.gain.exponentialRampToValueAtTime(level * 1e-3, when + dur);

    src.connect(band);
    band.connect(env);
    const pan = this.panner((random() * 2 - 1) * 0.35);
    env.connect(pan ?? this.out);
    if (pan !== null) pan.connect(this.out);
    src.start(when, random() * (this.white.duration - 0.5));
    src.stop(when + dur + 0.02);
    if (pan !== null) this.host.track(src, band, env, pan);
    else this.host.track(src, band, env);
  }

  /** The whump of canvas filling, and the crack as it comes taut. */
  sailFill(strength: number, when: number): void {
    if (!this.host.acquire()) return;
    const ctx = this.ctx;
    const s = clamp(strength, 0, 1);
    const src = ctx.createBufferSource();
    src.buffer = this.white;
    src.loop = true;
    const low = ctx.createBiquadFilter();
    low.type = 'lowpass';
    low.frequency.value = 240;
    low.Q.value = 1.2;
    const env = ctx.createGain();
    const level = FILL_LEVEL * (0.4 + 0.6 * s);
    env.gain.setValueAtTime(0, when);
    env.gain.linearRampToValueAtTime(level, when + 0.04);
    env.gain.exponentialRampToValueAtTime(level * 1e-3, when + 0.45);
    src.connect(low);
    low.connect(env);
    env.connect(this.out);
    src.start(when, this.random() * (this.white.duration - 1));
    src.stop(when + 0.5);
    this.host.track(src, low, env);
    this.flap(s, when + 0.03, true);
  }

  /**
   * Water slapping the hull: a bright smack whose band falls as the water
   * sheets away, over the hollow boom of the planking.
   */
  waveSlap(strength: number, pan: number, when: number): void {
    const s = clamp(strength, 0, 1);
    if (s < 0.03 || !this.host.acquire()) return;
    const ctx = this.ctx;
    const random = this.random;
    const dur = 0.09 + 0.12 * s;

    const src = ctx.createBufferSource();
    src.buffer = this.white;
    src.loop = true;
    src.playbackRate.value = 0.85 + 0.3 * random();
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.Q.value = 1.1;
    band.frequency.setValueAtTime(1300 + 500 * s, when);
    band.frequency.exponentialRampToValueAtTime(360, when + dur);
    const smack = ctx.createGain();
    const level = SLAP_LEVEL * Math.pow(s, 0.8);
    smack.gain.setValueAtTime(0, when);
    smack.gain.linearRampToValueAtTime(level, when + 0.004);
    smack.gain.exponentialRampToValueAtTime(level * 1e-3, when + dur);

    const boom = ctx.createOscillator();
    boom.frequency.setValueAtTime(92 + 20 * random(), when);
    boom.frequency.exponentialRampToValueAtTime(60, when + 0.16);
    const boomGain = ctx.createGain();
    const boomLevel = level * 0.8 * s;
    boomGain.gain.setValueAtTime(0, when);
    boomGain.gain.linearRampToValueAtTime(Math.max(1e-4, boomLevel), when + 0.006);
    boomGain.gain.exponentialRampToValueAtTime(Math.max(1e-7, boomLevel * 1e-3), when + 0.22);

    const panner = this.panner(clamp(pan, -1, 1) * 0.8);
    const dest = panner ?? this.out;
    src.connect(band);
    band.connect(smack);
    smack.connect(dest);
    boom.connect(boomGain);
    boomGain.connect(dest);
    if (panner !== null) panner.connect(this.out);

    src.start(when, random() * (this.white.duration - 0.5));
    src.stop(when + dur + 0.02);
    boom.start(when);
    boom.stop(when + 0.25);
    if (panner !== null) this.host.track(boom, src, band, smack, boomGain, panner);
    else this.host.track(boom, src, band, smack, boomGain);
  }

  /**
   * The hull hitting something: a deep thud through the whole frame, the timbers
   * answering, and with `grind` the long scrape of a keel over rock or sand.
   */
  hullImpact(strength: number, grind: boolean, when: number): void {
    const s = clamp(strength, 0, 1);
    if (s < 0.02 || !this.host.acquire()) return;
    const ctx = this.ctx;
    const random = this.random;
    const level = IMPACT_LEVEL * Math.pow(s, 0.7);

    const thud = ctx.createOscillator();
    thud.frequency.setValueAtTime(64, when);
    thud.frequency.exponentialRampToValueAtTime(34, when + 0.3);
    const thudGain = ctx.createGain();
    thudGain.gain.setValueAtTime(0, when);
    thudGain.gain.linearRampToValueAtTime(level, when + 0.008);
    thudGain.gain.exponentialRampToValueAtTime(level * 1e-3, when + 0.7);

    const src = ctx.createBufferSource();
    src.buffer = this.white;
    src.loop = true;
    const low = ctx.createBiquadFilter();
    low.type = 'lowpass';
    low.frequency.value = 320;
    low.Q.value = 0.8;
    const body = ctx.createGain();
    body.gain.setValueAtTime(0, when);
    body.gain.linearRampToValueAtTime(level * 0.7, when + 0.004);
    body.gain.exponentialRampToValueAtTime(level * 7e-4, when + 0.4);

    thud.connect(thudGain);
    thudGain.connect(this.out);
    src.connect(low);
    low.connect(body);
    body.connect(this.out);

    let end = when + 0.75;
    let grindGate: GainNode | null = null;
    let grindBand: BiquadFilterNode | null = null;
    if (grind) {
      // Gated band noise: the keel catching and releasing as it drags.
      const seconds = 0.7 + 0.9 * s;
      grindBand = ctx.createBiquadFilter();
      grindBand.type = 'bandpass';
      grindBand.frequency.value = 520 + 300 * random();
      grindBand.Q.value = 1.4;
      grindGate = ctx.createGain();
      const gg = grindGate.gain;
      gg.setValueAtTime(0, when);
      let t = when + 0.02;
      while (t < when + seconds) {
        const fade = 1 - (t - when) / seconds;
        gg.linearRampToValueAtTime(level * 0.5 * fade * (0.4 + 0.6 * random()), t);
        t += 0.03 + 0.05 * random();
      }
      gg.linearRampToValueAtTime(0, when + seconds + 0.05);
      src.connect(grindBand);
      grindBand.connect(grindGate);
      grindGate.connect(this.out);
      end = Math.max(end, when + seconds + 0.1);
    }

    src.start(when, random() * (this.white.duration - 2));
    src.stop(end);
    thud.start(when);
    thud.stop(end);
    const nodes: AudioNode[] = [thudGain, src, low, body];
    if (grindGate !== null && grindBand !== null) nodes.push(grindBand, grindGate);
    this.host.track(thud, ...nodes);

    // The frame answers the blow.
    this.creak(Math.min(1, s * 1.2), when + 0.06, true);
  }

  private panner(pan: number): StereoPannerNode | null {
    const factory = (this.ctx as BaseAudioContext & { createStereoPanner?: () => StereoPannerNode })
      .createStereoPanner;
    if (typeof factory !== 'function') return null;
    const node = this.ctx.createStereoPanner();
    node.pan.value = pan;
    return node;
  }
}

/**
 * An impulse train: every harmonic at the same level. Through a resonant filter
 * each pulse rings the filter once, which is a tick; a train of them at tens of
 * hertz is a creak.
 */
function buildPulseWave(ctx: BaseAudioContext): PeriodicWave {
  const n = 64;
  const real = new Float32Array(n + 1);
  const imag = new Float32Array(n + 1);
  for (let h = 1; h <= n; h++) real[h] = 1;
  return ctx.createPeriodicWave(real, imag);
}

/**
 * A wooden knock: two inharmonic damped modes plus a click, rendered once. The
 * upper mode dies far faster than the lower, which is what makes it wood and
 * not a bell.
 */
function renderKnock(
  ctx: BaseAudioContext,
  f1: number,
  f2: number,
  random: () => number,
): AudioBuffer {
  const sr = ctx.sampleRate;
  const length = Math.round(0.12 * sr);
  const buffer = ctx.createBuffer(1, length, sr);
  const data = buffer.getChannelData(0);
  const tau1 = 0.028;
  const tau2 = 0.009;
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const t = i / sr;
    const click = i < 48 ? (random() * 2 - 1) * (1 - i / 48) * 0.6 : 0;
    const v =
      Math.sin(2 * Math.PI * f1 * t) * Math.exp(-t / tau1) +
      0.6 * Math.sin(2 * Math.PI * f2 * t) * Math.exp(-t / tau2) +
      click;
    data[i] = v;
    peak = Math.max(peak, Math.abs(v));
  }
  const scale = peak > 0 ? 0.9 / peak : 0;
  for (let i = 0; i < length; i++) data[i] *= scale;
  return buffer;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
