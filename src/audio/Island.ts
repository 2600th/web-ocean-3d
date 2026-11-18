import { mulberry32 } from '../core/random';
import { NoiseBed } from './NoiseBed';
import { ParamTarget } from './ParamTarget';
import type { Spatial } from './Spatial';

/**
 * The island heard: surf on the shore, the camp fire, land birds by day and
 * insects by night (A8). It is heard before it is reached.
 *
 *  - **Shore surf.** Breakers are not the open sea's hiss: each wave builds,
 *    crashes and drains, so the level is a train of peaks on the swell period
 *    rather than a breathing bed. Louder as the breakers get closer (1/d from
 *    `SHORE_REF`), and darker with distance, since air takes the top off a
 *    crash long before it takes the body. Panned to the bearing of the nearest
 *    stretch of coast.
 *  - **The camp fire.** A low roar and a scatter of crackles, at the fire's
 *    own place, head-related, from a few tens of metres out.
 *  - **Land birds** call from the trees by day, **insects** chirr from them by
 *    night, both only within a few hundred metres of the shore.
 *
 * The coast is found, not assumed: a few times a second the ground height the
 * caller supplies is marched outward from the island's centre along the line
 * to the listener, and where it crosses the sea is the shore on that bearing.
 * Allocation-free except for the one-shots, which are events.
 */

/** The island, as `AudioSystem.setIsland` takes it. */
export interface AudioIsland {
  x: number;
  z: number;
  /** Mean shoreline radius, metres: where to start looking for the coast. */
  radius: number;
  /** The camp fire, world metres, or null for none. */
  fire?: { x: number; y: number; z: number } | null;
  /** Ground height at (x, z), metres above the sea; negative under it. */
  height?: (x: number, z: number) => number;
}

/** What the island borrows from the system: voices, cleanup, the noise. */
export interface IslandHost {
  readonly ctx: BaseAudioContext;
  readonly white: AudioBuffer;
  readonly brown: AudioBuffer;
  readonly spatial: Spatial;
  /** Voice for ambience, held short of the cap. */
  acquireAmbient(): boolean;
  track(source: AudioScheduledSourceNode, ...nodes: AudioNode[]): void;
  addBed(bed: NoiseBed): void;
}

/** What drives it each frame. */
export interface IslandFrame {
  now: number;
  dt: number;
  live: boolean;
  /** 1 in full day, 0 at night. */
  day: number;
  /** 1 at night, 0 in daylight. */
  night: number;
  waveHeight: number;
  /** Swell period, seconds. */
  swellPeriod: number;
  wind: number;
  rain: number;
  /** 1 above water, 0 under. */
  air: number;
  clock: number;
}

const SEED = 0x15_1a_4d;

/** Breakers within this are at full level, metres; past it 1/d. */
const SHORE_REF = 40;
/** Past this the shore is silent and its bed parks. */
const SHORE_SILENT = 1600;
const SHORE_LEVEL = 0.34;
/** Brightness: corner = BRIGHT * exp(-d / ABSORB) + DULL. */
const SHORE_BRIGHT_HZ = 2200;
const SHORE_ABSORB_M = 260;
const SHORE_DULL_HZ = 260;
/** How often the coast is looked for, seconds, and the march's step, metres. */
const SHORE_SEARCH_EVERY = 0.25;
const SHORE_STEP = 8;
/** Breakers stand this far off the waterline, metres. */
const BREAKER_OFFSET = 25;

const FIRE_REF = 6;
const FIRE_SILENT = 160;
const FIRE_ROAR_LEVEL = 0.5;
const FIRE_CRACKLE_RATE = 5;
const FIRE_CRACKLE_LEVEL = 0.5;

/** Birds and insects are heard this close to the shore, metres, and fade out by this. */
const LAND_NEAR = 60;
const LAND_FAR = 420;
const BIRD_RATE = 0.35;
const BIRD_LEVEL = 0.07;
const INSECT_LEVEL = 0.05;
const INSECT_HZ = 4300;
/** The chirp gate's rate, Hz: a field cricket's chirps a second at a warm night. */
const INSECT_CHIRP_HZ = 2.6;

export class Island {
  private readonly host: IslandHost;
  private readonly ctx: BaseAudioContext;
  private readonly destination: AudioNode;
  private desc: AudioIsland | null = null;
  private random = mulberry32(SEED);

  private readonly shore: NoiseBed;
  private readonly roar: NoiseBed;
  private readonly insects: NoiseBed;
  /** The fire: roar and crackles into an HRTF panner at the camp. */
  private readonly fireBus: GainNode;
  private readonly firePanner: PannerNode | null;
  private readonly fireGain: ParamTarget;
  /** The insects' chirp gate: a pulse LFO on a gain. */
  private readonly insectGate: GainNode;
  private readonly insectLfo: OscillatorNode;
  private readonly insectDepth: GainNode;
  private lfoStarted = false;
  private readonly crackles: AudioBuffer[];

  private searchTimer = 0;
  /** Nearest breakers: where, and how far from the listener. */
  private shoreX = 0;
  private shoreZ = 0;
  private shoreDistance = Number.POSITIVE_INFINITY;
  private crackleTimer = 0;
  private birdTimer = 3;

  constructor(host: IslandHost, destination: AudioNode, wet: AudioNode) {
    this.host = host;
    const ctx = host.ctx;
    this.ctx = ctx;
    this.destination = destination;

    this.shore = new NoiseBed(ctx, {
      buffer: host.white,
      destination,
      filter: 'lowpass',
      frequency: 900,
      q: 0.6,
      highpass: 70,
      playbackRate: 0.87,
      offset: 1.3,
      pan: 0,
      gainTau: 0.09,
      filterTau: 0.4,
    });
    host.addBed(this.shore);

    this.fireBus = ctx.createGain();
    this.fireBus.gain.value = 0;
    this.fireGain = new ParamTarget(this.fireBus.gain, 0, 0.3, 0.002, 0, 1);
    this.firePanner = host.spatial.panner(0, 0, 0, 'hrtf');
    if (this.firePanner !== null) {
      this.fireBus.connect(this.firePanner);
      this.firePanner.connect(destination);
    } else {
      this.fireBus.connect(destination);
    }
    const send = ctx.createGain();
    send.gain.value = 0.25;
    this.fireBus.connect(send);
    send.connect(wet);
    this.roar = new NoiseBed(ctx, {
      buffer: host.brown,
      spectrum: 'brown',
      destination: this.fireBus,
      filter: 'lowpass',
      frequency: 520,
      q: 0.7,
      highpass: 90,
      playbackRate: 1.21,
      offset: 2.1,
      pan: 0,
      gainTau: 0.15,
    });
    host.addBed(this.roar);

    this.insectGate = ctx.createGain();
    this.insectGate.gain.value = 0.5;
    this.insectGate.connect(destination);
    this.insects = new NoiseBed(ctx, {
      buffer: host.white,
      destination: this.insectGate,
      filter: 'bandpass',
      frequency: INSECT_HZ,
      q: 5,
      highpass: 2000,
      playbackRate: 1.31,
      offset: 0.7,
      pan: 0,
      gainTau: 0.5,
    });
    host.addBed(this.insects);
    // The gate: a pulse train at the chirp rate, 0..1, so the chorus chirps.
    this.insectLfo = ctx.createOscillator();
    this.insectLfo.setPeriodicWave(chirpWave(ctx));
    this.insectLfo.frequency.value = INSECT_CHIRP_HZ;
    this.insectDepth = ctx.createGain();
    this.insectDepth.gain.value = 0.5;
    this.insectLfo.connect(this.insectDepth);
    this.insectDepth.connect(this.insectGate.gain);

    const random = mulberry32(SEED ^ 0x3c);
    this.crackles = [0, 1, 2, 3].map(() => renderCrackle(ctx, random));
  }

  /** Every target here, for the system's flush. */
  get targets(): ParamTarget[] {
    return [this.fireGain];
  }

  set(desc: AudioIsland | null): void {
    this.desc = desc;
    this.searchTimer = 0;
    const fire = desc?.fire;
    if (fire && this.firePanner !== null) {
      const p = this.firePanner;
      if (p.positionX !== undefined) {
        p.positionX.value = fire.x;
        p.positionY.value = fire.y;
        p.positionZ.value = fire.z;
      }
    }
  }

  /** Logical loudness of the island's beds, for the window (`HdrMix`). */
  loudnessDb(): number {
    const a = this.shore.loudnessDb();
    const b = this.roar.loudnessDb() + 20 * Math.log10(Math.max(1e-6, this.fireGain.target));
    const hi = Math.max(a, b);
    if (hi === Number.NEGATIVE_INFINITY) return hi;
    return hi + 10 * Math.log10(1 + Math.pow(10, (Math.min(a, b) - hi) / 10));
  }

  /** Metres from the listener to the nearest breakers, Infinity with no island. */
  get distanceToShore(): number {
    return this.shoreDistance;
  }

  /** Starts the chirp gate with the beds, on the first resume. */
  start(when: number): void {
    if (this.lfoStarted) return;
    this.lfoStarted = true;
    this.insectLfo.start(when);
  }

  reset(): void {
    this.random = mulberry32(SEED);
    this.crackleTimer = 0;
    this.birdTimer = 3;
    this.searchTimer = 0;
  }

  update(f: Readonly<IslandFrame>): void {
    const desc = this.desc;
    const sp = this.host.spatial;
    if (desc === null) {
      this.shore.setLevel(0, f.now, f.live);
      this.roar.setLevel(0, f.now, f.live);
      this.insects.setLevel(0, f.now, f.live);
      this.fireGain.set(0, f.now, f.live);
      this.shoreDistance = Number.POSITIVE_INFINITY;
      return;
    }

    this.searchTimer -= f.dt;
    if (this.searchTimer <= 0) {
      this.searchTimer = SHORE_SEARCH_EVERY;
      this.findShore(desc, sp.x, sp.z);
    }
    const d = this.shoreDistance;

    // --- shore surf: a crash per wave, on the swell ----------------------------
    const near = Math.min(1, SHORE_REF / Math.max(SHORE_REF, d));
    const audible = d < SHORE_SILENT ? 1 - smoothstep(d, SHORE_SILENT * 0.6, SHORE_SILENT) : 0;
    const period = Math.max(3, f.swellPeriod);
    const phase = (f.clock / period) % 1;
    // Build, crash at a fifth of the period, drain: a skewed peak.
    const crash = phase < 0.2 ? Math.pow(phase / 0.2, 2) : Math.exp(-(phase - 0.2) * 4.5);
    const strength = 0.35 + 0.65 * Math.min(1, f.waveHeight / 2.5);
    this.shore.setLevel(SHORE_LEVEL * near * audible * strength * (0.35 + 0.65 * crash) * f.air, f.now, f.live);
    this.shore.setBand(SHORE_DULL_HZ + SHORE_BRIGHT_HZ * Math.exp(-d / SHORE_ABSORB_M) * (0.6 + 0.4 * crash), 0.6, f.now, f.live);
    this.shore.setPan(0.8 * sp.panOf(this.shoreX, this.shoreZ), f.now, f.live);

    // --- the camp fire -----------------------------------------------------------
    const fire = desc.fire;
    let fireNear = 0;
    if (fire) {
      const fd = Math.hypot(fire.x - sp.x, fire.y - sp.y, fire.z - sp.z);
      fireNear = fd < FIRE_SILENT ? Math.min(1, FIRE_REF / Math.max(FIRE_REF, fd)) * (1 - smoothstep(fd, FIRE_SILENT * 0.6, FIRE_SILENT)) : 0;
      // Rain damps a fire's crackle more than its roar.
      this.fireGain.set(fireNear * f.air, f.now, f.live);
      this.roar.setLevel(fireNear > 0 ? FIRE_ROAR_LEVEL * (0.8 + 0.2 * Math.sin(f.clock * 2.3) * Math.sin(f.clock * 0.7)) : 0, f.now, f.live);
      if (f.live && fireNear > 0.01) {
        this.crackleTimer -= f.dt;
        let offset = 0;
        while (this.crackleTimer <= 0) {
          this.crackle(f.now + offset, (1 - 0.7 * f.rain) * (0.4 + 0.6 * this.random()));
          offset += 0.03;
          this.crackleTimer += Math.max(0.03, -Math.log(1 - this.random()) / FIRE_CRACKLE_RATE);
        }
      }
    } else {
      this.roar.setLevel(0, f.now, f.live);
      this.fireGain.set(0, f.now, f.live);
    }

    // --- land: birds by day, insects by night -------------------------------------
    const land = (1 - smoothstep(d, LAND_NEAR, LAND_FAR)) * f.air * (1 - smoothstep(f.wind, 10, 18)) * (1 - smoothstep(f.rain, 0.2, 0.7));
    this.insects.setLevel(INSECT_LEVEL * land * f.night, f.now, f.live);
    this.insects.setPan(0.6 * sp.panOf(desc.x, desc.z), f.now, f.live);
    const birdRate = BIRD_RATE * land * f.day;
    if (f.live && birdRate > 1e-3) {
      this.birdTimer -= f.dt;
      if (this.birdTimer <= 0) {
        this.bird(f.now, d);
        this.birdTimer = 0.8 + -Math.log(1 - this.random()) / birdRate;
      }
    } else if (this.birdTimer < 1) {
      this.birdTimer = 1;
    }
  }

  dispose(): void {
    this.fireBus.disconnect();
    this.firePanner?.disconnect();
    this.insectGate.disconnect();
    this.insectDepth.disconnect();
    if (this.lfoStarted) {
      try {
        this.insectLfo.stop();
      } catch {
        // Already stopped.
      }
    }
    this.insectLfo.disconnect();
  }

  /** The coast on the bearing to the listener, by marching the ground height outward. */
  private findShore(desc: AudioIsland, lx: number, lz: number): void {
    const dx = lx - desc.x;
    const dz = lz - desc.z;
    const r = Math.hypot(dx, dz);
    if (r < 1e-3) {
      this.shoreDistance = 0;
      return;
    }
    const ux = dx / r;
    const uz = dz / r;
    let coast = desc.radius;
    const height = desc.height;
    if (height !== undefined) {
      // From well inside the island outward: the first point under the sea.
      coast = desc.radius * 1.4;
      for (let s = desc.radius * 0.5; s <= desc.radius * 1.4; s += SHORE_STEP) {
        if (height(desc.x + ux * s, desc.z + uz * s) < 0) {
          coast = s;
          break;
        }
      }
    }
    const breakers = coast + BREAKER_OFFSET;
    this.shoreX = desc.x + ux * breakers;
    this.shoreZ = desc.z + uz * breakers;
    // The listener on the beach or inland is among the breakers, not past them.
    this.shoreDistance = Math.max(0, r - breakers);
  }

  private crackle(when: number, strength: number): void {
    if (!this.host.acquireAmbient()) return;
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.crackles[Math.floor(this.random() * this.crackles.length)];
    src.playbackRate.value = 0.7 + 0.8 * this.random();
    const g = ctx.createGain();
    g.gain.value = FIRE_CRACKLE_LEVEL * strength;
    src.connect(g);
    g.connect(this.fireBus);
    src.start(when);
    this.host.track(src, g);
  }

  /**
   * One land bird: a whistled phrase of two to five notes, each a quick
   * sweep, from somewhere in the trees on the island's side.
   */
  private bird(now: number, shoreDistance: number): void {
    if (!this.host.acquireAmbient() || this.desc === null) return;
    const ctx = this.ctx;
    const random = this.random;
    const sp = this.host.spatial;
    const distance = shoreDistance + 30 + 120 * random();
    const level = BIRD_LEVEL * Math.min(1, 40 / distance);
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    const gain = ctx.createGain();
    const notes = 2 + Math.floor(random() * 4);
    const base = 2200 + 1800 * random();
    const kind = random();
    let t = now + 0.02;
    gain.gain.setValueAtTime(0, now);
    for (let i = 0; i < notes; i++) {
      const dur = 0.06 + 0.12 * random();
      const from = base * (kind < 0.5 ? 0.85 + 0.3 * random() : 1.15 - 0.1 * i);
      const to = from * (kind < 0.5 ? 1.25 : 0.78);
      osc.frequency.setValueAtTime(from, t);
      osc.frequency.exponentialRampToValueAtTime(to, t + dur);
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(level, t + 0.012);
      gain.gain.setValueAtTime(level, t + dur * 0.6);
      gain.gain.linearRampToValueAtTime(0, t + dur);
      t += dur + 0.04 + 0.1 * random();
    }
    const desc = this.desc;
    const pan = createPan(ctx, Math.max(-1, Math.min(1, 0.8 * sp.panOf(desc.x, desc.z) + 0.3 * (random() * 2 - 1))));
    gain.connect(pan ?? this.destination);
    osc.connect(gain);
    if (pan !== null) pan.connect(this.destination);
    osc.start(now);
    osc.stop(t + 0.05);
    if (pan !== null) this.host.track(osc, gain, pan);
    else this.host.track(osc, gain);
  }
}

function createPan(ctx: BaseAudioContext, pan: number): StereoPannerNode | null {
  const c = ctx as BaseAudioContext & { createStereoPanner?: () => StereoPannerNode };
  if (typeof c.createStereoPanner !== 'function') return null;
  const node = ctx.createStereoPanner();
  node.pan.value = pan;
  return node;
}

/** A pulse train, 0..1: the chirp gate's shape. */
function chirpWave(ctx: BaseAudioContext): PeriodicWave {
  const n = 12;
  const real = new Float32Array(n + 1);
  const imag = new Float32Array(n + 1);
  for (let h = 1; h <= n; h++) real[h] = 1 / (1 + h * 0.3);
  return ctx.createPeriodicWave(real, imag);
}

/** A crackle: a few milliseconds of bright noise with a fast decay, rendered once. */
function renderCrackle(ctx: BaseAudioContext, random: () => number): AudioBuffer {
  const sr = ctx.sampleRate;
  const length = Math.round(0.03 * sr);
  const buffer = ctx.createBuffer(1, length, sr);
  const data = buffer.getChannelData(0);
  const tau = 0.002 + 0.004 * random();
  let last = 0;
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const t = i / sr;
    // A one-pole high-pass on white noise: the pop is bright, not a thud.
    const w = random() * 2 - 1;
    const v = (w - last) * Math.exp(-t / tau);
    last = w * 0.6;
    data[i] = v;
    peak = Math.max(peak, Math.abs(v));
  }
  const scale = peak > 0 ? 0.9 / peak : 0;
  for (let i = 0; i < length; i++) data[i] *= scale;
  return buffer;
}

function smoothstep(x: number, a: number, b: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
