import { mulberry32 } from '../core/random';
import { ParamTarget } from './ParamTarget';

/**
 * The space around every sound that is in the world: one shared reverb send,
 * three rooms crossfaded by where the listener is.
 *
 *  - **deck**: the open sea is nearly dry. What a sailor hears is a handful of
 *    early reflections off the deck, the rails and the canvas within the first
 *    thirty milliseconds and a short tail that darkens fast, so this room is
 *    short, sparse and quiet. It is there so a creak sounds like it happened on
 *    a ship rather than inside the listener's head.
 *  - **under**: water carries sound four times faster and loses the top of it
 *    quickly, so under water a sound is dense, dark and smeared, with no
 *    distinct reflections.
 *  - **cliffs**: a rock face throws back a discrete echo a tenth to a quarter of
 *    a second later, darker than the sound, then a longer diffuse tail.
 *
 * Sends are per sound (`AudioSystem` decides how wet each one is) and sum into
 * `input`. The deck and cliff rooms return into the world above water, so
 * going under muffles them with everything else; the underwater room returns
 * past the muffle, because it *is* the underwater sound.
 *
 * The impulses are synthesised, seeded, at construction: filtered noise with
 * an exponential decay and hand-placed early reflections, the same method as
 * the music's room (`music/instruments.ts`, `buildImpulse`). Convolver
 * normalisation is left on, so the three rooms arrive at comparable power and
 * the per-room `level` sets the balance.
 *
 * **Cost.** A convolver costs whether or not anything reaches it, so a room
 * whose weight has sat at zero for longer than its tail is unhooked (the cliff
 * room in open water, the underwater room above it), the same way a silent
 * noise bed is. On Low there are no convolvers at all: one short feedback delay
 * stands in for the three, as the music's room does on that tier.
 */

export type RoomMode = 'convolver' | 'delay';

interface RoomSpec {
  seconds: number;
  /** Return level, relative; balances the normalised impulses. */
  level: number;
  /** Weight time constant: how fast this room is crossfaded in and out. */
  tau: number;
  build(ctx: BaseAudioContext, seconds: number, random: () => number): AudioBuffer;
}

const ROOMS = {
  deck: { seconds: 0.45, level: 1, tau: 0.6, build: deckImpulse },
  under: { seconds: 1.4, level: 1, tau: 0.15, build: underImpulse },
  cliffs: { seconds: 2.2, level: 0.4, tau: 1.2, build: cliffImpulse },
} as const satisfies Record<string, RoomSpec>;

export type RoomName = keyof typeof ROOMS;
const NAMES: readonly RoomName[] = ['deck', 'under', 'cliffs'];

/** A room is unhooked once its weight has been under this for its tail plus a margin. */
const PARK_WEIGHT = 1e-3;
const PARK_MARGIN = 0.5;
/** A swapped-out tier's rooms fade over this, and are unhooked after their tail. */
const SWAP_FADE = 0.08;
const SWAP_RETIRE = 2.5;

interface Path {
  readonly name: RoomName | 'delay';
  /** Weight of this room in the mix; the crossfade. */
  readonly gain: GainNode;
  readonly weight: ParamTarget;
  readonly nodes: AudioNode[];
  readonly tail: AudioNode;
  readonly destination: AudioNode;
  readonly parkAfter: number;
  connected: boolean;
  quietSince: number | null;
}

export class Room {
  /** Sends from sounds in the air sum here: the deck and the cliffs hear them. */
  readonly input: GainNode;
  /** Sends from sounds in the water: the underwater room hears them. */
  readonly underInput: GainNode;
  private readonly ctx: BaseAudioContext;
  private readonly air: AudioNode;
  private readonly under: AudioNode;
  private mode: RoomMode | null = null;
  private paths: Path[] = [];
  /** Inputs of the current paths, so a tier swap can fade them rather than cut them. */
  private feed: GainNode | null = null;
  private underFeed: GainNode | null = null;
  private readonly retiring: { feeds: GainNode[]; paths: Path[]; at: number }[] = [];
  private readonly mixWeights: Record<RoomName, number> = { deck: 1, under: 0, cliffs: 0 };

  /**
   * @param air   Where the deck and cliff rooms return: the world above water.
   * @param under Where the underwater room returns: past the muffle.
   */
  constructor(ctx: BaseAudioContext, air: AudioNode, under: AudioNode, mode: RoomMode) {
    this.ctx = ctx;
    this.air = air;
    this.under = under;
    this.input = ctx.createGain();
    this.underInput = ctx.createGain();
    this.setMode(mode);
  }

  /** Convolvers, or on Low one feedback delay. Swaps without a click. */
  setMode(mode: RoomMode): void {
    if (mode === this.mode) return;
    const first = this.mode === null;
    this.mode = mode;
    const ctx = this.ctx;
    const now = ctx.currentTime;
    if (this.feed !== null && this.underFeed !== null) {
      for (const old of [this.feed, this.underFeed]) {
        const g = old.gain;
        g.cancelScheduledValues(now);
        g.setValueAtTime(g.value, now);
        g.linearRampToValueAtTime(0, now + SWAP_FADE);
      }
      this.retiring.push({ feeds: [this.feed, this.underFeed], paths: this.paths, at: now + SWAP_RETIRE });
    }
    const makeFeed = (from: GainNode): GainNode => {
      const feed = ctx.createGain();
      if (first) {
        feed.gain.value = 1;
      } else {
        feed.gain.setValueAtTime(0, now);
        feed.gain.linearRampToValueAtTime(1, now + SWAP_FADE);
      }
      from.connect(feed);
      return feed;
    };
    const feed = makeFeed(this.input);
    const underFeed = makeFeed(this.underInput);
    this.feed = feed;
    this.underFeed = underFeed;
    this.paths =
      mode === 'convolver'
        ? NAMES.map((name) => this.buildRoom(name, name === 'under' ? underFeed : feed))
        : [this.buildDelay(feed, underFeed)];
    // Start every path at its current weight, not at a fade from zero.
    for (const path of this.paths) {
      const w = this.weightFor(path.name);
      path.weight.set(w, now, true);
      path.weight.flush(now);
    }
  }

  /**
   * How much of each room, 0..1 each. Called every frame; allocates nothing.
   * The weights need not sum to one: a sound by a cliff under water is mostly
   * the water.
   */
  setMix(deck: number, under: number, cliffs: number, now: number, live: boolean): void {
    const w = this.mixWeights;
    w.deck = deck;
    w.under = under;
    w.cliffs = cliffs;
    for (let i = 0; i < this.paths.length; i++) {
      const path = this.paths[i];
      const weight = this.weightFor(path.name);
      path.weight.set(weight, now, live);
      if (!live) continue;
      // Parked while silent for longer than its tail, so a convolver nobody
      // can hear costs nothing.
      if (weight > PARK_WEIGHT) {
        path.quietSince = null;
        if (!path.connected) {
          // Its weight ramps up from where it decayed to; no jump.
          path.tail.connect(path.destination);
          path.connected = true;
        }
      } else if (path.connected) {
        if (path.quietSince === null) path.quietSince = now;
        else if (now - path.quietSince >= path.parkAfter) {
          path.tail.disconnect();
          path.connected = false;
        }
      }
    }
  }

  /** Unhooks rooms a tier swap left ringing, once their tails are gone. */
  tick(now: number): void {
    while (this.retiring.length > 0 && now >= this.retiring[0].at) {
      const old = this.retiring.shift()!;
      this.input.disconnect(old.feeds[0]);
      this.underInput.disconnect(old.feeds[1]);
      for (const feed of old.feeds) feed.disconnect();
      for (const path of old.paths) disposePath(path);
    }
  }

  flush(now: number): void {
    for (const path of this.paths) path.weight.flush(now);
  }

  /** Rooms connected and so being rendered, for the bench. */
  get rendering(): number {
    let n = 0;
    for (const path of this.paths) if (path.connected) n++;
    return n;
  }

  dispose(): void {
    this.tick(Number.POSITIVE_INFINITY);
    for (const path of this.paths) disposePath(path);
    this.paths = [];
    this.feed?.disconnect();
    this.underFeed?.disconnect();
    this.input.disconnect();
    this.underInput.disconnect();
  }

  private weightFor(name: RoomName | 'delay'): number {
    const w = this.mixWeights;
    if (name === 'delay') return Math.min(1, w.deck * ROOMS.deck.level + w.cliffs * ROOMS.cliffs.level + w.under * ROOMS.under.level);
    return w[name] * ROOMS[name].level;
  }

  private buildRoom(name: RoomName, feed: AudioNode): Path {
    const ctx = this.ctx;
    const spec: RoomSpec = ROOMS[name];
    const gain = ctx.createGain();
    gain.gain.value = 0;
    const conv = ctx.createConvolver();
    conv.buffer = spec.build(ctx, spec.seconds, mulberry32(ROOM_SEEDS[name]));
    feed.connect(gain);
    gain.connect(conv);
    const destination = name === 'under' ? this.under : this.air;
    conv.connect(destination);
    return {
      name,
      gain,
      weight: new ParamTarget(gain.gain, 0, spec.tau, 0.002, 0, 1),
      nodes: [gain, conv],
      tail: conv,
      destination,
      parkAfter: spec.seconds + PARK_MARGIN,
      connected: true,
      quietSince: null,
    };
  }

  /** Low: one short, damped feedback delay for every room. */
  private buildDelay(feed: AudioNode, underFeed: AudioNode): Path {
    const ctx = this.ctx;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    const delay = ctx.createDelay(0.5);
    delay.delayTime.value = 0.045;
    const feedback = ctx.createGain();
    feedback.gain.value = 0.3;
    const damp = ctx.createBiquadFilter();
    damp.type = 'lowpass';
    damp.frequency.value = 2400;
    feed.connect(gain);
    underFeed.connect(gain);
    gain.connect(delay);
    delay.connect(damp);
    damp.connect(feedback);
    feedback.connect(delay);
    const out = ctx.createGain();
    out.gain.value = 0.5;
    damp.connect(out);
    out.connect(this.air);
    return {
      name: 'delay',
      gain,
      weight: new ParamTarget(gain.gain, 0, 0.6, 0.002, 0, 1),
      nodes: [gain, delay, feedback, damp, out],
      tail: out,
      destination: this.air,
      parkAfter: 1 + PARK_MARGIN,
      connected: true,
      quietSince: null,
    };
  }
}

const ROOM_SEEDS: Record<RoomName, number> = {
  deck: 0x0dec4a,
  under: 0x5ea0b1,
  cliffs: 0xc11ff5,
};

function disposePath(path: Path): void {
  for (const node of path.nodes) node.disconnect();
  path.connected = false;
}

// --- impulses ---------------------------------------------------------------

/** A one-pole low-passed noise tail, `from`..`to` seconds, decaying with `tau`. */
function addTail(
  data: Float32Array,
  sr: number,
  from: number,
  tau: number,
  level: number,
  brightness: (x: number) => number,
  random: () => number,
): void {
  let lp = 0;
  const start = Math.round(from * sr);
  for (let i = start; i < data.length; i++) {
    const t = (i - start) / sr;
    const x = i / data.length;
    lp += brightness(x) * (random() * 2 - 1 - lp);
    // A raised-cosine fade over the last tenth, so the buffer does not end on a step.
    const end = x > 0.9 ? 0.5 + 0.5 * Math.cos(((x - 0.9) / 0.1) * Math.PI) : 1;
    data[i] += level * lp * Math.exp(-t / tau) * end;
  }
}

/** A reflection: a few samples of darkened noise rather than one sample, so it is a surface and not a click. */
function addReflection(data: Float32Array, sr: number, at: number, level: number, smear: number, random: () => number): void {
  const start = Math.round(at * sr);
  const width = Math.max(2, Math.round(smear * sr));
  let lp = 0;
  for (let k = 0; k < width && start + k < data.length; k++) {
    lp += 0.5 * (random() * 2 - 1 - lp);
    const window = Math.sin((Math.PI * (k + 0.5)) / width);
    data[start + k] += level * window * (0.6 + 0.4 * lp);
  }
}

function deckImpulse(ctx: BaseAudioContext, seconds: number, random: () => number): AudioBuffer {
  const sr = ctx.sampleRate;
  const buffer = ctx.createBuffer(2, Math.round(seconds * sr), sr);
  // Deck, rails, bulwarks, the foot of the sails: 2 to 30 ms, each side its own.
  const reflections = [0.0034, 0.0061, 0.0098, 0.0142, 0.0195, 0.0263];
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    for (let r = 0; r < reflections.length; r++) {
      const at = reflections[r] * (0.85 + 0.3 * random());
      const sign = random() < 0.5 ? -1 : 1;
      addReflection(data, sr, at, sign * (0.55 - r * 0.07), 0.0006, random);
    }
    // A short tail that loses its top quickly: open air has nothing to sustain it.
    addTail(data, sr, 0.004, 0.055, 0.22, (x) => 0.7 - 0.6 * x, random);
  }
  return buffer;
}

function underImpulse(ctx: BaseAudioContext, seconds: number, random: () => number): AudioBuffer {
  const sr = ctx.sampleRate;
  const buffer = ctx.createBuffer(2, Math.round(seconds * sr), sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    // Dense and dark from the start: no early reflections, a one-pole near
    // 600 Hz closing further as it decays.
    addTail(data, sr, 0.003, 0.32, 1, (x) => 0.09 - 0.06 * x, random);
  }
  return buffer;
}

function cliffImpulse(ctx: BaseAudioContext, seconds: number, random: () => number): AudioBuffer {
  const sr = ctx.sampleRate;
  const buffer = ctx.createBuffer(2, Math.round(seconds * sr), sr);
  // Echoes off rock faces 15 to 45 m away, the nearer ones louder; the two
  // ears hear different faces, which is what makes it wide.
  const echoes = [0.088, 0.137, 0.204, 0.262];
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    addReflection(data, sr, 0.004, 0.25, 0.0008, random);
    for (let e = 0; e < echoes.length; e++) {
      const at = echoes[e] * (0.88 + 0.24 * random()) + ch * 0.011 * random();
      // Smeared wider and darker the further it travelled.
      addReflection(data, sr, at, (0.5 - e * 0.09) * (random() < 0.5 ? -1 : 1), 0.002 + e * 0.0015, random);
    }
    addTail(data, sr, 0.06, 0.42, 0.18, (x) => 0.35 - 0.3 * x, random);
  }
  return buffer;
}
