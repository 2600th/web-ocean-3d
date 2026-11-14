/**
 * The adaptive score: an original sea-song band, synthesised, whose arrangement
 * follows the game.
 *
 * **Shape.** One sequencer walks the score (`score.ts`) a quaver at a time and
 * hands notes to eight *stems* — drone, reeds, fiddle, concertina, plucked
 * strings, bodhrán, bass and glass. Each stem is a channel with its own body EQ
 * and pan, ending in a gain. A music *state* (`calm`, `sailing`, `storm`,
 * `underwater`, `tour`, `silent`) is a profile: which tune and phrase order,
 * the tempo range, the patterns each player uses, and a target gain per stem at
 * zero and at full *energy*. Changing state moves the stem gains on a
 * one-and-a-half-second crossfade and switches the material at the next
 * barline, so a transition is a band changing what it plays, not a track cut.
 * Energy (ship speed, under sail) moves the tempo, the stem balance and how
 * many drum and string notes are played.
 *
 * **Scheduling.** Look-ahead, driven from the frame loop: each `update` schedules
 * every quaver that starts within `LOOKAHEAD` seconds on the audio clock. The
 * audio thread's clock is the accurate one, so notes land exactly wherever the
 * frames do. A frame hitch longer than the look-ahead skips the missed quavers
 * rather than playing them late in a heap. `now` is passed in, not read, so the
 * same engine can be driven from an `OfflineAudioContext` for verification.
 *
 * **Cost.** The drone is five persistent oscillators; everything else is built
 * per note and released when it ends. Notes are counted against a cap, and when
 * the band is over it the lowest-priority notes (ghost drum strokes, inner
 * plucks) are dropped first — the tune is the last thing to go. A stem whose
 * target gain is zero schedules nothing at all, so `silent` and a music volume
 * of zero cost only the drone's idle oscillators.
 *
 * **Mix.** Stems sum into a ducking gain (thunder and stings pull the band down
 * under them), then a dry bus with a reverb send, a state low-pass (the
 * underwater arrangement is heard through water), and the music volume.
 */

import { mulberry32 } from '../../core/random';
import { ParamTarget } from '../ParamTarget';
import {
  buildImpulse,
  buildReedWave,
  playBass,
  playBell,
  playDrum,
  playFiddle,
  playGlass,
  playPluck,
  playReed,
  renderPluck,
  type DrumStroke,
  type NoteHost,
} from './instruments';
import {
  BARS_PER_PHRASE,
  ORDERS,
  SONGS,
  STEPS_PER_BAR,
  type Bar,
  type OrderId,
  type Phrase,
  type SongId,
} from './score';
import { atOrAbove, chordToneBelow, midiToHz, voicing } from './theory';

export type MusicState = 'calm' | 'sailing' | 'storm' | 'underwater' | 'tour' | 'silent';
export type MusicSting = 'cargo';
export type MusicReverb = 'convolver' | 'delay';

export const MUSIC_STATES: readonly MusicState[] = [
  'calm',
  'sailing',
  'storm',
  'underwater',
  'tour',
  'silent',
];

type StemId = 'drone' | 'reeds' | 'fiddle' | 'squeeze' | 'pluck' | 'drum' | 'bass' | 'glass';
const STEMS: readonly StemId[] = [
  'drone',
  'reeds',
  'fiddle',
  'squeeze',
  'pluck',
  'drum',
  'bass',
  'glass',
];

type Mix = Readonly<Record<StemId, number>>;

interface Profile {
  readonly song: SongId;
  readonly order: OrderId;
  /** Dotted-crotchet beats per minute at zero and at full energy. */
  readonly tempo: readonly [number, number];
  readonly low: Mix;
  readonly high: Mix;
  /** Who carries the tune, and who harmonises it. */
  readonly tune: 'fiddle' | 'squeeze' | 'glass';
  readonly harmony: 'squeeze' | 'fiddle' | null;
  readonly pluck: 'jig' | 'arp' | 'none';
  readonly drum: 'jig' | 'toms' | 'pulse' | 'none';
  readonly bass: 'beats' | 'tremolo' | 'none';
  readonly reeds: 'chords' | 'low' | 'none';
  /** Depth of the hurdy-gurdy's rhythmic buzz on the beats, 0..1. */
  readonly buzz: number;
  /** Semitones the drone sits below its home D3/A3. */
  readonly droneShift: number;
  /** Low-pass on the whole band, Hz. */
  readonly cutoff: number;
  readonly reverb: number;
}

function mix(m: Partial<Record<StemId, number>>): Mix {
  return {
    drone: 0,
    reeds: 0,
    fiddle: 0,
    squeeze: 0,
    pluck: 0,
    drum: 0,
    bass: 0,
    glass: 0,
    ...m,
  };
}

/**
 * The states. Gains are relative; the absolute level of each stem is set by its
 * per-note velocities, which were balanced against each other by measurement.
 */
const PROFILES: Readonly<Record<MusicState, Profile>> = {
  calm: {
    song: 'voyage',
    order: 'calm',
    tempo: [40, 46],
    low: mix({ drone: 0.55, reeds: 0.6, squeeze: 0.85, pluck: 0.8, glass: 0.3 }),
    high: mix({ drone: 0.55, reeds: 0.6, squeeze: 0.85, pluck: 0.85, glass: 0.3 }),
    tune: 'squeeze',
    harmony: null,
    pluck: 'arp',
    drum: 'none',
    bass: 'none',
    reeds: 'chords',
    buzz: 0,
    droneShift: 0,
    cutoff: 7000,
    reverb: 0.4,
  },
  sailing: {
    song: 'voyage',
    order: 'sailing',
    tempo: [70, 104],
    low: mix({ drone: 0.5, reeds: 0.45, fiddle: 0.75, squeeze: 0.5, pluck: 0.7, drum: 0.55, bass: 0.45 }),
    high: mix({ drone: 0.55, reeds: 0.35, fiddle: 0.9, squeeze: 0.6, pluck: 0.8, drum: 0.85, bass: 0.65 }),
    tune: 'fiddle',
    harmony: 'squeeze',
    pluck: 'jig',
    drum: 'jig',
    bass: 'beats',
    reeds: 'chords',
    buzz: 0.6,
    droneShift: 0,
    cutoff: 14000,
    reverb: 0.22,
  },
  storm: {
    song: 'squall',
    order: 'storm',
    tempo: [80, 92],
    low: mix({ drone: 1.4, reeds: 0.8, fiddle: 1.4, drum: 1.35, bass: 1.5 }),
    high: mix({ drone: 1.5, reeds: 0.9, fiddle: 1.5, drum: 1.6, bass: 1.7 }),
    tune: 'fiddle',
    harmony: 'fiddle',
    pluck: 'none',
    drum: 'toms',
    bass: 'tremolo',
    reeds: 'low',
    buzz: 0.25,
    droneShift: 12,
    cutoff: 6000,
    reverb: 0.32,
  },
  underwater: {
    song: 'voyage',
    order: 'underwater',
    tempo: [34, 34],
    low: mix({ drone: 0.25, glass: 1 }),
    high: mix({ drone: 0.25, glass: 1 }),
    tune: 'glass',
    harmony: null,
    pluck: 'none',
    drum: 'none',
    bass: 'none',
    reeds: 'none',
    buzz: 0,
    droneShift: 12,
    cutoff: 1300,
    reverb: 0.7,
  },
  tour: {
    song: 'voyage',
    order: 'tour',
    tempo: [62, 62],
    low: mix({ drone: 0.45, reeds: 0.55, fiddle: 0.8, squeeze: 0.6, pluck: 0.6, drum: 0.45, bass: 0.35, glass: 0.3 }),
    high: mix({ drone: 0.45, reeds: 0.55, fiddle: 0.8, squeeze: 0.6, pluck: 0.6, drum: 0.45, bass: 0.35, glass: 0.3 }),
    tune: 'fiddle',
    harmony: 'squeeze',
    pluck: 'arp',
    drum: 'pulse',
    bass: 'beats',
    reeds: 'chords',
    buzz: 0.2,
    droneShift: 0,
    cutoff: 12000,
    reverb: 0.35,
  },
  silent: {
    song: 'voyage',
    order: 'calm',
    tempo: [40, 40],
    low: mix({}),
    high: mix({}),
    tune: 'squeeze',
    harmony: null,
    pluck: 'none',
    drum: 'none',
    bass: 'none',
    reeds: 'none',
    buzz: 0,
    droneShift: 0,
    cutoff: 7000,
    reverb: 0.3,
  },
};

/** Seconds of audio scheduled ahead of the clock. Longer than any plausible frame. */
const LOOKAHEAD = 0.35;
/** Fell this far behind (a hitch, a background tab): skip ahead instead of catching up. */
const RESYNC_BEHIND = 0.12;
/** Stem crossfade time constant, seconds. */
const STEM_TAU = 1.4;
/** Seconds a requested state must persist before the band follows it. */
const STATE_HOLD = 2.0;
/** …except going under or coming up, which the band follows almost at once. */
const STATE_HOLD_WATER = 0.35;
/** Energy smoothing, seconds. Speed changes over seconds; the band follows. */
const ENERGY_TAU = 2.5;

/** Overall music level at a music volume of 1. Set by measurement against the ambience. */
const MUSIC_LEVEL = 0.45;

/** Base octave the plucked-string buffers are rendered in; others are replayed at 2^n. */
const PLUCK_BASE = 48;

/** Per-instrument note velocities, before the stem gain. */
const VEL = {
  fiddle: 0.2,
  squeeze: 0.16,
  reeds: 0.07,
  pluck: 0.65,
  drum: 0.42,
  bass: 0.2,
  glass: 0.075,
  bell: 0.13,
  buzz: 0.16,
} as const;

/** Note priorities against the voice cap. */
const PRIORITY_LOW = 0;
const PRIORITY_MID = 1;
const PRIORITY_HIGH = 2;

interface FilterSpec {
  type: BiquadFilterType;
  frequency: number;
  q: number;
  gain?: number;
}

const CHANNELS: Readonly<Record<Exclude<StemId, 'drone'>, { filters: FilterSpec[]; pan: number }>> = {
  fiddle: {
    // A violin body: the air resonance, the bridge hill, and the top of a
    // sawtooth taken off so it stops being a synth.
    filters: [
      { type: 'highpass', frequency: 190, q: 0.7 },
      { type: 'peaking', frequency: 480, q: 1.4, gain: 3 },
      { type: 'peaking', frequency: 2900, q: 1.1, gain: 4 },
      { type: 'lowpass', frequency: 6500, q: 0.6 },
    ],
    pan: -0.14,
  },
  squeeze: {
    filters: [
      { type: 'peaking', frequency: 1300, q: 1, gain: 3 },
      { type: 'lowpass', frequency: 4200, q: 0.6 },
    ],
    pan: 0.2,
  },
  reeds: {
    filters: [
      { type: 'highpass', frequency: 110, q: 0.7 },
      { type: 'lowpass', frequency: 2300, q: 0.6 },
    ],
    pan: 0.05,
  },
  pluck: {
    filters: [
      { type: 'peaking', frequency: 210, q: 1, gain: 3 },
      { type: 'lowpass', frequency: 3800, q: 0.6 },
    ],
    pan: 0.32,
  },
  drum: {
    filters: [{ type: 'lowpass', frequency: 6000, q: 0.6 }],
    pan: -0.04,
  },
  bass: {
    filters: [{ type: 'lowpass', frequency: 480, q: 0.9 }],
    pan: 0,
  },
  glass: {
    filters: [{ type: 'lowpass', frequency: 3200, q: 0.6 }],
    pan: 0,
  },
};

const DRONE_ROOT = 50; // D3
const DRONE_FIFTH = 57; // A3
const DRONE_BUZZ = 62; // D4

export interface MusicEngineOptions {
  /** Concurrent notes. */
  voices: number;
  reverb: MusicReverb;
  /** Music volume, 0..1. */
  volume?: number;
}

export class MusicEngine implements NoteHost {
  readonly ctx: BaseAudioContext;

  private readonly output: GainNode;
  private readonly volumeTarget: ParamTarget;
  private readonly filterTarget: ParamTarget;
  private readonly reverbTarget: ParamTarget;
  private readonly duckGain: GainNode;
  private readonly stemsBus: GainNode;
  private readonly dry: GainNode;
  private readonly wetSend: GainNode;
  private readonly filter: BiquadFilterNode;
  private wetPath: AudioNode[] = [];
  private reverbMode: MusicReverb | null = null;

  private readonly inputs: Record<Exclude<StemId, 'drone'>, GainNode>;
  private readonly stemGains: Record<StemId, ParamTarget>;
  private readonly stemLevel: Record<StemId, number>;
  private readonly sting: GainNode;
  private readonly chain: AudioNode[] = [];

  // --- drone ---------------------------------------------------------------
  private readonly droneOscs: OscillatorNode[] = [];
  private droneOut!: GainNode;
  private readonly droneRoot: ParamTarget;
  private readonly droneFifth: ParamTarget;
  private readonly droneSub: ParamTarget;
  private readonly buzzGate: GainNode;
  private readonly buzzPitch: ParamTarget;
  private droneStarted = false;

  private readonly white: AudioBuffer;
  private readonly reedWave: PeriodicWave;
  private readonly plucks = new Map<number, AudioBuffer>();

  // --- state -----------------------------------------------------------------
  private state: MusicState = 'calm';
  private requested: MusicState = 'calm';
  private profile: Profile = PROFILES.calm;
  private switchTimer = 0;
  private energy = 0;
  private energyTarget = 0;
  private volume = 0.5;
  /** Verification only: every other stem is held at zero. */
  private solo: StemId | null = null;
  private voiceCap: number;
  private activeVoices = 0;
  private readonly live = new Set<AudioScheduledSourceNode>();

  // --- sequencer -----------------------------------------------------------------
  private order: OrderId = 'calm';
  private song: SongId = 'voyage';
  private pendingOrder: OrderId | null = null;
  private phraseIndex = 0;
  private barIndex = 0;
  private stepIndex = 0;
  private nextStep = 0;
  private stepDur = 0.5;
  private wasLive = false;
  private random = mulberry32(0x50c1a1);
  private pluckRandom = mulberry32(0x91c4ed);
  private readonly scratch: number[] = [];

  constructor(
    ctx: BaseAudioContext,
    destination: AudioNode,
    white: AudioBuffer,
    options: MusicEngineOptions,
  ) {
    this.ctx = ctx;
    this.white = white;
    this.voiceCap = Math.max(4, options.voices);
    this.volume = clamp01(options.volume ?? this.volume);
    this.reedWave = buildReedWave(ctx);

    // stems -> duck -> dry -> filter -> output
    //               \-> wet send -> reverb -> filter
    this.output = ctx.createGain();
    this.output.gain.value = 0;
    this.output.connect(destination);
    this.volumeTarget = new ParamTarget(this.output.gain, 0, 0.08, 0.001, 0, 2);

    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.Q.value = 0.6;
    this.filter.connect(this.output);
    this.filterTarget = new ParamTarget(
      this.filter.frequency,
      this.profile.cutoff,
      0.6,
      5,
      200,
      ctx.sampleRate * 0.45,
    );

    this.dry = ctx.createGain();
    this.dry.connect(this.filter);
    this.wetSend = ctx.createGain();
    this.wetSend.gain.value = 0;
    this.dry.connect(this.wetSend);
    this.reverbTarget = new ParamTarget(this.wetSend.gain, 0, 0.8, 0.005, 0, 1);
    this.setReverb(options.reverb);

    this.duckGain = ctx.createGain();
    this.duckGain.connect(this.dry);
    this.stemsBus = ctx.createGain();
    this.stemsBus.connect(this.duckGain);

    // The sting bypasses the duck it causes.
    this.sting = ctx.createGain();
    this.sting.gain.value = 1;
    this.sting.connect(this.dry);

    const stemGain = {} as Record<StemId, ParamTarget>;
    const level = {} as Record<StemId, number>;
    const inputs = {} as Record<Exclude<StemId, 'drone'>, GainNode>;
    for (const stem of STEMS) {
      const gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(this.stemsBus);
      stemGain[stem] = new ParamTarget(gain.gain, 0, STEM_TAU, 0.002, 0, 2);
      level[stem] = 0;
      this.chain.push(gain);
      if (stem === 'drone') {
        this.droneOut = gain;
        continue;
      }
      const spec = CHANNELS[stem];
      const input = ctx.createGain();
      let tail: AudioNode = input;
      for (const f of spec.filters) {
        const node = ctx.createBiquadFilter();
        node.type = f.type;
        node.frequency.value = f.frequency;
        node.Q.value = f.q;
        if (f.gain !== undefined) node.gain.value = f.gain;
        tail.connect(node);
        tail = node;
        this.chain.push(node);
      }
      const pan = createPanner(ctx, spec.pan);
      if (pan !== null) {
        tail.connect(pan);
        tail = pan;
        this.chain.push(pan);
      }
      tail.connect(gain);
      inputs[stem] = input;
      this.chain.push(input);
    }
    this.stemGains = stemGain;
    this.stemLevel = level;
    this.inputs = inputs;

    // --- the hurdy-gurdy -----------------------------------------------------
    // Root and fifth on bowed (sawtooth) strings, a soft sine an octave under,
    // all through one warm low-pass, breathing slowly on an LFO. The trompette
    // is a brighter string whose gate is struck on the beats under sail.
    const droneMix = ctx.createGain();
    const droneLow = ctx.createBiquadFilter();
    droneLow.type = 'lowpass';
    droneLow.frequency.value = 900;
    droneLow.Q.value = 0.7;
    droneMix.connect(droneLow);
    droneLow.connect(this.droneOut);
    this.chain.push(droneMix, droneLow);

    const voice = (type: OscillatorType, midi: number, level: number): ParamTarget => {
      const osc = ctx.createOscillator();
      osc.type = type;
      const g = ctx.createGain();
      g.gain.value = level;
      osc.connect(g);
      g.connect(droneMix);
      this.droneOscs.push(osc);
      this.chain.push(g);
      return new ParamTarget(osc.frequency, midiToHz(midi), 0.9, 0.05, 10, 4000);
    };
    this.droneRoot = voice('sawtooth', DRONE_ROOT, 0.04);
    this.droneFifth = voice('sawtooth', DRONE_FIFTH, 0.025);
    this.droneSub = voice('sine', DRONE_ROOT - 12, 0.055);

    const breath = ctx.createOscillator();
    breath.frequency.value = 0.13;
    const breathDepth = ctx.createGain();
    breathDepth.gain.value = 0.18;
    breath.connect(breathDepth);
    breathDepth.connect(droneMix.gain);
    this.droneOscs.push(breath);
    this.chain.push(breathDepth);

    const buzz = ctx.createOscillator();
    buzz.type = 'sawtooth';
    const buzzHigh = ctx.createBiquadFilter();
    buzzHigh.type = 'bandpass';
    buzzHigh.frequency.value = 1700;
    buzzHigh.Q.value = 0.8;
    this.buzzGate = ctx.createGain();
    this.buzzGate.gain.value = 0;
    buzz.connect(buzzHigh);
    buzzHigh.connect(this.buzzGate);
    this.buzzGate.connect(this.droneOut);
    this.droneOscs.push(buzz);
    this.chain.push(buzzHigh, this.buzzGate);
    this.buzzPitch = new ParamTarget(buzz.frequency, midiToHz(DRONE_BUZZ), 0.9, 0.05, 10, 4000);
  }

  // ------------------------------------------------------------------ public

  get currentState(): MusicState {
    return this.state;
  }

  get voicesInUse(): number {
    return this.activeVoices;
  }

  /** Where the sequencer is, for tests and the debug overlay. */
  get position(): { order: OrderId; phrase: number; bar: number; step: number; energy: number } {
    return {
      order: this.order,
      phrase: this.phraseIndex,
      bar: this.barIndex,
      step: this.stepIndex,
      energy: this.energy,
    };
  }

  setState(state: MusicState): void {
    if (PROFILES[state] === undefined) return;
    this.requested = state;
  }

  /** 0..1. Ship speed under sail, typically; smoothed here over a few seconds. */
  setEnergy(energy: number): void {
    if (Number.isFinite(energy)) this.energyTarget = clamp01(energy);
  }

  setVolume(volume: number): void {
    if (Number.isFinite(volume)) this.volume = clamp01(volume);
  }

  getVolume(): number {
    return this.volume;
  }

  /**
   * Plays one stem alone (or all of them, with null). For measuring the balance
   * offline; nothing in the app calls it.
   */
  setSolo(stem: string | null): void {
    this.solo = stem !== null && (STEMS as readonly string[]).includes(stem) ? (stem as StemId) : null;
  }

  setVoices(voices: number): void {
    this.voiceCap = Math.max(4, Math.floor(voices));
  }

  /** Swaps the room. The convolver on capable tiers, a feedback delay on Low. */
  setReverb(mode: MusicReverb): void {
    if (mode === this.reverbMode) return;
    this.reverbMode = mode;
    this.wetSend.disconnect();
    for (const node of this.wetPath) node.disconnect();
    this.wetPath = [];
    const ctx = this.ctx;
    if (mode === 'convolver') {
      const conv = ctx.createConvolver();
      conv.buffer = buildImpulse(ctx, 2.3, mulberry32(0x2e7e2b));
      this.wetSend.connect(conv);
      conv.connect(this.filter);
      this.wetPath = [conv];
    } else {
      const delay = ctx.createDelay(1);
      delay.delayTime.value = 0.27;
      const feedback = ctx.createGain();
      feedback.gain.value = 0.33;
      const damp = ctx.createBiquadFilter();
      damp.type = 'lowpass';
      damp.frequency.value = 2200;
      this.wetSend.connect(delay);
      delay.connect(damp);
      damp.connect(feedback);
      feedback.connect(delay);
      damp.connect(this.filter);
      this.wetPath = [delay, feedback, damp];
    }
  }

  /** Starts the persistent drone. Idempotent; call from a running context. */
  start(when: number): void {
    if (this.droneStarted) return;
    this.droneStarted = true;
    for (const osc of this.droneOscs) osc.start(when);
  }

  /**
   * Pulls the band down for a moment — under thunder, under a sting.
   *
   * @param depth How far, 0..1 (0.5 is about -6 dB).
   * @param hold  Seconds before it starts to come back.
   */
  duck(depth: number, hold: number, when: number): void {
    const floor = 1 - clamp01(depth);
    const g = this.duckGain.gain;
    g.cancelScheduledValues(when);
    g.setTargetAtTime(floor, when, 0.06);
    g.setTargetAtTime(1, when + Math.max(0.05, hold), 0.55);
  }

  /**
   * A short musical flourish over the band, which ducks under it.
   *
   * Placed on the next quaver when that is close, so it lands in time with
   * whatever is playing; otherwise at once, because a reward that waits for a
   * slow bar to come round has stopped being a reward.
   */
  playSting(kind: MusicSting, now: number): void {
    if (this.volume <= 0) return;
    const t = this.nextStep > now && this.nextStep - now < 0.22 ? this.nextStep : now + 0.02;
    this.duck(0.6, 1.5, t);
    switch (kind) {
      case 'cargo': {
        // A run up the D major arpeggio, a held top D over a squeezebox chord.
        const run = [69, 74, 78, 81, 86];
        const gap = 0.072;
        for (let i = 0; i < run.length; i++) {
          const last = i === run.length - 1;
          playFiddle(
            this,
            this.sting,
            t + i * gap,
            run[i],
            last ? 0.95 : gap * 1.3,
            VEL.fiddle * (last ? 1.1 : 0.85),
            { cut: null, slide: last },
            this.random,
          );
        }
        const land = t + (run.length - 1) * gap;
        for (const m of [62, 66, 69, 74]) {
          playReed(this, this.sting, this.reedWave, land, m, 1.0, VEL.reeds * 1.4, 0.04, 0.35, (this.random() - 0.5) * 8);
        }
        playBell(this, this.sting, land, 86, VEL.bell * 0.8);
        playDrum(this, this.sting, this.white, land, VEL.drum * 1.1, 'down', this.random);
        break;
      }
    }
  }

  /**
   * Advances the arrangement and schedules every quaver inside the look-ahead.
   *
   * `live` false (context suspended, muted, disabled) writes nothing and
   * schedules nothing; on the frame it comes back the sequencer re-seats itself
   * just ahead of the clock rather than trying to play the silence it missed.
   */
  update(dt: number, now: number, live: boolean): void {
    const step = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.25) : 0;

    // --- state and energy (the model runs whether or not anything is heard) ---
    if (this.requested !== this.state) {
      this.switchTimer += step;
      const water = this.requested === 'underwater' || this.state === 'underwater';
      const hold = this.requested === 'silent' ? 0 : water ? STATE_HOLD_WATER : STATE_HOLD;
      if (this.switchTimer >= hold) this.applyState(this.requested);
    } else {
      this.switchTimer = 0;
    }
    this.energy += (this.energyTarget - this.energy) * (1 - Math.exp(-step / ENERGY_TAU));

    const p = this.profile;
    const e = this.energy;
    const audible = this.volume > 0;
    for (const stem of STEMS) {
      const soloed = this.solo === null || this.solo === stem;
      const target = audible && soloed ? p.low[stem] + (p.high[stem] - p.low[stem]) * e : 0;
      this.stemLevel[stem] = target;
      this.stemGains[stem].set(target, now, live);
    }
    this.volumeTarget.set(this.volume * MUSIC_LEVEL, now, live);
    this.filterTarget.set(p.cutoff, now, live);
    this.reverbTarget.set(p.reverb, now, live);
    const shift = p.droneShift;
    this.droneRoot.set(midiToHz(DRONE_ROOT - shift), now, live);
    this.droneFifth.set(midiToHz(DRONE_FIFTH - shift), now, live);
    this.droneSub.set(midiToHz(DRONE_ROOT - 12 - shift), now, live);
    this.buzzPitch.set(midiToHz(DRONE_BUZZ - shift), now, live);

    if (!live) {
      this.wasLive = false;
      return;
    }
    if (!this.wasLive) {
      this.wasLive = true;
      this.start(now);
      this.flush(now);
      this.nextStep = now + 0.06;
    }
    this.pump(now);
  }

  /**
   * Schedules every quaver that starts inside the look-ahead, and nothing else.
   *
   * `update` calls it every frame. `AudioSystem` also calls `update(0, …)`
   * from a 100 ms timer, because frames are not guaranteed — a shader compile
   * or a heavy scene can hold the render loop for a second or more, and a band
   * that stops whenever the GPU is busy is a band with a fault. With `dt` 0 the
   * model does not advance; only the scheduling does. Does nothing until
   * `update` has seen the context live.
   */
  pump(now: number): void {
    if (!this.wasLive) return;
    if (this.nextStep < now - RESYNC_BEHIND) this.nextStep = now + 0.02;
    const p = this.profile;
    const tempo = p.tempo[0] + (p.tempo[1] - p.tempo[0]) * this.energy;
    const targetDur = 60 / (Math.max(20, tempo) * 3);
    const audible = this.volume > 0;
    while (this.nextStep < now + LOOKAHEAD) {
      // A glide, not a jump: the band speeds up over a couple of bars.
      this.stepDur += (targetDur - this.stepDur) * 0.12;
      if (audible) this.scheduleStep(this.nextStep, this.stepDur);
      this.advance();
      this.nextStep += this.stepDur;
    }
  }

  /** Holds the sequencer until `update` next sees the context live (mute, disable). */
  pause(): void {
    this.wasLive = false;
  }

  /** Back to the top: silences every note and rewinds the sequencer and its random streams. */
  reset(): void {
    this.stopAll();
    this.random = mulberry32(0x50c1a1);
    this.pluckRandom = mulberry32(0x91c4ed);
    this.phraseIndex = 0;
    this.barIndex = 0;
    this.stepIndex = 0;
    this.energy = this.energyTarget;
    this.applyState(this.requested);
    this.order = this.profile.order;
    this.song = this.profile.song;
    this.pendingOrder = null;
    this.wasLive = false;
  }

  dispose(): void {
    this.stopAll();
    for (const osc of this.droneOscs) {
      try {
        osc.stop();
      } catch {
        // never started
      }
      osc.disconnect();
    }
    for (const node of this.chain) node.disconnect();
    for (const node of this.wetPath) node.disconnect();
    this.wetSend.disconnect();
    this.stemsBus.disconnect();
    this.duckGain.disconnect();
    this.sting.disconnect();
    this.dry.disconnect();
    this.filter.disconnect();
    this.output.disconnect();
  }

  /** `NoteHost`: counts a note and unhooks it when it ends. */
  track(source: AudioScheduledSourceNode, nodes: readonly AudioNode[]): void {
    this.activeVoices++;
    this.live.add(source);
    source.onended = () => {
      // Guarded: Chrome fires `ended` twice for a buffer source whose stop()
      // lands exactly on the buffer's natural end (measured), which counted
      // voices below zero and quietly lifted the cap.
      if (!this.live.delete(source)) return;
      source.disconnect();
      for (const node of nodes) node.disconnect();
      this.activeVoices--;
    };
  }

  // ---------------------------------------------------------------- internals

  private flush(now: number): void {
    for (const stem of STEMS) this.stemGains[stem].flush(now);
    this.volumeTarget.flush(now);
    this.filterTarget.flush(now);
    this.reverbTarget.flush(now);
    this.droneRoot.flush(now);
    this.droneFifth.flush(now);
    this.droneSub.flush(now);
    this.buzzPitch.flush(now);
  }

  private stopAll(): void {
    for (const source of this.live) {
      try {
        source.stop();
      } catch {
        // already stopped
      }
    }
  }

  private applyState(state: MusicState): void {
    this.state = state;
    this.switchTimer = 0;
    const next = PROFILES[state];
    this.profile = next;
    if (state !== 'silent' && next.order !== this.order) this.pendingOrder = next.order;
  }

  private advance(): void {
    this.stepIndex++;
    if (this.stepIndex < STEPS_PER_BAR) return;
    this.stepIndex = 0;
    this.barIndex++;
    if (this.barIndex < BARS_PER_PHRASE) return;
    this.barIndex = 0;
    this.phraseIndex = (this.phraseIndex + 1) % ORDERS[this.order].length;
  }

  private canPlay(priority: number): boolean {
    if (priority >= PRIORITY_HIGH) return this.activeVoices < this.voiceCap * 1.25;
    if (priority === PRIORITY_MID) return this.activeVoices < this.voiceCap;
    return this.activeVoices < this.voiceCap * 0.8;
  }

  private on(stem: StemId): boolean {
    return this.stemLevel[stem] > 0.01;
  }

  private scheduleStep(t: number, d: number): void {
    if (this.stepIndex === 0 && this.pendingOrder !== null) {
      this.order = this.pendingOrder;
      this.song = this.profile.song;
      this.pendingOrder = null;
      this.phraseIndex = 0;
      this.barIndex = 0;
    }
    const plan: Phrase = ORDERS[this.order][this.phraseIndex % ORDERS[this.order].length];
    const bars = SONGS[this.song].sections[plan.section];
    if (bars === undefined) return;
    const bar = bars[this.barIndex];
    const p = this.profile;
    const s = this.stepIndex;
    const e = this.energy;
    const random = this.random;

    if (s === 0) this.scheduleBar(t, d, bar);

    // --- the tune ---------------------------------------------------------
    if (plan.tune) {
      for (const note of bar.notes) {
        if (note.step !== s) continue;
        const dur = note.len * d;
        this.playTune(p.tune, t, note.midi, dur, note.len, false);
        if (plan.harmony && p.harmony !== null) {
          this.playTune(p.harmony, t, chordToneBelow(bar.chord, note.midi), dur, note.len, true);
        }
      }
    }

    // --- strings ---------------------------------------------------------------
    if (p.pluck !== 'none' && this.on('pluck')) {
      const v = this.scratch;
      if (p.pluck === 'jig') {
        // Boom, chick-chick: a bass note on each beat and the chord between.
        const root = atOrAbove(bar.chord.root, 43);
        if (s === 0) this.pluck(t, root, VEL.pluck, 2 * d, PRIORITY_MID);
        else if (s === 3) this.pluck(t, root + 7, VEL.pluck * 0.85, 2 * d, PRIORITY_MID);
        else if (s === 2 || s === 5) {
          voicing(bar.chord, 55, 66, v);
          const level = VEL.pluck * (0.45 + 0.25 * e);
          for (let i = 0; i < v.length && i < 3; i++) {
            // A strum: each string a few milliseconds after the last.
            this.pluck(t + i * 0.011, v[i], level, d, PRIORITY_LOW);
          }
        } else if (e > 0.7) {
          voicing(bar.chord, 59, 71, v);
          if (v.length > 0) this.pluck(t, v[(s + this.barIndex) % v.length], VEL.pluck * 0.3, d, PRIORITY_LOW);
        }
      } else {
        // A rolling arpeggio, harp-like, letting each note ring through the next.
        voicing(bar.chord, 50, 76, v);
        if (v.length > 0) {
          const shape = [0, 2, 4, 5, 4, 2];
          const idx = Math.min(v.length - 1, shape[s]);
          this.pluck(t, v[idx], VEL.pluck * (s === 0 ? 0.8 : 0.55), 3 * d, s === 0 ? PRIORITY_MID : PRIORITY_LOW);
        }
      }
    }

    // --- drum ----------------------------------------------------------------
    if (p.drum !== 'none' && this.on('drum')) {
      let level = 0;
      let stroke: DrumStroke = 'up';
      if (p.drum === 'jig') {
        // Accents on the two beats; the inner strokes fill in with speed.
        const accent = [1, 0.32, 0.55, 0.85, 0.32, 0.55][s];
        const gate = s === 0 || s === 3 ? 0 : s === 2 || s === 5 ? 0.2 : 0.55;
        if (e >= gate) level = accent * (0.75 + 0.25 * e);
        stroke = s === 0 || s === 3 ? 'down' : 'up';
      } else if (p.drum === 'toms') {
        const accent = [1, 0, 0, 0.8, 0, 0.45][s];
        level = accent;
        stroke = 'tom';
        // A roll into every fourth bar.
        if (this.barIndex % 4 === 3 && s >= 3) {
          level = 0.45 + 0.15 * (s - 3);
          this.drum(t + d * 0.5, level * 0.8 * VEL.drum, 'tom', PRIORITY_LOW);
        }
      } else if (p.drum === 'pulse') {
        level = s === 0 ? 0.7 : s === 3 ? 0.4 : 0;
        stroke = s === 0 ? 'down' : 'up';
      }
      if (level > 0) {
        const humanise = (random() - 0.5) * 0.008;
        this.drum(t + Math.max(0, humanise), level * VEL.drum, stroke, s === 0 ? PRIORITY_MID : PRIORITY_LOW);
      }
    }

    // --- bass ---------------------------------------------------------------------
    if (p.bass !== 'none' && this.on('bass')) {
      const root = atOrAbove(bar.chord.root, 38);
      if (p.bass === 'beats') {
        if ((s === 0 || s === 3) && this.canPlay(PRIORITY_MID)) {
          playBass(this, this.inputs.bass, t, s === 0 ? root : root + 7 > 50 ? root - 5 : root + 7, 2.6 * d, VEL.bass);
        }
      } else if (this.canPlay(PRIORITY_MID)) {
        // The storm's ostinato: every quaver, leaning on the beats.
        const v = s === 0 || s === 3 ? VEL.bass * 1.25 : VEL.bass * 0.8;
        playBass(this, this.inputs.bass, t, root, 0.85 * d, v);
      }
    }

    // --- the hurdy-gurdy's trompette -----------------------------------------
    if (p.buzz > 0 && this.on('drone')) {
      const beat = s === 0 || s === 3;
      if (beat || e > 0.6) {
        const level = VEL.buzz * p.buzz * (beat ? 1 : 0.45) * (0.6 + 0.4 * e);
        const g = this.buzzGate.gain;
        g.setValueAtTime(0, t);
        g.linearRampToValueAtTime(level, t + 0.012);
        g.setTargetAtTime(0, t + 0.035, 0.05);
      }
    }
  }

  /** Everything that happens once per bar: the chord. */
  private scheduleBar(t: number, d: number, bar: Bar): void {
    const p = this.profile;
    const v = this.scratch;
    const barDur = STEPS_PER_BAR * d;
    if (p.reeds !== 'none' && this.on('reeds')) {
      if (p.reeds === 'low') voicing(bar.chord, 45, 58, v);
      else voicing(bar.chord, 54, 67, v);
      const attack = Math.min(0.4, barDur * 0.15);
      for (let i = 0; i < v.length && i < 4; i++) {
        if (!this.canPlay(PRIORITY_MID)) break;
        playReed(
          this,
          this.inputs.reeds,
          this.reedWave,
          t,
          v[i],
          barDur * 0.98,
          VEL.reeds,
          attack,
          0.25,
          (this.random() - 0.5) * 12,
        );
      }
    }
    if (this.on('glass')) {
      voicing(bar.chord, 57, 71, v);
      for (let i = 0; i < v.length && i < 3; i++) {
        if (!this.canPlay(PRIORITY_MID)) break;
        playGlass(this, this.inputs.glass, t, v[i], barDur * 1.05, VEL.glass);
      }
    }
  }

  private playTune(
    who: 'fiddle' | 'squeeze' | 'glass',
    t: number,
    midi: number,
    dur: number,
    len: number,
    harmony: boolean,
  ): void {
    if (!this.on(who)) return;
    const priority = harmony ? PRIORITY_MID : PRIORITY_HIGH;
    if (!this.canPlay(priority)) return;
    const random = this.random;
    const scale = harmony ? 0.62 : 1;
    switch (who) {
      case 'fiddle': {
        // Ornaments on the tune only, and more of them the harder she is driven.
        const r = random();
        const ornate = !harmony && len >= 2 && r < 0.22 + 0.25 * this.energy;
        const cut = ornate && r < 0.6 * (0.22 + 0.25 * this.energy) ? midi + 2 : null;
        const slide = ornate && cut === null;
        const vel = VEL.fiddle * scale * (0.9 + 0.2 * random());
        playFiddle(this, this.inputs.fiddle, t, midi, dur * 1.02, vel, { cut, slide }, random);
        break;
      }
      case 'squeeze': {
        const vel = VEL.squeeze * scale * (0.9 + 0.15 * random());
        playReed(this, this.inputs.squeeze, this.reedWave, t, midi, dur * 0.97, vel, 0.035, 0.09, (random() - 0.5) * 6);
        break;
      }
      case 'glass': {
        // Under water only the long notes of the tune surface, as bells.
        if (len < 2 || random() < 0.25) return;
        playBell(this, this.inputs.glass, t, midi + 12, VEL.bell * scale);
        break;
      }
    }
  }

  private pluck(t: number, midi: number, vel: number, dur: number, priority: number): void {
    if (!this.canPlay(priority)) return;
    const pc = ((midi % 12) + 12) % 12;
    const base = PLUCK_BASE + pc;
    let buffer = this.plucks.get(pc);
    if (buffer === undefined) {
      buffer = renderPluck(this.ctx, base, this.pluckRandom);
      this.plucks.set(pc, buffer);
    }
    playPluck(this, this.inputs.pluck, buffer, Math.pow(2, (midi - base) / 12), t, vel, dur);
  }

  private drum(t: number, vel: number, stroke: DrumStroke, priority: number): void {
    if (!this.canPlay(priority)) return;
    playDrum(this, this.inputs.drum, this.white, t, vel, stroke, this.random);
  }
}

function createPanner(ctx: BaseAudioContext, pan: number): StereoPannerNode | null {
  const factory = (ctx as BaseAudioContext & { createStereoPanner?: () => StereoPannerNode })
    .createStereoPanner;
  if (typeof factory !== 'function') return null;
  const node = ctx.createStereoPanner();
  node.pan.value = pan;
  return node;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
