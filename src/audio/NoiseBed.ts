import { ParamTarget } from './ParamTarget';

/**
 * One continuous noise layer: a looping source, a high-pass, a shaping filter, a
 * gain and an optional pan.
 *
 * Surf, wind, rigging, rain, rain hiss, hull rush and the underwater rumble are
 * all this same chain with different filter settings and different numbers
 * driving them. That is not a coincidence or an economy — a broadband natural
 * sound *is* a noise source with a moving band, and the thing that distinguishes
 * wind from surf to the ear is almost entirely where that band sits and how it
 * moves.
 *
 * **Why the high-pass is a separate stage.** The shaping filter has to be free to
 * open right up — surf at 25 m/s runs to a couple of kilohertz — and a single
 * band-pass wide enough for that also passes everything underneath it. Sub-60 Hz
 * content in the air layers is not audible on the devices this runs on, but it is
 * very much present in the signal, and it eats the compressor's headroom on
 * behalf of sound nobody hears. The fixed high-pass removes it once, ahead of
 * everything.
 *
 * **Switching off means disconnecting, not muting.** A layer turned off by the
 * quality tier has to stop costing something, and a gain of zero costs exactly as
 * much as a gain of one — the audio thread still runs the source and both filters.
 * Disconnecting the tail makes the whole chain unreachable from the destination,
 * and an unreachable node is not rendered. The disconnect is deferred by
 * `DEACTIVATE_SECONDS` so the gain ramp finishes first; a disconnect mid-ramp is
 * a click, and a tier change is meant to be inaudible.
 *
 * The same goes for a bed asked for silence by the scene rather than the tier:
 * rain in a dry sky, the hull rush at anchor. Held at zero for ten of its own
 * gain time constants, it unhooks itself (`PARK_LEVEL`), and the next level
 * above zero hooks it back up from exactly zero.
 *
 * The source is started once and never stopped, because an `AudioBufferSourceNode`
 * cannot be restarted — a stopped one is dead and would have to be replaced, which
 * is an allocation on a path that can be hit by the adaptive quality manager.
 * Reconnecting resumes at wherever the shared clock says the loop has got to,
 * which for noise is indistinguishable from having never stopped.
 *
 * That one start is deliberately *not* in the constructor. Starting a source on a
 * context that has not been unlocked makes Chrome log an autoplay warning per
 * source, so building the graph up front would put eight warnings in the console
 * of every session before the viewer has done anything — indistinguishable, to
 * anyone reading that console, from a real problem. `start` is idempotent and the
 * system calls it on the first resume.
 */

/** Delay between a layer's gain reaching zero and its chain being unhooked. */
const DEACTIVATE_SECONDS = 0.4;
/**
 * A level at or under this is silence, and a bed held there this many of its
 * own gain time constants (at least `PARK_MIN_SECONDS`) is unhooked until it is
 * asked for a level again: rain in a dry sky, the hull rush at anchor and the
 * rumble above water were ten beds rendering into gains of zero.
 */
const PARK_LEVEL = 1e-4;
const PARK_TAUS = 10;
const PARK_MIN_SECONDS = 1;
/**
 * From a bed's estimated output power to the logical loudness scale: the
 * LUFS offset (-0.691) plus the master compressor's makeup, fitted against
 * the bench's world-only renders.
 */
const LOUDNESS_OFFSET_DB = 8;
/**
 * Loudness weighting on a brown bed: nearly all of its power is under 100 Hz,
 * where BS.1770's high-pass discounts it. Fitted against the underwater render.
 */
const BROWN_WEIGHTING_DB = -7;
/** The brown buffer's corner, Hz (`noise.ts`). */
const BROWN_CORNER_HZ = 20;

export interface NoiseBedOptions {
  /** Shared looping noise. Not owned — several beds read the same buffer. */
  buffer: AudioBuffer;
  /** Bus this layer feeds. */
  destination: AudioNode;
  /** Shape of the moving filter. */
  filter: BiquadFilterType;
  frequency: number;
  q: number;
  /** Corner of the fixed high-pass ahead of the band. */
  highpass: number;
  /**
   * Rate this bed reads the shared buffer at. Distinct per bed, and mutually
   * irrational where it can be arranged, so no two layers repeat in step.
   */
  playbackRate: number;
  /** Seconds into the shared buffer this bed starts, for the same reason. */
  offset: number;
  pan: number;
  /** Time constant for level moves. Longer for beds that should swell. */
  gainTau?: number;
  /** Time constant for filter moves. */
  filterTau?: number;
  /**
   * The buffer's spectrum, for `loudnessDb`: white (the default) or brown
   * (a one-pole at `BROWN_CORNER_HZ`, as `noise.ts` makes it).
   */
  spectrum?: 'white' | 'brown';
}

export class NoiseBed {
  private readonly source: AudioBufferSourceNode;
  private readonly highpassNode: BiquadFilterNode;
  private readonly bandNode: BiquadFilterNode;
  private readonly gainNode: GainNode;
  private readonly panNode: StereoPannerNode | null;
  /** Last node in the chain — the one connected to (and cut from) the bus. */
  private readonly tail: AudioNode;
  private readonly destination: AudioNode;

  private readonly level: ParamTarget;
  private readonly frequency: ParamTarget;
  private readonly resonance: ParamTarget;
  private readonly highpassFrequency: ParamTarget;
  private readonly pan: ParamTarget | null;

  /** Every target this bed owns, so the system can flush them all at once. */
  readonly targets: ParamTarget[];

  /** Seconds into the shared buffer this bed starts. See `start`. */
  private readonly startOffset: number;

  private active = true;
  private started = false;
  private disconnectAt: number | null = null;
  private disposed = false;
  /** Whether the tail is hooked to the bus, whatever the reason. */
  private connected = true;
  /** Unhooked for sitting at zero level (not by the tier). */
  private parked = false;
  /** When the level was last asked to be silence, or null while it is not. */
  private quietSince: number | null = null;
  private readonly parkAfter: number;
  private readonly filterType: BiquadFilterType;
  private readonly nyquist: number;
  /** Mean square of the buffer, for `loudnessDb`. */
  private readonly power: number;
  private readonly brown: boolean;
  /** The read rate's own wander (A5): base, period and phase. */
  private readonly baseRate: number;
  private readonly wanderPeriod: number;
  private readonly wanderPhase: number;
  private readonly rate: ParamTarget;

  constructor(ctx: BaseAudioContext, options: NoiseBedOptions) {
    // Ceiling on any filter corner: nine tenths of Nyquist, not Nyquist. A
    // biquad's bilinear-transform coefficients degenerate as the corner
    // approaches half the sample rate, and a resonant filter parked there is not
    // quiet — it is unstable, which is a burst of noise rather than the silence
    // the caller was reaching for.
    const maxFrequency = ctx.sampleRate * 0.45;

    this.destination = options.destination;
    this.filterType = options.filter;
    this.nyquist = ctx.sampleRate / 2;
    this.power = bufferPower(options.buffer);
    this.brown = options.spectrum === 'brown';
    this.baseRate = options.playbackRate;
    // Incommensurate per bed, from its own base rate and offset: 23 to 37 s.
    this.wanderPeriod = 23 + ((options.playbackRate * 97.13 + options.offset * 13.7) % 14);
    this.wanderPhase = (options.offset * 2.399) % (2 * Math.PI);

    this.source = ctx.createBufferSource();
    this.source.buffer = options.buffer;
    this.source.loop = true;
    this.source.playbackRate.value = options.playbackRate;

    this.highpassNode = ctx.createBiquadFilter();
    this.highpassNode.type = 'highpass';
    this.highpassNode.Q.value = 0.7;

    this.bandNode = ctx.createBiquadFilter();
    this.bandNode.type = options.filter;

    this.gainNode = ctx.createGain();
    this.gainNode.gain.value = 0;

    this.source.connect(this.highpassNode);
    this.highpassNode.connect(this.bandNode);
    this.bandNode.connect(this.gainNode);

    // `createStereoPanner` is universal on the browsers this project targets, but
    // it is the one node here that is not, and a missing panner should cost the
    // layer its width and nothing else.
    const factory = (ctx as BaseAudioContext & { createStereoPanner?: () => StereoPannerNode })
      .createStereoPanner;
    if (typeof factory === 'function') {
      this.panNode = ctx.createStereoPanner();
      this.gainNode.connect(this.panNode);
      this.tail = this.panNode;
    } else {
      this.panNode = null;
      this.tail = this.gainNode;
    }
    this.tail.connect(this.destination);

    const gainTau = options.gainTau ?? 0.15;
    const filterTau = options.filterTau ?? 0.25;
    this.parkAfter = Math.max(PARK_MIN_SECONDS, PARK_TAUS * gainTau);

    this.level = new ParamTarget(this.gainNode.gain, 0, gainTau, 0.002, 0, 4);
    this.frequency = new ParamTarget(
      this.bandNode.frequency,
      options.frequency,
      filterTau,
      3,
      20,
      maxFrequency,
    );
    this.resonance = new ParamTarget(this.bandNode.Q, options.q, filterTau, 0.02, 0.05, 24);
    this.highpassFrequency = new ParamTarget(
      this.highpassNode.frequency,
      options.highpass,
      filterTau,
      3,
      20,
      maxFrequency,
    );
    this.rate = new ParamTarget(this.source.playbackRate, options.playbackRate, 0.5, 0.002, 0.25, 4);
    this.pan =
      this.panNode === null
        ? null
        : new ParamTarget(this.panNode.pan, options.pan, 0.3, 0.01, -1, 1);

    this.targets = [this.level, this.frequency, this.resonance, this.highpassFrequency];
    if (this.pan !== null) this.targets.push(this.pan);

    // Offset into the shared buffer, so beds reading the same noise are not
    // reading the same *sample* of it. Taken modulo the duration here rather than
    // at `start`: an offset past the end of the buffer is a range error, not a
    // wrap.
    const duration = options.buffer.duration;
    this.startOffset = duration > 0 ? options.offset % duration : 0;
  }

  /**
   * Lets the read rate drift a few per cent around its base, on this bed's
   * own period, so its loop through the buffer is never exact (A5). Called
   * every frame; writes only when the rate has moved.
   */
  wander(now: number, depth: number, live: boolean): void {
    if (!this.started || !this.connected) return;
    const w = Math.sin((2 * Math.PI * now) / this.wanderPeriod + this.wanderPhase);
    this.rate.set(this.baseRate * (1 + depth * w), now, live);
  }

  /** Begins playback. Idempotent — a second call is a no-op, not an exception. */
  start(when: number): void {
    if (this.started || this.disposed) return;
    this.started = true;
    this.source.start(when, this.startOffset);
  }

  /**
   * Level, before the tier gate. An inactive bed pins itself to silence rather
   * than trusting the caller to stop asking for a level.
   */
  setLevel(value: number, now: number, live: boolean): void {
    const wanted = this.active ? value : 0;
    // Parking only moves on a running clock; a frozen one never reaches a deadline.
    if (this.active && live && this.started) {
      if (wanted > PARK_LEVEL) {
        this.quietSince = null;
        if (this.parked) {
          // Back from silence: rehooked at exactly zero, then ramped up.
          this.parked = false;
          this.level.silence(now);
          this.hook();
        }
      } else if (!this.parked) {
        if (this.quietSince === null) {
          this.quietSince = now;
        } else if (now - this.quietSince >= this.parkAfter) {
          // Ten time constants down, the gain is already under -80 dB of where
          // it was; zeroing it and unhooking is inaudible.
          this.parked = true;
          this.level.silence(now);
          this.unhook();
        }
      }
    }
    this.level.set(wanted, now, live);
  }

  /**
   * Logical loudness of what this bed is being asked to play, dB on the scale
   * `HdrMix` uses (roughly LUFS before the master stage, `LOUDNESS_OFFSET_DB`
   * fitted by the bench). Estimated from the level and the band: white noise's
   * power through a filter is its power per hertz times the filter's
   * equivalent noise bandwidth, less what the fixed high-pass takes away.
   * K-weighting is left out; the beds that matter sit in its flat middle.
   */
  loudnessDb(): number {
    const level = this.active ? this.level.target : 0;
    if (level <= 1e-5) return -Infinity;
    const f = this.frequency.target;
    const q = Math.max(0.05, this.resonance.target);
    const hp = this.highpassFrequency.target;
    if (this.brown) {
      // Power of a one-pole spectrum between the high-pass and the corner,
      // as a fraction of all of it: (atan(f/fc) - atan(hp/fc)) / (pi/2).
      const top = this.filterType === 'lowpass' ? f : this.nyquist;
      const share = Math.max(0.02, (Math.atan(top / BROWN_CORNER_HZ) - Math.atan(hp / BROWN_CORNER_HZ)) / (Math.PI / 2));
      return 10 * Math.log10(Math.max(1e-12, level * level * this.power * share)) + LOUDNESS_OFFSET_DB + BROWN_WEIGHTING_DB;
    }
    let band: number;
    switch (this.filterType) {
      case 'lowpass':
        band = Math.max(0.15 * f, 1.11 * f - hp);
        break;
      case 'highpass':
        band = Math.max(1, this.nyquist - Math.max(f, hp));
        break;
      default:
        // A band-pass's equivalent noise bandwidth is pi/2 of its -3 dB width.
        band = Math.max(1, (Math.PI / 2) * (f / q) - Math.max(0, hp - f / 2) * 0.5);
        break;
    }
    const power = level * level * this.power * (band / this.nyquist);
    return 10 * Math.log10(Math.max(1e-12, power)) + LOUDNESS_OFFSET_DB;
  }

  /** Whether this bed is hooked up and therefore being rendered. */
  get rendering(): boolean {
    return this.connected;
  }

  private hook(): void {
    if (this.connected || this.disposed) return;
    this.tail.connect(this.destination);
    this.connected = true;
  }

  private unhook(): void {
    if (!this.connected) return;
    this.tail.disconnect();
    this.connected = false;
  }

  setBand(frequency: number, q: number, now: number, live: boolean): void {
    this.frequency.set(frequency, now, live);
    this.resonance.set(q, now, live);
  }

  setHighpass(frequency: number, now: number, live: boolean): void {
    this.highpassFrequency.set(frequency, now, live);
  }

  setPan(value: number, now: number, live: boolean): void {
    this.pan?.set(value, now, live);
  }

  /** Tier gate. See the note on disconnecting rather than muting. */
  setActive(active: boolean, now: number, live: boolean): void {
    if (this.disposed || active === this.active) return;
    this.active = active;

    if (active) {
      // A pending disconnect means the chain is still hooked up and only the
      // timer needs cancelling.
      if (this.disconnectAt !== null) this.disconnectAt = null;
      else this.hook();
      this.parked = false;
      this.quietSince = null;
      this.level.silence(now);
      return;
    }

    if (live) {
      this.level.set(0, now, true);
      this.disconnectAt = this.connected ? now + DEACTIVATE_SECONDS : null;
    } else {
      // Nothing is audible, so there is no ramp to protect and no clock to wait
      // on — a suspended context would never reach the deadline anyway.
      this.level.silence(now);
      this.disconnectAt = null;
      this.unhook();
    }
  }

  /** Retires a deferred disconnect. Called every frame; almost always a compare. */
  tick(now: number): void {
    if (this.disconnectAt === null || now < this.disconnectAt) return;
    this.disconnectAt = null;
    this.unhook();
  }

  flush(now: number): void {
    for (const target of this.targets) target.flush(now);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.connected = false;
    if (this.started) {
      try {
        this.source.stop();
      } catch {
        // Already stopped, or the context died first.
      }
    }
    this.source.disconnect();
    this.highpassNode.disconnect();
    this.bandNode.disconnect();
    this.gainNode.disconnect();
    this.panNode?.disconnect();
  }
}

/** Mean square of a buffer's first channel, sampled sparsely. */
function bufferPower(buffer: AudioBuffer): number {
  const data = buffer.getChannelData(0);
  let sum = 0;
  let n = 0;
  for (let i = 0; i < data.length; i += 7) {
    sum += data[i] * data[i];
    n++;
  }
  return n > 0 ? sum / n : 0;
}
