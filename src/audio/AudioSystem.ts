import type { QualityTier } from '../core/QualityManager';
import { mulberry32 } from '../core/random';
import { NoiseBed, type NoiseBedOptions } from './NoiseBed';
import { NOISE_SECONDS, createNoiseBuffer } from './noise';
import { ParamTarget } from './ParamTarget';
import { MusicEngine, type MusicState, type MusicSting } from './music/MusicEngine';
import { ShipSfx, type ShipMotion, type SfxHost } from './ShipSfx';

/**
 * The scene's ambience, synthesised rather than sampled.
 *
 * There is no reliably-licensed CC0 ocean or wind recording this project could
 * fetch reproducibly — and the licensing is the smaller half of the argument. A
 * recording is a photograph of one sea. This scene changes wind speed, wave
 * height, rain rate, hull speed and submersion *live and continuously*, from a
 * slider the viewer is holding, and a fixed loop can only be cross-faded between:
 * which is audible as exactly what it is, two seas playing at once, each still
 * describing the wind it was recorded in. Filtered noise whose band and level
 * *are* functions of the wind speed has no such seam, because there is nothing to
 * cross-fade — there is one sea and the wind speed is a term in it. That is also
 * how a parameterised sea is actually built; sample libraries are for sounds that
 * do not have a parameter.
 *
 * The layers, and what drives each:
 *
 *  - **Surf.** Low-passed noise, level linear in wind speed and band opening with
 *    it, breathing on the swell period implied by the peak wavelength.
 *  - **Wind.** A separate, higher band that climbs far faster than the sea does,
 *    plus an aeolian tone off the rigging once it is blowing hard enough to sing.
 *  - **Rain.** Denser and brighter than the wind, with a separate hiss layer and a
 *    low drumming that only appears in a downpour.
 *  - **Hull.** Bow wash and the rush along the side, from hull speed.
 *  - **Underwater.** Not a switch: `submersion` continuously ducks and closes a
 *    low-pass over everything above, brings up a rumble, and starts entraining
 *    bubbles.
 *  - **Gulls.** The one thing in here that is alive, and the only layer that is
 *    not continuous: a synthesised cry, scheduled in sparse flurries against the
 *    flock the renderer is actually drawing, silent under water and thinned out
 *    by rain and by a gale.
 *  - **The ship.** Its own bus, gained by distance and panned by bearing: the
 *    hull rush and a bow wave that surges as the stem plunges, creaks and timber
 *    groans from how fast she rolls and pitches under load, blocks knocking,
 *    canvas flogging when it luffs and a *whump* when it fills (`ShipSfx`).
 *  - **The score.** An original, synthesised sea-song band whose arrangement
 *    follows the game (`music/MusicEngine`): calm, sailing (tempo and drive from
 *    speed), storm, underwater, tour and silent, crossfading by stem and
 *    switching material at the barline. It ducks under thunder and stings.
 *
 * **Hooks** — what other systems call. All are safe before the unlocking
 * gesture, with no Web Audio, and when muted (they do nothing):
 *
 *  - `update(dt, params)` — every frame. `AudioSceneParams` carries the sea, the
 *    weather, submersion and the ship (`shipDistance`, `shipPan`, `shipHeel`,
 *    `shipPitch`, `sailFill`, `sailLuff`); every ship field is optional.
 *  - `setMusicState(state, energy?)`, `setMusicEnergy(0..1)`,
 *    `setMusicVolume(0..1)`, `playSting('cargo')` — the score. Call
 *    `setMusicState` every frame with what the game implies; it holds a state
 *    for ~2 s (0.35 s for the waterline) before following.
 *  - `hullImpact(strength 0..1, grind = false)` — a slam, a rock, the shore.
 *  - `waveSlap(strength 0..1, pan -1..1 = 0)` — water hitting the planking.
 *    `SlapDetector` turns hull probes into these; a shore or impact system can
 *    call it directly.
 *  - `playSplash(0..1)`, `playThunder(km, delay)`, `playChime()` (the reward:
 *    chime plus the cargo sting), `playAnchorChain(seconds)`,
 *    `playGullCry(m, pan)`.
 *  - `uiTick(cue, value?)` — interface cues (`UiCue`); `src/ui/sounds.ts` maps
 *    DOM events to them so the UI never imports this.
 *  - `setGullLocator(fn)` — places cries on real birds (`Birds.positionOf`).
 *
 * **Mix.** Beds, ship and one-shots sum on the air bus (muffled under water);
 * the score joins them before a gentle 4:1 compressor; UI bypasses both. After
 * the master volume a 20:1 limiter at -1 dBFS and a soft clipper guarantee the
 * output never reaches full scale. Measured offline (master 0.8, music 0.5):
 * calm -32 dB RMS, under sail -27, storm -22, ambient scenes peaking below
 * -6.5 dBFS and a full-strength grounding at -4.5; the deliberate worst case
 * (volume 1, close strike + grounding + sting on one sample) peaks at
 * -0.6 dBFS with no clipped samples. The score sits about level with a calm
 * sea, ~3 dB under the sea at sail and ~5 dB under a storm.
 *
 * Three constraints shaped the implementation more than the sound design did.
 *
 * **Autoplay.** An `AudioContext` may not start before a user gesture. This one
 * is constructed and then explicitly suspended, its master gain is pinned at zero
 * until `resume()` has been called from a gesture, and if the platform has no Web
 * Audio at all every method here is a no-op that returns rather than throwing.
 * Audio is the one subsystem that must never be able to take the frame down.
 *
 * **`update` allocates nothing.** It runs on the render loop. Every node exists
 * from construction, every parameter write goes through a `ParamTarget` that
 * smooths, dedupes and drops non-finite values, and the layer model is plain
 * arithmetic on preallocated fields. One-shots do allocate — a source node cannot
 * be restarted, so an event has to build its voice — but they are events, not
 * frames, and the voice count is capped per tier so a storm cannot turn into an
 * allocation storm.
 *
 * **Silence has to be free.** Under the capture harness this is muted for the
 * whole run. Muted, and while the context is not running, no parameter is written
 * at all: the model still advances so `resetClock` means something, but nothing
 * touches the audio thread. That is not only cheaper, it removes any path by which
 * scheduling audio could perturb the frame timing of a measurement.
 */

/** Scene state the ambience is a function of. */
export interface AudioSceneParams {
  /** Wind speed at the 10 m reference height, m/s. `UiState.windSpeed`. */
  windSpeed: number;
  /** Significant wave height, metres. `significantWaveHeight( spectrumParams )`. */
  waveHeight: number;
  /** JONSWAP peak wavelength, metres. `UiState.peakWavelength`. */
  peakWavelength: number;
  /** Rain reaching the sea, 0..1 — the value `Wake.setRainAgitation` is given. */
  rain: number;
  /** Camera submersion, 0..1. `CameraDirector.submersion()`. */
  submersion: number;
  /** Hull speed over ground, m/s. `ShipControlState.speed`; 0 with no ship. */
  hullSpeed: number;
  /**
   * Gulls currently being drawn. `Birds.getCount()`, which is the quality tier's
   * `birds`.
   *
   * A count rather than a reference to the flock, because the audio has no
   * business knowing what a bird is — and because this is the one number that
   * matters to it. It is what makes Low, which draws none, hear none, and it is
   * what stops the sky sounding equally busy whether there are fourteen birds in
   * it or sixty-four.
   *
   * Optional, and 0 by default: a system that has not been told there are birds
   * does not invent them.
   */
  birdCount?: number;
  /**
   * Metres from the listener to the ship, or `Infinity` with no ship. Everything
   * the hull, rigging and canvas make — bow wash included — is gained by this,
   * so a ship across the bay is a murmur rather than a creak in the ear.
   */
  shipDistance?: number;
  /** Where the ship is, -1 (left) .. 1 (right), from the listener's view. */
  shipPan?: number;
  /** Hull roll, radians, positive starboard rail down. Drives creaks and groans. */
  shipHeel?: number;
  /** Hull pitch, radians, positive bow up. Drives creaks and the bow wash's surge. */
  shipPitch?: number;
  /** Canvas set, 0..1. */
  sailFill?: number;
  /** How hard the canvas is luffing (head to wind, or being set or taken in), 0..1. */
  sailLuff?: number;
}

export const DEFAULT_AUDIO_SCENE_PARAMS: AudioSceneParams = {
  windSpeed: 15,
  waveHeight: 5,
  peakWavelength: 47,
  rain: 0,
  submersion: 0,
  hullSpeed: 0,
  birdCount: 0,
  shipDistance: Number.POSITIVE_INFINITY,
  shipPan: 0,
  shipHeel: 0,
  shipPitch: 0,
  sailFill: 0,
  sailLuff: 0,
};

/** A gull's place relative to the listener, as `setGullLocator` reports it. */
export interface GullFix {
  /** Metres. */
  distance: number;
  /** -1 (left) .. 1 (right). */
  pan: number;
}

/**
 * Interface sounds, as `uiTick` takes them. `src/ui/sounds.ts` maps DOM events
 * onto these; the strings are the whole contract, so the UI never imports audio.
 */
export type UiCue =
  | 'hover'
  | 'press'
  | 'select'
  | 'slide'
  | 'toggleOn'
  | 'toggleOff'
  | 'open'
  | 'close'
  | 'info'
  | 'warn';

export type { MusicState, MusicSting };

export interface AudioQualitySettings {
  /** A second, opposite-panned surf bed. Without it the sea is mono. */
  stereoSurf: boolean;
  /** The resonant aeolian tone off the standing rigging. */
  rigging: boolean;
  /** A separate high-frequency hiss layer over the rain. */
  rainHiss: boolean;
  /**
   * Bubble blips per second at full submersion in a reference sea. 0 disables the
   * scheduler outright, which is what makes Low genuinely oscillator-free.
   */
  bubbleRate: number;
  /**
   * Calling flurries per second from a reference flock, in a calm. 0 silences
   * the gulls outright, which is what Low needs: it draws no birds at all, and
   * hearing one from an empty sky is worse than hearing nothing.
   *
   * A flurry is one to three overlapping calls — see `scheduleGullFlurry` — so
   * the rate of *cries* is roughly half again this.
   */
  gullRate: number;
  /**
   * Concurrent one-shot voices. Events past the cap are dropped, never queued —
   * a queued splash arrives after the thing that splashed has gone.
   */
  voices: number;
  /** Concurrent music notes. The score drops ghost notes first when it is over. */
  musicVoices: number;
  /** The music's room: a convolution reverb, or on Low a cheap feedback delay. */
  musicReverb: 'convolver' | 'delay';
}

/**
 * Tiers, in the same shape and order as `QUALITY_TIERS`.
 *
 * What changes between them is the number of sources actually being rendered:
 * Low runs five noise beds and no oscillators at all, Max runs eight beds and up
 * to twelve concurrent voices. Nothing here changes the *sound* of a layer, only
 * whether it is present — a tier drop should read as the sea getting simpler, not
 * as somebody moving the mix.
 */
export const AUDIO_QUALITY_TIERS: Record<QualityTier, AudioQualitySettings> = {
  // The one-shot caps were raised when the ship got a voice of its own: creaks,
  // blocks and canvas are ambience and are held two short of the cap (see
  // `SHIP_VOICE_RESERVE`), so the old caps would have left Low unable to creak.
  low: {
    stereoSurf: false, rigging: false, rainHiss: false, bubbleRate: 0, gullRate: 0, voices: 6,
    musicVoices: 14, musicReverb: 'delay',
  },
  medium: {
    stereoSurf: false, rigging: true, rainHiss: true, bubbleRate: 1.2, gullRate: 0.055, voices: 9,
    musicVoices: 22, musicReverb: 'convolver',
  },
  high: {
    stereoSurf: true, rigging: true, rainHiss: true, bubbleRate: 2.0, gullRate: 0.08, voices: 12,
    musicVoices: 30, musicReverb: 'convolver',
  },
  ultra: {
    stereoSurf: true, rigging: true, rainHiss: true, bubbleRate: 2.6, gullRate: 0.095, voices: 14,
    musicVoices: 34, musicReverb: 'convolver',
  },
  max: {
    stereoSurf: true, rigging: true, rainHiss: true, bubbleRate: 3.4, gullRate: 0.11, voices: 16,
    musicVoices: 40, musicReverb: 'convolver',
  },
};

export interface AudioSystemOptions {
  /** Master volume, 0..1. */
  volume?: number;
  /** Starts muted. Defaults to false; the capture harness sets it. */
  muted?: boolean;
  quality?: QualityTier;
  /** Music volume, 0..1, under the master volume. Defaults to 0.5. */
  musicVolume?: number;
  /**
   * Render into this context instead of creating a realtime one — an
   * `OfflineAudioContext`, for verification. The system then behaves as though
   * the gesture has happened and the context is running; the caller drives
   * `update` (e.g. from `suspend()` points) and renders.
   */
  context?: BaseAudioContext;
}

/**
 * Seeds for the procedural content here.
 *
 * Local rather than added to `SEEDS` in `core/random`, and literal constants for
 * the same reason that module gives: a value that drifts makes two runs
 * incomparable, so it is written where a change to it shows up in a diff.
 */
const AUDIO_SEEDS = {
  white: 0x0ce4a1,
  brown: 0x5b0d33,
  bubbles: 0xb0bb1f,
  gusts: 0x9c5751,
  oneShots: 0x5c1a5d,
  thunderShape: 0x7d0c4a,
  // Deliberately not the flock's own seed. The cries are not tied to individual
  // birds — nothing here knows where bird 7 is — and sharing a seed would imply
  // a correspondence that does not exist.
  gulls: 0x9b1ac5,
} as const;

/** Clock wrap, seconds. Matches the other animated systems in this project. */
const CLOCK_WRAP = 3600;

const TAU = Math.PI * 2;
const GRAVITY = 9.81;

// --- master ----------------------------------------------------------------

/**
 * Compressor settings for the master bus.
 *
 * Not for loudness. The layers are summed with fixed gains chosen so a typical
 * sea sits comfortably, but the *worst* case is a full storm with rain, a hull at
 * speed and a close thunder crack landing on the same sample — and that case is
 * both rare and roughly 4x the typical one, so mixing to leave headroom for it
 * would leave everything else inaudibly quiet. A gentle 4:1 above -12 dBFS with a
 * soft knee costs one node and makes the loud case merely loud instead of clipped.
 * The 6 ms attack is deliberately slower than a limiter's: it lets the leading
 * edge of a thunder crack through, which is the only part of a crack that sounds
 * like one.
 */
const COMPRESSOR_THRESHOLD_DB = -12;
const COMPRESSOR_KNEE_DB = 14;
const COMPRESSOR_RATIO = 4;
const COMPRESSOR_ATTACK = 0.006;
const COMPRESSOR_RELEASE = 0.28;

/** Fades applied by the volume, mute and enable controls, seconds. */
const VOLUME_RAMP = 0.05;
const MUTE_RAMP = 0.02;
const ENABLE_RAMP = 0.2;
/** Fade-in on the first resume. A sea that cuts in at full level reads as a bug. */
const RESUME_RAMP = 0.6;
/** Grace between a disable fading out and the context being suspended. */
const SUSPEND_DELAY = 0.35;

// --- swell -----------------------------------------------------------------

/**
 * Waves per group.
 *
 * What makes a sea audibly *surge* rather than hiss is not individual waves, it
 * is groups of them, and the folklore figure of every seventh wave is close
 * enough to the beat period of a narrow wind-sea spectrum to use directly. Both
 * periods run at once, which is why the surge never lands on the same beat twice.
 */
const GROUP_WAVES = 7;
/** Phase offset between the two modulators, so they cannot start in step. */
const GROUP_PHASE = 1.7;
/** Phase the far surf bed lags the near one by, so the stereo image is not pumped. */
const SURF_FAR_PHASE = Math.PI / 3;
/** Modulation depth at a flat calm, and the span added by a full sea. */
const SWELL_DEPTH_CALM = 0.1;
const SWELL_DEPTH_SPAN = 0.46;
/** Significant wave height, metres, at which the surge is fully developed. */
const SWELL_FULL_HEIGHT = 5;

// --- surf ------------------------------------------------------------------
//
// A note on the whole resting mix, because the numbers below were retuned once.
//
// The first pass was voiced for *presence*: play it for ten seconds and the sea
// is obviously there. That is the wrong target. This runs continuously while
// someone works in the scene, and the thing that makes a noise bed exhausting is
// broadband energy above about 2 kHz — the band the ear is most sensitive to and
// the one a filtered-noise sea has no business occupying. The complaint was that
// it hissed rather than breathed, which is exactly that band.
//
// So: every corner frequency below is darker, the wind layer — the harshest of
// them — is roughly halved, and the swell modulation is deeper so the level moves
// rather than sitting flat. The sea should be something you stop noticing.

/**
 * Wind speed the surf bed is referenced to, m/s, and the ceiling on that ratio.
 *
 * Amplitude linear in wind speed is not a taste decision. The Knudsen/Wenz
 * ambient-sea-noise curves put the broadband level up by about 6 dB per doubling
 * of wind speed across the band that matters here, and 6 dB *is* a factor of two
 * in amplitude. 12 m/s is a fresh breeze — near the middle of the 0.5..25 range
 * the slider offers, so neither end of it is extrapolating far.
 */
const SURF_REFERENCE_WIND = 12;
const SURF_MAX_DRIVE = 1.5;
/** Level at a flat calm. A glassy sea is not silent; it slops. */
const SURF_CALM = 0.1;
const SURF_LEVEL = 0.42;
/**
 * Low-pass corner, Hz, and how it moves.
 *
 * The Knudsen spectra fall at roughly 5–6 dB per octave above a few hundred hertz,
 * which is what a 12 dB/octave low-pass over white noise gives once the shoulder
 * is accounted for. It opens with wind speed because a stronger wind breaks the
 * surface at smaller scales and small breakers are what the high end is; it closes
 * with wave height because a big sea's energy is in slow, heavy water.
 */
const SURF_CUTOFF_BASE = 300;
const SURF_CUTOFF_PER_WIND = 16;
const SURF_CUTOFF_PER_HEIGHT = 34;
const SURF_CUTOFF_MIN = 150;
const SURF_CUTOFF_MAX = 950;
const SURF_Q = 0.6;
/** The second bed is quieter and darker: it is the sea further out. */
const SURF_FAR_LEVEL = 0.78;
const SURF_FAR_TILT = 0.72;
const SURF_PAN = 0.45;

// --- wind ------------------------------------------------------------------

/**
 * Wind noise climbs far faster than sea noise.
 *
 * Aerodynamic noise from an edge or a wire radiates as a power law in flow speed
 * with an exponent between 5 and 6 — that is in *power*, so somewhere near 3 in
 * amplitude. 1.6 is well under that on purpose: the real exponent applied across a
 * 0.5..25 m/s slider spans 15000:1, which is not a mix, it is a fault. What the
 * exponent has to do here is make the wind overtake the sea as it rises, and be
 * genuinely absent in a calm, and 1.6 does both.
 */
const WIND_EXPONENT = 1.6;
const WIND_REFERENCE_WIND = 16;
const WIND_MAX_DRIVE = 1.5;
// Halved. Wind is the layer that hisses: it is the widest band, the brightest,
// and the least like water. At 15 m/s — the default — the old level put it over
// the surf, so the resting scene sounded like an open window rather than a sea.
const WIND_LEVEL = 0.17;
const WIND_BAND_BASE = 260;
const WIND_BAND_PER_WIND = 34;
const WIND_BAND_MIN = 240;
const WIND_BAND_MAX = 1250;
const WIND_Q = 0.75;
const WIND_HIGHPASS = 180;

/**
 * Aeolian tone off the standing rigging: f = St * U / d.
 *
 * The Strouhal number of a circular cylinder is very nearly 0.2 over the whole
 * Reynolds range a 6 mm wire sees in wind — which is why a shroud's pitch is a
 * usable wind gauge, and why this is a tone rather than a texture. At 15 m/s it
 * sings at 500 Hz. It is gated on wind speed because the tone only establishes
 * itself once vortex shedding is periodic; below that the wire is just an
 * obstacle in turbulent air.
 */
const STROUHAL = 0.2;
const RIGGING_DIAMETER = 0.006;
const RIGGING_MIN_HZ = 120;
const RIGGING_MAX_HZ = 2600;
const RIGGING_Q = 7;
const RIGGING_LEVEL = 0.1;
const RIGGING_ONSET = 9;
const RIGGING_FULL = 20;

/** Gusting: depth, hold between new targets, and how fast it gets there. */
const GUST_DEPTH = 0.18;
const GUST_HOLD_MIN = 1.4;
const GUST_HOLD_SPAN = 2.6;
const GUST_TAU = 1.1;

// --- rain ------------------------------------------------------------------

/**
 * Level exponent for rain.
 *
 * Sub-linear, because rain noise is the sum of a very large number of independent
 * impacts and doubling the rate of a Poisson process adds 3 dB, not 6. Above about
 * half intensity what changes is mostly the *spectrum* — bigger drops, more low
 * drumming — which is why the band moves rather than the level continuing to climb.
 */
const RAIN_EXPONENT = 0.6;
const RAIN_LEVEL = 0.42;
const RAIN_CUTOFF_BASE = 3400;
const RAIN_CUTOFF_PER_RATE = 1100;
const RAIN_CUTOFF_MIN = 1600;
const RAIN_CUTOFF_MAX = 3600;
const RAIN_Q = 0.5;
/**
 * The rain layer's high-pass drops as the rain hardens.
 *
 * This is the drumming. Light rain on water is all splash and no body; heavy rain
 * is large drops hitting hard enough to make the deck and the sea itself respond
 * down at a couple of hundred hertz. Opening the high-pass downward is what
 * separates a shower from a downpour far more than the level does.
 */
const RAIN_HIGHPASS_BASE = 340;
const RAIN_HIGHPASS_PER_RATE = 200;
const RAIN_HIGHPASS_MIN = 110;
const RAIN_HISS_LEVEL = 0.09;
const RAIN_HISS_EXPONENT = 1.4;
const RAIN_HISS_HZ = 3400;
const RAIN_PAN = 0.25;

// --- hull ------------------------------------------------------------------

/**
 * Reference hull speed, m/s. The same 7 the wake's foam deposition saturates at,
 * because they are describing the same event: water being thrown by the hull.
 */
const HULL_REFERENCE_SPEED = 7;
/**
 * Wave-making resistance rises steeply with speed and the noise rides on it, so
 * this is super-linear like the wake's amplitude — but 1.5 rather than the wake's
 * quadratic, because what is heard is the entrained air and not the wave.
 */
const HULL_EXPONENT = 1.5;
const HULL_LEVEL = 0.3;
/** Faded in from rest, matching the wake's own 0.4..2.2 m/s amplitude fade. */
const HULL_ONSET = 0.4;
const HULL_FULL = 2.2;
const HULL_BAND_BASE = 200;
const HULL_BAND_PER_SPEED = 42;
const HULL_BAND_MIN = 200;
const HULL_BAND_MAX = 1100;
const HULL_Q = 0.55;
const HULL_HIGHPASS = 90;

// --- submersion ------------------------------------------------------------

/**
 * How much of the air mix survives at full submersion.
 *
 * Not all of it and not none. Sound crosses the air/water boundary badly — the
 * impedance mismatch reflects almost everything — but a swimmer under a breaking
 * sea is emphatically not in silence, because the breaking itself is happening in
 * the water. Ducking to a fifth and then closing the low-pass is what reads as
 * going under; ducking to zero reads as the audio failing.
 */
const AIR_DUCK = 0.8;
const AIR_OPEN_HZ = 18000;
const AIR_CLOSED_HZ = 320;
/**
 * Submersion exponent for the underwater bus.
 *
 * Above 1 would make the transition happen at the very end of the band and pop;
 * below 1 brings the rumble in as soon as the lens touches the water, which is
 * when it should arrive — the camera's submersion ramp is only 0.7 m wide and a
 * crest can cross it in a tenth of a second.
 */
const UNDER_EXPONENT = 0.7;
const RUMBLE_LEVEL = 0.55;
const RUMBLE_CALM = 0.45;
const RUMBLE_PER_WIND = 0.55;
const RUMBLE_HZ = 150;
const RUMBLE_Q = 1.1;
const RUMBLE_HIGHPASS = 25;

/**
 * Minnaert's constant, Hz·m: an air bubble in water rings at 3.26 / radius.
 *
 * That single relation is the whole reason synthesised bubbles are convincing —
 * the pitch is not a choice, it is the size of the bubble, and a spread of sizes
 * gives the scatter of pitches the ear expects. 0.8 to 5 mm is the range breaking
 * water and a hull entrain, which puts the blips between 650 Hz and 4 kHz.
 */
const MINNAERT_CONSTANT = 3.26;
const BUBBLE_MIN_RADIUS = 0.0008;
const BUBBLE_MAX_RADIUS = 0.005;
/**
 * How far the pitch rises over the blip.
 *
 * A bubble radiates while it is still detaching and shrinking, so its resonance
 * climbs — that upward chirp is the difference between a bubble and a beep.
 */
const BUBBLE_RISE = 1.6;
const BUBBLE_DECAY_MIN = 0.03;
const BUBBLE_DECAY_SPAN = 0.05;
const BUBBLE_LEVEL = 0.09;
const BUBBLE_CALM = 0.4;
const BUBBLE_PER_WIND = 0.6;
/**
 * Floor on the interval between scheduled bubbles, seconds.
 *
 * The scheduler drains its accumulated time in a `while` loop, and a Poisson
 * interval is `-ln(1 - u) / rate` — which is legitimately zero when `u` is. Without
 * a floor that is an infinite loop inside the frame update, i.e. a hung tab, for
 * one unlucky draw in a few billion. It also caps the worst case at 50 blips a
 * second whatever the rate is asked to be.
 */
const BUBBLE_MIN_GAP = 0.02;

// --- one-shots -------------------------------------------------------------

const SPLASH_LEVEL = 0.5;
const SPLASH_DECAY_MIN = 0.14;
const SPLASH_DECAY_SPAN = 0.4;
/**
 * The splash band sweeps down.
 *
 * An impact on water is bright at the instant of contact — the crown and the
 * spray — and then the cavity collapses, which is a larger, slower, lower event.
 * A splash on a fixed band sounds like a burst of static; the sweep is most of
 * what makes it read as water.
 */
const SPLASH_OPEN_HZ = 2600;
const SPLASH_CLOSE_HZ = 520;
const SPLASH_BODY_HZ = 190;
const SPLASH_BODY_LEVEL = 0.6;
/** Air entrained by the impact, surfacing over the following moment. */
const SPLASH_BUBBLES = 3;
const SPLASH_BUBBLE_WINDOW = 0.4;

const THUNDER_LEVEL = 0.85;
/**
 * e-folding distance for the crack, km.
 *
 * Atmospheric absorption is strongly frequency dependent — of order 2 dB/km at
 * 500 Hz and well over 20 dB/km at 4 kHz in ordinary conditions — so distance does
 * not merely quieten thunder, it removes the top of it. That is the entire
 * difference between a crack overhead and a roll on the horizon, and 1.4 km puts
 * the crossover where a viewer expects it.
 */
const THUNDER_ABSORPTION_KM = 1.4;
/** Geometric spreading plus scattering, as a simple inverse law. */
const THUNDER_SPREAD_PER_KM = 0.55;
const THUNDER_CRACK_SECONDS = 0.9;
const THUNDER_CRACK_HZ = 2600;
const THUNDER_CRACK_HIGHPASS = 260;
const THUNDER_ROLL_MIN_SECONDS = 1.5;
const THUNDER_ROLL_PER_KM = 0.5;
const THUNDER_ROLL_MAX_SECONDS = 8;
const THUNDER_ROLL_HZ = 110;
/**
 * Speed of sound, km/s. Not used by default — see `playThunder` — but it is the
 * number a caller wants when it triggers the sound off a lightning flash.
 */
export const SPEED_OF_SOUND_KM_S = 0.343;

/** Envelope curve resolution. 256 points over a roll is ~30 ms per point. */
const CURVE_POINTS = 256;

const UI_CLICK_HZ = 1250;
const UI_CLICK_FALL = 0.62;
const UI_CLICK_SECONDS = 0.045;
const UI_CLICK_LEVEL = 0.13;

/** Reward chime: root pitch, the arpeggio as ratios of it, and note spacing. */
const CHIME_ROOT_HZ = 784;
const CHIME_NOTES = [1, 1.25, 1.5] as const;
const CHIME_SPACING = 0.085;
const CHIME_LEVEL = 0.16;
/** [frequency ratio, level, decay seconds] per partial of each note. */
const CHIME_PARTIALS: readonly (readonly [number, number, number])[] = [
  [1, 1, 0.9],
  [2.76, 0.28, 0.35],
];

/** Anchor chain: the band the rattle sits in, its level and its link rate. */
const CHAIN_BAND_HZ = 2600;
const CHAIN_LEVEL = 0.35;
const CHAIN_LINKS_PER_SECOND = 14;

// --- output limiter ----------------------------------------------------------

/**
 * A brick wall after the master volume.
 *
 * The compressor above is a gentle 4:1 that lets thunder's leading edge through
 * on purpose, and the UI bus and the master volume sit after it — so with the
 * score, a close strike and a hull impact landing together the sum could still
 * cross full scale. This catches only that: 20:1, no knee, from -1 dBFS.
 *
 * The threshold is not lower because a `DynamicsCompressorNode` always applies
 * makeup gain — 0.6 of the gain it would take off a full-scale signal — so a
 * -4 dB threshold measured as +2.3 dB on *everything*, the whole mix louder in
 * exchange for a limiter. At -1 dB that is +0.6 dB. And since the node has a
 * finite attack, a soft clipper after it (`SOFT_CLIP_KNEE`) catches the few
 * samples of overshoot, so the output can never reach full scale.
 */
/**
 * Where the safety clipper starts bending, as a fraction of full scale. Linear
 * below it, a tanh shoulder above it that tops out at 0.93 (-0.6 dBFS) for any
 * input at all. In the measured worst case (volume 1, a 150 m strike, an
 * aground impact, a splash and the cargo sting on one sample) it bends about a
 * hundred samples; in every ordinary scenario measured, none.
 */
const LIMITER_THRESHOLD_DB = -1;
const LIMITER_RATIO = 20;
const LIMITER_ATTACK = 0.001;
const LIMITER_RELEASE = 0.12;
const SOFT_CLIP_KNEE = 0.7;

// --- the ship ------------------------------------------------------------------

/**
 * Distance at which the ship is heard at full level, metres; beyond it the
 * ship bus falls as 1/d. Twenty metres is about where the chase camera sits, so
 * at the helm everything is at its designed level and an orbit shot from across
 * the bay hears the hull as a murmur.
 */
const SHIP_REFERENCE_METRES = 20;
/** Below this the ship is inaudible and its event scheduler stops altogether. */
const SHIP_SILENT_PRESENCE = 0.04;
/** One-shot slots ship ambience leaves free; the same idea as `GULL_VOICE_RESERVE`. */
const SHIP_VOICE_RESERVE = 2;

/**
 * The bow wave, on top of the existing hull bed: a brighter, chuckling band
 * that surges as the bow drives down into the sea. The hull bed is the rush
 * along the side; this is the wave the stem is making.
 */
const BOW_LEVEL = 0.2;
const BOW_BAND_BASE = 620;
const BOW_BAND_PER_SPEED = 70;
const BOW_BAND_MAX = 1800;
const BOW_Q = 0.9;
/** Pitch rate, rad/s, at which a plunging bow doubles the wash. */
const BOW_PLUNGE_RATE = 0.12;

/** Continuous rustle under the canvas slaps while a sail is luffing. */
const LUFF_BED_LEVEL = 0.12;
const LUFF_BED_HZ = 520;

// --- ducking --------------------------------------------------------------------

/** How far a close strike pulls the score down, and the ambience beds. */
const THUNDER_MUSIC_DUCK = 0.55;
const THUNDER_BED_DUCK = 0.3;
/** Seconds held before recovering. */
const THUNDER_DUCK_HOLD = 1.6;

// --- crossing the surface ---------------------------------------------------------

/** Submersion at which going under or coming up is an event, with hysteresis. */
const PLUNGE_UNDER = 0.55;
const PLUNGE_SURFACE = 0.3;
const PLUNGE_COOLDOWN = 1.2;
const PLUNGE_BUBBLES = 7;

// --- interface ------------------------------------------------------------------

/** Minimum seconds between two UI cues of the same kind — a dragged slider is not a drumroll. */
const UI_CUE_GAP: Readonly<Record<UiCue, number>> = {
  hover: 0.06,
  press: 0.03,
  select: 0.04,
  slide: 0.035,
  toggleOn: 0.04,
  toggleOff: 0.04,
  open: 0.08,
  close: 0.08,
  info: 0.3,
  warn: 0.5,
};
const UI_TICK_LEVEL = 0.05;

/**
 * Period of the score's scheduling timer, ms. The look-ahead is 350 ms, so the
 * band keeps playing through a stalled render loop with room to spare.
 */
const MUSIC_PUMP_MS = 100;
const UI_HOVER_LEVEL = 0.022;

// --- gulls -----------------------------------------------------------------
//
// The hardest thing in this file to keep on the right side of the line between a
// bird and a synthesiser, for the same reason the bubble's rising chirp is the
// difference between a bubble and a beep: what identifies an animal is not its
// spectrum, it is what its pitch *does*.
//
// A herring gull's long call is a gesture, not a note. Each cry scoops up into a
// hard onset, holds while the bill is open, and glides back down as it closes;
// the cries run in a series that swells over the first two and then falls away,
// accelerating slightly as it goes. Every constant below describes one part of
// that gesture. A stack this rich played at a steady pitch is a car alarm.
//
// Structurally the whole call is **one oscillator**. The notes are stretches of
// automation on its frequency and on a gain that closes between them, not
// separate voices — which is what lets a seven-note call cost one slot out of a
// budget that has to keep room for a splash and a thunderclap.

/**
 * Partials in the cry's waveform, the slope they fall at, and the weight given
 * to the fundamental.
 *
 * A bird's syrinx is a pressure-driven valve, i.e. a relaxation oscillator: it
 * emits a pulse train, and a pulse train is a full harmonic stack falling at
 * about 6 dB per octave. 0.85 is a little shallower than that (a sawtooth is
 * exactly 1.0) because a gull's call is built to carry across open water, and
 * what carries is the 1–4 kHz band rather than the fundamental. The fundamental
 * is then pulled down below its own slope for the reason a spectrogram of the
 * species shows: the second and third partials are the loud ones, and a wave
 * with a dominant fundamental reads as a flute no matter what is done to it
 * afterwards.
 *
 * 18 partials on the highest fundamental here is 16 kHz, so the table is never
 * band-limited hard enough for the timbre to change across the pitch range.
 */
const GULL_PARTIALS = 18;
const GULL_PARTIAL_SLOPE = 0.85;
const GULL_FUNDAMENTAL_WEIGHT = 0.45;

/**
 * Fundamental of one bird, Hz, and the spread across individuals.
 *
 * Low for what is heard: the cry reads an octave up, at the second and third
 * partials — 1.2 to 2.7 kHz — which is where a herring gull's energy actually
 * sits and what the 800–1600 Hz a listener would name refers to. Drawn once per
 * call and held for the whole series, because a bird does not change size
 * between notes; the variation between calls is the flock having members.
 */
const GULL_F0_MIN = 620;
const GULL_F0_SPAN = 280;

/**
 * The contour of a single note, as fractions of its own length.
 *
 * `SCOOP` is where the note enters from — about a fourth below pitch. A gull
 * does not begin a note *on* the note; the same air pulse that makes the sound
 * has to spin the syrinx up, and starting on pitch is the single most synthetic
 * thing this could do. `PEAK` overshoots slightly, `FALL` is the glide the ear
 * would imitate if asked to do a gull, and `WAVER` is the small unsteadiness in
 * the held body — real notes are never flat, and a flat one at this harmonic
 * density is unmistakably an oscillator.
 *
 * `HOLD` is before `GLIDE` on purpose: the level starts falling while the pitch
 * is still up, so the note fades *into* its glide rather than doing both at once.
 */
const GULL_NOTE_MIN = 0.17;
const GULL_NOTE_SPAN = 0.11;
const GULL_NOTE_SCOOP = 0.72;
const GULL_NOTE_PEAK = 1.05;
const GULL_NOTE_FALL = 0.68;
const GULL_NOTE_RISE = 0.2;
const GULL_NOTE_HOLD = 0.5;
const GULL_NOTE_GLIDE = 0.6;
const GULL_NOTE_WAVER = 0.045;
/** Attack, seconds. Fast enough to read as a shout, slow enough not to click. */
const GULL_NOTE_ATTACK = 0.009;

/**
 * The series.
 *
 * A third of calls are a single drawn-out mew rather than a long call, because a
 * flock that only ever produces its full display call is a loop. The rest run
 * three to seven notes that accelerate (`ACCEL`), shorten (`SHRINK`) and drop in
 * pitch (`PITCH_FALL`) as the bird runs out of the breath it started with, under
 * an amplitude arc that swells over the first two notes and then fades. That arc
 * is most of what makes a series read as one utterance instead of a repeat.
 */
const GULL_SINGLE_CHANCE = 0.34;
const GULL_MEW_STRETCH = 1.45;
const GULL_NOTES_MIN = 3;
const GULL_NOTES_SPAN = 5;
const GULL_NOTE_GAP_MIN = 0.11;
const GULL_NOTE_GAP_SPAN = 0.08;
const GULL_NOTE_GAP_FLOOR = 0.06;
const GULL_NOTE_ACCEL = 0.9;
const GULL_NOTE_SHRINK = 0.95;
const GULL_PITCH_FALL = 0.972;
const GULL_SWELL_NOTES = 2;
const GULL_SERIES_FADE = 0.86;
/** Grace after the last note before the oscillator is stopped, seconds. */
const GULL_TAIL = 0.06;

/**
 * The formant, Hz, and how far it closes.
 *
 * The resonator is the bird's throat and its open bill, and the gape is not
 * fixed — a gull throws its head back and opens wide on the loud part of a note,
 * then closes as it glides down. So the emphasis sweeps with the note and lands
 * back where it started, which is why this is a peaking filter driven per note
 * rather than a fixed voicing. A formant that stays put while the pitch slides
 * under it is the sound of a filter, not of a throat.
 *
 * 1950 Hz with a moderate Q sits on the second and third partials of this pitch
 * range: +9 dB there is a strident cry, and the same +9 dB two octaves up would
 * be a whistle.
 */
const GULL_FORMANT_HZ = 1950;
const GULL_FORMANT_CLOSED = 0.62;
const GULL_FORMANT_Q = 1.5;
const GULL_FORMANT_DB = 9;

/**
 * Where the bird is.
 *
 * The flock flies 45–115 m circuits at 14–58 m up, so 30 to 240 m covers what is
 * plausibly audible; the distance is invented per cry rather than taken from any
 * particular bird, since nothing here knows where bird 7 is and a listener
 * cannot tell. It is drawn uniformly *over the annulus* — d = sqrt(near² + u
 * (far² − near²)) — because birds are spread over an area and there is far more
 * area far away. A uniform draw on the radius puts half the flock inside the
 * near half of it, which sounds like a colony on the rail.
 */
const GULL_NEAR_METRES = 30;
const GULL_FAR_METRES = 240;
const GULL_REFERENCE_METRES = 60;
const GULL_LEVEL = 0.15;

/**
 * Brightness against distance: corner = `BRIGHT * exp(-d / ABSORPTION)`.
 *
 * A distant gull is not a quiet near gull, and getting that difference right is
 * most of the sense of space this adds. Molecular absorption in ordinary sea air
 * is of order 1 dB per 100 m at 2 kHz and 3 dB per 100 m at 6 kHz, so across the
 * 200 m the flock spans it is a few dB of tilt on the top octave — real, but on
 * its own not enough to hear under a broadband sea. 170 m is deliberately
 * stronger than the molecular figure: turbulent scattering over water adds to
 * it, and the harmonics above 4 kHz are precisely the part that says *near*.
 * The floor keeps the formant band intact, so the farthest bird is dull and
 * still a bird rather than a hum.
 */
const GULL_BRIGHT_HZ = 9000;
const GULL_ABSORPTION_METRES = 170;
const GULL_DULL_HZ = 1900;

/**
 * Stereo placement, and how far it moves across one call.
 *
 * Not full width: a gull hard against one speaker is inside the listener's head,
 * and these are meant to be out over the water. The drift is a real cue and
 * nearly free — a bird cruising at 10 m/s covers 20 m during a long call, which
 * at 60 m is a visible change of bearing and at 240 m is almost none, so it
 * scales with the same inverse distance the level does.
 */
const GULL_PAN_SPREAD = 0.85;
const GULL_PAN_DRIFT = 0.14;

/**
 * Flock scaling. The rate goes as the square root of the count, not linearly.
 *
 * A tier that draws 64 birds instead of 26 does have more birds in the sky, so
 * the rate has to move — but linearly it would make Max nearly two and a half
 * times as talkative as High, and the constraint here is the listener rather
 * than the population. A gull every couple of seconds is a nesting colony; this
 * scene is a flock at sea. The cap is what a caller passing an implausible count
 * runs into.
 */
const GULL_REFERENCE_FLOCK = 26;
const GULL_MAX_FLOCK_SCALE = 2;

/**
 * Flurries: how many cries arrive together, and how far apart.
 *
 * Gulls answer each other, so cries come in loose clusters rather than singly —
 * but the count is drawn *squared*, so most flurries are one bird and an answer
 * is the exception. Birds answering every single time is a pattern, and a
 * pattern is exactly what this is trying not to be.
 */
const GULL_FLURRY_MAX = 3;
const GULL_ANSWER_MIN = 0.6;
const GULL_ANSWER_SPAN = 1.4;

/**
 * Floor on the gap between flurries, seconds, and the wait before the first one.
 *
 * The floor does the same job `BUBBLE_MIN_GAP` does — a Poisson interval is
 * legitimately zero for an unlucky draw — but it is seconds rather than
 * milliseconds because it is also doing the sound design: the thing that makes
 * sparse events read as wildlife is the length of the silences between them.
 *
 * The initial delay exists because a gull on the first frame after the viewer
 * unlocks the audio reads as a trigger rather than as a world. It is also what
 * the timer is held at while the gulls are gated off, so surfacing from a dive
 * does not fire one on the frame the lens clears the water.
 */
const GULL_FLURRY_FLOOR = 2.5;
const GULL_FIRST_DELAY = 6;

/**
 * One-shot slots kept clear of gulls.
 *
 * The voice cap is shared with splashes, thunder and bubbles, and those are
 * responses to something the viewer just did or just saw — a splash that is
 * dropped because three birds are mid-call is a missing sound with a visible
 * cause. Gulls are ambience and can afford to lose one, so they stop two slots
 * short of the cap and never take the last of them.
 */
const GULL_VOICE_RESERVE = 2;

/**
 * The gates.
 *
 * Under water there are no gulls: the surface reflects almost everything, and a
 * quarter submerged is already past the point where hearing one would be wrong.
 * In heavy rain and in a gale they sit it out — which is both true of the bird
 * and true of the mix, since by then the rain bed and the wind bed have taken
 * the band the cry lives in. 13 m/s is the top of a fresh breeze and 22 m/s is a
 * whole gale, by which point nothing is flying for pleasure.
 */
const GULL_SUBMERSION_SILENT = 0.25;
const GULL_RAIN_THIN = 0.25;
const GULL_RAIN_SILENT = 0.8;
const GULL_WIND_THIN = 13;
const GULL_WIND_SILENT = 22;

/**
 * Per-bed playback rates and buffer offsets.
 *
 * Every air layer reads the same white buffer, so without this they would be
 * filtered copies of one signal — which comb-filters where their bands overlap and
 * makes the whole mix repeat on one period. Mutually incommensurate rates and
 * offsets spread across the buffer cost nothing and remove both.
 */
const BED_RATES = {
  surfNear: 0.83,
  surfFar: 1.07,
  wind: 0.91,
  rigging: 1.19,
  rain: 1.13,
  rainHiss: 0.97,
  hull: 1.03,
  bowWash: 1.11,
  luff: 0.94,
  rumble: 0.89,
} as const;

type AudioContextConstructor = new (options?: AudioContextOptions) => AudioContext;

export class AudioSystem {
  /** False when the platform has no Web Audio. Every method is then a no-op. */
  readonly available: boolean;

  private ctx: BaseAudioContext | null = null;
  /** The same context when it is a realtime one; null when rendering offline. */
  private realtime: AudioContext | null = null;
  /** True when constructed on a caller's (offline) context. */
  private readonly offline: boolean = false;
  private disposed = false;

  // --- master chain ---------------------------------------------------------
  private master!: GainNode;
  private limiter!: DynamicsCompressorNode;
  private softClip!: WaveShaperNode;
  private compressor!: DynamicsCompressorNode;
  private mix!: GainNode;
  /** Everything above the waterline, before the muffle. */
  private airBus!: GainNode;
  private airMuffle!: BiquadFilterNode;
  private underBus!: GainNode;
  /**
   * UI feedback, routed past the muffle.
   *
   * A click is not in the world — it is the interface, and the interface does not
   * go underwater with the camera. Sending it through the air bus made pressing a
   * button while submerged sound like the panel was broken.
   */
  private uiBus!: GainNode;
  /** The ambience beds, so thunder can duck them without ducking itself. */
  private bedBus!: GainNode;
  /** Everything the ship makes: gained by distance, panned by bearing. */
  private shipBus!: GainNode;
  private shipPanner: StereoPannerNode | null = null;
  private shipGain!: ParamTarget;
  private shipPan: ParamTarget | null = null;

  private music!: MusicEngine;
  private shipSfx!: ShipSfx;
  private musicVolume = 0.5;
  private gullLocator: ((out: GullFix) => boolean) | null = null;
  private readonly gullFix: GullFix = { distance: GULL_REFERENCE_METRES, pan: 0 };
  private readonly shipMotion: ShipMotion = {
    heel: 0, pitch: 0, sailFill: 0, sailLuff: 0, windSpeed: 0, presence: 0,
  };
  private readonly uiLast: Record<UiCue, number> = {
    hover: -1, press: -1, select: -1, slide: -1, toggleOn: -1, toggleOff: -1,
    open: -1, close: -1, info: -1, warn: -1,
  };
  private wasUnder = false;
  private plungeCooldown = 0;
  /** The score's scheduling timer; see `MusicEngine.pump`. */
  private pumpTimer: ReturnType<typeof setInterval> | null = null;

  private airGain!: ParamTarget;
  private muffle!: ParamTarget;
  private underGain!: ParamTarget;

  // --- layers ---------------------------------------------------------------
  private surfNear!: NoiseBed;
  private surfFar!: NoiseBed;
  private windBed!: NoiseBed;
  private rigging!: NoiseBed;
  private rainBed!: NoiseBed;
  private rainHiss!: NoiseBed;
  private hull!: NoiseBed;
  private bowWash!: NoiseBed;
  private luffBed!: NoiseBed;
  private rumble!: NoiseBed;
  private readonly beds: NoiseBed[] = [];
  private readonly targets: ParamTarget[] = [];

  private white!: AudioBuffer;
  /**
   * The gull's waveform, built once and shared by every cry.
   *
   * A `PeriodicWave` is immutable and stateless — it is a wavetable, not a node —
   * so one instance serves every voice and a cry costs no allocation for its
   * timbre. Building it per call would be an allocation, and a large one, on a
   * path that can fire three times in a second.
   */
  private gullWave!: PeriodicWave;

  // --- state ----------------------------------------------------------------
  private volume = 0.8;
  private muted = false;
  private enabled = true;
  private gestured = false;
  private tier: QualityTier = 'high';
  private settings: AudioQualitySettings = AUDIO_QUALITY_TIERS.high;
  private suspendAt: number | null = null;

  private clock = 0;
  private gust = 1;
  private gustTarget = 1;
  private gustTimer = 0;
  private bubbleTimer = 0;
  private gullTimer = GULL_FIRST_DELAY;

  private bubbleRandom = mulberry32(AUDIO_SEEDS.bubbles);
  private gustRandom = mulberry32(AUDIO_SEEDS.gusts);
  private oneShotRandom = mulberry32(AUDIO_SEEDS.oneShots);
  private gullRandom = mulberry32(AUDIO_SEEDS.gulls);

  private crackCurve!: Float32Array;
  private rollCurve!: Float32Array;

  /** Live one-shot sources, so a reset or a dispose can silence them. */
  private readonly voices: AudioScheduledSourceNode[] = [];

  constructor(options: AudioSystemOptions = {}) {
    this.volume = clamp01(finite(options.volume, this.volume));
    this.musicVolume = clamp01(finite(options.musicVolume, this.musicVolume));
    this.muted = options.muted ?? false;
    this.tier = options.quality ?? 'high';
    this.settings = AUDIO_QUALITY_TIERS[this.tier] ?? AUDIO_QUALITY_TIERS.high;

    if (options.context !== undefined) {
      this.offline = true;
      this.ctx = options.context;
      this.gestured = true;
      this.build(this.ctx);
      this.available = true;
      return;
    }

    const Ctor = resolveAudioContext();
    if (Ctor === null) {
      this.available = false;
      return;
    }

    try {
      // 'interactive' rather than 'playback'. A larger buffer would be cheaper and
      // more robust against a heavy frame, but the UI click has to land with the
      // press — 'playback' can put a couple of hundred milliseconds between them,
      // which does not read as latency, it reads as the button not having worked.
      const realtime = new Ctor({ latencyHint: 'interactive' });
      this.ctx = realtime;
      this.realtime = realtime;
      this.build(realtime);
      this.available = true;
    } catch {
      // Construction can fail outright on a locked-down or headless platform.
      // Audio going missing is acceptable; taking the scene with it is not.
      this.ctx = null;
      this.realtime = null;
      this.available = false;
      return;
    }

    // Some platforms hand back a context that is already running — a page that has
    // been interacted with, or one with an autoplay exemption. Left alone, that
    // starts the sea before the application has said it may.
    if (this.realtime.state === 'running') void this.realtime.suspend().catch(() => {});
  }

  /** `'unavailable'` when there is no Web Audio at all. */
  get contextState(): 'unavailable' | AudioContextState {
    return this.ctx === null ? 'unavailable' : this.ctx.state;
  }

  /** True while audio exists but is still waiting for its unlocking gesture. */
  get needsGesture(): boolean {
    return this.ctx !== null && !this.gestured;
  }

  /**
   * Unlocks audio. Must be called from inside a user gesture handler.
   *
   * Safe to call repeatedly and safe to call when there is no audio; resolves
   * `true` only once the context is genuinely running.
   */
  async resume(): Promise<boolean> {
    const ctx = this.ctx;
    if (ctx === null || this.disposed) return false;

    this.gestured = true;
    this.suspendAt = null;
    if (!this.enabled) return false;

    try {
      if (this.realtime !== null) await this.realtime.resume();
    } catch {
      return false;
    }
    if (this.disposed || this.ctx === null) return false;

    // The model has been running since construction, so its targets are current;
    // pushing them in one go stops the scene fading in from a stale state.
    const now = ctx.currentTime;
    this.startBeds(now);
    this.flushAll(now);
    this.applyMasterGain(this.offline ? 0 : RESUME_RAMP);
    if (!this.offline && this.pumpTimer === null && typeof setInterval === 'function') {
      this.pumpTimer = setInterval(() => {
        const c = this.ctx;
        // dt 0: the model does not advance off the timer, but the band can go
        // live and keep scheduling while the render loop is stalled.
        if (c !== null && this.enabled && !this.muted && this.running()) this.music.update(0, c.currentTime, true);
      }, MUSIC_PUMP_MS);
    }
    return this.running();
  }

  /**
   * Convenience: resumes on the first gesture on `target`.
   *
   * Returns a disposer. Entirely optional — a caller with its own gesture plumbing
   * should just call `resume()` from it.
   */
  resumeOnGesture(target?: EventTarget): () => void {
    // Resolved in the body, not as a default argument: a default is evaluated on
    // every call, so `= window` would throw on a platform without a DOM before
    // the `available` guard below could stop it.
    const listener = target ?? (typeof window === 'undefined' ? null : window);
    if (this.ctx === null || listener === null) return () => {};

    const events = ['pointerdown', 'keydown', 'touchstart'] as const;
    const handler = (): void => {
      void this.resume();
      for (const name of events) listener.removeEventListener(name, handler);
    };
    for (const name of events) listener.addEventListener(name, handler, { passive: true });
    return () => {
      for (const name of events) listener.removeEventListener(name, handler);
    };
  }

  /**
   * Drives every layer from the current scene state.
   *
   * Allocates nothing. `params` is read and not retained, so a caller that hoists
   * one object and mutates its fields allocates nothing either.
   */
  update(dt: number, params: Readonly<AudioSceneParams>): void {
    const ctx = this.ctx;
    if (ctx === null || this.disposed) return;

    const step = clampNumber(finite(dt, 0), 0, 0.1);
    this.clock = (this.clock + step) % CLOCK_WRAP;

    const now = ctx.currentTime;

    // Deferred work first: it has to run even when nothing else does, or a
    // disabled system keeps its sources rendering forever.
    if (this.suspendAt !== null && now >= this.suspendAt) {
      this.suspendAt = null;
      void this.realtime?.suspend().catch(() => {});
    }
    for (let i = 0; i < this.beds.length; i++) this.beds[i].tick(now);

    const live = this.enabled && !this.muted && this.running();

    const wind = clampNumber(finite(params.windSpeed, 0), 0, 40);
    const waveHeight = clampNumber(finite(params.waveHeight, 0), 0, 20);
    const wavelength = clampNumber(finite(params.peakWavelength, 47), 4, 400);
    const rain = clamp01(finite(params.rain, 0));
    const submersion = clamp01(finite(params.submersion, 0));
    const hullSpeed = clampNumber(finite(params.hullSpeed, 0), 0, 40);
    const birdCount = clampNumber(finite(params.birdCount, 0), 0, 1024);

    // --- gusting ------------------------------------------------------------
    this.gustTimer -= step;
    if (this.gustTimer <= 0) {
      this.gustTimer = GUST_HOLD_MIN + GUST_HOLD_SPAN * this.gustRandom();
      this.gustTarget = 1 + GUST_DEPTH * (this.gustRandom() * 2 - 1);
    }
    // Framerate-independent approach, so a hitch does not jump the gust.
    this.gust += (this.gustTarget - this.gust) * (1 - Math.exp(-step / GUST_TAU));

    // --- swell --------------------------------------------------------------
    // Deep-water dispersion, the same relation the wake's crest spacing comes
    // from: a wave of length L has period sqrt(2*pi*L/g). The default 47 m sea
    // breathes at 5.5 s.
    const swellPeriod = Math.sqrt((TAU * wavelength) / GRAVITY);
    const groupPeriod = swellPeriod * GROUP_WAVES;
    const swellDepth =
      SWELL_DEPTH_CALM + SWELL_DEPTH_SPAN * clamp01(waveHeight / SWELL_FULL_HEIGHT);
    const swellNear = this.swellAt(swellDepth, swellPeriod, groupPeriod, 0);
    const swellFar = this.swellAt(swellDepth, swellPeriod, groupPeriod, SURF_FAR_PHASE);

    // --- surf ---------------------------------------------------------------
    const windDrive = Math.min(wind / SURF_REFERENCE_WIND, SURF_MAX_DRIVE);
    const seaLevel = SURF_LEVEL * (SURF_CALM + windDrive);
    const surfCutoff = clampNumber(
      SURF_CUTOFF_BASE + SURF_CUTOFF_PER_WIND * wind - SURF_CUTOFF_PER_HEIGHT * waveHeight,
      SURF_CUTOFF_MIN,
      SURF_CUTOFF_MAX,
    );
    this.surfNear.setLevel(seaLevel * swellNear, now, live);
    this.surfNear.setBand(surfCutoff, SURF_Q, now, live);
    this.surfFar.setLevel(seaLevel * SURF_FAR_LEVEL * swellFar, now, live);
    this.surfFar.setBand(surfCutoff * SURF_FAR_TILT, SURF_Q, now, live);

    // --- wind ---------------------------------------------------------------
    const windLoud = Math.pow(Math.min(wind / WIND_REFERENCE_WIND, WIND_MAX_DRIVE), WIND_EXPONENT);
    this.windBed.setLevel(WIND_LEVEL * windLoud * this.gust, now, live);
    this.windBed.setBand(
      clampNumber(WIND_BAND_BASE + WIND_BAND_PER_WIND * wind, WIND_BAND_MIN, WIND_BAND_MAX),
      WIND_Q,
      now,
      live,
    );

    const singing = smoothstepNumber(wind, RIGGING_ONSET, RIGGING_FULL);
    this.rigging.setLevel(RIGGING_LEVEL * windLoud * singing * this.gust, now, live);
    this.rigging.setBand(
      clampNumber((STROUHAL * wind) / RIGGING_DIAMETER, RIGGING_MIN_HZ, RIGGING_MAX_HZ),
      RIGGING_Q,
      now,
      live,
    );

    // --- rain ---------------------------------------------------------------
    // Rain gusts with the wind but less than the wind does: a squall arrives in
    // sheets, and the sheets are the same air movement.
    const rainGust = 0.85 + 0.15 * this.gust;
    this.rainBed.setLevel(RAIN_LEVEL * Math.pow(rain, RAIN_EXPONENT) * rainGust, now, live);
    this.rainBed.setBand(
      clampNumber(
        RAIN_CUTOFF_BASE - RAIN_CUTOFF_PER_RATE * rain,
        RAIN_CUTOFF_MIN,
        RAIN_CUTOFF_MAX,
      ),
      RAIN_Q,
      now,
      live,
    );
    this.rainBed.setHighpass(
      Math.max(RAIN_HIGHPASS_MIN, RAIN_HIGHPASS_BASE - RAIN_HIGHPASS_PER_RATE * rain),
      now,
      live,
    );
    this.rainHiss.setLevel(RAIN_HISS_LEVEL * Math.pow(rain, RAIN_HISS_EXPONENT), now, live);

    // --- hull ---------------------------------------------------------------
    const speedDrive = Math.min(hullSpeed / HULL_REFERENCE_SPEED, 1.3);
    this.hull.setLevel(
      HULL_LEVEL *
        Math.pow(speedDrive, HULL_EXPONENT) *
        smoothstepNumber(hullSpeed, HULL_ONSET, HULL_FULL),
      now,
      live,
    );
    this.hull.setBand(
      clampNumber(
        HULL_BAND_BASE + HULL_BAND_PER_SPEED * hullSpeed,
        HULL_BAND_MIN,
        HULL_BAND_MAX,
      ),
      HULL_Q,
      now,
      live,
    );

    // --- the ship -----------------------------------------------------------
    const shipDistance =
      params.shipDistance === undefined || Number.isNaN(params.shipDistance)
        ? Number.POSITIVE_INFINITY
        : Math.max(0, params.shipDistance);
    let presence =
      shipDistance === Number.POSITIVE_INFINITY
        ? 0
        : Math.min(1, SHIP_REFERENCE_METRES / Math.max(shipDistance, SHIP_REFERENCE_METRES));
    if (presence < SHIP_SILENT_PRESENCE) presence = 0;
    this.shipGain.set(presence, now, live);
    this.shipPan?.set(clampNumber(finite(params.shipPan, 0), -1, 1) * 0.7, now, live);

    const luff = clamp01(finite(params.sailLuff, 0));
    const motion = this.shipMotion;
    motion.heel = clampNumber(finite(params.shipHeel, 0), -1.5, 1.5);
    motion.pitch = clampNumber(finite(params.shipPitch, 0), -1.5, 1.5);
    motion.sailFill = clamp01(finite(params.sailFill, 0));
    motion.sailLuff = luff;
    motion.windSpeed = wind;
    motion.presence = presence;
    this.shipSfx.update(step, now, live, motion);

    // The bow wave surges as the stem drives down into the sea.
    const plunge = clampNumber(-this.shipSfx.pitchingRate / BOW_PLUNGE_RATE, 0, 1.5);
    this.bowWash.setLevel(
      BOW_LEVEL *
        Math.pow(speedDrive, 1.3) *
        smoothstepNumber(hullSpeed, 0.6, 2.5) *
        (0.6 + 0.6 * plunge),
      now,
      live,
    );
    this.bowWash.setBand(
      clampNumber(BOW_BAND_BASE + BOW_BAND_PER_SPEED * hullSpeed, BOW_BAND_BASE, BOW_BAND_MAX),
      BOW_Q,
      now,
      live,
    );
    this.luffBed.setLevel(LUFF_BED_LEVEL * luff * (0.5 + 0.5 * Math.min(1, wind / 14)), now, live);

    // --- submersion ---------------------------------------------------------
    this.plungeCooldown = Math.max(0, this.plungeCooldown - step);
    if (!this.wasUnder && submersion >= PLUNGE_UNDER) {
      this.wasUnder = true;
      if (live && this.plungeCooldown === 0) this.plunge(true);
    } else if (this.wasUnder && submersion <= PLUNGE_SURFACE) {
      this.wasUnder = false;
      if (live && this.plungeCooldown === 0) this.plunge(false);
    }
    this.airGain.set(1 - AIR_DUCK * submersion, now, live);
    // Geometric, not linear. Cutoff is heard in octaves: a linear sweep from
    // 18 kHz spends nine tenths of its travel in a range the ear still calls
    // bright and then collapses over the last stretch, so the water arrives all
    // at once instead of closing over the lens.
    this.muffle.set(AIR_OPEN_HZ * Math.pow(AIR_CLOSED_HZ / AIR_OPEN_HZ, submersion), now, live);
    this.underGain.set(Math.pow(submersion, UNDER_EXPONENT), now, live);
    this.rumble.setLevel(
      RUMBLE_LEVEL * (RUMBLE_CALM + RUMBLE_PER_WIND * windDrive) * swellNear,
      now,
      live,
    );

    // --- bubbles ------------------------------------------------------------
    const bubbleRate =
      this.settings.bubbleRate * submersion * (BUBBLE_CALM + BUBBLE_PER_WIND * windDrive);
    if (live && bubbleRate > 0.001) {
      this.bubbleTimer -= step;
      // Spread across the frame rather than all landing on its first sample. Two
      // bubbles at the same instant are one louder bubble, not two.
      let offset = 0;
      while (this.bubbleTimer <= 0) {
        this.spawnBubble(
          now + offset,
          this.underBus,
          BUBBLE_LEVEL * (0.55 + 0.45 * this.bubbleRandom()),
        );
        offset += BUBBLE_MIN_GAP;
        // Poisson inter-arrival, floored: see BUBBLE_MIN_GAP.
        const u = this.bubbleRandom();
        this.bubbleTimer += Math.max(BUBBLE_MIN_GAP, -Math.log(1 - u) / bubbleRate);
      }
    } else if (this.bubbleTimer < 0) {
      this.bubbleTimer = 0;
    }

    // --- gulls --------------------------------------------------------------
    // Everything about this is a product of gates rather than a branch, so a
    // rising wind or a closing squall thins the flock out continuously instead
    // of switching it off at a threshold — the same reason `submersion` is not a
    // boolean anywhere else in this file.
    const gullRate =
      this.settings.gullRate *
      Math.min(GULL_MAX_FLOCK_SCALE, Math.sqrt(birdCount / GULL_REFERENCE_FLOCK)) *
      (1 - smoothstepNumber(submersion, 0, GULL_SUBMERSION_SILENT)) *
      (1 - smoothstepNumber(rain, GULL_RAIN_THIN, GULL_RAIN_SILENT)) *
      (1 - smoothstepNumber(wind, GULL_WIND_THIN, GULL_WIND_SILENT));

    if (live && gullRate > 1e-4) {
      this.gullTimer -= step;
      // An `if` rather than the bubbles' `while`: the interval is floored at
      // seconds, so a frame can never owe more than one flurry, and a loop here
      // would be a loop that provably runs once.
      if (this.gullTimer <= 0) {
        this.scheduleGullFlurry(now);
        const u = this.gullRandom();
        this.gullTimer = GULL_FLURRY_FLOOR + -Math.log(1 - u) / gullRate;
      }
    } else if (this.gullTimer < GULL_FLURRY_FLOOR) {
      // Held at the floor rather than at zero while the gate is shut. Parked at
      // zero, coming up from a dive or out of a squall would fire a cry on the
      // very frame the gate opened, every time — which is a trigger, not a bird.
      this.gullTimer = GULL_FLURRY_FLOOR;
    }

    // --- score --------------------------------------------------------------
    this.music.update(step, now, live);
  }

  setVolume(value: number): void {
    this.volume = clamp01(finite(value, this.volume));
    this.applyMasterGain(VOLUME_RAMP);
  }

  getVolume(): number {
    return this.volume;
  }

  /**
   * Hard silence, without changing anything else.
   *
   * While muted no parameter is written at all — the model keeps running so
   * `resetClock` still means something, but the audio thread is left alone. That
   * makes a muted capture run cost nothing and, more importantly, removes any way
   * for audio scheduling to show up in a frame-time measurement.
   */
  setMuted(muted: boolean): void {
    if (this.muted === muted) return;
    this.muted = muted;
    if (muted && this.ctx !== null) this.music.pause();
    this.applyMasterGain(MUTE_RAMP);
    const ctx = this.ctx;
    // Unmuting resumes writing, and the graph has been frozen wherever it was
    // when the mute landed.
    if (!muted && ctx !== null && this.running()) this.flushAll(ctx.currentTime);
  }

  isMuted(): boolean {
    return this.muted;
  }

  /**
   * Turns the whole system off, and suspends the context once the fade is done —
   * so audio the viewer has switched off stops costing a thread, rather than
   * rendering eight noise beds into a gain of zero.
   */
  setEnabled(enabled: boolean): void {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    if (!enabled && this.ctx !== null) this.music.pause();
    this.applyMasterGain(ENABLE_RAMP);

    const ctx = this.ctx;
    if (ctx === null) return;

    if (!enabled) {
      // Only meaningful while time is moving; an already-suspended context would
      // never reach the deadline.
      this.suspendAt = this.running() && !this.offline ? ctx.currentTime + SUSPEND_DELAY : null;
      return;
    }

    this.suspendAt = null;
    if (!this.gestured) return;
    void (this.realtime?.resume() ?? Promise.resolve())
      .then(() => {
        if (this.disposed || this.ctx === null) return;
        const now = this.ctx.currentTime;
        this.startBeds(now);
        this.flushAll(now);
        this.applyMasterGain(ENABLE_RAMP);
      })
      .catch(() => {});
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Applies a quality tier. Never rebuilds the graph — layers are connected and
   * disconnected, so a tier change costs no allocation and makes no sound.
   */
  setQuality(tier: QualityTier): void {
    const settings = AUDIO_QUALITY_TIERS[tier];
    if (settings === undefined || this.tier === tier) return;
    this.tier = tier;
    this.settings = settings;
    this.applyQuality();
  }

  getQuality(): QualityTier {
    return this.tier;
  }

  /**
   * A splash, synthesised.
   *
   * @param intensity 0..1. Scales the level, how far the band sweeps, how long the
   *                  cavity takes to collapse, and how much air is entrained.
   */
  playSplash(intensity = 1): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    const amount = clamp01(finite(intensity, 0));
    if (amount < 0.02) return;
    if (!this.acquireVoice()) return;

    const random = this.oneShotRandom;
    const now = ctx.currentTime;
    const decay = SPLASH_DECAY_MIN + SPLASH_DECAY_SPAN * amount;
    const level = SPLASH_LEVEL * Math.pow(amount, 0.8);

    const source = ctx.createBufferSource();
    source.buffer = this.white;
    source.loop = true;
    source.playbackRate.value = 0.85 + 0.3 * random();

    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.Q.value = 0.7;
    band.frequency.setValueAtTime(SPLASH_OPEN_HZ, now);
    band.frequency.exponentialRampToValueAtTime(SPLASH_CLOSE_HZ, now + decay);

    const crown = ctx.createGain();
    envelope(crown.gain, now, 0.003, level, decay);

    // The body: the cavity closing, lower and slower than the crown that made it.
    const body = ctx.createBiquadFilter();
    body.type = 'lowpass';
    body.frequency.value = SPLASH_BODY_HZ;
    body.Q.value = 0.9;

    const thump = ctx.createGain();
    envelope(thump.gain, now, 0.01, level * SPLASH_BODY_LEVEL * amount, decay * 1.6);

    source.connect(band);
    band.connect(crown);
    crown.connect(this.airBus);
    source.connect(body);
    body.connect(thump);
    thump.connect(this.airBus);

    const tail = decay * 1.6 + 0.05;
    source.start(now, random() * this.white.duration);
    source.stop(now + tail);
    this.trackVoice(source, band, crown, body, thump);

    // Entrained air, surfacing behind the impact.
    const count = Math.round(1 + SPLASH_BUBBLES * amount);
    for (let i = 0; i < count; i++) {
      this.spawnBubble(
        now + 0.02 + random() * SPLASH_BUBBLE_WINDOW * (0.5 + amount),
        this.airBus,
        BUBBLE_LEVEL * (0.4 + 0.6 * amount),
      );
    }
  }

  /**
   * Thunder, synthesised as a crack and a roll whose balance is the distance.
   *
   * @param distanceKm   Distance to the strike, km. Distance does not merely
   *                     quieten it — see `THUNDER_ABSORPTION_KM`.
   * @param delaySeconds Delay before it is heard. Left at 0 by default so the
   *                     caller stays in control of its own timing; a caller
   *                     triggering off a lightning flash wants
   *                     `distanceKm / SPEED_OF_SOUND_KM_S`, which is ~2.9 s per km.
   */
  playThunder(distanceKm = 3, delaySeconds = 0): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    if (!this.acquireVoice()) return;

    const distance = clampNumber(finite(distanceKm, 3), 0.05, 40);
    const now = ctx.currentTime + Math.max(0, finite(delaySeconds, 0));

    // Everything above a kilohertz is absorbed out of a distant strike, so the
    // crack fades exponentially while the roll only spreads.
    const crackWeight = Math.exp(-distance / THUNDER_ABSORPTION_KM);
    const spread = 1 / (1 + distance * THUNDER_SPREAD_PER_KM);
    const rollSeconds = Math.min(
      THUNDER_ROLL_MAX_SECONDS,
      THUNDER_ROLL_MIN_SECONDS + THUNDER_ROLL_PER_KM * distance,
    );

    const source = ctx.createBufferSource();
    source.buffer = this.white;
    source.loop = true;
    // Under 1, so the noise is darker and the loop outlasts the longest roll.
    source.playbackRate.value = 0.7;

    const crackHigh = ctx.createBiquadFilter();
    crackHigh.type = 'highpass';
    crackHigh.frequency.value = THUNDER_CRACK_HIGHPASS;
    crackHigh.Q.value = 0.7;

    const crackLow = ctx.createBiquadFilter();
    crackLow.type = 'lowpass';
    crackLow.Q.value = 0.7;
    crackLow.frequency.value = Math.max(400, THUNDER_CRACK_HZ * crackWeight);

    // The curve carries the shape and the gain after it carries the level, since
    // `setValueCurveAtTime` writes absolute values and cannot be scaled.
    //
    // Nothing else is scheduled on these two parameters, deliberately: a curve
    // throws `NotSupportedError` if any other automation event falls inside its
    // window, and that includes the `setValueAtTime` that assigning `.value`
    // implies. What holds the path silent before `now` is the source not having
    // started, not the gain being zeroed.
    const crackEnv = ctx.createGain();
    crackEnv.gain.setValueCurveAtTime(this.crackCurve, now, THUNDER_CRACK_SECONDS);

    const crackLevel = ctx.createGain();
    crackLevel.gain.value = THUNDER_LEVEL * spread * crackWeight;

    const rollLow = ctx.createBiquadFilter();
    rollLow.type = 'lowpass';
    rollLow.frequency.value = THUNDER_ROLL_HZ;
    rollLow.Q.value = 1.2;

    const rollEnv = ctx.createGain();
    rollEnv.gain.setValueCurveAtTime(this.rollCurve, now, rollSeconds);

    const rollLevel = ctx.createGain();
    // A near strike is mostly crack; a far one is only roll.
    rollLevel.gain.value = THUNDER_LEVEL * spread * (0.35 + 0.65 * (1 - crackWeight));

    source.connect(crackHigh);
    crackHigh.connect(crackLow);
    crackLow.connect(crackEnv);
    crackEnv.connect(crackLevel);
    crackLevel.connect(this.airBus);

    source.connect(rollLow);
    rollLow.connect(rollEnv);
    rollEnv.connect(rollLevel);
    rollLevel.connect(this.airBus);

    // Sidechain, by hand: the score and the sea beds make room for the strike,
    // in proportion to how loud it arrives, at the moment it arrives.
    const loudness = clampNumber(spread * 1.6, 0.25, 1);
    this.music.duck(THUNDER_MUSIC_DUCK * loudness, THUNDER_DUCK_HOLD + rollSeconds * 0.3, now);
    const bed = this.bedBus.gain;
    bed.cancelScheduledValues(now);
    bed.setTargetAtTime(1 - THUNDER_BED_DUCK * loudness, now, 0.04);
    bed.setTargetAtTime(1, now + THUNDER_DUCK_HOLD, 0.6);

    source.start(now, this.oneShotRandom() * this.white.duration);
    source.stop(now + Math.max(THUNDER_CRACK_SECONDS, rollSeconds) + 0.1);
    this.trackVoice(
      source,
      crackHigh,
      crackLow,
      crackEnv,
      crackLevel,
      rollLow,
      rollEnv,
      rollLevel,
    );
  }

  /**
   * One gull's call, synthesised.
   *
   * Not needed to hear gulls — `update` schedules them from `birdCount` on its
   * own — but a caller that knows a bird just did something (broke off the mast,
   * dived on the wake) can place one itself, and it is the way to hear the voice
   * on demand without waiting on the scheduler.
   *
   * Refused outright on a tier whose `gullRate` is 0, so an explicit call cannot
   * put a bird in Low's empty sky.
   *
   * @param distanceMetres How far off the bird is, clamped to 30..240 m.
   *                       Distance does not merely quieten it — see
   *                       `GULL_ABSORPTION_METRES`.
   * @param pan            Placement, -1..1. 0 puts the bird dead ahead.
   */
  playGullCry(distanceMetres = GULL_REFERENCE_METRES, pan = 0): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay() || this.settings.gullRate <= 0) return;
    this.spawnGullCall(
      ctx.currentTime,
      clampNumber(
        finite(distanceMetres, GULL_REFERENCE_METRES),
        GULL_NEAR_METRES,
        GULL_FAR_METRES,
      ),
      clampNumber(finite(pan, 0), -1, 1),
    );
  }

  /** UI feedback. Deliberately not routed through the underwater muffle. */
  playUiClick(): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    if (!this.acquireVoice()) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    // A falling pitch, because a click with a fixed pitch is a beep. The drop is
    // what makes it read as something being struck rather than something sounding.
    osc.frequency.setValueAtTime(UI_CLICK_HZ, now);
    osc.frequency.exponentialRampToValueAtTime(UI_CLICK_HZ * UI_CLICK_FALL, now + UI_CLICK_SECONDS);

    const gain = ctx.createGain();
    envelope(gain.gain, now, 0.0015, UI_CLICK_LEVEL, UI_CLICK_SECONDS);

    osc.connect(gain);
    gain.connect(this.uiBus);
    osc.start(now);
    osc.stop(now + UI_CLICK_SECONDS + 0.01);
    this.trackVoice(osc, gain);
  }

  /**
   * A reward chime: a rising major arpeggio of struck-bell partials.
   *
   * Interface feedback rather than a sound in the world, so it rides the UI bus
   * past the underwater muffle like the click does. Three notes, each a
   * fundamental plus an inharmonic partial at 2.76x — the ratio that makes a
   * struck bar read as metal rather than as a sine beep.
   */
  playChime(): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    // The reward is also a moment in the score.
    this.music.playSting('cargo', ctx.currentTime);
    if (!this.acquireVoice()) return;

    const now = ctx.currentTime;
    const out = ctx.createGain();
    out.gain.value = CHIME_LEVEL;
    out.connect(this.uiBus);

    const nodes: AudioNode[] = [out];
    let first: OscillatorNode | null = null;
    let end = now;
    for (let i = 0; i < CHIME_NOTES.length; i++) {
      const start = now + i * CHIME_SPACING;
      for (const [ratio, level, decay] of CHIME_PARTIALS) {
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.value = CHIME_ROOT_HZ * CHIME_NOTES[i] * ratio;
        const gain = ctx.createGain();
        envelope(gain.gain, start, 0.004, level, decay);
        osc.connect(gain);
        gain.connect(out);
        osc.start(start);
        osc.stop(start + decay + 0.05);
        end = Math.max(end, start + decay + 0.05);
        if (first === null) first = osc;
        else nodes.push(osc);
        nodes.push(gain);
      }
    }
    // One voice for the whole chord: the first oscillator carries the cleanup,
    // and it is held open until the last partial has rung out.
    if (first !== null) {
      first.stop(end);
      this.trackVoice(first, ...nodes);
    }
  }

  // ------------------------------------------------------------------ score

  /**
   * Which arrangement the score plays. The band follows after a couple of
   * seconds (at once for going under or surfacing) and changes material at the
   * next barline, so this can be called every frame with whatever the game
   * state implies.
   *
   * @param energy Optional 0..1, as `setMusicEnergy`.
   */
  setMusicState(state: MusicState, energy?: number): void {
    if (this.ctx === null) return;
    this.music.setState(state);
    if (energy !== undefined) this.music.setEnergy(energy);
  }

  getMusicState(): MusicState | 'unavailable' {
    return this.ctx === null ? 'unavailable' : this.music.currentState;
  }

  /** 0..1: how hard she is being driven. Tempo, drums and density follow it. */
  setMusicEnergy(energy: number): void {
    if (this.ctx !== null) this.music.setEnergy(energy);
  }

  /** Music volume, 0..1, under the master volume. 0 stops the score scheduling anything. */
  setMusicVolume(value: number): void {
    this.musicVolume = clamp01(finite(value, this.musicVolume));
    if (this.ctx !== null) this.music.setVolume(this.musicVolume);
  }

  getMusicVolume(): number {
    return this.musicVolume;
  }

  /** A musical flourish over the score, which ducks under it. `playChime` plays the cargo one. */
  playSting(kind: MusicSting = 'cargo'): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    this.music.playSting(kind, ctx.currentTime);
  }

  /** Sequencer position and voice use, for tests. */
  get musicDebug(): {
    state: MusicState;
    voices: number;
    phrase: number;
    bar: number;
    energy: number;
  } | null {
    if (this.ctx === null) return null;
    const p = this.music.position;
    return {
      state: this.music.currentState,
      voices: this.music.voicesInUse,
      phrase: p.phrase,
      bar: p.bar,
      energy: p.energy,
    };
  }

  // ------------------------------------------------------------------ ship hooks

  /**
   * The hull striking something: a slam into a wave face, a rock, the shore.
   * A deep thud through the frame and the timbers answering it.
   *
   * @param strength 0..1. ~0.3 is a hard slam; 1 is running aground at speed.
   * @param grind    Add the drawn-out scrape of a keel dragging over the bottom.
   */
  hullImpact(strength: number, grind = false): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    this.shipSfx.hullImpact(clamp01(finite(strength, 0)), grind, ctx.currentTime);
  }

  /**
   * Water slapping the hull.
   *
   * @param strength 0..1, from the relative speed of water and planking. A
   *                 slow lap is ~0.15, a crest smacking the side ~0.6.
   * @param pan      -1..1 within the ship's own image (port/starboard as the
   *                 listener sees them). The ship bus adds the ship's bearing.
   */
  waveSlap(strength: number, pan = 0): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    this.shipSfx.waveSlap(
      clamp01(finite(strength, 0)),
      clampNumber(finite(pan, 0), -1, 1),
      ctx.currentTime,
    );
  }

  /**
   * Where the gulls are. Called when a cry is due; fills `out` with one real
   * bird's distance and bearing and returns true, or false when no bird is
   * within earshot (the cry is then skipped). Must not allocate. Null restores
   * the invented placement.
   */
  setGullLocator(locator: ((out: GullFix) => boolean) | null): void {
    this.gullLocator = locator;
  }

  // ------------------------------------------------------------------ interface

  /**
   * Interface feedback, past the underwater muffle and the compressor.
   *
   * @param cue   What happened. See `UiCue`.
   * @param value 0..1 for `slide`: the slider's position, which sets the pitch,
   *              so dragging a slider is heard as a scale rather than a buzz.
   */
  uiTick(cue: UiCue = 'press', value = 0.5): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    const gap = UI_CUE_GAP[cue];
    if (gap === undefined) return;
    const now = ctx.currentTime;
    if (now - this.uiLast[cue] < gap) return;
    this.uiLast[cue] = now;
    switch (cue) {
      case 'press':
        this.playUiClick();
        return;
      case 'hover':
        this.uiTone(now, 2400, 2150, 0.014, UI_HOVER_LEVEL, 'sine');
        return;
      case 'slide': {
        const f = 620 + 1000 * clamp01(finite(value, 0.5));
        this.uiTone(now, f, f * 0.92, 0.022, UI_TICK_LEVEL, 'triangle');
        return;
      }
      case 'toggleOn':
        this.uiTone(now, 587, 587, 0.05, UI_TICK_LEVEL * 1.3, 'triangle');
        this.uiTone(now + 0.06, 880, 880, 0.08, UI_TICK_LEVEL * 1.3, 'triangle');
        return;
      case 'toggleOff':
        this.uiTone(now, 880, 880, 0.05, UI_TICK_LEVEL * 1.2, 'triangle');
        this.uiTone(now + 0.06, 587, 560, 0.08, UI_TICK_LEVEL * 1.1, 'triangle');
        return;
      case 'select':
        // A wooden knock: two inharmonic modes, the upper one dying fast.
        this.uiTone(now, 760, 720, 0.06, UI_TICK_LEVEL * 1.4, 'sine');
        this.uiTone(now, 2080, 2000, 0.022, UI_TICK_LEVEL * 0.8, 'sine');
        return;
      case 'open':
        this.uiRustle(now, 1700, 3600, 0.15, UI_TICK_LEVEL * 1.3);
        return;
      case 'close':
        this.uiRustle(now, 3600, 1500, 0.12, UI_TICK_LEVEL * 1.2);
        return;
      case 'info':
        this.uiRustle(now, 2200, 3200, 0.22, UI_TICK_LEVEL);
        return;
      case 'warn':
        this.uiTone(now, 330, 322, 0.1, UI_TICK_LEVEL * 1.6, 'triangle');
        this.uiTone(now + 0.13, 262, 255, 0.16, UI_TICK_LEVEL * 1.6, 'triangle');
        return;
    }
  }

  /**
   * Anchor chain running out or being hove in: a rattle of metal on wood.
   *
   * Band-passed noise chopped by a fast gate, so it reads as links rather than
   * as hiss. In the world, so it goes through the air bus and is muffled under
   * water with everything else.
   *
   * @param seconds How long the chain runs.
   */
  playAnchorChain(seconds = 1.5): void {
    const ctx = this.ctx;
    if (ctx === null || !this.canPlay()) return;
    if (!this.acquireVoice()) return;

    const duration = clampNumber(finite(seconds, 1.5), 0.2, 6);
    const now = ctx.currentTime;
    const random = this.oneShotRandom;

    const source = ctx.createBufferSource();
    source.buffer = this.white;
    source.loop = true;

    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.frequency.value = CHAIN_BAND_HZ;
    band.Q.value = 2.2;

    // The gate: one click per link, with a little irregularity so it is a chain
    // and not a motor.
    const gate = ctx.createGain();
    gate.gain.setValueAtTime(0, now);
    let t = now;
    while (t < now + duration) {
      gate.gain.setValueAtTime(CHAIN_LEVEL * (0.6 + 0.4 * random()), t);
      gate.gain.exponentialRampToValueAtTime(CHAIN_LEVEL * 0.02, t + 0.035);
      t += 1 / (CHAIN_LINKS_PER_SECOND * (0.8 + 0.4 * random()));
    }
    gate.gain.setValueAtTime(0, now + duration);

    source.connect(band);
    band.connect(gate);
    gate.connect(this.airBus);
    source.start(now, random() * this.white.duration);
    source.stop(now + duration + 0.05);
    this.trackVoice(source, band, gate);
  }

  /**
   * Rewinds every clock this system owns and silences anything still ringing.
   *
   * The audible output is not what a capture compares, but the *state* is: the
   * gust walk, the bubble scheduler and the swell phase all persist across a
   * session, so without this a deterministic run would inherit whatever the
   * previous shot left behind. Note that the swell modulator is a CPU-side phase
   * for exactly this reason — an `OscillatorNode`'s phase cannot be set, so a
   * hardware LFO would be a piece of state no reset could reach.
   */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.bubbleRandom = mulberry32(AUDIO_SEEDS.bubbles);
    this.gustRandom = mulberry32(AUDIO_SEEDS.gusts);
    this.oneShotRandom = mulberry32(AUDIO_SEEDS.oneShots);
    this.gullRandom = mulberry32(AUDIO_SEEDS.gulls);
    this.gust = 1;
    this.gustTarget = 1;
    this.gustTimer = 0;
    this.bubbleTimer = 0;
    // Back to the initial delay, not to zero: a shot that opens on a gull cry is
    // a shot whose first second is about the gull.
    this.gullTimer = GULL_FIRST_DELAY;
    this.wasUnder = false;
    this.plungeCooldown = 0;
    this.stopVoices();
    this.shipSfx?.reset();
    this.music?.reset();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    const ctx = this.ctx;
    this.ctx = null;
    if (ctx === null) return;

    if (this.pumpTimer !== null) clearInterval(this.pumpTimer);
    this.pumpTimer = null;
    this.stopVoices();
    for (const bed of this.beds) bed.dispose();
    this.beds.length = 0;
    this.targets.length = 0;

    this.airBus.disconnect();
    this.airMuffle.disconnect();
    this.underBus.disconnect();
    this.uiBus.disconnect();
    this.mix.disconnect();
    this.master.disconnect();
    this.compressor.disconnect();

    this.music.dispose();
    this.limiter.disconnect();
    this.softClip.disconnect();
    this.bedBus.disconnect();
    this.shipBus.disconnect();
    this.shipPanner?.disconnect();
    if (this.realtime !== null) void this.realtime.close().catch(() => {});
    this.realtime = null;
  }

  // ------------------------------------------------------------------ internals

  private build(ctx: BaseAudioContext): void {
    this.white = createNoiseBuffer(ctx, 'white', AUDIO_SEEDS.white);
    const brown = createNoiseBuffer(ctx, 'brown', AUDIO_SEEDS.brown);
    this.gullWave = buildGullWave(ctx);

    const shape = mulberry32(AUDIO_SEEDS.thunderShape);
    // Two shapes, not one: a crack and a roll are different envelopes, and a
    // parameter cannot cross-fade between two curves. They are separate paths,
    // weighted by distance.
    this.crackCurve = buildThunderCurve(CURVE_POINTS, shape, 7, 0.015, 2, 0.35);
    this.rollCurve = buildThunderCurve(CURVE_POINTS, shape, 2.2, 0.08, 4, 0.55);

    this.master = ctx.createGain();
    // Pinned at zero until a gesture. Belt and braces against the suspend in the
    // constructor: a platform that ignores both still makes no sound.
    this.master.gain.value = this.offline ? this.volume : 0;
    this.limiter = ctx.createDynamicsCompressor();
    this.limiter.threshold.value = LIMITER_THRESHOLD_DB;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = LIMITER_RATIO;
    this.limiter.attack.value = LIMITER_ATTACK;
    this.limiter.release.value = LIMITER_RELEASE;
    this.softClip = ctx.createWaveShaper();
    this.softClip.curve = buildSoftClipCurve(SOFT_CLIP_KNEE);
    this.master.connect(this.limiter);
    this.limiter.connect(this.softClip);
    this.softClip.connect(ctx.destination);

    // The compressor sits *before* the master, so the volume control is a clean
    // output trim. Behind it, turning the volume down would also reduce what the
    // compressor sees, so quiet playback would be less compressed than loud
    // playback — a volume slider that changes the mix as well as the level.
    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = COMPRESSOR_THRESHOLD_DB;
    this.compressor.knee.value = COMPRESSOR_KNEE_DB;
    this.compressor.ratio.value = COMPRESSOR_RATIO;
    this.compressor.attack.value = COMPRESSOR_ATTACK;
    this.compressor.release.value = COMPRESSOR_RELEASE;
    this.compressor.connect(this.master);

    this.mix = ctx.createGain();
    this.mix.connect(this.compressor);

    // Past the compressor as well as past the muffle: an interface sound should
    // be the same every time it is heard, and routing it through the world bus
    // would have a thunderclap duck it.
    this.uiBus = ctx.createGain();
    this.uiBus.connect(this.master);

    this.airMuffle = ctx.createBiquadFilter();
    this.airMuffle.type = 'lowpass';
    this.airMuffle.frequency.value = AIR_OPEN_HZ;
    this.airMuffle.Q.value = 0.7;
    this.airMuffle.connect(this.mix);

    this.airBus = ctx.createGain();
    this.airBus.gain.value = 1;
    this.airBus.connect(this.airMuffle);

    this.underBus = ctx.createGain();
    this.underBus.gain.value = 0;
    this.underBus.connect(this.mix);

    this.bedBus = ctx.createGain();
    this.bedBus.connect(this.airBus);

    // Ship: distance gain, then bearing pan, then into the world with the rest.
    this.shipBus = ctx.createGain();
    this.shipBus.gain.value = 0;
    this.shipGain = new ParamTarget(this.shipBus.gain, 0, 0.25, 0.003, 0, 1);
    this.shipPanner = createStereoPanner(ctx);
    if (this.shipPanner !== null) {
      this.shipBus.connect(this.shipPanner);
      this.shipPanner.connect(this.airBus);
      this.shipPan = new ParamTarget(this.shipPanner.pan, 0, 0.2, 0.01, -1, 1);
    } else {
      this.shipBus.connect(this.airBus);
    }

    const nyquist = ctx.sampleRate * 0.45;
    this.airGain = new ParamTarget(this.airBus.gain, 1, 0.2, 0.004, 0, 1);
    this.muffle = new ParamTarget(
      this.airMuffle.frequency,
      AIR_OPEN_HZ,
      // Fast, because the waterline is crossed by a crest in a fraction of a
      // second and a lazy cutoff turns that into a slow wash rather than a duck.
      0.08,
      4,
      20,
      nyquist,
    );
    this.underGain = new ParamTarget(this.underBus.gain, 0, 0.12, 0.004, 0, 1);
    this.targets.push(this.airGain, this.muffle, this.underGain, this.shipGain);
    if (this.shipPan !== null) this.targets.push(this.shipPan);

    // Offsets spread across the buffer so no two beds read the same sample of it.
    const spacing = NOISE_SECONDS / 8;

    this.surfNear = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'lowpass',
      frequency: SURF_CUTOFF_BASE,
      q: SURF_Q,
      highpass: 60,
      playbackRate: BED_RATES.surfNear,
      offset: spacing * 0,
      pan: -SURF_PAN,
      // Slow: the sea's level is the swell, and a short time constant would make
      // the modulation a tremolo instead of a surge.
      gainTau: 0.35,
    });
    this.surfFar = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'lowpass',
      frequency: SURF_CUTOFF_BASE * SURF_FAR_TILT,
      q: SURF_Q,
      highpass: 60,
      playbackRate: BED_RATES.surfFar,
      offset: spacing * 1,
      pan: SURF_PAN,
      gainTau: 0.35,
    });
    this.windBed = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'bandpass',
      frequency: WIND_BAND_BASE,
      q: WIND_Q,
      highpass: WIND_HIGHPASS,
      playbackRate: BED_RATES.wind,
      offset: spacing * 2,
      pan: 0,
      gainTau: 0.2,
    });
    this.rigging = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'bandpass',
      frequency: 500,
      q: RIGGING_Q,
      highpass: 100,
      playbackRate: BED_RATES.rigging,
      offset: spacing * 3,
      pan: -0.2,
      gainTau: 0.3,
    });
    this.rainBed = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'lowpass',
      frequency: RAIN_CUTOFF_BASE,
      q: RAIN_Q,
      highpass: RAIN_HIGHPASS_BASE,
      playbackRate: BED_RATES.rain,
      offset: spacing * 4,
      pan: RAIN_PAN,
      gainTau: 0.25,
    });
    this.rainHiss = this.addBed(ctx, {
      buffer: this.white,
      destination: this.bedBus,
      filter: 'highpass',
      frequency: RAIN_HISS_HZ,
      q: 0.6,
      highpass: RAIN_HISS_HZ,
      playbackRate: BED_RATES.rainHiss,
      offset: spacing * 5,
      pan: -RAIN_PAN,
      gainTau: 0.25,
    });
    this.hull = this.addBed(ctx, {
      buffer: this.white,
      destination: this.shipBus,
      filter: 'bandpass',
      frequency: HULL_BAND_BASE,
      q: HULL_Q,
      highpass: HULL_HIGHPASS,
      playbackRate: BED_RATES.hull,
      offset: spacing * 6,
      pan: 0,
      gainTau: 0.18,
    });
    this.bowWash = this.addBed(ctx, {
      buffer: this.white,
      destination: this.shipBus,
      filter: 'bandpass',
      frequency: BOW_BAND_BASE,
      q: BOW_Q,
      highpass: 200,
      playbackRate: BED_RATES.bowWash,
      offset: spacing * 6.5,
      pan: 0.1,
      gainTau: 0.12,
    });
    this.luffBed = this.addBed(ctx, {
      buffer: this.white,
      destination: this.shipBus,
      filter: 'bandpass',
      frequency: LUFF_BED_HZ,
      q: 0.7,
      highpass: 150,
      playbackRate: BED_RATES.luff,
      offset: spacing * 7.5,
      pan: 0,
      gainTau: 0.2,
    });
    this.rumble = this.addBed(ctx, {
      // Brown, not white: what is left after a hundred metres of water is the
      // bottom of the spectrum, and a low-passed white bed to match would need a
      // filter steep enough to ring.
      buffer: brown,
      destination: this.underBus,
      filter: 'lowpass',
      frequency: RUMBLE_HZ,
      q: RUMBLE_Q,
      highpass: RUMBLE_HIGHPASS,
      playbackRate: BED_RATES.rumble,
      offset: spacing * 7,
      pan: 0,
      gainTau: 0.4,
    });

    const settings = this.settings;
    this.music = new MusicEngine(ctx, this.mix, this.white, {
      voices: settings.musicVoices,
      reverb: settings.musicReverb,
      volume: this.musicVolume,
    });

    const host: SfxHost = {
      ctx,
      acquire: () => this.acquireVoice(),
      acquireAmbient: () => this.voices.length + SHIP_VOICE_RESERVE < this.settings.voices,
      track: (source, ...nodes) => this.trackVoice(source, ...nodes),
    };
    this.shipSfx = new ShipSfx(host, this.white, this.shipBus);

    this.applyQuality();
  }

  private addBed(ctx: BaseAudioContext, options: NoiseBedOptions): NoiseBed {
    const bed = new NoiseBed(ctx, options);
    this.beds.push(bed);
    for (const target of bed.targets) this.targets.push(target);
    return bed;
  }

  private applyQuality(): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    const now = ctx.currentTime;
    const live = this.enabled && !this.muted && this.running();
    const settings = this.settings;

    this.surfFar.setActive(settings.stereoSurf, now, live);
    // With only one surf bed the sea has to come from the middle, or the whole
    // ocean is off to port.
    this.surfNear.setPan(settings.stereoSurf ? -SURF_PAN : 0, now, live);
    this.rigging.setActive(settings.rigging, now, live);
    this.rainHiss.setActive(settings.rainHiss, now, live);
    this.music.setVoices(settings.musicVoices);
    this.music.setReverb(settings.musicReverb);
  }

  private swellAt(depth: number, period: number, groupPeriod: number, phase: number): number {
    const wave = Math.sin((TAU * this.clock) / period + phase);
    const group = Math.sin((TAU * this.clock) / groupPeriod + phase + GROUP_PHASE);
    return 1 + depth * (0.62 * wave + 0.38 * group);
  }

  private applyMasterGain(rampSeconds: number): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    const now = ctx.currentTime;
    // `gestured` is in the target, not just in `resume`: until the viewer has
    // asked for audio, nothing this class can be told to do makes a sound.
    const target = this.enabled && !this.muted && this.gestured ? this.volume : 0;
    const param = this.master.gain;
    param.cancelScheduledValues(now);
    param.setValueAtTime(param.value, now);
    if (rampSeconds > 0) param.linearRampToValueAtTime(target, now + rampSeconds);
    else param.setValueAtTime(target, now);
  }

  private flushAll(now: number): void {
    for (const target of this.targets) target.flush(now);
  }

  /**
   * Starts the noise sources, on the first resume rather than at construction.
   *
   * See the note in `NoiseBed`: a source started on a locked context is an
   * autoplay warning in the console per source, and eight of those before the
   * viewer has touched anything is a false alarm the next person to read that
   * console has to work out for themselves.
   */
  private startBeds(now: number): void {
    for (const bed of this.beds) bed.start(now);
  }

  /** Whether a one-shot should be built at all. */
  private canPlay(): boolean {
    const ctx = this.ctx;
    return (
      !this.disposed && ctx !== null && this.enabled && !this.muted && this.running()
    );
  }

  /** The context is producing sound: running, or an offline render in progress. */
  private running(): boolean {
    return this.ctx !== null && (this.offline || this.ctx.state === 'running');
  }

  private acquireVoice(): boolean {
    return this.voices.length < this.settings.voices;
  }

  /** The same cap, held short so ambience cannot starve an event. See `GULL_VOICE_RESERVE`. */
  private acquireGullVoice(): boolean {
    return this.voices.length + GULL_VOICE_RESERVE < this.settings.voices;
  }

  /**
   * Registers a one-shot's nodes for cleanup.
   *
   * Without this every splash leaves its chain hanging off the bus: still
   * connected, still summed, and — because the source is stopped but not
   * disconnected — never collected. A busy sea would accumulate them until the
   * audio thread ran out of budget.
   */
  private trackVoice(source: AudioScheduledSourceNode, ...nodes: AudioNode[]): void {
    this.voices.push(source);
    source.onended = () => {
      source.disconnect();
      for (const node of nodes) node.disconnect();
      const index = this.voices.indexOf(source);
      if (index >= 0) this.voices.splice(index, 1);
    };
  }

  private stopVoices(): void {
    // Backwards: `onended` splices from this array, and although it is queued as a
    // task rather than fired synchronously, iterating forwards over a list that
    // something else shortens is a bug waiting for a scheduler change.
    for (let i = this.voices.length - 1; i >= 0; i--) {
      try {
        this.voices[i].stop();
      } catch {
        // Already stopped.
      }
    }
  }

  /** The lens going under (`down`) or breaking the surface. */
  private plunge(down: boolean): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    this.plungeCooldown = PLUNGE_COOLDOWN;
    const now = ctx.currentTime;
    if (down) {
      this.playSplash(0.55);
      // The air the lens drags down with it, heard from below.
      for (let i = 0; i < PLUNGE_BUBBLES; i++) {
        const at = now + 0.04 + i * 0.035 + this.bubbleRandom() * 0.03;
        this.spawnBubble(at, this.underBus, BUBBLE_LEVEL * 1.5);
      }
    } else {
      this.playSplash(0.3);
    }
  }

  private uiTone(
    when: number,
    from: number,
    to: number,
    seconds: number,
    level: number,
    type: OscillatorType,
  ): void {
    const ctx = this.ctx;
    if (ctx === null || !this.acquireVoice()) return;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(from, when);
    if (to !== from) osc.frequency.exponentialRampToValueAtTime(to, when + seconds);
    const gain = ctx.createGain();
    envelope(gain.gain, when, 0.0015, level, seconds);
    osc.connect(gain);
    gain.connect(this.uiBus);
    osc.start(when);
    osc.stop(when + seconds + 0.01);
    this.trackVoice(osc, gain);
  }

  /** A parchment rustle: band noise sweeping, for the ledger opening and closing. */
  private uiRustle(when: number, from: number, to: number, seconds: number, level: number): void {
    const ctx = this.ctx;
    if (ctx === null || !this.acquireVoice()) return;
    const src = ctx.createBufferSource();
    src.buffer = this.white;
    src.loop = true;
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.Q.value = 1.2;
    band.frequency.setValueAtTime(from, when);
    band.frequency.exponentialRampToValueAtTime(to, when + seconds);
    const gain = ctx.createGain();
    const g = gain.gain;
    g.setValueAtTime(0, when);
    g.linearRampToValueAtTime(level, when + seconds * 0.3);
    g.exponentialRampToValueAtTime(level * 1e-3, when + seconds);
    src.connect(band);
    band.connect(gain);
    gain.connect(this.uiBus);
    src.start(when, this.oneShotRandom() * (this.white.duration - 1));
    src.stop(when + seconds + 0.02);
    this.trackVoice(src, band, gain);
  }

  private spawnBubble(when: number, destination: AudioNode, level: number): void {
    const ctx = this.ctx;
    if (ctx === null || !this.acquireVoice()) return;

    const random = this.bubbleRandom;
    const radius = BUBBLE_MIN_RADIUS + (BUBBLE_MAX_RADIUS - BUBBLE_MIN_RADIUS) * random();
    const frequency = MINNAERT_CONSTANT / radius;
    const decay = BUBBLE_DECAY_MIN + BUBBLE_DECAY_SPAN * random();

    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(frequency, when);
    osc.frequency.exponentialRampToValueAtTime(frequency * BUBBLE_RISE, when + decay);

    const gain = ctx.createGain();
    envelope(gain.gain, when, 0.002, level, decay);

    osc.connect(gain);
    gain.connect(destination);
    osc.start(when);
    osc.stop(when + decay + 0.02);
    this.trackVoice(osc, gain);
  }

  /**
   * One cluster of cries, placed around the listener.
   *
   * Each call in the flurry gets its own distance and its own side, because a
   * flock is spread out and the answer comes from somewhere else — two birds at
   * the same distance and the same pan are one bird with an echo. The calls are
   * scheduled *ahead*, at their real offsets, rather than re-triggered from the
   * frame loop: the audio thread's clock is the accurate one, and a cry placed
   * by the render loop lands wherever the frame did.
   */
  private scheduleGullFlurry(now: number): void {
    const random = this.gullRandom;

    // Squared, so most flurries are a single bird. See GULL_FLURRY_MAX.
    const u = random();
    const calls = 1 + Math.floor(u * u * GULL_FLURRY_MAX);

    // Uniform over the annulus rather than over the radius: see GULL_NEAR_METRES.
    // `d = sqrt(near^2 + u (far^2 - near^2))`.
    const near = GULL_NEAR_METRES * GULL_NEAR_METRES;
    const far = GULL_FAR_METRES * GULL_FAR_METRES;

    let when = now;
    for (let i = 0; i < calls; i++) {
      let distance = Math.sqrt(near + random() * (far - near));
      let pan = GULL_PAN_SPREAD * (random() * 2 - 1);
      // With a locator the cry comes from a bird that is actually up there. No
      // bird within earshot, no cry: an empty patch of sky stays empty.
      const locator = this.gullLocator;
      if (locator !== null) {
        const fix = this.gullFix;
        if (!locator(fix) || !(fix.distance <= GULL_FAR_METRES * 1.25)) {
          when += GULL_ANSWER_MIN + GULL_ANSWER_SPAN * random();
          continue;
        }
        distance = clampNumber(finite(fix.distance, GULL_REFERENCE_METRES), GULL_NEAR_METRES, GULL_FAR_METRES);
        pan = GULL_PAN_SPREAD * clampNumber(finite(fix.pan, 0), -1, 1);
      }
      this.spawnGullCall(when, distance, pan);
      when += GULL_ANSWER_MIN + GULL_ANSWER_SPAN * random();
    }
  }

  /**
   * A whole call — one oscillator, one formant, one distance filter, one gain.
   *
   * The series lives entirely in the automation: `writeGullNote` appends one
   * note's worth of events to the three parameters, the gain closes between
   * notes, and nothing starts or stops until the last note has decayed. Seven
   * notes therefore cost seven envelopes and one voice, where seven voices would
   * empty the tier's budget on a single bird.
   */
  private spawnGullCall(when: number, distanceMetres: number, pan: number): void {
    const ctx = this.ctx;
    if (ctx === null || !this.acquireGullVoice()) return;

    const random = this.gullRandom;

    const single = random() < GULL_SINGLE_CHANCE;
    const notes = single ? 1 : GULL_NOTES_MIN + Math.floor(random() * GULL_NOTES_SPAN);

    // Spherical spreading, which is 1/d in pressure and therefore 1/d here.
    const level = GULL_LEVEL * (GULL_REFERENCE_METRES / distanceMetres);
    const cutoff = clampNumber(
      GULL_BRIGHT_HZ * Math.exp(-distanceMetres / GULL_ABSORPTION_METRES),
      GULL_DULL_HZ,
      // Nine tenths of Nyquist, for the reason `NoiseBed` gives: a biquad parked
      // at half the sample rate is unstable rather than transparent.
      Math.min(GULL_BRIGHT_HZ, ctx.sampleRate * 0.45),
    );

    const osc = ctx.createOscillator();
    osc.setPeriodicWave(this.gullWave);

    const formant = ctx.createBiquadFilter();
    formant.type = 'peaking';
    formant.Q.value = GULL_FORMANT_Q;
    formant.gain.value = GULL_FORMANT_DB;
    formant.frequency.value = GULL_FORMANT_HZ * GULL_FORMANT_CLOSED;

    const tone = ctx.createBiquadFilter();
    tone.type = 'lowpass';
    tone.Q.value = 0.7;
    tone.frequency.value = cutoff;

    const gain = ctx.createGain();

    let cursor = when;
    let duration = (GULL_NOTE_MIN + GULL_NOTE_SPAN * random()) * (single ? GULL_MEW_STRETCH : 1);
    let gap = GULL_NOTE_GAP_MIN + GULL_NOTE_GAP_SPAN * random();
    // One draw for the whole series: a bird is the size it is. See GULL_F0_MIN.
    let pitch = GULL_F0_MIN + GULL_F0_SPAN * random();

    for (let k = 0; k < notes; k++) {
      // The arc: up over the first two notes, then away. A lone mew has no
      // crescendo to make and lands at full level.
      const swell = single ? 1 : Math.min(1, (k + 1) / GULL_SWELL_NOTES);
      const fade = Math.pow(GULL_SERIES_FADE, Math.max(0, k - GULL_SWELL_NOTES + 1));
      writeGullNote(
        osc.frequency,
        formant.frequency,
        gain.gain,
        cursor,
        duration,
        pitch,
        level * swell * fade,
        random,
      );
      cursor += duration + gap;
      duration *= GULL_NOTE_SHRINK;
      gap = Math.max(GULL_NOTE_GAP_FLOOR, gap * GULL_NOTE_ACCEL);
      pitch *= GULL_PITCH_FALL;
    }

    const end = cursor + GULL_TAIL;

    osc.connect(formant);
    formant.connect(tone);
    tone.connect(gain);

    // Into the air bus rather than past it, unlike the UI click: a gull heard
    // with the lens half under the surface is supposed to be muffled, and
    // muffling what is above the water is exactly what that bus is for.
    //
    // The pan ramp is bearing drift across the call — the bird is flying — and it
    // is scaled by the same inverse distance the level is, because a far bird's
    // bearing hardly moves while a near one's sweeps.
    const panner = createStereoPanner(ctx);
    if (panner === null) {
      gain.connect(this.airBus);
    } else {
      const drift = GULL_PAN_DRIFT * (GULL_REFERENCE_METRES / distanceMetres) * (random() * 2 - 1);
      panner.pan.setValueAtTime(pan, when);
      panner.pan.linearRampToValueAtTime(clampNumber(pan + drift, -1, 1), end);
      gain.connect(panner);
      panner.connect(this.airBus);
    }

    osc.start(when);
    osc.stop(end);
    if (panner === null) this.trackVoice(osc, formant, tone, gain);
    else this.trackVoice(osc, formant, tone, gain, panner);
  }
}

// ---------------------------------------------------------------------------

/**
 * Attack-and-decay on a gain parameter.
 *
 * The decay is exponential and stops at a small non-zero value rather than at 0,
 * because `exponentialRampToValueAtTime` cannot reach zero — it throws on a target
 * of 0, and a linear tail in its place is audible as the decay changing character
 * right at the end. -80 dB is inaudible and the node is stopped immediately after.
 */
function envelope(
  param: AudioParam,
  start: number,
  attack: number,
  peak: number,
  decay: number,
): void {
  // An exponential ramp is also undefined from a starting value of zero, which is
  // what a zero peak would leave on the parameter after the attack.
  const level = Math.max(1e-4, peak);
  param.setValueAtTime(0, start);
  param.linearRampToValueAtTime(level, start + attack);
  param.exponentialRampToValueAtTime(level * 1e-4, start + attack + decay);
}

/**
 * One note of a gull's call, appended to three parameters that are already live.
 *
 * A note is a stretch of automation, not a voice: the notes of a call are
 * separated by this gain envelope closing, and the oscillator underneath runs
 * from the first note to the last. Every event written here is strictly after
 * the previous note's, which is what makes appending safe.
 *
 * The pitch contour has three parts and they are the whole difference between a
 * bird and a beep:
 *
 *  - **The scoop.** The note enters a fourth low and reaches pitch in its first
 *    fifth. The air pulse that makes the sound also has to spin the syrinx up;
 *    starting on pitch is the sound of an oscillator being switched on.
 *  - **The waver.** Two small, unequal excursions across the held body, drawn
 *    rather than fixed so no two notes wobble identically. A dead-steady pitch
 *    under a stack this rich is unmistakably synthetic.
 *  - **The glide.** The last two fifths fall by about a third. This is the part
 *    a person imitating a gull would produce, and it is the bill closing — which
 *    is why the formant is dragged down with it rather than held.
 *
 * The amplitude holds before it falls, and starts falling *before* the glide
 * does. A note that decays from its attack is plucked, and plucked is a
 * different family of sound entirely; a gull leans on the note.
 */
function writeGullNote(
  frequency: AudioParam,
  formant: AudioParam,
  gain: AudioParam,
  start: number,
  duration: number,
  pitch: number,
  level: number,
  random: () => number,
): void {
  const peak = pitch * GULL_NOTE_PEAK;
  const openAt = start + duration * GULL_NOTE_RISE;
  const holdAt = start + duration * GULL_NOTE_HOLD;
  const glideAt = start + duration * GULL_NOTE_GLIDE;
  const end = start + duration;

  // Unequal on purpose: a symmetric wobble is a vibrato, and vibrato is a
  // trained human, not a bird.
  const waverA = peak * (1 - GULL_NOTE_WAVER * (0.4 + 0.6 * random()));
  const waverB = peak * (1 + GULL_NOTE_WAVER * (0.2 + 0.5 * random()));

  frequency.setValueAtTime(pitch * GULL_NOTE_SCOOP, start);
  frequency.exponentialRampToValueAtTime(peak, openAt);
  frequency.exponentialRampToValueAtTime(waverA, openAt + (glideAt - openAt) * 0.45);
  frequency.exponentialRampToValueAtTime(waverB, glideAt);
  frequency.exponentialRampToValueAtTime(pitch * GULL_NOTE_FALL, end);

  // The `setValueAtTime` in the middle is what creates the hold: without it the
  // closing ramp would start at `openAt` and the gape would be sliding shut
  // through the whole note.
  formant.setValueAtTime(GULL_FORMANT_HZ * GULL_FORMANT_CLOSED, start);
  formant.exponentialRampToValueAtTime(GULL_FORMANT_HZ, openAt);
  formant.setValueAtTime(GULL_FORMANT_HZ, glideAt);
  formant.exponentialRampToValueAtTime(GULL_FORMANT_HZ * GULL_FORMANT_CLOSED, end);

  // Floored for the same reason `envelope` floors its peak: an exponential ramp
  // is undefined from zero, and the quietest note of a distant series is small.
  const loudness = Math.max(1e-4, level);
  gain.setValueAtTime(0, start);
  gain.linearRampToValueAtTime(loudness, start + Math.min(GULL_NOTE_ATTACK, duration * 0.2));
  gain.setValueAtTime(loudness, holdAt);
  gain.exponentialRampToValueAtTime(loudness * 1e-4, end);
}

/**
 * The gull's wavetable. See `GULL_PARTIALS` for the spectrum it describes.
 *
 * Normalisation is left on — the default — so these coefficients are purely a
 * *shape*: the wave comes out peaking at 1 whatever slope is chosen, and the
 * level of a cry stays the business of its envelope. Retuning the timbre then
 * cannot silently retune the mix, which is exactly the trap a hand-summed
 * harmonic stack sets.
 */
function buildGullWave(ctx: BaseAudioContext): PeriodicWave {
  // Index 0 is DC and stays zero; a wave with a DC term is an offset, not a
  // sound, and it would eat headroom on the bus for something inaudible.
  const real = new Float32Array(GULL_PARTIALS + 1);
  const imag = new Float32Array(GULL_PARTIALS + 1);
  for (let h = 1; h <= GULL_PARTIALS; h++) {
    const slope = Math.pow(h, -GULL_PARTIAL_SLOPE);
    imag[h] = h === 1 ? slope * GULL_FUNDAMENTAL_WEIGHT : slope;
  }
  return ctx.createPeriodicWave(real, imag);
}

/**
 * A panner, or `null` where there is none.
 *
 * `createStereoPanner` is the one node this file uses that is not universal on
 * the browsers targeted, and a platform without it should lose the flock its
 * spread and nothing else — the same guard, and the same reasoning, as `NoiseBed`.
 */
function createStereoPanner(ctx: BaseAudioContext): StereoPannerNode | null {
  const factory = (ctx as BaseAudioContext & { createStereoPanner?: () => StereoPannerNode })
    .createStereoPanner;
  return typeof factory === 'function' ? ctx.createStereoPanner() : null;
}

/**
 * A decaying envelope with a few slow bumps in it, normalised to peak at 1.
 *
 * Thunder is not one impulse. The channel is kilometres long and tortuous, so its
 * sections arrive at different times and from different directions — which is why
 * thunder rolls rather than bangs, and why a single exponential decay sounds like
 * a gunshot in a stairwell. The bumps are a handful of sinusoids at seeded,
 * mutually irrational rates: enough structure to read as a roll, slow enough that
 * they never turn into a tremolo.
 */
function buildThunderCurve(
  points: number,
  random: () => number,
  decayRate: number,
  attack: number,
  bumps: number,
  depth: number,
): Float32Array {
  const rates = new Float64Array(bumps);
  const phases = new Float64Array(bumps);
  for (let b = 0; b < bumps; b++) {
    rates[b] = 0.7 + 3.2 * random();
    phases[b] = TAU * random();
  }

  const curve = new Float32Array(points);
  let peak = 0;
  for (let i = 0; i < points; i++) {
    const t = i / (points - 1);
    let wobble = 0;
    for (let b = 0; b < bumps; b++) wobble += Math.sin(TAU * rates[b] * t + phases[b]);
    const shaped = Math.max(0, 1 + (depth * wobble) / bumps);
    const value = Math.min(1, t / attack) * Math.exp(-t * decayRate) * shaped;
    curve[i] = value;
    if (value > peak) peak = value;
  }

  const scale = peak > 1e-6 ? 1 / peak : 0;
  for (let i = 0; i < points; i++) curve[i] *= scale;
  // The curve holds its last value after it finishes, so it has to end at silence.
  curve[points - 1] = 0;
  return curve;
}

/**
 * Linear to `knee`, then a tanh shoulder. A `WaveShaperNode` holds its end
 * values for input beyond [-1, 1], so the curve's last point is the ceiling.
 */
function buildSoftClipCurve(knee: number): Float32Array<ArrayBuffer> {
  const n = 2049;
  const curve = new Float32Array(new ArrayBuffer(n * 4));
  const span = 1 - knee;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    const a = Math.abs(x);
    const y = a <= knee ? a : knee + span * Math.tanh((a - knee) / span);
    curve[i] = Math.sign(x) * y;
  }
  return curve;
}

function resolveAudioContext(): AudioContextConstructor | null {
  if (typeof window === 'undefined') return null;
  // Through `unknown`: `webkitAudioContext` is not on the `Window` lib type at
  // all, and intersecting an optional override onto a required member of it
  // produces a property type that is not what either half says.
  const scope = window as unknown as {
    AudioContext?: AudioContextConstructor;
    webkitAudioContext?: AudioContextConstructor;
  };
  return scope.AudioContext ?? scope.webkitAudioContext ?? null;
}

function finite(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function clamp01(v: number): number {
  return clampNumber(v, 0, 1);
}

function smoothstepNumber(x: number, edge0: number, edge1: number): number {
  const t = clampNumber((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
