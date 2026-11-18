import { mulberry32 } from '../core/random';

/**
 * Looping noise buffers. Every continuous layer in the ambience is filtered
 * noise, and all of it is drawn from one of these.
 *
 * **Buffers, not an `AudioWorklet`.** A worklet would produce genuinely endless
 * noise with no loop at all, but it has to be loaded from a separate module URL
 * at runtime: a second artefact to bundle, a path that has to survive whatever
 * base URL the site is deployed under, and a failure mode where the ambience is
 * silently missing on exactly the deployment nobody tested. A pre-filled
 * `AudioBuffer` is a `Float32Array` and nothing else — it cannot fail to load,
 * and it works identically under a test harness with no network.
 *
 * **The loop seam is folded, not spliced.** Butting the end of a buffer against
 * its start leaves a step discontinuity, which is a click. White noise is full of
 * steps so it survives, but the brown noise the underwater rumble uses is a
 * low-frequency signal whose endpoints are far apart, and there the splice is a
 * thud once per loop. So the generator runs `length + fade` samples and folds the
 * overrun back over the head with an equal-power crossfade: sample 0 becomes the
 * natural continuation of sample `length - 1`, and the loop point stops existing.
 * Equal-power (sin/cos) rather than linear, because the two ends are uncorrelated
 * noise — a linear crossfade of uncorrelated signals dips 3 dB in the middle,
 * which for a bed that loops every few seconds is an audible periodic sag.
 *
 * **Loop period (A5).** Four seconds of noise repeated, and the narrow rigging
 * band made that repeat audible as a texture. Two changes remove it. The
 * buffer is ten seconds (1.9 MB at 48 kHz for the white, the same for the
 * brown), so at the beds' rates of 0.83 to 1.31 the shortest repeat is 7.6 s;
 * and each bed's read rate now wanders slowly by a few per cent
 * (`NoiseBed.wander`), on its own incommensurate period, so the buffer is
 * never read the same way twice and no repeat is exact. For white noise the
 * rate changes nothing heard: the spectrum is flat either way and the band
 * is the filter's. Every bed also still reads from its own offset at its own
 * base rate, so no two layers repeat together.
 *
 * **Seeded.** Same reason every other procedural system in this project is: a
 * capture harness that mutes the audio still constructs it, and construction that
 * called `Math.random()` would allocate a different buffer every run for no
 * benefit whatsoever.
 */

export type NoiseColor = 'white' | 'brown';

/** Seconds of noise per buffer. See the loop-period note above. */
export const NOISE_SECONDS = 10;

/** Length of the fold that removes the loop seam, seconds. */
const LOOP_FADE_SECONDS = 0.03;

/**
 * Corner of the one-pole that turns white noise brown, Hz.
 *
 * Above the corner the pole falls at 6 dB/octave, which *is* brown noise. Below
 * it the pole stops integrating, which is the whole reason it is a leaky one: a
 * true integrator has infinite gain at DC, so a naive brown generator wanders off
 * to an arbitrary offset and spends most of its headroom there. 20 Hz is under
 * everything the rumble bed is filtered down to anyway.
 */
const BROWN_CORNER_HZ = 20;

/** Peak the finished buffer is normalised to. Headroom for the per-bed gains. */
const PEAK_TARGET = 0.85;

export function createNoiseBuffer(
  ctx: BaseAudioContext,
  color: NoiseColor,
  seed: number,
  seconds: number = NOISE_SECONDS,
): AudioBuffer {
  const rate = ctx.sampleRate;
  const length = Math.max(1, Math.round(Math.max(0.25, seconds) * rate));
  const fade = Math.min(Math.max(1, Math.round(LOOP_FADE_SECONDS * rate)), length >> 1);

  const random = mulberry32(seed);
  const scratch = new Float32Array(length + fade);

  if (color === 'brown') {
    const alpha = Math.exp((-2 * Math.PI * BROWN_CORNER_HZ) / rate);
    let last = 0;
    for (let i = 0; i < scratch.length; i++) {
      last = alpha * last + (1 - alpha) * (random() * 2 - 1);
      scratch[i] = last;
    }
  } else {
    for (let i = 0; i < scratch.length; i++) scratch[i] = random() * 2 - 1;
  }

  for (let i = 0; i < fade; i++) {
    const t = (i + 0.5) / fade;
    const head = Math.sin(t * Math.PI * 0.5);
    const tail = Math.cos(t * Math.PI * 0.5);
    scratch[i] = scratch[i] * head + scratch[length + i] * tail;
  }

  let peak = 0;
  for (let i = 0; i < length; i++) {
    const magnitude = Math.abs(scratch[i]);
    if (magnitude > peak) peak = magnitude;
  }
  const scale = peak > 1e-6 ? PEAK_TARGET / peak : 0;

  const buffer = ctx.createBuffer(1, length, rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < length; i++) channel[i] = scratch[i] * scale;
  return buffer;
}
