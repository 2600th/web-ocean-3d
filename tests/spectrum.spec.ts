import { test, expect } from '@playwright/test';
import {
  CASCADES,
  DEFAULT_SPECTRUM,
  generateInitialSpectrum,
  type SpectrumParams,
} from '../src/ocean/Spectrum';

/**
 * Node-only, like `fft.spec.ts` and `meshSampling.spec.ts`: `generateInitialSpectrum`
 * is CPU code and nothing here needs a GPU or a page.
 *
 * What it guards is the directional spreading, which is the one part of the
 * spectrum whose failure mode is *silent*. A wrong JONSWAP term makes the sea
 * obviously flat or obviously enormous and the sea-state test in `ocean.spec.ts`
 * catches it. A wrong spreading term produces a sea of exactly the right height
 * that is combed the wrong way, and nothing that measures amplitude will ever
 * notice.
 */

const SIZE = 256;

/** Summed |h0|^2 over a cascade — the variance the surface will carry from it. */
function energy(params: SpectrumParams, cascadeIndex: number): number {
  const data = generateInitialSpectrum(SIZE, CASCADES[cascadeIndex], params, 1337);
  let total = 0;
  for (let i = 0; i < data.length; i += 4) {
    total += data[i] * data[i] + data[i + 1] * data[i + 1];
  }
  return total;
}

/**
 * Energy-weighted mean of |cos(theta)| about the wind axis, where theta is the
 * heading of each wave component.
 *
 * A perfectly uniform sea gives 2/pi ~= 0.637, which is the mean of |cos| over a
 * circle; 1 would be every component running exactly along the wind. Measured
 * values come out a little under the uniform figure because the band edges cull
 * a square grid into a disc unevenly. It is a directionality score, not an
 * angle, and it is the thing a fixed-exponent lobe gets wrong at high frequency.
 */
function directionality(params: SpectrumParams, cascadeIndex: number): number {
  const cascade = CASCADES[cascadeIndex];
  const data = generateInitialSpectrum(SIZE, cascade, params, 1337);
  const deltaK = (2 * Math.PI) / cascade.tileSize;
  const half = SIZE / 2;
  const windX = Math.cos(params.windDirection);
  const windZ = Math.sin(params.windDirection);

  let weighted = 0;
  let total = 0;
  for (let z = 0; z < SIZE; z++) {
    for (let x = 0; x < SIZE; x++) {
      const kx = (x - half) * deltaK;
      const kz = (z - half) * deltaK;
      const kLength = Math.hypot(kx, kz);
      if (kLength < 1e-6) continue;
      const index = (z * SIZE + x) * 4;
      const e = data[index] * data[index] + data[index + 1] * data[index + 1];
      if (e <= 0) continue;
      total += e;
      weighted += e * Math.abs((kx * windX + kz * windZ) / kLength);
    }
  }
  return total > 0 ? weighted / total : 0;
}

test.describe('directional spreading', () => {
  /**
   * The whole point of the Hasselmann model over a fixed exponent: the spread is
   * a function of frequency, so the dominant sea tracks the wind while the
   * ripples do not. Cascade 0 carries 24 m and up, cascade 2 carries 6 m and
   * down, and the difference between them has to be large — with the fixed
   * `cos^8` lobe this test would report the same number three times.
   */
  test('short waves are far less directional than the dominant sea', () => {
    const swellAxis = directionality(DEFAULT_SPECTRUM, 0);
    const chopAxis = directionality(DEFAULT_SPECTRUM, 1);
    const rippleAxis = directionality(DEFAULT_SPECTRUM, 2);

    console.log(
      `directionality  swell=${swellAxis.toFixed(3)} chop=${chopAxis.toFixed(3)} ripple=${rippleAxis.toFixed(3)}`,
    );

    // Ordered, and by a wide margin at each step.
    expect(swellAxis).toBeGreaterThan(chopAxis + 0.05);
    expect(chopAxis).toBeGreaterThan(rippleAxis + 0.05);

    // Measured against a uniform sea on the same grid, not against 2/pi. The
    // ripple band starts five texels from the centre of a 16 m tile, so its
    // headings are quantised, and on the diagonal wind the default uses a
    // uniform spectrum scores 0.653 there, above the continuous 0.637. A fixed
    // bound under 2/pi failed a correct spectrum for that reason alone.
    const uniform = (cascade: number) => directionality({ ...DEFAULT_SPECTRUM, standingWaveRatio: 1 }, cascade);

    // The swell still knows where the wind is...
    expect(swellAxis).toBeGreaterThan(uniform(0) + 0.15);
    // ...and the ripples have very nearly forgotten.
    expect(rippleAxis).toBeLessThan(uniform(2) + 0.01);
  });

  /**
   * `spectralSharpness` is renormalised, so it redistributes energy between
   * headings without creating any. If `lobeNormalisation` were dropped or got
   * its gamma terms the wrong way up, this is what would catch it — and nothing
   * else would, because the sea would still be a plausible height.
   */
  test('spectral sharpness moves energy between headings without changing how much there is', () => {
    const broad: SpectrumParams = { ...DEFAULT_SPECTRUM, spectralSharpness: 0.5 };
    const calibrated = DEFAULT_SPECTRUM;
    const narrow: SpectrumParams = { ...DEFAULT_SPECTRUM, spectralSharpness: 3 };

    for (let cascade = 0; cascade < CASCADES.length; cascade++) {
      const base = energy(calibrated, cascade);
      for (const [label, params] of [
        ['broad', broad],
        ['narrow', narrow],
      ] as const) {
        const ratio = energy(params, cascade) / base;
        console.log(`cascade ${cascade} ${label}: energy x${ratio.toFixed(4)}`);
        // The tolerance is for the discrete sum over a finite grid, not for the
        // model: the continuous integral is exactly 1 for every spread. A
        // narrow lobe concentrates energy into fewer texels, so the quadrature
        // error grows with sharpness — hence a band rather than a bound.
        expect(ratio).toBeGreaterThan(0.75);
        expect(ratio).toBeLessThan(1.25);
      }
    }

    // ...and it did actually change the shape, or the test above proves nothing.
    expect(directionality(narrow, 0)).toBeGreaterThan(directionality(broad, 0) + 0.05);
  });

  /**
   * The normalisation is evaluated in log space precisely so that a large
   * spread cannot overflow. `gamma(2s + 1)` leaves double range around s = 85,
   * which `spectralSharpness: 12` on the dominant sea reaches comfortably.
   */
  test('a very sharp spread stays finite', () => {
    const extreme: SpectrumParams = { ...DEFAULT_SPECTRUM, spectralSharpness: 12 };
    const data = generateInitialSpectrum(SIZE, CASCADES[0], extreme, 1337);
    let finite = 0;
    let energyTotal = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (!Number.isFinite(data[i]) || !Number.isFinite(data[i + 1])) continue;
      finite++;
      energyTotal += data[i] * data[i] + data[i + 1] * data[i + 1];
    }
    expect(finite).toBe(data.length / 4);
    expect(energyTotal).toBeGreaterThan(0);
  });
});
