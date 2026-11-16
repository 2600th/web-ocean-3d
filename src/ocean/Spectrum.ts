import * as THREE from 'three/webgpu';

export const GRAVITY = 9.81;

export interface SpectrumParams {
  /** Wind speed at 10 m reference height, m/s. */
  windSpeed: number;
  /** Wind direction in radians, measured in the XZ plane. */
  windDirection: number;
  /** Wavelength of the spectral peak, metres. Drives the dominant swell size. */
  peakWavelength: number;
  /**
   * Fraction of the spectrum's own peak frequency below which energy is culled.
   * Keeps neighbouring cascades from double-counting the same wavelengths.
   */
  fetch: number;
  /** Peak enhancement (JONSWAP gamma). 1 = Pierson–Moskowitz, 3.3 = typical fetch-limited sea. */
  gamma: number;
  /**
   * Multiplier on the Hasselmann directional spread.
   *
   * 1 is the calibrated model and is what every preset uses. Above 1 narrows the
   * whole spectrum toward the wind axis, below 1 broadens it. It is a stylisation
   * knob, not a sea-state one: the lobe is renormalised after it is applied, so
   * turning it does not change how much energy the sea carries — only how that
   * energy is shared between headings.
   */
  spectralSharpness: number;
  /** Fraction of energy travelling against the wind (chop realism). */
  swell: number;
  /**
   * Blend from travelling waves (0) to standing waves (1).
   *
   * A travelling sea sweeps through the scene; a standing one oscillates in
   * place. Sheltered water — a bay, a lagoon, a harbour — is much closer to the
   * second, because the swell that would drive it has been broken up by the
   * land and what is left is the interference of reflections. 0.3 to 0.5 is the
   * range that reads as sheltered without looking like a ripple tank.
   *
   * It also broadens the spectrum: a standing wave is the sum of two components
   * running opposite ways, so the directional bias that the wind imposes only
   * applies to the part that is still travelling. Both halves of that are
   * implemented — the shape here, the motion in `OceanSimulation`.
   */
  standingWaveRatio: number;
  /** Overall amplitude scale. */
  amplitude: number;
}

export const DEFAULT_SPECTRUM: SpectrumParams = {
  windSpeed: 15,
  windDirection: Math.PI * 0.25,
  peakWavelength: 47,
  fetch: 1,
  gamma: 3.3,
  spectralSharpness: 1,
  swell: 0.2,
  standingWaveRatio: 0,
  amplitude: 1,
};

/**
 * One spectral band. Three cascades at decreasing tile sizes cover swell, chop
 * and ripple without any single tile becoming visibly periodic.
 */
export interface CascadeConfig {
  /** Physical size of the tile in metres. */
  tileSize: number;
  /** Wavelengths outside [minWavelength, maxWavelength] are culled from this band. */
  minWavelength: number;
  maxWavelength: number;
  /** Per-cascade weighting of horizontal displacement (choppiness). */
  choppiness: number;
  /**
   * Detail band: shading only, never displaced geometry. Uses the capillary
   * dispersion relation and web-ocean's millimetre short-wave cut instead of
   * the 2 cm-scale `capillaryCutoff` the geometric bands need. See
   * `detailCascades`.
   */
  detail?: boolean;
  /** Amplitude multiplier for the band. Detail bands are toned down; see `detailCascades`. */
  gain?: number;
}

/**
 * Surface tension over density, m^3/s^2 — the capillary term in
 * `w^2 = g k + (sigma/rho) k^3`. 7.4e-5 is web-ocean's value (sim.js `SURF`) and
 * the textbook one for sea water at 20 degrees. It moves a 1 cm ripple's
 * frequency by 70%, and a 12 cm one by under 2%, so only the detail bands use it.
 */
export const SURFACE_TENSION = 7.4e-5;

/**
 * A shading-only band below the shortest wave the geometric cascades carry.
 *
 * **Why it exists.** The critic's main complaint about our near water was
 * "jelly-like mottling", blurred blobs with no fine detail, and the cause is
 * spectral, not filtering. The ripple cascade's 16 m tile at N texels cannot hold
 * anything shorter than `2 * 16 / N` (12.5 cm at High), and the spectrum's own
 * `capillaryCutoff` had already removed most of the energy below about 8 cm.
 * Slope variance in a saturated wind sea is roughly constant per octave of
 * wavenumber (the k^-4 Phillips range), so the missing five octaves, 4 mm to
 * 12 cm, carry as much *slope* as everything from 12 cm to the swell. That is the
 * part of the surface that breaks a sun track into glints and a reflected cloud
 * into fragments. Without it a pixel 5 m away sees one smooth metre-scale facet,
 * and its bilinearly interpolated slope draws the stair-stepped "blocky glitter"
 * squares of refraction-1.
 *
 * **What is ported.** Noniv/web-ocean (MIT) runs cascades down to a 0.26 m tile
 * for exactly this ("The smallest cascade produces the sharp sun sparkles near
 * the camera"), with capillary dispersion, a 1.2 mm Gaussian cut, and its two
 * finest bands toned to 0.85 and 0.6 because at full amplitude they "read as
 * crumpled foil" (sim.js `amplitude`). This band spans both of those, so it takes
 * their mean, 0.72.
 *
 * **One band, not web-ocean's two, and that is a measured limit.** The water's
 * fragment stage sits at the WebGPU default of 16 sampled textures; a second
 * detail texture made it 17 and the pipeline failed to create. One band can span
 * the whole range because the well-resolved range of an N-texel tile of size L is
 * `[2L/N, L/4]`, a factor of N/8 (32 at High): the tile is `4 x` the band's top,
 * `128 / N` metres (0.5 m at High), and its bottom is then `256 / N^2` (3.9 mm).
 * The material reads it twice, the second time rotated by an irrational angle,
 * so the 0.5 m tile never repeats visibly (see the detail loop in
 * `OceanMaterial`).
 *
 * The edges are expressed in `size` so the band starts exactly where the ripple
 * cascade's resolution ends at every tier: 50 cm down to 6 cm at Low (N = 64),
 * 12.5 cm to 4 mm at High, 6 cm to 1 mm at Max.
 *
 * Fragment-only: the mesh is centimetres apart at best, so displacing it by
 * millimetre waves would only alias. Buoyancy, foam folding and the wake never
 * see this band, so the physics is unchanged.
 */
export function detailCascades(size: number): CascadeConfig[] {
  const rippleTile = CASCADES[CASCADES.length - 1].tileSize;
  const top = (2 * rippleTile) / size;
  const tile = 4 * top;
  return [{ tileSize: tile, minWavelength: (2 * tile) / size, maxWavelength: top, choppiness: 0, detail: true, gain: 0.72 }];
}

/**
 * The three bands, and why each tile is the size it is.
 *
 * A cascade of `N` texels over an `L`-metre tile carries wavenumbers on a grid
 * of spacing `2pi/L` out to `pi N / L`. In wavelengths that is a well-resolved
 * range of roughly `[2L/N, L/4]`: below `2L/N` the mode does not exist at all,
 * and above `L/4` the band is described by a disc only a few texels across, so
 * its directional content is quantised into a handful of headings. A band that
 * sits outside its own tile's range is not an error the code can report — it
 * simply renders less sea than the spectrum was asked for — so the tiles are
 * chosen against the bands rather than the other way round.
 *
 * At the tiers' resolutions (N = 128 / 256 / 512) the well-resolved ranges are:
 *
 *   L = 1024   [16, 256] m at N=128,  [4, 256] m at N=512
 *   L =  128   [ 2,  32] m at N=128,  [0.5, 32] m at N=512
 *   L =    8   [0.125, 2] m at N=128, [0.03, 2] m at N=512
 *
 * **The swell tile was 512 m and is now 1024.** Two things improve at once and
 * neither costs anything: the dominant sea repeats at twice the distance, which
 * is the tiling a wide shot actually shows, and the band occupies a disc of
 * radius 43 texels instead of 21 — four times the modes, on the same grid, for
 * the part of the spectrum carrying nearly all the energy. Nothing is lost at
 * the short end because the band stops at 24 m and the tile still resolves to
 * 4 m.
 *
 * **The 3-6 m octave stayed in the ripple cascade, and that is also measured.**
 * Moving it up to the chop cascade is attractive on paper: at the top of a 16 m
 * tile a 6 m wave is a disc of radius 2.7 texels with its heading quantised into
 * a handful of spokes, where on the 128 m tile it would have a radius of 42 and
 * a real spectrum. What that ignores is which cascades a tier actually runs.
 * Medium enables two of the three, so the ripple cascade is off there and those
 * wavelengths are simply not rendered — moving them into a cascade Medium *does*
 * run hands that tier a whole octave of short waves it never had, and
 * `gallery-jitter` measured the result at 2.352 against its 2.33 far-field
 * ceiling. Detail a tier cannot resolve is not detail; it is shimmer.
 *
 * **The ripple tile stayed at 16 m too, and for a similar reason.** It was
 * halved to 8, on the reasoning that the band asked for 5 cm
 * from a grid whose finest mode is 6.25 cm at the top tier — so the shortest
 * waves in the spectrum were never rendered anywhere. Halving the tile does fix
 * that, and it also quarters the number of modes in the 1-3 m annulus, which at
 * the High tier's 256 texels means 49 modes where there were 197. Those are the
 * waves that fold, and `tests/foam.spec.ts` caught it immediately: whitecap
 * coverage at 9 m/s fell from inside its factor-of-three band around Monahan to
 * a fifth of the prediction, because the same energy in a quarter of the modes
 * makes fewer, larger folds rather than many small ones.
 *
 * The honest way to finer ripples is to give this cascade more texels than the
 * other two, which is what the reference product does. That is not available
 * here: the three cascades share one spectrum atlas and one set of butterfly
 * passes — deliberately, and recently, for the performance it buys — so raising
 * one raises all three and costs 4x the transform. The band edge below is
 * therefore left at the value the spectrum wants rather than the value the grid
 * can reach, and the gap is a known limit rather than a bug.
 */
export const CASCADES: CascadeConfig[] = [
  { tileSize: 1024, minWavelength: 24, maxWavelength: 10000, choppiness: 1.1 },
  { tileSize: 128, minWavelength: 3, maxWavelength: 24, choppiness: 1.0 },
  { tileSize: 16, minWavelength: 0.05, maxWavelength: 3, choppiness: 0.85 },
];

/**
 * Builds the time-zero spectrum h0(k) for one cascade.
 *
 * Packing is RGBA = ( h0(k).re, h0(k).im, h0(-k).re, h0(-k).im ) so the per-frame
 * evolution pass can form the Hermitian pair with a single texture fetch.
 *
 * The model is JONSWAP in the frequency domain, mapped to wavenumber space via the
 * deep-water dispersion relation w^2 = g|k|, with a cosine-power directional
 * spreading term. Generated on the CPU: it is O(N^2) once per parameter change
 * (~1 ms at 256^2), and keeping it here avoids a GPU noise-generation pass and
 * gives byte-identical results across backends.
 */
export function generateInitialSpectrum(
  size: number,
  cascade: CascadeConfig,
  params: SpectrumParams,
  seed = 1337,
): Float32Array {
  const data = new Float32Array(size * size * 4);
  const anchor = anchorGaussians(seed);
  const anchorHalf = ANCHOR_SIZE / 2;

  const peakOmega = Math.sqrt((GRAVITY * 2 * Math.PI) / params.peakWavelength);
  const deltaK = (2 * Math.PI) / cascade.tileSize;
  const windX = Math.cos(params.windDirection);
  const windZ = Math.sin(params.windDirection);

  // Precompute the Gaussian pairs so that changing wind speed re-weights an
  // unchanged noise field — the sea keeps its identity instead of reshuffling.
  const half = size / 2;

  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      const kx = (x - half) * deltaK;
      const kz = (z - half) * deltaK;

      // This mode's draw, by its wavevector rather than by its place in the
      // texture. See `anchorGaussians`.
      const ax = x - half + anchorHalf;
      const az = z - half + anchorHalf;
      let ga: number;
      let gb: number;
      if (ax >= 0 && ax < ANCHOR_SIZE && az >= 0 && az < ANCHOR_SIZE) {
        const t = (az * ANCHOR_SIZE + ax) * 2;
        ga = anchor[t];
        gb = anchor[t + 1];
      } else {
        const g = gaussianPair(mulberry32(modeSeed(x - half, z - half, seed)));
        ga = g.a;
        gb = g.b;
      }

      const index = (z * size + x) * 4;
      const amplitude =
        spectrumAmplitude(kx, kz, cascade, params, peakOmega, windX, windZ, deltaK) * Math.SQRT1_2;

      data[index + 0] = ga * amplitude;
      data[index + 1] = gb * amplitude;
    }
  }

  // **h0(-k) is h0 at the mirrored texel, not a second independent draw.**
  //
  // The evolution forms h(k, t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}
  // (Tessendorf, "Simulating Ocean Water", 2001, eq. 26), and that is Hermitian
  // — h(-k) = conj(h(k)), so the field it transforms to is real — only when
  // h0(-k) is the *same* function evaluated at -k. This used to fill the second
  // pair with a fresh Gaussian draw. The transform of every field then came
  // out complex, and the simulation packs two fields per transform as P + iQ
  // on the promise that P and Q are real (`OceanSimulation`, "Field packing"):
  // each field's imaginary half leaked into its partner. The worst case is the
  // cross term dDx/dz, which shares a transform with the height and so carried
  // the height's imaginary half — metres, against a term that should be a few
  // tenths. Measured on the 1024 m cascade at 18 m/s, the stored Jacobian
  // correlated 0.17 with the Jacobian of the displacement actually rendered,
  // averaged 0.65 where any periodic displacement averages exactly 1, and read
  // 9% of the sea as folded; the slopes the surface is lit with correlated
  // only 0.58–0.82 with the slope of the geometry it is drawn on.
  //
  // `(size - i) % size` is -k on the centred lattice (index `half` is k = 0;
  // the Nyquist row maps to itself). The amplitude carries 1/√2 because the
  // old real part held half the variance of a complex field: with it the
  // height field keeps exactly the statistics it had, so the sea state, the
  // buoyancy and everything tuned against wave height are unchanged, and only
  // the parts that were noise — the leaked halves — are gone.
  for (let z = 0; z < size; z++) {
    const mz = (size - z) % size;
    for (let x = 0; x < size; x++) {
      const mirror = (mz * size + ((size - x) % size)) * 4;
      const index = (z * size + x) * 4;
      data[index + 2] = data[mirror + 0];
      data[index + 3] = data[mirror + 1];
    }
  }

  return data;
}

/**
 * Energy gain on the JONSWAP level for a sea younger than fully developed,
 * applied to the waves near the peak and released toward the short-wave tail.
 *
 * **Why.** The level was Phillips' 0.0081 at every sea state, which is the
 * Pierson-Moskowitz value for a *fully developed* sea. A preset's peak is set
 * independently of its wind, and at 15 m/s with a 47 m peak the sea is far from
 * developed (a developed 15 m/s sea peaks near 190 m): it is fetch-limited, and
 * JONSWAP's own fetch laws (Hasselmann et al. 1973) put its level well above
 * 0.0081. Written in terms of the peak instead of the fetch, eliminating the
 * dimensionless fetch between `f_p U / g = 3.5 x^-0.33` and
 * `alpha = 0.076 x^-0.22`:
 *
 *     alpha = 0.076 (f_p U / (3.5 g))^(2/3)
 *
 * which returns 0.0089 for a developed sea (so 0.0081 stays the floor), 0.0141
 * for the 15 m/s, 47 m preset and about 0.016 at 21 m/s, 60 m.
 *
 * **Near the peak only.** The level of the short waves is set by the wind's
 * friction velocity, not by fetch: Elfouhaily et al. (1997) split the
 * spectrum into a long-wave part, whose level grows with inverse wave age, and
 * a short-wave part on u* alone (the one `OceanMaterial.shortWaveGain` already
 * applies to the detail band). Their long-wave part is released above the peak
 * by `exp(-(Omega / sqrt(10)) (sqrt(k / k_p) - 1))`, Omega = U / c_p, and the
 * extra level here rides that same envelope, so a ripple keeps exactly the
 * energy it had and only the dominant waves grow.
 *
 * **What it was for.** waves-4 (round 6): at 15 m/s and 47 m the swell carried
 * 82% of the height variance but only 10% of the slope variance (rms slope
 * 0.051 against 0.088 for the 3-24 m chop and 0.129 under 3 m), and read as
 * "short chop" beside Water Pro's swell. At the peak this gives 1.74x the
 * energy (1.32x the height); at 12 m 1.4x; under 3 m nearly none. A calm 5 m/s
 * sea with a 20 m peak is almost developed and changes by about 10%.
 */
function youngSeaGain(windSpeed: number, peakOmega: number, omega: number): number {
  const u = Math.max(0.5, windSpeed);
  const peakFrequency = peakOmega / (2 * Math.PI);
  const level = 0.076 * Math.pow((peakFrequency * u) / (3.5 * GRAVITY), 2 / 3);
  const gain = Math.max(1, level / 0.0081);
  if (omega <= peakOmega) return gain;
  // k / k_p = (w / w_p)^2 in deep water, so sqrt(k / k_p) = w / w_p.
  const inverseWaveAge = (u * peakOmega) / GRAVITY;
  const release = Math.exp((-inverseWaveAge / Math.sqrt(10)) * (omega / peakOmega - 1));
  return 1 + (gain - 1) * release;
}

function spectrumAmplitude(
  kx: number,
  kz: number,
  cascade: CascadeConfig,
  params: SpectrumParams,
  peakOmega: number,
  windX: number,
  windZ: number,
  deltaK: number,
): number {
  const kLength = Math.hypot(kx, kz);
  if (kLength < 1e-6) return 0;

  // Band-limit so cascades tile different wavelength ranges without overlap.
  const wavelength = (2 * Math.PI) / kLength;
  if (wavelength < cascade.minWavelength || wavelength > cascade.maxWavelength) return 0;

  // Capillary dispersion on the detail bands only (web-ocean `omegaOf`); the
  // geometric bands keep pure gravity waves so their phases, and with them the
  // buoyancy readback and every captured frame, are unchanged.
  const detail = cascade.detail === true;
  const omega = detail
    ? Math.sqrt(GRAVITY * kLength + SURFACE_TENSION * kLength * kLength * kLength)
    : Math.sqrt(GRAVITY * kLength);

  // --- JONSWAP energy density in the frequency domain ---
  const sigma = omega <= peakOmega ? 0.07 : 0.09;
  const peakRatio = (omega - peakOmega) / (sigma * peakOmega);
  const peakEnhancement = Math.pow(params.gamma, Math.exp(-0.5 * peakRatio * peakRatio));
  const alpha = 0.0081 * youngSeaGain(params.windSpeed, peakOmega, omega);
  const jonswap =
    ((alpha * GRAVITY * GRAVITY) / Math.pow(omega, 5)) *
    Math.exp(-1.25 * Math.pow(peakOmega / omega, 4)) *
    peakEnhancement;

  // --- Map frequency density to wavenumber density ---
  // dw/dk = g / (2w); the extra 1/k converts a 1-D density to a 2-D one.
  // With the capillary term dw/dk = (g + 3 sigma k^2) / (2w) (web-ocean `dOmegaOf`).
  const jacobian = detail
    ? (GRAVITY + 3 * SURFACE_TENSION * kLength * kLength) / (2 * omega * kLength)
    : GRAVITY / (2 * omega * kLength);

  // --- Directional spreading ---
  //
  // Hasselmann, Dunckel & Ewing (1980), in the form Horvath collected for
  // graphics use (DigiPro 2015): the spreading exponent is a **function of
  // frequency**, not a constant.
  //
  //   s(w) = 6.97 (w/wp)^4.06                                   w <= wp
  //   s(w) = 9.77 (w/wp)^(-2.33 - 1.45 (U/cp - 1.17))           w >  wp
  //
  // paired with the Longuet-Higgins (1963) half-angle lobe, cos^2s(theta/2).
  //
  // This replaced a fixed `cos^8(theta)` lobe, and the fixed exponent was the
  // least physical thing left in the spectrum. It made a 3 cm ripple exactly as
  // directional as a 50 m swell, so all three cascades combed the same way and
  // the surface read as corduroy — an ordered texture that a moving light rakes
  // across instead of glinting off. Real short waves are nothing like that: they
  // are raised locally by turbulent wind stress and are close to omnidirectional,
  // which is why open water carries a chaotic sparkle *over* an ordered swell.
  // The model puts numbers on it. At the default 15 m/s and a 47 m peak, s runs
  // from about 9.8 at the dominant sea to under 0.01 by the time the wavelength
  // is a metre — that is the whole ripple cascade going effectively isotropic.
  //
  // The exponents look enormous because the argument is the *half* angle:
  // cos^2s(theta/2) at s = 9.77 is about as wide a lobe as cos^5(theta), not
  // cos^19.5(theta). The identity cos^2(theta/2) = (1 + cos theta) / 2 means it
  // costs one add and one multiply from the dot product we already have.
  const omegaRatio = omega / peakOmega;
  // Inverse wave age. cp is the phase speed at the peak, so U/cp > 1 is a sea
  // still being driven by its wind and U/cp ~ 1 is a fully developed one; the
  // model uses it to say that a young sea's short waves spread faster.
  const inverseWaveAge = params.windSpeed / (GRAVITY / peakOmega);
  const hasselmann =
    omegaRatio > 1
      ? 9.77 * Math.pow(omegaRatio, -2.33 - 1.45 * (inverseWaveAge - 1.17))
      : 6.97 * Math.pow(omegaRatio, 4.06);
  // Floored rather than clamped to zero: s = 0 is the uniform distribution and
  // is a perfectly good limit, but `pow(0, 0)` on the way there is not.
  const spread = Math.max(1e-4, hasselmann * params.spectralSharpness);

  const cosTheta = (kx * windX + kz * windZ) / kLength;
  // The forward lobe, plus `swell` of the same lobe reflected — the half-angle
  // form is exactly zero directly against the wind, and a real sea is not, so
  // the back lobe is what keeps a little energy running the other way. Both are
  // divided by (1 + swell) so adding it redistributes energy rather than adding
  // it, which is the same contract `spectralSharpness` honours.
  const forwardLobe = Math.pow(Math.max(0, (1 + cosTheta) * 0.5), spread);
  const backLobe = Math.pow(Math.max(0, (1 - cosTheta) * 0.5), spread);
  const windDriven =
    (lobeNormalisation(spread) * (forwardLobe + params.swell * backLobe)) / (1 + params.swell);

  // Standing water forgets the wind. A standing wave is two components running
  // opposite ways at equal amplitude, so the heading bias the wind imposes can
  // only apply to whatever is still travelling — which is why a sheltered bay
  // has no run to it even when the wind over it does. `1 / (2 pi)` is the
  // uniform distribution, and it is the correct other end of this blend rather
  // than a fudge: it is what `lobeNormalisation` returns as the spread goes to
  // zero, so the two limits meet exactly and total energy is unchanged either
  // way.
  const standing = 1 / (2 * Math.PI);
  const directional = windDriven + (standing - windDriven) * params.standingWaveRatio;

  // Suppress the very shortest waves — beyond this they are normal-map detail,
  // not geometry, and they alias badly.
  // The detail bands are exactly that normal-map detail, so they take
  // web-ocean's 1.2 mm cut (sim.js `misc.y = 0.0012`) instead.
  const capillaryCutoff = Math.exp(-kLength * kLength * (detail ? 0.0012 * 0.0012 : 0.0004));

  const energy = jonswap * jacobian * directional * capillaryCutoff * params.amplitude;
  if (!Number.isFinite(energy) || energy <= 0) return 0;

  // sqrt(2 * S(k) * dk^2) / sqrt(2) collapses to sqrt(S) * dk.
  return Math.sqrt(energy) * deltaK * (cascade.gain ?? 1);
}

/**
 * Q(s) for the `cos^2s(theta/2)` lobe — the factor that makes it integrate to 1
 * over the circle.
 *
 *   Q(s) = 2^(2s-1) / pi * gamma(s + 1)^2 / gamma(2s + 1)
 *
 * Without it the spread would be an energy control as well as a shape one, and
 * the sea state would change every time the wind age did. With it, `s` only ever
 * decides *where* the energy goes: JONSWAP alone decides how much there is.
 *
 * Evaluated in log space. `gamma(2s + 1)` overflows a double at about s = 85 —
 * reachable with `spectralSharpness` turned up on a long swell — while the ratio
 * it appears in stays entirely well behaved, so the two terms must never be
 * formed separately.
 */
function lobeNormalisation(spread: number): number {
  return Math.exp(
    (2 * spread - 1) * Math.LN2 -
      Math.log(Math.PI) +
      2 * logGamma(spread + 1) -
      logGamma(2 * spread + 1),
  );
}

/**
 * Lanczos approximation to log gamma, g = 7 with the standard nine-term series.
 * Better than 1e-13 relative for the positive arguments this file asks for.
 */
const LANCZOS_G7 = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

function logGamma(x: number): number {
  const z = x - 1;
  let series = LANCZOS_G7[0];
  for (let i = 1; i < LANCZOS_G7.length; i++) series += LANCZOS_G7[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(series);
}

/** Box–Muller. Returns two independent standard normals per call. */
function gaussianPair(random: () => number): { a: number; b: number } {
  let u = random();
  const v = random();
  if (u < 1e-9) u = 1e-9;
  const radius = Math.sqrt(-2 * Math.log(u));
  return { a: radius * Math.cos(2 * Math.PI * v), b: radius * Math.sin(2 * Math.PI * v) };
}

/**
 * The grid every cascade's random draws are keyed to, whatever its own size.
 *
 * A tier change rebuilds the simulation at another FFT size, and the draws were
 * taken in texel order: at 64, 128 and 256 the same wave mode got a different
 * Gaussian, so every tier drew an unrelated sea. Measured at one frame of
 * clear-day-wide, the height on a grid round the ship differed from High's by
 * 1.44 times its own RMS on Low and 1.10 on Medium — uncorrelated — and the
 * ship's heave by 0.56 m. Switching tier live dropped the hull onto a different
 * sea, which is a jump in the physics and a pop in the picture.
 *
 * Keyed to the wavevector instead, a mode has one draw at every size: a smaller
 * FFT keeps the modes it can carry and loses only the short waves past its
 * Nyquist, a larger one adds short waves round the same swell. The table is
 * the stream this function always drew at 256, two pairs per texel with the
 * second discarded (see below for why it can no longer be used), so High and
 * Ultra are byte-identical to what they were; Low, Medium and Max now show the
 * same swell as High rather than a sea of their own. Modes outside 256 (only
 * Max's 512 has any) are drawn from a hash of the wavevector.
 */
const ANCHOR_SIZE = 256;
const anchorTables = new Map<number, Float64Array>();

function anchorGaussians(seed: number): Float64Array {
  let table = anchorTables.get(seed);
  if (table) return table;
  table = new Float64Array(ANCHOR_SIZE * ANCHOR_SIZE * 2);
  const random = mulberry32(seed);
  for (let i = 0; i < ANCHOR_SIZE * ANCHOR_SIZE; i++) {
    const g = gaussianPair(random);
    gaussianPair(random);
    table[i * 2] = g.a;
    table[i * 2 + 1] = g.b;
  }
  anchorTables.set(seed, table);
  return table;
}

/** Seed for one mode outside the anchor grid, from its signed wavevector index. */
function modeSeed(ix: number, iz: number, seed: number): number {
  let h = Math.imul(ix ^ 0x27d4eb2d, 0x85ebca6b) ^ Math.imul(iz + 0x165667b1, 0xc2b2ae35) ^ Math.imul(seed, 0x9e3779b9);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  return (h ^ (h >>> 12)) >>> 0;
}

/** Small deterministic PRNG so a given seed always yields the same sea. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * `h0(k)` for every cascade side by side: `columns * size` wide, `size` tall.
 *
 * One texture rather than one per cascade so a single evolution pass can serve
 * all of them — see `OceanSimulation.build`. Cascade `c` occupies the columns
 * `[c * size, (c + 1) * size)`.
 */
export function createSpectrumTexture(
  size: number,
  columns: number,
  data: Float32Array,
): THREE.DataTexture {
  const texture = new THREE.DataTexture(data, size * columns, size, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Significant wave height implied by the spectrum, used to size the buoyancy
 * probe search and the camera's underwater test margin.
 */
export function significantWaveHeight(params: SpectrumParams): number {
  // Pierson–Moskowitz fully developed sea: Hs ~= 0.22 * U^2 / g.
  return (0.22 * params.windSpeed * params.windSpeed) / GRAVITY;
}
