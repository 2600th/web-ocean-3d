import { If, cameraPosition, float, fract, mix, sin, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { smoothstepDown, smoothstepDownClamped } from '../core/tslMath';
import { featherFoam, foamCells, foamPattern } from './FoamPattern';
// A cycle (Seafloor draws the swash from here), safe because both sides read
// the other only inside functions, never while the modules load.
import { TERRACES, shoreTerraceMask } from '../scene/Seafloor';

export { foamCells };

/**
 * The surf: breakers that roll in toward a beach in sets, and the swash that
 * runs up the sand after them.
 *
 * Three consumers draw the same surf and have to agree to the frame on where
 * every breaker is: the ocean's vertex stage lifts the water into the breaker
 * and the swash sheet, its fragment stage shades and foams them, and the island's
 * ground darkens the sand where the swash reached and dries it again. So the
 * surf is not a texture or a simulation but a **closed-form function of depth,
 * position and time**, defined once here, with its uniforms shared by every
 * material that calls it. A deterministic rewind of the sea clock rewinds the
 * surf with it, and nothing has to be kept in step by hand.
 *
 * **Shoaling and refraction, from one phase.** A wave entering water of depth
 * `d` travels at `sqrt(g d)` (shallow-water dispersion). On a beach of slope `s`
 * the time a crest takes to reach depth `d` from the shoreline is
 * `2 sqrt(d) / (s sqrt(g))`, so the breaker phase at a point is
 *
 *     phi = t / T + 2 sqrt(d) / (s sqrt(g) T) + (along-shore terms)
 *
 * and a crest is wherever `phi` is an integer. Everything a surf zone does
 * visually falls out of that: crests follow depth contours, so they arrive
 * square to the beach whatever the wind bearing (refraction); they bunch up as
 * the water shallows, because `sqrt(d)` compresses (shoaling — the wavelength
 * at 2 m is half what it is at 8 m); and they slow down as they come in. `s` is
 * a nominal 1:10, measured off the island's own beaches (0.10–0.13 at the
 * waterline, steepening offshore). The exceptions are the low-tide terraces
 * in `Seafloor`, where `shoalingRoot` adds the slower crossing of each shelf.
 *
 * **Sets.** `floor(phi)` numbers the breakers, so a wave's height can be a
 * function of its number: a seven-wave group envelope times a per-wave jitter.
 * A crest keeps its number along its whole length, so a big wave is big all
 * the way down the beach — and breaks further out, because a wave breaks where
 * the depth falls to `H / 0.78` (McCowan). The along-shore terms shift the
 * phase slowly round the island (so the whole coast is not hit by the same wave
 * at once) and quickly over tens of metres (so a crest breaks in sections and
 * peels rather than collapsing as one straight line).
 *
 * **Swash.** When a bore reaches the shoreline its wave number's run-up starts:
 * the front climbs the sand to `R_n` and falls back over most of a period
 * (`SWASH_SPAN`). Sand at height `h` is under the sheet between the up-crossing
 * and down-crossing of that front, both invertible in closed form, so the time
 * since the sand was last covered is known exactly for the last few waves —
 * which is what lets the ground darken where the swash reached and dry over
 * seconds, without any history buffer.
 *
 * **The pitch.** A heightfield cannot overturn, and a breaker that only grows
 * taller reads as a swell that stopped. Horizon Forbidden West's breakers
 * (Malan, SIGGRAPH 2022 Advances) are one cross-section animated in XYZ and
 * slid along the wavefront: the crest runs *ahead* of the face below it. The
 * same idea in closed form is a Gerstner-style horizontal offset — the crest
 * is pushed shoreward by a bump centred just behind it, so the water ahead of
 * the crest is compressed and the face stands up toward vertical. How far it
 * pitches is the surf similarity (Iribarren) number of the wave on this beach,
 * `xi = slope / sqrt(H / L0)` (Battjes 1974): below ~0.4 a wave *spills*,
 * foam tumbling down a face that never overhangs; from ~0.5 it *plunges*,
 * throwing the lip forward. The compression it causes is returned as a
 * Jacobian, and the ocean shades and foams the face from it — so the lip goes
 * white where the water is actually pitching, the way Tessendorf foam goes
 * white where the open sea folds.
 *
 * **Run-up and the marks it leaves.** Vertical run-up scales as
 * `slope * sqrt(H L0)` (Hunt 1959; the setup and incident-swash terms of
 * Stockdon et al. 2006 have the same form), so a longer-period swell climbs
 * further up the same beach than a wind sea of the same height. Each swash
 * strands a line of foam at its highest point that outlives the water by
 * seconds — the swash mark — and the sand behind a retreating sheet is a
 * mirror for a moment and then merely dark: a draining film and a damp bed,
 * two time constants, not one.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

const GRAVITY = 9.81;

/** Beach slope the phase assumes, rise over run. See the header. */
const NOMINAL_SLOPE = 0.1;

/** McCowan's breaker index: a wave breaks when H reaches this times the depth. */
export const BREAKER_INDEX = 0.78;

/** Waves in one set: the group envelope's period, in breakers. */
const SET_LENGTH = 7;

/**
 * Vertical run-up as a multiple of Hunt's `slope * sqrt(H L0)`.
 *
 * Hunt's form is for regular waves; for an irregular sea Stockdon's fit puts the
 * 2%-exceedance run-up at about 1.1 times it and the *mean* swash well below.
 * 0.75 is calibrated to keep the run-up this module drew before the period
 * term existed: 0.83 m for the average breaker of a 15 m/s sea at the default
 * 6.3 s period, where the old fixed fraction of breaker height gave 0.84 m.
 * What changes is the dependence — a long swell now climbs further than a short
 * sea of the same height, and run-up grows as the square root of height rather
 * than in proportion to it, so a glassy day still has a swash.
 */
const RUNUP_HUNT = 0.75;

/** Fraction of a wave period the swash front spends on the sand. */
const SWASH_SPAN = 0.82;

/**
 * Time warp of the swash front, and where its peak falls as a fraction of
 * `SWASH_SPAN`: `sin(pi v^0.7)` peaks at `v = 0.5^(1/0.7) = 0.3715`. See
 * `swashFront`.
 *
 * It replaces `sin(pi v)(1 - v/4)`, which was meant to climb fast and fall
 * slowly but peaked at v = 0.47 — barely skewed — and at 0.878 of the run-up,
 * so the front stopped 12% short of the height the wetness was computed for.
 */
const SWASH_SKEW = 0.7;
const SWASH_PEAK_AT = 0.3715;

/**
 * Two drying constants, seconds (e-folds).
 *
 * The *film* is the water sheet draining into the bed: a mirror for a second or
 * two, which is the glint on the sand behind every backwash. The *damp* is what
 * the bed holds after the film has gone. Reflectance falls steeply at the first
 * moisture and barely moves after (Nolet et al. 2014), so damp sand stays dark
 * long after it has stopped shining. Twelve seconds is two periods: the sand a
 * wave reached is still visibly darker when the next-but-one arrives, so the
 * swash zone reads as a band rather than flickering wave by wave.
 */
const FILM_SECONDS = 1.6;
const DAMP_SECONDS = 12;

/** Seconds a stranded swash mark takes to burst back to nothing, e-fold. */
const MARK_SECONDS = 4.5;

/**
 * The pitch: width of the horizontal bump in phase, where its centre sits
 * relative to the crest, and the deepest compression it may cause.
 *
 * `LEAN_WIDTH` 0.07 of a local wavelength is ~2 m at the break of a 15 m/s sea
 * — about the width of the face — and the offset puts the bump's peak a hair
 * behind the crest so the crest, not the face, is what runs ahead. The bump is
 * scaled so its steepest compression is exactly `1 - LEAN_MAX * lean`: never
 * below 0.15, so the face can approach vertical but the mesh never folds back
 * through itself, which a heightfield ocean has no way to draw.
 */
const LEAN_WIDTH = 0.07;
const LEAN_OFFSET = 0.02;
const LEAN_MAX = 0.85;
/** `2 x exp(-x^2)` peaks at `sqrt(2/e)`; see `breaker`. */
const BUMP_SLOPE_PEAK = Math.sqrt(2 / Math.E);

/**
 * How far out, in break depths of the largest wave, the surf is computed at
 * all. Past this a wave is a swell like any other and the FFT field owns it.
 */
export const SURF_REACH = 2.4;

/** Shared by every material that draws the surf. Written by `setSurfState`. */
export const surfUniforms = {
  /** The sea clock, seconds. */
  time: uniform(0),
  /** Breaker period, seconds. */
  period: uniform(6.3),
  /** Height of an average breaker arriving at the beach, metres. */
  height: uniform(1.5),
  /** Master, 0..1. */
  strength: uniform(1),
  /**
   * 1 when a shoreline is near enough to the viewer for the vertex stage to
   * look for one; the ocean's vertex stage skips the heightfield otherwise.
   */
  active: uniform(1),
  /** Deep-water wavelength of the breaker period, `g T^2 / 2 pi`, metres. */
  deepWavelength: uniform(62),
  /** `RUNUP_HUNT * slope * sqrt(L0)`: run-up per square-root metre of height. */
  runupScale: uniform(0.59),
};

/**
 * @param time        Sea clock, seconds — the one the FFT runs on.
 * @param period      Breaker period, seconds.
 * @param height      Average breaker height, metres.
 * @param strength    Master, 0..1.
 * @param active      Whether a shore is within reach of the viewer.
 */
export function setSurfState(
  time: number,
  period: number,
  height: number,
  strength: number,
  active: boolean,
): void {
  surfUniforms.time.value = time;
  const T = Math.max(2, period);
  surfUniforms.period.value = T;
  surfUniforms.height.value = Math.max(0.2, height);
  surfUniforms.strength.value = Math.max(0, strength);
  surfUniforms.active.value = active ? 1 : 0;
  const deep = (GRAVITY * T * T) / (2 * Math.PI);
  surfUniforms.deepWavelength.value = deep;
  surfUniforms.runupScale.value = RUNUP_HUNT * NOMINAL_SLOPE * Math.sqrt(deep);
}

/**
 * Highest the swash can climb, metres above still water: the run-up of the
 * largest wave a set can hold (see `waveHeight`, whose factors top out at
 * 1.2 x 1.15 x 1.15 x 1.5 = 2.38), with the 18% lobe `runup` adds.
 */
function maxRunup(): any {
  return surfUniforms.runupScale.mul(surfUniforms.height.mul(2.38).sqrt()).mul(1.18);
}

/** Deepest water, metres, in which anything here is non-zero. */
export function surfReachDepth(): any {
  // The biggest wave of a set, on a peak, is ~1.5 times the average. The
  // tallest peaks of the tallest sets go past that; their fade just starts
  // late. Left there so the reach does not grow into the lagoon floor.
  return surfUniforms.height.mul((1.5 / BREAKER_INDEX) * SURF_REACH);
}

/** Arithmetic value noise in [0,1] — no texture, so it costs nothing to bind. */
export function surfNoise(p: any): any {
  const i = p.floor();
  const f = fract(p);
  const u: any = f.mul(f).mul(f.mul(-2).add(3));
  const h = (o: any) => fract(sin(i.add(o).dot(vec2(127.1, 311.7))).mul(43758.5453));
  return (mix as any)(
    (mix as any)(h(vec2(0, 0)), h(vec2(1, 0)), u.x),
    (mix as any)(h(vec2(0, 1)), h(vec2(1, 1)), u.x),
    u.y,
  );
}

/** Hash of an integer-valued float to [0,1). */
function hash1(n: any): any {
  return fract(sin(n.mul(12.9898).add(4.1414)).mul(43758.5453));
}

/**
 * Phase offset that depends only on where along the coast a point is.
 *
 * Three periods of slow variation round the island, and a third of a period of
 * fast variation over about 45 m so a crest breaks in sections.
 *
 * Between them, half a period over about 80 m. A real swell is a spread of
 * directions and periods, so its crests bend and their spacing varies. One
 * direction and period drew every breaker parallel to the last and a
 * wavelength behind it: the "evenly ruled" bands of foam-4 and the ruler-
 * straight shore line of beach-transparency-2. The term is a 2-D field, so it
 * varies across the surf zone as well as along it. That bunches and spreads
 * successive crests by about a fifth either way. Its steepest slope is under a
 * third of the phase's own shoreward gradient on the terrace, so a crest never
 * turns back on itself.
 */
function alongShorePhase(xz: any): any {
  return surfNoise(xz.div(320))
    .mul(3)
    .add(surfNoise(xz.div(80).add(vec2(2.9, 13.7))).mul(0.5))
    .add(surfNoise(xz.div(45).add(vec2(17.3, 5.1))).mul(0.35));
}

/**
 * `sqrt(depth)`, rescaled to what it would be on the nominal 1:10 beach, so the
 * phase's `2 sqrt(d) / (s sqrt(g))` is the true travel time on this bed.
 *
 * On a plane of slope `s` a crest takes `2 sqrt(d) / (s sqrt(g))` to reach the
 * shoreline. On one of `Seafloor`'s low-tide terraces it crosses the shelf at
 * the terrace's slope and then the face at the nominal slope. The travel time
 * is the sum of the legs, so over the shelf the depth's root counts
 * `NOMINAL_SLOPE / slope` times over. The crest spacing,
 * `sqrt(g d) T`, does not depend on the slope, so `surfPhaseGradient` is right
 * on both. Without this term the breakers would cross the shelf at four times
 * their real spacing.
 */
function shoalingRoot(depth: any, xz: any): any {
  const root = depth.max(0).sqrt();
  let out: any = root;
  for (const tr of TERRACES) {
    // Face to the step, the shelf at its own slope to where it ends, and the
    // floor beyond at the nominal slope again.
    const step = Math.sqrt(tr.stepDepth);
    const end = Math.sqrt(tr.endDepth);
    const shelf = root
      .min(step)
      .add(root.clamp(step, end).sub(step).mul(NOMINAL_SLOPE / tr.slope))
      .add(root.sub(end).max(0));
    out = mix(out, shelf, shoreTerraceMask(xz, tr));
  }
  return out;
}

/** Breaker phase at a point; crests are at integer values. See the header. */
export function surfPhase(depth: any, xz: any): any {
  const k = float(2 / (NOMINAL_SLOPE * Math.sqrt(GRAVITY))).div(surfUniforms.period);
  return surfUniforms.time
    .div(surfUniforms.period)
    .add(shoalingRoot(depth, xz).mul(k))
    .add(alongShorePhase(xz));
}

/**
 * How fast the phase changes walking *shoreward*, per metre. Negative: the
 * crest ahead of you is the next one to arrive.
 */
export function surfPhaseGradient(depth: any): any {
  const k = float(2 / (NOMINAL_SLOPE * Math.sqrt(GRAVITY))).div(surfUniforms.period);
  return k.mul(NOMINAL_SLOPE * 0.5).div(depth.max(0.06).sqrt()).negate();
}

/** Height of breaker number `n` at `xz`, metres. */
export function waveHeight(n: any, xz: any): any {
  const envelope = sin(n.mul((Math.PI * 2) / SET_LENGTH).add(1.3)).mul(0.5).add(0.5);
  const jitter = hash1(n).mul(0.3).add(0.85);
  const along = surfNoise(xz.div(90).add(vec2(3.7, 9.2))).mul(0.3).add(0.85);
  // Peaks along the crest, re-drawn for every wave. A wave is never the same
  // height along its whole length: it has peaks, where it breaks first and
  // furthest out, and shoulders it peels toward. The 90 m term above is the
  // bathymetry's steady share; this one is the wave's own, at about the 25 m
  // a peak spans, so each breaker comes in as sections that go at different
  // depths instead of the one extruded tube beach-transparency-1 showed.
  //
  // The range runs from under half to one and a half times the wave's
  // height, with the middle squeezed out. A low section breaks well inside
  // the high ones beside it, or reaches the shore unbroken. That leaves gaps
  // in the line of white, where 0.75-1.25 only nudged a continuous line in
  // and out.
  const peaks = surfNoise(xz.div(30).add(vec2(hash1(n.add(11)).mul(53), hash1(n.add(17)).mul(41))))
    .smoothstep(0.12, 0.88)
    .mul(1.05)
    .add(0.45);
  return surfUniforms.height
    .mul(envelope.pow(1.5).mul(0.7).add(0.5))
    .mul(jitter)
    .mul(along)
    .mul(peaks)
    .mul(shelterAt(xz));
}

/** Breaker height kept where a terrace's sector is sheltered; see `Terrace.shelter`. */
function shelterAt(xz: any): any {
  let kept: any = float(1);
  for (const tr of TERRACES) {
    if (tr.shelter > 0) kept = kept.mul(shoreTerraceMask(xz, tr).mul(-tr.shelter).add(1));
  }
  return kept;
}

/**
 * The shape of a shoaling breaker across one period, and its derivative.
 *
 * `s` is the signed phase distance from the nearest crest, -0.5..0.5, negative
 * on the shoreward (front) face. Asymmetric: a long gentle back and a front that
 * steepens toward vertical as `steep` goes to 1 at the break, which is what a
 * wave about to break looks like from the beach. Mean-free, so it lifts crests
 * without lifting the sea.
 *
 * @returns vec2(elevation in units of amplitude, d elevation / d phase)
 */
export function crestProfile(s: any, steep: any, minWidth: any = float(0)): any {
  const width = s
    .lessThan(0)
    .select(mix(float(0.15), float(0.055), steep), float(0.24))
    .max(minWidth);
  const x = s.div(width);
  const bump = x.mul(x).negate().exp();
  const elevation = bump.sub(0.3);
  const slope = bump.mul(x).mul(-2).div(width);
  return vec2(elevation, slope);
}

/** Signed distance in phase from the nearest crest. Negative on the front face. */
export function crestDistance(phase: any): any {
  const f = fract(phase);
  return f.greaterThan(0.5).select(f.sub(1), f);
}

/** Everything about the breaker at one point that the ocean needs. */
export interface BreakerSample {
  /** Breaker phase; crests at integers. */
  phase: any;
  /** Signed phase distance to the nearest crest, negative on its front face. */
  s: any;
  /** Elevation the breaker adds to the sea, metres. */
  eta: any;
  /** d(eta) / d(phase), metres per period. */
  etaPerPhase: any;
  /** Depth the nearest crest breaks in, metres. */
  breakDepth: any;
  /** Depth the last crest to pass broke in, and the two before it. */
  breakDepthLast: any;
  breakDepthPrevious: any;
  breakDepthOlder: any;
  /** 0 far out, 1 at the break: how close the nearest crest is to going. */
  steep: any;
  /**
   * Surf similarity number of the nearest crest on this beach. Below ~0.4 it
   * spills, above ~0.5 it plunges. See the header.
   */
  xi: any;
  /** Horizontal offset of the water toward the beach, metres. See the header. */
  lean: any;
  /**
   * `d(x + lean) / dx` along the shoreward direction: 1 on undisturbed water,
   * falling toward `1 - LEAN_MAX` where the face is pitching. The slope of the
   * displaced surface is the undisplaced slope divided by this.
   */
  compression: any;
}

/**
 * The breaker at a point of seabed depth `depth` (>= 0). Must be called inside
 * a TSL flow; every intermediate is a variable of that flow.
 *
 * Shoaled height follows Green's law (`H ~ d^-1/4`) up to the break, after
 * which the wave is a bore whose height is held at the breaker index times the
 * depth — so it shrinks all the way in, which is the energy going into foam.
 * The amplitude fades out toward `SURF_REACH`, where the FFT field owns the
 * swell again.
 *
 * @param depth      Seabed depth, metres, >= 0.
 * @param xz         World position, for the along-shore terms.
 * @param beachSlope Local bottom slope, rise over run; the nominal 1:10 where the
 *                   caller has not measured it. Only the pitch reads it.
 * @param pitch      0..1 master on the horizontal pitch. At 0 the lean is zero
 *                   and the compression exactly 1.
 * @param footprint  Metres the caller's mesh (or pixel) spans here. The crest
 *                   is widened to at least this, so a mesh too coarse to carry
 *                   a 1.5 m front face draws a softer one rather than landing
 *                   every other vertex on the crest — the periodic saw-tooth
 *                   along the far breakers in foam-4 and beach-transparency-1.
 *                   The same band-limit the cascades take (`FOOTPRINT_SPACINGS`).
 */
export function breaker(
  depth: any,
  xz: any,
  beachSlope: any = float(NOMINAL_SLOPE),
  pitch: any = float(1),
  footprint: any = float(0),
): BreakerSample {
  const phase = surfPhase(depth, xz).toVar('surfPhase');
  const last = phase.floor().toVar('surfLast');
  const s = crestDistance(phase).toVar('surfS');
  // On the front face the nearest crest is the *next* one to arrive.
  const near = s.lessThan(0).select(last.add(1), last).toVar('surfNear');
  const height = waveHeight(near, xz).toVar('surfHeight');
  const breakDepth = height.div(BREAKER_INDEX).toVar('surfBreak');
  const breakDepthLast = waveHeight(last, xz).div(BREAKER_INDEX).toVar('surfBreakLast');
  const breakDepthPrevious = waveHeight(last.sub(1), xz).div(BREAKER_INDEX).toVar('surfBreakPrev');
  // Only the fragment stage's foam field reads this one; a graph that does not
  // reference it does not emit it.
  const breakDepthOlder = waveHeight(last.sub(2), xz).div(BREAKER_INDEX).toVar('surfBreakOlder');

  const d = depth.max(0).toVar('surfDepth');
  const steep = smoothstepDown(d, breakDepth, breakDepth.mul(1.7)).toVar('surfSteep');
  // `height` is a wave height, crest to trough, and `crestProfile` spans
  // exactly one unit from trough to crest — so the amplitude is the height. It
  // used to be half of it, which drew every breaker at half the height its own
  // break depth was computed from: a 2 m wave tripping in 2.5 m of water and
  // standing a metre tall while it did.
  const shoaled = height.mul(breakDepth.div(d.max(breakDepth)).pow(0.25)).toVar('surfShoaled');
  const bore = d.mul(BREAKER_INDEX).toVar('surfBore');
  const reachFade = smoothstepDown(d, breakDepth.mul(1.5), surfReachDepth())
    .mul(surfUniforms.strength.min(1))
    .toVar('surfReachFade');
  const amplitude = d.lessThan(breakDepth).select(bore, shoaled).mul(reachFade).toVar('surfAmplitude');
  // The footprint in phase: metres times phase per metre.
  const minWidth = footprint.mul(surfPhaseGradient(d).abs()).mul(0.6).toVar('surfMinWidth');
  const profile = crestProfile(s, steep, minWidth).toVar('surfProfile');

  // --- the pitch ---------------------------------------------------------------
  // Iribarren on the deep-water steepness of this crest. `beachSlope` is floored
  // so a flat patch of reef reads as spilling instead of dividing toward zero.
  const xi = beachSlope
    .max(0.005)
    .div(height.div(surfUniforms.deepWavelength).sqrt())
    .toVar('surfXi');
  const plunging = xi.smoothstep(0.3, 0.9).toVar('surfPlunging');
  // Builds as the crest steepens toward its break, peaks at the break, and
  // relaxes to a bore's lean as the broken wave runs in. A spilling wave
  // pitches at under half the strength of a plunging one.
  const leanAmount = d
    .lessThan(breakDepth)
    .select(mix(float(0.35), float(1), d.smoothstep(breakDepth.mul(0.6), breakDepth)), steep)
    .mul(mix(float(0.45), float(1), plunging))
    .mul(reachFade)
    // Nothing moves sideways at the waterline: the swash sheet and the sand
    // meet on a contour, and a lean there would drag it.
    .mul(d.smoothstep(0.12, 0.6))
    .mul(pitch)
    .toVar('surfLeanAmount');
  // Phase per metre shoreward is negative and its reciprocal is the local
  // wavelength. The bump is `LEAN_WIDTH` of that wide, so it shoals with the
  // wave, and its height is chosen so the steepest compression it causes is
  // exactly `1 - LEAN_MAX * leanAmount`:
  //   lean = A W exp(-x^2),  x = (s - offset) / w,  W = w / |dphase/dx|
  //   d(lean)/dx = A * 2x exp(-x^2), whose extreme is A * sqrt(2/e).
  const gradPhase = surfPhaseGradient(d).toVar('surfGradPhase');
  const bumpX = s.sub(LEAN_OFFSET).div(LEAN_WIDTH).toVar('surfBumpX');
  const bump = bumpX.mul(bumpX).negate().exp().toVar('surfBump');
  const leanScale = leanAmount
    .mul(LEAN_MAX / BUMP_SLOPE_PEAK)
    .mul(float(LEAN_WIDTH).div(gradPhase.abs().max(1e-3)))
    .toVar('surfLeanScale');
  const lean = leanScale.mul(bump).toVar('surfLean');
  // d(lean)/dx = d(lean)/ds * ds/dx, with ds/dx = `gradPhase`. Floored at what
  // the construction already guarantees, against rounding.
  const compression = bump
    .mul(bumpX)
    .mul(-2 / LEAN_WIDTH)
    .mul(leanScale)
    .mul(gradPhase)
    .add(1)
    .max(1 - LEAN_MAX)
    .toVar('surfCompression');

  return {
    phase,
    s,
    eta: amplitude.mul(profile.x),
    etaPerPhase: amplitude.mul(profile.y),
    breakDepth,
    breakDepthLast,
    breakDepthPrevious,
    breakDepthOlder,
    steep,
    xi,
    lean,
    compression,
  };
}

/**
 * Period of the wind sea riding in on the swell, as a fraction of the swell's.
 * NeuralSpecOcean's beach runs a 4.75 s wind-swell train under its 7.1 s and
 * 6.55 s swells (`CoastalWaves.js`, MIT): 0.67. This is the same ratio,
 * rounded down a little so the two trains' crests do not lock into step.
 */
export const CHOP_PERIOD = 0.62;
/** Its height as a fraction of the swell's average breaker. */
const CHOP_HEIGHT = 0.4;

/** What `chopBreaker` returns. */
export interface ChopSample {
  /** Signed phase distance to the nearest wind-sea crest, negative in front. */
  s: any;
  /** Its phase; crests at integers. */
  phase: any;
  /** Depth that crest breaks in, metres. */
  breakDepth: any;
  /** d(eta)/dx shoreward, for the normal. The mesh is not displaced. */
  slope: any;
}

/**
 * The wind sea on top of the swell, inside the surf zone only.
 *
 * A beach is not hit by one wave train. The local wind sea comes in with the
 * swell at about two thirds of its period and under half its height, and it
 * breaks too: in shallower water, as a row of small spilling bores between the
 * swell's. On this island's 1:9 beach face a 6 s swell holds only two bores at
 * a time, and the second train is what makes the inner surf zone a series of
 * lines rather than one or two (foam-4; NeuralSpecOcean superposes three
 * trains for the same reason).
 *
 * Drawn by the fragment stage only, as foam and as slope: these bores are a
 * few tens of centimetres high, which the mesh's vertex spacing would alias
 * rather than draw, and their height is small next to the swell's.
 */
export function chopBreaker(depth: any, xz: any): ChopSample {
  const d = depth.max(0).toVar('chopDepth');
  const k = float(2 / (NOMINAL_SLOPE * Math.sqrt(GRAVITY))).div(surfUniforms.period.mul(CHOP_PERIOD));
  // Its own along-shore terms: the wind sea comes in at its own angle, so its
  // crests cross the swell's rather than running parallel to them.
  const along = surfNoise(xz.div(240).add(vec2(41.7, 13.9)))
    .mul(4)
    .add(surfNoise(xz.div(30).add(vec2(7.3, 29.1))).mul(0.5));
  const phase = surfUniforms.time
    .div(surfUniforms.period.mul(CHOP_PERIOD))
    .add(shoalingRoot(d, xz).mul(k))
    .add(along)
    .toVar('chopPhase');
  const s = crestDistance(phase).toVar('chopS');
  const near = s.lessThan(0).select(phase.floor().add(1), phase.floor());
  // Its own groups, five waves long, and peaks along the crest.
  const envelope = sin(near.mul((Math.PI * 2) / 5).add(0.4)).mul(0.5).add(0.5);
  const peaks = surfNoise(xz.div(17).add(vec2(hash1(near.add(3)).mul(29), hash1(near.add(7)).mul(31))))
    .mul(0.6)
    .add(0.7);
  const height = surfUniforms.height
    .mul(CHOP_HEIGHT)
    .mul(envelope.mul(0.7).add(0.55))
    .mul(peaks)
    .toVar('chopHeight');
  const breakDepth = height.div(BREAKER_INDEX).toVar('chopBreak');
  const amplitude = height
    .min(d.mul(BREAKER_INDEX))
    .mul(smoothstepDown(d, breakDepth.mul(1.3), breakDepth.mul(2.2)))
    .mul(d.smoothstep(0.08, 0.35))
    .mul(surfUniforms.strength.min(1));
  const gradient = surfPhaseGradient(d).div(CHOP_PERIOD);
  const profile = crestProfile(s, smoothstepDown(d, breakDepth, breakDepth.mul(1.7)));
  return { s, phase, breakDepth, slope: amplitude.mul(profile.y).mul(gradient) };
}

/**
 * Height of the swash sheet over a point whose ground is `-depth` above still
 * water, or far below everything when the sheet is not there.
 *
 * Nearly level rather than following the sand, and that is about the mesh, not
 * the physics. A sheet a few centimetres thick that hugs the ground has to
 * agree with the *rendered* ground to within those centimetres, and the island
 * mesh is a triangulation of the analytic field this reads — off by more than
 * that on a curved beach. The first version did hug it, and the water came out
 * of the sand in a staircase along every ring of the ocean grid. A surface that
 * barely depends on the ground height meets it along a clean contour, the way
 * still water does; tilted a quarter of the way toward the sand so it still
 * thins toward the front rather than standing as a step. The caller takes the
 * max of this and the sea, which joins the two without a seam.
 */
export function swashSheetHeight(depth: any, xz: any): any {
  const h = depth.negate();
  const phase0 = surfPhase(float(0), xz);
  const front = swashFront(fract(phase0), runup(phase0.floor(), xz)).toVar('surfFront');
  const level = front.sub(front.sub(h).max(0).mul(0.25));
  return front.greaterThan(0.005).select(level, float(-1000));
}

/**
 * Vertical run-up of breaker `n`, metres above still water. See `RUNUP_HUNT`.
 *
 * Lobed along the shore: a swash front is never a contour. Real uprush splits
 * into tongues a few to twenty metres apart (the start of beach cusps; Guza &
 * Inman 1975 put edge-wave cusp spacing at that scale on reflective beaches),
 * so each wave's reach varies by about a fifth over ~15 m, re-drawn per wave so
 * the tongues do not stand still. The ocean's sheet and the ground's wetting
 * both read this, so the lobes they draw are the same lobes.
 */
export function runup(n: any, xz: any): any {
  const lobes = surfNoise(xz.div(14).add(vec2(hash1(n).mul(37), hash1(n.add(5)).mul(23))))
    .sub(0.5)
    .mul(0.36)
    .add(1);
  return surfUniforms.runupScale.mul(waveHeight(n, xz).sqrt()).mul(lobes);
}

/**
 * Centimetres of "seepage" added to the ground height for the drying terms, so
 * where a past wave's reach and the strand it left end is a fingered edge rather
 * than the contour of a 15 m triangle — which is a straight line, and was the
 * hard diagonal seam across the foreground sand of beach-transparency-1.
 * NeuralSpecOcean (MIT, `ShoreVisual.js`) perturbs its wet/dry test by three
 * scales of grain noise for the same reason; this is that, in closed form.
 */
function seepage(xz: any): any {
  return surfNoise(xz.mul(0.24).add(vec2(3.1, 0.9)))
    .sub(0.5)
    .mul(0.07)
    .add(surfNoise(xz.mul(1.35).add(vec2(0.4, 2.7))).sub(0.5).mul(0.025));
}

/**
 * Height of the swash front, `u` periods after the bore reached the shoreline.
 *
 * `sin(pi v^SWASH_SKEW)`: the uprush takes the first 37% of the swash and the
 * backwash the remaining 63%. The uprush is a collapsing bore driven up the
 * sand; the backwash is gravity alone, slowed by the water the bed takes in.
 * Written as a power warp of time rather than an envelope on the sine so the
 * front still reaches exactly `reach`, and so the crossing times at any height
 * invert in closed form — `swashOnSand` needs them to know how long ago sand
 * was last under water.
 */
export function swashFront(u: any, reach: any): any {
  const v = u.div(SWASH_SPAN).clamp(0, 1);
  const climb = v.pow(SWASH_SKEW).mul(Math.PI).sin();
  return reach.mul(climb).mul(u.lessThan(SWASH_SPAN).select(float(1), float(0)));
}

/**
 * When, in periods after the bore, the front of a swash reaching `reach` passes
 * height `h` going up and coming back down. Exact inverses of `swashFront`.
 */
function swashCrossings(h: any, reach: any): { up: any; down: any } {
  const a = h.div(reach.max(1e-3)).clamp(0, 1).asin().div(Math.PI);
  return {
    up: a.pow(1 / SWASH_SKEW).mul(SWASH_SPAN),
    down: a.oneMinus().pow(1 / SWASH_SKEW).mul(SWASH_SPAN),
  };
}

/**
 * The swash on a patch of ground at height `h` above still water.
 *
 * @returns vec4(damp 0..1, film 0..1, under the sheet 0..1, foam amount 0..1).
 *          The foam is an amount for `FoamPattern.featherFoam`, not coverage.
 */
export function swashOnSand(h: any, xz: any): any {
  const phase = surfPhase(float(0), xz);
  const n0 = phase.floor().toVar();
  const u0 = fract(phase).toVar();

  // Current wave: under the sheet between the up- and down-crossings.
  const r0 = runup(n0, xz).toVar();
  const front0 = swashFront(u0, r0).toVar();
  const under = smoothstepDown(h.sub(front0), -0.004, 0.012).toVar();

  // Drying and stranded foam from the last three waves. Sand above a wave's
  // run-up was never reached by it; for the current wave, sand the front has
  // not reached yet is not wet by it either.
  //
  // The oldest of the three is faded out over the second half of the current
  // period. When the next bore arrives it stops being tracked, and without the
  // fade any sand it alone had reached would snap dry in one frame. At the
  // same instant the wave that takes its slot arrives at full weight, so the
  // hand-over is continuous.
  const damp: any = float(0).toVar();
  const film: any = float(0).toVar();
  const mark: any = float(0).toVar();
  // The drying terms read the ground a few centimetres up or down by the
  // seepage noise, so the edge of what each wave reached fingers. `under`
  // above does not: it has to meet the ocean's sheet where the two meshes do.
  const hSeep = h.add(seepage(xz)).toVar();
  for (let k = 0; k < 3; k++) {
    const reach = k === 0 ? r0 : runup(n0.sub(k), xz).toVar();
    const crossing = swashCrossings(hSeep, reach);
    // Soft over the last few centimetres: the limit of a swash is a thin
    // bubble line soaking in, not a ruled edge.
    const reached = smoothstepDown(hSeep, reach.sub(0.035), reach.add(0.01));
    const arrived: any = k === 0 ? u0.greaterThan(crossing.up).select(float(1), float(0)) : float(1);
    const keep: any = k === 2 ? smoothstepDownClamped(u0, 0.5, 1) : float(1);
    const weight = reached.mul(arrived).mul(keep).toVar();
    // Seconds since the backwash uncovered this sand; negative while covered.
    const age = u0.add(k).sub(crossing.down).mul(surfUniforms.period).max(0).toVar();
    damp.assign(damp.max(age.div(-DAMP_SECONDS).exp().mul(weight)));
    film.assign(film.max(age.div(-FILM_SECONDS).exp().mul(weight)));

    // The swash mark: a thin line of foam stranded at this wave's highest
    // point, from the moment the front turns back. Just inside the run-up so it
    // sits on sand that was wetted, and a few centimetres of height wide —
    // a hand's width to a stride across the sand at this slope.
    const sincePeak = u0.add(k).sub(SWASH_SPAN * SWASH_PEAK_AT).mul(surfUniforms.period).toVar();
    const line = hSeep.sub(reach.mul(0.965)).div(reach.mul(0.025).add(0.008));
    mark.assign(
      mark.max(
        line
          .mul(line)
          .negate()
          .exp()
          .mul(sincePeak.max(0).div(-MARK_SECONDS).exp())
          .mul(sincePeak.greaterThan(0).select(float(1), float(0)))
          .mul(keep),
      ),
    );
  }

  // The front while it is moving: a band of foam just behind the leading edge,
  // dense on the uprush — the remains of the bore that drove it — and thinner
  // as it slides back. Wider behind the edge than ahead of it, because the
  // foam rides on the sheet and the dry sand ahead has none.
  const onSand = u0.lessThan(SWASH_SPAN).and(front0.greaterThan(0.01)).select(float(1), float(0));
  const rising = smoothstepDown(u0, SWASH_SPAN * 0.3, SWASH_SPAN * 0.55).mul(0.25).add(0.55);
  const edge = h.sub(front0).toVar();
  const edgeWidth = edge
    .lessThan(0)
    .select(r0.mul(0.12).add(0.03), r0.mul(0.02).add(0.008))
    .toVar();
  const edgeX = edge.div(edgeWidth);
  const frontFoam = edgeX.mul(edgeX).negate().exp().mul(onSand).mul(rising);

  // Behind the front, the sheet carries what is left of the bore's foam: thick
  // as it lands, bursting over a second or so, a thin scatter after that.
  const sheetFoam = under.mul(u0.div(-0.16).exp().mul(0.55).add(0.12));

  // And the wet sand the backwash has just left keeps a few bubbles for as long
  // as it shines.
  const foam = frontFoam.max(sheetFoam).max(mark.mul(0.62)).max(film.mul(0.14).mul(under.oneMinus()));

  // `y` is the film on sand the air can see. Under the sheet the bed is not a
  // mirror — the water surface above it is, and the ocean draws that — so a
  // film there painted a second, sky-bright reflection onto the bed that the
  // water then refracted back up: the glossy grey seabed under the shallows.
  return vec4(damp.max(under), film.mul(under.oneMinus()), under, foam.mul(surfUniforms.strength));
}

/**
 * Lowest ground the swash touches, metres relative to still water: about the
 * deepest a backwash trough uncovers on these beaches. Below it the bed is
 * permanently submerged.
 */
const SWASH_FLOOR = -0.7;

/**
 * The swash at a ground fragment, behind a real branch: only the strip the
 * swash can reach does any of this work, and the branch is uniform across most
 * of a frame — all of the island above the beach skips it.
 *
 * Exported so the ground can build it once and hand the same node to its
 * colour, roughness and normal graphs; three emits a shared node once per
 * shader, where three calls would pay for the three-wave loop three times.
 */
export function swashAtGround(position: any): any {
  const result = vec4(0, 0, 0, 0).toVar();
  If(position.y.lessThan(maxRunup().add(0.1)).and(position.y.greaterThan(SWASH_FLOOR)), () => {
    // Faded out under the still-water line. Below it the "sheet" is the sea
    // itself, and the bed is drawn in its own submerged palette; the swash's
    // wet-sand darkening on top of that took a sixth off every metre of sand
    // under the shallows (measured in `shore-break` with the sea hidden), and
    // its sheet foam was painted on the bed for the water to show through.
    const fade = position.y.smoothstep(SWASH_FLOOR, SWASH_FLOOR * 0.3);
    result.assign(swashOnSand(position.y, vec2(position.x, position.z)).mul(fade));
  });
  return result;
}


/** Albedo of foam lying on sand: a little under the open sea's, it is thinner. */
const SAND_FOAM_ALBEDO = 0.82;

/**
 * Darkening of fully wetted sand, on top of the ground's own static wet band.
 *
 * Wet sand is darker because light scattered by the grains is trapped by total
 * internal reflection in the water film (Lekner & Dorf 1988); measured drops run
 * from ~10% for bright translucent grains to a half for dark ones (Zhang & Voss
 * 2006). The ground already draws a damp band across the whole swash zone, so
 * this is the *saturated* step on top of it, and a third is what separates
 * sand the last wave reached from sand it did not without the swash zone going
 * muddy.
 */
const SWASH_DARKEN = 0.32;

/**
 * Hook for the island's ground shader: sand that the swash has just left is
 * dark and glossy, and dries over seconds; the front and the marks it strands
 * are foam. Self-contained on purpose — the ground's own palette is not this
 * module's to know, so this only scales it. Must be called inside the colour
 * node's `Fn`.
 *
 * @param color    The ground's albedo, vec3.
 * @param position World position of the fragment.
 * @returns        The albedo with the swash applied.
 */
export function applySwashToGround(color: any, position: any, state: any = null): any {
  const swash = (state ?? swashAtGround(position)).toVar();
  const strength = surfUniforms.strength.min(1);
  // Reflectance falls steeply with the first moisture and then flattens
  // (Nolet et al. 2014), so damp sand is most of the way to wet: a cubic ease
  // out of the dry end.
  const dampness = swash.x.mul(strength).oneMinus().pow(3).oneMinus();
  const darkened = color.mul(dampness.mul(-SWASH_DARKEN).add(1));
  // The sheet itself adds a little of the sea's colour.
  const sheeted = mix(darkened, darkened.mul(vec3(0.8, 0.95, 1.02)), swash.z);
  // Foam, through the same pattern and feather as the sea's, so the swash edge
  // and the lace on the water beside it are one material.
  const coverage = float(0).toVar();
  If(swash.w.greaterThan(0.002), () => {
    const pattern = foamPattern(vec2(position.x, position.z), cameraPosition.distance(position));
    coverage.assign(featherFoam(swash.w, pattern));
  });
  return mix(sheeted, vec3(SAND_FOAM_ALBEDO), coverage);
}

/**
 * Roughness for the same: the draining film is a mirror and damp sand is merely
 * smoother than dry. Under the sheet the bed stays rough — see `swashOnSand`.
 *
 * NeuralSpecOcean's sand runs from 0.86 dry to 0.17 under its film
 * (`ShoreVisual.js`, MIT); these are the same ends.
 */
export function swashRoughness(base: any, position: any, state: any = null): any {
  const swash = state ?? swashAtGround(position);
  const strength = surfUniforms.strength.min(1);
  const exposed = swash.z.oneMinus();
  const damp = mix(base, float(0.5), swash.x.mul(exposed).mul(strength));
  return mix(damp, float(0.12), swash.y.mul(strength));
}

/**
 * How far the ground's relief should be flattened toward its mesh normal, 0..1.
 *
 * A film of water a few millimetres deep fills the sand's relief and presents
 * its own surface, which is the plane of the beach, so the sky it mirrors is a
 * smooth sheet rather than the bumps of the sand under it. Reflected through the
 * full relief instead, every metre-scale hollow became a lens and the wet beach
 * read as melted plastic (beach-transparency-1). NeuralSpecOcean flattens its
 * film normal 74% toward the beach plane for the same reason (`ShoreVisual.js`,
 * MIT); damp sand is drained but packed smooth, so it takes a third of that.
 */
export function swashFlatten(state: any): any {
  const strength = surfUniforms.strength.min(1);
  return state.y.mul(0.78).max(state.x.mul(state.z.oneMinus()).mul(0.3)).mul(strength).clamp(0, 1);
}
