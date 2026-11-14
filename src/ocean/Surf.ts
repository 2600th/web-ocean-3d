import { If, float, fract, mix, sin, uniform, vec2, vec3 } from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';

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
 * waterline, steepening offshore).
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
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

const GRAVITY = 9.81;

/** Beach slope the phase assumes, rise over run. See the header. */
const NOMINAL_SLOPE = 0.1;

/** McCowan's breaker index: a wave breaks when H reaches this times the depth. */
export const BREAKER_INDEX = 0.78;

/** Waves in one set: the group envelope's period, in breakers. */
const SET_LENGTH = 7;

/** Vertical run-up of the swash, as a fraction of the breaker height. */
const RUNUP_FRACTION = 0.42;

/** Fraction of a wave period the swash front spends on the sand. */
const SWASH_SPAN = 0.82;

/** Seconds for swash-wetted sand to dry back to the damp it started at, e-fold. */
const DRY_SECONDS = 7;

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
  surfUniforms.period.value = Math.max(2, period);
  surfUniforms.height.value = Math.max(0.2, height);
  surfUniforms.strength.value = Math.max(0, strength);
  surfUniforms.active.value = active ? 1 : 0;
}

/** Deepest water, metres, in which anything here is non-zero. */
export function surfReachDepth(): any {
  // The biggest wave of a set is ~1.3 times the average.
  return surfUniforms.height.mul((1.3 / BREAKER_INDEX) * SURF_REACH);
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
 */
function alongShorePhase(xz: any): any {
  return surfNoise(xz.div(320))
    .mul(3)
    .add(surfNoise(xz.div(45).add(vec2(17.3, 5.1))).mul(0.35));
}

/** Breaker phase at a point; crests are at integer values. See the header. */
export function surfPhase(depth: any, xz: any): any {
  const k = float(2 / (NOMINAL_SLOPE * Math.sqrt(GRAVITY))).div(surfUniforms.period);
  return surfUniforms.time
    .div(surfUniforms.period)
    .add(depth.max(0).sqrt().mul(k))
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
  return surfUniforms.height
    .mul(envelope.pow(1.5).mul(0.7).add(0.5))
    .mul(jitter)
    .mul(along);
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
export function crestProfile(s: any, steep: any): any {
  const width = s.lessThan(0).select(mix(float(0.15), float(0.055), steep), float(0.24));
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
  /** Depth the last crest to pass broke in, and the one before it. */
  breakDepthLast: any;
  breakDepthPrevious: any;
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
 */
export function breaker(depth: any, xz: any): BreakerSample {
  const phase = surfPhase(depth, xz).toVar('surfPhase');
  const last = phase.floor().toVar('surfLast');
  const s = crestDistance(phase).toVar('surfS');
  // On the front face the nearest crest is the *next* one to arrive.
  const near = s.lessThan(0).select(last.add(1), last).toVar('surfNear');
  const height = waveHeight(near, xz).toVar('surfHeight');
  const breakDepth = height.div(BREAKER_INDEX).toVar('surfBreak');
  const breakDepthLast = waveHeight(last, xz).div(BREAKER_INDEX).toVar('surfBreakLast');
  const breakDepthPrevious = waveHeight(last.sub(1), xz).div(BREAKER_INDEX).toVar('surfBreakPrev');

  const d = depth.max(0).toVar('surfDepth');
  const steep = smoothstepDown(d, breakDepth, breakDepth.mul(1.7)).toVar('surfSteep');
  const shoaled = height
    .mul(0.5)
    .mul(breakDepth.div(d.max(breakDepth)).pow(0.25))
    .toVar('surfShoaled');
  const bore = d.mul(BREAKER_INDEX * 0.5).toVar('surfBore');
  const amplitude = d
    .lessThan(breakDepth)
    .select(bore, shoaled)
    .mul(smoothstepDown(d, breakDepth.mul(1.5), surfReachDepth()))
    .mul(surfUniforms.strength.min(1))
    .toVar('surfAmplitude');
  const profile = crestProfile(s, steep).toVar('surfProfile');
  return {
    phase,
    s,
    eta: amplitude.mul(profile.x),
    etaPerPhase: amplitude.mul(profile.y),
    breakDepth,
    breakDepthLast,
    breakDepthPrevious,
  };
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
 * Cellular foam: `vec2(F1, F2 - F1)` of a jittered grid. `F2 - F1` is zero on
 * the boundaries between cells, which is the shape of the bubble network a
 * surf zone is left covered in — polygons of clear water outlined in foam.
 * Nine cells, arithmetic only.
 */
export function foamCells(p: any): any {
  const cell = p.floor().toVar('foamCell');
  const local = fract(p).toVar('foamLocal');
  const f1 = float(8).toVar('foamF1');
  const f2 = float(8).toVar('foamF2');
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const o = vec2(i, j);
      const c = cell.add(o);
      const jitter = vec2(
        fract(sin(c.dot(vec2(127.1, 311.7))).mul(43758.5453)),
        fract(sin(c.dot(vec2(269.5, 183.3))).mul(43758.5453)),
      )
        .mul(0.8)
        .add(0.1);
      const dist = o.add(jitter).sub(local).length();
      f2.assign(dist.max(f1).min(f2));
      f1.assign(f1.min(dist));
    }
  }
  return vec2(f1, f2.sub(f1));
}

/** Vertical run-up of breaker `n`, metres above still water. */
export function runup(n: any, xz: any): any {
  return waveHeight(n, xz).mul(RUNUP_FRACTION);
}

/** Height of the swash front, `u` periods after the bore reached the shoreline. */
export function swashFront(u: any, reach: any): any {
  const v = u.div(SWASH_SPAN).clamp(0, 1);
  // Climbs fast and falls back slowly: the uprush is driven, the backwash is
  // gravity alone.
  const climb = v.mul(Math.PI).sin().mul(v.oneMinus().mul(0.25).add(0.75));
  return reach.mul(climb).mul(u.lessThan(SWASH_SPAN).select(float(1), float(0)));
}

/**
 * The swash on a patch of ground at height `h` above still water.
 *
 * @returns vec3(wet 0..1, under the sheet 0..1, foam at the front 0..1)
 */
export function swashOnSand(h: any, xz: any): any {
  const phase = surfPhase(float(0), xz);
  const n0 = phase.floor();
  const u0 = fract(phase);

  // Current wave: under the sheet between the up- and down-crossings.
  const r0 = runup(n0, xz);
  const front0 = swashFront(u0, r0);
  const under = smoothstepDown(h.sub(front0), -0.004, 0.012);

  // Wetness from the last three waves. For each, the down-crossing of the
  // backwash at height `h` is `SWASH_SPAN * (1 - asin(h / R) / pi)`, with the
  // small skew of `swashFront` ignored — it moves the answer by well under a
  // second. Sand above a wave's run-up was never reached by it.
  const wet: any = float(0).toVar();
  for (let k = 0; k < 3; k++) {
    const n = n0.sub(k);
    const reach = k === 0 ? r0 : runup(n, xz);
    const ratio = h.div(reach.max(1e-3)).clamp(0, 1);
    const down: any = float(SWASH_SPAN).mul(ratio.asin().div(Math.PI).oneMinus());
    const age = u0.add(k).sub(down).mul(surfUniforms.period);
    const reached = h.lessThan(reach).select(float(1), float(0));
    const dried: any = age.max(0).div(DRY_SECONDS).negate().exp();
    // For the current wave, sand the front has not reached yet is not wet by it.
    const up: any = float(SWASH_SPAN).mul(ratio.asin().div(Math.PI));
    const arrived: any = k === 0 ? u0.greaterThan(up).select(float(1), float(0)) : float(1);
    wet.assign(wet.max(dried.mul(reached).mul(arrived)));
  }

  // A line of white at the front while it is moving: bright on the uprush,
  // thinner and lacier as it slides back.
  const onSand = u0.lessThan(SWASH_SPAN).select(float(1), float(0));
  const rising = smoothstepDown(u0, SWASH_SPAN * 0.35, SWASH_SPAN * 0.6).mul(0.5).add(0.5);
  const edge = h
    .sub(front0)
    .div(r0.mul(0.05).add(0.02))
    .toVar();
  const foam = edge.mul(edge).negate().exp().mul(onSand).mul(rising).mul(front0.greaterThan(0.01).select(float(1), float(0)));

  return vec3(wet.max(under), under, foam.mul(surfUniforms.strength));
}

/**
 * The swash at a ground fragment, behind a real branch: only the strip the
 * swash can reach does any of this work, and the branch is uniform across most
 * of a frame — all of the island above the beach skips it.
 */
function swashAtGround(position: any): any {
  const result = vec3(0, 0, 0).toVar();
  If(position.y.lessThan(surfUniforms.height.mul(RUNUP_FRACTION * 1.6).add(0.1)), () => {
    result.assign(swashOnSand(position.y, vec2(position.x, position.z)));
  });
  return result;
}

/**
 * Hook for the island's ground shader: sand that the swash has just left is
 * dark and glossy, and dries over seconds. Self-contained on purpose — the
 * ground's own palette is not this module's to know, so this only scales it.
 * Must be called inside the colour node's `Fn`.
 *
 * @param color    The ground's albedo, vec3.
 * @param position World position of the fragment.
 * @returns        The albedo with the swash applied.
 */
export function applySwashToGround(color: any, position: any): any {
  const swash = swashAtGround(position);
  const wet = swash.x.mul(surfUniforms.strength);
  // A water film darkens sand to about half its dry albedo; the sheet itself
  // adds a little of the sea's colour; the front is foam.
  const darkened = color.mul(wet.mul(-0.2).add(1));
  const sheeted = mix(darkened, darkened.mul(vec3(0.8, 0.95, 1.02)), swash.y);
  return mix(sheeted, vec3(0.62, 0.66, 0.66), swash.z.mul(0.85));
}

/** Roughness for the same: wet sand is glossy and the sheet is a mirror. */
export function swashRoughness(base: any, position: any): any {
  const swash = swashAtGround(position);
  const wet = swash.x.mul(surfUniforms.strength);
  return mix(mix(base, float(0.5), wet), float(0.18), swash.y.mul(surfUniforms.strength));
}
