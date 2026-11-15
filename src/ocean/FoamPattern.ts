import * as THREE from 'three/webgpu';
import { If, float, fract, mix, sin, uniform, vec2 } from 'three/tsl';
import { smoothstepDownClamped } from '../core/tslMath';

/**
 * What foam looks like, as opposed to how much of it there is.
 *
 * Every foam source in the scene — whitecaps and their trails in the open sea,
 * the breakers, the lace a bore leaves behind, the swash on the sand, the ring
 * round a rock — produces an *amount*, 0..1, and until this module each of them
 * turned that amount into white its own way: a smoothstep on a soft field here,
 * a cell threshold there, a Gaussian line somewhere else. Thresholding a soft
 * field draws its contour, and the contour of a bilinearly filtered blob is an
 * oval, so the open sea was covered in flat white lozenges and the surf in
 * bands cut out of paper.
 *
 * **The feather, after Crest.** Crest (wave-harmonic, MIT; `OceanFoam.hlsl`)
 * does it the other way round, and so does every shipped ocean that looks
 * right: the amount does not draw anything, it sets a *threshold* on a foam
 * pattern, `white = smoothstep(1 - amount, 1 - amount + feather, pattern)`. Thick
 * foam clears the threshold almost everywhere and reads as a sheet; thinning
 * foam keeps only the pattern's ridges and tears into threads; the last of it is
 * a few specks. So a patch's *edge* is no longer a contour of the amount at all —
 * it is where the pattern's ridges run out, which is ragged at every scale the
 * pattern has — and the fading Sea of Thieves foam is known for (fresh white,
 * then lace, then nothing) falls out of the decay the amount already does.
 * Stage A and stage B whitecaps (Monahan) are the same thing seen by a
 * physicist: the actively breaking crest is dense, the decaying patch behind it
 * covers ten times the area at a fraction of the density.
 *
 * **The pattern is three scales, and each is there for a distance.**
 *
 *  - *Streaks*, 3–7 m, domain-warped and stretched along the wind. Foam that
 *    has been on the water a few seconds is pulled into lanes by the wind and
 *    the Langmuir cells under it; this is the part of the texture that survives
 *    out to a kilometre, and it is what replaces the "far-field foam reads as
 *    smooth patches" item in `docs/AAA_PLAN.md`: past the range where bubbles
 *    resolve, a patch still breaks into lanes instead of fading to its average.
 *  - *Rafts*, ~1.1 m Voronoi cell rims: the polygonal bubble network every
 *    photograph of a surf zone is covered in.
 *  - *Bubbles*, ~0.45 m rims, near only, so the network is not one regular net.
 *
 * Each finer scale fades to its own mean before its features fall under a
 * pixel. That is band-limiting the *field*, not widening the edge — the
 * distinction `OceanMaterial`'s foam comments record measuring the hard way —
 * and it keeps the far-field jitter budget the gallery test defends.
 *
 * **Cost.** Evaluated only where some foam amount is non-zero, behind a real
 * branch, and gated by tier through `foamUniforms.detail`: Low draws the streaks
 * alone (four value-noise lookups), Medium adds the rafts, High and up the
 * bubbles. A uniform branch is coherent across the whole frame, so a tier that
 * skips a scale pays one compare for it.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Stretch of the pattern along the wind, as a ratio. See `FOAM_WIND_STRETCH`. */
const STREAK_STRETCH = 2.4;

/**
 * The feather: how wide the ramp from water to white is, in units of the
 * normalised pattern (see `PATTERN_LOW`).
 *
 * 0.3, Crest's 0.4 tightened: at 0.4 the lace came out grey rather than
 * white-on-water, because most of a thin patch sat inside the ramp.
 */
const FEATHER = 0.3;

/**
 * Normalisation of the raw pattern to roughly uniform on [0, 1].
 *
 * The raw blend of streaks and rims is narrow: measured with a JS replica of
 * `foamPattern` at detail 2 over a 500² grid near the island, its 10th, 50th
 * and 90th percentiles are 0.12, 0.44 and 0.72 (Medium within 0.01 of that;
 * Low, with no cells, 0.16 / 0.44 / 0.68). A threshold written in raw units
 * therefore jumped from no foam to solid over a sliver of the amount's range,
 * and the first tuning of it — by eye, against an earlier histogram — left
 * the open sea with almost no whitecaps. `(raw - 0.045) / 0.75` maps those
 * percentiles onto 0.10 / 0.53 / 0.90, so the pattern is close to uniform and
 * an amount means what it says: **the fraction of the area that is white**.
 * Re-measure if the pattern's shape or its 0.62 / 0.38 blend changes.
 */
const PATTERN_LOW = 0.045;
const PATTERN_SPAN = 0.75;

/**
 * Means of the cell fields, for the tiers and distances that do not draw them.
 *
 * Measured off the fields as built, with a JS replica of `foamCells`, the
 * noise and the seam/snip modulation, swept over a 600² grid offset a
 * kilometre from the origin like the island. The plain bubble rim averages
 * 0.41 and is bimodal — 40% of the area cell interior near 0, 20% rim near 1;
 * the rafts, with their varying seams and snips, average 0.27; the 0.68/0.32
 * blend of the two, 0.31. Standing in the mean rather than zero keeps the same
 * amount of foam the same brightness whether or not its cells are drawn, so a
 * handover — by distance or by tier — is a loss of detail and never a change of
 * coverage. Re-measure if any of those shapes change.
 */
const BUBBLE_MEAN = 0.41;
const RIM_MEAN = 0.31;

/** Shared by every material that draws foam. */
export const foamUniforms = {
  /**
   * 0: streaks only (Low). 1: plus the 1 m rafts (Medium). 2: plus the bubble
   * scale, the foam's own relief and the surf's pitching lip (High and up).
   */
  detail: uniform(2),
  /** Unit wind axis in world xz, `(cos, sin)` of the bearing. */
  windAxis: uniform(new THREE.Vector2(1, 0)),
};

/** Tier knob; see `foamUniforms.detail`. */
export function setFoamDetail(level: number): void {
  foamUniforms.detail.value = Math.max(0, Math.min(2, Math.round(level)));
}

/** Wind bearing in radians, on the `(cos, sin) -> (x, z)` convention. */
export function setFoamWind(bearingRadians: number): void {
  (foamUniforms.windAxis.value as THREE.Vector2).set(Math.cos(bearingRadians), Math.sin(bearingRadians));
}

/**
 * Integer-lattice hash to [0,1). The lattice is wrapped at 289 first: the
 * island sits a kilometre and more from the origin, and `sin` of a dot product
 * in the hundreds of thousands has lost most of its fractional bits in fp32, so
 * the hash bands. A 289-cell period is hundreds of metres at every scale used
 * here, far past where any repeat could be seen.
 */
function latticeHash(c: any, k: any): any {
  const wrapped = c.sub(c.div(289).floor().mul(289));
  return fract(sin(wrapped.dot(k)).mul(43758.5453));
}

/** Value noise in [0,1], cubic fade, arithmetic only. */
export function foamNoise(p: any): any {
  const i = p.floor();
  const f = fract(p);
  const u: any = f.mul(f).mul(f.mul(-2).add(3));
  const k = vec2(127.1, 311.7);
  const h = (o: any) => latticeHash(i.add(o), k);
  return (mix as any)((mix as any)(h(vec2(0, 0)), h(vec2(1, 0)), u.x), (mix as any)(h(vec2(0, 1)), h(vec2(1, 1)), u.x), u.y);
}

/**
 * Cellular foam: `vec2(F1, F2 - F1)` of a jittered grid. `F2 - F1` is zero on
 * the boundaries between cells, which is the shape of the bubble network a
 * surf zone is left covered in — polygons of clear water outlined in foam.
 * Nine cells, arithmetic only.
 */
export function foamCells(p: any): any {
  const cell = p.floor().toVar();
  const local = fract(p).toVar();
  const f1 = float(8).toVar();
  const f2 = float(8).toVar();
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const o = vec2(i, j);
      const c = cell.add(o);
      const jitter = vec2(latticeHash(c, vec2(127.1, 311.7)), latticeHash(c, vec2(269.5, 183.3)))
        .mul(0.8)
        .add(0.1);
      const dist = o.add(jitter).sub(local).length();
      f2.assign(dist.max(f1).min(f2));
      f1.assign(f1.min(dist));
    }
  }
  return vec2(f1, f2.sub(f1));
}

/**
 * 1 on a cell boundary, falling to 0 up to `width` of a cell inside it.
 *
 * The width is varied by the caller. At one width the rims are a uniform
 * honeycomb — read from twenty metres, a fishnet laid on the water — where real
 * bubble rafts have thick seams and hair-thin ones side by side.
 */
function cellRim(p: any, width: any = float(0.38)): any {
  return smoothstepDownClamped(foamCells(p).y, 0.02, width);
}

/**
 * The foam pattern at world point `p` (metres, xz), in [0,1]. Ridges are where
 * thin foam survives longest. Must be called inside a TSL `Fn`.
 *
 * @param p            World xz, already moved by whatever advects the foam.
 * @param viewDistance Metres from the eye, for the per-scale fades.
 */
export function foamPattern(p: any, viewDistance: any): any {
  const along = vec2(foamUniforms.windAxis.x, foamUniforms.windAxis.y);
  const across = vec2(foamUniforms.windAxis.y.negate(), foamUniforms.windAxis.x);
  // The wind's own frame, stretched along it. A rotation by basis vectors —
  // two dot products — rather than a matrix.
  const q = vec2(p.dot(along).div(STREAK_STRETCH), p.dot(across)).toVar('foamQ');

  // Domain warp, metres. Without it the streaks are value-noise blobs in a
  // row; warped, they bend and pinch like foam being worked by the water.
  const warpBase = q.div(13).toVar('foamWarpBase');
  const warp = vec2(foamNoise(warpBase), foamNoise(warpBase.add(vec2(5.2, 1.3))))
    .sub(0.5)
    .mul(7)
    .toVar('foamWarp');
  const w = q.add(warp).toVar('foamW');

  // Streaks: two octaves; the finer fades out before it shimmers. Contrast is
  // pushed so the lanes have gaps between them rather than a gentle swell.
  const fine = mix(float(0.5), foamNoise(w.div(2.7).add(vec2(3.1, 7.7))), smoothstepDownClamped(viewDistance, 260, 650));
  const streak = foamNoise(w.div(6.5)).mul(0.65).add(fine.mul(0.35)).smoothstep(0.2, 0.8).toVar('foamStreak');

  // Rafts and bubbles. Each fades to the rim mean, so coverage is the same at
  // every tier and distance and only the structure changes.
  const rim = float(RIM_MEAN).toVar('foamRim');
  const raftNear = smoothstepDownClamped(viewDistance, 120, 360).toVar('foamRaftNear');
  If(foamUniforms.detail.greaterThan(0.5).and(raftNear.greaterThan(0.001)), () => {
    // The cells are drawn in the unstretched-by-warp frame scaled back toward
    // isotropic, so a raft is a polygon and not a sliver: `q` divided the
    // along-wind axis by the stretch, so multiply a little of it back.
    const cellUv = vec2(w.x.mul(STREAK_STRETCH * 0.55), w.y).toVar('foamCellUv');
    // Seams thicken where the streaks are dense and thin out between them, and
    // a second noise snips the thinnest so the net tears instead of closing.
    const seam = mix(float(0.16), float(0.46), streak);
    const snip = foamNoise(cellUv.div(1.9).add(vec2(2.2, 9.4))).smoothstep(0.2, 0.6).mul(0.6).add(0.4);
    const rafts = cellRim(cellUv.div(1.15), seam).mul(snip).toVar('foamRafts');
    const bubbles = float(BUBBLE_MEAN).toVar('foamBubbles');
    const bubbleNear = smoothstepDownClamped(viewDistance, 50, 170).toVar('foamBubbleNear');
    If(foamUniforms.detail.greaterThan(1.5).and(bubbleNear.greaterThan(0.001)), () => {
      bubbles.assign(mix(float(BUBBLE_MEAN), cellRim(cellUv.div(0.48).add(vec2(5.3, 1.7))), bubbleNear));
    });
    rim.assign(mix(float(RIM_MEAN), rafts.mul(0.68).add(bubbles.mul(0.32)), raftNear));
  });

  return streak.mul(0.62).add(rim.mul(0.38)).sub(PATTERN_LOW).div(PATTERN_SPAN);
}

/**
 * White coverage for a foam `amount` over a `pattern` value. See the header.
 * 0 is exactly no foam; 1 is solid.
 *
 * Over the normalised pattern the threshold runs from 1 (nothing clears it) at
 * amount 0 to `-FEATHER` (everything is past the ramp) at amount 1, linearly,
 * so the white fraction tracks the amount: 0.8 is white with the cells showing
 * through, 0.5 is lace, 0.2 is threads.
 */
export function featherFoam(amount: any, pattern: any): any {
  const threshold = amount.clamp(0, 1).mul(-(1 + FEATHER)).add(1).toVar();
  return pattern.smoothstep(threshold, threshold.add(FEATHER));
}

/**
 * Large-scale unevenness of foam density, 0..1, at world point `p`.
 *
 * A whitecap's patch is not evenly dense: the crest broke harder in some places
 * than others, and the wind has already begun pulling the patch into lanes. The
 * foam buffer cannot say so — it holds one smooth density per 0.4 m texel, and
 * where a fold saturates it the patch is a uniform plateau with an oval rim.
 * Scaling the amount by this before the feather turns the plateau into rafts
 * and lanes and tears the rim, at 5–11 m, the scale the eye reads a patch's
 * shape at. Wind-stretched like the pattern, and coarse enough to stay resolved
 * to a kilometre.
 */
export function foamPatchiness(p: any): any {
  const along = vec2(foamUniforms.windAxis.x, foamUniforms.windAxis.y);
  const across = vec2(foamUniforms.windAxis.y.negate(), foamUniforms.windAxis.x);
  const q = vec2(p.dot(along).div(STREAK_STRETCH * 1.3), p.dot(across));
  // The 26 m octave breaks up the buffer's saturated plates at a few hundred
  // metres. There the two finer octaves are two pixels wide or less, and a
  // fold region would otherwise read as one flat white oval.
  return foamNoise(q.div(26).add(vec2(4.1, 7.6)))
    .mul(0.4)
    .add(foamNoise(q.div(11).add(vec2(9.7, 3.3))).mul(0.4))
    .add(foamNoise(q.div(4.7).add(vec2(1.9, 6.1))).mul(0.2))
    .smoothstep(0.28, 0.72);
}
