import * as THREE from 'three/webgpu';
import { If, cos, float, fract, mix, sin, uniform, vec2 } from 'three/tsl';
import { smoothstepDownClamped } from '../core/tslMath';

/**
 * What foam looks like, as opposed to how much of it there is.
 *
 * Every foam source in the scene — whitecaps and their trails in the open sea,
 * the breakers, the lace a bore leaves behind, the swash on the sand, the ring
 * round a floating cask — produces an *amount*, 0..1. None of them draws
 * anything: the amount sets a *threshold* on one foam pattern, so thick foam
 * clears it almost everywhere and thinning foam keeps only the pattern's densest
 * parts. A patch's edge is then where the pattern runs out, ragged at every scale
 * the pattern has, rather than a contour of the amount (Crest, `OceanFoam.hlsl`,
 * does the same; so does web-ocean below).
 *
 * **The pattern is ported from Noniv/web-ocean** (MIT, `js/wgsl/sim.js`,
 * `FOAM_TEX`, and `foamPattern` in `js/wgsl/ocean.js`). web-ocean bakes it into a
 * 512² texture; it is evaluated here in the shader, in world space, because the
 * water's fragment stage already binds the WebGPU default maximum of 16 sampled
 * textures (`OceanMaterial`, where a 17th failed pipeline creation) and so has
 * no slot for one. Its structure, kept:
 *
 *  - *sheets with ragged holes*: a domain-warped fBm through a narrow
 *    smoothstep, 0.4 m features (`m1`), broken again at 15 cm (`m2`);
 *  - *round bubbles*, "swiss cheese": a jittered grid of discs with per-cell
 *    radius, 0 inside a bubble and 1 in the foam between (`foamBubbles`), at
 *    15 cm, 5 cm and 2 cm;
 *  - a 1 m density fBm under all of it.
 *
 * **What it replaces, and why.** The previous pattern was Voronoi *rims*:
 * `F2 - F1` of a jittered grid, 1 on the cell boundaries. Thresholded, a rim
 * field keeps its network of lines, and that network is what the critic read
 * as a "dendritic frost-crack web" (waves-3) and a "fishnet". Bubbles are the
 * complement: holes in a sheet. Thresholded, a sheet with holes keeps blobs of
 * foam that tear into lace as the amount falls, which is what thin foam on the
 * sea looks like.
 *
 * **Band-limited by distance and tier.** web-ocean's texture is mip-mapped, so
 * each scale averages away when it falls under a pixel. A shader-side pattern
 * has no mips, so each scale fades to its measured mean before that happens
 * (see `MEAN_*`), and the coarser scales keep the coverage. `foamUniforms.detail`
 * gates the cost by tier: Low draws the sheets and density only, Medium adds the
 * 15 cm scale, High and up the 5 cm and 2 cm bubbles. Evaluated only where some
 * foam amount is non-zero, behind a real branch, by every caller.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Stretch of the patchiness field along the wind, as a ratio. 1.5: at 3.1 the
 * far-field rafts read as a ladder of parallel bands (waves-4).
 */
const STREAK_STRETCH = 1.5;

/**
 * Normalisation of the raw pattern onto roughly uniform [0, 1], for
 * `featherFoam`.
 *
 * Measured with a JS replica of `foamPattern` (every scale drawn) over a 600²
 * grid at 0.17 m spacing a kilometre from the origin: the 10th, 25th, 50th,
 * 75th and 90th percentiles are 0.33, 0.40, 0.57, 0.70 and 0.79, and
 * `(raw - 0.29) / 0.53` maps them onto 0.08, 0.20, 0.53, 0.77 and 0.93. So an
 * amount passed to `featherFoam` still means what it always has: the fraction of
 * the area that is white. Re-measure if the pattern's shape changes.
 */
const PATTERN_LOW = 0.29;
const PATTERN_SPAN = 0.53;

/** The feather's ramp width, in normalised pattern units. */
const FEATHER = 0.3;

/**
 * Means of the scales that fade with distance or tier, measured with the same
 * replica: `m2` 0.673, the 15 cm bubbles 0.751, and the whole fine (G) term
 * `0.45 + 0.55 * g3 * (0.7 + 0.3 * g4)` 0.836. `m1` 0.619. Standing in the mean
 * rather than zero keeps the coverage of a given amount the same whether or not
 * its fine structure is drawn, so a hand-over by distance or tier is a loss of
 * detail and never a change of brightness.
 */
const MEAN_M1 = 0.619;
const MEAN_M2 = 0.673;
const MEAN_B3 = 0.751;
const MEAN_FINE = 0.836;

/**
 * Seabed depths, metres, over which open-sea whitecaps fade out toward the
 * shore: none at or above `SHELTER_DEPTH_FULL` (the lagoon floor is 5.5 m),
 * all of them past `SHELTER_DEPTH_NONE`. Both well inside the island's 17 m
 * plateau, so no open-water view changes. Shared by `Wake`'s whitecap deposit and
 * `OceanMaterial`'s instantaneous crest mask, which have to agree.
 */
export const SHELTER_DEPTH_FULL = 6.5;
export const SHELTER_DEPTH_NONE = 13;

/** Shared by every material that draws foam. */
export const foamUniforms = {
  /**
   * 0: sheets and density only (Low). 1: plus the 15 cm scale (Medium). 2: plus
   * the 5 cm and 2 cm bubbles, the foam's own relief and the surf's pitching
   * lip (High and up).
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
 * the hash bands. A 289-cell period is tens of metres even at the finest scale
 * used here, far past where a repeat could be seen in foam.
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
 * Gradient noise, about [-0.7, 0.7], quintic fade: web-ocean's `gnoise` /
 * `pnoise` (`js/wgsl/common.js`) without the period, with our lattice hash
 * giving the gradient angle. web-ocean's smoothstep thresholds below are written
 * for this distribution, which is why it is gradient and not value noise.
 */
function gradNoise(p: any): any {
  const i = p.floor().toVar();
  const f = fract(p).toVar();
  const u: any = f.mul(f).mul(f).mul(f.mul(f.mul(6).sub(15)).add(10));
  const k = vec2(127.1, 311.7);
  const g = (o: any, d: any) => {
    const a = latticeHash(i.add(o), k).mul(Math.PI * 2);
    return vec2(cos(a), sin(a)).dot(d);
  };
  const a = g(vec2(0, 0), f);
  const b = g(vec2(1, 0), f.sub(vec2(1, 0)));
  const c = g(vec2(0, 1), f.sub(vec2(0, 1)));
  const d = g(vec2(1, 1), f.sub(vec2(1, 1)));
  return (mix as any)((mix as any)(a, b, u.x), (mix as any)(c, d, u.x), u.y);
}

/** fBm of `gradNoise`, web-ocean's `pfbm` without the period. */
function gradFbm(p: any, octaves: number): any {
  let sum: any = float(0);
  let amplitude = 0.5;
  let q: any = p;
  for (let o = 0; o < octaves; o++) {
    sum = sum.add(gradNoise(q).mul(amplitude));
    q = q.mul(2);
    amplitude *= 0.5;
  }
  return sum;
}

/**
 * Cellular foam: `vec2(F1, F2 - F1)` of a jittered grid. Kept for the swash
 * (`Surf` re-exports it); the foam pattern itself no longer draws cell rims.
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
 * "Swiss cheese" foam, web-ocean's `bubbles`: 1 in the foam, 0 inside a bubble.
 * A jittered grid of discs, one per cell, each with its own radius between
 * `rMin` and `rMax` cells, the grid bent by a gradient noise at twice the cell
 * size so the bubbles do not sit in rows. web-ocean warps with a three-octave
 * fBm; one octave is indistinguishable at these sizes and a third of the cost.
 *
 * @param p    Position in cells.
 * @param salt Offsets the hash, so two calls at the same scale differ.
 */
function foamBubbles(p: any, salt: number, rMin: number, rMax: number): any {
  return bubbleDistance(p, salt, rMin, rMax).smoothstep(0.8, 1.25);
}

/**
 * Distance to the nearest bubble's centre in units of its radius: below 1
 * inside a bubble. `foamBubbles` thresholds it at 1.
 */
function bubbleDistance(p: any, salt: number, rMin: number, rMax: number): any {
  const half = p.mul(0.5);
  const warp = vec2(gradNoise(half.add(vec2(salt, 0))), gradNoise(half.add(vec2(17 + salt, 17))));
  const q = p.add(warp.mul(0.9)).toVar();
  const cell = q.floor().toVar();
  const local = fract(q).toVar();
  const v = float(9).toVar();
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const o = vec2(i, j);
      const c = cell.add(o).add(vec2(salt, 0));
      const jitter = vec2(latticeHash(c, vec2(127.1, 311.7)), latticeHash(c, vec2(269.5, 183.3)));
      const radius = mix(float(rMin), float(rMax), latticeHash(c, vec2(419.2, 371.9)));
      const r = o.add(jitter.mul(0.9)).add(0.05).sub(local);
      v.assign(v.min(r.length().div(radius)));
    }
  }
  return v;
}

/**
 * The filament field's frame: noise cells `FILAMENT_ACROSS` metres across the
 * axis and a stretch times that along it. Gradient noise has little structure
 * past a cell along a stretch, so what survives the threshold is drawn out by
 * about 0.8 of the nominal ratio: `FRESH_STRETCH` 2.8 reads as about 2.2:1,
 * `AGED_STRETCH` 4.5 as about 3.5:1.
 *
 * Round 8's single 5:1 frame (about 4:1 on screen) turned whole whitecaps into
 * thin brush strokes laid along the wind, which an earlier critic had already
 * marked down as "brushed streaks". A breaking whitecap is a dense patch with a
 * torn edge; only the foam it leaves behind is combed out. So fresh foam is
 * read in the short frame, and the long one takes over only as the amount
 * falls through the decay (`AGED_FROM`, `AGED_TO`).
 */
const FRESH_STRETCH = 2.8;
const AGED_STRETCH = 4.5;
const AGED_FROM = 0.2;
const AGED_TO = 0.65;
const FILAMENT_ACROSS = 1;

/**
 * Standard deviation of `filamentNoise`, measured with a JS replica over a
 * 700² grid (0.2404). The field is near enough normal that `|z| < t sd` covers
 * `erf(t / sqrt 2)` of the plane to within 0.02 at every threshold used here:
 * the replica's figures for t = 0.4, 0.8, 1.25 and 1.7 are 0.31, 0.58, 0.78
 * and 0.90.
 */
const FILAMENT_SD = 0.2404;

/**
 * The finer octave groups, as frequency ratio and weight: 0.34 m strands that
 * tear the metre ones (Medium and up), 12 cm fibres in those and 4 cm lace at
 * the camera (High and up). Read in a frame stretched only `DETAIL_STRETCH`
 * along the axis: the strands' shape is the coarse octave's business, and a
 * stretched fine grain was half of round 10's "near foam smeared and
 * stretched" (waves-4).
 *
 * Each fades out by the pixel's footprint on the water, not by distance, when
 * its cells fall under about six pixels: a footprint is what aliasing depends
 * on, and at a grazing angle it is several times what the distance suggests.
 */
const FILAMENT_MID = 2.9;
const FILAMENT_MID_WEIGHT = 0.95;
const FILAMENT_FINE = 8.4;
const FILAMENT_FINE_WEIGHT = 0.5;
const FILAMENT_CM = 25;
const FILAMENT_CM_WEIGHT = 0.4;
const DETAIL_STRETCH = 1.3;

/**
 * Gradient of the normalised field per metre with each group fully on,
 * measured on the replica (rms): 4 for the metre octave alone, and about 4, 6
 * and 20 more as the 0.34 m, 12 cm and 4 cm groups come in. The band's edge is
 * one pixel's worth of that, so it is as sharp as the screen allows at every
 * range: round 8's fixed edge, a fifth of the spread, was 6–11 cm of blur, and
 * at the camera that is twenty pixels of it.
 */
const GRAD_COARSE = 4;
const GRAD_MID = 4;
const GRAD_FINE = 6;
const GRAD_CM = 20;

/**
 * Warp of the strand frame, in noise cells, by a noise about two cells along
 * and two and a half across. Its gradient squeezes and stretches the strands
 * and the gaps between them by up to about half either way, so neither comes in
 * one length: the wake's gaps had been dashes of a single size (round 8's
 * boat-chase). Half as much across, which varies their width the same way.
 */
const FILAMENT_WARP = 0.6;

/**
 * Width variation along a strand: the band's half-width is scaled by
 * `0.65 + 0.7 n` for a value noise `n` at about 0.6 by 1.1 cells, so a strand
 * swells and pinches rather than being one stroke of a brush.
 */
const STRAND_WIDTH_LOW = 0.65;
const STRAND_WIDTH_SPAN = 0.7;

/** Two octaves of gradient noise, the filament field's raw value. */
function filamentNoise(q: any): any {
  return gradNoise(q).add(gradNoise(q.mul(2.03).add(vec2(5.1, 1.7))).mul(0.5));
}

/**
 * `P(|z| < t)` for a unit normal `z`: `erf(t / sqrt 2)`, by Winitzki's
 * approximation (good to 1e-3), so the far field's coverage follows the
 * threshold the near field draws.
 */
function bandCoverage(t: any): any {
  const x2 = t.mul(t).mul(0.5);
  const a = 0.147;
  return x2
    .mul(x2.mul(a).add(4 / Math.PI))
    .div(x2.mul(a).add(1))
    .negate()
    .exp()
    .oneMinus()
    .max(0)
    .sqrt();
}

/**
 * Foam lace: 1 in the foam, 0 in the gaps, at world point `p` for a foam
 * `amount` 0..1. Every open-sea foam and the wake are drawn through it.
 *
 * **Filaments, not holes.** Until round 8 this was web-ocean's swiss-cheese
 * bubble field (`bubbles` in `FOAM_TEX`) at three scales: holes punched into a
 * film, widening as it aged. However their sizes and spacing were varied,
 * round holes of a similar size in a white sheet are what the critic kept
 * reading as "leopard spots" (waves-4, foam-1, foam-3, boat-sea-1); the
 * smallest, 7 cm, were the worst, since every one of them came out the same
 * size on screen. A whitecap does not die that way. It is a dense white patch
 * with a torn edge, trailing downwind into decaying lace and a few streaks;
 * a wake's foam is drawn out along the ship's track.
 *
 * So the foam is where a gradient-noise field, read in a frame stretched along
 * `axis` and bent by a 7–14 m meander, is near zero: `|z| < t`. The zero set of
 * such a field is a web of strands running along the axis. The band's
 * half-width `t` grows with the amount, steeply at the top: fresh foam is
 * nearly solid, with only the torn edge and a few small gaps; aged foam is
 * lace; the last of it is strands. The frame's stretch grows as the amount
 * falls (`FRESH_STRETCH`, `AGED_STRETCH`), so only the thinning foam is combed
 * out, and the strands' length and width vary (`FILAMENT_WARP`,
 * `STRAND_WIDTH_*`).
 *
 * Three octave groups, 1 m, 0.34 m and 12 cm across the axis, summed into one
 * field and divided by its spread (`FILAMENT_SD`), so the finer groups tear
 * the outlines without changing how much water they cover. Each finer group
 * fades out with distance before its strands fall under a pixel, and is
 * tier-gated; the whole field fades to its analytic coverage (`bandCoverage`)
 * far off. Must be called inside a TSL `Fn`.
 *
 * @param axis      Unit xz direction the filaments run along; the wind by default.
 * @param footprint World metres per pixel at `p` (the larger screen axis). When
 *                  given, the detail is drawn to the pixel: the finer groups
 *                  are read nearly unstretched, fade by footprint, a 4 cm group
 *                  is added at the camera, and the strand edge is one pixel
 *                  wide. Without it, round 8b's distance-faded lace with a fixed
 *                  edge, which the surf's froth (`surfLace`) is tuned on.
 */
export function foamLace(p: any, amount: any, viewDistance: any, axis?: any, footprint?: any): any {
  const a = amount.clamp(0, 1).toVar('foamLaceAmount');
  // Half the larger axis: the band edge and the octave fades want the pixel's
  // size, and the larger axis alone overstates it by the foreshortening.
  const sharp = footprint !== undefined;
  const pixel = (footprint ?? viewDistance.mul(0.0011)).mul(0.5).toVar('foamLacePixel');
  const along = axis ?? vec2(foamUniforms.windAxis.x, foamUniforms.windAxis.y);
  const across = vec2(along.y.negate(), along.x);
  const frame = vec2(p.dot(along), p.dot(across)).toVar('foamLaceFrame');
  // The band's half-width, in units of the field's spread: the strands alone
  // (about half the water) at 0, nearly solid from about 0.85. Varied by ±20%
  // over 8–15 m, so one raft is not one texture: in places it has already
  // gone to strands, in others it still holds together.
  const region = foamNoise(p.mul(1 / 9).add(vec2(7.3, 2.1))).toVar('foamLaceRegion');
  const t0 = a
    .mul(1.35)
    .add(a.mul(a).mul(a).mul(0.6))
    .add(0.65)
    .mul(region.mul(0.4).add(0.8))
    .toVar('foamLaceBand');
  // The width variation is concave in coverage, so the mean is taken at the
  // two ends of its range rather than at its middle.
  const mean = bandCoverage(t0.mul(0.8)).add(bandCoverage(t0.mul(1.2))).mul(0.5).toVar('foamLaceMean');
  const strands = mean.toVar('foamLaceStrands');
  const near = smoothstepDownClamped(viewDistance, 60, 140).toVar('foamLaceNear');
  If(near.greaterThan(0.001), () => {
    // A strand meanders, as a streak follows the Langmuir cells rather than
    // a ruling; without it the stretched field reads as hatching.
    const bend = gradNoise(frame.mul(vec2(1 / 14, 1 / 7)).add(vec2(3.1, 7.7))).mul(1.6);
    const qy = frame.y.add(bend).div(FILAMENT_ACROSS).toVar('foamLaceQy');
    const qx = frame.x.div(FILAMENT_ACROSS * FRESH_STRETCH).toVar('foamLaceQx');
    const warp = vec2(
      gradNoise(vec2(qx.mul(0.55).add(4.3), qy.mul(0.4).add(1.9))).mul(FILAMENT_WARP),
      gradNoise(vec2(qx.mul(0.55).add(8.1), qy.mul(0.4).add(6.7))).mul(FILAMENT_WARP * 0.5),
    ).toVar('foamLaceWarp');
    const q = vec2(qx, qy).add(warp).toVar('foamLaceQ');
    const z = filamentNoise(q).toVar('foamLaceZ');
    // The long frame, for the foam that is thinning; its own field, blended in
    // and renormalised so the blend has the spread of either.
    const aged = a.smoothstep(AGED_FROM, AGED_TO).oneMinus().toVar('foamLaceAged');
    If(aged.greaterThan(0.001), () => {
      const qAged = vec2(frame.x.div(FILAMENT_ACROSS * AGED_STRETCH), qy).add(warp).add(vec2(13.7, 29.1));
      const fresh = aged.oneMinus();
      z.assign(mix(z, filamentNoise(qAged), aged).div(fresh.mul(fresh).add(aged.mul(aged)).sqrt()));
    });
    const detailQ = sharp
      ? vec2(frame.x.div(FILAMENT_ACROSS * DETAIL_STRETCH), qy).add(warp).toVar('foamLaceDetailQ')
      : q;
    const midOn = foamUniforms.detail
      .greaterThan(0.5)
      .select(sharp ? smoothstepDownClamped(pixel, 0.03, 0.06) : smoothstepDownClamped(viewDistance, 18, 45), float(0))
      .toVar('foamLaceMidOn');
    const fineOn = foamUniforms.detail
      .greaterThan(1.5)
      .select(sharp ? smoothstepDownClamped(pixel, 0.01, 0.02) : smoothstepDownClamped(viewDistance, 7, 18), float(0))
      .toVar('foamLaceFineOn');
    const cmOn = sharp
      ? foamUniforms.detail
          .greaterThan(1.5)
          .select(smoothstepDownClamped(pixel, 0.0035, 0.007), float(0))
          .toVar('foamLaceCmOn')
      : float(0);
    const midWeight = midOn.mul(FILAMENT_MID_WEIGHT);
    const fineWeight = fineOn.mul(FILAMENT_FINE_WEIGHT);
    const cmWeight = cmOn.mul(FILAMENT_CM_WEIGHT);
    If(midOn.greaterThan(0.001), () => {
      z.addAssign(filamentNoise(detailQ.mul(FILAMENT_MID).add(vec2(2.9, 8.3))).mul(midWeight));
    });
    If(fineOn.greaterThan(0.001), () => {
      z.addAssign(filamentNoise(detailQ.mul(FILAMENT_FINE).add(vec2(6.1, 3.7))).mul(fineWeight));
    });
    if (sharp) {
      If(cmOn.greaterThan(0.001), () => {
        z.addAssign(filamentNoise(detailQ.mul(FILAMENT_CM).add(vec2(1.3, 9.9))).mul(cmWeight));
      });
    }
    const spread = midWeight
      .mul(midWeight)
      .add(fineWeight.mul(fineWeight))
      .add(cmWeight.mul(cmWeight))
      .add(1)
      .sqrt()
      .mul(FILAMENT_SD);
    const t = t0
      .mul(foamNoise(q.mul(vec2(1.6, 0.9)).add(vec2(2.2, 5.5))).mul(STRAND_WIDTH_SPAN).add(STRAND_WIDTH_LOW))
      .toVar('foamLaceWidth');
    // The strand's edge: one pixel of the field's gradient, so a strand is as
    // crisp as the screen allows near the camera; never wider than round 8's
    // fifth of the spread far off, where that was what kept it from shimmering.
    const gradient = midOn
      .mul(GRAD_MID)
      .add(fineOn.mul(GRAD_FINE))
      .add(cmOn.mul(GRAD_CM))
      .add(GRAD_COARSE);
    const edge = sharp ? gradient.mul(pixel).mul(1.5).clamp(0.03, 0.22) : float(0.22);
    const band = z.abs().div(spread).smoothstep(t.sub(edge), t.add(edge)).oneMinus();
    strands.assign(mix(mean, band, near));
  });
  return strands;
}

/**
 * Bubble froth at the camera: an opacity multiplier, 1 where the foam is
 * thick and down to `1 - FROTH_DEPTH` in the hollows of the bubbles, for the
 * last few metres where a pixel is under about 4 mm and a whitecap seen that
 * close is visibly a mass of bubbles from a few millimetres to a few
 * centimetres (Crest's bubble foam layer, web-ocean's `bubbles` at their two
 * finest scales). Hollows, not holes: the water is only seen through them, so
 * it is the opacity they take, and the lace's coverage is left alone. Clustered
 * by a 30 cm noise, since bubbles of one size burst together. Faded to 1 before
 * the cells fall under a few pixels. Must be called inside a TSL `Fn`.
 */
const FROTH_DEPTH = 0.45;

export function foamFroth(p: any, footprint: any): any {
  const pixel = footprint.mul(0.5);
  const near = foamUniforms.detail
    .greaterThan(1.5)
    .select(smoothstepDownClamped(pixel, 0.002, 0.004), float(0))
    .toVar('foamFrothNear');
  const froth = float(1).toVar('foamFroth');
  If(near.greaterThan(0.001), () => {
    const cluster = foamNoise(p.mul(3.3).add(vec2(4.7, 1.3))).smoothstep(0.3, 0.75);
    const large = foamBubbles(p.mul(36).add(vec2(2.3, 7.1)), 61, 0.12, 0.45);
    const small = foamBubbles(p.mul(90).add(vec2(8.9, 0.7)), 73, 0.1, 0.4);
    const hollow = large.mul(small.mul(0.5).add(0.5)).oneMinus().mul(cluster.mul(0.6).add(0.4));
    froth.assign(hollow.mul(near).mul(-FROTH_DEPTH).add(1));
  });
  return froth;
}

/**
 * A blurred bubble cloud, 0..1, for the aerated layer under the white
 * (Crest's `BubbleFoamTexture`): the same structure at a metre scale, with no
 * fine detail, so read through a parallax offset it looks like bubbles hanging
 * in the water below the foam rather than painted on it.
 */
export function foamBubbleCloud(p: any): any {
  return foamNoise(p.mul(1.1).add(vec2(3.3, 9.2)))
    .mul(0.6)
    .add(foamNoise(p.mul(2.9).add(vec2(7.7, 1.4))).mul(0.4))
    .smoothstep(0.3, 0.75);
}

/**
 * The foam pattern at world point `p` (metres, xz), about [0.25, 0.85]: web-ocean's
 * `foamPattern`, `R * 0.78 + G * 0.22`, with R and G its `FOAM_TEX` channels
 * sampled at `xz * 0.23` and `xz * 0.71`. High values are where thin foam
 * survives longest. Must be called inside a TSL `Fn`.
 *
 * Raw, not normalised: `OceanMaterial` applies web-ocean's own threshold to it,
 * written for these values. `featherFoam` normalises it for the callers that
 * use the feather.
 *
 * @param p            World xz, already moved by whatever advects the foam.
 * @param viewDistance Metres from the eye, for the per-scale fades.
 */
export function foamPattern(p: any, viewDistance: any): any {
  // R: sheets with ragged holes, bubbles, density. web-ocean's texture uv.
  const u = p.mul(0.23).toVar('foamU');
  const dens = gradFbm(u.mul(4), 4).mul(0.5).add(0.5).toVar('foamDens');

  // Sheets, 0.4 m. Resolved to ~250 m at 1080p before the grazing angle
  // foreshortens them under a pixel; then their mean.
  const m1 = float(MEAN_M1).toVar('foamM1');
  const sheetNear = smoothstepDownClamped(viewDistance, 110, 260).toVar('foamSheetNear');
  If(sheetNear.greaterThan(0.001), () => {
    const w1 = vec2(gradFbm(u.mul(5), 2), gradFbm(u.mul(5).add(vec2(3.7, 9.1)), 2));
    const holes = gradFbm(u.mul(10).add(w1.mul(1.4)), 3).smoothstep(-0.14, 0.06);
    m1.assign(mix(float(MEAN_M1), holes, sheetNear));
  });

  // 15 cm: the sheets broken again, and the first bubbles. Medium and up.
  const m2 = float(MEAN_M2).toVar('foamM2');
  const b3 = float(MEAN_B3).toVar('foamB3');
  const midNear = smoothstepDownClamped(viewDistance, 30, 80).toVar('foamMidNear');
  If(foamUniforms.detail.greaterThan(0.5).and(midNear.greaterThan(0.001)), () => {
    const w2 = vec2(gradNoise(u.mul(12).add(1.3)), gradNoise(u.mul(12).add(7.9)));
    m2.assign(mix(float(MEAN_M2), gradFbm(u.mul(27).add(w2.mul(0.45)), 3).smoothstep(-0.16, 0.04), midNear));
    b3.assign(mix(float(MEAN_B3), foamBubbles(u.mul(29), 0, 0.06, 0.45), midNear));
  });

  // G: web-ocean's finer lace, bubbles at 5 cm and 2 cm. High and up, near only.
  const fine = float(MEAN_FINE).toVar('foamFine');
  const fineNear = smoothstepDownClamped(viewDistance, 10, 30).toVar('foamFineNear');
  If(foamUniforms.detail.greaterThan(1.5).and(fineNear.greaterThan(0.001)), () => {
    const u2 = p.mul(0.71);
    const g3 = foamBubbles(u2.mul(29).add(vec2(5.3, 1.7)), 0, 0.06, 0.45);
    const g4 = foamBubbles(u2.mul(61), 31, 0.05, 0.42);
    fine.assign(mix(float(MEAN_FINE), g3.mul(g4.mul(0.3).add(0.7)).mul(0.55).add(0.45), fineNear));
  });
  const fineD = gradFbm(p.mul(0.71 * 16), 2).mul(0.5).add(0.5);

  const big = m1.mul(m2.mul(0.45).add(0.55)).mul(b3.mul(0.25).add(0.75));
  const r = big.mul(0.6).add(dens.mul(0.4)).clamp(0, 1);
  const g = fine.mul(0.8).add(fineD.mul(0.2)).clamp(0, 1);
  return r.mul(0.78).add(g.mul(0.22));
}

/**
 * Wind streaks, about [0.35, 0.65], after web-ocean's `FOAM_TEX` alpha, which
 * web-ocean multiplies into *aged* foam as `smoothstep(0.25, 0.75, streak) * 1.5`
 * — what pulls a decaying patch into the lanes a wind sea leaves (Beaufort 7:
 * "foam blown in well-marked streaks").
 *
 * **Not web-ocean's 12:1 stretch.** Lanes 7 m by 0.6 m, ruler-straight, were
 * the critic's "ladder of over-regular parallel streak bands" (waves-4) and the
 * "hatch lines" in the streak foam at the horizon (waves-3): gradient noise
 * stretched that far has no features left along the stretch, so what survives
 * the threshold is a comb. Here the lanes are 4:1 (about 3.6 m by 0.9 m) and
 * meander: the across-wind coordinate is bent by a 12 m noise, the way real
 * streaks follow Langmuir cells rather than rulings. The finer octaves fade
 * before they fall under a pixel across the wind, so far off this flattens to
 * 0.5 and `foamPatchiness` takes over.
 */
export function foamStreaks(p: any, viewDistance: any): any {
  const along = vec2(foamUniforms.windAxis.x, foamUniforms.windAxis.y);
  const across = vec2(foamUniforms.windAxis.y.negate(), foamUniforms.windAxis.x);
  const frame = vec2(p.dot(along), p.dot(across)).toVar('foamStreakFrame');
  const meander = gradNoise(frame.mul(0.083).add(vec2(3.1, 7.7))).mul(2.4);
  const q = vec2(frame.x.mul(0.28), frame.y.add(meander).mul(1.1)).toVar('foamStreakQ');
  const first = gradNoise(q).mul(0.5);
  const rest = gradNoise(q.mul(2)).mul(0.25).add(gradNoise(q.mul(4)).mul(0.125));
  return first
    .mul(smoothstepDownClamped(viewDistance, 250, 600))
    .add(rest.mul(smoothstepDownClamped(viewDistance, 120, 300)))
    .mul(0.5)
    .add(0.5);
}

/**
 * White coverage for a foam `amount` over a raw `pattern` value, for the callers
 * that want the amount to be the white fraction (the swash on the sand, the
 * surf). 0 is exactly no foam; 1 is solid.
 *
 * Over the normalised pattern the threshold runs from 1 (nothing clears it) at
 * amount 0 to `-FEATHER` (everything is past the ramp) at amount 1, linearly, so
 * the white fraction tracks the amount: 0.8 is white with bubbles showing
 * through, 0.5 is lace, 0.2 is shreds.
 */
export function featherFoam(amount: any, pattern: any): any {
  const normalised = pattern.sub(PATTERN_LOW).div(PATTERN_SPAN);
  const threshold = amount.clamp(0, 1).mul(-(1 + FEATHER)).add(1).toVar();
  return normalised.smoothstep(threshold, threshold.add(FEATHER));
}

/**
 * Large-scale unevenness of foam density, 0..1, at world point `p`.
 *
 * The far-field stand-in for `foamStreaks`. Past a few hundred metres the 0.6 m
 * lanes are under a pixel, and a saturated patch of the foam buffer would read as
 * one flat white oval (the critic's "paper cutouts", foam-2). This breaks it into
 * rafts and lanes at 5–26 m, the scale the eye reads a patch's shape at from
 * there. Wind-stretched like the streaks.
 */
export function foamPatchiness(p: any): any {
  const along = vec2(foamUniforms.windAxis.x, foamUniforms.windAxis.y);
  const across = vec2(foamUniforms.windAxis.y.negate(), foamUniforms.windAxis.x);
  const q = vec2(p.dot(along).div(STREAK_STRETCH * 1.3), p.dot(across));
  return foamNoise(q.div(26).add(vec2(4.1, 7.6)))
    .mul(0.4)
    .add(foamNoise(q.div(11).add(vec2(9.7, 3.3))).mul(0.4))
    .add(foamNoise(q.div(4.7).add(vec2(1.9, 6.1))).mul(0.2))
    .smoothstep(0.28, 0.72);
}
