import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  cos,
  float,
  mix,
  pow,
  screenSize,
  sin,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';

/**
 * Rain on the lens, as a post-processing node graph.
 *
 * Water on glass is not simulated here. It is *sampled*: the screen is divided
 * into a lattice of cells and each cell hashes to at most one droplet — a
 * position, a radius, a birth phase — so the whole field is a pure function of
 * screen position, time and a handful of uniforms. Nothing is stored between
 * frames, which is what makes the effect free to start, stop, rewind and run on
 * a backend with no compute shaders.
 *
 * Four such lattices are composited at different scales and rates. One lattice
 * on its own always reads as a lattice; four, with different cell aspects, seeds
 * and turnover rates, do not.
 *
 * Three things carry most of the look:
 *
 *   1. **Droplets are lenses, not decals.** Each one is a spherical cap; the
 *      surface normal is derived from the distance to the cap's centre, the view
 *      ray is refracted through it at the air/water interface, and the *image*
 *      is read back from the scene colour at the displaced coordinate. A drop on
 *      glass is a plano-convex lens of focal length ~3r, so the plane it images
 *      sits a few radii behind it and the picture inside a drop comes out
 *      inverted and wide-angle — which is exactly what a drop on a window looks
 *      like, and what a normal-map smear never manages.
 *   2. **Big drops run, small ones do not.** Contact-angle hysteresis pins a
 *      drop's contact line with a force that scales with its perimeter, while
 *      gravity pulls on its volume, so there is a critical radius above which a
 *      drop breaks away and below which it sits (Furmidge's criterion). That
 *      single comparison is what sorts the field into static beads and runners,
 *      rather than a flag hung on a layer.
 *   3. **Runners stick and slip.** The same pinning that holds a small drop in
 *      place keeps grabbing a large one on its way down, so a real leak crawls
 *      and lurches. The travel curve is deliberately non-linear for that reason;
 *      a drop moving at constant speed reads as an animated texture.
 *
 * Cost is one branch on a clear frame. Everything below the top-level `If` is
 * skipped when the glass is dry, the layer count is a uniform-coherent branch
 * per lattice, and — this is the reason the field is accumulated before anything
 * is sampled — the number of texture reads does not depend on the layer count at
 * all. Adding a lattice costs arithmetic, never bandwidth.
 *
 * Backend note: the lattice is built in a square metric (screen *height* units)
 * derived from `screenSize`, so cells stay square and drops stay round at any
 * aspect ratio, with no resize hook to keep in sync. Every texture read sits
 * either at the top level or inside a branch whose condition is a uniform, which
 * both WGSL's uniformity analysis and GLSL's implicit-derivative rules require.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface LensRainParams {
  /**
   * Distance, in drop radii, to the plane a droplet images.
   *
   * A water drop on glass is a plano-convex lens with `f = R/(n-1) ≈ 3r`, so the
   * scene it shows sits a few radii behind it — and because that distance scales
   * with the drop, a bead and a runner both show a coherent picture rather than
   * the bead scrambling. Inversion sets in near 7; above that the image inside a
   * drop is inverted and magnified, which is what a real one does.
   */
  lensDepth: number;
  /** Misting on the glass between the droplets, 0..1. */
  fog: number;
  /**
   * Radius of the glass gather at the top tier, in screen heights.
   *
   * **Read this together with `GLASS_TAPS`.** A radius is only meaningful next
   * to the number of samples spread over it: the two together set the tap
   * density, and it is density — not strength — that decides whether the result
   * reads as a blur or as a handful of copies of the image. See `GLASS_TAPS`.
   */
  fogRadius: number;
  /**
   * How far the wet glass defocuses the world *outside* the droplets, 0..1.
   *
   * The droplets themselves stay sharp — see the glass block in the node graph.
   * Without this the beads are perfectly resolved objects sitting on a perfectly
   * resolved scene, which reads as decals rather than as lenses.
   */
  backdropBlur: number;
  /** Chromatic spread of the lens read, as a fraction of its own displacement. */
  dispersion: number;
  /** Specular strength on the droplet caps. */
  glint: number;
  /** Tightness of that highlight. */
  glintPower: number;
  glintColor: THREE.Color;
  /**
   * Radius above which a drop overcomes contact-line pinning and runs, in screen
   * heights. Scaled down by the gravity magnitude — tilt the glass or drive the
   * drops with wind and more of them break away.
   */
  rollRadius: number;
  /** sin(contact angle): how domed a bead is, and so how hard it refracts. */
  capSlope: number;
  /** Clumping of the density field. 0 spreads drops evenly over the frame. */
  clump: number;
  /** Seconds for the glass to wet up when rain starts. */
  wetTime: number;
  /** Seconds for it to dry off when rain stops. */
  dryTime: number;
}

export const DEFAULT_LENS_RAIN_PARAMS: LensRainParams = {
  lensDepth: 6.5,
  fog: 0.32,
  // 0.005 of a screen height — about 4.5 px at 900 lines, which is what
  // `GLASS_TAPS` samples can actually resolve. It was 0.012, a radius nothing in
  // this file had the taps to fill.
  fogRadius: 0.005,
  backdropBlur: 0.7,
  // Water's own dispersion, not a look. n runs 1.331 (700 nm) to 1.339
  // (400 nm), so (n − 1) spreads about ±1.2% about its mean, and the split
  // here is ± this fraction of the displacement. It was 0.05, four times
  // that, and the drops carried the rainbow rims the A/B critic called out.
  dispersion: 0.012,
  glint: 0.55,
  glintPower: 26,
  glintColor: new THREE.Color(0xffffff),
  rollRadius: 0.014,
  // sin(64°). A coated lens beads harder than this, but a cap steeper than 90°
  // overhangs its own contact circle and a height field cannot express that.
  capSlope: 0.9,
  clump: 0.55,
  wetTime: 1.1,
  dryTime: 7.5,
};

interface LayerConfig {
  /** Cells per screen height, across gravity and along it. */
  grid: [number, number];
  /** Drop radius range, in screen heights. */
  size: [number, number];
  /** Fraction of cells holding a drop at full rain. */
  density: number;
  /** Life cycles per second — turnover for beads, traverse rate for runners. */
  speed: number;
  /** Trail contribution. 0 for lattices whose drops are all below `rollRadius`. */
  trail: number;
  /** Lattice offset, so no two layers share a hash. */
  seed: number;
}

/**
 * The four lattices, ordered by how much each one is missed when it is dropped.
 *
 * The runners and the beads come first because those two are what makes an image
 * read as "seen through wet glass" at all: the beads carry the texture and the
 * runners carry the motion. The middle grade and the fine spray fill in between
 * them, and a tier that cannot afford them still looks like rain rather than
 * looking broken.
 *
 * Cell aspect is not decorative. The runners' lattice is almost three times taller
 * than it is wide because a trail lives inside its drop's own cell — bounding a
 * drop to its cell is what removes the neighbourhood search — and a square cell
 * would cap every streak at one cell width.
 */
const LAYERS: LayerConfig[] = [
  // Trail strength halved from 1. The runners' streaks read as smeared glass
  // rather than as water: a real trail is a thread of small residual beads, and
  // at full strength this one refracted enough of the image along its length to
  // look like a wipe. The drops themselves are untouched — it is only the tail
  // they leave that is quieter.
  { grid: [7, 2.6], size: [0.016, 0.05], density: 0.72, speed: 0.14, trail: 0.5, seed: 0 },
  { grid: [19, 19], size: [0.004, 0.013], density: 0.95, speed: 0.03, trail: 0, seed: 37.4 },
  { grid: [11, 7], size: [0.007, 0.026], density: 0.8, speed: 0.09, trail: 0.35, seed: 91.7 },
  { grid: [34, 34], size: [0.0022, 0.006], density: 1, speed: 0.02, trail: 0, seed: 143.1 },
];

/** Layers 0 and 1 run whenever the effect runs; the rest are tier-gated. */
const BASE_LAYERS = 2;

/** Refractive index ratio at the air/water interface. */
const ETA = 1 / 1.333;

/** Pinning events per traverse, and how close each one comes to a full stop. */
const STICK_EVENTS = 4;
const STICK_MIN = 0.5;
const STICK_MAX = 0.97;

/** Width of a trail just behind its drop, and at the far end, in drop radii. */
const TRAIL_NEAR = 0.7;
const TRAIL_FAR = 0.22;
/** A trail is a shallower ridge than the drop that laid it. */
const TRAIL_SLOPE = 0.75;
/** How much of a trail's width the beading modulates, peak to peak. */
const BEAD_DEPTH = 0.6;

/**
 * Taps in the glass gather, and the reason the number matters as much as the
 * radius does.
 *
 * This gather was eight samples on two fixed rings spanning a radius of 0.012
 * screen heights — about eleven pixels at 900 lines. Eight samples cannot
 * band-limit eleven pixels: each one lands far enough from its neighbours to
 * stay individually visible, so what the frame showed was not a defocused image
 * but eight superimposed copies of it. Against thin high-contrast geometry —
 * rigging, masts, the island's edge — that read as smearing and ghosting, and
 * it is what made the storm preset look broken.
 *
 * The controlling measurement, at a fixed weight of 0.7 and unchanged
 * everything else: at radius 0.012 the rigging ghosts badly, at 0.006 it still
 * ghosts, and at 0.003 it is clean. The strength was never the problem; the
 * sampling was.
 *
 * `DepthOfField` already states the rule this file was breaking — sample density
 * over a disc goes as `N / (pi r^2)`, so a radius chosen without reference to
 * the tap count "would start to read as individual samples rather than as a
 * defocused image". That is exactly what happened here. So the gather is now the
 * same sunflower spiral the depth of field uses, and `applyParams` scales the
 * radius with `sqrt(N)` so density stays fixed as the tier changes the count.
 *
 * Sixteen taps over ~4.5 px puts roughly two pixels between neighbours, which is
 * where a gather stops having visible structure.
 */
const GLASS_TAPS = 16;
/** The middle tier's count. Radius follows it down; see `applyParams`. */
const GLASS_TAPS_MID = 10;

/**
 * Golden angle, for the sunflower spiral. The same constant `DepthOfField`
 * uses, and for the same reason: it is the most uniform disc packing available
 * without a lookup table.
 */
const GOLDEN_ANGLE = 2.39996323;

/** Softness of a droplet rim, as a fraction of its radius. */
const RIM_SOFT = 0.22;

/** How sharply a cell drops out of the population as the glass dries. */
const POP_SOFT = 6;

/** Cells of the clumping field per screen height. Deliberately very low. */
const CLUMP_SCALE = 1.7;

/** Extra density in the lower half of the frame, where run-off collects. */
const GRAVITY_BIAS = 0.25;

/** Keeps a lens read off the very edge, where clamp-to-edge would smear. */
const UV_INSET = 0.002;

/** Clock wrap, seconds. Matches the rest of the project's animated systems. */
const CLOCK_WRAP = 3600;

/** Seconds the gravity direction takes to follow a change in wind or tilt. */
const GRAVITY_TAU = 0.9;

/**
 * The film of water a submerged lens carries, and how it behaves — a separate
 * quantity from rain, and separate for three reasons the previous single scalar
 * got wrong.
 *
 * It used to be the rain coverage itself, snapped to 1 whenever `submersion`
 * fell from above 0.5 to below 0.05 between two consecutive updates. Every part
 * of that was wrong:
 *
 *   - **The test almost never fired.** Crossing from 0.5 to 0.05 is 0.315 m of a
 *     0.7 m band, so it needed 19 m/s of vertical motion to happen inside one
 *     frame at 60 Hz. Measured: a camera lifted clear at 10.5 m/s left the lens
 *     completely dry. What *did* satisfy it was a teleport — which is to say a
 *     camera cut, which is the one time there is no water.
 *   - **Full rain coverage is not what a dive leaves.** It put a storm's whole
 *     droplet population on the glass, on a cloudless day, from one dunk.
 *   - **And then dried it as rain.** `dryTime` is 7.5 s: still visibly wet after
 *     15, clear only after 30.
 *
 * A film charges while the lens is under and drains when it is out, with no edge
 * test anywhere, so a slow swim up and a wave over the lens both do the obvious
 * thing and a cut does nothing.
 */
/** Seconds for immersion to load the front element. Short: it is instant. */
const FILM_WET_TAU = 0.14;
/**
 * Seconds for it to drain in air.
 *
 * Draining, not evaporating — which is the whole difference from `dryTime`.
 * Water runs off a vertical surface in about a second, leaving a scatter of
 * pinned beads behind it; it does not sit there for half a minute.
 */
const FILM_DRAIN_TAU = 0.85;
/**
 * Droplet coverage at a fully loaded film, as a fraction of the storm maximum.
 *
 * Below 1 on purpose. Surfacing leaves a scatter of large drops and a few
 * runners, not the four-lattice downpour that coverage 1 asks for.
 */
const FILM_COVERAGE = 0.7;
/**
 * How much of the glass treatment — the misting and the backdrop defocus — a
 * loaded film asks for, as a fraction of what the same coverage of rain asks
 * for.
 *
 * **This is the number that fixes the picture**, and it is low because the
 * defocus is what was destroying the frame rather than the droplets. Attributed
 * by capture: with the defocus alone at storm weight the surfacing frame is
 * ghosted and milky, and with it off the same frame is sharp and correctly
 * coloured with water on the lens. Nine taps on a ring of 0.012 screen heights
 * are nine ghosts, not a blur, once 70% of the pixel is coming from them, and in
 * linear space they smear the sun's glitter over everything before the tone
 * curve ever sees it.
 *
 * Not zero: a lens that has just broken the surface genuinely is not focusing,
 * and a beat of softness that clears as the water runs off is the effect. It is
 * brief because the film is.
 */
const FILM_GLASS = 0.3;

/**
 * Droplet coverage at full rain, as a fraction of what the lattices are tuned
 * to carry.
 *
 * The lattices were tuned for a lens that is the subject of the shot. In play
 * it is not — the sea and the ship are — and at full coverage the A/B critic
 * read the storm frame (waves-5) as "lens droplets with rainbow fringes
 * dominate the frame" and asked for about 80% fewer. Population rather than
 * opacity, so it is the same scaling a lighter shower already gets: fewer
 * drops, each one still a whole lens.
 */
const RAIN_DROP_SHARE = 0.2;

/**
 * How much of the glass treatment — misting and backdrop defocus — rain asks
 * for, as a fraction of the authored weights. The same reasoning as
 * `FILM_GLASS`, and the same value: attributed by capture on the storm shot,
 * the softness the critic took for depth of field was this defocus (turning
 * the depth of field off moved the frame by 0.14 levels on average; turning
 * the lens rain off moved it by 4.4), and at full weight it took the swell out
 * of the sea.
 */
const RAIN_GLASS = 0.3;

export class LensRain {
  private readonly params: LensRainParams;

  private clock = 0;
  private intensity = 0;
  /** Smoothed coverage. Rises on `wetTime`, falls on the much longer `dryTime`. */
  private wetness = 0;
  private submersion = 0;
  /** Water carried out of the sea on the front element, 0..1. See `FILM_*`. */
  private film = 0;
  private quality = 3;
  private disposed = false;

  private readonly gravityTarget = new THREE.Vector2(0, -1);
  private gravityMagnitude = 1;
  private gravityMagnitudeTarget = 1;

  // --- master ---------------------------------------------------------------
  /** Effective droplet coverage after drying, submersion and the tier policy. */
  private readonly uAmount = uniform(0);
  /**
   * Weight on the *glass* — the misting and the backdrop defocus — as opposed to
   * on the droplets.
   *
   * Equal to `uAmount` for rain, and well below it for a film carried out of the
   * water. The two used to be one uniform, which is what made a dive look like a
   * squall: see `FILM_GLASS`.
   */
  private readonly uGlass = uniform(0);
  private readonly uClock = uniform(0);
  /** Unit "down" on the glass, in the square screen metric. */
  private readonly uGravity = uniform(new THREE.Vector2(0, -1));

  // --- droplets -------------------------------------------------------------
  private readonly uRollRadius = uniform(DEFAULT_LENS_RAIN_PARAMS.rollRadius);
  private readonly uCapSlope = uniform(DEFAULT_LENS_RAIN_PARAMS.capSlope);
  private readonly uClump = uniform(DEFAULT_LENS_RAIN_PARAMS.clump);
  private readonly uLensDepth = uniform(DEFAULT_LENS_RAIN_PARAMS.lensDepth);
  private readonly uLayerWeights: any[] = [];

  // --- glass ----------------------------------------------------------------
  private readonly uFog = uniform(0);
  /**
   * The *derived* gather radius, not the authored one.
   *
   * `applyParams` scales `params.fogRadius` by `sqrt(taps / GLASS_TAPS)` before
   * writing it here, so tap density stays constant as the tier changes the count.
   * The same authored-versus-derived split `uRollRadius` already uses.
   */
  private readonly uFogRadius = uniform(DEFAULT_LENS_RAIN_PARAMS.fogRadius);
  /**
   * Tap count, twice — `Loop` needs an integer node for its bound and the
   * spiral's `sqrt(i / n)` needs the same number as a float. Both are uniforms
   * rather than one being derived in the shader, because this changes only on a
   * tier change and converting per fragment would be a cost paid every frame for
   * a value that almost never moves. `DepthOfField` carries the identical pair.
   */
  private readonly uGlassSampleCount: any = uniform(0, 'int');
  private readonly uGlassSamples = uniform(0);
  private readonly uBackdropBlur = uniform(0);
  private readonly uDispersion = uniform(0);

  // --- highlight ------------------------------------------------------------
  private readonly uGlint = uniform(DEFAULT_LENS_RAIN_PARAMS.glint);
  private readonly uGlintPower = uniform(DEFAULT_LENS_RAIN_PARAMS.glintPower);
  private readonly uGlintColor = uniform(new THREE.Color(0xffffff));
  private readonly uLightDir = uniform(new THREE.Vector3(0.34, 0.58, 0.74).normalize());

  /**
   * The lattice, as a TSL function. Built once in the constructor and called
   * once per layer, so the four layers share one body in the generated shader
   * instead of unrolling into four copies of it.
   */
  private readonly layerFn: any;

  constructor() {
    this.params = {
      ...DEFAULT_LENS_RAIN_PARAMS,
      glintColor: DEFAULT_LENS_RAIN_PARAMS.glintColor.clone(),
    };

    for (let i = 0; i < LAYERS.length; i++) {
      this.uLayerWeights.push(uniform(i < BASE_LAYERS ? 1 : 0));
    }

    /**
     * One lattice of droplets.
     *
     * @param q       Screen position in the gravity-aligned square metric; `q.y`
     *                increases downhill, so every drop in every layer travels the
     *                same way and the wind only has to rotate one frame.
     * @param grid    Cells per unit, across and along gravity.
     * @param size    Radius range, in the same units.
     * @param tune    `(density scale, cycles per second, trail strength, seed)`.
     * @param density Local coverage, already clumped and biased.
     *
     * Returns `(normal.x, normal.y, coverage, radius)` in the gravity frame. The
     * radius rides along because the refraction the caller performs needs it: a
     * droplet's image plane sits a fixed number of *its own radii* behind it.
     */
    this.layerFn = Fn(([q, grid, size, tune, density]: any) => {
      const cellSize = vec2(1).div(grid).toVar();
      const g = q.mul(grid).toVar();
      const cell = g.floor().toVar();
      const local = g.fract().toVar();

      const h = cellHash(cell.add(tune.w)).toVar();
      const h2 = cellHash(cell.add(tune.w).add(vec2(19.7, 7.3))).toVar();

      // Population, not opacity, is what rain rate controls: the cells with the
      // highest hash give up first, and because this scales the radius rather
      // than an alpha, a drying frame shows drops shrinking away one by one
      // instead of the whole field going transparent together.
      const act = density.mul(tune.x).sub(h.z).mul(POP_SOFT).clamp(0, 1).toVar();

      // Marshall–Palmer: drop counts fall exponentially with size, so inverting
      // that distribution rather than sampling the radius range uniformly is what
      // keeps every lattice looking like rain instead of like peas. Skewed hard —
      // the median drop lands about a fifth of the way up its range.
      const spread = float(1).sub(h.w.mul(0.985)).log().negate().div(3).clamp(0, 1).toVar();

      // A drop thinner than a pixel and a half is a sparkle, not a drop, and it
      // will crawl and flicker as the camera moves. Floored in screen space so
      // the lattice constants can be written in screen heights and still behave
      // at 720p.
      const rBase: any = mix(size.x, size.y, spread).max(float(1.6).div(screenSize.y)).toVar();

      // Margins keep the drop inside its own cell, which is the whole reason
      // there is no 3x3 neighbourhood search here. Computed from the *unshrunk*
      // radius so a drying drop stays where it landed rather than creeping toward
      // the cell centre.
      const margin: any = vec2(rBase).div(cellSize).add(0.04).min(0.42).toVar();

      // Furmidge's criterion: retention scales with the contact line and so with
      // r, weight scales with volume and so with r³, and the two cross at a
      // critical radius. Above it a drop runs; below it hysteresis holds it, which
      // is why a window in the rain is mostly still beads.
      const rolls = rBase.smoothstep(this.uRollRadius, this.uRollRadius.mul(1.35)).toVar();

      const speed = tune.y.mul(mix(0.55, 1.45, h2.y)).toVar();
      const life = this.uClock.mul(speed).add(h2.x).fract().toVar();

      // Stick–slip. `travel` is `life` minus a sinusoid of matched period, so its
      // derivative is `1 - stick·cos(...)`: strictly positive, near zero at each
      // pinning event, and back to 1 exactly at the cycle boundary. A drop
      // therefore crawls, tears free, and crawls again without the position ever
      // going backwards or jumping at the wrap.
      const stick: any = mix(STICK_MIN, STICK_MAX, h2.z).toVar();
      const wave = life.mul(2 * Math.PI * STICK_EVENTS).sin().toVar();
      const travel = life.sub(stick.mul(wave).div(2 * Math.PI * STICK_EVENTS)).toVar();

      const spanY = margin.y.mul(2).oneMinus().toVar();
      const cx: any = mix(margin.x, margin.x.oneMinus(), h.x).toVar();
      const still: any = mix(margin.y, margin.y.oneMinus(), h.y);
      const cy: any = mix(still, margin.y.add(travel.mul(spanY)), rolls).toVar();

      // Birth and death. The leading ramp is short — a drop lands in a frame or
      // two — and exists only so nothing appears at full size out of nothing. The
      // trailing one is long for a bead, which evaporates, and short for a runner,
      // which is simply leaving the bottom of its cell.
      const env = life
        .smoothstep(0, 0.05)
        .mul(smoothstepDown(life, rolls.mul(0.2).add(0.7), 1))
        .toVar();

      const radius = rBase.mul(env).mul(act).toVar();

      // --- the cap ----------------------------------------------------------
      const delta = local.sub(vec2(cx, cy)).mul(cellSize).toVar();
      const dist = delta.length().toVar();
      const dn = dist.div(radius.max(1e-5)).toVar();
      const cover = smoothstepDown(dn, float(1).sub(RIM_SOFT), 1).toVar();

      // Spherical cap: the lateral component of the normal is the normalised
      // distance from the centre times the sine of the contact angle, so it runs
      // from flat at the apex to `capSlope` at the contact line.
      const slope = dn.min(1).mul(this.uCapSlope).toVar();
      const capN = delta.div(dist.max(1e-5)).mul(slope).toVar();

      // --- the trail --------------------------------------------------------
      //
      // What a runner leaves behind is not a wet stripe but a chain of pinned
      // residue: the contact line snaps back at every slip and strands a bead.
      // So the ridge is modulated along its length at a frequency set by the
      // drop's own size, and it narrows going up as the residue thins out.
      const px = local.mul(cellSize).toVar();
      const dropX = cx.mul(cellSize.x).toVar();
      const behind = cy.mul(cellSize.y).sub(px.y).toVar();
      const span = travel.mul(spanY).mul(cellSize.y).max(1e-5).toVar();
      const along = behind.div(span).clamp(0, 1).toVar();

      // A runner wanders around obstacles it cannot see; without this the trail
      // is a ruled line and gives the whole thing away. Both frequencies are
      // written against the drop's own radius rather than against the cell, so a
      // big runner meanders slowly and a small one does not turn into noise: the
      // meander runs over eight radii and the beading repeats every two.
      const wobble = noise1(px.y.div(rBase.mul(8)).add(h2.w.mul(53)))
        .x.sub(0.5)
        .mul(rBase.mul(0.9))
        .toVar();
      const beading = noise1(px.y.div(rBase.mul(2)).add(h2.w.mul(91))).toVar();
      const bead = beading.x.mul(BEAD_DEPTH).add(1 - BEAD_DEPTH * 0.5).toVar();

      const taper = along.mul(TRAIL_FAR - TRAIL_NEAR).add(TRAIL_NEAR).toVar();
      const halfWidth = radius.mul(taper).mul(bead).toVar();
      const lateral = px.x.sub(dropX).sub(wobble).toVar();
      const tn = lateral.abs().div(halfWidth.max(1e-5)).toVar();
      const trailCover = smoothstepDown(tn, 0.55, 1)
        .mul(behind.smoothstep(0, radius.mul(0.3)))
        .mul(smoothstepDown(behind, span.mul(0.5), span))
        .mul(env)
        .mul(rolls)
        .mul(tune.z)
        .toVar();

      // The ridge's normal. Across it, the same cap profile the drop uses. Along
      // it, the slope of the beading: `d(halfWidth)/dy` scaled by the crown of
      // the cross-section, which is zero at the ridge's edges and full at its
      // spine. That second term is what makes a trail a chain of little lenses
      // rather than a length of flat pipe.
      const crossing = tn.min(1).toVar();
      const crown = crossing.mul(crossing).oneMinus().sqrt().toVar();
      const dWidth = radius
        .mul(taper)
        .mul(beading.y.mul(BEAD_DEPTH))
        .div(rBase.mul(2))
        .toVar();
      const trailN = vec2(
        lateral.sign().mul(crossing).mul(this.uCapSlope).mul(TRAIL_SLOPE),
        dWidth.mul(crown).mul(this.uCapSlope).negate(),
      ).toVar();

      // Nearest surface wins rather than the two summing: a trail passing under a
      // bead is occluded by it, and adding the two normals would give a slope
      // neither surface has.
      const pick = trailCover.step(cover).toVar();
      const wet = cover.max(trailCover).toVar();
      const depth: any = mix(radius, halfWidth, pick);

      // The image depth is faded out over the last two percent of the coverage
      // ramp. Not cosmetic: away from any droplet the ridge term saturates at its
      // rim slope, and leaving that in would have every pixel of a rainy frame
      // performing a scattered dependent read it then multiplies away — correct
      // output, ruinous cache behaviour. At zero depth the read collapses back
      // onto the pixel's own coordinate.
      return vec4(
        mix(capN, trailN, pick),
        wet,
        depth.mul(wet.smoothstep(0, 0.08)),
      );
    });
  }

  /**
   * Composites droplets over an incoming image.
   *
   * @param sceneColor The scene colour *texture* node, e.g.
   *   `scenePass.getTextureNode()`. It has to be a texture: a droplet is a lens,
   *   so the pass re-reads the image at displaced coordinates, which a computed
   *   colour node cannot supply. Wrap one in `rtt()` if the graph has already
   *   left texture form.
   * @returns The node to hand on — to `RenderPipeline.outputNode`, or to the next
   *   pass that accepts a colour node.
   */
  build(sceneColor: unknown): unknown {
    const colorNode: any = sceneColor;

    if (colorNode === null || colorNode === undefined || typeof colorNode.sample !== 'function') {
      throw new Error(
        'LensRain.build: sceneColor must be a texture node, e.g. scenePass.getTextureNode(). ' +
          'Droplets refract, so the pass has to re-read the image at displaced ' +
          'coordinates, and a computed colour node cannot be resampled.',
      );
    }

    // PassTextureNode leaves `uvNode` null and falls back to the quad's uv.
    const baseUv: any = colorNode.uvNode ?? uv();

    return Fn(() => {
      const suv: any = vec2(baseUv).toVar('lrUv');
      const src = colorNode.sample(suv).toVar('lrSrc');
      const outRgb = vec3(src.rgb).toVar('lrOut');

      // Uniform-coherent branch: dry glass costs one compare, and every texture
      // read below it is reached only through conditions that are themselves
      // uniform.
      If(this.uAmount.greaterThan(0.001), () => {
        const aspect: any = screenSize.x.div(screenSize.y);

        // Square metric: one unit is one screen height. Cells stay square and
        // drops stay round however wide the window is, and no resize hook is
        // needed because `screenSize` already tracks the target.
        const p = vec2(suv.x.mul(aspect), suv.y).toVar('lrP');

        // The lattice is built in the gravity frame rather than in screen axes,
        // so a change of wind or camera roll only rotates one basis instead of
        // needing every drop to steer. The direction is eased on the CPU side for
        // exactly that reason: rotating this basis slides the whole field, and a
        // step change in it would be a visible swim.
        const down = vec2(this.uGravity).toVar('lrDown');
        const tangent = vec2(down.y.negate(), down.x).toVar('lrTangent');
        const q = vec2(p.dot(tangent), p.dot(down)).toVar('lrQ');

        // Rain does not arrive as an even wash, and a lens that fogs evenly reads
        // as a dirty filter rather than as weather. A very low-frequency field
        // clumps the population into patches and leaves gaps for the image to
        // come through, and the same evaluation drives the misting so the two
        // agree about where the water is.
        const clump = noise2(
          p.mul(CLUMP_SCALE).add(vec2(this.uClock.mul(0.013), this.uClock.mul(-0.009))),
        ).toVar('lrClump');
        const patch = mix(float(1).sub(this.uClump), float(1).add(this.uClump), clump).toVar();

        // Run-off collects downhill, so the lower part of the frame carries more.
        const bias = float(1).add(p.sub(vec2(aspect.mul(0.5), 0.5)).dot(down).mul(GRAVITY_BIAS));
        const density = this.uAmount.mul(patch).mul(bias).clamp(0, 1.2).toVar('lrDensity');

        // --- the field ------------------------------------------------------
        //
        // Accumulated before anything is sampled. That ordering is the reason a
        // tier can add lattices without adding bandwidth: layers cost arithmetic,
        // and the texture reads below happen once regardless of how many ran.
        const field = vec4(0, 0, 0, 0).toVar('lrField');

        for (let i = 0; i < LAYERS.length; i++) {
          const emit = (): void => {
            const layer = this.layerFn(
              q,
              vec2(LAYERS[i].grid[0], LAYERS[i].grid[1]),
              vec2(LAYERS[i].size[0], LAYERS[i].size[1]),
              vec4(LAYERS[i].density, LAYERS[i].speed, LAYERS[i].trail, LAYERS[i].seed),
              density,
            ).mul(vec4(1, 1, this.uLayerWeights[i], 1)).toVar();
            // Frontmost wins, for the same reason the trail loses to its own
            // drop: two surfaces at the same pixel are one surface, not the sum
            // of two slopes.
            field.assign(mix(field, layer, layer.z.step(field.z)));
          };

          if (i < BASE_LAYERS) {
            emit();
          } else {
            If(this.uLayerWeights[i].greaterThan(0.0001), emit);
          }
        }

        const cover = field.z.clamp(0, 1).toVar('lrCover');
        const normal = tangent.mul(field.x).add(down.mul(field.y)).toVar('lrNormal');

        // --- refraction -------------------------------------------------------
        //
        // Snell at the air/water interface, for a view ray straight down the
        // screen axis. Entering a denser medium there is no total internal
        // reflection, so this stays bounded: the ray deviates by at most ~41°
        // however steep the cap gets, and the displacement can never run away.
        //
        // The displacement then scales with the drop's own radius, which is the
        // part that makes the image inside a drop hold together. A lens images a
        // plane at its focal length, and for a plano-convex water drop that is
        // ~3r — a distance that shrinks with the drop. Treating the background as
        // a plane at some fixed depth instead would leave a bead sampling half
        // the screen through a two-pixel aperture, which is noise, not a lens.
        const slope2 = normal.dot(normal).min(0.999).toVar('lrSlope2');
        const nz = float(1).sub(slope2).sqrt().toVar('lrNz');
        const k = float(1).sub(slope2.mul(ETA * ETA)).max(0).sqrt().toVar('lrK');
        const bend = nz.mul(ETA).sub(k).toVar('lrBend');
        const forward = k.sub(nz.mul(ETA)).mul(nz).add(ETA).max(0.2).toVar('lrForward');

        const offsetSq = normal
          .mul(bend.div(forward))
          .mul(field.w)
          .mul(this.uLensDepth)
          .toVar('lrOffset');
        const offsetUv = vec2(offsetSq.x.div(aspect), offsetSq.y).toVar('lrOffsetUv');

        // --- the glass --------------------------------------------------------
        //
        // Weighted by the same clumping field the droplets use, and taken to
        // zero in its low half rather than merely dimmed: a haze applied to
        // every pixel of the frame is a lens filter, not weather, and it is the
        // single easiest way to make a post effect look like one. Excluded from
        // the droplets themselves, because a drop is clear — it is the glass
        // around it that is fogged, and a runner wipes its own path clean.
        const mist = this.uFog
          .mul(this.uGlass)
          .mul(clump.smoothstep(0.2, 0.8))
          .mul(cover.oneMinus())
          .clamp(0, 1)
          .toVar('lrMist');

        // **The world outside the drops goes soft, and the world inside them
        // stays sharp.** That inversion is the effect, and without it the
        // droplets were decals: perfectly resolved beads sitting on a perfectly
        // resolved scene, which is not what a wet lens looks like and is exactly
        // how `storm.png` read.
        //
        // The physics is ordinary. A drop on the front element is far inside the
        // near focus distance, so the lens cannot resolve *it* — what it resolves
        // is the tiny inverted image the drop projects, which is why a raindrop
        // on glass shows a sharp little world in it while the film of water
        // around it defocuses everything behind. The bead is a lens with its own
        // very short focal length; the wet glass between beads is a diffuser.
        //
        // Driven by the rain amount rather than by `fog`, because the two are
        // different weather: `fog` is condensation and micro-spray, present in a
        // squall and absent in clean rain, while this is a property of there
        // being water on the element at all.
        const backdrop = this.uBackdropBlur
          .mul(this.uGlass)
          .mul(cover.oneMinus())
          .clamp(0, 1)
          .toVar('lrBackdrop');

        const softWeight = mist.max(backdrop).toVar('lrSoftW');

        // A sunflower spiral, not a pair of rings. See `GLASS_TAPS` for the
        // measurement that forced this: fixed rings put their samples far enough
        // apart to stay individually visible, and what the storm preset showed
        // was eight copies of the image rather than one blurred one.
        //
        // `sqrt((i + 0.5) / n)` for the radius rather than `(i + 0.5) / n`,
        // because a disc's area grows as the square of its radius — a linear
        // radius clusters every tap near the centre and leaves the rim
        // under-sampled, which is the same failure in a different place.
        //
        // The quadratic falloff gives the kernel a peak instead of a flat disc.
        // A box blur of a bright highlight is a bright disc, and this frame is
        // full of specular highlights on water.
        //
        // **The branch is on the uniforms, not on `softWeight`.** `softWeight` is
        // per-pixel — it carries the clumping field and the droplet coverage —
        // and a loop of texture reads inside a non-uniform branch is exactly what
        // the note at the top of this file says must never happen: WGSL's
        // uniformity analysis rejects it and GLSL's implicit derivatives are
        // undefined there. Gating on whether the *effect* is on and then
        // weighting the result per pixel gives the same image with a coherent
        // branch.
        const soft = vec3(src.rgb).toVar('lrSoft');
        If(this.uFog.add(this.uBackdropBlur).greaterThan(0.0001), () => {
          const r = vec2(this.uFogRadius.div(aspect), this.uFogRadius).toVar('lrR');
          // The centre tap, at full weight. Already in hand, so it costs nothing.
          const accum = vec3(src.rgb).toVar('lrAccum');
          const weight = float(1).toVar('lrAccumW');

          Loop(this.uGlassSampleCount, ({ i }: any) => {
            const fi = float(i).toVar('lrI');
            const theta = fi.mul(GOLDEN_ANGLE).toVar('lrTheta');
            const t = fi.add(0.5).div(this.uGlassSamples).sqrt().toVar('lrT');
            const offset = vec2(cos(theta), sin(theta)).mul(t).mul(r).toVar('lrOff');
            const w = t.mul(t).mul(0.8).oneMinus().toVar('lrTapW');
            accum.addAssign(
              colorNode.sample(suv.add(offset).clamp(UV_INSET, 1 - UV_INSET)).rgb.mul(w),
            );
            weight.addAssign(w);
          });

          soft.assign(accum.div(weight.max(1e-4)));
        });

        const glass = mix(src.rgb, soft, softWeight).toVar('lrGlass');

        // --- the lens ---------------------------------------------------------
        const lensUv = suv.add(offsetUv).clamp(UV_INSET, 1 - UV_INSET).toVar('lrLensUv');
        const lens = vec3(colorNode.sample(lensUv).rgb).toVar('lrLens');

        If(this.uDispersion.greaterThan(0.0001), () => {
          // Water's index falls across the visible band, so red bends least and
          // blue most. Splitting the *existing* displacement rather than
          // recomputing Snell per channel keeps this to two extra reads for a
          // fringe that is only ever visible at a drop's rim anyway.
          const shift = offsetUv.mul(this.uDispersion).toVar();
          lens.assign(
            vec3(
              colorNode.sample(lensUv.sub(shift).clamp(UV_INSET, 1 - UV_INSET)).r,
              lens.g,
              colorNode.sample(lensUv.add(shift).clamp(UV_INSET, 1 - UV_INSET)).b,
            ),
          );
        });

        // Grazing parts of the cap send most of their light out of the eye's
        // acceptance cone, which is why the outline of a real drop is darker than
        // its middle, and why the picture there is smeared rather than sharp.
        const rim = slope2.sqrt().smoothstep(0.3, 0.95).toVar('lrRim');
        lens.assign(mix(lens, soft, rim.mul(0.45)).mul(rim.mul(0.3).oneMinus()));

        const wet = mix(glass, lens, cover).toVar('lrWet');

        // The specular is what stops the drops reading as holes cut in the image.
        // Taken against the full reconstructed normal, so it tracks the cap and
        // slides across a runner as it goes down.
        const spec = pow(
          vec3(normal, nz).dot(this.uLightDir).max(0),
          this.uGlintPower,
        ).mul(this.uGlint).mul(cover);

        // A drop's highlight is a reflection of the bright surround, so it is
        // only as bright as the scene the glass sits in. A fixed white glint is
        // right on a sunlit frame and wrong under a storm deck, where it read as
        // a field of white specks ("stars") against the dim sky.
        const surround = glass.dot(vec3(0.2126, 0.7152, 0.0722)).clamp(0, 1);
        const glintLevel = surround.mul(1.6).min(1);

        outRgb.assign(wet.add(this.uGlintColor.mul(spec).mul(glintLevel)));
      });

      return vec4(outRgb, src.a);
    })();
  }

  /** Rain rate, 0..1. Drives droplet population, not droplet opacity. */
  setIntensity(value: number): void {
    this.intensity = clampNumber(value, 0, 1);
  }

  /**
   * Tier policy, as a layer count and a feature set.
   *
   * `0` disables the effect outright — the master uniform goes to zero and the
   * whole graph is skipped by a single compare, so a tier that does not want lens
   * rain pays nothing for the pass still being in the chain.
   *
   * `1` is the WebGL2 and Low floor: runners and beads only, no misting pass, no
   * dispersion. Two texture reads and two lattices.
   * `2` adds the middle grade and the misting. `3` adds the fine spray and the
   * chromatic fringe. Anything higher clamps to 3.
   */
  setQuality(level: number): void {
    this.quality = Math.max(0, Math.min(3, Math.round(level)));
    this.applyParams();
  }

  /** 0 = above water, 1 = fully submerged. */
  setSubmersion(value: number): void {
    this.submersion = clampNumber(value, 0, 1);
  }

  /**
   * Which way is downhill on the glass, and how hard.
   *
   * `(x, y)` is a direction in screen space with `+y` up, so the default is
   * `(0, -1)`. Feed it the camera's roll and the wind's screen-space projection
   * and the drops lean and run with the weather instead of always falling square
   * to the frame. `magnitude` scales the pinning threshold: harder driving breaks
   * more drops free, which is why a windscreen at speed sheets and a parked one
   * beads.
   */
  setGravity(x: number, y: number, magnitude = 1): void {
    const length = Math.hypot(x, y);
    if (length > 1e-4) {
      this.gravityTarget.set(x / length, y / length);
    }
    this.gravityMagnitudeTarget = clampNumber(magnitude, 0.05, 4);
  }

  /**
   * Direction the specular highlight comes from, in the same square screen frame
   * the droplets live in: `+x` right, `+y` up, `+z` toward the viewer. Project the
   * sun into screen space and the glints sit on the sun's side of every bead.
   */
  setLightDirection(x: number, y: number, z: number): void {
    const dir = this.uLightDir.value as THREE.Vector3;
    dir.set(x, y, z);
    if (dir.lengthSq() < 1e-8) dir.set(0, 0, 1);
    dir.normalize();
  }

  setParams(params: Partial<LensRainParams>): void {
    const p = this.params;
    if (params.glintColor !== undefined) p.glintColor.copy(params.glintColor);
    if (params.lensDepth !== undefined) p.lensDepth = params.lensDepth;
    if (params.fog !== undefined) p.fog = params.fog;
    if (params.fogRadius !== undefined) p.fogRadius = params.fogRadius;
    if (params.backdropBlur !== undefined) p.backdropBlur = params.backdropBlur;
    if (params.dispersion !== undefined) p.dispersion = params.dispersion;
    if (params.glint !== undefined) p.glint = params.glint;
    if (params.glintPower !== undefined) p.glintPower = params.glintPower;
    if (params.rollRadius !== undefined) p.rollRadius = params.rollRadius;
    if (params.capSlope !== undefined) p.capSlope = params.capSlope;
    if (params.clump !== undefined) p.clump = params.clump;
    if (params.wetTime !== undefined) p.wetTime = params.wetTime;
    if (params.dryTime !== undefined) p.dryTime = params.dryTime;
    this.applyParams();
  }

  getParams(): Readonly<LensRainParams> {
    return this.params;
  }

  /** Current smoothed coverage, 0..1. Exposed for tests and HUD readouts. */
  getWetness(): number {
    return this.wetness;
  }

  update(dt: number): void {
    if (this.disposed) return;
    // Not clamped to a maximum, unlike the accumulating systems in this project.
    // Nothing here integrates: the droplets are a closed-form function of the
    // clock and the two smoothers below are exponentials, which are stable for
    // any step. Capping the step would only desynchronise this from the rest of
    // the world on a slow frame.
    const step = Number.isFinite(dt) && dt > 0 ? dt : 0;

    this.clock = (this.clock + step) % CLOCK_WRAP;
    this.uClock.value = this.clock;

    // Wetting and drying are not the same process and must not share a constant.
    // Glass covers in a second or two once rain starts; clearing takes most of
    // half a minute, because what is left has to evaporate or run off.
    const tau = this.intensity > this.wetness ? this.params.wetTime : this.params.dryTime;
    this.wetness += (this.intensity - this.wetness) * (1 - Math.exp(-step / Math.max(0.01, tau)));

    // The film the lens carries out of the sea. Charged by *being under the
    // water* and drained by being out of it — a rate on both sides, with no edge
    // test, which is what makes a slow swim up, a crest washing over the lens and
    // a camera cut each do the obvious thing. `FILM_WET_TAU` documents what the
    // edge test this replaces could and could not detect.
    const filmTau = this.submersion > this.film ? FILM_WET_TAU : FILM_DRAIN_TAU;
    this.film += (this.submersion - this.film) * (1 - Math.exp(-step / filmTau));

    const blend = 1 - Math.exp(-step / GRAVITY_TAU);
    const gravity = this.uGravity.value as THREE.Vector2;
    gravity.x += (this.gravityTarget.x - gravity.x) * blend;
    gravity.y += (this.gravityTarget.y - gravity.y) * blend;
    const length = Math.hypot(gravity.x, gravity.y);
    if (length > 1e-4) {
      gravity.x /= length;
      gravity.y /= length;
    } else {
      gravity.set(0, -1);
    }
    this.gravityMagnitude += (this.gravityMagnitudeTarget - this.gravityMagnitude) * blend;
    this.uRollRadius.value = this.params.rollRadius / Math.max(0.05, this.gravityMagnitude);

    this.applyAmount();
  }

  /**
   * Rewinds to a known state.
   *
   * Snaps the smoothed coverage to the requested intensity and the gravity
   * direction to its target as well as rewinding the clock: those are the only
   * state this holds, and a capture that inherited a half-dried lens from the
   * previous shot would not be reproducible.
   */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uClock.value = this.clock;
    this.wetness = this.intensity;
    /**
     * The film is dropped rather than carried across, because a reset is a
     * *teleport* and no water crosses one.
     *
     * This is the same hazard the edge detector this replaces had, and it is
     * worth keeping the record of what it cost. That version fired when
     * submersion fell from above 0.5 to below 0.05, on the reasonable assumption
     * that the only way to do that is to come up through the surface — and a
     * camera cut satisfied it exactly. Cutting from the underwater shot to a
     * clear-day one flooded the lens to full coverage on the first settle step,
     * and since it then dried on the rain constant while a settle run is a second
     * and a half, the clear-day frame was photographed through a screenful of
     * droplets. That is what the `island-approach` baseline looked like, being
     * the shot that follows `underwater` in the list.
     *
     * The rate-based film cannot fire on a cut at all — there is no edge to
     * detect — but it *can* still be carried over one if it is not cleared here,
     * which is the half of the bug that survives the redesign.
     *
     * It is *not* what `tests/isolation.spec.ts` measures, which was worth
     * checking rather than assuming — that test goes storm to clear-day with
     * nothing submerged in between, and its figure was unmoved (1.598 before,
     * 1.595 after). Its residual is sea state surviving the reset, which is a
     * separate leak and still open.
     */
    this.film = 0;

    const gravity = this.uGravity.value as THREE.Vector2;
    gravity.copy(this.gravityTarget);
    this.gravityMagnitude = this.gravityMagnitudeTarget;
    this.uRollRadius.value = this.params.rollRadius / Math.max(0.05, this.gravityMagnitude);

    this.applyAmount();
  }

  /**
   * The pass owns no geometry, material or render target — it contributes a node
   * graph to whichever pipeline consumed it, and that owns the compiled material.
   * `dispose` therefore only stops the clock so a stale reference cannot keep
   * driving uniforms.
   */
  dispose(): void {
    this.disposed = true;
  }

  // -------------------------------------------------------------------------

  /**
   * The two master uniforms, and between them the only thing the shader is told
   * about rain rate, drying, immersion or tier.
   *
   * They are two rather than one because the water on a lens and the *state of
   * the lens* are not the same quantity. Rain moves both together. A film
   * carried out of the sea puts drops on the glass without asking for a squall's
   * worth of misting and defocus behind them, which is what `FILM_GLASS` sets
   * and what stops a dive on a clear day reading as weather.
   *
   * Underwater the lens is not in air and there is nothing on it to bead, so both
   * fade out over the first half of the crossing. Faded rather than cut: a hull
   * rolling through the waterline crosses `submersion` several times a second and
   * a hard switch would strobe.
   */
  private applyAmount(): void {
    if (this.quality === 0) {
      this.uAmount.value = 0;
      this.uGlass.value = 0;
      return;
    }

    // Two gates, because rain and a film need different amounts of daylight.
    //
    // Rain only needs the lens to be *mostly* out of the water — drops are
    // landing on it from above and the moment it clears they are there.
    const emerged = 1 - smoothstepNumber(this.submersion, 0.12, 0.45);
    // A film needs it *properly* clear, and the waterline shot is exactly where
    // the difference shows. An eye eight centimetres under with crests washing
    // over it sits at `submersion` around 0.47, dipping below 0.45 as each one
    // passes — enough for the looser gate to open. What that produced was a
    // half-submerged frame beaded from top to bottom, including over the half
    // that is under the water, where there is no air for a drop to bead against.
    // Water cannot drain off a lens that is still in the sea.
    const clear = 1 - smoothstepNumber(this.submersion, 0.02, 0.16);

    const rain = Math.max(0, this.wetness);
    // Whichever of the two has more water on the glass wins, rather than the two
    // summing: rain falling on a lens that is already wet from a dive does not
    // make it wetter than either, and adding them would push a surfacing frame in
    // a squall past the coverage the lattices are tuned for. Each is gated by its
    // own emergence before the comparison, not after — they are different
    // quantities and the looser gate must not carry the stricter one.
    this.uAmount.value = Math.max(rain * emerged * RAIN_DROP_SHARE, this.film * FILM_COVERAGE * clear);
    this.uGlass.value = Math.max(rain * emerged * RAIN_GLASS, this.film * FILM_GLASS * clear);
  }

  private applyParams(): void {
    const p = this.params;
    const tier = this.quality;

    this.uLensDepth.value = Math.max(0, p.lensDepth);
    this.uCapSlope.value = clampNumber(p.capSlope, 0.05, 0.98);
    this.uClump.value = clampNumber(p.clump, 0, 0.95);

    // Tap count from the tier, and the radius derived from the tap count.
    //
    // The second half is the part that was missing. Sample density over a disc
    // is `N / (pi r^2)`, so holding density constant means `r` scales with
    // `sqrt(N)`. Handing a lower tier the same radius on fewer taps does not buy
    // a cheaper blur, it buys a visibly broken one — which is the defect this
    // whole gather was rewritten to fix, and it would have come straight back at
    // the middle tier if only the count moved.
    const taps = tier >= 3 ? GLASS_TAPS : tier >= 2 ? GLASS_TAPS_MID : 0;
    this.uGlassSamples.value = taps;
    this.uGlassSampleCount.value = taps;
    this.uFogRadius.value = Math.max(0, p.fogRadius) * Math.sqrt(taps / GLASS_TAPS);
    this.uGlint.value = Math.max(0, p.glint);
    this.uGlintPower.value = Math.max(1, p.glintPower);
    (this.uGlintColor.value as THREE.Color).copy(p.glintColor);
    this.uRollRadius.value = p.rollRadius / Math.max(0.05, this.gravityMagnitude);

    for (let i = 0; i < LAYERS.length; i++) {
      this.uLayerWeights[i].value = tier === 0 ? 0 : i < BASE_LAYERS + tier - 1 ? 1 : 0;
    }

    // The misting is four extra reads and the fringe is two more, so both are
    // held back for the tiers that asked for them rather than being faded down
    // and still paid for.
    this.uFog.value = tier >= 2 ? p.fog * (tier >= 3 ? 1 : 0.6) : 0;
    // Same gate as the misting, and for the same reason: the gather is eight
    // reads, so it belongs to the tiers that asked for it rather than being
    // faded to nothing and still paid for.
    this.uBackdropBlur.value = tier >= 2 ? Math.max(0, p.backdropBlur) : 0;
    this.uDispersion.value = tier >= 3 ? Math.max(0, p.dispersion) : 0;

    // Tier gates the master uniform too, and a tier change must land on the very
    // next frame rather than waiting for an `update` that a paused capture may
    // never make.
    this.applyAmount();
  }
}

/**
 * Four decorrelated values per cell.
 *
 * The sine-hash is the project's usual one. It is safe here for the reason it is
 * not always safe: the argument is a screen-space cell index, which never exceeds
 * a few tens, so none of the precision loss that afflicts world-space hashing
 * applies. The exact bit pattern may differ between backends; the *statistics*
 * do not, and no two lattices share a seed.
 */
const cellHash = /*@__PURE__*/ Fn(([cell]: [any]) => {
  const h = vec4(
    cell.dot(vec2(127.1, 311.7)),
    cell.dot(vec2(269.5, 183.3)),
    cell.dot(vec2(419.2, 371.9)),
    cell.dot(vec2(97.3, 233.7)),
  ).toVar();
  return h.sin().mul(43758.5453).fract();
});

/**
 * Interpolated 1D noise and its exact derivative, as `(value, d/dx)`.
 *
 * The derivative is not a convenience. A trail is a ridge whose thickness varies
 * along its length, and the slope of the *surface* along the trail is that
 * variation's derivative — without it the ridge is a smooth half-cylinder, which
 * is flat on top, refracts nothing through its middle and reads as a barely-there
 * smear. Differentiating the smoothstep fade in closed form costs three multiplies
 * on top of the value; finite-differencing it would cost a second pair of hashes.
 */
const noise1 = /*@__PURE__*/ Fn(([x]: [any]) => {
  const i = x.floor().toVar();
  const f = x.fract().toVar();
  const u = f.mul(f).mul(f.mul(-2).add(3)).toVar();
  const a = i.mul(127.1).sin().mul(43758.5453).fract().toVar();
  const b = i.add(1).mul(127.1).sin().mul(43758.5453).fract().toVar();
  const d = b.sub(a).toVar();
  return vec2(a.add(d.mul(u)), d.mul(f.mul(f.oneMinus()).mul(6)));
});

/** One value from a lattice point. Four of these make one `noise2` sample. */
const cornerHash = /*@__PURE__*/ Fn(([p]: [any]) =>
  p.dot(vec2(127.1, 311.7)).sin().mul(43758.5453).fract(),
);

/**
 * Interpolated 2D noise, quintic-faded, for the clumping field.
 *
 * Built on the scalar hash rather than on `cellHash`: this needs one number per
 * corner and four corners, and reusing the four-component hash here would spend
 * sixteen transcendentals per pixel to throw twelve of them away.
 */
const noise2 = /*@__PURE__*/ Fn(([p]: [any]) => {
  const i = p.floor().toVar();
  const f = p.fract().toVar();
  const u = f.mul(f).mul(f.mul(f.mul(6).sub(15)).add(10)).toVar();

  const a = cornerHash(i).toVar();
  const b = cornerHash(i.add(vec2(1, 0))).toVar();
  const c = cornerHash(i.add(vec2(0, 1))).toVar();
  const d = cornerHash(i.add(vec2(1, 1))).toVar();

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
});

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstepNumber(x: number, lo: number, hi: number): number {
  const t = clampNumber((x - lo) / (hi - lo), 0, 1);
  return t * t * (3 - 2 * t);
}
