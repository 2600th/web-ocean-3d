import * as THREE from 'three/webgpu';
import { Fn, If, float, mix, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import { seafloorDepth } from '../scene/Seafloor';

/**
 * World-anchored wake buffer: foam in R, surface elevation in G.
 *
 * One texture covers a square of ocean centred on (usually) the camera. Every
 * frame it is decayed toward zero, resampled to compensate for the centre having
 * moved, and has fresh wake stamped into it wherever something is moving. The
 * water shader reads R as a foam mask and G as extra displacement — so a moving
 * hull both foams the water and visibly *deforms* it.
 *
 * Three decisions worth spelling out:
 *
 * **The texture scrolls, the world does not.** Anchoring the buffer to the
 * camera or the ship and letting the foam ride along with it would make the
 * wake follow the hull like a decal — the one thing a wake must never do. So the
 * buffer's centre is a world coordinate, and when it changes the previous
 * contents are resampled by exactly the offset, in the opposite direction.
 * Anything shifted off the edge is gone, which is correct: it is out of the
 * region the shader can sample anyway.
 *
 * **The pattern is stamped every frame, not accumulated from a point.**
 * Depositing a dot at the hull and letting motion draw the trail gives a
 * straight line, not a wake. Real ship wakes are a fixed pattern in the hull's
 * frame — two arms at the Kelvin half-angle of ~19.5 degrees plus the turbulent
 * band astern — so that pattern is what gets deposited. Accumulation and decay
 * then do what they are actually good at: persistence, and the smearing that
 * makes a turning wake curve.
 *
 * **Kelvin-inspired, not a Kelvin solution.** The dispersion is right — the
 * transverse system's `k = g/V^2` and the divergent system's `k0/cos^2(psi)` are
 * the deep-water relations for waves stationary in the hull's frame — and that is
 * what makes the pattern scale correctly with speed. Everything else is authored:
 * one parabolically warped cosine, one fixed-angle cosine, Gaussian envelopes and
 * a bow mound. It does not integrate a hull pressure distribution, reproduce the
 * stationary-phase cusp where the two systems meet — it borrows the cusp's
 * r^-1/3 amplitude decay for the divergent arms and nothing else of it — respond
 * to Froude number, handle finite depth, or propagate history at the group
 * velocity.
 *
 * **Two foam channels, one output.** The sea's foam and the hull's are not the
 * same material and cannot share a time constant. Whitecap foam is gone a second
 * or two after the crest that made it, while the churn behind a transom is still
 * white a hundred metres astern; running both at the whitecap's 1.5 s meant the
 * wake was erased within half a ship length of the hull that made it, and
 * lengthening the shared constant to fix that turned every breaking crest into a
 * streak. So the ping-pong buffer carries breaking-crest and rain foam in R,
 * hull foam in B, each with its own decay, and the resolve pass composites them
 * into the single foam value the water shader reads. The public texture's layout
 * is unchanged — foam in R, elevation in G — and the split costs nothing: the
 * target was already RGBA and the pass was already sampling all of it.
 *
 * **Foam accumulates; elevation does not.** Foam is history — it is deposited
 * and then decays, and adding this frame's deposit to last frame's remainder is
 * exactly right. Elevation is a *phase* field, and adding a wave pattern to a
 * slightly-decayed copy of the same pattern is how you destroy one: the two
 * differ by whatever the hull moved in a frame, they beat against each other,
 * and what should be a crest line becomes mush. So elevation is blended rather
 * than summed — the fresh stamp replaces the buffer wherever the stamp has any
 * strength, and the decayed history survives only outside it. Near the hull that
 * gives the clean, steady Kelvin pattern, which is correct because the pattern is
 * stationary in the hull's frame. Far astern, where the stamp has faded, the
 * world-space phase laid down on earlier frames persists — which is what lets a
 * turning ship leave a curved wake instead of swinging a rigid one around with it.
 *
 * **The surf line is a depth contour, not a distance from land.** A wave breaks
 * where it runs out of water under it, so the band of white along a shore is the
 * `depth = height / BREAKER_INDEX` contour of the bathymetry — it wanders in and
 * out along a headland because the bottom does, and it steps offshore when the
 * swell gets up. Nothing in this file knows where the island is, only how deep
 * the water is; the floor is handed in as a node and the term works against
 * whatever shape produces it. Evaluating that field is not free, so it sits
 * behind a uniform branch the CPU leaves false whenever the buffer's footprint
 * has no shoreline in it — which is nearly always, and is the case the frame
 * budget is set by.
 *
 * Runs as two fullscreen fragment passes per frame. No compute, no storage
 * textures — this has to work on the WebGL2 backend unchanged.
 */

/** Emissions coalesced into one pass. One hull plus a few props is plenty. */
const MAX_EMITTERS = 4;

/**
 * Exponential decay time constant for the *sea's* own foam, seconds.
 *
 * This is the single biggest lever on apparent whitecap coverage, and it is not
 * obvious why. Only about 5.5% of the surface is folding at any instant — the
 * measured figure for cascade 0 at 15 m/s — but a crest travels at its phase
 * speed, roughly 8.6 m/s for a 47 m swell, so over one time constant it sweeps a
 * band far wider than the patch that deposited it. Coverage is therefore
 * *deposit area times how far the crest runs while the foam survives*, and at
 * 1.7 s that multiplied 5.5% into something closer to a fifth of the frame.
 *
 * 1.5 s still reads as persistence — foam is visibly left behind the crest that
 * made it, which is the whole point of the buffer — without the streaks growing
 * long enough to merge into a sheet.
 */
const DECAY_TAU = 1.5;

/**
 * Decay time constant for hull foam, seconds.
 *
 * A wake is a *trail*, and a trail is the product of persistence and speed: what
 * the eye reads as its length is `V * tau`, not anything the deposit does. At the
 * whitecap's 1.5 s a hull at 8 m/s left 12 m of foam — under half its own length
 * — so the churn ended before the transom had finished passing over it and the
 * ship looked like it was being wiped clean behind. 5 s puts the trail at 40 m
 * per e-fold, which stays legible for around a hundred metres, or three to four
 * ship lengths. That is what a displacement hull actually leaves.
 *
 * Air in a wake genuinely does survive longer than air in a whitecap, and for a
 * reason: a breaking crest entrains a shallow film that dissolves as fast as it
 * rises, while a hull drives bubbles metres down and they come back up over
 * minutes. Five seconds is still far short of that; it is the point where the
 * trail reads as a trail.
 */
const WAKE_FOAM_TAU = 5;

/**
 * Hull foam deposited per second at reference speed.
 *
 * `DEPOSIT_RATE * WAKE_FOAM_TAU` is 5.5, and for the breaking-crest term a
 * number like that would be exactly the mistake `uBreakRate` records: an
 * equilibrium far above white, so anything the term touches saturates within a
 * frame or two. It is not that here, because nothing the hull deposits on is
 * exposed for anything like a time constant. The band deposits with an e-fold of
 * L = 1.2 beams abaft the transom and the hull tows it over the water at V, so a
 * texel is under it for about L/V — 1.1 s at 8 m/s against a 5 s constant. What
 * it ends up holding is the deposit it catches on the way past, thinned by the
 * decay while it is still catching it: about two thirds of `rate * L / V`, which
 * is 0.86 in the band's core and falls away from there.
 *
 * The two speed terms then cancel — `rate` rises with V through `intensity`
 * while the exposure falls as 1/V — so the band comes out about as bright at
 * four knots as at sixteen and only its *length* changes with speed. Which is
 * what a wake does.
 *
 * The guard against the failure this file has hit twice is not the rate, it is
 * the footprint. Both saturations came from depositing across the whole Kelvin
 * wedge, and a wedge that reaches white is a sheet of paper under tow. What
 * reaches white now is a band roughly one beam wide. The arms take `ARM_FOAM` of
 * the rate and are swept sideways across the water at `V * KELVIN_SLOPE`, which
 * is what keeps them feathered rather than filled; the only way to hold a texel
 * in the deposit long enough to saturate the wedge is to tell `emit` that a hull
 * which is not moving is doing eight metres a second.
 */
const DEPOSIT_RATE = 1.1;

/**
 * Relative deposit strength of the three things a hull does to the water.
 *
 * Ordered, and the order is the point: the turbulent band astern is the
 * brightest, most aerated feature of a real wake, the bow break is next, and the
 * arms are a feathered line an order of magnitude thinner than either. Flat
 * weights make a uniform grey V, which is what a wake looks like when it has been
 * drawn rather than shed.
 */
const STERN_FOAM = 1;
const BOW_FOAM = 0.55;
const ARM_FOAM = 0.3;

/**
 * Half the hull's length, in beams.
 *
 * `emit` is called with the hull's *centre* and its beam, so this is the only way
 * the pattern can find the ends of the ship — and every single thing a hull does
 * to the water happens at one end or the other. Measured from midships, the
 * turbulent band sat three-quarters of a beam abaft centre, which on a hull four
 * beams long is amidships: the brightest part of the wake was being deposited
 * *underneath the ship*, where the only view of it is whatever escapes past the
 * tumblehome. The bow mound was likewise half a hull short of the stem.
 *
 * 1.9 is the length/beam ratio of a displacement hull halved — 27 m over 7 m for
 * the ship this renders. It does not have to be exact. It has to put the churn
 * behind the transom instead of under the keel.
 */
const HALF_LENGTH_BEAMS = 1.9;

/** Speed, in m/s, at which foam generation saturates. */
const REFERENCE_SPEED = 7;

/** tan(19.47 deg) — the Kelvin wedge half-angle. */
const KELVIN_SLOPE = 0.3536;

/**
 * Decay time constant for the elevation channel, seconds.
 *
 * Longer than either foam constant, because the three decay for different
 * reasons. Foam disappears when the entrained air dissolves. Wake waves
 * disappear when they have radiated away and spread, which takes longer still —
 * a ship's transverse waves are legible for a while after the foam over them has
 * gone, which is the ordering this constant has to preserve.
 *
 * It was 3.4 s, chosen against a foam constant of 1.5. `WAKE_FOAM_TAU` is now 5,
 * so 3.4 would have inverted the ordering and left the water *foaming without
 * waving* along an old track. Only the history outside the fresh stamp is
 * affected — inside it the pattern is restamped every frame — so what this
 * actually governs is how long the phase a turning ship laid down survives once
 * the wedge has swung off it.
 */
const ELEVATION_DECAY_TAU = 6.5;

/**
 * Gravity, for the dispersion relation that sets the wake's wavelength.
 *
 * The wake is not decorative geometry with a tuned wavelength. A displacement
 * hull at speed V leaves a transverse wave system whose crests are stationary in
 * the hull's frame, which requires the wave's phase speed to equal V; in deep
 * water c = sqrt(g/k), so k = g/V². Everything about how the pattern scales with
 * speed follows from that one line, including the thing that makes a wake read as
 * a wake: go faster and the crests get *longer*, not just bigger.
 */
const GRAVITY = 9.81;

/**
 * Floor on V² in the dispersion relation, m²/s².
 *
 * k = g/V² diverges as the ship stops. Physically the waves genuinely do get
 * shorter, but below about 2 m/s their wavelength drops under a metre — a couple
 * of texels of this buffer — and all that survives resampling is aliasing. The
 * floor pins the wavelength at ~2.6 m there; the amplitude term below has already
 * faded the pattern to near nothing by that speed, so the pin is never visible.
 */
const MIN_SPEED_SQUARED = 4;

/**
 * Wake amplitude coefficient, metres per (m/s)².
 *
 * Wave-making resistance rises steeply with speed, so the crest height does too;
 * a linear-in-speed wake looks inert. Quadratic with a cap is the cheap stand-in
 * for the real curve, which flattens once the hull is at its own hull speed.
 *
 * Quadratic is also the only law that holds the wake's *steepness* fixed, which
 * is the thing that actually decides whether it is visible. Slope is `a * k`, and
 * `k` goes as `1/V²`, so `a` going as `V²` cancels it exactly: the divergent
 * system's steepness is `ELEVATION_PER_SPEED_SQUARED * DIVERGENT_WEIGHT *
 * DIVERGENT_K * GRAVITY` at every speed, which is 0.41 here. That number is the
 * real constraint on this constant, and it is why the arms cannot simply be
 * scaled up: waves fold at a steepness of about 0.44, and past that the wake
 * stops being a wake and becomes a crease in the mesh with a shading seam down
 * it. 0.41 is as steep as the arms can be and still be water.
 */
const ELEVATION_PER_SPEED_SQUARED = 0.0132;

/**
 * Cap on wake crest height, metres.
 *
 * At 0.62 the cap bound at 7 m/s — below the hull's own terminal speed — so the
 * quadratic law above was dead across the entire top third of the throttle and
 * the wake stopped answering the engine exactly where a viewer is most likely to
 * be looking for it. 0.95 binds at 8.5 m/s, just clear of terminal speed, so the
 * cap does what it is for (a bound on the arithmetic) rather than flattening the
 * curve it is bounding.
 */
const MAX_ELEVATION = 0.95;

/**
 * Relative amplitudes of the two wave systems.
 *
 * In a photograph of a hull at speed the arms are what you see, and the
 * transverse crests are the subtler thing between them: the divergent waves are
 * three times shorter for the same amplitude, so they carry three times the
 * slope, and slope is all a water surface shows. Weighting them 0.75/0.85 made
 * the two systems near enough equal in amplitude and therefore *unequal the wrong
 * way* in appearance — a wake with a strong ripple down the middle and a faint V.
 */
const TRANSVERSE_WEIGHT = 0.5;
const DIVERGENT_WEIGHT = 1.05;

/**
 * Divergent-system wave angle, radians, measured from the track.
 *
 * The Kelvin pattern is a superposition over wave angles from 0 to 90 degrees;
 * stationary phase picks out two families, and the divergent one is dominated by
 * angles near 55 degrees. Its wavenumber is k0/cos²(psi) — over three times the
 * transverse system's — which is why the feathered arms are visibly finer than
 * the crests running across the wake.
 */
const DIVERGENT_ANGLE = 0.96;
const DIVERGENT_COS = Math.cos(DIVERGENT_ANGLE);
const DIVERGENT_SIN = Math.sin(DIVERGENT_ANGLE);
const DIVERGENT_K = 1 / (DIVERGENT_COS * DIVERGENT_COS);

// ------------------------------------------------------------------ shore break

/**
 * Wave height over still-water depth at which a shoaling wave overturns.
 *
 * McCowan's limit for a solitary wave, and the number every surf-zone model
 * starts from. It is the whole breaking criterion: a wave of height H finds its
 * limit in `H / 0.78` metres of water, so the surf line is wherever the bottom
 * comes up to that depth. Everything else in this section is about how wide to
 * make the band and how hard to deposit inside it.
 */
const BREAKER_INDEX = 0.78;

/**
 * Shoaling amplification between deep water and the break point.
 *
 * A wave does not arrive at the bar with its offshore height. It slows as it
 * feels bottom, the energy it carries has to go somewhere, and by Green's law
 * `H ~ d^-1/4` it is a fifth or so taller by the time it overturns. Leaving this
 * out puts the surf line about a metre of depth too far inshore, which on a
 * steep face is the difference between a band and a rim.
 */
const SHOALING_GAIN = 1.2;

/**
 * Bounds on the break depth, metres.
 *
 * The cap is applied as `MAX * tanh(d/MAX)` rather than `min`, and the softness
 * is the point. Depth-limited breaking is real physics — a shelf can only
 * deliver `BREAKER_INDEX * depth` of wave height to the surf zone no matter what
 * is running at it from offshore, so past a certain sea state the breaker line
 * stops moving out. A hard `min` expresses that too, but it also kills the
 * response, and the response is most of what sells this: with the spectrum's
 * `Hs = 0.22 U^2/g` a hard cap at 8 m binds from 16 m/s upward, so the surf line
 * would sit dead still across the top third of the wind slider. `tanh` stays
 * monotonic all the way to a 13 m sea — measured along the island's windward
 * radial the break depth still walks from 1.2 m at 6 m/s to 7.9 m at 24, and the
 * outer edge of the white with it, 47 m further out at the top of the range than
 * the bottom — while never letting that edge past `MAX * SURF_OUTER_REACH`, 15 m
 * of water. That is the number that bounds the white region's width: divided by
 * the local bottom slope it *is* that width, and past 15 m there is no slope
 * gentle enough in this bathymetry for the result to still read as a line.
 *
 * The floor keeps a thread of white at the waterline on a glassy day, because
 * there is always a shorebreak.
 */
const SURF_DEPTH_MAX = 8;
const SURF_DEPTH_MIN = 0.5;

/**
 * Outer edge of the generating band, as a multiple of the break depth.
 *
 * The band has to be a gradient, not a wire, and the reason it is one is the
 * spectrum: `Hs` is the mean of the highest third, so the individual waves
 * arriving are Rayleigh-distributed about it and the largest in a few hundred is
 * roughly 1.9 times as tall. That wave breaks 1.9 times further out. Offshore of
 * the significant-wave contour is therefore the sets — thinning outward exactly
 * as the tail of the distribution does, and doing the thing a real surf line
 * does when it wanders seaward for a few waves and comes back.
 */
const SURF_OUTER_REACH = 1.9;

/**
 * How far seaward the shoaling test looks, metres, and the slopes it ramps over.
 *
 * Depth alone is not a breaking criterion, and assuming it was is what made the
 * first version of this term a sheet of paper: `band` is 1 at *every* depth
 * inshore of the break, so it whitened the whole of a shallow floor at once —
 * the lagoon behind the spit went solid, and over half the worst 420 m footprint
 * near the island was white at 15 m/s.
 *
 * What was missing is that waves break where they *shoal*. The energy that
 * becomes white water comes from the wave losing height as the bottom rises
 * under it, so the quantity that matters is the depth gradient along the
 * direction the swell is travelling, not the depth. One extra sample one
 * nearshore wavelength to seaward gives it: `(depthSeaward - depthHere) / reach`
 * is that slope, positive where the bottom is climbing toward the beach.
 *
 * Everything falls out of that one sample:
 *
 *  - a flat lagoon behind a bar reads zero slope and gets nothing;
 *  - a lee shore reads a *negative* slope, because seaward of it is the island
 *    it sits behind, so an island shelters its own back — which is both true and
 *    the single most convincing thing this term does;
 *  - a windward beach face reads a strong positive slope and breaks along its
 *    whole length, waterline included.
 *
 * With it, the worst 420 m footprint near the island goes from 55% white to 18%
 * at 15 m/s, and from 62% to 22% at 24. 55 m is roughly one shallow-water
 * wavelength for an 8 s swell over the surf zone. The ramp is wide and its exact
 * edges do not matter — 0.004 to 0.03 and 0.01 to 0.05 land within 0.3 points of
 * each other — because real beach faces sit an order of magnitude above it and
 * the floors this is rejecting sit at zero.
 */
const SHELTER_REACH = 55;
const SHOAL_INNER = 0.004;
const SHOAL_OUTER = 0.03;

/**
 * Foam deposited per second by fully-broken shore water.
 *
 * `SURF_RATE * WAKE_FOAM_TAU` is 1.4, which is above white — and unlike every
 * other deposit in this file, a texel in the impact zone really is exposed
 * indefinitely, because the band is pinned to the bottom instead of being towed
 * over the water by a hull. So this number is chosen knowing the equilibrium is
 * reached, and the surf band is *supposed* to reach white.
 *
 * What stops that from becoming the sheet of paper this file has twice been
 * burnt by is that nothing outside a band ever sees the full rate:
 *
 *  - `exposure` is zero wherever the bottom is not climbing toward the shore,
 *    which is what turns a shallow *region* into a shore-following *band*. See
 *    `SHELTER_REACH`; it is the load-bearing one.
 *  - `band` is zero beyond `SURF_OUTER_REACH` break depths, so the band's width
 *    on the ground is `SURF_OUTER_REACH * breakDepth / slope` and the
 *    *bathymetry* sets it, not this constant. Measured along the island's
 *    windward radial, the white runs 6 m wide at 6 m/s and 52 m at 24, always
 *    anchored at the waterline and growing seaward.
 *  - `wet` is zero on dry sand, so it cannot paint the beach.
 *  - `surge` and `cells` multiply to a mean near 0.45 and only approach 1 where a
 *    set is breaking through a boil, so the sustained level across the band is
 *    around 0.6 and only the cells go white.
 *  - the CPU gate holds the whole term at zero unless a shoreline is inside the
 *    footprint, which keeps it off the open-water plateau the whitecap coverage
 *    test measures.
 *
 * Together those hold the worst 420 m footprint anywhere near the island to 5%
 * white at 6 m/s and 22% at 24. Raising this constant does not widen any of
 * that, it only fills in the gaps between the cells — which is the same trade
 * the wedge lost twice, and the same answer.
 */
const SURF_RATE = 0.28;

/**
 * How much of the surf deposit the arriving swell modulates, versus a floor.
 *
 * All pulse and no floor and the beach blinks off between sets; all floor and no
 * pulse and it is a painted stripe. 0.55 leaves the band running at a bit under
 * half rate continuously with each set driving it to white, which against a 5 s
 * time constant reads as a band that breathes rather than one that flashes.
 */
const SURF_PULSE = 0.55;

/**
 * Fold values the swell modulation ramps between.
 *
 * Far looser than `uBreakThreshold`, and deliberately: that threshold asks "is
 * this water folding", which only a few percent of the sea ever is, while this
 * asks "is a crest arriving here", which is most of what a shore sees. Ramping
 * over the same tight window would have the surf pulse only on the rare texels
 * that whitecap on their own, which is not what makes a shore break.
 */
const SURF_FOLD_INNER = 0.15;
const SURF_FOLD_OUTER = 0.95;

/** World scale of the second `wakeBoil` octave, relative to the first. */
const SURF_CELL_SCALE = 0.31;

/**
 * Depth below which the CPU calls the footprint "coastal", metres.
 *
 * Fixed, rather than derived from the sea state, and that is the load-bearing
 * choice. A threshold that grew with the swell would switch the term — and its
 * per-texel evaluation of the seafloor field — on and off as the wind slider
 * moved, which is a frame-time cliff triggered by weather; worse, it would arm
 * the surf over open shelf that has no shoreline anywhere on it, including the
 * play area's own plateau, and that is the water Monahan's whitecap law is
 * measured against. 6 m is shallow enough that only real coast reaches it: the
 * shallowest water the gate sees at the camera's spawn is 13.7 m down, and no
 * centre within 260 m of the origin sees anything above 9.2 m, so there is a
 * clear factor of 1.5 between this threshold and the water the coverage test
 * reads.
 *
 * Conservative in the safe direction: a shoal whose crown never comes up to 6 m
 * gets no surf even if a big enough sea would technically break over it. The
 * alternative error is white water in open ocean.
 */
const SHORE_GATE_DEPTH = 6;

/**
 * Grid the gate samples the floor on, and how far past the footprint it reaches.
 *
 * 11 samples over 1.15 footprints is a 48 m stride, against a heightfield whose
 * finest octave is a ~25 m cell riding on shelf and bar features hundreds of
 * metres across. Checked against a 4 m dense scan of the same footprint at 1576
 * centres over the island: no misses, and 149 armed early, which is the margin
 * doing its job. The margin is what arms the term *before* the shore crosses
 * into the buffer, so the band has the whole approach to settle instead of
 * appearing at the edge of view.
 *
 * 121 evaluations of the CPU heightfield per frame — 5.7 us, measured, and the
 * entire cost of this feature in open water. Near a shore it early-outs sooner
 * and the GPU branch is what starts costing anything.
 */
const SHORE_GATE_SAMPLES = 11;
const SHORE_GATE_MARGIN = 1.15;

/**
 * Sea state assumed until `setSwell` is called: height in metres, bearing in
 * radians on the same `(cos, sin) -> (x, z)` convention as `Spectrum`.
 *
 * Non-zero height so the band exists as soon as a floor node is wired rather
 * than being a silent no-op until the caller notices the setter, and pi/4 to
 * match `DEFAULT_SPECTRUM.windDirection`, so an unwired default has the surf on
 * the same face of the island the swell is actually running at.
 */
const DEFAULT_SWELL_HEIGHT = 1.5;
const DEFAULT_SWELL_BEARING = Math.PI * 0.25;

/** Cascade slots the accumulate pass is built with. Matches `OceanMaterial`. */
const WAKE_CASCADES = 3;

/**
 * TSL node objects are structurally dynamic; the generated types cannot express
 * a uniform whose component type is only known at construction. Node-typed
 * fields are therefore `any` by design — the class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
interface EmitterUniforms {
  /** vec2 world position. */
  position: any;
  /** vec2 unit heading. */
  direction: any;
  /** vec4: x = deposit amount, y = hull width, z = arm length, w = speed (m/s). */
  params: any;
}

export interface WakeOptions {
  /**
   * Metres of water above the floor at a world position, as a TSL node.
   *
   * Same shape and same contract as `Seafloor.depthNode` — takes a vec3 world
   * position, reads xz, returns a float that goes negative over dry land. Bind
   * it and the shore-break term is compiled into the accumulate pass; omit it
   * and the term is not built at all, so an unwired `Wake` costs exactly what it
   * did before and renders exactly what it did before.
   */
  floorDepthNode?: ((worldPosition: any) => any) | null;
}

export class Wake {
  /** Foam accumulation texture. Stable reference — safe to bind once. */
  readonly texture: THREE.Texture;

  /** World-space size of the square the texture covers. */
  readonly extent: number;

  /** Add to the scene to make `setDebugVisible` do anything. */
  readonly debugObject: THREE.Object3D;

  /** Texture resolution per side. Updated by `setResolution` without replacing textures. */
  resolution: number;

  private readonly buffers: [THREE.RenderTarget, THREE.RenderTarget];
  private readonly output: THREE.RenderTarget;
  private readonly quad = new THREE.QuadMesh();

  private readonly accumulate: [THREE.NodeMaterial, THREE.NodeMaterial];
  private readonly copy: [THREE.NodeMaterial, THREE.NodeMaterial];
  private index = 0;

  private readonly uScroll = uniform(new THREE.Vector2());
  /**
   * Per-frame drift of the foam field, in uv, from the surface current.
   *
   * Foam floats. It does not sit where it was made: it is carried by the wind
   * drift and the wave orbital motion, which together move surface material
   * downwind at roughly 3% of the wind speed — the classic Stokes-drift-plus-
   * wind-drift figure. Without it the accumulation buffer is a stamp album, and
   * a patch of whitecap sits motionless on water that is visibly moving under it,
   * which is one of the strongest tells that foam is being painted rather than
   * transported.
   *
   * Applied as an offset on the *source* lookup, so it costs nothing: the pass
   * already resamples the previous frame to follow the moving centre.
   */
  private readonly uDrift = uniform(new THREE.Vector2());
  private readonly uDecay = uniform(1);
  /** Per-frame decay multiplier for the elevation channel. See `ELEVATION_DECAY_TAU`. */
  private readonly uElevationDecay = uniform(1);
  /** Per-frame decay multiplier for the hull-foam channel. See `WAKE_FOAM_TAU`. */
  private readonly uWakeFoamDecay = uniform(1);
  private readonly uCenter = uniform(new THREE.Vector2());
  /** Frame step, seconds, so deposits are a rate rather than a per-frame amount. */
  private readonly uStep = uniform(1 / 60);
  /**
   * Jacobian below which the surface counts as breaking.
   *
   * Low, because folding is rare: the measured sea state carries a mean Jacobian
   * of 0.86/0.95/0.98 per cascade and only about 0.1% of the surface is actually
   * folded at any instant, against a few percent whitecap coverage. A threshold
   * loose enough to catch "nearly folding" catches most of the ocean.
   */
  private readonly uBreakThreshold = uniform(0.14);
  /**
   * Foam deposited per second by fully-broken water.
   *
   * The value that matters is `rate * DECAY_TAU`, which is the coverage this
   * settles at under continuous breaking — an equilibrium, not a per-frame
   * amount. At 2.6 that product was 4.4 and clamped to 1, so anything that broke
   * even weakly saturated to solid white within a frame or two.
   *
   * 0.55 against a 1.5 s time constant settles at 0.83. That is deliberately well
   * short of white: a texel only reaches the equilibrium if it stays under a
   * breaking crest indefinitely, which nothing does, and the surface's own
   * breakup noise then decides how much of that reads as bubbles. Setting it so a
   * *continuously* breaking texel goes white leaves no headroom for the far more
   * common case of a crest passing over once.
   */
  private readonly uBreakRate = uniform(0.55);
  /**
   * Unit wind axis in world XZ, the direction the sea runs toward.
   *
   * Only the windward-face term uses it. `uDrift` cannot stand in: drift is a
   * velocity and goes to zero in a calm, and an axis that vanishes would swing
   * the foam bias through every heading on the way down.
   */
  private readonly uWindAxis = uniform(new THREE.Vector2(1, 0));
  /**
   * Foam per second deposited by a fully wind-facing wave face.
   *
   * A separate term from `uBreakRate`, and it has to be, because it answers a
   * different question. The fold asks "is this water overturning", which is true
   * of a fraction of a percent of the sea; this asks "is the wind driving up
   * this face", which is true of about half of it. So the rate is far lower and
   * what it produces is not whitecap but the faint aeration that a wind-driven
   * face carries before it breaks — and, once persistence has had it for a
   * second, the foam that is still there after the crest has passed over it.
   *
   * That last part is the point. Depositing only on the fold puts foam exactly
   * on the crest line and nowhere else, so a wave has a white top and clean
   * water either side of it. Real wind seas carry foam *up* the face and leave
   * it *behind* the crest, because the water that was at the crest a moment ago
   * is now on the back of it. Injecting on the rising face and letting the
   * existing decay carry it is the cheapest way to say that: no new buffer, no
   * new pass, one dot product in a shader that already samples the slope.
   *
   * 0.06 against the 1.5 s sea-foam constant settles at 0.09 — a wash, an order
   * of magnitude below what breaking deposits, which is the correct relationship
   * between "the wind is on this face" and "this face is overturning".
   */
  private readonly uWindwardRate = uniform(0.06);
  /**
   * Slope, in metres of rise per metre, at which a face counts as fully facing
   * the wind.
   *
   * 0.25 is a little under the 0.44 steepness at which waves fold, so the term
   * saturates on faces that are steep but not breaking — which is exactly the
   * band the fold term cannot see.
   */
  private readonly uWindwardSlope = uniform(0.25);
  /** Rain rate, 0..1. */
  private readonly uRainAgitation = uniform(0);
  /** Foam per second deposited by rain at full intensity. */
  private readonly uRainRate = uniform(0.22);

  /**
   * Still-water depth the significant wave breaks in, metres. See `BREAKER_INDEX`.
   *
   * A uniform rather than a constant because it is the whole coupling between the
   * sea state and the shore: raise the swell and this grows, and the surf line
   * walks offshore along the bathymetry instead of the band simply getting
   * brighter where it already was.
   */
  private readonly uSurfDepth = uniform(1);
  /**
   * 1 when the buffer's footprint contains a shoreline, 0 otherwise.
   *
   * A uniform, so the branch it guards is coherent across the entire draw and
   * the hardware skips the body rather than executing both sides — which matters
   * because the body is sixteen texture fetches of the seafloor's FBM per texel,
   * over a million texels, every frame. The CPU owns the buffer's centre, so the
   * CPU is what answers the question. See `SHORE_GATE_DEPTH`.
   */
  private readonly uShoreActive = uniform(0);
  /**
   * World-space offset, metres, from a texel to the point the shoaling test
   * samples: `SHELTER_REACH` seaward, against the swell's direction of travel.
   *
   * Carried as a vector rather than as a bearing so the shader adds it directly
   * to the world position — the trigonometry is the same on every texel of every
   * frame, and there are a million of them.
   */
  private readonly uSeaward = uniform(new THREE.Vector2());
  /** See `WakeOptions.floorDepthNode`. Null means the term was never built. */
  private readonly shoreDepthNode: ((worldPosition: any) => any) | null;
  private swellHeight = DEFAULT_SWELL_HEIGHT;
  private swellBearing = DEFAULT_SWELL_BEARING;

  /**
   * Wave derivative bindings for the breaking-crest term.
   *
   * Built once for the maximum cascade count and re-pointed by `setCascades`,
   * for the same reason `OceanMaterial` does it: a tier change recreates the
   * simulation's targets, and rebuilding this material to follow them would
   * recompile a shader mid-session and leak what it replaced.
   */
  private readonly derivativeNodes: any[] = [];
  private readonly uTileSizes: any[] = [];
  private readonly uCascadeWeights: any[] = [];
  private readonly emitters: EmitterUniforms[] = [];

  /** Pending emissions: [x, z, heading, speed, width] per slot. */
  private readonly queue = new Float32Array(MAX_EMITTERS * 5);
  private queued = 0;

  private driftX = 0;
  private driftZ = 0;

  private centerX_ = 0;
  private centerZ_ = 0;
  private appliedX = 0;
  private appliedZ = 0;

  private readonly debugMesh: THREE.Mesh;
  private readonly debugGeometry: THREE.PlaneGeometry;
  private readonly debugMaterial: THREE.MeshBasicNodeMaterial;
  private disposed = false;

  /**
   * @param waves Derivative fields and tile sizes, for the breaking-crest term.
   *
   * 1024 rather than 512 by default: the buffer now carries whitecaps as well as
   * the wake, and whitecap edges are decimetre features. Over a 420 m footprint
   * 512 is 0.82 m per texel, which turns a crest streak into a smear.
   */
  constructor(
    waves: { derivativeTextures: THREE.Texture[]; tileSizes: number[] },
    resolution = 1024,
    extent = 420,
    options: WakeOptions = {},
  ) {
    this.resolution = normalizeResolution(resolution);
    this.extent = extent;
    // Read before the accumulate materials are built: whether the shore-break
    // term exists is baked into their node graph, exactly as the cascade count
    // is, and for the same reason — a material rebuilt later recompiles a shader
    // mid-session and leaks the one it replaced.
    this.shoreDepthNode = options.floorDepthNode ?? null;

    for (let i = 0; i < WAKE_CASCADES; i++) {
      const source = Math.min(i, waves.derivativeTextures.length - 1);
      this.derivativeNodes.push(texture(waves.derivativeTextures[source]) as any);
      this.uTileSizes.push(uniform(waves.tileSizes[source]));
      this.uCascadeWeights.push(uniform(i < waves.derivativeTextures.length ? 1 : 0));
    }

    this.buffers = [makeTarget(this.resolution), makeTarget(this.resolution)];
    this.output = makeTarget(this.resolution);
    this.texture = this.output.texture;

    for (let i = 0; i < MAX_EMITTERS; i++) {
      this.emitters.push({
        position: uniform(new THREE.Vector2()),
        direction: uniform(new THREE.Vector2(1, 0)),
        params: uniform(new THREE.Vector4()),
      });
    }

    this.accumulate = [
      this.createAccumulateMaterial(this.buffers[0].texture),
      this.createAccumulateMaterial(this.buffers[1].texture),
    ];
    this.copy = [
      createCopyMaterial(this.buffers[0].texture),
      createCopyMaterial(this.buffers[1].texture),
    ];

    this.debugGeometry = new THREE.PlaneGeometry(extent, extent);
    this.debugGeometry.rotateX(-Math.PI / 2);
    this.debugMaterial = new THREE.MeshBasicNodeMaterial();
    this.debugMaterial.transparent = true;
    this.debugMaterial.depthWrite = false;
    this.debugMaterial.toneMapped = false;
    this.debugMaterial.colorNode = Fn(() => {
      const sampled = texture(this.texture, uv()).toVar();
      const foam = sampled.r.clamp(0, 1).toVar();
      // Elevation is signed and small; scaled so a 0.3 m crest reads clearly.
      const crest = sampled.g.mul(3).clamp(0, 1).toVar();
      const trough = sampled.g.mul(-3).clamp(0, 1).toVar();
      // Magenta foam so the overlay can never be mistaken for foam itself, with
      // crests in green and troughs in blue over it.
      return vec4(
        foam.mul(1.0).add(trough.mul(0.1)),
        foam.mul(0.25).add(crest.mul(0.9)),
        foam.mul(0.7).add(trough.mul(0.9)),
        foam.mul(0.85).add(crest.max(trough).mul(0.6)).add(0.06),
      );
    })();

    this.debugMesh = new THREE.Mesh(this.debugGeometry, this.debugMaterial);
    this.debugMesh.name = 'wake-debug';
    this.debugMesh.frustumCulled = false;
    this.debugMesh.renderOrder = 20;

    this.debugObject = new THREE.Group();
    this.debugObject.name = 'wake-probes';
    this.debugObject.visible = false;
    this.debugObject.add(this.debugMesh);
    this.debugMesh.position.y = 3;
  }

  /**
   * Resizes the three wake targets in place.
   *
   * `RenderTarget.setSize` releases the old GPU allocations, but it preserves
   * each `Texture` object. That distinction matters here: `wake.texture` is
   * already bound into the water node graph, so replacing the output target
   * would leave the graph sampling a disposed texture. The application must
   * call this after its submitted-work fence, because `setSize` necessarily
   * retires the old GPU resources.
   */
  setResolution(resolution: number): void {
    if (this.disposed) return;
    const next = normalizeResolution(resolution);
    if (next === this.resolution) return;

    for (const target of [this.buffers[0], this.buffers[1], this.output]) {
      target.setSize(next, next);
    }
    this.resolution = next;
    this.queued = 0;
  }

  /** Re-points the wave bindings after a tier change. See `derivativeNodes`. */
  setCascades(derivativeTextures: THREE.Texture[], tileSizes: number[]): void {
    const active = Math.min(derivativeTextures.length, WAKE_CASCADES);
    for (let i = 0; i < WAKE_CASCADES; i++) {
      const source = Math.min(i, active - 1);
      this.derivativeNodes[i].value = derivativeTextures[source];
      this.uTileSizes[i].value = tileSizes[source];
      this.uCascadeWeights[i].value = i < active ? 1 : 0;
    }
  }

  /**
   * Surface drift, m/s in world XZ, that foam is carried by.
   *
   * Wind drift plus Stokes drift is about 3% of the wind speed downwind; the
   * caller is expected to have applied that factor rather than passing the wind.
   */
  setDrift(x: number, z: number): void {
    this.driftX = x;
    this.driftZ = z;
  }

  /**
   * Bearing the sea runs toward, `atan2(dz, dx)` — the same convention and the
   * same number as `SpectrumParams.windDirection`.
   *
   * Kept apart from `setDrift` because that one takes a velocity and this one
   * takes a heading, and the heading has to survive the velocity going to zero.
   */
  setWindAxis(bearingRadians: number): void {
    const axis = this.uWindAxis.value as THREE.Vector2;
    axis.set(Math.cos(bearingRadians), Math.sin(bearingRadians));
  }

  /**
   * Foam deposited per second on a fully wind-facing face, and the slope at
   * which a face counts as fully facing. See `uWindwardRate`.
   */
  setWindward(rate: number, slope = 0.25): void {
    this.uWindwardRate.value = Math.max(0, rate);
    this.uWindwardSlope.value = Math.max(0.01, slope);
  }

  /** Rain rate, 0..1, driving the agitation term. */
  setRainAgitation(intensity: number): void {
    this.uRainAgitation.value = Math.max(0, Math.min(1, intensity));
  }

  /** Tunes how readily the surface is treated as breaking, and how fast it foams. */
  setBreaking(threshold: number, rate: number): void {
    this.uBreakThreshold.value = threshold;
    this.uBreakRate.value = Math.max(0, rate);
  }

  /**
   * The swell that is running at the shore: significant height in metres, and
   * the bearing it travels along, `atan2(dz, dx)` — the same convention and the
   * same number as `SpectrumParams.windDirection`.
   *
   * The height sets where the sea finds bottom and therefore how far offshore
   * the surf line stands (`BREAKER_INDEX`); the bearing decides which face of a
   * headland gets the surf and which one is in its lee (`SHELTER_REACH`).
   * `significantWaveHeight(spectrumParams)` is the height this expects, the same
   * value `AudioSystem` is given. No effect unless a floor node was supplied at
   * construction.
   */
  setSwell(significantHeight: number, bearingRadians: number): void {
    this.swellHeight = Number.isFinite(significantHeight) ? Math.max(0, significantHeight) : 0;
    if (Number.isFinite(bearingRadians)) this.swellBearing = bearingRadians;
  }

  /**
   * The resolved foam target, for CPU readback.
   *
   * Exposed so a test can measure the *rendered* whitecap coverage against the
   * empirical law that is supposed to be generating it. Monahan drove the deposit
   * rate for a long time with nothing checking the result, and the reference
   * image was rendering roughly ten times the coverage the law predicts.
   */
  get outputTarget(): THREE.RenderTarget {
    return this.output;
  }

  /** Recentres the world footprint. Contents are resampled on the next update. */
  setCenter(x: number, z: number): void {
    this.centerX_ = x;
    this.centerZ_ = z;
  }

  /** Current world centre. Scalars, not a vector, so reading it per frame is free. */
  get centerX(): number {
    return this.centerX_;
  }

  get centerZ(): number {
    return this.centerZ_;
  }

  /**
   * Queues a wake deposit.
   *
   * `headingRadians` is `atan2(forward.z, forward.x)` — the direction the hull
   * is pointing, which is also what `Ship.heading` returns. `width` is the beam;
   * the wedge is built outward from it.
   */
  emit(x: number, z: number, headingRadians: number, speed: number, width: number): void {
    if (this.queued >= MAX_EMITTERS) return;
    if (!Number.isFinite(x) || !Number.isFinite(z) || !Number.isFinite(headingRadians)) return;
    if (!(speed > 0.05)) return;

    const base = this.queued * 5;
    this.queue[base] = x;
    this.queue[base + 1] = z;
    this.queue[base + 2] = headingRadians;
    this.queue[base + 3] = speed;
    this.queue[base + 4] = Math.max(0.5, width);
    this.queued++;
  }

  setDebugVisible(v: boolean): void {
    this.debugObject.visible = v;
  }

  /** Converts a world point to this buffer's uv, for the water shader. */
  uvAt(x: number, z: number, out: THREE.Vector2): THREE.Vector2 {
    return out.set((x - this.centerX_) / this.extent + 0.5, (z - this.centerZ_) / this.extent + 0.5);
  }

  update(dt: number, renderer: THREE.WebGPURenderer): void {
    if (this.disposed) return;
    const step = Math.min(Math.max(dt, 0), 0.1);

    setVec2(
      this.uScroll.value as THREE.Vector2,
      (this.centerX_ - this.appliedX) / this.extent,
      (this.centerZ_ - this.appliedZ) / this.extent,
    );
    setVec2(
      this.uDrift.value as THREE.Vector2,
      (this.driftX * step) / this.extent,
      (this.driftZ * step) / this.extent,
    );
    this.uDecay.value = Math.exp(-step / DECAY_TAU);
    this.uElevationDecay.value = Math.exp(-step / ELEVATION_DECAY_TAU);
    this.uWakeFoamDecay.value = Math.exp(-step / WAKE_FOAM_TAU);
    this.uStep.value = step;
    setVec2(this.uCenter.value as THREE.Vector2, this.centerX_, this.centerZ_);

    if (this.shoreDepthNode !== null) {
      const raw = (this.swellHeight * SHOALING_GAIN) / BREAKER_INDEX;
      this.uSurfDepth.value = Math.max(
        SURF_DEPTH_MIN,
        SURF_DEPTH_MAX * Math.tanh(raw / SURF_DEPTH_MAX),
      );
      // Negated: the bearing is where the swell is going, and the shoaling test
      // has to look at where it came from.
      setVec2(
        this.uSeaward.value as THREE.Vector2,
        -Math.cos(this.swellBearing) * SHELTER_REACH,
        -Math.sin(this.swellBearing) * SHELTER_REACH,
      );
      // Recomputed every frame from the centre alone, with no hysteresis and no
      // cached scan. It is 121 heightfield samples, and making it cheaper by
      // remembering the last answer would make the gate a function of the path
      // taken to get here rather than of where the buffer is — which `reset`
      // plus a settle has to be able to reproduce exactly.
      this.uShoreActive.value = this.shoreInFootprint() ? 1 : 0;
    }

    for (let i = 0; i < MAX_EMITTERS; i++) {
      const slot = this.emitters[i];
      const params = slot.params.value as THREE.Vector4;
      if (i >= this.queued) {
        // Speed 0 in `w` is what switches the elevation stamp off for this slot;
        // the foam amount in `x` does the same for the foam term.
        params.set(0, 1, 1, 0);
        continue;
      }
      const base = i * 5;
      const heading = this.queue[base + 2];
      const speed = this.queue[base + 3];
      const width = this.queue[base + 4];

      setVec2(slot.position.value as THREE.Vector2, this.queue[base], this.queue[base + 1]);
      setVec2(slot.direction.value as THREE.Vector2, Math.cos(heading), Math.sin(heading));

      const intensity = Math.min(1.4, speed / REFERENCE_SPEED);
      // Arm length in beams, and longer than it was: the arms have to outlast the
      // turbulent band or the wedge ends before the churn inside it does, which
      // reads as a wake that has been cut off square. At 8 m/s this is about
      // 16 beams — a little over four ship lengths, which is also roughly where
      // `WAKE_FOAM_TAU` has taken the band down to nothing.
      params.set(DEPOSIT_RATE * step * intensity, width, width * (6 + intensity * 9), speed);
    }
    this.queued = 0;

    const previous = renderer.getRenderTarget();

    this.quad.material = this.accumulate[this.index];
    renderer.setRenderTarget(this.buffers[1 - this.index]);
    this.quad.render(renderer);
    this.index = 1 - this.index;

    // Resolve into a target whose texture reference never changes, so material
    // authors can bind `wake.texture` once at build time.
    this.quad.material = this.copy[this.index];
    renderer.setRenderTarget(this.output);
    this.quad.render(renderer);

    renderer.setRenderTarget(previous);

    this.appliedX = this.centerX_;
    this.appliedZ = this.centerZ_;
    this.debugObject.position.set(this.centerX_, 0, this.centerZ_);
  }

  /**
   * Clears the accumulation to empty water and re-anchors it at the current
   * centre.
   *
   * Foam persists for several seconds by design, so without this a capture would
   * carry in whatever the previous shot deposited. Both ping-pong buffers and the
   * resolved output are cleared, because the next `update` reads one of them.
   */
  reset(renderer: THREE.WebGPURenderer): void {
    if (this.disposed) return;

    const previousTarget = renderer.getRenderTarget();
    const previousClear = renderer.getClearColor(new THREE.Color());
    const previousAlpha = renderer.getClearAlpha();

    renderer.setClearColor(0x000000, 1);
    for (const target of [this.buffers[0], this.buffers[1], this.output]) {
      renderer.setRenderTarget(target);
      renderer.clear(true, false, false);
    }
    renderer.setRenderTarget(previousTarget);
    renderer.setClearColor(previousClear, previousAlpha);

    // Scroll compensation is a delta against the last applied centre; leaving it
    // stale would resample the freshly cleared buffer by an arbitrary offset.
    this.appliedX = this.centerX_;
    this.appliedZ = this.centerZ_;
    this.queued = 0;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.buffers[0].dispose();
    this.buffers[1].dispose();
    this.output.dispose();
    for (const material of this.accumulate) material.dispose();
    for (const material of this.copy) material.dispose();
    this.debugGeometry.dispose();
    this.debugMaterial.dispose();
    this.debugObject.removeFromParent();
    this.quad.geometry.dispose();
  }

  // ------------------------------------------------------------------ internals

  /**
   * Whether any water in (a little more than) the footprint is shore-shallow.
   *
   * Queries the same heightfield the floor mesh is built from, so the answer
   * follows whatever bathymetry that file produces without this one knowing
   * anything about the island's position or shape. No allocation: scalars only.
   */
  private shoreInFootprint(): boolean {
    const span = this.extent * SHORE_GATE_MARGIN;
    const stride = span / (SHORE_GATE_SAMPLES - 1);
    const originX = this.centerX_ - span * 0.5;
    const originZ = this.centerZ_ - span * 0.5;
    for (let iz = 0; iz < SHORE_GATE_SAMPLES; iz++) {
      const z = originZ + iz * stride;
      for (let ix = 0; ix < SHORE_GATE_SAMPLES; ix++) {
        // Dry land reports 0, which is below any threshold — so a beach arms the
        // gate for the same reason a shoal does, without a separate test.
        if (seafloorDepth(originX + ix * stride, z) < SHORE_GATE_DEPTH) return true;
      }
    }
    return false;
  }

  private createAccumulateMaterial(source: THREE.Texture): THREE.NodeMaterial {
    const extent = this.extent;
    const material = new THREE.NodeMaterial();
    material.depthTest = false;
    material.depthWrite = false;

    material.fragmentNode = Fn(() => {
      const coord = uv().toVar();
      const sourceUv = coord.add(this.uScroll).sub(this.uDrift).toVar();

      // Anything scrolled in from outside the previous footprint is unknown, and
      // clamp-to-edge would smear the border across the new region. Mask it.
      const edge = sourceUv.min(sourceUv.oneMinus()).toVar();
      const inside = edge.x.min(edge.y).smoothstep(0, 0.004).toVar();

      // Foam is read from the *drifted* lookup; elevation is not.
      //
      // Foam is material floating on the surface, so advecting it is the whole
      // point. Elevation is a phase field — the crest pattern a hull leaves —
      // and phase is not carried by the surface drift; a wake's waves propagate
      // at their own group velocity and stay where the hull put them for far
      // longer than a 3%-of-wind current would leave them. Sharing one lookup
      // between the two channels quietly dragged the wake downwind.
      const history = texture(source, sourceUv).toVar();
      const historyStill = texture(source, coord.add(this.uScroll)).toVar();
      const previous = history.r.mul(this.uDecay).mul(inside).toVar();
      // Hull foam rides the same drift — it floats, and it is the same material —
      // but decays on its own, much longer, constant. Same tap, no extra fetch.
      const previousWake = history.b.mul(this.uWakeFoamDecay).mul(inside).toVar();
      // Masked by *its own* footprint test, not the drifted one. Sharing `inside`
      // meant the drift decided where the elevation field's border was, which is
      // the coupling this split exists to remove.
      const edgeStill = coord.add(this.uScroll).toVar();
      const stillEdge = edgeStill.min(edgeStill.oneMinus()).toVar();
      const insideStill = stillEdge.x.min(stillEdge.y).smoothstep(0, 0.004).toVar();
      const previousElevation = historyStill.g
        .mul(this.uElevationDecay)
        .mul(insideStill)
        .toVar();

      const world = coord.sub(0.5).mul(extent).add(this.uCenter).toVar();
      /** Fresh foam from the sea itself — breaking crests and rain. */
      const deposit = float(0).toVar();
      /** Fresh foam from the hulls, kept apart for its own time constant. */
      const wakeDeposit = float(0).toVar();
      /** Fresh Kelvin elevation, metres, summed over emitters. */
      const stamped = float(0).toVar();
      /** How completely the fresh stamp owns this texel, 0..1. */
      const stampWeight = float(0).toVar();

      // --- breaking crests ----------------------------------------------------
      //
      // Whitecaps deposit into the same buffer as the wake, and for the same
      // reason they belong in a buffer at all: foam is *history*. Air entrained
      // by a wave that broke two seconds ago is still on the water, drifting and
      // dissolving, long after the wave itself has moved on.
      //
      // Evaluating the fold per fragment each frame — which is what the surface
      // used to do — cannot express that. It can only ever show where the water
      // is folding *now*, so foam appears and vanishes with the wave instead of
      // being left behind by it, and to read as continuous at all it has to be
      // spread far more widely than real whitecaps are. That is why the near
      // field was a third white.
      //
      // Here the same fold drives a *rate*, and persistence and dissipation are
      // left to the accumulation. Coverage can then be sparse and still read as
      // foam, because what the eye integrates is the trail, not the instant.
      const fold = float(1).toVar();
      // Surface slope, summed over the active cascades. The fold is a running
      // minimum because a texel breaks if *any* band folds it; slope is a sum
      // because the bands displace the same water and their gradients add.
      const slope = vec2(0, 0).toVar();
      for (let i = 0; i < WAKE_CASCADES; i++) {
        const d = this.derivativeNodes[i].sample(world.div(this.uTileSizes[i])).toVar();
        // Unused cascades are weighted to 1 — the neutral value for a running
        // minimum — rather than to 0, which would read as maximal folding
        // everywhere and paint the whole ocean white.
        fold.assign(fold.min(mix(float(1), d.z, this.uCascadeWeights[i])));
        slope.addAssign(d.xy.mul(this.uCascadeWeights[i]));
      }
      // Only genuinely folding water breaks. The threshold is deliberately
      // tighter than the old per-frame mask could afford to be.
      const breaking = smoothstepDown(fold, this.uBreakThreshold.sub(0.22), this.uBreakThreshold);
      deposit.addAssign(breaking.mul(this.uBreakRate).mul(this.uStep));

      // --- windward faces -----------------------------------------------------
      //
      // Positive slope along the wind axis is the face rising toward the crest
      // as you travel downwind — the face the wind is pushing on, and the one a
      // wind sea aerates before it breaks. Negative is the lee face, which gets
      // nothing here and gets its foam a second later when the crest that was
      // upwind of it has passed over and the persistence buffer still holds what
      // was deposited then. That handover is the whole mechanism; see
      // `uWindwardRate`.
      const windward = slope
        .dot(this.uWindAxis)
        .div(this.uWindwardSlope)
        .clamp(0, 1)
        .toVar();
      deposit.addAssign(windward.mul(this.uWindwardRate).mul(this.uStep));

      // Rain agitation. Heavy rain aerates a surface on its own — it goes white
      // in a downpour whether or not the waves are steep enough to break — so
      // this is a separate, unconditional contribution rather than a bias on the
      // fold threshold. Broken up so it reads as a stipple rather than a wash.
      const stipple = rainStipple(world.mul(0.55)).toVar();
      deposit.addAssign(
        this.uRainAgitation.mul(stipple.mul(0.7).add(0.3)).mul(this.uRainRate).mul(this.uStep),
      );

      // Structure for the turbulent band astern and for the surf line, evaluated
      // once because it depends only on where this texel is in the world. Both
      // are churned white water and both need the same metre-scale break-up. See
      // `wakeBoil`.
      const boil = wakeBoil(world).mul(0.35).add(0.75).toVar();

      // --- shore break --------------------------------------------------------
      //
      // The sea going white where it meets land. Two things decide where, and
      // both are properties of the bottom rather than of the island:
      //
      //  - a wave overturns once its height reaches `BREAKER_INDEX` times the
      //    depth under it, so the surf line is the `depth = height / gamma`
      //    contour, and it walks offshore when the swell gets up;
      //  - a wave only gets there by *shoaling*, so what breaks is the face the
      //    swell is climbing, not every shallow patch. See `SHELTER_REACH`.
      //
      // It is expressed here and not in the water shader because what a shore
      // actually produces is a *field of foam* — standing between sets, carried
      // in by the drift — and that is what this buffer is. The surface can only
      // ever draw where the water is white at this instant.
      //
      // Deposited into the hull-foam channel rather than the sea's, and for the
      // reason the two channels exist at all. Surf is deeply aerated — a bore
      // drives air metres down the way a transom does, not the shallow film a
      // whitecap entrains — and against the whitecap's 1.5 s the band went out
      // between sets instead of standing on the beach through them.
      if (this.shoreDepthNode !== null) {
        const floorDepth = this.shoreDepthNode;
        If(this.uShoreActive.greaterThan(0.5), () => {
          const depth = floorDepth(vec3(world.x, 0, world.y)).toVar();

          // Inner edge at the depth the significant wave breaks in, outer edge
          // where the biggest wave of a set does. Full strength everywhere
          // inshore of the break is not an oversight: past the bar the wave has
          // already broken and the whole surf zone is white water.
          const band = smoothstepDown(
            depth,
            this.uSurfDepth,
            this.uSurfDepth.mul(SURF_OUTER_REACH),
          ).toVar();

          // Which is exactly why the waterline needs its own cut. `band` is 1 at
          // every depth below the break including the negative ones over dry
          // sand, and without this the beach itself would be painted white.
          const wet = depth.smoothstep(-0.15, 0.5).toVar();

          // The shoaling test, and the second and last evaluation of the floor
          // field in this pass. `uSeaward` already carries the direction and the
          // reach, so this is one add and one subtract on top of the fetch.
          const seaward = floorDepth(
            vec3(world.x.add(this.uSeaward.x), 0, world.y.add(this.uSeaward.y)),
          ).toVar();
          const exposure = seaward
            .sub(depth)
            .div(SHELTER_REACH)
            .smoothstep(SHOAL_INNER, SHOAL_OUTER)
            .toVar();

          // The sets. `fold` is already this texel's steepness in the open-water
          // wave field, so borrowing it costs nothing and ties each pulse of the
          // band to a crest the viewer can watch arrive. The floor under it is
          // what keeps the surf running between sets. See `SURF_FOLD_INNER`.
          const surge = smoothstepDown(fold, SURF_FOLD_INNER, SURF_FOLD_OUTER)
            .mul(SURF_PULSE)
            .add(1 - SURF_PULSE)
            .toVar();

          // Two scales of `wakeBoil`, because a surf line is a row of breaking
          // cells with gaps between them and one scale gives an even stipple —
          // which is the tell that a band was drawn rather than broken. Sines
          // rather than a hash for the reason `wakeBoil` records: this buffer is
          // resampled by a fractional texel offset every frame, and a hash would
          // scintillate along the entire beach.
          const cells = boil.mul(wakeBoil(world.mul(SURF_CELL_SCALE)).mul(0.4).add(0.68)).toVar();

          wakeDeposit.addAssign(
            band.mul(wet).mul(exposure).mul(surge).mul(cells).mul(SURF_RATE).mul(this.uStep),
          );
        });
      }

      for (let i = 0; i < MAX_EMITTERS; i++) {
        const slot = this.emitters[i];
        const amount = slot.params.x;
        const width = slot.params.y;
        const armLength = slot.params.z;

        const delta = world.sub(slot.position).toVar();
        const forward = slot.direction;
        // Distance astern (positive behind the hull) and lateral offset.
        const along = delta.dot(forward).negate().toVar();
        const lateral = delta.dot(vec2(forward.y.negate(), forward.x)).abs().toVar();

        // Distances from the two ends of the hull, which is where everything a
        // ship does to water actually happens. See `HALF_LENGTH_BEAMS`.
        const fromStem = along.add(width.mul(HALF_LENGTH_BEAMS)).toVar();
        const fromTransom = along.sub(width.mul(HALF_LENGTH_BEAMS)).toVar();

        // The wedge, with its apex at the *stem*. The Kelvin half-angle is
        // measured from the disturbance that makes the pattern, and for a
        // displacement hull that is the bow: hanging the wedge off midships put
        // its apex half a hull length too far aft, which both narrowed the visible
        // V at the quarters and made the arms look like they were being emitted by
        // the middle of the ship rather than cut by the stem.
        const arm = width.mul(0.25).add(fromStem.mul(KELVIN_SLOPE)).toVar();
        const armWidth = width.mul(0.22).add(fromStem.mul(0.045)).max(0.4).toVar();
        const armOffset = lateral.sub(arm).div(armWidth).toVar();
        const armFoam = armOffset.mul(armOffset).min(24).negate().exp().toVar();

        // Fade along the arms, and cut everything ahead of the stem.
        const lengthFade = float(1).sub(along.div(armLength)).clamp(0, 1).toVar();
        const behind = fromStem.smoothstep(width.mul(-0.3), width.mul(0.5)).toVar();

        // --- dispersion ---------------------------------------------------------
        //
        // Both wave systems are stationary in the hull's frame, so both are
        // functions of `fromStem` and `lateral` alone — no time term anywhere.
        // That is the whole reason a stamped pattern works: what a wake *is*, is a
        // standing pattern that the ship drags along with it.
        //
        // Ahead of the foam, because the foam borrows the divergent phase: the two
        // have to agree about where a crest is or the foam sits in the troughs.
        const speed = slot.params.w;
        const speedSq = speed.mul(speed).toVar();

        // k0 = g/V². Wavelength grows as the square of speed, which is the single
        // most recognisable thing about a wake.
        const k0 = float(GRAVITY).div(speedSq.max(MIN_SPEED_SQUARED)).toVar();

        // Transverse system: crests across the track, bowing aft toward the arms.
        // The parabolic term is the leading-order curvature of the real crest
        // curves near the centreline — they are straight only in the limit.
        const stemSafe = fromStem.max(width.mul(0.5)).toVar();
        const transversePhase = k0
          .mul(fromStem.sub(lateral.mul(lateral).mul(1.35).div(stemSafe)))
          .toVar();

        // Divergent system: a plane wave at DIVERGENT_ANGLE to the track with
        // wavenumber k0/cos²(psi), mirrored across the centreline by using
        // |lateral|. Its crests are the feathered arms.
        const divergentPhase = k0
          .mul(DIVERGENT_K)
          .mul(fromStem.mul(DIVERGENT_COS).add(lateral.mul(DIVERGENT_SIN)))
          .toVar();

        // Amplitude: quadratic in speed, capped, and faded in from rest so a
        // drifting hull is not surrounded by half a metre of standing wave.
        const amplitude = speedSq
          .mul(ELEVATION_PER_SPEED_SQUARED)
          .min(MAX_ELEVATION)
          .mul(speed.smoothstep(0.4, 2.2))
          .toVar();

        // --- foam ---------------------------------------------------------------

        // Foam collects on the divergent crests, not evenly along the caustic.
        // This is what makes an arm read as a row of short feathers stepped back
        // from one another instead of a painted V, and taking the pitch from the
        // wave system that causes it means it lengthens with speed for free.
        //
        // Faded out below about 5 m/s: `k0/cos²(psi)` is pinned by
        // MIN_SPEED_SQUARED down there, so the divergent wavelength bottoms out
        // near a metre — two texels of this buffer — and modulating at that pitch
        // produces aliasing, not feathering.
        const feather = mix(
          float(1),
          divergentPhase.cos().mul(0.5).add(0.5),
          speed.smoothstep(3, 5.5).mul(0.6),
        ).toVar();

        // The bow break: white water thrown aside where the stem splits the
        // surface. Not a separate feature from the arms — the break *is* the head
        // of the divergent system, which is why it rides the same line — so it is
        // the same Gaussian weighted heavily over the first hull length and then
        // left to decay into the arms behind it. Without it the wedge began as two
        // faint threads out of nothing, which is the one thing a hull at speed
        // never looks like.
        const bowBreak = armFoam
          .mul(smoothstepDown(fromStem, width.mul(0.5), width.mul(3.8)))
          .mul(behind)
          .mul(BOW_FOAM)
          .toVar();

        // The turbulent band astern.
        //
        // This was one Gaussian a beam across, centred three-quarters of a beam
        // abaft the emitter — which is amidships — so the single most recognisable
        // part of a ship's wake was a blob under the ship. It is now a band that
        // starts at the transom and is dragged out astern by the accumulation.
        //
        // Note what actually draws the trail: the deposit falls away within about
        // three beams of the transom, and everything beyond that is the *history*
        // of texels the transom has already passed over, thinning at
        // `WAKE_FOAM_TAU`. Depositing along the whole visible length instead would
        // integrate to white over a hundred metres of water, which is the
        // saturation this file has twice been burnt by.
        const bandHalfWidth = width.mul(0.5).add(fromTransom.max(0).mul(0.055)).toVar();
        const bandOffset = lateral.div(bandHalfWidth).toVar();
        const bandSq = bandOffset.mul(bandOffset).min(8).toVar();
        // exp(-x⁴) rather than exp(-x²): the churn at a transom has a flat white
        // core with a soft edge to it, and a plain Gaussian is a ridge with a
        // bright line down the middle, which reads as a rope under tow.
        const bandProfile = bandSq.mul(bandSq).min(24).negate().exp().toVar();
        // Widening as it goes, but fading faster than it widens, so the bright
        // core narrows astern while the faint edges spread — which is how a real
        // band tapers rather than ending.
        const bandFade = fromTransom.max(0).div(width.mul(1.2)).min(12).negate().exp().toVar();
        const bandStart = fromTransom.smoothstep(width.mul(-0.75), width.mul(-0.05)).toVar();
        const sternFoam = bandProfile.mul(bandFade).mul(bandStart).mul(boil).toVar();

        wakeDeposit.addAssign(
          armFoam
            .mul(feather)
            .mul(lengthFade)
            .mul(behind)
            .mul(ARM_FOAM)
            .add(bowBreak)
            .add(sternFoam.mul(STERN_FOAM))
            .mul(amount),
        );

        // --- Kelvin elevation -------------------------------------------------
        //
        // The same wedge geometry, but carrying the actual wave systems rather
        // than a foam mask.
        const wedge = smoothstepDown(
          lateral,
          fromStem.mul(KELVIN_SLOPE).add(width.mul(0.5)),
          fromStem.mul(KELVIN_SLOPE).add(width.mul(2.1)).add(2),
        ).toVar();

        // The two systems do not decay at the same rate, and this is the reason
        // the arms are the part of a wake you can still see from a mile off.
        // Stationary phase gives the transverse waves the ordinary r^-1/2 of
        // energy spreading along a line, but the divergent waves pile up on the
        // cusp locus where the two families meet, and the cusp decays as r^-1/3.
        // Sharing one r^-1/2 between them threw away the difference and left the
        // arms fading at the same rate as the crests they are supposed to outlive.
        const radius = fromStem.max(width).div(width).toVar();
        const transverseSpread = float(1).div(radius.sqrt()).toVar();
        const divergentSpread = float(1).div(radius.pow(1 / 3)).toVar();

        const tail = float(1).sub(along.div(armLength.mul(1.35))).clamp(0, 1).toVar();
        const centreline = smoothstepDown(
          lateral,
          width.mul(0.8),
          // Held clear of the inner edge. Far enough ahead of the stem this outer
          // edge crosses below it, and a reversed pair is explicitly undefined in
          // both shading languages — see `smoothstepDown`. The region is
          // multiplied out by `behind` anyway, but "undefined" includes NaN, and
          // NaN times zero is still NaN.
          fromStem.mul(KELVIN_SLOPE).add(width.mul(2.2)).max(width.mul(1.2)),
        ).toVar();

        // Wider than the line the foam draws, and deliberately so: the divergent
        // crests fill the band just inside the caustic, they are not confined to
        // it. Reusing the foam's own Gaussian drew two wires where there should be
        // a feathered arm several crests deep.
        const armBand = armOffset.mul(0.5).toVar();
        const armEnvelope = armBand.mul(armBand).min(24).negate().exp().toVar();

        const transverse = transversePhase
          .cos()
          .mul(centreline)
          .mul(transverseSpread)
          .mul(TRANSVERSE_WEIGHT)
          .toVar();
        const divergent = divergentPhase
          .cos()
          .mul(armEnvelope.mul(0.85).add(0.15))
          .mul(divergentSpread)
          .mul(DIVERGENT_WEIGHT)
          .toVar();

        // Bow wave: the hull pushes a mound up ahead of itself and drags a trough
        // in behind the shoulder. Without it the pattern starts from nothing at
        // the hull, which reads as the ship floating over its own wake.
        //
        // The trough is not decoration. A displacement hull moving through water
        // has to put the water it displaces somewhere, and it comes back down
        // immediately abaft the bow — which is why a ship at speed sits visibly
        // *lower* amidships than the undisturbed surface around it. An earlier
        // version of this comment described the trough while the code added only
        // the mound, so the hull rode on a bulge with nothing under it.
        //
        // Both are now placed off the stem instead of off midships, which is where
        // a bow wave is: the mound was previously sitting a beam ahead of the
        // ship's centre, half a hull short of the stem that raises it, and the
        // trough was under the mainmast rather than at the shoulder.
        const bow = delta
          .sub(forward.mul(width.mul(HALF_LENGTH_BEAMS)))
          .length()
          .div(width)
          .toVar();
        const crest = bow.mul(bow).min(20).negate().exp().mul(1.25).toVar();

        // Centred about a beam abaft the mound, and wider — the shoulder trough is
        // a longer, shallower feature than the crest ahead of it, which is what
        // gives the two together the S-shape a bow wave has in profile.
        const shoulder = delta
          .sub(forward.mul(width.mul(0.7)))
          .length()
          .div(width.mul(1.5))
          .toVar();
        const bowWave = crest.sub(shoulder.mul(shoulder).min(20).negate().exp().mul(0.55)).toVar();

        const local = transverse
          .add(divergent)
          .mul(wedge)
          .mul(tail)
          .mul(behind)
          .add(bowWave)
          .mul(amplitude)
          .toVar();

        stamped.addAssign(local);
        // The stamp owns a texel wherever the pattern has structure there. `tail`
        // and `wedge` both go to zero at the pattern's edge, so the crossover to
        // decayed history happens exactly where the fresh stamp stops describing
        // anything.
        stampWeight.addAssign(
          wedge.mul(tail).max(bowWave.min(1)).mul(amplitude.mul(6).clamp(0, 1)),
        );
      }

      // Blend, do not sum: see the class comment.
      const elevation = mix(previousElevation, stamped, stampWeight.clamp(0, 1)).toVar();

      return vec4(
        previous.add(deposit).clamp(0, 1),
        elevation,
        previousWake.add(wakeDeposit).clamp(0, 1),
        1,
      );
    })();

    return material;
  }
}

/**
 * Cheap world-space stipple for the rain agitation term.
 *
 * A plain hash rather than interpolated noise: rain-aerated water is a spray of
 * discrete bright specks, and the hard edges a hash produces are closer to that
 * than a smooth field would be. It is also evaluated once per texel of the foam
 * buffer rather than per screen pixel, so it can afford to be blunt.
 */
const rainStipple = /*@__PURE__*/ Fn(([p]: [any]) => {
  const h = vec2(p.dot(vec2(127.1, 311.7)), p.dot(vec2(269.5, 183.3))).toVar();
  return h.sin().mul(43758.5453).fract().x;
});

/**
 * Slow world-space structure for the turbulent band astern, 0..1.
 *
 * The band cannot be a smooth mask. Water behind a transom is a field of boils
 * several metres across that surface, spread and collapse, and a flat white
 * ribbon is the tell that a wake was drawn rather than shed. The water shader
 * already breaks the foam's *edge* up at bubble scale, so what is missing at this
 * end is the metre scale between the two.
 *
 * Three sines rather than a hash, unlike `rainStipple` above, for two reasons.
 * Rain is a spray of discrete specks and wants hard edges; a boil is a smooth
 * mound and does not. And this buffer is resampled by a fractional texel offset
 * every frame as its centre follows the camera, which a hash does not survive —
 * it would scintillate along the whole length of the wake. The directions and
 * wavelengths are mutually incommensurate so the sum does not read as a grid.
 *
 * Anchored in the world and independent of time, so a texel is modulated the same
 * way on every frame the hull deposits on it and the structure accumulates
 * instead of averaging itself flat.
 */
const wakeBoil = /*@__PURE__*/ Fn(([p]: [any]) => {
  const a = p.dot(vec2(0.62, 0.27)).sin().toVar();
  const b = p.dot(vec2(-0.31, 0.74)).sin().toVar();
  const c = p.dot(vec2(1.13, -0.87)).sin().toVar();
  return a.add(b).add(c).mul(1 / 6).add(0.5);
});

/**
 * Resolves into the target whose texture reference never changes, and composites
 * the two foam channels on the way through.
 *
 * A screen, `1 - (1-a)(1-b)`, rather than a sum or a max. R and B are coverage
 * fractions of the same material laid down by two independent processes, and the
 * chance a texel is covered by either is exactly one minus the chance it is
 * covered by neither. Summing them puts wake over whitecap past white and clips
 * the difference away — the wake's own structure disappears wherever it crosses a
 * breaking crest — and `max` throws away the fact that two thin coverages
 * together are denser than one. The screen cannot exceed 1 by construction, so
 * the composite adds no saturation risk of its own.
 */
function createCopyMaterial(source: THREE.Texture): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial();
  material.depthTest = false;
  material.depthWrite = false;
  material.fragmentNode = Fn(() => {
    const s = texture(source, uv()).toVar();
    const sea = s.r.clamp(0, 1).toVar();
    const hull = s.b.clamp(0, 1).toVar();
    return vec4(sea.oneMinus().mul(hull.oneMinus()).oneMinus(), s.g, 0, 1);
  })();
  return material;
}

function normalizeResolution(size: number): number {
  return Number.isFinite(size) ? Math.max(1, Math.round(size)) : 1024;
}

function makeTarget(size: number): THREE.RenderTarget {
  const target = new THREE.RenderTarget(size, size, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    // Clamped, not repeated: foam must not wrap around to the far side of the
    // world when the ship leaves the footprint.
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  });
  return target;
}

function setVec2(target: THREE.Vector2, x: number, y: number): THREE.Vector2 {
  target.x = x;
  target.y = y;
  return target;
}
