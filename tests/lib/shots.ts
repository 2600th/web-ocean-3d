import type { Thresholds } from './compare';

/**
 * The canonical shot list.
 *
 * These are data, not code, on purpose: the visual harness, any future
 * before/after review loop, and the screenshots in the README should all be
 * framing the same seven views of this world, and the only way that stays true
 * is if there is exactly one place the framing lives.
 *
 * Every shot spells out **all** of the state it depends on, including values it
 * shares with the app defaults. `setState` mutates persistent app state, so a
 * shot that omitted `windSpeed` would inherit whatever the previous shot set and
 * would render differently depending on test ordering — which is the opposite of
 * what a baseline is for.
 */

export type QualityTier = 'low' | 'medium' | 'high' | 'ultra' | 'max';

export type PresetId =
  | 'skyPro'
  | 'arctic'
  | 'blackFlag'
  | 'dusk'
  | 'foggy'
  | 'moonlit'
  | 'seaOfThieves'
  | 'storm'
  | 'sunset';

export type CameraMode = 'orbit' | 'fly' | 'boat' | 'cinematic';

export type Vec3 = readonly [number, number, number];

export interface ShotState {
  quality: QualityTier;
  cameraMode: CameraMode;
  windSpeed: number;
  peakWavelength: number;
  cloudCoverage: number;
  preset: PresetId;
}

export interface Shot {
  id: string;
  title: string;
  /** What this shot exists to catch a regression in. */
  purpose: string;
  state: ShotState;
  /**
   * Exact camera pin, or `null` when the camera mode owns the pose (chase).
   *
   * Pinned before the settle run, not after: the underwater pass, the fog and
   * the particle system all derive their parameters from the camera during
   * `update()`, so a camera moved after the world settled would be photographed
   * with another camera's atmosphere.
   */
  camera: { position: Vec3; target: Vec3 } | null;
  /** Simulation time, in seconds, that the world is rewound to. */
  time: number;
  /** Steps of 1/60 s used to settle buoyancy, foam and the chase rig. */
  settleSteps: number;
  /**
   * Throttle and rudder held for the whole settle run, or omitted for a hull
   * that is only floating.
   *
   * Applied *inside* the reset, before the settle steps, because the ship has a
   * velocity time constant near 6.7 s: an input applied after the settle would
   * photograph a stationary hull with the throttle open. Only meaningful when
   * `state.cameraMode` is `boat` — the controller is disabled otherwise.
   */
  shipInput?: { throttle: number; rudder: number };
  /**
   * Position on the cinematic loop, seconds. Required when `state.cameraMode`
   * is `'cinematic'`, meaningless otherwise.
   *
   * This replaces the arrangement the two original tour shots used, where a
   * beat's key was copied into an *orbit* camera because this type did not
   * offer cinematic mode. That worked for those two only because the tour's
   * light there was close to the preset's — and it is useless for the beats
   * that matter most now that the flight has a day in it. A `cinematic-night`
   * shot captured as an orbit camera under `skyPro` is a **noon** frame at a
   * night camera position; a squall one is a clear-sky frame at a squall camera
   * position. Each would baseline the pose and nothing the beat exists to show.
   *
   * Naming a time instead of copying coordinates also retires the "if the beat
   * moves, this must move with it" hazard those shots' comments carried.
   */
  cinematicTime?: number;
}

/**
 * Capture resolution, and the viewport the visual project runs at.
 *
 * 1280x720 rather than the 1600x900 the performance budget is quoted at. The
 * comparison is per-pixel and full-resolution either way; the only thing the
 * larger frame would buy is 44% more bytes in every checked-in baseline, and
 * the defects these shots exist to catch are not one-pixel defects.
 */
export const SHOT_VIEWPORT = { width: 1280, height: 720 } as const;

export const SHOTS: readonly Shot[] = [
  {
    id: 'clear-day-wide',
    title: 'Clear day, wide',
    purpose:
      'The reference image. Horizon integration, sun glitter, cloud lighting, ' +
      'aerial perspective and the ship at a distance where its silhouette and ' +
      'its reflection both matter.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    // Looking roughly up-sun (skyPro's sun is toward +X/-Z) so the glitter track
    // runs through the frame instead of behind the camera.
    camera: { position: [-42, 21, 63], target: [0, 3, 0] },
    time: 12,
    settleSteps: 90,
  },
  {
    id: 'near-water-detail',
    title: 'Near-water detail',
    purpose:
      'Wave shape, normal detail, foam on breaking crests and the transition ' +
      'from resolved geometry to normal-mapped detail. This is where tiling, ' +
      'shimmer and over-bright specular show up first.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    // Away from the ship, cross-lit rather than up-sun, so the frame is water
    // and only water: this shot must fail for wave reasons, not hull reasons.
    camera: { position: [-46, 5, 44], target: [-64, 0, 26] },
    time: 23.5,
    settleSteps: 90,
  },
  {
    id: 'sunset',
    title: 'Sunset',
    purpose:
      'Low sun through a swell: the specular track, the warm-to-cool gradient ' +
      'across the water, and whether the tone mapping still holds highlight ' +
      'detail where the sun meets the horizon.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 6,
      peakWavelength: 24,
      cloudCoverage: 0.45,
      preset: 'sunset',
    },
    // The sunset sun sits almost due -X, so the camera looks straight down it.
    camera: { position: [58, 7, 9], target: [0, 4, 0] },
    time: 31.25,
    settleSteps: 90,
  },
  {
    id: 'storm',
    title: 'Storm',
    purpose:
      'Heavy sea state, rain, dense cloud and low contrast. Catches wave ' +
      'clipping at high amplitude, foam coverage running away, and rain that ' +
      'stops tracking the camera correctly.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 21,
      peakWavelength: 60,
      cloudCoverage: 0.95,
      preset: 'storm',
    },
    camera: { position: [34, 15, 54], target: [0, 4, 0] },
    time: 41,
    settleSteps: 90,
  },
  {
    id: 'boat-chase',
    title: 'Boat, chase camera',
    purpose:
      'The chase rig framing and the hull under way: buoyancy pose at speed, ' +
      'contact shadow, hull material response and the trailing wake.',
    state: {
      // `boat`, and this is load-bearing. `snapToTarget()` returns immediately
      // unless the director is actually in chase mode, so with `orbit` here the
      // shot never entered the chase rig at all: it photographed whatever camera
      // the previous shot happened to leave behind, and was stable only because
      // the suite runs in a fixed order. That is precisely the ordering
      // dependence the note at the top of this file exists to rule out.
      quality: 'high',
      cameraMode: 'boat',
      windSpeed: 13,
      peakWavelength: 40,
      cloudCoverage: 0.4,
      preset: 'seaOfThieves',
    },
    /**
     * `null` because the chase rig derives the pose from the hull; the harness
     * calls `director.snapToTarget()` after the settle so the pose is exact
     * rather than however far the damping happened to converge.
     *
     * The spec asks for this shot "while moving", and it now is. Full ahead is
     * held for the whole settle; 480 steps is 8 s, which against a ~6.7 s
     * velocity time constant and a 9.6 m/s terminal speed leaves the hull at
     * roughly 8 m/s having run about 38 m. That matters for what this shot can
     * catch: the wake buffer decays with a 1.7 s time constant, so a stationary
     * hull deposits a symmetric blob and nothing about the trailing wake, the
     * bow contact or the chase rig's lead is exercised at all.
     *
     * Rudder is zero deliberately. A turn would exercise the wake's curvature
     * too, but it also makes the pose depend on the yaw integral over 480 steps,
     * which is a far longer error lever than a straight run. The rudder's
     * behaviour — including its speed-dependent authority and its reversal when
     * making sternway — is covered by the four controller tests in
     * `tests/ocean.spec.ts`, where it can be asserted numerically instead of
     * photographed.
     */
    camera: null,
    time: 55.5,
    shipInput: { throttle: 1, rudder: 0 },
    settleSteps: 480,
  },
  {
    id: 'waterline',
    title: 'Waterline, split',
    purpose:
      'The eye *at* the surface: water below the line and sky above it in one ' +
      'frame, with the line following the crests rather than lying flat.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      // Calm on purpose, but no longer because the camera is fenced out of the
      // surface — that clamp is gone, and it was a wall that made diving
      // impossible. Calm because a shot with the eye a few centimetres under is
      // only reproducible if the crests near it are gentle; at 5 m/s they stay
      // under a metre.
      windSpeed: 5,
      peakWavelength: 20,
      cloudCoverage: 0.3,
      preset: 'skyPro',
    },
    /**
     * Eight centimetres under, looking very slightly up.
     *
     * This is the shot the spec always asked for and the renderer could not
     * produce. `submersion` used to cross-fade the *whole frame* on one scalar,
     * so a camera here rendered a uniformly half-tinted image rather than water
     * and air in the same picture. The underwater pass now integrates each eye
     * ray's own path below the surface: a ray angled down crosses metres of water
     * and comes back attenuated, a ray angled up leaves after eight centimetres
     * and does not, and the boundary between them is where the wave field crosses
     * the eye.
     */
    camera: { position: [-46, 0.02, 44], target: [-76, 1.0, 24] },
    time: 63.75,
    settleSteps: 90,
  },
  {
    id: 'underwater',
    title: 'Underwater',
    purpose:
      'Submerged look: extinction with depth, god rays, particulate, caustics ' +
      'on what is below, and the underside of the surface.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 10,
      peakWavelength: 36,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    // Well clear of the 0.35 m surface band, angled up so the surface underside
    // and the Snell window are both in frame.
    camera: { position: [-16, -7, 16], target: [-6, 2, 2] },
    time: 72.5,
    settleSteps: 90,
  },
  {
    id: 'island-approach',
    title: 'Island, from the water',
    purpose:
      'The island as a whole: shoreline rock, beach, canopy band and summit ' +
      'rock, the depth ramp from turquoise shallows to open blue, and the ' +
      'aerial perspective over about a kilometre. This is the frame the ' +
      'planting and the terrain biome are tuned against.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    /**
     * Sea level, roughly 400 m off the shore on the bearing the play area sees.
     *
     * `ISLAND` sits at (-1150, -780) with a mean shore radius of 500 m, so a
     * camera 900 m from its centre along the bearing toward the origin is about
     * that far off the beach — the distance at which the whole island fits the
     * frame and none of it is a silhouette. Six metres up rather than two: a
     * 1.5 m swell puts a wave crest through the lens at eye height, and the shot
     * is about the island rather than about the water in front of it.
     */
    /**
     * 780 m from the island centre, not the 900 the brief specifies.
     *
     * The brief gives both a camera — (-405, 6, -275) — and a framing target, the
     * island filling 85-90% of frame width, and on this terrain the two do not
     * agree: from 900 m out the island fills about two thirds. The reference
     * image is the ground truth for the look, so the framing wins and the camera
     * moves in along the same bearing until it matches. 660 m was tried first and
     * is too close: the island runs edge to edge and the foreground is all
     * lagoon, where the brief asks for substantial open ocean in the lower third.
     * 780 m keeps both headlands inside the frame and the deep water in front of
     * them.
     *
     * It also makes the colour comparison against the reference mean something.
     * Every "far sea" measurement taken from the old position was sampling water
     * a kilometre further away, and therefore at a far more grazing angle, than
     * the pixel it was being compared against — which is why three separate
     * attempts to explain a saturation gap of 0.20 against 0.66 each moved the
     * number by a rounding error.
     */
    camera: { position: [-505, 6, -342], target: [-1150, 58, -780] },
    time: 40,
    settleSteps: 90,
  },
  {
    id: 'shore-break',
    title: 'The shore break',
    purpose:
      'Where the sea meets the sand, from inside the surf zone. Depth-driven ' +
      'breaking foam, the swash band, wet sand, and rock crossing the ' +
      'waterline — none of which can be judged from the offshore shot, where ' +
      'the whole beach is forty pixels tall.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    /**
     * Standing in about a metre of water on the cove's bearing, looking up the
     * beach.
     *
     * `COVE_BEARING` in `Props` is 0.7 rad from the island centre and the shore
     * there is near the mean 500 m radius, which puts the waterline around
     * (-768, -458). This sits 35 m seaward of it at chest height on the swell,
     * angled along the beach rather than square at it so the surf line runs
     * across the frame instead of being a horizontal band.
     */
    camera: { position: [-726, 2.6, -424], target: [-830, 3, -516] },
    time: 52,
    settleSteps: 90,
  },
  {
    id: 'reef-dive',
    title: 'Over the reef',
    purpose:
      'The populated half of the underwater scene: coral, reef rock, kelp and ' +
      'the fish that live on it. The `underwater` shot deliberately looks *up* ' +
      'at the surface and the hull, so it can and does pass with an empty reef ' +
      'below it — which is exactly how a reef with no fish within visibility ' +
      'range went unnoticed.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 10,
      peakWavelength: 36,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    /**
     * Five metres off the bottom, fifteen out from the nearest reef patch, looking
     * back *at* it.
     *
     * The aim is the whole shot. `reefPatches` puts its closest patch at
     * (33, 29) and `findReefStation` gives reef school 0 that same patch, so the
     * first cut of this shot — standing at (34, 26) looking outward to (82, 54)
     * — had the school directly behind the lens and photographed empty sand
     * beyond it. It passed, twice, and was used as evidence that the reef had no
     * fish. Standing off and looking in puts the coral bed and its residents in
     * the same frame.
     *
     * Fifteen metres, not thirty. `schoolCentres` reports school 1 milling on an
     * 11 m circuit centred at (33, 31), so thirty metres did frame it — and a
     * 0.42 m fish at thirty metres, through water whose visibility is 45 m, is
     * four pixels of something the same colour as the water behind it. The
     * second cut of this shot proved the school was in frame and still showed
     * nothing, which is worth recording: "no fish visible" had two independent
     * causes and fixing the aim only removed the first.
     */
    /**
     * Aimed at where the school **is**, not at the middle of the circuit it
     * travels — and that distinction is the whole history of this shot.
     *
     * `schoolCentres` publishes an 11 m circuit centred on (33, 31), and both
     * previous cuts aimed at that centre. A resident school is a *body* that
     * moves round its circuit, not a cloud filling it, so aiming at the middle
     * photographs empty sand for most of the lap — which is indistinguishable
     * from a reef with no fish on it, and was twice concluded to be exactly
     * that. The gallery image and the `cinematic-reef` baseline both show it.
     *
     * `FishSchool` now publishes `schoolAnchors()`, the live position, which is
     * what a harness actually needs to frame one. At t = 68 this school sits at
     * (29, -13, 25). The camera stands eleven metres off it on the bearing
     * *away* from the coral bed, so the fish are between the lens and the reef
     * rather than beside it — coral behind, school in front, and both inside
     * the range where a 0.42 m body is thirty pixels of unmistakable fish
     * rather than four pixels of water-coloured speck.
     */
    camera: { position: [23, -12.4, 16], target: [29, -13.6, 25] },
    time: 68,
    settleSteps: 90,
  },
  {
    id: 'cinematic-reef',
    title: 'The tour, under the surface',
    purpose:
      "What the cinematic flight's `reef-run` beat actually sees. The tour is " +
      'the one camera a viewer does not steer, so a leg of it pointed at empty ' +
      'sand is a defect nobody can work around — and that is what it was, for ' +
      'the whole 26 s of it.',
    state: {
      quality: 'high',
      cameraMode: 'cinematic',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    /**
     * Captured by *running the tour*, not by copying its camera.
     *
     * Pinned as an orbit camera rather than by running the tour, because the
     * canonical shots cannot select cinematic mode — `ShotState.cameraMode` does
     * not offer it, since the flight owns the pose and a shot cannot pin one.
     * Copying the key gives the same frame from a camera the harness can hold
     * still, which is what a baseline needs. If the beat moves, this must move
     * with it; that is the cost of the arrangement and it is written down here
     * so the next person moving a key knows to.
     */
    camera: null,
    time: 44,
    // 53.5 s, and the precision is not fussiness — the window is genuinely that
    // narrow. The beat crosses the reef patch at (-97, -102) at about 20 m/s, so
    // the coral is at a useful size and inside the view cone for roughly two
    // seconds of the lap. 52.5 has it 34 m off and small; 55 has already passed
    // it and photographs open sand, which is the failure this shot exists to
    // catch and has twice been captured *as* the evidence for.
    //
    // It also has to be lit, and on a 60-second lap that is a real constraint:
    // the dive goes under at 03:57 and this is about the earliest the tour's
    // dawn has the sun high enough to reach 11 m of water.
    cinematicTime: 53.5,
    settleSteps: 90,
  },
  {
    id: 'cinematic-landfall',
    title: 'The tour, standing off the island',
    purpose:
      "The `landfall` beat's middle key: the whole landmass in one frame from " +
      '1.08 km, which is where the re-cut tour holds it. Headland, bay, canopy ' +
      'band and bare summit together, at the range the billboard canopy and the ' +
      'aerial perspective are both built to work at.',
    state: {
      quality: 'high',
      cameraMode: 'cinematic',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    camera: null,
    time: 30,
    cinematicTime: 26,
    settleSteps: 90,
  },
  {
    id: 'ship-and-island',
    title: 'The ship, with the island beyond',
    purpose:
      'The frame the README opens on: the hull and the landmass in one shot, ' +
      'which nothing in the gallery previously showed. Catches the two at their ' +
      'real relative scale — a 27 m ship a hundred metres off, a 1 km island a ' +
      'kilometre and a half beyond it — and the aerial perspective that has to ' +
      'separate them.',
    state: {
      quality: 'high',
      cameraMode: 'orbit',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    /**
     * Composed from the geometry rather than found by eye.
     *
     * The hull spawns at the origin and `ISLAND` sits at (-1150, -780), so the
     * two are on a bearing of (-0.828, -0.561) and any camera on the opposite
     * side of the ship has both in front of it. Standing 110 m off puts the 27 m
     * hull at 14 degrees — about a sixth of the frame width, big enough to read
     * as a ship rather than as a mark on the water — and the island at 1.5 km
     * subtends 37, a little under half the frame.
     *
     * The stand-off bearing is rotated 20 degrees off the ship-to-island line so
     * the hull does not eclipse the summit: it lands about 18 degrees off the
     * aim axis, which is the near third of the frame, with the island centred
     * behind it and open water in the foreground.
     *
     * The hull is floating rather than under way, and that is a real limitation
     * rather than a choice. `shipInput` only moves a ship whose controller is
     * enabled, which is Boat and Cinematic; and the chase rig re-derives its own
     * pose every frame, so a pinned camera cannot survive Boat mode. A wake here
     * would need the tour to happen to frame the island from astern, which it
     * does not.
     */
    /**
     * The aim point is 400 m out, **not** the island, and that is a constraint
     * rather than a preference: `OrbitControls.maxDistance` is 1200 m, and
     * `pin` drives the orbit rig, so aiming at the island 1493 m away silently
     * clamped the camera 240 m *past* the ship on the way to it. The first cut
     * of this shot photographed an empty sea with the island in it and no hull
     * anywhere — the pin had been honoured and then overruled.
     *
     * So the target sits on a bearing 45% of the way from the ship toward the
     * island, which puts the hull about 8 degrees left of the axis and the
     * island's centre about 10 to the right: the ship just overlapping the
     * island's near flank, with open water to its left.
     */
    camera: { position: [64, 15, 89], target: [-214, 23, -198] },
    time: 36,
    settleSteps: 90,
  },
  {
    id: 'cinematic-squall',
    title: 'The tour, in a squall',
    purpose:
      'The `squall` beat at its wettest. This is the only shot in the list ' +
      'that photographs the tour *changing the weather* — rain on the lens, ' +
      'cloud closed over, the key light killed and the hull wet — none of ' +
      'which the flight could do at all before, because it never touched ' +
      '`Weather`.',
    state: {
      quality: 'high',
      cameraMode: 'cinematic',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    camera: null,
    time: 76,
    // The rain envelope is centred on 40 s (`RAIN_CENTRE_SECONDS`) with a 9 s
    // half-width, so this sits exactly on the peak and the shot cannot be
    // quietly satisfied by a squall that arrives late.
    cinematicTime: 40,
    settleSteps: 90,
  },
  {
    id: 'cinematic-night',
    title: 'The tour, on the night watch',
    purpose:
      'The night half of the `squall` beat, which absorbed the night watch when ' +
      'the lap was re-cut to 60 s: moon glitter, the star field, and a hull lit ' +
      'by nothing else. The whole night half of the atmosphere, which the ' +
      "previous flight's 08:18-to-16:42 sun swing could not reach.",
    state: {
      quality: 'high',
      cameraMode: 'cinematic',
      windSpeed: 15,
      peakWavelength: 47,
      cloudCoverage: 0.32,
      preset: 'skyPro',
    },
    camera: null,
    time: 88,
    cinematicTime: 44,
    settleSteps: 90,
  },
] as const;

// --------------------------------------------------------------- noise floor

export interface NoiseFloor {
  meanDeltaE: number;
  p95DeltaE: number;
  fractionAbove: number;
}

/**
 * **Storm and boat-chase rose by more than an order of magnitude, and bloom is
 * why.** They went from 0.0387 and 0.0318 to 0.5691 and 0.3886. Those two shots
 * carry the heaviest foam in the list, and storm carries rain as well; a bloom
 * pyramid takes a single whitecap pixel that differed between two runs and
 * spreads it over a neighbourhood, so what used to be one isolated sparkle
 * difference is now a halo of them. That shows up exactly where it did: `p95`
 * and `fractionAbove` moved far more than the mean.
 *
 * The consequence is a genuine loss of sensitivity on those two shots — storm's
 * gate is now 1.24 mean ΔE where it was 0.18, so a subtle storm regression could
 * hide under it. It is recorded rather than tuned away, because the alternative
 * is choosing a bloom strength to suit a metric. If it needs recovering, the
 * lever is the harness (more warm-up captures, or a foam-masked comparison
 * region), not the renderer.
 *
 * The worst pairwise score over ten pairs, from five complete re-applications of
 * each shot, on this project's reference stack.
 *
 * This is not a fudge factor and it is not a guess. It is the cost of the fact
 * that a rendered frame is not perfectly reproducible: what moves between two
 * identical runs is specular sparkle on wave crests, where a difference far below
 * display precision in the wave field decides which facets catch the sun. A
 * regression smaller than these numbers cannot be detected, which is why they are
 * recorded rather than absorbed into a round tolerance.
 *
 * They are small — mean ΔE under 0.04 everywhere, with over 95% of pixels
 * bit-identical on every shot — but that is a property of the current renderer
 * and of the warm-up protocol in `applyShot`, not a guarantee. A future effect
 * with a genuinely stochastic or temporally-accumulated component will raise
 * them, and the correct response is to re-measure, not to widen the gate.
 *
 * Regenerate with `MEASURE_NOISE=1 npx playwright test --project=visual`; the
 * measurement writes `test-results/visual/noise-floor.json` and prints a block
 * ready to paste here. `docs/VERIFICATION.md` records the values, the stack they
 * came from, and how sensitive the resulting gate is to a real change.
 */
export const MEASURED_NOISE_FLOOR: Readonly<Record<string, NoiseFloor>> = {
  // Re-measured 2026-08-04 against the post chain, on the stack recorded in
  // docs/VERIFICATION.md. Two of these rose by more than an order of magnitude
  // and that is a real cost, not a rounding change -- see the note below.
  'clear-day-wide': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  'near-water-detail': { meanDeltaE: 0.0265, p95DeltaE: 0.0000, fractionAbove: 0.00229 },
  sunset: { meanDeltaE: 0.0337, p95DeltaE: 0.0000, fractionAbove: 0.00280 },
  storm: { meanDeltaE: 0.5691, p95DeltaE: 3.2400, fractionAbove: 0.06169 },
  'boat-chase': { meanDeltaE: 0.3886, p95DeltaE: 1.1000, fractionAbove: 0.01437 },
  waterline: { meanDeltaE: 0.0003, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  underwater: { meanDeltaE: 0.0210, p95DeltaE: 0.0000, fractionAbove: 0.00225 },
  'island-approach': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  'shore-break': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  'reef-dive': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  'cinematic-reef': { meanDeltaE: 0.0162, p95DeltaE: 0.0000, fractionAbove: 0.00213 },
  'cinematic-landfall': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
  'ship-and-island': { meanDeltaE: 0.0069, p95DeltaE: 0.0000, fractionAbove: 0.00062 },
  'cinematic-squall': { meanDeltaE: 0.0072, p95DeltaE: 0.0000, fractionAbove: 0.00051 },
  'cinematic-night': { meanDeltaE: 0.0000, p95DeltaE: 0.0000, fractionAbove: 0.00000 },
};

/**
 * How far above the measured noise a shot has to drift before it fails.
 *
 * 2x, plus a small absolute term. The multiplier covers the noise floor itself
 * being an estimate from ten sample pairs rather than a distribution; the
 * additive term is what keeps the gate meaningful on the shots that are almost
 * perfectly reproducible, where 2x of nearly nothing is still nearly nothing and
 * an honest rounding difference would fail the build.
 */
const NOISE_MULTIPLIER = 2;
const ABSOLUTE_FLOOR: NoiseFloor = {
  meanDeltaE: 0.1,
  p95DeltaE: 0.4,
  fractionAbove: 0.0015,
};

export function thresholdsFor(shotId: string): Thresholds {
  const noise = MEASURED_NOISE_FLOOR[shotId];
  if (!noise) throw new Error(`no measured noise floor for shot "${shotId}"`);
  return {
    meanDeltaE: noise.meanDeltaE * NOISE_MULTIPLIER + ABSOLUTE_FLOOR.meanDeltaE,
    p95DeltaE: noise.p95DeltaE * NOISE_MULTIPLIER + ABSOLUTE_FLOOR.p95DeltaE,
    fractionAbove: noise.fractionAbove * NOISE_MULTIPLIER + ABSOLUTE_FLOOR.fractionAbove,
  };
}
