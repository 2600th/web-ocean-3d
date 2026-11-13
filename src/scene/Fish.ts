import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  cos,
  faceDirection,
  float,
  mix,
  normalGeometry,
  positionGeometry,
  positionWorld,
  select,
  sin,
  texture,
  uniform,
  uv,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { dequantiseGeometry, type AssetLoader } from './AssetLoader';
import { ISLAND, reefPatches, seafloorHeight } from './Seafloor';

/**
 * Schools of fish, over the plateau, the reef and the island's shallows.
 *
 * **Why the geometry is procedural rather than a downloaded asset.** A rigged
 * glTF fish under a licence this project can verify was not obtainable, and for
 * a school at this scale it would have been the wrong technique anyway: a rig
 * means either one skinned draw call per fish or a bone-texture animation
 * system, to carry a mesh that is never seen closer than a few metres through
 * turbid water. Fish swim by passing a travelling wave down the body, which is
 * a closed-form displacement — so the whole animation is four lines of vertex
 * arithmetic on a hand-built silhouette, and the entire population is one
 * `InstancedMesh` and one draw call.
 *
 * Three ideas carry the module.
 *
 * **The body wave is the animation.** A carangiform swimmer's lateral
 * displacement is `a(s) sin(omega t - k s)` along the body axis `s`, with the
 * amplitude envelope `a` growing toward the tail — the nose barely moves and
 * the caudal fin sweeps about a tenth of a body length either side. That is
 * evaluated per vertex, and the normal is corrected by the analytic slope of
 * the same expression, so the body genuinely undulates instead of shearing
 * flat. Per-fish phase offsets desynchronise the school.
 *
 * **The school frame is CPU-side; everything inside it is GPU-side.** Each
 * school's circuit is a closed-form sum of sinusoids, so the CPU can evaluate
 * its centre, heading and bank for any `t` in a handful of trig calls — and,
 * crucially, can call `seafloorHeight` there. The floor clamp therefore uses
 * the *real* heightfield rather than a shader approximation of it, and only
 * three `vec4`s per school ever reach the GPU. Individual fish are placed as
 * offsets in that frame, weaving on their own phases, so the school has volume
 * and is not a rigid formation.
 *
 * **A school is not one behaviour.** The same closed-form machinery drives two
 * very different animals, because the difference between them is entirely in the
 * constants. An open-water shoal runs a two-hundred-metre circuit at cruise,
 * spread over thirty-four metres, holding a fixed depth below the surface. A
 * reef resident orbits a fixed piece of structure at a third of a metre a
 * second, packed into a third of the volume, and holds its depth above the
 * *floor* rather than below the surface — so it follows the bottom over an
 * outcrop instead of sliding across it. See `SCHOOL_LAYOUT`.
 *
 * **Nothing accumulates.** Every visible quantity is a pure function of the
 * clock: `resetClock(t)` and `update()`-ing to `t` produce a bit-identical
 * frame, which is what the visual regression harness requires. There is not a
 * single integrated velocity or remembered position anywhere in the file.
 *
 * That last point is also why the two GPU clocks are *phases* rather than
 * seconds. The other animated systems in this project push wrapped seconds into
 * a float32 uniform and take a discontinuity once an hour; here the CPU keeps
 * the clock in float64 and hands the shader `omega t mod 2pi`, with every
 * shader-side frequency an integer multiple of the fundamental. The wrap is
 * then exactly invisible — a sine does not care which turn it is on — and the
 * phase never grows large enough for `sin` to lose precision to range
 * reduction.
 *
 * Known limitation: the floor clamp is against the sand, not against the reef
 * outcrops `Props` scatters on it. A school can pass through a tall boulder.
 * Fixing that needs the prop transforms, which live in another module and would
 * couple the two for a collision the camera almost never sees side-on.
 */

/**
 * TSL node objects are structurally dynamic and the generated typings cannot
 * express a uniform whose component type is only known at construction, nor the
 * per-component expressions built out of `attribute()`. Node-typed values are
 * therefore `any` by design — the class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * `vec3` under a loose signature. Composing per-component expressions out of
 * `attribute()` values produces `any`, which the overloaded TSL typings then
 * resolve to the wrong constructor; the runtime behaviour is unaffected.
 */
const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;

/**
 * Local seed. `core/random` owns the shared `SEEDS` table and this module may
 * not edit it, so the constant lives here — a literal, for the same reason the
 * shared ones are: a baseline capture is only comparable to a later run if it
 * never drifts, and a drift has to be visible in a diff.
 */
const FISH_SEED = 0x5f15c4;

/**
 * Instance buffer capacity. `setCount` draws a prefix of it — see `setCount`.
 *
 * 600, raised from 480 when the tier budgets went up. It is a *silent* ceiling —
 * `setCount` clamps rather than warning — so a Max tier asking for 560 against a
 * 480 capacity would simply have drawn 480 and looked, from every angle except
 * this constant, like the tier table was being honoured.
 */
const MAX_FISH = 600;

/**
 * The schools, in the order the tier's budget populates them.
 *
 * Three ways of being a fish, not one:
 *
 *   `open`    a shoal crossing open water on a wide circuit round the origin.
 *             The original behaviour, unchanged in character.
 *   `reef`    residents holding station over one patch of the reef — milling
 *             about a fixed piece of structure rather than travelling anywhere,
 *             and holding their depth above the floor rather than below the
 *             surface.
 *   `island`  the same behaviour, in the island's shallows.
 *   `patrol`  one wide, slow lap round the island, so its shelf has something
 *             crossing it as well as something sitting on it.
 *
 * Every fish in this project used to be in the first kind, on a circuit centred
 * on the world origin, which meant that the two places a viewer will
 * deliberately go and dive — the reef and the island — had nothing living over
 * them. This is a redistribution and not an increase: the tier counts are
 * untouched and the same fish are spread over more places worth finding.
 *
 * **The order is the priority**, and the interleaving is not decorative — see
 * `SCHOOL_ACTIVATION`. At the lowest populated tier only the first four schools
 * exist, so an order that put the four open-water shoals first would leave
 * Medium with exactly the empty reef this change is about.
 *
 * Ten is still a compile-time constant, so the per-school uniforms are chosen
 * with a `select` chain rather than a dynamically indexed array — legal on both
 * backends, but the kind of thing WebGL2 drivers have historically been bad at,
 * and the chain costs a few ternaries on a population whose entire vertex count
 * is under eleven thousand.
 */
type SchoolKind = 'open' | 'reef' | 'island' | 'patrol';

const SCHOOL_LAYOUT: readonly SchoolKind[] = [
  'open',
  'reef',
  'reef',
  'island',
  'reef',
  'open',
  'island',
  'reef',
  'reef',
  'patrol',
];

const SCHOOLS = SCHOOL_LAYOUT.length;

/** How many of each kind there are, for the stratified layouts below. */
const OPEN_SCHOOLS = SCHOOL_LAYOUT.filter((kind) => kind === 'open').length;
const REEF_SCHOOLS = SCHOOL_LAYOUT.filter((kind) => kind === 'reef').length;
const ISLAND_SCHOOLS = SCHOOL_LAYOUT.filter((kind) => kind === 'island').length;

/**
 * Fish index at which each school starts to be populated.
 *
 * The problem: ten schools sharing Medium's forty fish is four fish each, and
 * four fish is not a school — it is four fish. Splitting a budget evenly is only
 * right when the budget is large, and this one runs from 40 to 220.
 *
 * So a school does not exist until the population can support it. `i % SCHOOLS`
 * becomes "give this fish to the emptiest school that has activated by now",
 * which still depends only on `i` — fish `i` therefore keeps its school at every
 * tier, and a tier change re-scales the population instead of reshuffling it,
 * exactly as before. The thresholds land each tier near twenty fish per active
 * school: 40 fills four schools, 90 fills six, 140 fills eight, 220 fills ten.
 */
const SCHOOL_ACTIVATION: readonly number[] = [0, 0, 14, 30, 48, 70, 96, 126, 160, 196];

// ------------------------------------------------------------------ the circuit

/**
 * Radial band the school centres wander in, metres from the origin.
 *
 * `Props` scatters the reef between 26 m and 260 m, and the base radii and
 * wander amplitudes below are chosen so `r(t)` stays inside 29..232 m without
 * ever needing a clamp. That matters more than it sounds: a clamp on `r` would
 * put a corner in the path, and the heading and bank are read from finite
 * differences of it, so a corner would be a visible flick of the whole school.
 *
 * The open-water base radii are *stratified* over the band — one per quarter,
 * jittered inside it — rather than drawn independently. Four independent draws
 * from this range land within 40 m of each other about a fifth of the time, and
 * the seed this module shipped with was one of those: every school ended up
 * beyond 165 m, where a 0.45 m fish through this water is a smudge. Stratifying
 * makes the coverage a property of the layout instead of a property of the seed.
 */
const RADIUS_MIN = 55;
const RADIUS_MAX = 200;

/** Cruise speed range, m/s. The angular rate is derived from it, not chosen. */
const CRUISE_MIN = 0.9;
const CRUISE_MAX = 1.4;

/** Radial wander: two out-of-phase terms, so the circuit is not a circle. */
const WANDER_R1 = 22;
const WANDER_W1 = 0.011;
const WANDER_R2 = 9;
const WANDER_W2 = 0.028;

/** Preferred depth below mean sea level, metres, before the floor clamp. */
const DEPTH_MIN = 6.5;
const DEPTH_MAX = 10.5;
const DEPTH_SWING = 2.8;

/**
 * Open-water school envelope, metres. Length runs along the heading, width
 * across it. Also the reference the weave amplitudes are scaled against, so a
 * tighter school automatically weaves less — see `WEAVE_ALONG`.
 *
 * A tension, resolved toward the tighter shoal: a wide school is in frame more
 * often, a tight one actually looks like a school. At 34 x 18 m a full-tier
 * shoal of 120 sits about three metres apart — loose for a reef fish, but this
 * is a school crossing open water between outcrops rather than one balled up
 * against a predator, and the separate circuits already buy back the coverage a
 * wider envelope would have.
 */
const SCHOOL_LENGTH = 34;
const SCHOOL_WIDTH = 18;
const SCHOOL_HALF_HEIGHT = 3.0;

/**
 * Metres of water kept between the lowest open-water fish and the sand.
 *
 * The floor under the reef band tops out at -14.7 m, so this leaves the school
 * at least 2 m of clearance in the worst place on the circuit and several times
 * that in the typical one. It is not a safety epsilon — it is the height a
 * school of fish visibly holds off the bottom. Residents get their own, much
 * smaller: see `RESIDENT_CLEARANCE`.
 */
const FLOOR_CLEARANCE = 2.5;

/**
 * Metres kept between the highest fish and mean sea level.
 *
 * Against the *displaced* surface, not the mean plane: a trough of a metre or
 * two passes overhead in any real sea state, and a fish that breaches reads as
 * a bug instantly. Three metres survives the swell this project generates.
 */
const SURFACE_CLEARANCE = 3.0;

// ---------------------------------------------------------------- the residents

/**
 * The resident circuit: a loop small enough to read as station-keeping.
 *
 * A reef fish's home range is metres, not hundreds of metres, and the thing that
 * makes a resident school look resident is that it never *goes* anywhere — it
 * turns over the same rock for as long as you watch it. A ten-metre orbit at a
 * third of a metre a second takes about three minutes to close, and the radial
 * wander below drifts the whole loop over a neighbourhood roughly twice that
 * wide, which together read as milling rather than as a fish on a rail.
 *
 * The wander stays well under the base radius on purpose. `r` is a polar radius
 * and a negative one turns the circuit inside out, taking the heading with it —
 * and the heading is read from finite differences, so it would arrive as the
 * whole school flipping end for end in a single frame.
 */
const RESIDENT_RADIUS_MIN = 9;
const RESIDENT_RADIUS_MAX = 15;
const RESIDENT_CRUISE_MIN = 0.3;
const RESIDENT_CRUISE_MAX = 0.45;
const RESIDENT_WANDER_R1 = 5;
const RESIDENT_WANDER_R2 = 2.5;

/**
 * Metres a resident school holds *above the floor beneath it*.
 *
 * The single most important difference between the two kinds, and the reason the
 * vertical reference is a per-school choice rather than a constant. A school at
 * a fixed depth below the surface passes over an outcrop without noticing it; a
 * school at a fixed height above the floor rises over the outcrop and settles
 * again behind it, which is what reef fish do and what makes the structure they
 * are holding to legible from a distance.
 */
const RESIDENT_HOVER_MIN = 2.2;
const RESIDENT_HOVER_MAX = 4.0;
const RESIDENT_HOVER_SWING = 1.0;

/**
 * Resident envelope, metres — a third of the open-water shoal's volume.
 *
 * Reef fish school tighter than pelagic ones because the structure is what they
 * are spacing themselves against, not each other. At 13 x 10 m a full school of
 * twenty-two sits a little over a metre and a half apart, which is dense enough
 * to read as one animal moving rather than as twenty-two.
 */
const RESIDENT_LENGTH = 13;
const RESIDENT_WIDTH = 10;
const RESIDENT_HALF_HEIGHT = 2.0;

/**
 * Water a resident school keeps between its lowest fish and the sand, metres.
 *
 * Far less than the open-water shoals' 2.5, and it has to be, or the clamp would
 * undo the hover: a school asking to sit 2.2 m off the bottom with a 2 m half
 * envelope and a 2.5 m clearance gets pushed to 4.5 m and is no longer near the
 * structure it is supposed to be holding to. A metre is what a reef fish
 * actually leaves under itself.
 */
const RESIDENT_CLEARANCE = 1.0;

/**
 * Resident tail beat and weave, as multiples of the open-water rates.
 *
 * The beat drops far less than the speed does, and deliberately. Scaled
 * linearly with ground speed, a school cruising at 0.35 m/s instead of 1.2 would
 * beat at 1 Hz, and a fish beating at 1 Hz does not look like it is holding
 * station — it looks dead and drifting. A fish holding position is not gliding:
 * it is working against the surge in short strokes and using its pectorals, so
 * the tail keeps moving even where the fish does not.
 *
 * The weave goes the other way. A resident mills, so its individuals cycle
 * through their own offsets faster than a cruising shoal's do, even though the
 * offsets themselves are smaller.
 */
const RESIDENT_BEAT_SCALE = 0.7;
const RESIDENT_WEAVE_SCALE = 1.55;

/**
 * The island patrol: a long shelf circuit rather than a lap of the island.
 *
 * The first version of this *was* a circumnavigation, at a fixed fraction of
 * `ISLAND.radius`, and it was wrong the moment the island stopped being a cone.
 * The coast now runs from 0.70 radii at the head of the bay to 1.25 at the tip
 * of the headland, so no circle centred on the island is entirely in water: at
 * 1.15 radii a fifth of the lap is over dry ground, and the floor clamp's answer
 * to a school over dry ground is to collapse its envelope and park it three
 * metres under mean sea level — which is to say, inside the hill.
 *
 * So the patrol is anchored to a station like the residents are, and simply
 * given a much larger circuit, a faster cruise and a surface-referenced depth.
 * It reads as a school working a stretch of shelf, which is the thing that was
 * wanted, and it cannot leave the water because `fitCircuit` shrinks it until it
 * does not.
 */
const PATROL_RADIUS = 90;
const PATROL_WANDER_R1 = 22;
const PATROL_WANDER_R2 = 9;
const PATROL_CRUISE = 0.75;
const PATROL_LENGTH = 26;
const PATROL_WIDTH = 14;
const PATROL_HALF_HEIGHT = 2.6;

/** Where a resident school may hold station: an annulus and a depth band. */
interface StationBand {
  x: number;
  z: number;
  minRadius: number;
  maxRadius: number;
  minDepth: number;
  maxDepth: number;
}

/**
 * The reef's stations.
 *
 * `Props` owns the rock scatter and does not export its extent, so the annulus
 * is stated rather than imported — but it only decides where to *look*, and the
 * heightfield decides what is accepted. The depth floor is what matters, and it
 * is derived rather than chosen: see `depthForSchool`.
 */
const REEF_STATION_BAND: StationBand = {
  x: 0,
  z: 0,
  // 22 m, not the 45 it was, and this is the difference between a reef with fish
  // on it and a reef that is provably populated somewhere the diver never gets
  // to. `Props` starts the coral at 26 m from the origin and the dive the scene
  // invites — the `underwater` shot's own camera — sits at a radius of 22. The
  // nearest school was therefore 45 m away through water whose visibility is 45
  // m, which is to say it was one extinction length out and had been attenuated
  // into the haze. Every fish in the scene was real, animated, and behind a wall
  // of blue.
  minRadius: 22,
  maxRadius: 190,
  minDepth: depthForSchool(RESIDENT_CLEARANCE, RESIDENT_HALF_HEIGHT) + 1,
  maxDepth: 22,
};

/**
 * The island's stations — its *shallows*, which is a narrower thing.
 *
 * In radii, because the island's size is not this module's to know, and because
 * it demonstrably changes. On the shelf as it stands the mean waterline is at
 * about one radius, so 0.85 to 1.35 brackets the shelf on both sides while the
 * depth band picks the part of it a school can actually occupy; about one sample
 * in ten is accepted, which forty-eight tries covers comfortably.
 *
 * The ceiling is what keeps the word "shallows" honest. Without it every station
 * slid to the outer edge of the annulus, into the twenty-odd metres of water off
 * the drop, and the island schools quietly became open-water schools that
 * happened to be near an island.
 */
const ISLAND_STATION_BAND: StationBand = {
  x: ISLAND.x,
  z: ISLAND.z,
  minRadius: ISLAND.radius * 0.85,
  maxRadius: ISLAND.radius * 1.35,
  minDepth: depthForSchool(RESIDENT_CLEARANCE, RESIDENT_HALF_HEIGHT) + 3,
  maxDepth: 18,
};

/**
 * The patrol's anchor: the outer shelf, and deliberately deep.
 *
 * 22 m is twice what the envelope strictly needs, and the margin is the whole
 * point. What shrinks a long circuit is not the mean depth at its centre but the
 * *relief* within ninety metres of it, and the island's inner shelf has plenty:
 * anchored at the minimum depth the school fits in, `fitCircuit` found a bump
 * every time and collapsed a 90 m lap to fifteen — a school that had stopped
 * patrolling anything and was simply a wide resident. Out here the slope is
 * smooth and the same nominal circuit survives at 63 m, which is a school
 * working a 170 m stretch of shelf.
 */
const PATROL_STATION_BAND: StationBand = {
  x: ISLAND.x,
  z: ISLAND.z,
  minRadius: ISLAND.radius,
  maxRadius: ISLAND.radius * 1.7,
  minDepth: 22,
  maxDepth: 44,
};

/** Candidate stations tried before the closest near-miss is taken. */
const STATION_TRIES = 48;

/**
 * How far a reef school's station may sit from its patch centre, metres.
 *
 * Comfortably inside the patch, because the school's own circuit is another nine
 * to fifteen metres on top of this and the point is that the fish are *over* the
 * coral rather than beside it. See `findReefStation`.
 */
const REEF_STATION_OFFSET = 7;

/** Bearings and radii the circuit fit probes, and how far it will shrink. */
const CIRCUIT_PROBES = 12;
const CIRCUIT_FIT_STEPS = 6;
const CIRCUIT_FIT_RATIO = 0.7;

/**
 * The shallowest water a school of this shape fits in, metres.
 *
 * Straight out of `solveAnchor`'s own arithmetic rather than judged: the centre
 * has to sit at least `clearance + halfHeight` above the floor and at least
 * `SURFACE_CLEARANCE + halfHeight` below the surface, so anything shallower than
 * their sum collapses the envelope and stacks the whole school into a plane.
 * Written as a function so the station bands and the circuit fit cannot drift
 * apart from the clamp they are both feeding.
 */
function depthForSchool(clearance: number, halfHeight: number): number {
  return clearance + 2 * halfHeight + SURFACE_CLEARANCE;
}

/**
 * Central-difference half-step for heading and bank, seconds.
 *
 * A quarter second is about half a metre of travel at cruise, which is a
 * well-conditioned baseline for `atan2`. Much shorter and the floor clamp's own
 * steps start to dominate the vertical component of the heading; much longer
 * and the bank lags the turn it is supposed to be anticipating.
 */
const DERIV_H = 0.25;

/**
 * Turn rate to roll, seconds, and the cap on the result.
 *
 * Not the coordinated-turn relation. `tan(roll) = v * omega / g` is the right
 * answer for something held up by a wing, and for a school circling at a
 * hundredth of a radian per second it evaluates to a fifth of a degree —
 * invisible. A neutrally buoyant fish is not fighting gravity at all; it rolls
 * because rolling is how it points its thrust, and it rolls hard. Ten seconds
 * puts the widest turn on the circuit a little past ten degrees and the typical
 * one at three or four, which is the "slightly" this is meant to read as. The
 * cap is there for the degenerate case, not for the circuit.
 */
const BANK_SECONDS = 10.0;
const BANK_MAX = 0.32;

/**
 * Ceiling on the school's climb angle, as a slope.
 *
 * The depth wander and the floor clamp together never produce more than a few
 * degrees, so this only exists to keep the shader's horizontal frame —
 * `normalize(forward.xz)` — away from a degenerate zero-length vector if the
 * heightfield ever handed the clamp a step.
 */
const PITCH_MAX = 0.25;

// ------------------------------------------------------------------- the weave

/**
 * Fundamental of the individual weave, rad/s. About a minute per cycle.
 *
 * Every other weave frequency in the shader is an integer multiple of *the
 * school's own* phase, which is what lets each phase uniform wrap at 2pi with no
 * discontinuity — and, because the CPU reduces each school's phase separately in
 * float64, is also what lets the residents weave at a different rate for free.
 * Fish do not really share a fundamental; the seeded phase offsets hide it
 * completely, and the alternative — independent rates per *fish* — is what
 * forces the float32 clock this module exists to avoid.
 */
const WEAVE_OMEGA = 0.11;

/**
 * Weave amplitudes within the school frame, metres, at the open-water envelope.
 *
 * Scaled in the shader by the school's own envelope against `SCHOOL_LENGTH` and
 * `SCHOOL_WIDTH`, so a resident school weaving inside a 13 x 10 m box swings
 * about a metre rather than two and a half. Deriving it rather than giving each
 * kind its own constant is what keeps a fish from weaving out of the envelope
 * the CPU proved clear of the sand.
 */
const WEAVE_ALONG = 2.5;
const WEAVE_LATERAL = 2.2;

/**
 * Yaw the weave induces, radians.
 *
 * A fish points where it is going. The lateral weave peaks at about 0.24 m/s
 * against a ~1.2 m/s cruise, so the heading offset is around a tenth of a
 * radian, and it is in quadrature with the offset itself because it is the
 * offset's derivative — which is why it reads as swimming rather than as
 * sliding sideways.
 */
const WEAVE_YAW = 0.2;

/** Roll the weave induces, radians. In antiphase with the lateral offset. */
const WEAVE_ROLL = 0.16;

// -------------------------------------------------------------------- the body

/**
 * Tail beat, Hz.
 *
 * Not decorative. Fish cover roughly two thirds of a body length per beat, so a
 * 0.42 m fish cruising at 1.2 m/s — which is what the circuit above implies —
 * beats a little over four times a second. 3.4 Hz is a shade under that, chosen
 * because the honest figure flickers at distance where the tail is a couple of
 * pixels wide and reads as aliasing rather than as swimming.
 */
const BEAT_HZ = 3.4;

/**
 * Body wavelength as a fraction of body length.
 *
 * 1.75, and the number that was here before — 1.1 — is why every fish in the
 * scene swam like an eel. The distinguishing measurement between the two gaits
 * is not the amplitude envelope, which both share the general shape of; it is
 * how many waves the body carries at once. Anguilliform swimmers hold **more
 * than one** full wave along the body, which is what produces the continuous
 * S travelling nose to tail that reads unmistakably as an eel. Carangiform
 * swimmers — jacks, tuna, and every reef fish in this scene — hold about **half
 * a wave**: one shallow bend, in the back half, and a head that stays pointed
 * where it is going.
 *
 * At 1.1 the body carried 1/1.1 = 0.91 waves, which is squarely anguilliform.
 * At 1.75 it carries 0.57, which is the middle of the measured carangiform
 * range. The tail-beat frequency is untouched: the gait is set by the wave
 * *number*, not by how fast it runs.
 */
const WAVE_LENGTH = 1.75;
const WAVE_K = (Math.PI * 2) / WAVE_LENGTH;

/**
 * Lateral amplitude at the tail base, in body lengths.
 *
 * The envelope is `AMP * s^3` with `s` running 0 at the nose to 1 at the tail
 * base and ~1.19 at the caudal tips, so the tips sweep about 0.14 L either
 * side. Measured carangiform envelopes land near 0.1 L at the peduncle and
 * — the part that matters — are almost flat over the front half: a tuna's
 * head yaws by under 2% of its length.
 *
 * Cubic rather than the quadratic this was. Both are "close to zero at the
 * nose", but a quadratic still gives the mid-body a quarter of the tail's
 * sweep, and a quarter of the sweep at mid-body is a visible second bend —
 * which, with the wavelength that used to sit above, was the other half of the
 * eel. The cubic halves it to an eighth and leaves the peduncle amplitude
 * exactly where it was, so the tail beat is unchanged and only the front of the
 * fish stiffens.
 */
const WAVE_AMP = 0.085;

/** Nominal body length, metres, before per-fish and per-school variation. */
const FISH_LENGTH = 0.42;

/**
 * Body silhouette: half-height and half-width at stations along the body.
 *
 * `u` runs 0 at the nose to 1 at the tail, and the two apexes are implicit — the
 * rings below are lofted between a nose point and a peduncle point. Taller than
 * wide by roughly two to one, because a reef fish is laterally compressed and
 * that compression is most of what makes the silhouette legible.
 */
const BODY_STATIONS: ReadonlyArray<{ u: number; halfHeight: number; halfWidth: number }> = [
  { u: 0.1, halfHeight: 0.062, halfWidth: 0.034 },
  { u: 0.24, halfHeight: 0.108, halfWidth: 0.055 },
  { u: 0.4, halfHeight: 0.118, halfWidth: 0.058 },
  { u: 0.56, halfHeight: 0.1, halfWidth: 0.046 },
  { u: 0.74, halfHeight: 0.07, halfWidth: 0.03 },
  { u: 0.9, halfHeight: 0.042, halfWidth: 0.016 },
];

/** Vertices per body ring. Six is enough at the sizes this is ever seen at. */
const RING = 6;

// ------------------------------------------------------------------- the colour

/** Dorsal, mid-flank and ventral tones. Counter-shading — see `buildMaterial`. */
const BACK_COLOR = new THREE.Color(0.055, 0.085, 0.1);
const FLANK_COLOR = new THREE.Color(0.42, 0.5, 0.53);
const BELLY_COLOR = new THREE.Color(0.8, 0.83, 0.8);

/**
 * The procedural body as several species, one per school, cycled in
 * `SCHOOL_LAYOUT` order: every school used to be the same silver jack, so ten
 * schools read as one animal repeated ten times. The swim, the shape and the
 * counter-shading are shared — what changes is the pigment, which is what
 * identifies a fish at the distances these are seen at.
 *
 *   silver jack        the original: dark back, silver flank
 *   blue fusilier      cobalt back over a yellow lateral band, pink belly
 *   yellowtail snapper slate back, a bright yellow flank stripe
 *   bigeye / soldier   rose-red all over, the reef's night shift
 *   barracuda          steel grey back, pale flank
 */
const PELAGIC_PALETTES: readonly (readonly [THREE.Color, THREE.Color, THREE.Color])[] = [
  [BACK_COLOR, FLANK_COLOR, BELLY_COLOR],
  [new THREE.Color(0.03, 0.08, 0.32), new THREE.Color(0.72, 0.6, 0.12), new THREE.Color(0.85, 0.68, 0.72)],
  [new THREE.Color(0.14, 0.18, 0.28), new THREE.Color(0.85, 0.68, 0.14), new THREE.Color(0.86, 0.84, 0.78)],
  [new THREE.Color(0.35, 0.06, 0.06), new THREE.Color(0.75, 0.22, 0.2), new THREE.Color(0.85, 0.55, 0.5)],
  [new THREE.Color(0.1, 0.12, 0.14), new THREE.Color(0.55, 0.58, 0.6), new THREE.Color(0.86, 0.87, 0.86)],
];

/** Skylight the fish sit in, and the direct sun on top of it. */
const AMBIENT_COLOR = new THREE.Color(0.24, 0.34, 0.38);
const SUN_COLOR = new THREE.Color(0.9, 0.88, 0.78);
const SUN_STRENGTH = 0.85;

/** Strength of the flank sheen. Small: it is a hint of colour, not a coating. */
const SHEEN = 0.07;

// ----------------------------------------------------------------- path records

/** One school's circuit. Every field is a constant of a closed-form path. */
interface SchoolPath {
  /** Circuit centre, world XZ. The origin for the open-water shoals. */
  cx: number;
  cz: number;
  theta0: number;
  omega: number;
  radius: number;
  wanderR1: number;
  wanderR2: number;
  r1Phase: number;
  r2Phase: number;
  /**
   * Which surface `depth` is measured from.
   *
   * `false` — metres below mean sea level, for anything swimming in the water
   * column. `true` — metres above the highest floor under the school's
   * footprint, for anything holding to structure. See `RESIDENT_HOVER_MIN`.
   */
  fromFloor: boolean;
  depth: number;
  depthSwing: number;
  depthRate: number;
  depthPhase: number;
  /** Envelope, metres, and the square the floor is probed over. */
  length: number;
  width: number;
  halfHeight: number;
  footprint: number;
  clearance: number;
  /** Weave and tail-beat rates, rad/s. Reduced to a phase every frame. */
  weaveOmega: number;
  beatOmega: number;
  /** Body-length multiplier for this school's fish. */
  scale: number;
}

/**
 * Which of the two bodies a school's fish wear.
 *
 * The distinction is the one a diver actually makes. Anything crossing open
 * water is seen at thirty metres through turbid water as a moving glint, and the
 * procedural silhouette with its counter-shading is a better model of that than
 * a texture would be — see `buildFishGeometry`. Anything holding station over
 * structure is seen from three metres, and at three metres a grey sliver is the
 * single most obvious piece of missing detail in the scene.
 *
 * So the residents get a real reef fish: `emperor_angelfish`, textured, rigged
 * in the source and stripped of its rig by `scripts/optimize-assets.mjs`,
 * animated here by the same travelling-wave vertex arithmetic as everything
 * else. It stays one draw call for the whole species.
 */
type Species = 'pelagic' | 'reef';

const SPECIES_OF_KIND: Readonly<Record<SchoolKind, Species>> = {
  open: 'pelagic',
  patrol: 'pelagic',
  reef: 'reef',
  island: 'reef',
};

/**
 * A body imported from a glTF, already normalised into this module's frame.
 *
 * `geometry` must run nose at `+0.5x`, tail at `-0.5x`, up `+y`, lateral `+z`,
 * one unit long — the same convention `buildFishGeometry` is authored in, so the
 * wave and the school frame apply to it unchanged. `normaliseImportedBody` does
 * the conversion; nothing else should have to know the source's axes.
 */
export interface FishSpecies {
  geometry: THREE.BufferGeometry;
  map: THREE.Texture | null;
}

export interface FishSchoolOptions {
  /** Overrides the seeded circuits and scatter; useful for A/B-ing a layout. */
  seed?: number;
}

/**
 * A population of fish in `SCHOOLS` independent shoals — see `SCHOOL_LAYOUT`
 * for where they are and how the three kinds differ.
 *
 * Add `object` to the scene. It carries an identity transform and the vertex
 * stage emits world coordinates directly, so it must be parented to something
 * untransformed — the scene root — exactly as `UnderwaterParticles` is.
 */
export class FishSchool {
  readonly object: THREE.Object3D;

  /**
   * One draw per species, each drawing a prefix of its own instances.
   *
   * `indices` is the ascending list of global fish indices this draw owns, which
   * is what lets `setCount` stay a pure function of the population: the tier
   * asks for the first `n` fish of the whole school and each species draws
   * however many of those belong to it. Fish `i` therefore keeps its school,
   * its body, its size and its phase at every tier, exactly as before.
   */
  private readonly draws: {
    species: Species;
    geometry: THREE.BufferGeometry;
    material: THREE.MeshBasicNodeMaterial;
    mesh: THREE.InstancedMesh;
    indices: Int32Array;
  }[] = [];

  private readonly paths: SchoolPath[] = [];


  private count: number;
  private wantVisible = true;
  private disposed = false;

  /**
   * Seconds, in float64, never wrapped.
   *
   * The GPU only ever sees phases derived from this, so there is no float32
   * range to protect and no reason to introduce a wrap discontinuity. `update`
   * advances it and `resetClock` sets it; nothing else reads or writes it.
   */
  private time = 0;

  // --- uniforms -------------------------------------------------------------
  /** Per school: xyz = centre, w = vertical half-extent the clamp allows. */
  private readonly uAnchors: Node[] = [];
  /** Per school: xyz = unit heading, w = bank angle. */
  private readonly uHeadings: Node[] = [];
  /**
   * Per school: x = envelope length, y = width, z = weave phase, w = beat phase.
   *
   * The two phases ride here rather than in their own global uniforms because
   * they now differ per school — a resident beats slower and weaves faster than
   * a cruiser — and this vec4 had two free slots. It costs nothing: the shader
   * already selects one of these per instance, so a per-school rate is the same
   * number of instructions as a shared one.
   */
  private readonly uShapes: Node[] = [];
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(SUN_COLOR));
  private readonly uAmbient = uniform(new THREE.Color(AMBIENT_COLOR));
  /** Per school: dorsal, flank and ventral tone — see `PELAGIC_PALETTES`. */
  private readonly uBacks: Node[] = [];
  private readonly uFlanks: Node[] = [];
  private readonly uBellies: Node[] = [];
  private readonly uSheen = uniform(SHEEN);

  // Scratch for the per-frame anchor solve. Reused — never reallocated.
  private readonly here = new THREE.Vector3();
  private readonly before = new THREE.Vector3();
  private readonly after = new THREE.Vector3();
  private readonly forward = new THREE.Vector3();

  /** Kept so a later `setSpecies` rebuilds instance data identically. */
  private readonly seed: number;

  constructor(count: number, options: FishSchoolOptions = {}) {
    this.count = Math.max(0, Math.min(MAX_FISH, Math.floor(count)));
    this.seed = options.seed ?? FISH_SEED;

    const random = mulberry32(this.seed);
    for (const path of buildSchools(random)) this.paths.push(path);

    for (const path of this.paths) {
      this.uAnchors.push(uniform(new THREE.Vector4(0, -8, 0, path.halfHeight)));
      this.uHeadings.push(uniform(new THREE.Vector4(1, 0, 0, 0)));
      this.uShapes.push(uniform(new THREE.Vector4(path.length, path.width, 0, 0)));
      const palette = PELAGIC_PALETTES[this.uBacks.length % PELAGIC_PALETTES.length];
      this.uBacks.push(uniform(palette[0].clone()));
      this.uFlanks.push(uniform(palette[1].clone()));
      this.uBellies.push(uniform(palette[2].clone()));
    }

    this.object = new THREE.Object3D();
    this.object.name = 'fish-schools';
    // Published for the visual harness, which has to be able to *frame* a school
    // to photograph one. Without it the only way to aim a shot at fish is to
    // guess, and a shot aimed at empty water is indistinguishable from a reef
    // with no fish in it — which is exactly the wrong conclusion this once
    // supported. Read-only data, no behaviour.
    this.object.userData.schoolCentres = this.paths.map((path) => ({
      cx: Math.round(path.cx),
      cz: Math.round(path.cz),
      radius: Math.round(path.radius),
    }));

    // And the *live* position of each school, as a function rather than a
    // snapshot.
    //
    // `schoolCentres` publishes the circuit a school travels; it does not say
    // where on that circuit the school is now, and a resident school is a body
    // that moves round its circuit rather than a cloud filling it. Framing one
    // therefore needs the anchor, not the centre — aiming at the centre of an
    // 11 m circuit from nine metres away photographs empty sand whenever the
    // school happens to be on the far side, which is indistinguishable from a
    // reef with no fish on it. That mistake has now been made three times in
    // this repository, twice in the shot list and once in a gallery image.
    this.object.userData.schoolAnchors = () =>
      this.uAnchors.map((u) => {
        const v = u.value as THREE.Vector4;
        return { x: v.x, y: v.y, z: v.z };
      });
    this.object.frustumCulled = false;
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();

    this.buildDraw('pelagic', null);
    this.buildDraw('reef', null);

    this.object.visible = this.count > 0;
    this.refresh();
  }

  /**
   * Swaps a species' body for an imported one, once the asset has arrived.
   *
   * Called from the async content load rather than the constructor, because the
   * schools have to exist and be swimming from the first frame whether or not a
   * 400 KB fish ever downloads. A 404 therefore costs the reef its markings and
   * nothing else, which is the same failure mode every other piece of dressing
   * in this project has.
   */
  setSpecies(species: Species, body: FishSpecies | null): void {
    if (this.disposed) return;
    const existing = this.draws.findIndex((draw) => draw.species === species);
    if (existing < 0) return;

    const previous = this.draws[existing];
    this.object.remove(previous.mesh);
    previous.mesh.dispose();
    previous.geometry.dispose();
    previous.material.dispose();
    this.draws.splice(existing, 1);

    this.buildDraw(species, body);
    this.applyCounts();
  }

  /** Builds the instanced draw for one species. */
  private buildDraw(species: Species, body: FishSpecies | null): void {
    const indices = instancesForSpecies(species);
    const geometry = body ? body.geometry.clone() : buildFishGeometry();
    attachInstanceAttributes(geometry, this.paths, this.seed, indices);

    const material = this.buildMaterial(body?.map ?? null);

    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, indices.length));
    mesh.name = species === 'pelagic' ? 'fish' : 'fish-reef';
    // The vertex stage writes world positions, so the instance matrix is dead
    // code — but `InstancedMesh` allocates it zero-filled, and an identity fill
    // costs one loop at startup and removes a whole category of "why is
    // everything at the origin" from anyone who later reads the buffer.
    identityInstanceMatrix(mesh.instanceMatrix, indices.length);
    // Shadows off deliberately: a 0.4 m fish under ten metres of water casts
    // nothing the eye can find, and it would double the vertex cost of the
    // whole population to render the depth pass.
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    // The population spans a 500 m circle and moves; a bound derived from the
    // geometry would be meaningless, and there is only one draw call to save.
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();

    this.draws.push({ species, geometry, material, mesh, indices });
    this.object.add(mesh);
    this.applyCounts();
  }

  /**
   * Points each draw at its share of the first `count` fish.
   *
   * `indices` is ascending, so "how many of mine are below `count`" is a binary
   * search rather than a scan, and the answer is the instance count directly —
   * the buffers are already laid out in global-index order.
   */
  private applyCounts(): void {
    for (const draw of this.draws) {
      draw.mesh.count = countBelow(draw.indices, this.count);
    }
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Population for the current quality tier.
   *
   * Free, unlike the equivalent on the particle systems: the instance buffers
   * are built once at `MAX_FISH` and this only moves the draw's instance count.
   * That also makes a tier change *re-scale* the population rather than
   * reshuffle it — fish `i` keeps its school, size, offset and phase whatever
   * the count is, so dropping a tier thins the schools instead of replacing
   * them with different ones.
   */
  setCount(count: number): void {
    const next = Math.max(0, Math.min(MAX_FISH, Math.floor(count)));
    if (next === this.count) return;
    this.count = next;
    this.applyCounts();
    this.object.visible = this.wantVisible && next > 0;
  }

  setVisible(v: boolean): void {
    this.wantVisible = v;
    this.object.visible = v && this.count > 0;
  }

  /** `dir` points *toward* the sun and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(dir);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();
  }

  /**
   * Advances the clock and re-solves the four school frames.
   *
   * Runs whether or not the schools are visible, and that is deliberate: making
   * the clock depend on visibility would make the pose depend on the history of
   * `setVisible` calls, which is exactly the class of hidden state
   * `resetClock` exists to rule out. The cost is 270 heightfield evaluations —
   * ten schools, three time samples, nine probes — which is a few hundredths of
   * a millisecond, and is the price of the floor clamp being exact rather than a
   * shader's guess at the same field.
   */
  update(dt: number): void {
    if (this.disposed) return;
    this.time += dt;
    this.refresh();
  }

  /** Jumps the clock, for reproducible captures. Identical to updating to `time`. */
  resetClock(time = 0): void {
    if (this.disposed) return;
    this.time = time;
    this.refresh();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const draw of this.draws) {
      this.object.remove(draw.mesh);
      draw.mesh.dispose();
      draw.geometry.dispose();
      draw.material.dispose();
    }
    this.draws.length = 0;
    this.object.removeFromParent();
  }

  // ------------------------------------------------------------------ internals

  /**
   * Pushes the current pose into the uniforms.
   *
   * Heading and bank come from a central difference of the path rather than
   * from its analytic derivative, because the path the fish actually follow is
   * the *clamped* one — differencing the unclamped circuit would have the
   * school pointing along a heading it is not travelling wherever the floor
   * pushed it up.
   */
  private refresh(): void {
    const t = this.time;

    for (let k = 0; k < SCHOOLS; k++) {
      const path = this.paths[k];
      const half = solveAnchor(path, t, this.here);
      solveAnchor(path, t - DERIV_H, this.before);
      solveAnchor(path, t + DERIV_H, this.after);

      const f = this.forward.copy(this.after).sub(this.before);
      const run = Math.hypot(f.x, f.z);
      if (run < 1e-6) {
        f.set(1, 0, 0);
      } else {
        f.y = clampNumber(f.y, -PITCH_MAX * run, PITCH_MAX * run);
        f.normalize();
      }

      // Two headings a half-step apart give the turn rate the bank follows.
      const h0 = Math.atan2(this.here.z - this.before.z, this.here.x - this.before.x);
      const h1 = Math.atan2(this.after.z - this.here.z, this.after.x - this.here.x);
      const turn = wrapPi(h1 - h0) / DERIV_H;
      const bank = clampNumber(turn * BANK_SECONDS, -BANK_MAX, BANK_MAX);

      (this.uAnchors[k].value as THREE.Vector4).set(this.here.x, this.here.y, this.here.z, half);
      (this.uHeadings[k].value as THREE.Vector4).set(f.x, f.y, f.z, bank);
      // Phases, reduced in float64 so the shader never sees a large argument.
      (this.uShapes[k].value as THREE.Vector4).set(
        path.length,
        path.width,
        wrapTau(t * path.weaveOmega),
        wrapTau(t * path.beatOmega),
      );
    }
  }

  /**
   * The body shader, for either species.
   *
   * Everything above the colour — the school frame, the weave, the travelling
   * body wave and the normal correction — is identical and deliberately shared:
   * the two species differ in what they are made of, not in how they swim. Pass
   * a `map` and the flank is textured; pass null and it is counter-shaded from
   * the baked `fishFlank` coordinate.
   */
  private buildMaterial(map: THREE.Texture | null): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    material.name = map ? 'fish-body-textured' : 'fish-body';
    // The fins are single-sided sheets in the body's midplane. Without
    // `DoubleSide` a fish's tail blinks out every time the beat carries it past
    // edge-on, which at 3.4 Hz is a strobe rather than a glitch.
    material.side = THREE.DoubleSide;
    material.fog = false;

    const seed: Node = attribute('fishSeed', 'vec4');
    const trait: Node = attribute('fishTrait', 'vec4');
    // The imported body has no counter-shading coordinate — it does not need
    // one, its pigment is in the texture — so the attribute is only read on the
    // procedural path. Reading a missing attribute is a shader compile error on
    // WebGPU, not a silent zero.
    const flank: Node = map ? float(0) : attribute('fishFlank', 'float');
    const p: Node = positionGeometry;
    const n: Node = normalGeometry;

    const anchor = pickPerSchool(this.uAnchors, trait.x);
    const heading = pickPerSchool(this.uHeadings, trait.x);
    const shape = pickPerSchool(this.uShapes, trait.x);

    // --- the school's horizontal frame -------------------------------------
    //
    // Offsets are laid out in the horizontal plane and along world up, not in
    // the school's tilted frame. That is what makes the vertical envelope the
    // CPU solved for *exact*: a pitched frame would convert some of the 17 m
    // along-axis offset into height, and the floor clearance would quietly
    // become a function of the climb angle.
    const f: Node = heading.xyz;
    const fh = vec3n(f.x, 0, f.z).normalize();
    const rh = vec3n(fh.z.negate(), 0, fh.x);

    // --- the individual's weave --------------------------------------------
    const phase = trait.w.mul(Math.PI * 2);
    const w: Node = shape.z;
    const weave = w.add(phase);
    const sLat = sin(weave);
    const cLat = cos(weave);
    const sVert = sin(w.mul(2).add(phase.mul(1.7)));
    const sAlong = sin(w.mul(3).add(phase.mul(2.3)));

    // The envelope and the weave inside it both come from the school's own
    // uniform, so a resident is genuinely a tighter school and not a cruiser
    // drawn smaller: its fish are packed closer *and* swing less far.
    const along = seed.x
      .sub(0.5)
      .mul(shape.x)
      .add(sAlong.mul(shape.x).mul(WEAVE_ALONG / SCHOOL_LENGTH));
    const lateral = seed.y
      .sub(0.5)
      .mul(shape.y)
      .add(sLat.mul(shape.y).mul(WEAVE_LATERAL / SCHOOL_WIDTH));
    // Scaled by the clamp's own half-extent and bounded by construction: the
    // fixed and weaving parts sum to at most one, so no fish can leave the band
    // the CPU proved clear of the sand and the surface.
    const vertical = anchor.w.mul(seed.z.sub(0.5).mul(1.52).add(sVert.mul(0.24)));

    const centre = anchor.xyz
      .add(fh.mul(along))
      .add(rh.mul(lateral))
      .add(vec3n(0, vertical, 0));

    // --- the individual's frame --------------------------------------------
    // Yaw leads the weave by a quarter cycle because it is the weave's
    // derivative; roll trails it by half, because a fish rolls into the turn
    // and the turn is sharpest where the lateral offset is at its extreme.
    const yaw = cLat.mul(WEAVE_YAW);
    const bank = heading.w.sub(sLat.mul(WEAVE_ROLL));

    const axisF = f.add(rh.mul(yaw)).normalize();
    const axisR0 = axisF.cross(vec3(0, 1, 0)).normalize();
    const axisU0 = axisR0.cross(axisF);
    const cb = cos(bank);
    const sb = sin(bank);
    const axisR = axisR0.mul(cb).add(axisU0.mul(sb));
    const axisU = axisU0.mul(cb).sub(axisR0.mul(sb));

    // --- the travelling body wave ------------------------------------------
    // `s` is the body axis: 0 at the nose, 1 at the peduncle, ~1.19 at the
    // caudal tips, which is why the tail sweeps hardest without needing a
    // separate term for the fin.
    const s = float(0.5).sub(p.x);
    const beat = shape.w.add(seed.w.mul(Math.PI * 2)).sub(s.mul(WAVE_K));
    const amp = s.mul(s).mul(s).mul(WAVE_AMP);
    const swing = sin(beat).mul(amp);

    // d(swing)/dx of `amp(s) sin(phi - k s)` with s = 0.5 - x. Both terms kept:
    // dropping the envelope's contribution leaves the tail's normal lagging its
    // own silhouette by a noticeable amount at the extremes of the beat.
    //
    // The envelope term follows the envelope: for `AMP s^3` its derivative is
    // `3 AMP s^2`, where the quadratic envelope's was `2 AMP s`. Getting this
    // wrong does not move the silhouette at all — it only mis-lights it, which
    // is exactly the kind of error that survives review.
    const slope = amp
      .mul(WAVE_K)
      .mul(cos(beat))
      .sub(s.mul(s).mul(3 * WAVE_AMP).mul(sin(beat)));

    // Inverse transpose of the shear this displacement is: for `z += g(x)` the
    // normal maps as `(nx - g' nz, ny, nz)`. Two operations, and without it a
    // swimming fish is lit as though it were rigid.
    const localN = vec3n(n.x.sub(slope.mul(n.z)), n.y, n.z).normalize();

    const size = trait.y;
    const world = centre
      .add(axisF.mul(p.x.mul(size)))
      .add(axisU.mul(p.y.mul(size)))
      .add(axisR.mul(p.z.add(swing).mul(size)));

    const worldNormal = axisF
      .mul(localN.x)
      .add(axisU.mul(localN.y))
      .add(axisR.mul(localN.z));

    material.positionNode = world;

    const vNormal: Node = varying(worldNormal, 'fishNormal');

    material.colorNode = Fn(() => {
      // `faceDirection` rather than a two-sided lighting hack: the fins are
      // genuinely seen from both faces and their normal is genuinely flipped.
      const normal = vNormal.normalize().mul(faceDirection).toVar();
      const view = cameraPosition.sub(positionWorld).normalize().toVar();

      const key = normal.dot(this.uSunDir).clamp(0, 1).toVar();
      // Skylight arrives from above and is what fills the shaded side; a
      // constant ambient would flatten the body into a cut-out.
      const sky = normal.y.mul(0.5).add(0.5).toVar();

      // --- counter-shading ---------------------------------------------------
      //
      // Dark back, silver flank, pale belly. This is not stylisation: pelagic
      // fish are pigmented exactly this way so that the shadow cast by their own
      // body is cancelled by the gradient, and it is the single strongest cue
      // that a small dark shape in blue water is a fish. It also happens to be
      // the thing that still reads at fifty metres through this project's water,
      // where a PBR treatment would resolve to one flat tone.
      //
      // Driven by a baked body coordinate, not by the normal: the pigment
      // gradient runs over the flank, which on a laterally compressed fish is
      // almost entirely surface whose normal points sideways.
      const base = (
        map
          ? // The imported body carries its own pigment, and it is the point of
            // importing it. `texture()` in the colour node rather than
            // `material.map`, because this is a `MeshBasicNodeMaterial` whose
            // colour is entirely replaced below and would otherwise ignore it.
            texture(map, uv()).rgb.toVar()
          : (() => {
              const t = flank.mul(0.5).add(0.5).toVar();
              const back = pickPerSchool(this.uBacks, trait.x);
              const flankTone = pickPerSchool(this.uFlanks, trait.x);
              const belly = pickPerSchool(this.uBellies, trait.x);
              const shaded = mix(belly, flankTone, t.smoothstep(0.0, 0.45)).toVar();
              shaded.assign(mix(shaded, back, t.smoothstep(0.52, 0.95)));
              return shaded;
            })()
      ) as Node;
      // A degree of per-fish tint, so a school is not one repeated animal. Held
      // much tighter on the textured species: an emperor angelfish is a
      // recognisable animal with recognisable colours, and tinting it by ±7%
      // reads as a rendering fault rather than as variation.
      const tintLow = map ? vec3(0.97, 0.99, 1.01) : vec3(0.93, 0.99, 1.04);
      const tintHigh = map ? vec3(1.03, 1.0, 0.98) : vec3(1.07, 1.0, 0.93);
      base.mulAssign(mix(tintLow, tintHigh, trait.w));

      const light = this.uAmbient
        .mul(sky.mul(0.65).add(0.35))
        .add(this.uSunColor.mul(key).mul(SUN_STRENGTH));

      // --- flank sheen -------------------------------------------------------
      //
      // A cosine palette over a view-dependent term: three cosines, no texture,
      // no thin-film integral. It is not a physical iridescence model and does
      // not claim to be — what it reproduces is the one behaviour that matters
      // at this distance, which is that the colour of the band slides as the
      // fish turns rather than staying painted on.
      //
      // Masked to the lateral line and multiplied by the key light, so it can
      // only ever brighten a lit flank and never haloes the silhouette.
      const fresnel = float(1).sub(normal.dot(view).abs()).toVar();
      const band = fresnel.mul(2.4).add(float(0.5).sub(p.x).mul(1.7)).add(trait.w);
      const iri = cos(vec3(band).add(vec3(0, 0.31, 0.62)).mul(Math.PI * 2)).mul(0.5).add(0.5);
      const sheen = iri
        .mul(this.uSheen)
        .mul(float(1).sub(flank.mul(flank)))
        .mul(fresnel)
        .mul(key.mul(0.7).add(0.3));

      return vec4(base.mul(light).add(sheen), 1);
    })();

    return material;
  }
}

// --------------------------------------------------------------------- the path

/**
 * Every school, in `SCHOOL_LAYOUT` order.
 *
 * Drawn from one generator in a fixed order, so the layout is a pure function of
 * the seed — the same property the instance attributes rely on, and for the same
 * reason: a baseline capture that cannot survive a reload is not a baseline.
 */
function buildSchools(random: () => number): SchoolPath[] {
  const paths: SchoolPath[] = [];
  let open = 0;
  let reef = 0;
  let island = 0;

  for (const kind of SCHOOL_LAYOUT) {
    switch (kind) {
      case 'open':
        paths.push(buildOpenPath(open++, random));
        break;
      case 'reef':
        paths.push(buildResidentPath(random, findReefStation(random, reef++)));
        break;
      case 'island':
        paths.push(
          buildResidentPath(
            random,
            findStation(random, ISLAND_STATION_BAND, island++, ISLAND_SCHOOLS),
          ),
        );
        break;
      case 'patrol':
        paths.push(buildPatrolPath(random, findStation(random, PATROL_STATION_BAND, 0, 1)));
        break;
    }
  }

  return paths;
}

function buildOpenPath(index: number, random: () => number): SchoolPath {
  // One school per quarter of the radial band, jittered within it. See RADIUS_MIN.
  const slice = (RADIUS_MAX - RADIUS_MIN) / OPEN_SCHOOLS;
  const radius = RADIUS_MIN + (index + 0.15 + random() * 0.7) * slice;
  const cruise = CRUISE_MIN + random() * (CRUISE_MAX - CRUISE_MIN);
  // Angular rate derived from a cruise speed rather than chosen directly, so a
  // school on the outer edge of the band does not lap one on the inner edge at
  // four times the tail-beat-implied speed.
  const direction = random() < 0.5 ? -1 : 1;

  return {
    cx: 0,
    cz: 0,
    theta0: random() * Math.PI * 2,
    omega: (direction * cruise) / radius,
    radius,
    wanderR1: WANDER_R1,
    wanderR2: WANDER_R2,
    r1Phase: random() * Math.PI * 2,
    r2Phase: random() * Math.PI * 2,
    fromFloor: false,
    depth: DEPTH_MIN + random() * (DEPTH_MAX - DEPTH_MIN),
    depthSwing: DEPTH_SWING,
    depthRate: 0.012 + random() * 0.018,
    depthPhase: random() * Math.PI * 2,
    length: SCHOOL_LENGTH,
    width: SCHOOL_WIDTH,
    halfHeight: SCHOOL_HALF_HEIGHT,
    footprint: footprintFor(SCHOOL_LENGTH, SCHOOL_WIDTH),
    clearance: FLOOR_CLEARANCE,
    weaveOmega: WEAVE_OMEGA,
    beatOmega: BEAT_HZ * Math.PI * 2,
    scale: 0.8 + random() * 0.55,
  };
}

/**
 * A school that lives somewhere rather than travelling through it.
 *
 * Same closed form, different constants — and one structural difference:
 * `fromFloor`, which measures the school's depth up from the bottom instead of
 * down from the surface. See `RESIDENT_HOVER_MIN`.
 */
function buildResidentPath(random: () => number, station: Station): SchoolPath {
  const nominal = RESIDENT_RADIUS_MIN + random() * (RESIDENT_RADIUS_MAX - RESIDENT_RADIUS_MIN);
  const cruise = RESIDENT_CRUISE_MIN + random() * (RESIDENT_CRUISE_MAX - RESIDENT_CRUISE_MIN);
  const direction = random() < 0.5 ? -1 : 1;

  const fit = fitCircuit(
    station,
    nominal,
    RESIDENT_WANDER_R1 + RESIDENT_WANDER_R2,
    footprintFor(RESIDENT_LENGTH, RESIDENT_WIDTH),
    depthForSchool(RESIDENT_CLEARANCE, RESIDENT_HALF_HEIGHT),
  );
  const radius = nominal * fit;

  return {
    cx: station.x,
    cz: station.z,
    theta0: random() * Math.PI * 2,
    omega: (direction * cruise) / radius,
    radius,
    wanderR1: RESIDENT_WANDER_R1 * fit,
    wanderR2: RESIDENT_WANDER_R2 * fit,
    r1Phase: random() * Math.PI * 2,
    r2Phase: random() * Math.PI * 2,
    fromFloor: true,
    depth: RESIDENT_HOVER_MIN + random() * (RESIDENT_HOVER_MAX - RESIDENT_HOVER_MIN),
    depthSwing: RESIDENT_HOVER_SWING,
    depthRate: 0.02 + random() * 0.03,
    depthPhase: random() * Math.PI * 2,
    length: RESIDENT_LENGTH,
    width: RESIDENT_WIDTH,
    halfHeight: RESIDENT_HALF_HEIGHT,
    footprint: footprintFor(RESIDENT_LENGTH, RESIDENT_WIDTH),
    clearance: RESIDENT_CLEARANCE,
    weaveOmega: WEAVE_OMEGA * RESIDENT_WEAVE_SCALE,
    beatOmega: BEAT_HZ * RESIDENT_BEAT_SCALE * Math.PI * 2,
    // Reef residents are smaller fish than the pelagic shoals; the size class is
    // most of what distinguishes them at the range they are seen from.
    scale: 0.6 + random() * 0.42,
  };
}

/** A long circuit of the island's shelf. See `PATROL_RADIUS`. */
function buildPatrolPath(random: () => number, station: Station): SchoolPath {
  const fit = fitCircuit(
    station,
    PATROL_RADIUS,
    PATROL_WANDER_R1 + PATROL_WANDER_R2,
    footprintFor(PATROL_LENGTH, PATROL_WIDTH),
    depthForSchool(FLOOR_CLEARANCE, PATROL_HALF_HEIGHT),
  );
  const radius = PATROL_RADIUS * fit;
  const direction = random() < 0.5 ? -1 : 1;

  return {
    cx: station.x,
    cz: station.z,
    theta0: random() * Math.PI * 2,
    omega: (direction * PATROL_CRUISE) / radius,
    radius,
    wanderR1: PATROL_WANDER_R1 * fit,
    wanderR2: PATROL_WANDER_R2 * fit,
    r1Phase: random() * Math.PI * 2,
    r2Phase: random() * Math.PI * 2,
    fromFloor: false,
    depth: DEPTH_MIN + random() * (DEPTH_MAX - DEPTH_MIN),
    depthSwing: DEPTH_SWING,
    depthRate: 0.012 + random() * 0.018,
    depthPhase: random() * Math.PI * 2,
    length: PATROL_LENGTH,
    width: PATROL_WIDTH,
    halfHeight: PATROL_HALF_HEIGHT,
    footprint: footprintFor(PATROL_LENGTH, PATROL_WIDTH),
    clearance: FLOOR_CLEARANCE,
    weaveOmega: WEAVE_OMEGA,
    beatOmega: BEAT_HZ * 0.85 * Math.PI * 2,
    scale: 0.85 + random() * 0.5,
  };
}

/** A patch of bottom worth holding station over. */
interface Station {
  x: number;
  z: number;
}

/**
 * Puts a reef school over the reef.
 *
 * The old version of this sampled the plateau's annulus for anything of the
 * right depth, which is how you place a school when there is nothing down there
 * to place it *on*. There is now: `Seafloor.reefPatches` is where `Props` grows
 * the coral and the rock, and a resident school's entire behaviour — holding
 * station, following the bottom over an outcrop, never travelling anywhere — is
 * about structure it can hold to. Stationing it anywhere else meant the two
 * things a diver goes looking for were in different places, and finding either
 * one told you nothing about where the other was.
 *
 * Patches are taken in order and wrapped, so with three reef schools over seven
 * patches the schools land on three different ones rather than stacking. The
 * offset is a short hop off the patch centre — enough that the school is not
 * bolted to a coordinate, small against the 15 m patch — and the depth is still
 * checked, because a patch is a place on the plateau and not a promise about it.
 */
function findReefStation(random: () => number, index: number): Station {
  const patches = reefPatches();
  if (patches.length === 0) return findStation(random, REEF_STATION_BAND, index, REEF_SCHOOLS);

  const patch = patches[index % patches.length];
  for (let attempt = 0; attempt < STATION_TRIES; attempt++) {
    const bearing = random() * Math.PI * 2;
    const offset = Math.sqrt(random()) * REEF_STATION_OFFSET;
    const x = patch.x + Math.cos(bearing) * offset;
    const z = patch.z + Math.sin(bearing) * offset;
    const depth = -seafloorHeight(x, z);
    if (depth >= REEF_STATION_BAND.minDepth && depth <= REEF_STATION_BAND.maxDepth) {
      return { x, z };
    }
  }
  return { x: patch.x, z: patch.z };
}

/**
 * Finds water a resident school can actually live in.
 *
 * Sampled and rejected against `seafloorHeight` rather than written down,
 * because the island's shape belongs to another module and is being changed —
 * a station in metres would be a station on dry land the moment it grew. The
 * band it samples is only a hint about where to look; the heightfield decides.
 *
 * Sectors rather than free angles, for the same reason the open-water radii are
 * stratified: three independent draws from a full circle put two schools on the
 * same side of the reef about half the time, and a viewer who dives on the other
 * side finds nothing.
 *
 * Always returns. If nothing in the band is the right depth, the closest
 * near-miss is taken and the floor clamp does the rest.
 */
function findStation(
  random: () => number,
  band: StationBand,
  sector: number,
  sectors: number,
): Station {
  let best: Station = { x: band.x + band.minRadius, z: band.z };
  let bestMiss = Number.POSITIVE_INFINITY;

  for (let attempt = 0; attempt < STATION_TRIES; attempt++) {
    const angle = ((sector + random()) / sectors) * Math.PI * 2;
    // Uniform in radius, not in area. An area-even draw is right for scattering
    // ninety rocks and wrong for choosing three places: it puts two thirds of
    // the draw in the outer third of the annulus, and all three reef stations
    // came out within fifteen metres of the same radius. Three points want
    // spread, not an even density.
    const r = band.minRadius + random() * (band.maxRadius - band.minRadius);
    const x = band.x + Math.cos(angle) * r;
    const z = band.z + Math.sin(angle) * r;
    const depth = -seafloorHeight(x, z);

    if (depth >= band.minDepth && depth <= band.maxDepth) return { x, z };

    const miss = Math.max(band.minDepth - depth, depth - band.maxDepth);
    if (miss < bestMiss) {
      bestMiss = miss;
      best = { x, z };
    }
  }

  return best;
}

/**
 * Shrinks a circuit until everywhere it can reach is deep enough, returning the
 * factor to scale its radius and both wander amplitudes by.
 *
 * `findStation` guarantees the *centre* is in usable water. It says nothing
 * about the circuit drawn around it, and on a coast like this one that gap is
 * not academic: the island's shelf runs at a gradient of a fifth, so a school
 * orbiting twenty metres out from a station in nine metres of water can find
 * itself over the beach. The floor clamp's response to that is to collapse the
 * school's envelope and hold it three metres under mean sea level — inside the
 * hill, where nothing can see it and nothing reports it.
 *
 * What is probed is the whole annulus the school's *volume* can occupy, which is
 * the circuit plus its radial wander plus the school's own footprint. All three
 * matter, and the third is the one that was originally missed: the patrol's
 * centre stayed in eight metres of water for its whole lap while the corner of
 * its 26 x 14 m box reached ground the clamp measured at five, so the band
 * between floor and surface closed and the school flattened into a single
 * horizontal plane. That does not look like a bug, it looks like a decal, and it
 * survived a thirty-minute sweep of the circuit unnoticed until the half-extent
 * itself was asserted on.
 *
 * **The footprint is not scaled and the circuit is.** Shrinking a school's
 * circuit does not shrink the school; scaling all three together was the second
 * version of this function and it made the fit *stop working as it converged* —
 * by the time the scale reached a quarter, the probe was covering a fifth of the
 * ground the clamp would actually sample, so it reported clear water and left a
 * fourteen-percent squeeze behind it. Everything else about the search stayed
 * the same and the numbers did not move at all, which is what gave it away.
 *
 * Scaling both the radius and the wander is what makes this converge — shrinking
 * the radius alone leaves a wander that can be larger than the circuit it is
 * perturbing. The limit is the station itself, which `findStation` already
 * checked, so six steps of 0.7 always terminate somewhere acceptable.
 */
function fitCircuit(
  station: Station,
  radius: number,
  wander: number,
  footprint: number,
  minDepth: number,
): number {
  let scale = 1;

  for (let step = 0; step < CIRCUIT_FIT_STEPS; step++) {
    let shallowest = Number.POSITIVE_INFINITY;

    for (let b = 0; b < CIRCUIT_PROBES; b++) {
      const angle = (b / CIRCUIT_PROBES) * Math.PI * 2;
      const ca = Math.cos(angle);
      const sa = Math.sin(angle);
      const reach = wander * scale + footprint;
      for (let edge = -1; edge <= 1; edge++) {
        const r = Math.max(0, radius * scale + edge * reach);
        const depth = -seafloorHeight(station.x + ca * r, station.z + sa * r);
        if (depth < shallowest) shallowest = depth;
      }
    }

    if (shallowest >= minDepth) return scale;
    scale *= CIRCUIT_FIT_RATIO;
  }

  return scale;
}

/**
 * Half-size of the square the floor is probed over, metres.
 *
 * Covers the school envelope at any heading — the corner of the box plus the
 * weave amplitude, whichever way it is rotated, plus a metre. For the
 * open-water envelope that comes to 22.7 m, which is the constant this replaced.
 * Probed on a 3x3 grid rather than densely: measured against a 3 m scan of the
 * same square over the whole radial band, the grid underestimates the true
 * maximum by at most 0.33 m, which the clearance absorbs several times over.
 */
function footprintFor(length: number, width: number): number {
  return Math.hypot(length, width) * 0.5 + WEAVE_ALONG + 1;
}

/**
 * The school centre at time `t`, and the vertical half-extent its members may
 * occupy around it.
 *
 * The band is solved rather than assumed. `bottom` is the highest floor found
 * under the school's footprint plus a clearance, `top` is the surface less its
 * own; the half-extent is whatever fits inside, and the centre is then placed
 * so that the whole envelope does. That ordering matters in the degenerate case
 * where the two limits cross — the band collapses to a single depth *under the
 * surface* rather than resolving in favour of the floor, because a fish in the
 * sand is a curiosity and a fish in the air is a bug report.
 *
 * `fromFloor` chooses what the school is asking for before any of that: metres
 * below the surface, or metres above the same `floorMax` the clamp is built
 * from. Reusing that value is what makes a resident follow the bottom exactly
 * rather than approximately — it is holding station against the surface the
 * clamp is measuring, not against a separate probe of it.
 */
function solveAnchor(path: SchoolPath, t: number, out: THREE.Vector3): number {
  const theta = path.theta0 + path.omega * t;
  const r =
    path.radius +
    path.wanderR1 * Math.sin(WANDER_W1 * t + path.r1Phase) +
    path.wanderR2 * Math.sin(WANDER_W2 * t + path.r2Phase);

  const x = path.cx + Math.cos(theta) * r;
  const z = path.cz + Math.sin(theta) * r;

  let floorMax = Number.NEGATIVE_INFINITY;
  for (let ix = -1; ix <= 1; ix++) {
    for (let iz = -1; iz <= 1; iz++) {
      const h = seafloorHeight(x + ix * path.footprint, z + iz * path.footprint);
      if (h > floorMax) floorMax = h;
    }
  }

  const swing = path.depthSwing * Math.sin(path.depthRate * t + path.depthPhase);
  const wanted = path.fromFloor ? floorMax + path.depth + swing : -(path.depth + swing);

  const top = -SURFACE_CLEARANCE;
  let bottom = floorMax + path.clearance;
  if (bottom > top) bottom = top;

  const half = Math.min(path.halfHeight, (top - bottom) * 0.5);
  out.set(x, clampNumber(wanted, bottom + half, top - half), z);
  return half;
}

// ----------------------------------------------------------------- the geometry

/**
 * One fish: 49 vertices, 77 triangles.
 *
 * A tube lofted between a nose point and a peduncle point through six elliptical
 * rings, plus a forked caudal fin and a dorsal and anal fin as flat triangles in
 * the midplane. The fins are what make the silhouette read; the rings are what
 * let the travelling wave bend something rather than shear a card.
 *
 * Authored with +x forward, +y up and the body one unit long, so the body axis
 * coordinate the wave needs is `0.5 - x` and needs no attribute of its own.
 *
 * The `fishFlank` attribute is the counter-shading coordinate: -1 on the ventral
 * line, +1 on the dorsal. It is baked rather than derived in the shader because
 * on a laterally compressed body it is a property of the *pigment*, which
 * follows the anatomy, not of the surface orientation, which does not.
 */
function buildFishGeometry(): THREE.BufferGeometry {
  const positions: number[] = [];
  const flanks: number[] = [];
  const indices: number[] = [];

  const push = (x: number, y: number, z: number, flank: number): number => {
    positions.push(x, y, z);
    flanks.push(flank);
    return flanks.length - 1;
  };

  const nose = push(0.5, 0, 0, 0);

  const rings: number[][] = [];
  for (const station of BODY_STATIONS) {
    const ring: number[] = [];
    const x = 0.5 - station.u;
    for (let j = 0; j < RING; j++) {
      // Offset by half a step so the ring has vertices exactly on the dorsal
      // and ventral lines, which is where the counter-shading gradient turns.
      const a = ((j + 0.5) / RING) * Math.PI * 2;
      const sy = Math.sin(a);
      ring.push(push(x, sy * station.halfHeight, Math.cos(a) * station.halfWidth, sy));
    }
    rings.push(ring);
  }

  const peduncle = push(-0.5, 0, 0, 0);

  const first = rings[0];
  for (let j = 0; j < RING; j++) indices.push(nose, first[(j + 1) % RING], first[j]);

  for (let k = 0; k + 1 < rings.length; k++) {
    const a = rings[k];
    const b = rings[k + 1];
    for (let j = 0; j < RING; j++) {
      const j2 = (j + 1) % RING;
      indices.push(a[j], a[j2], b[j2], a[j], b[j2], b[j]);
    }
  }

  const last = rings[rings.length - 1];
  for (let j = 0; j < RING; j++) indices.push(last[j], last[(j + 1) % RING], peduncle);

  // Fin vertices take their counter-shading from their height, so the tail
  // inherits the body's gradient across the fork instead of ending in a band.
  const finFlank = (y: number): number => clampNumber(y / 0.12, -1, 1);

  const tailUp = push(-0.46, 0.03, 0, finFlank(0.03));
  const tailDown = push(-0.46, -0.03, 0, finFlank(-0.03));
  const tailNotch = push(-0.6, 0, 0, 0);
  const tailTipUp = push(-0.69, 0.185, 0, finFlank(0.185));
  const tailTipDown = push(-0.69, -0.185, 0, finFlank(-0.185));
  indices.push(
    tailUp, tailTipUp, tailNotch,
    tailUp, tailNotch, tailDown,
    tailDown, tailNotch, tailTipDown,
  );

  indices.push(
    push(0.2, 0.112, 0, 1),
    push(0.1, 0.203, 0, 1),
    push(-0.12, 0.09, 0, 1),
  );

  indices.push(
    push(-0.12, -0.09, 0, -1),
    push(-0.32, -0.055, 0, -1),
    push(-0.2, -0.155, 0, -1),
  );

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('fishFlank', new THREE.Float32BufferAttribute(flanks, 1));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  // Every vertex is relocated by the vertex stage, so a bound derived from these
  // positions describes nothing. Culling is off; this exists so anything that
  // reads the sphere gets a defensible answer rather than a null.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

  return geometry;
}

/**
 * Per-instance seeds and traits.
 *
 * Drawn from one generator in index order, so instance `i` gets the same values
 * whatever `MAX_FISH` is and — because the buffer is never rebuilt — whatever
 * the tier is. Same reasoning as `UnderwaterParticles`: a baseline capture that
 * cannot survive a reload is not a baseline.
 */
function attachInstanceAttributes(
  geometry: THREE.BufferGeometry,
  paths: readonly SchoolPath[],
  seed: number,
  indices: Int32Array,
): void {
  const random = mulberry32(seed ^ 0x9e3779b9);
  const assignment = buildSchoolAssignment();

  // Every fish's values are drawn in *global* index order whether or not this
  // species keeps them, so splitting the population across two draws cannot
  // change what any individual fish looks like. Skipping a draw would shift the
  // whole stream and reshuffle both species.
  const seeds = new Float32Array(indices.length * 4);
  const traits = new Float32Array(indices.length * 4);
  let slot = 0;

  for (let i = 0; i < MAX_FISH; i++) {
    const school = assignment[i];

    const s0 = random();
    // Lateral and vertical are pulled toward the middle of the envelope. A
    // uniform draw gives a school with a hard rectangular edge and a hollow
    // core; real shoals are densest in the centre and ragged at the margin.
    const s1 = centreWeighted(random());
    const s2 = centreWeighted(random());
    const s3 = random();
    const size = FISH_LENGTH * (0.78 + random() * 0.5) * paths[school].scale;
    const t2 = random();
    const t3 = random();

    if (slot >= indices.length || indices[slot] !== i) continue;

    seeds[slot * 4 + 0] = s0;
    seeds[slot * 4 + 1] = s1;
    seeds[slot * 4 + 2] = s2;
    seeds[slot * 4 + 3] = s3;

    traits[slot * 4 + 0] = school;
    traits[slot * 4 + 1] = size;
    traits[slot * 4 + 2] = t2;
    traits[slot * 4 + 3] = t3;
    slot += 1;
  }

  geometry.setAttribute('fishSeed', new THREE.InstancedBufferAttribute(seeds, 4));
  geometry.setAttribute('fishTrait', new THREE.InstancedBufferAttribute(traits, 4));
}

/**
 * The global fish indices belonging to one species, ascending.
 *
 * A pure function of the index, like `buildSchoolAssignment` and for the same
 * reason: the split has to be stable across tiers, reloads and a late
 * `setSpecies` swap, or a baseline capture stops meaning anything.
 */
function instancesForSpecies(species: Species): Int32Array {
  const assignment = buildSchoolAssignment();
  const out: number[] = [];
  for (let i = 0; i < MAX_FISH; i++) {
    if (SPECIES_OF_KIND[SCHOOL_LAYOUT[assignment[i]]] === species) out.push(i);
  }
  return Int32Array.from(out);
}

/** Where the reef species lives, and which way it was authored. */
const REEF_FISH_URL = '/models/dressing/emperor_angelfish.glb';
const REEF_FISH_FORWARD = '+z' as const;

/**
 * Loads the reef species, or resolves null.
 *
 * Null on any failure, deliberately and quietly at `warn`: a missing 400 KB fish
 * must cost the reef its markings and nothing else. The caller keeps the
 * procedural body it was constructed with.
 *
 * The largest primitive wins when the file has several. `emperor_angelfish` is
 * body plus fins, and merging them would need matching attribute sets across a
 * material boundary for no gain — the fins are a fifth of the triangles and the
 * body is what carries the animal.
 */
export async function loadReefFish(assets: AssetLoader): Promise<FishSpecies | null> {
  try {
    const root = await assets.load(REEF_FISH_URL);
    let best: THREE.Mesh | null = null;
    let bestCount = 0;
    root.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      const count = mesh.geometry.getAttribute('position')?.count ?? 0;
      if (count > bestCount) {
        bestCount = count;
        best = mesh;
      }
    });
    if (!best) return null;

    const mesh = best as THREE.Mesh;
    // The glTF node transform is part of the pose, so it has to be baked before
    // the body is measured — otherwise a model authored under a scaled node is
    // normalised against the wrong length.
    mesh.updateWorldMatrix(true, false);
    const posed = mesh.geometry.clone();
    // Meshopt-quantised attributes cannot survive a baked transform — see
    // `dequantiseGeometry`.
    dequantiseGeometry(posed);
    posed.applyMatrix4(mesh.matrixWorld);

    const geometry = normaliseImportedBody(posed, REEF_FISH_FORWARD);
    posed.dispose();
    // Nothing survives of the source geometry's own bounds — every vertex is
    // relocated by the vertex stage — and culling is off for the same reason it
    // is off for the procedural body.
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

    const material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as
      | THREE.MeshStandardMaterial
      | undefined;
    return { geometry, map: material?.map ?? null };
  } catch (error) {
    console.warn(`[ocean] reef fish unavailable: ${REEF_FISH_URL}`, error);
    return null;
  }
}

/** How many entries of an ascending array are strictly below `limit`. */
function countBelow(sorted: Int32Array, limit: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < limit) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Rewrites an imported body into this module's frame.
 *
 * The frame is not negotiable — the wave, the weave and the school's axes are
 * all written against "nose at +0.5x, tail at -0.5x, up +y, lateral +z, one unit
 * long" — so the conversion happens once, here, rather than as a special case
 * scattered through the shader.
 *
 * `forwardAxis` names the source's own long axis and which end the nose is on.
 * `emperor_angelfish` is authored nose-toward +z, which is why the default is a
 * quarter turn about y.
 */
export function normaliseImportedBody(
  source: THREE.BufferGeometry,
  forwardAxis: '+z' | '-z' | '+x' | '-x' = '+z',
): THREE.BufferGeometry {
  const geometry = source.clone();

  if (forwardAxis === '+z') geometry.rotateY(Math.PI / 2);
  else if (forwardAxis === '-z') geometry.rotateY(-Math.PI / 2);
  else if (forwardAxis === '-x') geometry.rotateY(Math.PI);

  geometry.computeBoundingBox();
  const measured = geometry.boundingBox;
  if (!measured) return geometry;

  const length = measured.max.x - measured.min.x;
  if (!(length > 1e-6)) return geometry;

  geometry.scale(1 / length, 1 / length, 1 / length);

  /**
   * Re-measured after the scale, never derived from the measurement before it.
   *
   * `BufferGeometry.scale` goes through `applyMatrix4`, which recomputes a
   * non-null `boundingBox` *in place* — and `computeBoundingBox` hands back the
   * geometry's own `Box3`, not a copy. So a box read before the scale is
   * silently rescaled underneath every later use of it, and multiplying it by
   * the scale again applies the factor twice.
   *
   * That is not a rounding error. `emperor_angelfish` is authored 0.2 m long,
   * so the second application was a factor of five: the body landed at
   * x in -2.46..-1.46 instead of -0.5..0.5, which put the wave's body axis
   * `s = 0.5 - x` at 1.96..2.96 instead of 0..1. The envelope is `WAVE_AMP *
   * s^3`, so the lateral swing went from a designed 0.085 body lengths at the
   * tail to 2.2 at the tail and 0.64 *at the nose* — the fish was sheared into a
   * travelling S about two body lengths deep, which is exactly what an eel does
   * and nothing like what an angelfish does. It also displaced the body off the
   * origin the school frame places and banks it about.
   *
   * Measuring here keeps the scale factor out of the translation altogether, so
   * the mistake has nowhere left to hide.
   */
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box) return geometry;

  // The nose goes exactly on +0.5 so `s = 0.5 - x` starts at zero on the snout.
  // The tail then lands on -0.5, a little short of the -0.69 the procedural
  // caudal fin reaches — which is correct for this animal: an angelfish is a
  // pectoral swimmer with a short, stiff tail, and it should beat less than the
  // pelagic silhouette does.
  geometry.translate(
    0.5 - box.max.x,
    -(box.min.y + box.max.y) * 0.5,
    -(box.min.z + box.max.z) * 0.5,
  );

  geometry.computeBoundingSphere();
  return geometry;
}

/**
 * Which school each of the `MAX_FISH` instances belongs to.
 *
 * A pure function of the index — no seed, no randomness — because that is the
 * whole requirement: fish `i` must be in the same school whatever the tier is,
 * so that `setCount` thins the schools rather than reshuffling them.
 *
 * The rule is "give this fish to the emptiest school that has activated by now",
 * scanned in `SCHOOL_LAYOUT` order. Two properties fall out of it. A school that
 * has just activated is empty, so it fills first and catches up within a dozen
 * fish rather than staying starved for the rest of the buffer. And between
 * activations the assignment is a plain round robin, so the active schools stay
 * within one fish of each other at every prefix length — which is what makes
 * "40 fish across four schools" come out as exactly ten each rather than as a
 * tail-heavy distribution nobody chose.
 *
 * `SCHOOLS` is ten and `MAX_FISH` is 480, so this is five thousand comparisons,
 * once, at construction.
 */
function buildSchoolAssignment(): Uint8Array {
  const counts = new Int32Array(SCHOOLS);
  const out = new Uint8Array(MAX_FISH);

  for (let i = 0; i < MAX_FISH; i++) {
    let active = 0;
    while (active < SCHOOLS && SCHOOL_ACTIVATION[active] <= i) active++;

    let best = 0;
    for (let k = 1; k < active; k++) {
      if (counts[k] < counts[best]) best = k;
    }

    out[i] = best;
    counts[best]++;
  }

  return out;
}

/** Maps a uniform draw in [0, 1) to one biased toward 0.5, preserving the range. */
function centreWeighted(u: number): number {
  const s = u * 2 - 1;
  return (Math.sign(s) * Math.pow(Math.abs(s), 1.5)) * 0.5 + 0.5;
}

function identityInstanceMatrix(matrices: THREE.InstancedBufferAttribute, count: number): void {
  const array = matrices.array as Float32Array;
  for (let i = 0; i < count; i++) {
    const base = i * 16;
    array[base] = 1;
    array[base + 5] = 1;
    array[base + 10] = 1;
    array[base + 15] = 1;
  }
  matrices.needsUpdate = true;
}

/**
 * Selects one of `SCHOOLS` uniforms by a float instance attribute.
 *
 * A `select` chain rather than an indexed uniform array: the count is a
 * compile-time constant, the condition is uniform across every vertex of an
 * instance, and this lowers to a handful of ternaries on both backends with no
 * dependence on how a given WebGL2 driver handles dynamic array indexing.
 */
function pickPerSchool(values: readonly Node[], index: Node): Node {
  let node = values[values.length - 1];
  for (let k = values.length - 2; k >= 0; k--) {
    node = select(index.lessThan(k + 0.5), values[k], node);
  }
  return node;
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Reduces a phase to [0, 2pi) in float64, before it reaches a float32 uniform. */
function wrapTau(phase: number): number {
  const tau = Math.PI * 2;
  return ((phase % tau) + tau) % tau;
}

/** Shortest signed difference between two headings. */
function wrapPi(angle: number): number {
  const tau = Math.PI * 2;
  return ((((angle + Math.PI) % tau) + tau) % tau) - Math.PI;
}
