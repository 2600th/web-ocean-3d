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
  sin,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import { mulberry32 } from '../core/random';
import { ISLAND, seafloorHeight } from './Seafloor';

/**
 * Kelp and seagrass on the shallow bottom, animated entirely on the GPU.
 *
 * The submerged scene had rocks, fish and suspended matter and nothing that
 * grows. That is a specific kind of wrong rather than a missing feature: rocks
 * are static because rocks *are* static, so a seafloor made only of rock and
 * sand reads as a diorama however good the light on it is. A bed of weed is the
 * one thing down there that moves with the water, and it is therefore the thing
 * that tells the viewer the water is moving at all.
 *
 * Four ideas carry the module.
 *
 * **The sway is the swell, not a decoration.** Under a surface gravity wave the
 * water near the bed sweeps back and forth along the wave's direction of travel
 * with an excursion of `a / sinh(k h)` — linear wave theory, exactly the field
 * the ocean surface is built from. So the bed is driven by the same wavelength,
 * the same direction and the same clock the surface is, and the spatial term
 * `k · x` means the sway arrives at each plant as a travelling wave rather than
 * as a bed of metronomes. When the swell turns, the weed turns with it; when it
 * lengthens, the weed slows down. Nothing here is tuned to *resemble* the water.
 *
 * **A plant is a cantilever.** The response envelope is `s²` with `s` running 0
 * at the holdfast to 1 at the tip, which is the static deflection of a beam
 * under a distributed load and, more to the point, is exactly zero and exactly
 * flat at the anchor. A plant whose base moves at all reads as a decal sliding
 * on the sand. The phase then lags along the length, so the tip trails the base
 * and the plant carries its own travelling wave — that lag is the whole
 * difference between weed and a windscreen wiper.
 *
 * **Everything is one instanced draw and one pure function of two phases.**
 * Geometry is a single blade built in code; every blade in the forest is that
 * blade placed and deformed by the vertex stage from three per-instance vec4s.
 * The CPU writes two scalars per frame and never touches a transform. There is
 * no integrator and no remembered position, so `resetClock(t)` lands on the pose
 * for `t` exactly, which is what the visual regression harness needs.
 *
 * **Placement is derived from the heightfield, never from coordinates.** The
 * island is at a position and a radius this file does not know and must not
 * assume — it is `ISLAND`, and it changes. Sites are rejection-sampled against
 * `seafloorHeight`: a light-limited depth band, a slope limit, and a preference
 * for hollows over exposed flank. Move the island or deepen the shelf and the
 * beds follow, because the beds were never anywhere else.
 *
 * No compute, no storage textures, no per-frame allocation: the same node graph
 * has to compile on the WebGL2 fallback.
 *
 * Known limitation: the blades do not collide with anything. A reef boulder can
 * stand in the middle of a bed and have kelp growing through it. Fixing that
 * needs the prop transforms, which live in another module, for an intersection
 * that is a few blades out of thousands.
 */

/**
 * TSL node objects are structurally dynamic and the generated typings cannot
 * express per-component expressions built out of `attribute()`. Node-typed
 * locals are therefore `any` by design — the class's public API stays typed.
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
 * shared ones are: a baseline capture is only comparable to a later run if the
 * forest never drifts, and a drift has to be visible in a diff.
 */
const KELP_SEED = 0x6b31f5;

/** Standard gravity, for the deep-water dispersion relation. */
const GRAVITY = 9.81;

// ------------------------------------------------------------------ the budget

/**
 * Blades the instance buffers are built for. `setCount` draws a prefix.
 *
 * 4000 blades at 32 triangles is 128k triangles in one draw — about a sixth of
 * what the ocean surface costs at the same tier, for the thing that makes the
 * bottom of that ocean look inhabited. It is affordable because the blade is
 * nine rows of three vertices and because the far fade below means most of the
 * population rasterises nothing at all.
 */
const MAX_BLADES = 4000;

/**
 * Blades one patch is allowed.
 *
 * The budget is spent on *few dense patches* rather than on many thin ones, and
 * that is the single most important placement decision in the file. Weed grows
 * in beds: a hundred blades scattered over a hectare is litter, and the same
 * hundred inside a twelve-metre disc is a bed you can swim into. It also matters
 * for the tier ladder — dropping a tier removes whole patches and leaves the
 * survivors at full density, where thinning every patch equally would leave the
 * whole forest looking half-dead.
 */
const BLADES_PER_PATCH = 92;

// --------------------------------------------------------------- the placement

/** Sample regions, cycled so a count prefix is a fair sample of all of them. */
const REGION_ISLAND = 0;
const REGION_PLATEAU = 1;

/**
 * Island sampling annulus, in units of `ISLAND.radius`.
 *
 * Wider than the band that will actually be accepted, and expressed in radii
 * rather than metres, because the island's shape is another module's and is
 * being changed *while this is being written*. What this file relies on is only
 * that the shelf runs from dry land out to deep water somewhere in this bracket,
 * and that `seafloorHeight` will say where.
 *
 * That is not a hypothetical robustness argument. Between one revision and the
 * next the island's radius went from 260 m to 500 m, its summit from 30 m to
 * 72 m and then to 150 m, and its coastline from a circle to a bay, a headland
 * and a spit — and
 * the only thing that had to change here was this bracket, because everything
 * downstream reads the heightfield rather than a coordinate. On the shape as it
 * now stands the mean waterline sits at about one radius and the accepted sites
 * land between 0.86 and 1.39 of it, which 0.7 to 1.4 brackets with room on both
 * sides. Roughly 30% of the samples inside it are plantable.
 */
const ISLAND_INNER = 0.7;
const ISLAND_OUTER = 1.4;

/**
 * Origin plateau sampling annulus, metres.
 *
 * The inner edge clears the spawn point, where the ship sits and drops anchor;
 * the outer edge is past the reef scatter and out to where the plateau starts
 * falling away, which the depth test then trims to whatever is actually shallow.
 */
const PLATEAU_INNER = 24;
const PLATEAU_OUTER = 300;

/**
 * What grows where, cycled one patch at a time.
 *
 * Two plants over two regions, except that the fourth entry is not the one it
 * looks like it should be, and the reason is worth stating: **the origin plateau
 * has no shallow water on it at all.** Sampled over the annulus above, its floor
 * runs from 14.5 m to 21.0 m and never once comes up past 14 — the heightfield's
 * relief is nowhere near as wide as its amplitude suggests, because four octaves
 * of value noise concentrate hard around their mean. A seagrass meadow there
 * would be a seagrass meadow in twilight, which is not a thing.
 *
 * So the plateau's low growth is understorey turf rather than a meadow: the same
 * short many-bladed tuft, planted in the band the plateau actually offers, and
 * tinted between the two palettes because red and brown algae are what grows
 * under a kelp canopy at that depth. It is also, incidentally, the more useful
 * thing to have there — the reef's kelp stands need something between them, and
 * bare sand between them is what the scene already had.
 *
 * The depth bands themselves are the light limit. Kelp needs roughly 1% of
 * surface irradiance to hold a canopy, which in water this turbid is the top
 * twenty metres; seagrass is a higher-light plant and sits shallower still.
 * These are why a bed appears where the floor comes up and stops where it falls
 * away, and they are the cheapest possible way to make growth look like it chose
 * where to be.
 */
interface PatchSpec {
  region: number;
  /** Tall stipes off a tight holdfast, or a short many-bladed tuft. */
  tall: boolean;
  /** 0 = seagrass green, 1 = kelp brown. Also sets the lag and the twist. */
  kind: number;
  minDepth: number;
  maxDepth: number;
}

const PATCH_CYCLE: readonly PatchSpec[] = [
  { region: REGION_ISLAND, tall: true, kind: 1, minDepth: 4.0, maxDepth: 19.0 },
  { region: REGION_PLATEAU, tall: true, kind: 1, minDepth: 4.0, maxDepth: 19.0 },
  { region: REGION_ISLAND, tall: false, kind: 0, minDepth: 1.6, maxDepth: 11.0 },
  { region: REGION_PLATEAU, tall: false, kind: 0.4, minDepth: 13.0, maxDepth: 21.0 },
];

/**
 * Slope at which a site scores zero for shelter, as a gradient.
 *
 * Not a rejection threshold — a soft one, and set from a measurement rather than
 * from taste. Over the plantable depths, the origin plateau's gradient sits near
 * 0.05 and the island's shelf runs at a median of 0.22 with a ninetieth
 * percentile of 0.33: the island is genuinely the exposed ground and the plateau
 * genuinely the sheltered one, which is the distinction this is here to express.
 *
 * At 0.4 that distinction collapsed the wrong way — nine tenths of the island
 * scored at the 0.05 floor, so the shelf had no internal gradient left and the
 * flank was as thin as the crest of a boulder. 0.6 leaves the island's median at
 * 0.17 against the plateau's 0.42, which is a bed that thins toward the exposed
 * slope rather than one that gives up on it.
 */
const SLOPE_LIMIT = 0.6;

/**
 * Metres of local concavity that take a site from open ground to full shelter.
 *
 * Measured on a 22 m stencil, which is a tenth of the heightfield's coarsest
 * feature and so resolves the dips between bumps rather than the shelf itself.
 * The relief over that stencil runs to roughly ±0.9 m, so this is a little over
 * the full swing and the score never saturates at either end.
 */
const HOLLOW_SCALE = 1.2;

/** Half-step of the slope probe and radius of the concavity stencil, metres. */
const SLOPE_PROBE = 7;
const HOLLOW_PROBE = 22;

/**
 * Candidate sites tried per patch before the best one so far is taken.
 *
 * The depth test costs one heightfield evaluation and runs first; only survivors
 * pay the six evaluations the shelter score needs. At 160 tries the whole
 * forest's placement is a few thousand evaluations at construction, which is
 * under a millisecond and happens once.
 */
const SITE_TRIES = 160;

/** Shallowest water a plant will be seated in, metres. */
const MIN_PLANT_DEPTH = 1.2;

/** Water a blade tip must keep above it, metres. See `plantHeight`. */
const TIP_CLEARANCE = 0.7;

/**
 * Metres the holdfast is pushed into the sand.
 *
 * The floor *mesh* is a 256-segment grid over a 4 km extent, so it samples
 * `seafloorHeight` every fifteen metres and lerps between; the analytic field
 * this module seats against curves away from that by a decimetre or two in the
 * worst places. A plant seated at exactly the analytic height therefore hovers
 * over the rendered sand about half the time, and a hovering plant is instantly
 * a bug. Burying the holdfast costs nothing — the base of a blade is inside the
 * holdfast anyway.
 */
const HOLDFAST_SINK = 0.35;

// ---------------------------------------------------------------- the geometry

/**
 * Rows along the blade, and the three columns lofted between them.
 *
 * Nine rows means eight spans over a lag of up to 2.4 rad, so the travelling
 * wave is sampled about every 0.3 rad and the silhouette curves rather than
 * kinks. Three columns rather than two because a flat ribbon *disappears*
 * edge-on: the middle column is pushed out of the plane into a shallow midrib,
 * which both catches light from either side and gives `computeVertexNormals`
 * something to work with.
 */
const BLADE_ROWS = 9;
/** Depth of the midrib, as a fraction of the blade's width. */
const BLADE_CURL = 0.24;

/**
 * Row spacing exponent. Below 1 it packs rows toward the tip.
 *
 * That is where they are needed: the envelope is quadratic and the phase lag is
 * linear, so both the displacement and its curvature are largest in the last
 * third of the blade, and a uniform split spends half its rows on the part that
 * barely moves.
 */
const ROW_BIAS = 0.88;

// -------------------------------------------------------------------- the sway

/**
 * Ceiling on `1 / sinh(k h)`, the near-bed excursion per unit wave amplitude.
 *
 * The expression is exact linear theory and its shallow-water limit is `1/(k h)`,
 * which diverges as the water gets thin. A real plant in that regime is not
 * sweeping an unbounded arc — it is limited by its own stiffness and by the wave
 * that shallow having already broken. The cap is physics, not a guard against
 * a divide.
 */
const REACH_MAX = 1.6;

/**
 * Asymptote on tip displacement, as a fraction of blade height.
 *
 * A soft limit — `tanh` — and not a clamp, for two reasons.
 *
 * The first is that it has to apply to the *sum* of the wave components and the
 * mean lean, not to each of them. Capping them individually left the total free
 * to reach 1.09 of the blade's height in a 21 m/s sea, which is a blade being
 * carried further from its holdfast than its own length: it would have to
 * stretch to do it, and the arc-length correction below would go imaginary
 * trying to pay for it.
 *
 * The second is that a hard clamp puts a corner in the displacement, and the
 * surface normal is the analytic derivative of that displacement — so the corner
 * would arrive as a band of visibly wrong shading sweeping up the blade every
 * time a gust saturated it. `tanh` is smooth everywhere and its derivative is
 * `1 - tanh²`, which the slope term already has to hand.
 *
 * 0.7 puts a 21 m/s storm at about 40 degrees of lean and leaves anything under
 * a fresh breeze essentially untouched — `tanh(x) ≈ x` for the small arguments
 * ordinary conditions produce.
 */
const BEND_MAX = 0.7;

/**
 * Phase lag from holdfast to tip, radians.
 *
 * The single detail that separates weed from a hinged flap. Kelp is a long
 * flexible strap and its tip trails the base by most of a half cycle; a
 * seagrass blade is thirty centimetres of stiff leaf and barely lags at all.
 * Interpolated by the blade's kind, so one constant covers both.
 */
const GRASS_LAG = 0.55;
const KELP_LAG = 2.4;

/**
 * Roll of the blade about its own length, radians.
 *
 * Driven by the *cosine* of the sway phase, so it is in quadrature with the
 * displacement — a blade rolls hardest as it sweeps through the middle of its
 * stroke, which is where the flow across it is fastest. Free, because the cosine
 * is already computed for the slope, and it is what stops a bed of ribbons from
 * flashing as they pass edge-on.
 */
const TWIST = 0.9;

/**
 * Spread of each blade's own swing axis away from the swell direction.
 *
 * Real beds are stirred by turbulence at the scale of the plants, not by a
 * laminar sheet, so no two neighbours swing on quite the same line. Drawn from a
 * per-blade scalar that is independent of the blade's yaw, so the *mean* axis
 * over the bed is still exactly the swell direction — using the yaw itself
 * biased the whole forest toward +x regardless of where the swell was coming
 * from, because `cos(yaw)` weighted by `cos(yaw)` does not average to zero.
 */
const FLOW_JITTER = 0.4;

/**
 * The background swell, as a wavelength in metres.
 *
 * The presets run the peak wavelength from 20 m to 80 m, and at the short end
 * `1 / sinh(k h)` has closed the near-bed motion down to a few percent — which
 * is correct, chop genuinely does not reach the bottom, and would leave the weed
 * standing dead still on a windy day. What actually stirs a bed in those
 * conditions is the long-period swell that is present in any real sea whatever
 * the local wind is doing, so it is modelled as a second component: much longer,
 * much smaller, and permanent.
 */
const SURGE_WAVELENGTH = 200;
const SURGE_K = (Math.PI * 2) / SURGE_WAVELENGTH;
const SURGE_OMEGA = Math.sqrt(GRAVITY * SURGE_K);

// ------------------------------------------------------------------ the colour

/** Kelp: dark olive at the holdfast, amber at the tip where the blade thins. */
const KELP_BASE_COLOR = new THREE.Color(0.2, 0.16, 0.07);
const KELP_TIP_COLOR = new THREE.Color(0.44, 0.36, 0.13);
/** Seagrass: a colder, greener plant, and a much shorter gradient. */
const GRASS_BASE_COLOR = new THREE.Color(0.09, 0.16, 0.09);
const GRASS_TIP_COLOR = new THREE.Color(0.26, 0.42, 0.18);

/**
 * Skylight the bed sits in, and the direct sun on top of it.
 *
 * The same pair `Fish` uses, deliberately. Two submerged systems lit by
 * different ambients read as two different bodies of water sharing a frame.
 */
const AMBIENT_COLOR = new THREE.Color(0.24, 0.34, 0.38);
const SUN_COLOR = new THREE.Color(0.9, 0.88, 0.78);
const SUN_STRENGTH = 0.7;

/**
 * Light coming *through* a backlit blade.
 *
 * A kelp blade is a fraction of a millimetre thick and glows when the sun is
 * behind it. Underwater the sun is always more or less above, so this is on
 * every blade the viewer looks up at — which is most of them, since the viewer
 * is usually swimming above the bed rather than in it.
 */
const TRANSLUCENCY = 0.35;

/**
 * Attenuation of downwelling light with the plant's own depth, per metre.
 *
 * Zero now, and kept as a named constant so the history is legible. This used
 * to be the only place in the scene that knew light weakens on its way down —
 * `UnderwaterPass` integrated extinction along the *view* ray alone, so without
 * it a bed at nineteen metres was lit as brightly as one at two. The pass now
 * applies the column above every pixel's own world position (`uDownwell`), per
 * channel, to the whole submerged scene, and keeping this as well would darken
 * the weed twice and nothing else once.
 */
const DOWNWELLING_SIGMA = 0;

/**
 * Ambient occlusion at the holdfast.
 *
 * The inside of a bed is dark — that is why holdfasts are where the crabs live.
 * Driven by the blade coordinate rather than by any real occlusion query, which
 * is enough because the gradient is what reads, not its accuracy.
 */
const BASE_OCCLUSION = 0.42;

// ------------------------------------------------------------------- the fade

/**
 * Distance at which a plant starts to collapse toward its holdfast, metres.
 *
 * This is the module's culling. The population spans two beds fourteen hundred
 * metres apart, so no single bounding sphere can usefully reject either of them
 * — the viewer is nearly always inside it. Scaling a distant plant to zero in
 * the vertex stage is better than a cull anyway: it rasterises nothing, and it
 * removes the failure the alternative has, which is that a three-metre blade
 * two hundred metres away is a sub-pixel triangle and a bed of them is a field
 * of aliasing sparkle seen from the deck.
 *
 * Underwater visibility is around 38 m, so from below this is never reached
 * before the fog has already taken the bed. From above, through the surface,
 * 120 m of refracted water is well past anything resolvable.
 */
const FADE_FULL = 120;
const FADE_GONE = 210;

// --------------------------------------------------------------- instance data

/** One blade, before it is packed into the instance buffers. */
interface Blade {
  /** Holdfast, world space, already sunk into the sand. */
  x: number;
  y: number;
  z: number;
  /** Metres, tip to holdfast. */
  height: number;
  /** Metres across at the widest station. */
  width: number;
  yaw: number;
  /** Water depth at the holdfast, for the orbital attenuation and the light. */
  depth: number;
  /** 0 = seagrass, 1 = kelp. Interpolates the colour, the lag and the twist. */
  kind: number;
  /** -1..1. Drives the per-blade tint *and* its own swing axis. */
  variation: number;
  /** Radians, so no two blades in a plant are in step. */
  phase: number;
}

export interface KelpForestOptions {
  /** Overrides the seeded placement; useful for A/B-ing a layout. */
  seed?: number;
}

/**
 * Kelp and seagrass beds on the shallow bottom.
 *
 * Add `object` to the scene. It carries an identity transform and the vertex
 * stage emits world coordinates directly, so it must be parented to something
 * untransformed — the scene root — exactly as `FishSchool` is.
 */
export class KelpForest {
  /** Blades the buffers hold. `setCount` is clamped to this. */
  static readonly MAX_COUNT = MAX_BLADES;

  readonly object: THREE.Object3D;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;

  private count: number;
  private wantVisible = true;
  private disposed = false;

  /**
   * Angular rate of the driving swell, rad/s, from the deep-water dispersion
   * relation. Kept so `resetClock` can reconstruct the phase from a time.
   */
  private swellOmega = Math.sqrt(GRAVITY * ((Math.PI * 2) / 47));

  /**
   * Wave phases, radians in [0, 2pi), *accumulated* rather than recomputed.
   *
   * `omega` changes whenever the preset or the wind does. Recomputing the phase
   * as `omega * t` would then step the whole bed to a different point in its
   * cycle at that instant — a visible flinch across the forest, and one the
   * surface it is supposed to be following does not take, because changing a
   * spectrum changes the amplitude of each mode and not its phase.
   *
   * `resetClock` reconstructs both from the clock, which is exact for a capture
   * that applies its preset before rewinding — which is the order the harness
   * uses.
   */
  private swayPhase = 0;
  private surgePhase = 0;

  // --- uniforms -------------------------------------------------------------
  private readonly uSwayPhase = uniform(0);
  private readonly uSurgePhase = uniform(0);
  /** Unit XZ vector the swell travels along. */
  private readonly uSwellDir = uniform(new THREE.Vector2(1, 0));
  /** Wavenumber of the driving swell, rad/m. */
  private readonly uSwellK = uniform((Math.PI * 2) / 47);
  /** Wave amplitude at the surface, metres — the thing the excursion scales. */
  private readonly uSwellAmp = uniform(0.6);
  private readonly uSurgeAmp = uniform(0.4);
  /**
   * Mean lean along the flow, as a fraction of blade height.
   *
   * Stokes drift and the tidal set: a bed is never upright on average, it leans
   * downstream and oscillates about that. Without it the rest pose is a lawn of
   * vertical sticks, which is the pose weed is in for exactly no part of a wave
   * cycle.
   */
  private readonly uLean = uniform(0.07);

  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(SUN_COLOR));
  private readonly uAmbient = uniform(new THREE.Color(AMBIENT_COLOR));
  private readonly uKelpBase = uniform(new THREE.Color(KELP_BASE_COLOR));
  private readonly uKelpTip = uniform(new THREE.Color(KELP_TIP_COLOR));
  private readonly uGrassBase = uniform(new THREE.Color(GRASS_BASE_COLOR));
  private readonly uGrassTip = uniform(new THREE.Color(GRASS_TIP_COLOR));

  constructor(count: number, options: KelpForestOptions = {}) {
    this.count = clampCount(count);

    const blades = placeBlades(options.seed ?? KELP_SEED);

    this.geometry = buildBladeGeometry();
    attachInstanceAttributes(this.geometry, blades);
    this.geometry.instanceCount = this.count;
    this.geometry.boundingSphere = boundBlades(blades);

    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'kelp-blades';
    // Culled, unlike the fish and the birds, because unlike them the forest is
    // genuinely stationary and its bound is genuinely correct. It only ever
    // rejects the whole draw from well outside both beds, which is the one case
    // the vertex-stage fade cannot help with — the fade still has to run the
    // vertex stage to decide to collapse.
    this.mesh.frustumCulled = true;
    // Shadows off in both directions. A 0.2 m blade casts nothing legible under
    // this much water, and putting four thousand of them through the depth pass
    // would double the whole system's vertex cost to render it.
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();

    this.object = new THREE.Object3D();
    this.object.name = 'kelp-forest';
    // Positions are emitted in world space, so the container never moves and
    // never needs its matrix recomputed.
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Blades drawn, 0..`KelpForest.MAX_COUNT`.
   *
   * Free: the buffers are built once at `MAX_BLADES` and this only moves the
   * draw's instance count. Blade `i` therefore keeps its patch, plant, height
   * and phase whatever the tier is, so a tier change removes whole patches from
   * the tail of the layout and leaves every surviving bed at full density —
   * see `BLADES_PER_PATCH`.
   */
  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.count) return;
    this.count = next;
    this.geometry.instanceCount = next;
    this.applyVisibility();
  }

  setVisible(v: boolean): void {
    this.wantVisible = v;
    this.applyVisibility();
  }

  /** `dir` points *toward* the sun and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(dir);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();
  }

  /**
   * The sea the bed is standing in.
   *
   * Takes the *same three numbers the surface takes* — `preset.sea.windDirection`,
   * `state.windSpeed`, `state.peakWavelength` — rather than reaching into
   * `OceanSimulation` for them. That is the whole point: the water and the weed
   * agree because they are handed the same description of the sea, not because
   * one of them samples the other.
   *
   * What is derived here rather than passed in:
   *
   *   - `k` from the wavelength, and `omega` from `k` by the deep-water
   *     dispersion relation `omega² = g k`. The bed's period is therefore the
   *     swell's period, and a long swell visibly rolls through it more slowly
   *     than a short one.
   *   - the wave amplitude from the wind, by Pierson-Moskowitz's fully-developed
   *     `Hs = 0.0246 U²`. Taken at 0.4 Hs rather than 0.5, because a random sea's
   *     typical orbital amplitude is below its significant one.
   *   - the background swell's amplitude and the mean lean, both of which grow
   *     with the wind but neither of which goes to zero with it. A dead calm
   *     still has a bed that breathes, because a dead calm still has swell.
   */
  setSwell(bearingRadians: number, windSpeed: number, peakWavelength = 47): void {
    const dir = this.uSwellDir.value as THREE.Vector2;
    dir.set(Math.cos(bearingRadians), Math.sin(bearingRadians));

    const lambda = clampNumber(peakWavelength, 8, 400);
    const k = (Math.PI * 2) / lambda;
    this.uSwellK.value = k;
    this.swellOmega = Math.sqrt(GRAVITY * k);

    const u = Math.max(0, windSpeed);
    const significant = Math.min(9, 0.0246 * u * u);
    this.uSwellAmp.value = significant * 0.4;
    this.uSurgeAmp.value = 0.18 + 0.03 * u;
    this.uLean.value = Math.min(0.16, 0.05 + 0.006 * u);
  }

  /**
   * Advances the two wave phases.
   *
   * Runs whether or not the bed is visible, for the same reason `FishSchool`'s
   * does: making the clock depend on visibility would make the pose depend on
   * the history of `setVisible` calls, which is exactly the hidden state
   * `resetClock` exists to rule out. Two scalar writes, no allocation, no
   * heightfield evaluation — the placement was solved once at construction.
   */
  update(dt: number): void {
    if (this.disposed) return;
    this.swayPhase = wrapTau(this.swayPhase + this.swellOmega * dt);
    this.surgePhase = wrapTau(this.surgePhase + SURGE_OMEGA * dt);
    this.pushPhases();
  }

  /**
   * Jumps the bed to the pose for `time`, for reproducible captures.
   *
   * Exact, not approximate: every blade's position and normal is a closed-form
   * function of these two phases and its own constants, so there is no
   * integrated state to be out of step.
   */
  resetClock(time = 0): void {
    if (this.disposed) return;
    this.swayPhase = wrapTau(time * this.swellOmega);
    this.surgePhase = wrapTau(time * SURGE_OMEGA);
    this.pushPhases();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.remove(this.mesh);
    this.object.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }

  // ------------------------------------------------------------------ internals

  /** An empty forest is hidden outright, so it costs not even a culling test. */
  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.count > 0;
  }

  private pushPhases(): void {
    this.uSwayPhase.value = this.swayPhase;
    this.uSurgePhase.value = this.surgePhase;
  }

  private buildMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    material.name = 'kelp-blade';
    // A blade is a sheet with no thickness, so both faces have to rasterise.
    material.side = THREE.DoubleSide;
    // Opaque, and that is a lighting decision rather than a performance one.
    // `UnderwaterPass` reconstructs the water column from the *depth buffer*, so
    // anything that does not write depth is never fogged — a blended bed would
    // hang in front of the sand at full brightness and glow through fifty metres
    // of water that has already taken the rocks behind it. Writing depth is what
    // buys the weed the same Beer-Lambert extinction as everything else down
    // there, for free.
    material.transparent = false;
    material.depthWrite = true;
    // Vertex fog would then double it. See above.
    material.fog = false;

    const anchor: Node = attribute('kelpAnchor', 'vec4');
    const shape: Node = attribute('kelpShape', 'vec4');
    const trait: Node = attribute('kelpTrait', 'vec4');
    const p: Node = positionGeometry;
    const n: Node = normalGeometry;

    /** Blade coordinate: 0 at the holdfast, 1 at the tip. */
    const s = p.y;
    const height = anchor.w;
    const width = shape.x;
    const waterDepth = shape.w;
    const kind = trait.x;
    const variation = trait.y;

    // --- the blade's own frame ---------------------------------------------
    // `across` spans the blade's width, `face` is its resting normal. Both are
    // horizontal: a blade is authored standing up and is only ever tilted by the
    // sway, which is applied as a shear below rather than as a rotation of this
    // frame.
    const across = vec3n(shape.y, 0, shape.z);
    const face = vec3n(shape.z.negate(), 0, shape.y);
    const up = vec3(0, 1, 0);

    // --- the driving flow ---------------------------------------------------
    //
    // The swell direction, perturbed per blade so a bed is stirred rather than
    // swept. See `FLOW_JITTER`.
    const flow = vec3n(this.uSwellDir.x, 0, this.uSwellDir.y)
      .add(across.mul(variation.mul(FLOW_JITTER)))
      .normalize();

    // Distance along the wave's direction of travel, which is what turns a bed
    // of blades into a wave passing over one. Without it every plant in a
    // fifteen-metre patch is in step and the whole thing pulses.
    const downwave = anchor.x.mul(this.uSwellDir.x).add(anchor.z.mul(this.uSwellDir.y));

    // `1 / sinh(k h)` as `2 e^-kh / (1 - e^-2kh)`. Exact linear theory, in a form
    // with one exponential and no `sinh` — which WGSL has and GLSL ES 3.0 does
    // not, so writing it out is what keeps the same graph compiling on both.
    const reach = (k: Node, h: Node): Node => {
      const e = k.mul(h).negate().exp();
      return e.mul(2).div(float(1).sub(e.mul(e)).max(1e-3)).min(REACH_MAX);
    };

    // Tip displacement as a fraction of blade height: how far the water moves,
    // over how long the blade is. Unclamped here — the two components and the
    // mean lean are limited together, below, because it is their sum that a
    // blade cannot exceed. See `BEND_MAX`.
    const bendA = this.uSwellAmp.mul(reach(this.uSwellK, waterDepth)).div(height);
    const bendB = this.uSurgeAmp.mul(reach(float(SURGE_K), waterDepth)).div(height);

    // Lag along the length. Kelp whips, seagrass does not — see `KELP_LAG`.
    const lag = mix(float(GRASS_LAG), float(KELP_LAG), kind);

    const angA = this.uSwayPhase.add(trait.w).sub(downwave.mul(this.uSwellK)).sub(s.mul(lag));
    const angB = this.uSurgePhase
      .add(trait.w.mul(0.6))
      .sub(downwave.mul(SURGE_K))
      .sub(s.mul(lag).mul(0.55));

    const sinA = sin(angA);
    const cosA = cos(angA);
    const sinB = sin(angB);
    const cosB = cos(angB);

    // The two wave components plus the mean set, softly limited to a blade's
    // reach. Everything below is this one number and its derivative.
    const raw = bendA.mul(sinA).add(bendB.mul(sinB)).add(this.uLean);
    const sat = raw.div(BEND_MAX).tanh();
    const wave = sat.mul(BEND_MAX);
    // Quadratic cantilever envelope: exactly zero, with exactly zero slope, at
    // the holdfast. See the header.
    const sway = s.mul(s).mul(wave);

    // d(sway)/ds, in closed form. `d/ds [s² w(s)] = 2 s w + s² w'`, with
    // `w' = sech²(raw/M) · raw'` and `raw' = -lag (A cos a + 0.55 B cos b)`,
    // since the lag is the only part of the phase that depends on s. `sech²` is
    // `1 - tanh²`, so the limiter's derivative costs one multiply against a
    // value already computed.
    //
    // Both terms of the product rule are kept: dropping the envelope's own
    // contribution leaves the base of the blade lit as though it were straight.
    const dRaw = bendA.mul(cosA).add(bendB.mul(cosB).mul(0.55)).mul(lag.negate());
    const dWave = float(1).sub(sat.mul(sat)).mul(dRaw);
    const slope = s.mul(2).mul(wave).add(s.mul(s).mul(dWave));

    // --- the roll -----------------------------------------------------------
    // In quadrature with the sway, and stronger on kelp than on grass. Applied
    // as a rotation of the cross-section, so the authored normal rotates with it
    // by the same two lines and needs no correction of its own.
    const twist = cosA.mul(TWIST).mul(s).mul(kind.mul(0.7).add(0.3));
    const tc = cos(twist);
    const ts = sin(twist);

    const localX = p.x.mul(tc).sub(p.z.mul(ts));
    const localZ = p.x.mul(ts).add(p.z.mul(tc));

    // --- placement ----------------------------------------------------------
    //
    // A blade is inextensible. Leaning has to cost height or the plant grows as
    // it sways, which reads as the bed breathing in and out. `1 - sway²/2` is
    // the second-order arc-length correction — the same thing `sqrt(1 - sway²)`
    // says, without the square root and without the domain check.
    const shrink = float(1).sub(sway.mul(sway).mul(0.5));

    // See `FADE_FULL`. Collapsing the whole plant toward its holdfast costs no
    // fragments at all, unlike an alpha fade, which would also drag the forest
    // into the transparent pass and out of the depth buffer the fog needs.
    const fade = smoothstepDown(
      anchor.xyz.sub(cameraPosition).length(),
      float(FADE_FULL),
      float(FADE_GONE),
    );

    const offset = across
      .mul(localX.mul(width))
      .add(face.mul(localZ.mul(width)))
      .add(up.mul(s.mul(height).mul(shrink)))
      .add(flow.mul(sway.mul(height)));

    material.positionNode = anchor.xyz.add(offset.mul(fade));

    // --- the normal ---------------------------------------------------------
    //
    // The sway is a displacement that depends only on height, i.e. a shear
    // `p += flow * sway(y)`. Its Jacobian is `I + slope (flow ⊗ e_y)`, and
    // because `flow` is horizontal the inverse transpose collapses to
    // `n - e_y slope (flow · n)` — two operations, and without them a swaying
    // blade is lit as though it were rigid.
    //
    // The `shrink` term is a second shear in the same axis and is deliberately
    // absent: it is second order in the sway and its effect on the normal is
    // below the precision anything downstream cares about.
    const nLocal = vec3n(n.x.mul(tc).sub(n.z.mul(ts)), n.y, n.x.mul(ts).add(n.z.mul(tc)));
    const nWorld = across.mul(nLocal.x).add(up.mul(nLocal.y)).add(face.mul(nLocal.z));
    const bent = nWorld.sub(up.mul(slope.mul(flow.dot(nWorld))));

    const vNormal: Node = varying(bent, 'kelpNormal');

    material.colorNode = Fn(() => {
      // `faceDirection` rather than two-sided lighting: a blade is genuinely
      // seen from both faces and its normal is genuinely flipped on one of them.
      const normal = vNormal.normalize().mul(faceDirection).toVar();
      const ndl = normal.dot(this.uSunDir).toVar();

      // Wrapped diffuse, not a clamped one. A hard terminator across a blade two
      // centimetres wide is not shading, it is a flicker as the twist carries it
      // past — the terminator is narrower than the blade.
      const key = ndl.mul(0.5).add(0.5).toVar();
      const through = ndl.negate().max(0).mul(TRANSLUCENCY).toVar();
      // Skylight arrives from above; a constant ambient flattens a curved blade
      // into a cut-out strip.
      const sky = normal.y.mul(0.5).add(0.5).toVar();

      const light = this.uAmbient
        .mul(sky.mul(0.55).add(0.45))
        .add(this.uSunColor.mul(key.add(through)).mul(SUN_STRENGTH));

      // Base-to-tip gradient, then the plant kind. Both interpolations rather
      // than a branch: `kind` is a per-instance constant, so a `select` would
      // buy nothing and a `mix` lets the two populations share every line.
      const gradient = s.smoothstep(0.05, 0.85).toVar();
      const base = mix(
        mix(this.uGrassBase, this.uGrassTip, gradient),
        mix(this.uKelpBase, this.uKelpTip, gradient),
        kind,
      ).toVar();

      // A degree of per-blade tint, so a bed is not one plant repeated.
      base.mulAssign(
        mix(vec3(0.9, 1.0, 0.95), vec3(1.12, 0.97, 0.85), variation.mul(0.5).add(0.5)),
      );
      // Dark at the holdfast — see `BASE_OCCLUSION`.
      base.mulAssign(float(BASE_OCCLUSION).add(s.smoothstep(0, 0.25).mul(1 - BASE_OCCLUSION)));
      // How much daylight ever reached this plant. See `DOWNWELLING_SIGMA`.
      base.mulAssign(trait.z);

      return vec4(base.mul(light), 1);
    })();

    return material;
  }
}

// ----------------------------------------------------------------- the geometry

/**
 * One blade: 27 vertices, 32 triangles.
 *
 * Three columns lofted through nine rows, authored with `y` running 0 at the
 * holdfast to 1 at the tip and `x` spanning -0.5 to 0.5 of the blade's width —
 * so the shader's blade coordinate is `position.y` and needs no attribute of its
 * own, and the width parameter is a straight multiply.
 *
 * The width profile is narrow at the holdfast, widest a third of the way up and
 * tapering to a point: that is the silhouette of a leaf, and it is also what
 * puts the least geometry where the blade moves least. The middle column is
 * pushed out of the plane into a midrib, which is what stops the blade
 * disappearing edge-on and what gives `computeVertexNormals` two distinct faces
 * to average.
 *
 * Wound counter-clockwise seen from +z, so `faceDirection` is +1 on the face the
 * midrib bulges toward.
 */
function buildBladeGeometry(): THREE.InstancedBufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];

  for (let row = 0; row < BLADE_ROWS; row++) {
    const s = Math.pow(row / (BLADE_ROWS - 1), ROW_BIAS);
    const w = bladeWidth(s);
    positions.push(-0.5 * w, s, 0);
    positions.push(0, s, BLADE_CURL * w);
    positions.push(0.5 * w, s, 0);
  }

  for (let row = 0; row + 1 < BLADE_ROWS; row++) {
    for (let col = 0; col < 2; col++) {
      const a = row * 3 + col;
      const b = a + 1;
      const c = b + 3;
      const d = a + 3;
      indices.push(a, b, c, a, c, d);
    }
  }

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

/** Blade half-width at `s`, as a fraction of the instance's width. */
function bladeWidth(s: number): number {
  return (0.42 + 0.58 * smoothstep01(s / 0.28)) * (1 - 0.72 * s * s);
}

/**
 * Per-instance placement, packed.
 *
 * Three `vec4`s, laid out so the vertex stage's hot path — the anchor, the
 * height, the width and the blade frame — reads two of them and the fragment
 * stage's reads the third.
 */
function attachInstanceAttributes(
  geometry: THREE.InstancedBufferGeometry,
  blades: readonly Blade[],
): void {
  const anchors = new Float32Array(MAX_BLADES * 4);
  const shapes = new Float32Array(MAX_BLADES * 4);
  const traits = new Float32Array(MAX_BLADES * 4);

  for (let i = 0; i < MAX_BLADES; i++) {
    const blade = blades[i];
    const base = i * 4;

    anchors[base] = blade.x;
    anchors[base + 1] = blade.y;
    anchors[base + 2] = blade.z;
    anchors[base + 3] = blade.height;

    // The yaw is baked as its cosine and sine rather than as an angle: the
    // shader needs both, they are constant per blade, and computing them per
    // vertex would be twenty-seven transcendentals per blade per frame to
    // recover a number that never changes.
    shapes[base] = blade.width;
    shapes[base + 1] = Math.cos(blade.yaw);
    shapes[base + 2] = Math.sin(blade.yaw);
    shapes[base + 3] = blade.depth;

    traits[base] = blade.kind;
    traits[base + 1] = blade.variation;
    traits[base + 2] = Math.exp(-DOWNWELLING_SIGMA * blade.depth);
    traits[base + 3] = blade.phase;
  }

  geometry.setAttribute('kelpAnchor', new THREE.InstancedBufferAttribute(anchors, 4));
  geometry.setAttribute('kelpShape', new THREE.InstancedBufferAttribute(shapes, 4));
  geometry.setAttribute('kelpTrait', new THREE.InstancedBufferAttribute(traits, 4));
}

/**
 * A bound that actually contains the forest at full sway.
 *
 * Worth computing rather than declaring infinite, unlike the fish and the birds:
 * those are placed relative to a moving anchor and have no meaningful static
 * extent, while this is nailed to the seafloor and never moves. The margin is
 * the tallest plant plus the furthest that plant's tip can be thrown, which is
 * `BEND_MAX` of its own height.
 */
function boundBlades(blades: readonly Blade[]): THREE.Sphere {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  let tallest = 0;

  for (const blade of blades) {
    box.expandByPoint(point.set(blade.x, blade.y, blade.z));
    if (blade.height > tallest) tallest = blade.height;
  }

  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  sphere.radius += tallest * (1 + BEND_MAX);
  return sphere;
}

// ---------------------------------------------------------------- the placement

/**
 * The whole forest, in one pass, from one seeded generator.
 *
 * Patches are emitted cycling through (region, kind) so that *any prefix of the
 * result is a fair sample of all four combinations* — the same property
 * `FishSchool` and `Birds` rely on, and the reason `setCount` can be a single
 * integer write. A layout that filled one region and then the next would leave
 * the Medium tier with an island covered in kelp and a bare plateau.
 */
function placeBlades(seed: number): Blade[] {
  const random = mulberry32(seed);
  const blades: Blade[] = [];

  for (let patch = 0; blades.length < MAX_BLADES; patch++) {
    const spec = PATCH_CYCLE[patch % PATCH_CYCLE.length];
    const site = findSite(random, spec);

    // Kelp spreads; a mat does not. The radii differ by more than they look,
    // because the blade budget per patch is the same for both: eight tufts
    // inside a six-metre disc is a mat you cannot see the sand through, and the
    // same eight spread over a kelp patch would be nothing.
    const radius = spec.tall ? 9 + random() * 8 : 4 + random() * 4;
    const budget = Math.min(BLADES_PER_PATCH, MAX_BLADES - blades.length);
    const target = blades.length + budget;

    while (blades.length < target) {
      plant(blades, random, site, radius, spec, target);
    }
  }

  return blades;
}

/** A site the growth is plausible at: shallow enough, and sheltered. */
interface Site {
  x: number;
  z: number;
}

/**
 * Rejection-samples a site out of a region.
 *
 * Two tests, cheapest first. The depth band costs one heightfield evaluation and
 * rejects most candidates outright; only survivors pay the six the shelter score
 * needs. Acceptance is then *probabilistic in the score* rather than
 * thresholded, which is what turns "prefer shelter" into a density gradient
 * instead of a hard edge around every hollow.
 *
 * Always returns: the best-scoring candidate seen is kept, so a region whose
 * shape has moved out from under these bands still gets its patches, at the
 * closest thing to a good site it could find. That matters here specifically —
 * the island's geometry belongs to another module and is being changed.
 */
function findSite(random: () => number, spec: PatchSpec): Site {
  let best: Site = { x: 0, z: 0 };
  let bestScore = -1;

  for (let attempt = 0; attempt < SITE_TRIES; attempt++) {
    const angle = random() * Math.PI * 2;
    // Square-root radius keeps the scatter even in *area* rather than piling
    // every patch against the inner edge of the annulus.
    const unit = Math.sqrt(random());

    let x: number;
    let z: number;
    if (spec.region === REGION_ISLAND) {
      const r = ISLAND.radius * (ISLAND_INNER + unit * (ISLAND_OUTER - ISLAND_INNER));
      x = ISLAND.x + Math.cos(angle) * r;
      z = ISLAND.z + Math.sin(angle) * r;
    } else {
      const r = PLATEAU_INNER + unit * (PLATEAU_OUTER - PLATEAU_INNER);
      x = Math.cos(angle) * r;
      z = Math.sin(angle) * r;
    }

    const floor = seafloorHeight(x, z);
    const depth = -floor;

    if (depth < spec.minDepth || depth > spec.maxDepth) {
      // Out of band, but tracked anyway so `best` is never empty. Scored far
      // below anything in band, so an in-band candidate always wins.
      const miss = Math.min(spec.minDepth - depth, depth - spec.maxDepth);
      const score = Math.max(0, 1 - miss / 6) * 1e-3;
      if (score > bestScore) {
        bestScore = score;
        best = { x, z };
      }
      continue;
    }

    const score = shelterScore(x, z, floor);
    if (score > bestScore) {
      bestScore = score;
      best = { x, z };
    }
    if (random() < score) return { x, z };
  }

  return best;
}

/**
 * How sheltered a spot on the bottom is, 0..1.
 *
 * Two terms, both read off the heightfield rather than off the wind. The wind
 * changes at runtime and the planting does not, so a wind-relative "lee side"
 * would be wrong for most of the session; bathymetric shelter is a property of
 * the place. Slope is the exposure — an open flank is scoured and the surge runs
 * straight up it — and local concavity is the refuge, because a hollow keeps
 * both the sediment and the plants that are holding it.
 */
function shelterScore(x: number, z: number, floor: number): number {
  const gx =
    (seafloorHeight(x + SLOPE_PROBE, z) - seafloorHeight(x - SLOPE_PROBE, z)) / (2 * SLOPE_PROBE);
  const gz =
    (seafloorHeight(x, z + SLOPE_PROBE) - seafloorHeight(x, z - SLOPE_PROBE)) / (2 * SLOPE_PROBE);
  const slope = Math.hypot(gx, gz);

  const around =
    (seafloorHeight(x + HOLLOW_PROBE, z) +
      seafloorHeight(x - HOLLOW_PROBE, z) +
      seafloorHeight(x, z + HOLLOW_PROBE) +
      seafloorHeight(x, z - HOLLOW_PROBE)) *
    0.25;

  const open = 1 - Math.min(1, slope / SLOPE_LIMIT);
  const hollow = clamp01(0.45 + (around - floor) / HOLLOW_SCALE);
  // Squared, so the difference between a flank and a hollow is a factor of
  // several rather than a few percent. Floored well above zero: even an exposed
  // slope has weed on it, just not much.
  return Math.max(0.05, open * open * hollow);
}

/**
 * One plant: a holdfast and the blades rising from it.
 *
 * The blades of a plant are contiguous in the buffer, which is what makes a
 * `setCount` cut remove whole plants rather than shave blades off every plant in
 * the forest. `limit` is the patch's remaining budget — a plant is allowed to
 * finish short rather than overrun it, because a slightly sparse last plant is
 * invisible and a patch that spills into the next one is not.
 */
function plant(
  blades: Blade[],
  random: () => number,
  site: Site,
  patchRadius: number,
  spec: PatchSpec,
  limit: number,
): void {
  const angle = random() * Math.PI * 2;
  const spread = patchRadius * Math.sqrt(random());
  let x = site.x + Math.cos(angle) * spread;
  let z = site.z + Math.sin(angle) * spread;
  let depth = -seafloorHeight(x, z);

  // A patch on the island's flank is wider than the flank's own depth gradient
  // is gentle, so its edge can reach dry sand. Pulled back toward the centre
  // once rather than resampled, which is deterministic and terminates.
  if (depth < MIN_PLANT_DEPTH) {
    x = (x + site.x) * 0.5;
    z = (z + site.z) * 0.5;
    depth = -seafloorHeight(x, z);
    if (depth < MIN_PLANT_DEPTH) {
      x = site.x;
      z = site.z;
      depth = -seafloorHeight(x, z);
    }
  }

  const y = -depth - HOLDFAST_SINK;

  // Kelp reaches for the light and stops where it runs out of water; seagrass
  // has a length and keeps it. Both are capped so no tip ever breaches — a plant
  // sticking through the surface is the single most obvious failure this system
  // can have, and it would happen in the island shallows, which is the one place
  // a viewer is guaranteed to look.
  const tall = spec.tall;
  const headroom = Math.max(0.25, depth - TIP_CLEARANCE);
  const nominal = tall ? 1.9 + random() * 3.2 : 0.34 + random() * 0.62;
  const plantHeight = Math.min(nominal, headroom);

  const count = tall ? 4 + Math.floor(random() * 4) : 9 + Math.floor(random() * 6);
  const baseYaw = random() * Math.PI * 2;
  // Blades rise from one holdfast, so they share a position to within the width
  // of the holdfast itself and fan out around it.
  const cluster = tall ? 0.22 : 0.34;

  for (let b = 0; b < count && blades.length < limit; b++) {
    // Fanned rather than scattered: a kelp plant's blades come off the stipe at
    // regular intervals, and a tuft that happens to put four blades on the same
    // bearing reads as one thick blade.
    const yaw = baseYaw + (b / count) * Math.PI * 2 + (random() - 0.5) * 0.9;
    const spanned = plantHeight * (tall ? 0.72 + random() * 0.5 : 0.6 + random() * 0.7);

    blades.push({
      x: x + (random() - 0.5) * cluster,
      y,
      z: z + (random() - 0.5) * cluster,
      height: Math.max(0.2, Math.min(headroom, spanned)),
      width: tall ? 0.11 + random() * 0.11 : 0.022 + random() * 0.032,
      yaw,
      depth,
      kind: spec.kind,
      variation: random() * 2 - 1,
      phase: random() * Math.PI * 2,
    });
  }
}

// -------------------------------------------------------------------- utilities

function clampCount(count: number): number {
  if (!Number.isFinite(count)) return 0;
  return Math.max(0, Math.min(MAX_BLADES, Math.floor(count)));
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep01(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

/** Reduces a phase to [0, 2pi) in float64, before it reaches a float32 uniform. */
function wrapTau(phase: number): number {
  const tau = Math.PI * 2;
  return ((phase % tau) + tau) % tau;
}
