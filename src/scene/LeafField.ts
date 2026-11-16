import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  cameraViewMatrix,
  cos,
  dFdx,
  dFdy,
  faceDirection,
  float,
  mix,
  mrt,
  positionGeometry,
  select,
  sin,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { seafloorHeight, SEAFLOOR_EXTENT, SEAFLOOR_SEGMENTS } from './Seafloor';

/**
 * Instanced ribbon leaves on the seabed: the engine under `Kelp` and
 * `GroundCover`.
 *
 * Adapted from abyssal-garden, MIT, (c) 2026 Maksymilian Dendura — the leaf
 * vertex stage (a strip bent along a circular arc, rippled, twisted and
 * ruffled), the per-type palette and translucency, the buoyant `sway` of a
 * column as a function of height and time, the 36-byte instance record and the
 * cell-sorted buffers with distance LOD. See `THIRD_PARTY_NOTICES.md`.
 *
 * What changed in the port, and why:
 *
 * **The sway is our swell.** abyssal-garden drives its current from a clock
 * alone. Here the dominant term is the near-bed orbital motion of the same
 * swell the surface is built from — direction, wavenumber and period are the
 * ones `Kelp.setSwell` has always taken — travelling across the forest as
 * `k · x`, so a wave passing overhead visibly rolls through the columns below
 * it. The abyssal meander rides on top at a third of the amplitude so a calm
 * sea still has a forest that breathes.
 *
 * **Lighting is the scene's key, not a baked sun.** The key light's colour and
 * direction, the cloud deck's shade and the caustic field are all fed in from
 * `main`, and depth is left to `UnderwaterPass`, which already applies the
 * water column above every pixel (`uDownwell`) — attenuating here too would
 * darken the leaves twice. abyssal-garden's volume-texture skylight and
 * openness have no equivalent in this scene; a per-leaf occlusion baked at
 * placement (how crowded the column is, how low the leaf hangs) stands in.
 *
 * **Opaque, depth-writing glints.** abyssal-garden draws its sparkling specks
 * as additive sprites and fogs them itself. Here the fog is applied after the
 * fact from the depth buffer, so a sprite that wrote no depth would be fogged
 * as though it sat on whatever is behind it, i.e. vanish. A glint is therefore
 * one more instance in the leaf buffer — type `GLINT` — drawn as a tiny
 * camera-facing lens in the same draw, writing depth like every leaf.
 *
 * **Cells are meshes.** WebGPU in three r185 has no first-instance offset on an
 * instanced draw and the WebGL2 fallback has no compute, so a GPU compaction is
 * not on the table. Each 24 m cell owns its slice of the instance data as its
 * own attributes, shared by one mesh per level of detail, and the CPU attaches
 * only the mesh each visible cell needs — the scene graph holds a few dozen
 * meshes, never the thousands that exist. three's frustum test then runs per
 * cell against a bound that covers the sway.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;
const vec2n = vec2 as unknown as (x: unknown, y: unknown) => Node;

// ------------------------------------------------------------------- the types

/**
 * Leaf types. Three bits of the packed type byte; the other five carry the
 * baked occlusion. Order is the palette table's in `buildLeafMaterial`.
 */
export const LeafType = {
  /** Column blade: olive to golden green, strongly translucent, sparkles. */
  KELP: 0,
  /** Stipe segment: a narrow dark strap that reads as the stem. */
  STIPE: 1,
  /** Turtle grass (Thalassia): a flat, blunt-tipped green strap. */
  TURTLE_GRASS: 2,
  /** Manatee grass (Syringodium): hair-thin, long, pale green. */
  MANATEE_GRASS: 3,
  /** Sargassum gold: the golden fronds on columns and in drifts. */
  SARGASSUM: 4,
  /** Sea rod (a gorgonian): stiff purple-to-tan branches from one base. */
  SEA_ROD: 5,
  /** Halimeda: segmented, calcified pale-green algae in stiff upright clumps. */
  HALIMEDA: 6,
  /** A sparkling speck: camera-facing, emissive, carried by the sway. */
  GLINT: 7,
} as const;

/** One leaf before packing. */
export interface Leaf {
  x: number;
  y: number;
  z: number;
  /** Metres along the leaf. For a glint, the speck's radius. */
  len: number;
  /** Bearing of the leaf's lean, radians. */
  yaw: number;
  /** Lean from vertical, radians: 0 straight up, pi/2 flat, pi hanging. */
  tilt: number;
  /** Metres across at the widest point. */
  width: number;
  /** Height of the attachment above the plant's holdfast, for the sway. */
  hA: number;
  type: number;
  /** Two per-leaf seeds, 0..1. */
  s1: number;
  s2: number;
  /** Resting curvature, 0..1; 0.3 is straight. */
  bend: number;
  /** Baked occlusion, 0 (buried in a crowded bed) .. 1 (open water). */
  ao: number;
  /** Always drawn, whatever the density: stipes, so a thinned column keeps its stem. */
  keep?: boolean;
}

// ------------------------------------------------------------------- the cells

/**
 * Culling cell edge, metres. abyssal-garden's 24: at this scene's visibility
 * (38-70 m depending on preset) a cell is a few percent of the visible disc, so
 * the frustum test still rejects most of what is behind the viewer, while the
 * visible set at High stays around twenty to thirty cells — a draw each.
 */
export const CELL_SIZE = 24;

/** One level of detail: rows along the strip, and the drawn fraction. */
export interface LeafLod {
  rows: number;
  /**
   * [underwater-look-3] Vertices across the blade; 2 if absent. More than two
   * lets the blade cup (see `CUP`), so a near kelp blade has a rounded
   * section instead of reading as one flat polygon.
   */
  cols?: number;
  /** Multiplies the field's density for this level; widened to compensate. */
  thin: number;
  /** Upper distance bound, as a fraction of the field's range. */
  until: number;
}

export interface LeafFieldOptions {
  name: string;
  levels: readonly LeafLod[];
  /** Metres. Overridden per tier by `setRange`. */
  range: number;
  /**
   * Distance factor for the leaf-level collapse near the range, so short turf
   * goes before tall columns do. 1 for kelp.
   */
  fadeScale?: number;
  /** Extra bound margin for the sway, metres. */
  swayMargin: number;
  shading?: LeafShading;
}

/** Scene hooks the leaf lighting reads. All optional; absent ones are 1. */
export interface LeafShading {
  /** `(worldPosition) => ~1` caustic pattern. */
  caustics?: (worldPosition: Node) => Node;
  /** `(worldPosition) => 0..1` key-light shade (cloud deck). */
  keyShadow?: (worldPosition: Node) => Node;
}

interface Cell {
  /** One mesh per level, all sharing this cell's instance attributes. */
  meshes: THREE.Mesh[];
  /** Instances that are always drawn, then the shuffled thinnable rest. */
  keep: number;
  total: number;
  /** AABB, for the distance test. */
  min: THREE.Vector3;
  max: THREE.Vector3;
  /** Which level is attached, -1 for none. */
  level: number;
}

// --------------------------------------------------------------- the material

/** Above this eye height nothing on the bottom can resolve: `ReefLife`'s. */
const HIDE_ABOVE = 90;
/** Camera travel that re-runs the cell pass, metres. */
const REDEAL_DISTANCE = 1.5;
/**
 * Hysteresis on a level boundary, metres. Without it a camera parked on a band
 * edge flips a cell between levels every redeal, which is a visible pop.
 */
const LOD_HYSTERESIS = 2;

/** Standard gravity, for the dispersion relation. */
const GRAVITY = 9.81;
/** The background surge: see `Kelp.ts` history — a long, small, permanent swell. */
const SURGE_K = (Math.PI * 2) / 200;
const SURGE_OMEGA = Math.sqrt(GRAVITY * SURGE_K);

/**
 * Wraps the shader clock. Every per-leaf flutter term is `t * f` with its own
 * frequency, so the clock cannot be wrapped to any one period without a seam;
 * an hour keeps float32 resolution under a millisecond and puts the seam once
 * an hour, where a single frame of flutter phase jumping is not visible.
 */
const CLOCK_WRAP = 3600;

/**
 * Buoyant sway amplitude: abyssal-garden's `.024 h^1.25`. At 15 m of column
 * that is 0.7 m of tip travel, which is what a giant-kelp canopy does in a
 * moderate surge — and it is zero at the holdfast with zero slope, so nothing
 * slides on the sand.
 */
const SWAY_GAIN = 0.024;

/**
 * [underwater-look-3] Gain on the light a submerged blade passes through to
 * an eye looking toward the sun. The round-4 review found the kelp "flat,
 * opaque and polygonal, with no light through the blades" against
 * abyssal-garden's backlit forest. The term was there (abyssal's `trc * tr`
 * and its `pow(dot(-V, L), 3)` glow) but `UnderwaterPass` brings every
 * submerged surface down to `SURFACE_LIGHT` of its air-rig light, which took
 * the glow with it; transmitted light is what kelp is photographed for, so
 * it gets that back here.
 */
const BACKLIT = 6;
/**
 * [underwater-look-3] Round 5b. Under water, the share kept of the
 * view-independent translucency (`trc * tr`, every blade whose far face the
 * sun reaches) and of the ambient. Both lit the whole forest about evenly,
 * which is what made the kelp frame high-key next to abyssal-garden's dark
 * masses: the glow belongs to the blades the sun is behind, which the
 * view-dependent lobe (`BACKLIT`) already gives.
 */
const UNDER_DIFFUSE_TR = 0.25;
const UNDER_AMBIENT = 0.4;
/**
 * [underwater-look-3] Round 6. Under water, the share kept of the whole-leaf
 * sparkle and of the glints' emission. The round-6 review read the kelp as
 * "covered in emissive specks": both are self-lit and the pass's
 * `SURFACE_LIGHT` brings everything around them down to a third, so a speck
 * that sat quietly on a sunlit blade in air glowed like a lamp on a dark one.
 */
const UNDER_SPARKLE = 0.25;
/**
 * [underwater-look-3] Round 6. Per-channel loss per metre of water between the
 * eye and a submerged blade, over a path that saturates at
 * `NEAR_ABSORB_LENGTH`: a blade a metre off keeps 80% of its red, one at three
 * metres 60%, one at five half, and nothing past ten loses more than 63%.
 * The review read the foreground kelp as "unattenuated ochre, lit as if in
 * air": the medium's own extinction is set for range (red's 10% contrast at
 * 16 m), which leaves a blade three metres away nearly untouched. Stylised and
 * on the leaves only, which are what the eye swims through; the reef keeps
 * the medium's colour.
 */
const NEAR_ABSORB: readonly [number, number, number] = [0.22, 0.05, 0];
const NEAR_ABSORB_LENGTH = 4.5;
/**
 * [underwater-look-3] Round 6. `UNDER_DIFFUSE_TR` and `UNDER_AMBIENT` are the
 * contre-jour values: they apply where the eye looks toward the sun through
 * the blade (`dot(-V, L)` past `CONTRE_EDGES`), and these where it looks
 * across or away from it. Round 5b took the ambient and the diffuse
 * transmission down everywhere to give kelp-sun abyssal-garden's dark masses
 * against the light; seen across the light the same forest became columns of
 * the water's own navy — at a third of the air rig's light their faces were
 * darker than the water in front of them — and the review called the frame
 * monochrome. Against the sun the eye is adapted to the window and the
 * shaded sides are dark; across it they are lit by the water around them.
 */
const ACROSS_DIFFUSE_TR = 1;
/**
 * Twenty times the contre-jour value. The leaves' ambient is the scene's
 * hemisphere fill, set for air and a small term next to the key; across the
 * light under water it is most of what a blade receives, and at 2.5 the
 * forest was still navy. At 8 a column ten metres off is the dark green of
 * abyssal-garden's and the ones past thirty go to the water's blue.
 */
const ACROSS_AMBIENT = 8;
const CONTRE_EDGES: readonly [number, number] = [0.35, 0.8];

/**
 * [underwater-look-3] Round 12c. Relief on the near kelp blades, which the
 * final blind round read as flat cards in kelp-sun. Macrocystis blades are
 * bullate (blistered and wrinkled) with a raised midrib, and curl across their
 * width; all of it is a height field in metres, turned into a normal from its
 * screen derivatives (Mikkelsen's surface-gradient bump, as three's
 * `perturbNormalArb`), on kelp and Sargassum blades only, faded out between `BLADE_RELIEF_RANGE`
 * metres, past which it would only alias.
 * - `BULLATE_FREQ` along and across the half-width, `BULLATE_DEPTH` deep;
 * - the midrib `MIDRIB_HEIGHT` proud, veins `VEIN_HEIGHT`;
 * - a cross-blade curl, `BLADE_CURL` at the margin.
 * The same veins thicken the tissue the sun shines through (`VEIN_SHADOW` less
 * transmission) and the blisters vary its thickness by `BULLATE_THICKNESS`,
 * so a backlit blade shows its venation and wrinkles; in shade the hollows
 * and veins darken the blade by `BULLATE_SHADE` and `VEIN_SHADE`; the folded margin is
 * thicker and darker (`MARGIN_DARK`); and a soft sheen, `BLADE_SHEEN`, rides
 * the wrinkles.
 */
const BLADE_RELIEF_RANGE: readonly [number, number] = [5, 14];
const BULLATE_FREQ: readonly [number, number] = [34, 7];
const BULLATE_DEPTH = 0.007;
const MIDRIB_HEIGHT = 0.004;
const VEIN_HEIGHT = 0.0012;
const BLADE_CURL = 0.009;
const VEIN_SHADOW = 0.6;
const BULLATE_THICKNESS = 0.35;
const BULLATE_SHADE = 0.3;
const VEIN_SHADE = 0.22;
const MARGIN_DARK = 0.35;
const BLADE_SHEEN = 0.1;
/**
 * [underwater-look-3] Round 6. The ambient a submerged blade takes is the
 * water's light, not the sky's: blue-green rather than the sky cube's blue.
 * On the shaded side of a forest the sky's blue made every column navy, which
 * the review called monochrome; this keeps the luminance and turns the hue
 * toward green, so a near column reads green and a far one goes to the
 * water's blue.
 */
const UNDER_AMBIENT_TINT: readonly [number, number, number] = [0.75, 1.2, 0.8];
/**
 * [underwater-look-3] Round 6. A smooth field of lean across the forest, so
 * neighbouring columns stand at different angles to the current instead of
 * all leaning as one: a gain on the mean set (`uLean`) of `1 +- LEAN_SPREAD`
 * and a turn of its direction by up to `LEAN_TURN` radians, both varying over
 * five to eight metres — more than a column's spacing, much more than the
 * spread of one column's leaves, so each column still bends as one piece.
 * The swell and the surge are left alone: they are the water's motion and
 * travel through the forest as a wave.
 */
const LEAN_SPREAD = 0.6;
const LEAN_TURN = 0.7;
/**
 * [underwater-look-3] Round 10. Each plant's own lag in the swell and its own
 * meander, radians: a smooth field over seven to nine metres (two or three
 * columns' spacing; near the top, where the swing is, a column's strands are
 * within half a metre), so neighbours swing out of step and a column holds together.
 * A stiffer or longer plant answers the same water late; with one phase the
 * whole forest moved as a single sheet.
 */
const SWELL_LAG = 0.9;
const MEANDER_LAG = 2.2;
/** [underwater-look-3] How far a blade's edges curl back, as a share of its half-width. */
const CUP = 0.28;
const SWAY_POWER = 1.25;

/**
 * The shared state every leaf material reads: one set of uniforms, so the kelp
 * and the ground cover cannot disagree about the water they stand in.
 */
export class LeafUniforms {
  readonly uTime = uniform(0);
  readonly uPrevTime = uniform(0);
  readonly uSwayPhase = uniform(0);
  readonly uSurgePhase = uniform(0);
  readonly uPrevSwayPhase = uniform(0);
  readonly uPrevSurgePhase = uniform(0);
  /** Unit xz the swell travels along. */
  readonly uSwellDir = uniform(new THREE.Vector2(1, 0));
  readonly uSwellK = uniform((Math.PI * 2) / 47);
  /** Fraction of the sway envelope the swell contributes along `dir`. */
  readonly uSwell = uniform(0.9);
  readonly uSurge = uniform(0.6);
  /** Mean downstream set, same units. */
  readonly uLean = uniform(0.45);
  /** abyssal meander, same units: the forest breathing in a calm. */
  readonly uMeander = uniform(0.45);

  readonly uKeyDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  readonly uKeyColor = uniform(new THREE.Color(1, 0.96, 0.88));
  readonly uAmbient = uniform(new THREE.Color(0.1, 0.2, 0.26));
  /** Eye position the shadow pass collapses against; the main pass uses its own camera. */
  readonly uEye = uniform(new THREE.Vector3());
  /** Overall leaf brightness, so a preset can be balanced against the reef. */
  readonly uExposure = uniform(1.6);
  /** Emissive gain on the glints. */
  readonly uGlint = uniform(1);
  /** [underwater-look-3] See `BACKLIT`. */
  readonly uBacklit = uniform(BACKLIT);
  /** [underwater-look-3] See `UNDER_DIFFUSE_TR`. */
  readonly uUnderDiffuseTr = uniform(UNDER_DIFFUSE_TR);
  readonly uUnderAmbient = uniform(UNDER_AMBIENT);
  /** [underwater-look-3] See `UNDER_SPARKLE`. */
  readonly uUnderSparkle = uniform(UNDER_SPARKLE);
  /** [underwater-look-3] See `ACROSS_AMBIENT`. */
  readonly uAcrossDiffuseTr = uniform(ACROSS_DIFFUSE_TR);
  readonly uAcrossAmbient = uniform(ACROSS_AMBIENT);
  /** [underwater-look-3] See `NEAR_ABSORB`. */
  readonly uNearAbsorb = uniform(new THREE.Vector3(...NEAR_ABSORB));
  /** [underwater-look-3] See `UNDER_AMBIENT_TINT`. */
  readonly uUnderAmbientTint = uniform(new THREE.Vector3(...UNDER_AMBIENT_TINT));

  private swellOmega = Math.sqrt(GRAVITY * ((Math.PI * 2) / 47));
  private time = 0;
  private swayPhase = 0;
  private surgePhase = 0;

  setSwell(bearingRadians: number, windSpeed: number, peakWavelength: number): void {
    (this.uSwellDir.value as THREE.Vector2).set(Math.cos(bearingRadians), Math.sin(bearingRadians));
    const lambda = Math.min(400, Math.max(8, peakWavelength));
    const k = (Math.PI * 2) / lambda;
    this.uSwellK.value = k;
    this.swellOmega = Math.sqrt(GRAVITY * k);
    // Pierson-Moskowitz significant height, as `Kelp` always took it, mapped
    // onto the envelope: 10 m/s (Hs 2.5 m) is 0.9 of it, a gale saturates near
    // 1.6 rather than throwing a twenty-metre column flat.
    const u = Math.max(0, windSpeed);
    const hs = 0.0246 * u * u;
    this.uSwell.value = Math.min(1.6, 0.25 + 0.26 * hs);
    this.uSurge.value = 0.45 + 0.02 * u;
    this.uLean.value = Math.min(0.9, 0.25 + 0.025 * u);
  }

  setKey(direction: THREE.Vector3, color: THREE.Color): void {
    const d = this.uKeyDir.value as THREE.Vector3;
    d.copy(direction);
    if (d.lengthSq() < 1e-8) d.set(0, 1, 0);
    d.normalize();
    (this.uKeyColor.value as THREE.Color).copy(color);
  }

  setAmbient(color: THREE.Color): void {
    (this.uAmbient.value as THREE.Color).copy(color);
  }

  setEye(position: THREE.Vector3): void {
    (this.uEye.value as THREE.Vector3).copy(position);
  }

  update(dt: number): void {
    this.uPrevTime.value = this.uTime.value;
    this.uPrevSwayPhase.value = this.swayPhase;
    this.uPrevSurgePhase.value = this.surgePhase;
    this.time += dt;
    this.swayPhase = wrapTau(this.swayPhase + this.swellOmega * dt);
    this.surgePhase = wrapTau(this.surgePhase + SURGE_OMEGA * dt);
    this.push();
  }

  /** Exact: every pose is a closed form of these three numbers. */
  resetClock(time: number): void {
    this.time = time;
    this.swayPhase = wrapTau(time * this.swellOmega);
    this.surgePhase = wrapTau(time * SURGE_OMEGA);
    this.push();
    this.uPrevTime.value = this.uTime.value;
    this.uPrevSwayPhase.value = this.swayPhase;
    this.uPrevSurgePhase.value = this.surgePhase;
  }

  private push(): void {
    this.uTime.value = this.time % CLOCK_WRAP;
    this.uSwayPhase.value = this.swayPhase;
    this.uSurgePhase.value = this.surgePhase;
  }
}

/** `smoothstep` that tolerates reversed edges, as GLSL's and abyssal's `ss` do. */
function ss(a: number | Node, b: number | Node, x: Node): Node {
  const t = x.sub(a).div(float(b as any).sub(a)).clamp(0, 1);
  return t.mul(t).mul(t.mul(-2).add(3));
}

/**
 * The buoyant sway at a point of a plant: holdfast `(px, pz)`, height `h`
 * above it. Returns an xz displacement.
 *
 * The swell term is linear wave theory's near-bed excursion, along the swell's
 * direction, arriving at each plant as `k · x` and climbing the column with a
 * lag. The meander is abyssal-garden's two-axis term at a third of the swell's
 * weight.
 */
function swayAt(U: LeafUniforms, px: Node, pz: Node, h: Node, t: Node, swayPhase: Node, surgePhase: Node): Node {
  const dir = U.uSwellDir;
  const hh = h.max(0);
  const env = hh.pow(SWAY_POWER).mul(SWAY_GAIN);
  const downwave = px.mul(dir.x).add(pz.mul(dir.y));
  // [underwater-look-3] Round 10: see `SWELL_LAG`.
  const lag = sin(px.mul(0.79).add(pz.mul(0.37)).add(0.7))
    .mul(0.6)
    .add(sin(px.mul(-0.31).add(pz.mul(0.83)).add(2.1)).mul(0.4));
  const swell = sin(swayPhase.sub(downwave.mul(U.uSwellK)).sub(hh.mul(0.08)).sub(lag.mul(SWELL_LAG))).mul(U.uSwell);
  const surge = sin(surgePhase.sub(downwave.mul(SURGE_K)).sub(hh.mul(0.05)).sub(lag.mul(SWELL_LAG * 0.6))).mul(U.uSurge);
  const along = swell.add(surge);
  const ph = px.mul(0.031).add(pz.mul(0.017)).add(lag.mul(MEANDER_LAG));
  const mx = sin(t.mul(0.5).add(ph).add(hh.mul(0.09))).add(sin(t.mul(1.3).add(ph.mul(3)).add(hh.mul(0.3))).mul(0.35));
  const mz = cos(t.mul(0.37).add(ph.mul(1.3)).add(hh.mul(0.07))).add(cos(t.mul(1.1).add(ph.mul(2)).add(hh.mul(0.23))).mul(0.35));
  // [underwater-look-3] The mean set, varied across the forest; see `LEAN_SPREAD`.
  const leanGain = sin(px.mul(0.83).add(pz.mul(0.37)).add(1.1))
    .mul(0.6)
    .add(sin(px.mul(-0.29).add(pz.mul(1.07)).add(4.2)).mul(0.4))
    .mul(LEAN_SPREAD)
    .add(1);
  const turn = sin(px.mul(0.41).sub(pz.mul(0.73)).add(2.3))
    .mul(0.65)
    .add(sin(px.mul(1.13).add(pz.mul(0.19))).mul(0.35))
    .mul(LEAN_TURN);
  const lean = U.uLean.mul(leanGain);
  const lx = dir.x.mul(cos(turn)).sub(dir.y.mul(sin(turn))).mul(lean);
  const lz = dir.x.mul(sin(turn)).add(dir.y.mul(cos(turn))).mul(lean);
  // The swing with the swell, too, by half as much: a stiffer or shorter
  // column moves less in the same water.
  const swing = along.mul(leanGain.sub(1).mul(0.5).add(1));
  return vec2n(
    dir.x.mul(swing).add(lx).add(mx.mul(U.uMeander)),
    dir.y.mul(swing).add(lz).add(mz.mul(U.uMeander)),
  ).mul(env);
}

/** Per-type constants, as a select chain on the type. */
function byType(type: Node, values: readonly Node[]): Node {
  let out: Node = values[values.length - 1];
  for (let i = values.length - 2; i >= 0; i--) {
    out = select(type.lessThan(i + 0.5), values[i], out);
  }
  return out;
}

export interface LeafMaterialOptions {
  /** Width multiplier for thinned levels. */
  widen: Node;
  /** Range of the field, metres. */
  range: Node;
  fadeScale: number;
  shading?: LeafShading;
}

/**
 * The leaf material: the vertex stage of abyssal-garden's `W_LEAF.vs` in TSL,
 * and a fragment stage that keeps its albedo, translucency, sheen, iridescence
 * and sparkle terms but takes the scene's light.
 *
 * `MeshBasicNodeMaterial`, lit in the colour node, for the reasons the old kelp
 * was: two-sided translucent foliage has no counterpart in the standard model,
 * and the custom model is a fraction of its cost per fragment.
 */
export function buildLeafMaterial(U: LeafUniforms, options: LeafMaterialOptions): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial();
  material.name = 'seabed-leaf';
  material.side = THREE.DoubleSide;
  // Opaque and depth-writing: `UnderwaterPass` fogs by the depth buffer.
  material.transparent = false;
  material.depthWrite = true;
  material.fog = false;

  const A: Node = attribute('leafA', 'vec4');
  const B: Node = attribute('leafB', 'vec4');
  const C: Node = attribute('leafC', 'vec4');
  const u: Node = positionGeometry.x;
  const v: Node = positionGeometry.y;

  const code = C.x.mul(255).add(0.5).floor();
  const type = code.mod(8);
  const ao = code.div(8).floor().div(31);
  const s1 = C.y;
  const s2 = C.z;
  const bendRest = C.w;
  const anchor = A.xyz;
  const len = A.w;
  const yaw = B.x;
  const tilt = B.y;
  const width = B.z.mul(options.widen);
  const hA = B.w;
  const t = U.uTime;
  const isGlint = type.greaterThan(6.5);
  const isStipe = type.sub(1).abs().lessThan(0.5);

  // --- the per-type table (abyssal's `switch ty`) ---------------------------
  // Shape exponents (base, tip), wave, flutter.
  const shp = byType(type, [
    vec2(0.5, 0.32), // kelp
    vec2(0.05, 0.1), // stipe: nearly full width end to end
    vec2(0.25, 0.2), // turtle grass: a strap with a rounded tip
    vec2(0.08, 0.2), // manatee grass: a thread, near-constant width
    vec2(0.5, 0.4), // sargassum
    vec2(0.04, 0.12), // sea rod: a rod, blunt at the end
    vec2(0.15, 0.25), // halimeda: a chain of segments, blunt
    vec2(1, 1), // glint
  ]);
  const wav = byType(type, [float(1.9), float(0.3), float(0.6), float(1), float(1.8), float(0.15), float(0.2), float(0)]);
  const fls = byType(type, [float(1), float(0.3), float(0.6), float(0.8), float(1), float(0.2), float(0.25), float(0)]);

  // --- the strip, bent (abyssal `vs`) ----------------------------------------
  const cy = cos(yaw);
  const sy = sin(yaw);
  const ct = cos(tilt);
  const st = sin(tilt);
  const d = vec3n(st.mul(cy), ct, st.mul(sy));
  const pe = vec3n(ct.mul(cy), st.negate(), ct.mul(sy));
  const sd = vec3n(sy.negate(), 0, cy);

  const k = bendRest
    .sub(0.3)
    .mul(2)
    .add(sin(t.mul(s1.add(0.8)).mul(fls).add(s2.mul(40)).add(hA.mul(0.25))).mul(0.1).mul(fls));
  const waveRate = s2.mul(5).add(3);
  const wph = v.mul(waveRate).add(s1.mul(30)).add(t.mul(1.1));
  const wv = sin(wph).mul(0.12).mul(wav);
  const kq = k.mul(1.7);
  const ka = select(kq.abs().lessThan(0.06), float(0.06), kq);
  const sK = sin(ka.mul(v));
  const cK = cos(ka.mul(v));
  const centre = anchor.add(
    d.mul(sK.div(ka)).add(pe.mul(float(1).sub(cK).div(ka).add(wv.mul(v)))).mul(len),
  );
  const sl = sK.add(wv).add(cos(wph).mul(0.1).mul(wav).mul(v).mul(waveRate));
  const N = pe.mul(cK).sub(d.mul(sl)).normalize();
  const T0 = d.mul(cK).add(pe.mul(sl)).normalize();
  const tw = s1.sub(0.5).mul(2.4).mul(v).add(sin(t.mul(0.9).add(s2.mul(20))).mul(0.25).mul(v));
  const S = sd.mul(cos(tw)).add(N.mul(sin(tw)));
  const N2 = N.mul(cos(tw)).sub(sd.mul(sin(tw)));
  const profile = sin(v.pow(shp.x).mul(Math.PI)).max(0).pow(shp.y);
  // Halimeda is a chain of flat discs: pinch the width at the joints.
  const isHalimeda = type.sub(6).abs().lessThan(0.5);
  const segments = select(isHalimeda, sin(v.mul(Math.PI * 4.5)).abs().sqrt().mul(0.6).add(0.4), float(1));
  const halfWidth = width.mul(0.5).mul(profile).mul(segments);
  const cupAmount = byType(type, [float(CUP), float(0), float(0.12), float(0), float(CUP), float(0), float(0), float(0)]);

  /**
   * Leaf-level distance collapse, abyssal's `farK`: fades the width to zero
   * over the last fifth of the range so a cell leaving the set takes nothing
   * visible with it, and collapses a leaf the eye is inside of rather than
   * letting one blade fill the screen of a camera swimming through a bed.
   */
  const leafPosition = (eye: Node, shadow: boolean): Node => {
    const cd = anchor.distance(eye);
    const far = options.range.mul(options.fadeScale);
    let farK: Node = float(1).sub(ss(far.mul(0.8), far, cd));
    if (!shadow) {
      farK = farK
        .mul(ss(0.6, 2.2, centre.distance(eye)))
        .mul(ss(len.mul(1.1).add(0.5), len.mul(2.2).add(1.2), anchor.add(d.mul(len.mul(0.5))).distance(eye)));
    }
    const nearK = float(1).sub(ss(26, 40, cd));
    const ruffle = sin(v.mul(10).add(s2.mul(30)).add(t.mul(0.6)))
      .mul(width.mul(0.36))
      .mul(u)
      .mul(sin(v.mul(Math.PI)))
      .mul(nearK);
    // In the shadow pass a stipe casts nothing (it is a thin dark strap the
    // shafts do not need) and a blade casts at least 9 cm, abyssal's floor,
    // so a thinned far cell still darkens the water below it.
    const half = shadow
      ? select(isStipe, float(0), width.mul(0.5).max(0.09).mul(profile))
      : halfWidth;
    // [underwater-look-3] A cupped section; only visible with more than two
    // vertices across (`LeafLod.cols`), a constant offset otherwise. Edges
    // curl toward -N2, which is the side the shading normal already rounds.
    const cup = shadow ? float(0) : u.mul(u).mul(half).mul(cupAmount).negate();
    const p = centre.add(S.mul(u.mul(farK).mul(half))).add(N2.mul(shadow ? 0 : ruffle).add(N2.mul(cup)));
    const h = hA.add(p.y.sub(anchor.y));
    const sw = swayAt(U, anchor.x, anchor.z, h, t, U.uSwayPhase, U.uSurgePhase);
    return p.add(vec3n(sw.x, 0, sw.y));
  };

  // --- the glint ------------------------------------------------------------
  // A lens facing the eye, carried by its column's sway plus a slow drift of
  // its own, sized at least a pixel and a half wherever it is (abyssal's
  // `max(b.z, d * .0026)`), and gone past 40 m where it would only alias.
  const glintPosition = (eye: Node, shadow: boolean): Node => {
    const sw = swayAt(U, anchor.x, anchor.z, hA, t, U.uSwayPhase, U.uSurgePhase);
    const drift = vec3n(
      sin(t.mul(0.7).add(s1.mul(50))),
      sin(t.mul(0.9).add(s1.mul(70))),
      cos(t.mul(0.8).add(s1.mul(90))),
    ).mul(0.1);
    const centreG = anchor.add(vec3n(sw.x, 0, sw.y)).add(drift);
    if (shadow) return centreG;
    const toEye = eye.sub(centreG);
    const dist = toEye.length();
    const fwd = toEye.div(dist.max(1e-3));
    const right = vec3(0, 1, 0).cross(fwd).normalize();
    const up = fwd.cross(right);
    const twinkle = sin(t.mul(s2.mul(0.9).add(0.35)).add(s1.mul(100))).mul(0.5).add(0.5).pow(4);
    const size = len.max(dist.mul(0.0026)).min(dist.mul(0.0045));
    const alive = float(1).sub(ss(30, 40, dist)).mul(twinkle.mul(0.7).add(0.3));
    const lens = sin(v.mul(Math.PI));
    return centreG.add(right.mul(u.mul(lens)).add(up.mul(v.mul(2).sub(1))).mul(size.mul(alive)));
  };

  const mainPosition = select(isGlint, glintPosition(cameraPosition, false), leafPosition(cameraPosition, false));
  material.positionNode = mainPosition;
  // The shadow pass sees the shadow camera as `cameraPosition`, which would
  // collapse every leaf as out of range; it takes the eye from a uniform, and
  // glints collapse to a point.
  material.castShadowPositionNode = select(isGlint, glintPosition(U.uEye, true), leafPosition(U.uEye, true));

  const vWorld: Node = varying(mainPosition, 'leafWorld');
  const vNormal: Node = varying(
    N2.add(S.mul(u).mul(0.55))
      .sub(T0.mul(cos(v.mul(10).add(s2.mul(30)).add(t.mul(0.6)))).mul(u).mul(0.7).mul(float(1).sub(ss(26, 40, anchor.distance(cameraPosition)))))
      .normalize(),
    'leafNormal',
  );
  const vUv: Node = varying(vec2n(u, v), 'leafUv');
  const vSeed: Node = varying(vec4(s1, s2, ao, type), 'leafSeed');
  material.userData.leafPosition = mainPosition;
  material.userData.leafWorld = vWorld;
  material.userData.leafNormal = vNormal;
  material.userData.leafTranslucency = byType(type, [float(0.9), float(0.2), float(0.7), float(0.6), float(1), float(0.35), float(0.25), float(0)]);

  // --- palette (vertex stage, abyssal's per-type colours) --------------------
  // Height-varying: the same blade is darker and browner low in a column and
  // brighter and greener where it reaches the light, keyed on its own depth so
  // a shallow bed reads sunnier than a deep one.
  const lift = ss(-20, -3, anchor.y);
  // [underwater-look-3] Three tones rather than two, picked mostly per
  // column (`Kelp` folds a column tone into `s1`): golden-brown, olive, and a
  // greener blade, so neighbouring columns separate instead of the whole
  // forest being one olive.
  const tri = (a: Node, b: Node, c: Node): Node =>
    mix(mix(a, b, ss(0.0, 0.5, s1)), c, ss(0.5, 1.0, s1));
  const kelpCol = mix(
    tri(vec3(0.11, 0.065, 0.018), vec3(0.07, 0.085, 0.02), vec3(0.03, 0.08, 0.035)),
    tri(vec3(0.17, 0.125, 0.04), vec3(0.12, 0.145, 0.035), vec3(0.05, 0.15, 0.06)),
    lift,
  );
  const col = byType(type, [
    mix(kelpCol, vec3(0.25, 0.17, 0.025), s2.greaterThan(0.86).select(float(1), float(0))),
    vec3(0.08, 0.085, 0.03),
    // Turtle grass: bright Caribbean green, a fifth of the blades browned by
    // the epiphytes that crust old leaves.
    mix(
      mix(vec3(0.05, 0.17, 0.04), vec3(0.13, 0.3, 0.07), s1),
      vec3(0.16, 0.14, 0.05),
      s2.greaterThan(0.8).select(float(1), float(0)),
    ),
    mix(vec3(0.06, 0.2, 0.08), vec3(0.1, 0.27, 0.1), s1),
    mix(vec3(0.3, 0.17, 0.03), vec3(0.42, 0.27, 0.05), s1),
    // Sea rods run purple to tan (Plexaura, Eunicea); a quarter are tan.
    mix(
      mix(vec3(0.45, 0.06, 0.35), vec3(0.3, 0.08, 0.45), s1),
      vec3(0.5, 0.26, 0.14),
      s2.greaterThan(0.75).select(float(1), float(0)),
    ),
    // Halimeda: the chalky pale green of calcified segments.
    mix(vec3(0.16, 0.3, 0.1), vec3(0.3, 0.42, 0.2), s1),
    vec3(1, 0.62, 0.08),
  ]);
  // x sparkle probability, y translucency, z iridescence, w emissive.
  const pr = byType(type, [
    vec4(0.24, 0.9, 0, 0),
    vec4(0, 0.2, 0, 0),
    vec4(0.03, 0.7, 0, 0),
    vec4(0.02, 0.6, 0, 0),
    vec4(0.2, 1, 0.1, 0),
    vec4(0.06, 0.35, 0, 0.12),
    vec4(0.02, 0.25, 0, 0),
    vec4(0, 0, 0, 0),
  ]);
  const hsel = sin(s1.mul(13.7).add(s2.mul(71.3))).mul(43758.5).fract();
  const pal = (h: Node): Node =>
    select(
      h.lessThan(0.3),
      vec3(1, 0.62, 0.08),
      select(h.lessThan(0.6), vec3(0.12, 1, 0.22), select(h.lessThan(0.8), vec3(0.2, 0.85, 1), vec3(1, 0.3, 0.8))),
    );
  const spc = byType(type, [
    pal(hsel.mul(0.72)),
    vec3(0),
    vec3(0.6, 1, 0.5),
    vec3(0.6, 1, 0.5),
    vec3(1, 0.85, 0.5),
    vec3(1, 0.35, 0.8),
    vec3(0.7, 1, 0.6),
    pal(s1),
  ]);
  // Whole-leaf sparkle is a coin flip per leaf, abyssal's `step(hs(..), pr.x)`.
  const sparkleOn = sin(s1.mul(91.7).add(s2.mul(37.1)).add(yaw)).mul(43758.5).fract().lessThan(pr.x).select(float(1), float(0));
  const vCol: Node = varying(col, 'leafCol');
  const vPr: Node = varying(vec4(sparkleOn, pr.y, pr.z, pr.w), 'leafPr');
  const vSpc: Node = varying(spc, 'leafSpc');

  // --- the fragment ---------------------------------------------------------
  material.colorNode = Fn(() => {
    const wp = vWorld;
    const uvx = vUv.x.abs();
    const vv = vUv.y;
    const seed1 = vSeed.x;
    const seed2 = vSeed.y;
    const occl = vSeed.z;
    const glint = vSeed.w.greaterThan(6.5);
    const toEye = cameraPosition.sub(wp);
    const dist = toEye.length();
    const V = toEye.div(dist.max(1e-4));
    const n = vNormal.normalize().mul(faceDirection).toVar();
    // [underwater-look-3] Round 12c: a near blade's relief; see `BLADE_RELIEF_RANGE`.
    // Kelp and the golden Sargassum fronds on the columns (the near blades).
    const isKelp = vSeed.w.lessThan(0.5).or(vSeed.w.sub(4).abs().lessThan(0.5));
    const relief = select(isKelp, ss(BLADE_RELIEF_RANGE[1], BLADE_RELIEF_RANGE[0], dist), float(0)).toVar();
    const vein = ss(0.82, 1, sin(vv.mul(40).sub(uvx.mul(7)).add(seed1.mul(40)))).toVar();
    const midrib = ss(0.1, 0, uvx);
    const bullate: any = sin(vv.mul(BULLATE_FREQ[0]).add(seed2.mul(31))).mul(
      sin(uvx.mul(BULLATE_FREQ[1]).add(vv.mul(17)).add(seed1.mul(13))),
    );
    const height = bullate
      .mul(BULLATE_DEPTH)
      .add(midrib.mul(MIDRIB_HEIGHT))
      .add(vein.mul(VEIN_HEIGHT))
      .add(uvx.mul(uvx).mul(BLADE_CURL))
      .mul(relief)
      .toVar();
    const dpdx: any = dFdx(wp);
    const dpdy: any = dFdy(wp);
    const g1 = dpdy.cross(n);
    const g2 = n.cross(dpdx);
    const det = dpdx.dot(g1);
    const grad = g1.mul(dFdx(height)).add(g2.mul(dFdy(height))).mul(det.sign());
    n.assign(n.mul(det.abs().max(1e-12)).sub(grad).normalize());
    const margin = ss(0.86, 1, uvx).mul(select(isKelp, float(1), float(0))).toVar();
    // [underwater-look-3] Below the surface the sun stands where Snell puts
    // it, not where it is in the sky: 47 degrees up rather than 24 under
    // the skyPro sun, so the through-blade glow peaks where the burst is.
    const K = U.uKeyDir;
    const kh = vec2n(K.x, K.z).div(1.333);
    const Lw = vec3n(kh.x, float(1).sub(kh.dot(kh)).max(1e-4).sqrt(), kh.y);
    const underK = float(1).sub(ss(-1.2, 1.2, wp.y)).toVar();
    const L = mix(K, Lw, underK).normalize().toVar();

    // Albedo: midrib, a darker base, veins near the eye, a mottle in place of
    // abyssal's detail texture.
    const mottle = sin(vv.mul(seed2.mul(7).add(9)).add(seed1.mul(20)))
      .mul(sin(uvx.mul(2.1).add(vv.mul(23)).add(seed2.mul(40))))
      .mul(0.5)
      .add(0.5);
    const alb = vCol
      .mul(ss(0, 0.25, uvx).mul(0.28).add(0.72))
      .mul(mix(float(0.5), float(1.12), vv))
      .mul(float(1).sub(ss(30, 8, dist).mul(0.45).mul(ss(0.6, 1, sin(vv.mul(40).sub(uvx.mul(7)).add(seed1.mul(40)))))))
      .mul(ss(0, 0.13, uvx).mul(0.42).add(0.58))
      .mul(mottle.mul(0.45).add(0.75))
      .toVar();
    // Halimeda's joints are darker than its chalky segments.
    const joints = ss(0.0, 0.35, sin(vv.mul(Math.PI * 4.5)).abs());
    alb.mulAssign(select(vSeed.w.sub(6).abs().lessThan(0.5), joints.mul(0.45).add(0.55), float(1)));

    const ndl = n.dot(L).toVar();
    const ndv = n.dot(V).clamp(0, 1).toVar();
    const ir = cos(
      vec3(0, 0.33, 0.67).add(ndv.mul(1.1)).add(seed1.mul(0.22)).add(vv.mul(0.5)).add(0.55).mul(Math.PI * 2),
    )
      .mul(0.5)
      .add(0.5);
    alb.assign(mix(alb, alb.mul(0.5).add(ir.mul(vec3(0.15, 0.13, 0.3))), vPr.z.mul(ndv.oneMinus()).mul(0.8)));
    alb.mulAssign(float(1).sub(margin.mul(MARGIN_DARK)));
    // Round 12c: the wrinkles' hollows and the veins in the blade's own colour,
    // so a blade in the canopy's shade, lit by ambient alone, keeps its relief.
    alb.mulAssign(
      float(1).sub(
        relief.mul(bullate.mul(0.5).add(0.5).mul(BULLATE_SHADE).add(vein.mul(VEIN_SHADE)).add(midrib.mul(VEIN_SHADE))),
      ),
    );

    // Caustics ride the key light, and fade with depth as they do on the sand.
    let caustic: Node = float(1);
    if (options.shading?.caustics) {
      const depth = wp.y.negate();
      const reach = float(1).sub(ss(2, 48, depth)).mul(float(1).sub(ss(-1.2, 1.2, wp.y)));
      caustic = mix(float(1), options.shading.caustics(wp), reach);
    }
    const shade: Node = options.shading?.keyShadow ? options.shading.keyShadow(wp) : float(1);
    const E = U.uKeyColor.mul(shade).toVar();

    const translucency = vPr.y;
    const tr = ndl.negate().max(0).mul(0.85).add(0.15).mul(translucency);
    // [underwater-look-3] A thickness falloff: light crosses less tissue at
    // the thin margin than at the midrib, so the edges glow and the rib
    // reads as a darker line through the backlit blade.
    const thin = mix(float(0.55), float(1.3), ss(0.05, 0.85, uvx))
      .mul(float(1).sub(vein.mul(relief).mul(VEIN_SHADOW)))
      .mul(float(1).add(bullate.mul(relief).mul(BULLATE_THICKNESS)))
      .mul(float(1).sub(margin.mul(0.5)));
    // Under water a little less gold in the transmitted tint (round 5b).
    const trTint = mix(vec3(1.5, 1.4, 0.75), vec3(1.25, 1.4, 0.85), underK);
    const trc = alb.mul(mix(vec3(1), trTint, translucency.mul(vPr.z.oneMinus()))).mul(thin);
    // Occlusion: the baked crowding, and the base of every leaf is darker.
    const aoTerm = mix(float(0.3), float(1), occl).mul(mix(float(0.55), float(1), vv));
    const am = U.uAmbient
      .mul(n.y.mul(0.45).add(0.55))
      .add(U.uAmbient.mul(n.y.mul(-0.45).add(0.55)).mul(translucency.mul(0.55)).mul(vec3(1.1, 1.15, 0.7)))
      .mul(aoTerm);
    // [underwater-look-3] See `ACROSS_AMBIENT`.
    const contre = ss(CONTRE_EDGES[0], CONTRE_EDGES[1], V.negate().dot(L)).toVar();
    const underDiffuseTr = mix(U.uAcrossDiffuseTr, U.uUnderDiffuseTr, contre);
    const underAmbient = mix(U.uAcrossAmbient, U.uUnderAmbient, contre);
    const c = E.mul(caustic)
      .mul(alb.mul(ndl.max(0)).add(trc.mul(tr).mul(mix(float(1), underDiffuseTr, underK))))
      .mul(aoTerm.mul(0.6).add(0.4))
      .add(alb.mul(am).mul(mix(vec3n(1, 1, 1), U.uUnderAmbientTint.mul(underAmbient), underK)))
      .toVar();
    const H = V.add(L).normalize();
    const nh = n.dot(H).clamp(0, 1);
    c.addAssign(E.mul(nh.pow(40).mul(ir.mul(vPr.z).mul(2).add(0.45)).add(nh.pow(9).mul(0.07))).mul(0.25));
    // Round 12c: the blade's soft sheen; see `BLADE_SHEEN`.
    c.addAssign(E.mul(nh.pow(8).mul(relief).mul(BLADE_SHEEN)).mul(aoTerm));
    // Looking toward the key through a blade: the glow kelp is photographed for.
    // [underwater-look-3] Under water, a narrower lobe scaled back up (see
    // `BACKLIT`): only the blades between the eye and the sun glow, and the
    // rest of the forest stays the dark mass abyssal-garden's frames show.
    const back = V.negate().dot(L).clamp(0, 1).toVar();
    const lobeAir = back.pow(2).mul(2);
    // Round 5b: mu^12, from mu^6. At mu^6 every blade within forty degrees
    // of the sun glowed, which lit the near forest about evenly.
    const back2 = back.mul(back);
    const back6 = back2.mul(back2).mul(back2);
    const lobeWater = back6.mul(back6).mul(U.uBacklit);
    c.addAssign(E.mul(trc).mul(translucency).mul(mix(lobeAir, lobeWater, underK)));
    const lum = (x: Node): Node => x.dot(vec3(0.1, 0.5, 0.5));
    c.addAssign(alb.mul(vPr.w).mul(lum(am.add(E.mul(0.3))).mul(3).add(0.5)));
    // Whole-leaf sparkle that twinkles slowly.
    const tw = sin(U.uTime.mul(seed2.mul(1.4).add(0.5)).add(seed1.mul(90))).mul(0.5).add(0.5).pow(5);
    const band = ss(0.2, 0, vv.sub(seed2.mul(0.45).add(0.35)).abs()).mul(uvx.mul(uvx).oneMinus());
    const sparkleKeep = mix(float(1), U.uUnderSparkle, underK);
    c.addAssign(vSpc.mul(vPr.x).mul(tw.add(0.1)).mul(band).mul(lum(E).mul(1.2).add(0.35)).mul(0.6).mul(sparkleKeep));

    // The glint: emissive, falling off from its centre.
    const r2 = vUv.x.mul(vUv.x).add(vv.mul(2).sub(1).pow(2));
    const twG = sin(U.uTime.mul(seed2.mul(0.9).add(0.35)).add(seed1.mul(100))).mul(0.5).add(0.5).pow(4);
    const glow = vSpc
      .mul(twG.mul(1.7).add(0.12))
      .mul(r2.mul(-3).exp())
      .mul(lum(E).mul(1.8).add(0.3))
      .mul(U.uGlint)
      .mul(sparkleKeep)
      .mul(4);

    // [underwater-look-3] Round 6: the water between the eye and a near blade;
    // see `NEAR_ABSORB`.
    const nearPath = dist.div(NEAR_ABSORB_LENGTH).negate().exp().oneMinus().mul(NEAR_ABSORB_LENGTH);
    c.mulAssign(mix(vec3n(1, 1, 1), U.uNearAbsorb.mul(nearPath).negate().exp(), underK));
    return vec4(select(glint, glow, c.mul(U.uExposure)), 1);
  })();

  return material;
}

/**
 * Motion vectors and view normals for the TAA scene pass.
 *
 * Velocity reprojects the leaf through last frame's sway (`uPrev*`) and last
 * frame's view-projection: the flutter terms are left out, the column's sway —
 * which is most of the motion — is not, so a forest swaying in a still frame
 * does not smear. The normal is the leaf's shaded one, for GTAO.
 *
 * Set only when the scene pass has MRT: a material `mrtNode` *replaces* the
 * output of a render into a target without one (see `Clouds`).
 */
export function attachLeafMrt(
  material: THREE.MeshBasicNodeMaterial,
  U: LeafUniforms,
  previous: Node,
  current: Node,
): void {
  const A: Node = attribute('leafA', 'vec4');
  const B: Node = attribute('leafB', 'vec4');
  const world: Node = material.userData.leafPosition;
  const wp: Node = material.userData.leafWorld;
  const h = B.w.add(world.y.sub(A.y));
  const now = swayAt(U, A.x, A.z, h, U.uTime, U.uSwayPhase, U.uSurgePhase);
  const before = swayAt(U, A.x, A.z, h, U.uPrevTime, U.uPrevSwayPhase, U.uPrevSurgePhase);
  const prevWorld: Node = varying(world.sub(vec3n(now.x.sub(before.x), 0, now.y.sub(before.y))), 'leafPrevWorld');
  const normal: Node = material.userData.leafNormal;
  material.mrtNode = mrt({
    velocity: Fn(() => {
      const c = current.mul(vec4(wp, 1));
      const p = previous.mul(vec4(prevWorld, 1));
      return c.xy.div(c.w.max(1e-4)).sub(p.xy.div(p.w.max(1e-4)));
    })(),
    // [underwater-look-3] Bent toward the sky by the leaf's translucency. A
    // thin blade is lit through from either face, but `UnderwaterPass` reads
    // this normal for its hemisphere (dark undersides): seen from below, a
    // kelp blade's face toward the eye points down and lost two thirds of its
    // light. Varying, so the vertex stage computes it once.
    normal: cameraViewMatrix
      .mul(
        vec4(
          normal
            .normalize()
            .mul(faceDirection)
            .add(vec3(0, 1, 0).mul((varying(material.userData.leafTranslucency, 'leafTr') as Node).mul(0.7)))
            .normalize(),
          0,
        ),
      )
      .xyz.normalize(),
  });
  material.needsUpdate = true;
}

// ------------------------------------------------------------------- the field

/**
 * A cell-sorted, LOD'd set of leaves. Built once; per frame it moves which
 * cell meshes are attached and nothing else.
 */
export class LeafField {
  readonly object: THREE.Object3D;
  readonly leafCount: number;

  private readonly cells: Cell[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly materials: THREE.MeshBasicNodeMaterial[];
  private readonly levels: readonly LeafLod[];
  private readonly uWiden = uniform(1);
  private readonly uWidenThin: Node[];
  private readonly uRange: Node;
  private density = 1;
  private range: number;
  private castShadow = false;
  private wantVisible = true;
  private eyeAllows = true;
  private readonly lastEye = new THREE.Vector3(Infinity, Infinity, Infinity);
  private dirty = true;

  constructor(
    leaves: readonly Leaf[],
    private readonly uniforms: LeafUniforms,
    options: LeafFieldOptions,
  ) {
    this.levels = options.levels;
    this.range = options.range;
    this.uRange = uniform(options.range);
    this.leafCount = leaves.length;

    this.object = new THREE.Object3D();
    this.object.name = options.name;
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();

    // One material per distinct thinning factor: the widening is the only
    // thing that differs, and it is a uniform.
    this.uWidenThin = this.levels.map(() => uniform(1));
    this.materials = this.levels.map((_, i) =>
      buildLeafMaterial(uniforms, {
        widen: this.uWidenThin[i],
        range: this.uRange,
        fadeScale: options.fadeScale ?? 1,
        shading: options.shading,
      }),
    );
    void this.uWiden;

    const templates = this.levels.map((l) => buildStrip(l.rows, l.cols ?? 2));
    this.buildCells(leaves, templates, options.swayMargin, options.name);
    this.applyDensity();

    // The boot prewarm compiles what is attached and visible, and a cell is
    // attached only once the eye is near it — so every level of the first cell
    // is attached, unculled, until the first deal. Without it the first forest
    // the viewer swims to compiles its pipelines inline.
    if (this.cells.length > 0) {
      this.warm = this.cells[0].meshes;
      for (const mesh of this.warm) {
        mesh.frustumCulled = false;
        this.object.add(mesh);
      }
    }
  }

  private warm: THREE.Mesh[] | null = null;

  /** Fraction of the thinnable leaves drawn, 0..1. A uniform write and a count. */
  setDensity(density: number): void {
    const next = Math.max(0, Math.min(1, Number.isFinite(density) ? density : 0));
    if (next === this.density) return;
    this.density = next;
    this.applyDensity();
  }

  getDensity(): number {
    return this.density;
  }

  setRange(range: number): void {
    this.range = Math.max(10, range);
    this.uRange.value = this.range;
    this.dirty = true;
  }

  setCastShadow(cast: boolean): void {
    this.castShadow = cast;
    for (const cell of this.cells) for (const mesh of cell.meshes) mesh.castShadow = cast;
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** Attaches the level each cell needs for this eye. Cheap when the eye is still. */
  setEye(eye: THREE.Vector3): void {
    const allows = eye.y < HIDE_ABOVE;
    if (allows !== this.eyeAllows) {
      this.eyeAllows = allows;
      this.applyVisibility();
    }
    if (!allows) return;
    if (!this.dirty && eye.distanceToSquared(this.lastEye) < REDEAL_DISTANCE * REDEAL_DISTANCE) return;
    this.dirty = false;
    this.lastEye.copy(eye);
    if (this.warm) {
      for (const mesh of this.warm) {
        mesh.frustumCulled = true;
        this.object.remove(mesh);
      }
      this.warm = null;
    }
    const layers = this.object.layers.mask;
    for (const cell of this.cells) {
      const dx = Math.max(cell.min.x - eye.x, 0, eye.x - cell.max.x);
      const dy = Math.max(cell.min.y - eye.y, 0, eye.y - cell.max.y);
      const dz = Math.max(cell.min.z - eye.z, 0, eye.z - cell.max.z);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      let level = -1;
      for (let i = 0; i < this.levels.length; i++) {
        // Stay on the current level until the boundary is cleared by the margin.
        const edge = this.levels[i].until * this.range + (cell.level === i ? LOD_HYSTERESIS : -LOD_HYSTERESIS * (cell.level > i ? 1 : 0));
        if (dist < edge) {
          level = i;
          break;
        }
      }
      if (level === cell.level) continue;
      if (cell.level >= 0) this.object.remove(cell.meshes[cell.level]);
      cell.level = level;
      if (level >= 0) {
        const mesh = cell.meshes[level];
        mesh.layers.mask = layers;
        this.object.add(mesh);
      }
    }
  }

  /** Meshes currently attached, for diagnostics. */
  attachedCount(): number {
    return this.object.children.length;
  }

  setVelocity(previous: Node, current: Node): void {
    for (const material of this.materials) attachLeafMrt(material, this.uniforms, previous, current);
  }

  /** Every mesh, attached or not — for a compile pass that must see them all. */
  allMeshes(): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    for (const cell of this.cells) out.push(...cell.meshes);
    return out;
  }

  dispose(): void {
    this.object.removeFromParent();
    this.object.clear();
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
  }

  // ------------------------------------------------------------------ internals

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.eyeAllows && this.density > 0;
  }

  /**
   * Instance counts per level from the density, and the widening that keeps a
   * thinned bed's coverage: `1/sqrt(f)` — abyssal widens half density by 1.5
   * and a quarter by 2, which this tracks within a few percent.
   */
  private applyDensity(): void {
    for (let i = 0; i < this.levels.length; i++) {
      const f = Math.max(1e-3, this.density * this.levels[i].thin);
      this.uWidenThin[i].value = Math.min(2.2, 1 / Math.sqrt(f));
    }
    for (const cell of this.cells) {
      const rest = cell.total - cell.keep;
      for (let i = 0; i < this.levels.length; i++) {
        const geometry = cell.meshes[i].geometry as THREE.InstancedBufferGeometry;
        geometry.instanceCount = cell.keep + Math.round(rest * this.density * this.levels[i].thin);
      }
    }
    this.applyVisibility();
  }

  private buildCells(
    leaves: readonly Leaf[],
    templates: THREE.BufferGeometry[],
    swayMargin: number,
    name: string,
  ): void {
    // Bucket by cell.
    const buckets = new Map<number, Leaf[]>();
    const keyOf = (x: number, z: number) =>
      (Math.floor(x / CELL_SIZE) + 4096) * 8192 + (Math.floor(z / CELL_SIZE) + 4096);
    for (const leaf of leaves) {
      const key = keyOf(leaf.x, leaf.z);
      let list = buckets.get(key);
      if (!list) buckets.set(key, (list = []));
      list.push(leaf);
    }

    const keys = [...buckets.keys()].sort((a, b) => a - b);
    for (const key of keys) {
      const list = buckets.get(key)!;
      // Kept leaves first; the rest shuffled, deterministically, so any prefix
      // of them is a fair sample of every plant in the cell.
      const kept = list.filter((l) => l.keep);
      const rest = list.filter((l) => !l.keep);
      shuffle(rest, key);
      const ordered = kept.concat(rest);

      const n = ordered.length;
      const fa = new Float32Array(n * 4);
      const fb = new Float32Array(n * 4);
      const uc = new Uint8Array(n * 4);
      const min = new THREE.Vector3(Infinity, Infinity, Infinity);
      const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
      for (let i = 0; i < n; i++) {
        const l = ordered[i];
        fa[i * 4] = l.x;
        fa[i * 4 + 1] = l.y;
        fa[i * 4 + 2] = l.z;
        fa[i * 4 + 3] = l.len;
        fb[i * 4] = l.yaw;
        fb[i * 4 + 1] = l.tilt;
        fb[i * 4 + 2] = l.width;
        fb[i * 4 + 3] = l.hA;
        const aoLevel = Math.max(0, Math.min(31, Math.round(l.ao * 31)));
        uc[i * 4] = (l.type & 7) + aoLevel * 8;
        uc[i * 4 + 1] = Math.round(clamp01(l.s1) * 255);
        uc[i * 4 + 2] = Math.round(clamp01(l.s2) * 255);
        uc[i * 4 + 3] = Math.round(clamp01(l.bend) * 255);
        const reach = l.type === LeafType.GLINT ? 0.2 : l.len;
        min.x = Math.min(min.x, l.x - reach);
        min.y = Math.min(min.y, l.y - reach);
        min.z = Math.min(min.z, l.z - reach);
        max.x = Math.max(max.x, l.x + reach);
        max.y = Math.max(max.y, l.y + reach);
        max.z = Math.max(max.z, l.z + reach);
      }
      // The sway carries a column sideways; the bound has to cover it or the
      // frustum test clips a leaning canopy at the screen edge.
      const top = max.y - min.y;
      const sway = swayMargin + SWAY_GAIN * Math.pow(Math.max(0, top), SWAY_POWER) * 2.6;
      min.x -= sway;
      min.z -= sway;
      max.x += sway;
      max.z += sway;

      const a = new THREE.InstancedBufferAttribute(fa, 4);
      const b = new THREE.InstancedBufferAttribute(fb, 4);
      const c = new THREE.InstancedBufferAttribute(uc, 4, true);
      const sphere = new THREE.Sphere();
      new THREE.Box3(min, max).getBoundingSphere(sphere);

      const meshes = this.levels.map((_, level) => {
        const geometry = new THREE.InstancedBufferGeometry();
        const template = templates[level];
        geometry.setIndex(template.index);
        geometry.setAttribute('position', template.getAttribute('position'));
        geometry.setAttribute('normal', template.getAttribute('normal'));
        geometry.setAttribute('leafA', a);
        geometry.setAttribute('leafB', b);
        geometry.setAttribute('leafC', c);
        geometry.instanceCount = n;
        geometry.boundingSphere = sphere;
        geometry.boundingBox = new THREE.Box3(min.clone(), max.clone());
        this.geometries.push(geometry);
        const mesh = new THREE.Mesh(geometry, this.materials[level]);
        mesh.name = `${name}-cell`;
        mesh.frustumCulled = true;
        mesh.castShadow = this.castShadow;
        mesh.receiveShadow = false;
        mesh.matrixAutoUpdate = false;
        mesh.matrixWorldAutoUpdate = false;
        mesh.updateMatrix();
        mesh.updateMatrixWorld(true);
        return mesh;
      });

      this.cells.push({ meshes, keep: kept.length, total: n, min, max, level: -1 });
    }
  }
}

/**
 * A strip of `rows` rows of two vertices: `x` is the side (-1, 1), `y` the
 * position along the leaf, `sin(row / (rows - 1) * pi/2)` — abyssal's spacing,
 * which packs rows toward the tip where the curvature is.
 */
function buildStrip(rows: number, cols = 2): THREE.BufferGeometry {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  for (let r = 0; r < rows; r++) {
    const v = Math.sin((r / (rows - 1)) * Math.PI * 0.5);
    for (let c = 0; c < cols; c++) {
      positions.push(-1 + (2 * c) / (cols - 1), v, 0);
      normals.push(0, 0, 1);
    }
  }
  for (let r = 0; r + 1 < rows; r++) {
    for (let c = 0; c + 1 < cols; c++) {
      const a = r * cols + c;
      const b = a + cols;
      indices.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  return geometry;
}

// ------------------------------------------------------------------- utilities

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function wrapTau(phase: number): number {
  const tau = Math.PI * 2;
  return ((phase % tau) + tau) % tau;
}

/** Fisher-Yates from a seed, so a cell's order never depends on the session. */
function shuffle<T>(list: T[], seed: number): void {
  let s = (seed ^ 0x9e3779b9) >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = list[i];
    list[i] = list[j];
    list[j] = tmp;
  }
}

// ---------------------------------------------------------- the rendered floor

/**
 * Height of the rendered floor, memoising the mesh's vertex heights.
 *
 * `seafloorMeshHeight` is four `seafloorHeight` evaluations per call; ground
 * cover seats tens of thousands of tufts and most of them share their four
 * vertices with a neighbour, so the cache takes placement from four
 * evaluations per tuft to well under one.
 */
export class FloorCache {
  private readonly heights = new Map<number, number>();
  private readonly step = SEAFLOOR_EXTENT / SEAFLOOR_SEGMENTS;

  private vertex(ix: number, iz: number): number {
    const key = ix * 65536 + iz;
    let h = this.heights.get(key);
    if (h === undefined) {
      h = seafloorHeight(ix * this.step - SEAFLOOR_EXTENT / 2, iz * this.step - SEAFLOOR_EXTENT / 2);
      this.heights.set(key, h);
    }
    return h;
  }

  /** Mirrors `seafloorMeshHeight`'s triangulation exactly. */
  height(x: number, z: number): number {
    const gx = (x + SEAFLOOR_EXTENT / 2) / this.step;
    const gz = (z + SEAFLOOR_EXTENT / 2) / this.step;
    const ix = Math.floor(gx);
    const iz = Math.floor(gz);
    const u = gx - ix;
    const v = gz - iz;
    const ha = this.vertex(ix, iz);
    const hb = this.vertex(ix, iz + 1);
    const hc = this.vertex(ix + 1, iz + 1);
    const hd = this.vertex(ix + 1, iz);
    if (u + v <= 1) return ha + (hd - ha) * u + (hb - ha) * v;
    return hc + (hb - hc) * (1 - u) + (hd - hc) * (1 - v);
  }

  /** Slope of the rendered floor, as a gradient magnitude. */
  slope(x: number, z: number, e = 2): number {
    const gx = (this.height(x + e, z) - this.height(x - e, z)) / (2 * e);
    const gz = (this.height(x, z + e) - this.height(x, z - e)) / (2 * e);
    return Math.hypot(gx, gz);
  }
}

/** 2D value noise in [0, 1], for placement fields. */
export function valueNoise(x: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const h = (i: number, j: number) => {
    let n = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(seed, 2246822519);
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  };
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const a = h(ix, iz) + (h(ix + 1, iz) - h(ix, iz)) * ux;
  const b = h(ix, iz + 1) + (h(ix + 1, iz + 1) - h(ix, iz + 1)) * ux;
  return a + (b - a) * uz;
}

export function fbm(x: number, z: number, seed: number, octaves = 3): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise(x * f, z * f, seed + i * 101) * amp;
    norm += amp;
    amp *= 0.5;
    f *= 2.03;
  }
  return sum / norm;
}
