import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cos,
  faceDirection,
  float,
  mix,
  normalGeometry,
  positionGeometry,
  sin,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { ISLAND } from './Seafloor';

/**
 * Grass on the island, placed and animated entirely in the vertex stage.
 *
 * **Why this is not a scatter of models.** The island is about 0.8 km² of
 * vegetated ground. Covering that with instanced tufts at a density that reads
 * as grass rather than as litter needs somewhere north of a million instances,
 * and `Props` cannot pay for a thousandth of that — its meadow kinds run to a
 * few hundred each and, seen from ten metres, look exactly like a few hundred
 * plants standing on bare ground. That is the state this replaces.
 *
 * The way out is the one every open-world renderer takes: stop placing grass
 * over the world and start placing it around the *camera*. Only the ground
 * within a hundred metres can resolve a blade at all, so only that ground gets
 * any, and the same few tens of thousands of blades are reused wherever the
 * viewer happens to be.
 *
 * Three ideas carry the module.
 *
 * **The field tiles, and the tile follows the camera.** Each blade owns a fixed
 * offset inside one `FIELD_PERIOD` square. The vertex stage picks the copy of
 * that square nearest the camera — `round((camera - offset) / period)` — so a
 * blade's world position is either exactly where it was last frame or a whole
 * period away. It never slides, which is what a naive "origin follows the
 * camera" scheme does and what makes grass look like it is flowing downhill.
 * The teleport happens only at the edge of the field, where the fade has
 * already taken the blade to zero height.
 *
 * **The ground decides where grass grows, on the GPU.** `Seafloor` publishes its
 * heightfield as a TSL node — the same field the CPU samples and the floor mesh
 * is displaced by — so a blade can ask how high the ground is under itself and
 * how vegetated the terrain shader has painted it, and collapse to nothing if
 * the answer is "beach", "rock face" or "sea". No CPU placement pass, no
 * placement buffer, and the grass automatically follows the island if the
 * heightfield ever changes again.
 *
 * **Nothing accumulates.** Like the kelp and the fish, every visible quantity is
 * a closed-form function of one phase and the blade's own constants, so
 * `resetClock(t)` reproduces a frame exactly — which is what the visual
 * regression harness requires.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/** `vec3` under a loose signature; see the same alias in `Fish`. */
const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;
const vec2n = vec2 as unknown as (x: unknown, y: unknown) => Node;
/**
 * `mix` under a loose signature. Composing colour uniforms with varyings gives
 * nodes the overloads resolve to the scalar form; the runtime is unaffected.
 */
const mixn = mix as unknown as (a: unknown, b: unknown, t: unknown) => Node;

const MEADOW_SEED = 0x9d3a17;

/** Instance-buffer capacity. `setCount` draws a prefix of it. */
const MAX_BLADES = 90_000;

/**
 * Side of the tile a blade's offset lives in, metres — and therefore the width
 * of the field, since exactly one copy of each blade is ever within it.
 *
 * 72 m is a compromise between two things that both get worse in the same
 * direction. Larger, and the same tuft count spreads thinner, so the ground
 * near the camera — the only ground anyone can see a blade on — gets sparser.
 * Smaller, and the far edge of the field is close enough that the fade to bare
 * ground is inside the range where grass still reads, so the viewer sees a
 * circle of lawn following them about.
 *
 * The first cut of this was 150 m and gave two blades per square metre, which
 * renders as exactly what it is: individual spikes standing in bare ground. Real
 * turf is two orders of magnitude denser than that, and the gap is closed from
 * both ends — this halved, and each instance made a tuft rather than a blade.
 */
const FIELD_PERIOD = 72;

/**
 * Where the field starts fading, as a fraction of its half-width.
 *
 * The fade is on the Chebyshev distance rather than the Euclidean one, which is
 * to say it fades toward the edges of the *square* the blades tile in. Fading on
 * a circle inscribed in that square would throw away the corners — a fifth of
 * every blade in the buffer, permanently.
 */
const FADE_START = 0.78;

/** Blade height, metres, before per-blade variation. */
const BLADE_HEIGHT = 0.46;
const BLADE_HEIGHT_VARIATION = 0.5;
/** Blade half-width at the base, metres. */
const BLADE_WIDTH = 0.03;

/**
 * Rows along a blade, and the curl that stops it being a flat ribbon.
 *
 * Two rows are one span per column, four triangles for a blade 40 cm long. The
 * bend is still carried by the analytic vertex shear; a second span costs twice
 * the meadow geometry without adding a silhouette breakpoint at five metres.
 */
const BLADE_ROWS = 2;
const BLADE_CURL = 0.55;

/**
 * Blades in one tuft, and how far they spread from its centre.
 *
 * Grass grows in tillers from a crown, not as isolated blades on a grid, so a
 * tuft is both what the plant actually does and the cheapest way to buy density:
 * one instance, one set of heightfield evaluations, four blades. Uniform singles
 * at the same instance count read as a bed of nails — see `FIELD_PERIOD`.
 */
const TUFT_BLADES = 4;
const TUFT_SPREAD = 0.07;

/**
 * Elevation band the grass grows in, metres above mean sea level.
 *
 * The lower edge matches `Seafloor`'s own `BEACH_TOP_METRES`, because that is
 * where the terrain shader stops painting sand and starts painting vegetation —
 * grass that grew below it would be grass on a beach the ground colour says is
 * bare. The upper edge is the treeline, and it is derived rather than typed:
 * `Seafloor` fades its vegetation into bare summit rock between 0.72 and 0.95 of
 * `ISLAND.peak`, so the grass fades out over the same two heights. Written this
 * way because the alternative is two hard-coded numbers that have to be edited
 * together, and the first time the island's height changed they were not — a
 * lawn kept growing over a summit the terrain shader had already turned to
 * stone.
 */
const GRASS_MIN_HEIGHT = 3.2;
const GRASS_FULL_HEIGHT = 12;
const GRASS_FADE_HEIGHT = ISLAND.peak * 0.72;
const GRASS_MAX_HEIGHT = ISLAND.peak * 0.95;

/**
 * How far outside the island the field bothers to exist, in island radii.
 *
 * A cheap early-out that costs one length: the heightfield check below would
 * reject every blade anyway once the camera is out at sea, but only after
 * paying for four octaves of noise per vertex to find out. This rejects the
 * whole field on the instance's own offset.
 */
const ISLAND_REACH = 1.45;

/** Wind: metres per second of blade travel, and the wave that carries it. */
const GUST_WAVELENGTH = 26;
const GUST_K = (Math.PI * 2) / GUST_WAVELENGTH;
const GUST_SPEED = 5.2;
const GUST_OMEGA = GUST_K * GUST_SPEED;

/**
 * Bend at full wind, as a fraction of blade height, and the mean lean under it.
 *
 * Grass does not oscillate about vertical; it leans downwind and flutters about
 * that lean. A field animated as a symmetric sway reads as underwater weed,
 * which is the one thing this must not look like given what else is in the
 * scene.
 */
const BEND_MAX = 0.62;
const BEND_LEAN = 0.22;

/** Blade colours, base to tip, and the dry variant mixed in per blade. */
const GRASS_BASE_COLOR = new THREE.Color(0.055, 0.085, 0.035);
const GRASS_TIP_COLOR = new THREE.Color(0.29, 0.42, 0.16);
const GRASS_DRY_COLOR = new THREE.Color(0.44, 0.42, 0.2);

const SUN_COLOR = new THREE.Color(1.0, 0.96, 0.88);
const AMBIENT_COLOR = new THREE.Color(0.36, 0.44, 0.5);

export interface IslandMeadowOptions {
  seed?: number;
  /**
   * How much key light reaches a world point, 0..1 — the island's own shadow
   * times the cloud deck's. Sampled once per blade in the vertex stage.
   *
   * Without it the sward stayed fully lit across a hillside whose terrain now
   * has a lit face and a shaded one, which reads worse than the flat lighting
   * it replaced: grass glowing on ground that is in shadow.
   */
  sunOcclusion?: (worldPosition: unknown) => unknown;
}

/**
 * A field of grass that follows the camera across the island.
 *
 * Add `object` to the scene root: the vertex stage emits world coordinates, so
 * the container must carry an identity transform, exactly as `KelpForest` and
 * `FishSchool` do.
 */
export class IslandMeadow {
  static readonly MAX_COUNT = MAX_BLADES;

  readonly object: THREE.Object3D;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;

  private count: number;
  private wantVisible = true;
  private disposed = false;
  private phase = 0;

  private readonly uCamera = uniform(new THREE.Vector2(0, 0));
  private readonly uPhase = uniform(0);
  private readonly uWind = uniform(new THREE.Vector2(1, 0));
  private readonly uWindStrength = uniform(0.5);
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(SUN_COLOR));
  private readonly uAmbient = uniform(new THREE.Color(AMBIENT_COLOR));
  private readonly uBase = uniform(new THREE.Color(GRASS_BASE_COLOR));
  private readonly uTip = uniform(new THREE.Color(GRASS_TIP_COLOR));
  private readonly uDry = uniform(new THREE.Color(GRASS_DRY_COLOR));

  /**
   * The heightfield, as a TSL node factory.
   *
   * Injected rather than imported so this module never owns a second copy of the
   * terrain: it is handed `Seafloor`'s own node, which is the same field the
   * floor mesh is displaced by and the CPU samples for placement.
   */
  private readonly groundHeight: (worldPosition: Node) => Node;
  private readonly sunOcclusion: ((worldPosition: Node) => Node) | null;

  constructor(
    count: number,
    groundHeight: (worldPosition: Node) => Node,
    options: IslandMeadowOptions = {},
  ) {
    this.count = clampCount(count);
    this.groundHeight = groundHeight;
    this.sunOcclusion = (options.sunOcclusion as ((p: Node) => Node) | undefined) ?? null;

    this.geometry = buildBladeGeometry();
    attachInstanceAttributes(this.geometry, options.seed ?? MEADOW_SEED);
    this.geometry.instanceCount = this.count;
    // The field is defined relative to the camera, so no fixed bound describes
    // it and culling it against one would be culling it against last frame's
    // position. It is a single draw covering at most a 150 m square.
    this.geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(ISLAND.x, 0, ISLAND.z),
      Number.POSITIVE_INFINITY,
    );

    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'island-meadow';
    this.mesh.frustumCulled = false;
    // A 40 cm blade casts nothing legible at any distance this is visible from,
    // and ninety thousand of them through the depth pass would double the cost
    // of the whole system to render it.
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();

    this.object = new THREE.Object3D();
    this.object.name = 'island-grass-field';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Blades drawn, 0..`IslandMeadow.MAX_COUNT`.
   *
   * Free: the buffer is built once at capacity and this only moves the draw's
   * instance count. Blade `i` keeps its offset, height and phase at every tier,
   * so a tier change thins the field evenly rather than relocating it.
   */
  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.count) return;
    this.count = next;
    this.geometry.instanceCount = next;
    this.applyVisibility();
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** `dir` points *toward* the sun and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(dir);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();
  }

  setSunColor(color: THREE.Color): void {
    (this.uSunColor.value as THREE.Color).copy(color);
  }

  setAmbientColor(color: THREE.Color): void {
    (this.uAmbient.value as THREE.Color).copy(color);
  }

  /**
   * Wind direction and strength, 0..1.
   *
   * `direction` is the bearing the wind blows *toward*, in the same convention
   * the rest of the project uses for the sea state.
   */
  setWind(direction: THREE.Vector2, strength: number): void {
    const wind = this.uWind.value as THREE.Vector2;
    wind.copy(direction);
    if (wind.lengthSq() < 1e-8) wind.set(1, 0);
    wind.normalize();
    this.uWindStrength.value = Math.max(0, Math.min(1, strength));
  }

  /**
   * Moves the field to the camera and advances the wind.
   *
   * The camera position is written raw, not snapped: the field's *tiling* is
   * what keeps blades still in world space, so there is nothing here for a snap
   * to protect and snapping would only make the fade at the far edge move in
   * steps. Two scalar writes and no allocation.
   */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (this.disposed) return;
    this.phase = wrapTau(this.phase + GUST_OMEGA * dt);
    this.uPhase.value = this.phase;
    (this.uCamera.value as THREE.Vector2).set(cameraPosition.x, cameraPosition.z);
  }

  /** Jumps the wind clock, for reproducible captures. */
  resetClock(time = 0): void {
    if (this.disposed) return;
    this.phase = wrapTau(time * GUST_OMEGA);
    this.uPhase.value = this.phase;
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

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.count > 0;
  }

  private buildMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    material.name = 'island-grass';
    // A blade is a sheet with no thickness, so both faces have to rasterise.
    material.side = THREE.DoubleSide;
    material.fog = false;

    const seed: Node = attribute('bladeSeed', 'vec4');
    const p: Node = positionGeometry;
    const n: Node = normalGeometry;

    /** Blade coordinate: 0 at the root, 1 at the tip. */
    const s = p.y;

    // --- which copy of this blade is nearest the camera ---------------------
    //
    // The whole point of the module, in three lines. `offset` is fixed for the
    // life of the blade, so the only thing that changes as the camera moves is
    // *which* period it is picked in — and that is a whole-tile jump at the far
    // edge of the field rather than a slide underfoot.
    const offset = vec2n(seed.x, seed.y).mul(FIELD_PERIOD);
    const tile = this.uCamera.sub(offset).div(FIELD_PERIOD).round();
    const world = offset.add(tile.mul(FIELD_PERIOD));

    const toCamera = world.sub(this.uCamera);
    // Chebyshev distance: the blades tile a square, so the field is faded on the
    // square. See `FADE_START`.
    const edge = toCamera.x.abs().max(toCamera.y.abs()).div(FIELD_PERIOD * 0.5);
    const edgeFade = float(1).sub(edge.smoothstep(FADE_START, 1));

    // --- does grass grow here? ---------------------------------------------
    const ground = this.groundHeight(vec3n(world.x, 0, world.y)) as Node;

    // The same band the terrain shader paints vegetation over, so the grass and
    // the ground colour under it agree about where the beach ends.
    const band = ground
      .smoothstep(GRASS_MIN_HEIGHT, GRASS_FULL_HEIGHT)
      .mul(float(1).sub(ground.smoothstep(GRASS_FADE_HEIGHT, GRASS_MAX_HEIGHT)));

    // Off the island entirely — an early reject on the blade's own offset, so
    // the sea and the open plateau cost one length rather than a field of
    // collapsed blades.
    const fromIsland = world.sub(vec2(ISLAND.x, ISLAND.z)).length();
    const onIsland = float(1).sub(fromIsland.smoothstep(ISLAND.radius, ISLAND.radius * ISLAND_REACH));

    // Per-blade thinning, so the field has bare patches rather than an even
    // stipple. `seed.z` is already the height draw; reusing it correlates
    // sparseness with shortness, which is what thin ground actually looks like.
    const thin = seed.z.smoothstep(0.06, 0.3);

    const presence = edgeFade.mul(band).mul(onIsland).mul(thin);

    const height = float(BLADE_HEIGHT)
      .mul(float(1).sub(BLADE_HEIGHT_VARIATION).add(seed.z.mul(BLADE_HEIGHT_VARIATION * 2)))
      .mul(presence);

    // --- the blade's own frame ---------------------------------------------
    const spin = seed.w.mul(Math.PI * 2);
    const across = vec3n(cos(spin), 0, sin(spin));
    const face = vec3n(sin(spin).negate(), 0, cos(spin));

    // --- the wind -----------------------------------------------------------
    //
    // One travelling wave over the ground, so a gust crosses the field instead
    // of every blade breathing in time. The per-blade phase offset keeps
    // neighbours from being locked together at the wave's own scale.
    const downwind = world.x.mul(this.uWind.x).add(world.y.mul(this.uWind.y));
    const gust = this.uPhase.sub(downwind.mul(GUST_K)).add(seed.w.mul(Math.PI * 2));
    const flutter = sin(gust).mul(0.5).add(sin(gust.mul(2.3).add(seed.z.mul(9))).mul(0.22));

    // Lean plus flutter, limited together: it is their sum a blade cannot
    // exceed. `tanh` rather than a clamp because the normal below is the
    // analytic derivative of this, and a hard clamp puts a corner in it that
    // arrives as a band of visibly wrong shading sweeping up the blade.
    const drive = this.uWindStrength.mul(float(BEND_LEAN).add(flutter.mul(0.55)));
    const bend = drive.div(BEND_MAX).tanh().mul(BEND_MAX);

    // Quadratic cantilever envelope: zero displacement and zero slope at the
    // root, most of the motion in the top third — which is how a grass blade
    // actually bends.
    const sway = s.mul(s).mul(bend);
    const slope = s.mul(2).mul(bend);

    const windDir = vec3n(this.uWind.x, 0, this.uWind.y);

    // `p.xz` is already in metres — the tuft's own layout — so the only thing
    // applied here is the instance's bearing and the presence fade, which has to
    // reach the width as well as the height or a rejected tuft would leave a
    // mat of zero-height blades lying on the ground.
    const local = across.mul(p.x).add(face.mul(p.z)).mul(presence.max(0.0001));

    const position = vec3n(world.x, ground, world.y)
      .add(local)
      .add(vec3n(0, s.mul(height), 0))
      .add(windDir.mul(sway.mul(height)));

    material.positionNode = position;

    // The shear the sway is: for a displacement along `windDir` that grows with
    // height, the normal tilts back against it by the displacement's slope.
    const worldNormal = across
      .mul(n.x)
      .add(vec3n(0, 1, 0).mul(n.y))
      .add(face.mul(n.z))
      .sub(windDir.mul(slope.mul(n.y)))
      .normalize();

    const vNormal: Node = varying(worldNormal, 'meadowNormal');
    const vBlade: Node = varying(s, 'meadowBlade');
    const vTint: Node = varying(seed.z, 'meadowTint');
    const vKey: Node = varying(
      this.sunOcclusion === null
        ? (float(1) as Node)
        : this.sunOcclusion(vec3n(world.x, ground, world.y)),
      'meadowKey',
    );

    material.colorNode = Fn(() => {
      const normal = vNormal.normalize().mul(faceDirection).toVar();

      const key = normal.dot(this.uSunDir).clamp(0, 1).toVar();
      // Sky fill from the normal's vertical component: a blade lying over
      // catches less of it than one standing up, which is most of what gives a
      // wind-blown field its banding.
      const sky = normal.y.mul(0.5).add(0.5).toVar();

      // Base to tip, dark to light. Grass self-shadows heavily at the root and
      // a uniform blade reads as plastic; this gradient is the cheapest possible
      // stand-in for the occlusion a real sward has.
      const base = mixn(this.uBase, this.uTip, vBlade.smoothstep(0.05, 0.9)).toVar();
      // A fraction of the field is dry. Correlated with height again, because
      // the short thin grass is the grass that is struggling.
      base.assign(mixn(base, this.uDry, float(1).sub(vTint).mul(0.55).mul(vBlade)));

      const light = this.uAmbient
        .mul(sky.mul(0.6).add(0.4))
        .add(this.uSunColor.mul(key).mul(0.9).mul(vKey));

      return vec4(base.mul(light), 1);
    })();

    return material;
  }
}

// --------------------------------------------------------------------- geometry

/**
 * One blade: a tapered, curled strip.
 *
 * Three columns rather than two so the blade has a spine to curl about — a flat
 * ribbon catches the light as a single flat tone and a whole field of them
 * flickers as they turn. Two rows are enough for the bend because the envelope
 * is applied analytically in the vertex stage and the blade is 40 cm long.
 */
function buildBladeGeometry(): THREE.InstancedBufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];
  // Deterministic, and local: the tuft's shape is part of the mesh, so it must
  // not draw from the instance stream or every blade in the buffer would shift
  // when the tuft changed.
  const random = mulberry32(MEADOW_SEED ^ 0x7f4a2b);

  for (let blade = 0; blade < TUFT_BLADES; blade++) {
    const base = positions.length / 3;
    // Each tiller leaves the crown on its own bearing and leans out as it rises,
    // which is what gives a tuft its fountain shape rather than a bundle of
    // parallel sticks.
    const bearing = ((blade + random() * 0.6) / TUFT_BLADES) * Math.PI * 2;
    const dirX = Math.cos(bearing);
    const dirZ = Math.sin(bearing);
    const spread = TUFT_SPREAD * (0.35 + random() * 0.65);
    const height = 0.65 + random() * 0.35;
    const twist = random() * Math.PI;

    for (let row = 0; row < BLADE_ROWS; row++) {
      const s = row / (BLADE_ROWS - 1);
      // Taper to a point: a blade that ends in a flat edge reads as a strip of
      // tape at any distance where the tip is more than a pixel.
      const w = (1 - s * s * 0.92) * BLADE_WIDTH;
      // The blade's own width axis, turned so neighbouring tillers in a tuft do
      // not all present the same face to the light.
      const ax = Math.cos(twist);
      const az = Math.sin(twist);
      // Lean grows with height, so the crown stays tight and the tips fan.
      const outX = dirX * spread * s;
      const outZ = dirZ * spread * s;

      positions.push(outX - w * ax, s * height, outZ - w * az);
      positions.push(outX + BLADE_CURL * w * az, s * height, outZ - BLADE_CURL * w * ax);
      positions.push(outX + w * ax, s * height, outZ + w * az);
    }

    for (let row = 0; row + 1 < BLADE_ROWS; row++) {
      for (let col = 0; col < 2; col++) {
        const a = base + row * 3 + col;
        const b = a + 1;
        const c = b + 3;
        const d = a + 3;
        indices.push(a, b, c, a, c, d);
      }
    }
  }

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Per-blade constants: `xy` the offset inside the tile, `z` the height draw,
 * `w` the spin and phase.
 *
 * Drawn in index order from one generator, so blade `i` is the same blade at
 * every tier and between runs — the property the capture harness depends on,
 * and the same one `KelpForest` and `FishSchool` are built around.
 */
function attachInstanceAttributes(geometry: THREE.InstancedBufferGeometry, seed: number): void {
  const random = mulberry32(seed);
  const data = new Float32Array(MAX_BLADES * 4);

  for (let i = 0; i < MAX_BLADES; i++) {
    data[i * 4 + 0] = random();
    data[i * 4 + 1] = random();
    data[i * 4 + 2] = random();
    data[i * 4 + 3] = random();
  }

  geometry.setAttribute('bladeSeed', new THREE.InstancedBufferAttribute(data, 4));
}

function clampCount(count: number): number {
  return Math.max(0, Math.min(MAX_BLADES, Math.floor(count)));
}

/** Reduces a phase to [0, 2pi) in float64, before it reaches a float32 uniform. */
function wrapTau(phase: number): number {
  const tau = Math.PI * 2;
  return ((phase % tau) + tau) % tau;
}
