import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  float,
  fract,
  mix,
  mx_fractal_noise_float,
  normalize,
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
import { TRANSLUCENCY_GAIN, TRANSLUCENCY_POWER } from './groundShading';

/**
 * Distant canopy, drawn as billboards.
 *
 * **The problem this exists for.** From the play area the island is 1.4 km away,
 * where a 10 m tree is about eight pixels. `Props` plants a hundred and thirty
 * of them and every one is at LOD2, which is the right call for cost and the
 * wrong shape for the picture: a hundred and thirty eight-pixel objects on 0.8
 * km² of ground is a sprinkle, and the island reads as a bare hill with specks
 * on it. `Seafloor` carries the biome in the terrain colour, which fixes the
 * *hue* of the hill and can do nothing about its *silhouette* — a painted hill
 * is still a smooth dome against the sky.
 *
 * What is missing is canopy volume: the lumpy, broken edge a forest gives a
 * ridge, and the depth that stops a hillside being a gradient. Every open-world
 * renderer solves this the same way, and has since the first one — past the last
 * mesh LOD, trees become camera-facing cards. It is the cheapest geometry in the
 * scene per unit of silhouette and it is what makes distant forest read.
 *
 * **Why the cards are procedural rather than rendered from the trees.** The
 * usual pipeline bakes an impostor atlas: render each tree from a ring of angles
 * into a texture and pick the nearest view at runtime. That is the right answer
 * when the impostor has to hold up at fifty metres, and it is a build step, an
 * atlas, a licence question and an octahedral lookup. These are never seen
 * closer than 90 m, where a clump is a blob with a broken edge — so the card is
 * a blob with a broken edge, generated in the shader from the instance's own
 * seed. No asset, no atlas, no sampler, and nothing to keep in step with the
 * models when they change.
 *
 * **Placement is the terrain's, not a table's.** Like `IslandMeadow`, this is
 * handed `Seafloor`'s heightfield as a TSL node and asks it directly: a card
 * whose ground is below the beach line or above the treeline collapses to zero
 * size in the vertex stage. There is no CPU placement pass and no buffer to
 * rebuild if the island ever changes shape again.
 *
 * Nothing accumulates — every visible quantity is a function of the instance
 * seed and one phase uniform, so `resetClock(t)` reproduces a frame exactly,
 * which the visual regression harness requires.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;
const vec2n = vec2 as unknown as (x: unknown, y: unknown) => Node;

const CANOPY_SEED = 0x4f2b91;

/** Instance-buffer capacity. `setCount` draws a prefix of it. */
const MAX_CARDS = 24_000;

/**
 * Card size in metres, before per-instance variation.
 *
 * A card is a *clump* of canopy rather than one tree — roughly what a small
 * stand covers — because at this range individual crowns are not resolvable and
 * a card per tree would need ten times the count for the same coverage. 16 m
 * across and 11 m tall is a few trees' worth, which is the scale the broken edge
 * of a real treeline works at.
 */
const CARD_WIDTH = 19;
const CARD_HEIGHT = 12;
const CARD_VARIATION = 0.42;

/**
 * How far a card's *proportions* may depart from `CARD_WIDTH:CARD_HEIGHT`.
 *
 * `CARD_VARIATION` scales both axes together, so every card was the same ellipse
 * at a different size — and a field of one shape at one aspect is the strongest
 * of the bubble-wrap cues, stronger than either tone or size, because the eye
 * reads repeated *outline* long before it reads repeated colour. Applied as a
 * factor on the width and its reciprocal on the height, so a card's area is
 * unchanged and the field's coverage does not move with this number: it varies
 * squat stands against columnar ones and nothing else.
 */
const CARD_ASPECT_VARIATION = 0.34;

/**
 * Where the cards take over from the meshes, in metres from the camera.
 *
 * These were 320 and 520, and that was the single worst-looking thing about the
 * island: fly toward it and a bald crescent opened across the whole near half of
 * the landmass, travelling with the camera. From 750 m out — a routine approach —
 * everything from the near shore to past the summit was bare ground, because the
 * fade is per *card*, on the card's own distance, so what it cuts is a disc
 * around the viewer rather than a state the island is in.
 *
 * The old numbers were reasoned from the wrong quantity. They were set against
 * `Props`' *last* LOD switch at 420 m on the argument that the meshes take the
 * silhouette back over inside it — but the meshes are ~110 trees over 0.8 km²,
 * an accent layer two hundred times sparser than this field, and 420 m is where
 * they are at their *coarsest*, not where they are dense. There was never
 * anything on the other side of that handover to hand over to.
 *
 * What actually decides the near edge is when a card stops passing for canopy,
 * and that is an angle rather than a distance: a card is about 19 m across, so
 * it subtends 6° at 180 m and 12° at 90 m. Past 6° the flat sheet is still
 * reading as a crown; by 12° it is reading as a sheet. Hence 90 to 180 — and
 * inside 90 m the viewer is in among the trees, where the meshes are at LOD0 and
 * are the only honest answer.
 */
const FADE_IN_NEAR = 90;
const FADE_IN_FAR = 180;

/**
 * Elevation band, matching `Seafloor`'s vegetation ramp and treeline.
 *
 * Derived from `ISLAND.peak` for the upper edge rather than typed, for the same
 * reason `IslandMeadow` derives its: the two were hard-coded independently once
 * and the island's height changed without them.
 */
const CANOPY_MIN_HEIGHT = 6;
const CANOPY_FULL_HEIGHT = 21;
const CANOPY_FADE_HEIGHT = ISLAND.peak * 0.5;
const CANOPY_MAX_HEIGHT = ISLAND.peak * 0.78;

/** How far past the shore radius cards may sit, in island radii. */
const ISLAND_REACH = 1.22;

/** Canopy colours. Deliberately the same family as `Seafloor`'s ground biome. */
const CANOPY_SUNLIT = new THREE.Color(0.062, 0.078, 0.036);
const CANOPY_SHADE = new THREE.Color(0.026, 0.034, 0.02);
/** The flowering accent, on a small fraction of cards. See `bloom` below. */
const CANOPY_BLOOM = new THREE.Color(0.44, 0.1, 0.03);
/**
 * Dry, sun-bleached canopy — the yellower stand on an exposed spur.
 *
 * Warmer and lighter than either of the greens above, but not by much: it is a
 * different *stand* of the same forest, not a different biome, and pushing it
 * further turns the hillside patchy rather than varied.
 */
const CANOPY_DRY = new THREE.Color(0.072, 0.07, 0.03);

/**
 * Feature width of the stand-variation field, metres, and how far it and the
 * per-instance term are each allowed to push the tone.
 *
 * Two scales, because they do different jobs. The field makes one part of the
 * hillside browner than another, which is what a viewer reads as a forest having
 * regions; the per-instance term breaks up neighbouring crowns, which is what
 * stops a region reading as a flat wash. Either alone looks wrong — the field on
 * its own gives smooth blotches of identical cards, and the instance term on its
 * own gives uniform noise.
 */
const STAND_FIELD_METRES = 150;
const STAND_FIELD_WEIGHT = 0.62;
const STAND_INSTANCE_WEIGHT = 0.38;
/** Peak-to-peak brightness jitter the combined tone applies. */
const STAND_VALUE_RANGE = 0.42;
/** Dome normal lateral strength, upward bias, and foliage wrap. */
const DOME_BULGE = 0.55;
const CANOPY_UP_BIAS = 0.45;
const CANOPY_WRAP = 0.65;
const CANOPY_DETAIL_CHEAP = 0;
const CANOPY_DETAIL_FULL = 1;
const CANOPY_FULL_DETAIL_COUNT = 15_000;
/** Extra terrain taps used to make dry stands follow the sun-facing slope. */
const SLOPE_SAMPLE_METRES = 4;
const STAND_SLOPE_WEIGHT = 0.25;

/** Sway: metres of crown travel at full wind, and the wave that carries it. */
const SWAY_WAVELENGTH = 90;
const SWAY_K = (Math.PI * 2) / SWAY_WAVELENGTH;
const SWAY_SPEED = 7;
const SWAY_OMEGA = SWAY_K * SWAY_SPEED;
const SWAY_MAX = 0.9;

const CLOCK_WRAP = 3600;

function canopyDetailForCount(count: number): number {
  return count >= CANOPY_FULL_DETAIL_COUNT ? CANOPY_DETAIL_FULL : CANOPY_DETAIL_CHEAP;
}

export interface IslandCanopyOptions {
  seed?: number;
  /**
   * How much key light reaches a world point, 0..1. Sampled once per card, at
   * its foot, in the vertex stage.
   *
   * A constructor input rather than a setter because the material is built once
   * and the sample has to be inside its graph. Everything it needs — the cloud
   * field and the heightfield — exists before the canopy is planted.
   */
  sunOcclusion?: (worldPosition: unknown) => unknown;
}

function clampCount(count: number): number {
  return Math.max(0, Math.min(MAX_CARDS, Math.floor(count)));
}

/**
 * One quad, corners at +/-0.5, with the origin at the *bottom* centre.
 *
 * Bottom-centred because the card is planted: the ground decides where its foot
 * is and it grows upward from there. A centre-origin quad would have to be
 * lifted by half its own height, which is one more thing to get wrong when the
 * height varies per instance.
 */
function buildCardGeometry(): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  const position = new Float32Array([
    -0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0,
  ]);
  const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geometry;
}

function attachInstanceAttributes(geometry: THREE.InstancedBufferGeometry, seed: number): void {
  const random = mulberry32(seed);
  const data = new Float32Array(MAX_CARDS * 4);
  for (let i = 0; i < MAX_CARDS; i++) {
    // x,y: position on the island disc, drawn as sqrt(r) so the scatter is
    // uniform in *area* rather than crowding the middle.
    const angle = random() * Math.PI * 2;
    const radius = Math.sqrt(random()) * ISLAND.radius * ISLAND_REACH;
    data[i * 4 + 0] = ISLAND.x + Math.cos(angle) * radius;
    data[i * 4 + 1] = ISLAND.z + Math.sin(angle) * radius;
    // z: size and colour draw. w: sway phase and the flowering draw.
    data[i * 4 + 2] = random();
    data[i * 4 + 3] = random();
  }
  geometry.setAttribute('cardSeed', new THREE.InstancedBufferAttribute(data, 4));
}

/**
 * A field of canopy billboards standing on the island.
 *
 * Add `object` to the scene root: the vertex stage emits world coordinates, so
 * the container must carry an identity transform, exactly as `IslandMeadow` and
 * `FishSchool` do.
 */
export class IslandCanopy {
  static readonly MAX_COUNT = MAX_CARDS;

  readonly object: THREE.Object3D;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private material: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;

  private count: number;
  private detail = CANOPY_DETAIL_FULL;
  private wantVisible = true;
  private disposed = false;
  private phase = 0;

  private readonly uPhase = uniform(0);
  private readonly uWind = uniform(new THREE.Vector2(1, 0));
  private readonly uWindStrength = uniform(0.5);
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(0xfff2df));
  private readonly uAmbient = uniform(new THREE.Color(0x9dbbe0));
  private readonly uSunlit = uniform(new THREE.Color(CANOPY_SUNLIT));
  private readonly uShade = uniform(new THREE.Color(CANOPY_SHADE));
  private readonly uBloom = uniform(new THREE.Color(CANOPY_BLOOM));
  /**
   * The colour the dry patches go.
   *
   * Twenty-four thousand cards were two tones between them — `mix(shade,
   * sunlit, cornerY)` and nothing else — which is the whole of the bubble-wrap
   * read: identical round blobs at identical tone tile into a texture rather
   * than into a forest. `4ttSWf` gives each tree a per-instance material offset
   * *and* runs a low-frequency `brownAreas` field across the whole canopy, and
   * the two together are what make its hillside vary in tone across its width.
   */
  private readonly uDry = uniform(new THREE.Color(CANOPY_DRY));

  /**
   * How much key light reaches a card, 0..1. Evaluated per vertex.
   *
   * Per vertex rather than per fragment, and that is not a compromise: a card is
   * a few metres across, a cloud shadow is hundreds and the hill's own shadow is
   * larger still, so four samples per card is far above the rate either field
   * needs. It is also the only affordable place — the canopy is drawn with heavy
   * overdraw across a hillside, and a heightfield march per fragment would be
   * paid several times over for every pixel.
   */
  private readonly sunOcclusion: ((worldPosition: Node) => Node) | null;

  private readonly groundHeight: (worldPosition: Node) => Node;

  constructor(
    count: number,
    groundHeight: (worldPosition: Node) => Node,
    options: IslandCanopyOptions = {},
  ) {
    this.count = clampCount(count);
    this.detail = canopyDetailForCount(this.count);
    this.groundHeight = groundHeight;
    this.sunOcclusion = (options.sunOcclusion as ((p: Node) => Node) | undefined) ?? null;

    this.geometry = buildCardGeometry();
    attachInstanceAttributes(this.geometry, options.seed ?? CANOPY_SEED);
    this.geometry.instanceCount = this.count;
    // Bounded by the island, not by the camera — unlike the meadow these are
    // planted. A sphere around the island would be legitimate; infinity is used
    // because the cards billboard, so their bounds change every frame and a
    // tight sphere would have to be recomputed rather than reasoned about.
    this.geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(ISLAND.x, 0, ISLAND.z),
      Number.POSITIVE_INFINITY,
    );

    this.material = this.buildMaterial(this.detail);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'island-canopy';
    this.mesh.frustumCulled = false;
    // No shadows, and this is not a saving so much as a correctness point: a
    // card is a flat sheet facing the camera, so the shadow it casts is a flat
    // sheet facing the camera, which from the sun's direction is a line. The
    // meshes that these stand in for are 1.4 km from the shadow cascade anyway.
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();

    this.object = new THREE.Object3D();
    this.object.name = 'island-canopy-field';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
  }

  getCount(): number {
    return this.count;
  }

  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.count) return;
    this.count = next;
    this.geometry.instanceCount = next;
    // Existing main.ts quality application already routes through setCount.
    // Keep that path quality-aware until the caller supplies an explicit detail
    // field: current Low/Medium counts are below the High canopy budget.
    this.setDetail(canopyDetailForCount(next));
    this.applyVisibility();
  }

  /**
   * Selects the fragment graph: values below 1 use the cheap Low/Medium path;
   * values at or above 1 retain the full High+ silhouette detail. Call this
   * while the quality-change drain is paused, before compiling the new tier.
   */
  setDetail(detail: number): void {
    const next = detail >= CANOPY_DETAIL_FULL ? CANOPY_DETAIL_FULL : CANOPY_DETAIL_CHEAP;
    if (next === this.detail || this.disposed) return;
    this.detail = next;
    const previous = this.material;
    this.material = this.buildMaterial(next);
    this.mesh.material = this.material;
    previous.dispose();
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** `dir` points *toward* the sun and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    (this.uSunDir.value as THREE.Vector3).copy(dir).normalize();
  }

  setSunColor(color: THREE.Color): void {
    (this.uSunColor.value as THREE.Color).copy(color);
  }

  setAmbientColor(color: THREE.Color): void {
    (this.uAmbient.value as THREE.Color).copy(color);
  }

  /** `direction` is a unit vector in world xz; `strength` is 0..1. */
  setWind(direction: THREE.Vector2, strength: number): void {
    const wind = this.uWind.value as THREE.Vector2;
    wind.copy(direction);
    if (wind.lengthSq() < 1e-6) wind.set(1, 0);
    wind.normalize();
    this.uWindStrength.value = Math.max(0, Math.min(1, strength));
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.phase = (this.phase + dt) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  resetClock(time = 0): void {
    this.phase = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.remove(this.mesh);
    this.geometry.dispose();
    this.material.dispose();
  }

  private applyVisibility(): void {
    this.mesh.visible = this.wantVisible && this.count > 0;
  }

  private buildMaterial(detail: number): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    material.transparent = false;
    // Alpha *test*, not blend, and for the same reason the imported trees are
    // masked rather than blended: an instanced mesh is sorted as one object, so
    // twenty thousand blended cards over a hillside cannot resolve against each
    // other and the canopy turns inside out as the camera moves. Testing keeps
    // the depth write, which is what makes the field sort itself.
    // alphaToCoverage smooths the cutout edge while preserving that depth-write
    // ordering on the 8-20 px cards.
    material.alphaTest = 0.42;
    material.alphaToCoverage = true;
    // right x up = -flat points toward the camera for this cylindrical
    // billboard, so FrontSide removes the back-face rasterisation safely.
    material.side = THREE.FrontSide;

    const seed = attribute('cardSeed', 'vec4') as Node;
    const corner = positionGeometry as Node;
    const shade = varying(float(0), 'canopyShade') as Node;
    const bloomMix = varying(float(0), 'canopyBloom') as Node;
    /** Stand tone, 0 lush to 1 dry. See STAND_FIELD_METRES. */
    const stand = varying(float(0), 'canopyStand') as Node;
    /** Key light reaching this card, 0..1. */
    const keyLight = varying(float(1), 'canopyKey') as Node;
    /** Billboard basis, published so the fragment dome has no angle branch. */
    const cardRight = varying(vec3(0, 0, 0), 'canopyRight') as Node;
    const cardFacing = varying(vec3(0, 0, 0), 'canopyFacing') as Node;

    // `Fn`, not a plain closure. TSL only maintains an assignment stack while an
    // `Fn` callback is executing, and the two `varying(...).assign(...)` calls
    // below are assignments: built outside one they throw "No stack defined for
    // assign operation" on every boot, which is exactly what they did. `Seafloor`
    // carries the same note over its own graph.
    material.positionNode = Fn(() => {
      const foot = vec2n(seed.x, seed.y);
      const ground = this.groundHeight(vec3n(foot.x, 0, foot.y)) as Node;

      // --- does canopy grow here? -------------------------------------------
      // The same two ramps `Seafloor` paints its vegetation band with, so the
      // cards stop exactly where the green in the ground stops.
      const band = ground
        .smoothstep(CANOPY_MIN_HEIGHT, CANOPY_FULL_HEIGHT)
        .mul(float(1).sub(ground.smoothstep(CANOPY_FADE_HEIGHT, CANOPY_MAX_HEIGHT)));

      // Off the island — an early reject on the card's own foot.
      const fromIsland = foot.sub(vec2(ISLAND.x, ISLAND.z)).length();
      const onIsland = float(1).sub(
        fromIsland.smoothstep(ISLAND.radius * 0.98, ISLAND.radius * ISLAND_REACH),
      );

      // --- hand over from the meshes ----------------------------------------
      const toCamera = vec3n(foot.x, ground, foot.y).sub(cameraPosition);
      const distance = toCamera.length();
      const handover = distance.smoothstep(FADE_IN_NEAR, FADE_IN_FAR);

      const presence = band.mul(onIsland).mul(handover);

      const size = float(1)
        .sub(CARD_VARIATION)
        .add(seed.z.mul(CARD_VARIATION * 2))
        .mul(presence);

      // --- billboard ---------------------------------------------------------
      // Cylindrical, not spherical: the card spins about world up to face the
      // camera and never tips. A spherical billboard is correct for a particle
      // and wrong for a tree — fly over a spherical one and the canopy lies down
      // to look at you, which is the single most obvious way an impostor field
      // gives itself away.
      const flat = normalize(vec3n(toCamera.x, 0, toCamera.z).add(vec3(1e-4, 0, 0))) as Node;
      const right = normalize(vec3n(flat.z.negate(), 0, flat.x)) as Node;
      cardRight.assign(right);
      cardFacing.assign(flat.negate());

      // Proportions, from a hash of the two draws the instance already carries.
      // A third random per card would mean a wider instance buffer for one
      // scalar; the seed's own components are independent, so a hash of them is
      // as good a draw and costs nothing per instance.
      const aspect = fract(sin(seed.z.mul(74.7).add(seed.w.mul(219.3))).mul(21942.31))
        .sub(0.5)
        .mul(2 * CARD_ASPECT_VARIATION)
        .add(1);
      const width = size.mul(CARD_WIDTH).mul(aspect);
      const height = size.mul(CARD_HEIGHT).div(aspect);

      // Sway, as a shear at the top of the card. One travelling wave across the
      // island so neighbouring clumps move together — a canopy moves in gusts,
      // and per-instance random phase reads as static.
      const along = foot.dot(this.uWind);
      const gust = along
        .mul(SWAY_K)
        .sub(this.uPhase.mul(SWAY_OMEGA))
        .add(seed.w.mul(6.28))
        .sin();
      const lean = gust.mul(SWAY_MAX).mul(this.uWindStrength).mul(corner.y).mul(corner.y);

      const world = vec3n(foot.x, ground, foot.y)
        .add(right.mul(corner.x.mul(width)))
        .add(vec3n(0, corner.y.mul(height), 0))
        .add(vec3n(this.uWind.x.mul(lean), 0, this.uWind.y.mul(lean)));

      // Published for the fragment stage: crowns are lit from above, so the
      // bottom of a card is the shaded underside of the clump and the top is
      // what the sun reaches. This one term is most of why a flat card reads as
      // a volume.
      shade.assign(corner.y);
      // A small fraction of clumps carry the flame trees' colour, drawn from the
      // same seed the sway phase uses. `Props` plants twenty real ones; this is
      // what keeps them present in the mass rather than only where a mesh
      // happens to stand.
      bloomMix.assign(seed.w.smoothstep(0.978, 0.998).mul(0.7));

      // Stand tone: a broad field across the island, plus a per-crown offset.
      //
      // The hash is a sine hash, which `Seafloor` deliberately refuses to use —
      // and the distinction is real rather than inconsistent. There the noise has
      // to agree between a float64 CPU evaluation and a float32 GPU one, because
      // buoyancy and prop seating read the same field the mesh is built from.
      // Nothing on the CPU has an opinion about what colour a card is.
      const field = mx_fractal_noise_float(
        vec3n(foot.x, 0, foot.y).mul(1 / STAND_FIELD_METRES),
        3,
        2.0,
        0.5,
        1.0,
      )
        .mul(0.5)
        .add(0.5);
      const crown = fract(sin(seed.z.mul(127.1).add(seed.w.mul(311.7))).mul(43758.5453));
      // Two terrain taps approximate the local slope normal. The aspect term is
      // deliberately only a quarter of the stand tone: exposure should bias
      // CANOPY_DRY toward sun-facing spurs, not turn every ridge into a stripe.
      const eastGround = this.groundHeight(
        vec3n(foot.x.add(SLOPE_SAMPLE_METRES), 0, foot.y),
      ) as Node;
      const northGround = this.groundHeight(
        vec3n(foot.x, 0, foot.y.add(SLOPE_SAMPLE_METRES)),
      ) as Node;
      const slopeNormal = normalize(
        vec3n(ground.sub(eastGround), SLOPE_SAMPLE_METRES, ground.sub(northGround)),
      ) as Node;
      const slopeExposure = slopeNormal.dot(this.uSunDir).clamp(0, 1);
      const organicStand = field.mul(STAND_FIELD_WEIGHT).add(crown.mul(STAND_INSTANCE_WEIGHT));
      stand.assign(
        organicStand
          .mul(1 - STAND_SLOPE_WEIGHT)
          .add(slopeExposure.mul(STAND_SLOPE_WEIGHT))
          .clamp(0, 1),
      );

      // Cloud shade and the hill's own shadow, sampled at the foot and crown.
      // Interpolating the two taps gives a free vertical AO gradient across the
      // four vertices of a card: gullies are darker at their base than their
      // crown, while the fragment stage pays no heightfield march.
      const footKey =
        this.sunOcclusion === null
          ? (float(1) as Node)
          : this.sunOcclusion(vec3n(foot.x, ground, foot.y));
      const crownKey =
        this.sunOcclusion === null
          ? (float(1) as Node)
          : this.sunOcclusion(vec3n(foot.x, ground.add(CARD_HEIGHT * 0.85), foot.y));
      keyLight.assign(
        mix(footKey, crownKey, corner.y),
      );

      return world;
    })();

    material.colorNode = Fn(() => {
      const uv = attribute('uv', 'vec2') as Node;

      // --- the card's own shape ---------------------------------------------
      // A blob with a broken edge, built from the quad's uv rather than sampled.
      // The radial term gives the crown; the two sine lobes bite chunks out of
      // it at different frequencies so no two cards share a silhouette once the
      // instance phase is added, and the edge is ragged rather than round.
      const seedW = (attribute('cardSeed', 'vec4') as Node).w as Node;
      const seedZ = (attribute('cardSeed', 'vec4') as Node).z as Node;

      // Three outline families keep the field from repeating one ellipse. The
      // bands blend at their boundaries so deterministic seeds do not pop when
      // a card crosses a threshold: broad round, tall narrow, then flat topped.
      const tallOrFlat = seedZ.smoothstep(0.3, 0.7);
      const flatFamily = seedZ.smoothstep(0.68, 0.78);
      const roundFamily = float(1).sub(tallOrFlat);
      const tallFamily = tallOrFlat.mul(float(1).sub(flatFamily));
      const verticalCentre = roundFamily
        .mul(0.45)
        .add(tallFamily.mul(0.54))
        .add(flatFamily.mul(0.39));
      const horizontalScale = roundFamily
        .mul(1)
        .add(tallFamily.mul(1.35))
        .add(flatFamily.mul(0.92));
      const verticalScale = roundFamily
        .mul(1.15)
        .add(tallFamily.mul(0.98))
        .add(flatFamily.mul(1.05));
      const centred = vec2n(uv.x.sub(0.5), uv.y.sub(verticalCentre));
      const radial = vec2n(centred.x.mul(horizontalScale), centred.y.mul(verticalScale));
      const roundedRadius = radial.length();
      const flatRadius = radial.x
        .abs()
        .pow(4)
        .add(radial.y.abs().pow(4))
        .pow(0.25);
      const radius = roundedRadius
        .mul(float(1).sub(flatFamily))
        .add(flatRadius.mul(flatFamily));
      // The bite is driven by the *direction* around the card rather than by an
      // angle, which keeps `atan2` out of the shader — its branch cut would draw
      // a seam down one side of every card, and TSL has moved its spelling
      // across recent revisions. Products of the unit components are the same
      // harmonics an angle would have given, by the Chebyshev identities.
      const dir = normalize(vec2n(centred.x, centred.y).add(vec2(1e-4, 0)));
      const h2 = dir.x.mul(dir.x).mul(2).sub(1);
      const h3 = dir.x.mul(h2).mul(2).sub(dir.x);
      const h2Weight = roundFamily
        .mul(0.05)
        .add(tallFamily.mul(0.035))
        .add(flatFamily.mul(0.065));
      const h3Weight = roundFamily
        .mul(0.035)
        .add(tallFamily.mul(0.022))
        .add(flatFamily.mul(0.018));
      const verticalWeight = roundFamily
        .mul(0.025)
        .add(tallFamily.mul(0.04))
        .add(flatFamily.mul(0.012));
      const ragged = h2
        .mul(seedW.mul(6.28).sin())
        .mul(h2Weight)
        .add(h3.mul(seedW.mul(11.2).cos()).mul(h3Weight))
        .add(dir.y.mul(seedW.mul(3.1).sin()).mul(verticalWeight));

      // Low and Medium keep the analytic harmonic edge, but skip the two
      // fractal-noise samplers entirely. This graph is selected in JavaScript
      // at material-build time, so the cheap path does not evaluate or carry
      // dead noise work behind a runtime uniform branch.
      let raggedEdge: Node = float(0);
      let interiorHoles: Node = float(0);
      if (detail >= CANOPY_DETAIL_FULL) {
        // One two-octave field does double duty: it perturbs the rim only where
        // the edge is visible, and the same deterministic field opens a few holes
        // in the mass. Hole weight falls to zero at the rim, so the silhouette
        // stays broken without turning into frayed noise.
        const silhouetteNoise = mx_fractal_noise_float(
          vec3n(
            centred.x.mul(5.5).add(seedW.mul(1.7)),
            centred.y.mul(5.5).add(seedW.mul(3.1)),
            seedZ.mul(2.3).add(seedW.mul(7.9)),
          ),
          2,
          2.0,
          0.5,
        )
          .mul(0.5)
          .add(0.5);
        const edgeWeight = radius.smoothstep(0.16, 0.43);
        raggedEdge = silhouetteNoise.sub(0.5).mul(0.12).mul(edgeWeight);
        const holeWeight = float(1).sub(radius.smoothstep(0.18, 0.42));
        interiorHoles = silhouetteNoise.smoothstep(0.66, 0.86).mul(holeWeight);
      }

      // Alpha is a hard-ish edge because it is tested, not blended: a soft ramp
      // through an alpha test is a hard edge anyway, just one whose position
      // moves with the ramp. Alpha-to-coverage turns that stable depth edge into
      // a smoother MSAA transition without giving up the tested ordering.
      const alpha = float(1)
        .sub(radius.add(ragged).add(raggedEdge).smoothstep(0.3, 0.44))
        .mul(float(1).sub(interiorHoles.mul(0.95)));

      // --- shading ------------------------------------------------------------
      // A shallow ramp, and the shallowness is the point. Driving the card from
      // fully shaded at its foot to fully lit at its top is physically the right
      // idea and reads as a *ball*: every clump gets its own strong terminator.
      // Keep the dome shallow enough that the hillside does not become a tray of
      // broccoli; DOME_BULGE and CANOPY_WRAP are the two knobs that hold that
      // line while still letting the sun move across the canopy.
      // Real forest at this range has very little per-crown shading left in it —
      // what survives a kilometre of air is the mass, not the modelling.
      const lit = mix(this.uShade, this.uSunlit, shade.smoothstep(-0.45, 1.15));
      const flowering = mix(lit, this.uBloom, bloomMix.mul(shade.smoothstep(0.35, 1)));
      // Stand variation, in hue and in value. The hue term is weighted toward the
      // dry end so most of the forest stays green and the browner stands read as
      // exposure rather than as a second species; the value term is symmetric
      // about 1, so it varies the canopy without changing how bright it is on
      // average — which matters, because this field is most of the island's
      // luminance at range.
      const varied = mix(flowering, this.uDry, stand.smoothstep(0.5, 0.95).mul(0.7));
      const value = stand.sub(0.5).mul(STAND_VALUE_RANGE).add(1);

      // Build a hemisphere from centred uv and the published billboard basis.
      // The old seam warning was about an angle-derived normal: the dir and
      // harmonic block above is the atan2 branch-cut hazard (:538-545 in the
      // pre-change file). This hemisphere has no angle or seam.
      //
      // `keyLight` is the cloud deck and the hillside between the card and the
      // sun. Without it the forest went on being fully lit across a hill whose
      // own shading now has a lit face and a shaded one — a canopy floating over
      // its own shadow, which reads worse than no shadow at all.
      const domeUv = vec2n(centred.x.mul(2), centred.y.mul(2));
      const domeDepth = float(1).sub(domeUv.dot(domeUv)).max(0).sqrt();
      const canopyNormal = cardRight
        .mul(domeUv.x.mul(DOME_BULGE))
        .add(vec3(0, 1, 0).mul(domeUv.y.mul(DOME_BULGE)))
        .add(cardFacing.mul(domeDepth))
        .add(vec3(0, 1, 0).mul(CANOPY_UP_BIAS))
        .normalize();
      const wrapped = canopyNormal
        .dot(this.uSunDir)
        .add(CANOPY_WRAP)
        .div(1 + CANOPY_WRAP)
        .clamp(0, 1);
      const elevation = this.uSunDir.y.max(0).smoothstep(0, 0.4);
      const sun = wrapped.mul(elevation);

      // Back-scatter uses the same constants as mesh foliage, so a card
      // standing in for a tree glows by the same narrow, shadowed lobe.
      const backScatter = cardFacing
        .dot(this.uSunDir.negate())
        .clamp(0, 1)
        .pow(TRANSLUCENCY_POWER)
        .mul(TRANSLUCENCY_GAIN);
      const light = this.uAmbient
        .mul(0.55)
        .add(this.uSunColor.mul(sun.mul(0.8)).mul(keyLight));
      const litWithScatter = light.add(this.uSunColor.mul(backScatter).mul(keyLight));

      return vec4(varied.mul(litWithScatter).mul(value), alpha);
    })();

    return material;
  }
}
