import * as THREE from 'three/webgpu';
import { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';
import { positionWorld, shadowPositionWorld, vec3 } from 'three/tsl';
import { uniform } from '../core/uniform';

/**
 * Cascaded shadow maps for the key light.
 *
 * ### Why this replaced the single box
 *
 * The sun used to cast into one 520 m orthographic box centred on the viewer.
 * At 2048² that is 25 cm a texel *everywhere*, including the deck the viewer is
 * standing on, so the rigging — ropes of a few centimetres, spars of twenty —
 * cast nothing at all, and a mast cast a soft grey smear. And the bias that box
 * needed was worse than the resolution: `bias = -0.0006` over a 6 km depth range
 * is 3.6 m of depth offset, and `normalBias = 0.6` is 60 cm of world offset.
 * Anything within 3.6 m of its receiver along the light — a crate on the deck,
 * the hull against the water, a barrel on the sand — had no contact shadow, and
 * thin casters fell inside the normal offset entirely.
 *
 * Every shipped open-world renderer solves this the same way: split the view
 * frustum by distance and give each slice its own map, so texel density follows
 * screen density. three ships that as `CSMShadowNode`; this is that node with the
 * project's constraints applied:
 *
 * - **Stable fit.** The node sizes each cascade from the slice's bounding
 *   diameter, not its tight bounds, and snaps the origin to whole texels in light
 *   space — so the maps do not change size as the camera turns and do not crawl
 *   as it moves. That is the "no swimming" property, and it is the node's own.
 * - **Biases in metres, per cascade.** Both biases are set from each cascade's
 *   actual texel size after the fit, so the near cascade gets centimetres and the
 *   far one decimetres. A single number cannot be right for both.
 * - **Cascade count fixed at boot.** It is part of every lit material's node
 *   graph; changing it means recompiling all of them, which is the in-session
 *   compile this project rules out. A tier change moves map size and distance,
 *   both of which three reads live (`mapSize` by reference in the filter, the
 *   split distances from the node's own uniforms).
 * - **Bound to the scene camera before anything compiles.** The node otherwise
 *   adopts whatever camera first builds a material, and the planar reflector's
 *   mirrored camera can be that camera.
 */

export interface SunShadowOptions {
  cascades: number;
  /** Metres from the eye that the last cascade reaches. */
  distance: number;
  mapSize: number;
}

/**
 * Where each split falls, as a fraction of the shadow distance.
 *
 * Steeper than three's "practical" split at this range. The view near plane is
 * 0.1 m, so a logarithmic split puts the first break at under two metres — a
 * cascade spent on the inside of the lens. These put the first break where the
 * deck ends in the helm view and the second where the ship ends in a wide shot.
 */
const SPLITS: Record<number, number[]> = {
  1: [1],
  2: [0.16, 1],
  3: [0.08, 0.3, 1],
  4: [0.05, 0.15, 0.4, 1],
};

/**
 * Light-space depth kept above each cascade's box for casters outside the view.
 *
 * The island's summit stands 150 m and its spires another 60 above that; a sun
 * low over the island throws those shadows a long way across the slice. 400 m is
 * the margin, and with it the shadow camera's depth range only needs to span
 * margin plus the deepest slice — far tighter than the 6 km the box used, which
 * is what turns a depth bias into a sensible number of metres.
 */
const LIGHT_MARGIN = 400;
const SHADOW_FAR = 1600;

/** Constant depth bias, in texels of the cascade it is applied to. */
const DEPTH_BIAS_TEXELS = 1.2;
/** Receiver offset along the normal, in texels. One and a half is the usual figure for PCF 3x3. */
const NORMAL_BIAS_TEXELS = 1.5;

/**
 * Makes every lit material use one copy of the node's shading graph.
 *
 * Without this, `CSMShadowNode.setup` runs once per material and builds the
 * graph afresh each time. Each copy reads the split distances through a new
 * `reference('_cascades')`, which is a new uniform-array node with a new id.
 * The GLSL builder named that block `NodeBuffer_<id>`, so on the since-retired
 * WebGL2 fallback every lit material compiled a program that differed from its
 * neighbours by that name alone and could not share one with them.
 *
 * Sharing is how three treats every other shadow already. `AnalyticLightNode`
 * builds its shadow node once and hands it to every material, and the
 * per-cascade `shadow()` nodes this graph reads are shared the same way. The
 * per-build part, the shadow position, runs inside the graph's `Fn` and so
 * still runs for each material. Safe to fix once because the graph depends
 * only on `cascades` and `fade`, both set before anything compiles and never
 * changed afterwards (see the class notes).
 */
function shareOneGraph(node: CSMShadowNode): void {
  const target = node as unknown as { setup(builder: unknown): unknown };
  const setup = target.setup.bind(target);
  let graph: unknown = null;
  target.setup = (builder) => (graph ??= setup(builder));
}

/**
 * What the underwater shafts need to read one cascade from a post pass: its
 * depth texture (null until the first shadow render has created it), the
 * world-to-shadow-texture matrix three keeps current on `shadow.matrix`, and
 * the depth bias that matrix is used with.
 */
export interface ShadowVolumeSource {
  depthTexture(): THREE.DepthTexture | null;
  readonly matrix: THREE.Matrix4;
  bias(): number;
  /** Receiver offset along the normal, metres (one and a half texels). */
  normalBias(): number;
}

/** Refractive index of sea water. */
const WATER_IOR = 1.333;

/**
 * The cascade the shafts read is the first whose slice reaches this far.
 *
 * Shafts are integrated over at most ~120 m and the medium has taken them to a
 * few percent by 60, so the cascade that matters is the one covering roughly
 * the first hundred metres: at High that is cascade 1 (24-90 m slice, box ~220 m
 * wide, 0.11 m a texel) rather than the far one (734 m, 0.36 m). Points the
 * chosen box does not contain read as lit.
 */
const VOLUME_REACH = 80;

/**
 * Re-points the shadow lookup of submerged receivers along the refracted sun.
 *
 * The map is rendered along the sun's direction *in air*. Under water the light
 * that reaches a point arrived along the refracted direction, which for
 * `skyPro`'s 24-degree sun is 47 degrees — so a hull's shadow on sand 17 m down
 * landed 21 m from where the refracted caustics and shafts put it. The lookup
 * point is moved to the place at the same depth on the *air* ray through the
 * refracted ray's surface entry point: everything above the surface (the hull,
 * the island) is then tested along exactly the right path, and the short
 * submerged leg of the path is the only part still taken at the air angle.
 *
 * `max(0, -y)` is exactly zero above the sea, so every receiver above it keeps
 * its lookup bit for bit — the topside image does not change.
 */
function refractShadowLookup(node: CSMShadowNode, shift: unknown): void {
  const patch = (target: { setupShadowPosition(builder: any): void }): void => {
    target.setupShadowPosition = ({ context, material }: any) => {
      const base: any = material.receivedShadowPositionNode || context.shadowPositionWorld || positionWorld;
      const depth: any = base.y.negate().max(0);
      const s: any = shift;
      (shadowPositionWorld as any).assign(vec3(base.x.add(s.x.mul(depth)), base.y, base.z.add(s.y.mul(depth))));
    };
  };
  patch(node as unknown as { setupShadowPosition(builder: any): void });
  // Each cascade's own shadow node assigns the position again when the CSM
  // graph pulls it in, so the override has to be on all of them.
  for (const cascade of (node as unknown as { _shadowNodes: Array<{ setupShadowPosition(b: any): void }> })._shadowNodes) {
    patch(cascade);
  }
}

export class SunShadows {
  readonly node: CSMShadowNode;
  /**
   * Horizontal offset per metre of depth that moves a submerged lookup onto
   * the refracted light path. See `refractShadowLookup`.
   */
  private readonly uRefractShift = uniform(new THREE.Vector2(0, 0));
  /** Index of the cascade `volumeSource` reads. */
  private readonly volumeCascade: number;
  private readonly light: THREE.DirectionalLight;
  private readonly camera: THREE.PerspectiveCamera;
  private distance: number;
  private mapSize: number;
  /** Projection inputs at the last fit; the splits are rebuilt when any moves. */
  private fov = -1;
  private aspect = -1;
  private near = -1;
  private zoom = -1;

  constructor(
    light: THREE.DirectionalLight,
    camera: THREE.PerspectiveCamera,
    renderer: THREE.WebGPURenderer,
    options: SunShadowOptions,
  ) {
    this.light = light;
    this.camera = camera;
    this.distance = options.distance;
    this.mapSize = options.mapSize;

    const cascades = Math.max(1, Math.min(4, Math.round(options.cascades)));
    const splits = SPLITS[cascades];

    // The cascades clone the light's shadow, so its camera range and map size
    // are set first.
    light.shadow.camera.near = 1;
    light.shadow.camera.far = SHADOW_FAR;
    light.shadow.mapSize.set(options.mapSize, options.mapSize);
    light.shadow.camera.updateProjectionMatrix();

    this.node = new CSMShadowNode(light, {
      cascades,
      maxFar: options.distance,
      mode: 'custom',
      lightMargin: LIGHT_MARGIN,
      customSplitsCallback: (count: number, _near: number, _far: number, breaks: number[]) => {
        for (let i = 0; i < count; i++) breaks.push(splits[i]);
      },
    });
    // Blend across each split so the step in resolution is not a visible line
    // that slides across the deck as the camera dollies.
    this.node.fade = true;

    // Pre-bind the scene camera. `_init` is what `setup` would call with the
    // camera of whichever render first compiles a lit material.
    (this.node as unknown as { _init(args: { camera: THREE.Camera; renderer: THREE.WebGPURenderer }): void })._init({
      camera,
      renderer,
    });
    refractShadowLookup(this.node, this.uRefractShift);
    shareOneGraph(this.node);
    light.shadow.shadowNode = this.node;
    let volume = cascades - 1;
    for (let i = 0; i < cascades; i++) {
      if (splits[i] * options.distance >= VOLUME_REACH) {
        volume = i;
        break;
      }
    }
    this.volumeCascade = volume;
    this.refit();
  }

  /**
   * The direction *toward* the key light, in air. Feeds the refracted lookup;
   * a uniform write, no rebuild.
   */
  setLightDirection(dir: THREE.Vector3): void {
    const len = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const ax = dir.x / len;
    const ay = Math.max(0.05, dir.y / len);
    const az = dir.z / len;
    // Refracted: horizontal component over the index, renormalised.
    const rx = ax / WATER_IOR;
    const rz = az / WATER_IOR;
    const ry = Math.sqrt(Math.max(1e-4, 1 - rx * rx - rz * rz));
    // Clamped so a sun on the horizon cannot throw the lookup out of the box.
    const clamp = (v: number): number => Math.max(-4, Math.min(4, v));
    (this.uRefractShift.value as THREE.Vector2).set(clamp(rx / ry - ax / ay), clamp(rz / ry - az / ay));
  }

  /**
   * One cascade, for reading from a post pass — the underwater shafts. The
   * cascade count is fixed at boot, so the choice is too; the texture object
   * is created by the first shadow render and survives a tier change, which
   * only resizes it.
   */
  volumeSource(): ShadowVolumeSource {
    return this.cascadeSource(this.volumeCascade);
  }

  /**
   * The near cascades, nearest first, for the contact shadows' "is this pixel
   * in the sun at all" test (`ContactShadows`). Two at most: past the second
   * split a contact shadow is under a pixel anyway.
   */
  nearSources(): ShadowVolumeSource[] {
    const count = Math.min(2, this.node.lights.length);
    return Array.from({ length: count }, (_, i) => this.cascadeSource(i));
  }

  private cascadeSource(index: number): ShadowVolumeSource {
    const shadow = this.node.lights[index].shadow!;
    return {
      depthTexture: () => (shadow.map?.depthTexture as THREE.DepthTexture | null | undefined) ?? null,
      matrix: shadow.matrix,
      bias: () => shadow.bias,
      normalBias: () => shadow.normalBias,
    };
  }

  /** Per metre of depth; exposed for the shafts, which apply the same shift. */
  get refractShift(): THREE.Vector2 {
    return this.uRefractShift.value as THREE.Vector2;
  }

  /** The tier's map size and reach. Map size is read live; no recompile. */
  configure(distance: number, mapSize: number): void {
    const side = Math.max(256, Math.round(mapSize));
    if (distance === this.distance && side === this.mapSize) return;
    this.distance = distance;
    this.mapSize = side;
    this.light.shadow.mapSize.set(side, side);
    for (const cascade of this.node.lights) cascade.shadow?.mapSize.set(side, side);
    this.node.maxFar = distance;
    this.refit();
  }

  /** Re-fits the splits if the lens changed. Call once a frame, before rendering. */
  update(): void {
    const c = this.camera;
    if (c.fov !== this.fov || c.aspect !== this.aspect || c.near !== this.near || c.zoom !== this.zoom) this.refit();
  }

  /**
   * Whether `target` is one of the cascades' shadow maps. The sun's own
   * `shadow.map` stays null once this node renders its shadows (each cascade
   * renders a clone), so this is how a caller recognises a shadow render.
   */
  ownsTarget(target: unknown): boolean {
    for (const cascade of this.node.lights) if (cascade.shadow?.map === target) return true;
    return false;
  }

  get cascades(): number {
    return this.node.cascades;
  }

  /** World size of one texel in each cascade, metres. For the debug hooks. */
  texelSizes(): number[] {
    return this.node.lights.map((l) => {
      const cam = l.shadow!.camera;
      return (cam.right - cam.left) / l.shadow!.mapSize.x;
    });
  }

  private refit(): void {
    const c = this.camera;
    this.fov = c.fov;
    this.aspect = c.aspect;
    this.near = c.near;
    this.zoom = c.zoom;
    this.node.updateFrustums();
    // Biases from each cascade's own texel, in metres. The depth bias is a
    // fraction of the shadow camera's range, which is why that range is tight.
    const range = SHADOW_FAR - 1;
    for (const cascade of this.node.lights) {
      const shadow = cascade.shadow!;
      const cam = shadow.camera;
      const texel = (cam.right - cam.left) / shadow.mapSize.x;
      shadow.normalBias = texel * NORMAL_BIAS_TEXELS;
      shadow.bias = -(texel * DEPTH_BIAS_TEXELS) / range;
      cam.near = 1;
      cam.far = SHADOW_FAR;
      cam.updateProjectionMatrix();
    }
  }
}
