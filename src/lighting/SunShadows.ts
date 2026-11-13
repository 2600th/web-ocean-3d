import * as THREE from 'three/webgpu';
import { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';

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

export class SunShadows {
  readonly node: CSMShadowNode;
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
    light.shadow.shadowNode = this.node;
    this.refit();
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
