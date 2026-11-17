import * as THREE from 'three/webgpu';
import SSSNode from 'three/addons/tsl/display/SSSNode.js';
import { float, getViewPosition, texture, vec3, vec4 } from 'three/tsl';
import { uniform } from '../core/uniform';
import type { ShadowVolumeSource } from '../lighting/SunShadows';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * three's screen-space shadow march with the two changes `SequencedGTAO` makes
 * to GTAO, for the same reasons: its noise phase counts *renders* (which a
 * capture controls) instead of three's rAF frame counter, and it can be
 * switched off without leaving the graph, which a live tier change needs.
 */
class SequencedSSS extends (SSSNode as unknown as new (...args: any[]) => any) {
  enabled = true;
  private sequence = 0;

  constructor(depth: any, camera: THREE.Camera, light: THREE.DirectionalLight) {
    super(depth, camera, light);
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
    // Four ray offsets in rotation; the temporal resolve integrates them.
    this.useTemporalFiltering = true;
  }

  updateBefore(frame: any): any {
    if (!this.enabled) return undefined;
    return (SSSNode.prototype as any).updateBefore.call(this, {
      renderer: frame.renderer,
      frameId: this.sequence++,
    });
  }

  resetSequence(): void {
    this.sequence = 0;
  }
}

export interface ContactShadowSettings {
  enabled: boolean;
  /** Ray length, metres. */
  maxDistance: number;
  /** Fraction of the ray's pixel length that is stepped, 0..1. */
  quality: number;
  /** Fraction of the drawing buffer the march runs at; follows the scene pass. */
  resolutionScale: number;
}

/**
 * Hemisphere fill to total ambient irradiance. The scene's materials take the
 * environment cube at 0.6-1.2 on top of the hemisphere light, and both stand
 * in for the same dome, so the ambient a surface sees is a little under twice
 * the hemisphere term. A calibration for the sun-share estimate below, not a
 * look control: it moves how much of a lit surface the contact shadow may
 * take away, never whether a shadowed one is touched.
 */
const AMBIENT_SCALE = 1.8;

/**
 * Screen-space contact shadows from the sun (`SSSNode`), taken out of the
 * sun's share of each pixel rather than out of the whole pixel.
 *
 * ### Why post, and why a share
 *
 * The cascades are 2.5 cm a texel at best on High, so what they lose is
 * exactly what this finds: a coil of rope on the deck, a rock settled into the
 * sand, the strakes along the waterline. The physically right place to apply
 * it is the sun's own shadow term inside every lit material — three's
 * `builtinShadowContext`, which is how its example does it. That needs the
 * contact shadows *before* the scene draws, so a depth pre-pass of the whole
 * scene (ocean grid, canopy, meadow) every frame, and the context reaches
 * every render nested in the pass: the planar reflection would read a
 * screen-space texture at the wrong coordinates. It is the same trade
 * `AmbientOcclusion` describes, decided the same way.
 *
 * A plain multiply would darken ambient too: a crate standing in a sail's
 * shadow would grow a second, darker shadow inside the first. So the
 * composite removes only what the sun put there. For a diffuse surface the
 * sun's share of the outgoing light does not depend on its albedo —
 * `E_sun·N·L·V / (E_sun·N·L·V + E_ambient)`, with the albedo cancelling — and
 * every term is at hand in a post pass: N from the scene pass's normal target,
 * L and both irradiances from the lights, and V, whether the cascades already
 * have the pixel in shadow, from the two near cascades looked up at the
 * pixel's world position. Where the cascades say shadow, or the surface faces
 * away, the share is zero and nothing is darkened twice. Specular is left
 * alone; on these materials it is a small part of the sun's term.
 *
 * Sky has no depth and the march skips it (its target clears to 1); the sea
 * is masked by the caller with the water flag, because its shading models its
 * own self-shadowing.
 */
export class ContactShadows {
  private readonly node: SequencedSSS;
  private readonly uStrength = uniform(1);
  /** Sun irradiance over ambient irradiance, luminance; see `update`. */
  private readonly uSunRatio = uniform(3);
  /** Direction toward the key light, view space; refreshed per render. */
  private readonly uSunView: any;
  private readonly uInvProjection: any;
  private readonly uCameraWorld: any;
  private settings: ContactShadowSettings = { enabled: true, maxDistance: 0.35, quality: 0.5, resolutionScale: 1 };

  constructor(
    private readonly depth: any,
    camera: THREE.PerspectiveCamera,
    private readonly sun: THREE.DirectionalLight,
    private readonly ambient: THREE.HemisphereLight,
    private readonly cascades: ShadowVolumeSource[],
  ) {
    this.node = new SequencedSSS(depth, camera, sun);
    // Rope and spar are a few centimetres; anything thicker than this along
    // the view ray would count a silhouette's far side as an occluder and
    // throw a halo behind every edge.
    this.node.thickness.value = 0.08;
    this.node.shadowIntensity.value = 1;
    const sunView = new THREE.Vector3();
    this.uSunView = uniform(sunView).onRenderUpdate(() => {
      sunView.setFromMatrixPosition(sun.matrixWorld);
      sunView.sub(_target.setFromMatrixPosition(sun.target.matrixWorld)).normalize();
      return sunView.transformDirection(camera.matrixWorldInverse);
    });
    // Object references: read at render time, so inside the pipeline render
    // they carry the temporal jitter the depth was drawn with.
    this.uInvProjection = uniform(camera.projectionMatrixInverse);
    this.uCameraWorld = uniform(camera.matrixWorld);
  }

  /**
   * Multiplier for the scene colour at `p`, 1 where nothing is in contact.
   *
   * @param p Screen UV.
   * @param normalView The scene pass's view-space normal target.
   */
  factor(p: any, normalView: any): any {
    const shadow = (this.node.getTextureNode() as any).sample(p).r;
    const n = normalView.sample(p).xyz.normalize().toVar('cssN');
    const nDotL = n.dot(this.uSunView).max(0);
    const view = getViewPosition(p, this.depth.sample(p).r, this.uInvProjection).toVar('cssView');
    const world = this.uCameraWorld.mul(vec4(view, 1)).xyz;
    const normalWorld = this.uCameraWorld.mul(vec4(n, 0)).xyz;
    const visible = this.cascadeVisibility(world, normalWorld);
    const sunIrradiance = this.uSunRatio.mul(nDotL).mul(visible);
    const share = sunIrradiance.div(sunIrradiance.add(1));
    return float(1).sub(share.mul(float(1).sub(shadow)).mul(this.uStrength));
  }

  /**
   * Whether the near cascades have `world` in the sun, 0..1: the nearest that
   * contains it decides, and a point outside both counts as lit. One hardware
   * compare per cascade (a 2x2 PCF), offset along the normal by the cascade's
   * own normal bias so a lit surface does not shadow itself.
   */
  private cascadeVisibility(world: any, normalWorld: any): any {
    let result: any = float(1);
    // Built far to near, so the nearest cascade's test wraps the others.
    for (let i = this.cascades.length - 1; i >= 0; i--) {
      const source = this.cascades[i];
      const depthTexture = source.depthTexture();
      if (depthTexture === null) continue;
      const matrix = uniform(new THREE.Matrix4()).onRenderUpdate(() => source.matrix);
      const bias = uniform(0).onRenderUpdate(() => source.bias());
      const normalBias = uniform(0).onRenderUpdate(() => source.normalBias());
      const q = world.add(normalWorld.mul(normalBias));
      // Orthographic: w is 1. The y flip and the additive bias are
      // `ShadowNode.setupShadowCoord`'s, as in `UnderwaterPass.shadowVisibility`.
      const clip = matrix.mul(vec4(q, 1)).toVar();
      const coord = vec3(clip.x, clip.y.oneMinus(), clip.z.add(bias)).toVar();
      const inBox = coord.x
        .greaterThanEqual(0)
        .and(coord.x.lessThanEqual(1))
        .and(coord.y.greaterThanEqual(0))
        .and(coord.y.lessThanEqual(1))
        .and(coord.z.lessThanEqual(1));
      const lit = (texture(depthTexture, coord.xy) as any).compare(coord.z);
      result = inBox.select(lit, result);
    }
    return result;
  }

  apply(settings: ContactShadowSettings): void {
    this.settings = settings;
    this.node.enabled = settings.enabled;
    this.node.maxDistance.value = settings.maxDistance;
    this.node.quality.value = settings.quality;
    this.node.resolutionScale = settings.resolutionScale;
    this.uStrength.value = settings.enabled ? 1 : 0;
  }

  /**
   * The sun-to-ambient irradiance ratio, from the lights as they are now.
   * Per frame: the sun sets, clouds cover it and the moon takes over the same
   * light, and the share has to follow all three.
   */
  update(): void {
    const sun = this.sun.intensity * luminance(this.sun.color);
    _fill.copy(this.ambient.color).lerp(this.ambient.groundColor, 0.5);
    const ambient = Math.max(1e-3, this.ambient.intensity * luminance(_fill) * AMBIENT_SCALE);
    this.uSunRatio.value = sun / ambient;
  }

  get enabled(): boolean {
    return this.settings.enabled;
  }

  get texture(): any {
    return this.node.getTextureNode();
  }

  resetSequence(): void {
    this.node.resetSequence();
  }

  dispose(): void {
    this.node.dispose();
  }
}

const _target = new THREE.Vector3();

/** Rec. 709 luminance of a linear colour. */
function luminance(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}
const _fill = new THREE.Color();
