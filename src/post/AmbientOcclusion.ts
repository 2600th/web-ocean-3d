import * as THREE from 'three/webgpu';
import GTAONode from 'three/addons/tsl/display/GTAONode.js';
import { Fn, float, floor, mix, perspectiveDepthToViewZ, reference, smoothstep, uniform, uv, vec2, vec4 } from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Catmull-Rom sample of `tex` at `p`, in nine bilinear taps (the separable
 * weights folded so the middle two texels share a tap).
 *
 * For moving the scene colour back off TRAA's jitter. A plain bilinear read
 * does that too, but it is a tent filter laid over every frame, and the
 * accumulated image softened: the steepest step across the horizon fell 14%
 * (0.79 to 0.68 in `corr.mjs`'s log-luminance units, camera still). The
 * negative lobes can undershoot below zero next to a bright edge, so the
 * result is clamped; TRAA's neighbourhood clip takes care of any overshoot.
 */
function sampleCatmullRom(tex: any, p: any): any {
  const size = vec2(tex.size());
  const pos = p.mul(size);
  const t1 = floor(pos.sub(0.5)).add(0.5);
  const f = pos.sub(t1);
  const w0 = f.mul(f.mul(f.mul(-0.5).add(1)).sub(0.5));
  const w1 = f.mul(f).mul(f.mul(1.5).sub(2.5)).add(1);
  const w2 = f.mul(f.mul(f.mul(-1.5).add(2)).add(0.5));
  const w3 = f.mul(f).mul(f.mul(0.5).sub(0.5));
  const w12 = w1.add(w2);
  const t0 = t1.sub(1).div(size);
  const t3 = t1.add(2).div(size);
  const t12 = t1.add(w2.div(w12)).div(size);
  const tap = (x: any, y: any, w: any): any => tex.sample(vec2(x, y)).mul(w);
  return tap(t0.x, t0.y, w0.x.mul(w0.y))
    .add(tap(t12.x, t0.y, w12.x.mul(w0.y)))
    .add(tap(t3.x, t0.y, w3.x.mul(w0.y)))
    .add(tap(t0.x, t12.y, w0.x.mul(w12.y)))
    .add(tap(t12.x, t12.y, w12.x.mul(w12.y)))
    .add(tap(t3.x, t12.y, w3.x.mul(w12.y)))
    .add(tap(t0.x, t3.y, w0.x.mul(w3.y)))
    .add(tap(t12.x, t3.y, w12.x.mul(w3.y)))
    .add(tap(t3.x, t3.y, w3.x.mul(w3.y)))
    .max(0);
}

/**
 * GTAO from three's own node, with two things the stock node cannot do here.
 *
 * **Its temporal sequence is ours, not the frame counter's.** `GTAONode` rotates
 * its slice directions through six angles keyed on `frame.frameId`, which TRAA
 * then integrates into a clean result — the paper's intended pairing. But
 * `frameId` counts every render the session ever made, so a captured frame's
 * noise pattern would depend on how long the page had been open. The counter
 * here restarts with the temporal history in `resetSequence`, which the
 * capture path calls, so a capture sees the same six-frame phase every time.
 *
 * **It can be switched off without leaving the graph.** Removing it would
 * rebuild the output node and recompile mid-session, which the tier system
 * exists to avoid; a disabled node skips its pass and the composite's strength
 * goes to zero, which makes the result exactly the unoccluded colour.
 */
class SequencedGTAO extends GTAONode {
  enabled = true;
  private sequence = 0;

  constructor(depth: any, normal: any, camera: THREE.Camera) {
    super(depth, normal, camera);
    // Per pipeline render rather than per animation frame, so the rotation
    // phase counts renders — which a capture controls — and not rAF ticks,
    // which it does not. See `ScaledTRAA` in `PostChain.ts`.
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
  }

  updateBefore(frame: any): boolean | undefined {
    if (!this.enabled) return undefined;
    return (GTAONode.prototype as any).updateBefore.call(this, {
      renderer: frame.renderer,
      frameId: this.sequence++,
    });
  }

  resetSequence(): void {
    this.sequence = 0;
  }
}

export interface AoSettings {
  enabled: boolean;
  /** Fraction of the drawing buffer the AO target is rendered at. */
  resolutionScale: number;
  /** GTAO sample budget; the node splits it into slices and steps. */
  samples: number;
}

/**
 * Ambient occlusion on the opaque scene — ship, island, rocks, props — and never
 * on water or sky.
 *
 * **Applied to the scene colour before the temporal resolve, not to the
 * lighting.** The physically correct place for AO is the indirect term of each
 * material, which three supports through `builtinAOContext`. It needs the AO
 * *before* the scene draws, which means a second geometry pass of the whole
 * scene (ocean grid, canopy and meadow included) to produce depth first — and
 * the context applies to every nested render inside the pass, including the
 * planar reflector, where a screen-space AO texture would be read at the wrong
 * coordinates. A post multiply on the colour is the compromise most shipped
 * forward renderers make; strength is kept moderate because it also darkens
 * direct light in a crevice, which the correct version would not.
 *
 * Before the resolve because GTAO's per-pixel noise is designed to be
 * integrated over its six rotations by exactly that resolve.
 *
 * **Water and sky are masked, not merely tolerated.** The ocean writes a flag
 * into its velocity target's third channel (every other material writes 0
 * there, which is what three's vec2-to-vec4 conversion produces), and the sky
 * has no depth so GTAO leaves it at 1. Without the flag the sea would occlude
 * itself in every wave trough — a darkening the water shader already models
 * physically and would get twice.
 */
/** [underwater-look-3] GTAO with the eye under water; see `setSubmerged`. */
const UW_AO = { radius: 1.8, thickness: 3, strength: 1 };

export class AmbientOcclusion {
  private readonly node: SequencedGTAO;
  private readonly uStrength = uniform(0.75);
  private strength = 0.75;
  private settings: AoSettings = { enabled: true, resolutionScale: 0.5, samples: 12 };

  /** Metres. AO fades out across this range; see `build`. */
  private readonly fadeStart = uniform(35);
  private readonly fadeEnd = uniform(90);

  constructor(
    private readonly depth: any,
    private readonly camera: THREE.Camera,
    private readonly normal: any = null,
  ) {
    // Real normals from the scene pass when it has them; `null` falls back to
    // GTAO's depth reconstruction, which stripes distant grazing ground.
    this.node = new SequencedGTAO(depth, normal, camera);
    this.node.useTemporalFiltering = true;
    // Metres: this is a ship and a coastline, not a room. Half a metre finds
    // the deck furniture, the gunwale and the rock clefts; the falloff keeps
    // it from turning into a dirty halo on the hull at a distance.
    this.node.radius.value = 0.6;
    this.node.thickness.value = 1.5;
    this.node.distanceExponent.value = 1.6;
    this.node.distanceFallOff.value = 1;
    this.node.scale.value = 1;
    this.node.resolutionScale = this.settings.resolutionScale;
    this.node.samples.value = this.settings.samples;
  }

  /**
   * @param color Scene colour, linear HDR, texture node.
   * @param velocity The MRT velocity texture; its `.z` is the water flag.
   * @param unjitter UV offset of this frame's TRAA jitter; the colour is read
   *   back at the unjittered position. See `PostChain`'s `jitterUv`.
   * @returns The occluded colour as a composited node (the resolve converts it).
   */
  build(color: any, velocity: any, unjitter: any = null): any {
    const aoTexture = this.node.getTextureNode() as any;
    const depth = this.depth;
    const near = reference('near', 'float', this.camera);
    const far = reference('far', 'float', this.camera);
    return Fn(() => {
      const p = uv();
      const c = (unjitter === null ? color.sample(p) : sampleCatmullRom(color, p.sub(unjitter))).toVar('aoColor');
      const water = velocity.sample(p).z.saturate();
      const a = aoTexture.sample(p).r;
      // Near field only. A half-metre occluder is sub-pixel past a hundred
      // metres anyway, and beyond that the normals GTAO rebuilds from a
      // perspective depth buffer degrade into vertical streaks — measured on
      // the island's tree line at a kilometre before this fade existed.
      const distance = perspectiveDepthToViewZ(depth.sample(p).r, near, far).negate();
      const fade = float(1).sub(smoothstep(this.fadeStart, this.fadeEnd, distance));
      // Grazing surfaces get none. The depth buffer spans a 24 km sea, so at a
      // hundred metres it steps in terraces, and GTAO's horizon search reads
      // those steps as occluders on any flat ground seen edge-on — the beach
      // came out as a grid of stripes. Real normals did not fix it; the view
      // angle does. `|n.z|` is the view-space facing ratio, close enough to
      // n.v inside a 55-degree frustum.
      const facing: any = this.normal === null ? float(1) : this.normal.sample(p).z.abs().smoothstep(0.12, 0.4);
      const k: any = this.uStrength.mul(float(1).sub(water)).mul(fade).mul(facing);
      return vec4(c.rgb.mul(mix(float(1), a, k)), c.a);
    })();
  }

  apply(settings: AoSettings): void {
    this.settings = settings;
    this.node.enabled = settings.enabled;
    this.node.resolutionScale = settings.resolutionScale;
    this.node.samples.value = settings.samples;
    this.submergedRadius(this.submerged);
  }

  setStrength(strength: number): void {
    this.strength = strength;
    this.submergedRadius(this.submerged);
  }

  /**
   * [underwater-look-3] How submerged the eye is, 0..1. Under water the
   * occluders that matter are a reef's — a sponge on the sand, a coral head on
   * a rock, the rock on the bed — a metre or two across, not the ship's deck
   * furniture the 0.6 m radius was set for, and the light is almost all
   * ambient, so the occlusion the round-3 review missed ("no contact shadows
   * where reef life meets the rock and sand") is the term doing the most work.
   * Radius, thickness and strength blend toward `UW_AO`. Uniform writes only;
   * call per frame.
   */
  setSubmerged(amount: number): void {
    const a = Math.min(1, Math.max(0, amount));
    if (a === this.submerged) return;
    this.submerged = a;
    this.submergedRadius(a);
  }

  private submerged = 0;

  private submergedRadius(a: number): void {
    this.node.radius.value = 0.6 + (UW_AO.radius - 0.6) * a;
    this.node.thickness.value = 1.5 + (UW_AO.thickness - 1.5) * a;
    const strength = this.strength + (UW_AO.strength - this.strength) * a;
    this.uStrength.value = this.settings.enabled ? strength : 0;
  }

  get enabled(): boolean {
    return this.node.enabled;
  }

  setEnabled(on: boolean): void {
    this.apply({ ...this.settings, enabled: on });
  }

  /** The raw AO texture, for debug views. */
  get texture(): any {
    return this.node.getTextureNode();
  }

  get gtao(): GTAONode {
    return this.node;
  }

  resetSequence(): void {
    this.node.resetSequence();
  }

  dispose(): void {
    this.node.dispose();
  }
}
