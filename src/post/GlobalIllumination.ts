import * as THREE from 'three/webgpu';
import SSGINode from 'three/addons/tsl/display/SSGINode.js';
import { uniform } from '../core/uniform';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * `?gi=1`: build SSGI (and the scene pass's albedo target it needs) at boot.
 *
 * **Off by default, and not verified.** Its pipeline does not compile under
 * FXC — `FXC compile failed with error: E_FAIL` for the SSGI material, the
 * frame's command buffer invalid after it — and FXC is the compiler this
 * project's headless test Chrome runs (its bundled DXC will not load; see
 * `playwright.config.ts`), as does any Windows Chrome that falls back from
 * DXC. A pipeline that fails there takes the whole frame with it, so it
 * cannot ship on a tier. Behind the flag, sessions with a DXC (or Metal or
 * Vulkan) backend can A/B it; the tier table already says where it would go
 * (Ultra and Max, replacing GTAO), so turning it on is deleting this flag
 * once it has been seen to compile and look right.
 */
export const SSGI_REQUESTED =
  typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('gi') === '1';

/**
 * three's SSGI with GTAO's two changes (see `SequencedGTAO`): the slice
 * rotation and ray offset count renders, not rAF ticks, and the node can be
 * switched off in place.
 */
class SequencedSSGI extends (SSGINode as unknown as new (...args: any[]) => any) {
  enabled = true;
  private sequence = 0;

  constructor(color: any, depth: any, normal: any, camera: THREE.PerspectiveCamera) {
    super(color, depth, normal, camera);
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
    this.useTemporalFiltering = true;
  }

  updateBefore(frame: any): any {
    if (!this.enabled) return undefined;
    return (SSGINode.prototype as any).updateBefore.call(this, {
      renderer: frame.renderer,
      frameId: this.sequence++,
    });
  }

  resetSequence(): void {
    this.sequence = 0;
  }
}

export interface GiSettings {
  enabled: boolean;
  /** Hemisphere slices per pixel; the header's "with temporal filtering" presets. */
  slices: number;
  /** Steps along each side of a slice. */
  steps: number;
  /** Fraction of the scene pass's resolution. */
  resolutionScale: number;
}

/**
 * One bounce of diffuse light and its ambient occlusion, from three's
 * `SSGINode` (visibility-bitmask SSGI, after SSRT3): sunlit sand lifting the
 * hull's underside, green bounce under the canopy, light carried into the
 * fort's rooms off its sunlit floor.
 *
 * It replaces GTAO on the tiers that run it rather than stacking with it —
 * both are an occlusion term for the same ambient light, and the bitmask
 * version is the better one (it lets light through behind thin geometry,
 * where GTAO's horizon assumes a solid wall). `PostChain` switches GTAO off
 * whenever this is on.
 *
 * Composited the way three's example does — `colour·AO + albedo·GI` — which
 * is why the scene pass carries an albedo target (`sceneMrt`). Like the AO,
 * before the temporal resolve: one or two slices a pixel is noise that only
 * the accumulation makes into an image, which is what the node is designed
 * around.
 */
export class GlobalIllumination {
  private readonly node: SequencedSSGI;
  /** Composite weights; zero when off, so the result is exactly the input. */
  readonly uAoStrength = uniform(0);
  readonly uGiStrength = uniform(0);
  private settings: GiSettings = { enabled: false, slices: 2, steps: 8, resolutionScale: 1 };

  constructor(color: any, depth: any, normal: any, camera: THREE.PerspectiveCamera) {
    this.node = new SequencedSSGI(color, depth, normal, camera);
    // Metres. A deck, a hull and a fort's room are a few metres across; the
    // node's 12 m default reaches past the hull onto open water, which the
    // composite masks out but the march still pays for.
    this.node.radius.value = 5;
    // Screen-space step distribution, the node's default: detail up close,
    // where the bounce is visible, and fewer wasted taps far away.
    this.node.useScreenSpaceSampling.value = true;
    this.node.thickness.value = 0.6;
    this.node.expFactor.value = 2;
    this.node.aoIntensity.value = 1;
    // The GI is weighted again in the composite (`uGiStrength`); the node's
    // own scale stays at one so the two are not tuned against each other.
    this.node.giIntensity.value = 1;
    this.node.enabled = false;
  }

  get ao(): any {
    return this.node.getAONode();
  }

  get gi(): any {
    return this.node.getGINode();
  }

  apply(settings: GiSettings): void {
    this.settings = settings;
    this.node.enabled = settings.enabled;
    this.node.sliceCount.value = settings.slices;
    this.node.stepCount.value = settings.steps;
    this.node.resolutionScale = settings.resolutionScale;
    this.uAoStrength.value = settings.enabled ? 1 : 0;
    this.uGiStrength.value = settings.enabled ? GI_STRENGTH : 0;
  }

  get enabled(): boolean {
    return this.settings.enabled;
  }

  resetSequence(): void {
    this.node.resetSequence();
  }

  dispose(): void {
    this.node.dispose();
  }
}

/**
 * Bounce weight in the composite. The node returns the cosine-weighted
 * radiance its slices saw; times the receiver's albedo that is a Lambertian
 * surface's reflected bounce, up to the node's own normalisation. Not yet
 * calibrated against a frame — it has not been seen to compile here (see
 * `SSGI_REQUESTED`) — so 1, the physically motivated value, until it is.
 */
const GI_STRENGTH = 1;
