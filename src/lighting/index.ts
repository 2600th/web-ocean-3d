import * as THREE from 'three/webgpu';
import type { QualitySettings } from '../core/QualityManager';
import { LightningLight } from './LightningLight';
import { LocalLights } from './LocalLights';
import { SunShadows } from './SunShadows';

export { SunShadows } from './SunShadows';
export { LocalLights, SHIP_LANTERN_LOCAL } from './LocalLights';
export { LightningLight } from './LightningLight';
export { applySpecularAA, applySpecularAATo } from './SpecularAA';
export { sanitizeDielectrics } from './PbrAudit';
export { applyThinTransmission, applyThinTransmissionTo, SAIL_TRANSMITTANCE } from './Translucency';

/**
 * Everything the lighting pass added, behind one object so `main` wires it in a
 * handful of lines: the sun's cascades, the local lights, and the lightning's
 * light. All of it is created at boot — the set of lights in the scene is part
 * of every lit material's shader — and driven afterwards by uniforms and
 * intensities only.
 */
export class LightingRig {
  readonly shadows: SunShadows;
  readonly local: LocalLights;
  readonly flash: LightningLight;

  constructor(
    scene: THREE.Scene,
    renderer: THREE.WebGPURenderer,
    camera: THREE.PerspectiveCamera,
    sun: THREE.DirectionalLight,
    quality: QualitySettings,
  ) {
    this.shadows = new SunShadows(sun, camera, renderer, {
      cascades: quality.shadowCascades,
      distance: quality.shadowDistance,
      mapSize: quality.shadowMapSize,
    });
    this.local = new LocalLights(quality.localLights);
    this.flash = new LightningLight();
    scene.add(this.local.group, this.flash.channel, this.flash.channel.target, this.flash.deck);
  }

  /** A tier change: shadow reach and map size, and whether local lights shine. */
  applyQuality(quality: QualitySettings): void {
    this.shadows.configure(quality.shadowDistance, quality.shadowMapSize);
    this.local.setCount(quality.localLights);
  }

  /**
   * How dark it is for the purposes of lighting a lamp, 0..1: nothing in
   * daylight, all of it from deep dusk on, and most of it under a storm deck.
   */
  static darkness(sunY: number, overcast: number): number {
    const dusk = THREE.MathUtils.smoothstep(-sunY, -0.12, 0.04);
    const gloom = 0.75 * THREE.MathUtils.smoothstep(overcast, 0.55, 0.9);
    return Math.max(dusk, gloom);
  }

  update(
    elapsed: number,
    eye: THREE.Vector3,
    sunY: number,
    overcast: number,
    flash: number,
    strike: { x: number; z: number; distance: number },
    cloudBase: number,
  ): void {
    this.shadows.update();
    this.local.update(elapsed, eye, LightingRig.darkness(sunY, overcast));
    this.flash.update(flash, strike, cloudBase, eye);
  }

  dispose(): void {
    this.local.dispose();
    this.flash.channel.removeFromParent();
    this.flash.channel.target.removeFromParent();
    this.flash.deck.removeFromParent();
    this.flash.channel.dispose();
    this.flash.deck.dispose();
    this.shadows.node.dispose();
  }
}
