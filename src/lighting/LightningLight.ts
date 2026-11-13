import * as THREE from 'three/webgpu';

/**
 * The light a lightning stroke throws on the scene.
 *
 * `Lightning` drew the bolt and handed the colour grade a scalar, so a strike
 * lifted the whole frame evenly — the hull, the sea and the island brightened by
 * the same amount whichever way they faced, which is a camera exposure change,
 * not a light. A stroke is two lights at once and both are cheap:
 *
 * - **The channel itself**, a line source kilometres long; at the distances a
 *   strike lands from the viewer it lights the scene from one direction, so it
 *   is a directional light from the bolt toward the eye, a little above the
 *   horizon. This is what puts a bright edge on the side of the hull that faces
 *   the strike and leaves the other side in the dark.
 * - **The cloud deck it lights from inside**, which is a sky-sized area source
 *   overhead: a hemisphere light, cool above and near-black below.
 *
 * Neither casts a shadow (a second shadow-casting directional light is another
 * full set of cascades for a tenth of a second). Both exist from boot at zero
 * intensity, because a light's existence is part of every lit material's shader
 * and its intensity is not — see `LocalLights`.
 *
 * Intensities are in the sun's units (3.4 at noon). At the peak return stroke
 * of a strike a kilometre and a half away the channel is about three noon suns
 * on the faces turned to it — a stroke under a storm deck photographs brighter
 * than the day it interrupts for the length of the flash — and it falls off with
 * distance, floored so a strike on the horizon still reads. The colour grade's
 * frame-wide lift was cut to a little over half when this went in; see
 * `FLASH_GRADE_SHARE` in `main`.
 */

const FLASH_COLOR = new THREE.Color(0.78, 0.85, 1.0);
const DECK_COLOR = new THREE.Color(0.62, 0.7, 0.92);
const GROUND_COLOR = new THREE.Color(0.05, 0.06, 0.08);

const CHANNEL_PEAK = 12;
const DECK_PEAK = 2.4;
/** Distance at which the falloff starts, metres. */
const NEAR = 1500;

export class LightningLight {
  readonly channel: THREE.DirectionalLight;
  readonly deck: THREE.HemisphereLight;

  constructor() {
    this.channel = new THREE.DirectionalLight(FLASH_COLOR, 0);
    this.channel.name = 'lightning-flash';
    this.channel.castShadow = false;
    this.deck = new THREE.HemisphereLight(DECK_COLOR, GROUND_COLOR, 0);
    this.deck.name = 'lightning-deck';
  }

  /**
   * @param flash The stroke envelope, 0..~1.4, from `Lightning.flashIntensity`.
   * @param strike The stroke's ground point and its distance from the eye.
   * @param cloudBase Metres; the channel is lit from roughly halfway up it.
   */
  update(flash: number, strike: { x: number; z: number; distance: number }, cloudBase: number, eye: THREE.Vector3): void {
    if (!(flash > 0) || !Number.isFinite(strike.distance)) {
      this.channel.intensity = 0;
      this.deck.intensity = 0;
      return;
    }
    const falloff = Math.min(1, Math.max(0.3, Math.pow(NEAR / Math.max(1, strike.distance), 1.2)));
    this.channel.intensity = CHANNEL_PEAK * flash * falloff;
    this.deck.intensity = DECK_PEAK * flash * falloff;
    // From the bolt's mid-height toward the eye. Directional, so only the
    // direction matters; the target is the eye so the shadowless light needs no
    // further upkeep.
    this.channel.position.set(strike.x, eye.y + cloudBase * 0.45, strike.z);
    this.channel.target.position.copy(eye);
    this.channel.target.updateMatrixWorld();
  }
}
