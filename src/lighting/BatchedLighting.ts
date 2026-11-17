import * as THREE from 'three/webgpu';
import { dynamicLights } from 'three/addons/tsl/lighting/DynamicLightsNode.js';
import HemisphereLightDataNode from 'three/addons/tsl/lighting/data/HemisphereLightDataNode.js';
import { vec3 } from 'three/tsl';

/**
 * r187's batched hemisphere loop wiped the sky fill. It adds to the lighting
 * context's `irradiance`, and when nothing has written that variable before the
 * loop (the builtin hemisphere light used to be the first), the generated WGSL
 * initialises it to zero inside the loop body and again after the loop, so the
 * diffuse ambient from every hemisphere light was discarded. Measured: lit
 * surfaces up to 28/255 off by day. Touching it once before the loop puts
 * the initialisation there; adding zero leaves the sum bit-identical to the
 * unbatched one. Harmless if a later revision fixes the emission.
 */
const hemisphereSetup = HemisphereLightDataNode.prototype.setup;
HemisphereLightDataNode.prototype.setup = function (this: unknown, builder: { context: { irradiance: { addAssign(v: unknown): void } } }) {
  builder.context.irradiance.addAssign(vec3(0));
  return hemisphereSetup.call(this, builder as never);
} as typeof hemisphereSetup;

/**
 * The scene's analytic lights as uniform arrays and loops, instead of one
 * unrolled block of shader per light.
 *
 * three's default `LightsNode` emits each light's code separately into every lit
 * material, and puts each light's id in the material's cache key. With the sun,
 * five fire and lantern lights, the lightning's channel and two hemisphere fills
 * that was seven copies of the BRDF's direct term in each of the ~120 lit
 * fragment shaders the boot compiles, and a light joining or leaving the scene
 * recompiled all of them. `DynamicLightsNode` evaluates every light of a type in
 * one loop over uniform arrays, sized by a uniform count, and keys the material
 * only on which types are present. The sun keeps three's own path, because it
 * casts shadows; the arithmetic for the others is three's own lighting model
 * through the same `direct` call, so the shading does not change (captures
 * are bit-identical by day; at night the point lights' sum moves by one level
 * at most). Compiling the 63 lit main-pass and reflection pipelines one at a
 * time on FXC took 77 s before and 65 s after, about a sixth off the work the
 * boot's pipeline compiles are bound by.
 *
 * What this buys besides smaller shaders: a light can be added, removed or
 * hidden at any time for the price of a uniform write. `LocalLights` hides the
 * lights that are at zero, so by day the loop runs no iterations at all, and a
 * tier that turns lights on or off no longer recompiles anything.
 *
 * That holds once a type has been seen, because the set of types is in the
 * cache key. A boot by day, or on Low, has no point light visible, so the first
 * dusk would add the type and recompile every lit material. The main scene's
 * node is therefore created with all three batched types present, at zero
 * count. Other scenes (the sky's environment capture) keep three's behaviour.
 */
const SEEDED_TYPES = [
  new THREE.PointLight(),
  new THREE.DirectionalLight(),
  new THREE.HemisphereLight(),
];

/** Sized to what the scene holds: five local lights; the lightning's channel; sky and storm-deck fills. */
const LIMITS = { maxPointLights: 8, maxDirectionalLights: 2, maxSpotLights: 1, maxHemisphereLights: 2 };

export class BatchedLighting extends THREE.Lighting {
  constructor(private readonly scene: THREE.Scene) {
    super();
  }

  override createNode(lights: THREE.Light[] = []): THREE.LightsNode {
    return dynamicLights(LIMITS).setLights(lights);
  }

  override getNode(scene: THREE.Scene): THREE.LightsNode {
    const created = !(this as unknown as { _lightsNodeMap: WeakMap<object, unknown> })._lightsNodeMap.has(scene);
    const node = super.getNode(scene);
    // The renderer replaces the list on every render; the batched types stay.
    if (created && scene === this.scene) node.setLights(SEEDED_TYPES).setLights([]);
    return node;
  }
}
