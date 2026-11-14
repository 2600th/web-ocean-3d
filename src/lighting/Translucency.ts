import * as THREE from 'three/webgpu';
import { Fn, materialColor, materialEmissive, normalWorld, positionWorld, vec3 } from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * Light through a thin sheet: sails.
 *
 * Sailcloth is a few tenths of a millimetre of woven fibre. A real sail with the
 * sun behind it is the brightest surface on the ship — warm, even, and showing
 * every seam as a darker line — while ours rendered the shaded side flat
 * blue-grey, lit by the sky alone, exactly as if it were plate steel.
 *
 * What is missing is diffuse transmission: of the light arriving on the far
 * face, a fraction scatters through the weave and leaves the near face as if it
 * had been emitted there, tinted by the cloth. For a thin Lambertian sheet that
 * is the far face's irradiance times a transmittance, divided by pi — the same
 * form as the reflected term with the normal reversed — so it costs one dot
 * product. This is the "thin translucency" every current engine has for foliage
 * and cloth (Frostbite's and UE's two-sided foliage model are this term).
 *
 * It goes in through `emissiveNode`, which three adds before tone mapping, so it
 * is graded, fogged and bloomed like any other light. The key light's colour
 * node already carries its intensity and follows the moon at night, so a sail
 * glows at sunset and stops at dusk without anything here knowing what time it
 * is. Its one honest limit: it is not shadowed by the shadow map — a sail behind
 * another still glows — because the map tests the near face, and the near face of
 * a back-lit sail is in its own shadow by construction. The cloud shade is
 * applied, so a sail under a cloud does not.
 */

/**
 * Fraction of the far face's irradiance that comes through.
 *
 * Measured sailcloth transmittance runs from about 0.15 for heavy canvas to over
 * half for light nylon; this ship's flax would be toward the bottom, and the
 * term is further tinted by the cloth's own albedo on the way through.
 */
export const SAIL_TRANSMITTANCE = 0.32;

/** The key light's full-daylight intensity, which `keyColor` is normalised by. */
const KEY_DAYLIGHT = 3.4;

export interface TransmissionInputs {
  /** Direction toward the key light, world space. */
  keyDirection: Node;
  /** Key colour times intensity / 3.4 (so 1 = a clear noon sun). */
  keyColor: Node;
  /** Optional `(worldPosition) => 0..1` occlusion of the key, e.g. cloud shade. */
  occlusion?: (worldPosition: Node) => Node;
  transmittance?: number;
}

const applied = new WeakSet<THREE.Material>();

export function applyThinTransmission(material: THREE.Material, inputs: TransmissionInputs): void {
  if (applied.has(material)) return;
  applied.add(material);
  const target = material as unknown as { emissiveNode: Node; needsUpdate: boolean };
  const existing = target.emissiveNode ?? null;
  const t = (inputs.transmittance ?? SAIL_TRANSMITTANCE) * (KEY_DAYLIGHT / Math.PI);
  const through = Fn(() => {
    // `normalWorld` is the shading normal flipped to face the viewer on a
    // double-sided material, so a negative dot with the light means the light is
    // on the far face.
    const far = normalWorld.dot(vec3(inputs.keyDirection)).negate().max(0);
    let e: Node = vec3(materialColor).mul(vec3(inputs.keyColor)).mul(far.mul(t));
    if (inputs.occlusion) e = e.mul(inputs.occlusion(positionWorld));
    return e;
  })();
  // `materialEmissive` is what three would have used with no node set, so a
  // material with its own emissive keeps it.
  target.emissiveNode = vec3(existing ?? materialEmissive).add(through);
  target.needsUpdate = true;
}

/** Applies to every material under `root` whose mesh or material name matches `test`. */
export function applyThinTransmissionTo(root: THREE.Object3D, test: RegExp, inputs: TransmissionInputs): number {
  let count = 0;
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (!m || applied.has(m)) continue;
      if (!test.test(mesh.name) && !test.test(m.name)) continue;
      applyThinTransmission(m, inputs);
      count++;
    }
  });
  return count;
}
