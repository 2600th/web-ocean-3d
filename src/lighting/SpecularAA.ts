import * as THREE from 'three/webgpu';
import { float, materialRoughness, normalView } from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * Geometric specular anti-aliasing for normal-mapped materials.
 *
 * A normal map packs more surface detail into a pixel than the pixel can
 * resolve once the surface is a few metres away: planking grain, rope lay and
 * rock pitting all turn into normals that change direction from one pixel to
 * the next. The specular lobe is sampled once per pixel along whichever normal
 * the pixel happens to land on, so a narrow lobe flickers on and off as the
 * camera moves — the sparkle on the hull's wet planks and on the rocks in the
 * swash. TRAA averages some of it away and smears the rest.
 *
 * The fix every current engine ships (Tokuyoshi & Kaplanyan, "Improved
 * Geometric Specular Antialiasing", 2019; Filament's `specularAA`) is to widen
 * the lobe by exactly the normal variance the pixel's footprint covers: estimate
 * it from the screen-space derivatives of the shading normal and add it to the
 * squared roughness, clamped so a smooth surface cannot be pushed past a
 * plausible roughness. The highlight then shrinks in intensity instead of
 * aliasing, which is what a camera with a larger pixel would have seen.
 *
 * three already adds the *geometric* normal's curvature (`getGeometryRoughness`);
 * it does not account for the normal map, which is where the sparkle comes from.
 *
 * Wraps whatever roughness graph the material already has — wetting included —
 * so it composes, and must be applied before the material first compiles.
 */

/** Screen-space variance of the pixel filter (Filament's default). */
const SIGMA2 = 0.15;
/** Cap on the added squared roughness; keeps a polished surface polished. */
const KAPPA = 0.18;

const applied = new WeakSet<THREE.Material>();

export function applySpecularAA(material: THREE.Material): void {
  const target = material as unknown as { roughnessNode: Node; normalMap: unknown; needsUpdate: boolean; isNodeMaterial?: boolean };
  if (target.isNodeMaterial !== true || applied.has(material)) return;
  if (typeof (material as THREE.MeshStandardMaterial).roughness !== 'number') return;
  applied.add(material);
  const base: Node = target.roughnessNode ?? materialRoughness;
  const n = normalView;
  const du = n.dFdx();
  const dv = n.dFdy();
  const variance = du.dot(du).add(dv.dot(dv)).mul(SIGMA2);
  const kernel = variance.mul(2).min(KAPPA);
  // three's `roughness` is perceptual; the lobe width is its square.
  const r = float(base).clamp(0, 1);
  const alpha = r.mul(r);
  target.roughnessNode = alpha.mul(alpha).add(kernel).clamp(0, 1).sqrt().sqrt();
  target.needsUpdate = true;
}

/** Applies to every distinct normal-mapped node material under `root`. */
export function applySpecularAATo(root: THREE.Object3D): number {
  let count = 0;
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (!m || applied.has(m)) continue;
      // Normal-mapped or bump-noded only: without either the shading normal is
      // the geometric one, which three already filters. The island's generated
      // rock carries its detail as a `normalNode`, so that counts.
      const bumped = (m as THREE.MeshStandardMaterial).normalMap || (m as unknown as { normalNode?: unknown }).normalNode;
      if (!bumped) continue;
      applySpecularAA(m);
      count++;
    }
  });
  return count;
}
