import * as THREE from 'three/webgpu';

/**
 * Corrections from the PBR audit of the shipped models (map averages read back
 * on the GPU, linear):
 *
 *  - `rock_slab_a` / `rock_slab_b` (island crags, headland rocks, shore
 *    outcrops) ship a metalness map averaging 0.4 under a factor of 1. Rock is a
 *    dielectric; at 0.4 it lost 40% of its diffuse and wore an albedo-tinted
 *    specular. Stone gets metalness 0 whatever its maps say.
 *
 * Everything else measured inside plausible ranges (wood 0.04–0.14, canvas 0.17,
 * plaster 0.30, roughness 0.3–1, metalness ≤ 0.13 on wood and rope), so nothing
 * else is touched. The one outlier left alone is the glazed jug at 0.92 albedo,
 * a single small prop.
 */
const STONE = /rock|crag|outcrop|boulder|cliff|stone/i;

export function sanitizeDielectrics(root: THREE.Object3D): number {
  let fixed = 0;
  const seen = new Set<THREE.Material>();
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    let stone = false;
    for (let n: THREE.Object3D | null = mesh; n && n !== root.parent; n = n.parent) {
      if (STONE.test(n.name)) {
        stone = true;
        break;
      }
    }
    for (const m of ([] as THREE.Material[]).concat(mesh.material)) {
      if (seen.has(m)) continue;
      const s = m as THREE.MeshStandardMaterial;
      if (!s.isMeshStandardMaterial || s.metalness <= 0) continue;
      if (!stone && !STONE.test(s.name)) continue;
      seen.add(m);
      s.metalness = 0;
      fixed++;
    }
  });
  return fixed;
}
