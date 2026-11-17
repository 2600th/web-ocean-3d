import * as THREE from 'three/webgpu';
import { Fn, diffuseColor, float, mrt, normalView, output, vec3, vec4, velocity } from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * The scene pass's MRT — colour, motion vector, view normal, and albedo when
 * asked for — with the data targets blended by each material's own coverage.
 *
 * Albedo is for the screen-space GI (`GlobalIllumination`): bounce light is
 * irradiance, and what a surface reflects of it is its albedo times that, not
 * its lit colour times it. Eight bits a channel (`PostChain.buildResolve` sets
 * the type), which keeps the four targets inside WebGPU's 32 bytes a sample:
 * three half-float RGBA targets and one RGBA8. Only with `albedo` (a session
 * booted with SSGI): the MRT is part of every material's pipeline, so it is
 * fixed for the session, and without SSGI it would be a 4-byte write a
 * fragment that nothing reads.
 *
 * ### The bug this closes
 *
 * Three's MRT gives every target but `output` `NoBlending` (`MRTNode`'s
 * default blend mode). A transparent material's blending therefore fades only
 * its colour: every fragment it does not discard *overwrites* the motion vector,
 * the normal and the water's AO flag (velocity `.z`) underneath it, at full
 * weight, however close to zero its alpha. Under TRAA that is a hard-edged copy
 * of the quad laid over the sea: the history is reprojected with the quad's
 * motion instead of the water's, GTAO reads the quad's normal and loses the
 * water mask, and the resolve draws the outline. Round 10c found it as pale
 * rectangles under the bow spray; every transparent sprite, ribbon and halo in
 * the scene had the same fault.
 *
 * ### The fix, in one place
 *
 * The data targets take `MaterialBlending` — the drawing material's own blend
 * state, as `output` does — and their outputs carry the material's coverage in
 * alpha:
 *
 *  - **opaque** (including alpha-tested): no blending at all, exactly as before,
 *    so nothing changes for the bulk of the scene and nothing is paid for it;
 *  - **normal-blended transparent**: `src·a + dst·(1 − a)`, so a sprite at 5%
 *    moves the sea's motion vector and normal 5% of the way, and one at zero
 *    leaves them untouched;
 *  - **additive** (glows, halos, flashes; subtractive and multiply likewise):
 *    coverage 0, so `src·0 + dst` — light added in front of a surface is not a
 *    surface and has no motion of its own.
 *
 * Materials that set their own `mrtNode` (the water, clouds, leaves) keep their
 * outputs; those write alpha 1 and so still replace what is under them, as
 * before.
 *
 * Three r185 merges a material's `mrtNode` into this one per draw, but the
 * pipeline's blend state comes from the renderer's MRT — this node — so the
 * blend modes below hold for every draw into the scene pass.
 */
export function sceneMrt({ albedo = false }: { albedo?: boolean } = {}): any {
  const coverage = Fn((builder: any) => {
    const material = builder.material as THREE.Material | null;
    if (!material) return float(1);
    const blending = material.blending;
    // Unblended (three only blends a NormalBlending material when it is
    // transparent): it replaces, as it always did.
    if (blending === THREE.NoBlending) return float(1);
    if (blending === THREE.NormalBlending && material.transparent !== true) return float(1);
    // Light added, subtracted or multiplied over a surface: not a surface.
    if (
      blending === THREE.AdditiveBlending ||
      blending === THREE.SubtractiveBlending ||
      blending === THREE.MultiplyBlending
    ) {
      return float(0);
    }
    // Normal (and custom) blending: its own coverage.
    return (diffuseColor as any).a.clamp(0, 1);
  });
  // Premultiplied blending takes `src·1 + dst·(1 − a)`: weight the data first.
  const weighted = Fn(([value]: [any], builder: any) => {
    const material = builder.material as THREE.Material | null;
    const a: any = coverage();
    const v: any = material?.premultipliedAlpha === true ? value.mul(a) : value;
    return vec4(v, a);
  });
  const node: any = mrt({
    output,
    // `.z = 0`: not water, for the AO mask (`AmbientOcclusion`).
    velocity: weighted(vec3((velocity as any).xy, 0)),
    normal: weighted(normalView),
    ...(albedo ? { albedo: weighted((diffuseColor as any).rgb) } : {}),
  });
  const data = new THREE.BlendMode(THREE.MaterialBlending);
  node.setBlendMode('velocity', data);
  node.setBlendMode('normal', data);
  if (albedo) node.setBlendMode('albedo', data);
  return node;
}
