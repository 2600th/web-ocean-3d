import {
  Fn,
  cameraViewMatrix,
  faceDirection,
  float,
  mix,
  mx_noise_float,
  mx_worley_noise_float,
  mx_worley_noise_vec2,
  normalView,
  positionView,
  sin,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * Shading shared by the reef and the animals on it: the procedural surface
 * patterns, a bump from them, and the cheap submerged lighting model the
 * instanced animals use.
 *
 * Patterns are evaluated on the *template-space* position, so a pattern sits
 * still on a swaying coral or a flapping ray instead of swimming through it.
 * They are selected by the `reefSurf` channel `MeshBuilder` writes:
 *
 *   1  brain coral    iso-lines of a smooth noise field, which meander exactly
 *                     the way the grooves of Diploria do
 *   2  polyp pores    the near-cell distance of a fine Worley field
 *   3  scutes         Worley cell edges (F2 - F1): a turtle's carapace plates
 *   4  spots          bright Worley dots: an eagle ray's back, a moray's skin
 *   5  scales         fine Worley cells with pale seams: a turtle's head and
 *                     flippers
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/** `vec2(height, tint)` for a surface selector and a template-space position. */
export function lifePattern(surf: Node, pg: Node): Node {
  return Fn(() => {
    const isBrain = surf.sub(1).abs().lessThan(0.5);
    const isPores = surf.sub(2).abs().lessThan(0.5);
    const isScute = surf.sub(3).abs().lessThan(0.5);
    const isSpots = surf.sub(4).abs().lessThan(0.5);
    const isScales = surf.sub(5).abs().lessThan(0.5);

    const field = mx_noise_float(pg.mul(4.2));
    const groove = sin(field.mul(26)).abs().toVar('lifeGroove');

    const cell = mx_worley_noise_float(pg.mul(34));
    const pore = cell.smoothstep(0.05, 0.42).toVar('lifePore');

    const f = mx_worley_noise_vec2(pg.mul(7));
    const edge = f.y.sub(f.x).smoothstep(0.02, 0.14).toVar('lifeEdge');

    const dots = mx_worley_noise_float(pg.mul(16)).smoothstep(0.12, 0.22).oneMinus().toVar('lifeDots');

    // Plates darken toward their centre and carry a broken streak, like a
    // green turtle's radiating carapace markings.
    const streak = mx_noise_float(pg.mul(vec3(9, 3, 9))).mul(0.5).add(0.5);
    const plate = edge
      .mul(0.45)
      .add(0.55)
      .mul(f.x.smoothstep(0, 0.6).mul(0.45).add(0.8))
      .mul(streak.mul(0.45).add(0.8));

    // Small skin scales: dark centres with pale seams, the reverse of plates.
    const sf = mx_worley_noise_vec2(pg.mul(26));
    const seam = sf.y.sub(sf.x).smoothstep(0.02, 0.12).toVar('lifeSeam');

    const height = isBrain.select(
      groove,
      isPores.select(
        pore,
        isScute.select(edge, isSpots.select(dots.mul(0.4), isScales.select(seam.mul(0.6), float(0)))),
      ),
    );
    const tint = isBrain.select(
      groove.smoothstep(0.1, 0.7).mul(0.5).add(0.5),
      isPores.select(
        pore.mul(0.35).add(0.65),
        isScute.select(
          plate,
          isSpots.select(dots.mul(2.2).add(1), isScales.select(seam.mul(-0.8).add(1.55), float(1))),
        ),
      ),
    );
    return vec2(height, tint);
  })();
}

/**
 * A view-space normal bumped by a procedural height, by Mikkelsen's
 * surface-gradient method — what three's `BumpMapNode` does, which only accepts
 * a texture. Meant for `material.normalNode`.
 */
export function bumpedNormal(height: Node, scale: number, base: Node = normalView): Node {
  return Fn(() => {
    const h = height.mul(scale);
    const dHdx = h.dFdx();
    const dHdy = h.dFdy();
    const sigmaX = positionView.dFdx().normalize();
    const sigmaY = positionView.dFdy().normalize();
    const n = base;
    const r1 = sigmaY.cross(n);
    const r2 = n.cross(sigmaX);
    const det = sigmaX.dot(r1).mul(faceDirection);
    const grad = det.sign().mul(r1.mul(dHdx).add(r2.mul(dHdy)));
    return det.abs().mul(n).sub(grad).normalize();
  })();
}

/** A world-space direction to view space, for `normalNode`. */
export function worldNormalToView(n: Node): Node {
  return cameraViewMatrix.mul(vec4(n, 0)).xyz.normalize();
}

export interface SubmergedLightInputs {
  /** World normal, already facing the viewer. */
  normal: Node;
  worldPosition: Node;
  sunDirection: Node;
  sunColor: Node;
  ambient: Node;
  /** `(worldPosition) => ~1` caustic pattern, or null. */
  caustics: ((worldPosition: Node) => Node) | null;
  /** Light coming through a thin translucent body from behind, 0..1. */
  translucency?: number;
}

/** Where caustics stop, in metres of depth — `groundShading`'s band. */
const CAUSTIC_SHALLOW = 2;
const CAUSTIC_DEEP = 48;

/**
 * Wrapped diffuse sun with caustics, plus a sky-weighted ambient — the model
 * `Fish` and `Kelp` use, extended with the caustic pattern so an animal swimming
 * through a caustic web is lit by it the way the sand under it is.
 */
export function submergedLight(inputs: SubmergedLightInputs): Node {
  return Fn(() => {
    const n = inputs.normal;
    const ndl = n.dot(inputs.sunDirection).toVar('lifeNdl');
    const key = ndl.mul(0.6).add(0.4).max(0).toVar('lifeKey');
    const through = ndl.negate().max(0).mul(inputs.translucency ?? 0);
    const sky = n.y.mul(0.5).add(0.5);
    let caustic: Node = float(1);
    if (inputs.caustics) {
      const wp = inputs.worldPosition;
      const depth = wp.y.negate();
      const reach = float(1)
        .sub(depth.smoothstep(CAUSTIC_SHALLOW, CAUSTIC_DEEP))
        .mul(float(1).sub(wp.y.smoothstep(-1.2, 1.2)));
      caustic = mix(float(1), inputs.caustics(wp), reach);
    }
    return inputs.ambient
      .mul(sky.mul(0.6).add(0.4))
      .add(inputs.sunColor.mul(key.add(through)).mul(caustic));
  })();
}
