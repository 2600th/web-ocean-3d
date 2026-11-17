import { objectGroup, renderGroup, uniform as threeUniform } from 'three/tsl';

/**
 * The app's `uniform()`: three's, in the shared per-render uniform group.
 *
 * three r185 puts every `uniform()` in the per-object group unless told
 * otherwise, and a per-object group is compared value by value, and written,
 * for every draw of every object in every pass. Nearly all of this project's
 * uniforms are scene-wide — the sun, the sky, fog, wetness, time, the wave
 * field — and they reach every scene material through the shared shading
 * functions, so each draw carried 40 to 140 of them: measured on a High frame,
 * about 2 ms of main thread a frame spent finding that a rock and a palm see
 * the same sun. In the render group they are compared once per render call
 * and bound from one buffer per material.
 *
 * Correct for any value that does not change *between the draws of one
 * render call*, which is every uniform the app writes from JavaScript or from
 * a render- or frame-scoped callback: nothing here writes one per draw
 * (there is no `onBeforeRender` in the project). The exception is a uniform
 * given a per-object callback, whose value depends on the object being drawn;
 * `onObjectUpdate` puts it back in the per-object group. Import `uniform`
 * from here rather than from `three/tsl` in app code.
 *
 * No compute pass uses these: a render group is only refreshed by renders.
 */
export const uniform = ((...args: unknown[]) => {
  const node = (threeUniform as (...a: unknown[]) => any)(...args);
  node.setGroup(renderGroup);
  const onObjectUpdate = node.onObjectUpdate;
  node.onObjectUpdate = function (this: any, callback: unknown) {
    this.setGroup(objectGroup);
    return onObjectUpdate.call(this, callback);
  };
  return node;
}) as typeof threeUniform;
