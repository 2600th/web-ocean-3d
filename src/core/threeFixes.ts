import * as THREE from 'three/webgpu';

/**
 * Shared uniform groups, one copy per render context (three r185).
 *
 * The camera block and the other per-render uniforms live in a *shared* bind
 * group: every material with the same set of them binds one buffer, written
 * once per render. r185 picks that group when a material's nodes are built,
 * scoped by `renderer._currentRenderContext` at that moment, and keeps it in a
 * node build that any render context can reuse. Two things break the scoping:
 *
 * - Builds here are asynchronous (`installAsyncPipelines`), so a build started
 *   inside the planar reflection's render can finish after it, when the current
 *   context is the main pass again.
 * - A build is cached by material, not by context, so one made for one render
 *   is handed to another.
 *
 * Either way a reflection object and a main-pass object end up binding one
 * buffer. The reflection renders *inside* the main pass, so its write (the
 * mirrored camera) lands after the main pass's and before the main pass's
 * commands run: everything in the main pass on that buffer draws from under
 * the sea. In practice the camp smoke in the mirror took the cloud deck and
 * the rain with it (same uniform layout), and the storm sky went bare.
 *
 * Resolving the group when it is *bound* rather than built fixes both: each
 * render object swaps every shared group for its context's own copy, made on
 * first use and shared by everything else in that context, so sharing within
 * a render is kept and sharing across renders is gone. The copy is made the
 * way upstream clones a group, with fresh CPU-side buffers, so its first update
 * writes everything.
 *
 * r185 only: upstream moved the shared cache into `Bindings` with a context
 * argument (mrdoob/three.js#34707). Re-check, and likely delete, on upgrade.
 */
export function installThreeFixes(renderer: THREE.WebGPURenderer): void {
  if (THREE.REVISION !== '185') return;
  const objects = (renderer as any)._objects;
  if (!objects || objects.__perContextSharedGroups) return;
  objects.__perContextSharedGroups = true;

  const createRenderObject = objects.createRenderObject.bind(objects);
  let patched = false;
  objects.createRenderObject = (...args: unknown[]): unknown => {
    const renderObject = createRenderObject(...args);
    if (!patched) {
      patched = true;
      patchGetBindings(Object.getPrototypeOf(renderObject));
    }
    return renderObject;
  };
}

/**
 * Instance matrices as vertex attributes on the WebGL2 fallback (three r185).
 *
 * r185 puts an `InstancedMesh`'s matrices in a uniform block whenever they fit
 * — `mat4 buffer[count]` — so the instance **count** is part of the shader
 * source. Two meshes sharing one material but placing 20 and 36 instances
 * compile two programs, identical except for that number. The reef alone is
 * dozens of meshes on two materials, and on ANGLE's D3D11 path every program
 * costs FXC about 1.2 s to link: 66 copies of the reef-life shader were 80 s
 * of the 90 s the fallback spent compiling at boot, which is what put its time
 * to first frame at 65 s against main's 7.
 *
 * Above the block-size limit r185 already uses four instanced `vec4`
 * attributes, the way `WebGLRenderer` always did. That shader does not mention
 * the count, so meshes on one material share a program again, and the matrices
 * upload when they change rather than with every draw. Reporting no uniform
 * block space sends every instanced mesh down that path. The limit is read in
 * three places (`Instance.js` twice, `RangeNode.js`), all choosing between the
 * same two layouts.
 *
 * WebGL only, because the attribute path does not keep the *previous* frame's
 * matrices current — it copies them on the CPU but never bumps the
 * attribute's version — and this fallback has no velocity MRT to need them
 * (see `PostChain`). WebGPU keeps the uniform layout.
 */
export function installWebGLFixes(renderer: THREE.WebGPURenderer): void {
  if (THREE.REVISION !== '185') return;
  const capabilities = (renderer as any).backend?.capabilities;
  if (!capabilities || typeof capabilities.getUniformBufferLimit !== 'function') return;
  capabilities.getUniformBufferLimit = () => 0;
}

/** Per render context: original shared group -> that context's copy. */
const contextCopies = new WeakMap<object, WeakMap<object, any>>();

function patchGetBindings(proto: any): void {
  const getBindings = proto.getBindings;
  proto.getBindings = function (this: any): any[] {
    const bindings = getBindings.call(this);
    if (this.__sharedGroupsResolved === bindings) return bindings;
    this.__sharedGroupsResolved = bindings;

    let copies = contextCopies.get(this.context);
    if (copies === undefined) contextCopies.set(this.context, (copies = new WeakMap()));
    for (let i = 0; i < bindings.length; i++) {
      const group = bindings[i];
      if (group.bindings[0]?.groupNode?.shared !== true) continue;
      let copy = copies.get(group);
      if (copy === undefined) {
        copy = new group.constructor(group.name, group.bindings.map(cloneBinding));
        copies.set(group, copy);
      }
      bindings[i] = copy;
    }
    return bindings;
  };
}

/** As upstream's `UniformsGroup.clone`: shallow, with fresh CPU-side state. */
function cloneBinding(binding: any): any {
  const clone = binding.clone();
  if (binding.isUniformsGroup) {
    clone._buffer = null;
    clone._values = null;
    clone._updateRanges = [];
    clone._updateRangeCache = new Map();
    clone._addedIndices = new Set();
  }
  return clone;
}
