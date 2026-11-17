import * as THREE from 'three/webgpu';

/**
 * Patches to three's internals that upstream does not have, and the revision
 * whose source each hook was checked against. On any other revision they stay
 * off rather than patch code they were not written for: re-check, then add it.
 *
 * Two earlier patches are gone because r187 fixed the same bugs upstream: one
 * shared uniform group per render context (#34707), and one shader build per
 * instanced material rather than per mesh (#34872, #35007). A third, instance
 * matrices as attributes, existed only for the retired WebGL2 fallback.
 */
const VERIFIED_PERF_REVISIONS = new Set(['187']);

/**
 * Instance matrices a frame late on the attribute path.
 *
 * Any `InstancedMesh` whose matrices do not fit a uniform block (64 KiB: 1024
 * instances) draws them from instanced attributes — copies that share the
 * mesh's arrays, with versions synced from the mesh's in an `OnFrameUpdate`.
 * That runs in `Nodes.updateForRender`, which `Renderer._renderObjectDirect`
 * calls *after* `Geometries.updateForRender` has already decided what to
 * upload. The reef and rock fields are far past the limit, and since they
 * stopped re-uploading every draw (`DealtMatrices` in `reef/InstanceField`)
 * they upload on the version: a deal's new counts and fades were drawn with the
 * previous deal's matrices for a frame. The sync three does afterwards is done
 * here first, and three's then finds nothing to change.
 */
export function installInstanceCopySync(renderer: THREE.WebGPURenderer): void {
  if (!VERIFIED_PERF_REVISIONS.has(THREE.REVISION)) return;
  const geometries = (renderer as any)._geometries;
  if (!geometries || typeof geometries.updateForRender !== 'function') return;
  if (geometries.__instanceCopySync) return;
  geometries.__instanceCopySync = true;

  const updateForRender = geometries.updateForRender.bind(geometries);
  geometries.updateForRender = (renderObject: any): void => {
    const mesh = renderObject.object;
    if (mesh?.isInstancedMesh) syncInstanceCopies(mesh, renderObject.getAttributes());
    updateForRender(renderObject);
  };
}

function syncInstanceCopies(mesh: THREE.InstancedMesh, attributes: readonly any[]): void {
  const matrices = mesh.instanceMatrix;
  const colors = mesh.instanceColor;
  for (const attribute of attributes) {
    const source =
      attribute.isInterleavedBufferAttribute === true ? attribute.data : attribute;
    if (source === matrices || source === colors) continue;
    const owner =
      source.array === matrices.array ? matrices : colors && source.array === colors.array ? colors : null;
    if (owner === null || source.version === owner.version) continue;
    source.clearUpdateRanges();
    source.updateRanges.push(...owner.updateRanges);
    source.version = owner.version;
  }
}

/**
 * Shadow casters compiled their whole colour graph to throw it away (WebGPU).
 *
 * A shadow pass draws every caster with one override material, and
 * `Renderer._getShadowNodes` hands it `vec4(0, 0, 0, colorNode.a)` whenever the
 * source material has a `colorNode` or a `map`. The alpha is only there for
 * `alphaTest` and the mask nodes to discard by. Most of this scene's materials
 * are opaque procedural ones (the props, the island detail, the dressing
 * rocks), so each caster's shadow shader carried that material's noise and
 * palette, and nothing read the result. The map is depth-only: PCF compares
 * against the depth texture, and the colour target is sampled only with
 * `shadowMap.transmitted`. That made one shadow pipeline per material, 26 at
 * High, with the colour graph's attributes and varyings in the vertex stage too.
 *
 * Where nothing can read the alpha, the caster gets no colour node. Its shadow
 * shader is then the one any untextured opaque caster with the same vertex path
 * gets, and three shares the module and the pipeline. Depth is written exactly
 * as before, so the shadows do not change. `depthNode` and the position nodes
 * are kept, so displaced casters (the leaves, the swaying flora) still cast
 * where they are drawn.
 *
 * three caches the nodes per material version, and the `alphaTest` setter
 * bumps the version when it crosses zero, so a material that starts testing
 * alpha gets its colour node back on its next shadow draw.
 */
export function installShadowCasterSharing(renderer: THREE.WebGPURenderer): void {
  if (!VERIFIED_PERF_REVISIONS.has(THREE.REVISION)) return;
  const r = renderer as any;
  if (typeof r._getShadowNodes !== 'function' || r.__shadowCasterSharing) return;
  r.__shadowCasterSharing = true;

  const stripped = new WeakMap<object, object>();
  const getShadowNodes = r._getShadowNodes.bind(r);
  r._getShadowNodes = (material: any): any => {
    const nodes = getShadowNodes(material);
    if (nodes.colorNode === null) return nodes;
    const alphaRead =
      material.alphaTest > 0 ||
      material.maskNode?.isNode === true ||
      material.maskShadowNode?.isNode === true ||
      material.castShadowNode?.isNode === true ||
      r.shadowMap.transmitted === true ||
      r.shadowMap.type === THREE.VSMShadowMap;
    if (alphaRead) return nodes;
    let opaque = stripped.get(nodes);
    if (opaque === undefined) {
      const fresh: object = { ...nodes, colorNode: null };
      stripped.set(nodes, fresh);
      opaque = fresh;
    }
    return opaque;
  };
}

/**
 * Per-draw submission costs that three pays without needing to. Measured on a live High frame, 1920x1080:
 *
 * - **Shadow passes re-validated every caster's render object.** A shadow pass
 *   draws every caster with one shared override material and copies each
 *   source material's `alphaTest` onto it. `Material.alphaTest`'s setter bumps
 *   `version` whenever the value crosses zero, so in a pass that alternates
 *   alpha-tested casters (kelp, leaves) with opaque ones (rocks, the hull) the
 *   override's version moves on nearly every draw, and `RenderObjects.get`
 *   answers each move by rebuilding the render object's full material cache
 *   key: 163 keys a frame across the three cascades, about 1.5 ms of main
 *   thread, to find each time that nothing had changed. A render object is per
 *   caster, so its own source's `alphaTest` is all its key depends on. Here the
 *   override takes the value without the bump, and is bumped only when a
 *   source material's `alphaTest` really does cross zero, which still sends
 *   that caster's render object through the rebuild it needs.
 *
 * - **One uniform write per buffer (WebGPU).** A uniforms group records each
 *   changed uniform as a range, and the backend writes each run of contiguous
 *   ranges with its own `queue.writeBuffer`: about 2 150 writes a frame, most
 *   of a few bytes. The CPU copy mirrors the GPU buffer, so one write spanning
 *   the first to the last changed byte uploads the same contents. Uniform
 *   buffers are small (a few hundred bytes), so the gaps cost nothing.
 */
export function installSubmissionFixes(renderer: THREE.WebGPURenderer, options: { batchSubmits?: boolean } = {}): void {
  if (!VERIFIED_PERF_REVISIONS.has(THREE.REVISION)) return;
  const r = renderer as any;
  if (r.__submissionFixes) return;
  r.__submissionFixes = true;

  // Last `alphaTest > 0` seen per source material in a shadow pass.
  const alphaTested = new WeakMap<object, boolean>();
  const renderObject = r.renderObject;
  r.renderObject = function (
    this: any,
    object: unknown,
    scene: any,
    camera: unknown,
    geometry: unknown,
    material: any,
    ...rest: unknown[]
  ): unknown {
    const override = scene.overrideMaterial;
    if (override !== null && override !== undefined && override.isShadowPassMaterial === true && material.allowOverride === true) {
      const tested = material.alphaTest > 0;
      const last = alphaTested.get(material);
      if (last !== undefined && last !== tested) override.version++;
      alphaTested.set(material, tested);
      // Bypasses the setter; three assigns the same value next, and finds it
      // on the same side of zero.
      override._alphaTest = material.alphaTest;
    }
    return renderObject.call(this, object, scene, camera, geometry, material, ...rest);
  };

  const utils = r.backend?.bindingUtils;
  const device = r.backend?.device;
  if (!utils || !device || typeof utils.updateBinding !== 'function') return;
  installFramebufferCopyFixes(r);
  if (options.batchSubmits !== false) installSubmitBatching(device);
  const updateBinding = utils.updateBinding.bind(utils);
  utils.updateBinding = (binding: any): void => {
    const ranges = binding.updateRanges;
    const array = binding.buffer;
    if (ranges.length < 2 || !ArrayBuffer.isView(array)) {
      updateBinding(binding);
      return;
    }
    let start = Infinity;
    let end = 0;
    for (let i = 0; i < ranges.length; i++) {
      const range = ranges[i];
      if (range.start < start) start = range.start;
      if (range.start + range.count > end) end = range.start + range.count;
    }
    const typed = array as Float32Array;
    device.queue.writeBuffer(r.backend.get(binding).buffer, start * typed.BYTES_PER_ELEMENT, typed, start, end - start);
  };
}

/**
 * Framebuffer copies (WebGPU): 27 a frame at High, where 4 carry anything new.
 *
 * A viewport texture node (`viewportSharedTexture`, `viewportDepthTexture`)
 * copies the pass's colour or depth when the first object that samples it is
 * drawn, and three ends the render pass for the copy and begins a new one,
 * re-sending every pipeline, bind group and buffer for the draws after it. The
 * ocean samples the scene behind it through 18 shared-colour nodes and five
 * depth nodes, and each node copied for itself: at the ocean's draw the scene
 * pass split 23 times in a row, with no draw between, copying identical pixels.
 * In Chrome's GPU process those splits and copies were a large part of what
 * held the frame at half rate.
 *
 * Two changes, neither of which changes what any draw samples:
 *
 * - A copy is skipped when the same source was already copied into the same
 *   texture and nothing has been drawn, begun or cleared anywhere since. The
 *   destination then already holds exactly what the copy would write.
 * - Depth nodes copy into one texture per render target rather than one per
 *   node. Each node still copies *when it always did*: the frame-update key is
 *   kept per node, as it was when every node had a texture of its own, so a
 *   node that copies later on purpose (the spray, before its cores are drawn)
 *   still does. A copy overwrites the shared texture only after everything that
 *   sampled the earlier copy has been recorded, and WebGPU runs commands in
 *   order, so each draw reads the copy it read before. The copies that were in
 *   a row now land in one texture, and the rule above drops all but the first.
 */
function installFramebufferCopyFixes(r: any): void {
  const backend = r.backend;
  if (typeof backend.copyFramebufferToTexture !== 'function') return;

  // What could change what a copy reads: a draw into its source, a pass begun
  // on it, a clear. Counted per attachment, so a nested render into another
  // target (the reflection, inside the scene pass) does not force a recopy.
  // A clear without a context, or anything unrecognised, counts everywhere.
  const written = new WeakMap<object, number>();
  let everywhere = 0;
  const touch = (renderContext: any): void => {
    if (!renderContext?.renderTarget) {
      everywhere++;
      return;
    }
    for (const attachment of renderContext.textures ?? []) written.set(attachment, (written.get(attachment) ?? 0) + 1);
    const depth = renderContext.depthTexture;
    if (depth) written.set(depth, (written.get(depth) ?? 0) + 1);
  };
  const draw = backend.draw;
  backend.draw = function (this: unknown, renderObject: any, ...rest: unknown[]): unknown {
    touch(renderObject?.context);
    return draw.call(this, renderObject, ...rest);
  };
  const beginRender = backend.beginRender;
  backend.beginRender = function (this: unknown, renderContext: any, ...rest: unknown[]): unknown {
    touch(renderContext);
    return beginRender.call(this, renderContext, ...rest);
  };
  const clear = backend.clear;
  backend.clear = function (this: unknown, ...args: unknown[]): unknown {
    everywhere++;
    return clear.apply(this, args);
  };

  const lastCopy = new WeakMap<object, { stamp: number; source: unknown; x: number; y: number; z: number; w: number }>();
  const copyFramebufferToTexture = backend.copyFramebufferToTexture.bind(backend);
  backend.copyFramebufferToTexture = (texture: any, renderContext: any, rectangle: any): void => {
    const destination = backend.get(texture).texture;
    const sourceTexture = renderContext.renderTarget
      ? texture.isDepthTexture
        ? renderContext.depthTexture
        : renderContext.textures?.[0]
      : null;
    const source = sourceTexture ? backend.get(sourceTexture).texture : null;
    const stamp = sourceTexture ? everywhere * 1e9 + (written.get(sourceTexture) ?? 0) : -1;
    const last = destination ? lastCopy.get(destination) : undefined;
    if (
      source !== null &&
      source !== undefined &&
      last !== undefined &&
      last.stamp === stamp &&
      last.source === source &&
      last.x === rectangle.x &&
      last.y === rectangle.y &&
      last.z === rectangle.z &&
      last.w === rectangle.w
    ) {
      return;
    }
    copyFramebufferToTexture(texture, renderContext, rectangle);
    if (destination && source !== null) {
      lastCopy.set(destination, { stamp, source, x: rectangle.x, y: rectangle.y, z: rectangle.z, w: rectangle.w });
    }
  };

  const proto = (THREE as any).ViewportDepthTextureNode?.prototype;
  if (!proto || proto.__sharedDepthCopies) return;
  proto.__sharedDepthCopies = true;
  const sharedCopies = new WeakMap<object, unknown>();
  const getTextureForReference = proto.getTextureForReference;
  proto.getTextureForReference = function (this: any, reference: unknown = null): unknown {
    const base = this.referenceNode ?? this;
    base._cacheTextures = sharedCopies;
    return getTextureForReference.call(this, reference);
  };
  const updateReference = proto.updateReference;
  proto.updateReference = function (this: any, frame: unknown): unknown {
    updateReference.call(this, frame);
    return this.referenceNode ?? this;
  };
}

/**
 * One queue submit per batch of render calls instead of one per call (WebGPU).
 *
 * three submits a command buffer at the end of every `render()`: 93 a frame at
 * High, most of them a single full-screen quad. In Chrome every submit is a
 * round of work in the GPU process (Dawn's D3D12 queue submit, resource state
 * tracking, a fence), and measured on a High frame that per-submit overhead
 * was half of the GPU process's main thread: 11.3 ms a frame, 5.4 ms with the
 * same commands in one submit. That thread, not the page's JavaScript and not
 * the GPU, was what held High at half the display rate.
 *
 * So command buffers are held and submitted together, in order, at the end of
 * the task that encoded them. What makes this exact is WebGPU's ordering: a
 * `queue.writeBuffer` takes effect before any command buffer submitted after
 * it. A held command buffer must therefore be submitted before anything that
 * would change what it reads, and it is:
 *
 * - before a `writeBuffer` into any buffer a held command buffer references
 *   (bind groups, vertex and index buffers, copies, bundles: all tracked as
 *   they are recorded; anything recorded through an object made before this
 *   was installed counts as referencing every buffer);
 * - before any texture upload, buffer map, destroy, or `onSubmittedWorkDone`,
 *   whatever it touches;
 * - at the end of the task, so a frame is complete before it is presented.
 *
 * A write into a buffer nothing held reads (the next render call's own
 * uniforms, nearly always) goes straight through, which is where the batching
 * comes from. Same commands, same order, same data seen by each.
 */
function installSubmitBatching(device: any): void {
  const queue = device.queue;
  if (queue.__submitBatching) return;
  queue.__submitBatching = true;

  type Refs = Set<object> | 'all';
  const bindGroupRefs = new WeakMap<object, object[]>();
  const bundleRefs = new WeakMap<object, Refs>();
  const encoderRefs = new WeakMap<object, Refs>();
  const commandBufferRefs = new WeakMap<object, Refs>();
  const passEncoder = new WeakMap<object, object>();

  const add = (owner: object, buffer: unknown, map: WeakMap<object, Refs>): void => {
    const refs = map.get(owner);
    if (refs === undefined || refs === 'all' || !buffer) return;
    refs.add(buffer as object);
  };
  const addBindGroup = (owner: object, group: object, map: WeakMap<object, Refs>): void => {
    const buffers = bindGroupRefs.get(group);
    if (buffers === undefined) {
      if (group) map.set(owner, 'all');
      return;
    }
    for (const buffer of buffers) add(owner, buffer, map);
  };

  const createBindGroup = device.createBindGroup;
  device.createBindGroup = function (descriptor: any): unknown {
    const group = createBindGroup.call(this, descriptor);
    const buffers: object[] = [];
    for (const entry of descriptor.entries ?? []) {
      const resource = entry.resource;
      if (resource && typeof resource === 'object') {
        if ('buffer' in resource && resource.buffer) buffers.push(resource.buffer);
        else if (typeof (resource as any).mapAsync === 'function') buffers.push(resource);
      }
    }
    bindGroupRefs.set(group, buffers);
    return group;
  };

  // Render bundles: what they reference is what their encoder recorded.
  const createRenderBundleEncoder = device.createRenderBundleEncoder;
  device.createRenderBundleEncoder = function (descriptor: unknown): any {
    const encoder = createRenderBundleEncoder.call(this, descriptor);
    const refs = new Set<object>();
    bundleRefs.set(encoder, refs);
    const wrap = (name: string, use: (args: any[]) => void): void => {
      const original = encoder[name];
      encoder[name] = function (...args: any[]): unknown {
        use(args);
        return original.apply(this, args);
      };
    };
    wrap('setBindGroup', ([, group]) => addBindGroup(encoder, group, bundleRefs));
    wrap('setVertexBuffer', ([, buffer]) => add(encoder, buffer, bundleRefs));
    wrap('setIndexBuffer', ([buffer]) => add(encoder, buffer, bundleRefs));
    wrap('drawIndirect', ([buffer]) => add(encoder, buffer, bundleRefs));
    wrap('drawIndexedIndirect', ([buffer]) => add(encoder, buffer, bundleRefs));
    const finish = encoder.finish;
    encoder.finish = function (...args: unknown[]): unknown {
      const bundle = finish.apply(this, args);
      bundleRefs.set(bundle, bundleRefs.get(encoder) ?? 'all');
      return bundle;
    };
    return encoder;
  };

  const createCommandEncoder = device.createCommandEncoder;
  device.createCommandEncoder = function (descriptor: unknown): any {
    const encoder = createCommandEncoder.call(this, descriptor);
    encoderRefs.set(encoder, new Set());
    return encoder;
  };

  // Prototype-level, so passes and encoders three already holds are covered;
  // an encoder made before this ran has no entry and counts as 'all'.
  const patch = (proto: any, name: string, use: (self: any, args: any[]) => void): void => {
    const original = proto?.[name];
    if (typeof original !== 'function') return;
    proto[name] = function (...args: any[]): unknown {
      use(this, args);
      return original.apply(this, args);
    };
  };
  const ownerOf = (pass: object): object | undefined => passEncoder.get(pass);
  const passUse = (pass: object, buffer: unknown): void => {
    const encoder = ownerOf(pass);
    if (encoder !== undefined) add(encoder, buffer, encoderRefs);
  };
  const g = globalThis as any;
  for (const [ctor, begin] of [
    [g.GPUCommandEncoder, 'beginRenderPass'],
    [g.GPUCommandEncoder, 'beginComputePass'],
  ] as const) {
    const proto = ctor?.prototype;
    const original = proto?.[begin];
    if (typeof original !== 'function') continue;
    proto[begin] = function (this: object, ...args: unknown[]): unknown {
      const pass = original.apply(this, args);
      if (!encoderRefs.has(this)) encoderRefs.set(this, 'all');
      passEncoder.set(pass, this);
      return pass;
    };
  }
  for (const proto of [g.GPURenderPassEncoder?.prototype, g.GPUComputePassEncoder?.prototype]) {
    patch(proto, 'setBindGroup', (pass, [, group]) => {
      const encoder = ownerOf(pass);
      if (encoder !== undefined && group) addBindGroup(encoder, group, encoderRefs);
    });
    patch(proto, 'setVertexBuffer', (pass, [, buffer]) => passUse(pass, buffer));
    patch(proto, 'setIndexBuffer', (pass, [buffer]) => passUse(pass, buffer));
    patch(proto, 'drawIndirect', (pass, [buffer]) => passUse(pass, buffer));
    patch(proto, 'drawIndexedIndirect', (pass, [buffer]) => passUse(pass, buffer));
    patch(proto, 'dispatchWorkgroupsIndirect', (pass, [buffer]) => passUse(pass, buffer));
    patch(proto, 'executeBundles', (pass, [bundles]) => {
      const encoder = ownerOf(pass);
      if (encoder === undefined) return;
      for (const bundle of bundles ?? []) {
        const refs = bundleRefs.get(bundle);
        if (refs === undefined || refs === 'all') encoderRefs.set(encoder, 'all');
        else for (const buffer of refs) add(encoder, buffer, encoderRefs);
      }
    });
  }
  const encoderProto = g.GPUCommandEncoder?.prototype;
  const encoderUse = (encoder: object, buffer: unknown): void => {
    if (!encoderRefs.has(encoder)) encoderRefs.set(encoder, 'all');
    add(encoder, buffer, encoderRefs);
  };
  patch(encoderProto, 'copyBufferToBuffer', (e, [source, a, b]) => {
    encoderUse(e, source);
    // (source, destination[, size]) or (source, sourceOffset, destination, ...)
    encoderUse(e, typeof a === 'number' ? b : a);
  });
  patch(encoderProto, 'copyBufferToTexture', (e, [source]) => encoderUse(e, source?.buffer));
  patch(encoderProto, 'copyTextureToBuffer', (e, [, destination]) => encoderUse(e, destination?.buffer));
  patch(encoderProto, 'clearBuffer', (e, [buffer]) => encoderUse(e, buffer));
  patch(encoderProto, 'resolveQuerySet', (e, [, , , destination]) => encoderUse(e, destination));
  const finish = encoderProto?.finish;
  if (typeof finish === 'function') {
    encoderProto.finish = function (this: object, ...args: unknown[]): unknown {
      const commandBuffer = finish.apply(this, args);
      commandBufferRefs.set(commandBuffer, encoderRefs.get(this) ?? 'all');
      return commandBuffer;
    };
  }

  // The held batch.
  const submit = queue.submit.bind(queue);
  let held: object[] = [];
  let heldRefs: Set<object> | 'all' = new Set();
  let scheduled = false;
  const flush = (): void => {
    if (held.length === 0) return;
    const batch = held;
    held = [];
    heldRefs = new Set();
    submit(batch);
  };
  queue.submit = (commandBuffers: object[]): void => {
    // An empty submit is someone ticking the device on purpose
    // (`pipelinesSettled`); it goes now, after what is held.
    if (commandBuffers.length === 0) {
      flush();
      submit(commandBuffers);
      return;
    }
    for (const commandBuffer of commandBuffers) {
      held.push(commandBuffer);
      const refs = commandBufferRefs.get(commandBuffer) ?? 'all';
      if (heldRefs === 'all') continue;
      if (refs === 'all') heldRefs = 'all';
      else for (const buffer of refs) heldRefs.add(buffer);
    }
    if (!scheduled) {
      scheduled = true;
      queueMicrotask(() => {
        scheduled = false;
        flush();
      });
    }
  };
  const writeBuffer = queue.writeBuffer.bind(queue);
  queue.writeBuffer = (buffer: object, ...rest: unknown[]): unknown => {
    if (held.length > 0 && (heldRefs === 'all' || heldRefs.has(buffer))) flush();
    return writeBuffer(buffer, ...rest);
  };
  for (const name of ['writeTexture', 'copyExternalImageToTexture', 'onSubmittedWorkDone']) {
    const original = queue[name];
    if (typeof original !== 'function') continue;
    queue[name] = (...args: unknown[]): unknown => {
      flush();
      return original.apply(queue, args);
    };
  }
  patch(g.GPUBuffer?.prototype, 'mapAsync', () => flush());
  patch(g.GPUBuffer?.prototype, 'destroy', () => flush());
  patch(g.GPUTexture?.prototype, 'destroy', () => flush());
  patch(g.GPUQuerySet?.prototype, 'destroy', () => flush());
}
