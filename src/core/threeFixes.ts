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
 *
 * The attribute path also has a frame of lag, fixed below. It draws from
 * copies (an interleaved buffer for the matrices, an instanced attribute for
 * the colours) that share the mesh's arrays, and their versions are synced from
 * the mesh's in an `OnFrameUpdate`. That runs in `Nodes.updateForRender`, which
 * `Renderer._renderObjectDirect` calls *after* `Geometries.updateForRender` has
 * already decided what to upload, and the upload is checked once per render
 * call. So a mesh drawn in one pass a frame showed its new matrices a frame
 * late, and a burst that reused a slot flashed the previous one for a frame.
 * The uniform layout re-uploaded every draw and never had this. The sync
 * three does afterwards is done here first, and three's then finds nothing to
 * change.
 */
export function installWebGLFixes(renderer: THREE.WebGPURenderer): void {
  if (THREE.REVISION !== '185') return;
  const capabilities = (renderer as any).backend?.capabilities;
  if (!capabilities || typeof capabilities.getUniformBufferLimit !== 'function') return;
  capabilities.getUniformBufferLimit = () => 0;
  installInstanceCopySync(renderer);
}

/**
 * The frame-of-lag fix above, on its own, for every backend.
 *
 * WebGPU takes the attribute path too, for any mesh whose matrices do not fit
 * a uniform block (64 KiB: 1024 instances). The reef and rock fields are far
 * past that, and since they stopped re-uploading every draw (`DealtMatrices`
 * in `reef/InstanceField`) they upload on the version, so they had the same
 * lag: a deal's new counts and fades drawn with the previous deal's matrices
 * for a frame.
 */
export function installInstanceCopySync(renderer: THREE.WebGPURenderer): void {
  if (THREE.REVISION !== '185') return;
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

/**
 * Buffer names that do not change the shader from one mesh to the next
 * (WebGPU, three r185).
 *
 * r185 names a buffer binding after its node's global id: `NodeBuffer_76015`.
 * An `InstancedMesh` has a node build of its own (its uuid is in the material
 * cache key), and its own matrix buffer node, so two meshes on one material
 * produced WGSL identical but for that number. three keeps one shader module
 * per distinct source and one pipeline per distinct pair of modules, so each
 * mesh cost a pipeline of its own: at High, 36 copies of the seabed rock's
 * vertex shader, 32 of the reef's, and a few more for the flora, 67 modules of
 * 451 at boot. A pipeline is an FXC compile in Chrome's GPU process, and the
 * boot was bound by those: 15 compile threads at full load for the length of
 * the content compile.
 *
 * Named by first use within the build instead, the copies come out identical
 * and share a module and a pipeline. Only the name changes: the binding
 * layout, the group and the buffer each mesh binds are what they were. A name
 * a caller chose is left alone.
 */
export function installStableBufferNames(): void {
  if (THREE.REVISION !== '185') return;
  const proto = (THREE as any).WGSLNodeBuilder?.prototype;
  if (!proto || typeof proto.getUniformFromNode !== 'function' || proto.__stableBufferNames) return;
  proto.__stableBufferNames = true;
  const getUniformFromNode = proto.getUniformFromNode;
  const local = new WeakMap<object, Map<object, number>>();
  proto.getUniformFromNode = function (this: any, node: any, type: string, ...rest: unknown[]): any {
    const uniform = getUniformFromNode.call(this, node, type, ...rest);
    // The auto name is assigned once, when the uniform is first made for a
    // stage; the same node in the other stage gets the same local name.
    if (uniform?.name === 'NodeBuffer_' + node.id) {
      let names = local.get(this);
      if (names === undefined) local.set(this, (names = new Map()));
      let index = names.get(node);
      if (index === undefined) names.set(node, (index = names.size));
      uniform.name = 'NodeBuffer_L' + index;
    }
    return uniform;
  };
}

/** As `Instance.js`'s `OnFrameUpdate`, for the copies that share the mesh's arrays. */
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
 * Per-draw submission costs that three r185 pays without needing to (both
 * backends unless noted). Measured on a live High frame, 1920x1080:
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
  if (THREE.REVISION !== '185') return;
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
