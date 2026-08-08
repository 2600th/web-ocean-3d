import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

/**
 * Caching glTF loader.
 *
 * Three properties matter for this project:
 *
 *  - **Dedup.** `Ship` and `Props` are constructed in parallel from the same
 *    `AssetLoader`. Two concurrent `load()` calls for one URL must issue exactly
 *    one network fetch, so the cache stores the in-flight *promise*, not the
 *    resolved value.
 *  - **Isolation.** Callers get a `clone()` of the cached scene graph. Handing
 *    out the same `Group` twice would mean the second `scene.add()` silently
 *    reparents it away from the first. Clones share geometry and material, so
 *    the cost is a handful of `Object3D`s.
 *  - **Determinism.** Every GPU resource the loader creates is recorded, so
 *    `dispose()` frees the lot in one pass rather than relying on callers to
 *    traverse their own subtrees.
 *
 * Compression is configured defensively. The shipped dressing set uses Meshopt
 * geometry and KTX2 textures; the loader wires both decoders before first
 * request.
 *
 * **Draco is opt-in and is never probed for.** It used to auto-detect, HEAD-ing
 * three candidate paths before the first model could load — and since nothing
 * in this project has ever been Draco-encoded (`scripts/optimize-assets.mjs`
 * encodes Meshopt and contains the string zero times), all three always missed.
 * That cost three serial round trips on the critical boot path and put three
 * 404s in the console of every single session, which is not free: the test
 * suite asserts on console errors, so a probe that always fails was indistinguishable
 * from a real asset failure.
 *
 * The capability is kept for anyone who does serve a decoder — pass
 * `dracoDecoderPath` — because pointing at a CDN would add a third-party
 * runtime dependency to an otherwise self-contained build.
 */

const BASIS_TRANSCODER_PATH = '/basis/';
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * The deadline every asset request answers to, exported so that the sidecar and
 * manifest fetches outside this class answer to the same one.
 *
 * A request with no deadline is the failure this project set out to remove: the
 * boot awaits it, nothing rejects, and the visitor watches a progress bar that
 * will never move again. `Promise.allSettled` does not help — it handles
 * rejection, and a promise that never settles is not a rejection.
 */
export const ASSET_TIMEOUT_MS = DEFAULT_TIMEOUT_MS;

/** `fetch` with a deadline, rejecting rather than hanging. */
export async function fetchWithDeadline(
  url: string,
  timeoutMs = ASSET_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Request timed out after ${timeoutMs} ms: ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/** Rejects if `work` has not settled within `timeoutMs`. */
export async function withDeadline<T>(
  work: Promise<T>,
  label: string,
  timeoutMs = ASSET_TIMEOUT_MS,
): Promise<T> {
  let timer = 0 as unknown as ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`Timed out after ${timeoutMs} ms: ${label}`)),
      timeoutMs,
    );
  });
  try {
    return await Promise.race([work, deadline]);
  } finally {
    clearTimeout(timer);
  }
}
const DEFAULT_MAX_CONCURRENT = 6;

type MeshoptDecoder = Parameters<GLTFLoader['setMeshoptDecoder']>[0];

interface PendingAsset {
  promise: Promise<THREE.Group>;
  controller: AbortController;
  /** Resolves when the job leaves the concurrency queue. See `enqueue`. */
  started: Promise<void>;
}

interface QueueJob<T> {
  run: () => Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
  /** Called when the job leaves the queue, so its deadline can start there. */
  onStarted: () => void;
}

interface ProgressEntry {
  loaded: number;
  total: number;
  url: string;
}

export interface AssetProgress {
  /** 0..1 over every file this loader has seen, across all requests. */
  fraction: number;
  itemsLoaded: number;
  itemsTotal: number;
  /** The file that just completed. */
  url: string;
}

export type AssetProgressCallback = (progress: AssetProgress) => void;

export interface AssetLoadOptions {
  /** Cancels this URL and aborts its underlying request when set. */
  signal?: AbortSignal;
  /** Per-asset deadline. Defaults to the loader-wide timeout. */
  timeoutMs?: number;
}

export interface AssetLoaderOptions {
  /** Receives aggregate progress for the loader and its active asset requests. */
  onProgress?: AssetProgressCallback;
  /** The initialised renderer used by KTX2Loader.detectSupport(). */
  renderer?: THREE.WebGPURenderer;
  /** Deadline applied to every load unless overridden by AssetLoadOptions. */
  timeoutMs?: number;
  /** Maximum number of source assets allowed to load at once. Defaults to six. */
  maxConcurrent?: number;
  /**
   * Convert loaded glTF materials to `MeshPhysicalNodeMaterial` so downstream
   * systems (caustics, underwater tint) can attach TSL nodes to them. Defaults
   * to true.
   */
  convertToNodeMaterials?: boolean;
  /** Explicit Draco decoder directory; skips auto-detection when provided. */
  dracoDecoderPath?: string;
}

export class AssetLoader {
  private readonly transcoderManager = new THREE.LoadingManager();
  private readonly cache = new Map<string, PendingAsset>();
  private readonly queue: Array<QueueJob<THREE.Group>> = [];
  private readonly progressEntries = new Set<ProgressEntry>();
  private readonly maxConcurrent: number;
  private readonly timeoutMs: number;
  private activeLoads = 0;

  /** Everything we created and therefore must free. */
  private readonly geometries = new Set<THREE.BufferGeometry>();
  private readonly materials = new Set<THREE.Material>();
  private readonly textures = new Set<THREE.Texture>();

  /**
   * Loaded material -> the node material standing in for it.
   *
   * One entry per *source* material, which is the whole point. `GLTFLoader`
   * hands the same `Material` instance to every mesh that shares it in the
   * file, and downstream code relies on that identity: `Props.bakeParts` groups
   * geometry by material instance so that a multi-part asset merges into one
   * draw per material. Converting per mesh would fork one glTF material into
   * one node material per node — which is exactly what happened to
   * `grass_bermuda_01`, a single-material file of twenty-one separate blades:
   * every blade became its own material, so nothing merged and the tuft was
   * scattered as twenty-one lonely sprigs instead of being stacked into one.
   */
  private readonly converted = new Map<THREE.Material, THREE.Material>();

  private draco: DRACOLoader | null = null;
  private dracoProbe: Promise<string | null> | null = null;
  private meshoptProbe: Promise<MeshoptDecoder | null> | null = null;
  private meshoptDecoder: MeshoptDecoder | null = null;
  private ktx2: KTX2Loader | null = null;

  private readonly convertMaterials: boolean;
  private readonly explicitDracoPath: string | undefined;
  private onProgress: AssetProgressCallback | undefined;
  private disposed = false;

  /** Reused so the per-file progress callback never allocates. */
  private readonly progressState: AssetProgress = {
    fraction: 0,
    itemsLoaded: 0,
    itemsTotal: 0,
    url: '',
  };

  constructor(options: AssetLoaderOptions = {}) {
    this.convertMaterials = options.convertToNodeMaterials ?? true;
    this.explicitDracoPath = options.dracoDecoderPath;
    this.onProgress = options.onProgress;
    this.timeoutMs = Math.max(1, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    this.maxConcurrent = Math.max(1, Math.floor(options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT));

    this.bindProgress(this.transcoderManager, 'basis transcoder');
    if (options.renderer) this.setRenderer(options.renderer);
  }

  setProgressCallback(callback: AssetProgressCallback | undefined): void {
    this.onProgress = callback;
  }

  /** 0..1 across every file requested so far. */
  get progress(): number {
    return this.progressState.fraction;
  }

  /**
   * Supplies the initialised renderer used by KTX2Loader to select a GPU format.
   * Call this after `renderer.init()` and before the first KTX2 asset load. The
   * app's `WebGPURenderer` also owns the forced-WebGL2 backend, and Three's
   * feature bridge handles both paths here.
   */
  setRenderer(renderer: THREE.WebGPURenderer): void {
    if (this.disposed) throw new Error('AssetLoader disposed; cannot set renderer');
    this.ktx2?.dispose();
    this.ktx2 = new KTX2Loader(this.transcoderManager)
      .setTranscoderPath(BASIS_TRANSCODER_PATH)
      .setWorkerLimit(Math.max(1, Math.min(4, this.maxConcurrent)))
      .detectSupport(renderer);
  }

  /**
   * Loads `url` and resolves with an independent clone of its scene graph.
   * Concurrent calls for the same URL share one fetch.
   */
  async load(url: string, options: AssetLoadOptions = {}): Promise<THREE.Group> {
    if (this.disposed) throw new Error(`AssetLoader disposed; cannot load ${url}`);

    let pending = this.cache.get(url);
    if (!pending) {
      const controller = new AbortController();
      const { promise, started } = this.enqueue(() => this.loadSource(url, controller.signal));
      pending = { promise, controller, started };
      this.cache.set(url, pending);
      // A failed load must not poison the cache — the next attempt should retry.
      promise.catch(() => {
        if (this.cache.get(url)?.promise === promise) this.cache.delete(url);
      });
    }

    const source = await waitForAbort(
      pending.promise,
      options.signal,
      options.timeoutMs ?? this.timeoutMs,
      (reason) => pending?.controller.abort(reason),
      pending.started,
    );
    return source.clone(true) as THREE.Group;
  }

  /** Frees every geometry, material and texture this loader created. */
  dispose(): void {
    this.disposed = true;
    // Abort what is still in the air *before* freeing what has landed.
    //
    // Clearing the cache does not stop a request; it only forgets it. A load
    // that completes after this point would build geometries, materials and
    // textures into sets that have already been emptied and will never be
    // visited again — which is a leak that grows with how unlucky the timing
    // was, and is worst on the slow connections this project is trying to
    // support.
    const disposal = new Error('AssetLoader disposed');
    for (const pending of this.cache.values()) pending.controller.abort(disposal);
    for (const geometry of this.geometries) geometry.dispose();
    for (const material of this.materials) material.dispose();
    for (const texture of this.textures) texture.dispose();
    this.geometries.clear();
    this.materials.clear();
    this.textures.clear();
    this.converted.clear();
    this.cache.clear();
    for (const job of this.queue.splice(0)) job.reject(disposal);
    this.transcoderManager.abort();
    this.ktx2?.dispose();
    this.ktx2 = null;
    this.draco?.dispose();
    this.draco = null;
  }

  // ------------------------------------------------------------------ internals

  private async loadSource(url: string, signal: AbortSignal): Promise<THREE.Group> {
    throwIfAborted(signal);
    const [draco, meshopt] = await Promise.all([
      this.configureDraco(signal),
      this.configureMeshopt(signal),
    ]);
    throwIfAborted(signal);

    const manager = new THREE.LoadingManager();
    this.bindProgress(manager, url);
    const gltf = new GLTFLoader(manager);
    if (draco) gltf.setDRACOLoader(draco);
    if (meshopt) gltf.setMeshoptDecoder(meshopt);
    if (this.ktx2) gltf.setKTX2Loader(this.ktx2);

    const abortRequest = () => manager.abort();
    signal.addEventListener('abort', abortRequest, { once: true });
    let loaded: Awaited<ReturnType<GLTFLoader['loadAsync']>>;
    try {
      loaded = await gltf.loadAsync(url);
    } finally {
      signal.removeEventListener('abort', abortRequest);
    }

    throwIfAborted(signal);
    const scene = loaded.scene;
    scene.name = scene.name || url;

    // Bake the glTF's own node transforms into world matrices once, so callers
    // that read bounding boxes immediately get correct numbers.
    scene.updateMatrixWorld(true);

    scene.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;

      this.geometries.add(mesh.geometry);

      const source = mesh.material;
      if (Array.isArray(source)) {
        mesh.material = source.map((m) => this.adoptMaterial(m));
      } else {
        mesh.material = this.adoptMaterial(source);
      }
    });

    return scene;
  }

  private bindProgress(manager: THREE.LoadingManager, initialUrl: string): void {
    const entry: ProgressEntry = { loaded: 0, total: 0, url: initialUrl };
    this.progressEntries.add(entry);

    manager.onStart = (url, loaded, total) => {
      entry.url = url || initialUrl;
      entry.loaded = loaded;
      entry.total = total;
      this.emitProgress(entry.url);
    };
    manager.onProgress = (url, loaded, total) => {
      entry.url = url || initialUrl;
      entry.loaded = loaded;
      entry.total = total;
      this.emitProgress(entry.url);
    };
    manager.onLoad = () => {
      entry.loaded = entry.total;
      this.emitProgress(entry.url);
    };
    manager.onError = (url) => {
      entry.url = url || initialUrl;
      entry.loaded = Math.min(entry.total, entry.loaded + 1);
      this.emitProgress(entry.url);
    };
  }

  private emitProgress(url: string): void {
    let itemsLoaded = 0;
    let itemsTotal = 0;
    for (const entry of this.progressEntries) {
      itemsLoaded += entry.loaded;
      itemsTotal += entry.total;
    }
    const state = this.progressState;
    state.url = url;
    state.itemsLoaded = itemsLoaded;
    state.itemsTotal = itemsTotal;
    state.fraction = itemsTotal > 0 ? Math.min(1, itemsLoaded / itemsTotal) : 0;
    this.onProgress?.(state);
  }

  /**
   * Queues a load, returning both its result and a promise that resolves when
   * it actually *starts*.
   *
   * The second one exists because the deadline must not include queue time.
   * With six slots and a hundred-odd requests, a job near the back can spend
   * most of a 30-second budget waiting its turn and then be failed for a
   * timeout it never had a chance to beat — and the slower the connection, the
   * more of the batch dies without ever being attempted. The deadline is a
   * budget for the request, so it starts when the request does.
   */
  private enqueue(run: () => Promise<THREE.Group>): {
    promise: Promise<THREE.Group>;
    started: Promise<void>;
  } {
    let onStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      onStarted = resolve;
    });
    const promise = new Promise<THREE.Group>((resolve, reject) => {
      this.queue.push({ run, resolve, reject, onStarted });
      this.pumpQueue();
    });
    return { promise, started };
  }

  private pumpQueue(): void {
    while (!this.disposed && this.activeLoads < this.maxConcurrent && this.queue.length > 0) {
      const job = this.queue.shift()!;
      this.activeLoads += 1;
      job.onStarted();
      void job.run()
        .then(job.resolve, job.reject)
        .finally(() => {
          this.activeLoads -= 1;
          this.pumpQueue();
        });
    }
  }

  /**
   * Takes ownership of a loaded material, optionally converting it to a node
   * material, and registers it (and its textures) for disposal.
   */
  private adoptMaterial(material: THREE.Material): THREE.Material {
    const already = this.converted.get(material);
    if (already) return already;

    const converted =
      this.convertMaterials && (material as THREE.MeshStandardMaterial).isMeshStandardMaterial
        ? toPhysicalNodeMaterial(material as THREE.MeshStandardMaterial)
        : material;

    if (converted !== material) {
      // The original is dead the moment we swap it out; free it now rather than
      // holding a reference to something nothing renders. Keyed on it first, so
      // the next mesh sharing it still finds the replacement.
      this.converted.set(material, converted);
      material.dispose();
    } else {
      this.converted.set(material, converted);
    }

    this.materials.add(converted);
    collectTextures(converted, this.textures);
    return converted;
  }

  private configureDraco(signal: AbortSignal): Promise<DRACOLoader | null> {
    if (this.draco) return Promise.resolve(this.draco);

    // No probe: an unset path means this build does not serve a decoder, which
    // is the case for every asset this project ships. See the module header.
    if (!this.dracoProbe) {
      this.dracoProbe = Promise.resolve(this.explicitDracoPath ?? null);
    }

    return this.dracoProbe.then((path) => {
      throwIfAborted(signal);
      if (!path || this.draco || this.disposed) return this.draco;
      const draco = new DRACOLoader();
      draco.setDecoderPath(path);
      this.draco = draco;
      return draco;
    });
  }

  /**
   * Meshopt's decoder is a self-contained ES module inside the three package —
   * no sidecar files, so it can always be wired up. The dynamic import keeps it
   * out of the initial bundle for the (current) case where nothing uses it.
   */
  private configureMeshopt(signal: AbortSignal): Promise<MeshoptDecoder | null> {
    if (!this.meshoptProbe) {
      this.meshoptProbe = import('three/addons/libs/meshopt_decoder.module.js')
        .then((module) => {
          if (this.disposed) return null;
          this.meshoptDecoder = module.MeshoptDecoder as MeshoptDecoder;
          return this.meshoptDecoder;
        })
        .catch(() => {
          /* Meshopt support is optional; uncompressed assets load regardless. */
          return null;
        });
    }
    return this.meshoptProbe.then((decoder) => {
      throwIfAborted(signal);
      return decoder;
    });
  }
}

/**
 * Builds a `MeshPhysicalNodeMaterial` equivalent to a glTF-produced
 * `MeshStandardMaterial`.
 *
 * `Material.copy()` is not usable here: the node material does not extend
 * `MeshStandardMaterial`, so `copy` would skip exactly the map slots that
 * matter. The slots are therefore transferred explicitly. Textures are shared,
 * not cloned — the loader owns them either way.
 */
export function toPhysicalNodeMaterial(
  source: THREE.MeshStandardMaterial,
): THREE.MeshPhysicalNodeMaterial {
  const target = new THREE.MeshPhysicalNodeMaterial();

  target.name = source.name;
  target.color.copy(source.color);
  target.map = source.map;
  target.roughness = source.roughness;
  target.roughnessMap = source.roughnessMap;
  target.metalness = source.metalness;
  target.metalnessMap = source.metalnessMap;
  target.normalMap = source.normalMap;
  target.normalScale.copy(source.normalScale);
  target.normalMapType = source.normalMapType;
  target.aoMap = source.aoMap;
  target.aoMapIntensity = source.aoMapIntensity;
  target.emissive.copy(source.emissive);
  target.emissiveMap = source.emissiveMap;
  target.emissiveIntensity = source.emissiveIntensity;
  target.alphaMap = source.alphaMap;
  target.alphaTest = source.alphaTest;
  target.transparent = source.transparent;
  target.opacity = source.opacity;
  target.side = source.side;
  target.flatShading = source.flatShading;
  target.wireframe = source.wireframe;
  target.vertexColors = source.vertexColors;
  target.depthWrite = source.depthWrite;
  target.envMapIntensity = source.envMapIntensity;
  target.lightMap = source.lightMap;
  target.lightMapIntensity = source.lightMapIntensity;
  target.bumpMap = source.bumpMap;
  target.bumpScale = source.bumpScale;
  target.displacementMap = source.displacementMap;
  target.displacementScale = source.displacementScale;
  target.displacementBias = source.displacementBias;

  // KHR_materials_* extensions land on the standard material as loose
  // properties; carry over the ones the physical model understands.
  const extended = source as unknown as Record<string, unknown>;
  if (typeof extended.ior === 'number') target.ior = extended.ior;
  if (typeof extended.clearcoat === 'number') target.clearcoat = extended.clearcoat;
  if (typeof extended.transmission === 'number') target.transmission = extended.transmission;
  if (typeof extended.sheen === 'number') target.sheen = extended.sheen;
  if (typeof extended.iridescence === 'number') target.iridescence = extended.iridescence;

  return target;
}

/**
 * Rewrites every quantised vertex attribute of `geometry` as plain float32.
 *
 * Call this before baking a transform into loaded geometry. It is not an
 * optimisation — it is a correctness fix, and the bug it prevents is silent and
 * spectacular.
 *
 * `scripts/optimize-assets.mjs` Meshopt-encodes the scene dressing, and Meshopt
 * encoding quantises: positions arrive as an `Int16Array` with
 * `normalized = true`, holding values in [-1, 1], and the model's real size
 * lives in the glTF node's scale. `BufferAttribute.applyMatrix4` reads through
 * `getX/getY/getZ`, which de-normalise, and writes back through `setXYZ`, which
 * does not — so baking a node scale of 4 into the geometry tries to store 4.2 in
 * a buffer whose representable range stops at 1, and every vertex outside the
 * unit cube is clamped onto its faces.
 *
 * A tree came out of that as a hollow box of ribbons standing where the tree
 * should be — the exact silhouette of a mesh flattened onto a cube. Worth
 * knowing as a failure mode, because nothing errors: the model loads, the draw
 * succeeds, and only the shape is wrong.
 *
 * De-normalising also makes `mergeGeometries` work across an asset's parts,
 * since it refuses inputs whose attributes disagree about normalisation.
 */
export function dequantiseGeometry(geometry: THREE.BufferGeometry): void {
  for (const [name, attribute] of Object.entries(geometry.attributes)) {
    const source = attribute as THREE.BufferAttribute;
    if (!source.normalized && source.array instanceof Float32Array) continue;

    const items = source.itemSize;
    const values = new Float32Array(source.count * items);
    for (let i = 0; i < source.count; i++) {
      for (let c = 0; c < items; c++) values[i * items + c] = source.getComponent(i, c);
    }
    geometry.setAttribute(name, new THREE.BufferAttribute(values, items));
  }
}

/** Adds every texture referenced by `material` to `into`. */
function collectTextures(material: THREE.Material, into: Set<THREE.Texture>): void {
  const record = material as unknown as Record<string, unknown>;
  for (const key in record) {
    const value = record[key];
    if (value && (value as THREE.Texture).isTexture) {
      into.add(value as THREE.Texture);
    }
  }
}

function abortError(reason: unknown, fallback: string): Error {
  if (reason instanceof Error) return reason;
  const error = new Error(typeof reason === 'string' ? reason : fallback);
  error.name = 'AbortError';
  return error;
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError(signal.reason, 'Asset load aborted');
}

function waitForAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
  abort: (reason: Error) => void,
  /** Starts the deadline. Omitted means "start it now". See `enqueue`. */
  started?: Promise<void>,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const armTimer = () => {
      if (settled) return;
      timer = setTimeout(() => {
        const error = new Error(`Asset load timed out after ${timeoutMs} ms`);
        error.name = 'TimeoutError';
        abort(error);
        fail(error);
      }, Math.max(1, timeoutMs));
    };
    if (started) void started.then(armTimer, armTimer);
    else armTimer();

    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    };
    const succeed = (value: T) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };
    const fail = (reason: unknown) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(reason);
    };
    const onAbort = () => {
      const error = abortError(signal?.reason, 'Asset load aborted');
      abort(error);
      fail(error);
    };

    if (signal?.aborted) {
      onAbort();
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });
    promise.then(succeed, fail);
  });
}

