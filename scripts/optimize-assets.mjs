#!/usr/bin/env node
/**
 * optimize-assets.mjs — decimates and compresses the scene-dressing models.
 *
 * Poly Haven publishes film-quality geometry. That is the right choice for their
 * audience and the wrong one for ours: `jacaranda_tree` arrives as a 199 MB `.bin`
 * of raw vertex data, and it is a background tree seen from forty metres. The six
 * dressing models between them are 180 MB, against 57 MB for the entire rest of
 * the project — committing that would triple the repository to render silhouettes.
 *
 * So the fetch step keeps its job (get the authoritative source, verify it
 * against the publisher's manifest) and this step does what a game pipeline
 * does: decimate to the detail the camera can resolve, weld, strip what is not
 * read, and re-encode through Meshopt. `AssetLoader` already advertises the
 * Meshopt decoder — its own comment says re-exporting through gltf-transform is
 * expected and should not require a code change — so nothing at runtime changes.
 *
 * Only the raw downloads are large, and only the `.glb` outputs are committed;
 * `.gitignore` carries the split. Both are reproducible:
 *
 *   node scripts/fetch-assets.mjs      # authoritative source, verified
 *   node scripts/optimize-assets.mjs   # what actually ships
 *
 * Usage:
 *   node scripts/optimize-assets.mjs                       # build anything missing
 *   node scripts/optimize-assets.mjs --force               # rebuild everything
 *   node scripts/optimize-assets.mjs --force island_tree_  # rebuild matching slugs
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { availableParallelism, tmpdir } from 'node:os';
import { isMainThread, parentPort, Worker } from 'node:worker_threads';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodeIO, TextureChannel } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRTextureBasisu } from '@gltf-transform/extensions';
import {
  cloneDocument,
  dedup,
  getTextureChannelMask,
  join as joinDocument,
  joinPrimitives,
  listTextureSlots,
  meshopt,
  prune,
  simplifyPrimitive,
  textureCompress,
  weld,
} from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { encodeToKTX2 } from 'babylonpress-ktx2-encoder';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
/**
 * Where the raw downloads live, and deliberately not under `public/`.
 *
 * Vite copies the whole of `public/` into `dist/` verbatim, so while the sources
 * sat there this script's entire purpose was being undone one directory up: the
 * build shipped 818 MB, of which 784 MB was raw scan geometry that no frame
 * draws and no URL points at. Only the six directories the runtime actually
 * fetches — the ship, the buoy, the barrel, the two silhouette rocks, and
 * `dressing/` — are left in `public/`.
 */
const MODELS = join(ROOT, 'assets', 'source', 'models');
const OUT = join(ROOT, 'public', 'models', 'dressing');

const FORCE = process.argv.includes('--force');
/**
 * Bare arguments narrow the build to the slugs containing them, which is what
 * makes tuning a budget a ten-second loop instead of a four-minute one.
 */
const ONLY = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const UASTC_RDO_ARG = process.argv.find((arg) => arg.startsWith('--uastc-rdo='));
const UASTC_RDO_QUALITY = UASTC_RDO_ARG ? Number(UASTC_RDO_ARG.slice('--uastc-rdo='.length)) : 1;
if (!Number.isFinite(UASTC_RDO_QUALITY) || UASTC_RDO_QUALITY < 0 || UASTC_RDO_QUALITY > 10) {
  throw new Error('--uastc-rdo must be a number from 0 to 10; received ' + (UASTC_RDO_ARG ?? '(default)'));
}
const NORMAL_TEXTURE_SCALE = 0.5;

const MANIFEST_PATH = join(OUT, 'dressing-manifest.json');
const KTX2_CACHE = join(tmpdir(), 'web-ocean-3d-ktx2-cache-v1');
const KTX2_SIGNATURE = new Uint8Array([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);
const KTX2_WORKERS = Math.min(4, Math.max(1, availableParallelism() - 1));

async function decodeImage(buffer) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    width: info.width,
    height: info.height,
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  };
}

async function startKTX2Worker() {
  if (!parentPort) return;
  parentPort.on('message', async (job) => {
    try {
      const encoded = await encodeToKTX2(new Uint8Array(job.image), {
        ...job.options,
        imageDecoder: decodeImage,
      });
      const data = new Uint8Array(encoded);
      parentPort.postMessage({ id: job.id, ok: true, data }, [data.buffer]);
    } catch (error) {
      parentPort.postMessage({
        id: job.id,
        ok: false,
        error: error instanceof Error ? { message: error.message, name: error.name, stack: error.stack } : String(error),
      });
    }
  });
}

class KTX2WorkerPool {
  constructor(size) {
    this.workers = [];
    this.idle = [];
    this.queue = [];
    this.pending = new Map();
    this.nextId = 1;

    for (let i = 0; i < size; i++) {
      const worker = new Worker(new URL(import.meta.url), { type: 'module' });
      worker.on('message', (message) => this.handleMessage(worker, message));
      worker.on('error', (error) => this.handleError(worker, error));
      this.workers.push(worker);
      this.idle.push(worker);
    }
  }

  encode(image, options) {
    return new Promise((resolve, reject) => {
      this.queue.push({
        id: this.nextId++,
        image: new Uint8Array(image).slice(),
        options,
        resolve,
        reject,
      });
      this.dispatch();
    });
  }

  dispatch() {
    while (this.idle.length > 0 && this.queue.length > 0) {
      const worker = this.idle.shift();
      const job = this.queue.shift();
      this.pending.set(job.id, { worker, resolve: job.resolve, reject: job.reject });
      const image = job.image.buffer;
      worker.postMessage({ id: job.id, image, options: job.options }, [image]);
    }
  }

  handleMessage(worker, message) {
    const entry = this.pending.get(message.id);
    if (!entry) return;
    this.pending.delete(message.id);
    this.idle.push(worker);
    if (message.ok) {
      entry.resolve(new Uint8Array(message.data));
    } else {
      const error = new Error(typeof message.error === 'string' ? message.error : message.error.message);
      if (typeof message.error === 'object') {
        error.name = message.error.name ?? 'Error';
        error.stack = message.error.stack ?? error.stack;
      }
      entry.reject(error);
    }
    this.dispatch();
  }

  handleError(worker, error) {
    this.workers = this.workers.filter((candidate) => candidate !== worker);
    this.idle = this.idle.filter((candidate) => candidate !== worker);
    for (const [id, entry] of this.pending) {
      if (entry.worker !== worker) continue;
      this.pending.delete(id);
      entry.reject(error);
    }
    void worker.terminate();
    if (this.workers.length === 0) {
      for (const job of this.queue.splice(0)) job.reject(error);
    }
    this.dispatch();
  }

  async close() {
    await Promise.all(this.workers.map((worker) => worker.terminate()));
    this.workers = [];
    this.idle = [];
  }
}

/**
 * What counts as a scan's ground plate. See `stripGroundPlate`.
 *
 * Measured, not guessed: across the three trees the plate components run 0.07 to
 * 0.12 m thick over footprints of 0.25 to 1.57 m — a thickness-to-footprint
 * ratio between 0.05 and 0.28 — and every one of them bottoms out within a
 * centimetre of the model's base. The nearest *legitimate* geometry is a low
 * limb on a windswept scan at 0.53 x 0.12 x 0.60, which is a ratio of 0.23 and
 * sits 2.4 cm up. The margin is thin, which is why the rule needs all three
 * tests and why it is applied only where `ASSETS` names a `plate`.
 */
const PLATE_FLATNESS = 0.22;
const PLATE_GROUNDED = 0.03;
const PLATE_MIN_FOOTPRINT = 0.05;

/**
 * Per-asset simplification ratio and texture budget.
 *
 * `ratio` is the fraction of triangles to keep. The numbers are not uniform
 * because the assets are not: a tree's silhouette is carried by thousands of
 * separate leaf cards and collapses badly, while a rock is one closed surface
 * that decimates almost freely. `error` is the maximum allowed deviation as a
 * fraction of the mesh's extent — the simplifier stops early rather than exceed
 * it, so a low `ratio` on a shape that cannot take it degrades gracefully
 * instead of shredding.
 *
 * `texture` caps the longest edge. These are 1k downloads; at the distances they
 * are placed, 512 is generous and halves the payload again.
 *
 * Three optional keys handle the things a single ratio cannot:
 *
 *  - `parts` overrides the budget for the primitives whose material name
 *    matches, because a tree is two problems in one file — see below.
 *  - `plate` names the primitive that carries the scan's ground contact, so
 *    `stripGroundPlate` can cut it off.
 *  - `opaque` promotes materials that were exported `BLEND` without an alpha
 *    channel back to `OPAQUE`.
 */
const ASSETS = [
  // Trees are the worst case for a simplifier, and the reason is measurable
  // rather than aesthetic. `error` is a fraction of the *mesh radius*, so on a
  // 2.4 m canopy an error of 0.06 licenses 14 cm of deviation — and a leaf on
  // these models is about 4 cm across. Every leaf therefore fits inside the
  // error bound, the simplifier collapses whole leaves rather than simplifying
  // them, and the first tree through this script came out with 2% of its canopy
  // and the silhouette of a tree in February. The fix is not a higher ratio: it
  // is an error bound smaller than a leaf. At 0.004 (about 1 cm) the collapses
  // that delete leaves are refused and the ones that flatten a leaf's interior
  // are allowed, so the canopy survives whatever ratio it is given.
  //
  // The ratio then buys canopy against frame time, and 0.04 is where that trade
  // was settled: 116 trees stand on the island, so every thousand triangles of
  // leaf is 116,000 triangles of scene. At 0.022/0.06 the canopy was 23k and
  // the tree was bare; at 0.06/0.004 it was 64k and the island alone was seven
  // million triangles; 0.04 is a full canopy for a little over four.
  //
  // The trunk has the opposite problem — one closed surface, no small features
  // worth protecting — so it keeps the aggressive budget it always had. Hence
  // `parts`: one file, two materials, two completely different jobs.
  //
  // `plate` is the other thing Poly Haven ships that a scene does not want.
  // These are photogrammetry scans, and the scan includes the patch of ground
  // the tree was standing on: a flat sheet welded under the trunk, which reads
  // as a white dinner plate at the foot of every tree on the island.
  { slug: 'pachira_aquatica_01', ratio: 0.15, error: 0.012, texture: 512 },
  {
    slug: 'island_tree_02',
    ratio: 0.03,
    error: 0.06,
    texture: 512,
    parts: [{ match: /_leaves$/, ratio: 0.04, error: 0.004 }],
    plate: /^island_tree_02$/,
    opaque: /_leaves$/,
  },
  // The jacaranda is the same problem at four times the size, and the size is
  // what changes the arithmetic. `error` is a fraction of the mesh radius, so on
  // an 11.8 m canopy the 1 cm tolerance the small trees need is 0.0008 — and at
  // that tolerance the simplifier refuses almost everything and the model comes
  // out at half a million triangles and 8.5 MB, for one background tree. What
  // this canopy actually needs is a *ratio* bound rather than an error bound:
  // it is the largest crown on the island and the only tree whose leaf clusters
  // are big enough to survive being merged into each other. It is also the one
  // tree here with no scanned ground plate under it.
  {
    slug: 'jacaranda_tree',
    ratio: 0.008,
    error: 0.08,
    texture: 512,
    parts: [{ match: /_leaves$/, ratio: 0.012, error: 0.006 }],
    opaque: /_leaves$/,
  },
  { slug: 'fern_02', ratio: 0.35, error: 0.01, texture: 512 },
  { slug: 'shrub_sorrel_01', ratio: 0.35, error: 0.01, texture: 512 },
  // Ground cover is the highest instance count in the project — six hundred
  // tufts on the island — so these are budgeted like the coral carpet is, by
  // what the count multiplies out to rather than by how the tuft looks alone.
  // Both also ship `alphaMode: BLEND` over a JPEG diffuse, which is the same
  // spurious declaration the tree leaves carry: nothing to blend, and the whole
  // kind pushed out of the opaque pass for it.
  { slug: 'grass_medium_01', ratio: 0.12, error: 0.004, texture: 512, opaque: /^grass_medium/ },
  { slug: 'grass_medium_02', ratio: 0.2, error: 0.004, texture: 512, opaque: /^grass_medium/ },

  // Coastline edges carry the shoreline's silhouette, so they keep a little more
  // than the rock masses do — a decimated edge reads as a bitten one.
  { slug: 'coast_line_01', ratio: 0.08, error: 0.01, texture: 1024 },
  { slug: 'coast_line_02', ratio: 0.08, error: 0.01, texture: 1024 },
  { slug: 'coast_land_rocks_03', ratio: 0.06, error: 0.012, texture: 1024 },
  // Loaded raw, straight out of `public/models`, until the LOD pass — it predates
  // this script. It is also the single largest instanced cost in the scene:
  // `rock_07` is placed 154 times between the island skirt and the reef, and at
  // full detail with no LOD chain at any distance it was the one hole left in
  // "LODs for every high-poly model".
  { slug: 'rock_07', ratio: 0.35, error: 0.012, texture: 1024 },
  { slug: 'coast_rocks_01', ratio: 0.06, error: 0.012, texture: 1024 },
  { slug: 'coast_rocks_03', ratio: 0.06, error: 0.012, texture: 1024 },

  { slug: 'anthurium_botany_01', ratio: 0.3, error: 0.01, texture: 512 },
  { slug: 'calathea_orbifolia_01', ratio: 0.3, error: 0.01, texture: 512 },

  // Pirate cove. `ship_pinnace` and `cannon_01` are rigged in the source; the
  // rig is stripped by `prune` because nothing here animates them, and a boat
  // hauled up a beach does not need to.
  { slug: 'ship_pinnace', ratio: 0.2, error: 0.008, texture: 1024 },
  { slug: 'modular_wooden_pier', ratio: 0.25, error: 0.008, texture: 1024 },
  { slug: 'cannon_01', ratio: 0.25, error: 0.006, texture: 512 },
  { slug: 'wooden_barrels_01', ratio: 0.3, error: 0.008, texture: 512 },
  { slug: 'wooden_lantern_01', ratio: 0.35, error: 0.006, texture: 512 },
  { slug: 'wooden_crate_02', ratio: 0.3, error: 0.008, texture: 512 },

  // Pirate remains. Held to a tighter error than the rocks: these are handled
  // objects seen from a couple of metres, where a collapsed hilt or a faceted
  // jug is the thing the eye lands on.
  { slug: 'antique_estoc', ratio: 0.2, error: 0.004, texture: 512 },
  { slug: 'jug_01', ratio: 0.25, error: 0.005, texture: 512 },
  { slug: 'wooden_bucket_01', ratio: 0.3, error: 0.006, texture: 512 },
  { slug: 'modular_fort_01', ratio: 0.15, error: 0.01, texture: 1024 },

  { slug: 'treasure_chest', ratio: 0.3, error: 0.006, texture: 1024 },
  { slug: 'wooden_crate_01', ratio: 0.3, error: 0.008, texture: 512 },
  { slug: 'lambis_shell', ratio: 0.25, error: 0.008, texture: 512 },

  // ---- The reef (Sketchfab; see the SKETCHFAB manifest in fetch-assets.mjs) --
  //
  // The Smithsonian's three corals are 100k-face photogrammetry of a single
  // colony each, which is the same kind of asset as Poly Haven's rocks and takes
  // the same kind of decimation. They are also *small* — `stylaster_coral` is
  // 17 cm tall — because they are museum specimens; `Props` scales them to reef
  // heads. A tighter error than the rocks get, because a coral's whole
  // silhouette is its branching and that is exactly what a loose error eats.
  //
  // The budgets are set by instance count, not by how the model looks in
  // isolation. A coral head is placed sixty to ninety times and the reef already
  // carries ninety 15k rocks, so a 30k coral would double the scene's triangle
  // count to add detail nobody can resolve: these are seen from five metres and
  // up through water that has thrown away most of the contrast by ten. Three to
  // four thousand each is what survives that, and the error bounds are small in
  // absolute terms — a millimetre or two on a 17 cm specimen — so what is lost
  // is surface detail rather than the branching, which is the whole silhouette.
  { slug: 'stylaster_coral', ratio: 0.035, error: 0.01, texture: 512 },
  { slug: 'seriatopora_coral', ratio: 0.035, error: 0.01, texture: 512 },
  { slug: 'goniastrea_coral', ratio: 0.035, error: 0.012, texture: 512 },
  // The carpet, and therefore the one asset here whose budget is set by how many
  // of it there are. `Props` plants four hundred clumps of three forms each, so
  // a thousand triangles per form is fifteen hundred thousand triangles of
  // coral — and these are 1.5 m heads seen through water that has taken most of
  // the contrast out by ten metres. Decimated to roughly a quarter of that: the
  // shapes are convex blobs and tubes, which is the case simplification handles
  // best, and the silhouette survives where an equivalent cut to the branching
  // corals below would not.
  { slug: 'soft_coral_set', ratio: 0.12, error: 0.01, texture: 512 },
  // 2.5k faces for a fish 20 cm long. Nothing to decimate: `ratio: 1` keeps it
  // whole and the entry exists for `static` and the texture budget. See
  // `static` for why the rig goes.
  { slug: 'emperor_angelfish', ratio: 1, error: 0.001, texture: 512, static: true },

  // ---- The island's palms (Sketchfab) --------------------------------------
  //
  // Both replace `Remains.ts`'s procedural palm. `palm_coconut` is already
  // game-ready at 6.6k and needs no decimation at LOD0; `palm_tall` is 27k for a
  // 4.5 m tree, which is four times what it needs at the density it is planted.
  // Botanical trees, replacing the two Poly Haven scans that read as dead
  // scrub. Same two-budget treatment the scans get — the leaves need an error
  // bound smaller than a leaf or the simplifier deletes them wholesale — and
  // the flowers are held to the leaves' budget for the same reason.
  //
  // `cutOut` rather than `opaque`, which is the whole difference between these
  // and the scans above: a scanned leaf is geometry over a JPEG and an authored
  // one is a card over a PNG cut-out. See `cutOutFoliage`.
  {
    slug: 'tree_poinciana',
    ratio: 0.05,
    error: 0.02,
    texture: 1024,
    parts: [{ match: /leaf|flower/i, ratio: 0.12, error: 0.003 }],
    cutOut: /leaf|flower/i,
    atlas: /leaf|flower|bud/i,
    lods: [0.3, 0.09],
  },
  {
    slug: 'tree_orchid',
    ratio: 0.06,
    error: 0.02,
    texture: 1024,
    parts: [{ match: /leaf/i, ratio: 0.14, error: 0.003 }],
    cutOut: /leaf/i,
    atlas: /leaf|flower|bud/i,
    lods: [0.3, 0.09],
  },

  // Closed rock, replacing the coastal facades on the island — see the note in
  // the SKETCHFAB manifest. Decimated like the Poly Haven rocks are: one closed
  // surface with no small features to protect takes it freely.
  { slug: 'rock_slab_a', ratio: 0.16, error: 0.012, texture: 1024, ground: true, lods: [0.3, 0.09] },
  { slug: 'rock_slab_b', ratio: 0.16, error: 0.012, texture: 1024, ground: true, lods: [0.3, 0.09] },
  // `ground` matters most here: the scan rig left this one's origin 1.8 km above
  // the rock, so without the correction every instance is placed underground.
  { slug: 'rock_boulder', ratio: 0.12, error: 0.012, texture: 1024, ground: true, lods: [0.3, 0.09] },

  { slug: 'palm_coconut', ratio: 1, error: 0.001, texture: 1024, lods: [0.4, 0.14] },
  { slug: 'palm_tall', ratio: 0.35, error: 0.004, texture: 1024, lods: [0.4, 0.14] },
];

/**
 * Assets that get a LOD chain, and the ratios of each level against LOD0.
 *
 * Applied on top of `ASSETS` so the LOD0 budgets above stay readable as one
 * table. Only the kinds worth the extra files are listed: a 1.4k-triangle jug
 * has nothing to give back, and every entry here costs two more downloads and
 * two more draw calls' worth of pipeline.
 *
 * The ratios are deliberately aggressive. LOD1 is what the island wears from the
 * play area — 1.4 km away, where a 40 m cliff is fifty pixels tall — and LOD2 is
 * for the far half of the island seen from the same place. What matters at those
 * sizes is the silhouette and the average colour, and both survive a cut to a
 * tenth far better than the same cut survives at arm's length.
 */
const LOD_RATIOS = {
  island_tree_02: [0.3, 0.09],
  jacaranda_tree: [0.3, 0.09],
  pachira_aquatica_01: [0.35, 0.12],
  anthurium_botany_01: [0.35, 0.12],
  calathea_orbifolia_01: [0.35, 0.12],
  fern_02: [0.4],
  shrub_sorrel_01: [0.4],
  grass_medium_01: [0.4, 0.15],
  grass_medium_02: [0.4, 0.15],

  coast_line_01: [0.25, 0.07],
  coast_line_02: [0.25, 0.07],
  coast_land_rocks_03: [0.25, 0.07],
  coast_rocks_01: [0.25, 0.07],
  coast_rocks_03: [0.25, 0.07],
  rock_07: [0.3, 0.09],


  ship_pinnace: [0.3, 0.1],
  modular_wooden_pier: [0.3, 0.1],
  modular_fort_01: [0.35, 0.12],
  treasure_chest: [0.3, 0.1],
  cannon_01: [0.35, 0.12],
  wooden_barrels_01: [0.35, 0.12],

  // The reef is the one place the camera gets *close* to a high instance count,
  // so its LOD1 is gentler and there is no LOD2: a coral head is either in the
  // patch you are swimming through or invisible in the haze, with very little in
  // between.
  stylaster_coral: [0.4],
  seriatopora_coral: [0.3],
  goniastrea_coral: [0.3],
  soft_coral_set: [0.45],
};

// These species are replaced by billboard imposters before LOD_SWITCH_METRES
// reaches their first mesh LOD. Keeping the unused chains costs downloads and
// render-object compilation without ever being drawn.
const IMPOSTER_ONLY_ASSETS = new Set([
  'fern_02',
  'shrub_sorrel_01',
  'calathea_orbifolia_01',
  'anthurium_botany_01',
  'grass_medium_01',
  'grass_medium_02',
  'tree_orchid',
  'tree_poinciana',
  'jacaranda_tree',
  'island_tree_02',
  'palm_coconut',
  'palm_tall',
  'pachira_aquatica_01',
]);

for (const asset of ASSETS) {
  if (IMPOSTER_ONLY_ASSETS.has(asset.slug)) {
    asset.lods = [];
  } else if (asset.lods === undefined && LOD_RATIOS[asset.slug]) {
    asset.lods = LOD_RATIOS[asset.slug];
  }
}

const bytes = (n) =>
  n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

async function main() {
  await MeshoptDecoder.ready;
  await MeshoptEncoder.ready;
  await MeshoptSimplifier.ready;

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.decoder': MeshoptDecoder,
      'meshopt.encoder': MeshoptEncoder,
    });

  mkdirSync(OUT, { recursive: true });
  await mkdir(KTX2_CACHE, { recursive: true });
  const ktx2Pool = new KTX2WorkerPool(KTX2_WORKERS);
  const ktx2Stats = { converted: 0, cacheHits: 0, cacheMisses: 0, sharedJobs: 0, bytes: 0 };
  const ktx2Jobs = new Map();

  let inTotal = 0;
  let outTotal = 0;
  let built = 0;
  let failed = 0;

  for (const asset of ASSETS) {
    if (ONLY.length && !ONLY.some((needle) => asset.slug.includes(needle))) continue;

    // Poly Haven ships a `.gltf` plus a texture folder; Sketchfab hands back a
    // single self-contained `.glb`. Both are just "the authoritative download"
    // as far as this script is concerned.
    const candidates = [
      join(MODELS, asset.slug, `${asset.slug}_1k.gltf`),
      join(MODELS, asset.slug, `${asset.slug}.glb`),
    ];
    const dest = join(OUT, `${asset.slug}.glb`);
    // Sketchfab source downloads require a user token and are intentionally
    // not fetched by fetch-assets.mjs. When the committed dressing output is
    // present it is used as a local input so the pipeline can still run over
    // the shipped set; a clean checkout without that output keeps the normal
    // missing-source error.
    //
    // **This fallback is not regenerative, and calling it reproducible would be
    // a lie.** The destination is already the pipeline's own output: without
    // --force the build skips it because the file exists, and with --force it
    // feeds committed KTX2 bytes back through Sharp-based atlas and WebP
    // transforms that expect a decodable source. Re-encoding an encoded texture
    // is lossy at best and a hard failure at worst. Regenerating these assets
    // for real requires the token and the original Sketchfab downloads; this
    // path exists so that the *other* assets in a run are not blocked by them.
    const source = candidates.find((path) => existsSync(path)) ?? (existsSync(dest) ? dest : candidates[0]);

    if (!existsSync(source)) {
      console.error(`  MISSING  ${asset.slug} — run: node scripts/fetch-assets.mjs`);
      failed += 1;
      continue;
    }
    if (existsSync(dest) && !FORCE) {
      console.log(`  skip     ${asset.slug}.glb (${bytes(statSync(dest).size)})`);
      outTotal += statSync(dest).size;
      continue;
    }
    const sourceDirectory = join(MODELS, asset.slug);
    const sourceSize = existsSync(sourceDirectory) ? directorySize(sourceDirectory) : statSync(source).size;

    try {
      const document = await io.read(source);
      const before = countTriangles(document);

      // Welding first is what makes decimation work at all: an unwelded mesh
      // has no shared edges, so every triangle is an island and the simplifier
      // has nothing to collapse. It is also what makes `stripGroundPlate`
      // possible, since a connected component is not a meaningful idea until
      // the coincident vertices have been joined.
      // Before the weld, so the welder is not asked to reconcile joint weights
      // that are about to be deleted.
      if (asset.static) makeStatic(document);

      await document.transform(weld());

      const plateTris = stripGroundPlate(document, asset);
      simplifyParts(document, asset);
      const opaqued = promoteToOpaque(document, asset);
      const masked = cutOutFoliage(document, asset);
      const atlased = await atlasFoliage(document, asset);
      // Last of the geometry passes, because it measures the model: run it
      // before `stripGroundPlate` and it would centre on a slab that is about to
      // be cut, and before `simplifyParts` it would measure vertices that are
      // about to move.
      const shift = groundModel(document, asset);

      await document.transform(
        dedup(),
        textureCompress({ encoder: sharp, targetFormat: 'webp', resize: [asset.texture, asset.texture] }),
        // After decimation, whole primitives and their materials can end up
        // unreferenced. Pruning before the encoder means the earlier passes
        // decide what is dead rather than this one guessing.
        prune(),
      );
      await resizeNormalMaps(document, Math.max(64, Math.floor(asset.texture * NORMAL_TEXTURE_SCALE)));

      const after = countTriangles(document);
      const notes = [];
      if (plateTris > 0) notes.push(`-${plateTris} plate tris`);
      if (opaqued > 0) notes.push(`${opaqued} BLEND->OPAQUE`);
      if (masked > 0) notes.push(`${masked} BLEND->MASK`);
      if (atlased > 0) notes.push(`${atlased} foliage materials atlased`);
      if (shift) notes.push(`grounded by [${shift.map((v) => v.toFixed(2)).join(' ')}]`);

      // The LOD chain branches from the finished LOD0 document, *before* it is
      // Meshopt-encoded. Two reasons, and both were learned the hard way.
      //
      // Branching here rather than re-reading the source is what makes `ratio`
      // mean what it says: a level is a fraction of the triangles that ship at
      // LOD0, not a fraction of a 1.6-million-triangle download that a second
      // independent decimation pass would then compound against. Re-deriving
      // each level from the source gave `island_tree_02` a LOD1 at 9% when the
      // table asked for 30%.
      //
      // Before the encoder, because encoding quantises positions to normalised
      // int16 and simplifying quantised geometry bakes that error into every
      // level below it.
      for (let level = 1; level <= (asset.lods?.length ?? 0); level++) {
        const lodDest = join(OUT, `${asset.slug}_lod${level}.glb`);
        const lod = cloneDocument(document);

        // Sloppy, not edge collapse, and this is the whole reason the LOD chain
        // is worth having. The collapse pass that built LOD0 has already taken
        // everything the topology will give: rerunning it here reduced every
        // rock by four to one and every *tree* by almost nothing, because a
        // canopy is thousands of separate leaf islands with no interior edges to
        // collapse. `simplifySloppyPrimitive` drops the topology constraint and
        // reduces by spatial clustering instead, which is exactly the trade a
        // level seen from 1.4 km should be making — see its own note.
        const ratio = asset.lods[level - 1];
        for (const mesh of lod.getRoot().listMeshes()) {
          for (const primitive of mesh.listPrimitives()) {
            simplifySloppyPrimitive(primitive, ratio, 0.06);
          }
        }

        await lod.transform(
          // Half the texture budget per level. A tree at 1.4 km is fifty pixels
          // tall; a 512 map on it is a mip chain nobody reads past level four.
          textureCompress({
            encoder: sharp,
            targetFormat: 'webp',
            resize: [Math.max(64, asset.texture >> level), Math.max(64, asset.texture >> level)],
          }),
          prune(),
        );
        await resizeNormalMaps(
          lod,
          Math.max(64, Math.floor(asset.texture * NORMAL_TEXTURE_SCALE) >> level),
        );
        await compressDocumentTextures(lod, ktx2Pool, ktx2Stats, ktx2Jobs, asset);
        await lod.transform(meshopt({ encoder: MeshoptEncoder, level: 'high' }));
        await io.write(lodDest, lod);
        const lodSize = statSync(lodDest).size;
        outTotal += lodSize;
        notes.push(`lod${level} ${countTriangles(lod).toLocaleString('en-US')}t/${bytes(lodSize)}`);
      }

      // What the header of this file always claimed happened, and did not: the
      // encoder was registered with the I/O and the transform was never run, so
      // every model shipped as raw float32 vertex data. `AssetLoader` has
      // advertised the Meshopt decoder from the start, so this costs nothing at
      // runtime and is most of why the shipped set shrank while the canopies
      // got denser.
      await compressDocumentTextures(document, ktx2Pool, ktx2Stats, ktx2Jobs, asset);
      await document.transform(meshopt({ encoder: MeshoptEncoder, level: 'high' }));
      await io.write(dest, document);

      const destSize = statSync(dest).size;
      inTotal += sourceSize;
      outTotal += destSize;
      built += 1;

      console.log(
        `  build    ${asset.slug}.glb  ` +
          `${bytes(sourceSize)} -> ${bytes(destSize)}  ` +
          `${before.toLocaleString('en-US')} -> ${after.toLocaleString('en-US')} tris` +
          (notes.length ? `  (${notes.join(', ')})` : ''),
      );
    } catch (error) {
      console.error(`  ERROR    ${asset.slug}: ${error instanceof Error ? error.message : error}`);
      failed += 1;
    }
  }

  const manifest = { version: 1, basePath: '/models/dressing/', models: {} };
  for (const asset of ASSETS) {
    const base = join(OUT, asset.slug + '.glb');
    if (!existsSync(base)) continue;
    const lods = [{ level: 0, ratio: 1, url: manifest.basePath + asset.slug + '.glb' }];
    for (let level = 1; level <= (asset.lods?.length ?? 0); level++) {
      const lodPath = join(OUT, asset.slug + '_lod' + level + '.glb');
      if (!existsSync(lodPath)) continue;
      lods.push({
        level,
        ratio: asset.lods[level - 1],
        url: manifest.basePath + asset.slug + '_lod' + level + '.glb',
      });
    }
    manifest.models[asset.slug] = { lods };
  }
  await writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  await ktx2Pool.close();
  console.log('  manifest : ' + MANIFEST_PATH + ' (' + Object.keys(manifest.models).length + ' model(s))');
  console.log(
    '  KTX2     : ' +
      ktx2Stats.converted +
      ' texture use(s), ' +
      ktx2Stats.cacheMisses +
      ' encode(s), ' +
      ktx2Stats.cacheHits +
      ' cache hit(s), ' +
      bytes(ktx2Stats.bytes),
  );

  console.log('\nSummary');
  console.log(`  built    : ${built} file(s)`);
  if (inTotal > 0) console.log(`  source   : ${bytes(inTotal)}`);
  console.log(`  shipped  : ${bytes(outTotal)}`);
  if (failed > 0) {
    console.error(`  FAILED   : ${failed}`);
    process.exitCode = 1;
  }
}

/**
 * Strips the rig from a model that is going to be driven some other way.
 *
 * `emperor_angelfish` arrives skinned, with a 91-channel swim clip. Neither
 * survives contact with this project, and not because they are bad: three.js
 * has no skinned instancing, so a rigged fish is one draw call *per fish*, and
 * the school this mesh joins is two hundred of them in one. `src/scene/Fish.ts`
 * already animates a body by passing a travelling wave down it in the vertex
 * stage — the same closed form the procedural fish use — and that costs nothing
 * per instance. So the mesh is taken and the rig is dropped, deliberately.
 *
 * The bind pose is what remains, which for this model is the neutral swimming
 * pose the mesh was authored in.
 */
function makeStatic(document) {
  const root = document.getRoot();
  for (const animation of root.listAnimations()) animation.dispose();
  for (const skin of root.listSkins()) skin.dispose();
  for (const node of root.listNodes()) node.setSkin(null);
  for (const mesh of root.listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      for (const semantic of ['JOINTS_0', 'WEIGHTS_0', 'JOINTS_1', 'WEIGHTS_1']) {
        const attribute = primitive.getAttribute(semantic);
        if (attribute) primitive.setAttribute(semantic, null);
      }
    }
  }
}

/**
 * The budget that applies to one primitive: the asset's, unless a `parts` rule
 * claims its material by name.
 */
function budgetFor(asset, primitive) {
  const name = primitive.getMaterial()?.getName() ?? '';
  const override = (asset.parts ?? []).find((part) => part.match.test(name));
  return override ?? asset;
}

/**
 * Decimates every primitive against its own budget.
 *
 * `simplify()` is a document-wide transform and takes one ratio, which is the
 * right shape for a rock and the wrong one for a tree: the same call has to
 * flatten a trunk hard and barely touch a canopy. Driving `simplifyPrimitive`
 * directly is the whole difference — see the note on `error` in `ASSETS`.
 */
function simplifyParts(document, asset) {
  for (const mesh of document.getRoot().listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      const budget = budgetFor(asset, primitive);
      simplifyPrimitive(primitive, {
        simplifier: MeshoptSimplifier,
        ratio: budget.ratio,
        error: budget.error,
      });
    }
  }
}

/**
 * Decimates a primitive with meshopt's *sloppy* simplifier.
 *
 * The reason this exists is visible in the build log without it: the LOD chain
 * reduced every rock by four to one and every tree by almost nothing —
 * `island_tree_02` went 53,185 -> 46,324 -> 44,089, and `jacaranda_tree` moved
 * by thirty-four triangles across two levels. That is not the ratio being
 * ignored; it is edge collapse doing what it is defined to do. A canopy is
 * thousands of separate leaf islands with no shared edges between them, so there
 * are almost no interior edges to collapse, and the simplifier correctly refuses
 * to weld two leaves that merely happen to be adjacent in space.
 *
 * `simplifySloppy` drops the topology constraint entirely and reduces by spatial
 * clustering instead. It is the wrong tool for LOD0 — it does not preserve
 * boundaries, UV seams or the silhouette exactly — and the right one for a level
 * that exists to be seen from 1.4 km, which is the trade every game foliage
 * pipeline makes. It is used *only* for LOD levels; LOD0 is untouched.
 */
function simplifySloppyPrimitive(primitive, ratio, error) {
  const position = primitive.getAttribute('POSITION');
  const indices = primitive.getIndices();
  if (!position || !indices) return;

  const count = indices.getCount();
  const target = Math.max(3, Math.floor((count * ratio) / 3) * 3);
  if (target >= count) return;

  // `weld` leaves POSITION tightly packed float32, but the accessor is free not
  // to be, and handing meshopt a view over the wrong stride is silent garbage.
  let positions = position.getArray();
  if (!(positions instanceof Float32Array) || positions.length !== position.getCount() * 3) {
    positions = new Float32Array(position.getCount() * 3);
    const element = [0, 0, 0];
    for (let i = 0; i < position.getCount(); i++) {
      position.getElement(i, element);
      positions.set(element, i * 3);
    }
  }

  const source = new Uint32Array(count);
  for (let i = 0; i < count; i++) source[i] = indices.getScalar(i);

  // `null` is the vertex-lock mask: nothing here needs its border pinned, and
  // the argument sits between the stride and the target, so omitting it silently
  // passes the target as a lock and trips an assertion rather than a type error.
  const [simplified] = MeshoptSimplifier.simplifySloppy(source, positions, 3, null, target, error);
  if (simplified.length >= 3) indices.setArray(simplified);
}

/**
 * Cuts the scan's ground contact off the bottom of a model.
 *
 * Poly Haven's trees are photogrammetry, and a photogrammetry capture of a tree
 * includes the ground the tree was standing on: a flat sheet of scanned dirt
 * welded under the trunk, roughly a metre across on `island_tree_02`. Planted on
 * a hillside it reads as a white dinner plate at the foot of every tree, which
 * is one of the things that made the island look like objects sprinkled on a
 * dome rather than a place.
 *
 * Detection is by connected component rather than by a height-and-normal rule,
 * because the plate genuinely *is* a separate surface — the trunk does not share
 * an edge with it — and a rule stated in heights would have to be retuned per
 * asset and would still catch the low limbs that a windswept scan rests on the
 * ground. A component qualifies when it is flat (its height is under a quarter
 * of its footprint), grounded (its lowest point is within 3% of the model's
 * height of the model's own base) and broad enough to be scenery rather than a
 * leaf. Deleting it leaves the trunk open at the bottom, which is invisible:
 * `Props` seats trees with `sink`, so the cut is buried.
 *
 * Returns the number of triangles removed, so the build log can show it.
 */
function stripGroundPlate(document, asset) {
  if (!asset.plate) return 0;
  let removed = 0;

  for (const mesh of document.getRoot().listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      if (!asset.plate.test(primitive.getMaterial()?.getName() ?? '')) continue;

      const indices = primitive.getIndices();
      const position = primitive.getAttribute('POSITION');
      if (!indices || !position) continue;

      const count = indices.getCount();
      const parent = new Int32Array(position.getCount());
      for (let i = 0; i < parent.length; i++) parent[i] = i;
      const find = (x) => {
        while (parent[x] !== x) {
          parent[x] = parent[parent[x]];
          x = parent[x];
        }
        return x;
      };
      const union = (a, b) => {
        const ra = find(a);
        const rb = find(b);
        if (ra !== rb) parent[rb] = ra;
      };
      for (let i = 0; i < count; i += 3) {
        union(indices.getScalar(i), indices.getScalar(i + 1));
        union(indices.getScalar(i + 1), indices.getScalar(i + 2));
      }

      // Bounds per component, and the model's own base, in one pass.
      const components = new Map();
      const element = [0, 0, 0];
      let baseY = Infinity;
      let topY = -Infinity;
      for (let i = 0; i < count; i += 3) {
        const root = find(indices.getScalar(i));
        let box = components.get(root);
        if (!box) {
          box = { tris: 0, min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
          components.set(root, box);
        }
        box.tris += 1;
        for (let k = 0; k < 3; k++) {
          position.getElement(indices.getScalar(i + k), element);
          for (let d = 0; d < 3; d++) {
            if (element[d] < box.min[d]) box.min[d] = element[d];
            if (element[d] > box.max[d]) box.max[d] = element[d];
          }
        }
        if (box.min[1] < baseY) baseY = box.min[1];
        if (box.max[1] > topY) topY = box.max[1];
      }

      const height = topY - baseY;
      const doomed = new Set();
      for (const [root, box] of components) {
        const footprint = Math.max(box.max[0] - box.min[0], box.max[2] - box.min[2]);
        const thickness = box.max[1] - box.min[1];
        if (
          thickness < PLATE_FLATNESS * footprint &&
          box.min[1] < baseY + PLATE_GROUNDED * height &&
          footprint > PLATE_MIN_FOOTPRINT * height
        ) {
          doomed.add(root);
        }
      }
      if (doomed.size === 0) continue;

      const kept = [];
      for (let i = 0; i < count; i += 3) {
        if (doomed.has(find(indices.getScalar(i)))) {
          removed += 1;
          continue;
        }
        kept.push(indices.getScalar(i), indices.getScalar(i + 1), indices.getScalar(i + 2));
      }
      // The orphaned vertices stay in the buffer until `prune` sweeps them.
      indices.setArray(new Uint32Array(kept));
    }
  }

  return removed;
}

/**
 * Promotes `BLEND` materials that have no alpha to blend with back to `OPAQUE`.
 *
 * Every one of these trees declares its leaves `alphaMode: BLEND`, and every one
 * of them supplies the leaf colour as a **JPEG** — a format with no alpha
 * channel at all — with a base colour factor of 1. There is therefore nothing
 * for the blend to do, and the declaration is pure cost: `GLTFLoader` maps
 * `BLEND` to `transparent = true`, which moves the whole canopy out of the
 * opaque pass into the sorted one, where an instanced mesh is sorted as a single
 * object and the leaves of one tree cannot resolve against the leaves of the
 * next.
 *
 * Narrow on purpose. It only fires for materials the asset names, and it is not
 * an alpha-test conversion: these leaves are modelled geometry rather than cut
 * cards, so testing an absent alpha at 0.5 would erase them.
 */
function promoteToOpaque(document, asset) {
  if (!asset.opaque) return 0;
  let promoted = 0;
  for (const material of document.getRoot().listMaterials()) {
    if (!asset.opaque.test(material.getName())) continue;
    if (material.getAlphaMode() !== 'BLEND') continue;
    material.setAlphaMode('OPAQUE');
    promoted += 1;
  }
  return promoted;
}

/**
 * Turns `BLEND` foliage into alpha-tested `MASK` foliage.
 *
 * The opposite case to `promoteToOpaque`, and the reason both exist: whether a
 * `BLEND` declaration is spurious depends entirely on whether there is an alpha
 * channel behind it. Poly Haven's scans model each leaf as geometry over a JPEG,
 * so the blend does nothing and `OPAQUE` is free. The botanical trees model
 * their leaves as *cards* over a PNG cut-out, so the alpha is the leaf shape —
 * promote those to opaque and every tree grows rectangles.
 *
 * `MASK` is what a cut-out card actually wants. `BLEND` costs it depth writes,
 * which is what makes a canopy of instanced cards sort against itself at all:
 * without them three.js sorts the whole `InstancedMesh` as one object, so the
 * leaves of the near tree draw behind the leaves of the far one and the canopy
 * turns inside out as the camera moves. It also costs the shadow: a transparent
 * material is skipped by the depth pass, so a `BLEND` canopy casts the shadow of
 * its branches and nothing else. Alpha testing keeps both and is cheaper.
 *
 * The cutoff is 0.35 rather than 0.5 because these leaves are photographed with
 * soft edges — at 0.5 the outline erodes by a pixel or two all round, which on
 * a compound leaf is most of the leaflet.
 */
function cutOutFoliage(document, asset) {
  if (!asset.cutOut) return 0;
  let converted = 0;
  for (const material of document.getRoot().listMaterials()) {
    if (!asset.cutOut.test(material.getName())) continue;
    if (material.getAlphaMode() !== 'BLEND') continue;
    material.setAlphaMode('MASK');
    material.setAlphaCutoff(asset.cutOutAt ?? 0.35);
    converted += 1;
  }
  return converted;
}

const ATLAS_WIDTH = 2048;
const ATLAS_CELL_MAX = 512;
const ATLAS_PADDING = 4;

function align4(value) {
  return Math.max(4, Math.round(value / 4) * 4);
}

function nextPowerOfTwo(value) {
  return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

async function prepareAtlasImage(texture, width, height, normal) {
  const fallback = normal ? { r: 128, g: 128, b: 255, alpha: 1 } : { r: 0, g: 0, b: 0, alpha: 0 };
  const source = texture
    ? sharp(texture.getImage()).ensureAlpha().resize(width, height, { fit: 'fill' })
    : sharp({ create: { width, height, channels: 4, background: fallback } });
  const { data, info } = await source
    .extend({
      top: ATLAS_PADDING,
      bottom: ATLAS_PADDING,
      left: ATLAS_PADDING,
      right: ATLAS_PADDING,
      extendWith: 'copy',
    })
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

async function buildAtlas(entries, normal, layout = null) {
  const placements = layout
    ? layout.placements.map((placement) => ({ ...placement }))
    : [];
  let x = ATLAS_PADDING;
  let y = ATLAS_PADDING;
  let rowHeight = 0;

  if (!layout) {
    for (const entry of entries) {
      const sourceTexture = normal ? entry.material.getNormalTexture() : entry.material.getBaseColorTexture();
      const metadata = sourceTexture ? await sharp(sourceTexture.getImage()).metadata() : { width: 4, height: 4 };
      const sourceWidth = Math.max(4, metadata.width ?? 4);
      const sourceHeight = Math.max(4, metadata.height ?? 4);
      const scale = Math.min(1, ATLAS_CELL_MAX / Math.max(sourceWidth, sourceHeight));
      const width = align4(Math.max(4, sourceWidth * scale));
      const height = align4(Math.max(4, sourceHeight * scale));
      const outerWidth = width + ATLAS_PADDING * 2;
      const outerHeight = height + ATLAS_PADDING * 2;

      if (x + outerWidth > ATLAS_WIDTH - ATLAS_PADDING) {
        x = ATLAS_PADDING;
        y += rowHeight + ATLAS_PADDING;
        rowHeight = 0;
      }
      placements.push({ entry, x, y, width, height });
      x += outerWidth + ATLAS_PADDING;
      rowHeight = Math.max(rowHeight, outerHeight);
    }
  }

  const atlasHeight = layout?.height ?? nextPowerOfTwo(y + rowHeight + ATLAS_PADDING);
  const background = normal ? { r: 128, g: 128, b: 255, alpha: 1 } : { r: 0, g: 0, b: 0, alpha: 0 };
  const composites = [];
  for (const placement of placements) {
    const sourceTexture = normal ? placement.entry.material.getNormalTexture() : placement.entry.material.getBaseColorTexture();
    const prepared = await prepareAtlasImage(sourceTexture, placement.width, placement.height, normal);
    composites.push({
      input: prepared.data,
      raw: { width: prepared.width, height: prepared.height, channels: 4 },
      left: placement.x,
      top: placement.y,
    });
  }

  const image = await sharp({
    create: { width: ATLAS_WIDTH, height: atlasHeight, channels: 4, background },
  })
    .composite(composites)
    .png()
    .toBuffer();

  return { image, width: ATLAS_WIDTH, height: atlasHeight, placements };
}

/**
 * Atlases the existing UV-backed foliage maps. `palette()` is intentionally
 * not used here: glTF-Transform's palette transform only accepts materials
 * without TEXCOORD_0 and creates a palette for scalar factors, not image maps.
 * The explicit atlas below preserves the source material identity and then
 * `joinPrimitives`/`join` collapse the compatible foliage geometry.
 */
async function atlasFoliage(document, asset) {
  if (!asset.atlas) return 0;
  const root = document.getRoot();
  const materials = root
    .listMaterials()
    .filter((material) => asset.atlas.test(material.getName()) && material.getBaseColorTexture());
  if (materials.length < 2) return 0;

  const entries = materials.map((material) => ({ material }));
  const baseAtlas = await buildAtlas(entries, false);
  const hasNormal = entries.some((entry) => entry.material.getNormalTexture());
  // Normals must use exactly the base-colour layout. Repacking independently
  // would make the shared UVs point at the wrong normal image whenever the
  // source dimensions differ.
  const normalAtlas = hasNormal ? await buildAtlas(entries, true, baseAtlas) : null;

  const baseTexture = document.createTexture(`${asset.slug}_foliage_base_atlas`);
  const atlasMaterial = materials[0]
    .clone()
    .setName(`${asset.slug}_foliage_atlas`)
    .setBaseColorTexture(baseTexture);
  baseTexture
    .setImage(baseAtlas.image)
    .setMimeType('image/png')
    .setName(`${asset.slug}_foliage_base_atlas`);
  if (normalAtlas) {
    const normalTexture = document.createTexture(`${asset.slug}_foliage_normal_atlas`)
      .setImage(normalAtlas.image)
      .setMimeType('image/png');
    atlasMaterial.setNormalTexture(normalTexture);
  }
  atlasMaterial.setAlphaMode('MASK').setAlphaCutoff(asset.cutOutAt ?? 0.35);

  const placementByMaterial = new Map(
    baseAtlas.placements.map((placement) => [placement.entry.material, placement]),
  );
  for (const mesh of root.listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      const sourceMaterial = primitive.getMaterial();
      const placement = sourceMaterial ? placementByMaterial.get(sourceMaterial) : undefined;
      if (!placement) continue;
      const uv = primitive.getAttribute('TEXCOORD_0');
      if (!uv) throw new Error(`${asset.slug}: foliage primitive ${mesh.getName()} has no TEXCOORD_0`);

      const values = new Float32Array(uv.getCount() * 2);
      const element = [0, 0];
      const uOffset = placement.x + ATLAS_PADDING;
      const vOffset = baseAtlas.height - placement.y - ATLAS_PADDING - placement.height;
      for (let i = 0; i < uv.getCount(); i++) {
        uv.getElement(i, element);
        // The source leaf maps use REPEAT wrapping and contain UVs outside
        // [0, 1]. The atlas is one tile, so preserve that sampler behavior by
        // folding each coordinate into its fractional part.
        const u = element[0] - Math.floor(element[0]);
        const v = element[1] - Math.floor(element[1]);
        values[i * 2] = (uOffset + u * placement.width) / baseAtlas.width;
        values[i * 2 + 1] = (vOffset + v * placement.height) / baseAtlas.height;
      }
      primitive
        .setAttribute('TEXCOORD_0', document.createAccessor().setType('VEC2').setArray(values))
        .setMaterial(atlasMaterial);
    }
  }

  // Join any same-mesh group directly, then let glTF-Transform move compatible
  // sibling nodes into the destination node while accounting for their matrices.
  for (const mesh of root.listMeshes()) {
    const prims = mesh.listPrimitives().filter((primitive) => primitive.getMaterial() === atlasMaterial);
    if (prims.length < 2) continue;
    const joined = joinPrimitives(prims);
    for (const primitive of prims) mesh.removePrimitive(primitive);
    mesh.addPrimitive(joined);
  }
  await document.transform(
    joinDocument({
      filter: (node) => node.getMesh()?.listPrimitives().some((primitive) => primitive.getMaterial() === atlasMaterial) ?? false,
    }),
  );

  return materials.length;
}

const LEAF_CUTOUT_NAME = /leaf|foliage|flower|bud|fern|frond|palm|anthurium|calathea|shrub|grass/i;

function isLeafCutoutTexture(document, asset, texture, slots, channelMask) {
  if (!slots.includes('baseColorTexture') || !(channelMask & TextureChannel.A)) return false;

  const assetName = asset?.slug ?? '';
  return document
    .getRoot()
    .listMaterials()
    .filter((material) => material.getBaseColorTexture() === texture)
    .some(
      (material) =>
        material.getAlphaMode() === 'MASK' &&
        (LEAF_CUTOUT_NAME.test(assetName) ||
          LEAF_CUTOUT_NAME.test(material.getName()) ||
          LEAF_CUTOUT_NAME.test(texture.getName() ?? '')),
    );
}

async function resizeNormalMaps(document, maxDimension) {
  const textures = document
    .getRoot()
    .listTextures()
    .filter((texture) => listTextureSlots(texture).includes('normalTexture'));

  await Promise.all(
    textures.map(async (texture) => {
      const image = texture.getImage();
      if (!image) return;

      const metadata = await sharp(image).metadata();
      if (!metadata.width || !metadata.height || Math.max(metadata.width, metadata.height) <= maxDimension) return;

      const resized = await sharp(image)
        .resize({
          width: maxDimension,
          height: maxDimension,
          fit: 'inside',
          withoutEnlargement: true,
        })
        .png()
        .toBuffer();
      texture.setImage(resized).setMimeType('image/png');
    }),
  );
}

function isKTX2(data) {
  return data.byteLength >= KTX2_SIGNATURE.byteLength && KTX2_SIGNATURE.every((byte, index) => data[index] === byte);
}

function textureCacheKey(image, options) {
  return createHash('sha256')
    .update(image)
    .update(JSON.stringify(options))
    .digest('hex');
}

async function compressDocumentTextures(document, pool, stats, inFlight, asset) {
  const root = document.getRoot();
  const textures = root.listTextures();
  let converted = false;

  await Promise.all(
    textures.map(async (texture) => {
      if (texture.getMimeType() === 'image/ktx2') return;
      const image = texture.getImage();
      if (!image) return;

      const slots = listTextureSlots(texture);
      const channelMask = getTextureChannelMask(texture);
      const normal = slots.includes('normalTexture');
      const leafCutout = isLeafCutoutTexture(document, asset, texture, slots, channelMask);
      const uastc = normal || leafCutout;
      // Colour data is sRGB; everything that encodes *numbers* rather than
      // colour is linear. Normal maps store directions, metallic-roughness and
      // occlusion store coefficients — de-gamma any of them on read and a
      // stored roughness of 0.5 arrives as 0.21, which is a different material.
      const perceptual =
        !normal &&
        !slots.includes('metallicRoughnessTexture') &&
        !slots.includes('occlusionTexture');
      const options = {
        isKTX2File: true,
        isUASTC: uastc,
        isNormalMap: normal,
        isPerceptual: perceptual,
        // Must be set explicitly, and must match `isPerceptual`. This is the
        // flag that writes the transfer function into the file's own data
        // format descriptor, and it is what a loader reads to decide the
        // texture's colour space — `isPerceptual` alone only tunes the codec.
        // Left unset, every texture in the set was tagged sRGB, including all
        // 14 of the 21 maps in the pinnace that must not be: its hull and the
        // coast rocks rendered pale and over-shiny because their roughness and
        // occlusion were being de-gamma'd.
        isSetKTX2SRGBTransferFunc: perceptual,
        generateMipmap: true,
        enableRDO: uastc,
        rdoQualityLevel: uastc ? UASTC_RDO_QUALITY : undefined,
      };
      const key = textureCacheKey(image, options);

      let promise = inFlight.get(key);
      if (!promise) {
        promise = (async () => {
          await mkdir(KTX2_CACHE, { recursive: true });
          const cachedPath = join(KTX2_CACHE, key + '.ktx2');
          if (existsSync(cachedPath)) {
            const cached = new Uint8Array(await readFile(cachedPath));
            if (isKTX2(cached)) {
              stats.cacheHits += 1;
              return cached;
            }
          }

          const encoded = await pool.encode(image, options);
          await writeFile(cachedPath, encoded);
          stats.cacheMisses += 1;
          return encoded;
        })().finally(() => inFlight.delete(key));
        inFlight.set(key, promise);
      } else {
        stats.sharedJobs += 1;
      }

      const encoded = await promise;
      texture.setImage(encoded).setMimeType('image/ktx2');
      const uri = texture.getURI();
      if (uri && !uri.startsWith('data:')) texture.setURI(uri.replace(/\.[^./]+$/i, '.ktx2'));
      stats.converted += 1;
      stats.bytes += encoded.byteLength;
      converted = true;
    }),
  );

  if (converted) document.createExtension(KHRTextureBasisu).setRequired(true);
}

/**
 * Moves a model so it stands on y = 0, centred on x and z.
 *
 * `Props` places every instance by putting the model's origin on the terrain, so
 * where the origin sits inside the model is not a detail — it *is* the placement.
 * Most sources put it under the trunk or the base and this pass is a no-op for
 * them. Sketchfab's `rock_boulder` puts it 1.8 **kilometres** above the rock,
 * which is the author's scan rig showing through, and an instance of it lands
 * that far underground.
 *
 * The correction goes into the scene's root nodes rather than into the vertex
 * data, so it survives quantisation exactly and costs nothing: a translation on
 * a node is a translation on a node whatever the accessor beneath it is stored
 * as.
 */
function groundModel(document, asset) {
  if (!asset.ground) return null;

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const element = [0, 0, 0];

  const visit = (node, matrix) => {
    const world = multiply(matrix, node.getMatrix());
    const mesh = node.getMesh();
    if (mesh) {
      for (const primitive of mesh.listPrimitives()) {
        const position = primitive.getAttribute('POSITION');
        if (!position) continue;
        for (let i = 0; i < position.getCount(); i++) {
          position.getElement(i, element);
          for (let d = 0; d < 3; d++) {
            const v =
              world[d] * element[0] + world[4 + d] * element[1] + world[8 + d] * element[2] + world[12 + d];
            if (v < min[d]) min[d] = v;
            if (v > max[d]) max[d] = v;
          }
        }
      }
    }
    for (const child of node.listChildren()) visit(child, world);
  };

  const roots = [];
  for (const scene of document.getRoot().listScenes()) {
    for (const node of scene.listChildren()) {
      roots.push(node);
      visit(node, IDENTITY);
    }
  }
  if (!Number.isFinite(min[1])) return null;

  const shift = [-(min[0] + max[0]) / 2, -min[1], -(min[2] + max[2]) / 2];
  if (Math.hypot(...shift) < 1e-4) return null;

  for (const node of roots) {
    const t = node.getTranslation();
    node.setTranslation([t[0] + shift[0], t[1] + shift[1], t[2] + shift[2]]);
  }
  return shift;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Column-major 4x4 multiply, matching glTF's node matrices. */
function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

function countTriangles(document) {
  let n = 0;
  for (const mesh of document.getRoot().listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      const indices = primitive.getIndices();
      const position = primitive.getAttribute('POSITION');
      const count = indices ? indices.getCount() : (position?.getCount() ?? 0);
      n += Math.floor(count / 3);
    }
  }
  return n;
}

function directorySize(dir) {
  if (!existsSync(dir)) return 0;
  let total = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    total += entry.isDirectory() ? directorySize(path) : statSync(path).size;
  }
  return total;
}

if (isMainThread) {
  await main();
} else {
  await startKTX2Worker();
}
