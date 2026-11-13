#!/usr/bin/env node
/**
 * fetch-assets.mjs — reproducible asset downloader for web-ocean-3d.
 *
 * Downloads every third-party asset used by the demo into `public/`.
 * Every asset here is CC0 (Poly Haven, https://polyhaven.com/license) — see
 * ASSET_LICENSES.md at the repo root for the full per-file provenance table.
 *
 * Usage:
 *   node scripts/fetch-assets.mjs            # download anything missing, then verify
 *   node scripts/fetch-assets.mjs --force    # re-download everything
 *   node scripts/fetch-assets.mjs --verify   # verify on-disk files only, no network
 *
 * Idempotent: files that already exist with a non-zero size are skipped.
 * Exits non-zero if any download or verification step fails.
 *
 * No dependencies — Node built-ins only (global fetch, node:fs).
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
/**
 * Every `dir` in the manifests below is relative to the repository root, not to
 * `public/`, and which root an entry names is a decision rather than a detail.
 *
 * Vite copies the whole of `public/` into `dist/` verbatim. While the raw
 * downloads lived there, `scripts/optimize-assets.mjs` was being undone one
 * directory up — the build shipped 818 MB, 784 MB of it film-quality scan
 * geometry that no URL points at. So only what the runtime actually fetches
 * goes to `public/`, and the sources the optimiser eats go to `assets/source/`,
 * which nothing copies and `.gitignore` excludes.
 */
const DOWNLOAD_ROOT = ROOT;
/** Roots a `--verify` run walks when it has no manifest to check against. */
const VERIFY_ROOTS = ['public', 'assets/source'];
const API = 'https://api.polyhaven.com';
const SKETCHFAB_API = 'https://api.sketchfab.com/v3';

const args = new Set(process.argv.slice(2));
const FORCE = args.has('--force');
const VERIFY_ONLY = args.has('--verify');

/**
 * The asset manifest. `slug` values were all verified against
 * https://api.polyhaven.com/assets?t=models and ?t=hdris before being added here.
 *
 * Texture resolution notes:
 *  - The ship is the hero object, so its source is 2k; the build keeps 2k for
 *    the hull and drops the rest to 1k (see HERO_ASSETS in optimize-assets).
 *  - Props and rocks are background dressing at 1k to keep the payload small.
 *  - No HDRIs: the sky is procedural. See the note at the end of MANIFEST.
 */
const MANIFEST = [
  // ---- Hero: sailing ship -------------------------------------------------
  //
  // Source, like the dressing: `scripts/optimize-assets.mjs` (HERO_ASSETS)
  // builds the KTX2 + Meshopt `public/models/hero/*.glb` that actually ships.
  {
    kind: 'model',
    slug: 'dutch_ship_medium',
    res: '2k',
    dir: 'assets/source/models/dutch_ship_medium',
    note: 'Hero sailing ship',
  },

  // ---- Floating props -----------------------------------------------------
  { kind: 'model', slug: 'ocean_buoy', res: '1k', dir: 'assets/source/models/ocean_buoy', note: 'Floating marine buoy' },
  { kind: 'model', slug: 'barrel_03', res: '1k', dir: 'assets/source/models/barrel_03', note: 'Floating barrel' },

  // ---- Island silhouette --------------------------------------------------
  { kind: 'model', slug: 'rock_07', res: '1k', dir: 'assets/source/models/rock_07', note: 'Standalone rock' },


  // ---- Island and shore dressing -----------------------------------------
  //
  // These are the *source* downloads, not what ships. Poly Haven publishes film
  // -quality geometry — `jacaranda_tree` is a 199 MB `.bin` for a background tree —
  // so `scripts/optimize-assets.mjs` decimates and Meshopt-encodes them into
  // `public/models/dressing/*.glb`, and only those are committed. 177 MB of
  // source becomes 15 MB shipped.
  //
  // Almost all of these come from Poly Haven's `smugglers_cove` collection,
  // which is the same set the hero ship is from. That matters for more than
  // convenience: assets authored for one scene share a scale, a texel density
  // and a colour response, so they sit together without per-asset correction.
  // Mixing packs is how dressing ends up looking like dressing.
  //
  // 1k throughout. These are background geometry seen from tens of metres away;
  // the ship is the only thing that earns 2k.
  // Coastline. `coast_line_01/02` are the pieces that make a shore read as a
  // shore rather than as a hill meeting water: they are authored as *edges*,
  // with a wave-cut platform and a back slope, so they sit along a contour
  // instead of being another rock standing on one.
  { kind: 'model', slug: 'coast_line_01', res: '1k', dir: 'assets/source/models/coast_line_01', note: 'Shoreline edge' },
  { kind: 'model', slug: 'coast_line_02', res: '1k', dir: 'assets/source/models/coast_line_02', note: 'Shoreline edge' },
  { kind: 'model', slug: 'coast_land_rocks_03', res: '1k', dir: 'assets/source/models/coast_land_rocks_03', note: 'Shore rock mass' },
  { kind: 'model', slug: 'coast_rocks_01', res: '1k', dir: 'assets/source/models/coast_rocks_01', note: 'Shore rock cluster' },
  { kind: 'model', slug: 'coast_rocks_03', res: '1k', dir: 'assets/source/models/coast_rocks_03', note: 'Shore rock cluster' },
  //
  // `coastal_cliff_02`, `coastal_cliff_04` and `sand_rocks_small_01` used to be
  // here and are gone. All three are excellent scans that only read from one
  // direction — see the `rock_slab_a` note in the Sketchfab manifest below for
  // the measurement and the replacement.


  // ---- Tropical planting ---------------------------------------------------
  //
  // Poly Haven has no coconut palm, and there is no point pretending otherwise:
  // the tropical read comes from broadleaf shapes and density rather than from
  // the one silhouette everyone associates with it. `pachira_aquatica` above is
  // the closest tree they publish; these two are genuine tropical understorey
  // from the same cove collection.

  // ---- Pirate cove ---------------------------------------------------------
  //
  // All from `smugglers_cove`, which is what the collection is: the hero ship's
  // own set, so the pier, the cannon and the barrels share its scale, wood tone
  // and wear. That coherence is the whole reason to take dressing from one pack.
  { kind: 'model', slug: 'ship_pinnace', res: '1k', dir: 'assets/source/models/ship_pinnace', note: "Ship's boat, beached" },
  { kind: 'model', slug: 'modular_wooden_pier', res: '1k', dir: 'assets/source/models/modular_wooden_pier', note: 'Jetty on the island' },
  { kind: 'model', slug: 'cannon_01', res: '1k', dir: 'assets/source/models/cannon_01', note: 'Shore battery' },
  { kind: 'model', slug: 'wooden_barrels_01', res: '1k', dir: 'assets/source/models/wooden_barrels_01', note: 'Barrel stack' },
  { kind: 'model', slug: 'wooden_lantern_01', res: '1k', dir: 'assets/source/models/wooden_lantern_01', note: 'Lantern on the pier' },
  { kind: 'model', slug: 'wooden_crate_02', res: '1k', dir: 'assets/source/models/wooden_crate_02', note: 'Cargo crate' },

  // ---- Pirate remains ------------------------------------------------------
  //
  // A wreck site tells a story or it is litter, and the story needs objects a
  // person left behind rather than objects that washed up. Poly Haven has no
  // skeleton — that one is built procedurally, see `src/scene/Remains.ts`.
  { kind: 'model', slug: 'antique_estoc', res: '1k', dir: 'assets/source/models/antique_estoc', note: 'Sword, half-buried' },
  { kind: 'model', slug: 'jug_01', res: '1k', dir: 'assets/source/models/jug_01', note: 'Bottle / jug' },
  { kind: 'model', slug: 'wooden_bucket_01', res: '1k', dir: 'assets/source/models/wooden_bucket_01', note: 'Bucket' },
  // A ruin gives the island a landmark and a reason for the cannon to be where
  // it is. Modular in the source; one section is enough.
  { kind: 'model', slug: 'modular_fort_01', res: '1k', dir: 'assets/source/models/modular_fort_01', note: 'Ruined shore fort' },

  // ---- Underwater find ----------------------------------------------------
  { kind: 'model', slug: 'treasure_chest', res: '1k', dir: 'assets/source/models/treasure_chest', note: 'Sunken treasure chest' },
  { kind: 'model', slug: 'wooden_crate_01', res: '1k', dir: 'assets/source/models/wooden_crate_01', note: 'Sunken crate' },
  { kind: 'model', slug: 'lambis_shell', res: '1k', dir: 'assets/source/models/lambis_shell', note: 'Shell on the seabed' },

  // ---- Ground cover -------------------------------------------------------
  //
  // Field grasses at a metre, which is what an island meadow is actually made
  // of. `grass_bermuda_01` used to sit above them and is gone: it is a lawn
  // grass 15 cm tall, and scaled up far enough to be visible on a hillside it
  // stops being grass and becomes a bush made of knives. `IslandMeadow` owns
  // close-in turf now, so there was nothing left for it to do.

  // No environment maps. Four 2k HDRIs (19 MB) used to be fetched into
  // `public/hdris/` and shipped in every build, and nothing in `src/` ever
  // loaded them: the sky, its PMREM and every preset are procedural
  // (`src/sky/`). The `hdri` kind is still understood by `plan()` should one be
  // wanted again — add it under `assets/source/` unless a URL will point at it.
];

/**
 * Sketchfab models, which exist here for one reason: **the reef.**
 *
 * Poly Haven publishes no coral, no sea fan, no sponge, no anemone and no fish —
 * all 521 of their models were checked, and `lambis_shell` is the entire marine
 * catalogue. ambientCG is materials and HDRIs. So every underwater living thing
 * in this project was either procedural or absent, and "absent" is what the
 * reef and the island shallows actually were: bare sand with a few kelp stipes
 * on it.
 *
 * Sketchfab fills that gap and costs something for it. Its download API is
 * authenticated — `GET /v3/models/<uid>/download` answers 401 without a token —
 * so this section is **opt-in and skipped when no token is present**. That is
 * not a degradation for anyone cloning the repository: the decimated outputs in
 * `public/models/dressing/` are committed exactly as the Poly Haven ones are,
 * and this step only needs to run when the optimiser does.
 *
 * To run it, put a token from <https://sketchfab.com/settings/password> into
 * `SKETCHFAB_API_TOKEN` or into a `sketchfab-token` file at the repository root
 * (git-ignored). It is a personal credential; it must not be committed.
 *
 * `licence` and `author` are not decoration. Half of these are CC-BY, which
 * legally requires the credit — see the Sketchfab table in ASSET_LICENSES.md,
 * which this manifest is the source of truth for.
 */
const SKETCHFAB = [
  {
    slug: 'soft_coral_set',
    uid: '256355f15fcb4095af17b75ae572bff0',
    note: 'Reef coral kit: 24 forms — table, barrel, brain, tube, encrusting',
    licence: 'CC-BY-4.0',
    author: 'Kanna-Nakajima',
    page: 'https://sketchfab.com/3d-models/soft-coral-set-256355f15fcb4095af17b75ae572bff0',
  },
  {
    // ---- Rock that works from every side ---------------------------------
    //
    // Poly Haven's coastal set is authored for a *shore*, and it shows in the
    // shapes: `coast_line_01` is a 55 x 2.6 x 43 m wave-cut plate, and
    // `coastal_cliff_02` is a 41 m cliff face 8 m deep. Both are excellent at
    // the one job they are for — lying flat on a beach, or being set into a
    // hillside as a face — and both are wrong the moment the camera gets round
    // the back of them, which on an island a viewer can fly over is often.
    //
    // These three are closed solids. The test is not a matter of opinion:
    // counting edges used by a single triangle gives 0.0% for both `rock_slab`
    // models and 2.3% for `rock_boulder`, against 5–22% for the scans that were
    // rejected alongside them. A closed rock can be dropped at any angle, spun
    // freely, and seen from below without showing a hollow.
    slug: 'rock_slab_a',
    uid: 'e7778771c06d4705a80cccb23a471d5c',
    note: 'Layered sandstone outcrop, closed solid',
    licence: 'CC-BY-4.0',
    author: 'mohamedhussien',
    page: 'https://sketchfab.com/3d-models/rock-17-e7778771c06d4705a80cccb23a471d5c',
  },
  {
    slug: 'rock_slab_b',
    uid: 'f4f983f89c6f4b10a54ebb0a30787e56',
    note: 'Layered sandstone outcrop, closed solid',
    licence: 'CC-BY-4.0',
    author: 'mohamedhussien',
    page: 'https://sketchfab.com/3d-models/rock-6-f4f983f89c6f4b10a54ebb0a30787e56',
  },
  {
    slug: 'rock_boulder',
    uid: 'd25c7784bc68468d88add544db970e3f',
    note: 'Weathered boulder, photogrammetry, closed solid',
    licence: 'CC-BY-4.0',
    author: '3dhdscan',
    page: 'https://sketchfab.com/3d-models/big-boulder-d25c7784bc68468d88add544db970e3f',
  },
  {
    // The one imported *animal*. Everything that swims in this project is drawn
    // by `src/scene/Fish.ts` from a procedural silhouette, which is the right
    // call for a school of two hundred and the wrong one for the fish a diver
    // actually looks at: a 0.2 m emperor angelfish with its real markings reads
    // as a reef at a glance where a grey sliver reads as debris. The skin and
    // its swim clip are stripped by the optimiser — see `src/scene/Fish.ts`,
    // which drives this mesh from the same travelling-wave vertex shader as the
    // rest of the population, and so still costs one draw call for the school.
    slug: 'emperor_angelfish',
    uid: '3dc2d360d98c485496899121792ebcce',
    note: 'Reef fish, textured (Pomacanthus imperator)',
    licence: 'CC-BY-4.0',
    author: 'Mikhail Nesterov',
    page: 'https://sketchfab.com/3d-models/emperor-angelfish-update-v2-3dc2d360d98c485496899121792ebcce',
  },
];

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const bytes = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
};

let failures = 0;
const fail = (msg) => {
  failures += 1;
  console.error(`  ERROR  ${msg}`);
};

async function getJson(url, attempts = 3, headers = { 'user-agent': 'web-ocean-3d-asset-fetcher' }) {
  let lastErr;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      const res = await fetch(url, { headers });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      return await res.json();
    } catch (err) {
      lastErr = err;
      if (i < attempts) await new Promise((r) => setTimeout(r, 800 * i));
    }
  }
  throw new Error(`GET ${url} failed after ${attempts} attempts: ${lastErr.message}`);
}

async function download(url, destPath, expected = {}, attempts = 3) {
  mkdirSync(dirname(destPath), { recursive: true });
  let lastErr;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      const res = await fetch(url, { headers: { 'user-agent': 'web-ocean-3d-asset-fetcher' } });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length === 0) throw new Error('empty response body');
      if (expected.size && buf.length !== expected.size) {
        throw new Error(`size mismatch: got ${buf.length}, expected ${expected.size}`);
      }
      if (expected.md5) {
        const got = createHash('md5').update(buf).digest('hex');
        if (got !== expected.md5) throw new Error(`md5 mismatch: got ${got}, expected ${expected.md5}`);
      }
      writeFileSync(destPath, buf);
      return buf.length;
    } catch (err) {
      lastErr = err;
      if (i < attempts) await new Promise((r) => setTimeout(r, 800 * i));
    }
  }
  throw new Error(`download ${url} failed after ${attempts} attempts: ${lastErr.message}`);
}

/**
 * The Sketchfab token, or null.
 *
 * Environment first so CI can supply it without a file existing, then the
 * git-ignored file, which is what a human actually does.
 */
function sketchfabToken() {
  const fromEnv = process.env.SKETCHFAB_API_TOKEN?.trim();
  if (fromEnv) return fromEnv;
  const file = join(ROOT, 'sketchfab-token');
  if (!existsSync(file)) return null;
  const fromFile = readFileSync(file, 'utf8').trim();
  return fromFile.length > 0 ? fromFile : null;
}

/**
 * Downloads the Sketchfab manifest, or explains why it did not.
 *
 * Never fails the run for a missing token: the outputs these feed are committed,
 * so a contributor without a Sketchfab account can still build, test and render
 * the project. It only fails when a token *is* present and a download then goes
 * wrong, which is a real error rather than an absent optional step.
 */
async function fetchSketchfab() {
  console.log('[sketchfab] reef corals and the reef fish');

  const token = sketchfabToken();
  if (!token) {
    console.log('  skip     no SKETCHFAB_API_TOKEN and no ./sketchfab-token file.');
    console.log('           These sources are only needed to re-run scripts/optimize-assets.mjs;');
    console.log('           public/models/dressing/*.glb are committed. To enable, put a token');
    console.log('           from https://sketchfab.com/settings/password in ./sketchfab-token.');
    console.log('');
    return [];
  }

  const headers = { Authorization: `Token ${token}`, 'user-agent': 'web-ocean-3d-asset-fetcher' };
  const written = [];

  for (const entry of SKETCHFAB) {
    // Always a source download: everything here is decimated by the optimiser
    // before anything loads it, so none of it belongs under `public/`.
    const rel = `assets/source/models/${entry.slug}/${entry.slug}.glb`;
    const abs = join(DOWNLOAD_ROOT, rel);

    if (!FORCE && existsSync(abs) && statSync(abs).size > 0) {
      written.push(rel);
      console.log(`  skip     ${rel} (${bytes(statSync(abs).size)})`);
      continue;
    }

    try {
      // The metadata call is not ceremony: it is where the licence and the
      // author come from, and asserting them here is what stops a model being
      // silently relicensed under us between one fetch and the next.
      const meta = await getJson(`${SKETCHFAB_API}/models/${entry.uid}`, 3, headers);
      const label = meta?.license?.label ?? '(unknown)';
      const author = meta?.user?.displayName ?? meta?.user?.username ?? '(unknown)';
      const expected = entry.licence === 'CC0-1.0' ? 'CC0 Public Domain' : 'CC Attribution';
      if (label !== expected) {
        fail(`${entry.slug}: licence is now "${label}", manifest says ${entry.licence}`);
        continue;
      }
      if (author !== entry.author) {
        fail(`${entry.slug}: author is now "${author}", manifest says "${entry.author}"`);
        continue;
      }

      const links = await getJson(`${SKETCHFAB_API}/models/${entry.uid}/download`, 3, headers);
      // `glb` comes back as a bare .glb; `gltf` is a zip. Nothing here needs a
      // zip reader, so the manifest is written against the format that avoids one.
      if (!links?.glb?.url) throw new Error('no glb download link in the API response');

      const size = await download(links.glb.url, abs);
      written.push(rel);
      console.log(`  get      ${rel} (${bytes(size)})  ${label} — ${author}`);
    } catch (err) {
      fail(`${entry.slug}: ${err.message}`);
    }
  }

  console.log('');
  return written;
}

/** Resolve one manifest entry into a flat list of { url, rel, size, md5 }. */
async function plan(entry) {
  const files = await getJson(`${API}/files/${entry.slug}`);

  if (entry.kind === 'hdri') {
    const node = files?.hdri?.[entry.res]?.hdr;
    if (!node?.url) throw new Error(`${entry.slug}: no hdri/${entry.res}/hdr in API response`);
    const name = posix.basename(new URL(node.url).pathname);
    return [{ url: node.url, rel: `${entry.dir}/${name}`, size: node.size, md5: node.md5 }];
  }

  const node = files?.gltf?.[entry.res]?.gltf;
  if (!node?.url) throw new Error(`${entry.slug}: no gltf/${entry.res}/gltf in API response`);

  const out = [];
  const gltfName = posix.basename(new URL(node.url).pathname);
  out.push({ url: node.url, rel: `${entry.dir}/${gltfName}`, size: node.size, md5: node.md5 });

  // `include` maps a path *relative to the .gltf* -> the file to fetch.
  // Keeping these relative paths intact is what makes the .gltf resolvable.
  for (const [relPath, info] of Object.entries(node.include ?? {})) {
    out.push({ url: info.url, rel: `${entry.dir}/${relPath}`, size: info.size, md5: info.md5 });
  }
  return out;
}

// ---------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------

const MIN_SIZE = 256; // anything smaller than this is an error page, not an asset

/** Sanity-check a file on disk by extension. Returns null if OK, else a reason. */
function verifyFile(absPath) {
  if (!existsSync(absPath)) return 'missing';
  const size = statSync(absPath).size;
  if (size === 0) return 'zero bytes';
  if (size < MIN_SIZE) return `implausibly small (${size} B)`;

  const lower = absPath.toLowerCase();

  if (lower.endsWith('.glb')) {
    const head = readFileSync(absPath).subarray(0, 12);
    const magic = head.subarray(0, 4).toString('ascii');
    if (magic !== 'glTF') return `bad GLB magic: ${JSON.stringify(magic)}`;
    const declared = head.readUInt32LE(8);
    if (declared !== size) return `GLB length header ${declared} != file size ${size}`;
    return null;
  }

  if (lower.endsWith('.gltf')) {
    let doc;
    try {
      doc = JSON.parse(readFileSync(absPath, 'utf8'));
    } catch (err) {
      return `not valid JSON: ${err.message}`;
    }
    if (!doc.asset?.version) return 'glTF JSON has no asset.version';
    if (!Array.isArray(doc.meshes) || doc.meshes.length === 0) return 'glTF JSON has no meshes';
    // Every buffer/image URI referenced by the .gltf must exist next to it.
    const base = dirname(absPath);
    const uris = [
      ...(doc.buffers ?? []).map((b) => b.uri),
      ...(doc.images ?? []).map((i) => i.uri),
    ].filter((u) => u && !u.startsWith('data:'));
    for (const uri of uris) {
      const target = join(base, decodeURIComponent(uri));
      if (!existsSync(target) || statSync(target).size === 0) return `referenced file missing: ${uri}`;
    }
    return null;
  }

  if (lower.endsWith('.hdr')) {
    const head = readFileSync(absPath).subarray(0, 10).toString('ascii');
    if (!head.startsWith('#?RADIANCE') && !head.startsWith('#?RGBE')) {
      return `bad Radiance HDR signature: ${JSON.stringify(head)}`;
    }
    return null;
  }

  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
    const head = readFileSync(absPath).subarray(0, 3);
    if (head[0] !== 0xff || head[1] !== 0xd8 || head[2] !== 0xff) return 'bad JPEG SOI marker';
    return null;
  }

  if (lower.endsWith('.png')) {
    const head = readFileSync(absPath).subarray(0, 8).toString('hex');
    if (head !== '89504e470d0a1a0a') return 'bad PNG signature';
    return null;
  }

  if (lower.endsWith('.bin')) return null; // opaque buffer; size check above is all we can do

  return null;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main() {
  for (const root of VERIFY_ROOTS) mkdirSync(join(DOWNLOAD_ROOT, root), { recursive: true });

  console.log(`web-ocean-3d asset fetcher`);
  console.log(`  destination: ${DOWNLOAD_ROOT} (${VERIFY_ROOTS.join(', ')})`);
  console.log(`  mode:        ${VERIFY_ONLY ? 'verify only' : FORCE ? 'force re-download' : 'incremental'}`);
  console.log('');

  /** @type {string[]} every relative path we expect to exist afterwards */
  const allRel = [];
  let downloadedBytes = 0;
  let downloadedCount = 0;
  let skippedCount = 0;

  for (const entry of MANIFEST) {
    console.log(`[${entry.kind}] ${entry.slug} @ ${entry.res} — ${entry.note}`);

    let items;
    if (VERIFY_ONLY) {
      // Without network, fall back to whatever is on disk under the target dir.
      console.log('  (verify-only: skipping API lookup)');
      items = null;
    } else {
      try {
        items = await plan(entry);
      } catch (err) {
        fail(`${entry.slug}: ${err.message}`);
        continue;
      }
    }

    if (!items) continue;

    for (const item of items) {
      const abs = join(DOWNLOAD_ROOT, item.rel);
      allRel.push(item.rel);

      // Idempotency: skip files already on disk. The API gives us an expected
      // size, so we also use it to detect truncated/corrupt files and re-fetch
      // them rather than skipping a broken asset forever.
      if (!FORCE && existsSync(abs)) {
        const onDisk = statSync(abs).size;
        if (onDisk > 0 && (!item.size || onDisk === item.size)) {
          skippedCount += 1;
          console.log(`  skip     ${item.rel} (${bytes(onDisk)})`);
          continue;
        }
        console.log(`  stale    ${item.rel} (${bytes(onDisk)}, expected ${bytes(item.size)}) — refetching`);
      }

      try {
        const n = await download(item.url, abs, { size: item.size, md5: item.md5 });
        downloadedBytes += n;
        downloadedCount += 1;
        console.log(`  get      ${item.rel} (${bytes(n)})`);
      } catch (err) {
        fail(`${item.rel}: ${err.message}`);
      }
    }
    console.log('');
  }

  if (!VERIFY_ONLY) allRel.push(...(await fetchSketchfab()));

  // -- verification pass ----------------------------------------------------
  console.log('Verifying files on disk...');
  const toCheck = allRel.length
    ? allRel
    : VERIFY_ROOTS.flatMap((root) => listAllUnder(join(DOWNLOAD_ROOT, root), root));
  let okCount = 0;
  let totalBytes = 0;
  for (const rel of toCheck) {
    const abs = join(DOWNLOAD_ROOT, rel);
    const reason = verifyFile(abs);
    if (reason) {
      fail(`verify ${rel}: ${reason}`);
    } else {
      okCount += 1;
      totalBytes += statSync(abs).size;
    }
  }

  console.log('');
  console.log('Summary');
  console.log(`  downloaded : ${downloadedCount} file(s), ${bytes(downloadedBytes)}`);
  console.log(`  skipped    : ${skippedCount} file(s) already present`);
  console.log(`  verified   : ${okCount}/${toCheck.length} file(s) OK`);
  console.log(`  total size : ${bytes(totalBytes)} (${totalBytes} bytes)`);

  if (failures > 0) {
    console.error(`\n${failures} problem(s). See ERROR lines above.`);
    process.exitCode = 1;
    return;
  }
  console.log('\nAll assets present and verified.');
}

/**
 * Recursively list asset files under `dir`, as paths relative to `dir`.
 * Dotfiles (e.g. the `.gitattributes` that keeps git from mangling these binaries)
 * are not assets and are excluded from the count and the size total.
 */
function listAllUnder(dir, prefix = '') {
  const out = [];
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    if (ent.name.startsWith('.')) continue;
    const rel = prefix ? `${prefix}/${ent.name}` : ent.name;
    if (ent.isDirectory()) out.push(...listAllUnder(join(dir, ent.name), rel));
    else out.push(rel);
  }
  return out;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
