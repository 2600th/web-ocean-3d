#!/usr/bin/env node
/**
 * Strips the textures out of the dressing's LOD1/LOD2 files.
 *
 * At runtime every LOD level takes its LOD0's maps (`shareMaterialMaps` in
 * `src/scene/Props.ts`): one texture set per model on the GPU, and no change
 * of texture between levels. The LOD files' own textures were therefore
 * downloaded, transcoded and thrown away: 4.4 MB of the 5.7 MB the LOD files
 * weighed. `optimize-assets.mjs` no longer writes them; this rewrites the files
 * already in `public/` so the sources need not be fetched again.
 *
 * Geometry is untouched: the Meshopt-compressed, quantised buffers are decoded
 * and re-encoded at the same settings. Materials keep their factors and
 * extensions, only their texture references go.
 *
 * Usage: node scripts/strip-lod-textures.mjs [--dry]
 */
import { readdirSync, statSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIR = join(ROOT, 'public', 'models', 'dressing');
const DRY = process.argv.includes('--dry');

await MeshoptDecoder.ready;
await MeshoptEncoder.ready;
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });

let before = 0;
let after = 0;
for (const name of readdirSync(DIR).filter((f) => /_lod[12]\.glb$/.test(f)).sort()) {
  const path = join(DIR, name);
  const size = statSync(path).size;
  const document = await io.read(path);
  const textures = document.getRoot().listTextures();
  if (textures.length === 0) {
    before += size;
    after += size;
    continue;
  }
  for (const texture of textures) texture.dispose();
  // Extensions that only existed to carry textures go with them.
  for (const extension of document.getRoot().listExtensionsUsed()) {
    const id = extension.extensionName;
    if (id === 'KHR_texture_basisu' || id === 'EXT_texture_webp' || id === 'KHR_texture_transform') {
      extension.dispose();
    }
  }
  const bytes = await io.writeBinary(document);
  before += size;
  after += bytes.byteLength;
  console.log(`${name}: ${(size / 1024).toFixed(0)} KB -> ${(bytes.byteLength / 1024).toFixed(0)} KB (${textures.length} textures)`);
  if (!DRY) await writeFile(path, bytes);
}
console.log(`total ${(before / 1048576).toFixed(2)} MB -> ${(after / 1048576).toFixed(2)} MB${DRY ? ' (dry run)' : ''}`);
