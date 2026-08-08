#!/usr/bin/env node
/**
 * Corrects the transfer function recorded in every shipped KTX2 texture.
 *
 * `scripts/optimize-assets.mjs` encodes each texture with `isPerceptual` set
 * from its glTF slot — colour data perceptual, normals and coefficients not —
 * which is what tunes the codec. It did not also set
 * `isSetKTX2SRGBTransferFunc`, and that is the flag that writes the transfer
 * function into the file's own data format descriptor. A loader reads the DFD,
 * not the codec's tuning, so every texture shipped tagged sRGB.
 *
 * The consequence is not subtle. A normal map stores directions and a
 * metallic-roughness map stores coefficients; de-gamma either on read and a
 * stored roughness of 0.5 arrives as 0.21. The beached pinnace rendered pale
 * and over-glossy, and so did the coast rock — 14 of the pinnace's 21 maps were
 * mis-tagged.
 *
 * This is a **metadata-only** correction and is exactly equivalent to having
 * encoded with the right flag: the codec tuning was already correct, so the
 * compressed payload is untouched and only its declared interpretation changes.
 * It exists as a separate script because the assets cannot simply be rebuilt —
 * the original Sketchfab sources are token-gated and absent from a clean
 * checkout, which `optimize-assets.mjs` documents.
 *
 * Usage: node scripts/fix-ktx2-transfer.mjs [--check]
 */

import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';
import { listTextureSlots } from '@gltf-transform/functions';
import { read, write } from 'ktx-parse';

const KHR_DF_TRANSFER_LINEAR = 1;
const KHR_DF_TRANSFER_SRGB = 2;

/** Slots whose texels are colour. Everything else is numbers. */
const COLOUR_SLOTS = new Set(['baseColorTexture', 'emissiveTexture', 'specularColorTexture', 'sheenColorTexture']);

const CHECK_ONLY = process.argv.includes('--check');
const ROOT = 'public/models';

async function* glbFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* glbFiles(path);
    else if (entry.name.endsWith('.glb')) yield path;
  }
}

const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });

let filesChanged = 0;
let texturesChanged = 0;
let texturesChecked = 0;
const wrong = [];

for await (const file of glbFiles(ROOT)) {
  const document = await io.read(file);
  let touched = false;

  for (const texture of document.getRoot().listTextures()) {
    if (texture.getMimeType() !== 'image/ktx2') continue;
    const image = texture.getImage();
    if (!image) continue;

    const slots = listTextureSlots(texture);
    const wantSrgb = slots.some((slot) => COLOUR_SLOTS.has(slot));
    const want = wantSrgb ? KHR_DF_TRANSFER_SRGB : KHR_DF_TRANSFER_LINEAR;

    const container = read(image);
    const dfd = container.dataFormatDescriptor?.[0];
    if (!dfd) continue;
    texturesChecked++;
    if (dfd.transferFunction === want) continue;

    wrong.push(`${file} :: ${texture.getName() || '(unnamed)'} [${slots.join(',') || 'unused'}] ${dfd.transferFunction} -> ${want}`);
    if (!CHECK_ONLY) {
      dfd.transferFunction = want;
      texture.setImage(write(container, { keepWriter: true }));
      touched = true;
    }
    texturesChanged++;
  }

  if (touched) {
    await io.write(file, document);
    filesChanged++;
  }
}

console.log(`checked ${texturesChecked} KTX2 textures`);
console.log(`${CHECK_ONLY ? 'would correct' : 'corrected'} ${texturesChanged} in ${CHECK_ONLY ? '(dry run)' : filesChanged + ' file(s)'}`);
for (const line of wrong.slice(0, 8)) console.log('  ' + line);
if (wrong.length > 8) console.log(`  ... and ${wrong.length - 8} more`);
if (CHECK_ONLY && texturesChanged > 0) process.exitCode = 1;
