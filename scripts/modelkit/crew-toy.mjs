#!/usr/bin/env node
/**
 * crew-toy.mjs — compresses the crew built by `crew-toy.py` into the shipped
 * `public/models/crew/crew_toy.glb`.
 *
 *   node scripts/modelkit/crew-toy.mjs <crew_toy.raw.glb> [out.glb]
 *
 * Resamples the clips (drops keys a straight line would give back), welds,
 * prunes, and Meshopt-encodes with quantised attributes. The atlas stays a
 * PNG: it is a 25 KB palette of flat ramps, and the crew's loader has no KTX2
 * transcoder to set up for it.
 */
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { statSync } from 'node:fs';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, meshopt, prune, resample, weld } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const src = process.argv[2];
const out = process.argv[3] ?? resolve(ROOT, 'public/models/crew/crew_toy.glb');
if (!src) {
  console.error('usage: node scripts/modelkit/crew-toy.mjs <crew_toy.raw.glb> [out.glb]');
  process.exit(1);
}
await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.decoder': MeshoptDecoder,
  'meshopt.encoder': MeshoptEncoder,
});
const doc = await io.read(src);
await doc.transform(resample({ tolerance: 1e-4 }), weld(), dedup(), prune());
await doc.transform(meshopt({ encoder: MeshoptEncoder, level: 'medium' }));
await io.write(out, doc);
console.log(`${out}: ${statSync(src).size} -> ${statSync(out).size} bytes`);
