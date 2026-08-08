# Asset Loader and Asset Pipeline Implementation Plan

> **For agentic workers:** The supplied low-end optimisation design is the source of truth. This plan is limited to the asset-loader worktree ownership boundary.

**Goal:** Convert dressing textures to cached KTX2, collapse the cutout foliage materials into per-tree atlases, and make boot asset loading bounded, cancellable, progress-reporting, and manifest-driven.

**Architecture:** `AssetLoader` keeps one in-flight source promise per URL, schedules source loads through a six-slot queue, and wraps each load/probe in an abortable timeout. KTX2 support is configured against the caller’s `WebGPURenderer` before GLTF parsing and uses the locally published Basis transcoder. The optimizer resizes source images, builds a per-tree foliage atlas, joins compatible foliage primitives, emits the exact LOD manifest, and uses a content-addressed temporary KTX2 cache with a bounded worker pool.

**Tech Stack:** Three.js 0.185.1, TypeScript, glTF-Transform 4.4.2, `babylonpress-ktx2-encoder` 0.6.0, Sharp 0.35.3, Meshopt, Playwright, and the local Three.js Basis transcoder.

## Global Constraints

- Edit only `src/scene/AssetLoader.ts`, `scripts/optimize-assets.mjs`, package metadata/lockfile, `public/basis/**`, `public/models/**`, and asset-pipeline documentation.
- Do not edit `src/main.ts`, `src/scene/Props.ts`, `src/scene/Canopy.ts`, or `src/core/*`; document the integration API for their owners.
- Use ETC1S for base colour and ARM textures; use UASTC for normal maps and alpha-cutout foliage atlases.
- Keep `toktx` out of the pipeline; Node image decoding must use Sharp’s `ensureAlpha().raw().toBuffer({ resolveWithObject: true })` contract.
- Keep encoding bounded and content-hashed so unchanged images are not re-encoded and the previous memory-exhaustion failure mode is avoided.
- Produce a JSON LOD manifest beside the generated dressing models and omit LOD2 for `fern_02` and `shrub_sorrel_01`.
- Verify typecheck, end-to-end asset regeneration, measured before/after budgets, and real browser transcoding on WebGPU and WebGL2.

## Task 1: Establish red checks

- [ ] Run a focused contract check that currently fails because the loader has no KTX2 setup/abort scheduler and the optimizer has no KTX2, atlas, hash-cache, or manifest implementation.
- [ ] Use the failure output to define the exact public loader option and manifest schema before implementation.

## Task 2: Make `AssetLoader` bounded and KTX2-aware

- [ ] Add an optional renderer, timeout, and concurrency option while preserving the existing `onProgress`, material conversion, and clone/dedup behavior.
- [ ] Copy the local Basis transcoder path into the KTX2 loader and call `detectSupport` for the supplied renderer.
- [ ] Add abortable timeout wrappers around decoder probes and source loads, with cache eviction after failure and a six-load default queue.
- [ ] Keep the material conversion map keyed by source material identity and document the `main.ts` call that passes renderer and progress.
- [ ] Run `npm run typecheck` after the loader change.

## Task 3: Add optimizer transforms

- [ ] Add Sharp decoding and bounded KTX2 worker encoding with a content-addressed temporary cache.
- [ ] Classify normal and cutout foliage textures as UASTC and all remaining material/ARM textures as ETC1S, preserving KHR texture basisu metadata and embedded GLB images.
- [ ] Build one base-colour and one normal atlas for the foliage material group of `tree_orchid` and `tree_poinciana`, rewrite UVs with explicit atlas rectangles, share one material identity, and join compatible primitives without changing node transforms.
- [ ] Change the two redundant LOD entries, emit `dressing-manifest.json` with source kind, LOD URLs, and generated levels, and make output generation atomic enough that a failed asset is not published as a successful replacement.
- [ ] Run focused optimizer checks and typecheck before the full regeneration.

## Task 4: Regenerate and publish local runtime assets

- [ ] Run the optimizer end to end so `public/models/` contains regenerated GLBs and the manifest.
- [ ] Copy `basis_transcoder.js` and `basis_transcoder.wasm` from the installed Three.js package into `public/basis/`.
- [ ] Re-read generated GLB extension/material/LOD metadata and confirm the atlas group counts and manifest URLs.

## Task 5: Measure and verify

- [ ] Measure dressing and total model download bytes before and after.
- [ ] Measure per-texture dimensions and compute RGBA8 VRAM as `w × h × 4 × 1.333`; compute KTX2 VRAM with 0.5 bytes/pixel for ETC1S and 1 byte/pixel for UASTC, including mip overhead.
- [ ] Start the dev server and load a generated KTX2 GLB through `KTX2Loader` in a real WebGPU browser context and a forced WebGL2 context, asserting a compressed texture was decoded rather than accepting a white-material fallback.
- [ ] Run the brief’s required verification commands, inspect the final diff, write the measured report, and commit with the repository’s plain-spoken subject style.
