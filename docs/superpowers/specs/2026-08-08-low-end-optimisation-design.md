# Low-end device optimisation — design

Users on low-end devices report the experience getting stuck during loading, or the browser
tab crashing during load. This document says what is actually wrong, what we are changing,
and how the work is divided so that several agents can implement it at once.

Every measurement below was taken from this repository, on 2026-08-08, and is cited to the
file it came from. Nothing here is estimated unless it says so.

## What is wrong

### VRAM is the crash

The dressing GLBs decode to **762 MiB of texture memory**. The GLBs are Meshopt-compressed
with WebP textures and total 33.7 MiB on disk, but WebP, JPEG and PNG all decode to RGBA8
regardless of file size, so the download size says nothing about the cost. Three structural
causes:

- **183 MiB is duplication.** `optimize-assets.mjs:456-465` gives every LOD level its own
  halved texture set, and `AssetLoader` caches per *URL* (`AssetLoader.ts:135-148`), so
  `tree_orchid.glb`, `_lod1.glb` and `_lod2.glb` upload three independent copies of the same
  nine maps. The GPU already solves this for free with mipmaps.
- **112 MiB is one beached pinnace.** `ship_pinnace` carries 21 images at 1024², is drawn by
  `buildStatic` (`Props.ts:2546-2559`) with no LOD chain, and is never culled — despite
  `ship_pinnace_lod1.glb` and `_lod2.glb` existing on disk and being read by nothing.
- **Everything else is uncompressed.** No GPU-compressed texture format is used anywhere.

On top of that the renderer allocates 273–476 MB of render targets
(`docs/PERFORMANCE.md:408-424`), several of them tier-independent. On a unified-memory
integrated GPU or a phone, that total is the tab dying.

### Shader compilation is the hang

Roughly **73 pipelines** are compiled before the first frame, which the code's own note
measures at **~41 s** (`Props.ts:910-929`); the WebGL2 fallback path has been observed over
60 s (`main.ts:824-839`). 63 of the 141 instanced meshes are three tree kinds, purely because
`tree_orchid`, `tree_poinciana` and `palm_coconut` ship 7, 9 and 5 materials each — and
three.js keys its render-object cache on the mesh uuid, so each is its own pipeline.

### The tier is a hard-coded lie

`DEFAULT_UI_STATE.quality` is the literal `'high'` (`ui/types.ts:67`). There is no device
detection of any kind — no `deviceMemory`, no `hardwareConcurrency`, no adapter inspection,
no persistence. Every visitor on every device boots High.

Worse, **`applyQuality()` is never called during boot.** Its only call site is
`drainQualityRequests` (`main.ts:1897`). At boot the tier reaches the scene only through
constructor arguments, so everything whose only setter lives in `applyQuality` runs on its
class default:

| Setting | Boot default | `high` value |
|---|---|---|
| `fog.setSteps` | 24 (`post/VolumetricFog.ts:176`) | 18 |
| `seafloor.setShadowSteps` | 24 (`scene/Seafloor.ts:904`) | 20 |
| `water.setReflection` | 1 (`ocean/OceanMaterial.ts:486`) | 0.85 |
| `dof.setSamples` | 0 — DoF entirely off | 16 |
| `ssr.setQuality` | 0 — SSR entirely off | 24 |

The consequence that matters: the `backend === 'webgl' ? 0 : …` guards for refraction, DoF
and lens rain (`main.ts:1463, 1472, 1484`) also never run, so **the WebGL2 fallback boots
more expensive than the tier it believes it is on.**

### "Stuck" is partly a lie the UI tells

`AssetLoader` implements a progress callback (`AssetLoader.ts:40-49, 97-127`); `main.ts:1035`
constructs it without one. The boot bar therefore sits at 0.88 through the entire download
and jumps to 0.95 for compilation. That download is **131 HTTP requests and ~64 MB fired
essentially simultaneously**, of which **18 are guaranteed 404s** because `lodUrls()`
(`Props.ts:2648-2651`) derives `_lod1`/`_lod2` URLs by string construction for kinds that
have no LOD chain.

And nothing in the boot path has a timeout, an abort, or a cancellation — not the Draco HEAD
probes, not `loadAsync`, not `compileAsync`, not `renderer.init()`. The nested
`Promise.allSettled` batches handle *rejection*, but a single child that never settles pends
the whole boot forever, and the final `.catch` cannot see it.

### Nothing on the island is ever distance-culled

`switchesSq` is `LOD_SWITCH_METRES.slice(0, levels.length - 1)` (`Props.ts:2346`) and the last
level is explicitly unbounded (`Props.ts:613-617`). `seal` computes each kind's bounding
sphere over the full placed set *before* thinning and clones it to every LOD level
(`Props.ts:2304-2327`), so frustum culling cannot remove them by distance either. **1,422
instances and ~71 draws are submitted from 1.4 km away**, including 974 understorey instances
of 202–501-triangle plants that are invisible at that range.

Meanwhile `Canopy` fades *in* between 90 m and 180 m (`Canopy.ts:122-123`) and the Props tree
meshes never fade *out*. Past 180 m the two systems fully coexist: ~124 tree meshes draw
through up to 24,000 cards, permanently. Two comments still describe a 320–520 m handover
that no longer exists (`groundShading.ts:119-120`, `main.ts:165`).

## What we are building

Acceptance bar: **an integrated-GPU laptop and a mid-range Android phone must load and hold a
stable frame.**

### A. Pre-load gate — `src/ui/StartGate.ts`

A start screen that everybody sees before anything heavy allocates.

New module, dependency-free — nothing in `src/ui` imports three.js, and that stays true. It
renders into the **existing `#boot` overlay** so it inherits the almanac sheet already styled
in `styles.css`, rather than introducing a second visual language.

The entry point at `main.ts:2903` changes from `new App(...); app.start()` to awaiting the
gate and passing the result into the `App` constructor as a formal option, replacing the
private-state cast currently used for `?webgl=1` (`main.ts:2909-2913`).

Before showing, the gate runs a **cheap** probe: `navigator.gpu.requestAdapter()` and
`adapter.info`, `navigator.hardwareConcurrency`, `navigator.deviceMemory`, and
`matchMedia('(pointer: coarse)')`. It must **not create a device** — and the adapter probe is
hoisted out of `Renderer.probeWebGPU` (`Renderer.ts:63-71`) so it is not paid twice.

The preselected default is deliberately conservative, because the cost of guessing too high
is a dead tab and the cost of guessing too low is a visitor clicking one control:

| Condition | Default |
|---|---|
| No WebGPU adapter | WebGL2, Low |
| `pointer: coarse` | Low on WebGL2, Medium on WebGPU |
| `deviceMemory <= 4` or `hardwareConcurrency <= 4` | Low |
| Adapter vendor integrated (intel, qualcomm, arm, imagination) | Medium cap |
| Otherwise | High |

This is a **default selection, never a lock.** Every tier and both backends stay selectable;
a visitor who knows their machine can pick Max on an iGPU and find out. The choice persists
to a versioned `localStorage` key and preselects next visit — but the screen still shows,
because a device that has got worse, or a first choice that was wrong, must be able to be
corrected.

**The gate must be bypassable or it breaks the tooling.** `scripts/benchmark.mjs` and 17 spec
files boot the app directly. `?webgl=1` keeps working; `?gate=0&quality=<tier>` is added, and
the benchmark and specs pass it explicitly, so they exercise the same code path a real
visitor gets minus the click. A new spec drives the gate UI itself — selects a tier, presses
Start, asserts the tier actually applied — because the bypass would otherwise hide exactly
the class of bug the gate can introduce.

### B. Boot correctness

1. **Split `applyQuality()`** into `applyQualitySettings()` (pure setters) and
   `rebuildQualityResources()` (ocean mesh and FFT rebuild), and call the setter half during
   boot once the subsystems exist — after `buildUi()`/`applyPreset()` (~`main.ts:782-784`),
   before assets and compilation begin (`main.ts:842`). Calling `applyQuality` whole would
   work but redundantly rebuilds the ocean mesh once.
2. **Wire the progress callback** the loader already has, and drive the bar across real bytes
   instead of jumping 0.88 → 0.95.
3. **Bound every boot await** with an `AbortController` and a timeout — Draco HEAD probes,
   each `loadAsync`, `compileAsync`. A failed asset must degrade to an island without a fern,
   not pend the boot.
4. **Replace derived `lodUrls()` with a manifest** emitted by `optimize-assets.mjs`, which
   already knows exactly which LODs it produced (`LOD_RATIOS`, `optimize-assets.mjs:320-325`).
5. **Concurrency-limit fetches** to ~6 in flight rather than 131 at once.
6. **Progressive boot** — start the loop once ocean and sky exist and stream island content in
   behind it. The seam exists: `loadSceneContent()` already has a live-scene path
   (`main.ts:1217-1236`) that reveals roots and compiles LOD levels mid-session. Boot becomes
   build → apply tier → `loop.start()` → hide overlay → load content → reveal.

### C. Imposters — `scripts/bake-imposters.mjs`, `src/scene/Imposters.ts`

**Bake.** Playwright-driven, following `scripts/capture-shadertoy.mjs`. Each vegetation GLB is
rendered under an orthographic camera, front view, transparent background, into a
power-of-two cell; all species pack into one 2048² atlas — 16 species at 512² each — with a
JSON sidecar carrying per-species cell rect, world width and height, and pivot. **One view per
species.** No octahedral grid: these are understorey plants and mid-distance trees, and the
parallax an atlas would buy is not worth a bake step and 21 MiB.

Nothing new is licensed. The atlas is baked from GLBs the project already ships and already
documents in `ASSET_LICENSES.md`.

**Runtime.** `Imposters.ts` is modelled directly on `Canopy.ts`, which already solves
instancing, cylindrical billboarding, GPU-side placement and `FoliageWind` sway. One
`InstancedBufferGeometry` quad, one draw call, alpha-tested with the atlas as `colorNode`.
Billboarding is **cylindrical, not spherical** — `Canopy.ts:448-456` explains why, and it
applies identically here.

**Placement reuses what Props already computes.** Props hands its placed transforms for
imposter-managed kinds to `Imposters` and then does not build those `InstancedMesh`es at all.
This is the load-bearing part of the design: on tiers that use imposters, those GLBs are
never fetched, never decoded, never uploaded, and never compiled.

| Kind | Low / Medium | High and above |
|---|---|---|
| understorey — fern, sorrel, calathea, anthurium, meadow-grass, tussock (974 instances) | imposter always | mesh under 60 m, imposter beyond, cull at 400 m |
| trees — orchid, poinciana, jacaranda, island_tree_02, palms, palms-tall (218 instances, **63 InstancedMesh**) | imposter always | mesh under 120 m, imposter to 600 m, Canopy cards beyond |

On Low and Medium this removes ~11 MiB of download, 63 `InstancedMesh`, **~45 of the ~73 boot
pipelines**, and ~400 MiB of texture VRAM, in exchange for one atlas and one draw call.

### D. Props hygiene

- **Per-kind `cullSq`.** `updateLod` already computes `distanceSq` per instance
  (`Props.ts:882-903`) and the empty-level hide path already exists (`Props.ts:930-939`); a
  level index equal to `levels.length` meaning "drop it" costs one comparison. Hero kinds
  (coast rock, the silhouette layer) default to infinity.
- **Fade Props' trees out where Canopy fades in**, ending the permanent overlap, and fix the
  two stale 320–520 m comments.
- **Share LOD0's texture maps onto LOD1/LOD2 materials** after load. `AssetLoader.converted`
  is already a `Material → Material` map (`AssetLoader.ts:86`) and `toPhysicalNodeMaterial`
  already transfers every slot explicitly (`AssetLoader.ts:270-317`). **−183 MiB, no build
  change.**
- **Give the static cove/fort meshes their LOD chains**, which are already on disk and read by
  nothing, or cull them. The pinnace alone is 112 MiB and 48,001 triangles.
- **Drop `fern_02` and `shrub_sorrel_01` LOD2** — byte-identical to their LOD1 (462/462 and
  202/202 triangles, verified from the accessors). The simplifier bottomed out; it is two
  files, two downloads and two pipelines for zero triangles saved.
- **Two free wins in `Canopy.ts`:** `side = FrontSide` instead of `DoubleSide` — the winding
  was checked and the front face always faces the viewer, so this halves the scene's
  highest-overdraw pass — and `alphaToCoverage = true`, which kills the cutout edge crawl and
  costs nothing because `antialias: true` is already set (`Renderer.ts:28`).

### E. KTX2

`babylonpress-ktx2-encoder` as a devDependency with sharp (already present at 0.35.3, pulled
in by gltf-transform) as the image decoder. No external binary, so the asset build stays
reproducible — `toktx` is not installed here and requiring it would make the pipeline
unbuildable for anyone who has not installed KTX-Software by hand.

Measured on `barrel_03_diff_1k.jpg`, 1024²:

| | file on disk | VRAM |
|---|---|---|
| JPEG today | 146 KiB | **5.3 MiB** (RGBA8 + mips) |
| ETC1S | **132 KiB** | ~0.7 MiB |
| UASTC | 1105 KiB | ~1.4 MiB |

ETC1S is smaller on disk than the source JPEG *and* 7.5× cheaper in VRAM, so the usual
"KTX2 grows the download" caveat applies only to UASTC. Therefore: **ETC1S for base colour and
ARM, UASTC for normal maps and the alpha-cutout leaf atlases** — ETC1S handles alpha badly and
`tree_orchid`/`tree_poinciana` are `alphaMode: MASK` cutouts
(`optimize-assets.mjs:252-254`).

`KTX2Loader` is wired into `AssetLoader` beside the existing Draco and Meshopt probes. It has
a first-class WebGPU branch (`KTX2Loader.js:230-241`, checking `texture-compression-astc` /
`etc2` / `s3tc` / `bc`), and the transcoder ships locally in
`node_modules/three/examples/jsm/libs/basis/` — copied to `public/basis/`, so there is no CDN,
matching the constraint already stated at `AssetLoader.ts:22-28`. `public/.gitattributes:9`
already has a `*.ktx2 -text -diff` rule; someone anticipated this.

Encoding costs 2.5 s (ETC1S) to 4.7 s (UASTC) per 1024² texture, so the script needs content
hashing and parallelism. It is a manual script and already separate from `npm run build`.

Projected: **762 MiB → ~130 MiB.**

### F. Render-target budget

Make the tier-independent allocations tier-dependent: the three 1024² RGBA16F wake targets
(~24 MiB, `Wake.ts:714-737`), the 768² mipmapped RGBA16F caustics target
(`Caustics.ts:125-201`), and the 256²×6 half-float environment cube
(`Atmosphere.ts:419-428`). Cap DPR and drop MSAA on Low — `Renderer.ts:22-45` currently
enables antialiasing and sizes the framebuffer at up to DPR 2 before the tier is consulted at
all.

### G. Collapse the tree leaf materials

`tree_orchid` ships 7 materials (trunk, branches, twig, and four separate leaf atlases) and
`tree_poinciana` ships 9 (four flower/bud materials plus a leaf). `bakeParts` merges by
material *instance* (`Props.ts:2435-2441`), so it cannot collapse them, and three.js keys its
render-object cache on the mesh uuid — so each is its own pipeline.

| Kind | Now | After |
|---|---|---|
| `island-trees` | 7 mats × 3 levels = 21 meshes | 6 |
| `island-flame-trees` | 9 × 3 = 27 | 6 |
| `island-palms` | 5 × 3 = 15 | 6 |
| **total** | **63** | **18** |

45 fewer `InstancedMesh` and **~45 fewer pipelines**, against a boot the code measures at 73
pipelines and ~41 s. This is the largest available cut to High-tier load time, and it is a
build-side change: `@gltf-transform/functions` exports `palette` and `joinPrimitives`, both
already installed. Atlas the four leaf maps into one per tree at asset-build time.

### H. Canopy realism

`Canopy.ts` is the island's silhouette at every distance, and it has no directional lighting
at all — `Canopy.ts:584-587` derives light from sun *elevation* only. The sun crosses the sky
and the forest never changes which side of the hill is bright, while the terrain underneath it
does. The file patched the symptom with `keyLight` (`Canopy.ts:580-583`) but occlusion is not
lighting.

The comment at `Canopy.ts:574-578` rejects a normal because inventing one from the uv "gives a
lighting seam down the middle of every clump". That objection is specific to an *angle-derived*
normal — the same `atan2` branch cut the file already dodges at `:538-545`. A hemisphere normal
has no seam.

1. **Dome normal + wrap diffuse.** Publish the billboard basis (`right`, `-flat`) as varyings
   and build a hemisphere normal in the fragment stage from the centred uv, with no `atan2`.
   Blend toward world-up, because a canopy is lit from above far more than a sphere is. Then
   wrap-diffuse rather than Lambert: foliage multiple-scatters between leaves, so a canopy has
   almost no terminator. Starting constants `DOME_BULGE 0.55`, `CANOPY_UP_BIAS 0.45`,
   `CANOPY_WRAP 0.65` — the "tray of broccoli" warning at `:557-563` is real, and those are the
   two knobs that hold the line against it.
2. **Back-scatter.** `groundShading.ts:104-105` already implements leaf translucency for the
   tree *meshes* (`TRANSLUCENCY_POWER 4`, `TRANSLUCENCY_GAIN 0.55`) with the rationale at
   `:75-102`. The cards do not have it. Import those constants rather than retyping them, so a
   mesh and the card standing in for it agree when the sun is behind the island.
3. **Break the silhouette.** `alpha` is currently monotone in radius, so every card is a solid
   blob with no sky holes, and three harmonics at 0.05/0.035/0.025 is a wobble rather than a
   fractal edge. Add two octaves of `mx_fractal_noise_float` on the edge, plus interior holes
   weighted away from the rim so the mass breaks without the outline fraying. Two octaves only
   — this is a heavy-overdraw pass and `:279-284` is right to guard it.
4. **Silhouette archetypes.** Band `seed.z` into three outline families — round broadleaf,
   tall-narrow, flat-topped — switching the harmonic weights and the vertical centre. This
   attacks the bubble-wrap read from the one axis `CARD_ASPECT_VARIATION` does not.
5. **`alphaToCoverage = true`.** One line. `antialias: true` is already set
   (`Renderer.ts:28`), and this preserves the depth-write the sorting argument at `:396-402`
   depends on. Kills the cutout edge crawl on 8–20 px cards.
6. **`side = FrontSide`.** One line. The winding was checked — geometry at `:207-209` with
   `right` and `flat` at `:454-455` gives `right × up = -flat`, which points toward the camera,
   so the front face always faces the viewer. Halves the rasterisation on the scene's
   highest-overdraw pass.
7. **Two-tap vertical AO.** `sunOcclusion` is sampled once (`:517-521`); sample at foot and
   crown and the difference is a free ambient gradient, so canopy in a gully is darker at its
   base than canopy on a ridge. Per vertex, four verts a card.
8. **Slope-aware stand tone.** `stand` is position-only (`:502-514`). Two extra `groundHeight`
   taps give a slope aspect; fold in at ~0.25 weight so `CANOPY_DRY` lands on sun-exposed faces
   rather than in random blotches, which is the job its comment at `:145-152` claims.

Items 5 and 6 are two lines between them and are pure wins. Items 1 and 2 are where the
realism lives. **The comments at `:574-578`, `:557-563` and `:396-402` must be rewritten as
part of this** — this codebase treats its comments as load-bearing, and all three become false
the moment item 1 lands.

### I. Meadow triangle budget

`IslandMeadow` at Max is 90,000 instances × 40 triangles = **3.6 M triangles in one draw**
(`Meadow.ts:77`, `buildBladeGeometry:531-585`) — roughly three times the entire island dressing
at LOD2, and the largest single triangle consumer in the scene. `docs/PERFORMANCE.md:250-257`
reports whole-scene totals of 3.38 M at Low rising to 4.86 M at Max, which the meadow alone
explains.

It is well built — one draw call, frustum-culling off because it follows the camera, no shadow
cast — so the lever is the blade, not the system. `TUFT_BLADES` 5 → 4 and `BLADE_ROWS` 3 → 2
takes 8 triangles per blade to 4: **a 2× cut in the largest number in the frame.** The fade at
`FADE_START = 0.78` already makes the outer third near-invisible. Measure before and after with
the `hide('meadow', …)` toggle that `scripts/profile-frame.mjs:218-219` already provides.

### J. Benchmark honesty

Two gaps `docs/PERFORMANCE.md` already discloses about itself, now that we are changing the
things it measures:

- **The gate is a median.** Nothing fails on p95 or p99, and a renderer that misses vsync every
  twentieth frame stutters however good its median is. The percentiles are already recorded;
  make them gate.
- **The camera is never underwater**, so the submerged branch — god rays at full strength, the
  underwater particle system — contributes nothing to any published number. Add a submerged
  configuration.

Also add a **low-end configuration** to the matrix: WebGL2 at Low with DPR 1 and a CPU-throttled
profile, so the thing this whole document is about has a number attached to it rather than an
argument.

## How the work is divided

Five of the six workstreams touch `main.ts` (2,920 lines) and three touch `Props.ts` (2,879
lines). Dividing agents by workstream would put every agent in those two files at once, so
**work is assigned by file ownership instead.** Each wave's agents have disjoint file sets and
can run in parallel worktrees; waves are sequential.

Agents are `codex --model gpt-5.6-luna` at `model_reasoning_effort = max`, started by `herdr`
into a pane per worktree.

### Wave 1

| Agent | Owns | Contains |
|---|---|---|
| `imposter-bake` | `scripts/bake-imposters.mjs`, `public/imposters/*` | C, bake half. All new files, zero conflict. Longest-running, so it starts first. |
| `asset-loader` | `src/scene/AssetLoader.ts`, `scripts/optimize-assets.mjs`, `package.json` | E and G entire, plus B's loader half — KTX2, the leaf-material atlas, timeouts, aborts, progress, concurrency limit, LOD manifest emit, drop redundant LOD2s. Coherent because it is all "the loader and the thing that feeds it". |
| `boot-sequence` | `src/ui/StartGate.ts`, `src/main.ts`, `src/ui/types.ts`, `src/ui/styles.css`, `tests/**`, `scripts/benchmark.mjs` | A and J entire, plus B's `main.ts` half — the `applyQuality` split, progressive boot, progress wiring, and the benchmark's new gates and configurations. |

### Wave 2

| Agent | Owns | Contains |
|---|---|---|
| `props-imposters` | `src/scene/Props.ts`, `src/scene/Imposters.ts` | D entire, plus C's runtime half. |
| `canopy-meadow` | `src/scene/Canopy.ts`, `src/scene/Meadow.ts`, `src/scene/groundShading.ts` | H and I entire. |
| `render-targets` | `src/physics/Wake.ts`, `src/underwater/Caustics.ts`, `src/sky/Atmosphere.ts`, `src/core/QualityManager.ts`, `src/core/Renderer.ts` | F. |

Wave-2 agents need small `main.ts` edits to wire their work in; those are made by the
integrating session after all three land, not by the agents, so `main.ts` has exactly one owner
throughout.

## Verification

- `npm run typecheck`, `npm test`, `npm run bench` all pass.
- A new spec drives the gate UI: select a tier, press Start, assert the tier applied.
- A new boot-budget test asserts time-to-first-frame and peak `renderer.info.memory` at Low.
- Visual baselines **will** move — the island vegetation changes by design. They are
  regenerated deliberately and reviewed, not accepted silently.
- Codex reviews the final diff. This is a standing requirement on this repository, and it
  earned its place: Codex found the `applyQuality`-never-called-at-boot defect above, which
  reading the quality tier tables would never have surfaced.

## Documentation that must change with the code

This codebase treats its prose as load-bearing, and several passages become false as this
lands. They are part of the work, not follow-up:

- `groundShading.ts:119-120` and `main.ts:165` both assert a 320–520 m mesh-to-card handover.
  `Canopy.ts:122-123` replaced it with 90/180, and as established there is no handover at all
  because the meshes never fade. Both are wrong today, before any of this.
- `Canopy.ts:574-578`, `:557-563`, `:396-402` — see G.
- `QualitySettings.canopy` and `.propsDetail` (`QualityManager.ts:222-262`) describe a world in
  which the dressing is always meshes. Imposters change what those numbers mean.
- `docs/PERFORMANCE.md` — the Results tables, the cost model, the Memory section's 273/323/476
  MB figures, and the Limitations section's "one machine" caveat all move.
- `README.md` and `docs/SPEC.md` describe the canopy as "billboard impostors" in a project that
  did not have any; after C, it does, and they should say which is which.

## What this deliberately does not do

- **It does not add octahedral imposters.** One baked view per species, per the decision
  recorded in C. Revisit only if the near band reads badly *after* measuring.
- **It does not adopt `BatchedMesh` or GPU-driven culling.** `three@0.185.1` contains
  `drawIndirect`/`drawIndexedIndirect` but WebGPU `BatchedMesh` support is still upstream work
  in progress, and `InstancedMesh2` declares no WebGPU or TSL support — it patches WebGL shader
  chunks, which cannot compose with `applyGroundShading` and `FoliageWind` rewriting
  `positionNode` on every prop material.
