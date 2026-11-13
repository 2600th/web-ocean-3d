# Asset Licences

## Policy

`web-ocean-3d` is an **independent implementation**. Every binary asset shipped in
`public/` must be one of:

- **CC0 1.0** (public domain dedication), or
- **CC-BY** with the exact attribution string recorded in the table below, or
- **MIT / Apache-2.0**, or
- **SIL OFL 1.1**, for typefaces only — see [Typefaces](#typefaces-publicfonts).
  The OFL permits bundling and redistribution inside a work; it forbids selling
  the fonts on their own and requires a *modified* font be renamed. Neither
  condition is engaged here, since the files ship unmodified as part of the build.

Nothing else may be committed. In particular, **no assets are taken from
threejswaterpro.com or any other commercial product**. Assets are sourced from
[Poly Haven](https://polyhaven.com), [Sketchfab](https://sketchfab.com) (CC0 and
CC-BY only, licence-checked per model at fetch time), [ambientCG](https://ambientcg.com)
and the [Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
repository.

**Most of `public/` is CC0 1.0 from Poly Haven**, and for those no attribution is
legally required — we record the authors anyway, as a courtesy and because Poly
Haven asks contributors be credited where practical. The exceptions are the
twelve Sketchfab assets described under [Sketchfab assets](#sketchfab-assets):
three are CC0 from the Smithsonian and **nine are CC-BY 4.0, whose attribution is
a condition of the licence**. Those nine credits must survive redistribution, and
they are collected in one block at the end of that section for exactly that
reason.

Sources that were surveyed and not used, so the next reader does not repeat the
search:

| Source | Licence | Why not |
| --- | --- | --- |
| [Quaternius](https://quaternius.com) | CC0 | Has the animated bird and fish packs this scene wants, but the downloads are issued by client-side script rather than a stable URL, so they cannot be fetched reproducibly. |
| [Kenney](https://kenney.nl) | CC0 | Excellent, and stylistically incompatible: flat-shaded low-poly next to photogrammetry reads as two projects. |
| [ambientCG](https://ambientcg.com) | CC0 | Materials and HDRIs rather than props; its "3D model" category is scatter meshes. Worth revisiting for ground materials. |
| [Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) | per model | `BarramundiFish` is CC0 and genuinely good, but it is a 12 MB single fish. A school wants instanced geometry, not one hero mesh. |
| [Smithsonian Open Access](https://www.si.edu/openaccess) | CC0 | Now **used**, via their Sketchfab account, for the three coral colonies — see [Sketchfab assets](#sketchfab-assets). The scans are heavy (100k faces each) and take the same decimation pass everything else does. |

> Poly Haven's licence terms: <https://polyhaven.com/license> — *"All assets on Poly
> Haven are licensed as CC0, which is the same as public domain. This means you can use,
> modify and redistribute our assets for any purpose, including commercial use, without
> attribution or permission."*

### Reproducing this asset set

Run:

```sh
node scripts/fetch-assets.mjs
```

The script resolves the exact download URLs at run time through the public Poly Haven
API (`https://api.polyhaven.com/files/<slug>`), verifies each file's size and MD5
against the API manifest, checks glTF/HDR/JPEG magic bytes, and is idempotent (existing
non-empty files are skipped). It exits non-zero on any failure.

**Total size of `public/`: 55,947,175 bytes (53.36 MiB) across 85 files** — which
is everything a build ships. The raw downloads the optimiser eats are another
555 MB and live in `assets/source/`, outside the Vite public root and outside git;
see the note in `.gitignore`.

### Paths the application should load

| Purpose | Path (relative to the Vite public root, i.e. served at `/`) |
| --- | --- |
| Hero sailing ship | `/models/hero/dutch_ship_medium.glb` |
| Floating buoy | `/models/hero/ocean_buoy.glb` |
| Floating barrel | `/models/hero/barrel_03.glb` |
| Rock (island skirt and reef) | `/models/dressing/rock_07.glb` (+ `_lod1`, `_lod2`) |

The hero models are single self-contained `.glb` files built from the Poly Haven
downloads (fetched into `assets/source/models/`) by `scripts/optimize-assets.mjs`;
see `HERO_ASSETS` there for every encoding choice and why.

No environment maps ship. Four Poly Haven HDRIs (`kloofendal_43d_clear_puresky`,
`industrial_sunset_puresky`, `kloofendal_misty_morning_puresky`,
`satara_night_no_lamps`, 19 MB at 2k) used to sit in `public/hdris/` and were
referenced by nothing in `src/`: the sky and its lighting probe are procedural.
They were removed from `public/` and from `scripts/fetch-assets.mjs`.

---

## Asset inventory

One row per file shipped in `public/`.

| Asset | File path | Source URL | Author | Licence | Modifications |
| --- | --- | --- | --- | --- | --- |
| Dutch Ship Medium (`dutch_ship_medium`, 2k source) | `public/models/hero/dutch_ship_medium.glb` | https://polyhaven.com/a/dutch_ship_medium | James Ray Cock (model, textures, cleanup); Rico Cilliers (sails model, textures); Nicolò Zubbini (original model) | CC0 1.0 | Re-encoded by `scripts/optimize-assets.mjs`: hull colour and normal kept at 2k, the other seven maps resized to 1k, all nine KTX2 (UASTC colour/normal, ETC1S roughness), geometry Meshopt-compressed. No geometry or content edits. |
| Ocean Buoy (`ocean_buoy`, 1k source) | `public/models/hero/ocean_buoy.glb` | https://polyhaven.com/a/ocean_buoy | Mateusz Sadek | CC0 1.0 | Re-encoded: textures KTX2 (normal map at 512), geometry Meshopt-compressed. |
| Barrel 03 (`barrel_03`, 1k source) | `public/models/hero/barrel_03.glb` | https://polyhaven.com/a/barrel_03 | Serhii Khromov | CC0 1.0 | Re-encoded: textures KTX2 (normal map at 512), geometry Meshopt-compressed. |
| Rock 07 (`rock_07`, 1k source) | `public/models/dressing/rock_07.glb`, `rock_07_lod1.glb`, `rock_07_lod2.glb` | https://polyhaven.com/a/rock_07 | Jenelle van Heerden | CC0 1.0 | Built by the dressing pipeline (see below): decimated, welded, KTX2 textures, Meshopt geometry, two LOD levels. |

### Scene dressing (`public/models/dressing/`)

These are **modified**, and the modification is the point. Poly Haven publishes
film-quality geometry — `island_tree_01` arrives as 1.6 million triangles for a
background tree — so `scripts/optimize-assets.mjs` decimates each one, welds it,
resizes and re-encodes the textures to WebP, and compresses the result through
Meshopt. 226 MB of source becomes 24 MB shipped.

Only the `.glb` outputs are committed. The raw downloads are `.gitignore`d and
reproducible:

```sh
node scripts/fetch-assets.mjs      # authoritative source, verified against the publisher
node scripts/optimize-assets.mjs   # what actually ships
```

CC0 imposes no obligation to record any of this. It is recorded because a reader
comparing a shipped mesh against the publisher's page should be able to see why
they differ.

| Asset | File path | Source URL | Author | Licence | Modifications |
| --- | --- | --- | --- | --- | --- |
| Coast Rocks 01 (`coast_rocks_01`) | `public/models/dressing/coast_rocks_01.glb` | https://polyhaven.com/a/coast_rocks_01 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coast Rocks 03 (`coast_rocks_03`) | `public/models/dressing/coast_rocks_03.glb` | https://polyhaven.com/a/coast_rocks_03 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coastal Cliff 02 (`coastal_cliff_02`) | `public/models/dressing/coastal_cliff_02.glb` | https://polyhaven.com/a/coastal_cliff_02 | Rob Tuytel (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Sand Rocks Small 01 (`sand_rocks_small_01`) | `public/models/dressing/sand_rocks_small_01.glb` | https://polyhaven.com/a/sand_rocks_small_01 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Island Tree 01 (`island_tree_01`) | `public/models/dressing/island_tree_01.glb` | https://polyhaven.com/a/island_tree_01 | Rob Tuytel (scanning, processing); Rico Cilliers (cleanup, processing) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Grass Bermuda 01 (`grass_bermuda_01`) | `public/models/dressing/grass_bermuda_01.glb` | https://polyhaven.com/a/grass_bermuda_01 | Rico Cilliers (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Ship Pinnace (`ship_pinnace`) | `public/models/dressing/ship_pinnace.glb` | https://polyhaven.com/a/ship_pinnace | James Ray Cock (model, textures, cleanup); Rico Cilliers (sails model, textures); Nicolò Zubbini (original model); Yann Kervran (Rigging) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Modular Wooden Pier (`modular_wooden_pier`) | `public/models/dressing/modular_wooden_pier.glb` | https://polyhaven.com/a/modular_wooden_pier | Rico Cilliers (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Cannon 01 (`cannon_01`) | `public/models/dressing/cannon_01.glb` | https://polyhaven.com/a/cannon_01 | Yann Kervran (Rigging); James Ray Cock (Modeling & Texturing) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Wooden Barrels 01 (`wooden_barrels_01`) | `public/models/dressing/wooden_barrels_01.glb` | https://polyhaven.com/a/wooden_barrels_01 | James Ray Cock (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Wooden Lantern 01 (`wooden_lantern_01`) | `public/models/dressing/wooden_lantern_01.glb` | https://polyhaven.com/a/wooden_lantern_01 | James Ray Cock (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Wooden Crate 02 (`wooden_crate_02`) | `public/models/dressing/wooden_crate_02.glb` | https://polyhaven.com/a/wooden_crate_02 | James Ray Cock (modeling); Jurita Burger (graphic design) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Treasure Chest (`treasure_chest`) | `public/models/dressing/treasure_chest.glb` | https://polyhaven.com/a/treasure_chest | Rico Cilliers (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Wooden Crate 01 (`wooden_crate_01`) | `public/models/dressing/wooden_crate_01.glb` | https://polyhaven.com/a/wooden_crate_01 | James Ray Cock (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Lambis Shell (`lambis_shell`) | `public/models/dressing/lambis_shell.glb` | https://polyhaven.com/a/lambis_shell | Kuutti Siitonen (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coast Line 01 (`coast_line_01`) | `public/models/dressing/coast_line_01.glb` | https://polyhaven.com/a/coast_line_01 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coast Line 02 (`coast_line_02`) | `public/models/dressing/coast_line_02.glb` | https://polyhaven.com/a/coast_line_02 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coast Land Rocks 03 (`coast_land_rocks_03`) | `public/models/dressing/coast_land_rocks_03.glb` | https://polyhaven.com/a/coast_land_rocks_03 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Coastal Cliff 04 (`coastal_cliff_04`) | `public/models/dressing/coastal_cliff_04.glb` | https://polyhaven.com/a/coastal_cliff_04 | Rob Tuytel (Photography, processing); Rico Cilliers (cleanup) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Island Tree 03 (`island_tree_03`) | `public/models/dressing/island_tree_03.glb` | https://polyhaven.com/a/island_tree_03 | Rob Tuytel (scanning, processing); Rico Cilliers (cleanup, processing) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Antique Estoc (`antique_estoc`) | `public/models/dressing/antique_estoc.glb` | https://polyhaven.com/a/antique_estoc | James Ray Cock (Texturing); Ulan Cabanilla (modeling) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Jug 01 (`jug_01`) | `public/models/dressing/jug_01.glb` | https://polyhaven.com/a/jug_01 | Kuutti Siitonen (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Wooden Bucket 01 (`wooden_bucket_01`) | `public/models/dressing/wooden_bucket_01.glb` | https://polyhaven.com/a/wooden_bucket_01 | James Ray Cock (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |
| Modular Fort 01 (`modular_fort_01`) | `public/models/dressing/modular_fort_01.glb` | https://polyhaven.com/a/modular_fort_01 | Rico Cilliers (All) | CC0 1.0 | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP |

### Sketchfab assets

Every other asset in this project is CC0 from Poly Haven. These twelve are not,
and there are three separate reasons, none of them convenience.

**Poly Haven publishes no marine life.** All 521 of their
models were checked against `https://api.polyhaven.com/assets?t=models`, and
`lambis_shell` — one shell — is the entire catalogue. ambientCG is materials and
HDRIs. So the reef, the island shallows and the fish schools had nothing to be
made of, and the submerged half of the scene was bare sand with a few procedural
kelp stipes on it.

**Poly Haven publishes no coconut palm.** `src/scene/Remains.ts` grew a
procedural one because of it, and that palm was a good piece of engineering and a
bad tree: its fronds faked transmission through `emissiveNode`, which is neither
shadowed nor tone-mapped with the rest of the scene, so under a clear sky the
whole grove came out chrome blue. Two real palms replaced it and the procedural
one has been deleted.

**Two of Poly Haven's coastal scans only read from one direction, and two of its
trees are the wrong plant.** `coastal_cliff_02` and `_04` measure 0.21 and 0.28
deep over their own length: they are cliff *faces*, authored to be set into a
hillside, and on an island a viewer can circle they show their backs. Neither is
a bad asset; both had exactly one correct placement. `island_tree_01` and `_03`
are beautiful captures of half-bare coastal scrub, which on a tropical island
reads as a dead stick. All four are gone, replaced by the closed rocks and the
botanical trees below. The measurement is reproducible —
`node scripts/modelkit/shells.mjs <file>`.

Three of these are CC0. **Nine are CC-BY 4.0, and the attributions below are
therefore a licence condition rather than a courtesy** — they must survive into
any redistribution of this repository or a build of it.

Sketchfab's download API is authenticated, so `scripts/fetch-assets.mjs` skips
this section unless a token is present in `SKETCHFAB_API_TOKEN` or a git-ignored
`sketchfab-token` file at the repository root. That is not a barrier to building:
the decimated `.glb` outputs are committed exactly as the Poly Haven ones are,
and the token is only needed to re-run `scripts/optimize-assets.mjs`. The fetcher
re-checks each model's licence and author against this table on every download
and fails if either has changed.

| Asset | Path | Source | Author | Licence | Modifications |
| --- | --- | --- | --- | --- | --- |
| Soft Coral Set | `public/models/dressing/soft_coral_set.glb` | https://sketchfab.com/3d-models/soft-coral-set-256355f15fcb4095af17b75ae572bff0 | **Kanna-Nakajima** | **CC-BY 4.0** | Decimated, welded and Meshopt-encoded by `scripts/optimize-assets.mjs`; textures resized and re-encoded to WebP. `Props` plants subsets of its 24 forms as separate instanced kinds. |
| Emperor Angelfish (`Pomacanthus imperator`) | `public/models/dressing/emperor_angelfish.glb` | https://sketchfab.com/3d-models/emperor-angelfish-update-v2-3dc2d360d98c485496899121792ebcce | **Mikhail Nesterov** | **CC-BY 4.0** | Skin, skeleton and swim clip stripped by `scripts/optimize-assets.mjs` (`static: true`); geometry re-oriented and normalised to unit length by `src/scene/Fish.ts`, which animates it with the same travelling-wave vertex shader as the procedural fish. Textures resized and re-encoded to WebP. |
| Rock 17 (layered outcrop) | `public/models/dressing/rock_slab_a.glb` | https://sketchfab.com/3d-models/rock-17-e7778771c06d4705a80cccb23a471d5c | **mohamedhussien** | **CC-BY 4.0** | Decimated 40,000 -> 23,224 triangles; recentred onto its own base; two LOD levels generated; textures resized and re-encoded to WebP. |
| Rock 6 (layered outcrop) | `public/models/dressing/rock_slab_b.glb` | https://sketchfab.com/3d-models/rock-6-f4f983f89c6f4b10a54ebb0a30787e56 | **mohamedhussien** | **CC-BY 4.0** | Decimated 40,000 -> 7,478 triangles; recentred; two LOD levels generated; textures resized and re-encoded to WebP. |
| Big Boulder | `public/models/dressing/rock_boulder.glb` | https://sketchfab.com/3d-models/big-boulder-d25c7784bc68468d88add544db970e3f | **3dhdscan** | **CC-BY 4.0** | Decimated 65,029 -> 7,802 triangles. The source's origin sits 1.8 km above the rock, so `groundModel` in `scripts/optimize-assets.mjs` bakes a correcting translation into the scene nodes. Two LOD levels generated; textures resized and re-encoded to WebP. |

> Required attribution, in the form the licence asks for:
>
> - "Soft Coral Set" (https://sketchfab.com/3d-models/soft-coral-set-256355f15fcb4095af17b75ae572bff0)
>   by Kanna-Nakajima, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Emperor Angelfish (Update v2)" (https://sketchfab.com/3d-models/emperor-angelfish-update-v2-3dc2d360d98c485496899121792ebcce)
>   by Mikhail Nesterov, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Realistic HD Royal Poinciana (17/40)" (https://sketchfab.com/3d-models/realistic-hd-royal-poinciana-1740-066ca51810ad483aa34ef738c0b7ae6a)
>   by PlantCatalog, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Realistic HD Hong Kong Orchid Tree (40/40)" (https://sketchfab.com/3d-models/realistic-hd-hong-kong-orchid-tree-4040-160de59f02b946b0aa51f1c0f34ecbdd)
>   by PlantCatalog, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Rock 17" (https://sketchfab.com/3d-models/rock-17-e7778771c06d4705a80cccb23a471d5c)
>   by mohamedhussien, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Rock 6" (https://sketchfab.com/3d-models/rock-6-f4f983f89c6f4b10a54ebb0a30787e56)
>   by mohamedhussien, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Big Boulder" (https://sketchfab.com/3d-models/big-boulder-d25c7784bc68468d88add544db970e3f)
>   by 3dhdscan, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Coconut Palm" (https://sketchfab.com/3d-models/coconut-palm-26e787f2ff2e4c0fb004c3b0210805a3)
>   by evolveduk, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
> - "Tropical Palm" (https://sketchfab.com/3d-models/tropical-palm-08ccca74a0594fd999acaf4cfbd597e0)
>   by Šimon Ustal, licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/).
>
> All nine are modified — see the Modifications column.

### Substitutions

Some slugs floated during planning do not exist on Poly Haven. Verified against
`https://api.polyhaven.com/assets?t=models` (521 models) and `?t=hdris` (981 HDRIs):

| Slug considered | Exists? | Used instead | Why |
| --- | --- | --- | --- |
| `dutch_ship_medium` | yes | — | 2k source; ships as KTX2 with the hull at 2k and the rest at 1k. (`dutch_ship_large_01` / `_02` and `ship_pinnace` also exist.) |
| `buoy` | no | `ocean_buoy` | Closest real equivalent; `lifebuoy` is the only other buoy. |
| `wooden_barrel` | no | `barrel_03` | Real barrels are `barrel_03`, `barrel_stove`, `wine_barrel_01`, `wooden_barrels_01`. `barrel_03` is the cleanest single floating prop. |
| `rock_02` | no | `rock_07` | Only `rock_07` and `rock_09` exist in the plain `rock_NN` series. |
| `cliff_side_rock` | no | *(none — retired)* | `namaqualand_cliff_01` stood in for this and has since been dropped: it is a closed solid, but it is one 90 m mass with one composed silhouette, and five copies of one silhouette is a repeat no rotation hides. The island's crags are now built from the closed 1-2 m Sketchfab rocks, stretched and clumped. |

---

### Typefaces (`public/fonts/`)

All three are self-hosted rather than linked from a font CDN. That is a privacy
and a performance decision in equal parts: a CDN link discloses every visitor to
a third party, and it puts a DNS lookup plus a connection in front of the first
painted frame — which here is the boot chart, whose entire job is to appear
immediately.

All three are **SIL Open Font License 1.1**, which permits bundling and
redistribution in a build. The OFL requires that the fonts not be sold on their
own and that any *modified* version be renamed; no file is modified. Each is the
Latin subset exactly as Google Fonts serves it to a modern browser.

| File | Family | Author | Licence | Source |
| --- | --- | --- | --- | --- |
| `pirata-one.woff2` | Pirata One (display: names and headings) | Rodrigo Fuenzalida, Nicolás Massi | OFL-1.1 | https://fonts.google.com/specimen/Pirata+One |
| `alegreya.woff2` | Alegreya roman (variable, 400–900 wght; instrument: labels and figures) | Juan Pablo del Peral, Huerta Tipográfica | OFL-1.1 | https://fonts.google.com/specimen/Alegreya |
| `im-fell-english-italic.woff2` | IM FELL English **italic** (voice: the storm quote only) | Igino Marini (revival of the Fell types) | OFL-1.1 | https://fonts.google.com/specimen/IM+Fell+English |

9, 43 and 58 KB — 110 KB in total. The two faces in the first frame are
preloaded from `index.html`; the voice face speaks once, at the helm in a
storm, so the stylesheet is left to discover it.

No italic is shipped for Alegreya and none is set: `font-synthesis: none` on
the document means a stray `font-style: italic` renders upright rather than as
a sheared roman. See `DESIGN.md` for why these three families and what each one
is for.

## Software dependencies

| Software | Version | Licence | Project URL |
| --- | --- | --- | --- |
| three.js | ^0.185.1 | MIT | https://github.com/mrdoob/three.js |
| Vite | ^6.0.0 | MIT | https://github.com/vitejs/vite |
| Playwright (`@playwright/test`) | ^1.50.0 | Apache-2.0 | https://github.com/microsoft/playwright |
| TypeScript | ^5.7.0 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| `@types/three` | ^0.185.0 | MIT (DefinitelyTyped) | https://github.com/DefinitelyTyped/DefinitelyTyped |

All of the above are permissively licensed and redistributable in a bundled build.
