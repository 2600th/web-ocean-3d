# AAA pass — plan and task list

Started 2026-11-13. Goal: make the ocean, island and ship read as a shipped
AAA game rather than a graphics demo, load in a fraction of the time, and run
across devices — copying proven code from the reference projects rather than
reinventing it.

## What the references taught us

| Reference | License | What we take |
|---|---|---|
| [Noniv/web-ocean](https://github.com/Noniv/web-ocean) | MIT | Slope-variance (LEAN) sun glitter, slope-derivative specular AA, grazing-angle normal tilt + flattened-normal Fresnel, wrap-around world foam maps with drift/freshness, Karis-average bloom downsample, AgX |
| [bgbg108555/NeuralSpecOcean](https://github.com/bgbg108555/NeuralSpecOcean) | MIT | Crest sharpening `h += k(sqrt(h²+0.09)-0.3)`, roughness-capped Fresnel, per-band footprint fade with lost detail returned as roughness, chromatic caustics, quality-preset table |
| [mrdoob/slop-newyork](https://github.com/mrdoob/slop-newyork) | MIT | Zero-texture scene building: seeded procedural trees (tube + card), canvas leaf atlas with coverage mips, TSL noise + `bumpNormal`, `SkyMesh` baked into PMREM instead of HDRIs, staged loading with `compileAsync` pool, `modulepreload` |
| [Three.js Water Pro](https://threejsroadmap.com/assets/threejs-water-pro) | Commercial, closed | Feature checklist only: three foam layers (whitecap/ambient/contact), hull masking, object-contact foam, clipmap water |
| three.js r187 examples (`TRAANode`, `SMAANode`, `SkyMesh`, `LoftGeometry`) | MIT | Temporal AA for the foam/specular aliasing the renderer documents as unsolvable without it |

## Defects found in the current build (root-caused)

1. **Whitecaps render darker than the water around them.** Foam radiance was
   `albedo × bubbles(≈0.5) × (0.6–1.0)` with no sun or sky irradiance, so at
   grazing angles — where the water reflects a bright horizon — every whitecap
   was a grey (noon) or brown (sunset) hole. This is the hard-edged blotching in
   every gallery frame. *Fixed: foam is now lit by sun + sky dome.*
2. **Mid-distance foam/specular aliasing** — the code itself records that "the
   honest fix is a temporal resolve this renderer deliberately does not have".
   → add TRAA.
3. **Depth of field blurs the whole sky and the island** in the default views,
   which reads as "out of focus" not "cinematic".
4. **The island is a bald dome**: grey summit, mottled canopy, red blobs, flat
   grey sand.
5. **19 MB of HDRIs ship and are referenced by nothing.** Total payload 90 MB.

## Feature bar: Three.js Water Pro v3.5 — match, then exceed

| Water Pro feature | Us before this pass | Action |
|---|---|---|
| FFT waves, physical units | 3-cascade JONSWAP + Hasselmann spreading | — |
| Depth absorption, refraction, SSR | Beer–Lambert per channel, depth refraction, SSR over planar | — |
| Foam: whitecaps / ambient / shoreline | whitecaps + persistent buffer + depth-driven shore break | fix lighting (done), add drift-advected ambient streaks |
| Foam around objects | **missing** | **done** — depth-intersection contact foam on anything piercing the surface |
| Caustics on floor | animated caustics + world-space god rays | — (chromatic dispersion deferred: scalar field shared by four consumers) |
| Interacting wakes | Kelvin wake that displaces the surface | — |
| GPU buoyancy, multi-point | 4-probe solver on CPU sampler | — |
| Underwater + waterline | per-pixel waterline, Snell's window, TIR | — |
| Infinite water, LOD | camera-centred radial grid to 24 km | — |
| 8 presets, quality levels, determinism | 9 presets, 5 tiers + adaptive, deterministic hooks | — |
| Hull masking | not needed at this freeboard — checked from deck height in a 21 m/s storm, no water inside the hull | revisit if a low-freeboard boat is added |
| WebGL fallback | same TSL source on WebGL2 | — |
| *Beyond:* | | **done:** TRAA, crest glow, lightning + thunder, overcast-correct haze, cumulus erosion. In progress: procedural island + world, game loop, payload cut |

## Task list

### A. Load time and payload (asset agent)
- [ ] Delete unreferenced assets (HDRIs, unused dressing GLBs / LODs) and their fetch-script entries
- [ ] Ship textures → KTX2 at 1k (hull/sails/rigging), meshopt geometry
- [ ] Buoy/barrel textures → KTX2 or procedural
- [ ] Imposter atlas → KTX2/WebP
- [ ] Measure: `dist/` size and time-to-first-frame before/after

### B. Procedural island and vegetation (island agent)
- [ ] Seeded procedural palms (curved trunk tube + bent frond cards, canvas atlas, wind) replacing palm GLBs
- [ ] Procedural broadleaf canopy trees replacing the jacaranda/poinciana/orchid/island_tree GLBs
- [ ] Procedural rocks (displaced icosphere + TSL noise + bumpNormal) where scans are not hero pieces
- [ ] Island ground: warm coral sand, wet-sand band, grass/jungle, rock — no bald grey dome

### C. Water, sky and post (lead)
- [x] Foam lit by sun + sky
- [x] Grazing fix: topside normals bent toward the viewer (web-ocean) — and the underside branch no longer fires on them, which was the brown-blotch defect
- [ ] Slope-variance (LEAN) glitter — deferred: needs second-moment FFT outputs; current specular AA holds
- [ ] Crest sharpening — deferred: the CPU buoyancy sampler would have to apply the same warp
- [x] Stylised-real water: crest glow term, Tropical extinction re-cut, Tropical made the default look
- [x] TRAA temporal anti-aliasing (replaces 4x MSAA on the scene pass; ocean writes world-space velocity)
- [x] Soft sky/sea: not DOF (measured) — clouds re-shaped, clouds removed from the planar mirror, reflection mip cap lowered
- [x] Storm: re-graded, lightning + thunder, overcast-correct haze, sun disc hidden under the deck
- [x] Contact foam (Water Pro parity)
- [ ] Upgrade three r185 → r187 — **tried and deferred** (branch `three-r187`). Typecheck needs only a `QuadMesh(material)` fix (applied here too), but at runtime: with TRAA a material compiles to an empty WGSL `OutputType` struct; without it `copyFramebufferToTexture` hits bgra8unorm vs rgba16float (the refraction backdrop); `PCFSoftShadowMap` is gone. Each is its own investigation, and r185 costs nothing visible.

### D. UI and design language (UI agent)
- [x] Original nautical-adventure UI inspired by Sea of Thieves: parchment, brass, ink, hand-drawn map frames
- [x] Boot/start gate as a ship's-log page; HUD compass + wind + speed in knots
- [x] Mobile layout, reduced-motion honoured, `src/ui` stays three-free

### E. Gameplay and feel (gameplay agent)
- [x] A light loop: floating loot to collect under sail, objective marker and compass bearing (`src/gameplay`)
- [x] Sail trim against wind (point of sail matters), anchor, camera shake + spray on impacts
  (bow spray is a CPU billboard system in `gameplay/effects`; `physics/Spray` slam spray is still parked)
- [ ] Toasts/score in the new UI — `GameUiSink` is defined; the HUD has to implement it

### G. Cinematic camera (camera agent)
- [ ] Re-cut the tour: motivated shot list (establishing, ship hero, low wave-skim, island flyover, underwater dip, storm), eased splines, look-ahead framing, no clipping into water/terrain
- [ ] Smooth handoff between tour and free cameras

### H. Underwater and sea-floor life (underwater agent)
- [ ] Procedural sea-floor vegetation: seagrass beds, kelp variety, sea fans, anemones, coral clusters — instanced, wind/current sway
- [ ] Animals: more fish species and schools, a turtle, rays, jellyfish — procedural meshes, boids/paths
- [ ] Underwater look: god rays, particulates, colour depth grading, caustics on life

### I. Sky, clouds and birds (sky agent)
- [ ] Clouds: sunset/sunrise lighting, silver linings, cloud shadows; high cirrus layer
- [ ] Birds: gull flocks with flapping animation, circling the ship and island, landing on masts/rocks

### J. Audio and music (audio agent)
- [ ] Procedural adaptive score (original, synthesised — calm/sailing/storm/underwater states)
- [ ] SFX polish: hull creaks, sail luffing, rigging, wave slaps, seabirds, underwater muffling, UI clicks

### K. Shore and ship–wave interaction (shore agent)
- [ ] Island shore: breaking-wave bands, swash runup on the beach, wet-sand darkening that follows the swash, foam lines and lace
- [ ] Ship–wave impacts: bow slam splashes, spray sheets, hull-side foam, wave slap audio hooks, impact camera shake hooks

### L. Game feel and polish (lead)
- [ ] Onboarding hints, transitions between modes, pause/settings flow, touch for sails/anchor, end-to-end playtest

### F. Verification
- [ ] Progress screenshots each milestone
- [ ] `npm run typecheck`, functional suite, rebaseline visual suite deliberately
- [ ] Codex review of the final diff
