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
- [ ] Slope-variance glitter + grazing fixes (web-ocean)
- [ ] Crest sharpening + roughness-capped Fresnel (NeuralSpecOcean)
- [ ] Painterly-saturated, Sea-of-Thieves-inspired water palette: emerald SSS on crests, deep teal body
- [ ] TRAA temporal anti-aliasing
- [ ] DOF focus fix (no smeared sky)
- [ ] Upgrade three r185 → r187

### D. UI and design language (UI agent)
- [x] Original nautical-adventure UI inspired by Sea of Thieves: parchment, brass, ink, hand-drawn map frames
- [x] Boot/start gate as a ship's-log page; HUD compass + wind + speed in knots
- [x] Mobile layout, reduced-motion honoured, `src/ui` stays three-free

### E. Gameplay and feel (gameplay agent)
- [ ] A light loop: floating loot to collect under sail, objective marker and compass bearing
- [ ] Sail trim against wind (point of sail matters), anchor, camera shake + spray on impacts
- [ ] Toasts/score in the new UI

### F. Verification
- [ ] Progress screenshots each milestone
- [ ] `npm run typecheck`, functional suite, rebaseline visual suite deliberately
- [ ] Codex review of the final diff
