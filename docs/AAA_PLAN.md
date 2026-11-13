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
- [x] Seeded procedural palms (curved trunk tube + bent frond cards, canvas atlas, wind) replacing palm GLBs
- [x] Procedural broadleaf canopy trees replacing the jacaranda/poinciana/orchid/island_tree GLBs
- [x] Procedural rocks (displaced icosphere + TSL noise + bumpNormal) where scans are not hero pieces
- [x] Island ground: warm coral sand, wet-sand band, grass/jungle, rock — no bald grey dome

- [x] World detail: skull rock, sea arch, summit spires, strata outcrops, wet barnacled waterlines, beach wrack and driftwood, camp with fire/smoke/torches. 12.8 MB of vegetation GLBs and the imposter atlas removed.
- [x] Lower slopes no longer read as lawn: open ground is scrub (tufts, straw-dry patches, bare soil) in the terrain shader
- [ ] Follow-ups: sand swirl/grid pattern at close range (shore agent owns the sand); shell litter is at true scale and reads only up close — accepted

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
- [x] Re-cut the tour: motivated shot list (establishing, ship hero, low wave-skim, island flyover, underwater dip, storm), eased splines, look-ahead framing, no clipping into water/terrain
- [x] Smooth handoff between tour and free cameras (lens and Dutch angle ease back with the pose)

### H. Underwater and sea-floor life (underwater agent)
- [ ] Procedural sea-floor vegetation: seagrass beds, kelp variety, sea fans, anemones, coral clusters — instanced, wind/current sway
- [ ] Animals: more fish species and schools, a turtle, rays, jellyfish — procedural meshes, boids/paths
- [ ] Underwater look: god rays, particulates, colour depth grading, caustics on life

### I. Sky, clouds and birds (sky agent)
- [ ] Clouds: sunset/sunrise lighting, silver linings, cloud shadows; high cirrus layer
- [ ] Birds: gull flocks with flapping animation, circling the ship and island, landing on masts/rocks

### J. Audio and music (audio agent)
- [x] Procedural adaptive score (original, synthesised — calm/sailing/storm/underwater/tour states, cargo sting, Music slider)
- [x] SFX polish: hull creaks, sail luffing, rigging, wave slaps, seabirds, underwater muffling, UI clicks
- [x] Hooks for other systems: `hullImpact`, `waveSlap`, `uiTick`, `setMusicState`, `playSting` (documented at the top of `AudioSystem.ts`)

### K. Shore and ship–wave interaction (shore agent)
- [ ] Island shore: breaking-wave bands, swash runup on the beach, wet-sand darkening that follows the swash, foam lines and lace
- [ ] Ship–wave impacts: bow slam splashes, spray sheets, hull-side foam, wave slap audio hooks, impact camera shake hooks

### L. Game feel and polish (lead)
- [ ] Onboarding hints, transitions between modes, pause/settings flow, touch for sails/anchor, end-to-end playtest

### M. UI finish pass (UI agent, round 2)
- [x] Main menu with a living background: animated illustrated seascape (parallax swell, drifting clouds, gulls, sun glints, spray particles) that costs nothing on low-end devices and respects reduced motion — `src/ui/MenuSea.ts`, 2D canvas, cross-fades into the live sea
- [x] Nothing half-baked: every screen, state and transition finished (loading, errors, empty states, hover/press, focus) — WebGPU-unavailable gate, failed-boot retry, icon touch orders
- [x] HUD never blocks the view on PC or mobile: ledger collapsed by default, idle-fade of non-critical HUD, translucency over the scene, context-sensitive elements, compact mobile HUD, safe areas — the techniques AAA games use (DESIGN.md › Attention, Layout)
- [x] Sleek: consistent spacing, type scale and motion; reviewed at 1920x1080, 1280x720, 390x844, 844x390

### N. Post-processing to current AAA practice (post agent)
Audit 2026-11-13. Have: TRAA, HDR bloom (exposure-tracked threshold), thin-lens DOF, occludable lens flare, ASC CDL grade + cos⁴ vignette, volumetric fog/god rays, lens rain, triangular dither, ACES. Missing against what shipped AAA titles run:
- [x] Tone mapping: compared ACES vs AgX vs Khronos PBR Neutral, mid-grey matched (`?tonemap=aces|agx|neutral`, trims computed from three's own curve formulas: AgX ×1.008, Neutral ×1.406). **Kept ACES.** Over 11 shots (helm, tour 66/85, storm, sunset, moonlit, arctic, seaOfThieves, underwater, clear, dusk): AgX halves clipping where the sun is in frame (tour 66: 7.2% → 3.3% of pixels clipped; sunset 11.9% → 6.6%) but cuts mean saturation 20–40% on every preset (seaOfThieves 0.247 → 0.145, underwater 0.350 → 0.206) and lifts night blacks (moonlit mean luma 111 → 132) — every CDL grade would have to be re-authored to get the turquoise back. Neutral is the most saturated but clips far more (tour 66 26%, sunset 21%, helm 4.6% vs 0.05%). ACES plus eye adaptation fixes the sun-facing frames AgX was winning on: tour 66 is pulled down 0.6 EV and its clouds come back
- [x] Auto-exposure / eye adaptation (`src/post/AutoExposure.ts`): 16×16 centre-weighted log-luminance meter → 1×1 reduction → asymmetric adaptation (3/s toward bright, 0.9/s toward dark), all on the GPU, no readback. Corrects 60% of the deviation from a **per-preset anchor** (measured at each preset's canonical view, adaptation off) beyond a ±0.75 EV dead zone, clamped to [−1.5, +0.5] EV — so every preset's authored look is untouched at its own framing and only the sun, the night beat and dark close-ups move. Gain is applied *ahead* of the bloom so its exposure-derived threshold stays exact. Deterministic: the capture hook makes every frame adopt its own target (no history). Measured gain: tour 66 −0.61 EV, tour 85 night +0.5 EV (clamped — night stays night), deck close-up +0.32 EV, every gallery preset 0
- [x] Ambient occlusion: GTAO (`GTAONode`, its own temporal rotation driven by a counter the capture resets) composited into the scene colour *before* TRAA, masked off water (the ocean's velocity MRT now carries a flag in `.z`) and sky, faded out 60–160 m. The fade is load-bearing: without it, normals rebuilt from a perspective depth buffer drew vertical streaks down the island's tree line at 1 km. Post-multiply rather than `builtinAOContext`: the correct indirect-only path needs a depth prepass of the whole scene and the context leaks into the planar reflector's nested render
- [x] Motion blur (`src/post/PostResolve.ts`): gather along each pixel's own velocity, 180° shutter normalised to 60 Hz, clamped to 3.5% of frame height, taps weighted by whether they land on moving pixels so a static hull does not smear. Off under `prefers-reduced-motion` (live) and in deterministic captures unless `setMotionBlurInCapture(true)`
- [x] Post-TAA sharpening: RCAS (AMD FSR1's kernel, as `SharpenNode` ports it) run on a Karis-compressed signal so HDR glints keep the limiter's [0,1] assumption; fused with blur and CA into one pass. 0.3 at High+, 0.4 Medium, 0.5 Low
- [x] Below-native rendering for Low (0.67) and Medium (0.77). **`TAAUNode` was tried and rejected on evidence**: at 0.67 it dissolved the rigging and its thin-feature lock left dark specks all through the island canopy; at 1.0 it was still softer than TRAA and still speckled. TRAA at the reduced size (history follows its input) + bilinear upsample + RCAS beat it on every frame compared. TRAA's previous-depth copy is skipped below native in r185, which would silently turn the AA off on a Low boot — `ScaledTRAA` does the copy against the input's size. Measured: Medium 0.79 ms of 4.6 (17%), Low 0.25 ms of 0.96 (26%); the same knob on High would save 1.08 ms (0.77) / 1.44 ms (0.67) of 6.6. FSR1 not used: its EASU needs a low-res output from the resolve, which r185's TRAA cannot give without a fork
- [x] Screen-space contact shadows: **skipped, with reason.** `SSSNode` only makes sense on the *direct* term, which in three means the same `builtinShadowContext` route as AO — a second full raster of the scene (ocean grid, 38k meadow blades, 15k canopy at High) and a context that leaks into the planar reflector's nested render. A post multiply instead darkens sky light and reads as a second, harder AO. GTAO at a 0.6 m radius already grounds the deck furniture; revisit if a depth prepass ever exists for another reason
- [x] Lens character, High and up only: film grain in display space, mid-tone weighted, ~1.5 px value noise at 1.2% (static in captures and under reduced motion, animated otherwise); lateral CA, radial, r²-weighted, 0.001 UV at the corner
- [x] Every effect: per-tier table in `POST_TIERS` (`src/post/PostChain.ts`), A/B on `__ocean.post.chain.setOverride(name, bool|null)`, `?post=0` restores the pre-pass chain, rows in `scripts/profile-frame.mjs` (`post: …`). Costs measured A/B in one session (median of 4–5 alternations, RTX 5090, 1600×900, deck camera): GTAO 0.02–0.04 ms (High, half res), eye adaptation 0.01–0.02 ms, motion blur 0.01–0.04 ms at a 40 px pan, sharpen / CA / grain ≤ 0.01 ms each. WebGL2 gets eye adaptation and grain only (everything else hangs off the velocity MRT that path does not have)
- Not done / open: the AO multiply also darkens sunlit crevices a little (the price of not having a prepass); motion blur is per-pixel gather with no tile-max dilation, so a fast object's silhouette does not blur outward over a still background; the eye-adaptation anchors are per preset and need re-measuring if a preset's grade or exposure is re-authored

### F. Verification
- [ ] Progress screenshots each milestone
- [ ] `npm run typecheck`, functional suite, rebaseline visual suite deliberately
- [ ] Codex review of the final diff
- [x] Capture determinism: hero shot differs 0.22% run-to-run on this branch vs 0.18% on `main`, same patch beside the ship — pre-existing, not introduced here
