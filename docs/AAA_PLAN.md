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
| Foam: whitecaps / ambient / shoreline | whitecaps + persistent buffer + depth-driven shore break | fix lighting (done), add drift-advected ambient streaks; **done:** breakers in sets with lip, bore and cellular lace, swash sheet up the sand, wet sand that dries |
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
- [x] Delete unreferenced assets (HDRIs, unused dressing GLBs / LODs) and their fetch-script entries — HDRIs (19 MB) gone; every dressing GLB and LOD on disk is loaded, so none removed. Draco decoder and the duplicate `public/basis` transcoder dropped from the build.
- [x] Ship textures → KTX2 at 1k (hull/sails/rigging), meshopt geometry — one `/models/hero/dutch_ship_medium.glb`, 26.4 → 13.5 MB; hull colour and normal stay 2k (1k was visibly soft at the close-up).
- [x] Buoy/barrel textures → KTX2 or procedural — KTX2 + Meshopt; buoy 2.9 → 0.77 MB, barrel 0.56 MB either way at a quarter of the VRAM.
- [x] Imposter atlas → KTX2/WebP — WebP, 1.8 → 0.37 MB.
- [x] Measure: `dist/` size and time-to-first-frame before/after — `public/` 88 → 53 MiB, `dist/` 100 → 64 MiB; High on the FXC path: first frame ~7 s → ~6 s, `isReady` ~39 s → ~20 s.

### B. Procedural island and vegetation (island agent)
- [x] Seeded procedural palms (curved trunk tube + bent frond cards, canvas atlas, wind) replacing palm GLBs
- [x] Procedural broadleaf canopy trees replacing the jacaranda/poinciana/orchid/island_tree GLBs
- [x] Procedural rocks (displaced icosphere + TSL noise + bumpNormal) where scans are not hero pieces
- [x] Island ground: warm coral sand, wet-sand band, grass/jungle, rock — no bald grey dome

- [x] World detail: skull rock, sea arch, summit spires, strata outcrops, wet barnacled waterlines, beach wrack and driftwood, camp with fire/smoke/torches. 12.8 MB of vegetation GLBs and the imposter atlas removed.
- [x] Lower slopes no longer read as lawn: open ground is scrub (tufts, straw-dry patches, bare soil) in the terrain shader
- [x] Dry-sand swirls removed (ripple map faded out above the swash, relief bump damped on the dry beach); beach wrack blended instead of cut out (no MSAA under TRAA); meadow clumped with bare ground and dry patches
- Shell litter is at true scale and reads only up close — accepted

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
- [x] Toasts/score in the new UI — `GameUiSink` is defined; the HUD has to implement it

### G. Cinematic camera (camera agent)
- [x] Re-cut the tour: motivated shot list (establishing, ship hero, low wave-skim, island flyover, underwater dip, storm), eased splines, look-ahead framing, no clipping into water/terrain
- [x] Smooth handoff between tour and free cameras (lens and Dutch angle ease back with the pose)

### H. Underwater and sea-floor life (underwater agent)
- [x] Procedural sea-floor vegetation: seagrass beds, kelp variety, sea fans, anemones, coral clusters — instanced, wind/current sway
- [x] Animals: more fish species and schools, a turtle, rays, jellyfish — procedural meshes, boids/paths
- [x] Underwater look: god rays, particulates, colour depth grading, caustics on life

### I. Sky, clouds and birds (sky agent)
- [x] Clouds: sunset/sunrise lighting, silver linings, cloud shadows; high cirrus layer — clouds resolved by TRAA (own velocity), lit by the measured dome (no white deck at night/dusk), sunset underglow, storm base relief, 2D cirrus; twilight dome, milky way, 22° moon halo. Cost unchanged: cloud layer 2.7 ms at High / 1600x900, same as before the pass
- [x] Birds: gull flocks with flapping animation, circling the ship and island, landing on masts/rocks — M-wing flap/glide in the vertex stage, banking circuits round ship and island, soarers, five perched on the yards and taffrail that take off when she gets under way
- [ ] Follow-ups: perched and close gulls read as white shapes (no feather/grey-back detail); storm deck is moody but low in structure toward the sun; bright sea horizon line at night is the water/haze, not the sky

### J. Audio and music (audio agent)
- [x] Procedural adaptive score (original, synthesised — calm/sailing/storm/underwater/tour states, cargo sting, Music slider)
- [x] SFX polish: hull creaks, sail luffing, rigging, wave slaps, seabirds, underwater muffling, UI clicks
- [x] Hooks for other systems: `hullImpact`, `waveSlap`, `uiTick`, `setMusicState`, `playSting` (documented at the top of `AudioSystem.ts`)

### K. Shore and ship–wave interaction (shore agent)
- [x] Island shore: breaking-wave bands, swash runup on the beach, wet-sand darkening that follows the swash, foam lines and lace — `ocean/Surf` (closed-form breakers in sets, shared by the ocean's vertex and fragment stages, the wake buffer and the ground hook)
- [x] Ship–wave impacts: bow slam splashes, spray sheets, hull-side foam, wave slap audio hooks, impact camera shake hooks — `physics/Spray` (now visible; see its header for why it was not), routed through `onSprayImpact` in `main.ts`
- [x] Audio: `hullImpact` / `waveSlap` are called through an optional interface and stay silent until `AudioSystem` implements them

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
- Against `docs/REFERENCE_TARGETS.md` (captures `final-before_*` / `final-after_*`, same merged build, `?post=0` vs default): **shore by day** — the cyan rule is what rules AgX out; ACES keeps island-approach saturation at 0.380 (before 0.382). **Night** (tour 85) — +0.5 EV lifts mean luma 78 → 93 with no crushed and no clipped pixels; sky stays navy, sea teal; the bright horizon line is upstream (sky). **Storm** — untouched by design (anchor), saturation 0.121 → 0.120, teal-grey; the horizon haze is upstream. **Dusk** — the miss: the Sunset preset clips 17.8% white and its glitter path reads white, not gold. Attributed: bloom, flare and fog off each change it by < 0.05%, and −0.85 EV of exposure only takes it to 13.9% — it is scene radiance (the sun's Mie halo and the water under it), not the tone curve. AgX would halve it (11.9% → 6.6% in the comparison above) at the cost of the cyans; the right fix is in the atmosphere. Golden hour in the tour (t = 66) is helped: adaptation takes clipping 21.4% → 11.3%
- Determinism: two consecutive captures of the same shot match exactly on the tour shots (t = 25, 40, 85: ≤ 0.001% of pixels > 2 levels). Helm, deck and storm differ run to run by 12–33% of pixels — **pre-existing**: the `?post=0` chain on the same build differs by the same amount (12.5% / 29.5% / 15.1%), and the deck shot's mean luma shifts in both. TRAA, GTAO and the adaptation now update once per pipeline render rather than once per rAF tick, which removes one source of capture-timing dependence from this chain
- Visual baselines that will shift: every gallery shot (sharpening, grain and CA at High; GTAO on the hull, deck and rocks), most of all `boat-chase`/helm and the cinematic shots that face the sun or run at night (adaptation), and anything at Low/Medium (render scale)
- Not done / open: the AO multiply also darkens sunlit crevices a little (the price of not having a prepass); motion blur is per-pixel gather with no tile-max dilation, so a fast object's silhouette does not blur outward over a still background; the eye-adaptation anchors are per preset and need re-measuring if a preset's grade or exposure is re-authored

### O. Lighting, materials and shaders to current AAA practice (lighting agent)
Audit each against what shipped AAA titles do, fix on evidence, measure cost per tier:
- [x] **Shadows:** three's `CSMShadowNode` behind `src/lighting/SunShadows.ts` — bounding-sphere fit with light-space texel snapping (no swimming), 3×3 PCF-soft per cascade, fade between cascades, custom split table per count. Bias is set *per cascade in texels* (depth 1.2 texels, normal 1.5 texels) instead of the old −0.0006 depth bias that was 3.6 m over the 6 km shadow camera — the cause of the floating rigging and the missing deck shadows. Per tier (cascades / distance / map per cascade): Low 1 / 120 m / 512, Medium 3 / 200 m / 1024, High 3 / 300 m / 2048, Ultra 3 / 400 m / 2048, Max 4 / 500 m / 2048 (was one 4096). High texels 2.5 / 9.3 / 31 cm against the old 25 cm everywhere; the deck now carries rail, rigging and mast shadows and the palms shadow the sand. Contact hardening not done (PCSS per cascade is not in r185's node and the deck reads correctly without it)
- [x] **Image-based lighting:** audited, nothing to fix. `scene.environment` already follows the live sky (PMREM of the sky/cloud dome, so overcast and night are in it); three's `EnvironmentNode` applies AO-based specular occlusion and the roughness bend toward the normal. "Grey wash": the hemisphere fill measured at 5–10% of shadowed luminance (hull p5 0.0385 → 0.0341 with it off), so it is not one; at night it is now scaled with the moon. Horizon occlusion **skipped**: it needs a patched `EnvironmentNode`, and the only surfaces where it would show (normal-mapped hull planks at grazing angles) do not leak visibly
- [x] **Local lights** (`src/lighting/LocalLights.ts`): camp fire (flickering, deterministic in the clock), jetty torches, cove lantern, ship's stern lantern, as inverse-square point lights all created at boot (the light set is part of every lit shader) and switched by intensity with darkness: on from sun elevation +0.04, distance-faded at 26 / 16 / 16 / 10 m, culled outright beyond 150–280 m. Tier-gated by `quality.localLights` (Low 0). Lantern glass and a small additive halo give the bloom something to catch. Smoke above the fire picks up firelight. Night frames: the jetty planks and the fire's ground pool go from unlit to lit (camp_night region mean luma 0.007 → 0.048)
- [x] **Lightning** (`src/lighting/LightningLight.ts`): each strike is a directional light from the bolt's position (12 at the stroke peak, falling as (1500 m / d)^1.2, floor 0.3) plus a hemisphere "deck" fill (2.4), both driven by the existing stroke envelope; the colour grade's lift is cut to 55% of what it was, so the flash now has direction (backlit ship region linear luma 0.33 → 0.48 on a strike in front). Sky and water reflection lift with the deck light. Clouds are not lit by it (sky agent)
- [x] **Night:** the moon is the key at 0.6 (was 0.42), colour (0.44, 0.58, 0.9), fill 0.24; meadow and canopy (custom-lit) now take the key light rather than the sun below the horizon, which had lit the night grass orange. The sea's key glint gets ×3 at night (`MOON_GLINT_BOOST`) so the moon path reads: on-path / off-path luma 3.7× → 5.6× (moon_glitter19), and 11.6× once the sky agent's darker night sea merged in. Nothing black: tour 85 hull planks read. Still wide rather than a narrow column to the horizon — that width is the water's slope variance (OceanMaterial), not the light
- [x] **Materials (PBR):** map averages read back on the GPU for every textured material. Wood 0.04–0.14, canvas 0.17, plaster 0.30, roughness 0.3–1 — plausible. Fixed: rock slab models carried a metalness map averaging 0.4 (rock is a dielectric) → stone metalness 0 (`src/lighting/PbrAudit.ts`); dry sand albedo (0.5, 0.41, 0.25) → (0.6, 0.52, 0.37), beach linear luma 0.39 → 0.44 (cream, still under the 0.6 of white surf). Specular AA (Tokuyoshi–Kaplanyan normal-variance roughness widening) on every normal-mapped ship and prop material. Left: the glazed jug at 0.92 albedo
- [x] **Translucency** (`src/lighting/Translucency.ts`): thin-surface diffuse transmission on the sails (T = 0.32, through the cloud shade) and on the leaves (0.22, in the ground shader's leaf glow), using the back-face-negated normal so it lights the face away from the sun. Backlit sail region luma 0.196 → 0.238. Not shadow-mapped: a sail behind another sail transmits as if lit
- [x] **Wetness** (`src/scene/Wetness.ts`): per-object channels (ship, island, world) through an object-scope uniform, so dousing the deck no longer darkens the island props; the sand darkens (×0.62) and smooths (roughness → 0.5) with the island's rain level on up-facing exposed ground above the swash
- [x] **Shader hygiene:** no NaN/firefly found in any capture (2 × 24 shots, 9 repeated); the new terms are all clamped/saturated. Fog: point-light and transmission terms go through the materials' normal fog path. Compile cost: the extra cascades and four point lights make every lit material bigger — boot-to-ready 17.3 s → 21.8 s on High in one run each (noisy; assets agent to own). No synchronous rebuild after ready: the rAF trace shows the same post-ready stalls as `aaa-pass` (worst 296 vs 357 ms, both in the 25–45 s window when content streams in)
- Cost (GPU timestamps, median of 90 frames, 1280×720, same session pair, shared GPU): High +0.02–0.2 ms (helm 5.81 → 5.83, deck 5.82 → 5.90, jetty 5.66 → 5.82, jetty at night 5.64 → 5.84); Max −0.7 ms (10.96 → 10.23, 4096 map → 4 × 2048); Low: below what this method can resolve — the same build swings 0.65 ↔ 1.05 ms and 0.86 ↔ 2.17 ms between runs (GPU clock states at a ~1 ms load); by construction it renders the one 512 shadow pass it always had and no local lights, so only the material ALU (specular AA, transmission) is added. Draw calls +4–19 per view (the lights' glow meshes, the extra cascades)
- Determinism: two consecutive runs of 9 shots match bit-exactly on 7 (ship, deck, camp, fire, storm with a strike, moon path, island); helm and tour 85 differ run to run **on `aaa-pass` too** (pre-existing)
- Visual baselines that will shift: every shot with the ship or the island in it (shadows, sails, PBR), all night and dusk shots (moon key, local lights, moon path), storm (flash), shore shots (sand)

### P. Words: copy, tone and onboarding (copy agent)
User direction 2026-11-13: make the language more interesting, engaging and user-friendly, taking reference from the best games and repos.
- [x] Voice: an original seafaring-adventure voice for everything the player reads in play (menu, loading lines, objectives, toasts, hints, ledger labels, finish states) — warm, a little wry, never parody, never a pastiche of any one game; instrumentation (FPS, backend, tier) stays plain and truthful
- [x] Onboarding that teaches by doing: first-minute hints in the order a new player needs them, context-sensitive (keyboard vs touch), dismissed once learned
- [x] Loading: rotating lines that are tips or world flavour, not progress jargon; errors that say what happened and what to do
- [x] Every label and tooltip understandable without graphics knowledge; jargon (JONSWAP, TRAA) moves to the README and an "about the tech" corner
- [x] README rewritten to sell the experience first (hero gif/images, one-line pitch, play link, then the tech), modelled on the best showcase repos
- [x] `PRODUCT.md` updated: the voice/instrument split is redrawn (player-facing text gets a voice; readouts stay plain)
- Done 2026-11-13: voice guide in `PRODUCT.md` › Voice; first-minute lessons in `src/ui/Coach.ts`
  (look, helm, sail, steer, cargo, anchor; keyboard and touch wording; learned set kept in
  localStorage); loading lines in `src/ui/LoadingTips.ts`; ledger tooltips and *About the tech*.
  Not done: landscape phones show no lessons (no free slot); the storm quote is unchanged.

### Follow-ups found while merging
- [x] Post-ready main-thread stall (25 s on a contended GPU vs 3.8 s on `main`): prewarm times out and the rest compiles in one frame (assets agent) — fixed: the prewarm now compiles the frame's real contexts (scene pass target + TRAA MRT + velocity, pinned call depths, reflection), pipelines build in parallel, live frames never build a pipeline synchronously. Same session, High: `isReady` ~48 s → ~17 s, worst frame after ready ~1.8 s → <50 ms, loop start ~17 s → ~7 s.
- [x] Night/dusk clouds no longer outshine the sky (lit by measured sky radiance; moon as key)
- [x] Sea sun glint fades with overcast on the sky's disc curve
- [ ] Bright horizon band at night/storm (clear dome sky between the deck's far edge and the sea); dusk squall sea reads purple (sky agent)
- [ ] Underwater: jellyfish read as opaque blobs at distance; god rays faint; island shelf sparse; reef 17 m down not visible from just above the surface
- [ ] Bommie LOD (12k triangles each) for Ultra/Max triangle budget
- [ ] Unexplained striped light column on the water in a helm view — recheck once the GPU is quiet

### F. Verification
- [ ] Progress screenshots each milestone
- [ ] `npm run typecheck`, functional suite, rebaseline visual suite deliberately
- [ ] Codex review of the final diff
- [x] Capture determinism: hero shot differs 0.22% run-to-run on this branch vs 0.18% on `main`, same patch beside the ship — pre-existing, not introduced here
