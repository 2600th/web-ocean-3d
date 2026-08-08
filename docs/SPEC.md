# Web Ocean 3D — Design Spec & Feature Matrix

Target: recreate the *visible experience and functionality* of the Three.js Water Pro live
demo as an original, production-quality implementation.

**Provenance note.** The reference is a commercial product. Nothing in this repository is
derived from its source bundle or its shipped assets. Every technique here is implemented
from published literature (Tessendorf FFT ocean, JONSWAP/Pierson–Moskowitz spectra,
Hosek–Wilkie / Preetham sky models) and MIT-licensed Three.js examples. All 3D and HDRI
assets are sourced independently under CC0 and recorded in `ASSET_LICENSES.md`.

---

## 1. Feature matrix

Observed in the reference across the captures in `reference/shots/`.

| # | Feature | Observed behaviour | Priority |
|---|---------|--------------------|----------|
| F1 | Spectral ocean | Large-scale swell + wind chop + fine ripples, directional, wind-driven. Wavelength and wind speed are live-tunable. | P0 |
| F2 | Shoreline/shallow water | Turquoise shallows, visible seafloor, depth-graded absorption from teal → deep blue. | P0 |
| F3 | Water lighting | Fresnel sky reflection, sun specular glitter, subsurface scattering glow on wave backs, Beer–Lambert depth absorption. | P0 |
| F4 | Reflection & refraction | Sky/scene reflection on the surface; refracted seafloor and submerged hull, distorted by surface normals. | P0 |
| F5 | Foam | Whitecaps on wave crests (Jacobian/folding driven), shoreline foam, persistent wake foam trailing the ship. | P0 |
| F6 | Caustics | Animated light caustics on the seafloor and on submerged geometry; visible from above through clear shallow water. | P0 |
| F7 | Underwater state | Full transition when the camera crosses the surface: blue-green volumetric fog, god rays / light shafts, drifting particulates, bubble columns, desaturated distance. | P0 |
| F8 | Waterline transition | Correct half-submerged framing when the camera sits at the surface; no popping. | P1 |
| F9 | Atmosphere / sky | Physically based sky with sun position per preset, volumetric cloud layer with live coverage control, stars at night, rain in Storm. | P0 |
| F10 | Buoyancy | Ship and buoys ride the wave surface — heave, pitch, roll sampled from the displacement field. Debug "Buoyancy Probes" toggle. | P0 |
| F11 | Wakes | Ship generates a persistent foam wake and surface displacement. Debug "Wake Probes" toggle. | P1 |
| F12 | Presets | 9: Three.js Sky Pro, Arctic, Black Flag, Dusk, Foggy, Moonlit, Sea of Thieves, Storm, Sunset. Each sets sun/sky, water colour, wind, wavelength, cloud coverage, weather FX. | P0 |
| F13 | Camera modes | Orbit (LMB rotate / RMB pan / scroll zoom), Fly (WASD + mouse look), Boat (chase camera on the ship), Cinematic (a looping 60 s authored tour in six beats — open water, outbound, landfall, return, squall, reef run — carrying its own hour and weather). Keys 1/2/3/4. | P0 |
| F14 | Quality tiers | Low / Medium / High / Ultra / Max — scales cascade count, FFT resolution, mesh LOD, post FX, shadows. | P0 |
| F15 | Renderer fallback | WebGPU by default with a "Force WebGL" toggle producing a coherent — **not** visually equivalent — WebGL2 path. Refraction, planar reflection and SSR are absent there by policy; the surface falls back to analytic depth colour and an analytic sky reflection. | P0 |
| F16 | Pixel ratio control | Live 0.5×–2× resolution scale slider. | P1 |
| F17 | HUD | FPS counter (green→red by health), camera mode switcher, control hints. | P0 |
| F18 | Control panel | "Tide Table": an opaque printed reference page, not a glass pane — quality, preset, wind speed, peak wavelength, cloud coverage, time of day, volume, pixel ratio, three debug toggles, and a View Source link. The stylesheet refuses `backdrop-filter` and `border-radius` outright, both as a look and as a per-frame cost on the surface whose budget is the product. | P0 |
| F19 | Scene dressing | Sailing ship, buoys, barrels and an instanced rock/cliff island; over and around it a billboard canopy, island grass, kelp and seagrass on the shallow bottom, fish schools, a gull flock, and a pirate's remains in the sand. | P1 |
| F20 | Responsive | Panel collapses to a bottom sheet on narrow viewports; touch devices additionally get on-screen throttle and rudder (`ui/TouchControls.ts`) rather than only the orbit camera's built-in gestures. | P1 |

## 2. Visual-quality checklist

Derived from the reference captures. Each item is a pass/fail gate for the comparison loop.

**Water surface**
- [ ] Wave crests are sharp and slightly peaked, not sinusoidal — choppiness (horizontal displacement) is visibly present.
- [ ] Three distinct spatial scales are simultaneously legible: swell, chop, ripple.
- [ ] Whitecaps appear only where the surface folds, and dissipate over ~1–2 s.
- [ ] Sun glitter is anisotropic and stretched toward the viewer, not a round blob.
- [ ] Water colour transitions from turquoise (shallow, seafloor visible) to deep navy with distance/depth.
- [ ] Backlit wave faces glow with subsurface scattering.
- [ ] Horizon meets the sky without a visible seam or tiling repeat.

**Atmosphere**
- [ ] Sky gradient and sun disc match the preset's time of day.
- [ ] Clouds are volumetric-looking with lit tops and shadowed bases, and drift.
- [ ] Aerial perspective fades distant water into the horizon haze.

**Underwater**
- [ ] Crossing the surface is a continuous transition, not a hard cut.
- [ ] God rays originate from the sun direction and are occluded by the ship hull.
- [ ] Particulates drift slowly; bubble columns rise and wobble.
- [ ] Visibility falls off exponentially with distance in a blue-green tint.
- [ ] Looking up shows the surface underside with total internal reflection near the Snell window edge.

**Objects & motion**
- [ ] Ship heave/pitch/roll is phase-correct with the waves under it.
- [ ] Wake foam trails behind the ship and persists, widening astern.
- [ ] Buoys bob independently and correctly.

> Removed from this checklist: *"ship shadow lands on the water and reads through
> into the shallows"*. The surface is a `MeshBasicNodeMaterial` and receives no
> shadow at all, so the item could never pass. It returns only if the surface is
> given a lighting model that accepts one.

**UI**
- [ ] Panel typography, spacing, and glass treatment are crisp at 1× and 2× DPR.
- [ ] Sliders and toggles animate smoothly; values update live.
- [ ] FPS meter colour-codes by frame health.
- [ ] No layout shift or overflow at 360px–2560px widths.

## 3. Architecture

```
src/
  main.ts                 bootstrap, wires everything together
  core/
    Renderer.ts           WebGPURenderer + WebGL fallback, resize, pixel ratio
    Loop.ts               frame loop, clamped delta, frame timing
    QualityManager.ts     tier definitions + adaptive downscale on sustained low FPS
    tslMath.ts            ramps and helpers whose obvious spelling is undefined in WGSL
    random.ts             one seeded generator, shared by every system that places
    lightOcclusion.ts     an extra occlusion factor on one light, for one material
  ocean/
    Spectrum.ts           JONSWAP directional spectrum -> initial h0 texture
    FFT.ts                Stockham radix-2 IFFT fragment passes (TSL)
    OceanSimulation.ts    per-frame cascade evolution -> displacement/derivatives
    OceanMesh.ts          camera-centred radial grid with geometric ring spacing
    meshSampling.ts       what the mesh can resolve; derives ring/segment counts
    OceanMaterial.ts      surface shading node graph
    Reflections.ts        planar reflection of the scene in the surface
    ScreenSpaceReflection.ts  traced reflected ray, composited with the planar pass
    Sampler.ts            CPU-side height/normal readback for buoyancy
  sky/
    Atmosphere.ts         analytic sky + sun/moon disc + stars + env capture
    Clouds.ts             raymarched volumetric cloud layer
    AerialPerspective.ts  one per-channel distance haze for the whole frame
    Weather.ts            rain / snow particle systems
  post/
    DepthOfField.ts       thin-lens circle of confusion, spiral gather
    Bloom.ts              veiling glare (three's mip pyramid, capped input)
    LensFlare.ts          sun/moon-anchored, depth-occluded, above water only
    ColorGrade.ts         ASC CDL + saturation + cos^4 vignette, per preset
    SpatialAA.ts          spatial resolve between the transfer function and the dither
    OutputTransform.ts    ACES, sRGB and a triangular dither -- taken over from
                          the renderer so there is a stage late enough to dither
    VolumetricFog.ts      marched height fog and light shafts
    LensRain.ts           refracting droplets on the front element
  underwater/
    UnderwaterPass.ts     water column, god rays, per-pixel waterline
    Particles.ts          particulates + bubbles
    Caustics.ts           procedural caustic field shared by submerged materials
  scene/
    AssetLoader.ts        caching, deduplicating glTF loader
    Seafloor.ts           terrain + depth field feeding shallow-water shading
    groundShading.ts      caustics, contact darkening and wind for anything on the ground
    Props.ts              buoys, barrels, and an instanced rock/cliff island
    Ship.ts               ship model, hull normalisation, buoyancy probe layout
    Canopy.ts             distant canopy billboards over the island
    Meadow.ts             island grass, placed and animated in the vertex stage
    Kelp.ts               kelp and seagrass on the shallow bottom, GPU-animated
    Fish.ts               schools over the plateau, the reef and the shallows
    Birds.ts              a gull flock in one instanced draw
    Remains.ts            a pirate's remains, half-buried in the sand
    Wetness.ts            rain wetting for loaded glTF surfaces
  physics/
    Buoyancy.ts           probe-based rigid-body float
    ShipController.ts     throttle and rudder as forces into the buoyancy solver
    Wake.ts               world-anchored wake foam accumulation buffer
    Spray.ts              airborne spray thrown by a hull working in a seaway
  cameras/
    CameraDirector.ts     orbit / fly / boat / cinematic modes and the transitions
    Cinematic.ts          the authored tour: beats, cyclic spline, derived rudder
  audio/
    AudioSystem.ts        the scene's ambience, synthesised rather than sampled
    NoiseBed.ts           one continuous noise layer: source, filters, gain, pan
    ParamTarget.ts        an AudioParam bundled with the value it is driven toward
    noise.ts              the noise buffers the beds loop
  ui/
    Panel.ts, Hud.ts, TouchControls.ts, Quote.ts, types.ts, styles.css
  presets/
    index.ts              9 preset definitions
```

Barrel `index.ts` re-exports are omitted. Otherwise the tree above is the tree on
disk — and it is worth saying that this was once not true. Earlier revisions
named `core/Disposer.ts`, `scene/Fish.ts` and a `cameras/` module per mode, none
of which had been written; one `CameraDirector.ts` owns all three modes, and
`Fish.ts` exists today only because it was built later. A tree in a spec is a
claim like any other.

### Key decisions

**WebGPU + TSL with automatic WebGL2 fallback.** Three's `WebGPURenderer` compiles the same
TSL node graph to WGSL or GLSL, so one authored shader set serves both backends. This
directly supports the reference's "Force WebGL" toggle without a parallel codebase.
Rationale: avoids maintaining two shader languages, and compute-based FFT degrades to a
fragment-shader FFT on the WebGL path.

**FFT ocean over Gerstner sums.** A Gerstner sum needs hundreds of waves to look
non-repetitive; an FFT gives a full spectrum at fixed cost and yields the Jacobian needed
for physically motivated whitecaps. Cascades at three tile sizes (1024 m / 128 m / 16 m,
per `CASCADES` in `ocean/Spectrum.ts`) remove visible tiling.

**Radial ocean mesh, not a clipmap.** A camera-centred radial grid keeps triangle density
high near the viewer and cheap at the horizon, and avoids the seams and popping of discrete
LOD rings. This decision was originally written up as a clipmap, which the implementation
never was; `OceanMesh.ts` carries the reasoning that rejected one. Ring and segment counts
are derived rather than tuned — see `squareGrid` in `ocean/meshSampling.ts`, which keeps
radial and angular vertex spacing square so the coarser axis does not govern.

**GPU→CPU readback for buoyancy.** A small (64²) height slice is read back asynchronously
each frame so ship physics stays on the CPU without stalling the pipeline.

## 4. Measured behaviour

Numbers read directly off the GPU rather than judged by eye.

**These are readings, not assertions, and they predate the current spectrum.** The
distinction matters: a number someone once read off a GPU and a number the suite
will fail without look identical on a page, and this project has published the
first as though it were the second before. `tests/ocean.spec.ts`
gates the sea state on bounds wide enough to survive retuning — `peakHeight`
between 0.4 m and 12 m, folded area under 8 % — and never asserts any of the point
values below. They were read off a run on the reference GPU, last on 2026-08-04.
The spectrum was reworked on 2026-08-07 (`7d9a84c`), which moved every baseline in
the visual suite, so treat the column as indicative until it is re-measured.

| Quantity | Measured | Expected | Notes |
|---|---|---|---|
| Crest amplitude, cascade 0 | ±1.4 m | metres | 15 m/s wind, 47 m peak |
| Surface RMS elevation | 0.433 / 0.322 / 0.066 m | decreasing per cascade | swell / chop / ripple |
| Mean Jacobian | 0.86 / 0.95 / 0.98 | just under 1 | below 1 means net folding |
| Folded area (J < 0) | 0.1 % | a fraction of a percent | was 8.8 % before the FFT fix |
| Significant wave height | 2.17 m | see below | |

Whitecap coverage used to sit in this table at "4.6 %, matches observed sea state".
No test measured it; the figure was attached to a sea-state assertion that reads
displacement Jacobians and never touches the foam buffer. It is removed here for
the same reason `README.md` removed it. What replaced it is a real measurement:
`tests/foam.spec.ts` renders the surface at 6, 9, 15, 21 and 24 m/s and requires
coverage to sit within a factor of three of Monahan & O'Muircheartaigh's `U^3.41`
law and to rise monotonically with wind. A factor of three is not false precision —
the law is a fit to scattered ship observations, and the buffer holds a decaying
trail rather than an instantaneous census.

**On significant wave height.** The Pierson–Moskowitz relation `Hs ≈ 0.22 U²/g` gives
5.05 m at 15 m/s, which our 2.17 m appears to miss badly. It does not: PM describes a
*fully developed* sea, whose spectral peak at 15 m/s sits near 190 m. The demo exposes
peak wavelength as a control and defaults it to 47 m, which is a fetch-limited sea — a
shorter, steeper, lower-amplitude state. A fetch-limited spectrum is *expected* to carry
less variance than the fully developed one at the same wind speed, so the two figures
are not comparable and the simulation is not under-energised. Comparing against PM is
only valid with the peak wavelength set to the fully developed value for the chosen wind.
