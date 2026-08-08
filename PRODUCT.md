# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Someone who has followed a link to *look at something*. They arrive on a desktop
browser, usually from a repository listing, a graphics forum or a share, and they
are there for the render rather than for a task. They will spend somewhere between
fifteen seconds and a few minutes, and what they do in that time is drag the
camera around, pull a slider to see what changes, and try a different preset.

Confirmed by the user as the primary audience: **this is a showcase / portfolio
piece — the ocean is the product, and the interface should recede so the render
leads from the first viewport.**

Two smaller audiences exist and must not drive the design:

- Graphics engineers reading the source, who care about the FFT cascades, the
  tier ladder and the frame budget. They are served by the README and the docs,
  not by the on-screen UI.
- The project's own author, using the panel as an instrument while tuning. The
  debug toggles (`buoyancyProbes`, `wakeProbes`, `forceWebGL`) exist for this and
  are not part of the visitor's story.

## Product Purpose

A realtime spectral ocean and tropical island, rendered in the browser, that is
worth looking at without being explained.

It exists to demonstrate that a full offline-quality water pipeline — FFT wave
synthesis, physically motivated water optics, foam, caustics, buoyancy, wakes,
underwater transitions, a volumetric sky and a photographic lens model — runs at
interactive rates on the web from one shader source.

Success is that a visitor stays, moves the camera, and comes away believing the
water. Failure is a visitor who reads the frame counter and leaves.

## Positioning

Nothing here is a video, a baked flythrough or a captured sequence. Every wave is
solved from a JONSWAP spectrum on the GPU, every frame. The mechanism a
neighbouring demo could not truthfully copy is the combination of:

- a **three-cascade Cooley–Tukey IFFT** run as ping-ponged fullscreen passes, all
  cascades packed into one atlas so the whole transform costs 23 render passes;
- **one TSL source compiling to both WebGPU and WebGL2**, so the fallback is the
  same renderer rather than a different, simpler one;
- a **photographic finish** — thin-lens circle of confusion, veiling glare over a
  capped highlight, an occludable sun-anchored flare, and a per-preset ASC CDL
  grade — applied to a scene that is being simulated rather than played back.

## Operating Context

Desktop browser, one tab, mouse and keyboard, usually a single sitting with no
return visit and nothing saved. There is no account, no persistence and no back end
the app talks to. The one exception to "no data" is Vercel Web Analytics, injected
at boot from `main.ts` — aggregate page traffic, no account and nothing the visitor
enters, because there is nothing to enter. Touch and coarse-pointer use is real but secondary: below
900px the control panel becomes a bottom sheet and Boat mode gains an on-screen
throttle ring.

The visitor cannot break anything and there is nothing to recover from, which is
why the interface carries no confirmation, no undo and no error surface beyond the
renderer's own backend fallback.

## Capabilities and Constraints

**What the visitor can drive**

- Four camera modes — Orbit, Fly, Boat, Cinematic — on buttons and on keys 1–4.
  Cinematic is an automatic 60-second tour in six beats, carrying its own hour and
  weather so one lap runs a full day and a squall; Boat gives them the helm.
- Nine looks: `skyPro`, `arctic`, `blackFlag`, `dusk`, `foggy`, `moonlit`,
  `seaOfThieves`, `storm`, `sunset`.
- Five quality tiers: Low, Medium, High, Ultra, Max. High is the default. An
  adaptive watcher steps the tier *down* when the frame rate stays under target,
  one way only within a session.
- Sliders: wind speed (0.5–25 m/s), peak wavelength (10–150 m), cloud coverage,
  time of day (0–24 h, overrides the preset's sun once touched), fog density,
  master volume, pixel ratio (0.5–2).
- Debug toggles: buoyancy probes, wake probes, force WebGL.

**What the interface must report**

- Frame rate, written at ~4 Hz so a 120 Hz loop cannot force 120 UI repaints a
  second.
- Renderer backend, and specifically **whether WebGL2 fallback is in force** — on
  that path there is no planar reflection, no refraction and no lens-rain misting,
  so it is the difference between what the visitor is seeing and what the project
  can do.
- Boot progress, because first paint waits on shader compilation and asset decode
  and can take several seconds.

**Hard constraints**

- The UI is an overlay on a canvas that owns the whole viewport. `#ui-root` is
  click-through by default and each surface opts back in; anything full-bleed must
  not swallow pointer events, or it takes the camera drag with it.
- `src/ui` imports nothing from three.js. The control layer mounts and tests
  standalone, and that boundary is load-bearing.
- Every pixel of chrome competes with the render for GPU time and for attention.
  Backdrop blur over a full-screen 3D canvas is not free.
- 15 canonical screenshots are compared against baselines on every run. The visual
  suite excludes the UI by rendering offscreen, but layout changes are not free
  either — the interface shot is a real compositor screenshot.
- Reduced-motion is honoured globally today and must stay honoured.

## Brand Commitments

- The name is **Web Ocean 3D**. It appears in the boot overlay and the panel head.
- The only *voice* in the product is the storm quote — one line of serif italic
  over the sea when the visitor takes the helm in heavy weather. It is deliberately
  set in a different face from every other piece of text, because it is the one
  thing that is not instrumentation. This distinction is binding.
- Nothing else about the current visual treatment is binding. The redesign that
  this section once called for has shipped: the dark navy glass surface and its
  blue accent are gone, replaced by the printed-reference-page system recorded in
  `DESIGN.md`. That system is a record of what was built, not a second binding
  commitment — the only binding one is the voice/instrument split above.

## Evidence on Hand

Real, in the repository:

- `docs/images/` — gallery renders regenerated from the renderer by one command,
  and last regenerated against the current build.
- `tests/baselines/` — 15 canonical shots at 1280×720.
- `docs/PERFORMANCE.md` — measured frame budget, per-tier.
- Measured on an RTX 5090 at Max: 13.72 ms GPU p50 against a 16.7 ms budget, 48 FPS
  delivered. The frame roughly tripled when the fidelity work landed and still
  passes both gates; `docs/PERFORMANCE.md` carries the trade and the caveats.

Absences that must not be invented: there are no testimonials, no customers, no
pricing and no press. The project is a personal graphics demo. Any copy implying
adoption, a team, or a product business would be fabrication — and note that the
Vercel deployment and its traffic counter are **not** evidence against that. A
page-view number is not a user, and nothing here licenses copy that treats it as
one.

## Product Principles

1. **The render is the product; the interface is the frame around it.** Any pixel
   of UI has to earn its place against showing more ocean.
2. **Never lie about what is being shown.** The backend badge, the tier and the
   frame rate are truth-telling instruments, not decoration — a visitor on the
   WebGL2 fallback is seeing a genuinely reduced scene and has to be able to know.
3. **Nothing is destructive, so nothing needs guarding.** No confirmations, no
   undo, no modal interruptions. The cost of any action is one drag to put it back.
4. **The controls are an invitation, not a cockpit.** A visitor who touches
   nothing should still get the best version; a visitor who touches everything
   should not be able to reach a broken one.
5. **Instrumentation and voice stay separate.** Readouts read as software. The one
   line that is authorial does not.

## Accessibility & Inclusion

- Every interactive control carries a visible focus ring today; keep it.
- Camera modes are reachable by keyboard (1–4) as well as by pointer.
- `prefers-reduced-motion` is honoured globally.
- The FPS readout is `role="status"` with `aria-live="off"` — deliberately not
  announced, because a value changing four times a second would make a screen
  reader unusable.
- No product-specific standard (WCAG level, audit) has been established by the
  user. Contrast and target size are to be held to ordinary good practice rather
  than to a stated conformance target.
