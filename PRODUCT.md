# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Someone who has followed a link to a game they can play in a single sitting. They
arrive on a desktop browser, usually from a repository listing, a graphics forum
or a share, and the first thing they see is the sea. They will give it somewhere
between five and twenty minutes: long enough to learn the ship, sail to
something worth reaching, and stop when they choose to. What they do in that
time is steer, set the sail, take the camera off the ship to look at the water,
and try another look or another hour of the day.

The render still leads. The sea, the ship and the light are what make someone
follow the link, and the game exists to give them a reason to stay inside that
picture. This replaces the earlier brief, which treated the project as a
showcase and expected a stay of fifteen seconds to a few minutes. That was a
description of what visitors did when nothing was asked of them. It is not a
limit to design to now.

Two smaller audiences exist and must not drive the design:

- Graphics engineers reading the source, who care about the FFT cascades, the
  tier ladder and the frame budget. They are served by the README and the docs,
  not by the on-screen UI.
- The project's own author, using the panel as an instrument while tuning. The
  debug toggles (`buoyancyProbes`, `wakeProbes`) exist for this and are not part
  of the player's story.

## Product Purpose

A short-session sailing game on a realtime spectral ocean, set around 1700 in
the Caribbean and rendered entirely in the browser. It is worth looking at
without being explained, and worth playing for ten minutes without being taught
twice.

It exists to show that a full offline-quality water pipeline (FFT wave
synthesis, physically motivated water optics, foam, caustics, buoyancy, wakes,
underwater transitions, a volumetric sky and a photographic lens model) runs at
interactive rates on the web, and that it can carry a game: a ship that handles
like a ship on that water, and a voyage with a start, some risk and an end.

Success is a player who sails for a while, stops to look at the water, and
believes it. Failure is a player who reads the frame counter and leaves, or one
who finds the sea beautiful and has nothing to do on it.

The game is measured against shipped sailing games (Sea of Thieves, Assassin's
Creed IV: Black Flag, Skull and Bones, Dredge, Sailwind) for the qualities they
share: a sea, a ship and weather that feel alive. Their design principles are
studied. Their names, characters, missions, art, music and interface are not
taken. `docs/AAA_ROADMAP.md` carries the plan and `docs/REFERENCE_TARGETS.md`
the per-look comparison.

## Positioning

Nothing here is a video, a baked flythrough or a captured sequence. Every wave
is solved from a JONSWAP spectrum on the GPU, every frame, and the ship floats
on the same water the camera looks at. The mechanism a neighbouring demo could
not truthfully copy is the combination of:

- a **three-cascade Cooley-Tukey IFFT** run as ping-ponged fullscreen passes, all
  cascades packed into one atlas so the whole transform costs 23 render passes;
- **one TSL source on WebGPU**: a single renderer with the scene MRT, temporal
  antialiasing, the planar mirror, refraction and the underwater pass, and no
  lesser fallback (the WebGL2 path was retired with the r187 upgrade);
- a **photographic finish**: thin-lens circle of confusion, veiling glare over a
  capped highlight, an occludable sun-anchored flare, and a per-preset ASC CDL
  grade, applied to a scene that is being simulated rather than played back;
- a **ship that sails on that wind**: a sail polar with a no-go zone, keel drag,
  a rudder that needs way on, and heel as real torque against the buoyancy
  solver, so handling and water are one system.

It is a game, but a small one so far: one island, one repeating objective, no
combat. Copy must not describe it as more than it is. The plan to grow it is in
`docs/AAA_ROADMAP.md`.

## Operating Context

Desktop browser, one tab, mouse and keyboard, a single sitting of five to twenty
minutes. There is no account and no back end the app talks to. Two small things
are kept in the player's own browser (localStorage, and the app works without
it): the menu choice of graphics and quality, which first-minute lessons have
been learned, so a returning player is not taught to drag the camera twice, and
the player's own settings (sound levels, keys, comfort and access options).
The one exception to "no data" is Vercel Web Analytics, injected at boot from
`main.ts` — aggregate page traffic, no account and nothing the player enters,
because there is nothing to enter. Touch and coarse-pointer use is real but
secondary: below 900px the control panel becomes a bottom sheet and the helm
gains an on-screen stick.

The player cannot break anything outside the game, and nothing in it is lost for
good, which is why the interface carries no confirmation and no undo. The error
surfaces are a browser without WebGPU and a failed boot, each of which says what
happened and what to try next.

## Capabilities and Constraints

**What the player can drive**

- Four camera modes (Orbit, Fly, Boat, Cinematic) on buttons and on keys 1–4.
  Boat is the helm: sail setting (furled, half, full), steering and the anchor,
  with a drifting-cargo objective. Cinematic is an automatic 90-second tour in
  nine shots, carrying its own hour and weather so one lap runs a full day and a
  squall.
- Nine looks: `skyPro`, `arctic`, `blackFlag`, `dusk`, `foggy`, `moonlit`,
  `seaOfThieves`, `storm`, `sunset`.
- Five quality tiers: Low, Medium, High, Ultra, Max. High is the default. An
  adaptive watcher steps the tier *down* when the frame rate stays under target,
  one way only within a session.
- Sliders: wind speed (0.5–25 m/s), swell length (10–150 m), cloud cover, time
  of day (0–24 h, overrides the preset's sun once touched), sea fog, volume,
  music volume, sharpness (pixel ratio 0.5–2).
- Debug toggles: buoyancy probes, wake probes.

**What is not in the game yet**

Stated so that copy and the interface do not imply them. Each is a task in
`docs/AAA_ROADMAP.md`.

- No failure state, ending or saved progress. Gold buys nothing, and only the
  quality choice and learned lessons are kept.
- No other ships, crew, enemies or combat, and nothing on the island can be
  interacted with.
- No photo mode and no per-button gamepad rebinding. Pause, key rebinding,
  look speed and invert, rumble, shake, reduced motion, reduced flashes,
  interface size and shape cues are in (the shell, G8).

**What the interface must report**

- Frame rate, written at ~4 Hz so a 120 Hz loop cannot force 120 UI repaints a
  second.
- Renderer backend. WebGPU is the only one, so the badge reads WebGPU; a browser
  without it is told what it needs rather than given a lesser sea.
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
- The render stays inside its frame budget as the game grows. Each new system has
  a cap and a cheaper fallback per tier (`docs/BUDGETS.md`).
- Reduced-motion is honoured globally today and must stay honoured. Anything
  that flashes stays within three flashes a second.

## Brand Commitments

- The name is **Web Ocean 3D**. It appears in the boot overlay and the panel head.
- **Player-facing text has a voice; readouts stay plain.** Everything the
  player reads in play — the menu, the loading lines, objectives, toasts,
  lessons, ledger titles and labels, finish and error states — is written in
  one original seafaring voice (see *Voice* below). The readouts — frame rate,
  backend, quality tier, frame time, the probe line on the menu, compass and
  wind figures — are instrumentation: plain, exact, and never dressed up. This
  split is binding.
- **The storm quote is the most authorial line, and keeps its own face.** One
  line of serif italic over the sea when the player takes the helm in heavy
  weather. Other copy has a voice; this is the one line that is the author
  speaking, which is why it alone is set in IM Fell English italic.
- Nothing else about the current visual treatment is binding. The interface is
  now an original nautical-adventure language — parchment, brass, wood, ink and
  wax, in the spirit of the genre but borrowing no game's name, fonts, icons or
  art — recorded in `DESIGN.md`. It replaced the printed-reference-page system,
  which replaced the dark navy glass before it. That record is of what was
  built, not a second binding commitment — the binding ones are the
  voice/readout split and the storm quote's face above.

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
pricing and no press. The project is a personal graphics demo that has grown a
game. Any copy implying adoption, a team, or a product business would be
fabrication — and note that the Vercel deployment and its traffic counter are
**not** evidence against that. A page-view number is not a user, and nothing
here licenses copy that treats it as one.

## Product Principles

1. **The render leads; the game gives it a reason to stay.** The sea, the ship and
   the light are what earn the link. Any pixel of UI, and any system added for the
   game, has to earn its place against showing more ocean and holding its frame
   budget.
2. **Never lie about what is being shown.** The backend badge, the tier and the
   frame rate are truth-telling instruments, not decoration. A player on a lower
   tier is seeing a reduced scene and has to be able to know. The same goes for
   copy: it describes the game that exists.
3. **Short sessions, no punishment.** A visit is five to twenty minutes and can end
   at any moment. Failure costs time or cargo, never the session; there are no
   confirmations, no modal interruptions, and the cost of any action is one move to
   put it back.
4. **The controls are an invitation, not a cockpit.** A player who touches nothing
   should still get the best version; one who touches everything should not be able
   to reach a broken one. Options that help comfort and access (motion, flashing,
   sound, input) are part of the game, not a developer panel.
5. **Voice for the player, plain for the readouts.** Copy the player reads in
   play has a voice; instruments read as instruments. Nothing in the voice may
   blur a readout — a reduced tier is never softened into something kinder.
6. **Take qualities, not content.** Shipped sailing games are the bar for how the
   sea, ship and weather feel. Their names, art, music and interface are not
   ours to use.

## Voice

Original, seafaring, and the player's own: warm, a little wry, confident and
short. Never pirate-speak, never parody, never a pastiche of one game; no
line, name or catchphrase borrowed from any.

- **Say what to do, then stop.** "Shallows! Turn her away." The instruction
  comes first; the flavour is in the word choice, not in extra words.
- **Plain words over sailing words when they compete.** "Stalled head to wind"
  rather than "in irons"; "Hauling the anchor up" rather than "weighing". A
  nautical word stays only when the plain one would be longer or vaguer
  ("helm", "cargo").
- **Kind about failure.** A stall or a shoal is news and a way out, never a
  scolding. An error says what happened and what to try, and puts the raw
  message underneath for a bug report.
- **Teach by doing.** Controls are taught one at a time, at the moment they
  are needed, and the lesson leaves once it has been done. Lessons and tips
  are phrased for the hands in use: keys for a keyboard, buttons and gestures
  for touch.
- **No jargon in a label.** FFT, JONSWAP, TRAA and their kin live in the
  README and the ledger's folded *About the tech*, each explained in a
  sentence. A label or tooltip has to make sense to someone who has never
  heard of a shader.
- **Truthful flavour.** A tip may be charming, but it must be true of what is
  on screen ("Gulls ride the yards until she gets under way" is in the scene).
  No line implies an audience, a team or a product the project does not have.

## Accessibility & Inclusion

- Every interactive control carries a visible focus ring today; keep it.
- Camera modes are reachable by keyboard (1–4) as well as by pointer.
- `prefers-reduced-motion` is honoured globally. It is not yet honoured by the
  mode-change camera sweep, the beacon pulse or the lightning; those are listed
  in `docs/AAA_ROADMAP.md` (F10, X7).
- The FPS readout is `role="status"` with `aria-live="off"` — deliberately not
  announced, because a value changing four times a second would make a screen
  reader unusable.
- Lightning, muzzle flashes and any pulse stay within three flashes a second
  (WCAG 2.3.1), with a reduce-flashes option once they are built.
- No product-specific standard (WCAG level, audit) has been established by the
  user. Contrast and target size are to be held to ordinary good practice rather
  than to a stated conformance target.
