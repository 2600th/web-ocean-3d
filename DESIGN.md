---
name: Web Ocean 3D
description: A nautical adventure's instruments on the captain's table — torn chart paper, stained oak, polished brass and wax, laid over a live sea.
colors:
  parchment-hi: "#f6ead0"
  parchment: "#ecdcb6"
  parchment-lo: "#d8c08f"
  ink: "#2b1d10"
  ink-soft: "#5b4129"
  ink-faint: "#8f714d"
  wood-hi: "#3d2a18"
  wood: "#2a1b0f"
  wood-deep: "#170e07"
  on-wood: "#f1e2bf"
  on-wood-dim: "#c4ab7f"
  brass-hi: "#f7dc92"
  brass: "#c8973f"
  brass-lo: "#7a5519"
  brass-edge: "#2a1806"
  wax: "#8e1b17"
  wax-hi: "#c23a2c"
  gold: "#ffd36b"
  verdigris: "#2f7d6d"
  verdigris-hi: "#6cc4aa"
  wind: "#0f5f6b"
  ok: "#9ccf72"
  warn: "#f0b042"
  bad: "#ff7a5c"
  sea-night: "#0b1a20"
typography:
  display:
    fontFamily: "Pirata One, Palatino Linotype, Georgia, serif"
    fontSize: "clamp(54px, 10vw, 96px)"
    fontWeight: 400
    lineHeight: 0.9
    letterSpacing: "0.005em"
  heading:
    fontFamily: "Pirata One, Palatino Linotype, Georgia, serif"
    fontSize: "21px"
    fontWeight: 400
    lineHeight: 1
  label:
    fontFamily: "Alegreya, Palatino Linotype, Georgia, serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.13em"
  value:
    fontFamily: "Alegreya, Palatino Linotype, Georgia, serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.2
  figure:
    fontFamily: "Alegreya, Palatino Linotype, Georgia, serif"
    fontSize: "31px"
    fontWeight: 800
    lineHeight: 1
  body:
    fontFamily: "Alegreya, Palatino Linotype, Georgia, serif"
    fontSize: "16px"
    fontWeight: 500
    lineHeight: 1.45
  voice:
    fontFamily: "IM Fell English, Palatino Linotype, Georgia, serif"
    fontSize: "clamp(1.7rem, 3.6vw, 3.3rem)"
    fontWeight: 400
    fontStyle: italic
    lineHeight: 1.2
rounded:
  key: "4px"
  plaque: "7px"
  plate: "10px"
  tab: "10px"
  button: "15px"
  round: "50%"
spacing:
  hair: "4px"
  tight: "6px"
  row: "10px"
  group: "14px"
  edge: "20px"
  edge-mobile: "12px"
components:
  parchment:
    backgroundColor: "{colors.parchment}"
    textColor: "{colors.ink}"
    edge: "deckled mask, 10px tear, chamfered corners"
  plate:
    backgroundColor: "{colors.wood}"
    textColor: "{colors.on-wood}"
    border: "2px brass gradient"
    rounded: "{rounded.plate}"
  tab:
    backgroundColor: "{colors.wood-hi}"
    textColor: "{colors.on-wood-dim}"
    rounded: "{rounded.tab}"
    height: "48px"
  tab-active:
    backgroundColor: "{colors.brass}"
    textColor: "{colors.ink}"
  plaque:
    backgroundColor: "{colors.wood}"
    textColor: "{colors.brass-hi}"
    rounded: "{rounded.plaque}"
  plaque-selected:
    backgroundColor: "{colors.brass}"
    textColor: "{colors.ink}"
  cta:
    backgroundColor: "{colors.wax}"
    textColor: "#ffe6a6"
    rounded: "{rounded.button}"
    typography: "{typography.display}"
  range-track:
    backgroundColor: "#3b2614"
    height: "8px"
  range-fill:
    backgroundColor: "{colors.brass}"
  range-thumb:
    backgroundColor: "{colors.brass-hi}"
    size: "20px (28px coarse)"
    rounded: "{rounded.round}"
  toggle:
    border: "1.75px {colors.ink-soft}"
    mark: "{colors.wax}"
  keycap:
    backgroundColor: "{colors.parchment-hi}"
    textColor: "{colors.ink}"
    rounded: "{rounded.key}"
---

# Design

Recorded from the built interface, not from an intention written before it.

## Overview

The interface is **the instruments of a sailing ship laid out on the
captain's table**, over a live sea. It takes its genre cues from the nautical
adventure game — aged chart paper, brass fittings, ink line work, wax seals,
a hand-lettered display face — and builds its own versions of each from open
fonts and authored SVG. No mark, font, icon or artwork from any existing game
is used.

PRODUCT.md still governs: the sea is the product, and the chrome sits at the
corners and edges of the frame, never across its middle except for the moment
a toast is unrolled.

Three materials carry the whole system, each with exactly one job:

| Material | Job | Where |
|---|---|---|
| **Parchment** | Things you *read* | The ledger (panel), the objective scroll, toasts, the boot chart |
| **Wood in brass** | Things you *watch* | Compass tape, wind dial readings, score plaque, mode bar, plaques |
| **Polished brass** | Things you *touch*, and what is *selected* | Every rim, knob, active tab, chosen plaque, the fold button |

Two accents sit on top. **Wax red** means "this one": the north point, the
ledger's section heads, the Set Sail call, the seal on the chosen preset and
on the objective. **Gold** means reward: the objective mark on the compass,
the score figures, the doubloon toast.

Four rules:

1. **No glass.** There is no `backdrop-filter` anywhere. Blur over a
   full-screen 3D canvas costs GPU time on the surface whose frame budget is the
   product. Translucency is a flat alpha instead: wood plates are
   `--wood-a`/`--wood-deep-a` (84–88% oak), the hint ribbon and FPS corner
   `rgba(18,11,5,.7)`, and whole elements step back with `opacity` (see
   Attention). Parchment stays opaque: it is read, not looked through.
2. **Paper is torn, metal is turned.** Parchment never has a straight edge or
   a radius; wood and brass always have a radius. That one distinction is what
   tells a reading surface from an instrument at a glance.
3. **Selection is polish.** A chosen tab, plaque or preset becomes polished
   brass with ink engraved into it; an unchosen one is dark wood with brass
   inlay. There is no second selection colour.
4. **Tactile, never springy.** Buttons have a hard bottom edge (an offset
   shadow in the darker material) and press down by exactly that much on
   `:active`. Motion is exponential ease-out; nothing bounces.

## Colors

| Token | Value | Contrast | Use |
|---|---|---|---|
| `parchment` | `#ecdcb6` | — | Paper ground, with `-hi`/`-lo` for its gradient |
| `ink` | `#2b1d10` | 12.0:1 on parchment | All primary writing on paper |
| `ink-soft` | `#5b4129` | 6.9:1 on parchment (5.3:1 at the darkest edge) | Labels and secondary writing on paper |
| `ink-faint` | `#8f714d` | 3.3:1 | **Rules and drawn line work only, never text** |
| `wood` | `#2a1b0f` | — | Instrument ground, with `-hi`/`-deep` |
| `on-wood` | `#f1e2bf` | 13.0:1 on wood | Figures and primary text on wood |
| `on-wood-dim` | `#c4ab7f` | 7.5:1 on wood | Labels on wood |
| `brass` | `#c8973f` | — | Rims, knobs, fills; `--brass-grad` (rim) and `--brass-face` (polished face) |
| `wax` | `#8e1b17` | 6.7:1 on parchment | Section heads, seals, CTA ground, north |
| `gold` | `#ffd36b` | — | Reward: objective mark, score figures |
| `wind` | `#0f5f6b` | — | The wind arrow on the dial, and nothing else |
| `ok` / `warn` / `bad` | | | Frame-rate figure; `warn` also fills the WebGL2 fallback badge |

Focus rings are gold (`3px`, `3px` offset) on wood and wax-red on parchment,
because gold vanishes into paper.

## Typography

Three families, three files, 110 KB, self-hosted (see ASSET_LICENSES.md).

- **Pirata One** — the *display hand*. The project's name, the ledger's section
  heads, the objective title, plaque names, tab names, the Set Sail call. Never
  for a figure or a compass point: at small sizes its blackletter `S` reads as
  a `5`, which is why the compass cardinals are set in Alegreya 800.
- **Alegreya** — the *instrument*. Every label, value, figure, hint and toast.
  Labels are 12px/700 caps at `0.13em`; figures use `lining-nums
  tabular-nums` (Alegreya's default figures are old-style, which is right for
  prose and wrong for a column that must not jitter) with a `ch` min-width
  behind them.
- **IM Fell English italic** — the *authorial line*. The storm quote, and
  nothing else.

Player-facing copy has a voice (PRODUCT.md › Voice) and speaks it in Alegreya
and Pirata One like everything else; readouts stay plain. The storm quote is
the one line that is the author speaking, and its face is what says so: it
must never read as one more toast or one more readout. It used to be carried by Bodoni
against Bitter; it is now carried by a seventeenth-century italic against a
calligraphic book face and a display blackletter, which is a bigger gap than
before, not a smaller one.

**Scale.** Whole pixels on one ramp: 10 · 11 · 12 · 13 · 14 · 15 · 16 · 17 ·
18 · 20 · 21 · 25 · 31 · 38, and for the menu title only 40 · 50 · 66 and the
`display` clamp. 10px is the floor and is used only for letter-spaced caps
labels on a phone; running text never goes below 13px.

`font-synthesis: none` is set on the document. No italic of Alegreya ships,
so a stray `font-style: italic` renders upright rather than as a sheared roman.

## Layout

Desktop (≥ 900px), 20px edge gutter:

| Position | Surface |
|---|---|
| Top centre | Compass tape (≤ 580px, shrinks between the side columns) with the bearing plate hanging under it |
| Under the compass | Toast stack, newest on top, at most four |
| Top left (310px) | Objective scroll, then the score plaque |
| Top right (328px) | The ledger; folds to its title bar |
| Bottom left | Wind dial (116px) with the readings plate tucked behind it; the log appears at the helm |
| Bottom centre | Hint ribbon over the mode tabs |
| Bottom right | Instrument corner: FPS figure and backend badge |

Every edge is `--edge-t/r/b/l`: the gutter or the device's safe-area inset,
whichever is larger, so nothing sits under a notch, a rounded corner or the
home indicator.

The ledger starts **folded** to its title bar. The sea is the first thing a
visitor sees; the settings are one click away, not in the way.

Under 1240px the tab names drop and the tabs keep icon and key.

**Phone portrait (≤ 900px, 12px gutter).** The middle of the screen is the
ship's. Top: the compass tape, the instrument corner under its right end, and
the quest column on the left — the objective as a one-line **chip** (seal,
title, bearing) with a chevron that unfolds the detail; a new objective
arrives unfolded and settles to a chip after six seconds. The score is a
single ribbon of label–figure pairs. Toasts are a narrow strip, top centre,
clear of the quest column, over the mast tops rather than the hull. Bottom:
wind and log in one compact plate over an icon-only mode bar, and the 58px
brass Ledger button, which raises the ledger as a bottom sheet on
`transform`. At the helm the sail and anchor orders (icon over word) sit
above the steering stick on the right thumb.

**Phone landscape (height ≤ 520px).** Two columns and an empty middle: quest,
score and toasts (one at a time) down the left over wind, log and the mode
bar; the instrument corner and an icon Ledger button top right; the stick and
the orders bottom right. The ledger opens as a side sheet from the right.

Keycaps are hidden on coarse pointers.

`.panel__body` carries `min-height: 0` — a flex child refuses to shrink
without it, and the last controls get clipped instead of scrolled to. The body
also fades its last 22px out with a mask so a scrolled ledger says there is
more below.

## Elevation & Depth

Parchment lies on the sea and casts its shadow with
`filter: drop-shadow(0 10px 14px rgba(5,3,1,.42)) drop-shadow(0 2px 2px …)`,
so the shadow follows the torn outline. Wood plates use the box-shadow
equivalent (`0 8px 18px`) plus a 1px `brass-edge` hairline. Buttons add a hard
bottom edge in their darker material, which `:active` removes.

## Shapes and materials

- **Deckled edge.** `.parchment::before` is the paper; the content sits on
  top, so the tear never chews a scrollbar, a knob or a focus ring. The tear is
  a CSS mask of four tiling SVG strips (co-prime lengths 173/211/167/193 so no
  two edges repeat together) plus a solid core, intersected with four diagonal
  corner cuts so the strips do not leave a square tooth where they meet.
  Grain and foxing are two `feTurbulence` SVG tiles; the burnt rim is an inset
  shadow.
- **Brass.** A padding-box/border-box background pair: wood or a polished face
  inside, `--brass-grad` in a transparent 2–3px border. The dial bezel is a
  conic gradient, which reads as turned metal.
- **Iconography.** Authored SVG on a 24 grid, 1.75 stroke, round joins, in
  `src/ui/icons.ts`: orbit, gull (fly), ship's wheel (helm), spyglass (tour),
  compass needle, doubloon, hazard, skull, anchor, ledger, the X that marks the
  spot. The compass rose is drawn in code so its colours follow tokens.

## Components

- **Compass tape** — 2.6px per degree, ticks every 5° and 15° from two
  repeating gradients, labels every 15° with inter-cardinals bold and N in red.
  Moved by `translate3d` only. The objective is a gold X on the tape; off the
  visible arc it pins to the end and becomes an arrow. At the helm a verdigris
  triangle marks the ship's head.
- **Wind dial** — a compass card that turns against the view (so its north
  stays on the world's) with a fletched arrow on it showing where the wind
  goes; a brass lubber at the top marks the view. Readings: knots, "from" the
  sixteen-point name, and the Beaufort force and name. The arrow accumulates
  its angle so a change across north turns the short way.
- **Log** — labelled *Speed*: speed through the water in knots, one decimal,
  only at the helm.
- **Mode tabs** — wood buttons, the active one polished brass; icon, name in
  Pirata One, keycap digit. Keys 1–4 still drive them.
- **Instrument corner** — the frame rate (green/amber/red, written at ~4 Hz,
  `role=status`, `aria-live=off`) and the backend badge. On the WebGL2 fallback
  the badge reads "WebGL2 fallback" and fills amber: the scene is genuinely
  reduced (no planar reflection, refraction or lens misting) and the visitor
  must be able to know. These two stay truthful; they are demoted in size, not
  in honesty.
- **Toasts** — a parchment strip with a brass-rimmed medallion per kind (info
  verdigris, reward gold, warning amber, danger wax). Unrolled from the middle
  with `clip-path` and rolled up on leave; `role=status`, `aria-live=polite`.
- **Objective scroll** — wax seal, Pirata title, detail, and a bearing row with
  a needle that points at the objective relative to the view.
- **Score plaque** — `dl` rows of caps label and gold figure; a changed figure
  bumps once.
- **Ledger** — sections *The Look* (preset cards painted as tiny sky-and-sea
  vignettes, quality plaques), *Weather & Water* (wind with knots, swell
  length, cloud cover, sea fog, time of day), *Sound & Screen* (volume,
  music, sharpness), *Shipwright's tools* folded in a `<details>` because the
  debug toggles are the author's, not the visitor's, and *About the tech*,
  folded the same way: the jargon kept out of every label, a sentence each.
  Every slider and the quality field carry a plain-language tooltip. Presets and quality are native radios (arrow keys for
  free). Sliders are brass rails: an engraved groove filled with brass to the
  reading and a round brass knob. Toggles are an inked box ticked in wax.
- **Start gate** — a chart sheet with rhumb lines and a rose watermark, the
  title in Pirata One with "3D" in wax, brass plaques for renderer and quality
  with a wax "advised" tag from the device probe, a plain tooltip on every
  plaque, and the Set Sail button.
  The sheet lies on the **menu sea** (below). Loading folds the choices away
  and sails an ink ship from a lighthouse to an island along a dotted course,
  inking the course in wax behind it, with a wet-ink glint running along the
  line; the ship rides `--boot-progress` (never backwards), the bar is
  `transform: scaleX()`. Under the stage line a second line rotates a tip or
  a true piece of the world (`LoadingTips.ts`), each held 5.2s from when it
  went up, faded unless motion is reduced. States: *WebGPU unavailable* — the
  WebGPU plaque is disabled and captioned "not in this browser", WebGL2 is
  chosen, and a margin note with the hazard icon says why and where to get the
  full sea; *failed* — the course greys where it stopped, the status turns wax
  red and says what to try, the error's own words sit under it in small ink,
  and a brass "Try again" reloads.
- **Menu sea** (`MenuSea.ts`) — the gate's own living backdrop, painted on a
  2D canvas (never WebGL, so it costs the GPU nothing the real sea needs):
  a dusk sky, painted flat-bottomed clouds lit from below, stars, gulls, a sun
  with a squashed-ellipse road of glints on the water, five parallax swell
  bands with warm rims and running dashed foam, a headland whose lighthouse
  beam sweeps, and a square-rigger riding the third band with a lantern and
  bow spray. Scenery is pre-rendered once per size; per frame it is a few
  fills and sprites. DPR is capped at 1.5 and the backing store at 2.4 MP,
  ≤ 4-core devices draw at 30 fps, it pauses while the tab is hidden, and
  reduced motion gets a single still frame. Composition moves per shape:
  ship and sun right of the sheet on desktop, above the bottom card on a
  portrait phone, right of the compact sheet in landscape. When the game is
  ready the whole boot layer cross-fades (900ms opacity) into the live sea —
  never through black — and the canvas is released.

## Attention

The HUD steps back when it is not being used, the way console games treat
their HUD: nothing is hidden for good, but nothing sits at full strength over
the view unless it has something to say. `Hud.ts` sets three things on
`#ui-root` and CSS does the rest.

- **`is-idle`** — after 4.5s without input (pointer, key, wheel) the
  instruments fade over 1.4s: compass to 34%, wind and log 38%, mode bar 26%,
  instrument corner 45%, score 60%, folded ledger / Ledger button / touch
  controls 50%. The objective and any toast stay: they are being read. Any
  input brings everything back in 220ms. So does news — a toast, a new
  objective, a changed score, a mode change, the wind moving more than 2kn or
  the ship's speed more than 1.5kn. A keyboard focus ring or a live toast
  holds the HUD up; a real hover (not a stuck touch hover) restores the
  element under the pointer.
- **`is-dragging`** — once a press on the canvas travels 6px the viewer is
  looking at the sea: instruments drop to 18%, quest and ledger plate to 35%,
  an open ledger to 60%, until release.
- **`data-mode`** — the tour (`cinematic`) is a film: compass, instruments,
  quest and corner go entirely, and the mode bar goes too once idle.

Key hints are contextual: they appear with the first input after load and on
every mode change, for seven seconds, then hide until the mode bar is hovered
or focused. A tab changed from the keyboard pulses once (`is-arriving`).

**First-minute lessons** (`Coach.ts`) use the same ribbon. Until a lesson is
done the ribbon holds one line — look around, take the helm, set sail, steer,
find the cargo, drop anchor, in that order, each in the modes where it makes
sense — and stays up instead of timing out. Idle dims it to 62%, a camera
drag to 22%. Doing the thing is what retires it (a drag, the mode, the key,
the ship's own speed or swing, a score line), and what has been learned is
kept in localStorage. On a phone in portrait the ribbon is otherwise hidden;
a lesson shows in the left column above the wind plate, 186px at most, so it
clears the sail orders and the stick. Landscape phones show none.

## Motion

Exponential ease-out (`cubic-bezier(.16,1,.3,1)`) everywhere. The authored
moments are the toast unroll, the objective unfurl, the score bump, the
tab arrival, the chart laid on the table, the boot ship's bob, the ink
running on the course, and the menu sea. `prefers-reduced-motion` collapses every transition and
animation globally, stops the ship bobbing, and the Hud skips the unfurl and
bump classes entirely.

## Do's and Don'ts

**Do**

- Put reading on parchment and readings on wood. A new surface picks its
  material by what the visitor does with it.
- Spend wax on "this one" and gold on reward, nowhere else.
- Set every figure in Alegreya with tabular lining numerals and a `ch` floor.
- Give every new control a hard bottom edge and a press that removes it.

**Don't**

- Add `backdrop-filter`, or a glow (a zero-offset coloured halo).
- Put a straight edge or a border-radius on parchment.
- Set Pirata One on a figure or a compass point, or IM Fell English anywhere but
  the storm quote.
- Use `ink-faint` as a text colour.
- Animate `width`, `height`, `padding` or `margin`. The boot course is
  `scaleX()`, the ship and the compass tape are `translate`.
- Borrow any game's name, logo, fonts, icons or art.
