# Budgets

What a frame, a start-up and a download cost today, per quality tier, and what
each system planned in Milestones 1 and 2 of `docs/AAA_ROADMAP.md` is allowed to
add. Written 2026-11-17 against the roadmap's `main` at 2ec1e00.

Every figure comes from one machine (a Ryzen 7 9800X3D and an RTX 5090, Windows
11, Chrome; see `docs/PERFORMANCE.md`, Recorded hardware). Nothing here says
where the tiers stop working on other hardware. A number marked **unmeasured**
has no measurement in the repository; none was estimated to fill the gap.

## Today, per tier

Three different measurements are mixed here, and they are not interchangeable.
Read the notes under the table before comparing rows.

| Tier | GPU frame p50 (bench) | Live frame rate | Start-up (bench, first frame) | Ready to play (live) | Download |
|---|---|---|---|---|---|
| Low | 0.80 ms | unmeasured | 7.4 s | unmeasured | unmeasured |
| Medium | 2.41 ms | unmeasured | 7.5 s | unmeasured | unmeasured |
| **High** | **3.87 ms** | **113 to 119 fps** on a 120 Hz display | 7.8 s | **14 to 19 s** | **32.4 MB** |
| Ultra | 5.70 ms | unmeasured | 7.5 s | unmeasured | unmeasured |
| Max | 6.08 ms | unmeasured | 7.9 s | unmeasured | unmeasured |

Where each column comes from:

- **GPU frame p50 (bench).** `node scripts/benchmark.mjs`, the 2026-11-18 run
  in `docs/PERFORMANCE.md` ("fewer draws and fewer passes"). Measured at
  1600 x 900, DPR 1, on three.js **r187**, with GPU timestamps and the frame
  pacer off, after the optimisation pass of that day; before it the same
  machine read Low 2.74, Medium 2.70, High 4.64, Ultra 6.43 and Max 6.94 ms.
  The gate is High under 16.7 ms. The r185 figures this table used to carry
  (High 7.18 ms) and the roadmap's "about 8.5 ms" are superseded. The frame
  interval with the loop running free, which counts the page's work as well
  as the GPU's, is 3.8 ms at Low, 4.9 at Medium, 4.9 at High, 6.6 at Ultra and
  7.3 at Max.
- **Live frame rate.** `README.md` (Performance): High, 1920 x 1080, 120 Hz
  display, r187, production build, 113 to 119 fps. `docs/REFERENCE_TARGETS.md`
  (Scoreboard) records 111 to 113 fps with a p99 frame of 17.1 to 18.4 ms for
  the same configuration on the same day. The two differ by a few frames; the
  README figure is the later one. No other tier has a live figure.
- **Start-up (bench, first frame).** Time to first frame (TTFF) in the same
  2026-11-18 bench run, r187. It is not time to playable.
- **Ready to play (live).** `README.md`: High, r187, 14 to 19 s from clicking
  Set sail to ready. `docs/REFERENCE_TARGETS.md` records 17.3 to 19.7 s for the
  same build on 2026-11-17. The bench TTFF above is far shorter because it
  measures the first frame, not readiness, and on r185.
- **Download.** 32.4 MB at High, cold cache, r187 (`docs/REFERENCE_TARGETS.md`,
  Scoreboard; the roadmap's Guardrails repeat it). Whether other tiers fetch
  fewer or different assets has not been measured. Earlier figures of 36.9 MB
  (r185) are superseded.

Memory is not a budget in the roadmap. For reference, the 2026-11-15 bench
peaks at 208 MiB (Low), 264 (Medium), 403 (High), 410 (Ultra) and 511 MiB (Max).

## Totals the roadmap fixes

- **Frame time.** Milestone 1 may add at most **1.5 ms** in total at High. A task
  that cannot fit moves to Ultra and Max and gets a cheaper form below that.
  Decision 5 asks that High hold 100 fps or better on the development machine.
- **Start-up.** Ready to play is 14 to 19 s today. New compute pipelines go
  through the existing prewarm, and each milestone re-measures. No cap is set
  yet: **to be set when measured**.
- **Download.** 32.4 MB today. The roadmap says the cap is set in P0.3 and gives
  no figure, so none is written here: **to be set when measured**, after a model
  or a second ship has a size.
- **Audio.** No system in the audio lane may raise main-thread time. Beds at
  zero level are disconnected.
- **Flashing.** Three flashes a second at most, with a reduce-flashes option.

Of the 1.5 ms, the two caps the roadmap already writes for Milestone 1 (sails,
0.5 ms; spindrift, 0.5 ms) take 1.0 ms. That leaves 0.5 ms for every other
Milestone 1 system at High. Only measurement will say whether they fit.

## Per new system

"Cap" is the most a system may add to the frame at the tiers it runs on, in
milliseconds, taken from the roadmap. Where the roadmap gives no cap the cell
says **to be set when measured**. Costs the roadmap quotes from other projects
are noted as such; they are not caps for this build. "Tiers" and "Fallback" are
filled only where the roadmap names them. Otherwise they are also **to be set
when measured**, and the implementing task adds its tier-table entry and
override at that point, as the roadmap's Tiers guardrail requires.

### Milestone 1

| System | Cap | Tiers it runs on | Cheaper fallback |
|---|---|---|---|
| S1 Sails that set, fill and luff | 0.5 ms | Compute verlet cloth on the higher tiers; the cut-off tier is to be set when measured | Low and Medium: an analytic billow in the vertex stage |
| S2 Rigging that sags and tightens | to be set when measured | to be set when measured | CPU catenary with a vertex-stage bend, no physics (the method itself is the cheap form) |
| S3 Moving parts (wheel, rudder, yards, pennant, anchor cable) | to be set when measured | to be set when measured | to be set when measured |
| S4 Light aboard (stern windows, lanterns) | to be set when measured | to be set when measured | Fewer lanterns (the roadmap gives three to five) |
| S5 Wet hull and deck | to be set when measured | to be set when measured | to be set when measured |
| S6 The hull below the waterline | to be set when measured | to be set when measured | to be set when measured |
| W1 Light through crests, bubbles under foam | Near zero (the roadmap's word; no figure) | All | None needed |
| W2 Spindrift off the whitecaps | 0.5 ms for 50,000 to 100,000 particles | High and up | Off below High |
| W3 A stable glitter path | to be set when measured | to be set when measured | to be set when measured |
| W5 Fine detail and gusts | to be set when measured | to be set when measured | to be set when measured |
| W6 The horizon (distance fade, haze) | to be set when measured | to be set when measured | to be set when measured |
| F1 Speed you can feel (field of view, lag) | to be set when measured | All; off under reduced motion | Off |
| F2 Apparent wind and trim | to be set when measured (CPU) | All | None; it replaces the hidden engine |
| F3 Gamepad and rumble | to be set when measured (CPU) | All; rumble only where `playEffect` exists | No rumble |
| F6 One shake budget and hit-stop | to be set when measured (CPU) | All | Intensity slider to 0 |
| F8 A pickup worth making | to be set when measured | to be set when measured | to be set when measured |
| F9 Failure you can see (sand plume, camera kick) | to be set when measured | to be set when measured | to be set when measured |
| F10 One set of timings | None (interface and camera timing only) | All | None |
| X1 Spray that reaches the lens | to be set when measured | to be set when measured | to be set when measured |
| X2 Drops that land | to be set when measured | to be set when measured | to be set when measured |
| X3 Bubbles under the hull and wake | to be set when measured | to be set when measured | to be set when measured |
| X4 Soft edges everywhere (depth fade on smoke, burst, rain) | to be set when measured | to be set when measured | to be set when measured |
| A1 Buses, memory and lifecycle | 0 ms of main-thread time added | All | None |
| A2 A room around every sound | 0 ms of main-thread time added | All; reverb has tiers today (music) | A shorter impulse or fewer settings, to be set when measured |
| A3 Loudness and priority | 0 ms of main-thread time added | All | Reduced-dynamic-range option is a player setting, not a fallback |
| A4 Sound in space | 0 ms of main-thread time added | All | Plain panning for all but a few hero sources, which is the design |
| A5 No audible loop | 0 ms of main-thread time added | All | to be set when measured |
| A6 Thunder that differs | 0 ms of main-thread time added | All | Reserved voice so a distant strike cannot starve the six Low slots |
| A7 Wind you feel | 0 ms of main-thread time added | All | to be set when measured |

### Milestone 2

| System | Cap | Tiers it runs on | Cheaper fallback |
|---|---|---|---|
| I1 Remove the one-island assumption | None (data change) | All | None |
| I2 Island silhouette | to be set when measured | to be set when measured | to be set when measured |
| I3 Depth in the canopy | to be set when measured | to be set when measured | to be set when measured |
| I4 One wind field | to be set when measured | to be set when measured | to be set when measured |
| I5 An archipelago | to be set when measured | to be set when measured | to be set when measured |
| I6 A harbour with lights | to be set when measured | to be set when measured | to be set when measured |
| I7 The reef at the surface's standard | to be set when measured | to be set when measured | to be set when measured |
| K1 Clouds with variety | to be set when measured | to be set when measured | to be set when measured |
| K2 Weather that moves | to be set when measured | to be set when measured | to be set when measured |
| K3 A day in play | to be set when measured | to be set when measured | to be set when measured |
| K4 The night sky | to be set when measured | to be set when measured | to be set when measured |
| L1 Bounce light | to be set when measured for the probe grid. Screen-space global illumination: 1.5 to 3 ms, the roadmap's estimate | Screen-space GI on Ultra and Max only. Probe grid: to be set when measured | Probe grid below Ultra |
| W4 Waves that answer the hull | to be set when measured (Atlas measured 0.6 to 0.8 ms for its own; not a cap here) | to be set when measured | to be set when measured |
| S7 A crew (six to ten instanced figures) | to be set when measured | to be set when measured | Fewer figures; to be set when measured |
| X5 Fire that burns | to be set when measured | to be set when measured | to be set when measured |
| X6 Rain you sail through | to be set when measured | to be set when measured | to be set when measured |
| X7 Lightning that lights the sky | to be set when measured | to be set when measured | to be set when measured |
| X8 Spray at the shore | to be set when measured | to be set when measured | to be set when measured |
| X9 Smoke lit from six directions | to be set when measured | to be set when measured | Drawn at half resolution to bound overdraw |
| A8 The island heard | 0 ms of main-thread time added | All | to be set when measured |
| A9 Rain on wood and canvas | 0 ms of main-thread time added | All | to be set when measured |
| A10 An audio worklet | 0 ms of main-thread time added; no allocation inside its process call | All | to be set when measured |

## Systems that have no tier gating today

The roadmap's Tiers guardrail names four: bow spray, splashes, rain and the
pickup effects. Each gets a tier entry when it is next changed (P0.5, X1 to X4,
X6, F8). Today the bow spray is 1,792 particles plus a mesh sheet on every tier.

## How a cap gets set and checked

1. The task measures its system on the development machine with
   `scripts/profile-frame.mjs --tier <tier>`, before and after, and writes the
   difference here in place of "to be set when measured".
2. The roadmap also asks for rows for each new system in
   `scripts/profile-frame.mjs`. That script is not edited by this document.
3. After each milestone the High, Ultra and Max GPU p50, the live frame rate,
   ready-to-play time and download size are re-measured and the first table is
   updated, with its date, so a drift shows up as a changed row. The method
   for the live figures is in `docs/PERFORMANCE.md`, Results, 2026-11-17.
