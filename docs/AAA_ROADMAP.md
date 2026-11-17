# AAA roadmap: from the best web ocean to a game that stands beside shipped ones

Written 2026-11-17, against `main` at 2ec1e00. This file is the plan and task
list for the next pass. `docs/AAA_PLAN.md` stays as the record of the passes
before it.

## The bar has moved

The last pass was judged against open-source web demos, and won 30 of 32 blind
pairs (`AAA_PLAN.md`, section Q). That bar is met. This pass is judged against
shipped sailing games: Sea of Thieves, Assassin's Creed IV Black Flag and its
remake, Skull and Bones, Dredge and Sailwind. We take their qualities and
design principles. We do not take names, characters, missions, art, music or
interface from any of them.

Against that bar the build has three gaps, in this order of how quickly a new
player notices them:

1. **The ship is a statue.** Sails never set or fill, nothing on deck moves,
   nobody is aboard, and at night she is a black cut-out.
2. **There is five minutes of game.** One repeating task, no risk, no ending,
   nothing to spend gold on, nothing else afloat.
3. **The world is one green dome under repeating clouds.** The sea near the
   camera is the strongest thing in the frame; the island, the sky and the far
   sea do not hold up beside it.

Effects, sound and feel follow the same pattern as the picture: they are
strong at the water and thin everywhere else. Spray, splashes and the
underwater transition are well built. Fire is a static cone, rain is a flat
additive sprite, the island makes no sound at all, no sound has a room around
it, and the whole score is 40 bars of melody.

## What this is based on

- **The gallery.** All 15 images in `docs/images/`, recaptured today, and the
  nine in `docs/progress/`.
- **The code.** A read-only audit of gameplay, ship handling, cameras, the
  world's contents and the interface, with file and line pointers. Its findings
  are quoted in the tables below.
- **Effects, sound and feel.** Two further read-only audits: one of every
  particle, fire, weather and feedback system, and one of the audio engine,
  the sound effects and the score. Stills from `tests/baselines/` were checked
  for the effects that show in them.
- **Published sources.** Rare's SIGGRAPH 2018 paper on Sea of Thieves, the
  Atlas water talk (GDC 2019), Guerrilla's Horizon Forbidden West water notes
  (SIGGRAPH 2022), Crest's documentation, developer interviews on Black Flag,
  Dredge, Sailwind and Sea of Thieves, and the three.js r187 example index.
  Links are in the last section.

Not done, so the plan does not lean on it:

- **No motion review.** The repository holds no video, and the in-app browser
  drew 0 frames a second when I tried a live look. Everything said here about
  the picture comes from stills; everything about handling and camera comes
  from the code. Task V2 adds a way to review motion.
- **No reference video was watched.** The reference material is papers, slides
  and interviews.
- **Nothing was listened to.** Every statement about sound comes from reading
  the synthesis code, not from hearing it. Task V5 adds an offline render so
  loudness, clicks and repetition can be measured, and the tunes still need
  your ear.

## Visual audit: what the gallery shows

| Area | What the stills show | Where | What a shipped game does |
|---|---|---|---|
| Ship | Topsails permanently furled, courses hang flat; no flag, no crew, no lit windows. In dark looks she is a pure black silhouette with one lantern | `boat`, `night`, `squall`, `moonlit` | Canvas fills and luffs, rigging sags and tightens, lanterns and stern windows glow, figures move on deck |
| Ship from below | The hull underside is a smooth dark blob with no planking, keel or rudder | `underwater` | The hull is as detailed below the waterline as above |
| Island | One low dome. The canopy is a uniform green texture of same-sized crowns with no shading between them. A flat lawn band, then a thin flat beach. Props read as specks | `hero`, `island`, `landfall` | A sculpted silhouette you can name from a kilometre off: cliffs, a headland, a cove, emergent trees, a ragged vegetation edge |
| Clouds | Same-sized puffs with hard cut-out edges. They repeat in rows toward the horizon and tile visibly at night. No tall forms, no weather front | `waves`, `night`, `hero` | Variety in size and height, soft wispy edges, towers and anvils, a squall you can see coming |
| Open sea, mid and far | Smooth and gel-like with little fine detail, sparse blob foam, and a hard dark horizon line. A smeared band behind the ship | `waves`, `hero`, `boat` | Detail at every scale, wind streaks, gust patches, a horizon that melts into haze |
| Night sea | The moon's path breaks into hard white blocks. Stars are square specks | `moonlit` | A stable, narrow glitter column and point stars |
| Storm | The mood is right. No spindrift off the crests, no rain shafts, crests are not translucent green | `storm`, `squall` | Wind-torn spray, rain curtains under the cells, light through the crests |
| Shore | The breaker reads as a smear with one straight ruled ridge. A dark wedge sits on the water by the beached pinnace | `landfall` | A readable lip, bore and lace |
| Underwater | The reef is low-poly and stylised beside a realistic surface. Fish are black silhouettes | `reef` | One art direction above and below |
| Interface image | Shows debug probe dots on the hull and no gameplay | `interface` | The front-page image shows the game being played |

What already holds up: the near sea, shallow water and caustics (`shore`), the
sunset (`sunset`), the interface's design language, and the frame rate
(113 to 119 fps at High on a 120 Hz display).

## Game audit: what the code does today

| Area | Today | Pointer |
|---|---|---|
| Loop | One objective: sail into a drifting cask, gain gold, repeat forever. No fail state, no ending, gold buys nothing. The cask sequence is the same every run. About five minutes of content | `gameplay/Gameplay.ts:944`, `:769-825` |
| Saving | Quality choice and learned lessons only. Score is never saved | `ui/StartGate.ts:6`, `ui/Coach.ts:83` |
| Handling | Good: sail polar with a no-go zone, keel drag, a rudder that needs way on, heel as real torque, an anchor that swings her to the wind. Faked: no apparent wind, no trim, wind strength stops mattering above 9 m/s | `gameplay/sailing.ts`, `physics/ShipController.ts` |
| Hazards | Shallows push the ship away. No damage, no collision with rocks or props | `ShipController.ts:359-398` |
| Input | Keyboard and touch. No gamepad. **Bug:** the touch stick still drives the hidden engine at the helm | `ShipController.ts:222, 289`, `main.ts:2693` |
| World | One island, 500 m mean radius, 1.4 km from the start. A cove, jetty, beached pinnace, camp, fort, lighthouse, strongbox, skull rock, arch, skeleton. None can be interacted with. No other ships, crew or enemies | `scene/Seafloor.ts:282-291`, `scene/Props.ts` |
| One-island assumption | The `ISLAND` constant is read in 16 files, 71 times. The seabed is one analytic function with a hand-mirrored shader copy | `Seafloor.ts:687-711, 765` |
| Ship model | One Poly Haven model, no detail levels. Its only animation is a 1.2% sail scale. The sail setting reaches the audio and nothing visible | `scene/Ship.ts:178-200`, `main.ts:2601` |
| Camera | A chase rig with lag and mouse orbit. Field of view never changes with speed. No deck view, no photo mode | `cameras/CameraDirector.ts:679-683, :445` |
| Weather in play | Fixed. The day cycle and the squall exist only in the Tour | `main.ts:2964-2975, 3359` |
| Shell | A quality picker, then the sea. No pause, no key rebinding, no accessibility options | `ui/StartGate.ts` |

## Effects audit: what the code draws

| Area | Today | Pointer | What a shipped game does |
|---|---|---|---|
| Hull spray | Well built: 40 impact events of 64 particles, sun-lit with forward scatter, soft against depth, carrying the hull's velocity and the wind. Drops leave nothing where they land, and never reach the lens | `physics/Spray.ts`, `physics/sprayLook.ts` | Drops leave foam and rings, water streams off the hull as she rises, spray hits the camera |
| Bow spray | 1,792 particles and a mesh sheet. Not gated by quality tier. Hidden whenever the player is not at the helm, so the Tour shows a dry bow. Reads last frame's hull pose | `gameplay/effects.ts:331`, `Gameplay.ts:417`, `BowSheet.ts:333` | Always there, in step with the hull |
| Splashes | Crown, jet and mist with foam flecks and a ring wave. No depth pre-pass, so fog treats it as background in a storm | `physics/Splash.ts` | Fogged with the scene |
| Rain | 5,000 additive sprites of one constant colour, within 34 m of the camera. The volume is carried with the camera, so sailing fast does not make rain stream past. Not blocked by sails or deck, no splashes on deck, no distant sheets | `sky/Weather.ts:134, 174` | Lit by the scene and the lightning, streaming with speed, sheltered under canvas, splashing on the planks |
| Lightning | A branching ribbon whose forks are as bright as the main channel. No strike closer than 900 m. The flash lights the ship and sea, not the clouds, spray or rain. Flashes repeat about every 7 s with no reduced-flash option | `sky/Lightning.ts:30`, `main.ts:3626` | Clouds lit from inside, tapered forks, a close strike now and then |
| Fire | Camp fire and torches are static six-sided cones with a flickering emissive colour. No flame sheets, embers, sparks or heat haze. Lights flicker in brightness but never move, cast no shadows, and the two torches share one light | `scene/islandDetail/parts.ts:271`, `lighting/LocalLights.ts:145, 231` | Animated flame, embers on the wind, shadows that dance |
| Smoke | 34 billboards that drift with the wind. Hard edges where they meet the ground; rain does not affect them | `parts.ts:293` | Soft against geometry, lit from several directions |
| Shore | No spray from waves breaking, no haze over a rough sea | — | Spray off the lip and the rocks |
| Moments | None of these exist: canvas snapping, bubbles under the wake, a sand plume on grounding, treasure glints, fish or dolphin splashes, bird take-off puffs, a glowing wake at night | — | These are what make a sea feel alive |

## Sound and music audit: what the code plays

All sound is synthesised. There are no sample files.

| Area | Today | Pointer | What a shipped game does |
|---|---|---|---|
| Mix | Beds and ship into a compressor, then a limiter and a soft clipper. No separate effects, ambience or interface buses. Loudness figures exist only in a header comment; nothing reproduces them and there is no target | `audio/AudioSystem.ts:2107-2346, 70-78` | Buses per kind, a loudness target, priority ducking |
| Space | Stereo pan and a 1/distance gain for the ship and the gulls. No listener, no positional sources, no occlusion. Thunder has no bearing | `main.ts:2584-2591` | Creaks placed on the hull, thunder from where the bolt fell |
| Room | Reverb on the music only. Every sound effect and ambience is completely dry | `audio/music/instruments.ts:575` | A shared room for the deck, another below, another under water |
| Sea, wind, rain | Filtered-noise beds that breathe with the swell and gust. All of them loop the same 4 s of noise, which the narrow rigging band will expose | `audio/NoiseBed.ts`, `noise.ts:26` | No audible repeat |
| The island | Silent. No shore surf although breakers are drawn, no fire crackle, no insects at night, and the audio has no time-of-day input | `main.ts:3718, 3553` | The island is heard before it is reached |
| Thunder | Delayed by distance, which is right. The same two envelope curves on every strike | `AudioSystem.ts:1637, 2116` | No two alike |
| Ship | Good: creaks from roll and pitch rate, block knocks, sail flaps, wave slaps, hull impacts, an anchor chain. Barrel and buoy splashes make no sound | `audio/ShipSfx.ts`, `main.ts:1714-1726` | The same, placed in space |
| Score | Two tunes, both in D minor, 40 bars in all. Eight stems layered by ship speed. The order loops after 46 to 107 s depending on state. One sting. A fixed D drone under every section, including the F major one. A silent state exists and is never chosen | `audio/music/score.ts`, `MusicEngine.ts:154-259` | Hours of material, a sting for every event, and silence used on purpose |
| Instruments | Fiddle from two detuned saws with no legato; reeds from a wavetable; plucked strings; membrane drums; a five-oscillator drone. Drum timing varies by 0 to 4 ms late only; no swing | `audio/music/instruments.ts` | Players, not oscillators |
| Controls | Master and Music sliders. No effects slider, no mute, and nothing is saved: volume resets to 0.4 every visit | `ui/Panel.ts:133-150`, `ui/types.ts:128` | Separate volumes, remembered |
| Lifecycle | Nothing handles a hidden tab. Music is kept alive by a timer that browsers throttle in the background. If the first unlock fails, audio stays silent for good | `AudioSystem.ts:1197, 1221-1224` | Pauses and resumes cleanly |
| Tests | None | — | — |

## Feel audit: what answers the player

| Event | Today | Pointer |
|---|---|---|
| Hard impact | A good trauma-based shake with a cooldown. No hit-stop, no camera kick | `Gameplay.ts:147` |
| Picking up cargo | Burst, splash, chime, toast and a score bump. The cask vanishes and the next appears in the same frame; the gold figure snaps to its new value | `Gameplay.ts:786-827` |
| Grounding | Shake, a thud and a toast. Nothing to see | `Gameplay.ts:862-877` |
| Stalling head to wind | A toast and the luff sound | — |
| Changing mode | An 0.85 s camera ease. No fade or letterbox | `CameraDirector.ts:195` |
| Surfacing | Strong: a per-pixel waterline and a film that drains off the lens | `underwater/` |
| Interface timing | One pair of easing curves, but 23 different durations from 100 ms to 1.6 s, and the camera uses its own | `ui/styles.css:123` |
| Reduced motion | Honoured by the interface, the shake and the Tour. Ignored by the mode-change sweep, the beacon pulse and the lightning | `CameraDirector.ts:347-351`, `effects.ts:113-120` |

## Decisions

Decided 2026-11-17: the recommendation is taken on all eight, and on any
later decision that carries one, except where an entry below says otherwise.
Standing rule for every task: if a good open-source or free project, asset or
package already does the job, reuse it with credit instead of rebuilding it,
and download what the work needs without asking. Changing one moves tasks between milestones;
it does not invalidate the rest.

1. **Is this now a game first?** `PRODUCT.md` still defines it as a showcase
   whose visitor stays "fifteen seconds to a few minutes". Recommendation: yes,
   rewrite the brief as a short-session game (5 to 20 minutes), with the render
   still leading.
2. **Combat.** Broadside combat is the largest single item here and needs a
   second ship, damage, effects and enemy behaviour. Recommendation: build a
   rival sail without combat in Milestone 3, and treat combat as Milestone 4.
3. **Going ashore.** A walking character is a second game's worth of work.
   Recommendation: no. The player anchors and interacts from the ship, and the
   camera goes where the player cannot.
4. **Art direction below the surface.** Recommendation: bring the reef up to
   the surface's realism, not the surface down to the reef.
5. **Frame-rate floor.** Recommendation: High holds 100 fps or better on the
   development machine. Anything that cannot fit goes to Ultra and Max.
6. **New models.** Crew figures, cannons and extra buildings need assets.
   Decided: where a good free or open-licensed model exists, use it instead of
   building one, and record its licence in `ASSET_LICENSES.md`. Build
   procedurally only where nothing suitable exists.
7. **Recorded sound.** Everything is synthesised today. Published listening
   tests found the best synthesised thunder still told apart from a recording,
   and the audio lead on No Man's Sky has said procedural wind and rain work
   but lack feeling. Decided: use CC0 recordings wherever they beat synthesis,
   starting with the thunder crack, the gulls and the surf, layered with the
   synthesised beds so they still follow the sea state. The Sonniss bundle
   cannot go in a public repository under its licence.
8. **Period of the instruments.** The score uses concertina, hurdy-gurdy and
   bodhrán. Fiddle, fife, drum and trumpet are documented aboard ships of the
   time; the concertina and the work shanty are from the 1800s.
   Recommendation: keep them as stylistic licence, as the genre does, and say
   so in `docs/PERIOD_AUDIT.md`.

## Milestones

Sizes are relative effort: S is a short focused change, M a system, L a system
with content, XL several systems. Each task says what done looks like.

### Milestone 0: groundwork

- [x] **P0.1 Rewrite the brief** (S). `PRODUCT.md` describes a short-session
  game; `README.md` follows. Done when the two no longer contradict this file.
- [x] **P0.2 Extend the reference board** (S). Add rows to
  `docs/REFERENCE_TARGETS.md` for the ship under sail, the ship at night, the
  island from sea, and a squall front. View references; do not download them.
- [x] **P0.3 Set budgets** (S). A table per tier of frame time, start-up time
  and download size, with a cap for every new system below. Add rows to
  `scripts/profile-frame.mjs`.
- [ ] **P0.4 Fix what the audit found** (S each):
  - the touch throttle driving the engine at the helm;
  - the stale whitecap expectation in `tests/foam.spec.ts` (13.4% rendered
    against 3.9% expected at 15 m/s);
  - the dark wedge by the beached pinnace in `landfall`;
  - the straight ridge in the breaker;
  - the unused `'danger'` toast.
- [ ] **P0.5 Fix the effect defects** (S each):
  - the wash impact reported about 8 m off the starboard side, so its foam
    and its sound are in the wrong place (`Spray.ts:627`);
  - bow sheet and bow spray reading last frame's hull pose (`main.ts:4042`);
  - no bow spray outside helm mode, including the Tour (`Gameplay.ts:417`);
  - the splash missing its depth pre-pass;
  - rain streak tilt taken from world X and applied as a screen rotation
    (`Weather.ts:195, 236`), and lens drops leaning the same way whichever
    way the camera faces (`main.ts:3914`);
  - the objective arrow spinning the long way round when the bearing crosses
    180 degrees (`Hud.ts:1010`);
  - the beacon ring snapping back to full opacity every 2.2 s.
- [ ] **P0.6 Fix the audio defects** (S each):
  - spray impacts attenuated and panned twice (`main.ts:1736-1742` and the
    ship bus);
  - wave slaps probably firing twice, from the slap detector and from spray
    events; confirm first;
  - the unlock handler removing itself even when resume fails;
  - a distant thunder strike holding a voice slot for its whole delay, up to
    37 s, which starves the six slots on Low;
  - the hurdy-gurdy gate jumping to zero on every quaver, and a reverb tier
    change cutting the tail, both of which click;
  - wave height guessed from wavelength instead of the spectrum's own figure
    (`main.ts:3869`);
  - barrel and buoy splashes making no sound.
- [x] **V2 Motion clips** (S). A script that saves ten-second frame strips of
  four situations: helm in a breeze, a hard turn, the storm, and landfall. The
  texture-scroll foam fault was invisible in stills; this is how the next one
  is caught.
- [ ] **V5 An audio test bench** (M). The engine already accepts an offline
  context and nothing uses it. Render 60 s of each state offline and report
  integrated loudness, true peak, clicks, voices alive and the time before
  material repeats. Save the renders as files so they can be listened to.
  Every audio task below is judged with it.

### Milestone 1: a living ship on a sea that glows

The largest visible and audible jump for the least work. Five lanes that touch
different files, so they can run side by side.

**Ship lane** (`scene/Ship.ts`, new `scene/ship/`)

- [ ] **S1 Sails that set, fill and luff** (L). Replace the 1.2% scale with
  cloth pinned at the yards and clews, driven by apparent wind, and tie the
  furled, half and full settings to what is drawn. Start from three.js's
  `webgpu_compute_cloth` (verlet springs in a compute pass, MIT). Low and
  Medium get an analytic billow in the vertex stage. The sail meshes need a
  regular grid, so they are rebuilt from the yard positions. Budget 0.5 ms.
  Done when a close-hauled sail is visibly flatter than one running, and a
  ship head to wind shakes her canvas.
- [ ] **S2 Rigging that sags and tightens** (M). Sea of Thieves' method: solve
  the catenary on the CPU per rope with a slack value, and bend a tube in the
  vertex stage. No physics. Done when sheets and braces move with the sails.
- [ ] **S3 Moving parts** (M). The wheel turns with the helm, the rudder
  swings, the yards brace toward the wind, a pennant streams from the
  masthead, and the anchor cable is visible when it is down. The pennant is
  also the player's wind indicator.
- [ ] **S4 Light aboard** (S to M). Glowing stern windows and three to five
  swinging deck lanterns through `ClusteredLightsNode`, lighting the sails and
  reflecting in the sea. Done when the ship in `night` and `squall` reads as a
  lit vessel and not a cut-out.
- [ ] **S5 Wet hull and deck** (S to M). `scene/Wetness.ts` already darkens and
  glosses surfaces under rain. Add a waterline band that follows the sea and a
  spray-driven wet mask that dries over seconds.
- [ ] **S6 The hull below the waterline** (M). Planking, keel, rudder and
  period-correct bottom treatment, so `underwater` shows a ship.

**Water lane** (`ocean/`, `physics/Spray.ts`)

- [ ] **W1 Light through the crests and bubbles under the foam** (S). Blend
  deep and scatter colour by view angle, sun direction and a wave-peak mask
  (Sea of Thieves), and add a bubble colour scaled by foam density (Atlas).
  Done when storm crests are translucent green toward the light. Near zero
  cost.
- [ ] **W2 Spindrift off the whitecaps** (M). Atlas's method: emit particles
  across the view and kill any that are not on a whitecap in the foam
  texture. Compute particles, soft against depth, lit by the scene. Budget
  0.5 ms for 50,000 to 100,000 particles; High and up.
- [ ] **W3 A stable glitter path** (M). Slope-variance filtering: store first
  and second slope moments per wave band and mip them, with an area highlight
  for a low sun. `AAA_PLAN.md` section C deferred this for want of the second
  moments; this task adds them. Done when the moon's path in `moonlit` is a
  smooth column with no blocks.
- [ ] **W5 Fine detail and gusts** (S to M). A low-frequency gust mask scaling
  the smallest wave band (dark "cat's paws" crossing the water) and finer
  ripples near the camera. Done when `waves` no longer reads as gel.
- [ ] **W6 The horizon** (S to M). Fade each band's displacement with distance
  (Atlas uses about 30 wave periods) and dissolve the sea into haze by day and
  spray haze in a storm. This closes an item open since the first pass.

**Feel lane** (`cameras/`, `gameplay/sailing.ts`, `physics/ShipController.ts`, input)

- [ ] **F1 Speed you can feel** (S to M). Field of view widens with speed, the
  camera lags and looks into the turn, and bow spray and wind noise scale with
  it. All of it off under reduced motion.
- [ ] **F2 Apparent wind and trim** (M). Wind that is potent, not punishing: a
  floor speed on every heading, a trim control that rewards skill, and a no-go
  indicator that fades once learned. Remove the hidden engine. Turn response
  is exaggerated at speed.
- [ ] **F3 Gamepad and rumble** (S to M). Standard mapping, with rumble
  feature-detected: `playEffect` exists in Chrome, Edge and Safari 16.4 and
  later, and not in Firefox or iOS Safari.
- [ ] **F6 One shake budget and hit-stop** (S). Grounding, slams and thunder
  draw on the same decaying trauma value; a few frames of hit-stop on a hard
  impact, and an intensity slider from 0 to 100%.
- [ ] **F8 A pickup worth making** (S). The cask is hauled aboard instead of
  vanishing, it glints while afloat, the gold counts up, and the next beacon
  fades in after a beat.
- [ ] **F9 Failure you can see** (S). Grounding raises a sand plume and gives
  the camera a short kick; a stall shakes the canvas (through S1) and slackens
  the pennant.
- [ ] **F10 One set of timings** (S). Replace the 23 interface durations with
  a short scale shared by the interface and the camera. Reduced motion is
  honoured by the mode-change sweep and the beacon as well.

**Effects lane** (`physics/Spray.ts`, `physics/Splash.ts`, `post/LensRain.ts`)

- [ ] **X1 Spray that reaches the lens** (S). Slams and green water wet the
  camera through the existing lens-rain pass, as rain and surfacing already do.
- [ ] **X2 Drops that land** (M). Slam and bow drops leave foam flecks and
  rings where they fall, as splash drops already do, and water streams off
  the hull as she rises.
- [ ] **X3 Bubbles under the hull and wake** (S). Seen from below, the wake is
  a trail of rising bubbles.
- [ ] **X4 Soft edges everywhere** (S). Camp smoke, the pickup burst and rain
  fade against depth, as spray does.

**Audio lane** (`src/audio/`)

- [ ] **A1 Buses, memory and lifecycle** (S). Separate music, ambience,
  effects and interface buses, each with a slider and a mute, all saved
  locally. Suspend on a hidden tab, resume cleanly, and recover from a failed
  unlock. Music scheduling no longer depends on a throttled timer.
- [ ] **A2 A room around every sound** (S). A shared reverb send built from a
  synthesised impulse, with three settings crossfaded by state: open deck,
  under water, and near cliffs. Done when no effect is fully dry.
- [ ] **A3 Loudness and priority** (M). A target of about -18 LUFS integrated
  with true peak at or below -1 dB, measured by V5. Each sound gets a logical
  loudness, and the loudest present pulls the rest down (the method DICE
  called HDR audio). Music ducks under thunder, impacts and stings by
  scheduled gain ramps. A reduced dynamic range option for laptop speakers.
- [ ] **A4 Sound in space** (M). The listener sits on the camera. Hull creaks,
  blocks, the anchor chain, splashes and the camp fire become positional
  sources; thunder comes from the bolt's bearing. Head-related filtering for
  a few hero sources, plain panning for the rest.
- [ ] **A5 No audible loop** (S). Longer, decorrelated noise for the beds, or
  one worklet generator, so the 4 s repeat is gone.
- [ ] **A6 Thunder that differs** (S). Per-strike envelopes, several delayed
  rumbles, and a reserved voice so a distant strike cannot starve the rest.
- [ ] **A7 Wind you feel** (S). Apparent wind, not only true wind, and wind in
  the ears that rises with speed.

### Milestone 2: a place worth sailing to

- [ ] **I1 Remove the one-island assumption** (L). An array of island
  descriptors, and one height function that generates both the CPU and the
  shader copy. Everything else in this milestone that adds land depends on it.
- [ ] **I2 Give the island a silhouette** (M to L). Cliffs, a headland, a real
  cove, a beach with a berm and relief, rock faces breaking the canopy. Done
  when the island is recognisable from its outline alone in `hero`.
- [ ] **I3 Depth in the canopy** (M). Shading between crowns, variation in
  height and species, emergent trees and gaps, leaf translucency, and a ragged
  edge in place of the lawn band.
- [ ] **I4 One wind field** (M). After Ghost of Tsushima: a single gust
  texture that drives palms, grass, sails, the pennant, spray and the water's
  gust patches, so one gust front is seen crossing all of them.
- [ ] **I5 An archipelago** (L, after I1). Two more islands, a sandbar, a reef
  and sea stacks, spaced so something new is on the horizon every 45 to 90
  seconds at cruising speed (Dredge's rule: top speed sets the map).
- [ ] **I6 A harbour with lights** (M to L). A small settlement silhouette, lit
  at night, and a lighthouse beam visible from far off.
- [ ] **I7 The reef at the surface's standard** (M). Realistic coral shading
  and lit fish in place of black silhouettes.
- [ ] **K1 Clouds with variety** (M to L). Size and height variation, towers
  and anvils, wispy edges, and no visible repetition toward the horizon or at
  night. Add the dark-edge and inner-glow terms from Guerrilla's Nubis work.
- [ ] **K2 Weather that moves** (L). Pressure zones that redistribute cloud
  cover (Sea of Thieves), a squall front with rain shafts that the player can
  skirt or cross, and lightning that lights the clouds.
- [ ] **K3 A day in play** (S to M). An 8 to 12 minute day at the helm, not
  only in the Tour, with a golden hour inside the first five minutes.
- [ ] **K4 The night sky** (S to M). A moon with phase, point stars, the Milky
  Way.
- [ ] **L1 Bounce light** (M). A probe grid (`LightProbeGridNode`) for sky and
  sand bounce under the canopy, the sails and the hull. Screen-space global
  illumination costs 1.5 to 3 ms and stays on Ultra and Max.
- [ ] **W4 Waves that answer the hull** (M to L). A 256 to 512 texel wave grid
  that follows the ship: slam rings, splash rings, reflections off rocks and
  the jetty. Atlas measured 0.6 to 0.8 ms for theirs; three.js's
  `webgpu_compute_water` is the starting point.
- [ ] **S7 A crew** (L). Six to ten low-detail instanced figures with a few
  looped animations: a helmsman, a lookout, hands at the braces. Needs
  decision 6.
- [ ] **X5 Fire that burns** (M). Flame sheets in place of the static cones,
  embers and sparks carried by the wind, lights that shift with the flicker
  so shadows move, one light per torch, and heat haze over the camp fire.
  The ship's lanterns swing (with S4).
- [ ] **X6 Rain you sail through** (M). Rain streams past with the ship's
  speed, takes its colour from the scene and the lightning, is sheltered under
  sails and deck, and splashes on the planks. Distant sheets come with K2.
- [ ] **X7 Lightning that lights the sky** (M). Clouds lit from inside,
  tapered forks, a leader stroke, and now and then a strike within 300 m. No
  more than three flashes in any second, and a reduce-flashes option (WCAG
  2.3.1).
- [ ] **X8 Spray at the shore** (M). Spray off the breaker's lip and off the
  rocks, and haze over a rough sea (with W6).
- [ ] **X9 Smoke lit from six directions** (M). Baked directional lightmaps
  so smoke answers the sun, the fire and the lightning. Drawn at half
  resolution to bound the overdraw. Camp smoke first; cannon smoke uses it in
  Milestone 4.
- [ ] **A8 The island heard** (M). Shore surf by distance to the breakers,
  fire crackle, land birds by day, insects by night, the lighthouse. The
  audio gains a time-of-day input.
- [ ] **A9 Rain on wood and canvas** (S). Rain sounds different on the deck,
  on the sails and on the sea, mixed by where the camera is.
- [ ] **A10 A worklet for what nodes cannot do** (M). One audio worklet
  holding modal banks for wood knocks and creaks, granular droplets, and
  stick-slip rope. No allocation inside its process call.

### Milestone 3: a voyage

- [ ] **G1 A run with a beginning, a risk and an end** (M). A 3 to 5 minute
  voyage: take cargo to the cove before the squall or the dark closes in.
  Failure costs time or cargo, never the session.
- [ ] **G2 An event director** (M to L). Small events on a pacing timer: a
  wreck to salvage, a bottle with a chart fragment, a castaway, a whale, a
  light on the water at night. Black Flag's team added these because they
  feared an empty sea.
- [ ] **G3 Landmarks that answer** (M). Anchor in the cove to load stores,
  light the lighthouse, raise the strongbox, salute the fort.
- [ ] **G4 A chart and a log** (M). A chart table that records discovered
  landmarks and a ledger of finds, in the existing parchment language.
- [ ] **G5 A spyglass** (S to M). A zoom lens that identifies what it is held
  on.
- [ ] **G6 A rival sail** (L). A second ship that flees, races or shadows the
  player. It needs buoyancy, wake and detail levels for more than one hull.
- [ ] **G7 Progress that persists** (M). Gold buys sail colours, lanterns and
  a flag; discoveries and purchases are saved locally; runs vary by seed.
- [ ] **F4 A view from the deck** (M). A camera at the wheel with free look.
- [ ] **F5 Hazards with consequences** (M). Rocks and shoals damage a hull
  segment, with a repair to the next segment, in place of the push away.
- [ ] **F7 Sound of a crewed ship** (M). An original shanty layer and short
  calls as the crew's voice, sung from where the figures stand (A4, S7), and
  a switch to keep the crew quiet. Stings and silence are Mu3 and Mu4.
- [ ] **G8 The shell** (M). A title screen with continue and new voyage, pause
  on Escape, and settings: key rebinding, sensitivity, separate volumes, a
  haptics switch, motion reduction, markers that do not rely on colour alone,
  and interface scale.
- [ ] **G9 Photo mode** (S to M). Pause, free camera, field of view, depth of
  field, hidden interface, a time of day that does not leak into play, and
  image export.
- [ ] **I8 Wildlife as events** (M). Dolphins at the bow, a breaching whale,
  flying fish, each with its splash. Gulls scatter on a slam, on thunder and
  on the anchor, and dive into the wake.
- [ ] **X10 A glowing wake at night** (M). Bioluminescence in the bow wave and
  wake under a dark sky, as a rare event from the director (G2).
- [ ] **Mu1 More music** (L). Three or four more original tunes in other keys
  and metres, with transition phrases between them, so nothing repeats inside
  ten minutes. The drone follows the key of each section. Checked by V5.
- [ ] **Mu2 Transitions on the phrase** (M). State changes land on phrase
  boundaries with a closing flourish, in the manner of LucasArts' iMUSE, in
  place of restarting at the next bar.
- [ ] **Mu3 A sting for every event** (S to M). Landfall, discovery, danger,
  impact, voyage complete and voyage lost, each quantised to the bar. The
  code has one sting today.
- [ ] **Mu4 Silence on purpose** (S). The silent state is chosen: after a
  storm, at night, under water, and on a long calm reach, the sea carries the
  moment. A player switch for "music off, crew quiet".
- [ ] **Mu5 Players, not oscillators** (M to L). Legato and bow noise on the
  fiddle, a bowed-string or reed model in the worklet (A10), hummed voices
  from formant filters for the crew, and timing that pushes and drags instead
  of only arriving late. Rare's composer kept performances rough on purpose;
  roughness is the target.
- [ ] **Mu6 Music that follows the voyage** (M). Layers answer danger,
  nearness to land and time of day as well as ship speed. No fight music that
  tells the player how to feel before they do.

### Milestone 4: a fight (if decision 2 says yes)

- [ ] **C1 Broadsides** (L). The camera's side picks the battery, with an arc
  preview. Guns, recoil, smoke, a muzzle flash that lights the scene, and
  splash columns through W4.
- [ ] **C2 Damage you can see** (L). Three hull segments, holes, torn canvas
  and fire, with repair to the next segment.
- [ ] **C3 An enemy that sails** (L). Circling, raking and fleeing, with the
  wind position mattering so fights do not all play the same.
- [ ] **C4 The ending** (M). A disabled ship offers a quick sinking or a richer
  boarding by one button.
- [ ] **X11 Gun smoke and fire on the water** (M). Six-way lit smoke (X9)
  rolling downwind from each gun, splinters, embers, and burning wreckage.
- [ ] **A11 The sound of a broadside** (M). A layered shot: a crack, a
  pitched-down thump and a noise tail, filtered and delayed by distance, with
  the reply heard across the water.

### Later, on evidence

- **S9 Water on deck.** Sea of Thieves' texture-space shallow-water sim. Ultra
  and Max.
- **W8 Baked churn around rocks.** Horizon's flipbook volumes; needs offline
  simulation bakes.
- **W9 Crest sharpening.** Deferred in the last pass because the CPU buoyancy
  sampler must apply the same warp.
- **Meshlets.** three.js r187 ships a meshlet pipeline. It is poor value for
  this scene's size.

## Guardrails

- **Frame time.** High runs at about 8.5 ms today. Milestone 1 may add at most
  1.5 ms in total, and each task carries its own cap. A task that cannot fit
  moves to Ultra and Max and gets a cheaper form below.
- **Start-up.** Ready to play is 14 to 19 s. New compute pipelines must go
  through the existing prewarm, and each milestone re-measures.
- **Download.** 32.4 MB today. Crew, a second ship and new islands are built
  procedurally where possible; the cap is set in P0.3.
- **Tiers.** Every new system has an entry in the tier table and an override,
  as the post chain does. Bow spray, splashes, rain and the pickup effects
  have no tier gating today and get it.
- **Audio cost.** A busy bar creates 150 to 200 audio nodes a second, and
  about ten noise beds run even at zero level. Positional sources, reverb
  sends and the worklet must not raise main-thread time; V5 reports voices
  alive per state, and beds at zero level are disconnected.
- **Flashing.** Lightning, muzzle flashes and any pulse stay within three
  flashes a second, with a reduce-flashes option.
- **Motion.** Every new shake, kick and sweep has a strength setting and is
  off under reduced motion.
- **Tests.** The suite stays trimmed. Each new game state gets one functional
  test, and baselines are recaptured once per milestone.

## How each milestone is judged

- **V1 Reference rows.** One review per milestone by an art-director critic,
  scoring each row of `REFERENCE_TARGETS.md` as met, close or not met. Blind
  pairs do not work against a shipped game's stills, because the content
  differs.
- **V2 Motion clips.** The four clips from Milestone 0, reviewed before and
  after.
- **V3 The first five minutes.** A written playtest: what a new player sees,
  does and is told, minute by minute, checked against the "would notice
  missing" list from the code audit.
- **V4 Numbers.** Frame rate, 99th-percentile frame time, start-up and
  download, on the same machine and display as today's figures.
- **V5 The audio bench.** Loudness, peak, clicks, voices and time to repeat
  per state, before and after, with the rendered files kept for listening.
  The numbers catch faults. Whether a tune is good is your call by ear.

## Order of work

1. Milestone 0, then the five lanes of Milestone 1 side by side. `main.ts`
   wiring is merged by one owner. V5 comes before any audio task, and A1
   before the rest of the audio lane because it changes the bus layout they
   all route through.
2. I1 first in Milestone 2, because I2, I5 and I6 build on it. K1, K2, L1 and
   W4 are independent of it.
3. Milestone 3 needs F2 and K3 from earlier milestones; G6 needs I1 for
   somewhere to sail to.
4. Milestone 4 needs G6, W4 and S4.
5. Effects and sound that share an event land together: X6 with A9, X7 with
   A6, X5 with A8, X11 with A11. An effect without its sound, or the reverse,
   reads as unfinished.

## Sources

Rendering:

- Sea of Thieves, SIGGRAPH 2018:
  https://history.siggraph.org/wp-content/uploads/2022/09/2018-Talks-Ang_The-Technical-Art-of-Sea-of-Thieves.pdf
- Atlas water, GDC 2019:
  https://gpuopen.com/gdc-presentations/2019/gdc-2019-agtd6-interactive-water-simulation-in-atlas.pdf
- Horizon Forbidden West water, SIGGRAPH 2022:
  https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Water-Malan.pdf
- Crest (MIT): https://github.com/wave-harmonic/crest
- Nubis voxel clouds: https://www.guerrilla-games.com/read/nubis-cubed
  (abstract only; the deck was not read)
- Ghost of Tsushima wind: https://gdcvault.com/play/1027350/Blowing-from-the-West-Simulating
  (the page is gated; known from its summary)
- three.js r187 examples (MIT): `webgpu_compute_cloth`, `webgpu_compute_water`,
  `webgpu_compute_particles`, `webgpu_compute_particles_rain`,
  `webgpu_lights_clustered`, `webgpu_skinning_instancing`,
  `webgpu_postprocessing_ssgi`, `webgpu_tsl_vfx_tornado`

Design:

- Black Flag naval design: https://www.thesixthaxis.com/2013/09/30/interview-talking-to-the-high-seas-of-ac-iv-with-jean-sebastien-decant/
- Black Flag remake naval systems: https://www.ubisoft.com/en-gb/game/assassins-creed/news/1QoM4qTIi9ERlzDOrxeRgK/assassins-creed-black-flag-resynced-deep-dive-into-the-naval-gameplay
- Dredge: https://www.gamedeveloper.com/design/trawling-in-the-deep-how-black-salt-games-made-spooky-fishing-rpg-i-dredge-i-
- Sailwind: https://www.gamedeveloper.com/design/adventures-in-emergent-storytelling-from-sailwind-s-realistic-take-on-sailing
- Sea of Thieves design: https://www.gamedeveloper.com/design/how-emotional-psychology-and-the-goonies-inspired-i-sea-of-thieves-i-
- Camera shake from trauma: https://gdcvault.com/play/1023146/Math-for-Game-Programmers-Juicing
- Rumble support: https://developer.mozilla.org/en-US/docs/Web/API/GamepadHapticActuator/playEffect
- Accessibility baseline: https://gameaccessibilityguidelines.com/basic/

Effects, sound and feel:

- Six-way lit smoke: https://unity.com/blog/engine-platform/realistic-smoke-with-6-way-lighting-in-vfx-graph
- Soft particles: https://developer.download.nvidia.com/whitepapers/2007/SDK10/SoftParticles_hi.pdf
- Audio scheduling on the audio clock: https://web.dev/articles/audio-scheduling
- Loudness for games, ITU-R BS.2434: https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BS.2434-2018-PDF-E.pdf
- HDR audio, DICE: https://www.slideshare.net/slideshow/adaptive-mixing-in-frostbite/3128152
- Synthesised thunder listening test: https://arxiv.org/abs/2204.08026
- Procedural wind and rain, Paul Weir: https://www.audiokinetic.com/en/blog/behind-the-sound-of-no-mans-sky-a-qa-with-paul-weir-on-procedural-audio
- Sonniss bundle licence: https://sonniss.com/gdc-bundle-license/
- iMUSE: https://en.wikipedia.org/wiki/IMUSE
- Sea of Thieves music, Robin Beanland: https://gamemusic.net/robin-beanland-its-been-really-fantastic-to-make-use-of-these-weird-and-wonderful-instruments-on-sea-of-thieves/
- Musicians aboard ships: https://www.folger.edu/blogs/collation/musicians-on-ships/
- Flash safety, WCAG 2.3.1: https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html
- Motion options, Xbox guideline 117: https://learn.microsoft.com/en-us/gaming/accessibility/xbox-accessibility-guidelines/117

Several of these were seen only as search summaries, not read in full: the
six-way smoke post, the DICE slides, the loudness report, the Weir and
Beanland interviews and the Folger post. The -18 LUFS target in A3 is my
suggestion from the report's range for portable play, not a published browser
standard. All millisecond costs for effects are estimates until P0.3 measures
them.

Looked for and not found, so nothing here rests on them: a Rare talk on
particles or cannon smoke; sourced guidance on hit-stop, time dilation or
rumble design; how Sea of Thieves, Dredge or Sailwind adapt their music; and
evidence for Caribbean instruments around 1700 beyond the ship musicians
above. Also not found: an ocean talk for Black Flag or Skull and Bones;
primary sources for Sea of Thieves' sails, spray or hull masking; water talks
for games from 2023 to 2026; and sourced values for speed-linked field of view
or aiming models. Tasks F1, F6 and C1 are therefore tuned by eye against the
reference rows.
