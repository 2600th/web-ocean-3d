# Reference targets

What the best-shipped games do in each look we render, read off their screenshots
(Sea of Thieves, Assassin's Creed IV: Black Flag, and the community capture
galleries for both). These are *targets for qualities*, not art to copy: we take
the lighting relationships, value structure and palette logic, never assets,
names, logos or fonts.

Every look below lists what the reference does and where we stand, so a change
can be judged against a concrete picture rather than taste.

## Night at sea (Sea of Thieves)

| Reference does | We do (2026-11-13) |
|---|---|
| Whole frame sits in deep teal-navy; never black, never grey | Overhead near-black, sea mid blue |
| Horizon is a soft, slightly lighter teal band — no line | A thin near-white line along the whole horizon |
| Strong moon glitter path on the water, the brightest thing in frame after the moon | Lighting pass: moon glint ×3 at night — a path under the moon, 5.6× the sea beside it (11.6× with the sky pass); still broad, not a narrow column |
| Clouds dark teal with moonlit rims; stars between them | Fixed this pass: clouds no longer glow white |
| Warm lantern and fire points against the cool frame | Lighting pass: camp fire, jetty torches, cove lantern and the ship's stern lantern are real point lights with glow |

## Storm (Black Flag)

| Reference does | We do |
|---|---|
| Desaturated teal-green-grey palette | Navy-grey |
| Crests translucent green where light passes through them | Crest glow exists; storm water too dark to show it (lighting pass: a strike now lights the sea and ship from its side, crests still not green — water scatter colour, not light) |
| Wind-torn spray, long foam streaks along the wind, spindrift off crests | Streaks and hull spray yes; no spindrift off crests |
| Horizon dissolves into spray haze — no line, no hard edge | Bright horizon line |
| Low grey-green ceiling with lighter breaks; no visible sun | Dark flat ceiling (now correctly sunless) |

## Dusk and golden hour (Sea of Thieves)

| Reference does | We do |
|---|---|
| Warm orange-gold dominant; sun low with a broad glitter path | Golden-hour shot reads well |
| Sea in the sun's half is amber, away from it deep teal | — |
| Warm rim light on rigging and sails | Lighting pass: sails transmit the key light (backlit sail luma +21%); rim on the rigging only as the key colour allows |
| Clouds: orange undersides, purple-grey tops | Sky agent did undersides |
| Blue hour after sunset: sea reflects a deep blue sky, warm band only at the horizon | Squall at t=72 shows a flat purple-pink sea under a starry sky |

## Tropical shore by day (Sea of Thieves)

| Reference does | We do |
|---|---|
| Sand near-white cream, the brightest ground in frame | Lighting pass: dry sand albedo lifted to cream, linear luma 0.39 → 0.44, under the surf's white |
| Shallows bright turquoise to cyan, crisp drop to deep blue at the reef edge | Good |
| Palms saturated green with backlit translucent leaves | Palms good; leaves transmit the sun from behind (lighting pass) |
| Vegetation edge is ragged scrub and plants, not a lawn | Improved this pass, still a uniform band from the sea |
| Strong sunny contrast, soft blue aerial haze on far islands | Good |

## Shore break and sea foam (Sea of Thieves, with real surf as the ground truth)

| Reference does | We do (2026-11-15) |
|---|---|
| Foam is never one flat white. Each patch has a dense core, raft rims, and lace and threads at its edges, stretched along the wind | One feathered pattern (wind streaks, raft rims, bubbles) under every foam source, where the amount sets the white fraction. Near and mid field good |
| A whitecap is a small solid crest (stage A) inside a larger patch that thins as it decays (stage B) | Stage A from the crest fold, stage B from the buffer, capped and broken by a patchiness field. Strong seas at 150–300 m still show some smooth plates |
| A breaker shoals and steepens, its face goes thin and glows green, the lip pitches forward, then a white bore runs in | Swell faded per band in the surf so the sets read. The face leans with the Iribarren number, the thin face glows, and the lip and roller go white |
| The bore runs up the sand as a thin sheet. The uprush is fast and the backwash slow, and both leave a lacy edge | Hunt run-up with a skewed swash front, lace that thins with age, and a foam line at the front |
| Behind the swash the sand stays dark and glossy, then dries back to matte over seconds; a foam line is stranded at the top of the run-up | A glossy film (1.6 s) over a damp darkening (12 s), with a swash mark at maximum run-up |

Sources, and what each one gave:

- **Crest (Wave Harmonic, MIT-licensed source).** The feathered foam threshold:
  white = smoothstep(t, t + feather, pattern), so an amount reads as a
  fraction of white. Also per-band wave attenuation in shallow water,
  saturate(2h/λ).
- **Monahan and O'Muircheartaigh (1980), and Monahan (1971).** Whitecap stages
  A and B. Coverage against wind speed is the law the buffer's deposit rate is
  calibrated to.
- **Battjes (1974), the surf-similarity (Iribarren) number.**
  ξ = tan β / √(H/L₀) separates spilling breakers (ξ < 0.5) from plunging
  ones, and drives how far the face leans.
- **Hunt (1959) and Stockdon et al. (2006).** Run-up R ≈ ξ·H, written here as
  0.75·tan β·√(H·L₀). Stockdon's swash statistics set the skew between the
  fast uprush and the slow backwash.
- **Horizon Forbidden West (Malan, SIGGRAPH 2022 Advances course).** A breaker
  is one cross-section animated in XYZ and slid along the wavefront, so the
  crest runs ahead of the face beneath it. Here it is done in closed form, as
  a shoreward lean whose compression steepens the face.
- **Wet-sand reflectance studies (e.g. Nolet et al. 2014).** Darkening is
  steep at low moisture and saturates. A thin water film adds gloss, not
  darkness.
- **Sea of Thieves (Rare's talks and screenshots).** The art target:
  painterly foam with readable shapes, a translucent breaker face, and a
  bright cream beach under white surf. The talks say what it does, but give
  no formulas.

## Added 2026-11-17: the ship, the island, the squall, fire and sound

These rows came with `docs/AAA_ROADMAP.md`. "We do" is that file's audit tables,
read off the gallery stills and the code. "Reference does" is what shipped
sailing games are described as doing in the sources the roadmap lists: Rare's
SIGGRAPH 2018 talk on Sea of Thieves, the Atlas and Horizon Forbidden West
water talks, the Black Flag, Dredge, Sailwind and Sea of Thieves interviews, and
the DICE and No Man's Sky audio material. No game was watched or listened to
for this, and a few sources were read only as summaries (the roadmap says which).
Nothing here is art, a name or a tune to copy.

### The ship under sail (Sea of Thieves, Black Flag, Sailwind)

| Reference does | We do |
|---|---|
| Canvas fills and luffs with the wind. A sail close-hauled is flatter than one running, and a ship head to wind shakes her canvas | Topsails permanently furled, courses hang flat. The sail setting reaches the audio and nothing visible; the only animation is a 1.2% scale on the sails |
| Rigging sags and tightens with the sails (Sea of Thieves solves a catenary per rope on the CPU with a slack value and bends a tube in the vertex stage) | Rigging is rigid |
| Parts of the ship answer the player: the wheel turns, the rudder swings, the yards brace round | Nothing moves; wheel, rudder and yards are fixed |
| A pennant or flag shows the wind, and doubles as the player's indicator | No flag. Head to wind is shown by a toast and the luff sound |
| The hull is wet where the sea reaches it, and dries over seconds | `Wetness.ts` wets surfaces under rain only. No waterline band, no spray-driven wetting |
| Figures on deck: a helmsman, a lookout, hands at the braces | Nobody is aboard |
| The hull is as detailed below the waterline as above: planking, keel, rudder | `underwater` shows a smooth dark underside with no planking, keel or rudder |

### The ship at night (Sea of Thieves, Black Flag)

| Reference does | We do |
|---|---|
| Lit stern windows and lanterns make the ship a readable shape against a dark sea, and lantern light falls on sails, deck and rigging | In `night`, `squall` and `moonlit` she is a near-black silhouette with one lantern. No lit windows, no lanterns lighting the canvas |
| Lantern light reflects in the water around the hull | Lantern glow does not reach the sails or the sea |
| Lights move: lanterns swing with the roll | Lights flicker in brightness and never move |
| A glowing wake is one of the moments that make a sea feel alive | None |

### The island seen from the sea (Sea of Thieves, Black Flag)

| Reference does | We do |
|---|---|
| A sculpted silhouette you can name from a kilometre off: cliffs, a headland, a cove, trees standing above the canopy | One low dome. The cove, fort, lighthouse and skull rock exist as props, but the landmass itself has no cliffs or relief in its outline |
| Depth in the canopy: shading between crowns, variation in height and species, gaps and emergent trees | A uniform green texture of same-sized crowns with no shading between them |
| A ragged vegetation edge, then scrub, then a beach with some relief | A flat lawn band from the sea, then a thin flat beach. Props read as specks |
| New land on the horizon every 45 to 90 seconds at cruising speed (Dredge: top speed sets the map) | One island, 1.4 km from the start, and nothing else |
| Landmarks answer the player: anchor in the cove, light the lighthouse | Nothing on the island can be interacted with |
| Soft blue aerial haze on far land | Good (see the tropical shore table) |

### A squall front (Black Flag, Sea of Thieves)

| Reference does | We do |
|---|---|
| A front you can see coming: a dark wall of cloud with towers and anvils, and rain shafts under the cells | The squall exists only in the Tour. Clouds are same-sized puffs in rows toward the horizon, with hard cut-out edges and no tall forms |
| Weather moves, and can be skirted or sailed into (Sea of Thieves redistributes cloud cover with pressure zones) | Weather is fixed in play |
| Rain streams past with the ship's speed, takes its colour from the scene, is sheltered under canvas and splashes on the planks | 5,000 additive sprites of one constant colour within 34 m of the camera, carried with it. Not sheltered, no splashes on deck, no distant sheets |
| Lightning lights the clouds from inside, with tapered forks and now and then a close strike | A branching ribbon whose forks are as bright as the main channel. It lights the ship and sea, not the clouds. No strike closer than 900 m; about one flash every 7 s with no reduce-flashes option |
| Wind-torn spray off the crests, and crests translucent green toward the light | No spindrift; crests not green (see Storm above) |
| The horizon dissolves into spray haze | A bright horizon line |

### Fire and smoke (Black Flag, Sea of Thieves)

| Reference does | We do |
|---|---|
| Animated flame, with embers or sparks carried on the wind | The camp fire and torches are static six-sided cones with a flickering emissive colour. No flame sheets, embers, sparks or heat haze |
| Fire light that moves, so shadows dance | Lights flicker in brightness, never move, and cast no shadows. The two torches share one light |
| Smoke is soft where it meets geometry and is lit from several directions (the six-way lit smoke method in the roadmap's sources) | 34 billboards drifting with the wind, hard-edged against the ground. Rain does not affect them |

### Sound: the sea bed (Sea of Thieves, Black Flag, No Man's Sky)

| Reference does | We do |
|---|---|
| No audible repeat in the sea, wind and rain beds | All beds loop the same 4 s of noise |
| Every sound has a room: the deck, below decks, under water | Reverb on the music only. Every effect and ambience is dry |
| Sounds sit where they happen: creaks on the hull, thunder from the bolt | Stereo pan and a 1/distance gain for the ship and gulls only. No listener, no positional sources |
| Buses per kind, a loudness target and priority ducking (DICE's HDR audio) | One path into a compressor, a limiter and a soft clipper. Loudness figures exist only in a header comment |
| Volume per kind, remembered between visits | Master and Music sliders; volume resets to 0.4 every visit |

### Sound: the storm (Sea of Thieves, Black Flag)

| Reference does | We do |
|---|---|
| Thunder arrives after a delay set by distance, from the bearing of the strike, and no two are alike | Delayed by distance, which is right. No bearing, and the same two envelope curves on every strike. A distant strike holds a voice slot for up to 37 s |
| Rain sounds different on wood, canvas and sea, mixed by where the listener is | One rain bed |
| Wind in the ears rises with speed | True wind only, with no speed term |
| Music gives way to thunder and impacts | No ducking |

### Sound: the island (Sea of Thieves, Black Flag)

| Reference does | We do |
|---|---|
| The island is heard before it is reached: shore surf, fire crackle, land birds by day, insects by night | Silent. No surf although breakers are drawn, no crackle, no insects. The audio has no time-of-day input |
| Silence is used on purpose, and music follows the voyage | Two tunes in D minor, 40 bars in all, layered by ship speed. A silent state exists and is never chosen. One sting |

Every sound row is read from the synthesis code. Nothing has been listened to,
so none of them can be scored by ear yet. V5 in the roadmap, the audio test
bench, will give figures for loudness, clicks and repetition.

## Scoreboard against the reference demos (2026-11-16)

**Look.** Twelve blind A/B rounds of 32 pairs, judged by an art-director critic with sides randomised per pair (`docs/AAA_PLAN.md` section Q). Final round: **30 of 32**. All pairs won in waves, refraction, caustics, beach transparency, foam, boat-sea, sky and clouds; three of four reflections; two of three underwater (against abyssal-garden).

**Performance.** Same Chrome, 1920x1080, cold cache, each demo's default view, three runs each, on a ~100 Hz display with nothing else on the GPU. Ours is the production build at High, sailing at the start position; the references are single-feature demos with no ship, island or gameplay.

| Demo | Drawn fps | p99 frame | Hitches in 12 s | Download | Ready |
|---|---|---|---|---|---|
| Ours (High, full scene), 2026-11-16 | 50.4 (steady half rate) | 21.7 ms | 0-3 | 36.9 MB | 23.5 s to playable |
| Ours (High, full scene), 2026-11-17 | **98.5 (full rate)** | **10.6 ms** | 0-1 | 36.9 MB | 22.9 s to playable (warm; was 24.4 the same session) |
| Ours (High, full scene), r187 WebGPU-only, 2026-11-17, **120 Hz display** | **111-113** (full rate, ~5% doubled frames; one run of three settled at 76) | 17.1-18.4 ms | ~65 | **32.4 MB** | **17.3-19.7 s** to playable. Same display and session, r185: 60 (steady half rate), 23-24 s |
| web-ocean | 101 | 10.5 ms | 0 | ~0 (procedural) | 0.4 s |
| Three.js Water Pro | 101 | 10.5 ms | 0 | 19.4 MB | ~4 s |
| NeuralSpecOcean | 101 | 10.5 ms | 0 | 0.6 MB | ~1 s |
| abyssal-garden | 104 | 10.4 ms | 0 | ~0 | ~5 s |

Pacing now matches (the 28-32 fps judder came from re-uploading every instance buffer on every draw, fixed). As of 2026-11-17 the frame rate matches too, on this ~100 Hz display. The bottleneck had been replaying submission in Chrome's GPU process, not the GPU. Batched submits, scene-wide uniforms in the render group, fewer uniform writes and framebuffer copies, and a pacer gate on the slower pipeline stage took High from a steady 50 to 98.5 fps, p99 10.6 ms (`docs/PERFORMANCE.md`, Results, 2026-11-17). A 120 Hz display is likely still out of reach. Start-up is the remaining gap. It is bound by FXC pipeline compiles in the GPU process, and sharing the per-mesh shader copies cut those from 295 to 235. The reference demos are ready in 0.4-5 s. Method: `perf-compare.mjs` in the session scratchpad (rAF intervals for references, the app's own render calls for ours).

## Using this

- An agent changing a look attaches a frame of ours beside the row it is aiming
  at, and says which rows moved.
- Search terms that find good references quickly: `sea of thieves night ocean
  moon screenshot`, `assassin's creed black flag storm ocean screenshot`,
  `sea of thieves sunset ship`, `sea of thieves island beach shallow water`.
- View them; do not download them into the repository.
