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

## Scoreboard against the reference demos (2026-11-16)

**Look.** Twelve blind A/B rounds of 32 pairs, judged by an art-director critic with sides randomised per pair (`docs/AAA_PLAN.md` section Q). Final round: **30 of 32**. All pairs won in waves, refraction, caustics, beach transparency, foam, boat-sea, sky and clouds; three of four reflections; two of three underwater (against abyssal-garden).

**Performance.** Same Chrome, 1920x1080, cold cache, each demo's default view, three runs each, on a ~100 Hz display with nothing else on the GPU. Ours is the production build at High, sailing at the start position; the references are single-feature demos with no ship, island or gameplay.

| Demo | Drawn fps | p99 frame | Hitches in 12 s | Download | Ready |
|---|---|---|---|---|---|
| Ours (High, full scene) | 50.4 (steady half rate) | 21.7 ms | 0-3 | 36.9 MB | 23.5 s to playable |
| web-ocean | 101 | 10.5 ms | 0 | ~0 (procedural) | 0.4 s |
| Three.js Water Pro | 101 | 10.5 ms | 0 | 19.4 MB | ~4 s |
| NeuralSpecOcean | 101 | 10.5 ms | 0 | 0.6 MB | ~1 s |
| abyssal-garden | 104 | 10.4 ms | 0 | ~0 | ~5 s |

Pacing now matches (the 28-32 fps judder came from re-uploading every instance buffer on every draw, fixed). Frame rate and start-up do not: the frame is CPU-bound at 8-11 ms of render submission (three r185's per-object overhead), and start-up is dominated by pipeline compilation. Both are the next performance work. Method: `perf-compare.mjs` in the session scratchpad (rAF intervals for references, the app's own render calls for ours).

## Using this

- An agent changing a look attaches a frame of ours beside the row it is aiming
  at, and says which rows moved.
- Search terms that find good references quickly: `sea of thieves night ocean
  moon screenshot`, `assassin's creed black flag storm ocean screenshot`,
  `sea of thieves sunset ship`, `sea of thieves island beach shallow water`.
- View them; do not download them into the repository.
