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

## Using this

- An agent changing a look attaches a frame of ours beside the row it is aiming
  at, and says which rows moved.
- Search terms that find good references quickly: `sea of thieves night ocean
  moon screenshot`, `assassin's creed black flag storm ocean screenshot`,
  `sea of thieves sunset ship`, `sea of thieves island beach shallow water`.
- View them; do not download them into the repository.
