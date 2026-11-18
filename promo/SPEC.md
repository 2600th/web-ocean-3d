# SPEC: One day's voyage

```text
GOAL:    the viewer sees a ship, an island and a storm that look like a shipped sailing
         game, learns it runs live in a browser, and wants to open the address.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps
TIME:    60 s x 30 = 1800 frames. No beat grid: the score is the game's own, rendered on
         the cut's timeline, so its changes fall on the cuts (storm at 31 s, calm at 44.5 s).
LOOK:    ground = the game's own picture, full frame, ungraded (no filter, no vignette).
         ink = parchment #f6ead0 (the interface's --parchment-hi); accent = gold #ffd36b
         (--gold) on one word only ("live") and wax red #c23a2c (--wax-hi) on "3D" in the mark.
         display = Pirata One 400, 190 px (17.6% of height) for the three tagline words,
         130 px for the claim, 170 px for the mark; quiet line = Alegreya 500, 46 px, +4% tracking.
         Both faces are the game's own files. Legibility: one soft shadow, 28 px blur at 55%,
         no boxes, no bars.
OBJECTS: the ship (real footage), the island, the helm interface (the real HUD, captured
         from the page, not redrawn), lightning, lanterns. Nothing is drawn by hand.
MOTION:  the camera moves inside the footage; titles do not move. Title entrances:
         per letter 0.07 s apart (opacity 0 -> 1, blur 8 -> 0 px, 0.2 s, power3.out);
         per word 0.09 s apart; whole line blur-in 0.22 s. Exits 0.2 s blur-out. A landed
         title is dead still.
SHOTS:   (frames at 30 fps; source in brackets)
  01  0000-0120  [quarterp]  ship under full sail off the port quarter; "A ship" reads 14-113
  02  0120-0225  [tour wave-skim]  low pass along the hull, crew on deck
  03  0225-0375  [helm]  chase camera with the real HUD, a turn to starboard; "Take the helm"
  04  0375-0540  [tour reef-dive]  through the surface to light shafts and kelp
  05  0540-0675  [tour island-reveal]  the island's ridge behind the ship; "An island"
  06  0675-0795  [tour beach-flyover]  surf and the beach
  07  0795-0930  [tour golden-hour]  the hull on the sun's path
  08  0930-1005  [tour squall]  dip cut into the dark; lanterns on black water
  09  1005-1185  [storm]  camera boat in the gale; strikes at 1053 and 1167; "A sea raised live"
  10  1185-1335  [stormhelm]  at the helm into the seas; strike at 1245
  11  1335-1608  [moonlit]  calm under the moon, lanterns lit; the claim, two lines
  12  1608-1800  [tour night-watch]  the mark lands per letter, then "Set sail" and the address
BANNED:  dark void with floating elements; glow, bloom, particles added in post; gradients;
         glass cards; everything fading and sliding up; titles that drift after landing;
         corner labels, borders, HUD decoration drawn by the film; three cards in a row;
         a title over a dimmed screenshot; the logo docked in a corner; invented interface.
QA:      f0 (thumbnail: ship fills the right half, title not yet on); f60 "A ship" landed;
         f300 HUD legible, title clear of it; f600 "An island" clear of the ship;
         f1055 strike frame; f1110 "A sea raised live" legible on the storm; f1420 claim
         legible at 360 px wide; f1700 mark, CTA and address all on, nothing over the ship.
```
