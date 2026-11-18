# Review: One day's voyage (60 s, 1920×1080, 30 fps)

Delivery files, in `renders/`:

- `web-ocean-3d-voyage.mp4`: the master (250 MiB, about 35 Mbit/s).
- `web-ocean-3d-voyage-share.mp4`: the same film re-encoded for uploading (about 140 MiB, about 19 Mbit/s, H.264, AAC 256 kbit/s, fast start).
- `web-ocean-3d-voyage-preview-720p.mp4`: a 1280×720 copy for messaging apps and quick viewing (27 MiB).
  It is 3.7 Mbit/s, which is visibly soft on open sea and foam: a size-capped preview, not a copy to publish.
- `web-ocean-3d-voyage-1080p-excerpt.mp4` and `-1080p-hull-pass.mp4`: the master's first 5 seconds and its hull pass (4.0 to 7.5 s), untouched, to judge the real quality on another device.

## Fourth cut (2026-11-18)

Re-shot and re-rendered on the build where spray along the hull no longer
shows black specks (spray is kept out of the screen-space contact shadows and
ambient occlusion). The cut points, titles and soundtrack are unchanged. The
third cut's renders and footage are kept in `renders/v3/` and
`assets/footage-v3/`. On this cut: lint is clean, the capture logged no
console errors, a 24-frame sheet of the master and four close frames of the
hull pass were read by eye, duration is 60.000 s (1800 frames), and loudness
is -15.1 LUFS with a peak of -1.1 dBFS, measured, not heard. The full `check`
and the snapshot review were not run again, since no composition file
changed. Still visible in the hull pass: soft pale puffs of mist on the hull
side.

## Third cut (2026-11-18)

Re-shot and re-rendered on the build where the sails no longer break out in
black patches (the screen-space contact shadows are kept off the canvas) and a
hand can no longer be stood on a yard. The cut points, titles and soundtrack
are unchanged. The second cut's renders and footage are kept in `renders/v2/`
and `assets/footage-v2/`. On this cut: lint is clean, the capture logged no
console errors, a 24-frame sheet of the master and four close frames of the
hull pass (4.2 to 7.4 s) were read by eye, duration is 60.000 s (1800 frames),
and loudness is -15.1 LUFS with a peak of -1.1 dBFS, measured, not heard. The
full `check` and the snapshot review were not run again, since no composition
file changed.

## Second cut (2026-11-18)

Re-shot and re-rendered on the build with three reported faults fixed: the reef
dive no longer passes through rock, the sea off the beach has clean breaker
lines over a soft wash, and the crew are nine toy-style pirates. The cut
points, titles and soundtrack are unchanged. The first cut's renders and
footage are kept in `renders/v1/` and `assets/footage-v1/`. On this cut: lint
is clean, the beat sheet and 24 decoded frames were read by eye, duration is
60.000 s, and loudness is unchanged (-15.2 LUFS, true peak -1.1 dBTP). The full
`check` was not run again, since no composition file changed.

## What was checked (first cut)

| Check | Result |
| --- | --- |
| `npx hyperframes lint` | 0 errors, 0 warnings |
| `npx hyperframes check --json` | ok; lint, runtime, layout, craft and audio clean |
| Contrast | 16 of 17 samples pass on the final timeline. The one that fails is the "s" of "A ship" at 3.33 s, over a white cloud: 2.52 against a 3:1 target. Left as it is: 190 px display type with a dark shade, on screen for 3 s. |
| `npx hyperframes snapshot --review` | every scene's settled frame, all 12 cuts and the phone-width sheet read by eye |
| Render | 60.000 s, 1800 frames, H.264 yuv420p, AAC 48 kHz stereo |
| Decoded frames | 24 frames across the master and 4 from the share copy looked at |
| Sound | measured, not heard: -15.2 LUFS integrated, true peak -1.1 dBTP, loudness range 5.2 LU, no clipping, no silent stretch |

## The cut

| Film time | Shot | Source |
| --- | --- | --- |
| 0.0 to 4.0 | The ship under full sail, "A ship" | `quarterp.mp4` from 3.2 s |
| 4.0 to 7.5 | Close along the hull, crew on deck | tour, wave skim, from 13.3 s |
| 7.5 to 12.5 | The player's helm view with the real interface, "Take the helm" | `helm.mp4` from 1.6 s |
| 12.5 to 18.0 | Under the surface: reef, rays, kelp | tour, reef dive, from 22.3 s |
| 18.0 to 22.5 | The island behind the ship, "An island" | tour, island reveal, from 44.0 s |
| 22.5 to 26.5 | Along the beach to the camp and jetty | tour, beach flyover, from 52.5 s |
| 26.5 to 31.0 | Golden hour into a red sky | tour, golden hour, from 65.8 s |
| 31.0 to 33.5 | Dusk, lanterns lit, sea getting up | tour, squall, from 70.55 s |
| 33.5 to 39.5 | The squall, two lightning strikes, "A sea raised live" | `storm.mp4` from 1.0 s |
| 39.5 to 44.5 | The helm view in the squall, one strike | `stormhelm.mp4` from 1.0 s |
| 44.5 to 53.6 | Moonlit night, "Every wave, worked out live in your browser" | `moonlit.mp4` from 0.6 s |
| 53.6 to 60.0 | Night watch with the mark, "Set sail" and the address | tour, night watch, from 82.4 s |

## Known limits

- Nobody has listened to the soundtrack. It is the game's own score and sea,
  rendered offline by its audio engine on this cut's timeline, then raised
  2.8 dB through a limiter to reach social loudness.
- The click detector on the raw score flagged two moments (28.3 s and 29.5 s).
  The largest sample-to-sample step in the file is 0.23 of full scale, so
  neither is a hard discontinuity, but they have not been listened to.
- The cut from dusk (31.0 to 33.5) into the squall goes from a dark frame to a
  brighter grey one. A three-frame dip covers the cut.
- The squall is 11 s of grey sea across two shots; the second (the helm view)
  is the least varied shot in the film.
- The end card's address is the deployed site, which does not yet carry the
  work on the `aaa-roadmap-m1` branch shown in this footage.
- 16:9 only. A vertical cut needs its own framing pass, not a crop.
