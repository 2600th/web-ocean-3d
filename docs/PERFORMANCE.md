# Performance

Two different things measure this project, and they answer different questions.

- **`npm run bench`** — `scripts/benchmark.mjs`. A headed, focused Chrome window
  with vsync off, real WebGPU timestamp queries, and hundreds of samples per
  configuration. This is the only source of performance truth here.
- **`npm test`** — the Playwright suite. Functional and regression assertions.
  It runs in automated Chromium, which paces `requestAnimationFrame`
  independently of load, so it cannot settle a frame budget and does not try to.

Everything in the Results section below comes from the benchmark. The raw run is
checked in at [`bench-results/reference.json`](../bench-results/reference.json).

## Running the benchmark

```bash
npm run bench
```

That is the whole thing: it type-checks and builds, starts a `vite preview`
server on a free port, launches Chrome, and measures the matrix. Nothing needs to
be running first, and it does not reuse a server it did not start — an earlier
version of this harness silently benchmarked a stale `dist/` that another process
had left on port 4173.

Useful variations:

```bash
node scripts/benchmark.mjs --only webgpu-high,webgl-low   # just the gated pair
node scripts/benchmark.mjs --frames 1200 --warmup 300     # longer sample
node scripts/benchmark.mjs --dpr 2                        # 3200 x 1800
node scripts/benchmark.mjs --help
```

The run exits non-zero unless **every gated configuration is a measured PASS**.
An UNVERIFIED gate is not a pass and does not exit zero.

Take the measurement with nothing else on the GPU, and do not click away from the
window: the harness checks `document.hasFocus()` and refuses to report a pass
from a background window.

## What the harness does that the test suite cannot

**It gets real GPU time.** Three.js only allocates a timestamp query pool when
the backend's `trackTimestamp` flag is set, and this project constructs its
renderer without it. The flag does not have to be set at construction, though:
`WebGPUBackend` requests its device with *every* feature the adapter advertises,
so `timestamp-query` is already enabled on the device, and the pool is built
lazily on the first instrumented render pass. The harness therefore sets
`renderer.backend.trackTimestamp = true` at runtime, before the first sampled
frame, and no change to `src/` is required. (`renderer.trackTimestamp` is *not*
the flag — assigning it creates a stray property and changes nothing. The
property lives on the backend.)

Each sampled frame issues ~110 render passes at High. `resolveTimestampsAsync`
returns the summed GPU duration of the most recent frame's passes; the pool holds
2048 queries, so it has to be resolved every frame regardless.

**It pins the world.** Before sampling, the harness calls the deterministic reset
hook: every clock is rewound and every accumulation buffer cleared, and the world
is settled at simulation time 0 before being stepped forward at a fixed 1/60 s.
Without this the sampled scene is wherever the session happened to drift to.
Measured, before this was added: two runs of the Max tier differed by 505 000
triangles and 2.4 ms of GPU time, because the ship had wandered far enough
between them to change what the 4096² shadow frustum contained. With it, repeat
runs of Max produce byte-identical triangle counts and GPU medians within 1 %.

**It knows when the frame boundary is.** The harness pauses the app's own loop
and drives one frame per rAF tick through `Loop.step`, which runs the identical
update and render path and *awaits* the render. Owning the frame boundary is what
makes the rest possible: draw-call counters can be read before three.js resets
them, and a frame's timestamps can be resolved knowing the pool contains that
frame's passes and nothing else.

**It refuses to lie.** A configuration is reported UNVERIFIED, never PASS and
never FAIL, if any of these hold — a harness problem is not a regression:

| Condition | Why it invalidates the number |
|---|---|
| No WebGPU adapter | there is nothing to measure |
| `isFallbackAdapter`, or adapter/ANGLE strings naming SwiftShader, llvmpipe, lavapipe or a basic renderer | software rasterisation |
| Timestamp queries unavailable, or never returning a usable duration | no GPU time, only a CPU upper bound |
| rAF throttling detected | the browser is pacing the page, not the renderer |
| Window not focused, or page not visible | background tabs are throttled and descheduled |
| Fewer than 200 samples | percentiles over a handful of frames are noise |
| The requested backend is not the one that booted | measuring something else |
| The quality tier drifted mid-sample | `AdaptiveQuality` changed the thing under test |
| The ship did not load | a frame cost without the hero object is not this project's frame cost |
| The deterministic reset failed | the sample is not reproducible, so it cannot be a regression baseline |
| `--headless` | headless pacing and GPU scheduling are not representative |

Throttling is detected by comparing the delivered rAF interval against the
*measured* frame cost, not against the app's own `loop.stats.frameMs`. The app
times its render call, and on WebGPU that call returns once the work is
submitted — in the reference run it reads well under a millisecond for a High
frame that costs 3.09 ms on the GPU. Compare a 10.0 ms delivered interval against 0.5 ms and a
perfectly healthy 141 FPS run is classified as throttled.

### Chrome flags, and what they change

| Flag | Effect on the measurement |
|---|---|
| `--disable-gpu-vsync`, `--disable-frame-rate-limit` | rAF is not paced to the display refresh, so the delivered rate reflects the page rather than the monitor |
| `--disable-dawn-features=timestamp_quantization` | Chrome otherwise rounds every WebGPU timestamp to 100 µs. Rounding ~110 passes independently and then summing them puts several milliseconds of noise into the frame total — more than the gap between two quality tiers |
| `--enable-unsafe-webgpu`, `--enable-dawn-features=allow_unsafe_apis` | timestamp queries |
| `--enable-features=Vulkan` | matches the Playwright project configuration |

The full flag list is recorded in every results file, because each of them
changes what the numbers mean relative to a stock browser.

### Deliberate distortions

Both of these make the numbers slightly *cleaner* than the app achieves
unaided, and both are recorded in the output:

- Frames are stepped at a fixed 1/60 s rather than wall clock, so the simulation
  advances identically in every configuration.
- Resolving timestamps maps a buffer back every frame, which drains the pipeline
  between frames. GPU time is measured on the GPU and is unaffected; **CPU** frame
  time loses the overlap it would normally get with the previous frame's GPU work.

And one caveat on the reported delivered FPS: it is the rate at which the browser
delivered rAF callbacks during a three-second observation of the app's own loop,
which is an *upper bound* on presented frames rather than a count of them. It is
recorded to prove the browser was not throttling, and for nothing else. The
headline is GPU frame time.

## Recorded hardware

Detected by the browser at runtime, not read off the OS — this machine has two
GPUs and only the browser knows which one it bound.

| | |
|---|---|
| WebGPU adapter | vendor `nvidia`, architecture `blackwell` (Chrome blanks `device`/`description`) |
| ANGLE renderer | `ANGLE (NVIDIA, NVIDIA GeForce RTX 5090 (0x00002B85) Direct3D11 vs_5_0 ps_5_0, D3D11)` |
| Driver | 32.0.16.1062 — OS-reported, for provenance only |
| Also present | AMD Radeon(TM) Graphics (integrated); not the adapter Chrome selected |
| CPU | AMD Ryzen 7 9800X3D, 16 threads |
| OS | Windows 11 Pro 10.0.26300 |
| Browser | Chrome 157.0.8103.51, headed, focused, driven by Playwright 1.62.1 |
| Resolution | 1600 × 900 @ DPR 1 |

## Budgets

| Configuration | Gate | Rationale |
|---|---|---|
| WebGPU, High | GPU p50 < 16.7 ms | the 60 FPS target desktop budget |
| WebGL2, Low | GPU p50 < 33.3 ms | the 30 FPS fallback floor |

Those two are the gates from the brief. Every other tier is measured against its
backend's budget as well, but a failure there is informational: `max` is
deliberately allowed to cost more than 60 FPS on hardware that is not this.

The gate is on GPU frame time rather than on delivered FPS deliberately. Frame
time is a property of the renderer; delivered FPS is a property of the renderer,
the compositor, the display and the browser's scheduling policy, and this project
has already been burned once by treating the second as if it were the first.

## Results

### 2026-11-18: fewer draws and fewer passes, the picture unchanged

A pass over the whole frame with one rule: no canonical shot may move by more
than 1.0 mean Delta E (CIE94, the visual suite's measure, on a 0 to 100 scale)
and every shot must still pass its own threshold. It ran until five rounds in a
row each took less than 5% off the frame.

**How it was measured.** `node scripts/benchmark.mjs --only webgpu-high`, three
runs per round, on a build made into a scratch folder (`OCEAN_DIST`, below):
1600 x 900, DPR 1, r187, Chrome with vsync and the frame-rate limit off, nothing
else on the GPU. Three numbers per round:

- **Frame interval**: the mean time between frames with the loop running free,
  from the benchmark's pacing phase. This is the overall frame time, and the
  number the 5% rule was applied to.
- **CPU**: the p50 of one stepped frame on the page (updates plus encoding).
- **GPU**: the p50 of the frame's timestamp queries.

Frame interval moves by about 2% from run to run. GPU p50 has two modes about
0.18 ms apart from round 5 on (3.9 and 4.1 ms at High); the lower is quoted
where two runs of three agree.

| Round | Change | Interval | CPU | GPU | Draws | Passes | Interval, vs the round before | Worst shot, mean Delta E |
|---|---|---|---|---|---|---|---|---|
| | Before | 6.39 ms | 6.77 ms | 4.64 ms | 495 | 61 | | |
| 1 | Sun cascades drawn once a frame, not once per camera | 6.04 | 6.23 | 4.22 | 397 | 58 | -5.5% | 0.013 |
| 2 | Reef rock and kelp cast sun shadows only with the eye in the water | 5.79 | 5.93 | 4.22 | 381 | 58 | -4.2% | 0 |
| 3 | Empty level-of-detail meshes are not prepared | 5.73 | 5.43 | 4.23 | 381 | 58 | -1.0% | 0 |
| 4 | Ocean, spray, spindrift and splash depth in one draw, not two | 5.38 | 5.33 | 4.14 | 372 | 58 | -6.1% | 0 |
| 5 | The mirror stops drawing what cannot be made out in it | 5.21 | 5.20 | 3.92 | 356 | 58 | -3.1% | 0.046 |
| 6 | Casters that cannot show are dropped | 5.23 | 5.13 | 3.92 | 343 | 58 | +0.2% | 0.046 |
| 7 | Lantern ironwork merged, stern windows in one draw | 5.27 | 5.13 | 3.92 | 335 | 58 | +0.7% | 0.046 |
| 8 | Nested renders do not walk the scene's transforms again | 5.15 | 5.07 | 3.92 | 335 | 58 | -2.2% | 0.046 |
| 9 | Wave transform: a horizontal and a vertical stage per pass | **4.91** | **4.83** | **3.87** | 327 | 50 | -4.6% | 0.22 (see below) |

Before to after at High: **frame interval 6.39 to 4.91 ms (-23%), CPU 6.77 to
4.83 ms (-29%), GPU 4.64 to 3.87 ms (-17%)**, 495 to 327 draws, 61 to 50 render
passes, 8.7 M to 6.6 M triangles a frame. Rounds 5 to 9 are the five in a row
under 5%.

The other tiers, one run each of the whole matrix, before and after:

| Tier | Interval | CPU | GPU |
|---|---|---|---|
| Low | 4.49 to 3.84 ms | 4.8 to 3.1 ms | 2.74 to 0.80 ms |
| Medium | 6.10 to 4.94 ms | 6.3 to 4.6 ms | 2.70 to 2.41 ms |
| Ultra | 7.18 to 6.58 ms | 7.2 to 4.9 ms | 6.43 to 5.70 ms |
| Max | 8.46 to 7.32 ms | 8.0 to 5.9 ms | 6.94 to 6.08 ms |

Live, in plain Chrome at 1920 x 1080 on a 120 Hz display at High: 106.5 fps
drawn before (one run; the other sat in an occluded window), 112.7 and 113.6
after. The display is the limit there. Click to ready: 17.6 s before, 16.9 and
18.8 s after, so start-up did not move.

**What each round was.**

1. **Sun cascades once a frame** (`installShadowRefreshHold`,
   `holdShadowRefresh`). `ShadowNode.updateBefore` refreshes a light's map the
   first time each *camera* draws something it lights. The planar mirror has a
   camera of its own, so all three cascades were drawn twice a frame: 204
   caster draws for the view, then 98 for the mirror into the same maps. The
   second set was also the one the view ended up sampling, since the mirror's
   command buffer is submitted first, and it was the thinner one: three gives
   the shadow camera the layers of the camera it draws for, and the mirror's
   camera has only layer 0. The mirror now reads the maps the view drew.
2. **Reef casters only with a wet eye** (`SunShadows.setCasterLayers`). Round 1
   brought the reef rock and kelp back into the maps (they are on the
   underwater layer). From above the surface their shadows moved no shot by
   more than 0.013, for about a hundred caster draws a frame, so they cast only
   while the eye is within 0.75 m of the surface or under it. That is what the
   frame showed before round 1, by accident; now it is a rule.
3. **Empty levels are dropped** (`installSubmissionFixes`). An `InstancedMesh`
   with `count === 0` draws nothing, and three finds out last, after the
   render object lookup, the cache keys and the bindings refresh. 69 of the
   scene pass's 225 objects were empty levels of reef and seabed rock. They are
   let through once per render context, so pipelines are built when they were
   before, and never filtered during a compile.
4. **One draw, not two** (`forceSinglePass`). Three draws a transparent
   double-sided material as a back-face pass and a front-face pass. The ocean
   is opaque and writes depth, so the nearest face wins either way; the spray
   and spindrift quads face the eye, so the back-face pass culled everything it
   was given. 760 000 triangles a frame went through the vertex stage for
   nothing. The splash's coloured sheets keep both passes.
5. **A leaner mirror** (`VIEW_ONLY_LAYER`). The mirror's cost is objects and
   triangles, not pixels: a tenth of its resolution saved nothing measurable.
   Kept out of it: the meadow's grass (600 000 triangles), beach decals, reef
   shells, lantern ironwork, the rudder, the whipstaff, the depth-only spray
   copies. **Tried and put back:** the spindrift and the lantern glass. Both
   show; without them five shots failed and the night shot moved by 0.93.
6. **Hidden casters.** The whipstaff, below decks, and the reef's crate and
   chest from above the water. **Tried and put back:** the rudder, whose shadow
   shows in the underwater shot (0.148 against a limit of 0.1).
7. **Lanterns.** Each lantern's rope, cap and base are one mesh; the stern
   windows are additive, so one draw. Eight draws, nothing measurable.
8. **No repeated transform walks** (`withWorldMatricesHeld`). The mirror and
   the cascades render from inside the scene pass, which has just run
   `scene.updateMatrixWorld()`; each ran it again.
9. **Half the butterfly passes** (`butterflyBothPassNode`). The transform along
   x and the transform along y commute, so stage k of each runs in one pass
   with four reads instead of two: 8 passes at 256 where there were 16, and
   n + 1 where log2(N) is odd. The result differs in float rounding only. That
   is enough to move two things that amplify it: the sailcloth settles in a
   slightly different pose and the drops on the lens land elsewhere. The
   baselines for `ship-underway` and `waterline` were recaptured (0.18 and 0.16
   mean Delta E against the old ones, the same picture otherwise), and
   `boat-chase` now sits 0.22 from its baseline, inside its threshold.

**Where the frame goes now** (High, by subtraction with
`scripts/perf-subtract.mjs`, and a CPU profile with `scripts/perf-cpu.mjs`):

- GPU, of about 3.9 ms: the ocean surface 0.9, the clouds 0.75, the seabed
  0.44, the ship 0.34, the island's grass about 0.2 and canopy 0.2, props 0.3,
  the sun's shadows 0.27, spindrift about 0.2. Halving the render scale takes
  1.4 ms off, so about half the frame is fill.
- CPU, of about 4.8 ms: the scene pass about 2.0 (some 135 draws at 10 to 13
  microseconds each, which is three's cost per object), the mirror 0.9, the
  sun's shadows 0.7, the wave transform 0.35, the rest of the update 0.8.

What is left is in three's cost per draw and per pass, and in shading that
cannot change without the picture changing. The candidates not taken: the
cloud march below full resolution (0.34 ms of GPU at half, and the reason it is
at full is the crispness of an eroded edge), the mirror on alternate frames
(its image is looked up by screen position, so a stale one judders as the
camera turns), and the crew out of the mirror (five draws).

**The harness, fixed on the way.**

- `scripts/benchmark.mjs` and `scripts/profile-frame.mjs` warmed up with
  `renderer.compileAsync(scene, camera)`, a compile straight to the canvas,
  which the app never does. On r187 that left the canvas depth buffer
  destroyed and rebuilt every frame and a submit rejected on each sampled
  frame ("Destroyed texture depthBuffer used in a submit"): about 1 750 console
  errors a run and an UNVERIFIED verdict. Removed; the numbers did not move.
- `OCEAN_DIST=<folder>` makes both scripts measure a build made elsewhere and
  leave `dist/` alone.
- At Low and Medium the app itself raised one validation error on every boot:
  the prewarm's `compileAsync` ran the depth nodes' `updateBefore`, which
  copied the canvas depth buffer at the scene pass's size. No copy is made
  during a compile now, and the benchmark's Low gate passes.
- `scripts/profile-frame.mjs` no longer runs: its cases were written against
  an older scene. `scripts/perf-subtract.mjs` does its job for now.
- GPU time per pass, from three's timestamp queries, is not usable for
  attribution on this stack: a small pass that follows a heavy one is charged
  part of it (a 128 x 128 pass read 0.25 ms). The frame total is sound.

**Automatic quality.** Checked at the same time, because a drop is a rebuild
the viewer sits through. `AdaptiveQuality` used to read a smoothed frame rate
and drop after 2.5 s of the simulation clock under 41 fps. One slow frame could
not trip that, but a frame rate held under 41 by something other than load
could, and would walk the tier down to Low: a 30 Hz display, a remote desktop,
the browser's energy saver. It now needs all of: under 41 fps (smoothed over
about ten frames), the device busy for at least half of each frame (the cost
the pacer already measures), for four seconds of wall clock without a break,
each frame counting for at most 100 ms; and it waits 6 s after the first frame,
10 s after a drop and 2 s after frames stop and start again. Unit-tested
(`tests/adaptiveQuality.spec.ts`) and checked live at High: one 600 ms frame
and ten 200 ms frames in a row left the tier alone; a sustained load dropped
High to Medium at 4.0 s and no further once it was removed.

### 2026-11-17: High at the display rate, and a shorter boot

Not from `npm run bench`. The harness unhooks rAF from the display, and this
work is about holding the display rate, so it was measured with the pacer on, in
a plain headed Chrome on a ~100 Hz display. The page was the production build at
High, 1920x1080, sailing from the start position, with nothing else on the GPU.
The measuring scripts (CDP probes, CPU profiles, GPU-process thread sampling)
were scratch work and are not checked in. The method is below, so the numbers
can be reproduced.

**Before**, High held a divisor of 2: ~50 fps, 24-29 ms from callback to
finished frame. The GPU was not the limit (6.7 ms a frame). The page spent
~10 ms encoding the render and 1.6 ms in updates. Chrome's GPU-process main
thread spent 11.3-15 ms a frame replaying the commands, and that was the
bottleneck. Dawn's costs per command were measured by duplicating commands in
a live frame:

| Command | Cost in the GPU process |
|---|---|
| `queue.writeBuffer` | ~0.8 µs |
| `setBindGroup` | ~1.4 µs |
| empty `queue.submit` | ~6.5 µs |
| an extra render pass | ~1.8 µs |
| a three render call (encoder + submit) | ~35-40 µs |
| vertex-buffer binds, draws | negligible |

Five changes, none of which changes a pixel:

- **Scene-wide uniforms in the render group** (`core/uniform.ts`). By default,
  TSL's `uniform()` puts a value in the per-object group. That group is
  compared and written for every draw that uses it. The app's own uniforms
  (time, wind, sun, exposure and so on) now go to the render group, which is
  written once per render. A uniform with an `onObjectUpdate` callback still
  goes back to the object group.
- **Shadow passes stop re-validating every caster** (`installSubmissionFixes`).
  Alternating alpha-tested and opaque casters moved the shared override
  material's `version` on nearly every draw. That rebuilt 163 material cache
  keys a frame, about 1.5 ms.
- **One `writeBuffer` per uniform buffer**, not one per contiguous run of
  changed values. That was ~2,150 writes a frame.
- **Framebuffer copies: 27 a frame down to 4** (`installFramebufferCopyFixes`).
  The ocean's 18 shared-colour and 5 depth nodes each copied the same pixels
  and split the scene pass for each copy.
- **Submits batched** (`installSubmitBatching`). three submits once per render
  call. Command buffers are now held until a microtask and submitted together.
  The batch is flushed first by any `writeBuffer` to a buffer a held command
  buffer references, and by anything else whose order matters. This took the
  GPU-process main thread from 11.3 to 5.4 ms a frame on its own. `?batch=0`
  turns it off for A/B runs.

The pacer then had to stop refusing the faster rate (`FramePacer.gpuLatency`).
It gated its probes on the time from callback to finished frame, about 17 ms
here. But encoding frame N+1 overlaps the GPU working on frame N, so the device
holds a rate when *each stage* fits the interval. The gate is now the slower of
the two stages.

**After** (three runs):

| | Before | After |
|---|---|---|
| Drawn fps | ~50 (divisor 2) | **98.5-98.6** (divisor 1) |
| Frame interval p50 / p99 | ~20 / 21.7 ms (scoreboard, 2026-11-16) | **10.0 / 10.5-10.6 ms** |
| Render call, awaited | ~10 ms | 5.4 ms |
| Slower stage (pacer's input) | — | 10.5 ms |
| GPU-process main thread | 11.3-15 ms/frame | ~7.4 ms/frame |

**Boot.** Measured from clicking *Set sail* to `isReady()`, on a warm profile.
The click starts with ~2 s of island generation. Then the first view's node
builds run on the page's main thread (~5 s). The content's builds follow
(~8 s), and then pipeline compiles finish. Chrome's GPU process compiles
pipelines with FXC on a worker pool. For the whole content phase it had 15
threads at full load, so the boot was bound by the total compile work. At High
it made 447 node builds, 295 asynchronous pipelines and 423 shader modules.

Half the node builds were one material built again for each `InstancedMesh`,
because r185 puts the mesh's uuid in the material cache key. Their WGSL differed
only in buffer names, which r185 takes from global node ids
(`NodeBuffer_76015`). `installStableBufferNames` names them by first use within
the build, so the copies come out identical. three then shares a shader module
and a pipeline across them. The result: **295 → 235 pipelines and 423 → 362
modules, and click to ready went from 24.4 to 22.9 s** (two runs each, same
session). The node builds themselves are still made per mesh, because each one
binds its own mesh's matrix buffer.

**Still open**, measured but not fixed:

- **At 120 Hz the target is likely still missed.** The slower stage (~10 ms)
  is over that display's 8.3 ms interval (9.6 ms with the probe's margin).
  High would hold 60 there. The larger risk: if the pacer settles on a divisor
  of 3 (40 fps), `AdaptiveQuality` reads that as under 75% of its 55 fps
  target, drops a tier, and recompiles for about a minute at 1-2 fps.
- **Periodic stalls.** A 60-250 ms gap about every 15 s, with no long task on
  the page's main thread or the GPU process's. It is also present before these
  changes and absent on a trivial WebGPU page. Probably GPU-side; not
  diagnosed.
- **Hitches just after ready.** About 30 pipelines are first requested in the
  0.4 s after ready, which the prewarm does not reach: the ocean as a live frame
  draws it, the lantern glow, the spray and splash sprites, bloom's high-pass
  and composite, and one RTT. They build asynchronously, but their node builds
  cost frames of 50-80 ms.
- **The overlay lifts before the scene moves.** The boot overlay's hide timer
  is starved while the content builds run. It fires once they finish, and the
  loop then stays held for the 3-5 s the remaining compiles take. The user sees
  a still frame for that long before the first moving one.

### 2026-11-15, after the canopy rebuild, the foam rebuild and frame pacing

**Re-measured 2026-11-15, after the canopy rebuild, the foam rebuild and frame
pacing.** Full matrix in `bench-results/bench-2026-11-15T09-31-29-528Z.json`.
All nine configurations pass, both gates and the Low boot budget included. Every
GPU p50 but WebGPU Low's (below) is within 0.2 ms of the 2026-11-14 run (`bench-2026-11-14T08-47-41-374Z.json`),
so the leaf-shelled canopy and the feathered foam pattern cost nothing this
harness can resolve; the CPU-throttled WebGL Low is 2.8 ms cheaper.

The page is opened with `?pacing=0`. The flags above unhook rAF from the
display, so there is no refresh for the app's frame pacer (`core/FramePacer`)
to divide: run paced, it divided whatever cadence the GPU gave it, and the
delivered rate at WebGPU Low read 59 against 208 unpaced, with the GPU times
unchanged. A real browser ties rAF to the display, and there the pacer is on.

| Configuration | GPU p50 (2026-08-05) | now | p99 | delivered | TTFF | peak memory | Verdict |
|---|---|---|---|---|---|---|---|
| WebGPU · Low | 0.79 | 0.82 | 2.57 | 208 | 9.9 s | 208 MiB | PASS |
| WebGPU · Medium | 5.25 | 4.24 | 4.30 | 135 | 9.8 s | 264 MiB | PASS |
| **WebGPU · High** | **8.07** | **7.18** | **7.27** | **96** | 9.7 s | 403 MiB | **PASS** (16.7) |
| WebGPU · High, underwater | — | 13.88 | 14.05 | 61 | 9.8 s | 403 MiB | PASS |
| WebGPU · Ultra | 12.90 | 12.10 | 12.30 | 64 | 9.8 s | 410 MiB | PASS |
| WebGPU · Max | 13.55 | 12.68 | 12.86 | 62 | 10.2 s | 511 MiB | PASS |
| **WebGL2 · Low** | **7.59** | **5.00** | **5.99** | **213** | **6.5 s** | **211 MiB** | **PASS** (33.3; 15 s; 256 MiB) |
| **WebGL2 · Low, CPU 4x** | — | **11.84** | **15.74** | **67** | **8.3 s** | **211 MiB** | **PASS** |
| WebGL2 · High | 21.85 | 22.24 | 25.05 | 42 | 7.0 s | 356 MiB | PASS |

Low WebGPU read 2.08 ms on 2026-11-14 and 0.82 ms now; its p95 and p99 are
still near 2.1–2.6 ms, so the median moved across a bimodal distribution rather
than the tier getting cheaper; the 2026-11-14 median most likely sat in the upper mode. It
is the cheapest row by a factor of five either way.

**The WebGL2 boot was the finding of this run.** Its first frame had gone from
7.9 s on main to 65 s here. Program by program with `KHR_parallel_shader_compile`
disabled, the fallback linked 112 programs for 90 s of FXC time, and 66 of
them were the same reef-life shader. Three causes, all fixed:

- **The instance count was in the shader.** r185 puts an `InstancedMesh`'s
  matrices in a `mat4[count]` uniform block when they fit, so each distinct
  count is a new program. The fallback now reports no uniform-block space and
  takes three's instanced-attribute path (`installWebGLFixes`).
- **So was a node id.** `CSMShadowNode` rebuilt its graph for every material,
  and GLSL names the cascade array `NodeBuffer_<id>`. The graph is now built
  once and shared (`SunShadows`).
- **Programs linked one at a time.** The fallback skipped the asynchronous
  pipeline path, so `compileAsync` awaited each link and the first live frame
  linked what the prewarm missed, serially. With the WebGPU path enabled, the
  prewarm went from ~12.5 s to ~1.5 s on ANGLE's worker pool.

Together: 86 programs and 29 s of link work, all of it in parallel. First
frame 65 s → 6.3 s, fully ready 35 s → 7.1 s, with no frame over 100 ms in
the 25 s after.

The boot budget now gates **bytes**, not resource counts; see `LOW_BOOT_BUDGET`
in the script for the reasoning. This branch peaks at 210 MiB at WebGL2 Low.
Main peaks at 452 MiB, 425 MiB of it textures. The likely source of the
difference is that this branch ships the hero models as KTX2 where main
decodes JPEGs (the dressing is KTX2 on both, see Memory); it has not been
measured separately. The count gate (100 textures) had never actually
run: the harness never passed the boot measurements to the verdict, and its
observer stopped before the content's first draw. Both are fixed.

### 2026-08-05, after the wave transform was folded into one atlas
Full matrix in `bench-results/bench-2026-08-05T18-34-24-131Z.json`. All seven
configurations pass; both gates pass.

| Configuration | GPU p50, after fidelity | now | delivered | budget | Verdict |
|---|---|---|---|---|---|
| WebGPU · Low | 0.86 | 0.79 | 250 | — | PASS |
| WebGPU · Medium | 5.35 | 5.25 | 120 | — | PASS |
| **WebGPU · High** | **8.26** | **8.07** | **84** | **16.7** | **PASS** |
| WebGPU · Ultra | 13.01 | 12.90 | 52 | — | PASS |
| WebGPU · Max | 13.72 | 13.55 | 48 | — | PASS |
| **WebGL2 · Low** | **7.36** | **7.59** | **143** | **33.3** | **PASS** |
| WebGL2 · High | 20.80 | 21.85 | 35 | — | PASS |

**Read the GPU column and the delivered column against each other.** The
transform fold removed about 4.2 ms from the delivered frame at High — 61 FPS to
81, measured with `scripts/profile-frame.mjs` — and moved GPU p50 by 0.19 ms.
Both numbers are correct. The saving was CPU-side command submission, which
timestamp queries do not measure, and it is the clearest demonstration available
of what this gate can and cannot see. See the Cost model section.

The earlier fidelity-work comparison is kept below.

**The frame roughly tripled at High and both gates still pass, with more than
twice the budget to spare.** What it bought: a cloud layer
wrapped on a sphere so the deck converges instead of stopping in a band, one
per-channel aerial perspective on every material instead of three unrelated
treatments and one absence, the island's own shadow marched against the
heightfield because a +/-260 m shadow box cannot cover a kilometre of island,
crepuscular rays through the deck, and caustics and contact darkening on
everything standing on the ground.

Three things about the shape of that increase are worth recording.

**It is march-dominated and therefore tier-controlled.** Nearly all of it is in
raymarches whose step counts are `QualitySettings` fields — `cloudSteps`,
`fogSteps`, `godRaySteps` and the new `terrainShadowSteps`. Every tier above Low
was re-cut on measurement rather than on taste, and the cuts are invisible for a
structural reason: the cloud and fog marches integrate their segments
energy-conservingly, so the step count refines a cloud's *texture* and not how
much of it there is.

High first measured 10.46 ms and passed this gate while failing the *suite's*
own delivered-frame-rate assertion at 49.6 FPS against a 55 floor; 24 steps to 18
on both marches took it to 8.26 and satisfied both. Max first measured 20.48,
then 16.67 after one cut — which is inside a 16.7 budget by 0.2% and therefore
not inside it at all — and now sits at 13.72. `gallery-jitter` was re-run after
each cut: the far-field shimmer figures moved by under 0.5% and stayed inside
their ceilings, which is the evidence that fewer steps cost texture rather than
stability.

**The terrain shadow is gated on elevation**, so the sea — which is most of a
typical frame — never enters that loop. The numbers above are the canonical wide
shot, which contains no island at all; the island frames measure within noise of
them.

**Two of the reductions were free.** The cloud shadow's samples are a TSL `Loop`
rather than an unrolled JavaScript one, so `mx_fractal_noise_float` is emitted
once per consumer instead of three times, and the coarse evaluation the shadows
and shafts share runs two octaves instead of four. Together they took the first
frame of the scene from *minutes* back to seconds on the FXC shader path the
Playwright configuration forces, and 10.85 ms to 9.85 at High.

The earlier run is kept below for comparison.

### Before the fidelity work

Run of 2026-08-02, `bench-results/reference.json`. Full scene — ocean, sky,
volumetric clouds, seafloor, ship, island, buoys, barrels, wake and the
post-processing chain — at 1600 × 900, DPR 1, `skyPro` preset, camera pinned to
the canonical wide shot, world reset to simulation time 0. **600 samples per
configuration** after 150 discarded warm-up frames.

GPU frame time, milliseconds, from timestamp queries:

| Configuration | p50 | p90 | p95 | p99 | min | max | implied FPS | Verdict |
|---|---|---|---|---|---|---|---|---|
| WebGPU · Low | 0.49 | 0.50 | 0.50 | 0.69 | 0.46 | 1.04 | 2028 | PASS |
| WebGPU · Medium | 1.66 | 1.70 | 1.72 | 2.05 | 1.60 | 2.24 | 601 | PASS |
| **WebGPU · High** | **3.09** | 3.30 | 3.35 | 3.45 | 2.80 | 3.62 | **323** | **PASS** |
| WebGPU · Ultra | 4.21 | 4.46 | 4.53 | 4.65 | 4.01 | 4.79 | 237 | PASS |
| WebGPU · Max | 6.52 | 6.83 | 6.89 | 7.03 | 6.14 | 7.27 | 153 | PASS |
| **WebGL2 · Low** | **2.06** | 2.92 | 3.37 | 4.54 | 1.53 | 7.42 | **487** | **PASS** |
| WebGL2 · High | 5.39 | 5.99 | 6.20 | 6.88 | 4.48 | 8.11 | 185 | PASS |

Scene cost and CPU frame time for the same runs:

| Configuration | CPU p50 | CPU p99 | Draw calls | Triangles | Textures | Render targets | Programs |
|---|---|---|---|---|---|---|---|
| WebGPU · Low | 1.8 | 4.1 | 77 | 3 377 009 | 73 | 18 | 54 |
| WebGPU · Medium | 2.4 | 6.3 | 111 | 3 503 793 | 80 | 24 | 58 |
| WebGPU · High | 3.1 | 9.0 | 155 | 3 799 069 | 81 | 30 | 62 |
| WebGPU · Ultra | 3.1 | 9.0 | 155 | 4 167 965 | 81 | 30 | 62 |
| WebGPU · Max | 3.4 | 10.2 | 167 | 4 856 489 | 87 | 30 | 62 |
| WebGL2 · Low | 1.2 | 1.9 | 65 | 1 807 467 | 65 | 17 | 52 |
| WebGL2 · High | 1.9 | 2.9 | 142 | 2 227 687 | 73 | 29 | 60 |

Reading these:

- **Both gates pass with a wide margin on this GPU.** WebGPU High costs 3.09 ms
  against a 16.7 ms budget — 5.4x headroom; WebGL2 Low costs 2.06 ms against
  33.3 ms. That is an RTX 5090 result and it should be read as one; see
  Limitations.
- **GPU time tracks the tier cleanly**, 0.49 -> 1.66 -> 3.09 -> 4.21 -> 6.52 ms.
  Whatever else is true of these numbers, they are responding to the thing the
  quality tiers change. Low no longer sits near zero because it now casts a
  shadow like every other tier — see `QualitySettings.shadowMapSize` for why that
  is not negotiable.
- **No console errors during any sample.** That is a gate, not an observation: the
  harness reports UNVERIFIED if a configuration logs one, which is how the
  `Destroyed texture used in a submit` validation error was caught.
- **The gate is on p50 alone, and that is weaker than a shipping frame-time
  target.** A median says nothing about hitches, and a game that misses vsync
  every twentieth frame is a game that stutters. The p95 and p99 columns are
  recorded and are within 15% of the median on every WebGPU configuration here,
  but nothing *fails* on them. An independent review raised this and it is a real
  gap in the harness rather than in the renderer.
- **CPU frame time tracks the tier weakly**, 1.8 to 3.4 ms from Low to Max. Most
  of it is JS update work and command submission, both roughly tier-independent;
  the rise is the extra render passes. The renderer is GPU-bound at High and above.
- **Ultra costs 35% more GPU time than High** for 10% more triangles and the same
  draw-call count. Ultra raises raymarch step counts, not texture resolution.
- **Max is the only tier that moves memory materially** — 512² FFT cascades, for
  12 extra render passes.
- **WebGL2 High costs 86% more GPU time than WebGPU High** for an identical
  scene, and WebGL2 Low costs 2.6× WebGPU Low. That is the price of the fallback
  path, measured rather than assumed.
- **Distributions are tight on WebGPU and looser on WebGL2.** p99/p50 sits between
  1.1 and 1.7 on every WebGPU tier — no compilation stalls, no periodic hitch. The
  WebGL2 configurations reach 2.8, with single frames at 10.8 ms against a 2.6 ms
  median; that is one frame in several hundred and it is disclosed rather than
  smoothed away.

### Cross-check: the number responds to workload

A GPU timer that does not move with load is not measuring anything. WebGPU High
re-run at DPR 2 (3200 × 1800, four times the pixels) costs **6.76 ms** against
2.89 ms — 2.3×, which is what a mix of resolution-independent FFT passes and
fragment-bound surface shading should do. Both figures are from the run that
established the ratio; the tier's absolute cost has moved since (3.09 ms), and
the ratio is the claim here, not the absolute. Reproduce with:

```bash
node scripts/benchmark.mjs --only webgpu-high --dpr 2
```

## Results file format

Every run writes `bench-results/bench-<timestamp>.json` and overwrites
`bench-results/latest.json`. Schema id `web-ocean-3d/bench@1`:

| Field | Contents |
|---|---|
| `schema`, `startedAt`, `finishedAt`, `durationMs`, `command`, `argv` | provenance |
| `host` | platform, OS release, CPU model, RAM, Node version |
| `osReportedGpus` | `Win32_VideoController` — driver versions, for provenance only |
| `browser` | channel, version, the full flag list, headless flag |
| `budgets`, `minSamples` | the thresholds this run was judged against |
| `configurations[]` | one entry per `{backend, tier, resolution, dpr}` |
| `summary` | pass/fail/unverified counts, plus per-gate verdicts and reasons |

Each `configurations[]` entry carries:

| Field | Contents |
|---|---|
| `id`, `gate`, `requestedBackend`, `backend`, `tier`, `preset`, `camera` | what was measured |
| `resolution`, `drawingBuffer` | requested size and DPR, and the buffer actually allocated |
| `adapter` | browser-reported WebGPU adapter, its feature list, `isFallbackAdapter` |
| `gl` | WebGL2 version and the unmasked ANGLE vendor/renderer strings |
| `focus` | `hasFocus`, `visibilityState` at the end of the sample |
| `gpuTiming` | whether timestamps were enabled, by what route, and any error |
| `pacing` | classification, delivered FPS, rAF interval percentiles, the app loop's own `frameMs` |
| `sampling` | warm-up and sample counts, GPU-sample count and misses, truncation flag, step size, deterministic-reset outcome, simulation time at the end of the sample |
| `cpuFrameMs`, `gpuFrameMs` | `{samples, min, p50, p90, p95, p99, max, mean}` |
| `fps` | delivered p50, and the rates implied by GPU p50 and CPU p50 |
| `render`, `memory` | `renderer.info.render` and `renderer.info.memory` snapshots |
| `boot` | time to first frame, and peak renderer memory from page load to the end of the settle: `geometries`, `textures`, `bytes` (`info.memory.total`), `textureBytes` |
| `sceneContent` | whether the ship loaded, and which binaries the harness served |
| `budget`, `verdict`, `reasons`, `consoleErrors` | the judgement and its evidence |

The schema is additive-stable: fields may be added at `@1`, and anything that
changes or removes a field bumps the id.

## Cost model

Where the frame goes, and the lever for each.

| Stage | Scales with | Lever |
|---|---|---|
| Spectrum evolution | `fftSize²` × cascades × 2 | quality tier |
| IFFT butterfly passes | `fftSize²` × log2(fftSize) × 2 × cascades × 2 | quality tier |
| Output assembly + mipmaps | `fftSize²` × cascades × 2 | quality tier |
| Ocean surface raster | `meshSegments` × (2·`meshRings` + 1) triangles, then fragment cost at screen resolution | quality tier, pixel-ratio slider |
| Sky dome | one fullscreen-ish pass | negligible |
| Volumetric clouds | `cloudSteps` × screen pixels | `cloudSteps`, 0 disables |
| Underwater god rays | `godRaySteps` × screen pixels, only when submerged | `godRaySteps`, 0 disables |

**This section used to claim the transform was deliberately not the bottleneck,
and that surface shading was. That was wrong, and the way it was wrong is worth
keeping.**

The reasoning was that 96 fullscreen passes at 256² are only ~6.3 M trivial
fragment invocations against ~2 M for one 1080p raster pass — true, and
irrelevant, because the cost was never the fragments. Measured against delivered
frame rate, the transform was **40% of the frame**, and stubbing it dropped the
frame's *CPU* update phase from 5.7 ms to 0.22. A fullscreen render pass costs
about **53 microseconds** here near enough regardless of what it draws — measured
directly by adding 96 empty ones — and the transform was issuing 108 of them.
Command submission, not arithmetic.

Folding the spectra pairs and then the cascades into one atlas took it to 23
passes and the transform from 7.24 ms to 2.55 ms. Re-measure any of this with
`node scripts/profile-frame.mjs`, which attributes the delivered frame stage by
stage rather than gating on GPU p50.

**Why this document could assert the opposite for so long is the important
part**, and it is a property of the gate above rather than of anyone's judgement:
`npm run bench` measures GPU p50 through timestamp queries, and GPU timestamps
cannot see CPU-side command submission. The same work that took the delivered
frame from ~61 to ~81 FPS moved GPU p50 at High from 8.26 ms to 8.07 — under
2%. A metric that cannot observe a cost will report that the cost is not there.

Two consequences worth carrying:

- **Pass count is a first-class cost here**, alongside texels and triangles. The
  table above prices work per pixel and per triangle and says nothing about how
  many passes it takes; at 53 microseconds each that omission was worth 5.7 ms.
- **GPU p50 is the right budget gate and the wrong diagnostic.** It is a real,
  reproducible property of the renderer and the brief's budgets are stated in
  it. It just cannot tell you where a frame goes. That is what
  `scripts/profile-frame.mjs` is for.

## Adaptive quality

`AdaptiveQuality` steps the tier down when the smoothed rate stays below 75 % of
target for 2.5 s. It is deliberately one-way within a session, with a 6 s
debounce and a 4 s startup grace period: oscillating between tiers is more
distracting than running one notch below optimal, and the first seconds of a
session are dominated by compilation rather than by steady-state cost.

It also has to be neutralised during a benchmark. It reads `loop.stats.fps`,
which `Loop.step` never writes, so on a slow configuration it would otherwise act
on a stale rAF-era number and change the tier *during* the sample. The harness
pins the value it watches, and then asserts afterwards that the tier did not
move — a drifted tier is an UNVERIFIED result, not a quiet one.

## Memory

The suite cycles Low↔High and asserts the renderer's texture and geometry counts
have not grown beyond a small allowance. A tier change disposes the previous FFT
targets and ocean geometry before allocating replacements.

The surface **material** is deliberately not among them. It used to be rebuilt on
every tier change, which both leaked node-graph textures across cycles and was the
mechanism behind defect D1 (a rebuild dropped the seafloor depth node). Cascade
count is now re-pointed in place through `OceanMaterial.setCascades()` against a
fixed `MAX_CASCADES = 3` binding, so one material instance survives the whole
session. An earlier revision of this section still claimed the material was
disposed per tier; that was corrected after an independent review.

The benchmark records `renderer.info.memory` per configuration, so the tier cost
is visible directly: 273 MB of textures at Low, 323 MB at High and Ultra, 476 MB
at Max, with render-target counts of 14/27/27 respectively.

### The low-end pass, and where the memory actually went

Those figures predate the low-end work and are kept because the benchmark rows
above are from the same run. What that work found is that **the numbers this
document was watching were not the ones killing the tab.**

The dressing GLBs decoded to **762.3 MiB of texture memory** — roughly twice the
entire render-target budget, and invisible to every measurement here, because
`renderer.info.memory` counts textures the renderer has been handed and this
document only ever quoted its totals per tier. Three causes: every LOD level
shipped its own copy of the same maps (183 MiB of pure duplication, since the
loader caches per URL), one beached pinnace carried 112 MiB, and nothing was
GPU-compressed at all.

Measured after: **79.9 MiB**, an 89.5% reduction, via KTX2 — ETC1S for colour and
ARM, UASTC for normals and alpha-cutout leaf atlases, with rate-distortion
optimisation and half-resolution normals. Dressing download went 33.7 → 40.7 MiB,
which is the price of the UASTC half.

Two lessons worth keeping, both of which cost a real defect first:

- **`isPerceptual` tunes the codec; `isSetKTX2SRGBTransferFunc` tells the loader.**
  Setting only the first shipped 239 of 367 textures tagged sRGB, including every
  normal and metallic-roughness map. De-gamma'ing a roughness of 0.5 gives 0.21,
  and the beached pinnace rendered pale and over-glossy for it.
  `scripts/fix-ktx2-transfer.mjs --check` now gates this.
- **A regenerated visual baseline cannot catch a rendering regression**, because
  it is compared against the render that produced it. All 20 visual tests passed
  against the broken textures. The pale hull was only visible by diffing the new
  baseline against `main`.

Render targets are now tier-dependent too — wake 13.5 MiB at Low and Medium
against 24.0 above, caustics 256²/512²/768², and Low drops MSAA and caps DPR at
1 before the framebuffer is allocated. The environment cube is **not**: shrinking
it was measured and rejected, see `QualitySettings.environmentResolution`.

## What these numbers do not establish

An independent review pushed back on the framing, correctly. "All seven
configurations pass on this RTX 5090 under the harness's p50 rules" is supported
by the checked artifact. "The renderer meets a broadly meaningful AAA performance
target" is not, for four reasons:

- **The gate is a median.** Nothing fails on p95 or p99, and a game that misses
  vsync every twentieth frame stutters however good its median is. The
  percentiles are recorded and are within 15% of the median on every WebGPU
  configuration here, but that is an observation, not a gate.
- **One preset, one camera.** The matrix measures `skyPro` from the canonical
  wide shot. It does not establish the cost of the storm, of near-water SSR at a
  grazing angle, or of the underwater state — all of which are more expensive.
- **The underwater pass is never sampled.** It is behind a uniform branch that is
  false above the surface, so the benchmark measures the compare and nothing else.
- **One GPU, and a large one.** High already issues 155 draws over 3.8 million
  triangles with 30 render targets. Full-resolution single-ray SSR, the cloud
  march, the fog march and the shaft march are all serial fragment loops; they
  will scale far worse on a laptop or integrated GPU, at high DPR, or on a
  tile-based mobile renderer than the headroom here suggests.

## Limitations, and what is still unverified

- **One machine.** Every number here is from an RTX 5090, and a 2.89 ms frame on
  that part says very little about a laptop iGPU. The budgets are *met*, not
  *stressed*: nothing in this run establishes where the tiers stop working. The
  harness is the deliverable; the numbers describe one host, which is why the
  hardware block above is as detailed as it is.
- **Adaptive downgrade is untested against real pressure.** On this GPU no tier
  gets close to the trigger, so the downgrade path has not been exercised by a
  genuine frame-rate drop here.
- **Delivered frame rate is not presented frame rate.** With vsync disabled the
  rAF callback rate runs ahead of what the compositor puts on screen — the WebGL2
  Low run reports 1000 delivered FPS, which is a main-thread spin rate, not 1000
  images. Only GPU frame time is treated as a measurement.
- **Timestamp quantisation is disabled by a flag.** These GPU numbers are not
  what a stock browser would report; a stock browser would report the same frames
  rounded to 100 µs per pass.
- **The camera is never underwater.** The canonical shot is above the waterline,
  so the submerged branch — god rays at full strength, the underwater particle
  system, which is skipped entirely while dry — contributes nothing to these
  numbers. A submerged benchmark configuration is the obvious next addition.
- **Environment quirk on this host: `.bin` responses return HTTP 204.** In a
  *headed* browser on this machine, every glTF binary payload served by
  `vite preview` arrives as `204, zero bytes`, while `curl` and the same Chrome
  build in headless mode both receive the full 200 from the same server and port.
  Something outside the browser is eating `application/octet-stream` on visible
  sessions. Untreated, `GLTFLoader` reports `Failed to load buffer`, the ship and
  props are absent, and the benchmark measures an empty ocean — 113 draw calls
  and 394 493 triangles at High instead of 138 and 633 229, which is a 38 % lie
  in the direction nobody notices. `scripts/benchmark.mjs` therefore fulfils those requests from
  `dist/` itself, records every interception in `sceneContent.binariesServedFromDisk`,
  and reports UNVERIFIED if the ship is missing anyway. This is an observation
  about this host, not a defect in the application — but it is exactly the kind
  of thing that turns a benchmark into fiction, so it is written down.

## Running the test suite: hardware matters

The Playwright suite is only fully meaningful on a machine with a real GPU
adapter.

Observed on a software-only runner (Playwright's bundled Chromium, no WebGPU
adapter, so WebGL2 backed by a software rasteriser): **9 failed, 6 passed,
1 skipped in 21.7 minutes**. Nearly every failure was a timeout, not an
assertion — a single frame of this scene through software rasterisation can take
tens of seconds, so anything that waits on rendering runs out of time.

Which is which:

- **Trustworthy anywhere** — the sea-state assertions (read straight off the GPU
  via `readRenderTargetPixelsAsync`), the WebGL fallback boot, and typecheck.
  These passed on the software runner.
- **Needs a GPU** — everything that waits on presented frames: screenshots,
  preset comparison and the interaction tests that poll after a state change.
  These time out on a software runner and their failure says nothing about the
  code.

Timeouts are set to 240 s to give a software runner a chance, but the honest
recommendation is to run on GPU hardware and treat a software-runner result as
inconclusive rather than as a regression. Frame budgets are not the suite's job
at all — that is `npm run bench`.
