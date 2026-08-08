/**
 * Where the frame actually goes, by subtraction.
 *
 * `npm run bench` answers "what does a frame cost". It cannot answer "what is
 * the frame *spending it on*", and every attribution in `docs/PERFORMANCE.md`
 * before this script existed was read off the source rather than measured. This
 * measures it: pin the world, render it, then render it again with exactly one
 * subsystem removed, and attribute the difference.
 *
 * Subtraction is not free of assumptions and the important one is stated here
 * rather than discovered later: **the deltas do not have to sum to the frame.**
 * Removing an object removes its raster cost and also whatever it was occluding
 * for — hide the island and the sea behind it starts shading. A negative delta
 * is therefore a real result and not a bug, and the total of the parts can
 * exceed or fall short of the whole. What each number honestly means is "what
 * this frame would cost if this system were not in it", which is the number a
 * cut is actually judged on.
 *
 * Everything else is borrowed from `scripts/benchmark.mjs`, deliberately and
 * identically: the same Chrome flags, the same `.bin` interception for this
 * host's 204 quirk, the same deterministic reset, the same
 * `resolveTimestampsAsync` route to real GPU time. A number produced here is
 * comparable with one produced there.
 *
 * Usage:
 *   node scripts/profile-frame.mjs                       # High, canonical wide shot
 *   node scripts/profile-frame.mjs --tier ultra
 *   node scripts/profile-frame.mjs --camera waterline    # the grazing-angle shot
 *   node scripts/profile-frame.mjs --frames 200 --repeats 3
 */

import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from '@playwright/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RESULTS_DIR = join(ROOT, 'bench-results');
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

/** Identical to the benchmark's, so the two harnesses measure the same thing. */
const CHROME_FLAGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--disable-frame-rate-limit',
  '--disable-gpu-vsync',
  '--enable-dawn-features=allow_unsafe_apis',
  '--disable-dawn-features=timestamp_quantization',
  '--window-position=0,0',
  '--autoplay-policy=no-user-gesture-required',
];

/**
 * Cameras worth profiling, and why each one is here.
 *
 * `wide` is the benchmark's canonical shot, so this script's baseline can be
 * checked against `npm run bench`. The other two are the views
 * `docs/PERFORMANCE.md` names as unmeasured and more expensive: a sea-level
 * camera puts the whole ocean at a grazing angle, and the island shot is the
 * only one that enters the terrain-shadow march.
 */
const CAMERAS = {
  wide: { position: [0, 14, 48], target: [0, 2, 0] },
  waterline: { position: [-46, 9, 44], target: [-90, 6, 8] },
  island: { position: [-120, 26, 150], target: [-260, 30, -40] },
};

function parseArgs(argv) {
  const options = {
    tier: 'high',
    preset: 'skyPro',
    camera: 'wide',
    width: 1600,
    height: 900,
    dpr: 1,
    warmup: 40,
    frames: 150,
    /** Baseline is re-measured this many times, interleaved, to expose drift. */
    repeats: 2,
    settleSeconds: 3,
    only: null,
    /** 'timestamp' (the benchmark's metric) or 'fence' (queue.onSubmittedWorkDone wall clock). */
    metric: 'timestamp',
  };
  for (let i = 0; i < argv.length; i++) {
    const next = () => argv[++i];
    switch (argv[i]) {
      case '--tier': options.tier = next(); break;
      case '--preset': options.preset = next(); break;
      case '--camera': options.camera = next(); break;
      case '--width': options.width = Number(next()); break;
      case '--height': options.height = Number(next()); break;
      case '--dpr': options.dpr = Number(next()); break;
      case '--warmup': options.warmup = Number(next()); break;
      case '--frames': options.frames = Number(next()); break;
      case '--repeats': options.repeats = Number(next()); break;
      case '--only': options.only = new Set(next().split(',').map((s) => s.trim())); break;
      case '--metric': options.metric = next(); break;
      case '--help':
        process.stdout.write(
          'node scripts/profile-frame.mjs [--tier high] [--camera wide|waterline|island]\n' +
          '  [--frames 150] [--warmup 40] [--repeats 2] [--only name,name] [--dpr 1]\n',
        );
        process.exit(0);
        break;
      default: throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  if (!CAMERAS[options.camera]) {
    throw new Error(`unknown camera "${options.camera}" — have ${Object.keys(CAMERAS).join(', ')}`);
  }
  return options;
}

function findFreePort() {
  return new Promise((resolvePort, reject) => {
    const probe = createServer();
    probe.unref();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolvePort(port));
    });
  });
}

async function waitForServer(url, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(url, { redirect: 'follow' })).ok) return;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error(`preview server did not respond at ${url}`);
}

async function startPreview() {
  const port = await findFreePort();
  const url = `http://127.0.0.1:${port}/`;
  const child = spawn(
    process.execPath,
    [join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js'), 'preview',
      '--port', String(port), '--strictPort', '--host', '127.0.0.1'],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  child.stdout.on('data', () => {});
  child.stderr.on('data', () => {});
  await waitForServer(url);
  return { url, stop: () => child.kill() };
}

/** This host returns 204 for `.bin` in a headed browser. See docs/PERFORMANCE.md. */
function serveBinariesFromDisk(page) {
  return page.route('**/*.bin', async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const file = join(ROOT, 'dist', decodeURIComponent(pathname).replace(/^\/+/, ''));
    if (!existsSync(file) || !statSync(file).isFile()) return route.continue();
    return route.fulfill({
      status: 200,
      contentType: 'application/octet-stream',
      body: readFileSync(file),
    });
  });
}

const percentile = (sorted, f) =>
  sorted.length === 0 ? null : sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(f * sorted.length) - 1))];

function summarise(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const round = (v) => Math.round(v * 1000) / 1000;
  return {
    samples: sorted.length,
    p50: round(percentile(sorted, 0.5)),
    p90: round(percentile(sorted, 0.9)),
    min: round(sorted[0]),
    max: round(sorted[sorted.length - 1]),
  };
}

/**
 * The scenarios, and the shape of one.
 *
 * `apply` runs in the page and returns a function that undoes itself, so a
 * scenario cannot leak into the next one. Anything that would recompile a shader
 * is not allowed here — a recompile costs milliseconds that would be attributed
 * to the subsystem rather than to the edit. `visible = false` and an
 * `instanceCount` change are both free.
 */
const SCENARIOS = `
(() => {
  const ocean = window.__ocean;
  const scene = ocean.scene;
  const byName = (name) => scene.getObjectByName(name) ?? null;

  const hide = (label, name, note) => ({
    label, note,
    apply: () => {
      const object = byName(name);
      if (!object) return null;
      const was = object.visible;
      object.visible = false;
      return () => { object.visible = was; };
    },
  });

  return [
    hide('sea surface', 'ocean-surface',
      'water fragment shading, the planar reflector and SSR all hang off this draw'),
    hide('sky dome', 'sky-dome', 'the analytic atmosphere'),
    hide('cloud layer', 'cloud-layer', 'the volumetric cloud march, cloudSteps'),
    hide('seafloor', 'seafloor', 'the island heightfield and the seabed are one mesh'),
    hide('meadow', 'island-grass-field', 'instanced grass blades, QualitySettings.meadow'),
    hide('canopy', 'island-canopy-field', 'instanced billboards, QualitySettings.canopy'),
    hide('kelp', 'kelp-forest', 'instanced blades, QualitySettings.kelp'),
    hide('fish', 'fish-schools', 'instanced, QualitySettings.fish'),
    hide('birds', 'birds', 'instanced, QualitySettings.birds'),
    hide('props', 'props', 'island dressing, cove, fort, reef garden'),
    hide('remains', 'remains', 'the bone field'),
    hide('ship', 'ship', 'the hero glTF'),
    hide('weather', 'weather', 'rain and snow particles'),
    hide('underwater particles', 'underwater-particles', 'skipped while dry, so expect ~0'),

    {
      label: 'sun shadow',
      note: 'the whole shadow-map pass: every caster is submitted a second time',
      apply: () => {
        const light = ocean.scene.getObjectByName('sun');
        if (!light) return null;
        const was = light.castShadow;
        light.castShadow = false;
        return () => { light.castShadow = was; };
      },
    },
    {
      label: 'planar reflection @ 0.1',
      note: 'not a removal — shrinks the mirrored view to a tenth, so the delta is its pixel cost only',
      apply: () => {
        const reflections = ocean.reflections;
        if (!reflections) return null;
        const was = reflections.resolutionScale;
        reflections.setQuality(0.1);
        return () => { reflections.setQuality(was); };
      },
    },

    // --- splitting the shadow ------------------------------------------------
    // "sun shadow" above removes both halves at once: the depth pass that draws
    // the map, and the filtered lookup every receiving fragment does. These two
    // separate them, and the answer decides which lever is worth pulling.
    {
      label: 'no shadow casters',
      note: 'the light keeps its shadow node so receivers still sample; the depth pass just draws nothing. '
        + 'This is the decisive split — against "sun shadow" it says whether the cost is rastering the map or reading it',
      apply: () => {
        const restore = [];
        ocean.scene.traverse((object) => {
          if (object.castShadow) {
            restore.push(object);
            object.castShadow = false;
          }
        });
        return () => { for (const object of restore) object.castShadow = true; };
      },
    },
    {
      label: 'shadow map at 512',
      note: 'ShadowNode assigns its render target to light.shadow.map, and the filter reads mapSize live, '
        + 'so both can be moved without a recompile. Delta = how much of the pass is its own pixels',
      apply: () => {
        const light = ocean.scene.getObjectByName('sun');
        const map = light?.shadow?.map;
        if (!map) return null;
        const wasW = map.width, wasH = map.height;
        const wasSize = light.shadow.mapSize.clone();
        map.setSize(512, 512);
        light.shadow.mapSize.set(512, 512);
        return () => {
          map.setSize(wasW, wasH);
          light.shadow.mapSize.copy(wasSize);
        };
      },
    },
    {
      label: 'one 48-triangle caster',
      note: 'the pass still runs at full map size and draws almost nothing. Against "no shadow casters" this '
        + 'separates the cost of the geometry from the cost of having the pass at all',
      apply: () => {
        const casters = [];
        ocean.scene.traverse((object) => { if (object.castShadow) casters.push(object); });
        if (casters.length === 0) return null;
        // Smallest by triangle count, so the pass has a caster and no work.
        const triangles = (o) => {
          const g = o.geometry;
          const n = g?.index ? g.index.count : g?.attributes?.position?.count ?? 0;
          return (n / 3) * (o.isInstancedMesh ? o.count : 1);
        };
        const keep = casters.reduce((a, b) => (triangles(a) <= triangles(b) ? a : b));
        const disabled = casters.filter((o) => o !== keep);
        for (const object of disabled) object.castShadow = false;
        return () => { for (const object of disabled) object.castShadow = true; };
      },
    },

    // --- candidate fixes, measured before they are written ------------------
    {
      label: 'clouds behind depth test',
      note: 'draw the dome after the opaque scene with depth testing on, so the ocean and the island reject '
        + 'cloud fragments before the march runs instead of after. Delta = what the occluded march is costing',
      apply: () => {
        const mesh = byName('cloud-layer');
        if (!mesh) return null;
        const wasTest = mesh.material.depthTest;
        const wasOrder = mesh.renderOrder;
        mesh.material.depthTest = true;
        mesh.material.needsUpdate = true;
        mesh.renderOrder = 900;
        return () => {
          mesh.material.depthTest = wasTest;
          mesh.material.needsUpdate = true;
          mesh.renderOrder = wasOrder;
        };
      },
    },
    {
      label: 'clouds after the water',
      note: 'the water is transparent+depthWrite, so it draws after the whole opaque queue and cannot reject '
        + 'an opaque-queue cloud. This puts the dome in the transparent queue behind it, where it can',
      apply: () => {
        const mesh = byName('cloud-layer');
        if (!mesh) return null;
        const m = mesh.material;
        const was = { test: m.depthTest, transparent: m.transparent, order: mesh.renderOrder };
        m.depthTest = true;
        m.transparent = true;
        m.needsUpdate = true;
        mesh.renderOrder = 1;
        return () => {
          m.depthTest = was.test;
          m.transparent = was.transparent;
          m.needsUpdate = true;
          mesh.renderOrder = was.order;
        };
      },
    },
    {
      label: 'sky dome behind depth test',
      note: 'the same question for the analytic sky, which is also drawn first with no depth test',
      apply: () => {
        const mesh = byName('sky-dome');
        if (!mesh) return null;
        const wasTest = mesh.material.depthTest;
        const wasOrder = mesh.renderOrder;
        mesh.material.depthTest = true;
        mesh.material.needsUpdate = true;
        mesh.renderOrder = 899;
        return () => {
          mesh.material.depthTest = wasTest;
          mesh.material.needsUpdate = true;
          mesh.renderOrder = wasOrder;
        };
      },
    },

    {
      label: 'SSR trace disabled',
      note: 'a 24-step screen-space march plus refinement, per water fragment, at High. 0 returns the fallback '
        + 'after one uniform compare, so this is the trace and nothing else',
      apply: () => {
        const ssr = ocean.ssr;
        if (!ssr || typeof ssr.setQuality !== 'function') return null;
        ssr.setQuality(0);
        return () => { ssr.setQuality(24); };
      },
    },
    {
      label: 'reflection without the seabed',
      note: 'the reflector clips at the water plane, so everything below it is submitted, vertex-shaded and '
        + 'then thrown away. This hides it for the mirrored render only — the image must not change',
      apply: () => {
        const base = ocean.reflections?.node?.reflector;
        if (!base || typeof base.updateBefore !== 'function') return null;
        const names = ['fish-schools', 'underwater-particles', 'kelp-forest'];
        const was = base.updateBefore.bind(base);
        base.updateBefore = (frame) => {
          const hidden = [];
          for (const name of names) {
            const object = scene.getObjectByName(name);
            if (object && object.visible) { object.visible = false; hidden.push(object); }
          }
          try { return was(frame); } finally { for (const object of hidden) object.visible = true; }
        };
        return () => { base.updateBefore = was; };
      },
    },
    {
      label: 'reflection pass skipped',
      note: 'stubs the reflector node\\'s updateBefore, so the mirrored scene render never runs and the water '
        + 'samples a stale texture. Delta = the whole second view of the scene, which resolutionScale does not touch',
      apply: () => {
        const base = ocean.reflections?.node?.reflector;
        if (!base || typeof base.updateBefore !== 'function') return null;
        const was = base.updateBefore;
        base.updateBefore = () => {};
        return () => { base.updateBefore = was; };
      },
    },

    // --- the post chain and the simulation ----------------------------------
    // None of this is reachable by hiding a scene object, which is why the
    // first rounds of this profile accounted for only about nine of fourteen
    // milliseconds. Each of these is a fullscreen cost with a step count, a tap
    // count or an enable behind it.
    {
      label: 'volumetric fog march',
      note: 'fogSteps is 18 at High — a fullscreen raymarch. 0 is documented as a bit-exact pass-through',
      apply: () => {
        const fog = ocean.post?.fog;
        if (!fog || typeof fog.setSteps !== 'function') return null;
        fog.setSteps(0);
        return () => { fog.setSteps(18); };
      },
    },
    {
      label: 'depth of field',
      note: 'dofSamples is 16 at High, on a golden-angle spiral. 0 is documented as a bit-exact pass-through',
      apply: () => {
        const dof = ocean.post?.dof;
        if (!dof || typeof dof.setSamples !== 'function') return null;
        dof.setSamples(0);
        return () => { dof.setSamples(16); };
      },
    },
    {
      label: 'bloom pyramid',
      note: 'five downsample-and-blur passes; setEnabled(false) also drops the pyramid to an eighth',
      apply: () => {
        const bloom = ocean.post?.bloom;
        if (!bloom || typeof bloom.setEnabled !== 'function') return null;
        bloom.setEnabled(false);
        return () => { bloom.setEnabled(true); };
      },
    },
    {
      label: 'lens rain',
      note: 'lensRainQuality 3 at High. A clear frame is documented as costing one compare at any level',
      apply: () => {
        const rain = ocean.post?.lensRain;
        if (!rain || typeof rain.setQuality !== 'function') return null;
        rain.setQuality(0);
        return () => { rain.setQuality(3); };
      },
    },
    // --- what a render pass costs, and whether the material matters ---------
    // These two ADD work rather than removing it, so their delta is negative and
    // its magnitude is the cost of the work added. They settle the question the
    // rest of the transform plan hangs on: is the ~50 us per fullscreen pass the
    // *pass boundary* — setRenderTarget, end/begin, the render-object machinery
    // — or is it the fact that 96 distinct materials are cycled through?
    //
    // Non-destructive by construction. The extra passes run *before* the real
    // update, writing into pairA.write, which the real update's first
    // butterfly pass then overwrites from a freshly evolved pairA.read. The
    // wave field the frame displaces from is unchanged.
    {
      label: 'plus 96 passes / one material',
      note: 'the same material and target 96 extra times. Magnitude of the delta / 96 is the per-pass floor',
      apply: () => {
        const sim = ocean.simulation;
        const first = sim?.cascades?.[0];
        if (!first || typeof sim.runPass !== 'function') return null;
        const was = sim.update.bind(sim);
        const material = first.butterflyA[0];
        const target = first.pairA.write;
        sim.update = (t) => {
          for (let i = 0; i < 96; i++) sim.runPass(material, target);
          was(t);
        };
        return () => { sim.update = was; };
      },
    },
    {
      label: 'plus 96 passes / 96 materials',
      note: 'the same 96 extra passes, cycled across every even-stage material in the scene. Against the row '
        + 'above this says whether material identity costs anything once the pipelines are cached',
      apply: () => {
        const sim = ocean.simulation;
        const first = sim?.cascades?.[0];
        if (!first || typeof sim.runPass !== 'function') return null;
        // Even-index stages only. Odd stages bind pair.write as their source,
        // and writing into the target they read is a WebGPU validation error —
        // "includes writable usage and another usage in the same synchronization
        // scope". Even stages read pair.read, so writing to pairA.write is
        // safe for every one of them.
        const materials = [];
        for (const cascade of sim.cascades) {
          for (const list of [cascade.butterflyA, cascade.butterflyB]) {
            for (let i = 0; i < list.length; i += 2) materials.push(list[i]);
          }
        }
        if (materials.length < 8) return null;
        const was = sim.update.bind(sim);
        const target = first.pairA.write;
        sim.update = (t) => {
          for (let i = 0; i < 96; i++) sim.runPass(materials[i % materials.length], target);
          was(t);
        };
        return () => { sim.update = was; };
      },
    },

    // --- deciding between amortising and cutting the pass count -------------
    // Per cascade per frame the simulation runs 2 evolve passes, 2*(2*log2 N)
    // butterfly passes and 2 assemble passes — 36 at 256 squared, so 108 for
    // three cascades, of which 96 are butterfly. Whether the fix is "run it less
    // often" or "run it in fewer passes" depends entirely on whether the cost
    // follows the pass count or the texel count, and these two measure exactly
    // that. Both leave a wrong wave field on purpose; only their cost is read.
    {
      label: 'half the butterfly stages',
      note: 'same texels, same targets, half the passes. If this saves about half the transform, the cost is '
        + 'per-pass and folding stages (radix-4, MRT, array layers) is the lever rather than update rate',
      apply: () => {
        const sim = ocean.simulation;
        if (!sim || typeof sim.runFFT !== 'function') return null;
        const was = sim.runFFT.bind(sim);
        sim.runFFT = (pair, materials) => was(pair, materials.slice(0, materials.length >> 1));
        return () => { sim.runFFT = was; };
      },
    },
    {
      label: 'skip the 3rd cascade',
      note: 'drops 36 of the 108 passes. Against the two rows above this gives the per-cascade cost, which is '
        + 'what amortising one band at a time would be worth',
      apply: () => {
        const sim = ocean.simulation;
        const cascades = sim?.cascades;
        if (!Array.isArray(cascades) || cascades.length < 3) return null;
        const removed = cascades.splice(2, 1);
        return () => { cascades.push(...removed); };
      },
    },
    {
      label: 'skip the 2nd and 3rd cascades',
      note: 'one cascade left, 36 of 108 passes',
      apply: () => {
        const sim = ocean.simulation;
        const cascades = sim?.cascades;
        if (!Array.isArray(cascades) || cascades.length < 3) return null;
        const removed = cascades.splice(1, 2);
        return () => { cascades.push(...removed); };
      },
    },
    {
      label: 'FFT on alternate frames',
      note: 'the whole transform re-runs every frame. This halves its rate to test whether amortising it is '
        + 'worth pursuing — not a shippable change, the wave field would need interpolating',
      apply: () => {
        const simulation = ocean.simulation;
        if (!simulation || typeof simulation.update !== 'function') return null;
        const was = simulation.update.bind(simulation);
        let n = 0;
        simulation.update = (t) => { if ((n++ & 1) === 0) was(t); };
        return () => { simulation.update = was; };
      },
    },
    {
      label: 'FFT + spectrum',
      note: 'the whole wave simulation: ~96 fullscreen passes per frame at three cascades. The surface keeps '
        + 'displacing from the last field it computed, so this is the transform cost and nothing else',
      apply: () => {
        const simulation = ocean.simulation;
        if (!simulation || typeof simulation.update !== 'function') return null;
        const was = simulation.update;
        simulation.update = () => {};
        return () => { simulation.update = was; };
      },
    },

    // --- the frame's own plumbing -------------------------------------------
    {
      label: 'buoyancy readback',
      note: 'Sampler.update maps a render target back to host memory every frame. The read is not awaited, '
        + 'but a mapAsync in flight can still bound how many frames the driver will queue',
      apply: () => {
        const sampler = ocean.sampler;
        if (!sampler || typeof sampler.update !== 'function') return null;
        const was = sampler.update;
        sampler.update = () => {};
        return () => { sampler.update = was; };
      },
    },

    // --- how the cloud march scales -----------------------------------------
    // Hiding the layer says what it costs. This says whether the cost is in the
    // step count — which is a tier knob — or in the per-pixel setup, which is not.
    {
      label: 'cloud march at 4 steps',
      note: 'High runs 18. If most of the cost survives this, the march is not step-bound',
      apply: () => {
        const clouds = ocean.clouds;
        if (!clouds) return null;
        const was = clouds.getParams().steps;
        clouds.setParams({ steps: 4 });
        return () => { clouds.setParams({ steps: was }); };
      },
    },
  ];
})()
`;

async function measureScenarios(page, options) {
  return page.evaluate(async ([opts, scenarioSource]) => {
    const ocean = window.__ocean;
    const renderer = ocean.renderer;
    const backend = renderer.backend;
    const raf = () => new Promise((done) => requestAnimationFrame(done));

    // eslint-disable-next-line no-eval
    const scenarios = eval(scenarioSource);

    /**
     * One measurement: rewind the world, then render a fixed number of frames
     * from it.
     *
     * The reset is per-scenario rather than once at the start, and that is the
     * whole basis of the comparison: every scenario then sees byte-identical
     * world state at frame N, so the only difference between two runs is the
     * mutation. Timestamp tracking is switched off across the reset because a
     * 60-step settle would queue ~13 000 queries into a 2048-entry pool.
     */
    /**
     * Two instruments, and the second exists because the first was not believed.
     *
     * `timestamp` is the benchmark's metric: `resolveTimestampsAsync` summed over
     * the frame's render passes. It is the right thing to compare against
     * `npm run bench`, but it is a *sum of pass durations*, so passes the GPU
     * overlaps are counted twice and a pass that spends its bracket waiting
     * charges the wait to itself. That matters here: this script's headline
     * finding was a render pass that appeared to cost 5.9 ms while drawing
     * nothing, which is not a thing a 5090 can do, and a metric cannot be used to
     * confirm its own anomaly.
     *
     * `fence` uses no timestamps at all. It submits the frame and then awaits
     * `queue.onSubmittedWorkDone()`, so the wall clock it measures includes the
     * GPU actually finishing. It is noisier and it carries submission latency
     * the GPU number does not, but it is *independent*, and two instruments
     * disagreeing is the only way to find out that one of them is wrong.
     */
    const useFence = opts.metric === 'fence';
    const useRaf = opts.metric === 'raf';
    const device = backend.device ?? null;

    /**
     * The third instrument, and the one that answers the question actually
     * asked: how fast does the application run.
     *
     * Neither of the other two does. `timestamp` sums pass durations with the
     * pipeline drained every frame; `fence` serialises CPU and GPU by awaiting
     * completion. Both destroy the overlap a real frame gets. This one lets the
     * app's own rAF loop run untouched and counts frames that actually
     * completed — `Loop.tick` skips while a render is in flight, so one call of
     * a registered update is one delivered frame. With vsync off, the rate is
     * the renderer's throughput.
     *
     * It is the noisiest of the three and the only one that is not a frame
     * *cost*. It is here as the tie-breaker.
     */
    const runRaf = async () => {
      // `resetDeterministic` rewinds the simulation clock to 0, and
      // AdaptiveQuality ignores everything before elapsed = 4 s. The whole
      // window below is 3.5 s, so the tier provably cannot move mid-sample —
      // which is the same hazard `npm run bench` pins `loop.stats.fps` for.
      await ocean.resetDeterministic(0, 60);
      let frames = 0;
      let updateTotal = 0;
      let updateStart = 0;

      /**
       * Bracket the update phase.
       *
       * `Loop.updates` is `private` in TypeScript, which is a compile-time
       * claim and not a runtime one — the array is an ordinary property here.
       * Putting a probe at each end of it is the only way to separate the time
       * the frame spends in JavaScript from the time it spends waiting on the
       * render, and that split is the difference between "make the shaders
       * cheaper" and "the shaders are not the problem".
       */
      const first = () => { updateStart = performance.now(); };
      const last = () => { updateTotal += performance.now() - updateStart; frames++; };
      ocean.loop.updates.unshift(first);
      ocean.loop.add(last);

      ocean.setPaused(false);
      await new Promise((done) => setTimeout(done, 1000));   // discard the ramp

      const windows = [];
      const renderMs = [];
      const updateMs = [];
      for (let w = 0; w < 5; w++) {
        frames = 0;
        updateTotal = 0;
        const started = performance.now();
        await new Promise((done) => setTimeout(done, 500));
        const wall = performance.now() - started;
        if (frames > 0) { windows.push(wall / frames); updateMs.push(updateTotal / frames); }
        // `Loop.stats.frameMs` is wall time *inside* renderFn. On WebGPU
        // `renderAsync` is supposed to return once the work is submitted, so
        // this should be small and the CPU should be free to build the next
        // frame while the GPU draws this one. If it is close to the whole frame
        // period instead, the render call is waiting on the GPU and there is no
        // overlap at all — which is worth more than any single subsystem here.
        renderMs.push(ocean.loop.stats.frameMs);
      }

      ocean.setPaused(true);
      ocean.loop.remove(last);
      ocean.loop.updates.splice(ocean.loop.updates.indexOf(first), 1);
      // Milliseconds per delivered frame, so it lands in the same column as the
      // other two metrics and a delta there means the same thing here.
      return { gpu: windows, counters: null, renderMs, updateMs };
    };

    const run = async () => {
      if (useRaf) return runRaf();
      backend.trackTimestamp = false;
      await ocean.resetDeterministic(0, 60);
      ocean.setPaused(true);
      ocean.loop.stats.fps = 1000;   // AdaptiveQuality must not act mid-sample
      if (!useFence) {
        backend.trackTimestamp = true;
        try { await renderer.resolveTimestampsAsync('render'); } catch { /* drain */ }
      }

      const gpu = [];
      let counters = null;
      for (let i = 0; i < opts.warmup + opts.frames; i++) {
        await raf();

        let value = null;
        if (useFence) {
          const started = performance.now();
          await ocean.loop.step(1 / 60, 1);
          if (device) await device.queue.onSubmittedWorkDone();
          value = performance.now() - started;
        } else {
          await ocean.loop.step(1 / 60, 1);
        }

        const render = renderer.info.render;
        const frameCounters = {
          drawCalls: render.drawCalls,
          triangles: render.triangles,
          frameCalls: render.frameCalls,
        };

        if (!useFence) {
          try {
            const resolved = await renderer.resolveTimestampsAsync('render');
            if (typeof resolved === 'number' && Number.isFinite(resolved) && resolved > 0) {
              value = resolved;
            }
          } catch { /* not available */ }
        }

        if (i < opts.warmup) continue;
        if (value !== null) gpu.push(value);
        counters = frameCounters;
      }
      return { gpu, counters };
    };

    const results = [];
    const baselines = [];

    /**
     * A fresh baseline immediately before every scenario, and the delta taken
     * against that one rather than against a global average.
     *
     * This is not caution, it is a correction. The first version took one
     * baseline at the start and it read 8.356 ms while every neutral scenario
     * later in the same run clustered at 7.41-7.53 — the GPU had not finished
     * clocking up, and 0.8 ms of ramp was being handed out as if it were a
     * saving. Pairing each measurement with its own neighbour makes the ramp
     * common-mode, and it is why the first baseline below is discarded.
     */
    baselines.push(await run());   // discarded: absorbs the clock ramp

    for (const scenario of scenarios) {
      if (opts.only && !opts.only.includes(scenario.label)) continue;
      const local = await run();
      baselines.push(local);
      const undo = scenario.apply();
      if (!undo) {
        results.push({ label: scenario.label, note: scenario.note, absent: true });
        continue;
      }
      let measured;
      try {
        measured = await run();
      } finally {
        undo();
      }
      results.push({ label: scenario.label, note: scenario.note, local, ...measured });
    }

    /**
     * What is in the scene and how it is flagged.
     *
     * Recorded because two scenarios in the first run of this script were silent
     * no-ops — `shadow.mapSize` after the node is built, and `receiveShadow` on a
     * material that samples the shadow itself — and both produced a plausible
     * small number that would have been read as a finding. An audit of the flags
     * makes that class of mistake visible instead of convincing.
     */
    const sun = ocean.scene.getObjectByName('sun');
    const shadowMap = sun?.shadow?.map;
    const shadowTarget = shadowMap ? {
      width: shadowMap.width,
      height: shadowMap.height,
      depth: shadowMap.depth,
      samples: shadowMap.samples,
      colorTextureType: shadowMap.texture?.type,
      colorTextureFormat: shadowMap.texture?.format,
      generateMipmaps: shadowMap.texture?.generateMipmaps,
      depthTextureType: shadowMap.depthTexture?.type,
      depthBuffer: shadowMap.depthBuffer,
      mapSize: sun.shadow.mapSize.toArray(),
      autoUpdate: sun.shadow.autoUpdate,
      filterType: ocean.renderer.shadowMap.type,
    } : null;

    const audit = [];
    ocean.scene.traverse((object) => {
      if (!object.isMesh) return;
      const geometry = object.geometry;
      const index = geometry?.index;
      const position = geometry?.attributes?.position;
      const perInstance = index ? index.count / 3 : position ? position.count / 3 : 0;
      audit.push({
        name: object.name || `(${object.type})`,
        visible: object.visible,
        castShadow: object.castShadow,
        receiveShadow: object.receiveShadow,
        instances: object.isInstancedMesh ? object.count : 1,
        trianglesPerInstance: Math.round(perInstance),
        triangles: Math.round(perInstance * (object.isInstancedMesh ? object.count : 1)),
        frustumCulled: object.frustumCulled,
      });
    });

    ocean.setPaused(false);
    return {
      baselines,
      results,
      audit,
      shadowTarget,
      state: ocean.getState(),
      memory: { ...renderer.info.memory },
      drawingBuffer: {
        width: renderer.domElement.width,
        height: renderer.domElement.height,
        pixelRatio: renderer.getPixelRatio(),
      },
    };
  }, [{ warmup: options.warmup, frames: options.frames, repeats: options.repeats, metric: options.metric, only: options.only ? [...options.only] : null }, SCENARIOS]);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  const build = spawn('npm', ['run', 'build'], { cwd: ROOT, stdio: 'inherit', shell: true });
  const buildCode = await new Promise((done) => build.on('exit', done));
  if (buildCode !== 0) throw new Error(`build failed with code ${buildCode}`);

  const preview = await startPreview();
  const browser = await chromium.launch({
    headless: false,
    channel: 'chrome',
    args: CHROME_FLAGS,
  });

  try {
    const context = await browser.newContext({
      viewport: { width: options.width, height: options.height },
      deviceScaleFactor: options.dpr,
    });
    const page = await context.newPage();
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 300)); });
    page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
    await serveBinariesFromDisk(page);

    const launchUrl = new URL(preview.url);
    launchUrl.searchParams.set('gate', '0');
    launchUrl.searchParams.set('quality', options.tier);
    await page.goto(launchUrl.toString(), { waitUntil: 'domcontentloaded' });
    await page.bringToFront();
    await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 120_000 });
    await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 180_000 });
    await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 180_000 });
    await page.waitForFunction(() => window.__ocean.sceneContentLoaded(), undefined, { timeout: 180_000 });

    const camera = CAMERAS[options.camera];
    await page.evaluate(async ([tier, preset, cam]) => {
      const ocean = window.__ocean;
      await ocean.setState({ quality: tier, preset, pixelRatio: 1, cameraMode: 'orbit' });
      ocean.setCamera(...cam.position, ...cam.target);
      await ocean.renderer.compileAsync(ocean.scene, ocean.camera);
    }, [options.tier, options.preset, camera]);
    await page.waitForTimeout(options.settleSeconds * 1000);

    const raw = await measureScenarios(page, options);

    // The leading baseline is the clock-ramp absorber and is excluded from the
    // spread, which would otherwise report the ramp as measurement noise.
    const baseSummaries = raw.baselines.map((b) => summarise(b.gpu));
    const settled = baseSummaries.slice(1).map((s) => s?.p50).filter((v) => typeof v === 'number');
    const baseline = settled.reduce((a, b) => a + b, 0) / settled.length;
    const drift = settled.length > 1 ? Math.max(...settled) - Math.min(...settled) : 0;

    const rows = raw.results.map((r) => {
      if (r.absent) return { ...r };
      const gpu = summarise(r.gpu);
      const local = summarise(r.local?.gpu ?? [])?.p50 ?? baseline;
      return {
        label: r.label,
        note: r.note,
        gpu,
        localBaselineP50: local,
        deltaMs: gpu ? Math.round((local - gpu.p50) * 1000) / 1000 : null,
        percent: gpu ? Math.round(((local - gpu.p50) / local) * 1000) / 10 : null,
        drawCalls: r.counters?.drawCalls ?? null,
        triangles: r.counters?.triangles ?? null,
        renderPasses: r.counters?.frameCalls ?? null,
        renderMs: r.renderMs ? summarise(r.renderMs)?.p50 ?? null : null,
        updateMs: r.updateMs ? summarise(r.updateMs)?.p50 ?? null : null,
      };
    });

    const baseCounters = raw.baselines[0].counters ?? {};
    rows.sort((a, b) => (b.deltaMs ?? -Infinity) - (a.deltaMs ?? -Infinity));

    const report = {
      schema: 'web-ocean-3d/profile@1',
      startedAt: new Date().toISOString(),
      options: { ...options, only: options.only ? [...options.only] : null },
      camera,
      baseline: {
        gpuP50: Math.round(baseline * 1000) / 1000,
        driftMs: Math.round(drift * 1000) / 1000,
        repeats: baseSummaries,
        drawCalls: baseCounters.drawCalls ?? null,
        triangles: baseCounters.triangles ?? null,
        renderMsP50: raw.baselines[1]?.renderMs ? summarise(raw.baselines[1].renderMs)?.p50 ?? null : null,
        updateMsP50: raw.baselines[1]?.updateMs ? summarise(raw.baselines[1].updateMs)?.p50 ?? null : null,
      },
      drawingBuffer: raw.drawingBuffer,
      memory: raw.memory,
      scenarios: rows,
      shadowTarget: raw.shadowTarget ?? null,
      audit: [...(raw.audit ?? [])].sort((a, b) => b.triangles - a.triangles),
      consoleErrors,
    };

    mkdirSync(RESULTS_DIR, { recursive: true });
    const file = join(RESULTS_DIR, `profile-${options.tier}-${options.camera}-${report.startedAt.replace(/[:.]/g, '-')}.json`);
    writeFileSync(file, JSON.stringify(report, null, 2));

    const pad = (s, n) => String(s).padEnd(n);
    const num = (v, n) => String(v === null || v === undefined ? '—' : v).padStart(n);
    process.stdout.write(
      `\n${options.tier} · ${options.camera} · ${raw.drawingBuffer.width}x${raw.drawingBuffer.height} @ DPR ${raw.drawingBuffer.pixelRatio}\n` +
      `baseline p50 ${report.baseline.gpuP50} ms  (spread across ${settled.length} settled baselines: ${report.baseline.driftMs} ms)\n` +
      `baseline draws ${baseCounters.drawCalls}, triangles ${baseCounters.triangles?.toLocaleString?.() ?? baseCounters.triangles}\n\n` +
      `${pad('removed', 26)}${num('Δ ms', 8)}${num('Δ %', 7)}${num('base', 8)}${num('draws', 8)}${num('passes', 8)}${num('tris', 13)}\n` +
      `${'-'.repeat(78)}\n`,
    );
    for (const row of rows) {
      if (row.absent) {
        process.stdout.write(`${pad(row.label, 26)}${num('absent', 8)}\n`);
        continue;
      }
      process.stdout.write(
        `${pad(row.label, 26)}${num(row.deltaMs, 8)}${num(row.percent, 7)}` +
        `${num(row.localBaselineP50, 8)}${num(row.renderMs, 9)}${num(row.updateMs, 9)}${num(row.drawCalls, 8)}` +
        `${num(row.triangles?.toLocaleString?.() ?? row.triangles, 13)}\n`,
      );
    }
    if (consoleErrors.length) {
      process.stdout.write(`\nconsole errors during the run:\n  ${consoleErrors.join('\n  ')}\n`);
    }
    process.stdout.write(`\nwritten to ${file}\n`);
  } finally {
    await browser.close();
    preview.stop();
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
});
