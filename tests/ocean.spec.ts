import { test, expect } from '@playwright/test';
import {
  collectConsoleErrors,
  hasGpuAdapter,
  measureFrameRate,
  setCamera,
  setState,
  waitForOcean,
} from './helpers';
import { CINEMATIC_LOOP_SECONDS } from '../src/cameras/Cinematic';
import { capture } from './lib/capture';
import { compareImages } from './lib/compare';
import type { RgbaImage } from './lib/png';

test.describe('boot and rendering', () => {
  test('boots with no console errors and draws a non-empty frame', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);

    // Whether WebGPU is actually reachable depends on the machine and, notably,
    // on the browser build: Playwright's bundled Chromium commonly exposes no
    // WebGPU adapter at all. Asserting "backend === webgpu" unconditionally would
    // therefore fail on a correctly-behaving fallback. Instead, ask the page what
    // is available and require the renderer to have made the right choice.
    const { backend, adapterAvailable } = await page.evaluate(async () => {
      let adapter = false;
      try {
        adapter = navigator.gpu ? (await navigator.gpu.requestAdapter()) !== null : false;
      } catch {
        adapter = false;
      }
      return {
        backend: (window as unknown as { __ocean: { backend: string } }).__ocean.backend,
        adapterAvailable: adapter,
      };
    });

    if (adapterAvailable) {
      expect(backend, 'a WebGPU adapter exists but the renderer did not use it').toBe('webgpu');
    } else {
      expect(backend, 'no WebGPU adapter, so the renderer must fall back').toBe('webgl');
    }

    // The canvas must contain a rendered scene, not a cleared buffer.
    //
    // Deliberately NOT via drawImage on the canvas: without preserveDrawingBuffer
    // that reads back blank on both WebGL and WebGPU, so it measures a readback
    // limitation rather than the render. The compositor screenshot is the honest
    // source. PNG is entropy-coded, so a flat frame compresses to a few KB while
    // a detailed ocean is orders of magnitude larger — size is a sound proxy for
    // "there is structure on screen".
    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: software rasterisation cannot deliver a screenshot in time',
    );
    const shot = await page.screenshot();
    expect(
      shot.byteLength,
      `frame compressed to ${shot.byteLength} bytes, which indicates a blank or uniform image`,
    ).toBeGreaterThan(120_000);
  });

  test('falls back to WebGL2 and still renders', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high&webgl=1');
    await waitForOcean(page);

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);

    const backend = await page.evaluate(
      () => (window as unknown as { __ocean: { backend: string } }).__ocean.backend,
    );
    expect(backend).toBe('webgl');

    // "still renders" has to mean a frame was rendered.
    //
    // This asserted the backend string and an empty console and stopped there,
    // which is satisfied by an app that boots cleanly onto WebGL2 and then draws
    // a flat blue rectangle — or nothing at all. Read the frame and require
    // structure in it.
    const frame = await capture(page);
    let min = 255;
    let max = 0;
    let sum = 0;
    for (let i = 0; i < frame.data.length; i += 4) {
      const luma =
        0.2126 * frame.data[i] + 0.7152 * frame.data[i + 1] + 0.0722 * frame.data[i + 2];
      if (luma < min) min = luma;
      if (luma > max) max = luma;
      sum += luma;
    }
    const mean = sum / (frame.data.length / 4);
    expect(mean, 'the WebGL2 frame is black').toBeGreaterThan(4);
    expect(
      max - min,
      `the WebGL2 frame spans only ${(max - min).toFixed(1)} levels of luminance, ` +
        'which is a cleared buffer rather than a rendered scene',
    ).toBeGreaterThan(40);
  });
});

test.describe('wave simulation', () => {
  test('produces a physically plausible sea state', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    const stats = await page.evaluate(async () => {
      const ocean = (window as unknown as { __ocean: Record<string, never> }).__ocean;
      const renderer = ocean.renderer as unknown as {
        readRenderTargetPixelsAsync: (
          t: unknown, x: number, y: number, w: number, h: number,
        ) => Promise<ArrayLike<number>>;
      };
      const simulation = ocean.simulation as unknown as {
        displacementTargets: unknown[];
        tileSizes: number[];
      };

      const halfToFloat = (bits: number) => {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits & 0x7c00) >> 10;
        const mantissa = bits & 0x03ff;
        if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
        if (exponent === 31) return mantissa ? NaN : sign * Infinity;
        return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
      };

      let peakHeight = 0;
      let nonFinite = 0;
      let foldedFraction = 0;
      let samples = 0;

      for (const target of simulation.displacementTargets) {
        // 64 wide keeps bytesPerRow a multiple of 256 so no padding rows appear.
        const raw = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 64, 64);
        const isHalf = (raw as ArrayLike<number> & { BYTES_PER_ELEMENT?: number })
          .BYTES_PER_ELEMENT === 2;
        for (let i = 0; i < raw.length; i += 4) {
          const height = isHalf ? halfToFloat(raw[i + 1]) : raw[i + 1];
          const jacobian = isHalf ? halfToFloat(raw[i + 3]) : raw[i + 3];
          if (!Number.isFinite(height)) nonFinite++;
          peakHeight = Math.max(peakHeight, Math.abs(height));
          if (jacobian < 0) foldedFraction++;
          samples++;
        }
      }

      return { peakHeight, nonFinite, foldedPercent: (100 * foldedFraction) / samples };
    });

    expect(stats.nonFinite, 'displacement field contains non-finite values').toBe(0);

    // At the default 15 m/s wind a fully developed sea has Hs ~= 5 m, so crest
    // amplitude should land in metres — not centimetres, and not tens of metres.
    expect(stats.peakHeight).toBeGreaterThan(0.4);
    expect(stats.peakHeight).toBeLessThan(12);

    // Whitecaps cover a few percent of a real sea at this wind speed. A large
    // number here means the surface is folding everywhere, which is the
    // signature of a broken transform or excessive choppiness.
    expect(stats.foldedPercent).toBeLessThan(8);
  });

  test('the surface actually animates', async ({ page }) => {
    // Each compositor screenshot can block for seconds when the browser is
    // pacing frames at ~1 Hz, and this test takes two of them.
    test.setTimeout(180_000);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);
    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: software rasterisation cannot deliver a screenshot in time',
    );
    await setCamera(page, [0, 12, 40], [0, 0, 0]);

    // Stepped, not waited on.
    //
    // This used to take two compositor screenshots three seconds apart and
    // require them to differ. That is a bet on the browser delivering a frame in
    // the gap, and with an island in the scene it began losing — reporting a
    // frozen simulation when what had actually happened was that the same frame
    // was captured twice. Advancing the clock explicitly asks the question the
    // test means to ask: does the surface change when time passes?
    const first = await capture(page);
    await page.evaluate(() => window.__ocean.step(1 / 60, 30));
    const second = await capture(page);

    let changed = 0;
    for (let i = 0; i < first.width * first.height; i++) {
      if (Math.abs(first.data[i * 4] - second.data[i * 4]) > 2) changed++;
    }
    expect(
      changed / (first.width * first.height),
      'the frame is identical after half a second of simulated time — the sea is frozen',
    ).toBeGreaterThan(0.02);
  });
});

test.describe('interaction', () => {
  /**
   * Every preset must produce a *distinguishable* image, pairwise.
   *
   * The previous version of this test hashed bytes sampled out of the compressed
   * PNG that `page.screenshot()` returns and called the result an average colour.
   * It is not one: PNG is entropy-coded, so those bytes are deflate output and
   * the number means nothing about what is on screen. It went unnoticed because
   * the test also required a WebGPU adapter, which Playwright's Chromium did not
   * have until the launch flags were fixed — so it skipped rather than ran, and
   * the first time it actually executed all nine presets "fingerprinted"
   * identically.
   *
   * This compares real decoded pixels through the same CIE94 metric the visual
   * suite gates on, and checks every pair rather than counting distinct hashes —
   * a hash count cannot tell you *which* two looks collapsed together.
   */
  test('every preset applies without error and produces a distinct image', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    const presets = [
      'skyPro', 'arctic', 'blackFlag', 'dusk', 'foggy',
      'moonlit', 'seaOfThieves', 'storm', 'sunset',
    ] as const;

    const images = new Map<string, RgbaImage>();
    for (const preset of presets) {
      await setState(page, { preset });
      // Pinned time and camera, so what separates two captures is the preset and
      // nothing else — otherwise a pair could differ merely by wave phase.
      await page.evaluate(() => window.__ocean.resetDeterministic(30, 60));
      await setCamera(page, [0, 14, 48], [0, 2, 0]);
      images.set(preset, await capture(page));
    }

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);

    // Well clear of the measured run-to-run noise floor, which tops out around
    // mean ΔE 0.04 — two presets an order of magnitude apart in appearance
    // should be separated by far more than that.
    const MIN_SEPARATION = 1.0;
    const collapsed: string[] = [];
    for (let i = 0; i < presets.length; i++) {
      for (let j = i + 1; j < presets.length; j++) {
        const a = presets[i];
        const b = presets[j];
        const score = compareImages(images.get(a)!, images.get(b)!);
        if (score.meanDeltaE < MIN_SEPARATION) {
          collapsed.push(`${a} vs ${b}: mean ΔE ${score.meanDeltaE.toFixed(3)}`);
        }
      }
    }

    expect(
      collapsed,
      `preset pairs that render too similarly (mean ΔE < ${MIN_SEPARATION}):\n${collapsed.join('\n')}`,
    ).toEqual([]);
  });

  /**
   * The wake buffer must reach the water.
   *
   * `physics/Wake` maintained a correct, world-anchored foam accumulation every
   * frame that `OceanMaterial` never sampled — two fullscreen passes of cost for
   * a texture nothing read, visible only through the debug overlay. Nothing
   * failed when it was disconnected, which is exactly why this test exists: it
   * deposits a wake and requires the rendered surface to change because of it.
   *
   * Deliberately not asserting on the wake's own texture. That would pass just as
   * happily with the binding removed again — the claim under test is that the
   * *water* shows it.
   */
  test('a deposited wake changes the rendered water', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    await setState(page, { preset: 'seaOfThieves', quality: 'high' });

    // Looking down at open water well clear of the hull, so the only thing that
    // can differ between the two captures is the deposit.
    const look = async () => {
      await setCamera(page, [-30, 40, 55], [-25, 0, 0]);
      return capture(page);
    };

    await page.evaluate(() => window.__ocean.resetDeterministic(20, 60));
    const clean = await look();

    // Drive an emitter along a track, stepping between deposits so the buffer
    // accumulates a trail rather than a single stamp.
    await page.evaluate(async () => {
      const ocean = window.__ocean;
      for (let i = 0; i < 90; i++) {
        const x = -60 + i * (8 / 60);
        ocean.wake.emit(x, 0, 0, 8, 7);
        await ocean.step(1 / 60, 1);
      }
    });
    const withWake = await look();

    const score = compareImages(clean, withWake);

    // A Kelvin wedge across open water is a large, bright, unmistakable feature.
    // The bar is set far above the measured run-to-run noise floor (mean ΔE
    // around 0.04) but well below what the wedge actually produces, so this fails
    // on the binding being lost rather than on the foam being retuned.
    expect(
      score.meanDeltaE,
      `depositing a wake changed the water by mean ΔE ${score.meanDeltaE.toFixed(3)}; ` +
        'a value near zero means OceanMaterial is not sampling the wake buffer',
    ).toBeGreaterThan(0.5);
  });

  /**
   * The wake must *deform* the water, not only foam it.
   *
   * The test above cannot tell those apart: it compares clean water against
   * wake-covered water, and a working foam mask alone satisfies it. So this one
   * holds the wake buffer completely fixed — same deposit, same decay state, no
   * simulation steps between the two captures — and toggles only the surface's
   * elevation term. The foam is bit-identical across the pair by construction,
   * which leaves the Kelvin displacement as the one thing that can move the
   * number.
   *
   * A low camera on purpose. Wake elevation shows up as a change in surface
   * *normal*, and a normal only changes what you see through Fresnel and the
   * specular lobe — both of which flatten out toward a top-down view. Looking
   * down at 40 m, the same displacement moves mean ΔE by a fraction of what it
   * does from near the waterline.
   */
  test('the wake displaces the surface, not just its foam', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    await setState(page, { preset: 'seaOfThieves', quality: 'high' });
    await page.evaluate(() => window.__ocean.resetDeterministic(20, 60));

    // Lay a trail running toward +x, then look back down it from low and astern.
    await page.evaluate(async () => {
      const ocean = window.__ocean;
      for (let i = 0; i < 120; i++) {
        const x = -70 + i * (9 / 60);
        ocean.wake.emit(x, 0, 0, 9, 7);
        await ocean.step(1 / 60, 1);
      }
    });
    await setCamera(page, [-90, 6, 26], [-48, 0.5, 0]);

    const withDisplacement = await capture(page);
    await page.evaluate(() => window.__ocean.water.setWakeDisplacement(0));
    const flat = await capture(page);
    // Leave the app as the tier says it should be, so a later test in the same
    // page context is not silently running with the effect off.
    await page.evaluate(() => window.__ocean.water.setWakeDisplacement(1));

    const score = compareImages(flat, withDisplacement);

    // Well above the measured run-to-run noise floor (mean ΔE ~0.04) and well
    // below what the wedge produces, so this fails on the elevation channel
    // being dropped rather than on the wake being retuned.
    expect(
      score.meanDeltaE,
      `enabling wake displacement changed the water by mean ΔE ${score.meanDeltaE.toFixed(3)}; ` +
        'near zero means the surface is not reading the wake buffer\'s elevation channel',
    ).toBeGreaterThan(0.3);
  });

  /**
   * The water must reflect the *scene*, not just a sky gradient.
   *
   * `SPEC.md` has claimed scene reflection as a P0 feature throughout, while the
   * surface reflected `mix(horizonColor, skyColor, reflectDir.y)` — an analytic
   * ramp containing no geometry at all. Asserting that a reflection node exists
   * would not have caught that; what distinguishes the two is whether hiding the
   * ship changes the water underneath it.
   *
   * The camera is placed low and close so the hull's reflection occupies a real
   * part of the frame, and the comparison is restricted to the lower half, below
   * the horizon — otherwise hiding the ship would trivially change the image by
   * removing the ship itself.
   */
  test('the water reflects nearby scene geometry', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    const backend = await page.evaluate(() => window.__ocean.backend);
    test.skip(
      backend !== 'webgpu',
      'planar reflection is a WebGPU-only path; WebGL2 keeps the analytic sky by design',
    );

    await setState(page, { preset: 'seaOfThieves', quality: 'high' });

    const look = async () => {
      await page.evaluate(() => window.__ocean.resetDeterministic(28, 90));
      await setCamera(page, [28, 4.5, 22], [0, 3, 0]);
      return capture(page);
    };

    const withShip = await look();
    await page.evaluate(() => {
      const ship = window.__ocean.scene.getObjectByName('ship');
      if (ship) ship.visible = false;
    });
    const withoutShip = await look();
    await page.evaluate(() => {
      const ship = window.__ocean.scene.getObjectByName('ship');
      if (ship) ship.visible = true;
    });

    // Water only: the bottom half of the frame, which at this camera is entirely
    // below the horizon.
    const half = Math.floor(withShip.height / 2);
    const crop = (image: RgbaImage): RgbaImage => ({
      width: image.width,
      height: image.height - half,
      data: image.data.slice(half * image.width * 4),
    });

    const score = compareImages(crop(withShip), crop(withoutShip));

    expect(
      score.meanDeltaE,
      `hiding the ship changed the water by mean ΔE ${score.meanDeltaE.toFixed(3)}; ` +
        'a value near zero means the surface is reflecting a sky gradient, not the scene',
    ).toBeGreaterThan(0.4);
  });

  /**
   * The screen-space trace has to carry its own weight.
   *
   * The test above cannot show that. Planar reflection and SSR are composited,
   * not chosen between — they fail in opposite places, which is why engines layer
   * them — and both are driven from one tier number. So "hiding the ship changes
   * the water" is satisfied by the planar layer alone, and would pass with the
   * trace completely broken.
   *
   * This turns the planar layer off and asks the same question of what is left.
   * SSR is the layer that has the *displaced* surface and can put the hull's
   * reflection on the wave under it, which one mirror plane at y = 0 cannot do at
   * all, so it is the layer worth having a separate assertion for.
   */
  test('screen-space reflection alone still reflects the hull', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );
    const backend = await page.evaluate(() => window.__ocean.backend);
    test.skip(backend !== 'webgpu', 'SSR is a WebGPU-only path by policy');

    await setState(page, { preset: 'seaOfThieves', quality: 'high' });

    const hasSsr = await page.evaluate(() => window.__ocean.ssr !== null);
    expect(hasSsr, 'no SSR instance on the WebGPU path, so this test proves nothing').toBe(true);

    // Planar off, screen-space at full strength. `applyQuality` drives both from
    // the tier, so these have to be set after the state change, not before.
    await page.evaluate(() => {
      window.__ocean.water.setReflection(0);
      window.__ocean.ssr!.setStrength(1);
    });

    const look = async () => {
      await page.evaluate(() => window.__ocean.resetDeterministic(28, 90));
      await setCamera(page, [28, 4.5, 22], [0, 3, 0]);
      return capture(page);
    };

    const withShip = await look();
    await page.evaluate(() => {
      const ship = window.__ocean.scene.getObjectByName('ship');
      if (ship) ship.visible = false;
    });
    const withoutShip = await look();
    await page.evaluate(() => {
      const ship = window.__ocean.scene.getObjectByName('ship');
      if (ship) ship.visible = true;
    });

    const half = Math.floor(withShip.height / 2);
    const crop = (image: RgbaImage): RgbaImage => ({
      width: image.width,
      height: image.height - half,
      data: image.data.slice(half * image.width * 4),
    });

    const score = compareImages(crop(withShip), crop(withoutShip));

    // Lower bar than the layered test on purpose: SSR only reflects what is
    // already on screen, so it recovers less of the hull than the mirrored view
    // does. Still far above the measured run-to-run noise floor of ~0.04.
    expect(
      score.meanDeltaE,
      `with the planar layer off, hiding the ship changed the water by mean ΔE ` +
        `${score.meanDeltaE.toFixed(3)}; near zero means the screen-space trace is ` +
        'contributing nothing and the layered test above was passing on planar alone',
    ).toBeGreaterThan(0.15);
  });

  /**
   * Boat mode has to select the *ship*, not just a camera.
   *
   * The HUD advertised W/S throttle and A/D steering from the beginning while
   * `Ship.update` only billowed the sails, so these assertions are the ones that
   * would have caught the gap: signed speed responds to throttle, heading
   * responds to rudder, and neither happens in Orbit.
   */
  test.describe('ship control', () => {
    const drive = async (
      page: import('@playwright/test').Page,
      throttle: number,
      rudder: number,
      seconds: number,
    ) =>
      page.evaluate(
        async ({ t, r, s }) => {
          window.__ocean.setShipInput(t, r);
          await window.__ocean.step(1 / 60, Math.round(s * 60));
          return window.__ocean.shipState();
        },
        { t: throttle, r: rudder, s: seconds },
      );

    const boot = async (page: import('@playwright/test').Page) => {
      await page.goto('/?gate=0&quality=high');
      await waitForOcean(page);
      await page.waitForFunction(() => window.__ocean.shipState() !== null, undefined, {
        timeout: 60_000,
      });
      await setState(page, { preset: 'seaOfThieves', quality: 'high', cameraMode: 'boat' });
      await page.evaluate(() => window.__ocean.resetDeterministic(10, 60));
    };

    test('W drives the ship forward and S drives it astern', async ({ page }) => {
      await boot(page);

      const ahead = await drive(page, 1, 0, 20);
      expect(ahead!.forwardSpeed, 'full throttle produced no headway').toBeGreaterThan(2);

      // From rest, not from ahead — otherwise this only measures deceleration.
      await page.evaluate(() => window.__ocean.resetDeterministic(10, 60));
      const astern = await drive(page, -1, 0, 20);
      expect(astern!.forwardSpeed, 'reverse throttle produced no sternway').toBeLessThan(-0.5);
    });

    test('A and D steer, and the turn needs way on', async ({ page }) => {
      await boot(page);

      const before = await page.evaluate(() => window.__ocean.shipState());
      // A rudder is a foil: hard over from a standstill should do almost nothing.
      const stopped = await drive(page, 0, 1, 6);
      const stoppedTurn = Math.abs(stopped!.heading - before!.heading);
      expect(stoppedTurn, 'the ship turned on the spot with no way on').toBeLessThan(0.2);

      await page.evaluate(() => window.__ocean.resetDeterministic(10, 60));
      const straight = await drive(page, 1, 0, 12);
      const starboard = await drive(page, 1, 1, 10);
      const port = await page.evaluate(async () => {
        await window.__ocean.resetDeterministic(10, 60);
        window.__ocean.setShipInput(1, 0);
        await window.__ocean.step(1 / 60, 720);
        window.__ocean.setShipInput(1, -1);
        await window.__ocean.step(1 / 60, 600);
        return window.__ocean.shipState();
      });

      const delta = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
      const toStarboard = delta(starboard!.heading, straight!.heading);
      const toPort = delta(port!.heading, straight!.heading);

      expect(Math.abs(toStarboard), 'rudder to starboard did not change heading').toBeGreaterThan(0.3);
      expect(Math.abs(toPort), 'rudder to port did not change heading').toBeGreaterThan(0.3);
      expect(
        Math.sign(toStarboard),
        `A and D turned the same way (starboard ${toStarboard.toFixed(2)}, port ${toPort.toFixed(2)})`,
      ).not.toBe(Math.sign(toPort));
    });

    test('other camera modes do not steer the ship', async ({ page }) => {
      await boot(page);
      await setState(page, { cameraMode: 'orbit' });

      const enabled = await page.evaluate(() => window.__ocean.shipControlsEnabled());
      expect(enabled, 'ship input is still live outside Boat mode').toBe(false);

      const before = await page.evaluate(() => window.__ocean.shipState());
      const after = await drive(page, 1, 1, 12);

      expect(
        Math.abs(after!.forwardSpeed),
        'the ship accelerated while the camera was in Orbit',
      ).toBeLessThan(Math.abs(before!.forwardSpeed) + 0.5);
    });

    test('the hull stays finite and afloat while driven through a storm', async ({ page }) => {
      await boot(page);
      await setState(page, { preset: 'storm' });
      await page.evaluate(() => window.__ocean.resetDeterministic(10, 90));

      // Driven hard, turning, in the heaviest sea state the presets offer.
      const state = await drive(page, 1, 0.8, 30);
      const pose = await page.evaluate(() => {
        const ship = window.__ocean.scene.getObjectByName('ship') as unknown as {
          position: { x: number; y: number; z: number };
        };
        return { x: ship.position.x, y: ship.position.y, z: ship.position.z };
      });

      for (const [name, value] of Object.entries({ ...pose, ...state! })) {
        expect(Number.isFinite(value), `${name} went non-finite under load`).toBe(true);
      }
      // Riding the swell, not launched out of it or sunk under it.
      expect(Math.abs(pose.y), `hull settled at y = ${pose.y}`).toBeLessThan(20);
    });

    /**
     * A phone has no W/S/A/D, so Boat mode needs an on-screen stick.
     *
     * Two things have to hold and they fail differently. The pad has to be
     * *there* — built, and shown only in Boat mode on a coarse pointer — and
     * dragging it has to reach the same `ShipController.setInput` the keyboard
     * resolves to. Asserting only the first would pass on a decorative pad, which
     * is the more likely failure of the two.
     *
     * Runs in its own touch-enabled context. `hasTouch` sets
     * `navigator.maxTouchPoints`, which is what `TouchControls.isTouchDevice`
     * reads; the default desktop context deliberately does not build the pad, and
     * a test that forced it into existence would not be testing the gate.
     */
    test('touch devices get an on-screen throttle and rudder that drives the ship', async ({
      browser,
    }) => {
      const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 900, height: 800 },
      });
      const page = await context.newPage();
      try {
        await boot(page);

        // Orbit is the default: no pad, because a stick that steers a ship the
        // viewer is not driving is worse than no stick.
        await setState(page, { cameraMode: 'orbit' });
        expect(
          await page.evaluate(() => window.__ocean.touchControlsVisible()),
          'the pad is showing in Orbit mode',
        ).toBe(false);

        await setState(page, { cameraMode: 'boat' });
        expect(
          await page.evaluate(() => window.__ocean.touchControlsVisible()),
          'no on-screen controls on a touch device in Boat mode',
        ).toBe(true);

        const pad = page.locator('.touchpad');
        await expect(pad).toBeVisible();
        const box = (await pad.boundingBox())!;
        const centreX = box.x + box.width / 2;
        const centreY = box.y + box.height / 2;

        // Push the stick forward and to the right: ahead, starboard rudder.
        await page.mouse.move(centreX, centreY);
        await page.mouse.down();
        await page.mouse.move(centreX + box.width * 0.4, centreY - box.height * 0.4, { steps: 4 });

        // `shipState().throttle` is the *smoothed* value the controller ramps
        // toward the input, and under automation rAF is throttled to about 1 Hz,
        // so the frame loop cannot be relied on to have advanced. Step the sim
        // explicitly — this is testing that the input arrives, not how fast the
        // ramp is.
        await page.evaluate(() => window.__ocean.step(1 / 60, 20));
        const held = await page.evaluate(() => window.__ocean.shipState());
        expect(held, 'no ship controller').not.toBeNull();
        expect(held!.throttle, 'dragging the pad forward did not open the throttle')
          .toBeGreaterThan(0.2);
        expect(held!.rudder, 'dragging the pad right did not put the rudder over')
          .toBeGreaterThan(0.2);

        // A stick springs back. Leaving the rudder latched hard over on release
        // leaves the ship circling, which is exactly what a spring-centred
        // control exists to prevent.
        await page.mouse.up();
        await page.evaluate(() => window.__ocean.step(1 / 60, 40));
        const released = await page.evaluate(() => window.__ocean.shipState());
        expect(Math.abs(released!.throttle), 'throttle stayed open after release')
          .toBeLessThan(0.05);
        expect(Math.abs(released!.rudder), 'rudder stayed over after release')
          .toBeLessThan(0.05);
      } finally {
        await context.close();
      }
    });

    /**
     * Freeboard, measured on the hull's own geometry, against the water that is
     * actually drawn.
     *
     * `the hull stays finite and afloat while driven through a storm` above only
     * asserts `|y| < 20`, which a hull sitting six metres under the surface
     * passes comfortably — and that is exactly the reported defect. "Afloat" is
     * not a statement about the origin's altitude; it is a statement about the
     * hull relative to the water above it, and the two differ by the whole wave
     * field.
     *
     * So this measures what the eye judges: for every corner of the hull's own
     * bounding box, the height of that corner above the surface directly over
     * it. The hull is afloat as long as at least one corner is dry. The box
     * comes from the same meshes `Ship` measures its waterline from — `Ship`
     * puts y = 0 at 31% of the box's height, so the deck edge sits about
     * 0.69 * height above the origin and a buried deck is unambiguous.
     *
     * **Two water lines, and they are not the same water line.** The cheap one
     * is `sampler.height()`, which is what the physics floats the hull on. The
     * expensive one is rebuilt here from the displacement targets themselves —
     * the same fields the surface is drawn from, inverted the same way — and it
     * is the one the assertions use, because a hull can only *look* submerged
     * against water that is on screen. Reporting both is the point: the gap
     * between them is a defect of its own, and a test that consulted only the
     * sampler would certify a hull the viewer can plainly see underwater.
     */
    const measureFreeboard = async (
      page: import('@playwright/test').Page,
      options: { steps: number; throttle: number; rudder: number; renderEvery: number },
    ) =>
      page.evaluate(async (o) => {
        interface Vec3Like {
          x: number;
          y: number;
          z: number;
        }
        interface Node3 {
          isMesh?: boolean;
          name: string;
          parent: { name: string } | null;
          matrixWorld: { elements: ArrayLike<number> };
          position: Vec3Like;
          quaternion: { x: number; y: number; z: number; w: number };
          geometry?: {
            computeBoundingBox(): void;
            boundingBox: { min: Vec3Like; max: Vec3Like } | null;
          };
          traverse(callback: (node: Node3) => void): void;
          updateMatrixWorld(force?: boolean): void;
        }
        const hooks = window.__ocean as unknown as {
          scene: { getObjectByName(name: string): Node3 | undefined };
          sampler: { ready: boolean; height(x: number, z: number): number };
          simulation: { displacementTargets: { width: number }[]; tileSizes: number[] };
          renderer: {
            readRenderTargetPixelsAsync(
              target: unknown, x: number, y: number, width: number, height: number,
            ): Promise<ArrayLike<number>>;
          };
          step(dt: number, steps?: number): Promise<void>;
          setShipInput(throttle: number, rudder: number): void;
        };

        const ship = hooks.scene.getObjectByName('ship');
        if (!ship) throw new Error('no object named "ship" in the scene');
        ship.updateMatrixWorld(true);

        // --- the surface as drawn --------------------------------------------
        // The displacement targets are RGBA16F; reading the raw Uint16 as a
        // number gives tens of thousands of metres instead of tenths of one.
        const halfToFloat = (bits: number) => {
          const sign = bits & 0x8000 ? -1 : 1;
          const exponent = (bits & 0x7c00) >> 10;
          const mantissa = bits & 0x03ff;
          if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
          if (exponent === 31) return mantissa ? NaN : sign * Infinity;
          return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
        };
        const targets = hooks.simulation.displacementTargets;
        const tileSizes = hooks.simulation.tileSizes;
        const fieldSize = targets[0].width;
        // Allocated once; the read runs hundreds of times.
        const fields = targets.map(() => new Float64Array(fieldSize * fieldSize * 3));
        const readFields = async () => {
          for (let t = 0; t < targets.length; t++) {
            const raw = await hooks.renderer.readRenderTargetPixelsAsync(
              targets[t], 0, 0, fieldSize, fieldSize,
            );
            const half =
              (raw as ArrayLike<number> & { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT === 2;
            const field = fields[t];
            for (let i = 0; i < fieldSize * fieldSize; i++) {
              field[i * 3] = half ? halfToFloat(raw[i * 4]) : raw[i * 4];
              field[i * 3 + 1] = half ? halfToFloat(raw[i * 4 + 1]) : raw[i * 4 + 1];
              field[i * 3 + 2] = half ? halfToFloat(raw[i * 4 + 2]) : raw[i * 4 + 2];
            }
          }
        };
        const texel = [0, 0, 0];
        const bilinear = (field: Float64Array, u: number, v: number) => {
          const fx = (((u % 1) + 1) % 1) * fieldSize - 0.5;
          const fz = (((v % 1) + 1) % 1) * fieldSize - 0.5;
          const x0 = Math.floor(fx);
          const z0 = Math.floor(fz);
          const tx = fx - x0;
          const tz = fz - z0;
          const wrap = (n: number) => ((n % fieldSize) + fieldSize) % fieldSize;
          const x0w = wrap(x0);
          const x1w = wrap(x0 + 1);
          const z0w = wrap(z0);
          const z1w = wrap(z0 + 1);
          for (let c = 0; c < 3; c++) {
            const a = field[(z0w * fieldSize + x0w) * 3 + c];
            const b = field[(z0w * fieldSize + x1w) * 3 + c];
            const d = field[(z1w * fieldSize + x0w) * 3 + c];
            const e = field[(z1w * fieldSize + x1w) * 3 + c];
            texel[c] = (a + (b - a) * tx) * (1 - tz) + (d + (e - d) * tx) * tz;
          }
        };
        const total = [0, 0, 0];
        const displacementAt = (x: number, z: number) => {
          total[0] = 0;
          total[1] = 0;
          total[2] = 0;
          for (let i = 0; i < fields.length; i++) {
            bilinear(fields[i], x / tileSizes[i], z / tileSizes[i]);
            total[0] += texel[0];
            total[1] += texel[1];
            total[2] += texel[2];
          }
        };
        // The surface is choppy — the vertex at grid point p is drawn at
        // p + D(p) — so the height *above* a world point needs that map
        // inverted. The same fixed-point iteration `OceanSampler.height` uses,
        // so the two numbers differ only where the underlying data does.
        const renderedHeight = (x: number, z: number) => {
          let gx = x;
          let gz = z;
          for (let i = 0; i < 4; i++) {
            displacementAt(gx, gz);
            gx = x - total[0];
            gz = z - total[2];
          }
          displacementAt(gx, gz);
          return total[1];
        };

        // --- the hull box -----------------------------------------------------
        // From the hull meshes alone. Including the rigging would put the
        // mastheads into it and nothing would ever read as submerged. Corners
        // are kept in each mesh's own space and transformed per sample, so a
        // rolled hull is measured exactly rather than through an inflated
        // world-axis-aligned box.
        const localCorners: number[][] = [];
        const matrices: { elements: ArrayLike<number> }[] = [];
        let boxMinY = Infinity;
        let boxMaxY = -Infinity;
        ship.traverse((node) => {
          if (!node.isMesh || !node.geometry) return;
          const label = `${node.name} ${node.parent ? node.parent.name : ''}`.toLowerCase();
          if (!label.includes('hull') && !label.includes('base')) return;
          node.geometry.computeBoundingBox();
          const box = node.geometry.boundingBox;
          if (!box) return;
          const flat: number[] = [];
          for (const x of [box.min.x, box.max.x]) {
            for (const y of [box.min.y, box.max.y]) {
              for (const z of [box.min.z, box.max.z]) flat.push(x, y, z);
            }
          }
          localCorners.push(flat);
          matrices.push(node.matrixWorld);
        });
        if (localCorners.length === 0) throw new Error('no hull mesh found under "ship"');

        hooks.setShipInput(o.throttle, o.rudder);

        let worstFreeboard = Infinity;
        let worstAtStep = -1;
        let freeboardSum = 0;
        let submergedSamples = 0;
        let worstRendered = Infinity;
        let worstRenderedAtStep = -1;
        let renderedSum = 0;
        let renderedSubmerged = 0;
        let renderedSamples = 0;
        let disagreementSquares = 0;
        let worstDisagreement = 0;
        let ySum = 0;
        let minY = Infinity;
        let maxY = -Infinity;
        let minRoll = Infinity;
        let maxRoll = -Infinity;
        let minPitch = Infinity;
        let maxPitch = -Infinity;
        // The sea the measurement was actually taken in. Without it a passing
        // run is unfalsifiable: "the hull kept its freeboard" means nothing if
        // the wave field turned out to be a millpond.
        let waveSum = 0;
        let waveSquares = 0;
        let waveMin = Infinity;
        let waveMax = -Infinity;

        for (let i = 0; i < o.steps; i++) {
          await hooks.step(1 / 60, 1);
          ship.updateMatrixWorld(true);

          // Reading three full displacement targets costs tens of milliseconds,
          // so the drawn surface is consulted on a stride while the sampler —
          // which is free — is consulted every step.
          const useRendered = i % o.renderEvery === 0;
          if (useRendered) await readFields();

          // Highest dry corner: the hull is under only when even this is wet.
          let highest = -Infinity;
          let highestRendered = -Infinity;
          let deckTop = -Infinity;
          for (let m = 0; m < localCorners.length; m++) {
            const e = matrices[m].elements;
            const c = localCorners[m];
            for (let k = 0; k < c.length; k += 3) {
              const lx = c[k];
              const ly = c[k + 1];
              const lz = c[k + 2];
              const wx = e[0] * lx + e[4] * ly + e[8] * lz + e[12];
              const wy = e[1] * lx + e[5] * ly + e[9] * lz + e[13];
              const wz = e[2] * lx + e[6] * ly + e[10] * lz + e[14];
              if (wy > deckTop) deckTop = wy;
              const freeboard = wy - hooks.sampler.height(wx, wz);
              if (freeboard > highest) highest = freeboard;
              if (useRendered) {
                const drawn = wy - renderedHeight(wx, wz);
                if (drawn > highestRendered) highestRendered = drawn;
              }
            }
          }
          if (deckTop > boxMaxY) boxMaxY = deckTop;
          if (deckTop < boxMinY) boxMinY = deckTop;

          freeboardSum += highest;
          if (highest < worstFreeboard) {
            worstFreeboard = highest;
            worstAtStep = i;
          }
          if (highest <= 0) submergedSamples++;

          if (useRendered) {
            renderedSamples++;
            renderedSum += highestRendered;
            if (highestRendered < worstRendered) {
              worstRendered = highestRendered;
              worstRenderedAtStep = i;
            }
            if (highestRendered <= 0) renderedSubmerged++;
            const gap = highest - highestRendered;
            disagreementSquares += gap * gap;
            if (Math.abs(gap) > Math.abs(worstDisagreement)) worstDisagreement = gap;
          }

          const y = ship.position.y;
          ySum += y;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;

          const here = hooks.sampler.height(ship.position.x, ship.position.z);
          waveSum += here;
          waveSquares += here * here;
          if (here < waveMin) waveMin = here;
          if (here > waveMax) waveMax = here;

          const q = ship.quaternion;
          // Local +X is the bow and local +Z is starboard; the world-y component
          // of each rotated axis is the sine of pitch and of roll.
          const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (q.x * q.y + q.w * q.z))));
          const roll = Math.asin(Math.max(-1, Math.min(1, 2 * (q.y * q.z - q.w * q.x))));
          if (pitch < minPitch) minPitch = pitch;
          if (pitch > maxPitch) maxPitch = pitch;
          if (roll < minRoll) minRoll = roll;
          if (roll > maxRoll) maxRoll = roll;
        }

        const degrees = 180 / Math.PI;
        const waveMean = waveSum / o.steps;
        return {
          samples: o.steps,
          renderedSamples,
          // 4 * sigma is the significant wave height, the number sea states are
          // quoted in.
          significantWaveHeight:
            4 * Math.sqrt(Math.max(0, waveSquares / o.steps - waveMean * waveMean)),
          waveMin,
          waveMax,
          worstFreeboard,
          worstAtSeconds: worstAtStep / 60,
          meanFreeboard: freeboardSum / o.steps,
          submergedFraction: submergedSamples / o.steps,
          worstRenderedFreeboard: worstRendered,
          worstRenderedAtSeconds: worstRenderedAtStep / 60,
          meanRenderedFreeboard: renderedSum / Math.max(1, renderedSamples),
          renderedSubmergedFraction: renderedSubmerged / Math.max(1, renderedSamples),
          waterLineDisagreementRms: Math.sqrt(disagreementSquares / Math.max(1, renderedSamples)),
          worstWaterLineDisagreement: worstDisagreement,
          meanY: ySum / o.steps,
          minY,
          maxY,
          highestDeckPoint: boxMaxY,
          lowestDeckPoint: boxMinY,
          rollAmplitudeDeg: ((maxRoll - minRoll) / 2) * degrees,
          pitchAmplitudeDeg: ((maxPitch - minPitch) / 2) * degrees,
        };
      }, options);

    test('the hull keeps freeboard driven through the heaviest sea', async ({ page }) => {
      await boot(page);
      // The heaviest sea this app can actually be put into, which is not the
      // one it looks like from the preset table. `windSpeed` does not appear in
      // `spectrumAmplitude` at all — it only sets the wind *axis*, the foam
      // rate, the glitter stretch and the clouds. Wave height comes entirely
      // from `peakWavelength`, because JONSWAP energy at the peak goes as
      // g^2 / omega^5: at the storm preset's own 60 m the field measures
      // Hs 2.5 m, and at the slider's maximum of 150 m it measures Hs 6.7 m
      // with crests over 5 m — which is the first sea in range of the ship's
      // 5.2 m of freeboard. Both sliders are pushed anyway, because the wind
      // still decides which way the sea runs relative to the hull.
      await setState(page, { preset: 'storm', windSpeed: 25, peakWavelength: 150 });
      await page.evaluate(() =>
        window.__ocean.resetDeterministic(10, 120, { throttle: 1, rudder: 0.6 }),
      );

      // Full ahead and hard over, so the hull takes the sea on the bow, the
      // beam and the quarter in turn over ~40 s — around six peak periods.
      const report = await measureFreeboard(page, {
        steps: 2400,
        throttle: 1,
        rudder: 0.6,
        renderEvery: 8,
      });

      const summary =
        `Hs ${report.significantWaveHeight.toFixed(2)} m (surface under the hull ` +
        `${report.waveMin.toFixed(2)}..${report.waveMax.toFixed(2)} m); ` +
        `deck top ${report.lowestDeckPoint.toFixed(2)}..${report.highestDeckPoint.toFixed(2)} m; ` +
        `drawn water line: worst freeboard ${report.worstRenderedFreeboard.toFixed(2)} m at t+` +
        `${report.worstRenderedAtSeconds.toFixed(1)} s, mean ` +
        `${report.meanRenderedFreeboard.toFixed(2)} m, fully submerged on ` +
        `${(report.renderedSubmergedFraction * 100).toFixed(1)}% of ${report.renderedSamples} ` +
        `samples; physics water line: worst ${report.worstFreeboard.toFixed(2)} m at t+` +
        `${report.worstAtSeconds.toFixed(1)} s, mean ${report.meanFreeboard.toFixed(2)} m; ` +
        `the two disagree by rms ${report.waterLineDisagreementRms.toFixed(2)} m, worst ` +
        `${report.worstWaterLineDisagreement.toFixed(2)} m; ` +
        `y in [${report.minY.toFixed(2)}, ${report.maxY.toFixed(2)}], ` +
        `roll +-${report.rollAmplitudeDeg.toFixed(1)} deg`;

      // Printed, not just asserted: these are measurements, and the numbers are
      // the only way to tell "comfortably afloat" from "one wave away from the
      // failure this test exists for".
      console.log(`[freeboard/storm] ${summary}`);

      expect(
        Number.isFinite(report.worstRenderedFreeboard) && Number.isFinite(report.worstFreeboard),
        `freeboard went non-finite: ${summary}`,
      ).toBe(true);
      // The whole assertion: at every instant *some* part of the hull is above
      // the water directly over it. Not "the origin stayed near zero".
      expect(
        report.worstRenderedFreeboard,
        `the hull went completely under the water that is drawn — ${summary}`,
      ).toBeGreaterThan(0);
      expect(
        report.worstFreeboard,
        `the hull went completely under the water the physics sees — ${summary}`,
      ).toBeGreaterThan(0);
    });

    /**
     * The counterweight to the test above.
     *
     * Every cheap way to stop a hull submerging — more buoyancy, a stiffer
     * spring, a floor under the position — also lifts it out of the water in a
     * normal sea, and a ship riding on top of the surface is a worse defect than
     * one that occasionally ducks under, because it is visible in every frame.
     * So the rest waterline and the motion in a moderate sea are pinned here,
     * with bounds tight enough that raising `buoyancyStrength` would break them.
     */
    test('a moderate sea leaves the hull on its designed waterline', async ({ page }) => {
      await boot(page);
      await page.evaluate(() =>
        window.__ocean.resetDeterministic(10, 120, { throttle: 0.5, rudder: 0 }),
      );

      const report = await measureFreeboard(page, {
        steps: 1200,
        throttle: 0.5,
        rudder: 0,
        renderEvery: 8,
      });

      const summary =
        `Hs ${report.significantWaveHeight.toFixed(2)} m; mean y ${report.meanY.toFixed(3)} m, ` +
        `y in [${report.minY.toFixed(2)}, ${report.maxY.toFixed(2)}], mean freeboard ` +
        `${report.meanFreeboard.toFixed(3)} m (drawn water line ` +
        `${report.meanRenderedFreeboard.toFixed(3)} m), worst ` +
        `${report.worstFreeboard.toFixed(3)} m (drawn ` +
        `${report.worstRenderedFreeboard.toFixed(3)} m), roll +-` +
        `${report.rollAmplitudeDeg.toFixed(3)} deg, pitch +-` +
        `${report.pitchAmplitudeDeg.toFixed(3)} deg`;

      console.log(`[freeboard/moderate] ${summary}`);

      // The design waterline is y = 0 by construction — `Ship` shifts the model
      // so that it is. A hull floating high shows up here first.
      expect(report.meanY, `the hull is not sitting on its designed waterline — ${summary}`)
        .toBeGreaterThan(-0.75);
      expect(report.meanY, `the hull is riding high out of the water — ${summary}`)
        .toBeLessThan(0.75);
      // Still a live, moving hull rather than one nailed to the surface.
      expect(report.rollAmplitudeDeg, `the hull stopped rolling — ${summary}`)
        .toBeGreaterThan(0.5);
      expect(report.rollAmplitudeDeg, `the hull is bobbing like a cork — ${summary}`)
        .toBeLessThan(25);
      expect(report.worstFreeboard, `the hull went under in a moderate sea — ${summary}`)
        .toBeGreaterThan(0);
      expect(
        report.worstRenderedFreeboard,
        `the hull went under the drawn water line in a moderate sea — ${summary}`,
      ).toBeGreaterThan(0);
    });
  });

  /**
   * The flock has to be *in* the frame, not merely in the scene graph.
   *
   * Birds are the easiest kind of feature to ship broken: an instanced draw with
   * every transform derived on the GPU will construct, add to the scene, report a
   * healthy instance count and render nothing at all if a single uniform is
   * unset. Counting objects would pass on that; comparing the rendered frame with
   * the flock hidden against the same frame with it shown cannot.
   *
   * The camera is aimed *up*, away from the water, so the only thing that can
   * differ between the two captures is the sky and what is flying in it.
   */
  test('the bird flock renders, and the tier scales it', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    // A preset with a dark sky, so a pale gull against it is a large per-pixel
    // difference rather than white-on-white. Up-sun on a clear day the flock is
    // genuinely hard to see, which is a real limitation of the shot and not of
    // the flock.
    await setState(page, { preset: 'moonlit', quality: 'ultra' });
    // Looking up and out across the flock's altitude band: the circuits sit
    // 9-62 m up and 35-142 m out from the anchor.
    await setCamera(page, [0, 26, 0], [40, 64, 40]);
    await page.evaluate(() => window.__ocean.resetDeterministic(30, 60));

    const withFlock = await capture(page);
    await page.evaluate(() => {
      const birds = window.__ocean.scene.getObjectByName('birds');
      if (birds) birds.visible = false;
    });
    const without = await capture(page);
    await page.evaluate(() => {
      const birds = window.__ocean.scene.getObjectByName('birds');
      if (birds) birds.visible = true;
    });

    let changed = 0;
    for (let i = 0; i < withFlock.data.length; i += 4) {
      if (Math.abs(withFlock.data[i] - without.data[i]) > 6) changed++;
    }

    // The bar is 80, and it is low on purpose.
    //
    // Measured, this frame moves ~135 pixels out of 1.44 million: forty gulls at
    // 60-140 m are three or four pixels each. That is what a flock at that
    // distance *is*, and inflating the birds until the number looks impressive
    // would be tuning the scene to the test. What the assertion has to separate
    // is "drawn small" from "not drawn", and a disconnected flock gives exactly
    // zero — the gap between 0 and 135 is the whole signal.
    expect(
      changed,
      `hiding the flock changed ${changed} pixels; near zero means the birds are ` +
        'in the scene graph and not in the image',
    ).toBeGreaterThan(80);

    // And the tier is a real cost lever, not a label: Low flies none.
    //
    // Compared against Low *with the flock hidden*, not against the Ultra capture
    // above. A tier change moves the cascade count, the fog, the mesh density and
    // the shadow map, so an across-tier diff is a million pixels of everything
    // else — the first version of this assertion measured exactly that and read
    // it as birds.
    await setState(page, { quality: 'low' });
    await page.evaluate(() => window.__ocean.resetDeterministic(30, 60));
    const lowWith = await capture(page);
    await page.evaluate(() => {
      const birds = window.__ocean.scene.getObjectByName('birds');
      if (birds) birds.visible = false;
    });
    const lowWithout = await capture(page);

    let lowChanged = 0;
    for (let i = 0; i < lowWith.data.length; i += 4) {
      if (Math.abs(lowWith.data[i] - lowWithout.data[i]) > 6) lowChanged++;
    }
    expect(
      lowChanged,
      `Low still drew ${lowChanged} pixels of birds; the tier sets the count to zero, ` +
        'so hiding them there must change nothing at all',
    ).toBeLessThan(20);
  });

  /**
   * Rain wets what it lands on.
   *
   * The hull was listed as a known limitation for exactly as long as rain has
   * existed: it reached the water and the lens, but wood and canvas were as dry
   * in a squall as at noon.
   *
   * The assertion is on the materials rather than on a pixel count, because the
   * effect is a pair of scalar multipliers and a screenshot would mostly be
   * measuring whatever else the storm preset changed. What it does check is the
   * thing most likely to break: `SurfaceWetness` duck-types the materials it
   * adopts, and the asset loader converts everything to
   * `MeshPhysicalNodeMaterial` — a flag test against `isMeshStandardMaterial`
   * adopts nothing at all and leaves the whole effect silently inert.
   */
  /**
   * Wetting has to reach the image, and it has to vary over the object.
   *
   * This used to read `material.roughness` and `material.color` off the CPU,
   * because wetting was a scalar written onto every material. It is a node graph
   * now — evaluated per fragment against the geometric world normal, so the deck
   * soaks while the underside of a beam stays dry — and the only place the result
   * exists is the rendered image. Reading the scalars would now pass or fail on
   * an implementation detail rather than on whether the ship looks wet.
   *
   * So it photographs the hull instead, driving wetness directly through
   * `setSurfaceWetness` with the preset, camera, sea state and lighting all held
   * fixed. Two things are then true of a wet hull that are not true of a dry one:
   * it is darker, and — the part the old scalar version could not express — the
   * darkening is *not uniform*, because a surface turned away from the sky does
   * not collect rain. A uniform albedo multiplier scales every pixel by the same
   * factor whatever its texture, so the wet/dry ratio would be constant across
   * the hull; the normal-driven mask spreads it. Both assertions fail if the
   * effect is disconnected, and the second one fails if it regresses to uniform.
   */
  test('rain darkens the hull, unevenly, and it dries afterwards', async ({ page }) => {
    // The drying leg steps 900 frames of a scene that now carries an island,
    // and deterministic stepping is synchronous with the GPU — so this test's
    // cost is set by how much there is to draw, not by how much it asserts.
    test.setTimeout(600_000);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    // Close on the hull under a clear sky, so the ship fills enough of the frame
    // to measure and nothing about the weather moves between the two captures.
    await setState(page, { preset: 'skyPro', cameraMode: 'orbit' });
    await page.evaluate(() => window.__ocean.resetDeterministic(10, 60));
    await setCamera(page, [14, 5, 16], [0, 2.5, 0]);

    // No `step` between the two: `capturePixels` renders a fresh frame from the
    // current state, so stepping would advance the waves, clouds and spray as
    // well and the difference would be dominated by ordinary motion rather than
    // by wetting. Holding the clock still is what makes this a measurement of
    // one variable.
    const shoot = async (wetness: number) => {
      await page.evaluate((w: number) => window.__ocean.setSurfaceWetness(w), wetness);
      return capture(page);
    };

    const dry = await shoot(0);
    const wet = await shoot(1);

    // Pixels the wetting actually moved. Sea and sky are unaffected by it, so
    // this selects the ship without needing to know where on screen it is.
    const ratios: number[] = [];
    let darkened = 0;
    for (let i = 0; i < dry.width * dry.height; i++) {
      const o = i * 4;
      const a = 0.2126 * dry.data[o] + 0.7152 * dry.data[o + 1] + 0.0722 * dry.data[o + 2];
      const b = 0.2126 * wet.data[o] + 0.7152 * wet.data[o + 1] + 0.0722 * wet.data[o + 2];
      // A floor on the dry value keeps the ratio meaningful: a pixel at level 2
      // can halve on rounding alone and would otherwise dominate the spread.
      if (a < 24 || Math.abs(a - b) < 2) continue;
      ratios.push(b / a);
      if (b < a) darkened++;
    }

    expect(
      ratios.length,
      'wetting changed nothing on screen, so it is not reaching the image at all',
    ).toBeGreaterThan(400);

    const mean = ratios.reduce((t, r) => t + r, 0) / ratios.length;
    const spread = Math.sqrt(
      ratios.reduce((t, r) => t + (r - mean) * (r - mean), 0) / ratios.length,
    );
    // The median, not the mean, carries the darkening. Wetting does two things
    // at once and they pull opposite ways in luminance: the albedo drops, and the
    // roughness drops too, which *brightens* every pixel that catches a specular
    // highlight. Those highlights are a minority of the surface and several times
    // the brightness of the rest, so they drag the mean back toward 1 while the
    // typical pixel is clearly darker. Asserting on the mean would be asserting
    // that the gloss is weak.
    const sorted = [...ratios].sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];

    console.log(
      `[wetness] n=${ratios.length} median=${median.toFixed(4)} mean=${mean.toFixed(4)} ` +
        `spread=${spread.toFixed(4)} darker=${((darkened / ratios.length) * 100).toFixed(0)}%`,
    );
    expect(median, `median wet/dry luminance ratio was ${median.toFixed(3)}`).toBeLessThan(0.97);
    expect(
      spread,
      `wet/dry ratio spread was ${spread.toFixed(4)} across ${ratios.length} pixels — ` +
        'a uniform multiplier would give ~0, so the wetting is not normal-driven',
    ).toBeGreaterThan(0.02);

    // Drying is still a CPU-side time constant, and still worth asserting: 26 s
    // means this is a partial recovery on purpose, and asserting it returns
    // exactly to dry would be asserting the wrong physics.
    await page.evaluate(() => window.__ocean.setSurfaceWetness(1));
    await setState(page, { preset: 'skyPro' });
    await page.evaluate(() => window.__ocean.step(1 / 30, 900));
    const dried = await page.evaluate(() => window.__ocean.surfaceWetness());
    expect(dried, `wetness stayed at ${dried.toFixed(3)} after 30 s of clear weather`).toBeLessThan(
      0.85,
    );
  });

  /**
   * Rain has to reach the water.
   *
   * It was an overlay: a camera-following particle volume that disturbed
   * nothing. The distinguishing test is not whether rain is drawn — it always
   * was — but whether the *surface* changes when it falls, and whether that
   * change goes away when the rain stops.
   */
  test('rain disturbs the ocean surface, and the surface recovers', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    test.skip(
      !(await hasGpuAdapter(page)),
      'no GPU adapter: the software path cannot render these frames in time',
    );

    await setState(page, { preset: 'storm', quality: 'high' });

    // Low over the water, where impact rings are decimetre features and legible.
    // The rain particles themselves are excluded by hiding them, so what is
    // being measured is the surface and not the curtain in front of it.
    const look = async (rain: number) =>
      page.evaluate(async (intensity) => {
        const ocean = window.__ocean;
        const weather = ocean.scene.getObjectByName('weather');
        if (weather) weather.visible = false;
        // Before the reset, not after. The settle run is what fills the foam
        // buffer, so setting the rate afterwards would settle the world under
        // the *previous* call's rain and then photograph it under this one's.
        ocean.setRainOverride(intensity);
        await ocean.resetDeterministic(35, 60);
        ocean.setCamera(0, 4.5, 26, 0, 1.5, 0);
        await ocean.step(1 / 60, 30);
        await ocean.capturePixels();
        await ocean.capturePixels();
        return ocean.capturePixels();
      }, rain);

    const dry = await look(0);
    const light = await look(0.35);
    const heavy = await look(1);
    const dryAgain = await look(0);

    const atLight = compareImages(dry as never, light as never);
    const atHeavy = compareImages(dry as never, heavy as never);
    const recovered = compareImages(dry as never, dryAgain as never);

    expect(
      atHeavy.meanDeltaE,
      `rain changed the surface by mean ΔE ${atHeavy.meanDeltaE.toFixed(3)}; ` +
        'near zero means impacts are not reaching the water',
    ).toBeGreaterThan(0.25);

    // Monotonic, not merely different.
    //
    // Asserting only that rain changes the surface is too weak to be worth
    // running: it passed while the impact mask was *inverted*, activating cells
    // as the rain eased, because an inverted mask changes the water just as much
    // as a correct one. Heavier rain has to disturb the surface more than
    // lighter rain, which an inversion cannot satisfy.
    expect(
      atHeavy.meanDeltaE,
      `heavy rain (ΔE ${atHeavy.meanDeltaE.toFixed(3)}) did not disturb the surface ` +
        `more than light rain (ΔE ${atLight.meanDeltaE.toFixed(3)}) — the impact ` +
        'mask may be inverted',
    ).toBeGreaterThan(atLight.meanDeltaE);

    expect(
      recovered.meanDeltaE,
      `the surface did not return to its dry state after the rain stopped ` +
        `(mean ΔE ${recovered.meanDeltaE.toFixed(3)})`,
    ).toBeLessThan(atHeavy.meanDeltaE * 0.25);
  });

  test('camera modes switch via keyboard', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    // '4' included: a mode that is selectable by button and not by key is a
    // mode half of the viewers cannot reach.
    for (const [key, expected] of [
      ['2', 'fly'],
      ['3', 'boat'],
      ['4', 'cinematic'],
      ['1', 'orbit'],
    ] as const) {
      await page.keyboard.press(key);
      await page.waitForTimeout(250);
      const mode = await page.evaluate(
        () =>
          (window as unknown as { __ocean: { director: { currentMode: string } } }).__ocean.director
            .currentMode,
      );
      expect(mode).toBe(expected);
    }
  });

  test('quality tiers apply cleanly and rebuild GPU resources', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    for (const quality of ['low', 'medium', 'high', 'ultra', 'max', 'high']) {
      await setState(page, { quality });
    }

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);
  });

  /**
   * Shadow and reflection resolution have to *follow* the tier, not just survive
   * it.
   *
   * Both were latched shut for a long time — honoured once at boot, ignored
   * afterwards — as a workaround for a crash that turned out to be a GPU
   * resource-lifetime race, now fixed properly by the drain in
   * `App.drainQualityRequests`. The workaround was invisible to every existing
   * test: the tier-change test above only asserts that nothing throws, so it
   * passed just as happily with both setters returning on their first line.
   *
   * This asserts the resolutions actually move, and it moves *up* then *down*,
   * because a setter that reallocates only when growing would pass a one-way
   * check. Low boots first so the interesting direction is exercised on a
   * session that started small — the case the latch was hiding.
   */
  test('shadow and reflection resolution follow the tier', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    const sample = async (quality: string) => {
      await setState(page, { quality });
      return page.evaluate(() => ({
        shadow: window.__ocean.atmosphere.shadowMapSize,
        // Null on the WebGL2 path, which has no planar reflector at all.
        reflection: window.__ocean.reflections?.resolutionScale ?? null,
      }));
    };

    const low = await sample('low');
    const max = await sample('max');
    const backDown = await sample('low');

    expect(max.shadow, `low=${low.shadow} max=${max.shadow}`).toBeGreaterThan(low.shadow);
    expect(backDown.shadow, `max=${max.shadow} back=${backDown.shadow}`).toBe(low.shadow);

    expect(max.reflection).not.toBeNull();
    expect(
      max.reflection as number,
      `low=${low.reflection} max=${max.reflection}`,
    ).toBeGreaterThan(low.reflection as number);
    expect(backDown.reflection).toBe(low.reflection);

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);
  });

  /**
   * The cinematic mode has to be a *mode*, not a camera animation.
   *
   * The distinction is the whole design: the flight publishes engine orders and
   * the ship sails itself along the tour, so the wake, the buoyancy and the spray
   * are the ones the physics produced. Teleporting the hull along a curve would
   * look identical in a single screenshot and wrong in every frame after it — the
   * hull would slide without heeling and tow no wake.
   *
   * So this asserts on the hull rather than on the camera: it must be under
   * power and moving, and it must be released cleanly when the viewer takes over.
   * Handing back a ship still holding the tour's throttle was a real hazard here,
   * because `setInput` is a latch that disabling the controller does not clear.
   */
  test('cinematic mode sails the ship and releases it cleanly', async ({ page }) => {
    // Deterministic stepping is synchronous with the GPU, so this test's cost is
    // set by how loaded the machine is rather than by how much it does. It runs
    // in 35 s alone and timed out at four minutes inside the full suite.
    test.setTimeout(600_000);
    const errors = collectConsoleErrors(page);
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    await setState(page, { cameraMode: 'cinematic' });
    expect(await page.evaluate(() => window.__ocean.director.currentMode)).toBe('cinematic');

    await page.evaluate(() => window.__ocean.resetDeterministic(0, 30));
    const early = await page.evaluate(() => ({
      orders: { ...window.__ocean.director.shipInput },
      beat: window.__ocean.director.cinematicBeat,
      ship: window.__ocean.shipState(),
    }));
    expect(early.orders.throttle, 'the flight is not calling for any power').toBeGreaterThan(0.1);

    // 1200 steps — twenty seconds — and the number is load-bearing.
    //
    // This was 420 steps, which is seven, described in its own message as ten.
    // The opening beat is fourteen seconds long, so seven never left it: the
    // assertion below passed on a floating-point artefact instead. The settle
    // left the loop clock at 119.99999999999997 against a 120 s lap — a hair
    // short of the wrap — so `beatAt` reported the *last* beat, and stepping
    // seven seconds into `open-water` therefore looked like a beat change. Re-
    // authoring the flight to 166 s changed that accumulation by one ulp, the
    // clock landed exactly on the wrap, and the test failed with everything
    // about the tour working correctly.
    //
    // Twenty seconds genuinely crosses the first boundary, whatever the lap
    // length and wherever the reset happens to land.
    await page.evaluate(() => window.__ocean.step(1 / 60, 1200));
    const later = await page.evaluate(() => ({
      beat: window.__ocean.director.cinematicBeat,
      ship: window.__ocean.shipState(),
    }));

    expect(
      later.ship?.forwardSpeed ?? 0,
      `hull speed was ${(later.ship?.forwardSpeed ?? 0).toFixed(2)} m/s after 20 s of tour`,
    ).toBeGreaterThan(1);
    expect(later.beat, `the tour stayed on beat "${early.beat}" for 20 s`).not.toBe(early.beat);

    // Hand back to the viewer, and assert *immediately*.
    //
    // An earlier version of this stepped five seconds first, on the reasoning
    // that the spooled throttle cannot drop instantly. That reasoning is sound
    // and the test was still wrong: throttle spools at 0.7 per second, so full
    // ahead decays on its own inside about a second and a half, and waiting five
    // proved only that a first-order lag is a first-order lag. It passed against
    // a handoff that did *not* clear the spool, which is exactly the defect it
    // was written to catch. Taking the helm must hand over a stopped engine on
    // the same frame.
    await setState(page, { cameraMode: 'boat' });
    const released = await page.evaluate(() => window.__ocean.shipState());
    expect(
      Math.abs(released?.throttle ?? 1),
      `Boat mode inherited throttle ${released?.throttle} from the tour`,
    ).toBeLessThan(1e-6);
    expect(
      Math.abs(released?.rudder ?? 1),
      `Boat mode inherited rudder ${released?.rudder} from the tour`,
    ).toBeLessThan(1e-6);

    // And the keys must be inert *during* the tour, which is the other half of
    // "the flight holds the wheel". The keyboard deliberately outranks
    // `setInput` so a viewer at the helm beats the on-screen throttle; applied
    // during a cinematic that rule let a held S command full astern against a
    // full-ahead beat.
    await setState(page, { cameraMode: 'cinematic' });
    await page.evaluate(() => window.__ocean.resetDeterministic(0, 30));
    await page.keyboard.down('s');
    // 90 steps is 1.5 s, and the throttle spools at 0.7 per second — long enough
    // for a working S key to have driven it hard astern, which is the thing this
    // has to be able to see.
    await page.evaluate(() => window.__ocean.step(1 / 60, 90));
    const underKey = await page.evaluate(() => window.__ocean.shipState());
    await page.keyboard.up('s');
    expect(
      underKey?.throttle ?? 0,
      `holding S during the tour drove the throttle to ${underKey?.throttle}`,
    ).toBeGreaterThan(0.5);

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);
  });

  /**
   * The tour loops, so its ends have to meet.
   *
   * A cut at the wrap is the one artefact a looping camera cannot hide, and it is
   * invisible to any test that only samples the middle. `resetCinematic` makes
   * the loop addressable, so this simply asks where the camera is just before the
   * seam and just after it, and requires the two to be a frame apart rather than
   * a jump apart.
   */
  test('the cinematic loop closes without a cut', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);
    await setState(page, { cameraMode: 'cinematic' });

    // Run the loop continuously across the seam rather than resetting either
    // side of it and comparing. Resetting and snapping measures the camera's own
    // settling transient, not the curve: the rig eases toward the pose the
    // director publishes, so the first frame after a reset is dominated by that
    // ease and reads as a metre-scale jump wherever the reset landed.
    //
    // Stepping through the wrap the way a viewer meets it asks the only question
    // that matters — is the seam frame distinguishable from its neighbours?
    //
    // `resetDeterministic` first, because it is what pauses the loop. Without it
    // the render loop keeps running on wall-clock time between round-trips, and
    // the per-frame distances come out around fifteen metres — a measurement of
    // this test's own latency rather than of the camera. The whole walk then runs
    // inside a single `evaluate` so no round-trip can get between two frames.
    await page.evaluate(() => window.__ocean.resetDeterministic(0, 30));

    const path = await page.evaluate(
      async ([seamStart, frames]: [number, number]) => {
        const ocean = window.__ocean;
        ocean.director.resetCinematic(seamStart);
        ocean.director.snapToTarget();
        await ocean.step(1 / 60, 30);
        const samples: { x: number; y: number; z: number }[] = [];
        for (let frame = 0; frame < frames; frame++) {
          await ocean.step(1 / 60, 1);
          const p = ocean.camera.position;
          samples.push({ x: p.x, y: p.y, z: p.z });
        }
        return samples;
      },
      [CINEMATIC_LOOP_SECONDS - 0.5, 90] as [number, number],
    );

    const steps: number[] = [];
    for (let i = 1; i < path.length; i++) {
      steps.push(
        Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y, path[i].z - path[i - 1].z),
      );
    }
    const sorted = [...steps].sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    const worst = sorted[sorted.length - 1];

    console.log(
      `[cinematic] seam crossing: median step ${median.toFixed(4)} m, worst ${worst.toFixed(4)} m`,
    );
    expect(median, 'the camera is not moving, so this proves nothing').toBeGreaterThan(1e-3);
    // A frame at 60 Hz on a camera doing tens of metres a second is centimetres;
    // anything several times the median at the wrap is a cut.
    expect(
      worst,
      `the worst frame across the wrap moved ${worst.toFixed(3)} m against a median of ` +
        `${median.toFixed(3)} m — a cut, not a loop`,
    ).toBeLessThan(median * 4);
  });

  test('sliders drive the simulation', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    // Stepped between setting and reading, and that is not a formality.
    // `updateSpectrum` rewrites h0 and marks it dirty; the field the readback
    // measures is whatever the *last rendered frame* produced from it. Reading
    // straight after `setState` therefore measures the previous sea state, and
    // whether it happens to be right depends on whether a frame landed in the
    // gap. It always used to; once the island put sixteen million triangles in
    // the scene it stopped, and the test failed with `calm` and `rough`
    // bit-identical — which is the signature of a race, not of a frozen sim.
    await setState(page, { windSpeed: 3, peakWavelength: 20 });
    await page.evaluate(() => window.__ocean.step(1 / 60, 2));
    const calm = await peakWaveHeight(page);

    await setState(page, { windSpeed: 24, peakWavelength: 120 });
    await page.evaluate(() => window.__ocean.step(1 / 60, 2));
    const rough = await peakWaveHeight(page);

    expect(rough, `calm=${calm} rough=${rough}`).toBeGreaterThan(calm * 1.5);
  });
});

test.describe('performance', () => {
  /**
   * These gates key on per-frame WORK, not on delivered frame rate.
   *
   * Automated Chromium throttles requestAnimationFrame independently of load —
   * this project measured 1.1 "FPS" while spending 0.8 ms per frame, and the
   * same 1.00 fps appears with every effect switched off. Asserting on FPS here
   * would test the harness, not the renderer. When the browser is NOT throttling
   * we additionally assert the frame rate, so a real regression on an
   * interactive run still fails.
   */
  const budget = (target: number) => (sample: {
    fps: number;
    frameMs: number;
    rafThrottled: boolean;
  }) => {
    expect(
      sample.frameMs,
      `frame work was ${sample.frameMs.toFixed(2)} ms (budget ${target} ms); ` +
        `delivered ${sample.fps.toFixed(1)} FPS, rafThrottled=${sample.rafThrottled}`,
    ).toBeLessThan(target);
  };

  test('stays within the frame budget at High on WebGPU', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);
    await setState(page, { quality: 'high' });

    const sample = await measureFrameRate(page, 4);
    budget(16.7)(sample);
    if (!sample.rafThrottled) {
      expect(sample.fps, `median FPS was ${sample.fps.toFixed(1)}`).toBeGreaterThan(55);
    }
  });

  test('stays within the fallback frame budget at Low on WebGL', async ({ page }) => {
    await page.goto('/?gate=0&quality=high&webgl=1');
    await waitForOcean(page);
    await setState(page, { quality: 'low' });

    const sample = await measureFrameRate(page, 4);
    budget(33.3)(sample);
    if (!sample.rafThrottled) {
      expect(sample.fps, `median FPS was ${sample.fps.toFixed(1)}`).toBeGreaterThan(30);
    }
  });

  test('does not leak GPU memory across quality changes', async ({ page }) => {
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page);

    const sample = async () =>
      page.evaluate(() => {
        const info = (
          window as unknown as {
            __ocean: { renderer: { info: { memory: { textures: number; geometries: number } } } };
          }
        ).__ocean.renderer.info.memory;
        return { textures: info.textures, geometries: info.geometries };
      });

    const cycle = async (times: number) => {
      for (let i = 0; i < times; i++) {
        await setState(page, { quality: 'low' });
        await setState(page, { quality: 'high' });
      }
    };

    // Warm-up, so lazily-created internal targets exist before anything is
    // counted.
    await cycle(2);
    const baseline = await sample();

    await cycle(4);
    const mid = await sample();

    await cycle(8);
    const after = await sample();

    /*
     * Measured as a *rate*, not as a total.
     *
     * The previous version ran 8 transitions and allowed 8 extra textures, which
     * is one per transition — so the exact leak this test exists to catch, a
     * single texture per tier change, sat precisely on the limit and passed. That
     * is not a slack allowance, it is the failure mode written down as the
     * budget.
     *
     * Two segments instead. The second is twice as long as the first, so a
     * per-cycle leak has to show at least twice the growth; anything created once
     * and reused shows up in the first segment and not the second. The absolute
     * bound stays as a backstop for a leak so large it saturates both.
     */
    const firstSegment = mid.textures - baseline.textures;
    const secondSegment = after.textures - mid.textures;

    expect(
      secondSegment,
      `textures grew by ${firstSegment} over 4 cycles and ${secondSegment} over the next 8; ` +
        'growth that continues at rate is a leak, growth that stops is lazy allocation',
    ).toBeLessThanOrEqual(Math.max(2, firstSegment));

    expect(after.textures).toBeLessThanOrEqual(baseline.textures + 8);
    expect(after.geometries).toBeLessThanOrEqual(baseline.geometries + 4);
  });
});

test.describe('responsiveness', () => {
  for (const [label, width, height] of [
    ['desktop', 1920, 1080],
    ['laptop', 1440, 900],
    ['tablet', 834, 1112],
    ['phone', 390, 844],
    ['narrow', 360, 720],
  ] as const) {
    test(`${label} (${width}x${height}) has no layout overflow`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.goto('/?gate=0&quality=high');
      await waitForOcean(page, 1200);

      const overflow = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth + 1);
    });
  }

  /**
   * Every control the panel offers must be reachable.
   *
   * The overflow tests above pass whether or not this holds, and that is not a
   * hypothetical: the panel body is a flex child, defaulted to `min-height:
   * auto`, and so refused to shrink to the panel's `max-height`. It grew past its
   * container, `.panel { overflow: hidden }` cut the bottom off, and the Time of
   * Day and Pixel Ratio sliders were simply not on screen — while
   * `document.scrollWidth` stayed exactly equal to `clientWidth`, because clipped
   * content does not overflow.
   *
   * Two assertions, because they fail for different reasons. The body must sit
   * inside the panel, which catches the clipping directly. And scrolling the body
   * to its end must bring the last control into the panel's box, which catches
   * the subtler version where the body scrolls but not far enough.
   *
   * 700 px is chosen to be shorter than the panel's natural content height at the
   * current control count, so the constraint is actually exercised. If controls
   * are removed until it fits, this test stops proving anything — hence the
   * assertion that the body really is overflowing to begin with.
   */
  test('every panel control is reachable on a short viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 700 });
    await page.goto('/?gate=0&quality=high');
    await waitForOcean(page, 1200);

    const geometry = await page.evaluate(() => {
      const panel = document.querySelector('.panel') as HTMLElement;
      const body = document.querySelector('.panel__body') as HTMLElement;
      if (!panel || !body) return null;
      const controls = [...body.querySelectorAll('input[type="range"]')] as HTMLElement[];

      body.scrollTop = body.scrollHeight;
      const last = controls[controls.length - 1];
      return {
        controlCount: controls.length,
        // Ids are `panelN-<key>`, so the key is what follows the first dash.
        controlKeys: controls.map((c) => c.id.replace(/^panel\d+-/, '')),
        bodyScrolls: body.scrollHeight > body.clientHeight + 1,
        bodyBottom: body.getBoundingClientRect().bottom,
        panelBottom: panel.getBoundingClientRect().bottom,
        lastControlBottom: last?.getBoundingClientRect().bottom ?? 0,
        lastControlLabel: last?.getAttribute('aria-label') ?? last?.id ?? '(unnamed)',
      };
    });

    expect(geometry, '.panel or .panel__body is missing').not.toBeNull();
    const g = geometry!;

    // Named rather than counted. The panel used to place its sliders by array
    // index — `SLIDERS.slice(0, 3)` and `SLIDERS[3]` — so inserting `fogDensity`
    // and `timeOfDay` ahead of `pixelRatio` pushed both of the last two off the
    // end: they were declared, they had labels and formatters, and they were
    // never built. A count would have caught that only by accident.
    expect(
      g.controlKeys.sort(),
      'a declared slider is missing from the panel',
    ).toEqual(
      [
        'cloudCoverage',
        'fogDensity',
        'peakWavelength',
        'pixelRatio',
        'timeOfDay',
        'volume',
        'windSpeed',
      ],
    );
    expect(
      g.bodyScrolls,
      'the panel fits this viewport without scrolling, so the constraint this test ' +
        'exists for is not being exercised — shorten the viewport or it is vacuous',
    ).toBe(true);

    expect(
      g.bodyBottom,
      'the panel body extends past the panel, so overflow:hidden is clipping controls',
    ).toBeLessThanOrEqual(g.panelBottom + 1);

    expect(
      g.lastControlBottom,
      `scrolled to the end, the last control (${g.lastControlLabel}) is still below ` +
        'the panel and cannot be reached',
    ).toBeLessThanOrEqual(g.panelBottom + 1);
  });
});

// ---------------------------------------------------------------------- utils

async function peakWaveHeight(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(async () => {
    const ocean = (window as unknown as { __ocean: Record<string, never> }).__ocean;
    const renderer = ocean.renderer as unknown as {
      readRenderTargetPixelsAsync: (
        t: unknown, x: number, y: number, w: number, h: number,
      ) => Promise<ArrayLike<number>>;
    };
    const simulation = ocean.simulation as unknown as { displacementTargets: unknown[] };
    const halfToFloat = (bits: number) => {
      const sign = bits & 0x8000 ? -1 : 1;
      const exponent = (bits & 0x7c00) >> 10;
      const mantissa = bits & 0x03ff;
      if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
      if (exponent === 31) return mantissa ? NaN : sign * Infinity;
      return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
    };
    const raw = await renderer.readRenderTargetPixelsAsync(
      simulation.displacementTargets[0], 0, 0, 64, 64,
    );
    const isHalf = (raw as ArrayLike<number> & { BYTES_PER_ELEMENT?: number })
      .BYTES_PER_ELEMENT === 2;
    let peak = 0;
    for (let i = 0; i < raw.length; i += 4) {
      const height = isHalf ? halfToFloat(raw[i + 1]) : raw[i + 1];
      if (Number.isFinite(height)) peak = Math.max(peak, Math.abs(height));
    }
    return peak;
  });
}
