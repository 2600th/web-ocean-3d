import type { Page } from '@playwright/test';
import type { RgbaImage } from './png';
import type { Shot } from './shots';

/**
 * Drives the app's deterministic hooks and turns a frame into pixels Node can
 * work with.
 *
 * The capture path is `__ocean.capturePixels()`, never `page.screenshot()`. A
 * screenshot is whatever the compositor last presented: it can be a frame stale,
 * it goes through the browser's own colour management, and under automation —
 * where requestAnimationFrame is throttled to around 1 Hz — it may simply not
 * arrive before the test times out. `capturePixels` renders the post chain into
 * an offscreen target and reads it straight back, so it is the pixels the shader
 * wrote, on demand, with no dependence on frame pacing.
 */

interface CapturedPixels {
  width: number;
  height: number;
  /**
   * RGBA8. Pinned to a plain `ArrayBuffer` rather than the default
   * `ArrayBufferLike`, because `Blob` will not accept a view that might be
   * backed by a `SharedArrayBuffer` — and this one never is. Row order is
   * discussed on `capture` below; it is not what the hook's own docs say.
   */
  data: Uint8Array<ArrayBuffer>;
}

/** Only the parts of `window.__ocean` the tests use. */
interface OceanHooks {
  backend: 'webgpu' | 'webgl';
  scene: { getObjectByName(name: string): { visible: boolean } | undefined };
  setRainOverride(intensity: number | null): void;
  /** Forces foam and surf strength together; `null` returns them to the weather. */
  setFoamOverride(strength: number | null): void;
  director: {
    snapToTarget(): void;
    currentMode: string;
    /** Engine orders published by the cinematic flight; zeroes in every other mode. */
    readonly shipInput: { readonly throttle: number; readonly rudder: number };
    readonly cinematicBeat: string;
    readonly cinematicTime: number;
    resetCinematic(time?: number): void;
  };
  /** The tour's environment curves at a loop time. */
  cinematicEnvironment(time?: number): {
    hours: number;
    rain: number;
    cloudCoverage: number;
    fogDensity: number;
    weatherKind: 'clear' | 'rain';
  };
  /** Beat names, start times and durations, in order. */
  cinematicBeats(): Array<{ name: string; start: number; duration: number }>;
  /** Nominal hull position at a loop time, world XZ. */
  nominalShipAt(time: number): { x: number; z: number };
  camera: { position: { x: number; y: number; z: number } };
  /** The surface material. Only the knobs a test drives are declared. */
  water: {
    setWakeDisplacement(value: number): void;
    setReflection(amount: number): void;
  };
  wake: {
    emit(x: number, z: number, heading: number, speed: number, width: number): void;
    setCenter(x: number, z: number): void;
    readonly centerX: number;
    readonly centerZ: number;
    readonly extent: number;
  };
  isReady(): boolean;
  shadersReady(): boolean;
  setState(partial: Record<string, unknown>): void | Promise<void>;
  setCamera(px: number, py: number, pz: number, tx: number, ty: number, tz: number): void;
  resetDeterministic(
    time?: number,
    settleSteps?: number,
    shipInput?: { throttle: number; rudder: number } | null,
    cinematicTime?: number | null,
  ): Promise<void>;
  step(dt: number, steps?: number): Promise<void>;
  capturePixels(): Promise<CapturedPixels>;
  /** Test-only dither amplitude in output levels; 0 disables. */
  setDitherLevels(levels: number): void;
  setAaStrength(strength: number): void;
  /** Test-only lens override: gather tap count and f-number. */
  setDof(samples: number, fNumber: number): void;
  /** Test-only flare override. */
  setFlareEnabled(on: boolean): void;
  /** Test-only bloom override. */
  setBloomEnabled(on: boolean): void;
  /** Test-only grade override. Colours are plain triples; see the hook's note. */
  setGrade(g: {
    slope: [number, number, number];
    offset: [number, number, number];
    power: [number, number, number];
    saturation: number;
  }): void;
  setShipInput(throttle: number, rudder: number): void;
  shipControlsEnabled(): boolean;
  touchControlsVisible(): boolean;
  /** Reflection layers. Null on the WebGL2 path, which has neither. */
  reflections: { setQuality(scale: number): void; readonly resolutionScale: number } | null;
  /** Sky and sun. Declared for the per-tier shadow resolution check. */
  atmosphere: {
    readonly shadowMapSize: number;
    readonly sunDirection: { x: number; y: number; z: number };
  };
  ssr: { setStrength(amount: number): void } | null;
  surfaceWetness(): number;
  setSurfaceWetness(value: number): void;
  shipState(): {
    throttle: number;
    rudder: number;
    speed: number;
    forwardSpeed: number;
    heading: number;
  } | null;
}

declare global {
  interface Window {
    __ocean: OceanHooks;
  }
}

/** Waits for the scene to have loaded, prewarmed and drawn at least one frame. */
export async function bootOcean(page: Page): Promise<void> {
  await page.goto('/?gate=0&quality=high');
  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 60_000 });
  await page.waitForFunction(() => window.__ocean.isReady(), undefined, { timeout: 180_000 });
  // Without this a first capture can be taken while pipelines are still being
  // built, and the frame it produces is not the frame the same call produces a
  // second later.
  await page.waitForFunction(() => window.__ocean.shadersReady(), undefined, { timeout: 180_000 });
}

/**
 * Puts the world into the state a shot describes and leaves it there, settled.
 *
 * Order matters and is not arbitrary:
 *
 * 1. `preset` is applied **last** in the state object. `applyPreset` rebuilds
 *    the whole wave spectrum from the preset's sea parameters merged with the
 *    current wind values, so applying it after the wind sliders gives one
 *    complete spectrum update instead of a preset update followed by a partial
 *    one.
 * 2. The camera is pinned **before** the settle, because the underwater pass,
 *    the fog and the particle field all read the camera during `update()`.
 * 3. `resetDeterministic` rewinds every clock and clears the wake buffer, then
 *    settles by stepping, so the frame does not depend on what ran before it.
 */
export async function applyShot(page: Page, shot: Shot): Promise<void> {
  // Awaited: a tier change now drains the GPU queue before it destroys anything,
  // so `setState` is genuinely asynchronous when `quality` moves. Without the
  // await, a shot can be captured while the previous tier is still being torn
  // down — which showed up as the UI-exclusion check seeing a 255-level
  // difference between two captures that should have been identical.
  await page.evaluate(async (state) => {
    await window.__ocean.setState({
      quality: state.quality,
      cameraMode: state.cameraMode,
      windSpeed: state.windSpeed,
      peakWavelength: state.peakWavelength,
      cloudCoverage: state.cloudCoverage,
      preset: state.preset,
    });
  }, shot.state);

  if (shot.camera) {
    await page.evaluate(
      ({ position, target }) => {
        window.__ocean.setCamera(
          position[0], position[1], position[2],
          target[0], target[1], target[2],
        );
      },
      { position: shot.camera.position, target: shot.camera.target },
    );
  }

  if (shot.state.cameraMode === 'cinematic' && shot.cinematicTime === undefined) {
    throw new Error(`shot "${shot.id}" is cinematic but names no cinematicTime`);
  }

  // Two clocks, and they are not the same one. `shot.time` is the *simulation*
  // clock — what the sea, the foam and the wake have been doing — while
  // `cinematicTime` is a position on the flight's own 166 s lap, which decides
  // the pose, the hour and the weather together. `resetDeterministic` rewinds
  // both, each so that the settle *ends* on the requested value.
  await page.evaluate(
    ({ time, settleSteps, shipInput, cinematicTime }) =>
      window.__ocean.resetDeterministic(time, settleSteps, shipInput, cinematicTime),
    {
      time: shot.time,
      settleSteps: shot.settleSteps,
      shipInput: shot.shipInput ?? null,
      cinematicTime: shot.cinematicTime ?? null,
    },
  );

  if (!shot.camera && shot.state.cameraMode === 'boat') {
    // The chase rig damps toward its ideal pose, so after a settle it is close
    // but not equal to it. `snapToTarget` places it exactly. Chase-only: it
    // returns immediately in any other mode, and the cinematic rig has its own
    // deterministic entry point below.
    await page.evaluate(() => window.__ocean.director.snapToTarget());
  }

  // Warm-up frames, discarded.
  //
  // These are the single largest lever on how sensitive this harness can be, and
  // the count is measured rather than guessed. Re-applying a shot five times and
  // scoring all ten pairs:
  //
  //   1 warm-up capture   storm mean ΔE 1.346, waterline 1.103
  //   2 warm-up captures  storm mean ΔE 0.035, waterline 0.000
  //   3 warm-up captures  storm mean ΔE 0.037, waterline 0.000
  //
  // — a factor of about thirty, and it flattens out immediately after. So the
  // frame right after a state change is not converged, the one after that is,
  // and what looked like irreducible chaos in the specular glitter was mostly
  // this. Three, because the second and third measure the same and the extra
  // costs about ten milliseconds a shot.
  //
  // `capturePixels` already renders twice internally to settle the wave-texture
  // mip chains, so this is six further full renders of the post chain.
  await page.evaluate(async () => {
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
    await window.__ocean.capturePixels();
  });
}

/**
 * Reads back the current frame as a top-down RGBA8 image.
 *
 * No vertical flip, despite `capturePixels` documenting its buffer as bottom-up
 * "render-target origin". On the WebGPU backend it is not: three's WebGPU
 * readback copies texture rows in texture order, which puts row 0 at the top,
 * and flipping produced an ocean above a sky. The claim may well hold on the
 * WebGL2 path — but this suite only compares WebGPU captures, so rather than
 * carry a conditional flip that nothing here can exercise, this asserts the one
 * orientation it has actually observed.
 */
export async function capture(page: Page): Promise<RgbaImage> {
  const frame = await page.evaluate(async () => {
    const { width, height, data } = await window.__ocean.capturePixels();
    // Base64 through a Blob so the browser does the encoding in native code.
    // Building the string in JS with String.fromCharCode over 3.7 MB is roughly
    // an order of magnitude slower, and this runs once per capture per shot.
    const url = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(new Blob([data]));
    });
    return { width, height, base64: url.slice(url.indexOf(',') + 1) };
  });

  return {
    width: frame.width,
    height: frame.height,
    data: new Uint8Array(Buffer.from(frame.base64, 'base64')),
  };
}
