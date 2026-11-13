export type Updatable = (dt: number, elapsed: number) => void;

export interface FrameStats {
  /** Smoothed frames per second. */
  fps: number;
  /** Milliseconds of wall clock spent in the last frame. */
  frameMs: number;
}

/**
 * Frame driver. Simulation receives a clamped delta so that a tab switch or a
 * long shader compile cannot teleport the ship across the ocean.
 */
/**
 * Draws one frame. `live` is true for the free-running frames `tick` draws and
 * false for `step`'s deterministic ones, which must come out complete.
 */
export type RenderFn = (dt: number, elapsed: number, live: boolean) => void | Promise<void>;

export class Loop {
  private readonly updates: Updatable[] = [];
  private readonly renderFn: RenderFn;
  private running = false;
  private paused = false;
  private last = 0;
  private elapsed = 0;
  private handle = 0;
  private inFlight = false;
  /**
   * The frame currently inside `renderFn`, so a caller can join it.
   *
   * `inFlight` on its own is a flag with no join operation: pausing the loop
   * stops *new* frames but says nothing about the one already awaiting its
   * render. Anything that then destroys a GPU resource — a quality change — can
   * do so while that frame's command buffer still references it, which WebGPU
   * reports as "Destroyed texture used in a submit". `settle()` is the join.
   */
  private inFlightPromise: Promise<void> | null = null;
  private frameWaiters: Array<() => void> = [];

  readonly stats: FrameStats = { fps: 0, frameMs: 0 };

  /** Exponential moving average keeps the readout stable without lagging real drops. */
  private static readonly FPS_SMOOTHING = 0.1;
  private static readonly MAX_DELTA = 1 / 15;

  constructor(renderFn: RenderFn) {
    this.renderFn = renderFn;
  }

  add(update: Updatable): void {
    this.updates.push(update);
  }

  remove(update: Updatable): void {
    const i = this.updates.indexOf(update);
    if (i >= 0) this.updates.splice(i, 1);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    this.handle = requestAnimationFrame(this.tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.handle);
    for (const resolve of this.frameWaiters.splice(0)) resolve();
  }

  /**
   * Simulation clock, in seconds. Every animated system in the project is a
   * function of this value, so reading and restoring it is what makes a frame
   * reproducible.
   */
  get elapsedTime(): number {
    return this.elapsed;
  }

  /**
   * Resolves once no frame is inside `renderFn`.
   *
   * Pair with `setPaused(true)`: pause first so no *new* frame starts, then
   * settle so the one already running has finished. Doing it the other way round
   * leaves a window in which a fresh frame begins between the two calls.
   */
  async settle(): Promise<void> {
    while (this.inFlightPromise !== null) {
      await this.inFlightPromise;
    }
  }

  /**
   * Resolves once the loop has finished drawing a frame it started after this
   * call. Resolves immediately if the loop is not running, since no frame is
   * coming; while paused it waits for the resume.
   */
  nextFrame(): Promise<void> {
    if (!this.running) return Promise.resolve();
    return new Promise((resolve) => this.frameWaiters.push(resolve));
  }

  get isPaused(): boolean {
    return this.paused;
  }

  /**
   * True once `start()` has been called and `stop()` has not.
   *
   * Distinct from `isPaused`: a paused loop is running and choosing not to
   * advance, while a loop that has never started cannot draw a frame at all.
   * Work that must not be interrupted by a render only needs to stop the loop
   * when there is one — see `loadSceneContent`, which does that dance during
   * gameplay and skips it entirely when it runs behind the boot overlay.
   */
  get isRunning(): boolean {
    return this.running;
  }

  /**
   * Detaches the simulation from wall clock.
   *
   * While paused the rAF callback keeps firing but neither advances time nor
   * renders; `step()` becomes the only way the world moves. This is the basis of
   * every deterministic capture: real frame pacing varies by machine and by
   * whatever else the GPU is doing, so a baseline taken against wall-clock time
   * can never be reproduced exactly.
   */
  setPaused(paused: boolean): void {
    this.paused = paused;
    // Resuming must not integrate the entire pause as one delta.
    this.last = performance.now();
  }

  /** Hard-sets the simulation clock without running any update. */
  setElapsed(seconds: number): void {
    this.elapsed = seconds;
  }

  /**
   * Advances the simulation by exactly `steps` increments of `dt` and renders
   * once, awaiting the render.
   *
   * Stepping several small increments rather than one large one matters: the
   * buoyancy solver, the wake decay and the foam advection all integrate over
   * `dt`, and a single 1 s step lands somewhere a 60 × 16.7 ms sequence never
   * would.
   */
  async step(dt: number, steps = 1): Promise<void> {
    for (let s = 0; s < steps; s++) {
      this.elapsed += dt;
      for (const update of this.updates) update(dt, this.elapsed);
    }
    const started = performance.now();
    await this.renderFn(dt, this.elapsed, false);
    this.stats.frameMs = performance.now() - started;
  }

  private tick = async (now: number): Promise<void> => {
    if (!this.running) return;
    this.handle = requestAnimationFrame(this.tick);

    // Paused: `step()` owns the clock. Keep the rAF chain alive so resuming does
    // not need to restart it.
    if (this.paused) return;

    // Skip if the previous async render has not resolved — prevents unbounded
    // queueing of GPU work when the device is the bottleneck.
    if (this.inFlight) return;

    const rawDelta = (now - this.last) / 1000;
    this.last = now;
    const dt = Math.min(rawDelta, Loop.MAX_DELTA);
    this.elapsed += dt;

    for (const update of this.updates) update(dt, this.elapsed);

    this.inFlight = true;
    const waiters = this.frameWaiters.splice(0);
    const started = performance.now();
    const render = this.renderFn(dt, this.elapsed, true);
    this.inFlightPromise = Promise.resolve(render).then(
      () => undefined,
      () => undefined,
    );
    try {
      await render;
    } finally {
      this.inFlight = false;
      this.inFlightPromise = null;
      this.stats.frameMs = performance.now() - started;
      const instantaneous = rawDelta > 0 ? 1 / rawDelta : 0;
      this.stats.fps =
        this.stats.fps === 0
          ? instantaneous
          : this.stats.fps + (instantaneous - this.stats.fps) * Loop.FPS_SMOOTHING;
      for (const resolve of waiters) resolve();
    }
  };
}
