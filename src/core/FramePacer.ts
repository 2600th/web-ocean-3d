/**
 * Chooses which display refreshes get a frame, and how much time each frame
 * stands for.
 *
 * **Why the loop needs this.** On a 120 Hz display at High, the GPU finishes a
 * frame in about 10.7 ms against an 8.3 ms refresh. The browser does not wait
 * politely for it: it queues three frames deep, 36 ms behind the input, and
 * hands out animation frames at whatever moment the queue has room. Measured
 * live, about a third of the frames were shown for two refreshes and the rest
 * for one, in no pattern. Every frame was simulated for the time its callback
 * began, and a frame that is then shown a refresh late is a frame of motion in
 * the wrong place. That is the jitter on the hull and on the island when the
 * fly camera moves: not the physics, which was measured smooth to a fifth of a
 * pixel, but the presentation.
 *
 * A steady 60 reads as smoother than an uneven 93, which is why consoles lock
 * to a divisor of the refresh rate, and why Android's frame-pacing library
 * (Swappy) and Alen Ladavac's GDC 2019 talk on frame timing land on the same two
 * rules this class applies:
 *
 *  1. **Render on every `divisor`-th refresh**, and raise the divisor when
 *     frames keep missing it. A frame that misses is one shown later than the
 *     time it was simulated for, so a miss is the thing being measured.
 *  2. **Advance time by whole refreshes.** The callback timestamp wobbles
 *     around the vsync, and the frame is shown on a vsync, so the time it
 *     stands for is a whole number of refresh periods.
 *
 * The divisor comes back down by probation: try one less and keep it if
 * frames stop missing. A GPU latency well over the faster rate's budget skips
 * a probe that cannot succeed, and failed probes back off, so a machine that
 * cannot make the faster rate does not judder every few seconds proving it.
 *
 * Pure bookkeeping on timestamps, so it is unit-tested without a browser.
 */
export class FramePacer {
  /** Longest interval a divisor may ask for: about 28 fps. */
  static readonly MAX_INTERVAL_MS = 1000 / 28;
  /**
   * Rendered frames, and the span of time they must cover, before a miss rate
   * is believed. Frames rather than time alone so that a slow device, at 15 fps,
   * still gets a decision within its second.
   */
  static readonly MIN_FRAMES = 8;
  static readonly MIN_SPAN_MS = 750;
  /** Missed fraction in the last second that raises the divisor. */
  static readonly MISS_RATE = 0.1;
  /** How long a lowered divisor runs before it is kept, ms. */
  static readonly PROBATION_MS = 1500;
  /** First and longest wait before a probe at a lower divisor, ms. */
  static readonly BACKOFF_MIN_MS = 4000;
  static readonly BACKOFF_MAX_MS = 60000;
  /**
   * A probe is skipped when the median frame cost is over this many times
   * the faster rate's interval. The cost is the slower of a frame's two stages
   * (see `gpuLatency`); the GPU stage still counts the browser's GPU process
   * and the GPU end to end, which overlap across frames, so it overstates a
   * little, and the margin keeps a machine that could make the rate from being
   * kept off it. Not much margin: a probe that fails is a second and a half of
   * the judder this class exists to remove, and a device within 15% of the
   * budget fails it.
   */
  static readonly PROBE_LATENCY_RATIO = 1.15;
  /** Slowest display refresh believed, ms: 45 Hz. */
  private static readonly MAX_PERIOD_MS = 1000 / 45;
  /** Refresh intervals kept for the period estimate. */
  private static readonly PERIOD_WINDOW = 120;
  /** Gaps longer than this are stalls (a tab switch, a compile), not misses. */
  private static readonly STALL_MS = 100;

  private readonly intervals: number[] = [];
  private intervalCursor = 0;
  private lastVsync = Number.NaN;
  private sinceEstimate = 0;
  private periodMs = 1000 / 60;

  private divisorValue = 1;
  /** Render times and whether each missed, for the last second. */
  private readonly history: Array<{ at: number; missed: boolean }> = [];
  /** Recent GPU latencies, ms. */
  private readonly latencies: number[] = [];
  private changedAt = Number.NEGATIVE_INFINITY;
  /** Set while a lowered divisor is on trial: the divisor to go back to. */
  private probationFrom = 0;
  private backoffMs = FramePacer.BACKOFF_MIN_MS;
  private nextProbeAt = Number.NEGATIVE_INFINITY;

  /** Estimated display refresh interval, ms. */
  get period(): number {
    return this.periodMs;
  }

  /** Frames are drawn on every `divisor`-th refresh. */
  get divisor(): number {
    return this.divisorValue;
  }

  /** The frame rate the pacer is holding, frames per second. */
  get targetFps(): number {
    return 1000 / (this.periodMs * this.divisorValue);
  }

  /**
   * Records a refresh. Call on every animation frame callback, rendered or
   * not, with its timestamp: the spacing of the callbacks is how the refresh
   * rate is found.
   */
  vsync(now: number): void {
    const gap = now - this.lastVsync;
    this.lastVsync = now;
    if (!(gap > 2 && gap < 60)) return;
    if (this.intervals.length < FramePacer.PERIOD_WINDOW) this.intervals.push(gap);
    else this.intervals[this.intervalCursor] = gap;
    this.intervalCursor = (this.intervalCursor + 1) % FramePacer.PERIOD_WINDOW;
    if (++this.sinceEstimate >= 15 && this.intervals.length >= 30) {
      this.sinceEstimate = 0;
      this.estimatePeriod();
    }
  }

  /**
   * The refresh interval is the shortest one callbacks keep arriving at. A busy
   * GPU stretches some to two or three refreshes and never shortens any, so a
   * low percentile finds the display rate under load; the median would find
   * the load. The percentile only picks the cluster: its mean is the estimate,
   * since a timestamp late on one callback is early on the next and the
   * percentile alone would be biased short by the wobble.
   */
  private estimatePeriod(): void {
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const floor = sorted[Math.floor(sorted.length * 0.2)];
    let sum = 0;
    let count = 0;
    for (const gap of sorted) {
      if (gap < floor * 0.75) continue;
      if (gap > floor * 1.25) break;
      sum += gap;
      count++;
    }
    const estimate = sum / count;
    // No display refreshes slower than this; a cluster past it is a device so
    // overloaded that every callback is late, and says nothing about the rate.
    if (estimate > FramePacer.MAX_PERIOD_MS) return;
    // A longer interval than before is believed only when callbacks arrive at
    // it steadily. An overloaded GPU holds callbacks back too, two refreshes
    // here and three there, and that is the load to pace against, not a slower
    // display; a slower display delivers them evenly.
    if (estimate > this.periodMs * 1.15) {
      const high = sorted[Math.floor(sorted.length * 0.8)];
      if (high > floor * 1.1) return;
    }
    if (Math.abs(estimate - this.periodMs) > this.periodMs * 0.15) {
      // A different display (the window moved), or a first estimate refined.
      // Either way the frame interval being held is what the viewer sees, so
      // it is kept: 60 fps at a divisor of 2 on 120 Hz is a divisor of 1 on 60.
      const interval = this.divisorValue * this.periodMs;
      this.divisorValue = Math.max(1, Math.round(interval / estimate));
      while (this.divisorValue > 1 && this.divisorValue * estimate > FramePacer.MAX_INTERVAL_MS) {
        this.divisorValue--;
      }
      this.history.length = 0;
      this.latencies.length = 0;
      this.probationFrom = 0;
    }
    this.periodMs = estimate;
  }

  /**
   * Whether the refresh at `now` should get a frame, given that the last one
   * was drawn at `last`. Half a period of slack: a callback lands either side
   * of the vsync it belongs to.
   */
  shouldRender(now: number, last: number): boolean {
    return now - last >= (this.divisorValue - 0.5) * this.periodMs;
  }

  /**
   * The time, in seconds, a frame drawn at `now` stands for: the whole number
   * of refreshes since `last`. A gap that is not near a whole number of them
   * (a stall, a resume) is passed through as measured.
   */
  delta(now: number, last: number): number {
    const gap = now - last;
    const refreshes = Math.round(gap / this.periodMs);
    if (refreshes >= 1 && Math.abs(gap - refreshes * this.periodMs) < this.periodMs * 0.3) {
      return (refreshes * this.periodMs) / 1000;
    }
    return gap / 1000;
  }

  /**
   * Records a frame drawn at `now` after one drawn at `last`, and moves the
   * divisor if the last second says it should.
   */
  rendered(now: number, last: number): void {
    const gap = now - last;
    if (gap > FramePacer.STALL_MS) {
      // A stall says nothing about the rate the GPU can hold.
      this.history.length = 0;
      return;
    }
    const refreshes = Math.round(gap / this.periodMs);
    this.history.push({ at: now, missed: refreshes > this.divisorValue });
    while (this.history.length > 0 && now - this.history[0].at > 1000) this.history.shift();

    const settled = now - this.changedAt > 1000;
    const span = now - this.history[0].at;
    if (settled && this.history.length >= FramePacer.MIN_FRAMES && span >= FramePacer.MIN_SPAN_MS) {
      let missed = 0;
      for (const frame of this.history) if (frame.missed) missed++;
      if (missed / this.history.length > FramePacer.MISS_RATE) {
        this.raise(now);
        return;
      }
    }

    if (this.probationFrom > 0 && now - this.changedAt > FramePacer.PROBATION_MS) {
      // Survived: the lower divisor stays, and the next probe may come soon.
      this.probationFrom = 0;
      this.backoffMs = FramePacer.BACKOFF_MIN_MS;
      this.nextProbeAt = now + this.backoffMs;
    }

    if (this.divisorValue > 1 && this.probationFrom === 0 && now >= this.nextProbeAt) {
      this.probe(now);
    }
  }

  /**
   * Records how long a frame took from its callback to the GPU finishing it,
   * and how much of that was the page encoding it (callback to last submit).
   *
   * What is kept is the slower *stage*, not the sum. Encoding frame N+1
   * overlaps the browser's GPU process and the GPU working on frame N, so a
   * device holds a rate when each stage fits the interval, though the whole
   * trip from callback to finished frame takes longer: measured at High on a
   * 100 Hz display, 17 ms end to end with 99.8 fps delivered unpaced. Gating
   * probes on the sum kept that device at half rate. A caller that does not
   * know its encode time passes none, and the latency is the cost as before.
   */
  gpuLatency(ms: number, encodeMs = 0): void {
    if (!(ms > 0 && ms < 1000)) return;
    const encode = encodeMs > 0 && encodeMs < ms ? encodeMs : 0;
    this.latencies.push(Math.max(encode, ms - encode));
    if (this.latencies.length > 60) this.latencies.shift();
  }

  private raise(now: number): void {
    if (this.probationFrom > 0) {
      // A failed probe: back to where it was, and wait longer before the next.
      this.divisorValue = this.probationFrom;
      this.probationFrom = 0;
      this.backoffMs = Math.min(this.backoffMs * 2, FramePacer.BACKOFF_MAX_MS);
    } else {
      const next = this.divisorValue + 1;
      if (next * this.periodMs > FramePacer.MAX_INTERVAL_MS) return;
      this.divisorValue = next;
    }
    this.changedAt = now;
    this.history.length = 0;
    this.latencies.length = 0;
    this.nextProbeAt = now + this.backoffMs;
  }

  private probe(now: number): void {
    const lower = this.divisorValue - 1;
    if (this.latencies.length >= 20) {
      const sorted = [...this.latencies].sort((a, b) => a - b);
      const median = sorted[sorted.length >> 1];
      if (median > lower * this.periodMs * FramePacer.PROBE_LATENCY_RATIO) {
        // Hopeless; look again later without spending a judder on it.
        this.nextProbeAt = now + this.backoffMs;
        return;
      }
    }
    this.probationFrom = this.divisorValue;
    this.divisorValue = lower;
    this.changedAt = now;
    this.history.length = 0;
    this.latencies.length = 0;
  }
}
