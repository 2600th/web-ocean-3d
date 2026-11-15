import { expect, test } from '@playwright/test';
import { FramePacer } from '../src/core/FramePacer';

/**
 * Pure arithmetic — no browser, no GPU. A small model of a browser driving a
 * device that needs `costMs` of GPU time per frame: a callback arrives on every
 * refresh unless two frames are already queued on the GPU, which is the
 * backpressure that, measured in Chrome, turned a 10.7 ms frame on a 120 Hz
 * display into an uneven mix of one- and two-refresh frames.
 */
function drive(
  pacer: FramePacer,
  opts: { periodMs: number; costMs: (t: number) => number; seconds: number; start?: number; noise?: number },
): { gaps: number[]; end: number } {
  const { periodMs, seconds } = opts;
  const noise = opts.noise ?? 0;
  let t = opts.start ?? 0;
  let last = t;
  let gpuFree = t;
  const queued: number[] = [];
  const gaps: number[] = [];
  let k = 0;
  const end = t + seconds * 1000;
  while (t < end) {
    t += periodMs;
    k++;
    while (queued.length > 0 && queued[0] <= t) queued.shift();
    if (queued.length >= 2) continue;
    // Callback timestamps wobble around the vsync.
    const stamp = t + (noise ? Math.sin(k * 12.9898) * noise : 0);
    pacer.vsync(stamp);
    if (!pacer.shouldRender(stamp, last)) continue;
    pacer.rendered(stamp, last);
    gaps.push(stamp - last);
    last = stamp;
    gpuFree = Math.max(t, gpuFree) + opts.costMs(t);
    queued.push(gpuFree);
    pacer.gpuLatency(gpuFree - t);
  }
  return { gaps, end: t };
}

const P120 = 1000 / 120;
const P60 = 1000 / 60;

test('a device that cannot hold 120 Hz is held at a steady 60', () => {
  const pacer = new FramePacer();
  const { gaps } = drive(pacer, { periodMs: P120, costMs: () => 10.7, seconds: 6 });
  expect(pacer.divisor).toBe(2);
  expect(pacer.period).toBeCloseTo(P120, 1);
  // The last two seconds: every frame shown for exactly two refreshes.
  const tail = gaps.slice(-240 / 2);
  for (const gap of tail) expect(Math.round(gap / P120)).toBe(2);
});

test('a device that holds the refresh rate is left at it', () => {
  const pacer = new FramePacer();
  drive(pacer, { periodMs: P120, costMs: () => 5, seconds: 10 });
  expect(pacer.divisor).toBe(1);
});

test('the divisor comes back down once the device can hold the faster rate', () => {
  const pacer = new FramePacer();
  const first = drive(pacer, { periodMs: P120, costMs: () => 10.7, seconds: 4 });
  expect(pacer.divisor).toBe(2);
  drive(pacer, { periodMs: P120, costMs: () => 5, seconds: 12, start: first.end });
  expect(pacer.divisor).toBe(1);
});

test('a hopeless probe is skipped rather than juddered through', () => {
  const pacer = new FramePacer();
  const { gaps } = drive(pacer, { periodMs: P120, costMs: () => 14, seconds: 20 });
  expect(pacer.divisor).toBe(2);
  // After settling, no frame is shown for more than its two refreshes.
  const tail = gaps.slice(-14 * 60);
  expect(tail.filter((gap) => Math.round(gap / P120) !== 2)).toHaveLength(0);
});

test('an overloaded 60 Hz display is still read as 60 Hz, and paced no lower than 30', () => {
  const pacer = new FramePacer();
  drive(pacer, { periodMs: P60, costMs: () => 60, seconds: 6 });
  expect(pacer.period).toBeCloseTo(P60, 1);
  expect(pacer.divisor).toBe(2);
});

test('time advances by whole refreshes despite callback noise', () => {
  const pacer = new FramePacer();
  drive(pacer, { periodMs: P120, costMs: () => 3, seconds: 2, noise: 0.6 });
  expect(pacer.delta(1000 + P120 + 0.5, 1000)).toBeCloseTo(P120 / 1000, 5);
  expect(pacer.delta(1000 + 2 * P120 - 0.7, 1000)).toBeCloseTo((2 * P120) / 1000, 5);
  // Not near a whole number of refreshes: passed through.
  expect(pacer.delta(1000 + 1.5 * P120, 1000)).toBeCloseTo((1.5 * P120) / 1000, 6);
});

test('a stall is not counted as a run of misses', () => {
  const pacer = new FramePacer();
  const first = drive(pacer, { periodMs: P120, costMs: () => 5, seconds: 2 });
  // One 400 ms compile.
  drive(pacer, {
    periodMs: P120,
    costMs: (t) => (t < first.end + 20 ? 400 : 5),
    seconds: 3,
    start: first.end,
  });
  expect(pacer.divisor).toBe(1);
});

test('moving to a display with another refresh rate keeps the frame rate', () => {
  const pacer = new FramePacer();
  const first = drive(pacer, { periodMs: P120, costMs: () => 10.7, seconds: 4 });
  expect(pacer.divisor).toBe(2);
  drive(pacer, { periodMs: P60, costMs: () => 10.7, seconds: 4, start: first.end });
  expect(pacer.period).toBeCloseTo(P60, 1);
  expect(pacer.divisor).toBe(1);
});

test('a heavier load on 120 Hz settles on 40 fps without losing the refresh rate', () => {
  const pacer = new FramePacer();
  const { gaps } = drive(pacer, { periodMs: P120, costMs: () => 20, seconds: 12 });
  expect(pacer.period).toBeCloseTo(P120, 1);
  expect(pacer.divisor).toBe(3);
  const tail = gaps.slice(-3 * 40);
  expect(tail.filter((gap) => Math.round(gap / P120) !== 3)).toHaveLength(0);
});
