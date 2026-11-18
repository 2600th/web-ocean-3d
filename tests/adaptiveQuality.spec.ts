import { expect, test } from '@playwright/test';
import { AdaptiveQuality, type QualityTier } from '../src/core/QualityManager';

/**
 * Pure arithmetic — no browser, no GPU. The automatic quality drop is fed a
 * sequence of frames (how long since the last one, what it cost the device)
 * and the test reads off when it stepped the tier down, if it did.
 *
 * The rule under test, from `AdaptiveQuality`: under 41 fps, with the device
 * busy for at least half of each frame, for four seconds without a break. One
 * slow frame, or a second or two of them, must not do it.
 */

interface Segment {
  /** How long this stretch lasts, seconds of wall clock. */
  seconds: number;
  /** Time between frames, ms. */
  intervalMs: number;
  /** What a frame costs the device, ms. Defaults to nearly the whole interval. */
  costMs?: number;
}

interface Drop {
  at: number;
  tier: QualityTier;
  why: string;
}

/** Runs the segments in order from `tier`; returns each drop and when it came. */
function drive(segments: Segment[], tier: QualityTier = 'high'): { drops: Drop[]; adaptive: AdaptiveQuality } {
  let current = tier;
  let now = 1000;
  const drops: Drop[] = [];
  const adaptive = new AdaptiveQuality(55, (next, why) => {
    drops.push({ at: now, tier: next, why });
    current = next;
  });
  for (const segment of segments) {
    const end = now + segment.seconds * 1000;
    do {
      now += segment.intervalMs;
      adaptive.update({
        now,
        intervalMs: segment.intervalMs,
        costMs: segment.costMs ?? segment.intervalMs * 0.9,
        tier: current,
      });
    } while (now < end);
  }
  return { drops, adaptive };
}

const SMOOTH: Segment = { seconds: 20, intervalMs: 1000 / 60, costMs: 9 };

test('one slow frame never drops the tier, however long it is', () => {
  for (const stall of [40, 80, 300, 1000, 5000, 60_000]) {
    const { drops, adaptive } = drive([SMOOTH, { seconds: 0, intervalMs: stall }, SMOOTH]);
    expect(drops, `a single ${stall} ms frame dropped the tier`).toEqual([]);
    expect(adaptive.pressureMs).toBe(0);
  }
});

test('a slow frame every two seconds, for a minute, never drops the tier', () => {
  const segments: Segment[] = [SMOOTH];
  for (let i = 0; i < 30; i++) {
    segments.push({ seconds: 0, intervalMs: 120 }, { seconds: 2, intervalMs: 1000 / 60, costMs: 9 });
  }
  expect(drive(segments).drops).toEqual([]);
});

test('a burst of hitches does not drop the tier', () => {
  // Ten 300 ms frames in a row: three seconds of the worst a compile does.
  expect(drive([SMOOTH, { seconds: 3, intervalMs: 300 }, SMOOTH]).drops).toEqual([]);
  // Three seconds at 25 fps, then recovered.
  expect(drive([SMOOTH, { seconds: 3, intervalMs: 40 }, SMOOTH]).drops).toEqual([]);
});

test('a device that cannot keep up for four seconds drops one tier, once', () => {
  const { drops } = drive([SMOOTH, { seconds: 8, intervalMs: 1000 / 30 }]);
  expect(drops.map((d) => d.tier)).toEqual(['medium']);
  // 20 s of smooth frames, then about four seconds of slow ones (plus the few
  // frames the smoothed rate takes to cross the line).
  const after = (drops[0].at - 1000) / 1000 - SMOOTH.seconds;
  expect(after).toBeGreaterThan(4);
  expect(after).toBeLessThan(5);
  expect(drops[0].why).toMatch(/30 fps for 4\.\d s, frames costing 30 ms/);
});

test('still slow after a drop: the next tier waits ten seconds, then four more', () => {
  const { drops } = drive([SMOOTH, { seconds: 60, intervalMs: 1000 / 30 }], 'ultra');
  expect(drops.map((d) => d.tier)).toEqual(['high', 'medium', 'low']);
  for (let i = 1; i < drops.length; i++) {
    const gap = (drops[i].at - drops[i - 1].at) / 1000;
    expect(gap).toBeGreaterThan(13.9);
    expect(gap).toBeLessThan(14.5);
  }
});

test('the lowest tier has nowhere to go', () => {
  expect(drive([SMOOTH, { seconds: 60, intervalMs: 50 }], 'low').drops).toEqual([]);
});

test('a frame rate held down by something else is not the device being slow', () => {
  // A 30 Hz display, a remote desktop or the energy saver: 33 ms between
  // frames that cost 8 ms each. A lower tier would still run at 30.
  expect(drive([{ seconds: 120, intervalMs: 1000 / 30, costMs: 8 }]).drops).toEqual([]);
  // The same rate with the device working for most of each frame is real.
  expect(drive([{ seconds: 120, intervalMs: 1000 / 30, costMs: 28 }], 'medium').drops.length).toBe(1);
});

test('without a cost to go on, the frame rate alone is believed', () => {
  const { drops } = drive([{ ...SMOOTH, costMs: 0 }, { seconds: 8, intervalMs: 1000 / 30, costMs: 0 }]);
  expect(drops.map((d) => d.tier)).toEqual(['medium']);
});

test('start-up does not count', () => {
  // Five seconds of 10 fps while the first shaders build, then smooth.
  expect(drive([{ seconds: 5, intervalMs: 100 }, SMOOTH]).drops).toEqual([]);
});

test('frames stopping and starting again (a load, a hidden tab) starts the count over', () => {
  // Three seconds of slow frames, a 30 s gap, three more: neither stretch is
  // four seconds, and they must not add up.
  const { drops } = drive([
    SMOOTH,
    { seconds: 3, intervalMs: 40 },
    { seconds: 0, intervalMs: 30_000 },
    { seconds: 3, intervalMs: 40 },
    SMOOTH,
  ]);
  expect(drops).toEqual([]);
});

test('a device at two frames a second does get dropped, eventually', () => {
  const { drops } = drive([SMOOTH, { seconds: 30, intervalMs: 500 }]);
  expect(drops.length).toBeGreaterThan(0);
  // Forty frames at the least: each one counts for a tenth of a second.
  expect((drops[0].at - 1000) / 1000 - SMOOTH.seconds).toBeGreaterThan(19);
});

test('switched off, it does nothing', () => {
  let dropped = false;
  const adaptive = new AdaptiveQuality(55, () => (dropped = true));
  adaptive.enabled = false;
  for (let now = 0; now < 60_000; now += 50) adaptive.update({ now, intervalMs: 50, costMs: 45, tier: 'high' });
  expect(dropped).toBe(false);
});
