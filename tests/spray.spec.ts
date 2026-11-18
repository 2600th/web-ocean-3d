import { test, expect } from '@playwright/test';
import { bootOcean } from './lib/capture';

/**
 * Hull impact spray.
 *
 * Two failures look identical in a screenshot — the hull is not slamming, and
 * the hull is slamming and nothing is drawn — so this asserts both halves
 * separately: that impacts are being detected at all, and that the detected
 * impacts put light on the screen.
 */

/**
 * How the spray is told apart from everything else that moves in a storm.
 *
 * The first version captured twice in a row, spray on and then off, and
 * counted the pixels that changed. That cannot work, for two reasons that
 * have nothing to do with the spray:
 *
 *   - A capture is a whole frame with its own TRAA jitter and history step, so
 *     two captures in a row differ by the resolve alone; in a storm with rain
 *     that was about a third of the frame, far above a sheet of spray.
 *   - Switching the spray off for one frame leaves it in the TRAA history,
 *     which is most of what the resolve shows.
 *
 * What the harness does reproduce is a capture *from a reset*: the same reset
 * and the same steps give the same frame. So the voyage is sailed three times
 * from one reset, step for step: twice with the spray on (the control pair,
 * which measures what two identical runs differ by) and once with it off for
 * the whole run. The spray has to change many times more pixels than the
 * control pair differs by.
 *
 * History: the system used to draw too little to see. Real impacts were weak
 * (a half-metre puff at a fifth of its opacity), and in this preset they were
 * fogged out entirely, because a particle that writes no depth is fogged as
 * whatever is behind it; `Spray` now writes its cores' depth in a separate
 * pass.
 */
test('a hull driven through a storm throws spray, and it is drawn', async ({ page }) => {
  test.setTimeout(600_000);
  await bootOcean(page);

  const result = await page.evaluate(async () => {
    const ocean = window.__ocean as unknown as {
      setState(p: Record<string, unknown>): Promise<void> | void;
      resetDeterministic(
        t?: number,
        steps?: number,
        shipInput?: { throttle: number; rudder: number } | null,
      ): Promise<void>;
      step(dt: number, steps?: number): Promise<void>;
      capturePixels(): Promise<{ width: number; height: number; data: Uint8Array }>;
      setSprayStrength(s: number): void;
      sprayLiveEvents(): number;
      director: { snapToTarget(): void };
      setShipInput(throttle: number, rudder: number): void;
      shipState(): { speed: number } | null;
    };

    // A storm, and the ship driven hard into it. Spray is a *slam* — the hull
    // arriving where the water already is — so a drifting hull produces none by
    // design and would prove nothing either way.
    //
    // Orbit mode because it targets the hull: the camera has to be looking at
    // the bow for the second assertion to mean anything, and hard-coding a
    // position would need the ship's, which it has moved away from by then.
    await ocean.setState({
      preset: 'storm',
      windSpeed: 21,
      quality: 'high',
      cameraMode: 'orbit',
    });

    /**
     * One voyage from the reset: the run-up, then `rounds` two-step rounds,
     * then (unless scouting) the capture. Every voyage issues the same calls
     * in the same order up to its last round, which is what makes two of them
     * comparable.
     */
    const voyage = async (strength: number, rounds: number, capture: boolean) => {
      ocean.setSprayStrength(strength);
      await ocean.resetDeterministic(40, 90, { throttle: 1, rudder: 0 });
      // The reset seeds the input; this holds it. Without it the throttle
      // decays back to neutral and the hull never gets past a knot.
      ocean.setShipInput(1, 0);
      // Six seconds for the hull to start working in the sea. Stepped rather
      // than run on the wall clock so the count is the same on every machine
      // and on every voyage here.
      for (let i = 0; i < 60; i++) await ocean.step(1 / 60, 6);
      ocean.director.snapToTarget();
      const live: number[] = [];
      for (let r = 0; r < rounds; r++) {
        await ocean.step(1 / 60, 2);
        live.push(ocean.sprayLiveEvents());
      }
      const speed = ocean.shipState()?.speed ?? 0;
      const frame = capture ? await ocean.capturePixels() : null;
      return { frame, live, speed };
    };

    // Pixels that differ by more than rounding, and how many of those are
    // brighter in `a`. Counted pixel by pixel rather than by mean brightness:
    // a sheet of spray is a few hundred pixels out of a million, so its effect
    // on the mean is below the fourth decimal.
    const compare = (a: Uint8Array, b: Uint8Array) => {
      let changed = 0;
      let brighter = 0;
      for (let i = 0; i < a.length; i += 4) {
        const d = a[i] - b[i] + (a[i + 1] - b[i + 1]) + (a[i + 2] - b[i + 2]);
        if (Math.abs(d) > 6) {
          changed++;
          if (d > 0) brighter++;
        }
      }
      return { changed, brighter };
    };

    // Capture *while a sheet is in the air*, not merely after one has been.
    // Spray lives about a second, so a fixed step count photographs whatever
    // phase it lands on, usually an empty sky, and the instant an impact is
    // first registered photographs a sheet that has not left the hull yet. A
    // scouting voyage finds the round with the most impacts live in five
    // seconds of sailing, and the captures are taken a fifth of a second
    // after it, once those sheets have opened.
    const SCOUT_ROUNDS = 150;
    const scout = await voyage(1, SCOUT_ROUNDS, false);
    let best = 0;
    for (let r = 1; r < scout.live.length; r++) if (scout.live[r] > scout.live[best]) best = r;
    const rounds = Math.min(SCOUT_ROUNDS, best + 1 + 6);

    const first = await voyage(1, rounds, true);
    const again = await voyage(1, rounds, true);
    const without = await voyage(0, rounds, true);
    ocean.setSprayStrength(1);

    const control = compare(first.frame!.data, again.frame!.data);
    const spray = compare(first.frame!.data, without.frame!.data);

    return {
      peakLive: scout.live[best],
      liveAtCapture: first.live[first.live.length - 1],
      liveAgain: again.live[again.live.length - 1],
      rounds,
      control: control.changed,
      changed: spray.changed,
      brighter: spray.brighter,
      total: first.frame!.data.length / 4,
      shipSpeed: first.speed,
      speedAgain: again.speed,
    };
  });

  console.log(
    `[spray] peak ${result.peakLive} live-at-capture ${result.liveAtCapture} (again ${result.liveAgain}) ` +
      `after ${result.rounds} rounds | ship ${result.shipSpeed.toFixed(3)} m/s (again ${result.speedAgain.toFixed(3)}) | ` +
      `control pair differs in ${result.control} px | spray changed ${result.changed} of ${result.total} ` +
      `(${result.brighter} brighter)`,
  );

  // The hull has to be working in the sea, or the test proves nothing. Left
  // deliberately low: what spray needs is *vertical* motion through the surface,
  // which a hull gets from the swell whether or not it is making way.
  expect(result.shipSpeed, 'the ship never moved at all').toBeGreaterThan(0.1);

  // Impacts are being detected. One is enough to prove the crossing test fires;
  // the interesting failure is zero, which means the probe logic never triggers.
  expect(result.peakLive, 'no impact registered in a 21 m/s sea').toBeGreaterThan(0);
  expect(result.liveAtCapture, 'no sheet was in the air at the capture').toBeGreaterThan(0);

  // The replay is a replay. If two voyages from one reset do not arrive at the
  // same instant, nothing below compares like with like.
  expect(result.liveAgain, 'the second voyage did not repeat the first').toBe(result.liveAtCapture);
  expect(result.speedAgain).toBeCloseTo(result.shipSpeed, 6);

  // And the impacts reach the frame. Measured when this was written: the
  // control pair identical to the pixel, and the spray changing 18 695 pixels
  // of 1 440 000 with nine impacts live. The floor is a ninth of that, far
  // above anything two identical voyages have differed by (148 pixels, in an
  // earlier form of this test that captured at a different instant).
  expect(
    result.changed,
    'spray registered impacts but changed almost no pixels: it is emitting and not drawing',
  ).toBeGreaterThan(2000);
  expect(
    result.changed,
    `the spray changed ${result.changed} pixels, but two identical voyages differed by ${result.control}: ` +
      'the measurement is inside its own noise',
  ).toBeGreaterThan(result.control * 20);

  // Spray is white water, and mostly it adds light. Not all of it: it is
  // blended, not added, so a thin sheet in front of a white crest is darker
  // than the crest (63% brighter when measured). A majority going the other
  // way would mean the quads are occluding rather than accumulating, which is
  // what a depth-write or blend-mode mistake looks like from outside.
  expect(result.brighter, 'most of what the spray changed got darker').toBeGreaterThan(result.changed * 0.5);
});
