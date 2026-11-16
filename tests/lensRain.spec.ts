import { expect, test } from '@playwright/test';
import { LensRain } from '../src/post/LensRain';

/**
 * What the lens does when the camera goes under the water and comes back.
 *
 * CPU-only, for the same reason `fft.spec` is: the thing that decides this
 * behaviour is a handful of scalars updated on the CPU, and the shader below
 * them is a pure function of the uniforms they write. Asserting on the uniforms
 * is therefore asserting on what is drawn, without a GPU in the loop.
 *
 * **Three defects are pinned here, all of them observed rather than imagined.**
 *
 *   1. Surfacing was detected as a single-frame *edge* — `submersion` falling
 *      from above 0.5 to below 0.05 between two consecutive updates. That is
 *      0.315 m of a 0.7 m band in one step, so a camera rising at 10.5 m/s did
 *      not trigger it and nothing but a teleport ever did. Measured: a 60-frame
 *      ascent through the surface left the lens completely dry.
 *   2. When it *did* fire it set the rain coverage to 1 — the storm maximum —
 *      so a dunk on a cloudless day put a full downpour on the glass.
 *   3. And then dried it on the rain constant, which is 7.5 s: still visibly
 *      wet 15 s later, effectively clear only after 30.
 *
 * The replacement is a film that charges while the lens is under water and
 * drains when it is not, with no edge test anywhere.
 */

/** Runs `seconds` of updates at 60 Hz with submersion held, and returns the rig. */
function run(rain: LensRain, submersion: number, seconds: number): void {
  const dt = 1 / 60;
  for (let i = 0; i < Math.round(seconds / dt); i++) {
    rain.setSubmersion(submersion);
    rain.update(dt);
  }
}

/** Coverage the droplet lattices are driven with — what the shader reads. */
const coverage = (rain: LensRain): number => (rain as any).uAmount.value as number;
/** Weight on the misting and the backdrop defocus. */
const glass = (rain: LensRain): number => (rain as any).uGlass.value as number;

/** Submerged, then lifted clear over `riseSeconds` — a swim up, not a cut. */
function ascend(rain: LensRain, riseSeconds = 0.5): void {
  const dt = 1 / 60;
  run(rain, 1, 1.5);
  const steps = Math.round(riseSeconds / dt);
  for (let i = 1; i <= steps; i++) {
    rain.setSubmersion(1 - i / steps);
    rain.update(dt);
  }
}

test.describe('the lens coming out of the water', () => {
  test('a smooth ascent wets the glass', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(0); // Not raining. Nothing but the dive may wet this.

    ascend(rain);

    // The defect this replaces left it at exactly zero: the edge test needed the
    // whole band crossed inside one frame, which a swim never does.
    expect(
      coverage(rain),
      'rising through the surface left the lens dry — the wetting is keyed on a ' +
        'single-frame edge again rather than on having been under the water',
    ).toBeGreaterThan(0.3);
  });

  test('and clears within a few seconds, not half a minute', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(0);

    ascend(rain);
    run(rain, 0, 4);

    // Water drains off a lens; it does not evaporate off one. The rain constant
    // is 7.5 s and left this at 0.59 here, still plainly wet.
    expect(
      coverage(rain),
      'the film is still on the glass four seconds after surfacing — it is ' +
        'draining on the rain constant rather than on its own',
    ).toBeLessThan(0.05);
  });

  test('never reaches the strength a storm does', () => {
    const wet = new LensRain();
    wet.setQuality(3);
    wet.setIntensity(0);
    ascend(wet);

    const storm = new LensRain();
    storm.setQuality(3);
    storm.setIntensity(1);
    run(storm, 0, 20);

    // Drop count is not compared: rain's drops were cut to `RAIN_DROP_SHARE` of
    // the lattice after the A/B critic found them dominating the storm frame,
    // while a fresh film is meant to bead heavily and drain in a few seconds.
    // The defocus is the term that ruins the frame: at full authored weight it
    // replaces 70% of every pixel outside a droplet with a ring of taps, which
    // ghosts rigging and turns sun glitter into a milky wash. Neither a dive
    // nor a storm may ask for more than a share of it, and a dive no more than
    // the storm does — see the attribution captures in the plan doc.
    expect(
      glass(wet),
      'surfacing is driving the glass defocus harder than a storm does',
    ).toBeLessThanOrEqual(glass(storm) + 1e-6);
    expect(
      glass(wet),
      'surfacing is driving the glass defocus near full authored weight',
    ).toBeLessThan(0.45);
  });

  test('a splash wets less than a dive', () => {
    const splash = new LensRain();
    splash.setQuality(3);
    splash.setIntensity(0);
    run(splash, 1, 0.05);
    run(splash, 0, 0.2);

    const dive = new LensRain();
    dive.setQuality(3);
    dive.setIntensity(0);
    run(dive, 1, 3);
    run(dive, 0, 0.2);

    expect(
      coverage(splash),
      'a crest crossing the lens for a twentieth of a second leaves as much ' +
        'water as a dive does — the film is charging as a switch, not a rate',
    ).toBeLessThan(coverage(dive) * 0.75);
  });

  test('does not bead while the lens is awash at the waterline', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(0);

    // An eye a few centimetres under, crests washing over it: `submersion` sits
    // near 0.47 and dips below 0.45 as each one passes. That dip is enough to
    // open the gate rain uses, and it must not open this one — the `waterline`
    // baseline came back beaded from top to bottom, over its submerged half
    // included, where there is no air for a drop to bead against.
    run(rain, 1, 2);
    for (let i = 0; i < 120; i++) {
      rain.setSubmersion(0.47 + 0.06 * Math.sin(i / 6));
      rain.update(1 / 60);
    }

    expect(
      coverage(rain),
      'the lens is beading while it is still in the sea',
    ).toBeLessThan(0.01);
  });

  test('shows nothing while the lens is still under', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(1); // Raining hard, and none of it reaches a submerged lens.
    run(rain, 1, 3);
    expect(coverage(rain)).toBe(0);
    expect(glass(rain)).toBe(0);
  });

  test('leaves the rain path exactly as it was', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(1);
    run(rain, 0, 20);

    // Rain drives coverage and glass together, each a fixed share of the
    // wetness (`RAIN_DROP_SHARE` drops, `RAIN_GLASS` misting and defocus) set
    // from the storm-shot attribution. The dive's film must not leak into it.
    expect(coverage(rain)).toBeCloseTo(rain.getWetness() * 0.2, 6);
    expect(glass(rain)).toBeCloseTo(rain.getWetness() * 0.3, 6);
    expect(rain.getWetness()).toBeGreaterThan(0.99);
  });

  test('a reset leaves no film behind', () => {
    const rain = new LensRain();
    rain.setQuality(3);
    rain.setIntensity(0);
    ascend(rain);
    expect(coverage(rain)).toBeGreaterThan(0.3);

    // A capture is a teleport. Carrying a film across one is how the clear-day
    // gallery shot got photographed through a wet lens; see `resetClock`.
    rain.resetClock(0);
    expect(coverage(rain)).toBe(0);
    expect(glass(rain)).toBe(0);
  });
});
