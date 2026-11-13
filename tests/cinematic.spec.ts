import { test, expect } from '@playwright/test';
import * as THREE from 'three';
import { setState } from './helpers';
import { bootOcean } from './lib/capture';
import { CINEMATIC_LOOP_SECONDS, CinematicDirector } from '../src/cameras/Cinematic';
import { seafloorHeight } from '../src/scene/Seafloor';

/**
 * What the tour promises, asserted rather than trusted.
 *
 * The seam test in `ocean.spec.ts` already proves the *camera* is continuous at
 * the wrap. These cover the two things the re-authored flight added — a day that
 * genuinely reaches night, and weather that genuinely arrives — plus the one
 * property whose failure mode is completely silent: the hull sailing somewhere
 * boring during the beats that frame it.
 */

test.describe('the cinematic tour', () => {
  /**
   * Every shot moves smoothly, stays out of the ground, and the loop point is
   * a move rather than an edit.
   *
   * Run against `CinematicDirector` directly rather than through the browser:
   * with no hull and no wave field supplied the rig falls back on the nominal
   * circuit and flat water, so a full lap at 60 Hz is a few thousand CPU samples.
   * The live clearance — real crests, the real hull — is a browser measurement
   * and is not what this asserts.
   *
   * The tour is *cut*, so the walk is per shot: frames either side of a shot
   * boundary are skipped, and so are the whip pans, which are fast by design.
   * Everything else is held to the bounds the continuous flight was held to,
   * which still catch a key edited into a lurch.
   */
  test('moves smoothly within every shot and keeps clear of the terrain', () => {
    const director = new CinematicDirector();
    director.setEnabled(true);
    const pose = { position: new THREE.Vector3(), target: new THREE.Vector3(), fov: 50, roll: 0 };

    const STEPS = Math.round(CINEMATIC_LOOP_SECONDS * 60);
    const dt = CINEMATIC_LOOP_SECONDS / STEPS;
    const eye: THREE.Vector3[] = [];
    const dir: THREE.Vector3[] = [];
    const beat: string[] = [];
    for (let i = 0; i < STEPS; i++) {
      director.resetClock(i * dt);
      director.update(0, pose);
      eye.push(pose.position.clone());
      dir.push(pose.target.clone().sub(pose.position));
      beat.push(director.beatName);
    }

    const at = <T>(a: T[], i: number): T => a[((i % STEPS) + STEPS) % STEPS];
    // Inside a shot, and clear of the whip windows at either end of one.
    const settled = (i: number) => {
      for (let k = -24; k <= 24; k++) if (at(beat, i + k) !== beat[i]) return false;
      return true;
    };

    let peakSpeed = 0;
    let peakAccel = 0;
    let peakTurn = 0;
    let minReach = Infinity;
    let lowestOverGround = Infinity;
    const underwater = new Set<string>();
    for (let i = 0; i < STEPS; i++) {
      const b = eye[i];
      lowestOverGround = Math.min(lowestOverGround, b.y - seafloorHeight(b.x, b.z));
      if (b.y < 0) underwater.add(beat[i]);
      if (!settled(i)) continue;
      const a = at(eye, i - 1);
      const c = at(eye, i + 1);
      peakSpeed = Math.max(peakSpeed, c.distanceTo(a) / (2 * dt));
      peakAccel = Math.max(
        peakAccel,
        new THREE.Vector3().addVectors(a, c).addScaledVector(b, -2).length() / (dt * dt),
      );
      const u = at(dir, i - 1).clone().normalize();
      const w = at(dir, i + 1).clone().normalize();
      peakTurn = Math.max(peakTurn, (Math.acos(Math.min(1, Math.max(-1, u.dot(w)))) / (2 * dt)) * (180 / Math.PI));
      minReach = Math.min(minReach, dir[i].length());
    }

    // The wrap is the one boundary that must not be a cut.
    const seam: number[] = [];
    for (let i = STEPS - 30; i < STEPS + 60; i++) seam.push(at(eye, i + 1).distanceTo(at(eye, i)));
    const seamMedian = [...seam].sort((x, y) => x - y)[seam.length >> 1];
    const seamWorst = Math.max(...seam);

    console.log(
      `[cinematic] within shots: speed ${peakSpeed.toFixed(1)} m/s, accel ${peakAccel.toFixed(1)} m/s², ` +
        `view ${peakTurn.toFixed(1)} °/s, reach ${minReach.toFixed(0)} m; ` +
        `lowest over ground ${lowestOverGround.toFixed(1)} m; seam ${seamWorst.toFixed(3)} / ${seamMedian.toFixed(3)} m`,
    );

    expect(peakSpeed, 'a shot lurches').toBeLessThan(70);
    expect(peakAccel, 'a shot has a corner in it').toBeLessThan(80);
    expect(minReach, 'a look target passes too close to the lens').toBeGreaterThan(8);
    expect(peakTurn, 'a shot whips where it should pan').toBeLessThan(60);
    expect(lowestOverGround, 'the camera goes into the ground').toBeGreaterThan(2);
    expect([...underwater], 'only the dive goes under').toEqual(['reef-dive']);
    expect(seamWorst, 'the loop point is a cut').toBeLessThan(seamMedian * 4);
  });

  test('reaches night, rains, and closes its environment on the loop', async ({ page }) => {
    await bootOcean(page);

    const samples = await page.evaluate((loop) => {
      const out: Array<{ t: number; hours: number; rain: number; kind: string; cloud: number }> = [];
      for (let t = 0; t < loop; t += 0.25) {
        const e = window.__ocean.cinematicEnvironment(t);
        out.push({ t, hours: e.hours, rain: e.rain, kind: e.weatherKind, cloud: e.cloudCoverage });
      }
      return out;
    }, CINEMATIC_LOOP_SECONDS);

    // Sun elevation, from the same diurnal arc `sunFromClock` uses. Asserting on
    // the elevation rather than on the hour is what makes this mean "it gets
    // dark" instead of "a number went outside a range".
    const elevation = (h: number) => Math.sin(((h - 6) / 24) * Math.PI * 2) * 1.32;

    // It actually reaches night — the previous flight's sun bottomed out at
    // 08:18 and never came close.
    expect(Math.min(...samples.map((s) => elevation(s.hours)))).toBeLessThan(-0.5);
    // And it actually reaches the middle of the day, which a widened sine about
    // noon could not have done at the same time.
    expect(Math.max(...samples.map((s) => elevation(s.hours)))).toBeGreaterThan(1.0);

    // It actually rains, and declares the kind that makes rain visible at all.
    // Without the kind, `Weather` draws nothing and main's `raining` scalar is
    // zero, so every downstream effect stays dry — which would look like a bug
    // in the curves rather than a missing field.
    expect(samples.some((s) => s.rain > 0.5 && s.kind === 'rain')).toBe(true);
    // And it is dry for most of the lap, or it is not a *change*.
    expect(samples.filter((s) => s.rain < 0.001).length / samples.length).toBeGreaterThan(0.6);
    // Cloud follows the rain up.
    const wettest = samples.reduce((a, b) => (b.rain > a.rain ? b : a));
    const driest = samples.find((s) => s.rain === 0)!;
    expect(wettest.cloud).toBeGreaterThan(driest.cloud + 0.3);

    // Continuous at the wrap, to the same standard the pose is held to. `hours`
    // is an angle — 24 and 0 are the same sun — so the comparison is on the
    // circle rather than on the number.
    const first = samples[0];
    const last = samples[samples.length - 1];
    // Signed difference on the 24-hour circle, folded into [-12, 12). The naive
    // `a - b` reports nearly a full day across the wrap, which is the whole
    // reason this is done on the circle: the tour advances by exactly one day
    // per lap, so hour 24 and hour 0 are the same sun and the same frame.
    const circular = (a: number, b: number) => (((a - b + 12) % 24) + 24) % 24 - 12;
    expect(Math.abs(circular(first.hours, last.hours))).toBeLessThan(0.6);
    expect(Math.abs(first.rain - last.rain)).toBeLessThan(0.05);
  });

  test('keeps the hull over the plateau whenever it frames it', async ({ page }) => {
    await bootOcean(page);

    // The shots framed on the hull. They follow the live hull wherever it is,
    // but the circuit is what puts reef and turquoise water under it while they
    // do; the others are free to let it wander.
    const SHIP_BEATS = ['dawn-crane', 'wave-skim', 'hero-orbit', 'island-reveal', 'golden-hour', 'squall', 'night-watch'];

    const worst = await page.evaluate((names) => {
      let max = 0;
      for (const beat of window.__ocean.cinematicBeats()) {
        if (!names.includes(beat.name)) continue;
        for (let t = beat.start; t < beat.start + beat.duration; t += 0.25) {
          const { x, z } = window.__ocean.nominalShipAt(t);
          max = Math.max(max, Math.hypot(x, z));
        }
      }
      return max;
    }, SHIP_BEATS);

    // The shallow plateau runs to 320 m; past it the hull is over deep blue
    // water with no reef under it. This is asserted rather than left to the
    // radius constant because the failure is completely silent — the tour would
    // still loop perfectly and would simply have sailed somewhere with nothing
    // to look at. Note the hull's far point is *twice* the track radius: the
    // circuit passes through the origin rather than being centred on it.
    expect(worst).toBeLessThan(320);
  });

  test('the tour drives the sky, and hands it back on the way out', async ({ page }) => {
    await bootOcean(page);
    await setState(page, { preset: 'skyPro', cameraMode: 'cinematic' });

    // Park the tour in the night watch — about 01:00 on the tour's day — and
    // let it apply. A literal rather than a lookup, so it has to be revisited
    // when the tour is re-cut; it was 44 on the 60-second lap.
    const night = await page.evaluate(async () => {
      await window.__ocean.resetDeterministic(0, 30);
      window.__ocean.director.resetCinematic(85);
      await window.__ocean.step(1 / 60, 4);
      return window.__ocean.atmosphere.sunDirection.y;
    });
    expect(night).toBeLessThan(0);

    // Leaving hands the sun back to the preset, which is daylight.
    await setState(page, { cameraMode: 'orbit' });
    const day = await page.evaluate(async () => {
      await window.__ocean.step(1 / 60, 4);
      return window.__ocean.atmosphere.sunDirection.y;
    });
    expect(day).toBeGreaterThan(0);
  });
});
