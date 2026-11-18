import { test, expect, type Page } from '@playwright/test';
import { setState, waitForOcean } from './helpers';

/**
 * The voyage (`src/gameplay/Voyage.ts`): it opens, gathers three casks, turns
 * for home, can lose a cask in a squall and win it back, lands with a tally
 * when the anchor goes down off the cove, keeps the result in localStorage,
 * and varies by seed while staying frozen under a deterministic capture.
 *
 * Driven through the gameplay test surface rather than by sailing a real
 * three-minute course: what is under test is the bookkeeping and the HUD it
 * feeds, not the handling, which `ocean.spec.ts` covers.
 */

const KEY = 'web-ocean:voyage:v1';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const gameplay = (page: Page) => page.evaluate(() => (window as any).__ocean.gameplay.state());

test('a voyage gathers its cargo, lands with a tally, keeps the result and varies by seed', async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto('/?gate=0&quality=low');
  await waitForOcean(page);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  await page.waitForFunction(() => (window as any).__ocean.gameplay !== undefined, undefined, { timeout: 120_000 });
  await page.evaluate((key) => localStorage.removeItem(key), KEY);
  await setState(page, { preset: 'seaOfThieves', cameraMode: 'boat' });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  await page.evaluate(() => (window as any).__ocean.resetDeterministic(10, 30));

  // Under a capture every clock is frozen: no time, no squall, no day.
  const frozen = await page.evaluate(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = (window as any).__ocean.gameplay;
    g.voyage.skip(300);
    return g.state().voyage;
  });
  expect(frozen.frozen).toBe(true);
  expect(frozen.seconds).toBe(0);
  expect(frozen.squallRain).toBe(0);

  // The same seed lays the same first cask and the same squall; another does not.
  const runs = await page.evaluate(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = (window as any).__ocean.gameplay;
    const run = (seed: number) => {
      g.voyage.begin(seed);
      const first = g.state().target;
      let waited = 0;
      while (g.state().voyage.squall === 'none' && waited < 400) {
        g.voyage.skip(1);
        waited += 1;
      }
      return { x: first.x, z: first.z, squallWarnedAfter: waited };
    };
    return [run(1234), run(1234), run(999)];
  });
  expect(runs[0]).toEqual(runs[1]);
  expect(Math.hypot(runs[0].x - runs[2].x, runs[0].z - runs[2].z)).toBeGreaterThan(1);
  expect(runs[0].squallWarnedAfter).toBeGreaterThan(30);
  expect(runs[0].squallWarnedAfter).toBeLessThan(200);

  // A fresh voyage: the ship's log opens it.
  await page.evaluate(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).__ocean.gameplay.voyage.begin(4321);
  });
  await expect(page.locator('.toast__text', { hasText: "Ship's log" }).first()).toBeVisible();
  expect((await gameplay(page)).voyage.phase).toBe('gather');

  // Three casks hauled aboard, then she turns for home.
  const haul = async () => {
    await page.evaluate(async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const o = (window as any).__ocean;
      o.gameplay.placeTargetAhead(6);
      await o.step(1 / 60, 30);
    });
  };
  for (let i = 1; i <= 3; i++) {
    await haul();
    expect((await gameplay(page)).cargo, `cask ${i} was not hauled aboard`).toBe(i);
  }
  expect((await gameplay(page)).voyage.phase).toBe('homeward');
  await expect(page.locator('.objective__title')).toHaveText('Bring the cargo home');

  // The squall comes, the HUD says so, and a cask goes over the side and
  // floats astern as the target; hauling it back sends her home again.
  const squall = await page.evaluate(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = (window as any).__ocean.gameplay;
    for (let i = 0; i < 80 && g.state().voyage.squall !== 'here'; i++) g.voyage.skip(2);
    return g.state().voyage;
  });
  expect(squall.squall).toBe('here');
  expect(squall.squallRain).toBeGreaterThan(0.2);
  await expect(page.locator('.objective__warning')).toContainText('Squall');
  const goldBefore = (await gameplay(page)).gold;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  await page.evaluate(() => (window as any).__ocean.gameplay.voyage.loseCask());
  const lost = await gameplay(page);
  expect(lost.cargo).toBe(2);
  expect(lost.gold).toBeLessThan(goldBefore);
  expect(lost.voyage.phase).toBe('gather');
  await expect(page.locator('.toast__text', { hasText: 'over the side' }).first()).toBeVisible();
  await haul();
  const recovered = await gameplay(page);
  expect(recovered.cargo).toBe(3);
  expect(recovered.gold).toBe(goldBefore);
  expect(recovered.voyage.phase).toBe('homeward');

  // Anchor off the cove: the voyage ends, the tally shows, the result is kept.
  await page.evaluate(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = (window as any).__ocean.gameplay;
    g.voyage.anchorHere();
    g.dropAnchor();
  });
  const landed = await gameplay(page);
  expect(landed.voyage.phase).toBe('landed');
  expect(landed.voyage.tally.cargo).toBe(3);
  const tally = page.locator('.tally');
  await expect(tally).toBeVisible();
  await expect(tally).toContainText('3 of 3');
  await expect(page.locator('.objective')).toBeHidden();
  const kept = JSON.parse((await page.evaluate((key) => localStorage.getItem(key), KEY)) ?? 'null');
  expect(kept).toEqual({ gold: landed.voyage.tally.gold, voyages: 1, bestTimeS: expect.any(Number) });

  // Sail again: a second voyage, numbered from the save.
  await page.locator('.tally__again').click();
  const again = await gameplay(page);
  expect(again.voyage.phase).toBe('gather');
  expect(again.voyage.number).toBe(2);
  expect(again.cargo).toBe(0);
  await expect(tally).toBeHidden();

  // A capture's rewind freezes it again on the fixed seed.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  await page.evaluate(() => (window as any).__ocean.resetDeterministic(10, 30));
  const reset = await gameplay(page);
  expect(reset.voyage.frozen).toBe(true);
  expect(reset.voyage.phase).toBe('gather');
  await page.evaluate((key) => localStorage.removeItem(key), KEY);
});
