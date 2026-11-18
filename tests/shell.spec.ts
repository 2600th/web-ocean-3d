import { test, expect } from '@playwright/test';
import { collectConsoleErrors, setState, waitForOcean } from './helpers';

/**
 * The shell: Escape pauses the simulation and takes the keys from the game, the
 * settings are saved and come back after a reload, and a rebound key is the key
 * the helm answers to.
 */
test('Escape pauses, the settings are saved, and a rebound key steers the helm', async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await page.goto('/?gate=0&quality=low');
  await waitForOcean(page, 1200);
  await setState(page, { cameraMode: 'boat' });

  const sail = () =>
    page.evaluate(() => (window as unknown as { __ocean: { gameplay: { state(): { sail: string } } } })
      .__ocean.gameplay.state().sail);
  const clock = () =>
    page.evaluate(() => (window as unknown as { __ocean: { elapsedTime(): number } }).__ocean.elapsedTime());
  const dialog = page.getByRole('dialog');

  // ---- pause: the clock stops, the keys no longer reach the helm ---------------
  await page.evaluate(() => (window as unknown as { __ocean: { gameplay: { reset(): void } } }).__ocean.gameplay.reset());
  await page.locator('#viewport').focus();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Resume' })).toBeFocused();
  const frozen = await clock();
  const before = await sail();
  await page.keyboard.press('KeyW');
  await page.waitForTimeout(600);
  expect(await clock()).toBe(frozen);
  expect(await sail(), 'W in the menu must not set sail').toBe(before);

  // Focus stays inside the menu however many times Tab is pressed.
  for (let i = 0; i < 7; i++) await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.activeElement?.closest('.shell') !== null)).toBe(true);

  // ---- settings: a switch, a slider, a key --------------------------------------
  await dialog.getByRole('button', { name: 'Settings' }).click();
  await dialog.getByRole('switch', { name: /Reduce motion/ }).click();
  await expect(page.locator('html')).toHaveAttribute('data-reduce-motion', 'on');
  await dialog.getByRole('switch', { name: /Reduce flashes/ }).click();
  await dialog.getByRole('radio', { name: 'Large', includeHidden: true }).check({ force: true });

  await dialog.getByRole('button', { name: /^Steer to port: A/ }).click();
  await page.keyboard.press('KeyJ');
  await expect(dialog.getByRole('button', { name: /^Steer to port: J/ })).toBeVisible();
  // A key another action holds is swapped, and the screen says so.
  await dialog.getByRole('button', { name: /^Drop or weigh anchor: R/ }).click();
  await page.keyboard.press('KeyJ');
  await expect(page.locator('.settings__live')).toContainText('took');
  await expect(dialog.getByRole('button', { name: /^Steer to port: R/ })).toBeVisible();

  const stored = await page.evaluate(() => ({
    settings: JSON.parse(localStorage.getItem('web-ocean-3d:settings') ?? '{}'),
    keys: JSON.parse(localStorage.getItem('web-ocean-3d:keys') ?? '{}'),
  }));
  expect(stored.settings).toMatchObject({ reduceMotion: true, reduceFlashes: true, uiScale: 1.15 });
  expect(stored.keys).toMatchObject({ anchor: 'KeyJ', steerPort: 'KeyR' });

  // Escape steps back, then resumes, and the clock runs again.
  await page.keyboard.press('Escape');
  await expect(dialog.getByRole('button', { name: 'Resume' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect.poll(clock).toBeGreaterThan(frozen);

  // The rebound anchor key now does the anchor's job; the old one does not.
  await page.keyboard.press('KeyR');
  expect((await page.evaluate(() =>
    (window as unknown as { __ocean: { gameplay: { state(): { anchor: string } } } }).__ocean.gameplay.state().anchor))).toBe('up');
  await page.keyboard.press('KeyJ');
  await expect.poll(() => page.evaluate(() =>
    (window as unknown as { __ocean: { gameplay: { state(): { anchor: string } } } }).__ocean.gameplay.state().anchor)).not.toBe('up');

  // ---- remembered after a reload ---------------------------------------------------
  await page.reload();
  await waitForOcean(page, 1200);
  await expect(page.locator('html')).toHaveAttribute('data-reduce-motion', 'on');
  const state = await page.evaluate(() =>
    (window as unknown as { __ocean: { getState(): Record<string, unknown> } }).__ocean.getState());
  expect(state).toMatchObject({ reduceMotion: true, reduceFlashes: true, uiScale: 1.15 });

  expect(errors).toEqual([]);
});
