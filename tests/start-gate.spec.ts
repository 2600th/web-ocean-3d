import { test, expect } from '@playwright/test';

test('the start gate applies the selected quality tier before boot continues', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('#boot-gate')).toBeVisible({ timeout: 5_000 });
  await page.getByRole('button', { name: 'Low' }).click();
  await page.getByRole('button', { name: 'New voyage' }).click();

  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 120_000 });
  await expect.poll(
    () => page.evaluate(() =>
      (window as unknown as { __ocean: { getState(): { quality: string } } })
        .__ocean.getState().quality,
    ),
    { timeout: 120_000 },
  ).toBe('low');
});

test('the title offers Continue only when a voyage is on record', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#boot-gate')).toBeVisible({ timeout: 5_000 });
  await expect(page.getByRole('button', { name: 'Continue' })).toHaveCount(0);
  await expect(page.locator('.boot__record')).toHaveText('No voyage on record yet.');

  // A malformed record is no record.
  await page.evaluate(() => localStorage.setItem('web-ocean:voyage:v1', '{not json'));
  await page.reload();
  await expect(page.locator('#boot-gate')).toBeVisible({ timeout: 5_000 });
  await expect(page.getByRole('button', { name: 'Continue' })).toHaveCount(0);

  await page.evaluate(() =>
    localStorage.setItem('web-ocean:voyage:v1', JSON.stringify({ gold: 120, voyages: 3, bestTimeS: null })));
  await page.reload();
  await expect(page.getByRole('button', { name: 'Continue' })).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.boot__record')).toContainText('Gold 120');
  await expect(page.locator('.boot__record')).toContainText('Voyages completed 3');
  await expect(page.getByRole('button', { name: 'New voyage' })).toBeVisible();
});
