import { test, expect } from '@playwright/test';

test('the start gate applies the selected quality tier before boot continues', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('#boot-gate')).toBeVisible({ timeout: 5_000 });
  await page.getByRole('button', { name: 'Low' }).click();
  await page.getByRole('button', { name: 'Start the sea' }).click();

  await page.waitForFunction(() => '__ocean' in window, undefined, { timeout: 120_000 });
  await expect.poll(
    () => page.evaluate(() =>
      (window as unknown as { __ocean: { getState(): { quality: string } } })
        .__ocean.getState().quality,
    ),
    { timeout: 120_000 },
  ).toBe('low');
});
