/**
 * The comfort and access settings, remembered between visits.
 *
 * The same pattern as `soundPrefs`: only the keys listed here are kept, every
 * read and write is guarded, and a stored value from an older build that is
 * missing or the wrong type leaves the default standing. The look, the weather
 * and the camera mode are what a visit is for and are not kept; the start gate
 * keeps the quality.
 *
 * `applyDisplayPrefs` is the one place the display-side settings reach the
 * page: the interface size, the shape cues and the reduced-motion override
 * become attributes on `<html>`, which the stylesheet reads, and the override
 * is handed to `motion.ts` so every part of the game that asks the system's
 * `prefers-reduced-motion` hears the player's choice too.
 */

import { setReducedMotionOverride } from './motion.ts';
import { UI_SCALES, type UiScale, type UiState } from './types.ts';

const STORAGE_KEY = 'web-ocean-3d:settings';

const NUMBERS = {
  shake: [0, 1],
  lookSensitivity: [0.4, 2],
} as const;
const FLAGS = ['haptics', 'invertLook', 'reduceMotion', 'reduceFlashes', 'shapeCues'] as const;

export type SettingsKey = keyof typeof NUMBERS | (typeof FLAGS)[number] | 'uiScale';

export const SETTINGS_KEYS: readonly SettingsKey[] = [
  ...(Object.keys(NUMBERS) as (keyof typeof NUMBERS)[]),
  ...FLAGS,
  'uiScale',
];

export function isSettingsKey(key: string): key is SettingsKey {
  return (SETTINGS_KEYS as readonly string[]).includes(key);
}

/** The remembered settings, or nothing. Never throws. */
export function loadSettingsPrefs(): Partial<Pick<UiState, SettingsKey>> {
  const out: Partial<Pick<UiState, SettingsKey>> = {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === null) return out;
    const stored = JSON.parse(raw) as Record<string, unknown>;
    for (const key of Object.keys(NUMBERS) as (keyof typeof NUMBERS)[]) {
      const v = stored[key];
      const [min, max] = NUMBERS[key];
      if (typeof v === 'number' && Number.isFinite(v)) out[key] = Math.min(max, Math.max(min, v));
    }
    for (const key of FLAGS) {
      const v = stored[key];
      if (typeof v === 'boolean') out[key] = v;
    }
    const scale = stored.uiScale;
    if (typeof scale === 'number' && (UI_SCALES as readonly number[]).includes(scale)) {
      out.uiScale = scale as UiScale;
    }
  } catch {
    // Unavailable or unreadable: the defaults stand.
  }
  return out;
}

/** Remembers the settings in `state`. Never throws. */
export function saveSettingsPrefs(state: Readonly<UiState>): void {
  try {
    const stored: Record<string, number | boolean> = {};
    for (const key of SETTINGS_KEYS) stored[key] = state[key];
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  } catch {
    // Storage refused: the settings last for this visit only.
  }
}

/** True where the browser can scale the interface (CSS `zoom`: Chromium, Safari, Firefox 126+). */
export function canScaleInterface(): boolean {
  try {
    return typeof CSS !== 'undefined' && CSS.supports('zoom', '1.15');
  } catch {
    return false;
  }
}

/**
 * The size actually applied. The interface is laid out for a smallest viewport
 * in each of its three layouts, and scaling it up shrinks the room it has, so
 * a step is held back where it would push the instruments into each other: on
 * a 720p laptop "Larger" becomes the most that still fits. Never below 1.
 */
export function effectiveScale(step: number): number {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const [minW, minH] = h <= 520 && w > h ? [700, 360] : w < 900 ? [340, 560] : [1100, 640];
  return Math.max(1, Math.min(step, w / minW, h / minH));
}

let scaleStep = 1;
let resizeWatched = false;

function paintScale(): void {
  document.documentElement.style.setProperty(
    '--ui-scale',
    canScaleInterface() ? effectiveScale(scaleStep).toFixed(3) : '1',
  );
}

/**
 * Puts the display-side settings on the page. Safe to call before the app
 * exists (the start gate does, so the menu honours them) and on every change.
 */
export function applyDisplayPrefs(
  prefs: Partial<Pick<UiState, 'reduceMotion' | 'uiScale' | 'shapeCues'>>,
): void {
  const html = document.documentElement;
  if (prefs.reduceMotion !== undefined) {
    setReducedMotionOverride(prefs.reduceMotion);
    if (prefs.reduceMotion) html.setAttribute('data-reduce-motion', 'on');
    else html.removeAttribute('data-reduce-motion');
  }
  if (prefs.uiScale !== undefined) {
    scaleStep = prefs.uiScale;
    paintScale();
    if (!resizeWatched) {
      resizeWatched = true;
      window.addEventListener('resize', paintScale, { passive: true });
    }
  }
  if (prefs.shapeCues !== undefined) {
    if (prefs.shapeCues) html.setAttribute('data-shape-cues', 'on');
    else html.removeAttribute('data-shape-cues');
  }
}
