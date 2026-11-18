/**
 * The sound settings, remembered between visits.
 *
 * The master volume used to reset to its default every visit, and the
 * per-kind levels and mutes are worth nothing if they do not stick. Only the
 * sound keys are kept: the look, the weather and the camera are what a visit
 * is for, and the start gate chooses the quality.
 *
 * Every read and write is guarded. Private browsing and locked-down embeds can
 * refuse `localStorage`, and a stored value from an older build may be missing
 * or the wrong type; either way the defaults stand.
 */

import type { UiState } from './types.ts';

const STORAGE_KEY = 'web-ocean-3d:sound';

const LEVELS = ['volume', 'musicVolume', 'ambienceVolume', 'effectsVolume', 'interfaceVolume'] as const;
const MUTES = ['musicMuted', 'ambienceMuted', 'effectsMuted', 'interfaceMuted', 'reducedRange'] as const;

export type SoundKey = (typeof LEVELS)[number] | (typeof MUTES)[number];

/** The state keys this module remembers. */
export const SOUND_KEYS: readonly SoundKey[] = [...LEVELS, ...MUTES];

export function isSoundKey(key: string): key is SoundKey {
  return (SOUND_KEYS as readonly string[]).includes(key);
}

/** The remembered sound settings, or nothing. Never throws. */
export function loadSoundPrefs(): Partial<Pick<UiState, SoundKey>> {
  const out: Partial<Pick<UiState, SoundKey>> = {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === null) return out;
    const stored = JSON.parse(raw) as Record<string, unknown>;
    for (const key of LEVELS) {
      const v = stored[key];
      if (typeof v === 'number' && Number.isFinite(v)) out[key] = Math.min(1, Math.max(0, v));
    }
    for (const key of MUTES) {
      const v = stored[key];
      if (typeof v === 'boolean') out[key] = v;
    }
  } catch {
    // Unavailable or unreadable: the defaults stand.
  }
  return out;
}

/** Remembers the sound settings in `state`. Never throws. */
export function saveSoundPrefs(state: Readonly<UiState>): void {
  try {
    const stored: Record<string, number | boolean> = {};
    for (const key of SOUND_KEYS) stored[key] = state[key];
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  } catch {
    // Storage refused: the settings last for this visit only.
  }
}
