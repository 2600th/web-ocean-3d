/**
 * The saved voyage, as far as the title screen needs to know it.
 *
 * The voyage code keeps its progress in localStorage under
 * `web-ocean:voyage:v1`. The title screen only asks two questions of it: is
 * there a record, and what are the gold and the voyages completed. This is the
 * one function that answers, so that when the voyage module exports its own
 * typed reader (`src/gameplay/`), swapping it in is one line here.
 *
 * Read defensively: the key may be absent, the storage may refuse, the JSON
 * may be malformed or from another build. Anything that is not a plain object
 * is "no record". Fields are taken by the names the voyage code is expected to
 * use and ignored if they are not finite, non-negative numbers; a record with
 * no figures still counts as a record, and the title shows no line for them.
 *
 * Pure: no three.js.
 */

export const VOYAGE_KEY = 'web-ocean:voyage:v1';

export interface VoyageRecord {
  /** Gold in the hold, when the record says. */
  readonly gold: number | null;
  /** Voyages completed, when the record says. */
  readonly voyages: number | null;
}

function figure(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : null;
}

function pick(source: Record<string, unknown>, names: readonly string[]): number | null {
  for (const name of names) {
    const found = figure(source[name]);
    if (found !== null) return found;
  }
  return null;
}

/** The saved voyage, or `null` when there is none worth continuing. Never throws. */
export function readVoyageRecord(): VoyageRecord | null {
  try {
    const raw = window.localStorage.getItem(VOYAGE_KEY);
    if (raw === null) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
    const record = parsed as Record<string, unknown>;
    // The figures may sit at the top or in a `profile` / `progress` object.
    const nested = [record.profile, record.progress, record.stats].find(
      (candidate): candidate is Record<string, unknown> =>
        typeof candidate === 'object' && candidate !== null && !Array.isArray(candidate),
    );
    const sources = nested === undefined ? [record] : [record, nested];
    let gold: number | null = null;
    let voyages: number | null = null;
    for (const source of sources) {
      gold ??= pick(source, ['gold', 'doubloons']);
      voyages ??= pick(source, ['voyagesCompleted', 'voyages', 'completed', 'completedVoyages']);
    }
    return { gold, voyages };
  } catch {
    return null;
  }
}
