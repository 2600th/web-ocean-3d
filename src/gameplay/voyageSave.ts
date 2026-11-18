/**
 * What a player keeps between visits: the gold landed, the voyages brought
 * home and the fastest one. One small typed module so the voyage (which
 * writes it) and the title screen's "continue" (which reads it) agree on the
 * key and the shape.
 *
 * **No three.js, no DOM beyond `localStorage`.** `src/ui` may import this file
 * directly (not through `gameplay/index.ts`, which pulls in the renderer).
 *
 * **The game works without it.** Private windows and locked-down embeds can
 * refuse storage or throw on access; every read falls back to an empty save
 * and every write reports whether it stuck. A stored value that is missing a
 * field, or holds a number that makes no sense, is read as empty for that
 * field rather than trusted.
 */

/** The localStorage key. Bump the suffix, never the shape, if the format changes. */
export const VOYAGE_SAVE_KEY = 'web-ocean:voyage:v1';

export interface VoyageSave {
  /** Gold landed at the cove across every voyage, whole coins. */
  gold: number;
  /** Voyages brought home (the anchor dropped off the cove with cargo aboard). */
  voyages: number;
  /** Fastest voyage that landed all its cargo, seconds; null until one has. */
  bestTimeS: number | null;
}

/** What a voyage reports when it lands. */
export interface VoyageResult {
  /** Gold landed this voyage. */
  gold: number;
  /** Seconds from the opening log line to the anchor dropping. */
  seconds: number;
  /** True when every cask the voyage laid out came home. Only these set a best time. */
  fullCargo: boolean;
}

export function emptyVoyageSave(): VoyageSave {
  return { gold: 0, voyages: 0, bestTimeS: null };
}

/** The browser's storage, or null where it is refused. */
function storage(): Storage | null {
  try {
    return typeof window !== 'undefined' && window.localStorage ? window.localStorage : null;
  } catch {
    return null;
  }
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
}

/** The save on this browser, or an empty one. Never throws. */
export function readVoyageSave(): VoyageSave {
  const store = storage();
  if (!store) return emptyVoyageSave();
  try {
    const raw = store.getItem(VOYAGE_SAVE_KEY);
    if (!raw) return emptyVoyageSave();
    const parsed = JSON.parse(raw) as Partial<Record<keyof VoyageSave, unknown>> | null;
    if (!parsed || typeof parsed !== 'object') return emptyVoyageSave();
    const best = parsed.bestTimeS;
    return {
      gold: count(parsed.gold),
      voyages: count(parsed.voyages),
      bestTimeS: typeof best === 'number' && Number.isFinite(best) && best > 0 ? best : null,
    };
  } catch {
    return emptyVoyageSave();
  }
}

/** Writes the save. Returns false where storage is refused or full. */
export function writeVoyageSave(save: VoyageSave): boolean {
  const store = storage();
  if (!store) return false;
  try {
    store.setItem(
      VOYAGE_SAVE_KEY,
      JSON.stringify({ gold: count(save.gold), voyages: count(save.voyages), bestTimeS: save.bestTimeS }),
    );
    return true;
  } catch {
    return false;
  }
}

/** True once anything worth continuing from has been kept: the title screen's "continue". */
export function hasVoyageSave(): boolean {
  const save = readVoyageSave();
  return save.voyages > 0 || save.gold > 0;
}

/** Forgets the save: a new game from the title screen. */
export function clearVoyageSave(): void {
  try {
    storage()?.removeItem(VOYAGE_SAVE_KEY);
  } catch {
    // Nothing kept is the state being asked for.
  }
}

/**
 * Adds a landed voyage to the save and writes it. Returns the save as it now
 * stands (also when the write was refused, so the tally can still show it) and
 * whether this voyage set a new best time.
 */
export function recordVoyage(result: VoyageResult): { save: VoyageSave; newBest: boolean } {
  const save = readVoyageSave();
  save.gold += count(result.gold);
  save.voyages += 1;
  let newBest = false;
  if (result.fullCargo && Number.isFinite(result.seconds) && result.seconds > 0) {
    if (save.bestTimeS === null || result.seconds < save.bestTimeS) {
      newBest = true;
      save.bestTimeS = Math.round(result.seconds * 10) / 10;
    }
  }
  writeVoyageSave(save);
  return { save, newBest };
}
