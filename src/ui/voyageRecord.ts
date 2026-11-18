/**
 * The saved voyage, as far as the title screen needs to know it.
 *
 * The voyage keeps its progress through `gameplay/voyageSave` (localStorage,
 * `web-ocean:voyage:v1`); that module owns the key and the shape and reads
 * defensively, so a missing, refused or malformed save is simply "no record".
 * The title screen asks two things of it: is there a record worth continuing,
 * and what are the gold and the voyages brought home.
 *
 * Pure: no three.js.
 */
import { clearVoyageSave, hasVoyageSave, readVoyageSave } from '../gameplay/voyageSave.ts';

export interface VoyageRecord {
  /** Gold landed across every voyage. */
  readonly gold: number;
  /** Voyages brought home. */
  readonly voyages: number;
}

/** The saved voyage, or `null` when there is none worth continuing. Never throws. */
export function readVoyageRecord(): VoyageRecord | null {
  if (!hasVoyageSave()) return null;
  const save = readVoyageSave();
  return { gold: save.gold, voyages: save.voyages };
}

/** Strikes the record: a new voyage chosen over an old one on the title screen. */
export function clearVoyageRecord(): void {
  clearVoyageSave();
}
