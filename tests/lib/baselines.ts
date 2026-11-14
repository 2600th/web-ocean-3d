import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePng, encodePng, type RgbaImage } from './png';
import type { StackFingerprint } from './stack';
import type { Shot } from './shots';

/**
 * Where baselines live and what travels with them.
 *
 * The PNG alone is not enough to know whether a comparison is meaningful, so
 * each one has a sidecar recording the stack it came off and the shot definition
 * it was taken under. Both can drift independently of the renderer — someone
 * moves a camera in `shots.ts`, someone else runs the suite on a different GPU —
 * and both produce a large, entirely legitimate diff that must not be reported
 * as a regression without saying why.
 */

const HERE = dirname(fileURLToPath(import.meta.url));

/** `tests/baselines`. Checked in. */
export const BASELINE_DIR = join(HERE, '..', 'baselines');

/** `test-results/visual`. Gitignored; holds actual/baseline/diff artefacts. */
export const ARTIFACT_DIR = join(HERE, '..', '..', 'test-results', 'visual');

export interface BaselineMeta {
  shot: string;
  capturedAt: string;
  width: number;
  height: number;
  /** Snapshot of the shot definition, so a moved camera is distinguishable
   *  from a changed renderer. */
  definition: Pick<Shot, 'state' | 'camera' | 'lens' | 'time' | 'settleSteps' | 'shipInput'>;
  stack: StackFingerprint;
}

export interface Baseline {
  image: RgbaImage;
  meta: BaselineMeta;
}

export function baselinePath(shotId: string): string {
  return join(BASELINE_DIR, `${shotId}.png`);
}

export function baselineMetaPath(shotId: string): string {
  return join(BASELINE_DIR, `${shotId}.json`);
}

export function hasBaseline(shotId: string): boolean {
  return existsSync(baselinePath(shotId)) && existsSync(baselineMetaPath(shotId));
}

export function readBaseline(shotId: string): Baseline {
  return {
    image: decodePng(readFileSync(baselinePath(shotId))),
    meta: JSON.parse(readFileSync(baselineMetaPath(shotId), 'utf8')) as BaselineMeta,
  };
}

export function writeBaseline(shot: Shot, image: RgbaImage, stack: StackFingerprint): void {
  mkdirSync(BASELINE_DIR, { recursive: true });
  const meta: BaselineMeta = {
    shot: shot.id,
    capturedAt: new Date().toISOString(),
    width: image.width,
    height: image.height,
    definition: {
      state: shot.state,
      camera: shot.camera,
      lens: shot.lens,
      time: shot.time,
      settleSteps: shot.settleSteps,
      shipInput: shot.shipInput,
    },
    stack,
  };
  writeFileSync(baselinePath(shot.id), encodePng(image));
  writeFileSync(baselineMetaPath(shot.id), `${JSON.stringify(meta, null, 2)}\n`);
}

/** Writes a full-resolution artefact and returns its path, for attachment. */
export function writeArtifact(shotId: string, kind: string, image: RgbaImage): string {
  mkdirSync(ARTIFACT_DIR, { recursive: true });
  const path = join(ARTIFACT_DIR, `${shotId}-${kind}.png`);
  writeFileSync(path, encodePng(image));
  return path;
}

export function writeReport(name: string, body: unknown): string {
  mkdirSync(ARTIFACT_DIR, { recursive: true });
  const path = join(ARTIFACT_DIR, name);
  writeFileSync(path, `${JSON.stringify(body, null, 2)}\n`);
  return path;
}

/** Reports a definition change as a list of human-readable differences. */
export function definitionDifferences(shot: Shot, meta: BaselineMeta): string[] {
  const recorded = JSON.stringify(meta.definition);
  const current = JSON.stringify({
    state: shot.state,
    camera: shot.camera,
    lens: shot.lens,
    time: shot.time,
    settleSteps: shot.settleSteps,
    shipInput: shot.shipInput,
  });
  if (recorded === current) return [];
  return [`shot definition changed since the baseline was captured:\n  was ${recorded}\n  now ${current}`];
}
