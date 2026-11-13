/**
 * First-minute lessons: what a new player needs, in the order they need it,
 * taught by doing rather than by a page of controls.
 *
 *   look   → take the helm → set sail → steer → find the cargo → anchor
 *
 * One lesson shows at a time, in the hint ribbon over the mode tabs (a slot
 * the HUD already has), and it stays until the player has *done* the thing —
 * not until a timer runs out. Doing a later thing first counts: someone who
 * presses 3 before dragging the camera is never told to press 3.
 *
 * Lessons are phrased for the hands in use. A coarse pointer gets the gesture
 * and the on-screen button by name; a keyboard gets the key.
 *
 * What has been learned is remembered in localStorage, so a returning visitor
 * is not re-taught. Storage can throw (private windows, locked-down embeds);
 * every access is wrapped and the coach simply forgets at the end of the
 * session instead.
 *
 * Pure logic and strings: no DOM, no three.js. `Hud` owns the rendering and
 * reports the actions.
 */

import type { CameraMode } from './types.ts';

export type LessonId = 'look' | 'helm' | 'sail' | 'steer' | 'cargo' | 'anchor';

export interface Lesson {
  readonly id: LessonId;
  /** The key or gesture, shown on a keycap. Omitted: the line stands alone. */
  readonly key?: string;
  readonly text: string;
}

interface LessonSpec {
  readonly id: LessonId;
  /** Modes in which the lesson makes sense to show. */
  readonly modes: readonly CameraMode[];
  readonly keyboard: Omit<Lesson, 'id'>;
  readonly touch: Omit<Lesson, 'id'>;
}

const LESSONS: readonly LessonSpec[] = [
  {
    id: 'look',
    // Orbit only: Fly looks with the mouse once it is captured, not by dragging.
    modes: ['orbit'],
    keyboard: { key: 'Drag', text: 'Look around' },
    touch: { key: 'Drag', text: 'Look around' },
  },
  {
    id: 'helm',
    modes: ['orbit', 'fly'],
    keyboard: { key: '3', text: 'Take the helm' },
    touch: { text: 'Tap the wheel below to take the helm' },
  },
  {
    id: 'sail',
    modes: ['boat'],
    keyboard: { key: 'W', text: 'Set sail' },
    touch: { key: 'More', text: 'Set sail' },
  },
  {
    id: 'steer',
    modes: ['boat'],
    keyboard: { key: 'A D', text: 'Steer her round' },
    touch: { text: 'Push the stick left or right to steer' },
  },
  {
    id: 'cargo',
    modes: ['boat'],
    keyboard: { text: 'Steer for the gold mark and come alongside the cargo' },
    touch: { text: 'Steer for the gold mark and come alongside the cargo' },
  },
  {
    id: 'anchor',
    modes: ['boat'],
    keyboard: { key: 'R', text: 'Drop anchor to stop' },
    touch: { key: 'Anchor', text: 'Drop anchor to stop' },
  },
];

const STORAGE_KEY = 'web-ocean:learned:v1';

function readLearned(): Set<LessonId> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    const known = new Set<string>(LESSONS.map((lesson) => lesson.id));
    return new Set(parsed.filter((id): id is LessonId => typeof id === 'string' && known.has(id)));
  } catch {
    return new Set();
  }
}

function writeLearned(learned: ReadonlySet<LessonId>): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify([...learned]));
  } catch {
    // Nowhere to keep it: the lessons are remembered for this session only.
  }
}

export class Coach {
  private readonly learned: Set<LessonId>;
  private readonly touch: boolean;

  constructor(touch: boolean) {
    this.touch = touch;
    this.learned = readLearned();
  }

  /** True when every lesson has been learned, now or on an earlier visit. */
  get finished(): boolean {
    return LESSONS.every((lesson) => this.learned.has(lesson.id));
  }

  has(id: LessonId): boolean {
    return this.learned.has(id);
  }

  /**
   * Marks a lesson learned. Returns true when that is news, so the caller
   * knows to redraw. Idempotent and cheap: safe to call from every input.
   */
  learn(id: LessonId): boolean {
    if (this.learned.has(id)) return false;
    this.learned.add(id);
    writeLearned(this.learned);
    return true;
  }

  /**
   * The lesson to show now: the first one not yet learned, provided it makes
   * sense in this mode. Earlier lessons that belong to another mode do not
   * hold back the ones that belong here — at the helm, "look around" is not
   * what a new sailor needs next — but within a mode the order is kept.
   */
  current(mode: CameraMode): Lesson | null {
    for (const spec of LESSONS) {
      if (this.learned.has(spec.id)) continue;
      if (!spec.modes.includes(mode)) continue;
      return { id: spec.id, ...(this.touch ? spec.touch : spec.keyboard) };
    }
    return null;
  }
}
