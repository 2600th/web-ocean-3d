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
 * and the on-screen button by name; a keyboard gets the key; a gamepad, once
 * it is used, gets its buttons.
 *
 * What has been learned is remembered in localStorage, so a returning visitor
 * is not re-taught. Storage can throw (private windows, locked-down embeds);
 * every access is wrapped and the coach simply forgets at the end of the
 * session instead.
 *
 * Pure logic and strings: no DOM, no three.js. `Hud` owns the rendering and
 * reports the actions.
 */

import { keybinds } from '../input/keybinds.ts';
import type { CameraMode } from './types.ts';

/**
 * `nogo` is never shown in the ribbon: it is the compass's no-go arc, which
 * fades once the player has sailed her out of the zone. It is kept here so it
 * is remembered with the rest.
 */
export type LessonId = 'look' | 'helm' | 'sail' | 'steer' | 'trim' | 'cargo' | 'anchor' | 'nogo';

/** The hands in use: lessons and key hints are phrased for them. */
export type InputKind = 'keyboard' | 'touch' | 'pad';

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
  /** Omitted: a pad player is shown the keyboard line (the mode keys, the mouse). */
  readonly pad?: Omit<Lesson, 'id'>;
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
    // Read when shown, not when loaded: the keys are the player's to rebind.
    get keyboard() {
      return { key: keybinds.labelOf('modeHelm'), text: 'Take the helm' };
    },
    touch: { text: 'Tap the wheel below to take the helm' },
  },
  {
    id: 'sail',
    modes: ['boat'],
    get keyboard() {
      return { key: keybinds.labelOf('sailMore'), text: 'Set sail' };
    },
    touch: { key: 'More', text: 'Set sail' },
    pad: { key: 'RT', text: 'Set sail' },
  },
  {
    id: 'steer',
    modes: ['boat'],
    get keyboard() {
      return { key: keybinds.labelsOf('steerPort', 'steerStarboard'), text: 'Steer her round' };
    },
    touch: { text: 'Push the stick left or right to steer' },
    pad: { key: 'Left stick', text: 'Steer her round' },
  },
  {
    id: 'trim',
    modes: ['boat'],
    get keyboard() {
      return {
        key: keybinds.labelsOf('trimIn', 'trimOut'),
        text: 'Trim the sails: in toward the wind, out away from it',
      };
    },
    touch: { text: 'Push the stick up to haul the sails in, down to ease them out' },
    pad: { key: 'LB RB', text: 'Trim the sails: in toward the wind, out away from it' },
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
    get keyboard() {
      return { key: keybinds.labelOf('anchor'), text: 'Drop anchor to stop' };
    },
    touch: { key: 'Anchor', text: 'Drop anchor to stop' },
    pad: { key: 'A', text: 'Drop anchor to stop' },
  },
  {
    id: 'nogo',
    // Never in the ribbon: the compass's red arc is this lesson.
    modes: [],
    keyboard: { text: 'Keep her bow out of the red arc' },
    touch: { text: 'Keep her bow out of the red arc' },
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
  private input: InputKind;

  constructor(input: InputKind) {
    this.input = input;
    this.learned = readLearned();
  }

  /** Rephrases the lessons for other hands. Returns true when that changed anything. */
  setInput(input: InputKind): boolean {
    if (input === this.input) return false;
    this.input = input;
    return true;
  }

  get inputKind(): InputKind {
    return this.input;
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
      const phrase =
        this.input === 'touch' ? spec.touch
          : this.input === 'pad' ? (spec.pad ?? spec.keyboard)
            : spec.keyboard;
      return { id: spec.id, ...phrase };
    }
    return null;
  }
}
