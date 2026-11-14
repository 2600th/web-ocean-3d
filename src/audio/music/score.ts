/**
 * The score: two original tunes and the order their phrases are played in.
 *
 * Everything here was written for this project. It is in the idiom of a sea
 * song — 6/8, a modal major with a flattened seventh that comes and goes, a
 * minor tune for weather — but it quotes no existing song, shanty or
 * traditional air. If a phrase here ever starts to sound like something with a
 * name, change it: that is a licensing problem, not a taste one.
 *
 * - **"The Lantern Reach"** (`voyage`), in D. Section `I` is a slow air, `A` the
 *   tune proper, `B` its answer, which climbs to the high B and leans on the C
 *   natural for the mixolydian colour.
 * - **"Glass Falling"** (`squall`), in D minor, for weather. `S` sits low and
 *   ends on the dominant so it never resolves; `T` climbs a sixth higher.
 *
 * **Time.** One step is a quaver; a bar of 6/8 is six of them, felt as two
 * dotted-crotchet beats (steps 0 and 3). Every bar is checked to sum to six at
 * load, so a typo is an exception on the first frame rather than a tune that
 * drifts off the barline.
 *
 * **Notation.** A bar is `chord` plus `"note:len note:len …"`, where a note is
 * a name (`F#5`, `Bb4`) or `r` for a rest and `len` is in quavers.
 */

import { parseChord, parseNote, type Chord } from './theory';

export const STEPS_PER_BAR = 6;
export const BARS_PER_PHRASE = 8;

export interface MelodyNote {
  /** Quaver within the bar, 0..5. */
  readonly step: number;
  readonly midi: number;
  /** Length in quavers. */
  readonly len: number;
}

export interface Bar {
  readonly chord: Chord;
  readonly notes: readonly MelodyNote[];
}

export type SectionId = 'I' | 'A' | 'B' | 'S' | 'T';

/** One phrase of an arrangement: which section, and what the melody instruments do with it. */
export interface Phrase {
  readonly section: SectionId;
  /** The tune is played. False is a breathing phrase: harmony and rhythm only. */
  readonly tune: boolean;
  /** A second instrument adds a harmony line under the tune. */
  readonly harmony: boolean;
}

export type SongId = 'voyage' | 'squall';

export interface Song {
  readonly id: SongId;
  readonly sections: Readonly<Record<SectionId, readonly Bar[] | undefined>>;
}

function bar(chord: string, notes: string): Bar {
  const parsed: MelodyNote[] = [];
  let step = 0;
  for (const token of notes.trim().split(/\s+/)) {
    const [name, lenText] = token.split(':');
    const len = Number(lenText);
    if (!Number.isInteger(len) || len < 1) throw new Error(`music: bad length in "${token}"`);
    if (name !== 'r') parsed.push({ step, midi: parseNote(name), len });
    step += len;
  }
  if (step !== STEPS_PER_BAR) {
    throw new Error(`music: bar "${notes}" is ${step} quavers, not ${STEPS_PER_BAR}`);
  }
  return { chord: parseChord(chord), notes: parsed };
}

const VOYAGE_I: readonly Bar[] = [
  bar('D', 'F#4:3 A4:3'),
  bar('G', 'B4:4 A4:2'),
  bar('D', 'F#4:3 E4:2 D4:1'),
  bar('A', 'E4:6'),
  bar('D', 'F#4:3 A4:2 D5:1'),
  bar('Bm', 'D5:3 C#5:2 B4:1'),
  bar('G', 'A4:2 B4:1 G4:2 E4:1'),
  bar('A', 'A4:6'),
];

const VOYAGE_A: readonly Bar[] = [
  bar('D', 'A4:2 D5:1 F#5:2 E5:1'),
  bar('D', 'D5:2 B4:1 A4:3'),
  bar('G', 'G4:2 B4:1 D5:2 C5:1'),
  bar('A', 'B4:1 A4:1 G4:1 A4:3'),
  bar('D', 'A4:2 D5:1 F#5:2 A5:1'),
  bar('G', 'G5:2 F#5:1 E5:2 D5:1'),
  bar('A', 'E5:1 F#5:1 E5:1 C#5:2 A4:1'),
  bar('D', 'D5:6'),
];

const VOYAGE_B: readonly Bar[] = [
  bar('D', 'F#5:2 A5:1 A5:2 F#5:1'),
  bar('C', 'G5:2 E5:1 C5:3'),
  bar('D', 'D5:2 F#5:1 E5:2 D5:1'),
  bar('G', 'B4:2 G4:1 A4:3'),
  bar('Bm', 'F#5:2 A5:1 B5:2 A5:1'),
  bar('Em', 'G5:1 F#5:1 E5:1 D5:2 B4:1'),
  bar('A', 'A4:2 F#4:1 G4:1 A4:1 C#5:1'),
  bar('D', 'D5:3 r:3'),
];

const SQUALL_S: readonly Bar[] = [
  bar('Dm', 'D5:3 C5:2 A4:1'),
  bar('Bb', 'Bb4:4 A4:2'),
  bar('Gm', 'G4:3 A4:2 Bb4:1'),
  bar('A', 'A4:6'),
  bar('Dm', 'D5:3 E5:2 F5:1'),
  bar('A', 'E5:3 D5:2 C#5:1'),
  bar('Gm', 'D5:2 A4:1 Bb4:2 G4:1'),
  bar('A', 'A4:6'),
];

const SQUALL_T: readonly Bar[] = [
  bar('Dm', 'F5:3 E5:2 D5:1'),
  bar('A', 'C#5:4 A4:2'),
  bar('Bb', 'D5:2 E5:1 F5:2 G5:1'),
  bar('Dm', 'A5:6'),
  bar('Gm', 'Bb5:3 A5:2 G5:1'),
  bar('Dm', 'F5:3 E5:2 D5:1'),
  bar('A', 'E5:2 C#5:1 D5:2 E5:1'),
  bar('A', 'A4:6'),
];

export const SONGS: Readonly<Record<SongId, Song>> = {
  voyage: {
    id: 'voyage',
    sections: { I: VOYAGE_I, A: VOYAGE_A, B: VOYAGE_B, S: undefined, T: undefined },
  },
  squall: {
    id: 'squall',
    sections: { I: undefined, A: undefined, B: undefined, S: SQUALL_S, T: SQUALL_T },
  },
};

function phrase(section: SectionId, tune = true, harmony = false): Phrase {
  return { section, tune, harmony };
}

/**
 * Phrase orders, per arrangement.
 *
 * Every order has at least one phrase with the tune off. A melody that never
 * stops is the fastest way to make game music something people turn off; the
 * breathing phrases are what let it run for an hour.
 */
export const ORDERS = {
  calm: [phrase('I'), phrase('A'), phrase('I', false), phrase('B'), phrase('A', false)],
  sailing: [
    phrase('A'),
    phrase('A', true, true),
    phrase('B'),
    phrase('B', true, true),
    phrase('A', false),
  ],
  tour: [phrase('I'), phrase('A', true, true), phrase('B', true, true), phrase('A')],
  storm: [phrase('S'), phrase('S', true, true), phrase('T'), phrase('S', false)],
  underwater: [phrase('I'), phrase('I', false), phrase('A'), phrase('I', false)],
} as const satisfies Record<string, readonly Phrase[]>;

export type OrderId = keyof typeof ORDERS;
