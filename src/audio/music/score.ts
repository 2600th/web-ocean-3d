/**
 * The score: two original tunes and the order their phrases are played in.
 *
 * Everything here was written for this project. It is in the idiom of the sea
 * — a minor 6/8 that drives, a hook that leaps, a fiddle calling and a
 * squeezebox answering — but it quotes no existing song, shanty, film or game
 * theme, and no progression with the rhythm that makes one recognisable. If a
 * phrase here ever starts to sound like something with a name, change it: that
 * is a licensing problem, not a taste one.
 *
 * - **"Ballast and Brass"** (`voyage`), in D minor. The hook is two bars: a
 *   leap up a fifth to a held note, then a dotted fall (`D5:1 A5:2 G5:1.5
 *   F5:.5 E5:1`), answered a step lower. `A` is the tune proper over
 *   i–bVII–bVI–V: the fiddle's call, the squeezebox's answer (the call's leap
 *   widened to a sixth), the call again reaching up through the flat sixth, and
 *   the answer taken up an octave for the climax on the high D. `B` lifts into
 *   F major with the hook a third higher and comes home through a half close on
 *   A. `I` is the calm air: the same leap, slower and Dorian, with the B natural
 *   of the major IV.
 * - **"Close-Reefed"** (`squall`), in D minor, for weather. `S` is the hook's
 *   leap made urgent — a snapped repeated note before it — and ends on the
 *   dominant so it never settles; `T` climbs over i–bVI–bIII–bVII.
 *
 * **Time.** One step is a quaver; a bar of 6/8 is six of them, felt as two
 * dotted-crotchet beats (steps 0 and 3). A note may start on a semiquaver, so
 * a step can be fractional and the engine places it inside the quaver. Every
 * bar is checked to sum to six at load, so a typo is an exception on the first
 * frame rather than a tune that drifts off the barline.
 *
 * **Notation.** A bar is a chord, `"note:len note:len …"` and who plays it,
 * where a note is a name (`F#5`, `Bb4`) or `r` for a rest and `len` is in
 * quavers, in halves (`1.5`, `.5`). Who plays it is the call (the lead), the
 * answer (the second melody instrument) or both, the second in harmony.
 */

import { parseChord, parseNote, type Chord } from './theory';

export const STEPS_PER_BAR = 6;
export const BARS_PER_PHRASE = 8;

export interface MelodyNote {
  /** Quaver within the bar, 0..5.5; a fractional step starts on the semiquaver. */
  readonly step: number;
  readonly midi: number;
  /** Length in quavers. */
  readonly len: number;
}

/** Who has the tune for a bar: the lead, the answering instrument, or both. */
export type BarRole = 'call' | 'answer' | 'tutti';

export interface Bar {
  readonly chord: Chord;
  readonly notes: readonly MelodyNote[];
  readonly role: BarRole;
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
  /** Pitch classes of the key the harmony line moves in. */
  readonly scale: readonly number[];
  readonly sections: Readonly<Record<SectionId, readonly Bar[] | undefined>>;
}

function bar(chord: string, notes: string, role: BarRole = 'call'): Bar {
  const parsed: MelodyNote[] = [];
  let step = 0;
  for (const token of notes.trim().split(/\s+/)) {
    const [name, lenText] = token.split(':');
    const len = Number(lenText);
    if (!(len > 0) || !Number.isInteger(len * 2)) throw new Error(`music: bad length in "${token}"`);
    if (name !== 'r') parsed.push({ step, midi: parseNote(name), len });
    step += len;
  }
  if (step !== STEPS_PER_BAR) {
    throw new Error(`music: bar "${notes}" is ${step} quavers, not ${STEPS_PER_BAR}`);
  }
  return { chord: parseChord(chord), notes: parsed, role };
}

const ANSWER = 'answer';
const TUTTI = 'tutti';

/** D natural minor. The bars' own chords bring in the B natural and the C sharp. */
const D_MINOR = [2, 4, 5, 7, 9, 10, 0];

const VOYAGE_I: readonly Bar[] = [
  bar('Dm', 'D5:1 A5:2 F5:3'),
  bar('G', 'G5:1 B5:2 A5:1.5 G5:.5 E5:1'),
  bar('Dm', 'F5:2 E5:1 D5:2 A4:1', ANSWER),
  bar('C', 'G4:1 C5:2 E5:3', ANSWER),
  bar('Dm', 'D5:1 A5:2 C6:3'),
  bar('G', 'B5:1.5 A5:.5 G5:1 D5:3'),
  bar('C', 'E5:1 G5:1 E5:1 C5:1.5 B4:.5 C5:1', TUTTI),
  bar('Dm', 'D5:6', TUTTI),
];

const VOYAGE_A: readonly Bar[] = [
  bar('Dm', 'D5:1 A5:2 G5:1.5 F5:.5 E5:1'),
  bar('C', 'C5:1 G5:2 F5:1.5 E5:.5 D5:1'),
  bar('Bb', 'F4:1 D5:2 C5:1.5 Bb4:.5 A4:1', ANSWER),
  bar('A', 'E4:1 C#5:2 E5:3', ANSWER),
  bar('Dm', 'D5:1 A5:2 Bb5:1.5 A5:.5 G5:1'),
  bar('Bb', 'F5:1 D6:2 C6:1.5 Bb5:.5 A5:1'),
  bar('A', 'G5:1 E5:1 C#5:1 A4:1.5 B4:.5 C#5:1', TUTTI),
  bar('Dm', 'D5:3 r:3', TUTTI),
];

const VOYAGE_B: readonly Bar[] = [
  bar('F', 'F5:1 C6:2 Bb5:1.5 A5:.5 G5:1'),
  bar('C', 'G5:1 E5:1 G5:1 C6:3'),
  bar('Bb', 'D5:1 F5:2 D5:1.5 C5:.5 Bb4:1', ANSWER),
  bar('F', 'C5:1 F5:2 A5:3', ANSWER),
  bar('Dm', 'D6:2 C6:1 A5:2 F5:1'),
  bar('Gm', 'G5:1 Bb5:2 D6:1.5 C6:.5 Bb5:1'),
  bar('Bb', 'D5:1 F5:1 Bb5:1 A5:1.5 G5:.5 F5:1', TUTTI),
  bar('A', 'E5:2 C#5:1 A4:3', TUTTI),
];

const SQUALL_S: readonly Bar[] = [
  bar('Dm', 'D5:1.5 D5:.5 A5:1 F5:1.5 E5:.5 D5:1'),
  bar('Bb', 'Bb4:1.5 Bb4:.5 F5:1 D5:3'),
  bar('Gm', 'G4:1.5 G4:.5 D5:1 Bb4:1.5 A4:.5 G4:1', ANSWER),
  bar('A', 'C#5:1 E5:1 A5:1 G5:1.5 F5:.5 E5:1', ANSWER),
  bar('Dm', 'D5:1.5 D5:.5 A5:1 D6:3'),
  bar('Bb', 'C6:1.5 Bb5:.5 A5:1 F5:1.5 E5:.5 D5:1'),
  bar('Gm', 'G5:1 Bb5:1 G5:1 E5:1.5 F5:.5 G5:1', TUTTI),
  bar('A', 'A5:3 A4:3', TUTTI),
];

const SQUALL_T: readonly Bar[] = [
  bar('Dm', 'F5:1 A5:1 D6:1 C6:1.5 A5:.5 F5:1'),
  bar('Bb', 'D5:1 F5:1 Bb5:1 A5:3'),
  bar('F', 'C5:1 F5:1 A5:1 G5:1.5 F5:.5 E5:1', ANSWER),
  bar('C', 'E5:1.5 G5:.5 C6:1 G5:3', ANSWER),
  bar('Dm', 'A5:1.5 A5:.5 D6:1 A5:1.5 F5:.5 D5:1'),
  bar('Bb', 'Bb5:1.5 Bb5:.5 D6:1 Bb5:1.5 F5:.5 D5:1'),
  bar('Gm', 'G5:1 D5:1 Bb5:1 A5:1.5 G5:.5 F5:1', TUTTI),
  bar('A', 'E5:1 C#5:1 A4:1 E5:3', TUTTI),
];

export const SONGS: Readonly<Record<SongId, Song>> = {
  voyage: {
    id: 'voyage',
    scale: D_MINOR,
    sections: { I: VOYAGE_I, A: VOYAGE_A, B: VOYAGE_B, S: undefined, T: undefined },
  },
  squall: {
    id: 'squall',
    scale: D_MINOR,
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
 * breathing phrases are what let it run for an hour. The harmony comes in on
 * a section's second time round, so the repeat is never just a repeat.
 */
export const ORDERS = {
  calm: [phrase('I'), phrase('A'), phrase('I', false), phrase('B'), phrase('I', true, true), phrase('A', false)],
  sailing: [
    phrase('A'),
    phrase('A', true, true),
    phrase('B'),
    phrase('B', true, true),
    phrase('A', false),
  ],
  tour: [phrase('I'), phrase('A', true, true), phrase('B', true, true), phrase('A')],
  storm: [phrase('S'), phrase('S', true, true), phrase('T'), phrase('T', true, true), phrase('S', false)],
  underwater: [phrase('I'), phrase('I', false), phrase('A'), phrase('I', false)],
} as const satisfies Record<string, readonly Phrase[]>;

export type OrderId = keyof typeof ORDERS;

/** The chord of the bar after `barIndex` in this order, for a bass line walking into it. */
export function nextChord(
  song: Song,
  order: OrderId,
  phraseIndex: number,
  barIndex: number,
): Chord | undefined {
  const phrases = ORDERS[order];
  let p = phraseIndex % phrases.length;
  let b = barIndex + 1;
  if (b >= BARS_PER_PHRASE) {
    b = 0;
    p = (p + 1) % phrases.length;
  }
  return song.sections[phrases[p].section]?.[b]?.chord;
}
