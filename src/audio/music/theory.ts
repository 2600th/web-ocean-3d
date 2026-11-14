/**
 * Just enough music theory for the score: note names, chords, voicings.
 *
 * Pitches are MIDI note numbers throughout (60 = middle C, 69 = A 440). Chords
 * are sets of pitch classes, which is all a voicing or a harmony line needs.
 */

const LETTER: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

export function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** `"F#5"`, `"Bb4"`, `"D3"` to a MIDI note. Throws on anything else: the score is static data. */
export function parseNote(name: string): number {
  const match = /^([A-G])([#b]?)(-?\d)$/.exec(name);
  if (match === null) throw new Error(`music: bad note "${name}"`);
  const accidental = match[2] === '#' ? 1 : match[2] === 'b' ? -1 : 0;
  return 12 * (Number(match[3]) + 1) + LETTER[match[1]] + accidental;
}

export interface Chord {
  readonly name: string;
  /** Pitch class of the root, 0..11. */
  readonly root: number;
  /** Pitch classes of every chord tone, root first. */
  readonly tones: readonly number[];
}

const QUALITY: Record<string, readonly number[]> = {
  '': [0, 4, 7],
  m: [0, 3, 7],
  '7': [0, 4, 7, 10],
  sus4: [0, 5, 7],
  m7: [0, 3, 7, 10],
};

/** `"D"`, `"Bm"`, `"Bb"`, `"A7"`, `"Gm"`. */
export function parseChord(name: string): Chord {
  const match = /^([A-G])([#b]?)(m7|m|7|sus4)?$/.exec(name);
  if (match === null) throw new Error(`music: bad chord "${name}"`);
  const accidental = match[2] === '#' ? 1 : match[2] === 'b' ? -1 : 0;
  const root = (LETTER[match[1]] + accidental + 12) % 12;
  const intervals = QUALITY[match[3] ?? ''];
  return { name, root, tones: intervals.map((i) => (root + i) % 12) };
}

function pitchClass(midi: number): number {
  return ((midi % 12) + 12) % 12;
}

/** The lowest MIDI note at or above `low` with pitch class `pc`. */
export function atOrAbove(pc: number, low: number): number {
  return low + ((pc - pitchClass(low) + 12) % 12);
}

/**
 * Every chord tone in `[low, high]`, ascending, written into `out`.
 *
 * Into a caller's array so the scheduler can reuse one per instrument instead of
 * allocating a voicing per bar.
 */
export function voicing(chord: Chord, low: number, high: number, out: number[]): number[] {
  out.length = 0;
  for (let m = low; m <= high; m++) {
    if (chord.tones.includes(pitchClass(m))) out.push(m);
  }
  return out;
}

/**
 * The nearest chord tone at least `minGap` semitones below `midi`.
 *
 * The harmony line under the tune. Taken from the chord rather than as a fixed
 * diatonic third, so it can never land on a note the bar's harmony rejects —
 * the tune has both C and C sharp in it, and a scale-based third would have to
 * guess which one the bar means.
 */
export function chordToneBelow(chord: Chord, midi: number, minGap = 3): number {
  for (let m = midi - minGap; m > midi - 13; m--) {
    if (chord.tones.includes(pitchClass(m))) return m;
  }
  return midi - 12;
}
