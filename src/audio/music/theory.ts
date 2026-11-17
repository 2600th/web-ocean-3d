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
 * The harmony line's note under `midi`: a third below, or a sixth below where
 * the third would leave the chord. That is the folk second part — parallel
 * thirds, turning to sixths on the root — and it is why the line sounds sung
 * rather than computed.
 *
 * A chord tone wins at either distance. A melody note outside the chord (a
 * passing note, an appoggiatura) takes the third that lies in `scale`, so the
 * pair moves together; the chord's own tones override the scale, because the
 * tunes have both C and C sharp in them and only the bar knows which it means.
 */
export function harmonyBelow(chord: Chord, scale: readonly number[], midi: number): number {
  for (const m of [midi - 3, midi - 4, midi - 8, midi - 9]) {
    if (chord.tones.includes(pitchClass(m))) return m;
  }
  return inKey(chord, scale, midi - 3) ? midi - 3 : midi - 4;
}

/**
 * `midi` belongs over this chord in this key: a chord tone, or a scale note
 * the chord has not altered. A scale note a semitone from a chord tone the
 * scale lacks is the one the chord replaced — C under an A major chord.
 */
export function inKey(chord: Chord, scale: readonly number[], midi: number): boolean {
  const pc = pitchClass(midi);
  if (chord.tones.includes(pc)) return true;
  for (const tone of chord.tones) {
    const gap = (tone - pc + 12) % 12;
    if (!scale.includes(tone) && (gap === 1 || gap === 11)) return false;
  }
  return scale.includes(pc);
}
