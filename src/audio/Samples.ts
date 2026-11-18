/**
 * Recordings, where they beat synthesis (roadmap decision 7).
 *
 * Published listening tests still tell synthesised thunder from a recording,
 * and a gull's cry and a wave breaking on a beach are the same kind of sound:
 * a gesture with a shape no filter on noise reproduces. So these three are
 * recorded, and each is *layered* with the synthesis it replaces rather than
 * swapped for it, so the sound still follows the sea state: the recording
 * supplies the texture, the synthesised bed underneath still moves with the
 * wind, the swell and the distance, and the level and filter on the recording
 * are driven by the same numbers.
 *
 * All are CC0, from OpenGameArt; URLs, authors and licences are in
 * `ASSET_LICENSES.md`. Mono Opus at 40-48 kb/s, about 165 KB together.
 *
 * Loading is lazy and can fail: until a recording has decoded, or if it never
 * does (an old Safari without Opus, a blocked request), the synthesis plays
 * alone, exactly as before. Every buffer is peak-normalised on decode, so the
 * levels in `AudioSystem` and `Island` do not depend on how a file was
 * mastered.
 */

export type SampleKind = 'thunder' | 'gull' | 'surf';

const FILES: Readonly<Record<SampleKind, readonly string[]>> = {
  thunder: ['thunder-crack.ogg'],
  gull: ['gull-1.ogg', 'gull-2.ogg', 'gull-3.ogg', 'gull-4.ogg', 'gull-5.ogg', 'gull-6.ogg', 'gull-7.ogg'],
  surf: ['surf-1.ogg', 'surf-2.ogg', 'surf-3.ogg', 'surf-4.ogg'],
};

/** Where the recordings are served from, by default: `public/audio/`. */
export const DEFAULT_SAMPLES_URL = '/audio/';

/** Peak every decoded buffer is scaled to. */
const PEAK = 0.9;

export class Samples {
  private readonly ctx: BaseAudioContext;
  private readonly base: string;
  private readonly buffers: Record<SampleKind, AudioBuffer[]> = { thunder: [], gull: [], surf: [] };
  private loading: Promise<void> | null = null;

  constructor(ctx: BaseAudioContext, base: string) {
    this.ctx = ctx;
    this.base = base.endsWith('/') ? base : `${base}/`;
  }

  /** Starts loading, once; resolves when every file has decoded or failed. Never rejects. */
  load(): Promise<void> {
    if (this.loading !== null) return this.loading;
    if (typeof fetch !== 'function') {
      this.loading = Promise.resolve();
      return this.loading;
    }
    const jobs: Promise<void>[] = [];
    for (const kind of Object.keys(FILES) as SampleKind[]) {
      FILES[kind].forEach((file, index) => {
        jobs.push(
          this.fetchOne(`${this.base}${file}`).then((buffer) => {
            // Kept in file order whatever order they arrive in, so a seeded
            // choice picks the same file every run.
            if (buffer !== null) this.buffers[kind][index] = buffer;
          }),
        );
      });
    }
    this.loading = Promise.all(jobs).then(() => {
      for (const kind of Object.keys(this.buffers) as SampleKind[]) {
        this.buffers[kind] = this.buffers[kind].filter((b) => b !== undefined);
      }
    });
    return this.loading;
  }

  /** How many of a kind have decoded. */
  count(kind: SampleKind): number {
    return this.buffers[kind].length;
  }

  /** One of a kind, chosen by `u` in 0..1, or null while none has decoded. */
  pick(kind: SampleKind, u: number): AudioBuffer | null {
    const list = this.buffers[kind];
    if (list.length === 0) return null;
    return list[Math.min(list.length - 1, Math.floor(u * list.length))] ?? null;
  }

  private async fetchOne(url: string): Promise<AudioBuffer | null> {
    try {
      const response = await fetch(url);
      if (!response.ok) return null;
      const data = await response.arrayBuffer();
      const buffer = await this.ctx.decodeAudioData(data);
      normalise(buffer);
      return buffer;
    } catch {
      return null;
    }
  }
}

function normalise(buffer: AudioBuffer): void {
  let peak = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]);
      if (a > peak) peak = a;
    }
  }
  if (peak < 1e-6) return;
  const scale = PEAK / peak;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) data[i] *= scale;
  }
}
