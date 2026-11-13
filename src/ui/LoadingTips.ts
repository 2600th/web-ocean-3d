/**
 * Lines for the loading chart: a tip or a piece of the world, one at a time,
 * under the status line while the sea is being built.
 *
 * The status line above them stays the truthful part — what is being built,
 * and the course on the chart is how far along it is. These are for the wait:
 * each one is something true about the sea, the ship or the controls, so a
 * visitor who reads three of them arrives knowing a little more than they did.
 *
 * Phrased for the hands in use, like the coach: a phone has no number keys.
 * The order is shuffled once per visit so a reload does not replay the same
 * opening line, and the first is always about the helm, the most useful thing
 * to know before arriving.
 */

interface Tip {
  readonly keyboard: string;
  readonly touch?: string;
}

const FIRST: Tip = {
  keyboard: 'Press 3 to take the helm. W sets sail, A and D steer, R drops anchor.',
  touch: 'Tap the wheel to take the helm, then More to set sail.',
};

const TIPS: readonly Tip[] = [
  { keyboard: 'No ship sails straight into the wind. If she stalls, turn either way and the canvas fills.' },
  { keyboard: 'She is fastest with the wind across her side, not behind her.' },
  { keyboard: 'A column of gold light marks drifting cargo. Come alongside to haul it in.' },
  { keyboard: 'Bright turquoise water is shallow water. Keep her in the deep blue.' },
  { keyboard: 'The anchor takes a few seconds to come up. Plan your stops.' },
  {
    keyboard: 'Press 4 and she sails herself, through a whole day and a squall.',
    touch: 'Tap the spyglass and she sails herself, through a whole day and a squall.',
  },
  { keyboard: 'Take any camera below the surface to see the sea from underneath.' },
  { keyboard: 'Gulls ride the yards until she gets under way.' },
  { keyboard: 'Whitecaps break only where a wave really folds over. Nothing here is painted on.' },
  { keyboard: 'Rain soaks the deck and the canvas, and they stay dark a while after it passes.' },
  {
    keyboard: 'The ledger, top right, changes the weather, the hour and the sea.',
    touch: 'The ledger button changes the weather, the hour and the sea.',
  },
  { keyboard: 'Every wave is worked out live, every frame. Nothing here is a video.' },
];

/** How long each line holds, ms. Long enough to read twice. */
const HOLD_MS = 5200;
/** Matches the CSS fade on `.boot__tip`. */
const FADE_MS = 360;

function shuffled<T>(items: readonly T[]): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * Starts rotating lines in `#boot-tip`, if the page has one. Stops by itself
 * once the boot layer is hidden or the status reports a failure.
 */
export function startLoadingTips(root: HTMLElement): void {
  const el = root.querySelector<HTMLElement>('#boot-tip');
  if (!el) return;
  const touch = typeof window.matchMedia === 'function'
    && window.matchMedia('(pointer: coarse)').matches;
  const reduced = typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const lines = [FIRST, ...shuffled(TIPS)].map((tip) => (touch ? tip.touch ?? tip.keyboard : tip.keyboard));

  let index = 0;
  el.textContent = lines[0] ?? '';
  el.hidden = false;

  // Chained timeouts, not an interval. Boot blocks the main thread for seconds
  // at a time while shaders compile, and an interval that missed its ticks
  // fires them back to back afterwards — a line would flash past unread. Each
  // line is held for its full time from when it actually went up.
  const next = (): void => {
    const failed = root.querySelector('.boot__status--error') !== null;
    if (failed || root.classList.contains('boot--hidden')) {
      if (failed) el.hidden = true;
      return;
    }
    index = (index + 1) % lines.length;
    const line = lines[index] ?? '';
    if (reduced) {
      el.textContent = line;
      window.setTimeout(next, HOLD_MS);
      return;
    }
    el.classList.add('is-changing');
    window.setTimeout(() => {
      el.textContent = line;
      el.classList.remove('is-changing');
      window.setTimeout(next, HOLD_MS);
    }, FADE_MS);
  };
  window.setTimeout(next, HOLD_MS);
}
