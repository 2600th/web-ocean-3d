/**
 * The reduced-motion setting, as the player's choice on top of the system's.
 *
 * Fifteen places in the game ask `matchMedia('(prefers-reduced-motion:
 * reduce)')`: the camera, the sails, the lens rain, the post chain, the HUD,
 * the menu sea. Giving each a second source would mean editing all of them and
 * every one written after. So the question is answered in one place instead:
 * `installReducedMotionOverride` wraps `window.matchMedia` so that this one
 * query reports a match when the system says so *or* the player has switched
 * "Reduce motion" on, and tells its listeners when the player flips it. Every
 * other query goes to the browser untouched.
 *
 * The wrapper hands back the browser's own list for every other query, and for
 * this one a thin stand-in with the same `matches`, `media`, `change` events
 * and the legacy `addListener`. Nothing else in `MediaQueryList` is used for
 * this query anywhere in the game. The stylesheet cannot ask JavaScript, so it
 * reads `html[data-reduce-motion]` for the same thing (see `prefs.ts`).
 *
 * Installed once, from `main.ts`, before anything asks.
 */

const QUERY = /prefers-reduced-motion\s*:\s*reduce/i;

let override = false;
let installed = false;
const instances = new Set<{ fire(): void }>();

/** True when the player has chosen reduced motion, whatever the system says. */
export function reducedMotionOverride(): boolean {
  return override;
}

/** The player's choice. Listeners on every live query hear it if the answer changed. */
export function setReducedMotionOverride(on: boolean): void {
  if (override === on) return;
  const before = effective();
  override = on;
  if (effective() !== before) for (const instance of [...instances]) instance.fire();
}

/** The system's own preference, without the player's override. */
export function systemReducedMotion(): boolean {
  return realMatchMedia !== null
    ? systemMatches(realMatchMedia)
    : typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function systemMatches(real: (query: string) => MediaQueryList): boolean {
  return real('(prefers-reduced-motion: reduce)').matches;
}

let realMatchMedia: ((query: string) => MediaQueryList) | null = null;

function effective(): boolean {
  return override || (realMatchMedia !== null && systemMatches(realMatchMedia));
}

class ReducedMotionQuery extends EventTarget {
  readonly media: string;
  private handler: ((this: MediaQueryList, event: MediaQueryListEvent) => unknown) | null = null;
  private readonly real: MediaQueryList;
  private last: boolean;
  private armed = false;

  constructor(media: string, real: MediaQueryList) {
    super();
    this.media = media;
    this.real = real;
    this.last = this.matches;
  }

  get matches(): boolean {
    return override || this.real.matches;
  }

  get onchange(): ((this: MediaQueryList, event: MediaQueryListEvent) => unknown) | null {
    return this.handler;
  }

  set onchange(handler: ((this: MediaQueryList, event: MediaQueryListEvent) => unknown) | null) {
    this.handler = handler;
    if (handler !== null) this.arm();
  }

  /**
   * Only a list somebody listens to is tracked. The HUD asks the question
   * afresh on every toast and score change and drops the answer; keeping each
   * of those would leak a list and a listener per call.
   */
  private arm(): void {
    if (this.armed) return;
    this.armed = true;
    this.last = this.matches;
    // The system changing its mind is heard through the real list.
    this.real.addEventListener('change', () => this.fire());
    instances.add(this);
  }

  override addEventListener(
    type: string,
    callback: EventListenerOrEventListenerObject | null,
    options?: AddEventListenerOptions | boolean,
  ): void {
    super.addEventListener(type, callback, options);
    if (type === 'change') this.arm();
  }

  fire(): void {
    const now = this.matches;
    if (now === this.last) return;
    this.last = now;
    const event = new Event('change') as Event & { matches: boolean; media: string };
    Object.defineProperties(event, { matches: { value: now }, media: { value: this.media } });
    this.dispatchEvent(event);
    this.handler?.call(this as unknown as MediaQueryList, event as unknown as MediaQueryListEvent);
  }

  addListener(callback: ((event: MediaQueryListEvent) => void) | null): void {
    if (callback) this.addEventListener('change', callback as EventListener);
  }

  removeListener(callback: ((event: MediaQueryListEvent) => void) | null): void {
    if (callback) this.removeEventListener('change', callback as EventListener);
  }
}

/** Wraps `window.matchMedia`. Idempotent. */
export function installReducedMotionOverride(): void {
  if (installed || typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
  installed = true;
  const real = window.matchMedia.bind(window);
  realMatchMedia = real;
  window.matchMedia = (query: string): MediaQueryList => {
    const list = real(query);
    return QUERY.test(query)
      ? (new ReducedMotionQuery(query, list) as unknown as MediaQueryList)
      : list;
  };
}
