/**
 * Interface sounds, as cue names.
 *
 * The UI says *what happened* — a hover, a press, a slider moved to 0.3 — and
 * the host decides what that sounds like. One delegated listener per event type
 * on the UI root covers every control the panel and HUD build, including ones
 * added later, without either of them knowing sound exists. Like the rest of
 * `src/ui`, this imports neither three.js nor the audio system.
 *
 * Listeners run in the bubbling phase, after the control's own handler, so a
 * switch or a fold button is read in its *new* state: toggling on sounds like
 * on.
 */

export type UiSoundCue =
  | 'hover'
  | 'press'
  | 'select'
  | 'slide'
  | 'toggleOn'
  | 'toggleOff'
  | 'open'
  | 'close';

/** `value` is 0..1 for `slide` (the slider's position); unused otherwise. */
export type UiSoundSink = (cue: UiSoundCue, value?: number) => void;

const INTERACTIVE = 'button, [role="switch"], [role="tab"], a[href], label, input[type="range"]';

/** Wires `play` to every control under `root`. Returns a detach function. */
export function attachUiSounds(root: HTMLElement, play: UiSoundSink): () => void {
  const controller = new AbortController();
  const signal = controller.signal;
  let hovered: Element | null = null;

  root.addEventListener(
    'pointerover',
    (event) => {
      // Mouse only: a touch has no hover, and a tick before every tap is noise.
      if (event.pointerType !== 'mouse') return;
      const target = closest(event.target, INTERACTIVE);
      if (target === hovered) return;
      hovered = target;
      if (target !== null && !isDisabled(target)) play('hover');
    },
    { signal, passive: true },
  );

  root.addEventListener(
    'click',
    (event) => {
      const target = closest(event.target, 'button, [role="switch"], [role="tab"], a[href]');
      if (target === null || isDisabled(target)) return;
      if (target.getAttribute('role') === 'switch') {
        play(target.getAttribute('aria-checked') === 'true' ? 'toggleOn' : 'toggleOff');
      } else if (target.hasAttribute('aria-expanded')) {
        play(target.getAttribute('aria-expanded') === 'true' ? 'open' : 'close');
      } else if (target.hasAttribute('aria-pressed') || target.getAttribute('role') === 'tab') {
        play('select');
      } else {
        play('press');
      }
    },
    { signal },
  );

  root.addEventListener(
    'change',
    (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement && (target.type === 'radio' || target.type === 'checkbox')) {
        play('select');
      }
    },
    { signal },
  );

  root.addEventListener(
    'input',
    (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement) || target.type !== 'range') return;
      const min = Number(target.min || 0);
      const max = Number(target.max || 100);
      const value = Number(target.value);
      play('slide', max > min && Number.isFinite(value) ? (value - min) / (max - min) : 0.5);
    },
    { signal, passive: true },
  );

  return () => controller.abort();
}

function closest(target: EventTarget | null, selector: string): Element | null {
  return target instanceof Element ? target.closest(selector) : null;
}

function isDisabled(element: Element): boolean {
  return element.getAttribute('aria-disabled') === 'true' || (element as HTMLButtonElement).disabled === true;
}
