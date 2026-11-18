/**
 * The settings: keys, look, comfort, screen, and a way to the sound controls.
 *
 * Everything here is a setting the player keeps between visits (`prefs.ts`,
 * `input/keybinds.ts`); this file owns only the markup and the capture of a
 * new key. Each control reports through {@link UiCallbacks.onChange} like the
 * ledger's, and the host echoes values back with {@link SettingsScreen.setState}.
 *
 * Sound is not repeated here. The ledger already carries every volume and
 * mute, so this screen points at them (`openSound`) instead of keeping a
 * second set that could disagree.
 *
 * Every control has its states: hover and press from the shared brass and
 * paper classes, a focus ring on each, and a disabled state with its reason
 * written beside it where the control cannot apply (no gamepad to rumble, the
 * system already asking for reduced motion, a browser that cannot resize the
 * interface).
 */

import {
  KEY_ACTIONS,
  KEY_GROUPS,
  keybinds,
  type KeyAction,
  type KeyActionSpec,
} from '../input/keybinds.ts';
import { icon } from './icons.ts';
import { systemReducedMotion } from './motion.ts';
import { canScaleInterface } from './prefs.ts';
import { UI_SCALES, type UiCallbacks, type UiScale, type UiState } from './types.ts';

export interface SettingsHooks {
  /** Closes the menu and opens the ledger at its sound controls. */
  openSound(): void;
}

type SliderKey = 'lookSensitivity' | 'shake';
type SwitchKey = 'invertLook' | 'haptics' | 'reduceMotion' | 'reduceFlashes' | 'shapeCues';

const SCALE_LABELS: Record<UiScale, string> = { 1: 'Standard', 1.15: 'Large', 1.3: 'Larger' };

interface SliderRow {
  readonly input: HTMLInputElement;
  readonly value: HTMLElement;
  readonly note: HTMLElement | null;
  readonly min: number;
  readonly max: number;
  readonly format: (v: number) => string;
}

interface SwitchRow {
  readonly button: HTMLButtonElement;
  readonly note: HTMLElement;
}

let uid = 0;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className !== undefined) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function motionReduced(): boolean {
  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function gamepadsSupported(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.getGamepads === 'function';
}

export class SettingsScreen {
  readonly element: HTMLElement;

  private readonly state: UiState;
  private readonly callbacks: UiCallbacks;
  private readonly hooks: SettingsHooks;
  private readonly controller = new AbortController();
  private readonly sliders = new Map<SliderKey, SliderRow>();
  private readonly switches = new Map<SwitchKey, SwitchRow>();
  private readonly scaleInputs = new Map<UiScale, HTMLInputElement>();
  private readonly keyButtons = new Map<KeyAction, HTMLButtonElement>();
  private readonly keyResets = new Map<KeyAction, HTMLButtonElement>();
  private resetAll!: HTMLButtonElement;
  private readonly noteEl: HTMLElement;
  private captureAction: KeyAction | null = null;
  private scaleNote!: HTMLElement;
  private stopKeyWatch: (() => void) | null = null;
  private readonly motionQuery: MediaQueryList | null;

  constructor(initial: UiState, callbacks: UiCallbacks, hooks: SettingsHooks) {
    this.state = { ...initial };
    this.callbacks = callbacks;
    this.hooks = hooks;
    const id = `settings${++uid}`;
    const signal = this.controller.signal;

    const root = el('div', 'settings');
    this.noteEl = el('p', 'settings__live');
    this.noteEl.setAttribute('role', 'status');
    this.noteEl.setAttribute('aria-live', 'polite');

    // ---- Keys ---------------------------------------------------------------
    const keys = this.section('Keys', `${id}-keys`);
    for (const group of KEY_GROUPS) {
      const block = el('div', 'settings__keys');
      block.setAttribute('role', 'group');
      block.setAttribute('aria-label', group);
      block.append(el('h4', 'settings__group', group));
      for (const spec of KEY_ACTIONS.filter((s) => s.group === group)) block.append(this.keyRow(spec));
      keys.append(block);
    }
    const keyFoot = el('div', 'settings__foot');
    this.resetAll = el('button', 'settings__btn', 'Put every key back');
    this.resetAll.type = 'button';
    this.resetAll.addEventListener('click', () => this.onResetAll(), { signal });
    keyFoot.append(this.resetAll);
    keys.append(keyFoot, this.noteEl);

    // ---- Look ---------------------------------------------------------------
    const look = this.section('Looking around', `${id}-look`);
    this.buildSlider(look, `${id}-look-speed`, 'lookSensitivity', {
      label: 'Look speed',
      tip: 'How far the camera turns for each move of the mouse or the right stick',
      min: 0.4,
      max: 2,
      step: 0.05,
      format: (v) => `${Math.round(v * 100)}%`,
    });
    this.buildSwitch(look, 'invertLook', 'Invert up and down',
      'Pull back to look up. Applies at the helm and in Fly, and to the right stick.');

    // ---- Comfort ------------------------------------------------------------
    const comfort = this.section('Comfort', `${id}-comfort`);
    this.buildSlider(comfort, `${id}-shake`, 'shake', {
      label: 'Camera shake',
      tip: 'How hard the camera jolts when you strike a shoal, a wave slams or thunder rolls. Zero holds it steady',
      min: 0,
      max: 1,
      step: 0.01,
      format: (v) => (v === 0 ? 'Steady' : `${Math.round(v * 100)}%`),
      note: true,
    });
    this.buildSwitch(comfort, 'haptics', 'Rumble',
      'Gamepads that can rumble feel hard knocks in the hands.');
    this.buildSwitch(comfort, 'reduceMotion', 'Reduce motion',
      'Calmer camera sweeps and no screen kicks. Also follows your system setting.');
    this.buildSwitch(comfort, 'reduceFlashes', 'Reduce flashes',
      'Softer lightning and fewer bright pulses.');

    // ---- Screen -------------------------------------------------------------
    const screen = this.section('Screen', `${id}-screen`);
    const size = el('div', 'field');
    const sizeLabel = el('span', 'field__label', 'Interface size');
    sizeLabel.id = `${id}-size-label`;
    const sizeGroup = el('div', 'plaques plaques--three');
    sizeGroup.setAttribute('role', 'radiogroup');
    sizeGroup.setAttribute('aria-labelledby', sizeLabel.id);
    for (const scale of UI_SCALES) {
      const label = el('label', 'plaque');
      const input = el('input', 'sr-radio');
      input.type = 'radio';
      input.name = `${id}-size`;
      input.value = String(scale);
      input.checked = this.state.uiScale === scale;
      input.setAttribute('aria-label', SCALE_LABELS[scale]);
      input.addEventListener('change', () => {
        if (!input.checked) return;
        this.state.uiScale = scale;
        this.callbacks.onChange('uiScale', scale);
      }, { signal });
      label.append(input, el('span', 'plaque__name', SCALE_LABELS[scale]));
      sizeGroup.append(label);
      this.scaleInputs.set(scale, input);
    }
    this.scaleNote = el('p', 'settings__hint');
    size.append(sizeLabel, sizeGroup, this.scaleNote);
    screen.append(size);
    this.buildSwitch(screen, 'shapeCues', 'Shapes as well as colours',
      'Danger, stalls and the frame rate gain a pattern or a mark, so none depends on colour alone.');

    // ---- Sound --------------------------------------------------------------
    const sound = this.section('Sound', `${id}-sound`);
    sound.append(el('p', 'settings__hint', 'Volume, music, sea, ship and clicks, each with its own mute, are in the ledger.'));
    const link = el('button', 'settings__btn settings__btn--link');
    link.type = 'button';
    link.append(icon('sound', 'settings__btn-icon'), el('span', undefined, 'Open the sound controls'));
    link.addEventListener('click', () => this.hooks.openSound(), { signal });
    const soundFoot = el('div', 'settings__foot');
    soundFoot.append(link);
    sound.append(soundFoot);

    const columns = [el('div', 'settings__col'), el('div', 'settings__col')];
    const sections = [...this.sectionEls];
    columns[0]!.append(sections[0]!);
    for (const s of sections.slice(1)) columns[1]!.append(s);
    root.append(...columns);
    this.element = root;

    this.motionQuery = typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
    this.motionQuery?.addEventListener('change', () => this.paintDisabled(), { signal });
    this.stopKeyWatch = keybinds.onChange(() => this.paintKeys());
    this.paintKeys();
    this.paintDisabled();
  }

  /** True while a key button waits for its key. The menu leaves Escape to this. */
  get capturing(): boolean {
    return this.captureAction !== null;
  }

  /** The first control, for the menu to focus on opening. */
  firstControl(): HTMLElement | null {
    return this.element.querySelector<HTMLElement>('button:not(:disabled), input:not(:disabled)');
  }

  setState(partial: Partial<UiState>): void {
    Object.assign(this.state, partial);
    for (const [key, row] of this.sliders) {
      const next = partial[key];
      if (typeof next !== 'number') continue;
      row.input.value = String(next);
      this.paintSlider(row);
    }
    for (const [key, row] of this.switches) {
      const next = partial[key];
      if (typeof next !== 'boolean') continue;
      this.paintSwitch(key, row);
    }
    if (partial.uiScale !== undefined) {
      const input = this.scaleInputs.get(partial.uiScale);
      if (input) input.checked = true;
    }
    this.paintDisabled();
  }

  /**
   * One key press while a key button is waiting. Returns true when it was
   * handled (and the menu must not act on it). Wired to the menu's own key
   * listener so Escape cancels the capture before it closes anything.
   */
  handleKey(event: KeyboardEvent): boolean {
    const action = this.captureAction;
    if (action === null) return false;
    if (event.key === 'Tab') {
      this.endCapture();
      return false;
    }
    event.preventDefault();
    if (event.code === 'Escape') {
      this.endCapture();
      this.say('Cancelled. The key is unchanged.');
      return true;
    }
    // A modifier on its own is the start of a shortcut, not a binding.
    if (/^(Shift|Control|Alt|Meta|OS|CapsLock)/.test(event.code)) return true;
    const label = KEY_ACTIONS.find((s) => s.id === action)?.label ?? '';
    const before = keybinds.codeOf(action);
    const result = keybinds.bind(action, event.code);
    if (!result.ok) {
      this.say('That key cannot be used. Try a letter, a number or an arrow.');
      return true;
    }
    this.endCapture();
    const now = keybinds.labelOf(action);
    if (result.swappedWith !== null) {
      const other = KEY_ACTIONS.find((s) => s.id === result.swappedWith)?.label ?? '';
      this.say(`${label} is now ${now}. ${other} took ${keybinds.labelOfCode(before)}, which this key held.`);
    } else {
      this.say(`${label} is now ${now}.`);
    }
    return true;
  }

  /** Stops waiting for a key, if it was. Returns whether it was. */
  cancelCapture(): boolean {
    if (this.captureAction === null) return false;
    this.endCapture();
    return true;
  }

  dispose(): void {
    this.controller.abort();
    this.stopKeyWatch?.();
    this.stopKeyWatch = null;
    this.element.remove();
  }

  // ------------------------------------------------------------------ builders

  private readonly sectionEls: HTMLElement[] = [];

  private section(title: string, id: string): HTMLElement {
    const section = el('section', 'ledger-section settings__section');
    const heading = el('h3', 'ledger-section__title', title);
    heading.id = id;
    section.setAttribute('aria-labelledby', id);
    section.append(heading);
    this.sectionEls.push(section);
    return section;
  }

  private keyRow(spec: KeyActionSpec): HTMLElement {
    const row = el('div', 'keyrow');
    const label = el('span', 'keyrow__label', spec.label);
    const button = el('button', 'keybtn');
    button.type = 'button';
    button.addEventListener('click', () => {
      if (this.captureAction === spec.id) this.endCapture();
      else this.startCapture(spec.id);
    }, { signal: this.controller.signal });
    // Leaving the button while it waits ends the wait.
    button.addEventListener('blur', () => {
      if (this.captureAction === spec.id) this.endCapture();
    }, { signal: this.controller.signal });
    const reset = el('button', 'keyreset');
    reset.type = 'button';
    reset.append(icon('close', 'keyreset__icon'));
    reset.addEventListener('click', () => {
      const swapped = keybinds.reset(spec.id);
      const now = keybinds.labelOf(spec.id);
      this.say(swapped === null
        ? `${spec.label} is back on ${now}.`
        : `${spec.label} is back on ${now}. ${KEY_ACTIONS.find((s) => s.id === swapped)?.label ?? ''} swapped places with it.`);
      button.focus();
    }, { signal: this.controller.signal });
    this.keyButtons.set(spec.id, button);
    this.keyResets.set(spec.id, reset);
    row.append(label, button, reset);
    return row;
  }

  private buildSlider(
    parent: HTMLElement,
    id: string,
    key: SliderKey,
    spec: {
      label: string;
      tip: string;
      min: number;
      max: number;
      step: number;
      format: (v: number) => string;
      note?: boolean;
    },
  ): void {
    const field = el('div', 'field field--slider');
    field.title = spec.tip;
    const row = el('div', 'field__row');
    const label = el('label', 'field__label', spec.label);
    label.htmlFor = id;
    const value = el('span', 'field__value');
    row.append(label, value);
    const input = el('input', 'range');
    input.type = 'range';
    input.id = id;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(this.state[key]);
    const note = spec.note ? el('p', 'settings__hint') : null;
    const refs: SliderRow = { input, value, note, min: spec.min, max: spec.max, format: spec.format };
    this.sliders.set(key, refs);
    input.addEventListener('input', () => {
      const next = Math.min(spec.max, Math.max(spec.min, Number(input.value)));
      this.state[key] = next;
      this.paintSlider(refs);
      this.callbacks.onChange(key, next);
    }, { signal: this.controller.signal });
    field.append(row, input);
    if (note) field.append(note);
    parent.append(field);
    this.paintSlider(refs);
  }

  private buildSwitch(parent: HTMLElement, key: SwitchKey, labelText: string, hint: string): void {
    const wrap = el('div', 'switchfield');
    const button = el('button', 'switchrow');
    button.type = 'button';
    button.setAttribute('role', 'switch');
    const box = el('span', 'switch');
    box.append(el('span', 'switch__knob'));
    button.append(box, el('span', 'switchrow__label', labelText));
    const note = el('p', 'settings__hint', hint);
    note.id = `settings-switch-${key}-${uid}-note`;
    button.setAttribute('aria-describedby', note.id);
    button.addEventListener('click', () => {
      const next = !this.state[key];
      this.state[key] = next;
      this.paintSwitch(key, this.switches.get(key)!);
      this.callbacks.onChange(key, next);
      this.paintDisabled();
    }, { signal: this.controller.signal });
    const refs: SwitchRow = { button, note };
    this.switches.set(key, refs);
    wrap.append(button, note);
    parent.append(wrap);
    this.paintSwitch(key, refs);
  }

  // ---------------------------------------------------------------------- paint

  private paintSlider(row: SliderRow): void {
    const current = Number(row.input.value);
    const pct = ((current - row.min) / (row.max - row.min)) * 100;
    row.input.style.setProperty('--fill', `${Math.min(100, Math.max(0, pct)).toFixed(2)}%`);
    const text = row.format(current);
    row.value.textContent = text;
    row.input.setAttribute('aria-valuetext', text);
  }

  private paintSwitch(key: SwitchKey, row: SwitchRow): void {
    const on = this.state[key];
    row.button.setAttribute('aria-checked', on ? 'true' : 'false');
    row.button.classList.toggle('is-on', on);
  }

  /** Disabled states, each with the reason written where the player will see it. */
  private paintDisabled(): void {
    const reduced = motionReduced();
    const shake = this.sliders.get('shake');
    if (shake) {
      shake.input.disabled = reduced;
      shake.input.closest('.field')?.classList.toggle('is-disabled', reduced);
      if (shake.note) {
        shake.note.textContent = reduced ? 'Motion is reduced, so the camera holds steady.' : '';
        shake.note.hidden = !reduced;
      }
    }

    const motion = this.switches.get('reduceMotion');
    if (motion) {
      const system = systemReducedMotion();
      motion.button.disabled = system;
      motion.button.setAttribute('aria-checked', system || this.state.reduceMotion ? 'true' : 'false');
      motion.button.classList.toggle('is-on', system || this.state.reduceMotion);
      motion.note.textContent = system
        ? 'Your system already asks for reduced motion.'
        : 'Calmer camera sweeps and no screen kicks. Also follows your system setting.';
    }

    const haptics = this.switches.get('haptics');
    if (haptics) {
      const supported = gamepadsSupported();
      haptics.button.disabled = !supported;
      haptics.note.textContent = supported
        ? 'Gamepads that can rumble feel hard knocks in the hands.'
        : 'This browser cannot see gamepads.';
    }

    const scalable = canScaleInterface();
    for (const input of this.scaleInputs.values()) input.disabled = !scalable;
    this.scaleNote.textContent = scalable ? '' : 'This browser cannot resize the interface.';
    this.scaleNote.hidden = scalable;

    this.resetAll.disabled = !keybinds.customised;
  }

  private paintKeys(): void {
    for (const spec of KEY_ACTIONS) {
      const button = this.keyButtons.get(spec.id);
      const reset = this.keyResets.get(spec.id);
      if (!button || !reset) continue;
      const waiting = this.captureAction === spec.id;
      const key = keybinds.labelOf(spec.id);
      button.textContent = waiting ? 'Press a key' : key;
      button.classList.toggle('is-capturing', waiting);
      button.classList.toggle('is-changed', !keybinds.isDefault(spec.id));
      button.setAttribute('aria-label', waiting
        ? `${spec.label}: press the new key, or Escape to cancel`
        : `${spec.label}: ${key}. Activate to change.`);
      const changed = !keybinds.isDefault(spec.id);
      reset.hidden = !changed;
      reset.setAttribute('aria-label',
        `Put ${spec.label.toLowerCase()} back on ${keybinds.labelOfCode(spec.default)}`);
      reset.title = `Back to ${keybinds.labelOfCode(spec.default)}`;
    }
    this.paintDisabled();
  }

  // ------------------------------------------------------------------- capture

  private startCapture(action: KeyAction): void {
    this.captureAction = action;
    this.paintKeys();
    const spec = KEY_ACTIONS.find((s) => s.id === action);
    this.say(`Press the key for "${spec?.label ?? ''}". Escape cancels.`);
  }

  private endCapture(): void {
    if (this.captureAction === null) return;
    this.captureAction = null;
    this.paintKeys();
  }

  private onResetAll(): void {
    keybinds.resetAll();
    this.say('Every key is back where it began.');
  }

  private say(text: string): void {
    this.noteEl.textContent = text;
  }
}
