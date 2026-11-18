/**
 * Key bindings: which physical key does what, remembered between visits.
 *
 * Bindings are `KeyboardEvent.code` values (the position of the key, not what
 * it prints), as the helm's keys have always been, so a French keyboard still
 * has its sail keys under the left hand. The label shown to the player asks the
 * browser what that key prints where it can (Chromium's keyboard layout map)
 * and falls back to the code's own name.
 *
 * One module-level instance, `keybinds`, is shared by the helm's input
 * (`gameplay`, `physics`), the HUD's hints and the settings screen, so a key
 * rebound in the settings is the key the ship answers to, with nothing to
 * pass around. It reads localStorage once, defensively: unavailable, malformed
 * or from an older build, the defaults stand.
 *
 * Conflicts: binding a key another action already holds swaps the two, so
 * no state ever has two actions on one key and no action is left without one.
 * The settings screen says so when it happens. Fly mode's WASD, Shift, Space
 * and Ctrl are the free camera's own and are not in this list.
 *
 * Pure: no three.js, no DOM beyond `localStorage` and `navigator`.
 */

export type KeyAction =
  | 'sailMore'
  | 'sailLess'
  | 'trimIn'
  | 'trimOut'
  | 'steerPort'
  | 'steerStarboard'
  | 'anchor'
  | 'modeOrbit'
  | 'modeFly'
  | 'modeHelm'
  | 'modeTour';

export type KeyGroup = 'Sails' | 'Steering' | 'Anchor' | 'Camera';

export interface KeyActionSpec {
  readonly id: KeyAction;
  readonly group: KeyGroup;
  /** What the settings screen calls it. */
  readonly label: string;
  /** What the HUD's hint says it does. */
  readonly hint: string;
  readonly default: string;
}

export const KEY_ACTIONS: readonly KeyActionSpec[] = [
  { id: 'sailMore', group: 'Sails', label: 'More sail', hint: 'More sail', default: 'KeyW' },
  { id: 'sailLess', group: 'Sails', label: 'Less sail', hint: 'Less sail', default: 'KeyS' },
  { id: 'trimIn', group: 'Sails', label: 'Haul the sheets in', hint: 'Sheets in', default: 'KeyQ' },
  { id: 'trimOut', group: 'Sails', label: 'Ease the sheets out', hint: 'Sheets out', default: 'KeyE' },
  { id: 'steerPort', group: 'Steering', label: 'Steer to port', hint: 'Steer to port', default: 'KeyA' },
  { id: 'steerStarboard', group: 'Steering', label: 'Steer to starboard', hint: 'Steer to starboard', default: 'KeyD' },
  { id: 'anchor', group: 'Anchor', label: 'Drop or weigh anchor', hint: 'Anchor', default: 'KeyR' },
  { id: 'modeOrbit', group: 'Camera', label: 'Orbit view', hint: 'Orbit', default: 'Digit1' },
  { id: 'modeFly', group: 'Camera', label: 'Fly view', hint: 'Fly', default: 'Digit2' },
  { id: 'modeHelm', group: 'Camera', label: 'Helm view', hint: 'Helm', default: 'Digit3' },
  { id: 'modeTour', group: 'Camera', label: 'Tour view', hint: 'Tour', default: 'Digit4' },
];

export const KEY_GROUPS: readonly KeyGroup[] = ['Sails', 'Steering', 'Anchor', 'Camera'];

const STORAGE_KEY = 'web-ocean-3d:keys';

/**
 * Codes a binding may take: letters, digits, arrows, the main punctuation, the
 * numpad and Space. Escape is the pause key, Tab and Enter move focus, and the
 * modifiers and function keys belong to the browser and the system.
 */
const BINDABLE = /^(Key[A-Z]|Digit[0-9]|Numpad[A-Za-z0-9]+|Arrow(Up|Down|Left|Right)|Space|Comma|Period|Slash|Semicolon|Quote|Bracket(Left|Right)|Backslash|Minus|Equal|Backquote|Home|End|PageUp|PageDown|Insert|Delete)$/;

export function isBindableCode(code: string): boolean {
  return BINDABLE.test(code);
}

const DEFAULTS: Record<KeyAction, string> = Object.fromEntries(
  KEY_ACTIONS.map((spec) => [spec.id, spec.default]),
) as Record<KeyAction, string>;

/** The plain name of a key by its code, for when the browser will not say. */
export function codeName(code: string): string {
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return 'Num ' + code.slice(6).replace('Add', '+').replace('Subtract', '-').replace('Multiply', '*').replace('Divide', '/').replace('Decimal', '.').replace('Enter', 'Enter');
  switch (code) {
    case 'ArrowUp': return 'Up';
    case 'ArrowDown': return 'Down';
    case 'ArrowLeft': return 'Left';
    case 'ArrowRight': return 'Right';
    case 'Space': return 'Space';
    case 'Comma': return ',';
    case 'Period': return '.';
    case 'Slash': return '/';
    case 'Semicolon': return ';';
    case 'Quote': return "'";
    case 'BracketLeft': return '[';
    case 'BracketRight': return ']';
    case 'Backslash': return '\\';
    case 'Minus': return '-';
    case 'Equal': return '=';
    case 'Backquote': return '`';
    case 'PageUp': return 'Page up';
    case 'PageDown': return 'Page down';
    default: return code;
  }
}

export type BindResult =
  /** The key is bound; `swappedWith` is the action that gave up the key, when one did. */
  | { readonly ok: true; readonly swappedWith: KeyAction | null }
  | { readonly ok: false; readonly reason: 'unbindable' };

type Listener = () => void;

export class KeyBindings {
  private readonly codes: Record<KeyAction, string> = { ...DEFAULTS };
  private readonly lower: Record<KeyAction, string> = { ...lowered(DEFAULTS) };
  private readonly byCode = new Map<string, KeyAction>();
  private readonly listeners = new Set<Listener>();
  /** What a code prints on this keyboard, once the browser has said. */
  private layout: { get(code: string): string | undefined } | null = null;
  private layoutAsked = false;

  constructor() {
    this.load();
    this.reindex();
  }

  /** The code bound to `action`. */
  codeOf(action: KeyAction): string {
    return this.codes[action];
  }

  /** The same, lower-cased: how the controllers key their held-key sets. */
  lowerCodeOf(action: KeyAction): string {
    return this.lower[action];
  }

  /** The action a physical key is bound to, or null. */
  actionFor(code: string): KeyAction | null {
    return this.byCode.get(code) ?? null;
  }

  isDefault(action: KeyAction): boolean {
    return this.codes[action] === DEFAULTS[action];
  }

  /** True when any binding differs from the defaults. */
  get customised(): boolean {
    return KEY_ACTIONS.some((spec) => this.codes[spec.id] !== spec.default);
  }

  /**
   * What `action`'s key is called on screen. Layout-aware where the browser
   * supports it: the key printed "Z" where a QWERTY keyboard prints "W".
   */
  labelOf(action: KeyAction): string {
    return this.labelOfCode(this.codes[action]);
  }

  labelOfCode(code: string): string {
    this.askLayout();
    const printed = this.layout?.get(code);
    if (printed !== undefined && printed.length > 0 && !code.startsWith('Numpad')) {
      return printed.length === 1 ? printed.toUpperCase() : printed;
    }
    return codeName(code);
  }

  /** Space-separated labels, as the HUD's keycaps want them ("A D"). */
  labelsOf(...actions: KeyAction[]): string {
    return actions.map((action) => this.labelOf(action)).join(' ');
  }

  bind(action: KeyAction, code: string): BindResult {
    if (!isBindableCode(code)) return { ok: false, reason: 'unbindable' };
    const current = this.codes[action];
    if (current === code) return { ok: true, swappedWith: null };
    const holder = this.byCode.get(code) ?? null;
    if (holder !== null && holder !== action) this.codes[holder] = current;
    this.codes[action] = code;
    this.commit();
    return { ok: true, swappedWith: holder !== null && holder !== action ? holder : null };
  }

  /**
   * Puts one action back on its default key. If another action has taken that
   * key since, the two swap, so the reset never leaves a duplicate.
   */
  reset(action: KeyAction): KeyAction | null {
    const result = this.bind(action, DEFAULTS[action]);
    return result.ok ? result.swappedWith : null;
  }

  resetAll(): void {
    for (const spec of KEY_ACTIONS) this.codes[spec.id] = spec.default;
    this.commit();
  }

  /** Called after any change; returns the call that stops listening. */
  onChange(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // ---------------------------------------------------------------- internals

  private commit(): void {
    this.reindex();
    this.save();
    for (const listener of [...this.listeners]) listener();
  }

  private reindex(): void {
    this.byCode.clear();
    for (const spec of KEY_ACTIONS) {
      this.lower[spec.id] = this.codes[spec.id].toLowerCase();
      this.byCode.set(this.codes[spec.id], spec.id);
    }
  }

  private load(): void {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw === null) return;
      const stored = JSON.parse(raw) as Record<string, unknown>;
      const next: Record<KeyAction, string> = { ...DEFAULTS };
      for (const spec of KEY_ACTIONS) {
        const value = stored[spec.id];
        if (typeof value === 'string' && isBindableCode(value)) next[spec.id] = value;
      }
      // A stored set with a duplicate (hand-edited, or from a build with other
      // defaults) is discarded whole rather than half-applied.
      if (new Set(Object.values(next)).size !== KEY_ACTIONS.length) return;
      Object.assign(this.codes, next);
    } catch {
      // Unavailable or unreadable: the defaults stand.
    }
  }

  private save(): void {
    try {
      if (!this.customised) {
        window.localStorage.removeItem(STORAGE_KEY);
        return;
      }
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.codes));
    } catch {
      // Storage refused: the bindings last for this visit only.
    }
  }

  /** Asks the browser once what each key prints; the labels refresh when it answers. */
  private askLayout(): void {
    if (this.layoutAsked) return;
    this.layoutAsked = true;
    try {
      const keyboard = (navigator as Navigator & {
        keyboard?: { getLayoutMap?: () => Promise<{ get(code: string): string | undefined }> };
      }).keyboard;
      void keyboard?.getLayoutMap?.().then((map) => {
        this.layout = map;
        for (const listener of [...this.listeners]) listener();
      }).catch(() => undefined);
    } catch {
      // No layout map (Firefox, Safari): the codes' own names.
    }
  }
}

function lowered(source: Record<KeyAction, string>): Record<KeyAction, string> {
  const out = {} as Record<KeyAction, string>;
  for (const spec of KEY_ACTIONS) out[spec.id] = source[spec.id].toLowerCase();
  return out;
}

/** The bindings in force. Shared by the helm, the HUD and the settings screen. */
export const keybinds = new KeyBindings();
