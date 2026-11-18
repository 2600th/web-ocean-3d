/**
 * The pause menu and the settings it opens.
 *
 * Escape (or a gamepad's menu button) opens it from anywhere once the sea is
 * showing. While it is open the simulation is held and the sound ducked (the
 * host does both, through `onPause`), nothing behind it can be reached, and
 * focus stays inside it. Closing it puts focus back where it was.
 *
 *   Resume · Settings · Restart voyage · Quit to title
 *
 * **Keys.** One capture-phase listener on `window` runs while the menu is
 * open and stops every key before the game's own listeners (the helm, the
 * camera, the mode keys) can see it, so a W pressed in the menu is not a sail
 * order. It handles Escape (back, or resume), Tab (kept inside) and the
 * capture of a new key for the settings; everything else carries on to the
 * focused control, so Enter, Space and the arrows still work natively.
 *
 * **Behind it.** Everything else in the interface root is made `inert`, which
 * removes it from the tab order and from the pointer, and a scrim catches the
 * rest of the screen.
 *
 * **Gamepad.** The loop is held, so the host's gamepad poll is not running;
 * the menu reads the pad itself while it is open: d-pad or left stick to move
 * focus, left and right to adjust a slider or step a choice, A to press, B to
 * go back, menu to resume. The menu button that opened it has to be let go of
 * before it will close, or the same press would open it again.
 *
 * Pure DOM, no three.js.
 */

import { icon } from './icons.ts';
import { SettingsScreen, type SettingsHooks } from './SettingsScreen.ts';
import type { UiCallbacks, UiState } from './types.ts';

export interface ShellHooks extends SettingsHooks {
  /** The menu opened (true) or closed (false): hold the simulation and duck the sound. */
  onPause(paused: boolean): void;
  /** Puts the ship and the voyage back at the start. */
  restartVoyage(): void;
  /** Leaves for the title screen. */
  quitToTitle(): void;
}

type View = 'menu' | 'settings';

const FOCUSABLE = 'button:not(:disabled):not([hidden]), input:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])';

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

function isBootShowing(): boolean {
  const boot = document.getElementById('boot');
  return boot !== null && !boot.classList.contains('boot--hidden');
}

export class PauseMenu {
  private readonly root: HTMLElement;
  private readonly hooks: ShellHooks;
  private readonly controller = new AbortController();
  private readonly shell: HTMLElement;
  private readonly sheet: HTMLElement;
  private readonly menuView: HTMLElement;
  private readonly settingsView: HTMLElement;
  private readonly settings: SettingsScreen;
  private readonly resumeButton: HTMLButtonElement;
  private readonly settingsButton: HTMLButtonElement;
  private readonly menuTitle: HTMLElement;
  private readonly settingsTitle: HTMLElement;
  private view: View = 'menu';
  private opened = false;
  private returnFocus: HTMLElement | null = null;
  private readonly made: HTMLElement[] = [];
  private padFrame = 0;
  private disposed = false;

  // Pad state: last frame's pressed buttons, and the stick's repeat clock.
  private padDown = new Set<number>();
  private padStickAt = 0;
  private closeOnRelease = false;

  constructor(root: HTMLElement, state: UiState, callbacks: UiCallbacks, hooks: ShellHooks) {
    this.root = root;
    this.hooks = hooks;
    const signal = this.controller.signal;

    this.settings = new SettingsScreen(state, callbacks, {
      openSound: () => {
        this.close();
        hooks.openSound();
      },
    });

    const shell = el('div', 'shell');
    shell.hidden = true;
    const sheet = el('div', 'shell__sheet parchment');
    sheet.setAttribute('role', 'dialog');
    sheet.setAttribute('aria-modal', 'true');

    // ---- The menu -----------------------------------------------------------
    const menu = el('div', 'shell__view shell__view--menu');
    const menuTitle = el('h2', 'shell__title', 'Paused');
    menuTitle.id = 'shell-title-menu';
    menuTitle.tabIndex = -1;
    const flourish = el('div', 'boot__flourish shell__flourish');
    flourish.setAttribute('aria-hidden', 'true');
    flourish.append(el('span'));
    const list = el('div', 'shell__menu');
    this.resumeButton = this.button('Resume', 'shell__btn shell__btn--primary', 'anchor', () => this.close());
    this.settingsButton = this.button('Settings', 'shell__btn', 'ledger', () => this.show('settings'));
    const restart = this.button('Restart voyage', 'shell__btn', 'mark', () => {
      this.close();
      hooks.restartVoyage();
    });
    // No close: the page is about to reload, and a resumed frame in between
    // would only flicker.
    const quit = this.button('Quit to title', 'shell__btn shell__btn--quiet', 'close', () => {
      hooks.quitToTitle();
    });
    list.append(this.resumeButton, this.settingsButton, restart, quit);
    menu.append(menuTitle, flourish, list);

    // ---- The settings -------------------------------------------------------
    const settingsView = el('div', 'shell__view shell__view--settings');
    settingsView.hidden = true;
    const head = el('header', 'shell__head');
    const settingsTitle = el('h2', 'shell__title shell__title--small', 'Settings');
    settingsTitle.id = 'shell-title-settings';
    settingsTitle.tabIndex = -1;
    const back = this.button('Back', 'shell__btn shell__btn--small', 'chevron', () => this.show('menu'));
    back.classList.add('shell__back');
    head.append(settingsTitle, back);
    const scroll = el('div', 'shell__scroll');
    scroll.append(this.settings.element);
    settingsView.append(head, scroll);

    sheet.append(menu, settingsView);
    shell.append(sheet);
    root.append(shell);

    this.shell = shell;
    this.sheet = sheet;
    this.menuView = menu;
    this.settingsView = settingsView;
    this.menuTitle = menuTitle;
    this.settingsTitle = settingsTitle;
    sheet.setAttribute('aria-labelledby', menuTitle.id);

    // Escape opens it. Captured, so nothing downstream sees the key first, and
    // only once there is a sea to pause: not behind the boot overlay.
    window.addEventListener('keydown', this.onOpenKey, { signal, capture: true });
  }

  get isOpen(): boolean {
    return this.opened;
  }

  /** Opens the menu, or its settings view. A no-op while the boot overlay is up. */
  open(view: View = 'menu'): void {
    if (this.disposed || this.opened || isBootShowing()) return;
    this.opened = true;
    this.returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    this.closeOnRelease = false;

    // Everything else in the interface root is put out of reach.
    for (const child of Array.from(this.root.children)) {
      if (child === this.shell || !(child instanceof HTMLElement) || child.inert) continue;
      child.inert = true;
      this.made.push(child);
    }
    this.shell.hidden = false;
    window.addEventListener('keydown', this.onKey, { signal: this.controller.signal, capture: true });
    this.show(view);
    this.hooks.onPause(true);
    this.startPad();
  }

  /** Closes it and gives focus back to where it was. */
  close(): void {
    if (!this.opened) return;
    this.opened = false;
    this.settings.cancelCapture();
    window.removeEventListener('keydown', this.onKey, { capture: true });
    window.cancelAnimationFrame(this.padFrame);
    for (const child of this.made.splice(0)) child.inert = false;
    this.shell.hidden = true;
    const target = this.returnFocus;
    this.returnFocus = null;
    // After the inert flags are lifted, or the element could not take focus.
    if (target !== null && target.isConnected) target.focus({ preventScroll: true });
    this.hooks.onPause(false);
  }

  toggle(): void {
    if (this.opened) this.close();
    else this.open();
  }

  setState(partial: Partial<UiState>): void {
    this.settings.setState(partial);
  }

  dispose(): void {
    if (this.disposed) return;
    this.close();
    this.disposed = true;
    this.controller.abort();
    this.settings.dispose();
    this.shell.remove();
  }

  // -------------------------------------------------------------------- views

  private show(view: View): void {
    this.settings.cancelCapture();
    const fromSettings = this.view === 'settings' && view === 'menu';
    this.view = view;
    this.menuView.hidden = view !== 'menu';
    this.settingsView.hidden = view !== 'settings';
    this.shell.dataset.view = view;
    this.sheet.classList.toggle('shell__sheet--wide', view === 'settings');
    this.sheet.setAttribute('aria-labelledby', view === 'menu' ? this.menuTitle.id : this.settingsTitle.id);
    if (view === 'menu') {
      (fromSettings ? this.settingsButton : this.resumeButton).focus({ preventScroll: true });
    } else {
      this.settingsView.querySelector<HTMLElement>('.shell__back')?.focus({ preventScroll: true });
      this.settingsView.querySelector<HTMLElement>('.shell__scroll')?.scrollTo({ top: 0 });
    }
  }

  /** Back from settings, or resume from the menu. */
  private back(): void {
    if (this.view === 'settings') this.show('menu');
    else this.close();
  }

  private button(
    label: string,
    className: string,
    iconName: 'anchor' | 'ledger' | 'mark' | 'close' | 'chevron',
    onClick: () => void,
  ): HTMLButtonElement {
    const button = el('button', className);
    button.type = 'button';
    button.append(icon(iconName, 'shell__btn-icon'), el('span', 'shell__btn-label', label));
    button.addEventListener('click', onClick, { signal: this.controller.signal });
    return button;
  }

  // --------------------------------------------------------------------- keys

  private readonly onOpenKey = (event: KeyboardEvent): void => {
    if (this.opened || event.code !== 'Escape' || event.repeat) return;
    if (event.ctrlKey || event.metaKey || event.altKey || isBootShowing()) return;
    event.preventDefault();
    event.stopPropagation();
    this.open();
  };

  private readonly onKey = (event: KeyboardEvent): void => {
    // Nothing reaches the game while the menu is up.
    event.stopPropagation();
    if (this.settings.handleKey(event)) return;
    if (event.code === 'Escape') {
      event.preventDefault();
      if (!event.repeat) this.back();
      return;
    }
    if (event.key === 'Tab') this.keepFocus(event);
    else if (!this.shell.contains(event.target as Node | null)) {
      // Focus fell to the page (a click on the scrim): bring it back.
      this.focusables()[0]?.focus({ preventScroll: true });
    }
  };

  /**
   * The visible view's tab stops, in order. A radio group is one stop, as the
   * platform treats it: the checked radio, or the first when none is checked.
   */
  private focusables(): HTMLElement[] {
    const view = this.view === 'menu' ? this.menuView : this.settingsView;
    const nodes = Array.from(view.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
      (node) => node.offsetParent !== null || node === document.activeElement,
    );
    const groups = new Map<string, HTMLInputElement[]>();
    for (const node of nodes) {
      if (node instanceof HTMLInputElement && node.type === 'radio') {
        const list = groups.get(node.name) ?? [];
        list.push(node);
        groups.set(node.name, list);
      }
    }
    return nodes.filter((node) => {
      if (!(node instanceof HTMLInputElement) || node.type !== 'radio') return true;
      const group = groups.get(node.name) ?? [];
      const stop = group.find((r) => r.checked) ?? group[0];
      return node === stop;
    });
  }

  /** Tab wraps inside the visible view. Radio groups count once, as the platform does. */
  private keepFocus(event: KeyboardEvent): void {
    const nodes = this.focusables();
    if (nodes.length === 0) {
      event.preventDefault();
      return;
    }
    const first = nodes[0]!;
    const last = nodes[nodes.length - 1]!;
    const active = document.activeElement;
    if (!this.shell.contains(active)) {
      event.preventDefault();
      first.focus({ preventScroll: true });
    } else if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus({ preventScroll: true });
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus({ preventScroll: true });
    }
  }

  // ---------------------------------------------------------------------- pad

  private startPad(): void {
    const pad = this.pad();
    // Whatever is held now (the menu button that opened this) is not a press.
    this.padDown = new Set<number>();
    pad?.buttons.forEach((b, i) => {
      if (b.pressed) this.padDown.add(i);
    });
    this.closeOnRelease = false;
    this.padFrame = window.requestAnimationFrame(this.pollPad);
  }

  private pad(): Gamepad | null {
    try {
      if (typeof navigator.getGamepads !== 'function') return null;
      for (const pad of navigator.getGamepads()) if (pad?.connected) return pad;
    } catch {
      // No gamepad API here.
    }
    return null;
  }

  private readonly pollPad = (now: number): void => {
    if (!this.opened) return;
    this.padFrame = window.requestAnimationFrame(this.pollPad);
    const pad = this.pad();
    if (pad === null) return;

    const down = new Set<number>();
    pad.buttons.forEach((b, i) => {
      if (b.pressed) down.add(i);
    });
    const pressed = (i: number): boolean => down.has(i) && !this.padDown.has(i);

    if (this.closeOnRelease && !down.has(9)) {
      this.padDown = down;
      this.close();
      return;
    }
    // Start: resume, once the button is up again.
    if (pressed(9)) this.closeOnRelease = true;
    else if (pressed(1)) this.back();
    else if (pressed(0)) this.activate();

    let vertical = (pressed(13) ? 1 : 0) - (pressed(12) ? 1 : 0);
    let horizontal = (pressed(15) ? 1 : 0) - (pressed(14) ? 1 : 0);
    // The left stick repeats while held, a little slower than a key would.
    const sx = pad.axes[0] ?? 0;
    const sy = pad.axes[1] ?? 0;
    if (Math.max(Math.abs(sx), Math.abs(sy)) > 0.6 && now - this.padStickAt > 260) {
      this.padStickAt = now;
      if (Math.abs(sy) >= Math.abs(sx)) vertical = vertical || Math.sign(sy);
      else horizontal = horizontal || Math.sign(sx);
    }
    if (vertical !== 0) this.moveFocus(vertical);
    if (horizontal !== 0) this.nudge(horizontal);
    this.padDown = down;
  };

  private moveFocus(step: number): void {
    const nodes = this.focusables();
    if (nodes.length === 0) return;
    const index = nodes.findIndex((node) => node === document.activeElement);
    const next = nodes[(index + step + nodes.length) % nodes.length] ?? nodes[0]!;
    next.focus({ preventScroll: true });
    next.scrollIntoView({ block: 'nearest' });
  }

  /** Left or right on the focused control: a slider steps, a choice moves. */
  private nudge(step: number): void {
    const active = document.activeElement;
    if (active instanceof HTMLInputElement && active.type === 'range') {
      if (step > 0) active.stepUp();
      else active.stepDown();
      active.dispatchEvent(new Event('input', { bubbles: true }));
    } else if (active instanceof HTMLInputElement && active.type === 'radio') {
      const group = Array.from(this.shell.querySelectorAll<HTMLInputElement>(`input[type="radio"][name="${active.name}"]`))
        .filter((r) => !r.disabled);
      const at = group.indexOf(active);
      const target = group[at + step];
      if (target) {
        target.checked = true;
        target.focus();
        target.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }
  }

  private activate(): void {
    const active = document.activeElement;
    if (active instanceof HTMLElement && this.shell.contains(active) && !(active instanceof HTMLInputElement && active.type === 'range')) {
      active.click();
    }
  }
}
