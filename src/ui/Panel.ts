/**
 * The captain's ledger: the right-hand control panel.
 *
 * Pure DOM — no three.js, no framework. It owns nothing but its own markup and
 * reports every user edit through {@link UiCallbacks.onChange}. The host app is
 * free to echo values back with {@link Panel.setState} (which never re-fires the
 * callbacks, so it is safe to call from inside a change handler).
 *
 * Sections, in the order a visitor reaches for them: the *look* (preset cards
 * and quality), weather and water (the sliders that change the render), sound
 * and screen (volume, music and sharpness), then two folds: the shipwright's
 * debug toggles, which are the author's instruments rather than the visitor's,
 * and "About the tech", where the jargon kept out of every label lives. The ledger folds to its title on desktop and is a bottom sheet under
 * 900px.
 *
 * Presets and quality are native radio inputs styled as cards and plaques, so
 * arrow keys, focus and form semantics come from the platform rather than from
 * a hand-rolled roving tabindex.
 */

import { icon } from './icons.ts';
import {
  PRESET_IDS,
  QUALITY_TIERS,
  type PresetId,
  type QualityTier,
  type UiCallbacks,
  type UiState,
} from './types.ts';

/** Where "View Source" goes. */
const SOURCE_URL = 'https://github.com/2600th/web-ocean-3d';

type SliderKey =
  | 'windSpeed'
  | 'peakWavelength'
  | 'cloudCoverage'
  | 'timeOfDay'
  | 'fogDensity'
  | 'volume'
  | 'musicVolume'
  | 'pixelRatio';
type ToggleKey = 'buoyancyProbes' | 'wakeProbes' | 'forceWebGL';

interface SliderSpec {
  readonly key: SliderKey;
  readonly label: string;
  /** What moving it does, in words that need no graphics knowledge. Tooltip. */
  readonly tip: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  format(value: number): string;
}

interface SliderRefs {
  readonly spec: SliderSpec;
  readonly input: HTMLInputElement;
  readonly value: HTMLElement;
}

/**
 * Which sliders go in which section.
 *
 * Named, not sliced. This used to be `SLIDERS.slice(0, 3)` and `SLIDERS[3]`,
 * which silently stopped building Time of Day and Pixel Ratio when two sliders
 * were inserted ahead of them. `assertSlidersPlaced` makes every declared
 * slider appear in exactly one group, or construction throws.
 */
const SEA_SKY_SLIDERS = [
  'windSpeed',
  'peakWavelength',
  'cloudCoverage',
  'fogDensity',
  'timeOfDay',
] as const satisfies readonly SliderKey[];

const CABIN_SLIDERS = ['volume', 'musicVolume', 'pixelRatio'] as const satisfies readonly SliderKey[];

const SLIDERS: readonly SliderSpec[] = [
  {
    key: 'windSpeed',
    label: 'Wind',
    tip: 'How hard it blows. More wind raises bigger seas and more whitecaps',
    min: 0.5,
    max: 25,
    step: 0.5,
    // Knots alongside m/s: the HUD reads the wind in knots, and the ledger
    // should not disagree with it about the same number.
    format: (v) => `${v.toFixed(1)} m/s · ${Math.round(v * 1.943844)} kn`,
  },
  {
    key: 'peakWavelength',
    label: 'Swell length',
    tip: 'Distance between the big waves. Long swell rolls; short swell chops',
    min: 10,
    max: 150,
    step: 1,
    format: (v) => `${Math.round(v)} m`,
  },
  {
    key: 'cloudCoverage',
    label: 'Cloud cover',
    tip: 'How much of the sky is cloud',
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${Math.round(v * 100)}%`,
  },
  {
    key: 'fogDensity',
    label: 'Sea fog',
    tip: 'Haze over the water. Clear shows the far horizon',
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => (v === 0 ? 'Clear' : `${Math.round(v * 100)}%`),
  },
  {
    key: 'timeOfDay',
    label: 'Time of day',
    tip: 'Moves the sun, or the moon. Each look keeps its own hour until you move this',
    min: 0,
    max: 24,
    step: 0.1,
    format: (v) => {
      const hours = Math.floor(v) % 24;
      const minutes = Math.round((v - Math.floor(v)) * 60);
      return `${String(hours).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
    },
  },
  {
    key: 'volume',
    label: 'Volume',
    tip: 'Everything you hear: sea, ship, gulls and music',
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => (v === 0 ? 'Muted' : `${Math.round(v * 100)}%`),
  },
  {
    key: 'musicVolume',
    label: 'Music',
    tip: 'The score, under the sea and the ship. Zero turns it off',
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => (v === 0 ? 'Off' : `${Math.round(v * 100)}%`),
  },
  {
    key: 'pixelRatio',
    label: 'Sharpness',
    tip: 'How many pixels are drawn. Lower runs smoother on a slow device; higher is crisper',
    min: 0.5,
    max: 2,
    step: 0.05,
    format: (v) => `${v.toFixed(2)}×`,
  },
];

const TOGGLES: readonly { key: ToggleKey; label: string }[] = [
  { key: 'buoyancyProbes', label: 'Show where the hull floats' },
  { key: 'wakeProbes', label: 'Show where the wake is sampled' },
  { key: 'forceWebGL', label: 'Draw with WebGL2' },
];

/**
 * "About the tech": the jargon, kept out of every label and collected here
 * for the visitor who wants it. Each line says the plain thing first and
 * names the technique after it. The README carries the long version.
 */
const ABOUT_TECH: readonly string[] = [
  'Every wave is worked out live on your graphics card, every frame: swell, chop and ripple, '
    + 'summed by a fast Fourier transform (FFT) from a measured ocean-wave spectrum (JONSWAP).',
  'Whitecaps form only where the surface really folds over, read from the wave field itself.',
  'The ship floats on the same waves you see: points along her hull sample the water and the '
    + 'forces are solved each step.',
  'Edges are smoothed over several frames (temporal anti-aliasing, TRAA), which is what keeps '
    + 'the sun glitter from sparkling.',
  'One set of shaders runs on WebGPU, and on WebGL2 where WebGPU is not available.',
];

const QUALITY_LABELS: Record<QualityTier, string> = {
  low: 'Low',
  medium: 'Med',
  high: 'High',
  ultra: 'Ultra',
  max: 'Max',
};

const QUALITY_TIP =
  'How much detail to draw. Lower runs smoother; if frames drop for a while, the sea steps down a tier on its own';

const QUALITY_NAMES: Record<QualityTier, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  ultra: 'Ultra',
  max: 'Max',
};

// Neutral, descriptive names. The preset *ids* are stable API; the labels
// describe the look rather than borrowing product or game titles.
const PRESET_LABELS: Record<PresetId, string> = {
  skyPro: 'Clear Day',
  arctic: 'Arctic',
  blackFlag: 'High Seas',
  dusk: 'Dusk',
  foggy: 'Foggy',
  moonlit: 'Moonlit',
  seaOfThieves: 'Tropical',
  storm: 'Storm',
  sunset: 'Sunset',
};

let instanceCounter = 0;

function sliderSpec(key: SliderKey): SliderSpec {
  const spec = SLIDERS.find((s) => s.key === key);
  if (spec === undefined) throw new Error(`Panel: no slider spec for "${key}"`);
  return spec;
}

/** Every declared slider must be placed exactly once. */
function assertSlidersPlaced(): void {
  const placed = [...SEA_SKY_SLIDERS, ...CABIN_SLIDERS] as readonly SliderKey[];
  for (const spec of SLIDERS) {
    const count = placed.filter((key) => key === spec.key).length;
    if (count !== 1) {
      throw new Error(
        `Panel: slider "${spec.key}" is placed ${count} times; it must be placed exactly once`,
      );
    }
  }
  for (const key of placed) {
    if (!SLIDERS.some((spec) => spec.key === key)) {
      throw new Error(`Panel: slider "${key}" is placed but has no spec`);
    }
  }
}

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

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

export class Panel {
  private readonly root: HTMLElement;
  private readonly callbacks: UiCallbacks;
  private readonly controller = new AbortController();
  private readonly state: UiState;
  private readonly sliders = new Map<SliderKey, SliderRefs>();
  private readonly toggles = new Map<ToggleKey, HTMLButtonElement>();
  private readonly qualityInputs = new Map<QualityTier, HTMLInputElement>();
  private readonly presetInputs = new Map<PresetId, HTMLInputElement>();
  private readonly panelEl: HTMLElement;
  private readonly bodyEl: HTMLElement;
  private readonly collapseButton: HTMLButtonElement;
  private readonly sheetButton: HTMLButtonElement;
  private disposed = false;
  private sheetOpen = false;
  private collapsed = false;

  constructor(root: HTMLElement, initial: UiState, callbacks: UiCallbacks) {
    this.root = root;
    this.callbacks = callbacks;
    this.state = { ...initial };

    assertSlidersPlaced();

    const uid = `panel${++instanceCounter}`;
    const signal = this.controller.signal;

    const panel = el('section', 'panel parchment');
    panel.setAttribute('aria-label', 'Ocean controls');
    panel.id = `${uid}-sheet`;

    // ---- Title block -------------------------------------------------------
    const head = el('header', 'panel__head');
    const title = el('h1', 'panel__title', 'Web Ocean 3D');
    const collapse = el('button', 'panel__collapse');
    collapse.type = 'button';
    collapse.setAttribute('aria-expanded', 'true');
    collapse.setAttribute('aria-controls', `${uid}-body`);
    collapse.setAttribute('aria-label', 'Fold the ledger');
    collapse.append(icon('chevron'));
    collapse.addEventListener('click', () => this.onCollapseClick(), { signal });
    head.append(title, collapse);
    panel.append(head);

    const body = el('div', 'panel__body');
    body.id = `${uid}-body`;
    panel.append(body);

    // ---- The look ----------------------------------------------------------
    const look = this.section(body, 'The Look');
    const presets = el('div', 'presets');
    presets.setAttribute('role', 'radiogroup');
    presets.setAttribute('aria-label', 'Preset');
    for (const id of PRESET_IDS) {
      const input = this.radio(presets, `${uid}-preset`, id, id === this.state.preset, 'preset');
      input.setAttribute('aria-label', PRESET_LABELS[id]);
      const card = input.parentElement as HTMLElement;
      card.dataset.preset = id;
      card.append(el('span', 'preset__art'), el('span', 'preset__name', PRESET_LABELS[id]));
      input.addEventListener(
        'change',
        () => {
          if (!input.checked) return;
          this.state.preset = id;
          this.callbacks.onChange('preset', id);
        },
        { signal },
      );
      this.presetInputs.set(id, input);
    }
    look.append(presets);

    const qualityField = el('div', 'field');
    const qualityLabel = el('span', 'field__label', 'Quality');
    qualityLabel.id = `${uid}-quality-label`;
    qualityField.title = QUALITY_TIP;
    const quality = el('div', 'plaques');
    quality.setAttribute('role', 'radiogroup');
    quality.setAttribute('aria-labelledby', qualityLabel.id);
    for (const tier of QUALITY_TIERS) {
      const input = this.radio(quality, `${uid}-quality`, tier, tier === this.state.quality, 'plaque');
      input.setAttribute('aria-label', QUALITY_NAMES[tier]);
      (input.parentElement as HTMLElement).append(el('span', 'plaque__name', QUALITY_LABELS[tier]));
      input.addEventListener(
        'change',
        () => {
          if (!input.checked) return;
          this.state.quality = tier;
          this.callbacks.onChange('quality', tier);
        },
        { signal },
      );
      this.qualityInputs.set(tier, input);
    }
    qualityField.append(qualityLabel, quality);
    look.append(qualityField);

    // ---- Sea & sky ---------------------------------------------------------
    const seaSky = this.section(body, 'Weather & Water');
    for (const key of SEA_SKY_SLIDERS) {
      this.buildSlider(seaSky, `${uid}-${key}`, sliderSpec(key));
    }

    // ---- Cabin -------------------------------------------------------------
    const cabin = this.section(body, 'Sound & Screen');
    for (const key of CABIN_SLIDERS) {
      this.buildSlider(cabin, `${uid}-${key}`, sliderSpec(key));
    }

    // ---- Shipwright (debug) ------------------------------------------------
    const tools = el('details', 'ledger-tools');
    const summary = el('summary', 'ledger-tools__summary');
    summary.append(el('span', undefined, "Shipwright's tools"), icon('chevron', 'ledger-tools__chev'));
    tools.append(summary);
    for (const toggle of TOGGLES) {
      this.buildToggle(tools, toggle.key, toggle.label);
    }
    body.append(tools);

    // ---- About the tech ----------------------------------------------------
    // The same fold as the tools: the jargon is here for whoever opens it and
    // out of the way of everyone else.
    const about = el('details', 'ledger-tools ledger-about');
    const aboutSummary = el('summary', 'ledger-tools__summary');
    aboutSummary.append(el('span', undefined, 'About the tech'), icon('chevron', 'ledger-tools__chev'));
    about.append(aboutSummary);
    for (const line of ABOUT_TECH) about.append(el('p', 'ledger-about__line', line));
    body.append(about);

    // ---- Actions -----------------------------------------------------------
    const actions = el('div', 'panel__actions');
    const source = el('a', 'btn btn--brass', 'View Source');
    source.title = 'The code, on GitHub';
    source.href = SOURCE_URL;
    source.rel = 'noopener';
    source.target = '_blank';
    actions.append(source);
    body.append(actions);

    // ---- Mobile bottom-sheet trigger --------------------------------------
    const sheetButton = el('button', 'panel-toggle');
    sheetButton.type = 'button';
    sheetButton.setAttribute('aria-controls', panel.id);
    sheetButton.setAttribute('aria-expanded', 'false');
    sheetButton.setAttribute('aria-label', 'Ledger');
    sheetButton.append(icon('ledger', 'panel-toggle__icon'), el('span', 'panel-toggle__text', 'Ledger'));
    sheetButton.addEventListener('click', () => this.setSheetOpen(!this.sheetOpen), { signal });

    this.panelEl = panel;
    this.bodyEl = body;
    this.collapseButton = collapse;
    this.sheetButton = sheetButton;
    this.root.append(panel, sheetButton);

    // Folded by default. The sea is the product, and an open ledger is a
    // third of a laptop screen; the title plate says where the controls are,
    // and one click opens them.
    this.setCollapsed(true);
  }

  /** Opens or folds the ledger (desktop) — for a harness or a host shortcut. */
  setOpen(open: boolean): void {
    if (this.disposed) return;
    this.setCollapsed(!open);
  }

  /** Push values in from the host without re-emitting change callbacks. */
  setState(partial: Partial<UiState>): void {
    if (this.disposed) return;

    if (partial.quality !== undefined) {
      this.state.quality = partial.quality;
      const input = this.qualityInputs.get(partial.quality);
      if (input) input.checked = true;
    }
    if (partial.preset !== undefined) {
      this.state.preset = partial.preset;
      const input = this.presetInputs.get(partial.preset);
      if (input) input.checked = true;
    }
    if (partial.cameraMode !== undefined) {
      this.state.cameraMode = partial.cameraMode;
    }

    for (const [key, refs] of this.sliders) {
      const next = partial[key];
      if (next === undefined) continue;
      // `timeOfDay` is nullable — null means the preset still owns the sun.
      if (next === null) continue;
      const value = clamp(next, refs.spec.min, refs.spec.max);
      this.state[key] = value;
      refs.input.value = String(value);
      this.paintSlider(refs);
    }

    for (const [key, button] of this.toggles) {
      const next = partial[key];
      if (next === undefined) continue;
      this.state[key] = next;
      button.setAttribute('aria-checked', next ? 'true' : 'false');
      button.classList.toggle('is-on', next);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.controller.abort();
    this.panelEl.remove();
    this.sheetButton.remove();
    this.sliders.clear();
    this.toggles.clear();
    this.qualityInputs.clear();
    this.presetInputs.clear();
  }

  // -- builders -------------------------------------------------------------

  private section(parent: HTMLElement, title: string): HTMLElement {
    const section = el('section', 'ledger-section');
    const heading = el('h2', 'ledger-section__title', title);
    section.append(heading);
    parent.append(section);
    return section;
  }

  /** A visually-hidden native radio inside its own label. */
  private radio(
    parent: HTMLElement,
    name: string,
    value: string,
    checked: boolean,
    className: string,
  ): HTMLInputElement {
    const label = el('label', className);
    const input = el('input', 'sr-radio');
    input.type = 'radio';
    input.name = name;
    input.value = value;
    input.checked = checked;
    label.append(input);
    parent.append(label);
    return input;
  }

  private buildSlider(parent: HTMLElement, id: string, spec: SliderSpec): void {
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
    input.value = String(this.state[spec.key]);

    const refs: SliderRefs = { spec, input, value };
    this.sliders.set(spec.key, refs);
    this.paintSlider(refs);

    input.addEventListener(
      'input',
      () => {
        const next = clamp(Number(input.value), spec.min, spec.max);
        this.state[spec.key] = next;
        this.paintSlider(refs);
        this.callbacks.onChange(spec.key, next);
      },
      { signal: this.controller.signal },
    );

    field.append(row, input);
    parent.append(field);
  }

  private buildToggle(parent: HTMLElement, key: ToggleKey, labelText: string): void {
    const button = el('button', 'switchrow');
    button.type = 'button';
    button.setAttribute('role', 'switch');
    const on = this.state[key];
    button.setAttribute('aria-checked', on ? 'true' : 'false');
    button.classList.toggle('is-on', on);

    const box = el('span', 'switch');
    box.append(el('span', 'switch__knob'));
    button.append(box, el('span', 'switchrow__label', labelText));

    button.addEventListener(
      'click',
      () => {
        const next = !this.state[key];
        this.state[key] = next;
        button.setAttribute('aria-checked', next ? 'true' : 'false');
        button.classList.toggle('is-on', next);
        this.callbacks.onChange(key, next);
      },
      { signal: this.controller.signal },
    );

    this.toggles.set(key, button);
    parent.append(button);
  }

  // -- helpers --------------------------------------------------------------

  private paintSlider(refs: SliderRefs): void {
    const { spec, input, value } = refs;
    const current = Number(input.value);
    const pct = ((current - spec.min) / (spec.max - spec.min)) * 100;
    input.style.setProperty('--fill', `${clamp(pct, 0, 100).toFixed(2)}%`);
    value.textContent = spec.format(current);
    input.setAttribute('aria-valuetext', spec.format(current));
  }

  /**
   * The head button folds the ledger on a wide screen and closes the sheet on
   * a narrow one — the same control, doing the same job, "put this away".
   */
  private onCollapseClick(): void {
    if (this.sheetOpen && window.matchMedia('(max-width: 900px)').matches) {
      this.setSheetOpen(false);
      this.sheetButton.focus();
      return;
    }
    this.setCollapsed(!this.collapsed);
  }

  private setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    this.panelEl.classList.toggle('panel--collapsed', collapsed);
    this.bodyEl.hidden = collapsed;
    this.collapseButton.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    this.collapseButton.setAttribute('aria-label', collapsed ? 'Unfold the ledger' : 'Fold the ledger');
  }

  private setSheetOpen(open: boolean): void {
    this.sheetOpen = open;
    // Opening the sheet always shows its contents, whatever it was folded to.
    if (open && this.collapsed) this.setCollapsed(false);
    this.panelEl.classList.toggle('panel--open', open);
    this.sheetButton.setAttribute('aria-expanded', open ? 'true' : 'false');
    this.sheetButton.classList.toggle('is-active', open);
  }
}
