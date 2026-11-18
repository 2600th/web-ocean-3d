import { icon, type IconName } from './icons.ts';
import { startLoadingTips } from './LoadingTips.ts';
import { MenuSea } from './MenuSea.ts';
import { installReducedMotionOverride } from './motion.ts';
import { applyDisplayPrefs, loadSettingsPrefs } from './prefs.ts';
import { QUALITY_TIERS, type QualityTier } from './types.ts';
import { readVoyageRecord, type VoyageRecord } from './voyageRecord.ts';

const STORAGE_KEY = 'web-ocean:start-selection:v1';

interface StoredSelection {
  quality: QualityTier;
}

interface DeviceProbe {
  adapter: GPUAdapter | null;
  vendor: string;
  hardwareConcurrency: number;
  deviceMemory: number | null;
  coarsePointer: boolean;
}

/** Which way the player left the title screen: back to their voyage, or out on a new one. */
export type VoyageChoice = 'continue' | 'new';

export interface StartSelection {
  quality: QualityTier;
  /**
   * The title screen's choice. The tooling bypass (`?gate=0`) is always a new
   * voyage unless it says `?voyage=continue` and a record exists, so a saved
   * game on a developer's machine cannot change what a test sees.
   */
  voyage: VoyageChoice;
  /**
   * The adapter metadata probe result. No device is created by the gate.
   * `undefined` means "not probed" (the tooling bypass), and `Renderer` probes
   * for itself. The gate never resolves without an adapter: a browser with no
   * WebGPU is told what it needs instead.
   */
  adapter: GPUAdapter | undefined;
}

const QUALITY_LABELS: Record<QualityTier, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  ultra: 'Ultra',
  max: 'Max',
};

/** What each tier costs, in a word a visitor can choose by. */
const QUALITY_DETAILS: Record<QualityTier, string> = {
  low: 'smoothest',
  medium: 'balanced',
  high: 'rich',
  ultra: 'lavish',
  max: 'everything',
};

/** The same, as a sentence, for the plaque's tooltip. */
const QUALITY_TIPS: Record<QualityTier, string> = {
  low: 'Lightest on the device. For phones and older laptops',
  medium: 'A good balance for most laptops',
  high: 'Rich detail for a desktop graphics card',
  ultra: 'Lavish. For a fast graphics card',
  max: 'Everything at once. For the fastest graphics cards',
};

function isQualityTier(value: string | null): value is QualityTier {
  return value !== null && QUALITY_TIERS.includes(value as QualityTier);
}

function readStoredSelection(): StoredSelection | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredSelection>;
    const quality = parsed.quality ?? null;
    if (!isQualityTier(quality)) return null;
    return { quality };
  } catch {
    return null;
  }
}

function writeStoredSelection(selection: StoredSelection): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // Private browsing and locked-down embeds can reject localStorage. The gate
    // remains useful as a session choice when persistence is unavailable.
  }
}

async function probeDevice(): Promise<DeviceProbe> {
  let adapter: GPUAdapter | null = null;
  try {
    if (navigator.gpu) {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    }
  } catch {
    adapter = null;
  }

  const info = adapter?.info;
  const extendedNavigator = navigator as Navigator & { deviceMemory?: number };
  return {
    adapter,
    vendor: info?.vendor ?? '',
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
    deviceMemory: typeof extendedNavigator.deviceMemory === 'number'
      ? extendedNavigator.deviceMemory
      : null,
    coarsePointer: typeof window.matchMedia === 'function'
      && window.matchMedia('(pointer: coarse)').matches,
  };
}

function isIntegratedVendor(vendor: string): boolean {
  return /intel|qualcomm|arm|imagination/i.test(vendor);
}

function defaultSelection(probe: DeviceProbe): StoredSelection {
  const constrained = (probe.deviceMemory !== null && probe.deviceMemory <= 4)
    || (probe.hardwareConcurrency > 0 && probe.hardwareConcurrency <= 4);
  if (constrained) return { quality: 'low' };
  if (probe.coarsePointer) return { quality: 'medium' };
  if (isIntegratedVendor(probe.vendor)) return { quality: 'medium' };
  return { quality: 'high' };
}

/**
 * What the probe found, as a plain readout. Instrumentation, not voice: it
 * says only what the browser reported, and says so when it reported nothing.
 */
function describeProbe(probe: DeviceProbe): string {
  // `deviceMemory` is rounded and capped by the browser (8 GB in Chrome), so
  // it is quoted as "at least".
  const memory = probe.deviceMemory === null
    ? 'memory not reported'
    : String(probe.deviceMemory) + ' GB+ memory';
  const cores = probe.hardwareConcurrency > 0
    ? String(probe.hardwareConcurrency) + ' CPU threads'
    : 'CPU not reported';
  const vendor = probe.vendor ? probe.vendor + ' graphics' : 'graphics not named';
  return 'WebGPU on ' + vendor + ' · ' + cores + ' · ' + memory;
}

/**
 * A brass-framed plaque. Selected, its face is polished brass with the name
 * engraved in ink; otherwise it is dark wood with the name inlaid in brass.
 * `advised` adds the small wax tag that marks what the probe recommends.
 */
function choiceButton(
  label: string,
  detail: string,
  selected: boolean,
  onSelect: () => void,
  advised = false,
  tip = '',
): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'boot__choice';
  button.classList.toggle('is-selected', selected);
  button.setAttribute('aria-label', label);
  button.setAttribute('aria-pressed', selected ? 'true' : 'false');
  if (tip) button.title = tip;
  const name = document.createElement('span');
  name.className = 'boot__choice-name';
  name.textContent = label;
  const caption = document.createElement('small');
  caption.className = 'boot__choice-detail';
  caption.textContent = detail;
  button.append(name, caption);
  if (advised) {
    const tag = document.createElement('span');
    tag.className = 'boot__advised';
    tag.textContent = 'advised';
    button.append(tag);
  }
  button.addEventListener('click', onSelect);
  return button;
}

function markSelected(buttons: readonly HTMLButtonElement[], selected: string): void {
  for (const button of buttons) {
    const active = button.getAttribute('aria-label') === selected;
    button.classList.toggle('is-selected', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

function showProgress(root: HTMLElement): void {
  const gate = root.querySelector<HTMLElement>('#boot-gate');
  const progress = root.querySelector<HTMLElement>('#boot-progress');
  if (gate) gate.hidden = true;
  if (progress) progress.hidden = false;
  // The sheet shrinks from the menu to the loading chart; CSS animates the
  // change of content rather than cutting to it.
  root.classList.add('boot--loading');
  startLoadingTips(root);
}

/**
 * The painted seascape behind the gate and the loading chart, and the retry
 * control the failure state shows. Neither may stop the boot: a browser
 * without a 2D canvas still gets the gate on its plain background.
 */
function dressBoot(root: HTMLElement): void {
  try {
    new MenuSea(root).start();
  } catch {
    // No 2D context: the gradient behind the sheet stands in for the sea.
  }
  const retry = root.querySelector<HTMLButtonElement>('#boot-retry');
  retry?.addEventListener('click', () => window.location.reload(), { once: true });
}

/**
 * The page for a browser without WebGPU: what it needs, and no way forward that
 * would end on a dead canvas. Exported for the boot, which shows the same page
 * when an adapter was found but the device would not open.
 */
export function showWebGPURequired(root: HTMLElement, detail = ''): void {
  const gate = root.querySelector<HTMLElement>('#boot-gate');
  const progress = root.querySelector<HTMLElement>('#boot-progress');
  const status = root.querySelector<HTMLElement>('#boot-status');
  root.classList.remove('boot--loading');
  if (progress) progress.hidden = true;
  if (status) status.textContent = 'WebGPU required';
  if (!gate) return;
  gate.hidden = false;
  const notice = document.createElement('p');
  notice.className = 'boot__notice';
  notice.id = 'boot-notice';
  notice.setAttribute('role', 'alert');
  notice.append(
    icon('warning', 'boot__notice-icon'),
    document.createTextNode(
      'This sea is drawn with WebGPU, which this browser does not offer'
        + (detail ? ' (' + detail + ')' : '')
        + '. Open it in a current Chrome or Edge, Safari 26 or later, or Firefox 141 or later on Windows, '
        + 'with hardware acceleration switched on.',
    ),
  );
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'cta boot__start';
  const label = document.createElement('span');
  label.className = 'cta__label';
  label.textContent = 'Try again';
  retry.append(icon('anchor', 'cta__icon'), label);
  retry.addEventListener('click', () => window.location.reload(), { once: true });
  gate.replaceChildren(notice, retry);
}

/** A call-to-action: the wax one for the main choice, a quieter brass one beside it. */
function startButton(label: string, iconName: IconName, quiet: boolean): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = quiet ? 'cta cta--quiet boot__start' : 'cta boot__start';
  const text = document.createElement('span');
  text.className = 'cta__label';
  text.textContent = label;
  button.append(icon(iconName, 'cta__icon'), text);
  return button;
}

/** "Gold 120 · Voyages completed 3", from whatever the record says. Plain figures, not voice. */
function describeRecord(record: VoyageRecord): string {
  const parts: string[] = [];
  if (record.gold !== null) parts.push('Gold ' + record.gold.toLocaleString('en-GB'));
  if (record.voyages !== null) {
    parts.push('Voyages completed ' + record.voyages.toLocaleString('en-GB'));
  }
  return parts.join(' · ');
}

function renderGate(
  root: HTMLElement,
  probe: DeviceProbe,
  initial: StoredSelection,
  advised: StoredSelection,
): Promise<StartSelection> {
  const gate = root.querySelector<HTMLElement>('#boot-gate');
  if (!gate) throw new Error('Failed to start: page markup is missing #boot-gate.');

  const progress = root.querySelector<HTMLElement>('#boot-progress');
  const status = root.querySelector<HTMLElement>('#boot-status');
  const bar = root.querySelector<HTMLElement>('#boot-bar');
  gate.hidden = false;
  if (progress) progress.hidden = true;
  if (status) status.textContent = 'Choose your settings';
  if (bar) bar.style.transform = 'scaleX(0)';

  let selectedQuality = initial.quality;
  const record = readVoyageRecord();

  // What is on record, or that nothing is. Quiet writing under the title: the
  // choices below are the point of the page.
  const ledger = document.createElement('p');
  ledger.className = 'boot__record';
  if (record === null) {
    ledger.classList.add('is-empty');
    ledger.textContent = 'No voyage on record yet.';
  } else {
    const figures = describeRecord(record);
    ledger.textContent = figures === '' ? 'A voyage is on record.' : figures;
  }

  // Continue leads when there is something to continue; otherwise the new
  // voyage does, and there is only one way to go.
  const resume = record === null ? null : startButton('Continue', 'helm', false);
  const fresh = startButton('New voyage', record === null ? 'anchor' : 'mark', record !== null);
  const startGroup = document.createElement('div');
  startGroup.className = 'boot__starts';
  startGroup.append(...(resume === null ? [fresh] : [resume, fresh]));

  // The graphics choice stays, below and smaller: it is set once and the
  // advised tier is usually right.
  const qualityLabel = document.createElement('div');
  qualityLabel.className = 'boot__gate-label boot__gate-label--quiet';
  qualityLabel.textContent = 'Graphics';
  const qualityGroup = document.createElement('div');
  qualityGroup.className = 'boot__choices boot__choices--quality boot__choices--compact';
  qualityGroup.setAttribute('role', 'group');
  qualityGroup.setAttribute('aria-label', 'Quality tier');
  const qualityButtons = QUALITY_TIERS.map((tier) =>
    choiceButton(QUALITY_LABELS[tier], QUALITY_DETAILS[tier], selectedQuality === tier, () => {
      selectedQuality = tier;
      markSelected(qualityButtons, QUALITY_LABELS[tier]);
    }, advised.quality === tier, QUALITY_TIPS[tier]),
  );
  qualityGroup.append(...qualityButtons);

  const probeReadout = document.createElement('p');
  probeReadout.className = 'boot__gate-readout';
  probeReadout.textContent = describeProbe(probe);
  const adviceNote = document.createElement('p');
  adviceNote.className = 'boot__gate-readout boot__gate-readout--note';
  adviceNote.textContent = 'The wax tag marks what suits this device. It can be changed later from the ledger.';

  gate.replaceChildren(ledger, startGroup, qualityLabel, qualityGroup, adviceNote, probeReadout);

  return new Promise((resolve) => {
    const go = (voyage: VoyageChoice): void => {
      for (const button of [resume, fresh, ...qualityButtons]) if (button) button.disabled = true;
      const selection = { quality: selectedQuality } satisfies StoredSelection;
      writeStoredSelection(selection);
      showProgress(root);
      resolve({ ...selection, voyage, adapter: probe.adapter ?? undefined });
    };
    resume?.addEventListener('click', () => go('continue'), { once: true });
    fresh.addEventListener('click', () => go('new'), { once: true });
    // The first control takes focus, so a keyboard needs no Tab to begin.
    if (window.matchMedia('(pointer: fine)').matches) (resume ?? fresh).focus({ preventScroll: true });
  });
}

/**
 * Shows the pre-load gate, or resolves immediately for the explicit tooling
 * bypass. The adapter probe only asks for metadata and never creates a device.
 */
export async function openStartGate(root: HTMLElement): Promise<StartSelection> {
  // The player's motion and display settings apply to the menu as well, before
  // anything asks the system's reduced-motion query (the menu sea does).
  installReducedMotionOverride();
  applyDisplayPrefs(loadSettingsPrefs());

  const params = new URL(window.location.href).searchParams;
  const stored = readStoredSelection();
  const requestedQuality = isQualityTier(params.get('quality'))
    ? params.get('quality') as QualityTier
    : null;

  if (params.get('gate') === '0') {
    // Tooling already declares the tier. Do not pay the adapter metadata round
    // trip before a bypassed boot; Renderer probes for itself.
    const quality = requestedQuality ?? stored?.quality ?? 'high';
    dressBoot(root);
    showProgress(root);
    const voyage: VoyageChoice =
      params.get('voyage') === 'continue' && readVoyageRecord() !== null ? 'continue' : 'new';
    return { quality, voyage, adapter: undefined };
  }

  dressBoot(root);
  const probe = await probeDevice();
  if (!probe.adapter) {
    showWebGPURequired(root);
    // Nothing to start: the page stays on the notice.
    return new Promise<StartSelection>(() => undefined);
  }
  const defaults = defaultSelection(probe);
  const initial: StoredSelection = {
    quality: requestedQuality ?? stored?.quality ?? defaults.quality,
  };
  return renderGate(root, probe, initial, defaults);
}
