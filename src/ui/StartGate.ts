import { icon } from './icons.ts';
import { MenuSea } from './MenuSea.ts';
import {
  QUALITY_TIERS,
  type QualityTier,
  type RendererBackend,
} from './types.ts';

const STORAGE_KEY = 'web-ocean:start-selection:v1';

interface StoredSelection {
  backend: RendererBackend;
  quality: QualityTier;
}

interface DeviceProbe {
  adapter: GPUAdapter | null;
  vendor: string;
  hardwareConcurrency: number;
  deviceMemory: number | null;
  coarsePointer: boolean;
}

export interface StartSelection {
  quality: QualityTier;
  backend: RendererBackend;
  forceWebGL: boolean;
  /**
   * The adapter metadata probe result. No device is created by the gate.
   *
   * Three-valued on purpose, and `Renderer` reads all three: an adapter means
   * "use this one", `null` means "probed, and there is none — take WebGL2", and
   * `undefined` means "not probed, decide for yourself". The tooling bypass
   * returns `undefined` because it skips the probe; returning `null` there
   * silently forced every bypassed launch onto WebGL2.
   */
  adapter: GPUAdapter | null | undefined;
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
  low: 'lightest',
  medium: 'balanced',
  high: 'rich',
  ultra: 'lavish',
  max: 'everything',
};

function isQualityTier(value: string | null): value is QualityTier {
  return value !== null && QUALITY_TIERS.includes(value as QualityTier);
}

function isBackend(value: unknown): value is RendererBackend {
  return value === 'webgpu' || value === 'webgl';
}

function readStoredSelection(): StoredSelection | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredSelection>;
    const quality = parsed.quality ?? null;
    if (!isBackend(parsed.backend) || !isQualityTier(quality)) return null;
    return { backend: parsed.backend, quality };
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
  if (!probe.adapter) return { backend: 'webgl', quality: 'low' };

  const constrained = (probe.deviceMemory !== null && probe.deviceMemory <= 4)
    || (probe.hardwareConcurrency > 0 && probe.hardwareConcurrency <= 4);
  if (constrained) return { backend: 'webgpu', quality: 'low' };
  if (probe.coarsePointer) return { backend: 'webgpu', quality: 'medium' };
  if (isIntegratedVendor(probe.vendor)) return { backend: 'webgpu', quality: 'medium' };
  return { backend: 'webgpu', quality: 'high' };
}

function describeProbe(probe: DeviceProbe): string {
  const memory = probe.deviceMemory === null
    ? 'memory n/a'
    : String(probe.deviceMemory) + ' GB RAM hint';
  const cores = probe.hardwareConcurrency > 0
    ? String(probe.hardwareConcurrency) + ' threads'
    : 'threads n/a';
  if (!probe.adapter) return 'WebGPU unavailable · WebGL2 ready · ' + cores + ' · ' + memory;
  const vendor = probe.vendor || 'unreported adapter';
  return 'WebGPU ' + vendor + ' · ' + cores + ' · ' + memory;
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
): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'boot__choice';
  button.classList.toggle('is-selected', selected);
  button.setAttribute('aria-label', label);
  button.setAttribute('aria-pressed', selected ? 'true' : 'false');
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
  if (status) status.textContent = 'Choose a profile';
  if (bar) bar.style.transform = 'scaleX(0)';

  let selectedBackend = initial.backend;
  let selectedQuality = initial.quality;

  const intro = document.createElement('p');
  intro.className = 'boot__gate-intro';
  intro.textContent = 'Choose how this device should render the sea. Both can be changed later from the ledger.';

  const backendLabel = document.createElement('div');
  backendLabel.className = 'boot__gate-label';
  backendLabel.textContent = 'Renderer';
  const backendGroup = document.createElement('div');
  backendGroup.className = 'boot__choices boot__choices--backend';
  backendGroup.setAttribute('role', 'group');
  backendGroup.setAttribute('aria-label', 'Renderer backend');
  // Without an adapter WebGPU is not a choice, and saying so plainly is the
  // gate's error state: the plaque is struck through and a note says what the
  // fallback costs, rather than letting the visitor pick a backend that would
  // silently become WebGL2 anyway.
  const webgpuAvailable = probe.adapter !== null;
  if (!webgpuAvailable) selectedBackend = 'webgl';
  const backendButtons = [
    choiceButton('WebGPU', webgpuAvailable ? 'full fidelity' : 'unavailable', selectedBackend === 'webgpu', () => {
      selectedBackend = 'webgpu';
      markSelected(backendButtons, 'WebGPU');
    }, advised.backend === 'webgpu'),
    choiceButton('WebGL2', 'compatibility', selectedBackend === 'webgl', () => {
      selectedBackend = 'webgl';
      markSelected(backendButtons, 'WebGL2');
    }, advised.backend === 'webgl'),
  ];
  backendGroup.append(...backendButtons);
  const webgpuButton = backendButtons[0];
  let notice: HTMLElement | null = null;
  if (!webgpuAvailable && webgpuButton) {
    webgpuButton.disabled = true;
    webgpuButton.classList.add('is-unavailable');
    webgpuButton.setAttribute('aria-describedby', 'boot-notice');
    notice = document.createElement('p');
    notice.className = 'boot__notice';
    notice.id = 'boot-notice';
    notice.setAttribute('role', 'note');
    notice.append(
      icon('warning', 'boot__notice-icon'),
      document.createTextNode(
        'This browser does not offer WebGPU, so the sea will be drawn with WebGL2 — '
          + 'the same waves, without planar reflections, refraction or lens rain.',
      ),
    );
  }

  const qualityLabel = document.createElement('div');
  qualityLabel.className = 'boot__gate-label';
  qualityLabel.textContent = 'Quality';
  const qualityGroup = document.createElement('div');
  qualityGroup.className = 'boot__choices boot__choices--quality';
  qualityGroup.setAttribute('role', 'group');
  qualityGroup.setAttribute('aria-label', 'Quality tier');
  const qualityButtons = QUALITY_TIERS.map((tier) =>
    choiceButton(QUALITY_LABELS[tier], QUALITY_DETAILS[tier], selectedQuality === tier, () => {
      selectedQuality = tier;
      markSelected(qualityButtons, QUALITY_LABELS[tier]);
    }, advised.quality === tier),
  );
  qualityGroup.append(...qualityButtons);

  const probeReadout = document.createElement('p');
  probeReadout.className = 'boot__gate-readout';
  probeReadout.textContent = describeProbe(probe);

  const start = document.createElement('button');
  start.type = 'button';
  start.className = 'cta boot__start';
  const startLabel = document.createElement('span');
  startLabel.className = 'cta__label';
  startLabel.textContent = 'Set Sail';
  start.append(icon('anchor', 'cta__icon'), startLabel);

  gate.replaceChildren(
    intro,
    backendLabel,
    backendGroup,
    ...(notice ? [notice] : []),
    qualityLabel,
    qualityGroup,
    probeReadout,
    start,
  );

  return new Promise((resolve) => {
    start.addEventListener('click', () => {
      start.disabled = true;
      const selection = {
        backend: selectedBackend,
        quality: selectedQuality,
      } satisfies StoredSelection;
      writeStoredSelection(selection);
      showProgress(root);
      resolve({
        ...selection,
        forceWebGL: selection.backend === 'webgl',
        adapter: probe.adapter,
      });
    }, { once: true });
  });
}

/**
 * Shows the pre-load gate, or resolves immediately for the explicit tooling
 * bypass. The adapter probe only asks for metadata and never creates a device.
 */
export async function openStartGate(root: HTMLElement): Promise<StartSelection> {
  const params = new URL(window.location.href).searchParams;
  const stored = readStoredSelection();
  const requestedQuality = isQualityTier(params.get('quality'))
    ? params.get('quality') as QualityTier
    : null;

  if (params.get('gate') === '0') {
    const backend: RendererBackend = params.get('webgl') === '1' ? 'webgl' : 'webgpu';
    // Tooling already declares the backend and tier. Do not pay the adapter
    // metadata round trip before a bypassed boot; Renderer will perform its
    // normal backend probe, while a visible gate still gets the cheap probe
    // below before presenting the device recommendation.
    //
    // `undefined`, emphatically not `null`. Renderer reads null as "already
    // probed, and there is no adapter" and takes the WebGL2 fallback on the
    // strength of it. Returning null here therefore forced every `gate=0`
    // launch onto WebGL2 — including every benchmark and profile run nominally
    // measuring WebGPU.
    const quality = requestedQuality ?? stored?.quality ?? 'high';
    dressBoot(root);
    showProgress(root);
    return { quality, backend, forceWebGL: backend === 'webgl', adapter: undefined };
  }

  dressBoot(root);
  const probe = await probeDevice();
  const defaults = defaultSelection(probe);
  const initial: StoredSelection = {
    backend: params.get('webgl') === '1' ? 'webgl' : stored?.backend ?? defaults.backend,
    quality: requestedQuality ?? stored?.quality ?? defaults.quality,
  };
  return renderGate(root, probe, initial, defaults);
}
