/**
 * The heads-up display: the ship's instruments, laid over the sea.
 *
 *   top centre    compass tape — the bearing the camera looks along, with the
 *                 objective and (at the helm) the ship's head marked on it
 *   under it      toasts, unrolled as strips of chart paper
 *   top left      the objective, a sealed scroll, and the score plaque
 *   bottom left   wind dial (a turning compass card with the wind's arrow on it)
 *                 and, at the helm, the log — speed through the water in knots
 *   bottom centre camera-mode tabs with their keys, and the hints for the mode
 *   bottom right  the instrument corner: frame rate and renderer backend
 *
 * Pure DOM, no three.js. Every value the host feeds per frame is diffed and the
 * text writes are throttled — the frame-rate figure to ~4 Hz, as before, so a
 * 120 Hz loop cannot force 120 UI repaints a second. The compass tape and dial
 * card move by `transform` only, which is composited and never lays out.
 *
 * The whole `.hud` is a full-viewport, click-through layer; only the mode tabs
 * opt back into hit-testing, so nothing here can swallow a camera drag.
 */

import { keybinds, type KeyAction } from '../input/keybinds.ts';
import { Coach, type InputKind, type LessonId } from './Coach.ts';
import { compassRose, icon, type IconName } from './icons.ts';
import {
  CAMERA_MODES,
  type CameraMode,
  type NavReadout,
  type Objective,
  type RendererBackend,
  type ToastKind,
  type ToastOptions,
  type UiCallbacks,
  type VoyageTally,
} from './types.ts';

interface Hint {
  readonly action: string;
  readonly key: string;
}

const MODE_LABELS: Record<CameraMode, string> = {
  orbit: 'Orbit',
  fly: 'Fly',
  boat: 'Helm',
  cinematic: 'Tour',
};

/** The key bound to each mode, which the player may have changed. */
const MODE_ACTIONS: Record<CameraMode, KeyAction> = {
  orbit: 'modeOrbit',
  fly: 'modeFly',
  boat: 'modeHelm',
  cinematic: 'modeTour',
};

function modeKey(mode: CameraMode): string {
  return keybinds.labelOf(MODE_ACTIONS[mode]);
}

/** What each mode is for, in a sentence a first-time visitor can choose by. Tab tooltips. */
const MODE_TIPS: Record<CameraMode, string> = {
  orbit: 'Circle the ship and the island',
  fly: 'Fly anywhere, above the waves or under them',
  boat: 'Take the wheel and sail her yourself',
  cinematic: 'Sit back for a guided voyage through a day and a squall',
};

const MODE_ICONS: Record<CameraMode, IconName> = {
  orbit: 'orbit',
  fly: 'fly',
  boat: 'helm',
  cinematic: 'tour',
};

/** The key list for a mode. The helm's and the tour's follow the player's bindings. */
function hintsFor(mode: CameraMode): readonly Hint[] {
  switch (mode) {
    case 'orbit':
      return [
        { action: 'Look around', key: 'Drag' },
        { action: 'Slide', key: 'Right drag' },
        { action: 'Zoom', key: 'Scroll' },
      ];
    case 'fly':
      return [
        { action: 'Fly', key: 'WASD' },
        { action: 'Look', key: 'Click, then mouse' },
        { action: 'Set speed', key: 'Scroll' },
        { action: 'Boost', key: 'Shift' },
      ];
    // More and less sail step the canvas (furled, half, full) and the anchor
    // key lets go or weighs it; see `src/gameplay`.
    case 'boat':
      return [
        { action: 'More or less sail', key: keybinds.labelsOf('sailMore', 'sailLess') },
        { action: 'Steer', key: keybinds.labelsOf('steerPort', 'steerStarboard') },
        { action: 'Trim sails', key: keybinds.labelsOf('trimIn', 'trimOut') },
        { action: 'Anchor', key: keybinds.labelOf('anchor') },
        { action: 'Look', key: 'Drag' },
      ];
    // Nothing to drive: the tour holds the wheel. The hint names what the
    // viewer *can* do, which is leave.
    case 'cinematic':
      return [
        { action: 'Take the helm', key: modeKey('boat') },
        { action: 'Fly free', key: modeKey('fly') },
      ];
  }
}

/** The helm's hints once a gamepad is in use (standard mapping, Xbox names). */
const PAD_BOAT_HINTS: readonly Hint[] = [
  { action: 'More or less sail', key: 'RT LT' },
  { action: 'Steer', key: 'Left stick' },
  { action: 'Trim sails', key: 'LB RB' },
  { action: 'Anchor', key: 'A' },
  { action: 'Look', key: 'Right stick' },
];

/**
 * Half-width of the compass's no-go arc, degrees either side of the wind:
 * `IRONS_DEG` in `gameplay/sailing.ts`, repeated because `src/ui` stands alone.
 */
const NO_GO_DEG = 35;

/** Minimum ms between FPS text writes (~4 per second). */
const FPS_WRITE_INTERVAL = 250;
/** Minimum ms between heading / speed / distance text writes. */
const NAV_WRITE_INTERVAL = 120;

/** Pixels of compass tape per degree. */
const TAPE_PX_PER_DEG = 2.6;
/** Tape spans this many degrees either side of 0..360 so it never runs out. */
const TAPE_PAD_DEG = 180;

const TOAST_LIMIT = 4;

/**
 * Attention model. AAA HUDs earn their pixels by not keeping them: anything
 * the viewer is not using fades back so the sea fills the frame.
 *
 *  - IDLE_MS without input and the non-critical chrome (compass, dial, mode
 *    bar, instrument corner) fades to a low opacity: `.is-idle` on #ui-root;
 *  - while the camera is being dragged it drops further: `.is-dragging`;
 *  - key hints show for HINT_MS after a mode is entered, then go;
 *  - any input, a new objective, a toast, a score change or a real change in
 *    speed or wind wakes everything at once.
 *
 * The classes live on #ui-root rather than on `.hud`, so the ledger and the
 * touch controls take part in the same fade from CSS alone.
 */
const IDLE_MS = 4500;
const HINT_MS = 7000;
/** Pointer travel, px, before a press on the canvas counts as a camera drag. */
const DRAG_SLOP = 6;
/** Speed change, knots, that counts as news worth waking the HUD for. */
const SPEED_WAKE_KNOTS = 1.5;
const TOAST_DEFAULT_MS = 3800;

const POINTS_16 = [
  'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
] as const;

/** Beaufort upper bounds in knots, force 0..11; anything above is 12. */
const BEAUFORT_KNOTS = [1, 4, 7, 11, 17, 22, 28, 34, 41, 48, 56, 64];
const BEAUFORT_NAMES = [
  'Calm', 'Light air', 'Light breeze', 'Gentle breeze', 'Moderate breeze', 'Fresh breeze',
  'Strong breeze', 'Near gale', 'Gale', 'Strong gale', 'Storm', 'Violent storm', 'Hurricane',
];

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

/** 0..360. */
function wrap360(deg: number): number {
  const d = deg % 360;
  return d < 0 ? d + 360 : d;
}

/** -180..180. */
function wrap180(deg: number): number {
  const d = wrap360(deg);
  return d > 180 ? d - 360 : d;
}

function pointName(deg: number): string {
  return POINTS_16[Math.round(wrap360(deg) / 22.5) % 16] ?? 'N';
}

function threeDigits(deg: number): string {
  return String(Math.round(wrap360(deg)) % 360).padStart(3, '0');
}

function beaufort(knots: number): number {
  const index = BEAUFORT_KNOTS.findIndex((limit) => knots < limit);
  return index < 0 ? 12 : index;
}

/** Minutes and seconds, for a voyage's time: 4:07. */
function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

function formatDistance(metres: number): string {
  if (metres < 1000) return `${Math.max(0, Math.round(metres / 5) * 5)} m`;
  return `${(metres / 1852).toFixed(1)} nmi`;
}

function fpsClass(fps: number): string {
  if (fps >= 55) return 'is-good';
  if (fps >= 30) return 'is-warn';
  return 'is-bad';
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'OPTION';
}

/** One line of the score plate. `shown` is the figure on screen while it counts up. */
interface ScoreRow {
  row: HTMLElement;
  value: HTMLElement;
  /** The text the line is settling on. */
  text: string;
  shown: number | null;
  frame: number;
}

/** How long a rising figure takes to count up to its new value, ms. */
const SCORE_COUNT_MS = 900;

function reducedMotion(): boolean {
  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Writes `text` only when it differs, so unchanged readings cost nothing. */
function setText(node: Node, text: string): void {
  if (node.textContent !== text) node.textContent = text;
}

export class Hud {
  private readonly root: HTMLElement;
  private readonly callbacks: UiCallbacks;
  private readonly controller = new AbortController();
  private readonly hudEl: HTMLElement;

  // instrument corner
  private readonly fpsValue: HTMLElement;
  private readonly backendBadge: HTMLElement;
  private readonly pauseButton: HTMLButtonElement;

  // modes
  private readonly hintList: HTMLElement;
  private readonly modeButtons = new Map<CameraMode, HTMLButtonElement>();
  private readonly modeKeys = new Map<CameraMode, HTMLElement>();
  private stopKeyWatch: (() => void) | null = null;

  // compass
  private readonly compassWindow: HTMLElement;
  private readonly tape: HTMLElement;
  private readonly headingText: HTMLElement;
  private readonly headingPoint: HTMLElement;
  private readonly objectiveMark: HTMLElement;
  private readonly shipMark: HTMLElement;
  private compassHalf = 0;

  // wind dial + log
  private readonly instruments: HTMLElement;
  private readonly dialCard: HTMLElement;
  private readonly windArrow: HTMLElement;
  private readonly windKnotsText: HTMLElement;
  private readonly windFromText: HTMLElement;
  private readonly windForceText: HTMLElement;
  private readonly logBlock: HTMLElement;
  private readonly logSub: HTMLElement;
  private readonly logKnotsText: HTMLElement;

  // toasts
  private readonly toastStack: HTMLElement;

  // objective + score
  private readonly objectiveEl: HTMLElement;
  private readonly objectiveTitle: HTMLElement;
  private readonly objectiveDetail: HTMLElement;
  private readonly objectiveBearingRow: HTMLElement;
  private readonly objectiveArrow: HTMLElement;
  private readonly objectiveBearingText: HTMLElement;
  private readonly objectiveWarning: HTMLElement;
  private readonly scoreEl: HTMLElement;
  // voyage tally
  private readonly tallyEl: HTMLElement;
  private readonly tallyTitle: HTMLElement;
  private readonly tallySub: HTMLElement;
  private readonly tallyRows: HTMLElement;
  private readonly tallyAgain: HTMLButtonElement;
  private onSailAgain: (() => void) | null = null;
  private readonly scoreRows = new Map<string, ScoreRow>();

  private mode: CameraMode;
  private pendingFps = 0;
  private writtenFps = -1;
  private lastFpsWrite = 0;
  private fpsTimer = 0;
  private disposed = false;

  private lastInput = performance.now();
  private idle = false;
  /** Whether the viewer has given any input since the HUD was built. */
  private engaged = false;
  private idleTimer = 0;
  private hintTimer = 0;
  private dragStart: { x: number; y: number } | null = null;
  private dragging = false;
  private wakeKnots: number | null = null;
  private wakeWind = Number.NaN;
  private objectiveChipTimer = 0;
  private readonly objectiveMore: HTMLButtonElement;

  private heading = 0;
  private writtenTapeX = Number.NaN;
  private writtenCard = Number.NaN;
  private windArrowAngle = Number.NaN;
  private writtenWindKnots = Number.NaN;
  private writtenWindFrom = Number.NaN;
  private lastNavWrite = 0;
  private shipKnots: number | null = null;
  private objective: Objective | null = null;
  private writtenObjectiveArrow = Number.NaN;
  /** The needle's unwrapped angle, so the transition turns the short way across 180. */
  private objectiveArrowTurn = 0;
  private lastObjectiveWrite = 0;
  private shipHeading: number | null = null;

  // first-minute lessons
  private readonly coach: Coach;
  /** The lesson the ribbon is showing, so a redraw happens only on a change. */
  private shownLesson: LessonId | null = null;
  /** Ship's head when the steering lesson went up; a real turn from it is the lesson done. */
  private steerFrom: number | null = null;
  /** Hands to go back to when the gamepad is put down. */
  private readonly baseInput: InputKind;
  /** The compass's no-go arc either side of the wind, and how it is placed. */
  private readonly noGoArc: HTMLElement;
  private windFrom = 0;
  /** True once her bow has been inside the no-go zone; leaving it under way is the lesson. */
  private wasInNoGo = false;
  /** Seconds-ish of sailing at way, for learning the arc without ever stalling. */
  private sailedTime = 0;
  private lastNavAt = 0;

  constructor(root: HTMLElement, initial: CameraMode, callbacks: UiCallbacks) {
    this.root = root;
    this.callbacks = callbacks;
    this.mode = initial;

    const signal = this.controller.signal;

    const hud = el('section', 'hud');
    hud.setAttribute('aria-label', 'Ship instruments and camera');
    hud.dataset.mode = initial;

    // ---- Compass tape ------------------------------------------------------
    const compass = el('div', 'hud-compass');
    const compassWindow = el('div', 'hud-compass__window');
    compassWindow.setAttribute('aria-hidden', 'true');
    const tape = el('div', 'hud-compass__tape');
    tape.style.width = `${(360 + TAPE_PAD_DEG * 2) * TAPE_PX_PER_DEG}px`;
    for (let deg = -TAPE_PAD_DEG; deg <= 360 + TAPE_PAD_DEG; deg += 15) {
      const bearing = wrap360(deg);
      const cardinal = bearing % 90 === 0;
      const inter = !cardinal && bearing % 45 === 0;
      const label = el(
        'span',
        `hud-compass__label${cardinal ? ' is-cardinal' : inter ? ' is-inter' : ''}${bearing === 0 ? ' is-north' : ''}`,
        cardinal || inter ? pointName(bearing) : String(bearing),
      );
      label.style.left = `${(deg + TAPE_PAD_DEG) * TAPE_PX_PER_DEG}px`;
      tape.append(label);
    }
    const objectiveMark = el('span', 'hud-compass__mark hud-compass__mark--objective');
    objectiveMark.append(icon('mark'));
    objectiveMark.hidden = true;
    const shipMark = el('span', 'hud-compass__mark hud-compass__mark--ship');
    shipMark.hidden = true;
    // The no-go zone: a red band on the tape either side of where the wind
    // comes from. Shown at the helm until the player has sailed out of it.
    const noGoArc = el('span', 'hud-compass__mark hud-compass__nogo');
    noGoArc.style.width = `${NO_GO_DEG * 2 * TAPE_PX_PER_DEG}px`;
    noGoArc.style.marginLeft = `${-NO_GO_DEG * TAPE_PX_PER_DEG}px`;
    noGoArc.hidden = true;
    compassWindow.append(tape, noGoArc, objectiveMark, shipMark);

    const lubber = el('span', 'hud-compass__lubber');
    lubber.setAttribute('aria-hidden', 'true');
    const readout = el('div', 'hud-compass__readout');
    const headingText = el('span', 'hud-compass__deg', '000');
    const headingPoint = el('span', 'hud-compass__point', 'N');
    readout.append(headingText, el('span', 'hud-compass__degsign', '°'), headingPoint);
    readout.setAttribute('aria-hidden', 'true');
    compass.append(compassWindow, lubber, readout);

    // ---- Toasts ------------------------------------------------------------
    const toastStack = el('div', 'hud-toasts');
    toastStack.setAttribute('role', 'status');
    toastStack.setAttribute('aria-live', 'polite');
    toastStack.setAttribute('aria-relevant', 'additions');

    // ---- Objective + score -------------------------------------------------
    const quest = el('div', 'hud-quest');
    const objectiveEl = el('article', 'objective parchment');
    objectiveEl.hidden = true;
    objectiveEl.setAttribute('aria-label', 'Current objective');
    const seal = el('span', 'objective__seal');
    seal.append(icon('mark'));
    const objectiveText = el('div', 'objective__text');
    const objectiveTitle = el('h2', 'objective__title');
    const objectiveDetail = el('p', 'objective__detail');
    const bearingRow = el('p', 'objective__bearing');
    const objectiveArrow = el('span', 'objective__arrow');
    objectiveArrow.setAttribute('aria-hidden', 'true');
    const bearingText = el('span', 'objective__bearing-text');
    bearingRow.append(objectiveArrow, bearingText);
    // What is coming (a squall, the dark), said plainly under the bearing.
    const objectiveWarning = el('p', 'objective__warning');
    objectiveWarning.hidden = true;
    objectiveText.append(objectiveTitle, objectiveDetail, bearingRow, objectiveWarning);
    // On a phone the scroll folds to a chip (seal, title, bearing); this
    // button unfolds it. A wide screen has room for the whole scroll and
    // never shows it.
    const objectiveMore = el('button', 'objective__more');
    objectiveMore.type = 'button';
    objectiveMore.setAttribute('aria-expanded', 'false');
    objectiveMore.setAttribute('aria-label', 'Show objective details');
    objectiveMore.append(icon('chevron'));
    objectiveMore.addEventListener(
      'click',
      () => this.setObjectiveExpanded(!this.objectiveEl.classList.contains('is-expanded')),
      { signal },
    );
    objectiveEl.append(seal, objectiveText, objectiveMore);

    const scoreEl = el('dl', 'score plate');
    scoreEl.hidden = true;
    scoreEl.setAttribute('aria-label', 'Score');

    // The voyage's close: what came home, and the choice to go again. Takes
    // the objective's place while it is up; the only part of the quest column
    // that takes clicks.
    const tallyEl = el('article', 'tally parchment');
    tallyEl.hidden = true;
    tallyEl.setAttribute('aria-label', 'Voyage tally');
    const tallyTitle = el('h2', 'tally__title');
    const tallySub = el('p', 'tally__sub');
    const tallyRows = el('dl', 'tally__rows');
    const tallyAgain = el('button', 'tally__again');
    tallyAgain.type = 'button';
    tallyAgain.append(icon('anchor', 'tally__icon'), el('span', 'tally__again-label', 'Sail again'));
    tallyAgain.addEventListener('click', () => this.onSailAgain?.(), { signal });
    tallyEl.append(tallyTitle, tallySub, tallyRows, tallyAgain);
    quest.append(objectiveEl, tallyEl, scoreEl);

    // ---- Wind dial + log ---------------------------------------------------
    const instruments = el('div', 'hud-instruments');
    const dial = el('div', 'dial');
    dial.setAttribute('aria-hidden', 'true');
    const dialCard = el('div', 'dial__card');
    dialCard.append(compassRose('dial__rose'));
    const windArrow = el('span', 'dial__wind');
    dialCard.append(windArrow);
    dial.append(dialCard, el('span', 'dial__lubber'));

    const readings = el('div', 'hud-readings plate');
    const windBlock = el('div', 'reading');
    windBlock.append(el('span', 'reading__label', 'Wind'));
    const windFigure = el('span', 'reading__figure');
    const windKnotsText = el('span', 'reading__value', '0');
    windFigure.append(windKnotsText, el('span', 'reading__unit', 'kn'));
    const windFromText = el('span', 'reading__sub', 'from N');
    const windForceText = el('span', 'reading__sub reading__sub--force', 'Calm');
    windBlock.append(windFigure, windFromText, windForceText);

    const logBlock = el('div', 'reading reading--log');
    logBlock.append(el('span', 'reading__label', 'Speed'));
    const logFigure = el('span', 'reading__figure');
    const logKnotsText = el('span', 'reading__value', '0.0');
    logFigure.append(logKnotsText, el('span', 'reading__unit', 'kn'));
    const logSub = el('span', 'reading__sub', 'through water');
    logBlock.append(logFigure, logSub);
    logBlock.hidden = true;
    readings.append(windBlock, logBlock);
    instruments.append(dial, readings);

    // ---- Camera modes ------------------------------------------------------
    const helm = el('div', 'hud-helm');
    const hintList = el('ul', 'hud-hints');
    hintList.setAttribute('aria-label', 'Controls');
    const modes = el('div', 'hud-modes');
    modes.setAttribute('role', 'group');
    modes.setAttribute('aria-label', 'Camera mode');
    for (const mode of CAMERA_MODES) {
      const button = el('button', 'tab');
      button.type = 'button';
      button.setAttribute('aria-pressed', mode === this.mode ? 'true' : 'false');
      button.classList.toggle('is-active', mode === this.mode);
      const keycap = el('kbd', 'tab__key');
      button.append(
        icon(MODE_ICONS[mode], 'tab__icon'),
        el('span', 'tab__name', MODE_LABELS[mode]),
        keycap,
      );
      this.modeKeys.set(mode, keycap);
      button.addEventListener('click', () => this.selectMode(mode), { signal });
      this.modeButtons.set(mode, button);
      modes.append(button);
    }
    helm.append(hintList, modes);

    // ---- Instrument corner: FPS + backend ----------------------------------
    const corner = el('div', 'hud-corner');
    const fpsValue = el('span', 'hud-corner__fps is-good', '0');
    fpsValue.setAttribute('role', 'status');
    fpsValue.setAttribute('aria-live', 'off');
    fpsValue.setAttribute('aria-label', 'Frames per second');
    const backendBadge = el('span', 'hud-corner__backend', 'WebGPU');
    backendBadge.setAttribute('aria-label', 'Renderer backend: WebGPU');
    // Pause, for hands with no Escape key. Shown once the host says what pausing does.
    const pauseButton = el('button', 'hud-corner__pause');
    pauseButton.type = 'button';
    pauseButton.hidden = true;
    pauseButton.setAttribute('aria-label', 'Pause and settings');
    pauseButton.title = 'Pause and settings (Esc)';
    pauseButton.append(icon('pause', 'hud-corner__pause-icon'));
    corner.append(pauseButton, fpsValue, el('span', 'hud-corner__unit', 'fps'), backendBadge);
    this.pauseButton = pauseButton;

    hud.append(compass, toastStack, quest, instruments, helm, corner);

    this.hudEl = hud;
    this.fpsValue = fpsValue;
    this.backendBadge = backendBadge;
    this.hintList = hintList;
    this.compassWindow = compassWindow;
    this.tape = tape;
    this.headingText = headingText;
    this.headingPoint = headingPoint;
    this.objectiveMark = objectiveMark;
    this.shipMark = shipMark;
    this.noGoArc = noGoArc;
    this.instruments = instruments;
    this.dialCard = dialCard;
    this.windArrow = windArrow;
    this.windKnotsText = windKnotsText;
    this.windFromText = windFromText;
    this.windForceText = windForceText;
    this.logBlock = logBlock;
    this.logSub = logSub;
    this.logKnotsText = logKnotsText;
    this.toastStack = toastStack;
    this.objectiveEl = objectiveEl;
    this.objectiveTitle = objectiveTitle;
    this.objectiveDetail = objectiveDetail;
    this.objectiveBearingRow = bearingRow;
    this.objectiveArrow = objectiveArrow;
    this.objectiveBearingText = bearingText;
    this.objectiveWarning = objectiveWarning;
    this.scoreEl = scoreEl;
    this.tallyEl = tallyEl;
    this.tallyTitle = tallyTitle;
    this.tallySub = tallySub;
    this.tallyRows = tallyRows;
    this.tallyAgain = tallyAgain;

    this.objectiveMore = objectiveMore;
    // Lessons are phrased for the hands in use: a phone has no W key.
    this.baseInput =
      typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches
        ? 'touch'
        : 'keyboard';
    this.coach = new Coach(this.baseInput);
    this.paintModeKeys();
    // A key rebound in the settings is the key the tabs and the lessons name.
    this.stopKeyWatch = keybinds.onChange(() => {
      this.paintModeKeys();
      this.renderHints(this.mode);
    });
    this.renderHints(this.mode);
    this.root.append(hud);
    this.measureCompass();
    this.placeTape(0);

    window.addEventListener('keydown', this.onKeyDown, { signal });

    // Every kind of input counts as attention. Captured and passive: these
    // only read a timestamp and must never delay the camera's own handlers.
    const opts = { signal, capture: true, passive: true } as const;
    const touch = (): void => this.noteInput();
    window.addEventListener('pointermove', touch, opts);
    window.addEventListener('keydown', touch, opts);
    window.addEventListener('wheel', touch, opts);
    window.addEventListener('pointerdown', this.onPointerDown, opts);
    window.addEventListener('pointerup', this.onPointerUp, opts);
    window.addEventListener('pointercancel', this.onPointerUp, opts);
    this.root.dataset.mode = initial;
    this.idleTimer = window.setInterval(() => this.checkIdle(), 400);
    // The HUD is built behind the boot overlay, so nothing here may start
    // counting yet: the first hints stay up, and nothing fades, until the
    // viewer has actually done something.
    this.hudEl.classList.add('hud--hints');
    window.addEventListener('resize', () => this.measureCompass(), { signal });
  }

  // ======================================================================
  // Truth-telling instruments
  // ======================================================================

  /** Feed the current frame rate; the DOM write is throttled internally. */
  setFps(fps: number): void {
    if (this.disposed) return;
    this.pendingFps = Math.max(0, Math.round(fps));

    const now = performance.now();
    const elapsed = now - this.lastFpsWrite;
    if (elapsed >= FPS_WRITE_INTERVAL) {
      this.flushFps(now);
      return;
    }
    if (this.fpsTimer === 0) {
      this.fpsTimer = window.setTimeout(
        () => {
          this.fpsTimer = 0;
          this.flushFps(performance.now());
        },
        FPS_WRITE_INTERVAL - elapsed,
      );
    }
  }

  /** Gives the corner's pause button something to do, and shows it. */
  setPauseHandler(open: () => void): void {
    if (this.disposed) return;
    this.pauseButton.hidden = false;
    this.pauseButton.addEventListener('click', open, { signal: this.controller.signal });
  }

  /**
   * The backend badge. WebGPU is the only backend; the badge stays as a
   * readout beside the frame rate.
   */
  setBackend(backend: RendererBackend): void {
    if (this.disposed) return;
    this.backendBadge.textContent = 'WebGPU';
    this.backendBadge.setAttribute('aria-label', backend === 'webgpu' ? 'Drawn with WebGPU' : '');
  }

  // ======================================================================
  // Navigation
  // ======================================================================

  /**
   * Per-frame navigation readings. Cheap to call every frame: transforms are
   * written only when they move, and text at most every ~120 ms.
   */
  setNav(nav: NavReadout): void {
    if (this.disposed) return;
    const heading = wrap360(nav.headingDeg);
    this.heading = heading;
    this.placeTape(heading);

    // The card turns against the view, so its north stays on the world's.
    const card = Math.round(-heading * 10) / 10;
    if (card !== this.writtenCard) {
      this.writtenCard = card;
      this.dialCard.style.transform = `rotate(${card}deg)`;
    }

    // The wind arrow lives on the card and points where the wind is going.
    // Accumulated rather than wrapped, so a change across north turns the
    // short way instead of spinning the arrow through a full circle.
    const windTo = wrap360(nav.windFromDeg + 180);
    const arrow = Number.isNaN(this.windArrowAngle)
      ? windTo
      : this.windArrowAngle + wrap180(windTo - this.windArrowAngle);
    if (Math.abs(arrow - this.windArrowAngle) > 0.05 || Number.isNaN(this.windArrowAngle)) {
      this.windArrowAngle = arrow;
      this.windArrow.style.transform = `rotate(${arrow.toFixed(1)}deg)`;
    }

    const windKnots = Math.round(nav.windKnots);
    if (windKnots !== this.writtenWindKnots) {
      this.writtenWindKnots = windKnots;
      setText(this.windKnotsText, String(windKnots));
      const force = beaufort(nav.windKnots);
      setText(this.windForceText, `Force ${force} · ${BEAUFORT_NAMES[force] ?? ''}`);
    }
    const windFrom = Math.round(wrap360(nav.windFromDeg) / 22.5);
    if (windFrom !== this.writtenWindFrom) {
      this.writtenWindFrom = windFrom;
      setText(this.windFromText, `from ${pointName(nav.windFromDeg)}`);
    }

    // Ship mark and log appear only while the viewer has the helm.
    this.shipHeading = nav.shipHeadingDeg === null ? null : wrap360(nav.shipHeadingDeg);
    this.noteHelmProgress(nav.shipKnots);
    this.placeMark(this.shipMark, this.shipHeading);
    // News worth waking for: the wind changing (a new preset) or the ship
    // gaining or losing real way.
    if (!Number.isNaN(this.wakeWind) && Math.abs(nav.windKnots - this.wakeWind) > 2) this.wake();
    this.wakeWind = nav.windKnots;
    if (nav.shipKnots !== null) {
      const knots = Math.abs(nav.shipKnots);
      if (this.wakeKnots === null) {
        this.wakeKnots = knots;
      } else if (Math.abs(knots - this.wakeKnots) > SPEED_WAKE_KNOTS) {
        this.wakeKnots = knots;
        this.wake();
      }
    } else {
      this.wakeKnots = null;
    }

    const atHelm = nav.shipKnots !== null;
    if (this.logBlock.hidden === atHelm) {
      this.logBlock.hidden = !atHelm;
      this.instruments.classList.toggle('has-log', atHelm);
    }
    this.shipKnots = nav.shipKnots;

    if (this.objective !== null) this.placeObjective(false);
    this.windFrom = wrap360(nav.windFromDeg);
    this.placeNoGo();

    const now = performance.now();
    if (now - this.lastNavWrite >= NAV_WRITE_INTERVAL) {
      this.lastNavWrite = now;
      setText(this.headingText, threeDigits(heading));
      setText(this.headingPoint, pointName(heading));
      if (this.shipKnots !== null) setText(this.logKnotsText, Math.abs(this.shipKnots).toFixed(1));
    }
  }

  /**
   * A gamepad was used: lessons and the key list switch to its buttons until
   * a key or a pointer is used again.
   */
  setPadInUse(): void {
    if (this.disposed || !this.coach.setInput('pad')) return;
    this.renderHints(this.mode);
  }

  /**
   * An order given by hands the HUD cannot see (the touch stick's sheets, a
   * gamepad), reported so its lesson counts as done.
   */
  noteLesson(id: LessonId): void {
    if (this.disposed) return;
    this.learn(id);
  }

  // ======================================================================
  // Gameplay surfaces
  // ======================================================================

  /**
   * Shows a toast under the compass. Returns a function that dismisses it
   * early. At most four are shown; a fifth pushes the oldest out.
   */
  toast(text: string, kind: ToastKind = 'info', options: ToastOptions = {}): () => void {
    if (this.disposed) return () => {};
    const node = el('div', `toast toast--${kind} parchment`);
    const medal = el('span', 'toast__medal');
    medal.append(icon(kind));
    const body = el('div', 'toast__body');
    body.append(el('p', 'toast__text', text));
    if (options.detail) body.append(el('p', 'toast__detail', options.detail));
    node.append(medal, body);
    this.toastStack.prepend(node);
    this.wake();

    // Oldest out first. They leave immediately rather than animating, so the
    // stack never grows past the limit even for a burst.
    const live = [...this.toastStack.querySelectorAll<HTMLElement>('.toast:not(.is-leaving)')];
    for (const stale of live.slice(TOAST_LIMIT)) stale.remove();

    const duration = Math.min(15000, Math.max(1200, options.durationMs ?? TOAST_DEFAULT_MS));
    let gone = false;
    const dismiss = (): void => {
      if (gone) return;
      gone = true;
      window.clearTimeout(timer);
      node.classList.add('is-leaving');
      window.setTimeout(() => node.remove(), reducedMotion() ? 0 : 360); // the roll-away is --dur-move, 320 ms
    };
    const timer = window.setTimeout(dismiss, duration);
    return dismiss;
  }

  /**
   * Sets, updates or (with `null`) clears the objective. Safe to call every
   * frame with a moving bearing and distance: the scroll only re-renders its
   * text when the title or detail change, and the distance is throttled.
   */
  setObjective(objective: Objective | null): void {
    if (this.disposed) return;
    const previous = this.objective;
    this.objective = objective === null ? null : { ...objective };

    if (objective === null) {
      this.objectiveEl.hidden = true;
      this.objectiveMark.hidden = true;
      return;
    }

    const fresh = previous === null || previous.title !== objective.title;
    if (fresh) {
      this.wake();
      // A new objective arrives unfolded, then settles to a chip on a phone.
      this.setObjectiveExpanded(true);
      window.clearTimeout(this.objectiveChipTimer);
      this.objectiveChipTimer = window.setTimeout(() => this.setObjectiveExpanded(false), 6000);
    }
    setText(this.objectiveTitle, objective.title);
    setText(this.objectiveDetail, objective.detail ?? '');
    this.objectiveDetail.hidden = !objective.detail;
    const warning = objective.warning ?? '';
    if (warning !== '' && warning !== this.objectiveWarning.textContent) this.wake();
    setText(this.objectiveWarning, warning);
    this.objectiveWarning.hidden = warning === '';
    this.objectiveEl.hidden = false;
    if (fresh && !reducedMotion()) {
      this.objectiveEl.classList.remove('is-new');
      // Reflow so the unfurl replays for a new objective.
      void this.objectiveEl.offsetWidth;
      this.objectiveEl.classList.add('is-new');
    }
    this.placeObjective(true);
  }

  /**
   * Shows the voyage's tally in the objective's place, or puts it away with
   * `null`. `onSailAgain` is what its one button does; the button takes focus
   * so a keyboard can answer it with Enter.
   */
  setTally(tally: VoyageTally | null, onSailAgain?: () => void): void {
    if (this.disposed) return;
    if (tally === null) {
      this.tallyEl.hidden = true;
      this.onSailAgain = null;
      return;
    }
    this.onSailAgain = onSailAgain ?? null;
    const full = tally.cargo >= tally.casks;
    setText(this.tallyTitle, full ? 'All cargo home' : 'Cargo home');
    setText(this.tallySub, `Voyage ${tally.voyage} · at anchor off the cove`);
    const best = tally.newBest
      ? `${formatClock(tally.seconds)} · new best`
      : tally.bestSeconds !== null
        ? formatClock(tally.bestSeconds)
        : 'Land every cask to set one';
    const rows: readonly (readonly [string, string])[] = [
      ['Cargo landed', `${tally.cargo} of ${tally.casks}`],
      ['Gold', `+${tally.gold.toLocaleString('en-GB')}`],
      ['Time', formatClock(tally.seconds)],
      ['Best time', best],
      ['Purse', `${tally.purse.toLocaleString('en-GB')} gold`],
      ['Voyages home', tally.voyage.toLocaleString('en-GB')],
    ];
    this.tallyRows.replaceChildren(
      ...rows.map(([label, value]) => {
        const row = el('div', 'tally__row');
        row.append(el('dt', 'tally__label', label), el('dd', 'tally__value', value));
        return row;
      }),
    );
    const fresh = this.tallyEl.hidden;
    this.tallyEl.hidden = false;
    this.objectiveEl.hidden = true;
    this.objectiveMark.hidden = true;
    this.wake();
    if (fresh) {
      if (!reducedMotion()) {
        this.tallyEl.classList.remove('is-new');
        void this.tallyEl.offsetWidth;
        this.tallyEl.classList.add('is-new');
      }
      this.tallyAgain.focus({ preventScroll: true });
    }
  }

  /**
   * Sets one line of the score plaque, creating it on first use. Values are
   * shown as given — `3`, `'3 / 8'`, `'120 doubloons'`.
   */
  setScore(label: string, value: number | string): void {
    if (this.disposed) return;
    this.noteScore(label, value);
    const text = typeof value === 'number' ? value.toLocaleString('en-GB') : value;
    let entry = this.scoreRows.get(label);
    if (entry === undefined) {
      const row = el('div', 'score__row');
      const value = el('dd', 'score__value');
      row.append(el('dt', 'score__label', label), value);
      this.scoreEl.append(row);
      entry = { row, value, text: '', shown: null, frame: 0 };
      this.scoreRows.set(label, entry);
    } else if (entry.text === text) {
      return;
    }
    window.cancelAnimationFrame(entry.frame);
    const from = entry.shown;
    entry.text = text;
    if (typeof value === 'number' && from !== null && value > from && !reducedMotion()) {
      // Gold and cargo count up to the new figure rather than snapping to it.
      this.countScore(entry, from, value);
    } else {
      entry.value.textContent = text;
      entry.shown = typeof value === 'number' ? value : null;
    }
    this.wake();
    this.scoreEl.hidden = false;
    if (!reducedMotion()) {
      entry.row.classList.remove('is-bump');
      void entry.row.offsetWidth;
      entry.row.classList.add('is-bump');
    }
  }

  private countScore(entry: ScoreRow, from: number, to: number): void {
    const started = performance.now();
    const step = (now: number): void => {
      const p = Math.min(1, (now - started) / SCORE_COUNT_MS);
      const eased = 1 - (1 - p) ** 3;
      entry.shown = p >= 1 ? to : Math.round(from + (to - from) * eased);
      entry.value.textContent = entry.shown.toLocaleString('en-GB');
      if (p < 1) entry.frame = window.requestAnimationFrame(step);
    };
    entry.frame = window.requestAnimationFrame(step);
  }

  /**
   * Marks the ship as stalled head to wind: the wind arrow on the dial turns
   * amber and the speed reading says why she is not moving. Cleared when she
   * is free. The toast still tells the visitor what to do about it.
   */
  setStalled(stalled: boolean): void {
    if (this.disposed) return;
    this.instruments.classList.toggle('is-stalled', stalled);
    setText(this.logSub, stalled ? 'head to wind' : 'through water');
  }

  /** Removes one score line, or every line when `label` is omitted. */
  clearScore(label?: string): void {
    if (this.disposed) return;
    for (const [key, entry] of this.scoreRows) {
      if (label !== undefined && key !== label) continue;
      window.cancelAnimationFrame(entry.frame);
      entry.row.remove();
      this.scoreRows.delete(key);
    }
    this.scoreEl.hidden = this.scoreRows.size === 0;
  }

  // ======================================================================
  // Camera mode
  // ======================================================================

  /** Programmatic mode change — does not fire `onChange`. */
  setCameraMode(m: CameraMode): void {
    if (this.disposed || m === this.mode) return;
    this.mode = m;
    this.hudEl.dataset.mode = m;
    this.root.dataset.mode = m;
    if (m === 'boat') this.coach.learn('helm');
    this.steerFrom = null;
    this.wake();
    this.showHints();
    for (const [mode, button] of this.modeButtons) {
      const active = mode === m;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
    this.renderHints(m);
    // The newly chosen tab settles in with a small press, so a mode change
    // made from the keyboard is seen on the bar too.
    const tab = this.modeButtons.get(m);
    if (tab && !reducedMotion()) {
      tab.classList.remove('is-arriving');
      void tab.offsetWidth;
      tab.classList.add('is-arriving');
    }
  }

  /** Brings every faded element back, as if the viewer had moved the mouse. */
  wake(): void {
    if (this.disposed) return;
    this.lastInput = performance.now();
    if (this.idle) {
      this.idle = false;
      this.root.classList.remove('is-idle');
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.fpsTimer !== 0) {
      window.clearTimeout(this.fpsTimer);
      this.fpsTimer = 0;
    }
    this.controller.abort();
    this.stopKeyWatch?.();
    this.stopKeyWatch = null;
    window.clearInterval(this.idleTimer);
    window.clearTimeout(this.hintTimer);
    window.clearTimeout(this.objectiveChipTimer);
    this.root.classList.remove('is-idle', 'is-dragging');
    delete this.root.dataset.mode;
    this.hudEl.remove();
    this.modeButtons.clear();
    this.modeKeys.clear();
    for (const entry of this.scoreRows.values()) window.cancelAnimationFrame(entry.frame);
    this.scoreRows.clear();
  }

  // -- internals ------------------------------------------------------------

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    if (isTypingTarget(event.target)) return;
    this.leavePad();

    // The helm's orders, as lessons done. Pressing the key is knowing it; the
    // gameplay layer carries out the order itself.
    const action = keybinds.actionFor(event.code);
    if (this.mode === 'boat') {
      const lesson: LessonId | null =
        action === 'sailMore' ? 'sail'
          : action === 'steerPort' || action === 'steerStarboard' ? 'steer'
            : action === 'anchor' ? 'anchor'
              : action === 'trimIn' || action === 'trimOut' ? 'trim'
                : null;
      if (lesson !== null) this.learn(lesson);
    }

    const mode = CAMERA_MODES.find((candidate) => MODE_ACTIONS[candidate] === action);
    if (mode === undefined) return;

    event.preventDefault();
    this.selectMode(mode);
  };

  /** Writes each tab's key, and the words that name it, from the bindings in force. */
  private paintModeKeys(): void {
    for (const [mode, button] of this.modeButtons) {
      const key = modeKey(mode);
      const keycap = this.modeKeys.get(mode);
      if (keycap) setText(keycap, key);
      button.setAttribute('aria-label', `${MODE_LABELS[mode]}: ${MODE_TIPS[mode]} (key ${key})`);
      button.title = `${MODE_TIPS[mode]} (${key})`;
    }
  }

  private selectMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    this.setCameraMode(mode);
    this.callbacks.onChange('cameraMode', mode);
  }

  private noteInput(): void {
    this.lastInput = performance.now();
    if (!this.engaged) {
      this.engaged = true;
      this.showHints();
    }
    if (this.idle) this.wake();
  }

  private checkIdle(): void {
    if (!this.engaged || this.idle || this.dragging) return;
    if (performance.now() - this.lastInput < IDLE_MS) return;
    // Something the viewer is reading or operating keeps the chrome up.
    if (this.toastStack.querySelector('.toast:not(.is-leaving)') !== null) return;
    if (!this.tallyEl.hidden) return;
    if (this.root.querySelector(':focus-visible') !== null) return;
    this.idle = true;
    this.root.classList.add('is-idle');
  }

  /** A press on the canvas becomes a camera drag once it travels a few pixels. */
  private readonly onPointerDown = (event: PointerEvent): void => {
    this.noteInput();
    this.leavePad();
    if (!(event.target instanceof HTMLCanvasElement)) return;
    this.dragStart = { x: event.clientX, y: event.clientY };
    window.addEventListener('pointermove', this.onDragMove, {
      passive: true,
      signal: this.controller.signal,
    });
  };

  private readonly onDragMove = (event: PointerEvent): void => {
    if (this.dragStart === null || this.dragging) return;
    const travel = Math.hypot(event.clientX - this.dragStart.x, event.clientY - this.dragStart.y);
    if (travel < DRAG_SLOP) return;
    this.dragging = true;
    this.root.classList.add('is-dragging');
    this.learn('look');
  };

  private readonly onPointerUp = (): void => {
    window.removeEventListener('pointermove', this.onDragMove);
    this.dragStart = null;
    if (!this.dragging) return;
    this.dragging = false;
    this.root.classList.remove('is-dragging');
    this.lastInput = performance.now();
  };

  private showHints(): void {
    this.hudEl.classList.add('hud--hints');
    window.clearTimeout(this.hintTimer);
    this.hintTimer = window.setTimeout(() => {
      // A lesson stays up until it is done; only the key list times out.
      if (this.shownLesson === null) this.hudEl.classList.remove('hud--hints');
    }, HINT_MS);
  }

  /** Keys or a pointer again: back to their phrasing. */
  private leavePad(): void {
    if (this.coach.inputKind === 'pad' && this.coach.setInput(this.baseInput)) {
      this.renderHints(this.mode);
    }
  }

  /**
   * Places the no-go arc on the tape, centred on where the wind comes from,
   * and learns it: once her bow has been in the zone and she has sailed out
   * of it with way on, or after two minutes of sailing that never went near
   * it, the arc fades and stays gone.
   */
  private placeNoGo(): void {
    const now = performance.now();
    const elapsed = this.lastNavAt > 0 ? Math.min(0.5, (now - this.lastNavAt) / 1000) : 0;
    this.lastNavAt = now;

    const ship = this.shipHeading;
    if (ship === null) {
      this.noGoArc.hidden = true;
      return;
    }
    const off = Math.abs(wrap180(ship - this.windFrom));
    const knots = Math.abs(this.shipKnots ?? 0);
    if (!this.coach.has('nogo')) {
      if (off < NO_GO_DEG) this.wasInNoGo = true;
      if (knots > 2) this.sailedTime += elapsed;
      if ((this.wasInNoGo && off > NO_GO_DEG + 10 && knots > 2) || this.sailedTime > 120) {
        this.learn('nogo');
      }
    }
    const learned = this.coach.has('nogo');
    this.noGoArc.classList.toggle('is-learned', learned);
    this.noGoArc.classList.toggle('is-inside', off < NO_GO_DEG);
    this.placeMark(this.noGoArc, this.windFrom);
  }

  /** Marks a lesson done and moves the ribbon on to the next one. */
  private learn(id: LessonId): void {
    if (!this.coach.learn(id)) return;
    this.renderHints(this.mode);
    // The key list for the mode gets its usual few seconds once the last
    // lesson that applies here is done.
    if (this.shownLesson === null) this.showHints();
  }

  /** Score lines the host writes are also the evidence that an order was given. */
  private noteScore(label: string, value: number | string): void {
    if (label === 'Sails' && value !== 'Furled') this.learn('sail');
    else if (label === 'Cargo' && typeof value === 'number' && value > 0) this.learn('cargo');
    else if (label === 'Anchor') this.learn('anchor');
  }

  /**
   * At the helm, the ship's own motion proves a lesson: way on her means the
   * sail is set, and her head swung well round from where the steering lesson
   * found it means she has been steered — the same proof for a keyboard and a
   * thumb on the stick.
   */
  private noteHelmProgress(shipKnots: number | null): void {
    if (shipKnots === null || this.shipHeading === null) {
      this.steerFrom = null;
      return;
    }
    if (Math.abs(shipKnots) > 2) this.learn('sail');
    if (this.shownLesson !== 'steer') return;
    if (this.steerFrom === null) {
      this.steerFrom = this.shipHeading;
    } else if (Math.abs(wrap180(this.shipHeading - this.steerFrom)) > 30) {
      this.learn('steer');
    }
  }

  private setObjectiveExpanded(expanded: boolean): void {
    this.objectiveEl.classList.toggle('is-expanded', expanded);
    this.objectiveMore.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    this.objectiveMore.setAttribute(
      'aria-label',
      expanded ? 'Hide objective details' : 'Show objective details',
    );
  }

  private flushFps(now: number): void {
    this.lastFpsWrite = now;
    if (this.pendingFps === this.writtenFps) return;
    this.writtenFps = this.pendingFps;
    this.fpsValue.textContent = String(this.pendingFps);
    this.fpsValue.className = `hud-corner__fps ${fpsClass(this.pendingFps)}`;
  }

  private renderHints(mode: CameraMode): void {
    const lesson = this.coach.current(mode);
    this.shownLesson = lesson?.id ?? null;
    this.hintList.replaceChildren();
    this.hintList.classList.toggle('is-coach', lesson !== null);
    // A lesson is news a screen reader should hear; a key list is not.
    this.hintList.setAttribute('aria-live', lesson !== null ? 'polite' : 'off');
    if (lesson !== null) {
      const row = el('li', 'hud-hints__item hud-hints__item--lesson');
      if (lesson.key !== undefined) row.append(el('kbd', 'keycap', lesson.key));
      row.append(el('span', 'hud-hints__action', lesson.text));
      this.hintList.append(row);
      this.hudEl.classList.add('hud--hints');
      return;
    }
    const hints = mode === 'boat' && this.coach.inputKind === 'pad' ? PAD_BOAT_HINTS : hintsFor(mode);
    for (const hint of hints) {
      const row = el('li', 'hud-hints__item');
      row.append(el('kbd', 'keycap', hint.key), el('span', 'hud-hints__action', hint.action));
      this.hintList.append(row);
    }
  }

  private measureCompass(): void {
    this.compassHalf = this.compassWindow.clientWidth / 2;
    // Marks are placed against the half-width, so re-place them.
    this.placeMark(this.shipMark, this.shipHeading);
    if (this.objective !== null) this.placeObjective(true);
  }

  private placeTape(heading: number): void {
    const x = Math.round(-(heading + TAPE_PAD_DEG) * TAPE_PX_PER_DEG * 10) / 10;
    if (x === this.writtenTapeX) return;
    this.writtenTapeX = x;
    this.tape.style.transform = `translate3d(${x}px, 0, 0)`;
  }

  /**
   * Places a mark on the tape at `bearing`. Off the visible arc it pins to the
   * nearer end and turns into an arrow, so a target behind the viewer still
   * says which way to turn.
   */
  private placeMark(mark: HTMLElement, bearing: number | null): void {
    if (bearing === null) {
      mark.hidden = true;
      return;
    }
    mark.hidden = false;
    const limit = Math.max(0, this.compassHalf - 16);
    const offset = wrap180(bearing - this.heading) * TAPE_PX_PER_DEG;
    const clamped = Math.max(-limit, Math.min(limit, offset));
    mark.style.transform = `translate3d(${clamped.toFixed(1)}px, 0, 0)`;
    mark.classList.toggle('is-off-left', offset < -limit);
    mark.classList.toggle('is-off-right', offset > limit);
  }

  private placeObjective(force: boolean): void {
    const objective = this.objective;
    if (objective === null) return;
    this.placeMark(this.objectiveMark, objective.bearingDeg);

    const hasBearing = objective.bearingDeg !== null;
    this.objectiveBearingRow.hidden = !hasBearing && objective.distanceM === null;
    this.objectiveArrow.hidden = !hasBearing;
    if (hasBearing) {
      const relative = Math.round(wrap180(objective.bearingDeg! - this.heading));
      if (relative !== this.writtenObjectiveArrow) {
        // A wrapped angle jumping from 179 to -179 would spin the whole circle
        // under the transition; step by the shortest difference instead.
        if (Number.isNaN(this.writtenObjectiveArrow)) this.objectiveArrowTurn = relative;
        else this.objectiveArrowTurn += wrap180(relative - this.writtenObjectiveArrow);
        this.writtenObjectiveArrow = relative;
        this.objectiveArrow.style.transform = `rotate(${this.objectiveArrowTurn}deg)`;
      }
    }

    const now = performance.now();
    if (!force && now - this.lastObjectiveWrite < NAV_WRITE_INTERVAL * 2) return;
    this.lastObjectiveWrite = now;
    const parts: string[] = [];
    if (hasBearing) parts.push(`${threeDigits(objective.bearingDeg!)}° ${pointName(objective.bearingDeg!)}`);
    if (objective.distanceM !== null) parts.push(formatDistance(objective.distanceM));
    setText(this.objectiveBearingText, parts.join('  ·  '));
  }
}
