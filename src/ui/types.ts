/**
 * State contract shared between the UI layer and the application shell.
 *
 * This module is deliberately dependency-free: nothing in `src/ui` imports
 * three.js, so the whole control layer can be mounted and tested standalone.
 */

export type QualityTier = 'low' | 'medium' | 'high' | 'ultra' | 'max';

export type PresetId =
  | 'skyPro'
  | 'arctic'
  | 'blackFlag'
  | 'dusk'
  | 'foggy'
  | 'moonlit'
  | 'seaOfThieves'
  | 'storm'
  | 'sunset';

export type CameraMode = 'orbit' | 'fly' | 'boat' | 'cinematic';

export interface UiState {
  quality: QualityTier;
  preset: PresetId;
  /** Wind speed at 10 m reference height, m/s. Range 0.5..25. */
  windSpeed: number;
  /** JONSWAP peak wavelength in metres. Range 10..150. */
  peakWavelength: number;
  /** Volumetric cloud layer coverage, 0..1. */
  cloudCoverage: number;
  /**
   * Time of day, 0..24 hours. 12 is noon, 0 and 24 are midnight.
   *
   * Overrides the preset's own sun elevation once the viewer touches it — a
   * preset describes a *look*, and the time of day is the one part of that look
   * the viewer is being invited to drive. `null` means the preset still owns it.
   */
  timeOfDay: number | null;
  /** Volumetric fog density, 0..1. 0 is a bit-exact pass-through. */
  fogDensity: number;
  /**
   * Master audio level, 0..1. 0 is silence.
   *
   * The audio graph stays suspended until a user gesture whatever this says —
   * browsers require one — so a non-zero default is a statement of intent
   * rather than a promise of noise.
   */
  volume: number;
  /**
   * Music level, 0..1, under the master volume. 0 stops the score outright
   * (nothing is scheduled), so it is also the "music off" switch.
   */
  musicVolume: number;
  buoyancyProbes: boolean;
  wakeProbes: boolean;
  forceWebGL: boolean;
  /** Device pixel ratio multiplier, 0.5..2. */
  pixelRatio: number;
  cameraMode: CameraMode;
}

export interface UiCallbacks {
  onChange<K extends keyof UiState>(key: K, value: UiState[K]): void;
}

/** Renderer backend reported back to the HUD badge. */
export type RendererBackend = 'webgpu' | 'webgl';

/**
 * Navigation readings for the HUD's compass, wind dial and log.
 *
 * Every bearing is a **compass bearing**: degrees clockwise from north, where
 * north is world −Z and east is world +X. Any value is accepted and wrapped, so
 * a caller never has to normalise. The HUD owns every presentation decision
 * (wrapping, smoothing, rounding, throttled text writes); the host just reports
 * what is true each frame.
 */
export interface NavReadout {
  /** Bearing the camera is looking toward. Drives the compass tape and dial card. */
  headingDeg: number;
  /** Bearing the wind blows *from* — the sailor's convention ("wind from the NE"). */
  windFromDeg: number;
  /** Wind speed in knots. */
  windKnots: number;
  /** The ship's own heading while the viewer has the helm; `null` hides its mark. */
  shipHeadingDeg: number | null;
  /** Speed through the water in knots while at the helm; `null` hides the log. */
  shipKnots: number | null;
}

/**
 * Toast flavour. Each has its own drawn icon and colour; none is an error
 * surface (PRODUCT.md: nothing here is destructive).
 */
export type ToastKind = 'info' | 'reward' | 'warning' | 'danger';

export interface ToastOptions {
  /** A second, smaller line under the headline. */
  detail?: string;
  /** How long it stays, in ms. Default 3800; clamped to 1200..15000. */
  durationMs?: number;
}

/** The current objective, shown as a sealed scroll top-left and a mark on the compass. */
export interface Objective {
  title: string;
  detail?: string;
  /** Compass bearing from the viewer to the objective, or `null` if it has none. */
  bearingDeg: number | null;
  /** Distance in metres, or `null` to hide the distance. */
  distanceM: number | null;
}

/**
 * Default state: the stylised-real Tropical look a visitor lands on.
 *
 * The canonical shots name every value they depend on, so moving the first
 * impression off the reference frame does not move any baseline.
 */
export const DEFAULT_UI_STATE: UiState = {
  quality: 'high',
  preset: 'seaOfThieves',
  windSpeed: 11,
  peakWavelength: 40,
  cloudCoverage: 0.36,
  timeOfDay: null,
  fogDensity: 0.35,
  volume: 0.4,
  musicVolume: 0.5,
  buoyancyProbes: false,
  wakeProbes: false,
  forceWebGL: false,
  pixelRatio: 1,
  cameraMode: 'orbit',
};

export const QUALITY_TIERS: readonly QualityTier[] = ['low', 'medium', 'high', 'ultra', 'max'];

export const PRESET_IDS: readonly PresetId[] = [
  'skyPro',
  'arctic',
  'blackFlag',
  'dusk',
  'foggy',
  'moonlit',
  'seaOfThieves',
  'storm',
  'sunset',
];

export const CAMERA_MODES: readonly CameraMode[] = ['orbit', 'fly', 'boat', 'cinematic'];
