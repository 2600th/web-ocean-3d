import * as THREE from 'three/webgpu';
import { QUALITY_TIERS, type QualitySettings, type QualityTier } from './QualityManager';

/**
 * The one backend. The WebGL2 fallback was retired with the r187 upgrade: every
 * effect that sets the look — the scene MRT under TRAA, the planar mirror,
 * refraction, the underwater pass, lens rain, depth of field — was either off
 * or reduced there, so a browser without WebGPU got a different, lesser game.
 * It now gets a page that says what it needs instead.
 */
export type Backend = 'webgpu';

export interface RendererBootstrap {
  renderer: THREE.WebGPURenderer;
  backend: Backend;
}

/** Why the renderer could not start: no adapter at all, or a device that would not open. */
export type WebGPUUnavailableReason = 'no-adapter' | 'no-device';

export class WebGPUUnavailableError extends Error {
  constructor(
    readonly reason: WebGPUUnavailableReason,
    readonly cause?: unknown,
  ) {
    super(reason === 'no-adapter' ? 'WebGPU is not available in this browser' : 'The WebGPU device could not be created');
    this.name = 'WebGPUUnavailableError';
  }
}

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  pixelRatio?: number;
  /**
   * Quality tier, or the already-resolved framebuffer policy for that tier.
   * Resolving it before construction is useful to callers that do not want the
   * renderer to depend on the quality manager at boot.
   */
  quality?: QualityTier | Pick<QualitySettings, 'maxPixelRatio' | 'antialias'>;
  /** Adapter already obtained by the start gate; avoids a second requestAdapter(). */
  adapter?: GPUAdapter | null;
}

export type RendererQualityPolicy = Pick<QualitySettings, 'maxPixelRatio' | 'antialias'>;

/** Upper bound on DPR — beyond 2x the cost is real and the gain is not. */
export const MAX_PIXEL_RATIO = 2;

/**
 * Creates the WebGPU renderer, or throws `WebGPUUnavailableError`.
 *
 * Never falls back. `WebGPURenderer` installs a WebGL2 fallback in its
 * constructor (`_getFallback`) and takes it silently when the device fails to
 * open — a blocklisted driver, a lost GPU process — so it is cleared before
 * `init()`, and a failure surfaces as an error the boot can explain.
 */
export async function createRenderer(options: RendererOptions): Promise<RendererBootstrap> {
  const { canvas } = options;
  const quality = resolveQualityPolicy(options.quality);

  // An undefined adapter means the caller has not probed yet. A supplied null
  // is a real result from the start gate.
  const available = options.adapter === undefined ? await probeWebGPU() : options.adapter !== null;
  if (!available) throw new WebGPUUnavailableError('no-adapter');

  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: quality.antialias,
    powerPreference: 'high-performance',
    alpha: false,
  });
  (renderer as unknown as { _getFallback: unknown })._getFallback = null;

  renderer.setPixelRatio(
    Math.min(quality.maxPixelRatio, clampPixelRatio(options.pixelRatio ?? window.devicePixelRatio)),
  );
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  // Linear-space rendering with filmic tonemapping — the ocean has a very wide
  // dynamic range between sun glitter and shadowed troughs.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  // `PCFSoftShadowMap` was removed in r186 and maps to this anyway.
  renderer.shadowMap.type = THREE.PCFShadowMap;

  try {
    await renderer.init();
  } catch (error) {
    renderer.dispose();
    throw new WebGPUUnavailableError('no-device', error);
  }
  if ((renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend !== true) {
    renderer.dispose();
    throw new WebGPUUnavailableError('no-device');
  }

  return { renderer, backend: 'webgpu' };
}

/**
 * `navigator.gpu` existing is not sufficient — adapter request can still fail on
 * blocklisted drivers. Probe first.
 */
async function probeWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    return adapter !== null;
  } catch {
    return false;
  }
}

export function clampPixelRatio(value: number): number {
  return Math.min(MAX_PIXEL_RATIO, Math.max(0.5, value));
}

function resolveQualityPolicy(
  quality: RendererOptions['quality'],
): RendererQualityPolicy {
  if (quality === undefined) {
    return { maxPixelRatio: MAX_PIXEL_RATIO, antialias: true };
  }
  if (typeof quality === 'string') {
    const settings = QUALITY_TIERS[quality];
    return {
      maxPixelRatio: settings.maxPixelRatio,
      antialias: settings.antialias,
    };
  }
  return {
    maxPixelRatio: Math.min(MAX_PIXEL_RATIO, Math.max(0.5, quality.maxPixelRatio)),
    antialias: quality.antialias,
  };
}
