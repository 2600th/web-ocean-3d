import * as THREE from 'three/webgpu';
import { QUALITY_TIERS, type QualitySettings, type QualityTier } from './QualityManager';

export type Backend = 'webgpu' | 'webgl';

export interface RendererBootstrap {
  renderer: THREE.WebGPURenderer;
  backend: Backend;
  /** True when WebGPU was requested but the device could not be acquired. */
  fellBack: boolean;
}

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  /** Force the WebGL2 backend even when WebGPU is available. */
  forceWebGL?: boolean;
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

export async function createRenderer(options: RendererOptions): Promise<RendererBootstrap> {
  const { canvas, forceWebGL = false } = options;
  const quality = resolveQualityPolicy(options.quality);

  // An undefined adapter means the caller has not probed yet. A supplied null
  // is a real result from the start gate and must take the WebGL2 fallback
  // without paying for another adapter request. Three's backend still owns
  // device creation; this avoids the redundant availability probe that used to
  // precede it.
  const webgpuAvailable =
    !forceWebGL &&
    (options.adapter === undefined ? await probeWebGPU() : options.adapter !== null);
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: quality.antialias,
    forceWebGL: !webgpuAvailable,
    powerPreference: 'high-performance',
    alpha: false,
  });

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
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  await renderer.init();

  // `isWebGPUBackend` is set on the concrete backend instance but is not part of
  // the exported base type, so read it defensively.
  const isWebGPU = (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend === true;
  const backend: Backend = isWebGPU ? 'webgpu' : 'webgl';

  return {
    renderer,
    backend,
    fellBack: !forceWebGL && backend === 'webgl',
  };
}

/**
 * `navigator.gpu` existing is not sufficient — adapter request can still fail on
 * blocklisted drivers, and a failed WebGPU init leaves a dead canvas. Probe first.
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
