import type { Page } from '@playwright/test';

/**
 * Provenance for a baseline.
 *
 * A rendered frame is a property of the whole stack, not of this repository:
 * change the GPU, the driver, the browser build or the backend and the same
 * source produces a different, equally correct image. Comparing across stacks
 * and calling the difference a regression is the single easiest way to make a
 * visual harness untrustworthy, so every baseline records what produced it and
 * the harness says so — loudly — when the current run does not match.
 */
export interface StackFingerprint {
  /** Which renderer backend the app actually chose, not which one was asked for. */
  backend: 'webgpu';
  /** WebGL's unmasked renderer string. The most specific GPU identity available. */
  gpu: string;
  /** WebGPU adapter identity, or `none` when there is no adapter. */
  adapter: string;
  /** Browser product and build, from the user-agent. */
  browser: string;
  width: number;
  height: number;
  devicePixelRatio: number;
}

/** Fields whose difference invalidates a comparison. `gpu` is the sharpest. */
const COMPARED_FIELDS: readonly (keyof StackFingerprint)[] = [
  'backend',
  'gpu',
  'adapter',
  'browser',
  'width',
  'height',
  'devicePixelRatio',
];

export async function describeStack(page: Page): Promise<StackFingerprint> {
  return page.evaluate(async () => {
    const gl = document.createElement('canvas').getContext('webgl2');
    const debugInfo = gl?.getExtension('WEBGL_debug_renderer_info') ?? null;
    const gpu =
      gl && debugInfo
        ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
        : (gl && String(gl.getParameter(gl.RENDERER))) || 'unknown';

    let adapter = 'none';
    try {
      const found = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
      if (found) {
        const info = found.info;
        adapter = info
          ? [info.vendor, info.architecture, info.device, info.description]
              .filter(Boolean)
              .join('/') || 'present'
          : 'present';
      }
    } catch {
      adapter = 'error';
    }

    // Chrome/141.0.0.0 out of the UA, which is what actually changes the image
    // when Chromium updates. The rest of the UA is constant noise.
    const build = /(Chrome|Chromium|Firefox|Version)\/([\d.]+)/.exec(navigator.userAgent);
    const browser = build ? `${build[1]}/${build[2]}` : navigator.userAgent;

    return {
      backend: window.__ocean.backend,
      gpu,
      adapter,
      browser,
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio,
    };
  });
}

/**
 * True when the GPU string names a CPU rasteriser.
 *
 * Playwright's Chromium defaults to SwiftShader unless `--enable-gpu` is passed,
 * and SwiftShader renders this scene correctly but not identically to a GPU —
 * different filtering, different floating-point rounding, no anisotropy. A
 * baseline captured on it would be a baseline for the software rasteriser.
 */
export function isSoftwareRenderer(gpu: string): boolean {
  return /swiftshader|llvmpipe|software|basic render|microsoft basic/i.test(gpu);
}

/** Names the fields on which `current` differs from `recorded`. */
export function stackDifferences(
  recorded: StackFingerprint,
  current: StackFingerprint,
): string[] {
  return COMPARED_FIELDS.filter((field) => recorded[field] !== current[field]).map(
    (field) => `${field}: baseline "${recorded[field]}" vs current "${current[field]}"`,
  );
}

export function formatStack(stack: StackFingerprint): string {
  return (
    `${stack.backend} on ${stack.gpu} (adapter ${stack.adapter}), ` +
    `${stack.browser}, ${stack.width}x${stack.height} @ ${stack.devicePixelRatio}x`
  );
}
