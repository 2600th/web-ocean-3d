import * as THREE from 'three/webgpu';

/**
 * A 64x64 blue-noise tile, one byte per texel, ranks 0..255 (sixteen texels per
 * rank).
 *
 * Generated for this project with Ulichney's void-and-cluster method ("The
 * void-and-cluster method for dither array generation", SPIE 1993): toroidal
 * Gaussian energy with sigma 1.5, seeded initial pattern, the usual three
 * ranking phases. The generator is a forty-line numpy script; what ships is its
 * output, because running it in the browser would cost startup time for a
 * table that never changes. Its spectrum has 0.02% of its energy in the lowest
 * 9x9 frequencies, against 2% for white noise of the same histogram.
 *
 * Used for the cloud march's per-pixel start offset. Blue noise rather than
 * interleaved gradient noise, because IGN's structure is diagonal and a ray
 * march offset by it draws diagonal hatching along every soft edge; blue noise
 * leaves isotropic grain with no low-frequency energy for the eye to group into
 * lines, and that is what a 3x3 neighbourhood clamp and a temporal blend remove.
 */
const BLUE_NOISE_64 =
  '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';

export const BLUE_NOISE_SIZE = 64;

let shared: THREE.DataTexture | null = null;

/** The tile as an R8 texture: nearest filtering, repeat wrapping. Shared. */
export function blueNoiseTexture(): THREE.DataTexture {
  if (shared !== null) return shared;
  const binary = atob(BLUE_NOISE_64);
  const data = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) data[i] = binary.charCodeAt(i);
  const tex = new THREE.DataTexture(data, BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, THREE.RedFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.name = 'clouds.blueNoise';
  tex.needsUpdate = true;
  shared = tex;
  return tex;
}
