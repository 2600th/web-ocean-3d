import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  acos,
  clamp,
  cos,
  dot,
  exp,
  float,
  fwidth,
  length,
  max,
  min,
  mix,
  pow,
  rtt,
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * Physically based clear-sky atmosphere: Hillaire's LUT model ("A Scalable and
 * Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020).
 *
 * **Ported from web-ocean** (github.com/Noniv/web-ocean, `js/wgsl/sky.js` and
 * `js/wgsl/common.js`, MIT, (c) 2026 Maksymilian Dendura — see
 * THIRD_PARTY_NOTICES.md). The constants, the medium, the transmittance and
 * sky-view parameterisations, the multiple-scattering LUT and the limb-darkened
 * sun disc are theirs, transcribed from WGSL. What changed in the port, and why:
 *
 * - Their transmittance and multiple-scattering LUTs are compute passes. Here
 *   they are built on the CPU and uploaded as half-float `DataTexture`s, because
 *   the same tables must also exist on the CPU (`radiance`, `transmittance`):
 *   the sun's colour, the hemisphere fill and the clouds' ambient are all read
 *   from this model on the CPU, and two copies of a table cannot disagree. It
 *   also keeps the WebGL2 fallback on the same code path — no storage textures.
 *   Both depend on haze alone, so they are rebuilt only on a preset change.
 * - Their sky-view LUT is a compute pass run every frame. Here it is a fragment
 *   pass through three's `rtt` with `autoUpdate` off, re-rendered only when the
 *   sun's elevation, the haze or the Mie anisotropy actually moved — a static
 *   sun costs nothing, the tour's moving one costs one 256x144 quad.
 * - The camera altitude is pinned at 10 m (theirs is `max(cam.y, 10 m)`): our
 *   viewer never leaves the sea surface by more than a mast, which is a
 *   hundredth of the 1.2 km aerosol scale height.
 * - Their Mie g is fixed at 0.84; ours stays a preset parameter
 *   (`mieDirectionalG`, clamped to 0.6..0.9) because the presets author it.
 *
 * Units are theirs: kilometres, scattering per km, radiance per unit solar
 * illuminance. `Atmosphere` scales the result into this renderer's range.
 */

// ---------------------------------------------------------------------------
// Constants (web-ocean `ATMO_CORE_WGSL`).
// ---------------------------------------------------------------------------
export const RG = 6360.0;
export const RT = 6460.0;
const RAY_S = [5.802e-3, 13.558e-3, 33.1e-3] as const;
const MIE_S = 3.996e-3;
const MIE_E = 4.4e-3;
const OZO_A = [0.65e-3, 1.881e-3, 0.085e-3] as const;
/** Angular radius of the sun, radians. */
export const SUN_ANG_R = 0.004675;
const TRANS_W = 256;
const TRANS_H = 64;
const MS_SIZE = 32;
const SKY_W = 256;
const SKY_H = 144;
const H_TOP = Math.sqrt(RT * RT - RG * RG);
/** Viewer altitude, km. See the header. */
const CAM_ALT = 0.01;
export const CAM_R = RG + CAM_ALT;
/** Sky-view march steps (web-ocean: 48). */
const SKY_STEPS = 48;
/** CPU radiance march steps. The CPU answers a handful of low-order questions per frame. */
const CPU_STEPS = 32;

/** Rayleigh, Mie and ozone at height `h` km. Writes [extR, extG, extB, rayScale, mieS]. */
function mediumCPU(h: number, haze: number, out: Float64Array): void {
  const hh = Math.max(h, 0);
  const dR = Math.exp(-hh / 8.0);
  const dM = Math.exp(-hh / 1.2);
  const dO = Math.max(0, 1 - Math.abs(hh - 25) / 15);
  const mieE = MIE_E * dM * haze;
  out[0] = RAY_S[0] * dR + mieE + OZO_A[0] * dO;
  out[1] = RAY_S[1] * dR + mieE + OZO_A[1] * dO;
  out[2] = RAY_S[2] * dR + mieE + OZO_A[2] * dO;
  out[3] = dR;
  out[4] = MIE_S * dM * haze;
}

/** Far intersection distance of a ray from radius `r` (cos zenith `mu`) with sphere `R`, or -1. */
function sphereFar(r: number, mu: number, R: number): number {
  const b = r * mu;
  const d = b * b - (r * r - R * R);
  if (d < 0) return -1;
  return -b + Math.sqrt(d);
}
function sphereNear(r: number, mu: number, R: number): number {
  const b = r * mu;
  const d = b * b - (r * r - R * R);
  if (d < 0) return -1;
  return -b - Math.sqrt(d);
}

function transRMuToUV(r: number, mu: number, out: number[]): void {
  const rho = Math.sqrt(Math.max(0, (r - RG) * (r + RG)));
  const disc = r * r * (mu * mu - 1) + RT * RT;
  const d = Math.max(0, -r * mu + Math.sqrt(Math.max(disc, 0)));
  const dMin = RT - r;
  const dMax = rho + H_TOP;
  out[0] = (d - dMin) / (dMax - dMin);
  out[1] = rho / H_TOP;
}

function transUVToRMu(u: number, v: number, out: number[]): void {
  const rho = H_TOP * v;
  const r = Math.sqrt(rho * rho + RG * RG);
  const dMin = RT - r;
  const dMax = rho + H_TOP;
  const d = dMin + u * (dMax - dMin);
  let mu = 1;
  if (d > 0) mu = (H_TOP * H_TOP - rho * rho - d * d) / (2 * r * d);
  out[0] = r;
  out[1] = Math.min(1, Math.max(-1, mu));
}

const fromUnitToSubUvs = (u: number, res: number): number => (u + 0.5 / res) * (res / (res + 1));
const fromSubUvsToUnit = (u: number, res: number): number => (u - 0.5 / res) * (res / (res - 1));

function sunBelowHorizon(r: number, mu: number): boolean {
  return mu < -Math.sqrt(Math.max(0, (r - RG) * (r + RG))) / r;
}

function phaseRayleigh(c: number): number {
  return (3 / (16 * Math.PI)) * (1 + c * c);
}
function phaseMie(c: number, g: number): number {
  const g2 = g * g;
  return (
    ((3 / (8 * Math.PI)) * ((1 - g2) * (1 + c * c))) /
    ((2 + g2) * Math.pow(Math.max(1 + g2 - 2 * g * c, 1e-4), 1.5))
  );
}

/** Bilinear lookup with clamp-to-edge and texel-centre convention, matching the GPU sampler. */
function bilinear(table: Float32Array, w: number, h: number, u: number, v: number, out: Float64Array): void {
  const x = Math.min(w - 1, Math.max(0, u * w - 0.5));
  const y = Math.min(h - 1, Math.max(0, v * h - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  for (let c = 0; c < 3; c++) {
    const a = table[(y0 * w + x0) * 3 + c];
    const b = table[(y0 * w + x1) * 3 + c];
    const d = table[(y1 * w + x0) * 3 + c];
    const e = table[(y1 * w + x1) * 3 + c];
    out[c] = (a + (b - a) * fx) * (1 - fy) + (d + (e - d) * fx) * fy;
  }
}

function makeHalfTexture(w: number, h: number): { tex: THREE.DataTexture; data: Uint16Array } {
  const data = new Uint16Array(w * h * 4);
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  return { tex, data };
}

// Module scratch: nothing below allocates per call.
const _m = new Float64Array(5);
const _t = new Float64Array(3);
const _ms = new Float64Array(3);
const _uv = [0, 0];
const _rm = [0, 0];

export class PhysicalAtmosphere {
  /** Transmittance to the top of the atmosphere, CPU copy, rgb. */
  private readonly trans = new Float32Array(TRANS_W * TRANS_H * 3);
  /** Second-and-higher-order scattering, CPU copy, rgb. */
  private readonly ms = new Float32Array(MS_SIZE * MS_SIZE * 3);
  private readonly transTex: THREE.DataTexture;
  private readonly transData: Uint16Array;
  private readonly msTex: THREE.DataTexture;
  private readonly msData: Uint16Array;
  private haze = -1;
  private readonly cache = new Map<number, { trans: Float32Array; ms: Float32Array }>();

  readonly uHaze = uniform(1);
  readonly uMieG = uniform(0.84);
  /** Sun direction (world, unit). */
  readonly uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  /** The sky-view LUT, a `rtt` node; sampled through `skyView`. */
  private readonly skyViewNode: any;
  private lutSunY = NaN;
  private lutHaze = NaN;
  private lutG = NaN;

  constructor() {
    const t = makeHalfTexture(TRANS_W, TRANS_H);
    this.transTex = t.tex;
    this.transData = t.data;
    const m = makeHalfTexture(MS_SIZE, MS_SIZE);
    this.msTex = m.tex;
    this.msData = m.data;
    this.skyViewNode = rtt(this.buildSkyViewNode(), SKY_W, SKY_H, { type: THREE.HalfFloatType });
    this.skyViewNode.autoUpdate = false;
  }

  get currentHaze(): number {
    return this.haze;
  }

  /**
   * Rebuilds the transmittance and multiple-scattering tables for `haze` (the
   * Mie multiplier, web-ocean's "Haze" slider: 3 is their default sea air).
   * A no-op if unchanged.
   *
   * Cached per haze value, because a preset switch is the only caller and nine
   * presets use five values. The first build of a value costs ~60 ms of main
   * thread in Chrome (measured: 110 ms at web-ocean's 64-step transmittance and
   * 8x8-direction MS, before the cuts noted in each builder), so a preset is
   * paid for once per session.
   */
  setHaze(haze: number): void {
    const h = Math.max(0, haze);
    if (h === this.haze) return;
    this.haze = h;
    this.uHaze.value = h;
    const cached = this.cache.get(h);
    if (cached) {
      this.trans.set(cached.trans);
      this.ms.set(cached.ms);
      packHalf(this.trans, this.transData);
      packHalf(this.ms, this.msData);
      this.transTex.needsUpdate = true;
      this.msTex.needsUpdate = true;
    } else {
      this.buildTransmittance(h);
      this.buildMultipleScattering(h);
      this.cache.set(h, { trans: this.trans.slice(), ms: this.ms.slice() });
    }
    this.skyViewNode.textureNeedsUpdate = true;
  }

  /** Sun direction and Mie anisotropy; marks the sky-view LUT stale only when it changed. */
  setSun(sunDir: THREE.Vector3, g: number): void {
    (this.uSunDir.value as THREE.Vector3).copy(sunDir);
    const gc = Math.min(0.9, Math.max(0.6, g));
    this.uMieG.value = gc;
    if (sunDir.y !== this.lutSunY || this.haze !== this.lutHaze || gc !== this.lutG) {
      this.lutSunY = sunDir.y;
      this.lutHaze = this.haze;
      this.lutG = gc;
      this.skyViewNode.textureNeedsUpdate = true;
    }
  }

  // -------------------------------------------------------------------------
  // CPU tables (web-ocean TRANS_LUT / MS_LUT, transcribed)
  // -------------------------------------------------------------------------

  private buildTransmittance(haze: number): void {
    // 40 steps, from web-ocean's 64: the integrand is a sum of exponentials in
    // height, and 40 steps match 64 within 1% wherever T > 0.001 (checked at
    // cos-zenith -0.01..1, altitudes 10 m, 1 km and 10 km, haze 3; the worst
    // case is the horizon ray).
    const STEPS = 40;
    for (let y = 0; y < TRANS_H; y++) {
      for (let x = 0; x < TRANS_W; x++) {
        transUVToRMu((x + 0.5) / TRANS_W, (y + 0.5) / TRANS_H, _rm);
        const r = _rm[0];
        const mu = _rm[1];
        const tMax = sphereFar(r, mu, RT);
        const dt = tMax / STEPS;
        let o0 = 0;
        let o1 = 0;
        let o2 = 0;
        for (let i = 0; i < STEPS; i++) {
          const t = (i + 0.5) * dt;
          const pr = Math.sqrt(r * r + t * t + 2 * r * mu * t);
          mediumCPU(pr - RG, haze, _m);
          o0 += _m[0] * dt;
          o1 += _m[1] * dt;
          o2 += _m[2] * dt;
        }
        const k = (y * TRANS_W + x) * 3;
        this.trans[k] = Math.exp(-o0);
        this.trans[k + 1] = Math.exp(-o1);
        this.trans[k + 2] = Math.exp(-o2);
      }
    }
    packHalf(this.trans, this.transData);
    this.transTex.needsUpdate = true;
  }

  /** Transmittance from radius `r` along cos-zenith `mu` to space, into `out` (rgb). */
  private sampleTrans(r: number, mu: number, out: Float64Array): void {
    transRMuToUV(r, mu, _uv);
    bilinear(this.trans, TRANS_W, TRANS_H, _uv[0], _uv[1], out);
  }

  private buildMultipleScattering(haze: number): void {
    // web-ocean uses 8x8 directions and 20 steps on the GPU; on the CPU that is
    // 1.3 M medium evaluations. 4x4 stratified directions and 16 steps: the
    // second-order term is a near-isotropic integral that is then divided
    // into `1 / (1 - f_ms)`, so it is smooth in both LUT axes, and it is a
    // fraction of the sky's radiance in the first place.
    const SQ = 4;
    const STEPS = 16;
    const L = new Float64Array(3);
    const F = new Float64Array(3);
    for (let iy = 0; iy < MS_SIZE; iy++) {
      for (let ix = 0; ix < MS_SIZE; ix++) {
        const u = fromSubUvsToUnit((ix + 0.5) / MS_SIZE, MS_SIZE);
        const v = fromSubUvsToUnit((iy + 0.5) / MS_SIZE, MS_SIZE);
        const cosSun = u * 2 - 1;
        const r = RG + Math.max(v * (RT - RG), 0.01);
        const sunX = Math.sqrt(Math.max(0, 1 - cosSun * cosSun));
        const sunY = cosSun;
        let ls0 = 0, ls1 = 0, ls2 = 0, fs0 = 0, fs1 = 0, fs2 = 0;
        for (let a = 0; a < SQ; a++) {
          for (let b = 0; b < SQ; b++) {
            const th = Math.acos(1 - (2 * (b + 0.5)) / SQ);
            const ph = (2 * Math.PI * (a + 0.5)) / SQ;
            const rdx = Math.sin(th) * Math.cos(ph);
            const rdy = Math.cos(th);
            const tTop = sphereFar(r, rdy, RT);
            const tGr = sphereNear(r, rdy, RG);
            const hitG = tGr > 0;
            const tMax = hitG ? tGr : tTop;
            const dt = tMax / STEPS;
            let T0 = 1, T1 = 1, T2 = 1;
            L[0] = L[1] = L[2] = 0;
            F[0] = F[1] = F[2] = 0;
            for (let s = 0; s < STEPS; s++) {
              const t = (s + 0.5) * dt;
              const px = rdx * t;
              const py = r + rdy * t;
              const pz = Math.sin(th) * Math.sin(ph) * t;
              const pr = Math.sqrt(px * px + py * py + pz * pz);
              mediumCPU(pr - RG, haze, _m);
              const smu = (px * sunX + py * sunY) / pr;
              if (sunBelowHorizon(pr, smu)) _t[0] = _t[1] = _t[2] = 0;
              else this.sampleTrans(pr, smu, _t);
              for (let c = 0; c < 3; c++) {
                const scat = RAY_S[c] * _m[3] + _m[4];
                const ext = Math.max(_m[c], 1e-9);
                const Ts = Math.exp(-_m[c] * dt);
                const Tc = c === 0 ? T0 : c === 1 ? T1 : T2;
                const S = (scat * _t[c]) / (4 * Math.PI);
                L[c] += (Tc * (S - S * Ts)) / ext;
                F[c] += (Tc * (scat - scat * Ts)) / ext;
              }
              T0 *= Math.exp(-_m[0] * dt);
              T1 *= Math.exp(-_m[1] * dt);
              T2 *= Math.exp(-_m[2] * dt);
            }
            if (hitG) {
              // Ground bounce, albedo 0.06 (the sea), as web-ocean.
              const smu = sunY; // the ground normal at the hit is close enough to vertical over 100 km
              this.sampleTrans(RG, smu, _t);
              const k = (Math.max(smu, 0) * 0.06) / Math.PI;
              L[0] += T0 * _t[0] * k;
              L[1] += T1 * _t[1] * k;
              L[2] += T2 * _t[2] * k;
            }
            ls0 += L[0]; ls1 += L[1]; ls2 += L[2];
            fs0 += F[0]; fs1 += F[1]; fs2 += F[2];
          }
        }
        const inv = 1 / (SQ * SQ);
        const k = (iy * MS_SIZE + ix) * 3;
        this.ms[k] = (ls0 * inv) / (1 - fs0 * inv);
        this.ms[k + 1] = (ls1 * inv) / (1 - fs1 * inv);
        this.ms[k + 2] = (ls2 * inv) / (1 - fs2 * inv);
      }
    }
    packHalf(this.ms, this.msData);
    this.msTex.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // CPU queries
  // -------------------------------------------------------------------------

  /** Transmittance from the viewer toward cos-zenith `mu`, rgb into `out`. */
  transmittance(mu: number, out: THREE.Color): THREE.Color {
    if (sunBelowHorizon(CAM_R, mu)) return out.setRGB(0, 0, 0);
    this.sampleTrans(CAM_R, mu, _t);
    return out.setRGB(_t[0], _t[1], _t[2]);
  }

  /**
   * Clear-sky radiance toward `dir` for unit solar illuminance — the same
   * integral the sky-view LUT stores, marched directly for one direction.
   */
  radiance(dir: THREE.Vector3, sun: THREE.Vector3, g: number, out: THREE.Color): THREE.Color {
    const dl = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const dx = dir.x / dl;
    const dy = Math.max(dir.y / dl, 0.0005);
    const dz = dir.z / dl;
    const r = CAM_R;
    const tGr = sphereNear(r, dy, RG);
    const tMax = tGr > 0 ? tGr : sphereFar(r, dy, RT);
    const cosT = dx * sun.x + dy * sun.y + dz * sun.z;
    const pR = phaseRayleigh(cosT);
    const pM = phaseMie(cosT, Math.min(0.9, Math.max(0.6, g)));
    let L0 = 0, L1 = 0, L2 = 0, T0 = 1, T1 = 1, T2 = 1;
    let tPrev = 0;
    for (let i = 0; i < CPU_STEPS; i++) {
      const fa = (i + 1) / CPU_STEPS;
      const tn = tMax * fa * fa;
      const dt = tn - tPrev;
      const t = tPrev + dt * 0.3;
      tPrev = tn;
      const px = dx * t;
      const py = r + dy * t;
      const pz = dz * t;
      const pr = Math.sqrt(px * px + py * py + pz * pz);
      const h = pr - RG;
      mediumCPU(h, this.haze, _m);
      const pmu = (px * sun.x + py * sun.y + pz * sun.z) / pr;
      if (sunBelowHorizon(pr, pmu)) _t[0] = _t[1] = _t[2] = 0;
      else this.sampleTrans(pr, pmu, _t);
      bilinear(
        this.ms,
        MS_SIZE,
        MS_SIZE,
        fromUnitToSubUvs(pmu * 0.5 + 0.5, MS_SIZE),
        fromUnitToSubUvs(Math.min(1, Math.max(0, h / (RT - RG))), MS_SIZE),
        _ms,
      );
      for (let c = 0; c < 3; c++) {
        const rayS = RAY_S[c] * _m[3];
        const S = _t[c] * (rayS * pR + _m[4] * pM) + _ms[c] * (rayS + _m[4]);
        const ext = Math.max(_m[c], 1e-9);
        const Ts = Math.exp(-_m[c] * dt);
        if (c === 0) { L0 += (T0 * (S - S * Ts)) / ext; T0 *= Ts; }
        else if (c === 1) { L1 += (T1 * (S - S * Ts)) / ext; T1 *= Ts; }
        else { L2 += (T2 * (S - S * Ts)) / ext; T2 *= Ts; }
      }
    }
    return out.setRGB(L0, L1, L2);
  }

  // -------------------------------------------------------------------------
  // GPU (TSL)
  // -------------------------------------------------------------------------

  /** Transmittance LUT lookup, (r km, cos zenith) -> rgb. */
  private transNode(r: any, mu: any): any {
    const rho = sqrt(max(r.mul(r).sub(RG * RG), 0));
    const disc = r.mul(r).mul(mu.mul(mu).sub(1)).add(RT * RT);
    const d = max(r.negate().mul(mu).add(sqrt(max(disc, 0))), 0);
    const dMin = float(RT).sub(r);
    const dMax = rho.add(H_TOP);
    const u = d.sub(dMin).div(dMax.sub(dMin));
    const v = rho.div(H_TOP);
    return texture(this.transTex, vec2(u, v)).level(float(0)).rgb;
  }

  /** web-ocean SKYVIEW_LUT, as a fragment over the LUT's own uv. */
  private buildSkyViewNode(): any {
    return Fn(() => {
      const q = uv();
      const ux = q.x.sub(0.5 / SKY_W).mul(SKY_W / (SKY_W - 1));
      const uy = q.y.sub(0.5 / SKY_H).mul(SKY_H / (SKY_H - 1));
      const haze = this.uHaze;
      const r = float(CAM_R);
      const vHorizon = Math.sqrt(Math.max(0, (CAM_R - RG) * (CAM_R + RG)));
      const beta = Math.acos(Math.min(1, Math.max(-1, vHorizon / CAM_R)));
      const zha = Math.PI - beta;
      const loC = float(1).sub(uy.mul(2));
      const lo = float(1).sub(loC.mul(loC)).mul(zha);
      const hiC = uy.mul(2).sub(1);
      const hi = hiC.mul(hiC).mul(beta).add(zha);
      const vza = select(uy.lessThan(0.5), lo, hi);
      const vzc = cos(vza);
      const vzs = sin(vza);
      const cx = ux.mul(ux);
      const lvc = cx.mul(2).sub(1).negate();
      const lvs = sqrt(max(float(1).sub(lvc.mul(lvc)), 0));
      const rd = vec3(vzs.mul(lvc), vzc, vzs.mul(lvs)).toVar('svRd');
      const smu = this.uSunDir.y;
      const sun = vec3(sqrt(max(float(1).sub(smu.mul(smu)), 0)), smu, 0).toVar('svSun');

      // Ray vs ground (near) / top (far), from radius r along rd.y.
      const b = r.mul(rd.y);
      const dG = b.mul(b).sub(CAM_R * CAM_R - RG * RG);
      const tG = b.negate().sub(sqrt(max(dG, 0)));
      const dT = b.mul(b).sub(CAM_R * CAM_R - RT * RT);
      const tT = b.negate().add(sqrt(max(dT, 0)));
      const tMax = select(dG.greaterThanEqual(0).and(tG.greaterThan(0)), tG, tT).toVar('svTMax');

      const cosT = dot(rd, sun);
      const pR = cosT.mul(cosT).add(1).mul(3 / (16 * Math.PI));
      const g = this.uMieG;
      const g2 = g.mul(g);
      const pM = float(3 / (8 * Math.PI))
        .mul(float(1).sub(g2).mul(cosT.mul(cosT).add(1)))
        .div(g2.add(2).mul(pow(max(g2.add(1).sub(g.mul(cosT).mul(2)), 1e-4), 1.5)))
        .toVar('svPM');

      const L = vec3(0).toVar('svL');
      const T = vec3(1).toVar('svT');
      const tPrev = float(0).toVar('svTPrev');
      Loop(SKY_STEPS, ({ i }: any) => {
        const fa = float(i).add(1).div(SKY_STEPS);
        // `toVar` before `tPrev` is reassigned: TSL emits an expression where it
        // is first *used*, so an inline `tn - tPrev` read after the assignment
        // is zero (measured: an all-black LUT).
        const tn = tMax.mul(fa).mul(fa).toVar('svTn');
        const dt = tn.sub(tPrev).toVar('svDt');
        const t = tPrev.add(dt.mul(0.3)).toVar('svT0');
        tPrev.assign(tn);
        const p = vec3(0, r, 0).add(rd.mul(t));
        const pr = length(p);
        const h = max(pr.sub(RG), 0);
        const dR = exp(h.div(-8.0));
        const dM = exp(h.div(-1.2));
        const dO = max(float(1).sub(h.sub(25).abs().div(15)), 0);
        const rayS = vec3(RAY_S[0], RAY_S[1], RAY_S[2]).mul(dR);
        const mieS = dM.mul(MIE_S).mul(haze);
        const ext = rayS.add(dM.mul(MIE_E).mul(haze)).add(vec3(OZO_A[0], OZO_A[1], OZO_A[2]).mul(dO));
        const pmu = dot(p, sun).div(pr);
        const below = pmu.lessThan(sqrt(max(pr.mul(pr).sub(RG * RG), 0)).div(pr).negate());
        const sunT = select(below, vec3(0), this.transNode(pr, pmu));
        const msUV = vec2(
          pmu.mul(0.5).add(0.5).add(0.5 / MS_SIZE).mul(MS_SIZE / (MS_SIZE + 1)),
          clamp(h.div(RT - RG), 0, 1).add(0.5 / MS_SIZE).mul(MS_SIZE / (MS_SIZE + 1)),
        );
        const ms = texture(this.msTex, msUV).level(float(0)).rgb;
        const S = sunT.mul(rayS.mul(pR).add(mieS.mul(pM))).add(ms.mul(rayS.add(mieS)));
        const Ts = exp(ext.mul(dt).negate());
        L.addAssign(T.mul(S.sub(S.mul(Ts))).div(max(ext, vec3(1e-9))));
        T.mulAssign(Ts);
      });
      return vec4(L, 1);
    })();
  }

  /**
   * Clear-sky radiance toward `dir` (unit, world) for unit solar illuminance,
   * from the sky-view LUT (web-ocean `skyViewUV` / `skyAtmo`).
   */
  skyView(dir: any): any {
    const vHorizon = Math.sqrt(Math.max(0, (CAM_R - RG) * (CAM_R + RG)));
    const cosBeta = vHorizon / CAM_R;
    const beta = Math.acos(Math.min(1, Math.max(-1, cosBeta)));
    const zha = Math.PI - beta;
    const vzc = clamp(dir.y, -1, 1);
    const above = float(1).sub(sqrt(max(float(1).sub(acos(vzc).div(zha)), 0))).mul(0.5);
    const belowV = sqrt(max(acos(vzc).sub(zha).div(beta), 0)).mul(0.5).add(0.5);
    const v = select(vzc.greaterThanEqual(-cosBeta), above, belowV);
    const sun = this.uSunDir;
    const dl = length(dir.xz);
    const sl = length(sun.xz);
    const lvc = select(
      dl.greaterThan(1e-6).and(sl.greaterThan(1e-6)),
      clamp(dot(dir.xz, sun.xz).div(dl.mul(sl).max(1e-6)), -1, 1),
      float(1),
    );
    const u = sqrt(max(lvc.mul(-0.5).add(0.5), 0));
    const suv = vec2(
      u.add(0.5 / SKY_W).mul(SKY_W / (SKY_W + 1)),
      v.add(0.5 / SKY_H).mul(SKY_H / (SKY_H + 1)),
    );
    // Not `.sample(suv)`: that returns a plain TextureNode clone, and the
    // `rtt` node itself — whose `updateBefore` renders the LUT — would then not
    // be in any graph, so the LUT was never drawn (measured: a black sky). The
    // node is sampled once, by the dome, so it takes the lookup uv directly.
    this.skyViewNode.uvNode = suv;
    return this.skyViewNode.rgb;
  }

  /**
   * The sun's disc, limb darkened (web-ocean `sunDisk`): the limb-darkening
   * profile times the transmittance along `dir`, without the disc's radiance,
   * which the caller sets (`Atmosphere` caps it for display). `angularRadius`
   * is the drawn radius in radians; web-ocean draws the real `SUN_ANG_R`. Its
   * edge is anti-aliased over one pixel of angle via `fwidth` rather than
   * web-ocean's screen-size uniform, so the same node serves every camera (and
   * the cube capture, which hides the disc anyway).
   */
  sunDisc(dir: any, angularRadius: number = SUN_ANG_R): any {
    const c = dot(dir, this.uSunDir);
    const ang = sqrt(max(float(1).sub(c).mul(2), 0));
    const pix = fwidth(ang).max(1e-5).mul(0.75);
    const edge = float(1).sub(smoothstep(float(angularRadius).sub(pix), float(angularRadius).add(pix), ang));
    const x = min(ang.div(angularRadius), 1);
    const mu = sqrt(max(float(1).sub(x.mul(x)), 0));
    const limb = vec3(1).sub(vec3(0.397, 0.503, 0.652).mul(float(1).sub(mu)));
    return limb.mul(edge).mul(this.transNode(float(CAM_R), max(dir.y, -0.01)));
  }

  /**
   * The solar aureole's angular profile outside the disc of `discRadius`,
   * times the transmittance along `dir` (so the halo reddens toward the
   * horizon the way the disc does). 1 at the disc's edge; the caller sets its
   * level. Two exponentials in angle from the limb: a tight one of `core`
   * radians and a broad one of `wide`, weighted `wideShare`.
   *
   * The shape follows measured solar aureoles, whose radiance falls steeply
   * (roughly as the inverse square of angle) over the first degree or two and
   * then slowly out to ten or more: aerosol forward scattering, the same Mie
   * lobe the sky-view LUT integrates but resolved at a scale the LUT's 1-degree
   * texels cannot carry.
   */
  sunAureole(dir: any, discRadius: number, core: any, wide: any, wideShare: any): any {
    const c = dot(dir, this.uSunDir);
    const ang = sqrt(max(float(1).sub(c).mul(2), 0));
    const beyond = ang.sub(discRadius).max(0);
    const profile = mix(exp(beyond.div(core).negate()), exp(beyond.div(wide).negate()), wideShare);
    return this.transNode(float(CAM_R), max(dir.y, -0.01)).mul(profile);
  }

  dispose(): void {
    this.transTex.dispose();
    this.msTex.dispose();
    this.skyViewNode.renderTarget?.dispose?.();
  }
}

function packHalf(src: Float32Array, dst: Uint16Array): void {
  const n = src.length / 3;
  const one = THREE.DataUtils.toHalfFloat(1);
  for (let i = 0; i < n; i++) {
    dst[i * 4] = THREE.DataUtils.toHalfFloat(src[i * 3]);
    dst[i * 4 + 1] = THREE.DataUtils.toHalfFloat(src[i * 3 + 1]);
    dst[i * 4 + 2] = THREE.DataUtils.toHalfFloat(src[i * 3 + 2]);
    dst[i * 4 + 3] = one;
  }
}
