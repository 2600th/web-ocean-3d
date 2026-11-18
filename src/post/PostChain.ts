import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  dot,
  float,
  floor,
  fract,
  mix,
  rtt,
  screenCoordinate,
  sin,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import TRAANode, { traa } from 'three/addons/tsl/display/TRAANode.js';
import type { QualityTier } from '../core/QualityManager';
import type { ShadowVolumeSource } from '../lighting/SunShadows';
import { AmbientOcclusion } from './AmbientOcclusion';
import { AutoExposure } from './AutoExposure';
import { ContactShadows, type ContactShadowSettings } from './ContactShadows';
import { GlobalIllumination, SSGI_REQUESTED, type GiSettings } from './GlobalIllumination';
import { PostResolve } from './PostResolve';
import { DeterministicTAAU } from './TemporalUpscale';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * What each tier spends on the post stages this file owns.
 *
 * Every field is a uniform write or a resolution change at runtime — nothing
 * here can recompile a shader, which is the property a live tier change needs.
 * Costs are measured, not estimated; see `docs/AAA_PLAN.md` section N.
 */
export interface PostTierSettings {
  /**
   * Scene-pass resolution as a fraction of the drawing buffer. Below 1 the
   * temporal resolve brings it back to the drawing buffer (TAAU, or TRAA in a
   * session that booted at native; see `PostChain.buildResolve`); at 1 it is
   * a plain temporal AA. The rest of the chain always runs at the drawing
   * buffer.
   */
  renderScale: number;
  /**
   * `renderScale` when the drawing buffer is dense — a pixel ratio above
   * `HI_DPI_RATIO`, a Retina panel or a 4K monitor at 150-200% — where a
   * native-resolution scene pass shades two to four times the pixels the eye
   * can separate. See `PostChain.effectiveRenderScale`.
   */
  hiDpiRenderScale: number;
  ao: { enabled: boolean; resolutionScale: number; samples: number };
  /** Sun contact shadows; resolution follows the scene pass. */
  contactShadows: Omit<ContactShadowSettings, 'resolutionScale'>;
  /**
   * SSGI; replaces GTAO where enabled. `resolutionScale` is of the scene pass.
   * Only in a session booted with `?gi=1` (see `SSGI_REQUESTED`); elsewhere
   * these settings have nothing to drive and GTAO keeps the tier's AO.
   */
  gi: GiSettings;
  /** Motion-blur taps; 0 is off. */
  motionBlurSamples: number;
  /** RCAS lobe scale, 0..1. */
  sharpen: number;
  /**
   * Film-grain amplitude in display units; 0 is off. Off at every tier — see
   * `PostChain.applyGrain` — and kept as a field so `setOverride('grain', true)`
   * (which restores the old 0.012) still has something to A/B against.
   */
  grain: number;
  /** Lateral CA at the frame corner, UV units; 0 is off. */
  aberration: number;
  autoExposure: boolean;
}

export const POST_TIERS: Record<QualityTier, PostTierSettings> = {
  low: {
    renderScale: 0.67,
    hiDpiRenderScale: 0.67,
    ao: { enabled: false, resolutionScale: 0.5, samples: 8 },
    contactShadows: { enabled: false, maxDistance: 0.35, quality: 0.4 },
    gi: { enabled: false, slices: 1, steps: 8, resolutionScale: 0.5 },
    motionBlurSamples: 0,
    sharpen: 0.5,
    grain: 0,
    aberration: 0,
    autoExposure: true,
  },
  medium: {
    renderScale: 0.77,
    hiDpiRenderScale: 0.77,
    ao: { enabled: true, resolutionScale: 0.5, samples: 8 },
    contactShadows: { enabled: false, maxDistance: 0.35, quality: 0.4 },
    gi: { enabled: false, slices: 1, steps: 8, resolutionScale: 0.5 },
    motionBlurSamples: 8,
    sharpen: 0.4,
    grain: 0,
    aberration: 0,
    autoExposure: true,
  },
  high: {
    renderScale: 1,
    hiDpiRenderScale: 0.75,
    ao: { enabled: true, resolutionScale: 0.5, samples: 12 },
    contactShadows: { enabled: true, maxDistance: 0.35, quality: 0.5 },
    gi: { enabled: false, slices: 1, steps: 12, resolutionScale: 0.5 },
    motionBlurSamples: 10,
    sharpen: 0.3,
    grain: 0,
    aberration: 0.001,
    autoExposure: true,
  },
  ultra: {
    renderScale: 1,
    hiDpiRenderScale: 1,
    ao: { enabled: true, resolutionScale: 0.75, samples: 16 },
    contactShadows: { enabled: true, maxDistance: 0.4, quality: 0.5 },
    gi: { enabled: true, slices: 2, steps: 8, resolutionScale: 1 },
    motionBlurSamples: 12,
    sharpen: 0.3,
    grain: 0,
    aberration: 0.001,
    autoExposure: true,
  },
  max: {
    renderScale: 1,
    hiDpiRenderScale: 1,
    ao: { enabled: true, resolutionScale: 1, samples: 16 },
    contactShadows: { enabled: true, maxDistance: 0.5, quality: 0.6 },
    gi: { enabled: true, slices: 3, steps: 12, resolutionScale: 1 },
    motionBlurSamples: 14,
    sharpen: 0.3,
    grain: 0,
    aberration: 0.001,
    autoExposure: true,
  },
};

/** Shutter angle, degrees. 180° is the film convention: half the frame interval. */
const SHUTTER_ANGLE = 180;

/** Pixel ratio above which a tier's `hiDpiRenderScale` replaces its `renderScale`. */
const HI_DPI_RATIO = 1.5;

/**
 * `?aa=taau` / `?aa=traa` force the temporal resolve, for A/B; otherwise it
 * follows the boot tier's render scale (see `PostChain.buildResolve`).
 */
const FORCED_TEMPORAL: 'taau' | 'traa' | null = (() => {
  if (typeof window === 'undefined') return null;
  const aa = new URLSearchParams(window.location.search).get('aa');
  return aa === 'taau' || aa === 'traa' ? aa : null;
})();

const _drawingBuffer = new THREE.Vector2();
/**
 * TRAA that keeps its disocclusion test working below native resolution.
 *
 * TRAA sizes its history to its input, so a scene pass rendered at 0.77
 * resolves at 0.77 and the chain upsamples it from there. What it does not do
 * is keep the previous depth: its copy is guarded by "history matches the
 * drawing buffer", which is never true below native, so the depth it rejects
 * history against would stay whatever the last native frame left — or empty on
 * a session that booted on Low, which rejects everything and silently turns
 * the AA off. This performs the same copy against the size that actually
 * matters, the input's.
 */
class ScaledTRAA extends (TRAANode as unknown as new (...args: any[]) => any) {
  /** Written with this frame's jitter as a UV offset; see `PostChain.jitterUv`. */
  jitterUv: { value: THREE.Vector2 } | null = null;

  constructor(...args: any[]) {
    super(...args);
    // Resolve once per *pipeline render*, not once per animation frame.
    //
    // Stock TRAA is a FRAME node, and three's frame counter advances on the
    // renderer's own rAF loop, not on renders. The deterministic harness
    // renders several times between rAF ticks — settle steps, warm-up
    // captures — so a FRAME node resolves for some of those renders and is
    // skipped for others, depending on how the browser happened to schedule
    // them. The jitter, meanwhile, advances on every render. RENDER makes the
    // resolve follow the same count the jitter does. The same change is made
    // to GTAO's rotation and to eye adaptation for the same reason.
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
  }

  /**
   * Publishes the jitter TRAA is about to put on the camera, so the colour
   * TRAA reads can be taken back to the unjittered grid (`AmbientOcclusion.build`).
   *
   * Stock TRAA samples the current frame where it landed, half a pixel either
   * way of the true position. That only cancels while the history carries the
   * pixel; once the camera moves, the resolve gives the current frame weight —
   * the subpixel term alone adds up to 0.25 — and variance clipping pulls the
   * history into the jittered 3x3 box, so the Halton offsets reach the screen.
   * Measured on the horizon (`imgshift.mjs`, High, 1920x945): with the camera
   * still, 18% of the vertical jitter reached the output (0.09 px rms per
   * frame); turning at 0.25 rad/s, 73% (0.36 px rms) — every edge in the frame
   * stepping up and down by a fraction of a pixel, frame to frame, in no
   * rhythm, which is the whole-frame jitter seen whenever the view moves.
   * Reading the current frame at the unjittered position takes the offsets
   * out of both the current sample and the clipping box, as reference TAA
   * implementations do (Karis 2014; Playdead's `_JitterUV`). Same harness,
   * after: 10% still, 8% turning (0.10 px rms, the rest being the scene's own
   * motion), with the horizon's edge as steep as before.
   */
  setViewOffset(width: number, height: number): void {
    super.setViewOffset(width, height);
    if (this.jitterUv === null) return;
    const index = (this as any)._jitterIndex as number;
    // TRAA's own sequence: Halton (2, 3) at index + 1, centred on the pixel.
    // A view offset of +d pixels moves the image content by -d, in screen
    // space (y down), which is also the UV convention of the passes here.
    this.jitterUv.value.set((halton(index + 1, 2) - 0.5) / width, (halton(index + 1, 3) - 0.5) / height);
  }

  clearViewOffset(): void {
    super.clearViewOffset();
    this.jitterUv?.value.set(0, 0);
  }

  updateBefore(frame: any): any {
    const self = this as any;
    const renderer = frame.renderer;
    const beauty = self.beautyNode.isRTTNode ? self.beautyNode.renderTarget : self.beautyNode.passNode.renderTarget;
    const restart =
      self._historyRenderTarget.width !== beauty.texture.width ||
      self._historyRenderTarget.height !== beauty.texture.height;
    const result = super.updateBefore(frame);
    // A restart (a resize, or `resetTemporal`) seeds the history with this frame
    // and then resolves against it anyway. That resolve still reprojects through
    // the velocity buffer, and on this one frame the velocities describe the
    // move from wherever the camera last drew — after a cut, the old shot. The
    // history is sampled at the wrong place, clamped, and blended in, and a
    // TRAA's history decays slowly enough that the error was still flipping
    // 8-bit values 92 frames later: the first capture after any camera jump
    // differed from the next one. On a restart the frame *is* the resolve.
    if (restart) {
      renderer.copyTextureToTexture(beauty.texture, self._resolveRenderTarget.texture);
      renderer.copyTextureToTexture(beauty.texture, self._historyRenderTarget.texture);
    }
    renderer.getDrawingBufferSize(_drawingBuffer);
    const history = self._historyRenderTarget;
    if (history.width === _drawingBuffer.x && history.height === _drawingBuffer.y) return result;
    const depth = self.depthNode.value;
    const image = depth?.image;
    if (image && image.width === history.width && image.height === history.height) {
      renderer.copyTextureToTexture(depth, history.depthTexture);
      self._previousDepthNode.value = history.depthTexture;
    }
    return result;
  }
}

/** Radical inverse in `base`: TRAA's jitter sequence, which it keeps private. */
function halton(index: number, base: number): number {
  let fraction = 1;
  let result = 0;
  for (let i = index; i > 0; i = Math.floor(i / base)) {
    fraction /= base;
    result += fraction * (i % base);
  }
  return result;
}

function hash21(p: any): any {
  return fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));
}

export interface PostChainOptions {
  renderer: THREE.WebGPURenderer;
  camera: THREE.PerspectiveCamera;
  /** The scene pass. With temporal AA it must carry an `{ output, velocity }` MRT. */
  scenePass: any;
  /** The scene pass's depth texture node. */
  depth: any;
  /** Temporal resolve available (WebGPU with the velocity MRT). */
  temporal: boolean;
  /** The tier the session boots at; picks the temporal resolve. */
  bootTier: QualityTier;
  /** The key light (sun or moon), the fill, and the near shadow cascades, for the contact shadows. */
  sun: THREE.DirectionalLight;
  ambient: THREE.HemisphereLight;
  sunCascades: ShadowVolumeSource[];
}

/**
 * The parts of the post chain added in the AAA pass, kept out of `main.ts`.
 *
 * `main.ts` still owns the order of the stages it always had — underwater, fog,
 * depth of field, bloom, flare, grade, lens rain, output — and calls in here at
 * four seams:
 *
 * 1. `buildResolve` replaces the bare TRAA call: sun contact shadows, SSGI
 *    (bounce and its AO) or GTAO are composited into the scene colour, the
 *    temporal resolve runs (TAAU or TRAA, either of which also brings a scene
 *    pass rendered below native back up), and the resolved image gets sharpening,
 *    motion blur and lateral CA in one pass.
 * 2. `buildExposure` meters the fogged, underwater-composited scene and
 *    `applyExposure` applies the adapted gain ahead of the bloom.
 * 3. `applyGrain` adds film grain in display space, before the dither.
 * 4. `setTier` / `update` / `resetTemporal` / `setDeterministic` from the
 *    corresponding places in the app.
 *
 * Every effect has an A/B toggle on `__ocean.post.chain`; with temporal AA off
 * (`?taa=0`) the chain gets auto-exposure and grain only, because everything
 * else hangs off the velocity MRT that only the temporal scene pass carries.
 */
export class PostChain {
  readonly resolve = new PostResolve();
  ao: AmbientOcclusion | null = null;
  contactShadows: ContactShadows | null = null;
  gi: GlobalIllumination | null = null;
  exposure: AutoExposure | null = null;

  private readonly options: PostChainOptions;
  private temporalNode: any = null;
  private aoComposite: any = null;
  private tier: PostTierSettings = POST_TIERS.high;
  private tierId: QualityTier = 'high';
  private readonly reducedMotionQuery: MediaQueryList | null;
  private deterministic = false;
  /** Opt-in for captures that want the blur in frame (a whip-pan test, say). */
  private motionBlurInCapture = false;
  /** A/B overrides; `null` follows the tier. */
  readonly overrides: {
    ao: boolean | null;
    motionBlur: boolean | null;
    sharpen: boolean | null;
    grain: boolean | null;
    aberration: boolean | null;
    autoExposure: boolean | null;
    renderScale: number | null;
    contactShadows: boolean | null;
    gi: boolean | null;
  } = {
    contactShadows: null,
    gi: null,
    ao: null,
    motionBlur: null,
    sharpen: null,
    grain: null,
    aberration: null,
    autoExposure: null,
    renderScale: null,
  };

  /**
   * `?post=0`: the chain as it was before this file existed — TRAA straight off
   * the scene pass, no AO, no resolve pass, no adaptation, no grain. The
   * fixed-cost half of the A/B in `docs/AAA_PLAN.md` section N, and a kill
   * switch.
   */
  private readonly legacy =
    typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('post') === '0';

  /** This frame's TRAA jitter as a UV offset (screen space, y down). */
  private readonly jitterUv = uniform(new THREE.Vector2());
  /** `?unjitter=0`: the TRAA path reads its input where it landed, for A/B. */
  private readonly unjitterAllowed =
    typeof window === 'undefined' || new URLSearchParams(window.location.search).get('unjitter') !== '0';
  /** Which resolve this session built; see `buildResolve`. */
  private temporalMode: 'taau' | 'traa' = 'traa';

  private readonly uGrain = uniform(0);
  private readonly uGrainSeed = uniform(0);
  private grainFrame = 0;
  private lastDt = 1 / 60;

  constructor(options: PostChainOptions) {
    this.options = options;
    this.reducedMotionQuery =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : null;
    this.reducedMotionQuery?.addEventListener?.('change', this.onMotionPreference);
  }

  private readonly onMotionPreference = (): void => this.applyTier();

  /** Whether the visitor asked for reduced motion. Live: follows the OS setting. */
  get reducedMotion(): boolean {
    return this.reducedMotionQuery?.matches ?? false;
  }

  /**
   * The scene-referred image the rest of the chain starts from.
   *
   * Without temporal AA this is the scene pass's own colour and nothing here
   * runs. With it: occluded colour → temporal resolve → sharpen/blur/CA.
   */
  buildResolve(): any {
    const { scenePass, depth, camera, temporal } = this.options;
    if (!temporal) return scenePass.getTextureNode();
    if (this.legacy) {
      this.temporalNode = traa(
        scenePass.getTextureNode('output'),
        depth,
        scenePass.getTextureNode('velocity'),
        camera,
      );
      return this.temporalNode.getTextureNode();
    }

    const color = scenePass.getTextureNode('output');
    const velocity = scenePass.getTextureNode('velocity');
    const normal = scenePass.getTextureNode('normal');

    // The temporal resolve, chosen once: one or the other is in the graph,
    // never both, because each drives the camera's jitter and they disagree
    // on its units. TAAU where the boot tier renders below native, TRAA where
    // it renders at native — each is the better of the two there; see the
    // comparison at the end of this function. A live tier change across
    // native keeps the session's choice, and both work at any scale.
    this.tierId = this.options.bootTier;
    this.tier = POST_TIERS[this.tierId];
    this.temporalMode = FORCED_TEMPORAL ?? (this.effectiveRenderScale() < 1 ? 'taau' : 'traa');
    const unjitter = this.temporalMode === 'traa' && this.unjitterAllowed;

    this.ao = new AmbientOcclusion(depth, camera, normal);
    const { sun, ambient, sunCascades } = this.options;
    this.contactShadows = new ContactShadows(depth, camera, sun, ambient, sunCascades);
    const contact = this.contactShadows;
    // SSGI only when the session asked for it at boot; see `SSGI_REQUESTED`.
    let gi: GlobalIllumination | null = null;
    let albedo: any = null;
    if (SSGI_REQUESTED) {
      // RGBA8, so the four targets fit in 32 bytes a sample; see `sceneMrt`.
      scenePass.getTexture('albedo').type = THREE.UnsignedByteType;
      albedo = scenePass.getTextureNode('albedo');
      gi = this.gi = new GlobalIllumination(color, depth, normal, camera);
    }
    // Built at boot, at every tier with the temporal path, and switched by
    // uniform and by skipping their passes; see the header.
    const lighting = (p: any, rgb: any, mask: { solid: any; field: any; contact: any }): any => {
      // The contact shadow takes only the sun's share; see `ContactShadows`.
      // Not on cloth: see `SurfaceLighting`.
      const shadowed = rgb.mul(mix(float(1), contact.factor(p, normal), mask.contact));
      if (gi === null) return shadowed;
      // `colour·AO + albedo·GI`, three's composite for the node.
      const occlusion = mix(float(1), gi.ao.sample(p).r, gi.uAoStrength.mul(mask.field));
      const bounce = albedo.sample(p).rgb.mul(gi.gi.sample(p).rgb).mul(gi.uGiStrength.mul(mask.field));
      return shadowed.mul(occlusion).add(bounce);
    };
    // Resolved to a texture explicitly rather than by the resolve's own
    // `convertToTexture`, because its resolution has to follow the scene pass
    // when the tier renders below native — TAAU sizes its input from this
    // target, and an RTT left at the drawing buffer would tell it there was
    // nothing to upscale.
    // On the TRAA path it is read at the unjittered position, so TRAA's input
    // stops carrying the jitter; see `ScaledTRAA.setViewOffset`. `?unjitter=0`
    // keeps the stock behaviour, for A/B.
    this.aoComposite = rtt(this.ao.build(color, velocity, unjitter ? this.jitterUv : null, lighting));

    // Measured on aaa-pass at 1280x720 against the TRAA path (mean |Laplacian|
    // of luma on fixed crops, deterministic captures, contact shadows off):
    //
    // - **Below native, TAAU.** At Medium's 0.77, TAAU with the Catmull-Rom
    //   reconstruction (`DeterministicTAAU`) kept the shrouds as continuous
    //   lines where TRAA's bilinear upsample broke them up, and was sharper
    //   on the ship-and-island (+8%), beach (+40%) and canopy (+53%) crops;
    //   the deck gratings came out 12% under. The dark specks r185's TAAU
    //   left through the canopy's alpha-tested edges did not appear; r187
    //   gates its thin-feature lock on a two-sided depth test, the likely fix.
    // - **At native, TRAA.** TAAU was 18-34% under on the deck and ship crops
    //   with either filter (7% over on the beach). The likely reason: TRAA's
    //   input is a Catmull-Rom resample onto the unjittered grid
    //   (`ScaledTRAA.setViewOffset`) and its history clip is built on that,
    //   where TAAU clips to the raw jittered 3x3, and the one-pixel highlights
    //   along the deck gratings came out dulled.
    if (this.temporalMode === 'taau') {
      // No unjittered read of the input here, unlike TRAA: TAAU weighs each
      // tap by its distance from the output pixel *including* the jitter, so
      // a jittered sample already counts where it was rendered. Moving the
      // input back onto the grid first would apply the offset twice.
      this.temporalNode = new DeterministicTAAU(this.aoComposite, depth, velocity, camera);
    } else {
      // TRAA resolves into a history sized to its input and samples it by UV,
      // so below native the chain upsamples the resolved image from there.
      this.temporalNode = new ScaledTRAA(this.aoComposite, depth, velocity, camera);
      if (unjitter) this.temporalNode.jitterUv = this.jitterUv;
    }

    return rtt(this.resolve.build(this.temporalNode.getTextureNode(), velocity));
  }

  /** Below-native rendering needs the temporal resolve to bring it back up. */
  get canUpscale(): boolean {
    return this.temporalNode !== null && !this.legacy;
  }

  /**
   * Meters `source` and returns nothing; call `applyExposure` where the gain
   * belongs.
   *
   * @param source Linear HDR scene as a texture node, *before* the gain.
   */
  buildExposure(source: any): void {
    if (this.legacy) return;
    this.exposure = new AutoExposure(source);
  }

  /** `node × adapted gain`, alpha untouched. A pass-through if never metered. */
  applyExposure(node: any): any {
    if (this.exposure === null) return node;
    const gain = this.exposure.gain;
    const v = vec4(node);
    return vec4(v.rgb.mul(gain), v.a);
  }

  /**
   * Film grain, in display space and before the dither.
   *
   * Display space because grain is a property of the recording medium, not of
   * the light: its visibility follows the *picture's* tones — strongest in the
   * mid-tones, fading in deep shadow and toward white — and that is a curve on
   * display values. It is value noise at about a pixel and a half rather than
   * per-pixel white noise, which reads as sensor noise instead of grain.
   *
   * Static in captures and under reduced motion; otherwise the seed advances
   * every frame, which is what makes grain read as grain and not as dirt.
   *
   * **Off at every tier.** At 0.012 it moved a mid-tone pixel by up to three
   * output levels (corrected capture, `hull-refraction`: 47% of pixels moved,
   * 9% by two or more), three times the dither, and it was what the A/B critic
   * called "grain over the whole image" on almost every shot. Nothing in this
   * frame needs it: the only noise the 8-bit output needs is the dither in
   * `OutputTransform`, which removes banding on its own. The node stays, behind
   * a uniform-coherent branch so it costs one compare, for `setOverride`.
   */
  applyGrain(display: any): any {
    if (this.legacy) return display;
    const src: any = display;
    return Fn(() => {
      const c = vec4(src).toVar('grainIn');
      const out = vec3(c.rgb).toVar('grainOut');
      If(this.uGrain.greaterThan(0), () => {
        const p = screenCoordinate.xy.div(1.5).add(vec2(this.uGrainSeed.mul(17.0), this.uGrainSeed.mul(31.0)));
        const i = floor(p);
        const f = fract(p);
        const s = f.mul(f).mul(float(3).sub(f.mul(2)));
        const n00 = hash21(i);
        const n10 = hash21(i.add(vec2(1, 0)));
        const n01 = hash21(i.add(vec2(0, 1)));
        const n11 = hash21(i.add(vec2(1, 1)));
        const n = mix(mix(n00, n10, s.x), mix(n01, n11, s.x), s.y).sub(0.5).mul(2);
        const luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)).saturate();
        // Mid-tone weighted, with a floor so shadows keep a trace of texture.
        const response = luma.mul(float(1).sub(luma)).mul(4).mul(0.8).add(0.2);
        out.addAssign(n.mul(this.uGrain).mul(response));
      });
      return vec4(out, c.a);
    })();
  }

  setTier(tier: QualityTier): void {
    this.tierId = tier;
    this.tier = POST_TIERS[tier];
    this.applyTier();
  }

  get currentTier(): QualityTier {
    return this.tierId;
  }

  /** Re-derives every uniform from the tier, the overrides and the context. */
  applyTier(): void {
    const t = this.tier;
    const o = this.overrides;
    const pick = (override: boolean | null, tierOn: boolean) => (override === null ? tierOn : override);

    const scale = this.canUpscale ? (o.renderScale ?? this.effectiveRenderScale()) : 1;
    this.options.scenePass.setResolutionScale?.(scale);
    this.aoComposite?.setResolutionScale?.(scale);
    // Resize both now rather than at their next `updateBefore`. They resize at
    // different points of the frame, and TRAA reads its input's size before
    // the input has re-rendered — for one frame the depth it copies and the
    // target it copies into disagree, which WebGPU rejects as a validation
    // error on every live tier change across the native boundary.
    this.options.renderer.getDrawingBufferSize(_drawingBuffer);
    if (_drawingBuffer.x > 0 && _drawingBuffer.y > 0) {
      this.options.scenePass.setSize?.(_drawingBuffer.x, _drawingBuffer.y);
      this.aoComposite?.setSize?.(_drawingBuffer.x, _drawingBuffer.y);
    }

    // SSGI carries its own AO, so GTAO stands down wherever it runs.
    const giOn = this.gi !== null && pick(o.gi, t.gi.enabled);
    this.gi?.apply({ ...t.gi, enabled: giOn, resolutionScale: t.gi.resolutionScale * scale });
    if (this.ao) {
      const on = pick(o.ao, t.ao.enabled) && !giOn;
      this.ao.apply({ enabled: on, resolutionScale: t.ao.resolutionScale * scale, samples: t.ao.samples });
    }
    this.contactShadows?.apply({
      ...t.contactShadows,
      enabled: pick(o.contactShadows, t.contactShadows.enabled),
      resolutionScale: scale,
    });

    const sharpenOn = pick(o.sharpen, t.sharpen > 0);
    this.resolve.uSharpen.value = sharpenOn ? (t.sharpen > 0 ? t.sharpen : 0.3) : 0;

    // Motion blur: the tier's budget, never under reduced motion, and never in
    // a deterministic capture unless that capture asked for it.
    const blurAllowed =
      !this.reducedMotion && (!this.deterministic || this.motionBlurInCapture);
    const blurOn = pick(o.motionBlur, t.motionBlurSamples > 0) && blurAllowed;
    this.resolve.uBlurSamples.value = Math.max(4, t.motionBlurSamples || 10);
    this.blurOn = blurOn;
    this.updateShutter();

    const caOn = pick(o.aberration, t.aberration > 0);
    this.resolve.uAberration.value = caOn ? (t.aberration > 0 ? t.aberration : 0.001) : 0;

    const grainOn = pick(o.grain, t.grain > 0);
    this.uGrain.value = grainOn ? (t.grain > 0 ? t.grain : 0.012) : 0;

    this.exposure?.setEnabled(pick(o.autoExposure, t.autoExposure));
  }

  private blurOn = false;

  /**
   * The tier's scene-pass scale for this display.
   *
   * High drops to 0.75 above a pixel ratio of 1.5. At a ratio of 2 that is
   * still 1.5 shaded samples per CSS pixel along each axis — more than a 1x
   * display gets at native — and TAAU (which a session booting below native
   * resolves with) reconstructs the 2x grid from the jittered samples, while
   * the scene pass and everything sized from it shade 56% of the pixels.
   * Ultra and Max keep native: they are opted into, and the headroom is
   * theirs to spend.
   */
  effectiveRenderScale(): number {
    const t = this.tier;
    return this.options.renderer.getPixelRatio() > HI_DPI_RATIO ? t.hiDpiRenderScale : t.renderScale;
  }

  private updateShutter(): void {
    if (!this.blurOn) {
      this.resolve.uShutter.value = 0;
      return;
    }
    // Exposure time of a 60 Hz camera at this shutter angle, as a fraction of
    // the frame this velocity was measured over. Capped at the whole frame.
    const exposureTime = (SHUTTER_ANGLE / 360) / 60;
    this.resolve.uShutter.value = Math.min(1, exposureTime / Math.max(1e-3, this.lastDt));
  }

  /** Per-frame. `dt` is the simulation step. */
  update(dt: number): void {
    this.lastDt = dt > 0 ? dt : this.lastDt;
    this.exposure?.setDeltaTime(dt);
    this.contactShadows?.update();
    if (this.blurOn) this.updateShutter();
    const animate = !this.deterministic && !this.reducedMotion;
    if (animate) {
      this.grainFrame = (this.grainFrame + 1) % 4096;
      this.uGrainSeed.value = this.grainFrame;
    } else {
      this.uGrainSeed.value = 0;
    }
  }

  setPreset(presetId: string): void {
    this.exposure?.setPreset(presetId);
  }

  /**
   * A capture has taken the clock: no history-dependent state from here on.
   *
   * Auto-exposure adopts each frame's own target, grain stops moving, and
   * motion blur is withheld unless `setMotionBlurInCapture(true)`.
   */
  setDeterministic(on: boolean): void {
    this.deterministic = on;
    this.exposure?.setSnap(on);
    if (on) {
      this.grainFrame = 0;
      this.uGrainSeed.value = 0;
    }
    this.applyTier();
  }

  setMotionBlurInCapture(on: boolean): void {
    this.motionBlurInCapture = on;
    this.applyTier();
  }

  /**
   * Restart every temporal accumulator: the resolve's jitter and history, the
   * AO's rotation phase, and the exposure (one snap).
   *
   * Reaches into two fields TRAA and TAAU both carry and do not document
   * (`_jitterIndex`, `_historyRenderTarget`); shrinking the history target is
   * what either's `updateBefore` reads as "restart".
   */
  resetTemporal(): void {
    const node = this.temporalNode as unknown as {
      _jitterIndex: number;
      _historyRenderTarget: { setSize(w: number, h: number): void };
    } | null;
    if (node) {
      node._jitterIndex = 0;
      node._historyRenderTarget.setSize(1, 1);
    }
    this.ao?.resetSequence();
    this.contactShadows?.resetSequence();
    this.gi?.resetSequence();
    this.exposure?.snapOnce();
  }

  /** Test/A-B surface: flip one effect, leaving the tier in charge of the rest. */
  setOverride(effect: keyof PostChain['overrides'], value: any): void {
    (this.overrides as any)[effect] = value;
    this.applyTier();
  }

  /** Snapshot of what is live, for the panel and for tests. */
  describe(): Record<string, unknown> {
    return {
      tier: this.tierId,
      temporal: this.temporalNode ? (this.legacy ? 'traa' : this.temporalMode) : 'none',
      renderScale: this.options.scenePass.getResolutionScale?.() ?? 1,
      ao: this.ao?.enabled ?? false,
      contactShadows: this.contactShadows?.enabled ?? false,
      gi: this.gi?.enabled ?? false,
      motionBlur: this.resolve.uShutter.value > 0,
      sharpen: this.resolve.uSharpen.value,
      aberration: this.resolve.uAberration.value,
      grain: this.uGrain.value,
      autoExposure: this.exposure?.enabled ?? false,
      deterministic: this.deterministic,
      reducedMotion: this.reducedMotion,
    };
  }

  dispose(): void {
    this.reducedMotionQuery?.removeEventListener?.('change', this.onMotionPreference);
    this.ao?.dispose();
    this.contactShadows?.dispose();
    this.gi?.dispose();
    this.exposure?.dispose();
    this.temporalNode?.dispose();
    this.aoComposite?.dispose?.();
  }
}
