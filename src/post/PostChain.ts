import * as THREE from 'three/webgpu';
import {
  Fn,
  dot,
  float,
  floor,
  fract,
  mix,
  rtt,
  screenCoordinate,
  sin,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import TRAANode, { traa } from 'three/addons/tsl/display/TRAANode.js';
import type { QualityTier } from '../core/QualityManager';
import { AmbientOcclusion } from './AmbientOcclusion';
import { AutoExposure } from './AutoExposure';
import { PostResolve } from './PostResolve';

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
   * temporal resolve (TRAA) upsamples to the drawing buffer; at 1 it is a plain
   * temporal AA. The rest of the chain always runs at the drawing buffer.
   */
  renderScale: number;
  ao: { enabled: boolean; resolutionScale: number; samples: number };
  /** Motion-blur taps; 0 is off. */
  motionBlurSamples: number;
  /** RCAS lobe scale, 0..1. */
  sharpen: number;
  /** Film-grain amplitude in display units; 0 is off. */
  grain: number;
  /** Lateral CA at the frame corner, UV units; 0 is off. */
  aberration: number;
  autoExposure: boolean;
}

export const POST_TIERS: Record<QualityTier, PostTierSettings> = {
  low: {
    renderScale: 0.67,
    ao: { enabled: false, resolutionScale: 0.5, samples: 8 },
    motionBlurSamples: 0,
    sharpen: 0.5,
    grain: 0,
    aberration: 0,
    autoExposure: true,
  },
  medium: {
    renderScale: 0.77,
    ao: { enabled: true, resolutionScale: 0.5, samples: 8 },
    motionBlurSamples: 8,
    sharpen: 0.4,
    grain: 0,
    aberration: 0,
    autoExposure: true,
  },
  high: {
    renderScale: 1,
    ao: { enabled: true, resolutionScale: 0.5, samples: 12 },
    motionBlurSamples: 10,
    sharpen: 0.3,
    grain: 0.012,
    aberration: 0.001,
    autoExposure: true,
  },
  ultra: {
    renderScale: 1,
    ao: { enabled: true, resolutionScale: 0.75, samples: 16 },
    motionBlurSamples: 12,
    sharpen: 0.3,
    grain: 0.012,
    aberration: 0.001,
    autoExposure: true,
  },
  max: {
    renderScale: 1,
    ao: { enabled: true, resolutionScale: 1, samples: 16 },
    motionBlurSamples: 14,
    sharpen: 0.3,
    grain: 0.012,
    aberration: 0.001,
    autoExposure: true,
  },
};

/** Shutter angle, degrees. 180° is the film convention: half the frame interval. */
const SHUTTER_ANGLE = 180;

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
}

/**
 * The parts of the post chain added in the AAA pass, kept out of `main.ts`.
 *
 * `main.ts` still owns the order of the stages it always had — underwater, fog,
 * depth of field, bloom, flare, grade, lens rain, output — and calls in here at
 * four seams:
 *
 * 1. `buildResolve` replaces the bare TRAA call: GTAO is composited into the
 *    scene colour, the temporal resolve runs (TRAA, which also upsamples when the
 *    tier renders below native), and the resolved image gets sharpening,
 *    motion blur and lateral CA in one pass.
 * 2. `buildExposure` meters the fogged, underwater-composited scene and
 *    `applyExposure` applies the adapted gain ahead of the bloom.
 * 3. `applyGrain` adds film grain in display space, before the dither.
 * 4. `setTier` / `update` / `resetTemporal` / `setDeterministic` from the
 *    corresponding places in the app.
 *
 * Every effect has an A/B toggle on `__ocean.post.chain`; the WebGL2 fallback
 * gets auto-exposure and grain only, because everything else hangs off the
 * velocity MRT that path does not have.
 */
export class PostChain {
  readonly resolve = new PostResolve();
  ao: AmbientOcclusion | null = null;
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
  } = {
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

    this.ao = new AmbientOcclusion(depth, camera, scenePass.getTextureNode('normal'));
    // Resolved to a texture explicitly rather than by the resolve's own
    // `convertToTexture`, because its resolution has to follow the scene pass
    // when the tier renders below native — TAAU sizes its input from this
    // target, and an RTT left at the drawing buffer would tell it there was
    // nothing to upscale.
    this.aoComposite = rtt(this.ao.build(color, velocity));

    // TRAA at every render scale, including the ones below native.
    //
    // r185 ships `TAAUNode` for exactly the below-native case and it was tried
    // first. On this scene it lost on every frame compared: at 0.67 it was
    // visibly softer than TRAA (the rigging dissolved), and its thin-feature
    // lock left dark specks through the island canopy's alpha-tested edges,
    // which TRAA does not. TRAA resolves into a drawing-buffer-sized history
    // and samples its input by UV, so a scene pass rendered smaller is
    // upsampled by the same accumulation that anti-aliases it — the jitter is
    // half an *output* pixel, which still lands every frame's samples at a new
    // sub-pixel offset of the smaller grid. `docs/AAA_PLAN.md` section N has
    // the comparison frames.
    this.temporalNode = new ScaledTRAA(this.aoComposite, depth, velocity, camera);

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
   */
  applyGrain(display: any): any {
    if (this.legacy) return display;
    const src: any = display;
    return Fn(() => {
      const c = vec4(src).toVar('grainIn');
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
      return vec4(c.rgb.add(n.mul(this.uGrain).mul(response)), c.a);
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

    const scale = this.canUpscale ? (o.renderScale ?? t.renderScale) : 1;
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

    if (this.ao) {
      const on = pick(o.ao, t.ao.enabled);
      this.ao.apply({ enabled: on, resolutionScale: t.ao.resolutionScale * scale, samples: t.ao.samples });
    }

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
   * Reaches into two fields TRAA carries and does not document
   * (`_jitterIndex`, `_historyRenderTarget`); shrinking the history target is
   * what its own `updateBefore` reads as "restart".
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
      temporal: this.temporalNode ? 'traa' : 'none',
      renderScale: this.options.scenePass.getResolutionScale?.() ?? 1,
      ao: this.ao?.enabled ?? false,
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
    this.exposure?.dispose();
    this.temporalNode?.dispose();
    this.aoComposite?.dispose?.();
  }
}
