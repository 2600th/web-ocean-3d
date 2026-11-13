import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  clamp,
  dot,
  exp,
  exp2,
  float,
  int,
  ivec2,
  log2,
  max,
  mix,
  nodeObject,
  rtt,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Eye adaptation: a log-average luminance meter, an asymmetric temporal
 * response, and an EV correction clamped around each preset's authored exposure.
 *
 * **Why on the GPU, end to end.** The obvious implementation reads the average
 * back to the CPU and sets `toneMappingExposure`. That is a `mapAsync` per frame
 * — either a stall, or a result two or three frames late whose latency depends
 * on how deep the driver happens to be queueing, which is a non-determinism the
 * capture harness cannot tolerate. Here the meter, the reduction and the
 * adaptation are three small render passes and the result never leaves the GPU:
 * the chain samples a 1×1 texture.
 *
 * **What it corrects, and what it leaves alone.** Every preset was graded by
 * eye at a known exposure, and that look is the product. So the meter does not
 * drive the image toward one global middle grey — that would make Moonlit as
 * bright as Clear Day. It measures how far *this frame* sits from the luminance
 * the preset was authored at (its anchor), and corrects a fraction of the
 * difference, inside a clamp. At the preset's own canonical view the correction
 * is zero and the authored look is untouched; looking into the sun, dropping
 * below the surface or riding the tour into night is what moves it.
 *
 * **Why the gain goes in before the bloom.** `SceneBloom` derives its threshold
 * from the preset's exposure, so "white" in the bloom's input is `1/exposure`.
 * Applying the adaptation as a linear gain *ahead* of the bloom keeps that
 * relationship exact: the pyramid sees light already scaled to where the tone
 * curve will put it. Applying it at the tone curve instead would leave the
 * threshold tracking the preset while the image tracked the eye.
 *
 * **Determinism.** Adaptation is history by definition, and a capture must not
 * have any. `setSnap(true)` makes every frame adopt its own target with no
 * temporal filter, so a captured frame's exposure is a function of that frame
 * alone — the same rule the dither follows. The harness turns it on the moment
 * it takes the clock. `snapOnce()` covers the live case after a cut or a reset.
 */

/** Meter grid. Each texel integrates a 4×4 block of bilinear taps. */
const METER_SIZE = 16;
const TAPS_PER_AXIS = 4;
/** Clamp on per-tap log2 luminance, so the sun disc cannot swing the mean. */
const LOG_MIN = -14;
const LOG_MAX = 10;

const LUMA_709 = vec3(0.2126, 0.7152, 0.0722);

/** One small node that runs the reduction and the adaptation once per frame. */
class ExposureAdaptNode extends THREE.TempNode {
  static get type(): string {
    return 'ExposureAdaptNode';
  }

  readonly current: THREE.RenderTarget;
  private readonly previous: THREE.RenderTarget;
  private readonly material = new THREE.NodeMaterial();
  private readonly quad = new THREE.QuadMesh();
  private readonly textureNode: any;
  private rendererState: any = undefined;
  private initialised = false;
  private wroteDisabled = false;

  constructor(
    private readonly meter: any,
    private readonly u: AutoExposure['uniforms'],
  ) {
    super('vec4');
    const options = { depthBuffer: false, type: THREE.HalfFloatType };
    this.current = new THREE.RenderTarget(1, 1, options);
    this.previous = new THREE.RenderTarget(1, 1, options);
    this.current.texture.name = 'AutoExposure.current';
    this.previous.texture.name = 'AutoExposure.previous';
    this.textureNode = texture(this.current.texture);
    this.material.name = 'AutoExposure.adapt';
    // Once per pipeline render, not once per animation frame — see the note on
    // `ScaledTRAA` in `PostChain.ts` for why FRAME is not deterministic here.
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
  }

  setup(builder: any): any {
    const meter = this.meter;
    const previous = texture(this.previous.texture);
    const u = this.u;
    const adapt = Fn(() => {
      const sumLog = float(0).toVar('aeSumLog');
      const sumWeight = float(0).toVar('aeSumW');
      Loop({ start: int(0), end: int(METER_SIZE), type: 'int', condition: '<' }, ({ i }: any) => {
        Loop({ start: int(0), end: int(METER_SIZE), type: 'int', condition: '<', name: 'j' } as any, ({ j }: any) => {
          const cell = meter.load(ivec2(i, j));
          sumLog.addAssign(cell.r);
          sumWeight.addAssign(cell.g);
        });
      });
      const avgLog = sumLog.div(sumWeight.max(1e-4)).toVar('aeAvgLog');
      // How far this frame is from where the preset was graded, corrected by a
      // fraction and clamped. Positive means "brighten".
      // A dead zone first: ordinary reframing inside a preset — a hull filling
      // more of the frame, a little more sky — is part of the authored look and
      // is left alone. Only the excess beyond it is corrected.
      const deviation = u.anchorLog.sub(avgLog).toVar('aeDev');
      const excess = deviation.sign().mul(deviation.abs().sub(u.deadZone).max(0));
      const target = clamp(excess.mul(u.strength), u.minEv, u.maxEv).toVar('aeTarget');
      const prev = previous.load(ivec2(0, 0)).r.toVar('aePrev');
      // Asymmetric: a scene getting brighter (target below the current
      // correction) is answered quickly, as the eye's pupil does; adapting to
      // the dark is slow, as rod adaptation is.
      const rate = target.lessThan(prev).select(u.rateBright, u.rateDark);
      const k = float(1).sub(exp(u.dt.negate().mul(rate)));
      const blended = mix(prev, target, k);
      const adapted = u.snap.greaterThan(0.5).select(target, blended).mul(u.enabled).toVar('aeAdapted');
      return vec4(adapted, avgLog, target, 1);
    });
    this.material.fragmentNode = adapt().context(builder.getSharedContext());
    this.material.needsUpdate = true;
    return this.textureNode;
  }

  updateBefore(frame: any): boolean | undefined {
    const renderer = frame.renderer;
    // Disabled: one more pass to write a zero correction, then nothing — the
    // meter and the reduction cost nothing while the gain sits at exactly 1.
    const enabled = this.u.enabled.value > 0.5;
    if (!enabled && this.wroteDisabled) return undefined;
    this.wroteDisabled = !enabled;
    if (!this.initialised) {
      renderer.initRenderTarget(this.previous);
      this.initialised = true;
    }
    this.rendererState = THREE.RendererUtils.resetRendererState(renderer, this.rendererState);
    renderer.setRenderTarget(this.current);
    this.quad.material = this.material;
    this.quad.name = 'AutoExposure';
    this.quad.render(renderer);
    renderer.copyTextureToTexture(this.current.texture, this.previous.texture);
    THREE.RendererUtils.restoreRendererState(renderer, this.rendererState);
    // A one-shot snap lasts exactly one adaptation step.
    if (this.u.snapOnce.value > 0) {
      this.u.snapOnce.value = 0;
      this.u.snap.value = this.u.snapHeld.value;
    }
    return undefined;
  }

  dispose(): void {
    this.current.dispose();
    this.previous.dispose();
    this.material.dispose();
    super.dispose();
  }
}

/**
 * Anchors, in log2 of the meter's centre-weighted average linear luminance, at
 * which each preset was graded. Measured with adaptation off from the preset's
 * canonical framing (the gallery camera for presets that have one, the clear-day
 * wide camera otherwise); see `docs/AAA_PLAN.md` section N for the run.
 *
 * A preset missing from this table adapts around `DEFAULT_ANCHOR`, which is the
 * median of the measured ones.
 */
export const EXPOSURE_ANCHORS: Record<string, number> = {
  skyPro: -1.39,
  seaOfThieves: -1.76,
  storm: -2.85,
  sunset: -1.0,
  moonlit: -4.75,
  arctic: -2.35,
  blackFlag: -1.5,
  dusk: -3.79,
  foggy: -1.52,
};
const DEFAULT_ANCHOR = -1.76;

export class AutoExposure {
  readonly uniforms = {
    anchorLog: uniform(DEFAULT_ANCHOR),
    /** Fraction of the deviation beyond the dead zone that is corrected. */
    strength: uniform(0.6),
    /** EV either side of the anchor that is left exactly as authored. */
    deadZone: uniform(0.75),
    /** Never darken by more than this, EV. */
    minEv: uniform(-1.5),
    /**
     * Never brighten by more than this, EV. Smaller than the darkening limit on
     * purpose: night and dusk are meant to read as dark, and an eye that fully
     * adapted to them would erase the look the tour spends a beat on.
     */
    maxEv: uniform(0.5),
    /** 1/s. Toward a brighter scene. */
    rateBright: uniform(3.0),
    /** 1/s. Toward a darker scene. */
    rateDark: uniform(0.9),
    dt: uniform(1 / 60),
    snap: uniform(1),
    /** The snap state to return to after a one-shot snap. */
    snapHeld: uniform(0),
    snapOnce: uniform(1),
    enabled: uniform(1),
  };

  private readonly meter: any;
  private readonly node: ExposureAdaptNode;
  /** `exp2(adapted EV)`, a linear gain on scene radiance. */
  readonly gain: any;

  /**
   * @param sourceNode Linear HDR scene, as a texture node. Metered before the gain
   *   is applied — measuring after it would make the loop chase its own tail.
   */
  constructor(sourceNode: any) {
    // A plain texture node on the same GPU texture, *not* the RTT node itself.
    // `RTTNode` re-renders once per render call that samples it, and this
    // meter is its own render call — sampling the RTT here re-ran the whole
    // fog and underwater composite a second time, measured at 0.95 ms on its
    // own. By the time this node's `updateBefore` runs, the chain's consumer
    // has already rendered the texture this frame.
    const source = sourceNode.isRTTNode === true ? texture(sourceNode.value) : sourceNode;
    const tap = Fn(() => {
      const cellUv = uv();
      const cellSize = float(1 / METER_SIZE);
      const sumLog = float(0).toVar('aeTapLog');
      const sumWeight = float(0).toVar('aeTapW');
      for (let y = 0; y < TAPS_PER_AXIS; y++) {
        for (let x = 0; x < TAPS_PER_AXIS; x++) {
          const offset = vec2((x + 0.5) / TAPS_PER_AXIS - 0.5, (y + 0.5) / TAPS_PER_AXIS - 0.5);
          const p = cellUv.add(offset.mul(cellSize));
          const rgb = source.sample(p).rgb;
          const lum = max(dot(rgb, LUMA_709), 1e-6);
          const l = clamp(log2(lum), LOG_MIN, LOG_MAX);
          // Centre-weighted metering: the frame's centre counts about five
          // times its corners, which is what keeps a bright sky at the top edge
          // from darkening a ship in the middle.
          const c = p.sub(0.5);
          const w = exp(dot(c, c).mul(-3.2));
          sumLog.addAssign(l.mul(w));
          sumWeight.addAssign(w);
        }
      }
      return vec4(sumLog, sumWeight, 0, 1);
    });
    this.meter = rtt(tap(), METER_SIZE, METER_SIZE, { type: THREE.HalfFloatType });
    this.node = new ExposureAdaptNode(this.meter, this.uniforms);
    this.gain = exp2((nodeObject(this.node) as any).r);
  }

  /** Which preset's anchor to adapt around. Snaps, since a preset change is a cut. */
  setPreset(presetId: string): void {
    this.uniforms.anchorLog.value = EXPOSURE_ANCHORS[presetId] ?? DEFAULT_ANCHOR;
    this.snapOnce();
  }

  setEnabled(on: boolean): void {
    this.uniforms.enabled.value = on ? 1 : 0;
  }

  get enabled(): boolean {
    return this.uniforms.enabled.value > 0.5;
  }

  /** Per-frame simulation step, for the adaptation's time constant. */
  setDeltaTime(dt: number): void {
    this.uniforms.dt.value = Math.min(0.25, Math.max(0, dt));
  }

  /** Every frame adopts its own target: no history. For deterministic capture. */
  setSnap(on: boolean): void {
    this.uniforms.snapHeld.value = on ? 1 : 0;
    this.uniforms.snap.value = on ? 1 : 0;
  }

  /** The next frame adopts its target outright; adaptation resumes after it. */
  snapOnce(): void {
    this.uniforms.snapOnce.value = 1;
    this.uniforms.snap.value = 1;
  }

  /** Debug read of [adapted EV, metered log2 luminance, target EV]. Stalls; not for frame paths. */
  async read(renderer: THREE.WebGPURenderer): Promise<{ ev: number; avgLog: number; target: number }> {
    const rt = this.node.current;
    const px = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, 1, 1)) as any;
    const f = (i: number) => (px instanceof Uint16Array ? THREE.DataUtils.fromHalfFloat(px[i]) : px[i]);
    return { ev: f(0), avgLog: f(1), target: f(2) };
  }

  dispose(): void {
    this.node.dispose();
    this.meter.dispose?.();
  }
}
