import * as THREE from 'three/webgpu';
import TAAUNode from 'three/addons/tsl/display/TAAUNode.js';
import {
  Fn,
  OnAfterRenderPipeline,
  OnBeforeRenderPipeline,
  context,
  float,
  ivec2,
  luminance,
  max,
  mix,
  outputStruct,
  property,
  texture,
  uv,
  vec2,
  vec4,
} from 'three/tsl';
// @ts-expect-error -- three ships no typings for its TAA helpers.
import { clipAABB, flickerReduction, sampleCurrentDepth, samplePreviousDepth } from 'three/addons/tsl/utils/TAAUtils.js';

/* eslint-disable @typescript-eslint/no-explicit-any */

const _drawingBuffer = new THREE.Vector2();
const _seedQuad = new (THREE.QuadMesh as unknown as new () => THREE.QuadMesh)();
let _seedState: any;

/** Catmull-Rom weight at `x` texels; the 3x3 taps only ever reach |x| <= 1.5. */
function catmullRom(x: any): any {
  const a = x.abs();
  const near = a.mul(1.5).sub(2.5).mul(a).mul(a).add(1);
  const far = a.mul(-0.5).add(2.5).mul(a).sub(4).mul(a).add(2);
  return a.lessThan(1).select(near, far);
}

/**
 * three's TAAU (r187), with what `ScaledTRAA` adds to TRAA that TAAU does not
 * already have, and a sharper reconstruction filter.
 *
 * Of `ScaledTRAA`'s three behaviours, the previous-depth copy at the input's
 * size is upstream's own: TAAU keeps the previous depth in a target sized to
 * the depth it copies, so the disocclusion test works below native without
 * help. The jitter needs no publishing either — see `PostChain.buildResolve` —
 * because TAAU's resolve places every input sample at its jittered position
 * when it reconstructs, which is the correction `ScaledTRAA` makes by reading
 * its input back on the unjittered grid. What is left:
 *
 * - **Once per pipeline render**, not per animation frame; the jitter
 *   already advances per render (`clearViewOffset` runs after every pipeline
 *   render). See `ScaledTRAA`'s constructor for the harness this matters to.
 * - **A restart resolves to the seed**; see `updateBefore`.
 * - **A Catmull-Rom reconstruction filter** in place of upstream's Gaussian
 *   fit to Blackman-Harris (`exp(-2.29·d²)`, σ = 0.47 input pixels). The
 *   Gaussian has no negative lobe, so every frame is low-passed before it is
 *   accumulated. At Medium's 0.77 the Catmull-Rom was the sharper on all three
 *   gallery crops measured (mean |Laplacian| of luma: deck +5%,
 *   ship-and-island +12%, beach +9%), with no ringing on the rigging against
 *   the sky; the clamp to the taps' range is FSR2's de-ringing, for the hard
 *   HDR edges where it would. At native neither filter reached TRAA's
 *   sharpness, which is why a session that boots at native resolves with
 *   TRAA (`PostChain.buildResolve`). The weight is an expression inside
 *   `setup`, so `setup` is upstream's, copied; everything else in it is r187's
 *   `TAAUNode.setup` line for line (MIT, three.js authors).
 */
export class DeterministicTAAU extends (TAAUNode as unknown as new (...args: any[]) => any) {
  constructor(beauty: any, depth: any, velocity: any, camera: THREE.Camera) {
    super(beauty, depth, velocity, camera);
    this.updateBeforeType = THREE.NodeUpdateType.RENDER;
  }

  /**
   * On a restart (a resize, a tier change across native, or `resetTemporal`)
   * TAAU seeds its history with this frame and then resolves against it,
   * reprojecting through velocities that, on that one frame, describe the
   * move from wherever the camera last drew — after a cut, the old shot. At a
   * 2.5% current-frame weight that error decays over a hundred frames, so the
   * capture after a camera jump would differ from the one after it. On a
   * restart the seed *is* the resolve: the seed pass is drawn again over the
   * resolve's history write and copied into the resolve target.
   */
  updateBefore(frame: any): any {
    const self = this as any;
    const renderer = frame.renderer;
    renderer.getDrawingBufferSize(_drawingBuffer);
    const history = self._historyRenderTarget;
    const restart = history.width !== _drawingBuffer.x || history.height !== _drawingBuffer.y;
    const result = super.updateBefore(frame);
    if (restart) {
      _seedState = THREE.RendererUtils.resetRendererState(renderer, _seedState);
      renderer.setRenderTarget(history);
      _seedQuad.material = self._seedMaterial;
      _seedQuad.name = 'TAAU.reseed';
      _seedQuad.render(renderer);
      renderer.setRenderTarget(null);
      renderer.copyTextureToTexture(history.textures[0], self._resolveRenderTarget.texture);
      THREE.RendererUtils.restoreRendererState(renderer, _seedState);
    }
    return result;
  }

  setup(builder: any): any {
    const self = this as any;
    self.depthNode.build(builder);
    self.velocityNode.build(builder);
    self.beautyNode.build(builder);

    if (builder.renderPipeline && !builder.context.renderPipelineState.viewOffsetOwner) {
      builder.context.renderPipelineState.viewOffsetOwner = this;
      OnBeforeRenderPipeline(() => {
        const target = self.beautyNode.isRTTNode ? self.beautyNode.renderTarget : self.beautyNode.passNode.renderTarget;
        self.setViewOffset(target.texture.width, target.texture.height);
      });
      OnAfterRenderPipeline(() => {
        self.clearViewOffset();
      });
    }

    self._previousDepthRenderTarget.depthTexture.format = self.depthNode.value.format;
    self._previousDepthRenderTarget.depthTexture.type = self.depthNode.value.type;

    const historyNode: any = texture(self._historyRenderTarget.textures[0]);
    const lockNode: any = texture(self._historyRenderTarget.textures[1]);
    const colorOutput: any = property('vec4');
    const lockOutput: any = property('vec4');
    const outputNode = outputStruct(colorOutput, lockOutput);
    const beauty = self.beautyNode;
    const jitter = self._jitterOffset;

    const resolve = Fn(() => {
      const uvNode: any = uv();
      const inputSizeF = vec2(beauty.size());
      const pIn = uvNode.mul(inputSizeF);
      const closestTapF = pIn.sub(vec2(0.5).add(jitter)).round();
      const closestTap = ivec2(closestTapF);

      const currentDepth = sampleCurrentDepth(self.depthNode, closestTapF, self._cameraNearFar);
      const closestDepth = currentDepth.get('closestDepth');
      const closestPositionTexel = currentDepth.get('closestPositionTexel');
      const farthestDepth = currentDepth.get('farthestDepth');

      const offsetUV = self.velocityNode.load(closestPositionTexel).xy.mul(vec2(0.5, -0.5));
      const historyUV = uvNode.sub(offsetUV);
      const previousDepth = samplePreviousDepth(
        self._previousDepthNode,
        historyUV,
        self._previousCameraProjectionMatrixInverse,
        self._previousCameraWorldMatrix,
        self._cameraWorldMatrixInverse,
        self._cameraNearFar,
        self.camera,
      );

      const isValidUV = historyUV.greaterThanEqual(0).all().and(historyUV.lessThanEqual(1).all());
      const isEdge = farthestDepth.sub(closestDepth).greaterThan(self.edgeDepthDiff);
      const isDisocclusion = closestDepth.sub(previousDepth).greaterThan(self.depthThreshold);
      const hasValidHistory = isValidUV.and(isEdge.or(isDisocclusion.not()));

      const sumColor = vec4(0).toVar();
      const sumWeight = float(0).toVar();
      const moment1 = vec4(0).toVar();
      const moment2 = vec4(0).toVar();
      const tapMin = vec4(1e9).toVar();
      const tapMax = vec4(0).toVar();
      for (const [x, y] of [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]) {
        const tap = closestTap.add(ivec2(x, y));
        const tapCenter = vec2(tap).add(vec2(0.5).add(jitter));
        const delta = pIn.sub(tapCenter);
        // The change from upstream: Catmull-Rom for `exp(-2.29·d²)`.
        const w = catmullRom(delta.x).mul(catmullRom(delta.y));
        const c = beauty.load(tap).max(0);
        sumColor.addAssign(c.mul(w));
        sumWeight.addAssign(w);
        moment1.addAssign(c);
        moment2.addAssign(c.pow2());
        tapMin.assign(tapMin.min(c));
        tapMax.assign(tapMax.max(c));
      }
      // The cubic's negative lobes ring across a hard HDR edge (a sunlit sail
      // against the sky); clamping to the taps it was built from is FSR2's
      // de-ringing, and a no-op for the Gaussian, which cannot overshoot.
      const currentColor = sumColor.div(sumWeight.max(1e-5)).clamp(tapMin, tapMax);

      const N = float(9);
      const mean = moment1.div(N);
      const motionFactor = uvNode.sub(historyUV).mul(inputSizeF).length().div(self.maxVelocityLength).saturate();
      const varianceGamma = mix(0.5, 1, motionFactor.oneMinus().pow2());
      const variance = moment2.div(N).sub(mean.pow2()).max(0).sqrt().mul(varianceGamma);
      const minColor = mean.sub(variance);
      const maxColor = mean.add(variance);
      const historyColor = historyNode.sample(historyUV);
      const clippedHistoryColor = clipAABB(mean.clamp(minColor, maxColor), historyColor, minColor, maxColor);

      const currentLuma = luminance(currentColor.rgb);
      const meanLuma = luminance(mean.rgb).toConst();
      const thinFeature = currentLuma.sub(meanLuma).abs().div(meanLuma).smoothstep(0, 0.2);
      const isDepthChanged = closestDepth.sub(previousDepth).abs().greaterThan(self.depthThreshold);
      const canLock = isValidUV.and(isDepthChanged.not());
      const gatedThinFeature = canLock.select(thinFeature, float(0));
      const decay = isDisocclusion.select(0, 0.5);
      const lock = max(gatedThinFeature, lockNode.r.mul(decay)).saturate();
      const lockedHistoryColor = mix(clippedHistoryColor, historyColor, lock);

      const currentWeight = float(self.currentFrameWeight).toVar();
      currentWeight.assign(hasValidHistory.select(currentWeight.add(motionFactor).saturate(), 1));
      colorOutput.assign(flickerReduction(currentColor, lockedHistoryColor, currentWeight));
      lockOutput.assign(lock);
      return vec4(0);
    });

    const sharedContext = context(builder.getSharedContext());
    self._resolveMaterial.contextNode = sharedContext;
    self._resolveMaterial.colorNode = resolve();
    self._resolveMaterial.outputNode = outputNode;

    self._seedMaterial.colorNode = Fn(() => {
      colorOutput.assign(beauty.sample(uv()));
      lockOutput.assign(0);
      return vec4(0);
    })();
    self._seedMaterial.contextNode = sharedContext;
    self._seedMaterial.outputNode = outputNode;

    return self._textureNode;
  }
}
