import { Vector2 } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  dot,
  float,
  int,
  max,
  min,
  screenSize,
  smoothstep,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * The stages that act on the resolved image before anything else touches it:
 * contrast-adaptive sharpening, camera motion blur and lateral chromatic
 * aberration — fused into one fullscreen pass, because each of them is a few
 * taps of the same texture and three separate passes would pay three target
 * writes for work that fits in one.
 *
 * **Sharpening is RCAS, run on a reversibly tone-mapped signal.** The kernel is
 * AMD's Robust Contrast-Adaptive Sharpening from FSR 1, as three's `SharpenNode`
 * ports it: a five-tap cross whose negative lobe is limited by how much local
 * contrast the pixel can take without ringing. Its limiter assumes a `[0, 1]`
 * signal, and this chain is linear HDR — a specular glint at 40 would drive the
 * `1 - max` term negative and the lobe with it. So each tap is compressed with
 * `c / (1 + max(c))` (Karis 2014), sharpened, and expanded back. That is
 * exactly invertible, keeps the kernel's guarantees, and stops the sun's glitter
 * from growing a dark halo. Three's node cannot be used as-is for that reason,
 * and because it would cost a pass of its own.
 *
 * It sits after the temporal resolve on purpose: sharpening is not fed back into
 * the history, so it cannot accumulate into ringing frame over frame.
 *
 * **Motion blur is a gather along this pixel's own velocity, shutter-angle
 * scaled.** The velocity target is in NDC per frame; a 180° shutter exposes for
 * half a frame, so the blur spans half the motion. It is normalised to a 60 Hz
 * frame so a slow device does not get twice the smear for being slow, clamped
 * so a camera cut cannot paint the screen, and each tap is weighted by whether
 * the pixel it lands on is itself moving — a static hull in the chase camera
 * does not bleed into the sea streaming past it.
 *
 * **Chromatic aberration is lateral only, radial, and small.** Red and blue are
 * re-read at magnified and minified radii growing with `r²`, which is the
 * shape transverse CA has in a real lens; the centre of frame is untouched.
 * Applied here rather than at the end of the chain because this is the last
 * place the image is a plain texture that can be resampled for free — the
 * stages after it (fog, bloom, grade) are all low-frequency, which is where CA
 * is invisible anyway.
 */
export class PostResolve {
  /** RCAS lobe scale, 0..1. FSR's default (0.2 stops) is 0.87; this is gentler. */
  readonly uSharpen = uniform(0.35);
  /** Fraction of a 60 Hz frame the shutter is open, after the frame-rate normalisation. */
  readonly uShutter = uniform(0);
  /** Longest permitted blur, in UV units of frame height. */
  readonly uMaxBlur = uniform(0.035);
  readonly uBlurSamples = uniform(10);
  /** Lateral CA displacement at the frame corner, in UV units. */
  readonly uAberration = uniform(0);
  /**
   * Added to every pixel's velocity, NDC per frame. Zero in use; the profiler
   * and the blur test set it to stand in for a camera pan, because a
   * deterministic capture has a still camera and would otherwise never take
   * the blur branch.
   */
  readonly uMotionBias = uniform(new Vector2());

  build(source: any, velocity: any): any {
    return Fn(() => {
      const p = uv().toVar('prUv');
      const texel = vec2(float(1).div(screenSize.x), float(1).div(screenSize.y)).toVar('prTexel');

      const e = source.sample(p).level(0).toVar('prE');
      const b = source.sample(p.sub(vec2(0, texel.y))).level(0).rgb;
      const d = source.sample(p.sub(vec2(texel.x, 0))).level(0).rgb;
      const f = source.sample(p.add(vec2(texel.x, 0))).level(0).rgb;
      const h = source.sample(p.add(vec2(0, texel.y))).level(0).rgb;

      // Karis compression into [0, 1) so the RCAS limiter's assumptions hold.
      const squash = (c: any) => c.div(max(c.r, max(c.g, c.b)).add(1));
      const eS = squash(e.rgb).toVar('prEs');
      const bS = squash(b);
      const dS = squash(d);
      const fS = squash(f);
      const hS = squash(h);

      const mn4 = min(min(bS, dS), min(fS, hS)).toVar('prMn');
      const mx4 = max(max(bS, dS), max(fS, hS)).toVar('prMx');
      const hitMin = min(mn4, eS).div(mx4.mul(4).max(1e-5));
      const hitMax = vec3(1).sub(max(mx4, eS)).div(mn4.mul(4).sub(4).min(-1e-5));
      const lobeRgb = max(hitMin.negate(), hitMax);
      const lobe = max(float(-0.25 + 1 / 16), min(max(lobeRgb.r, max(lobeRgb.g, lobeRgb.b)), 0))
        .mul(this.uSharpen)
        .toVar('prLobe');
      const sharpS = bS.add(dS).add(fS).add(hS).mul(lobe).add(eS).div(lobe.mul(4).add(1)).toVar('prSharpS');
      // Expand back to HDR. The squashed value is strictly below 1 by
      // construction, the clamp only guards against rounding.
      const sharp = sharpS.div(float(1).sub(max(sharpS.r, max(sharpS.g, sharpS.b)).min(0.9999)))
        .toVar('prSharp');

      const color = vec3(sharp).toVar('prColor');

      // --- lateral chromatic aberration ------------------------------------
      If(this.uAberration.greaterThan(0), () => {
        const centred = p.sub(0.5).toVar('prCaC');
        const aspect = screenSize.x.div(screenSize.y.max(1));
        const r2 = dot(vec2(centred.x.mul(aspect), centred.y), vec2(centred.x.mul(aspect), centred.y))
          .div(aspect.mul(aspect).add(1).mul(0.25));
        const shift = centred.mul(this.uAberration.mul(r2).mul(2));
        const red = source.sample(p.add(shift)).level(0).r;
        const blue = source.sample(p.sub(shift)).level(0).b;
        color.assign(vec3(color.r.add(red.sub(e.r)), color.g, color.b.add(blue.sub(e.b))));
      });

      // --- motion blur ------------------------------------------------------
      const motion = velocity.sample(p).level(0).xy.add(this.uMotionBias).mul(vec2(0.5, -0.5)).mul(this.uShutter).toVar('prMotion');
      const motionLen = motion.length().toVar('prMotionLen');
      const clampedLen = min(motionLen, this.uMaxBlur).toVar('prClampLen');
      const blurPixels = clampedLen.mul(screenSize.y).toVar('prBlurPx');
      If(blurPixels.greaterThan(0.75), () => {
        const dir = motion.div(motionLen.max(1e-6)).mul(clampedLen).toVar('prDir');
        const centreSpeed = velocity.sample(p).level(0).xy.add(this.uMotionBias).length().max(1e-6).toVar('prSpeed');
        const sum = vec3(color).toVar('prSum');
        const weight = float(1).toVar('prWeight');
        const n = this.uBlurSamples;
        Loop({ start: int(0), end: int(n), type: 'int', condition: '<' }, ({ i }: any) => {
          const t = float(i).add(0.5).div(float(n)).sub(0.5);
          const q = p.add(dir.mul(t));
          const tapSpeed = velocity.sample(q).level(0).xy.add(this.uMotionBias).length();
          const w = smoothstep(0.15, 0.6, tapSpeed.div(centreSpeed));
          sum.addAssign(source.sample(q).level(0).rgb.mul(w));
          weight.addAssign(w);
        });
        // Fade in over the first few pixels of blur so the switch at the
        // threshold is invisible.
        const fade = smoothstep(0.75, 3.0, blurPixels);
        color.assign(color.add(sum.div(weight).sub(color).mul(fade)));
      });

      return vec4(color.max(0), e.a);
    })();
  }
}
