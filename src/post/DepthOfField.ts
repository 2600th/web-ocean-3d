import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  cos,
  float,
  perspectiveDepthToViewZ,
  screenSize,
  sin,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';

/**
 * Depth of field, from a lens rather than from a blur radius.
 *
 * **The circle of confusion is the thin-lens one, and that is the whole point.**
 * Three ships a `DepthOfFieldNode` whose CoC is
 * `smoothstep(0, focalLength, |viewZ − focus|)` scaled by a bokeh factor: a
 * linear ramp in distance from the focal plane, with no sensor and no aperture
 * in it. An aperture can be made to *drive* those numbers, so the honest
 * objection is not that it cannot be done but that the curve is the wrong shape.
 * A real lens' CoC is hyperbolic in object distance and strongly asymmetric
 * about the focal plane — near-field blur grows far faster than far-field, and
 * the far field saturates at the hyperfocal limit rather than growing without
 * bound. A symmetric ramp expresses neither, and both are exactly what makes a
 * wide lens over open water read as a lens instead of as a blur filter.
 *
 * So:
 *
 * ```
 * f    = (sensorHeight / 2) / tan(fovY / 2)          focal length, metres
 * A    = f / N                                       aperture diameter, N = f-number
 * c(d) = A · f · |d − focus| / (d · (focus − f))     CoC diameter on the sensor
 * r    = c(d) · (frameHeightPx / sensorHeight) / 2   CoC radius, pixels
 * ```
 *
 * `f` is derived from the live camera's field of view rather than authored, so
 * the CoC stays correct if the field of view is ever changed.
 *
 * **The gather is single-frame, deliberately.** The obvious better answer is to
 * jitter the ray per sample and accumulate temporally — it is what a path tracer
 * does and what one of this project's reference shaders does. It is unusable
 * here: the entire visual harness rests on a frame being reproducible from a
 * rewound clock, and a temporally accumulated image is a function of how many
 * frames preceded it. Every baseline would become a function of test ordering,
 * which is the one property the shot list's own header exists to rule out.
 *
 * **The tap rejection is not optional.** Each tap contributes only if its *own*
 * circle of confusion is large enough to reach the centre pixel. Without that
 * test a blurred foreground smears over a sharp background at every silhouette —
 * the classic "halo around everything" that makes cheap depth of field obvious.
 * It is one comparison per tap and it is the difference between this looking
 * like a lens and looking like a mistake.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Full-frame sensor height, metres. Sets the scale the CoC is measured in. */
const SENSOR_HEIGHT = 0.024;

/**
 * The golden angle, radians.
 *
 * Successive taps at this angular increment with radius `sqrt(i/n)` produce a
 * sunflower spiral — the most uniform disc packing available without a lookup
 * table, and crucially one with no rotational symmetry, so under-sampling shows
 * up as noise rather than as a polygonal bokeh.
 */
const GOLDEN_ANGLE = 2.39996323;

/** Gather radius ceiling in pixels at the top tier, and the tap count it assumes. */
const MAX_RADIUS_AT_FULL_TAPS = 14;
const FULL_TAPS = 32;

export class DepthOfField {
  private camera: THREE.PerspectiveCamera | null = null;

  private readonly uNear = uniform(0.1);
  private readonly uFar = uniform(40000);
  /** Focal length, metres. */
  private readonly uFocal = uniform(0.023);
  /** Aperture diameter, metres. */
  private readonly uAperture = uniform(0.023 / 5.6);
  /** Focus distance, metres. */
  private readonly uFocus = uniform(120);
  /** Sensor metres to pixels. */
  private readonly uPixelsPerMetre = uniform(720 / SENSOR_HEIGHT);
  /** Hard ceiling on the gather radius, pixels. */
  private readonly uMaxRadius = uniform(14);
  /**
   * Tap count, twice.
   *
   * `Loop` needs an integer node for its bound; the spiral's `sqrt(i / n)`
   * radius and the top-level gate need the same number as a float. Deriving one
   * from the other in the shader would cost a conversion per fragment for a
   * value that changes only on a tier change, so both are uniforms and
   * `setSamples` is the single place that writes them.
   */
  private readonly uSampleCount: any = uniform(0, 'int');
  private readonly uSamples = uniform(0);

  /**
   * f-number, and a note on why the effect is deliberately invisible in the
   * wide shots.
   *
   * A 55-degree field on a full-frame sensor is a 23 mm lens, and a 23 mm lens
   * has enormous depth of field: at f/2.8 the hyperfocal distance is about six
   * metres, so anything past a dozen metres is sharp when the focus is at four
   * hundred. That is not a defect to be tuned out — it is the correct answer,
   * and it is also the right *look*, because a softened island in a landscape
   * shot would read as a mistake.
   *
   * Where this earns its cost is the close work: the reef at twelve metres with
   * coral at three, the underwater dive, the surf. There the circle of
   * confusion is a few pixels and the frame gains the depth cue it should have.
   *
   * Written down because the tempting "fix" for a wide shot that shows no bokeh
   * is a scale factor on the CoC, and that would be a lie applied everywhere to
   * solve a problem that exists nowhere.
   */
  private fNumber = 2.8;
  private focusDistance = 120;
  private samples = 0;

  setCamera(camera: THREE.PerspectiveCamera): void {
    this.camera = camera;
    this.uNear.value = camera.near;
    this.uFar.value = camera.far;
  }

  /**
   * @param sceneColor Must be a *texture* node — the gather re-samples the image
   *   at offset coordinates, which a composited colour node cannot do.
   * @param sceneDepth Must be the depth *texture* node, for the same reason: the
   *   rejection test needs each tap's own depth.
   */
  build(sceneColor: unknown, sceneDepth: unknown): unknown {
    const colorNode: any = sceneColor;
    const depthNode: any = sceneDepth;
    if (colorNode === null || colorNode === undefined || typeof colorNode.sample !== 'function') {
      throw new Error('DepthOfField.build: sceneColor must be a texture node.');
    }
    if (depthNode === null || depthNode === undefined || typeof depthNode.sample !== 'function') {
      throw new Error(
        'DepthOfField.build: sceneDepth must be a texture node, e.g. scenePass.getTextureNode( "depth" ).',
      );
    }

    const baseUv: any = colorNode.uvNode ?? uv();

    /** Distance from the eye to the nearest surface at a screen uv, metres. */
    const viewDistance = (p: any): any =>
      perspectiveDepthToViewZ(depthNode.sample(p).r, this.uNear, this.uFar).negate();

    /** Circle-of-confusion radius in pixels at a view distance. */
    const cocRadius = (d: any): any => {
      // Guarded away from zero: a sky pixel comes back at the far plane and a
      // degenerate one at zero, and `d` is a denominator.
      const dist = d.max(0.01);
      const numerator = this.uAperture.mul(this.uFocal).mul(dist.sub(this.uFocus).abs());
      // `focus - f` is positive by construction — `update` clamps the focus
      // above the focal length, because a lens cannot focus closer than that.
      const denominator = dist.mul(this.uFocus.sub(this.uFocal));
      return numerator
        .div(denominator.max(1e-6))
        .mul(this.uPixelsPerMetre)
        .mul(0.5)
        .min(this.uMaxRadius);
    };

    return Fn(() => {
      const suv = vec2(baseUv).toVar('dofUv');
      const src = colorNode.sample(suv).toVar('dofSrc');
      const outRgb = vec3(src.rgb).toVar('dofOut');

      // Uniform-coherent branch, exactly as `UnderwaterPass` gates its whole
      // effect on submersion. This is what makes `dofSamples: 0` a bit-exact
      // pass-through rather than merely a cheap one: with the branch not taken,
      // the output is the source texel unmodified, not the source texel summed
      // with itself and divided by one.
      If(this.uSamples.greaterThan(0.5), () => {
        const centreDist = viewDistance(suv).toVar('dofCentreDist');
        const centreRadius = cocRadius(centreDist).toVar('dofCentreR');

        const texel = vec2(float(1).div(screenSize.x), float(1).div(screenSize.y))
          .toVar('dofTexel');

        const accum = vec3(src.rgb).toVar('dofAccum');
        const weight = float(1).toVar('dofWeight');

        Loop(this.uSampleCount, ({ i }: any) => {
          const fi = float(i).toVar('dofI');
          const theta = fi.mul(GOLDEN_ANGLE).toVar('dofTheta');
          // `sqrt(i / n)` rather than `i / n`: a disc's area grows as the square
          // of its radius, so a linear radius clusters every tap near the centre
          // and leaves the outside of the bokeh under-sampled.
          const rho = fi.add(0.5).div(this.uSamples).sqrt().mul(centreRadius).toVar('dofRho');
          const offset = vec2(cos(theta), sin(theta)).mul(rho).mul(texel).toVar('dofOffset');
          const tapUv = suv.add(offset).clamp(0, 1).toVar('dofTapUv');

          const tapDist = viewDistance(tapUv).toVar('dofTapDist');
          const tapRadius = cocRadius(tapDist).toVar('dofTapR');

          // The rejection. A tap counts only if its own circle of confusion is
          // wide enough to have reached this pixel — that is what "scattering,
          // gathered" means. Comparing against `max(tapRadius, 1)` lets a
          // perfectly sharp neighbour still contribute its own texel's worth,
          // which keeps in-focus regions from losing energy at their edges.
          const reaches = rho.lessThanEqual(tapRadius.max(1)).select(float(1), float(0))
            .toVar('dofReach');

          accum.addAssign(colorNode.sample(tapUv).rgb.mul(reaches));
          weight.addAssign(reaches);
        });

        outRgb.assign(accum.div(weight.max(1e-4)));
      });

      return vec4(outRgb, src.a);
    })();
  }

  /**
   * Recomputes the lens from the live camera. Called every frame; allocates
   * nothing.
   */
  update(): void {
    const camera = this.camera;
    if (camera === null) return;

    const fovY = THREE.MathUtils.degToRad(camera.fov);
    // Focal length that produces this field of view on a full-frame sensor.
    const f = SENSOR_HEIGHT / 2 / Math.tan(fovY / 2);
    this.uFocal.value = f;
    this.uAperture.value = f / Math.max(0.7, this.fNumber);
    // A lens cannot focus closer than its own focal length, and `focus - f` is
    // the CoC's denominator. Clamped just above it rather than at it.
    this.uFocus.value = Math.max(f * 1.001, this.focusDistance);
    this.uNear.value = camera.near;
    this.uFar.value = camera.far;
  }

  /** Frame height in pixels, for the sensor-metres-to-pixels conversion. */
  setFrameHeight(pixels: number): void {
    this.uPixelsPerMetre.value = Math.max(1, pixels) / SENSOR_HEIGHT;
  }

  setFocusDistance(metres: number): void {
    this.focusDistance = Number.isFinite(metres) ? Math.max(0.1, metres) : 120;
  }

  /** f-number. Smaller is a wider aperture and a shallower depth of field. */
  setAperture(fNumber: number): void {
    this.fNumber = Math.max(0.7, fNumber);
  }

  /** Tap count from the quality tier. 0 is a bit-exact pass-through. */
  setSamples(count: number): void {
    this.samples = Math.max(0, Math.floor(count));
    this.uSamples.value = this.samples;
    this.uSampleCount.value = this.samples;
    // The radius ceiling follows the tap count, so tap *density* stays roughly
    // constant across the tiers. Sample density over a disc goes as N / (pi r^2),
    // so holding it fixed means r scales with the square root of N. A tier that
    // kept the full radius on a quarter of the taps would not be a cheaper
    // depth of field, it would be a visibly noisy one — the spiral would start
    // to read as individual samples rather than as a defocused image.
    this.uMaxRadius.value = MAX_RADIUS_AT_FULL_TAPS * Math.sqrt(this.samples / FULL_TAPS);
  }

  /** Owns no GPU resources; present so the chain's teardown stays uniform. */
  dispose(): void {}
}
