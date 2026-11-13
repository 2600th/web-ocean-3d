import * as THREE from 'three/webgpu';
import { reflector } from 'three/tsl';

/**
 * Planar reflection of the scene in the water surface.
 *
 * The surface previously reflected `mix(horizonColor, skyColor, reflectDir.y)` —
 * a two-colour analytic gradient. It is a reasonable stand-in for open ocean
 * under an empty sky and it is completely wrong the moment anything is *on* the
 * water: a ship sitting in a mirror that does not contain it reads as pasted on,
 * and no amount of Fresnel tuning fixes that.
 *
 * **Planar, not screen-space.** SSR can only reflect what is already on screen,
 * and the geometry that matters here is the hull — which, seen from a low camera,
 * has its reflection below the waterline while the object itself is near the top
 * of the frame or out of it entirely. That is precisely the case SSR cannot
 * serve. A mirrored camera has the whole scene available, and the ocean is a
 * plane, which is the one situation where planar reflection is exactly right
 * rather than an approximation.
 *
 * The cost is a second view of the scene. It is charged only where it buys
 * something: see `setQuality`.
 */
export class Reflections {
  /** Add to the scene; its transform defines the mirror plane. */
  readonly plane: THREE.Object3D;

  /** TSL texture node carrying the reflected scene. Bind once. */
  readonly node: any;

  private readonly base: any;
  constructor(resolutionScale = 0.5) {
    // `reflector` mirrors about the target's local XY plane, so the target is
    // rotated to put its normal along +Y. Sea level, not the camera's height:
    // the displaced surface oscillates about y = 0 and reflecting about the mean
    // plane is what keeps the reflection stable as waves pass through it.
    // Mipmapped, so the surface can sample it at a roughness-driven level.
    //
    // `generateMipmaps: false` was fine while the reflection was only ever
    // sampled as a sharp mirror, and that is exactly the problem: a rough sea
    // does not reflect sharply, and without a mip chain there is nothing to
    // sample instead. A single level costs a third again in bandwidth on a target
    // that is already at half resolution, and it is what lets a chopped-up sea
    // reflect as a chopped-up sea rather than as polished metal.
    const node = reflector({ resolutionScale, generateMipmaps: true, bounces: false });
    node.target.rotateX(-Math.PI / 2);
    node.target.name = 'ocean-reflector';

    this.node = node;
    this.base = node.reflector;
    this.plane = node.target;
  }

  /**
   * Per-tier cost policy.
   *
   * `resolutionScale` is the real lever. The reflection is sampled through a
   * surface normal that is being perturbed by every ripple in the wave field, so
   * it is never seen sharp — half resolution is indistinguishable from full in
   * motion, and a quarter is acceptable wherever the sea state is rough enough
   * to break the image up anyway.
   *
   * This was latched shut for a while — honoured once at boot and ignored after
   * — because resizing a mipmapped target during a tier change crashed the
   * renderer: the reflector renders the whole scene from its own `updateBefore`,
   * so it reached node graphs three had not finished rebuilding. The latch
   * treated the symptom. The cause was that a tier change destroyed resources
   * still referenced by a submitted command buffer, and it is now fixed where it
   * belongs — `App.drainQualityRequests` pauses the loop, joins the frame in
   * flight, fences on `onSubmittedWorkDone`, and only then applies the tier. With
   * the GPU provably idle there is nothing left for the latch to protect, and
   * someone who starts on Low and switches to Max gets the reflection they paid
   * for instead of a quarter-resolution one.
   *
   * Dropping the old targets is the point rather than a side effect. `reflector`
   * keys its targets by camera, and changing `resolutionScale` alone leaves the
   * existing entry at its old size, so the scale would apply only to a camera
   * the reflector had not seen before — which, in a session with one camera, is
   * never.
   */
  setQuality(scale: number): void {
    const next = Math.max(0.1, Math.min(1, scale));
    if (next === this.base.resolutionScale) return;
    this.base.resolutionScale = next;
    for (const target of this.base.renderTargets.values()) target.dispose();
    this.base.renderTargets.clear();
  }

  /**
   * Keeps a layer out of the mirror for `camera`.
   *
   * The reflector builds its virtual camera once per real camera and never
   * copies layers onto it, so disabling a layer there sticks.
   */
  excludeLayer(camera: THREE.Camera, layer: number): void {
    this.base.getVirtualCamera(camera).layers.disable(layer);
  }

  /** Current resolution scale, so a test can prove the tier reached it. */
  get resolutionScale(): number {
    return this.base.resolutionScale;
  }

  /**
   * What the reflector renders `camera`'s reflection with: its mirrored camera
   * and its target. Read for the pipeline prewarm, which compiles that render
   * ahead of time; the camera's pose is whatever the last frame left it in.
   */
  renderSetup(camera: THREE.Camera): { camera: THREE.Camera; target: THREE.RenderTarget } {
    const virtualCamera = this.base.getVirtualCamera(camera) as THREE.Camera;
    return { camera: virtualCamera, target: this.base.getRenderTarget(virtualCamera) };
  }

  /** Whether `target` is one of the reflector's own render targets. */
  ownsTarget(target: unknown): boolean {
    for (const own of this.base.renderTargets.values()) if (own === target) return true;
    return false;
  }

  dispose(): void {
    this.plane.removeFromParent();
    for (const target of this.base.renderTargets.values()) target.dispose();
    this.base.renderTargets.clear();
  }
}
