import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  exp,
  float,
  perspectiveDepthToViewZ,
  screenSize,
  smoothstep,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * Flare from the key light — anchored to it, occluded by the scene, and above
 * water only.
 *
 * **Not the pseudo-flare.** Three ships a `LensflareNode` implementing John
 * Chapman's technique: take the bloom texture, and pivot a chain of ghosts about
 * screen centre from *every* surviving bright sample. It is a good effect on a
 * scene with one or two isolated highlights. This scene is a sea covered in sun
 * glitter and breaking whitecaps, and every one of them would throw its own
 * ghost chain. So the source here is not "whatever is bright" but the light
 * itself, projected.
 *
 * **The anchor is a projection, not a search.** The key light's direction is
 * projected to screen coordinates on the CPU once per frame into a uniform.
 * There is no readback and no per-pixel projection.
 *
 * **The gate is angular, and it is the part most worth getting right.** The
 * whole effect is scaled by `smoothstep(-0.25, 0.45, dot(viewDir, sunDir))` —
 * how nearly the camera is pointed at the light — rather than by an
 * in-front-of-the-camera test. A boolean there is correct and looks wrong: it
 * switches the flare on the frame the light crosses the 90-degree plane, which
 * is a pop in the middle of a pan. The angular form fades it out over tens of
 * degrees, so a camera sweeping past the sun sees the flare arrive and leave.
 * This is the construction `reinder`'s *Himalayas* uses, itself crediting mu6k.
 *
 * **The occlusion is a depth test, so the flare belongs to the scene.** Eight
 * taps on a small disc at the light's screen position; a tap is clear when its
 * depth reads at the far plane. Behind the island, behind the hull, behind the
 * rigging, the flare goes with it. Without this it floats over solid geometry
 * and reads as an overlay — which is exactly what it would be.
 *
 * A second, frame gate closes the effect as the light leaves the frame (see
 * `OFFSCREEN_FADE`): the angular one alone kept it at full strength 60° off
 * axis, where its long-tailed halo lay a veil over a shot with no sun in it.
 *
 * **Ghosts sit on the line through the light and the frame centre**, at signed
 * magnifications (`GHOST_SCALES`), and are faint (`GHOST_GAIN`). Subtle by
 * design: the A/B critic read the previous, brighter chain as blobs and a
 * streak, and a flare that is noticed is a flare that is too strong.
 *
 * **The moon is the same lens, and reading it wrong is an easy trap.**
 * `Atmosphere` owns *one* directional light and retargets it to the moon once
 * the sun is down, overwriting its colour with `MOON_LIGHT_COLOR`. Meanwhile
 * `atmosphere.sunColor` goes on reporting the solar extinction colour whatever
 * the hour. So this class takes a direction and a colour from its caller and
 * makes no assumption about which body it is looking at — `main` hands it the
 * live key light, which follows the retarget.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Ghost positions, as signed multiples of the light's offset from frame centre.
 *
 * A ghost is an image of the source formed by a pair of internal reflections,
 * and every such image lies on the line through the source and the optical
 * axis — at its own magnification, which can be either sign (Chapman's
 * "pseudo lens flare" uses the same construction: `ghostUv = -uv * k`).
 * Negative is the far side of centre, which is where most of a real chain
 * lands. The previous placement put every ghost *between* the source and the
 * centre, on the warped radius of the pixel being shaded rather than of the
 * source, so the chain smeared into a band near the sun.
 */
const GHOST_SCALES = [-0.45, -0.9, -1.35, 0.4];
/** Ghost disc radii, frame heights. Small: these are aperture images, not haloes. */
const GHOST_RADII = [0.022, 0.035, 0.05, 0.016];
/**
 * Peak of one ghost relative to the source radiance factor.
 *
 * Was 0.03, which against Clear Day's white point (1/0.42 ≈ 2.4) made each
 * ghost about 1.3% of white — a visible disc in open sky. A ghost is two
 * reflections off coated glass, each well under 1%, so it should sit near the
 * threshold of visibility against a bright sky: 0.006 is about a quarter of a
 * percent of white there.
 */
const GHOST_GAIN = 0.006;

/**
 * Glare core amplitude and radius (frame heights).
 *
 * Shaped as `1 / (1 + (r / r0)^2)`: the near field of a lens's veiling-glare
 * point-spread function falls roughly as the inverse square of the angle from
 * the source (Spencer et al., "Physically-Based Glare Effects for Digital
 * Images", SIGGRAPH 1995, after Vos). The previous `1 / (1 + 14r)` fell as
 * the inverse *first* power, so its tail never ended — measured on corrected
 * captures it lifted every pixel of `sky-day-b`, with the sun 60° out of
 * frame, by about one output level, which is a veil, not a flare.
 */
const HALO_GAIN = 0.04;
const HALO_RADIUS = 0.035;

/**
 * How far outside the frame, in NDC units past the edge, the light can be
 * before the flare has gone. Past the edge the light is not imaged by the lens
 * at all and the only glare left is stray light off the barrel, which this
 * effect does not model.
 */
const OFFSCREEN_FADE = 0.3;

export class LensFlare {
  private camera: THREE.PerspectiveCamera | null = null;

  private readonly uNear = uniform(0.1);
  private readonly uFar = uniform(40000);
  /** Light position in screen uv, top-down like every other uv here. */
  private readonly uSourceUv: any = uniform(new THREE.Vector2(0.5, 0.5));
  /** Cosine of the angle between the view axis and the light. */
  private readonly uAxisDot = uniform(0);
  private readonly uSourceColor: any = uniform(new THREE.Color(1, 1, 1));
  private readonly uSourceIntensity = uniform(0);
  /** 0 above water, 1 below. The effect is above-water only. */
  private readonly uSubmersion = uniform(0);
  /** Master scale, from the quality tier. */
  private readonly uEnabled = uniform(1);
  /** Elevation of the light, for the horizon fade. */
  private readonly uElevation = uniform(1);
  /** 1 with the light in frame, falling to 0 `OFFSCREEN_FADE` past the edge. */
  private readonly uInFrame = uniform(0);

  /** Scratch — this runs every frame and must not allocate. */
  private readonly _world = new THREE.Vector3();
  private readonly _forward = new THREE.Vector3();

  setCamera(camera: THREE.PerspectiveCamera): void {
    this.camera = camera;
    this.uNear.value = camera.near;
    this.uFar.value = camera.far;
  }

  /**
   * The key light this frame.
   *
   * @param direction Unit vector *toward* the light, world space.
   * @param color     The light's own colour. Must come from the live light, not
   *                  from `atmosphere.sunColor` — see the note on the moon above.
   * @param intensity Roughly 0..1, scaling how strongly the flare reads.
   */
  setSource(direction: THREE.Vector3, color: THREE.Color, intensity: number): void {
    const camera = this.camera;
    if (camera === null) return;

    // A direction, not a position: the light is at infinity. Projected from a
    // point far along it, which for a perspective projection is the same thing
    // and avoids dividing by a vanishing w.
    this._world.copy(direction).multiplyScalar(1e5).add(camera.position);
    this._world.project(camera);

    // `project` returns NDC, whose y runs bottom-up; screen uv here runs
    // top-down, the same as everywhere else in this project. That asymmetry is
    // what `ScreenSpaceReflection.clipToScreenUV` documents and what once had
    // the lens-rain droplets running up the screen.
    this.uSourceUv.value.set(this._world.x * 0.5 + 0.5, 1 - (this._world.y * 0.5 + 0.5));

    // The angular gate. Behind the camera this goes negative and the smoothstep
    // in the shader closes, so there is no separate in-front test to disagree
    // with it at the boundary.
    camera.getWorldDirection(this._forward);
    this.uAxisDot.value = this._forward.dot(direction);

    // The frame gate. The angular one below fades over tens of degrees, so on
    // its own it left the flare at full strength with the sun 60° off axis and
    // well outside the frame — a glare veil over a shot that has no sun in it.
    // `project` is only meaningful in front of the camera; behind it the
    // angular gate is already shut.
    const outside = Math.max(Math.abs(this._world.x) - 1, Math.abs(this._world.y) - 1, 0);
    const t = THREE.MathUtils.clamp(1 - outside / OFFSCREEN_FADE, 0, 1);
    this.uInFrame.value = this.uAxisDot.value > 0 ? t * t * (3 - 2 * t) : 0;

    this.uSourceColor.value.copy(color);
    this.uSourceIntensity.value = Math.max(0, intensity);
    this.uElevation.value = direction.y;
  }

  setSubmersion(value: number): void {
    this.uSubmersion.value = THREE.MathUtils.clamp(value, 0, 1);
  }

  setEnabled(on: boolean): void {
    this.uEnabled.value = on ? 1 : 0;
  }

  /**
   * @param sceneColor Any composited colour node — the flare is purely additive
   *   and never re-samples the image, so this imposes no `rtt`.
   * @param sceneDepth The depth *texture* node, for the occlusion taps.
   */
  build(sceneColor: unknown, sceneDepth: unknown): unknown {
    const colorNode: any = sceneColor;
    const depthNode: any = sceneDepth;
    if (colorNode === null || colorNode === undefined) {
      throw new Error('LensFlare.build: sceneColor is required.');
    }
    if (depthNode === null || depthNode === undefined || typeof depthNode.sample !== 'function') {
      throw new Error(
        'LensFlare.build: sceneDepth must be a texture node, e.g. scenePass.getTextureNode( "depth" ).',
      );
    }

    const viewDistance = (p: any): any =>
      perspectiveDepthToViewZ(depthNode.sample(p).r, this.uNear, this.uFar).negate();

    return Fn(() => {
      const src = vec4(colorNode).toVar('flareSrc');
      const out = vec3(src.rgb).toVar('flareOut');

      // Everything that can switch the effect off, resolved before any texture
      // work. All of these are uniform across the frame, so the branch below is
      // uniform-coherent and a submerged or night frame pays one compare.
      const angular = smoothstep(-0.25, 0.45, this.uAxisDot).toVar('flareAngular');
      const horizon = smoothstep(-0.02, 0.08, this.uElevation).toVar('flareHorizon');
      const dry = float(1).sub(this.uSubmersion).toVar('flareDry');
      const gate = angular
        .mul(this.uInFrame)
        .mul(horizon)
        .mul(dry)
        .mul(this.uEnabled)
        .mul(this.uSourceIntensity)
        .toVar('flareGate');

      If(gate.greaterThan(0.001), () => {
        const aspect = screenSize.x.div(screenSize.y.max(1)).toVar('flareAspect');
        /** Aspect-corrected offset from a point to the light, so discs stay round. */
        const toSource = (p: any): any => {
          const d = p.sub(this.uSourceUv);
          return vec2(d.x.mul(aspect), d.y);
        };

        // --- occlusion ------------------------------------------------------
        //
        // Eight taps on a small disc at the light's screen position. A tap is
        // clear when it reads at the far plane, which is what the sky returns.
        // The disc rather than a single tap is what makes a mast or a spar
        // *dim* the flare rather than switch it, so rigging crossing the sun
        // ramps instead of flickering.
        const clear = float(0).toVar('flareClear');
        const farEnough = this.uFar.mul(0.98).toVar('flareFarEnough');
        Loop(8, ({ i }: any) => {
          const a = float(i).mul(0.7853981634).toVar('flareOccA');
          const offset = vec2(a.cos(), a.sin()).mul(0.012).toVar('flareOccOff');
          const p = this.uSourceUv.add(vec2(offset.x.div(aspect), offset.y)).clamp(0, 1);
          clear.addAssign(viewDistance(p).greaterThan(farEnough).select(float(1), float(0)));
        });
        const visibility = smoothstep(0.0, 0.6, clear.div(8)).toVar('flareVis');

        const strength = gate.mul(visibility).toVar('flareStrength');
        const tint = vec3(this.uSourceColor).toVar('flareTint');

        const here = vec2(uv()).toVar('flareUv');
        const d = toSource(here).toVar('flareD');
        const r = d.length().toVar('flareR');

        // --- glare core -----------------------------------------------------
        //
        // Inverse-square falloff with a finite core; see `HALO_GAIN`.
        const rr = r.div(HALO_RADIUS).toVar('flareRr');
        const halo = float(HALO_GAIN).div(rr.mul(rr).add(1)).toVar('flareHalo');

        // --- anamorphic streak ----------------------------------------------
        //
        // One horizontal bar, faint and short. It was 0.05 tall and ran a fifth
        // of the frame each way; it is the element that most quickly turns a
        // flare into a costume, and a spherical lens barely has one.
        const streak = exp(d.x.mul(d.x).mul(-260))
          .mul(exp(d.y.mul(d.y).mul(-9000)))
          .mul(0.015)
          .toVar('flareStreak');

        // --- ghosts ---------------------------------------------------------
        //
        // On the line through the light and the optical centre; see
        // `GHOST_SCALES`. Unrolled: four ghosts, constants per ghost.
        const centred = vec2(here.x.sub(0.5).mul(aspect), here.y.sub(0.5)).toVar('flareCentred');
        const sourceCentred = vec2(this.uSourceUv.x.sub(0.5).mul(aspect), this.uSourceUv.y.sub(0.5))
          .toVar('flareSrcC');
        const ghostTint = vec3(0).toVar('flareGhostTint');
        for (let i = 0; i < GHOST_SCALES.length; i++) {
          const gd = centred.sub(sourceCentred.mul(GHOST_SCALES[i])).length();
          const g = smoothstep(float(GHOST_RADII[i]), float(0), gd).pow(2).mul(GHOST_GAIN);
          // Ghosts disperse: the further down the chain, the cooler. A single
          // white ghost chain is the other tell of a synthetic flare.
          ghostTint.addAssign(vec3(1 - i * 0.12, 1, 1 + i * 0.16).mul(g));
        }

        // Ghosts fade toward the frame edge rather than clipping at it.
        const edge = smoothstep(float(0.9), float(0.35), centred.length()).toVar('flareEdge');

        out.addAssign(tint.mul(halo.add(streak)).mul(strength));
        out.addAssign(ghostTint.mul(tint).mul(strength).mul(edge));
      });

      return vec4(out, src.a);
    })();
  }

  /** Owns no GPU resources; present so the chain's teardown stays uniform. */
  dispose(): void {}
}
