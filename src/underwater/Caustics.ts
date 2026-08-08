import * as THREE from 'three/webgpu';
import {
  Fn,
  clamp,
  exp,
  float,
  int,
  mx_worley_noise_float,
  pow,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * Procedural light caustics.
 *
 * The classic interlocking-web look comes from cellular (worley) noise rather
 * than from a scrolling texture: the distance-to-nearest-feature field of a
 * jittered point lattice already has exactly the right topology — closed cells
 * separated by thin bright ridges. Two octaves at different scales, drifting in
 * different directions, are multiplied together so the ridges of one break the
 * ridges of the other into the short, restless arcs real caustics have. A power
 * curve then crushes the midtones so only the ridges survive.
 *
 * Everything is a pure function of world position, so the same node can be
 * dropped into a seafloor material, a rock material and a hull material and the
 * pattern stays continuous across all of them — there is no projection texture
 * and no second render pass to keep in sync.
 *
 * Refraction at the surface is approximated by shearing the sample coordinate
 * along the sun's tilt: a point `d` metres below the surface is lit by the patch
 * of surface at `xz + (sunDir.xz / sunDir.y) * d`, which is what makes caustics
 * slide sideways as the sun moves toward the horizon.
 *
 * The spatial field is baked into one mipmapped render target per frame, so the
 * volumetric pass can sample it instead of re-running both Worley lattices at
 * every march step. There are no external assets or compute-only dependencies;
 * the bake and all consumers compile to WGSL and GLSL alike.
 */

export interface CausticsParams {
  /** Peak intensity of the web, before depth fade. */
  strength: number;
  /** Cells per metre. Larger = finer, busier caustics. */
  scale: number;
  /** Animation rate of the cell drift. */
  speed: number;
  /** Extinction per metre applied to the caustic below the surface. */
  depthFade: number;
  /** Exponent of the sharpening curve. Higher = thinner, brighter filaments. */
  sharpness: number;
  /** World Y of the water surface the caustics are cast from. */
  surfaceLevel: number;
}

export const DEFAULT_CAUSTICS_PARAMS: CausticsParams = {
  strength: 1.0,
  scale: 0.34,
  speed: 0.35,
  depthFade: 0.055,
  sharpness: 9,
  surfaceLevel: 0,
};

/** Clock wrap, seconds — long enough to be invisible, short enough for float32. */
const CLOCK_WRAP = 3600;

/** The sun tilt is clamped so a sun near the horizon cannot shear to infinity. */
const MAX_TILT = 3;

export class Caustics {
  private readonly params: CausticsParams;

  // --- uniforms ------------------------------------------------------------
  private readonly uTime = uniform(0);
  private readonly uStrength = uniform(DEFAULT_CAUSTICS_PARAMS.strength);
  private readonly uScale = uniform(DEFAULT_CAUSTICS_PARAMS.scale);
  private readonly uSpeed = uniform(DEFAULT_CAUSTICS_PARAMS.speed);
  private readonly uDepthFade = uniform(DEFAULT_CAUSTICS_PARAMS.depthFade);
  private readonly uSharpness = uniform(DEFAULT_CAUSTICS_PARAMS.sharpness);
  private readonly uSurfaceY = uniform(DEFAULT_CAUSTICS_PARAMS.surfaceLevel);
  /** Metres of horizontal shear per metre of depth, from the sun's elevation. */
  private readonly uSunTilt = uniform(new THREE.Vector2(0, 0));

  /**
   * The node graph, built exactly once. TSL functions are lazily compiled at
   * shader build time, so constructing this in the constructor is safe even
   * though there is no assign stack yet.
   */
  private readonly fn: any;
  private readonly fnLod: any;
  /** The web as a function of surface xz, before depth shear and extinction. */
  private readonly surfaceFn: any;

  /** Baked field. Stable reference — safe for materials to bind once. */
  readonly texture: THREE.Texture;
  /** World size of the square the baked field covers. */
  readonly extent: number;
  /** Texels per side. Updated by `setResolution` without replacing the texture. */
  resolution: number;

  private readonly target: THREE.RenderTarget;
  private readonly bakeMaterial: THREE.NodeMaterial;
  private readonly quad = new THREE.QuadMesh();
  /** World centre of the baked region. */
  private readonly uCenter = uniform(new THREE.Vector2());

  private clock = 0;
  private disposed = false;

  /**
   * @param resolution Texels per side of the baked field.
   * @param extent World metres the field spans.
   *
   * The pair is chosen for metres-per-texel, not for either number alone. Caustic
   * filaments are decimetre-scale features; at the first attempt's 512 over 480 m
   * — nearly a metre per texel — they came out as a soft radial glow rather than
   * distinct blades. 768 over 320 m is 0.42 m/texel and resolves them. Shrinking
   * the extent costs nothing real: extinction has closed the pattern down to a
   * few percent by 50 m of depth, so a region much larger than the water you can
   * actually see through is texels spent on black.
   */
  constructor(resolution = 768, extent = 320) {
    this.params = { ...DEFAULT_CAUSTICS_PARAMS };
    this.resolution = normalizeResolution(resolution);
    this.extent = extent;

    /**
     * The web itself, as a function of a point *on the surface*.
     *
     * Deliberately knows nothing about depth. Depth enters twice — as the
     * horizontal shear of the refracted ray and as the extinction along it — and
     * both are cheap closed forms applied at sample time. Keeping them out of
     * here is what lets the expensive part be evaluated once per surface texel
     * per frame instead of once per shaded fragment.
     */
    this.surfaceFn = Fn(([surfaceXZ]: any) => {
      const s = vec2(surfaceXZ).mul(this.uScale).toVar('causticsS');
      const t = this.uTime.mul(this.uSpeed);

      // `mx_worley_noise_float` is overloaded on the point type; the typings
      // only model the two-argument form, hence the loose view.
      const worley = mx_worley_noise_float as unknown as (
        p: unknown,
        jitter: number,
        metric: unknown,
      ) => any;

      // Octave A: the large cells. Time is the third noise axis, so the pattern
      // boils rather than sliding rigidly.
      const a = worley(vec3(s.x, s.y, t), 1.0, int(0));

      // Octave B: finer, counter-drifting, offset so the lattices never align.
      const q = s.mul(1.87).add(vec2(13.7, -5.3));
      const b = worley(vec3(q.x, q.y, t.mul(-1.43).add(31.0)), 1.0, int(0));

      // The filaments live where the two distance fields agree: `a - b == 0` is
      // a set of curves through the plane, and because the two lattices have
      // different scales and drift in opposite directions those curves close on
      // themselves and cross — which is exactly the interlocking web. Raising it
      // to a high power is what turns a broad ramp into a thin bright line.
      const filament = pow(clamp(float(1).sub(a.sub(b).abs()), 0.0, 1.0), this.uSharpness);

      // A much dimmer pool of light inside each cell, so the web sits on top of
      // a soft dapple instead of floating on black.
      const ridgeA = clamp(float(1).sub(a), 0.0, 1.0);
      const ridgeB = clamp(float(1).sub(b), 0.0, 1.0);
      const dapple = pow(ridgeA.mul(ridgeB), 2.5).mul(0.45);

      return filament.add(dapple);
    });

    // --- the baked field ----------------------------------------------------
    this.target = new THREE.RenderTarget(this.resolution, this.resolution, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      // Mipmapped, and the god-ray march depends on it.
      //
      // The march steps several metres between samples while the field's texel
      // is 0.42 m, so it point-samples a signal it is nowhere near resolving.
      // With no mip chain to fall back on, the result is variance — and because
      // the march's start offset is an interleaved gradient noise, that variance
      // arrives as a *stationary screen-space lattice* rather than as noise,
      // which is what showed up as a fine grid over the seafloor.
      //
      // Implicit LOD cannot fix it either: inside the loop, adjacent pixels are
      // sampling positions metres apart, so the derivative the hardware infers is
      // meaningless. The consumer has to ask for the level that matches its own
      // sampling rate, which is what `intensityNode`'s `lod` is for.
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      // Clamped, not repeated. The pattern is anchored in the world; wrapping it
      // would tile a visibly identical web onto the far side of the region.
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: true,
    });
    this.texture = this.target.texture;

    this.bakeMaterial = new THREE.NodeMaterial();
    this.bakeMaterial.depthTest = false;
    this.bakeMaterial.depthWrite = false;
    this.bakeMaterial.fragmentNode = Fn(() => {
      const world = uv().sub(0.5).mul(this.extent).add(this.uCenter);
      return vec4(this.surfaceFn(world), 0, 0, 1);
    })();

    /**
     * Sampling entry point: the same quantity the old analytic version returned,
     * but as one texture fetch plus two closed forms.
     */
    // Two variants of one field: one that lets the hardware pick the mip level
    // from its own derivatives, and one that is told the level explicitly.
    //
    // A surface being shaded wants the first. Its neighbouring fragments are
    // neighbouring points on the floor, so the derivative is exactly the right
    // footprint — and at a grazing view over the seafloor that footprint spans
    // many texels, which is the case that was aliasing into a visible lattice.
    // Forcing level 0 there defeated the mip chain entirely.
    //
    // The volumetric march wants the second. Its neighbouring fragments sample
    // points metres apart along different rays, so the inferred derivative is
    // meaningless; it has to state the level its own step length implies.
    this.fn = Fn(([worldPosition]: any) => this.field(worldPosition, null));
    this.fnLod = Fn(([worldPosition, lod]: any) => this.field(worldPosition, lod));

    this.setSunDirection(new THREE.Vector3(0.35, 0.62, 0.7));
  }

  /**
   * Resizes the mipmapped bake target in place.
   *
   * The render target is already referenced by both the seafloor and the
   * underwater node graphs through `texture`. `RenderTarget.setSize` retires
   * the old GPU allocation while retaining that texture object, which makes a
   * tier change safe after the application's submitted-work fence. The next
   * `bake` repopulates the resized mip chain.
   */
  setResolution(resolution: number): void {
    if (this.disposed) return;
    const next = normalizeResolution(resolution);
    if (next === this.resolution) return;
    this.target.setSize(next, next);
    this.resolution = next;
  }

  /** Shared body of both sampler variants. `lod` null selects implicit LOD. */
  private field(worldPosition: any, lod: any): any {
    return (() => {
      const p = vec3(worldPosition).toVar('causticsP');

      // Depth below the surface; above water there is nothing to cast.
      const below = this.uSurfaceY.sub(p.y).max(0.0).toVar('causticsDepth');

      // Walk the refracted ray back up to the surface patch that lit this point.
      const surfaceXZ = vec2(p.x, p.z).add(this.uSunTilt.mul(below)).toVar('causticsXZ');

      const coord = surfaceXZ.sub(this.uCenter).div(this.extent).add(0.5).toVar('causticsUv');

      // Outside the baked region there is no pattern to read, and clamp-to-edge
      // would streak the border across the whole seafloor. Fade instead.
      const edge = coord.min(coord.oneMinus()).toVar();
      const inside = edge.x.min(edge.y).smoothstep(0, 0.03).clamp(0, 1).toVar();

      const sampled = texture(this.texture, coord);
      const web = (lod === null ? sampled : sampled.level(lod)).r.mul(inside).toVar();
      const fade = exp(below.mul(this.uDepthFade).negate());

      return web.mul(this.uStrength).mul(fade);
    })();
  }

  /**
   * Re-bakes the field around `centerX, centerZ`.
   *
   * One 512² pass per frame, against the alternative of evaluating two 3D worley
   * lattices per shaded fragment. That was affordable while the only consumer
   * was the seafloor; it stopped being affordable the moment the volumetric
   * shafts began asking for the same value at every step of a 24-sample march,
   * which is around fifty hash lookups per step per pixel and took the frame
   * from milliseconds to seconds.
   *
   * Must run outside an active render target; it saves and restores its own.
   */
  bake(renderer: THREE.WebGPURenderer, centerX: number, centerZ: number): void {
    if (this.disposed) return;

    // Quantised to a texel, so the pattern does not crawl sub-texel against the
    // world as the camera creeps — the same reason the ocean mesh snaps.
    const texel = this.extent / this.resolution;
    const centre = this.uCenter.value as THREE.Vector2;
    centre.x = Math.round(centerX / texel) * texel;
    centre.y = Math.round(centerZ / texel) * texel;

    const previous = renderer.getRenderTarget();
    this.quad.material = this.bakeMaterial;
    renderer.setRenderTarget(this.target);
    this.quad.render(renderer);
    renderer.setRenderTarget(previous);
  }

  /**
   * Caustic intensity, in the range `[0, strength]`, at a world-space point.
   *
   * Pure: the result depends only on `worldPosition` and the uniforms, so it can
   * be shared by any number of materials.
   *
   * @param worldPosition A `vec3` node — typically `positionWorld`.
   * @param lod Explicit mip level. 0 for a surface being shaded, where the
   *   hardware's own derivatives would be right anyway; higher for a consumer
   *   that samples far apart, such as the volumetric march. Passing the level
   *   that matches the caller's sampling rate is what turns aliasing into blur,
   *   and blur is what a shaft sampled every few metres should look like.
   */
  intensityNode(worldPosition: unknown, lod: unknown = null): unknown {
    return lod === null ? this.fn(worldPosition) : this.fnLod(worldPosition, lod);
  }

  setParams(p: Partial<CausticsParams>): void {
    if (p.strength !== undefined) this.params.strength = p.strength;
    if (p.scale !== undefined) this.params.scale = p.scale;
    if (p.speed !== undefined) this.params.speed = p.speed;
    if (p.depthFade !== undefined) this.params.depthFade = p.depthFade;
    if (p.sharpness !== undefined) this.params.sharpness = p.sharpness;
    if (p.surfaceLevel !== undefined) this.params.surfaceLevel = p.surfaceLevel;

    this.uStrength.value = Math.max(0, this.params.strength);
    this.uScale.value = Math.max(1e-4, this.params.scale);
    this.uSpeed.value = this.params.speed;
    this.uDepthFade.value = Math.max(0, this.params.depthFade);
    this.uSharpness.value = Math.max(1, this.params.sharpness);
    this.uSurfaceY.value = this.params.surfaceLevel;
  }

  getParams(): Readonly<CausticsParams> {
    return this.params;
  }

  /** `dir` points *toward* the sun and need not be normalised. */
  setSunDirection(dir: THREE.Vector3): void {
    // A sun below ~9 degrees would shear the projection arbitrarily far; clamp
    // the elevation rather than the result so the tilt stays continuous.
    const y = Math.max(0.15, Math.abs(dir.y));
    const tilt = this.uSunTilt.value;
    tilt.set(
      clampNumber(dir.x / y, -MAX_TILT, MAX_TILT),
      clampNumber(dir.z / y, -MAX_TILT, MAX_TILT),
    );
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  /** Rewinds the animation clock, for reproducible captures. */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.target.dispose();
    this.bakeMaterial.dispose();
    this.quad.geometry.dispose();
  }
}

function normalizeResolution(size: number): number {
  return Number.isFinite(size) ? Math.max(1, Math.round(size)) : 768;
}

function clampNumber(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
