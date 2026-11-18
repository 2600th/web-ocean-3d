import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  cameraPosition,
  exp,
  float,
  hash,
  instanceIndex,
  smoothstep,
  storage,
  texture,
  uniform as threeUniform,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDown } from '../core/tslMath';
import { particleFragment, particleVertex, sprayMaterial } from './sprayLook';

/**
 * Spindrift: spray the wind tears off breaking crests and streams downwind.
 *
 * After the method in "Interactive Water Simulation in Atlas" (GDC 2019): emit
 * particles on the sea surface across the view, and kill every one that is not
 * sitting on a whitecap in the foam texture. What survives lifts off the crest
 * and is carried by the wind. The whitecap signal is the persistent foam
 * buffer's *freshness* channel (`physics/Wake`, B): a texel is fresh while a
 * crest is breaking over it and for about a second after, and rain foam never
 * is, so spray comes only off water that is actually breaking now.
 *
 * **GPU only.** Two storage buffers, stepped by one compute pass a frame, drawn
 * as one instanced draw of camera-facing quads. The compute-then-instanced-
 * sprite structure is three.js's `webgpu_compute_particles` example (r187,
 * MIT; see THIRD_PARTY_NOTICES.md): `storage` buffers stepped by a
 * `Fn().compute(count)` and read back in the vertex stage through
 * `toAttribute()`. What a particle looks like once it is somewhere is
 * `sprayLook`, shared with the hull and bow spray: velocity-stretched quads,
 * soft against the depth behind them, lit by the sun with a forward-scatter
 * lobe and by the sky, so it darkens at night like every other spray.
 *
 * **Per-slot respawn.** A dead slot tries to spawn with a probability per
 * frame proportional to `dt` (so the rate does not depend on frame rate) and to
 * how far the wind is above the threshold. Its spawn point is a uniform random
 * point in the camera's horizontal view sector, out to the foam buffer's edge;
 * the foam there decides whether it lives. The pool is the budget: in a storm
 * most slots are alive, so the count per tier sets how dense the spindrift is.
 *
 * **Deterministic.** The only randomness is a PCG hash of the slot index and a
 * frame counter, and `reset` puts every slot back to dead and the counter to
 * zero, so the same steps after `resetDeterministic` give the same spray.
 *
 * **Cost.** One compute dispatch over `count` slots (a texture tap and three
 * cascade taps for a slot that spawns, a dozen flops for one in flight) and
 * `count` instanced quads through `particleVertex`; budgeted at about 0.5 ms
 * for High's 60,000. Nothing at all is dispatched or drawn below the wind
 * threshold once the last particle has expired, nor at Low and Medium.
 *
 * **Depth.** The dense classes (drops and clusters) are drawn a second time
 * into depth only, before the colour, as the hull spray's are: without it the
 * volumetric fog fogged spindrift as whatever lay behind it, a crest's spray
 * 150 m off as the sea at the horizon or the sky past it, and a dense patch of
 * breaking crests in daylight came out a grey smudge. The mist never clears
 * the depth pass's alpha test and still soft-fades into the sea. The second
 * draw is vertex work over the same slots, about a fifth of the colour pass.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Slots allocated, the largest tier's count. 2 x 16 B each: 3.2 MB. */
export const SPINDRIFT_MAX = 100_000;

/** Longest a particle of any class lives, seconds. */
const MAX_LIFE = 3.4;

/** Wind, m/s, below which no spindrift is torn off; full by `WIND_FULL`. */
const WIND_ONSET = 9;
const WIND_FULL = 20;

/** Furthest spawn from the eye, metres (also capped by the foam buffer). */
const RANGE = 230;

/** Nearest spawn from the eye, metres. */
const NEAR = 4;

/** Spawn attempts per dead slot per second at full wind. */
const ATTEMPT_RATE = 30;

/** Near-surface wind as a share of the 10 m wind the sliders set. */
const SURFACE_WIND = 0.75;

const GRAVITY = 9.81;

/** Marks a slot dead: any age past every lifespan. */
const DEAD_AGE = 1e4;

/**
 * The spray's brightness against the display's white, as the water's foam is
 * drawn: `sprayLook` lights in scene units, and under a daylight exposure of
 * 0.42 its whitest cluster sat at sRGB 140 beside a whitecap at 240, so the
 * spray off a crest read grey against the crest it came from. Scaled by
 * `1 / exposure`, it is white in daylight and as dim as the sea at night.
 */
const DISPLAY_WHITE = 0.9;

/**
 * Distance from the eye over which mist and clusters fade in, metres. A crest
 * breaking beside the camera threw quads metres wide across the lens, and in
 * the storm the nearest crest stood as a white wall over a third of the frame.
 */
const MIST_NEAR: readonly [number, number] = [5, 16];
const CLUSTER_NEAR: readonly [number, number] = [2, 7];

/** Fragment classes, as `sprayLook` numbers them. */
const CLS_MIST = 2;
const CLS_CLUSTER = 3;

export class Spindrift {
  readonly mesh: THREE.Mesh;
  /** The compute passes, for the boot prewarm. */
  readonly computeNodes: any[];

  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly geometry: THREE.InstancedBufferGeometry;
  /** xyz = position, w = age (s). */
  private readonly posAge: any;
  /** xyz = velocity, w = the surface height it left. */
  private readonly velBase: any;
  private readonly updateNode: any;
  private readonly resetNode: any;

  // Compute uniforms: three's own, in the per-object group, because the app's
  // render-group `uniform` is only refreshed by renders (see `core/uniform`).
  private readonly cDt = threeUniform(0);
  private readonly cFrame = (threeUniform as any)(0, 'uint');
  private readonly cCount = (threeUniform as any)(0, 'uint');
  private readonly cEye = threeUniform(new THREE.Vector3());
  private readonly cHeading = threeUniform(0);
  private readonly cHalfAngle = threeUniform(Math.PI);
  private readonly cRange = threeUniform(RANGE);
  private readonly cAttempt = threeUniform(0);
  private readonly cGate = threeUniform(0);
  private readonly cWind = threeUniform(new THREE.Vector3());
  private readonly cFoamCenter = threeUniform(new THREE.Vector2());
  private readonly cFoamExtent = threeUniform(420);

  /** Draw-side wind gate, so a dying gale's last spray thins as it goes. */
  private readonly uGate = uniform(0);
  /** `DISPLAY_WHITE / exposure`; see there. */
  private readonly uDisplayGain = uniform(DISPLAY_WHITE / 0.42);
  private readonly depthMaterial: THREE.MeshBasicNodeMaterial;

  private count = 0;
  private frame = 0;
  /** Seconds since the gate was last open; past `MAX_LIFE` nothing is alive. */
  private quiet = Number.POSITIVE_INFINITY;
  private disposed = false;

  /**
   * @param foam         The wake buffer's resolved texture (R sea foam, B freshness).
   * @param displacement `OceanSimulation.displacementNode()`: the surface's
   *                     displacement at a rest-grid XZ, so a particle starts on
   *                     the displaced crest the foam texel belongs to.
   */
  constructor(foam: THREE.Texture, displacement: (worldXZ: any) => any) {
    const initial = new Float32Array(SPINDRIFT_MAX * 4);
    for (let i = 0; i < SPINDRIFT_MAX; i++) initial[i * 4 + 3] = DEAD_AGE;
    const posAttr = new THREE.StorageInstancedBufferAttribute(initial, 4);
    const velAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(SPINDRIFT_MAX * 4), 4);
    this.posAge = (storage as any)(posAttr, 'vec4', SPINDRIFT_MAX);
    this.velBase = (storage as any)(velAttr, 'vec4', SPINDRIFT_MAX);

    this.updateNode = (Fn(() => this.buildUpdate(foam, displacement))() as any)
      .compute(SPINDRIFT_MAX)
      .setName('Spindrift update');
    this.resetNode = (Fn(() => {
      If(instanceIndex.lessThan(uint(SPINDRIFT_MAX)), () => {
        this.posAge.element(instanceIndex).assign(vec4(0, -1000, 0, DEAD_AGE));
        this.velBase.element(instanceIndex).assign(vec4(0, 0, 0, 0));
      });
    })() as any)
      .compute(SPINDRIFT_MAX)
      .setName('Spindrift reset');
    this.computeNodes = [this.updateNode, this.resetNode];

    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    this.geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.geometry.instanceCount = 0;
    // Placed entirely in the vertex stage.
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

    this.material = sprayMaterial(() => this.buildVertex(), false);
    // Display-referred, as the foam it comes off (DISPLAY_WHITE).
    this.material.colorNode = Fn(() => {
      const c: any = particleFragment();
      return vec4(c.xyz.mul(this.uDisplayGain), c.w);
    })() as any;
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'spindrift';
    this.mesh.frustumCulled = false;
    // With the hull spray: after the ocean, so the water's refraction never sees it.
    this.mesh.renderOrder = 6;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;

    // The dense classes into depth, before the colour (see the header).
    this.depthMaterial = sprayMaterial(() => this.buildVertex(), true);
    const depth = new THREE.Mesh(this.geometry, this.depthMaterial);
    depth.name = 'spindrift-depth';
    depth.frustumCulled = false;
    depth.renderOrder = 5;
    depth.castShadow = false;
    depth.receiveShadow = false;
    this.mesh.add(depth);
  }

  /** Slots in use, from the quality tier (0 at Low and Medium). */
  setCount(count: number): void {
    this.count = Math.max(0, Math.min(SPINDRIFT_MAX, Math.round(count)));
    this.cCount.value = this.count;
    if (this.count === 0) this.geometry.instanceCount = 0;
  }

  /**
   * Steps the particles. Call after the wake buffer has updated this frame, so
   * the foam it tests is the foam the water is drawing.
   *
   * @param windSpeed   10 m wind, m/s.
   * @param windBearing Direction the wind blows toward, radians in XZ.
   */
  update(
    dt: number,
    camera: THREE.PerspectiveCamera,
    windSpeed: number,
    windBearing: number,
    foamCenterX: number,
    foamCenterZ: number,
    foamExtent: number,
    renderer: THREE.WebGPURenderer,
  ): void {
    if (this.disposed) return;
    this.frame = (this.frame + 1) >>> 0;
    const t = Math.min(1, Math.max(0, (windSpeed - WIND_ONSET) / (WIND_FULL - WIND_ONSET)));
    const gate = t * t * (3 - 2 * t);
    this.uGate.value = gate;
    this.uDisplayGain.value = DISPLAY_WHITE / Math.max(0.05, renderer.toneMappingExposure);
    if (gate > 0) this.quiet = 0;
    else this.quiet += Math.max(0, dt);
    const live = this.count > 0 && this.quiet < MAX_LIFE + 0.25;
    this.geometry.instanceCount = live ? this.count : 0;
    if (!live || dt <= 0) return;

    camera.getWorldDirection(_dir);
    const horizontal = Math.hypot(_dir.x, _dir.z);
    this.cHeading.value = Math.atan2(_dir.z, _dir.x);
    const halfV = (camera.fov * Math.PI) / 360;
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    // Looking steeply down, the view's footprint surrounds the eye.
    this.cHalfAngle.value = horizontal < 0.45 ? Math.PI : Math.min(Math.PI, halfH + 0.25);
    this.cRange.value = Math.min(RANGE, foamExtent * 0.48);
    (this.cEye.value as THREE.Vector3).copy(camera.position);
    (this.cWind.value as THREE.Vector3).set(
      Math.cos(windBearing) * windSpeed * SURFACE_WIND,
      0,
      Math.sin(windBearing) * windSpeed * SURFACE_WIND,
    );
    (this.cFoamCenter.value as THREE.Vector2).set(foamCenterX, foamCenterZ);
    this.cFoamExtent.value = foamExtent;
    this.cDt.value = Math.min(dt, 0.1);
    this.cAttempt.value = Math.min(1, ATTEMPT_RATE * Math.min(dt, 0.1) * gate);
    this.cGate.value = gate;
    this.cFrame.value = this.frame;
    renderer.compute(this.updateNode);
  }

  /** Every slot dead and the hash counter rewound. See `resetDeterministic`. */
  reset(renderer: THREE.WebGPURenderer): void {
    if (this.disposed) return;
    this.frame = 0;
    this.quiet = Number.POSITIVE_INFINITY;
    this.geometry.instanceCount = 0;
    renderer.compute(this.resetNode);
  }

  dispose(): void {
    this.disposed = true;
    this.geometry.dispose();
    this.material.dispose();
    this.depthMaterial.dispose();
    this.updateNode.dispose?.();
    this.resetNode.dispose?.();
  }

  // ------------------------------------------------------------------ compute

  private buildUpdate(foam: THREE.Texture, displacement: (worldXZ: any) => any): any {
    If(instanceIndex.lessThan(this.cCount), () => {
      const pa: any = this.posAge.element(instanceIndex);
      const vb: any = this.velBase.element(instanceIndex);
      const k = classOf(instanceIndex);
      const lifeSpan: any = lifeSpanOf(instanceIndex, k);
      const age: any = pa.w.toVar('sdAge');

      If(age.lessThan(lifeSpan), () => {
        // In flight: linear drag toward the wind, exact over the step (the
        // closed form `Spray` uses, here one frame at a time).
        const drag: any = k.drop.mul(0.5).add(k.cluster.mul(0.8)).add(k.mist.mul(1.6)).toVar('sdDrag');
        // Mist is held up by the turbulence it rides in: a fraction of g.
        const g: any = k.mist.mul(0.8 - GRAVITY).add(GRAVITY);
        const vInf: any = vec3(this.cWind.x, g.negate().div(drag), this.cWind.z).toVar('sdVInf');
        const decay: any = exp(drag.negate().mul(this.cDt)).toVar('sdDecay');
        const v: any = vb.xyz.toVar('sdV');
        const p: any = pa.xyz
          .add(vInf.mul(this.cDt))
          .add(v.sub(vInf).mul(decay.oneMinus().div(drag)))
          .toVar('sdP');
        const vNew: any = vInf.add(v.sub(vInf).mul(decay));
        // Back into the sea: drops a little below the crest they left, mist further.
        const floor: any = vb.w.sub(k.mist.mul(1.0).add(0.5));
        const nextAge: any = p.y.lessThan(floor).select(float(DEAD_AGE), age.add(this.cDt));
        pa.assign(vec4(p, nextAge));
        vb.assign(vec4(vNew, vb.w));
      }).Else(() => {
        const base: any = instanceIndex.mul(uint(8)).add(this.cFrame.mul(uint(0x9e3779b1)));
        const rnd = (n: number): any => hash(base.add(uint(n)));
        If(rnd(0).lessThan(this.cAttempt), () => {
          // A point in the view sector, with density falling as 1/sqrt(r):
          // between uniform by area (which puts nearly everything far off,
          // where it is a few pixels) and uniform on screen (Atlas), so the
          // near crests get their share. A third within 40 m, 60% within 100.
          const theta: any = this.cHeading.add(rnd(1).mul(2).sub(1).mul(this.cHalfAngle));
          const rootNear = Math.sqrt(NEAR);
          const root: any = rnd(2).mul(this.cRange.sqrt().sub(rootNear)).add(rootNear);
          const rest: any = vec2(this.cEye.x, this.cEye.z)
            .add(vec2(theta.cos(), theta.sin()).mul(root.mul(root)))
            .toVar('sdRest');
          const foamUv: any = rest.sub(this.cFoamCenter).div(this.cFoamExtent).add(0.5).toVar('sdUv');
          const edge: any = foamUv.min(foamUv.oneMinus());
          const inside: any = edge.x.min(edge.y).greaterThan(0.01);
          const f: any = (texture(foam, foamUv) as any).level(0).toVar('sdFoam');
          // Breaking now (fresh) and white (foam): a crest, not a lace patch.
          // Freshness above 0.6 is 7% of the buffer in the storm shot and 3% at
          // 18 m/s; above 0.25 it was 17% and 8%, the whole aged sea.
          const whitecap: any = smoothstep(0.45, 0.9, f.z).mul(smoothstep(0.2, 0.6, f.x));
          If(inside.and(rnd(3).lessThan(whitecap)), () => {
            const d: any = displacement(rest).toVar('sdDisp');
            const windLen: any = vec2(this.cWind.x, this.cWind.z).length().max(1e-3);
            const along: any = vec3(this.cWind.x, 0, this.cWind.z).div(windLen);
            const across: any = vec3(along.z.negate(), 0, along.x);
            // Torn off at a share of the wind, with a little scatter across it.
            const horizontal: any = along
              .mul(windLen.mul(rnd(4).mul(0.35).add(0.25)))
              .add(across.mul(rnd(5).sub(0.5).mul(1.6)));
            const up: any = k.drop
              .mul(rnd(6).mul(2.3).add(1.2))
              .add(k.cluster.mul(rnd(6).mul(1.8).add(1.0)))
              .add(k.mist.mul(rnd(6).mul(1.4).add(0.6)))
              .mul(this.cGate.mul(0.4).add(0.6));
            const start: any = vec3(rest.x.add(d.x), d.y.add(rnd(7).mul(0.3).add(0.05)), rest.y.add(d.z));
            pa.assign(vec4(start, rnd(7).mul(this.cDt)));
            vb.assign(vec4(horizontal.add(vec3(0, up, 0)), d.y));
          });
        });
      });
    });
  }

  // ------------------------------------------------------------------ shading

  private buildVertex(): any {
    const pa: any = this.posAge.toAttribute().toVar('sdPA');
    const vb: any = this.velBase.toAttribute().toVar('sdVB');
    const k = classOf(instanceIndex);
    const seed: any = hash(instanceIndex.add(uint(0x5bd1e995))).toVar('sdSeed');
    const lifeSpan: any = lifeSpanOf(instanceIndex, k);
    const life: any = pa.w.div(lifeSpan).toVar('sdLife');
    const alive: any = life.lessThan(1).select(float(1), float(0)).toVar('sdAlive');
    const cls: any = k.mist.mul(CLS_MIST).add(k.cluster.mul(CLS_CLUSTER));

    // Widths, metres: fine droplets streak, clusters are a handful of drops
    // in one quad, mist is the smoking haze a gale strips off a crest.
    const width: any = k.drop
      .mul(seed.mul(0.03).add(0.03))
      .add(k.cluster.mul(seed.mul(0.35).add(0.25)).mul(life.mul(0.6).add(0.8)))
      .add(k.mist.mul(seed.mul(1.2).add(0.8)).mul(life.mul(1.5).add(0.5)))
      .toVar('sdWidth');
    const density: any = k.drop.mul(0.85).add(k.cluster.mul(0.75)).add(k.mist.mul(seed.mul(0.04).add(0.06)));
    const fadeIn: any = smoothstep(0, 0.08, life);
    const fadeOut: any = smoothstepDown(life, 0.55, 1);
    // The far field is the sea's own haze, not individual wisps: thousands of
    // mist quads 150 m off pile into one band along the horizon, and those
    // above the eye stood against the sky as a grey smear.
    const distance: any = cameraPosition.distance(pa.xyz);
    const farFade: any = k.drop
      .add(k.cluster.mul(smoothstepDown(distance, 90, 200)).mul(smoothstep(CLUSTER_NEAR[0], CLUSTER_NEAR[1], distance)))
      .add(k.mist.mul(smoothstepDown(distance, 45, 140)).mul(smoothstep(MIST_NEAR[0], MIST_NEAR[1], distance)));
    const opacity: any = density
      .mul(fadeIn)
      .mul(fadeOut)
      .mul(farFade)
      .mul(this.uGate.mul(0.6).add(0.4))
      .toVar('sdOpacity');

    return particleVertex({
      centre: pa.xyz,
      velocity: vb.xyz,
      life,
      alive,
      isDrop: k.drop,
      isSheet: float(0),
      // Lit partly as droplet clusters. Spindrift fresh off a crest is dense
      // spray that scatters many times, not the thin single-scattering mist a
      // hull slam leaves hanging: lit as pure mist (diffuse 0.2) it came out a
      // grey veil, half the brightness of the whitecap it rose from, over a
      // sunlit sea. The fragment shape is still mist's, from `cls`.
      isMist: k.mist.mul(0.4),
      isCluster: k.cluster.add(k.mist.mul(0.6)),
      cls,
      seed,
      width,
      streak: k.drop.mul(0.035).add(k.cluster.mul(0.03)).add(k.mist.mul(0.05)),
      opacity,
      soft: k.drop.mul(0.1).add(k.cluster.mul(0.3)).add(k.mist.mul(1.0)),
      range: RANGE,
    });
  }
}

/**
 * The class of a slot, from its index: half fine droplets, a quarter droplet
 * clusters, a quarter mist, interleaved with period four so any `count` keeps
 * the same mix.
 */
function classOf(index: any): { drop: any; cluster: any; mist: any } {
  const m: any = index.mod(uint(4)).toFloat();
  const mist: any = m.greaterThan(2.5).select(float(1), float(0));
  const cluster: any = m.greaterThan(0.5).and(m.lessThan(1.5)).select(float(1), float(0));
  return { drop: float(1).sub(mist).sub(cluster), cluster, mist };
}

/** Lifespan of a slot, seconds; fixed per slot. Mist hangs longest. */
function lifeSpanOf(index: any, k: { drop: any; cluster: any; mist: any }): any {
  const h: any = hash(index.add(uint(0x27d4eb2d)));
  return k.drop
    .mul(h.mul(0.6).add(0.7))
    .add(k.cluster.mul(h.mul(0.7).add(0.9)))
    .add(k.mist.mul(h.mul(1.5).add(1.8)));
}

const _dir = new THREE.Vector3();
