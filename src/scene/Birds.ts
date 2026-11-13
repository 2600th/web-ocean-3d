import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  faceDirection,
  mix,
  positionGeometry,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';

/**
 * A flock of gulls, animated entirely on the GPU.
 *
 * **No downloaded asset.** A rigged glTF gull was not obtainable under a licence
 * this project can verify, and at the distance these are seen — a 1.4 m wingspan
 * at 50 to 250 m, so a couple of dozen pixels across — instanced procedural
 * geometry is the right technique anyway. A skinned mesh would spend
 * a bone palette and a per-bird draw on detail that never reaches the framebuffer,
 * and would still need a flight model written by hand on top of it. Seven
 * triangles and a vertex shader get the two things that actually read at this
 * size: the silhouette, and the way it moves.
 *
 * The whole system is **one instanced draw**. Geometry is a single gull built in
 * code — a body, a delta tail, and one quad per wing — and every bird is that
 * geometry deformed and placed by the vertex stage. The CPU never writes a
 * transform.
 *
 * `InstancedBufferGeometry` on a plain `Mesh` rather than `InstancedMesh`, for
 * the same reason `Weather` and `Particles` do it: there is no per-instance
 * matrix to carry. Every transform here is derived on the GPU from the clock and
 * a handful of per-instance constants, so an `instanceMatrix` would be a buffer
 * of identity matrices multiplied into every vertex for nothing.
 *
 * **Everything is a pure function of one clock.** No integrator, no per-frame
 * state, no boids. Each bird runs a closed circuit whose position, velocity and
 * acceleration all have closed forms in the path parameter, which is what makes
 * `resetClock(t)` land on the exact pose for time `t` — the property the visual
 * regression harness depends on, and the property a simulation could not offer at
 * any price. It is also why the flock costs a single uniform write per frame.
 *
 * No compute, no storage textures, nothing WebGPU-only: the same node graph has
 * to compile on the WebGL2 fallback.
 */

/**
 * TSL node objects are structurally dynamic; composing per-component expressions
 * out of `attribute()` values produces `any`, which the overloaded typings then
 * resolve to the wrong constructor. Node-typed locals are therefore `any` by
 * design — the class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

/** `vec3` under a loose signature. See the note above. */
const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => any;

/**
 * Seeded, and seeded *here*.
 *
 * `core/random`'s `SEEDS` table is shared and owned elsewhere, so this file keeps
 * its own constant rather than adding a row to it. What matters is only that it
 * is a literal that never drifts: the flock must be identical on every load or a
 * sky baseline is comparing two different flocks and every diff is noise.
 */
const SEED = 0x9e11a5;

/**
 * Clock wrap, seconds. Matches the other animated systems.
 *
 * Unlike the particle fields, **every frequency in this file is quantised to a
 * whole number of cycles per wrap**, so the entire flock is exactly periodic with
 * this period and the wrap is invisible. That is not fussiness. A particle that
 * jumps when the clock reseats is one dot among six thousand; a gull that jumps
 * is a gull teleporting across the sky, and it would happen once an hour with no
 * way to reproduce it on demand.
 */
const CLOCK_WRAP = 3600;

/**
 * Instances the buffers are built for. `setCount` only moves `instanceCount`,
 * so a tier change costs nothing and — more importantly — bird `i` keeps the
 * same circuit whatever the count is. A tier change re-scales the flock instead
 * of reshuffling it.
 */
const MAX_BIRDS = 128;

/** Sub-flocks the birds are strung out into, so they travel as loose skeins. */
const FLOCK_GROUPS = 3;
/** Spread of a bird's start phase around its group's, radians. */
const GROUP_SPREAD = 0.55;

/**
 * How far the viewer may roam before the flock starts to follow, metres.
 *
 * Inside this radius the circuits are anchored to the world origin and the birds
 * are genuinely stationary objects — they parallax as the ship moves, which is
 * most of what sells their distance. Only once the viewer has sailed out of the
 * play area does the anchor slide, and then it slides *with* the viewer at
 * exactly their speed, so the flock never falls out of the world.
 *
 * The crossover is continuous in position, so nothing teleports: at the boundary
 * the anchor is already at the origin and only its velocity changes. A grid-snap
 * or a wrap volume — the trick `Particles` uses — would be cheaper and would pop,
 * which is survivable for marine snow and not for a bird.
 */
const HOME_RADIUS = 140;

/** Standard gravity, for the coordinated-turn bank angle. */
const GRAVITY = 9.81;

/**
 * Bank exaggeration.
 *
 * A coordinated turn banks at `atan(a_lat / g)`, and swept over these circuits
 * the mean lateral acceleration is 1.6 m/s² — a nine degree roll, which at this
 * angular size is indistinguishable from level flight. Birds also genuinely
 * over-bank relative to the coordinated ideal, because they use roll to *change*
 * heading rather than to hold a steady turn. Three takes the mean to 26 degrees,
 * which reads, and leaves the clamp below doing almost nothing.
 */
const BANK_GAIN = 3;

/**
 * Ceiling on roll, radians. Beyond ~55 degrees a flat gull is edge-on and reads
 * as a stick. Swept over every circuit it binds on 2% of samples, so it is a
 * guard on the tightest corners rather than the thing setting the look.
 */
const MAX_ROLL = 0.95;

/** Peak wing rotation about the shoulder, radians. */
const FLAP_AMPLITUDE = 0.7;

/**
 * Resting wing angle, radians. Gulls hold a shallow V, and it is what a bird
 * settles into when the flap amplitude gates off — without it a gliding bird is a
 * flat cross.
 */
const DIHEDRAL = 0.12;

/**
 * Phase lag from shoulder to wingtip, radians.
 *
 * The wing is not a plank. The tip trails the shoulder by roughly a quarter of a
 * stroke, and that lag is the single detail that separates a flapping bird from a
 * pair of hinged boards — it is what puts the travelling wave along the span.
 * Free here, because the flap angle is already evaluated per vertex.
 */
const TIP_LAG = 0.9;

/**
 * Wrist flex during the recovery stroke: the tip is drawn inboard and swept back.
 *
 * On the **upstroke**, not the downstroke. The downstroke is the power stroke and
 * the wing is held fully extended through it; it is the recovery that flexes, to
 * cut the drag of dragging a full aerofoil back up through the air. Folding on
 * the power stroke instead reads as a broken wing — the silhouette narrows at
 * exactly the moment the eye expects it to be widest.
 */
const FOLD_SHRINK = 0.28;
const FOLD_SWEEP = 0.1;

/** Softness of the flap/glide gate. Wide enough to take about a second. */
const GLIDE_WIDTH = 0.3;

/**
 * Collapse radius, metres.
 *
 * A seven-triangle gull is convincing at sixty metres and a paper aeroplane at
 * six. The near fade scales a bird to zero before the viewer can get inside it,
 * which costs one smoothstep and — because it happens in the vertex stage —
 * rasterises no fragments at all, unlike an alpha fade that would also drag the
 * whole flock into the transparent pass.
 */
const NEAR_HIDE = 6;
const NEAR_SHOW = 18;

/**
 * Strength of the light coming *through* a backlit wing.
 *
 * A gull's remiges are thin enough to be translucent, and a bird between the
 * viewer and a low sun visibly glows rather than going black. Cheap, and it is
 * the single most recognisable thing about seabirds at golden hour.
 */
const TRANSMISSION = 0.55;

/** Wing half-span of the authored geometry, metres, at instance scale 1. */
const HALF_SPAN = 0.65;
/** Lateral offset of the shoulder joint — the pivot the wing rotates about. */
const SHOULDER_X = 0.045;

export class Birds {
  /** Instances the buffers hold. `setCount` is clamped to this. */
  static readonly MAX_COUNT = MAX_BIRDS;

  readonly object: THREE.Object3D;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly mesh: THREE.Mesh;

  private count: number;
  private clock = 0;
  private disposed = false;
  /** What the caller asked for, kept apart from whether there is anything to draw. */
  private wantVisible = true;

  // --- uniforms --------------------------------------------------------------
  private readonly uTime = uniform(0);
  /** Centre of every circuit, world space. See `HOME_RADIUS`. */
  private readonly uAnchor = uniform(new THREE.Vector3());
  /** The viewer, for the near collapse only. */
  private readonly uViewer = uniform(new THREE.Vector3());
  /** Unit vector *toward* the sun. */
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(1.0, 0.95, 0.88));
  /** What an unlit bird settles to: sky-lit, not black. */
  private readonly uShadeColor = uniform(new THREE.Color(0.24, 0.3, 0.38));
  /** Direct-sun weight, faded out as the sun sets. See `setSunDirection`. */
  private readonly uSunGain = uniform(1);

  constructor(count = 24) {
    this.count = clampCount(count);

    this.geometry = buildFlockGeometry();
    this.geometry.instanceCount = this.count;

    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'gulls';
    // The flock is placed in world space by the vertex stage against a bound that
    // moves with the viewer, so a fixed bounding sphere is meaningless.
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;

    this.object = new THREE.Object3D();
    this.object.name = 'birds';
    // Positions are computed in world space, exactly as `Particles` does it, so
    // the container itself never moves and never needs its matrix recomputed.
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Birds actually drawn, 0..`Birds.MAX_COUNT`.
   *
   * Cheap: it moves an instance count, it does not rebuild a buffer. At 0 the
   * renderer skips the draw entirely rather than submitting an empty one.
   */
  setCount(count: number): void {
    const next = clampCount(count);
    if (next === this.count) return;
    this.count = next;
    this.geometry.instanceCount = next;
    this.applyVisibility();
  }

  setVisible(v: boolean): void {
    this.wantVisible = v;
    this.applyVisibility();
  }

  /**
   * The sun, for shading.
   *
   * The direct term is faded out as the sun drops below the horizon — derived
   * here rather than exposed, so callers only have to push the vector, and so the
   * flock cannot end up lit like midday against a night sky.
   */
  setSunDirection(dir: THREE.Vector3): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(dir);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();

    const above = clamp01((sun.y + 0.09) / 0.18);
    this.uSunGain.value = above * above * (3 - 2 * above);
  }

  /**
   * Sunlight and sky-light colours.
   *
   * `sun` is a hue, not a brightness — `Atmosphere.sunColor` is already
   * normalised to a peak of 1 and is exactly what this wants.
   *
   * `shade` is the light an *unlit* bird receives, and it is a blue-grey rather
   * than black on purpose: an underwing away from the sun is still lit by the
   * whole sky dome and by the water under it. It is emphatically **not** the
   * sky's own radiance. Handing this `Atmosphere.zenithColor` directly puts a
   * white belly at roughly the same luminance as the sky behind it and the flock
   * vanishes; a third to a half of it is the range that reads as a silhouette.
   */
  setLightColors(sun: THREE.Color, shade: THREE.Color): void {
    (this.uSunColor.value as THREE.Color).copy(sun);
    (this.uShadeColor.value as THREE.Color).copy(shade);
  }

  /**
   * Advances the clock and re-anchors the flock.
   *
   * Two `Vector3.copy`-class writes and one scalar, with no allocation. The clock
   * advances whether or not the flock is visible, so hiding the birds for a while
   * and showing them again resumes the flight they would have flown rather than
   * the one they were paused mid-stroke of.
   */
  update(dt: number, cameraPosition: THREE.Vector3): void {
    if (this.disposed) return;
    this.clock = (this.clock + dt) % CLOCK_WRAP;
    this.uTime.value = this.clock;
    this.anchorTo(cameraPosition);
  }

  /**
   * Jumps the flock to the pose for `time`.
   *
   * Exact, not approximate: every bird's position, heading, bank, wing angle and
   * glide state is a closed-form function of this clock and its own constants, so
   * there is no integrated state to be out of step. Rewinding the harness to the
   * same time twice produces the same frame to the bit.
   *
   * The anchor is not touched — it is a function of the camera, which the caller
   * owns and rewinds itself.
   */
  resetClock(time = 0): void {
    this.clock = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uTime.value = this.clock;
  }

  /**
   * Where bird `index` is now, world space, computed on the CPU from the same
   * closed form the vertex stage evaluates (the body only — no wing pose).
   *
   * For the audio, which places a gull's cry on a bird that is actually up
   * there. Allocation-free. False for an index the flock is not drawing.
   */
  positionOf(index: number, out: THREE.Vector3): boolean {
    if (!(index >= 0 && index < this.count)) return false;
    const orbit = this.geometry.getAttribute('orbit').array;
    const harmonics = this.geometry.getAttribute('harmonics').array;
    const bob = this.geometry.getAttribute('bob').array;
    const b = index * 4;
    const theta = orbit[b + 1] * this.clock + orbit[b + 2];
    const second = theta * 2 + harmonics[b + 1];
    const third = theta * 3 + harmonics[b + 3];
    const climb = theta * 2 + bob[b + 1];
    const radius = orbit[b];
    const anchor = this.uAnchor.value as THREE.Vector3;
    out.set(
      anchor.x + radius * Math.cos(theta) + harmonics[b] * Math.cos(second) + harmonics[b + 2] * Math.cos(third),
      anchor.y + orbit[b + 3] + bob[b] * Math.sin(climb),
      anchor.z + radius * Math.sin(theta) + harmonics[b] * Math.sin(second) + harmonics[b + 2] * Math.sin(third),
    );
    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.remove(this.mesh);
    this.object.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }

  // ------------------------------------------------------------------ internals

  /** An empty flock is hidden outright, so it costs not even a culling test. */
  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.count > 0;
  }

  /**
   * Slides the circuit centre so the flock stays reachable. See `HOME_RADIUS`.
   *
   * Horizontal only — the birds fly at an altitude above the sea, not above the
   * camera, so a viewer climbing the mast has to see them get closer.
   */
  private anchorTo(cameraPosition: THREE.Vector3): void {
    const anchor = this.uAnchor.value as THREE.Vector3;
    const x = cameraPosition.x;
    const z = cameraPosition.z;
    const distance = Math.sqrt(x * x + z * z);

    if (distance <= HOME_RADIUS) {
      anchor.set(0, 0, 0);
    } else {
      const k = (distance - HOME_RADIUS) / distance;
      anchor.set(x * k, 0, z * k);
    }

    (this.uViewer.value as THREE.Vector3).copy(cameraPosition);
  }

  private buildMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    // A wing is a sheet with no thickness, so both faces have to rasterise.
    material.side = THREE.DoubleSide;
    material.transparent = false;
    material.depthWrite = true;
    // Aerial perspective is applied by the volumetric fog pass from scene depth,
    // and these are opaque, so they get it for free. Vertex fog would double it.
    material.fog = false;

    // Two independent graphs rather than one shared set of `.toVar()` locals.
    // A TSL variable belongs to the flow it was declared in, and a varying opens
    // its own vertex-stage flow — sharing vars across the two is how you get a
    // shader that references a name it never declared. Rebuilding the frame costs
    // a few dozen vector ops on ~1800 vertices, which is nothing.
    material.positionNode = Fn(() => this.buildVertex().world)() as any;

    const shading: any = varying(
      Fn(() => {
        const vertex = this.buildVertex();
        return vec4(vertex.normal, vertex.outboard);
      })(),
    );

    const normal: any = shading.xyz.normalize();
    /** Distance out the span, 0 at the shoulder and 1 at the tip. */
    const outboard: any = shading.w;

    // `faceDirection` is +1 on the bird's back and -1 on its belly, because the
    // geometry is wound counter-clockwise seen from above. That one sign carries
    // both halves of a gull's plumage.
    const facing: any = faceDirection;
    const seen: any = normal.mul(facing);
    const ndl: any = seen.dot(this.uSunDir);

    // Wrapped diffuse rather than a clamped one. A hard terminator on a body this
    // small in screen space is not shading, it is a flicker as the wing crosses
    // it — the terminator is narrower than the bird.
    const direct: any = ndl.mul(0.5).add(0.5).mul(this.uSunGain);
    const through: any = ndl.negate().max(0).mul(TRANSMISSION).mul(this.uSunGain);
    const lit: any = direct.add(through).clamp(0, 1);

    // Grey mantle above, white below — the countershading that makes a gull read
    // as a gull the instant it banks and shows you the other side.
    const topness: any = facing.mul(0.5).add(0.5);
    const plumage: any = mix(vec3(0.95, 0.96, 0.97), vec3(0.6, 0.66, 0.72), topness);
    // Black primaries. Shaped rather than taken raw so the darkening stays out on
    // the hand of the wing instead of washing the whole span grey — the wing quad
    // only has vertices at root and tip, so the raw ramp is linear across it.
    const tipped: any = mix(plumage, vec3(0.1, 0.11, 0.13), outboard.smoothstep(0.6, 0.96));

    const light: any = mix(this.uShadeColor, this.uSunColor, lit);
    material.colorNode = vec4(tipped.mul(light), 1) as any;

    return material;
  }

  /**
   * The whole animation, per vertex.
   *
   * Read as: solve the circuit for where this bird is and which way it is
   * pointing, build a flight frame from that, deform the wing in the bird's own
   * axes, then place the deformed vertex in the frame. Nothing here reads a
   * texture or a previous frame.
   */
  private buildVertex(): { world: any; normal: any; outboard: any } {
    // `attribute()` is typed as an opaque node; swizzles need the loose view.
    const orbit: any = attribute('orbit', 'vec4');
    const harmonics: any = attribute('harmonics', 'vec4');
    const bob: any = attribute('bob', 'vec4');
    const style: any = attribute('style', 'vec4');
    const span: any = attribute('span', 'float');
    const rest: any = positionGeometry;

    const radius = orbit.x;
    const rate = orbit.y;
    const altitude = orbit.w;
    const secondAmp = harmonics.x;
    const thirdAmp = harmonics.z;
    const climbAmp = bob.x;

    // --- the circuit ---------------------------------------------------------
    //
    // A circle plus a second and third harmonic, all in the same parameter, so
    // the path is closed by construction: every term returns to itself after one
    // turn of theta. That is what keeps the flock in the play area with no
    // simulation and no bounds check — there is nowhere for a bird to drift *to*.
    //
    // `buildInstanceData` holds `2*B + 3*C < A`, which bounds the harmonics'
    // contribution to the derivative below the circle's own. The tangent can
    // therefore never vanish, and a vanishing tangent is a cusp: a bird that
    // stops dead and swings through ninety degrees of heading in one frame.
    const theta = rate.mul(this.uTime).add(orbit.z).toVar();
    const second = theta.mul(2).add(harmonics.y).toVar();
    const third = theta.mul(3).add(harmonics.w).toVar();
    const climb = theta.mul(2).add(bob.y).toVar();

    const s1 = theta.sin().toVar();
    const c1 = theta.cos().toVar();
    const s2 = second.sin().toVar();
    const c2 = second.cos().toVar();
    const s3 = third.sin().toVar();
    const c3 = third.cos().toVar();
    const sc = climb.sin().toVar();
    const cc = climb.cos().toVar();

    const offset = vec3n(
      radius.mul(c1).add(secondAmp.mul(c2)).add(thirdAmp.mul(c3)),
      altitude.add(climbAmp.mul(sc)),
      radius.mul(s1).add(secondAmp.mul(s2)).add(thirdAmp.mul(s3)),
    ).toVar();

    // dP/dtheta and d²P/dtheta², term by term. Both are wanted: the first is the
    // heading, the second is the bank. Differentiating the path analytically
    // rather than sampling it twice avoids the atan2 wrap that makes a
    // finite-difference turn rate spike once per lap.
    const tangent = vec3n(
      radius.mul(s1).negate().sub(secondAmp.mul(s2).mul(2)).sub(thirdAmp.mul(s3).mul(3)),
      climbAmp.mul(cc).mul(2),
      radius.mul(c1).add(secondAmp.mul(c2).mul(2)).add(thirdAmp.mul(c3).mul(3)),
    ).toVar();

    const bend = vec3n(
      radius.mul(c1).negate().sub(secondAmp.mul(c2).mul(4)).sub(thirdAmp.mul(c3).mul(9)),
      climbAmp.mul(sc).mul(-4),
      radius.mul(s1).negate().sub(secondAmp.mul(s2).mul(4)).sub(thirdAmp.mul(s3).mul(9)),
    ).toVar();

    // Chain rule into real units. `rate` carries the sign, so a bird with a
    // negative rate flies its circuit backwards and still faces the right way.
    const velocity = tangent.mul(rate).toVar();
    const accel = bend.mul(rate).mul(rate).toVar();

    // --- the flight frame ----------------------------------------------------
    const forward = velocity.normalize().toVar();
    const across = forward.cross(vec3(0, 1, 0)).toVar();
    // Guarded rather than normalised outright: the cross product collapses for a
    // vertically climbing bird. These never climb past a few degrees, but a NaN
    // frame would take the whole instance to infinity and there is no cheaper
    // insurance than a max.
    const right = across.div(across.length().max(1e-3)).toVar();
    const up = right.cross(forward).toVar();

    // Coordinated turn: bank until the lift vector's horizontal component
    // supplies the lateral acceleration the path demands. Dotting the
    // acceleration with `right` picks out exactly that component and discards the
    // tangential one, which is the part that changes speed rather than heading.
    const roll = accel
      .dot(right)
      .mul(BANK_GAIN / GRAVITY)
      .atan()
      .clamp(-MAX_ROLL, MAX_ROLL)
      .toVar();
    const rollCos = roll.cos().toVar();
    const rollSin = roll.sin().toVar();
    const bankRight = right.mul(rollCos).sub(up.mul(rollSin)).toVar();
    const bankUp = up.mul(rollCos).add(right.mul(rollSin)).toVar();

    // --- the wing ------------------------------------------------------------
    const outboard = span.abs().toVar();
    // Exactly 0 on the body and tail, so `sign` is 0 there and the fuselage is
    // untouched by everything below without needing a branch or a second
    // attribute to mark it.
    const side = span.sign().toVar();

    // The glide gate. A slow oscillator per bird, thresholded: below the bird's
    // own bias it stops flapping and holds the wings out. The *amplitude* is
    // gated, not the phase — freezing the phase would snap the wings to wherever
    // the stroke happened to be, while fading the amplitude settles them onto the
    // dihedral, which is the pose a gliding gull actually holds.
    const glide = style.y.mul(this.uTime).add(style.z).sin().toVar();
    const amplitude = glide.smoothstep(style.w.sub(GLIDE_WIDTH), style.w.add(GLIDE_WIDTH)).toVar();

    const stroke = bob.z.mul(this.uTime).add(bob.w).sub(outboard.mul(TIP_LAG)).toVar();
    const flap = stroke
      .sin()
      .mul(amplitude)
      .mul(FLAP_AMPLITUDE)
      .add(DIHEDRAL)
      .mul(side)
      .toVar();

    // The stroke's angular velocity is its cosine, so this is positive exactly
    // through the recovery half. See `FOLD_SHRINK`.
    const fold = stroke.cos().max(0).mul(amplitude).mul(outboard).toVar();
    const folded = vec3n(
      rest.x.mul(fold.mul(FOLD_SHRINK).oneMinus()),
      rest.y,
      rest.z.sub(fold.mul(FOLD_SWEEP)),
    ).toVar();

    // Rotate about the shoulder, not the centreline. Hinging at x = 0 would make
    // the body pump up and down with the wings.
    const pivot = side.mul(SHOULDER_X).toVar();
    const arm = folded.x.sub(pivot).toVar();
    const flapCos = flap.cos().toVar();
    const flapSin = flap.sin().toVar();

    // --- placement -----------------------------------------------------------
    const centre = this.uAnchor.add(offset).toVar();
    // See `NEAR_HIDE`. Scaling to zero costs no fragments; an alpha fade would.
    const size = style.x
      .mul(centre.sub(this.uViewer).length().smoothstep(NEAR_HIDE, NEAR_SHOW))
      .toVar();

    const local = vec3n(
      arm.mul(flapCos).sub(folded.y.mul(flapSin)).add(pivot),
      arm.mul(flapSin).add(folded.y.mul(flapCos)),
      folded.z,
    )
      .mul(size)
      .toVar();

    const world = centre
      .add(bankRight.mul(local.x))
      .add(bankUp.mul(local.y))
      .add(forward.mul(local.z));

    // The authored sheet's normal is +Y everywhere, so the flap rotation is the
    // only thing that moves it: rotating (0,1,0) about the local Z axis by the
    // flap angle gives this directly, with no cross products and no normal matrix.
    const normal = bankRight.mul(flapSin.negate()).add(bankUp.mul(flapCos));

    return { world, normal, outboard };
  }
}

/**
 * The gull: seven triangles, authored nose-forward.
 *
 * Local axes are **+X right wing, +Y up, +Z nose** — chosen to match the frame
 * the vertex stage builds, so placement is three multiply-adds and never a matrix.
 * Every triangle is wound counter-clockwise seen from above, which is what makes
 * `faceDirection` mean "this is the bird's back" in the fragment stage.
 *
 * `span` is the signed distance out the wing, normalised to the tip: its sign
 * mirrors the flap and its magnitude drives the tip lag, the wrist fold and the
 * dark primaries. It is exactly 0 on the body and tail, which is what excludes
 * them from all three.
 */
function buildFlockGeometry(): THREE.InstancedBufferGeometry {
  const positions = new Float32Array([
    //  x       y      z
     0.000,  0.000,  0.260, // 0  nose
     0.045,  0.000,  0.020, // 1  right flank
     0.000,  0.000, -0.170, // 2  hip
    -0.045,  0.000,  0.020, // 3  left flank
     0.085,  0.000, -0.320, // 4  right tail corner
    -0.085,  0.000, -0.320, // 5  left tail corner

     0.045,  0.000,  0.100, // 6  right shoulder, leading
     0.045,  0.000, -0.070, // 7  right shoulder, trailing
     0.620,  0.000, -0.050, // 8  right tip, leading
     0.650,  0.000, -0.140, // 9  right tip, trailing

    -0.045,  0.000,  0.100, // 10 left shoulder, leading
    -0.045,  0.000, -0.070, // 11 left shoulder, trailing
    -0.620,  0.000, -0.050, // 12 left tip, leading
    -0.650,  0.000, -0.140, // 13 left tip, trailing
  ]);

  const spans = new Float32Array(positions.length / 3);
  for (let i = 0; i < spans.length; i++) {
    // Body and tail are the first six vertices and stay at exactly 0.
    spans[i] = i < 6 ? 0 : positions[i * 3] / HALF_SPAN;
  }

  // The sheet is flat, so one normal serves every vertex; the flap rotation is
  // applied to it analytically in the shader. Supplied only so nothing three
  // reaches for it later finds the attribute missing.
  const normals = new Float32Array(positions.length);
  for (let i = 1; i < normals.length; i += 3) normals[i] = 1;

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('span', new THREE.Float32BufferAttribute(spans, 1));
  geometry.setIndex([
    0, 1, 3,  1, 2, 3,   // body
    2, 4, 5,             // tail
    6, 8, 9,  6, 9, 7,   // right wing
    10, 13, 12,  10, 11, 13, // left wing
  ]);

  const data = buildInstanceData();
  geometry.setAttribute('orbit', new THREE.InstancedBufferAttribute(data.orbit, 4));
  geometry.setAttribute('harmonics', new THREE.InstancedBufferAttribute(data.harmonics, 4));
  geometry.setAttribute('bob', new THREE.InstancedBufferAttribute(data.bob, 4));
  geometry.setAttribute('style', new THREE.InstancedBufferAttribute(data.style, 4));

  geometry.instanceCount = 0;
  // The flock is anchored near the viewer and placed in the vertex stage, so a
  // fixed bound is meaningless; culling is off on the mesh.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

  return geometry;
}

interface InstanceData {
  /** radius, angular rate (signed), start phase, altitude. */
  orbit: Float32Array;
  /** second harmonic amplitude and phase, third harmonic amplitude and phase. */
  harmonics: Float32Array;
  /** climb amplitude, climb phase, flap rate, flap phase. */
  bob: Float32Array;
  /** scale, glide-gate rate, glide-gate phase, glide bias. */
  style: Float32Array;
}

/**
 * One bird's worth of constants, `MAX_BIRDS` times, from a fixed seed.
 *
 * Drawn once at construction and never regenerated, which is what makes bird `i`
 * the same bird at every quality tier — `setCount` only changes how many of these
 * are drawn.
 *
 * Every rate is quantised by `perWrap` to a whole number of cycles per
 * `CLOCK_WRAP`. See the constant: it is what makes the clock's wrap seamless
 * instead of a once-an-hour teleport.
 */
function buildInstanceData(): InstanceData {
  const random = mulberry32(SEED);

  const orbit = new Float32Array(MAX_BIRDS * 4);
  const harmonics = new Float32Array(MAX_BIRDS * 4);
  const bob = new Float32Array(MAX_BIRDS * 4);
  const style = new Float32Array(MAX_BIRDS * 4);

  // Drawn before the per-bird loop so a bird's own draws stay at a fixed offset
  // in the sequence.
  //
  // Direction is per group, not per bird. Mixed directions inside one skein
  // would have half of it flying through the other half; separating them by
  // group gives the variety without the collisions.
  const groupPhase: number[] = [];
  const groupDirection: number[] = [];
  for (let g = 0; g < FLOCK_GROUPS; g++) {
    groupPhase.push(random() * Math.PI * 2);
    groupDirection.push(random() < 0.35 ? -1 : 1);
  }

  for (let i = 0; i < MAX_BIRDS; i++) {
    const base = i * 4;
    const radius = 45 + random() * 70;

    // Laps are derived from an *airspeed*, not drawn independently.
    //
    // A gull cruises at 10 to 13 m/s and a wide circuit at that speed simply takes
    // longer than a tight one. Drawing radius and rate separately instead put a
    // 45 m circuit and a 115 m circuit at the same lap time, which meant the
    // inner birds crawled at 3 m/s — slow enough that they read as hanging in the
    // air rather than flying, and slow enough that the bank angle, which goes as
    // the square of the rate, all but vanished.
    //
    // Rounding to a whole number of laps per wrap is what keeps the clock's wrap
    // seamless; it moves the realised speed by at most a few percent.
    const speed = 8.5 + random() * 4.5;
    const laps = Math.max(1, Math.round((speed * CLOCK_WRAP) / (2 * Math.PI * radius)));

    // The harmonics that turn the circle into a wandering circuit. Bounded so
    // that 2*second + 3*third stays below the radius: see the tangent note in
    // `buildVertex`.
    const second = radius * (0.08 + random() * 0.14);
    const third = radius * (0.03 + random() * 0.05);

    orbit[base] = radius;
    orbit[base + 1] = (groupDirection[i % FLOCK_GROUPS] * 2 * Math.PI * laps) / CLOCK_WRAP;
    // Strung out along the circuit in a few loose skeins rather than scattered:
    // a flock that is evenly spread around its own path reads as traffic.
    orbit[base + 2] =
      groupPhase[i % FLOCK_GROUPS] + (random() - 0.5) * GROUP_SPREAD * 2;
    orbit[base + 3] = 14 + random() * 44;

    harmonics[base] = second;
    harmonics[base + 1] = random() * Math.PI * 2;
    harmonics[base + 2] = third;
    harmonics[base + 3] = random() * Math.PI * 2;

    bob[base] = 1.5 + random() * 5;
    bob[base + 1] = random() * Math.PI * 2;
    // 2.3 to 3.6 Hz. A herring gull cruises around 3.
    bob[base + 2] = perWrap(2.3 + random() * 1.3);
    bob[base + 3] = random() * Math.PI * 2;

    // Wingspan 1.1 to 1.7 m, which is a herring gull either side of average.
    // Scale, and deliberately not a gull's.
    //
    // A herring gull spans 1.3 m; at the 60-140 m these circuits fly, that is
    // under ten pixels, and a measured 38 pixels of the *entire* flock survived
    // a 1600x900 frame. The viewer sees dust. Doubling it puts the span near an
    // albatross's, which is the readable size — the honest alternative is to fly
    // the flock close enough to intersect the ship's rigging.
    style[base] = 1.7 + random() * 0.9;
    // Glide gate period, 14 to 30 seconds.
    style[base + 1] = perWrap(1 / (14 + random() * 16));
    style[base + 2] = random() * Math.PI * 2;
    // Duty cycle, as a threshold on that oscillator. Skewed well negative: the
    // typical bird flaps about three quarters of the time, the most negative
    // never fully stops and only softens its stroke, and only the few at the top
    // of the range spend most of their circuit soaring. An even spread put half
    // the flock gliding at any instant, which reads as a flock of paper darts.
    style[base + 3] = -1.0 + random() * 1.3;
  }

  return { orbit, harmonics, bob, style };
}

/**
 * Rounds a frequency in Hz to the nearest whole number of cycles per
 * `CLOCK_WRAP` and returns it as radians per second.
 *
 * The rounding is the point: it is what makes every oscillator in the flock share
 * an exact common period with the clock, so `clock % CLOCK_WRAP` lands on a pose
 * identical to the one it left.
 */
function perWrap(hz: number): number {
  const cycles = Math.max(1, Math.round(hz * CLOCK_WRAP));
  return (2 * Math.PI * cycles) / CLOCK_WRAP;
}

function clampCount(count: number): number {
  if (!Number.isFinite(count)) return 0;
  return Math.max(0, Math.min(MAX_BIRDS, Math.floor(count)));
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
