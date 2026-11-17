import * as THREE from 'three/webgpu';
import {
  Fn,
  PI,
  attribute,
  cos,
  cross,
  float,
  int,
  mix,
  normalLocal,
  select,
  sin,
  smoothstep,
  uniform,
  uniformArray,
  vec3,
} from 'three/tsl';
import { CLOTH_STEP, ClothSolver, type ClothPatch } from './ClothSolver';
import { Pennant } from './Pennant';
import { readSailPlan, type SailSpec } from './sailRig';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * Canvas that sets, fills and luffs.
 *
 * Each sail is rebuilt as a regular grid between its yard and its clews (see
 * `sailRig`), with texture coordinates interpolated from the model's own sail
 * corners so the canvas texture still lands on it. Two ways to shape it, one
 * material:
 *
 *   - **Cloth** (High and up, `sailCloth`): verlet springs in compute
 *     (`ClothSolver`), the head pinned along the yard and the clews pinned,
 *     pressed by the apparent wind on each vertex's own normal. About 900
 *     vertices and 5,800 springs for the four sails and the pennant, four
 *     steps a frame at 60 Hz, eight small dispatches: well under the 0.5 ms
 *     budget.
 *   - **Analytic** (Low, Medium): a belly evaluated in the vertex stage, its
 *     depth and luff driven from the CPU with the same wind model.
 *
 * Both read the same per-sail table: how far the sail is let fall (furled is a
 * roll of canvas on the yard, half and full let the courses and then the
 * topsails down), its belly depth, its luff and its forward normal.
 *
 * The point of sail shows two ways. A sail with the wind across it at a shallow
 * angle is trimmed flat (its slack is taken out, which is what bracing sharp up
 * and hauling the bowlines does) and the pressure on it is small; one with the
 * wind from astern is eased and full. With the wind ahead it stops drawing and
 * shakes.
 *
 * **The yards do not brace.** They are not separate nodes: all four sit inside
 * the one joined rigging mesh with their lifts, footropes and braces as separate
 * islands that end at the yardarms, and `Birds` perches gulls on fixed yardarm
 * points. Turning a yard means bending those ropes with it (a vertex skin on the
 * rigging) and moving the perches, which is S2's territory. Until then the
 * sails keep the model's own brace, about 15 degrees with the starboard arm
 * forward, and the point of sail is expressed through trim, pressure and luff.
 */

export type SailSetting = 'furled' | 'half' | 'full';

export interface SailStateInput {
  /**
   * Gameplay's canvas fraction (0 furled, 0.55 half, 1 full, ramped as the crew
   * work), or `null` when nobody is at the helm and the ship chooses by wind.
   */
  fill: number | null;
  /** True wind, world space, m/s, the direction it blows toward. */
  trueWind: THREE.Vector3;
  /** Hull velocity, world space, m/s. */
  velocity: THREE.Vector3;
  /** Gameplay's luff, 0..1 (in irons, anchored, handling sail). */
  luff: number;
  /** Whether the tier runs the compute cloth (`QualitySettings.sailCloth`). */
  cloth: boolean;
}

/**
 * Apparent wind: true wind minus the hull's own velocity. One function so it
 * can be swapped for `gameplay/sailing.ts`'s when that exists.
 */
export function apparentWind(trueWind: THREE.Vector3, velocity: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
  return out.copy(trueWind).sub(velocity);
}

const NX = 17;
const NY = 13;
const SAIL_STRIDE = 6;
/** Canvas fraction at which the courses are fully down and the topsails start. */
const HALF_FILL = 0.55;
/** Above this true wind, a ship with nobody at the helm carries courses only. */
const AUTO_REEF_WIND = 15;
/** Fraction per second, the crew's pace (matches `Gameplay`'s SAIL_RATE). */
const SAIL_RATE = 0.45;
/** Simulated seconds of cloth after a reset, before the first frame. */
const WARMUP_STEPS = 240;
const MAX_STEPS_PER_FRAME = 8;
/** Wind the canvas is sized for; above it the crew would reef. */
const PRESSURE_WIND_CAP = 14;
/** How sharp a square-rigger's yards brace before they foul the shrouds. */
const BRACE_LIMIT = THREE.MathUtils.degToRad(40);

interface SailRuntime {
  spec: SailSpec;
  /** Arc length head to foot over the straight distance: the cut-in slack. */
  slack: number;
  height: number;
  /** Horizontal unit normal of the yard, pointing forward. */
  forward: THREE.Vector3;
  /** The model's brace of this yard, radians about +Y, positive toward port. */
  braceBaked: number;
  drop: number;
  trim: number;
  bundle: THREE.Mesh | null;
}

export class ShipSails {
  readonly object = new THREE.Group();

  private readonly sails: SailRuntime[] = [];
  private readonly solver: ClothSolver | null = null;
  private readonly pennant: Pennant | null = null;
  private readonly sailValues: THREE.Vector4[] = [];
  private readonly uCloth: Node = uniform(0);
  private readonly uTime: Node = uniform(0);
  private readonly material: THREE.Material | null = null;
  private readonly geometry: THREE.BufferGeometry | null = null;
  private readonly bundleGeometries: THREE.BufferGeometry[] = [];
  private readonly hull: THREE.Object3D;
  private readonly unitsPerMetre: number;

  private renderer: THREE.WebGPURenderer | null = null;
  private compiled = false;
  private clothOn = false;
  private time = 0;
  private stepAccumulator = 0;
  private fill = 0;
  private pendingReset = true;
  private inputsFresh = false;
  private warmPending = true;
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private readonly onMotion = () => (this.reducedMotion = this.motionQuery?.matches ?? false);
  private readonly stepLists = new Map<number, Node[]>();

  private readonly input: SailStateInput = {
    fill: null,
    trueWind: new THREE.Vector3(),
    velocity: new THREE.Vector3(),
    luff: 0,
    cloth: false,
  };
  private readonly windModel = new THREE.Vector3();
  private readonly invHull = new THREE.Quaternion();

  /**
   * @param model the GLB root, whose frame the sails are built in
   * @param metresPerUnit the ship's normalising scale
   * @param hull the object whose rotation is the ship's heading and heel
   */
  constructor(model: THREE.Object3D, metresPerUnit: number, hull: THREE.Object3D) {
    this.hull = hull;
    this.unitsPerMetre = 1 / metresPerUnit;
    this.object.name = 'ship-sails';
    this.motionQuery =
      typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    this.reducedMotion = this.motionQuery?.matches ?? false;
    this.motionQuery?.addEventListener('change', this.onMotion);

    const plan = readSailPlan(model);
    if (!plan || plan.sails.length === 0) return;
    model.add(this.object);

    const sourceMaterial = plan.sourceMesh.material as THREE.Material;
    for (const spec of plan.sails) {
      const headMid = spec.headL.clone().add(spec.headR).multiplyScalar(0.5);
      const clewMid = spec.clewL.clone().add(spec.clewR).multiplyScalar(0.5);
      const straight = headMid.distanceTo(clewMid);
      const slack = spec.kind === 'course' ? 1.1 : 1.07;
      const along = spec.headR.clone().sub(spec.headL).setY(0).normalize();
      const forward = new THREE.Vector3(0, 1, 0).cross(along).normalize();
      const braceBaked = Math.atan2(-forward.z, forward.x);
      this.sails.push({ spec, slack, height: straight * slack, forward, braceBaked, drop: 0, trim: 0, bundle: null });
    }

    // The furled topsail bundles stay the model's own meshes; the courses and
    // the set topsails are the grids.
    const src = plan.sourceMesh;
    src.visible = false;
    for (const sail of this.sails) {
      if (!sail.spec.bundle || !src.parent) continue;
      const g = new THREE.BufferGeometry();
      for (const [name, attr] of Object.entries(src.geometry.attributes)) g.setAttribute(name, attr);
      g.setIndex(new THREE.BufferAttribute(sail.spec.bundle, 1));
      g.computeBoundingSphere();
      this.bundleGeometries.push(g);
      const m = new THREE.Mesh(g, sourceMaterial);
      m.name = `${src.name}-${sail.spec.name}-furled`;
      m.position.copy(src.position);
      m.quaternion.copy(src.quaternion);
      m.scale.copy(src.scale);
      m.castShadow = true;
      m.receiveShadow = true;
      src.parent.add(m);
      sail.bundle = m;
    }

    // --- grids
    const patches: ClothPatch[] = [];
    const vertsPerSail = NX * NY;
    const total = vertsPerSail * this.sails.length;
    const positions = new Float32Array(total * 3);
    const normals = new Float32Array(total * 3);
    const uvs = new Float32Array(total * 2);
    const ids = new Uint32Array(total);
    const grid = new Float32Array(total * 3);
    const index: number[] = [];
    const tmp = new THREE.Vector3();
    const tmpB = new THREE.Vector3();
    this.sails.forEach((sail, si) => {
      const { spec } = sail;
      const rest = new Float32Array(vertsPerSail * 3);
      const start = new Float32Array(vertsPerSail * 3);
      const wHead = spec.headL.distanceTo(spec.headR);
      const wFoot = spec.clewL.distanceTo(spec.clewR) * 1.06;
      for (let r = 0; r < NY; r++) {
        const v = r / (NY - 1);
        for (let c = 0; c < NX; c++) {
          const u = c / (NX - 1);
          const k = r * NX + c;
          rest[k * 3] = (u - 0.5) * (wHead + (wFoot - wHead) * v);
          rest[k * 3 + 1] = -v * sail.height;
          rest[k * 3 + 2] = 0;
          bilinear(spec, u, v, tmp);
          // Starts a little filled, forward, so the first pressure has a side.
          tmp.addScaledVector(sail.forward, 0.18 * sail.height * Math.sin(Math.PI * u) * Math.sin(0.5 * Math.PI * v));
          start.set([tmp.x, tmp.y, tmp.z], k * 3);
          const g = si * vertsPerSail + k;
          positions.set([tmp.x, tmp.y, tmp.z], g * 3);
          normals.set([sail.forward.x, sail.forward.y, sail.forward.z], g * 3);
          const uvTop = tmpB.set(spec.uvHeadL.x, spec.uvHeadL.y, 0).lerp(tmp.set(spec.uvHeadR.x, spec.uvHeadR.y, 0), u).clone();
          const uvBot = tmpB.set(spec.uvClewL.x, spec.uvClewL.y, 0).lerp(tmp.set(spec.uvClewR.x, spec.uvClewR.y, 0), u);
          uvTop.lerp(uvBot, v);
          uvs.set([uvTop.x, uvTop.y], g * 2);
          ids[g] = g;
          grid.set([u, v, si], g * 3);
          if (r > 0 && c > 0) {
            const a = g - NX - 1;
            const b = g - NX;
            const d = g - 1;
            // Wound so the front face agrees with the normal the shader
            // computes (across x down, i.e. forward); DoubleSide flips by it.
            index.push(a, b, d, b, g, d);
          }
        }
      }
      patches.push({
        nx: NX,
        ny: NY,
        rest,
        start,
        group: si,
        kind: 0,
        pinned: (c, r) => r === 0 || (r === NY - 1 && (c === 0 || c === NX - 1)),
      });
      for (let k = 0; k < SAIL_STRIDE; k++) this.sailValues.push(new THREE.Vector4());
    });

    const pennant = new Pennant(plan.masthead, this.unitsPerMetre, this.sails.length);
    patches.push(pennant.patch);

    const solver = new ClothSolver(patches, this.sails.length + 1, this.unitsPerMetre);
    this.solver = solver;
    pennant.build(solver, solver.offsets[patches.length - 1], this.uCloth);
    this.pennant = pennant;
    this.object.add(pennant.mesh);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setAttribute('clothId', new THREE.BufferAttribute(ids, 1));
    geometry.setAttribute('grid', new THREE.BufferAttribute(grid, 3));
    geometry.setIndex(index);
    geometry.computeBoundingSphere();
    // Room for the belly and the furl either side of the start pose.
    if (geometry.boundingSphere) geometry.boundingSphere.radius += 3;
    this.geometry = geometry;

    const material = sourceMaterial.clone();
    material.side = THREE.DoubleSide;
    material.shadowSide = THREE.DoubleSide;
    (material as unknown as { positionNode: Node }).positionNode = this.buildPositionNode(solver);
    this.material = material;

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'ship-sails-cloth';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.object.add(mesh);

    if (import.meta.env.DEV && typeof window !== 'undefined') {
      (window as unknown as { __sails?: ShipSails }).__sails = this;
    }
  }

  /**
   * Builds the compute pipelines and takes the renderer that will dispatch
   * them. Called from the boot prewarm, so the first frame compiles nothing.
   */
  async compile(renderer: THREE.WebGPURenderer): Promise<void> {
    if (!this.solver || this.compiled) return;
    await this.solver.compile(renderer);
    this.renderer = renderer;
    this.compiled = true;
    this.warmPending = true;
  }

  setState(state: SailStateInput): void {
    this.input.fill = state.fill;
    this.input.trueWind.copy(state.trueWind);
    this.input.velocity.copy(state.velocity);
    this.input.luff = state.luff;
    this.input.cloth = state.cloth;
    this.inputsFresh = true;
  }

  /** Back to the rest pose at `time`; the next update warms the cloth up from it. */
  reset(time: number): void {
    this.time = time;
    this.stepAccumulator = 0;
    this.pendingReset = true;
    this.inputsFresh = false;
  }

  /** A test override: fixes the fill and the apparent wind (model frame, m/s). */
  debugOverride: { fill?: number; wind?: [number, number, number] } | null = null;

  update(dt: number): void {
    const solver = this.solver;
    if (!solver) return;
    this.time += dt;
    const input = this.input;
    const snap = this.pendingReset && this.inputsFresh;

    // --- what is set
    const trueSpeed = Math.hypot(input.trueWind.x, input.trueWind.z);
    let fill = input.fill ?? (trueSpeed > AUTO_REEF_WIND ? HALF_FILL : 1);
    if (this.debugOverride?.fill !== undefined) fill = this.debugOverride.fill;
    this.fill = snap ? fill : approach(this.fill, fill, SAIL_RATE * dt);

    // --- apparent wind in the ship's frame
    apparentWind(input.trueWind, input.velocity, this.windModel);
    this.invHull.copy(this.hull.quaternion).invert();
    this.windModel.applyQuaternion(this.invHull);
    if (this.debugOverride?.wind) this.windModel.fromArray(this.debugOverride.wind);
    const speed = this.windModel.length();
    const capped = SCRATCH_WIND.copy(this.windModel);
    if (speed > PRESSURE_WIND_CAP) capped.multiplyScalar(PRESSURE_WIND_CAP / speed);
    const cappedSpeed = Math.min(speed, PRESSURE_WIND_CAP);
    const motion = this.reducedMotion ? 0.3 : 1;
    // A slow gust envelope; pure in time so a reset reproduces it.
    const t = this.time;
    const gust = 1 + 0.18 * Math.sin(t * 0.41) + 0.09 * Math.sin(t * 1.37 + 1.1);
    // Where the yards would be braced for this wind: square before it, sharp up
    // to BRACE_LIMIT as it comes forward, the lee arm forward. Angles here are
    // about +Y, positive toward port, from the bow.
    const fromAngle = Math.atan2(Math.abs(this.windModel.z), -this.windModel.x);
    const braceSide = this.windModel.z <= 0 ? 1 : -1;
    const brace = braceSide * Math.min(BRACE_LIMIT, (Math.PI - fromAngle) * 0.35);

    for (let i = 0; i < this.sails.length; i++) {
      const sail = this.sails[i];
      const dropTarget =
        sail.spec.kind === 'course'
          ? clamp01(this.fill / HALF_FILL)
          : clamp01((this.fill - HALF_FILL) / (1 - HALF_FILL));
      sail.drop = dropTarget;

      // The yards stay at the model's brace (see the header), so the canvas is
      // shown the wind as a yard braced to `brace` would meet it: rotated by
      // the difference into the model's yard frame.
      const turn = sail.braceBaked - brace;
      const c = Math.cos(turn);
      const sn = Math.sin(turn);
      const felt = SCRATCH_FELT.set(capped.x * c + capped.z * sn, capped.y, capped.z * c - capped.x * sn);
      // Angle of attack: 1 with the wind square on the canvas from aft, 0 along
      // it, negative when it is aback.
      const sinA = cappedSpeed > 1e-3 ? felt.dot(sail.forward) / cappedSpeed : 0;
      const windy = smooth(0.5, 3.5, speed);
      const draw = smooth(0.08, 0.45, sinA) * windy;
      const luffWind = (1 - smooth(0.1, 0.4, sinA)) * smooth(1.5, 4, speed);
      const luff = Math.max(input.luff, luffWind) * (dropTarget > 0 ? 1 : 0);
      // Close-hauled is trimmed flat; running is eased.
      const trimTarget = 1 - smooth(0.3, 0.9, sinA);
      sail.trim = snap ? trimTarget : approach(sail.trim, trimTarget, 0.6 * dt);
      const slackScale = 1 + (1.01 / sail.slack - 1) * sail.trim;
      const aback = sinA < 0 ? 0.15 : 1;

      solver.setGroup(i, felt, gust * aback, luff * motion, slackScale, 0.9);

      // The analytic belly, for the tiers without the cloth.
      const full = sail.height * (0.36 - 0.24 * sail.trim);
      const depth = full * draw * Math.min(1, cappedSpeed / 6) * gust - sail.height * 0.06 * (sinA < 0 ? windy : 0);
      const luffAmp = luff * motion * sail.height * 0.07 * Math.min(1, speed / 7);
      const s = sail.spec;
      const o = i * SAIL_STRIDE;
      this.sailValues[o].set(s.headL.x, s.headL.y, s.headL.z, sail.drop);
      this.sailValues[o + 1].set(s.headR.x, s.headR.y, s.headR.z, sail.drop > 0.005 ? 1 : 0);
      this.sailValues[o + 2].set(s.clewL.x, s.clewL.y, s.clewL.z, Math.max(depth, -sail.height * 0.08));
      this.sailValues[o + 3].set(s.clewR.x, s.clewR.y, s.clewR.z, luffAmp);
      this.sailValues[o + 4].set(sail.forward.x, sail.forward.y, sail.forward.z, 0.3);
      this.sailValues[o + 5].set(i * 1.7, 0, 0, 0);
      if (sail.bundle) sail.bundle.visible = sail.drop <= 0.005;
    }

    this.pennant?.update(this.windModel, motion, this.time);
    if (this.pennant) solver.setGroup(this.sails.length, this.windModel, 1, 0.7 * motion, 1, 1);

    this.uTime.value = this.time;
    solver.uTime.value = this.time;

    // --- cloth
    const wantCloth = input.cloth && this.compiled && this.renderer !== null;
    if (wantCloth !== this.clothOn) {
      this.clothOn = wantCloth;
      if (wantCloth) this.warmPending = true;
    }
    if (snap) {
      this.pendingReset = false;
      this.warmPending = true;
    }
    if (this.clothOn && this.renderer) {
      if (this.warmPending && (this.inputsFresh || !this.pendingReset)) {
        solver.reset();
        this.dispatch(WARMUP_STEPS);
        this.warmPending = false;
        this.stepAccumulator = 0;
      } else {
        this.stepAccumulator += dt;
        let steps = Math.floor(this.stepAccumulator / CLOTH_STEP);
        this.stepAccumulator -= steps * CLOTH_STEP;
        if (steps > MAX_STEPS_PER_FRAME) steps = MAX_STEPS_PER_FRAME;
        this.dispatch(steps);
      }
    }
    this.uCloth.value = this.clothOn ? 1 : 0;
  }

  dispose(): void {
    this.motionQuery?.removeEventListener('change', this.onMotion);
    this.object.removeFromParent();
    this.geometry?.dispose();
    this.material?.dispose();
    for (const g of this.bundleGeometries) g.dispose();
    this.pennant?.dispose();
    this.solver?.dispose();
  }

  // ------------------------------------------------------------------ internals

  private dispatch(steps: number): void {
    if (!this.solver || !this.renderer || steps <= 0) return;
    let list = this.stepLists.get(steps);
    if (!list) {
      list = [];
      for (let i = 0; i < steps; i++) list.push(this.solver.computeSprings, this.solver.computeVertices);
      this.stepLists.set(steps, list);
    }
    void this.renderer.compute(list);
  }

  private buildPositionNode(solver: ClothSolver): Node {
    const table: Node = uniformArray(this.sailValues, 'vec4');
    const uCloth = this.uCloth;
    const uTime = this.uTime;
    const du = 1 / (NX - 1);
    const dv = 1 / (NY - 1);

    const row = (s: Node, k: number) => table.element(s.mul(SAIL_STRIDE).add(k));

    /** The sail's shape at (u, v) before furling: analytic or simulated. */
    const surface = (id: Node, u: Node, v: Node, s: Node): Node => {
      const r0 = row(s, 0);
      const r1 = row(s, 1);
      const r2 = row(s, 2);
      const r3 = row(s, 3);
      const r4 = row(s, 4);
      const phase = row(s, 5).x;
      const top = mix(r0.xyz, r1.xyz, u);
      const bottom = mix(r2.xyz, r3.xyz, u);
      const base = mix(top, bottom, v);
      // Fullest low in the sail, the head held straight by the yard.
      const shape = sin(u.mul(PI)).mul(sin(v.mul(Math.PI * 0.5))).mul(float(1).sub(v.mul(0.3)));
      const t = uTime;
      const ripple = sin(u.mul(9).add(v.mul(6)).sub(t.mul(14)).add(phase))
        .mul(0.6)
        .add(sin(u.mul(17).sub(v.mul(4)).sub(t.mul(23)).add(phase.mul(1.7))).mul(0.4));
      const leech = float(0.35).add(sin(u.mul(PI)).oneMinus().mul(0.65));
      const offset = r2.w.mul(shape).add(r3.w.mul(ripple).mul(v).mul(leech));
      const analytic = base.add(r4.xyz.mul(offset));
      const simulated = solver.positions.element(id).xyz;
      const shaped = select(uCloth.greaterThan(0.5), simulated, analytic);

      // Let fall by `drop`: hauled up toward the yard, and below a third
      // rolled into a bundle along it.
      const drop = r0.w;
      const hoisted = top.add(shaped.sub(top).mul(drop));
      const angle = v.mul(3 * Math.PI);
      const radius = r4.w.mul(float(1).add(sin(u.mul(31).add(1.7)).mul(0.3)));
      const tube = top
        .add(vec3(0, -1, 0).mul(radius.mul(cos(angle).oneMinus()).add(0.05)))
        .add(r4.xyz.mul(radius.mul(sin(angle))));
      const placed = mix(tube, hoisted, smoothstep(0, 0.3, drop));
      return mix(top, placed, r1.w);
    };

    return Fn(() => {
      const id = attribute('clothId', 'uint');
      const grid = attribute('grid', 'vec3');
      const u = grid.x;
      const v = grid.y;
      const s = int(grid.z.add(0.5));
      const hasL = u.greaterThan(du * 0.5);
      const hasR = u.lessThan(1 - du * 0.5);
      const hasU = v.greaterThan(dv * 0.5);
      const hasD = v.lessThan(1 - dv * 0.5);
      const p = surface(id, u, v, s).toVar();
      const pl = surface(select(hasL, id.sub(1), id), select(hasL, u.sub(du), u), v, s);
      const pr = surface(select(hasR, id.add(1), id), select(hasR, u.add(du), u), v, s);
      const pu = surface(select(hasU, id.sub(NX), id), u, select(hasU, v.sub(dv), v), s);
      const pd = surface(select(hasD, id.add(NX), id), u, select(hasD, v.add(dv), v), s);
      const n = cross(pr.sub(pl), pd.sub(pu)).toVar();
      normalLocal.assign(n.div(n.length().max(1e-6)));
      return p;
    })();
  }
}

const SCRATCH_WIND = /*@__PURE__*/ new THREE.Vector3();
const SCRATCH_FELT = /*@__PURE__*/ new THREE.Vector3();

function bilinear(spec: SailSpec, u: number, v: number, out: THREE.Vector3): THREE.Vector3 {
  const top = SCRATCH_A.copy(spec.headL).lerp(spec.headR, u);
  const bottom = SCRATCH_B.copy(spec.clewL).lerp(spec.clewR, u);
  return out.copy(top).lerp(bottom, v);
}
const SCRATCH_A = /*@__PURE__*/ new THREE.Vector3();
const SCRATCH_B = /*@__PURE__*/ new THREE.Vector3();

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function smooth(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

function approach(current: number, target: number, step: number): number {
  if (current < target) return Math.min(target, current + step);
  return Math.max(target, current - step);
}
