import * as THREE from 'three/webgpu';
import {
  EPSILON,
  Fn,
  If,
  Loop,
  Return,
  cross,
  float,
  instanceIndex,
  instancedArray,
  select,
  triNoise3D,
  uniform,
  uniformArray,
  vec4,
} from 'three/tsl';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * Verlet cloth in two compute passes, for the sails and the pennant.
 *
 * Adapted from three.js r187's `webgpu_compute_cloth` example (MIT, see
 * THIRD_PARTY_NOTICES.md): the same layout of a spring buffer, a per-vertex
 * spring list and two dispatches per step, one computing each spring's force
 * and one gathering them per vertex. Gathering rather than scattering keeps it
 * free of atomics, so the same inputs give the same bits every run, which the
 * visual baselines need.
 *
 * What changed from the example:
 *   - several grids (patches) share one set of buffers, each with a group index
 *     into a small uniform table carrying its wind, pressure, luff and slack;
 *   - pinned vertices follow a pin buffer the CPU writes, so clews can be trimmed;
 *   - the wind is a pressure on the local normal (from the four grid
 *     neighbours) for sails, a drag along the wind for the pennant, plus a
 *     flutter term that shakes the canvas when it is not drawing;
 *   - a per-group slack scales the rest lengths, which is how a sail hauled
 *     close to the wind is flatter than one eased off before the wind;
 *   - vec4 buffers throughout, so no vec3 padding is involved anywhere;
 *   - the step is in physical units (m/s, m/s^2) converted by the model scale.
 */

/** Simulation step, seconds. Four steps per frame at 60 Hz. */
export const CLOTH_STEP = 1 / 240;

export interface ClothPatch {
  /** Columns (u, across) and rows (v, down). Vertex = offset + row * nx + col. */
  nx: number;
  ny: number;
  /** Flat layout the rest lengths come from, nx*ny*3. */
  rest: Float32Array;
  /** Pose at reset, nx*ny*3. */
  start: Float32Array;
  pinned: (col: number, row: number) => boolean;
  group: number;
  /** 0 sail (pressure on the normal), 1 pennant (drag along the wind). */
  kind: 0 | 1;
}

/** Uniform slots per group. */
const GROUP_STRIDE = 2;

export class ClothSolver {
  readonly vertexCount: number;
  readonly offsets: number[] = [];
  /** xyz position, model space. Read by the sail and pennant materials. */
  readonly positions: Node;
  readonly computeSprings: Node;
  readonly computeVertices: Node;
  /** Clock for the gust and flutter noise. */
  readonly uTime: Node = uniform(0);

  private readonly velocities: Node;
  private readonly pins: Node;
  private readonly startArray: Float32Array;
  private readonly pinArray: Float32Array;
  private readonly groupValues: THREE.Vector4[];

  constructor(patches: readonly ClothPatch[], groupCount: number, unitsPerMetre: number) {
    let count = 0;
    for (const p of patches) {
      this.offsets.push(count);
      count += p.nx * p.ny;
    }
    this.vertexCount = count;

    // --- springs: structural, shear and bend, as in the example plus bend.
    const springs: [number, number, number][] = [];
    const perVertex: number[][] = Array.from({ length: count }, () => []);
    const fixed = new Uint8Array(count);
    const group = new Uint32Array(count);
    const kind = new Uint32Array(count);
    const nbr = new Uint32Array(count * 4);
    this.startArray = new Float32Array(count * 4);
    patches.forEach((p, pi) => {
      const o = this.offsets[pi];
      const id = (c: number, r: number) => o + r * p.nx + c;
      const restLen = (a: number, b: number) => {
        const ax = (a - o) * 3;
        const bx = (b - o) * 3;
        return Math.hypot(p.rest[ax] - p.rest[bx], p.rest[ax + 1] - p.rest[bx + 1], p.rest[ax + 2] - p.rest[bx + 2]);
      };
      const add = (a: number, b: number) => {
        const s = springs.length;
        springs.push([a, b, restLen(a, b)]);
        perVertex[a].push(s);
        perVertex[b].push(s);
      };
      for (let r = 0; r < p.ny; r++) {
        for (let c = 0; c < p.nx; c++) {
          const v = id(c, r);
          fixed[v] = p.pinned(c, r) ? 1 : 0;
          group[v] = p.group;
          kind[v] = p.kind;
          nbr[v * 4] = id(Math.max(0, c - 1), r);
          nbr[v * 4 + 1] = id(Math.min(p.nx - 1, c + 1), r);
          nbr[v * 4 + 2] = id(c, Math.max(0, r - 1));
          nbr[v * 4 + 3] = id(c, Math.min(p.ny - 1, r + 1));
          const s = (v - o) * 3;
          this.startArray.set([p.start[s], p.start[s + 1], p.start[s + 2], 1], v * 4);
          if (c > 0) add(v, id(c - 1, r));
          if (r > 0) add(v, id(c, r - 1));
          if (c > 0 && r > 0) add(v, id(c - 1, r - 1));
          if (c > 0 && r < p.ny - 1) add(v, id(c - 1, r + 1));
          if (c > 1) add(v, id(c - 2, r));
          if (r > 1) add(v, id(c, r - 2));
        }
      }
    });

    const springList: number[] = [];
    const params = new Uint32Array(count * 4);
    for (let v = 0; v < count; v++) {
      params[v * 4] = fixed[v];
      params[v * 4 + 3] = group[v] * 16 + kind[v];
      if (fixed[v]) continue;
      params[v * 4 + 1] = perVertex[v].length;
      params[v * 4 + 2] = springList.length;
      springList.push(...perVertex[v]);
    }
    const springIds = new Uint32Array(springs.length * 2);
    const springRest = new Float32Array(springs.length);
    springs.forEach(([a, b, l], s) => {
      springIds[s * 2] = a;
      springIds[s * 2 + 1] = b;
      springRest[s] = l;
    });

    this.positions = instancedArray(this.startArray.slice(), 'vec4');
    this.velocities = instancedArray(count, 'vec4');
    this.pinArray = this.startArray.slice();
    this.pins = instancedArray(this.pinArray, 'vec4');
    const paramBuf = instancedArray(params, 'uvec4');
    const nbrBuf = instancedArray(nbr, 'uvec4');
    const listBuf = instancedArray(new Uint32Array(springList.length > 0 ? springList : [0]), 'uint');
    const idBuf = instancedArray(springIds, 'uvec2');
    const restBuf = instancedArray(springRest, 'float');
    const forceBuf = instancedArray(springs.length, 'vec4');
    const normalBuf = instancedArray(count, 'vec4');

    this.groupValues = Array.from({ length: groupCount * GROUP_STRIDE }, () => new THREE.Vector4());
    const groups: Node = uniformArray(this.groupValues, 'vec4');

    // Per-step constants. Velocities are stored as displacement per step, as in
    // the example, so an acceleration a becomes a * h^2 in model units.
    const h2 = CLOTH_STEP * CLOTH_STEP * unitsPerMetre;
    const stiffness = float(0.32);
    const damping = float(0.982);
    const gravity = float(9.81 * h2);
    // Dynamic pressure over canvas mass: q = 0.5 rho V^2, flax about 0.6 kg/m^2.
    const kPress = float(((0.5 * 1.225) / 0.6) * h2);
    // A bunting pennant is lighter and feels drag rather than pressure.
    const kDrag = float(((0.5 * 1.225 * 0.25) / 0.2) * h2);

    const slackOf = (g: Node) => groups.element(g.mul(GROUP_STRIDE).add(1)).z;

    // The first pass also takes each vertex's normal from its neighbours. It
    // has to be here, where positions are only read: taken in the vertex pass
    // it raced the neighbours' own writes, and the pennant's flutter amplified
    // that into a different pose on every reset.
    const springCount = springs.length;
    this.computeSprings = Fn(() => {
      If(instanceIndex.lessThan(springCount), () => {
        const ids = idBuf.element(instanceIndex);
        const rest = restBuf.element(instanceIndex);
        const g = paramBuf.element(ids.x).w.div(16);
        const p0 = this.positions.element(ids.x).xyz;
        const p1 = this.positions.element(ids.y).xyz;
        const delta = p1.sub(p0).toVar();
        const dist = delta.length().max(EPSILON).toVar();
        const target = rest.mul(slackOf(g));
        const force = dist.sub(target).mul(stiffness).mul(delta).mul(0.5).div(dist);
        forceBuf.element(instanceIndex).assign(vec4(force, 0));
      });
      If(instanceIndex.lessThan(count), () => {
        const n4 = nbrBuf.element(instanceIndex);
        const across = this.positions.element(n4.y).xyz.sub(this.positions.element(n4.x).xyz);
        const down = this.positions.element(n4.w).xyz.sub(this.positions.element(n4.z).xyz);
        const nRaw = cross(across, down).toVar();
        normalBuf.element(instanceIndex).assign(vec4(nRaw.div(nRaw.length().max(1e-6)), 0));
      });
    })()
      .compute(Math.max(springCount, count))
      .setName('Sail springs and normals');

    this.computeVertices = Fn(() => {
      const prm = paramBuf.element(instanceIndex).toVar();
      If(prm.x.equal(1), () => {
        this.positions.element(instanceIndex).assign(this.pins.element(instanceIndex));
        this.velocities.element(instanceIndex).assign(vec4(0));
        Return();
      });
      const g = prm.w.div(16).toVar();
      const isPennant = prm.w.mod(16).equal(1);
      const position = this.positions.element(instanceIndex).xyz.toVar();
      const vel = this.velocities.element(instanceIndex).xyz.mul(damping).toVar();

      const ptrStart = prm.z.toVar();
      const ptrEnd = ptrStart.add(prm.y).toVar();
      Loop({ start: ptrStart, end: ptrEnd, type: 'uint', condition: '<' }, ({ i }: { i: Node }) => {
        const springId = listBuf.element(i).toVar();
        const f = forceBuf.element(springId).xyz;
        const sign = select(idBuf.element(springId).x.equal(instanceIndex), 1.0, -1.0);
        vel.addAssign(f.mul(sign));
      });

      vel.y.subAssign(gravity);

      // Local normal, from the grid neighbours in the first pass.
      const normal = normalBuf.element(instanceIndex).xyz.toVar();

      const g0 = groups.element(g.mul(GROUP_STRIDE)).toVar();
      const g1 = groups.element(g.mul(GROUP_STRIDE).add(1)).toVar();
      const wind = g0.xyz;
      const speed = wind.length().toVar();
      // Gusts: a slow field drifting through the canvas; flutter: a fast one.
      const gust = triNoise3D(position.mul(0.3), float(0.8), this.uTime).mul(0.8).add(0.65);
      const shake = triNoise3D(position.mul(1.3).add(7.1), float(6.0), this.uTime).sub(0.32).mul(2.0);
      const flutter = normal.mul(shake.mul(g1.y).mul(speed).mul(speed).mul(kPress).mul(g1.w));

      If(isPennant, () => {
        vel.addAssign(wind.mul(speed).mul(kDrag).mul(gust));
        vel.addAssign(flutter);
      }).Else(() => {
        const pressure = normal.dot(wind).mul(speed).mul(kPress).mul(g1.x).mul(gust);
        vel.addAssign(normal.mul(pressure));
        vel.addAssign(flutter);
      });

      this.velocities.element(instanceIndex).assign(vec4(vel, 0));
      this.positions.element(instanceIndex).assign(vec4(position.add(vel), 1));
    })()
      .compute(count)
      .setName('Sail vertices');
  }

  /**
   * Group parameters for the next steps.
   * @param wind apparent wind, m/s, model space (the direction it blows toward)
   * @param pressure scale on the pressure (gust, aback)
   * @param luff 0..1, how hard the canvas shakes
   * @param slack rest-length scale, 1 eased, below 1 hauled flat
   * @param flutterGain per-kind flutter strength
   */
  setGroup(index: number, wind: THREE.Vector3, pressure: number, luff: number, slack: number, flutterGain: number): void {
    this.groupValues[index * GROUP_STRIDE].set(wind.x, wind.y, wind.z, 0);
    this.groupValues[index * GROUP_STRIDE + 1].set(pressure, luff, slack, flutterGain);
  }

  setPin(vertex: number, x: number, y: number, z: number): void {
    this.pinArray[vertex * 4] = x;
    this.pinArray[vertex * 4 + 1] = y;
    this.pinArray[vertex * 4 + 2] = z;
  }

  /** Marks the pin buffer for upload before the next dispatch. */
  commitPins(): void {
    (this.pins.value as THREE.BufferAttribute).needsUpdate = true;
  }

  /** Back to the start pose, at rest. */
  reset(): void {
    const pos = this.positions.value as THREE.BufferAttribute;
    (pos.array as Float32Array).set(this.startArray);
    pos.needsUpdate = true;
    const vel = this.velocities.value as THREE.BufferAttribute;
    (vel.array as Float32Array).fill(0);
    vel.needsUpdate = true;
  }

  /** Overwrites the start pose of one vertex (used before `reset`). */
  setStart(vertex: number, x: number, y: number, z: number): void {
    this.startArray[vertex * 4] = x;
    this.startArray[vertex * 4 + 1] = y;
    this.startArray[vertex * 4 + 2] = z;
  }

  /** `steps` simulation steps, in one compute submission. */
  run(renderer: THREE.WebGPURenderer, steps: number): void {
    if (steps <= 0) return;
    const list: Node[] = [];
    for (let i = 0; i < steps; i++) list.push(this.computeSprings, this.computeVertices);
    void renderer.compute(list);
  }

  compile(renderer: THREE.WebGPURenderer): Promise<void> {
    return renderer.compileComputeAsync([this.computeSprings, this.computeVertices]);
  }

  dispose(): void {
    this.computeSprings.dispose?.();
    this.computeVertices.dispose?.();
  }
}
