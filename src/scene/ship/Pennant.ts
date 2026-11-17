import * as THREE from 'three/webgpu';
import { Fn, attribute, cross, float, mix, normalLocal, select, sin, uniform, vec3 } from 'three/tsl';
import type { ClothPatch, ClothSolver } from './ClothSolver';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * A long masthead pennant, streaming down the apparent wind: the player's wind
 * indicator, as a real ship's was.
 *
 * Same solver as the sails (a two-row strip pinned at the hoist, with drag
 * along the wind and flutter), and the same analytic fallback for the tiers
 * without compute: a strip laid down the wind with a travelling wave, drooping
 * as the wind falls light.
 */

const NU = 18;
const NV = 2;
/** Metres. Period pennants ran to many fathoms; this one reads at chase range. */
const LENGTH = 6;
const HOIST = 0.45;
const FLY = 0.06;

export class Pennant {
  readonly patch: ClothPatch;
  mesh!: THREE.Mesh;

  private readonly length: number;
  private readonly hoistTop: THREE.Vector3;
  private readonly uDir: Node = uniform(new THREE.Vector3(-1, 0, 0));
  private readonly uAcross: Node = uniform(new THREE.Vector3(0, -1, 0));
  private readonly uSide: Node = uniform(new THREE.Vector3(0, 0, 1));
  private readonly uAmp: Node = uniform(0);
  private readonly uTime: Node = uniform(0);
  private material: THREE.Material | null = null;
  private geometry: THREE.BufferGeometry | null = null;

  constructor(masthead: THREE.Vector3, unitsPerMetre: number, group: number) {
    this.length = LENGTH * unitsPerMetre;
    const hoist = HOIST * unitsPerMetre;
    const fly = FLY * unitsPerMetre;
    this.hoistTop = masthead.clone().add(new THREE.Vector3(0, -0.12, 0));
    const rest = new Float32Array(NU * NV * 3);
    const start = new Float32Array(NU * NV * 3);
    for (let r = 0; r < NV; r++) {
      for (let c = 0; c < NU; c++) {
        const u = c / (NU - 1);
        const v = r / (NV - 1);
        const w = hoist + (fly - hoist) * u;
        const k = (r * NU + c) * 3;
        rest.set([u * this.length, -v * w, 0], k);
        start.set([this.hoistTop.x - u * this.length, this.hoistTop.y - v * w - u * 0.3, this.hoistTop.z], k);
      }
    }
    this.patch = { nx: NU, ny: NV, rest, start, group, kind: 1, pinned: (c) => c === 0 };
    this.hoist = hoist;
    this.fly = fly;
  }

  private readonly hoist: number;
  private readonly fly: number;

  build(solver: ClothSolver, offset: number, uCloth: Node): void {
    const n = NU * NV;
    const positions = new Float32Array(n * 3);
    const ids = new Uint32Array(n);
    const grid = new Float32Array(n * 3);
    const index: number[] = [];
    for (let r = 0; r < NV; r++) {
      for (let c = 0; c < NU; c++) {
        const k = r * NU + c;
        positions.set(this.patch.start.subarray(k * 3, k * 3 + 3), k * 3);
        ids[k] = offset + k;
        grid.set([c / (NU - 1), r / (NV - 1), 0], k * 3);
        // Wound to agree with the shader's normal (along x across).
        if (r > 0 && c > 0) index.push(k - NU - 1, k - NU, k - 1, k - NU, k, k - 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(n * 3).fill(0.5), 3));
    geometry.setAttribute('clothId', new THREE.BufferAttribute(ids, 1));
    geometry.setAttribute('grid', new THREE.BufferAttribute(grid, 3));
    geometry.setIndex(index);
    geometry.computeBoundingSphere();
    if (geometry.boundingSphere) geometry.boundingSphere.radius += this.length;
    this.geometry = geometry;

    const material = new THREE.MeshStandardNodeMaterial({ color: 0x8e1d17, roughness: 0.85, side: THREE.DoubleSide });
    material.name = 'ship-pennant';
    const du = 1 / (NU - 1);
    const L = this.length;
    const hoist = this.hoist;
    const fly = this.fly;
    const top = vec3(this.hoistTop.x, this.hoistTop.y, this.hoistTop.z);
    const surface = (id: Node, u: Node, v: Node): Node => {
      const width = mix(float(hoist), float(fly), u);
      const wave = sin(u.mul(10).sub(this.uTime.mul(12))).mul(this.uAmp).mul(u);
      const analytic = top
        .add(this.uDir.mul(u.mul(L)))
        .add(this.uAcross.mul(v.mul(width)))
        .add(this.uSide.mul(wave));
      return select(uCloth.greaterThan(0.5), solver.positions.element(id).xyz, analytic);
    };
    material.positionNode = Fn(() => {
      const id = attribute('clothId', 'uint');
      const grid = attribute('grid', 'vec3');
      const u = grid.x;
      const v = grid.y;
      const hasL = u.greaterThan(du * 0.5);
      const hasR = u.lessThan(1 - du * 0.5);
      const p = surface(id, u, v).toVar();
      const pl = surface(select(hasL, id.sub(1), id), select(hasL, u.sub(du), u), v);
      const pr = surface(select(hasR, id.add(1), id), select(hasR, u.add(du), u), v);
      const p0 = surface(select(v.greaterThan(0.5), id.sub(NU), id), u, float(0));
      const p1 = surface(select(v.lessThan(0.5), id.add(NU), id), u, float(1));
      const nrm = cross(pr.sub(pl), p1.sub(p0)).toVar();
      normalLocal.assign(nrm.div(nrm.length().max(1e-6)));
      return p;
    })();
    this.material = material;

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'ship-pennant';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.mesh = mesh;
  }

  /** Analytic pose for the tiers without compute. `wind` is model space, m/s. */
  update(wind: THREE.Vector3, motion: number, time: number): void {
    const s = wind.length();
    const dir = this.uDir.value as THREE.Vector3;
    // Streams level in a working breeze and droops as it falls light.
    if (s > 1e-3) dir.copy(wind).multiplyScalar(s).add(SCRATCH.set(0, -9, 0));
    else dir.set(0, -1, 0);
    dir.normalize();
    const side = this.uSide.value as THREE.Vector3;
    side.crossVectors(dir, UP);
    if (side.lengthSq() < 1e-6) side.set(0, 0, 1);
    side.normalize();
    (this.uAcross.value as THREE.Vector3).crossVectors(dir, side).normalize();
    const fresh = Math.min(1, Math.max(0, (s - 2) / 8));
    this.uAmp.value = this.length * 0.05 * (0.4 + 0.6 * fresh) * motion;
    this.uTime.value = time;
  }

  dispose(): void {
    this.mesh?.removeFromParent();
    this.geometry?.dispose();
    this.material?.dispose();
  }
}

const UP = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const SCRATCH = /*@__PURE__*/ new THREE.Vector3();
