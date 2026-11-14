import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraViewMatrix,
  float,
  mix,
  mx_noise_float,
  positionGeometry,
  sin,
  uniform,
  vec3,
  vec4,
  vertexColor,
} from 'three/tsl';
import { getLeafAtlas, tileUv, type TileName } from '../flora/leafAtlas';
import { mulberry32 } from '../../core/random';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;
const vec3n = vec3 as unknown as (x: unknown, y?: unknown, z?: unknown) => Node;

/**
 * Small geometry kit for the island's set dressing: a builder that appends
 * transformed parts into one geometry per material, and the materials those
 * parts are drawn with. Everything here ends up merged — a whole camp is three
 * draws — because the point of it is detail that costs no draw calls.
 */

export class PartBuilder {
  private readonly pos: number[] = [];
  private readonly nrm: number[] = [];
  private readonly uv: number[] = [];
  private readonly col: number[] = [];
  private readonly idx: number[] = [];

  /** Appends `g` (indexed or not) transformed by `m`, tinted by `colour`. */
  add(g: THREE.BufferGeometry, m: THREE.Matrix4, colour: [number, number, number] = [1, 1, 1]): void {
    const src = g.index ? g : g;
    const p = src.getAttribute('position') as THREE.BufferAttribute;
    const n = src.getAttribute('normal') as THREE.BufferAttribute | undefined;
    const t = src.getAttribute('uv') as THREE.BufferAttribute | undefined;
    const c = src.getAttribute('color') as THREE.BufferAttribute | undefined;
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const base = this.pos.length / 3;
    const v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).applyMatrix4(m);
      this.pos.push(v.x, v.y, v.z);
      if (n) v.fromBufferAttribute(n, i).applyMatrix3(nm).normalize();
      else v.set(0, 1, 0);
      this.nrm.push(v.x, v.y, v.z);
      this.uv.push(t ? t.getX(i) : 0, t ? t.getY(i) : 0);
      const k = c ? c.getX(i) : 1;
      const kg = c ? c.getY(i) : 1;
      const kb = c ? c.getZ(i) : 1;
      this.col.push(colour[0] * k, colour[1] * kg, colour[2] * kb);
    }
    if (src.index) {
      for (let i = 0; i < src.index.count; i++) this.idx.push(base + src.index.getX(i));
    } else {
      for (let i = 0; i < p.count; i++) this.idx.push(base + i);
    }
  }

  /** Appends a quad from four corners (counter-clockwise seen from the front). */
  quad(
    corners: [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3],
    uvs: [number, number][],
    colour: [number, number, number],
    normal?: THREE.Vector3,
  ): void {
    const base = this.pos.length / 3;
    const n =
      normal ??
      new THREE.Vector3()
        .subVectors(corners[1], corners[0])
        .cross(new THREE.Vector3().subVectors(corners[3], corners[0]))
        .normalize();
    for (let i = 0; i < 4; i++) {
      this.pos.push(corners[i].x, corners[i].y, corners[i].z);
      this.nrm.push(n.x, n.y, n.z);
      this.uv.push(uvs[i][0], uvs[i][1]);
      this.col.push(colour[0], colour[1], colour[2]);
    }
    this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  get empty(): boolean {
    return this.idx.length === 0;
  }

  build(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.pos.length / 3 > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    return g;
  }
}

// ------------------------------------------------------------ shapes

/** A tapered, gently bent log along +x, `length` long, with a few branch stubs. */
export function buildLog(seed: number, length: number, radius: number, stubs: number): THREE.BufferGeometry {
  const rng = mulberry32(seed);
  const b = new PartBuilder();
  const pts: THREE.Vector3[] = [];
  const bend = (rng() - 0.5) * 0.3;
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    pts.push(new THREE.Vector3(t * length, Math.sin(t * Math.PI) * bend * length * 0.1, Math.sin(t * Math.PI * 1.3) * bend * length * 0.15));
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  const tube = new THREE.TubeGeometry(curve, 10, radius, 7, false);
  // Taper toward the tip by scaling ring radii about the curve.
  const p = tube.getAttribute('position') as THREE.BufferAttribute;
  const centre = new THREE.Vector3();
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const t = Math.min(1, Math.max(0, v.x / length));
    curve.getPointAt(t, centre);
    const k = 1.15 - 0.55 * t;
    v.sub(centre).multiplyScalar(k).add(centre);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  tube.computeVertexNormals();
  b.add(tube, new THREE.Matrix4());
  tube.dispose();
  // Root-plate end cap and branch stubs.
  const cap = new THREE.SphereGeometry(radius * 1.25, 7, 5);
  b.add(cap, new THREE.Matrix4().makeScale(0.5, 1, 1));
  cap.dispose();
  for (let i = 0; i < stubs; i++) {
    const t = 0.3 + rng() * 0.6;
    const at = curve.getPointAt(t);
    const len = radius * (2.5 + rng() * 4);
    const stub = new THREE.CylinderGeometry(radius * 0.18, radius * 0.38, len, 5, 1, true);
    stub.translate(0, len / 2, 0);
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler((rng() - 0.5) * 1.6, 0, -0.7 - rng() * 0.6));
    const m = new THREE.Matrix4().compose(at, q, new THREE.Vector3(1, 1, 1));
    b.add(stub, m);
    stub.dispose();
  }
  return b.build();
}

/** An open-topped wooden crate, plank seams in the vertex colour. */
export function buildCrate(size: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(size, size * 0.8, size * 0.9, 1, 4, 1);
  const p = g.getAttribute('position') as THREE.BufferAttribute;
  const c = new Float32Array(p.count * 3);
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i) / (size * 0.8) + 0.5;
    const seam = Math.abs(((y * 4) % 1) - 0.5) < 0.08 ? 0.6 : 1;
    c.set([seam, seam, seam], i * 3);
  }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  g.translate(0, size * 0.4, 0);
  return g;
}

// ------------------------------------------------------------ materials

/**
 * Weathered wood: vertex colour for the species and bleaching, a model-space
 * streak noise for grain. Model space so instanced logs keep their grain when
 * they move — the instance matrix is applied after `positionGeometry`.
 */
export function createWoodMaterial(): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.88, metalness: 0 });
  m.name = 'island-wood';
  m.envMapIntensity = 0.7;
  m.colorNode = Fn(() => {
    const p = positionGeometry;
    const grain = mx_noise_float(vec3n(p.x.mul(1.2), p.y.mul(9), p.z.mul(9))).mul(0.5).add(0.5);
    const knot = mx_noise_float(p.mul(2.7)).smoothstep(0.55, 0.8);
    const base: Node = vec3n(0.42, 0.37, 0.31);
    return base.mul(grain.mul(0.35).add(0.78)).mul(float(1).sub(knot.mul(0.35))).mul(vertexColor());
  })();
  return m;
}

/** Sail canvas, double sided: an off-white with a drawn-on weave and stains. */
export function createClothMaterial(): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0, side: THREE.DoubleSide });
  m.name = 'island-canvas';
  m.envMapIntensity = 0.7;
  m.colorNode = Fn(() => {
    const p = positionGeometry;
    const stain = mx_noise_float(p.mul(0.9)).smoothstep(0.2, 0.7);
    const weave = sin(p.x.mul(90)).mul(sin(p.z.mul(90))).mul(0.04).add(1);
    const base: Node = vec3n(0.68, 0.58, 0.42);
    return mix(base, vec3n(0.42, 0.34, 0.24), stain.mul(0.5)).mul(weave).mul(vertexColor());
  })();
  return m;
}

/** Flat decals lying on the sand: wrack and shell litter from the leaf atlas. */
export function createDecalMaterial(): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({
    map: getLeafAtlas(),
    vertexColors: true,
    // Blended, not cut out. Alpha-to-coverage needs MSAA and the scene pass has
    // none since TRAA replaced it, so a 0.5 cutout drew the wrack as hard,
    // pixel-stepped smears at every viewing distance. Lying flat on the sand,
    // these never overlap anything they would need sorting against.
    transparent: true,
    depthWrite: false,
    alphaTest: 0.02,
    roughness: 0.9,
    metalness: 0,
  });
  m.name = 'island-beach-decal';
  m.polygonOffset = true;
  m.polygonOffsetFactor = -2;
  m.polygonOffsetUnits = -2;
  m.envMapIntensity = 0.6;
  return m;
}

/** uv corners for a tile, in the PartBuilder quad order (bottom-left, bottom-right, top-right, top-left). */
export function tileQuadUv(tile: TileName, mirror = false): [number, number][] {
  const t = tileUv(tile);
  const [u0, u1] = mirror ? [t.u1, t.u0] : [t.u0, t.u1];
  return [
    [u0, t.vBot],
    [u1, t.vBot],
    [u1, t.vTop],
    [u0, t.vTop],
  ];
}

// ------------------------------------------------------------ fire and smoke

/** Shared clock for everything animated on the island's set dressing. */
export class DetailClock {
  readonly uTime = uniform(0);
  readonly uWind = uniform(new THREE.Vector2(1, 0));
  readonly uWindStrength = uniform(0.5);
  /** Light on the smoke from sky and key; ~1 at a clear noon. See `IslandDetail.setLighting`. */
  readonly uSmokeLight: any = uniform(new THREE.Color(1, 1, 1));
  /** 0..1, the fire's glow on the plume's underside after dark. */
  readonly uFireGlow = uniform(0);
  private time = 0;

  update(dt: number): void {
    this.time = (this.time + dt) % 3600;
    this.uTime.value = this.time;
  }

  reset(time: number): void {
    this.time = ((time % 3600) + 3600) % 3600;
    this.uTime.value = this.time;
  }

  setWind(direction: THREE.Vector2, strength: number): void {
    const w = this.uWind.value as THREE.Vector2;
    w.copy(direction);
    if (w.lengthSq() < 1e-6) w.set(1, 0);
    w.normalize();
    this.uWindStrength.value = Math.max(0, Math.min(1, strength));
  }
}

/** Glowing coals and flame tongues: emissive, flickering on the shared clock. */
export function createEmberMaterial(clock: DetailClock): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 1, metalness: 0, color: 0x1a0d06 });
  m.name = 'island-embers';
  m.emissiveNode = Fn(() => {
    const p = positionGeometry;
    const t = clock.uTime;
    const flicker = mx_noise_float(vec3n(p.x.mul(3), p.y.mul(2).sub(t.mul(3.1)), p.z.mul(3).add(t.mul(0.7))))
      .mul(0.5)
      .add(0.75);
    const hot: Node = vec3n(4.2, 1.5, 0.35);
    const warm: Node = vec3n(1.6, 0.35, 0.05);
    return mix(warm, hot, vertexColor().x).mul(flicker);
  })();
  m.userData.noGroundShading = true;
  return m;
}

/**
 * A column of smoke as camera-facing sprites. Each sprite rises on its own
 * phase of the shared clock, drifts downwind, grows and fades: a plume that
 * never accumulates state, so a clock reset reproduces a frame exactly.
 */
export function buildSmoke(clock: DetailClock, origin: THREE.Vector3, count: number, height: number): THREE.Mesh {
  const pos: number[] = [];
  const corner: number[] = [];
  const seed: number[] = [];
  const idx: number[] = [];
  const rng = mulberry32(0x5e0e);
  for (let i = 0; i < count; i++) {
    const s = [i / count, rng(), rng(), rng()];
    const base = pos.length / 3;
    for (const [cx, cy] of [
      [-0.5, -0.5],
      [0.5, -0.5],
      [0.5, 0.5],
      [-0.5, 0.5],
    ]) {
      pos.push(origin.x, origin.y, origin.z);
      corner.push(cx, cy);
      seed.push(...s);
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('corner', new THREE.Float32BufferAttribute(corner, 2));
  g.setAttribute('smokeSeed', new THREE.Float32BufferAttribute(seed, 4));
  g.setIndex(idx);
  g.boundingSphere = new THREE.Sphere(origin.clone().add(new THREE.Vector3(0, height / 2, 0)), height * 1.5);

  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  m.name = 'island-smoke';
  m.userData.noGroundShading = true;
  const c = attribute('corner', 'vec2') as Node;
  const sd = attribute('smokeSeed', 'vec4') as Node;
  const RISE_SECONDS = 9;
  const age = clock.uTime.div(RISE_SECONDS).add(sd.x).fract().toVar('smokeAge') as Node;
  m.positionNode = Fn(() => {
    const o = positionGeometry as Node;
    const w = clock.uWind as Node;
    const drift = age.mul(age).mul(height * 0.9).mul(clock.uWindStrength.add(0.25));
    const wobble = sin(clock.uTime.mul(0.9).add(sd.y.mul(6.28))).mul(age).mul(height * 0.12);
    // Each puff also spreads out on its own heading as it climbs, so the
    // plume widens into a drift rather than standing as a column.
    const spread = age.mul(height * 0.18).mul(sd.w.mul(0.6).add(0.4));
    const heading = sd.y.mul(6.28).add(sd.z.mul(2));
    const centre = vec3n(
      o.x.add(w.x.mul(drift)).add(wobble.mul(w.y)).add(heading.cos().mul(spread)),
      o.y.add(age.mul(height)),
      o.z.add(w.y.mul(drift)).sub(wobble.mul(w.x)).add(heading.sin().mul(spread)),
    );
    const size = age.mul(9).add(0.7).mul(sd.z.mul(0.5).add(0.75));
    // Camera right and up, from the view matrix's rows.
    const view = cameraViewMatrix as Node;
    const right = vec3n(view.element(0).x, view.element(1).x, view.element(2).x);
    const up = vec3n(view.element(0).y, view.element(1).y, view.element(2).y);
    return centre.add(right.mul(c.x.mul(size))).add(up.mul(c.y.mul(size)));
  })();
  m.colorNode = Fn(() => {
    const r = c.length().mul(2);
    const puff = float(1).sub(r.smoothstep(0.35, 1));
    const fade = age.smoothstep(0, 0.1).mul(float(1).sub(age.smoothstep(0.3, 0.9)));
    const grey: Node = mix(vec3n(0.36, 0.33, 0.3), vec3n(0.78, 0.78, 0.8), age.smoothstep(0, 0.5));
    // Lit, not self-coloured: the sky and key on a grey card, plus the fire on
    // the low part of the plume once it is dark. Unlit, the column stayed noon
    // grey all night and was the brightest thing on the beach.
    const firelight: Node = vec3n(1.0, 0.45, 0.16).mul(clock.uFireGlow).mul(float(1).sub(age.smoothstep(0.05, 0.35)).mul(1.0));
    const lit: Node = grey.mul(vec3n(clock.uSmokeLight)).add(grey.mul(firelight));
    return vec4(lit, puff.mul(fade).mul(0.13));
  })();
  const mesh = new THREE.Mesh(g, m);
  mesh.name = 'camp-smoke';
  mesh.frustumCulled = true;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.renderOrder = 2;
  return mesh;
}
