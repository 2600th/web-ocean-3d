import * as THREE from 'three/webgpu';
import {
  Fn,
  float,
  materialColor,
  materialRoughness,
  mix,
  mx_noise_float,
  normalViewGeometry,
  normalWorldGeometry,
  positionView,
  positionWorld,
  vec3,
  vec4,
  vertexColor,
} from 'three/tsl';
import { mulberry32 } from '../../core/random';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * Procedural rock: geometry for the island's landmarks and a material that
 * paints strata, moss and the waterline onto anything made of it.
 *
 * The geometry is a displaced icosphere, sculpted per landmark by a signed
 * offset function, terraced into ledges, and given a cavity term in its vertex
 * colour. Nothing is textured: the material is world-space noise, so one rock
 * and the next share a geology and no two faces repeat.
 */

// ------------------------------------------------------------ CPU noise

/** Integer hash to [0, 1). CPU only, so a float hash's portability does not matter. */
function hash3(x: number, y: number, z: number, seed: number): number {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647 + seed * 144665) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const c = (dx: number, dy: number, dz: number) => hash3(ix + dx, iy + dy, iz + dz, seed);
  return l(
    l(l(c(0, 0, 0), c(1, 0, 0), ux), l(c(0, 1, 0), c(1, 1, 0), ux), uy),
    l(l(c(0, 0, 1), c(1, 0, 1), ux), l(c(0, 1, 1), c(1, 1, 1), ux), uy),
    uz,
  );
}

/** Signed fbm in about [-1, 1]. */
export function fbm3(x: number, y: number, z: number, seed: number, octaves = 4): number {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let f = 1;
  for (let o = 0; o < octaves; o++) {
    sum += (valueNoise3(x * f, y * f, z * f, seed + o * 31) * 2 - 1) * amp;
    norm += amp;
    amp *= 0.5;
    f *= 2.03;
  }
  return sum / norm;
}

// ------------------------------------------------------------ geometry

export interface RockShape {
  /** Subdivision of the base icosphere: 20 * (detail + 1)^2 triangles, so 9 is 2k and 44 is 40k. */
  detail: number;
  /** Half-extents of the base ellipsoid, metres. */
  size: [number, number, number];
  /** Noise displacement, metres, and the feature size it is drawn at. */
  rough: number;
  roughScale: number;
  /** Ledge height in metres, and how hard the surface snaps to it (0..1). */
  terrace?: number;
  terraceStrength?: number;
  /** Fraction of the height below which the rock is cut flat (its buried base). */
  base?: number;
  seed: number;
  /** How much narrower the top is than the bottom (0 none, 0.6 a spire). */
  taper?: number;
  /**
   * Extra signed offset along the normal, given the undisplaced unit-sphere
   * direction and the current position. This is where a landmark is sculpted.
   */
  sculpt?: (dir: THREE.Vector3, p: THREE.Vector3) => number;
}

/**
 * Builds a rock as an indexed geometry with position, normal, uv and colour.
 * The colour carries cavity occlusion — darker in hollows, lighter on edges —
 * which the material multiplies in.
 */
export function buildRock(shape: RockShape): THREE.BufferGeometry {
  const ico = new THREE.IcosahedronGeometry(1, shape.detail);
  // Icosahedron geometry is non-indexed; weld so displacement keeps it closed.
  const g = mergeByPosition(ico);
  ico.dispose();
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const dir = new THREE.Vector3();
  const p = new THREE.Vector3();
  const [sx, sy, sz] = shape.size;
  const cavity = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    dir.fromBufferAttribute(pos, i).normalize();
    p.set(dir.x * sx, dir.y * sy, dir.z * sz);
    const n = fbm3(
      (p.x / shape.roughScale) + 11.3,
      (p.y / shape.roughScale) + 3.7,
      (p.z / shape.roughScale) + 5.1,
      shape.seed,
    );
    let d = n * shape.rough;
    // Carved hollows (orbits, nostrils) are shadowed as well as recessed, so
    // they read as holes from a distance where the geometry alone does not.
    let carved = 0;
    if (shape.sculpt) {
      const sculpted = shape.sculpt(dir, p);
      d += sculpted;
      carved = Math.min(1, Math.max(0, -sculpted / (0.12 * sy)));
    }
    cavity[i] = n - carved * 0.9;
    p.addScaledVector(dir, d);
    if (shape.taper) {
      const up = Math.pow(dir.y * 0.5 + 0.5, 1.5);
      const k = 1 - shape.taper * up;
      p.x *= k;
      p.z *= k;
    }
    if (shape.terrace) {
      const t = shape.terrace;
      const snapped = Math.round(p.y / t) * t;
      p.y += (snapped - p.y) * (shape.terraceStrength ?? 0.4);
    }
    if (shape.base !== undefined) {
      const floor = -sy * shape.base;
      if (p.y < floor) p.y = floor + (p.y - floor) * 0.15;
    }
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  g.computeVertexNormals();
  // Cavity from the noise sign plus a little height: hollows dark, crowns light.
  const colours = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i) / sy;
    const ao = THREE.MathUtils.clamp(0.72 + cavity[i] * 0.55 + y * 0.12, 0.12, 1.1);
    colours.set([ao, ao, ao], i * 3);
  }
  g.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  g.computeBoundingSphere();
  return g;
}

/** Welds a non-indexed geometry by exact position, dropping its other attributes. */
function mergeByPosition(src: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = src.getAttribute('position') as THREE.BufferAttribute;
  const map = new Map<string, number>();
  const verts: number[] = [];
  const index: number[] = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const key = `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`;
    let k = map.get(key);
    if (k === undefined) {
      k = verts.length / 3;
      verts.push(x, y, z);
      map.set(key, k);
    }
    index.push(k);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  const uv = new Float32Array((verts.length / 3) * 2);
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(index);
  return g;
}

/** Smooth bump: 1 at the centre of a sphere of radius r around c, 0 outside. */
export function blob(p: THREE.Vector3, c: THREE.Vector3, r: number): number {
  const d = p.distanceTo(c) / r;
  if (d >= 1) return 0;
  const t = 1 - d * d;
  return t * t;
}

/**
 * Skull rock. A cranium, a narrower jaw, two deep orbits, a nasal cavity and a
 * row of teeth, all as offsets on one closed rock so it weathers like the rest
 * of the coast. Faces +z.
 */
export function buildSkullRock(seed: number, height: number): THREE.BufferGeometry {
  const k = height / 2;
  const eyeL = new THREE.Vector3(-0.36 * k, 0.12 * k, 0.86 * k);
  const eyeR = new THREE.Vector3(0.36 * k, 0.12 * k, 0.86 * k);
  const nose = new THREE.Vector3(0, -0.2 * k, 0.98 * k);
  const brow = new THREE.Vector3(0, 0.34 * k, 0.9 * k);
  const cheekL = new THREE.Vector3(-0.62 * k, -0.18 * k, 0.6 * k);
  const cheekR = new THREE.Vector3(0.62 * k, -0.18 * k, 0.6 * k);
  const mouth = new THREE.Vector3(0, -0.56 * k, 0.86 * k);
  return buildRock({
    detail: 44,
    size: [0.92 * k, k, 1.0 * k],
    rough: 0.035 * k,
    roughScale: 0.3 * k,
    terrace: 0.09 * k,
    terraceStrength: 0.25,
    base: 0.82,
    seed,
    sculpt: (dir, p) => {
      let d = 0;
      // Jaw: the lower third pinches in toward the front, cranium stays round.
      const lower = THREE.MathUtils.smoothstep(-p.y / k, 0.15, 0.85);
      d -= lower * (0.18 + 0.2 * Math.max(0, dir.z)) * k * Math.abs(dir.x);
      d += blob(p, brow, 0.38 * k) * 0.09 * k;
      d += (blob(p, cheekL, 0.32 * k) + blob(p, cheekR, 0.32 * k)) * 0.08 * k;
      // Orbits: deep, and slightly larger than life so they read at a mile.
      d -= (blob(p, eyeL, 0.3 * k) + blob(p, eyeR, 0.3 * k)) * 0.36 * k;
      // Nasal cavity: a narrow inverted heart.
      const nx = (p.x - nose.x) / (0.13 * k);
      const ny = (p.y - nose.y) / (0.2 * k);
      const nz = Math.max(0, dir.z);
      const nasal = Math.max(0, 1 - nx * nx - ny * ny * (1 + 0.6 * Math.sign(ny))) * nz;
      d -= nasal * 0.22 * k;
      // Teeth: a row of bumps along the mouth line, a dark slot either side.
      const mouthBand = blob(p, mouth, 0.42 * k);
      if (mouthBand > 0 && dir.z > 0.3) {
        const tooth = Math.cos((p.x / (0.085 * k)) * Math.PI);
        const row = Math.exp(-(((p.y - mouth.y) / (0.07 * k)) ** 2));
        const slot = Math.exp(-(((p.y - mouth.y - 0.1 * k) / (0.03 * k)) ** 2));
        d += mouthBand * (row * (0.03 + 0.03 * tooth) - slot * 0.12) * k;
      }
      return d;
    },
  });
}

/**
 * A sea arch: a fat, rough tube bent over a span, standing on two feet. Built
 * as a swept ring rather than a displaced sphere because the hole is the point.
 */
export function buildArch(seed: number, span: number, height: number, thickness: number): THREE.BufferGeometry {
  const rng = mulberry32(seed);
  const segs = 40;
  const radial = 18;
  const pos: number[] = [];
  const idx: number[] = [];
  const centre = (t: number): THREE.Vector3 => {
    const a = Math.PI * t;
    // A slightly flattened, asymmetric arc: one leg steeper than the other.
    const lean = 0.12 * Math.sin(Math.PI * t);
    return new THREE.Vector3(-Math.cos(a) * span * 0.5 + lean * span, Math.sin(a) * height, 0);
  };
  const p = new THREE.Vector3();
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const c = centre(t);
    const ahead = centre(Math.min(1, t + 0.01)).sub(centre(Math.max(0, t - 0.01))).normalize();
    const side = new THREE.Vector3(0, 0, 1);
    const up = new THREE.Vector3().crossVectors(side, ahead).normalize();
    // Thicker at the feet, thinnest at the crown.
    const r = thickness * (0.62 + 0.55 * Math.pow(Math.abs(t - 0.5) * 2, 1.6));
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      p.copy(c)
        .addScaledVector(up, Math.cos(a) * r)
        .addScaledVector(side, Math.sin(a) * r * 1.35);
      const n = fbm3(p.x / 7 + 3.1, p.y / 7 + 1.7, p.z / 7 + 9.2, seed) * thickness * 0.35;
      const dir = p.clone().sub(c).normalize();
      p.addScaledVector(dir, n + (rng() - 0.5) * 0.15);
      // Ledges.
      p.y += (Math.round(p.y / 2.4) * 2.4 - p.y) * 0.3;
      pos.push(p.x, p.y, p.z);
    }
  }
  for (let i = 0; i < segs; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * (radial + 1) + j;
      const b = a + radial + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  const count = pos.length / 3;
  const colours = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const n = fbm3(pos[i * 3] / 4, pos[i * 3 + 1] / 4, pos[i * 3 + 2] / 4, seed + 5);
    const ao = THREE.MathUtils.clamp(0.8 + n * 0.5, 0.4, 1.1);
    colours.set([ao, ao, ao], i * 3);
  }
  g.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(count * 2), 2));
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------ material

/**
 * Surface-gradient bump (Mikkelsen), in view space. After the slop-newyork
 * `bumpNormal` (MIT, Copyright (c) 2026 Mr.doob), with the geometric normal in
 * place of the shading one so it can sit in `normalNode` without recursion.
 */
const bumpNormal = Fn(([height]: [Node]) => {
  const h = float(height).toVar();
  const n = normalViewGeometry;
  const dpdx = positionView.dFdx();
  const dpdy = positionView.dFdy();
  const r1 = dpdy.cross(n);
  const r2 = n.cross(dpdx);
  const det = dpdx.dot(r1);
  const grad = det.sign().mul(h.dFdx().mul(r1).add(h.dFdy().mul(r2)));
  return det.abs().mul(n).sub(grad).normalize();
});

export interface RockMaterialOptions {
  /** Moss on upward faces, 0..1. */
  moss?: number;
  /** Base tint multiplier. */
  tint?: [number, number, number];
}

/**
 * Volcanic and sedimentary rock in one: banded by world elevation and warped
 * by noise so the strata run continuously across neighbouring rocks, moss on
 * the upper faces, and the waterline written in — darker and glossy where the
 * swash reaches, barnacle-pale just above mean sea level, weed-green below it.
 */
export function createRockMaterial(options: RockMaterialOptions = {}): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 });
  m.name = 'island-rock';
  m.envMapIntensity = 0.75;
  const tint = options.tint ?? [1, 1, 1];
  const mossAmount = options.moss ?? 0.6;
  const vec3n = vec3 as unknown as (x: unknown, y?: unknown, z?: unknown) => Node;

  const field = Fn(() => {
    const wp = positionWorld;
    const warp = mx_noise_float(wp.mul(0.035)).toVar();
    const fine = mx_noise_float(wp.mul(0.45)).toVar();
    const grain = mx_noise_float(wp.mul(2.1)).toVar();
    return vec3n(warp, fine, grain);
  })();

  m.colorNode = Fn(() => {
    const wp = positionWorld.toVar();
    const f = vec3n(field).toVar();
    const light: Node = vec3n(0.36 * tint[0], 0.31 * tint[1], 0.25 * tint[2]);
    const dark: Node = vec3n(0.17 * tint[0], 0.14 * tint[1], 0.12 * tint[2]);
    const ochre: Node = vec3n(0.36 * tint[0], 0.21 * tint[1], 0.13 * tint[2]);
    const layer = wp.y.mul(0.55).add(f.x.mul(5)).sin().mul(0.5).add(0.5).toVar();
    const band = wp.y.mul(0.17).add(f.x.mul(2.2)).sin().toVar();
    const strata: Node = mix(dark, light, layer.smoothstep(0.15, 0.85).mul(0.6).add(f.y.mul(0.25)).add(0.2));
    const banded: Node = mix(strata, ochre, band.smoothstep(0.75, 0.95).mul(0.55));
    const grained: Node = banded.mul(f.z.mul(0.12).add(1));

    // Moss and lichen on faces that catch rain, above the spray line.
    const up = normalWorldGeometry.y.toVar();
    const mossMask = up
      .add(f.y.mul(0.25))
      .smoothstep(0.45, 0.75)
      .mul(wp.y.smoothstep(2.5, 6))
      .mul(mossAmount)
      .toVar();
    const mossy: Node = mix(grained, vec3n(0.06, 0.1, 0.028), mossMask);

    // The waterline.
    const y = wp.y;
    const wet = float(1).sub(y.smoothstep(0.6, 2.6)).toVar();
    const barnacle = y
      .smoothstep(-0.9, -0.2)
      .mul(float(1).sub(y.smoothstep(0.5, 1.3)))
      .mul(f.z.add(f.y.mul(0.5)).smoothstep(0.1, 0.45))
      .toVar();
    const weed = float(1).sub(y.smoothstep(-0.6, 0.2)).mul(f.y.smoothstep(-0.4, 0.2)).toVar();
    const tide: Node = mix(mossy, mossy.mul(0.5), wet);
    const barnacled: Node = mix(tide, vec3n(0.42, 0.4, 0.36) as Node, barnacle.mul(0.75) as Node);
    const weeded: Node = mix(barnacled, vec3n(0.07, 0.1, 0.035) as Node, weed.mul(0.7) as Node);
    return weeded.mul(vertexColor());
  })();

  m.roughnessNode = Fn(() => {
    const y = positionWorld.y;
    const wet = float(1).sub(y.smoothstep(0.6, 2.6));
    return mix(float(0.9), float(0.32), wet);
  })();

  m.normalNode = Fn(() => {
    const f = vec3n(field);
    return bumpNormal(f.y.mul(0.35).add(f.z.mul(0.08)));
  })();
  // `vertexColors` stays false: the colour node above multiplies the vertex
  // colour in itself, and leaving the flag on would apply it twice.
  m.vertexColors = false;
  return m;
}

/**
 * Gives an existing rock material the waterline the generated rock has: wet
 * and dark through the swash, a pale crust of barnacles just above the still
 * water, green weed below it. Wraps the material's own colour (map and all),
 * so call it before `SurfaceWetness.adopt`, which wraps whatever it finds.
 * `tint` multiplies the albedo, for scans whose colour fights the island's.
 */
export function applyTideline(material: THREE.Material, tint: [number, number, number] = [1, 1, 1]): void {
  const node = material as THREE.MeshStandardNodeMaterial;
  if ((node as unknown as { isNodeMaterial?: boolean }).isNodeMaterial !== true) return;
  if (material.userData.tideline === true) return;
  material.userData.tideline = true;
  const vec3n = vec3 as unknown as (x: unknown, y?: unknown, z?: unknown) => Node;
  const own: Node = (node as unknown as { colorNode: Node }).colorNode ?? materialColor;
  const ownRoughness: Node = (node as unknown as { roughnessNode: Node }).roughnessNode ?? materialRoughness;
  node.colorNode = Fn(() => {
    const base: Node = vec4(own.xyz.mul(vec3n(...tint)), own.w);
    const wp = positionWorld.toVar();
    const crust = mx_noise_float(wp.mul(0.9)).add(mx_noise_float(wp.mul(3.3)).mul(0.5)).toVar();
    const y = wp.y;
    const wet = float(1).sub(y.smoothstep(0.5, 2.4)).toVar();
    const barnacle = y
      .smoothstep(-0.8, -0.2)
      .mul(float(1).sub(y.smoothstep(0.4, 1.1)))
      .mul(crust.smoothstep(0.05, 0.4))
      .toVar();
    const weed = float(1).sub(y.smoothstep(-0.7, 0.1)).mul(crust.smoothstep(-0.3, 0.2)).toVar();
    const tide: Node = mix(base, base.mul(vec3n(0.48, 0.5, 0.5)), wet);
    const crusted: Node = mix(tide, vec3n(0.4, 0.38, 0.33) as Node, barnacle.mul(0.7) as Node);
    const weeded: Node = mix(crusted, vec3n(0.06, 0.09, 0.03) as Node, weed.mul(0.65) as Node);
    return vec4(weeded.xyz, base.w);
  })();
  node.roughnessNode = Fn(() => {
    const y = positionWorld.y;
    const wet = float(1).sub(y.smoothstep(0.5, 2.4));
    const r: Node = ownRoughness;
    // Wet rock glistens; barnacle crust does not, so it stops short of a mirror.
    return mix(r, r.mul(0.45).max(0.3), wet);
  })();
  node.needsUpdate = true;
}
