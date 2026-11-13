import * as THREE from 'three/webgpu';

/**
 * A small mesh accumulator for the procedural reef and its animals.
 *
 * Every plant and animal under the water is built from three primitives — a
 * tube swept along a polyline, a surface of revolution, and a displaced blob —
 * written into one set of flat arrays and handed to the GPU as a single
 * `BufferGeometry`. That is the whole asset pipeline: no downloads, no texture
 * fetches, and a template is a few hundred triangles that an `InstancedMesh`
 * repeats.
 *
 * The tube sweep with parallel-transported frames and the per-vertex sway weight
 * are the pattern `mrdoob/slop-newyork` (MIT, (c) 2026 Mr.doob) uses for its
 * procedural trees — `growBranch` / `addTube` / `swayWeight` in `js/trees.js` —
 * adapted here to TypeScript and to the reef's extra per-vertex channels.
 *
 * Per-vertex channels, beyond position and normal:
 *
 *   `color`     albedo multiplier, with ambient occlusion baked in. The instance
 *               colour is multiplied on top by three, so a template is authored
 *               near white wherever its hue should come from the instance.
 *   `reefSway`  x: metres a vertex travels at full current (0 on anything
 *               rigid, growing toward a tip); y: 0..1 how much of that travel is
 *               a fast local wriggle (tentacles) rather than the slow bend of the
 *               whole plant with the swell.
 *   `reefSurf`  surface pattern selector for the shader: 0 plain, 1 meandering
 *               brain-coral grooves, 2 polyp pores, 3 scute plates (turtle).
 */
export type RGB = readonly [number, number, number];

export interface PartStyle {
  /** Albedo multiplier at a point along the part (t 0..1 along, a 0..1 around). */
  color: (t: number, a: number) => RGB;
  /** Sway amplitude, metres, at t along the part. */
  sway?: (t: number) => number;
  /** 0 slow bend, 1 fast wriggle. */
  wiggle?: number;
  /** Surface pattern selector, see the header. */
  surf?: number;
}

const WHITE: RGB = [1, 1, 1];

export class MeshBuilder {
  readonly pos: number[] = [];
  readonly col: number[] = [];
  readonly sway: number[] = [];
  readonly surf: number[] = [];
  readonly idx: number[] = [];
  /** Optional per-vertex scalar channel for animals (part id, span, ...). */
  readonly aux: number[] = [];

  get vertexCount(): number {
    return this.pos.length / 3;
  }

  vertex(
    x: number,
    y: number,
    z: number,
    color: RGB = WHITE,
    swayAmp = 0,
    wiggle = 0,
    surf = 0,
    aux = 0,
  ): number {
    this.pos.push(x, y, z);
    this.col.push(color[0], color[1], color[2]);
    this.sway.push(swayAmp, wiggle);
    this.surf.push(surf);
    this.aux.push(aux);
    return this.pos.length / 3 - 1;
  }

  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }

  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  /** Index of the next vertex, for `transformFrom`. */
  mark(): number {
    return this.vertexCount;
  }

  /** Applies a matrix to every vertex written since `mark`. */
  transformFrom(mark: number, m: THREE.Matrix4): void {
    const e = m.elements;
    for (let i = mark * 3; i < this.pos.length; i += 3) {
      const x = this.pos[i];
      const y = this.pos[i + 1];
      const z = this.pos[i + 2];
      this.pos[i] = e[0] * x + e[4] * y + e[8] * z + e[12];
      this.pos[i + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
      this.pos[i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
    }
  }

  /** Overwrites the aux channel for every vertex since `mark`. */
  setAuxFrom(mark: number, value: number): void {
    for (let i = mark; i < this.aux.length; i++) this.aux[i] = value;
  }

  /**
   * A tube swept along a polyline, with frames carried by parallel transport so
   * the rings never twist — after slop-newyork's `addTube`.
   *
   * `radius[i]` is the radius at `path[i]`. A zero last radius closes the tip to
   * a point; `capBase` closes the first ring with a fan.
   */
  addTube(
    path: readonly THREE.Vector3[],
    radius: readonly number[],
    radial: number,
    style: PartStyle,
    capBase = false,
  ): void {
    const n = path.length;
    if (n < 2) return;
    const base = this.vertexCount;
    const tangent = (i: number): THREE.Vector3 =>
      path[Math.min(n - 1, i + 1)].clone().sub(path[Math.max(0, i - 1)]).normalize();

    let T = tangent(0);
    let N =
      Math.abs(T.y) < 0.9
        ? new THREE.Vector3(0, 1, 0).cross(T).normalize()
        : new THREE.Vector3(1, 0, 0).cross(T).normalize();
    const q = new THREE.Quaternion();
    const dir = new THREE.Vector3();
    const B = new THREE.Vector3();

    for (let i = 0; i < n; i++) {
      if (i > 0) {
        const T2 = tangent(i);
        q.setFromUnitVectors(T, T2);
        N.applyQuaternion(q).normalize();
        T = T2;
      }
      B.crossVectors(T, N);
      const t = i / (n - 1);
      const s = style.sway ? style.sway(t) : 0;
      for (let j = 0; j <= radial; j++) {
        const a = j / radial;
        const ang = a * Math.PI * 2;
        dir.copy(N).multiplyScalar(Math.cos(ang)).addScaledVector(B, Math.sin(ang));
        const p = path[i];
        const r = radius[i];
        this.vertex(
          p.x + dir.x * r,
          p.y + dir.y * r,
          p.z + dir.z * r,
          style.color(t, a),
          s,
          style.wiggle ?? 0,
          style.surf ?? 0,
        );
      }
    }
    for (let k = 0; k < n - 1; k++) {
      for (let j = 0; j < radial; j++) {
        const a = base + k * (radial + 1) + j;
        const c = a + radial + 1;
        // Counter-clockwise seen from outside: the ring advances from N toward
        // B = T x N, so (a, a + 1, c) has normal B x T = N.
        this.idx.push(a, a + 1, c, a + 1, c + 1, c);
      }
    }
    if (capBase) {
      const centre = this.vertex(
        path[0].x,
        path[0].y,
        path[0].z,
        style.color(0, 0),
        style.sway ? style.sway(0) : 0,
        style.wiggle ?? 0,
        style.surf ?? 0,
      );
      for (let j = 0; j < radial; j++) this.tri(centre, base + j + 1, base + j);
    }
  }

  /**
   * A surface of revolution about +Y. `profile` runs bottom to top as
   * `[radius, y]`; `displace(angle01, t)` scales the radius per vertex, which is
   * how a sponge gets its ridges and a dome its lumps.
   */
  addLathe(
    profile: readonly (readonly [number, number])[],
    segments: number,
    style: PartStyle,
    displace?: (a: number, t: number) => number,
  ): void {
    const base = this.vertexCount;
    const rows = profile.length;
    for (let i = 0; i < rows; i++) {
      const [r, y] = profile[i];
      const t = i / (rows - 1);
      const s = style.sway ? style.sway(t) : 0;
      for (let j = 0; j <= segments; j++) {
        const a = j / segments;
        const ang = a * Math.PI * 2;
        const k = displace ? displace(a % 1, t) : 1;
        this.vertex(
          Math.cos(ang) * r * k,
          y,
          Math.sin(ang) * r * k,
          style.color(t, a),
          s,
          style.wiggle ?? 0,
          style.surf ?? 0,
        );
      }
    }
    for (let i = 0; i < rows - 1; i++) {
      for (let j = 0; j < segments; j++) {
        const a = base + i * (segments + 1) + j;
        const c = a + segments + 1;
        // Outward-facing: the angle runs (cos, sin) and the rows run upward.
        this.idx.push(a, c, a + 1, a + 1, c, c + 1);
      }
    }
  }

  /**
   * A flat strip — a seagrass blade, a kelp frond, a ray's wing — from a
   * centreline, a width per point and an across direction. Two-sided by
   * material, not by geometry.
   */
  addRibbon(
    path: readonly THREE.Vector3[],
    width: readonly number[],
    across: THREE.Vector3,
    style: PartStyle,
  ): void {
    const n = path.length;
    const base = this.vertexCount;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const s = style.sway ? style.sway(t) : 0;
      const p = path[i];
      const w = width[i] * 0.5;
      for (const side of [-1, 1]) {
        this.vertex(
          p.x + across.x * w * side,
          p.y + across.y * w * side,
          p.z + across.z * w * side,
          style.color(t, side < 0 ? 0 : 1),
          s,
          style.wiggle ?? 0,
          style.surf ?? 0,
        );
      }
    }
    for (let i = 0; i < n - 1; i++) {
      const a = base + i * 2;
      this.idx.push(a, a + 1, a + 3, a, a + 3, a + 2);
    }
  }

  /**
   * A displaced UV sphere, `rings` x `segments`. `shape(dir)` returns the radius
   * multiplier along a unit direction, so lumps and flattening are one function.
   * `yMin` clips the bottom (a dome sitting on the sand loses its buried half).
   */
  addBlob(
    centre: THREE.Vector3,
    radii: THREE.Vector3,
    rings: number,
    segments: number,
    style: PartStyle,
    shape: (dir: THREE.Vector3) => number = () => 1,
    yMin = -1,
  ): void {
    const base = this.vertexCount;
    const dir = new THREE.Vector3();
    const theta0 = Math.acos(Math.max(-1, Math.min(1, yMin)));
    for (let i = 0; i <= rings; i++) {
      // Latitude from the pole (theta 0) down to the clip.
      const theta = (i / rings) * theta0;
      const t = 1 - i / rings;
      for (let j = 0; j <= segments; j++) {
        const a = j / segments;
        const phi = a * Math.PI * 2;
        dir.set(Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi));
        const k = shape(dir);
        this.vertex(
          centre.x + dir.x * radii.x * k,
          centre.y + dir.y * radii.y * k,
          centre.z + dir.z * radii.z * k,
          style.color(t, a),
          style.sway ? style.sway(t) : 0,
          style.wiggle ?? 0,
          style.surf ?? 0,
        );
      }
    }
    for (let i = 0; i < rings; i++) {
      for (let j = 0; j < segments; j++) {
        const a = base + i * (segments + 1) + j;
        const c = a + segments + 1;
        this.idx.push(a, a + 1, c, a + 1, c + 1, c);
      }
    }
  }

  /** Multiplies every vertex colour by `tint` — for composites, where the hue is baked. */
  tint(tint: RGB): this {
    for (let i = 0; i < this.col.length; i += 3) {
      this.col[i] *= tint[0];
      this.col[i + 1] *= tint[1];
      this.col[i + 2] *= tint[2];
    }
    return this;
  }

  /** Appends another builder's contents, optionally transformed. */
  append(other: MeshBuilder, m?: THREE.Matrix4): void {
    const mark = this.mark();
    const offset = this.vertexCount;
    this.pos.push(...other.pos);
    this.col.push(...other.col);
    this.sway.push(...other.sway);
    this.surf.push(...other.surf);
    this.aux.push(...other.aux);
    for (const i of other.idx) this.idx.push(i + offset);
    if (m) this.transformFrom(mark, m);
  }

  /**
   * Bakes to a geometry. Normals are computed from the final triangles rather
   * than authored, so parts can be transformed freely before this.
   */
  build(options: { aux?: boolean } = {}): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setAttribute('reefSway', new THREE.Float32BufferAttribute(this.sway, 2));
    g.setAttribute('reefSurf', new THREE.Float32BufferAttribute(this.surf, 1));
    if (options.aux) g.setAttribute('reefAux', new THREE.Float32BufferAttribute(this.aux, 1));
    g.setIndex(this.idx);
    g.computeVertexNormals();
    g.computeBoundingSphere();
    return g;
  }
}

/** Linear blend of two colours. */
export function mixRGB(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function scaleRGB(a: RGB, k: number): RGB {
  return [a[0] * k, a[1] * k, a[2] * k];
}

/** Cheap deterministic value noise on the sphere of directions, 0..1. */
export function dirNoise(x: number, y: number, z: number, seed: number): number {
  const s =
    Math.sin(x * 3.1 + seed) * Math.sin(y * 2.7 + seed * 1.7) * Math.sin(z * 3.3 + seed * 0.6) +
    0.5 * Math.sin(x * 6.3 + y * 5.1 + seed * 2.3) * Math.sin(z * 5.7 - y * 4.9 + seed);
  return 0.5 + 0.5 * Math.max(-1, Math.min(1, s));
}

export function smoothstep(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/**
 * A branch as a polyline from `p0` along `dir`: `len` metres in `segs` steps,
 * bending toward `up` and wandering by `jitter` — slop-newyork's `growBranch`,
 * reduced to what corals and fans need.
 */
export function growBranch(
  p0: THREE.Vector3,
  dir: THREE.Vector3,
  len: number,
  segs: number,
  up: number,
  jitter: number,
  random: () => number,
  planar: THREE.Vector3 | null = null,
): THREE.Vector3[] {
  const pts = [p0.clone()];
  const d = dir.clone().normalize();
  const p = p0.clone();
  const step = len / segs;
  for (let i = 1; i <= segs; i++) {
    d.x += (random() - 0.5) * jitter;
    d.y += up * step + (random() - 0.5) * jitter * 0.6;
    d.z += (random() - 0.5) * jitter;
    // A sea fan grows in one plane: project the wander back into it.
    if (planar) d.addScaledVector(planar, -d.dot(planar));
    d.normalize();
    p.addScaledVector(d, step);
    pts.push(p.clone());
  }
  return pts;
}

/** Rotates `dir` by `angle` away from itself toward an azimuth around it. */
export function deflect(dir: THREE.Vector3, angle: number, azimuth: number): THREE.Vector3 {
  const ref = Math.abs(dir.y) < 0.95 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(dir, ref).normalize();
  const w = new THREE.Vector3().crossVectors(dir, u);
  return dir
    .clone()
    .multiplyScalar(Math.cos(angle))
    .addScaledVector(u, Math.sin(angle) * Math.cos(azimuth))
    .addScaledVector(w, Math.sin(angle) * Math.sin(azimuth))
    .normalize();
}
