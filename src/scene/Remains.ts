import * as THREE from 'three/webgpu';
import { attribute, cameraPosition, float, mix, normalWorld, positionWorld, vec3, vec4 } from 'three/tsl';
import { uniform } from '../core/uniform';
import { mulberry32 } from '../core/random';
import { seafloorHeight } from './Seafloor';

/**
 * A pirate's remains, half-buried in the sand.
 *
 * **Why this is built rather than downloaded.** Neither Poly Haven nor the CC0
 * end of Sketchfab publishes a weathered skeleton at a scale that would sit next
 * to 40k-triangle photogrammetry; what they do publish is low-poly stylised, and
 * next to the scans it reads as a different game. The shape is also unusually
 * tractable — a skull is a deformed sphere with two holes in it — so this
 * follows the route `Birds` and `Fish` already took for the same reason:
 * procedural geometry, animated from a clock so `resetClock(t)` reproduces a
 * frame exactly.
 *
 * It is one object, placed once: a single merged geometry and a single draw, so
 * it can afford 3,430 triangles on detail that only pays off when the camera is
 * a metre away — orbits, a tooth row, ribs that thin as they curve.
 *
 * No compute and no storage textures: the same node graph has to compile on the
 * WebGL2 fallback.
 *
 * This file used to hold a procedural coconut palm as well, and the grove that
 * scattered it. Both are gone. The palm was a good piece of engineering and a
 * bad tree — its fronds rode on `emissiveNode` to fake transmission, which is
 * neither shadowed nor tone-mapped with the rest of the scene, so under a clear
 * sky the whole grove came out chrome blue. `Props` plants two real palms in its
 * place; see `DRESSING_URLS` there.
 */

/**
 * TSL node objects are structurally dynamic, and expressions composed out of
 * `attribute()` values resolve to `any` — which the overloaded typings then
 * narrow to the wrong constructor. Node-typed locals are therefore `any` by
 * design; the class's public API stays typed.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * A `vec3` node from a `THREE.Color`. Legal at runtime and the way every colour
 * constant in this file reaches a shader; the generated typings simply do not
 * describe the constructor overload that accepts one.
 */
const rgb = vec3 as unknown as (c: THREE.Color) => Node;

const TAU = Math.PI * 2;

// ---------------------------------------------------------------------------
// mesh accumulator
// ---------------------------------------------------------------------------

/**
 * A growing indexed triangle mesh with two free per-vertex channels.
 *
 * Both halves of this file build geometry out of the same handful of shapes, and
 * both need exactly two extra floats a vertex — the palm carries `(part, angle)`
 * and the skeleton carries `(stain, cavity)` — so one accumulator serves both.
 * The channels are deliberately unnamed here: naming them would mean two
 * near-identical classes.
 *
 * `setMatrix` bakes a transform into everything pushed after it, which is how
 * the skeleton assembles thirty separately-authored bones into one geometry and
 * one draw call without a scene graph.
 */
class MeshData {
  readonly position: number[] = [];
  readonly normal: number[] = [];
  readonly channel: number[] = [];
  readonly index: number[] = [];

  private readonly matrix = new THREE.Matrix4();
  private readonly normalMatrix = new THREE.Matrix3();
  private transformed = false;
  private readonly p = new THREE.Vector3();
  private readonly n = new THREE.Vector3();

  /** Bakes `m` into every subsequent vertex. Pass null to stop transforming. */
  setMatrix(m: THREE.Matrix4 | null): void {
    if (m === null) {
      this.transformed = false;
      return;
    }
    this.matrix.copy(m);
    this.normalMatrix.getNormalMatrix(this.matrix);
    this.transformed = true;
  }

  /** Appends a vertex and returns its index. */
  vertex(
    px: number, py: number, pz: number,
    nx: number, ny: number, nz: number,
    c0: number, c1: number,
  ): number {
    if (this.transformed) {
      this.p.set(px, py, pz).applyMatrix4(this.matrix);
      this.n.set(nx, ny, nz).applyMatrix3(this.normalMatrix).normalize();
      this.position.push(this.p.x, this.p.y, this.p.z);
      this.normal.push(this.n.x, this.n.y, this.n.z);
    } else {
      this.position.push(px, py, pz);
      this.normal.push(nx, ny, nz);
    }
    this.channel.push(c0, c1);
    return this.channel.length / 2 - 1;
  }

  tri(a: number, b: number, c: number): void {
    this.index.push(a, b, c);
  }

  /** Two triangles over a quad wound `a -> b -> c -> d`. */
  quad(a: number, b: number, c: number, d: number): void {
    this.index.push(a, b, c, a, c, d);
  }

  get vertexCount(): number {
    return this.channel.length / 2;
  }

  get triangleCount(): number {
    return this.index.length / 3;
  }

  /**
   * Replaces every normal with the area-weighted average of the faces meeting
   * at that vertex.
   *
   * For shapes whose normals are cheaper to derive from the winding than to
   * write down — palm leaflets, which are separate quads with no shared
   * vertices, so this gives each blade a flat normal consistent with the face
   * it was wound as. That consistency is what makes `faceDirection` mean
   * "you are looking at the underside" in the fragment stage rather than
   * "the author guessed".
   */
  recomputeNormals(): void {
    const p = this.position;
    const n = this.normal;
    for (let i = 0; i < n.length; i++) n[i] = 0;

    for (let i = 0; i < this.index.length; i += 3) {
      const a = this.index[i] * 3;
      const b = this.index[i + 1] * 3;
      const c = this.index[i + 2] * 3;
      const ax = p[b] - p[a];
      const ay = p[b + 1] - p[a + 1];
      const az = p[b + 2] - p[a + 2];
      const bx = p[c] - p[a];
      const by = p[c + 1] - p[a + 1];
      const bz = p[c + 2] - p[a + 2];
      // Not normalised: the cross product's length is twice the triangle area,
      // which is exactly the weight a big face should have over a sliver.
      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;
      n[a] += nx; n[a + 1] += ny; n[a + 2] += nz;
      n[b] += nx; n[b + 1] += ny; n[b + 2] += nz;
      n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
    }

    for (let i = 0; i < n.length; i += 3) {
      const len = Math.hypot(n[i], n[i + 1], n[i + 2]);
      if (len < 1e-9) {
        n[i] = 0; n[i + 1] = 1; n[i + 2] = 0;
      } else {
        n[i] /= len; n[i + 1] /= len; n[i + 2] /= len;
      }
    }
  }

  /** Bakes the accumulator into a geometry, naming the two free channels. */
  toGeometry(channelName: string, instanced: boolean): THREE.BufferGeometry {
    const geometry = instanced ? new THREE.InstancedBufferGeometry() : new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.position, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(this.normal, 3));
    geometry.setAttribute(channelName, new THREE.Float32BufferAttribute(this.channel, 2));
    geometry.setIndex(this.index);
    return geometry;
  }
}

/** Fills `out` with a surface point for parameters in [0, 1]. */
type SurfacePoint = (u: number, v: number, out: THREE.Vector3) => void;

const _s0 = new THREE.Vector3();
const _s1 = new THREE.Vector3();
const _s2 = new THREE.Vector3();
const _du = new THREE.Vector3();
const _dv = new THREE.Vector3();
const _sn = new THREE.Vector3();

/**
 * A lathed surface: `lon + 1` columns (the seam is duplicated) by `lat + 1`
 * rows, with the first and last rows collapsed to poles.
 *
 * `point(lon01, lat01)` may be any deformation of a sphere — the normal is taken
 * from central differences of that function rather than from the sphere it
 * started as, which is what lets the cranium have eye sockets pushed into it and
 * still shade like a solid object. The differencing step is clamped away from
 * the poles because the parameterisation degenerates there; the error that
 * introduces is one 1e-3 step of latitude, which is invisible.
 *
 * The seam column is duplicated rather than shared so that anything driven by
 * the longitude parameter — the palm trunk's helical leaf scars — does not have
 * to interpolate the whole way back around the object between the last column
 * and the first.
 */
function lathe(
  md: MeshData,
  lon: number,
  lat: number,
  point: SurfacePoint,
  channel: (u: number, v: number) => readonly [number, number],
): void {
  const h = 1e-3;
  const base = md.vertexCount;

  for (let iy = 0; iy <= lat; iy++) {
    const v = iy / lat;
    const vs = Math.min(1 - h, Math.max(h, v));
    for (let ix = 0; ix <= lon; ix++) {
      const u = ix / lon;
      point(u, v, _s0);
      point(u + h, vs, _s1);
      point(u - h, vs, _s2);
      _du.copy(_s1).sub(_s2);
      point(u, vs + h, _s1);
      point(u, vs - h, _s2);
      _dv.copy(_s1).sub(_s2);
      // du x dv, in that order: the other order is the inward normal, and on a
      // closed solid that is a surface lit from the inside — every face dark
      // except the ones the sun happens to be behind.
      _sn.copy(_du).cross(_dv);
      if (_sn.lengthSq() < 1e-18) _sn.set(0, 1, 0);
      else _sn.normalize();
      const c = channel(u, v);
      md.vertex(_s0.x, _s0.y, _s0.z, _sn.x, _sn.y, _sn.z, c[0], c[1]);
    }
  }

  const stride = lon + 1;
  for (let iy = 0; iy < lat; iy++) {
    for (let ix = 0; ix < lon; ix++) {
      const a = base + iy * stride + ix;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      // The pole rows collapse to a point, so half of each quad there is
      // degenerate; emitting one triangle instead keeps the index buffer honest.
      if (iy === 0) md.tri(a, d, c);
      else if (iy === lat - 1) md.tri(a, b, c);
      else md.quad(a, b, d, c);
    }
  }
}

/** Fills `out` with a point on a swept path, for `s` in [0, 1]. */
type PathPoint = (s: number, out: THREE.Vector3) => void;

const _t0 = new THREE.Vector3();
const _t1 = new THREE.Vector3();
const _tan = new THREE.Vector3();
const _ref = new THREE.Vector3();
const _bi = new THREE.Vector3();
const _nrm = new THREE.Vector3();
const _ring = new THREE.Vector3();

/**
 * A tapered tube swept along `path`, capped with a shallow dome at each end.
 *
 * Every bone in the skeleton is one of these. The frame is built against a fixed
 * reference direction rather than parallel-transported: bones are short arcs
 * that never turn far enough for the frame to degenerate, and a transported
 * frame would make the tube's seam — and therefore any per-vertex channel that
 * follows it — depend on where the sweep started.
 *
 * The caps are what stop a femur from being a drinking straw seen end-on. They
 * are a single fan to a point pushed `0.55 r` past the end along the tangent,
 * which reads as a rounded epiphysis at any distance a viewer can get to.
 */
function tube(
  md: MeshData,
  path: PathPoint,
  radius: (s: number) => number,
  stations: number,
  sides: number,
  reference: THREE.Vector3,
  channel: (s: number, around: number) => readonly [number, number],
): void {
  const base = md.vertexCount;
  const h = 1e-4;

  for (let i = 0; i < stations; i++) {
    const s = i / (stations - 1);
    path(s, _t0);
    path(Math.min(1, s + h), _t1);
    _tan.copy(_t1).sub(_t0);
    path(Math.max(0, s - h), _t1);
    _tan.sub(_t1);
    if (_tan.lengthSq() < 1e-18) _tan.set(0, 1, 0);
    _tan.normalize();

    _ref.copy(reference);
    _bi.copy(_ref).cross(_tan);
    if (_bi.lengthSq() < 1e-8) {
      // The reference happened to line up with the tangent. Any perpendicular
      // will do; this only ever fires on a caller that passed a bad reference.
      _ref.set(_tan.z, _tan.x, _tan.y);
      _bi.copy(_ref).cross(_tan);
    }
    _bi.normalize();
    _nrm.copy(_tan).cross(_bi).normalize();

    const r = radius(s);
    for (let j = 0; j < sides; j++) {
      const a = (j / sides) * TAU;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      _ring.copy(_bi).multiplyScalar(ca).addScaledVector(_nrm, sa);
      const c = channel(s, j / sides);
      md.vertex(
        _t0.x + _ring.x * r, _t0.y + _ring.y * r, _t0.z + _ring.z * r,
        _ring.x, _ring.y, _ring.z,
        c[0], c[1],
      );
    }
  }

  for (let i = 0; i + 1 < stations; i++) {
    for (let j = 0; j < sides; j++) {
      const j2 = (j + 1) % sides;
      const a = base + i * sides + j;
      const b = base + i * sides + j2;
      const c = base + (i + 1) * sides + j;
      const d = base + (i + 1) * sides + j2;
      // Wound against the (binormal, normal, tangent) frame this sweep builds,
      // which is right-handed — so the outward face is a -> b -> d -> c and the
      // mirror of it is a tube you can only see the inside of.
      md.quad(a, b, d, c);
    }
  }

  // Caps. Wound so that the start cap faces backwards along the sweep and the
  // end cap forwards, which is what `quad`'s ordering above implies for the wall.
  for (const end of [0, 1] as const) {
    const s = end;
    path(s, _t0);
    path(end === 0 ? h : 1 - h, _t1);
    _tan.copy(_t0).sub(_t1);
    if (_tan.lengthSq() < 1e-18) _tan.set(0, 1, 0);
    _tan.normalize();
    const r = radius(s);
    const c = channel(s, 0);
    const tip = md.vertex(
      _t0.x + _tan.x * r * 0.55, _t0.y + _tan.y * r * 0.55, _t0.z + _tan.z * r * 0.55,
      _tan.x, _tan.y, _tan.z,
      c[0], c[1],
    );
    const ring = base + (end === 0 ? 0 : (stations - 1) * sides);
    for (let j = 0; j < sides; j++) {
      const j2 = (j + 1) % sides;
      if (end === 0) md.tri(tip, ring + j2, ring + j);
      else md.tri(tip, ring + j, ring + j2);
    }
  }
}

// ---------------------------------------------------------------------------
// shared shading helpers
// ---------------------------------------------------------------------------

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * How much direct sun a ground object should receive, from the sun's elevation.
 *
 * Derived rather than exposed, for the same reason `Birds` derives it: a caller
 * that has to remember to fade the sun term is a caller that will eventually
 * ship a set piece lit like noon against a night sky.
 */
function sunGainFor(sun: THREE.Vector3): number {
  const above = clamp01((sun.y + 0.09) / 0.18);
  return above * above * (3 - 2 * above);
}

// ===========================================================================
// remains
// ===========================================================================

/**
 * Layout seed. Fixed, so it is the same skeleton every session — the mottled
 * staining and the scatter of loose bones are drawn from it, and a set piece
 * whose blotches move between loads is a set piece no screenshot baseline can
 * be compared against.
 */
const BONE_SEED = 0x5be1e7;

/**
 * How far the whole arrangement is sunk below the local sand, metres.
 *
 * Small on purpose. The burial is authored *into* the layout — the pelvis and
 * the lower ribs sit below y = 0 in the skeleton's own frame and the skull only
 * just clears it — rather than achieved by dropping a complete skeleton into the
 * ground, because sinking a whole body uniformly buries the parts that read
 * (the orbits, the mandible line) at the same rate as the parts that do not.
 */
const BONE_SINK = 0.03;

/** Half-width of the terrain sample the arrangement is levelled against. */
const BONE_GROUND_SPAN = 3;

/** Bone tessellation. Ribs are 15 mm across; five sides is plenty. */
const RIB_SIDES = 5;
const RIB_STATIONS = 9;
const LIMB_SIDES = 6;
const LIMB_STATIONS = 9;

/** Cranium tessellation. */
const SKULL_LON = 20;
const SKULL_LAT = 14;

/**
 * Bone colour, in three states, because a uniform bone-white plastic look is the
 * failure mode here and one albedo cannot avoid it.
 *
 * `BONE_CLEAN` is sun-bleached cortical bone — off-white with a warm cast, never
 * pure white; nothing in nature is, and 1.0 albedo under a 3.2-intensity sun
 * clips to a white silhouette with no form in it at all. `BONE_STAINED` is what
 * a decade in wet sand does: the mineral takes up iron and the buried half comes
 * out ochre. `BONE_CAVITY` is not a pigment at all, it is the ambient occlusion
 * of a hole — the orbits, the nasal aperture and the shadowed inner faces of the
 * ribs, baked as a vertex channel because a 20-segment sphere has no way to
 * cast a shadow into its own eye socket.
 */
const BONE_CLEAN = new THREE.Color(0.78, 0.75, 0.67);
const BONE_STAINED = new THREE.Color(0.46, 0.38, 0.26);
const BONE_CAVITY = new THREE.Color(0.1, 0.085, 0.07);

/**
 * Light that comes through the thin edges of bone, and how much of it there is.
 *
 * Cortical bone a couple of millimetres thick genuinely passes light, and it is
 * warm and orange when it does — which is exactly what stops the close look from
 * resolving into painted plastic. Driven by a Fresnel term, so it appears only
 * at grazing angles where the path through the material is short: the rim of the
 * cranium, the edge of a rib, the thin arch of the zygomatic. Weaker than the
 * fronds' by a factor of four, because bone is not a leaf.
 */
const BONE_TRANSMISSION = new THREE.Color(0.95, 0.55, 0.34);
const BONE_TRANSMISSION_GAIN = 0.2;

/**
 * The tide line, in the skeleton's own frame.
 *
 * `place` seats local y = 0 at `BONE_SINK` below the sand, so the sand surface
 * is at local `+BONE_SINK` and these two straddle it: clean above 6 cm, fully
 * stained at 0, and a 6 cm gradient across the middle. The gradient is the
 * whole trick. Bone that has sat half in wet sand has a *line* on it, and the
 * line is what says "buried" rather than "dropped here this morning" — a
 * uniformly stained skeleton and a uniformly clean one fail the same way.
 */
const STAIN_TOP = 0.06;
const STAIN_BOTTOM = 0;

export interface RemainsOptions {
  /** Overrides the layout and mottling seed. */
  seed?: number;
}

const _boneMatrix = new THREE.Matrix4();
const _boneEuler = new THREE.Euler();
const _boneQuat = new THREE.Quaternion();
const _bonePos = new THREE.Vector3();
const _boneScale = new THREE.Vector3(1, 1, 1);
const _bonePath = new THREE.Vector3();

/** Composes a placement matrix for one bone, in the skeleton's own frame. */
function bonePlace(
  x: number, y: number, z: number,
  rx: number, ry: number, rz: number,
): THREE.Matrix4 {
  _bonePos.set(x, y, z);
  _boneEuler.set(rx, ry, rz, 'YXZ');
  _boneQuat.setFromEuler(_boneEuler);
  return _boneMatrix.compose(_bonePos, _boneQuat, _boneScale);
}

/**
 * How stained a vertex at local height `y` is. See `STAIN_TOP`.
 *
 * Sampled from the *authored* height rather than the world height, so the tide
 * line survives the whole arrangement being tilted onto a slope — which is
 * correct, because the sand it was buried in tilted with it.
 */
function burialStain(y: number, mottle: number): number {
  const t = clamp01((STAIN_TOP - y) / (STAIN_TOP - STAIN_BOTTOM));
  const smooth = t * t * (3 - 2 * t);
  // Mottling on top of it, or the line reads as a dip-dye. Weighted so it can
  // only ever add stain to a clean bone and never bleach a buried one.
  return clamp01(smooth * 0.82 + mottle * 0.28 * (1 - smooth * 0.5));
}

/**
 * The long-bone radius profile: bulbous at both ends, waisted in the middle.
 *
 * `s` runs 0 to 1 along the shaft. This is the single shape that makes a tapered
 * tube read as a femur rather than as a stick — the epiphyses are nearly twice
 * the diameter of the diaphysis, and it is the only anatomical detail visible on
 * a leg bone from more than two metres away.
 */
function limbProfile(s: number, shaft: number, endGain: number): number {
  const ends = Math.pow(Math.abs(s * 2 - 1), 2.4);
  return shaft * (1 + (endGain - 1) * ends);
}

/** Straight-line path between two points, for the limb sweeps. */
function segment(
  x0: number, y0: number, z0: number,
  x1: number, y1: number, z1: number,
): PathPoint {
  return (s, out) => out.set(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, z0 + (z1 - z0) * s);
}

/**
 * Frame references for the sweeps. See `tube`: the reference only has to stay
 * away from parallel with the tangent, and picking it per bone group rather than
 * transporting a frame keeps every tube's seam in a predictable place.
 */
const _boneRefY = new THREE.Vector3(0, 1, 0);
const _boneRefZ = new THREE.Vector3(0, 0, 1);

const _skullDir = new THREE.Vector3();
const _orbitR = new THREE.Vector3(0.4, 0.03, 0.92).normalize();
const _orbitL = new THREE.Vector3(-0.4, 0.03, 0.92).normalize();
const _nasal = new THREE.Vector3(0, -0.3, 0.95).normalize();

/** Hermite ramp, 1 inside `inner` and 0 outside `outer`. */
function falloff(d: number, inner: number, outer: number): number {
  const t = clamp01((outer - d) / (outer - inner));
  return t * t * (3 - 2 * t);
}

/**
 * How deep the skull surface is pushed in at direction `n`, and how much of a
 * cavity it is.
 *
 * Three recesses, and only three, because they are the three that make a lump of
 * bone read instantly as a skull: the two orbits and the nasal aperture. The
 * measure is the *tangential* distance from each recess's axis, with the
 * vertical component scaled so the orbits come out wider than they are tall —
 * a circular socket reads as a cartoon.
 *
 * Displacement and darkening come out of the same function because they have to
 * agree: a socket that is geometrically deep but shaded like the brow beside it
 * disappears the moment the sun is anywhere but behind the viewer, and a socket
 * that is painted dark but not sunk is a decal.
 */
function skullRecess(n: THREE.Vector3): { depth: number; cavity: number } {
  let depth = 0;
  let cavity = 0;

  for (const axis of [_orbitR, _orbitL]) {
    const along = n.dot(axis);
    if (along <= 0) continue;
    // Tangential offset from the socket axis, with the vertical squashed so the
    // socket is an ellipse lying on its side.
    const tx = n.x - axis.x * along;
    const ty = (n.y - axis.y * along) * 1.3;
    const tz = n.z - axis.z * along;
    const e = Math.hypot(tx, ty, tz);
    const m = falloff(e, 0.16, 0.44);
    depth += 0.021 * m;
    cavity = Math.max(cavity, m);
  }

  const nasalAlong = n.dot(_nasal);
  if (nasalAlong > 0) {
    // Narrow and tall, unlike the orbits — the aperture is a keyhole.
    const tx = (n.x - _nasal.x * nasalAlong) * 2.8;
    const ty = (n.y - _nasal.y * nasalAlong) * 0.9;
    const tz = (n.z - _nasal.z * nasalAlong) * 2.8;
    const m = falloff(Math.hypot(tx, ty, tz), 0.05, 0.34);
    depth += 0.016 * m;
    cavity = Math.max(cavity, m * 0.9);
  }

  return { depth, cavity };
}

/**
 * The cranium: a sphere with a face pulled out of the front of it.
 *
 * Authored with +Z anterior and +Y superior, in metres, at life size — a human
 * cranium is 19 cm long, 15 wide and 14 tall, and getting that right matters
 * more than any amount of surface detail, because the viewer has a very
 * accurate idea of how big a skull is and will read a wrong one as a prop.
 *
 * The deformations, in the order they are applied:
 *  - the neurocranium as a tri-axial ellipsoid;
 *  - the face pulled forward and down, and narrowed, which is the whole of the
 *    maxilla — a separate mesh for it would double the vertex count for a shape
 *    that is continuous with the cranium anyway;
 *  - the cranial base flattened, so the skull sits on sand instead of rocking;
 *  - the temporal fossae hollowed, which is what puts the corner in the
 *    silhouette between the brow and the ear;
 *  - the orbits and the nasal aperture, from `skullRecess`.
 */
function addCranium(md: MeshData, place: THREE.Matrix4): void {
  md.setMatrix(place);
  lathe(
    md,
    SKULL_LON,
    SKULL_LAT,
    (u, v, out) => {
      const phi = u * TAU;
      const theta = v * Math.PI;
      const st = Math.sin(theta);
      _skullDir.set(Math.cos(phi) * st, Math.cos(theta), Math.sin(phi) * st);
      const n = _skullDir;

      let x = n.x * 0.0735;
      let y = n.y * 0.0700;
      let z = n.z * 0.0930;

      const face = clamp01((n.z - 0.1) / 0.55) * clamp01((-n.y + 0.05) / 0.6);
      z += face * 0.026;
      y -= face * 0.034;
      x *= 1 - face * 0.3;

      // The cranial base is nearly flat and the occiput is not a hemisphere.
      if (y < -0.049) y = -0.049 - (y + 0.049) * 0.25;

      const temple = clamp01((Math.abs(n.x) - 0.68) / 0.3) * falloff(Math.abs(n.y - 0.1), 0.1, 0.5);
      x *= 1 - temple * 0.11;

      const recess = skullRecess(n);
      out.set(x - n.x * recess.depth, y - n.y * recess.depth, z - n.z * recess.depth);
    },
    (u, v) => {
      const phi = u * TAU;
      const theta = v * Math.PI;
      const st = Math.sin(theta);
      _skullDir.set(Math.cos(phi) * st, Math.cos(theta), Math.sin(phi) * st);
      return [0, skullRecess(_skullDir).cavity];
    },
  );
  md.setMatrix(null);
}

/**
 * The dental arch, shared by the mandible and the tooth rows.
 *
 * `t` runs -1 at the left condyle through 0 at the chin to +1 at the right. The
 * arch is a parabola in plan and the ramus rises only over the last quarter,
 * which is what gives the jaw its L-shaped profile rather than a banana's.
 */
function jawPoint(t: number, out: THREE.Vector3): THREE.Vector3 {
  const a = Math.abs(t);
  const rise = clamp01((a - 0.72) / 0.28);
  return out.set(
    t * 0.051,
    -0.026 + rise * rise * (3 - 2 * rise) * 0.062,
    0.052 - a * a * 0.105,
  );
}

/** The mandible, plus the two tooth rows it and the maxilla carry. */
function addJaw(md: MeshData, place: THREE.Matrix4): void {
  md.setMatrix(place);

  tube(
    md,
    (s, out) => jawPoint(s * 2 - 1, out),
    // Thinner at the condyles than at the body, and the body is the part the
    // silhouette is made of.
    (s) => 0.0135 * (1 - 0.3 * Math.abs(s * 2 - 1)),
    15,
    6,
    _boneRefZ,
    () => [0, 0.12],
  );

  // Teeth, on both jaws. Sixteen small nubs is a lot of draw for a 4 cm arch,
  // and it is the single most recognisable thing on a skull at arm's length —
  // a jaw modelled as a smooth tube reads as a horseshoe of driftwood.
  for (const upper of [false, true] as const) {
    for (let i = 0; i < 7; i++) {
      const t = (i / 6) * 2 - 1;
      jawPoint(t * 0.86, _bonePath);
      const baseY = _bonePath.y + (upper ? 0.031 : 0.006);
      const dir = upper ? -1 : 1;
      const x = _bonePath.x * 0.9;
      const z = _bonePath.z * 0.94;
      tube(
        md,
        segment(x, baseY, z, x, baseY + dir * 0.011, z),
        (s) => 0.0046 - s * 0.0014,
        2,
        4,
        _boneRefZ,
        // Teeth are enamel, not bone: paler and far less stained than the jaw
        // they sit in, which is why they are the last thing to disappear.
        () => [-0.55, 0],
      );
    }
  }

  md.setMatrix(null);
}

/**
 * The spine and the ribcage, lying supine with the lower half in the sand.
 *
 * Each rib is an elliptical arc in the body's cross-section, swept from the
 * vertebra at the bottom — the spine is the *lowest* part of a body on its back,
 * which is why the cage reads as a row of hoops coming up out of the sand rather
 * than as a basket sitting on it — up the side and over toward the sternum. The
 * radius falls by 40% along the way, because a rib genuinely does thin as it
 * curves forward and it is the only thing that keeps twelve identical arcs from
 * looking machined.
 *
 * Three ribs are missing and one is short. That is the difference between
 * remains and an anatomical model: something ate here.
 */
function addTorso(md: MeshData, random: () => number): void {
  // The spine. Buried, with only the transverse processes breaking the sand, so
  // its job is to be the thing the ribs are obviously attached to.
  tube(
    md,
    (s, out) => out.set(0, -0.024 + Math.sin(s * Math.PI) * 0.012, 0.1 + s * 0.56),
    // Six lobes along the length, which at this station count reads as
    // segmentation rather than as a smooth pipe. Not twelve: at four stations a
    // cycle the higher count aliases into a beat pattern.
    (s) => 0.0195 + 0.0055 * Math.cos(s * 6 * TAU),
    21,
    5,
    _boneRefY,
    () => [0.25, 0.15],
  );

  const missing = new Set(['1:-1', '3:1', '5:-1']);
  for (let k = 0; k < 6; k++) {
    for (const side of [1, -1]) {
      if (missing.has(`${k}:${side}`)) continue;
      const shortened = k === 2 && side === 1;
      const z0 = 0.52 - k * 0.058;
      // The cage is widest at the fourth rib and tapers both ways.
      const w = 0.108 + 0.036 * Math.sin((Math.PI * (k + 0.9)) / 7);
      const h = 0.086 + 0.012 * Math.sin((Math.PI * (k + 0.9)) / 7);
      // Lower ribs are floating: they stop well short of the midline in front.
      const arc = (shortened ? 1.5 : 2.62 - k * 0.16) as number;

      tube(
        md,
        (s, out) => {
          const ang = -1.5 + s * arc;
          out.set(side * w * Math.cos(ang), 0.062 + h * Math.sin(ang), z0 - s * 0.05);
        },
        (s) => 0.0102 * (1 - 0.42 * s),
        RIB_STATIONS,
        RIB_SIDES,
        _boneRefZ,
        // The vertebral end sits deep inside the cage and inside the sand; the
        // sternal end is out in the light.
        (s) => [0, 0.3 * (1 - clamp01(s * 2.2))],
      );
    }
  }

  // The pelvis: two blades, sunk almost to the crest. Flattened ellipsoids
  // rather than anything anatomical — from anywhere a viewer can stand, an
  // ilium is a curved plate, and this is a curved plate.
  for (const side of [1, -1]) {
    md.setMatrix(bonePlace(side * 0.082, -0.012, 0.095, 0.18, side * 0.5, side * -0.35));
    lathe(
      md,
      9,
      6,
      (u, v, out) => {
        const phi = u * TAU;
        const theta = v * Math.PI;
        const st = Math.sin(theta);
        out.set(Math.cos(phi) * st * 0.026, Math.cos(theta) * 0.082, Math.sin(phi) * st * 0.062);
      },
      () => [0.1, 0.1],
    );
    md.setMatrix(null);
  }

  // Loose bones — phalanges, a shed vertebra, a fragment. Scattered rather than
  // arranged, and drawn from the seeded stream so the scatter is the same one
  // every session.
  for (let i = 0; i < 11; i++) {
    const ang = random() * TAU;
    const r = 0.18 + Math.sqrt(random()) * 0.55;
    const x = Math.cos(ang) * r * 1.15;
    const z = 0.28 + Math.sin(ang) * r;
    const len = 0.022 + random() * 0.05;
    const lie = random() * TAU;
    const y = 0.004 + random() * 0.032;
    // Drawn here and not inside the profile callback: the callback runs once a
    // station, so a draw in there would give the bone a randomly lumpy radius
    // and would make the PRNG's position depend on the tessellation.
    const thickness = 0.0062 + random() * 0.002;
    tube(
      md,
      segment(x, y, z, x + Math.cos(lie) * len, y + 0.004, z + Math.sin(lie) * len),
      (s) => limbProfile(s, thickness, 1.5),
      5,
      5,
      _boneRefY,
      () => [0.18, 0],
    );
  }
}

/**
 * Arms and legs, arranged as a fall rather than as a diagram.
 *
 * Every joint is a coincident endpoint rather than a real articulation, which is
 * exactly what a body that has been lying in sand for years looks like: the
 * cartilage is long gone and the bones have settled into contact. The asymmetry
 * is the point — one leg is straight and one is folded out sideways, one arm is
 * thrown up past the head and the other has dropped across the chest. A
 * symmetric skeleton reads as a display case.
 */
function addLimbs(md: MeshData): void {
  const bones: ReadonlyArray<readonly [number, number, number, number, number, number, number, number]> = [
    // x0, y0, z0, x1, y1, z1, shaft radius, epiphysis gain
    [0.085, 0.014, 0.075, 0.2, 0.026, -0.34, 0.0155, 1.9], // right femur
    [0.2, 0.026, -0.34, 0.162, 0.012, -0.7, 0.0132, 1.85], // right tibia, sinking at the ankle
    [-0.09, 0.01, 0.07, -0.27, 0.024, -0.24, 0.0155, 1.9], // left femur, flung out
    [-0.27, 0.024, -0.24, -0.15, 0.004, -0.57, 0.0132, 1.85], // left tibia, folded back and under
    [0.145, 0.026, 0.5, 0.33, 0.03, 0.65, 0.0125, 1.7], // right humerus, thrown up past the head
    [0.33, 0.03, 0.65, 0.45, 0.008, 0.86, 0.0092, 1.65], // right ulna, hand lost in the sand
    [-0.15, 0.048, 0.49, -0.06, 0.052, 0.28, 0.0125, 1.7], // left humerus, across the chest
  ];

  for (const [x0, y0, z0, x1, y1, z1, shaft, gain] of bones) {
    tube(
      md,
      segment(x0, y0, z0, x1, y1, z1),
      (s) => limbProfile(s, shaft, gain),
      LIMB_STATIONS,
      LIMB_SIDES,
      _boneRefY,
      () => [0, 0],
    );
  }
}

/**
 * The whole arrangement, baked into one geometry and therefore one draw.
 *
 * The stain is applied here, after everything is placed, because it is a
 * function of where a vertex ended up relative to the sand and not of which bone
 * it belongs to. `channel[0]` up to this point is a per-part *bias* — negative on
 * the teeth, positive on the parts that spent their whole time buried — which
 * this pass adds to the burial gradient rather than replacing.
 */
function buildRemainsGeometry(seed: number): THREE.BufferGeometry {
  const md = new MeshData();
  const random = mulberry32(seed);

  // Blotch phases, drawn before anything else so the layout stays stable if the
  // skeleton later gains a bone.
  const p0 = random() * TAU;
  const p1 = random() * TAU;
  const p2 = random() * TAU;
  const p3 = random() * TAU;

  // The head has rolled to one side and the jaw has dropped away from it, which
  // is what a mandible does once the muscle is gone. Between them they are the
  // silhouette: the cranium is barely proud of the sand, and what the eye
  // catches first is the dark of the orbits and the open line of the jaw.
  addCranium(md, bonePlace(0.03, 0.055, 0.74, -0.35, 0.75, 0.3));
  // The mandible has dropped away from the cranium and turned a little further
  // than it did — which is what a jaw does once the muscle holding it is gone,
  // and it is the reason the gap between the two rows of teeth reads as a mouth
  // rather than as a modelling seam. It also sits lower, so the chin is under
  // the sand and the ramus is not: the jaw line is half of the silhouette.
  addJaw(md, bonePlace(0.055, 0.022, 0.775, -0.18, 0.9, 0.22));
  addTorso(md, random);
  addLimbs(md);

  const bias = md.channel;
  const pos = md.position;
  for (let i = 0; i < md.vertexCount; i++) {
    const x = pos[i * 3];
    const y = pos[i * 3 + 1];
    const z = pos[i * 3 + 2];
    // Two octaves of smooth blotching at roughly 0.9 m and 0.45 m, which is the
    // scale sediment staining actually comes in. White noise per vertex would be
    // cheaper and would read as dirt on the lens.
    const mottle = clamp01(
      0.5 +
        0.5 * (Math.sin(x * 7.3 + p0) * Math.sin(z * 6.1 + p1) * 0.62 +
               Math.sin(x * 13.7 + p2) * Math.sin(y * 11.3 + p3) * 0.38),
    );
    bias[i * 2] = clamp01(burialStain(y, mottle) + bias[i * 2]);
  }

  const geometry = md.toGeometry('boneTint', false);
  geometry.name = 'remains';
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return geometry;
}

/**
 * A pirate's remains, half-buried in the sand: one mesh, one draw, ~3,400
 * triangles.
 *
 * Deliberately *not* an instanced field. There is one of these in the world and
 * the whole point of it is that it rewards walking up to it, so the budget goes
 * on the things that survive a close look — the orbits, the tooth row, ribs that
 * thin as they curve — rather than on making it cheap to repeat. At this size
 * that is 3,400 triangles, which is a twentieth of one of the photogrammetry
 * rocks `Props` scatters ninety of.
 *
 * There is no clock and nothing to animate: bones do not move. `place` seats the
 * whole arrangement on `seafloorHeight` and levels it to the local slope, so it
 * follows the island wherever the island goes.
 */
export class Remains {
  readonly object: THREE.Object3D;

  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.MeshStandardNodeMaterial;
  private readonly mesh: THREE.Mesh;
  private disposed = false;

  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(1, 0.96, 0.9));
  private readonly uSunGain = uniform(1);

  constructor(options: RemainsOptions = {}) {
    this.geometry = buildRemainsGeometry(options.seed ?? BONE_SEED);
    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'remains';
    // Set as the object deserves rather than as today's shadow camera is framed;
    // see the same note on `Palms`. A skull with no contact shadow floats, and
    // the contact shadow is most of what says "half-buried".
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = true;

    this.object = new THREE.Object3D();
    this.object.name = 'remains';
    this.object.add(this.mesh);
  }

  /**
   * Seats the remains on the sand at (x, z).
   *
   * The whole arrangement is levelled to the local ground normal, taken from a
   * `BONE_GROUND_SPAN`-wide sample of the same heightfield the floor mesh is
   * built from — so a body on a slope lies along it instead of having one hip in
   * the air and the other underground. `yaw` turns the body about its own axis;
   * 0 puts the head toward +z.
   *
   * Nothing here assumes where the beach is. Hand it a point and it will find
   * the ground under it.
   */
  place(x: number, z: number, yaw = 0): void {
    if (this.disposed) return;
    const span = BONE_GROUND_SPAN;
    const dx = (seafloorHeight(x + span, z) - seafloorHeight(x - span, z)) / (2 * span);
    const dz = (seafloorHeight(x, z + span) - seafloorHeight(x, z - span)) / (2 * span);
    _boneNormal.set(-dx, 1, -dz).normalize();

    this.object.position.set(x, seafloorHeight(x, z) - BONE_SINK, z);
    this.object.quaternion.setFromUnitVectors(_boneUp, _boneNormal);
    this.object.quaternion.multiply(_boneSpin.setFromAxisAngle(_boneUp, yaw));
    this.object.updateMatrix();
    this.object.updateMatrixWorld(true);
  }

  setVisible(v: boolean): void {
    this.object.visible = v;
  }

  /**
   * The sun, for the thin-edge translucency. `direction` points toward the sun.
   *
   * Optional, like `Palms.setSun` and for the same reason: the default is a
   * defensible mid-morning sun, so a caller that never wires this gets a
   * skeleton that looks right rather than one that looks broken.
   */
  setSun(direction: THREE.Vector3, color?: THREE.Color): void {
    const sun = this.uSunDir.value as THREE.Vector3;
    sun.copy(direction);
    if (sun.lengthSq() < 1e-8) sun.set(0, 1, 0);
    sun.normalize();
    this.uSunGain.value = sunGainFor(sun);
    if (color) (this.uSunColor.value as THREE.Color).copy(color);
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

  private buildMaterial(): THREE.MeshStandardNodeMaterial {
    const material = new THREE.MeshStandardNodeMaterial();
    material.name = 'bone';
    material.metalness = 0;

    const tint: Node = attribute('boneTint', 'vec2');
    const stain: Node = tint.x.clamp(0, 1);
    const cavity: Node = tint.y.clamp(0, 1);

    // Three tones, not one. See `BONE_CLEAN`: a single albedo is precisely the
    // uniform-plastic failure this is written to avoid, and the cavity term is
    // doing the job a shadow cannot — a 20-segment sphere has no way to occlude
    // its own eye socket.
    const base = mix(rgb(BONE_CLEAN), rgb(BONE_STAINED), stain).toVar();
    const albedo: Node = mix(base, rgb(BONE_CAVITY), cavity.mul(0.85));
    material.colorNode = vec4(albedo, 1);
    // Bleached bone is chalky; sand-stained bone is chalkier still. Neither is
    // anywhere near smooth, and a glossy skull is a plastic one.
    material.roughnessNode = mix(float(0.6), float(0.88), stain);

    // No custom `positionNode` or `normalNode` here — nothing deforms — so the
    // standard `normalWorld` is the real surface normal and can be used directly.
    const view = cameraPosition.sub(positionWorld).normalize().toVar();
    const nrm: Node = normalWorld;
    // Grazing angles only: that is where the path through the material is short
    // enough for anything to get through, and it is why this shows up on the rim
    // of the cranium and the edge of a rib and nowhere else.
    const rim = float(1).sub(nrm.dot(view).abs()).pow(2.6).toVar();
    const lit = nrm.dot(this.uSunDir).negate().max(0).mul(0.6).add(0.4).toVar();
    material.emissiveNode = rgb(BONE_TRANSMISSION)
      .mul(this.uSunColor)
      .mul(rim.mul(lit).mul(cavity.oneMinus()).mul(BONE_TRANSMISSION_GAIN).mul(this.uSunGain));

    return material;
  }
}

const _boneNormal = new THREE.Vector3();
const _boneUp = new THREE.Vector3(0, 1, 0);
const _boneSpin = new THREE.Quaternion();
