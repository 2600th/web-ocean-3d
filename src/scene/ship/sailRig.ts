import * as THREE from 'three/webgpu';

/**
 * Reads the sail plan out of the hero model.
 *
 * `dutch_ship_medium.glb` has three meshes: hull, rigging and sails, each one
 * joined. Inspected (model units, +X bow, +Y up, +Z starboard):
 *
 *   - The sails mesh is four separate islands. Two are set courses, main and
 *     fore, about 3.9 and 2.5 units deep and bellied well forward. Two are the
 *     topsails, which exist only as furled bundles about one unit deep hanging
 *     from their yards, so the model has no set topsail to animate at all.
 *   - The rigging mesh is one mesh of 916 islands. The four yards are islands
 *     of their own (168 vertices each), already braced about 15 degrees with
 *     the starboard arm forward, but the lifts, footropes and braces that
 *     meet them are separate islands that end at the yardarms. Rotating a
 *     yard therefore needs those ropes bent with it, and `Birds` perches gulls
 *     on fixed yardarm points; see `Sails` for why the yards stay put.
 *
 * So the sails are rebuilt rather than deformed: each course keeps its head on
 * its yard and its clews where the model put them, and each topsail is made
 * from its yard down to the course yardarms below it, using the course's
 * canvas texture region. Every number is measured from the meshes here rather
 * than written in, so `dutch_ship_large_01` would not need new constants.
 */

export interface SailSpec {
  name: string;
  kind: 'course' | 'topsail';
  /** Head corners on the yard, port then starboard, model space. */
  headL: THREE.Vector3;
  headR: THREE.Vector3;
  /** Clews as the model has them (eased, filled). */
  clewL: THREE.Vector3;
  clewR: THREE.Vector3;
  /** Texture coordinates of the head and clew corners, same order. */
  uvHeadL: THREE.Vector2;
  uvHeadR: THREE.Vector2;
  uvClewL: THREE.Vector2;
  uvClewR: THREE.Vector2;
  /** Triangle indices of the furled bundle in the source mesh, or null. */
  bundle: Uint32Array | null;
}

export interface SailPlan {
  sails: SailSpec[];
  /** The original sails mesh, which keeps drawing the furled topsail bundles. */
  sourceMesh: THREE.Mesh;
  /** Model-space transform of the source mesh's geometry. */
  sourceMatrix: THREE.Matrix4;
  /** Main truck: the highest point of the rigging, model space. */
  masthead: THREE.Vector3;
}

interface Island {
  ids: number[];
  min: THREE.Vector3;
  max: THREE.Vector3;
  tris: number[];
}

export function readSailPlan(model: THREE.Object3D): SailPlan | null {
  model.updateMatrixWorld(true);
  const toModel = new THREE.Matrix4().copy(model.matrixWorld).invert();

  let sailMesh: THREE.Mesh | null = null;
  let rigMesh: THREE.Mesh | null = null;
  model.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    const name = `${mesh.name} ${mesh.parent?.name ?? ''} ${mat?.name ?? ''}`.toLowerCase();
    if (name.includes('sail') && !sailMesh) sailMesh = mesh;
    else if (name.includes('rigging') && !rigMesh) rigMesh = mesh;
  });
  if (!sailMesh) return null;
  const sail = sailMesh as THREE.Mesh;

  const sourceMatrix = new THREE.Matrix4().multiplyMatrices(toModel, sail.matrixWorld);
  const geo = sail.geometry;
  const pos = geo.getAttribute('position');
  const uv = geo.getAttribute('uv');
  const index = geo.getIndex();
  if (!pos || !uv || !index) return null;

  const P: THREE.Vector3[] = [];
  for (let i = 0; i < pos.count; i++) P.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(sourceMatrix));
  const islands = splitIslands(P, index);

  const sets = islands.filter((s) => s.max.y - s.min.y > 0.4 * (s.max.z - s.min.z));
  const bundles = islands.filter((s) => !sets.includes(s));
  const uvOf = (i: number) => new THREE.Vector2(uv.getX(i), uv.getY(i));
  // Corners by diagonal extremes: the head corners are the highest points at
  // either side, the clews the lowest.
  const corner = (s: Island, sz: number, sy: number) => {
    let best = -Infinity;
    let bi = s.ids[0];
    for (const i of s.ids) {
      const v = P[i].z * sz + P[i].y * sy;
      if (v > best) {
        best = v;
        bi = i;
      }
    }
    return bi;
  };

  const courses: SailSpec[] = sets.map((s, n) => {
    const hl = corner(s, -1, 1);
    const hr = corner(s, 1, 1);
    const cl = corner(s, -1, -1);
    const cr = corner(s, 1, -1);
    return {
      name: `course-${n}`,
      kind: 'course' as const,
      headL: P[hl].clone(),
      headR: P[hr].clone(),
      clewL: P[cl].clone(),
      clewR: P[cr].clone(),
      uvHeadL: uvOf(hl),
      uvHeadR: uvOf(hr),
      uvClewL: uvOf(cl),
      uvClewR: uvOf(cr),
      bundle: null,
    };
  });

  const topsails: SailSpec[] = [];
  for (const b of bundles) {
    const mid = (b.min.x + b.max.x) / 2;
    // The course on the same mast is the one nearest fore and aft.
    let course: SailSpec | null = null;
    let bestD = Infinity;
    for (const c of courses) {
      const d = Math.abs((c.headL.x + c.headR.x) / 2 - mid);
      if (d < bestD) {
        bestD = d;
        course = c;
      }
    }
    if (!course) continue;
    const hl = corner(b, -1, 1);
    const hr = corner(b, 1, 1);
    // The bundle hangs from the yard and stops a little short of its arms; the
    // set sail spreads to the full yard.
    const headL = P[hl].clone();
    const headR = P[hr].clone();
    const c = headL.clone().add(headR).multiplyScalar(0.5);
    headL.sub(c).multiplyScalar(1.12).add(c);
    headR.sub(c).multiplyScalar(1.12).add(c);
    // Topsail sheets lead to the course yardarms, a little inboard of the ends.
    const cl = course.headL.clone().lerp(course.headR, 0.03);
    const cr = course.headR.clone().lerp(course.headL, 0.03);
    cl.y -= 0.05;
    cr.y -= 0.05;
    topsails.push({
      name: `topsail-${topsails.length}`,
      kind: 'topsail',
      headL,
      headR,
      clewL: cl,
      clewR: cr,
      uvHeadL: course.uvHeadL.clone(),
      uvHeadR: course.uvHeadR.clone(),
      uvClewL: course.uvClewL.clone(),
      uvClewR: course.uvClewR.clone(),
      bundle: Uint32Array.from(b.tris),
    });
  }

  const masthead = new THREE.Vector3(0, -Infinity, 0);
  const rig = rigMesh as THREE.Mesh | null;
  const rigPos = rig?.geometry.getAttribute('position');
  if (rig && rigPos) {
    const m = new THREE.Matrix4().multiplyMatrices(toModel, rig.matrixWorld);
    const v = new THREE.Vector3();
    for (let i = 0; i < rigPos.count; i++) {
      v.fromBufferAttribute(rigPos, i).applyMatrix4(m);
      if (v.y > masthead.y) masthead.copy(v);
    }
  } else {
    // No rigging: above the tallest sail.
    for (const s of [...courses, ...topsails]) if (s.headL.y > masthead.y) masthead.copy(s.headL).setZ(0);
    masthead.y += 2;
  }

  return { sails: [...courses, ...topsails], sourceMesh: sail, sourceMatrix, masthead };
}

/** Connected islands of an indexed mesh, welding coincident positions. */
function splitIslands(P: THREE.Vector3[], index: THREE.BufferAttribute): Island[] {
  const parent = new Int32Array(P.length);
  for (let i = 0; i < P.length; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let t = 0; t < index.count; t += 3) {
    union(index.getX(t), index.getX(t + 1));
    union(index.getX(t), index.getX(t + 2));
  }
  const weld = new Map<string, number>();
  for (let i = 0; i < P.length; i++) {
    const key = `${P[i].x.toFixed(4)},${P[i].y.toFixed(4)},${P[i].z.toFixed(4)}`;
    const seen = weld.get(key);
    if (seen === undefined) weld.set(key, i);
    else union(i, seen);
  }
  const byRoot = new Map<number, Island>();
  for (let i = 0; i < P.length; i++) {
    const r = find(i);
    let s = byRoot.get(r);
    if (!s) {
      s = { ids: [], min: P[i].clone(), max: P[i].clone(), tris: [] };
      byRoot.set(r, s);
    }
    s.ids.push(i);
    s.min.min(P[i]);
    s.max.max(P[i]);
  }
  for (let t = 0; t < index.count; t += 3) {
    const a = index.getX(t);
    byRoot.get(find(a))!.tris.push(a, index.getX(t + 1), index.getX(t + 2));
  }
  // Tiny islands (stray seams) are not sails.
  return [...byRoot.values()].filter((s) => s.ids.length > 50);
}
