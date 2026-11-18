import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import {
  attribute,
  exp2,
  float,
  fract,
  instanceIndex,
  mix,
  positionGeometry,
  positionLocal,
  select,
  step,
  texture,
  uint,
  uniformArray,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import { CREW_GROUPS, CREW_ORDER, type CrewGroupSpec, type CrewLook, type CrewStation } from './stations';

/**
 * The ship's company on deck: a captain on the quarterdeck, a helmsman at the
 * whipstaff, a bosun pacing the waist, hands hauling at the masts and standing
 * watch at the rails, and a lookout in the bows.
 *
 * ### The figures
 *
 * Chunky toy pirates: KayKit's CC0 "Adventurers" (Kay Lousberg) taken apart
 * and re-dressed by `scripts/modelkit/crew-toy.py` into one skinned mesh that
 * holds every head, body, pair of arms and legs, and the pirate gear (tricorn,
 * Monmouth cap, kerchief, eye patch, sash). A figure's look (`stations.ts`)
 * says which parts it wears; the vertex shader collapses the rest to a point,
 * and each group's index buffer only lists the parts its figures wear, so a
 * figure costs the triangles it shows. The vertex colour carries each
 * corner's part (G) and tint slot (R); tinted faces sample a neutral ramp in
 * the atlas and take the figure's own shirt, coat, breeches, headwear, sash,
 * skin or hair colour. Glossy, like moulded plastic, but lit by the same
 * sun, sky and lanterns as the ship.
 *
 * ### Drawing them
 *
 * Instanced skinning, after three.js r187's `webgpu_skinning_instancing`
 * example (MIT, mrdoob and contributors): a `SkinnedMesh` flagged as instanced
 * and given an `instanceMatrix`, so one skeleton poses every instance and the
 * instance matrix places each one after skinning. One instanced draw per *pose
 * group* (`stations.ts`), each with its own skeleton, clip, phase and motion.
 * Five groups, so five draws (and their shadow draws) for nine figures.
 *
 * All per-figure data (part mask, stripes, seven colours) is one uniform
 * array per group, read in both stages: the lit fragment stage sits at its
 * limit of twelve uniform buffers with one, as the previous crew did.
 *
 * ### Costs and tiers
 *
 * `QualitySettings.crew` is the number of figures; Low has none. A tier only
 * changes each group's instance count (a group at zero draws nothing and is
 * not posed), so every tier compiles the same one material and nothing
 * recompiles on a tier change. CPU: five mixers of 23 bones and a handful of
 * bone nudges.
 *
 * ### Clock
 *
 * The animation time is a clock of its own, advanced by `update(dt)` and
 * rewound by `reset(time)`, and every pose and every walker's place is a pure
 * function of it, so a capture after `resetDeterministic` is the same frame
 * every time. Under reduced motion the clock stands still and the crew hold
 * their pose.
 */

const CREW_URL = '/models/crew/crew_toy.glb';

/** Height of a standard figure standing, metres, after the head is scaled. */
const STANDING_HEIGHT = 1.6;
/** Head bone scale: the pack's heads are nearly half the figure; a toy's are a third. */
const HEAD_SCALE = 0.8;
/** How far below the measured deck the feet are planted, metres (planks have relief). */
const FOOT_SINK = 0.02;
/** Stations in the largest pose group; sizes the per-instance arrays. */
const MAX_GROUP = Math.max(...CREW_GROUPS.map((g) => g.stations.length));
/** vec4s per figure in the group's uniform array: flags, then slots 1..7. */
const STRIDE = 8;
/** Seconds a walker takes to turn about at the end of his beat. */
const TURN_TIME = 1.1;

interface Group {
  spec: CrewGroupSpec;
  root: THREE.Object3D;
  mesh: THREE.SkinnedMesh;
  mixer: THREE.AnimationMixer;
  action: THREE.AnimationAction;
  duration: number;
  bones: Record<string, THREE.Bone | undefined>;
  /**
   * The layered bones' rotations as the clip left them. The mixer only writes
   * a bone whose clip value changed since it last wrote it, so without putting
   * these back first a still channel would keep, and pile up, our own turns.
   */
  clipPose: Map<THREE.Object3D, THREE.Quaternion>;
  stations: ResolvedStation[];
  instanceMatrix: THREE.InstancedBufferAttribute;
  meshInRoot: THREE.Matrix4;
  meshInRootInv: THREE.Matrix4;
  /** Walkers: metres per second along the deck at rate 1, from the clip's stride. */
  walkSpeed: number;
}

export class Crew {
  readonly object = new THREE.Group();
  private readonly groups: Group[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly geometries = new Set<THREE.BufferGeometry>();
  private readonly scale: number;
  private readonly deck: DeckProbe;
  private time = 0;
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private readonly onMotionChange = (e: MediaQueryListEvent): void => {
    this.reducedMotion = e.matches;
    this.pose();
  };

  /**
   * Loads the figures and puts the crew aboard `ship` (the buoyant node, +X
   * forward, y up from the waterline, metres). Rejects if the model fails; the
   * caller sails without a crew.
   */
  static async load(ship: THREE.Object3D, hullBeam: number, count: number): Promise<Crew> {
    const { MeshoptDecoder } = await import('three/addons/libs/meshopt_decoder.module.js');
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(CREW_URL);
    return new Crew(ship, hullBeam, gltf.scene, gltf.animations, count);
  }

  private constructor(
    ship: THREE.Object3D,
    hullBeam: number,
    model: THREE.Group,
    clips: THREE.AnimationClip[],
    count: number,
  ) {
    this.object.name = 'ship-crew';
    this.motionQuery = typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
    this.reducedMotion = this.motionQuery?.matches ?? false;
    this.motionQuery?.addEventListener?.('change', this.onMotionChange);

    let source: THREE.SkinnedMesh | null = null;
    model.traverse((n) => {
      if ((n as THREE.SkinnedMesh).isSkinnedMesh && !source) source = n as THREE.SkinnedMesh;
    });
    if (!source) throw new Error('crew model has no skinned mesh');
    const sourceMesh: THREE.SkinnedMesh = source;
    const atlas = (sourceMesh.material as THREE.MeshStandardMaterial).map;
    if (!atlas) throw new Error('crew model has no atlas');
    // Flat ramps in 64 px cells: mip levels would only bleed neighbouring cells together.
    atlas.generateMipmaps = false;
    atlas.minFilter = THREE.LinearFilter;
    const partOf = cornerParts(sourceMesh.geometry);

    this.scale = STANDING_HEIGHT / standingHeight(model, HEAD_SCALE);
    ship.updateWorldMatrix(true, false);
    this.deck = new DeckProbe(ship);
    const wheel = findWheel(ship);

    for (const spec of CREW_GROUPS) {
      const root = cloneSkinned(model) as THREE.Group;
      root.name = `ship-crew-${spec.id}`;
      let mesh: THREE.SkinnedMesh | null = null;
      root.traverse((n) => {
        if ((n as THREE.SkinnedMesh).isSkinnedMesh && !mesh) mesh = n as THREE.SkinnedMesh;
      });
      if (!mesh) throw new Error('crew clone has no skinned mesh');
      const skinned: THREE.SkinnedMesh = mesh;
      const stations = spec.stations.map((s) => resolveStation(s, hullBeam, wheel, this.deck));

      // Only the parts this group's figures wear.
      let union = 0;
      for (const st of stations) for (const p of st.look.parts) union |= 1 << p;
      skinned.geometry = subsetByParts(sourceMesh.geometry, partOf, union);
      this.geometries.add(skinned.geometry);

      const clip = THREE.AnimationClip.findByName(clips, spec.clip) ?? clips[0];
      const mixer = new THREE.AnimationMixer(root);
      const action = mixer.clipAction(clip);
      action.play();

      this.object.add(root);
      root.updateMatrixWorld(true);
      // Where the skinned mesh sits within its own glTF root; the instance
      // matrix is applied in the mesh's frame, after skinning.
      const meshInRoot = skinned.matrixWorld.clone().premultiply(_m.copy(root.matrixWorld).invert());
      const meshInRootInv = meshInRoot.clone().invert();
      const instanceMatrix = new THREE.InstancedBufferAttribute(new Float32Array(stations.length * 16), 16);
      instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      // `webgpu_skinning_instancing`'s hack: the node material reads these.
      const instanced = skinned as unknown as {
        isInstancedMesh: boolean;
        instanceMatrix: THREE.InstancedBufferAttribute;
        instanceColor: null;
        count: number;
      };
      instanced.isInstancedMesh = true;
      instanced.instanceMatrix = instanceMatrix;
      instanced.instanceColor = null;
      instanced.count = stations.length;
      skinned.material = this.makeMaterial(spec.id, atlas, stations.map((s) => s.look));
      skinned.castShadow = true;
      skinned.receiveShadow = true;
      // Instances spread over the deck; the bind-pose bounds cover one figure.
      skinned.frustumCulled = false;

      const bones: Record<string, THREE.Bone | undefined> = {};
      // GLTFLoader drops the dots from node names ('upperarm.l' is 'upperarml').
      for (const name of ['hips', 'spine', 'chest', 'head', 'upperarm.l', 'upperarm.r', 'lowerarm.l', 'lowerarm.r']) {
        bones[name.replace('.', '_')] = (root.getObjectByName(name.replace('.', ''))
          ?? root.getObjectByName(name)) as THREE.Bone | undefined;
      }
      const group: Group = {
        spec,
        root,
        mesh: skinned,
        mixer,
        action,
        duration: clip.duration,
        bones,
        clipPose: new Map(
          Object.values(bones).filter((b): b is THREE.Bone => !!b).map((b) => [b, b.quaternion.clone()]),
        ),
        stations,
        instanceMatrix,
        meshInRoot,
        meshInRootInv,
        walkSpeed: spec.motion === 'walk' ? strideSpeed(root, mixer, action, clip.duration) * this.scale : 0,
      };
      // The bones now hold what the mixer last wrote (`strideSpeed` ran it).
      mixer.update(0);
      for (const [bone, q] of group.clipPose) q.copy(bone.quaternion);
      this.groups.push(group);
      this.place(group, 0);
    }
    ship.add(this.object);
    this.setCount(count);
    this.pose();
  }

  /**
   * One material per group, for its own per-figure data. They generate the
   * same shader, so the five share one pipeline.
   */
  private makeMaterial(id: string, atlas: THREE.Texture, looks: CrewLook[]): THREE.Material {
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.42, metalness: 0 });
    material.name = `ship-crew-${id}`;
    const data = Array.from({ length: MAX_GROUP * STRIDE }, () => new THREE.Vector4(1, 1, 1, 1));
    looks.forEach((look, i) => {
      let mask = 0;
      for (const p of look.parts) mask |= 1 << p;
      data[i * STRIDE].set(mask & 0xfff, mask >>> 12, look.stripes ? 1 : 0, 0);
      const slots = [look.shirt, look.coat ?? look.shirt, look.breeches, look.hat ?? look.hair, look.sash ?? look.shirt, look.skin, look.hair];
      slots.forEach((hex, k) => {
        _c.setHex(hex).convertSRGBToLinear();
        data[i * STRIDE + 1 + k].set(_c.r, _c.g, _c.b, 1);
      });
    });
    // One array for both stages: the lit fragment stage is one buffer short of its limit of twelve.
    const figure = uniformArray(data, 'vec4');
    const base = instanceIndex.mul(STRIDE);
    const vc = attribute<'vec4'>('color', 'vec4');
    const flags = vec4(figure.element(base) as never);

    // Vertex: parts this figure does not wear collapse to a point.
    const part = vc.y.mul(255).round();
    const high = part.greaterThanEqual(12);
    const bits = select(high, flags.y, flags.x);
    const bit = select(high, part.sub(12), part);
    const worn = bits.div(exp2(bit)).floor().mod(2);
    material.positionNode = positionLocal.mul(worn);

    // Fragment: the atlas, and on tinted faces the figure's own colour over a neutral ramp.
    const slot = vc.x.mul(255).round();
    const tint = vec3(figure.element(base.add(uint(slot))) as never);
    const band = step(float(0.5), fract(positionGeometry.y.mul(9)));
    const striped = flags.z.mul(step(float(0.5), slot)).mul(step(slot, float(1.5)));
    const cloth = mix(tint, vec3(0.86, 0.82, 0.72), band.mul(striped));
    const atlasColour = texture(atlas, uv()).rgb;
    material.colorNode = select(slot.greaterThan(0.5), atlasColour.mul(cloth), atlasColour);
    this.materials.push(material);
    return material;
  }

  /**
   * Figures aboard, from the tier (`QualitySettings.crew`), taken in
   * `CREW_ORDER`. Only instance counts move; a group at zero draws nothing
   * and is not posed.
   */
  setCount(count: number): void {
    const n = Math.max(0, Math.min(CREW_ORDER.length, Math.round(count)));
    this.groups.forEach((g, i) => {
      let k = 0;
      for (let j = 0; j < n; j++) if (CREW_ORDER[j] === i) k++;
      (g.mesh as unknown as { count: number }).count = Math.min(k, g.stations.length);
    });
    this.pose();
  }

  /** Rewinds the crew's clock, for deterministic capture. */
  reset(time = 0): void {
    this.time = time;
    this.pose();
  }

  update(dt: number): void {
    if (!this.reducedMotion) this.time += Math.max(0, dt);
    this.pose();
  }

  /** Figures drawn now, with their roles; for tests and captions. */
  roster(): string[] {
    const out: string[] = [];
    for (const g of this.groups) {
      const n = (g.mesh as unknown as { count: number }).count;
      for (let i = 0; i < n; i++) out.push(g.stations[i].look.role);
    }
    return out;
  }

  private pose(): void {
    const t = this.reducedMotion ? 0 : this.time;
    for (const g of this.groups) {
      if ((g.mesh as unknown as { count: number }).count === 0) continue;
      const local = g.spec.phase + t * g.spec.rate;
      g.action.time = ((local % g.duration) + g.duration) % g.duration;
      for (const [bone, q] of g.clipPose) bone.quaternion.copy(q);
      g.mixer.update(0);
      for (const [bone, q] of g.clipPose) q.copy(bone.quaternion);
      g.bones.head?.scale.setScalar(HEAD_SCALE);
      this.layer(g, local);
      if (g.spec.motion === 'walk') this.place(g, t);
    }
  }

  /**
   * The motion on top of each group's clip, as rotations about the figure's
   * own axes (x to its left, y up, z ahead), so it reads the same whatever
   * the bones' local frames are. `s` is the group's local time, seconds.
   */
  private layer(g: Group, s: number): void {
    const b = g.bones;
    const ph = g.spec.phase * 1.7;
    // Everyone looks about now and then.
    const look = 0.42 * Math.sin(s * 0.31 + ph) * smoothPulse(Math.sin(s * 0.23 + ph * 2)) + 0.12 * Math.sin(s * 0.9 + ph);
    switch (g.spec.motion) {
      case 'idle': {
        // Weight from foot to foot, the chest answering, and a look about.
        const sway = Math.sin(s * 0.9 + ph);
        turn(g.root, b.hips, _z, 0.07 * sway);
        turn(g.root, b.chest, _z, -0.05 * sway);
        turn(g.root, b.head, _y, look);
        turn(g.root, b.head, _x, 0.08 * Math.sin(s * 0.47 + ph));
        break;
      }
      case 'helm': {
        // Both hands on the whipstaff, leaning into it as the ship answers.
        const lean = Math.sin(s * 0.45);
        turn(g.root, b.hips, _y, 0.16 * lean);
        turn(g.root, b.spine, _x, 0.12);
        turn(g.root, b.chest, _z, -0.08 * lean);
        turn(g.root, b.head, _y, 0.5 * look - 0.1 * lean);
        turn(g.root, b.head, _x, -0.18 * smoothPulse(Math.sin(s * 0.19 + 1)));
        break;
      }
      case 'haul': {
        // Hand over hand on a fall: reach up (0 to 0.4), heave down and back (to 0.75), recover.
        const c = fract01(s / 1.6);
        const reach = c < 0.4 ? ease(c / 0.4) : c < 0.75 ? 1 - ease((c - 0.4) / 0.35) : 0;
        const heave = c >= 0.4 && c < 0.75 ? Math.sin(((c - 0.4) / 0.35) * Math.PI) : 0;
        turn(g.root, b.upperarm_l, _x, -1.35 * reach);
        turn(g.root, b.upperarm_r, _x, -1.35 * reach);
        turn(g.root, b.lowerarm_l, _x, 0.35 * reach);
        turn(g.root, b.lowerarm_r, _x, 0.35 * reach);
        turn(g.root, b.spine, _x, 0.22 * reach - 0.3 * heave);
        turn(g.root, b.hips, _x, -0.12 * heave);
        turn(g.root, b.head, _x, -0.2 * reach + 0.1 * heave);
        break;
      }
      case 'walk': {
        turn(g.root, b.head, _y, 0.6 * look);
        break;
      }
    }
  }

  /** Writes the group's instance matrices; walkers from the clock. */
  private place(g: Group, t: number): void {
    g.stations.forEach((st, i) => {
      let x = st.x;
      let facing = st.facing;
      let y = st.deckY;
      if (st.walkTo !== undefined && st.heights) {
        const len = Math.abs(st.walkTo - st.x);
        const speed = Math.max(0.2, g.walkSpeed * g.spec.rate);
        const leg = len / speed;
        const period = 2 * (leg + TURN_TIME);
        const u = fract01((t + g.spec.phase) / period) * period;
        const dir = Math.sign(st.walkTo - st.x) || 1;
        if (u < leg) {
          x = st.x + dir * u * speed;
          facing = dir > 0 ? 0 : Math.PI;
        } else if (u < leg + TURN_TIME) {
          x = st.walkTo;
          facing = (dir > 0 ? 0 : Math.PI) + Math.PI * ease((u - leg) / TURN_TIME);
        } else if (u < 2 * leg + TURN_TIME) {
          x = st.walkTo - dir * (u - leg - TURN_TIME) * speed;
          facing = dir > 0 ? Math.PI : 0;
        } else {
          x = st.x;
          facing = (dir > 0 ? Math.PI : 0) + Math.PI * ease((u - 2 * leg - TURN_TIME) / TURN_TIME);
        }
        y = sampleHeights(st.heights, (x - Math.min(st.x, st.walkTo)) / HEIGHT_STEP);
      }
      const [w, h] = st.look.build ?? [1, 1];
      _q.setFromAxisAngle(_up, Math.PI / 2 - facing);
      _m.compose(_p.set(x, y - FOOT_SINK, st.z), _q, _s.set(w * this.scale, h * this.scale, w * this.scale));
      _m.premultiply(g.meshInRootInv).multiply(g.meshInRoot);
      _m.toArray(g.instanceMatrix.array, i * 16);
    });
    g.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.motionQuery?.removeEventListener?.('change', this.onMotionChange);
    this.object.removeFromParent();
    for (const g of this.groups) g.mixer.stopAllAction();
    for (const geometry of this.geometries) geometry.dispose();
    for (const m of this.materials) m.dispose();
  }
}

/** Spacing of a walker's deck-height samples, metres. */
const HEIGHT_STEP = 0.25;

/** A station in the ship's frame, metres, with the deck under it measured. */
interface ResolvedStation extends CrewStation {
  x: number;
  z: number;
  deckY: number;
  /** Walkers: the deck along the beat, every `HEIGHT_STEP` from the aft end. */
  heights?: number[];
}

function resolveStation(s: CrewStation, hullBeam: number, wheel: THREE.Vector3 | null, deck: DeckProbe): ResolvedStation {
  if (s.atWheel && wheel) {
    // Abaft the whipstaff, facing forward with both hands on it.
    const x = wheel.x - 0.62;
    return { ...s, x, z: wheel.z, facing: 0, deckY: deck.heightAt(x, wheel.z, wheel.y + 2) };
  }
  const z = s.zBeam * hullBeam * 0.5;
  const from = s.probeFrom ?? 12;
  const r: ResolvedStation = { ...s, z, deckY: deck.heightAt(s.x, z, from) };
  if (s.walkTo !== undefined) {
    const x0 = Math.min(s.x, s.walkTo);
    const n = Math.ceil(Math.abs(s.walkTo - s.x) / HEIGHT_STEP) + 1;
    r.heights = Array.from({ length: n }, (_, i) => deck.heightAt(x0 + i * HEIGHT_STEP, z, from));
  }
  return r;
}

function sampleHeights(h: number[], f: number): number {
  const i = Math.max(0, Math.min(h.length - 1, Math.floor(f)));
  const j = Math.min(h.length - 1, i + 1);
  const t = Math.max(0, Math.min(1, f - i));
  return h[i] + (h[j] - h[i]) * t;
}

/** The helm's position in the ship's frame (a wheel, or this ship's whipstaff), if the model carries it as a node. */
function findWheel(ship: THREE.Object3D): THREE.Vector3 | null {
  let found: THREE.Object3D | null = null;
  ship.traverse((n) => {
    if (!found && /wheel|whipstaff/i.test(n.name)) found = n;
  });
  if (!found) return null;
  const box = new THREE.Box3().setFromObject(found, true);
  if (box.isEmpty()) return null;
  const centre = box.getCenter(new THREE.Vector3());
  const bottom = new THREE.Vector3(centre.x, box.min.y, centre.z);
  return ship.worldToLocal(bottom);
}

/**
 * The deck under a point, by a ray down onto the hull. Measured once, at load;
 * the sails, the rigging and anything of ours are not deck.
 */
class DeckProbe {
  private readonly raycaster = new THREE.Raycaster();
  private readonly targets: THREE.Object3D[] = [];

  constructor(private readonly ship: THREE.Object3D) {
    ship.traverse((n) => {
      const mesh = n as THREE.Mesh;
      if (!mesh.isMesh || (n as THREE.SkinnedMesh).isSkinnedMesh) return;
      const name = `${mesh.name} ${mesh.parent?.name ?? ''}`.toLowerCase();
      if (/sail|rig|rope|crew|lantern|cable|probe|pennant|flag|window|halo|glow/.test(name)) return;
      this.targets.push(mesh);
    });
  }

  /** Deck height at (x, z), searching down from `fromY`; 2.6 m if nothing is hit. */
  heightAt(x: number, z: number, fromY: number): number {
    const origin = this.ship.localToWorld(_o.set(x, fromY, z));
    const down = _d.set(0, -1, 0).transformDirection(this.ship.matrixWorld);
    this.raycaster.set(origin, down);
    this.raycaster.far = fromY + 2;
    const hits = this.raycaster.intersectObjects(this.targets, false);
    for (const hit of hits) {
      if (!hit.face) continue;
      const n = _n.copy(hit.face.normal).transformDirection(hit.object.matrixWorld);
      _inv.copy(this.ship.matrixWorld).invert();
      n.transformDirection(_inv);
      if (n.y < 0.7) continue;
      return this.ship.worldToLocal(hit.point.clone()).y;
    }
    return 2.6;
  }
}

/** Part bit of each vertex, from the vertex colour's G. */
function cornerParts(geometry: THREE.BufferGeometry): Uint8Array {
  const col = geometry.getAttribute('color');
  const out = new Uint8Array(col.count);
  for (let i = 0; i < col.count; i++) out[i] = Math.round(col.getY(i) * 255);
  return out;
}

/** A geometry sharing `source`'s attributes whose index lists only triangles of the parts in `mask`. */
function subsetByParts(source: THREE.BufferGeometry, partOf: Uint8Array, mask: number): THREE.BufferGeometry {
  const index = source.getIndex();
  if (!index) throw new Error('crew mesh is not indexed');
  const keep: number[] = [];
  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i);
    if (mask & (1 << partOf[a])) keep.push(a, index.getX(i + 1), index.getX(i + 2));
  }
  const geometry = new THREE.BufferGeometry();
  for (const [name, attr] of Object.entries(source.attributes)) geometry.setAttribute(name, attr);
  const vertexCount = source.getAttribute('position').count;
  geometry.setIndex(vertexCount > 65535 ? new THREE.Uint32BufferAttribute(keep, 1) : new THREE.Uint16BufferAttribute(keep, 1));
  return geometry;
}

/**
 * Height of the figure standing in its bind pose with the head bone scaled by
 * `headScale` about its own origin, in the model's units.
 */
function standingHeight(scene: THREE.Object3D, headScale: number): number {
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(scene, true);
  const head = scene.getObjectByName('head');
  if (!head) return box.max.y - box.min.y;
  const neck = head.getWorldPosition(new THREE.Vector3()).y;
  // The hats stand above the hair; measure to the crown of a bare head.
  const crown = Math.min(box.max.y, neck + 0.95);
  return neck - box.min.y + (crown - neck) * headScale;
}

/**
 * Ground speed of an in-place walk clip at rate 1, model units per second:
 * the distance a planted foot travels back under the body in a stride.
 */
function strideSpeed(root: THREE.Object3D, mixer: THREE.AnimationMixer, action: THREE.AnimationAction, duration: number): number {
  const foot = root.getObjectByName('footl') ?? root.getObjectByName('foot.l');
  if (!foot) return 1;
  let lo = Infinity;
  let hi = -Infinity;
  const steps = 32;
  for (let i = 0; i < steps; i++) {
    action.time = (duration * i) / steps;
    mixer.update(0);
    root.updateMatrixWorld(true);
    const z = foot.getWorldPosition(_o).applyMatrix4(_inv.copy(root.matrixWorld).invert()).z;
    lo = Math.min(lo, z);
    hi = Math.max(hi, z);
  }
  return ((hi - lo) * 2) / duration;
}

/**
 * Rotates `bone` by `angle` about `axis`, given in `root`'s frame (the
 * figure's own: x left, y up, z ahead), after the clip has posed it.
 */
function turn(root: THREE.Object3D, bone: THREE.Object3D | undefined, axis: THREE.Vector3, angle: number): void {
  if (!bone || !bone.parent || angle === 0) return;
  bone.parent.updateWorldMatrix(true, false);
  root.getWorldQuaternion(_qr);
  bone.parent.getWorldQuaternion(_qp);
  // The parent's frame relative to the root's.
  _qp.premultiply(_qr.invert());
  _ql.setFromAxisAngle(axis, angle);
  // Axis in the parent's frame: qp^-1 * axisRotation * qp.
  _ql.premultiply(_qp.clone().invert()).multiply(_qp);
  bone.quaternion.premultiply(_ql);
}

function fract01(x: number): number {
  return x - Math.floor(x);
}

function ease(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return c * c * (3 - 2 * c);
}

/** 0..1 from a sine, held near the top: a look that lingers. */
function smoothPulse(s: number): number {
  return ease(s * 0.8 + 0.5);
}

const _up = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const _x = /*@__PURE__*/ new THREE.Vector3(1, 0, 0);
const _y = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const _z = /*@__PURE__*/ new THREE.Vector3(0, 0, 1);
const _p = /*@__PURE__*/ new THREE.Vector3();
const _s = /*@__PURE__*/ new THREE.Vector3();
const _o = /*@__PURE__*/ new THREE.Vector3();
const _d = /*@__PURE__*/ new THREE.Vector3();
const _n = /*@__PURE__*/ new THREE.Vector3();
const _q = /*@__PURE__*/ new THREE.Quaternion();
const _qr = /*@__PURE__*/ new THREE.Quaternion();
const _qp = /*@__PURE__*/ new THREE.Quaternion();
const _ql = /*@__PURE__*/ new THREE.Quaternion();
const _m = /*@__PURE__*/ new THREE.Matrix4();
const _inv = /*@__PURE__*/ new THREE.Matrix4();
const _c = /*@__PURE__*/ new THREE.Color();
