import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import { attribute, float, instanceIndex, mix, select, step, uniformArray, vec3 } from 'three/tsl';
import { CREW_GROUPS, CREW_ORDER, type CrewGroupSpec, type CrewModel, type CrewStation } from './stations';

/**
 * The ship's company on deck: a man at the wheel, a lookout in the bows, hands
 * standing watch at the rails and hauling at the foot of the masts.
 *
 * ### Drawing them
 *
 * Instanced skinning, after three.js r187's `webgpu_skinning_instancing`
 * example (MIT, mrdoob and contributors): a `SkinnedMesh` flagged as instanced
 * and given an `instanceMatrix`, so one skeleton poses every instance and the
 * instance matrix places each one after skinning. One instanced draw per *pose
 * group* (`stations.ts`), each with its own skeleton and clip at its own phase,
 * so the hands hauling at the mainmast are not in step with the ones at the
 * fore. Five groups, so five draws (and their shadow draws) for nine figures.
 *
 * The figures are Quaternius' CC0 "Ultimate Animated Character Pack", baked
 * by Blender to one vertex-coloured material each (`public/models/crew/`).
 * The vertex colour's alpha marks the shirt (1) and the kerchief (0.5), which
 * take a per-instance colour, so one mesh dresses every hand differently.
 * The pack's characters are drawn with heads a third of their height; the head
 * bone is scaled down every frame to give them a grown man's proportions.
 *
 * ### Costs and tiers
 *
 * `QualitySettings.crew` is the number of figures; Low has none. A tier only
 * changes each group's instance count (a group at zero draws nothing and its
 * mixer is not run), so every tier compiles the same one material and nothing
 * recompiles on a tier change. The material is a plain `MeshStandardNodeMaterial`
 * lit through the scene's batched loop (`lighting/BatchedLighting`), so the
 * lanterns light the crew at night. CPU: five mixers of 23 bones.
 *
 * ### Clock
 *
 * The animation time is a clock of its own, advanced by `update(dt)` and
 * rewound by `reset(time)`, and each group's pose is a pure function of it
 * (`phase + time × rate`, wrapped to the clip), so a capture after
 * `resetDeterministic` is the same frame every time. Under reduced motion the
 * clock stands still and the crew hold their pose.
 */

const CREW_URLS: Record<CrewModel, string> = {
  officer: '/models/crew/crew_officer.glb',
  hand: '/models/crew/crew_hand.glb',
};

/** Height of a hand standing, metres, after the head is scaled. */
const STANDING_HEIGHT = 1.72;
/** Head bone scale: the pack's heads are a third of the figure. */
const HEAD_SCALE = 0.55;
/** How far below the measured deck the feet are planted, metres (planks have relief). */
const FOOT_SINK = 0.02;
/** Stations in the largest pose group; sizes the per-instance colour arrays. */
const MAX_GROUP = Math.max(...CREW_GROUPS.map((g) => g.stations.length));

interface Group {
  spec: CrewGroupSpec;
  root: THREE.Object3D;
  mesh: THREE.SkinnedMesh;
  mixer: THREE.AnimationMixer;
  action: THREE.AnimationAction;
  duration: number;
  head: THREE.Object3D | null;
  /** Instances drawn on each tier step; see `setCount`. */
  stations: CrewStation[];
}

interface LoadedModel {
  scene: THREE.Group;
  clips: THREE.AnimationClip[];
}

export class Crew {
  readonly object = new THREE.Group();
  private readonly groups: Group[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly geometries = new Set<THREE.BufferGeometry>();
  private time = 0;
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private readonly onMotionChange = (e: MediaQueryListEvent): void => {
    this.reducedMotion = e.matches;
    this.pose();
  };

  /**
   * Loads both figures and puts the crew aboard `ship` (the buoyant node, +X
   * forward, y up from the waterline, metres). Rejects if a model fails; the
   * caller sails without a crew.
   */
  static async load(ship: THREE.Object3D, hullBeam: number, count: number): Promise<Crew> {
    const { MeshoptDecoder } = await import('three/addons/libs/meshopt_decoder.module.js');
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const [officer, hand] = await Promise.all(
      (['officer', 'hand'] as const).map(async (k) => {
        const gltf = await loader.loadAsync(CREW_URLS[k]);
        return { scene: gltf.scene, clips: gltf.animations } satisfies LoadedModel;
      }),
    );
    return new Crew(ship, hullBeam, { officer, hand }, count);
  }

  private constructor(ship: THREE.Object3D, hullBeam: number, models: Record<CrewModel, LoadedModel>, count: number) {
    this.object.name = 'ship-crew';
    this.motionQuery = typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
    this.reducedMotion = this.motionQuery?.matches ?? false;
    this.motionQuery?.addEventListener?.('change', this.onMotionChange);

    // Scale from the hand's bind pose: neck to crown shrinks with the head.
    const scale = STANDING_HEIGHT / standingHeight(models.hand.scene, HEAD_SCALE);
    ship.updateWorldMatrix(true, false);
    const deck = new DeckProbe(ship);
    const wheel = findWheel(ship);

    for (const spec of CREW_GROUPS) {
      const model = models[spec.model];
      const root = cloneSkinned(model.scene) as THREE.Group;
      root.name = `ship-crew-${spec.id}`;
      let mesh: THREE.SkinnedMesh | null = null;
      root.traverse((n) => {
        if ((n as THREE.SkinnedMesh).isSkinnedMesh && !mesh) mesh = n as THREE.SkinnedMesh;
      });
      if (!mesh) throw new Error(`crew model ${spec.model} has no skinned mesh`);
      const skinned: THREE.SkinnedMesh = mesh;
      this.geometries.add(skinned.geometry);
      const clip = THREE.AnimationClip.findByName(model.clips, spec.clip) ?? model.clips[0];
      const mixer = new THREE.AnimationMixer(root);
      const action = mixer.clipAction(clip);
      action.play();

      const stations = spec.stations.map((s) => resolveStation(s, hullBeam, wheel));
      this.object.add(root);
      root.updateMatrixWorld(true);
      // Where the skinned mesh sits within its own glTF root; the instance
      // matrix is applied in the mesh's frame, after skinning.
      const meshInRoot = skinned.matrixWorld.clone().premultiply(_m.copy(root.matrixWorld).invert());
      const meshInRootInv = meshInRoot.clone().invert();
      const n = stations.length;
      const instanceMatrix = new THREE.InstancedBufferAttribute(new Float32Array(n * 16), 16);
      // Padded to the largest group so every group generates the same shader.
      const shirts = Array.from({ length: MAX_GROUP }, () => new THREE.Color(1, 1, 1));
      const caps = Array.from({ length: MAX_GROUP }, () => new THREE.Color(1, 1, 1));
      stations.forEach((st, i) => {
        const y = deck.heightAt(st.x, st.z, st.y) - FOOT_SINK;
        _q.setFromAxisAngle(_up, Math.PI / 2 - st.facing);
        _m.compose(_p.set(st.x, y, st.z), _q, _s.setScalar(scale));
        _m.premultiply(meshInRootInv).multiply(meshInRoot);
        _m.toArray(instanceMatrix.array, i * 16);
        const shirtColor = _c.setHex(st.shirt ?? 0xffffff).convertSRGBToLinear();
        shirts[i].copy(shirtColor);
        const capColor = _c.setHex(st.kerchief ?? 0xffffff).convertSRGBToLinear();
        caps[i].copy(capColor);
      });

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
      skinned.material = this.makeMaterial(spec.id, shirts, caps);
      skinned.castShadow = true;
      skinned.receiveShadow = true;
      // Instances spread over the deck; the bind-pose bounds cover one figure.
      skinned.frustumCulled = false;

      this.groups.push({
        spec,
        root,
        mesh: skinned,
        mixer,
        action,
        duration: clip.duration,
        head: root.getObjectByName('Head') ?? null,
        stations,
      });
    }
    ship.add(this.object);
    this.setCount(count);
    this.pose();
  }

  /**
   * One material per group, for its own per-instance colours. They generate
   * the same shader, so the five share one pipeline.
   */
  private makeMaterial(id: string, shirts: THREE.Color[], caps: THREE.Color[]): THREE.Material {
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.88, metalness: 0 });
    material.name = `ship-crew-${id}`;
    // Per-instance shirt and kerchief, chosen by the vertex colour's alpha.
    const vc = attribute<'vec4'>('color', 'vec4');
    const shirt = vec3(uniformArray(shirts, 'color').element(instanceIndex) as never);
    const cap = vec3(uniformArray(caps, 'color').element(instanceIndex) as never);
    const tint = select(vc.w.greaterThan(0.75), shirt, mix(vec3(1), cap, step(float(0.25), vc.w)));
    material.colorNode = vc.xyz.mul(tint);
    this.materials.push(material);
    return material;
  }

  /**
   * Figures aboard, from the tier (`QualitySettings.crew`), taken in
   * `CREW_ORDER`: the helmsman and the lookout first. Only instance counts
   * move; a group at zero draws nothing and is not posed.
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

  private pose(): void {
    const t = this.reducedMotion ? 0 : this.time;
    for (const g of this.groups) {
      if ((g.mesh as unknown as { count: number }).count === 0) continue;
      const local = g.spec.phase + t * g.spec.rate;
      g.action.time = ((local % g.duration) + g.duration) % g.duration;
      g.mixer.update(0);
      g.head?.scale.setScalar(HEAD_SCALE);
    }
  }

  dispose(): void {
    this.motionQuery?.removeEventListener?.('change', this.onMotionChange);
    this.object.removeFromParent();
    for (const g of this.groups) g.mixer.stopAllAction();
    for (const geometry of this.geometries) geometry.dispose();
    for (const m of this.materials) m.dispose();
  }
}

/** A station in the ship's frame, metres, with the helm placed from the wheel if the model has one. */
interface ResolvedStation extends CrewStation {
  x: number;
  z: number;
  /** Ray start height for the deck probe. */
  y: number;
}

function resolveStation(s: CrewStation, hullBeam: number, wheel: THREE.Vector3 | null): ResolvedStation {
  if (s.atWheel && wheel) {
    // Behind the wheel, facing forward with the spokes in his hands.
    return { ...s, x: wheel.x - 0.55, z: wheel.z, y: wheel.y + 2, facing: 0 };
  }
  return { ...s, x: s.x, z: s.zBeam * hullBeam * 0.5, y: s.probeFrom ?? 12 };
}

/** The wheel's position in the ship's frame, if the model carries it as a node. */
function findWheel(ship: THREE.Object3D): THREE.Vector3 | null {
  let found: THREE.Object3D | null = null;
  ship.traverse((n) => {
    if (!found && /wheel/i.test(n.name)) found = n;
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

/**
 * Height of the figure standing in its bind pose with the head bone scaled by
 * `headScale` about its own origin, in the model's units.
 */
function standingHeight(scene: THREE.Object3D, headScale: number): number {
  scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(scene, true);
  const head = scene.getObjectByName('Head');
  if (!head) return box.max.y - box.min.y;
  const neck = head.getWorldPosition(new THREE.Vector3()).y;
  return neck - box.min.y + (box.max.y - neck) * headScale;
}

const _up = /*@__PURE__*/ new THREE.Vector3(0, 1, 0);
const _p = /*@__PURE__*/ new THREE.Vector3();
const _s = /*@__PURE__*/ new THREE.Vector3();
const _o = /*@__PURE__*/ new THREE.Vector3();
const _d = /*@__PURE__*/ new THREE.Vector3();
const _n = /*@__PURE__*/ new THREE.Vector3();
const _q = /*@__PURE__*/ new THREE.Quaternion();
const _m = /*@__PURE__*/ new THREE.Matrix4();
const _inv = /*@__PURE__*/ new THREE.Matrix4();
const _c = /*@__PURE__*/ new THREE.Color();
