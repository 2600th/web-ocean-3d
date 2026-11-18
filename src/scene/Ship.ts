import * as THREE from 'three/webgpu';
import { abs, dot, float, materialColor, mix, mx_noise_float, positionWorld, smoothstep, uniform, vec3, vec4 } from 'three/tsl';
import { HERO_ANISOTROPY, type AssetLoader } from './AssetLoader';
import { ShipSails, type SailStateInput } from './ship/Sails';

/**
 * The hero sailing ship.
 *
 * The Poly Haven model arrives in its own units and its own frame: the hull runs
 * along +X from x = -10.33 (transom) to x = +11.63 (stem), the bowsprit reaches
 * further still, and the origin sits somewhere inside the hull rather than at
 * the waterline. None of that is usable directly, so `load()` normalises it:
 *
 *   - uniform scale so the hull is `TARGET_HULL_LENGTH` metres stem to stern;
 *   - recentred on the hull's horizontal centroid;
 *   - shifted vertically so the design waterline is exactly y = 0.
 *
 * After normalisation `object` is a plain `Object3D` whose origin is the point
 * buoyancy drives, and whose **forward direction is +X** (see `forwardLocal`).
 * That is the model's native heading; rotating it to some other convention would
 * mean carrying a correction into every probe point and every wake emission for
 * no gain.
 */

/**
 * Built by `scripts/optimize-assets.mjs` from Poly Haven's 2k glTF: KTX2 at 1k
 * and Meshopt geometry, 7 MB against the 27 MB of JPEGs it replaces.
 */
export const SHIP_URL = '/models/hero/dutch_ship_medium.glb';

/** Stem-to-transom length of the hull mesh, in metres, after normalisation. */
const TARGET_HULL_LENGTH = 27;

/**
 * Where the waterline sits within the hull's vertical extent, 0 = keel,
 * 1 = top of the hull mesh. A little under a third puts the wales just clear of
 * the water, which is how the reference sits.
 */
const WATERLINE_FRACTION = 0.31;

/** Probe inset from the hull extremities, as a fraction of the half-extent. */
const PROBE_LONGITUDINAL = 0.82;
const PROBE_LATERAL = 0.86;

/**
 * The helm's node in the model. This ship steers with a whipstaff, not a
 * wheel: a vertical lever through the quarterdeck whose foot works the tiller,
 * which is what the model carries and what a Dutch ship of about 1700 had (the
 * wheel came into the Royal Navy around 1703-1710 and spread slowly). The node
 * is the pivot where the staff passes through the deck; see `helmStation`.
 * Split out by `scripts/modelkit/split-hero.mjs`, as are the rudder and yards.
 */
export const HELM_NODE = 'whipstaff';
export const RUDDER_NODE = 'rudder';
/** Rudder hard over, radians. A ship's rudder stalls past about 35 degrees. */
const RUDDER_MAX = THREE.MathUtils.degToRad(30);
/**
 * Whipstaff hard over, radians. The staff leans toward the turn: its head to
 * starboard puts its foot, the tiller, to port, and the blade to starboard.
 */
const WHIPSTAFF_MAX = THREE.MathUtils.degToRad(22);

export class Ship {
  readonly object: THREE.Object3D;

  /** Bow, stern, port, starboard — local space, at the waterline plane. */
  readonly probePoints: THREE.Vector3[];

  readonly hullLength: number;
  readonly hullBeam: number;
  /** Highest point of the hull above the waterline (the stern castle), metres. */
  readonly hullFreeboard: number;

  /** Unit vector, in `object` space, that the bow points along. */
  readonly forwardLocal = new THREE.Vector3(1, 0, 0);

  /**
   * The helm (a whipstaff, `HELM_NODE`), for placing the helmsman. Null if the
   * model was not split. It swings athwartships with the rudder; its rest pose
   * is upright.
   */
  readonly helmObject: THREE.Object3D | null;
  /**
   * Where the helmsman stands, ship frame (`object` space, metres): on the
   * quarterdeck planking, 0.55 m abaft the staff, facing the bow (+X). Zero
   * if the model was not split.
   */
  readonly helmStation = new THREE.Vector3();
  /** Head of the whipstaff at rest, ship frame, metres: where the hands go. */
  readonly helmGrip = new THREE.Vector3();

  private readonly model: THREE.Group;
  private readonly rudderObject: THREE.Object3D | null;
  private readonly rudderAxis = new THREE.Vector3(0, 1, 0);
  private readonly helmAxis = new THREE.Vector3(1, 0, 0);
  /** ShipController's rudder, -1 hard to port .. 1 hard to starboard. */
  private helm = 0;
  /** World to ship frame, for the bottom treatment. */
  private readonly uShipInverse = uniform(new THREE.Matrix4());
  private readonly probeGroup: THREE.Group;
  private readonly probeGeometry: THREE.SphereGeometry;
  private readonly probeMaterial: THREE.MeshBasicNodeMaterial;
  /** Canvas, cloth and pennant; see `ship/Sails`. */
  private readonly sails: ShipSails;
  private disposed = false;

  private constructor(model: THREE.Group, hullBox: THREE.Box3) {
    this.model = model;

    const size = hullBox.getSize(new THREE.Vector3());
    const scale = TARGET_HULL_LENGTH / size.x;

    const center = hullBox.getCenter(new THREE.Vector3());
    const waterlineY = hullBox.min.y + size.y * WATERLINE_FRACTION;

    // Order matters: the offset is applied in the model's own (pre-scale) units,
    // so it goes on the model and the scale goes on its parent.
    model.position.set(-center.x, -waterlineY, -center.z);

    const scaled = new THREE.Group();
    scaled.name = 'ship-model';
    scaled.scale.setScalar(scale);
    scaled.add(model);

    this.object = new THREE.Object3D();
    this.object.name = 'ship';
    this.object.add(scaled);

    // Moving parts. Each pivot node carries its hinge axis in its glTF extras.
    const axisOf = (node: THREE.Object3D, out: THREE.Vector3) => {
      const a = node.userData.axis as number[] | undefined;
      if (Array.isArray(a) && a.length === 3) out.fromArray(a).normalize();
    };
    this.rudderObject = model.getObjectByName(RUDDER_NODE) ?? null;
    this.helmObject = model.getObjectByName(HELM_NODE) ?? null;
    if (this.rudderObject) axisOf(this.rudderObject, this.rudderAxis);
    if (this.helmObject) {
      axisOf(this.helmObject, this.helmAxis);
      this.object.updateMatrixWorld(true);
      // `object` sits at the origin here, so world space is ship space.
      const pivot = this.helmObject.getWorldPosition(new THREE.Vector3());
      const staff = new THREE.Box3().setFromObject(this.helmObject, true);
      this.helmGrip.set(pivot.x, staff.max.y - 0.06, pivot.z);
      this.helmStation.set(pivot.x - 0.55, pivot.y, pivot.z);
    }

    this.hullLength = size.x * scale;
    this.hullBeam = size.z * scale;
    this.hullFreeboard = size.y * (1 - WATERLINE_FRACTION) * scale;

    const halfL = (this.hullLength / 2) * PROBE_LONGITUDINAL;
    const halfB = (this.hullBeam / 2) * PROBE_LATERAL;

    // Derived from the measured hull box, never hard-coded: swapping the model
    // for `dutch_ship_large_01` must not require touching the physics.
    this.probePoints = [
      new THREE.Vector3(halfL, 0, 0), // bow
      new THREE.Vector3(-halfL, 0, 0), // stern
      new THREE.Vector3(0, 0, -halfB), // port
      new THREE.Vector3(0, 0, halfB), // starboard
    ];

    this.probeGeometry = new THREE.SphereGeometry(Math.max(0.3, this.hullBeam * 0.06), 10, 8);
    this.probeMaterial = new THREE.MeshBasicNodeMaterial();
    this.probeMaterial.color.setHex(0x38e1ff);
    this.probeMaterial.toneMapped = false;
    this.probeMaterial.depthTest = false;
    this.probeMaterial.transparent = true;
    this.probeMaterial.opacity = 0.9;

    this.probeGroup = new THREE.Group();
    this.probeGroup.name = 'ship-buoyancy-probes';
    this.probeGroup.visible = false;
    this.probeGroup.renderOrder = 10;
    for (const point of this.probePoints) {
      const sphere = new THREE.Mesh(this.probeGeometry, this.probeMaterial);
      sphere.position.copy(point);
      sphere.matrixAutoUpdate = false;
      sphere.updateMatrix();
      this.probeGroup.add(sphere);
    }
    this.object.add(this.probeGroup);

    this.sails = new ShipSails(model, scale, this.object);
    this.applyRenderFlags();
    this.applyBottom();
  }

  static async load(loader: AssetLoader): Promise<Ship> {
    const model = await loader.load(SHIP_URL, { anisotropy: HERO_ANISOTROPY });
    model.updateMatrixWorld(true);

    // Measure the *hull* specifically. Including the rigging would put the
    // bowsprit into the length and the mastheads into the vertical extent, and
    // the waterline would end up several metres above the deck.
    const hullBox = measureHull(model);
    return new Ship(model, hullBox);
  }

  setDebugProbesVisible(v: boolean): void {
    this.probeGroup.visible = v;
  }

  /**
   * World heading of the bow, in radians, as
   * `atan2(forward.z, forward.x)` — the convention `Wake.emit()` expects.
   */
  get heading(): number {
    const f = SCRATCH_FORWARD.copy(this.forwardLocal).applyQuaternion(this.object.quaternion);
    return Math.atan2(f.z, f.x);
  }

  /**
   * Rewinds the sails: the cloth goes back to its rest pose and is warmed up
   * by a fixed number of steps on the next update, so every reset to the same
   * state draws the same canvas.
   */
  resetClock(time = 0): void {
    this.sails.reset(time);
  }

  /** Sail setting, wind and luff, each frame; see `SailStateInput`. */
  setSailState(state: SailStateInput): void {
    this.sails.setState(state);
  }

  /** Builds the sail cloth's compute pipelines; part of the boot prewarm. */
  compileSails(renderer: THREE.WebGPURenderer): Promise<void> {
    return this.sails.compile(renderer);
  }

  /**
   * The helm, each frame: `ShipController`'s rudder, -1 hard to port to 1 hard
   * to starboard. Already spooled by the controller, so it is drawn as given.
   */
  setHelm(rudder: number): void {
    this.helm = THREE.MathUtils.clamp(rudder, -1, 1);
  }

  update(dt: number): void {
    if (this.disposed) return;
    // Rudder to starboard: the blade's trailing edge (aft, -X) goes to +Z, a
    // positive turn about the near-vertical hinge.
    this.rudderObject?.quaternion.setFromAxisAngle(this.rudderAxis, this.helm * RUDDER_MAX);
    // The staff's head to starboard: positive about +X takes +Y toward +Z.
    this.helmObject?.quaternion.setFromAxisAngle(this.helmAxis, this.helm * WHIPSTAFF_MAX);
    (this.uShipInverse.value as THREE.Matrix4).copy(this.object.matrixWorld).invert();
    this.sails.update(dt);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.removeFromParent();
    this.probeGeometry.dispose();
    this.probeMaterial.dispose();
    this.sails.dispose();
    // Geometry, materials and textures under `model` belong to the AssetLoader
    // and are shared with any other clone of this asset.
    this.model.clear();
  }

  // ------------------------------------------------------------------ internals

  /**
   * The bottom, below the waterline (S6).
   *
   * The model paints the whole hull the same tarred brown, and from below,
   * lit only by what the water scatters down there, that reads as a black
   * blob. A ship of about 1700 had her bottom paid with "white stuff" when she
   * was careened: tallow, sulphur and train oil (or resin), a dirty cream,
   * renewed every few months against weed and shipworm. Copper sheathing is
   * the 1760s and later. So below the waterline the hull's own texture is
   * re-tinted toward that cream, keeping its planking and seams as relief in
   * the tint, with:
   *
   *   - the boot-top a little above the design waterline, so a heeled ship
   *     shows a pale stripe to windward as period paintings do;
   *   - a fouling band of weed and slime just below the waterline, greener
   *     and patchier near the surface, where the light is;
   *   - the keel and the rudder's heel left darker, tarred, so the keel reads
   *     as a line along the bottom.
   *
   * One shader path on the hull material (shared with the rudder and the
   * whipstaff, whose head is far above the band), every tier: a matrix
   * multiply, a smoothstep or two and one noise lookup per fragment of the
   * hull. Wetness wraps this as its dry colour.
   */
  private applyBottom(): void {
    let material: THREE.Material | null = null;
    this.model.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh || material) return;
      if (`${mesh.name} ${mesh.parent?.name ?? ''}`.toLowerCase().includes('hull')) material = materialsOf(mesh)[0];
    });
    const target = material as (THREE.Material & { colorNode?: unknown }) | null;
    if (!target || !('colorNode' in target)) return;
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const p: any = this.uShipInverse.mul(vec4(positionWorld, 1)).xyz;
    const base: any = (target.colorNode as any) ?? materialColor;
    // Treatment from 6 cm above the design waterline down, feathered over 8 cm.
    const below = float(1).sub(smoothstep(-0.02, 0.06, p.y));
    const luma = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
    // Linear cream. The texture's light and dark (planks against seams, about
    // 0.02 to 0.12 in luma) become a 0.55 to 1.15 relief on it.
    const relief = luma.mul(6).add(0.45).clamp(0.4, 1.2);
    const stuff = vec3(0.46, 0.41, 0.3).mul(relief);
    // Weed and slime: strongest in the top 0.9 m, in patches streaked fore and
    // aft by the flow.
    const patch = mx_noise_float(vec3(p.x.mul(0.35), p.y.mul(2.2), p.z.mul(0.9))).mul(0.5).add(0.5);
    const band = smoothstep(-1.4, -0.25, p.y).mul(below);
    const weed = band.mul(smoothstep(0.35, 0.75, patch)).mul(0.85);
    const fouled = mix(stuff, vec3(0.16, 0.19, 0.08).mul(relief), weed);
    // Keel: the bottom 35 cm within its sided width, kept tarred.
    const keel = float(1).sub(smoothstep(0.17, 0.24, abs(p.z))).mul(float(1).sub(smoothstep(-2.75, -2.6, p.y)));
    const bottom = mix(fouled, base.rgb.mul(0.8), keel);
    (target as unknown as { colorNode: unknown }).colorNode = vec4(mix(base.rgb, bottom, below), base.a);
    /* eslint-enable @typescript-eslint/no-explicit-any */
    target.needsUpdate = true;
  }

  private applyRenderFlags(): void {
    this.model.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;

      mesh.castShadow = true;
      // Self-shadowing across the rigging is what sells the deck clutter.
      mesh.receiveShadow = true;

      // glTF nodes, meshes and materials are named inconsistently across
      // exporters, so match against all three rather than trusting one.
      const materials = materialsOf(mesh);
      const name = [mesh.name, mesh.parent?.name ?? '', ...materials.map((m) => m.name)]
        .join(' ')
        .toLowerCase();
      const isSail = name.includes('sail');
      const isRigging = name.includes('rigging') || name.includes('rope');

      for (const material of materials) {
        // Canvas and rope are surfaces with no back face to speak of; without
        // DoubleSide the sails vanish when the camera crosses the wind.
        if (isSail || isRigging) material.side = THREE.DoubleSide;
        material.shadowSide = THREE.DoubleSide;
      }
    });
  }
}

const SCRATCH_FORWARD = /*@__PURE__*/ new THREE.Vector3();

/**
 * Bounding box of the hull mesh alone, in the root's local space.
 *
 * Falls back to the whole model if no mesh is named like a hull, so a different
 * asset still produces something sane rather than throwing.
 */
function measureHull(root: THREE.Object3D): THREE.Box3 {
  const box = new THREE.Box3();
  let found = false;

  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const name = `${mesh.name} ${mesh.parent?.name ?? ''}`.toLowerCase();
    // The rudder and the whipstaff were cut out of the hull mesh; the hull's
    // box is still the box of all three.
    if (!name.includes('hull') && !name.includes('base') && !name.includes(RUDDER_NODE) && !name.includes(HELM_NODE)) return;

    mesh.geometry.computeBoundingBox();
    const local = mesh.geometry.boundingBox;
    if (!local) return;

    SCRATCH_BOX.copy(local).applyMatrix4(mesh.matrixWorld);
    if (found) box.union(SCRATCH_BOX);
    else box.copy(SCRATCH_BOX);
    found = true;
  });

  if (!found) box.setFromObject(root, true);
  return box;
}

const SCRATCH_BOX = /*@__PURE__*/ new THREE.Box3();

function materialsOf(mesh: THREE.Mesh): THREE.Material[] {
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}
