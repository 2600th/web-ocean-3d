import * as THREE from 'three/webgpu';
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

  private readonly model: THREE.Group;
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

  update(dt: number): void {
    if (this.disposed) return;
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
    if (!name.includes('hull') && !name.includes('base')) return;

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
