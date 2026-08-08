import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  float,
  positionGeometry,
  texture,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import { fetchWithDeadline, withDeadline } from './AssetLoader';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

const vec3n = vec3 as unknown as (x: unknown, y: unknown, z: unknown) => Node;

const SWAY_WAVELENGTH = 90;
const SWAY_K = (Math.PI * 2) / SWAY_WAVELENGTH;
const SWAY_SPEED = 7;
const SWAY_OMEGA = SWAY_K * SWAY_SPEED;
const SWAY_MAX = 0.9;
const CLOCK_WRAP = 3600;
const RANGE_FADE = 24;
const NO_CULL_DISTANCE = 1_000_000;

export interface ImposterUvRect {
  u0: number;
  u1: number;
  v0: number;
  v1: number;
  origin: 'bottom-left';
}

export interface ImposterSpecies {
  sourceGlb: string;
  sourceGlbSlug: string;
  uv: ImposterUvRect;
  worldWidth: number;
  worldHeight: number;
  pivot: {
    offsetX: number;
    heightAboveBottom: number;
  };
  frame: {
    viewSize: number;
  };
}

export interface ImposterSidecar {
  schema: 'web-ocean-3d/imposters@1';
  atlas: {
    file: string;
    width: number;
    height: number;
  };
  species: Record<string, ImposterSpecies>;
}

export interface ImpostersLoadOptions {
  metadataUrl?: string;
  textureUrl?: string;
}

export interface ImposterPlacement {
  species: string;
  position: THREE.Vector3 | readonly [number, number, number];
  /** The uniform x/z scale written by Props. */
  scale: number;
  /** The extra y stretch written by Props. */
  stretch?: number;
  /** The source model's yaw about world up. */
  yaw: number;
  minDistance?: number;
  maxDistance?: number;
  /** Per-kind index used to mirror Props' density thinning. */
  detailSlot?: number;
  detailCapacity?: number;
}

interface PendingPlacement {
  speciesIndex: number;
  position: [number, number, number];
  scale: number;
  stretch: number;
  yaw: number;
  minDistance: number;
  maxDistance: number;
  detailSlot: number;
  detailCapacity: number;
}

/**
 * One instanced, alpha-tested draw for the vegetation atlas.
 *
 * The card is bottom-anchored at the transform Props calculated for the mesh.
 * Its square is deliberately sized from `frame.viewSize`, not the recorded
 * mesh bounds: the bake letterboxes each plant inside that square, and mapping
 * the full cell onto the mesh bounds would scale the plant twice.
 */
export class Imposters {
  readonly object: THREE.Object3D;

  private readonly atlas: THREE.Texture;
  private readonly sidecar: ImposterSidecar;
  private readonly speciesIndex = new Map<string, number>();
  private readonly placements: PendingPlacement[] = [];
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly uWind = uniform(new THREE.Vector2(1, 0));
  private readonly uWindStrength = uniform(0.5);
  private readonly uPhase = uniform(0);

  private geometry: THREE.InstancedBufferGeometry | null = null;
  private mesh: THREE.Mesh | null = null;
  private phase = 0;
  private detail = 1;
  private wantVisible = true;
  private sealed = false;
  private disposed = false;

  static async load(options: ImpostersLoadOptions = {}): Promise<Imposters> {
    const metadataUrl = options.metadataUrl ?? '/imposters/vegetation.json';
    // Both of these are awaited by the boot, so both carry the same deadline
    // every other asset request does. A sidecar or atlas that never answers
    // would otherwise pend the island load forever.
    const response = await fetchWithDeadline(metadataUrl);
    if (!response.ok) {
      throw new Error(`Imposter sidecar request failed: ${response.status} ${metadataUrl}`);
    }

    const sidecar = (await response.json()) as ImposterSidecar;
    if (sidecar.schema !== 'web-ocean-3d/imposters@1') {
      throw new Error(`Unsupported imposter sidecar schema: ${String(sidecar.schema)}`);
    }

    const textureUrl = options.textureUrl ?? `/${sidecar.atlas.file.replace(/^\/+/, '')}`;
    const atlas = await withDeadline(
      new THREE.TextureLoader().loadAsync(textureUrl),
      textureUrl,
    );
    // The sidecar's UVs use the normal Three.js/WebGL bottom-left convention.
    // TextureLoader's default flip is what maps the PNG's top-left pixel rects
    // into that convention; make it explicit before the first upload.
    atlas.flipY = true;
    atlas.colorSpace = THREE.SRGBColorSpace;
    atlas.premultiplyAlpha = true;
    atlas.needsUpdate = true;

    return new Imposters(atlas, sidecar);
  }

  constructor(atlas: THREE.Texture, sidecar: ImposterSidecar) {
    this.atlas = atlas;
    this.sidecar = sidecar;

    const ordered = Object.entries(sidecar.species).sort(
      ([, a], [, b]) => a.sourceGlbSlug.localeCompare(b.sourceGlbSlug),
    );
    for (let index = 0; index < ordered.length; index++) {
      this.speciesIndex.set(ordered[index][0], index);
    }

    this.material = this.buildMaterial();
    this.object = new THREE.Object3D();
    this.object.name = 'island-imposters-field';
    this.object.matrixAutoUpdate = false;
    this.object.updateMatrix();
  }

  addPlacement(placement: ImposterPlacement): void {
    if (this.disposed) throw new Error('Imposters disposed; cannot add placement');
    if (this.sealed) throw new Error('Imposters already finalised; cannot add placement');

    const speciesIndex = this.speciesIndex.get(placement.species);
    if (speciesIndex === undefined) {
      throw new Error(`No imposter atlas species for ${placement.species}`);
    }

    const position =
      placement.position instanceof THREE.Vector3
        ? [placement.position.x, placement.position.y, placement.position.z] as const
        : placement.position;
    const scale = Math.max(0, placement.scale);
    const stretch = Math.max(0.01, placement.stretch ?? 1);
    const minDistance = Math.max(0, placement.minDistance ?? 0);
    const maxDistance = Math.max(minDistance + 1, placement.maxDistance ?? NO_CULL_DISTANCE);
    const detailCapacity = Math.max(1, Math.floor(placement.detailCapacity ?? 1));
    const detailSlot = Math.max(0, Math.floor(placement.detailSlot ?? 0));

    this.placements.push({
      speciesIndex,
      position: [position[0], position[1], position[2]],
      scale,
      stretch,
      yaw: placement.yaw,
      minDistance,
      maxDistance,
      detailSlot,
      detailCapacity,
    });
  }

  /** Builds the one geometry and one mesh after Props has handed over all placements. */
  finalize(): void {
    if (this.disposed) return;
    if (this.sealed) return;
    this.sealed = true;

    const count = this.placements.length;
    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(
        new Float32Array([-0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0]),
        3,
      ),
    );
    this.geometry.setAttribute(
      'uv',
      new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 2),
    );

    const position = new Float32Array(count * 3);
    // Pack instance values to stay below WebGPU vertex-buffer limits.
    const transform = new Float32Array(count * 4);
    const uv = new Float32Array(count * 4);
    const frame = new Float32Array(count * 4);
    const active = new Float32Array(count);

    const orderedSpecies = Object.entries(this.sidecar.species).sort(
      ([, a], [, b]) => a.sourceGlbSlug.localeCompare(b.sourceGlbSlug),
    );
    for (let i = 0; i < count; i++) {
      const item = this.placements[i];
      const metadata = orderedSpecies[item.speciesIndex][1];
      const unitCardBottom =
        metadata.worldHeight * 0.5 -
        metadata.pivot.heightAboveBottom -
        metadata.frame.viewSize * 0.5;

      position.set(item.position, i * 3);
      transform[i * 4] = item.scale;
      transform[i * 4 + 1] = item.scale * item.stretch;
      transform[i * 4 + 2] = item.yaw;
      transform[i * 4 + 3] = metadata.frame.viewSize;
      uv.set([metadata.uv.u0, metadata.uv.u1, metadata.uv.v0, metadata.uv.v1], i * 4);
      // The bake frames the cell around the bounds centre. `offsetX` is the
      // negative centre x, so its negation moves that frame back to Props'
      // source origin. The y term is the bottom of the square relative to the
      // source origin and includes the bake's intentional transparent margin.
      frame[i * 4] = -metadata.pivot.offsetX;
      frame[i * 4 + 1] = unitCardBottom;
      frame[i * 4 + 2] = item.minDistance;
      frame[i * 4 + 3] = item.maxDistance;
      active[i] = this.isActive(item, this.detail) ? 1 : 0;
    }

    this.geometry.setAttribute('imposterPosition', new THREE.InstancedBufferAttribute(position, 3));
    this.geometry.setAttribute('imposterTransform', new THREE.InstancedBufferAttribute(transform, 4));
    this.geometry.setAttribute('imposterUv', new THREE.InstancedBufferAttribute(uv, 4));
    this.geometry.setAttribute('imposterFrame', new THREE.InstancedBufferAttribute(frame, 4));
    this.geometry.setAttribute('imposterActive', new THREE.InstancedBufferAttribute(active, 1));
    this.geometry.instanceCount = count;
    this.geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(0, 0, 0),
      Number.POSITIVE_INFINITY,
    );

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'island-imposters';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();
    this.object.add(this.mesh);
    this.applyVisibility();
  }

  getCount(): number {
    return this.placements.length;
  }

  setDetailScale(scale: number): void {
    this.detail = Math.max(0, Math.min(1, scale));
    if (!this.geometry) return;
    const active = this.geometry.getAttribute('imposterActive') as THREE.InstancedBufferAttribute;
    const data = active.array as Float32Array;
    for (let i = 0; i < this.placements.length; i++) {
      data[i] = this.isActive(this.placements[i], this.detail) ? 1 : 0;
    }
    active.needsUpdate = true;
    this.applyVisibility();
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  /** `direction` is a world-xz unit vector; `strength` is 0..1. */
  setWind(direction: THREE.Vector2, strength: number): void {
    const wind = this.uWind.value as THREE.Vector2;
    wind.copy(direction);
    if (wind.lengthSq() < 1e-6) wind.set(1, 0);
    wind.normalize();
    this.uWindStrength.value = Math.max(0, Math.min(1, strength));
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.phase = (this.phase + dt) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  resetClock(time = 0): void {
    if (this.disposed) return;
    this.phase = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.object.removeFromParent();
    if (this.mesh) this.object.remove(this.mesh);
    this.geometry?.dispose();
    this.material.dispose();
    this.atlas.dispose();
    this.placements.length = 0;
  }

  private isActive(placement: PendingPlacement, detail: number): boolean {
    const kept = Math.max(1, Math.ceil(placement.detailCapacity * 0.25));
    const wanted = Math.max(kept, Math.round(placement.detailCapacity * detail));
    return placement.detailSlot < Math.min(placement.detailCapacity, wanted);
  }

  private applyVisibility(): void {
    if (!this.mesh) return;
    this.mesh.visible = this.wantVisible && this.placements.length > 0;
  }

  private buildMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial();
    material.name = 'vegetation-imposter-material';
    material.transparent = false;
    material.alphaTest = 0.42;
    material.alphaToCoverage = true;
    material.side = THREE.FrontSide;

    const corner = positionGeometry as Node;
    const placement = attribute('imposterPosition', 'vec3') as Node;
    const transform = attribute('imposterTransform', 'vec4') as Node;
    const frame = attribute('imposterFrame', 'vec4') as Node;
    const size = vec2(transform.x, transform.y) as Node;
    const yaw = transform.z;
    const viewSize = transform.w;
    const pivot = vec2(frame.x, frame.y) as Node;
    const range = vec2(frame.z, frame.w) as Node;
    const active = attribute('imposterActive', 'float') as Node;

    material.positionNode = Fn(() => {
      const base = vec3n(placement.x, placement.y, placement.z);
      const toCamera = cameraPosition.sub(base);
      // Away from the camera, not toward it, and the sign is load-bearing.
      // `right x worldUp` is `-flat`, so that is the face normal; with
      // `FrontSide` the card is only drawn when it points at the viewer. Built
      // from `toCamera` directly it points away, every card is back-face
      // culled, and the entire vegetation field is invisible on exactly the
      // tiers that have no meshes to fall back on.
      //
      // `Canopy` gets this right by computing the same vector with the opposite
      // subtraction (`Canopy.ts:481`) — where the local is also called
      // `toCamera` and also is not.
      const flat = toCamera
        .negate()
        .mul(vec3(1, 0, 1))
        .normalize()
        .add(vec3(1e-4, 0, 0))
        .normalize() as Node;
      const right = vec3n(flat.z.negate(), 0, flat.x);

      // The frame centre offset is in the source model's local x axis. Yaw is
      // retained even though the card itself turns cylindrically to the camera:
      // it is what keeps asymmetric baked bounds anchored to the same source
      // origin as the mesh it replaces.
      const sourceX = vec3n(yaw.cos(), 0, yaw.sin().negate());
      const frameOffset = sourceX.mul(pivot.x.mul(size.x));
      const width = size.x.mul(viewSize);
      const height = size.y.mul(viewSize);

      const distance = toCamera.length();
      const near = distance.smoothstep(range.x, range.x.add(2));
      const far = float(1).sub(
        distance.smoothstep(range.y.sub(RANGE_FADE), range.y),
      );
      const presence = active.mul(near).mul(far);

      const along = base.x.mul(this.uWind.x).add(base.z.mul(this.uWind.y));
      const gust = along
        .mul(SWAY_K)
        .sub(this.uPhase.mul(SWAY_OMEGA))
        .add(yaw.mul(0.37))
        .sin();
      const lean = gust.mul(SWAY_MAX).mul(this.uWindStrength).mul(corner.y.mul(corner.y));

      const card = base
        .add(frameOffset)
        .add(right.mul(corner.x.mul(width)))
        .add(vec3n(0, pivot.y.mul(size.y).add(corner.y.mul(height)), 0))
        .add(vec3n(this.uWind.x.mul(lean), 0, this.uWind.y.mul(lean)));
      return base.add(card.sub(base).mul(presence));
    })();

    material.colorNode = Fn(() => {
      const cardUv = attribute('uv', 'vec2') as Node;
      const rect = attribute('imposterUv', 'vec4') as Node;
      const atlasUv = vec2(
        rect.x.add(cardUv.x.mul(rect.y.sub(rect.x))),
        rect.z.add(cardUv.y.mul(rect.w.sub(rect.z))),
      );
      // Alpha is deliberately left in the sampled vec4 so alphaTest can keep
      // depth-write. Blending would make one instanced draw order-dependent.
      return texture(this.atlas, atlasUv);
    })();

    return material;
  }
}
