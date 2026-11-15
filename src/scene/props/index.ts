import * as THREE from 'three/webgpu';
import { PropBuilder } from './kit';
import { PropMaterials } from './materials';

export { beaconLevel } from './materials';
export { M } from './kit';
export { buildCask, buildBucket, CASK_PRESETS } from './cask';
export { buildChest } from './chest';
export { buildLighthouse, buildKeepersHut, buildBeaconHalo, LIGHTHOUSE_LAMP_Y, LIGHTHOUSE_CLEARING } from './lighthouse';
export {
  buildAnchor,
  buildBartmann,
  buildCannon,
  buildCaseBottle,
  buildConch,
  buildCutlass,
  buildGrating,
  buildLantern,
  buildOliveJar,
  buildRopeCoil,
  buildStoveCask,
  buildWreckage,
} from './stores';

export type Lod = 0 | 1 | 2;
/** Writes one level of a prop into the builder. */
export type PropRecipe = (pb: PropBuilder, lod: Lod) => void;

export interface PropPart {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
}

/**
 * Turns recipes into meshes, and owns everything it makes.
 *
 * A prop is a recipe called once per level of detail; each level comes back as
 * one geometry per material slot. Static props become a `THREE.LOD` whose
 * levels share the library's materials, so every level of every prop compiles
 * to the same handful of pipelines — the boot prewarm compiles a level the
 * first time it sees *any* prop drawn with that material, and a level switch
 * never meets a material it has not already built. Repeated props hand their
 * parts to `Props`' instancing instead.
 *
 * Generated on the main thread at load: 46 recipe-levels, 75-89 ms of
 * geometry building over three boots (headless Chrome, WebGPU, `quality=high`)
 * against the 5.5 MB of glTF it replaced, whose fetch and Meshopt/KTX2 decode
 * it no longer waits on.
 */
export class PropLibrary {
  readonly materials = new PropMaterials();
  private readonly geometries: THREE.BufferGeometry[] = [];

  /** One level of a recipe as material-grouped parts. */
  parts(recipe: PropRecipe, lod: Lod): PropPart[] {
    const pb = new PropBuilder();
    recipe(pb, lod);
    const out: PropPart[] = [];
    for (const [slot, geometry] of pb.build()) {
      this.geometries.push(geometry);
      out.push({ geometry, material: this.materials.get(slot) });
    }
    return out;
  }

  /** One level as a group of plain meshes. */
  group(name: string, recipe: PropRecipe, lod: Lod, castShadow = true): THREE.Group {
    const group = new THREE.Group();
    group.name = name;
    for (const part of this.parts(recipe, lod)) {
      const mesh = new THREE.Mesh(part.geometry, part.material);
      mesh.name = `${name}-${part.material.name}`;
      mesh.castShadow = castShadow;
      mesh.receiveShadow = true;
      group.add(mesh);
    }
    return group;
  }

  /**
   * A static prop with up to three levels. `distances[i]` is where level i
   * takes over, metres; level 0 at 0. A level whose distance is omitted is not
   * built.
   *
   * Every level casts: a prop's shadow is a large part of how it sits on the
   * sand, and a level switch that dropped the shadow would be the visible pop.
   * The levels cost a few hundred triangles each in the shadow pass.
   */
  lod(name: string, recipe: PropRecipe, distances: readonly number[]): THREE.LOD {
    const lod = new THREE.LOD();
    lod.name = name;
    for (let i = 0; i < distances.length && i < 3; i++) {
      lod.addLevel(this.group(`${name}-lod${i}`, recipe, i as Lod), distances[i]);
    }
    return lod;
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.materials.dispose();
  }
}
