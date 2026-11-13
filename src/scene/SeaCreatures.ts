import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraPosition,
  cos,
  faceDirection,
  float,
  interleavedGradientNoise,
  mix,
  modelWorldMatrix,
  normalGeometry,
  normalWorld,
  positionGeometry,
  positionWorld,
  screenCoordinate,
  select,
  sin,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { mulberry32 } from '../core/random';
import { seafloorHeight } from './Seafloor';
import {
  EAGLE_RAY,
  MANTA,
  TURTLE_PART,
  TURTLE_PIVOTS,
  buildCrab,
  buildJellyfish,
  buildRay,
  buildReefFish,
  buildTurtle,
  type RayStyle,
} from './reef/CreatureGeometry';
import { lifePattern, submergedLight } from './reef/LifeShading';

/**
 * The reef's animals: turtles, manta and eagle rays, jellyfish, clouds of small
 * reef fish over every coral head, and crabs on the sand.
 *
 * Everything here follows the rules `Fish` and `Kelp` set, for the same reasons:
 *
 * - **Procedural, no rig.** Each animal is a template from
 *   `reef/CreatureGeometry` with its moving parts tagged per vertex. A turtle's
 *   flipper stroke is a rotation of the tagged vertices about a shoulder pivot;
 *   a ray's flap is a travelling wave out along its span; a fish's swim is a
 *   wave down its body; a jelly's pulse is its rim contracting. All of it is in
 *   the vertex stage, in TSL, and none of it is skinned.
 * - **Closed-form in the clock.** No integrated velocity or remembered position
 *   anywhere: every pose is a pure function of `t`, so `resetClock(t)` lands on
 *   exactly the frame a capture expects.
 * - **Instanced where it is a crowd.** The reef fish, the jellies and the crabs
 *   are one draw each, placed entirely on the GPU from per-instance constants —
 *   the CPU writes a clock and nothing else. The large animals are a handful of
 *   ordinary meshes whose frame the CPU solves, because the CPU can ask
 *   `seafloorHeight` where the bottom is and the shader cannot cheaply.
 * - **Placed where the reef is.** Homes come from `ReefLife`'s sites, so fish
 *   mill over coral, crabs work the sand at its edge and the turtles and rays
 *   cruise circuits that pass over it.
 *
 * Lighting is the submerged model in `reef/LifeShading` — wrapped sun, sky
 * ambient and the caustic web — so an animal swimming through a caustic pattern
 * is lit by it the way the sand below is.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/** `vec3` under a loose signature — see the note in `Kelp`. */
const v3 = vec3 as unknown as (...args: unknown[]) => Node;

const CREATURE_SEED = 0x7e57a1;

/** Above this eye height every creature is hidden and none is updated. */
const HIDE_ABOVE = 40;

/** Highest a swimming animal may come, metres — clear of a trough overhead. */
const CEILING = -2.6;

export interface CreatureSite {
  x: number;
  z: number;
  radius: number;
}

export interface SeaCreaturesOptions {
  /** Reef sites to live over: the plateau's and the island's. */
  sites: { plateau: readonly CreatureSite[]; island: readonly CreatureSite[] };
  /** `(worldPosition) => ~1` caustic pattern. */
  caustics?: (worldPosition: Node) => Node;
}

/** One large animal on a closed-form circuit. */
interface Cruiser {
  mesh: THREE.Mesh;
  cx: number;
  cz: number;
  radius: number;
  /** Angular rate, rad/s, signed. */
  omega: number;
  phase: number;
  hover: number;
  flapHz: number;
  /** Order in which a tier's budget keeps it. */
  rank: number;
}

const _p0 = new THREE.Vector3();
const _p1 = new THREE.Vector3();
const _p2 = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _up = new THREE.Vector3(0, 1, 0);
const _right = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _roll = new THREE.Quaternion();

export class SeaCreatures {
  readonly object = new THREE.Group();

  private detail = 1;
  private clock = 0;
  private wantVisible = true;
  private cameraAllows = true;
  private disposed = false;

  private readonly uTime = uniform(0);
  private readonly uFrame = uniform(0);
  private readonly uSunDir = uniform(new THREE.Vector3(0.35, 0.62, 0.7).normalize());
  private readonly uSunColor = uniform(new THREE.Color(0.9, 0.88, 0.78));
  private readonly uAmbient = uniform(new THREE.Color(0.24, 0.34, 0.38));

  private readonly cruisers: Cruiser[] = [];
  private readonly crowds: { mesh: THREE.Mesh; geometry: THREE.InstancedBufferGeometry; capacity: number }[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly caustics: ((worldPosition: Node) => Node) | null;

  constructor(detail: number, options: SeaCreaturesOptions) {
    this.object.name = 'sea-creatures';
    this.caustics = options.caustics ?? null;
    const random = mulberry32(CREATURE_SEED);
    const plateau = options.sites.plateau;
    const island = options.sites.island;
    const all = plateau.concat(island);

    this.buildReefFish(random, all);
    this.buildJellies(random, plateau, island);
    this.buildCrabs(random, all);
    this.buildTurtles(random, plateau, island);
    this.buildRays(random, plateau, island);

    this.setDetail(detail);
    this.resetClock(0);
  }

  setDetail(detail: number): void {
    this.detail = Math.max(0, Math.min(1, detail));
    for (const crowd of this.crowds) {
      crowd.geometry.instanceCount = Math.round(crowd.capacity * this.detail);
      crowd.mesh.visible = crowd.geometry.instanceCount > 0;
    }
    // The big animals are few, so a tier keeps the first of each kind by rank
    // (0..1 within the kind) — half detail is half the rays, not no turtles.
    for (const c of this.cruisers) c.mesh.visible = this.detail > 0 && c.rank < this.detail;
    this.applyVisibility();
    this.push();
  }

  setVisible(visible: boolean): void {
    this.wantVisible = visible;
    this.applyVisibility();
  }

  setCameraHeight(height: number): void {
    const allows = height < HIDE_ABOVE;
    if (allows !== this.cameraAllows) {
      this.cameraAllows = allows;
      this.applyVisibility();
    }
  }

  setSun(direction: THREE.Vector3, color: THREE.Color): void {
    const d = this.uSunDir.value as THREE.Vector3;
    d.copy(direction);
    if (d.lengthSq() < 1e-8) d.set(0, 1, 0);
    d.normalize();
    (this.uSunColor.value as THREE.Color).copy(color);
  }

  setAmbient(color: THREE.Color): void {
    (this.uAmbient.value as THREE.Color).copy(color);
  }

  update(dt: number): void {
    if (this.disposed) return;
    this.clock += dt;
    this.uFrame.value = ((this.uFrame.value as number) + 1) % 64;
    this.push();
  }

  resetClock(time = 0): void {
    this.clock = time;
    this.uFrame.value = 0;
    this.push();
  }

  get visible(): boolean {
    return this.object.visible;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.object.removeFromParent();
  }

  // ------------------------------------------------------------------- frame

  private applyVisibility(): void {
    this.object.visible = this.wantVisible && this.cameraAllows && this.detail > 0;
  }

  private push(): void {
    // Wrapped once an hour, the same compromise the particles and the pass make.
    this.uTime.value = this.clock % 3600;
    if (!this.object.visible) return;
    for (const c of this.cruisers) {
      if (!c.mesh.visible) continue;
      this.poseCruiser(c, this.clock);
    }
  }

  private cruiserAt(c: Cruiser, t: number, out: THREE.Vector3): THREE.Vector3 {
    const th = c.phase + c.omega * t;
    const x = c.cx + c.radius * (Math.cos(th) + 0.22 * Math.cos(0.41 * th + c.phase));
    const z = c.cz + c.radius * (0.82 * Math.sin(th) + 0.22 * Math.sin(0.57 * th + 1.3));
    const floor = seafloorHeight(x, z);
    const y = Math.min(CEILING, floor + c.hover + 0.8 * Math.sin(0.23 * th * 3 + c.phase));
    return out.set(x, Math.max(floor + 0.8, y), z);
  }

  private poseCruiser(c: Cruiser, t: number): void {
    const p = this.cruiserAt(c, t, _p0);
    const ahead = this.cruiserAt(c, t + 0.6, _p1);
    const behind = this.cruiserAt(c, t - 0.6, _p2);
    _fwd.subVectors(ahead, behind);
    const horiz = Math.hypot(_fwd.x, _fwd.z);
    // Pitch is damped — an animal climbing over a bommie noses up a little, not
    // along the full slope of its path.
    _fwd.y *= 0.5;
    _fwd.normalize();
    if (horiz < 1e-5) return;
    // Bank into the turn: the signed change of heading over the window.
    const h0 = Math.atan2(p.x - behind.x, p.z - behind.z);
    const h1 = Math.atan2(ahead.x - p.x, ahead.z - p.z);
    let turn = h1 - h0;
    if (turn > Math.PI) turn -= Math.PI * 2;
    if (turn < -Math.PI) turn += Math.PI * 2;
    const bank = Math.max(-0.45, Math.min(0.45, -turn * 4));
    _right.crossVectors(_up, _fwd).normalize();
    const up = _p2.crossVectors(_fwd, _right).normalize();
    _m.makeBasis(_right, up, _fwd);
    _q.setFromRotationMatrix(_m);
    _roll.setFromAxisAngle(_fwd, bank);
    c.mesh.quaternion.copy(_roll).multiply(_q);
    c.mesh.position.copy(p);
    c.mesh.userData.flapPhase = (Math.PI * 2 * c.flapHz * t + c.phase * 3) % (Math.PI * 2);
    c.mesh.updateMatrix();
  }

  // --------------------------------------------------------------- materials

  /** The submerged light for a world normal and position, as an albedo multiplier. */
  private light(normal: Node, worldPosition: Node, translucency = 0): Node {
    return submergedLight({
      normal,
      worldPosition,
      sunDirection: this.uSunDir,
      sunColor: this.uSunColor,
      ambient: this.uAmbient,
      caustics: this.caustics,
      translucency,
    });
  }

  private track<T extends THREE.Material>(m: T): T {
    this.materials.push(m);
    return m;
  }

  // ---------------------------------------------------------------- the fish

  /**
   * Small reef fish milling over the coral: chromis, anthias, tangs, sergeant
   * majors, butterflyfish. One species per swarm, a few dozen swarms, one draw.
   */
  private buildReefFish(random: () => number, sites: readonly CreatureSite[]): void {
    interface FishKind {
      a: number;
      b: number;
      pattern: number;
      size: [number, number];
      depth: number;
      count: [number, number];
      radius: [number, number];
    }
    const KINDS: FishKind[] = [
      // Blue-green chromis: the clouds over every coral head.
      { a: 0x46b8e8, b: 0x9fe0f0, pattern: 0, size: [0.07, 0.1], depth: 1, count: [14, 26], radius: [0.6, 2.2] },
      // Anthias: orange-pink, lavender belly.
      { a: 0xf0743a, b: 0xd890e0, pattern: 0, size: [0.07, 0.1], depth: 1, count: [12, 22], radius: [0.5, 1.8] },
      // Yellow tang: a deep yellow disc.
      { a: 0xf2d42c, b: 0xf6e070, pattern: 0, size: [0.13, 0.18], depth: 1.6, count: [4, 8], radius: [1.2, 3] },
      // Sergeant major: yellow-backed with black bars.
      { a: 0xd8d470, b: 0x1a1a1a, pattern: 1, size: [0.12, 0.16], depth: 1.3, count: [5, 10], radius: [1, 2.5] },
      // Blue tang: royal blue, yellow tail.
      { a: 0x2a5ae0, b: 0xf0d030, pattern: 2, size: [0.15, 0.2], depth: 1.4, count: [3, 6], radius: [1.5, 3.5] },
      // Butterflyfish: white with a yellow rear and a dark band.
      { a: 0xf2f0e6, b: 0xf0c020, pattern: 2, size: [0.11, 0.14], depth: 1.7, count: [2, 4], radius: [0.8, 2] },
      // Clownfish-coloured damsels: orange with white bars.
      { a: 0xf06818, b: 0xf6f6f0, pattern: 1, size: [0.07, 0.09], depth: 1.2, count: [2, 5], radius: [0.3, 0.8] },
    ];
    // Swarms per site, and the order kinds are dealt in so a tier prefix keeps
    // the common species.
    const SWARMS_PER_SITE = 4;
    const swarms: { x: number; y: number; z: number; kind: FishKind }[] = [];
    let k = 0;
    for (const site of sites) {
      for (let s = 0; s < SWARMS_PER_SITE; s++) {
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(random()) * site.radius * 0.8;
        const x = site.x + Math.cos(a) * r;
        const z = site.z + Math.sin(a) * r;
        const kind = KINDS[k++ % KINDS.length];
        swarms.push({ x, z, y: seafloorHeight(x, z), kind });
      }
    }
    const fish: number[][] = [];
    for (const swarm of swarms) {
      const n = swarm.kind.count[0] + Math.floor(random() * (swarm.kind.count[1] - swarm.kind.count[0] + 1));
      const dir = random() < 0.5 ? -1 : 1;
      for (let i = 0; i < n; i++) {
        const kind = swarm.kind;
        const size = kind.size[0] + random() * (kind.size[1] - kind.size[0]);
        const radius = kind.radius[0] + random() * (kind.radius[1] - kind.radius[0]);
        // Swim at two to four body lengths a second around the orbit.
        const speed = size * (2 + random() * 2);
        const omega = (dir * speed) / radius;
        const hx = swarm.x + (random() - 0.5) * 1.2;
        const hz = swarm.z + (random() - 0.5) * 1.2;
        const hy = Math.min(CEILING, seafloorHeight(hx, hz) + 0.5 + random() * 1.8);
        const a = new THREE.Color(kind.a);
        const b = new THREE.Color(kind.b);
        const jitter = 0.85 + random() * 0.3;
        fish.push([
          hx, hy, hz, size,
          radius, omega, random() * Math.PI * 2, 0.15 + random() * 0.35,
          a.r * jitter, a.g * jitter, a.b * jitter, kind.pattern,
          b.r, b.g, b.b, 5 + random() * 3,
          kind.depth, 0, 0, 0,
        ]);
      }
    }
    // Shuffle, so a tier's prefix thins every swarm rather than deleting some.
    for (let i = fish.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [fish[i], fish[j]] = [fish[j], fish[i]];
    }

    const template = buildReefFish().build({ aux: true });
    const geometry = toInstanced(template, fish, ['fHome', 'fMotion', 'fColA', 'fColB', 'fShape']);
    this.geometries.push(template, geometry);

    const material = this.track(new THREE.MeshBasicNodeMaterial());
    material.name = 'reef-fish';
    material.vertexColors = true;
    material.side = THREE.DoubleSide;
    material.fog = false;

    const home: Node = attribute('fHome', 'vec4');
    const motion: Node = attribute('fMotion', 'vec4');
    const colA: Node = attribute('fColA', 'vec4');
    const colB: Node = attribute('fColB', 'vec4');
    const shape: Node = attribute('fShape', 'vec4');
    const s: Node = attribute('reefAux', 'float');

    const frame = (): { centre: Node; fwd: Node; right: Node; up: Node } => {
      const t = this.uTime;
      const R = motion.x;
      const th = motion.z.add(motion.y.mul(t));
      const bob = motion.w;
      const offset = v3(
        R.mul(cos(th).add(cos(th.mul(2).add(motion.z)).mul(0.15))),
        bob.mul(sin(th.mul(1.3).add(motion.z.mul(2)))),
        R.mul(sin(th).add(sin(th.mul(3)).mul(0.15))),
      );
      const sign = motion.y.sign();
      const tangent = v3(
        R.mul(sin(th).negate().sub(sin(th.mul(2).add(motion.z)).mul(0.3))),
        bob.mul(1.3).mul(cos(th.mul(1.3).add(motion.z.mul(2)))).mul(0.5),
        R.mul(cos(th).add(cos(th.mul(3)).mul(0.45))),
      ).mul(sign);
      const fwd = tangent.normalize();
      const right = v3(0, 1, 0).cross(fwd).normalize();
      const up = fwd.cross(right);
      return { centre: home.xyz.add(offset), fwd, right, up };
    };

    const f = frame();
    const size = home.w;
    const scale = v3(size.mul(shape.x.mul(0.0).add(1)), size.mul(shape.x), size);
    // The swim: a wave down the body, amplitude growing to the tail.
    const beat = this.uTime.mul(colB.w).mul(Math.PI * 2).add(motion.z.mul(7));
    const sway = sin(beat.sub(s.mul(5))).mul(s.mul(s)).mul(0.13);
    const local = positionGeometry.add(v3(sway, 0, 0)).mul(scale);
    const world = f.centre
      .add(f.right.mul(local.x))
      .add(f.up.mul(local.y))
      .add(f.fwd.mul(local.z));
    // Collapse past the visible range — see `Kelp.FADE_FULL`.
    const fade = float(1).sub(home.xyz.sub(cameraPosition).length().smoothstep(70, 110));
    material.positionNode = mix(home.xyz, world, fade);

    const nLocal = normalGeometry.div(scale);
    const nWorld: Node = varying(
      f.right.mul(nLocal.x).add(f.up.mul(nLocal.y)).add(f.fwd.mul(nLocal.z)).normalize(),
      'rfNormal',
    );
    material.colorNode = Fn(() => {
      const pg = positionGeometry;
      const pattern = colA.w;
      const isBars = pattern.sub(1).abs().lessThan(0.5);
      const isTail = pattern.sub(2).abs().lessThan(0.5);
      const bars = sin(pg.z.mul(30).add(1)).smoothstep(0.35, 0.65);
      const tail = pg.z.negate().smoothstep(-0.05, 0.08);
      const belly = pg.y.smoothstep(-0.12, 0.1);
      const plain = mix(colB.rgb, colA.rgb, belly);
      const albedo = select(isBars, mix(colA.rgb, colB.rgb, bars.mul(0.85)), select(isTail, mix(colA.rgb, colB.rgb, tail), plain));
      const normal = nWorld.normalize().mul(faceDirection);
      return vec4(albedo.mul(this.light(normal, positionWorld, 0.15)), 1);
    })();

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'reef-fish-swarms';
    mesh.frustumCulled = false;
    this.object.add(mesh);
    this.crowds.push({ mesh, geometry, capacity: fish.length });
  }

  // ------------------------------------------------------------- the jellies

  private buildJellies(random: () => number, plateau: readonly CreatureSite[], island: readonly CreatureSite[]): void {
    const blooms: CreatureSite[] = [];
    if (plateau.length > 0) blooms.push(plateau[0], plateau[Math.min(plateau.length - 1, 3)]);
    if (island.length > 0) blooms.push(island[Math.min(island.length - 1, 2)]);
    const jellies: number[][] = [];
    const tints = [0xe8d8f8, 0xd8ecff, 0xf8d8e8, 0xe0f0f0];
    for (const bloom of blooms) {
      const n = 14;
      for (let i = 0; i < n; i++) {
        const a = random() * Math.PI * 2;
        const r = 4 + random() * 16;
        const x = bloom.x + Math.cos(a) * r;
        const z = bloom.z + Math.sin(a) * r;
        const floor = seafloorHeight(x, z);
        const y = floor + 3 + random() * Math.max(0.5, -floor - 6.5);
        const tint = new THREE.Color(tints[Math.floor(random() * tints.length)]);
        jellies.push([
          x, Math.min(-3.2, y), z, 0.18 + random() * 0.22,
          random() * Math.PI * 2, 0.45 + random() * 0.35, 0.6 + random() * 1.6, 0.05 + random() * 0.05,
          tint.r, tint.g, tint.b, 0,
        ]);
      }
    }
    for (let i = jellies.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [jellies[i], jellies[j]] = [jellies[j], jellies[i]];
    }
    const template = buildJellyfish().build({ aux: true });
    const geometry = toInstanced(template, jellies, ['jHome', 'jParam', 'jColor']);
    this.geometries.push(template, geometry);

    const material = this.track(new THREE.MeshBasicNodeMaterial());
    material.name = 'jellyfish';
    material.vertexColors = true;
    material.side = THREE.DoubleSide;
    material.fog = false;

    const home: Node = attribute('jHome', 'vec4');
    const param: Node = attribute('jParam', 'vec4');
    const tint: Node = attribute('jColor', 'vec4');
    const part: Node = attribute('reefAux', 'float');
    const along: Node = (attribute('reefSway', 'vec2') as Node).x;

    const t = this.uTime;
    const phi = t.mul(param.y).mul(Math.PI * 2).add(param.x);
    // Contraction, 0 relaxed to 1 closed — a sharp squeeze and a slow refill.
    const squeeze = sin(phi).mul(0.5).add(0.5).pow(1.6);
    const p: Node = positionGeometry;
    const radial = vec2(p.x, p.z).length();
    const isBell = part.lessThan(0.5);
    // The bell's rim pulls in and the dome rises; appendages follow the rim at
    // their root and trail a wave behind it.
    const rim = select(isBell, radial.smoothstep(0.2, 1.0), float(1).sub(along).max(0));
    const pull = float(1).sub(squeeze.mul(0.2).mul(rim));
    const lift = select(isBell, p.y.mul(squeeze.mul(0.14)), float(0));
    const wave = phi.mul(0.5).sub(along.mul(3.2)).add(p.x.add(p.z).mul(2));
    const trail = along.mul(along).mul(0.28);
    const local = v3(
      p.x.mul(pull).add(sin(wave).mul(trail)),
      p.y.add(lift).add(along.mul(squeeze).mul(0.12)),
      p.z.mul(pull).add(cos(wave.mul(0.9)).mul(trail)),
    );
    const drift = v3(
      cos(t.mul(param.w).add(param.x)).mul(param.z),
      sin(t.mul(0.13).add(param.x.mul(1.7))).mul(0.7).add(sin(phi.sub(1.3)).mul(0.08)),
      sin(t.mul(param.w).add(param.x)).mul(param.z),
    );
    const centre = home.xyz.add(drift);
    const fade = float(1).sub(home.xyz.sub(cameraPosition).length().smoothstep(60, 95));
    material.positionNode = centre.add(local.mul(home.w).mul(fade));

    // Opaque and dithered rather than blended. The underwater pass fogs by the
    // depth buffer, so a blended bell — which writes none — would be fogged as
    // though it were the open water behind it and vanish into it. A stochastic
    // mask writes the bell's own depth where it is opaque and leaves the
    // background's where it is not, and the temporal resolve integrates the
    // pattern into translucency, re-seeded every frame so it does.
    const nW = normalWorld.mul(faceDirection);
    const view = cameraPosition.sub(positionWorld).normalize();
    const facing = nW.dot(view).abs();
    const fresnel = float(1).sub(facing).pow(1.5);
    const alpha = select(
      isBell,
      fresnel.mul(0.7).add(0.22),
      select(part.lessThan(1.5), float(0.55), float(0.4)),
    ).mul(fade);
    const threshold = interleavedGradientNoise(screenCoordinate.add(vec2(this.uFrame.mul(5.588), this.uFrame.mul(3.11))));
    material.maskNode = alpha.greaterThan(threshold);
    material.colorNode = Fn(() => {
      const light = this.light(nW, positionWorld, 0.8);
      const glow = fresnel.mul(0.25).add(0.05);
      return vec4(tint.rgb.mul(light.mul(0.55).add(glow)), 1);
    })();

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'jellyfish';
    mesh.frustumCulled = false;
    this.object.add(mesh);
    this.crowds.push({ mesh, geometry, capacity: jellies.length });
  }

  // --------------------------------------------------------------- the crabs

  private buildCrabs(random: () => number, sites: readonly CreatureSite[]): void {
    const crabs: number[][] = [];
    const shells = [0xc8402a, 0xd8702a, 0xb85a3a, 0xe0a050, 0x9a4a3a];
    for (const site of sites) {
      for (let i = 0; i < 6; i++) {
        const a = random() * Math.PI * 2;
        const r = site.radius * (0.7 + random() * 0.9);
        const x = site.x + Math.cos(a) * r;
        const z = site.z + Math.sin(a) * r;
        const c = new THREE.Color(shells[Math.floor(random() * shells.length)]);
        crabs.push([
          x, seafloorHeight(x, z) - 0.015, z, 0.8 + random() * 0.8,
          random() * Math.PI * 2, random() * Math.PI * 2, 0.4 + random() * 0.9, 0.25 + random() * 0.3,
          c.r, c.g, c.b, 0,
        ]);
      }
    }
    for (let i = crabs.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [crabs[i], crabs[j]] = [crabs[j], crabs[i]];
    }
    const template = buildCrab(CREATURE_SEED).build({ aux: true });
    const geometry = toInstanced(template, crabs, ['cHome', 'cParam', 'cColor']);
    this.geometries.push(template, geometry);

    const material = this.track(new THREE.MeshBasicNodeMaterial());
    material.name = 'crab';
    material.vertexColors = true;
    material.fog = false;
    const home: Node = attribute('cHome', 'vec4');
    const param: Node = attribute('cParam', 'vec4');
    const color: Node = attribute('cColor', 'vec4');
    const part: Node = attribute('reefAux', 'float');

    const t = this.uTime;
    // Scuttle sideways and pause: a clamped sine is a walk with rests at each end.
    const cycle = t.mul(param.w).add(param.y);
    const travel = sin(cycle).mul(1.6).clamp(-1, 1).mul(param.z);
    const moving = cos(cycle).abs().mul(sin(cycle).abs().lessThan(0.62).select(float(1), float(0)));
    const yaw = param.x;
    const across = v3(cos(yaw), 0, sin(yaw).negate());
    const fwd = v3(sin(yaw), 0, cos(yaw));
    const p: Node = positionGeometry;
    // Legs lift in two alternating sets while walking.
    const isLeg = part.greaterThan(0.5).and(part.lessThan(8.5));
    const legPhase = t.mul(14).add(part.mul(Math.PI * 0.5));
    const lift = select(isLeg, sin(legPhase).max(0).mul(0.02).mul(moving).mul(p.y.smoothstep(0.06, 0.0)), float(0));
    // Claws wave a little at rest.
    const isClaw = part.greaterThan(8.5);
    const wave = select(isClaw, sin(t.mul(2.1).add(param.y)).mul(0.012), float(0));
    const local = v3(p.x, p.y.add(lift).add(wave), p.z).mul(home.w);
    const world = home.xyz.add(across.mul(travel)).add(across.mul(local.x)).add(v3(0, local.y, 0)).add(fwd.mul(local.z));
    const fade = float(1).sub(home.xyz.sub(cameraPosition).length().smoothstep(35, 55));
    material.positionNode = mix(home.xyz, world, fade);
    const n: Node = normalGeometry;
    const nWorld: Node = varying(across.mul(n.x).add(v3(0, n.y, 0)).add(fwd.mul(n.z)).normalize(), 'crabNormal');
    material.colorNode = Fn(() => {
      const pattern = lifePattern(attribute('reefSurf', 'float'), positionGeometry);
      return vec4(color.rgb.mul(pattern.y).mul(this.light(nWorld.normalize(), positionWorld)), 1);
    })();

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'crabs';
    mesh.frustumCulled = false;
    this.object.add(mesh);
    this.crowds.push({ mesh, geometry, capacity: crabs.length });
  }

  // ------------------------------------------------------- the large animals

  /** A material for a large animal, posed per object and lit like the rest. */
  private cruiserMaterial(name: string, animate: (p: Node, phase: Node) => { position: Node; normal?: Node }): THREE.MeshBasicNodeMaterial {
    const material = this.track(new THREE.MeshBasicNodeMaterial());
    material.name = name;
    material.fog = false;
    material.vertexColors = true;
    const phase: Node = (uniform(0) as Node).onObjectUpdate(({ object }: { object: THREE.Object3D }) => object.userData.flapPhase ?? 0);
    const result = animate(positionGeometry, phase);
    material.positionNode = result.position;
    const nLocal = result.normal ?? normalGeometry;
    const nWorld: Node = varying(modelWorldMatrix.mul(vec4(nLocal, 0)).xyz.normalize(), `${name}Normal`);
    material.colorNode = Fn(() => {
      const pattern = lifePattern(attribute('reefSurf', 'float'), positionGeometry);
      return vec4(v3(pattern.y).mul(this.light(nWorld.normalize().mul(faceDirection), positionWorld)), 1);
    })();
    return material;
  }

  private buildTurtles(random: () => number, plateau: readonly CreatureSite[], island: readonly CreatureSite[]): void {
    const template = buildTurtle().build({ aux: true });
    this.geometries.push(template);
    const material = this.cruiserMaterial('turtle', (p, phase) => {
      const part: Node = attribute('reefAux', 'float');
      const front = part.greaterThan(0.5).and(part.lessThan(2.5));
      const rear = part.greaterThan(2.5).and(part.lessThan(4.5));
      const side = select(part.equal(1).or(part.equal(3)), float(-1), float(1));
      const pivot = select(
        front,
        v3(side.mul(TURTLE_PIVOTS.front.x), TURTLE_PIVOTS.front.y, TURTLE_PIVOTS.front.z),
        v3(side.mul(TURTLE_PIVOTS.rear.x), TURTLE_PIVOTS.rear.y, TURTLE_PIVOTS.rear.z),
      );
      // Front flippers fly — a deep up-and-down stroke with a feathering sweep;
      // rear flippers trail and steer.
      const roll = select(front, sin(phase).mul(0.62), sin(phase.add(1.6)).mul(0.22)).mul(side);
      const sweep = select(front, cos(phase).mul(0.3), float(0)).mul(side);
      const q = p.sub(pivot);
      const cr = cos(roll);
      const sr = sin(roll);
      const x1 = q.x.mul(cr).sub(q.y.mul(sr));
      const y1 = q.x.mul(sr).add(q.y.mul(cr));
      const cs = cos(sweep);
      const ss = sin(sweep);
      const x2 = x1.mul(cs).add(q.z.mul(ss));
      const z2 = x1.negate().mul(ss).add(q.z.mul(cs));
      const moved = v3(x2, y1, z2).add(pivot);
      const isLimb = front.or(rear);
      // A slow look around for the head.
      const isHead = part.equal(TURTLE_PART.head);
      const look = sin(phase.mul(0.25)).mul(0.12);
      const headed = v3(p.x.mul(cos(look)).add(p.z.sub(0.38).mul(sin(look))), p.y, p.z.sub(0.38).mul(cos(look)).sub(p.x.mul(sin(look))).add(0.38));
      return { position: select(isLimb, moved, select(isHead, headed, p)) };
    });
    const homes = [plateau[0], island[0], plateau[Math.min(plateau.length - 1, 4)]].filter(Boolean);
    homes.forEach((site, i) => {
      const mesh = new THREE.Mesh(template, material);
      mesh.name = `turtle-${i}`;
      mesh.matrixAutoUpdate = false;
      mesh.frustumCulled = true;
      const s = 0.95 + random() * 0.35;
      mesh.scale.setScalar(s);
      this.object.add(mesh);
      this.cruisers.push({
        mesh,
        cx: site.x,
        cz: site.z,
        radius: 14 + random() * 10,
        omega: (random() < 0.5 ? -1 : 1) * (0.45 / 18),
        phase: random() * Math.PI * 2,
        hover: 1.6 + random() * 2.5,
        flapHz: 0.32,
        rank: i / homes.length,
      });
    });
  }

  private buildRays(random: () => number, plateau: readonly CreatureSite[], island: readonly CreatureSite[]): void {
    const make = (style: RayStyle, name: string, amplitude: number): THREE.MeshBasicNodeMaterial =>
      this.cruiserMaterial(name, (p, phase) => {
        const u: Node = attribute('reefAux', 'float');
        const a = u.abs();
        // A wave travelling out along the span: the tip trails the root.
        const arg = phase.sub(a.mul(1.4));
        const lift = a.pow(1.5).mul(sin(arg)).mul(style.halfSpan * amplitude);
        // Body counter-heave, so the wings do not look bolted to a fixed root.
        const heave = sin(phase).mul(style.halfSpan * amplitude * -0.08);
        const position = v3(p.x, p.y.add(lift).add(heave), p.z);
        // d(lift)/dx for the normal: the wing surface tilts with its own slope.
        const dLift = a
          .pow(0.5)
          .mul(1.5)
          .mul(sin(arg))
          .sub(a.pow(1.5).mul(cos(arg)).mul(1.4))
          .mul(amplitude)
          .mul(u.sign());
        const n: Node = normalGeometry;
        const normal = v3(n.x.sub(dLift.mul(n.y)), n.y, n.z).normalize();
        return { position, normal };
      });
    const manta = buildRay(MANTA).build({ aux: true });
    const eagle = buildRay(EAGLE_RAY).build({ aux: true });
    this.geometries.push(manta, eagle);
    const mantaMat = make(MANTA, 'manta', 0.3);
    const eagleMat = make(EAGLE_RAY, 'eagleRay', 0.34);
    const plan: { geo: THREE.BufferGeometry; mat: THREE.Material; site: CreatureSite | undefined; hz: number; hover: number; speed: number }[] = [
      { geo: eagle, mat: eagleMat, site: plateau[0], hz: 0.42, hover: 2.2, speed: 1.1 },
      { geo: manta, mat: mantaMat, site: plateau[Math.min(plateau.length - 1, 2)], hz: 0.2, hover: 6, speed: 1.3 },
      { geo: eagle, mat: eagleMat, site: island[Math.min(island.length - 1, 1)], hz: 0.45, hover: 2.5, speed: 1.0 },
      { geo: eagle, mat: eagleMat, site: plateau[Math.min(plateau.length - 1, 5)], hz: 0.4, hover: 2, speed: 1.05 },
      { geo: manta, mat: mantaMat, site: island[0], hz: 0.22, hover: 5, speed: 1.2 },
    ];
    plan.forEach((entry, i) => {
      if (!entry.site) return;
      const mesh = new THREE.Mesh(entry.geo, entry.mat);
      mesh.name = `ray-${i}`;
      mesh.matrixAutoUpdate = false;
      const radius = 22 + random() * 18;
      this.object.add(mesh);
      this.cruisers.push({
        mesh,
        cx: entry.site.x + (random() - 0.5) * 10,
        cz: entry.site.z + (random() - 0.5) * 10,
        radius,
        omega: (random() < 0.5 ? -1 : 1) * (entry.speed / radius),
        phase: random() * Math.PI * 2,
        hover: entry.hover,
        flapHz: entry.hz,
        rank: i / plan.length,
      });
    });
  }
}

/**
 * Wraps a template as an instanced geometry with one `vec4` instance attribute
 * per name, filled from rows of `4 x names.length` numbers.
 */
function toInstanced(template: THREE.BufferGeometry, rows: number[][], names: string[]): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = template.index;
  for (const [name, attr] of Object.entries(template.attributes)) geometry.setAttribute(name, attr);
  // One interleaved buffer for every per-instance attribute: WebGPU allows eight
  // vertex buffers per pipeline and the template already spends six of them.
  const stride = names.length * 4;
  const data = new Float32Array(Math.max(1, rows.length) * stride);
  rows.forEach((row, i) => data.set(row.slice(0, stride), i * stride));
  const buffer = new THREE.InstancedInterleavedBuffer(data, stride, 1);
  names.forEach((name, k) => geometry.setAttribute(name, new THREE.InterleavedBufferAttribute(buffer, 4, k * 4)));
  geometry.instanceCount = rows.length;
  return geometry;
}
