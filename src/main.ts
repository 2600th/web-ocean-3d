import * as THREE from 'three/webgpu';
import {
  Fn,
  mrt,
  normalView,
  output,
  pass,
  positionWorld,
  rtt,
  uniform,
  vec4,
  velocity,
} from 'three/tsl';
import { createRenderer, clampPixelRatio, type Backend } from './core/Renderer';
import { installThreeFixes, installWebGLFixes } from './core/threeFixes';
import { Caustics, UnderwaterParticles, UnderwaterPass } from './underwater';
import {
  AssetLoader,
  Birds,
  FishSchool,
  loadReefFish,
  IslandCanopy,
  IslandMeadow,
  KelpForest,
  ReefLife,
  SeaCreatures,
  Props,
  Remains,
  SEAFLOOR_EXTENT,
  Seafloor,
  seafloorHeight,
  ISLAND,
  Ship,
  SurfaceWetness,
  FoliageWind,
  applyGroundShadingTo,
} from './scene';
import {
  AudioSystem,
  DEFAULT_AUDIO_SCENE_PARAMS,
  SlapDetector,
  type GullFix,
  type MusicState,
} from './audio';
import { attachUiSounds } from './ui/sounds';
import { Gameplay, type GameUiSink, type SailSetting } from './gameplay';
import {
  BuoyancySystem,
  BuoyantBody,
  ShipController,
  Spray,
  Wake,
  createRadialProbes,
  type ShipControlState,
  type SprayImpact,
} from './physics';
import { Loop } from './core/Loop';
import { AdaptiveQuality, QUALITY_TIERS, type QualityTier } from './core/QualityManager';
import { OceanSimulation } from './ocean/OceanSimulation';
import { OceanMesh } from './ocean/OceanMesh';
import { DEFAULT_APPEARANCE, OceanMaterial } from './ocean/OceanMaterial';
import { Reflections } from './ocean/Reflections';
import { DEFAULT_SSR_STEPS, ScreenSpaceReflection } from './ocean/ScreenSpaceReflection';
import { OceanSampler } from './ocean/Sampler';
import { DEFAULT_SPECTRUM, significantWaveHeight } from './ocean/Spectrum';
import { AerialPerspective, Atmosphere, Clouds, Weather } from './sky';
import { Lightning } from './sky/Lightning';
import { LightingRig, applySpecularAATo, applyThinTransmissionTo, sanitizeDielectrics } from './lighting';
import { DEFAULT_VOLUMETRIC_FOG_PARAMS, VolumetricFog } from './post/VolumetricFog';
import { LensRain } from './post/LensRain';
import { ColorGrade } from './post/ColorGrade';
import { SceneBloom } from './post/Bloom';
import { DepthOfField } from './post/DepthOfField';
import { LensFlare } from './post/LensFlare';
import {
  OUTPUT_COLOR_SPACE,
  OUTPUT_TONE_MAPPING,
  OutputTransform,
} from './post/OutputTransform';
import { SpatialAA } from './post/SpatialAA';
import { PostChain } from './post/PostChain';
import { CameraDirector } from './cameras/CameraDirector';
import { CINEMATIC_BEATS, nominalShipXZ } from './cameras/Cinematic';
import { getPreset } from './presets';
import { Panel } from './ui/Panel';
import { Hud } from './ui/Hud';
import { StormQuote } from './ui/Quote';
import { TouchControls } from './ui/TouchControls';
import { DEFAULT_UI_STATE, type UiState } from './ui/types';
import { openStartGate, type StartSelection } from './ui/StartGate';
import { inject } from '@vercel/analytics';

// Vercel Web Analytics, and only where Vercel is actually serving it.
//
// `inject()` adds a script tag pointing at `/_vercel/insights/script.js`, which
// exists only on a Vercel deployment. Everywhere else — `vite preview`, the
// Playwright suite, the benchmark harness, anyone self-hosting the build — it
// is a guaranteed 404 in the console of every session.
//
// That is not merely untidy. The suite asserts that a boot produces no console
// errors, which is how it catches a missing asset or a failed shader, and a
// permanent 404 makes a real failure indistinguishable from the background
// noise. Six tests failed on it.
// Excluded by environment rather than allow-listed by domain: the deployment
// may well be on a custom domain, and an allow-list would quietly turn
// analytics off there, which is the failure nobody notices.
const localHost = /^(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)$/.test(window.location.hostname);
if (!localHost && !navigator.webdriver) {
  inject();
}

/** How the HUD plaque names each sail setting. See the gameplay block in `loadSceneContent`. */
const SAIL_LABEL: Record<SailSetting, string> = { furled: 'Furled', half: 'Half sail', full: 'Full sail' };

/** Scratch for the test-hook camera pin; the hook must not allocate either. */
const _pinPosition = new THREE.Vector3();
const _pinTarget = new THREE.Vector3();
/** Scratch for the water's key-light direction, read every frame. */
const _keyDirection = new THREE.Vector3();
/** Scratch for the camera forward axis, read every frame. */
const _keyDirection2 = new THREE.Vector3();
const _impactForward = new THREE.Vector3();
const _impactRight = new THREE.Vector3();
const _impactOffset = new THREE.Vector3();

/** The impact sounds `AudioSystem` is gaining in parallel; optional until it has them. */
interface ImpactAudio {
  hullImpact?(strength: number): void;
  waveSlap?(strength: number, pan: number): void;
}
/** Scratch for the hull occluder pushed to the underwater pass each frame. */
const _hullCenter = new THREE.Vector3();
const _hullRadius = new THREE.Vector3(1, 1, 1);
/** Scratch for the birds' ambient colour, recomputed every frame. */
const _birdShade = new THREE.Color();
const _meadowAmbient = new THREE.Color();
const _meadowWind = new THREE.Vector2();
/** Scratch for the key light's colour at its strength, for the custom-lit foliage and the smoke. */
const _keyLight = new THREE.Color();
const _smokeLight = new THREE.Color();
const _smokeSky = new THREE.Color();
/** How much of a lightning flash still goes to the grade as a frame-wide lift. */
const FLASH_GRADE_SHARE = 0.55;
/** Night multiplier on the sea's key-light glint, so the moon path reads (see `water.setSun`). */
const MOON_GLINT_BOOST = 3;
/** Scratch for the sky colours lifted by a lightning flash before the water reads them. */
const _flashZenith = new THREE.Color();
const _flashHorizon = new THREE.Color();
/** Scratch for the drawing-buffer size, read on resize and at startup. */
const _drawingBuffer = new THREE.Vector2();
/**
 * How far the sun must move before the tour re-captures the environment cube.
 *
 * 0.035 in elevation, which over the flight's full day is about twenty captures
 * a lap — a few milliseconds each, spread out, and never two in a row.
 */
const TOUR_ENV_ELEVATION_STEP = 0.035;

/**
 * Which of the dressing's scatters sway in the wind.
 *
 * Matched on the mesh name because that is where `Props` records what a scatter
 * *is* — `buildScatter` names every mesh after the scatter that placed it. The
 * alternative would be a flag threaded through the placement tables, which is
 * more code for the same information and one more thing to forget when a kind is
 * added.
 *
 * Trunks and fronds only. A rock that swayed would be memorable for the wrong
 * reason, and the fort, the jetty and the wreck are all structures.
 */
const FOLIAGE_MESHES = /^flora-/;

/** Render layer the cloud dome lives on, so the planar mirror can skip it. */
const CLOUD_LAYER = 3;

/**
 * Render layer for the life under the surface — reef, kelp, fish, creatures.
 *
 * The planar mirror renders the whole scene again from below the surface plane
 * and clips it, so without this every coral, blade and fish was drawn a second
 * time per frame to produce pixels the clip plane then threw away. The surface
 * seen from *below* takes its mirror image from the screen-space backdrop, not
 * from this reflector, so nothing visible depends on it.
 */
const UNDERWATER_LAYER = 4;

/** Scratch for the sea creatures' key light. */
const _creatureSun = new THREE.Color();

/** Haze sky colours, pulled toward the deck's grey under an overcast. */
const _hazeHorizon = new THREE.Color();
const _hazeZenith = new THREE.Color();
/** Mean dome radiance for the cloud ambient; see `App.measureSkyRadiance`. */
const _skyRadiance = new THREE.Color();
const _skyHorizon = new THREE.Color();
/** Typical dome radiance at the horizon, for the volumetric fog; see `measureSkyRadiance`. */
const _skyTypical = new THREE.Color();
const _skyBearings = [new THREE.Color(), new THREE.Color(), new THREE.Color(), new THREE.Color()];
const _fogColor = new THREE.Color();
/** Dome radiance overhead; see `measureSkyRadiance`. */
const _skyZenith = new THREE.Color();
/** The sky the water and the haze are given; see `seenSky`. */
const _seenHorizon = new THREE.Color();
const _seenZenith = new THREE.Color();
const _cloudFog = new THREE.Color();
const _cloudZenith = new THREE.Color();

/**
 * A fill tint dimmed to the dome's own radiance, once the sun is down.
 *
 * `Atmosphere.horizonColor` and `zenithColor` are the hemisphere *fill* — hues
 * at a daylight scale — and by day the dome around the horizon is about as
 * bright, so the water and the haze were tuned against them and keep them.
 * After sunset the dome falls a hundredfold and the tints do not: the sea went
 * on mirroring a mid-blue sky above a near-black one, meeting in a hard step at
 * the horizon. So, by how far the sun is down, the tint is dimmed to the dome's
 * luma (`dark` = 1 gives exactly that luma) and its hue drawn toward the
 * dome's. Never brightened.
 */
function seenSky(tint: THREE.Color, dome: THREE.Color, dark: number, out: THREE.Color): THREE.Color {
  const tintLuma = luma(tint);
  const fit = tintLuma > 1e-6 ? Math.min(1, luma(dome) / tintLuma) : 1;
  const s = 1 - dark * (1 - fit);
  return out.copy(tint).multiplyScalar(s).lerp(dome, 1 - s);
}
const _skyDirection = new THREE.Vector3();
const _litFog = new THREE.Color();
/** Rec. 709 luminance of a linear colour. */
const luma = (c: THREE.Color): number => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

/** The MRT node type `setMRT` takes; `null` clears it. */
type SceneMRT = Parameters<THREE.WebGPURenderer['setMRT']>[0];

/**
 * The render-context call depths `installStableContexts` pins the scene's own
 * passes to. Far above any depth the post chain reaches, so a pinned context
 * never shares a key with an ordinary one.
 */
const PINNED_DEPTH = { scene: 1001, reflection: 1002, shadow: 1003 } as const;

/**
 * How long `compileForFrame` waits for the pipelines it started to land. The
 * prewarm's own 30 s timeout is the one that normally bites first.
 */
const PIPELINE_SETTLE_MS = 30_000;


/** Scratch for the nominal-hull test hook. */
const _shipProbe = new THREE.Vector2();
/** Scratch for the HUD's view heading. */
const _hudView = new THREE.Vector3();

// Audio scratch: the ship's axes, the listener's right, a bird, a hull probe.
const _audioAxis = new THREE.Vector3();
const _audioRight = new THREE.Vector3();
const _audioRel = new THREE.Vector3();
const _audioBird = new THREE.Vector3();
const _audioProbe = new THREE.Vector3();

/**
 * Hull probes for wave slaps, in the hull's frame as fractions of (half length,
 * height above the waterline in metres, half beam). Bow and waist, both sides:
 * the bow is where a sea arrives first, the waist is where a beam sea slaps.
 */
const SLAP_PROBES: readonly (readonly [number, number, number])[] = [
  [0.7, 0.35, -0.75],
  [0.7, 0.35, 0.75],
  [0.05, 0.3, -1],
  [0.05, 0.3, 1],
];

/**
 * Compass bearing of a world XZ direction: degrees clockwise from north, with
 * north = −Z and east = +X (the HUD's convention, see `NavReadout`).
 */
function compassBearing(x: number, z: number): number {
  return (Math.atan2(x, -z) * 180) / Math.PI;
}

const MS_TO_KNOTS = 1.943844;

const boot = {
  root: document.getElementById('boot'),
  bar: document.getElementById('boot-bar'),
  status: document.getElementById('boot-status'),
  /** Furthest the course has been inked; the ship never sails backwards. */
  reached: 0,
  set(progress: number, message: string) {
    // Scaled rather than resized. Animating `width` puts a layout pass and a
    // paint on the main thread for every update, during the one stretch where
    // that thread is already compiling shaders and decoding assets; a transform
    // is composited and costs neither. The bar is `transform-origin: left`.
    //
    // Monotonic: stages are reported from more than one place (0.85 for the
    // controls lands before 0.82 for the first frame), and a bar that slides
    // back reads as something having gone wrong.
    const clamped = Math.max(this.reached, Math.min(1, progress));
    this.reached = clamped;
    if (this.bar) {
      (this.bar as HTMLElement).style.transform = `scaleX(${clamped})`;
    }
    // The ship on the boot chart sails to the same fraction.
    this.root?.style.setProperty('--boot-progress', String(clamped));
    if (this.status) this.status.textContent = message;
  },
  hide() {
    this.root?.classList.add('boot--hidden');
  },
  /**
   * The failure state: what happened and what to do, in plain words, with the
   * error's own text under it, smaller, for whoever needs to report it.
   */
  fail(message: string, detail = '') {
    if (this.status) {
      this.status.textContent = message;
      if (detail) {
        const small = document.createElement('small');
        small.className = 'boot__status-detail';
        small.textContent = detail;
        this.status.append(small);
      }
      this.status.classList.add('boot__status--error');
    }
  },
};

class App {
  private readonly canvas: HTMLCanvasElement;
  private readonly uiRoot: HTMLElement;

  private renderer!: THREE.WebGPURenderer;
  private backend!: Backend;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private director!: CameraDirector;

  private simulation!: OceanSimulation;
  /**
   * Surface elevation as a TSL function, bound once at build time.
   *
   * Rebound by `rebuildQualityResources` after `resize` recreates the displacement targets,
   * for the same reason the surface material re-points its cascades: the node
   * graph holds the binding, and a stale one samples a disposed texture.
   */
  private waveHeightNode!: (worldXZ: any) => any;
  private water!: OceanMaterial;
  /** Planar reflection. Null on the WebGL2 path, which keeps the analytic sky. */
  private reflections: Reflections | null = null;
  /** Screen-space reflection, layered over the planar one. WebGPU only. */
  private ssr: ScreenSpaceReflection | null = null;
  private oceanMesh!: OceanMesh;
  private sampler!: OceanSampler;

  private atmosphere!: Atmosphere;
  private aerial!: AerialPerspective;
  /**
   * The gust the island's planting sways on. Three systems share it — the
   * procedural plants, the meadow and the distant canopy cards — so the
   * handover between them is not a change of character. The plants carry their
   * own LOD chain to any range; the canopy field fades in over 90-180 m behind
   * them.
   */
  private readonly foliageWind = new FoliageWind();
  /**
   * The key light as nodes, for graphs that need it and are not lit by three's
   * own lighting pipeline — the leaf back-scatter in `groundShading`.
   *
   * Written from the same `sunLight` every frame, after `Atmosphere` has decided
   * whether the sun or the moon is driving it, so a night frond glows by
   * moonlight or not at all rather than by a sun that has set.
   */
  private readonly uKeyDirection: any = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uKeyColor: any = uniform(new THREE.Color(1, 1, 1));
  private clouds!: Clouds;
  private weather!: Weather;
  private lightning!: Lightning;
  /** Sun cascades, fire and lantern lights, the lightning's light. See `lighting/`. */
  private lighting!: LightingRig;
  /** Gulls over the play area. Owns no simulation state — see `Birds`. */
  private birds!: Birds;
  private spray!: Spray;
  /** Reef school. Parented to the scene root: its vertex stage emits world space. */
  private fish!: FishSchool;
  private kelp!: KelpForest;
  private reefLife!: ReefLife;
  private creatures!: SeaCreatures;
  private meadow!: IslandMeadow;
  private canopy!: IslandCanopy;
  private remains!: Remains;
  /**
   * Procedural audio.
   *
   * Constructed suspended and silent. Browsers refuse to start an `AudioContext`
   * without a user gesture, so `resume()` is bound to the first click or key —
   * which also means an automated run, which never gestures, is silent by
   * construction and needs no test-only mute.
   */
  private audio!: AudioSystem;
  /** Hoisted: `update` reads these every frame and must not allocate. */
  private readonly audioParams = { ...DEFAULT_AUDIO_SCENE_PARAMS };
  /** Sail state for the audio, polled from gameplay without allocating. */
  private readonly sailAudio = { active: false, sailFill: 0, sailLuff: 0 };
  private readonly slaps = new SlapDetector(SLAP_PROBES.length);
  /** Weather hysteresis for the score, so a squall at the threshold does not flip it. */
  private musicStormy = false;
  private detachUiSounds: (() => void) | null = null;

  private post!: THREE.RenderPipeline;
  private taaEnabled = false;
  /** Last frame's view-projection, for the ocean's world-space velocity. */
  private readonly prevViewProjection = uniform(new THREE.Matrix4());
  /** This frame's *unjittered* view-projection; see the ocean's velocity. */
  private readonly currViewProjection = uniform(new THREE.Matrix4());
  /** GTAO, the temporal resolve (TRAA/TAAU), sharpen, motion blur, CA, eye adaptation, grain. */
  private postChain!: PostChain;
  /** The scene pass the frame draws through; see `compileForFrame`. */
  private scenePass: ReturnType<typeof pass> | null = null;
  /** True while a live (loop-driven) frame renders; see `installAsyncPipelines`. */
  private liveFrame = false;
  /** Pipelines a live frame asked for that are still being built. */
  private pendingPipelines = 0;
  /** `compileForFrame` calls still running; see `installAsyncPipelines`. */
  private compilesInFlight = 0;
  /** Bumped when a compile's held state is released early; see `compileForFrame`. */
  private compileGeneration = 0;
  /** Restores the renderer state a running `compileForFrame` holds. */
  private frameStateRelease: (() => void) | null = null;
  /**
   * Compiles still running after their held state was released, by the render
   * context they build for: each entry puts that state back and returns the
   * undo. See `installOrphanedCompileGuard`.
   */
  private readonly orphanedCompiles = new Map<object, () => () => void>();
  private underwater!: UnderwaterPass;
  private fog!: VolumetricFog;
  private lensRain!: LensRain;
  private colorGrade!: ColorGrade;
  private bloom!: SceneBloom;
  private dof!: DepthOfField;
  private lensFlare!: LensFlare;
  private outputTransform!: OutputTransform;
  private spatialAa!: SpatialAA;
  private particles!: UnderwaterParticles;
  private caustics!: Caustics;

  private assets!: AssetLoader;
  /** The model requests, started early; see `requestSceneContent`. */
  private contentRequests: ReturnType<App['startContentRequests']> | null = null;
  /** Whether the boot bar belongs to the content load yet; see `requestSceneContent`. */
  private contentOwnsBootBar = false;
  private seafloor!: Seafloor;
  private ship: Ship | null = null;
  private props: Props | null = null;
  private buoyancy!: BuoyancySystem;
  private wake!: Wake;
  private shipBody: BuoyantBody | null = null;
  private shipControls: ShipController | null = null;
  /** Helm-mode sailing, anchor and cargo. See `src/gameplay`. */
  private gameplay: Gameplay | null = null;
  /** Rain wetting for the ship and the floating props. */
  private readonly wetness = new SurfaceWetness();

  /** Scratch for the exposed controller state; reading it must not allocate. */
  private readonly shipStateOut: ShipControlState = {
    throttle: 0,
    rudder: 0,
    speed: 0,
    forwardSpeed: 0,
    heading: 0,
  };

  /** Scratch — the frame path must not allocate. */
  private readonly previousShipPosition = new THREE.Vector3();
  private readonly chaseTarget = { position: new THREE.Vector3(), heading: 0 };

  private panel!: Panel;
  private hud!: Hud;
  private stormQuote!: StormQuote;
  /** On-screen throttle/rudder. Null on devices with a fine pointer. */
  private touchControls: TouchControls | null = null;
  private loop!: Loop;
  private adaptive!: AdaptiveQuality;

  /** See `StartSelection.adapter` — three-valued, and `undefined` is not `null`. */
  private readonly adapter: GPUAdapter | null | undefined;
  private state: UiState;
  private disposed = false;
  /** True once the async scene-content load has settled, whatever the outcome. */
  private sceneContentLoaded = false;
  /**
   * The in-flight scene load, so a tier change can wait for it.
   *
   * Models arrive after the first frames are on screen, which means the load
   * adds meshes to a live scene and compiles their pipelines while the loop is
   * running. A tier change that lands in that window would be destroying FFT,
   * shadow and reflection resources while pipeline creation for the new content
   * is still in flight — the same lifetime class as the crash the drain exists
   * to prevent, reached by a different route. Holding the promise is what lets
   * the drain rule it out instead of hoping.
   */
  private contentReady: Promise<void> | null = null;
  /** In-flight compile of LOD levels that just became drawable; see `compileLodLevels`. */
  private lodCompile: Promise<void> | null = null;
  /** True once `compileAsync` has built the initial pipeline set. */
  private shadersReady = false;
  /**
   * A `compileAsync` that outlived the prewarm timeout and is still running.
   *
   * Null whenever nothing is in flight. `drainQualityRequests` joins it before
   * tearing a tier down, because the alternative is destroying resources out
   * from under an in-flight pipeline creation — which is the exact crash this
   * file already carries two long comments about.
   */
  private pendingCompile: Promise<void> | null = null;
  /** Lazily created offscreen target for `capturePixels`. */
  private captureTarget: THREE.RenderTarget | null = null;
  /** True while `stepDeterministic` owns the wave-field readback. */
  private deterministic = false;
  /** Test-only rain rate override; null means the weather system decides. */
  private rainOverride: number | null = null;
  /**
   * Test-only master on foam and surf strength, or `null` for the weather's own.
   *
   * Exists to answer a question a single frame cannot. At a sea-level camera
   * every wave face is at grazing incidence, where Fresnel drives reflectance
   * toward one and the water legitimately returns the pale sky — so white in
   * such a frame is not evidence of foam, and tuning foam on the strength of it
   * would be tuning the wrong term. Differencing a frame against one with both
   * forced to zero separates them.
   */
  private foamOverride: number | null = null;
  /**
   * Foam strength the weather last derived, so the override can be lifted
   * without re-running `applyPreset` to find out what it was.
   */
  private weatherFoamStrength = 1;
  /**
   * The tour's multiplier on the preset's volumetric fog. 1 outside Cinematic.
   *
   * A field rather than a second `fog.setParams` call, because the density is
   * one expression assembled from three independent scalars and splitting it
   * across two call sites is how one of them silently stops applying.
   */
  private cinematicFog = 1;
  /**
   * Sun elevation at the last environment-cube capture, for the throttle in
   * `refreshTourEnvironment`.
   */
  private lastEnvElevation = Number.NaN;
  /** Tier change waiting for a safe moment. See `drainQualityRequests`. */
  private pendingQuality: QualityTier | null = null;
  /** The in-flight drain, so concurrent requests coalesce into one. */
  private qualityApply: Promise<void> | null = null;
  /**
   * True from the moment `resetDeterministic` takes the clock until the page
   * goes away. Weather that would make a baseline depend on chance (lightning)
   * reads it.
   *
   * It used to guard the pause as well: tier changes and the scene load each
   * restored the pause state they found on entry, so one that ended after a
   * capture began handed the clock back to wall time under it (the tour moved
   * 109 s during 7 s of driven stepping). Those now take `Loop.hold()`, which
   * cannot undo the capture's `setPaused(true)`.
   */
  private captureOwnsClock = false;

  constructor(canvas: HTMLCanvasElement, uiRoot: HTMLElement, options: StartSelection) {
    this.canvas = canvas;
    this.uiRoot = uiRoot;
    this.adapter = options.adapter;
    this.state = {
      ...DEFAULT_UI_STATE,
      quality: options.quality,
      forceWebGL: options.forceWebGL,
    };
  }

  async start(): Promise<void> {
    boot.set(0.05, 'Waking the graphics card…');
    const rendererOptions: Parameters<typeof createRenderer>[0] = {
      canvas: this.canvas,
      forceWebGL: this.state.forceWebGL,
      pixelRatio: window.devicePixelRatio * this.state.pixelRatio,
      // The tier decides the framebuffer before it is allocated, rather than
      // after: antialiasing and a DPR-2 buffer are four times the pixels, and
      // committing to them before anyone knows what machine this is was how a
      // low-end device paid a high-end cost it could never use.
      quality: this.state.quality,
      // The gate has already resolved this. `undefined` would mean "probe",
      // `null` means "probed, and there is no adapter" — the distinction
      // decides whether the WebGL2 fallback is taken, so it is passed through
      // rather than collapsed.
      adapter: this.adapter,
    };
    const bootstrap = await createRenderer(rendererOptions);
    this.renderer = bootstrap.renderer;
    if (bootstrap.backend === 'webgpu') installThreeFixes(this.renderer);
    else installWebGLFixes(this.renderer);
    this.backend = bootstrap.backend;
    console.info(`[ocean] renderer backend: ${this.backend}`);

    // Set once, never toggled — see `rebuildQualityResources` for why toggling it crashes.
    this.renderer.shadowMap.enabled = true;

    this.scene = new THREE.Scene();
    /**
     * Near 0.5, not 0.1, and the far plane is why.
     *
     * Depth precision goes as `z^2 / near`, so with a 40 km far plane — which
     * the 24 km ocean disc in `OceanMesh` requires and which therefore cannot
     * come in — the near plane is the only lever there is. At 0.1 the depth
     * buffer resolved 1.17 m at the island, 1.4 km out, and 54 cm at the
     * `ISLAND_STAND_OFF` the tour keeps. Two nearly coplanar surfaces at that
     * range separated by less than one depth code is a shoreline: the seafloor
     * heightfield breaks the surface there and the water writes depth, so the
     * two meet at a grazing angle inside the quantum. 0.5 divides all of it by
     * five, for one number and no format change.
     *
     * **It stays at 0.1 anyway, and the reason is worth recording because the
     * obvious argument for moving it is wrong.**
     *
     * The plan was 0.5, on the reasoning that `OceanMesh.innerRadius` is 0.6 m
     * so nothing of the sea lives closer. That is a misreading of the mesh:
     * `buildRadialGrid` fans from a *centre vertex at radius zero* out to the
     * first ring, so the surface reaches the lens no matter what the inner
     * radius says. The grid is re-centred on the camera every frame, so what a
     * near plane of `n` removes is a disc of water of radius roughly `n`
     * directly under the viewer. At 0.1 that is a 10 cm hole nobody will find;
     * at 0.5 it is half a metre, and the shot that finds it is already in the
     * gallery — `waterline` sits the eye 2 cm above the surface, which is
     * exactly the case where that disc fills the bottom of frame and the
     * per-pixel waterline has nothing left to resolve against.
     *
     * The underwater particle boxes and the rain field have no exclusion radius
     * either, so they would start popping at half a metre as well.
     *
     * The precision is a real problem and this is not the lever for it. The
     * lever is `reversedDepthBuffer`, which r185 supports and which would give
     * 0.16 mm at the island against the 1.17 m above — every depth consumer here
     * goes through `perspectiveDepthToViewZ` or `linearDepth` and both branch on
     * it, so it is genuinely a one-flag change. What stops it being taken here is
     * that the WebGL2 backend silently reverts without `EXT_clip_control`, which
     * would leave the two backends resolving depth differently, and that is a
     * commitment to make deliberately rather than at the end of a batch.
     */
    this.camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.1,
      40000,
    );
    /**
     * The opening frame, chosen against captures rather than against taste.
     *
     * This was `(0, 14, 46)` — looking along a bearing of 180 degrees, straight
     * down -Z. Three things were wrong with it, and they were all the same
     * thing: the sun in the default preset sits at azimuth 137.5 degrees and 26
     * degrees elevation, so that bearing pointed the camera 43 degrees *into* the
     * light. The sea's near faces were turned away from it, the hull the orbit
     * rig targets was a flat silhouette, and the island — which is 1.4 km out on
     * a bearing of 236 degrees — sat 56 degrees off axis, reduced to a smudge at
     * the very edge of frame. The first thing a visitor saw was a dark seascape
     * with a black cut-out in it.
     *
     * 250 degrees puts the sun 113 degrees off the view axis: behind the camera's
     * shoulder, so the hull and the swell faces are lit and modelled rather than
     * silhouetted, while enough of the specular path stays in frame that the
     * water still glitters. It also brings the island to 14 degrees off centre,
     * clear of the ship instead of behind it.
     *
     * **Rotating the camera rather than the sun is deliberate.** The two are the
     * same relative change, but `sunAzimuth` lives on the `skyPro` preset, which
     * more than twenty canonical shots are captured under; moving it would
     * rewrite every one of those baselines to fix one frame nobody photographs.
     *
     * Standoff and eye height are unchanged at 47.5 m and 14 m, so only the
     * bearing moves. `OrbitControls` derives its own spherical from this on first
     * update, which is why it is set here and not on the rig.
     */
    this.camera.position.set(44.6, 14, 16.2);

    const quality = QUALITY_TIERS[this.state.quality];

    boot.set(0.25, 'Raising the swell…');
    this.simulation = new OceanSimulation(this.renderer, {
      size: quality.fftSize,
      cascadeCount: quality.cascades,
      params: DEFAULT_SPECTRUM,
    });
    this.sampler = new OceanSampler(this.renderer, this.simulation);
    this.waveHeightNode = this.simulation.heightNode();

    // Before the seafloor, the water and the canopy, all three of which sample
    // the cloud shadow field — and every one of those bindings is baked into a
    // node graph at construction. Built here rather than with the rest of the
    // sky purely so the ordering is impossible to get wrong.
    this.clouds = new Clouds();

    boot.set(0.4, 'Laying the seabed…');
    // The seafloor must exist before the water material: the surface samples its
    // depth to shade shallows, and that binding is baked into the node graph.
    this.seafloor = new Seafloor(SEAFLOOR_EXTENT);
    // And the cloud shade before anything asks the seafloor for its combined key
    // occlusion — the canopy does, at its own construction.
    // Three octaves rather than four: the seafloor is shaded under the whole
    // ocean as well as on the island, so it is closer to the water's cost profile
    // than to a prop's. Three keeps features down to 455 m, which is a cloud
    // rather than a weather system.
    this.seafloor.setCloudShadow(this.clouds.shadowNode({ octaves: 3 }));
    this.scene.add(this.seafloor.mesh);

    // The foam buffer must exist before the water material too, and for the same
    // reason as the seafloor: the surface samples it, and that binding is baked
    // into the node graph. It used to be created during the async asset load,
    // which is why nothing could ever sample it — by the time it existed the
    // shader had already been built without it. It owns no assets, so there is
    // nothing to wait for.
    this.buoyancy = new BuoyancySystem();
    this.wake = new Wake(
      {
        derivativeTextures: this.simulation.derivativeTextures,
        tileSizes: this.simulation.tileSizes,
      },
      quality.wakeResolution,
      420,
      {
        // The same depth field the water shades its shallows from, which is what
        // lets the foam buffer know land exists at all. Passing it is what builds
        // the surf term into the node graph; without it the term is not compiled
        // and the buffer costs exactly what it did before.
        floorDepthNode: (worldPosition) => this.seafloor.depthNode(worldPosition),
      },
    );
    this.scene.add(this.wake.debugObject);

    // Planar reflection, on WebGPU only. Whether it exists is baked into the
    // surface's node graph, so it is decided here, once, from the backend — and
    // the WebGL2 path keeps the analytic sky reflection it always had, which is
    // a coherent simpler image rather than a broken richer one.
    if (this.backend === 'webgpu') {
      this.reflections = new Reflections(QUALITY_TIERS[this.state.quality].reflectionScale);
      this.scene.add(this.reflections.plane);
      this.reflections.excludeLayer(this.camera, CLOUD_LAYER);
      this.ssr = new ScreenSpaceReflection();
      this.ssr.setCamera(this.camera);
    }

    boot.set(0.5, 'Teaching the water to catch the light…');
    this.water = this.buildWaterMaterial();
    this.oceanMesh = new OceanMesh(this.water.material, {
      radialSegments: quality.meshRings,
      angularSegments: quality.meshSegments,
    });
    // How much of the wave field this mesh is entitled to be displaced by. Must
    // follow every mesh construction; see `OceanMaterial.setVertexSpacing`.
    this.water.setVertexSpacing(this.oceanMesh.spacingPerMetre);
    this.scene.add(this.oceanMesh.mesh);

    boot.set(0.7, 'Hanging the sky…');
    this.atmosphere = new Atmosphere(this.renderer, quality.environmentResolution);
    this.scene.add(this.atmosphere.mesh, this.atmosphere.sunLight, this.atmosphere.ambientLight);
    // Before anything lit compiles: the cascades and the extra lights are part
    // of every lit material's shader. See `lighting/index.ts`.
    this.lighting = new LightingRig(this.scene, this.renderer, this.camera, this.atmosphere.sunLight, quality);

    // One haze for the whole frame.
    //
    // As `scene.fogNode`, so three applies it after lighting to every material
    // that has not opted out — terrain, props, hull, canopy and water all at
    // once, with no per-material plumbing and no chance of one of them being
    // forgotten. Which is exactly what had happened: the water had an authored
    // exponential, the volumetric pass is a sea fog that clear presets switch
    // off, and the island had nothing at all.
    //
    // Assigned before any material compiles, because the fog node is part of
    // every one of their cache keys.
    this.aerial = new AerialPerspective();
    this.scene.fogNode = this.aerial.node;
    // The cloud layer draws with `fog = false` on a camera-locked dome, so it
    // never reaches `setupFog` and has to be handed the same function directly.
    this.clouds.setAerialPerspective(this.aerial);

    this.scene.add(this.clouds.mesh);
    // Clouds stay out of the planar mirror. Through a roughness-blurred, half-
    // resolution, haze-tinted reflection a cumulus came back as a khaki smear
    // lying on the sea — the pale blotches in the near field. SSR still
    // reflects the clouds that are on screen, sharply and in the right place.
    this.clouds.mesh.layers.set(CLOUD_LAYER);
    this.camera.layers.enable(CLOUD_LAYER);

    this.weather = new Weather();
    this.scene.add(this.weather.object);
    this.lightning = new Lightning();
    this.scene.add(this.lightning.object);

    // Birds go in with the sky rather than with the props: they are lit by the
    // atmosphere and nothing else, and they never touch the water.
    this.birds = new Birds(quality.birds);
    this.scene.add(this.birds.object);

    // Built here with the rest of the always-present scene rather than with the
    // ship, so its shader is compiled by the boot prewarm whether or not a hull
    // ever loads. Its probes are attached later, if one does.
    this.spray = new Spray();
    this.spray.setStrength(quality.spray);
    this.scene.add(this.spray.mesh);
    // One detector, several consumers: the impact a probe registers drives the
    // spray it draws, the camera shake, the audio, the foam it leaves on the
    // water and the water it puts on the deck. See `onSprayImpact`.
    this.spray.onImpact((impact) => this.onSprayImpact(impact));

    this.fish = new FishSchool(quality.fish);
    this.scene.add(this.fish.object);

    this.kelp = new KelpForest(quality.kelp);
    this.scene.add(this.kelp.object);

    // Grass, placed on the GPU from the seafloor's own heightfield node rather
    // than scattered as instances — the island is 0.8 km2 and a scatter dense
    // enough to read as a sward is a million draws. See `src/scene/Meadow.ts`.
    this.meadow = new IslandMeadow(
      quality.meadow,
      (worldPosition) => this.seafloor.heightNode(worldPosition),
      {
        // Cloud shade only, for the same reason the dressing takes it: this is
        // evaluated per *blade* vertex, and High plants thirty-eight thousand of
        // them. A twenty-four-step heightfield march there is roughly two hundred
        // texture fetches per vertex before the cloud term, which is a poor place
        // to replicate a march the ground under it has already done.
        //
        // The visible cost is a terminator: grass inside the hill's own shadow
        // stays lit while the ground it stands on darkens. The field is a 150 m
        // square around the camera, so that boundary is only ever in frame when
        // the viewer is standing on one — and the sward is a near-camera detail
        // whose own self-shadow gradient dominates it at that range.
        sunOcclusion: (worldPosition) => this.seafloor.cloudShadowNode(worldPosition),
      },
    );
    this.scene.add(this.meadow.object);

    // Distant canopy. The island's trees are LOD2 specks from the play area and
    // the biome in the terrain colour cannot give a dome a broken edge, so past
    // ninety metres the forest becomes billboards. See `src/scene/Canopy.ts`.
    this.canopy = new IslandCanopy(
      quality.canopy,
      (worldPosition) => this.seafloor.heightNode(worldPosition),
      {
        // The forest takes the cloud deck's shade and the hill's own shadow.
        // Without this it goes on being fully lit across a hillside that now has
        // a shaded face — a canopy floating over its own shadow, which reads
        // worse than the flat lighting it replaced.
        sunOcclusion: (worldPosition) =>
          this.seafloor.keyShadowNode(worldPosition),
      },
    );
    this.scene.add(this.canopy.object);

    // The wreck's owner. Poly Haven publishes no skeleton and every CC0 source
    // that does is low-poly stylised and would sit badly against photoscanned
    // rock — so it is built, which is also how this project solves gulls and
    // the pelagic fish. The island's palms are built too, in `scene/flora`, and
    // scattered by `Props.dressIsland`.

    this.remains = new Remains();
    // Above the tideline on the cove's own bearing, but off to one side of the
    // landing: a body found on the way somewhere, rather than staged in the
    // middle of the beach where it would read as a signpost. The class seats
    // itself on whatever ground is under the point, so this survives the island
    // being reshaped again.
    {
      const bearing = 0.52;
      const radius = ISLAND.radius * 0.94;
      this.remains.place(
        ISLAND.x + Math.cos(bearing) * radius,
        ISLAND.z + Math.sin(bearing) * radius,
        bearing + 1.9,
      );
    }
    this.scene.add(this.remains.object);

    this.audio = new AudioSystem({
      volume: this.state.volume,
      musicVolume: this.state.musicVolume,
      quality: this.state.quality,
    });
    this.audio.resumeOnGesture();
    // Gull cries come from a real bird in the flock, placed by distance and
    // bearing from the listener. Tries a few birds; a cry with none in
    // earshot is skipped rather than invented.
    this.audio.setGullLocator((out: GullFix) => this.locateGull(out));

    boot.set(0.8, 'Filling the deep…');
    this.underwater = new UnderwaterPass();
    // Required, not optional: a post pass is drawn with the post-processor's own
    // orthographic quad camera, so the built-in camera nodes resolve to that quad
    // rather than the scene camera. Without this the depth buffer cannot be
    // linearised and the sun cannot be projected to screen space.
    this.underwater.setCamera(this.camera);

    this.particles = new UnderwaterParticles(quality.underwaterParticles);
    this.scene.add(this.particles.object);


    this.caustics = new Caustics(quality.causticsResolution);
    // Rebuilds the sand shader once, so it happens here at setup and never in a
    // frame path.
    this.seafloor.setCaustics(this.caustics.intensityNode(positionWorld));
    // The two occlusion terms the land had neither of: its own hillside, and the
    // cloud deck overhead. Both rebuild the sand shader, so like the caustics
    // they are wired here at setup and never from a frame path.
    this.seafloor.setKeyLight(this.atmosphere.sunLight);
    // The volumetric march picks its caustics mip level from this, so it has to
    // be told after the field exists — see `uCausticsTexel`.
    this.underwater.setCausticsTexelSize(this.caustics.extent / this.caustics.resolution);

    // The procedural reef. Built after the caustics because it takes the same
    // ground treatment the props do — caustics on every coral, the cloud deck's
    // shade, contact darkening at the base — and that binding is compiled in.
    this.reefLife = new ReefLife(quality.reefLife, {
      shading: {
        light: this.atmosphere.sunLight,
        keyShadow: (worldPosition) => this.seafloor.cloudShadowNode(worldPosition),
        caustics: (worldPosition) => this.caustics.intensityNode(worldPosition),
        groundHeight: (worldPosition) => this.seafloor.heightNode(worldPosition),
      },
    });
    this.scene.add(this.reefLife.object);

    // The animals that live on it, homed on the same sites.
    this.creatures = new SeaCreatures(quality.seaCreatures, {
      sites: this.reefLife.sites,
      caustics: (worldPosition) => this.caustics.intensityNode(worldPosition),
    });
    this.scene.add(this.creatures.object);

    // Everything that lives under the surface goes on its own layer, kept out of
    // the planar mirror — see `UNDERWATER_LAYER`.
    this.camera.layers.enable(UNDERWATER_LAYER);
    this.reflections?.excludeLayer(this.camera, UNDERWATER_LAYER);
    for (const root of [
      this.reefLife.object,
      this.creatures.object,
      this.kelp.object,
      this.fish.object,
      this.particles.object,
    ]) {
      root.traverse((node) => node.layers.set(UNDERWATER_LAYER));
    }

    // `RenderPipeline`, not the `PostProcessing` alias: the latter is deprecated
    // as of r183 and warns on every boot.
    this.post = new THREE.RenderPipeline(this.renderer);
    // Temporal reprojection AA, WebGPU only and opt-out with `?taa=0`.
    //
    // The water's remaining aliasing — thresholded foam at mid distance, glitter
    // spikes, the rigging — is sub-pixel signal that no spatial filter can
    // recover; the shader comments say so in as many words. TRAA integrates it
    // over frames instead. It also replaces 4x MSAA on the scene pass, which is
    // a bandwidth saving that pays for most of the resolve.
    this.taaEnabled =
      this.backend === 'webgpu' && new URLSearchParams(window.location.search).get('taa') !== '0';
    this.clouds.setTemporalJitter(this.taaEnabled);
    const scenePass = this.taaEnabled
      ? pass(this.scene, this.camera, { samples: 0 })
      : pass(this.scene, this.camera);
    this.scenePass = scenePass;
    if (this.taaEnabled) {
      // View-space normals for GTAO. Without them it reconstructs normals from
      // depth, which breaks down on distant ground seen at a grazing angle: the
      // beach came out as a grid of stripes that TRAA could not average away.
      scenePass.setMRT(mrt({ output, velocity, normal: normalView }));
      this.clouds.setPreviousViewProjection(this.prevViewProjection, this.currViewProjection);
      // The ocean mesh is re-centred under the camera every frame, so its model
      // matrix moves with the eye while the surface it draws is anchored to the
      // world. Stock velocity would read that re-centring as motion and smear
      // the whole sea; reproject its world position through last frame's
      // camera instead.
      //
      // Both ends are *unjittered*. TRAA jitters the projection only for the
      // duration of the scene pass and hands the stock velocity its original
      // matrix, so velocity is jitter-free everywhere else; using the jittered
      // in-shader projection here put up to half a pixel of Halton offset into
      // every sea pixel's motion, which resampled (blurred) its history each
      // frame and raised the current-frame weight enough to undo the AA on the
      // very glints it was added for. Both matrices are captured outside the
      // pipeline render — see `renderFrame`.
      const prevViewProjection = this.prevViewProjection;
      const currViewProjection = this.currViewProjection;
      this.water.material.mrtNode = mrt({
        velocity: Fn(() => {
          const world = vec4(positionWorld, 1);
          const current = (currViewProjection as any).mul(world);
          const previous = (prevViewProjection as any).mul(world);
          // `.z = 1` marks water for the AO mask in `PostChain`; every other
          // material's vec2 velocity is widened to (x, y, 0, 1).
          return vec4(
            current.xy.div(current.w.max(1e-4)).sub(previous.xy.div(previous.w.max(1e-4))),
            1,
            1,
          );
        })(),
      });
    }
    // Must be the depth *texture* node: both passes linearise it to find where
    // the scene stops, which is what bounds their marches.
    const sceneDepth = scenePass.getTextureNode('depth');
    // AO, the temporal resolve and the post-resolve stages live in `PostChain`.
    this.postChain = new PostChain({
      renderer: this.renderer,
      camera: this.camera,
      scenePass,
      depth: sceneDepth,
      temporal: this.taaEnabled,
    });
    const sceneBeauty: any = this.postChain.buildResolve();

    this.fog = new VolumetricFog();
    // Required: a post pass draws with the post-processor's own orthographic
    // quad camera, so the scene camera has to be handed over explicitly.
    this.fog.setCamera(this.camera);
    // Crepuscular rays above the water. The march had every ingredient except an
    // occlusion term; this is it, and the deck casting it is the same field the
    // clouds are drawn from, so a shaft lands under the cloud that made it.
    //
    // One sample rather than three. This is evaluated once per march cell and
    // Max marches 56 of them, so the surface variant would be a hundred and
    // sixty-eight noise evaluations a pixel for a feature that is by nature
    // low-frequency and integrated along the ray anyway.
    this.fog.setSunOcclusion(this.clouds.shadowNode({ samples: 1, octaves: 2 }));

    // Fog wraps the underwater pass, not the other way round.
    //
    // `UnderwaterPass.build` samples its colour input and reads `uvNode`, so it
    // has to consume the raw texture node; the fog accepts either that or an
    // already-composited colour. Above water the underwater pass is a bit-exact
    // pass-through, so the ordering only matters below the surface — where the
    // fog is faded out anyway, since there is no atmosphere down there.
    this.lensRain = new LensRain();

    const graded = this.fog.build(
      this.underwater.build(
        sceneBeauty,
        sceneDepth,
        // The shafts are an integral of this field along the view ray, so
        // passing it here is what makes them and the seafloor pattern the same
        // light.
        (worldPosition) => this.caustics.intensityNode(worldPosition),
        // And this is what makes the waterline per-pixel: the pass can ask where
        // the surface actually is along each eye ray instead of assuming a plane.
        this.waveHeightNode,
        // Cloud shade reaches under the surface too — the light the caustics
        // field refracts is the light that got past the deck.
        this.clouds.shadowNode(),
      ),
      sceneDepth,
    );

    // Lens rain comes last, and needs the graded image resolved to a texture
    // first.
    //
    // Both it and the underwater pass re-sample the image at displaced
    // coordinates, so both need a *texture* node rather than a composited colour
    // — and a composited node is exactly what each produces. They therefore
    // cannot be nested directly in either order, and putting the droplets first
    // was not merely wrong but silently fatal: the underwater pass called
    // `.sample()` on something that has no such method and the whole frame came
    // out black.
    //
    // `rtt` resolves the chain into a texture at the cost of one fullscreen
    // pass. That is the right place to spend it: droplets are lenses, and what
    // they should be refracting is the finished image — fog, grade and all — not
    // the raw scene behind it.
    //
    // All of this lives inside `outputNode`, so the droplets are part of the
    // rendered frame while the DOM HUD stays crisp on top of them.
    //
    // The grade goes *inside* that `rtt`, which is to say before the lens rather
    // than after it. Two reasons, and both are about the ordering being physical
    // rather than convenient. A colourist grades the image a camera recorded, not
    // the water on its front element — so the droplets should be refracting a
    // graded frame, which is what LensRain's own note asks for when it says the
    // thing it refracts should be the finished image. And the grade is in linear:
    // `outputColorTransform` applies ACES after `outputNode`, so this shapes what
    // the tone curve then compresses. See `src/post/ColorGrade.ts`.
    this.bloom = new SceneBloom();
    this.colorGrade = new ColorGrade();

    // Resolved to a texture before the bloom, and that costs a fullscreen pass
    // worth having. `BloomNode` evaluates its input inside its own high-pass
    // material, and the additive base evaluates it a second time — so handing it
    // a composited expression computes that expression twice per frame. Once the
    // depth-of-field gather is in this position that is up to 32 texture taps
    // paid for twice; resolving once and sampling a texture is strictly cheaper
    // from the first tap onward.
    // Depth of field first, because it is the only stage that is a property of
    // the *lens* rather than of the light: everything after it — the bloom's
    // spill, the flare's ghosts, the grade — happens to an image that has
    // already been focused, which is the order a camera does it in. Blooming
    // first and defocusing afterwards would smear a sharp glow across an
    // out-of-focus background.
    this.dof = new DepthOfField();
    // Required, like the fog's and the underwater pass's: a post pass draws with
    // the post-processor's own orthographic quad camera, so the scene camera has
    // to be handed over explicitly or the depth buffer cannot be linearised.
    this.dof.setCamera(this.camera);
    this.dof.setFrameHeight(this.renderer.getDrawingBufferSize(_drawingBuffer).y);
    const gradedTexture = rtt(graded as THREE.Node);
    // Eye adaptation meters this — fog and the underwater pass included, so the
    // dip below the surface is what it reacts to — and applies its gain ahead
    // of the bloom, whose threshold assumes the preset's exposure.
    this.postChain.buildExposure(gradedTexture);
    const focused = this.dof.build(gradedTexture, sceneDepth);

    const bloomed = this.bloom.build(rtt(this.postChain.applyExposure(focused) as THREE.Node));

    // Flare after the bloom and before the grade: it is light arriving at the
    // sensor through the glass, so it belongs with the other things the lens
    // does to light, and it must be *graded* rather than sitting on top of the
    // grade. Purely additive and it never re-samples the image, so unlike the
    // depth of field it imposes no resolve of its own.
    this.lensFlare = new LensFlare();
    this.lensFlare.setCamera(this.camera);
    const flared = this.lensFlare.build(bloomed, sceneDepth);

    // Tone mapping, sRGB and the dither are ours now, not the renderer's.
    //
    // `outputColorTransform` is switched off below and `OutputTransform` does
    // the same three steps `RenderPipeline` would have — using `renderOutput`,
    // which is literally the node it would have used — plus a dither. That
    // fourth step is the whole reason for the takeover: quantisation to 8 bits
    // is the last thing that happens to a frame, so while every node ran before
    // the tone curve there was nowhere late enough to stand. Sky and open water
    // are big smooth gradients and they are most of every gallery frame, which
    // is the content that bands.
    //
    // The lens rain still comes before it, so the droplets refract linear scene
    // light rather than a tone-mapped picture — which is both what its own
    // comment asks for and what a drop of water on glass actually does.
    this.outputTransform = new OutputTransform();
    this.spatialAa = new SpatialAA();
    // TRAA already resolved the edges; FXAA on top only softens them again.
    if (this.taaEnabled) this.spatialAa.setStrength(0);
    this.post.outputColorTransform = false;
    // Tone map, anti-alias, dither — in that order, and none of the three can
    // move. See `SpatialAA` for why the filter has to sit between the other two
    // rather than at either end of the chain.
    this.post.outputNode = this.outputTransform.applyDither(
      this.postChain.applyGrain(
        this.spatialAa.build(
          this.outputTransform.buildDisplay(
            this.lensRain.build(rtt(this.colorGrade.build(flared) as THREE.Node)),
            OUTPUT_TONE_MAPPING,
            OUTPUT_COLOR_SPACE,
          ),
        ),
      ),
    ) as THREE.Node;

    boot.set(0.85, 'Rigging the helm…');
    this.director = new CameraDirector({
      camera: this.camera,
      domElement: this.canvas,
      surfaceHeight: (x, z) => this.sampler.height(x, z),
    });

    this.buildUi();
    this.applyPreset();
    // Constructors already used the selected tier for resource sizes. Apply
    // the remaining live settings once, before any asset or shader work.
    this.applyQualitySettings(this.state.quality);

    this.adaptive = new AdaptiveQuality(55, (tier) => {
      console.info(`[ocean] adaptive quality: dropping to "${tier}"`);
      this.state.quality = tier;
      this.panel.setState({ quality: tier });
      this.requestQuality(tier);
    });

    window.addEventListener('resize', this.onResize);

    // `renderAsync`, not `render`. The synchronous form queues GPU work and
    // returns immediately, so `Loop`'s `await` completed before the frame did:
    // `frameMs` was timing the submit rather than the work, and the `inFlight`
    // guard that is supposed to stop unbounded queueing never actually held
    // anything back. It also left the deterministic capture path racing a render
    // that was still in flight, which is what made repeated captures of an
    // identical world disagree.
    this.loop = new Loop((_dt, _elapsed, live) => this.renderFrame(live));
    this.installAsyncPipelines();
    this.installStableContexts();
    this.installOrphanedCompileGuard();
    this.loop.add(this.update);

    // Models load **behind the boot overlay**, not after it.
    //
    // This used to start after `loop.start()`, on the reasoning that the ocean
    // is the headline and nobody should wait on 26 MB of ship textures before
    // seeing anything. What that traded away was worse than what it bought: the
    // ship and the dressing arrive seconds into the experience, and the reveal
    // at the end of `loadSceneContent` has to stop the loop to compile roughly
    // sixty models' materials without a frame being drawn between the reveal and
    // the compile. A viewer already flying the camera sees that as a hitch — and
    // it lands at the exact moment the scene finally has something in it, which
    // is the worst possible time to drop frames.
    //
    // Waiting here costs a longer boot screen, which is a progress bar doing its
    // job, and buys a session with no compile pause in it. The overlay says what
    // it is waiting for so the extra seconds read as loading rather than as a
    // hang.
    //
    // Kept in a field, not dropped. A tier change destroys GPU resources and
    // this task creates them, so the two must never overlap;
    // `drainQualityRequests` awaits it.
    // Published *before* the content load, and that ordering is load-bearing.
    //
    // Every object this exposes by reference — renderer, scene, camera, loop,
    // the post stages — exists by now; only the ship and the dressing do not,
    // and those are reached through closures. What moving the load ahead of
    // `loop.start()` accidentally also moved was this line, which put the whole
    // of a sixty-model parse and compile in front of the hook being installed at
    // all. On WebGPU that is absorbed; on the WebGL2 fallback it ran past the
    // 60 s the harness waits for `__ocean` to appear, so the fallback path
    // failed to boot as far as every test was concerned.
    //
    // Nothing is weakened by publishing early: `isReady()` gates on
    // `sceneContentLoaded`, so a caller that waits for readiness still gets a
    // fully dressed scene. The hook now appears when the *renderer* is up, which
    // is what it is for, and the loading it used to hide behind is a wait the
    // harness can observe rather than a silence it has to time out on.
    this.exposeTestHooks();

    // The model requests go out now, under the prewarm, rather than after it.
    // See `requestSceneContent`.
    void this.requestSceneContent();

    boot.set(0.82, 'Readying the first view…');
    await this.prewarm();
    this.loop.start();
    boot.set(0.84, 'Bringing the ship and island in…');
    window.setTimeout(() => boot.hide(), 350);

    boot.set(0.86, 'Bringing the ship and island in…');
    this.contentReady = this.loadSceneContent();
    await this.contentReady;

    // No second pass here. This used to deal the LODs, mark every level the
    // deal handed back as compiled, and prewarm again — written for a boot that
    // loaded the content before `loop.start()`. The loop now starts first, so by
    // this line frames are running, and marking a level compiled *before* the
    // compile under it lets a frame draw it with no pipeline: the inline compile
    // `compileLodLevels` exists to prevent. The deal now happens in
    // `loadSceneContent`, ahead of its paused compile, and any level the camera
    // deals itself after that takes the path every other late level does — the
    // frame update hands it to `compileLodLevels`.

    boot.set(0.98, 'Island in sight…');
    boot.set(1, 'Fair winds');
  }

  /**
   * Advances the world by `steps` increments of `dt`, awaiting a fresh wave-field
   * readback before each one.
   *
   * The interactive path fires the sampler readback and does not wait for it, so
   * buoyancy runs on data that is a frame or two old and *how* old depends on the
   * machine. That is invisible in motion and fatal to reproducibility: the hull
   * ends up somewhere slightly different every run, and so does its wake. Here we
   * pay the stall, once per step, and get the same hull position every time.
   */
  /**
   * One pipeline render, bracketed by the view-projections the ocean's velocity
   * needs. Both are read *outside* the render, where TRAA has not yet applied
   * (or has already cleared) its jitter.
   */
  private async renderFrame(live = false): Promise<void> {
    this.camera.updateMatrixWorld();
    this.currViewProjection.value.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
    this.liveFrame = live;
    try {
      await this.post.renderAsync();
    } finally {
      this.liveFrame = false;
    }
    this.prevViewProjection.value.copy(this.currViewProjection.value);
  }

  /**
   * Builds the rest of a released compile under the state it was started with.
   *
   * `compileAsync` cannot be cancelled, and it builds each object's nodes
   * later, a task at a time, reading the target and the MRT from the renderer
   * as it goes (`NodeMaterial.setup`). A prewarm that times out has to give
   * the renderer back so a frame can draw, and every build after that saw no
   * target and no MRT: one output for a three-attachment pass, cached against
   * the scene pass's context for the rest of the session. Frames cannot simply
   * borrow the renderer around a held state either — tried, and the compile's
   * open context collided with the frame's ("RenderPassEncoder was already
   * ended").
   *
   * So a released compile's remaining builds for its context are made
   * synchronously here, with its state put back for exactly the length of
   * each build: no frame can run inside one, and none is built against the
   * wrong state. A main-thread cost per object, on a path that only a device
   * slow enough to time out the prewarm ever takes.
   */
  private installOrphanedCompileGuard(): void {
    if (this.backend !== 'webgpu') return;
    const nodes = (this.renderer as unknown as {
      _nodes: {
        getForRender(renderObject: unknown): unknown;
        getForRenderAsync(renderObject: unknown): Promise<unknown>;
      };
    })._nodes;
    const getForRenderAsync = nodes.getForRenderAsync.bind(nodes);
    nodes.getForRenderAsync = (renderObject) => {
      const hold = this.orphanedCompiles.get((renderObject as { context: object }).context);
      if (hold === undefined) return getForRenderAsync(renderObject);
      const restore = hold();
      try {
        return Promise.resolve(nodes.getForRender(renderObject));
      } finally {
        restore();
      }
    };
  }

  /**
   * Gives the scene pass, the reflection and the shadow map one render context
   * each, whatever depth they are drawn at.
   *
   * three keys a render context on the target's format, the MRT and the
   * renderer's call depth, and keys render objects on the context. The frame
   * draws the scene pass five calls deep — inside the underwater, fog, TRAA and
   * post passes — and the reflection and shadow map six; `compileAsync` asks
   * for depth 0, and the renders it sets off from `updateBefore` for depth 1.
   * So every render object the prewarm built belonged to a context no frame
   * ever used, and the first frame to draw the content built them all again:
   * measured on aaa-pass, a second and a half of node building on the main
   * thread for the scene pass alone, plus a pipeline per instanced mesh, whose
   * instance buffer is a new node with a new id in the WGSL every time.
   *
   * Depth is in the key so that a render nested inside another render into a
   * target of the same format does not take over the outer one's context
   * mid-pass. None of these three targets is ever drawn into from inside a
   * render of itself, and each gets a depth of its own, so pinning them gives
   * that up nowhere. Keyed on the targets themselves, so a change to the post
   * chain that moves them deeper or shallower changes nothing here.
   */
  private installStableContexts(): void {
    if (this.backend !== 'webgpu') return;
    const contexts = (this.renderer as unknown as {
      _renderContexts: { get(target: unknown, mrt?: unknown, depth?: number): unknown };
    })._renderContexts;
    const get = contexts.get.bind(contexts);
    contexts.get = (target, mrtArg, depth) => {
      // Depth -1 is `Renderer.clear()` asking for a context of its own, apart
      // from any render in progress on the same target. Pinning it would hand
      // a clear the reflection's live context (the reflector clears instead of
      // rendering with the eye under water).
      if (depth !== -1 && target !== null && target !== undefined) {
        if (target === this.scenePass?.renderTarget) {
          return get(target, mrtArg, PINNED_DEPTH.scene);
        }
        if (this.reflections?.ownsTarget(target)) {
          return get(target, mrtArg, PINNED_DEPTH.reflection);
        }
        if (this.lighting?.shadows.ownsTarget(target) || target === this.atmosphere?.sunLight.shadow.map) {
          return get(target, mrtArg, PINNED_DEPTH.shadow);
        }
      }
      return get(target, mrtArg, depth);
    };
  }

  /**
   * Makes a live frame build any pipeline it is missing asynchronously, and
   * skip that object until it is ready, rather than stopping to build it.
   *
   * The prewarm is meant to have built everything first, and with
   * `compileForFrame` it builds the main pass. But the frame draws the scene
   * more than once — the planar reflection renders it again from a mirrored
   * camera into a target of its own, the sun's shadow map renders the casters,
   * and post is a dozen full-screen passes — and each of those is a render
   * context with render objects (and, for an instanced mesh, pipelines) of its
   * own. Any of them a frame met unbuilt was built *synchronously*, on the GPU
   * process's main thread, where FXC takes 0.1-1.3 s a pipeline; the command
   * buffer filled up behind it and Blink blocked in `WaitForToken`, the main
   * thread frozen — no rAF, no input, no CSS transitions — for the whole batch.
   *
   * three already does the rest: `_renderObjectDirect` draws an object only
   * once `Pipelines.isReady` says its pipeline exists, and handing
   * `getForRender` a promise list is how `compileAsync` gets
   * `createRenderPipelineAsync`. So a live frame hands it one too. Whatever is
   * not ready is left out of that frame and drawn from the first frame after
   * its pipeline lands, which TRAA fades in rather than pops.
   *
   * Only live frames. `step` and the capture path draw frames that have to be
   * complete — the deterministic harness compares them — so they keep the
   * synchronous build, and `isReady` waits for the asynchronous builds to
   * drain, so a harness that waits for it never captures a frame with a hole.
   */
  private installAsyncPipelines(): void {
    if (this.backend !== 'webgpu') return;
    const internals = this.renderer as unknown as {
      _pipelines: {
        getForRender(renderObject: unknown, promises?: Promise<unknown>[] | null): unknown;
      };
      backend: {
        createRenderPipeline(renderObject: unknown, promises: Promise<unknown>[] | null): void;
      };
    };
    const pipelines = internals._pipelines;
    const getForRender = pipelines.getForRender.bind(pipelines);
    pipelines.getForRender = (renderObject, promises = null) => {
      if (promises === null && (this.liveFrame || this.compilesInFlight > 0)) {
        return getForRender(renderObject, []);
      }
      return getForRender(renderObject, promises);
    };

    // Every asynchronous build, the compile's and the frame's, is counted
    // until its pipeline exists — `isReady` and `prewarm` wait on the count —
    // and the compile is not made to wait for any one of them.
    //
    // `compileAsync` awaits each object's pipeline before it starts the next,
    // so a compile is the sum of its FXC builds, one after another, while
    // Dawn has a worker pool for exactly this. Handing it a settled promise
    // lets the compile do what only it can do — build every object's nodes,
    // a task at a time — and start every pipeline as it goes, so they build in
    // parallel; `prewarm` then waits for the count to reach zero. Measured at
    // High on the FXC path: the content compile from ~31 s to ~4 s.
    //
    // It also takes the compile off a hook it used to hang on. A pipeline
    // Dawn already holds (a second mesh with the same material) comes back
    // from its cache with no worker task, and its callback waits for the next
    // device tick, which Chrome only runs while GPU work is in flight — under a
    // paused loop, never. Traced on aaa-pass: a fish pipeline asked for at
    // 2.3 s resolved at 31.4 s, when the prewarm's 30 s timeout let a frame
    // submit; every prewarm timeout this project has logged was that.
    // `compileForFrame` keeps the device ticking while it waits for the count;
    // see `pipelinesSettled`.
    const backend = internals.backend;
    const createRenderPipeline = backend.createRenderPipeline.bind(backend);
    backend.createRenderPipeline = (renderObject, promises) => {
      const before = promises?.length ?? 0;
      createRenderPipeline(renderObject, promises);
      if (!promises || promises.length === before) return;
      const built = promises[promises.length - 1];
      this.pendingPipelines++;
      void built.finally(() => {
        this.pendingPipelines--;
      });
      promises[promises.length - 1] = Promise.resolve();
    };
  }

  /**
   * Resolves once every asynchronous pipeline build has landed, or after
   * `timeoutMs`, whichever is first; true if they all landed.
   *
   * While it waits it keeps submitting an empty command buffer and asking for
   * the queue fence, which is what gets Chrome to tick the device and deliver
   * the callbacks of pipelines Dawn answered from its cache (see
   * `installAsyncPipelines`) when no frame is running to do it.
   */
  private async pipelinesSettled(timeoutMs: number): Promise<boolean> {
    const queue = (this.renderer as unknown as {
      backend?: { device?: { queue?: GPUQueue } };
    }).backend?.device?.queue;
    const deadline = performance.now() + timeoutMs;
    while (this.pendingPipelines > 0 && !this.disposed) {
      if (performance.now() > deadline) return false;
      if (queue) {
        queue.submit([]);
        await Promise.race([
          queue.onSubmittedWorkDone().catch(() => undefined),
          new Promise((resolve) => setTimeout(resolve, 50)),
        ]);
      }
      await new Promise((resolve) => setTimeout(resolve, 16));
    }
    return true;
  }

  /**
   * Restarts temporal accumulation from the next frame: jitter sequence back to
   * its start, history re-seeded from the current frame, and no motion assumed
   * since the last one. The deterministic capture path calls it, because
   * otherwise the final frame's sub-pixel offset — and the history under it —
   * depend on how many free-running frames the session drew before the reset.
   *
   * `PostChain.resetTemporal` restarts TRAA, the GTAO rotation and the eye
   * adaptation together.
   */
  private resetTemporal(): void {
    this.postChain.resetTemporal();
    this.camera.updateMatrixWorld();
    this.prevViewProjection.value.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
  }

  /**
   * Fans one hull or float impact out to everything that reacts to it.
   *
   * The audio calls go through an optional-method view of `AudioSystem` because
   * `hullImpact` and `waveSlap` are being added to it in parallel; this compiles
   * and runs silently until they exist, and plays them as soon as they do.
   */
  private onSprayImpact(impact: SprayImpact): void {
    const s = impact.strength;
    const p = impact.position;
    if (!impact.hull) {
      // A buoy or a barrel: a small patch of foam where it came down.
      this.wake.splash(p.x, p.z, 1.2, 0.35 + 0.4 * s);
      return;
    }

    this.gameplay?.reportImpact(impact.kind, s);

    // Stereo position of the impact relative to the listener's view, -1..1.
    this.camera.getWorldDirection(_impactForward);
    _impactRight.set(-_impactForward.z, 0, _impactForward.x).normalize();
    _impactOffset.copy(p).sub(this.camera.position);
    const distance = Math.max(1, _impactOffset.length());
    const pan = Math.max(-1, Math.min(1, _impactOffset.dot(_impactRight) / distance));
    // Heard less the further away it is, not at all past a few hundred metres.
    const audible = s * Math.min(1, 30 / distance);
    const audio = this.audio as unknown as ImpactAudio | undefined;
    if (audible > 0.02) {
      if (impact.kind === 'slap') audio?.waveSlap?.(audible, pan);
      else audio?.hullImpact?.(audible);
    }

    // White water where it hit, and water on the deck when it came aboard.
    const radius = impact.kind === 'wash' ? 7 : impact.kind === 'slap' ? 3.5 : 4.5;
    this.wake.splash(p.x, p.z, radius, 0.4 + 0.6 * s);
    // The ship's channel only: green water on the deck does not wet the island.
    if (impact.kind === 'wash') this.wetness.douse(0.55 * s, 'ship');
    else if (impact.kind === 'slam' && s > 0.5) this.wetness.douse(0.08 * s, 'ship');
  }

  /**
   * Waits until no LOD compile or tier change holds the renderer's frame state.
   *
   * Both pause the live loop for the length of their compile, because the
   * compile holds the target and MRT it builds for across awaits (see
   * `compileForFrame`). `step` and `capturePixels` draw whether the loop is
   * paused or not, so a frame they drew meanwhile rendered into whatever the
   * compile was holding. Traced in the cinematic test: the tour reached the
   * island, an LOD compile took the reflection's target, and the next step's
   * output pass ran inside the reflection's render context, ending its pass
   * twice ("RenderPassEncoder was already ended").
   */
  private async compilesIdle(): Promise<void> {
    while (this.lodCompile !== null || this.qualityApply !== null) {
      await Promise.all([
        this.lodCompile?.catch(() => undefined),
        this.qualityApply?.catch(() => undefined),
      ]);
    }
  }

  private async stepDeterministic(dt: number, steps = 1): Promise<void> {
    this.deterministic = true;
    try {
      for (let i = 0; i < steps; i++) {
        // A step can start an LOD compile; the next one waits it out.
        await this.compilesIdle();
        await this.sampler.readNow();
        this.beginNodeFrame();
        await this.loop.step(dt, 1);
      }
    } finally {
      this.deterministic = false;
    }
  }

  /**
   * Starts a new node frame, as a browser tick does for a live frame.
   *
   * three advances its frame counter on its own rAF tick (and in
   * `compileAsync`), never on a render, and the nodes it gates on that counter
   * include the scene pass and the bloom. A harness step is a frame that no
   * tick has to have started: two steps issued inside one tick drew the scene
   * pass once, and the second step's TRAA resolved the first one's image at a
   * new jitter offset. How many steps shared a tick depended on how fast each
   * ran, so at Low, where steps are quick, two captures from the same reset
   * could differ across every edge in the frame. One counter step per harness
   * frame makes each draw its own scene, however the browser schedules them.
   */
  private beginNodeFrame(): void {
    (this.renderer as unknown as { _nodes: { nodeFrame: { update(): void } } })._nodes.nodeFrame.update();
  }

  /**
   * Renders the full post-processed frame into an offscreen target and reads the
   * pixels straight back.
   *
   * This is the capture path the visual harness uses, in preference to a
   * compositor screenshot. A screenshot goes through the browser's own
   * presentation path — it can arrive a frame late, it is subject to whatever
   * the compositor decides about colour management, and under automation it may
   * not arrive at all. Reading the render target is the same pixels the shader
   * wrote, on demand, with no frame-pacing dependency.
   *
   * The returned buffer is RGBA8, **top-down** — row 0 is the top of the image,
   * ready to hand to a PNG encoder without flipping. That is the WebGPU
   * backend's readback order, and it is worth stating because the opposite is
   * the reasonable guess: render-target space is conventionally bottom-up, and
   * assuming so here produces an ocean above a sky.
   */
  async capturePixels(): Promise<{ width: number; height: number; data: Uint8Array }> {
    await this.compilesIdle();
    const target = this.ensureCaptureTarget();
    const { width, height } = target;

    const previous = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    // Exactly one render, and that matters.
    //
    // This used to render twice and discard the first, to let the wave textures'
    // generated mip chains settle. That became actively wrong once the surface
    // started sampling the framebuffer for refraction: the second render's
    // backdrop can contain the first render's output, so the image feeds back
    // into itself and two consecutive captures of an unchanged world no longer
    // agree — which is precisely the property the whole visual harness rests on.
    //
    // Settling is the caller's job instead, and the harness already does it by
    // discarding warm-up captures before the one it keeps.
    //
    // Its own node frame, like a step: see `beginNodeFrame`. So a capture is a
    // whole frame, with its own TRAA jitter and history step, and two in a row
    // differ by the resolve. What is reproducible is a capture from a reset:
    // the same reset gives the same pixels, which is what every comparison in
    // the harness takes. Before this, whether the capture redrew the scene at
    // all depended on whether the browser had ticked since the last step.
    this.beginNodeFrame();
    await this.renderFrame();
    this.renderer.setRenderTarget(previous);

    const raw = (await this.renderer.readRenderTargetPixelsAsync(
      target,
      0,
      0,
      width,
      height,
    )) as ArrayBufferView;

    // Copied to an exactly-sized buffer, honouring the view's offset and length.
    //
    // Not `new Uint8Array(raw.buffer)`: the readback allocation is pooled and can
    // be larger than the image, so taking the whole backing store appends stale
    // bytes from a previous read. The pixels are identical either way, but the
    // trailing garbage is not, and it made two captures of an unchanged frame
    // compare unequal — a false regression in every byte-exact check.
    const view = new Uint8Array(raw.buffer, raw.byteOffset, width * height * 4);
    return { width, height, data: new Uint8Array(view) };
  }

  /** The RGBA8 target captures render into, sized to the drawing buffer. */
  private ensureCaptureTarget(): THREE.RenderTarget {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const width = Math.max(1, Math.floor(size.x));
    const height = Math.max(1, Math.floor(size.y));
    if (
      this.captureTarget === null ||
      this.captureTarget.width !== width ||
      this.captureTarget.height !== height
    ) {
      this.captureTarget?.dispose();
      this.captureTarget = new THREE.RenderTarget(width, height, {
        type: THREE.UnsignedByteType,
        format: THREE.RGBAFormat,
        colorSpace: THREE.SRGBColorSpace,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
      });
    }
    return this.captureTarget;
  }

  /**
   * Compiles every material in the scene against the current camera before the
   * loop starts.
   *
   * `compileAsync` walks the visible graph and builds the render pipelines
   * without drawing, so the cost lands behind the loading overlay where the user
   * is already waiting. It is best-effort: a backend that cannot honour it must
   * not stop the app from starting.
   */
  /**
   * Builds pipelines for LOD levels that have just become drawable.
   *
   * The dance is the one `loadSceneContent` and the tier change already use, and
   * it is forced from both ends: `compileAsync` walks `traverseVisible`, so a
   * hidden mesh compiles to nothing and the level has to be revealed before it
   * can be built — and a frame drawn between the reveal and the compile
   * resolving is exactly the inline compile being avoided. The only gap that is
   * safe between them is one no frame can be drawn in, so the loop is paused and
   * settled first.
   *
   * Rare by construction: the boot prewarm covers everything drawable from the
   * opening camera, so this runs only when the viewer crosses an LOD switch onto
   * a level that was skipped — in practice, flying in to the island. The pause
   * is a few frames rather than the ~11 s that compiling every level at boot
   * would have cost every session, including the sessions that never go there.
   */
  private async compileLodLevels(levels: readonly THREE.InstancedMesh[]): Promise<void> {
    const resume = this.loop.hold();
    try {
      await this.loop.settle();
      if (this.disposed) return;
      for (const mesh of levels) mesh.visible = true;
      try {
        await this.compileForFrame();
      } catch {
        // A compile can fail while the device is reconfiguring. Marking the
        // levels compiled anyway would let an unbuilt pipeline reach a draw, so
        // they stay unclaimed and the next deal offers them again.
        for (const mesh of levels) mesh.visible = false;
        return;
      }
      this.props?.markCompiled(levels);
    } finally {
      resume();
    }
  }

  /**
   * `compileAsync` for the pass the frame actually draws the scene through.
   *
   * Called bare, `compileAsync(scene, camera)` compiles for whatever the
   * renderer holds — outside a frame that is no target and no MRT, so it falls
   * back to the frame buffer target: one colour attachment, one output. The
   * frame draws the scene through `scenePass` instead, into its own target and,
   * with TRAA, through an `output` + `velocity` MRT, which changes every
   * material's fragment stage and, for velocity, its vertex stage too. None of
   * the prewarm's pipelines were the ones the frame then asked for, so every
   * one was built again synchronously on the first frame that drew it.
   * Measured on aaa-pass at High on the FXC path: a hundred pipelines in the
   * first frame after `isReady`, and the main thread frozen while the GPU
   * process compiled them.
   *
   * So the frame's state is reproduced for the compile, and held for all of
   * it, not just the call: `compileAsync` takes its render context up front but
   * builds each object's nodes later, one per task, and the node builder reads
   * the target and the MRT from the renderer when it does. The same goes for
   * the velocity node, which TRAA hands the unjittered projection for the
   * length of every frame (`TRAANode.setViewOffset`) and takes back after;
   * `VelocityNode.setup` emits different code with and without one.
   *
   * Holding renderer state across awaits is only safe because nothing renders
   * meanwhile: every caller compiles with the loop not yet started or paused.
   * A caller that has to give up waiting calls `releaseFrameState` before it
   * lets a frame run.
   */
  private compileForFrame(): Promise<void> {
    this.releaseFrameState();
    const scenePass = this.scenePass;
    // While it runs, the renders the compile itself sets off — the reflection
    // and the shadow map, from inside `updateBefore` — build asynchronously
    // too; see `installAsyncPipelines`.
    this.compilesInFlight++;
    const run = async (): Promise<void> => {
      // WebGL keeps the plain compile: it has neither the MRT nor the FXC
      // build cost, and compiling into the scene pass's multisampled target
      // there sets off an invalid depth-stencil blit.
      if (!scenePass || this.backend !== 'webgpu') {
        await this.renderer.compileAsync(this.scene, this.camera);
        return;
      }
      const generation = ++this.compileGeneration;
      await this.compileReflection();
      // Released while the reflection compiled: the caller gave up waiting
      // and frames are running, so the main pass cannot hold the renderer.
      if (generation !== this.compileGeneration) return;
      await this.compileWithState(
        scenePass.renderTarget,
        scenePass.getMRT(),
        this.camera,
        this.taaEnabled,
      );
      // The compile has only *started* the pipelines (see
      // `installAsyncPipelines`); this is where they land. A caller that
      // reveals what it compiled, or tears resources down, after this
      // resolves needs nothing still building. Not an error if they run long:
      // a live frame leaves out what is not ready rather than stopping for it.
      if (!(await this.pipelinesSettled(PIPELINE_SETTLE_MS))) {
        console.warn('[ocean] pipelines still building after the compile:', this.pendingPipelines);
      }
    };
    return run().finally(() => {
      this.compilesInFlight--;
    });
  }

  /**
   * Compiles the planar reflection's render ahead of the main pass.
   *
   * The reflection renders the whole scene again, from a mirrored camera into
   * a target of its own, and it does so from the ocean's `updateBefore` — so
   * the main compile met it when it reached the ocean, and built every render
   * object in it synchronously, inside a single task: 1.7 s with the main
   * thread blocked, measured at High. Compiled first, through `compileAsync`,
   * the same objects are built one task at a time, and the nested render finds
   * them done. Same target (pinned, see `installStableContexts`), same mirrored
   * camera, posed by the last frame; the ocean is hidden for the projection,
   * as the reflector hides it.
   */
  private async compileReflection(): Promise<void> {
    // Before the first frame the mirrored camera has never been posed, so a
    // compile culled against it would build an arbitrary set — and the boot
    // prewarm is on the critical path to that first frame. The first frames
    // build the reflection's few boot-time objects themselves, asynchronously.
    if (!this.reflections || this.loop.stats.frameMs === 0) return;
    const setup = this.reflections.renderSetup(this.camera);
    const ocean = this.water.material;
    const visible = ocean.visible;
    // Only for the projection, which `compileAsync` does before it first
    // yields; what it compiles after that is the list it made then.
    ocean.visible = false;
    let compile: Promise<void>;
    try {
      compile = this.compileWithState(setup.target, null, setup.camera, false);
    } finally {
      ocean.visible = visible;
    }
    await compile;
  }

  /**
   * One `compileAsync` with the renderer holding `target` and `mrtNode` (and
   * TRAA's velocity projection, if `velocityProjection`) until it settles or
   * `releaseFrameState` gives them back; see `compileForFrame`. The projection
   * is over by the time this returns, so state that only has to hold for it
   * can be restored as soon as the call does.
   */
  private compileWithState(
    target: THREE.RenderTarget,
    mrtNode: SceneMRT | null,
    camera: THREE.Camera,
    velocityProjection: boolean,
  ): Promise<void> {
    const renderer = this.renderer;
    const previousTarget = renderer.getRenderTarget();
    const previousMRT = renderer.getMRT();
    const traaVelocity = velocityProjection
      ? (velocity as unknown as { setProjectionMatrix(m: THREE.Matrix4 | null): void })
      : null;
    // The context `compileAsync` will build for: the same lookup it makes.
    const context = (
      renderer as unknown as { _renderContexts: { get(t: unknown, m: unknown): object } }
    )._renderContexts.get(target, mrtNode);
    const hold = (): (() => void) => {
      const heldTarget = renderer.getRenderTarget();
      const heldMRT = renderer.getMRT();
      renderer.setRenderTarget(target);
      renderer.setMRT(mrtNode as SceneMRT);
      traaVelocity?.setProjectionMatrix(this.camera.projectionMatrix);
      return () => {
        renderer.setRenderTarget(heldTarget);
        renderer.setMRT(heldMRT as SceneMRT);
        traaVelocity?.setProjectionMatrix(null);
      };
    };
    hold();
    let released = false;
    const release = (): void => {
      if (released) return;
      released = true;
      renderer.setRenderTarget(previousTarget);
      renderer.setMRT(previousMRT as SceneMRT);
      traaVelocity?.setProjectionMatrix(null);
      if (this.frameStateRelease === release) this.frameStateRelease = null;
      // Still building: what is left is built under this state, one object at
      // a time, by `installOrphanedCompileGuard`.
      this.orphanedCompiles.set(context, hold);
    };
    this.frameStateRelease = release;
    let compile: Promise<void>;
    try {
      compile = renderer.compileAsync(this.scene, camera);
    } catch (error) {
      release();
      this.orphanedCompiles.delete(context);
      throw error;
    }
    return compile.finally(() => {
      release();
      if (this.orphanedCompiles.get(context) === hold) this.orphanedCompiles.delete(context);
    });
  }

  /** Puts back what `compileForFrame` holds, if a compile still holds it. */
  private releaseFrameState(): void {
    this.compileGeneration++;
    this.frameStateRelease?.();
  }

  private async prewarm(timeoutMs = 30_000): Promise<void> {
    let compile: Promise<void> | null = null;
    const timeoutError = new Error(
      'pipeline prewarm timed out after ' + timeoutMs + ' ms',
    );
    let timer = 0;
    try {
      compile = this.compileForFrame();
      const timeout = new Promise<never>((_, reject) => {
        timer = window.setTimeout(() => reject(timeoutError), timeoutMs);
      });
      await Promise.race([compile, timeout]);
      this.shadersReady = true;
    } catch (error) {
      console.warn('[ocean] pipeline prewarm skipped', error);
      // The compile may still be running, and it holds the renderer on the
      // scene pass's target; the caller is about to let a frame draw.
      this.releaseFrameState();
      // `compileAsync` has no cancellation API, so a timed-out compile is still
      // running — and on a slow device, which is exactly where the timeout
      // fires, it can still be running when a tier change tears its resources
      // down. That is the async-pipeline-versus-destroyed-resource crash
      // `drainQualityRequests` documents at length. Publish it so the drain can
      // join it before it destroys anything.
      if (error === timeoutError && compile) {
        this.pendingCompile = compile.then(
          () => { this.shadersReady = true; },
          () => undefined,
        );
        void this.pendingCompile.finally(() => {
          this.pendingCompile = null;
        });
      }
    } finally {
      if (timer !== 0) window.clearTimeout(timer);
    }
  }

  /**
   * Starts the model requests — ship, dressing, reef fish — and returns them.
   * Idempotent: the first call starts them, every later one gets the same
   * promise.
   *
   * Split from `loadSceneContent` so the boot can start these *before* the
   * first prewarm rather than after it. That prewarm is the sea and the sky
   * compiling on the GPU, five-odd seconds in which the main thread is mostly
   * idle — and the requests are network, worker transcodes and `Props`'
   * placement, all of which can happen underneath it. Nothing here touches the
   * scene: the results wait, unadded, until `loadSceneContent` takes them, so
   * the order in which things become visible and get compiled is unchanged.
   */
  private requestSceneContent(): NonNullable<App['contentRequests']> {
    this.contentRequests ??= this.startContentRequests();
    return this.contentRequests;
  }

  private startContentRequests() {
    this.assets = new AssetLoader({
      // KTX2Loader has to ask the device which compressed formats it can
      // transcode to — ASTC, ETC2, BC — and it can only ask a renderer that
      // exists. Without this the loader falls back and every dressing texture
      // arrives untranscoded, which is the whole saving handed straight back.
      renderer: this.renderer,
      onProgress: ({ fraction, itemsLoaded, itemsTotal }) => {
        // The requests start under the first prewarm, while the boot bar is
        // still reporting it; they take the bar over once `start` hands it on.
        if (!this.contentOwnsBootBar) return;
        const progress = 0.86 + fraction * 0.1;
        boot.set(
          progress,
          'Bringing the ship and island in · ' + itemsLoaded + ' of ' + itemsTotal,
        );
      },
    });

    // Settled, not `Promise.all`. The ship and the scene dressing are separate
    // features and must fail separately: one unreachable prop asset previously
    // rejected the whole batch, taking the ship, buoyancy and the wake with it
    // and leaving an empty ocean with nothing but a console error to show for it.
    return Promise.allSettled([
      Ship.load(this.assets),
      Props.load(this.assets, {
        // The tier decides whether the island's vegetation is meshes or cards,
        // and on Low and Medium it decides that those GLBs are never requested
        // at all. Passing only `detailScale` leaves Props inferring it.
        qualityTier: this.state.quality,
        detailScale: QUALITY_TIERS[this.state.quality].propsDetail,
      }),
      // The reef species. Already settled-by-construction — `loadReefFish`
      // resolves null rather than rejecting — but it rides in the same batch so
      // its request overlaps the ship's rather than queueing behind it.
      loadReefFish(this.assets),
    ]);
  }

  private async loadSceneContent(): Promise<void> {
    /** Roots added hidden, revealed together once their pipelines exist. */
    const loaded: THREE.Object3D[] = [];

    this.contentOwnsBootBar = true;
    const [shipResult, propsResult, reefFishResult] = await this.requestSceneContent();
    if (this.disposed) return;

    // One frame first, before any of it joins the scene. The requests start
    // under the first prewarm and are usually done by the time the loop starts,
    // so without this the compile pause below landed before a single frame had
    // been drawn: the overlay lifted on an empty canvas, and the camera director
    // had never run, so the LODs were dealt for a camera still at its
    // constructor position and every level the real opening shot needs was
    // compiled again, mid-session, the moment it moved. Waiting *here* rather
    // than just before the pause matters too: once `this.props` is set, the
    // frame's own LOD pass claims the new levels and starts a second compile
    // racing this one.
    if (this.loop.isRunning && !this.loop.isPaused) await this.loop.nextFrame();
    if (this.disposed) return;

    if (reefFishResult.status === 'fulfilled' && reefFishResult.value) {
      this.fish.setSpecies('reef', reefFishResult.value);
      // The species arrives as a new mesh; it joins the rest of the life on the
      // underwater layer.
      this.fish.object.traverse((node) => node.layers.set(UNDERWATER_LAYER));
    }

    if (shipResult.status === 'fulfilled') {
      const ship = shipResult.value;
      this.ship = ship;
      // Gulls work the ship and sit on her yards; see `Birds.setShip`.
      this.birds.setShip(ship.object);
      // Hidden until compiled — see the reveal at the end of this function.
      ship.object.visible = false;
      loaded.push(ship.object);
      this.scene.add(ship.object);
      this.wetness.adopt(ship.object, 'ship');
      // After wetting, so the filter widens the wet roughness too.
      applySpecularAATo(ship.object);
      // Backlit canvas. The key uniforms follow the moon at night.
      applyThinTransmissionTo(ship.object, /sail/i, {
        keyDirection: this.uKeyDirection,
        keyColor: this.uKeyColor,
        occlusion: (worldPosition) => this.seafloor.cloudShadowNode(worldPosition),
      });
      this.lighting.local.setShip(ship.object);
      // The hull gets the same treatment the dressing does, and the part of it
      // that matters here is the caustics: `underwater.png` shows the submerged
      // hull as a flat black cutout, because the one thing lighting a shape down
      // there — the pattern the surface focuses onto everything below it — was
      // wired to the seafloor and to nothing else. `Caustics.intensityNode`'s own
      // header names "a hull material" as an intended consumer.
      applyGroundShadingTo(ship.object, {
        light: this.atmosphere.sunLight,
        keyShadow: (worldPosition) => this.seafloor.cloudShadowNode(worldPosition),
        caustics: (worldPosition) => this.caustics.intensityNode(worldPosition),
        groundHeight: (worldPosition) => this.seafloor.heightNode(worldPosition),
      });

      this.shipBody = new BuoyantBody({
        object: ship.object,
        probePoints: ship.probePoints,
        mass: 90_000,
        // A hull, not a barrel: it resists heave and is shaped not to resist
        // surge. Its longitudinal and lateral resistance comes from
        // `ShipController`, which knows which way the bow is pointing and can
        // make the two differ by the order of magnitude a keel actually does.
        // Nearly none. What remains is the genuine coupling to the water's own
        // orbital motion — a hull does get shoved about by a passing swell — but
        // the resistance to being *driven* belongs to the controller. Even a
        // tenth of the probe damping was around 83 kN at cruising speed, which is
        // most of the engine.
        horizontalDamping: 0.03,
        horizontalDrag: 0,
      });
      this.buoyancy.add(this.shipBody);

      // Where this hull meets the water hard enough to throw it: the stem, the
      // shoulders either side of it, the topsides amidships for a beam sea, and
      // the quarters. Weighted forward — a hull slams where it is finest and
      // moving fastest through the vertical, so the forefoot does most of it and
      // the quarters very little. The hull's forward direction is +X and its
      // beam runs along Z (see `forwardLocal` in `Ship`). Heights are above the
      // waterline: the side probes sit up the topsides so a crest has to climb
      // the hull to reach them, which is what a slap is.
      {
        const halfLength = ship.hullLength * 0.5;
        const halfBeam = ship.hullBeam * 0.5;
        this.spray.setHull({
          object: ship.object,
          velocity: this.shipBody.velocity,
          hullLength: ship.hullLength,
          hullBeam: ship.hullBeam,
          probes: [
            { offset: new THREE.Vector3(halfLength * 0.9, 0.35, 0), weight: 1, role: 'bow' },
            { offset: new THREE.Vector3(halfLength * 0.6, 0.45, halfBeam * 0.85), weight: 0.75, role: 'shoulder' },
            { offset: new THREE.Vector3(halfLength * 0.6, 0.45, -halfBeam * 0.85), weight: 0.75, role: 'shoulder' },
            { offset: new THREE.Vector3(0, 0.7, halfBeam * 0.98), weight: 0.6, role: 'side' },
            { offset: new THREE.Vector3(0, 0.7, -halfBeam * 0.98), weight: 0.6, role: 'side' },
            { offset: new THREE.Vector3(-halfLength * 0.7, 0.4, halfBeam * 0.8), weight: 0.3, role: 'quarter' },
            { offset: new THREE.Vector3(-halfLength * 0.7, 0.4, -halfBeam * 0.8), weight: 0.3, role: 'quarter' },
          ],
        });
      }

      // The controller exists from load but stays inert until Boat mode selects
      // it, so W/S and A/D cannot steer a ship the viewer is not driving.
      // The heightfield goes in with her: the hull is turned away from shoaling
      // water by the same sand the floor mesh is displaced by, which is what
      // stops a viewer sailing the ship onto the island. See `applyShoal`.
      this.shipControls = new ShipController(this.shipBody, (x, z) => seafloorHeight(x, z));
      // Both flags from the current mode, not just `enabled`. The mode change
      // that would have set them may already have happened — the models take
      // seconds to arrive, and a viewer who selects Cinematic during the
      // download changed mode while this object did not yet exist. It was
      // therefore born with the keyboard live, and holding S once the ship
      // appeared drove it straight through the tour.
      this.applyShipControlMode();

      ship.setDebugProbesVisible(this.state.buoyancyProbes);
      this.wake.setDebugVisible(this.state.wakeProbes);
      this.previousShipPosition.copy(ship.object.position);
    } else {
      console.error('[ocean] ship failed to load', shipResult.reason);
    }

    if (propsResult.status === 'fulfilled') {
      const props = propsResult.value;
      this.props = props;
      props.object.visible = false;
      loaded.push(props.object);
      this.scene.add(props.object);
      // Props share materials with the hull through the loader's cache;
      // `adopt` de-duplicates, so this is a no-op for anything already tracked.
      this.wetness.adopt(props.object, 'island');
      sanitizeDielectrics(props.object);
      applySpecularAATo(props.object);
      this.lighting.local.setIsland(props.lightAnchors(), props.object);
      // Everything placed on the ground learns about the ground: contact
      // darkening at its base, the caustic pattern if it is under water, and the
      // island's own shadow and the cloud deck's if it is not. See
      // `scene/groundShading`. One pass over the tree, one treatment per
      // distinct material.
      applyGroundShadingTo(props.object, {
        light: this.atmosphere.sunLight,
        // Cloud shade only, **not** the island's own heightfield march, and the
        // reason is compile time rather than frame time.
        //
        // The dressing bakes to roughly twenty distinct materials, and a
        // twenty-four-step raymarch compiled into every one of them took the
        // first frame from seconds to minutes on the FXC path the test harness
        // forces. It is also nearly redundant: the sun's shadow map is a
        // +/-260 m box that follows the viewer, so a prop close enough for its
        // own shadow to read is inside it, and past that range the canopy cards
        // — which do carry the march, in their vertex stage, at four samples a
        // card — are what the viewer is actually looking at.
        keyShadow: (worldPosition) => this.seafloor.cloudShadowNode(worldPosition),
        caustics: (worldPosition) => this.caustics.intensityNode(worldPosition),
        groundHeight: (worldPosition) => this.seafloor.heightNode(worldPosition),
        foliage: {
          wind: this.foliageWind,
          test: (name) => FOLIAGE_MESHES.test(name),
          // Uniform nodes rather than values: the sun moves every frame and
          // these are inside a compiled graph.
          sunDirection: this.uKeyDirection,
          sunColor: this.uKeyColor,
        },
      });

      for (const floater of props.floaters) {
        const body = new BuoyantBody({
          object: floater.object,
          probePoints: createRadialProbes(floater.radius),
          // Rough displacement for a hollow float of this size.
          mass: 40 * floater.radius * floater.radius * floater.radius,
        });
        this.buoyancy.add(body);
        // A buoy coming down off a swell smacks the water too. One probe, a
        // small crown, and only tested within range of the eye.
        this.spray.addFloat(floater.object, floater.radius, body.velocity);
      }
    } else {
      console.error('[ocean] scene props failed to load', propsResult.reason);
    }

    // --- gameplay (src/gameplay) --------------------------------------------
    // Helm-mode sails, anchor and cargo. Needs the hull; the floating barrel it
    // clones for cargo is optional. Added hidden with the rest of `loaded`, so
    // the reveal-and-compile below builds its pipelines before it is drawn.
    if (this.ship && this.shipBody && this.shipControls) {
      // The HUD is the sink. Speed, heading and wind are already on its dial via
      // `updateHudNav`, so the instruments add only what the dial cannot show —
      // the sail set and the anchor — as plaque lines, and leaving the helm
      // clears the plaque the gameplay filled.
      const ui: GameUiSink = {
        toast: (text, kind) => {
          this.hud.toast(text, kind);
          // The reward toast has its chime already; the others get a cue of their own.
          if (kind === 'warning') this.audio.uiTick('warn');
          else if (kind !== 'reward') this.audio.uiTick('info');
        },
        setObjective: (objective) => this.hud.setObjective(objective),
        setScore: (label, value) => this.hud.setScore(label, value),
        setShipTelemetry: (telemetry) => {
          if (telemetry === null) {
            this.hud.clearScore();
            return;
          }
          this.hud.setScore('Sails', SAIL_LABEL[telemetry.sail]);
          if (telemetry.anchored) this.hud.setScore('Anchor', 'Down');
          else this.hud.clearScore('Anchor');
        },
      };
      this.gameplay = new Gameplay({
        body: this.shipBody,
        controller: this.shipControls,
        hullLength: this.ship.hullLength,
        hullBeam: this.ship.hullBeam,
        mass: 90_000,
        sampler: this.sampler,
        floorHeight: (x, z) => seafloorHeight(x, z),
        wind: () => ({
          towardRad: getPreset(this.state.preset).sea.windDirection,
          speed: this.state.windSpeed,
        }),
        isHelm: () => this.state.cameraMode === 'boat',
        camera: this.camera,
        lootTemplate: this.props?.object.getObjectByName('barrel-0') ?? null,
        audio: this.audio,
        ui,
        shake: this.director,
      });
      this.gameplay.object.visible = false;
      loaded.push(this.gameplay.object);
      this.scene.add(this.gameplay.object);
      const hooks = (window as unknown as { __ocean?: Record<string, unknown> }).__ocean;
      if (hooks) hooks.gameplay = this.gameplay.testHandle;
    }
    // --- end gameplay ---------------------------------------------------------

    // Compile what just arrived, before it is first drawn.
    //
    // `prewarm` runs before this function is even called — it has to, because
    // the loop must start without waiting on the network — so every material the
    // ship and the dressing bring with them was compiling inline on the frame
    // that first rendered it. That is a shader compile during gameplay, which
    // this project's constraints rule out, and placing the scene dressing made
    // it considerably worse: twenty more models, each with its own materials.
    //
    // Awaiting `compileAsync` is not by itself enough, and the first version of
    // this was wrong about that: the objects were already in the scene, the loop
    // was already running, and a frame drawn between adding them and the compile
    // resolving hits exactly the inline compile this is meant to avoid.
    //
    // So they are added hidden, and the reveal happens with the loop stopped.
    // The ordering is forced from both ends: `compileAsync` walks the scene with
    // `traverseVisible`, so compiling before the reveal compiles nothing at all,
    // and revealing before the compile is the original bug. The only gap between
    // them that is safe is one no frame can be drawn in.
    //
    // **None of which is needed when the loop has not started yet.** On the boot
    // path this runs behind the overlay, before `loop.start()`, so there is no
    // frame that could be drawn between the reveal and the compile and no loop
    // to pause; the caller prewarms once, immediately after this resolves, and
    // that single pass covers the sea, the sky and everything loaded here. The
    // branch below is what the *other* caller would need — a tier change or any
    // future mid-session load — and it is kept working rather than deleted,
    // because the bug it closes is invisible until it is not.
    if (!this.loop.isRunning) {
      for (const object of loaded) object.visible = true;
      this.sceneContentLoaded = true;
      return;
    }

    const resume = this.loop.hold();
    try {
      await this.revealAndCompileContent(loaded);
    } finally {
      // A hold, not a flag: a tier change that starts during this load pauses
      // the loop on its own account, and must not find frames running because
      // this load ended first. Likewise a harness's `resetDeterministic`.
      resume();
    }
    this.sceneContentLoaded = true;
  }

  /** The part of `loadSceneContent` that runs with the loop held. */
  private async revealAndCompileContent(loaded: readonly THREE.Object3D[]): Promise<void> {
    await this.loop.settle();
    for (const object of loaded) object.visible = true;
    // Deal the LODs for where the camera is *before* compiling, so the compile
    // builds the levels the next frame will draw and nothing else.
    //
    // This is the boot path now — the loop starts before the content load, so
    // this branch, not the one above, is what every session takes — and it was
    // compiling every level of every kind: `seal` leaves LOD0 holding the whole
    // population and the lower levels visible, and `compileAsync` walks
    // `traverseVisible` without looking at `count`. `start` deals the LODs too,
    // but only after this pass had already built the lot. Measured at High on
    // the FXC path the harness forces: the content compile was ~30 s of a ~38 s
    // boot. `updateLod` hides every level the camera cannot see and hands back
    // the ones it can; levels skipped here are compiled when a camera first
    // needs them, by the same path a fly-in to the island always used.
    this.props?.updateLod(this.camera.position);
    const wanted = this.props?.takeUncompiledLevels() ?? [];
    for (const mesh of wanted) mesh.visible = true;
    await this.prewarm();
    this.props?.markCompiled(wanted);
  }

  /**
   * Compass, wind dial and log. Bearings go out as compass bearings and speeds
   * in knots; the HUD diffs and throttles its own writes, so this is called
   * every frame.
   */
  private updateHudNav(): void {
    this.camera.getWorldDirection(_hudView);
    // The spectrum's wind direction is the way the sea runs, so the wind
    // blows *from* the opposite bearing.
    const windAngle = getPreset(this.state.preset).sea.windDirection;
    const windTo = compassBearing(Math.cos(windAngle), Math.sin(windAngle));
    const atHelm = this.state.cameraMode === 'boat' && (this.shipControls?.isEnabled ?? false);
    const ship = atHelm ? this.shipControls?.getState(this.shipStateOut) ?? null : null;
    this.hud.setNav({
      headingDeg: compassBearing(_hudView.x, _hudView.z),
      windFromDeg: windTo + 180,
      windKnots: this.state.windSpeed * MS_TO_KNOTS,
      shipHeadingDeg: ship ? compassBearing(Math.cos(ship.heading), Math.sin(ship.heading)) : null,
      shipKnots: ship ? ship.speed * MS_TO_KNOTS : null,
    });
  }

  /**
   * The ship as the audio hears it: how far and which side, how she is
   * rolling and pitching, what the canvas is doing, and wave slaps from four
   * hull probes. Allocation-free.
   */
  private updateShipAudio(dt: number, audio: typeof this.audioParams): void {
    const ship = this.ship;
    if (ship === null) {
      audio.shipDistance = Number.POSITIVE_INFINITY;
      audio.sailFill = 0;
      audio.sailLuff = 0;
      return;
    }
    const hull = ship.object;
    const cam = this.camera.position;
    _audioRel.copy(hull.position).sub(cam);
    const distance = _audioRel.length();
    _audioRight.setFromMatrixColumn(this.camera.matrixWorld, 0);
    audio.shipDistance = distance;
    audio.shipPan = distance > 1e-3 ? _audioRel.dot(_audioRight) / distance : 0;

    // Roll: the beam axis (+Z, starboard) tilting out of the horizontal;
    // pitch: the bow axis (+X) rising. The same frame `Gameplay` uses.
    _audioAxis.set(0, 0, 1).applyQuaternion(hull.quaternion);
    audio.shipHeel = -Math.asin(THREE.MathUtils.clamp(_audioAxis.y, -1, 1));
    _audioAxis.set(1, 0, 0).applyQuaternion(hull.quaternion);
    audio.shipPitch = Math.asin(THREE.MathUtils.clamp(_audioAxis.y, -1, 1));

    const sails = this.gameplay?.audioState(this.sailAudio) ?? null;
    audio.sailFill = sails?.sailFill ?? 0;
    audio.sailLuff = sails?.sailLuff ?? 0;

    // Wave slaps: water climbing past a point on the planking, fast. Only
    // within earshot, since each probe is a sea-surface sample.
    if (!this.sampler.ready || distance > 120 || !(dt > 0)) return;
    hull.updateMatrixWorld();
    const halfLength = ship.hullLength * 0.5;
    const halfBeam = ship.hullBeam * 0.5;
    for (let i = 0; i < SLAP_PROBES.length; i++) {
      const [along, up, across] = SLAP_PROBES[i];
      _audioProbe.set(along * halfLength, up, across * halfBeam).applyMatrix4(hull.matrixWorld);
      const immersion = this.sampler.height(_audioProbe.x, _audioProbe.z) - _audioProbe.y;
      const strength = this.slaps.update(i, immersion, dt);
      if (strength > 0) {
        // Placed within the ship's own image by which side of the listener the probe is on.
        _audioRel.copy(_audioProbe).sub(hull.position);
        const pan = THREE.MathUtils.clamp(_audioRel.dot(_audioRight) / Math.max(1, halfBeam), -1, 1);
        this.audio.waveSlap(strength, pan);
      }
    }
  }

  /**
   * Which arrangement the score should play. Priority: under water, then
   * weather, then the tour, then sailing (at the helm and moving or with
   * canvas set), else calm. The audio system holds a state for a couple of
   * seconds before following it, so calling this every frame is the intent.
   */
  private chooseMusicState(raining: number, submersion: number): MusicState {
    if (submersion > 0.5) return 'underwater';
    const coverage = THREE.MathUtils.clamp((this.state.cloudCoverage - 0.7) / 0.25, 0, 1);
    const gale = THREE.MathUtils.clamp((this.state.windSpeed - 12) / 8, 0, 1);
    const weather = Math.max(raining, coverage * gale, this.state.preset === 'storm' ? 1 : 0);
    if (weather > 0.5) this.musicStormy = true;
    else if (weather < 0.3) this.musicStormy = false;
    if (this.musicStormy) return 'storm';
    if (this.state.cameraMode === 'cinematic') return 'tour';
    if (this.state.cameraMode === 'boat' && this.sailAudio.active) {
      const speed = this.audioParams.hullSpeed;
      if (speed > 1 || this.sailAudio.sailFill > 0.2) return 'sailing';
    }
    return 'calm';
  }

  /** 0..1 for the score: ship speed against a brisk 7.5 m/s (~15 knots). */
  private musicEnergy(): number {
    return THREE.MathUtils.clamp(this.audioParams.hullSpeed / 7.5, 0, 1);
  }

  /** `setGullLocator`'s callback: one drawn bird near enough to hear. */
  private locateGull(out: GullFix): boolean {
    const count = this.birds.getCount();
    if (count === 0) return false;
    const cam = this.camera.position;
    _audioRight.setFromMatrixColumn(this.camera.matrixWorld, 0);
    for (let attempt = 0; attempt < 4; attempt++) {
      // Deterministic per draw is not needed here: which bird calls is not
      // part of any captured frame, and the audio itself is muted in captures.
      const index = Math.floor(Math.random() * count);
      if (!this.birds.positionOf(index, _audioBird)) continue;
      _audioRel.copy(_audioBird).sub(cam);
      const distance = _audioRel.length();
      if (distance > 240) continue;
      out.distance = distance;
      out.pan = distance > 1e-3 ? _audioRel.dot(_audioRight) / distance : 0;
      return true;
    }
    return false;
  }

  private buildUi(): void {
    const callbacks = {
      onChange: <K extends keyof UiState>(key: K, value: UiState[K]) => {
        this.state[key] = value;
        this.onStateChange(key);
      },
    };
    this.panel = new Panel(this.uiRoot, this.state, callbacks);
    this.hud = new Hud(this.uiRoot, this.state.cameraMode, callbacks);
    // Clicks, ticks and rustles for the ledger and the HUD. The UI reports
    // what happened as a cue name and never sees the audio system.
    this.detachUiSounds = attachUiSounds(this.uiRoot, (cue, value) => this.audio.uiTick(cue, value));
    this.stormQuote = new StormQuote(this.uiRoot);
    this.hud.setBackend(this.backend);

    // Built only where it will be used. A mouse user has better controls and a
    // stick over the frame would only be in the way; constructing it anyway and
    // hiding it would leave a pointer target on the canvas for every desktop
    // viewer to discover by accident.
    if (TouchControls.isTouchDevice()) {
      this.touchControls = new TouchControls(this.uiRoot, {
        onInput: (throttle, rudder) => this.shipControls?.setInput(throttle, rudder),
        onSail: (direction) => this.gameplay?.stepSail(direction),
        onAnchor: () => this.gameplay?.toggleAnchor(),
      });
      this.touchControls.setVisible(this.state.cameraMode === 'boat');
    }
  }

  private onStateChange(key: keyof UiState): Promise<void> | void {
    switch (key) {
      case 'quality':
        return this.requestQuality(this.state.quality);
      case 'preset': {
        // A preset brings its own sky. Coverage lives in UI state because the
        // slider owns it afterwards, and nothing ever copied the preset's value
        // into it — so choosing Storm drew the storm's deck at whatever the
        // slider last said, 0.36 by default: blue sky between fair-weather
        // puffs under a sky lit for a 0.88 overcast. The slider still wins
        // after this, which is what a slider is for.
        const coverage = getPreset(this.state.preset).clouds.coverage;
        if (this.state.cloudCoverage !== coverage) {
          this.state.cloudCoverage = coverage;
          this.panel?.setState({ cloudCoverage: coverage });
        }
        this.applyPreset();
        break;
      }
      case 'windSpeed':
      case 'peakWavelength':
        // Through `applyPreset`, not straight to the spectrum.
        //
        // Wind is not only a wave parameter. It stretches the surface's glitter
        // along the crests, leans the rain, scuds the clouds, and — via Monahan —
        // sets the rate at which whitecaps are deposited. All of those read
        // `state.windSpeed` from inside `applyPreset`, and this case did not call
        // it, so every one of them kept the value from whenever the *preset* last
        // changed. Dragging the wind slider from a calm sea to a gale raised the
        // waves and left the foam, the glitter and the clouds behaving as though
        // it were still calm; the whitecap rate in particular was only ever
        // correct for the wind a preset happened to load with.
        //
        // Without the environment capture: this does not move the sun.
        this.applyPreset(false);
        break;
      case 'cloudCoverage':
        this.clouds.setParams({ coverage: this.state.cloudCoverage });
        break;
      case 'timeOfDay':
        // Through `applyPreset`, not straight to the atmosphere: moving the sun
        // changes the environment capture, the ambient fill and the water's sky
        // colours, and routing it through one place is what keeps those in step.
        this.applyPreset();
        break;
      case 'pixelRatio':
        this.renderer.setPixelRatio(
          clampPixelRatio(window.devicePixelRatio * this.state.pixelRatio),
        );
        this.onResize();
        break;
      case 'cameraMode': {
        // Read before `setMode`, which is what changes it. The environment-cube
        // restore below has to know whether the tour is being *left*, and by the
        // time the director has been told the new mode that is unanswerable.
        const leftTheTour = this.director.currentMode === 'cinematic';
        this.director.setMode(this.state.cameraMode);
        this.hud.setCameraMode(this.state.cameraMode);
        // Selecting Boat selects the *ship*, not just a camera. Cinematic drives
        // it too — the flight steers the hull rather than teleporting it, so the
        // wake, the buoyancy and the spray are the real ones. Orbit and Fly
        // release it, so they keep their own keys and a hull cannot be left under
        // power while the viewer is somewhere else.
        //
        this.applyShipControlMode();
        // The on-screen throttle stays with the mode that can actually use it.
        this.touchControls?.setVisible(this.state.cameraMode === 'boat');
        // Leaving the tour hands the sun back. The flight drives the sun's
        // position directly every frame (see `update`), so without this the hour
        // it happened to be at when the viewer pressed a key would stick until
        // something else re-applied the preset.
        // `true` on the way *out of* the tour, and only there.
        //
        // The flight now moves the environment cube as well as the sun, so
        // leaving it at night with the capture suppressed would restore the
        // preset's daylight sky to the dome and leave the night IBL bound to
        // every PBR surface in the scene, with nothing that would ever clear it.
        // Entering the tour, and every other mode change, still skips the
        // capture — it is a whole extra scene render and the sky has not moved.
        if (this.state.cameraMode !== 'cinematic') {
          this.applyPreset(leftTheTour);
          if (leftTheTour) this.lastEnvElevation = Number.NaN;
        }
        break;
      }
      case 'volume':
        this.audio.setVolume(this.state.volume);
        break;
      case 'musicVolume':
        this.audio.setMusicVolume(this.state.musicVolume);
        break;
      case 'buoyancyProbes':
        this.ship?.setDebugProbesVisible(this.state.buoyancyProbes);
        break;
      case 'wakeProbes':
        this.wake?.setDebugVisible(this.state.wakeProbes);
        break;
      case 'forceWebGL':
        // Switching backend means tearing down every GPU resource and rebuilding
        // against a new device; a reload is both simpler and more reliable than
        // trying to migrate a live scene graph between backends.
        this.restartWithBackend(this.state.forceWebGL);
        break;
      default:
        break;
    }
  }

  /**
   * The one place the water surface is constructed.
   *
   * Both the initial build and every tier rebuild go through here, so a new
   * input can never be wired into one path and forgotten in the other.
   */
  private buildWaterMaterial(): OceanMaterial {
    return new OceanMaterial({
      ...this.simulation.fields,
      floorDepthNode: (worldPosition) => this.seafloor.depthNode(worldPosition),
      foam: {
        texture: this.wake.texture,
        extent: this.wake.extent,
        resolution: this.wake.resolution,
      },
      reflectionNode: this.reflections?.node ?? null,
      // Same field the cloud march samples, so the shade lands under the cloud
      // that casts it rather than merely correlating with it.
      //
      // Two samples and three octaves, which is the cheapest setting that still
      // fixes the low-sun case. The sea is most of the frame at every tier and
      // *all* of it on WebGL2 Low, where the cloud layer is not drawn at all —
      // so this one call site is worth more than the rest put together. Measured:
      // the full four-octave three-sample form costs WebGL2 Low about four
      // milliseconds, on a shadow cast by clouds that tier never renders.
      cloudShadowNode: this.clouds.shadowNode({ samples: 2, octaves: 3 }),
      ssrNode: this.ssr
        ? (worldPosition, worldNormal, fallback) =>
            this.ssr!.reflectionNode(worldPosition, worldNormal, fallback)
        : null,
    });
  }

  private rebuildQualityResources(tier: QualityTier): void {
    const quality = QUALITY_TIERS[tier];
    this.simulation.resize(quality.fftSize, quality.cascades);
    this.sampler.rebuild();
    this.water.setCascades(this.simulation.fields);
    this.wake.setCascades(this.simulation.derivativeTextures, this.simulation.tileSizes);
    this.waveHeightNode = this.simulation.heightNode();

    this.scene.remove(this.oceanMesh.mesh);
    this.oceanMesh.dispose();
    this.oceanMesh = new OceanMesh(this.water.material, {
      radialSegments: quality.meshRings,
      angularSegments: quality.meshSegments,
    });
    this.water.setVertexSpacing(this.oceanMesh.spacingPerMetre);
    this.scene.add(this.oceanMesh.mesh);
    this.applyPreset(false);
  }

  private applyQualitySettings(tier: QualityTier): void {
    const quality = QUALITY_TIERS[tier];

    // The wave textures are recreated by `resize`, so the material's bindings are
    // stale — re-point them. The reflection target is *not* recreated: its
    // resolution is written once at startup, see `Reflections.setQuality`.
    //
    // Rebuilding the material instead, as this used to, was wrong in three ways.
    // Every input had to be re-supplied and one silently was not, which is how
    // `floorDepthNode` went missing after any tier change. The node graph
    // recompiled mid-session, which is exactly the in-gameplay shader compile the
    // performance work is meant to avoid. And once the surface began sampling the
    // framebuffer for refraction, each rebuild leaked the backdrop texture that
    // came with it — measured at forty textures over eight tier changes, which
    // the leak test caught.
    this.clouds.setParams({ steps: quality.cloudSteps });

    // Shadows are a *light* setting here, never a renderer one.
    //
    // `renderer.shadowMap.enabled` looks like the natural tier switch and is a
    // trap. Three builds a light's shadow node lazily inside `setup()`, and that
    // setup returns early while the flag is false — so switching it off and back
    // on leaves the node cached with a null `shadowMap`, and the next thing to
    // render the scene reads `depthTexture` off it. The planar reflector renders
    // the scene from its own `updateBefore`, so it got there first and the crash
    // landed on a Low-to-Medium transition.
    //
    // The flag is therefore set true once at startup and never touched. The tier
    // expresses "no shadows" as `castShadow = false`, which three handles on the
    // light rather than on the renderer.
    this.atmosphere.setShadowMapSize(quality.shadowMapSize);
    // Cascade map size (read live by the filter) and reach; local lights' gate.
    this.lighting.applyQuality(quality);

    // The tier-sized targets, resized rather than replaced.
    //
    // Without these a runtime downgrade keeps every allocation the tier it came
    // from made, which defeats the entire point of `AdaptiveQuality`: the
    // machine that drops High to Low under load is the machine that most needs
    // Low's memory back, and it would have got Low's step counts while still
    // holding High's targets.
    //
    // All three resize in place and keep their existing texture object, so no
    // node graph loses its binding — the constraint the shadow comment above
    // explains at length.
    this.wake?.setResolution(quality.wakeResolution);
    this.caustics?.setResolution(quality.causticsResolution);
    this.atmosphere.setEnvironmentResolution(quality.environmentResolution);
    // A uniform write, not a rebuild — the march reads its own bound.
    this.seafloor.setShadowSteps(quality.terrainShadowSteps);

    // Particle budget is a live setting, not a construction-time one; `setCount`
    // rebuilds the instanced geometry against the new tier.
    this.particles.setCount(quality.underwaterParticles);
    // Moves `instanceCount` only; the flock is not rebuilt and each gull keeps
    // its own circuit across the change.
    this.birds.setCount(quality.birds);
    // A uniform write, not a rebuild: the geometry and the shader are the same
    // at every tier and only the master scale moves, so changing tier cannot
    // compile anything.
    this.spray.setStrength(quality.spray);
    this.fish.setCount(quality.fish);
    this.kelp.setCount(quality.kelp);
    this.reefLife.setDetail(quality.reefLife);
    this.creatures.setDetail(quality.seaCreatures);
    this.meadow.setCount(quality.meadow);
    this.canopy.setCount(quality.canopy);
    this.audio.setQuality(tier);

    // WebGL2 gets the analytic path regardless of tier. The backdrop and depth
    // reads are the least portable part of the surface, and a fallback that
    // renders a coherent simpler image beats one that renders a broken richer
    // one — which is the whole point of having a declared fallback policy.
    this.water.setRefraction(this.backend === 'webgl' ? 0 : quality.refraction);
    this.water.setDepthRange(this.camera.far - this.camera.near);
    this.water.setReflection(quality.reflection);
    this.reflections?.setQuality(quality.reflectionScale);
    this.ssr?.setQuality(DEFAULT_SSR_STEPS[tier]);
    this.ssr?.setStrength(quality.reflection);
    this.fog.setSteps(quality.fogSteps);
    // WebGL2 gets the two-lattice floor whatever the tier, matching the rest of
    // the fallback policy.
    this.lensRain.setQuality(this.backend === 'webgl' ? 1 : quality.lensRainQuality);
    // Bloom keeps its tier on both backends, and that is a decision rather than
    // an omission. The pyramid is ordinary texture sampling into ordinary render
    // targets — there is no depth read, no backdrop read and no compute in it —
    // so it is one of the few effects here with nothing backend-specific to go
    // wrong. The stages that *do* read depth are gated below.
    this.bloom.setEnabled(quality.bloom === 1);
    // WebGL2 gets no depth of field, and this is the fallback policy rather than
    // an oversight. The gather takes a depth read per tap — up to 32 of them —
    // and the depth-texture path is the least portable part of this renderer;
    // the same reasoning already keeps refraction off that backend. A coherent
    // sharp image beats a richer one with the silhouettes wrong.
    this.dof.setSamples(this.backend === 'webgl' ? 0 : quality.dofSamples);
    // Kept on WebGL2: the occlusion disc is eight depth taps, which is the same
    // read the volumetric fog already performs there, so nothing new is being
    // asked of the backend.
    this.lensFlare.setEnabled(quality.lensFlare === 1);
    this.postChain.setTier(tier);
    this.water.setWakeDisplacement(quality.wakeDisplacement);
    // Instance counts only, and safe on a live scene: no geometry is rebuilt and
    // nothing allocates, so this is a tier knob rather than a reload. Null until
    // the models arrive — `Props.load` is passed the same number, so a tier
    // chosen before loading finishes is still the one that applies.
    this.props?.setDetailScale(quality.propsDetail);

    // Without the environment capture: the tier does not move the sun, and the
    // capture is a whole extra scene render into a cube target. See `applyPreset`.
  }

  /**
   * Sun elevation, azimuth and night blend for a clock time.
   *
   * A simple diurnal arc rather than a real ephemeris: no latitude, no season,
   * no equation of time. Those would change where the sun is by degrees, and
   * what this control exists for is to let someone drag from dawn to dusk and
   * watch the water follow — a model that answers that convincingly is worth
   * more here than one that is astronomically correct and looks the same.
   *
   * Elevation peaks at noon and goes negative at night; azimuth sweeps a full
   * turn so the light comes from the east in the morning and the west in the
   * evening. `nightIntensity` ramps once the sun is below the horizon, which is
   * what brings up the stars and the moon.
   */
  private sunFromClock(hours: number): {
    sunElevation: number;
    sunAzimuth: number;
    nightIntensity: number;
    moonElevation: number;
    moonAzimuth: number;
  } {
    // Noon at its highest, midnight at its lowest.
    const dayPhase = ((hours - 6) / 24) * Math.PI * 2;
    const elevation = Math.sin(dayPhase) * 1.32;
    // Civil twilight is roughly the first six degrees below the horizon; the
    // night blend follows it rather than snapping at zero, so dusk is a gradual
    // handover to the moon rather than a light switch.
    const night = THREE.MathUtils.clamp(-elevation / 0.22 + 0.15, 0, 1);
    const azimuth = ((hours - 6) / 24) * Math.PI * 2 + Math.PI;

    // The moon rides opposite the sun, so dragging into night finds it already
    // up. Without this the clock inherits whatever moon the preset declared —
    // and most declare none, so night was simply black, which is a poor answer
    // for a control whose whole purpose is to be dragged into it.
    return {
      sunElevation: elevation,
      sunAzimuth: azimuth,
      nightIntensity: night,
      moonElevation: -elevation * 0.82,
      moonAzimuth: azimuth + Math.PI,
    };
  }

  /**
   * @param captureEnvironment Re-render the environment cube. Skipped by a tier
   *   change, which does not alter the sky: that capture is a *whole extra scene
   *   render*, and issuing one in the middle of a teardown is what produced
   *   "Destroyed texture [ShadowDepthTexture] used in a submit" — the cube render
   *   referenced the shadow map in a command buffer that outlived it.
   */
  /**
   * Points the ship controller at the current camera mode.
   *
   * One place, because there are two callers and they must not drift: the mode
   * change, and the controller's own construction — which happens when the
   * models finish downloading and so can land either side of a mode change.
   */
  private applyShipControlMode(): void {
    const controls = this.shipControls;
    if (controls === null) return;
    const mode = this.state.cameraMode;
    // `resetInput`, not `setInput(0, 0)`. Throttle and rudder are spooled, and
    // `setEnabled` only clears the spool on a *transition* — Cinematic and Boat
    // both keep the controller enabled, so switching between them would
    // otherwise hand the viewer a ship still carrying the tour's full ahead.
    controls.resetInput();
    // Cinematic selects the *ship* as much as Boat does: the flight steers the
    // hull rather than teleporting it, so the wake, the buoyancy and the spray
    // are the real ones.
    controls.setEnabled(mode === 'boat' || mode === 'cinematic');
    // But not the viewer's keys. They deliberately outrank `setInput` so someone
    // at the helm beats the on-screen throttle; during a cinematic that rule
    // would let a held S command full astern against a full-ahead beat.
    controls.setKeyboardEnabled(mode !== 'cinematic');
  }

  /**
   * Re-captures the environment cube while the tour moves the sun.
   *
   * The tour drives `Atmosphere.setParams` directly, sixty times a second, and
   * never touched the cube. That was invisible while the flight only swept
   * 08:18 to 16:42 — a frozen daylight IBL is close enough to a moving daylight
   * one that nothing shows. It is badly wrong now that the flight reaches night:
   * every PBR surface in the scene, hull and rigging and cannon and wet rock,
   * would go on being lit by a noon sky while the sky being drawn behind them is
   * black.
   *
   * Not every frame. `updateEnvironment` renders six cube faces and then rebuilds
   * the PMREM chain, which is milliseconds. Gated on the sun having actually
   * moved, so the cost is paid a handful of times a lap instead of sixty times a
   * second — and the gate is on *elevation* rather than on a timer, so the fast
   * sweep through dawn gets the updates and the slow dwell at noon does not need
   * them.
   */
  private refreshTourEnvironment(force = false): void {
    const elevation = this.atmosphere.sunDirection.y;
    if (
      !force &&
      Number.isFinite(this.lastEnvElevation) &&
      Math.abs(elevation - this.lastEnvElevation) < TOUR_ENV_ELEVATION_STEP
    ) {
      return;
    }
    this.lastEnvElevation = elevation;
    // `force` has to reach *both* throttles or it is not a force. This one is on
    // sun elevation and lives here; `Atmosphere` has a second on how far the key
    // direction has moved since it last captured, and a reset that happened to
    // land inside it would otherwise silently keep the previous shot's
    // irradiance.
    if (force) this.atmosphere.invalidateEnvironment();
    this.atmosphere.updateEnvironment(this.renderer, this.scene);
  }

  private applyPreset(captureEnvironment = true): void {
    const preset = getPreset(this.state.preset);

    // The preset owns the sun until the viewer takes it, and then the clock
    // does. Spread over the preset rather than replacing it, so a preset's
    // turbidity, Mie and overcast — the things that make it *that place* —
    // survive being re-timed.
    const clock = this.state.timeOfDay;
    this.atmosphere.setParams(
      clock === null ? preset.atmosphere : { ...preset.atmosphere, ...this.sunFromClock(clock) },
    );
    this.clouds.setParams({
      ...preset.clouds,
      coverage: this.state.cloudCoverage,
      steps: QUALITY_TIERS[this.state.quality].cloudSteps,
      // Driven by the same wind that raises the sea, so a storm's clouds scud
      // and its swell runs the same way. Evolution scales with it too: a squall
      // boils, a calm day drifts.
      windDirection: preset.sea.windDirection,
      windSpeed: 6 + this.state.windSpeed * 1.6,
      evolutionRate: 0.006 + this.state.windSpeed * 0.0022,
    });
    this.weather.setKind(preset.weather.kind);
    // The override wins where it is set. The per-frame path and
    // `resetDeterministic` both already defer to it, and this did not — so a
    // caller that forced a rain rate and then touched anything routed through
    // here got the preset's rain back in the particles while the *surface* kept
    // showing the override. Widened by wind now coming through this function,
    // and wrong before that too.
    this.weather.setIntensity(this.rainOverride ?? preset.weather.intensity);

    // Rain leans with the wind that is driving the sea. Reading the shear off
    // the same wind speed and direction the spectrum uses is what stops a storm
    // from having waves running one way and rain slanting another — the two
    // being visibly independent is a strong tell that the weather is a costume.
    const windAngle = preset.sea.windDirection;
    const lean = Math.min(0.55, this.state.windSpeed * 0.022);
    this.weather.setShear(Math.cos(windAngle) * lean, Math.sin(windAngle) * lean);

    // `windwardFoam` rides in `sea` because it is a property of the sea state,
    // but it is not a spectrum parameter — it drives the foam buffer's deposit.
    // Split off explicitly rather than relying on the spread to be ignored.
    const { windwardFoam, ...seaSpectrum } = preset.sea;
    this.simulation.updateSpectrum({
      ...seaSpectrum,
      windSpeed: this.state.windSpeed,
      peakWavelength: this.state.peakWavelength,
    });

    // `setAppearance` only writes what it is given, and only some presets name
    // `crestGlow` — so it is reset here or Tropical's value leaks into every
    // preset visited after it.
    this.water.setAppearance({
      crestGlow: DEFAULT_APPEARANCE.crestGlow,
      ...preset.water,
    });
    // The surface needs the wind to stretch its glitter along the crests.
    this.water.setWind(preset.sea.windDirection, this.state.windSpeed);

    // --- whitecaps ----------------------------------------------------------
    //
    // Neither of these was called at all, which meant the accumulation buffer
    // ran at one fixed threshold and one fixed rate for every preset and every
    // wind speed: a glassy dusk foamed exactly as hard as a 21 m/s storm. The
    // preset's `foamThreshold` reached only the surface's instantaneous mask,
    // which since the near field went over to the buffer is the one place it no
    // longer decides anything.
    //
    // Coverage against wind follows Monahan & O'Muircheartaigh's fit to ship
    // observations, W = 3.84e-6 * U^3.41 — a very steep law, and the reason a
    // linear wind response never looks right. It gives 3.9% at 15 m/s, which is
    // what the sea state assertions measure, 0.2% at 6 m/s and 12% at 21 m/s.
    // Normalising by the 15 m/s value turns it into a multiplier on how strongly
    // the deposit reads, with the clamp keeping a calm sea faintly streaked
    // rather than surgically clean.
    const whitecapAt = (u: number) => 3.84e-6 * Math.pow(Math.max(0, u), 3.41);
    const foamThreshold = preset.water.foamThreshold ?? DEFAULT_APPEARANCE.foamThreshold;
    // The buffer's threshold is far tighter than the surface mask's. It has to
    // be: the mask asks "is this water folding *now*", which it answers for
    // every frame the crest is overhead, while the buffer asks "did it break
    // here", and then keeps the answer for a time constant afterwards. Measured
    // on cascade 0 at 15 m/s, fold < 0.14 covers 5.5% of the surface and fold < 0
    // covers 3.8%, so the buffer sees only genuinely folded water and the trail
    // it leaves supplies the rest of the coverage.
    // Monahan drives the *deposit rate*, not just the surface's read strength.
    //
    // It previously scaled only `setFoamStrength`, which changes how strongly an
    // existing deposit reads — so the empirical coverage law was decorating the
    // opacity of a foam field whose generation was a fixed constant for every
    // wind speed. The rate is what the law is about, so that is what it now
    // moves; the strength keeps a gentler share of it, because at a given
    // coverage heavier seas also entrain more air per breaking event.
    const whitecapRatio = whitecapAt(this.state.windSpeed) / whitecapAt(15);
    // 0.32, down from 0.55, and this is the first version of the number that was
    // measured rather than judged. `tests/foam.spec.ts` reads the accumulation
    // buffer back and compares the standing coverage against the law that is
    // supposed to be producing it: at 0.55 a 15 m/s sea stood at 7.96% against
    // Monahan's 3.93%, which is what made the reference image a sheet of white
    // rather than a sea with whitecaps on it.
    //
    // The limits are soft — bounded, but never flat. A hard
    // `max(0.45, min(2.2, ratio))` binds below about 11.9 m/s and above about
    // 18.9, and inside those regions the deposit rate stopped responding to the
    // wind at all: measured coverage was *identical* at 6 and 9 m/s, and
    // identical again at 21 and 24. That is a dead zone at each end of the
    // slider, where dragging the wind moved the waves and not the foam. Adding a
    // small share of the unclamped ratio keeps the bound while leaving the
    // response monotonic everywhere.
    //
    // The rate deliberately does not carry the whole of U^3.41, and the
    // measurements are what say so: the *area* that folds rises with wind too,
    // so a rate following the full law on top of it double-counts. At 21 m/s the
    // clamped rate is a fifth of what the law alone would ask for, and the
    // rendered coverage lands within 0.02 points of it.
    // 0.50, up from 0.32, and the increase is a consequence of the spectrum
    // rather than a change of taste.
    //
    // 0.32 was measured against a directional lobe with a fixed `cos^8`
    // exponent, which made every cascade run downwind together. Replacing it
    // with the Hasselmann frequency-dependent spread left the *energy*
    // untouched — the lobe is renormalised, see `Spectrum` — but spread the
    // short waves across every heading, and waves that do not agree on a
    // direction do not pile into steep crests. The fold threshold is unchanged
    // and the sea state is unchanged; what fell is the fraction of the surface
    // steep enough to trip it, measured at 0.60 to 0.67 of the old value at
    // every wind speed from 6 to 24 m/s.
    //
    // `tests/foam.spec.ts` is what caught it and what set this number: it reads
    // the accumulation buffer back and compares the standing coverage against
    // Monahan & O'Muircheartaigh, so the rate is a calibration against a
    // measurement rather than a free parameter. A more realistic spectrum
    // needing a higher deposit rate to reach the same observed coverage is not a
    // contradiction — the law predicts coverage, and it is coverage that is held
    // fixed.
    this.wake.setBreaking(
      foamThreshold * 0.34,
      // Recalibrated 2026-11-14. The surface work since the last calibration
      // left fewer texels steep enough to fold, and the old rate (0.5 x
      // (clamp(0.34, 2.2) + 0.08 r)) measured 0.38x, 0.14x, 0.38x, 0.37x and
      // 0.24x Monahan at 6, 9, 15, 21 and 24 m/s — two outside the band.
      // Doubling it fixed the middle and not the ends: 6 m/s rose to 2.96x,
      // because 6 and 9 m/s shared the floor and moved together, while 21 and
      // 24 m/s barely moved at all. Above ~7% the buffer saturates, and a
      // higher rate whitens folds that are already white rather than adding
      // new ones.
      //
      // So the floor drops below 9 m/s's own ratio (0.175) and the linear share
      // rises, which separates the calm pair: 9 m/s now gets more than 6. The
      // cap stays because past it the rate buys nothing. Measured, against
      // Monahan and in the test's own units:
      //
      //   6 m/s  0.15%  0.86x     15 m/s  4.79%  1.22x     24 m/s  7.30%  0.37x
      //   9 m/s  0.32%  0.47x     21 m/s  6.96%  0.56x
      //
      // Monotonic, and the 6-to-24 range is 49x against the 38x the law's
      // shape demands. 24 m/s is the tight one: it is held by saturation, not
      // by this rate.
      Math.max(0.2, Math.min(4, whitecapRatio)) + 0.46 * whitecapRatio,
    );

    // The windward-face term rides the same law, and has to: it is the same air
    // being entrained by the same wind, just before the fold rather than at it.
    //
    // **It does overlap the breaking term, and the rate is set knowing that.** A
    // wind-facing crest that is also folding satisfies both tests and receives
    // both deposits — they are not disjoint and cannot be, because a crest
    // steepens on its windward face before it breaks there. What keeps that from
    // being a bug is the size of the two: the windward rate is an order of
    // magnitude below the breaking rate, so on the small fraction of the surface
    // that is doing both, the windward share is a rounding error on top of a
    // deposit that was already going to saturate. What it buys is the *other*
    // 99% — the wind-facing water that never breaks at all, which the fold term
    // by construction cannot see.
    //
    // `tests/foam.spec.ts` measures the total, so the calibration above is
    // against both terms together rather than against the fold alone.
    this.wake.setWindward(windwardFoam * Math.max(0.15, Math.min(3, whitecapRatio)));

    // The surface's own mask is re-scaled from the same number.
    //
    // The preset values (0.42 to 0.55) were authored when that mask painted the
    // whole ocean, and against the measured fold distribution they select about
    // 18% of the surface — which is what a 15 m/s clear day was rendering, and it
    // reads as a gale. Now that the near field belongs to the buffer, the mask's
    // only job is the far field beyond the buffer's 420 m square, so it is scaled
    // to select roughly what the buffer does and the two meet without a seam.
    //
    // Derived here rather than by editing the nine presets so the two thresholds
    // cannot drift apart: they describe one physical property of one sea.
    this.water.setAppearance({
      foamThreshold: foamThreshold * 0.62,
      foamSoftness: 0.42,
    });
    this.weatherFoamStrength = Math.max(0.35, Math.min(1.2, Math.sqrt(whitecapRatio)));
    this.water.setFoamStrength(this.foamOverride ?? this.weatherFoamStrength);
    // Sky and horizon come from the atmosphere, not from preset constants.
    //
    // The first argument used to be the *sun* colour, which is not the sky by
    // any reading, and the other two were a single authored fog colour. So the
    // water's reflection and its aerial perspective were describing a different
    // sky from the one being drawn behind it, and where they disagreed the
    // horizon showed a hard step. Both now derive from the state that lights the
    // scene, so a preset cannot pull them apart. `fog.color` survives as the
    // aerial-perspective tint, which is genuinely an authored choice.
    this.water.setSky(
      this.atmosphere.zenithColor,
      this.atmosphere.horizonColor,
      preset.fog.color,
      preset.fog.density,
    );

    this.renderer.toneMappingExposure = preset.toneMappingExposure;
    // The bloom needs to know where white is, and only the exposure knows.
    //
    // Tone mapping happens after this whole post chain, so "1.0" in here is not
    // the sensor limit — `1 / exposure` is. The presets span 0.42 to 1.35, so a
    // fixed threshold would mean something different in each of the nine.
    this.bloom.setExposure(preset.toneMappingExposure);
    this.postChain.setPreset(this.state.preset);
    // With the exposure, not instead of it, and the pair is not redundant:
    // exposure decides where the scene sits on the tone curve, the grade decides
    // what colour it is once it gets there. Collapsing them into one number is
    // how a preset ends up choosing between being the right brightness and being
    // the right colour.
    this.colorGrade.setParams(preset.grade);
    if (captureEnvironment) this.atmosphere.updateEnvironment(this.renderer, this.scene);
  }

  /**
   * Coalesces tier changes and applies them with nothing in flight.
   *
   * A tier change disposes and recreates the FFT targets, the ocean geometry and
   * the particle field. Destroying any of those while a submitted command buffer
   * still references them is a use-after-free, which WebGPU states plainly:
   * "Destroyed texture used in a submit". Three earlier attempts each removed one
   * *trigger* — the renderer's shadow flag, `shadow.dispose()`, `castShadow` —
   * and the next appeared, because none of them addressed the lifetime.
   *
   * Requests coalesce into one drain. Note that the drain applies *every* pending
   * request in turn rather than one per frame: dragging a quality selector across
   * five tiers with the loop paused would otherwise take five frames that are not
   * being rendered anyway.
   */
  private requestQuality(tier: QualityTier): Promise<void> {
    this.pendingQuality = tier;
    if (this.qualityApply === null) this.qualityApply = this.drainQualityRequests();
    return this.qualityApply;
  }

  /**
   * Applies pending tier changes with no frame in flight.
   *
   * A tier change destroys GPU resources — the FFT targets, the ocean geometry,
   * the particle field — and destroying anything three has already referenced in
   * a submitted command buffer is a use-after-free. WebGPU says so out loud:
   * *"Destroyed texture [ShadowDepthTexture] used in a submit"*. It surfaced on
   * the shadow map because the teardown cascades into three rebuilding the
   * light's node graph, but the shadow map is the symptom, not the cause — which
   * is why three earlier fixes aimed at shadow state each removed one trigger and
   * the next appeared.
   *
   * So: stop the loop, wait for the queue to finish everything already submitted,
   * apply, restart. `onSubmittedWorkDone` is the fence that makes it safe, and
   * pausing is what stops a rAF callback from submitting a new frame in the gap.
   *
   * Requests coalesce: only the most recent tier is applied, which is also what a
   * viewer dragging a quality selector across five tiers wants.
   */
  private async drainQualityRequests(): Promise<void> {
    // An LOD compile already running owns the pause and gives it back when it
    // ends; taking the loop before that would read its pause as ours and let
    // its `finally` restart frames in the middle of the teardown.
    await this.lodCompile?.catch(() => undefined);
    const resume = this.loop.hold();
    // Pause *then* settle, in that order. Pausing stops new frames; settling
    // joins the one already inside `renderAsync`. Requesting the queue fence
    // without settling first registers it before that frame has even submitted,
    // so the fence resolves against an empty queue and the teardown lands on a
    // command buffer that is still being built.
    await this.loop.settle();
    // And the scene load, if one is still running. See `contentReady`.
    await this.contentReady?.catch(() => undefined);
    // And a prewarm that outlived its own timeout. `compileAsync` cannot be
    // cancelled, so the only safe thing to do with one still in flight is wait
    // for it — destroying the resources it is compiling against is precisely
    // the crash the comment below describes.
    //
    // WebGPU only. The crash is an asynchronous pipeline build meeting a texture
    // destroyed under it, which WebGL's synchronous programs cannot do; and
    // there a timed-out boot compile can run for a minute and more (measured
    // ~80 s), so waiting for it left a tier change hanging that long.
    if (this.backend === 'webgpu') await this.pendingCompile?.catch(() => undefined);
    try {
      while (this.pendingQuality !== null && !this.disposed) {
        const tier = this.pendingQuality;
        this.pendingQuality = null;
        const queue = (
          this.renderer as unknown as {
            backend?: { device?: { queue?: { onSubmittedWorkDone?: () => Promise<void> } } };
          }
        ).backend?.device?.queue;
        await queue?.onSubmittedWorkDone?.();
        if (this.disposed) return;
        this.rebuildQualityResources(tier);
        this.applyQualitySettings(tier);

        // Build the new tier's pipelines *while the loop is still paused*.
        //
        // A tier change replaces the ocean geometry, so three compiles its
        // pipeline asynchronously — and an async pipeline creation that is still
        // in flight when the next frame's resources turn over is the last form
        // this bug took: "Async render pipeline creation failed ... Destroyed
        // texture [ShadowDepthTexture] used in a submit". Compiling here means
        // there is nothing outstanding when the loop restarts, and it removes the
        // hitch a first-frame compile would otherwise cause anyway.
        try {
          await this.compileForFrame();
        } catch {
          // A compile can fail while the device is reconfiguring. The next frame
          // rebuilds what it needs; dropping this is better than aborting the
          // tier change half-applied.
        }

        // And drain again: the change itself submits work — the FFT rebuild and
        // the compile — and resuming the loop on top of that puts a new frame in
        // the queue beside resources that are still being replaced.
        await queue?.onSubmittedWorkDone?.();
      }
    } finally {
      this.qualityApply = null;
      // A hold, not a flag, so neither a capture that took the clock meanwhile
      // nor a scene load still running finds frames resumed under it.
      resume();
    }
  }

  private update = (dt: number, elapsed: number): void => {
    this.director.update(dt);
    this.postChain.update(dt);

    /**
     * The tour owns the sun while it is running.
     *
     * Driven straight into `Atmosphere.setParams` rather than through
     * `applyPreset`, and the distinction matters twice. `applyPreset` also
     * rebuilds the clouds, the weather and the water appearance and optionally
     * re-captures the environment cube, none of which the hour of the day needs
     * and all of which would be paid for sixty times a second. And `setParams`
     * merges — its own header says so — so the preset's turbidity, Mie and
     * overcast survive being re-timed, which is what keeps the tour recognisably
     * *this* place at every hour it passes through.
     *
     * `cinematicTimeOfDay` is a pure function of the loop clock, so this stays
     * inside the determinism guarantee the capture harness depends on.
     */
    if (this.state.cameraMode === 'cinematic') {
      const env = this.director.cinematicEnvironment();
      this.atmosphere.setParams(this.sunFromClock(env.hours));
      this.clouds.setParams({ coverage: env.cloudCoverage });
      // Kind *and* intensity. Setting the intensity alone produces no weather
      // at all under a clear preset: `Weather` will not draw while its kind is
      // 'clear', and the `raining` scalar a dozen lines below — which is what
      // reaches the lens beads, the surface stipple, the foam agitation and the
      // hull wetting — is gated on the same test.
      this.weather.setKind(env.weatherKind);
      this.weather.setIntensity(env.rain);
      this.cinematicFog = env.fogDensity;
      // Before the capture, not after it. The environment cube is graded by how
      // cloudy the sky is, and this used to be copied across at the *end* of the
      // update — so a capture triggered by the sun moving carried the new sun
      // and the previous frame's coverage. On a shot boundary that is the
      // previous *shot's* coverage, which is precisely the ordering dependence
      // the deterministic harness exists to rule out.
      this.atmosphere.setCloudCoverage(this.clouds.getParams().coverage);
      this.refreshTourEnvironment();
    } else {
      this.cinematicFog = 1;
    }

    // The lens focuses on whatever this mode's shot is about. Closed-form in
    // every mode — see `CameraDirector.focusDistance` — so a capture cannot
    // depend on how many frames of focus-pulling preceded it.
    this.dof.setFocusDistance(this.director.focusDistance());
    this.dof.update();

    // Before `update`, which is what reads it to place the light and its target.
    // The shadowed region follows the viewer rather than sitting on the origin,
    // so the island 1.4 km out gets the same shadows the ship does.
    this.atmosphere.setShadowFocus(this.camera.position.x, this.camera.position.z);
    this.atmosphere.update(dt);
    // The deck is lit in the sun's own colour, and by the moon once the sun has
    // gone — see `Clouds.setLighting`.
    this.clouds.setLighting(
      this.atmosphere.sunDirection,
      this.atmosphere.sunColor,
      this.atmosphere.moonDirection,
      this.atmosphere.getParams().nightIntensity,
    );
    // The island shadows itself from whatever is currently the key light, which
    // after sunset is the moon — `Atmosphere` retargets the same light.
    this.seafloor.setKeyDirection(this.atmosphere.keyDirection);

    this.birds.setSunDirection(this.atmosphere.sunDirection);
    // The second colour is the light an *unlit* bird receives, not the sky's own
    // radiance: handing over `zenithColor` raw puts a white belly at the sky's
    // luminance and the whole flock vanishes into it.
    this.birds.setLightColors(this.atmosphere.sunColor, _birdShade.copy(this.atmosphere.zenithColor).multiplyScalar(0.4));
    this.birds.update(dt, this.camera.position);

    // Cheap by construction: it early-outs unless the camera has moved far
    // enough for a level boundary to have plausibly been crossed. See
    // `Props.updateLod`.
    this.props?.updateLod(this.camera.position);
    // A level the boot prewarm skipped has just become drawable — fly in to the
    // island and the near LODs are wanted for the first time. `updateLod` leaves
    // them hidden and hands them over here; drawing one before its pipeline
    // exists is the inline gameplay compile this project rules out.
    // Never beside another compile: a tier change destroys what an LOD compile
    // is building against, and either one releasing the renderer's frame state
    // would leave the other building with none. The levels stay hidden and are
    // offered again once nothing else is compiling.
    if (
      this.props !== null &&
      this.lodCompile === null &&
      this.qualityApply === null &&
      this.pendingCompile === null
    ) {
      const pending = this.props.takeUncompiledLevels();
      if (pending.length > 0) {
        // Cleared by its own promise only: the harness calls `compileLodLevels`
        // directly, and that call must not mark this one finished.
        const compile = this.compileLodLevels(pending).finally(() => {
          if (this.lodCompile === compile) this.lodCompile = null;
        });
        this.lodCompile = compile;
      }
    }

    this.fish.setSunDirection(this.atmosphere.sunDirection);
    this.fish.update(dt);

    // The bed follows the same swell the surface does, so the water and the
    // weed under it agree about which way the sea is running.
    this.kelp.setSunDirection(this.atmosphere.sunDirection);
    this.kelp.setSwell(
      getPreset(this.state.preset).sea.windDirection,
      this.state.windSpeed,
      this.state.peakWavelength,
    );
    this.kelp.update(dt);

    // The reef sways with the same swell; it is hidden outright when the eye is
    // high enough above the sea that none of it could resolve.
    this.reefLife.setSwell(
      getPreset(this.state.preset).sea.windDirection,
      this.state.windSpeed,
      this.state.peakWavelength,
    );
    this.reefLife.setCameraPosition(this.camera.position);
    this.reefLife.update(dt);

    // The animals take the same sun the fish do — its colour, scaled as `Fish`
    // scales its own, and gone when the sun is.
    {
      const sunUp = THREE.MathUtils.smoothstep(this.atmosphere.sunDirection.y, -0.06, 0.1);
      this.creatures.setSun(
        this.atmosphere.sunDirection,
        _creatureSun.copy(this.atmosphere.sunColor).multiplyScalar(0.7 * sunUp),
      );
      this.creatures.setCameraHeight(this.camera.position.y);
      this.creatures.update(dt);
    }

    // The grass field follows the camera, so unlike everything else on the
    // island it has to be told where that is every frame. Two uniform writes.
    // The *key* light, at its strength — not the sun's direction and its
    // normalised colour. Every lit material follows the key, which is the moon
    // after dark and is dimmed by an overcast; the custom-lit grass and canopy
    // were the two things in the scene still lit by a full-strength sun under a
    // storm deck and by a sun below the horizon at night.
    const keyLight = this.atmosphere.sunLight;
    _keyLight.copy(keyLight.color).multiplyScalar(keyLight.intensity / 3.4);
    this.meadow.setSunDirection(this.atmosphere.keyDirection);
    this.meadow.setSunColor(_keyLight);
    this.meadow.setAmbientColor(_meadowAmbient.copy(this.atmosphere.zenithColor).multiplyScalar(0.5));
    {
      const bearing = getPreset(this.state.preset).sea.windDirection;
      _meadowWind.set(Math.cos(bearing), Math.sin(bearing));
      // Blade travel saturates well below a gale: past about 12 m/s grass is
      // already lying flat and more wind moves the *sea*, not the sward.
      this.meadow.setWind(_meadowWind, Math.min(1, this.state.windSpeed / 12));
    }
    this.meadow.update(dt, this.camera.position);

    // The canopy shares the meadow's light and wind — they are the same
    // vegetation under the same sky, and two sets of uniforms drifting apart is
    // how a sward ends up lit at a different hour from the trees above it.
    this.canopy.setSunDirection(this.atmosphere.keyDirection);
    this.canopy.setSunColor(_keyLight);
    this.canopy.setAmbientColor(_meadowAmbient);
    this.canopy.setWind(_meadowWind, Math.min(1, this.state.windSpeed / 14));
    this.canopy.update(dt);
    // The planted meshes sway on the same gust the cards do — same wavelength,
    // same speed, same amplitude. Not the same phase: `groundShading` explains
    // why a mesh has no per-instance constant to offset by, so a plant crossing
    // its handover keeps the character of its motion and may jump within the
    // cycle. That is what the handover distances are chosen around.
    this.foliageWind.setWind(_meadowWind, Math.min(1, this.state.windSpeed / 14));
    this.foliageWind.update(dt);
    // The island's animated set dressing reads the same wind.
    this.props?.setWind(_meadowWind, Math.min(1, this.state.windSpeed / 14));
    this.props?.update(dt);

    this.remains.setSun(this.atmosphere.sunDirection, this.atmosphere.sunColor);


    this.clouds.update(dt);
    this.weather.update(dt, this.camera.position);

    // Lightning, in heavy weather only, and never while a capture owns the
    // clock — a baseline must not depend on whether a strike happened to land.
    {
      const coverage = THREE.MathUtils.clamp((this.state.cloudCoverage - 0.7) / 0.25, 0, 1);
      const wind = THREE.MathUtils.clamp((this.state.windSpeed - 12) / 8, 0, 1);
      this.lightning.setCloudBase(getPreset(this.state.preset).clouds.altitude ?? 900);
      const strike = this.lightning.update(
        dt,
        this.camera,
        coverage * wind,
        !this.captureOwnsClock && this.director.submersion() < 0.5,
      );
      if (strike) {
        this.audio.playThunder(strike.distance / 1000, strike.distance / 343); // speed of sound, m/s
      }
      // Less of the flash as a grade lift now the strike lights the scene for
      // real (`lighting/LightningLight`); what is left is the eye's response to
      // a bright sky, which a frame-wide lift is the right model of.
      this.colorGrade.setFlash(this.lightning.flashIntensity * FLASH_GRADE_SHARE);
    }

    // The lights `lighting/` owns: cascade refit, fire and lanterns after dark,
    // and the strike's own light on the scene.
    this.lighting.update(
      elapsed,
      this.camera.position,
      this.atmosphere.sunDirection.y,
      this.atmosphere.overcast,
      this.lightning.flashIntensity,
      this.lightning.lastStrike,
      this.lightning.cloudBaseHeight,
    );
    // The camp smoke takes the key and the sky like a grey card, and the fire
    // from underneath once it is dark.
    _smokeLight
      .copy(this.atmosphere.sunLight.color)
      .multiplyScalar((this.atmosphere.sunLight.intensity / 3.4) * 0.75)
      .add(_smokeSky.copy(this.atmosphere.zenithColor).multiplyScalar(this.atmosphere.ambientLight.intensity * 0.6));
    this.props?.setDetailLighting(_smokeLight, this.lighting.local.fireLevel);

    // The key light, whichever body is providing it, at its actual strength.
    //
    // This used to be the sun direction, the sun colour and a hardcoded
    // intensity of 6 — so the water was lit as though by a noon sun at every
    // hour, and its subsurface scattering glowed green under the hull at half
    // past nine at night. It also went on taking its direction from a sun that
    // was below the horizon while the moon was the only thing actually casting.
    const key = this.atmosphere.sunLight;
    _keyDirection.copy(key.position).normalize();
    // 3.4 is the sun's full-daylight intensity in `Atmosphere`, so this is
    // "how close to full daylight is it" rather than an arbitrary scale.
    // The *direct* glint fades with the overcast on the same curve the sky uses
    // to hide the disc (`Atmosphere`: gone by three-quarters cover). The key
    // light keeps a quarter of its strength under a full deck as diffuse
    // daylight, but a mirror-sharp sun path on the sea under a sky with no sun
    // in it is the storm's most obvious lie.
    const clearSky = Math.max(0, 1 - THREE.MathUtils.clamp(this.atmosphere.overcast, 0, 1) / 0.75);
    // At night the moon's path has to be the brightest thing on the sea, but the
    // moon key (0.6) is a smaller fraction of the sun than the night sky fill is
    // of daylight, so at a straight 1.8 its glitter sank into the moonlit swell.
    const moonPath = 1 + (MOON_GLINT_BOOST - 1) * LightingRig.darkness(this.atmosphere.sunDirection.y, 0);
    const glint = (0.06 + 0.94 * clearSky * clearSky) * moonPath;
    this.water.setSun(_keyDirection, key.color, key.intensity * 1.8 * glint, key.intensity / 3.4);
    (this.uKeyDirection.value as THREE.Vector3).copy(_keyDirection);
    // Scaled by the light's own intensity so the term follows the sun down: at
    // dusk a backlit frond should stop glowing along with everything else.
    (this.uKeyColor.value as THREE.Color).copy(key.color).multiplyScalar(key.intensity / 3.4);
    // The sun moves and the sky follows it, so these have to be refreshed every
    // frame rather than only when a preset is applied.
    //
    // The water's own aerial perspective is wound back as the volumetric fog
    // comes up. Both describe the same air between the viewer and the horizon,
    // and running them at full strength together fogs it twice — the surface
    // term is a cheap analytic far-field haze and the volumetric pass is the
    // better answer wherever it is active, so the analytic one yields to it
    // rather than the two being summed.
    const fogPreset = getPreset(this.state.preset).fog;
    const hazeDensity = fogPreset.density * (1 - 0.75 * this.state.fogDensity);
    // Measured first: the water and the haze below are both given the dome's
    // own brightness after sunset, and the clouds further down read it too.
    this.measureSkyRadiance(_skyRadiance, _skyHorizon, _skyTypical, _skyZenith);
    const sunDown = 1 - THREE.MathUtils.smoothstep(this.atmosphere.sunDirection.y, -0.1, 0.02);
    seenSky(this.atmosphere.horizonColor, _skyTypical, sunDown, _seenHorizon);
    seenSky(this.atmosphere.zenithColor, _skyZenith, sunDown, _seenZenith);
    // A strike lights the cloud deck from inside, and the sea mirrors the deck:
    // the analytic sky the water reflects at grazing angles brightens with it.
    // Without this the flash lit the hull and left the sea dark under it.
    const deckFlash = this.lighting.flash.deck.intensity * 1.5;
    this.water.setSky(
      _flashZenith.copy(_seenZenith).multiplyScalar(1 + deckFlash),
      _flashHorizon.copy(_seenHorizon).multiplyScalar(1 + deckFlash),
      fogPreset.color,
      // Zero: the shared aerial perspective owns this now, for the sea as well
      // as for the land. Two independent haze terms meeting at the shoreline is
      // precisely the horizon step the gap analysis describes, and the water's
      // was the one that could not know what colour the sky was in a given
      // direction. The uniform stays wired so a future caller can put a
      // surface-local term back without re-plumbing the material.
      0,
    );
    // Everything the haze has to agree with, from the same atmosphere the dome
    // and the water read. Uniform writes; no rebuild.
    //
    // Under an overcast the light the haze scatters toward the eye is the
    // deck's grey, not the clear-sky blue the CPU-side horizon and zenith still
    // describe. Fed raw, a kilometre of storm haze repainted the dark cloud base
    // in clear-sky blue and the storm read as a fine evening.
    //
    // The preset's fog colour is authored for *its* hour, and the tour and the
    // time slider take it to others. At night or deep dusk it is far brighter
    // than any sky there is, and under a squall the pull above leaned the whole
    // haze toward it: the far cloud deck lit pale while the near clouds, lit by
    // the real dark sky, stood out as black blobs, and the sea mirrored it pink.
    // So the fog is dimmed to the sky's own brightness whenever the sky is the
    // darker of the two — never brightened, so a preset's daytime look is kept.
    const skyLuma = luma(_seenHorizon);
    const fogLuma = luma(fogPreset.color);
    _litFog.copy(fogPreset.color).multiplyScalar(fogLuma > 1e-5 ? Math.min(1, skyLuma / fogLuma) : 1);
    const overcastPull = THREE.MathUtils.clamp(this.atmosphere.overcast, 0, 1) * 0.85;
    _hazeHorizon.copy(_seenHorizon).lerp(_litFog, overcastPull);
    _hazeZenith.copy(_seenZenith).lerp(_litFog, overcastPull);
    // The deck takes the *tint* and scales it by the measured sky itself
    // (`Clouds.setSkyColors`), so it is handed the undimmed one.
    const tintLuma = luma(this.atmosphere.horizonColor);
    _cloudFog.copy(fogPreset.color).multiplyScalar(fogLuma > 1e-5 ? Math.min(1, tintLuma / fogLuma) : 1);
    _cloudZenith.copy(this.atmosphere.zenithColor).lerp(_cloudFog, overcastPull);
    this.aerial.setSky(
      this.atmosphere.betaRayleigh,
      this.atmosphere.betaMie,
      _litFog,
      hazeDensity,
      _hazeHorizon,
      _hazeZenith,
      this.atmosphere.sunDirection,
      this.atmosphere.sunColor,
    );
    // Cloud tops are lit by the sky over them, and the environment capture has
    // to know how much of that sky is cloud.
    //
    // And by how bright that sky actually *is*: the dome's own radiance at the
    // zenith and around the horizon, so the deck can never be lit brighter
    // than the sky it sits in by more than the sun or moon on it justifies.
    this.clouds.setSkyColors(_cloudZenith, _skyRadiance, _skyHorizon);
    // Idempotent, and the tour has already done it before its own capture — this
    // covers the free-camera path, where the coverage comes from the UI.
    this.atmosphere.setCloudCoverage(this.clouds.getParams().coverage);
    // From the *scene* camera, so the planar reflector's mirrored camera cannot
    // take the haze off the reflected island.
    this.aerial.setEyeHeight(this.camera.position.y);

    // Rain reaches the water. Only rain does — snow settles far too slowly to
    // punch a ring into a surface, and driving this from `weather.intensity`
    // alone would have the Arctic preset stippling the sea with snowflakes.
    const raining =
      this.rainOverride ??
      (this.weather.getKind() === 'rain' ? this.weather.getIntensity() : 0);
    this.water.setRain(raining, elapsed);
    // The shore break runs off the same clock the rest of the sea does, so a
    // deterministic rewind puts the surf sets back where they were.
    this.water.setSurf(
      elapsed,
      this.foamOverride ?? 1,
      this.state.peakWavelength,
      this.wake.shoreNear,
    );
    // Agitation: heavy rain whitens a sea surface on its own, independently of
    // whether the waves are steep enough to break.
    this.wake.setRainAgitation(raining);
    // Foam floats, and floating things move. Wind drift plus Stokes drift carries
    // surface material downwind at roughly 3% of the wind speed.
    const driftBearing = getPreset(this.state.preset).sea.windDirection;
    const driftSpeed = this.state.windSpeed * 0.03;
    // Where the shore breaks depends on how big the waves arriving at it are:
    // a bigger swell trips in deeper water, so the surf line walks seaward with
    // the sea state instead of sitting on a fixed contour.
    this.wake.setSwell(significantWaveHeight(this.simulation.spectrumParams), driftBearing);
    this.wake.setDrift(
      Math.cos(driftBearing) * driftSpeed,
      Math.sin(driftBearing) * driftSpeed,
    );
    // The axis the windward-face term biases foam along. Same bearing, but it
    // has to be set separately because the drift above goes to zero in a calm
    // and an axis cannot.
    this.wake.setWindAxis(driftBearing);

    // Spray reads the hull's world matrix, so it has to run after whatever moved
    // it this frame — the buoyancy solve above — and it samples the same wave
    // field the buoyancy did, so the surface it tests against is the one the
    // hull is actually floating on rather than a second opinion about it.
    // Spray drifts with the near-surface wind and is lit by the same key light
    // and sky as the foam it comes off.
    this.spray.setWind(
      Math.cos(driftBearing) * this.state.windSpeed * 0.6,
      Math.sin(driftBearing) * this.state.windSpeed * 0.6,
    );
    this.spray.setLighting(
      _keyDirection,
      this.atmosphere.sunLight.color,
      this.atmosphere.sunLight.intensity / 3.4,
      this.atmosphere.zenithColor,
    );
    this.spray.update(dt, elapsed, (x, z) => this.sampler.height(x, z), this.camera.position);
    // Wood and canvas darken and gloss in a squall, and stay damp well after it
    // passes — see `SurfaceWetness` for the asymmetric time constants.
    this.wetness.update(dt, raining);
    // The island's ground follows the island channel: rain, never spray.
    this.seafloor.setRainWetness(this.wetness.level('island'));

    this.lensRain.setIntensity(raining);

    this.simulation.update(elapsed);
    // Deterministic stepping owns the readback and awaits it; kicking off a
    // second, unawaited one here would put the race straight back.
    if (!this.deterministic) this.sampler.update();

    this.oceanMesh.recenter(this.camera.position);
    this.water.setWorldOffset(this.oceanMesh.mesh.position.x, this.oceanMesh.mesh.position.z);
    // Depth-buffer distances are measured along this axis; the surface needs it
    // to convert them into distance along each pixel's own ray.
    this.water.setCameraForward(this.camera.getWorldDirection(_keyDirection2));

    // --- underwater state -----------------------------------------------------
    // `submersion` is a soft band around the surface rather than a boolean, so
    // crossing the waterline cross-fades instead of popping.
    const submersion = this.director.submersion();
    const surface = this.sampler.height(this.camera.position.x, this.camera.position.z);
    const preset = getPreset(this.state.preset);
    this.water.setEyeSubmerged(this.camera.position.y < surface);

    this.underwater.setParams({
      submersion,
      // Signed, and the sign is the point: it tells the pass which side of the
      // surface each ray starts on, which is what a per-pixel waterline needs.
      eyeHeight: this.camera.position.y - surface,
      cameraDepth: Math.max(0, surface - this.camera.position.y),
      sunDirection: this.atmosphere.sunDirection,
      sunColor: this.atmosphere.sunColor,
      waterColor: preset.underwater.color,
      extinction: preset.underwater.extinction,
      visibility: preset.underwater.visibility,
      godRayStrength: preset.underwater.godRayStrength,
      godRaySteps: QUALITY_TIERS[this.state.quality].godRaySteps,
      // The lens the submerged view is looking through. Per preset, because a
      // storm's surface is a disturbed one and a glassy sunset's is very nearly
      // flat — see `Preset.underwater`.
      distortion: preset.underwater.distortion,
    });
    // The hull, as the one occluder the shafts need. A diver under a ship
    // should be in its shadow; the caustics field cannot know that, because it
    // describes the surface and not what floats on it.
    if (this.ship) {
      const hull = this.ship.object;
      this.underwater.setHullOccluder(
        _hullCenter.copy(hull.position).setY(hull.position.y + 1),
        // Beam, draught, length — in the hull's own frame, so the long axis
        // follows the bow rather than pointing at world Z whatever the heading.
        _hullRadius.set(this.ship.hullBeam * 1.6, 4, this.ship.hullBeam * 3.4),
        this.ship.heading,
        true,
      );
    } else {
      this.underwater.setHullOccluder(_hullCenter, _hullRadius, 0, false);
    }
    this.underwater.update(dt);

    // Audio reads the same scene state the renderer does, so the bed tracks the
    // sea rather than approximating it: wind drives the surf band, the wake's
    // own rain rate drives the rain layer, and `submersion` cross-fades the
    // whole graph under water on the same continuous value the visuals use.
    const audio = this.audioParams;
    audio.windSpeed = this.state.windSpeed;
    audio.waveHeight = this.state.peakWavelength * 0.045;
    audio.peakWavelength = this.state.peakWavelength;
    audio.rain = raining;
    audio.submersion = submersion;
    audio.hullSpeed = this.shipControls?.getState(this.shipStateOut).speed ?? 0;
    // The gull scheduler takes the flock's actual size rather than the tier's,
    // so a sky with no birds in it stays silent instead of the audio inventing
    // some. Low draws none, and hears none.
    audio.birdCount = this.birds.getCount();
    this.updateShipAudio(dt, audio);
    this.audio.setMusicState(this.chooseMusicState(raining, submersion), this.musicEnergy());
    this.audio.update(dt, audio);

    // Rain on the lens. Placed here rather than with the other rain wiring
    // because it needs `submersion`, which is only known once the camera has
    // been resolved against the surface — the effect suppresses itself as the
    // viewer goes under, since there is no lens above water to bead on.
    // The flare's source is the *live key light*, and that distinction is the
    // whole of it.
    //
    // `Atmosphere` owns one directional light and retargets it to the moon once
    // the sun is down, overwriting its colour with `MOON_LIGHT_COLOR` — while
    // `atmosphere.sunColor` goes on reporting the solar extinction colour at
    // every hour of the night. So the pair (`sunDirection`, `sunColor`) is the
    // obvious thing to hand over and is wrong after dark twice over: it would
    // anchor the flare where the sun is not, and paint it warm on a blue moon.
    // `_keyDirection` is already derived from `sunLight.position` further up, so
    // it follows the retarget; the colour has to come from the same object.
    // A deck thick enough to call an overcast leaves no disc to flare on.
    const flareThrough = 1 - THREE.MathUtils.clamp(this.atmosphere.overcast, 0, 1) * 0.97;
    this.lensFlare.setSource(_keyDirection, key.color, (key.intensity / 3.4) * flareThrough);
    this.lensFlare.setSubmersion(submersion);

    this.lensRain.setSubmersion(submersion);
    // Screen-space "down" for the droplets, leaned by the wind.
    //
    // **+y is down here.** The quad's uv runs top-down, the same way screen uv
    // does everywhere else in this project — it is NDC that runs bottom-up, which
    // is the asymmetry `clipToScreenUV` in `ScreenSpaceReflection` documents and
    // that the fog and underwater passes had to be corrected for. This was set to
    // -1 on the opposite assumption and the beads ran *up* the screen.
    //
    // Set explicitly rather than left to the effect's own default, because which
    // way rain runs down a lens is exactly the kind of thing that is obvious to a
    // viewer and invisible to every test we have.
    const lean = Math.min(0.5, this.state.windSpeed * 0.018);
    this.lensRain.setGravity(Math.sin(preset.sea.windDirection) * lean, 1, 1 + lean);
    this.lensRain.update(dt);

    // Volumetric fog follows the key light, and fades out as the camera goes
    // under. There is no atmosphere below the waterline — the medium down there
    // is the underwater pass's job, and leaving both on would stack two
    // different descriptions of the same water on top of each other.
    // The fog's sky-lit colour. Its far field settles on this colour, so it is
    // what the horizon shows wherever the fog runs out to the far plane. It was
    // never set and stayed at the default pale grey-blue at every hour — at
    // night a hundred times the dome's own radiance there, which drew a stark
    // white line along the night horizon between the dark sea and the dark
    // deck. Same rule as `_litFog`: never brightened, so daylight is untouched;
    // where the sky is the darker of the two, the colour goes over to the
    // dome's own horizon radiance, so the haze ends on the sky behind it.
    const fogBase = DEFAULT_VOLUMETRIC_FOG_PARAMS.color;
    const fogBaseLuma = luma(fogBase);
    const fogLit = Math.min(1, luma(_skyTypical) / Math.max(1e-5, fogBaseLuma));
    _fogColor.copy(_skyTypical).lerp(fogBase, fogLit);
    this.fog.setParams({
      color: _fogColor,
      sunDirection: _keyDirection,
      sunColor: this.atmosphere.sunLight.color,
      sunIntensity: Math.max(0.05, this.atmosphere.sunLight.intensity / 3.4),
      // Extinction per metre, from the preset, scaled by the slider.
      //
      // The slider's default of 0.35 is the *neutral* point — it means "as thick
      // as this place is" — and it scales to about 2.9x at the top. That is what
      // a fog control should do: a clear day made foggy is still a clear day's
      // light, and Foggy at the same slider position is still much thicker than
      // Sea of Thieves at it.
      //
      // This used to be `slider * 0.021` with no preset term at all, which put
      // 0.0074/m under every preset — around 400 m of visibility — and rendered
      // even Clear Day as a white-out. The preset numbers now carry the medium;
      // see `Preset.fog.volumetric`.
      // `cinematicFog` is 1 outside the tour, so this term vanishes there. Inside
      // it, the squall thickens the air between the viewer and everything else.
      density:
        preset.fog.volumetric *
        (this.state.fogDensity / DEFAULT_UI_STATE.fogDensity) *
        this.cinematicFog *
        (1 - submersion),
      windDirection: preset.sea.windDirection,
      windSpeed: 0.4 + this.state.windSpeed * 0.06,
    });
    this.fog.update(dt);

    // Particles only cost anything while they can actually be seen.
    this.particles.setVisible(submersion > 0.01);
    if (submersion > 0.01) this.particles.update(dt, this.camera.position);

    this.ssr?.update(dt);
    this.caustics.setSunDirection(this.atmosphere.sunDirection);
    this.caustics.update(dt);
    // Re-bake around the viewer. Must run outside an active render target, so it
    // sits here in the update rather than inside the post chain.
    this.caustics.bake(this.renderer, this.camera.position.x, this.camera.position.z);

    this.updateSceneContent(dt);

    // *After* the wake has been recentred and re-rendered, not before.
    //
    // `updateSceneContent` moves the buffer's world centre to the camera and
    // resamples the texture to match. Publishing the centre ahead of that handed
    // the surface the *previous* frame's anchor for a texture that had already
    // been re-anchored, so while the camera was moving the wake slid against the
    // hull by exactly one frame of camera travel — the one thing the whole
    // world-anchored design exists to prevent, reintroduced by an ordering.
    this.water.setFoamCenter(this.wake.centerX, this.wake.centerZ);

    this.hud.setFps(this.loop.stats.fps);
    this.updateHudNav();

    // The line, when the viewer takes the helm in weather. Driven from `elapsed`
    // rather than a wall clock so a deterministic rewind puts it back where it
    // was instead of leaving one fading across a capture.
    this.stormQuote.update(
      {
        preset: this.state.preset,
        windSpeed: this.state.windSpeed,
        atHelm: this.state.cameraMode === 'boat' && (this.shipControls?.isEnabled ?? false),
      },
      elapsed,
    );

    // Adaptive quality answers sustained *real-time* frame pressure, so it has
    // no business running while the clock is detached. `Loop.step` deliberately
    // does not write `stats.fps` — there is no wall-clock rate to report when
    // frames are being issued on demand — so the value here would be whatever
    // the live loop last saw. A capture taken after a heavy moment would then
    // inherit that reading and downgrade the tier partway through, quietly
    // changing the thing being measured.
    if (!this.loop.isPaused) {
      this.adaptive.update(dt, this.loop.stats.fps, this.state.quality, elapsed);
    }
  };

  /** Physics, wake and chase camera. No-ops cleanly until the models land. */
  private updateSceneContent(dt: number): void {
    // The controller runs *before* the solver, so this frame's throttle and
    // rudder are integrated by this frame's substeps.
    //
    // It used to sit inside the `ship` branch below, which is after
    // `buoyancy.update` — and the comment there claimed the opposite of what the
    // code did. The external force persists between frames, so the ship still
    // sailed and no test could see it; every input was simply acted on one frame
    // late. An independent review caught it by reading the call order rather
    // than the comment.
    // The cinematic flight's engine orders, forwarded only while it is running.
    //
    // Deliberately not "every frame in every mode", which is the obvious way to
    // guarantee a clean handoff: `setInput` is the same latch the on-screen
    // throttle writes to, so publishing the director's zeroes unconditionally
    // would stamp on the touch controls once per frame and leave them dead. The
    // handoff is made clean at the mode change instead, where it belongs.
    if (this.shipControls !== null && this.state.cameraMode === 'cinematic') {
      const orders = this.director.shipInput;
      this.shipControls.setInput(orders.throttle, orders.rudder);
    }
    this.shipControls?.update(dt);
    // Sails, anchor and cargo: adds to the forces the controller just set, so
    // it runs between the controller and the solve. See `src/gameplay`.
    this.gameplay?.update(dt);

    // Safe before the sampler's first readback resolves: it reports height 0 and
    // bodies simply settle to flat water rather than producing NaN.
    this.buoyancy.update(dt, this.sampler);

    const ship = this.ship;
    if (ship) {
      ship.update(dt);

      const position = ship.object.position;
      // Speed along the bow, from the body, rather than frame-to-frame distance.
      // Distance travelled cannot tell ahead from astern and counts the hull's
      // heave on a swell as forward motion, so a ship sitting still in a seaway
      // laid down a wake.
      const state = this.shipControls?.getState(this.shipStateOut) ?? null;
      const speed = state ? Math.abs(state.forwardSpeed) : 0;
      this.previousShipPosition.copy(position);

      this.wake.emit(position.x, position.z, ship.heading, speed, ship.hullBeam);

      this.chaseTarget.position.copy(position);
      this.chaseTarget.heading = ship.heading;
      this.director.setChaseTarget(this.chaseTarget);
    }

    // Centred on the viewer, not on the hull.
    //
    // It was the hull's while it only held the wake. Now that breaking crests
    // deposit into the same buffer it has to cover what is being *looked at*,
    // or the whitecaps stop at an invisible circle a few hundred metres from a
    // ship that may not even be on screen. The wake still deposits at the hull's
    // world position and simply falls out of the footprint when the camera
    // leaves it behind, which is the correct trade: foam you can see beats foam
    // you cannot.
    this.wake.setCenter(this.camera.position.x, this.camera.position.z);

    // Outside the `ship` branch: the foam has to keep decaying whether or not
    // anything is depositing into it, or a wake left by a ship that has since
    // been disposed would hang on the water forever.
    //
    // Must run outside an active render target; it saves and restores its own.
    this.wake.update(dt, this.renderer);
  }

  /**
   * Mean of the dome's radiance at the zenith and at four bearings 6 degrees
   * above the horizon. Five CPU evaluations of the sky model a frame.
   *
   * `horizon` receives the darkest of the four bearings: the dome away from
   * the sun's glow, which is the colour of the air under a continuous deck
   * (see the composite in `Clouds`). `typical` receives the mean of the middle
   * two by brightness: the horizon as most of it looks, with neither the sun's
   * bearing (64x the others at sunset, from the disc) nor the darkest one
   * deciding it. `zenith` receives the overhead sample.
   */
  private measureSkyRadiance(
    out: THREE.Color,
    horizon: THREE.Color,
    typical: THREE.Color,
    zenith: THREE.Color,
  ): THREE.Color {
    this.atmosphere.skyRadiance(_skyDirection.set(0, 1, 0), zenith);
    out.copy(zenith).multiplyScalar(0.5);
    for (let k = 0; k < 4; k++) {
      const a = (k * Math.PI) / 2;
      _skyDirection.set(Math.cos(a), 0.1, Math.sin(a)).normalize();
      const sample = this.atmosphere.skyRadiance(_skyDirection, _skyBearings[k]);
      out.r += sample.r * 0.125;
      out.g += sample.g * 0.125;
      out.b += sample.b * 0.125;
    }
    const sorted = _skyBearings.sort((a, b) => luma(a) - luma(b));
    horizon.copy(sorted[0]);
    typical.copy(sorted[1]).add(sorted[2]).multiplyScalar(0.5);
    return out;
  }

  private onResize = (): void => {
    // The circle of confusion is measured on a sensor and drawn in pixels, so
    // the conversion between them moves with the frame.
    this.dof?.setFrameHeight(this.renderer.getDrawingBufferSize(_drawingBuffer).y);
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight, false);
  };

  private restartWithBackend(forceWebGL: boolean): void {
    const url = new URL(window.location.href);
    if (forceWebGL) url.searchParams.set('webgl', '1');
    else url.searchParams.delete('webgl');
    window.location.replace(url.toString());
  }

  /** Deterministic control surface for the Playwright verification loop. */
  private exposeTestHooks(): void {
    Object.assign(window, {
      __ocean: {
        renderer: this.renderer,
        scene: this.scene,
        camera: this.camera,
        director: this.director,
        /** The audio system: `musicDebug`, `getMusicState()`, and the hooks. */
        audio: this.audio,
        simulation: this.simulation,
        /**
         * Reflection layers, exposed so a test can isolate them.
         *
         * The two are composited, not chosen between, and the tier drives both
         * from one number — so a test that only asks "does hiding the ship change
         * the water" is satisfied by the planar layer alone and would pass with
         * the screen-space trace completely broken. Null on the WebGL2 path,
         * which has neither.
         */
        reflections: this.reflections,
        ssr: this.ssr,
        /** Storm lightning, so a test can wait for a strike. */
        lightning: this.lightning,
        /** Sun cascades, local lights and the strike's light — see `lighting/`. */
        lighting: this.lighting,
        sampler: this.sampler,
        atmosphere: this.atmosphere,
        /**
         * The cloud layer, so `scripts/profile-frame.mjs` can vary its step
         * count without going through a tier change.
         *
         * A tier moves twenty other things at once, which is exactly what an
         * attribution measurement must not do — the whole point is to change one
         * variable. `setParams({ steps })` costs a uniform write and no
         * recompile, so the frames either side of it are comparable.
         */
        clouds: this.clouds,
        /**
         * The post chain, exposed for `scripts/profile-frame.mjs` only.
         *
         * Each of these has a step count, a tap count or an enable that can be
         * moved without a recompile, which is the property the attribution
         * harness needs — a tier change would move twenty things at once and
         * measure none of them. They are read-only handles; nothing here writes
         * through them outside the profiler.
         */
        post: {
          fog: this.fog,
          bloom: this.bloom,
          dof: this.dof,
          lensRain: this.lensRain,
          underwater: this.underwater,
          /** AO, TAAU, sharpen, motion blur, CA, grain, eye adaptation; `setOverride` for A/B. */
          chain: this.postChain,
        },
        loop: this.loop,
        backend: this.backend,
        wake: this.wake,
        water: this.water,
        /** The procedural reef, so a harness can find its sites and count it. */
        reefLife: this.reefLife,
        creatures: this.creatures,
        /**
         * Readiness signals. `sceneContentLoaded` settles whether or not the
         * models arrived, so a harness can distinguish "still loading" from
         * "loaded, and the ship legitimately is not there".
         */
        /**
         * The gameplay-facing HUD surfaces: toasts, the objective scroll and
         * the score plaque. See `Hud` for the signatures.
         */
        ui: {
          toast: (...args: Parameters<Hud['toast']>) => this.hud.toast(...args),
          setObjective: (...args: Parameters<Hud['setObjective']>) => this.hud.setObjective(...args),
          setScore: (...args: Parameters<Hud['setScore']>) => this.hud.setScore(...args),
          clearScore: (...args: Parameters<Hud['clearScore']>) => this.hud.clearScore(...args),
        },
        isReady: () =>
          this.sceneContentLoaded &&
          this.sampler.ready &&
          this.loop.stats.frameMs > 0 &&
          this.pendingPipelines === 0,
        sceneContentLoaded: () => this.sceneContentLoaded,
        hasShip: () => this.ship !== null,
        getState: () => ({ ...this.state }),
        /**
         * Returns a promise, and callers that change `quality` must await it.
         *
         * A tier change now waits for the GPU queue to drain before destroying
         * anything, so it genuinely is asynchronous. A test that asserted on
         * `renderer.info` straight after would otherwise be reading the tier it
         * had just left.
         */
        setState: async (partial: Partial<UiState>) => {
          // Quality first, and awaited before anything else runs.
          //
          // A tier change pauses the loop and waits on a GPU fence. Applying the
          // remaining keys synchronously on top of that is not safe: `preset`
          // re-captures the environment cube, which is a whole scene render
          // submitted *after* the fence was requested and before it resolved.
          // `{ quality, preset }` in one call is exactly what the visual harness
          // sends, so this was reachable on every shot.
          if (partial.quality !== undefined) {
            (this.state as unknown as Record<string, unknown>).quality = partial.quality;
            await this.onStateChange('quality');
          }
          // The preset next, before the keys it seeds. Choosing a preset now
          // resets the cloud coverage to the preset's own, so a caller sending
          // `{ cloudCoverage, preset }` — which is the order the visual harness
          // sends them in — has to see its explicit coverage applied after
          // the preset, not overwritten by it.
          const entries = Object.entries(partial).sort(
            ([a], [b]) => Number(b === 'preset') - Number(a === 'preset'),
          );
          for (const [key, value] of entries) {
            if (key === 'quality') continue;
            (this.state as unknown as Record<string, unknown>)[key] = value;
            this.onStateChange(key as keyof UiState);
          }
          this.panel.setState(partial);
        },
        /**
         * Places the camera exactly, for reproducible screenshots.
         *
         * Works in every mode, not just Orbit — see `CameraDirector.pin`.
         */
        setCamera: (px: number, py: number, pz: number, tx: number, ty: number, tz: number) => {
          _pinPosition.set(px, py, pz);
          _pinTarget.set(tx, ty, tz);
          this.director.pin(_pinPosition, _pinTarget);
        },

        // --- deterministic stepping ------------------------------------------
        /** Detaches the simulation from wall clock. `step` then owns the clock. */
        setPaused: (paused: boolean) => this.loop.setPaused(paused),
        isPaused: () => this.loop.isPaused,
        elapsedTime: () => this.loop.elapsedTime,
        /** Advances by exactly `steps` increments of `dt` and renders once. */
        step: (dt: number, steps = 1) => this.stepDeterministic(dt, steps),

        /**
         * Returns the world to a known state at simulation time `time`.
         *
         * Everything with a clock is rewound and every accumulation buffer is
         * cleared, so two calls with the same arguments produce the same frame
         * regardless of what the session did in between.
         */
        resetDeterministic: async (
          time = 0,
          settleSteps = 90,
          shipInput: { throttle: number; rudder: number } | null = null,
          cinematicTime: number | null = null,
        ) => {
          const settleDt = 1 / 60;
          // Claim the clock before anything else, and settle any tier change
          // that is already running — a drain in flight restores the pause state
          // it captured on entry, so both halves are needed to close the race.
          this.captureOwnsClock = true;
          this.loop.setPaused(true);
          this.postChain.setDeterministic(true);
          // Settle both things that pause the loop on their own account: the
          // scene load's compile pass and any tier change in flight. Awaiting
          // them is what makes the pause stick — the flag above only stops them
          // *undoing* it, and a capture that begins while shaders are still
          // compiling is not reproducible anyway.
          await this.contentReady?.catch(() => undefined);
          await this.qualityApply?.catch(() => undefined);
          this.loop.setPaused(true);
          // And the live frame that may still be drawing, which the pause does
          // not stop: the first step must not overlap it.
          await this.loop.settle();
          // One frame drawn into the capture target and thrown away, before the
          // history is cleared. The first render into a new capture target
          // changes what the frames after it draw — measured, at Low, as 3.3k
          // pixels between the first reset of a session and every later one,
          // and gone once any capture had been taken. Drawing it here, where
          // the history is about to be reset anyway, makes the first capture of
          // a session the same as the rest.
          {
            const previous = this.renderer.getRenderTarget();
            this.renderer.setRenderTarget(this.ensureCaptureTarget());
            this.beginNodeFrame();
            await this.renderFrame();
            this.renderer.setRenderTarget(previous);
          }
          this.resetTemporal();

          // Settle the LOD levels for wherever this shot's camera is, and build
          // any pipeline it newly needs, *before* the capture rather than during
          // it.
          //
          // Levels the boot prewarm skipped are compiled lazily when the viewer
          // crosses a switch, which for a harness means: a shot that moves the
          // camera in toward the island can start that compile mid-settle and
          // finish it somewhere unpredictable. The props would then appear
          // between two captures of the same shot, and the baseline would depend
          // on how fast the machine compiled — the exact class of order- and
          // timing-dependence the shot list exists to keep out.
          this.props?.updateLod(this.camera.position);
          // A compile a frame started first: two at once would each hold the
          // renderer for its own pass.
          await this.lodCompile?.catch(() => undefined);
          const lodLevels = this.props?.takeUncompiledLevels() ?? [];
          if (lodLevels.length > 0) await this.compileLodLevels(lodLevels);

          // Rewind far enough that the settle run *ends* exactly at `time`.
          // Setting the clock to `time` and then stepping forward would leave the
          // world at `time + settleSteps * dt`, and the caller's chosen time is
          // the one thing that has to be exact.
          const start = time - settleSteps * settleDt;
          this.loop.setElapsed(start);

          // The rain rate this shot should show, pushed in *before* anything
          // rewinds — several of these snap their own state to the current
          // intensity, and until now they were snapping to the previous shot's.
          //
          // `LensRain.resetClock` sets its coverage to `this.intensity`, which
          // `update()` had not yet been called to change: a capture taken after
          // the storm shot therefore inherited a soaked lens and dried it over a
          // 26 s constant that eight seconds of settling could not touch. The
          // droplets in a clear-sky gallery image were that, and it made the
          // canonical shots depend on the order they ran in — exactly what the
          // shot list's own header warns against.
          // The cinematic clock first, because with the tour driving the
          // weather, "what should the world look like at `start`" is a question
          // only the flight can answer.
          //
          // This used to sit forty lines below, after the seeds. That ordering
          // was correct while the preset owned the weather and is wrong now: a
          // night or squall capture would be seeded bone dry from a clear
          // preset, and wetness dries on a 26 s constant, so no number of settle
          // steps would recover it. The seeds have to come from the tour when
          // the tour is what is being photographed.
          // Two clocks, rewound together and from different origins.
          //
          // `start` is the *simulation* clock — the sea, the foam, the wake.
          // The flight rides its own 60 s lap, and which second of that lap a
          // shot wants is an independent choice: a caller can ask for a settled
          // sea at t = 40 framed by the tour's night squall. Passing `null` keeps
          // the old behaviour of running both from the same number, which is
          // what every non-cinematic shot wants.
          const cinematicStart =
            cinematicTime === null ? start : cinematicTime - settleSteps * settleDt;
          this.director.resetCinematic(cinematicStart);
          const tour =
            this.state.cameraMode === 'cinematic'
              ? this.director.cinematicEnvironment(cinematicStart)
              : null;
          if (tour) this.weather.setKind(tour.weatherKind);
          const resetRain =
            this.rainOverride ?? tour?.rain ?? getPreset(this.state.preset).weather.intensity;
          this.lensRain.setIntensity(resetRain);
          this.weather.setIntensity(resetRain);
          this.wake?.setRainAgitation(resetRain);

          this.weather.resetClock(start);
          this.particles.resetClock(start);
          this.underwater.resetClock(start);
          this.fog.resetClock(start);
          this.lensRain.resetClock(start);
          this.caustics.resetClock(start);
          this.atmosphere.resetClock(start);
          // Force the environment cube, rather than waiting for the elevation
          // threshold to be crossed some frames into the settle.
          //
          // The throttle in `refreshTourEnvironment` exists so the cost is not
          // paid every frame, and it means the *first* settled frame after a
          // rewind would otherwise be lit by whichever hour the previous shot
          // left in the cube. For a night capture taken after a daylight one
          // that is the whole difference between the shot and a bug.
          if (this.state.cameraMode === 'cinematic') {
            this.atmosphere.setParams(this.sunFromClock(this.director.cinematicEnvironment(cinematicStart).hours));
            this.atmosphere.update(0);
            this.refreshTourEnvironment(true);
          }
          this.clouds.resetWind();
          this.birds.resetClock(start);
          this.spray.resetClock(start);
          this.fish.resetClock(start);
          this.kelp.resetClock(start);
          this.reefLife.resetClock(start);
          this.creatures.resetClock(start);
          this.meadow.resetClock(start);
          this.stormQuote.reset();
          this.canopy.resetClock(start);
          this.foliageWind.resetClock(start);
          this.props?.resetClock(start);
          this.audio.resetClock(start);
          // `resetCinematic` is deliberately *not* here any more — it now runs
          // before the weather seeds above, because those seeds are read from the
          // flight and cannot be taken before its clock is set.
          this.wake?.reset(this.renderer);

          // Floating bodies carry position and momentum across a whole session;
          // returning them to their spawn poses is what stops a capture from
          // inheriting wherever the hull happened to have drifted.
          this.buoyancy?.resetToHome();
          // Zero first, then re-apply, so a caller that passes no input always
          // gets a stationary hull regardless of what the previous shot left on
          // the throttle. A caller that *does* pass one gets it applied before
          // the settle rather than after, which is the whole point: the ship has
          // a ~6.7 s velocity time constant, so an input applied after settling
          // would photograph a hull that has not begun to move.
          //
          // `resetInput`, not `setInput(0, 0)`: the throttle and rudder spool
          // toward their order over seconds, so clearing the order alone left the
          // spool open and the hull under way from the first settle step. See
          // `ShipController.resetInput` — this was the whole of the shot-ordering
          // dependence.
          this.shipControls?.resetInput();
          // Sails furled, anchor up, score and cargo sequence back to the seed.
          this.gameplay?.reset();
          // Set, not settled. Wetness dries with a 26 s time constant, so a
          // capture that inherited a storm's wet hull would still be visibly damp
          // three hundred settle steps later.
          this.wetness.setWetness(resetRain);
          if (shipInput) this.shipControls?.setInput(shipInput.throttle, shipInput.rudder);
          this.ship?.resetClock(start);
          this.previousShipPosition.copy(this.ship?.object.position ?? this.previousShipPosition);

          // The sampler holds a readback of the *previous* wave field; drop it so
          // buoyancy re-derives from the field at `time`.
          //
          // And put the field itself at the rewound clock first. Each step's
          // physics reads the field the step before it computed, so the first
          // settle step would otherwise float the hull on whatever the GPU last
          // evaluated — the previous shot's sea, or the live session's. That
          // made the first capture after boot differ from every later one, and
          // made each capture depend on the shot that ran before it.
          this.simulation.update(start);
          this.sampler.rebuild();

          // Settle: the FFT is stateless in time (h(k, t) is evaluated directly),
          // but buoyancy, wake and foam all integrate, so they need real steps to
          // reach the state that simulation time implies. Stepping goes through
          // the synchronous sampler path — see `stepDeterministic`.
          await this.stepDeterministic(settleDt, settleSteps);

          // And again, because the settle can *move* the camera. A cinematic
          // shot pins no pose — the tour drives it — so the levels wanted at the
          // end of the settle are not the ones wanted at its start, and the
          // capture that follows must not be the frame a compile happens to land
          // in. Cheap when nothing has changed: the deal early-exits until the
          // camera has moved 25 m, and the queue is empty on every ordinary shot.
          this.props?.updateLod(this.camera.position);
          // The last settle step may have started one of its own.
          await this.lodCompile?.catch(() => undefined);
          const settledLevels = this.props?.takeUncompiledLevels() ?? [];
          // Compiled, not stepped. Revealing a level changes what the next
          // *render* draws and nothing about the simulation, and `capturePixels`
          // renders on demand — so the levels are in the captured frame without
          // advancing any clock. An extra step here would have ended the settle
          // at `time + settleDt` and quietly broken the one guarantee this whole
          // function is built around.
          if (settledLevels.length > 0) await this.compileLodLevels(settledLevels);
        },

        shadersReady: () => this.shadersReady,
        /** Live ship controller state — throttle, rudder, speed, heading. */
        shipState: () =>
          this.shipControls ? { ...this.shipControls.getState(this.shipStateOut) } : null,
        shipControlsEnabled: () => this.shipControls?.isEnabled ?? false,
        /** Whether the on-screen throttle/rudder is built and showing. */
        touchControlsVisible: () => this.touchControls?.isVisible ?? false,
        /** Current rain wetting of the hull and props, 0..1. */
        surfaceWetness: () => this.wetness.value,
        /**
         * Drives wetting directly, without going through the weather.
         *
         * Wetness used to be observable from the CPU, because it was a scalar
         * written onto every material — so a test could read `material.roughness`
         * and see it move. It is a node graph now, evaluated per fragment against
         * the world normal, and the only place the result exists is the rendered
         * image. Reaching it through the rain rate would change the sky, the fog,
         * the lens and the sea state at the same time, which is no way to measure
         * one thing.
         */
        setSurfaceWetness: (value: number) => this.wetness.setWetness(value),
        /** Direct throttle/rudder input, bypassing the keyboard. */
        setShipInput: (throttle: number, rudder: number) =>
          this.shipControls?.setInput(throttle, rudder),
        /**
         * Forces the rain rate the surface sees, independently of the preset's
         * weather. Lets a test vary one input while holding the sea state,
         * lighting and camera fixed, which comparing two presets could not.
         * `null` returns control to the weather system.
         */
        setRainOverride: (intensity: number | null) => {
          this.rainOverride = intensity;
        },
        /**
         * Forces foam and surf strength together, or `null` to hand both back to
         * the weather system. Zero is the "no foam anywhere" reference frame that
         * separates whitecaps from the sky the water is reflecting.
         */
        setFoamOverride: (strength: number | null) => {
          this.foamOverride = strength;
          // Written straight through, and both halves of that matter.
          //
          // **It must be written now.** The surf master is set every frame, but
          // foam strength is set by `applyPreset`, which runs on a *state
          // change* rather than per frame. Setting the field alone left the
          // uniform holding its old value until something unrelated happened to
          // change state, and the override read as completely inert — foam 0 and
          // foam 3 rendered bit-identically, which a measurement with no control
          // would have reported as "there is no foam here".
          //
          // **And it must not go through `applyPreset`,** which is what the
          // first fix did. That re-applies the atmosphere, the clouds, the
          // weather, every spectrum texture, the wake's breaking parameters, the
          // exposure, the bloom and the colour grade — so a hook that claims to
          // move one uniform would silently reset any other override a test had
          // set, `setGrade` and `setBloomEnabled` included. An independent review
          // caught that; `weatherFoamStrength` is cached instead so lifting the
          // override needs nothing recomputed.
          this.water.setFoamStrength(strength ?? this.weatherFoamStrength);
        },
        /**
         * Test-only grade override.
         *
         * Colours arrive as plain triples because this crosses `page.evaluate`,
         * which structured-clones its argument and would strip a `THREE.Color`
         * down to a bare object with no `copy` on it.
         */
        setGrade: (g: {
          slope: [number, number, number];
          offset: [number, number, number];
          power: [number, number, number];
          saturation: number;
        }) =>
          this.colorGrade.setParams({
            slope: new THREE.Color(g.slope[0], g.slope[1], g.slope[2]),
            offset: new THREE.Color(g.offset[0], g.offset[1], g.offset[2]),
            power: new THREE.Color(g.power[0], g.power[1], g.power[2]),
            saturation: g.saturation,
          }),
        /**
         * Test-only dither amplitude, in output levels. 0 disables it exactly,
         * which is how a test proves the output takeover reproduces the
         * renderer's own path.
         */
        setDitherLevels: (levels: number) => this.outputTransform.setDitherLevels(levels),
        /**
         * Test-only anti-alias blend, 0..1. 0 is the unfiltered frame exactly,
         * which is how a test proves the filter is in the chain and doing
         * something rather than being wired up and inert.
         */
        setAaStrength: (strength: number) => this.spatialAa.setStrength(strength),
        /**
         * Test-only spray master scale. 0 removes it from the frame exactly,
         * which is how a test attributes pixels to it rather than to the wake
         * and the whitecaps it appears on top of.
         */
        setSprayStrength: (strength: number) => this.spray.setStrength(strength),
        /** Live impact events, for the spray test. */
        sprayLiveEvents: () => this.spray.liveEvents(),
        /** The spray system itself, for probing its event table. */
        spray: this.spray,
        /** The tour's environment curves, for the seam and coverage tests. */
        cinematicEnvironment: (time?: number) => ({
          ...this.director.cinematicEnvironment(time),
        }),
        /** Beat names, start times and durations, for the framing tests. */
        cinematicBeats: () => CINEMATIC_BEATS.map((b) => ({ ...b })),
        /** Nominal hull position at a loop time; see `Cinematic.nominalShipXZ`. */
        nominalShipAt: (time: number) => {
          const p = nominalShipXZ(time, _shipProbe);
          return { x: p.x, z: p.y };
        },
        /** Test-only flare override. */
        setFlareEnabled: (on: boolean) => this.lensFlare.setEnabled(on),
        /** Test-only lens override: tap count and f-number. */
        setDof: (samples: number, fNumber: number) => {
          this.dof.setSamples(samples);
          this.dof.setAperture(fNumber);
        },
        /** Test-only bloom override, so a test can prove the stage contributes. */
        setBloomEnabled: (on: boolean) => this.bloom.setEnabled(on),
        /** Exact-pixel frame capture; see `App.capturePixels`. */
        capturePixels: () => this.capturePixels(),
        dispose: () => this.dispose(),
      },
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    window.removeEventListener('resize', this.onResize);
    this.loop?.stop();
    this.panel?.dispose();
    this.detachUiSounds?.();
    this.detachUiSounds = null;
    this.hud?.dispose();
    this.touchControls?.dispose();
    this.director?.dispose();
    this.sampler?.dispose();
    this.simulation?.dispose();
    this.oceanMesh?.dispose();
    this.water?.dispose();
    this.atmosphere?.dispose();
    this.clouds?.dispose();
    this.birds?.dispose();
    this.fish?.dispose();
    this.kelp?.dispose();
    this.meadow?.dispose();
    this.stormQuote?.dispose();
    this.canopy?.dispose();
    this.remains?.dispose();
    this.audio?.dispose();
    this.weather?.dispose();
    this.underwater?.dispose();
    this.fog?.dispose();
    this.lensRain?.dispose();
    // Every stage in `src/post/` is torn down here, including the ones that own
    // nothing. `RenderPipeline.dispose` disposes only its own quad material and
    // does not traverse the node graph, so anything in the chain that holds a
    // render target has to be released from this list or it is simply leaked.
    this.colorGrade?.dispose();
    this.bloom?.dispose();
    this.dof?.dispose();
    this.lensFlare?.dispose();
    this.outputTransform?.dispose();
    this.spatialAa?.dispose();
    this.postChain?.dispose();
    this.spray?.dispose();
    this.particles?.dispose();
    this.caustics?.dispose();
    this.shipControls?.dispose();
    this.gameplay?.dispose();
    // Restores the dry roughness and colour on materials the loader's cache
    // shares, so a re-created App does not inherit a permanently wet ship.
    this.wetness.dispose();
    this.lighting?.dispose();
    this.buoyancy?.dispose();
    this.wake?.dispose();
    this.ship?.dispose();
    this.props?.dispose();
    this.seafloor?.dispose();
    this.reflections?.dispose();
    this.ssr?.dispose();
    this.assets?.dispose();
    this.captureTarget?.dispose();
    this.renderer?.dispose();
  }
}

const canvas = document.getElementById('viewport') as HTMLCanvasElement | null;
const uiRoot = document.getElementById('ui-root');

if (!canvas || !uiRoot) {
  boot.fail('The page did not load completely. Reload to try again.');
} else {
  void openStartGate(boot.root as HTMLElement)
    .then((selection) => new App(canvas, uiRoot, selection).start())
    .catch((error: unknown) => {
    console.error(error);
    // What happened and what to do about it; the error's own words go under
    // it. Try again reloads to the menu, where a lighter choice can be made.
    boot.fail(
      'The sea could not be launched on this device. '
        + 'Try again, and if it fails twice, choose WebGL2 and Low on the menu.',
      error instanceof Error ? error.message : '',
    );
    });
}
