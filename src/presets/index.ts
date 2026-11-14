import * as THREE from 'three';
import type { ColorGradeParams } from '../post/ColorGrade';
import type { PresetId } from '../ui/types';
import type { WaterAppearance } from '../ocean/OceanMaterial';
import type { SpectrumParams } from '../ocean/Spectrum';

export type WeatherKind = 'clear' | 'rain' | 'snow';

export interface Preset {
  id: PresetId;
  label: string;
  /** Sky and lighting. */
  atmosphere: {
    sunElevation: number;
    sunAzimuth: number;
    turbidity: number;
    rayleigh: number;
    mieCoefficient: number;
    mieDirectionalG: number;
    exposure: number;
    /**
     * How far toward flat overcast light the sky is pulled, 0..1.
     *
     * Required, not optional, and that is the entire point. `Atmosphere.setParams`
     * merges rather than replaces — it has to, because the time-of-day slider
     * spreads a sun position over whatever preset is loaded — so a key that only
     * some presets mention is a key that leaks. Two of the ten set this and the
     * other eight left it out, so selecting Storm and then anything else kept the
     * storm's flat grey sky forever: the sun, the clouds and the water all changed
     * and the light did not. Making it non-optional means the compiler, not a
     * reviewer, is what notices the next preset that forgets it.
     */
    overcast: number;
    nightIntensity: number;
    moonElevation: number;
    moonAzimuth: number;
  };
  clouds: {
    coverage: number;
    density: number;
    altitude: number;
    thickness: number;
    color: THREE.Color;
    shadowColor: THREE.Color;
    /**
     * Relief on the deck's underside, 0..1 — flat fair-weather bases at the
     * low end, a storm's hanging lobes at the top. Required for the reason
     * `atmosphere.overcast` is: `Clouds.setParams` merges, so a key only the
     * storm set would leak into every preset chosen after it.
     */
    baseRoughness: number;
    /** High cirrus veil, 0..1. Required for the same reason. */
    cirrus: number;
  };
  weather: { kind: WeatherKind; intensity: number };
  /**
   * Sea state.
   *
   * `standingWaveRatio` and `spectralSharpness` are here rather than left on
   * their defaults because they are the two parameters that separate one body of
   * water from another at the same wind speed. Sharpness decides whether the sea
   * is combed downwind or confused; the standing ratio decides whether it runs
   * at all. A sheltered lagoon and an open gale can share a wind vector and
   * agree on nothing else.
   *
   * `windwardFoam` is the deposit rate for the wind-facing-face term in
   * `Wake.setWindward`, before the wind law scales it.
   */
  sea: Pick<
    SpectrumParams,
    | 'windSpeed'
    | 'windDirection'
    | 'peakWavelength'
    | 'gamma'
    | 'swell'
    | 'standingWaveRatio'
    | 'spectralSharpness'
  > & { windwardFoam: number };
  water: Partial<WaterAppearance>;
  /**
   * Aerial perspective applied to the water surface (`density`), and the
   * extinction of the *volumetric* fog pass (`volumetric`). Both are per metre.
   *
   * Two numbers rather than one because they are two different media doing two
   * different jobs. `density` is the distance haze baked into the water shader:
   * flat, cheap, and the thing that makes the horizon recede. `volumetric` is
   * the marched pass, which is what gives light shafts and lets the fog have
   * structure — and being a real integral along the view ray, it stacks on top
   * of the first rather than replacing it.
   *
   * `volumetric` is per preset for the same reason `density` is. It was a single
   * global constant of 0.0074/m, which is roughly 400 m of visibility — sixty
   * times Sea of Thieves' aerial perspective and four times Foggy's — so every
   * preset, including the clear ones, rendered as a white-out. Anything that
   * describes the medium belongs to the place, not to the renderer.
   */
  fog: { color: THREE.Color; density: number; volumetric: number };
  /**
   * Underwater medium.
   *
   * `distortion` is the refractive warp of the submerged image, in uv. It is per
   * preset because it is a property of the water: a storm's surface is a
   * disturbed lens and an arctic calm is very nearly a flat one, and a submerged
   * frame that swims by the same amount in both is one of the tells that the two
   * are the same render with a different palette.
   */
  underwater: {
    color: THREE.Color;
    distortion: number;
    extinction: THREE.Vector3;
    visibility: number;
    godRayStrength: number;
  };
  toneMappingExposure: number;
  /**
   * Global colour grade, applied in linear before ACES. See `src/post/ColorGrade.ts`.
   *
   * Required rather than optional, for the reason `atmosphere.overcast` above is:
   * `ColorGrade.setParams` merges, so a key only some presets mention is a key
   * that leaks from whichever preset was selected before.
   *
   * Identity means "this preset is already the colour it wants to be", which is
   * a real answer and not a placeholder — the grade reinforces a look, it does
   * not supply one.
   */
  grade: ColorGradeParams;
}

const color = (hex: number) => new THREE.Color(hex);
const vec = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
// Every preset now declares its own grade, so there is no shared identity to
// clone. Note that each one builds fresh `Color` objects: sharing them between
// presets would mean a single mutation anywhere silently regraded the whole set.

/**
 * Nine environment looks. Each one moves the sun, the sea state, the medium and
 * the aerial perspective together — changing only the sky is what makes preset
 * switches in most demos look like a filter rather than a different place.
 *
 * Extinction triples follow real water: red is absorbed roughly an order of
 * magnitude faster than blue, so the third component is always the smallest.
 */
export const PRESETS: Record<PresetId, Preset> = {
  skyPro: {
    id: 'skyPro',
    label: 'Clear Day',
    atmosphere: {
      sunElevation: 0.46,
      sunAzimuth: 2.4,
      turbidity: 2.2,
      rayleigh: 1.5,
      mieCoefficient: 0.004,
      mieDirectionalG: 0.8,
      overcast: 0,
      exposure: 1,
      nightIntensity: 0,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.32,
      density: 0.9,
      altitude: 1600,
      thickness: 700,
      color: color(0xffffff),
      shadowColor: color(0x8fa8c4),
      baseRoughness: 0.1,
      cirrus: 0.45,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 15,
      windDirection: Math.PI * 0.25,
      peakWavelength: 47,
      gamma: 3.3,
      swell: 0.2,
      standingWaveRatio: 0.05,
      spectralSharpness: 1,
      windwardFoam: 0.03,
    },
    water: {
      deepColor: color(0x04283c),
      shallowColor: color(0x1fa9a6),
      scatterColor: color(0x36bda8),
      extinction: vec(0.32, 0.1, 0.06),
      roughness: 0.07,
      foamThreshold: 0.42,
    },
    /**
     * A correctness fix to the haze, and explicitly **not** a fix to the far
     * sea — which is what it was attempted as.
     *
     * The far sea measures sRGB (123, 141, 154), a saturation of 0.20, while the
     * near sea measures (7, 79, 108) at 0.94 and matches the reference almost
     * exactly. A defect that grows with distance looks like a haze defect, and
     * 0xbfd8ee is very nearly white, so the reasoning was that every kilometre of
     * it drove the water toward grey. Moving the colour to the horizon band this
     * preset actually renders and nearly halving the density moved the far sea
     * by **three levels**, from 0.19 saturation to 0.20. The hypothesis was
     * wrong, and it is the second one to be: weighting the haze by `1 - fresnel`
     * was tried before it and moved the same sample by one level.
     *
     * What the two null results together say is that the pale far water is not
     * the haze at all — it is grazing-angle reflection of a horizon sky, plus the
     * absence of the shallow turquoise the reference carries in the same band.
     * That is recorded as open rather than papered over.
     *
     * The change is kept on its own merits, which are real but smaller than they
     * were hoped to be: aerial perspective is scattered *skylight*, so a neutral
     * is the wrong colour for it whatever it happens to be washing out, and
     * 0.00016/m is about 27% haze at two kilometres — a hazy day, not the clear
     * one this preset is for.
     */
    fog: { color: color(0x7ea8cc), density: 0.00009, volumetric: 0.00012 },
    /**
     * Lifted when the surface grade came down.
     *
     * `toneMappingExposure` went from 1 to 0.42 to stop the above-water frame
     * sitting on the ACES shoulder, and that cut lands on the underwater frame
     * *as well as* the medium's own attenuation — so the dive, which was tuned
     * against the old exposure, came out two and a half stops under. Measured on
     * the tour's own reef leg, the coral fifteen metres away was within a few
     * levels of the water behind it.
     *
     * Visibility from 45 m to 62 m and a brighter inscatter, rather than a
     * larger `godRayStrength`: the shafts were never the problem. What was
     * missing is the diffuse reach that separates a coral head from the blue
     * behind it, and that is what the extinction length buys.
     */
    underwater: {
      color: color(0x2f8fb5),
      distortion: 0.01,
      extinction: vec(0.22, 0.072, 0.045),
      visibility: 62,
      godRayStrength: 1.1,
    },
    /**
     * 0.42, and this is a correction rather than a look.
     *
     * The scene's key is 3.4, its hemisphere fill is 1, and on top of both sits
     * a full sky-cube IBL — which together put a sunlit dielectric well up the
     * ACES shoulder at an exposure of 1. That is not a small stylistic
     * difference. Measured on the hero island frame against
     * `docs/ref/tropical-island-sea-level-v2.png`: the open sea came out
     * (39, 125, 158) against the reference's (10, 62, 95), the sky 15% down from
     * the top came out (214, 221, 228) — white — against (105, 149, 190), and
     * the vegetated slopes rendered at a saturation of 0.05 against 0.38.
     *
     * The saturation is the tell, and it is why the answer is exposure rather
     * than albedo. ACES desaturates as it compresses: everything pushed onto the
     * shoulder converges on white, so a scene rendered two and a half stops hot
     * does not read as "bright" — it reads as *milky*, which is exactly what the
     * brief lists as a failure mode. Darkening the albedos instead would have
     * fixed one surface and left the sky, the sea and every prop where they
     * were.
     *
     * Verified by isolation: with the terrain's albedo forced to black and its
     * environment contribution to zero, the island renders black. There is no
     * veil in the post chain. The frame was simply over-exposed.
     */
    toneMappingExposure: 0.42,
    grade: {
      // The reference image, so barely touched: a whisker of contrast and
      // saturation, and the lightest vignette of the nine. A preset the whole
      // project is tuned against should not be the one carrying a look.
      slope: color(0xffffff),
      offset: new THREE.Color(0, 0, 0),
      power: new THREE.Color(1.03, 1.02, 1.0),
      saturation: 1.04,
      vignette: 0.16,
    },
  },

  arctic: {
    id: 'arctic',
    label: 'Arctic',
    atmosphere: {
      sunElevation: 0.16,
      sunAzimuth: 1.6,
      turbidity: 1.6,
      rayleigh: 2.4,
      mieCoefficient: 0.003,
      mieDirectionalG: 0.75,
      overcast: 0,
      exposure: 1.1,
      nightIntensity: 0,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.55,
      density: 0.7,
      altitude: 1200,
      thickness: 500,
      color: color(0xeaf3ff),
      shadowColor: color(0x9db4cc),
      baseRoughness: 0.15,
      cirrus: 0.6,
    },
    weather: { kind: 'snow', intensity: 0.45 },
    sea: {
      windSpeed: 11,
      windDirection: Math.PI * 0.8,
      peakWavelength: 38,
      gamma: 2.4,
      swell: 0.3,
      standingWaveRatio: 0.05,
      spectralSharpness: 0.85,
      windwardFoam: 0.05,
    },
    water: {
      deepColor: color(0x081f2e),
      shallowColor: color(0x2f8fa8),
      scatterColor: color(0x59c8d4),
      extinction: vec(0.4, 0.16, 0.1),
      roughness: 0.09,
      foamThreshold: 0.5,
    },
    fog: { color: color(0xd8e6f2), density: 0.00034, volumetric: 0.00028 },
    underwater: {
      color: color(0x14526e),
      distortion: 0.014,
      extinction: vec(0.34, 0.13, 0.08),
      visibility: 30,
      godRayStrength: 0.7,
    },
    toneMappingExposure: 1.05,
    grade: {
      // Cold light on ice. Blue slope, a hair of blue lift in the shadows so
      // they read as snow-shadow rather than as black, and saturation up
      // slightly because the only colour in the frame is the sea.
      slope: new THREE.Color(0.96, 1.0, 1.07),
      offset: new THREE.Color(0.0, 0.002, 0.006),
      power: new THREE.Color(1.02, 1.0, 0.98),
      saturation: 1.06,
      vignette: 0.24,
    },
  },

  blackFlag: {
    id: 'blackFlag',
    label: 'High Seas',
    atmosphere: {
      sunElevation: 0.55,
      sunAzimuth: 3.1,
      turbidity: 3.4,
      rayleigh: 1.2,
      mieCoefficient: 0.006,
      mieDirectionalG: 0.82,
      overcast: 0,
      exposure: 1,
      nightIntensity: 0,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.4,
      density: 1.1,
      altitude: 1800,
      thickness: 900,
      color: color(0xfff6e8),
      shadowColor: color(0x7f93ad),
      baseRoughness: 0.1,
      cirrus: 0.35,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 13,
      windDirection: Math.PI * 0.1,
      peakWavelength: 55,
      gamma: 3.3,
      swell: 0.25,
      standingWaveRatio: 0.03,
      spectralSharpness: 1.15,
      windwardFoam: 0.05,
    },
    water: {
      deepColor: color(0x04303f),
      shallowColor: color(0x18b5a4),
      scatterColor: color(0x3fd0b0),
      extinction: vec(0.28, 0.085, 0.05),
      roughness: 0.06,
      foamThreshold: 0.44,
    },
    fog: { color: color(0xc8e0ee), density: 0.00014, volumetric: 0.0001 },
    underwater: {
      color: color(0x1c7f92),
      distortion: 0.012,
      extinction: vec(0.24, 0.075, 0.045),
      visibility: 55,
      godRayStrength: 1.15,
    },
    toneMappingExposure: 1,
    grade: {
      // Warm midtones over deep blacks — the look the preset is named for.
      // Negative offset is doing the work: it crushes the aerial haze out of
      // the shadows, which is what separates this from Sea of Thieves.
      slope: new THREE.Color(1.06, 1.0, 0.93),
      offset: new THREE.Color(-0.008, -0.006, -0.004),
      power: new THREE.Color(1.06, 1.04, 1.02),
      saturation: 1.02,
      vignette: 0.28,
    },
  },

  dusk: {
    id: 'dusk',
    label: 'Dusk',
    atmosphere: {
      sunElevation: 0.045,
      sunAzimuth: 4.3,
      turbidity: 5,
      rayleigh: 2.6,
      mieCoefficient: 0.009,
      mieDirectionalG: 0.86,
      overcast: 0,
      exposure: 0.95,
      nightIntensity: 0.25,
      moonElevation: 0.35,
      moonAzimuth: 1.4,
    },
    clouds: {
      coverage: 0.42,
      density: 1,
      altitude: 2000,
      thickness: 800,
      color: color(0xffd9b3),
      shadowColor: color(0x5a5470),
      baseRoughness: 0.1,
      cirrus: 0.7,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 7,
      windDirection: Math.PI * 1.2,
      peakWavelength: 42,
      gamma: 2.8,
      swell: 0.35,
      standingWaveRatio: 0.25,
      spectralSharpness: 1.25,
      windwardFoam: 0.02,
    },
    water: {
      deepColor: color(0x08192b),
      shallowColor: color(0x1a5f74),
      scatterColor: color(0x8a6a52),
      extinction: vec(0.34, 0.13, 0.09),
      roughness: 0.05,
      foamThreshold: 0.46,
    },
    fog: { color: color(0x9a8ea6), density: 0.00026, volumetric: 0.00022 },
    underwater: {
      color: color(0x11364f),
      distortion: 0.008,
      extinction: vec(0.36, 0.14, 0.09),
      visibility: 26,
      godRayStrength: 0.55,
    },
    toneMappingExposure: 0.95,
    grade: {
      // Cool shadows under a warm sky, which is what dusk actually looks like:
      // the sun is reddened and everything it is not lighting is lit by the
      // blue half of the sky instead. Mild desaturation as the eye gives up
      // colour with the light.
      slope: new THREE.Color(0.99, 0.99, 1.05),
      offset: new THREE.Color(0.0, 0.002, 0.008),
      power: new THREE.Color(1.0, 1.0, 0.98),
      saturation: 0.94,
      vignette: 0.3,
    },
  },

  foggy: {
    id: 'foggy',
    label: 'Foggy',
    atmosphere: {
      sunElevation: 0.3,
      sunAzimuth: 2,
      // Same correction as Storm, less of it: fog is a bright diffuse overcast
      // rather than a dark one.
      turbidity: 5,
      rayleigh: 1.3,
      mieCoefficient: 0.012,
      mieDirectionalG: 0.7,
      overcast: 0.7,
      exposure: 1,
      nightIntensity: 0,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.85,
      density: 0.6,
      altitude: 900,
      thickness: 600,
      color: color(0xdfe6ea),
      shadowColor: color(0xa8b4bd),
      baseRoughness: 0.3,
      cirrus: 0.0,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 5,
      windDirection: Math.PI * 0.6,
      peakWavelength: 30,
      gamma: 2,
      swell: 0.4,
      standingWaveRatio: 0.4,
      spectralSharpness: 0.9,
      windwardFoam: 0.02,
    },
    water: {
      deepColor: color(0x14252c),
      shallowColor: color(0x3f7b82),
      scatterColor: color(0x5f9296),
      extinction: vec(0.42, 0.2, 0.14),
      roughness: 0.1,
      foamThreshold: 0.5,
    },
    fog: { color: color(0xccd6da), density: 0.0016, volumetric: 0.0028 },
    underwater: {
      color: color(0x2a5560),
      distortion: 0.016,
      extinction: vec(0.46, 0.24, 0.17),
      visibility: 16,
      godRayStrength: 0.35,
    },
    toneMappingExposure: 1,
    grade: {
      // Flattened deliberately. Power below 1 lifts the midtones, a positive
      // offset lifts the blacks off the floor, and the saturation comes well
      // down — fog is a low-contrast, low-chroma medium and a preset that
      // renders it punchy is not rendering fog.
      slope: new THREE.Color(0.98, 0.99, 1.0),
      offset: new THREE.Color(0.014, 0.015, 0.017),
      power: new THREE.Color(0.96, 0.96, 0.96),
      saturation: 0.84,
      vignette: 0.14,
    },
  },

  moonlit: {
    id: 'moonlit',
    label: 'Moonlit',
    atmosphere: {
      sunElevation: -0.35,
      sunAzimuth: 5,
      turbidity: 1.4,
      rayleigh: 0.6,
      mieCoefficient: 0.002,
      mieDirectionalG: 0.8,
      overcast: 0,
      exposure: 1.4,
      nightIntensity: 1,
      moonElevation: 0.7,
      moonAzimuth: 2.1,
    },
    clouds: {
      coverage: 0.22,
      density: 0.8,
      altitude: 1900,
      thickness: 700,
      color: color(0x9fb0c8),
      shadowColor: color(0x18202e),
      baseRoughness: 0.1,
      cirrus: 0.5,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 6,
      windDirection: Math.PI * 1.7,
      peakWavelength: 80,
      gamma: 2.2,
      swell: 0.45,
      standingWaveRatio: 0.35,
      spectralSharpness: 1.2,
      windwardFoam: 0.02,
    },
    water: {
      deepColor: color(0x020a12),
      shallowColor: color(0x0d3f4a),
      scatterColor: color(0x18525a),
      extinction: vec(0.36, 0.14, 0.09),
      roughness: 0.045,
      foamThreshold: 0.48,
    },
    fog: { color: color(0x0d1926), density: 0.0003, volumetric: 0.00024 },
    underwater: {
      color: color(0x07202e),
      distortion: 0.01,
      extinction: vec(0.38, 0.16, 0.1),
      visibility: 22,
      godRayStrength: 0.3,
    },
    toneMappingExposure: 1.35,
    grade: {
      // Strongly blue and strongly desaturated, which is the Purkinje shift
      // rather than a stylistic choice: at scotopic levels the eye's rods take
      // over, sensitivity moves toward blue and colour discrimination all but
      // disappears. Contrast is raised because the alternative at these levels
      // is an even grey.
      slope: new THREE.Color(0.89, 0.97, 1.13),
      offset: new THREE.Color(0.0, 0.0, 0.003),
      power: new THREE.Color(1.07, 1.04, 1.0),
      saturation: 0.78,
      vignette: 0.34,
    },
  },

  seaOfThieves: {
    id: 'seaOfThieves',
    label: 'Tropical',
    atmosphere: {
      sunElevation: 0.62,
      sunAzimuth: 1.9,
      turbidity: 2.6,
      rayleigh: 1.3,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      overcast: 0,
      exposure: 1.05,
      nightIntensity: 0,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.36,
      density: 1.2,
      altitude: 1500,
      thickness: 1000,
      color: color(0xffffff),
      shadowColor: color(0x86a6c8),
      baseRoughness: 0.08,
      cirrus: 0.5,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 9,
      windDirection: Math.PI * 0.45,
      peakWavelength: 34,
      gamma: 3,
      swell: 0.2,
      standingWaveRatio: 0.1,
      spectralSharpness: 1.3,
      windwardFoam: 0.06,
    },
    water: {
      // Deep water goes properly blue within a few tens of metres, so the open
      // sea reads as ocean and only the island's shallows turn turquoise. At
      // the old (0.22, 0.06, 0.035) thirty metres of water still showed the
      // sand through it, and the whole sea read as a lagoon.
      deepColor: color(0x022c48),
      shallowColor: color(0x27d0bf),
      scatterColor: color(0x3fe6b4),
      extinction: vec(0.46, 0.12, 0.07),
      roughness: 0.055,
      foamThreshold: 0.42,
      crestGlow: 0.85,
    },
    fog: { color: color(0xcfeaf4), density: 0.00012, volumetric: 9e-05 },
    underwater: {
      color: color(0x1fa0ab),
      distortion: 0.011,
      extinction: vec(0.18, 0.05, 0.03),
      visibility: 70,
      godRayStrength: 1.35,
    },
    toneMappingExposure: 1.05,
    grade: {
      // The stylised one, and the only place saturation goes meaningfully above
      // 1. Warm slope with a cool-ish blue kept intact, so the turquoise stays
      // turquoise rather than going green as it saturates.
      slope: new THREE.Color(1.05, 1.02, 0.99),
      offset: new THREE.Color(0.0, 0.0, 0.0),
      power: new THREE.Color(1.03, 1.0, 0.99),
      saturation: 1.18,
      vignette: 0.2,
    },
  },

  storm: {
    id: 'storm',
    label: 'Storm',
    atmosphere: {
      sunElevation: 0.22,
      sunAzimuth: 3.6,
      // Turbidity down from 12, rayleigh up from 0.7, Mie down from 0.024. Those
      // values were being used to fake an overcast and produced a sandy brown
      // horizon instead — Preetham reads high turbidity as *dust*, not cloud, and
      // the ocean then reflected the dust. `overcast` is the parameter that
      // actually describes this sky.
      turbidity: 5.5,
      rayleigh: 1.6,
      mieCoefficient: 0.009,
      mieDirectionalG: 0.72,
      overcast: 0.88,
      exposure: 0.85,
      nightIntensity: 0.1,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.95,
      density: 1.6,
      altitude: 800,
      thickness: 1400,
      color: color(0x7d8791),
      shadowColor: color(0x1e2329),
      baseRoughness: 1.0,
      cirrus: 0.0,
    },
    weather: { kind: 'rain', intensity: 0.9 },
    sea: {
      windSpeed: 21,
      windDirection: Math.PI * 1.1,
      peakWavelength: 60,
      gamma: 3.3,
      swell: 0.15,
      standingWaveRatio: 0,
      spectralSharpness: 0.75,
      windwardFoam: 0.07,
    },
    water: {
      deepColor: color(0x071820),
      shallowColor: color(0x1d5a5a),
      scatterColor: color(0x2f7f72),
      extinction: vec(0.4, 0.18, 0.12),
      roughness: 0.12,
      // Raised with the foam relighting: lit as bubbles rather than at 40 % of
      // albedo, a 21 m/s sea at 0.55 read as a snowfield.
      foamThreshold: 0.66,
    },
    fog: { color: color(0x4a555f), density: 0.00042, volumetric: 0.00055 },
    underwater: {
      color: color(0x16323f),
      distortion: 0.02,
      extinction: vec(0.44, 0.2, 0.14),
      visibility: 14,
      godRayStrength: 0.25,
    },
    toneMappingExposure: 0.85,
    grade: {
      // Low contrast, low chroma, lifted blacks. Weather is not dramatic to
      // look at, it is *flat*: the deck kills the key light, everything is lit
      // by an overcast hemisphere, and the air between the viewer and anything
      // worth seeing is full of water. The heaviest vignette of the nine.
      // Re-cut darker and harder for the relit foam and the lightning: a storm
      // read as a pale wash, and a flash has nothing to land on in a frame that
      // is already white. Slate water, white crests, a heavy deck.
      slope: new THREE.Color(0.92, 0.97, 1.0),
      offset: new THREE.Color(0.0, 0.002, 0.004),
      power: new THREE.Color(1.1, 1.08, 1.06),
      saturation: 0.86,
      vignette: 0.32,
    },
  },

  sunset: {
    id: 'sunset',
    label: 'Sunset',
    atmosphere: {
      sunElevation: 0.075,
      sunAzimuth: 4.6,
      turbidity: 6,
      rayleigh: 2.9,
      mieCoefficient: 0.012,
      mieDirectionalG: 0.88,
      overcast: 0,
      exposure: 1,
      nightIntensity: 0.05,
      moonElevation: -0.5,
      moonAzimuth: 0,
    },
    clouds: {
      coverage: 0.45,
      density: 1,
      altitude: 2200,
      thickness: 900,
      color: color(0xffcfa8),
      shadowColor: color(0x6b5f78),
      baseRoughness: 0.1,
      cirrus: 0.75,
    },
    weather: { kind: 'clear', intensity: 0 },
    sea: {
      windSpeed: 2.5,
      windDirection: Math.PI * 1.35,
      peakWavelength: 20,
      gamma: 2.2,
      swell: 0.5,
      standingWaveRatio: 0.45,
      spectralSharpness: 1.1,
      windwardFoam: 0.01,
    },
    water: {
      deepColor: color(0x0a2230),
      shallowColor: color(0x1c6f7c),
      scatterColor: color(0xb5825a),
      extinction: vec(0.3, 0.12, 0.08),
      roughness: 0.04,
      foamThreshold: 0.52,
    },
    fog: { color: color(0xd0aa9c), density: 0.00022, volumetric: 0.00018 },
    underwater: {
      color: color(0x14455c),
      distortion: 0.006,
      extinction: vec(0.32, 0.12, 0.08),
      visibility: 30,
      godRayStrength: 0.7,
    },
    toneMappingExposure: 1,
    grade: {
      // Warm, and — the part that matters — *contrastier*. Shooting into a low
      // sun puts a large amount of scattered light into every part of the
      // frame, and with a bloom on top the first cut of this rendered as a pale
      // wash with the rigging lost in it. Power above 1 in linear steepens the
      // midtones and deepens the shadows without touching the highlights that
      // ACES is about to roll off anyway; the negative offset takes the haze
      // off the floor. The measurement is the sails: if they are not legible
      // against the sun, this is too weak.
      slope: new THREE.Color(1.06, 0.99, 0.92),
      offset: new THREE.Color(-0.012, -0.010, -0.006),
      power: new THREE.Color(1.16, 1.13, 1.08),
      saturation: 1.06,
      vignette: 0.26,
    },
  },
};

export const PRESET_LIST: Preset[] = Object.values(PRESETS);

export function getPreset(id: PresetId): Preset {
  return PRESETS[id];
}
