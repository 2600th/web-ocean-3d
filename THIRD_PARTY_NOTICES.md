# Third-party notices

Source code adapted from the open-source projects below is used under their
licences. Art assets (models, textures, fonts, audio) are listed separately in
`ASSET_LICENSES.md`. Each port is also credited in a comment at the point of use.

## abyssal-garden

https://github.com/Noniv/abyssal-garden

The underwater medium (analytic height-dependent in-scatter), the volumetric
light shafts (ray march, depth-aware blur and upsample), the chromatic caustics
and the submerged hemisphere ambient are adapted from abyssal-garden's WGSL
shaders and ported to three.js TSL. Affected files:

- `src/underwater/UnderwaterPass.ts`
- `src/underwater/Caustics.ts`

The underwater look also adapts these parts of abyssal-garden:

- The fluorescence term, the water-scattered side light and the grade's
  luminance S-curve, from `W_POST` and its lighting helpers, in
  `src/underwater/UnderwaterPass.ts`.
- The caustics arrangement, in `src/underwater/Caustics.ts`: one periodic tile
  read at metre scale for surfaces (`caus`), with a second read at another
  scale and orientation, and read again at shaft scale with abyssal's
  thresholds (`W_RAYS`). The pattern inside the tile is our own.
- The water surface seen from below (`W_SURF`): Snell's window, the Fresnel
  term at the window edge, the ripple brightness, the refracted sun, and total
  internal reflection as the medium's own in-scatter. These are in the fenced
  `[underwater-look]` underside branch of `src/ocean/OceanMaterial.ts`.
- The shaft pattern's blurred, thresholded pair of reads (`W_RAYS`), now at
  the heavier blur that turns the web into beams, in
  `src/underwater/Caustics.ts`.
- The submerged tone curve: the per-channel ACES fit abyssal-garden's
  `W_POST` uses (Narkowicz 2015), in the fenced `[underwater-look-2]` branch
  of `src/post/OutputTransform.ts`.

The instanced leaf renderer for kelp and seabed cover (leaf strips, sway, palette,
two-sided translucent lighting, glints) is adapted from abyssal-garden's `W_LEAF`
shader and its world layout:

- `src/scene/LeafField.ts`
- `src/scene/Kelp.ts`
- `src/scene/GroundCover.ts`

The kelp forest's density follows abyssal-garden's `world.js` layout: holdfasts
on a minimum-spacing occupancy grid where a thresholded fbm field says the
forest is solid (`tryColumn`, `kelpDens`), outlying stands past the rim, and
viewpoints kept clear of columns. Its canopy shade is a per-leaf stand-in for
abyssal's skylight bake (each column's absorption propagated down through the
cells below it), in `src/scene/Kelp.ts`.

The seabed geology and its reef dressing are adapted from abyssal-garden's
world builder: icosphere rock chunks, naive surface nets and layered-rock
shading (`src/scene/reef/RockKit.ts`), the outcrop density profile of segment
columns with overhang, waist, base flare, strata and relief
(`src/scene/Rocks.ts`), and sponges and corals clustered on rock surfaces
(`src/scene/ReefLife.ts`):

- `src/scene/reef/RockKit.ts`
- `src/scene/Rocks.ts`
- `src/scene/ReefLife.ts`

The prop kit's swept profiles and tubes are ported from abyssal-garden's
`math.js` (`Mesh.sweep`, `Mesh.tube`) as `sweep` and `tube` in
`src/scene/props/kit.ts`, used for the chest straps, the lighthouse railing,
the lantern bail, the cutlass blade and guard, the rope coils and the anchor
ring:

- `src/scene/props/kit.ts`

```
MIT License

Copyright (c) 2026 Maksymilian Dendura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## web-ocean

https://github.com/Noniv/web-ocean

The physical sky (transmittance and sky-view atmosphere model, limb-darkened sun
disc) is ported from `js/wgsl/sky.js` and `common.js`, and the high cloud layer
from its cloud shader, to three.js TSL. The volumetric cumulus take three more
pieces of that cloud shader: the rotated fBm with a 2.03 lacunarity and
per-octave offset (`cloudFbm`, used for the 3D shape field and the 2D weather
map), and the clear far band where the low layer thins out before the horizon:

- `src/sky/PhysicalAtmosphere.ts`
- `src/sky/Atmosphere.ts`
- `src/sky/Clouds.ts`

The water surface shading (LEAN slope statistics, Gaussian slope-space sun
glitter with Smith shadowing, Fresnel on a flattened normal, the sky
reflection's grazing tilt and fold), the capillary detail band (dispersion,
cut-off and band gains from `js/wgsl/sim.js`), and the foam model (the
`FOAM_TEX` pattern, the cover and soft-threshold composite, wind streaks, the
foam-freshness channel, and the foam buffer's injection: breaking on the
surface's total Jacobian, `max(previous, inject)` with growth and a 1.6 cap,
and the 0.22 / s decay, from `FOAM_UPDATE`; and the swiss-cheese `bubbles` of
`FOAM_TEX`; and the gradient noise, `gnoise` / `pfbm` in `js/wgsl/common.js`,
that the filament lace every open-sea foam and wake is drawn through is built
on) are ported from `js/wgsl/ocean.js` and `js/wgsl/sim.js`:

- `src/ocean/OceanMaterial.ts`
- `src/ocean/FoamPattern.ts`
- `src/ocean/Spectrum.ts`
- `src/ocean/OceanSimulation.ts`
- `src/physics/Wake.ts`

```
MIT License

Copyright (c) 2026 Maksymilian Dendura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Crest

https://github.com/wave-harmonic/crest

The aerated layer under the foam reads a blurred bubble field through a
parallax offset of the view over the surface normal, after Crest's
`BubbleFoamTexture` (`Crest/Shaders/OceanFoam.hlsl`); the white foam's
threshold of a foam pattern by the foam amount ("black point fade",
`WhiteFoamTexture`) is the same construction:

- `src/ocean/OceanMaterial.ts`
- `src/ocean/FoamPattern.ts`

```
MIT License

Copyright (c) 2019 Wave Harmonic and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hash without Sine

https://www.shadertoy.com/view/4djSRW

The `hash22` lattice hash that seeds the water's stochastic sun glints is
Dave Hoskins' "Hash without Sine", ported to three.js TSL:

- `src/ocean/OceanMaterial.ts`

```
MIT License

Copyright (c) 2014 David Hoskins

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mrdoob/slop-newyork

https://github.com/mrdoob/slop-newyork

Ported from its TSL materials: the surface-gradient `bumpNormal` (Mikkelsen),
`hash2D` and `gridLine`, and the general approach of procedural TSL surfaces
with no texture downloads.

- `src/scene/props/materials.ts`: `bumpNormal`, `hash21` (after `hash2D`),
  `gridLine`.
- `src/scene/islandDetail/rock.ts`: `bumpNormal`, with the geometric normal in
  place of the shading one.

```
MIT License

Copyright (c) 2026 Mr.doob

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## three.quarks

The velocity-aligned ("stretched") billboard used by the spray particles takes
its axis construction from three.quarks' `stretched_bb_particle_vert.glsl.ts`:
the long axis along the particle's velocity in the view plane, the short axis
the cross product of that with the view ray. It is ported to TSL and evaluated
in world space; the length rule is this project's own. Affected files:

- `src/physics/sprayLook.ts`

```
MIT License

Copyright (c) 2019 Forrest Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NeuralSpecOcean

https://github.com/bgbg108555/NeuralSpecOcean

The shallow-water optics of the sea surface near a shore are adapted from
NeuralSpecOcean's beach optics (`source/src/Offshore.js`): the view ray refracted
at the surface and followed to the bed, the bed point projected to choose the
backdrop pixel (offset clamped, faded at the frame edge, stepped back when the
sample lands in front of the water), the optical path taken along the refracted
ray (`depth / cos(refracted)`, floored at 0.66), and the bed attenuated on the
way down by a diffuse coefficient derived from the same extinction. The wet-sand
model is adapted from its `source/src/ShoreVisual.js`: the dry-to-film roughness
range, the water film's normal flattened toward the beach plane, and the
wet/dry edge perturbed by multi-scale seepage noise. The surf's second, wind-sea
wave train follows the secondary train in its `source/src/CoastalWaves.js`, at
about two thirds of the swell's period, and the surf foam's long-tail wake
behind each bore follows that file's roller envelope. The surf white water's
relief shading follows its `source/src/FoamShader.js`: a foam height field
whose gradient tilts the foam normal, one sample toward the sun for the lumps'
self-shadow, a multiple-scattering transmission term, and three albedos for
shadowed folds, lace and sunlit crests. Ported to three.js TSL;
the surf's closed form, the foam and the wetting state are this project's own.
Affected files:

- `src/ocean/OceanMaterial.ts`
- `src/ocean/Surf.ts`
- `src/scene/Seafloor.ts`

```
MIT License

Copyright (c) 2026 bgbg108555 and NeuralSpecOcean Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## three.js examples: webgpu_compute_particles

https://github.com/mrdoob/three.js/blob/r187/examples/webgpu_compute_particles.html

The spindrift's structure follows this example: particle state in `storage`
buffers, stepped each frame by a `Fn().compute(count)` pass and read in the
vertex stage of an instanced draw through `toAttribute()`. The emission,
physics and look are this project's own. Affected files:

- `src/physics/Spindrift.ts`

## three.js examples: webgpu_compute_cloth

https://github.com/mrdoob/three.js/blob/r187/examples/webgpu_compute_cloth.html

The sail and pennant cloth is adapted from three.js r187's
`webgpu_compute_cloth` example: verlet springs in two compute passes (spring
forces, then a per-vertex gather over a spring list), the buffer layout and the
integration. Several grids share the buffers, the pins follow a CPU buffer, and
the wind is an apparent-wind pressure with flutter; see the file header.
Affected files:

- `src/scene/ship/ClothSolver.ts`
- `src/scene/ship/Sails.ts`
- `src/scene/ship/Pennant.ts`

```
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
