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

The instanced leaf renderer for kelp and seabed cover (leaf strips, sway, palette,
two-sided translucent lighting, glints) is adapted from abyssal-garden's `W_LEAF`
shader and its world layout:

- `src/scene/LeafField.ts`
- `src/scene/Kelp.ts`
- `src/scene/GroundCover.ts`

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
from its cloud shader, to three.js TSL:

- `src/sky/PhysicalAtmosphere.ts`
- `src/sky/Atmosphere.ts`
- `src/sky/Clouds.ts`

The water surface shading (LEAN slope statistics, Gaussian slope-space sun
glitter with Smith shadowing, Fresnel on a flattened normal, the sky
reflection's grazing tilt and fold), the capillary detail band (dispersion,
cut-off and band gains from `js/wgsl/sim.js`), and the foam model (the
`FOAM_TEX` pattern, the cover and soft-threshold composite, wind streaks and
the foam-freshness channel) are ported from `js/wgsl/ocean.js` and
`js/wgsl/sim.js`:

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
