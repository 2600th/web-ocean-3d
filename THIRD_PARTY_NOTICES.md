# Third-party notices

Source code ported into this project from other open-source projects. Art
assets (models, textures, fonts) are listed separately in `ASSET_LICENSES.md`.

Each port is also credited in a comment at the point of use.

## Noniv/abyssal-garden

https://github.com/Noniv/abyssal-garden

Ported: `Mesh.sweep` (hard-edged profile swept along a path) and `Mesh.tube`
(round tube along a polyline with a parallel-transported frame) from `math.js`,
as `sweep` and `tube` in `src/scene/props/kit.ts`. Used for the chest straps,
the lighthouse railing, the lantern bail, the cutlass blade and guard, the rope
coils and the anchor ring.

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
