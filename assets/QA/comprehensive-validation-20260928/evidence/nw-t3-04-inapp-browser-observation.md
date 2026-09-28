# In-app Browser observation: NW-T3-04

Date: 2026-09-28
URL observed: `http://127.0.0.1:6431/builder`
Profile controls: PyTorch execution backend, Keras enabled, CPU lane available

The live Builder rendered the current catalog shell with `149 available` nodes
and an expanded `CONVOLUTIONAL 18 nodes` section. The rendered list visibly
included the nine NW-T3-04 definitions:

- Average Pooling 2D (`conv.average_pool2d`)
- Cropping (`conv.cropping`)
- Depthwise Convolution (`conv.depthwise`)
- Max Pooling 1D (`conv.max_pool1d`)
- Max Pooling 3D (`conv.max_pool3d`)
- Padding (`conv.padding`)
- Separable Convolution (`conv.separable`)
- Transposed Convolution (`conv.transposed`)
- Upsampling (`conv.upsampling`)

The page loaded without an application error and showed the empty Builder
canvas, catalog search, backend controls, validation/save controls, and the
Properties inspector. The serial E2E run then materialized each selected node
from this rendered catalog and captured the Builder/runtime screenshots under
this evidence directory.
