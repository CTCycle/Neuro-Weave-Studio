# Convolutional Layers

Last updated: 2026-09-01

The catalog includes Conv1D, Conv2D, Conv3D, transposed, depthwise, and separable convolution, max and average pooling, global pooling, upsampling, padding, and cropping. Keras mappings retain the relevant spatial, activation, kernel, regularization, and base-layer parameters. Shape inference enforces convolution rank, channel replacement, and known spatial transitions where the rule is defined; unresolved padding, dilation, stride, or backend-specific behavior remains dynamic/unknown and is never fabricated. Native mappings are limited to the verified target subset.
