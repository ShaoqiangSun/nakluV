#version 450

// Caustic photon-splatting fragment shader.
//
// Each rasterized fragment deposits a small irradiance contribution into
// the caustic accumulation map; additive blending in the pipeline sums
// contributions from overlapping triangles, producing bright spots where
// reflected rays focus on the receiver plane.

layout(location = 0) in float vContrib;
layout(location = 0) out vec4 outCaustic;

void main() {
    outCaustic = vec4(vContrib, 0.0, 0.0, 1.0);
}
