#version 450

// Write the fragment's clip-space depth to the R32_SFLOAT shadow map
layout(location=0) out float out_depth;

void main() {
    // Slope-scaled bias: surfaces angled relative to the light get a larger offset,
    // preventing shadow acne on steep faces
    float slope_bias = 2.0 * max(abs(dFdx(gl_FragCoord.z)), abs(dFdy(gl_FragCoord.z)));
    out_depth = gl_FragCoord.z + max(slope_bias, 0.0002);
}
