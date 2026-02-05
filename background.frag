#version 450

layout(push_constant) uniform Push {
    float time;
};

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 outColor;

void main() {
    //outColor = vec4( fract(gl_FragCoord.x / 100), gl_FragCoord.y / 400, 0.2, 1.0 );
    //outColor = vec4( fract((position.x + position.y) * 500 / 100), position.y * 500 / 400, 0.2, 1.0 );
    //outColor = vec4(position, 0.0, 1.0);

    //outColor = vec4(fract(position.x + time), position.y, 0.0, 1.0);    

    vec2 p = position * 2.0 - 1.0;
    float t = time;

    float r = length(p);

    float wave = sin(15.0 * r - 3.0 * t);

    float damp = exp(-1.7 * r);

    float h = 0.5 + 0.5 * wave * damp;

    vec3 deep = vec3(0.04, 0.05, 0.15);
    vec3 shallow = vec3(0.08, 0.66, 0.90);

    outColor = vec4(mix(deep, shallow, h), 1.0);
}