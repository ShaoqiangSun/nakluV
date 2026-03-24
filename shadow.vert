#version 450

// Push constant carries the light's clip-from-world transform
layout(push_constant) uniform LightPC {
    mat4 CLIP_FROM_WORLD;
};

// All object instances' world-from-local matrices
layout(set=0, binding=0, std430) readonly buffer AllWorldTransforms {
    mat4 WORLD_FROM_LOCAL[];
};

// Must match PosNorTanTexVertex layout; only Position is used
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 Tangent;
layout(location=3) in vec2 TexCoord;

void main() {
    gl_Position = CLIP_FROM_WORLD * WORLD_FROM_LOCAL[gl_InstanceIndex] * vec4(Position, 1.0);
}
