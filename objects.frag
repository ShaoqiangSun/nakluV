#version 450

layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION;
	vec3 SKY_ENERGY; //energy supplied by sky to a surface patch with normal = SKY_DIRECTION

    vec3 SUN_DIRECTION;
	vec3 SUN_ENERGY; //energy supplied by sun to a surface patch with normal = SUN_DIRECTION

    vec3 EYE;
};
layout(set=0,binding=1) uniform samplerCube ENV;
layout(set=0,binding=2) uniform samplerCube ENV_LAMBERTIAN;

//layout(set=2,binding=0) uniform sampler2D TEXTURE;
layout(set=2,binding=0,std140) uniform MaterialParams {
    vec4 ALBEDO;
    uint TYPE;
    uint padding1_;
    uint padding2_;
    uint padding3_;
};
layout(set=2,binding=1) uniform sampler2D ALBEDO_TEXTURE;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 tangent;
layout(location=3) in vec2 texCoord;

layout(location=0) out vec4 outColor;

void main() {
    vec3 n = normalize(normal);

    if (TYPE == 0u) {
        vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb * ALBEDO.rgb;
        //vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb;

        //hemisphere sky + directional sun:
        //vec3 e = SKY_ENERGY * (0.5 * dot(n,SKY_DIRECTION) + 0.5)
            + SUN_ENERGY * max(0.0, dot(n,SUN_DIRECTION)) ;

        //outColor = vec4(e * albedo / 3.1415926, 1.0);

        //environment cubemap
        vec3 e = texture(ENV_LAMBERTIAN, n).rgb;
        outColor = vec4(e * albedo, 1.0);
    }

    else if (TYPE == 1u) {
        // Environment material
        outColor = vec4(texture(ENV, n).rgb, 1.0);
    }
    else if (TYPE == 2u) {
        // Mirror material
        vec3 v = normalize(EYE - position);
        vec3 r = reflect(-v, n);
        outColor = vec4(texture(ENV, r).rgb, 1.0);
    }
    else if (TYPE == 3u){
        vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb * ALBEDO.rgb;

        //hemisphere sky + directional sun:
        vec3 e = SKY_ENERGY * (0.5 * dot(n,SKY_DIRECTION) + 0.5)
            + SUN_ENERGY * max(0.0, dot(n,SUN_DIRECTION)) ;

        outColor = vec4(e * albedo / 3.1415926, 1.0);
    }
    
}