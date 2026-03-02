#version 450

layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION;
	vec3 SKY_ENERGY; //energy supplied by sky to a surface patch with normal = SKY_DIRECTION

    vec3 SUN_DIRECTION;
	vec3 SUN_ENERGY; //energy supplied by sun to a surface patch with normal = SUN_DIRECTION

    vec3 EYE;

    vec3 TONE;

    uvec3 MAX_MIP;
};
layout(set=0,binding=1) uniform samplerCube ENV;
layout(set=0,binding=2) uniform samplerCube ENV_LAMBERTIAN;
layout(set=0,binding=3) uniform sampler2D BRDF_LUT;

//layout(set=2,binding=0) uniform sampler2D TEXTURE;
layout(set=2,binding=0,std140) uniform MaterialParams {
    vec4 ALBEDO;
    vec4 PBR;
    uint TYPE;
    uint padding1_;
    uint padding2_;
    uint padding3_;
};
layout(set=2,binding=1) uniform sampler2D ALBEDO_TEXTURE;
layout(set=2,binding=2) uniform sampler2D ROUGHNESS_TEXTURE;
layout(set=2,binding=3) uniform sampler2D METALLIC_TEXTURE;
layout(set=2,binding=4) uniform sampler2D NORMAL_TEXTURE;
layout(set=2,binding=5) uniform sampler2D DISPLACEMENT_TEXTURE;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 tangent;
layout(location=3) in vec2 texCoord;

layout(location=0) out vec4 outColor;

vec3 tonemapping_linear(vec3 x) {
    return x;
}

vec3 tonemapping_reinhard(vec3 x) {
    return x / (vec3(1.0) + x);
}

vec3 apply_tonemapping(vec3 radiance, float exposure, uint op) {
    vec3 x = radiance * exp2(exposure);

    if (op == 0u) return tonemapping_linear(x);
    if (op == 1u) return tonemapping_reinhard(x);

    // fallback
    return tonemapping_linear(x);
}

void main() {
    vec3 N = normalize(normal);
    vec3 T = normalize(tangent.xyz);
    T = normalize(T - N * dot(N, T));
    vec3 B = cross(N, T) * tangent.w;
    mat3 TBN = mat3(T, B, N);

    vec3 n_ts = texture(NORMAL_TEXTURE, texCoord).xyz;
    n_ts = n_ts * 2.0 - 1.0;

    vec3 n = normalize(TBN * normalize(n_ts));

    //vec3 n = normalize(normal);

    if (TYPE == 0u) {
        vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb * ALBEDO.rgb;
        //vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb;

        //hemisphere sky + directional sun:
        //vec3 e = SKY_ENERGY * (0.5 * dot(n,SKY_DIRECTION) + 0.5)
            + SUN_ENERGY * max(0.0, dot(n,SUN_DIRECTION)) ;

        //outColor = vec4(e * albedo / 3.1415926, 1.0);

        //environment cubemap
        vec3 e = texture(ENV_LAMBERTIAN, n).rgb;

        vec3 radiance = e * albedo;
        float exposure = TONE.x;
        uint op = uint(TONE.y + 0.5);
        vec3 mapped = apply_tonemapping(radiance, exposure, op);
        outColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);

        //outColor = vec4(e * albedo, 1.0);
        
    }

    else if (TYPE == 1u) {
        // Environment material
        //vec3 radiance = texture(ENV, n).rgb;
        vec3 radiance = textureLod(ENV, n, 0.0).rgb;
        float exposure = TONE.x;
        uint op = uint(TONE.y + 0.5);
        vec3 mapped = apply_tonemapping(radiance, exposure, op);
        outColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);

        //outColor = vec4(texture(ENV, n).rgb, 1.0);
    }
    else if (TYPE == 2u) {
        // Mirror material
        vec3 v = normalize(EYE - position);
        vec3 r = reflect(-v, n);

        //vec3 radiance = texture(ENV, r).rgb;
        vec3 radiance = textureLod(ENV, r, 0.0).rgb;
        float exposure = TONE.x;
        uint op = uint(TONE.y + 0.5);
        vec3 mapped = apply_tonemapping(radiance, exposure, op);
        outColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);

        //outColor = vec4(texture(ENV, r).rgb, 1.0);
    }
    else if (TYPE == 3u){
        //PBR Material
        vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb * ALBEDO.rgb;

        float roughness = texture(ROUGHNESS_TEXTURE, texCoord).r * PBR.x;
        float metallic  = texture(METALLIC_TEXTURE,  texCoord).r * PBR.y;

        vec3 V = normalize(EYE - position);
        float NdotV = clamp(dot(n, V), 0.0, 1.0);

        vec3 irradiance = texture(ENV_LAMBERTIAN, n).rgb;
        vec3 diffuse_IBL = irradiance * albedo;

        vec3 R = reflect(-V, n);
        float mip = roughness * float(MAX_MIP.x);
        vec3 prefiltered = textureLod(ENV, R, mip).rgb;
        vec2 brdf = texture(BRDF_LUT, vec2(NdotV, roughness)).rg;
        
        vec3 F0 = mix(vec3(0.04), albedo, metallic);
        vec3 specularIBL = prefiltered * (F0 * brdf.x + brdf.y);
        
        //hemisphere sky + directional sun:
        //vec3 e = SKY_ENERGY * (0.5 * dot(n,SKY_DIRECTION) + 0.5)
            + SUN_ENERGY * max(0.0, dot(n,SUN_DIRECTION)) ;

        //vec3 radiance = e * albedo;

        vec3 kS = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);   // Schlick Fresnel at NdotV
        vec3 kD = (1.0 - kS) * (1.0 - metallic);

        vec3 radiance = kD * diffuse_IBL + specularIBL;
        
        float exposure = TONE.x;
        uint op = uint(TONE.y + 0.5);
        vec3 mapped = apply_tonemapping(radiance, exposure, op);
        outColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);

        //outColor = vec4(e * albedo / 3.1415926, 1.0);
    }
    
}