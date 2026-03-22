#version 450

const uint LIGHT_SUN    = 0u;
const uint LIGHT_SPHERE = 1u;
const uint LIGHT_SPOT   = 2u;

const float PI = 3.14159265359;

struct Light {
    vec4 POSITION_TYPE;
    vec4 DIRECTION_SHADOW;
    vec4 TINT_STRENGTH;
    vec4 PARAMS;
};

layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION;
	vec3 SKY_ENERGY; //energy supplied by sky to a surface patch with normal = SKY_DIRECTION

    vec3 SUN_DIRECTION;
	vec3 SUN_ENERGY; //energy supplied by sun to a surface patch with normal = SUN_DIRECTION

    vec3 EYE;

    vec3 TONE;

    uvec3 MIP_AND_LIGHT;
};
layout(set=0,binding=1) uniform samplerCube ENV;
layout(set=0,binding=2) uniform samplerCube ENV_LAMBERTIAN;
layout(set=0,binding=3) uniform sampler2D BRDF_LUT;
layout(set=0,binding=4,std430) readonly buffer LightsBlock {
    Light LIGHTS[];
};

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


float spot_falloff(vec3 Ldir, vec3 spot_dir, float fov, float blend) {
    float cosTheta = dot(normalize(-Ldir), normalize(spot_dir));
    float inner = cos(fov * 0.5 * (1.0 - blend));
    float outer = cos(fov * 0.5);

    if (cosTheta <= outer) return 0.0;
    if (cosTheta >= inner) return 1.0;

    return smoothstep(outer, inner, cosTheta);
}

float limit_falloff(float d, float limit) {
    if (limit <= 0.0) return 1.0;

    float x = max(0.0, 1.0 - pow(d / limit, 4.0));
    return x;
}

vec3 evaluate_direct_diffuse(vec3 P, vec3 N) {
    vec3 result = vec3(0.0);

    uint light_count = MIP_AND_LIGHT.y;

    for (uint i = 0u; i < light_count; ++i) {
        Light light = LIGHTS[i];

        uint type = uint(light.POSITION_TYPE.w);
        vec3 light_tint = light.TINT_STRENGTH.rgb;
        float intensity = light.TINT_STRENGTH.w;

        vec3 L = vec3(0.0);
        vec3 radiance = vec3(0.0);

        if (type == LIGHT_SUN) {
            vec3 sun_dir = normalize(light.DIRECTION_SHADOW.xyz);

            // light emits along -z, so shading uses opposite direction
            L = normalize(-sun_dir);

            // sun strength is W/m^2, no distance falloff
            radiance = light_tint * intensity;
        }
        else if (type == LIGHT_SPHERE) {
            vec3 light_pos = light.POSITION_TYPE.xyz;
            float limit = light.PARAMS.y;

            vec3 to_light = light_pos - P;
            float dist = length(to_light);
            if (dist <= 1e-5) continue;

            L = to_light / dist;

            float attenuation = 1.0 / max(4.0 * PI * dist * dist, 1e-4);
            attenuation *= limit_falloff(dist, limit);

            // sphere power is total emitted power in watts
            radiance = light_tint * intensity * attenuation;
        }
        else if (type == LIGHT_SPOT) {
            vec3 light_pos = light.POSITION_TYPE.xyz;
            vec3 spot_dir = normalize(light.DIRECTION_SHADOW.xyz);
            float limit = light.PARAMS.y;
            float fov   = light.PARAMS.z;
            float blend = light.PARAMS.w;

            vec3 to_light = light_pos - P;
            float dist = length(to_light);
            if (dist <= 1e-5) continue;

            L = to_light / dist;

            float attenuation = 1.0 / max(4.0 * PI * dist * dist, 1e-4);
            attenuation *= limit_falloff(dist, limit);

            float cone = spot_falloff(L, spot_dir, fov, blend);

            // spot power is defined relative to sphere power
            radiance = light_tint * intensity * attenuation * cone;
        }

        float NoL = max(dot(N, L), 0.0);
        if (NoL <= 0.0) continue;

        // Lambert BRDF
        result += radiance * NoL * (1.0 / PI);
    }

    return result;
}

vec3 representative_point_on_sphere(vec3 P, vec3 R, vec3 C, float radius) {
    float t = max(dot(C - P, R), 0.0);
    vec3 closest_on_ray = P + t * R;

    vec3 v = closest_on_ray - C;
    float len2 = dot(v, v);

    if (len2 < 1e-8) {
        return C - R * radius;
    }

    return C + normalize(v) * radius;
}

vec3 representative_direction_in_sun(vec3 Lsun, vec3 R, float angle) {
    float half_angle = 0.5 * angle;

    float cosTheta = clamp(dot(Lsun, R), -1.0, 1.0);
    float theta = acos(cosTheta);

    if (theta <= half_angle) {
        return normalize(R);
    }

    vec3 axis = R - Lsun * dot(Lsun, R);
    float axis_len = length(axis);

    if (axis_len < 1e-6) {
        return Lsun;
    }

    axis /= axis_len;

    return normalize(cos(half_angle) * Lsun + sin(half_angle) * axis);
}

float D_GGX(float NoH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G1_Smith(float NoX, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NoX + sqrt(a2 + (1.0 - a2) * NoX * NoX);
    return 2.0 * NoX / max(denom, 1e-5);
}

float G_Smith(float NoV, float NoL, float roughness) {
    return G1_Smith(NoV, roughness) * G1_Smith(NoL, roughness);
}

vec3 F_Schlick(vec3 F0, float VoH) {
    return F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
}

vec3 ggx_specular(vec3 N, vec3 V, vec3 L, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);

    float NoV = max(dot(N, V), 0.0);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);

    if (NoV <= 0.0 || NoL <= 0.0) return vec3(0.0);

    float D = D_GGX(NoH, roughness);
    float G = G_Smith(NoV, NoL, roughness);
    vec3  F = F_Schlick(F0, VoH);

    return (D * G * F) / max(4.0 * NoV * NoL, 1e-4);
}

vec3 evaluate_direct_specular(vec3 P, vec3 N, vec3 V, float roughness, vec3 F0) {
    vec3 result = vec3(0.0);
    vec3 R = reflect(-V, N);

    uint light_count = MIP_AND_LIGHT.y;

    for (uint i = 0u; i < light_count; ++i) {
        Light light = LIGHTS[i];
        uint type = uint(light.POSITION_TYPE.w);
        vec3 tint = light.TINT_STRENGTH.rgb;
        float intensity = light.TINT_STRENGTH.w;

        if (type == LIGHT_SUN) {
            vec3 Lsun = normalize(-light.DIRECTION_SHADOW.xyz);
            float angle = light.PARAMS.x;

            vec3 Lrep = representative_direction_in_sun(Lsun, R, angle);
            vec3 spec = ggx_specular(N, V, Lrep, roughness, F0);
            result += tint * intensity * spec * max(dot(N, Lrep), 0.0);
        }
        else if (type == LIGHT_SPHERE) {
            vec3 C = light.POSITION_TYPE.xyz;
            float radius = light.PARAMS.x;
            float limit  = light.PARAMS.y;

            vec3 to_light = C - P;
            float dist = length(to_light);
            if (dist <= 1e-5) continue;

            float attenuation = 1.0 / max(4.0 * PI * dist * dist, 1e-4);
            attenuation *= limit_falloff(dist, limit);

            vec3 rep_point = representative_point_on_sphere(P, R, C, radius);
            vec3 Lrep = normalize(rep_point - P);

            vec3 spec = ggx_specular(N, V, Lrep, roughness, F0);
            result += tint * intensity * attenuation * spec * max(dot(N, Lrep), 0.0);
        }
        else if (type == LIGHT_SPOT) {
            vec3 C = light.POSITION_TYPE.xyz;
            vec3 spot_dir = normalize(light.DIRECTION_SHADOW.xyz);
            float radius = light.PARAMS.x;
            float limit  = light.PARAMS.y;
            float fov    = light.PARAMS.z;
            float blend  = light.PARAMS.w;

            vec3 to_light = C - P;
            float dist = length(to_light);
            if (dist <= 1e-5) continue;

            vec3 Lcenter = to_light / dist;
            float cone = spot_falloff(Lcenter, spot_dir, fov, blend);
            if (cone <= 0.0) continue;

            float attenuation = 1.0 / max(4.0 * PI * dist * dist, 1e-4);
            attenuation *= limit_falloff(dist, limit);

            vec3 rep_point = representative_point_on_sphere(P, R, C, radius);
            vec3 Lrep = normalize(rep_point - P);

            vec3 spec = ggx_specular(N, V, Lrep, roughness, F0);
            result += tint * intensity * attenuation * cone * spec * max(dot(N, Lrep), 0.0);
        }
    }

    return result;
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
        // Lambertian Material
        vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb * ALBEDO.rgb;
        //vec3 albedo = texture(ALBEDO_TEXTURE, texCoord).rgb;

        //hemisphere sky + directional sun:
        //vec3 e = SKY_ENERGY * (0.5 * dot(n,SKY_DIRECTION) + 0.5)
            + SUN_ENERGY * max(0.0, dot(n,SUN_DIRECTION)) ;

        //outColor = vec4(e * albedo / 3.1415926, 1.0);

        //environment cubemap
        vec3 env_e = texture(ENV_LAMBERTIAN, n).rgb;

        vec3 direct_e = evaluate_direct_diffuse(position, n);

        vec3 radiance = (env_e + direct_e) * albedo;
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
        float mip = roughness * float(MIP_AND_LIGHT.x);
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

        vec3 direct_diffuse = evaluate_direct_diffuse(position, n);
        vec3 direct_specular = evaluate_direct_specular(position, n, V, roughness, F0);

        //vec3 radiance = kD * diffuse_IBL + specularIBL;

        vec3 radiance =
            kD * diffuse_IBL +
            specularIBL +
            kD * direct_diffuse * albedo +
            direct_specular;
        
        float exposure = TONE.x;
        uint op = uint(TONE.y + 0.5);
        vec3 mapped = apply_tonemapping(radiance, exposure, op);
        outColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);

        //outColor = vec4(e * albedo / 3.1415926, 1.0);
    }
    
}