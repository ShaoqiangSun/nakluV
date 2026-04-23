#version 450

#include "water_waves.glsl"

// Caustic photon-splatting: procedural water grid, reflect dominant light,
// intersect one axis-aligned room face per draw (faceIndex push constant),
// rasterize into that face's layer of the caustic map.

layout(push_constant) uniform Push {
    uint faceIndex;
} push;

layout(set = 0, binding = 0, std140) uniform WaterParams {
    mat4 WORLD_FROM_LOCAL;
    vec4 SIZE_RES_RECV;
    vec4 CAUSTIC_CEI;
    vec4 SUN_TIME;
    vec4 WAVE_COUNT;
    vec4 WAVES[WATER_MAX_WAVES];
    vec4 ROOM_MIN;
    vec4 ROOM_MAX;
    vec4 CAUSTIC_LIGHT_POINT;
    vec4 WATER_NM;
};

layout(set = 0, binding = 1) uniform sampler2D WATER_NORMAL_MAP;

layout(location = 0) out float vContrib;

bool intersect_room_face(vec3 P, vec3 Rd, uint face, out vec2 uv01) {
    vec3 mn = ROOM_MIN.xyz;
    vec3 mx = ROOM_MAX.xyz;
    const float eps = 1.0e-4;
    float t = -1.0;

    if (face == 0u) {
        if (abs(Rd.x) < eps) return false;
        t = (mx.x - P.x) / Rd.x;
    } else if (face == 1u) {
        if (abs(Rd.x) < eps) return false;
        t = (mn.x - P.x) / Rd.x;
    } else if (face == 2u) {
        if (abs(Rd.y) < eps) return false;
        t = (mx.y - P.y) / Rd.y;
    } else if (face == 3u) {
        if (abs(Rd.y) < eps) return false;
        t = (mn.y - P.y) / Rd.y;
    } else if (face == 4u) {
        if (abs(Rd.z) < eps) return false;
        t = (mx.z - P.z) / Rd.z;
    } else if (face == 5u) {
        if (abs(Rd.z) < eps) return false;
        t = (mn.z - P.z) / Rd.z;
    } else {
        return false;
    }

    if (t < 0.0) return false;

    vec3 H = P + t * Rd;

    if (face == 0u || face == 1u) {
        if (H.y < mn.y - 1e-3 || H.y > mx.y + 1e-3) return false;
        if (H.z < mn.z - 1e-3 || H.z > mx.z + 1e-3) return false;
        float ey = mx.y - mn.y;
        float ez = mx.z - mn.z;
        if (ey < 1e-6 || ez < 1e-6) return false;
        uv01 = vec2((H.y - mn.y) / ey, (H.z - mn.z) / ez);
    } else if (face == 2u || face == 3u) {
        if (H.x < mn.x - 1e-3 || H.x > mx.x + 1e-3) return false;
        if (H.z < mn.z - 1e-3 || H.z > mx.z + 1e-3) return false;
        float ex = mx.x - mn.x;
        float ez = mx.z - mn.z;
        if (ex < 1e-6 || ez < 1e-6) return false;
        uv01 = vec2((H.x - mn.x) / ex, (H.z - mn.z) / ez);
    } else {
        if (H.x < mn.x - 1e-3 || H.x > mx.x + 1e-3) return false;
        if (H.y < mn.y - 1e-3 || H.y > mx.y + 1e-3) return false;
        float ex = mx.x - mn.x;
        float ey = mx.y - mn.y;
        if (ex < 1e-6 || ey < 1e-6) return false;
        uv01 = vec2((H.x - mn.x) / ex, (H.y - mn.y) / ey);
    }

    if (uv01.x < 0.0 || uv01.x > 1.0 || uv01.y < 0.0 || uv01.y > 1.0) return false;
    return true;
}

void main() {
    int R = int(SIZE_RES_RECV.z + 0.5);
    R = max(R, 2);
    int Q = R - 1;

    const ivec2 offsets[6] = ivec2[6](
        ivec2(0, 0), ivec2(1, 0), ivec2(0, 1),
        ivec2(1, 0), ivec2(1, 1), ivec2(0, 1)
    );

    int vi = gl_VertexIndex;
    int tri = vi / 6;
    int lv  = vi % 6;
    int qx  = tri % Q;
    int qy  = tri / Q;
    ivec2 gxy = ivec2(qx, qy) + offsets[lv];

    float u = float(gxy.x) / float(Q);
    float v = float(gxy.y) / float(Q);

    float w = SIZE_RES_RECV.x;
    float h = SIZE_RES_RECV.y;
    vec2 local_xy = vec2((u - 0.5) * w, (v - 0.5) * h);

    vec3 normal_local;
    vec3 disp;
    water_waves_disp_and_normal(local_xy, SUN_TIME.w, WAVE_COUNT, WAVES, disp, normal_local);

    if (WATER_NM.z > 0.5) {
        vec2 tuv = local_xy * WATER_NM.y;
        vec3 tsn = texture(WATER_NORMAL_MAP, tuv).xyz * 2.0 - 1.0;
        float bw = clamp(WATER_NM.x, 0.0, 1.0);
        normal_local = normalize(normal_local + vec3(tsn.xy * bw, 0.0));
    }

    vec3 P_local = vec3(local_xy + disp.xy, disp.z);

    vec4 P_world4 = WORLD_FROM_LOCAL * vec4(P_local, 1.0);
    vec3 P_world  = P_world4.xyz;

    vec3 N_world = normalize((WORLD_FROM_LOCAL * vec4(normal_local, 0.0)).xyz);

    vec3 L;
    vec3 I;
    if (CAUSTIC_LIGHT_POINT.w > 0.5) {
        vec3 toL = CAUSTIC_LIGHT_POINT.xyz - P_world;
        float dtl = length(toL);
        if (dtl < 1e-4) {
            vec2 cCenter = CAUSTIC_CEI.xy;
            float cExtent = max(CAUSTIC_CEI.z, 1e-4);
            vec2 uv_fallback = (P_world.xy - cCenter) / cExtent + 0.5;
            vec2 ndc_fb = uv_fallback * 2.0 - 1.0;
            gl_Position = vec4(ndc_fb, 0.5, 1.0);
            vContrib = 0.0;
            return;
        }
        L = toL / dtl;
        I = -L;
    } else {
        L = normalize(SUN_TIME.xyz);
        I = -L;
    }
    vec3 Rd = reflect(I, N_world);

    vec2 uv01;
    vec2 hit_xy = P_world.xy;
    bool valid = intersect_room_face(P_world, Rd, push.faceIndex, uv01);
    if (!valid) {
        vec2 cCenter = CAUSTIC_CEI.xy;
        float cExtent = max(CAUSTIC_CEI.z, 1e-4);
        vec2 uv_fallback = (hit_xy - cCenter) / cExtent + 0.5;
        vec2 ndc_fb = uv_fallback * 2.0 - 1.0;
        gl_Position = vec4(ndc_fb, 0.5, 1.0);
        vContrib = 0.0;
        return;
    }

    vec2 ndc = uv01 * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.5, 1.0);

    float ndotL = max(0.0, dot(N_world, L));
    float cIntensity = CAUSTIC_CEI.w;
    float density_scale = 1.0 / float(Q * Q);
    float att = 1.0;
    if (CAUSTIC_LIGHT_POINT.w > 0.5) {
        float dist = length(CAUSTIC_LIGHT_POINT.xyz - P_world);
        att = 1.0 / max(4.0 * 3.14159265359 * dist * dist, 1e-4);
        att = min(att, 12.0);
    }
    vContrib = ndotL * cIntensity * density_scale * att;
}
