#include "PrecomputedIBL.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <glm/gtc/constants.hpp>

#include <random>
#include <fstream>
#include <filesystem>

static inline glm::vec3 face_uv_to_dir(int face, float u, float v) {
    float a = 2.0f * u - 1.0f;
    float b = 2.0f * v - 1.0f;

    glm::vec3 dir;
    switch(face) {
        case 0: dir = {  1, -b, -a }; break;
        case 1: dir = { -1, -b,  a }; break;
        case 2: dir = {  a,  1,  b }; break;
        case 3: dir = {  a, -1, -b }; break;
        case 4: dir = {  a, -b,  1 }; break;
        case 5: dir = { -a, -b, -1 }; break;
    }

    return glm::normalize(dir);
}

static glm::vec3 fetch_face(const std::vector<glm::vec4>& env, uint32_t face_size, int face, int x, int y)
{
    x = std::max(0, std::min(x, int(face_size) - 1));
    y = std::max(0, std::min(y, int(face_size) - 1));

    size_t idx = size_t(face) * face_size * face_size
               + size_t(y) * face_size + size_t(x);

    return glm::vec3(env[idx]);
}

static glm::vec3 bilinear_face(const std::vector<glm::vec4>& env, uint32_t face_size, int face, float u, float v)
{
    u = std::max(0.0f, std::min(u, 1.0f));
    v = std::max(0.0f, std::min(v, 1.0f));

    float fx = u * float(face_size) - 0.5f;
    float fy = v * float(face_size) - 0.5f;

    int x0 = int(std::floor(fx));
    int y0 = int(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = fx - float(x0);
    float ty = fy - float(y0);

    glm::vec3 c00 = fetch_face(env, face_size, face, x0, y0);
    glm::vec3 c10 = fetch_face(env, face_size, face, x1, y0);
    glm::vec3 c01 = fetch_face(env, face_size, face, x0, y1);
    glm::vec3 c11 = fetch_face(env, face_size, face, x1, y1);

    glm::vec3 c0 = glm::mix(c00, c10, tx);
    glm::vec3 c1 = glm::mix(c01, c11, tx);
    return glm::mix(c0, c1, ty);
}

static glm::vec3 sample_env_cubemap(const std::vector<glm::vec4>& env, uint32_t face_size, glm::vec3 dir)
{
    dir = glm::normalize(dir);

    float ax = std::abs(dir.x), ay = std::abs(dir.y), az = std::abs(dir.z);
    int face;
    float u, v;

    if (ax >= ay && ax >= az) {
        if (dir.x > 0) { face = 0; u = (-dir.z / ax + 1) * 0.5f; v = (-dir.y / ax + 1) * 0.5f; }
        else           { face = 1; u = ( dir.z / ax + 1) * 0.5f; v = (-dir.y / ax + 1) * 0.5f; }
    }
    else if (ay >= ax && ay >= az) {
        if (dir.y > 0) { face = 2; u = ( dir.x / ay + 1) * 0.5f; v = ( dir.z / ay + 1) * 0.5f; }
        else           { face = 3; u = ( dir.x / ay + 1) * 0.5f; v = (-dir.z / ay + 1) * 0.5f; }
    }
    else {
        if (dir.z > 0) { face = 4; u = ( dir.x / az + 1) * 0.5f; v = (-dir.y / az + 1) * 0.5f; }
        else           { face = 5; u = (-dir.x / az + 1) * 0.5f; v = (-dir.y / az + 1) * 0.5f; }
    }

    return bilinear_face(env, face_size, face, u, v);
}

static inline glm::vec3 cosine_sample(float r1, float r2) {
    float phi = 2.0f * glm::pi<float>() * r1;
    //r2 = sin_theta ^ 2
    float sin_theta = std::sqrt(r2);

    return glm::vec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), std::sqrt(1.0f - r2));
}

static inline glm::vec3 ggx_importance_sample(float r1, float r2, float alpha) {
    float phi = 2.0f * glm::pi<float>() * r1;

    float a2 = alpha * alpha;
    float cos_theta = std::sqrt((1.0f - r2) / (1.0f + (a2 - 1.0f) * r2));
    float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

    return glm::vec3(
        sin_theta * std::cos(phi),
        sin_theta * std::sin(phi),
        cos_theta
    );
}

static inline void make_basis(glm::vec3 n, glm::vec3 &t, glm::vec3 &b) {
    glm::vec3 up = (std::abs(n.z) < 0.999f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
    t = glm::normalize(glm::cross(up, n));
    b = glm::cross(n, t);
}

static inline glm::vec3 to_world(glm::vec3 local, glm::vec3 n) {
    glm::vec3 t, b;
    make_basis(n, t, b);
    return glm::normalize(t * local.x + b * local.y + n * local.z);
}

static inline void encode_rgbe(glm::vec3 c, uint8_t *dst4) {
    float maxc = std::max({c.r, c.g, c.b});
    if (maxc < 1e-32f) {
        dst4[0] = dst4[1] = dst4[2] = dst4[3] = 0;
        return;
    }
    int exp;
    std::frexp(maxc, &exp);               // maxc = m * 2^exp
    float scale = std::ldexp(1.0f, -exp); // 1 / 2^exp

    dst4[0] = uint8_t(glm::clamp(scale * c.r * 256.0f - 0.5f, 0.0f, 255.0f));
    dst4[1] = uint8_t(glm::clamp(scale * c.g * 256.0f - 0.5f, 0.0f, 255.0f));
    dst4[2] = uint8_t(glm::clamp(scale * c.b * 256.0f - 0.5f, 0.0f, 255.0f));
    dst4[3] = uint8_t(exp + 128);
}

static inline uint16_t float_to_unorm16(float v) {
    v = std::max(0.0f, std::min(v, 1.0f));
    uint32_t u = uint32_t(v * 65535.0f + 0.5f);
    if (u > 65535u) u = 65535u;
    return uint16_t(u);
}

static float D_ggx(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = glm::pi<float>() * denom * denom;

    return a2 / denom;
}

static float G1_smith(float NdotX, float alpha)
{
    float a2 = alpha * alpha;
    float denom = NdotX + std::sqrt(a2 + (1.0f - a2) * NdotX * NdotX);
    return (2.0f * NdotX) / denom;
}

static float G_smith(float NdotV, float NdotL, float alpha)
{
    return G1_smith(NdotV, alpha) * G1_smith(NdotL, alpha);
}

static float radical_inverse_vdc(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    // scale to [0,1)
    return float(bits) * 2.3283064365386963e-10f; // 1 / 2^32
}

static glm::vec2 hammersley_2d(uint32_t i, uint32_t N) {
    return glm::vec2((float(i) + 0.5f) / float(N), radical_inverse_vdc(i));
}

void PrecomputedIBL::load_environment_map(std::string env_cube_in_path) {
    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(env_cube_in_path.c_str(), &w, &h, &comp, 4);
    if (!pixels) throw std::runtime_error("Failed to load RGBE cubemap.");

    size_t pixel_count = size_t(w) * size_t(h);

    std::vector<float> decoded(pixel_count * 4);

    for (size_t i = 0; i < pixel_count; ++i) {
        uint8_t r = pixels[4 * i + 0];
        uint8_t g = pixels[4 * i + 1];
        uint8_t b = pixels[4 * i + 2];
        uint8_t e = pixels[4 * i + 3];

        float* out = &decoded[4 * i];
        if (e == 0) {
            out[0] = out[1] = out[2] = 0.0f;
        } 
        else {
            float scale = std::ldexp(1.0f, int(e) - 128);
            out[0] = scale * float(r + 0.5f) / 256 ;
            out[1] = scale * float(g + 0.5f) / 256;
            out[2] = scale * float(b + 0.5f) / 256;
            out[3] = 1.0f;
        }
    }

    stbi_image_free(pixels);
    
    uint32_t face = uint32_t(w);
    assert(h == int(6 * face));

    //store cubemap data
    env_cube_rgba.resize(size_t(face) * size_t(face) * 6);
    for (size_t i = 0; i < env_cube_rgba.size(); ++i) {
        env_cube_rgba[i] = glm::vec4(decoded[4 * i], decoded[4 * i + 1], decoded[4 * i + 2], decoded[4 * i + 3]);
    }
    env_cube_face_size = face;
    env_cube_path = env_cube_in_path;

}

void PrecomputedIBL::precompute_ibl_diffuse_monte_carlo(uint32_t out_face_size, uint32_t samples, std::string env_cube_out_path) {
    if (env_cube_rgba.empty()) {
        throw std::runtime_error("Call load_environment_map() first.");
    }

    std::vector<uint8_t> out(out_face_size * out_face_size * 6 * 4);

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(0.0f,1.0f);

    for (int face = 0; face < 6; ++face) {
        for (uint32_t y = 0; y < out_face_size; ++y) {
            for (uint32_t x = 0; x < out_face_size; ++x) {

                float u = (float(x) + 0.5f) / float(out_face_size);
                float v = (float(y) + 0.5f) / float(out_face_size);

                glm::vec3 n = face_uv_to_dir(face, u, v);

                glm::vec3 sum(0);

                for (uint32_t s = 0; s < samples; ++s) {
                    glm::vec3 l  = cosine_sample(uni(rng), uni(rng));
                    glm::vec3 wi = to_world(l, n);
                    sum += sample_env_cubemap(env_cube_rgba, env_cube_face_size, wi);
                }

                glm::vec3 E = sum * (glm::pi<float>() / float(samples));

                size_t out_idx = (face * out_face_size * out_face_size + y * out_face_size + x) * 4;

                encode_rgbe(E, &out[out_idx]);
            }
        }
    }

    std::string path = env_cube_out_path.empty() ? env_cube_path.substr(0, env_cube_path.find_last_of('.')) + ".lambertian.png" : env_cube_out_path;
    stbi_write_png(path.c_str(), out_face_size, out_face_size * 6, 4, out.data(), out_face_size * 4);
}


void PrecomputedIBL::precompute_ibl_diffuse_direct(uint32_t out_face_size, uint32_t src_down_size, std::string env_cube_out_path) {
    if (env_cube_rgba.empty()) {
        throw std::runtime_error("Call load_environment_map() first.");
    }

    std::vector<uint8_t> out(size_t(out_face_size) * out_face_size * 6 * 4, 0);

    std::vector<glm::vec4> env_cube_down(size_t(src_down_size) * src_down_size * 6);

    for (int face = 0; face < 6; ++face) {
        for (uint32_t y = 0; y < src_down_size; ++y) {
            for (uint32_t x = 0; x < src_down_size; ++x) {
                float u = (float(x) + 0.5f) / float(src_down_size);
                float v = (float(y) + 0.5f) / float(src_down_size);

                glm::vec3 c = bilinear_face(env_cube_rgba, env_cube_face_size, face, u, v);

                size_t idx = size_t(face) * src_down_size * src_down_size
                           + size_t(y) * src_down_size
                           + size_t(x);
                env_cube_down[idx] = glm::vec4(c, 1.0f);
            }
        }
    }

    //texel_area(x,y) = atan2(x * y, sqrt(x * x + y * y + 1))
    auto texel_area = [&](float x, float y) -> float {
        return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0f));
    };

    auto texel_solid_angle = [&](uint32_t ix, uint32_t iy, uint32_t N) -> float {
        //corners in [-1,1]
        float u0 = float(ix)     / float(N);
        float v0 = float(iy)     / float(N);
        float r1 = float(ix + 1) / float(N);
        float v1 = float(iy + 1) / float(N);

        float x0 = 2.0f * u0 - 1.0f;
        float y0 = 2.0f * v0 - 1.0f;
        float x1 = 2.0f * r1 - 1.0f;
        float y1 = 2.0f * v1 - 1.0f;

        //solid_angle(texel) = A(x1,y1) - A(x0,y1) - A(x1,y0) + A(x0,y0)
        return texel_area(x1, y1) - texel_area(x0, y1) - texel_area(x1, y0) + texel_area(x0, y0);
    };

    //Direct integral over cubemap texels
    for (int face = 0; face < 6; ++face) {
        for (uint32_t y = 0; y < out_face_size; ++y) {
            for (uint32_t x = 0; x < out_face_size; ++x) {

                float uo = (float(x) + 0.5f) / float(out_face_size);
                float vo = (float(y) + 0.5f) / float(out_face_size);
                glm::vec3 n = face_uv_to_dir(face, uo, vo);

                glm::vec3 sum(0.0f);

                for (int in_face = 0; in_face < 6; ++in_face) {
                    for (uint32_t iy = 0; iy < src_down_size; ++iy) {
                        for (uint32_t ix = 0; ix < src_down_size; ++ix) {

                            float ui = (float(ix) + 0.5f) / float(src_down_size);
                            float vi = (float(iy) + 0.5f) / float(src_down_size);
                            glm::vec3 wi = face_uv_to_dir(in_face, ui, vi);

                            float cos_theta = glm::dot(n, wi);
                            if (cos_theta <= 0.0f) continue;

                            float d_omega = texel_solid_angle(ix, iy, src_down_size);

                            size_t in_idx = size_t(in_face) * src_down_size * src_down_size
                                          + size_t(iy) * src_down_size
                                          + size_t(ix);

                            glm::vec3 Li = glm::vec3(env_cube_down[in_idx]);

                            sum += Li * (cos_theta * d_omega);
                        }
                    }
                }

                //store irradiance divided by pi (so shader shouldn't divide again)
                glm::vec3 E = sum * (1.0f / glm::pi<float>());
                // glm::vec3 E = sum;

                //encode E to rgbe
                size_t out_idx = (size_t(face) * out_face_size * out_face_size + size_t(y) * out_face_size + size_t(x)) * 4;

                encode_rgbe(E, &out[out_idx]);
            }
        }
    }

    std::string path = env_cube_out_path.empty() ? env_cube_path.substr(0, env_cube_path.find_last_of('.')) + ".lambertian.png" : env_cube_out_path;
    stbi_write_png(path.c_str(), int(out_face_size), int(out_face_size * 6), 4, out.data(), int(out_face_size * 4));
}

void PrecomputedIBL::precompute_ibl_specular_ggx(uint32_t samples, std::string env_cube_out_path) {
    if (env_cube_rgba.empty()) {
        throw std::runtime_error("Call load_environment_map() first.");
    }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(0.0f,1.0f);

    // mip1 is half resolution, mipN is 1x1
    uint32_t base = env_cube_face_size;
    if ((base & (base - 1)) != 0) {
        throw std::runtime_error("env_cube_face_size must be power-of-two for mip chain.");
    }

    uint32_t max_mip = 0;
    {
        uint32_t t = base;
        while (t > 1) { t = t >> 1; max_mip++; }
    }

    for (uint32_t mip = 1; mip <= max_mip; ++mip) {
        uint32_t out_face_size = base >> mip;

        float roughness = float(mip) / float(max_mip);
        float alpha = std::max(0.001f, roughness * roughness);

        std::vector<uint8_t> out(out_face_size * out_face_size * 6 * 4);

        for (int face = 0; face < 6; ++face) {
            for (uint32_t y = 0; y < out_face_size; ++y) {
                for (uint32_t x = 0; x < out_face_size; ++x) {

                    float u = (float(x) + 0.5f) / float(out_face_size);
                    float v = (float(y) + 0.5f) / float(out_face_size);

                    glm::vec3 N = face_uv_to_dir(face, u, v);
                    glm::vec3 V = N;

                    glm::vec3 prefiltered(0.0f);
                    float totalWeight = 0.0f;

                    for (uint32_t s = 0; s < samples; ++s) {

                        glm::vec2 Xi = hammersley_2d(s, samples);
                        glm::vec3 H_local = ggx_importance_sample(Xi.x, Xi.y, alpha);
 
                        glm::vec3 H = to_world(H_local, N);

                        glm::vec3 L = glm::normalize(2.0f * glm::dot(V, H) * H - V);

                        float NdotL = glm::max(glm::dot(N, L), 0.0f);
                        if (NdotL > 0.0f) {
                            prefiltered += sample_env_cubemap(env_cube_rgba, env_cube_face_size, L) * NdotL;
                            totalWeight += NdotL;
                        }
                    }

                    glm::vec3 E = (totalWeight > 0.0f) ? (prefiltered / totalWeight) : glm::vec3(0.0f);

                    size_t out_idx = (face * out_face_size * out_face_size + y * out_face_size + x) * 4;

                    encode_rgbe(E, &out[out_idx]);
                }
            }
        }

        auto make_mip_path = [&](std::string const& base_path, uint32_t mip) -> std::string {
            auto dot = base_path.find_last_of('.');
            std::string stem = (dot == std::string::npos) ? base_path : base_path.substr(0, dot);
            std::string ext  = (dot == std::string::npos) ? "png"     : base_path.substr(dot + 1);
            return stem + "." + std::to_string(mip) + "." + ext;
        };

        std::string base_out = env_cube_out_path.empty() ? env_cube_path : env_cube_out_path;
        std::string path = make_mip_path(base_out, mip);

        stbi_write_png(path.c_str(), int(out_face_size), int(out_face_size * 6), 4, out.data(), int(out_face_size * 4));
    }

}

void PrecomputedIBL::precompute_ibl_specular_ggx_mip(uint32_t samples, std::string env_cube_out_path) {
    if (env_cube_rgba.empty()) {
        throw std::runtime_error("Call load_environment_map() first.");
    }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(0.0f,1.0f);

    // mip1 is half resolution, mipN is 1x1
    uint32_t base = env_cube_face_size;
    if ((base & (base - 1)) != 0) {
        throw std::runtime_error("env_cube_face_size must be power-of-two for mip chain.");
    }

    uint32_t max_mip = 0;
    {
        uint32_t t = base;
        while (t > 1) { t = t >> 1; max_mip++; }
    }

    std::vector<std::vector<glm::vec4>> env_mips;
    env_mips.resize(max_mip + 1);
    env_mips[0] = env_cube_rgba;

    auto face_index = [](uint32_t face_size, int face, uint32_t x, uint32_t y) -> size_t {
        return (size_t(face) * face_size * face_size + size_t(y) * face_size + size_t(x));
    };

    for (uint32_t mip = 1; mip <= max_mip; ++mip) {
        uint32_t prev_size = base >> (mip - 1);
        uint32_t this_size = base >> mip;

        env_mips[mip].assign(size_t(this_size) * this_size * 6, glm::vec4(0.0f));

        for (int face = 0; face < 6; ++face) {
            for (uint32_t y = 0; y < this_size; ++y) {
                for (uint32_t x = 0; x < this_size; ++x) {
                    // 2x2 box filter inside each face (minimal, matches "ignore distortion" spirit)
                    uint32_t x0 = std::min(2u * x + 0u, prev_size - 1);
                    uint32_t x1 = std::min(2u * x + 1u, prev_size - 1);
                    uint32_t y0 = std::min(2u * y + 0u, prev_size - 1);
                    uint32_t y1 = std::min(2u * y + 1u, prev_size - 1);

                    glm::vec3 c00 = glm::vec3(env_mips[mip - 1][face_index(prev_size, face, x0, y0)]);
                    glm::vec3 c10 = glm::vec3(env_mips[mip - 1][face_index(prev_size, face, x1, y0)]);
                    glm::vec3 c01 = glm::vec3(env_mips[mip - 1][face_index(prev_size, face, x0, y1)]);
                    glm::vec3 c11 = glm::vec3(env_mips[mip - 1][face_index(prev_size, face, x1, y1)]);

                    glm::vec3 avg = 0.25f * (c00 + c10 + c01 + c11);
                    env_mips[mip][face_index(this_size, face, x, y)] = glm::vec4(avg, 1.0f);
                }
            }
        }
    }

    auto sample_env_cubemap_lod = [&](glm::vec3 dir, float lod) -> glm::vec3 {
        lod = std::max(0.0f, std::min(lod, float(max_mip)));
        int m0 = int(std::floor(lod));
        int m1 = std::min(m0 + 1, int(max_mip));
        float t = lod - float(m0);

        uint32_t s0 = base >> uint32_t(m0);
        uint32_t s1 = base >> uint32_t(m1);

        glm::vec3 c0 = sample_env_cubemap(env_mips[m0], s0, dir);
        glm::vec3 c1 = sample_env_cubemap(env_mips[m1], s1, dir);

        return (1.0f - t) * c0 + t * c1;
    };

    for (uint32_t mip = 1; mip <= max_mip; ++mip) {
        uint32_t out_face_size = base >> mip;

        float roughness = float(mip) / float(max_mip);
        float alpha = std::max(0.001f, roughness * roughness);

        std::vector<uint8_t> out(size_t(out_face_size) * out_face_size * 6 * 4);

        for (int face = 0; face < 6; ++face) {
            for (uint32_t y = 0; y < out_face_size; ++y) {
                for (uint32_t x = 0; x < out_face_size; ++x) {

                    float u = (float(x) + 0.5f) / float(out_face_size);
                    float v = (float(y) + 0.5f) / float(out_face_size);

                    glm::vec3 N = face_uv_to_dir(face, u, v);
                    glm::vec3 V = N;

                    glm::vec3 prefiltered(0.0f);
                    float totalWeight = 0.0f;


                    for (uint32_t s = 0; s < samples; ++s) {
                        glm::vec2 Xi = hammersley_2d(s, samples);

                        glm::vec3 H_local = ggx_importance_sample(Xi.x, Xi.y, alpha);
                        glm::vec3 H = to_world(H_local, N);

                        glm::vec3 L = glm::normalize(2.0f * glm::dot(V, H) * H - V);

                        float NdotV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f);
                        float NdotL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
                        float NdotH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f);
                        float VdotH = glm::clamp(glm::dot(V, H), 0.0f, 1.0f);

                        if (NdotL > 0.0f && NdotV > 0.0f && NdotH > 0.0f && VdotH > 0.0f) {

                            //footprint estimate:
                            float D = D_ggx(NdotH, alpha);
                            float p = (NdotH * D) / (4.0f * VdotH + 1e-8f);

                            int size = int(env_cube_face_size); // mip0 face size
                            float l = 0.5f * std::log2(3.0f * float(size) * float(size))
                                    - 0.5f * std::log2(float(samples) * p + 1e-8f);

                            l += 0.25f;
                            if (roughness < 0.01f) l *= (roughness / 0.01f);

                            l = std::max(0.0f, std::min(l, float(max_mip)));

                            glm::vec3 Li = sample_env_cubemap_lod(L, l);

                            prefiltered += Li * NdotL;
                            totalWeight += NdotL;
                        }
                    }

                    glm::vec3 E = (totalWeight > 0.0f) ? (prefiltered / totalWeight) : glm::vec3(0.0f);

                    size_t out_idx = (size_t(face) * out_face_size * out_face_size + size_t(y) * out_face_size + size_t(x)) * 4;
                    encode_rgbe(E, &out[out_idx]);
                }
            }
        }

        auto make_mip_path = [&](std::string const& base_path, uint32_t mip_level) -> std::string {
            auto dot = base_path.find_last_of('.');
            std::string stem = (dot == std::string::npos) ? base_path : base_path.substr(0, dot);
            std::string ext  = (dot == std::string::npos) ? "png"     : base_path.substr(dot + 1);
            return stem + "." + std::to_string(mip_level) + "." + ext;
        };

        std::string base_out = env_cube_out_path.empty() ? env_cube_path : env_cube_out_path;
        std::string path = make_mip_path(base_out, mip);

        stbi_write_png(path.c_str(), int(out_face_size), int(out_face_size * 6), 4, out.data(), int(out_face_size * 4));
    }
}

void PrecomputedIBL::precompute_brdf_lut(uint32_t size, uint32_t samples, std::string out_path) {
    std::vector<float> lut(size * size * 2, 0.0f);

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (uint32_t y = 0; y < size; ++y) {
        float roughness = (float(y) + 0.5f) / float(size);
        float alpha = roughness * roughness;

        for (uint32_t x = 0; x < size; ++x) {
            float NdotV = (float(x) + 0.5f) / float(size);

            glm::vec3 V;
            V.x = std::sqrt(std::max(0.0f, 1.0f - NdotV * NdotV));
            V.y = 0.0f;
            V.z = NdotV;

            float A = 0.0f;
            float B = 0.0f;

            for (uint32_t s = 0; s < samples; ++s) {

                glm::vec2 Xi = hammersley_2d(s, samples);
                glm::vec3 H = ggx_importance_sample(Xi.x, Xi.y, alpha);
                

                glm::vec3 L = glm::normalize(2.0f * glm::dot(V, H) * H - V);

                float NdotL = std::max(L.z, 0.0f);
                float NdotH = std::max(H.z, 0.0f);
                float VdotH = std::max(glm::dot(V, H), 0.0f);

                if (NdotL > 0.0f) {
                    float G = G_smith(NdotV, NdotL, alpha);

                    float G_Vis = (G * VdotH) / (NdotH * NdotV + 1e-6f);

                    float Fc = std::pow(1.0f - VdotH, 5.0f);

                    A += (1.0f - Fc) * G_Vis;
                    B += Fc * G_Vis;
                }
            }

            A /= float(samples);
            B /= float(samples);

            uint32_t y_flipped = size - 1 - y;
            size_t idx = (y_flipped * size + x) * 2;
            lut[idx + 0] = A;
            lut[idx + 1] = B;
        }
    }

    std::vector<uint16_t> out(size * size * 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float_to_unorm16(lut[i]);
    }

    {
        std::ofstream f(out_path, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open output file: " + out_path);

        f.write(reinterpret_cast<char const*>(out.data()),
                std::streamsize(out.size() * sizeof(uint16_t)));

        if (!f) throw std::runtime_error("Failed to write BRDF LUT raw: " + out_path);
    }


    std::vector<uint8_t> debug(size * size * 3);

    for (uint32_t i = 0; i < size * size; ++i)
    {
        float A = lut[i*2+0];
        float B = lut[i*2+1];

        debug[i*3+0] = uint8_t(glm::clamp(A,0.f,1.f)*255);
        debug[i*3+1] = uint8_t(glm::clamp(B,0.f,1.f)*255);
        debug[i*3+2] = 0;
    }

    std::filesystem::path out_file(out_path);
    std::filesystem::path dir = out_file.parent_path();
    std::string stem = out_file.stem().string();
    std::filesystem::path debug_file = dir / (stem + "_debug.png");

    stbi_write_png(debug_file.string().c_str(), size, size, 3, debug.data(), size*3);
}