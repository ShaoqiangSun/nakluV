#include "PrecomputedIBL.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <glm/gtc/constants.hpp>

#include <random>

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

static inline glm::vec3 cosine_sample(float r1, float r2) {
    float phi = 2.0f * glm::pi<float>() * r1;
    //r2 = sin_theta ^ 2
    float sin_theta = std::sqrt(r2);

    return glm::vec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), std::sqrt(1.0f - r2));
}

static inline void make_basis(glm::vec3 n, glm::vec3 &t, glm::vec3 &b) {
    glm::vec3 up = (std::abs(n.z) < 0.999f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
    t = glm::normalize(glm::cross(up, n));
    b = glm::cross(n, t);
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

    auto sample_env_cube = [&](glm::vec3 dir)->glm::vec3 {
        dir = glm::normalize(dir);

        float ax = std::abs(dir.x), ay = std::abs(dir.y), az = std::abs(dir.z);
        int face;
        float u,v;

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

        uint32_t x = std::min(uint32_t(u * env_cube_face_size), env_cube_face_size - 1);
        uint32_t y = std::min(uint32_t(v * env_cube_face_size), env_cube_face_size - 1);

        size_t idx = (size_t(face) * env_cube_face_size * env_cube_face_size + y * env_cube_face_size + x);

        return glm::vec3(env_cube_rgba[idx]);
    };

    for (int face = 0; face < 6; ++face) {
        for (uint32_t y = 0; y < out_face_size; ++y) {
            for (uint32_t x = 0; x < out_face_size; ++x) {

                float u = (x + 0.5f) / out_face_size;
                float v = (y + 0.5f) / out_face_size;

                glm::vec3 n = face_uv_to_dir(face, u, v);
                glm::vec3 t, b;
                make_basis(n, t, b);

                glm::vec3 sum(0);

                for (uint32_t s = 0; s < samples; ++s) {
                    glm::vec3 l  = cosine_sample(uni(rng), uni(rng));
                    glm::vec3 wi = t * l.x + b * l.y + n * l.z;
                    sum += sample_env_cube(wi);
                }

                glm::vec3 E = sum * (glm::pi<float>() / float(samples));

                size_t out_idx = (face * out_face_size * out_face_size + y * out_face_size + x) * 4;

                float maxc = std::max({E.r, E.g, E.b});

                if (maxc < 1e-32f) {
                    out[out_idx + 0] = 0;
                    out[out_idx + 1] = 0;
                    out[out_idx + 2] = 0;
                    out[out_idx + 3] = 0;
                } else {
                    int exp;
                    std::frexp(maxc, &exp);

                    float scale = std::ldexp(1.0f, -exp);

                    out[out_idx + 0] = uint8_t(glm::clamp(scale * E.r * 256 - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 1] = uint8_t(glm::clamp(scale * E.g * 256 - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 2] = uint8_t(glm::clamp(scale * E.b * 256 - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 3] = uint8_t(exp + 128);
                }
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

    auto fetch_face = [&](int face, int x, int y) -> glm::vec3 {
        x = std::max(0, std::min(x, int(env_cube_face_size) - 1));
        y = std::max(0, std::min(y, int(env_cube_face_size) - 1));
        size_t idx = size_t(face) * env_cube_face_size * env_cube_face_size + size_t(y) * env_cube_face_size + size_t(x);

        return glm::vec3(env_cube_rgba[idx]);
    };

    auto bilinear_face = [&](int face, float u, float v) -> glm::vec3 {
        u = std::max(0.0f, std::min(u, 1.0f));
        v = std::max(0.0f, std::min(v, 1.0f));

        float fx = u * float(env_cube_face_size) - 0.5f;
        float fy = v * float(env_cube_face_size) - 0.5f;

        int x0 = int(std::floor(fx));
        int y0 = int(std::floor(fy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float tx = fx - float(x0);
        float ty = fy - float(y0);

        glm::vec3 c00 = fetch_face(face, x0, y0);
        glm::vec3 c10 = fetch_face(face, x1, y0);
        glm::vec3 c01 = fetch_face(face, x0, y1);
        glm::vec3 c11 = fetch_face(face, x1, y1);

        glm::vec3 c0 = glm::mix(c00, c10, tx);
        glm::vec3 c1 = glm::mix(c01, c11, tx);
        return glm::mix(c0, c1, ty);
    };

    for (int face = 0; face < 6; ++face) {
        for (uint32_t y = 0; y < src_down_size; ++y) {
            for (uint32_t x = 0; x < src_down_size; ++x) {
                float u = (float(x) + 0.5f) / float(src_down_size);
                float v = (float(y) + 0.5f) / float(src_down_size);

                glm::vec3 c = bilinear_face(face, u, v);

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
        float u1 = float(ix + 1) / float(N);
        float v1 = float(iy + 1) / float(N);

        float x0 = 2.0f * u0 - 1.0f;
        float y0 = 2.0f * v0 - 1.0f;
        float x1 = 2.0f * u1 - 1.0f;
        float y1 = 2.0f * v1 - 1.0f;

        //solid_angle(texel) = A(x1,y1) - A(x0,y1) - A(x1,y0) + A(x0,y0)
        return texel_area(x1, y1) - texel_area(x0, y1) - texel_area(x1, y0) + texel_area(x0, y0);
    };

    //Direct integral over cubemap texels
    for (int out_face = 0; out_face < 6; ++out_face) {
        for (uint32_t y = 0; y < out_face_size; ++y) {
            for (uint32_t x = 0; x < out_face_size; ++x) {

                float uo = (float(x) + 0.5f) / float(out_face_size);
                float vo = (float(y) + 0.5f) / float(out_face_size);
                glm::vec3 n = face_uv_to_dir(out_face, uo, vo);

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
                size_t out_idx = (size_t(out_face) * out_face_size * out_face_size + size_t(y) * out_face_size + size_t(x)) * 4;

                float maxc = std::max({E.r, E.g, E.b});
                if (maxc < 1e-32f) {
                    out[out_idx + 0] = 0;
                    out[out_idx + 1] = 0;
                    out[out_idx + 2] = 0;
                    out[out_idx + 3] = 0;
                } else {
                    int exp;
                    std::frexp(maxc, &exp);               //maxc = m * 2^exp
                    float scale = std::ldexp(1.0f, -exp); //1 / 2 ^ exp

                    out[out_idx + 0] = uint8_t(glm::clamp(scale * E.r * 256.0f - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 1] = uint8_t(glm::clamp(scale * E.g * 256.0f - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 2] = uint8_t(glm::clamp(scale * E.b * 256.0f - 0.5f, 0.0f, 255.0f));
                    out[out_idx + 3] = uint8_t(exp + 128);
                }
            }
        }
    }

    std::string path = env_cube_out_path.empty() ? env_cube_path.substr(0, env_cube_path.find_last_of('.')) + ".lambertian.png" : env_cube_out_path;
    stbi_write_png(path.c_str(), int(out_face_size), int(out_face_size * 6), 4, out.data(), int(out_face_size * 4));
}