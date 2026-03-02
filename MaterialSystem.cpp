#include "MaterialSystem.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Tutorial.hpp"
#include "VK.hpp"
#include <iostream>

static inline void decode_rgbe(uint8_t const* src_rgbe, size_t pixel_count, std::vector<float>& dst_rgba) {
    dst_rgba.resize(pixel_count * 4);

    for (size_t i = 0; i < pixel_count; ++i) {
        uint8_t r = src_rgbe[4 * i + 0];
        uint8_t g = src_rgbe[4 * i + 1];
        uint8_t b = src_rgbe[4 * i + 2];
        uint8_t e = src_rgbe[4 * i + 3];

        float* out = &dst_rgba[4 * i];

        if (e == 0) {
            out[0] = out[1] = out[2] = 0.0f;
            out[3] = 1.0f;
        } else {
            float scale = std::ldexp(1.0f, int(e) - 128);
            out[0] = scale * (float(r) + 0.5f) / 256.0f;
            out[1] = scale * (float(g) + 0.5f) / 256.0f;
            out[2] = scale * (float(b) + 0.5f) / 256.0f;
            out[3] = 1.0f;
        }
    }
}

MaterialSystem::MaterialSystem(RTG &rtg_) : rtg(rtg_) {
    
}

MaterialSystem::~MaterialSystem() {
    if (material_descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, material_descriptor_pool, nullptr);
		material_descriptor_pool = nullptr;

		//this also frees the descriptor sets allocated from the pool:
		material_descriptors.clear();
	}

    if (texture_sampler) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

    if (env_cube_sampler) {
		vkDestroySampler(rtg.device, env_cube_sampler, nullptr);
		env_cube_sampler = VK_NULL_HANDLE;
	}

    for (VkImageView &view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

    for (auto &texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

    if (brdf_lut_view != VK_NULL_HANDLE) {
        vkDestroyImageView(rtg.device, brdf_lut_view, nullptr);
        brdf_lut_view = VK_NULL_HANDLE;
    }
    if (brdf_lut.handle != VK_NULL_HANDLE) {
        rtg.helpers.destroy_image(std::move(brdf_lut));
    }

    if (diffuse_cube_view != VK_NULL_HANDLE) {
        vkDestroyImageView(rtg.device, diffuse_cube_view, nullptr);
        diffuse_cube_view = VK_NULL_HANDLE;
    }

    if (diffuse_cube.handle != VK_NULL_HANDLE) {
        rtg.helpers.destroy_image(std::move(diffuse_cube));
    }

    if (env_cube_view != VK_NULL_HANDLE) {
        vkDestroyImageView(rtg.device, env_cube_view, nullptr);
        env_cube_view = VK_NULL_HANDLE;
    }

    if (env_cube.handle != VK_NULL_HANDLE) {
        rtg.helpers.destroy_image(std::move(env_cube));
    }

    for (auto &material_param : material_params) {
		rtg.helpers.destroy_buffer(std::move(material_param));
	}
	material_params.clear();
}

void MaterialSystem::build_material_texture(S72 const &s72) {
	materials_list.clear();
	materials_list.reserve(s72.materials.size());
	textures_list.clear();
	textures_list.reserve(s72.textures.size());

	for (auto const& [name, mat] : s72.materials) {
		materials_list.push_back(&mat);
	}

	for (auto const& [src, tex] : s72.textures) {
		textures_list.push_back(&tex);
	}

	std::sort(materials_list.begin(), materials_list.end(),
			[](auto a, auto b){ return a->name < b->name; });
	std::sort(textures_list.begin(), textures_list.end(),
		[](auto a, auto b){
			if (a->src != b->src) return a->src < b->src;
			if (a->type != b->type) return int(a->type) < int(b->type);
			return int(a->format) < int(b->format);
		});


	material_id.clear();
	for (uint32_t i = 0; i < (uint32_t)materials_list.size(); ++i) {
		//default material has index 0
		material_id[materials_list[i]] = i + 1;
	}
	texture_id.clear();
	for (uint32_t i = 0; i < (uint32_t)textures_list.size(); ++i) {
		//5 default textures
		texture_id[textures_list[i]] = i + 5;
	}

	//default material has index 0
	material_params.push_back(rtg.helpers.create_buffer(
			sizeof(Tutorial::ObjectsPipeline::Material), 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
	));

	{
        //default material
        Tutorial::ObjectsPipeline::Material m{};
        m.ALBEDO = {0.8f, 0.8f, 0.8f, 0.0f};
        //0=Lambertian, 1=Environment, 2=Mirror
        m.TYPE = 0u;
        std::memcpy(material_params[0].allocation.data(), &m, sizeof(m));
    }

	material_tex_info.emplace_back(MaterialTextureInfo{0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

	for (uint32_t i = 0; i < (uint32_t)materials_list.size(); ++i) {
		S72::Material const* mat = materials_list[i];

		material_params.push_back(rtg.helpers.create_buffer(
			sizeof(Tutorial::ObjectsPipeline::Material), 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		));

		MaterialTextureInfo info{};

        Tutorial::ObjectsPipeline::Material out{};
        out.ALBEDO = {0.8f, 0.8f, 0.8f, 0.0f};
        out.PBR = {1.0f, 0.0f, 0.0f, 0.0f};
        out.TYPE = 0u;

        auto read_albedo = [&](auto const& v, auto &out_albedo, uint32_t &out_tex_index, uint32_t &out_has_tex)
        {
            out_tex_index = 0;
            out_has_tex   = 0;

            if (std::holds_alternative<S72::color>(v)) {
                auto const& c = std::get<S72::color>(v);
                out_albedo = {c.r, c.g, c.b, 0.0f};
            } else {
                auto const* tex = std::get<S72::Texture*>(v);
                out_tex_index = texture_id.at(tex);
                out_has_tex   = 1;
                out_albedo = {1.0f, 1.0f, 1.0f, 0.0f};
            }
        };


        auto read_pbr = [&](auto const& v, float &out_scalar, uint32_t &out_tex_index, uint32_t &out_has_tex)
        {
            out_tex_index = 0;
            out_has_tex  = 0;

            if (std::holds_alternative<float>(v)) {
                out_scalar = std::get<float>(v);
            } else {
                auto const* tex = std::get<S72::Texture*>(v);
                out_tex_index = texture_id.at(tex);
                out_has_tex = 1;
                out_scalar = 1.0f;
            }
        };

		if (std::holds_alternative<S72::Material::Lambertian>(mat->brdf)) {
            out.TYPE = 0u;
			auto const& lam = std::get<S72::Material::Lambertian>(mat->brdf);

            read_albedo(
                lam.albedo,
                out.ALBEDO,
                info.albedo_tex_index,
                info.has_albedo_tex
            );
		}
        else if (std::holds_alternative<S72::Material::Environment>(mat->brdf)) {
            // environment: look up env map by normal
            out.TYPE = 1u;
            
        }
        else if (std::holds_alternative<S72::Material::Mirror>(mat->brdf)) {
            // mirror: look up env map by reflection vector
            out.TYPE = 2u;
        
        }
        else if (std::holds_alternative<S72::Material::PBR>(mat->brdf)) {
            // PBR
            out.TYPE = 3u;
            auto const& pbr = std::get<S72::Material::PBR>(mat->brdf);

            read_albedo(
                pbr.albedo,
                out.ALBEDO,
                info.albedo_tex_index,
                info.has_albedo_tex
            );

            // roughness (default 1.0)
            read_pbr(
                pbr.roughness,
                out.PBR.ROUGHNESS,
                info.roughness_tex_index,
                info.has_roughness_tex
            );

            // metallic (default 0.0)
            read_pbr(
                pbr.metalness,
                out.PBR.METALLIC,
                info.metallic_tex_index,
                info.has_metallic_tex
            );
        }

        if (mat->normal_map) {
            info.normal_tex_index = texture_id.at(mat->normal_map);
            info.has_normal_tex = 1;
        }

        if (mat->displacement_map) {
            info.displacement_tex_index = texture_id.at(mat->displacement_map);
            info.has_displacement_tex = 1;
        }

        std::memcpy(material_params[i + 1].allocation.data(), &out, sizeof(out));
		material_tex_info.push_back(info);
	
	}
}

void MaterialSystem::load_all_textures() {
	// textures.clear();
    // textures.reserve(textures_list.size());

	stbi_set_flip_vertically_on_load(true);

    // { //the default first texture will be a white texture.
    //     //actually make the texture:
    //     uint32_t size = 128;
    //     std::vector< uint32_t > data;
    //     data.reserve(size * size);
    //     for (uint32_t y = 0; y < size; ++y) {
    //         for (uint32_t x = 0; x < size; ++x) {
    //             data.emplace_back(0xffffffff); //white
    //         }
    //     }
    //     assert(data.size() == size*size);

    //     //make a place for the texture to live on the GPU:
    //     textures.emplace_back(rtg.helpers.create_image(
    //         VkExtent2D{ .width = size , .height = size }, //size of image
    //         VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
    //         VK_IMAGE_TILING_OPTIMAL,
    //         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
    //         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
    //         Helpers::Unmapped
    //     ));

    //     //transfer data:
    //     rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());

    //     //
    //     default_texture_index = 0;
    // }

    auto make_solid_rgba8 = [&](uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> uint32_t {
        return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
    };

    auto push_solid_texture = [&](uint32_t &out_index, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        uint32_t size = 128;
        std::vector<uint32_t> data;
        data.reserve(size * size);
        uint32_t px = make_solid_rgba8(r,g,b,a);
        for (uint32_t i = 0; i < size * size; ++i) data.emplace_back(px);

        textures.emplace_back(rtg.helpers.create_image(
            VkExtent2D{ .width = size , .height = size },
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped
        ));
        rtg.helpers.transfer_to_image(data.data(), sizeof(uint32_t) * data.size(), textures.back());
        out_index = uint32_t(textures.size() - 1);
    };

    // default albedo texture = 0
    push_solid_texture(default_albedo_texture_index, 255,255,255,255);

    // default roughness = 1
    push_solid_texture(default_roughness_texture_index, 255,255,255,255);

    // default metallic = 0
    push_solid_texture(default_metallic_texture_index, 255,255,255,255);

    // default normal = (0.5,0.5,1.0)
    push_solid_texture(default_normal_texture_index, 128,128,255,255);

    // default displacement = 0
    push_solid_texture(default_displacement_texture_index, 0,0,0,255);
    
    for (size_t i = 0; i < textures_list.size(); ++i) {
        S72::Texture const* tex = textures_list[i];

        int w = 0, h = 0, comp = 0;
        stbi_uc* pixels = stbi_load(tex->path.c_str(), &w, &h, &comp, 4);
        if (!pixels) throw std::runtime_error(std::string("stbi_load failed: ") + tex->path);

        std::vector<uint32_t> data;
        data.resize(size_t(w) * size_t(h));
        std::memcpy(data.data(), pixels, data.size() * sizeof(uint32_t));
        stbi_image_free(pixels);

        VkFormat fmt = (tex->format == S72::Texture::Format::srgb)
            ? VK_FORMAT_R8G8B8A8_SRGB
            : VK_FORMAT_R8G8B8A8_UNORM;

        textures.emplace_back(rtg.helpers.create_image(
            VkExtent2D{ (uint32_t)w, (uint32_t)h },
            fmt,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped
        ));

        rtg.helpers.transfer_to_image(data.data(), data.size() * sizeof(uint32_t), textures.back());

    }
}

void MaterialSystem::load_environment_map(S72 const &s72) {
    if (s72.environments.empty()) return;

    stbi_set_flip_vertically_on_load(false);

    S72::Environment const &env = s72.environments.begin()->second;
    S72::Texture const* tex = env.radiance;

    assert(tex->type == S72::Texture::Type::cube);
    assert(tex->format == S72::Texture::Format::rgbe);

    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(tex->path.c_str(), &w, &h, &comp, 4);
    if (!pixels) throw std::runtime_error("Failed to load RGBE cubemap.");

    size_t pixel_count = size_t(w) * size_t(h);

    std::vector<float> decoded(pixel_count * 4);

    decode_rgbe(pixels, pixel_count, decoded);

    stbi_image_free(pixels);
    
    uint32_t face = uint32_t(w);
    assert(h == int(6 * face));

    uint32_t max_mip = 0;
    uint32_t t = face;
    while (t > 1) { t >>= 1; max_mip++; }
    env_specular_max_mip = max_mip;
    

    env_cube = rtg.helpers.create_image(
        VkExtent2D{face, face},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        Helpers::Unmapped,
        6, //arrayLayers
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, //flags
        max_mip + 1
    );

    rtg.helpers.transfer_to_cubemap(decoded.data(), decoded.size() * sizeof(float), env_cube);

    std::string base = tex->path.substr(0, tex->path.find_last_of('.'));

    for (uint32_t mip = 1; mip <= max_mip; ++mip) {
        std::string path = base + "." + std::to_string(mip) + ".png";

        int mw = 0, mh = 0, mcomp = 0;
        stbi_uc* mpixels = stbi_load(path.c_str(), &mw, &mh, &mcomp, 4);
        if (!mpixels) break;

        size_t mcount = size_t(mw) * size_t(mh);
        std::vector<float> mdecoded(mcount * 4);

        decode_rgbe(mpixels, mcount, mdecoded);

        
        stbi_image_free(mpixels);

        rtg.helpers.transfer_to_cubemap_level(mdecoded.data(),mdecoded.size() * sizeof(float), env_cube, mip);
    }

    {
        VkImageViewCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = env_cube.handle,
            .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = max_mip + 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            },
        };

        VK( vkCreateImageView(rtg.device, &create_info, nullptr, &env_cube_view) );
    }
    
    {
        VkSamplerCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,

            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,

            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,

            .minLod = 0.0f,
            .maxLod = float(max_mip),

            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VK(vkCreateSampler(rtg.device, &create_info, nullptr, &env_cube_sampler));
    }
    
    diffuse_path = tex->path.substr(0, tex->path.find_last_of('.')) + ".lambertian.png";

}

void MaterialSystem::load_environment_map_diffuse() {

    stbi_set_flip_vertically_on_load(false);

    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(diffuse_path.c_str(), &w, &h, &comp, 4);
    if (!pixels) throw std::runtime_error("Failed to load RGBE cubemap.");

    size_t pixel_count = size_t(w) * size_t(h);

    std::vector<float> decoded(pixel_count * 4);

    decode_rgbe(pixels, pixel_count, decoded);

    stbi_image_free(pixels);
    
    uint32_t face = uint32_t(w);
    assert(h == int(6 * face));

    diffuse_cube = rtg.helpers.create_image(
        VkExtent2D{face, face},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        Helpers::Unmapped,
        6, //arrayLayers
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT //flags
    );

    rtg.helpers.transfer_to_cubemap(decoded.data(), decoded.size() * sizeof(float), diffuse_cube);

    {
        VkImageViewCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = diffuse_cube.handle,
            .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            },
        };

        VK( vkCreateImageView(rtg.device, &create_info, nullptr, &diffuse_cube_view) );
    }
    
}

void MaterialSystem::load_brdf_lut(std::string const& path, uint32_t size) {
    //raw layout: size * size * 2 uint16 (RG)
    std::vector<uint16_t> rg(size * size * 2);
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open BRDF LUT raw: " + path);
        f.read(reinterpret_cast<char*>(rg.data()), rg.size() * sizeof(uint16_t));
        if (!f) throw std::runtime_error("Failed to read BRDF LUT raw fully.");
    }

    brdf_lut = rtg.helpers.create_image(
        VkExtent2D{size, size},
        VK_FORMAT_R16G16_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        Helpers::Unmapped
    );

    rtg.helpers.transfer_to_image(rg.data(), rg.size()*sizeof(uint16_t), brdf_lut);

    VkImageViewCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = brdf_lut.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R16G16_UNORM,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };
    VK( vkCreateImageView(rtg.device, &view, nullptr, &brdf_lut_view) );
}