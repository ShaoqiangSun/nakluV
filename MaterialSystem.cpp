#include "MaterialSystem.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Tutorial.hpp"

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

    for (VkImageView &view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

    for (auto &texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

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
		//default texture has index 0
		texture_id[textures_list[i]] = i + 1;
	}

	//default material has index 0
	material_params.push_back(rtg.helpers.create_buffer(
			sizeof(Tutorial::ObjectsPipeline::Material), 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
	));

	S72::color c{0.8f, 0.8f, 0.8f};
	std::memcpy(material_params[0].allocation.data(), &c, sizeof(c));

	material_tex_info.emplace_back(MaterialTextureInfo{0, 0});

	for (uint32_t i = 0; i < (uint32_t)materials_list.size(); ++i) {
		S72::Material const* mat = materials_list[i];

		material_params.push_back(rtg.helpers.create_buffer(
			sizeof(Tutorial::ObjectsPipeline::Material), 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		));

		MaterialTextureInfo info{};

		if (std::holds_alternative<S72::Material::Lambertian>(mat->brdf)) {
			auto const& lam = std::get<S72::Material::Lambertian>(mat->brdf);
			if (std::holds_alternative<S72::color>(lam.albedo)) {
				auto const& c = std::get<S72::color>(lam.albedo);
				//default material has index 0
				std::memcpy(material_params[i + 1].allocation.data(), &c, sizeof(c));
			} 
			else {
				auto const* tex = std::get<S72::Texture*>(lam.albedo);
				info.albedo_tex_index = texture_id.at(tex);
				info.has_albedo_tex = 1;

				S72::color c{1.0f, 1.0f, 1.0f};
				//default material has index 0
				std::memcpy(material_params[i + 1].allocation.data(), &c, sizeof(c));
			}
		}

		material_tex_info.push_back(info);

		
	}

}

void MaterialSystem::load_all_textures() {
	// textures.clear();
    // textures.reserve(textures_list.size());

	stbi_set_flip_vertically_on_load(true);

    { //the default first texture will be a white texture.
        //actually make the texture:
        uint32_t size = 128;
        std::vector< uint32_t > data;
        data.reserve(size * size);
        for (uint32_t y = 0; y < size; ++y) {
            for (uint32_t x = 0; x < size; ++x) {
                data.emplace_back(0xffffffff); //white
            }
        }
        assert(data.size() == size*size);

        //make a place for the texture to live on the GPU:
        textures.emplace_back(rtg.helpers.create_image(
            VkExtent2D{ .width = size , .height = size }, //size of image
            VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
            Helpers::Unmapped
        ));

        //transfer data:
        rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());

        //
        default_texture_index = 0;
    }
    
    for (size_t i = 0; i < textures_list.size(); ++i) {
        S72::Texture const* tex = textures_list[i];

        int w=0, h=0, comp=0;
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