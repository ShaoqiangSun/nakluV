#pragma once

#include <vector>
#include <unordered_map>
#include "RTG.hpp"
#include "Helpers.hpp"
#include "S72.hpp"

struct MaterialTextureInfo {
	uint32_t albedo_tex_index = 0;
	uint32_t has_albedo_tex = 0;
		
};

struct MaterialSystem {
	MaterialSystem(RTG &);
	~MaterialSystem();
    
    void build_material_texture(S72 const &s72);
	void load_all_textures();

	RTG &rtg;

    uint32_t default_texture_index;

    std::unordered_map<S72::Material const*, uint32_t> material_id;
	std::unordered_map<S72::Texture const*, uint32_t> texture_id;

	std::vector<S72::Material const*> materials_list;
	std::vector<S72::Texture const*> textures_list;
	std::vector< Helpers::AllocatedBuffer > material_params;
	std::vector<MaterialTextureInfo> material_tex_info;
	std::vector< Helpers::AllocatedImage > textures;
	std::vector< VkImageView > texture_views;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkDescriptorPool material_descriptor_pool = VK_NULL_HANDLE;
	std::vector< VkDescriptorSet > material_descriptors; //allocated from material_descriptor_pool
};