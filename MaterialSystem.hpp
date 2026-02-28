#pragma once

#include <vector>
#include <unordered_map>
#include "RTG.hpp"
#include "Helpers.hpp"
#include "S72.hpp"

// struct MaterialTextureInfo {
// 	uint32_t albedo_tex_index = 0;
// 	uint32_t has_albedo_tex = 0;	
// };

struct MaterialSystem {

	struct MaterialTextureInfo {
		uint32_t albedo_tex_index = 0;
		uint32_t has_albedo_tex = 0;	
	};

	
	MaterialSystem(RTG &);
	~MaterialSystem();
    
    void build_material_texture(S72 const &s72);
	void load_all_textures();
	void load_environment_map(S72 const &s72);
	void load_environment_map_diffuse();
	

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
	Helpers::AllocatedImage env_cube;
	VkImageView env_cube_view;

	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkSampler env_cube_sampler = VK_NULL_HANDLE;
	VkDescriptorPool material_descriptor_pool = VK_NULL_HANDLE;
	std::vector< VkDescriptorSet > material_descriptors; //allocated from material_descriptor_pool

	std::string diffuse_path;
	Helpers::AllocatedImage diffuse_cube;
    VkImageView diffuse_cube_view = VK_NULL_HANDLE;

    std::vector<Helpers::AllocatedImage> specular_levels;
    std::vector<VkImageView> specular_level_views;

};