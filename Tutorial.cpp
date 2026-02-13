#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


//Reference from https://github.com/15-472/s72-loader print_scene.cpp
void print_info(S72 &s72){
	std::cout << "--- Scene Objects ---"<< std::endl;
	std::cout << "Scene: " << s72.scene.name << std::endl;
	std::cout << "Roots: ";
	for (S72::Node* root : s72.scene.roots) {
		std::cout << root->name << ", ";
	}
	std::cout << std::endl;

	std::cout << "Nodes: ";
	for (auto const& pair : s72.nodes) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Meshes: ";
	for (auto const& pair : s72.meshes) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Cameras: ";
	for (auto const& pair : s72.cameras) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	 std::cout << "Drivers: ";
	for (auto const& driver : s72.drivers) {
		std::cout << driver.name << ", ";
	}
	std::cout << std::endl;

	std::cout << "Materials: ";
	for (auto const& pair : s72.materials) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Environment: ";
	for (auto const& pair : s72.environments) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;

	std::cout << "Lights: ";
	for (auto const& pair : s72.lights) {
		std::cout << pair.first << ", ";
	}
	std::cout << std::endl;
}

//Reference from https://github.com/15-472/s72-loader print_scene.cpp
void traverse_children(S72 &s72, S72::Node* node, std::string prefix){
	//Print node information
	std::cout << prefix << node->name << ": {";
	if(node->camera != nullptr){
		std::cout << "Camera: " << node->camera->name;
	}
	if(node->mesh != nullptr){
		std::cout << "Mesh: " << node->mesh->name;
		if(node->mesh->material != nullptr){
			std::cout << " {Material: " <<node->mesh->material->name << "}";
		}
	}
	if(node->environment != nullptr){
		std::cout << "Environment: " << node->environment->name;
	}
	if(node->light != nullptr){
		std::cout << "Light: " << node->light->name;
	}

	std::cout << "}" <<std::endl;

	std::string new_prefix = prefix + "- ";
	for(S72::Node* child : node->children){
		traverse_children(s72, child, new_prefix);
	}
}
//Reference from https://github.com/15-472/s72-loader print_scene.cpp
void print_scene_graph(S72 &s72){
	std::cout << std::endl << "--- Scene Graph ---"<< std::endl;
	for (S72::Node* root : s72.scene.roots) {
		std::cout << "Root: ";
		std::string prefix = "";
		traverse_children(s72, root, prefix);
	}
}

void print_mesh_file(S72 &s72){
	std::cout << std::endl << "--- Mesh File ---"<< std::endl;
	for (auto pair : s72.meshes) {
		std::cout << "Mesh: " << pair.first << std::endl;
		std::cout << "Src: " << pair.second.attributes.at("POSITION").src.src << std::endl;
		std::cout << "Path: " << pair.second.attributes.at("POSITION").src.path << std::endl;
	}
}

void print_data_file(S72 &s72){
	std::cout << std::endl << "--- Mesh File ---"<< std::endl;
	for (auto pair : s72.data_files) {
		std::cout << "Key: " << pair.first << std::endl;
		std::cout << "Src: " << pair.second.src << std::endl;
		std::cout << "Path: " << pair.second.path << std::endl;
	}
}

static void read_bytes(std::ifstream &f, uint64_t off, void *dst, size_t n) {
    f.seekg(std::streamoff(off), std::ios::beg);
    if (!f) throw std::runtime_error("seek failed");
    f.read(reinterpret_cast<char*>(dst), std::streamsize(n));
    if (!f) throw std::runtime_error("read failed");
}


void Tutorial::load_mesh_vertices(S72 const &s72, std::vector< PosNorTanTexVertex > &vertices_pool) {
	for (auto const& [name, mesh] : s72.meshes) {
		std::cout << "Material: " << mesh.material->name << std::endl;

		auto const &position = mesh.attributes.at("POSITION");
		auto const &normal = mesh.attributes.at("NORMAL");
		auto const &tangent = mesh.attributes.at("TANGENT");
		auto const &texcoord  = mesh.attributes.at("TEXCOORD");

		ObjectVertices range;
		range.first = uint32_t(vertices_pool.size());
		range.count = mesh.count;
		mesh_vertices.emplace(name, range);

		vertices_pool.resize(vertices_pool.size() + mesh.count);

		std::string path = position.src.path;
		std::ifstream f(path, std::ios::binary);
		if (!f) throw std::runtime_error("Failed to open " + path);

		AABB aabb; // local-space

		for (uint32_t i = 0; i < mesh.count; ++i) {
			auto &v = vertices_pool[range.first + i];
		
			read_bytes(f, uint64_t(position.offset) + uint64_t(i) * position.stride, &v.Position, sizeof(v.Position));
			
			read_bytes(f, uint64_t(normal.offset) + uint64_t(i) * normal.stride, &v.Normal, sizeof(v.Normal));
	
			read_bytes(f, uint64_t(tangent.offset) + uint64_t(i) * tangent.stride, &v.Tangent, sizeof(v.Tangent));
	
			read_bytes(f, uint64_t(texcoord.offset) + uint64_t(i) * texcoord.stride, &v.TexCoord, sizeof(v.TexCoord));

			glm::vec3 p(v.Position.x, v.Position.y, v.Position.z);
            aabb.min = glm::min(aabb.min, p);
            aabb.max = glm::max(aabb.max, p);
		}

		mesh_aabb_local[name] = aabb;
		
	}
}

void Tutorial::append_bbox_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	glm::vec3 corners[8] = {
        {local.min.x, local.min.y, local.min.z},
        {local.max.x, local.min.y, local.min.z},
        {local.max.x, local.max.y, local.min.z},
        {local.min.x, local.max.y, local.min.z},
        {local.min.x, local.min.y, local.max.z},
        {local.max.x, local.min.y, local.max.z},
        {local.max.x, local.max.y, local.max.z},
        {local.min.x, local.max.y, local.max.z},
    };

	glm::vec3 p[8];
	for (int i = 0; i < 8; ++i) {
        p[i] = WORLD_FROM_LOCAL * glm::vec4(corners[i], 1.0f);
    }

	auto edge = [&](int idx1, int idx2) { 
		PosColVertex v1, v2;
		v1.Position.x = p[idx1].x; v1.Position.y = p[idx1].y; v1.Position.z = p[idx1].z;
		v1.Color.r = r; v1.Color.g = g; v1.Color.b = b; v1.Color.a = a;

		v2.Position.x = p[idx2].x; v2.Position.y = p[idx2].y; v2.Position.z = p[idx2].z;
		v2.Color.r = r; v2.Color.g = g; v2.Color.b = b; v2.Color.a = a;

		bbox_lines_vertices.push_back(v1);
    	bbox_lines_vertices.push_back(v2);

		// lines_vertices.push_back(v1);
    	// lines_vertices.push_back(v2);
	};


    // bottom
    edge(0,1); edge(1,2); edge(2,3); edge(3,0);
    // top
    edge(4,5); edge(5,6); edge(6,7); edge(7,4);
    // verticals
    edge(0,4); edge(1,5); edge(2,6); edge(3,7);

}

void Tutorial::append_frustum_lines_world(glm::mat4 const &CLIP_FROM_WORLD, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	// glm::mat4 CAMERA_FROM_CLIP = glm::inverse(PROJ);
	// glm::mat4 WORLD_FROM_CAMERA = glm::inverse(VIEW);
	glm::mat4 WORLD_FROM_CLIP = glm::inverse(CLIP_FROM_WORLD);

	glm::vec3 ndc[8] = {
        {-1,-1,0}, {+1,-1,0}, {+1,+1,0}, {-1,+1,0}, // near
        {-1,-1,1}, {+1,-1,1}, {+1,+1,1}, {-1,+1,1}, // far
    };

	glm::vec3 p[8];
	for (int i = 0; i < 8; ++i) {
		// glm::vec4 ndc_homo(ndc[i], 1.0f);

		// glm::vec4 clip_homo = CAMERA_FROM_CLIP * ndc_homo;

		// glm::vec3 cam = glm::vec3(clip_homo) / clip_homo.w;

		// glm::vec4 world_homo = WORLD_FROM_CAMERA * glm::vec4(cam, 1.0f);
		// p[i] = glm::vec3(world_homo);

		glm::vec4 ndc_homo(ndc[i], 1.0f);
		glm::vec4 world = WORLD_FROM_CLIP * ndc_homo;
        p[i] = glm::vec3(world) / world.w;
    }

	auto edge = [&](int idx1, int idx2) { 
		PosColVertex v1, v2;
		v1.Position.x = p[idx1].x; v1.Position.y = p[idx1].y; v1.Position.z = p[idx1].z;
		v1.Color.r = r; v1.Color.g = g; v1.Color.b = b; v1.Color.a = a;

		v2.Position.x = p[idx2].x; v2.Position.y = p[idx2].y; v2.Position.z = p[idx2].z;
		v2.Color.r = r; v2.Color.g = g; v2.Color.b = b; v2.Color.a = a;

		frustum_lines_vertices.push_back(v1);
    	frustum_lines_vertices.push_back(v2);

		// lines_vertices.push_back(v1);
    	// lines_vertices.push_back(v2);
	};


    // near
    edge(0,1); edge(1,2); edge(2,3); edge(3,0);
    // far
    edge(4,5); edge(5,6); edge(6,7); edge(7,4);
   // sides
    edge(0,4); edge(1,5); edge(2,6); edge(3,7);
}

static inline mat4 to_mat4(glm::mat4 const& m) {
    mat4 out;
    static_assert(sizeof(out) == sizeof(glm::mat4), "sizes must match");
    std::memcpy(out.data(), &m[0][0], sizeof(mat4));
    return out;
}

static inline glm::mat4 to_glm(mat4 const& m) {
    glm::mat4 out;
    static_assert(sizeof(out) == sizeof(mat4), "sizes must match");
    std::memcpy(&out[0][0], m.data(), sizeof(mat4));
    return out;
}

void Tutorial::traverse_node(S72::Node* node, glm::mat4 const& parent) {
	glm::mat4 local =
		glm::translate(glm::mat4(1.0f), node->translation) *
		glm::mat4_cast(node->rotation) *
		glm::scale(glm::mat4(1.0f), node->scale);

	glm::mat4 world_glm = parent * local; 
	glm::mat3 world_normal3 = glm::transpose(glm::inverse(glm::mat3(world_glm)));
	glm::mat4 world_normal4(1.0f);
	world_normal4[0] = glm::vec4(world_normal3[0], 0.0f);
	world_normal4[1] = glm::vec4(world_normal3[1], 0.0f);
	world_normal4[2] = glm::vec4(world_normal3[2], 0.0f);

	mat4 world_matrix = to_mat4(world_glm);
	mat4 world_matrix_normal = to_mat4(world_normal4);

	if (node->mesh != nullptr) {
		ObjectVertices vertices = mesh_vertices[node->mesh->name];

		AABB local = mesh_aabb_local.at(node->mesh->name);

		append_bbox_lines_world(local, world_glm, 255, 0, 0, 255); // red

		object_instances.emplace_back(ObjectInstance{
			.vertices = vertices,
			.transform{
				//.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_matrix,
				.WORLD_FROM_LOCAL = world_matrix,
				.WORLD_FROM_LOCAL_NORMAL = world_matrix_normal,
			},
			.material = material_id.at(node->mesh->material),
		});
	}

	if (node->light != nullptr) {
		if (std::holds_alternative<S72::Light::Sun>(node->light->source)) {
			auto const& sun = std::get<S72::Light::Sun>(node->light->source);

			glm::vec3 light_dir = -glm::normalize(glm::vec3(world_glm * glm::vec4(0,0,-1,0)));
			if (sun.angle < 1e-4f) {
				world.SUN_DIRECTION.x = light_dir.x;
				world.SUN_DIRECTION.y = light_dir.y;
				world.SUN_DIRECTION.z = light_dir.z;
				world.SUN_ENERGY.r = node->light->tint.r * sun.strength;
				world.SUN_ENERGY.g = node->light->tint.g * sun.strength;
				world.SUN_ENERGY.b = node->light->tint.b * sun.strength;
				// world.SUN_ENERGY.r = 0.0f;
				// world.SUN_ENERGY.g = 0.0f;
				// world.SUN_ENERGY.b = 0.0f;
			}
			if (sun.angle > 3.14f) {
				world.SKY_DIRECTION.x = light_dir.x;
				world.SKY_DIRECTION.y = light_dir.y;
				world.SKY_DIRECTION.z = light_dir.z;
				world.SKY_ENERGY.r = node->light->tint.r * sun.strength;
				world.SKY_ENERGY.g = node->light->tint.g * sun.strength;
				world.SKY_ENERGY.b = node->light->tint.b * sun.strength;
				// world.SKY_ENERGY.r = 0.0f;
				// world.SKY_ENERGY.g = 0.0f;
				// world.SKY_ENERGY.b = 0.0f;
			}
		}
	}

	if (node->camera != nullptr) {
		auto const& cam = *node->camera;

		mat4 view = to_mat4(glm::inverse(world_glm));

		auto const& persp = std::get<S72::Camera::Perspective>(cam.projection);

		mat4 proj = perspective(
			persp.vfov,
			persp.aspect,
			persp.near,
			persp.far
		);

		camera_indices[cam.name] = uint32_t(camera_view_matrices.size());

		camera_view_matrices.push_back(view);
		camera_proj_matrices.push_back(proj);
		camera_aspects.push_back(persp.aspect);

	}

	for (S72::Node* child : node->children) {
		traverse_node(child, world_glm);
	}

}

void Tutorial::build_scene_objects() {
	object_instances.clear();

	glm::mat4 parent(1.0f);

	for (S72::Node* root: s72.scene.roots) {
		traverse_node(root, parent);
	}

	auto it = camera_indices.find(rtg.configuration.camera_name);
	if (it != camera_indices.end()) {
		current_camera_index = it->second;
	}
	
	lines_vertices.clear();
	for (auto & v : bbox_lines_vertices) {
		lines_vertices.push_back(v);
	}

	debug_camera.radius = 20.0f;
	debug_camera.far = 10000.0f;
}

void Tutorial::update_object_instances_camera() {
	for (auto & obj_inst : object_instances) {
		obj_inst.transform.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * obj_inst.transform.WORLD_FROM_LOCAL;
	}
}

void Tutorial::build_material_texture_indices() {
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
		material_id[materials_list[i]] = i;
	}
	texture_id.clear();
	for (uint32_t i = 0; i < (uint32_t)textures_list.size(); ++i) {
		texture_id[textures_list[i]] = i;
	}

	for (uint32_t i = 0; i < (uint32_t)materials_list.size(); ++i) {
		S72::Material const* mat = materials_list[i];

		material_params.push_back(rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::Material), 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		));

		MaterialTextureInfo info{};

		if (std::holds_alternative<S72::Material::Lambertian>(mat->brdf)) {
			auto const& lam = std::get<S72::Material::Lambertian>(mat->brdf);
			if (std::holds_alternative<S72::color>(lam.albedo)) {
				auto const& c = std::get<S72::color>(lam.albedo);
				
				std::memcpy(material_params[i].allocation.data(), &c, sizeof(c));
			} 
			else {
				auto const* tex = std::get<S72::Texture*>(lam.albedo);
				info.albedo_tex_index = texture_id.at(tex);
				info.has_albedo_tex = 1;

				S72::color c{1.0f, 1.0f, 1.0f};
				std::memcpy(material_params[i].allocation.data(), &c, sizeof(c));
			}
		}

		material_tex_info.push_back(info);

		
	}
}

void Tutorial::load_all_textures() {
	// textures.clear();
    // textures.reserve(textures_list.size());

	stbi_set_flip_vertically_on_load(true);
    
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

Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_) {
	//load .s72 scene:
	try {
		if (!rtg.configuration.scene_file.empty()) {
			s72 = S72::load(rtg.configuration.scene_file);
			print_info(s72);
			print_scene_graph(s72);
			print_mesh_file(s72);
			print_data_file(s72);

			if (!rtg.configuration.camera_name.empty()) {
            if (s72.cameras.count(rtg.configuration.camera_name) == 0) {
                throw std::runtime_error(
                    "Camera '" + rtg.configuration.camera_name + "' not found in scene."
                );
            }
            camera_mode = CameraMode::Scene;
        }

		}
	} catch (std::exception &e) {
		std::cerr << "Failed to load s72-format scene from '" << rtg.configuration.scene_file << "':\n" << e.what() << std::endl;

		std::exit(EXIT_FAILURE);
	}

	//select a depth format:
	//  (at least one of these two must be supported, according to the spec; but neither are required)
	depth_format = rtg.helpers.find_image_format(
		{VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32 },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);

	{ //create render pass
		std::array< VkAttachmentDescription, 2 > attachments{
			VkAttachmentDescription{ //0 - color attachment:
				.format = rtg.surface_format.format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = rtg.present_layout,
			},
			VkAttachmentDescription{ //1 - depth attachment:
				.format = depth_format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			}
		};

		VkAttachmentReference color_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_attachment_ref{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};

		//this defers the image load actions for the attachments:
		std::array< VkSubpassDependency, 2 > dependencies {
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			},
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			}
		};

		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};

		VK( vkCreateRenderPass(rtg.device, &create_info, nullptr, &render_pass) );
	}

	{ //create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK( vkCreateCommandPool(rtg.device, &create_info, nullptr, &command_pool) );
	}

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass, 0);

	{ //create descriptor pool:
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());  //for easier-to-read counting
		std::array< VkDescriptorPoolSize, 2> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2 * per_workspace, //one descriptor per set, two sets per workspace
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1 * per_workspace, //one descriptor per set, one set per workspace
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = 3 * per_workspace, //three sets per workspace
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool) );
	}

	workspaces.resize(rtg.workspaces.size());
	for (Workspace &workspace : workspaces) {
		{ //allocate command buffer:
			VkCommandBufferAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			VK( vkAllocateCommandBuffers(rtg.device, &alloc_info, &workspace.command_buffer) );
		}

		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,  //host-visible memory, coherent (no special sync needed)
			Helpers::Mapped //get a pointer to the memory
		);

		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as a uniform buffer, also going to have GPU copy into this memory
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
			Helpers::Unmapped //don't get a pointer to the memory
		);

		{ //allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lines_pipeline.set0_Camera,
			};

			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Camera_descriptors) );
		}

		workspace.World_src = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};

			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors) );
			//NOTE: will actually fill in this descriptor set just a bit lower
		}

		{ //allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};

			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors) );
			//NOTE: will fill in this descriptor set in render when buffers are [re-]allocated
		}



		{ //point descriptor to Camera buffer:
			VkDescriptorBufferInfo Camera_Info{
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};

			std::array< VkWriteDescriptorSet, 2 > writes {
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Camera_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &Camera_Info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.World_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &World_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device, //device
				uint32_t(writes.size()), //descriptorWriteCount
				writes.data(), //pDescriptorWrites
				0, //descriptorCopyCount
				nullptr //pDescriptorCopies
			);
		}

	}

	build_material_texture_indices();

	{ //create object vertices
		// std::vector< PosNorTexVertex > vertices;
		std::vector< PosNorTanTexVertex > vertices;
		load_mesh_vertices(s72, vertices);
		
		// { //A [-1,1]x[-1,1]x{0} quadrilateral:
		// 	plane_vertices.first = uint32_t(vertices.size());
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 	// 	.TexCoord{ .s = 0.0f, .t = 0.0f },
		// 	// });
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 	// 	.TexCoord{ .s = 1.0f, .t = 0.0f },
		// 	// });
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 	// 	.TexCoord{ .s = 0.0f, .t = 1.0f },
		// 	// });
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 	// 	.TexCoord{ .s = 1.0f, .t = 1.0f },
		// 	// });
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
		// 	// 	.TexCoord{ .s = 0.0f, .t = 1.0f },
		// 	// });
		// 	// vertices.emplace_back(PosNorTexVertex{
		// 	// 	.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
		// 	// 	.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
		// 	// 	.TexCoord{ .s = 1.0f, .t = 0.0f },
		// 	// });


		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 0.0f, .t = 0.0f },
		// 	});
		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 1.0f, .t = 0.0f },
		// 	});
		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 0.0f, .t = 1.0f },
		// 	});
		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 1.0f, .t = 1.0f },
		// 	});
		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 0.0f, .t = 1.0f },
		// 	});
		// 	vertices.emplace_back(PosNorTanTexVertex{
		// 		.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
		// 		.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
		// 		.Tangent{ .x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f },
		// 		.TexCoord{ .s = 1.0f, .t = 0.0f },
		// 	});
			
		// 	plane_vertices.count = uint32_t(vertices.size()) - plane_vertices.first;
		// }
		
		// { //A torus:
		// 	torus_vertices.first = uint32_t(vertices.size());

		// 	//TODO: torus!
		// 	//will parameterize with (u,v) where:
		// 	// - u is angle around main axis (+z)
		// 	// - v is angle around the tube

		// 	constexpr float R1 = 0.75f; //main radius
		// 	constexpr float R2 = 0.15f; //tube radius

		// 	constexpr uint32_t U_STEPS = 20;
		// 	constexpr uint32_t V_STEPS = 16;

		// 	//texture repeats around the torus:
		// 	constexpr float V_REPEATS = 2.0f;
		// 	constexpr float U_REPEATS = int(V_REPEATS / R2 * R1 + 0.999f); //approximately square, rounded up

		// 	auto emplace_vertex = [&](uint32_t ui, uint32_t vi) {
		// 		//convert steps to angles:
		// 		// (doing the mod since trig on 2 M_PI may not exactly match 0)
		// 		float ua = (ui % U_STEPS) / float(U_STEPS) * 2.0f * float(M_PI);
		// 		float va = (vi % V_STEPS) / float(V_STEPS) * 2.0f * float(M_PI);

		// 		// vertices.emplace_back( PosNorTexVertex{
		// 		// 	.Position{
		// 		// 		.x = (R1 + R2 * std::cos(va)) * std::cos(ua),
		// 		// 		.y = (R1 + R2 * std::cos(va)) * std::sin(ua),
		// 		// 		.z = R2 * std::sin(va),
		// 		// 	},
		// 		// 	.Normal{
		// 		// 		.x = std::cos(va) * std::cos(ua),
		// 		// 		.y = std::cos(va) * std::sin(ua),
		// 		// 		.z = std::sin(va),
		// 		// 	},
		// 		// 	.TexCoord{
		// 		// 		.s = ui / float(U_STEPS) * U_REPEATS,
		// 		// 		.t = vi / float(V_STEPS) * V_REPEATS,
		// 		// 	},
		// 		// });

		// 		vertices.emplace_back( PosNorTanTexVertex{
		// 			.Position{
		// 				.x = (R1 + R2 * std::cos(va)) * std::cos(ua),
		// 				.y = (R1 + R2 * std::cos(va)) * std::sin(ua),
		// 				.z = R2 * std::sin(va),
		// 			},
		// 			.Normal{
		// 				.x = std::cos(va) * std::cos(ua),
		// 				.y = std::cos(va) * std::sin(ua),
		// 				.z = std::sin(va),
		// 			},
		// 			.Tangent{
		// 				.x = 0.0f,
		// 				.y = 0.0f,
		// 				.z = 0.0f,
		// 				.w = 0.0f,
		// 			},
		// 			.TexCoord{
		// 				.s = ui / float(U_STEPS) * U_REPEATS,
		// 				.t = vi / float(V_STEPS) * V_REPEATS,
		// 			},
		// 		});
		// 	};

		// 	for (uint32_t ui = 0; ui < U_STEPS; ++ui) {
		// 		for (uint32_t vi = 0; vi < V_STEPS; ++vi) {
		// 			emplace_vertex(ui, vi);
		// 			emplace_vertex(ui+1, vi);
		// 			emplace_vertex(ui, vi+1);

		// 			emplace_vertex(ui, vi+1);
		// 			emplace_vertex(ui+1, vi);
		// 			emplace_vertex(ui+1, vi+1);
		// 		}
		// 	}

		// 	torus_vertices.count = uint32_t(vertices.size()) - torus_vertices.first;
		// }
		

		size_t bytes = vertices.size() * sizeof(vertices[0]);

		object_vertices = rtg.helpers.create_buffer(bytes, 
													VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
													VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
													Helpers::Unmapped
		);

		//copy data to buffer:
		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}

	build_scene_objects();

	{
		auto it = camera_indices.find(rtg.configuration.camera_name);
		if (it != camera_indices.end()) {
			current_camera_index = it->second;
		}
		
		lines_vertices.clear();
		for (auto & v : bbox_lines_vertices) {
			lines_vertices.push_back(v);
		}

		debug_camera.radius = 20.0f;
		debug_camera.far = 10000.0f;
	}

	{ //make some textures
		// textures.reserve(2);

		// { //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
		// 	//actually make the texture:
		// 	uint32_t size = 128;
		// 	std::vector< uint32_t > data;
		// 	data.reserve(size * size);
		// 	for (uint32_t y = 0; y < size; ++y) {
		// 		float fy = (y + 0.5f) / float(size);
		// 		for (uint32_t x = 0; x < size; ++x) {
		// 			float fx = (x + 0.5f) / float(size);
		// 			//highlight the origin:
		// 			if      (fx < 0.05f && fy < 0.05f) data.emplace_back(0xff0000ff); //red
		// 			else if ( (fx < 0.5f) == (fy < 0.5f)) data.emplace_back(0xff444444); //dark grey
		// 			else data.emplace_back(0xffbbbbbb); //light grey
		// 		}
		// 	}
		// 	assert(data.size() == size*size);

		// 	//make a place for the texture to live on the GPU:
		// 	textures.emplace_back(rtg.helpers.create_image(
		// 		VkExtent2D{ .width = size , .height = size }, //size of image
		// 		VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
		// 		VK_IMAGE_TILING_OPTIMAL,
		// 		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
		// 		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
		// 		Helpers::Unmapped
		// 	));

		// 	//transfer data:
		// 	rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		// }

		// { //texture 1 will be a classic 'xor' texture:
		// 	//actually make the texture:
		// 	uint32_t size = 256;
		// 	std::vector< uint32_t > data;
		// 	data.reserve(size * size);
		// 	for (uint32_t y = 0; y < size; ++y) {
		// 		for (uint32_t x = 0; x < size; ++x) {
		// 			uint8_t r = uint8_t(x) ^ uint8_t(y);
		// 			uint8_t g = uint8_t(x + 128) ^ uint8_t(y);
		// 			uint8_t b = uint8_t(x) ^ uint8_t(y + 27);
		// 			uint8_t a = 0xff;
		// 			data.emplace_back( uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24) );
		// 		}
		// 	}
		// 	assert(data.size() == size*size);

		// 	//make a place for the texture to live on the GPU:
		// 	textures.emplace_back(rtg.helpers.create_image(
		// 		VkExtent2D{ .width = size , .height = size }, //size of image
		// 		VK_FORMAT_R8G8B8A8_SRGB, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
		// 		VK_IMAGE_TILING_OPTIMAL,
		// 		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
		// 		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
		// 		Helpers::Unmapped
		// 	));

		// 	//transfer data:
		// 	rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		// }

		load_all_textures();

		{ //the last texture will be a white texture.
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
			white_texture_index = uint32_t(textures.size() - 1);
		}
		
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				// .components sets swizzling and is fine when zero-initialized
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view) );

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ // make a sampler for the textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler) );
	}
		
	{ // create the material descriptor pool
		uint32_t per_material = uint32_t(s72.materials.size()); //for easier-to-read counting

		std::array< VkDescriptorPoolSize, 2> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1 * 1 * per_material, //one descriptor per set, one set per texture
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1 * 1 * per_material, //one descriptor per set, one set per texture
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = 1 * per_material, //one set per texture
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &material_descriptor_pool) );
	}

	{ //allocate and write the material descriptor sets
		
		//allocate the descriptors (using the same alloc_info):
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = material_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_Material,
		};
		material_descriptors.assign(s72.materials.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet &descriptor_set : material_descriptors) {
			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set) );
		}

		std::vector<VkWriteDescriptorSet> writes;
		size_t N = material_descriptors.size();
		std::vector<VkDescriptorBufferInfo> material_params_infos(N);
		std::vector<VkDescriptorImageInfo>  albedo_infos(N);
		writes.reserve(N * 2);

		for (size_t i = 0; i < N; ++i) {
			material_params_infos[i] = VkDescriptorBufferInfo{
				.buffer = material_params[i].handle,
				.offset = 0,
				.range  = material_params[i].size,
			};

			writes.push_back(VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptors[i],
					.dstBinding = 0, // MaterialParams
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &material_params_infos[i],
			});

			uint32_t tex_index = material_tex_info[i].has_albedo_tex ? material_tex_info[i].albedo_tex_index : white_texture_index;

			

			albedo_infos[i] = VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			writes.push_back(VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptors[i],
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &albedo_infos[i],
			});
		}

		//write descriptors for textures:
		// std::vector< VkDescriptorImageInfo > infos(textures.size());
		// std::vector< VkWriteDescriptorSet > writes(textures.size());

		// for (Helpers::AllocatedImage const &image : textures) {
		// 	size_t i = &image - &textures[0];

		// 	infos[i] = VkDescriptorImageInfo{
		// 		.sampler = texture_sampler,
		// 		.imageView = texture_views[i],
		// 		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		// 	};
		// 	writes[i] = VkWriteDescriptorSet{
		// 		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		// 		.dstSet = material_descriptors[i],
		// 		.dstBinding = 1,
		// 		.dstArrayElement = 0,
		// 		.descriptorCount = 1,
		// 		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		// 		.pImageInfo = &infos[i],
		// 	};
		// }

		vkUpdateDescriptorSets( rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr );
	}

}

Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

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

	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces) {
		if (workspace.command_buffer != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(rtg.device, command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}

		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}

		if (workspace.Camera_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}
		//Camera_descriptors freed when pool is destroyed.

		if (workspace.World_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		//World_descriptors freed when pool is destroyed.

		if (workspace.Transforms_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		//Transforms_descriptors freed when pool is destroyed.
	}
	workspaces.clear();

	if (descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
		//(this also frees the descriptor sets allocated from the pool)
	}

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);

	if (command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain) {
	//clean up existing framebuffers (and depth image):
	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	//Allocate depth image for framebuffers to share:
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent,
		depth_format,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);

	{ //create depth image view:
		VkImageViewCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_depth_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
		};

		VK( vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view) );
	}

	//Make framebuffers for each swapchain image:
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i) {
		std::array< VkImageView, 2 > attachments{
			swapchain.image_views[i],
			swapchain_depth_image_view,
		};
		VkFramebufferCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = render_pass,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.width = swapchain.extent.width,
			.height = swapchain.extent.height,
			.layers = 1,
		};

		VK( vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]) );
	}
}

void Tutorial::destroy_framebuffers() {
	for (VkFramebuffer &framebuffer : swapchain_framebuffers) {
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}


void Tutorial::render(RTG &rtg_, RTG::RenderParams const &render_params) {
	//assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	//get more convenient names for the current workspace and target framebuffer:
	Workspace &workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	
	//reset the command buffer (clear old commands);
	VK( vkResetCommandBuffer(workspace.command_buffer, 0) );
	{ //begin recording:
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			//.pNext set to nullptr by zero-initialization!
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, //will record again every submit
		};
		VK( vkBeginCommandBuffer(workspace.command_buffer, &begin_info) );
	}

	if (!lines_vertices.empty()) { ////upload lines vertices:
		//[re-]allocate lines buffers if needed:
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE || workspace.lines_vertices_src.size < needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;

			if (workspace.lines_vertices_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}
			workspace.lines_vertices_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory, coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as vertex buffer, also going to have GPU into this memory
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			std::cout << "Re-allocated lines buffers to " << new_bytes << " bytes." << std::endl;
		}

		assert(workspace.lines_vertices_src.size == workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);

		//host-side copy into lines_vertices_src:
		assert(workspace.lines_vertices_src.allocation.mapped);
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		//device-side copy from lines_vertices_src -> lines_vertices:
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.handle, 1, &copy_region);

		{ //upload camera info:
			LinesPipeline::Camera camera{
				.CLIP_FROM_WORLD = CLIP_FROM_WORLD
			};
			assert(workspace.Camera_src.size == sizeof(camera));

			//host-side copy into Camera_src:
			memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

			//add device-side copy from Camera_src -> Camera:
			assert(workspace.Camera_src.size == workspace.Camera.size);
			VkBufferCopy copy_region{
				.srcOffset = 0,
				.dstOffset = 0,
				.size = workspace.Camera_src.size,
			};
			vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle, workspace.Camera.handle, 1, &copy_region);
		}

		{ //upload world info:
			// assert(workspace.Camera_src.size == sizeof(world));
			assert(workspace.World_src.size == sizeof(world));

			//host-side copy into World_src:
			memcpy(workspace.World_src.allocation.data(), &world, sizeof(world));

			//add device-side copy from World_src -> World:
			assert(workspace.World_src.size == workspace.World.size);
			VkBufferCopy copy_region{
				.srcOffset = 0,
				.dstOffset = 0,
				.size = workspace.World_src.size,
			};
			vkCmdCopyBuffer(workspace.command_buffer, workspace.World_src.handle, workspace.World.handle, 1, &copy_region);
		}
	}

	if (!object_instances.empty()) { //upload object transforms:
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size < needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.Transforms_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}
			workspace.Transforms_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory, coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.Transforms = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as storage buffer, also going to have GPU into this memory
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			//update the descriptor set:
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};

			std::array< VkWriteDescriptorSet, 1 > writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Transforms_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Transforms_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()), writes.data(), //descriptorWrites count, data
				0, nullptr //descriptorCopies count, data
			);

			std::cout << "Re-allocated object transforms buffers to " << new_bytes << " bytes." << std::endl;
		}

		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);

		{ //copy transforms into Transforms_src:
			assert(workspace.Transforms_src.allocation.mapped);
			ObjectsPipeline::Transform *out = reinterpret_cast< ObjectsPipeline::Transform * >(workspace.Transforms_src.allocation.data()); // Strict aliasing violation, but it doesn't matter
			for (ObjectInstance const &inst : object_instances) {
				*out = inst.transform;
				++out;
			}
		}

		//device-side copy from Transforms_src -> Transforms:
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle, workspace.Transforms.handle, 1, &copy_region);
	}


	{ //memory barrier to make sure copies complete before rendering happens:
		VkMemoryBarrier memory_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		};

		vkCmdPipelineBarrier( workspace.command_buffer, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, //srcStageMask
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, //dstStageMask
			0, //dependencyFlags
			1, &memory_barrier, //memoryBarriers (count, data)
			0, nullptr, //bufferMemoryBarriers (count, data)
			0, nullptr //imageMemoryBarriers (count, data)
		);
	}

	//TODO: put GPU commands here!
	{ //render pass
		std::array< VkClearValue, 2 > clear_values{
			VkClearValue{ .color{ .float32{0.0f, 0.0f, 0.0f, 1.0f} } },
			VkClearValue{ .depthStencil{ .depth = 1.0f, .stencil = 0 } },
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			},
			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};

		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		uint32_t fb_w = rtg.swapchain_extent.width;
		uint32_t fb_h = rtg.swapchain_extent.height;
		float fb_aspect = fb_w / float(fb_h);

		float target_aspect = fb_aspect;
		bool constrain_to_camera_aspect = (camera_mode == CameraMode::Scene) && (!debug_camera_mode);
		if (constrain_to_camera_aspect && current_camera_index < camera_aspects.size()) {
			target_aspect = camera_aspects[current_camera_index];
		}

		target_aspect = camera_aspects[current_camera_index];

		uint32_t vp_x = 0, vp_y = 0, vp_w = fb_w, vp_h = fb_h;

		if (constrain_to_camera_aspect && target_aspect > 0.0f) {
			

			if (fb_aspect > target_aspect) {
				// window too wide -> pillarbox
				vp_h = fb_h;
				vp_w = uint32_t(std::round(float(vp_h) * target_aspect));
				vp_x = (fb_w - vp_w) / 2;
				vp_y = 0;

			} else if (fb_aspect < target_aspect) {
				// window too tall/narrow -> letterbox
				vp_w = fb_w;
				vp_h = uint32_t(std::round(float(vp_w) / target_aspect));
				vp_x = 0;
				vp_y = (fb_h - vp_h) / 2;

			}
		}

		
		//TODO: run pipelines here
		{ //set scissor rectangle:
			// VkRect2D scissor{
			// 	.offset = {.x = 0, .y = 0},
			// 	.extent = rtg.swapchain_extent,
			// };
			VkRect2D scissor{
				.offset = { .x = int32_t(vp_x), .y = int32_t(vp_y) },
				.extent = { .width = vp_w, .height = vp_h },
			};
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &scissor);
		}
		{ //configure viewport transform:
			// VkViewport viewport{
			// 	.x = 0.0f,
			// 	.y = 0.0f,
			// 	.width = float(rtg.swapchain_extent.width),
			// 	.height = float(rtg.swapchain_extent.height),
			// 	.minDepth = 0.0f,
			// 	.maxDepth = 1.0f,
			// };
			VkViewport viewport{
				.x = float(vp_x),
				.y = float(vp_y),
				.width = float(vp_w),
				.height = float(vp_h),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &viewport);
		}

		// { //draw with the background pipeline:
		// 	vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background_pipeline.handle);

		// 	{ //push time:
		// 		BackgroundPipeline::Push push{
		// 			.time = time,
		// 		};
		// 		vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 
		// 							0, sizeof(push), &push);
		// 	}

		// 	vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		// }

		if (debug_camera_mode){ //draw with the lines pipeline:
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);

			{ //use lines_vertices (offset 0) as vertex buffer binding 0:
				std::array< VkBuffer, 1 > vertex_buffers{ workspace.lines_vertices.handle};
				std::array< VkDeviceSize, 1 > offsets{ 0 };
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}

			{ //bind Camera descriptor set:
				std::array< VkDescriptorSet, 1 > descriptor_sets {
					workspace.Camera_descriptors, //0: Camera
				};

				vkCmdBindDescriptorSets(
					workspace.command_buffer, //command buffer
					VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
					lines_pipeline.layout, //pipeline layout
					0, //first set
					uint32_t(descriptor_sets.size()), descriptor_sets.data(),  //descriptor sets count, ptr
					0, nullptr //dynamic offsets count, ptr
				);
			}

			//draw lines vertices:
			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
		}

		if (!object_instances.empty()) { //draw with the objects pipeline:
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.handle);

			{ //use object_vertices (offset 0) as vertex buffer binding 0:
				std::array< VkBuffer, 1 > vertex_buffers{ object_vertices.handle };
				std::array< VkDeviceSize, 1 > offsets{ 0 };
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}

			{ //bind World and Transforms descriptor sets:
				std::array< VkDescriptorSet, 2 > descriptor_sets{
					workspace.World_descriptors, //0: World
					workspace.Transforms_descriptors, //1: Transforms
				};
				vkCmdBindDescriptorSets(
					workspace.command_buffer, //command buffer
					VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
					objects_pipeline.layout, //pipeline layout
					0, //first set
					uint32_t(descriptor_sets.size()), descriptor_sets.data(), //descriptor sets count, ptr
					0, nullptr //dynamic offsets count, ptr
				);
			}			

			//Camera descriptor set is still bound, but unused(!)

			//draw all instances:
			for (ObjectInstance const &inst : object_instances) {
				uint32_t index = uint32_t(&inst - &object_instances[0]);

				//bind texture descriptor set:
				vkCmdBindDescriptorSets(
					workspace.command_buffer, //command buffer
					VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
					objects_pipeline.layout, //pipeline layout
					2, //second set
					1, &material_descriptors[inst.material], //descriptor sets count, ptr
					0, nullptr //dynamic offsets count, ptr
				);

				vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
			}			
			// vkCmdDraw(workspace.command_buffer, uint32_t(object_vertices.size / sizeof(ObjectsPipeline::Vertex)), 1, 0, 0);
			// vkCmdDraw(workspace.command_buffer, torus_vertices.count, 1, torus_vertices.first, 0);
			// vkCmdDraw(workspace.command_buffer, plane_vertices.count, 1, plane_vertices.first, 0);

		}
		vkCmdEndRenderPass(workspace.command_buffer);
	}

	//end recording:
	VK( vkEndCommandBuffer(workspace.command_buffer) );

	
	{ //submit `workspace.command buffer` for the GPU to run:
		std::array< VkSemaphore, 1 > wait_semaphores{
			render_params.image_available
		};
		std::array< VkPipelineStageFlags, 1 > wait_stages{
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		};
		static_assert(wait_semaphores.size() == wait_stages.size(), "every semaphore needs a stage");

		std::array< VkSemaphore, 1 > signal_semaphores{
			render_params.image_done
		};
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = uint32_t(wait_semaphores.size()),
			.pWaitSemaphores = wait_semaphores.data(),
			.pWaitDstStageMask = wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &workspace.command_buffer,
			.signalSemaphoreCount = uint32_t(signal_semaphores.size()),
			.pSignalSemaphores = signal_semaphores.data(),
		};

		VK( vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, render_params.workspace_available) );
	}
}


void Tutorial::update(float dt) {
	time += dt;

	if (camera_mode == CameraMode::Scene) { 
		//camera rotating around the origin:
		// float ang = float(M_PI) * 2.0f * 10.0f * (time / 60.0f);
		
		// CLIP_FROM_WORLD = perspective(
		// 	60.0f * float(M_PI) / 180.0f, //vfov
		// 	rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), //aspect
		// 	0.1f, //near
		// 	1000.0f //far
		// ) * look_at(
		// 	3.0f * std::cos(ang), 3.0f * std::sin(ang), 1.0f, //eye
		// 	0.0f, 0.0f, 0.5f, //target
		// 	0.0f, 0.0f, 1.0f //up
		// );

		CLIP_FROM_WORLD = camera_proj_matrices[current_camera_index] * camera_view_matrices[current_camera_index];
		
	} else if (camera_mode == CameraMode::Free) {
		CLIP_FROM_WORLD = perspective(
			free_camera.fov,
			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), //aspect
			free_camera.near,
			free_camera.far
		) * orbit(
			free_camera.target_x, free_camera.target_y, free_camera.target_z,
			free_camera.azimuth, free_camera.elevation, free_camera.radius
		);
	} else {
		assert(0 && "only two camera modes");
	} 

	if (debug_camera_mode) {
		frustum_lines_vertices.clear();
		append_frustum_lines_world(to_glm(CLIP_FROM_WORLD), 255, 255, 0, 255);

		CLIP_FROM_WORLD = perspective(
			debug_camera.fov,
			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), //aspect
			debug_camera.near,
			debug_camera.far
		) * orbit(
			debug_camera.target_x, debug_camera.target_y, debug_camera.target_z,
			debug_camera.azimuth, debug_camera.elevation, debug_camera.radius
		);

		lines_vertices.clear();
		for (auto & v : bbox_lines_vertices) {
			lines_vertices.push_back(v);
		}

		for (auto & v : frustum_lines_vertices) {
			lines_vertices.push_back(v);
		}
	}

	{ //static sun and sky:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.1f;
		world.SKY_ENERGY.g = 0.1f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = 6.0f / 23.0f;
		world.SUN_DIRECTION.y = 13.0f / 23.0f;
		world.SUN_DIRECTION.z = 18.0f / 23.0f;

		world.SUN_ENERGY.r = 1.0f;
		world.SUN_ENERGY.g = 1.0f;
		world.SUN_ENERGY.b = 0.9f;
	}

	

	// { //Time of Day Effect:

	// 	float t = time * 1.0f;

	// 	float sx = std::cos(t);
	// 	float sy = 0.0f;
	// 	float sz = std::sin(t);

	// 	//normalize
	// 	float len = std::sqrt(sx*sx + sz*sz);
	// 	if (len < 1e-6f) len = 1.0f;
	// 	sx /= len; sz /= len;

	// 	world.SUN_DIRECTION = { sx, sy, sz, 0.0f };

	// 	float day = sz;
	// 	if (day < 0.0f) day = 0.0f;
	// 	if (day > 1.0f) day = 1.0f;

	// 	//sunset factor peaks near horizon when sun is still above it:
	// 	float sunset = 1.0f - std::fabs(sz); //0 at noon/midnight, 1 near horizon
	// 	if (sunset < 0.0f) sunset = 0.0f;
	// 	if (sunset > 1.0f) sunset = 1.0f;

	// 	//only apply sunset warmth when sun is above horizon:
	// 	float sunset_warm = sunset * day;

	// 	//Base sun brightness:
	// 	float sunI = day * day;
	// 	world.SUN_ENERGY = {
	// 		(4.2f * sunI + 2.2f * sunset_warm),
	// 		(2.4f * sunI + 0.9f * sunset_warm),
	// 		(1.2f * sunI + 0.1f * sunset_warm),
	// 		0.0f
	// 	};

	// 	world.SKY_DIRECTION = { 0.0f, 0.0f, 1.0f, 0.0f };

	// 	//Want this when sz is slightly negative, e.g. [-0.25, 0]
	// 	//Map sz in [-0.25, 0] -> blueHour in [0, 1]
	// 	float blueHour = (sz + 0.25f) / 0.25f;
	// 	if (blueHour < 0.0f) blueHour = 0.0f;
	// 	if (blueHour > 1.0f) blueHour = 1.0f;

	// 	//Also include dawn/dusk when sun is slightly above horizon:
	// 	//Make it strongest near horizon regardless of sign, but bias to below-horizon:
	// 	float nearHorizon = 1.0f - std::fabs(sz);
	// 	if (nearHorizon < 0.0f) nearHorizon = 0.0f;
	// 	if (nearHorizon > 1.0f) nearHorizon = 1.0f;

	// 	float blueTone = 0.7f * blueHour + 0.3f * nearHorizon;

	// 	float nightR = 0.00f, nightG = 0.01f, nightB = 0.03f;
	// 	float dayR = 0.03f, dayG = 0.08f, dayB = 0.28f;
	// 	float blueR = 0.00f, blueG = 0.06f, blueB = 0.40f;

	// 	float skyR = nightR + dayR * day + blueR * blueTone;
	// 	float skyG = nightG + dayG * day + blueG * blueTone;
	// 	float skyB = nightB + dayB * day + blueB * blueTone;

	// 	world.SKY_ENERGY = { skyR, skyG, skyB, 0.0f };
	// }



	// { //engine
	// 	float s = 1.7f;

	// 	mat4 SCALE = mat4{
	// 		s, 0, 0, 0,
	// 		0, s, 0, 0,
	// 		0, 0, s, 0,
	// 		0, 0, 0, 1
	// 	};

	// 	float z0 = 0.6f;

	// 	CLIP_FROM_WORLD =
	// 		perspective(
	// 			60.0f * float(M_PI) / 180.0f,
	// 			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
	// 			0.1f,
	// 			1000.0f
	// 		) * look_at(
	// 			0.0f, 0.0f, z0 + 4.0f,
	// 			0.0f, 0.0f, z0,
	// 			0.0f, 1.0f, 0.0f
	// 		) * SCALE;
	// }

	// //make an 'x':
	// lines_vertices.clear();
	// lines_vertices.reserve(4);
	// lines_vertices.emplace_back(PosColVertex{
	// 	.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f},
	// 	.Color{ .r = 0xff, .g = 0xff, .b = 0xff, .a = 0xff}
	// });
	// lines_vertices.emplace_back(PosColVertex{
	// 	.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f},
	// 	.Color{ .r = 0xff, .g = 0x00, .b = 0x00, .a = 0xff}
	// });
	// lines_vertices.emplace_back(PosColVertex{
	// 	.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
	// 	.Color{ .r = 0x00, .g = 0x00, .b = 0xff, .a = 0xff }
	// });
	// lines_vertices.emplace_back(PosColVertex{
	// 	.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
	// 	.Color{ .r = 0x00, .g = 0x00, .b = 0xff, .a = 0xff }
	// });
	// assert(lines_vertices.size() == 4);

	// { //make some crossing lines at different depths:
	// 	lines_vertices.clear();
	// 	constexpr size_t count = 2 * 30 + 2 * 30;
	// 	lines_vertices.reserve(count);
	// 	//horizontal lines at z = 0.5f:
	// 	for (uint32_t i = 0; i < 30; ++i) {
	// 		float y = (i + 0.5f) / 30.0f * 2.0f - 1.0f;
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{.x = -1.0f, .y = y, .z = 0.5f},
	// 			.Color{ .r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff},
	// 		});
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{.x = 1.0f, .y = y, .z = 0.5f},
	// 			.Color{ .r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff},
	// 		});
	// 	}
	// 	//vertical lines at z = 0.0f (near) through 1.0f (far):
	// 	for (uint32_t i = 0; i < 30; ++i) {
	// 		float x = (i + 0.5f) / 30.0f * 2.0f - 1.0f;
	// 		float z = (i + 0.5f) / 30.0f;
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{.x = x, .y =-1.0f, .z = z},
	// 			.Color{ .r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff},
	// 		});
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{.x = x, .y = 1.0f, .z = z},
	// 			.Color{ .r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff},
	// 		});
	// 	}

	// 	assert(lines_vertices.size() == count);
	// }
	
	update_object_instances_camera();
	// { //make some objects:
	// 	object_instances.clear();

	// 	{ //plane translated +x by one unit:
	// 		mat4 WORLD_FROM_LOCAL{
	// 			1.0f, 0.0f, 0.0f, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, 1.0f, 0.0f,
	// 			1.0f, 0.0f, 0.0f, 1.0f,
	// 		};

	// 		object_instances.emplace_back(ObjectInstance{
	// 			.vertices = plane_vertices,
	// 			.transform{
	// 				.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
	// 				.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
	// 				.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
	// 			},
	// 			.texture = 1,
	// 		});
	// 	}

	// 	{ //torus translated -x by one unit and rotated CCW around +y:
	// 		// float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
	// 		float ang = 2.0f * float(M_PI) * 10.0f;
	// 		float ca = std::cos(ang);
	// 		float sa = std::sin(ang);
	// 		mat4 WORLD_FROM_LOCAL{
	// 			  ca, 0.0f,  -sa, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			  sa, 0.0f,   ca, 0.0f,
	// 			-1.0f,0.0f, 0.0f, 1.0f,
	// 		};

	// 		object_instances.emplace_back(ObjectInstance{
	// 			.vertices = torus_vertices,
	// 			.transform{
	// 				.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
	// 				.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
	// 				.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
	// 			},
	// 		});
	// 	}
	// }

	// {//jet engine front view (intake ring + spinner + curved blades)
	// 	lines_vertices.clear();

	// 	auto add_line = [&](float ax, float ay, float az,
	// 						float bx, float by, float bz,
	// 						uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff) {
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{ .x = ax, .y = ay, .z = az },
	// 			.Color{ .r = r, .g = g, .b = b, .a = a }
	// 		});
	// 		lines_vertices.emplace_back(PosColVertex{
	// 			.Position{ .x = bx, .y = by, .z = bz },
	// 			.Color{ .r = r, .g = g, .b = b, .a = a }
	// 		});
	// 	};

	// 	auto add_ring = [&](float z, float radius, int segments, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff) {
	// 		for (int i = 0; i < segments; ++i) {
	// 			float t0 = float(i) / float(segments) * 2.0f * float(M_PI);
	// 			float t1 = float(i + 1) / float(segments) * 2.0f * float(M_PI);
	// 			float x0 = radius * std::cos(t0), y0 = radius * std::sin(t0);
	// 			float x1 = radius * std::cos(t1), y1 = radius * std::sin(t1);
	// 			add_line(x0, y0, z, x1, y1, z, r, g, b, a);
	// 		}
	// 	};

	// 	auto add_arc = [&](float z, float radius, float a0, float a1, int steps,
	// 					uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff) {
	// 		float prevx = radius * std::cos(a0);
	// 		float prevy = radius * std::sin(a0);
	// 		for (int i = 1; i <= steps; ++i) {
	// 			float t = float(i) / float(steps);
	// 			float ang = a0 + (a1 - a0) * t;
	// 			float x = radius * std::cos(ang);
	// 			float y = radius * std::sin(ang);
	// 			add_line(prevx, prevy, z, x, y, z, r, g, b, a);
	// 			prevx = x; prevy = y;
	// 		}
	// 	};

	// 	const float z0 = 0.6f;

	// 	//Dimensions (tweak these):
	// 	const float R_outer = 1.05f; //nacelle / intake lip
	// 	const float R_inner = 0.98f; //inner intake ring
	// 	const float R_fan   = 0.85f; //blade tip radius
	// 	const float R_hub   = 0.28f; //spinner / hub radius
	// 	const float R_root  = 0.40f; //blade root radius

	// 	//Rotation:
	// 	float angular_accel = 0.1f;
	// 	float max_omega     = 0.2f;

	// 	float omega = angular_accel * time;
	// 	omega = std::min(omega, max_omega);

	// 	static float spin = 0.0f;
	// 	spin += omega * dt;
	// 	// float spin = time * 10.0f; // rad/sec (slow-ish like a display)

	// 	//Draw intake rings (outer bright, inner darker):
	// 	add_ring(z0, R_outer, 160, 0xcc, 0xcc, 0xcc); //silver-ish
	// 	add_ring(z0, R_inner, 160, 0x66, 0x88, 0xaa); //cooler inner ring

	// 	//Draw fan boundary ring (where blade tips sit):
	// 	add_ring(z0, R_fan, 140, 0x22, 0x22, 0x22);

	// 	//Draw hub/spinner rings:
	// 	add_ring(z0, R_hub, 120, 0xee, 0xee, 0xee);
	// 	add_ring(z0, R_hub * 0.55f, 90, 0xff, 0xff, 0xff);

	// 	//Draw a simple "spiral" mark on spinner (like the logo swirl):
	// 	{ 
	// 		const int steps = 120;
	// 		float prevx = 0.0f, prevy = 0.0f;
	// 		for (int i = 0; i <= steps; ++i) {
	// 			float t = float(i) / float(steps);
	// 			float ang = spin + t * 5.0f * float(M_PI);
	// 			float rad = (R_hub * 0.50f) * t;
	// 			float x = rad * std::cos(ang);
	// 			float y = rad * std::sin(ang);
	// 			if (i > 0) add_line(prevx, prevy, z0, x, y, z0, 0xff, 0xff, 0xff);
	// 			prevx = x; prevy = y;
	// 		}
	// 	}

	// 	//Fan blades (curved / swept arcs + root connector)
	// 	//Fake a “curved blade” using:
	// 	//- a small arc near the root
	// 	//- a larger arc near the tip, with a sweep angle offset
	// 	//- connect root->tip edges
	// 	const int blades = 22;
	// 	lines_vertices.reserve(lines_vertices.size() + size_t(blades) * 60);

	// 	for (int i = 0; i < blades; ++i) {
	// 		float base = spin + float(i) / float(blades) * 2.0f * float(M_PI);

	// 		//Blade angular span and sweep (tweak for more/less curvature):
	// 		float width = 0.09f;              //how wide each blade is in angle
	// 		float sweep = 0.28f;              //tip swept forward
	// 		float a_root0 = base - width;
	// 		float a_root1 = base + width;
	// 		float a_tip0  = base - width + sweep;
	// 		float a_tip1  = base + width + sweep;

	// 		//Root arc segment (small, darker):
	// 		add_arc(z0, R_root, a_root0, a_root1, 8, 0x88, 0x88, 0x88);

	// 		//Tip arc segment (bright):
	// 		add_arc(z0, R_fan, a_tip0, a_tip1, 10, 0xdd, 0xdd, 0xdd);

	// 		//Connect root edges to tip edges (leading/trailing edges):
	// 		float rx0 = R_root * std::cos(a_root0), ry0 = R_root * std::sin(a_root0);
	// 		float rx1 = R_root * std::cos(a_root1), ry1 = R_root * std::sin(a_root1);
	// 		float tx0 = R_fan  * std::cos(a_tip0),  ty0 = R_fan  * std::sin(a_tip0);
	// 		float tx1 = R_fan  * std::cos(a_tip1),  ty1 = R_fan  * std::sin(a_tip1);

	// 		add_line(rx0, ry0, z0, tx0, ty0, z0, 0xbb, 0xbb, 0xbb); //one side
	// 		add_line(rx1, ry1, z0, tx1, ty1, z0, 0xbb, 0xbb, 0xbb); //other side

	// 		//Root to hub connector (gives “blade attaches to spinner” feel):
	// 		float hx = R_hub * std::cos(base), hy = R_hub * std::sin(base);
	// 		float mx = R_root * std::cos(base), my = R_root * std::sin(base);
	// 		add_line(hx, hy, z0, mx, my, z0, 0x66, 0x66, 0x66);
	// 	}

	// 	//Optional: inner dark “engine core” ring to add depth:
	// 	add_ring(z0, 0.12f, 60, 0x11, 0x11, 0x11);
	// }

}


void Tutorial::on_input(InputEvent const &evt) {
	//if there is a current action, it gets input priority:
	if (action) {
		action(evt);
		return;
	}

	//general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB) {
		//switch camera modes
		camera_mode = CameraMode((int(camera_mode) + 1) % 2);
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_D) {
		//switch camera modes
		debug_camera_mode = !debug_camera_mode;
		return;
	}

	if (camera_mode == CameraMode::Scene &&
		evt.type == InputEvent::KeyDown) {

		if (evt.key.key == GLFW_KEY_LEFT_BRACKET) {	// [
			if (!camera_indices.empty()) {
				current_camera_index =
					(current_camera_index + camera_indices.size() - 1)
					% camera_indices.size();
			}
			return;
		}

		if (evt.key.key == GLFW_KEY_RIGHT_BRACKET) {  // ]
			if (!camera_indices.empty()) {
				current_camera_index =
					(current_camera_index + 1)
					% camera_indices.size();
			}
			return;
		}
	}

	//debug camera controls:
	if (debug_camera_mode) {

		if (evt.type == InputEvent::MouseWheel) {
			//change distance by 10% every scroll click:
			debug_camera.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			//make sure camera isn't too close or too far from target:
			debug_camera.radius = std::max(debug_camera.radius, 0.5f * debug_camera.near);
			debug_camera.radius = std::min(debug_camera.radius, 2.0f * debug_camera.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT)) {
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = debug_camera;

			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//image height at plane of target point:
					float height = 2.0f * std::tan(debug_camera.fov * 0.5f) * debug_camera.radius;

					//motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height * height; //note: negated because glfw uses y-down coordinate system

					//compute camera transform to extract right (first row) and up (second row):
					mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius
					);

					//move the desired distance:
					debug_camera.target_x = init_camera.target_x - dx * camera_from_world[0] - dy * camera_from_world[1];
					debug_camera.target_y = init_camera.target_y - dx * camera_from_world[4] - dy * camera_from_world[5];
					debug_camera.target_z = init_camera.target_z - dx * camera_from_world[8] - dy * camera_from_world[9];

					return;
				}
			};

			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
			//start tumbling

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = debug_camera;
			
			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;

					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//motion, normalized so 1.0 is window height:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height; //note: negated because glfw uses y-down coordinate system

					//rotate camera based on motion:
					float speed = float(M_PI); //how much rotation happens at one full window height
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);
					debug_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					debug_camera.elevation = init_camera.elevation - dy * speed;

					//reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					debug_camera.azimuth -= std::round(debug_camera.azimuth / twopi) * twopi;
					debug_camera.elevation -= std::round(debug_camera.elevation / twopi) * twopi;
					return;
				}
			};

			return;
		}

	}


	//free camera controls:
	if (camera_mode == CameraMode::Free) {

		if (evt.type == InputEvent::MouseWheel) {
			//change distance by 10% every scroll click:
			free_camera.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			//make sure camera isn't too close or too far from target:
			free_camera.radius = std::max(free_camera.radius, 0.5f * free_camera.near);
			free_camera.radius = std::min(free_camera.radius, 2.0f * free_camera.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT)) {
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//image height at plane of target point:
					float height = 2.0f * std::tan(free_camera.fov * 0.5f) * free_camera.radius;

					//motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height * height; //note: negated because glfw uses y-down coordinate system

					//compute camera transform to extract right (first row) and up (second row):
					mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius
					);

					//move the desired distance:
					free_camera.target_x = init_camera.target_x - dx * camera_from_world[0] - dy * camera_from_world[1];
					free_camera.target_y = init_camera.target_y - dx * camera_from_world[4] - dy * camera_from_world[5];
					free_camera.target_z = init_camera.target_z - dx * camera_from_world[8] - dy * camera_from_world[9];

					return;
				}
			};

			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
			//start tumbling

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;
			
			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;

					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//motion, normalized so 1.0 is window height:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height; //note: negated because glfw uses y-down coordinate system

					//rotate camera based on motion:
					float speed = float(M_PI); //how much rotation happens at one full window height
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					//reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					free_camera.azimuth -= std::round(free_camera.azimuth / twopi) * twopi;
					free_camera.elevation -= std::round(free_camera.elevation / twopi) * twopi;
					return;
				}
			};

			return;
		}

	}

	
}
