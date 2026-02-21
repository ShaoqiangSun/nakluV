#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "SceneViewer.hpp"


static inline glm::mat4 to_glm(mat4 const& m) {
    glm::mat4 out;
    static_assert(sizeof(out) == sizeof(mat4), "sizes must match");
    std::memcpy(&out[0][0], m.data(), sizeof(mat4));
    return out;
}


static S72 build_cpu_bottleneck_scene(std::string const& base_s72_file, std::string const& mesh_name, uint32_t N_culled) {
	S72 s72 = S72::load(base_s72_file);

	S72::Mesh* shared_mesh = &s72.meshes.at(mesh_name);

    auto make_culled_name = [](uint32_t i){
        return "C_" + std::to_string(i);
    };

	for (uint32_t i = 0; i < N_culled; ++i) {
        S72::Node n;
        n.name = make_culled_name(i);
        n.mesh = shared_mesh;

		n.translation = S72::vec3(+10.0f + float(i) * 0.1f, 0.0f, -8.0f);

        n.rotation = S72::quat(1.0f, 0.0f, 0.0f, 0.0f);
        n.scale = S72::vec3(1.0f);

        auto [it, ok] = s72.nodes.emplace(n.name, std::move(n));
        (void)ok;

        s72.scene.roots.push_back(&it->second);
    }

	return s72;
}


Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_), material_system(rtg_) {


	scene_viewer = std::make_unique<SceneViewer>(material_system);
	
	try {
		if (!rtg.configuration.test_mode.empty()) {
			
			if (rtg.configuration.test_mode == "cpu") {
				uint32_t N_culled = std::stoul(rtg.configuration.cpu_test_culled_count);
				scene_viewer->s72 = build_cpu_bottleneck_scene("scenes/cpu-bottleneck.s72", "Rounded-Cube" ,N_culled);
				
			}
			else {
				if (!rtg.configuration.scene_file.empty()) scene_viewer->s72 = S72::load(rtg.configuration.scene_file);
			}

			material_system.build_material_texture(scene_viewer->s72);
			material_system.load_all_textures();
			
			if (!rtg.configuration.camera_name.empty()) {
				if (scene_viewer->s72.cameras.count(rtg.configuration.camera_name) == 0) {
					throw std::runtime_error(
						"Camera '" + rtg.configuration.camera_name + "' not found in scene."
					);
				}
				camera_mode = CameraMode::Scene;
				
        	}

			if (!rtg.configuration.culling_mode.empty()) {
				if (rtg.configuration.culling_mode == "none") culling_mode = CullingMode::None;
				else if (rtg.configuration.culling_mode == "frustum") culling_mode = CullingMode::Frustum;
				else if (rtg.configuration.culling_mode == "bvh") culling_mode = CullingMode::BVH;
				else throw std::runtime_error(
					"Culling mode '" + rtg.configuration.culling_mode + "' is incorrect."
				);
			}

			std::string csv_file_name = rtg.configuration.csv_file_name.empty() ? "perf.csv" : rtg.configuration.csv_file_name;

			perf_log.open(csv_file_name);

			if (!perf_log) {
				throw std::runtime_error("Failed to open perf.csv");
			}

			perf_log << "frame,instances,visible,cpu_cull_ms,cpu_frame_ms,cpu_wait_gpu_ms,gpu_draw_ms\n";
		}
		else if (!rtg.configuration.scene_file.empty()) {
			scene_viewer->s72 = S72::load(rtg.configuration.scene_file);
			
			material_system.build_material_texture(scene_viewer->s72);
			material_system.load_all_textures();

			if (!rtg.configuration.camera_name.empty()) {
				if (scene_viewer->s72.cameras.count(rtg.configuration.camera_name) == 0) {
					throw std::runtime_error(
						"Camera '" + rtg.configuration.camera_name + "' not found in scene."
					);
				}
				camera_mode = CameraMode::Scene;
				
        	}

			if (!rtg.configuration.culling_mode.empty()) {
				if (rtg.configuration.culling_mode == "none") culling_mode = CullingMode::None;
				else if (rtg.configuration.culling_mode == "frustum") culling_mode = CullingMode::Frustum;
				else throw std::runtime_error(
					"Culling mode '" + rtg.configuration.culling_mode + "' is incorrect."
				);
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

	{//query
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(rtg.physical_device, &props);
		timestamp_period = props.limits.timestampPeriod;

		VkQueryPoolCreateInfo qpci{
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = VK_QUERY_TYPE_TIMESTAMP,
			.queryCount = 2,
		};
		VK( vkCreateQueryPool(rtg.device, &qpci, nullptr, &timestamp_query_pool) );
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

	{ //create object vertices
		// std::vector< PosNorTexVertex > vertices;
		std::vector< PosNorTanTexVertex > vertices;
		scene_viewer->load_mesh_vertices(vertices);
		
		size_t bytes = vertices.size() * sizeof(vertices[0]);

		object_vertices = rtg.helpers.create_buffer(bytes, 
													VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
													VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
													Helpers::Unmapped
		);

		//copy data to buffer:
		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}

	// build_material_texture();
	scene_viewer->build_scene_objects();
	scene_viewer->cache_rest_pose_and_duration(anim_duration);

	{ //initialization
		auto it = scene_viewer->camera_indices.find(rtg.configuration.camera_name);
		if (it != scene_viewer->camera_indices.end()) {
			current_camera_index = it->second;
		}
		
		lines_vertices.clear();
		for (auto & v : scene_viewer->bbox_lines_vertices) {
			lines_vertices.push_back(v);
		}

		debug_camera.radius = 20.0f;
		debug_camera.far = 10000.0f;
	}


	{ //make image views for the textures
		material_system.texture_views.reserve(material_system.textures.size());
		for (Helpers::AllocatedImage const &image : material_system.textures) {
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

			material_system.texture_views.emplace_back(image_view);
		}
		assert(material_system.texture_views.size() == material_system.textures.size());
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
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &material_system.texture_sampler) );
	}
		
	{ // create the material descriptor pool
		uint32_t per_material = uint32_t(material_system.material_params.size()); //for easier-to-read counting

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

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &material_system.material_descriptor_pool) );
	}

	{ //allocate and write the material descriptor sets
		
		//allocate the descriptors (using the same alloc_info):
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = material_system.material_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_Material,
		};
		material_system.material_descriptors.assign(material_system.material_params.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet &descriptor_set : material_system.material_descriptors) {
			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set) );
		}

		std::vector<VkWriteDescriptorSet> writes;
		size_t N = material_system.material_descriptors.size();
		std::vector<VkDescriptorBufferInfo> material_params_infos(N);
		std::vector<VkDescriptorImageInfo>  albedo_infos(N);
		writes.reserve(N * 2);

		for (size_t i = 0; i < N; ++i) {
			material_params_infos[i] = VkDescriptorBufferInfo{
				.buffer = material_system.material_params[i].handle,
				.offset = 0,
				.range  = material_system.material_params[i].size,
			};

			writes.push_back(VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_system.material_descriptors[i],
					.dstBinding = 0, // MaterialParams
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &material_params_infos[i],
			});

			uint32_t tex_index = material_system.material_tex_info[i].has_albedo_tex ? material_system.material_tex_info[i].albedo_tex_index : material_system.default_texture_index;

			

			albedo_infos[i] = VkDescriptorImageInfo{
				.sampler = material_system.texture_sampler,
				.imageView = material_system.texture_views[tex_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			writes.push_back(VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_system.material_descriptors[i],
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &albedo_infos[i],
			});
		}


		vkUpdateDescriptorSets( rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr );
	}

}

Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}


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

	if (timestamp_query_pool != VK_NULL_HANDLE) {
		vkDestroyQueryPool(rtg.device, timestamp_query_pool, nullptr);
		timestamp_query_pool = VK_NULL_HANDLE;
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
	double cpu_frame_ms = 0.0;
	CPUTimer frame_timer;
	frame_timer.start();

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
			assert(workspace.World_src.size == sizeof(scene_viewer->world));

			//host-side copy into World_src:
			memcpy(workspace.World_src.allocation.data(), &scene_viewer->world, sizeof(scene_viewer->world));

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

	double cpu_cull_ms = 0.0;
	CPUTimer cull_timer;
	cull_timer.start();

	std::vector<uint32_t> draw_list;
	draw_list.reserve(scene_viewer->object_instances.size());
	if (culling_mode == CullingMode::None) {
		for (uint32_t i = 0; i < (uint32_t)scene_viewer->object_instances.size(); ++i) draw_list.push_back(i);
	} else if (culling_mode == CullingMode::Frustum) { //Frustum
		glm::mat4 CLIP_FROM_WORLD_CULL_glm = to_glm(CLIP_FROM_WORLD_FOR_CULLING);

		for (uint32_t i = 0; i < (uint32_t)scene_viewer->object_instances.size(); ++i) {
			SceneViewer::ObjectInstance const& inst = scene_viewer->object_instances[i];
			SceneViewer::AABB const& box_local = scene_viewer->mesh_aabb_local.at(inst.mesh);

			glm::mat4 WORLD_FROM_LOCAL_glm = to_glm(inst.transform.WORLD_FROM_LOCAL);
			glm::mat4 CLIP_FROM_LOCAL_glm  = CLIP_FROM_WORLD_CULL_glm * WORLD_FROM_LOCAL_glm;

			if (scene_viewer->aabb_intersects_frustum(box_local, CLIP_FROM_LOCAL_glm)) {
				draw_list.push_back(i);
			}
		}
	} else if (culling_mode == CullingMode::BVH) {
		glm::mat4 CLIP_FROM_WORLD_CULL_glm = to_glm(CLIP_FROM_WORLD_FOR_CULLING);

		// bvh_lines_vertices.clear();
		scene_viewer->cull_with_bvh(CLIP_FROM_WORLD_CULL_glm, draw_list);

		//dynamic objects
		for (uint32_t inst_idx : scene_viewer->dynamic_instances) {
			SceneViewer::ObjectInstance const& inst = scene_viewer->object_instances[inst_idx];
			SceneViewer::AABB const& box_local = scene_viewer->mesh_aabb_local.at(inst.mesh);

			glm::mat4 WORLD_FROM_LOCAL_glm = to_glm(inst.transform.WORLD_FROM_LOCAL);
			glm::mat4 CLIP_FROM_LOCAL_glm = CLIP_FROM_WORLD_CULL_glm * WORLD_FROM_LOCAL_glm;

			if (scene_viewer->aabb_intersects_frustum(box_local, CLIP_FROM_LOCAL_glm)) {
				draw_list.push_back(inst_idx);
			}
		}
	}

	cpu_cull_ms = cull_timer.ms();

	// std::cout << "instances: " << object_instances.size()
	// 		<< " visible: " << draw_list.size() << "\n";
	
	//if (!object_instances.empty())
	if (!draw_list.empty()) { //upload object transforms:
		// size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		size_t needed_bytes = draw_list.size() * sizeof(ObjectsPipeline::Transform);

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
			// for (ObjectInstance const &inst : object_instances) {
			// 	*out = inst.transform;
			// 	++out;
			// }

			for (uint32_t slot = 0; slot < (uint32_t)draw_list.size(); ++slot) {
				SceneViewer::ObjectInstance const &inst = scene_viewer->object_instances[draw_list[slot]];
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

		//
		vkCmdResetQueryPool(workspace.command_buffer, timestamp_query_pool, 0, 2);

		vkCmdWriteTimestamp(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			timestamp_query_pool,
			0
		);

		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		uint32_t fb_w = rtg.swapchain_extent.width;
		uint32_t fb_h = rtg.swapchain_extent.height;
		float fb_aspect = fb_w / float(fb_h);

		float target_aspect = fb_aspect;
		bool constrain_to_camera_aspect = (camera_mode == CameraMode::Scene) && (!debug_camera_mode);
		if (constrain_to_camera_aspect && current_camera_index < scene_viewer->camera_aspects.size()) {
			target_aspect = scene_viewer->camera_aspects[current_camera_index];
		}

		target_aspect = scene_viewer->camera_aspects[current_camera_index];

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

		// if (!object_instances.empty())
		if (!draw_list.empty()) { //draw with the objects pipeline:
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
			// for (ObjectInstance const &inst : object_instances) {
			// 	uint32_t index = uint32_t(&inst - &object_instances[0]);

			// 	//bind material descriptor set:
			// 	vkCmdBindDescriptorSets(
			// 		workspace.command_buffer, //command buffer
			// 		VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
			// 		objects_pipeline.layout, //pipeline layout
			// 		2, //second set
			// 		1, &material_descriptors[inst.material], //descriptor sets count, ptr
			// 		0, nullptr //dynamic offsets count, ptr
			// 	);

			// 	vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
			// }

			for (uint32_t slot = 0; slot < (uint32_t)draw_list.size(); ++slot) {
				SceneViewer::ObjectInstance const& inst = scene_viewer->object_instances[draw_list[slot]];

				//bind material descriptor set:
				vkCmdBindDescriptorSets(
					workspace.command_buffer, //command buffer
					VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
					objects_pipeline.layout, //pipeline layout
					2, //second set
					1, &material_system.material_descriptors[inst.material], //descriptor sets count, ptr
					0, nullptr //dynamic offsets count, ptr
				);

				vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, slot);
			}		
			// vkCmdDraw(workspace.command_buffer, uint32_t(object_vertices.size / sizeof(ObjectsPipeline::Vertex)), 1, 0, 0);
			// vkCmdDraw(workspace.command_buffer, torus_vertices.count, 1, torus_vertices.first, 0);
			// vkCmdDraw(workspace.command_buffer, plane_vertices.count, 1, plane_vertices.first, 0);

		}
		vkCmdEndRenderPass(workspace.command_buffer);

		vkCmdWriteTimestamp(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			timestamp_query_pool,
			1
		);
	}

	//end recording:
	VK( vkEndCommandBuffer(workspace.command_buffer) );

	cpu_frame_ms = frame_timer.ms();

	
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

	double cpu_wait_gpu_ms = 0.0;
	CPUTimer cpu_wait_timer;
	cpu_wait_timer.start();

	uint64_t timestamps[2] = {};

	VK(vkGetQueryPoolResults(
		rtg.device,
		timestamp_query_pool,
		0, 2,
		sizeof(timestamps),
		timestamps,
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	));

	double gpu_ms = double(timestamps[1] - timestamps[0]) * timestamp_period * 1e-6;

	cpu_wait_gpu_ms = cpu_wait_timer.ms();


	// std::cout
	// << "instances: " << object_instances.size()
	// << " visible: " << draw_list.size() << "\n"
	// << "cpu cull ms: " << cpu_cull_ms << "\n"
	// << "cpu frame ms: " << cpu_frame_ms << "\n"
	// << "cpu wait for gpu ms: " << cpu_wait_gpu_ms << "\n"
	// << "gpu draw ms: " << gpu_ms << "\n\n";

	if (perf_log.is_open()) {
		perf_log
		<< frame_id++ << ","
		<< scene_viewer->object_instances.size() << ","
		<< draw_list.size() << ","
		<< cpu_cull_ms << ","
		<< cpu_frame_ms << ","
		<< cpu_wait_gpu_ms << ","
		<< gpu_ms
		<< "\n";
	}

}


void Tutorial::update(float dt) {
	time += dt;

	if (!anim_paused) anim_time += dt;

	// if (anim_loop && anim_duration > 0.0f) {
	// 	anim_time = std::fmod(anim_time, anim_duration);
	// 	if (anim_time < 0.0f) anim_time += anim_duration;
	// }

	if (anim_duration > 0.0f) {
		if (anim_loop) {
			anim_time = std::fmod(anim_time, anim_duration);
			if (anim_time < 0.0f) anim_time += anim_duration;
		} else {
			anim_time = std::min(anim_time, anim_duration);
		}
	}


	if (!scene_viewer->s72.drivers.empty()) {
		scene_viewer->apply_drivers(anim_time);
		scene_viewer->update_scene_objects();
	}

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

		CLIP_FROM_WORLD = scene_viewer->camera_proj_matrices[current_camera_index] * scene_viewer->camera_view_matrices[current_camera_index];
		
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

	CLIP_FROM_WORLD_FOR_CULLING = CLIP_FROM_WORLD;

	if (debug_camera_mode) {
		scene_viewer->frustum_lines_vertices.clear();
		scene_viewer->append_frustum_lines_world(to_glm(CLIP_FROM_WORLD), 255, 255, 0, 255);

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
		for (auto & v : scene_viewer->bbox_lines_vertices) {
			lines_vertices.push_back(v);
		}

		for (auto & v : scene_viewer->frustum_lines_vertices) {
			lines_vertices.push_back(v);
		}

		// std::cout << "bvh_lines_vertices = " << bvh_lines_vertices.size() << "\n";

		// for (auto & v : bvh_lines_vertices) {
		// 	lines_vertices.push_back(v);
		// }
	}

	// { //static sun and sky:
	// 	scene_viewer->world.SKY_DIRECTION.x = 0.0f;
	// 	scene_viewer->world.SKY_DIRECTION.y = 0.0f;
	// 	scene_viewer->world.SKY_DIRECTION.z = 1.0f;

	// 	scene_viewer->world.SKY_ENERGY.r = 0.1f;
	// 	scene_viewer->world.SKY_ENERGY.g = 0.1f;
	// 	scene_viewer->world.SKY_ENERGY.b = 0.2f;

	// 	scene_viewer->world.SUN_DIRECTION.x = 6.0f / 23.0f;
	// 	scene_viewer->world.SUN_DIRECTION.y = 13.0f / 23.0f;
	// 	scene_viewer->world.SUN_DIRECTION.z = 18.0f / 23.0f;

	// 	scene_viewer->world.SUN_ENERGY.r = 1.0f;
	// 	scene_viewer->world.SUN_ENERGY.g = 1.0f;
	// 	scene_viewer->world.SUN_ENERGY.b = 0.9f;
	// }

	
	scene_viewer->update_object_instances_camera(CLIP_FROM_WORLD);

}


void Tutorial::on_input(InputEvent const &evt) {
	//if there is a current action, it gets input priority:
	if (action) {
		action(evt);
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_SPACE) {
		anim_paused = !anim_paused;
		return;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_R) {
		anim_time = 0.0f;
		return;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_L) {
		anim_loop = !anim_loop;
		return;
	}

	//general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB && scene_viewer->camera_indices.size() != 0) {
		//switch camera modes
		camera_mode = CameraMode((int(camera_mode) + 1) % 2);
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_D) {
		//switch camera modes
		debug_camera_mode = !debug_camera_mode;
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_C) {
		culling_mode = CullingMode((int(culling_mode) + 1) % 3);
		return;
	}

	if (camera_mode == CameraMode::Scene &&
		evt.type == InputEvent::KeyDown) {

		if (evt.key.key == GLFW_KEY_LEFT_BRACKET) {	// [
			if (!scene_viewer->camera_indices.empty()) {
				current_camera_index =
					(current_camera_index + scene_viewer->camera_indices.size() - 1)
					% scene_viewer->camera_indices.size();
			}
			return;
		}

		if (evt.key.key == GLFW_KEY_RIGHT_BRACKET) {  // ]
			if (!scene_viewer->camera_indices.empty()) {
				current_camera_index =
					(current_camera_index + 1)
					% scene_viewer->camera_indices.size();
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



