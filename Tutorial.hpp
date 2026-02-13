#pragma once

#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "mat4.hpp"

#include "RTG.hpp"
#include "S72.hpp"

struct Tutorial : RTG::Application {

	Tutorial(RTG &);
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	S72 s72;
	struct AABB {
		glm::vec3 min = glm::vec3( std::numeric_limits<float>::infinity());
		glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

	};

	void load_mesh_vertices(S72 const &s72, std::vector< PosNorTanTexVertex > &vertices_pool);
	void append_bbox_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff);
	void append_frustum_lines_world(glm::mat4 const &CLIP_FROM_WORLD, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff);
	void traverse_node(S72::Node* node, glm::mat4 const& parent);
	void build_scene_objects();
	void update_object_instances_camera();
	void build_material_texture_indices();
	void load_all_textures();
	

	//kept for use in destructor:
	RTG &rtg;

	//--------------------------------------------------------------------
	//Resources that last the lifetime of the application:

	//chosen format for depth buffer:
	VkFormat depth_format{};
	//Render passes describe how pipelines write to images:
	VkRenderPass render_pass = VK_NULL_HANDLE;

	//Pipelines:
	struct BackgroundPipeline {
		//no descriptor set layouts

		struct Push {
			float time;
		};

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//no vertex bindings

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} background_pipeline;

	struct LinesPipeline {
		//descriptor set layouts:
		VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;

		//types for descriptors:
		struct Camera {
			mat4 CLIP_FROM_WORLD;
		};
		static_assert(sizeof(Camera) == 16*4, "camera buffer structure is packed");

		//no push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		using Vertex = PosColVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} lines_pipeline;

	struct ObjectsPipeline {
		//descriptor set layouts:
		VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_Material = VK_NULL_HANDLE;

		//types for descriptors:
		struct World {
			struct { float x, y, z, padding_; } SKY_DIRECTION;
			struct { float r, g, b, padding_; } SKY_ENERGY;
			struct { float x, y, z, padding_; } SUN_DIRECTION;
			struct { float r, g, b, padding_; } SUN_ENERGY;
		};
		static_assert(sizeof(World) == 4*4 + 4*4 + 4*4 + 4*4, "World is the expected size.");

		struct Transform {
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;
		};
		static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform is the expected size.");

		struct Material {
			struct { float r, g, b, padding_;} ALBEDO;
		};

		//no push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		// using Vertex = PosNorTexVertex;
		using Vertex = PosNorTanTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} objects_pipeline;

	struct MaterialTextureInfo {
		uint32_t albedo_tex_index = 0;
		uint32_t has_albedo_tex = 0;
		
	};

	uint32_t white_texture_index;

	//pools from which per-workspace things are allocated:
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

	//workspaces hold per-render resources:
	struct Workspace {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; //from the command pool above; reset at the start of every render.

		//location for lines data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer lines_vertices_src; //host coherent; mapped
		Helpers::AllocatedBuffer lines_vertices; //device-local

		//location for LinesPipeline::Camera data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Camera_src; //host coherent; mapped
		Helpers::AllocatedBuffer Camera; //device-local
		VkDescriptorSet Camera_descriptors; //references Camera

		//location for ObjectsPipeline::World data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer World_src; //host coherent; mapped
		Helpers::AllocatedBuffer World; //device-local
		VkDescriptorSet World_descriptors; //references World

		//location for ObjectsPipeline::Transforms data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Transforms_src; //host coherent; mapped
		Helpers::AllocatedBuffer Transforms; //device-local
		VkDescriptorSet Transforms_descriptors; //references Transforms
	};
	std::vector< Workspace > workspaces;

	//-------------------------------------------------------------------
	//static scene resources:
	Helpers::AllocatedBuffer object_vertices;
	struct ObjectVertices {
		uint32_t first = 0;
		uint32_t count = 0;
	};
	ObjectVertices plane_vertices;
	ObjectVertices torus_vertices;
	//load .s72 scene mesh vertices
	std::unordered_map<std::string, ObjectVertices> mesh_vertices;

	std::unordered_map<std::string, AABB> mesh_aabb_local;

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

	//--------------------------------------------------------------------
	//Resources that change when the swapchain is resized:

	virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

	Helpers::AllocatedImage swapchain_depth_image;
	VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
	std::vector< VkFramebuffer > swapchain_framebuffers;
	//used from on_swapchain and the destructor: (framebuffers are created in on_swapchain)
	void destroy_framebuffers();

	//--------------------------------------------------------------------
	//Resources that change when time passes or the user interacts:

	virtual void update(float dt) override;
	virtual void on_input(InputEvent const &) override;

	//modal action, intercepts inputs:
	std::function< void(InputEvent const &) > action;

	float time = 0.0f;

	//for selecting between cameras:
	enum class CameraMode {
		Scene = 0,
		Free = 1,
	};

	CameraMode camera_mode = CameraMode::Free;
	bool debug_camera_mode = false;

	//used when camera_mode == CameraMode::Free:
	struct OrbitCamera {
		float target_x = 0.0f, target_y = 0.0f, target_z = 0.0f; //where the camera is looking + orbiting
		float radius = 2.0f; //distance from camera to target
		float azimuth = 0.0f; //counterclockwise angle around z axis between x axis and camera direction (radians)
		float elevation = 0.25f * float(M_PI); //angle up from xy plane to camera direction (radians)

		float fov = 60.0f / 180.0f * float(M_PI); //vertical field of view (radians)
		float near = 0.1f; //near clipping plane
		float far = 1000.0f; //far clipping plane
	};

	OrbitCamera free_camera;
	//
	OrbitCamera debug_camera;

	//computed from the current camera (as set by camera_mode) during update():
	mat4 CLIP_FROM_WORLD;
	std::vector<mat4> camera_view_matrices;
	std::vector<mat4> camera_proj_matrices;
	std::vector<float> camera_aspects;
	std::unordered_map<std::string, uint32_t> camera_indices;
	uint32_t current_camera_index = 0;

	std::vector< LinesPipeline::Vertex > lines_vertices;
	std::vector< LinesPipeline::Vertex > bbox_lines_vertices;
	std::vector< LinesPipeline::Vertex > frustum_lines_vertices;

	ObjectsPipeline::World world;

	struct ObjectInstance {
		ObjectVertices vertices;
		ObjectsPipeline::Transform transform;
		uint32_t material = 0;
	};
	std::vector< ObjectInstance > object_instances;

	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;
};
