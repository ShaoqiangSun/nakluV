#pragma once

#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "mat4.hpp"

#include "RTG.hpp"
#include "S72.hpp"

#include <fstream>
#include <unordered_set>
#include <memory>

#include "MaterialSystem.hpp"

struct SceneViewer;

struct Tutorial : RTG::Application {

	Tutorial(RTG &);
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	struct CPUTimer {
		using clock = std::chrono::steady_clock;
		clock::time_point t0;

		void start() { t0 = clock::now(); }

		double ms() const {
			auto t1 = clock::now();
			return std::chrono::duration<double, std::milli>(t1 - t0).count();
		}
	};
	

	//kept for use in destructor:
	RTG &rtg;
	MaterialSystem material_system;
	std::unique_ptr<SceneViewer> scene_viewer;
	

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
			struct { float x, y, z, padding_; } EYE;
			struct { float x, y, z, padding_; } TONE;
			uint32_t MAX_MIP, padding1_, padding2_, padding3_;
		};
		static_assert(sizeof(World) == 7 * 4 * 4, "World is the expected size.");

		struct Transform {
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;
		};
		static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform is the expected size.");

		struct Material {
			struct { float r, g, b, padding_;} ALBEDO;
			struct {float ROUGHNESS, METALLIC, padding2_, padding3_;} PBR;
			uint32_t TYPE, padding1_, padding2_, padding3_;
		};

		//no push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		// using Vertex = PosNorTexVertex;
		using Vertex = PosNorTanTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} objects_pipeline;


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

	bool anim_paused = false;
	bool anim_loop = true;
	float anim_time = 0.0f;
	float anim_duration = 0.0f;

	//for selecting between cameras:
	enum class CameraMode {
		Scene = 0,
		Free = 1,
	};

	enum class CullingMode {
		None,
		Frustum,
		BVH,
	};

	CameraMode camera_mode = CameraMode::Free;
	bool debug_camera_mode = false;
	CullingMode culling_mode = CullingMode::None;

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
	mat4 CLIP_FROM_WORLD_FOR_CULLING;
	
	uint32_t current_camera_index = 0;

	std::vector< LinesPipeline::Vertex > lines_vertices;
	std::vector< LinesPipeline::Vertex > bbox_lines_vertices;
	std::vector< LinesPipeline::Vertex > frustum_lines_vertices;
	std::vector< LinesPipeline::Vertex > bvh_lines_vertices;


	VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
	float timestamp_period = 0.0f;

	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;

	std::ofstream perf_log;
	uint64_t frame_id = 0;

	float exposure_stops = 0.0f; //default 0
    uint32_t tone_map_op = 0; 
};
