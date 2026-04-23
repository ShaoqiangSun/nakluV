#pragma once

#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "mat4.hpp"

#include "RTG.hpp"
#include "S72.hpp"

#include <array>
#include <fstream>
#include <unordered_set>
#include <memory>

#include "MaterialSystem.hpp"

#include <glm/glm.hpp>

struct SceneViewer;

struct Tutorial : RTG::Application {

	Tutorial(RTG &);
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	// Maximum number of spot lights that can cast shadows simultaneously.
	static constexpr uint32_t MAX_SHADOW_CASTERS = 7;

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
			uint32_t MAX_MIP, LIGHT_COUNT, padding2_, padding3_;
		};
		static_assert(sizeof(World) == 7 * 4 * 4, "World is the expected size.");

		struct Transform {
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;
		};
		static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform is the expected size.");

		struct Material {
			struct { float r, g, b, padding_;} ALBEDO; // padding_ = 1 for pbr:water (procedural waves in frag)
			struct {float ROUGHNESS, METALLIC, padding2_, padding3_;} PBR;
			uint32_t TYPE, padding1_, padding2_, padding3_;
		};

		struct Light {
			vec4 POSITION_TYPE;     // xyz = position, w = type
			vec4 DIRECTION_SHADOW;  // xyz = direction, w = shadow slot index (-1 if no shadow)
			vec4 TINT_STRENGTH;     // rgb = tint, w = strength / power
			vec4 PARAMS;            // angle / radius, limit, fov, blend
		};
		static_assert(sizeof(Light) == 4 * 4 * 4, "Light is the expected size.");

		// Shadow matrices for up to MAX_SHADOW_CASTERS spot lights (std140 uniform buffer).
		struct ShadowMatricesUniform {
			glm::mat4 LIGHT_FROM_WORLD[MAX_SHADOW_CASTERS];
		};
		static_assert(sizeof(ShadowMatricesUniform) == MAX_SHADOW_CASTERS * 64,
		              "ShadowMatricesUniform is the expected size.");

		// Caustic apply uniform — tells the main pass how to project a world
		// XY into the caustic map.  One shared map across all waters, so
		// objects.frag samples it once per fragment.
		struct CausticApplyUniform {
			// x,y,z reserved / unused for six-face mode; w = 1 if caustics active
			glm::vec4 CENTER_EXTENT_ACTIVE;
			// x = water node world z (reference plane); y = receiver_z;
			// z = world half-thickness of water slab (caustics off horizontal sheet)
			glm::vec4 WATER_Z_RECEIVER_Z;
			glm::vec4 ROOM_MIN;
			glm::vec4 ROOM_MAX;
			// rgb scales reflected caustics when the scene has no sun (e.g. sphere light)
			glm::vec4 CAUSTIC_TINT;
		};
		static_assert(sizeof(CausticApplyUniform) == 5 * 16, "CausticApplyUniform std140 size");

		//no push constants for main pipeline

		VkPipelineLayout layout = VK_NULL_HANDLE;

		// using Vertex = PosNorTexVertex;
		using Vertex = PosNorTanTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} objects_pipeline;

	// Pipeline used to render shadow maps for spot lights
	// Renders a R32_SFLOAT color attachment so that the result can be sampled
	// on macOS/MoltenVK without the depth-texture sampling restriction
	struct ShadowPipeline {
		VkDescriptorSetLayout set0_WorldTransforms = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass shadow_render_pass);
		void destroy(RTG &);
	} shadow_pipeline;

	// Caustic photon-splatting pipeline.  Draws a procedurally-generated
	// water grid into a 2D R16_SFLOAT caustic map: each vertex computes its
	// reflected ray and outputs the intersection with the receiver plane as
	// a clip-space position.  Additive blending accumulates overlapping
	// triangles into bright focus spots.
	struct CausticPipeline {
		VkDescriptorSetLayout set0_WaterParams = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
		VkPipeline handle = VK_NULL_HANDLE;

		// std140 uniform buffer; matches `WaterParams` in caustic.vert.
		struct WaterParamsUniform {
			glm::mat4 WORLD_FROM_LOCAL;
			glm::vec4 SIZE_RES_RECV;   // width, height, resolution, receiver_z (legacy)
			glm::vec4 CAUSTIC_CEI;     // center.xy, extent, intensity
			glm::vec4 SUN_TIME;        // surface->sun dir.xyz, time
			glm::vec4 WAVE_COUNT;      // count, pad, pad, pad
			glm::vec4 WAVES[4];        // amplitude, frequency, direction, speed
			glm::vec4 ROOM_MIN;        // xyz = world AABB min for six-face caustics
			glm::vec4 ROOM_MAX;        // xyz = world AABB max
			// xyz = world position for point-light caustics; w = 1 sphere, 0 sun (SUN_TIME.xyz)
			glm::vec4 CAUSTIC_LIGHT_POINT;
			// x = normal-map detail weight, y = UV tiling, z = use_nm (1/0), w unused
			glm::vec4 WATER_NM;
		};
		static_assert(sizeof(WaterParamsUniform) == 256, "WaterParamsUniform std140 size");

		// First water surface: shared waves + transforms for objects.frag (PBR water)
		// and matching caustic WATER_NM / normal texture when enabled.
		struct WaterSurfaceUniform {
			glm::mat4 WORLD_FROM_LOCAL;
			glm::mat4 LOCAL_FROM_WORLD;
			glm::vec4 SIZE_TIME_BLEND; // width, height, time, unused (NM: WAVE_COUNT.yzw)
			glm::vec4 WAVE_COUNT;
			glm::vec4 WAVES[4];
			glm::vec4 ACTIVE; // x = 1 if valid
		};
		static_assert(sizeof(WaterSurfaceUniform) == 240, "WaterSurfaceUniform std140 size");

		void create(RTG &, VkRenderPass caustic_render_pass);
		void destroy(RTG &);
	} caustic_pipeline;

	// Render pass for the caustic accumulation pass.  Single R16_SFLOAT
	// color attachment cleared to 0 at pass start and loaded as sampled
	// image afterwards.
	VkRenderPass caustic_render_pass = VK_NULL_HANDLE;

	// Sampler for reading the caustic map in objects.frag.  Linear + clamp.
	VkSampler caustic_sampler = VK_NULL_HANDLE;

	// Dimensions of the caustic map (edge length; square image).
	// Higher resolution reduces internal caustic-map aliasing when projected
	// onto walls at grazing angles (cost: 6× passes × pixels).
	static constexpr uint32_t CAUSTIC_MAP_SIZE = 1024;
	// Six axis-aligned room faces -> six layers in the caustic texture array.
	static constexpr uint32_t CAUSTIC_ROOM_FACE_COUNT = 6;

	// Max number of water surfaces supported per scene (upper bound on the
	// caustic-pass draw loop).  Only one caustic map is generated; multiple
	// waters accumulate into the same target.
	static constexpr uint32_t MAX_WATER_SURFACES = 4;

	// Render pass for shadow map generation:
	//   attachment 0 – R32_SFLOAT color (the shadow map, sampled in main pass)
	//   attachment 1 – D32_SFLOAT depth (depth testing during shadow pass, not sampled)
	VkRenderPass shadow_render_pass = VK_NULL_HANDLE;

	// Sampler for reading R32_SFLOAT shadow maps (no comparison — done manually in GLSL)
	VkSampler shadow_sampler = VK_NULL_HANDLE;

	// Per-light shadow map resources (one entry per slot in [0, MAX_SHADOW_CASTERS))
	struct ShadowLight {
		// R32_SFLOAT color image — stores depth values, sampled in the main pass
		Helpers::AllocatedImage shadow_map;
		VkImageView shadow_map_view = VK_NULL_HANDLE;
		// D32_SFLOAT depth image — used only for depth testing during the shadow pass
		Helpers::AllocatedImage depth_buffer;
		VkImageView depth_buffer_view = VK_NULL_HANDLE;
		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		uint32_t size = 0;        // edge length of the shadow map in pixels
		uint32_t light_index = 0; // index into scene_viewer->lights
	};
	std::vector< ShadowLight > shadow_lights; // length <= MAX_SHADOW_CASTERS

	// Maps scene_viewer->lights[i] to its shadow slot, or -1 if none
	std::vector< int32_t > light_shadow_slot;

	// 1×1 dummy R32_SFLOAT image (value 1.0 = fully lit) for unused shadow map slots
	Helpers::AllocatedImage dummy_shadow_image;
	VkImageView dummy_shadow_image_view = VK_NULL_HANDLE;


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

		//location for ObjectsPipeline::Light data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Lights_src; // host coherent; mapped
		Helpers::AllocatedBuffer Lights;     // device-local
		VkDescriptorSet Lights_descriptors;  // fold into set0_World

		// All object instances' WORLD_FROM_LOCAL matrices (mat4 each), used by shadow pipeline
		// Uploaded per-frame; size grows lazily with object count
		Helpers::AllocatedBuffer AllWorldTransforms_src; // host coherent; mapped
		Helpers::AllocatedBuffer AllWorldTransforms;     // device-local
		VkDescriptorSet AllWorldTransforms_descriptors;  // set0 for shadow pipeline

		// Shadow matrices uniform buffer: CLIP_FROM_WORLD for each shadow light slot
		Helpers::AllocatedBuffer ShadowMatrices_src; // host coherent; mapped
		Helpers::AllocatedBuffer ShadowMatrices;     // device-local
		// (referenced from World_descriptors binding 6 after allocation)

		// Caustic apply uniform (for main pass binding 8 on set0_World):
		Helpers::AllocatedBuffer CausticApply_src; //host coherent; mapped
		Helpers::AllocatedBuffer CausticApply;     //device-local

		// Water surface waves (binding 9 on set0_World, objects.frag):
		Helpers::AllocatedBuffer WaterSurface_src;
		Helpers::AllocatedBuffer WaterSurface;

		//---- caustic-pass per-workspace resources ----
		// Packed array of WaterParams uniforms (one per water, aligned to
		// minUniformBufferOffsetAlignment).  Used with a dynamic-offset
		// descriptor so a single set is rebound per draw.
		Helpers::AllocatedBuffer CausticWater_src; // host coherent; mapped
		Helpers::AllocatedBuffer CausticWater;     // device-local
		VkDescriptorSet CausticWater_descriptors = VK_NULL_HANDLE; // set0 for caustic pipeline

		// R16_SFLOAT caustic accumulation: 2D array (one layer per room face).
		Helpers::AllocatedImage caustic_map;
		std::array< VkImageView, CAUSTIC_ROOM_FACE_COUNT > caustic_map_layer_views{};
		VkImageView caustic_map_array_view = VK_NULL_HANDLE;
		std::array< VkFramebuffer, CAUSTIC_ROOM_FACE_COUNT > caustic_framebuffers{};
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

	// Default pose matches caustic-pool.s72 Pool-Camera: eye at (0,-28,6),
	// looking toward +Y (into the open side of the room).
	OrbitCamera free_camera{
		.target_x = 0.0f, .target_y = 0.0f, .target_z = 6.0f,
		.radius = 28.0f,
		.azimuth = -0.5f * float(M_PI),
		.elevation = 0.0f,
		.fov = 0.9f,
	};
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

	// Stride (in bytes) between consecutive WaterParamsUniform entries in the
	// per-workspace CausticWater buffer, rounded up to the device's
	// minUniformBufferOffsetAlignment so it can be used as a dynamic offset.
	uint32_t caustic_uniform_stride = 0;

	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;

	std::ofstream perf_log;
	uint64_t frame_id = 0;

	float exposure_stops = 0.0f; //default 0
    uint32_t tone_map_op = 0; 
};
