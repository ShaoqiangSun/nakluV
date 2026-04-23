#pragma once

#include <vector>
#include "S72.hpp"
#include "PosNorTanTexVertex.hpp"
#include "MaterialSystem.hpp"
#include "mat4.hpp"
#include <unordered_set>
#include "Tutorial.hpp"

struct SceneViewer {

    SceneViewer(MaterialSystem &);
    ~SceneViewer();

    struct AABB {
		glm::vec3 min = glm::vec3( std::numeric_limits<float>::infinity());
		glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

	};

    struct BVHNode {
		AABB box_world;
		int left = -1;
		int right = -1;
		uint32_t start = 0;
		uint32_t count = 0;
	};

    struct ObjectVertices {
		uint32_t first = 0;
		uint32_t count = 0;
	};

    struct ObjectInstance {
		ObjectVertices vertices;
		Tutorial::ObjectsPipeline::Transform transform;
		uint32_t mesh = 0;
		uint32_t material = 0;
	};

	struct LightInfo {
		enum class Type : uint32_t {
            Sun = 0,
            Sphere = 1,
            Spot = 2,
        };

        Type type;

        uint32_t shadow = 0;

        glm::vec3 position = glm::vec3(0.0f);   // sphere / spot
        glm::vec3 direction = glm::vec3(0.0f);  // sun / spot
        glm::vec3 tint = glm::vec3(1.0f);

        float angle = 0.0f;     // sun
        float strength = 0.0f;  // sun

        float radius = 0.0f;    // sphere / spot
        float power = 0.0f;     // sphere / spot
        float limit = std::numeric_limits<float>::infinity(); // sphere / spot

        float fov = 0.0f;       // spot
        float blend = 0.0f;     // spot
	};

	// Resolved per-scene water instance derived from an S72::Water attached to
	// some scene node.  Caustic generation reads these.
	struct WaterInfo {
		glm::mat4 world_from_local = glm::mat4(1.0f);

		float width = 10.0f;
		float height = 10.0f;
		uint32_t resolution = 64;
		float receiver_z = 5.0f;
		float caustic_extent = 12.0f;
		float caustic_intensity = 1.0f;

		// When true, use caustic_tint (linear RGB) as main-pass caustic colour instead
		// of auto sun / sphere light colour.
		bool caustic_tint_set = false;
		glm::vec3 caustic_tint{1.0f, 1.0f, 1.0f};

		struct WaveOctave {
			float amplitude = 0.05f;
			float frequency = 1.0f;
			float direction = 0.0f;
			float speed = 1.0f;
		};
		std::vector< WaveOctave > waves;

		// Axis-aligned room for six-face caustic splats (world space).
		glm::vec3 room_min{0.0f};
		glm::vec3 room_max{1.0f};
		bool room_aabb_set = false;
	};

    bool aabb_intersects_frustum(AABB const &box_target_space, glm::mat4 const& transform_matrix);

    void load_mesh_vertices(std::vector< PosNorTanTexVertex > &vertices_pool);
    void append_bbox_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff);
	void append_frustum_lines_world(glm::mat4 const &CLIP_FROM_WORLD, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff);
	void append_bvh_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff);
    void traverse_node(S72::Node* node, glm::mat4 const& parent, bool inherited_dynamic);
	void update_transforms_node(S72::Node* node, glm::mat4 const& parent);
	void build_scene_objects();
	void update_scene_objects();
    void update_object_instances_camera(mat4 const &CLIP_FROM_WORLD);
    void cache_rest_pose_and_duration(float &anim_duration);
	void apply_drivers(float t);
	void mark_driven_nodes();
	int build_bvh(std::vector<std::pair<AABB,uint32_t>> &items, int start, int count);
	void build_bvh_for_static();
	void cull_with_bvh(glm::mat4 const& CLIP_FROM_WORLD_CULL_glm, std::vector<uint32_t> &draw_list);

    S72 s72;
    MaterialSystem &material_system;

    //load .s72 scene mesh vertices
	std::unordered_map<std::string, uint32_t> mesh_id;
	std::vector<ObjectVertices> mesh_vertices;
	std::vector<AABB> mesh_aabb_local;

    std::vector<mat4> camera_view_matrices;
	std::vector<mat4> camera_proj_matrices;
	std::vector<float> camera_aspects;
	std::unordered_map<std::string, uint32_t> camera_indices;

    std::vector< Tutorial::LinesPipeline::Vertex > bbox_lines_vertices;
	std::vector< Tutorial::LinesPipeline::Vertex > frustum_lines_vertices;
	std::vector< Tutorial::LinesPipeline::Vertex > bvh_lines_vertices;

    Tutorial::ObjectsPipeline::World world;

    std::vector< ObjectInstance > object_instances;
	std::unordered_map<S72::Node*, uint32_t> node_to_instance;
	std::vector<uint32_t> dynamic_instances;
	std::unordered_set<S72::Node*> driven_nodes;

    std::vector<S72::Node*> all_nodes;
	std::vector<S72::vec3> rest_T;
	std::vector<S72::quat> rest_R;
	std::vector<S72::vec3> rest_S;

    std::vector<uint32_t> bvh_indices;
	std::vector<BVHNode> bvh_nodes;
	bool bvh_built = false;

	std::vector<LightInfo> lights;

	// All water surfaces discovered during traversal. Used by the caustic pass.
	std::vector<WaterInfo> waters;
};