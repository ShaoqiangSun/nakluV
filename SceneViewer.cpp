#include "SceneViewer.hpp"
#include "Tutorial.hpp"
#include <fstream>

static void read_bytes(std::ifstream &f, uint64_t off, void *dst, size_t n) {
    f.seekg(std::streamoff(off), std::ios::beg);
    if (!f) throw std::runtime_error("seek failed");
    f.read(reinterpret_cast<char*>(dst), std::streamsize(n));
    if (!f) throw std::runtime_error("read failed");
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

static inline bool aabb_outside_plane(SceneViewer::AABB const &box, glm::vec4 const &plane) {
	glm::vec3 n(plane.x, plane.y, plane.z);

	glm::vec3 v;
    v.x = (n.x >= 0.0f) ? box.max.x : box.min.x;
    v.y = (n.y >= 0.0f) ? box.max.y : box.min.y;
    v.z = (n.z >= 0.0f) ? box.max.z : box.min.z;

	float dist = plane.x * v.x + plane.y * v.y + plane.z * v.z + plane.w;
    return dist < 0.0f;
}


bool SceneViewer::aabb_intersects_frustum(AABB const &box_target_space, glm::mat4 const& transform_matrix) {
	//Vulkan clip constraints (x,y in [-w,w], z in [0,w]):
    //left:   x + w >= 0
    //right: -x + w >= 0
    //bottom: y + w >= 0
    //top:   -y + w >= 0
    //near:  z >= 0
    //far:  -z + w >= 0
	const glm::vec4 planes_clip[6] = {
        glm::vec4( 1,  0,  0, 1), //left
        glm::vec4(-1,  0,  0, 1), //right
        glm::vec4( 0,  1,  0, 1), //bottom
        glm::vec4( 0, -1,  0, 1), //top
        glm::vec4( 0,  0,  1, 0), //near
        glm::vec4( 0,  0, -1, 1), //far
    };

	glm::mat4 transform_matrix_T = glm::transpose(transform_matrix);

	for (uint32_t i = 0; i < 6; ++i) {
        glm::vec4 plane_target_space = transform_matrix_T * planes_clip[i];
        if (aabb_outside_plane(box_target_space, plane_target_space)) return false;
    }
    return true;
}

static inline SceneViewer::AABB aabb_local_to_world(SceneViewer::AABB const& box_local, glm::mat4 const& WORLD_FROM_LOCAL) {
	glm::vec3 corners[8] = {
        {box_local.min.x, box_local.min.y, box_local.min.z},
        {box_local.max.x, box_local.min.y, box_local.min.z},
        {box_local.min.x, box_local.max.y, box_local.min.z},
        {box_local.max.x, box_local.max.y, box_local.min.z},
        {box_local.min.x, box_local.min.y, box_local.max.z},
        {box_local.max.x, box_local.min.y, box_local.max.z},
        {box_local.min.x, box_local.max.y, box_local.max.z},
        {box_local.max.x, box_local.max.y, box_local.max.z},
    };

	SceneViewer::AABB out;

	for (int i = 0; i < 8; ++i) {
        glm::vec3 p = glm::vec3(WORLD_FROM_LOCAL * glm::vec4(corners[i], 1.0f));
        out.min = glm::min(out.min, p);
        out.max = glm::max(out.max, p);
    }
    return out;
}

static inline SceneViewer::AABB aabb_merge(SceneViewer::AABB const& a, SceneViewer::AABB const& b) {
    SceneViewer::AABB out;
    out.min = glm::min(a.min, b.min);
    out.max = glm::max(a.max, b.max);
    return out;
}

static inline float clamp01(float x) {
	return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

static inline S72::vec3 read_vec3(std::vector<float> const &v, size_t k) {
	return S72::vec3(v[3*k+0], v[3*k+1], v[3*k+2]);
}

static inline S72::quat read_quat_wxyz(std::vector<float> const &v, size_t k) {
	return S72::quat(v[4*k+3], v[4*k+0], v[4*k+1], v[4*k+2]);
}

SceneViewer::SceneViewer(MaterialSystem &material_system_) : material_system(material_system_) {

}

SceneViewer::~SceneViewer() {

}

void SceneViewer::load_mesh_vertices(std::vector< PosNorTanTexVertex > &vertices_pool) {
	for (auto const& [name, mesh] : s72.meshes) {
		auto const &position = mesh.attributes.at("POSITION");
		auto const &normal = mesh.attributes.at("NORMAL");
		auto const &tangent = mesh.attributes.at("TANGENT");
		auto const &texcoord  = mesh.attributes.at("TEXCOORD");

		ObjectVertices range;
		range.first = uint32_t(vertices_pool.size());
		range.count = mesh.count;

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

		uint32_t idx = (uint32_t)mesh_aabb_local.size();
		mesh_id.emplace(name, idx);
		mesh_vertices.push_back(range);
		mesh_aabb_local.push_back(aabb);
		
	}
}

void SceneViewer::append_bbox_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
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

void SceneViewer::append_frustum_lines_world(glm::mat4 const &CLIP_FROM_WORLD, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
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

void SceneViewer::append_bvh_lines_world(AABB const &local, glm::mat4 const &WORLD_FROM_LOCAL, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
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

		bvh_lines_vertices.push_back(v1);
    	bvh_lines_vertices.push_back(v2);

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

void SceneViewer::traverse_node(S72::Node* node, glm::mat4 const& parent, bool inherited_dynamic) {
	bool dynamic_here = inherited_dynamic || (driven_nodes.count(node) != 0);

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
		uint32_t mesh_idx = mesh_id.at(node->mesh->name);
		ObjectVertices vertices = mesh_vertices[mesh_idx];
		AABB local = mesh_aabb_local[mesh_idx];

		append_bbox_lines_world(local, world_glm, 255, 0, 0, 255); // red

		uint32_t inst_idx = uint32_t(object_instances.size());

		object_instances.emplace_back(ObjectInstance{
			.vertices = vertices,
			.transform{
				//.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_matrix,
				.WORLD_FROM_LOCAL = world_matrix,
				.WORLD_FROM_LOCAL_NORMAL = world_matrix_normal,
			},
			.mesh = mesh_idx,
			.material = node->mesh->material != nullptr ? material_system.material_id.at(node->mesh->material) : 0,
		});

		node_to_instance[node] = inst_idx;

		if (dynamic_here) dynamic_instances.push_back(inst_idx);
        else bvh_indices.push_back(inst_idx);
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
		traverse_node(child, world_glm, dynamic_here);
	}

}

void SceneViewer::update_transforms_node(S72::Node* node, glm::mat4 const& parent) {
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
		uint32_t mesh_idx = mesh_id.at(node->mesh->name);
		AABB local = mesh_aabb_local[mesh_idx];

		append_bbox_lines_world(local, world_glm, 255, 0, 0, 255); // red

		uint32_t inst_idx = node_to_instance.at(node);
		//object_instances[inst_idx].CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_matrix;
		object_instances[inst_idx].transform.WORLD_FROM_LOCAL = world_matrix;
		object_instances[inst_idx].transform.WORLD_FROM_LOCAL_NORMAL = world_matrix_normal;
	}

	if (node->camera != nullptr) {
		auto const& cam = *node->camera;

		mat4 view = to_mat4(glm::inverse(world_glm));

		uint32_t camera_idx = camera_indices[cam.name];

		camera_view_matrices[camera_idx] = view;
	}

	for (S72::Node* child : node->children) {
		update_transforms_node(child, world_glm);
	}

}

void SceneViewer::build_scene_objects() {
	object_instances.clear();

	glm::mat4 parent(1.0f);

	mark_driven_nodes();

	for (S72::Node* root: s72.scene.roots) {
		traverse_node(root, parent, false);
	}

	build_bvh_for_static();
}

void SceneViewer::update_scene_objects() {
	bbox_lines_vertices.clear();

	glm::mat4 parent(1.0f);

	for (S72::Node* root: s72.scene.roots) {
		update_transforms_node(root, parent);
	}
}

void SceneViewer::update_object_instances_camera(mat4 const &CLIP_FROM_WORLD) {
	for (auto & obj_inst : object_instances) {
		obj_inst.transform.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * obj_inst.transform.WORLD_FROM_LOCAL;
	}
}

void SceneViewer::cache_rest_pose_and_duration(float &anim_duration) {
	all_nodes.clear();
	rest_T.clear();
	rest_R.clear();
	rest_S.clear();
	
	all_nodes.reserve(s72.nodes.size());
	rest_T.reserve(s72.nodes.size());
	rest_R.reserve(s72.nodes.size());
	rest_S.reserve(s72.nodes.size());

	for (auto &pair : s72.nodes) {
		S72::Node *n = &pair.second;
		all_nodes.push_back(n);
		rest_T.push_back(n->translation);
		rest_R.push_back(n->rotation);
		rest_S.push_back(n->scale);
	}

	anim_duration = 0.0f;
	for (auto const &d : s72.drivers) {
		if (!d.times.empty()) anim_duration = std::max(anim_duration, d.times.back());
	}
}

void SceneViewer::apply_drivers(float t) {
	//reset to rest pose:
	for (size_t i = 0; i < all_nodes.size(); ++i) {
		all_nodes[i]->translation = rest_T[i];
		all_nodes[i]->rotation    = rest_R[i];
		all_nodes[i]->scale       = rest_S[i];
	}

	for (auto const &d : s72.drivers) {
		auto const &times = d.times;
		if (times.empty()) continue;

		if (t <= times.front()) {
			size_t k = 0;
			if (d.channel == S72::Driver::Channel::translation) d.node.translation = read_vec3(d.values, k);
			else if (d.channel == S72::Driver::Channel::scale) d.node.scale = read_vec3(d.values, k);
			else if (d.channel == S72::Driver::Channel::rotation) d.node.rotation = glm::normalize(read_quat_wxyz(d.values, k));
			continue;
		}

		if (t >= times.back()) {
			size_t k = times.size() - 1;
			if (d.channel == S72::Driver::Channel::translation) d.node.translation = read_vec3(d.values, k);
			else if (d.channel == S72::Driver::Channel::scale) d.node.scale = read_vec3(d.values, k);
			else if (d.channel == S72::Driver::Channel::rotation) d.node.rotation = glm::normalize(read_quat_wxyz(d.values, k));
			continue;
		}

		auto it = std::upper_bound(times.begin(), times.end(), t);
		size_t i = size_t(it - times.begin()) - 1;
		size_t j = i + 1;

		float t0 = times[i], t1 = times[j];
		float u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;
		u = clamp01(u);

		if (d.interpolation == S72::Driver::Interpolation::STEP) u = 0.0f;

		if (d.channel == S72::Driver::Channel::translation) {
			S72::vec3 a = read_vec3(d.values, i);
			S72::vec3 b = read_vec3(d.values, j);
			d.node.translation = (1.0f - u) * a + u * b;
		} 
		else if (d.channel == S72::Driver::Channel::scale) {
			S72::vec3 a = read_vec3(d.values, i);
			S72::vec3 b = read_vec3(d.values, j);
			d.node.scale = (1.0f - u) * a + u * b;
		} 
		else if (d.channel == S72::Driver::Channel::rotation) {
			S72::quat qa = glm::normalize(read_quat_wxyz(d.values, i));
			S72::quat qb = glm::normalize(read_quat_wxyz(d.values, j));

			if (d.interpolation == S72::Driver::Interpolation::SLERP) {
				d.node.rotation = glm::slerp(qa, qb, u);
			} else {
				d.node.rotation = glm::normalize(glm::mix(qa, qb, u));
			}
		}
		
	}

}

void SceneViewer::mark_driven_nodes() {
    driven_nodes.clear();
    for (auto const &d : s72.drivers) {
        driven_nodes.insert(&d.node);
    }
}

int SceneViewer::build_bvh(std::vector<std::pair<AABB,uint32_t>> &items, int start, int count) {
	BVHNode node;

	AABB box;

	for (uint32_t i = 0; i < count; ++i) box = aabb_merge(box, items[start + i].first);
    node.box_world = box;

	int node_idx = int(bvh_nodes.size());
    bvh_nodes.push_back(node);

	if (count <= 4) {
        //write leaf range into bvh_indices
        uint32_t leaf_start = uint32_t(bvh_indices.size());
        for (int i = 0; i < count; ++i) bvh_indices.push_back(items[start + i].second);

        bvh_nodes[node_idx].start = leaf_start;
        bvh_nodes[node_idx].count = uint32_t(count);
        return node_idx;
    }

	glm::vec3 ext = box.max - box.min;
    int axis = (ext.y > ext.x) ? 1 : 0;
    if (ext.z > ext[axis]) axis = 2;

    auto centroid = [&](AABB const& b)->float {
        return 0.5f*(b.min[axis] + b.max[axis]);
    };

	int mid = start + count / 2;
    std::nth_element(
        items.begin() + start,
        items.begin() + mid,
        items.begin() + start + count,
        [&](auto const& A, auto const& B){ return centroid(A.first) < centroid(B.first); }
    );

	int left = build_bvh(items, start, mid - start);
    int right = build_bvh(items, mid, start + count - mid);

	bvh_nodes[node_idx].left = left;
    bvh_nodes[node_idx].right = right;
    bvh_nodes[node_idx].count = 0;
    return node_idx;
}

void SceneViewer::build_bvh_for_static() {
	std::vector<uint32_t> static_in = bvh_indices;
    bvh_indices.clear();
    bvh_nodes.clear();

	if (static_in.empty()) return;

    std::vector<std::pair<AABB,uint32_t>> items;
    items.reserve(static_in.size());

	for (uint32_t inst_idx : static_in) {
        ObjectInstance const& inst = object_instances[inst_idx];
        AABB const& box_local = mesh_aabb_local.at(inst.mesh);
        glm::mat4 W = to_glm(inst.transform.WORLD_FROM_LOCAL);
        AABB box_world = aabb_local_to_world(box_local, W);
        items.emplace_back(box_world, inst_idx);
    }

	build_bvh(items, 0, int(items.size()));
}

void SceneViewer::cull_with_bvh(glm::mat4 const& CLIP_FROM_WORLD_CULL_glm, std::vector<uint32_t> &draw_list){
	if (bvh_nodes.empty()) return;

    std::vector<int> stack;
    stack.reserve(bvh_nodes.size());
    stack.push_back(0);

	while (!stack.empty()) {
		int node_idx = stack.back();
        stack.pop_back();
		BVHNode const &node = bvh_nodes[node_idx];
		if (!aabb_intersects_frustum(node.box_world, CLIP_FROM_WORLD_CULL_glm)) continue;

		if (node.count > 0) {
			for (uint32_t k = 0; k < node.count; ++k) {
				uint32_t inst_idx = bvh_indices[node.start + k];

				ObjectInstance const &inst = object_instances[inst_idx];
				AABB const &box_local = mesh_aabb_local.at(inst.mesh);
				glm::mat4 W = to_glm(inst.transform.WORLD_FROM_LOCAL);
                glm::mat4 CLIP_FROM_LOCAL_glm = CLIP_FROM_WORLD_CULL_glm * W;
				if (aabb_intersects_frustum(box_local, CLIP_FROM_LOCAL_glm)) {
					// append_bvh_lines_world(box_local, W, 255, 255, 255);
                    draw_list.push_back(inst_idx);
                }
			}
		} else {
            if (node.left >= 0) stack.push_back(node.left);
            if (node.right >= 0) stack.push_back(node.right);
        }
	}
}