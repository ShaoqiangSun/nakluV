#include "Helpers.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct PrecomputedIBL {

    void load_environment_map(std::string env_cube_in_path);
    void precompute_ibl_diffuse_direct(uint32_t out_face_size = 32, uint32_t src_down_size = 256, std::string env_cube_out_path = "");
	void precompute_ibl_diffuse_monte_carlo(uint32_t out_face_size = 32, uint32_t samples = 1024, std::string env_cube_out_path = "");


    std::vector<glm::vec4> env_cube_rgba;
	uint32_t env_cube_face_size = 0;
	std::string env_cube_path;
};