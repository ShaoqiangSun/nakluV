#include "PrecomputedIBL.hpp"
#include <string>
#include <iostream>


int main(int argc, char **argv) {

    std::string cube_in_file = "";
    std::string lambertian_out_file = "";
    std::string ggx_out_file = "";
    std::string brdf_out_file = "";

    cube_in_file = argv[1];

    for (int argi = 2; argi < argc; ++argi) {
        std::string arg = argv[argi];

        if (arg == "--lambertian") {
			if (argi + 1 >= argc) throw std::runtime_error("--lambertian requires a parameter (a file name).");
			argi += 1;
			lambertian_out_file = argv[argi];
		}
        else if (arg == "--ggx") {
            if (argi + 1 >= argc) throw std::runtime_error("--ggx requires a parameter (a file name).");
            argi += 1;
            ggx_out_file = argv[argi];
        }
        else if (arg == "--brdf-lut") {
            if (argi + 1 >= argc)
                throw std::runtime_error("--brdf-lut requires a parameter (a file name).");
            argi += 1;
            brdf_out_file = argv[argi];
        }
    }

        

    PrecomputedIBL pre_ibl;

    if (!cube_in_file.empty()) pre_ibl.load_environment_map(cube_in_file);
    if (!lambertian_out_file.empty()) pre_ibl.precompute_ibl_diffuse_direct(32, 256, lambertian_out_file);
    if (!ggx_out_file.empty()) pre_ibl.precompute_ibl_specular_ggx_mip(1024, ggx_out_file);
    if (!brdf_out_file.empty()) pre_ibl.precompute_brdf_lut(256, 1024, brdf_out_file);


    // std::vector<uint32_t> sizes = {4, 8, 16, 32, 64};

    // std::cout << "===== Lambertian Timing =====" << std::endl;

    // for (uint32_t size : sizes) {

    //     auto start = std::chrono::high_resolution_clock::now();

    //     pre_ibl.precompute_ibl_diffuse_direct(size, size * 4, "");

    //     auto end = std::chrono::high_resolution_clock::now();

    //     float time_ms = std::chrono::duration<float, std::milli>(end - start).count();

    //     std::cout << "Size: " << size << "  Time: " << time_ms << " ms" << std::endl;
    // }

    return 0;
}