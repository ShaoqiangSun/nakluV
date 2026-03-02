#include "PrecomputedIBL.hpp"
#include <string>


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


    return 0;
}