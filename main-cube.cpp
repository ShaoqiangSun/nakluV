#include "PrecomputedIBL.hpp"
#include <string>


int main(int argc, char **argv) {

    std::string cube_in_file = "";
    std::string lambertian_out_file = "";

    cube_in_file = argv[1];

    for (int argi = 2; argi < argc; ++argi) {
        std::string arg = argv[argi];

        if (arg == "--lambertian") {
			if (argi + 1 >= argc) throw std::runtime_error("--lambertian requires a parameter (a file name).");
			argi += 1;
			lambertian_out_file = argv[argi];
		}
    }

        

    PrecomputedIBL pre_ibl;

    pre_ibl.load_environment_map(cube_in_file);
    pre_ibl.precompute_ibl_diffuse_direct(32, 256, lambertian_out_file);

    return 0;
}