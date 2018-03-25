#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace tiramisu;

constexpr auto it_type = p_int32;
constexpr auto data_type = p_float32;
constexpr auto block_size = 16;
constexpr auto kern_size = 5;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function gaussian_tiramisu("gaussiangpu_tiramisu");

    computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &gaussian_tiramisu};


    gaussian_tiramisu.set_arguments({});
    gaussian_tiramisu.gen_time_space_domain();
    gaussian_tiramisu.gen_isl_ast();
    gaussian_tiramisu.gen_cuda_stmt();
    gaussian_tiramisu.gen_halide_stmt();
    gaussian_tiramisu.dump_halide_stmt();
    gaussian_tiramisu.gen_halide_obj("build/generated_fct_gaussiangpu.o");

    return 0;
}
