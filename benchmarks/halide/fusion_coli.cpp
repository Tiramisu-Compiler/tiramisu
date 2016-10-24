#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    int  SIZE0 = 0;
    int  SIZE1 = 0;


    // COLi generated code here.


    // Generate code
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion.o");

    // Some debugging
    fusion_coli.dump_iteration_domain();
    fusion_coli.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    fusion_coli.dump(true);

    return 0;
}
