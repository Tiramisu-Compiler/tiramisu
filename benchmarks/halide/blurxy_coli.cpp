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

    // A hack to represent the image size.  TODO: this should be retrieved from
    // the function arguments.
    Halide::Image<uint16_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int  SIZE0 = in_image.extent(0) - 8;
    int  SIZE1 = in_image.extent(1) - 2;

    #include "blurxy_algorithm.cpp"

    blurxy.set_context_set("[N,M]->{: N>1 and M>1}]");
    c_blurx.set_schedule("[N,M,NM]->{c_blurx[i,j]->[i,0,j1,j2]: 0<i<N and 0<j<NM and j1=floor(j/8) and j2=j%8 and NM%8=0; c_blurx[i,j]->[i,2,j,0]: 0<i<N and NM<j<M}");
    c_blury.set_schedule("[N,M,NM]->{c_blury[i,j]->[i,1,j1,j2]: 0<i<N and 0<j<NM and j1=floor(j/8) and j2=j%8 and NM%8=0; c_blury[i,j]->[i,3,j,0]: 0<i<N and NM<j<M}");
    c_blury.tag_parallel_dimension(0);
   // c_blury.tag_vector_dimension(2);
   // c_blurx.tag_vector_dimension(2);

    // Generate code
    blurxy.gen_isl_ast();
    blurxy.gen_halide_stmt();
    blurxy.gen_halide_obj("build/generated_fct_blurxy.o");

    // Some debugging
    blurxy.dump_iteration_domain();
    blurxy.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    blurxy.dump(true);

    return 0;
}
