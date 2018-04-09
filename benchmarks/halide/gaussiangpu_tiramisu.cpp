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
constexpr auto data_type = p_uint8;
constexpr auto work_type = p_float32;
constexpr auto bs = 32;
#define BS "32"

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function gaussian_tiramisu("gaussiangpu_tiramisu");

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), c("c");

    computation sizes{"{sizes[i]: 0 <= i < 3}", expr(), false, it_type, &gaussian_tiramisu};
    constant N{"N", sizes(0), it_type, true, nullptr, computation::root_dimension, &gaussian_tiramisu};
    constant M{"M", sizes(1), it_type, true, nullptr, computation::root_dimension, &gaussian_tiramisu};
    constant NB{"NB", N - 5, it_type, true, nullptr, computation::root_dimension, &gaussian_tiramisu};
    constant MB{"MB", M - 5, it_type, true, nullptr, computation::root_dimension, &gaussian_tiramisu};
    constant C{"C", sizes(2), it_type, true, nullptr, computation::root_dimension, &gaussian_tiramisu};
    computation gpu_input{"[N, M, C] -> {gpu_input[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < C}", expr(), false, data_type, &gaussian_tiramisu};
    computation kernel_x{"{kernel_x[i]: 0 <= i < 5}", expr(), false, work_type, &gaussian_tiramisu};
    computation kernel_y{"{kernel_y[i]: 0 <= i < 5}", expr(), false, work_type, &gaussian_tiramisu};

    tiramisu::expr e = value_cast(work_type, 0);
    for (int k = 0; k < 5; k++)
        e = e + cast(work_type, gpu_input(i + k, j, c)) * kernel_x(k);
    computation gaussian_x{"[NB, MB, C] -> {gaussian_x[i, j, c]: 0 <= i < NB and 0 <= j < " EXTRA_J " and 0 <= c < C}", e, false, work_type, &gaussian_tiramisu};
    gaussian_x.set_inline();

    e = value_cast(work_type, 0);
    for (int k = 0; k < 5; k++)
        e = e + gaussian_x(i, j + k, c) * kernel_y(k);
    e = cast(data_type, e);
    computation gaussian_y{"[NB, MB, C] -> {gaussian_y[i, j, c]: 0 <= i < NB and 0 <= j < MB and 0 <= c < C}", e, true, data_type, &gaussian_tiramisu};


    gaussian_y.tile(i, j, bs, bs, i0, j0, i1, j1);

    gaussian_y.tag_gpu_level(i0, j0, i1, j1);

    buffer sizes_b{"sizes_b", {3}, it_type, a_input, &gaussian_tiramisu};
    buffer input{"input", {C, M, N}, data_type, a_input, &gaussian_tiramisu};
    buffer output{"output", {C, MB, NB}, data_type, a_output, &gaussian_tiramisu};
    sizes.set_access("{sizes[i] -> sizes_b[i]}");
    buffer kxo{"kxo", {5}, work_type, a_input, &gaussian_tiramisu};
    buffer kx{"kx", {5}, work_type, a_temporary, &gaussian_tiramisu};
    kx.tag_gpu_constant();
    buffer kyo{"kyo", {5}, work_type, a_input, &gaussian_tiramisu};
    buffer ky{"ky", {5}, work_type, a_temporary, &gaussian_tiramisu};
    ky.tag_gpu_constant();
    kernel_x.set_access("{kernel_x[i] -> kx[i]}");
    kernel_y.set_access("{kernel_y[i] -> ky[i]}");


    buffer gpu_in{"gpu_in", {C, M, N}, data_type, a_temporary, &gaussian_tiramisu};
    gpu_in.tag_gpu_global();
    gpu_input.set_access("{gpu_input[i, j, c] -> gpu_in[c, j, i]}");
    buffer gpu_out{"gpu_out", {C, MB, NB}, data_type, a_temporary, &gaussian_tiramisu};
    gpu_out.tag_gpu_global();

    gaussian_y.set_access("{gaussian_y[i, j, c] -> gpu_out[c, j, i]}");





    computation copy_to_gpu_in{"{copy_to_gpu_in[0]}", expr(o_memcpy, var(input.get_name()), var(gpu_in.get_name())), true, p_none, &gaussian_tiramisu};
    computation copy_kx{"{copy_kx[0]}", expr(o_memcpy, var(kxo.get_name()), var(kx.get_name())), true, p_none, &gaussian_tiramisu};
    computation copy_ky{"{copy_ky[0]}", expr(o_memcpy, var(kyo.get_name()), var(ky.get_name())), true, p_none, &gaussian_tiramisu};
    computation copy_to_out{"{copy_to_out[0]}", expr(o_memcpy, var(gpu_out.get_name()), var(output.get_name())), true, p_none, &gaussian_tiramisu};

    copy_kx.between(copy_to_gpu_in, computation::root, copy_ky, computation::root);
    copy_ky.before(gaussian_y, computation::root);
    copy_to_out.after(gaussian_y, computation::root);


    gaussian_tiramisu.set_arguments({&sizes_b, &input, &kxo, &kyo, &output});
    gaussian_tiramisu.gen_time_space_domain();
    gaussian_tiramisu.gen_isl_ast();
    gaussian_tiramisu.gen_c_code();
    gaussian_tiramisu.gen_cuda_stmt();
    gaussian_tiramisu.gen_halide_stmt();
    gaussian_tiramisu.dump_halide_stmt();
    gaussian_tiramisu.gen_halide_obj("build/generated_fct_gaussiangpu.o");

    return 0;
}
