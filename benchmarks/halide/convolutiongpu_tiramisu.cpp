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
constexpr auto work_type = p_float32;
constexpr auto data_type = p_uint8;
constexpr auto block_size = 16;
#define BLOCK_SIZE "16"

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function convolutiongpu_tiramisu("convolutiongpu_tiramisu");

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), c("c");

    // Layer I

    computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &convolutiongpu_tiramisu};
    constant N{"N", sizes(0), it_type, true, nullptr, 0, &convolutiongpu_tiramisu};
    constant M{"M", sizes(1), it_type, true, nullptr, 0, &convolutiongpu_tiramisu};

    computation kernel{"{kernel[i, j]: 0 <= i < 3 and 0 <= j < 3}",
                       expr(), false, work_type, &convolutiongpu_tiramisu};
    computation in{"[N, M] -> {in[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                   expr(), false, data_type, &convolutiongpu_tiramisu};
    computation shared_in{"[N, M] -> {shared_in[i, o_i, j, o_j]: 0 <= i < N - 2 and 0 <= j < M - 2}",
                          expr(), false, work_type, &convolutiongpu_tiramisu};
    expr e = value_cast(work_type, 0);
    for (int _j = 0; _j < 3; _j ++)
    {
        for (int _i = 0; _i < 3; _i ++)
        {
            e = e + shared_in(i, _i, j, _j) * kernel(_i, _j);
        }
    }
    computation out{"[N, M] -> {out[i, j, c]: 0 <= i < N - 2 and 0 <= j < M - 2 and 0 <= c < 3}",
                    cast(data_type, e), true, data_type, &convolutiongpu_tiramisu};

    // Layer II
    out.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    out.tag_gpu_level(i0, j0, i1, j1);

    // Layer III
    buffer sizes_host{"sizes_host", {2}, it_type, a_input, &convolutiongpu_tiramisu};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");

    buffer kernel_host{"kernel_host", {3, 3}, work_type, a_input, &convolutiongpu_tiramisu};
    buffer kernel_gpu{"kernel_gpu", {3, 3}, work_type, a_temporary, &convolutiongpu_tiramisu};
    kernel_gpu.tag_gpu_constant();
    kernel.set_access("{kernel[i, j] -> kernel_gpu[j, i]}");

    buffer in_host{"in_host", {3, M, N}, data_type, a_input, &convolutiongpu_tiramisu};
    buffer in_gpu{"in_gpu", {3, M, N}, data_type, a_temporary, &convolutiongpu_tiramisu};
    in_gpu.tag_gpu_global();
    in.set_access("[N, M] -> {in[i, j, c] -> in_gpu[c, min(j, M - 1), min(i, N - 1)]}");

    buffer shared{"shared", {block_size + 2, block_size + 2}, work_type, a_temporary, &convolutiongpu_tiramisu};
    shared.tag_gpu_shared();
    shared_in.set_access("{shared_in[i, o_i, j, o_j] -> shared[i % " BLOCK_SIZE " + o_i, j % " BLOCK_SIZE" + o_j]}");

    buffer out_host{"out_host", {3, M - 2, N - 2}, data_type, a_output, &convolutiongpu_tiramisu};
    buffer out_gpu{"out_gpu", {3, M - 2, N - 2}, data_type, a_temporary, &convolutiongpu_tiramisu};
    out_gpu.tag_gpu_global();
    out.set_access("{out[i, j, c] -> out_gpu[c, j, i]}");


    // Layer IV
    computation copy_kernel{"{copy_kernel[0]}", memcpy(kernel_host, kernel_gpu), true, p_none, &convolutiongpu_tiramisu};
    computation copy_in{"{copy_in[0]}", memcpy(in_host, in_gpu), true, p_none, &convolutiongpu_tiramisu};
    copy_in.after(copy_kernel, computation::root);
    computation shared_dec{"[N, M] -> {shared_dec[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                           allocate(shared), true, data_type, &convolutiongpu_tiramisu};
    shared_dec.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    shared_dec.tag_gpu_level(i0, j0, i1, j1);
    shared_dec.after(copy_in, computation::root);
    computation shared_copy{"[N, M] -> {shared_copy[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                            cast(work_type, in(i, j, c)), true, data_type, &convolutiongpu_tiramisu};
    shared_copy.set_access("{shared_copy[i, j, c] -> shared[i, j]}");
    shared_copy.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    shared_copy.tag_gpu_level(i0, j0, i1, j1);
    shared_copy.after(shared_dec, c);

    computation shared_copy_right{"[N, M] -> {shared_copy_right[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                                  cast(work_type, in(i + block_size, j, c)), true, data_type, &convolutiongpu_tiramisu};
    shared_copy_right.add_predicate(i % block_size < 2);
    shared_copy_right.set_access("{shared_copy_right[i, j, c] -> shared[i + " BLOCK_SIZE ", j]}");
    shared_copy_right.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    shared_copy_right.after(shared_copy, c);

    computation shared_copy_bottom{"[N, M] -> {shared_copy_bottom[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                                   cast(work_type, in(i, j + block_size, c)), true, data_type, &convolutiongpu_tiramisu};
    shared_copy_bottom.add_predicate(j % block_size < 2);
    shared_copy_bottom.set_access("{shared_copy_bottom[i, j, c] -> shared[i, j + " BLOCK_SIZE "]}");
    shared_copy_bottom.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    shared_copy_bottom.after(shared_copy_right, c);


    computation shared_copy_diag{"[N, M] -> {shared_copy_diag[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                                 cast(work_type, in(i + block_size, j + block_size, c)), true, data_type, &convolutiongpu_tiramisu};
    shared_copy_diag.add_predicate(i % block_size < 2 && j % block_size < 2);
    shared_copy_diag.set_access("{shared_copy_diag[i, j, c] -> shared[i + " BLOCK_SIZE ", j + " BLOCK_SIZE "]}");
    shared_copy_diag.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    shared_copy_diag.after(shared_copy_bottom, c);

    computation synchronize1{"[N, M] -> {synchronize1[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                             tiramisu::sync{}, true, p_none, &convolutiongpu_tiramisu};
    synchronize1.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    synchronize1.after(shared_copy_diag, c);

    out.after(synchronize1, c);

    computation synchronize2{"[N, M] -> {synchronize2[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}",
                             tiramisu::sync{}, true, p_none, &convolutiongpu_tiramisu};
    synchronize2.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    synchronize2.after(out, c);

    computation copy_out{"{copy_out[0]}", memcpy(out_gpu, out_host), true, p_none, &convolutiongpu_tiramisu};
    copy_out.after(synchronize2, computation::root);


    convolutiongpu_tiramisu.set_arguments({&sizes_host, &in_host, &kernel_host, &out_host});
    convolutiongpu_tiramisu.gen_time_space_domain();
    convolutiongpu_tiramisu.gen_isl_ast();
    convolutiongpu_tiramisu.gen_cuda_stmt();
    convolutiongpu_tiramisu.gen_halide_stmt();
    convolutiongpu_tiramisu.dump_halide_stmt();
    convolutiongpu_tiramisu.gen_halide_obj("build/generated_fct_convolutiongpu.o");

    return 0;
}

