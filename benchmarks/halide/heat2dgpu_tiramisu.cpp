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
constexpr auto block_size = 32;
constexpr auto block_size_small = 30;

#define BS "32"
#define BSS "30"


int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function heat2d_tiramisu("heat2dgpu_tiramisu");

    // Input params.
    float alpha = 0.3;
    float beta = 0.4;

    computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &heat2d_tiramisu};
    constant N{"N", sizes(0), it_type, true, nullptr, 0, &heat2d_tiramisu};
    constant M{"M", sizes(1), it_type, true, nullptr, 0, &heat2d_tiramisu};
    constant NB{"NB", N - 2, it_type, true, nullptr, 0, &heat2d_tiramisu};
    constant MB{"MB", M - 2, it_type, true, nullptr, 0, &heat2d_tiramisu};

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");

    computation in{"[N, M] -> {in[i, j]: 0 <= i < N and 0 <= j < M}", expr(), false, data_type, &heat2d_tiramisu};
    computation shared_in{"[N, M] -> {shared_in[i, o_i, j, o_j]: 0 <= i < N and 0 <= j < M}", expr(), false, data_type, &heat2d_tiramisu};
    computation out_init_i{"[N, M] -> {out_init_i[j, i]: 0 <= j < M and (i = 0 or i = M - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    computation out_init_j{"[N, M] -> {out_init_j[i, j]: 0 <= i < N and (j = 0 or j = N - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    expr comp_expr = shared_in(i, 0, j, 0) * alpha + (shared_in(i, 1, j, 0) + shared_in(i, 0, j, 1) + shared_in(i, -1, j, 0) + shared_in(i, 0, j, -1)) * beta;
    computation out_comp{"[NB, MB] -> {out_comp[i, j]: 0 <= i < NB and 0 <= j < MB}", comp_expr, true, data_type, &heat2d_tiramisu};

    buffer buff_shared{"buff_shared", {block_size, block_size}, data_type, a_temporary, &heat2d_tiramisu};
    buff_shared.tag_gpu_shared();
    computation shared_dec{"[NB, MB] -> {shared_dec[i, j] : 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}",
                           expr(o_allocate, buff_shared.get_name()), true, p_none, &heat2d_tiramisu};
    computation shared_init{"[NB, MB] -> {shared_init[i, j] : 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}",
                                   in(i, j),
                                   true, data_type, &heat2d_tiramisu};

    shared_dec.interchange(i, j);
    shared_init.interchange(i, j);
    out_comp.interchange(i, j);

    buffer buff_out_gpu{"buff_out_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    buff_out_gpu.tag_gpu_global();
    buffer buff_in_gpu{"buff_in_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    buff_in_gpu.tag_gpu_global();
    // in.set_access("{in[i,j] -> buff_in_gpu[(i / " BS ") * " BSS " + i % " BS ", (j / " BS ") * " BSS " + j % " BS "]}");
    in.set_access("{in[i,j] -> buff_in_gpu[i - 2 * floor(i / " BS "), j - 2 * floor(j / " BS ")]}");
    shared_init.set_access("{shared_init[i,j] -> buff_shared[i % " BS ", j % " BS "]}");
    shared_in.set_access("{shared_in[i, o_i, j, o_j] -> buff_shared[i % " BSS " + o_i + 1, j % " BSS " + o_j + 1]}");
    out_init_i.set_access("{out_init_i[j,i] -> buff_out_gpu[i,j]}");
    out_init_j.set_access("{out_init_j[i,j] -> buff_out_gpu[i,j]}");

    buffer sizes_b{"sizes_b", {2}, it_type, a_input, &heat2d_tiramisu};
    buffer buff_in{"buff_in", {N, M}, data_type, a_input, &heat2d_tiramisu};
    buffer buff_out{"buff_out", {N, M}, data_type, a_output, &heat2d_tiramisu};

    sizes.set_access("{sizes[i] -> sizes_b[i]}");
    out_comp.set_access("{out_comp[i,j] -> buff_out_gpu[i + 1,j + 1]}");

    out_comp.tile(j, i, block_size_small, block_size_small, i0, j0, i1, j1);
    shared_dec.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    shared_init.tile(j, i, block_size, block_size, i0, j0, i1, j1);


    computation synchronize{"[MB, NB] -> {synchronize[i, j]: 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}", tiramisu::sync{}, true, p_none, &heat2d_tiramisu};
    synchronize.interchange(i, j);
    synchronize.tile(j, i, block_size, block_size, i0, j0, i1, j1);


    out_init_i.before(out_init_j, computation::root);
    shared_dec.between(out_init_j, computation::root, out_comp, j1);
    shared_init.between(shared_dec, j1, out_comp, j1);

    synchronize.between(shared_init, j1, out_comp, j1);


    out_init_i.split(j, block_size, j0, j1);
    out_init_j.split(i, block_size, i0, i1);
    out_init_i.before(out_init_j, computation::root);
    out_init_i.tag_gpu_level(j0, j1);
    out_init_j.tag_gpu_level(i0, i1);
    shared_dec.tag_gpu_level(i0, j0, i1, j1);

    computation copy_to_device{"{copy_to_device[0]}", expr(o_memcpy, var(buff_in.get_name()), var(buff_in_gpu.get_name())), true, p_none, &heat2d_tiramisu};
    computation copy_to_host{"{copy_to_host[0]}", expr(o_memcpy, var(buff_out_gpu.get_name()), var(buff_out.get_name())), true, p_none, &heat2d_tiramisu};

    copy_to_device.before(out_init_i, computation::root);
    copy_to_host.after(out_comp, computation::root);

    



    heat2d_tiramisu.set_arguments({&sizes_b, &buff_in, &buff_out});
    heat2d_tiramisu.gen_time_space_domain();
    heat2d_tiramisu.gen_isl_ast();
    heat2d_tiramisu.gen_cuda_stmt();
    heat2d_tiramisu.gen_halide_stmt();
    heat2d_tiramisu.dump_halide_stmt();
    heat2d_tiramisu.gen_halide_obj("build/generated_fct_heat2dgpu.o");

    return 0;
}
