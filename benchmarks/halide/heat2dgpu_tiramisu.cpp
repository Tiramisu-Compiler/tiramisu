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
    computation out_init_i{"[N, M] -> {out_init_i[j, i]: 0 <= j < M and (i = 0 or i = M - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    computation out_init_j{"[N, M] -> {out_init_j[i, j]: 0 <= i < N and (j = 0 or j = N - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    // computation out_init_j{"[N, M] -> {out_init[i, j]: 0 <= i < N and (0 = j  or j = M - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    expr comp_expr = in(i, j) * alpha + (in(i + 1, j) + in(i, j + 1) + in(i - 1, j) + in(i, j - 1)) * beta;
    computation out_comp{"[NB, MB] -> {out_comp[i, j]: 0 <= i < NB and 0 <= j < MB}", comp_expr, true, data_type, &heat2d_tiramisu};

    // out_comp.after(out_init_i, computation::root);

    buffer sizes_b{"sizes_b", {2}, it_type, a_input, &heat2d_tiramisu};
    buffer buff_in{"buff_in", {N, M}, data_type, a_input, &heat2d_tiramisu};
    buffer buff_out{"buff_out", {N, M}, data_type, a_output, &heat2d_tiramisu};

    sizes.set_access("{sizes[i] -> sizes_b[i]}");
    in.set_access("{in[i,j] -> buff_in[i,j]}");
    out_init_i.set_access("{out_init_i[j,i] -> buff_out[i,j]}");
    out_init_j.set_access("{out_init_j[i,j] -> buff_out[i,j]}");
    // out_init_j.set_access("{out_init_j[i,j] -> buff_out[i,j]}");
    out_comp.set_access("{out_comp[i,j] -> buff_out[i,j]}");

    out_comp.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation in_gpu{"[N, M] -> {in_gpu[i, j]: 0 <= i + 1 < N and 0 <= j + 1< M}", expr(), false, data_type, &heat2d_tiramisu};

    buffer buff_out_gpu{"buff_out_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    buff_out_gpu.tag_gpu_global();
    buffer buff_in_gpu{"buff_in_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    buff_in_gpu.tag_gpu_global();

    buffer buff_shared{"buff_shared", {block_size + 2, block_size + 2}, data_type, a_temporary, &heat2d_tiramisu};
    buff_shared.tag_gpu_shared();
    auto i_str = "i0 * " + std::to_string(block_size) + " + i1";
    auto j_str = "j0 * " + std::to_string(block_size) + " + j1";
    auto i0_j0_cond = "0 <= i0 * " + std::to_string(block_size) + " < NB and 0 <= j0 * " + std::to_string(block_size) + " < MB";
    auto i1_j1_cond = "0 <= i1 < " + std::to_string(block_size) + " and 0 <= j1 < " + std::to_string(block_size);
    auto whole_condition = i0_j0_cond + " and " + i1_j1_cond + " and 1 <= " + i_str + " < NB and 1 <= " + j_str + " < MB";
    computation shared_dec{"[NB, MB] -> {shared_init_center[i, j] : 0 <= i < NB and 0 <= j < MB}",
                           expr(o_allocate, buff_shared.get_name()), true, p_none, &heat2d_tiramisu};

    shared_dec.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation shared_init_top_edge{"[NB, MB] -> {shared_init_top_edge[i, j] : 0 <= i < NB and 0 <= j < MB and (i % " + std::to_string(block_size) + " = 0)}",
                                     in_gpu(i - 1, j),
                                     true, data_type, &heat2d_tiramisu};
    shared_init_top_edge.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation shared_init_left_edge{"[NB, MB] -> {shared_init_left_edge[i, j] : 0 <= i < NB and 0 <= j < MB and (j % " + std::to_string(block_size) + " = 0)}",
                                     in_gpu(i, j - 1),
                                     true, data_type, &heat2d_tiramisu};
    shared_init_left_edge.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation shared_init_bot_edge{"[NB, MB] -> {shared_init_bot_edge[i, j] : 0 <= i < NB and 0 <= j < MB and ((i + 1) % " + std::to_string(block_size) + " = 0 or i + 1 = NB)}",
                                     in_gpu(i + 1, j),
                                     true, data_type, &heat2d_tiramisu};
    shared_init_bot_edge.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation shared_init_right_edge{"[NB, MB] -> {shared_init_right_edge[i, j] : 0 <= i < NB and 0 <= j < MB and ((j + 1) % " + std::to_string(block_size) + " = 0 or j + 1 = MB)}",
                                     in_gpu(i, j + 1),
                                     true, data_type, &heat2d_tiramisu};
    shared_init_right_edge.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    computation shared_init_center{"[NB, MB] -> {shared_init_center[i, j] : 0 <= i < NB and 0 <= j < MB}",
                                   in_gpu(i, j),
                                   true, data_type, &heat2d_tiramisu};
    shared_init_center.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    shared_init_top_edge.set_access(
            "{  shared_init_top_edge[i, j] -> buff_shared[0,                                          j % " + std::to_string(block_size) + " + 1]}");
    shared_init_bot_edge.set_access(
            "{  shared_init_bot_edge[i, j] -> buff_shared[i % " + std::to_string(block_size) + " + 2, j % " + std::to_string(block_size) + " + 1]}");
    shared_init_left_edge.set_access(
            "{ shared_init_left_edge[i, j] -> buff_shared[i % " + std::to_string(block_size) + " + 1, 0                                         ]}");
    shared_init_right_edge.set_access(
            "{shared_init_right_edge[i, j] -> buff_shared[i % " + std::to_string(block_size) + " + 1, j % " + std::to_string(block_size) + " + 2]}");
    shared_init_center.set_access(
            "{    shared_init_center[i, j] -> buff_shared[i % " + std::to_string(block_size) + " + 1, j % " + std::to_string(block_size) + " + 1]}");

    computation shared_buff{"[MB, NB] -> {shared_buff[i, o_i, j, o_j]: 0 <= i < MB and 0 <= j < NB}",
                            expr(), false, data_type, &heat2d_tiramisu};
    shared_buff.set_access("{shared_buff[i, o_i, j, o_j] -> buff_shared[i % " + std::to_string(block_size) + " + o_i + 1, j % " + std::to_string(block_size) + " + o_j + 1]}");
    comp_expr = shared_buff(i, 0, j, 0) * alpha + (shared_buff(i, 1, j, 0) + shared_buff(i, 0, j, 1) + shared_buff(i, -1, j, 0) + shared_buff(i, 0, j, -1)) * beta;
    out_comp.set_expression(comp_expr);

    computation synchronize{"[MB, NB] -> {synchronize[i, j]: 0 <= i < NB and 0 <= j < MB}", tiramisu::sync{}, true, p_none, &heat2d_tiramisu};
    synchronize.tile(i, j, block_size, block_size, i0, j0, i1, j1);


    out_init_i.before(out_init_j, computation::root);
    shared_dec.between(out_init_j, computation::root, out_comp, j1);
    shared_init_top_edge.between(shared_dec, j1, out_comp, j1);
    shared_init_bot_edge.between(shared_init_top_edge, j1, out_comp, j1);
    shared_init_left_edge.between(shared_init_bot_edge, j1, out_comp, j1);
    shared_init_right_edge.between(shared_init_left_edge, j1, out_comp, j1);
    shared_init_center.between(shared_init_right_edge, j1, out_comp, j1);

    synchronize.between(shared_init_center, j1, out_comp, j1);

    in_gpu.set_access("{in_gpu[i,j] -> buff_in_gpu[i + 1, j + 1]}");
    out_init_i.set_access("{out_init_i[j,i] -> buff_out_gpu[i,j]}");
    out_init_j.set_access("{out_init_j[i,j] -> buff_out_gpu[i,j]}");
    // out_init_j.set_access("{out_init_j[i,j] -> buff_out_gpu[i,j]}");
    out_comp.set_access("{out_comp[i,j] -> buff_out_gpu[i + 1,j + 1]}");
    out_comp.set_expression(out_comp.get_expr().substitute_access(in.get_name(), in_gpu.get_name()));

    out_init_i.split(j, block_size, j0, j1);
    out_init_j.split(i, block_size, i0, i1);
    //out_init_i.before(out_init_j, computation::root);
    out_init_i.tag_gpu_level(j0, j1);
    out_init_j.tag_gpu_level(i0, i1);
    shared_init_top_edge.tag_gpu_level(i0, j0, i1, j1);
    shared_init_bot_edge.tag_gpu_level(i0, j0, i1, j1);
    shared_init_left_edge.tag_gpu_level(i0, j0, i1, j1);
    shared_init_right_edge.tag_gpu_level(i0, j0, i1, j1);

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
