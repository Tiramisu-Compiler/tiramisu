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
constexpr auto block_size = 16;
#define BS "16"

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function fusiongpu_tiramisu("fusiongpu_tiramisu");

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), c("c");

    tiramisu::computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &fusiongpu_tiramisu};
    tiramisu::constant N{"N", sizes(0), it_type, true, nullptr, 0, &fusiongpu_tiramisu};
    tiramisu::constant M{"M", sizes(1), it_type, true, nullptr, 0, &fusiongpu_tiramisu};
    tiramisu::computation in{"[N, M] -> {in[i, j, c]}", expr(), false, data_type, &fusiongpu_tiramisu};
    tiramisu::computation f_const{"[N, M] -> {f_const[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", 255 - in(i, j, c), true, data_type, &fusiongpu_tiramisu};
    tiramisu::computation g_const{"[N, M] -> {g_const[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", 2 * in(i, j, c), true, data_type, &fusiongpu_tiramisu};
    tiramisu::computation f{"[N, M] -> {f[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", f_const(i, j, c), true, data_type, &fusiongpu_tiramisu};
    tiramisu::computation g{"[N, M] -> {g[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", g_const(i, j, c), true, data_type, &fusiongpu_tiramisu};
    tiramisu::computation h{"[N, M] -> {h[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", f_const(i, j, c) + g_const(i, j, c), true, data_type, &fusiongpu_tiramisu};
    tiramisu::computation k{"[N, M] -> {k[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", f_const(i, j, c) - g_const(i, j, c), true, data_type, &fusiongpu_tiramisu};

    f_const.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    g_const.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    f.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    g.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    h.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    k.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    buffer sizes_host{"sizes_host", {2}, it_type, a_input, &fusiongpu_tiramisu};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");
    buffer in_host{"in_host", {3, M, N}, data_type, a_input, &fusiongpu_tiramisu};
    buffer in_gpu{"in_gpu", {3, M, N}, data_type, a_temporary, &fusiongpu_tiramisu};
    in_gpu.tag_gpu_global();
    in.set_access("{in[i, j, c] -> in_gpu[c, j, i]}");

    buffer f_const_gpu{"f_const_gpu", {1}, data_type, a_temporary, &fusiongpu_tiramisu};
    f_const_gpu.tag_gpu_register();
    f_const.set_access("{f_const[i, j, c] -> f_const_gpu[0]}");
    buffer g_const_gpu{"g_const_gpu", {1}, data_type, a_temporary, &fusiongpu_tiramisu};
    g_const_gpu.tag_gpu_register();
    g_const.set_access("{g_const[i, j, c] -> g_const_gpu[0]}");
    buffer f_gpu{"f_gpu", {3, M, N}, data_type, a_temporary, &fusiongpu_tiramisu};
    f_gpu.tag_gpu_global();
    f.set_access("{f[i, j, c] -> f_gpu[c, j, i]}");
    buffer g_gpu{"g_gpu", {3, M, N}, data_type, a_temporary, &fusiongpu_tiramisu};
    g_gpu.tag_gpu_global();
    g.set_access("{g[i, j, c] -> g_gpu[c, j, i]}");
    buffer h_gpu{"h_gpu", {3, M, N}, data_type, a_temporary, &fusiongpu_tiramisu};
    h_gpu.tag_gpu_global();
    h.set_access("{h[i, j, c] -> h_gpu[c, j, i]}");
    buffer k_gpu{"k_gpu", {3, M, N}, data_type, a_temporary, &fusiongpu_tiramisu};
    k_gpu.tag_gpu_global();
    k.set_access("{k[i, j, c] -> k_gpu[c, j, i]}");
    buffer f_host{"f_host", {3, M, N}, data_type, a_output, &fusiongpu_tiramisu};
    buffer g_host{"g_host", {3, M, N}, data_type, a_output, &fusiongpu_tiramisu};
    buffer h_host{"h_host", {3, M, N}, data_type, a_output, &fusiongpu_tiramisu};
    buffer k_host{"k_host", {3, M, N}, data_type, a_output, &fusiongpu_tiramisu};

    tiramisu::computation f_const_dec{"[N, M] -> {f_const_dec[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", allocate(f_const_gpu), true, p_none, &fusiongpu_tiramisu};
    f_const_dec.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    tiramisu::computation g_const_dec{"[N, M] -> {g_const_dec[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", allocate(g_const_gpu), true, p_none, &fusiongpu_tiramisu};
    g_const_dec.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    f_const_dec.tag_gpu_level(i0, j0, i1, j1);

    k.after(h, c);
    h.after(g, c);
    g.after(f, c);
    f.after(g_const, c);
    g_const.after(f_const, c);
    f_const.after(g_const_dec, c);
    g_const_dec.after(f_const_dec, c);

    tiramisu::computation copy_to_device{"{copy_to_device[0]}", memcpy(in_host, in_gpu), true, p_none, &fusiongpu_tiramisu};
    tiramisu::computation copy_f_to_host{"{copy_f_to_host[0]}", memcpy(f_gpu, f_host), true, p_none, &fusiongpu_tiramisu};
    tiramisu::computation copy_g_to_host{"{copy_g_to_host[0]}", memcpy(g_gpu, g_host), true, p_none, &fusiongpu_tiramisu};
    tiramisu::computation copy_h_to_host{"{copy_h_to_host[0]}", memcpy(h_gpu, h_host), true, p_none, &fusiongpu_tiramisu};
    tiramisu::computation copy_k_to_host{"{copy_k_to_host[0]}", memcpy(k_gpu, k_host), true, p_none, &fusiongpu_tiramisu};

    copy_to_device.before(f_const_dec, computation::root);
    copy_f_to_host.after(k, computation::root);
    copy_g_to_host.after(copy_f_to_host, computation::root);
    copy_h_to_host.after(copy_g_to_host, computation::root);
    copy_k_to_host.after(copy_h_to_host, computation::root);


    fusiongpu_tiramisu.set_arguments({&sizes_host, &in_host, &f_host, &g_host, &h_host, &k_host});
    fusiongpu_tiramisu.gen_time_space_domain();
    fusiongpu_tiramisu.gen_isl_ast();
    fusiongpu_tiramisu.gen_cuda_stmt();
    fusiongpu_tiramisu.gen_halide_stmt();
    fusiongpu_tiramisu.dump_halide_stmt();
    fusiongpu_tiramisu.gen_halide_obj("build/generated_fct_fusiongpu.o");


    return 0;
}

