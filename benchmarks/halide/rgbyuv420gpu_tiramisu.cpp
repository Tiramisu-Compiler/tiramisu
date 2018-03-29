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
constexpr auto work_type = p_int32;
constexpr auto data_type = p_uint8;
constexpr auto block_size = 16;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function rgbyuv420gpu("rgbyuv420gpu_tiramisu");

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), c("c");

    // Layer I
    computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &rgbyuv420gpu};
    constant N{"N", sizes(0), it_type, true, nullptr, 0, &rgbyuv420gpu};
    constant M{"M", sizes(1), it_type, true, nullptr, 0, &rgbyuv420gpu};
    computation in{"[N, M] -> {in[i, j, c]: 0 <= i < N and 0 <= j < M and 0 <= c < 3}", expr(), false, data_type, &rgbyuv420gpu};
    computation y{"[N, M] -> {y[i, j]: 0 <= i < N and 0 <= j < M}",
                 cast(data_type, ((66 * cast(work_type, in(i, j, 0)) + 129 * cast(work_type, in(i, j, 1)) +  25 * cast(work_type, in(i, j, 2)) + 128) % 256) +  16),
                 true, data_type, &rgbyuv420gpu};
    computation copy_r{"[N, M] -> {copy_r[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", cast(work_type, in(2 * i, 2 * j, 0)), true, work_type, &rgbyuv420gpu};
    computation copy_g{"[N, M] -> {copy_g[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", cast(work_type, in(2 * i, 2 * j, 1)), true, data_type, &rgbyuv420gpu};
    computation copy_b{"[N, M] -> {copy_b[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", cast(work_type, in(2 * i, 2 * j, 2)), true, work_type, &rgbyuv420gpu};
    computation u{"[N, M] -> {u[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}",
                  cast(data_type, ((-38 * copy_r(i, j) - cast(work_type, 74 * copy_g(i, j)) +  112 * copy_b(i, j) + 128) % 256) +  128),
                  true, data_type, &rgbyuv420gpu};
    computation v{"[N, M] -> {v[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}",
                  cast(data_type, ((112 * copy_r(i, j) - cast(work_type, 94 * copy_g(i, j)) -   18 * copy_b(i, j) + 128) % 256) +  128),
                  true, data_type, &rgbyuv420gpu};

    // Layer II
    y.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    u.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    v.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    copy_r.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    copy_g.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    copy_b.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    y.tag_gpu_level(i0, j0, i1, j1);
    copy_r.after(y, computation::root);
    copy_r.tag_gpu_level(i0, j0, i1, j1);
    copy_g.after(copy_r, j1);
    copy_b.after(copy_g, j1);
    u.after(copy_b, j1);
    v.after(u, j1);

    // Layer III
    buffer sizes_host{"sizes_host", {2}, it_type, a_input, &rgbyuv420gpu};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");

    buffer in_host{"in_host", {3, M, N}, data_type, a_input, &rgbyuv420gpu};
    buffer y_host{"y_host", {M, N}, data_type, a_output, &rgbyuv420gpu};
    buffer u_host{"u_host", {M/2, N/2}, data_type, a_output, &rgbyuv420gpu};
    buffer v_host{"v_host", {M/2, N/2}, data_type, a_output, &rgbyuv420gpu};

    buffer in_gpu{"in_gpu", {3, M, N}, data_type, a_temporary, &rgbyuv420gpu};
    in_gpu.tag_gpu_global();
    in.set_access("{in[i, j, c] -> in_gpu[c, j, i]}");
    buffer y_gpu{"y_gpu", {M, N}, data_type, a_temporary, &rgbyuv420gpu};
    y_gpu.tag_gpu_global();
    y.set_access("{y[i, j] -> y_gpu[j, i]}");
    buffer u_gpu{"u_gpu", {M/2, N/2}, data_type, a_temporary, &rgbyuv420gpu};
    u.set_access("{u[i, j] -> u_gpu[j, i]}");
    u_gpu.tag_gpu_global();
    buffer v_gpu{"v_gpu", {M/2, N/2}, data_type, a_temporary, &rgbyuv420gpu};
    v.set_access("{v[i, j] -> v_gpu[j, i]}");
    v_gpu.tag_gpu_global();

    buffer copy_r_gpu{"copy_r_gpu", {1}, work_type, a_temporary, &rgbyuv420gpu};
    copy_r_gpu.tag_gpu_register();
    copy_r.set_access("{copy_r[i, j] -> copy_r_gpu[0]}");
    buffer copy_g_gpu{"copy_g_gpu", {1}, work_type, a_temporary, &rgbyuv420gpu};
    copy_g_gpu.tag_gpu_register();
    copy_g.set_access("{copy_g[i, j] -> copy_g_gpu[0]}");
    buffer copy_b_gpu{"copy_b_gpu", {1}, work_type, a_temporary, &rgbyuv420gpu};
    copy_b_gpu.tag_gpu_register();
    copy_b.set_access("{copy_b[i, j] -> copy_b_gpu[0]}");

    // Layer IV
    computation declare_copy_r{"[N, M] -> {declare_copy_r[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", expr(o_allocate, copy_r_gpu.get_name()), true, p_none, &rgbyuv420gpu};
    computation declare_copy_g{"[N, M] -> {declare_copy_g[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", expr(o_allocate, copy_g_gpu.get_name()), true, p_none, &rgbyuv420gpu};
    computation declare_copy_b{"[N, M] -> {declare_copy_b[i, j]: 0 <= 2 * i < N and 0 <= 2 * j < M}", expr(o_allocate, copy_b_gpu.get_name()), true, p_none, &rgbyuv420gpu};

    declare_copy_r.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    declare_copy_g.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    declare_copy_b.tile(i, j, block_size, block_size, i0, j0, i1, j1);

    declare_copy_r.tag_gpu_level(i0, j0, i1, j1);

    declare_copy_b.between(y, computation::root, copy_r, j1);
    declare_copy_g.between(y, computation::root, declare_copy_b, j1);
    declare_copy_r.between(y, computation::root, declare_copy_g, j1);

    computation copy_in{"{copy_in[0]}", expr(o_memcpy, var(in_host.get_name()), var(in_gpu.get_name())), true, p_none, &rgbyuv420gpu};
    computation copy_y{"{copy_y[0]}", expr(o_memcpy, var(y_gpu.get_name()), var(y_host.get_name())), true, p_none, &rgbyuv420gpu};
    computation copy_u{"{copy_u[0]}", expr(o_memcpy, var(u_gpu.get_name()), var(u_host.get_name())), true, p_none, &rgbyuv420gpu};
    computation copy_v{"{copy_v[0]}", expr(o_memcpy, var(v_gpu.get_name()), var(v_host.get_name())), true, p_none, &rgbyuv420gpu};

    copy_in.before(y, computation::root);
    copy_y.after(v, computation::root);
    copy_v.after(copy_y, computation::root);
    copy_u.after(copy_v, computation::root);


    rgbyuv420gpu.set_arguments({&sizes_host, &in_host, &y_host, &u_host, &v_host});
    rgbyuv420gpu.gen_time_space_domain();
    rgbyuv420gpu.gen_isl_ast();
    rgbyuv420gpu.gen_cuda_stmt();
    rgbyuv420gpu.gen_c_code();
    rgbyuv420gpu.gen_halide_stmt();
    rgbyuv420gpu.dump_halide_stmt();
    rgbyuv420gpu.gen_halide_obj("build/generated_fct_rgbyuv420gpu.o");

    return 0;
}
