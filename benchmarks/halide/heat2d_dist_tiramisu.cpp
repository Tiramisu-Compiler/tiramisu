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
#include "wrapper_heat2d_dist.h"

using namespace tiramisu;

constexpr auto it_type = p_int32;
constexpr auto data_type = p_float32;
constexpr auto block_size = 32;
constexpr auto block_size_small = 30;

#ifdef USE_GPU2
#define BS "32"
#define BSS "30"
#else
#define BS "1"
#define BSS "1"
#endif

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

#ifdef USE_GPU2
    tiramisu::function heat2d_tiramisu("heat2dgpu_tiramisu");
#else
    tiramisu::function heat2d_tiramisu("heat2d_dist_tiramisu");
#endif

    // Input params.
    float alpha = 0.3;
    float beta = 0.4;
    constant("N", N, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("CRANKS", NUM_CPU_RANKS, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant NB{"NB", N - 1, it_type, true, nullptr, 0, &heat2d_tiramisu};

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");

    computation in{"[N] -> {in[i, j]: 0 <= i < N and 0 <= j < N}", expr(), false, data_type, &heat2d_tiramisu};
#ifdef USE_GPU2
    computation shared_in{"[N, M] -> {shared_in[i, o_i, j, o_j]: 0 <= i < N and 0 <= j < M}", expr(), false, data_type, &heat2d_tiramisu};
    //    computation out_init_i{"[N] -> {out_init_i[j, i]: 0 <= j < N and (i = 0 or i = N - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    //    computation out_init_j{"[N] -> {out_init_j[i, j]: 0 <= i < N and (j = 0 or j = N - 1)}", 0.f, true, data_type, &heat2d_tiramisu};
    expr comp_expr = shared_in(i, 0, j, 0) * alpha + (shared_in(i, 1, j, 0) + shared_in(i, 0, j, 1) + shared_in(i, -1, j, 0) + shared_in(i, 0, j, -1)) * beta;
#else
    computation init("[NB]->{init[i,j]: 1<=i<NB and 1<=j<NB}", expr(0.0f), true, data_type, &heat2d_tiramisu);
    expr comp_expr = in(i,j) * alpha + (in(i-1, j) + in(i+1, j) + in(i, j-1) + in(i, j+1)) * beta;
#endif
    computation out_comp{"[NB] -> {out_comp[i, j]: 1 <= i < NB and 1 <= j < NB}", comp_expr, true, data_type, &heat2d_tiramisu};  

#ifdef USE_GPU2
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
    in.set_access("{in[i,j] -> buff_in_gpu[i - 2 * floor(i / " BS "), j - 2 * floor(j / " BS ")]}");
    shared_init.set_access("{shared_init[i,j] -> buff_shared[i % " BS ", j % " BS "]}");
    shared_in.set_access("{shared_in[i, o_i, j, o_j] -> buff_shared[i % " BSS " + o_i + 1, j % " BSS " + o_j + 1]}");
    //    out_init_i.set_access("{out_init_i[j,i] -> buff_out_gpu[i,j]}");
    //    out_init_j.set_access("{out_init_j[i,j] -> buff_out_gpu[i,j]}");
#endif

    // split into the number of ranks by distributing
    var i2("i2");
    if (NUM_CPU_RANKS + NUM_GPU_RANKS > 1) {
      init.shift(i, -1);
      out_comp.shift(i, -1);
      init.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      out_comp.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
    }
    // insert communication
    var q("q");
    // TODO collapse this send
    // This communication sends a row backwords (from rank q to rank q-1)
    xfer send_recv_bkwd = computation::create_xfer("[N,CRANKS]->{send_bkwd[q,i,j]: 1<=q<CRANKS and 0<=i<1 and 0<=j<N}", "[N,CRANKS]->{recv_bkwd[q,i,j]: 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<N}", q-1, q+1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}), xfer_prop(data_type, {SYNC, BLOCK, MPI}), in(i,j), &heat2d_tiramisu);
    send_recv_bkwd.s->collapse_many({collapse_group(2,0,N-2), collapse_group(1,0,1)});
    send_recv_bkwd.r->collapse_many({collapse_group(2,0,N-2), collapse_group(1,0,1)});
    // This communication sends a row forwards (from rank q to q+1);
    xfer send_recv_fwd = computation::create_xfer("[N,CRANKS]->{send_fwd[q,i,j]: 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<N}", "[N,CRANKS]->{recv_fwd[q,i,j]: 1<=q<CRANKS and 0<=i<1 and 0<=j<N}", q+1, q-1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}), xfer_prop(data_type, {SYNC, BLOCK, MPI}), in(i,j), &heat2d_tiramisu);
    send_recv_fwd.s->collapse_many({collapse_group(2,0,N-2), collapse_group(1,0,1)});
    send_recv_fwd.r->collapse_many({collapse_group(2,0,N-2), collapse_group(1,0,1)});
    
#ifndef USE_GPU2
    // Optimize the CPU sections
    // don't tag out_comp too because the fusion makes the wrong stuff get parallelized
    init.tag_parallel_level(i2);    
#endif    
    out_comp.tag_distribute_level(i1);
    init.drop_rank_iter();
    out_comp.drop_rank_iter();
    send_recv_bkwd.s->tag_distribute_level(q);
    send_recv_bkwd.r->tag_distribute_level(q);
    send_recv_fwd.s->tag_distribute_level(q);
    send_recv_fwd.r->tag_distribute_level(q);
    // Need to send one row back also
    // Need another set of computations to compute border

    buffer buff_in{"buff_in", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS) + 2, N}, data_type, a_input, &heat2d_tiramisu};
    buffer buff_out{"buff_out", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS), N}, data_type, a_output, &heat2d_tiramisu};
#ifdef USE_GPU2
    out_comp.set_access("{out_comp[i,j] -> buff_out_gpu[i + 1,j + 1]}");
    in.set_access("{in[i,j] -> buff_in[i - 2 * floor(i / " BS "), j - 2 * floor(j / " BS ")]}");
#else
    in.set_access("{in[i,j]->buff_in[i,j]}");
    init.set_access("{init[i,j]->buff_out[i,j]}");
    out_comp.set_access("{out_comp[i,j] -> buff_out[i,j]}");
    send_recv_bkwd.r->set_access("{recv_bkwd[q,i,j]->buff_in[i+" + std::to_string(N/(NUM_CPU_RANKS+NUM_GPU_RANKS)) + ",j]}");
    send_recv_fwd.r->set_access("{recv_fwd[q,i,j]->buff_in[i,j]}");
#endif

#ifdef USE_GPU2
    out_comp.tile(j, i, block_size_small, block_size_small, i0, j0, i1, j1);
    shared_dec.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    shared_init.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    computation synchronize{"[MB, NB] -> {synchronize[i, j]: 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}", tiramisu::sync{}, true, p_none, &heat2d_tiramisu};
    synchronize.interchange(i, j);
    synchronize.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    //    out_init_i.before(out_init_j, computation::root);
    //    shared_dec.between(out_init_j, computation::root, out_comp, j1);
    shared_init.between(shared_dec, j1, out_comp, j1);
    synchronize.between(shared_init, j1, out_comp, j1);
#else
    send_recv_fwd.s->before(*send_recv_fwd.r, computation::root);
    send_recv_fwd.r->before(*send_recv_bkwd.s, computation::root);
    send_recv_bkwd.s->before(*send_recv_bkwd.r, computation::root);
    send_recv_bkwd.r->before(init, computation::root);
    init.before(out_comp, j);//computation::root);
#endif

#ifdef USE_GPU2
    //    out_init_i.split(j, block_size, j0, j1);
    //    out_init_j.split(i, block_size, i0, i1);
    //    out_init_i.before(out_init_j, computation::root);
    //    out_init_i.tag_gpu_level(j0, j1);
    //    out_init_j.tag_gpu_level(i0, i1);
    shared_dec.tag_gpu_level(i0, j0, i1, j1);
    computation copy_to_device{"{copy_to_device[0]}", expr(o_memcpy, var(buff_in.get_name()), var(buff_in_gpu.get_name())), true, p_none, &heat2d_tiramisu};
    computation copy_to_host{"{copy_to_host[0]}", expr(o_memcpy, var(buff_out_gpu.get_name()), var(buff_out.get_name())), true, p_none, &heat2d_tiramisu};
    copy_to_device.after(out_init_i, computation::root);
    copy_to_host.after(out_comp, computation::root);
#endif

    heat2d_tiramisu.set_arguments({&buff_in, &buff_out});
    heat2d_tiramisu.lift_dist_comps(); // MUST go before gen_isl_ast
    heat2d_tiramisu.gen_time_space_domain();
    heat2d_tiramisu.gen_isl_ast();
    // TODO This breaks in the create_tiramisu_assignment because it assumes a send has an access function, which it does not
    //    heat2d_tiramisu.gen_cuda_stmt();
    heat2d_tiramisu.gen_halide_stmt();
    heat2d_tiramisu.dump_halide_stmt();
#ifdef USE_GPU2
    heat2d_tiramisu.gen_halide_obj("build/generated_fct_heat2dgpu.o");
#else
    heat2d_tiramisu.gen_halide_obj("build/generated_fct_heat2d_dist.o");
#endif

    return 0;
}
