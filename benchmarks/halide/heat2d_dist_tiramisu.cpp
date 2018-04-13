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

#define data_type p_float32
#define it_type p_int64
#define cit_type int64_t

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

#ifdef USE_MPI
    tiramisu::function heat2d_tiramisu("heat2d_dist_tiramisu");
#elif defined(USE_GPU)
    tiramisu::function heat2d_tiramisu("heat2d_gpu_tiramisu");
#elif defined(USE_COOP)
    tiramisu::function heat2d_tiramisu("heat2d_coop_tiramisu");
#endif
    
    // Input params.
    float alpha = 0.3;
    float beta = 0.4;
    constant("N", N, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("CRANKS", NUM_CPU_RANKS, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("NB", N - 1, it_type, true, nullptr, 0, &heat2d_tiramisu);
    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), i2("i2"), q("q");

    computation in("[N] -> {in[i, j]: 0 <= i < N and 0 <= j < N}", expr(), false, data_type, &heat2d_tiramisu);
    computation init("[NB]->{init[i,j]: 1<=i<NB and 1<=j<NB}", expr(0.0f), true, data_type, &heat2d_tiramisu);
    
    expr comp_expr = in(i,j) * alpha + (in(i-1, j) + in(i+1, j) + in(i, j-1) + in(i, j+1)) * beta;
    computation out_comp("[NB] -> {out_comp[i, j]: 1 <= i < NB and 1 <= j < NB}", comp_expr, true, data_type, &heat2d_tiramisu);  

#ifdef USE_MPI
    // split into the number of ranks by distributing
    init.shift(i, -1);
    out_comp.shift(i, -1);
    if (NUM_CPU_RANKS+NUM_GPU_RANKS == 1) {
      init.split(i, 8, i1, i2);
      out_comp.split(i, 8, i1, i2);      
    } else {
      init.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      out_comp.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);      
    }

    
    // Insert communication
    // This communication sends a row backwords (from rank q to rank q-1)
    xfer send_recv_bkwd;
    // This communication sends a row forwards (from rank q to q+1);
    xfer send_recv_fwd;
    std::cerr << "NUM CPU RANKS " << NUM_CPU_RANKS << std::endl;
    std::cerr << "NUM GPU RANKS " << NUM_GPU_RANKS << std::endl;
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_bkwd = computation::create_xfer("[N,CRANKS]->{send_bkwd[q,i,j]: 1<=q<CRANKS and 0<=i<1 and 0<=j<N}",
						"[N,CRANKS]->{recv_bkwd[q,i,j]: 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<N}",
						q-1, q+1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						xfer_prop(data_type, {SYNC, BLOCK, MPI}), in(i,j), &heat2d_tiramisu);
      
      send_recv_fwd = computation::create_xfer("[N,CRANKS]->{send_fwd[q,i,j]: 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<N}",
					       "[N,CRANKS]->{recv_fwd[q,i,j]: 1<=q<CRANKS and 0<=i<1 and 0<=j<N}",
						  q+1, q-1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						  xfer_prop(data_type, {SYNC, BLOCK, MPI}), in(i,j), &heat2d_tiramisu);
    }
    // Optimizations
    // don't tag both out_comp and init because there is a bug in the code generator for when there is fusion
    //    init.tag_parallel_level(i2);
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      out_comp.tag_distribute_level(i1);
      init.drop_rank_iter();
      out_comp.drop_rank_iter();
    }
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_bkwd.s->collapse_many({collapse_group(2, (cit_type)0, (cit_type)(N-2)), collapse_group(1, (cit_type)0, (cit_type)1)});
      send_recv_bkwd.r->collapse_many({collapse_group(2, (cit_type)0, (cit_type)(N-2)), collapse_group(1, (cit_type)0, (cit_type)1)});
      send_recv_fwd.s->collapse_many({collapse_group(2, (cit_type)0, (cit_type)(N-2)), collapse_group(1, (cit_type)0, (cit_type)1)});
      send_recv_fwd.r->collapse_many({collapse_group(2, (cit_type)0, (cit_type)(N-2)), collapse_group(1, (cit_type)0, (cit_type)1)});
      send_recv_bkwd.s->tag_distribute_level(q);
      send_recv_bkwd.r->tag_distribute_level(q);
      send_recv_fwd.s->tag_distribute_level(q);
      send_recv_fwd.r->tag_distribute_level(q);
    }

    // Ordering
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {    
      send_recv_fwd.s->before(*send_recv_fwd.r, computation::root);
      send_recv_fwd.r->before(*send_recv_bkwd.s, computation::root);
      send_recv_bkwd.s->before(*send_recv_bkwd.r, computation::root);
      send_recv_bkwd.r->before(init, computation::root);
    }
    init.before(out_comp, j);

    // Buffers
    buffer buff_in{"buff_in", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS) + 2, N}, data_type, a_input, &heat2d_tiramisu};
    buffer buff_out{"buff_out", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS), N}, data_type, a_output, &heat2d_tiramisu};

    in.set_access("{in[i,j]->buff_in[i,j]}");
    init.set_access("{init[i,j]->buff_out[i,j]}");
    out_comp.set_access("{out_comp[i,j] -> buff_out[i,j]}");
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_bkwd.r->set_access("{recv_bkwd[q,i,j]->buff_in[i+" +
				   std::to_string(N/(NUM_CPU_RANKS+NUM_GPU_RANKS)) + ",j]}");
      send_recv_fwd.r->set_access("{recv_fwd[q,i,j]->buff_in[i,j]}");
    }
#endif

    heat2d_tiramisu.set_arguments({&buff_in, &buff_out});
#if defined(USE_MPI) || defined(USE_COOP)
    heat2d_tiramisu.lift_dist_comps(); // MUST go before gen_isl_ast
#endif
#if defined(USE_GPU) || defined(USE_COOP)
    heat2d_tiramisu.gen_cuda_stmt();
#endif
    heat2d_tiramisu.gen_time_space_domain();
    heat2d_tiramisu.gen_isl_ast();
    heat2d_tiramisu.gen_halide_stmt();
    heat2d_tiramisu.dump_halide_stmt();

    heat2d_tiramisu.gen_halide_obj("build/generated_fct_heat2d_dist.o");


    return 0;
}
