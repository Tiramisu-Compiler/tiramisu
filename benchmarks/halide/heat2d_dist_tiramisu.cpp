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

constexpr auto block_size = 32;
constexpr auto block_size_small = 30;

#define BS "32"
#define BSS "30"



int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    //#ifdef USE_MPI
    tiramisu::function heat2d_tiramisu("heat2d_dist_tiramisu");
    //#elif defined(USE_GPU)
    //    tiramisu::function heat2d_tiramisu("heat2d_gpu_tiramisu");
    //#elif defined(USE_COOP)
    //    tiramisu::function heat2d_tiramisu("heat2d_coop_tiramisu");
    //#endif
    
    // Input params.
    float alpha = 0.3;
    float beta = 0.4;
    constant("N", N, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("M", M, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("CRANKS", NUM_CPU_RANKS, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("NB", N - 1, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("MB", M - 1, it_type, true, nullptr, 0, &heat2d_tiramisu);
    constant("ITERS", TOTAL_ITERATIONS, it_type, true, nullptr, 0, &heat2d_tiramisu);
    
    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), i2("i2"), q("q"), t("t");

    // Initial input
    computation in("[N,M] -> {in[i,j]: 0 <= i < N and 0 <= j < M}", expr(), false, data_type, &heat2d_tiramisu);
    // even iterations
    computation init_even("[NB, MB, ITERS]->{init_even[t,i,j]: 0<=t<ITERS/2 and 1<=i<NB and 1<=j<MB}", expr(0.0f), true, data_type, &heat2d_tiramisu); //this does not mean what you think it means
    // odd iterations
    computation init_odd("[NB, MB, ITERS]->{init_odd[t,i,j]: 0<=t<ITERS/2 and 1<=i<NB and 1<=j<MB}", expr(0.0f), true, data_type, &heat2d_tiramisu);

    //#ifdef USE_GPU
    /*    var o_i("o_i"), o_j("o_j");
    computation shared_in{"[N, M] -> {shared_in[i, o_i, j, o_j]: 0 <= i < N and 0 <= j < M}",
	expr(), false, data_type, &heat2d_tiramisu};

    expr comp_expr = shared_in(i, 0, j, 0) * alpha +
      (shared_in(i, 1, j, 0) +
       shared_in(i, 0, j, 1) +
       shared_in(i, -1, j, 0) +
       shared_in(i, 0, j, -1)) * beta;
    computation out_comp{"[NB, MB, ITERS] -> {out_comp[t, i, j]: 0<=t<ITERS/2 and 1 <= i < NB and 1 <= j < MB}", comp_expr, true, data_type, &heat2d_tiramisu};
    
    computation shared_dec{"[NB, MB, ITERS] -> {shared_dec[t, i, j] : 0<=t<ITERS/2 and 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}",
	expr(o_allocate, buff_shared.get_name()), true, p_none, &heat2d_tiramisu};
    computation shared_init{"[NB, MB, ITERS] -> {shared_init[i, j] : 0<=t<ITERS/2 and 0 <= i < NB + floor(NB / " BSS " + 1) * 2 and 0 <= j < MB + floor(MB / " BSS " + 1) * 2}",
	in(i, j),
	true, data_type, &heat2d_tiramisu};
    
    shared_dec.interchange(i, j);
    shared_init.interchange(i, j);
    out_comp.interchange(i, j);

    buffer buff_shared{"buff_shared", {block_size, block_size}, data_type, a_temporary, &heat2d_tiramisu};
    buff_shared.tag_gpu_shared();    */
    
      //#else 
    expr comp_expr_even = in(i,j) * alpha + (in(i-1, j) + in(i+1, j) + in(i, j-1) + in(i, j+1)) * beta;
    computation out_comp_even("[NB,MB,ITERS] -> {out_comp_even[t, i, j]: 0<=t<ITERS/2 and 1 <= i < NB and 1 <= j < MB}",
			      comp_expr_even, true, data_type, &heat2d_tiramisu);
    computation wrapper("[N,M] -> {wrapper[i,j]: 0 <= i < N and 0 <= j < M}", expr(), false, data_type, &heat2d_tiramisu);
    expr comp_expr_odd = wrapper(i,j) * alpha + (wrapper(i-1, j) + wrapper(i+1, j) + wrapper(i, j-1) + wrapper(i, j+1)) * beta;
    computation out_comp_odd("[NB,MB,ITERS] -> {out_comp_odd[t, i, j]: 0<=t<ITERS/2 and 1 <= i < NB and 1 <= j < MB}",
			     comp_expr_odd, true, data_type, &heat2d_tiramisu);
    //#endif


#ifdef USE_MPI
    // split into the number of ranks by distributing
    init_even.shift(i, -1);
    init_odd.shift(i, -1);
    out_comp_even.shift(i, -1);
    out_comp_odd.shift(i, -1);
    if (NUM_CPU_RANKS+NUM_GPU_RANKS == 1) {
      init_even.split(i, 8, i1, i2);
      init_odd.split(i, 8, i1, i2);
      out_comp_even.split(i, 8, i1, i2);
      out_comp_odd.split(i, 8, i1, i2);      
      init_even.tag_parallel_level(i2);
      init_odd.tag_parallel_level(i2);
    } else {
      init_even.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      init_odd.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      out_comp_even.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      out_comp_odd.split(i, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS), i1, i2);
      var i3("i3"), i4("i4");
      init_even.split(i2, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS)/8, i3, i4);
      init_odd.split(i2, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS)/8, i3, i4);
      out_comp_even.split(i2, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS)/8, i3, i4);
      out_comp_odd.split(i2, (N-2)/(NUM_CPU_RANKS+NUM_GPU_RANKS)/8, i3, i4);
      init_even.tag_parallel_level(i3);
      init_odd.tag_parallel_level(i3);
    }
   
    // Insert communication
    // This communication sends a row backwards (from rank q to rank q-1)
    xfer send_recv_bkwd_even_mpi;
    xfer send_recv_bkwd_odd_mpi;
    // This communication sends a row forwards (from rank q to q+1);
    xfer send_recv_fwd_even_mpi;
    xfer send_recv_fwd_odd_mpi;
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_bkwd_even_mpi = computation::create_xfer("[N,M,CRANKS,ITERS]->{send_bkwd_even_mpi[t,q,i,j]: 0<=t<ITERS/2 and 1<=q<CRANKS and 0<=i<1 and 0<=j<=M}",
						     "[N,M,CRANKS,ITERS]->{recv_bkwd_even_mpi[t,q,i,j]: 0<=t<ITERS/2 and 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<M}",
						     q-1, q+1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						     xfer_prop(data_type, {SYNC, BLOCK, MPI}), in(i,j), &heat2d_tiramisu);
      
      send_recv_bkwd_odd_mpi = computation::create_xfer("[N,M,CRANKS,ITERS]->{send_bkwd_odd_mpi[t,q,i,j]: 0<=t<ITERS/2 and 1<=q<CRANKS and 0<=i<1 and 0<=j<=M}",
						    "[N,M,CRANKS,ITERS]->{recv_bkwd_odd_mpi[t,q,i,j]: 0<=t<ITERS/2 and 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<M}",
						    q-1, q+1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						    xfer_prop(data_type, {SYNC, BLOCK, MPI}), wrapper(i,j), &heat2d_tiramisu);
      
      send_recv_fwd_even_mpi = computation::create_xfer("[N,M,CRANKS,ITERS]->{send_fwd_even_mpi[t,q,i,j]: 0<=t<ITERS/2 and 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<M}",
						    "[N,M,CRANKS,ITERS]->{recv_fwd_even_mpi[t,q,i,j]: 0<=t<ITERS/2 and 1<=q<CRANKS and 0<=i<1 and 0<=j<M}",
						    q+1, q-1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						    xfer_prop(data_type, {SYNC, BLOCK, MPI}),
						    in(i + N / (NUM_CPU_RANKS + NUM_GPU_RANKS) - 1, j), &heat2d_tiramisu);
      
      send_recv_fwd_odd_mpi = computation::create_xfer("[N,M,CRANKS,ITERS]->{send_fwd_odd_mpi[t,q,i,j]: 0<=t<ITERS/2 and 0<=q<CRANKS-1 and 0<=i<1 and 0<=j<M}",
						   "[N,M,CRANKS,ITERS]->{recv_fwd_odd_mpi[t,q,i,j]: 0<=t<ITERS/2 and 1<=q<CRANKS and 0<=i<1 and 0<=j<M}",
						   q+1, q-1, xfer_prop(data_type, {ASYNC, BLOCK, MPI}),
						   xfer_prop(data_type, {SYNC, BLOCK, MPI}),
						   wrapper(i + N / (NUM_CPU_RANKS + NUM_GPU_RANKS) - 1, j), &heat2d_tiramisu);
      

      
    }
    // Optimizations
    // don't tag both out_comp and init because there is a bug in the code generator for when there is fusion
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      out_comp_even.tag_distribute_level(i1);
      out_comp_odd.tag_distribute_level(i1);
      init_even.tag_distribute_level(i1);
      init_odd.tag_distribute_level(i1);
      init_even.drop_rank_iter();
      init_odd.drop_rank_iter();
      out_comp_even.drop_rank_iter();
      out_comp_odd.drop_rank_iter();      
      send_recv_bkwd_even_mpi.s->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_bkwd_odd_mpi.s->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_bkwd_even_mpi.r->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_bkwd_odd_mpi.r->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_fwd_even_mpi.s->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_fwd_odd_mpi.s->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_fwd_even_mpi.r->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_fwd_odd_mpi.r->collapse_many({collapse_group(3, (cit_type)0, (cit_type)(M-2)), collapse_group(2, (cit_type)0, (cit_type)1)});
      send_recv_bkwd_even_mpi.s->tag_distribute_level(q);
      send_recv_bkwd_odd_mpi.s->tag_distribute_level(q);
      send_recv_bkwd_even_mpi.r->tag_distribute_level(q);
      send_recv_bkwd_odd_mpi.r->tag_distribute_level(q);
      send_recv_fwd_even_mpi.s->tag_distribute_level(q);
      send_recv_fwd_odd_mpi.s->tag_distribute_level(q);
      send_recv_fwd_even_mpi.r->tag_distribute_level(q);
      send_recv_fwd_odd_mpi.r->tag_distribute_level(q);
    }

        buffer buff_out_even{"buff_out_even", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS) + 2, M}, data_type, a_output, &heat2d_tiramisu};
    buffer buff_out_odd{"buff_out_odd", {N/(NUM_CPU_RANKS+NUM_GPU_RANKS), M}, data_type, a_output, &heat2d_tiramisu};
    
    //#ifdef USE_GPU
    assert(TOTAL_ITERATIONS % 2 == 0); // so that the output buffer is the last odd computation    
    // Buffers
//    buffer buff_out_gpu{"buff_out_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    //    buff_out_gpu.tag_gpu_global();
    //    buffer buff_in_gpu{"buff_in_gpu", {N, M}, data_type, a_temporary, &heat2d_tiramisu};
    //    buff_in_gpu.tag_gpu_global();    

    // Do main copy once    
    //    computation copy_to_device{"{copy_to_device[0]}", expr(o_memcpy, var(buff_out_odd.get_name()), var(buff_in_gpu.get_name())),
    //	true, p_none, &heat2d_tiramisu};
    //    computation copy_to_host{"{copy_to_host[0]}", expr(o_memcpy, var(buff_out_gpu.get_name()), var(buff_out_odd.get_name())),
    //	true, p_none, &heat2d_tiramisu};
    //  buffer buff_in{"buff_in", {N, M}, data_type, a_input, &heat2d_tiramisu};
    //  buffer buff_out{"buff_out", {N, M}, data_type, a_output, &heat2d_tiramisu};    
    //#endif

    // Ordering
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_fwd_even_mpi.s->before(*send_recv_fwd_even_mpi.r, t);
      send_recv_fwd_even_mpi.r->before(*send_recv_bkwd_even_mpi.s, t);
      send_recv_bkwd_even_mpi.s->before(*send_recv_bkwd_even_mpi.r, t);
      send_recv_bkwd_even_mpi.r->before(init_even, t);
      init_even.before(out_comp_even, j);
      out_comp_even.before(*send_recv_fwd_odd_mpi.s, t);
      send_recv_fwd_odd_mpi.s->before(*send_recv_fwd_odd_mpi.r, t);
      send_recv_fwd_odd_mpi.r->before(*send_recv_bkwd_odd_mpi.s, t);
      send_recv_bkwd_odd_mpi.s->before(*send_recv_bkwd_odd_mpi.r, t);
      send_recv_bkwd_odd_mpi.r->before(init_odd, t);
      init_odd.before(out_comp_odd, j);
    } else {
      //#ifdef USE_GPU
      //      copy_to_device.before(init_even, computation::root);
      //#endif
      init_even.before(out_comp_even, j);
      out_comp_even.before(init_odd, t);
      init_odd.before(out_comp_odd, j);
#ifdef USE_GPU
      //      out_comp_odd.before(copy_to_host, computation::root);
#endif      
    }

    // even writes to odd, 
    in.set_access("{in[i,j]->buff_out_odd[i,j]}");
    wrapper.set_access("{wrapper[i,j]->buff_out_even[i,j]}");
    init_even.set_access("{init_even[t,i,j]->buff_out_even[i,j]}");
    init_odd.set_access("{init_odd[t,i,j]->buff_out_odd[i,j]}");
    out_comp_even.set_access("{out_comp_even[t,i,j] -> buff_out_even[i,j]}");
    out_comp_odd.set_access("{out_comp_odd[t,i,j] -> buff_out_odd[i,j]}");
    if (NUM_CPU_RANKS+NUM_GPU_RANKS > 1) {
      send_recv_bkwd_even_mpi.r->set_access("{recv_bkwd_even_mpi[t, q, i,j]->buff_out_odd[i+" +
				   std::to_string(N/(NUM_CPU_RANKS+NUM_GPU_RANKS)) + ",j]}");
      send_recv_fwd_even_mpi.r->set_access("{recv_fwd_even_mpi[t, q, i,j]->buff_out_odd[i,j]}");
      send_recv_bkwd_odd_mpi.r->set_access("{recv_bkwd_odd_mpi[t, q, i,j]->buff_out_even[i+" +
				   std::to_string(N/(NUM_CPU_RANKS+NUM_GPU_RANKS)) + ",j]}");
      send_recv_fwd_odd_mpi.r->set_access("{recv_fwd_odd_mpi[t, q, i,j]->buff_out_even[i,j]}");

      //      send_recv_fwd_even_mpi.s->unschedule_this_computation();
      //      send_recv_fwd_even_mpi.r->unschedule_this_computation();
      //      send_recv_bkwd_even_mpi.s->unschedule_this_computation();
      //      send_recv_bkwd_even_mpi.r->unschedule_this_computation();

      //      send_recv_fwd_odd_mpi.s->unschedule_this_computation();
      //      send_recv_fwd_odd_mpi.r->unschedule_this_computation();
      //      send_recv_bkwd_odd_mpi.s->unschedule_this_computation();
      //      send_recv_bkwd_odd_mpi.r->unschedule_this_computation();
    }
#endif

    heat2d_tiramisu.set_arguments({&buff_out_even, &buff_out_odd});
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
