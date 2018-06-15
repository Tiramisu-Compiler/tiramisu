// This tutorial does a simple distributed blur, assuming the data starts distributed across nodes.
// Data is distributed by chunks of contiguous rows, for example with 12 rows and 3 nodes, we would distribute as so:
// |--------------------------|
// |         Rows 0-3         | ==> Node 0
// |--------------------------|
// |         Rows 4-7         | ==> Node 1
// |--------------------------|
// |         Rows 8-11        | ==> Node 2
// |--------------------------|
//
//
// For simplicity, assume that NUM_ROWS % NUM_NODES == 0

#include <Halide.h>
#include "../include/tiramisu/core.h"

using namespace tiramisu;

#define NUM_ROWS 1280
#define NUM_COLS 768
#define NUM_NODES 5

int main(int argc, char **argv) {

    static_assert(NUM_ROWS % NUM_NODES == 0, "Rows should be dividable by the number of nodes.");

    global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    /*
     * Declare a function blurxy.
     * Declare two arguments (tiramisu buffers) for the function: b_input and b_blury
     * Declare an invariant for the function.
     */
    function blurxy("blurxy");

    constant p0("N", expr((int32_t) NUM_ROWS), p_int32, true, nullptr, 0, &blurxy);
    constant p1("M", expr((int32_t) NUM_COLS), p_int32, true, nullptr, 0, &blurxy);
    constant p2("NODES", expr((int32_t) NUM_NODES), p_int32, true, nullptr, 0, &blurxy);

    // Declare a wrapper around the input.
    computation c_input("[N,M]->{c_input[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_uint8, &blurxy);

    var i("i"), j("j"), i0("i0"), i1("i1"), j0("j0"), j1("j1");

    // Declare the computations c_blurx and c_blury.
    expr e1 = (c_input(i - 1, j) +
               c_input(i    , j) +
               c_input(i + 1, j)) / ((uint8_t) 3);

    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint8, &blurxy);

    expr e2 = (c_blurx(i, j - 1) +
               c_blurx(i, j) +
               c_blurx(i, j + 1)) / ((uint8_t) 3);

    computation c_blury("[N,M]->{c_blury[i,j]: 0<=i<N and 0<=j<M}", e2, true, p_uint8, &blurxy);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    c_input.split(i, NUM_ROWS/NUM_NODES, i0, i1);
    c_blurx.split(i, NUM_ROWS/NUM_NODES, i0, i1);
    c_blury.split(i, NUM_ROWS/NUM_NODES, i0, i1);

    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);
    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    var p("p"), q("q");
    xfer border_comm = computation::create_xfer("[NODES,N,M]->{border_send[p,i,j]: 0<=p<NODES-1 and 0<=i<1 and 0<=j<M}",
                                                "[NODES,N,M]->{border_recv[q,i,j]: 1<=q<NODES and 0<=i<1 and 0<=j<M}",
                                                p+1, q-1, xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                                xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                                c_input((NUM_ROWS - 1)/NUM_NODES * NUM_COLS, j), &blurxy);

    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);

    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(c_blurx, computation::root);
    c_blurx.before(c_blury, i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_input("b_input", {tiramisu::expr(NUM_ROWS/NUM_NODES) + 2, tiramisu::expr(NUM_COLS) + 2}, p_uint8, a_input, &blurxy);
    buffer b_blurx("b_blurx", {tiramisu::expr(NUM_ROWS/NUM_NODES) + 2, tiramisu::expr(NUM_COLS) + 2}, p_uint8, a_temporary, &blurxy);
    buffer b_blury("b_blury", {tiramisu::expr(NUM_ROWS/NUM_NODES) + 2, tiramisu::expr(NUM_COLS) + 2}, p_uint8, a_output, &blurxy);

    // Map the computations to a buffer.
    c_input.set_access("{c_input[i,j]->b_input[i+1,j+1]}");
    border_comm.r->set_access("{border_recv[q,i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i+1,j+1]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i+1,j+1]}");

    blurxy.codegen({&b_input, &b_blury}, "build/generated_fct_tutorial_11.o");

}