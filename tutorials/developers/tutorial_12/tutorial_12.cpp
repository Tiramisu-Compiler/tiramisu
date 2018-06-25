// This tutorial does a distributed blur. We assume data starts distributed across rows, so communication is only
// required between adjacent nodes for sending the borders (a more detailed description follows below).
// Data is distributed by chunks of contiguous rows, for example with 12 rows and 3 nodes, we would distribute as so:
// 
// |++++++++++++++++++++++++++|
// |         Rows 0-3         | ==> Node 0
// |++++++++++++++++++++++++++|
// |         Rows 4-7         | ==> Node 1
// |++++++++++++++++++++++++++|
// |         Rows 8-11        | ==> Node 2
// |++++++++++++++++++++++++++|
//
// For simplicity, assume that NUM_ROWS % NUM_NODES == 0
//
// The following diagram shows how rows communicate their border regions with each other. We use the same
// example as above with 12 rows and 3 nodes
//
// 
// |++++++++++++++++++++++++++|
// |                          |
// |        Rows 0-3          |
// |                          |
// |--------------------------|
// |  Space for 2 extra rows  | <-- 
// |++++++++++++++++++++++++++|   | Node 1 sends rows 4 and 5 to Node 0
// |       Rows 4 and 5       | ---
// |--------------------------|
// |       Rows 6 and 7       |
// |--------------------------|
// |  Space for 2 extra rows  | <--
// |++++++++++++++++++++++++++|   | Node 2 sends rows 8 and 9 to Node 1
// |      Rows 8 and 9        | ---
// |--------------------------|
// |                          |
// |      Rows 10 and 11      |
// |                          |
// |++++++++++++++++++++++++++|
//
// This shows that each node (except for node 0) send two rows to node n-1.
//
// Handling out-of-bounds accesses at the end of each row and for the last node are handled
// in the C++ code in wrapper_tutorial_12.cpp.

#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "wrapper_tutorial_12.h"

using namespace tiramisu;

int main(int argc, char **argv) {

    static_assert(NUM_ROWS % NUM_NODES == 0, "Rows should be dividable by the number of nodes.");

    global::set_default_tiramisu_options();
    
    // Create vars that we will use throughout.
    var i("i"), j("j"), i0("i0"), i1("i1"), ii("ii"), j0("j0"), j1("j1"), jj("jj"), p("p"), q("q");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // For sake of explanation, the algorithm below differs slightly from that in tutorial_02.
    // The computation is the same, but the box stencil is accessed as {c_input[i,j], c_input[i+1,j], 
    // c_input[i+2,j], c_blurx[i,j], c_blurx[i,j+1], c_blurx[i,j+2]} instead of {c_input[i-1,j], c_input[i,j], 
    // c_input[i+1,j], c_blurx[i,j-1], c_blurx[i,j], c_blurx[i,j+1]}

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

    // Declare the computations c_blurx and c_blury.
    expr e1 = (c_input(i, j) + c_input(i + 1, j) + c_input(i + 2, j)) / ((uint8_t) 3);

    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint8, &blurxy);

    expr e2 = (c_blurx(i, j) + c_blurx(i, j + 1) + c_blurx(i, j + 2)) / ((uint8_t) 3);

    computation c_blury("[N,M]->{c_blury[i,j]: 0<=i<N and 0<=j<M}", e2, true, p_uint8, &blurxy);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Prepare the computations for distributing by splitting to create an outer loop over the 
    // number of nodes
    c_input.split(i, NUM_ROWS/NUM_NODES, i0, i1);
    c_blurx.split(i, NUM_ROWS/NUM_NODES, i0, i1);
    c_blury.split(i, NUM_ROWS/NUM_NODES, i0, i1);

    // Tag the outer loop level over the number of nodes so that it is distributed. Internally,
    // this creates a new Var called "rank"
    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);

    // Tell the code generator to not include the "rank" var when computing linearized indices.
    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    // Create the communication (i.e. data transfer) for the borders
    xfer border_comm = computation::create_xfer(/*Iteration domain for the send. All nodes except Node 0 send two rows.*/
                                                "[NODES,N,M]->{border_send[p,ii,jj]: 1<=p<NODES and 0<=ii<2 and 0<=jj<M}",
                                                /*Iteration domain for the receive. All nodes except the last Node receive two rows.*/
                                                "[NODES,N,M]->{border_recv[q,ii,jj]: 0<=q<NODES-1 and 0<=ii<2 and 0<=jj<M}",
                                                /*The dest of each send relative to the send's iteration domain.*/ 
                                                p-1,   
                                                /*The src of each receive relative to receive's iteration domain.*/
                                                q+1, 
                                                /*Properties defining the type of transfer to perform for send and receive, respectively.
                                                  We choose to do a blocking asynchronous send and receive.*/
                                                xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                                /*The access that says where to start the transfer. We define it relative to each node.*/
                                                c_input(ii, jj), &blurxy);

    // Distribute the communication
    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);

    // Order computations and communication
    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(c_blurx, computation::root);
    c_blurx.before(c_blury, i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Make each buffer's size relative to a single node. We add 2 extra rows and columns to account for the extra
    // data sent in the border and to prevent out-of-bounds accesses.
    buffer b_input("b_input", {tiramisu::expr(NUM_ROWS/NUM_NODES) + 2, tiramisu::expr(NUM_COLS) + 2}, p_uint8, a_input, &blurxy);
    buffer b_blurx("b_blurx", {tiramisu::expr(NUM_ROWS/NUM_NODES), tiramisu::expr(NUM_COLS) + 2}, p_uint8, a_temporary, &blurxy);
    buffer b_blury("b_blury", {tiramisu::expr(NUM_ROWS/NUM_NODES), tiramisu::expr(NUM_COLS)}, p_uint8, a_output, &blurxy);

    // Map the computations to a buffer.
    // The send doesn't explicitly write out data (MPI handles that internally), but the receive does write to a buffer,
    // so it requires an access function as well.
    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    border_comm.r->set_access("{border_recv[q,ii,jj]->b_input[ii,jj]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    blurxy.codegen({&b_input, &b_blury}, "build/generated_fct_developers_tutorial_12.o");

}
