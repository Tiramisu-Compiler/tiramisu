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

    global::set_default_tiramisu_options();
    
    // Create vars that we will use throughout.
    var i("i"), j("j"), i0("i0"), i1("i1"), ii("ii"), j0("j0"), j1("j1"), jj("jj"), p("p"), q("q");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // The algorithm below differs slightly from that in tutorial_02.
    // The computation is the same, but the box stencil is accessed as {c_input[i,j], c_input[i+1,j], 
    // c_input[i+2,j], c_blurx[i,j], c_blurx[i,j+1], c_blurx[i,j+2]} instead of {c_input[i-1,j], c_input[i,j], 
    // c_input[i+1,j], c_blurx[i,j-1], c_blurx[i,j], c_blurx[i,j+1]}

    function blurxy("blurxy");

    constant ROWS("ROWS", expr((int32_t) _ROWS), p_int32, true, nullptr, 0, &blurxy);
    constant COLS("COLS", expr((int32_t) _COLS), p_int32, true, nullptr, 0, &blurxy);

    // Declare a wrapper around the input.
    computation c_input("[ROWS,COLS]->{c_input[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", expr(), false, p_uint32, &blurxy);

    // Declare the computations c_blurx and c_blury.
    expr e1 = (c_input(i, j) + c_input(i + 1, j) + c_input(i + 2, j)) / ((uint32_t) 3);

    computation c_blurx("[ROWS,COLS]->{c_blurx[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", e1, true, p_uint32, &blurxy);

    expr e2 = (c_blurx(i, j) + c_blurx(i, j + 1) + c_blurx(i, j + 2)) / ((uint32_t) 3);

    computation c_blury("[ROWS,COLS]->{c_blury[i,j]: 0<=i<ROWS and 0<=j<COLS}", e2, true, p_uint32, &blurxy);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Prepare the computations for distributing by splitting to create an outer loop over the 
    // number of nodes
    c_input.split(i, _ROWS/10, i0, i1);
    c_blurx.split(i, _ROWS/10, i0, i1);
    c_blury.split(i, _ROWS/10, i0, i1);

    // Tag the outer loop level over the number of nodes so that it is distributed. Internally,
    // this creates a new Var called "rank"
    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);

    // Tell the code generator to not include the "rank" var when computing linearized indices (where the rank var is the tagged loop)
    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    // Create the communication (i.e. data transfer) for the borders
    xfer border_comm = computation::create_xfer(/*Iteration domain for the send. All nodes except Node 0 send two rows.*/
                                                "[COLS]->{border_send[p,ii,jj]: 1<=p<10 and 0<=ii<2 and 0<=jj<COLS+2}",
                                                /*Iteration domain for the receive. All nodes except the last Node receive two rows.*/
                                                "[COLS]->{border_recv[q,ii,jj]: 0<=q<9 and 0<=ii<2 and 0<=jj<COLS+2}",
                                                /*The dest of each send relative to the send's iteration domain.*/ 
                                                p-1,   
                                                /*The src of each receive relative to receive's iteration domain.*/
                                                q+1, 
                                                /*Properties defining the type of transfer to perform for send and receive, respectively.
                                                  We choose to do a blocking asynchronous send and receive.*/
                                                xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}), xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}),
                                                /*The access that says where to start the transfer. We define it relative to each node.*/
                                                c_input(ii, jj), &blurxy);

    // Distribute the communication
    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);
    border_comm.s->collapse(2, 0, -1, COLS+2);
    border_comm.s->collapse(1, 0, -1, 2);
    border_comm.r->collapse(2, 0, -1, COLS+2);
    border_comm.r->collapse(1, 0, -1, 2);

    // Order computations and communication
    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(c_blurx, computation::root);
    c_blurx.before(c_blury, i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Make each buffer's size relative to a single node. We add 2 extra rows and columns to account for the extra
    // data sent in the border and to prevent out-of-bounds accesses.
    buffer b_input("b_input", {tiramisu::expr(_ROWS/10) + 2, tiramisu::expr(_COLS) + 2}, p_uint32, a_input, &blurxy);
    buffer b_blurx("b_blurx", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS) + 2}, p_uint32, a_temporary, &blurxy);
    buffer b_blury("b_blury", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS)}, p_uint32, a_output, &blurxy);

    // Map the computations to a buffer.
    // The send doesn't explicitly write out data (MPI handles that internally), but the receive does write to a buffer,
    // so it requires an access function as well.
    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    border_comm.r->set_access("{border_recv[q,ii,jj]->b_input[" + std::to_string(_ROWS/10) + "+ii,jj]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    blurxy.codegen({&b_input, &b_blury}, "build/generated_fct_developers_tutorial_12.o");

}
