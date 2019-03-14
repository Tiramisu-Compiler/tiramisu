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
#include "wrapper_sobel_dist.h" // has the sizes in it

// TODO need to compute an extra row of each (other than last node) since you need a computed row for sobel_y
// should add an additional computation to each node (except the last one) to compute the one extra row.
// That way, we don't need to chagne the original algorithm

using namespace tiramisu;

#define E(e) tiramisu::expr(e)
#define S(s) std::to_string(s)
#define MIN(bound, actual) expr(o_min, bound, actual)
#define MAX(bound, actual) expr(o_max, bound, actual)

// Current format: assumes data is already pre-distributed. Communication is only needed to transfer over shared zones

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    tiramisu::function sobel_dist("sobel_dist");

    int ROWS_PER_NODE = NROWS / NNODES;

    var x("x"), y("y");
    computation input("{input[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, 
		      tiramisu::p_float32, &sobel_dist);
    computation sobel_x("{sobel_x[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                        E(-1.0f) * input(expr(o_max, 0, y-1), expr(o_max, 0, x-1)) + 
			input(MIN(NROWS-1, y+1), MAX(0, x-1)) - 
			E(2.0f) * input(MAX(0, y-1), x) + 
			E(2.0f) * input(MIN(NROWS-1, y+1), x) - 
			E(1.0f) * input(MAX(0, y-1), MIN(NCOLS-1, x+1)) + 
			input(MIN(NROWS-1, y+1), MIN(NCOLS-1, x+1)),
			true, tiramisu::p_float32, &sobel_dist);
    computation sobel_y("{sobel_y[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                        E(-1.0f) * input(MAX(0,y-1), MAX(0,x-1)) - 
			E(2.0f) * input(y, MAX(0,x-1)) - 
			E(1.0f) * input(MIN(NROWS-1, y+1), MAX(0,x-1)) + 
			input(MAX(0, y-1), MIN(NCOLS-1,x+1)) + 
			E(2.0f) * input(y, MIN(NCOLS-1,x+1)) + 
			input(MIN(NROWS-1, y+1), MIN(NCOLS-1,x+1)),
                        true, tiramisu::p_float32, &sobel_dist);
    computation sobel("{sobel[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                      expr(tiramisu::o_sqrt, sobel_x(y,x) * sobel_x(y,x) + sobel_y(y,x) * 
			   sobel_y(y,x)), true, tiramisu::p_float32, &sobel_dist);

    constant nodes("nodes", expr(NNODES), p_int32, true, NULL, 0, &sobel_dist);
    constant nodesp("nodes_p", expr(NNODES-1), p_int32, true, NULL, 0, &sobel_dist);

    var y1("y1"), y2("y2"), d("d"), q("q");
    // split out the loop to distribute (y->y1)
    sobel_x.split(y, ROWS_PER_NODE, y1, y2);
    sobel_y.split(y, ROWS_PER_NODE, y1, y2);
    sobel.split(y, ROWS_PER_NODE, y1, y2);
    sobel_x.vectorize(x, 8);
    sobel_y.vectorize(x, 8);
    sobel.vectorize(x, 8);
    // TODO make sure to match Halide schedule. 
    
    xfer exchange = 
      computation::create_xfer("[nodes]->{exchange_s[q,y,x]: 1<=q<nodes and 0<=y<1 and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes]->{exchange_r[q,y,x]: 0<=q<(nodes-1) and 0<=y<1 and 0<=x<" + S(NCOLS) + "}",
			       q-1, q+1, 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_float32, {SYNC, BLOCK, MPI}), input(y,x), &sobel_dist);


    sobel_x.tag_distribute_level(y1);
    sobel_x.drop_rank_iter(y1);
    sobel_y.tag_distribute_level(y1);
    sobel_y.drop_rank_iter(y1);
    sobel.tag_distribute_level(y1);
    sobel.drop_rank_iter(y1);

    exchange.s->tag_distribute_level(q);
    exchange.r->tag_distribute_level(q);

    // Need a row copy to buffer out the last row in the last rank
    computation copy("[nodes,nodes_p]->{copy[q,x]: (nodes-1)<=q<=nodes_p and 0<=x<" + S(NCOLS) + "}",
		     input(ROWS_PER_NODE-1,x), true, tiramisu::p_float32, &sobel_dist);
    copy.tag_distribute_level(q);

    // For the ranks > 0, their 0th row will be incorrect for sobel_x, sobel_y, and sobel, so we need to repair those
    xfer exchange_repair = 
      computation::create_xfer("[nodes]->{exchange_repair_s[q,x]: 0<=q<nodes-1 and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes]->{exchange_repair_r[q,x]: 1<=q<nodes and 0<=x<" + S(NCOLS) + "}",
			       q+1, q-1, 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_float32, {SYNC, BLOCK, MPI}), input(ROWS_PER_NODE-1, x), &sobel_dist);

    computation sobel_x_repair("[nodes]->{sobel_x_repair[q,x]: 1<=q<nodes and 0<=x<" + S(NCOLS) + "}",
			       E(-1.0f) * input(2, expr(o_max, 0, x-1)) + // y-1 = row 2
			       input(1, MAX(0, x-1)) - 
			       E(2.0f) * input(2, x) + 
			       E(2.0f) * input(1, x) - 
			       E(1.0f) * input(2, MIN(NCOLS-1, x+1)) + 
			       input(1, MIN(NCOLS-1, x+1)),
			       true, tiramisu::p_float32, &sobel_dist);
    computation sobel_y_repair("[nodes]->{sobel_y_repair[q,x]: 1<=q<nodes and 0<=x<" + S(NCOLS) + "}",
			       E(-1.0f) * input(2, MAX(0,x-1)) - 
			       E(2.0f) * input(0, MAX(0,x-1)) - 
			       E(1.0f) * input(1, MAX(0,x-1)) + 
			       input(2, MIN(NCOLS-1,x+1)) + 
			       E(2.0f) * input(0, MIN(NCOLS-1,x+1)) + 
			       input(1, MIN(NCOLS-1,x+1)),
			       true, tiramisu::p_float32, &sobel_dist);
    computation sobel_repair("[nodes]->{sobel_repair[q,x]: 1<=q<nodes and 0<=x<" + S(NCOLS) + "}",
			     expr(tiramisu::o_sqrt, sobel_x_repair(0,x) * sobel_x_repair(0,x) + sobel_y_repair(0,x) * 
			     sobel_y_repair(0,x)), true, tiramisu::p_float32, &sobel_dist);

    sobel_x_repair.tag_distribute_level(q);
    sobel_y_repair.tag_distribute_level(q);
    sobel_repair.tag_distribute_level(q);
    exchange_repair.s->tag_distribute_level(q);
    exchange_repair.r->tag_distribute_level(q);

    
    //    exchange.s->collapse_many({collapser(2, 0, COLS)});
    //    exchange.r->collapse_many({collapser(2, 0, COLS)});

    copy.before(*exchange.s, computation::root);
    exchange.s->before(*exchange.r, computation::root);
    exchange.r->before(sobel_x, computation::root);
    sobel_x.before(sobel_y, computation::root);
    sobel_y.before(sobel, computation::root);
    sobel.before(*exchange_repair.s, computation::root);
    exchange_repair.s->before(*exchange_repair.r, computation::root);
    exchange_repair.r->before(sobel_x_repair, computation::root);
    sobel_x_repair.before(sobel_y_repair, computation::root);
    sobel_y_repair.before(sobel_repair, computation::root);

    buffer buff_input("buff_input", {ROWS_PER_NODE + 1, NCOLS}, p_float32, a_input, &sobel_dist);
    buffer buff_sobel_x("buff_sobel_x", {ROWS_PER_NODE, NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel_y("buff_sobel_y", {ROWS_PER_NODE, NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel("buff_sobel", {ROWS_PER_NODE, NCOLS}, p_float32, a_output, &sobel_dist);

    input.set_access("{input[y,x]->buff_input[y,x]}");
    sobel_x.set_access("{sobel_x[y,x]->buff_sobel_x[y,x]}");
    sobel_y.set_access("{sobel_y[y,x]->buff_sobel_y[y,x]}");
    sobel.set_access("{sobel[y,x]->buff_sobel[y,x]}");
    exchange.r->set_access("{exchange_r[q,y,x]->buff_input[" + std::to_string(ROWS_PER_NODE) + "+ y,x]}");  // 0<=y<1
    exchange_repair.r->set_access("{exchange_repair_r[q,x]->buff_input[2,x]}");
    sobel_x_repair.set_access("{sobel_x_repair[q,x]->buff_sobel_x[0,x]}");
    sobel_y_repair.set_access("{sobel_y_repair[q,x]->buff_sobel_y[0,x]}");
    sobel_repair.set_access("{sobel_repair[q,x]->buff_sobel[0,x]}");
    copy.set_access("{copy[q,x]->buff_input[" + std::to_string(ROWS_PER_NODE) + ",x]}");
    sobel_dist.codegen({&buff_input, &buff_sobel}, "build/generated_fct_sobel_dist.o");
    sobel_dist.dump_halide_stmt();

    return 0;
}
