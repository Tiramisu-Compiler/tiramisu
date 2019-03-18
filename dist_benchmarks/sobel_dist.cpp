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

    var y1("y1"), y2("y2"), d("d"), q("q");
    // split out the loop to distribute (y->y1)
    sobel_x.split(y, ROWS_PER_NODE, y1, y2);
    sobel_y.split(y, ROWS_PER_NODE, y1, y2);
    sobel.split(y, ROWS_PER_NODE, y1, y2);
    sobel_x.vectorize(x, 8);
    sobel_y.vectorize(x, 8);
    sobel.vectorize(x, 8);
    sobel.parallelize(y2);
       
    xfer exchange_back = 
      computation::create_xfer("[nodes,rank]->{exchange_back_s[q,y,x]: 1<=q<nodes and (rank*" + S(ROWS_PER_NODE) + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+1) and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_back_r[q,y,x]: 0<=q<(nodes-1) and (rank*" + S(ROWS_PER_NODE) + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+1) and 0<=x<" + S(NCOLS) + "}",
			       q-1, q+1, 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_float32, {SYNC, BLOCK, MPI}), input(y,x), &sobel_dist);

    xfer exchange_fwd = 
      computation::create_xfer("[nodes,rank]->{exchange_fwd_s[q,y,x]: 0<=q<(nodes-1) and ((rank+1)*" + S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_fwd_r[q,y,x]: 1<=q<nodes and ((rank+1)*" + S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}",
			       q+1, q-1, 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_float32, {SYNC, BLOCK, MPI}), input(y,x), &sobel_dist);
			       

    sobel_x.tag_distribute_level(y1);
    sobel_y.tag_distribute_level(y1);
    sobel.tag_distribute_level(y1);
    exchange_back.s->tag_distribute_level(q);
    exchange_back.r->tag_distribute_level(q);
    exchange_fwd.s->tag_distribute_level(q);
    exchange_fwd.r->tag_distribute_level(q);    
    
    exchange_back.s->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    exchange_back.r->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    exchange_fwd.s->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    exchange_fwd.r->collapse_many({collapse_group(2, 0, -1, NCOLS)});

    exchange_fwd.s->before(*exchange_fwd.r, computation::root);
    exchange_fwd.r->before(*exchange_back.s, computation::root);
    exchange_back.s->before(*exchange_back.r, computation::root);
    exchange_back.r->before(sobel_x, computation::root);
    sobel_x.before(sobel_y, y2);
    sobel_y.before(sobel, y2);

    buffer buff_input("buff_input", {ROWS_PER_NODE+2, NCOLS}, p_float32, a_input, &sobel_dist);
    buffer buff_sobel_x("buff_sobel_x", {ROWS_PER_NODE, NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel_y("buff_sobel_y", {ROWS_PER_NODE, NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel("buff_sobel", {ROWS_PER_NODE, NCOLS}, p_float32, a_output, &sobel_dist);

    input.set_access("[rank]->{input[y,x]->buff_input[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank)+1)}");
    sobel_x.set_access("[rank]->{sobel_x[y,x]->buff_sobel_x[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    sobel_y.set_access("[rank]->{sobel_y[y,x]->buff_sobel_y[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    sobel.set_access("[rank]->{sobel[y,x]->buff_sobel[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    exchange_back.r->set_access("{exchange_back_r[q,y,x]->buff_input[" + S(ROWS_PER_NODE) + "+1,x]}"); 
    exchange_fwd.r->set_access("{exchange_fwd_r[q,y,x]->buff_input[0,x]}"); 

    sobel_dist.codegen({&buff_input, &buff_sobel}, "build/generated_fct_sobel_dist.o");
    sobel_dist.dump_halide_stmt();

    return 0;
}
