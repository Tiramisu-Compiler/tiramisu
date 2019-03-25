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
#define LOOP_TYPE p_int64
#define T(val) expr(o_cast, LOOP_TYPE, val)
#define VAR(name) var name(LOOP_TYPE, #name)
// Current format: assumes data is already pre-distributed. Communication is only needed to transfer over shared zones


int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int64);

    tiramisu::function sobel_dist("sobel_dist");

    int64_t ROWS_PER_NODE = NROWS / NNODES;

    VAR(x); VAR(y);
    computation input("{input[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, 
		      tiramisu::p_float32, &sobel_dist);
    computation sobel_x("{sobel_x[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                        E(-1.0f) * input(expr(o_max, T(0), y-T(1)), expr(o_max, T(0), x-T(1))) +
			input(MIN(T(NROWS-1), y+T(1)), MAX(T(0), x-T(1))) - 
			E(2.0f) * input(MAX(T(0), y-T(1)), x) + 
			E(2.0f) * input(MIN(T(NROWS-1), y+T(1)), x) - 
			E(1.0f) * input(MAX(T(0), y-T(1)), MIN(T(NCOLS-1), x+T(1))) + 
			input(MIN(T(NROWS-1),y+T(1)), MIN(T(NCOLS-1), x+T(1))),
			true, tiramisu::p_float32, &sobel_dist);
    computation sobel_y("{sobel_y[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                        E(-1.0f) * input(MAX(T(0),y-T(1)), MAX(T(0),x-T(1))) -
			E(2.0f) * input(y, MAX(T(0),x-T(1))) - 
			E(1.0f) * input(MIN(T(NROWS-1), y+1), MAX(T(0),x-T(1))) + 
			input(MAX(T(0), y-T(1)), MIN(T(NCOLS-1),x+T(1))) + 
			E(2.0f) * input(y, MIN(T(NCOLS-1),x+T(1))) + 
			input(MIN(T(NROWS-1), y+T(1)), MIN(T(NCOLS-1),x+T(1))),
                        true, tiramisu::p_float32, &sobel_dist);
    computation sobel("{sobel[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
                      expr(tiramisu::o_sqrt, sobel_x(y,x) * sobel_x(y,x) + sobel_y(y,x) * 
			   sobel_y(y,x)), true, tiramisu::p_float32, &sobel_dist);
    
    constant nodes("nodes", expr((int64_t)NNODES), LOOP_TYPE /*This doesn't seem to actually do anything*/, true, NULL, 0, &sobel_dist);

    VAR(y1); VAR(y2); VAR(d); VAR(q);
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
			       xfer_prop(p_float32, {ASYNC, NONBLOCK, MPI, NOWAIT}), 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), input(y,x), &sobel_dist);

    xfer exchange_fwd = 
      computation::create_xfer("[nodes,rank]->{exchange_fwd_s[q,y,x]: 0<=q<(nodes-1) and ((rank+1)*" + S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_fwd_r[q,y,x]: 1<=q<nodes and ((rank+1)*" + S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}",
			       q+1, q-1, 
			       xfer_prop(p_float32, {ASYNC, NONBLOCK, MPI, NOWAIT}), 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), input(y,x), &sobel_dist);
    
    sobel_x.tag_distribute_level(y1);
    sobel_y.tag_distribute_level(y1);
    sobel.tag_distribute_level(y1);
    exchange_back.s->tag_distribute_level(q);
    exchange_back.r->tag_distribute_level(q);
    exchange_fwd.s->tag_distribute_level(q);
    exchange_fwd.r->tag_distribute_level(q);    
    
    exchange_back.s->collapse_many({collapse_group(2, (int64_t)0, -1, (int64_t)NCOLS)});
    exchange_back.r->collapse_many({collapse_group(2, (int64_t)0, -1, (int64_t)NCOLS)});
    exchange_fwd.s->collapse_many({collapse_group(2, (int64_t)0, -1, (int64_t)NCOLS)});
    exchange_fwd.r->collapse_many({collapse_group(2, (int64_t)0, -1, (int64_t)NCOLS)});

    buffer buff_input("buff_input", {(int64_t)ROWS_PER_NODE+2, (int64_t)NCOLS}, p_float32, a_input, &sobel_dist);
    buffer buff_sobel_x("buff_sobel_x", {(int64_t)NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel_y("buff_sobel_y", {(int64_t)NCOLS}, p_float32, a_temporary, &sobel_dist);
    buffer buff_sobel("buff_sobel", {(int64_t)ROWS_PER_NODE, (int64_t)NCOLS}, p_float32, a_output, &sobel_dist);

    computation c1("{c1[y]: 0<=y<" + S(NROWS) + " }", expr(o_allocate, buff_sobel_x.get_name()), true, 
		   tiramisu::p_none, &sobel_dist);
    computation c2("{c2[y]: 0<=y<" + S(NROWS) + " }", expr(o_allocate, buff_sobel_y.get_name()), true, 
		   tiramisu::p_none, &sobel_dist);
    buff_sobel_x.set_auto_allocate(false);
    buff_sobel_y.set_auto_allocate(false);
    c1.split(y, ROWS_PER_NODE, y1, y2);
    c2.split(y, ROWS_PER_NODE, y1, y2);
    c1.tag_distribute_level(y1);
    c2.tag_distribute_level(y1);

    exchange_fwd.s->before(*exchange_fwd.r, computation::root);
    exchange_fwd.r->before(*exchange_back.s, computation::root);
    exchange_back.s->before(*exchange_back.r, computation::root);
    exchange_back.r->before(c1, computation::root);
    c1.before(c2, y2);
    c2.before(sobel_x, y2);
    sobel_x.before(sobel_y, y2);
    sobel_y.before(sobel, y2);

    input.set_access("[rank]->{input[y,x]->buff_input[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank)+1)}");
    sobel_x.set_access("[rank]->{sobel_x[y,x]->buff_sobel_x[x]}");//: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    sobel_y.set_access("[rank]->{sobel_y[y,x]->buff_sobel_y[x]}");//: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    sobel.set_access("[rank]->{sobel[y,x]->buff_sobel[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    exchange_back.r->set_access("{exchange_back_r[q,y,x]->buff_input[" + S(ROWS_PER_NODE) + "+1,x]}"); 
    exchange_fwd.r->set_access("{exchange_fwd_r[q,y,x]->buff_input[0,x]}"); 

    sobel_dist.codegen({&buff_input, /*&buff_sobel_x, &buff_sobel_y,*/ &buff_sobel}, "build/generated_fct_sobel_dist.o");
    sobel_dist.dump_halide_stmt();

    return 0;
}
