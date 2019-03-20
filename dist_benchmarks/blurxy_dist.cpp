// Current format: assumes data is already pre-distributed. Communication is only needed to transfer over shared zones

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
#include "wrapper_blurxy_dist.h"

using namespace tiramisu;

#define E(e) tiramisu::expr(e)
#define S(s) std::to_string(s)
#define MIN(bound, actual) expr(o_min, bound, actual)
#define MAX(bound, actual) expr(o_max, bound, actual)

int main() {
    global::set_default_tiramisu_options();

    function blurxy_dist("blurxy_dist");

    int ROWS_PER_NODE = NROWS / NNODES;

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    var y("y"), x("x");
    
    computation input("{input[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, 
		      p_int32, &blurxy_dist);
    
    computation bx("{bx[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",
		   (input(y,x) + input(y,MAX(0,x-1)) + input(y,MIN(NCOLS-1, x+1)))/3,
                   true, p_int32, &blurxy_dist);
    
    computation by("{by[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}",		  
		   (bx(MAX(0,y-1),x) + bx(y,x) + bx(MIN(NROWS-1,y+1),x))/3,
                   true, p_int32, &blurxy_dist);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    constant nodes("nodes", expr(NNODES), p_int32, true, NULL, 0, &blurxy_dist);
    var y1("y1"), y2("y2"), d("d"), q("q");

    bx.split(y, ROWS_PER_NODE, y1, y2);
    by.split(y, ROWS_PER_NODE, y1, y2);
    //    bx.vectorize(x, 8);
    //    by.vectorize(x, 8);
    //    by.parallelize(y2);
    //    bx.parallelize(y2);

    xfer exchange_back = 
      computation::create_xfer("[nodes,rank]->{exchange_back_s[q,y,x]: 1<=q<nodes and (rank*" + S(ROWS_PER_NODE) 
			       + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+1) and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_back_r[q,y,x]: 0<=q<(nodes-1) and (rank*" + 
			       S(ROWS_PER_NODE) + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+1) and 0<=x<" + S(NCOLS) + "}",
			       q-1, q+1, 
			       xfer_prop(p_int32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_int32, {SYNC, BLOCK, MPI}), bx(y,x), &blurxy_dist);

    xfer exchange_fwd = 
      computation::create_xfer("[nodes,rank]->{exchange_fwd_s[q,y,x]: 0<=q<(nodes-1) and ((rank+1)*" + 
			       S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_fwd_r[q,y,x]: 1<=q<nodes and ((rank+1)*" + 
			       S(ROWS_PER_NODE) + "-1)<=y<((rank+1)*" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}",
			       q+1, q-1, 
			       xfer_prop(p_int32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_int32, {SYNC, BLOCK, MPI}), bx(y,x), &blurxy_dist);

    bx.tag_distribute_level(y1);
    by.tag_distribute_level(y1);
    exchange_back.s->tag_distribute_level(q);
    exchange_back.r->tag_distribute_level(q);
    exchange_fwd.s->tag_distribute_level(q);
    exchange_fwd.r->tag_distribute_level(q);    
    
    //    exchange_back.s->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    //    exchange_back.r->collapse_many({collapse_group(2, 0, -1, NCOLS)});
//    exchange_fwd.s->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    //    exchange_fwd.r->collapse_many({collapse_group(2, 0, -1, NCOLS)});
    bx.before(*exchange_fwd.s, computation::root);
    exchange_fwd.s->before(*exchange_fwd.r, computation::root);
    exchange_fwd.r->before(*exchange_back.s, computation::root);
    exchange_back.s->before(*exchange_back.r, computation::root);
    exchange_back.r->before(by, computation::root);
    
    buffer buff_input("buff_input", {ROWS_PER_NODE, NCOLS}, p_int32, a_input, &blurxy_dist);
    buffer buff_bx("buff_bx", {ROWS_PER_NODE+2, NCOLS}, p_int32, a_input, &blurxy_dist);
    buffer buff_by("buff_by", {ROWS_PER_NODE, NCOLS}, p_int32, a_output, &blurxy_dist);

    input.set_access("[rank]->{input[y,x]->buff_input[t,x]: t=y-(" + S(ROWS_PER_NODE) + "*rank)}");
    bx.set_access("[rank]->{bx[y,x]->buff_bx[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank)+1)}");
    by.set_access("[rank]->{by[y,x]->buff_by[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    exchange_back.r->set_access("{exchange_back_r[q,y,x]->buff_bx[" + S(ROWS_PER_NODE) + "+1,x]}"); 
    exchange_fwd.r->set_access("{exchange_fwd_r[q,y,x]->buff_bx[0,x]}"); 

    blurxy_dist.codegen({&buff_input, &buff_bx, &buff_by}, "build/generated_fct_blurxy_dist.o");
    blurxy_dist.dump_halide_stmt();

    
    return 0;

}
