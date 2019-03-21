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
#include "wrapper_gaussian_dist.h"

using namespace tiramisu;

// Current format: assumes data is already pre-distributed. Communication is only needed to transfer over shared zones
#define S(s) std::to_string(s)
#define MIN(bound, actual) expr(o_min, bound, actual)
#define MAX(bound, actual) expr(o_max, bound, actual)
#define fcast(val) expr(o_cast, p_float32, val)
#define icast(val) expr(o_cast, p_int32, val)

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    tiramisu::function gaussian_dist("gaussian_dist");

    int ROWS_PER_NODE = NROWS / NNODES;

    float kernel_x[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float kernel_y[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    computation input("{input[c,y,x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, p_float32, &gaussian_dist);

    var x("x"), y("y"), c("c");
    expr e = 0.0f;    
    for (int i = 0; i < 5; i++) {
      e = e + input(c, y, MIN(NCOLS-1, x + i)) * kernel_x[i];
    }

    computation gaussian_x("{gaussian_x[c,y,x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", e, true, p_float32, &gaussian_dist);

    expr f = 0.0f;
    for (int j = 0; j < 5; j++) {
      f = f + gaussian_x(c, MIN(NROWS-1, y + j), x) * kernel_y[j];
    }

    computation gaussian("{gaussian[c,y,x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", f, true, p_float32, &gaussian_dist);

    constant nodes("nodes", expr(NNODES), p_float32, true, NULL, 0, &gaussian_dist);
    var y1("y1"), y2("y2"), q("q");
    gaussian_x.interchange(c, y);
    gaussian.interchange(c, y);
    gaussian_x.split(y, ROWS_PER_NODE, y1, y2);
    gaussian.split(y, ROWS_PER_NODE, y1, y2);    
    gaussian_x.interchange(c, y2);
    gaussian.interchange(c, y2);
    gaussian_x.parallelize(c);
    gaussian.parallelize(c);

    // Need to transfer 4 rows backwards
    xfer exchange_back = 
      computation::create_xfer("[nodes,rank]->{exchange_back_s[q,c,y,x]: 1<=q<nodes and 0<=c<3 and (rank*" +
			       S(ROWS_PER_NODE) 
			       + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+4) and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_back_r[q,c,y,x]: 0<=q<(nodes-1) and 0<=c<3 and (rank*" + 
			       S(ROWS_PER_NODE) + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+4) and 0<=x<" + S(NCOLS) + "}",
			       q-1, q+1, 
			       xfer_prop(p_float32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_float32, {SYNC, BLOCK, MPI}), gaussian_x(c,y,x), &gaussian_dist);

    gaussian_x.tag_distribute_level(y1);
    gaussian.tag_distribute_level(y1);
    exchange_back.s->tag_distribute_level(q);
    exchange_back.r->tag_distribute_level(q);
    exchange_back.s->collapse_many({collapse_group(3, 0, -1, NCOLS)});
    exchange_back.r->collapse_many({collapse_group(3, 0, -1, NCOLS)});

    gaussian_x.before(*exchange_back.s, computation::root);
    exchange_back.s->before(*exchange_back.r, computation::root);
    exchange_back.r->before(gaussian, computation::root);
    
    tiramisu::buffer buff_input("buff_input", {3, ROWS_PER_NODE, NCOLS}, tiramisu::p_float32, tiramisu::a_input, &gaussian_dist);
    tiramisu::buffer buff_gaussian_x("buff_gaussian_x", {3, ROWS_PER_NODE+4, NCOLS}, tiramisu::p_float32, tiramisu::a_input, &gaussian_dist);
    tiramisu::buffer buff_gaussian("buff_gaussian", {3, ROWS_PER_NODE, NCOLS}, tiramisu::p_float32, tiramisu::a_output, &gaussian_dist);
    
    input.set_access("[rank]->{input[c, y, x]->buff_input[c, t, x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    gaussian_x.set_access("[rank]->{gaussian_x[c,y,x]->buff_gaussian_x[c,t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    gaussian.set_access("[rank]->{gaussian[c,y,x]->buff_gaussian[c,t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    exchange_back.r->set_access("[rank]->{exchange_back_r[q,c,y,x]->buff_gaussian_x[c,t,x]: t=(" + S(ROWS_PER_NODE) + "+(y-(" + S(ROWS_PER_NODE) + "*rank)))}");
    
    gaussian_dist.codegen({&buff_input, &buff_gaussian_x, &buff_gaussian}, "build/generated_fct_gaussian_dist.o");
    gaussian_dist.dump_halide_stmt();

    return 0;
}
