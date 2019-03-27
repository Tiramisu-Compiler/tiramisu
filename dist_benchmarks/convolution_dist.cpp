#include <tiramisu/tiramisu.h>

#include <Halide.h>
#include "halide_image_io.h"
#include "wrapper_convolution_dist.h"

using namespace tiramisu;

#define S(s) std::to_string(s)
#define MIN(bound, actual) expr(o_min, bound, actual)
#define MAX(bound, actual) expr(o_max, bound, actual)
#define LOOP_TYPE p_int64
#define C_TYPE int64_t
#define T(val) expr(o_cast, LOOP_TYPE, val)
#define TC(val) (C_TYPE)val
#define VAR(name) var name(LOOP_TYPE, #name)

int main(int argc, char **argv)
{

    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(LOOP_TYPE);

    tiramisu::function convolution_dist("convolution_dist");

    int64_t ROWS_PER_NODE = NROWS / NUM_MPI_RANKS;

    int32_t kernel[3][3] = {{2, 1, 2},
			    {1, 1, 1},
			    {2, 1, 2}};
    
    computation input("{input[c,y,x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, 
		      p_int32, &convolution_dist);

    VAR(x); VAR(y); VAR(c);

    expr e = 0;
    for (int64_t j = 0; j < 3; j++) {
        for (int64_t i = 0; i < 3; i++) {
	  e = e + input(c, MIN(T(NROWS-1), y + j), MIN(T(NCOLS-1), x + i)) * kernel[i][j];
        }
    }

    constant nodes("nodes", expr(o_cast, LOOP_TYPE, NUM_MPI_RANKS), LOOP_TYPE, true, NULL, 0, &convolution_dist);
    constant channels("channels", expr(o_cast, LOOP_TYPE, 3), LOOP_TYPE, true, NULL, 0, &convolution_dist);
    computation convolution("{convolution[c,y,x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", e, true, p_int32, 
			    &convolution_dist);

    VAR(y1); VAR(y2); VAR(q);
    convolution.interchange(c, y);
    convolution.split(y, ROWS_PER_NODE, y1, y2);
    convolution.interchange(c, y2);
    convolution.vectorize(x, 8);
    convolution.parallelize(c);

    xfer exchange_back = 
      computation::create_xfer("[nodes,rank]->{exchange_back_s[q,c,y,x]: 1<=q<nodes and 0<=c<3 and (rank*" +
			       S(ROWS_PER_NODE) 
			       + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+2) and 0<=x<" + S(NCOLS) + "}", 
			       "[nodes,rank]->{exchange_back_r[q,c,y,x]: 0<=q<(nodes-1) and 0<=c<3 and (rank*" + 
			       S(ROWS_PER_NODE) + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+2) and 0<=x<" + S(NCOLS) + "}",
			       q-1, q+1, 
			       xfer_prop(p_int32, {ASYNC, BLOCK, MPI}), 
			       xfer_prop(p_int32, {SYNC, BLOCK, MPI}), input(c,y,x), &convolution_dist);

    convolution.tag_distribute_level(y1);
    exchange_back.s->tag_distribute_level(q);
    exchange_back.r->tag_distribute_level(q);
    exchange_back.s->collapse_many({collapse_group(3, TC(0), -1, TC(NCOLS))});
    exchange_back.r->collapse_many({collapse_group(3, TC(0), -1, TC(NCOLS))});
    exchange_back.s->before(*exchange_back.r, computation::root);
    exchange_back.r->before(convolution, computation::root);

    tiramisu::buffer buff_input("buff_input", {channels, TC(ROWS_PER_NODE+2), TC(NCOLS)}, tiramisu::p_int32, tiramisu::a_input, &convolution_dist);
    tiramisu::buffer buff_convolution("buff_convolution", {channels, TC(ROWS_PER_NODE), TC(NCOLS)}, tiramisu::p_int32, tiramisu::a_output, &convolution_dist);
    
    input.set_access("[rank]->{input[c, y, x]->buff_input[c,t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    convolution.set_access("[rank]->{convolution[c,y,x]->buff_convolution[c,t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
    exchange_back.r->set_access("[rank]->{exchange_back_r[q,c,y,x]->buff_input[c,t,x]: t=(" + S(ROWS_PER_NODE) + "+(y-(" + S(ROWS_PER_NODE) + "*rank)))}");
    
    convolution_dist.codegen({&buff_input, &buff_convolution}, "build/generated_fct_convolution_dist.o");
    convolution_dist.dump_halide_stmt();


    return 0;
}

