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
#include "wrapper_cvtcolor_dist.h"

using namespace tiramisu;

#define U(val) tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32, val)
#define LOOP_TYPE p_int64
#define T(val) expr(o_cast, LOOP_TYPE, val)
#define VAR(name) var name(LOOP_TYPE, #name)
#define S(s) std::to_string(s)
#define CV_DESCALE(x,n) (((x) + (U(1) << ((n)-U(1)))) >> (n))

int main() {
  // Set default tiramisu options.
  global::set_default_tiramisu_options();
  global::set_loop_iterator_type(LOOP_TYPE);

  tiramisu::function cvtcolor_dist("cvtcolor_dist");
    
  int64_t ROWS_PER_NODE = NROWS / NUM_MPI_RANKS;

  VAR(x); VAR(y); VAR(c);
  tiramisu::computation input("{input[c, y, x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", 
			      expr(), false, tiramisu::p_uint32, &cvtcolor_dist);

  tiramisu::expr rgb_expr(input(2, y, x) * U(1868) + input(1, y, x) * U(9617) + input(0, y, x) * U(4899));
  tiramisu::expr cv_descale = CV_DESCALE(rgb_expr, U(14));
  tiramisu::computation RGB2Gray_s0("{RGB2Gray_s0[y, x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", 
				    cv_descale, true, tiramisu::p_uint32, &cvtcolor_dist);
  
  // distribute stuff
  VAR(y1); VAR(y2); VAR(y3); VAR(y4);
  
  RGB2Gray_s0.split(y, ROWS_PER_NODE, y1, y2);    
  RGB2Gray_s0.tag_distribute_level(y1);
  RGB2Gray_s0.split(y2, ROWS_PER_NODE/12, y3, y4);
  RGB2Gray_s0.parallelize(y3);
  RGB2Gray_s0.vectorize(x, 8);
  constant chan("chan", expr((int64_t)3), LOOP_TYPE /*This doesn't seem to actually do anything*/, true, NULL, 0, &cvtcolor_dist);  
  // If you put a "3" in for "chan" instead when the dimensions for rows and cols are big, then constant
  // folding is applied somewhere in halide and it overflows. I can't figure out where though, so this 
  // seems to get around the problematic area. folding still happens, but somewhere else. ahhhhh
  tiramisu::buffer buff_input("buff_input", {chan, (int64_t)(ROWS_PER_NODE), (int64_t)(NCOLS)}, tiramisu::p_uint32, 
			      tiramisu::a_input, &cvtcolor_dist);
  tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {(int64_t)(ROWS_PER_NODE), (int64_t)(NCOLS)}, tiramisu::p_uint32, 
				 tiramisu::a_output, &cvtcolor_dist);
  input.set_access("[rank]->{input[c, y, x]->buff_input[c, t, x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
  RGB2Gray_s0.set_access("[rank]->{RGB2Gray_s0[y,x]->buff_RGB2Gray[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
  
  cvtcolor_dist.codegen({&buff_input, &buff_RGB2Gray}, "build/generated_fct_cvtcolor_dist.o");
  cvtcolor_dist.dump_halide_stmt();
  
  return 0;
}

