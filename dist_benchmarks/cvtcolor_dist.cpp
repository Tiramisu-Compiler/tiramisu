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

// Current format: assumes data is already pre-distributed.

#define S(s) std::to_string(s)
#define CV_DESCALE(x,n) (((x) + (U(1) << ((n)-U(1)))) >> (n))

int main() {
    // Set default tiramisu options.
  global::set_default_tiramisu_options();

  tiramisu::function cvtcolor_dist("cvtcolor_dist");
    
  int ROWS_PER_NODE = NROWS / NNODES;

  var y("y"), x("x"), c("c");
  tiramisu::computation input("{input[c, y, x]: 0<=c<3 and 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", 
			      expr(), false, tiramisu::p_uint32, &cvtcolor_dist);

  tiramisu::expr rgb_expr(input(2, y, x) * U(1868) + input(1, y, x) * U(9617) + input(0, y, x) * U(4899));
  tiramisu::expr cv_descale = CV_DESCALE(rgb_expr, U(14));
  tiramisu::computation RGB2Gray_s0("{RGB2Gray_s0[y, x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", 
				    cv_descale, true, tiramisu::p_uint32, &cvtcolor_dist);
  
  // distribute stuff
  var y1("y1"), y2("y2");
  
  RGB2Gray_s0.split(y, ROWS_PER_NODE, y1, y2);    
  RGB2Gray_s0.tag_distribute_level(y1);
  RGB2Gray_s0.parallelize(y2);
  RGB2Gray_s0.vectorize(x, 8);
    
  tiramisu::buffer buff_input("buff_input", {3, ROWS_PER_NODE, NCOLS}, tiramisu::p_uint32, tiramisu::a_input, &cvtcolor_dist);
  tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {ROWS_PER_NODE, NCOLS}, tiramisu::p_uint32, tiramisu::a_output, &cvtcolor_dist);
  input.set_access("[rank]->{input[c, y, x]->buff_input[c, t, x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
  RGB2Gray_s0.set_access("[rank]->{RGB2Gray_s0[y,x]->buff_RGB2Gray[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
  
  cvtcolor_dist.codegen({&buff_input, &buff_RGB2Gray}, "build/generated_fct_cvtcolor_dist.o");
  cvtcolor_dist.dump_halide_stmt();
  
  return 0;
}

