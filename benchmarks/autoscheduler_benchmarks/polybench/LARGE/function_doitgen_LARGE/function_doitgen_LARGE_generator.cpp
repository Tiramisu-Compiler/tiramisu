#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_doitgen_LARGE_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv) {
    tiramisu::init("function_doitgen_LARGE");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

  // Iteration variables
  var r("r", 0, 150), q("q", 0, 140), p("p", 0, 160), s("s", 0, 160);

  // inputs
  input A("A", {r, q, s}, p_float64);
  input x("x", {p, s}, p_float64);
  input sum("sum", {p}, p_float64);

  // Computations
  computation sum_init("sum_init", {r, q, p}, 0.0);
  computation sum_comp("sum_comp", {r, q, p, s}, sum(p) + A(r, q, s) * x(s, p));
  computation A_out("A_out", {r, q, p}, sum(p));


  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------
  sum_init.then(sum_comp, p)
            .then(A_out, q);
  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------
  // Input Buffers
  buffer b_A("b_A", {150, 140, 160}, p_float64, a_output);
  buffer b_sum("b_sum", {160}, p_float64, a_temporary);
  buffer b_x("b_x", {160, 160}, p_float64, a_input);


  // Store inputs
  sum.store_in(&b_sum);
  A.store_in(&b_A);
  x.store_in(&b_x);

  // Store computations
  sum_init.store_in(&b_sum, {p});
  sum_comp.store_in(&b_sum, {p});
  A_out.store_in(&b_A, {r, q, p});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_x}, "function_doitgen_LARGE.o");

    return 0;
}
