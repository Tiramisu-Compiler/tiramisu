#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_bicg_MINI_wrapper.h"


using namespace tiramisu;

int main(int argc, char **argv) {
  tiramisu::init("function_bicg_MINI");

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Iteration variables
  var i("i", 0, 42), j("j", 0, 38);

  // inputs
  input A("A", {i, j}, p_float64);
  input p("p", {i}, p_float64);
  input r("r", {j}, p_float64);

  // Computations
  computation q_init("q_init", {i}, 0.0);
  computation q("q", {i, j}, p_float64);
  q.set_expression(q(i, j) + A(i, j) * p(j));
  computation s_init("s_init", {j}, 0.0);
  computation s("s", {i, j}, p_float64);
  s.set_expression(s(i, j) + A(i, j) * r(i));

  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------
  q_init.then(q, computation::root)
        .then(s_init, computation::root)
        .then(s, computation::root);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------
  // Input Buffers
  buffer b_A("b_A", {42, 38}, p_float64, a_input);
  buffer b_p("b_p", {38}, p_float64, a_input);
  buffer b_r("b_r", {42}, p_float64, a_input);
  buffer b_q("b_q", {42}, p_float64, a_output);
  buffer b_s("b_s", {38}, p_float64, a_output);

  // Store inputs
  A.store_in(&b_A);
  p.store_in(&b_p);
  r.store_in(&b_r);
  q.store_in(&b_q);
  s.store_in(&b_s);

  // Store computations
  q_init.store_in(&b_q);
  q.store_in(&b_q, {i});
  s_init.store_in(&b_s);
  s.store_in(&b_s, {j});

  // -------------------------------------------------------
  // Code Generation
  // -------------------------------------------------------
  tiramisu::codegen({&b_A, &b_p, &b_r, &b_q, &b_s}, "function_bicg_MINI.o");

  return 0;
}
