#include "wrapper_test_1004.h"
#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <Halide.h>

using namespace tiramisu;

// For computation C(i, k) = A(i, j) * B(j, k)
void gen(std::string name, uint64_t i_dim_size, uint64_t j_dim_size, uint64_t k_dim_size) {

  tiramisu::init(name);
  function *fn0 = global::get_implicit_function();

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Create constants for matrix dimension sizes.
  constant I("I", expr((int32_t) i_dim_size));
  constant J("J", expr((int32_t) j_dim_size));
  constant K("K", expr((int32_t) k_dim_size));
  var i("i", 0, I), j("j", 0, J), k("k", 0, K);

  // Create computation to initialize inputs and outputs.
  computation a_init("a_init", {}, tiramisu::expr(
      o_call, "obj_init_a", {I, J}, p_byte_obj));
  computation b_init("b_init", {}, tiramisu::expr(
      o_call, "obj_init_b", {J, K}, p_byte_obj));
  computation c_init("c_init", {}, tiramisu::expr(
      o_call, "obj_init_c", {I, K}, p_byte_obj));

  // Create object computations for inputs.
  input a("a", {i, j}, p_byte_obj);
  input b("b", {j, k}, p_byte_obj);

  // Create computation to perform matrix multiplication.
  computation c("c", {j, i, k}, tiramisu::expr(
      o_call, "obj_matmul", {a(i, j), b(j, k)}, p_byte_obj));

  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------

  // Tile along i and k.
  var i0("i0"), k0("k0"), i1("i1"), k1("k1");
  c.tile(i, k, 3, 3, i0, k0, i1, k1);

  // Order computations.
  a_init.before(b_init, computation::root);
  b_init.before(c_init, computation::root);
  c_init.before(c, computation::root);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  // Make state buffers for object computations and dummy buffers for function call outputs.
  buffer buf_a("buf_a", {expr(sizeof(obj_mat) + i_dim_size * j_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);
  buffer buf_b("buf_b", {expr(sizeof(obj_mat) + j_dim_size * k_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);
  buffer buf_c("buf_c", {expr(sizeof(obj_mat) + i_dim_size * k_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);

  // Map computations to buffers.
  a.set_object_buffer(&buf_a);
  b.set_object_buffer(&buf_b);
  c.set_object_buffer(&buf_c);
  a_init.set_object_buffer(&buf_a);
  b_init.set_object_buffer(&buf_b);
  c_init.set_object_buffer(&buf_c);

  tiramisu::codegen({&buf_a, &buf_b, &buf_c}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}





int main(int argc, char **argv)
{
  gen("mm_obj", _I_DIM_SIZE, _J_DIM_SIZE, _K_DIM_SIZE);
  return 0;
}
