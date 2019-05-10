#include "wrapper_test_1003.h"
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
  var i("i", 0, I), j("j", 0, J), k("k", 0, K), init("init", 0, 1);

  // Create object computations for inputs and outputs.
  input a("a", {i, j}, p_byte_obj);
  input b("b", {j, k}, p_byte_obj);
  input c("c", {i, k}, p_byte_obj);

  // Create computation to initialize inputs and outputs.
  expr e1(o_call, "obj_init_a", {expr(o_address, var(p_byte_obj, a.get_name())), I, J}, p_int32);
  computation a_init("a_init", {init}, e1);
  expr e2(o_call, "obj_init_b", {expr(o_address, var(p_byte_obj, b.get_name())), J, K}, p_int32);
  computation b_init("b_init", {init}, e2);
  expr e3(o_call, "obj_init_c", {expr(o_address, var(p_byte_obj, c.get_name())), I, K}, p_int32);
  computation c_init("c_init", {init}, e3);

  // Create computation to perform matrix multiplication.
  expr e4(o_call, "obj_matmul", {c(i, k), a(i, j), b(j, k)}, p_int32);
  computation matmul("matmul", {j, i, k}, e4);

  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------

  // Order computations.
  a_init.before(b_init, computation::root);
  b_init.before(c_init, computation::root);
  c_init.before(matmul, computation::root);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  // Make state buffers for object computations and dummy buffers for function call outputs.
  buffer buf_a("buf_a", {expr(sizeof(obj_mat) + i_dim_size * j_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);
  buffer buf_b("buf_b", {expr(sizeof(obj_mat) + j_dim_size * k_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);
  buffer buf_c("buf_c", {expr(sizeof(obj_mat) + i_dim_size * k_dim_size * sizeof(uint64_t))}, p_byte_obj, a_input);
  buffer buf_dummy("buf_dummy", {1}, p_int32, a_output);

  // Map computations to buffers.
  a.set_object_buffer(&buf_a);
  b.set_object_buffer(&buf_b);
  c.set_object_buffer(&buf_c);
  a_init.store_in(&buf_dummy, {0});
  b_init.store_in(&buf_dummy, {0});
  c_init.store_in(&buf_dummy, {0});
  matmul.store_in(&buf_dummy, {0});

  tiramisu::codegen({&buf_a, &buf_b, &buf_c, &buf_dummy}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}





int main(int argc, char **argv)
{
  gen("mm_obj", _I_DIM_SIZE, _J_DIM_SIZE, _K_DIM_SIZE);
  return 0;
}
