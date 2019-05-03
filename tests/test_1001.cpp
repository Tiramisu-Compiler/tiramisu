#include "wrapper_test_1001.h"
#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <Halide.h>

using namespace tiramisu;

void gen(std::string name, int num_ranks) {

  tiramisu::init(name);
  function *fn0 = global::get_implicit_function();

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  constant NUM_RANKS("NUM_RANKS", expr(num_ranks));

  var i("i", expr(0), expr(NUM_RANKS));
  var j("j", expr(0), expr(NUM_RANKS));
  var k("k", expr(0), expr(NUM_RANKS));

  input input("input", {k}, p_int64);

  xfer comm = computation::create_xfer(
      "[NUM_RANKS]->{send[i,j]: 0<=i<NUM_RANKS and 0<=j<NUM_RANKS}",
      "[NUM_RANKS]->{recv[i,j]: 0<=i<NUM_RANKS and 0<=j<NUM_RANKS}",
      expr(j),
      expr(i),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      input((i*j + i/(j+1) + j/(i+1)) % NUM_RANKS),
      fn0);

  tiramisu::wait wait_send((*comm.s)(i,j), xfer_prop(p_wait_ptr, {MPI}), fn0);
  tiramisu::wait wait_recv((*comm.r)(i,j), xfer_prop(p_wait_ptr, {MPI}), fn0);

  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------
  
  // Distribute MPI sends/recvs.
  comm.s->tag_distribute_level(i);
  comm.r->tag_distribute_level(j);

  // Distribute MPI waits.
  wait_send.tag_distribute_level(i);
  wait_recv.tag_distribute_level(j);

  // Schedule computations.
  comm.r->before(*comm.s, computation::root);
  // BUG: The generated Halide IR has the first loop with both indices 
  // distributed, despite only i being tagged distributed for comm.s. This is
  // oddly fixed if we change the j in the below line to computation::root.
  comm.s->before(wait_send, j);
  wait_send.before(wait_recv, computation::root);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  buffer buf_input ("buf_input",  {expr(NUM_RANKS)}, p_int64, a_input);
  buffer buf_output("buf_output", {expr(NUM_RANKS)}, p_int64, a_output);
  buffer buf_wait_send("buf_wait_send", {expr(NUM_RANKS)}, p_wait_ptr, a_input);
  buffer buf_wait_recv("buf_wait_recv", {expr(NUM_RANKS)}, p_wait_ptr, a_input);

  input.store_in(&buf_input);
  comm.r->store_in(&buf_output, {expr((2*j+i) % num_ranks)});
  comm.s->set_wait_access("{send[i,j]->buf_wait_send[(3*i + j) % " + std::to_string(num_ranks) + "]}");
  comm.r->set_wait_access("{recv[i,j]->buf_wait_recv[(4*j + i) % " + std::to_string(num_ranks) + "]}");

  tiramisu::codegen({&buf_input, &buf_output, &buf_wait_send, &buf_wait_recv}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}

int main(int argc, char **argv)
{
  gen("double_distribute_bug", _NUM_RANKS);
  return 0;
}
