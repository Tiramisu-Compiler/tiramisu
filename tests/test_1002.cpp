#include "wrapper_test_1002.h"
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
  var round("round", expr(0), expr(NUM_RANKS));

  input input("input", {k}, p_int64);

  xfer comm = computation::create_xfer(
      "[NUM_RANKS]->{send[round,i,j]: 0<=round<NUM_RANKS and 0<=i<NUM_RANKS and 0<=j<NUM_RANKS}",
      "[NUM_RANKS]->{recv[round,i,j]: 0<=round<NUM_RANKS and 0<=i<NUM_RANKS and 0<=j<NUM_RANKS}",
      expr(j),
      expr(i),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      input((i*j + i/(j+1) + j/(i+1)) % NUM_RANKS),
      fn0);

  tiramisu::wait wait_send((*comm.s)(round,i,j), xfer_prop(p_wait_ptr, {MPI}), fn0);
  tiramisu::wait wait_recv((*comm.r)(round,i,j), xfer_prop(p_wait_ptr, {MPI}), fn0);

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
  comm.r->before(*comm.s, round);
  comm.s->before(wait_recv, round);
  wait_recv.before(wait_send, round);

  // BUG: Shifting the loops for send/recv causes them to split here into two
  // loops, but the second send loop does not calculate the address into
  // buf_input correctly. Instead of using the expression
  // (i*j + i/(j+1) + j/(i+1)) % NUM_RANKS
  // which was given in the create_xfer command, it uses
  // (3*i + j)
  // which is in the wait access relation for the send.
  comm.r->shift(round, -3);
  comm.s->shift(round, -3);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  buffer buf_input ("buf_input",  {expr(NUM_RANKS)}, p_int64, a_input);
  buffer buf_output("buf_output", {expr(NUM_RANKS)}, p_int64, a_output);
  buffer buf_wait_send("buf_wait_send", {expr(NUM_RANKS)}, p_wait_ptr, a_input);
  buffer buf_wait_recv("buf_wait_recv", {expr(NUM_RANKS)}, p_wait_ptr, a_input);

  input.store_in(&buf_input);
  comm.r->store_in(&buf_output, {expr((2*j+i) % num_ranks)});
  comm.s->set_wait_access("{send[round,i,j]->buf_wait_send[(3*i + j) % " + std::to_string(num_ranks) + "]}");
  comm.r->set_wait_access("{recv[round,i,j]->buf_wait_recv[(4*j + i) % " + std::to_string(num_ranks) + "]}");

  tiramisu::codegen({&buf_input, &buf_output, &buf_wait_send, &buf_wait_recv}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}

int main(int argc, char **argv)
{
  gen("split_isend_bug", _NUM_RANKS);
  return 0;
}
