#include "wrapper_test_1000.h"
#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <Halide.h>

using namespace tiramisu;





// For computation C(i, k) = A(i, j) * B(j, k)
void gen(std::string name, int num_ranks, uint64_t i_dim_size, uint64_t j_dim_size, uint64_t k_dim_size, uint64_t comm_shift) {

  tiramisu::init(name);
  function *fn0 = global::get_implicit_function();

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Create constants for inputs.
  constant NUM_RANKS("NUM_RANKS", expr((int32_t) num_ranks));
  constant I("I", expr((int32_t) i_dim_size));
  constant J("J", expr((int32_t) j_dim_size));
  constant K("K", expr((int32_t) k_dim_size));
  constant COMM_SHIFT("COMM_SHIFT", expr((int32_t) comm_shift));

  // Create constant for mapping key sizes and max number of tasks.
  constant MAP_C_SIZE("MAP_C_SIZE", expr(expr(I) * expr(K)));
  constant MAP_A_SIZE("MAP_A_SIZE", expr(expr(I) * expr(J)));
  constant MAP_B_SIZE("MAP_B_SIZE", expr(expr(J) * expr(K)));
  constant NUM_TASKS_C("NUM_TASKS_C", expr(((MAP_C_SIZE - 1) / NUM_RANKS) + 1));
  constant NUM_TASKS_A("NUM_TASKS_A", expr(((MAP_A_SIZE - 1) / NUM_RANKS) + 1));
  constant NUM_TASKS_B("NUM_TASKS_B", expr(((MAP_B_SIZE - 1) / NUM_RANKS) + 1));

  // Create vars for distributing computation of C.
  var j("j", expr((int32_t) 0), expr(J));
  var map_c("map_c", expr((int32_t) 0), expr(MAP_C_SIZE)), rnk_c("rnk_c"), tsk_c("tsk_c");

  // Create vars for distributing original blocks of A and B.
  var map_a("map_a", expr((int32_t) 0), expr(MAP_A_SIZE)), rnk_a("rnk_a"), tsk_a("tsk_a");
  var map_b("map_b", expr((int32_t) 0), expr(MAP_B_SIZE)), rnk_b("rnk_b"), tsk_b("tsk_b");

  // Create computation for original input blocks (srcs of MPI send).
  input a_in("a_in", {map_a}, p_int64);
  input b_in("b_in", {map_b}, p_int64);

  // Create computation for local input blocks (dsts of MPI recv).
  input a_local("a_local", {j, map_c}, p_int64);
  input b_local("b_local", {j, map_c}, p_int64);

  // Create computation to initialize output blocks.
  computation c_local_init("c_local_init", {map_c}, expr((int64_t) 0));

  // Create computation to sum into output blocks.
  computation c_local("c_local", {j, map_c}, p_int64);
  c_local.set_expression(c_local(j-1, map_c) + a_local(j, map_c) * b_local(j, map_c));

  // Create (non-blocking, non-synchronous) MPI send/recv for A blocks.
  constant ONE("ONE", expr((int32_t) 1));
  std::string map_a_str("[map_c/" + std::to_string(k_dim_size) + "]*" + std::to_string(j_dim_size) + "+j");
  std::string rnk_a_str("(" + map_a_str + ")%" + std::to_string(num_ranks));
  std::string rnk_a_plus_one_str("(" + rnk_a_str + ")+ONE");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{a_send[j,map_c,rnk_a]: 0<=j<J and 0<=map_c<MAP_C_SIZE and ([map_c/K]*J+j)%NUM_RANKS<=rnk_a<(([map_c/K]*J+j)%NUM_RANKS)+1}
  std::string a_iter_send("[J,K,NUM_RANKS,MAP_C_SIZE,ONE]->{a_send[j,map_c,rnk_a]: 0<=j<J and 0<=map_c<MAP_C_SIZE and " + rnk_a_str + "<=rnk_a<" + rnk_a_plus_one_str + "}");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{a_recv[j,map_c]: 0<=j<J and 0<=map_c<MAP_C_SIZE}
  std::string a_iter_recv("[J,K,NUM_RANKS,MAP_C_SIZE]->{a_recv[j,map_c]: 0<=j<J and 0<=map_c<MAP_C_SIZE}");
  xfer comm_a = computation::create_xfer(
      a_iter_send,
      a_iter_recv,
      expr(map_c % NUM_RANKS),
      expr(((map_c / K) * J + j) % NUM_RANKS),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      a_in((map_c / K) * J + j),
      fn0);

  // Create waits for MPI send/recv for A blocks.
  tiramisu::wait wait_a_send((*comm_a.s)(j,map_c,rnk_a), xfer_prop(p_wait_ptr, {MPI}), fn0);
  tiramisu::wait wait_a_recv((*comm_a.r)(j,map_c), xfer_prop(p_wait_ptr, {MPI}), fn0);

  // Create (non-blocking, non-synchronous) MPI send/recv for B blocks.
  std::string map_b_str("j*" + std::to_string(k_dim_size) + "+(map_c%" + std::to_string(k_dim_size) + ")");
  std::string rnk_b_str("(" + map_b_str + ")%" + std::to_string(num_ranks));
  std::string rnk_b_plus_one_str("(" + rnk_b_str + ")+ONE");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{b_send[j,map_c,rnk_b]: 0<=j<J and 0<=map_c<MAP_C_SIZE and (j*K+(map_c%K))%NUM_RANKS<=rnk_b<((j*K+(map_c%K))%NUM_RANKS)+1}
  std::string b_iter_send("[J,K,NUM_RANKS,MAP_C_SIZE,ONE]->{b_send[j,map_c,rnk_b]: 0<=j<J and 0<=map_c<MAP_C_SIZE and " + rnk_b_str + "<=rnk_b<" + rnk_b_plus_one_str + "}");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{b_recv[j,map_c]: 0<=j<J and 0<=map_c<MAP_C_SIZE}
  std::string b_iter_recv("[J,K,NUM_RANKS,MAP_C_SIZE]->{b_recv[j,map_c]: 0<=j<J and 0<=map_c<MAP_C_SIZE}");
  xfer comm_b = computation::create_xfer(
      b_iter_send,
      b_iter_recv,
      expr(map_c % NUM_RANKS),
      expr((j * K + (map_c % K)) % NUM_RANKS),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, NONBLOCK, ASYNC}),
      b_in(j * K + (map_c % K)),
      fn0);

  // Create waits for MPI send/recv for B blocks.
  tiramisu::wait wait_b_send((*comm_b.s)(j,map_c,rnk_b), xfer_prop(p_wait_ptr, {MPI}), fn0);
  tiramisu::wait wait_b_recv((*comm_b.r)(j,map_c), xfer_prop(p_wait_ptr, {MPI}), fn0);



  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------

  // Distribute computation of output blocks.
  a_local.split(map_c, num_ranks, tsk_c, rnk_c);
  b_local.split(map_c, num_ranks, tsk_c, rnk_c);
  c_local_init.split(map_c, num_ranks, tsk_c, rnk_c);
  c_local.split(map_c, num_ranks, tsk_c, rnk_c);

  a_local.tag_distribute_level(rnk_c);
  b_local.tag_distribute_level(rnk_c);
  c_local_init.tag_distribute_level(rnk_c);
  c_local.tag_distribute_level(rnk_c);
  
  // Distribute MPI sends/recvs.
  comm_a.r->split(map_c, num_ranks, tsk_c, rnk_c);
  comm_b.r->split(map_c, num_ranks, tsk_c, rnk_c);

  comm_a.s->tag_distribute_level(rnk_a);
  comm_a.r->tag_distribute_level(rnk_c);
  comm_b.s->tag_distribute_level(rnk_b);
  comm_b.r->tag_distribute_level(rnk_c);

  // Distribute MPI waits.
  wait_a_recv.split(map_c, num_ranks, tsk_c, rnk_c);
  wait_b_recv.split(map_c, num_ranks, tsk_c, rnk_c);

  wait_a_send.tag_distribute_level(rnk_a);
  wait_a_recv.tag_distribute_level(rnk_c);
  wait_b_send.tag_distribute_level(rnk_b);
  wait_b_recv.tag_distribute_level(rnk_c);

  // Order computations and communication
  c_local_init.before(*comm_a.r, computation::root);
  comm_a.r->before(*comm_b.r, rnk_c);
  comm_b.r->before(*comm_a.s, j);
  comm_a.s->before(*comm_b.s, j);
  comm_b.s->before(wait_a_recv, j);
  wait_a_recv.before(wait_b_recv, rnk_c);
  wait_b_recv.before(c_local, rnk_c);
  c_local.before(wait_a_send, j);
  wait_a_send.before(wait_b_send, j);

  // Shift the sends/recvs backward in j so they overlap with computation.
  comm_a.r->shift(j, (int32_t) -comm_shift);
  comm_b.r->shift(j, (int32_t) -comm_shift);
  comm_a.s->shift(j, (int32_t) -comm_shift);
  comm_b.s->shift(j, (int32_t) -comm_shift);



  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  // Make buffers long enough to store each task separately.
  buffer buf_a_in("buf_a_in", {expr(NUM_TASKS_A)}, p_int64, a_input);
  buffer buf_b_in("buf_b_in", {expr(NUM_TASKS_B)}, p_int64, a_input);
  buffer buf_wait_a_send("buf_wait_a_send", {expr(COMM_SHIFT + 1), expr(K), expr(NUM_TASKS_A)}, p_wait_ptr, a_input);
  buffer buf_wait_a_recv("buf_wait_a_recv", {expr(COMM_SHIFT + 1), expr(NUM_TASKS_C)}, p_wait_ptr, a_input);
  buffer buf_wait_b_send("buf_wait_b_send", {expr(COMM_SHIFT + 1), expr(I), expr(NUM_TASKS_B)}, p_wait_ptr, a_input);
  buffer buf_wait_b_recv("buf_wait_b_recv", {expr(COMM_SHIFT + 1), expr(NUM_TASKS_C)}, p_wait_ptr, a_input);
  buffer buf_a_local("buf_a_local", {expr(COMM_SHIFT + 1), expr(NUM_TASKS_C)}, p_int64, a_input);
  buffer buf_b_local("buf_b_local", {expr(COMM_SHIFT + 1), expr(NUM_TASKS_C)}, p_int64, a_input);
  buffer buf_c_local("buf_c_local", {expr(NUM_TASKS_C)}, p_int64, a_output);

  // Map computations to buffers.
  a_in.store_in(&buf_a_in, {expr(map_a / num_ranks)});
  b_in.store_in(&buf_b_in, {expr(map_b / num_ranks)});
  comm_a.r->store_in(&buf_a_local, {expr(j % ((int32_t) comm_shift + 1)), expr(map_c / num_ranks)});
  comm_b.r->store_in(&buf_b_local, {expr(j % ((int32_t) comm_shift + 1)), expr(map_c / num_ranks)});
  //comm_a.s->set_wait_access("[J,K,NUM_RANKS,COMM_SHIFT]->{a_send[j,map_c,rnk_a]->buf_wait_a_send[j%(COMM_SHIFT+1), map_c%NUM_RANKS, [([map_c/K]*J+j)/NUM_RANKS]]}");
  //comm_a.r->set_wait_access("[NUM_RANKS,COMM_SHIFT]->{a_recv[j,map_c]->buf_wait_a_recv[j%(COMM_SHIFT+1), [map_c/NUM_RANKS]]}");
  //comm_b.s->set_wait_access("[K,NUM_RANKS,COMM_SHIFT]->{b_send[j,map_c,rnk_b]->buf_wait_b_send[j%(COMM_SHIFT+1), [map_c/NUM_RANKS], [(j*K+(map_c%K))/NUM_RANKS]]}");
  //comm_b.r->set_wait_access("[NUM_RANKS,COMM_SHIFT]->{b_recv[j,map_c]->buf_wait_b_recv[j%(COMM_SHIFT+1), [map_c/NUM_RANKS]]}");
  std::string tsk_a_str("[(" + map_a_str + ")/" + std::to_string(num_ranks) + "]");
  std::string tsk_b_str("[(" + map_b_str + ")/" + std::to_string(num_ranks) + "]");
  std::string tsk_c_str("[map_c/" + std::to_string(num_ranks) + "]");
  std::string i_str("[map_c/" + std::to_string(k_dim_size) + "]");
  std::string k_str("map_c%" + std::to_string(k_dim_size));
  std::string comm_shift_str("j%" + std::to_string(comm_shift + 1));
  comm_a.s->set_wait_access("{a_send[j,map_c,rnk_a]->buf_wait_a_send[(" + comm_shift_str + "), (" + k_str + "), (" + tsk_a_str + ")]}");
  comm_a.r->set_wait_access("{a_recv[j,map_c]->buf_wait_a_recv[(" + comm_shift_str + "), (" + tsk_c_str + ")]}");
  comm_b.s->set_wait_access("{b_send[j,map_c,rnk_b]->buf_wait_b_send[(" + comm_shift_str + "), (" + i_str + "), (" + tsk_b_str + ")]}");
  comm_b.r->set_wait_access("{b_recv[j,map_c]->buf_wait_b_recv[(" + comm_shift_str + "), (" + tsk_c_str + ")]}");
  a_local.store_in(&buf_a_local, {expr(j % ((int32_t) comm_shift + 1)), expr(map_c / num_ranks)});
  b_local.store_in(&buf_b_local, {expr(j % ((int32_t) comm_shift + 1)), expr(map_c / num_ranks)});
  c_local_init.store_in(&buf_c_local, {expr(map_c / num_ranks)});
  c_local.store_in(&buf_c_local, {expr(map_c / num_ranks)});

  tiramisu::codegen({&buf_a_in, &buf_b_in, &buf_wait_a_send, &buf_wait_a_recv, &buf_wait_b_send, &buf_wait_b_recv, &buf_a_local, &buf_b_local, &buf_c_local}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}





int main(int argc, char **argv)
{
  gen("spmm", _NUM_RANKS, _I_DIM_SIZE, _J_DIM_SIZE, _K_DIM_SIZE, _COMM_SHIFT);
  return 0;
}
