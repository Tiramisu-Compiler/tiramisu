#include "wrapper_test_1000.h"
#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <Halide.h>

using namespace tiramisu;

// For computation C(i, k) = A(i, j) * B(j, k)
void gen(std::string name, int num_ranks, uint64_t i_dim_size, uint64_t j_dim_size, uint64_t k_dim_size) {

  tiramisu::init(name);
  function *fn0 = global::get_implicit_function();

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Create constants for matrix dimension sizes.
  constant I("I", expr((int32_t) i_dim_size));
  constant J("J", expr((int32_t) j_dim_size));
  constant K("K", expr((int32_t) k_dim_size));

  // Create constant for max number of nodes and tasks.
  constant NUM_RANKS("NUM_RANKS", expr((int32_t) num_ranks));
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

  // Distribute input blocks.
  // a_in.split(map_a, num_ranks, tsk_a, rnk_a);
  // b_in.split(map_b, num_ranks, tsk_b, rnk_b);

  // a_in.tag_distribute_level(rnk_a);
  // b_in.tag_distribute_level(rnk_b);

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
  c_local_init.before(*comm_a.s, computation::root);
  comm_a.s->before(*comm_b.s, j);
  comm_b.s->before(*comm_a.r, j);
  comm_a.r->before(*comm_b.r, rnk_c);
  comm_b.r->before(wait_a_recv, rnk_c);
  wait_a_recv.before(wait_b_recv, rnk_c);
  wait_b_recv.before(c_local, rnk_c);
  c_local.before(wait_a_send, j);
  wait_a_send.before(wait_b_send, j);

  // c_local_init.before(*comm_a.s, computation::root);
  // comm_a.s->before(*comm_b.s, j);
  // comm_b.s->before(*comm_a.r, j);
  // comm_a.r->before(*comm_b.r, rnk_c);
  // comm_b.r->before(c_local, rnk_c);





  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------

  // Make buffers long enough to store each task separately.
  buffer buf_a_in("buf_a_in", {expr(NUM_TASKS_A)}, p_int64, a_input);
  buffer buf_b_in("buf_b_in", {expr(NUM_TASKS_B)}, p_int64, a_input);
  buffer buf_wait_a_send("buf_wait_a_send", {expr(NUM_TASKS_A)}, p_wait_ptr, a_input);
  buffer buf_wait_a_recv("buf_wait_a_recv", {expr(NUM_TASKS_C)}, p_wait_ptr, a_input);
  buffer buf_wait_b_send("buf_wait_b_send", {expr(NUM_TASKS_B)}, p_wait_ptr, a_input);
  buffer buf_wait_b_recv("buf_wait_b_recv", {expr(NUM_TASKS_C)}, p_wait_ptr, a_input);
  buffer buf_a_local("buf_a_local", {expr(NUM_TASKS_C)}, p_int64, a_input);
  buffer buf_b_local("buf_b_local", {expr(NUM_TASKS_C)}, p_int64, a_input);
  buffer buf_c_local("buf_c_local", {expr(NUM_TASKS_C)}, p_int64, a_output);

  // Map computations to buffers.
  a_in.store_in(&buf_a_in, {expr(map_a / num_ranks)});
  b_in.store_in(&buf_b_in, {expr(map_b / num_ranks)});
  comm_a.r->store_in(&buf_a_local, {expr(map_c / num_ranks)});
  comm_b.r->store_in(&buf_b_local, {expr(map_c / num_ranks)});
  //comm_a.s->set_wait_access("[J,K,NUM_RANKS]->{a_send[j,map_c,rnk_a]->buf_wait_a_send[ [([map_c/K]*J+j)/NUM_RANKS] ]}");
  //comm_a.r->set_wait_access("[NUM_RANKS]->{a_recv[j,map_c]->buf_wait_a_recv[ [map_c/NUM_RANKS] ]}");
  //comm_b.s->set_wait_access("[K,NUM_RANKS]->{b_send[j,map_c,rnk_b]->buf_wait_b_send[ [(j*K+(map_c%K))/NUM_RANKS] ]}");
  //comm_b.r->set_wait_access("[NUM_RANKS]->{b_recv[j,map_c]->buf_wait_b_recv[ [map_c/NUM_RANKS] ]}");
  std::string tsk_a_str("[(" + map_a_str + ")/" + std::to_string(num_ranks) + "]");
  std::string tsk_b_str("[(" + map_b_str + ")/" + std::to_string(num_ranks) + "]");
  std::string tsk_c_str("[map_c/" + std::to_string(num_ranks) + "]");
  comm_a.s->set_wait_access("{a_send[j,map_c,rnk_a]->buf_wait_a_send[(" + tsk_a_str + ")]}");
  comm_a.r->set_wait_access("{a_recv[j,map_c]->buf_wait_a_recv[(" + tsk_c_str + ")]}");
  comm_b.s->set_wait_access("{b_send[j,map_c,rnk_b]->buf_wait_b_send[(" + tsk_b_str + ")]}");
  comm_b.r->set_wait_access("{b_recv[j,map_c]->buf_wait_b_recv[(" + tsk_c_str + ")]}");
  a_local.store_in(&buf_a_local, {expr(map_c / num_ranks)});
  b_local.store_in(&buf_b_local, {expr(map_c / num_ranks)});
  c_local_init.store_in(&buf_c_local, {expr(map_c / num_ranks)});
  c_local.store_in(&buf_c_local, {expr(map_c / num_ranks)});

  tiramisu::codegen({&buf_a_in, &buf_b_in, &buf_wait_a_send, &buf_wait_a_recv, &buf_wait_b_send, &buf_wait_b_recv, &buf_a_local, &buf_b_local, &buf_c_local}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
  fn0->dump_halide_stmt();
}





int main(int argc, char **argv)
{
  gen("spmm", _NUM_RANKS, _I_DIM_SIZE, _J_DIM_SIZE, _K_DIM_SIZE);
  return 0;
}












////////////////////////////////////////////////////////////////////////////////
// Code that almost worked and then didn't, but may be useful later
////////////////////////////////////////////////////////////////////////////////

  /*
  // Manually form iteration domain point-wise for A block MPI send/recv.
  std::string a_iter;
  for (uint64_t j_str = 0; j_str < j_dim_size; j_str++) {
    for (uint64_t map_c_str = 0; map_c_str < i_dim_size * k_dim_size; map_c_str++) {
      uint64_t i_str = map_c_str / k_dim_size;
      uint64_t k_str = map_c_str % k_dim_size;
      uint64_t map_a_str = i_str * j_dim_size + j_str;
      
      uint64_t tsk_c_str = map_c_str / num_ranks;
      uint64_t rnk_c_str = map_c_str % num_ranks;

      uint64_t tsk_a_str = map_a_str / num_ranks;
      uint64_t rnk_a_str = map_a_str % num_ranks;

      if (!a_iter.empty()) {
        a_iter += " or ";
      }
      a_iter += "j=" + std::to_string(j_str) + " and ";
      a_iter += "tsk_c=" + std::to_string(tsk_c_str) + " and ";
      a_iter += "rnk_c=" + std::to_string(rnk_c_str) + " and ";
      a_iter += "tsk_a=" + std::to_string(tsk_a_str) + " and ";
      a_iter += "rnk_a=" + std::to_string(rnk_a_str);
    }
  }

  std::string a_iter_send("{a_send[j,tsk_c,rnk_c,tsk_a,rnk_a]: ");
  a_iter_send += a_iter;
  a_iter_send += "}";
  std::string a_iter_recv("{a_recv[j,tsk_c,rnk_c,tsk_a,rnk_a]: ");
  a_iter_recv += a_iter;
  a_iter_recv += "}";
  */

  /*
  std::string a_iter_constants("[NUM_RANKS,J,K,MAP_C_SIZE]");
  std::string a_iter_constraints("[j,map_c,tsk_c_derived,rnk_c_derived,tsk_a_derived,rnk_a_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and tsk_c_derived=[map_c/NUM_RANKS] and rnk_c_derived=map_c%NUM_RANKS and tsk_a_derived=[([map_c/K]*J+j)/NUM_RANKS] and rnk_a_derived=([map_c/K]*J+j)%NUM_RANKS");
  std::string a_iter_send(a_iter_constants + "->{a_send" + a_iter_constraints + "}");
  std::string a_iter_recv(a_iter_constants + "->{a_recv" + a_iter_constraints + "}");
  */




  /*
  std::string map_a_str("[map_c/" + std::to_string(k_dim_size) + "]*" + std::to_string(j_dim_size) + "+j");
  std::string rnk_c_derived_str("map_c%" + std::to_string(num_ranks));
  std::string rnk_a_derived_str("(" + map_a_str + ")%" + std::to_string(num_ranks));
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{a_send[j,map_c,map_a,rnk_c_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and map_a=[map_c/K]*J+j and rnk_c_derived=map_c%NUM_RANKS}
  std::string a_iter_send("[J,K,NUM_RANKS,MAP_C_SIZE]->{a_send[j,map_c,map_a,rnk_c_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and map_a=" + map_a_str + " and rnk_c_derived=" + rnk_c_derived_str + "}");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{a_recv[j,map_c,rnk_a_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and rnk_a_derived=([map_c/K]*J+j)%NUM_RANKS}
  std::string a_iter_recv("[J,K,NUM_RANKS,MAP_C_SIZE]->{a_recv[j,map_c,rnk_a_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and rnk_a_derived=" + rnk_a_derived_str + "}");
  // std::cout << "A_ITER_SEND\n";
  // std::cout << a_iter_send;
  // std::cout << "\n\n";
  // std::cout << "A_ITER_RECV\n";
  // std::cout << a_iter_recv;
  // std::cout << "\n\n";

  // Create (non-blocking, non-synchronous) MPI send/recv for A blocks.
  xfer comm_a = computation::create_xfer(
      a_iter_send,
      a_iter_recv,
      rnk_c_derived,
      rnk_a_derived,
      xfer_prop(p_int64, {MPI, BLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, BLOCK, ASYNC}),
      a_in(map_a),
      fn0);
  */




  // Manually form iteration domain point-wise for B block MPI send/recv.
  /*
  std::string b_iter;
  for (uint64_t j_str = 0; j_str < j_dim_size; j_str++) {
    for (uint64_t map_c_str = 0; map_c_str < i_dim_size * k_dim_size; map_c_str++) {
      uint64_t i_str = map_c_str / k_dim_size;
      uint64_t k_str = map_c_str % k_dim_size;
      uint64_t map_b_str = j_str * k_dim_size + k_str;
      
      uint64_t tsk_c_str = map_c_str / num_ranks;
      uint64_t rnk_c_str = map_c_str % num_ranks;

      uint64_t tsk_b_str = map_b_str / num_ranks;
      uint64_t rnk_b_str = map_b_str % num_ranks;

      if (!b_iter.empty()) {
        b_iter += " or ";
      }
      b_iter += "j=" + std::to_string(j_str) + " and ";
      b_iter += "tsk_c=" + std::to_string(tsk_c_str) + " and ";
      b_iter += "rnk_c=" + std::to_string(rnk_c_str) + " and ";
      b_iter += "tsk_b=" + std::to_string(tsk_b_str) + " and ";
      b_iter += "rnk_b=" + std::to_string(rnk_b_str);
    }
  }

  std::string b_iter_send("{b_send[j,tsk_c,rnk_c,tsk_b,rnk_b]: ");
  b_iter_send += b_iter;
  b_iter_send += "}";
  std::string b_iter_recv("{b_recv[j,tsk_c,rnk_c,tsk_b,rnk_b]: ");
  b_iter_recv += b_iter;
  b_iter_recv += "}";
  */

  /*
  std::string b_iter_constants("[NUM_RANKS,J,K,MAP_C_SIZE]");
  std::string b_iter_constraints("[j,map_c,tsk_c,rnk_c,tsk_b,rnk_b]: 0<=j<J and 0<=map_c<MAP_C_SIZE and tsk_c=[map_c/NUM_RANKS] and rnk_c=map_c%NUM_RANKS and tsk_b=[(j*K+[map_c%K])/NUM_RANKS] and rnk_b=(j*K+[map_c%K])%NUM_RANKS");
  std::string b_iter_send(b_iter_constants + "->{b_send" + b_iter_constraints + "}");
  std::string b_iter_recv(b_iter_constants + "->{b_recv" + b_iter_constraints + "}");
  */





  /*
  std::string map_b_str("j*" + std::to_string(k_dim_size) + "+(map_c%" + std::to_string(k_dim_size) + ")");
  std::string rnk_b_derived_str("(" + map_b_str + ")%" + std::to_string(num_ranks));
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{b_send[j,map_c,map_b,rnk_c_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and map_b=j*K+(map_c%K) and rnk_c_derived=map_c%NUM_RANKS}
  std::string b_iter_send("[J,K,NUM_RANKS,MAP_C_SIZE]->{b_send[j,map_c,map_b,rnk_c_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and map_b=" + map_b_str + " and rnk_c_derived=" + rnk_c_derived_str + "}");
  // [J,K,NUM_RANKS,MAP_C_SIZE]->{b_recv[j,map_c,rnk_b_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and rnk_b_derived=(j*K+(map_c%K))%NUM_RANKS}
  std::string b_iter_recv("[J,K,NUM_RANKS,MAP_C_SIZE]->{b_recv[j,map_c,rnk_b_derived]: 0<=j<J and 0<=map_c<MAP_C_SIZE and rnk_b_derived=" + rnk_b_derived_str + "}");

  // Create (non-blocking, non-synchronous) MPI send/recv for B blocks.
  xfer comm_b = computation::create_xfer(
      b_iter_send,
      b_iter_recv,
      rnk_c_derived,
      rnk_b_derived,
      xfer_prop(p_int64, {MPI, BLOCK, ASYNC}),
      xfer_prop(p_int64, {MPI, BLOCK, ASYNC}),
      b_in(map_b),
      fn0);
  */
