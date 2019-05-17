#include "wrapper_test_1000.h"
#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <Halide.h>
#include <iostream>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>





uint64_t encode_point(char name, uint32_t i, uint32_t j) {
  union {
    struct {
      uint8_t name1:4;
      uint32_t i:28;
      uint8_t name2:4;
      uint32_t j:28;
    };
    uint64_t bits;
  } bit_pack;
  assert(sizeof(bit_pack) == 8);
  bit_pack.name1 = name / 16;
  bit_pack.i = i;
  bit_pack.name2 = name % 16;
  bit_pack.j = j;
  return bit_pack.bits;
}





int main(int argc, char **argv) {
#ifdef WITH_MPI
  int rank = tiramisu_MPI_init();

  uint64_t num_ranks = _NUM_RANKS;
  uint64_t i_dim_size = _I_DIM_SIZE;
  uint64_t j_dim_size = _J_DIM_SIZE;
  uint64_t k_dim_size = _K_DIM_SIZE;
  uint64_t comm_shift = _COMM_SHIFT;

  uint64_t num_tasks_c = ((i_dim_size * k_dim_size - 1) / num_ranks) + 1;  
  uint64_t num_tasks_a = ((i_dim_size * j_dim_size - 1) / num_ranks) + 1;
  uint64_t num_tasks_b = ((j_dim_size * k_dim_size - 1) / num_ranks) + 1;
  
  Halide::Buffer<int64_t> buf_a(num_tasks_a, "buf_a");
  Halide::Buffer<int64_t> buf_b(num_tasks_b, "buf_b");
  Halide::Buffer<void *>  buf_wait_a_send(num_tasks_a, k_dim_size, comm_shift + 1, "buf_wait_a_send");
  Halide::Buffer<void *>  buf_wait_a_recv(num_tasks_c, comm_shift + 1, "buf_wait_a_recv");
  Halide::Buffer<void *>  buf_wait_b_send(num_tasks_b, i_dim_size, comm_shift + 1, "buf_wait_b_send");
  Halide::Buffer<void *>  buf_wait_b_recv(num_tasks_c, comm_shift + 1, "buf_wait_b_recv");
  Halide::Buffer<int64_t> buf_a_local(num_tasks_c, comm_shift + 1, "buf_a_local");
  Halide::Buffer<int64_t> buf_b_local(num_tasks_c, comm_shift + 1, "buf_b_local");
  Halide::Buffer<int64_t> buf_c(num_tasks_c, "buf_c");

  init_buffer(buf_a_local, (int64_t) 0); 
  init_buffer(buf_b_local, (int64_t) 0);
  init_buffer(buf_c, (int64_t) 0);

  {
    uint64_t tsk_a = 0;
    for (uint64_t map_a = rank; map_a < i_dim_size * j_dim_size; map_a += num_ranks, tsk_a++) {
      uint64_t i = map_a / j_dim_size;
      uint64_t j = map_a % j_dim_size;
      buf_a(tsk_a) = (int64_t) encode_point('A', (uint32_t) i, (uint32_t) j);
    }
    for (; tsk_a < num_tasks_a; tsk_a++) {
      buf_a(tsk_a) = 0;
    }
  }

  {
    uint64_t tsk_b = 0;
    for (uint64_t map_b = rank; map_b < j_dim_size * k_dim_size; map_b += num_ranks, tsk_b++) {
      uint64_t j = map_b / k_dim_size;
      uint64_t k = map_b % k_dim_size;
      buf_b(tsk_b) = (int64_t) encode_point('B', (uint32_t) j, (uint32_t) k);
    }
    for (; tsk_b < num_tasks_b; tsk_b++) {
      buf_b(tsk_b) = 0;
    }
  }

  spmm(buf_a.raw_buffer(), buf_b.raw_buffer(), buf_wait_a_send.raw_buffer(), buf_wait_a_recv.raw_buffer(), buf_wait_b_send.raw_buffer(), buf_wait_b_recv.raw_buffer(), buf_a_local.raw_buffer(), buf_b_local.raw_buffer(), buf_c.raw_buffer());

  MPI_Barrier(MPI_COMM_WORLD);

  Halide::Buffer<int64_t> buf_c_ref(num_tasks_c, "buf_c_ref");
  {
    uint64_t tsk_c = 0;
    for (uint64_t map_c = rank; map_c < i_dim_size * k_dim_size; map_c += num_ranks, tsk_c++) {
      uint64_t i = map_c / k_dim_size;
      uint64_t k = map_c % k_dim_size;
      int64_t sum = 0;
      for (uint64_t j = 0; j < j_dim_size; j++) {
        sum += (int64_t) encode_point('A', (uint32_t) i, (uint32_t) j) * 
            (int64_t) encode_point('B', (uint32_t) j, (uint32_t) k);
      }
      buf_c_ref(tsk_c) = sum;
    }
    for (; tsk_c < num_tasks_c; tsk_c++) {
      buf_c_ref(tsk_c) = 0;
    }
  }

  compare_buffers(TEST_NAME_STR, buf_c, buf_c_ref);
  
  tiramisu_MPI_cleanup();
#endif
    return 0;
}
