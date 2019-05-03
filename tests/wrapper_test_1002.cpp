#include "wrapper_test_1002.h"
#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <Halide.h>



int main(int argc, char **argv) {
#ifdef WITH_MPI
  int rank = tiramisu_MPI_init();

  int num_ranks = _NUM_RANKS;

  Halide::Buffer<int64_t> buf_input(num_ranks, "buf_input");
  Halide::Buffer<int64_t> buf_output(num_ranks, "buf_output");
  Halide::Buffer<void *>  buf_wait_send(num_ranks, "buf_wait_send");
  Halide::Buffer<void *>  buf_wait_recv(num_ranks, "buf_wait_recv");

  init_buffer(buf_output, (int64_t) 0);
  for (int r = 0; r < num_ranks; r++) {
    buf_input(r) = rank*r;
  }

  split_isend_bug(buf_input.raw_buffer(), buf_output.raw_buffer(), buf_wait_send.raw_buffer(), buf_wait_recv.raw_buffer());

  MPI_Barrier(MPI_COMM_WORLD);

  Halide::Buffer<int64_t> buf_output_ref(num_ranks, "buf_output_ref");
  {
    int j = rank;
    for (int i = 0; i < num_ranks; i++) {
      int input_idx = (i*j + i/(j+1) + j/(i+1)) % num_ranks;
      int input_val = i*input_idx;
      int output_idx = (2*j+i) % num_ranks;
      buf_output_ref(output_idx) = input_val;
    }
  }

  compare_buffers(TEST_NAME_STR, buf_output, buf_output_ref);
  
  tiramisu_MPI_cleanup();
#endif
    return 0;
}
