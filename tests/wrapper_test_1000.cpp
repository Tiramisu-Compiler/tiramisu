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





////////////////////////////////////////////////////////////////////////////////
// Code that was useful for debugging in the past, but isn't needed now
////////////////////////////////////////////////////////////////////////////////





/*
uint64_t encode_point_round(char name, uint32_t i, uint32_t j, uint32_t round) {
  union {
    struct {
      uint8_t name1:4;
      uint32_t round1:10;
      uint32_t i:18;
      uint8_t name2:4;
      uint32_t round2:10;
      uint32_t j:18;
    };
    uint64_t bits;
  } bit_pack;
  assert(sizeof(bit_pack) == 8);
  bit_pack.name1 = (uint8_t) name / 16;
  bit_pack.round1 = round / 1024;
  bit_pack.i = i;
  bit_pack.name2 = (uint8_t) name % 16;
  bit_pack.round2 = round % 1024;
  bit_pack.j = j;
  return bit_pack.bits;
}

void decode_point_round(uint64_t bits, char *name, uint32_t *i, uint32_t *j, uint32_t *round) {
  union {
    struct {
      uint8_t name1:4;
      uint32_t round1:10;
      uint32_t i:18;
      uint8_t name2:4;
      uint32_t round2:10;
      uint32_t j:18;
    };
    uint64_t bits;
  } bit_unpack;
  assert(sizeof(bit_unpack) == 8);
  bit_unpack.bits = bits;
  *name = (char) (bit_unpack.name1 * 16 + bit_unpack.name2);
  *i = bit_unpack.i;
  *j = bit_unpack.j;
  *round = bit_unpack.round1 * 1024 + bit_unpack.round2;
}

#define DECODE_POINT_ROUND(__bits, __name, __i, __j, __round)           \
  do {                                                                  \
    uint64_t _bits = __bits;                                            \
    char *_name = __name;                                               \
    uint32_t *_i = __i;                                                 \
    uint32_t *_j = __j;                                                 \
    uint32_t *_round = __round;                                         \
    union {                                                             \
      struct {                                                          \
        uint8_t name1:4;                                                \
        uint32_t round1:10;                                             \
        uint32_t i:18;                                                  \
        uint8_t name2:4;                                                \
        uint32_t round2:10;                                             \
        uint32_t j:18;                                                  \
      };                                                                \
      uint64_t bits;                                                    \
    } bit_unpack;                                                       \
    assert(sizeof(bit_unpack) == 8);                                    \
    bit_unpack.bits = _bits;                                            \
    *_name = (char) (bit_unpack.name1 * 16 + bit_unpack.name2);         \
    *_i = bit_unpack.i;                                                 \
    *_j = bit_unpack.j;                                                 \
    *_round = bit_unpack.round1 * 1024 + bit_unpack.round2;             \
  } while (0)
*/






/*
{
  std::cout << "RANK: " << rank << " : buf_a = [";
  for (uint64_t i = 0; i < num_tasks_a; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buf_a(i);
  }
  std::cout << "]\n";

  std::cout << "RANK: " << rank << " : buf_b = [";
  for (uint64_t i = 0; i < num_tasks_b; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buf_b(i);
  }
  std::cout << "]\n";

  std::cout << "RANK: " << rank << " : buf_a_local = [";
  for (uint64_t i = 0; i < num_tasks_c; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buf_a_local(i);
  }
  std::cout << "]\n";

  std::cout << "RANK: " << rank << " : buf_b_local = [";
  for (uint64_t i = 0; i < num_tasks_c; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buf_b_local(i);
  }
  std::cout << "]\n";

  std::cout << "RANK: " << rank << " : buf_c = [";
  for (uint64_t i = 0; i < num_tasks_c; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buf_c(i);
  }
  std::cout << "]\n";
}
*/





/*
#define make_Send(suffix, c_datatype, mpi_datatype)                     \
  void tiramisu_MPI_Send_##suffix(int count, int dest, int tag, c_datatype *data) \
  {                                                                     \
    uint64_t bits = *((uint64_t *)data);                                \
    char name = 0;                                                      \
    uint32_t i = 0, j = 0, round = 0;                                   \
    DECODE_POINT_ROUND(bits, &name, &i, &j, &round);                    \
    int src = -1;                                                       \
    MPI_Comm_rank(MPI_COMM_WORLD, &src);                                \
                                                                        \
    printf("ROUND %02" PRIu32 ": RANK %02d: SEND(%02d): %c(%02" PRIu32 ", %02" PRIu32 ")" , round, src, dest, name, i, j); \
    if (tag) {                                                          \
      printf(": TAG %02d", tag);                                        \
    }                                                                   \
    printf("\n");                                                       \
                                                                        \
    check_MPI_error(MPI_Send(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD)); \
  }

#define make_Recv(suffix, c_datatype, mpi_datatype)                     \
  void tiramisu_MPI_Recv_##suffix(int count, int source, int tag, c_datatype *store_in) \
  {                                                                     \
    uint64_t bits = *((uint64_t *)store_in);                            \
    char name = 0;                                                      \
    uint32_t i = 0, j = 0, round = 0;                                   \
    DECODE_POINT_ROUND(bits, &name, &i, &j, &round);                    \
    int dst = -1;                                                       \
    MPI_Comm_rank(MPI_COMM_WORLD, &dst);                                \
                                                                        \
    MPI_Status status;                                                  \
    check_MPI_error(MPI_Recv(store_in, count, mpi_datatype, source, tag, MPI_COMM_WORLD, &status)); \
                                                                        \
    uint64_t bits_recv = *((uint64_t *)store_in);                       \
    char name_recv = 0;                                                 \
    uint32_t i_recv = 0, j_recv = 0, round_recv = 0;                    \
    DECODE_POINT_ROUND(bits_recv, &name_recv, &i_recv, &j_recv, &round_recv); \
                                                                        \
    printf("ROUND %02" PRIu32 ": RANK %02d: RECV(%02d): %c(%02" PRIu32 ", %02" PRIu32 ") = %c(%02" PRIu32 ", %02" PRIu32 ")" , round, dst, source, name, i, j, name_recv, i_recv, j_recv); \
    if (round != round_recv) {                                          \
      printf(" (ROUND %02" PRIu32 ")", round_recv);                     \
    }                                                                   \
    if (tag) {                                                          \
      printf(": TAG %02d", tag);                                        \
    }                                                                   \
    printf("\n");                                                       \
  }
*/





/*
#define make_Isend(suffix, c_datatype, mpi_datatype)                    \
void tiramisu_MPI_Isend_##suffix(int count, int dest, int tag, c_datatype *data, long *reqs) \
{                                                                       \
    int src = -1;                                                       \
    MPI_Comm_rank(MPI_COMM_WORLD, &src);                                \
                                                                        \
    uint64_t bits = *((uint64_t *)data);                                \
    char name = 0;                                                      \
    uint32_t i = 0, j = 0, round = 0;                                   \
    DECODE_POINT_ROUND(bits, &name, &i, &j, &round);                    \
                                                                        \
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request)); \
    check_MPI_error(MPI_Isend(data, count, mpi_datatype, dest, tag, MPI_COMM_WORLD, ((MPI_Request**)reqs)[0])); \
                                                                        \
    printf("RANK %02d: WAIT(%016" PRIxPTR "): ROUND %02" PRIu32 ": SEND(%02d, %02d): %c(%02" PRIu32 ", %02" PRIu32 ")" , \
           src, (uintptr_t)(((MPI_Request**)reqs)[0]), round, dest, tag, name, i, j); \
    printf("\n");                                                       \
}

#define make_Irecv(suffix, c_datatype, mpi_datatype)                    \
void tiramisu_MPI_Irecv_##suffix(int count, int source, int tag,        \
                                 c_datatype *store_in, long *reqs)      \
{                                                                       \
    int dst = -1;                                                       \
    MPI_Comm_rank(MPI_COMM_WORLD, &dst);                                \
                                                                        \
    uint64_t bits = *((uint64_t *)store_in);                            \
    char name = 0;                                                      \
    uint32_t i = 0, j = 0, round = 0;                                   \
    DECODE_POINT_ROUND(bits, &name, &i, &j, &round);                    \
                                                                        \
    ((MPI_Request**)reqs)[0] = (MPI_Request*)malloc(sizeof(MPI_Request)); \
    check_MPI_error(MPI_Irecv(store_in, count, mpi_datatype, source, tag, MPI_COMM_WORLD, \
                              ((MPI_Request**)reqs)[0]));               \
                                                                        \
    uint64_t bits_recv = *((uint64_t *)store_in);                       \
    char name_recv = 0;                                                 \
    uint32_t i_recv = 0, j_recv = 0, round_recv = 0;                    \
    DECODE_POINT_ROUND(bits_recv, &name_recv, &i_recv, &j_recv, &round_recv); \
                                                                        \
    printf("RANK %02d: WAIT(%016" PRIxPTR "): ROUND %02" PRIu32 ": RECV(%02d, %02d): %c(%02" PRIu32 ", %02" PRIu32 ") = %c(%02" PRIu32 ", %02" PRIu32 ")" , \
           dst, (uintptr_t)(((MPI_Request**)reqs)[0]), round, source, tag, name, i, j, name_recv, i_recv, j_recv); \
    if (round != round_recv) {                                          \
      printf(" (ROUND %02" PRIu32 ")", round_recv);                     \
    }                                                                   \
    printf("\n");                                                       \
}

void tiramisu_MPI_Wait(void *request)
{
    int rnk = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

    MPI_Status status;
    check_MPI_error(MPI_Wait((MPI_Request*)request, &status));
    free(request);

    printf("RANK %02d: WAIT(%016" PRIxPTR "): COMPLETION" , rnk, (uintptr_t) request);
    printf("\n");
}
*/





/*
  {
    uint64_t tsk_a = 0;
    for (uint64_t map_a = rank; map_a < i_dim_size * j_dim_size; map_a += num_ranks, tsk_a++) {
      uint64_t i = map_a / j_dim_size;
      uint64_t j = map_a % j_dim_size;
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_a(tsk_a, round) = (int64_t) encode_point_round('A', (uint32_t) i, (uint32_t) j, (uint32_t) round);

        uint64_t bits_test = (uint64_t) buf_a(tsk_a, round);
        char name_test;
        uint32_t i_test, j_test, round_test;
        DECODE_POINT_ROUND(bits_test, &name_test, &i_test, &j_test, &round_test);
        if (name_test != 'A' ||
            i_test != i ||
            j_test != j ||
            round_test != round) {
          std::cout << "DECODE IS WRONG";
          return -1;
        }
      }
    }
    for (; tsk_a < num_tasks_a; tsk_a++) {
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_a(tsk_a, round) = 0;
      }
    }
  }

  {
    uint64_t tsk_b = 0;
    for (uint64_t map_b = rank; map_b < j_dim_size * k_dim_size; map_b += num_ranks, tsk_b++) {
      uint64_t j = map_b / k_dim_size;
      uint64_t k = map_b % k_dim_size;
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_b(tsk_b, round) = (int64_t) encode_point_round('B', (uint32_t) j, (uint32_t) k, (uint32_t) round);

        uint64_t bits_test = (uint64_t) buf_b(tsk_b, round);
        char name_test;
        uint32_t j_test, k_test, round_test;
        DECODE_POINT_ROUND(bits_test, &name_test, &j_test, &k_test, &round_test);
        if (name_test != 'B' ||
            j_test != j ||
            k_test != k ||
            round_test != round) {
          std::cout << "DECODE IS WRONG";
          return -1;
        }
      }
    }
    for (; tsk_b < num_tasks_b; tsk_b++) {
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_b(tsk_b, round) = 0;
      }
    }
  }

  {
    uint64_t tsk_c = 0;
    for (uint64_t map_c = rank; map_c < i_dim_size * k_dim_size; map_c += num_ranks, tsk_c++) {
      uint64_t i = map_c / k_dim_size;
      uint64_t k = map_c % k_dim_size;
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_a_local(tsk_c, round) = (int64_t) encode_point_round('a', (uint32_t) i, (uint32_t) k, (uint32_t) round);
      }
    }
    for (; tsk_c < num_tasks_c; tsk_c++) {
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_a_local(tsk_c, round) = 0;
      }
    }
  }

  {
    uint64_t tsk_c = 0;
    for (uint64_t map_c = rank; map_c < i_dim_size * k_dim_size; map_c += num_ranks, tsk_c++) {
      uint64_t i = map_c / k_dim_size;
      uint64_t k = map_c % k_dim_size;
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_b_local(tsk_c, round) = (int64_t) encode_point_round('b', (uint32_t) i, (uint32_t) k, (uint32_t) round);
      }
    }
    for (; tsk_c < num_tasks_c; tsk_c++) {
      for (uint64_t round = 0; round < j_dim_size; round++) {
        buf_b_local(tsk_c, round) = 0;
      }
    }
  }
*/





/*
  {
    uint64_t tsk_c = 0;
    for (uint64_t map_c = rank; map_c < i_dim_size * k_dim_size; map_c += num_ranks, tsk_c++) {
      uint64_t i = map_c / k_dim_size;
      uint64_t k = map_c % k_dim_size;
      int64_t sum = 0;
      for (uint64_t j = 0; j < j_dim_size; j++) {
        sum += (int64_t) encode_point_round('A', (uint32_t) i, (uint32_t) j, (uint32_t) j) * (int64_t) encode_point_round('B', (uint32_t) j, (uint32_t) k, (uint32_t) j);
      }
      buf_c_ref(tsk_c) = sum;
    }
    for (; tsk_c < num_tasks_c; tsk_c++) {
      buf_c_ref(tsk_c) = 0;
    }
  }
 */
