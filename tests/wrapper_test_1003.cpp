#include "wrapper_test_1003.h"
#include <tiramisu/utils.h>
#include <Halide.h>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstdlib>



#ifdef __cplusplus
extern "C" {
#endif

uint64_t encode_point(int32_t i, int32_t j) {
  union {
    struct {
      int32_t i:32;
      int32_t j:32;
    };
    uint64_t bits;
  } bit_pack;
  assert(sizeof(bit_pack) == 8);
  bit_pack.i = i;
  bit_pack.j = j;
  return bit_pack.bits;
}

int32_t init_zero(obj_mat *m, int32_t dim0, int32_t dim1) {
  m->dim0 = dim0;
  m->dim1 = dim1;
  memset(m->data, 0, dim0 * dim1 * sizeof(uint64_t));
  return 0;
}

int32_t init_encode(obj_mat *m, int32_t dim0, int32_t dim1) {
  m->dim0 = dim0;
  m->dim1 = dim1;
  for (int32_t i = 0; i < dim0; i++) {
    for (int32_t j = 0; j < dim1; j++) {
      m->data[i * dim1 + j] = encode_point(i, j);
    }
  }
  return 0;
}

int32_t obj_init_a(halide_buffer_t *buf, int32_t dim0, int32_t dim1) {
  obj_mat *m = (obj_mat *)buf->host;
  return init_encode(m, dim0, dim1);
}

int32_t obj_init_b(halide_buffer_t *buf, int32_t dim0, int32_t dim1) {
  obj_mat *m = (obj_mat *)buf->host;
  return init_encode(m, dim0, dim1);
}

int32_t obj_init_c(halide_buffer_t *buf, int32_t dim0, int32_t dim1) {
  obj_mat *m = (obj_mat *)buf->host;
  return init_zero(m, dim0, dim1);
}

int32_t obj_matmul(halide_buffer_t *buf_c, int32_t c_i, int32_t c_k,
                   halide_buffer_t *buf_a, int32_t a_i, int32_t a_j,
                   halide_buffer_t *buf_b, int32_t b_j, int32_t b_k) {
  obj_mat *c = (obj_mat *)buf_c->host;
  obj_mat *a = (obj_mat *)buf_a->host;
  obj_mat *b = (obj_mat *)buf_b->host;
  c->data[c_i * c->dim1 + c_k] += a->data[a_i * a->dim1 + a_j] * b->data[b_j * b->dim1 + b_k];
  return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif



int main(int argc, char **argv) {
  uint64_t i_dim_size = _I_DIM_SIZE;
  uint64_t j_dim_size = _J_DIM_SIZE;
  uint64_t k_dim_size = _K_DIM_SIZE;

  Halide::Buffer<uint8_t> buf_a(sizeof(obj_mat) + i_dim_size * j_dim_size * sizeof(uint64_t), "buf_a");
  Halide::Buffer<uint8_t> buf_b(sizeof(obj_mat) + j_dim_size * k_dim_size * sizeof(uint64_t), "buf_b");
  Halide::Buffer<uint8_t> buf_c(sizeof(obj_mat) + i_dim_size * k_dim_size * sizeof(uint64_t), "buf_c");
  Halide::Buffer<int32_t> buf_dummy(1, "buf_dummy");

  init_buffer(buf_c, (uint8_t) 0xFF);
  init_buffer(buf_c, (uint8_t) 0xFF);
  init_buffer(buf_c, (uint8_t) 0xFF);

  mm_obj(buf_a.raw_buffer(), buf_b.raw_buffer(), buf_c.raw_buffer(), buf_dummy.raw_buffer());

  Halide::Buffer<uint8_t> buf_c_ref(sizeof(obj_mat) + i_dim_size * k_dim_size * sizeof(uint64_t), "buf_c_ref");
  init_buffer(buf_c_ref, (uint8_t) 0);
  {
    obj_mat *c_ref = (obj_mat *)buf_c_ref.get()->data();

    c_ref->dim0 = i_dim_size;
    c_ref->dim1 = k_dim_size;
    for (int32_t i = 0; i < i_dim_size; i++) {
      for (int32_t k = 0; k < k_dim_size; k++) {
        for (int32_t j = 0; j < j_dim_size; j++) {
          c_ref->data[i * k_dim_size + k] += encode_point(i, j) * encode_point(j, k);
        }
      }
    }
  }
  compare_buffers(TEST_NAME_STR, buf_c, buf_c_ref);

  return 0;
}
