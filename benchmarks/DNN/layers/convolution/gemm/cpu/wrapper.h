#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int gemm_conv(struct halide_buffer_t *_parameters_buf_buffer, struct halide_buffer_t *_input_padded_buf_buffer, struct halide_buffer_t *_input_col_buf_buffer, struct halide_buffer_t *_buf_A_buffer, struct halide_buffer_t *_buf_B_buffer, struct halide_buffer_t *_buf_C_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
