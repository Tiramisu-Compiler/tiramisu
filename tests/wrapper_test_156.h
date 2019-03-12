#ifndef TIRAMISU_WRAPPER_TEST_156_H
#define TIRAMISU_WRAPPER_TEST_156_H

#define TEST_NAME_STR       "Distributed BoxBlur with the new API"
#define TEST_NUMBER_STR     "156"

//data size
#define _ROWS 100
#define _COLS 10

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int boxblur(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);
int boxblur_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
}
#endif


#endif
