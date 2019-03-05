#ifndef HALIDE__build___blurxyautodist_o_h
#define HALIDE__build___blurxyautodist_o_h

#include <tiramisu/utils.h>

#define _ROWS 100000
#define _COLS 10000

#ifdef __cplusplus
extern "C" {
#endif

int blurxyautodist_tiramisu(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);
int blurxyautodist_ref(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_TUTORIAL_12_H
