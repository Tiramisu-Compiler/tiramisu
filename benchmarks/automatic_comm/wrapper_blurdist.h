#ifndef HALIDE__build___blurdist_o_h
#define HALIDE__build___blurdist_o_h

#include <tiramisu/utils.h>

#define _ROWS 1000
#define _COLS 1000

#ifdef __cplusplus
extern "C" {
#endif

int blurdist_tiramisu(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);
int blurdist_ref(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}
#endif

#endif
