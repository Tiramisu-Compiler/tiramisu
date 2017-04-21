#ifndef TIRAMISU_test_17_h
#define TIRAMISU_test_17_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int test_tag_gpu_level(buffer_t *_p0_buffer);
int test_tag_gpu_level_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
