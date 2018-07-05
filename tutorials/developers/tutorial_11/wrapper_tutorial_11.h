#ifndef TIRAMISU_WRAPPER_TUTORIAL_11_H
#define TIRAMISU_WRAPPER_TUTORIAL_11_H

#define _ROWS 1000
#define _COLS 100
// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int cvtcolor(halide_buffer_t *, halide_buffer_t *);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_TUTORIAL_11_H
