#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "baryon"

// Data size
#if TIRAMISU_XLARGE
#define SIZE (1024*1024*128)
#elif TIRAMISU_LARGE
#define SIZE (1024*1024)
#elif TIRAMISU_MEDIUM
#define SIZE (1024)
#elif TIRAMISU_SMALL
#define SIZE (128)
#endif

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

// BARYON_N is used for loop iterators.
#define BARYON_N 16
#define BX 16
#define BY 16
#define BZ 16
#define BT 16
#define BK 16

// BARYON_P1 is used for the size of first dimension
// of array and possible value of the parameters used
// in that first dimension.
#define BARYON_P1 3
// BARYON_P is used for the size of the other array
// dimensions that are not of size BARYON_N
#define BARYON_P 1

int tiramisu_generated_code(halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *);

int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
