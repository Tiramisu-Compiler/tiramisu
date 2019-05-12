#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

#define SMALL_BARYON_DATA_SET 0
#define LARGE_BARYON_DATA_SET 1

#if SMALL_BARYON_DATA_SET

#define Nq 3
#define Nc 3
#define Ns 2
#define Nw 9
#define twoNw 81
#define Nperms 36
#define Lt 2
#define Vsrc 2
#define Vsnk 4
#define Nsrc 2
#define Nsnk 2
#define mq 1.0

#elif LARGE_BARYON_DATA_SET

#define Nq 3
#define Nc 3
#define Ns 2
#define Nw 9
#define twoNw 81
#define Nperms 36
#define Lt 2 // 1..32
#define Vsrc 16 //8, 64, 512
#define Vsnk 16 //8, 64, 512
#define Nsrc 6
#define Nsnk 6
#define mq 1.0

#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// Define these values for each new test
#define TEST_NAME_STR       "dibaryon"

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int tiramisu_generated_code(halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
	   		    halide_buffer_t *,
			    halide_buffer_t *,
	   		    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
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
