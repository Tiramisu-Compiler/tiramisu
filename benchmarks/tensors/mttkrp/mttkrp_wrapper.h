#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

// Define these values for each new test
#define TEST_NAME_STR       "mttkrp"

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

#define SIZE 256

int tiramisu_generated_code(halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *);

int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
