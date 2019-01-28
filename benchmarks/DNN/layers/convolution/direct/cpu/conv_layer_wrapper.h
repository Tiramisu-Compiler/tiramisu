#ifndef HALIDE__build___wrapper_conv_o_h
#define HALIDE__build___wrapper_conv_o_h

#ifdef __cplusplus
extern "C" {
#endif

int conv_tiramisu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
int conv_tiramisu_argv(void **args);

// Result is never null and points to constant static data
const struct halide_filter_metadata_t *conv_tiramisu_metadata();
const struct halide_filter_metadata_t *conv_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
