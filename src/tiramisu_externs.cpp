#include "tiramisu/tiramisu_externs.h"

extern "C" {

float *tiramisu_address_of_float32(halide_buffer_t *buffer, unsigned long index) {
    return &(((float*)(buffer->host))[index]);
}

double *tiramisu_address_of_float64(halide_buffer_t *buffer, unsigned long index) {
    return &(((double*)(buffer->host))[index]);
}

}