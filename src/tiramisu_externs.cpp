#include "tiramisu/tiramisu_externs.h"

extern "C" {

int8_t *tiramisu_address_of_int8(halide_buffer_t *buffer, unsigned long index) {
    return &(((int8_t*)(buffer->host))[index]);
}

int16_t *tiramisu_address_of_int16(halide_buffer_t *buffer, unsigned long index) {
    return &(((int16_t*)(buffer->host))[index]);
}

int32_t *tiramisu_address_of_int32(halide_buffer_t *buffer, unsigned long index) {
    return &(((int32_t*)(buffer->host))[index]);
}

int64_t *tiramisu_address_of_int64(halide_buffer_t *buffer, unsigned long index) {
    return &(((int64_t*)(buffer->host))[index]);
}

float *tiramisu_address_of_float32(halide_buffer_t *buffer, unsigned long index) {
    return &(((float*)(buffer->host))[index]);
}

double *tiramisu_address_of_float64(halide_buffer_t *buffer, unsigned long index) {
    return &(((double*)(buffer->host))[index]);
}

}