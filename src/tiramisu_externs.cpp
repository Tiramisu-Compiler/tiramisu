#include "tiramisu/externs.h"
#ifdef WITH_MPI
#include <mpi.h>
#endif

extern "C" {

#ifdef USE_HALIDE
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

uint8_t *tiramisu_address_of_uint8(halide_buffer_t *buffer, unsigned long index) {
    return &(((uint8_t*)(buffer->host))[index]);
}

uint16_t *tiramisu_address_of_uint16(halide_buffer_t *buffer, unsigned long index) {
    return &(((uint16_t*)(buffer->host))[index]);
}

uint32_t *tiramisu_address_of_uint32(halide_buffer_t *buffer, unsigned long index) {
    return &(((uint32_t*)(buffer->host))[index]);
}

uint64_t *tiramisu_address_of_uint64(halide_buffer_t *buffer, unsigned long index) {
    return &(((uint64_t*)(buffer->host))[index]);
}

float *tiramisu_address_of_float32(halide_buffer_t *buffer, unsigned long index) {
    return &(((float*)(buffer->host))[index]);
}

double *tiramisu_address_of_float64(halide_buffer_t *buffer, unsigned long index) {
    return &(((double*)(buffer->host))[index]);
}
#endif

#ifdef WITH_MPI
#ifdef USE_HALIDE
void *tiramisu_address_of_wait(halide_buffer_t *buffer, unsigned long index) {
  return &(((MPI_Request*)(buffer->host))[index]);
}
#endif
#endif

}
