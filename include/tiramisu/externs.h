#ifndef TIRAMISU_EXTERNS_H
#define TIRAMISU_EXTERNS_H

#ifdef USE_HALIDE
    #include "Halide.h"
#endif

extern "C" {

#ifdef USE_HALIDE
int8_t *tiramisu_address_of_int8(halide_buffer_t *buffer, unsigned long index);

int16_t *tiramisu_address_of_int16(halide_buffer_t *buffer, unsigned long index);

int32_t *tiramisu_address_of_int32(halide_buffer_t *buffer, unsigned long index);

int64_t *tiramisu_address_of_int64(halide_buffer_t *buffer, unsigned long index);

uint8_t *tiramisu_address_of_uint8(halide_buffer_t *buffer, unsigned long index);

uint16_t *tiramisu_address_of_uint16(halide_buffer_t *buffer, unsigned long index);

uint32_t *tiramisu_address_of_uint32(halide_buffer_t *buffer, unsigned long index);

uint64_t *tiramisu_address_of_uint64(halide_buffer_t *buffer, unsigned long index);

float *tiramisu_address_of_float32(halide_buffer_t *buffer, unsigned long index);

double *tiramisu_address_of_float64(halide_buffer_t *buffer, unsigned long index);

#ifdef WITH_MPI
void *tiramisu_address_of_wait(halide_buffer_t *buffer, unsigned long index);
#endif
#endif

}

#endif //TIRAMISU_EXTERNS_H
