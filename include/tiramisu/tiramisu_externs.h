#ifndef TIRAMISU_EXTERNS_H
#define TIRAMISU_EXTERNS_H

#include "Halide.h"

extern "C" {

float *tiramisu_address_of_float32(halide_buffer_t *buffer, unsigned long index);

double *tiramisu_address_of_float64(halide_buffer_t *buffer, unsigned long index);

}

#endif //TIRAMISU_EXTERNS_H