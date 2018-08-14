#ifndef PENCIL_KERNEL_C_H
#define PENCIL_KERNEL_C_H

/* PENCIL functionality that require an #include in C */
#include <stdbool.h>/* bool */
#include <stdint.h> /* int32_t, int64_t */
#include <stdlib.h> /* abs, labs, llabs */
#include <math.h>   /* sqrt, sqrtf, ... */
#include <assert.h> /* assert */

#include "pencil_mathimpl.h"

#ifndef __has_extension
#define __has_extension(...) 0
#endif

#if (__GNUC__) > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || __has_extension(c_static_assert)
_Static_assert(sizeof(int)==4, "PENCIL: int must be 32 bit to match OpenCL's");
_Static_assert(sizeof(long)==8, "PENCIL: long must be 64 bit to match OpenCL's");
#endif

#endif /* PENCIL_KERNEL_C_H */
