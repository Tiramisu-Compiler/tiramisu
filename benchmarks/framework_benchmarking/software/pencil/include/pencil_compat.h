#ifndef PENCIL_COMPAT_H
#define PENCIL_COMPAT_H

#if 0
/* This file is also included by _host.c which is already preprocessed. That it,
 * header guards #defines disappeared such that we cannot #include headers
 * anymore that maybe already have been included. */
#include <assert.h> /* assert */
#include <stdbool.h>/* bool */
#include <stdlib.h> /* abs, labs, llabs */
#include <math.h>   /* sqrt, sqrtf, ... */
#endif

/* PENCIL builtin functions */
#define __pencil_kill(...)
#define __pencil_use(...)
#define __pencil_def(...)
#define __pencil_maybe() 1
#define __pencil_assume(...)
#define __pencil_assert(...) assert(__VA_ARGS__)

/* Avoid gcc warning for __attribute__(((pencil_access(func_summary))) */
#define pencil_access(X) unused


/* Non-standard */
#define PENCIL_ACCESS(X)
#define PENCIL_BEGIN_SCOP
#define PENCIL_END_SCOP
#define PENCIL_INDEPENDENT(X)
#define pencil_array static const restrict
#define PENCIL_ARRAY static const restrict
#define PENCIL_SUMMARY_FUNC static __attribute__((unused))

/* Additional PENCIL types not in C99 */
/* half */
#if __ARM_FP16_ARGS
  /* use __fp16 only if usable as arguments (some older ARM targets only supports it in structs or arrays) */
  #define half __fp16
#else /* __ARM_FP16_ARGS */
  /* 16-bit floating-point is not supported: fallback to float */
  #define half float
#endif /* __ARM_FP16_ARGS */

#endif /* PENCIL_COMPAT_H */
