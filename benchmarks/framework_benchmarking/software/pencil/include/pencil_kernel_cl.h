#ifndef PENCIL_OPENCL_H
#define PENCIL_OPENCL_H

/* Do not #include other files here; ppcg just pastes this file literally into the target file. */

/* These sizes are fixed by the OpenCL specification. */
typedef int int32_t;
typedef long int64_t;

int32_t abs32(int32_t x) { return abs(x); }
int64_t abs64(int64_t x) { return abs(x); }

int32_t min32(int32_t x, int32_t y) { return min(x, y); }
int64_t min64(int64_t x, int64_t y) { return min(x, y); }

int32_t max32(int32_t x, int32_t y) { return max(x, y); }
int64_t max64(int64_t x, int64_t y) { return max(x, y); }

int32_t clamp32(int32_t x, int32_t minvalue, int32_t maxvalue) { return clamp(x, minvalue, maxvalue); }
int64_t clamp64(int64_t x, int64_t minvalue, int64_t maxvalue) { return clamp(x, minvalue, maxvalue); }


#ifdef cl_khr_fp64
#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Here be math functions with double precision. */

#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#endif
#endif

#endif /* ifndef PENCIL_OPENCL_H */
