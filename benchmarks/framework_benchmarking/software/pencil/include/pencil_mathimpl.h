#ifndef PENCIL_MATHDEFS_H
#define PENCIL_MATHDEFS_H

#ifdef __CUDACC__
#define PENCIL_MATHDEF __host__ __device__ static
/* TODO: Prefer CUDA intrinisics over own definitions (min, max). */
#else
#define PENCIL_MATHDEF static inline
#endif

/* PENCIL standard library functions NOT in math.h or stdlib.h */
PENCIL_MATHDEF int32_t abs32(int32_t x) { return (x >= 0) ? x : -x; }
PENCIL_MATHDEF int64_t abs64(int64_t x) { return (x >= 0) ? x : -x; }

#ifndef __CUDACC__
PENCIL_MATHDEF int min(int x, int y) { return (x<=y) ? x : y; }
#endif
PENCIL_MATHDEF long lmin(long x, long y)  { return (x<=y) ? x : y; }
//PENCIL_MATHDEF long long llmin(long long x, long long y) { return (x<=y) ? x : y; }
PENCIL_MATHDEF int32_t min32(int32_t x, int32_t y) { return (x<=y) ? x : y; }
PENCIL_MATHDEF int64_t min64(int64_t x, int64_t y) { return (x<=y) ? x : y; }

#ifndef __CUDACC__
PENCIL_MATHDEF int max(int x, int y) { return (x>=y) ? x : y; }
#endif
PENCIL_MATHDEF long lmax(long x, long y) { return (x>=y) ? x : y; }
//PENCIL_MATHDEF long long llmax(long long x, long long y) { return (x>=y) ? x : y; }
PENCIL_MATHDEF int32_t max32(int32_t x, int32_t y) { return (x>=y) ? x : y; }
PENCIL_MATHDEF int64_t max64(int64_t x, int64_t y) { return (x>=y) ? x : y; }


PENCIL_MATHDEF int clamp(int x, int minval, int maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
PENCIL_MATHDEF long lclamp(long x, long minval, long maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
//static inline long long llclamp(long long x, long long minval, long long maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
PENCIL_MATHDEF int32_t clamp32(int32_t x, int32_t minval, int32_t maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
PENCIL_MATHDEF int64_t clamp64(int64_t x, int64_t minval, int64_t maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
PENCIL_MATHDEF double fclamp(double x, double minval, double maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }
PENCIL_MATHDEF float fclampf(float x, float minval, float maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }

PENCIL_MATHDEF double mix(double x, double y, double a) { return x + (x-y) * a; }
PENCIL_MATHDEF float mixf(float x, float y, float a) { return x + (x-y) * a; }

PENCIL_MATHDEF double atan2pi(double x, double y) { return atan2(x, y) / 3.14159265358979323846; }
PENCIL_MATHDEF float atan2pif(float x, float y) { return atan2f(x, y) / 3.14159265358979323846f; }

#endif /* PENCIL_MATHDEFS_H */
