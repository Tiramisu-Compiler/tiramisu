#ifndef PENCIL_MATHDECLS_H
#define PENCIL_MATHDECLS_H

PENCIL_MATHDECL int32_t abs32(int32_t x);
PENCIL_MATHDECL int64_t abs64(int64_t x);

PENCIL_MATHDECL int min(int x, int y);
PENCIL_MATHDECL long lmin(long x, long y);
//PENCIL_MATHDECL long long llmin(long long x, long long y);
PENCIL_MATHDECL int32_t min32(int32_t x, int32_t y);
PENCIL_MATHDECL int64_t min64(int64_t x, int64_t y);

PENCIL_MATHDECL int max(int x, int y);
PENCIL_MATHDECL long lmax(long x, long y);
//PENCIL_MATHDECL long long llmax(long long x, long long y);
PENCIL_MATHDECL int32_t max32(int32_t x, int32_t y);
PENCIL_MATHDECL int64_t max64(int64_t x, int64_t y);

PENCIL_MATHDECL int clamp(int x, int minval, int maxval);
PENCIL_MATHDECL long lclamp(long x, long minval, long maxval);
//PENCIL_MATHDECL long long llclamp(long long x, long long minval, long long maxval);
PENCIL_MATHDECL int32_t clamp32(int32_t x, int32_t minval, int32_t maxval);
PENCIL_MATHDECL int64_t clamp64(int64_t x, int64_t minval, int64_t maxval);
PENCIL_MATHDECL double fclamp(double x, double minval, double maxval);
PENCIL_MATHDECL float fclampf(float x, float minval, float maxval);

PENCIL_MATHDECL double mix(double x, double y, double a);
PENCIL_MATHDECL float mixf(float x, float y, float a);

PENCIL_MATHDECL double atan2pi(double x, double y);
PENCIL_MATHDECL float atan2pif(float x, float y);

#endif /* PENCIL_MATHDECLS_H */
