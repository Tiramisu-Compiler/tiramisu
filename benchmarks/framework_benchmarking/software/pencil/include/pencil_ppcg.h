#ifndef _PENCIL_PPCG_H
#define _PENCIL_PPCG_H

/* TODO: Replace by a PPCG builtin. */
#define __pencil_def(X) ((X)=0)
#define __pencil_use(X) ((void)(X))
#define __pencil_maybe() (0.1f != strtod("NaN", NULL))

/* TODO: Make ppcg skip summary functions */
#define PENCIL_SUMMARY_FUNC
#define PENCIL_ACCESS(...) __attribute__((pencil_access(__VA_ARGS__)))

#define PENCIL_MATHDECL

#endif /* _PENCIL_PPCG_H */
