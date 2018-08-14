#ifndef PENCIL_CPP_H
#define PENCIL_CPP_H

/* These are mandatory #includes that have to be added before proprocessing (otherwise the #include guard doesn't work). */
/* In contrast to those in pencil.h, which is optional to the user. */

#ifndef PENCIL_NOFORCEDINCLUDES
/* int32_t,int64_t must be declared for math builtins */
#include <stdint.h> /* int32_t, int64_t */
#include <math.h>   /* For pencil_mathimpl.h */
#endif

#endif /* PENCIL_CPP_H */
