#ifndef _H_TIRAMISU_MACROS_
#define _H_TIRAMISU_MACROS_

#define cast(TYPE, EXPRESSION) (tiramisu::expr(tiramisu::o_cast, TYPE, EXPRESSION))
#define floor(EXPRESSION) (tiramisu::expr(o_floor, EXPRESSION))
#define clamp(EXPRESSION, MIN_VAL, MAX_VAL) (tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, EXPRESSION, MAX_VAL), MIN_VAL))

#endif
