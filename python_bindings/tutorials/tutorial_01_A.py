import tiramisu as tm

tm.init("test")

p0 = tm.constant("N", tm.expr(20))
i = tm.var("i", tm.expr(0), p0)
j = tm.var("j", 0, p0)

S0 = tm.computation("S0", [i], tm.expr(3) + tm.expr(4))

S0.parallelize(i)

buf0 = tm.buffer("buf0", [10], tm.primitive_t.p_uint8, tm.argument_t.a_output)

S0.store_in(buf0)

tm.pycodegen([buf0], "generated_fct_developers_tutorial_01.o", False)
import test

print(test.test)
import numpy as np

temp = np.zeros((20), dtype=np.uint8)
test.test(temp)
print(temp)
