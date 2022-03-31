import tiramisu as tm

tm.init("tut_02")
NN = 10
MM = 20

N = tm.constant("N", NN)
M = tm.constant("M", MM)

A = tm.input("A", ["i", "j"], {N, M}, tm.primitive_t.p_uint8)

i, j = tm.var("i", 0, N), tm.var("j", 0, M)

# Casting issue?
output = tm.computation("output", [i,j], A(i, j) + cast(p_uint8, i) + (uint8_t)4)


i0, i1, j0, j1 = tm.var("i0"), tm.var("i1"), tm.var("j0"), tm.var("j1")

output.tile(i, j, 2, 2, i0, j0, i1, j1)
output.parallelize(i0)

b_A = tm.buffer("b_A", [NN, MM], tm.primitive_t.p_uint8, tm.argument_t.a_input)
b_output = tm.buffer("b_output", [NN, MM], tm.primitive_t.p_uint8, tm.argument_t.a_output)

A.store_in(b_A)
output.store_in(b_output)

tm.codegen([b_A, b_output], "generated_fct_developers_tutorial_02.o")
