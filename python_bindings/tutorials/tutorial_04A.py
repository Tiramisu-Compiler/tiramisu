'''
    This tutorial shows how to write a simple matrix multiplication (C = A * B)
    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = 0;
            for k = 0 .. N
                C[i,j] = C[i,j] + A[i,k] * B[k,j];
     
     To run this tutorial
     
     cd build/
     make run_developers_tutorial_04A
'''

from tiramisu import *

SIZE0 = 100

init("matmul")

# -------------------------------------------------------
# Layer I
# -------------------------------------------------------

p0 = constant("N", expr(SIZE0).cast(primitive_t.p_int32))

i, j, k = var("i", 0, p0), var("j", 0, p0), var("k", 0, p0)

# Declare computations that represents the input buffers.  The actual
# input buffers will be declared later.
A = input("A", ["i", "j"], [SIZE0, SIZE0], primitive_t.p_uint8)
B = input("B", ["i", "j"], [SIZE0, SIZE0], primitive_t.p_uint8)

# Declare a computation to initialize the reduction.
C_init = computation("C_init", [i,j], expr(0).cast(primitive_t.p_uint8))

# Declare the reduction operation.  Do not provide any expression during declaration.
C = computation("C", [i,j,k], primitive_t.p_uint8)
# Note that the previous computation has an empty expression (because we can only use C in an expression after its declaration)
C.set_expression(C[i, j, k - 1] + A[i, k] * B[k, j])

# In this example, C does not read the value of C_init, but later
# we indicate that C_init and C both are stored in the same buffer,
# therefore C will read values written by C_init.
# We are working on adding an operator for reduction to perform reduction
# in a straight forward way.

# -------------------------------------------------------
# Layer II
# -------------------------------------------------------

# Tile both computations: C_init and C
# This tiles the loop levels i and j and produces the loop levels by a 32x32 tile.
# i0, j0, i1 and j1 where i0 is the outermost loop level and j1 is the innermost.

i0, j0, i1, j1 = var("i0"), var("j0"), var("i1"), var("j1")
C_init.tile(i, j, 32, 32, i0, j0, i1, j1)
C.tile(i, j, 32, 32, i0, j0, i1, j1)

# Parallelize the outermost loop level i0
C.parallelize(i0)

# Indicate that C is after C_init at the loop level j0 (this means,
# they share the two outermost loops i0 and j0 and starting from j0 C
# is ordered after C_init).
C.after(C_init, j1)

# -------------------------------------------------------
# Layer III
# -------------------------------------------------------

# Declare the buffers.
b_A = buffer("b_A", [expr(SIZE0), expr(SIZE0)], primitive_t.p_uint8, argument_t.a_input)
b_B = buffer("b_B", [expr(SIZE0), expr(SIZE0)], primitive_t.p_uint8, argument_t.a_input)
b_C = buffer("b_C", [expr(SIZE0), expr(SIZE0)], primitive_t.p_uint8, argument_t.a_output)

# Map the computations to a buffer.
A.store_in(b_A)
B.store_in(b_B)

# Store C_init[i,j] in b_C[i,j]
C_init.store_in(b_C, [i,j])
# Store c_C[i,j,k] in b_C[i,j]
C.store_in(b_C, [i,j])
# Note that both of the computations C_init and C store their
# results in the buffer b_C.

# -------------------------------------------------------
# Code Generation
# -------------------------------------------------------

codegen([b_A, b_B, b_C], "build/generated_fct_developers_tutorial_04A.o")