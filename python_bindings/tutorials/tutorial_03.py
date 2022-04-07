'''
This example defines the following sequence of computations.
for (i = 0; i < M; i++)
  S0(i) = 4;
  S1(i) = 3;
  for (j = 0; j < N; j++)
    S2(i, j) = 2;
  S3(i) = 1;
 
 The goal of this tutorial is to show how one can indicate
 the order of computations in Tiramisu.
'''

from tiramisu import *

SIZE0 = 10
    
init("sequence")

# -------------------------------------------------------
# Layer I
# -------------------------------------------------------

M = constant("M", expr(SIZE0).cast(primitive_t.uint32_t))

i, j = var("i", 0, M),  var("j", 0, M)

# Declare the four computations: c0, c1, c2 and c3.
c0 = computation("c0", [i], expr(4).cast(primitive_t.p_int8))
c1 = computation("c1", [i], expr(3).cast(primitive_t.p_int8))
c2 = computation("c2", [i,j], expr(2).cast(primitive_t.p_int8))
c3 = computation("c3", [i], expr(1).cast(primitive_t.p_int8))

# -------------------------------------------------------
# Layer II
# -------------------------------------------------------

# By default computations are unordered in Tiramisu. The user has to specify
# the order exlplicitely (automatic ordering is being developed and will be
# available soon).
#
# The following calls define the order between the computations c3, c2, c1 and c0.
# c1 is set to be after c0 in the loop level i.  That is, both have the same outer loops
# up to the loop level i (they share i also) but starting from i, all the
# computations c1 are ordered after the computations c0.
c1.after(c0, i)
c2.after(c1, i)
c3.after(c2, i)

# -------------------------------------------------------
# Layer III
# -------------------------------------------------------

b0 = buffer("b0", [expr(SIZE0)], primitive_t.p_uint8, argument_t.a_output)
b1 = buffer("b1", [expr(SIZE0)], primitive_t.p_uint8, argument_t.a_output)
b2 = buffer("b2", [expr(SIZE0), expr(SIZE0)], primitive_t.p_uint8, argument_t.a_output)
b3 = buffer("b3", [expr(SIZE0)], primitive_t.p_uint8, argument_t.a_output)

c0.store_in(b0)
c1.store_in(b1)
c2.store_in(b2)
c3.store_in(b3)

# -------------------------------------------------------
# Code Generator
# -------------------------------------------------------

codegen([b0, b1, b2, b3], "build/generated_fct_developers_tutorial_03.o")
