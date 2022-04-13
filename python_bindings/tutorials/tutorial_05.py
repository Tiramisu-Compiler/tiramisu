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

init("a")

function("function0")

print("a")
