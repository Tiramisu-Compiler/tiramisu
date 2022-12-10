#include "polybench-tiramisu.h"

#define N 40




//initializes a positive semi-definite matrix
int init_array(Halide::Buffer<double> A)
{
  int i, j;
  int n = N;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A(i, j) = (double)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A(i, j) = 0;
      }
      A(i, i) = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  Halide::Buffer<double> B(N,N);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      B(r, s) = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	B(r, s) += A(r, t) * A(s, t);
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A(r, s) = B(r, s);

return 0;
}
int get_nb_exec(){
    if (std::getenv("NB_EXEC")!=NULL)
        return std::stoi(std::getenv("NB_EXEC"));
    else{
        return 30;
    }
}
