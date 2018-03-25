
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER
/////////////////////////////////////////////////////////////////////////

// Routine to compute an approximate solution to Ax = b where:

// A - known matrix stored as an HPC_Sparse_Matrix struct

// b - known right hand side vector

// x - On entry is initial guess, on exit new approximate solution

// max_iter - Maximum number of iterations to perform, even if
//            tolerance is not met.

// tolerance - Stop and assert convergence if norm of residual is <=
//             to tolerance.

// niters - On output, the number of iterations actually performed.

/////////////////////////////////////////////////////////////////////////

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cmath>
#include "mytimer.hpp"
#include "HPCCG.hpp"

#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0
int HPCCG(HPC_Sparse_Matrix * A,
	  const double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times)

{
  double t_begin = mytimer();  // Start timing right away

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
#ifdef USING_MPI
  double t5 = 0.0;
#endif
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

  double * r = new double [nrow];
  double * p = new double [ncol]; // In parallel case, A is rectangular
  double * Ap = new double [nrow];

  normr = 0.0;
  double rtrans = 0.0;
  double oldrtrans = 0.0;

#ifdef USING_MPI
  int rank; // Number of MPI processes, My process ID
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank = 0; // Serial case (not using MPI)
#endif

  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;

  // p is of length ncols, copy x to p for sparse MV operation
  TICK(); waxpby(nrow, 1.0, x, 0.0, x, p); TOCK(t2);
#ifdef USING_MPI
  TICK(); exchange_externals(A,p); TOCK(t5); 
#endif
  TICK(); HPC_sparsemv(A, p, Ap); TOCK(t3);
  TICK(); waxpby(nrow, 1.0, b, -1.0, Ap, r); TOCK(t2);
  TICK(); ddot(nrow, r, r, &rtrans, t4); TOCK(t1);
  normr = sqrt(rtrans);

  if (rank==0) cout << "Initial Residual = "<< normr << endl;

  for(int k=1; k<max_iter && normr > tolerance; k++ )
    {
      if (k == 1)
	{
	  TICK(); waxpby(nrow, 1.0, r, 0.0, r, p); TOCK(t2);
	}
      else
	{
	  oldrtrans = rtrans;
	  TICK(); ddot (nrow, r, r, &rtrans, t4); TOCK(t1);// 2*nrow ops
	  double beta = rtrans/oldrtrans;
	  TICK(); waxpby (nrow, 1.0, r, beta, p, p);  TOCK(t2);// 2*nrow ops
	}
      normr = sqrt(rtrans);
      if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
      cout << "Iteration = "<< k << "   Residual = "<< normr << endl;
     

#ifdef USING_MPI
      TICK(); exchange_externals(A,p); TOCK(t5); 
#endif
      TICK(); HPC_sparsemv(A, p, Ap); TOCK(t3); // 2*nnz ops
      double alpha = 0.0;
      TICK(); ddot(nrow, p, Ap, &alpha, t4); TOCK(t1); // 2*nrow ops
      alpha = rtrans/alpha;
      TICK(); waxpby(nrow, 1.0, x, alpha, p, x);// 2*nrow ops
      waxpby(nrow, 1.0, r, -alpha, Ap, r);  TOCK(t2);// 2*nrow ops
      niters = k;
    }

  // Store times
  times[1] = t1; // ddot time
  times[2] = t2; // waxpby time
  times[3] = t3; // sparsemv time
  times[4] = t4; // AllReduce time
#ifdef USING_MPI
  times[5] = t5; // exchange boundary time
#endif
  delete [] p;
  delete [] Ap;
  delete [] r;
  times[0] = mytimer() - t_begin;  // Total time. All done...
  return(0);
}
