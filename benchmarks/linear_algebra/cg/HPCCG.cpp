
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
#include <chrono>
#include <vector>
#include <algorithm>
#include "Halide.h"
#include "generated_waxpby.o.h"
#include "generated_cg.o.h"
#include "generated_dot.o.h"
#include "generated_spmv.o.h"


#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0

template<typename T>
inline void init_buffer(Halide::Buffer<T> &buf, T val)
{
    for (int z = 0; z < buf.channels(); z++)
    {
        for (int y = 0; y < buf.height(); y++)
        {
            for (int x = 0; x < buf.width(); x++)
            {
                buf(x, y, z) = val;
            }
        }
    }
}


double median(std::vector<std::chrono::duration<double, std::milli>> scores)
{
    double median;
    size_t size = scores.size();

    std::sort(scores.begin(), scores.end());

    if (size % 2 == 0)
        median = (scores[size / 2 - 1].count() + scores[size / 2].count()) / 2;
    else
        median = scores[size / 2].count();

    return median;
}

int HPCCG_matrix_to_tiramisu_matrix(HPC_Sparse_Matrix *A,
				    Halide::Buffer<int> &row_start,
				    Halide::Buffer<int> &col_idx,
				    Halide::Buffer<double> &A_tiramisu)
{
  const int nrow = (const int) A->local_nrow;
  const int debug_HPCCG_matrix_to_tiramisu_matrix = 0;

  row_start(0) = 0;

  std::cerr << "row_start.size(): " << row_start.number_of_elements() << ". " <<
	       "col_idx.size(): " << col_idx.number_of_elements() << ". " <<
               "A_tiramisu.size(): " <<  A_tiramisu.number_of_elements() << ". " <<
	       "A->total_nnz: " << A->total_nnz << std::endl;
  std::cerr.flush();

  for (int i=0; i< nrow; i++)
    {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int    * const cur_inds = (const int    * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];

      row_start(i+1) = row_start(i) + cur_nnz;

      if (debug_HPCCG_matrix_to_tiramisu_matrix) {
      		std::cerr << "i = " << i << ", row_start(i) = " << row_start(i) << std::endl;
      		std::cerr.flush();
      }

      for (int j=0; j< cur_nnz; j++)
      {
          col_idx(row_start(i)+j) = cur_inds[j];
          A_tiramisu(row_start(i)+j) = cur_vals[j];

	  if (debug_HPCCG_matrix_to_tiramisu_matrix) {
		std::cerr<<"	j = "<<j<< ", col_idx(" << row_start(i)+j << ") = " << col_idx(row_start(i)+j) << ", A_tiramisu(" << row_start(i)+j << ") = " << A_tiramisu(row_start(i)+j) << std::endl;
		std::cerr.flush();
	  }
      }
    }
}

int     print_spmv(// Input HPCCG format
		   HPC_Sparse_Matrix *HPCCG_A,
		   const double * const x,
 		   double * const y,
                   // Input Tiramisu format
                   int tiramisu_NROW,
		   Halide::Buffer<int> &row_start,
		   Halide::Buffer<int> &col_idx,
		   Halide::Buffer<double> &A_tiramisu,
		   Halide::Buffer<double> &p,
		   Halide::Buffer<double> &Ap)
{

  const int nrow = (const int) HPCCG_A->local_nrow;
  std::cerr << "[HPPCG_A]: nrow = " << nrow << std::endl;
  std::cerr << "[Tiramisu_A]: tiramisu_NROW = " << tiramisu_NROW << std::endl;
  std::cerr << "[HPPCG_A]: i iterating in [0, " << nrow << "]" << std::endl;

  for (int i=0; i< nrow; i++)
    {
      std::cerr << "[HPPCG_A]: i is " << i << std::endl;

      double sum = 0.0;
      const double * const cur_vals = (const double * const) HPCCG_A->ptr_to_vals_in_row[i];
      const int    * const cur_inds = (const int    * const) HPCCG_A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) HPCCG_A->nnz_in_row[i];

      std::cerr << "[HPPCG_A]: j iterating in [0, " << cur_nnz << "]" << std::endl;
      for (int j=0; j< cur_nnz; j++)
	{
          std::cerr << "	[HPPCG_A]: j is " << j << std::endl;
          std::cerr << "	[HPPCG_A]: sum = sum + cur_vals[" << j << "]*x[cur_inds[" << j << "]]" << std::endl;
          std::cerr << "	[HPPCG_A]: sum = " << sum << " + cur_vals[" << j << "]*x[" <<   cur_inds[j] << "]" << std::endl;
          std::cerr << "	[HPPCG_A]: sum = " << sum << " + " << cur_vals[j]     <<  "*"   << x[cur_inds[j]] << std::endl;

          sum += cur_vals[j]*x[cur_inds[j]];

	  std::cerr << "	[HPPCG_A]: sum is " << sum << std::endl;
	}

      std::cerr << "[HPPCG_A]: y[" << i << "] = " << sum << std::endl;
      //y[i] = sum;

      // ------------------------------------------------------------
      std::cerr << "[Tiramisu_A]: j iterating in [" << row_start(i) << ", " << row_start(i+1) << "]" << std::endl;
      for (int j=row_start(i); j<row_start(i+1); j++)
	{
          std::cerr << "	[Tiramisu_A]: j is " << j << std::endl;
          std::cerr << "	[Tiramisu_A]: Ap(i) = Ap(i) + A_tiramisu[" << j << "]*p[col_idx[" << j << "]]" << std::endl;
          std::cerr << "	[Tiramisu_A]: Ap(i) = " << Ap(i) << " + A_tiramisu[" << j << "]*p[" << col_idx(j) << "]" << std::endl;
          std::cerr << "	[Tiramisu_A]: Ap(i) = " << Ap(i) << " + " << A_tiramisu(j) << "*" << p(col_idx(j)) << std::endl;

          Ap(i) += A_tiramisu(j) * p(col_idx(j));

	  std::cerr << "	[Tiramisu_A]: Ap(i) is " << Ap(i) << std::endl;
	}

      std::cerr << "[Tiramisu_A]: Ap(" << i << ") = " << Ap(i) << std::endl;
    }
  return(0);
}

int HPCCG_tiramisu(HPC_Sparse_Matrix * A,
	  const double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times, Halide::Buffer<double> &r)
{
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

  Halide::Buffer<double> p(A->total_nnz); // In parallel case, A is rectangular
  Halide::Buffer<double> Ap(A->total_nnz);
  Halide::Buffer<double> rtrans(1);
  Halide::Buffer<double> alpha(1);
  Halide::Buffer<int> NROW(1);
  NROW(0) = nrow;
  Halide::Buffer<double> beta(1);
  Halide::Buffer<double> a(1);
  a(0) = 1.0;
  Halide::Buffer<int> row_start(A->total_nnz+1);
  Halide::Buffer<int> col_idx(A->total_nnz+1);
  Halide::Buffer<double> A_tiramisu(A->total_nnz+1);

  HPCCG_matrix_to_tiramisu_matrix(A, row_start, col_idx, A_tiramisu);

  normr = 0.0;
  rtrans(0) = 0.0;
  alpha(0) = 0.0;
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
  hpccg_waxpby(nrow, 1.0, x, 0.0, x, p.data());

#ifdef USING_MPI
  exchange_externals(A,p.data());
#endif
  HPC_sparsemv(A, p.data(), Ap.data());
  hpccg_waxpby(nrow, 1.0, b, -1.0, Ap.data(), r.data());
  hpccg_ddot(nrow, r.data(), r.data(), rtrans.data());
  normr = sqrt(rtrans(0));

  if (rank==0) cout << "Initial Residual = "<< normr << endl;
  
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_one_iter;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_comm;

  for(int k=1; k<=1 && normr > tolerance; k++ )
    {
      Halide::Buffer<double> Ap1(A->total_nnz);
      alpha(0) = 0.0;
      hpccg_waxpby(nrow, 1.0, r.data(), 0.0, r.data(), p.data()); // r + 0*p -> p
      HPC_sparsemv(A, p.data(), Ap1.data()); // A*p -> Ap
      hpccg_ddot(nrow, p.data(), Ap1.data(), alpha.data()); // p*Ap -> alpha
      alpha(0) = rtrans(0)/alpha(0);
      hpccg_waxpby(nrow, 1.0, x, alpha(0), p.data(), x);// x + alpha*p -> x
      hpccg_waxpby(nrow, 1.0, r.data(), -alpha(0), Ap1.data(), r.data());  //  r - alpha.Ap -> r

      niters = k;
      normr = sqrt(rtrans(0));
    }


  for(int k=2; k<=max_iter && normr > tolerance; k++ )
    {
      Halide::Buffer<double> Ap2(A->total_nnz);
      auto start_one_iter = std::chrono::high_resolution_clock::now();


      alpha(0) = 0.0;
      oldrtrans = rtrans(0);
      dot (NROW.raw_buffer(), r.raw_buffer(), r.raw_buffer(), rtrans.raw_buffer()); // r*r -> rtrans
      beta(0) = rtrans(0)/oldrtrans;

      cg(NROW.raw_buffer(), a.raw_buffer(), r.raw_buffer(), beta.raw_buffer(), p.raw_buffer(), p.raw_buffer(), row_start.raw_buffer(), col_idx.raw_buffer(), A_tiramisu.raw_buffer(), Ap2.raw_buffer(), alpha.raw_buffer()); // r + beta*p -> p; A*p -> Ap; p*Ap -> alpha

      alpha(0) = rtrans(0)/alpha(0);
      hpccg_waxpby(nrow, 1.0, x, alpha(0), p.data(), x);// x + alpha*p -> x
      hpccg_waxpby(nrow, 1.0, r.data(), -alpha(0), Ap2.data(), r.data());  //  r - alpha.Ap -> r


      niters = k;
      normr = sqrt(rtrans(0));
      if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
      cout << "Iteration = "<< k << "   Residual = "<< normr << endl;
      auto end_one_iter = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration_one_iter = end_one_iter - start_one_iter;
      duration_vector_one_iter.push_back(duration_one_iter);
#ifdef USING_MPI
      std::chrono::duration<double,std::milli> duration_one_comm = end_comm - start_comm;
      duration_vector_one_iter.push_back(duration_one_iter);
#endif
    }

  // Store times
  times[1] = median(duration_vector_one_iter); // Iteration total time
#ifdef USING_MPI
  times[5] = median(duration_vector_comm); // exchange boundary time
#endif

  return(0);
}

int HPCCG_ref(HPC_Sparse_Matrix * A,
	  const double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times, double *r)
{
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

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
  hpccg_waxpby(nrow, 1.0, x, 0.0, x, p);
#ifdef USING_MPI
  exchange_externals(A,p);
#endif
  HPC_sparsemv(A, p, Ap);
  hpccg_waxpby(nrow, 1.0, b, -1.0, Ap, r);
  hpccg_ddot(nrow, r, r, &rtrans);
  normr = sqrt(rtrans);

  if (rank==0) cout << "Initial Residual = "<< normr << endl;
  
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_one_iter;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_comm;

  for(int k=1; k<=max_iter && normr > tolerance; k++ )
    {
      auto start_one_iter = std::chrono::high_resolution_clock::now();
      if (k == 1)
	  hpccg_waxpby(nrow, 1.0, r, 0.0, r, p);
      else
	{
	  oldrtrans = rtrans;
	  hpccg_ddot (nrow, r, r, &rtrans); // 2*nrow ops
	  double beta = rtrans/oldrtrans;
	  hpccg_waxpby (nrow, 1.0, r, beta, p, p); // 2*nrow ops
	}
      normr = sqrt(rtrans);
      if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
      cout << "Iteration = "<< k << "   Residual = "<< normr << endl;

#ifdef USING_MPI
      auto start_comm = std::chrono::high_resolution_clock::now();
      exchange_externals(A,p);
      auto end_comm = std::chrono::high_resolution_clock::now();
#endif

      HPC_sparsemv(A, p, Ap); // 2*nnz ops
      double alpha = 0.0;
      hpccg_ddot(nrow, p, Ap, &alpha); // 2*nrow ops
      alpha = rtrans/alpha;
      hpccg_waxpby(nrow, 1.0, x, alpha, p, x);// 2*nrow ops
      hpccg_waxpby(nrow, 1.0, r, -alpha, Ap, r);  // 2*nrow ops
      niters = k;
      auto end_one_iter = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration_one_iter = end_one_iter - start_one_iter;
      duration_vector_one_iter.push_back(duration_one_iter);

#ifdef USING_MPI
      std::chrono::duration<double,std::milli> duration_one_comm = end_comm - start_comm;
      duration_vector_one_iter.push_back(duration_one_iter);
#endif
    }

  // Store times
  times[1] = median(duration_vector_one_iter); // Iteration total time
#ifdef USING_MPI
  times[5] = median(duration_vector_comm); // exchange boundary time
#endif
  delete [] p;
  delete [] Ap;
  return(0);
}
