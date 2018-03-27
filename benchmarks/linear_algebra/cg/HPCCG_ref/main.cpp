
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

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// read_HPC_row - Reads in linear system

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <cmath>
#ifdef USING_MPI
#include <mpi.h> // If this routine is compiled with -DUSING_MPI
                 // then include mpi.h
#include "make_local_matrix.hpp" // Also include this function
#endif
#ifdef USING_OMP
#include <omp.h>
#endif
#include "generate_matrix.hpp"
#include "mytimer.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
//#include "read_HPC_row.hpp"
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"
#include "dump_matlab_matrix.hpp"

#include "YAML_Element.hpp"
#include "YAML_Doc.hpp"
#include "Halide.h"

#undef DEBUG

#define MAX_ITER 150

int main_ref(int argc, char *argv[], double **r, double &normr)
{
  HPC_Sparse_Matrix *A;
  double *x, *b, *xexact;
  double norm, d;
  int ierr = 0;
  int i, j;
  int ione = 1;
  double times[7];
  double t6 = 0.0;
  int nx,ny,nz;

#ifdef USING_MPI
  MPI_Init(&argc, &argv);
  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //  if (size < 100) cout << "Process "<<rank<<" of "<<size<<" is alive." <<endl;
#else
  int size = 1; // Serial case (not using MPI)
  int rank = 0; 
#endif

  if (argc==4) 
  {
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
  }
  else
  {
    //read_HPC_row(argv[1], &A, &x, &b, &xexact);
    std::cerr << "Reading a linear system is disabled\n";
    exit(0);
  }


  bool dump_matrix = false;
  if (dump_matrix && size<=4) dump_matlab_matrix(A, rank);

#ifdef USING_MPI
  // Transform matrix indices from global to local values.
  // Define number of columns for the local matrix.
  t6 = mytimer(); make_local_matrix(A);  t6 = mytimer() - t6;
  times[6] = t6;
#endif

  double t1 = mytimer();   // Initialize it (if needed)
  int niters = 0;
  int max_iter = MAX_ITER; //150 is the default.
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  int nrow = A->local_nrow;
  *r = (double *) malloc(sizeof(double)*nrow);

  ierr = HPCCG_ref(A, b, x, max_iter, tolerance, niters, normr, times, *r);
  if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      double fniters = 1; //niters; 
      double fnrow = A->total_nrow;
      double fnnz = A->total_nnz;
      double fnops_ddot = fniters*4*fnrow;
      double fnops_waxpby = fniters*6*fnrow;
      double fnops_sparsemv = fniters*2*fnnz;
      double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv;


#ifdef USING_MPI
          std::cout << "MPI (Number of ranks = " << size << ") - ";
#else
          std::cout << "MPI (not enabled) - ";
#endif

#ifdef USING_OMP
      int nthreads = 1;
      #pragma omp parallel
      nthreads = omp_get_num_threads();
      std::cout << "OpenMP (Number of threads = " << nthreads << ")" << std::endl;
#else
      std::cout << "OpenMP (not enabled)" << std::endl; 
#endif

      std::cout << "Dimensions : (nx, ny, nz) = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;
      std::cout << "Number of iterations: " << niters  << ".  ";
      std::cout << "Final residual: " << normr  << std::endl;
      std::cout << "Time (per iteration): " << times[1] << ";" << std::endl;
      std::cout << "Total number of Floating operations (FLOPS): " << fnops << std::endl;
      std::cout << "MFLOPS: " << fnops/times[1]/1.0E6 << std::endl;
    }


  // Compute difference between known exact solution and computed solution
  // All processors are needed here.

  double residual = 0;
  //  if ((ierr = compute_residual(A->local_nrow, x, xexact, &residual)))
  //  cerr << "Error in call to compute_residual: " << ierr << ".\n" << endl;

  // if (rank==0)
  //   cout << "Difference between computed and exact  = " 
  //        << residual << ".\n" << endl;


  // Finish up 
#ifdef USING_MPI
  MPI_Finalize();
#endif
  return nrow ;
}

int main_tiramisu(int argc, char *argv[], Halide::Buffer<double> &r_tiramisu, double &normr)
{
  HPC_Sparse_Matrix *A;
  double *x, *b, *xexact;
  double norm, d;
  int ierr = 0;
  int i, j;
  int ione = 1;
  double times[7];
  double t6 = 0.0;
  int nx,ny,nz;

#ifdef USING_MPI
  MPI_Init(&argc, &argv);
  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //  if (size < 100) cout << "Process "<<rank<<" of "<<size<<" is alive." <<endl;
#else
  int size = 1; // Serial case (not using MPI)
  int rank = 0; 
#endif

  if (argc==4) 
  {
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
  }
  else
  {
    //read_HPC_row(argv[1], &A, &x, &b, &xexact);
    std::cerr << "Reading a linear system is disabled\n";
    exit(0);
  }


  bool dump_matrix = false;
  if (dump_matrix && size<=4) dump_matlab_matrix(A, rank);

#ifdef USING_MPI
  // Transform matrix indices from global to local values.
  // Define number of columns for the local matrix.
  t6 = mytimer(); make_local_matrix(A);  t6 = mytimer() - t6;
  times[6] = t6;
#endif

  double t1 = mytimer();   // Initialize it (if needed)
  int niters = 0;
  int max_iter = MAX_ITER; //150 is the default.
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  int nrow = A->local_nrow;
  //*r = (double *) malloc(sizeof(double)*nrow);

  ierr = HPCCG_tiramisu(A, b, x, max_iter, tolerance, niters, normr, times, r_tiramisu);
  if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      double fniters = 1; //niters; 
      double fnrow = A->total_nrow;
      double fnnz = A->total_nnz;
      double fnops_ddot = fniters*4*fnrow;
      double fnops_waxpby = fniters*6*fnrow;
      double fnops_sparsemv = fniters*2*fnnz;
      double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv;

#ifdef USING_MPI
      std::cout << "MPI (Number of ranks = " << size << ") - ";
#else
      std::cout << "MPI (not enabled) - ";
#endif

#ifdef USING_OMP
      int nthreads = 1;
      #pragma omp parallel
      nthreads = omp_get_num_threads();
      std::cout << "OpenMP (Number of threads = " << nthreads << ")" << std::endl;
#else
      std::cout << "OpenMP (not enabled)" << std::endl; 
#endif

      std::cout << "Dimensions : (nx, ny, nz) = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;
      std::cout << "Number of iterations: " << niters  << ".  ";
      std::cout << "Final residual: " << normr  << std::endl;
      std::cout << "Time (per iteration): " << times[1] << ";" << std::endl;
      std::cout << "Total number of Floating operations (FLOPS): " << fnops << std::endl;
      std::cout << "MFLOPS: " << fnops/times[1]/1.0E6 << std::endl;
    }

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.

  double residual = 0;
  //  if ((ierr = compute_residual(A->local_nrow, x, xexact, &residual)))
  //  cerr << "Error in call to compute_residual: " << ierr << ".\n" << endl;

  // if (rank==0)
  //   cout << "Difference between computed and exact  = " 
  //        << residual << ".\n" << endl;


  // Finish up 
#ifdef USING_MPI
  MPI_Finalize();
#endif
  return nrow;
} 

int flush_cache()
{
  int cs = (1024 * 1024 * 1024);
  double* flush = (double*) calloc (cs, sizeof(double));
  double* flush2 = (double*) calloc (cs, sizeof(double));
  double* flush3 = (double*) calloc (cs, sizeof(double));
  double* flush4 = (double*) calloc (cs, sizeof(double));

  int i;
  double tmp = 0.0;
#pragma omp parallel for
  for (i = 0; i < cs; i++)
  {
    tmp += flush[i];
    tmp += flush2[i] + flush3[i] + flush4[i];
  }
  free (flush);
  free (flush2);
  free (flush3);
  free (flush4);

  return tmp;
}

int main(int argc, char *argv[])
{
  assert(argc >= 3);

  double * r_ref;
  Halide::Buffer<double> r_tiramisu(atoi(argv[1])*atoi(argv[2])*atoi(argv[3]));
  double normr_tiramisu = 0.0;
  double normr_ref = 0.0;

  int res = flush_cache();
  std::cout << "*************************** Tiramisu CG **************************************" << std::endl;
  main_tiramisu(argc, argv, r_tiramisu, normr_tiramisu);
  std::cout << "*************************** Reference CG ***************************************" << std::endl;
  res += flush_cache();
  int nrow = main_ref(argc, argv, &r_ref, normr_ref);
  std::cout << "******************************************************************" << std::endl;

  if (std::abs(normr_ref - normr_tiramisu) >= 0.000000000000000000001)
  {
	std::cerr << "Residuals of the two computations are not equal" << std::endl;
        std::cerr << "normr_ref = " << normr_ref << " and normr_tiramisu = " << normr_tiramisu << std::endl;
        std::cerr << "normr_ref - normr_tiramisu = " << normr_ref - normr_tiramisu << std::endl;
        exit(1);
  }

  // Compare r_ref and r_tiramisu
  for (int i = 0; i < nrow; i++)
    if (std::abs(r_ref[i] - r_tiramisu(i)) >= 0.0001)
    {
	std::cerr << "r_ref[" << i << "] != r_tiramisu[" << i << "]" << std::endl;
        exit(1);
    }
  std::cout << "Correct computations." << std::endl;
  std::cout << "Residuals equal." << std::endl;


  free(r_ref);

  return res ;
}
