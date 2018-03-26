
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

#undef DEBUG

int main_ref(int argc, char *argv[], double **r)
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


#ifdef DEBUG
  if (rank==0)
   {
    int junk = 0;
    cout << "Press enter to continue"<< endl;
    cin >> junk;
   }

  MPI_Barrier(MPI_COMM_WORLD);
#endif


  if(argc != 2 && argc!=4) {
    if (rank==0)
      cerr << "Usage:" << endl
	   << "Mode 1: " << argv[0] << " nx ny nz" << endl
	   << "     where nx, ny and nz are the local sub-block dimensions, or" << endl
	   << "Mode 2: " << argv[0] << " HPC_data_file " << endl
	   << "     where HPC_data_file is a globally accessible file containing matrix data." << endl;
    exit(1);
  }

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
  double normr = 0.0;
  int max_iter = 45; //150 is the default.
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  int nrow = A->local_nrow;
  *r = (double *) malloc(sizeof(double)*nrow);

  ierr = HPCCG_ref(A, b, x, max_iter, tolerance, niters, normr, times);
  if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;

#ifdef USING_MPI
      double t4 = times[4];
      double t4min = 0.0;
      double t4max = 0.0;
      double t4avg = 0.0;
      MPI_Allreduce(&t4, &t4min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      t4avg = t4avg/((double) size);
#endif

// initialize YAML doc

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      double fniters = niters; 
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

      std::cout << "Dimensions : (nx, ny, nz) = (" << nx << ", ";
      std::cout << ny << ", ";
      std::cout << nz << ")" << std::endl;

      std::cout << "Number of iterations: " << niters  << ".  ";
      std::cout << "Final residual: " << normr  << std::endl;
 
      std::cout << "Time (sec) - Total: " << times[0] << ";  DDOT: " << times[1] << ";  WAXPBY: " << times[2] << ";  SPARSEMV: " << times[3] << std::endl;

      std::cout << "FLOPS Summary - Total: " << fnops;
      std::cout << ";  DDOT: " << fnops_ddot;
      std::cout << ";  WAXPBY: " << fnops_waxpby;
      std::cout << ";  SPARSEMV: " << fnops_sparsemv  << std::endl;

      std::cout << "MFLOPS Summary - Total: " << fnops/times[0]/1.0E6;
      std::cout << ";  DDOT: " << fnops_ddot/times[1]/1.0E6;
      std::cout << ";  WAXPBY: " << fnops_waxpby/times[2]/1.0E6;
      std::cout << ";  SPARSEMV: " << fnops_sparsemv/(times[3])/1.0E6  << std::endl;

#ifdef USING_MPI
      std::cout << "DDOT Timing Variations - Min DDOT MPI_Allreduce time: " << t4min << ";  ";
      std::cout << "Max DDOT MPI_Allreduce time: " << t4max << ";  ";
      std::cout << "Avg DDOT MPI_Allreduce time: " << t4avg << std::endl;

      double totalSparseMVTime = times[3] + times[5]+ times[6];
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV MFLOPS W OVERHEAD: " << fnops_sparsemv/(totalSparseMVTime)/1.0E6 << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Time: " << (times[5]+times[6]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Pct: " << (times[5]+times[6])/totalSparseMVTime*100.0 << ";  " << std::endl;
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV PARALLEL OVERHEAD Setup Time: " << (times[6]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Setup Pct: " << (times[6])/totalSparseMVTime*100.0 << ";  " << std::endl;
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV PARALLEL OVERHEAD Bdry Exch Time: " << (times[5]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Bdry Exch Pct: " << (times[5])/totalSparseMVTime*100.0 << std::endl;
#endif

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

int main_tiramisu(int argc, char *argv[], double **r)
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


#ifdef DEBUG
  if (rank==0)
   {
    int junk = 0;
    cout << "Press enter to continue"<< endl;
    cin >> junk;
   }

  MPI_Barrier(MPI_COMM_WORLD);
#endif


  if(argc != 2 && argc!=4) {
    if (rank==0)
      cerr << "Usage:" << endl
	   << "Mode 1: " << argv[0] << " nx ny nz" << endl
	   << "     where nx, ny and nz are the local sub-block dimensions, or" << endl
	   << "Mode 2: " << argv[0] << " HPC_data_file " << endl
	   << "     where HPC_data_file is a globally accessible file containing matrix data." << endl;
    exit(1);
  }

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
  double normr = 0.0;
  int max_iter = 45; //150 is the default.
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  int nrow = A->local_nrow;
  *r = (double *) malloc(sizeof(double)*nrow);

  ierr = HPCCG_tiramisu(A, b, x, max_iter, tolerance, niters, normr, times);
  if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;

#ifdef USING_MPI
      double t4 = times[4];
      double t4min = 0.0;
      double t4max = 0.0;
      double t4avg = 0.0;
      MPI_Allreduce(&t4, &t4min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      t4avg = t4avg/((double) size);
#endif

// initialize YAML doc

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      double fniters = niters; 
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

      std::cout << "Dimensions : (nx, ny, nz) = (" << nx << ", ";
      std::cout << ny << ", ";
      std::cout << nz << ")" << std::endl;

      std::cout << "Number of iterations: " << niters  << ".  ";
      std::cout << "Final residual: " << normr  << std::endl;
 
      std::cout << "Time (sec) - Total: " << times[0] << ";  DDOT: " << times[1] << ";  WAXPBY: " << times[2] << ";  SPARSEMV: " << times[3] << std::endl;

      std::cout << "FLOPS Summary - Total: " << fnops;
      std::cout << ";  DDOT: " << fnops_ddot;
      std::cout << ";  WAXPBY: " << fnops_waxpby;
      std::cout << ";  SPARSEMV: " << fnops_sparsemv  << std::endl;

      std::cout << "MFLOPS Summary - Total: " << fnops/times[0]/1.0E6;
      std::cout << ";  DDOT: " << fnops_ddot/times[1]/1.0E6;
      std::cout << ";  WAXPBY: " << fnops_waxpby/times[2]/1.0E6;
      std::cout << ";  SPARSEMV: " << fnops_sparsemv/(times[3])/1.0E6  << std::endl;

#ifdef USING_MPI
      std::cout << "DDOT Timing Variations - Min DDOT MPI_Allreduce time: " << t4min << ";  ";
      std::cout << "Max DDOT MPI_Allreduce time: " << t4max << ";  ";
      std::cout << "Avg DDOT MPI_Allreduce time: " << t4avg << std::endl;

      double totalSparseMVTime = times[3] + times[5]+ times[6];
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV MFLOPS W OVERHEAD: " << fnops_sparsemv/(totalSparseMVTime)/1.0E6 << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Time: " << (times[5]+times[6]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Pct: " << (times[5]+times[6])/totalSparseMVTime*100.0 << ";  " << std::endl;
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV PARALLEL OVERHEAD Setup Time: " << (times[6]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Setup Pct: " << (times[6])/totalSparseMVTime*100.0 << ";  " << std::endl;
      std::cout << "SPARSEMV OVERHEADS - SPARSEMV PARALLEL OVERHEAD Bdry Exch Time: " << (times[5]) << ";  ";
      std::cout << "SPARSEMV PARALLEL OVERHEAD Bdry Exch Pct: " << (times[5])/totalSparseMVTime*100.0 << std::endl;
#endif

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

int main(int argc, char *argv[])
{
  double * r_ref;
  double * r_tiramisu;

  std::cout << "*************************** Reference CG ***************************************" << std::endl;
  int nrow = main_ref(argc, argv, &r_tiramisu);
  std::cout << "*************************** Tiramisu CG **************************************" << std::endl;
  main_tiramisu(argc, argv, &r_ref);
  std::cout << "******************************************************************" << std::endl;

  // Compare r_ref and r_tiramisu
  for (int i = 0; i < nrow; i++)
    if (r_ref[i] != r_tiramisu[i])
    {
	std::cerr << "r_ref[" << i << "] != r_tiramisu[" << i << "]" << std::endl;
        exit(1);
    }
  std::cerr << "Correct computations." << std::endl;

  free(r_ref);
  free(r_tiramisu);

  return 0 ;
}
