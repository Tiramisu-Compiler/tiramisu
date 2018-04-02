
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

// Routine to dump matrix in row, col, val format for analysis with Matlab
// Writes to mat.dat
// NOTE:  THIS CODE ONLY WORKS ON SINGLE PROCESSOR RUNS
// Read into matlab using:
//   load mat.dat
//   A=spconvert(mat);

// A - known matrix 

/////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include "dump_matlab_matrix.hpp"

int dump_matlab_matrix( HPC_Sparse_Matrix *A, int rank) {
  const int nrow = A->local_nrow;
  int start_row = nrow*rank; // Each processor gets a section of a chimney stack domain

  FILE * handle = 0;
  if (rank==0) 
    handle = fopen("mat0.dat", "w");
  else if (rank==1)
    handle = fopen("mat1.dat", "w");
  else if (rank==2)
    handle = fopen("mat2.dat", "w");
  else if (rank==3)
    handle = fopen("mat3.dat", "w");
  else return(0);

  for (int i=0; i< nrow; i++) {
      const double * const cur_vals = A->ptr_to_vals_in_row[i];
      const int    * const cur_inds = A->ptr_to_inds_in_row[i];
      const int cur_nnz = A->nnz_in_row[i];
      for (int j=0; j< cur_nnz; j++) fprintf(handle, " %d %d %22.16e\n",start_row+i+1,cur_inds[j]+1,cur_vals[j]);
    }

  fclose(handle);
  return(0);
}
