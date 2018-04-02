
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
#include "HPC_Sparse_Matrix.hpp"

#ifdef USING_MPI
#include <mpi.h>
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void destroyMatrix(HPC_Sparse_Matrix * &A)
{
  if(A->title)
  {
    delete [] A->title;
  }
  if(A->nnz_in_row)
  {
    delete [] A->nnz_in_row;
  }
  if(A->list_of_vals)
  {
    delete [] A->list_of_vals;
  }
  if(A->ptr_to_vals_in_row !=0)
  {
    delete [] A->ptr_to_vals_in_row;
  }
  if(A->list_of_inds)
  {
    delete [] A->list_of_inds;
  }
  if(A->ptr_to_inds_in_row !=0)
  {
    delete [] A->ptr_to_inds_in_row;
  }
  if(A->ptr_to_diags)
  {
    delete [] A->ptr_to_diags;
  }

#ifdef USING_MPI
  if(A->external_index)
  {
    delete [] A->external_index;
  }
  if(A->external_local_index)
  {
    delete [] A->external_local_index;
  }
  if(A->elements_to_send)
  {
    delete [] A->elements_to_send;
  }
  if(A->neighbors)
  {
    delete [] A->neighbors;
  }
  if(A->recv_length)
  {
    delete [] A->recv_length;
  }
  if(A->send_length)
  {
    delete [] A->send_length;
  }
  if(A->send_buffer)
  {
    delete [] A->send_buffer;
  }
#endif

  delete A;
  A = 0;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_SHAREDMEM_MPI
#ifndef SHAREDMEM_ALTERNATIVE
void destroySharedMemMatrix(HPC_Sparse_Matrix * &A)
{
  if(A==0)
  {
    return; //noop
  }

  if(A->title)
  {
    delete [] A->title;
  }

  if(A->nnz_in_row)
  {
    MPI_Comm_free_mem(MPI_COMM_NODE,A->nnz_in_row);
  }
  if(A->list_of_vals)
  {
    MPI_Comm_free_mem(MPI_COMM_NODE,A->list_of_vals);
  }
  if(A->ptr_to_vals_in_row !=0)
  {
    MPI_Comm_free_mem(MPI_COMM_NODE,A->ptr_to_vals_in_row);
  }
  if(A->list_of_inds)
  {
    MPI_Comm_free_mem(MPI_COMM_NODE,A->list_of_inds);
  }
  if(A->ptr_to_inds_in_row !=0)
  {
    MPI_Comm_free_mem(MPI_COMM_NODE,A->ptr_to_inds_in_row);
  }

  // currently not allocated with shared memory
  if(A->ptr_to_diags)
  {
    delete [] A->ptr_to_diags;
  }


#ifdef USING_MPI
  if(A->external_index)
  {
    delete [] A->external_index;
  }
  if(A->external_local_index)
  {
    delete [] A->external_local_index;
  }
  if(A->elements_to_send)
  {
    delete [] A->elements_to_send;
  }
  if(A->neighbors)
  {
    delete [] A->neighbors;
  }
  if(A->recv_length)
  {
    delete [] A->recv_length;
  }
  if(A->send_length)
  {
    delete [] A->send_length;
  }
  if(A->send_buffer)
  {
    delete [] A->send_buffer;
  }
#endif

  MPI_Comm_free_mem(MPI_COMM_NODE,A); A=0;

}
#endif
#endif
////////////////////////////////////////////////////////////////////////////////

