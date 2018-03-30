
# @HEADER
# ***********************************************************************
#  
#                HPCCG: Simple Conjugate Gradient Benchmark Code
#                  Copyright (2006) Sandia Corporation
#  
#  Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
#  license for use of this work by or on behalf of the U.S. Government.
#  
#  BSD 3-Clause License
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
#  
#  ************************************************************************
# @HEADER

# Simple hand-tuned makefile.  Modify as necessary for your environment.
# Questions? Contact Mike Heroux (maherou@sandia.gov).
#

#
# 0) Specify compiler and linker:

CXX=g++
LINKER=g++
#CXX=mpicxx
#LINKER=mpicxx


# 1) Build with MPI or not?
#    If you want to run the program with MPI, make sure USE_MPI is set 
#    to -DUSING_MPI

USE_MPI =
#USE_MPI = -DUSING_MPI


# 2) MPI headers:  
#    If you:
#    - Are building MPI mode (-DUSING_MPI is set above).
#    - Do not have the MPI headers installed in a default search directory and
#    - Are not using MPI compiler wrappers
#    Then specify the path to your MPI header file (include a -I)

#MPI_INC = -I/usr/MPICH/SDK.gcc/include


# 3) Specify C++ compiler optimization flags (if any)
#    Typically some reasonably high level of optimization should be used to 
#    enhance performance.

#IA32 with GCC: 
#CPP_OPT_FLAGS = -O3 -funroll-all-loops -malign-double -ftree-vectorizer-verbose=2
CPP_OPT_FLAGS = -O3 -ftree-vectorize

#
# 4) MPI library:
#    If you:
#    - Are building MPI mode (-DUSING_MPI is set above).
#    - Do not have the MPI library installed a default search directory and
#    - Are not using MPI compiler wrappers for linking
#    Then specify the path to your MPI library (include -L and -l directives)

#MPI_LIB = -L/usr/MPICH/SDK.gcc/lib -lmpich

#
# 5) Build with OpenMP or not?
#    If you want to run the program with OpenMP, make sure USING_OMP is set
#    to -DUSING_OMP

#USE_OMP = 
USE_OMP = -DUSING_OMP

#
# 6) OpenMP Compiler argument
#    GCC and Intel compilers require -fopenmp and -openmp, resp.  Other compilers may differ.

OMP_FLAGS = -fopenmp
#OMP_FLAGS = -openmp

#
# 7) System libraries: (May need to add -lg2c before -lm)

SYS_LIB =-lm -L ../../../../Halide/lib/ -lHalide -lz -lpthread -ldl

#
# 6) Specify name if executable (optional):

TARGET = test_HPCCG

################### Derived Quantities (no modification required) ##############

CXXFLAGS= -std=c++11 -I ../../../../Halide/include/ -I ../../blas/level1/waxpby/ $(CPP_OPT_FLAGS) $(OMP_FLAGS) $(USE_OMP) $(USE_MPI) $(MPI_INC)

LIB_PATHS= $(SYS_LIB)

#read_HPC_row.cpp
TEST_CPP = main.cpp generate_matrix.cpp \
	  compute_residual.cpp mytimer.cpp dump_matlab_matrix.cpp \
          HPC_sparsemv.cpp HPCCG.cpp waxpby.cpp ddot.cpp \
          make_local_matrix.cpp exchange_externals.cpp \
          YAML_Element.cpp YAML_Doc.cpp

TEST_OBJ          = $(TEST_CPP:.cpp=.o)

$(TARGET): $(TEST_OBJ)
	$(LINKER) $(CPP_OPT_FLAGS) $(OMP_FLAGS) $(TEST_OBJ) $(LIB_PATHS) ../../blas/level1/waxpby/generated_waxpby.o -o $(TARGET)

test:
	@echo "Not implemented yet..."

clean:
	@rm -f *.o  *~ $(TARGET) $(TARGET).exe test_HPCPCG 
