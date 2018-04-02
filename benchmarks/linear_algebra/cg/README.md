# HPCCG
High Performance Computing Conjugate Gradients:  The original Mantevo miniapp
------------------------------------------------
Description:
------------------------------------------------
HPCCG: A simple conjugate gradient benchmark code for a 3D chimney 
domain on an arbitrary number of processors.

Author: Michael A. Heroux, Sandia National Laboratories (maherou@sandia.gov)

This simple benchmark code is a self-contained piece of C++ software 
that generates a 27-point finite difference matrix with a user-prescribed 
sub-block size on each processor.

It is implemented to be very scalable (in a weak sense).  Any 
reasonable parallel computer should be able to achieve excellent 
scaled speedup (weak scaling).  

Kernel performance should be reasonable, but no attempts have been made
to provide special kernel optimizations.

------------------------------------------------
Compiling the code:
------------------------------------------------

There is a simple Makefile that should be easily modified for most 
Unix-like environments.  There are also a few Makefiles with extensions 
that indicate the target machine and compilers. Read the Makefile for 
further instructions.  If you generate a Makefile for your platform 
and care to share it, please send it to the author.

By default the code compiles with MPI support and can be run on one 
or more processors.  If you don't have MPI, or want to compile without 
MPI support, you may change the definition of USE_MPI in the 
makefile, or use make as follows:

`make USE_MPI=`

To remove all output files, type:

`make clean`

------------------------------------------------
Running the code:
------------------------------------------------

Usage:

`test_HPCCG nx ny nz` (serial mode)

`mpirun -np numproc test_HPCCG nx ny nz` (MPI mode)

where nx, ny, nz are the number of nodes in the x, y and z 
dimension respectively on a each processor.
The global grid dimensions will be nx, ny and numproc * nz.  
In other words, the domains are stacked in the z direction.

Example:

`mpirun -np 16 ./test_HPCCG 20 30 10`

This will construct a local problem of dimension 20-by-30-by-10 
whose global problem has dimension 20-by-30-by-160.

--------------------
Using OpenMP and MPI
--------------------

The values of nx, ny and nz are the local problem size.  The global size
is nx-by-ny-by-(nz * number of MPI ranks).

The number of OpenMP threads is defined by the standard OpenMP mechanisms.
Typically this value defaults to the maximum number of reasonable threads a
compute node can support.  The number of threads can be modified by defining
the environment variable OMP_NUM_THREADS. 
To set the number of threads to 4:

In tcsh or csh: `setenv OMP_NUM_THREADS 4`
In sh or bash: `export OMP_NUM_THREADS=4`

You can also define it when executing the run of HPCCG:

`ENV OMP_NUM_THREADS=4 mpirun -np 16 ./test_HPCCG 50 50 50`

---------------------------------
What size problem is a good size?
---------------------------------

I think the best way to give this guidance is to pick the problems so that 
the data size is over a range from 25% of total system memory up to 75%.

If nx=ny=nz and n = nx * ny * nz, local to each MPI rank, then the number of bytes 
used for each rank works like this:

Matrix storage: 336 * n bytes total (27 pt stencil), 96 * n bytes total (7 pt stencil)
27 * n  or 7 * n, 12 bytes per nonzero: 324 * n bytes total or 84 * n bytes total
n pointers for start of rows, 8 bytes per pointer: 8 * n bytes total
n integers for nnz per row: 4 * n bytes.

Preconditioner: Roughly same as matrix

Algorithm vectors: 48 * n bytes total
6 * n double vectors

Total memory per MPI rank:720 * n bytes for 27 pt stencil, 240 * n bytes for 7 pt stencil.

On an 16GB system with 4 MPI ranks running with the 27 pt stencil: 
- 25% of the memory would allow 1GB per MPI rank.  
  n would approximately be 1GB/720, so 1.39M and nx=ny=nz=100.

- 75% of the memory would allow 3GB per MPI rank.  
  n would approximately be 3GB/720, so 4.17M and nx=ny=nz=161.

Alternate usage:

There is an alternate mode that allows specification of a data 
file containing a general sparse matrix.  This usage is deprecated.  
Please contact the author if you have need for this more general case.


-------------------------------------------------
Changing the sparse matrix structure:
-------------------------------------------------

HPCCG supports two sparse matrix data structures: a 27-pt 3D grid based
structure and a 7-pt 3D grid based structure.  To switch between the two
change the bool value for use_7pt_stencil in generate_matrix.cpp.
