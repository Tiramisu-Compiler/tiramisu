
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

// Function to return time in seconds.
// If compiled with no flags, return CPU time (user and system).
// If compiled with -DWALL, returns elapsed time.

/////////////////////////////////////////////////////////////////////////
#ifdef USING_MPI
#include <mpi.h> // If this routine is compiled with -DUSING_MPI
                 // then include mpi.h
double mytimer(void)
{
   return(MPI_Wtime());
}


#elif defined(UseClock)

#include <time.hpp>
double mytimer(void)
{
   clock_t t1;
   static clock_t t0=0;
   static double CPS = CLOCKS_PER_SEC;
   double d;

   if (t0 == 0) t0 = clock();
   t1 = clock() - t0;
   d = t1 / CPS;
   return(d);
}

#elif defined(WALL)

#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
double mytimer(void)
{
   struct timeval tp;
   static long start=0, startu;
   if (!start)
   {
      gettimeofday(&tp, NULL);
      start = tp.tv_sec;
      startu = tp.tv_usec;
      return(0.0);
   }
   gettimeofday(&tp, NULL);
   return( ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
}

#elif defined(UseTimes)

#include <cstdlib>
#include <sys/times.h>
#include <unistd.h>
double mytimer(void)
{
   struct tms ts;
   static double ClockTick=0.0;

   if (ClockTick == 0.0) ClockTick = (double) sysconf(_SC_CLK_TCK);
   times(&ts);
   return( (double) ts.tms_utime / ClockTick );
}

#else

#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
double mytimer(void)
{
   struct rusage ruse;
   getrusage(RUSAGE_SELF, &ruse);
   return( (double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0) );
}

#endif
