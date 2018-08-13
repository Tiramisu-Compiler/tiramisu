/*
 * benchmarks.h
 *
 *  Created on: Oct 13, 2016
 *      Author: b
 */

#ifndef BENCHMARKS_BENCHMARKS_H_
#define BENCHMARKS_BENCHMARKS_H_


#define NB_TESTS 60
#define CHECK_CORRECTNESS 1
#define PRINT_OUTPUT 0 
#define SIZE_IS_MULTIPLE_OF_TILE 1


// BARYON_N is used for loop iterators.
#define BARYON_N 16
// BARYON_P1 is used for the size of first dimension
// of array and possible value of the parameters used
// in that first dimension.
#define BARYON_P1 3
// BARYON_P is used for the size of the other array
// dimensions that are not of size BARYON_N
#define BARYON_P 1

// Data size
#if TIRAMISU_XLARGE

#define N 4096
#define M N
#define K N
#define SIZE (65536*4096)

#elif TIRAMISU_LARGE

#if SIZE_IS_MULTIPLE_OF_TILE
#define N 1024
#define M 1024
#define K 1024
#define SIZE (1024*1024)
#else
#define N 1060
#define M 1060
#define K 1060
#endif

#elif TIRAMISU_MEDIUM

#define N 512
#define M 512
#define K 512
#define SIZE (1024)

#elif TIRAMISU_SMALL

#define N 128
#define M 128
#define K 128
#define SIZE (128)

#else

#define N 1024
#define M 1024
#define K 1024

#endif

#endif /* BENCHMARKS_BENCHMARKS_H_ */
