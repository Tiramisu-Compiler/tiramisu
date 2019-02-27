#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define LARGE_DATA_SET	0
#define MEDIUM_DATA_SET	0
#define SMALL_DATA_SET	1

#if LARGE_DATA_SET
	#define BATCH_SIZE 100 
#elif MEDIUM_DATA_SET
	#define BATCH_SIZE 32
#elif SMALL_DATA_SET
	#define BATCH_SIZE 8
#endif

// Size of one data dimension
// Data is NxNx16

#if LARGE_DATA_SET
	#define N 512
#elif MEDIUM_DATA_SET
	#define N 64
#elif SMALL_DATA_SET
	#define N 32
#endif


// Number of features in the input
#define FIn 16
// Number of features in the output
#define FOut 16

// Size of convolution filter ( FOut xFIn x K x K)
#define K 4


// Number of tests 
#define NB_TESTS 3

/** define relative to the sizes**/ 
#define N1 N+K
#define N2 N-K
#define N3 N-2*K
#define pK K+1


static double rtclock()
{
struct timeval Cnv;
gettimeofday (&Cnv, NULL);
return (Cnv.tv_sec* 1.0e3 + Cnv.tv_usec * 1.0e-3);
}

static float Max(float a, float b) {
    if (a < b)
        return b;
    return a;
}
static double median(int n, double x[NB_TESTS])
{
    double temp;
    int i, j;

    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}



