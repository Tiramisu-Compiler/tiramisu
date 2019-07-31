#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#include <sys/time.h>

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

// Width and height of an input tensor
#define N 112

// Number of features in the input
#define FIn 3
// Number of features in the output
#define FOut 32

// Size of convolution filter (KxK)
#define K 3

// Parameters for Tiramisu code
#define FIN2_BLOCKING 8
#define FOUT_BLOCKING 8

#define FIN1_NB_BLOCKS FIn/FIN1_BLOCKING
#define FIN2_NB_BLOCKS FOut/FIN2_BLOCKING
#define FOUT_NB_BLOCKS FOut/FOUT_BLOCKING

#if N >= 224
    #define X_BLOCKING 8
    #define Y_BLOCKING 2
    #define SCHEDULE_PREFETCH_WEIGHTS true
#else
    #define X_BLOCKING 4
    #define Y_BLOCKING 1
    #define SCHEDULE_PREFETCH_WEIGHTS false
#endif

#define X_NB_BLOCKS N/X_BLOCKING
#define Y_NB_BLOCKS N/Y_BLOCKING

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 1

#define NB_TESTS 101

#ifdef __cplusplus
double median(std::vector<double> scores)
{
    double median;
    size_t size = scores.size();

    sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
    }
    else
    {
        median = scores[size / 2];
    }

    return median;
}
#else
double median(int n, double x[])
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
#endif

double rtclock()
{
    struct timeval Tp;
    gettimeofday(&Tp, NULL);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

#endif
