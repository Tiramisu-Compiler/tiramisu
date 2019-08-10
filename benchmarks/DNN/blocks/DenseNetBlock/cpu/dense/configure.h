#ifndef __DENSENET_BLOCK_CONF_HEADER_
#define __DENSENET_BLOCK_CONF_HEADER_

#include <sys/time.h>

#define LARGE_DATA_SET	0
#define MEDIUM_DATA_SET	1
#define SMALL_DATA_SET	0

#if LARGE_DATA_SET
    #define BATCH_SIZE 100
#elif MEDIUM_DATA_SET
    #define BATCH_SIZE 32
#elif SMALL_DATA_SET
    #define BATCH_SIZE 8
#endif

// Growth Rate of the block (see the original DenseNet paper for a definition)
// This block receives an input tensor of size NxNx4*GR and outputs a tensor of size NxNxGR
#define GR 32

// Width and height of an input tensor
#define BLOCK_NUMBER 1

#if BLOCK_NUMBER == 0
    #define N 112
#elif BLOCK_NUMBER == 1
    #define N 56
#elif BLOCK_NUMBER == 2
    #define N 28
#elif BLOCK_NUMBER == 3
    #define N 14
#elif BLOCK_NUMBER == 4
    #define N 7
#endif

// Convolution kernel size
#define K_X 3 
#define K_Y 3

#define EPSILON 1e-05

// Parameters for Tiramisu code
#define FOUT_BLOCKING 8
#define FOUT_NB_BLOCKS GR/FOUT_BLOCKING

#define FIN_BLOCKING 8
#define FIN_NB_BLOCKS (4*GR)/FIN_BLOCKING

#define X_BLOCKING 3
#define X_NB_BLOCKS N/X_BLOCKING

#define X_BOUND X_NB_BLOCKS*X_BLOCKING

#define VEC_LEN 8

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 0

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

    // The following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // Swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // If there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        // Else return the element in the middle
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
