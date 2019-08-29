#ifndef __RESIZE_CONV_CONF_HEADER_
#define __RESIZE_CONV_CONF_HEADER_

#include <sys/time.h>

#define WRITE_RESULTS_TO_FILE 1
#define CHECK_CORRECTNESS 1

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

// Width and height of an input image
#define IMG_W 600
#define IMG_H 400

// Size of one data dimension
#define N 224

// Number of features in the input
#define FIn 3
// Number of features in the output
#define FOut 32

// Size of convolution filter (KxK)
#define K 3
#define WEIGHTS_DENSITY 0.2

// Parameters for Tiramisu code
#define FOUT_BLOCKING 16
#define FOUT_NB_BLOCKS FOut/FOUT_BLOCKING

#define FIN_BLOCKING 8
#define FIN_NB_BLOCKS FIn/FIN_BLOCKING

// Parameters for MKL Sparse's IM2COL
#define H_BL 32 // Must be a divisor of N
#define H_NB_BL N/H_BL
#define W_BL 32 // Must be a divisor of N
#define W_NB_BL N/W_BL

#if N >= 224
    #define X_BL 8
    #define Y_BL 4
#else
    #define X_BL 4
    #define Y_BL 2
#endif

#define X_NB_BL (N/X_BL)
#define Y_NB_BL (N/Y_BL)

#define NB_TESTS 301

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
