#ifndef __SPCONV_CONF_HEADER_
#define __SPCONV_CONF_HEADER_

#include <sys/time.h>

#define IMPORT_CSR_FROM_FILE 0
#define SHOW_OUTPUT 0
#define CHECK_CORRECTNESS 1 // If set to 1, MKLDNN will write results to a file, and the tiramisu wrapper will compare the outputs

#define PAD_OUTPUT 1 // Needed to make ResNet End to End. When set to 1, MKLDNN doesn't add the padding, we just compare results without padding
#define STRIDE2_FORMATTED_OUTPUT 0

#define LARGE_DATA_SET	0
#define MEDIUM_DATA_SET	1
#define SMALL_DATA_SET	0

#define LARGE_N 0
#define MEDIUM_N 0
#define SMALL_N 1

#define NO_BATCH 0
#if NO_BATCH
	#define BATCH_SIZE 1
#else
	#if LARGE_DATA_SET
		#define BATCH_SIZE 100
	#elif MEDIUM_DATA_SET
		#define BATCH_SIZE 32
	#elif SMALL_DATA_SET
		#define BATCH_SIZE 8
	#endif
#endif

// Size of one data dimension
#if LARGE_N
	#define N 224
#elif MEDIUM_N
	#define N 112
#elif SMALL_N
	#define N 56
#endif

// Number of features in the input
#define FIn 16
#define FIN_BL 2 // Used only in the case where IMPORT_CSR_FROM_FILE = 0, to block FIN when IMPORT_CSR_FROM_FILE = 1, you need to do it through the python script

// Number of features in the output
#define FOut 32
#if NO_BATCH
	#define FOUT_BL 8
#else
	#define FOUT_BL 16
#endif
#define FOUT_NB_BL (FOut / FOUT_BL)

#if NO_BATCH
	#define FOUT_BL2 8
#else
	#define FOUT_BL2 16
#endif
#define FOUT_NB_BL2 (FOut / FOUT_BL2)

// Size of convolution filter (KxK)
#define K 3
#define STRIDE 2
#define STRIDE2_INPUT_TRANSFORMATION 0 // Has to be set to 0 for better performance in the E2E ResNet. This ResNet block is used in resnet End to End architecture

#define WEIGHTS_DENSITY 0.2682
#define WEIGHTS_DENSITY2 0.2027

#define NB_TESTS 101

// Parameters to tune
#define X_BL 28 // Must be a multiple of N
#define X_NB_BL (N/STRIDE/X_BL)

#define Y_BL 2 // Must be a multiple of N
#define Y_NB_BL (N/STRIDE/Y_BL)

#define X_BL2 28 // Must be a multiple of N
#define X_NB_BL2 (N/STRIDE/X_BL2)

#define Y_BL2 2 // Must be a multiple of N
#define Y_NB_BL2 (N/STRIDE/Y_BL2)

#define EPSILON 1e-05

// Parameters for MKL Sparse's IM2COL,
#define H_BL 32 // Must be a divisor of N
#define H_NB_BL N/H_BL
#define W_BL 32 // Must be a divisor of N
#define W_NB_BL N/W_BL

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
