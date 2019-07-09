#ifndef __CONV_RELU_MAXPOOL_CONF_HEADER_
#define __CONV_RELU_MAXPOOL_CONF_HEADER_

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

// Width and height of an input tensor
#define N 56

// Number of features in the input
#define FIn 32
// Number of features in the output
#define FOut 64

// Convolution kernel size
#define K_X 3 
#define K_Y 3

// Parameters for Tiramisu code
#define FIN_BLOCKING 8
#define FOUT_BLOCKING 8

#define FIN_NB_BLOCKS FIn/FIN_BLOCKING
#define FOUT_NB_BLOCKS FOut/FOUT_BLOCKING

#define SCHEDULE_FUSION true

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 0

#define NB_TESTS 1

#ifdef __cplusplus
double median(std::vector<std::chrono::duration<double, std::milli>> scores)
{
    double median;
    size_t size = scores.size();

    sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1].count() + scores[size / 2].count()) / 2;
    }
    else
    {
        median = scores[size / 2].count();
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

#endif