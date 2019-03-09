#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#define LARGE_DATA_SET	0
#define MEDIUM_DATA_SET	0
#define SMALL_DATA_SET	1
#define C11 0
#define C12 0
#define C13 0
#define C21 0
#define C22 0
#define C23 0
#define C41 0
#define C42 0
#define C43 0
#define C61 0
#define C62 0
#define C63 0
#define C71 0
#define C72 0
#define C73 0
#define C91 0
#define C92 0
#define C93 0
#define C101 0
#define C102 0
#define C103 0
#define C121 0
#define C122 0
#define C123 0

#if LARGE_DATA_SET or C13 or C23 or C43 or C63 or C73 or C83 or C93 or C103 or C123
	#define BATCH_SIZE 100
#elif MEDIUM_DATA_SET or C12 or C22 or C42 or C62 or C72 or C82 or C92 +C102 or C122
	#define BATCH_SIZE 32
#elif SMALL_DATA_SET or C11 or C21 or C41 or C61 or C71 or C81 or C91 or C101 or C121
	#define BATCH_SIZE 8 
#endif

// Size of one data dimension

#if LARGE_DATA_SET
	#define N 512
#elif MEDIUM_DATA_SET
	#define N 64
#elif SMALL_DATA_SET
	#define N 32
#endif

#if SMALL_DATA_SET or MEDIUM_DATA_SET or LARGE_DATA_SET
    // Number of features in the input
    #define FIn 16
    // Number of features in the output
    #define FOut 16
#endif

// Size of convolution filter (KxK)
#define K 5

#if C11 or C12 or C13
    #define N 224
    #define FIn 3
    #define FOut 64
#elif C21 or C22 or C23
    #define N 56
    #define FIn 64
    #define FOut 64
#elif C41 or C42 or C43
    #define N 56
    #define FIn 64
    #define FOut 128
#elif C61 or C62 or C63
    #define N 28
    #define FIn 128
    #define FOut 128
#elif C71 or C72 or C73
    #define N 28
    #define FIn 128
    #define FOut 256
#elif C91 or C92 or C93
    #define N 14
    #define FIn 256
    #define FOut 256
#elif C101 or C102 or C103
    #define N 14
    #define FIn 256
    #define FOut 512
#elif C121 or C122 or C123
    #define N 7
    #define FIn 512
    #define FOut 512

#endif

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 1

#define NB_TESTS 10

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

#endif
