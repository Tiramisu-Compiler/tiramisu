#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#define C12	0
#define C11	0
#define C10	0
#define C9	0
#define C8	0
#define C7	0
#define C6	0
#define C5	0
#define C4	0
#define C3	0
#define C2	0
#define C1	1

#if C12
	#define BATCH_SIZE 7
    #define N 7
    #define FIn 512
    #define FOut 512
    #define KERNEL 3
    #define STRIDES 1

#elif C11
	#define BATCH_SIZE 14
    #define N 14
    #define FIn 256
    #define FOut 512
    #define KERNEL 3
    #define STRIDES 1

#elif C10
	#define BATCH_SIZE 14
    #define N 14
    #define FIn 256
    #define FOut 512
    #define KERNEL 3
    #define STRIDES 2

#elif C9
	#define BATCH_SIZE 14
    #define N 14
    #define FIn 256
    #define FOut 256
    #define KERNEL 3
    #define STRIDES 1
  
#elif C8
	#define BATCH_SIZE 28
    #define N 28
    #define FIn 128
    #define FOut 256
    #define KERNEL 3
    #define STRIDES 2

#elif C7
	#define BATCH_SIZE 28
    #define N 28
    #define FIn 128
    #define FOut 256
    #define KERNEL 3
    #define STRIDES 2

#elif C6
	#define BATCH_SIZE 28
    #define N 28
    #define FIn 128
    #define FOut 128
    #define KERNEL 3
    #define STRIDES 1

#elif C5
	#define BATCH_SIZE 56
    #define N 56
    #define FIn 64
    #define FOut 128
    #define KERNEL 2
    #define STRIDES 1

#elif C4
	#define BATCH_SIZE 56
    #define N 56
    #define FIn 64
    #define FOut 128
    #define KERNEL 3
    #define STRIDES 2

#elif C3
	#define BATCH_SIZE 100
    #define N 152
    #define FIn 32
    #define FOut 64
    #define STRIDES 1
    #define KERNEL 3

#elif C2
	#define BATCH_SIZE 32
    #define N 64
    #define FIn 16
    #define FOut 32 
    #define STRIDES 1
    #define KERNEL 3

#elif C1
	#define BATCH_SIZE 8
    #define N 32
    #define FIn 3
    #define FOut 64
    #define STRIDES 1
    #define KERNEL 3
    
#endif

#define PADDING 1

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 100

#define NB_TESTS 100

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
