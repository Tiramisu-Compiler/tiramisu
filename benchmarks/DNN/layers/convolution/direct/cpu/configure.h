#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#define LARGE_DATA_SET	1
#define MEDIUM_DATA_SET	0
#define SMALL_DATA_SET	0
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

// Size of convolution filter (KxK)
#define K 5

// If this is defined, print 10 array elements only
#define PRINT_ONLY_10 1

#define NB_TESTS 1

// Maxilam size for the sizes[][] array.
#define NB_MAX_SIZES 100

void fill_sizes_array(int sizes[NB_MAX_SIZES][4], int &nb_sizes)
{
	// N
    	// BATCH_SIZE
    	// FIn
    	// FOut

	nb_sizes = 0;

	if (LARGE_DATA_SET)
	{
		sizes[nb_sizes][0] = 512;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 16;
		sizes[nb_sizes][3] = 16;
		nb_sizes++;
	}

	if (MEDIUM_DATA_SET)
	{
		sizes[nb_sizes][0] = 64;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 16;
		sizes[nb_sizes][3] = 16;
		nb_sizes++;
	}

	if (SMALL_DATA_SET)
	{
		sizes[nb_sizes][0] = 32;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 16;
		sizes[nb_sizes][3] = 16;
		nb_sizes++;
	}

	if (C11)
	{
		sizes[nb_sizes][0] = 224;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 3;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C12)
	{
		sizes[nb_sizes][0] = 224;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 3;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C13)
	{
		sizes[nb_sizes][0] = 224;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 3;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C21)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C22)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C23)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 64;
		nb_sizes++;
	}

	if (C41)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C42)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C43)
	{
		sizes[nb_sizes][0] = 56;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 64;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C61)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 128;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C62)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 128;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C63)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 128;
		sizes[nb_sizes][3] = 128;
		nb_sizes++;
	}

	if (C71)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 100;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C72)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 100;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C73)
	{
		sizes[nb_sizes][0] = 28;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 100;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C91)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 256;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C92)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 256;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C93)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 256;
		sizes[nb_sizes][3] = 256;
		nb_sizes++;
	}

	if (C101)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 310;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}

	if (C102)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 310;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}

	if (C103)
	{
		sizes[nb_sizes][0] = 14;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 310;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}

	if (C121)
	{
		sizes[nb_sizes][0] = 7;
		sizes[nb_sizes][1] = 8;
		sizes[nb_sizes][2] = 512;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}

	if (C122)
	{
		sizes[nb_sizes][0] = 7;
		sizes[nb_sizes][1] = 32;
		sizes[nb_sizes][2] = 512;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}

	if (C123)
	{
		sizes[nb_sizes][0] = 7;
		sizes[nb_sizes][1] = 100;
		sizes[nb_sizes][2] = 512;
		sizes[nb_sizes][3] = 512;
		nb_sizes++;
	}
}

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
