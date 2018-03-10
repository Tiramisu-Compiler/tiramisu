#include "generated_spmv.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#define DATATYPE double

/* Fill an array with random values.
 */
void fillArray(int n, DATATYPE *array) {
    for (int i = 0; i < n; i++) {
        array[i] = (DATATYPE)(10.0 * rand() / (RAND_MAX + 1.0f));
    }
}

// Original version by: Kyle Spafford
void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;
    int i,j;
    for (i = 0; i < dim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = 1;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}

/* Generate a square matrix in CSR format with the given number of rows and 1% nonzero entries.
 */
void generateCsrMatrix(int nRows, int *nNonzero, int **rowDelims, int **columns, DATATYPE **values) {
    *nNonzero = nRows * nRows / 100;    // As with SHOC, make 1% of matrix entries nonzero
    *values = (DATATYPE *) malloc(*nNonzero * sizeof(DATATYPE));
    *rowDelims = (int *) malloc((nRows+1) * sizeof(int));
    *columns = (int *) malloc(*nNonzero * sizeof(int));
    fillArray(*nNonzero, *values);
    initRandomMatrix(*columns, *rowDelims, *nNonzero, nRows);
}

void spmv_CSR(int nRows, int nNonzero,
        int *rowDelims,
        int *columns,
        double *mat,
        double *vec,
        double *out)
{
    for (int i = 0; i < nRows; i++) {
        int start = rowDelims[i];
        int end = rowDelims[i+1];
        double accum = 0.0;

        for (int j = start; j < end; j++) {
            int col = columns[j];

            accum += mat[j] * vec[col];
        }
        out[i] = accum;
    }
}

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    bool run_ref = false;
    bool run_tiramisu = false;

    const char* env_ref = std::getenv("RUN_MKL");
    if ((env_ref != NULL) && (env_ref[0] == '1'))
	run_ref = true;
    const char* env_tira = std::getenv("RUN_TIRAMISU");
    if ((env_tira != NULL) && (env_tira[0] == '1'))
	run_tiramisu = true;

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    int *rowDelims;
    int *columns;
    double *values;
    int nNonzero;
    generateCsrMatrix(N, &nNonzero, &rowDelims, &columns, &values);

    std::vector<int> sz = {N};
    std::vector<int> sz2 = {N*N/100};

    Halide::Buffer<int> b_row_start_buf(rowDelims, sz);
    Halide::Buffer<int> b_col_idx_buf(columns, sz);
    Halide::Buffer<double> b_values_buf(values, sz2);
    Halide::Buffer<double> b_x_buf(N);
    init_buffer(b_x_buf, (double) 1);
    Halide::Buffer<double> b_y_buf(N);
    init_buffer(b_y_buf, (double) 0);
    Halide::Buffer<double> b_y_ref_buf(N);
    init_buffer(b_y_ref_buf, (double) 0);

    // Calling HPCCG spmv
    {
        for (int i = 0; i < NB_TESTS; i++)
	{
	    init_buffer(b_y_ref_buf, (double)0);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    if (run_ref == true)
	    	spmv_CSR(N, nNonzero, b_row_start_buf.data(), b_col_idx_buf.data(),
			 b_values_buf.data(), b_x_buf.data(), b_y_ref_buf.data());
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
	}
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(b_y_buf, (double)0);
	    auto start2 = std::chrono::high_resolution_clock::now();
 	    if (run_tiramisu == true)
    		spmv(b_row_start_buf.raw_buffer(), b_col_idx_buf.raw_buffer(), b_values_buf.raw_buffer(), b_x_buf.raw_buffer(), b_y_buf.raw_buffer());
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "sgemm",
               {"Ref", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
 	if (run_ref == 1 && run_tiramisu == 1)
	{
		compare_buffers("sgemm", b_y_buf, b_y_ref_buf);
        }

    if (PRINT_OUTPUT)
    {
	std::cout << "Tiramisu " << std::endl;
	print_buffer(b_y_buf);
	std::cout << "Reference " << std::endl;
	print_buffer(b_y_ref_buf);
    }

    return 0;
}
