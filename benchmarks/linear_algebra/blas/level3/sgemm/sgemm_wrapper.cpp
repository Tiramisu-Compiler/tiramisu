#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "mkl_cblas.h"

#include "sgemm_wrapper.h"
#include "benchmarks.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif


int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    float a = 3, b = 3;


#if 1
    bool run_mkl = false;
    bool run_tiramisu = false;

    const char* env_mkl = std::getenv("RUN_MKL");
    if ((env_mkl != NULL) && (env_mkl[0] == '1'))
	run_mkl = true;
    const char* env_tira = std::getenv("RUN_TIRAMISU");
    if ((env_tira != NULL) && (env_tira[0] == '1'))
	run_tiramisu = true;
#else
    bool run_mkl = true;
    bool run_tiramisu = true;
#endif

    Halide::Buffer<int> SIZES(3, "SIZES");
    Halide::Buffer<float> alpha(1, "alpha");
    Halide::Buffer<float> beta(1, "beta");
    Halide::Buffer<float> A(N, K, "A");
    Halide::Buffer<float> B(K, M, "B");
    Halide::Buffer<float> C(N, M, "C");
    Halide::Buffer<float> C_mkl(N, M, "C_mkl");

    SIZES(0) = N; SIZES(1) = M; SIZES(2) = K;
    alpha(0) = a; beta(0) = b;
    init_buffer(A, (float)1);
    init_buffer(B, (float)1);
    init_buffer(C, (float)1);
    init_buffer(C_mkl, (float)1);

    // Calling MKL
    {
	long long int   lda, ldb, ldc;
      	long long int   rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
      	CBLAS_LAYOUT    layout = CblasRowMajor;
      	CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasNoTrans;
      	long long int    ma, na, mb, nb;

        if( transA == CblasNoTrans ) {
        	rmaxa = M + 1;
         	cmaxa = K;
		ma    = M;
		na    = K;
	} else {
		rmaxa = K + 1;
		cmaxa = M;
		ma    = K;
		na    = M;
	}
	if( transB == CblasNoTrans ) {
		rmaxb = K + 1;
		cmaxb = N;
		mb    = K;
		nb    = N;
	} else {
		rmaxb = N + 1;
		cmaxb = K;
		mb    = N;
		nb    = K;
	}
	rmaxc = M + 1;
	cmaxc = N;
	if (layout == CblasRowMajor) {
		lda=cmaxa;
		ldb=cmaxb;
		ldc=cmaxc;
	} else {
		lda=rmaxa;
		ldb=rmaxb;
		ldc=rmaxc;
	}

        for (int i = 0; i < NB_TESTS; i++)
	{
	    init_buffer(C_mkl, (float)1);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    if (run_mkl == true)
	    	cblas_sgemm(layout, transA, transB, M, N, K, a, (float *) A.raw_buffer()->host, lda, (float *) B.raw_buffer()->host, ldb, b, (float *) C_mkl.raw_buffer()->host, ldc);
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
	}
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(C, (float)1);
	    auto start2 = std::chrono::high_resolution_clock::now();
 	    if (run_tiramisu == true)
	    	sgemm_tiramisu(SIZES.raw_buffer(), alpha.raw_buffer(), beta.raw_buffer(), A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "sgemm",
               {"MKL", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
 	if (run_mkl == 1 && run_tiramisu == 1)
	{
		compare_buffers("sgemm", C, C_mkl);
        }

    if (PRINT_OUTPUT)
    {
	std::cout << "Tiramisu sgemm " << std::endl;
	print_buffer(C);
	std::cout << "MKL sgemm " << std::endl;
	print_buffer(C_mkl);
    }

    return 0;
}
