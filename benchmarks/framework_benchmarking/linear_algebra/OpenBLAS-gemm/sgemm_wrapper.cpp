#include "Halide.h"
#include "HalideRuntime.h"
#include <cstdlib>
#include <iostream>

#include <cblas.h>
#include "benchmarks.h"
#include <tiramisu/utils.h>
#include "generated_sgemm_halide.h"

#ifdef __cplusplus
extern "C" {
#endif

//int sgemm_halide(halide_buffer_t *alpha, halide_buffer_t *beta, halide_buffer_t *A, halide_buffer_t *B, halide_buffer_t *C);

#ifdef __cplusplus
}  // extern "C"
#endif


int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    float a = 3, b = 3;

    Halide::Buffer<int> SIZES(3, "SIZES");
    Halide::Buffer<float> alpha(1, "alpha");
    Halide::Buffer<float> beta(1, "beta");
    Halide::Buffer<float> A(N, K, "A");
    Halide::Buffer<float> B(K, M, "B");
    Halide::Buffer<float> C(N, M, "C");
    Halide::Buffer<float> C_openblas(N, M, "C_openblas");

    SIZES(0) = N; SIZES(1) = M; SIZES(2) = K;
    alpha(0) = a; beta(0) = b;
    init_buffer(A, (float)1);
    init_buffer(B, (float)1);
    init_buffer(C, (float)1);
    init_buffer(C_openblas, (float)1);

    // Calling OpenBLAS
    {
	long long int   lda, ldb, ldc;
      	long long int   rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
      	long long int    ma, na, mb, nb;

        rmaxa = M + 1;
        cmaxa = K;
	ma    = M;
	na    = K;

	rmaxb = K + 1;
	cmaxb = N;
	mb    = K;
	nb    = N;

	rmaxc = M + 1;
	cmaxc = N;

	lda=cmaxa;
	ldb=cmaxb;
	ldc=cmaxc;

        for (int i = 0; i < NB_TESTS; i++)
	{
	    init_buffer(C_openblas, (float)1);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, a, (float *) A.raw_buffer()->host, lda, (float *) B.raw_buffer()->host, ldb, b, (float *) C_openblas.raw_buffer()->host, ldc);
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
	}
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(C, (float)1);
	    auto start2 = std::chrono::high_resolution_clock::now();
	    //halide_sgemm_notrans(a, A.raw_buffer(), B.raw_buffer(), b, C.raw_buffer(), C.raw_buffer());
	    //sgemm_halide(alpha.raw_buffer(), beta.raw_buffer(), A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "sgemm",
               {"OpenBLAS", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

//    if (CHECK_CORRECTNESS)
//	compare_buffers("sgemm", C, C_openblas);

    if (PRINT_OUTPUT)
    {
	std::cout << "Halide sgemm " << std::endl;
	print_buffer(C);
	std::cout << "OpenBLAS sgemm " << std::endl;
	print_buffer(C_openblas);
    }

    return 0;
}
