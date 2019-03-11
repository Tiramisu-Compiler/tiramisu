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

    const char* env_mkl = std::getenv("RUN_REF");
    if ((env_mkl != NULL) && (env_mkl[0] == '1'))
	run_mkl = true;
    const char* env_tira = std::getenv("RUN_TIRAMISU");
    if ((env_tira != NULL) && (env_tira[0] == '1'))
	run_tiramisu = true;
#else
    bool run_mkl = true;
    bool run_tiramisu = true;
#endif


	int sizes[27][4], D = 1060 * 1060; 

	/************ SIZE_IS_MULTIPLE_OF_TILE 1 ******************/

	sizes[0][0] = 4096;
	sizes[0][1] = 4096;
	sizes[0][2] = 4096;
	sizes[0][3] = 65536 * 4096;

	sizes[1][0] = 128;
	sizes[1][1] = 128;
	sizes[1][2] = 128;
	sizes[1][3] = 128;

	sizes[2][0] = 512;
	sizes[2][1] = 512;
	sizes[2][2] = 512;
	sizes[2][3] = 1024;

	sizes[3][0] = 1024;
	sizes[3][1] = 1024;
	sizes[3][2] = 1024;
	sizes[3][3] = 1024 * 1024;

	sizes[4][0] = 128;
	sizes[4][1] = 128;
	sizes[4][2] = 256;
	sizes[4][3] = D;

	sizes[5][0] = 2048;
	sizes[5][1] = 2048;
	sizes[5][2] = 2048;
	sizes[5][3] = D;

	sizes[6][0] = 2048;
	sizes[6][1] = 2048;
	sizes[6][2] = 1024;
	sizes[6][3] = D;

	sizes[7][0] = 1024;
	sizes[7][1] = 1024;
	sizes[7][2] = 2048;
	sizes[7][3] = D;

	sizes[8][0] = 1024;
	sizes[8][1] = 1024;
	sizes[8][2] = 512;
	sizes[8][3] = D;

	sizes[9][0] = 512;
	sizes[9][1] = 512;
	sizes[9][2] = 1024;
	sizes[9][3] = D;

	sizes[10][0] = 512;
	sizes[10][1] = 512;
	sizes[10][2] = 256;
	sizes[10][3] = D;

	sizes[11][0] = 128;
	sizes[11][1] = 128;
	sizes[11][2] = 2048;
	sizes[11][3] = D;

	sizes[12][0] = 256;
	sizes[12][1] = 256;
	sizes[12][2] = 256;
	sizes[12][3] = D;

	sizes[13][0] = 128;
	sizes[13][1] = 128;
	sizes[13][2] = 512;
	sizes[13][3] = D;

	sizes[14][0] = 128;
	sizes[14][1] = 128;
	sizes[14][2] = 1024;
	sizes[14][3] = D;

	sizes[15][0] = 128;
	sizes[15][1] = 128;
	sizes[15][2] = 64;
	sizes[15][3] = D;

	/**************  SIZE_IS_MULTIPLE_OF_TILE 0  ****************/

	sizes[16][0] = 1060;
	sizes[16][1] = 1060;
	sizes[16][2] = 1060;
	sizes[16][3] = D;

	sizes[17][0] = 4;
	sizes[17][1] = 4;
	sizes[17][2] = 4;
	sizes[17][3] = D;

	sizes[18][0] = 8;
	sizes[18][1] = 8;
	sizes[18][2] = 4;
	sizes[18][3] = D;

	sizes[19][0] = 16;
	sizes[19][1] = 16;
	sizes[19][2] = 4;
	sizes[19][3] = D;

	sizes[20][0] = 16;
	sizes[20][1] = 16;
	sizes[20][2] = 16;
	sizes[20][3] = D;

	sizes[21][0] = 8;
	sizes[21][1] = 8;
	sizes[21][2] = 16;
	sizes[21][3] = D;

	int p, q;

	if (SIZE_IS_MULTIPLE_OF_TILE)
	{
		p = 0;
		q = 16;
	}
	else
	{
		p = 16;
		q = 22;
	}

	for (int j = p; j < q; j++)
	{

	int local_N = sizes[j][0];
	int local_M = sizes[j][1];
	int local_K = sizes[j][2];
	int local_size = sizes[j][3];

	Halide::Buffer<int> SIZES(3, "SIZES");
	Halide::Buffer<float> alpha(1, "alpha");
	Halide::Buffer<float> beta(1, "beta");
	Halide::Buffer<float> A(local_N, local_K, "A");
	Halide::Buffer<float> B(local_K, local_M, "B");
	Halide::Buffer<float> C(local_N, local_M, "C");
	Halide::Buffer<float> C_mkl(local_N, local_M, "C_mkl");

	SIZES(0) = local_N; SIZES(1) = local_M; SIZES(2) = local_K;
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
		long long int ma, na, mb, nb;

		if( transA == CblasNoTrans ) {
			rmaxa = local_M + 1;
			cmaxa = local_K;
			ma    = local_M;
			na    = local_K;
		} else {
			rmaxa = local_K + 1;
			cmaxa = local_M;
			ma    = local_K;
			na    = local_M;
		}
		if( transB == CblasNoTrans ) {
			rmaxb = local_K + 1;
			cmaxb = local_N;
			mb    = local_K;
			nb    = local_N;
		} else {
			rmaxb = local_N + 1;
			cmaxb = local_K;
			mb    = local_N;
			nb    = local_K;
		}
		rmaxc = local_M + 1;
		cmaxc = local_N;
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
	    	cblas_sgemm(layout, transA, transB, local_M, local_N, local_K, a, (float *) A.raw_buffer()->host, lda, (float *) B.raw_buffer()->host, ldb, b, (float *) C_mkl.raw_buffer()->host, ldc);
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
    }
    return 0;
}
