#include <cstdint>
#include <mkl.h>

/**
 * This is a reduced interface for the MKL GEMM API.
 * It multiplies row-major matrices A and B of size MxK and KxN.
 */
extern "C"
int tiramisu_cblas_sgemm(float *A, float *B, float *C,
                         uint64_t M, uint64_t N, uint64_t K,
                         float alpha, float beta,
                         uint64_t ldA, uint64_t ldB, uint64_t ldC,
                         uint64_t offsetA, uint64_t offsetB, uint64_t offsetC,
                         bool transposeA, bool transposeB)
{
    // Default values for tight packing:
    if (ldA == 0) {
        ldA = transposeA ? M : K;
    }
    if (ldB == 0) {
        ldB = transposeB ? K : N;
    }
    if (ldC == 0) {
        ldC = N;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, alpha, A+offsetA, K, B+ offsetB, N, beta, C+ offsetC, N);
    return 0;
}

/**
 * This is a reduced interface for the MKL GEMM API.
 * It multiplies row-major matrices A and B of size MxK and KxN.
 */
extern "C"
int tiramisu_cblas_dgemm(double *A, double *B, double *C,
                         uint64_t M, uint64_t N, uint64_t K,
                         double alpha, double beta,
                         uint64_t ldA, uint64_t ldB, uint64_t ldC,
                         uint64_t offsetA, uint64_t offsetB, uint64_t offsetC,
                         bool transposeA, bool transposeB)
{
    // Default values for tight packing:
    if (ldA == 0) {
        ldA = transposeA ? M : K;
    }
    if (ldB == 0) {
        ldB = transposeB ? K : N;
    }
    if (ldC == 0) {
        ldC = N;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, alpha, A+offsetA, K, B+ offsetB, N, beta, C+ offsetC, N);
    return 0;
}