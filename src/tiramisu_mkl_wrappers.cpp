#include <cstdint>
#include <mkl.h>

/**
 * This is a reduced interface for the MKL GEMM API.
 * It multiplies row-major matrices A and B of size MxK and KxN.
 */
extern "C"
uint8_t tiramisu_cblas_sgemm(float *A, float *B, float *C,
                             int M, int N, int K,
                             float alpha, float beta,
                             int ldA, int ldB, int ldC,
                             int offsetA, int offsetB, int offsetC,
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
                M, N, K, alpha, A + offsetA, ldA, B + offsetB, ldB, beta, C + offsetC, ldC);
    return 0;
}

/**
 * This is a reduced interface for the MKL GEMM API.
 * It multiplies row-major matrices A and B of size MxK and KxN.
 */
extern "C"
uint8_t tiramisu_cblas_dgemm(double *A, double *B, double *C,
                             int M, int N, int K,
                             double alpha, double beta,
                             int ldA, int ldB, int ldC,
                             int offsetA, int offsetB, int offsetC,
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
                M, N, K, alpha, A + offsetA, ldA, B + offsetB, ldB, beta, C + offsetC, ldC);
    return 0;
}