#include <cstdint>
#include <mkl.h>
#include <mkl_spblas.h>
#include <stdio.h>
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

/**
 * This is a reduced interface for the MKL SPMV API.
 * It multiplies a sparse matrix by a dense matrix.
 * This version is custotmized for LSTM
 */
extern "C"
uint8_t tiramisu_spmv(bool transposeA,
                      float alpha,
                      float* csrA, // Sparse matrix handle
                      float* descrA, // Sparse matrix descriptor
                      int layer_num,
                      int weight_type,
                      float* B, // vector
                      float beta,
                      float* C, // vector
                      int offsetB, int offsetC
                    )
{

    sparse_operation_t transposeA_mkl = transposeA ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
    // Get the sparse matrix's handle prealably created
    sparse_matrix_t** csrA_mkl = (sparse_matrix_t**)csrA;
    // Get the sparse matrix's descriptor prealably created
    struct matrix_descr** descrA_mkl = (matrix_descr**)descrA;

    mkl_sparse_s_mv(
                transposeA_mkl,
                alpha,
                *(csrA_mkl[layer_num * 2 + weight_type]),
                *(descrA_mkl[layer_num * 2 + weight_type]),
                B + offsetB,
                beta,
                C + offsetC
    );
    return 0;
}
