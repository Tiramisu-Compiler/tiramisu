#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define M 3072
#define N 3072
#define K 3072

float* new_matrix(int rows, int cols) {
    float* m = (float*)malloc(rows * cols * sizeof(float));
    // Random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i * cols + j] = rand() % 100;
        }
    }
    return m;
}

int clock_t_comparator(const void* a, const void* b) {
    return (*(clock_t*)a > *(clock_t*)b) - (*(clock_t*)a < *(clock_t*)b);
}

void sgemm(float *A, float *B, float *C, float alpha, float beta) {
#pragma scop
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = beta * C[i * N + j];
            for (int k = 0; k < K; k++) {
                C[i * N + j] += alpha * A[i * K + k] * B[k * N + j];
            }
        }
    }
#pragma endscop
}

int main() {
    float *A = new_matrix(M, K);
    float *B = new_matrix(K, N);
    float *C = new_matrix(M, N);
    float alpha = 3;
    float beta = 2;
    // Warm up
    for (int i = 0; i< 10; i++) {
        sgemm(A, B, C, alpha, beta);
    }
    int testN = 100;
    clock_t times[testN];
    for (int i = 0; i < testN; i++) {
        clock_t t = clock();
        sgemm(A, B, C, alpha, beta);
        times[i] = clock() - t;
    }
    qsort(times, testN, sizeof(clock_t), clock_t_comparator);
    printf("median time: %.2fms\n", ((float)times[testN / 2]) / CLOCKS_PER_SEC * 1000);
}
