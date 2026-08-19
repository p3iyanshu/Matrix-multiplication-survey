#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul_algorithms.h"

double** alloc_matrix(int n) {
    double** M = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        M[i] = (double*)calloc(n, sizeof(double));
    }
    return M;
}

void free_matrix(double** M, int n) {
    if (!M) return;
    for (int i = 0; i < n; i++) free(M[i]);
    free(M);
}

void fill_random(double** M, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = (double)(rand() % 10);
}

void fill_zero(double** M, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = 0.0;
}

void copy_matrix(double** dst, double** src, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[i][j] = src[i][j];
}

void print_matrix(double** M, int n, const char* label) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%8.2f ", M[i][j]);
        printf("\n");
    }
}

int matrices_equal(double** A, double** B, int n, double tol) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (fabs(A[i][j] - B[i][j]) > tol) return 0;
    return 1;
}

double** matrix_add(double** A, double** B, int n) {
    double** R = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A[i][j] + B[i][j];
    return R;
}

double** matrix_sub(double** A, double** B, int n) {
    double** R = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A[i][j] - B[i][j];
    return R;
}
