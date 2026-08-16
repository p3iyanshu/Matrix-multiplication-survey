# 2. Strassen's Algorithm

| Field | Details |
|---|---|
| **Author(s)** | Volker Strassen |
| **Year** | 1969 |
| **Time Complexity** | O(n^log2(7)) ~ O(n^2.807) |
| **Approach** | Serial |
| **Original Paper Link** | https://doi.org/10.1007/BF02165411 (Springer -- Numerische Mathematik, 13(4), 354-356) |

## Pseudocode

```
FUNCTION Strassen(A, B):                    // A, B are n x n, n a power of 2
    IF n == 1:
        RETURN A * B                            // scalar multiplication
    Partition A, B into four (n/2) x (n/2) quadrants: A11,A12,A21,A22 / B11,B12,B21,B22

    M1 = Strassen(A11+A22, B11+B22)
    M2 = Strassen(A21+A22, B11)
    M3 = Strassen(A11, B12-B22)
    M4 = Strassen(A22, B21-B11)
    M5 = Strassen(A11+A12, B22)
    M6 = Strassen(A21-A11, B11+B12)
    M7 = Strassen(A12-A22, B21+B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    RETURN C assembled from quadrants [C11 C12; C21 C22]
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   2. STRASSEN'S ALGORITHM â O(n^log2(7)) ~ O(n^2.807)
   Requires n to be a power of 2. Falls back to classical multiplication
   below a base-case threshold (standard practical optimization, since
   Strassen's recursive overhead outweighs its FLOP savings for small n).
   ===================================================================== */
#define STRASSEN_BASE_CASE 64

static double** strassen_recursive(double** A, double** B, int n) {
    double** C = alloc_matrix(n);

    if (n <= STRASSEN_BASE_CASE) {
        classical_matmul(A, B, C, n);
        return C;
    }

    int half = n / 2;

    /* Allocate quadrant sub-matrices */
    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);

    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }

    double** S1 = matrix_add(A11, A22, half);
    double** T1 = matrix_add(B11, B22, half);
    double** M1 = strassen_recursive(S1, T1, half);

    double** S2 = matrix_add(A21, A22, half);
    double** M2 = strassen_recursive(S2, B11, half);

    double** T3 = matrix_sub(B12, B22, half);
    double** M3 = strassen_recursive(A11, T3, half);

    double** T4 = matrix_sub(B21, B11, half);
    double** M4 = strassen_recursive(A22, T4, half);

    double** S5 = matrix_add(A11, A12, half);
    double** M5 = strassen_recursive(S5, B22, half);

    double** S6 = matrix_sub(A21, A11, half);
    double** T6 = matrix_add(B11, B12, half);
    double** M6 = strassen_recursive(S6, T6, half);

    double** S7 = matrix_sub(A12, A22, half);
    double** T7 = matrix_add(B21, B22, half);
    double** M7 = strassen_recursive(S7, T7, half);

    /* C11 = M1 + M4 - M5 + M7 ; C12 = M3 + M5
       C21 = M2 + M4           ; C22 = M1 - M2 + M3 + M6 */
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C[i][j]               = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + half]        = M3[i][j] + M5[i][j];
            C[i + half][j]        = M2[i][j] + M4[i][j];
            C[i + half][j + half] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    /* Cleanup */
    double** temps[] = {S1, T1, M1, S2, M2, T3, M3, T4, M4, S5, M5, S6, T6, M6, S7, T7, M7};
    for (int t = 0; t < 17; t++) free_matrix(temps[t], half);
    free_matrix(A11, half); free_matrix(A12, half); free_matrix(A21, half); free_matrix(A22, half);
    free_matrix(B11, half); free_matrix(B12, half); free_matrix(B21, half); free_matrix(B22, half);

    return C;
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

double** strassen_matmul(double** A, double** B, int n) {
    int padded = next_pow2(n);
    if (padded == n) {
        return strassen_recursive(A, B, n);
    }
    /* Pad to next power of two with zeros, run, then the caller should
       only read the top-left n x n block of the result. */
    double** Apad = alloc_matrix(padded);
    double** Bpad = alloc_matrix(padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Apad[i][j] = A[i][j]; Bpad[i][j] = B[i][j]; }
    double** Cpad = strassen_recursive(Apad, Bpad, padded);
    double** C = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cpad[i][j];
    free_matrix(Apad, padded); free_matrix(Bpad, padded); free_matrix(Cpad, padded);
    return C;
}

```

---
[<- Back to index](../README.md)
