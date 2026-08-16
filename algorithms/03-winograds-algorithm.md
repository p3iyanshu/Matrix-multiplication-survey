# 3. Winograd's Algorithm (Strassen-Winograd Variant)

| Field | Details |
|---|---|
| **Author(s)** | Shmuel Winograd |
| **Year** | 1971 |
| **Time Complexity** | O(n^2.807) -- same asymptotic class as Strassen; reduces additions from 18 to 15 |
| **Approach** | Serial |
| **Original Paper Link** | https://doi.org/10.1016/0024-3795(71)90009-7 (Elsevier -- Linear Algebra and its Applications, 4(4), 381-388) |

## Pseudocode

```
FUNCTION WinogradBlock2x2(A, B):
    S1 = A21 + A22 ; S2 = S1 - A11 ; S3 = A11 - A21 ; S4 = A12 - S2
    T1 = B12 - B11 ; T2 = B22 - T1 ; T3 = B22 - B12 ; T4 = T2 - B21

    M1 = S2 * T2 ; M2 = A11 * B11 ; M3 = A12 * B21
    M4 = S3 * T3 ; M5 = S1 * T1  ; M6 = S4 * B22  ; M7 = A22 * T4

    U1 = M2 + M3 ; U2 = M2 + M5 ; U3 = U2 + M4

    C11 = U1
    C12 = U3 + M6
    C21 = U3 + M7
    C22 = U2 + M1
    RETURN C   // Same 7 multiplications as Strassen, only 15 add/sub instead of 18
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   3. WINOGRAD'S ALGORITHM (Strassen-Winograd variant)
   Same 7 multiplications as Strassen, but only 15 add/sub instead of 18
   (S1..S4, T1..T4, then combine â matches the paper's reduced-addition form).
   ===================================================================== */
static double** winograd_recursive(double** A, double** B, int n) {
    double** C = alloc_matrix(n);
    if (n <= STRASSEN_BASE_CASE) {
        classical_matmul(A, B, C, n);
        return C;
    }
    int half = n / 2;
    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];             A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];      A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];             B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];      B22[i][j] = B[i + half][j + half];
        }

    /* S1 = A21+A22 ; S2 = S1-A11 ; S3 = A11-A21 ; S4 = A12-S2 */
    double** S1 = matrix_add(A21, A22, half);
    double** S2 = matrix_sub(S1, A11, half);
    double** S3 = matrix_sub(A11, A21, half);
    double** S4 = matrix_sub(A12, S2, half);
    /* T1 = B12-B11 ; T2 = B22-T1 ; T3 = B22-B12 ; T4 = T2-B21 */
    double** T1 = matrix_sub(B12, B11, half);
    double** T2 = matrix_sub(B22, T1, half);
    double** T3 = matrix_sub(B22, B12, half);
    double** T4 = matrix_sub(T2, B21, half);

    double** M1 = winograd_recursive(S2, T2, half);
    double** M2 = winograd_recursive(A11, B11, half);
    double** M3 = winograd_recursive(A12, B21, half);
    double** M4 = winograd_recursive(S3, T3, half);
    double** M5 = winograd_recursive(S1, T1, half);
    double** M6 = winograd_recursive(S4, B22, half);
    double** M7 = winograd_recursive(A22, T4, half);

    /* U1 = M2+M3 ; U2 = M2+M5 ; U3 = U2+M4
       C11 = U1 ; C12 = U3+M6 ; C21 = U3+M7 ; C22 = U2+M1 */
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            double U1 = M2[i][j] + M3[i][j];
            double U2 = M2[i][j] + M5[i][j];
            double U3 = U2 + M4[i][j];
            C[i][j]               = U1;
            C[i][j + half]        = U3 + M6[i][j];
            C[i + half][j]        = U3 + M7[i][j];
            C[i + half][j + half] = U2 + M1[i][j];
        }
    }

    double** temps[] = {S1,S2,S3,S4,T1,T2,T3,T4,M1,M2,M3,M4,M5,M6,M7};
    for (int t = 0; t < 15; t++) free_matrix(temps[t], half);
    free_matrix(A11, half); free_matrix(A12, half); free_matrix(A21, half); free_matrix(A22, half);
    free_matrix(B11, half); free_matrix(B12, half); free_matrix(B21, half); free_matrix(B22, half);
    return C;
}

double** winograd_matmul(double** A, double** B, int n) {
    int padded = next_pow2(n);
    if (padded == n) return winograd_recursive(A, B, n);
    double** Apad = alloc_matrix(padded);
    double** Bpad = alloc_matrix(padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Apad[i][j] = A[i][j]; Bpad[i][j] = B[i][j]; }
    double** Cpad = winograd_recursive(Apad, Bpad, padded);
    double** C = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cpad[i][j];
    free_matrix(Apad, padded); free_matrix(Bpad, padded); free_matrix(Cpad, padded);
    return C;
}

```

---
[<- Back to index](../README.md)
