# 1. Classical (Naive) Matrix Multiplication

| Field | Details |
|---|---|
| **Author(s)** | N/A -- standard algebraic definition, not attributable to a single author |
| **Year** | Not Verified -- no single original paper; standard definition from linear algebra |
| **Time Complexity** | O(n^3) for square matrices; O(m*n*p) for general rectangular matrices |
| **Approach** | Serial |
| **Original Paper Link** | Not Applicable -- see Golub & Van Loan, *Matrix Computations*, Johns Hopkins University Press |

## Pseudocode

```
FUNCTION ClassicalMatMul(A, B):            // A is m x n, B is n x p
    Initialize C as an m x p matrix of zeros
    FOR i = 1 TO m:
        FOR j = 1 TO p:
            sum = 0
            FOR k = 1 TO n:
                sum = sum + A[i][k] * B[k][j]
            C[i][j] = sum
    RETURN C
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   1. CLASSICAL (NAIVE) MATRIX MULTIPLICATION â O(n^3)
   ===================================================================== */
void classical_matmul(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

```

---
[<- Back to index](../README.md)
