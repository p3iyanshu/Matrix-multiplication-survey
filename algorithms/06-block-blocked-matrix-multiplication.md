# 6. Block (Blocked) Matrix Multiplication

| Field | Details |
|---|---|
| **Author(s)** | Not attributable to a single author -- classical HPC/numerical linear algebra technique |
| **Year** | Not Verified -- technique matured alongside Level-3 BLAS design (late 1970s-1990) |
| **Time Complexity** | O(n^3) -- same asymptotic complexity as classical MM, much better constant factor |
| **Approach** | Serial |
| **Original Paper Link** | https://doi.org/10.1145/77626.79170 (ACM -- Dongarra, Du Croz, Hammarling & Duff, 'A Set of Level 3 Basic Linear Algebra Subprograms,' 1990) |

## Pseudocode

```
FUNCTION BlockMatMul(A, B, b):               // b = block size fitting in cache
    Initialize C as an n x n matrix of zeros
    FOR ii = 1 TO n STEP b:
      FOR jj = 1 TO n STEP b:
        FOR kk = 1 TO n STEP b:
          FOR i = ii TO min(ii+b-1, n):
            FOR j = jj TO min(jj+b-1, n):
              sum = 0
              FOR k = kk TO min(kk+b-1, n):
                sum = sum + A[i][k] * B[k][j]
              C[i][j] = C[i][j] + sum
    RETURN C
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   6. BLOCK (BLOCKED) MATRIX MULTIPLICATION â cache-tiled, O(n^3)
   ===================================================================== */
void block_matmul(double** A, double** B, double** C, int n, int block_size) {
    fill_zero(C, n);
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                int i_max = (ii + block_size < n) ? ii + block_size : n;
                int j_max = (jj + block_size < n) ? jj + block_size : n;
                int k_max = (kk + block_size < n) ? kk + block_size : n;
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

```

---
[<- Back to index](../README.md)
