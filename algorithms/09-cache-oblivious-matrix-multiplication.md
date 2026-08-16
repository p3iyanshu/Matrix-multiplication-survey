# 9. Cache-Oblivious Matrix Multiplication

| Field | Details |
|---|---|
| **Author(s)** | Matteo Frigo, Charles E. Leiserson, Harald Prokop, Sridhar Ramachandran |
| **Year** | 1999 (FOCS conference); 2012 (extended ACM TALG journal version) |
| **Time Complexity** | O(n^3) work; asymptotically optimal cache-miss complexity Theta(1 + n^2/L + n^3/(L*sqrt(Z))), Z = cache size, L = cache-line length |
| **Approach** | Serial |
| **Original Paper Link** | Primary: https://doi.org/10.1109/SFFCS.1999.814600 (FOCS 1999) | Extended: https://doi.org/10.1145/2071379.2071383 (ACM Trans. Algorithms, 8(1), 2012) |

## Pseudocode

```
FUNCTION CacheObliviousMatMul(A, B, m, n, p):   // A is m x n, B is n x p
    IF max(m, n, p) <= base-case threshold:
        Compute C = A * B directly (classical triple loop)
        RETURN C
    IF m is the largest dimension:
        Split A horizontally into A1, A2; recurse on each half, concatenate results
    ELSE IF p is the largest dimension:
        Split B vertically into B1, B2; recurse on each half, concatenate results
    ELSE:                                        // n is the largest dimension
        Split A vertically and B horizontally into halves along n
        C = CacheObliviousMatMul(A1,B1,...) + CacheObliviousMatMul(A2,B2,...)
    RETURN C
```

## CPU (C) Implementation

Full compilable source: [`src/algo_07_to_09.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   9. CACHE-OBLIVIOUS MATRIX MULTIPLICATION
   Recursively splits whichever of {m, n, p} is currently largest,
   with NO tuning parameter -- this is what makes it "cache-oblivious"
   (Frigo, Leiserson, Prokop, Ramachandran, FOCS 1999). Operates via
   offsets into full-size backing arrays so no sub-matrix copies are
   needed (unlike the simpler Strassen/recursive implementations above).
   ===================================================================== */
#define CACHE_OBLIVIOUS_BASE 32

void cache_oblivious_matmul(double** A, double** B, double** C,
                             int row_off_a, int col_off_a,
                             int row_off_b, int col_off_b,
                             int row_off_c, int col_off_c,
                             int m, int n, int p, int lda, int ldb, int ldc) {
    (void)lda; (void)ldb; (void)ldc; /* offsets used directly; kept for API clarity */

    if (m <= CACHE_OBLIVIOUS_BASE && n <= CACHE_OBLIVIOUS_BASE && p <= CACHE_OBLIVIOUS_BASE) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = C[row_off_c+i][col_off_c+j];
                for (int k = 0; k < n; k++)
                    sum += A[row_off_a+i][col_off_a+k] * B[row_off_b+k][col_off_b+j];
                C[row_off_c+i][col_off_c+j] = sum;
            }
        }
        return;
    }

    if (m >= n && m >= p) {
        /* split A horizontally (split m) */
        int h = m / 2;
        cache_oblivious_matmul(A, B, C, row_off_a, col_off_a, row_off_b, col_off_b,
                                row_off_c, col_off_c, h, n, p, 0,0,0);
        cache_oblivious_matmul(A, B, C, row_off_a+h, col_off_a, row_off_b, col_off_b,
                                row_off_c+h, col_off_c, m-h, n, p, 0,0,0);
    } else if (p >= m && p >= n) {
        /* split B vertically (split p) */
        int h = p / 2;
        cache_oblivious_matmul(A, B, C, row_off_a, col_off_a, row_off_b, col_off_b,
                                row_off_c, col_off_c, m, n, h, 0,0,0);
        cache_oblivious_matmul(A, B, C, row_off_a, col_off_a, row_off_b, col_off_b+h,
                                row_off_c, col_off_c+h, m, n, p-h, 0,0,0);
    } else {
        /* n is largest: split the shared dimension and accumulate both halves */
        int h = n / 2;
        cache_oblivious_matmul(A, B, C, row_off_a, col_off_a, row_off_b, col_off_b,
                                row_off_c, col_off_c, m, h, p, 0,0,0);
        cache_oblivious_matmul(A, B, C, row_off_a, col_off_a+h, row_off_b+h, col_off_b,
                                row_off_c, col_off_c, m, n-h, p, 0,0,0);
    }
}

void cache_oblivious_matmul_wrapper(double** A, double** B, double** C, int n) {
    fill_zero(C, n);
    cache_oblivious_matmul(A, B, C, 0,0,0,0,0,0, n, n, n, n, n, n);
}

```

---
[<- Back to index](../README.md)
