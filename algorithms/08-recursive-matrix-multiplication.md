# 8. Recursive Matrix Multiplication (Plain Divide-and-Conquer)

| Field | Details |
|---|---|
| **Author(s)** | Not attributable to one author for the plain scheme |
| **Year** | Not Verified -- concept predates a single canonical paper; rigorously formalized in 1999 (see Cache-Oblivious MM) |
| **Time Complexity** | O(n^3) without Strassen's trick; O(n^2.807) if Strassen's 7-multiplication trick is layered on top |
| **Approach** | Serial |
| **Original Paper Link** | Nearest rigorous treatment: https://doi.org/10.1109/SFFCS.1999.814600 (IEEE -- Frigo, Leiserson, Prokop & Ramachandran, FOCS 1999) |

## Pseudocode

```
FUNCTION RecursiveMatMul(A, B):              // n x n, n a power of 2
    IF n == 1:
        RETURN A * B
    Partition A, B into four (n/2) x (n/2) quadrants each
    C11 = RecursiveMatMul(A11,B11) + RecursiveMatMul(A12,B21)
    C12 = RecursiveMatMul(A11,B12) + RecursiveMatMul(A12,B22)
    C21 = RecursiveMatMul(A21,B11) + RecursiveMatMul(A22,B21)
    C22 = RecursiveMatMul(A21,B12) + RecursiveMatMul(A22,B22)
    RETURN C assembled from quadrants
```

## CPU (C) Implementation

Full compilable source: [`src/algo_07_to_09.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   8. RECURSIVE MATRIX MULTIPLICATION â plain divide-and-conquer,
   NO Strassen trick (8 recursive multiplications, not 7), O(n^3).
   Requires n to be a power of 2 for clean quadrant splitting; the
   wrapper below pads non-power-of-2 sizes with zeros.
   ===================================================================== */
#define RECURSIVE_BASE_CASE 64

static void recursive_matmul_inplace(double** A, double** B, double** C,
                                      int ra, int ca, int rb, int cb, int rc, int cc,
                                      int n, int lda, int ldb, int ldc) {
    if (n <= RECURSIVE_BASE_CASE) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = C[rc+i][cc+j];
                for (int k = 0; k < n; k++)
                    sum += A[ra+i][ca+k] * B[rb+k][cb+j];
                C[rc+i][cc+j] = sum;
            }
        }
        return;
    }
    int h = n / 2;
    /* C11 += A11*B11 + A12*B21 ; C12 += A11*B12 + A12*B22
       C21 += A21*B11 + A22*B21 ; C22 += A21*B12 + A22*B22 */
    recursive_matmul_inplace(A, B, C, ra,   ca,   rb,   cb,   rc,   cc,   h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra,   ca+h, rb+h, cb,   rc,   cc,   h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra,   ca,   rb,   cb+h, rc,   cc+h, h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra,   ca+h, rb+h, cb+h, rc,   cc+h, h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra+h, ca,   rb,   cb,   rc+h, cc,   h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra+h, ca+h, rb+h, cb,   rc+h, cc,   h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra+h, ca,   rb,   cb+h, rc+h, cc+h, h, lda, ldb, ldc);
    recursive_matmul_inplace(A, B, C, ra+h, ca+h, rb+h, cb+h, rc+h, cc+h, h, lda, ldb, ldc);
}

static int next_pow2_local(int n) { int p = 1; while (p < n) p <<= 1; return p; }

void recursive_matmul(double** A, double** B, double** C, int n) {
    int padded = next_pow2_local(n);
    if (padded == n) {
        fill_zero(C, n);
        recursive_matmul_inplace(A, B, C, 0,0,0,0,0,0, n, n, n, n);
        return;
    }
    double** Apad = alloc_matrix(padded);
    double** Bpad = alloc_matrix(padded);
    double** Cpad = alloc_matrix(padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Apad[i][j] = A[i][j]; Bpad[i][j] = B[i][j]; }
    recursive_matmul_inplace(Apad, Bpad, Cpad, 0,0,0,0,0,0, padded, padded, padded, padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cpad[i][j];
    free_matrix(Apad, padded); free_matrix(Bpad, padded); free_matrix(Cpad, padded);
}

```

---
[<- Back to index](../README.md)
