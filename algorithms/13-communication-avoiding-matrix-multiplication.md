# 13. Communication-Avoiding Matrix Multiplication (CA-MM / 2.5D)

| Field | Details |
|---|---|
| **Author(s)** | Grey Ballard, James Demmel, Olga Holtz, Oded Schwartz |
| **Year** | 2011 (builds on Hong & Kung's 1981 communication lower bound) |
| **Time Complexity** | Computation O(n^3/p); communication lower bound Omega(n^2/sqrt(M)) per processor (M = local memory size) |
| **Approach** | Serial (Distributed-Memory) |
| **Original Paper Link** | https://doi.org/10.1137/090769156 (SIAM -- SIAM J. Matrix Anal. Appl., 32(3), 866-901) |

> **Implementation note:** CA-MM is inherently distributed-memory. The implementation below **simulates** the c-layer replication and final reduction step that characterizes the 2.5D approach.

## Pseudocode

```
FUNCTION CA25DMatMul(A, B, c):   // c = replication factor over a base q x q grid
    Partition the sqrt(p/c) x sqrt(p/c) processor grid into c layers
    Distribute A, B blocks across the base 2-D grid; replicate across the c layers
    EACH of the c layers independently computes a partial product using only
        its local 1/c share of the k-dimension (SUMMA- or Cannon-style)
    Perform a reduction (sum) across the c layers to combine partial results
    RETURN C     // achieves the Omega(n^2/sqrt(M)) communication lower bound
```

## CPU (C) Implementation

Full compilable source: [`src/algo_10_to_13.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   13. COMMUNICATION-AVOIDING MATRIX MULTIPLICATION (2.5D)
   The shared (k) dimension is split into c layers; each layer computes
   an independent partial product using only its 1/c share of k, then
   all c partial results are summed (the "reduction" step) to form C.
   This models the 2.5D algorithm's core idea: trade extra memory
   (c-way replication) for reduced communication volume.
   ===================================================================== */
void ca_mm_2_5d_simulated(double** A, double** B, double** C, int n, int q, int c) {
    (void)q; /* base grid dimension retained for interface symmetry/documentation */
    if (n % c != 0) { fprintf(stderr, "[CA-MM 2.5D] n must be divisible by c\n"); return; }
    int kchunk = n / c;
    fill_zero(C, n);

    double** partial = alloc_matrix(n);
    for (int layer = 0; layer < c; layer++) {
        fill_zero(partial, n);
        int k0 = layer * kchunk, k1 = k0 + kchunk;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = k0; k < k1; k++) sum += A[i][k] * B[k][j];
                partial[i][j] = sum;
            }
        }
        /* Reduction across layers: accumulate into C */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                C[i][j] += partial[i][j];
    }
    free_matrix(partial, n);
}

```

---
[<- Back to index](../README.md)
