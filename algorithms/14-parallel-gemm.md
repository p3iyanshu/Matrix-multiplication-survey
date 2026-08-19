# 14. Parallel GEMM (General Category)

| Field | Details |
|---|---|
| **Author(s)** | Not a single algorithm/author -- umbrella term standardized through BLAS/PBLAS/ScaLAPACK |
| **Year** | Not Verified as a single algorithm -- covers Cannon (1969), Fox (1987), SUMMA (1997), PUMMA (1994), and CA-MM (2011) |
| **Time Complexity** | O(n^3/p) computation; exact communication term depends on the underlying algorithm |
| **Approach** | Serial (Distributed-Memory) |
| **Original Paper Link** | Representative reference: Choi, Walker & Dongarra (1994), 'PUMMA: Parallel Universal Matrix Multiplication Algorithms,' Concurrency: Practice and Experience, 6(7), 543-570 |

> **Implementation note:** Parallel GEMM is an umbrella category, not a single algorithm. The implementation below is a dispatcher over the four concrete distributed-memory algorithms above.

## Pseudocode

```
FUNCTION ParallelGEMM(A, B, algorithm):
    SWITCH algorithm:
        CASE Cannon:      RETURN CannonMatMul(A,B)
        CASE Fox:         RETURN FoxMatMul(A,B)
        CASE SUMMA:       RETURN SUMMA(A,B)
        CASE CA-MM(2.5D): RETURN CA25DMatMul(A,B,c)
```

## CPU (C) Implementation

Full compilable source: [`src/algo_14_to_15.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   14. PARALLEL GEMM — general umbrella category, not a single algorithm.
   This dispatcher demonstrates that "Parallel GEMM" simply means picking
   one of the concrete distributed-memory algorithms above (Cannon, Fox,
   SUMMA, or CA-MM/2.5D) as the underlying communication scheme.
   ===================================================================== */
void parallel_gemm(double** A, double** B, double** C, int n, ParallelGemmAlgo algo) {
    int q = 2; /* demo grid dimension; n must be divisible by q (and by q*c for CA-MM) */
    switch (algo) {
        case PGEMM_CANNON:
            cannon_matmul_simulated(A, B, C, n, q);
            break;
        case PGEMM_FOX:
            fox_matmul_simulated(A, B, C, n, q);
            break;
        case PGEMM_SUMMA:
            summa_matmul_simulated(A, B, C, n, q, q);
            break;
        case PGEMM_CA25D:
            ca_mm_2_5d_simulated(A, B, C, n, q, q); /* c = q for this demo */
            break;
        default:
            fprintf(stderr, "[Parallel GEMM] Unknown algorithm selector\n");
    }
}

```

---
[<- Back to index](../README.md)
