# 4. Coppersmith-Winograd Algorithm

| Field | Details |
|---|---|
| **Author(s)** | Don Coppersmith, Shmuel Winograd |
| **Year** | 1990 (journal); 1987 (STOC conference version) |
| **Time Complexity** | O(n^2.376) |
| **Approach** | Serial (theoretical only -- no practical implementation exists) |
| **Original Paper Link** | Primary: https://doi.org/10.1016/S0747-7171(08)80013-2 (Journal of Symbolic Computation, 9(3), 251-280) | Extended: https://doi.org/10.1145/28395.28396 (STOC '87) |

> **Implementation note:** No practical implementation of Coppersmith-Winograd exists anywhere -- it is a **"galactic algorithm"**: the hidden constant factors are so large it never outperforms even classical multiplication for any matrix size that could ever be materialized. Rather than fabricate a fake implementation, the code below documents this fact and safely falls back to Strassen's algorithm (the best *practical* sub-cubic algorithm) so it can still run end-to-end.

## Pseudocode

```
FUNCTION CoppersmithWinograd(A, B):
    Construct a trilinear algebraic identity based on a Salem-Spencer
    arithmetic-progression-free set of integers
    Represent the matrix product implicitly as a tensor T with low
    "border rank" relative to naive multiplication
    Amplify the base identity via repeated tensor (Kronecker) powers
    T^(x)k to drive the effective exponent toward 2.376
    Recover C = A x B from the amplified tensor decomposition
    RETURN C
    // NOT implemented in practice: hidden constants are astronomically large
    // ("galactic algorithm") -- see implementation note below
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   4. COPPERSMITH-WINOGRAD â THEORETICAL ONLY
   No practical implementation exists anywhere (it is a "galactic
   algorithm": the hidden constant factors are so large that it never
   outperforms even classical multiplication for any matrix size that
   could ever be materialized). Per the survey's own findings, we do
   NOT fabricate a fake "CW implementation" â that would misrepresent
   the algorithm. Instead this function documents that fact and safely
   falls back to Strassen (the best PRACTICAL sub-cubic algorithm)
   so the driver program can still run end-to-end.
   ===================================================================== */
double** coppersmith_winograd_matmul(double** A, double** B, int n) {
    fprintf(stderr,
        "[NOTE] Coppersmith-Winograd (1990, exponent 2.376) has no practical "
        "implementation on any hardware -- its hidden constants are "
        "astronomically large ('galactic algorithm'). Falling back to "
        "Strassen's algorithm as the best PRACTICAL sub-cubic substitute.\n");
    return strassen_matmul(A, B, n);
}

```

---
[<- Back to index](../README.md)
