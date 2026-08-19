# 11. Fox's Algorithm (Broadcast-Multiply-Roll)

| Field | Details |
|---|---|
| **Author(s)** | Geoffrey C. Fox, Steve W. Otto, Anthony J. G. Hey |
| **Year** | 1987 |
| **Time Complexity** | O(n^3/p) computation, with communication cost analyzed for hypercube/mesh topologies |
| **Approach** | Serial (Distributed-Memory) |
| **Original Paper Link** | https://doi.org/10.1016/0167-8191(87)90060-3 (Elsevier -- Parallel Computing, 4(1), 17-31) |

> **Implementation note:** Fox's Algorithm is inherently distributed-memory. The implementation below **simulates** a q x q virtual processor grid, executing the real broadcast-multiply-roll pattern sequentially.

## Pseudocode

```
FUNCTION FoxMatMul(A, B):   // executed identically on each processor P(i,j) in a sqrt(p) x sqrt(p) grid
    C(i,j) = 0
    FOR step = 0 TO sqrt(p)-1:
        bcast_col = (i + step) MOD sqrt(p)
        Processor P(i, bcast_col) broadcasts its A-block along row i
        C(i,j) = C(i,j) + A_broadcast * B_local
        Circularly shift B_local one step up (within processor column)
    RETURN C(i,j)
```

## CPU (C) Implementation

Full compilable source: [`src/algo_10_to_13.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   11. FOX'S ALGORITHM (broadcast-multiply-roll), simulated q x q grid
   ===================================================================== */
void fox_matmul_simulated(double** A, double** B, double** C, int n, int q) {
    if (n % q != 0) { fprintf(stderr, "[Fox] n must be divisible by q\n"); return; }
    int s = n / q;
    fill_zero(C, n);

    double**** Agrid = (double****)malloc(q * sizeof(double***));
    double**** Bgrid = (double****)malloc(q * sizeof(double***));
    double**** Cgrid = (double****)malloc(q * sizeof(double***));
    for (int i = 0; i < q; i++) {
        Agrid[i] = (double***)malloc(q * sizeof(double**));
        Bgrid[i] = (double***)malloc(q * sizeof(double**));
        Cgrid[i] = (double***)malloc(q * sizeof(double**));
        for (int j = 0; j < q; j++) {
            Agrid[i][j] = alloc_matrix(s);
            Bgrid[i][j] = alloc_matrix(s);
            Cgrid[i][j] = alloc_matrix(s);
            for (int a = 0; a < s; a++)
                for (int b = 0; b < s; b++) {
                    Agrid[i][j][a][b] = A[i*s+a][j*s+b];
                    Bgrid[i][j][a][b] = B[i*s+a][j*s+b];
                }
        }
    }

    double**** curBg = (double****)malloc(q * sizeof(double***));
    for (int i = 0; i < q; i++) {
        curBg[i] = (double***)malloc(q * sizeof(double**));
        for (int j = 0; j < q; j++) curBg[i][j] = Bgrid[i][j];
    }

    for (int step = 0; step < q; step++) {
        for (int i = 0; i < q; i++) {
            int bcast_col = (i + step) % q;
            double** Abroadcast = Agrid[i][bcast_col]; /* row i broadcast source */
            for (int j = 0; j < q; j++)
                block_mult_accumulate(Cgrid[i][j], Abroadcast, curBg[i][j], s);
        }
        /* shift B up: each row takes B from the row below it (wraparound) */
        double**** nextBg = (double****)malloc(q * sizeof(double***));
        for (int i = 0; i < q; i++) {
            nextBg[i] = (double***)malloc(q * sizeof(double**));
            for (int j = 0; j < q; j++) nextBg[i][j] = curBg[(i+1) % q][j];
        }
        for (int i = 0; i < q; i++) free(curBg[i]);
        free(curBg);
        curBg = nextBg;
    }

    for (int i = 0; i < q; i++)
        for (int j = 0; j < q; j++)
            for (int a = 0; a < s; a++)
                for (int b = 0; b < s; b++)
                    C[i*s+a][j*s+b] = Cgrid[i][j][a][b];

    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            free_matrix(Agrid[i][j], s);
            free_matrix(Bgrid[i][j], s);
            free_matrix(Cgrid[i][j], s);
        }
        free(Agrid[i]); free(Bgrid[i]); free(Cgrid[i]); free(curBg[i]);
    }
    free(Agrid); free(Bgrid); free(Cgrid); free(curBg);
}

```

---
[<- Back to index](../README.md)
