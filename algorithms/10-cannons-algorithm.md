# 10. Cannon's Algorithm

| Field | Details |
|---|---|
| **Author(s)** | Lynn Elliot Cannon |
| **Year** | 1969 |
| **Time Complexity** | O(n^3/p) computation per processor on a sqrt(p) x sqrt(p) mesh; O(n^2/sqrt(p)) communication volume |
| **Approach** | Serial (Distributed-Memory) |
| **Original Paper Link** | https://dl.acm.org/doi/10.5555/905686 (Ph.D. Thesis, Montana State University -- ACM Guide/ProQuest record; theses generally carry no DOI) |

> **Implementation note:** Cannon's Algorithm is inherently distributed-memory. The implementation below **simulates** a q x q virtual processor grid on a single CPU, executing the algorithm's real skew-then-shift communication pattern sequentially.

## Pseudocode

```
FUNCTION CannonMatMul(A, B):    // executed identically on each processor P(i,j)
    Initially skew A: processor P(i,j) holds A-block shifted left by i
    Initially skew B: processor P(i,j) holds B-block shifted up by j
    C(i,j) = 0
    FOR step = 1 TO sqrt(p):
        C(i,j) = C(i,j) + A_local * B_local
        Circularly shift A_local one step left (within processor row)
        Circularly shift B_local one step up (within processor column)
    RETURN C(i,j)    // full C is the union of all local blocks
```

## CPU (C) Implementation

Full compilable source: [`src/algo_10_to_13.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   10. CANNON'S ALGORITHM (simulated q x q grid, n must be divisible by q)
   ===================================================================== */
void cannon_matmul_simulated(double** A, double** B, double** C, int n, int q) {
    if (n % q != 0) { fprintf(stderr, "[Cannon] n must be divisible by q\n"); return; }
    int s = n / q;
    fill_zero(C, n);

    /* Extract original blocks */
    double*** Ablk = (double***)malloc(q * sizeof(double**));
    double*** Bblk = (double***)malloc(q * sizeof(double**));
    double*** Cblk = (double***)malloc(q * sizeof(double**));
    double*** curA = (double***)malloc(q * sizeof(double**));
    double*** curB = (double***)malloc(q * sizeof(double**));
    for (int i = 0; i < q; i++) {
        Ablk[i] = (double**)malloc(q * sizeof(double*));
        Bblk[i] = (double**)malloc(q * sizeof(double*));
        Cblk[i] = (double**)malloc(q * sizeof(double*));
        curA[i] = (double**)malloc(q * sizeof(double*));
        curB[i] = (double**)malloc(q * sizeof(double*));
    }
    for (int i = 0; i < q; i++)
        for (int j = 0; j < q; j++) {
            Ablk[i][j] = (double*)0; /* placeholder pointers not used directly below */
        }

    /* Use simple 2D arrays of double** blocks instead (clearer) */
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

    /* Initial skew: skewA[i][j] = Agrid[i][(j+i)%q] ; skewB[i][j] = Bgrid[(i+j)%q][j] */
    double**** curAg = (double****)malloc(q * sizeof(double***));
    double**** curBg = (double****)malloc(q * sizeof(double***));
    for (int i = 0; i < q; i++) {
        curAg[i] = (double***)malloc(q * sizeof(double**));
        curBg[i] = (double***)malloc(q * sizeof(double**));
        for (int j = 0; j < q; j++) {
            curAg[i][j] = Agrid[i][(j+i) % q];
            curBg[i][j] = Bgrid[(i+j) % q][j];
        }
    }

    for (int step = 0; step < q; step++) {
        for (int i = 0; i < q; i++)
            for (int j = 0; j < q; j++)
                block_mult_accumulate(Cgrid[i][j], curAg[i][j], curBg[i][j], s);

        /* circular shift A left (each proc takes from its right neighbor) */
        double**** nextAg = (double****)malloc(q * sizeof(double***));
        double**** nextBg = (double****)malloc(q * sizeof(double***));
        for (int i = 0; i < q; i++) {
            nextAg[i] = (double***)malloc(q * sizeof(double**));
            nextBg[i] = (double***)malloc(q * sizeof(double**));
            for (int j = 0; j < q; j++) {
                nextAg[i][j] = curAg[i][(j+1) % q];
                nextBg[i][j] = curBg[(i+1) % q][j];
            }
        }
        for (int i = 0; i < q; i++) { free(curAg[i]); free(curBg[i]); }
        free(curAg); free(curBg);
        curAg = nextAg; curBg = nextBg;
    }

    /* Gather C */
    for (int i = 0; i < q; i++)
        for (int j = 0; j < q; j++)
            for (int a = 0; a < s; a++)
                for (int b = 0; b < s; b++)
                    C[i*s+a][j*s+b] = Cgrid[i][j][a][b];

    /* Cleanup */
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            free_matrix(Agrid[i][j], s);
            free_matrix(Bgrid[i][j], s);
            free_matrix(Cgrid[i][j], s);
        }
        free(Agrid[i]); free(Bgrid[i]); free(Cgrid[i]);
        free(curAg[i]); free(curBg[i]);
        free(Ablk[i]); free(Bblk[i]); free(Cblk[i]); free(curA[i]); free(curB[i]);
    }
    free(Agrid); free(Bgrid); free(Cgrid); free(curAg); free(curBg);
    free(Ablk); free(Bblk); free(Cblk); free(curA); free(curB);
}

```

---
[<- Back to index](../README.md)
