#include <stdio.h>
#include <stdlib.h>
#include "matmul_algorithms.h"

/* All four algorithms in this file are distributed-memory algorithms.
   Since the requested platform is CPU-only (no MPI), each is SIMULATED
   as a q x q (or pr x pc) grid of "virtual processors," where each
   virtual processor's local block is just an array slot processed
   sequentially. Communication steps (broadcast, circular shift,
   layer reduction) are modeled explicitly as data movement between
   these slots, so the algorithm's actual communication PATTERN is
   preserved and visible in the code -- only true network communication
   and parallel execution are absent (they're serialized on one CPU). */

static void block_mult_accumulate(double** Cblk, double** Ablk, double** Bblk, int s) {
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) {
            double sum = 0.0;
            for (int k = 0; k < s; k++) sum += Ablk[i][k] * Bblk[k][j];
            Cblk[i][j] += sum;
        }
}

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

/* =====================================================================
   12. SUMMA (broadcast column-of-A-blocks / row-of-B-blocks, rank-b update)
   Simulated over a q x q grid (pr = pc = q here for simplicity).
   ===================================================================== */
void summa_matmul_simulated(double** A, double** B, double** C, int n, int pr, int pc) {
    if (pr != pc) { fprintf(stderr, "[SUMMA] demo requires pr == pc\n"); return; }
    int q = pr;
    if (n % q != 0) { fprintf(stderr, "[SUMMA] n must be divisible by grid size\n"); return; }
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

    /* At each panel step kp, the processor column owning A-block-column kp
       "broadcasts" A[i][kp] across row i; the processor row owning
       B-block-row kp "broadcasts" B[kp][j] across column j. In this
       sequential simulation, broadcasting simply means every processor
       reads that shared block directly (no physical copy needed on a
       single machine) -- this is the key structural difference from
       Cannon/Fox, which instead physically SHIFT blocks between steps. */
    for (int kp = 0; kp < q; kp++) {
        for (int i = 0; i < q; i++)
            for (int j = 0; j < q; j++)
                block_mult_accumulate(Cgrid[i][j], Agrid[i][kp], Bgrid[kp][j], s);
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
        free(Agrid[i]); free(Bgrid[i]); free(Cgrid[i]);
    }
    free(Agrid); free(Bgrid); free(Cgrid);
}

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
