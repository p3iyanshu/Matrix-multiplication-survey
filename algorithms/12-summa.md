# 12. SUMMA (Scalable Universal Matrix Multiplication Algorithm)

| Field | Details |
|---|---|
| **Author(s)** | Robert A. van de Geijn, Jerrell Watts |
| **Year** | 1997 |
| **Time Complexity** | O(n^3/p) computation; O(n^2/sqrt(p)) communication volume (grid-shape dependent) |
| **Approach** | Serial (Distributed-Memory) |
| **Original Paper Link** | https://doi.org/10.1002/(SICI)1096-9128(199704)9:4%3C255::AID-CPE250%3E3.0.CO;2-2 (Wiley -- Concurrency: Practice and Experience, 9(4), 255-274) |

> **Implementation note:** SUMMA is inherently distributed-memory. The implementation below **simulates** a q x q virtual processor grid using SUMMA's broadcast-then-rank-1-update pattern (structurally distinct from Cannon's/Fox's shift-based approach). The genuine **author-provided original source code** (Robert van de Geijn's PLAPACK `PLA_Gemm.c`, GNU GPL) is also included below for reference.

## Pseudocode

```
FUNCTION SUMMA(A, B):   // executed identically on each processor P(i,j) in a pr x pc grid
    C(i,j) = 0
    FOR k = 0 TO n-1:            // or in blocks of width b for blocked SUMMA
        The processor column owning column k of A broadcasts that column (A[:,k]) across its processor row
        The processor row owning row k of B broadcasts that row (B[k,:]) across its processor column
        C(i,j) = C(i,j) + A(i,k) * B(k,j)     // rank-1 update
    RETURN C(i,j)
```

## Author-Provided Original Source Code (SUMMA)

Actual production C source from Robert van de Geijn's PLAPACK library (GNU GPL), directly implementing SUMMA. Verified against [the live file](https://www.cs.utexas.edu/~plapack/Code/PLAPACK20/BLAS3/PLA_Gemm.c).

```c
/***************************************************************************
  Parallel Linear Algebra Package Release R2.0.2 -- 6 Feb 2000
  Copyright (c) 1997,1998,1999,2000 Robert van de Geijn and
  The University of Texas at Austin.  (GNU General Public License)

  Written under the direction of: Robert van de Geijn,
  Department of Computer Sciences, University of Texas at Austin

  SOURCE: https://www.cs.utexas.edu/~plapack/Code/PLAPACK20/BLAS3/PLA_Gemm.c
***************************************************************************/

#include "PLA.h"

int PLA_Gemm( int transa, int transb,
              PLA_Obj alpha, PLA_Obj A, PLA_Obj B,
              PLA_Obj beta,  PLA_Obj C )
/*
  Purpose : Parallel matrix multiplication
  C <- alpha * A * B + beta * C   (plus transposed variants)

  NOTE: For details on how to implement matrix-matrix multiplication, see
        R. van de Geijn, Using PLAPACK, The MIT Press, 1997.
        R. van de Geijn and J. Watts,
        "SUMMA: Scalable Universal Matrix Multiplication Algorithm,"
        Concurrency: Practice and Experience, Vol 9(4), pp. 255-274 (1997).
*/
{
  int value = PLA_SUCCESS, owner_row, owner_col;
  int length_A, width_A, length_B, width_B, length_C, width_C, nb_alg;
  PLA_Obj alpha_cpy = NULL, beta_cpy = NULL;
  PLA_Template templ = NULL;

  if ( PLA_ERROR_CHECKING )
    value = PLA_Gemm_enter( transa, transb, alpha, A, B, beta, C );

  if ( !value ){
    /* If necessary, duplicate alpha and beta to all nodes */
    PLA_Obj_owner_row( alpha, &owner_row );
    PLA_Obj_owner_col( alpha, &owner_col );
    if ( owner_row != PLA_ALL_ROWS || owner_col != PLA_ALL_COLS ){
      PLA_Mscalar_create_conf_to( alpha, PLA_ALL_ROWS, PLA_ALL_COLS, &alpha_cpy );
      PLA_Copy( beta, beta_cpy );
    }
    PLA_Obj_owner_row( beta, &owner_row );
    PLA_Obj_owner_col( beta, &owner_col );
    if ( owner_row != PLA_ALL_ROWS || owner_col != PLA_ALL_COLS ){
      PLA_Mscalar_create_conf_to( beta, PLA_ALL_ROWS, PLA_ALL_COLS, &beta_cpy );
      PLA_Copy( beta, beta_cpy );
    }

    PLA_Obj_template( A, &templ );
    PLA_Obj_global_length( A, &length_A );  PLA_Obj_global_width( A, &width_A );
    PLA_Obj_global_length( B, &length_B );  PLA_Obj_global_width( B, &width_B );
    PLA_Obj_global_length( C, &length_C );  PLA_Obj_global_width( C, &width_C );

    /* SUMMA's core idea: whichever matrix (A, B, or C) holds the most data
       is left in place, and the other two are the ones communicated/broadcast */
    if ( length_A*width_A > length_B*width_B && length_A*width_A > length_C*width_C ){
      PLA_Environ_nb_alg( PLA_OP_MAT_PAN, templ, &nb_alg );
      PLA_Gemm_A( nb_alg, transa, transb,
                  (alpha_cpy==NULL?alpha:alpha_cpy), A, B,
                  (beta_cpy==NULL?beta:beta_cpy), C );
    }
    else if ( length_B*width_B > length_A*width_A && length_B*width_B > length_C*width_C ){
      PLA_Environ_nb_alg( PLA_OP_PAN_MAT, templ, &nb_alg );
      PLA_Gemm_B( nb_alg, transa, transb,
                  (alpha_cpy==NULL?alpha:alpha_cpy), A, B,
                  (beta_cpy==NULL?beta:beta_cpy), C );
    }
    else {
      PLA_Environ_nb_alg( PLA_OP_PAN_PAN, templ, &nb_alg );
      PLA_Gemm_C( nb_alg, transa, transb,
                  (alpha_cpy==NULL?alpha:alpha_cpy), A, B,
                  (beta_cpy==NULL?beta:beta_cpy), C );
    }
    PLA_Obj_free( &alpha_cpy );
    PLA_Obj_free( &beta_cpy );
  }
  if ( PLA_ERROR_CHECKING )
    value = PLA_Gemm_exit( transa, transb, alpha, A, B, beta, C );
  return value;
}

```

## CPU (C) Implementation

Full compilable source: [`src/algo_10_to_13.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
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

```

---
[<- Back to index](../README.md)
