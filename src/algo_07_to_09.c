#include <stdio.h>
#include <stdlib.h>
#include "matmul_algorithms.h"

/* =====================================================================
   7. TILED MATRIX MULTIPLICATION — CUDA shared-memory tiling pattern,
   SIMULATED ON CPU.

   On a real GPU, each CUDA thread block cooperatively loads a TxT tile
   of A and a TxT tile of B into __shared__ memory, synchronizes with
   __syncthreads(), then every thread in the block reuses those tiles
   for T multiply-adds before the next tile "phase" is loaded. Here we
   reproduce that exact phase structure on the CPU using explicit tile
   buffers (playing the role of shared memory) so the algorithm's data
   movement pattern matches the CUDA kernel exactly -- only the
   execution model (sequential CPU loop vs. thousands of parallel GPU
   threads) differs. The real .cu kernel is given separately.
   ===================================================================== */
void tiled_matmul_cpu_simulated(double** A, double** B, double** C, int n, int tile_size) {
    fill_zero(C, n);
    double** As = alloc_matrix(tile_size); /* plays the role of __shared__ As */
    double** Bs = alloc_matrix(tile_size); /* plays the role of __shared__ Bs */

    for (int row_tile = 0; row_tile < n; row_tile += tile_size) {
        for (int col_tile = 0; col_tile < n; col_tile += tile_size) {
            int rmax = (row_tile + tile_size < n) ? row_tile + tile_size : n;
            int cmax = (col_tile + tile_size < n) ? col_tile + tile_size : n;

            /* accumulator per output element in this output tile */
            int rsize = rmax - row_tile, csize = cmax - col_tile;
            double** acc = alloc_matrix(tile_size);

            int num_phases = (n + tile_size - 1) / tile_size;
            for (int ph = 0; ph < num_phases; ph++) {
                int k0 = ph * tile_size;
                int kmax = (k0 + tile_size < n) ? k0 + tile_size : n;
                int ksize = kmax - k0;

                /* "Load" phase: cooperative tile load into shared memory */
                for (int i = 0; i < rsize; i++)
                    for (int k = 0; k < ksize; k++)
                        As[i][k] = A[row_tile + i][k0 + k];
                for (int k = 0; k < ksize; k++)
                    for (int j = 0; j < csize; j++)
                        Bs[k][j] = B[k0 + k][col_tile + j];
                /* __syncthreads() equivalent: tiles are now fully loaded */

                /* "Compute" phase: every thread reuses the shared tiles */
                for (int i = 0; i < rsize; i++)
                    for (int j = 0; j < csize; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < ksize; k++)
                            sum += As[i][k] * Bs[k][j];
                        acc[i][j] += sum;
                    }
                /* second __syncthreads() equivalent before next phase */
            }

            for (int i = 0; i < rsize; i++)
                for (int j = 0; j < csize; j++)
                    C[row_tile + i][col_tile + j] = acc[i][j];

            free_matrix(acc, tile_size);
        }
    }
    free_matrix(As, tile_size);
    free_matrix(Bs, tile_size);
}

/* =====================================================================
   8. RECURSIVE MATRIX MULTIPLICATION — plain divide-and-conquer,
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
