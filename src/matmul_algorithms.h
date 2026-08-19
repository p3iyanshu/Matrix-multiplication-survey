#ifndef MATMUL_ALGORITHMS_H
#define MATMUL_ALGORITHMS_H

/* =====================================================================
   Matrix Multiplication Algorithms — CPU (C) Reference Implementations
   NIT Warangal Research Internship — Literature Survey Companion Code
   =====================================================================
   All matrices are square, doubles, stored as double** (array of row
   pointers) unless noted otherwise. Distributed-memory algorithms
   (Cannon, Fox, SUMMA, CA-MM/2.5D) are SIMULATED on a single CPU
   process: each "processor" is represented as an array slot, and
   communication steps (broadcast, shift, reduce) are modeled explicitly
   as data movement between these slots, executed sequentially. This is
   a standard way to study distributed algorithms without an actual
   cluster/MPI environment, and is noted at each relevant function.
   ===================================================================== */

/* ---------- Matrix utility functions ---------- */
double** alloc_matrix(int n);
void free_matrix(double** M, int n);
void fill_random(double** M, int n, unsigned int seed);
void fill_zero(double** M, int n);
void copy_matrix(double** dst, double** src, int n);
void print_matrix(double** M, int n, const char* label);
int matrices_equal(double** A, double** B, int n, double tol);
double** matrix_add(double** A, double** B, int n);
double** matrix_sub(double** A, double** B, int n);

/* 1. Classical (Naive) Matrix Multiplication — O(n^3) */
void classical_matmul(double** A, double** B, double** C, int n);

/* 2. Strassen's Algorithm — O(n^log2(7)) */
double** strassen_matmul(double** A, double** B, int n);

/* 3. Winograd's Algorithm (Strassen-Winograd variant, reduced additions) */
double** winograd_matmul(double** A, double** B, int n);

/* 4. Coppersmith-Winograd — THEORETICAL ONLY. No practical CPU implementation
      exists or is attempted; function documents why and falls back safely. */
double** coppersmith_winograd_matmul(double** A, double** B, int n);

/* 5. Schonhage-Strassen — INTEGER multiplication, NOT matrix multiplication.
      Included per requirement; multiplies two big integers (digit arrays)
      via FFT-based convolution, demonstrating the algorithm on its actual
      domain rather than forcing it into a matrix-multiplication API. */
void schonhage_strassen_multiply(int* x, int lenx, int* y, int leny,
                                  int* result, int* result_len);

/* 6. Block (Blocked/Tiled) Matrix Multiplication — cache-blocked, O(n^3) */
void block_matmul(double** A, double** B, double** C, int n, int block_size);

/* 7. Tiled Matrix Multiplication — CUDA shared-memory tiling pattern,
      SIMULATED on CPU (phase-by-phase tile loads, mirroring the CUDA
      kernel structure exactly, but executed by the CPU instead of
      GPU threads). See README for the real .cu kernel. */
void tiled_matmul_cpu_simulated(double** A, double** B, double** C, int n, int tile_size);

/* 8. Recursive Matrix Multiplication — plain divide-and-conquer, no
      Strassen trick, O(n^3) */
void recursive_matmul(double** A, double** B, double** C, int n);

/* 9. Cache-Oblivious Matrix Multiplication — recursive split of the
      largest dimension, asymptotically optimal cache behavior */
void cache_oblivious_matmul(double** A, double** B, double** C,
                             int row_off_a, int col_off_a,
                             int row_off_b, int col_off_b,
                             int row_off_c, int col_off_c,
                             int m, int n, int p, int lda, int ldb, int ldc);
void cache_oblivious_matmul_wrapper(double** A, double** B, double** C, int n);

/* 10. Cannon's Algorithm — distributed-memory, SIMULATED as a q x q
       virtual processor grid (q = sqrt(num_procs)) on a single CPU */
void cannon_matmul_simulated(double** A, double** B, double** C, int n, int q);

/* 11. Fox's Algorithm — broadcast-multiply-roll, SIMULATED as a q x q
       virtual processor grid */
void fox_matmul_simulated(double** A, double** B, double** C, int n, int q);

/* 12. SUMMA — SIMULATED as a pr x pc virtual processor grid */
void summa_matmul_simulated(double** A, double** B, double** C, int n, int pr, int pc);

/* 13. Communication-Avoiding Matrix Multiplication (2.5D) — SIMULATED
       with c replicated layers over a q x q base grid */
void ca_mm_2_5d_simulated(double** A, double** B, double** C, int n, int q, int c);

/* 14. Parallel GEMM — general umbrella dispatcher over the four
       distributed-memory algorithms above */
typedef enum { PGEMM_CANNON, PGEMM_FOX, PGEMM_SUMMA, PGEMM_CA25D } ParallelGemmAlgo;
void parallel_gemm(double** A, double** B, double** C, int n, ParallelGemmAlgo algo);

/* 15. cuBLAS GEMM — GPU-only, closed-source; CANNOT run on CPU.
       This function provides the best practical CPU analog (aggressively
       blocked + loop-restructured GEMM) purely as a performance reference
       point, and is NOT a reimplementation of cuBLAS itself. */
void cublas_style_cpu_reference_gemm(double** A, double** B, double** C, int n);

#endif
