#include <stdio.h>
#include "matmul_algorithms.h"

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

/* =====================================================================
   15. cuBLAS GEMM — GPU-only, closed-source. It CANNOT run on a CPU and
   is not reimplemented here (that would misrepresent it). What follows
   is an aggressively blocked, cache- and loop-order-optimized CPU GEMM
   used purely as a "best practical CPU analog" performance reference
   point -- NOT a reimplementation of cuBLAS, and NOT claimed to be
   equivalent to it. The real GPU-side reference implementation
   NVIDIA officially provides is CUTLASS (see README / earlier chat).
   ===================================================================== */
void cublas_style_cpu_reference_gemm(double** A, double** B, double** C, int n) {
    fprintf(stderr,
        "[NOTE] cuBLAS itself is closed-source and GPU-only; it cannot run "
        "on a CPU and is not reimplemented here. This function is only a "
        "best-practical-effort CPU analog (blocked + cache-friendly loop "
        "order), used as a performance reference point.\n");

    const int BS = 32; /* block size tuned for typical L1/L2 cache */
    fill_zero(C, n);
    for (int ii = 0; ii < n; ii += BS) {
        int imax = (ii + BS < n) ? ii + BS : n;
        for (int kk = 0; kk < n; kk += BS) {
            int kmax = (kk + BS < n) ? kk + BS : n;
            for (int jj = 0; jj < n; jj += BS) {
                int jmax = (jj + BS < n) ? jj + BS : n;
                /* i-k-j loop order: inner j-loop is stride-1 on both
                   B[k][j] and C[i][j], maximizing cache-line reuse --
                   this is the single biggest lever available on CPU
                   before resorting to explicit SIMD intrinsics. */
                for (int i = ii; i < imax; i++) {
                    for (int k = kk; k < kmax; k++) {
                        double a_ik = A[i][k];
                        for (int j = jj; j < jmax; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}
