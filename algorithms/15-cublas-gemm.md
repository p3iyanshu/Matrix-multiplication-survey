# 15. cuBLAS GEMM (Optimized Library Implementation -- Not a Novel Algorithm)

| Field | Details |
|---|---|
| **Author(s)** | NVIDIA Corporation (proprietary library engineering team; no single academic inventor) |
| **Year** | First released 2007 (initial CUDA Toolkit); continuously updated since |
| **Time Complexity** | O(n^3) -- same asymptotic complexity as classical GEMM; near-peak hardware throughput via micro-architectural optimization, not a new algorithm |
| **Approach** | CUDA |
| **Original Paper Link** | https://docs.nvidia.com/cuda/cublas/ (NVIDIA Developer Documentation, living page, no fixed publication year) |

> **Implementation note:** cuBLAS is closed-source and GPU-only; it cannot run on a CPU and is not reimplemented here. The implementation below is an aggressively blocked, cache-friendly CPU GEMM used only as a **best-practical-effort performance analog**, not a reimplementation of cuBLAS. The closest official NVIDIA open-source code (CUTLASS) is included below for reference.

## Pseudocode

```
FUNCTION cuBLAS_GEMM(A, B, C, alpha, beta):
    Select a pre-tuned kernel variant based on GPU architecture, matrix shape,
        and data type (FP32/FP16/BF16/TF32/INT8, via autotuning/heuristics)
    Load tiles of A and B into shared memory using warp-level and register-blocked tiling
    ON architectures with Tensor Cores: issue mma/wgmma warp-matrix-multiply-accumulate
        instructions instead of scalar FMA
    Accumulate partial tile products into registers/shared memory
    Apply epilogue: C = alpha * (A x B) + beta * C
    RETURN C
```

## Closest Official NVIDIA Source Code (CUTLASS, cuBLAS-Adjacent)

cuBLAS itself is closed-source. NVIDIA states CUTLASS "incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS and cuDNN." Verified against [the official repo](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu).

```cpp
/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SOURCE: https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu
 *
 * NOTE: cuBLAS itself is closed-source. NVIDIA officially states that
 * CUTLASS "incorporates strategies for hierarchical decomposition and
 * data movement similar to those used to implement cuBLAS and cuDNN."
 * This is the closest official, open-source NVIDIA GEMM code available.
 **************************************************************************************************/

#include "cutlass/gemm/device/gemm.h"

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    ColumnMajor, // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    ColumnMajor, // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  CutlassGemm::Arguments args({M, N, K},   // Gemm Problem dimensions
                               {A, lda},   // Tensor-ref for source matrix A
                               {B, ldb},   // Tensor-ref for source matrix B
                               {C, ldc},   // Tensor-ref for source matrix C
                               {C, ldc},   // Tensor-ref for destination matrix D
                               {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel.
  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

/// Naive reference GEMM computation (included in the same official file,
/// used by NVIDIA to verify the CUTLASS kernel's correctness).
__global__ void ReferenceGemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

```

## CPU (C) Implementation

Full compilable source: [`src/algo_14_to_15.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
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

```

---
[<- Back to index](../README.md)
