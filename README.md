# Matrix Multiplication Algorithms: Literature Survey & Reference Implementations

A literature survey and companion CPU (C) implementation of 15 matrix multiplication
algorithms spanning classical numerical linear algebra, fast (sub-cubic) algorithms,
cache-aware algorithms, distributed-memory parallel algorithms, and GPU/CUDA-oriented
approaches. Prepared as part of a research internship (NIT Warangal).

Every entry below has its **own file** containing that algorithm's pseudocode, the
original author-provided source code (where one genuinely exists and is publicly
available), and a working, tested CPU (C) implementation. Click an algorithm name to
open it directly.

## Index

| # | Algorithm | Approach | Year | Complexity |
|---|---|---|---|---|
| 1 | [Classical (Naive) Matrix Multiplication](algorithms/01-classical-naive-matrix-multiplication.md) | Serial | -- | O(n^3) |
| 2 | [Strassen's Algorithm](algorithms/02-strassens-algorithm.md) | Serial | 1969 | O(n^2.807) |
| 3 | [Winograd's Algorithm (Strassen-Winograd Variant)](algorithms/03-winograds-algorithm.md) | Serial | 1971 | O(n^2.807) |
| 4 | [Coppersmith-Winograd Algorithm](algorithms/04-coppersmith-winograd-algorithm.md) | Serial (theoretical) | 1990 | O(n^2.376) |
| 5 | [Schonhage-Strassen Algorithm](algorithms/05-schonhage-strassen-algorithm.md) (integer mult.) | Serial | 1971 | O(n log n log log n) |
| 6 | [Block (Blocked) Matrix Multiplication](algorithms/06-block-blocked-matrix-multiplication.md) | Serial | -- | O(n^3) |
| 7 | [Tiled Matrix Multiplication (CUDA)](algorithms/07-tiled-matrix-multiplication-cuda.md) | CUDA | -- | O(n^3) |
| 8 | [Recursive Matrix Multiplication](algorithms/08-recursive-matrix-multiplication.md) | Serial | -- | O(n^3) |
| 9 | [Cache-Oblivious Matrix Multiplication](algorithms/09-cache-oblivious-matrix-multiplication.md) | Serial | 1999 | O(n^3) |
| 10 | [Cannon's Algorithm](algorithms/10-cannons-algorithm.md) | Serial (Distributed-Memory) | 1969 | O(n^3/p) |
| 11 | [Fox's Algorithm](algorithms/11-foxs-algorithm.md) | Serial (Distributed-Memory) | 1987 | O(n^3/p) |
| 12 | [SUMMA](algorithms/12-summa.md) | Serial (Distributed-Memory) | 1997 | O(n^3/p) |
| 13 | [Communication-Avoiding Matrix Multiplication (CA-MM / 2.5D)](algorithms/13-communication-avoiding-matrix-multiplication.md) | Serial (Distributed-Memory) | 2011 | O(n^3/p) |
| 14 | [Parallel GEMM (General Category)](algorithms/14-parallel-gemm.md) | Serial (Distributed-Memory) | -- | O(n^3/p) |
| 15 | [cuBLAS GEMM](algorithms/15-cublas-gemm.md) (library, not a novel algorithm) | CUDA | 2007+ | O(n^3) |

## Repository Structure

```
.
|-- README.md                  <- you are here
|-- algorithms/                <- one file per algorithm: pseudocode + author code + implementation
|   |-- 01-classical-naive-matrix-multiplication.md
|   |-- 02-strassens-algorithm.md
|   |-- ... (15 files total)
|-- src/                       <- full compilable CPU (C) project (all 15 algorithms + test suite)
|   |-- matmul_algorithms.h
|   |-- utils.c
|   |-- algo_01_to_06.c
|   |-- algo_07_to_09.c
|   |-- algo_10_to_13.c
|   |-- algo_14_to_15.c
|   |-- main.c
|   `-- Makefile
|-- docs/
|   |-- Matrix_Multiplication_Full_Report.pdf   <- combined pseudocode + code, all 15, one PDF
|   `-- Matrix_Multiplication_Algorithms.xlsx   <- literature survey spreadsheet
```

## Building & Running the Code

```bash
cd src
make
./matmul_test
```

Requires `gcc` and a C11-capable toolchain; no dependencies beyond the standard C
math library (`-lm`). All 15 algorithms are correctness-tested against the classical
result on every run.

## Important Notes on Scope

- **Coppersmith-Winograd** has no practical implementation anywhere (a "galactic
  algorithm"); its file documents this and falls back to Strassen so the code still runs.
- **Schonhage-Strassen** is an *integer*-multiplication algorithm, not matrix
  multiplication -- implemented on its actual domain (big-integer FFT multiplication).
- **Cannon's, Fox's, SUMMA, and CA-MM** are inherently distributed-memory algorithms.
  Since this project targets CPU only (no MPI/cluster), each is *simulated* as a virtual
  processor grid on a single machine, executing the real communication pattern
  (skew/shift, broadcast/roll, broadcast/rank-1-update, layer-replication/reduction)
  sequentially. This is stated explicitly in each relevant file.
- **Tiled Matrix Multiplication** is a CUDA shared-memory pattern; the CPU file
  simulates the exact phase-by-phase tile-load/sync/compute structure of the real kernel.
- **cuBLAS GEMM** is closed-source and GPU-only; it cannot run on a CPU and is not
  reimplemented. Its file includes a labeled "best practical CPU analog" instead, plus
  NVIDIA's own open-source CUTLASS code as the closest official reference.
- Only **SUMMA** and **cuBLAS** have genuine author-provided/official original source
  code publicly available; both are included in full in their respective files.

## License / Attribution

This repository's own code (pseudocode text and the `src/` CPU implementations) is
provided for academic/research use. Third-party author code included for reference
(PLAPACK's `PLA_Gemm.c`, GNU GPL; NVIDIA CUTLASS, BSD-3-Clause) retains its original
license and copyright, noted in each relevant file.
