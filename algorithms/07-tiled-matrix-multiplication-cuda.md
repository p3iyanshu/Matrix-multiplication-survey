# 7. Tiled Matrix Multiplication (CUDA Shared-Memory Tiling)

| Field | Details |
|---|---|
| **Author(s)** | Not attributable to a single author -- standard CUDA programming pattern |
| **Year** | Not Verified -- formalized as a CUDA teaching pattern from the mid-2000s CUDA era onward |
| **Time Complexity** | O(n^3) -- same complexity, optimized constant factor for GPU memory hierarchy |
| **Approach** | CUDA |
| **Original Paper Link** | Not Applicable -- standard treatment in Kirk & Hwu (& El Hajj in later editions), *Programming Massively Parallel Processors*, Morgan Kaufmann (textbook, no DOI) |

> **Implementation note:** Tiled matrix multiplication is a CUDA shared-memory pattern. The implementation below *simulates* the exact phase-by-phase tile-load / synchronize / compute structure of a real CUDA kernel using CPU arrays in place of `__shared__` memory (the deliverable target for this repo's implementation was CPU-only).

## Pseudocode

```
FUNCTION TiledMatMulKernel(A, B, C, n, T):   // T = tile width; runs per CUDA thread
    row = blockIdx.y * T + threadIdx.y
    col = blockIdx.x * T + threadIdx.x
    __shared__ As[T][T], Bs[T][T]
    sum = 0
    FOR ph = 0 TO (n/T)-1:                    // loop over tile "phases"
        As[threadIdx.y][threadIdx.x] = A[row][ph*T + threadIdx.x]
        Bs[threadIdx.y][threadIdx.x] = B[ph*T + threadIdx.y][col]
        __syncthreads()
        FOR k = 0 TO T-1:
            sum = sum + As[threadIdx.y][k] * Bs[k][threadIdx.x]
        __syncthreads()
    C[row][col] = sum
```

## CPU (C) Implementation

Full compilable source: [`src/algo_07_to_09.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   7. TILED MATRIX MULTIPLICATION â CUDA shared-memory tiling pattern,
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

```

---
[<- Back to index](../README.md)
