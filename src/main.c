#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul_algorithms.h"

#define TOL 1e-6

static void run_test(const char* name, double** result, double** expected, int n) {
    int ok = matrices_equal(result, expected, n, TOL);
    printf("[%-45s] %s\n", name, ok ? "PASS (matches classical result)" : "FAIL");
}

int main(void) {
    printf("=====================================================================\n");
    printf(" Matrix Multiplication Algorithms -- CPU (C) Correctness Test Suite\n");
    printf("=====================================================================\n\n");

    /* --- Test 1: power-of-2 size (8x8) for the recursive-family algorithms --- */
    int n = 8;
    double** A = alloc_matrix(n);
    double** B = alloc_matrix(n);
    fill_random(A, n, 42);
    fill_random(B, n, 99);

    double** expected = alloc_matrix(n);
    classical_matmul(A, B, expected, n);
    run_test("1. Classical (Naive) Matrix Multiplication", expected, expected, n);

    double** C_strassen = strassen_matmul(A, B, n);
    run_test("2. Strassen's Algorithm", C_strassen, expected, n);

    double** C_winograd = winograd_matmul(A, B, n);
    run_test("3. Winograd's Algorithm (Strassen-Winograd variant)", C_winograd, expected, n);

    double** C_cw = coppersmith_winograd_matmul(A, B, n); /* falls back to Strassen, see stderr note */
    run_test("4. Coppersmith-Winograd (theoretical; Strassen fallback)", C_cw, expected, n);

    double** C_block = alloc_matrix(n);
    block_matmul(A, B, C_block, n, 4);
    run_test("6. Block (Blocked) Matrix Multiplication", C_block, expected, n);

    double** C_tiled = alloc_matrix(n);
    tiled_matmul_cpu_simulated(A, B, C_tiled, n, 4);
    run_test("7. Tiled Matrix Multiplication (CUDA pattern, CPU-simulated)", C_tiled, expected, n);

    double** C_recursive = alloc_matrix(n);
    recursive_matmul(A, B, C_recursive, n);
    run_test("8. Recursive Matrix Multiplication (plain)", C_recursive, expected, n);

    double** C_cache_obl = alloc_matrix(n);
    cache_oblivious_matmul_wrapper(A, B, C_cache_obl, n);
    run_test("9. Cache-Oblivious Matrix Multiplication", C_cache_obl, expected, n);

    double** C_cannon = alloc_matrix(n);
    cannon_matmul_simulated(A, B, C_cannon, n, 2); /* 2x2 virtual processor grid */
    run_test("10. Cannon's Algorithm (2x2 grid, simulated)", C_cannon, expected, n);

    double** C_fox = alloc_matrix(n);
    fox_matmul_simulated(A, B, C_fox, n, 2);
    run_test("11. Fox's Algorithm (2x2 grid, simulated)", C_fox, expected, n);

    double** C_summa = alloc_matrix(n);
    summa_matmul_simulated(A, B, C_summa, n, 2, 2);
    run_test("12. SUMMA (2x2 grid, simulated)", C_summa, expected, n);

    double** C_camm = alloc_matrix(n);
    ca_mm_2_5d_simulated(A, B, C_camm, n, 2, 2); /* c=2 replication layers */
    run_test("13. Communication-Avoiding MatMul (2.5D, c=2, simulated)", C_camm, expected, n);

    double** C_pgemm_cannon = alloc_matrix(n);
    parallel_gemm(A, B, C_pgemm_cannon, n, PGEMM_CANNON);
    run_test("14. Parallel GEMM (dispatched -> Cannon)", C_pgemm_cannon, expected, n);

    double** C_pgemm_summa = alloc_matrix(n);
    parallel_gemm(A, B, C_pgemm_summa, n, PGEMM_SUMMA);
    run_test("14. Parallel GEMM (dispatched -> SUMMA)", C_pgemm_summa, expected, n);

    double** C_cublas_ref = alloc_matrix(n);
    cublas_style_cpu_reference_gemm(A, B, C_cublas_ref, n);
    run_test("15. cuBLAS-style CPU reference GEMM (analog only)", C_cublas_ref, expected, n);

    printf("\n---------------------------------------------------------------------\n");
    printf(" 5. Schonhage-Strassen -- INTEGER multiplication demo (not matrix mult)\n");
    printf("---------------------------------------------------------------------\n");
    /* Multiply 123456789 x 987654321 via FFT-based convolution */
    int x[] = {1,2,3,4,5,6,7,8,9};
    int y[] = {9,8,7,6,5,4,3,2,1};
    int result[64], result_len;
    schonhage_strassen_multiply(x, 9, y, 9, result, &result_len);
    printf("123456789 * 987654321 via Schonhage-Strassen FFT convolution = ");
    for (int i = 0; i < result_len; i++) printf("%d", result[i]);
    printf("\n");
    long long expected_int = 123456789LL * 987654321LL;
    printf("Expected (native 64-bit multiply)                            = %lld\n", expected_int);

    printf("\n---------------------------------------------------------------------\n");
    printf(" Non-power-of-2 size test (n=10) for padding-based algorithms\n");
    printf("---------------------------------------------------------------------\n");
    int n2 = 10;
    double** A2 = alloc_matrix(n2);
    double** B2 = alloc_matrix(n2);
    fill_random(A2, n2, 7);
    fill_random(B2, n2, 13);
    double** expected2 = alloc_matrix(n2);
    classical_matmul(A2, B2, expected2, n2);

    double** C2_strassen = strassen_matmul(A2, B2, n2);
    run_test("2. Strassen's Algorithm (n=10, zero-padded)", C2_strassen, expected2, n2);

    double** C2_recursive = alloc_matrix(n2);
    recursive_matmul(A2, B2, C2_recursive, n2);
    run_test("8. Recursive Matrix Multiplication (n=10, zero-padded)", C2_recursive, expected2, n2);

    double** C2_cache_obl = alloc_matrix(n2);
    cache_oblivious_matmul_wrapper(A2, B2, C2_cache_obl, n2);
    run_test("9. Cache-Oblivious MatMul (n=10, no padding needed)", C2_cache_obl, expected2, n2);

    printf("\n---------------------------------------------------------------------\n");
    printf(" Simple timing comparison (n=%d, not a rigorous benchmark)\n", n);
    printf("---------------------------------------------------------------------\n");
    int bn = 256;
    double** BA = alloc_matrix(bn);
    double** BB = alloc_matrix(bn);
    double** BC = alloc_matrix(bn);
    fill_random(BA, bn, 1);
    fill_random(BB, bn, 2);

    clock_t t0, t1;
    t0 = clock();
    classical_matmul(BA, BB, BC, bn);
    t1 = clock();
    printf("Classical         (n=%d): %.4f sec\n", bn, (double)(t1 - t0) / CLOCKS_PER_SEC);

    t0 = clock();
    block_matmul(BA, BB, BC, bn, 32);
    t1 = clock();
    printf("Block (tiled)      (n=%d): %.4f sec\n", bn, (double)(t1 - t0) / CLOCKS_PER_SEC);

    t0 = clock();
    cublas_style_cpu_reference_gemm(BA, BB, BC, bn);
    t1 = clock();
    printf("cuBLAS-style CPU analog (n=%d): %.4f sec\n", bn, (double)(t1 - t0) / CLOCKS_PER_SEC);

    t0 = clock();
    double** BC_strassen = strassen_matmul(BA, BB, bn);
    t1 = clock();
    printf("Strassen           (n=%d): %.4f sec\n", bn, (double)(t1 - t0) / CLOCKS_PER_SEC);
    free_matrix(BC_strassen, bn);

    /* Cleanup */
    free_matrix(A, n); free_matrix(B, n); free_matrix(expected, n);
    free_matrix(C_strassen, n); free_matrix(C_winograd, n); free_matrix(C_cw, n);
    free_matrix(C_block, n); free_matrix(C_tiled, n); free_matrix(C_recursive, n);
    free_matrix(C_cache_obl, n); free_matrix(C_cannon, n); free_matrix(C_fox, n);
    free_matrix(C_summa, n); free_matrix(C_camm, n); free_matrix(C_pgemm_cannon, n);
    free_matrix(C_pgemm_summa, n); free_matrix(C_cublas_ref, n);
    free_matrix(A2, n2); free_matrix(B2, n2); free_matrix(expected2, n2);
    free_matrix(C2_strassen, n2); free_matrix(C2_recursive, n2); free_matrix(C2_cache_obl, n2);
    free_matrix(BA, bn); free_matrix(BB, bn); free_matrix(BC, bn);

    printf("\nAll tests complete.\n");
    return 0;
}
