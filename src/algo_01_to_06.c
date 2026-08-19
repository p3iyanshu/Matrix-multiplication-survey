#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul_algorithms.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =====================================================================
   1. CLASSICAL (NAIVE) MATRIX MULTIPLICATION — O(n^3)
   ===================================================================== */
void classical_matmul(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

/* =====================================================================
   2. STRASSEN'S ALGORITHM — O(n^log2(7)) ~ O(n^2.807)
   Requires n to be a power of 2. Falls back to classical multiplication
   below a base-case threshold (standard practical optimization, since
   Strassen's recursive overhead outweighs its FLOP savings for small n).
   ===================================================================== */
#define STRASSEN_BASE_CASE 64

static double** strassen_recursive(double** A, double** B, int n) {
    double** C = alloc_matrix(n);

    if (n <= STRASSEN_BASE_CASE) {
        classical_matmul(A, B, C, n);
        return C;
    }

    int half = n / 2;

    /* Allocate quadrant sub-matrices */
    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);

    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }

    double** S1 = matrix_add(A11, A22, half);
    double** T1 = matrix_add(B11, B22, half);
    double** M1 = strassen_recursive(S1, T1, half);

    double** S2 = matrix_add(A21, A22, half);
    double** M2 = strassen_recursive(S2, B11, half);

    double** T3 = matrix_sub(B12, B22, half);
    double** M3 = strassen_recursive(A11, T3, half);

    double** T4 = matrix_sub(B21, B11, half);
    double** M4 = strassen_recursive(A22, T4, half);

    double** S5 = matrix_add(A11, A12, half);
    double** M5 = strassen_recursive(S5, B22, half);

    double** S6 = matrix_sub(A21, A11, half);
    double** T6 = matrix_add(B11, B12, half);
    double** M6 = strassen_recursive(S6, T6, half);

    double** S7 = matrix_sub(A12, A22, half);
    double** T7 = matrix_add(B21, B22, half);
    double** M7 = strassen_recursive(S7, T7, half);

    /* C11 = M1 + M4 - M5 + M7 ; C12 = M3 + M5
       C21 = M2 + M4           ; C22 = M1 - M2 + M3 + M6 */
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C[i][j]               = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + half]        = M3[i][j] + M5[i][j];
            C[i + half][j]        = M2[i][j] + M4[i][j];
            C[i + half][j + half] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    /* Cleanup */
    double** temps[] = {S1, T1, M1, S2, M2, T3, M3, T4, M4, S5, M5, S6, T6, M6, S7, T7, M7};
    for (int t = 0; t < 17; t++) free_matrix(temps[t], half);
    free_matrix(A11, half); free_matrix(A12, half); free_matrix(A21, half); free_matrix(A22, half);
    free_matrix(B11, half); free_matrix(B12, half); free_matrix(B21, half); free_matrix(B22, half);

    return C;
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

double** strassen_matmul(double** A, double** B, int n) {
    int padded = next_pow2(n);
    if (padded == n) {
        return strassen_recursive(A, B, n);
    }
    /* Pad to next power of two with zeros, run, then the caller should
       only read the top-left n x n block of the result. */
    double** Apad = alloc_matrix(padded);
    double** Bpad = alloc_matrix(padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Apad[i][j] = A[i][j]; Bpad[i][j] = B[i][j]; }
    double** Cpad = strassen_recursive(Apad, Bpad, padded);
    double** C = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cpad[i][j];
    free_matrix(Apad, padded); free_matrix(Bpad, padded); free_matrix(Cpad, padded);
    return C;
}

/* =====================================================================
   3. WINOGRAD'S ALGORITHM (Strassen-Winograd variant)
   Same 7 multiplications as Strassen, but only 15 add/sub instead of 18
   (S1..S4, T1..T4, then combine — matches the paper's reduced-addition form).
   ===================================================================== */
static double** winograd_recursive(double** A, double** B, int n) {
    double** C = alloc_matrix(n);
    if (n <= STRASSEN_BASE_CASE) {
        classical_matmul(A, B, C, n);
        return C;
    }
    int half = n / 2;
    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];             A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];      A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];             B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];      B22[i][j] = B[i + half][j + half];
        }

    /* S1 = A21+A22 ; S2 = S1-A11 ; S3 = A11-A21 ; S4 = A12-S2 */
    double** S1 = matrix_add(A21, A22, half);
    double** S2 = matrix_sub(S1, A11, half);
    double** S3 = matrix_sub(A11, A21, half);
    double** S4 = matrix_sub(A12, S2, half);
    /* T1 = B12-B11 ; T2 = B22-T1 ; T3 = B22-B12 ; T4 = T2-B21 */
    double** T1 = matrix_sub(B12, B11, half);
    double** T2 = matrix_sub(B22, T1, half);
    double** T3 = matrix_sub(B22, B12, half);
    double** T4 = matrix_sub(T2, B21, half);

    double** M1 = winograd_recursive(S2, T2, half);
    double** M2 = winograd_recursive(A11, B11, half);
    double** M3 = winograd_recursive(A12, B21, half);
    double** M4 = winograd_recursive(S3, T3, half);
    double** M5 = winograd_recursive(S1, T1, half);
    double** M6 = winograd_recursive(S4, B22, half);
    double** M7 = winograd_recursive(A22, T4, half);

    /* U1 = M2+M3 ; U2 = M2+M5 ; U3 = U2+M4
       C11 = U1 ; C12 = U3+M6 ; C21 = U3+M7 ; C22 = U2+M1 */
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            double U1 = M2[i][j] + M3[i][j];
            double U2 = M2[i][j] + M5[i][j];
            double U3 = U2 + M4[i][j];
            C[i][j]               = U1;
            C[i][j + half]        = U3 + M6[i][j];
            C[i + half][j]        = U3 + M7[i][j];
            C[i + half][j + half] = U2 + M1[i][j];
        }
    }

    double** temps[] = {S1,S2,S3,S4,T1,T2,T3,T4,M1,M2,M3,M4,M5,M6,M7};
    for (int t = 0; t < 15; t++) free_matrix(temps[t], half);
    free_matrix(A11, half); free_matrix(A12, half); free_matrix(A21, half); free_matrix(A22, half);
    free_matrix(B11, half); free_matrix(B12, half); free_matrix(B21, half); free_matrix(B22, half);
    return C;
}

double** winograd_matmul(double** A, double** B, int n) {
    int padded = next_pow2(n);
    if (padded == n) return winograd_recursive(A, B, n);
    double** Apad = alloc_matrix(padded);
    double** Bpad = alloc_matrix(padded);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Apad[i][j] = A[i][j]; Bpad[i][j] = B[i][j]; }
    double** Cpad = winograd_recursive(Apad, Bpad, padded);
    double** C = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cpad[i][j];
    free_matrix(Apad, padded); free_matrix(Bpad, padded); free_matrix(Cpad, padded);
    return C;
}

/* =====================================================================
   4. COPPERSMITH-WINOGRAD — THEORETICAL ONLY
   No practical implementation exists anywhere (it is a "galactic
   algorithm": the hidden constant factors are so large that it never
   outperforms even classical multiplication for any matrix size that
   could ever be materialized). Per the survey's own findings, we do
   NOT fabricate a fake "CW implementation" — that would misrepresent
   the algorithm. Instead this function documents that fact and safely
   falls back to Strassen (the best PRACTICAL sub-cubic algorithm)
   so the driver program can still run end-to-end.
   ===================================================================== */
double** coppersmith_winograd_matmul(double** A, double** B, int n) {
    fprintf(stderr,
        "[NOTE] Coppersmith-Winograd (1990, exponent 2.376) has no practical "
        "implementation on any hardware -- its hidden constants are "
        "astronomically large ('galactic algorithm'). Falling back to "
        "Strassen's algorithm as the best PRACTICAL sub-cubic substitute.\n");
    return strassen_matmul(A, B, n);
}

/* =====================================================================
   5. SCHONHAGE-STRASSEN — INTEGER multiplication, NOT matrix multiplication
   Multiplies two big integers (given as arrays of decimal digits, most
   significant digit first) via a simplified FFT-based convolution,
   demonstrating the algorithm on the domain it actually applies to.
   This is a compact, correctness-focused DFT implementation (O(n^2)
   naive DFT, not the full O(n log n) radix-2 FFT) purely for
   pedagogical/demonstration purposes within this CPU reference project.
   ===================================================================== */
typedef struct { double re, im; } Complex;

static Complex c_add(Complex a, Complex b) { Complex r = {a.re+b.re, a.im+b.im}; return r; }
static Complex c_mul(Complex a, Complex b) {
    Complex r = { a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
    return r;
}

/* Naive O(n^2) DFT / inverse-DFT (sufficient for demonstrating the
   convolution-based multiplication principle at small digit counts) */
static void dft(Complex* in, Complex* out, int n, int invert) {
    for (int k = 0; k < n; k++) {
        Complex sum = {0.0, 0.0};
        for (int t = 0; t < n; t++) {
            double angle = (invert ? 2.0 : -2.0) * M_PI * k * t / n;
            Complex w = { cos(angle), sin(angle) };
            sum = c_add(sum, c_mul(in[t], w));
        }
        if (invert) { sum.re /= n; sum.im /= n; }
        out[k] = sum;
    }
}

void schonhage_strassen_multiply(int* x, int lenx, int* y, int leny,
                                  int* result, int* result_len) {
    int conv_len = lenx + leny - 1;
    int fft_len = 1;
    while (fft_len < conv_len) fft_len <<= 1;
    fft_len <<= 1; /* extra padding to reduce aliasing in the naive DFT */

    Complex* X = (Complex*)calloc(fft_len, sizeof(Complex));
    Complex* Y = (Complex*)calloc(fft_len, sizeof(Complex));
    Complex* Xf = (Complex*)calloc(fft_len, sizeof(Complex));
    Complex* Yf = (Complex*)calloc(fft_len, sizeof(Complex));
    Complex* Zf = (Complex*)calloc(fft_len, sizeof(Complex));
    Complex* Z  = (Complex*)calloc(fft_len, sizeof(Complex));

    /* least-significant digit first for convolution convenience */
    for (int i = 0; i < lenx; i++) X[i].re = x[lenx - 1 - i];
    for (int i = 0; i < leny; i++) Y[i].re = y[leny - 1 - i];

    dft(X, Xf, fft_len, 0);
    dft(Y, Yf, fft_len, 0);
    for (int i = 0; i < fft_len; i++) Zf[i] = c_mul(Xf[i], Yf[i]);
    dft(Zf, Z, fft_len, 1);

    /* Round to nearest integer per convolution coefficient, then
       propagate carries to obtain the final base-10 product */
    int* digits = (int*)calloc(fft_len + 2, sizeof(int));
    long long carry = 0;
    for (int i = 0; i < fft_len; i++) {
        long long val = (long long)llround(Z[i].re) + carry;
        digits[i] = (int)(val % 10);
        carry = val / 10;
        if (digits[i] < 0) { digits[i] += 10; carry -= 1; }
    }
    int pos = fft_len;
    while (carry > 0) { digits[pos++] = (int)(carry % 10); carry /= 10; }

    /* strip leading zeros (from the most-significant end) */
    int top = pos - 1;
    while (top > 0 && digits[top] == 0) top--;

    *result_len = top + 1;
    for (int i = 0; i < *result_len; i++) result[i] = digits[top - i];

    free(X); free(Y); free(Xf); free(Yf); free(Zf); free(Z); free(digits);
}

/* =====================================================================
   6. BLOCK (BLOCKED) MATRIX MULTIPLICATION — cache-tiled, O(n^3)
   ===================================================================== */
void block_matmul(double** A, double** B, double** C, int n, int block_size) {
    fill_zero(C, n);
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                int i_max = (ii + block_size < n) ? ii + block_size : n;
                int j_max = (jj + block_size < n) ? jj + block_size : n;
                int k_max = (kk + block_size < n) ? kk + block_size : n;
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}
