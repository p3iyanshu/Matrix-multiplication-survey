# 5. Schonhage-Strassen Algorithm (Integer Multiplication)

| Field | Details |
|---|---|
| **Author(s)** | Arnold Schonhage, Volker Strassen |
| **Year** | 1971 |
| **Time Complexity** | O(n log n log log n) -- for multiplying large INTEGERS, not matrices |
| **Approach** | Serial |
| **Original Paper Link** | https://doi.org/10.1007/BF02242355 (Springer -- Computing, 7(3-4), 281-292) |

> **Implementation note:** This is an **integer**-multiplication algorithm, not a matrix-multiplication algorithm -- it is included per the survey's requirement, but implemented on its actual domain (big-integer multiplication via FFT-based convolution) rather than being forced into a matrix-multiplication interface it was never designed for.

## Pseudocode

```
FUNCTION SchonhageStrassen(x, y):            // x, y are N-digit integers
    Split x, y into digit blocks; treat as polynomial coefficient vectors
    X_hat = FFT(x) ; Y_hat = FFT(y)          // modular FFT over Z/(2^N+1)Z
    Z_hat = X_hat .* Y_hat                    // pointwise multiplication (convolution theorem)
    z = InverseFFT(Z_hat)
    Perform carry propagation on z to obtain the final integer product
    RETURN z
```

## CPU (C) Implementation

Full compilable source: [`src/algo_01_to_06.c`](../src/) (build with `cd ../src && make && ./matmul_test`).

```c
/* =====================================================================
   5. SCHONHAGE-STRASSEN â INTEGER multiplication, NOT matrix multiplication
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

```

---
[<- Back to index](../README.md)
