# Exact verification for fourth-order finite-difference Laplacians

This repository contains the four Python experiments supporting the
computer-assisted statements in the accompanying paper on the spectral
reality of nonsymmetric finite-difference Laplacians.  All certification
steps use exact integer, rational, or algebraic-number arithmetic.  No
floating-point eigenvalue computation is used to establish a claim.

## Requirements

- Python 3.10 or newer
- SymPy 1.14.0
- Matplotlib 3.10.8 (used only to reproduce the high-order figure)

Install the pinned dependencies from the repository root:

```bash
python -m pip install .
```

## Experiments

### 1. Bramble--Hubbard characteristic polynomials

```bash
python verify_bh_charpoly.py
```

For each `n` in `{5, 6, 7}`, this script constructs the matrix directly from
its finite-difference stencils, forms its upper and lower parity blocks, and
checks the two characteristic-polynomial formulas from the
Bramble--Hubbard characteristic-polynomial lemma coefficient by coefficient.
The comparison is performed in the exact field
`QQ(exp(I*pi/(n+1)))`; this is important for `n=6`, where a numerical
simplification would not constitute a proof.

### 2. Fourth-order one-sided characteristic polynomials

```bash
python verify_onesided_charpoly.py
```

For `n = 5, 6, 7`, this performs the corresponding exact coefficient checks
for the characteristic-polynomial lemma of the fourth-order one-sided scheme.

### 3. Fourth-order one-sided positive-root counts

```bash
python verify_onesided_root_counts.py
```

For `n = 5, 6, 7`, this applies exact rational Sturm sequences to the upper
and lower characteristic polynomials.  It verifies that they have
`ceil(n/2)` and `floor(n/2)` distinct roots in `(0,+infinity)`, respectively.
It also checks square-freeness and, for `n=5`, verifies

```text
f_L(z) = (z - 13)(z - 36).
```

### 4. High-order nonreal-eigenvalue counts

```bash
python sweep_nonreal_eigs.py
```

For matrix dimension `n=50` and stencil widths
`m=5,7,...,41`, this script constructs the one-sided finite-difference
matrices with `Fraction` coefficients, computes their characteristic
polynomials and exact real-root counts, and reports the number of nonreal
eigenvalues.  The expected sequence is

```text
0, 4, 4, 4, 8, 8, 8, 10, 12, 12, 14, 16, 16, 18, 20, 20, 22, 24, 24.
```

It writes the exact counts and the reproduced plot to `results/` as CSV, PNG,
and PDF files.  Matplotlib is used only for rendering the plot, not for the
spectral certification.

## Scaling convention

The fourth-order experiments omit the common positive factor
`1/(12*h**2)`, as in the spectral-analysis sections of the paper.  The
high-order experiment uses the weights for `+u''`, whereas the paper uses
`-u''`, and it clears rational denominators before computing the
characteristic polynomial.  Thus its matrix differs from the paper's matrix
by a nonzero real scalar.  Such a scaling changes the eigenvalue scale and,
possibly, their signs, but not whether an eigenvalue is real or nonreal.

Every script prints `RESULT: PASS` or completes the stated sweep on success.
The three verification scripts return a nonzero exit status if any exact
identity or root count fails.
