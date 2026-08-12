#!/usr/bin/env python3
"""Verify the Bramble--Hubbard characteristic-polynomial formulas.

This script checks the exceptional dimensions n = 5, 6, 7 occurring in the
paper.  Matrices and characteristic polynomials are constructed exactly.  The
closed formulas are expanded in the cyclotomic algebraic number field
QQ(exp(i*pi/(n+1))) and compared coefficient by coefficient; no floating-point
eigenvalue calculation is used.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence

import sympy as sp
from sympy import QQ


Z = sp.Symbol("z")
CENTRAL_STENCIL = (1, -16, 30, -16, 1)
BH_BOUNDARY_STENCIL = (24, -12, 0, 0)


def bh_matrix(n: int) -> sp.Matrix:
    """Assemble the unscaled n-by-n Bramble--Hubbard matrix exactly."""
    if n < 5:
        raise ValueError("the experiment is intended for n >= 5")

    matrix = sp.zeros(n)
    for column, value in enumerate(BH_BOUNDARY_STENCIL):
        matrix[0, column] = value
        matrix[n - 1, n - 1 - column] = value

    # Grid values at x_0 and x_{n+1} vanish by the Dirichlet condition, so
    # columns outside 0,...,n-1 are omitted.
    for row in range(1, n - 1):
        for offset, value in zip(range(-2, 3), CENTRAL_STENCIL):
            column = row + offset
            if 0 <= column < n:
                matrix[row, column] = value
    return matrix


def parity_blocks(matrix: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Restrict a centrosymmetric matrix to exact integer parity bases."""
    n = matrix.rows
    n_lower = n // 2
    n_upper = n - n_lower

    plus_basis = sp.zeros(n, n_upper)
    minus_basis = sp.zeros(n, n_lower)
    for index in range(n_lower):
        plus_basis[index, index] = 1
        plus_basis[n - 1 - index, index] = 1
        minus_basis[index, index] = 1
        minus_basis[n - 1 - index, index] = -1
    if n % 2:
        plus_basis[n_lower, n_lower] = 1

    plus_rows = list(range(n_lower))
    if n % 2:
        plus_rows.append(n_lower)
    upper = (matrix * plus_basis).extract(plus_rows, range(n_upper))
    lower = (matrix * minus_basis).extract(range(n_lower), range(n_lower))

    if matrix * plus_basis != plus_basis * upper:
        raise AssertionError("the symmetric parity subspace is not invariant")
    if matrix * minus_basis != minus_basis * lower:
        raise AssertionError("the antisymmetric parity subspace is not invariant")
    return upper, lower


def multiply_polynomials(
    left: Sequence[object], right: Sequence[object], field: object
) -> list[object]:
    """Multiply ascending coefficient lists over an algebraic field."""
    result = [field.zero for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] += left_value * right_value
    return result


def spectral_data(n: int) -> tuple[object, list[object], list[object]]:
    """Return lambda_k and the BH residues in an exact cyclotomic field."""
    primitive_root = sp.exp(sp.I * sp.pi / (n + 1))
    field = QQ.algebraic_field(primitive_root)
    root = field.convert(primitive_root)
    two = field.convert(2)

    eigenvalues: list[object] = []
    residues: list[object] = []
    for k in range(1, n + 1):
        cosine = (root**k + root ** (-k)) / two
        eigenvalues.append(4 * cosine**2 - 32 * cosine + 28)
        residues.append(
            field.convert(sp.Rational(16, n + 1))
            * (field.one - cosine) ** 2
            * (field.one - cosine**2)
        )
    return field, eigenvalues, residues


def product_of_linear_factors(
    field: object, eigenvalues: Sequence[object], indices: Iterable[int]
) -> list[object]:
    result = [field.one]
    for index in indices:
        result = multiply_polynomials(
            result, [-eigenvalues[index], field.one], field
        )
    return result


def closed_formula(
    field: object,
    eigenvalues: Sequence[object],
    residues: Sequence[object],
    indices: Sequence[int],
) -> list[object]:
    """Expand prod(z-lambda_k)+sum alpha_k prod_{j != k}(z-lambda_j)."""
    result = product_of_linear_factors(field, eigenvalues, indices)
    for omitted in indices:
        quotient = product_of_linear_factors(
            field,
            eigenvalues,
            [index for index in indices if index != omitted],
        )
        for degree, coefficient in enumerate(quotient):
            result[degree] += residues[omitted] * coefficient
    return result


def coefficients_agree(
    polynomial: sp.Poly, field: object, ascending_formula: Sequence[object]
) -> bool:
    actual = [field.convert(value) for value in reversed(polynomial.all_coeffs())]
    return len(actual) == len(ascending_formula) and all(
        left == right for left, right in zip(actual, ascending_formula)
    )


def verify_dimension(n: int) -> None:
    matrix = bh_matrix(n)
    upper, lower = parity_blocks(matrix)
    upper_polynomial = upper.charpoly(Z).as_poly()
    lower_polynomial = lower.charpoly(Z).as_poly()

    field, eigenvalues, residues = spectral_data(n)
    upper_indices = list(range(0, n, 2))  # k = 1, 3, 5, ... in the paper
    lower_indices = list(range(1, n, 2))  # k = 2, 4, 6, ... in the paper

    upper_expected = closed_formula(
        field, eigenvalues, residues, upper_indices
    )
    lower_expected = closed_formula(
        field, eigenvalues, residues, lower_indices
    )
    if not coefficients_agree(upper_polynomial, field, upper_expected):
        raise AssertionError(f"n={n}: upper characteristic formula failed")
    if not coefficients_agree(lower_polynomial, field, lower_expected):
        raise AssertionError(f"n={n}: lower characteristic formula failed")

    print(
        f"PASS  n={n}: upper and lower BH characteristic-polynomial "
        "formulas agree coefficient by coefficient"
    )


def main() -> int:
    try:
        for n in (5, 6, 7):
            verify_dimension(n)
    except Exception as error:
        print(f"FAIL  {type(error).__name__}: {error}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
