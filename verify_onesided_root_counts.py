#!/usr/bin/env python3
"""Verify the positive-root counts used in the one-sided block theorem.

The experiment covers n = 5, 6, 7.  It constructs the upper and lower parity
blocks over the integers, computes their characteristic polynomials exactly,
and applies a rational Sturm sequence on (0,+infinity).  A square-freeness
check ensures that the root counts also certify distinctness.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import sympy as sp
from sympy import QQ


Z = sp.Symbol("z")
CENTRAL_STENCIL = (1, -16, 30, -16, 1)
ONESIDED_BOUNDARY_STENCIL = (20, -6, -4, 1)


def onesided_matrix(n: int) -> sp.Matrix:
    if n < 5:
        raise ValueError("the experiment is intended for n >= 5")
    matrix = sp.zeros(n)
    for column, value in enumerate(ONESIDED_BOUNDARY_STENCIL):
        matrix[0, column] = value
        matrix[n - 1, n - 1 - column] = value
    for row in range(1, n - 1):
        for offset, value in zip(range(-2, 3), CENTRAL_STENCIL):
            column = row + offset
            if 0 <= column < n:
                matrix[row, column] = value
    return matrix


def parity_blocks(matrix: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
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


def sign_variations(signs: Iterable[int]) -> int:
    nonzero = [sign for sign in signs if sign]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def exact_sign(value: sp.Expr) -> int:
    value = sp.cancel(value)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def positive_sturm_count(polynomial: sp.Poly) -> int:
    """Count distinct roots in (0,+infinity) using exact rational signs."""
    if polynomial.eval(0) == 0:
        raise AssertionError("the characteristic polynomial vanishes at zero")
    sequence = [
        sp.Poly(item, Z, domain=QQ)
        for item in sp.sturm(polynomial.as_expr(), Z)
    ]
    signs_at_zero = [exact_sign(item.eval(0)) for item in sequence]
    signs_at_infinity = [exact_sign(item.LC()) for item in sequence]
    return sign_variations(signs_at_zero) - sign_variations(signs_at_infinity)


def verify_block(polynomial: sp.Poly, expected: int, label: str) -> None:
    common_factor = sp.gcd(polynomial, polynomial.diff())
    if common_factor.degree() != 0:
        raise AssertionError(f"{label}: the polynomial has repeated roots")
    count = positive_sturm_count(polynomial)
    if count != expected:
        raise AssertionError(
            f"{label}: expected {expected} positive roots, obtained {count}"
        )


def verify_dimension(n: int) -> None:
    upper, lower = parity_blocks(onesided_matrix(n))
    upper_polynomial = upper.charpoly(Z).as_poly()
    lower_polynomial = lower.charpoly(Z).as_poly()
    n_upper = (n + 1) // 2
    n_lower = n // 2

    verify_block(upper_polynomial, n_upper, f"n={n} upper block")
    verify_block(lower_polynomial, n_lower, f"n={n} lower block")
   
    print(
        f"PASS  n={n}: upper has {n_upper} and lower has {n_lower} "
        "distinct roots in (0,+infinity)"
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
