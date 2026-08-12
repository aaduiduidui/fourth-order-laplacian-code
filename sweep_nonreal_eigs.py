from fractions import Fraction
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from sympy import Matrix, gcd


def characteristic_polynomial(matrix):
    denom_lcm = 1
    for row in matrix:
        for value in row:
            denom_lcm = math.lcm(denom_lcm, value.denominator)

    integer_matrix = [
        [value.numerator * (denom_lcm // value.denominator) for value in row]
        for row in matrix
    ]
    sympy_matrix = Matrix(integer_matrix)
    char_poly = sympy_matrix.charpoly()
    num_real_roots = char_poly.count_roots()
    return char_poly, int(num_real_roots)


def fornberg_origin(z, x, m=0):
    n = len(x)
    if m >= n:
        m = n - 1

    coeffs = [[Fraction(0) for _ in range(m + 1)] for _ in range(n)]
    coeffs[0][0] = Fraction(1)

    c1 = Fraction(1)
    c4 = Fraction(x[0] - z)

    for i in range(1, n):
        mn = min(i, m)
        c2 = Fraction(1)
        c5 = c4
        c4 = Fraction(x[i] - z)
        for j in range(i):
            c3 = Fraction(x[i] - x[j])
            c2 *= c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    coeffs[i][s] = c1 * (
                        s * coeffs[i - 1][s - 1] - c5 * coeffs[i - 1][s]
                    ) / c2
                coeffs[i][0] = -c1 * c5 * coeffs[i - 1][0] / c2
            for s in range(mn, 0, -1):
                coeffs[j][s] = (c4 * coeffs[j][s] - s * coeffs[j][s - 1]) / c3
            coeffs[j][0] = c4 * coeffs[j][0] / c3
        c1 = c2

    return [[coeffs[j][i] for j in range(n)] for i in range(m + 1)]


def fd_mat_dirichlet_origin(m, n):
    mat_temp = [[Fraction(0) for _ in range(n + 2)] for _ in range(n + 2)]
    alpha = [Fraction(i) for i in range(-(m - 1) // 2, (m - 1) // 2 + 1)]
    coe = [fornberg_origin(i, alpha, 2)[2] for i in alpha[: (m + 1) // 2]]
    half_len = len(coe)
    for i in range(half_len - 1):
        coe.append(coe[half_len - 2 - i][::-1])

    row_idx = 1
    coe_idx = 1
    while row_idx <= (m - 1) // 2:
        mat_temp[row_idx - 1][0:m] = coe[coe_idx - 1]
        row_idx += 1
        coe_idx += 1

    while row_idx <= n + 2 - (m - 1) // 2:
        left = row_idx - (m + 1) // 2
        right = row_idx + (m - 1) // 2
        mat_temp[row_idx - 1][left:right] = coe[(m + 1) // 2 - 1]
        row_idx += 1

    coe_idx += 1
    while row_idx <= n + 2:
        start = n + 2 - m
        mat_temp[row_idx - 1][start : start + m] = coe[coe_idx - 1]
        row_idx += 1
        coe_idx += 1

    return [row[1 : n + 1] for row in mat_temp[1 : n + 1]]


def sweep_counts(n=50, k_min=2, k_max=20):
    rows = []
    for k in range(k_min, k_max + 1):
        m = 2 * k + 1
        char_poly, real_count = characteristic_polynomial(fd_mat_dirichlet_origin(m, n))
        repeated = gcd(char_poly, char_poly.diff()).degree() > 0
        if repeated:
            raise RuntimeError(f"Repeated roots detected for m={m}, n={n}")
        rows.append(
            {
                "n": n,
                "k": k,
                "m": m,
                "degree": int(char_poly.degree()),
                "real_eigenvalues": real_count,
                "nonreal_eigenvalues": int(char_poly.degree()) - real_count,
            }
        )
        print(
            f"m={m:2d} | real={real_count:2d} | nonreal={int(char_poly.degree()) - real_count:2d}",
            flush=True,
        )
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "n",
                "k",
                "m",
                "degree",
                "real_eigenvalues",
                "nonreal_eigenvalues",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, png_path, pdf_path):
    ms = [row["m"] for row in rows]
    nonreal = [row["nonreal_eigenvalues"] for row in rows]

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(
            figsize=(3.5, 2.45),
            dpi=220,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(
            ms,
            nonreal,
            color="black",
            linewidth=0.9,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )

        ax.set_xlabel(r"Stencil width $m = 2\ell + 1$")
        ax.set_ylabel("Number of nonreal eigenvalues")
        ax.xaxis.set_major_locator(FixedLocator(ms[::2]))
        ax.xaxis.set_minor_locator(FixedLocator(ms))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", which="major", color="0.85", linestyle=":", linewidth=0.5)
        ax.margins(x=0.03, y=0.14)

        for i, (m, y) in enumerate(zip(ms, nonreal)):
            if i > 0 and y == nonreal[i - 1]:
                continue
            ax.annotate(
                f"{y}",
                (m, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
            )

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "n50_nonreal_eigs.csv"
    png_path = out_dir / "n50_nonreal_eigs.png"
    pdf_path = out_dir / "n50_nonreal_eigs.pdf"

    rows = sweep_counts()
    write_csv(rows, csv_path)
    make_plot(rows, png_path, pdf_path)

    print(f"Saved data to {csv_path}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
