#!/usr/bin/env python3
"""
SPLAY – Single-case profile plotter.

Reads a SPLAY CSV snapshot and plots density, velocity, pressure, temperature,
Mach number, viscosity, and conductivity.

Usage
-----
    python scripts/plot_profile.py <csv_file> [--title "My case"]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            header = [h.strip() for h in line.strip().split(",")]
            break
        rows = []
        for line in f:
            if line.strip():
                rows.append([float(v) for v in line.strip().split(",")])
    if not rows:
        raise RuntimeError(f"No data in {path}")
    arr = np.array(rows)
    for k, col in enumerate(header):
        data[col] = arr[:, k]
    return data


def shock_thickness_10_90(x, q):
    q_min, q_max = q.min(), q.max()
    if q_max == q_min:
        return 0.0
    lo_val = q_min + 0.1 * (q_max - q_min)
    hi_val = q_min + 0.9 * (q_max - q_min)
    x_lo = x[np.argmin(np.abs(q - lo_val))]
    x_hi = x[np.argmin(np.abs(q - hi_val))]
    return abs(x_hi - x_lo)


def main():
    parser = argparse.ArgumentParser(description="SPLAY profile plotter")
    parser.add_argument("csv_file", help="Path to combined CSV snapshot")
    parser.add_argument("--title",  default="", help="Plot title")
    parser.add_argument("--out",    default="", help="Save figure to file (png/pdf)")
    args = parser.parse_args()

    d = load_csv(args.csv_file)
    x_um = d["x"] * 1e6  # micrometres

    # Sort by x (in case ranks are interleaved)
    idx = np.argsort(x_um)
    x_um = x_um[idx]
    for k in d:
        d[k] = d[k][idx]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    plots = [
        ("rho",   r"$\rho$ [kg/m³]",     "tab:blue"),
        ("u",     "u [m/s]",             "tab:orange"),
        ("p",     "p [Pa]",              "tab:green"),
        ("T",     "T [K]",              "tab:red"),
        ("Mach",  "Mach",               "purple"),
        ("mu",    r"$\mu$ [Pa·s]",       "tab:brown"),
        ("kappa", r"$\kappa$ [W/(m·K)]", "tab:pink"),
    ]

    for ax, (col, ylabel, color) in zip(axes, plots):
        if col not in d:
            ax.set_visible(False)
            continue
        ax.plot(x_um, d[col], color=color, lw=1.5)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)

    title = args.title or os.path.basename(args.csv_file)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    # Print statistics
    print(f"\nSnapshot statistics:")
    print(f"  Cells      : {len(d['x'])}")
    print(f"  x range    : {d['x'].min():.4e} – {d['x'].max():.4e} m")
    for col, label, *_ in plots:
        if col in d:
            print(f"  {label.replace('$','').replace(chr(92),''):20s}: "
                  f"{d[col].min():.4e} – {d[col].max():.4e}")

    thick = shock_thickness_10_90(d["x"], d["rho"])
    dx    = d["x"][1] - d["x"][0] if len(d["x"]) > 1 else 1.0
    print(f"\n  Shock thickness (10–90% rho): {thick*1e6:.4f} µm  "
          f"({thick/dx:.1f} cells)")

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"\nSaved: {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
