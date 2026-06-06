#!/usr/bin/env python3
"""
SPLAY – Basic test cases plotter.

Plots pressure, density, and temperature vs x for each of the three
basic test cases, with a colour gradient showing the time evolution.

Usage (from repo root or any directory):
    python tutorials/basic_tests/plot.py
    python tutorials/basic_tests/plot.py --save basic_tests.png
    python tutorials/basic_tests/plot.py --output_dir /path/to/output
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

CASES = [
    ("basic_constant_state",        "Constant state",         "Blues"),
    ("basic_contact_disc",          "Contact discontinuity",  "Greens"),
    ("basic_pressure_perturbation", "Pressure perturbation",  "Oranges"),
]

VARS = [
    ("p",   "Pressure [Pa]"),
    ("rho", "Density [kg/m³]"),
    ("T",   "Temperature [K]"),
]


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_csv(path):
    rows, header = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            rows.append([float(v) for v in line.split(",")])
    if not rows:
        return None
    arr = np.array(rows)
    d   = {col: arr[:, k] for k, col in enumerate(header)}
    idx = np.argsort(d["x"])
    return {k: v[idx] for k, v in d.items()}


def meta_from_comment(path):
    with open(path) as f:
        line = f.readline().strip()
    t, s = None, None
    for tok in line.split():
        if tok.startswith("time="): t = float(tok.split("=")[1])
        if tok.startswith("step="): s = int(tok.split("=")[1])
    return t, s


def load_all_snapshots(case_dir):
    pattern = os.path.join(case_dir, "snapshots", "step_*_combined.csv")
    files   = sorted(glob.glob(pattern))
    snaps   = []
    for f in files:
        t, s = meta_from_comment(f)
        d    = load_csv(f)
        if d is not None and t is not None:
            snaps.append((t, s, d))
    snaps.sort(key=lambda x: x[0])
    return snaps


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SPLAY basic-tests plotter")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                        help="Directory containing output/<case_name>/ folders")
    parser.add_argument("--save", default="",
                        help="Save figure to file instead of displaying (e.g. out.png)")
    args = parser.parse_args()

    # Load data for each case
    all_data = {}
    for case_name, label, cmap_name in CASES:
        snaps = load_all_snapshots(os.path.join(args.output_dir, case_name))
        if not snaps:
            print(f"  [skip] {label} ({case_name}): no snapshots found.")
            print(f"         Run first: bash tutorials/basic_tests/run.sh --no-plot")
            continue
        all_data[case_name] = (label, cmap_name, snaps)
        print(f"  {label}: {len(snaps)} snapshots  "
              f"t={snaps[0][0]:.2e} → {snaps[-1][0]:.2e} s")

    if not all_data:
        print("\nNothing to plot.")
        sys.exit(1)

    n_cases = len(all_data)
    n_vars  = len(VARS)

    # Layout: n_vars rows × n_cases columns, with a narrow colorbar column per case
    col_widths = []
    for _ in range(n_cases):
        col_widths += [5.5, 0.28]

    fig = plt.figure(figsize=(sum(col_widths) + 0.4 * n_cases, 3.0 * n_vars))
    fig.suptitle("SPLAY — Basic test cases: time evolution",
                 fontsize=13, fontweight="bold", y=0.999)
    gs = gridspec.GridSpec(n_vars, 2 * n_cases, figure=fig,
                           width_ratios=col_widths,
                           hspace=0.48, wspace=0.10)

    for col_idx, (case_name, (label, cmap_name, snaps)) in enumerate(all_data.items()):
        t_all        = np.array([s[0] for s in snaps])
        t_min, t_max = t_all.min(), t_all.max()
        # Avoid degenerate colour norm when only one snapshot
        if t_min == t_max:
            t_max = t_min + 1.0

        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=t_min, vmax=t_max)

        plot_col = col_idx * 2
        cbar_col = col_idx * 2 + 1

        for row, (col_key, ylabel) in enumerate(VARS):
            ax = fig.add_subplot(gs[row, plot_col])

            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", pad=6)

            for t, step, d in snaps:
                color   = cmap(norm(t))
                is_edge = (t == t_min or t == t_max)
                lw      = 2.0 if is_edge else 0.9
                alpha   = 1.0 if is_edge else 0.55
                ax.plot(d["x"], d[col_key], color=color, lw=lw, alpha=alpha)

            ax.set_xlabel("x  [m]", fontsize=9)
            ax.set_ylabel(ylabel,   fontsize=9)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=8)

        # Single colorbar spanning all variable rows for this case
        cbar_ax = fig.add_subplot(gs[:, cbar_col])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("time [s]", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
