#!/usr/bin/env python3
"""
SPLAY – Flattening effect comparison plot.

Compares PPM and MUSCL results with and without CW84/PeleC-style shock
flattening, for both Euler and Navier-Stokes cases.

Layout: 3 variables (rho, p, T)  x  2 panels (Euler comparison | NS comparison)
Each panel overlays: no-flatten (solid) vs flatten (dashed) for PPM and MUSCL.

Usage (from repo root):
    python tutorials/argon_target_shock/plot_flatten.py
    python tutorials/argon_target_shock/plot_flatten.py --save flatten.png
    python tutorials/argon_target_shock/plot_flatten.py --output_dir /path/to/output
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

# (case_name, label, color, linestyle, panel)
# panel: "euler" or "ns"
CASES = [
    ("argon_shock_ppm_euler_quick",          "PPM Euler",              "tab:orange", "-",  "euler"),
    ("argon_shock_ppm_euler_flatten_quick",  "PPM Euler + flatten",    "tab:orange", "--", "euler"),
    ("argon_shock_muscl_euler_quick",        "MUSCL Euler",            "tab:blue",   "-",  "euler"),
    ("argon_shock_muscl_euler_flatten_quick","MUSCL Euler + flatten",  "tab:blue",   "--", "euler"),
    ("argon_shock_ppm_ns_quick",             "PPM NS",                 "tab:red",    "-",  "ns"),
    ("argon_shock_ppm_ns_flatten_quick",     "PPM NS + flatten",       "tab:red",    "--", "ns"),
    ("argon_shock_muscl_ns_quick",           "MUSCL NS",               "tab:green",  "-",  "ns"),
    ("argon_shock_muscl_ns_flatten_quick",   "MUSCL NS + flatten",     "tab:green",  "--", "ns"),
]

VARS = [
    ("rho", r"Density  [kg/m³]"),
    ("p",   "Pressure  [Pa]"),
    ("T",   "Temperature  [K]"),
]

PANEL_TITLES = {
    "euler": "Euler (inviscid) — flatten on vs off",
    "ns":    "Navier-Stokes — flatten on vs off  (PeleC-like = dashed)",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────
def find_latest_csv(case_dir):
    files = sorted(glob.glob(os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


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


def shock_thickness(x, rho):
    rmin, rmax = rho.min(), rho.max()
    if rmax == rmin:
        return 0.0
    lo = rmin + 0.1 * (rmax - rmin)
    hi = rmin + 0.9 * (rmax - rmin)
    return abs(x[np.argmin(np.abs(rho - hi))] - x[np.argmin(np.abs(rho - lo))])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SPLAY flattening comparison plot")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save", default="", help="Save figure to file (png/pdf)")
    args = parser.parse_args()

    # Load data
    datasets = {}
    for case_name, label, color, ls, panel in CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip] {label}: no snapshots in {args.output_dir}/{case_name}")
            continue
        d = load_csv(csv)
        if d is not None:
            datasets[case_name] = d
            print(f"  {label}: {csv.split('/')[-1]}")

    if not datasets:
        print("\nNo data found. Run the cases first:")
        print("  bash tutorials/argon_target_shock/run_flatten.sh")
        sys.exit(1)

    panels = ["euler", "ns"]
    n_vars   = len(VARS)
    n_panels = len(panels)

    col_widths = [5.5] * n_panels
    fig = plt.figure(figsize=(sum(col_widths) + 1.5, 3.2 * n_vars))
    fig.suptitle("SPLAY — CW84/PeleC flattening effect on argon shock (M≈5.03)",
                 fontsize=13, fontweight="bold", y=0.999)
    gs = gridspec.GridSpec(n_vars, n_panels, figure=fig,
                           hspace=0.44, wspace=0.28)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_panels)] for r in range(n_vars)]

    for row, (col_key, ylabel) in enumerate(VARS):
        for col, panel in enumerate(panels):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(PANEL_TITLES[panel], fontsize=10, fontweight="bold", pad=6)

            plotted = False
            for case_name, label, color, ls, case_panel in CASES:
                if case_panel != panel or case_name not in datasets:
                    continue
                d  = datasets[case_name]
                x_um = d["x"] * 1e6
                lw = 2.0 if ls == "-" else 1.5
                ax.plot(x_um, d[col_key], color=color, ls=ls, lw=lw,
                        label=label, alpha=0.9)
                plotted = True

            ax.set_xlabel("x  [µm]", fontsize=9)
            ax.set_ylabel(ylabel,    fontsize=9)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=8)
            if plotted and row == 0:
                ax.legend(fontsize=7.5, loc="upper right")

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()

    # ── Shock thickness table ─────────────────────────────────────────────────
    print("\nShock thickness (10–90% density rise):")
    print(f"  {'Case':<45}  {'delta_x [nm]':>13}  {'cells':>6}")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d     = datasets[case_name]
        x     = d["x"]
        dx    = x[1] - x[0] if len(x) > 1 else 1.0
        thick = shock_thickness(x, d["rho"])
        print(f"  {label:<45}  {thick*1e9:>13.1f}  {thick/dx:>6.1f}")


if __name__ == "__main__":
    main()
