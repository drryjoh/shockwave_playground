#!/usr/bin/env python3
"""
SPLAY – Tutorial comparison plot.

Reads the final combined CSV snapshot from each of the five tutorial cases
and overlays them for comparison.

Usage
-----
    python tutorials/argon_target_shock/plot_compare.py [output_dir]

Default output_dir is "output/".
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"

CASES = [
    ("argon_shock_central_navier_stokes",  "Central NS",          "k",            "-",   2.5),
    ("argon_shock_muscl_euler",            "MUSCL Euler n=500",   "tab:blue",     "--",  1.5),
    ("argon_shock_ppm_euler",              "PPM Euler n=500",     "tab:orange",   "--",  1.5),
    ("argon_shock_muscl_navier_stokes",    "MUSCL NS n=500",      "tab:green",    ":",   1.5),
    ("argon_shock_ppm_navier_stokes",      "PPM NS n=500",        "tab:red",      ":",   1.5),
    # DG p=2 cases: run with argon_shock_dg_p2_n{500,1000}_quick.yml
    ("argon_shock_dg_p2_n500_quick",       "DG p=2 n=500",        "tab:purple",   "-",   2.0),
    ("argon_shock_dg_p2_n1000_quick",      "DG p=2 n=1000",       "darkviolet",   "-",   2.0),
]

VARS = ["rho", "u", "p", "T", "Mach", "mu"]
LABELS = {
    "x":    "x [m]",
    "rho":  r"$\rho$ [kg/m³]",
    "u":    "u [m/s]",
    "p":    "p [Pa]",
    "T":    "T [K]",
    "Mach": "Mach",
    "mu":   r"$\mu$ [Pa·s]",
    "kappa": r"$\kappa$ [W/(m·K)]",
}


def find_latest_csv(case_dir):
    """Find the most recent combined CSV snapshot in case_dir/snapshots/."""
    snap_dir = os.path.join(case_dir, "snapshots")
    pattern  = os.path.join(snap_dir, "step_*_combined.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_csv(path):
    """Load a SPLAY CSV snapshot into a dict of arrays."""
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            header = [h.strip() for h in line.strip().split(",")]
            break
        rows = []
        for line in f:
            rows.append([float(v) for v in line.strip().split(",")])
    arr = np.array(rows)
    for k, col in enumerate(header):
        data[col] = arr[:, k]
    return data


def shock_thickness(x, q, lo=0.1, hi=0.9):
    """
    Estimate shock thickness as x-distance between 10% and 90% of
    the total variation in quantity q.
    """
    q_min, q_max = q.min(), q.max()
    if q_max == q_min:
        return 0.0
    lo_val = q_min + lo * (q_max - q_min)
    hi_val = q_min + hi * (q_max - q_min)
    x_lo = x[np.argmin(np.abs(q - lo_val))]
    x_hi = x[np.argmin(np.abs(q - hi_val))]
    return abs(x_hi - x_lo)


def main():
    print(f"Reading output from: {OUTPUT_DIR}")

    datasets = {}
    for case_name, label, color, ls, lw in CASES:
        case_dir = os.path.join(OUTPUT_DIR, case_name)
        csv_path = find_latest_csv(case_dir)
        if csv_path is None:
            print(f"  [skip] No combined CSV found for {case_name}")
            continue
        print(f"  [{case_name}] {csv_path}")
        datasets[case_name] = load_csv(csv_path)

    if not datasets:
        print("No data found.  Run the tutorial cases first.")
        sys.exit(1)

    # ── Comparison plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    plot_vars = ["rho", "u", "p", "T", "Mach", "mu"]

    # Auto-detect shock x-centre from density in the first available dataset.
    shock_centre_um = None
    for case_name, *_ in CASES:
        if case_name in datasets:
            d0   = datasets[case_name]
            rho  = d0["rho"]
            x_um = d0["x"] * 1e6
            mid  = 0.5 * (rho.min() + rho.max())
            idx  = np.argmin(np.abs(rho - mid))
            shock_centre_um = x_um[idx]
            break

    for ax, var in zip(axes, plot_vars):
        for case_name, label, color, ls, lw in CASES:
            if case_name not in datasets:
                continue
            d = datasets[case_name]
            x = d["x"] * 1e6  # convert to micrometres for readability
            ax.plot(x, d[var], color=color, ls=ls, lw=lw, label=label)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel(LABELS.get(var, var))
        ax.set_title(LABELS.get(var, var))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if shock_centre_um is not None:
            ax.set_xlim([shock_centre_um - 0.2, shock_centre_um + 0.2])

    fig.suptitle("SPLAY: Argon shock comparison (M≈5.03)", fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "argon_shock_comparison.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_png}")
    plt.show()

    # ── Shock thickness estimate ──────────────────────────────────────────────
    print("\nShock thickness estimates (10%–90% density variation):")
    print(f"  {'Case':<45}  {'delta_x [µm]':>12}  {'dx [µm]':>8}  {'cells':>6}")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d = datasets[case_name]
        x = d["x"]
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        thick = shock_thickness(x, d["rho"])
        ncells = thick / dx if dx > 0 else 0.0
        print(f"  {label:<45}  {thick*1e6:>12.4f}  {dx*1e6:>8.4f}  {ncells:>6.1f}")

    # ── Min/max sanity check ─────────────────────────────────────────────────
    print("\nMin/max sanity check:")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d = datasets[case_name]
        nonphys = []
        if np.any(d["rho"] <= 0): nonphys.append("rho<=0")
        if np.any(d["p"]   <= 0): nonphys.append("p<=0")
        if np.any(d["T"]   <= 0): nonphys.append("T<=0")
        status = "OK" if not nonphys else "NONPHYSICAL: " + ", ".join(nonphys)
        print(f"  {label:<45} {status}")


if __name__ == "__main__":
    main()
