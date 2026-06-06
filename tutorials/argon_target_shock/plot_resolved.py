#!/usr/bin/env python3
"""
SPLAY – Resolved shock structure plot.

Compares PPM+NS (resolved viscous structure), PPM+NS+flatten (PeleC-like
captured shock), Central+NS, and MUSCL+NS — all from the fine (N=2000) cases.

The x-axis is centred at the shock interface (argmin |T - 1000 K|) so profiles
from different runs can be meaningfully overlaid even if the shock has drifted
slightly.  Window: ±0.20 µm.

Usage (from repo root):
    python tutorials/argon_target_shock/plot_resolved.py
    python tutorials/argon_target_shock/plot_resolved.py --save resolved.png
    python tutorials/argon_target_shock/plot_resolved.py --window 0.5
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

# (case_name, label, color, linestyle, linewidth)
# Central NS is first (black solid) as reference.
CASES = [
    ("argon_shock_central_navier_stokes",         "Central NS",             "k",          "-",  2.5),
    ("argon_shock_ppm_navier_stokes",             "PPM NS",                 "tab:orange", "-",  1.8),
    ("argon_shock_ppm_navier_stokes_flatten",     "PPM NS + flatten",       "tab:red",    "--", 1.8),
    ("argon_shock_muscl_navier_stokes",           "MUSCL NS",               "tab:blue",   "-",  1.8),
    ("argon_shock_muscl_navier_stokes_flatten",   "MUSCL NS + flatten",     "tab:cyan",   "--", 1.8),
]

VARS = [
    ("rho", r"Density  [kg/m³]"),
    ("p",   "Pressure  [Pa]"),
    ("T",   "Temperature  [K]"),
]

T_SHOCK_MID = 1000.0   # K — temperature level used to locate shock centre


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


def shock_centre(d, T_mid=T_SHOCK_MID):
    """Return x-coordinate [m] where T is closest to T_mid."""
    return d["x"][np.argmin(np.abs(d["T"] - T_mid))]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SPLAY resolved shock structure plot")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save",   default="", help="Save to file (png/pdf)")
    parser.add_argument("--window", type=float, default=0.20,
                        help="Half-width of x window in µm (default 0.20)")
    args = parser.parse_args()

    # Load data
    datasets = {}
    for case_name, label, *_ in CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip] {label}: no snapshots in {args.output_dir}/{case_name}")
            continue
        d = load_csv(csv)
        if d is not None:
            datasets[case_name] = d
            xc_um = shock_centre(d) * 1e6
            print(f"  {label}: {csv.split('/')[-1]}  shock@{xc_um:.3f} µm")

    if not datasets:
        print("\nNo data found. Run the fine cases first:")
        print("  bash tutorials/argon_target_shock/fine/run.sh --no-plot")
        sys.exit(1)

    n_vars = len(VARS)
    fig, axes = plt.subplots(1, n_vars, figsize=(5.0 * n_vars, 5.0))
    if n_vars == 1:
        axes = [axes]

    fig.suptitle(
        "SPLAY — Resolved argon shock structure (N=2000, M≈5.03)\n"
        "x centred at shock interface  (T = 1000 K level)",
        fontsize=12, fontweight="bold"
    )

    half_um = args.window  # µm

    for ax, (col_key, ylabel) in zip(axes, VARS):
        for case_name, label, color, ls, lw in CASES:
            if case_name not in datasets:
                continue
            d   = datasets[case_name]
            xc  = shock_centre(d)            # m
            x_r = (d["x"] - xc) * 1e6       # relative position in µm
            mask = np.abs(x_r) <= half_um
            ax.plot(x_r[mask], d[col_key][mask],
                    color=color, ls=ls, lw=lw, label=label, alpha=0.9)

        ax.set_xlabel("x − x_shock  [µm]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-half_um, half_um)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=9)
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.4)  # mark shock centre

    axes[0].legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()

    # ── Shock thickness summary ───────────────────────────────────────────────
    print(f"\nShock thickness (10–90%% density, within ±{half_um:.2f} µm window):")
    print(f"  {'Case':<45}  {'delta_x [nm]':>13}  {'cells (dx=3.8nm)':>18}")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d    = datasets[case_name]
        xc   = shock_centre(d)
        x_r  = (d["x"] - xc) * 1e6
        mask = np.abs(x_r) <= half_um
        rho  = d["rho"][mask]
        x_w  = d["x"][mask]
        rmin, rmax = rho.min(), rho.max()
        if rmax == rmin:
            continue
        lo = rmin + 0.1 * (rmax - rmin)
        hi = rmin + 0.9 * (rmax - rmin)
        dx = x_w[1] - x_w[0] if len(x_w) > 1 else 1.0
        thick = abs(x_w[np.argmin(np.abs(rho - hi))] - x_w[np.argmin(np.abs(rho - lo))])
        print(f"  {label:<45}  {thick*1e9:>13.1f}  {thick/dx:>18.1f}")


if __name__ == "__main__":
    main()
