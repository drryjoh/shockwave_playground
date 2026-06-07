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
    ("argon_shock_central_navier_stokes",               "Central NS",                  "k",          "-",  2.5),
    ("argon_shock_ppm_navier_stokes",                   "PPM NS",                      "tab:orange", "-",  1.8),
    ("argon_shock_ppm_navier_stokes_flatten",           "PPM NS + flatten",            "tab:red",    "--", 1.8),
    ("argon_shock_ppm_navier_stokes_flatten_split",     "PPM NS + flatten + split",    "tab:purple", ":",  2.0),
    ("argon_shock_ppm_pele_navier_stokes",              "PPM_pele NS",                 "tab:pink",   "-.", 2.0),
    ("argon_shock_muscl_navier_stokes",                 "MUSCL NS",                    "tab:blue",   "-",  1.8),
    ("argon_shock_muscl_navier_stokes_flatten",         "MUSCL NS + flatten",          "tab:cyan",   "--", 1.8),
]

VARS = [
    ("rho", r"Density  [kg/m³]"),
    ("p",   "Pressure  [Pa]"),
    ("T",   "Temperature  [K]"),
]

T_SHOCK_MID = 1500.0   # K — temperature level used to locate shock centre


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

    # Load MD data — x already in µm, shock-centred; col 1 = T [K]
    md_path = os.path.join(SCRIPT_DIR, "MD_data.txt")
    md_data = None
    if os.path.exists(md_path):
        md_arr = np.loadtxt(md_path, delimiter=",")
        md_data = {"x": md_arr[:, 0], "T": md_arr[:, 1]}  # x in µm, already centred
        print(f"  MD data: {len(md_arr)} points loaded from {md_path}")
    else:
        print(f"  [skip] MD data not found at {md_path}")

    # Load DG (p=2) data — x in metres, needs shock-centering
    dg_dir = os.path.join(SCRIPT_DIR, "DG")
    dg_data = None
    dg_x_files = sorted(glob.glob(os.path.join(dg_dir, "x_*.npy")))
    dg_T_files = sorted(glob.glob(os.path.join(dg_dir, "Temperature_*.npy")))
    if dg_x_files and dg_T_files:
        dg_x = np.load(dg_x_files[-1])
        dg_T = np.load(dg_T_files[-1])
        dg_xc = dg_x[np.argmin(np.abs(dg_T - T_SHOCK_MID))]
        dg_data = {"x_rel": (dg_x - dg_xc) * 1e6, "T": dg_T}  # x_rel in µm, centred
        print(f"  DG data: {len(dg_x)} points loaded, shock@{dg_xc*1e6:.3f} µm")
    else:
        print(f"  [skip] DG data not found in {dg_dir}")

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

        # Overlay reference data on the T panel only
        if col_key == "T":
            # MD data — x already in µm, shock-centred
            if md_data is not None:
                mask = np.abs(md_data["x"]) <= half_um
                if mask.any():
                    ax.scatter(md_data["x"][mask], md_data["T"][mask],
                               s=18, color="tab:green", marker="o",
                               zorder=5, label="MD", alpha=0.85)
            # DG (p=2) data — x in µm, shock-centred via T_SHOCK_MID
            if dg_data is not None:
                mask = np.abs(dg_data["x_rel"]) <= half_um
                if mask.any():
                    ax.plot(dg_data["x_rel"][mask], dg_data["T"][mask],
                            color="tab:brown", ls="-", lw=1.5,
                            zorder=4, label="DG p=2", alpha=0.85)

        ax.set_xlabel("x − x_shock  [µm]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-half_um, half_um)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=9)
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.4)  # mark shock centre

    for ax in axes:
        ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()

    # ── Shock thickness summary ───────────────────────────────────────────────
    # L = |p_left - p_right| / |dp/dx|_max  (pressure-gradient definition)
    # dp/dx computed with second-order central differences (one-sided at boundaries)
    print(f"\nShock thickness  L = |Δp| / |dp/dx|_max  (within ±{half_um:.2f} µm window):")
    print(f"  {'Case':<50}  {'L [nm]':>8}  {'cells (dx=3.8nm)':>18}")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d    = datasets[case_name]
        xc   = shock_centre(d)
        x_r  = (d["x"] - xc) * 1e6
        mask = np.abs(x_r) <= half_um
        xi   = d["x"][mask]
        pi   = d["p"][mask]
        if len(xi) < 3:
            continue

        # Second-order accurate dp/dx
        dpi_dx = np.zeros_like(pi)
        dpi_dx[1:-1] = (pi[2:] - pi[:-2]) / (xi[2:] - xi[:-2])
        dpi_dx[0]    = (-3*pi[0] + 4*pi[1] - pi[2])     / (xi[2]  - xi[0])
        dpi_dx[-1]   = ( 3*pi[-1] - 4*pi[-2] + pi[-3])  / (xi[-1] - xi[-3])

        dpi_dx_max = np.max(np.abs(dpi_dx))
        if dpi_dx_max < 1e-10:
            continue
        L  = abs((pi[0] - pi[-1]) / dpi_dx_max)
        dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
        print(f"  {label:<50}  {L*1e9:>8.1f}  {L/dx:>18.1f}")


if __name__ == "__main__":
    main()
