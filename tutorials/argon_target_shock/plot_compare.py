#!/usr/bin/env python3
"""
SPLAY – Tutorial comparison plot.

Reads the final combined CSV snapshot from each of the tutorial cases
and overlays them, centred at T = 1500 K with a ±window µm x-axis.

Usage
-----
    python tutorials/argon_target_shock/plot_compare.py
    python tutorials/argon_target_shock/plot_compare.py --output_dir output --window 0.15
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

CASES = [
    ("argon_shock_central_navier_stokes",  "Central NS",          "tab:blue",     "-",   2.0),
    ("argon_shock_muscl_euler",            "MUSCL Euler",         "tab:cyan",     "--",  2.0),
    ("argon_shock_ppm_euler",              "PPM Euler",           "tab:olive",    "--",  2.0),
    ("argon_shock_muscl_navier_stokes",    "MUSCL NS",            "tab:red",      ":",   2.0),
    ("argon_shock_ppm_navier_stokes",      "PPM NS",              "tab:orange",   ":",   2.0),
    ("argon_shock_dg_p2_n800_quick",        "SPLAY DG p=2, n=800", "tab:green",    "-",   2.0),
]

VARS = ["rho", "u", "p", "T", "Mach", "mu"]
LABELS = {
    "rho":  r"Density [kg/m³]",
    "u":    "Velocity [m/s]",
    "p":    "Pressure [Pa]",
    "T":    "Temperature [K]",
    "Mach": "Mach",
    "mu":   r"$\mu$ [Pa·s]",
}

T_SHOCK_MID = 1500.0   # K — shock-centring threshold


def find_latest_csv(case_dir):
    files = sorted(glob.glob(os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


def load_csv(path):
    """Load snapshot CSV.  DG files (have 'cell' column) keep original row order;
    FVM files are sorted by x."""
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
    if "cell" not in d:
        idx = np.argsort(d["x"])
        d   = {k: v[idx] for k, v in d.items()}
    return d


def shock_centre(d):
    """x [m] where T crosses T_SHOCK_MID, searching from the right."""
    x = d["x"]
    T = d["T"]
    crossings = np.where((T[:-1] - T_SHOCK_MID) * (T[1:] - T_SHOCK_MID) < 0)[0]
    if len(crossings):
        i    = crossings[-1]   # rightmost crossing — avoids spurious left-side hits
        frac = (T_SHOCK_MID - T[i]) / (T[i + 1] - T[i])
        return x[i] + frac * (x[i + 1] - x[i])
    rho  = d["rho"]
    drho = np.zeros_like(rho)
    drho[1:-1] = (rho[2:] - rho[:-2]) / (x[2:] - x[:-2])
    drho[0]    = (rho[1]  - rho[0])   / (x[1]  - x[0])
    drho[-1]   = (rho[-1] - rho[-2])  / (x[-1] - x[-2])
    return x[np.argmax(np.abs(drho))]


def make_by_cell(d, xc_m, half_um, col):
    """Build (x_rel [µm], y) with NaN breaks between DG cells."""
    x_rel = (d["x"] - xc_m) * 1e6
    cells = d["cell"].astype(int)
    x_out, y_out = [], []
    for c in np.unique(cells):
        mask   = cells == c
        xc_pts = x_rel[mask]
        y_pts  = d[col][mask]
        order  = np.argsort(xc_pts)
        xc_pts = xc_pts[order]
        y_pts  = y_pts[order]
        if np.any(np.abs(xc_pts) <= half_um):
            x_out.extend(xc_pts.tolist())
            y_out.extend(y_pts.tolist())
        x_out.append(np.nan)
        y_out.append(np.nan)
    return np.array(x_out), np.array(y_out)


def main():
    parser = argparse.ArgumentParser(description="SPLAY tutorial comparison plot")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=float, default=0.1,
                        help="Half-width of x window [µm] (default 0.1)")
    parser.add_argument("--save", default="",
                        help="Save path (default: output_dir/argon_shock_comparison.png)")
    args = parser.parse_args()

    half_um = args.window

    print(f"Reading output from: {args.output_dir}")

    datasets = {}
    for case_name, label, *_ in CASES:
        case_dir = os.path.join(args.output_dir, case_name)
        csv_path = find_latest_csv(case_dir)
        if csv_path is None:
            print(f"  [skip] {label}")
            continue
        d = load_csv(csv_path)
        if d is not None:
            datasets[case_name] = d
            xc = shock_centre(d)
            print(f"  {label}: {os.path.basename(csv_path)}"
                  f"  shock@{xc*1e6:.3f} µm")

    if not datasets:
        print("No data found. Run the tutorial cases first.")
        sys.exit(1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, var in zip(axes, VARS):
        for case_name, label, color, ls, lw in CASES:
            if case_name not in datasets:
                continue
            d  = datasets[case_name]
            xc = shock_centre(d)

            if "cell" in d:
                xp, yp = make_by_cell(d, xc, half_um, var)
                ax.plot(xp, yp, color=color, ls=ls, lw=lw, label=label, zorder=3)
            else:
                x_rel = (d["x"] - xc) * 1e6
                mask  = np.abs(x_rel) <= half_um
                ax.plot(x_rel[mask], d[var][mask],
                        color=color, ls=ls, lw=lw, label=label, zorder=3)

        ax.set_xlabel("x − x_shock  [µm]", fontsize=10)
        ax.set_ylabel(LABELS.get(var, var), fontsize=10)
        ax.set_xlim(-half_um, half_um)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.35)
        ax.grid(True, alpha=0.22, linestyle=":")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=7, loc="best", framealpha=0.85)

    fig.suptitle(
        f"Argon shock comparison — M ≈ 5.03\n"
        f"Centred at T = {T_SHOCK_MID:.0f} K  |  ±{half_um:.2f} µm",
        fontsize=12, fontweight="bold"
    )

    save_path = args.save or os.path.join(args.output_dir, "argon_shock_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.show()

    # ── Shock thickness summary ───────────────────────────────────────────────
    print(f"\nShock thickness  L = |Δp| / |dp/dx|_max  (±{half_um:.2f} µm window):")
    print(f"  {'Case':<35}  {'L [nm]':>8}  {'N pts':>7}  {'dx [nm]':>8}")
    for case_name, label, *_ in CASES:
        if case_name not in datasets:
            continue
        d  = datasets[case_name]
        xc = shock_centre(d)
        x_rel = (d["x"] - xc) * 1e6
        mask  = np.abs(x_rel) <= half_um
        xi = d["x"][mask]
        pi = d["p"][mask]
        if len(xi) < 3:
            continue
        dpi = np.zeros_like(pi)
        dpi[1:-1] = (pi[2:] - pi[:-2]) / (xi[2:] - xi[:-2])
        dpi[0]    = (-3*pi[0] + 4*pi[1] - pi[2])    / (xi[2]  - xi[0])
        dpi[-1]   = ( 3*pi[-1] - 4*pi[-2] + pi[-3]) / (xi[-1] - xi[-3])
        mx = np.max(np.abs(dpi))
        if mx < 1e-10:
            continue
        L  = abs((pi[0] - pi[-1]) / mx)
        dx = (xi[-1] - xi[0]) / max(len(xi) - 1, 1)
        print(f"  {label:<35}  {L*1e9:>8.1f}  {len(xi):>7d}  {dx*1e9:>8.2f}")


if __name__ == "__main__":
    main()
