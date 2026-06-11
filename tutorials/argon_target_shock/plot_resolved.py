#!/usr/bin/env python3
"""
SPLAY – Resolved shock structure.

Produces TWO figures from one run:
  Fig 1  (_with_dg):   DG p=2 n=800 by cell (underneath) + FVM + Ref DG + MD
  Fig 2  (_fvm_only):  FVM only + Ref DG + MD

DG data is plotted cell-by-cell (NaN breaks at element faces) using the
'cell' column written by write_csv_dg.  FVM data is plotted as a plain line.

Usage (from repo root):
    python tutorials/argon_target_shock/plot_resolved.py
    python tutorials/argon_target_shock/plot_resolved.py --save output/resolved.png
    python tutorials/argon_target_shock/plot_resolved.py --window 0.15
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

# ── FVM reference cases ────────────────────────────────────────────────────────
# (case_name, label, color, linestyle, linewidth, alpha)
FVM_CASES = [
    ("argon_shock_central_navier_stokes",  "Central NS",  "tab:blue",    "-",   2.0, 0.85),
    ("argon_shock_ppm_navier_stokes",      "PPM NS",      "tab:orange",  "--",  2.0, 0.85),
    ("argon_shock_muscl_navier_stokes",    "MUSCL NS",    "tab:red",     ":",   2.0, 0.85),
]

# ── SPLAY DG cases ─────────────────────────────────────────────────────────────
# (case_name, label, color, linewidth, alpha)
DG_CASES = [
    ("argon_shock_dg_p2_n800_quick",  "SPLAY DG p=2, n=800",  "tab:green",  2.0, 1.0),
]

VARS = [
    ("rho", r"Density  [kg/m³]"),
    ("p",   "Pressure  [Pa]"),
    ("T",   "Temperature  [K]"),
]

T_SHOCK_MID = 1500.0   # K — shock-centring threshold


# ── I/O helpers ────────────────────────────────────────────────────────────────
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
    """x [m] where T crosses T_SHOCK_MID by linear interpolation."""
    x = d["x"]
    T = d["T"]
    crossings = np.where((T[:-1] - T_SHOCK_MID) * (T[1:] - T_SHOCK_MID) < 0)[0]
    if len(crossings):
        i    = crossings[0]
        frac = (T_SHOCK_MID - T[i]) / (T[i + 1] - T[i])
        return x[i] + frac * (x[i + 1] - x[i])
    rho  = d["rho"]
    drho = np.zeros_like(rho)
    drho[1:-1] = (rho[2:] - rho[:-2]) / (x[2:] - x[:-2])
    drho[0]    = (rho[1]  - rho[0])   / (x[1]  - x[0])
    drho[-1]   = (rho[-1] - rho[-2])  / (x[-1] - x[-2])
    return x[np.argmax(np.abs(drho))]


def make_by_cell(d, xc_m, half_um, col):
    """Build (x_rel [µm], y) arrays with NaN between each DG cell.

    Groups rows by the 'cell' integer column so each element is a separate
    line segment, making inter-element discontinuities visible as gaps.
    """
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


def plot_figure(axes, half_um, fvm_datasets, dg_splay, ref_dg, md_data,
                include_dg_splay):
    """Populate a row of axes (one per variable).

    Layer order when include_dg_splay=True:
      1. DG by cell  (bottom)
      2. FVM         (above)
      3. Ref DG npy  (T panel only)
      4. MD scatter  (T panel only, top)
    """
    for ax, (col_key, ylabel) in zip(axes, VARS):

        # ── DG by cell (underneath FVM) ───────────────────────────────────────
        if include_dg_splay:
            for case_name, label, color, lw, alpha in DG_CASES:
                if case_name not in dg_splay:
                    continue
                d  = dg_splay[case_name]
                xc = shock_centre(d)
                xp, yp = make_by_cell(d, xc, half_um, col_key)
                ax.plot(xp, yp, color=color, lw=lw, alpha=alpha,
                        label=label, zorder=2)

        # ── FVM (on top of DG) ────────────────────────────────────────────────
        for case_name, label, color, ls, lw, alpha in FVM_CASES:
            if case_name not in fvm_datasets:
                continue
            d      = fvm_datasets[case_name]
            xc     = shock_centre(d)
            x_rel  = (d["x"] - xc) * 1e6
            mask   = np.abs(x_rel) <= half_um
            ax.plot(x_rel[mask], d[col_key][mask],
                    color=color, ls=ls, lw=lw, alpha=alpha,
                    label=label, zorder=3)

        # ── T panel: Ref DG npy and MD ────────────────────────────────────────
        if col_key == "T":
            if ref_dg is not None:
                ax.plot(ref_dg["x_rel"], ref_dg["T"],
                        color="tab:brown", ls="-", lw=2.0, alpha=0.9,
                        label="Ref. DG p=2", zorder=5)
            if md_data is not None:
                mask = np.abs(md_data["x"]) <= half_um
                ax.scatter(md_data["x"][mask], md_data["T"][mask],
                           s=20, color="tab:green", marker="o",
                           alpha=0.85, zorder=6, label="MD")

        ax.set_xlabel("x − x_shock  [µm]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-half_um, half_um)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.35)
        ax.grid(True, alpha=0.22, linestyle=":")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SPLAY resolved shock: DG+FVM figure and FVM-only figure")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save",   default="",
                        help="Base path for saved figures (e.g. output/resolved.png)."
                             " Two files are written: _with_dg and _fvm_only.")
    parser.add_argument("--window", type=float, default=0.1,
                        help="Half-width of x window [µm] (default 0.1)")
    args = parser.parse_args()

    half_um = args.window

    # ── Load data ─────────────────────────────────────────────────────────────
    fvm_datasets = {}
    for case_name, label, *_ in FVM_CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip-fvm] {label}")
            continue
        d = load_csv(csv)
        if d is not None:
            fvm_datasets[case_name] = d
            print(f"  FVM {label}: {os.path.basename(csv)}"
                  f"  shock@{shock_centre(d)*1e6:.3f} µm")

    dg_splay = {}
    for case_name, label, *_ in DG_CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip-dg] {label}")
            continue
        d = load_csv(csv)
        if d is not None:
            dg_splay[case_name] = d
            xc      = shock_centre(d)
            rho_max = np.nanmax(d["rho"])
            print(f"  {label}: {os.path.basename(csv)}"
                  f"  shock@{xc*1e6:.3f} µm  rho_max={rho_max:.1f}")

    md_data = None
    md_path = os.path.join(SCRIPT_DIR, "MD_data.txt")
    if os.path.exists(md_path):
        md_arr  = np.loadtxt(md_path, delimiter=",")
        md_data = {"x": md_arr[:, 0], "T": md_arr[:, 1]}
        print(f"  MD: {len(md_arr)} points")

    ref_dg = None
    dg_dir  = os.path.join(SCRIPT_DIR, "DG")
    x_files = sorted(glob.glob(os.path.join(dg_dir, "x_*.npy")))
    T_files = sorted(glob.glob(os.path.join(dg_dir, "Temperature_*.npy")))
    if x_files and T_files:
        rx   = np.load(x_files[-1])
        rT   = np.load(T_files[-1])
        crossings = np.where((rT[:-1] - T_SHOCK_MID) * (rT[1:] - T_SHOCK_MID) < 0)[0]
        if len(crossings):
            ci   = crossings[0]
            frac = (T_SHOCK_MID - rT[ci]) / (rT[ci + 1] - rT[ci])
            rxc  = rx[ci] + frac * (rx[ci + 1] - rx[ci])
        else:
            rxc  = rx[np.argmax(np.abs(np.gradient(rT, rx)))]
        xrel = (rx - rxc) * 1e6
        mask = np.abs(xrel) <= half_um
        ref_dg = {"x_rel": xrel[mask], "T": rT[mask]}
        print(f"  Reference DG (.npy): {len(rx)} pts  shock@{rxc*1e6:.3f} µm")

    if not fvm_datasets and not dg_splay:
        print("\nNo data found.")
        sys.exit(1)

    # ── Figure 1: DG underneath FVM ───────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, len(VARS), figsize=(5.2 * len(VARS), 5.2),
                               constrained_layout=True)
    fig1.suptitle(
        "Argon shock — SPLAY DG p=2 (by cell) + FVM + Ref DG + MD  (M ≈ 5.03)\n"
        f"DG underneath FVM  |  centred at T = {T_SHOCK_MID:.0f} K"
        f"  |  ±{half_um:.2f} µm",
        fontsize=11, fontweight="bold"
    )
    plot_figure(axes1, half_um, fvm_datasets, dg_splay, ref_dg, md_data,
                include_dg_splay=True)

    # ── Figure 2: FVM only (+ Ref DG + MD) ────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(VARS), figsize=(5.2 * len(VARS), 5.2),
                               constrained_layout=True)
    fig2.suptitle(
        "Argon shock — FVM schemes + Ref DG + MD  (M ≈ 5.03)\n"
        f"Centred at T = {T_SHOCK_MID:.0f} K  |  ±{half_um:.2f} µm",
        fontsize=11, fontweight="bold"
    )
    plot_figure(axes2, half_um, fvm_datasets, dg_splay, ref_dg, md_data,
                include_dg_splay=False)

    # ── Save or show ──────────────────────────────────────────────────────────
    if args.save:
        base, ext = os.path.splitext(args.save)
        if not ext:
            ext = ".png"
        p1 = base + "_with_dg" + ext
        p2 = base + "_fvm_only" + ext
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {p1}")
        print(f"Saved: {p2}")
    else:
        plt.show()

    # ── Shock thickness summary ────────────────────────────────────────────────
    all_cases = [(n, l) for n, l, *_ in DG_CASES] + [(n, l) for n, l, *_ in FVM_CASES]
    all_data  = {**dg_splay, **fvm_datasets}
    print(f"\nShock thickness  L = |Δp| / |dp/dx|_max  (±{half_um:.2f} µm window):")
    print(f"  {'Case':<35}  {'L [nm]':>8}  {'N pts':>7}  {'dx [nm]':>8}")
    for case_name, label in all_cases:
        if case_name not in all_data:
            continue
        d  = all_data[case_name]
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
