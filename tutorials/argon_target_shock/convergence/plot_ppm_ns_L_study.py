#!/usr/bin/env python3
"""
SPLAY – PPM + NS domain-size study: shock structure vs Δx at fixed n=100.

Domain L is increased by factors of 10 (7.6 µm → 76 mm), keeping n=100.
As Δx grows relative to the physical shock thickness (~40 nm), the shock
transitions from viscous-smooth to Euler-like (2–3 cell jump).

Figure 1: T profiles in NORMALISED coordinates (x−x_shock)/Δx.
          If the shock spans 2–3 x/Δx units, it is Euler-like.

Figure 2: Shock thickness [nm] vs Δx [nm] on a log-log plot.
          References: physical NS thickness (horizontal) and 2·Δx Euler limit.

Usage (from repo root):
    python tutorials/argon_target_shock/convergence/plot_ppm_ns_L_study.py
    python tutorials/argon_target_shock/convergence/plot_ppm_ns_L_study.py \\
        --save_profiles output/L_study_profiles.png \\
        --save_thickness output/L_study_thickness.png
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")
TUTORIAL_DIR   = os.path.join(REPO_ROOT, "tutorials", "argon_target_shock")

# (case_name, L [m], label, color)
# Lx1 re-uses the existing n100_conv run
CASES = [
    ("argon_shock_ppm_ns_n100_conv",      7.600403652e-6,  "Δx = 76 nm  (L = 7.6 µm)",   "tab:purple"),
    ("argon_shock_ppm_ns_n100_Lx10",      7.600403652e-5,  "Δx = 760 nm  (L = 76 µm)",   "tab:blue"),
    ("argon_shock_ppm_ns_n100_Lx100",     7.600403652e-4,  "Δx = 7.6 µm  (L = 760 µm)",  "tab:green"),
    ("argon_shock_ppm_ns_n100_Lx1000",    7.600403652e-3,  "Δx = 76 µm  (L = 7.6 mm)",   "tab:orange"),
    ("argon_shock_ppm_ns_n100_Lx10000",   7.600403652e-2,  "Δx = 760 µm  (L = 76 mm)",   "tab:red"),
]
N_CELLS = 100
T_SHOCK_MID = 1500.0   # K


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_latest_csv(case_dir):
    files = sorted(glob.glob(
        os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
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


def shock_centre(d):
    x, T = d["x"], d["T"]
    crossings = np.where((T[:-1] - T_SHOCK_MID) * (T[1:] - T_SHOCK_MID) < 0)[0]
    if len(crossings):
        i    = crossings[-1]
        frac = (T_SHOCK_MID - T[i]) / (T[i + 1] - T[i])
        return x[i] + frac * (x[i + 1] - x[i])
    rho  = d["rho"]
    drho = np.zeros_like(rho)
    drho[1:-1] = (rho[2:] - rho[:-2]) / (x[2:] - x[:-2])
    drho[0]    = (rho[1]  - rho[0])   / (x[1]  - x[0])
    drho[-1]   = (rho[-1] - rho[-2])  / (x[-1] - x[-2])
    return x[np.argmax(np.abs(drho))]


def shock_thickness_nm(d, xc, dx_m, n_win=15, lo=0.1, hi=0.9):
    """10–90% T-rise width [nm] within ±n_win cell widths of shock centre."""
    half_m = n_win * dx_m
    x_rel  = d["x"] - xc
    mask   = np.abs(x_rel) <= half_m
    xi, Ti = x_rel[mask], d["T"][mask]
    if len(xi) < 4:
        return np.nan
    T_lo = Ti.min() + lo * (Ti.max() - Ti.min())
    T_hi = Ti.min() + hi * (Ti.max() - Ti.min())
    cross_lo = np.where((Ti[:-1] - T_lo) * (Ti[1:] - T_lo) < 0)[0]
    cross_hi = np.where((Ti[:-1] - T_hi) * (Ti[1:] - T_hi) < 0)[0]
    if not len(cross_lo) or not len(cross_hi):
        return np.nan
    def interp(crossings, level):
        i    = crossings[-1]
        frac = (level - Ti[i]) / (Ti[i + 1] - Ti[i])
        return xi[i] + frac * (xi[i + 1] - xi[i])
    return abs(interp(cross_hi, T_hi) - interp(cross_lo, T_lo)) * 1e9


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPM+NS domain-size study: shock structure vs Δx")
    parser.add_argument("--output_dir",      default=DEFAULT_OUTPUT)
    parser.add_argument("--norm_half",       type=float, default=8.0,
                        help="Half-window in cell widths for normalised profile plot (default 8)")
    parser.add_argument("--save_profiles",   default="")
    parser.add_argument("--save_thickness",  default="")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    datasets = {}   # case_name → (d, L_m, dx_m, label, color)
    for case_name, L_m, label, color in CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip] {label}")
            continue
        d = load_csv(csv)
        if d is None or not np.all(np.isfinite(d["T"])):
            print(f"  [bad]  {label}")
            continue
        dx_m = L_m / N_CELLS
        xc   = shock_centre(d)
        th   = shock_thickness_nm(d, xc, dx_m)
        print(f"  {label:<35}  dx={dx_m*1e9:8.1f} nm  xc={xc*1e6:.2f} µm  δ={th:.1f} nm")
        datasets[case_name] = (d, L_m, dx_m, label, color)

    if not datasets:
        print("No data found.")
        sys.exit(1)

    # Physical NS shock thickness from the fine convergence reference (n=3200)
    # Use the finest available PPM NS conv case as an estimate
    phys_thick_nm = None
    for n_ref in [3200, 1600, 800]:
        ref_dir = os.path.join(args.output_dir, f"argon_shock_ppm_ns_n{n_ref}_conv")
        ref_csv = find_latest_csv(ref_dir)
        if ref_csv is None:
            continue
        dr  = load_csv(ref_csv)
        if dr is None:
            continue
        dx_r = 7.600403652e-6 / n_ref
        xc_r = shock_centre(dr)
        phys_thick_nm = shock_thickness_nm(dr, xc_r, dx_r)
        print(f"  Physical NS thickness (n={n_ref}): {phys_thick_nm:.1f} nm")
        break

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1 – T profiles in normalised (x − x_shock) / Δx coordinates
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, len(datasets), figsize=(3.5 * len(datasets), 4.5),
                                constrained_layout=True, sharey=True)
    if len(datasets) == 1:
        axes1 = [axes1]

    for ax, (case_name, (d, L_m, dx_m, label, color)) in zip(axes1, datasets.items()):
        xc    = shock_centre(d)
        x_ndx = (d["x"] - xc) / dx_m      # normalised to cell widths
        mask  = np.abs(x_ndx) <= args.norm_half

        ax.plot(x_ndx[mask], d["T"][mask], color=color, lw=2.0)
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.4)
        ax.set_xlim(-args.norm_half, args.norm_half)
        ax.set_xlabel("(x − x_shock) / Δx", fontsize=9)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(True, ls=":", lw=0.5, alpha=0.4)
        # mark cell boundaries
        for k in range(-int(args.norm_half), int(args.norm_half) + 1):
            ax.axvline(k, color="gray", lw=0.4, ls="--", alpha=0.3)

    axes1[0].set_ylabel("Temperature  [K]", fontsize=10)
    fig1.suptitle(
        "Argon shock — PPM + NS  (M ≈ 5.03)  |  n = 100 fixed, L increasing\n"
        "x-axis normalised to cell width Δx  |  vertical dashes = cell boundaries",
        fontsize=10, fontweight="bold"
    )

    if args.save_profiles:
        fig1.savefig(args.save_profiles, dpi=300, bbox_inches="tight")
        print(f"\nSaved profiles: {args.save_profiles}")
    else:
        plt.show()

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2 – Shock thickness vs Δx
    # ═══════════════════════════════════════════════════════════════════════════
    dx_nm_list, thick_nm_list, colors_list = [], [], []
    for case_name, (d, L_m, dx_m, label, color) in datasets.items():
        xc = shock_centre(d)
        th = shock_thickness_nm(d, xc, dx_m)
        if np.isfinite(th):
            dx_nm_list.append(dx_m * 1e9)
            thick_nm_list.append(th)
            colors_list.append(color)

    if len(dx_nm_list) >= 2:
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        dx_arr  = np.array(dx_nm_list)
        th_arr  = np.array(thick_nm_list)
        L_arr   = dx_arr * N_CELLS          # L [nm]
        ncells_arr = th_arr / dx_arr        # cells across shock

        # ── Left panel: thickness vs Δx ───────────────────────────────────────
        for dx_i, th_i, c in zip(dx_arr, th_arr, colors_list):
            ax2.scatter(dx_i, th_i, color=c, s=60, zorder=4)
        ax2.loglog(dx_arr, th_arr, "k-", lw=1.5, alpha=0.5, zorder=3)

        for case_name, (d, L_m, dx_m, label, color) in datasets.items():
            dx_i = dx_m * 1e9
            th_i = shock_thickness_nm(d, shock_centre(d), dx_m)
            if np.isfinite(th_i):
                ax2.annotate(label.split("(")[1].rstrip(")"),
                             (dx_i, th_i), textcoords="offset points",
                             xytext=(6, 4), fontsize=7)

        x_line = np.array([dx_arr.min() * 0.5, dx_arr.max() * 2])
        ax2.loglog(x_line, 2.0 * x_line, "k--", lw=1.5, alpha=0.7,
                   label="Euler limit  (δ = 2·Δx)")
        if phys_thick_nm is not None and np.isfinite(phys_thick_nm):
            ax2.axhline(phys_thick_nm, color="tab:brown", lw=1.5, ls="--",
                        alpha=0.8, label=f"Physical NS  (δ ≈ {phys_thick_nm:.0f} nm)")

        ax2.set_xlabel("Δx  [nm]", fontsize=12)
        ax2.set_ylabel("Shock thickness  (10–90% T rise)  [nm]", fontsize=11)
        ax2.legend(fontsize=9, framealpha=0.9)
        ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        ax2.set_title("Shock thickness vs Δx", fontsize=11, fontweight="bold")

        # ── Right panel: cells across shock vs Δx  (L-study + n-convergence) ──
        L_BASE_M = 7.600403652e-6   # fixed domain for n-convergence cases
        N_CONV   = [200, 400, 800, 1600, 3200]

        # L-study points  (circles, n=100 fixed)
        Lstudy_dx, Lstudy_nc, Lstudy_col = [], [], []
        for case_name, (d, L_m, dx_m, label, color) in datasets.items():
            th_i = shock_thickness_nm(d, shock_centre(d), dx_m)
            if np.isfinite(th_i):
                Lstudy_dx.append(dx_m * 1e9)
                Lstudy_nc.append(th_i / (dx_m * 1e9))
                Lstudy_col.append(color)

        Lstudy_dx = np.array(Lstudy_dx)
        Lstudy_nc = np.array(Lstudy_nc)
        order = np.argsort(Lstudy_dx)
        ax3.plot(Lstudy_dx[order], Lstudy_nc[order], "-", color="k",
                 lw=1.2, alpha=0.4, zorder=2)
        for dx_i, nc_i, c in zip(Lstudy_dx, Lstudy_nc, Lstudy_col):
            ax3.scatter(dx_i, nc_i, color=c, s=70, marker="o", zorder=4,
                        label=f"L-study  Δx={dx_i:.0f} nm")

        # n-convergence points  (squares, L=7.6 µm fixed)
        conv_dx, conv_nc = [], []
        for n_i in N_CONV:
            csv = find_latest_csv(
                os.path.join(args.output_dir, f"argon_shock_ppm_ns_n{n_i}_conv"))
            if csv is None:
                continue
            dc = load_csv(csv)
            if dc is None or not np.all(np.isfinite(dc["T"])):
                continue
            dx_m_c = L_BASE_M / n_i
            xc_c   = shock_centre(dc)
            th_c   = shock_thickness_nm(dc, xc_c, dx_m_c)
            if np.isfinite(th_c):
                conv_dx.append(dx_m_c * 1e9)
                conv_nc.append(th_c / (dx_m_c * 1e9))

        if conv_dx:
            conv_dx = np.array(conv_dx)
            conv_nc = np.array(conv_nc)
            order_c = np.argsort(conv_dx)
            ax3.plot(conv_dx[order_c], conv_nc[order_c], "-", color="k",
                     lw=1.2, alpha=0.4, zorder=2)
            ax3.scatter(conv_dx, conv_nc, color="tab:gray", s=70, marker="s",
                        zorder=4, label="n-convergence  (L = 7.6 µm fixed)")

        # Reference lines
        all_dx = np.concatenate([Lstudy_dx, conv_dx if len(conv_dx) > 0 else np.array([])])
        x_ref  = np.array([all_dx.min() * 0.5, all_dx.max() * 2])
        ax3.axhline(2.0, color="k", lw=1.5, ls="--", alpha=0.7,
                    label="Euler limit  (2 cells)")
        if phys_thick_nm is not None and np.isfinite(phys_thick_nm):
            ax3.loglog(x_ref, phys_thick_nm / x_ref, color="tab:brown",
                       lw=1.5, ls="--", alpha=0.8,
                       label=f"Physical NS  (δ ≈ {phys_thick_nm:.0f} nm / Δx)")

        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Δx  [nm]", fontsize=12)
        ax3.set_ylabel("Cells across shock  (δ / Δx)", fontsize=11)
        ax3.legend(fontsize=7.5, framealpha=0.9, loc="lower left")
        ax3.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        ax3.set_title("Cells across shock vs Δx\n"
                      "(circles = L-study n=100;  squares = n-convergence L=7.6 µm)",
                      fontsize=10, fontweight="bold")

        fig2.suptitle(
            "PPM + NS — n=100 fixed, L varies  (Δx = L/100)\n"
            "Transition from viscous-smooth (many cells) to Euler-like (2 cells) as L grows",
            fontsize=11, fontweight="bold"
        )

        if args.save_thickness:
            fig2.savefig(args.save_thickness, dpi=300, bbox_inches="tight")
            print(f"Saved thickness plot: {args.save_thickness}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
