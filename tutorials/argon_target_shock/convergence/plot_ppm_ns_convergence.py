#!/usr/bin/env python3
"""
SPLAY – PPM + NS convergence study: shock profiles and L1 error vs dx.

Figure 1: Temperature profiles for n=100..3200, centred at T=1500 K, ±window µm.
          Reference DG p=2 (n=800) npy overlaid.

Figure 2: L1 error of T (and rho) vs dx [nm] on a log-log plot.
          Reference solution is the finest available PPM NS case (n=3200).
          Order-1 and order-2 reference lines are drawn.

Usage (from repo root):
    python tutorials/argon_target_shock/convergence/plot_ppm_ns_convergence.py
    python tutorials/argon_target_shock/convergence/plot_ppm_ns_convergence.py --window 0.15
    python tutorials/argon_target_shock/convergence/plot_ppm_ns_convergence.py --output_dir /path/to/output
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TUTORIAL_DIR   = os.path.join(REPO_ROOT, "tutorials", "argon_target_shock")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

DOMAIN_L_M = 7.600403652e-4 * 1e-2   # cm → m = 7.6 µm

# n values in order coarse → fine
N_VALUES = [100, 200, 400, 800, 1600, 3200]

# Sequential colors coarse (light) → fine (dark)
_cmap  = cm.plasma
COLORS = {n: _cmap(0.15 + 0.7 * i / (len(N_VALUES) - 1))
          for i, n in enumerate(N_VALUES)}

T_SHOCK_MID = 1500.0   # K


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_latest_csv(case_dir):
    files = sorted(glob.glob(
        os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


def load_csv(path):
    """Load FVM CSV; sort by x (no cell column for FVM)."""
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


def shock_thickness_nm(d, xc, half_m, lo=0.1, hi=0.9):
    """10%–90% temperature-rise width [nm] within ±half_m of shock centre."""
    x_rel = d["x"] - xc
    mask  = np.abs(x_rel) <= half_m
    xi, Ti = x_rel[mask], d["T"][mask]
    if len(xi) < 4:
        return np.nan
    T_lo = Ti.min() + lo * (Ti.max() - Ti.min())
    T_hi = Ti.min() + hi * (Ti.max() - Ti.min())
    # interpolate x positions of lo and hi crossings (search from right → left
    # so we catch the shock front, not upstream oscillations)
    crossings_lo = np.where((Ti[:-1] - T_lo) * (Ti[1:] - T_lo) < 0)[0]
    crossings_hi = np.where((Ti[:-1] - T_hi) * (Ti[1:] - T_hi) < 0)[0]
    if not len(crossings_lo) or not len(crossings_hi):
        return np.nan
    def interp_cross(crossings, level):
        i    = crossings[-1]
        frac = (level - Ti[i]) / (Ti[i + 1] - Ti[i])
        return xi[i] + frac * (xi[i + 1] - xi[i])
    x_lo = interp_cross(crossings_lo, T_lo)
    x_hi = interp_cross(crossings_hi, T_hi)
    return abs(x_hi - x_lo) * 1e9   # → nm


def shock_centre(d):
    """x [m] where T crosses T_SHOCK_MID, searching from the right."""
    x = d["x"]
    T = d["T"]
    crossings = np.where((T[:-1] - T_SHOCK_MID) * (T[1:] - T_SHOCK_MID) < 0)[0]
    if len(crossings):
        i    = crossings[-1]   # rightmost crossing
        frac = (T_SHOCK_MID - T[i]) / (T[i + 1] - T[i])
        return x[i] + frac * (x[i + 1] - x[i])
    # fallback: max |dρ/dx|
    rho  = d["rho"]
    drho = np.zeros_like(rho)
    drho[1:-1] = (rho[2:] - rho[:-2]) / (x[2:] - x[:-2])
    drho[0]    = (rho[1]  - rho[0])   / (x[1]  - x[0])
    drho[-1]   = (rho[-1] - rho[-2])  / (x[-1] - x[-2])
    return x[np.argmax(np.abs(drho))]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPM+NS convergence: profiles + L1 error vs dx")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=float, default=0.2,
                        help="Half-width of x window [µm] (default 0.2)")
    parser.add_argument("--save_profiles", default="",
                        help="Save profiles figure to file")
    parser.add_argument("--save_error", default="",
                        help="Save error figure to file")
    args = parser.parse_args()

    half_um = args.window

    # ── Load PPM NS convergence data ─────────────────────────────────────────
    datasets = {}   # n → dict of arrays
    for n in N_VALUES:
        case_name = f"argon_shock_ppm_ns_n{n}_conv"
        case_dir  = os.path.join(args.output_dir, case_name)
        csv       = find_latest_csv(case_dir)
        if csv is None:
            print(f"  [skip] n={n:4d}  (no CSV in {case_dir})")
            continue
        d = load_csv(csv)
        if d is None:
            continue
        if np.any(~np.isfinite(d["T"])):
            print(f"  [NaN]  n={n:4d}  {os.path.basename(csv)}")
            continue
        xc = shock_centre(d)
        dx_nm = DOMAIN_L_M / n * 1e9
        print(f"  n={n:4d}  dx={dx_nm:6.1f} nm  shock@{xc*1e6:.3f} µm"
              f"  rho_max={d['rho'].max():.1f}  {os.path.basename(csv)}")
        datasets[n] = d

    if not datasets:
        print("\nNo data found. Run the convergence cases first.")
        sys.exit(1)

    # ── Load reference DG T profile (.npy) ───────────────────────────────────
    dg_dir  = os.path.join(TUTORIAL_DIR, "DG")
    x_files = sorted(glob.glob(os.path.join(dg_dir, "x_*.npy")))
    T_files = sorted(glob.glob(os.path.join(dg_dir, "Temperature_*.npy")))
    ref_dg  = None
    if x_files and T_files:
        rx = np.load(x_files[-1])
        rT = np.load(T_files[-1])
        crossings = np.where((rT[:-1] - T_SHOCK_MID) * (rT[1:] - T_SHOCK_MID) < 0)[0]
        if len(crossings):
            ci   = crossings[-1]
            frac = (T_SHOCK_MID - rT[ci]) / (rT[ci + 1] - rT[ci])
            rxc  = rx[ci] + frac * (rx[ci + 1] - rx[ci])
        else:
            rxc = rx[np.argmax(np.abs(np.gradient(rT, rx)))]
        ref_xrel = (rx - rxc) * 1e6
        mask     = np.abs(ref_xrel) <= half_um
        ref_dg   = (ref_xrel[mask], rT[mask])
        print(f"  Ref DG (.npy): {len(rx)} pts  shock@{rxc*1e6:.3f} µm")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1 – Temperature profiles
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for n in N_VALUES:
        if n not in datasets:
            continue
        d     = datasets[n]
        xc    = shock_centre(d)
        x_rel = (d["x"] - xc) * 1e6
        mask  = np.abs(x_rel) <= half_um
        dx_nm = DOMAIN_L_M / n * 1e9
        ax.plot(x_rel[mask], d["T"][mask],
                color=COLORS[n], lw=2.0,
                label=f"PPM NS  n={n}  (Δx={dx_nm:.0f} nm)")

    # Reference DG
    if ref_dg is not None:
        ax.plot(ref_dg[0], ref_dg[1],
                color="tab:brown", lw=2.0, ls="--",
                label="Ref. DG p=2 (n=800)", zorder=5)

    ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("x − x_shock  [µm]", fontsize=11)
    ax.set_ylabel("Temperature  [K]", fontsize=11)
    ax.set_xlim(-half_um, half_um)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    ax.set_title(
        f"Argon shock — PPM + NS convergence  (M ≈ 5.03)\n"
        f"Centred at T = {T_SHOCK_MID:.0f} K  |  ±{half_um:.2f} µm",
        fontsize=11, fontweight="bold"
    )

    if args.save_profiles:
        fig1.savefig(args.save_profiles, dpi=300, bbox_inches="tight")
        print(f"\nSaved profiles: {args.save_profiles}")
    else:
        plt.show()

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2 – L1 error vs dx
    # ═══════════════════════════════════════════════════════════════════════════

    # Use finest available as reference
    n_ref = max(datasets.keys())
    d_ref = datasets[n_ref]
    xc_ref   = shock_centre(d_ref)
    x_ref_m  = d_ref["x"] - xc_ref
    T_ref    = d_ref["T"]
    rho_ref  = d_ref["rho"]

    dx_list, errT_list, errRho_list, thick_list = [], [], [], []
    half_m = half_um * 1e-6

    # Thickness for reference case too (for horizontal reference line)
    thick_ref_nm = shock_thickness_nm(d_ref, shock_centre(d_ref), half_m)

    for n in N_VALUES:
        if n not in datasets or n == n_ref:
            continue
        d  = datasets[n]
        xc = shock_centre(d)
        x_rel_m = d["x"] - xc
        dx_nm   = DOMAIN_L_M / n * 1e9

        mask = np.abs(x_rel_m) <= half_m
        if mask.sum() < 3:
            continue

        xi = x_rel_m[mask]
        Ti = d["T"][mask]
        Ri = d["rho"][mask]

        T_ref_interp   = np.interp(xi, x_ref_m, T_ref)
        rho_ref_interp = np.interp(xi, x_ref_m, rho_ref)

        errT   = np.mean(np.abs(Ti - T_ref_interp))
        errRho = np.mean(np.abs(Ri - rho_ref_interp))
        thick  = shock_thickness_nm(d, xc, half_m)

        dx_list.append(dx_nm)
        errT_list.append(errT)
        errRho_list.append(errRho)
        thick_list.append(thick)
        print(f"  n={n:4d}  dx={dx_nm:6.1f} nm"
              f"  L1(T)={errT:8.2f} K  L1(rho)={errRho:.4f} kg/m³"
              f"  δ={thick:.1f} nm")

    print(f"  n={n_ref:4d}  dx={DOMAIN_L_M/n_ref*1e9:6.1f} nm"
          f"  (reference)  δ={thick_ref_nm:.1f} nm")

    if len(dx_list) >= 2:
        fig2, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

        dx_arr   = np.array(dx_list)
        errT_arr = np.array(errT_list)
        errR_arr = np.array(errRho_list)
        thick_arr = np.array(thick_list)

        n_labels = [n for n in N_VALUES if n in datasets and n != n_ref]

        for ax, err, var_label in [
            (axes[0], errT_arr, "L1 error — Temperature [K]"),
            (axes[1], errR_arr, r"L1 error — Density [kg/m³]"),
        ]:
            ax.loglog(dx_arr, err, "o-", color="tab:blue", lw=2.0,
                      ms=7, label="PPM NS")

            x0, e0 = dx_arr[0], err[0]
            x_ref_line = np.array([dx_arr[0], dx_arr[-1]])
            ax.loglog(x_ref_line, e0 * (x_ref_line / x0) ** 1,
                      "k--", lw=1.2, alpha=0.6, label="Order 1")
            ax.loglog(x_ref_line, e0 * (x_ref_line / x0) ** 2,
                      "k:",  lw=1.2, alpha=0.6, label="Order 2")

            ax.set_xlabel("Δx  [nm]", fontsize=11)
            ax.set_ylabel(var_label, fontsize=10)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

            for dx_i, e_i, n_i in zip(dx_arr, err, n_labels):
                ax.annotate(f"n={n_i}", (dx_i, e_i),
                            textcoords="offset points", xytext=(5, 3), fontsize=7)

        # ── Third panel: shock thickness vs dx ───────────────────────────────
        ax3 = axes[2]
        valid = np.isfinite(thick_arr)
        ax3.semilogx(dx_arr[valid], thick_arr[valid], "s-",
                     color="tab:red", lw=2.0, ms=7, label="PPM NS (10–90% T rise)")
        if np.isfinite(thick_ref_nm):
            ax3.axhline(thick_ref_nm, color="tab:red", lw=1.2, ls="--", alpha=0.6,
                        label=f"n={n_ref} ref  ({thick_ref_nm:.1f} nm)")
        ax3.set_xlabel("Δx  [nm]", fontsize=11)
        ax3.set_ylabel("Shock thickness  [nm]", fontsize=10)
        ax3.legend(fontsize=9, framealpha=0.9)
        ax3.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        for dx_i, th_i, n_i in zip(dx_arr[valid], thick_arr[valid], np.array(n_labels)[valid]):
            ax3.annotate(f"n={n_i}", (dx_i, th_i),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

        fig2.suptitle(
            f"PPM + NS convergence — Argon shock  (M ≈ 5.03)\n"
            f"Reference: n={n_ref}  |  shock window ±{half_um:.2f} µm  |  thickness = 10–90% T rise",
            fontsize=11, fontweight="bold"
        )

        if args.save_error:
            fig2.savefig(args.save_error, dpi=300, bbox_inches="tight")
            print(f"Saved error plot: {args.save_error}")
        else:
            plt.show()
    else:
        print("\nNeed at least 2 coarser cases to compute L1 errors.")


if __name__ == "__main__":
    main()
