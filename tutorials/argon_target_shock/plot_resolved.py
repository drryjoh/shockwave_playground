#!/usr/bin/env python3
"""
SPLAY – Resolved shock structure: SPLAY DG vs MD vs reference DG.

Panels: density, pressure, temperature — all shock-centred at T = T_SHOCK_MID.

Layers (back to front):
  1. FVM reference cases  (thin, muted)  — for context only
  2. SPLAY DG p=2 n=500 and n=1000      — primary comparison
  3. Reference DG .npy (T only)         — existing DG reference on T panel
  4. MD scatter (T only)                — ground truth on T panel

Usage (from repo root):
    python tutorials/argon_target_shock/plot_resolved.py
    python tutorials/argon_target_shock/plot_resolved.py --save resolved_dg.png
    python tutorials/argon_target_shock/plot_resolved.py --window 0.4
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

# ── FVM reference cases ───────────────────────────────────────────────────────
# (case_name, label, color, linestyle, linewidth, alpha)
FVM_CASES = [
    ("argon_shock_central_navier_stokes",  "Central NS",  "tab:blue",    "-",   1.5, 0.75),
    ("argon_shock_ppm_navier_stokes",      "PPM NS",      "tab:orange",  "--",  1.5, 0.75),
    ("argon_shock_muscl_navier_stokes",    "MUSCL NS",    "tab:red",     ":",   1.5, 0.75),
]

# ── SPLAY DG cases (primary comparison, plotted prominently) ──────────────────
# (case_name, label, color, linestyle, linewidth, alpha)
DG_CASES = [
    ("argon_shock_dg_p2_n800_quick",  "SPLAY DG p=2, n=800",  "tab:purple",  "-",  2.5, 1.0),
]

VARS = [
    ("rho", r"Density  [kg/m³]"),
    ("p",   "Pressure  [Pa]"),
    ("T",   "Temperature  [K]"),
]

T_SHOCK_MID = 1500.0   # K — fallback threshold (see shock_centre below)


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


def shock_centre(d):
    """
    Locate shock centre at the x [m] where T = T_SHOCK_MID (1500 K) by linear
    interpolation between the two bracketing points.  Falls back to max |dρ/dx|
    if no crossing is found (e.g., fully post-shock data).
    """
    x = d["x"]
    T = d["T"]
    crossings = np.where((T[:-1] - T_SHOCK_MID) * (T[1:] - T_SHOCK_MID) < 0)[0]
    if len(crossings):
        i    = crossings[0]
        frac = (T_SHOCK_MID - T[i]) / (T[i + 1] - T[i])
        return x[i] + frac * (x[i + 1] - x[i])
    # fallback: max |dρ/dx|
    rho  = d["rho"]
    drho = np.zeros_like(rho)
    drho[1:-1] = (rho[2:] - rho[:-2]) / (x[2:] - x[:-2])
    drho[0]    = (rho[1]  - rho[0])   / (x[1]  - x[0])
    drho[-1]   = (rho[-1] - rho[-2])  / (x[-1] - x[-2])
    return x[np.argmax(np.abs(drho))]


def rel_x_um(d, half_um):
    """Return (x_rel [µm], mask) for points within ±half_um of shock centre."""
    xc    = shock_centre(d)
    x_rel = (d["x"] - xc) * 1e6
    return x_rel, np.abs(x_rel) <= half_um


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SPLAY DG vs MD vs reference DG resolved shock comparison")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save",   default="", help="Save to file (png/pdf)")
    parser.add_argument("--window", type=float, default=0.1,
                        help="Half-width of x window [µm] (default 0.1)")
    args = parser.parse_args()

    half_um = args.window

    # ── Load FVM reference data ───────────────────────────────────────────────
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

    # ── Load SPLAY DG data ────────────────────────────────────────────────────
    dg_splay = {}
    dg_overshoot = {}   # track rho overshoot as indicator of oscillations
    for case_name, label, *_ in DG_CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip-dg] {label}  (run: ./build/splay "
                  f"tutorials/argon_target_shock/coarse/{case_name.replace('argon_shock_', '')}.yml)")
            continue
        d = load_csv(csv)
        if d is not None:
            dg_splay[case_name] = d
            xc = shock_centre(d)
            # Density overshoot: physical post-shock rho ~ 28.6 kg/m³ for this case
            rho_max  = d["rho"].max()
            overshoot = rho_max / 28.6
            dg_overshoot[case_name] = overshoot
            flag = f"  *** rho overshoot {overshoot:.2f}×" if overshoot > 1.15 else ""
            print(f"  {label}: {os.path.basename(csv)}"
                  f"  shock@{xc*1e6:.3f} µm  rho_max={rho_max:.1f}{flag}")

    if not dg_splay and not fvm_datasets:
        print("\nNo data found. Run the cases first.")
        sys.exit(1)

    # ── Load MD reference data (x in µm, already shock-centred) ──────────────
    md_data = None
    md_path = os.path.join(SCRIPT_DIR, "MD_data.txt")
    if os.path.exists(md_path):
        md_arr  = np.loadtxt(md_path, delimiter=",")
        md_data = {"x": md_arr[:, 0], "T": md_arr[:, 1]}   # x µm, T K
        print(f"  MD: {len(md_arr)} points")
    else:
        print(f"  [skip] MD data not found: {md_path}")

    # ── Load reference DG .npy (T only, x in metres, not centred) ─────────────
    ref_dg = None
    dg_dir  = os.path.join(SCRIPT_DIR, "DG")
    x_files = sorted(glob.glob(os.path.join(dg_dir, "x_*.npy")))
    T_files = sorted(glob.glob(os.path.join(dg_dir, "Temperature_*.npy")))
    if x_files and T_files:
        rx   = np.load(x_files[-1])
        rT   = np.load(T_files[-1])
        # Centre at T = T_SHOCK_MID by interpolation
        crossings = np.where((rT[:-1] - T_SHOCK_MID) * (rT[1:] - T_SHOCK_MID) < 0)[0]
        if len(crossings):
            ci   = crossings[0]
            frac = (T_SHOCK_MID - rT[ci]) / (rT[ci + 1] - rT[ci])
            rxc  = rx[ci] + frac * (rx[ci + 1] - rx[ci])
        else:
            rxc  = rx[np.argmax(np.abs(np.gradient(rT, rx)))]
        xrel = (rx - rxc) * 1e6   # µm, centred
        mask = np.abs(xrel) <= half_um
        ref_dg = {"x_rel": xrel[mask], "T": rT[mask]}
        print(f"  Reference DG (.npy): {len(rx)} pts  shock@{rxc*1e6:.3f} µm")
    else:
        print(f"  [skip] Reference DG .npy not found in {dg_dir}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_vars = len(VARS)
    fig, axes = plt.subplots(1, n_vars, figsize=(5.2 * n_vars, 5.2),
                             constrained_layout=True)

    fig.suptitle(
        "Argon shock structure — SPLAY DG p=2 vs FVM vs MD vs reference DG  (M ≈ 5.03)\n"
        f"x centred at T = {T_SHOCK_MID:.0f} K  |  window ±{half_um:.2f} µm",
        fontsize=12, fontweight="bold"
    )

    for ax, (col_key, ylabel) in zip(axes, VARS):
        # ── Layer 1: FVM reference (thin, muted) ──────────────────────────────
        for case_name, label, color, ls, lw, alpha in FVM_CASES:
            if case_name not in fvm_datasets:
                continue
            d      = fvm_datasets[case_name]
            x_rel, mask = rel_x_um(d, half_um)
            ax.plot(x_rel[mask], d[col_key][mask],
                    color=color, ls=ls, lw=lw, alpha=alpha, label=label)

        # ── Layer 2: SPLAY DG (prominent) ─────────────────────────────────────
        for case_name, label, color, ls, lw, alpha in DG_CASES:
            if case_name not in dg_splay:
                continue
            d      = dg_splay[case_name]
            x_rel, mask = rel_x_um(d, half_um)
            ax.plot(x_rel[mask], d[col_key][mask],
                    color=color, ls=ls, lw=lw, alpha=alpha, label=label, zorder=4)

        # ── Layer 3 & 4: T panel only — reference DG .npy and MD scatter ──────
        if col_key == "T":
            if ref_dg is not None:
                mask = np.abs(ref_dg["x_rel"]) <= half_um
                ax.plot(ref_dg["x_rel"][mask], ref_dg["T"][mask],
                        color="tab:brown", ls="-", lw=1.8, alpha=0.9,
                        label="Ref. DG p=2", zorder=5)

            if md_data is not None:
                mask = np.abs(md_data["x"]) <= half_um
                ax.scatter(md_data["x"][mask], md_data["T"][mask],
                           s=20, color="tab:green", marker="o",
                           alpha=0.85, zorder=6, label="MD")

        # Annotate DG overshoot on the density panel
        if col_key == "rho":
            for case_name, label, color, *_ in DG_CASES:
                if case_name not in dg_overshoot:
                    continue
                ov = dg_overshoot[case_name]
                if ov > 1.15:
                    ax.annotate(
                        f"{label.split(',')[1].strip()}\nρ overshoot {ov:.2f}×\n(AV too small)",
                        xy=(0.97, 0.97 - 0.18 * list(dg_overshoot).index(case_name)),
                        xycoords="axes fraction",
                        ha="right", va="top", fontsize=7.5,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8)
                    )

        ax.set_xlabel("x − x_shock  [µm]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-half_um, half_um)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.35)
        ax.grid(True, alpha=0.22, linestyle=":")
        ax.tick_params(labelsize=9)

        # Legend: DG + reference entries first, FVM at bottom
        handles, labels = ax.get_legend_handles_labels()
        # Sort: DG cases first, then reference/MD, then FVM
        order = []
        dg_labels  = {e[1] for e in DG_CASES}
        ref_labels = {"Ref. DG p=2", "MD"}
        for i, lbl in enumerate(labels):
            if lbl in dg_labels or lbl in ref_labels:
                order.insert(0, i)
            else:
                order.append(i)
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  fontsize=8, loc="upper right", framealpha=0.85)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()

    # ── Shock thickness: L = |Δp| / |dp/dx|_max ──────────────────────────────
    all_cases = (
        [(n, l) for n, l, *_ in DG_CASES] +
        [(n, l) for n, l, *_ in FVM_CASES]
    )
    all_data = {**dg_splay, **fvm_datasets}
    print(f"\nShock thickness  L = |Δp| / |dp/dx|_max  (±{half_um:.2f} µm window):")
    print(f"  {'Case':<35}  {'L [nm]':>8}  {'N cells':>8}  {'dx [nm]':>8}")
    for case_name, label in all_cases:
        if case_name not in all_data:
            continue
        d      = all_data[case_name]
        x_rel, mask = rel_x_um(d, half_um)
        xi     = d["x"][mask]
        pi     = d["p"][mask]
        if len(xi) < 3:
            continue
        dpi_dx = np.zeros_like(pi)
        dpi_dx[1:-1] = (pi[2:] - pi[:-2]) / (xi[2:] - xi[:-2])
        dpi_dx[0]    = (-3*pi[0] + 4*pi[1] - pi[2])    / (xi[2]  - xi[0])
        dpi_dx[-1]   = ( 3*pi[-1] - 4*pi[-2] + pi[-3]) / (xi[-1] - xi[-3])
        mx = np.max(np.abs(dpi_dx))
        if mx < 1e-10:
            continue
        L  = abs((pi[0] - pi[-1]) / mx)
        dx = (xi[-1] - xi[0]) / max(len(xi) - 1, 1)
        print(f"  {label:<35}  {L*1e9:>8.1f}  {len(d['x']):>8d}  {dx*1e9:>8.2f}")


if __name__ == "__main__":
    main()
