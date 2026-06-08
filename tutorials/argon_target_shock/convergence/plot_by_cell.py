#!/usr/bin/env python3
"""
SPLAY – DG convergence: plot shock structure cell-by-cell.

Each DG cell is drawn as a separate line segment (GLL nodes connected within
the cell), with a NaN break between adjacent cells so the DG discontinuities
at element faces are clearly visible as gaps.

Panels: density [kg/m³] and temperature [K], both centred at T = 1500 K.

Cases plotted:
  - SPLAY DG p=2 convergence cases (n=400, 800) from output/
  - Reference DG p=2 .npy  (Temperature only, on T panel)

Usage (from repo root):
    python tutorials/argon_target_shock/convergence/plot_by_cell.py
    python tutorials/argon_target_shock/convergence/plot_by_cell.py --save output/conv_by_cell.png
    python tutorials/argon_target_shock/convergence/plot_by_cell.py --window 0.15
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TUTORIAL_DIR   = os.path.join(REPO_ROOT, "tutorials", "argon_target_shock")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

# ── Convergence cases (in desired legend order, coarse → fine) ────────────────
# (case_name, label, color)
CONV_CASES = [
    ("argon_shock_dg_p2_n100_conv",  "DG p=2, n=100",  "#aaaaff"),
    ("argon_shock_dg_p2_n200_conv",  "DG p=2, n=200",  "#7777dd"),
    ("argon_shock_dg_p2_n400_conv",  "DG p=2, n=400",  "#4444bb"),
    ("argon_shock_dg_p2_n800_conv",  "DG p=2, n=800",  "#110099"),
]

T_SHOCK_MID = 1500.0   # K — shock centring threshold


# ── I/O helpers ──────────────────────────────────────────────────────────────
def find_latest_csv(case_dir):
    files = sorted(glob.glob(os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


def load_csv(path):
    """Load CSV written by write_csv_dg.  Returns dict of arrays, sorted by x."""
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
    return {col: arr[:, k] for k, col in enumerate(header)}


def shock_centre(d):
    """x [m] where T crosses T_SHOCK_MID (1500 K), interpolated."""
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


def make_by_cell(d, xc, half_um, col):
    """
    Build (x_plot [µm], y_plot) arrays with NaN between each cell.

    The CSV already contains N_PLOT_PER_CELL evaluated points per cell
    (polynomial evaluated in C++), so we just group by cell index and
    append a NaN break between cells.
    """
    x_rel = (d["x"] - xc) * 1e6
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DG convergence — plot by cell with NaN breaks at element faces")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save",   default="", help="Save to file (png/pdf)")
    parser.add_argument("--window", type=float, default=0.1,
                        help="Half-width of x window [µm] (default 0.1)")
    args = parser.parse_args()

    half_um = args.window

    # ── Load convergence DG data ──────────────────────────────────────────────
    datasets = {}
    for case_name, label, color in CONV_CASES:
        csv = find_latest_csv(os.path.join(args.output_dir, case_name))
        if csv is None:
            print(f"  [skip] {label}  (not found in {args.output_dir})")
            continue
        d = load_csv(csv)
        if d is None:
            continue
        if np.any(np.isnan(d["T"])):
            print(f"  [NaN]  {label}  (blew up — sub-cell shock, needs AV or finer mesh)")
            continue
        xc = shock_centre(d)
        print(f"  {label}: {os.path.basename(csv)}"
              f"  shock@{xc*1e6:.3f} µm  rho_max={d['rho'].max():.1f}")
        datasets[case_name] = (d, label, color)

    if not datasets:
        print("\nNo valid data found. Run the convergence cases first.")
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
            ci   = crossings[0]
            frac = (T_SHOCK_MID - rT[ci]) / (rT[ci + 1] - rT[ci])
            rxc  = rx[ci] + frac * (rx[ci + 1] - rx[ci])
        else:
            rxc  = rx[np.argmax(np.abs(np.gradient(rT, rx)))]
        ref_xrel = (rx - rxc) * 1e6
        ref_mask = np.abs(ref_xrel) <= half_um
        ref_dg   = (ref_xrel[ref_mask], rT[ref_mask])
        print(f"  Reference DG (.npy): {len(rx)} pts  shock@{rxc*1e6:.3f} µm")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_rho = fig.add_subplot(gs[0])
    ax_T   = fig.add_subplot(gs[1])

    for case_name, (d, label, color) in datasets.items():
        xc = shock_centre(d)
        n_str = case_name.split("_n")[1].split("_")[0]   # e.g. "800"

        xr, yr = make_by_cell(d, xc, half_um, "rho")
        xT, yT = make_by_cell(d, xc, half_um, "T")

        ax_rho.plot(xr, yr, color=color, lw=1.2, label=label)
        ax_T  .plot(xT, yT, color=color, lw=1.2, label=label)

    # Reference DG (T panel only)
    if ref_dg is not None:
        ax_T.plot(ref_dg[0], ref_dg[1],
                  color="tab:brown", lw=1.5, ls="--", label="Ref. DG p=2", zorder=5)

    # ── Formatting ────────────────────────────────────────────────────────────
    for ax, ylabel in [(ax_rho, r"Density  [kg/m³]"), (ax_T, "Temperature  [K]")]:
        ax.set_xlabel("x − x_shock  [µm]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-half_um, half_um)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")

    ax_rho.set_title("Density — by cell (NaN breaks at faces)")
    ax_T  .set_title("Temperature — by cell (NaN breaks at faces)")

    fig.suptitle(
        "Argon shock convergence — SPLAY DG p=2  (M ≈ 5.03)\n"
        f"Each segment = one DG cell  |  window ±{half_um:.2f} µm"
        f"  |  centred at T = {T_SHOCK_MID:.0f} K",
        fontsize=11, fontweight="bold"
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
