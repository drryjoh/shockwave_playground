#!/usr/bin/env python3
"""
SPLAY – Quick-run shock structure plotter.

Shows T, p, u, and rho as functions of x across all saved time snapshots
for each completed quick case, using a colour gradient to indicate time.

Usage (from any directory):
    python tutorials/argon_target_shock/plot_quick.py
    python tutorials/argon_target_shock/plot_quick.py --case muscl  # filter one case
    python tutorials/argon_target_shock/plot_quick.py --save shock_evolution.png
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "output")

QUICK_CASES = [
    ("argon_shock_muscl_euler_quick", "MUSCL Euler", "Blues"),
    ("argon_shock_ppm_euler_quick",   "PPM Euler",   "Oranges"),
]

VARS = [
    ("T",   "Temperature [K]"),
    ("p",   "Pressure [Pa]"),
    ("u",   "Velocity [m/s]"),
    ("rho", r"Density [kg/m³]"),
]


# ── I/O helpers ───────────────────────────────────────────────────────────────
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


def meta_from_comment(path):
    """Return (time, step) from the first comment line of a CSV."""
    with open(path) as f:
        line = f.readline().strip()
    t, s = None, None
    for tok in line.split():
        if tok.startswith("time="): t = float(tok.split("=")[1])
        if tok.startswith("step="): s = int(tok.split("=")[1])
    return t, s


def load_all_snapshots(case_dir):
    """Return sorted list of (time, step, data_dict) for all combined CSVs."""
    pattern = os.path.join(case_dir, "snapshots", "step_*_combined.csv")
    files   = sorted(glob.glob(pattern))
    snaps   = []
    for f in files:
        t, s = meta_from_comment(f)
        d    = load_csv(f)
        if d is not None and t is not None:
            snaps.append((t, s, d))
    snaps.sort(key=lambda x: x[0])
    return snaps


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SPLAY quick-run time-evolution plotter")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--case",  default="",
                        help="Filter cases: 'muscl', 'ppm', or leave blank for both")
    parser.add_argument("--save",  default="",   help="Save figure to file (png/pdf)")
    parser.add_argument("--zoom",  action="store_true",
                        help="Zoom x-axis tightly around the shock")
    args = parser.parse_args()

    # Select cases to plot
    cases = [c for c in QUICK_CASES
             if not args.case or args.case.lower() in c[0].lower()]
    if not cases:
        print(f"No cases match --case={args.case!r}")
        sys.exit(1)

    # Load data
    all_data = {}
    for case_name, label, cmap_name in cases:
        snaps = load_all_snapshots(os.path.join(args.output_dir, case_name))
        if not snaps:
            print(f"  [skip] No snapshots for {case_name}  "
                  f"(run:  bash run_serial.sh --quick)")
            continue
        all_data[case_name] = (label, cmap_name, snaps)
        print(f"  {label}: {len(snaps)} snapshots  "
              f"t={snaps[0][0]:.2e} → {snaps[-1][0]:.2e} s")

    if not all_data:
        print("Nothing to plot.")
        sys.exit(1)

    n_cases = len(all_data)
    n_vars  = len(VARS)

    # Layout: n_vars rows × n_cases columns, plus a narrow colorbar column per case.
    # Use gridspec for precise control.
    import matplotlib.gridspec as gridspec

    col_widths = []
    for _ in range(n_cases):
        col_widths += [6, 0.35]           # plot col + cbar col
    fig = plt.figure(figsize=(sum(col_widths) + 0.4 * n_cases, 3.2 * n_vars))
    fig.suptitle("SPLAY – Shock structure evolution over time", fontsize=13,
                 fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(n_vars, 2 * n_cases, figure=fig,
                           width_ratios=col_widths,
                           hspace=0.42, wspace=0.12)

    for col_idx, (case_name, (label, cmap_name, snaps)) in enumerate(all_data.items()):
        t_all        = np.array([s[0] for s in snaps])
        t_min, t_max = t_all.min(), t_all.max()
        cmap         = plt.get_cmap(cmap_name)
        norm         = mcolors.Normalize(vmin=t_min, vmax=t_max)

        plot_col = col_idx * 2          # gridspec column for the plot
        cbar_col = col_idx * 2 + 1     # gridspec column for the colorbar

        plot_axes = []
        for row, (col_key, ylabel) in enumerate(VARS):
            ax = fig.add_subplot(gs[row, plot_col])
            plot_axes.append(ax)

            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", pad=6)

            for t, step, d in snaps:
                color   = cmap(norm(t))
                x_um    = d["x"] * 1e6
                is_edge = (t == t_min or t == t_max)
                lw      = 2.0 if is_edge else 0.9
                alpha   = 1.0 if is_edge else 0.55
                ax.plot(x_um, d[col_key], color=color, lw=lw, alpha=alpha)

            ax.set_xlabel("x  [µm]", fontsize=9)
            ax.set_ylabel(ylabel,    fontsize=9)
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=8)

            if args.zoom:
                d_last = snaps[-1][2]
                rho    = d_last["rho"]
                x_um_z = d_last["x"] * 1e6
                lo_v   = rho.min() + 0.15 * (rho.max() - rho.min())
                hi_v   = rho.min() + 0.85 * (rho.max() - rho.min())
                mask   = (rho > lo_v) & (rho < hi_v)
                if mask.any():
                    cx  = x_um_z[mask]
                    pad = max(0.4 * (cx.max() - cx.min()), 0.3)
                    ax.set_xlim(cx.min() - pad, cx.max() + pad)

        # Single colourbar spanning all rows for this case
        cbar_ax = fig.add_subplot(gs[:, cbar_col])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("time [s]", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
