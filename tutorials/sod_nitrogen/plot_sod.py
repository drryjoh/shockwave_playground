#!/usr/bin/env python3
"""
Nitrogen Sod shock tube — compare SPLAY output against the exact Riemann solution.

Usage:
    python tutorials/sod_nitrogen/plot_sod.py
    python tutorials/sod_nitrogen/plot_sod.py --save sod_nitrogen.png
    python tutorials/sod_nitrogen/plot_sod.py --output_dir /path/to/output
"""

import sys, os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "output")

# ── Gas properties ────────────────────────────────────────────────────────────
GAMMA  = 1.4            # diatomic ideal gas
R_GAS  = 296.8          # J/(kg·K)  — specific gas constant for nitrogen

# ── IC (must match nitrogen_sod.yml) ─────────────────────────────────────────
RHO_L = 1.0;    P_L = 1e5;  U_L = 0.0
RHO_R = 0.125;  P_R = 1e4;  U_R = 0.0
X0    = 0.5   # diaphragm position (m)

# ── Exact Riemann solver (Toro, "Riemann Solvers", ch. 4) ─────────────────────
def exact_riemann(rhoL, pL, uL, rhoR, pR, uR, gamma, x, t, x0):
    """
    Return (rho, u, p, T) arrays for the exact Riemann solution at time t.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    def fK_rarefaction(p, pK, cK):
        return 2.0 * cK / gm1 * ((p / pK) ** (gm1 / (2.0 * gamma)) - 1.0)

    def fK_shock(p, pK, rhoK):
        AK = 2.0 / (gp1 * rhoK)
        BK = gm1 / gp1 * pK
        return (p - pK) * np.sqrt(AK / (p + BK))

    def f(p):
        fL = fK_rarefaction(p, pL, cL) if p <= pL else fK_shock(p, pL, rhoL)
        fR = fK_rarefaction(p, pR, cR) if p <= pR else fK_shock(p, pR, rhoR)
        return fL + fR + (uR - uL)

    p_star = max(1e-6 * (pL + pR), 0.5 * (pL + pR))
    for _ in range(100):
        fp = f(p_star)
        dp = max(p_star * 1e-7, 1e-3)
        fprime = (f(p_star + dp) - f(p_star - dp)) / (2.0 * dp)
        p_star -= fp / fprime
        p_star = max(p_star, 1e-6 * pL)
        if abs(fp) < 1e-12 * (pL + pR):
            break

    # u* = uR + fR(p*): fK_shock > 0 for right shock → u* > 0.
    if p_star <= pR:
        u_star = uR + fK_rarefaction(p_star, pR, cR)
    else:
        u_star = uR + fK_shock(p_star, pR, rhoR)

    if p_star <= pL:
        rho_starL = rhoL * (p_star / pL) ** (1.0 / gamma)
        c_starL   = cL   * (p_star / pL) ** (gm1 / (2.0 * gamma))
    else:
        rho_starL = rhoL * (p_star / pL + gm1 / gp1) / (gm1 / gp1 * p_star / pL + 1.0)
        c_starL   = None

    if p_star <= pR:
        rho_starR = rhoR * (p_star / pR) ** (1.0 / gamma)
        c_starR   = cR   * (p_star / pR) ** (gm1 / (2.0 * gamma))
    else:
        rho_starR = rhoR * (p_star / pR + gm1 / gp1) / (gm1 / gp1 * p_star / pR + 1.0)
        c_starR   = None

    if p_star <= pL:
        S_HL = uL - cL
        S_TL = u_star - c_starL
    else:
        S_HL = S_TL = uL - cL * np.sqrt((gp1 / (2.0 * gamma)) * p_star / pL
                                          + gm1 / (2.0 * gamma))

    if p_star <= pR:
        S_HR = uR + cR
        S_TR = u_star + c_starR
    else:
        S_HR = S_TR = uR + cR * np.sqrt((gp1 / (2.0 * gamma)) * p_star / pR
                                          + gm1 / (2.0 * gamma))

    S_contact = u_star
    xi = (x - x0) / t

    rho_ex = np.empty_like(x)
    u_ex   = np.empty_like(x)
    p_ex   = np.empty_like(x)

    for k in range(len(x)):
        s = xi[k]
        if s <= S_HL:
            rho_ex[k], u_ex[k], p_ex[k] = rhoL, uL, pL
        elif s <= S_TL:
            u_f = 2.0 / gp1 * (cL + gm1 / 2.0 * uL + s)
            c_f = cL - gm1 / 2.0 * (u_f - uL)
            rho_ex[k] = rhoL * (c_f / cL) ** (2.0 / gm1)
            u_ex[k]   = u_f
            p_ex[k]   = pL   * (c_f / cL) ** (2.0 * gamma / gm1)
        elif s <= S_contact:
            rho_ex[k], u_ex[k], p_ex[k] = rho_starL, u_star, p_star
        elif s <= S_TR:
            rho_ex[k], u_ex[k], p_ex[k] = rho_starR, u_star, p_star
        elif s <= S_HR:
            u_f = 2.0 / gp1 * (-cR + gm1 / 2.0 * uR + s)
            c_f = cR + gm1 / 2.0 * (u_f - uR)
            rho_ex[k] = rhoR * (c_f / cR) ** (2.0 / gm1)
            u_ex[k]   = u_f
            p_ex[k]   = pR   * (c_f / cR) ** (2.0 * gamma / gm1)
        else:
            rho_ex[k], u_ex[k], p_ex[k] = rhoR, uR, pR

    T_ex = p_ex / (rho_ex * R_GAS)
    return rho_ex, u_ex, p_ex, T_ex


def find_latest_csv(case_dir):
    files = sorted(glob.glob(os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


def load_csv(path):
    rows, header, time_val = [], None, 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("time="):
                        try:
                            time_val = float(tok.split("=")[1])
                        except ValueError:
                            pass
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            rows.append([float(v) for v in line.split(",")])
    if not rows:
        return None, time_val
    arr = np.array(rows)
    d   = {col: arr[:, k] for k, col in enumerate(header)}
    idx = np.argsort(d["x"])
    return {k: v[idx] for k, v in d.items()}, time_val


def main():
    parser = argparse.ArgumentParser(description="Nitrogen Sod shock tube plot")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--save", default="", help="Save figure to file")
    args = parser.parse_args()

    case_dir = os.path.join(args.output_dir, "nitrogen_sod")
    csv_path = find_latest_csv(case_dir)
    if csv_path is None:
        print("No snapshot found. Run first:")
        print("  bash tutorials/sod_nitrogen/run.sh --no-plot")
        sys.exit(1)

    d, t_sim = load_csv(csv_path)
    print(f"Loaded: {os.path.basename(csv_path)}  t={t_sim:.4e} s")

    x_ex  = np.linspace(0.0, 1.0, 2000)
    rho_ex, u_ex, p_ex, T_ex = exact_riemann(
        RHO_L, P_L, U_L, RHO_R, P_R, U_R, GAMMA, x_ex, t_sim, X0)

    x_sim = d["x"]
    rho_int = np.interp(x_sim, x_ex, rho_ex)
    u_int   = np.interp(x_sim, x_ex, u_ex)
    p_int   = np.interp(x_sim, x_ex, p_ex)

    dx_sim = x_sim[1] - x_sim[0]
    L1_rho = np.sum(np.abs(d["rho"] - rho_int)) * dx_sim
    L1_u   = np.sum(np.abs(d["u"]   - u_int))   * dx_sim
    L1_p   = np.sum(np.abs(d["p"]   - p_int))   * dx_sim

    print(f"\nL1 errors (integrated):  rho={L1_rho:.4e}  u={L1_u:.4e}  p={L1_p:.4e}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Nitrogen Sod shock tube — MUSCL+HLLC vs exact Riemann  (t={t_sim:.3e} s, N=400)",
        fontsize=11, fontweight="bold")

    panels = [
        ("rho", rho_ex, r"Density  [kg/m³]"),
        ("u",   u_ex,   "Velocity  [m/s]"),
        ("p",   p_ex,   "Pressure  [Pa]"),
        ("T",   T_ex,   "Temperature  [K]"),
    ]

    for ax, (key, ex_arr, ylabel) in zip(axes, panels):
        ax.plot(x_ex, ex_arr, "k-", lw=1.5, label="Exact", alpha=0.8)
        ax.plot(d["x"], d[key], "tab:orange", ls="--", lw=1.2, label="SPLAY MUSCL")
        ax.set_xlabel("x  [m]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, ls=":")
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
