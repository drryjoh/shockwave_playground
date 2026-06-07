#!/usr/bin/env python3
"""
Sod shock tube regression test.

Runs the argon and nitrogen Sod cases with SPLAY, loads the final snapshot,
and checks that L1 errors in density, velocity, and pressure against the
exact Riemann solution are below acceptable tolerances for N=400 MUSCL+HLLC.

Exit code 0 on pass, 1 on failure.

Usage (from repo root):
    python tests/test_sod_regression.py
    python tests/test_sod_regression.py --build_dir build --output_dir output
"""

import sys, os, glob, argparse, subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ── Exact Riemann solver ──────────────────────────────────────────────────────
def exact_riemann(rhoL, pL, uL, rhoR, pR, uR, gamma, x, t, x0=0.5):
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    cL  = np.sqrt(gamma * pL / rhoL)
    cR  = np.sqrt(gamma * pR / rhoR)

    def fK(p, pK, rhoK, cK):
        if p <= pK:
            return 2.0 * cK / gm1 * ((p / pK) ** (gm1 / (2.0 * gamma)) - 1.0)
        AK = 2.0 / (gp1 * rhoK)
        BK = gm1 / gp1 * pK
        return (p - pK) * np.sqrt(AK / (p + BK))

    def ftot(p):
        return fK(p, pL, rhoL, cL) + fK(p, pR, rhoR, cR) + (uR - uL)

    p_star = 0.5 * (pL + pR)
    for _ in range(100):
        fp  = ftot(p_star)
        dp  = max(p_star * 1e-7, 1e-3)
        fps = (ftot(p_star + dp) - ftot(p_star - dp)) / (2.0 * dp)
        p_star -= fp / fps
        p_star = max(p_star, 1e-10 * pL)
        if abs(fp) < 1e-12 * (pL + pR):
            break

    u_star = uR + fK(p_star, pR, rhoR, cR)

    def star_rho(p_st, pk, rhok, gamma):
        if p_st <= pk:
            return rhok * (p_st / pk) ** (1.0 / gamma)
        return rhok * (p_st / pk + gm1 / gp1) / (gm1 / gp1 * p_st / pk + 1.0)

    rho_sL = star_rho(p_star, pL, rhoL, gamma)
    rho_sR = star_rho(p_star, pR, rhoR, gamma)

    c_sL = cL * (p_star / pL) ** (gm1 / (2.0 * gamma)) if p_star <= pL else None
    c_sR = cR * (p_star / pR) ** (gm1 / (2.0 * gamma)) if p_star <= pR else None

    S_HL = uL - cL                    if p_star <= pL else (
           uL - cL * np.sqrt(gp1 / (2.0 * gamma) * p_star / pL + gm1 / (2.0 * gamma)))
    S_TL = u_star - c_sL              if p_star <= pL else S_HL

    S_HR = uR + cR                    if p_star <= pR else (
           uR + cR * np.sqrt(gp1 / (2.0 * gamma) * p_star / pR + gm1 / (2.0 * gamma)))
    S_TR = u_star + c_sR              if p_star <= pR else S_HR

    xi = (x - x0) / t
    rho_ex = np.empty_like(x); u_ex = np.empty_like(x); p_ex = np.empty_like(x)

    for k in range(len(x)):
        s = xi[k]
        if s <= S_HL:
            rho_ex[k], u_ex[k], p_ex[k] = rhoL, uL, pL
        elif s <= S_TL:
            u_f = 2.0/gp1 * (cL + gm1/2.0*uL + s)
            c_f = cL - gm1/2.0*(u_f - uL)
            rho_ex[k] = rhoL*(c_f/cL)**(2.0/gm1); u_ex[k]=u_f
            p_ex[k]   = pL  *(c_f/cL)**(2.0*gamma/gm1)
        elif s <= u_star:
            rho_ex[k], u_ex[k], p_ex[k] = rho_sL, u_star, p_star
        elif s <= S_TR:
            rho_ex[k], u_ex[k], p_ex[k] = rho_sR, u_star, p_star
        elif s <= S_HR:
            u_f = 2.0/gp1*(-cR + gm1/2.0*uR + s)
            c_f = cR + gm1/2.0*(u_f - uR)
            rho_ex[k] = rhoR*(c_f/cR)**(2.0/gm1); u_ex[k]=u_f
            p_ex[k]   = pR  *(c_f/cR)**(2.0*gamma/gm1)
        else:
            rho_ex[k], u_ex[k], p_ex[k] = rhoR, uR, pR

    return rho_ex, u_ex, p_ex


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv(path):
    rows, header, t = [], None, 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("time="):
                        try: t = float(tok.split("=")[1])
                        except ValueError: pass
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]; continue
            rows.append([float(v) for v in line.split(",")])
    if not rows: return None, t
    arr = np.array(rows)
    d   = {col: arr[:, k] for k, col in enumerate(header)}
    idx = np.argsort(d["x"])
    return {k: v[idx] for k, v in d.items()}, t


def find_latest_csv(case_dir):
    files = sorted(glob.glob(os.path.join(case_dir, "snapshots", "step_*_combined.csv")))
    return files[-1] if files else None


# ── Single-case runner ────────────────────────────────────────────────────────
def run_case(splay_bin, yml_path, output_dir, case_name,
             rhoL, pL, uL, rhoR, pR, uR, gamma,
             L1_tol_rho, L1_tol_u, L1_tol_p):
    print(f"\n{'='*60}")
    print(f"  Case: {case_name}")
    print(f"{'='*60}")

    # Run solver
    result = subprocess.run(
        [splay_bin, yml_path],
        cwd=os.path.dirname(splay_bin),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  FAIL: splay exited with code {result.returncode}")
        return False

    # Load final snapshot
    csv_path = find_latest_csv(os.path.join(output_dir, case_name))
    if csv_path is None:
        print(f"  FAIL: no snapshot found in {output_dir}/{case_name}/snapshots/")
        return False

    d, t_sim = load_csv(csv_path)
    print(f"  Snapshot: {os.path.basename(csv_path)}  t={t_sim:.4e} s")

    # Exact solution
    x_ex  = np.linspace(0.0, 1.0, 4000)
    rho_ex, u_ex, p_ex = exact_riemann(rhoL, pL, uL, rhoR, pR, uR, gamma, x_ex, t_sim)

    # Interpolate exact to simulation x-grid
    x_sim   = d["x"]
    dx      = x_sim[1] - x_sim[0]
    L1_rho  = np.sum(np.abs(d["rho"] - np.interp(x_sim, x_ex, rho_ex))) * dx
    L1_u    = np.sum(np.abs(d["u"]   - np.interp(x_sim, x_ex, u_ex)))   * dx
    L1_p    = np.sum(np.abs(d["p"]   - np.interp(x_sim, x_ex, p_ex)))   * dx

    print(f"  L1 rho = {L1_rho:.4e}  (tol={L1_tol_rho:.2e})")
    print(f"  L1 u   = {L1_u:.4e}  (tol={L1_tol_u:.2e})")
    print(f"  L1 p   = {L1_p:.4e}  (tol={L1_tol_p:.2e})")

    passed = (L1_rho < L1_tol_rho and L1_u < L1_tol_u and L1_p < L1_tol_p)
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir",  default=os.path.join(REPO_ROOT, "build"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "output"))
    args = parser.parse_args()

    splay = os.path.join(args.build_dir, "splay")
    if not os.path.isfile(splay):
        print(f"ERROR: splay binary not found at {splay}")
        print("Build first: cmake --build build -j")
        sys.exit(1)

    # Shared IC (same for both gases, only γ differs)
    RHO_L, P_L, U_L = 1.0, 1e5, 0.0
    RHO_R, P_R, U_R = 0.125, 1e4, 0.0

    # Tolerances for N=400 MUSCL+HLLC
    # These are loose enough to be robust but tight enough to catch major regressions.
    TOL_RHO, TOL_U, TOL_P = 0.012, 8.0, 1200.0

    all_pass = True

    all_pass &= run_case(
        splay,
        os.path.join(REPO_ROOT, "tutorials", "sod_argon",    "argon_sod.yml"),
        args.output_dir, "argon_sod",
        RHO_L, P_L, U_L, RHO_R, P_R, U_R,
        gamma=5.0/3.0,
        L1_tol_rho=TOL_RHO, L1_tol_u=TOL_U, L1_tol_p=TOL_P,
    )

    all_pass &= run_case(
        splay,
        os.path.join(REPO_ROOT, "tutorials", "sod_nitrogen", "nitrogen_sod.yml"),
        args.output_dir, "nitrogen_sod",
        RHO_L, P_L, U_L, RHO_R, P_R, U_R,
        gamma=1.4,
        L1_tol_rho=TOL_RHO, L1_tol_u=TOL_U, L1_tol_p=TOL_P,
    )

    print(f"\n{'='*60}")
    print(f"  Overall: {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
