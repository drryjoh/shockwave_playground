// SPLAY test: Sod shock tube initialization and exact Riemann consistency.
//
// 1. Verify init_step sets the correct left/right primitive state.
// 2. Verify that the star-region pressure from the exact Rankine–Hugoniot
//    relations is self-consistent for both argon and nitrogen.
#include "splay/gas.hpp"
#include "splay/mesh.hpp"
#include "splay/state.hpp"
#include "splay/config.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>

static void check(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
}
static bool near_rel(double a, double b, double tol) {
    return std::abs(a - b) / (std::abs(b) + 1e-30) < tol;
}

// ── init_step unit test ───────────────────────────────────────────────────────
void test_init_step(const splay::GasModel& gas) {
    // Build a tiny 10-cell mesh on [0, 1].
    splay::GridConfig gc;
    gc.x_min      = 0.0;
    gc.x_max      = 1.0;
    gc.n_cells    = 10;
    gc.ghost_cells = 1;
    splay::Mesh m = splay::Mesh::build(gc, 0, 1);
    splay::State s(m.n_total);

    const double rhoL = 1.0,  pL = 1e5, uL = 0.0;
    const double rhoR = 0.125, pR = 1e4, uR = 0.0;
    const double x0   = 0.5;

    splay::init_step(s, m, gas, rhoL, pL, uL, rhoR, pR, uR, x0);

    for (int i = m.interior_begin(); i < m.interior_end(); ++i) {
        const bool left = (m.x_cell[i] <= x0);
        const double rho_exp = left ? rhoL : rhoR;
        const double p_exp   = left ? pL   : pR;
        const double u_exp   = left ? uL   : uR;
        const double T_exp   = p_exp / (gas.R * rho_exp);

        check(near_rel(s.rho[i],  rho_exp, 1e-12), "init_step rho");
        check(near_rel(s.p[i],    p_exp,   1e-12), "init_step p");
        check(near_rel(s.u[i],    u_exp,   1e-12), "init_step u");
        check(near_rel(s.T[i],    T_exp,   1e-10), "init_step T");
        check(near_rel(s.rhou[i], rho_exp * u_exp, 1e-12), "init_step rhou");
    }
    std::cout << "  [PASS] init_step (" << gas.name << ")\n";
}

// ── Exact Riemann star-state consistency ─────────────────────────────────────
// Verify Rankine–Hugoniot and entropy relations across each wave for given IC.
void test_riemann_star(const splay::GasModel& gas) {
    const double gamma = gas.gamma;
    const double gm1   = gamma - 1.0;
    const double gp1   = gamma + 1.0;

    const double rhoL = 1.0,   pL = 1e5, uL = 0.0;
    const double rhoR = 0.125, pR = 1e4, uR = 0.0;

    const double cL = gas.sound_speed(pL, rhoL);
    const double cR = gas.sound_speed(pR, rhoR);

    // ── Solve for p* ──
    auto fK = [&](double p, double pK, double rhoK, double cK) {
        if (p <= pK) return 2.0*cK/gm1 * (std::pow(p/pK, gm1/(2.0*gamma)) - 1.0);
        const double AK = 2.0/(gp1*rhoK), BK = gm1/gp1*pK;
        return (p - pK) * std::sqrt(AK / (p + BK));
    };
    auto ftot = [&](double p) { return fK(p,pL,rhoL,cL) + fK(p,pR,rhoR,cR); };

    double p_star = 0.5*(pL + pR);
    for (int iter = 0; iter < 100; ++iter) {
        const double fp  = ftot(p_star);
        const double dp  = p_star * 1e-7;
        const double fps = (ftot(p_star+dp) - ftot(p_star-dp)) / (2.0*dp);
        p_star -= fp / fps;
        if (p_star < 1e-10*pL) p_star = 1e-10*pL;
        if (std::abs(fp) < 1e-12*(pL+pR)) break;
    }

    // u* from the right wave (right shock: fK_shock > 0, so u* > uR = 0).
    // Equivalently, from the left rarefaction: u* = uL - fK(p*, WL)
    // (the sign flip relative to the right wave is Toro's convention).
    const double u_star = uR + fK(p_star, pR, rhoR, cR);

    // Left wave is a rarefaction (p_star < pL) — check entropy relation.
    check(p_star < pL, "p_star < pL (left rarefaction)");
    const double rho_sL = rhoL * std::pow(p_star/pL, 1.0/gamma);
    const double c_sL   = cL   * std::pow(p_star/pL, gm1/(2.0*gamma));

    // Entropy: p/rho^gamma = const across rarefaction
    check(near_rel(p_star / std::pow(rho_sL, gamma),
                   pL     / std::pow(rhoL,   gamma), 1e-8),
          "left rarefaction entropy");

    // Riemann invariant u + 2c/(γ-1) = const on C_+ characteristics across left rarefaction.
    // Holds algebraically when u* = uL - fL(p*); verified here numerically.
    check(near_rel(uL + 2.0*cL/gm1, u_star + 2.0*c_sL/gm1, 1e-8),
          "left rarefaction Riemann invariant");

    // Right wave is a shock (p_star > pR) — check RH relations.
    check(p_star > pR, "p_star > pR (right shock)");
    const double rho_sR = rhoR * (p_star/pR + gm1/gp1) / (gm1/gp1 * p_star/pR + 1.0);

    // Rankine–Hugoniot: mass, momentum, energy across right shock.
    const double S_R = uR + cR * std::sqrt(gp1/(2.0*gamma) * p_star/pR + gm1/(2.0*gamma));
    const double v1  = uR     - S_R;  // pre-shock velocity in shock frame
    const double v2  = u_star - S_R;  // post-shock velocity in shock frame

    check(near_rel(rhoR*v1,          rho_sR*v2,              1e-6), "RH mass");
    check(near_rel(rhoR*v1*v1 + pR,  rho_sR*v2*v2 + p_star,  1e-5), "RH momentum");

    const double cvR  = gas.cv;
    const double e1   = cvR * gas.temperature(rhoR,  pR)     + 0.5*v1*v1;
    const double e2   = cvR * gas.temperature(rho_sR,p_star) + 0.5*v2*v2;
    const double flux1 = (rhoR  * e1 + pR)      * v1;
    const double flux2 = (rho_sR* e2 + p_star)  * v2;
    check(near_rel(flux1, flux2, 1e-5), "RH energy");

    std::cout << "  [PASS] exact Riemann star-state (" << gas.name
              << ")  p*=" << p_star << " Pa  u*=" << u_star << " m/s\n";
}

int main() {
    std::cout << "=== test_sod ===\n";

    auto ar = splay::GasModel::from_name("argon");
    auto n2 = splay::GasModel::from_name("nitrogen");

    test_init_step(ar);
    test_init_step(n2);

    test_riemann_star(ar);
    test_riemann_star(n2);

    std::cout << "=== PASSED ===\n";
    return 0;
}
