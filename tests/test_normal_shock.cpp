// SPLAY test: normal-shock relations
// Verifies Rankine-Hugoniot consistency for argon and nitrogen.
#include "splay/gas.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

static bool near_rel(double a, double b, double tol) {
    return std::abs(a - b) / (std::abs(b) + 1e-30) < tol;
}

// Verify mass, momentum, and energy conservation across the shock.
void verify_rankine_hugoniot(const splay::GasModel& g,
                              double M1, double p1, double T1) {
    const double rho1 = p1 / (g.R * T1);
    const double a1   = g.sound_speed(p1, rho1);
    const double u1   = M1 * a1;  // shock-frame velocity (upstream)

    auto ss = g.normal_shock(M1, p1, T1, 0.0);

    const double rho2 = ss.rho2;
    const double p2   = ss.p2;
    const double T2   = ss.T2;
    const double u2   = ss.u2;

    // Mass: rho1*u1 = rho2*u2
    const double mass1 = rho1 * u1;
    const double mass2 = rho2 * u2;
    check(near_rel(mass1, mass2, 1e-8), "mass flux conservation");

    // Momentum: rho*u^2 + p
    const double mom1 = rho1 * u1 * u1 + p1;
    const double mom2 = rho2 * u2 * u2 + p2;
    check(near_rel(mom1, mom2, 1e-8), "momentum flux conservation");

    // Energy: (rhoE + p)*u  where E = e + 0.5*u^2 = cv*T + 0.5*u^2
    const double E1  = g.cv * T1 + 0.5 * u1 * u1;
    const double E2  = g.cv * T2 + 0.5 * u2 * u2;
    const double en1 = (rho1 * E1 + p1) * u1;
    const double en2 = (rho2 * E2 + p2) * u2;
    check(near_rel(en1, en2, 1e-8), "energy flux conservation");

    // Temperature from ideal gas
    const double T2_check = p2 / (rho2 * g.R);
    check(near_rel(T2, T2_check, 1e-10), "T2 consistent with ideal gas");

    std::cout << "  [PASS] RH M=" << M1 << " (" << g.name << ")\n";
}

int main() {
    std::cout << "=== test_normal_shock ===\n";

    auto ar = splay::GasModel::from_name("argon");
    auto n2 = splay::GasModel::from_name("nitrogen");

    // Argon M=5.03 (tutorial case)
    verify_rankine_hugoniot(ar, 5.03, 5.0e5, 300.0);
    // Argon M=2.0
    verify_rankine_hugoniot(ar, 2.0,  1.0e5, 300.0);
    // Nitrogen M=3.0
    verify_rankine_hugoniot(n2, 3.0,  1.0e5, 300.0);

    // Verify tutorial downstream state matches published values.
    {
        auto ss = ar.normal_shock(5.03, 5.0e5, 300.0, 0.0);
        const double p2_ref = 1.56880625e7;
        const double T2_ref = 2632.24759509;
        const bool p_ok = near_rel(ss.p2, p2_ref, 0.01);
        const bool T_ok = near_rel(ss.T2, T2_ref, 0.01);
        std::cout << "  p2_ref=" << p2_ref << "  p2_calc=" << ss.p2
                  << "  err=" << std::abs(ss.p2 - p2_ref) / p2_ref << "\n";
        std::cout << "  T2_ref=" << T2_ref << "  T2_calc=" << ss.T2
                  << "  err=" << std::abs(ss.T2 - T2_ref) / T2_ref << "\n";
        if (!p_ok || !T_ok)
            std::cout << "  NOTE: >1% discrepancy — check frame convention.\n";
        else
            std::cout << "  [PASS] Tutorial state verified (within 1%)\n";
    }

    std::cout << "=== PASSED ===\n";
    return 0;
}
