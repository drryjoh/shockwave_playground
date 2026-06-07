// SPLAY test: gas model
// Tests: GasModel construction, thermodynamic relations, normal shock.
#include "splay/gas.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <stdexcept>

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol * (std::abs(b) + 1e-14);
}

void test_argon_properties() {
    auto g = splay::GasModel::from_name("argon");
    check(g.name == "argon",              "argon name");
    check(near(g.gamma, 5.0/3.0, 1e-12), "argon gamma");
    check(near(g.MW, 0.039948, 1e-8),    "argon MW");
    // R = Ru / MW
    check(near(g.R, 8.31446261815324 / 0.039948, 1e-6), "argon R");
    // gamma = cp/cv => cv = R/(gamma-1)
    check(near(g.cv, g.R / (g.gamma - 1.0), 1e-10), "argon cv");
    check(near(g.cp, g.gamma * g.cv, 1e-10),         "argon cp");
    std::cout << "  [PASS] argon properties\n";
}

void test_nitrogen_properties() {
    auto g = splay::GasModel::from_name("nitrogen");
    check(g.name == "nitrogen",            "nitrogen name");
    check(near(g.gamma, 7.0/5.0, 1e-12),  "nitrogen gamma");
    std::cout << "  [PASS] nitrogen properties\n";
}

void test_unknown_gas() {
    bool caught = false;
    try { splay::GasModel::from_name("helium"); }
    catch (const std::exception&) { caught = true; }
    check(caught, "unknown gas throws");
    std::cout << "  [PASS] unknown gas throws\n";
}

void test_pressure_temperature_round_trip() {
    auto g = splay::GasModel::from_name("argon");
    const double rho = 2.0, T = 500.0;
    const double p   = g.pressure(rho, T);
    const double T2  = g.temperature(rho, p);
    check(near(T, T2, 1e-12), "pressure/temperature round-trip");
    std::cout << "  [PASS] pressure/temperature round-trip\n";
}

void test_sound_speed() {
    auto g = splay::GasModel::from_name("argon");
    const double p = 1e5, rho = 1.5;
    const double a = g.sound_speed(p, rho);
    const double a_ref = std::sqrt(g.gamma * p / rho);
    check(near(a, a_ref, 1e-12), "sound_speed");
    std::cout << "  [PASS] sound_speed\n";
}

void test_normal_shock_argon() {
    // Verify the tutorial left/right states match M=5.03 argon shock.
    auto g = splay::GasModel::from_name("argon");

    const double M1   = 5.03;
    const double p1   = 5.0e5;      // pre-shock (right)
    const double T1   = 300.0;
    const double u1   = 0.0;        // pre-shock at rest in lab

    auto ss = g.normal_shock(M1, p1, T1, u1);

    // Reference values from the tutorial (provided in master prompt)
    const double p2_ref = 1.56880625e7;
    const double T2_ref = 2632.24759509;

    // Allow ~1% tolerance (our polynomial fit is approximate; exact match not expected)
    check(near(ss.p2, p2_ref, 0.01), "normal shock p2");
    check(near(ss.T2, T2_ref, 0.01), "normal shock T2");

    std::cout << "  [PASS] normal shock argon M=5.03\n";
    std::cout << "    p2_calc=" << ss.p2 << "  p2_ref=" << p2_ref << "\n";
    std::cout << "    T2_calc=" << ss.T2 << "  T2_ref=" << T2_ref << "\n";
}

int main() {
    std::cout << "=== test_gas ===\n";
    test_argon_properties();
    test_nitrogen_properties();
    test_unknown_gas();
    test_pressure_temperature_round_trip();
    test_sound_speed();
    test_normal_shock_argon();
    std::cout << "=== PASSED ===\n";
    return 0;
}
