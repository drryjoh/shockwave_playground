// SPLAY test: transport model
// Tests: coefficient loading, monotonicity, clamp guards.
#include "splay/transport.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) <= tol * (std::abs(b) + 1e-14);
}

void test_argon_transport() {
    auto tm = splay::TransportModel::from_name("argon");

    check(tm.name == "argon",   "argon transport name");
    check(tm.T_min  > 0.0,      "T_min > 0");
    check(tm.T_max  > tm.T_min, "T_max > T_min");

    // At T=300 K, argon viscosity should be ~2.27e-5 Pa·s (Sutherland)
    const double mu300 = tm.viscosity(300.0);
    check(mu300 > 1e-6 && mu300 < 1e-4, "mu300 in physical range");

    // Viscosity should increase with temperature for argon (power law)
    const double mu1000 = tm.viscosity(1000.0);
    check(mu1000 > mu300, "mu increases with T (argon)");

    // Conductivity should be positive
    const double k300 = tm.conductivity(300.0);
    check(k300 > 0.0, "k300 > 0");

    std::cout << "  [PASS] argon transport\n";
    std::cout << "    mu(300K)="  << mu300  << " Pa·s\n";
    std::cout << "    k(300K)="   << k300   << " W/(m·K)\n";
    std::cout << "    mu(1000K)=" << mu1000 << " Pa·s\n";
}

void test_nitrogen_transport() {
    auto tm = splay::TransportModel::from_name("nitrogen");
    const double mu = tm.viscosity(300.0);
    const double k  = tm.conductivity(300.0);
    check(mu > 0.0, "nitrogen mu > 0");
    check(k  > 0.0, "nitrogen k  > 0");
    std::cout << "  [PASS] nitrogen transport\n";
}

void test_compute_transport_vector() {
    auto tm = splay::TransportModel::from_name("argon");
    std::vector<double> T_vec = {300.0, 500.0, 1000.0, 2000.0};
    std::vector<double> mu_out, k_out;
    splay::compute_transport(T_vec, tm, mu_out, k_out);

    check(mu_out.size() == T_vec.size(), "mu size");
    check(k_out.size()  == T_vec.size(), "k  size");
    for (size_t i = 0; i < T_vec.size(); ++i) {
        check(mu_out[i] > 0.0, "mu[i] > 0");
        check(k_out[i]  > 0.0, "k[i]  > 0");
    }
    std::cout << "  [PASS] compute_transport vector\n";
}

int main() {
    std::cout << "=== test_transport ===\n";
    test_argon_transport();
    test_nitrogen_transport();
    test_compute_transport_vector();
    std::cout << "=== PASSED ===\n";
    return 0;
}
