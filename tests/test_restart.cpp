// SPLAY test: restart round-trip
// Writes a state to a restart file and reads it back; checks consistency.
#include "splay/config.hpp"
#include "splay/gas.hpp"
#include "splay/mesh.hpp"
#include "splay/state.hpp"
#include "splay/mpi_decomp.hpp"
#include "splay/io.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol * (std::abs(b) + 1.0);
}

int main() {
    std::cout << "=== test_restart ===\n";

    // Build a minimal config + mesh.
    splay::GridConfig gc;
    gc.n_cells    = 20;
    gc.x_min      = 0.0;
    gc.x_max      = 1.0;
    gc.ghost_cells = 2;

    auto gas    = splay::GasModel::from_name("argon");
    auto m      = splay::Mesh::build(gc, 0, 1);
    auto decomp = splay::MPIDecomp::build(0, 1, m.n_ghost, m.n_local);

    // Initialise a recognisable non-trivial state.
    splay::State s(m.n_total);
    const int ib = m.interior_begin(), ie = m.interior_end();
    for (int i = ib; i < ie; ++i) {
        const double x  = m.x_cell[i];
        s.p[i]   = 1e5 + x * 1e5;
        s.T[i]   = 300.0 + x * 100.0;
        s.u[i]   = 50.0 * x;
        s.rho[i] = s.p[i] / (gas.R * s.T[i]);
        const double E = gas.cv * s.T[i] + 0.5 * s.u[i] * s.u[i];
        s.rhou[i] = s.rho[i] * s.u[i];
        s.rhoE[i] = s.rho[i] * E;
    }

    // Create a minimal Config for the restart writer.
    splay::Config cfg;
    cfg.gas         = "argon";
    cfg.case_name   = "test_restart_case";
    cfg.grid        = gc;

    const std::string restart_dir = "/tmp/splay_test_restart";
    const int  step_out = 42;
    const double time_out = 3.14159;

    splay::write_restart(s, m, cfg, decomp, restart_dir, step_out, time_out);
    std::cout << "  Wrote restart to: " << restart_dir << "\n";

    // Read back.
    splay::State s2(m.n_total);
    double t_in;
    int    step_in;
    splay::read_restart(s2, m, decomp, restart_dir, t_in, step_in);

    check(step_in == step_out, "restart step round-trip");
    check(near(t_in, time_out, 1e-12), "restart time round-trip");

    for (int i = ib; i < ie; ++i) {
        check(near(s2.rho[i],  s.rho[i]),  "restart rho round-trip");
        check(near(s2.rhou[i], s.rhou[i]), "restart rhou round-trip");
        check(near(s2.rhoE[i], s.rhoE[i]), "restart rhoE round-trip");
    }

    std::cout << "  [PASS] restart round-trip\n";
    std::cout << "=== PASSED ===\n";

    // Cleanup
    std::filesystem::remove_all(restart_dir);
    return 0;
}
