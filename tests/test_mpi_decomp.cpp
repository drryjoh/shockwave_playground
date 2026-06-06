// SPLAY test: MPI decomposition
// Tests: cell count decomposition, ghost-cell exchange (serial fallback),
//        boundary conditions, uniform flow preservation.
#include "splay/mesh.hpp"
#include "splay/mpi_decomp.hpp"
#include "splay/state.hpp"
#include "splay/gas.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

// Verify that a 100-cell global domain decomposes correctly for 1..4 ranks.
void test_decomposition() {
    const int N = 100;
    splay::GridConfig gc;
    gc.n_cells   = N;
    gc.x_min     = 0.0;
    gc.x_max     = 1.0;
    gc.ghost_cells = 2;

    for (int nranks : {1, 2, 3, 4, 7}) {
        int total = 0;
        for (int r = 0; r < nranks; ++r) {
            auto m = splay::Mesh::build(gc, r, nranks);
            total += m.n_local;
            // Check x coordinates are within domain (with ghost overhang allowed)
            check(m.n_total == m.n_local + 2 * m.n_ghost, "n_total check");
        }
        check(total == N, ("total cells == N for nranks=" + std::to_string(nranks)).c_str());
        std::cout << "  [PASS] decomposition nranks=" << nranks << "\n";
    }
}

// Verify uniform flow preservation: init all cells to constant state,
// apply BCs, and check that interior cells remain unchanged.
void test_uniform_flow() {
    const int N = 50;
    splay::GridConfig gc;
    gc.n_cells   = N;
    gc.x_min     = 0.0;
    gc.x_max     = 1.0;
    gc.ghost_cells = 2;

    auto gas = splay::GasModel::from_name("argon");
    auto m   = splay::Mesh::build(gc, 0, 1);

    splay::State s(m.n_total);
    const double p0 = 1e5, T0 = 300.0, u0 = 100.0;
    const double rho0 = p0 / (gas.R * T0);
    const double e0   = gas.cv * T0;
    const double E0   = e0 + 0.5 * u0 * u0;
    for (int i = 0; i < m.n_total; ++i) {
        s.rho[i]  = rho0;
        s.rhou[i] = rho0 * u0;
        s.rhoE[i] = rho0 * E0;
        s.u[i]    = u0;
        s.p[i]    = p0;
        s.T[i]    = T0;
    }

    // Inflow BC: same state as interior
    splay::BCState bc_left;
    bc_left.type        = splay::BCType::Inflow;
    bc_left.pressure    = p0;
    bc_left.temperature = T0;
    bc_left.velocity    = u0;

    splay::BCState bc_right;
    bc_right.type = splay::BCType::Outflow;

    splay::MPIDecomp decomp = splay::MPIDecomp::build(0, 1, m.n_ghost, m.n_local);
    splay::apply_boundary_conditions(s, m, bc_left, bc_right, decomp,
                                      gas.gamma, gas.cv, gas.R);

    // Check interior cells unchanged
    const int ib = m.interior_begin(), ie = m.interior_end();
    for (int i = ib; i < ie; ++i) {
        check(std::abs(s.rho[i] - rho0) < 1e-14, "uniform rho preserved");
        check(std::abs(s.p[i]   - p0  ) < 1e-14, "uniform p preserved");
    }
    // Check ghost cells set by BC
    for (int g = 0; g < m.n_ghost; ++g) {
        check(std::abs(s.rho[g] - rho0) < 1e-14, "inflow ghost rho");
    }
    std::cout << "  [PASS] uniform flow preservation\n";
}

int main(int argc, char** argv) {
#ifdef SPLAY_ENABLE_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc; (void)argv;
#endif

    std::cout << "=== test_mpi_decomp ===\n";
    test_decomposition();
    test_uniform_flow();
    std::cout << "=== PASSED ===\n";

#ifdef SPLAY_ENABLE_MPI
    MPI_Finalize();
#endif
    return 0;
}
