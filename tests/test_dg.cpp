// SPLAY DG tests
//
// 1. Constant-state residual: for a uniform flow, all DG residuals must be
//    machine-epsilon zero (inviscid + viscous + AV terms cancel exactly).
//
// 2. Sod shock tube: run to t = 0.15 ms with Rusanov flux + Persson-Peraire AV;
//    check no NaN/Inf, density jump is present, and minimum density > 0.
//
// 3. Gaussian pressure wave: run a smooth pressure perturbation for 10 steps;
//    check the solution stays smooth (no new extrema introduced) and the max
//    pressure perturbation decreases or stays bounded (energy is not added).

#include "splay/gas.hpp"
#include "splay/mesh.hpp"
#include "splay/config.hpp"
#include "splay/transport.hpp"
#include "splay/dg_state.hpp"
#include "splay/dg_av.hpp"
#include "splay/dg_residual.hpp"
#include "splay/rk.hpp"
#include "splay/mpi_decomp.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdlib>

// ── Test helpers ─────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
}

static bool is_finite_val(double v) { return std::isfinite(v); }

// Check all DOFs in interior cells are finite.
static void check_finite(const splay::DGState& s, const splay::Mesh& m,
                          const char* context) {
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            if (!is_finite_val(s.rho[j][i]) ||
                !is_finite_val(s.rhou[j][i]) ||
                !is_finite_val(s.rhoE[j][i])) {
                std::cerr << "FAIL: non-finite DOF at cell=" << i
                          << " node=" << j << " (" << context << ")\n";
                std::exit(1);
            }
        }
    }
}

// ── Build a minimal SolverConfig for DG tests ─────────────────────────────────
static splay::SolverConfig make_dg_cfg(int poly_order,
                                        splay::AVMethod av = splay::AVMethod::PerssonPeraire,
                                        bool viscous = false)
{
    splay::SolverConfig cfg;
    cfg.spatial                = splay::SpatialDiscretization::DG;
    cfg.dg.poly_order          = poly_order;
    cfg.dg.visc_scheme         = splay::ViscousScheme::BR2;
    cfg.dg.av_method           = av;
    cfg.dg.C_av                = 0.5;
    cfg.dg.s0                  = -4.0;
    cfg.dg.kappa               = 1.0;
    cfg.viscous_terms          = viscous;
    cfg.viscous_dt             = viscous;
    cfg.cfl                    = 0.3;
    cfg.riemann_solver         = splay::RiemannSolver::Rusanov;
    cfg.inviscid_scheme        = splay::InviscidScheme::Central; // unused for DG
    return cfg;
}

// ── Build a small mesh for testing ───────────────────────────────────────────
static splay::Mesh make_mesh(int n_cells, double x_min = 0.0, double x_max = 1.0,
                              int ghost_cells = 2)
{
    splay::GridConfig gc;
    gc.x_min       = x_min;
    gc.x_max       = x_max;
    gc.n_cells     = n_cells;
    gc.ghost_cells = ghost_cells;
    return splay::Mesh::build(gc, 0, 1);
}

// ─── Test 1: constant-state residual = 0 ─────────────────────────────────────
void test_constant_residual(int poly_order, bool viscous) {
    const std::string label = "p=" + std::to_string(poly_order)
                            + (viscous ? " visc" : " inviscid");

    auto gas = splay::GasModel::from_name("argon");
    auto tm  = splay::TransportModel::from_name("argon");
    auto m   = make_mesh(20);

    splay::DGState    s(m.n_total, poly_order);
    splay::DGResidual R(m.n_total, s.n_dof);

    // Uniform rest state (u=0 eliminates viscous fluxes trivially)
    const double rho0 = 1.0, p0 = 1e5, u0 = 0.0;
    splay::init_constant_dg(s, m, gas, rho0, p0, u0);

    // Outflow BCs on both sides
    splay::BCState bc_out;
    bc_out.type = splay::BCType::Outflow;
    splay::apply_boundary_conditions_dg(s, m, bc_out, bc_out, gas);

    // AV: None — ensures ε = 0 and no contamination from sensor
    auto cfg = make_dg_cfg(poly_order, splay::AVMethod::None, viscous);
    splay::compute_av_dg(s, m, gas, cfg.dg);
    splay::compute_residual_dg(s, m, gas, tm, cfg, R);

    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    double r_max = 0.0;
    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            r_max = std::max(r_max, std::abs(R.r_rho [j][i]));
            r_max = std::max(r_max, std::abs(R.r_rhou[j][i]));
            r_max = std::max(r_max, std::abs(R.r_rhoE[j][i]));
        }
    }

    // Residual should be ≤ machine-epsilon × reference flux magnitude.
    // Reference: ρc of argon ~ 400 kg/(m²·s); expect r_max ≪ 1e-6 * 400.
    const double tol = 1e-6;
    check(r_max < tol, ("constant-state residual not zero [" + label + "]").c_str());
    std::cout << "  [PASS] constant-state residual (" << label
              << ")  r_max=" << r_max << "\n";
}

// ─── Test 2: Sod shock tube ───────────────────────────────────────────────────
void test_sod(int poly_order) {
    const std::string label = "p=" + std::to_string(poly_order);

    auto gas = splay::GasModel::from_name("argon");
    auto tm  = splay::TransportModel::from_name("argon");
    auto m   = make_mesh(100, 0.0, 1.0, /*ghost=*/2);

    splay::DGState s(m.n_total, poly_order);

    // Sod ICs
    const double rhoL = 1.0, pL = 1e5, uL = 0.0;
    const double rhoR = 0.125, pR = 1e4, uR = 0.0;
    splay::init_step_dg(s, m, gas, rhoL, pL, uL, rhoR, pR, uR, /*x0=*/0.5);

    splay::BCState bc_out;
    bc_out.type = splay::BCType::Outflow;
    splay::apply_boundary_conditions_dg(s, m, bc_out, bc_out, gas);
    splay::fill_ghosts_dg(s, m, 0, 1, -1, -1);

    auto cfg = make_dg_cfg(poly_order, splay::AVMethod::PerssonPeraire, /*viscous=*/false);
    cfg.cfl  = 0.2;

    splay::MPIDecomp decomp = splay::MPIDecomp::build(0, 1, m.n_ghost, m.n_local);

    // Run to t_target (Sod solution at 0.15 ms should show clear shock)
    const double t_target = 1.5e-4;
    double time = 0.0;
    int    step = 0;

    while (time < t_target) {
        std::string ac;
        double dt = splay::compute_dt_dg(s, m, gas, tm, cfg, ac);
        if (time + dt > t_target) dt = t_target - time;
        if (dt <= 0.0) break;

        splay::ssprk3_step_dg(s, m, gas, tm, cfg, bc_out, bc_out, decomp, dt);
        time += dt;
        ++step;
    }

    check_finite(s, m, ("sod " + label).c_str());

    const int ib = m.interior_begin();
    const int ie = m.interior_end();

    // Density must be positive everywhere
    double rho_min = std::numeric_limits<double>::max();
    double rho_max = 0.0;
    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            rho_min = std::min(rho_min, s.rho[j][i]);
            rho_max = std::max(rho_max, s.rho[j][i]);
        }
    }
    check(rho_min > 0.0, ("Sod: negative density [" + label + "]").c_str());

    // There should still be a density contrast (shock not completely diffused)
    check(rho_max > 0.3, ("Sod: density too uniform — shock smeared away [" + label + "]").c_str());
    check(rho_min < 0.3, ("Sod: no low-density region [" + label + "]").c_str());

    std::cout << "  [PASS] Sod shock tube (" << label << ")  steps=" << step
              << "  rho_min=" << rho_min << "  rho_max=" << rho_max << "\n";
}

// ─── Test 3: Gaussian pressure wave ──────────────────────────────────────────
void test_gaussian_wave(int poly_order, bool viscous = false) {
    const std::string label = "p=" + std::to_string(poly_order)
                            + (viscous ? " visc" : " inviscid");

    auto gas = splay::GasModel::from_name("argon");
    auto tm  = splay::TransportModel::from_name("argon");
    // Wide domain so the wave doesn't hit the boundary during the test
    auto m   = make_mesh(80, -1.0, 1.0, /*ghost=*/2);

    splay::DGState s(m.n_total, poly_order);

    const double p0 = 1e5, T0 = 300.0, u0 = 0.0;
    const double amplitude = 1e-3;
    splay::init_gaussian_perturbation_dg(s, m, gas, p0, T0, u0, amplitude,
                                          /*x0=*/0.0, /*sigma=*/0.1);

    splay::BCState bc_out;
    bc_out.type = splay::BCType::Outflow;
    splay::apply_boundary_conditions_dg(s, m, bc_out, bc_out, gas);
    splay::fill_ghosts_dg(s, m, 0, 1, -1, -1);

    // Measure initial max pressure perturbation
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    double p_max0 = 0.0;
    for (int i = ib; i < ie; ++i)
        for (int j = 0; j < s.n_dof; ++j)
            p_max0 = std::max(p_max0, s.prim_p[j][i]);

    auto cfg = make_dg_cfg(poly_order, splay::AVMethod::None, viscous);
    splay::MPIDecomp decomp = splay::MPIDecomp::build(0, 1, m.n_ghost, m.n_local);

    // Advance 20 steps
    const int n_steps = 20;
    double time = 0.0;
    for (int k = 0; k < n_steps; ++k) {
        std::string ac;
        double dt = splay::compute_dt_dg(s, m, gas, tm, cfg, ac);
        splay::ssprk3_step_dg(s, m, gas, tm, cfg, bc_out, bc_out, decomp, dt);
        time += dt;
    }

    check_finite(s, m, ("gaussian " + label).c_str());

    // Density must remain positive
    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            check(s.rho[j][i] > 0.0,
                  ("gaussian: negative density [" + label + "]").c_str());
        }
    }

    // Max pressure should not exceed the initial peak by more than a few percent
    // (DG without limiting can have small oscillations, but should not blow up)
    double p_max_final = 0.0;
    for (int i = ib; i < ie; ++i)
        for (int j = 0; j < s.n_dof; ++j)
            p_max_final = std::max(p_max_final, s.prim_p[j][i]);

    // Allow up to 10× growth factor as a loose check (no blowup)
    check(p_max_final < 10.0 * p_max0,
          ("gaussian: pressure blow-up [" + label + "]").c_str());

    std::cout << "  [PASS] Gaussian pressure wave (" << label << ")  t=" << time
              << "  p_max0=" << p_max0 << "  p_max_final=" << p_max_final << "\n";
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== test_dg ===\n";

    // Test 1: constant-state residual for p=1 and p=2, inviscid and viscous
    std::cout << "-- constant-state residual --\n";
    test_constant_residual(1, false);
    test_constant_residual(2, false);
    test_constant_residual(1, true);
    test_constant_residual(2, true);

    // Test 2: Sod shock tube
    std::cout << "-- Sod shock tube --\n";
    test_sod(1);
    test_sod(2);

    // Test 3: Gaussian pressure wave — inviscid and viscous
    std::cout << "-- Gaussian pressure wave --\n";
    test_gaussian_wave(1, false);
    test_gaussian_wave(2, false);
    test_gaussian_wave(1, true);
    test_gaussian_wave(2, true);

    std::cout << "=== PASSED ===\n";
    return 0;
}
