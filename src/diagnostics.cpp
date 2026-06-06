#include "splay/diagnostics.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace splay {

static const char* scheme_name(InviscidScheme s) {
    switch (s) {
        case InviscidScheme::Central: return "central";
        case InviscidScheme::MUSCL:   return "muscl";
        case InviscidScheme::PPM:     return "ppm";
        default: return "unknown";
    }
}

static const char* riemann_name(RiemannSolver r) {
    switch (r) {
        case RiemannSolver::Central: return "central/none";
        case RiemannSolver::Rusanov: return "rusanov";
        case RiemannSolver::HLLC:    return "hllc";
        default: return "unknown";
    }
}

static const char* limiter_name(Limiter l) {
    switch (l) {
        case Limiter::None:    return "none";
        case Limiter::Minmod:  return "minmod";
        case Limiter::VanLeer: return "vanleer";
        case Limiter::MC:      return "mc";
        case Limiter::PPM:     return "ppm";
        default: return "unknown";
    }
}

void print_banner(const Config& cfg, int nranks) {
    if (/* rank */ true) {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  SPLAY v1.0.0  –  Shock Playground 1D FV Solver             ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << std::left << std::setw(30) << "  Case name"
                  << ": " << cfg.case_name << "\n";
        std::cout << std::setw(30) << "  Gas"
                  << ": " << cfg.gas << "\n";
        std::cout << std::setw(30) << "  MPI ranks"
                  << ": " << nranks << "\n";
        std::cout << std::setw(30) << "  Grid cells (global)"
                  << ": " << cfg.grid.n_cells << "\n";
        std::cout << std::setw(30) << "  Ghost cells"
                  << ": " << cfg.grid.ghost_cells << "\n";
        std::cout << std::setw(30) << "  Domain [m]"
                  << ": [" << cfg.grid.x_min << ", " << cfg.grid.x_max << "]\n";
        std::cout << std::setw(30) << "  Inviscid scheme"
                  << ": " << scheme_name(cfg.solver.inviscid_scheme) << "\n";
        std::cout << std::setw(30) << "  Riemann solver"
                  << ": " << riemann_name(cfg.solver.riemann_solver) << "\n";
        std::cout << std::setw(30) << "  Limiter"
                  << ": " << limiter_name(cfg.solver.limiter) << "\n";
        std::cout << std::setw(30) << "  Viscous terms"
                  << ": " << (cfg.solver.viscous_terms ? "YES (Navier-Stokes)" : "NO (Euler)") << "\n";
        std::cout << std::setw(30) << "  CFL"
                  << ": " << cfg.solver.cfl << "\n";
        std::cout << std::setw(30) << "  Final time [s]"
                  << ": " << cfg.solver.time_end << "\n";
        std::cout << std::setw(30) << "  Time integrator"
                  << ": SSPRK3\n";
        std::cout << "\n";
    }
}

void print_step_diag(int step, double time, double dt,
                     const std::string& active_constraint,
                     const State& s, const Mesh& m,
                     const GasModel& gas, const MPIDecomp& decomp,
                     const std::vector<double>& r_rho,
                     const std::vector<double>& r_rhou,
                     const std::vector<double>& r_rhoE)
{
    if (decomp.rank != 0) return;

    const int ib = m.interior_begin();
    const int ie = m.interior_end();

    double rho_min  =  1e300, rho_max  = -1e300;
    double p_min    =  1e300, p_max    = -1e300;
    double T_min    =  1e300, T_max    = -1e300;
    double sum_rho = 0.0;
    int count = 0;

    for (int i = ib; i < ie; ++i) {
        rho_min = std::min(rho_min, s.rho[i]);
        rho_max = std::max(rho_max, s.rho[i]);
        p_min   = std::min(p_min,   s.p[i]);
        p_max   = std::max(p_max,   s.p[i]);
        T_min   = std::min(T_min,   s.T[i]);
        T_max   = std::max(T_max,   s.T[i]);
        sum_rho += r_rho[i] * r_rho[i];
        ++count;
    }
    const double res_norm = (count > 0) ? std::sqrt(sum_rho / count) : 0.0;

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  step=" << std::setw(7) << step
              << "  t="    << time
              << "  dt="   << dt
              << "  [" << active_constraint << "]"
              << "  |R_rho|=" << res_norm
              << "  rho=[" << rho_min << "," << rho_max << "]"
              << "  p=["   << p_min   << "," << p_max   << "]"
              << "  T=["   << T_min   << "," << T_max   << "]"
              << "\n";
}

} // namespace splay
