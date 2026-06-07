#pragma once
#include <vector>
#include <string>
#include "splay/mesh.hpp"
#include "splay/gas.hpp"

namespace splay {

/// Conservative and primitive state arrays for the 1D solver.
///
/// Arrays are of length mesh.n_total (interior + ghost on both sides).
/// Index 0 is the first ghost cell on the left.
struct State {
    // Conservative variables
    std::vector<double> rho;   // kg/m³
    std::vector<double> rhou;  // kg/(m²·s)
    std::vector<double> rhoE;  // J/m³

    // Primitive variables (derived; kept in sync)
    std::vector<double> u;     // m/s
    std::vector<double> p;     // Pa
    std::vector<double> T;     // K

    int size() const { return static_cast<int>(rho.size()); }

    /// Allocate arrays for n_total cells.
    explicit State(int n_total);

    /// Recompute primitive variables from conservative (all cells).
    void update_primitives(const GasModel& gas);

    /// Recompute primitive variables for a single cell index.
    void update_primitives_cell(int i, const GasModel& gas);

    /// Return L2 norm of a residual vector (same size as state).
    static double l2_norm(const std::vector<double>& r);
};

/// Initialize state from analytic tanh profile.
void init_tanh(State& s, const Mesh& m, const GasModel& gas,
               double p_L, double T_L, double u_L,
               double p_R, double T_R, double u_R,
               double x_shock, double delta);

/// Initialize state as a step function (Sod / Riemann problem ICs).
/// Cells with x <= x_diaphragm get the left state; cells to the right get the right state.
void init_step(State& s, const Mesh& m, const GasModel& gas,
               double rho_L, double p_L, double u_L,
               double rho_R, double p_R, double u_R,
               double x_diaphragm);

/// Initialize state as uniform base + small Gaussian pressure perturbation.
/// p(x) = p0 * (1 + amplitude * exp(-((x-x0)/sigma)^2))
/// Density is held constant at p0/(R*T0); temperature varies with pressure.
void init_gaussian_perturbation(State& s, const Mesh& m, const GasModel& gas,
                                 double p0, double T0, double u0,
                                 double amplitude, double x0, double sigma);

} // namespace splay
