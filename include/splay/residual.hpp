#pragma once
#include <vector>
#include "splay/config.hpp"
#include "splay/gas.hpp"
#include "splay/mesh.hpp"
#include "splay/state.hpp"
#include "splay/transport.hpp"

namespace splay {

/// Residual arrays for the three conservative equations.
/// dU/dt = -R(U)   (negative sign: residual = -dF/dx + viscous)
/// Actually we store R = dU/dt so that U^{n+1} = U^n + dt * R.
struct Residual {
    std::vector<double> r_rho;   // d(rho)/dt contribution
    std::vector<double> r_rhou;  // d(rhou)/dt contribution
    std::vector<double> r_rhoE;  // d(rhoE)/dt contribution

    explicit Residual(int n_total);

    void zero();
};

/// Selects which flux contributions to include in compute_residual().
enum class ResidualPart {
    Full,         ///< inviscid + viscous  (normal operation)
    InviscidOnly, ///< inviscid fluxes only (used by Godunov splitting, inviscid sub-step)
    ViscousOnly,  ///< viscous  fluxes only (used by Godunov splitting, viscous  sub-step)
};

/// Compute R(U): the right-hand side of dU/dt = R(U).
///
/// Steps:
///   1. Reconstruct face states (inviscid scheme + limiter).
///   2. Compute inviscid fluxes at each face (Riemann solver).
///   3. Optionally compute viscous fluxes.
///   4. Subtract divergence: R[i] = -(F[i+1/2] - F[i-1/2]) / dx + viscous divergence.
///
/// Only interior cells (mesh.interior_begin..interior_end) are filled.
void compute_residual(
    const State&          s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    Residual&             R,
    ResidualPart          part = ResidualPart::Full,
    double                dt   = 0.0);

/// Compute stable time-step satisfying CFL (convective + diffusive when viscous).
/// Returns global min dt across all local cells.
double compute_dt(
    const State&          s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    std::string&          active_constraint);

} // namespace splay
