#pragma once
#include <array>
#include "splay/config.hpp"

namespace splay {

using Flux3 = std::array<double, 3>;

/// Compute the inviscid flux at a single face given left/right primitive states.
///
/// @param rho_L  left density  (kg/m³)
/// @param u_L    left velocity (m/s)
/// @param p_L    left pressure (Pa)
/// @param rho_R  right density
/// @param u_R    right velocity
/// @param p_R    right pressure
/// @param gamma  ratio of specific heats
/// @param solver selected Riemann solver
/// @returns [F_rho, F_rhou, F_rhoE]
Flux3 compute_inviscid_flux(
    double rho_L, double u_L, double p_L, double E_L,
    double rho_R, double u_R, double p_R, double E_R,
    double gamma,
    RiemannSolver solver);

/// Central (average) flux.
Flux3 flux_central(double rho_L, double u_L, double p_L, double E_L,
                   double rho_R, double u_R, double p_R, double E_R);

/// Rusanov (local Lax-Friedrichs) flux.
Flux3 flux_rusanov(double rho_L, double u_L, double p_L, double E_L,
                   double rho_R, double u_R, double p_R, double E_R,
                   double gamma);

/// HLLC flux.
Flux3 flux_hllc(double rho_L, double u_L, double p_L, double E_L,
                double rho_R, double u_R, double p_R, double E_R,
                double gamma);

/// Physical inviscid flux for a single state.
Flux3 inviscid_flux(double rho, double u, double p, double E);

} // namespace splay
