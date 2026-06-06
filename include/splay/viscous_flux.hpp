#pragma once
#include <array>
#include <vector>
#include "splay/riemann.hpp"

namespace splay {

/// Compute viscous flux at face i+1/2 using central differences.
///
/// Convention:  F_visc = [0,  tau_xx,  u*tau_xx - q_x]^T
///
/// For 1D Newtonian / Stokes hypothesis:
///   tau_xx = (4/3) * mu * du/dx
///   q_x    = -k * dT/dx
///   F_visc_energy = u * tau_xx + k * dT/dx
///
/// du/dx and dT/dx are approximated as centred differences between cell centres
/// at i and i+1.
///
/// @param mu_L   viscosity at left cell (Pa·s)
/// @param mu_R   viscosity at right cell
/// @param k_L    conductivity at left cell (W/(m·K))
/// @param k_R    conductivity at right cell
/// @param u_L    velocity at left cell (m/s)
/// @param u_R    velocity at right cell
/// @param T_L    temperature at left cell (K)
/// @param T_R    temperature at right cell
/// @param dx     cell spacing (m)
/// @returns [0, tau_xx, u_face*tau_xx + k_face*dT/dx]
Flux3 compute_viscous_flux(
    double mu_L, double mu_R,
    double k_L,  double k_R,
    double u_L,  double u_R,
    double T_L,  double T_R,
    double dx);

/// Compute viscous fluxes for all interior faces.
void compute_all_viscous_fluxes(
    const std::vector<double>& mu,
    const std::vector<double>& kappa,
    const std::vector<double>& u,
    const std::vector<double>& T,
    double dx,
    int face_begin, int face_end,
    std::vector<Flux3>& visc_flux);

} // namespace splay
