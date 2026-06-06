#include "splay/viscous_flux.hpp"
#include <cmath>

namespace splay {

// Face i+1/2 is between cell i and cell i+1.
// du/dx ≈ (u_{i+1} - u_i) / dx
// dT/dx ≈ (T_{i+1} - T_i) / dx
// mu_face = 0.5*(mu_i + mu_{i+1})
// k_face  = 0.5*(k_i  + k_{i+1})
// u_face  = 0.5*(u_i  + u_{i+1})
//
// tau_xx = (4/3)*mu*du/dx
// F_visc = [0, tau_xx, u_face*tau_xx + k_face*dT/dx]

Flux3 compute_viscous_flux(
    double mu_L, double mu_R,
    double k_L,  double k_R,
    double u_L,  double u_R,
    double T_L,  double T_R,
    double dx)
{
    const double mu_face = 0.5 * (mu_L + mu_R);
    const double k_face  = 0.5 * (k_L  + k_R);
    const double u_face  = 0.5 * (u_L  + u_R);

    const double dudx  = (u_R - u_L) / dx;
    const double dTdx  = (T_R - T_L) / dx;

    const double tau_xx   = (4.0 / 3.0) * mu_face * dudx;
    const double q_x      = -k_face * dTdx;  // heat flux (positive in +x direction)

    // F_visc_energy = u*tau_xx - q_x
    const double F_energy = u_face * tau_xx - q_x;

    return {0.0, tau_xx, F_energy};
}

void compute_all_viscous_fluxes(
    const std::vector<double>& mu,
    const std::vector<double>& kappa,
    const std::vector<double>& u,
    const std::vector<double>& T,
    double dx,
    int face_begin, int face_end,
    std::vector<Flux3>& visc_flux)
{
    for (int f = face_begin; f < face_end; ++f) {
        visc_flux[f] = compute_viscous_flux(
            mu[f], mu[f+1], kappa[f], kappa[f+1],
            u[f],  u[f+1],  T[f],    T[f+1],
            dx);
    }
}

} // namespace splay
