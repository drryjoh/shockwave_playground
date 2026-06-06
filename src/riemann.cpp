#include "splay/riemann.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace splay {

// Physical inviscid flux for a single state.
Flux3 inviscid_flux(double rho, double u, double p, double E) {
    return {rho * u,
            rho * u * u + p,
            u * (rho * E + p)};
}

// ─── Central flux ─────────────────────────────────────────────────────────────
Flux3 flux_central(double rho_L, double u_L, double p_L, double E_L,
                   double rho_R, double u_R, double p_R, double E_R) {
    auto FL = inviscid_flux(rho_L, u_L, p_L, E_L);
    auto FR = inviscid_flux(rho_R, u_R, p_R, E_R);
    return {0.5 * (FL[0] + FR[0]),
            0.5 * (FL[1] + FR[1]),
            0.5 * (FL[2] + FR[2])};
}

// ─── Rusanov (local Lax-Friedrichs) ──────────────────────────────────────────
Flux3 flux_rusanov(double rho_L, double u_L, double p_L, double E_L,
                   double rho_R, double u_R, double p_R, double E_R,
                   double gamma) {
    auto FL = inviscid_flux(rho_L, u_L, p_L, E_L);
    auto FR = inviscid_flux(rho_R, u_R, p_R, E_R);

    const double a_L  = std::sqrt(gamma * p_L / rho_L);
    const double a_R  = std::sqrt(gamma * p_R / rho_R);
    const double s_max = std::max(std::abs(u_L) + a_L, std::abs(u_R) + a_R);

    return {0.5 * (FL[0] + FR[0]) - 0.5 * s_max * (rho_R               - rho_L),
            0.5 * (FL[1] + FR[1]) - 0.5 * s_max * (rho_R * u_R         - rho_L * u_L),
            0.5 * (FL[2] + FR[2]) - 0.5 * s_max * (rho_R * E_R         - rho_L * E_L)};
}

// ─── HLLC ────────────────────────────────────────────────────────────────────
// Reference: Toro, "Riemann Solvers and Numerical Methods", 3rd ed., §10.4.
Flux3 flux_hllc(double rho_L, double u_L, double p_L, double E_L,
                double rho_R, double u_R, double p_R, double E_R,
                double gamma) {
    const double a_L = std::sqrt(gamma * p_L / rho_L);
    const double a_R = std::sqrt(gamma * p_R / rho_R);

    // Pressure estimate (Roe-average or two-rarefaction; use arithmetic here)
    const double p_pvrs = 0.5 * (p_L + p_R)
                        - 0.5 * (u_R - u_L) * 0.5 * (rho_L + rho_R) * 0.5 * (a_L + a_R);
    const double p_star = std::max(0.0, p_pvrs);

    // Wave speeds using pressure-based estimate (Toro eq. 10.58-10.60)
    auto q_K = [&](double p_K, double a_K) {
        if (p_star <= p_K) return 1.0;
        return std::sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / p_K - 1.0));
    };

    const double S_L = u_L - a_L * q_K(p_L, a_L);
    const double S_R = u_R + a_R * q_K(p_R, a_R);

    // Contact wave speed
    const double S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                        / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));

    // Physical fluxes
    auto FL = inviscid_flux(rho_L, u_L, p_L, E_L);
    auto FR = inviscid_flux(rho_R, u_R, p_R, E_R);

    if (S_L >= 0.0) return FL;
    if (S_R <= 0.0) return FR;

    // HLLC intermediate states (Toro eq. 10.73)
    auto hllc_star = [&](double rho_K, double u_K, double p_K, double E_K,
                         double S_K, const Flux3& FK) -> Flux3 {
        const double coeff  = rho_K * (S_K - u_K) / (S_K - S_star);
        const double E_star = E_K + (S_star - u_K) * (S_star + p_K / (rho_K * (S_K - u_K)));
        // U_star = coeff * [1, S_star, E_star]
        // F_star = FK + S_K * (U_star - UK)
        const double drho    = coeff         - rho_K;
        const double drhou   = coeff * S_star - rho_K * u_K;
        const double drhoE   = coeff * E_star - rho_K * E_K;
        return {FK[0] + S_K * drho,
                FK[1] + S_K * drhou,
                FK[2] + S_K * drhoE};
    };

    if (S_star >= 0.0) {
        return hllc_star(rho_L, u_L, p_L, E_L, S_L, FL);
    } else {
        return hllc_star(rho_R, u_R, p_R, E_R, S_R, FR);
    }
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────
Flux3 compute_inviscid_flux(
    double rho_L, double u_L, double p_L, double E_L,
    double rho_R, double u_R, double p_R, double E_R,
    double gamma,
    RiemannSolver solver)
{
    switch (solver) {
        case RiemannSolver::Central: return flux_central(rho_L, u_L, p_L, E_L,
                                                         rho_R, u_R, p_R, E_R);
        case RiemannSolver::Rusanov: return flux_rusanov(rho_L, u_L, p_L, E_L,
                                                         rho_R, u_R, p_R, E_R, gamma);
        case RiemannSolver::HLLC:    return flux_hllc(rho_L, u_L, p_L, E_L,
                                                      rho_R, u_R, p_R, E_R, gamma);
        default:
            throw std::runtime_error("compute_inviscid_flux: unknown solver.");
    }
}

} // namespace splay
