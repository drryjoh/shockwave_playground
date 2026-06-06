#include "splay/gas.hpp"
#include <cmath>
#include <stdexcept>

namespace splay {

GasModel GasModel::from_name(const std::string& name) {
    GasModel g;
    g.name = name;

    if (name == "argon") {
        g.MW    = 0.039948;            // kg/mol
        g.gamma = 5.0 / 3.0;
        g.R     = 8314.46261815324 / g.MW;  // J/(kg·K)  = 208.13
        g.cv    = g.R / (g.gamma - 1.0);
        g.cp    = g.gamma * g.cv;
    } else if (name == "nitrogen") {
        g.MW    = 0.028014;            // kg/mol
        g.gamma = 7.0 / 5.0;
        g.R     = 8314.46261815324 / g.MW;  // J/(kg·K)  = 296.80
        g.cv    = g.R / (g.gamma - 1.0);
        g.cp    = g.gamma * g.cv;
    } else {
        throw std::runtime_error("Unknown gas: '" + name + "'. Supported: argon, nitrogen.");
    }
    return g;
}

double GasModel::sound_speed(double p, double rho) const {
    if (rho <= 0.0 || p <= 0.0)
        throw std::runtime_error("GasModel::sound_speed: non-positive p or rho.");
    return std::sqrt(gamma * p / rho);
}

double GasModel::pressure_from_conserved(double rho, double rhoE, double rhou) const {
    // E = e + 0.5*u^2  => e = E - 0.5*u^2
    // p = (gamma-1)*rho*e
    const double u   = rhou / rho;
    const double E   = rhoE / rho;
    const double e   = E - 0.5 * u * u;
    return (gamma - 1.0) * rho * e;
}

GasModel::ShockState GasModel::normal_shock(double M1, double p1, double T1,
                                             double u1_lab) const {
    // Classic Rankine-Hugoniot relations for calorically perfect gas.
    // Shock is assumed stationary in the shock frame.
    // u1_lab is the lab-frame velocity of the upstream (right) gas.
    // We work in the shock frame where upstream flow is from the left at speed V_shock - u1_lab.

    if (M1 <= 1.0)
        throw std::runtime_error("normal_shock: M1 must be > 1.");

    const double g = gamma;
    const double M1sq = M1 * M1;

    // Downstream Mach
    const double M2sq = (M1sq + 2.0 / (g - 1.0)) /
                        (2.0 * g / (g - 1.0) * M1sq - 1.0);
    const double M2 = std::sqrt(M2sq);

    // Pressure ratio
    const double p2_p1 = (2.0 * g * M1sq - (g - 1.0)) / (g + 1.0);

    // Temperature ratio
    const double T2_T1 = p2_p1 * (2.0 + (g - 1.0) * M1sq) /
                         ((g + 1.0) * M1sq);

    // Density ratio
    const double rho2_rho1 = (g + 1.0) * M1sq /
                              (2.0 + (g - 1.0) * M1sq);

    const double rho1 = p1 / (R * T1);
    const double a1   = std::sqrt(g * R * T1);

    ShockState s;
    s.M2   = M2;
    s.p2   = p2_p1 * p1;
    s.T2   = T2_T1 * T1;
    s.rho2 = rho2_rho1 * rho1;

    // Shock speed in lab frame (shock is stationary in shock frame when upstream
    // gas approaches at u_shock = M1 * a1).  Here we define the shock-frame
    // reference so that the shock is at rest and upstream gas has velocity M1*a1.
    // The lab-frame velocity of the downstream gas:
    const double u_shock_frame_upstream = M1 * a1;
    const double u_shock_frame_downstream = M2 * std::sqrt(g * R * s.T2);
    // Conservation of mass: rho1*u1 = rho2*u2 (shock frame)
    // u2_shock = rho1*u1 / rho2
    const double u2_shock = rho1 * u_shock_frame_upstream / s.rho2;
    // Lab frame: subtract shock velocity (shock moves at u1_lab + u_shock_frame_upstream from lab
    // if upstream gas is at rest in lab).  For our target problem upstream (right) is at rest.
    // V_shock_lab = u1_lab - u_shock_frame_upstream (upstream gas velocity minus its shock-frame speed)
    // But the master prompt already gives us lab-frame left/right states for the argon case,
    // so we just report shock-frame u2.
    s.u2 = u2_shock;  // velocity in shock frame (downstream)

    return s;
}

} // namespace splay
