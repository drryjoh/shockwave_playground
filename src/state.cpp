#include "splay/state.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace splay {

State::State(int n_total) {
    rho.assign(n_total, 0.0);
    rhou.assign(n_total, 0.0);
    rhoE.assign(n_total, 0.0);
    u.assign(n_total, 0.0);
    p.assign(n_total, 0.0);
    T.assign(n_total, 0.0);
}

void State::update_primitives(const GasModel& gas) {
    const int n = static_cast<int>(rho.size());
    for (int i = 0; i < n; ++i) {
        update_primitives_cell(i, gas);
    }
}

void State::update_primitives_cell(int i, const GasModel& gas) {
    if (rho[i] <= 0.0)
        throw std::runtime_error("State: non-positive density at cell " + std::to_string(i) +
                                 " rho=" + std::to_string(rho[i]));
    u[i] = rhou[i] / rho[i];
    p[i] = gas.pressure_from_conserved(rho[i], rhoE[i], rhou[i]);
    if (p[i] <= 0.0)
        throw std::runtime_error("State: non-positive pressure at cell " + std::to_string(i) +
                                 " p=" + std::to_string(p[i]));
    T[i] = gas.temperature(rho[i], p[i]);
    if (T[i] <= 0.0)
        throw std::runtime_error("State: non-positive temperature at cell " + std::to_string(i) +
                                 " T=" + std::to_string(T[i]));
}

double State::l2_norm(const std::vector<double>& r) {
    double sum = 0.0;
    for (double v : r) sum += v * v;
    return std::sqrt(sum / static_cast<double>(r.size()));
}

void init_tanh(State& s, const Mesh& m, const GasModel& gas,
               double p_L, double T_L, double u_L,
               double p_R, double T_R, double u_R,
               double x_shock, double delta) {
    const int n = m.n_total;
    for (int i = 0; i < n; ++i) {
        const double x    = m.x_cell[i];
        // phi(x) = 0.5*(L+R) + 0.5*(R-L)*tanh((x-xs)/delta)
        // => for x << xs gives L; for x >> xs gives R.
        const double th   = std::tanh((x - x_shock) / delta);
        const double fac  = 0.5 * (1.0 + th);  // 0 at left, 1 at right

        const double p_i  = p_L  + (p_R  - p_L)  * fac;
        const double T_i  = T_L  + (T_R  - T_L)  * fac;
        const double u_i  = u_L  + (u_R  - u_L)  * fac;

        s.p[i] = p_i;
        s.T[i] = T_i;
        s.u[i] = u_i;

        s.rho[i]  = p_i / (gas.R * T_i);
        const double e_i  = gas.cv * T_i;
        const double E_i  = e_i + 0.5 * u_i * u_i;
        s.rhou[i] = s.rho[i] * u_i;
        s.rhoE[i] = s.rho[i] * E_i;
    }
}

} // namespace splay
