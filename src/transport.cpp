#include "splay/transport.hpp"
#include "splay/transport_coeffs.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace splay {

double TransportModel::poly4(const std::array<double, 5>& c, double T) {
    // Horner's method
    return c[0] + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])));
}

double TransportModel::viscosity(double T) const {
    if (T < T_min * 0.95 || T > T_max * 1.05) {
        std::cerr << "[SPLAY] Warning: T=" << T << " K is outside transport fit range ["
                  << T_min << ", " << T_max << "] for " << name << "\n";
    }
    const double val = poly4(mu_coeff, T);
    if (val <= 0.0) {
        std::cerr << "[SPLAY] Warning: viscosity evaluated to " << val
                  << " Pa·s at T=" << T << " K; clamping to zero.\n";
        return 0.0;
    }
    return val;
}

double TransportModel::conductivity(double T) const {
    if (T < T_min * 0.95 || T > T_max * 1.05) {
        std::cerr << "[SPLAY] Warning: T=" << T << " K is outside transport fit range ["
                  << T_min << ", " << T_max << "] for " << name << "\n";
    }
    const double val = poly4(k_coeff, T);
    if (val <= 0.0) {
        std::cerr << "[SPLAY] Warning: conductivity evaluated to " << val
                  << " W/(m·K) at T=" << T << " K; clamping to zero.\n";
        return 0.0;
    }
    return val;
}

TransportModel TransportModel::from_name(const std::string& gas_name) {
    using namespace transport_data;
    TransportModel tm;
    tm.name = gas_name;

    if (gas_name == "argon") {
        tm.T_min = argon_T_min;
        tm.T_max = argon_T_max;
        for (int k = 0; k < 5; ++k) {
            tm.mu_coeff[k] = argon_mu[k];
            tm.k_coeff[k]  = argon_k[k];
        }
    } else if (gas_name == "nitrogen") {
        tm.T_min = nitrogen_T_min;
        tm.T_max = nitrogen_T_max;
        for (int k = 0; k < 5; ++k) {
            tm.mu_coeff[k] = nitrogen_mu[k];
            tm.k_coeff[k]  = nitrogen_k[k];
        }
    } else {
        throw std::runtime_error("TransportModel: unknown gas '" + gas_name + "'.");
    }
    return tm;
}

void compute_transport(const std::vector<double>& T_vec,
                       const TransportModel& tm,
                       std::vector<double>& mu,
                       std::vector<double>& kappa) {
    const int n = static_cast<int>(T_vec.size());
    mu.resize(n);
    kappa.resize(n);
    for (int i = 0; i < n; ++i) {
        mu[i]    = tm.viscosity(T_vec[i]);
        kappa[i] = tm.conductivity(T_vec[i]);
    }
}

} // namespace splay
