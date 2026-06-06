#pragma once
#include <array>
#include <string>
#include <vector>

namespace splay {

/// Temperature-dependent viscosity and conductivity via 4th-order polynomial.
///
/// mu(T) = sum_{k=0}^{4} mu_coeff[k] * T^k   (Pa·s)
/// k (T) = sum_{k=0}^{4} k_coeff[k]  * T^k   (W/(m·K))
///
/// Values are clamped to a minimum of zero and a warning is issued if T is
/// outside [T_min, T_max].
struct TransportModel {
    std::string name;
    double      T_min;
    double      T_max;

    std::array<double, 5> mu_coeff;   // viscosity polynomial coefficients
    std::array<double, 5>  k_coeff;   // conductivity polynomial coefficients

    /// Evaluate dynamic viscosity mu(T) in Pa·s.
    double viscosity(double T) const;

    /// Evaluate thermal conductivity k(T) in W/(m·K).
    double conductivity(double T) const;

    /// Load coefficients by gas name ("argon" or "nitrogen").
    static TransportModel from_name(const std::string& gas_name);

private:
    static double poly4(const std::array<double, 5>& c, double T);
};

/// Evaluate transport for all cells (interior + ghost).
void compute_transport(const std::vector<double>& T_vec,
                       const TransportModel& tm,
                       std::vector<double>& mu,
                       std::vector<double>& kappa);

} // namespace splay
