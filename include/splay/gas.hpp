#pragma once
#include <string>
#include <stdexcept>

namespace splay {

// ─── Gas properties for a calorically/thermally perfect gas ──────────────────
struct GasModel {
    std::string name;
    double      MW;     // kg/mol
    double      R;      // J/(kg·K)  specific gas constant
    double      gamma;  // ratio of specific heats
    double      cp;     // J/(kg·K)
    double      cv;     // J/(kg·K)

    /// Construct from name: "argon" or "nitrogen".
    static GasModel from_name(const std::string& name);

    // ── Primitive ↔ Conservative ─────────────────────────────────────────────
    /// p from rho, T
    double pressure(double rho, double T) const { return rho * R * T; }
    /// T from rho, p
    double temperature(double rho, double p) const { return p / (rho * R); }
    /// internal energy per unit mass from T
    double internal_energy(double T) const { return cv * T; }
    /// total energy per unit mass from T, u
    double total_energy(double T, double u) const { return cv * T + 0.5 * u * u; }
    /// speed of sound
    double sound_speed(double p, double rho) const;
    /// p from rho, E (total), u
    double pressure_from_conserved(double rho, double rhoE, double rhou) const;

    // ── Normal-shock relations (given upstream Mach M1) ───────────────────────
    struct ShockState {
        double M2;    // downstream Mach
        double p2;    // downstream pressure   (Pa)
        double T2;    // downstream temperature (K)
        double rho2;  // downstream density     (kg/m³)
        double u2;    // downstream velocity    (m/s) — shock frame
    };
    ShockState normal_shock(double M1, double p1, double T1, double u1_lab) const;
};

} // namespace splay
