#include "splay/dg_av.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace splay {

// ── Persson-Peraire smoothness sensor ────────────────────────────────────────
//
// For each cell, project the density onto the normalised Legendre modal basis
// using the precomputed modal_proj tables in DGBasis:
//
//   a_j = Σ_k modal_proj[j][k] · ρ_k
//
// Then compute:
//   ||ρ_e||²        = Σ_j a_j²           (total modal energy)
//   ||ρ_e-Π_{p-1}||² = a_p²             (energy in highest mode)
//   Se = log10(a_p² / ||ρ_e||²)
//
// The AV coefficient follows a smooth sigmoid:
//
//   factor = 0                                  if Se < s0 - κ
//   factor = ½(1 - cos(π(Se-s0+κ)/(2κ)))       if s0-κ ≤ Se ≤ s0+κ
//   factor = 1                                  if Se > s0 + κ
//
//   ε_i = C_av · h · c_max · factor
//
// We use density as the indicator variable (captures contact discontinuities
// and shocks; for pure acoustic waves this will give ε ≈ 0 as desired).

static void persson_peraire(DGState& s, const Mesh& m,
                             const GasModel& gas, const DGConfig& cfg) {
    const int    ib    = m.interior_begin();
    const int    ie    = m.interior_end();
    const int    nd    = s.n_dof;
    const double h     = m.dx;
    const double s0    = cfg.s0;
    const double kappa = cfg.kappa;
    const double C_av  = cfg.C_av;
    const auto&  mp    = s.basis.modal_proj;

    for (int i = ib; i < ie; ++i) {
        // ── Modal projection of density ──────────────────────────────────────
        double modal_energy = 0.0;
        double a_p          = 0.0;  // highest-mode coefficient

        for (int mode = 0; mode < nd; ++mode) {
            double a = 0.0;
            for (int k = 0; k < nd; ++k)
                a += mp[mode][k] * s.rho[k][i];
            modal_energy += a * a;
            if (mode == nd - 1) a_p = a;  // highest mode is last
        }

        // ── Smoothness indicator ─────────────────────────────────────────────
        double Se;
        if (modal_energy < 1e-30 || a_p * a_p < 1e-60) {
            Se = s0 - 2.0 * kappa;   // treat as fully smooth
        } else {
            Se = std::log10(a_p * a_p / modal_energy);
        }

        // ── Sigmoid ramp ─────────────────────────────────────────────────────
        double factor;
        if (Se < s0 - kappa) {
            factor = 0.0;
        } else if (Se > s0 + kappa) {
            factor = 1.0;
        } else {
            const double phi = (Se - s0 + kappa) / (2.0 * kappa);
            factor = 0.5 * (1.0 - std::cos(M_PI * phi));
        }

        // ── Maximum wave speed in cell ───────────────────────────────────────
        double c_max = 0.0;
        for (int j = 0; j < nd; ++j) {
            const double a  = gas.sound_speed(s.prim_p[j][i], s.rho[j][i]);
            c_max = std::max(c_max, std::abs(s.u[j][i]) + a);
        }

        s.epsilon[i] = C_av * h * c_max * factor;
    }
}

// ── Residual-based sensor ─────────────────────────────────────────────────────
//
// Estimate ε from the magnitude of the strong-form flux divergence:
//
//   (∂F/∂x)_k ≈ (2/h) · Σ_j ρu[j][i] · D[k][j]   (mass flux divergence)
//
// Using the mass equation as the indicator (density changes fastest at shocks).
//
//   r_K = max_k |(∂F_ρ/∂x)_k|
//   ref_K = max(max_k |ρu_k| / h,  small)          (prevents division by zero
//                                                     in trivial regions)
//   ε_K = C_av · h² · r_K / ref_K

static void residual_based(DGState& s, const Mesh& m,
                            const DGConfig& cfg) {
    const int    ib  = m.interior_begin();
    const int    ie  = m.interior_end();
    const int    nd  = s.n_dof;
    const double h   = m.dx;
    const double inv_h = 2.0 / h;
    const auto&  D   = s.basis.D;
    const double C_av = cfg.C_av;

    for (int i = ib; i < ie; ++i) {
        double r_max   = 0.0;
        double ref_max = 1e-20;

        for (int k = 0; k < nd; ++k) {
            // Strong-form mass-flux divergence at node k
            double dFdx = 0.0;
            for (int j = 0; j < nd; ++j)
                dFdx += s.rhou[j][i] * D[k][j];
            dFdx *= inv_h;

            r_max   = std::max(r_max,   std::abs(dFdx));
            ref_max = std::max(ref_max, std::abs(s.rhou[k][i]) / h);
        }

        s.epsilon[i] = C_av * h * h * r_max / ref_max;
    }
}

// ── Public interface ─────────────────────────────────────────────────────────

void compute_av_dg(DGState& s, const Mesh& m,
                   const GasModel& gas, const DGConfig& cfg) {
    // Zero out ghost cells (they are not needed for AV flux assembly)
    std::fill(s.epsilon.begin(), s.epsilon.end(), 0.0);

    switch (cfg.av_method) {
        case AVMethod::PerssonPeraire:
            persson_peraire(s, m, gas, cfg);
            break;
        case AVMethod::ResidualBased:
            residual_based(s, m, cfg);
            break;
        case AVMethod::None:
            break;  // epsilon remains zero
    }
}

} // namespace splay
