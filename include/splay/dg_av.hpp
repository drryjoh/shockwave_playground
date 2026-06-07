#pragma once
#include "splay/dg_state.hpp"
#include "splay/mesh.hpp"
#include "splay/gas.hpp"
#include "splay/config.hpp"

namespace splay {

// ── Artificial Viscosity for DG ───────────────────────────────────────────────
//
// AV adds a diffusion term ε(x) ∂²Q/∂x² to all conserved equations.
// It is needed for p ≥ 1 DG on Euler problems where no limiter is applied.
//
// Two sensor methods are supported:
//
// 1. PerssonPeraire (default for Euler)
//    Measures the ratio of energy in the highest polynomial mode to total
//    energy via a modal projection.  Smooth solutions have Se << s0 (ε ≈ 0);
//    shocks have Se ≈ s0 (ε = C_av · h · c_max).
//    Reference: Persson & Peraire, AIAA-2006-3requested.
//
// 2. ResidualBased
//    Uses the strong-form PDE residual magnitude as the sensor.  More natural
//    for NS problems where physical diffusion already scales with gradients.
//    ε_K = C_av · h² · ||∂F/∂x||_K / max(||F||_K / h, small)
//
// The computed ε values are stored in DGState::epsilon for use in dg_residual.

// ── DGConfig is defined in config.hpp ────────────────────────────────────────

/// Compute per-cell artificial viscosity coefficients and store in s.epsilon.
/// Called once per RK stage, before compute_residual_dg.
void compute_av_dg(DGState& s,
                   const Mesh& m,
                   const GasModel& gas,
                   const DGConfig& cfg);

} // namespace splay
