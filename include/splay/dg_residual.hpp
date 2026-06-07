#pragma once
#include "splay/dg_state.hpp"
#include "splay/mesh.hpp"
#include "splay/gas.hpp"
#include "splay/transport.hpp"
#include "splay/config.hpp"
#include "splay/residual.hpp"   // ResidualPart

namespace splay {

// ── DG Right-hand side ────────────────────────────────────────────────────────
//
// Assembles the DG residual dU_j/dt for each DOF j of each interior cell.
//
// Four contributions (each guarded by ResidualPart and cfg.viscous_terms):
//
//   1. Inviscid volume:  +(2/(dx·w_j)) · Σ_k w_k · F_inv(U_k) · D[k][j]
//   2. Inviscid face:    -(2/(dx·w_j)) · (F*_inv_R·δ_{j,p} - F*_inv_L·δ_{j,0})
//                         F* = Riemann solver (same solvers as FVM: Rusanov / HLLC)
//
//   3. Viscous volume:   -(2/(dx·w_j)) · Σ_k w_k · F_vis(U_k,∇U_k) · D[k][j]
//                         ∇U_k = (2/h) · Σ_j U_j · D[k][j]  (exact DG gradient)
//   4. Viscous face:     +(2/(dx·w_j)) · (F*_vis_R·δ_{j,p} - F*_vis_L·δ_{j,0})
//                         SIPG: F*_vis = μ_avg·{∂u/∂x} − (η/h)·μ_avg·[u]
//                         BR2:  F*_vis = μ_avg·({∂u/∂x} + [u]/(h·w_endpoint))
//
//   5. AV volume:        -(2/(dx·w_j)) · Σ_k w_k · ε_k · (∇Q_k) · D[k][j]
//   6. AV face:          +(2/(dx·w_j)) · ({ε}·{∂Q/∂x}) · (δ_{j,p} - δ_{j,0})
//
// Sign convention: R_j = dU_j/dt, so the time integrator advances U_j += dt·R_j.

/// Compute DG residual for all interior cells.
/// s.epsilon must already be populated (call compute_av_dg before this).
void compute_residual_dg(
    const DGState&        s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    DGResidual&           R,
    ResidualPart          part = ResidualPart::Full);

/// DG-aware timestep: applies CFL factor 1/(2p+1) on top of the wave-speed
/// constraint, and optionally the viscous parabolic constraint.
double compute_dt_dg(
    const DGState&        s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    std::string&          active_constraint);

} // namespace splay
