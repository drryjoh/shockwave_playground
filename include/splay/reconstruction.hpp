#pragma once
#include <vector>
#include "splay/config.hpp"

namespace splay {

/// Reconstruct left/right face-states for cell interface i+1/2.
///
/// Given cell-centred primitive variables q[0..n-1],
/// returns q_L (left state at face i+1/2) and q_R (right state at face i+1/2).
///
/// All arrays are indexed over the full local+ghost array.
/// face_L[i] = left state  at interface between cell i and i+1
/// face_R[i] = right state at interface between cell i and i+1

struct FaceStates {
    // One entry per face: face i lives between cell i and i+1.
    // Allocated for n_faces = n_total - 1.
    std::vector<double> rho_L, rhou_L, rhoE_L;
    std::vector<double> rho_R, rhou_R, rhoE_R;

    // Primitive helpers (filled from conservative by caller)
    std::vector<double> p_L, u_L, T_L;
    std::vector<double> p_R, u_R, T_R;

    explicit FaceStates(int n_faces);
};

/// Compute face reconstructions using the selected scheme.
/// Works on cell index range [face_begin, face_end).
/// face_begin is the first face index to reconstruct (= interior_begin - 1 typically).
void reconstruct(
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& p,
    const std::vector<double>& T,
    InviscidScheme scheme,
    Limiter        limiter,
    double         gamma,
    int            face_begin,
    int            face_end,
    FaceStates&    fs);

// ── Individual limiters ────────────────────────────────────────────────────
double limiter_minmod(double r);
double limiter_vanleer(double r);
double limiter_mc(double r);

} // namespace splay
