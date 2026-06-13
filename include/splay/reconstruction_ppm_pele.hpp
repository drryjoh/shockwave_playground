#pragma once
#include <vector>
#include "splay/reconstruction.hpp"

namespace splay {

/// PeleC-style PPM reconstruction with characteristic tracing.
///
/// Produces time-centered face states at t^{n+1/2} by integrating each
/// characteristic wave's contribution over the fraction of the cell it
/// traverses in dt, following Colella & Woodward 1984 (CW84) §3 and
/// PeleC (AMReX-Combustion/PeleC) Source/PPM.cpp.
///
/// Key difference from the existing splay PPM:
///   Existing PPM:  purely spatial — left state = aR[i], right state = aL[i+1]
///   PPM_pele:      characteristic tracing — face states are sigma-weighted
///                  averages under the parabola, time-centred at t^{n+1/2}
///
/// Approximations vs full PeleC (documented here for reproducibility):
///   1. Tracing uses cell-centre wave speeds (u, c) evaluated at t^n,
///      not iterated to the characteristic foot — O(dt) approximation.
///   2. Temperature is derived from the traced (rho, p) via ideal gas
///      T = p/(rho*R) rather than being traced independently; T = p/(rho*R)
///      is exact for an ideal gas so this introduces no error.
///   3. Flattening (if enabled) is applied after tracing, identical to the
///      existing splay CW84 flattening path.
///   4. 1D only: no transverse-gradient corrections, no multi-D flattening.
///
/// @param rho          cell-centred density   [kg/m³]  (size n_total)
/// @param u            cell-centred velocity  [m/s]    (size n_total)
/// @param p            cell-centred pressure  [Pa]     (size n_total)
/// @param T            cell-centred temperature [K]    (size n_total, read-only)
/// @param gamma        ratio of specific heats
/// @param R_gas        specific gas constant  [J/(kg·K)]
/// @param dt           current time-step size [s]  (0 → degrades to spatial PPM)
/// @param dx           uniform cell width     [m]
/// @param face_begin   first face index to reconstruct
/// @param face_end     one-past-last face index
/// @param fs           output FaceStates (pre-allocated to n_total)
/// @param flatten        enable CW84/PeleC-style shock flattening
/// @param flatten_z1     lower pressure-ratio threshold (ramp start)
/// @param flatten_z2     upper pressure-ratio threshold (fully flattened)
/// @param flatten_shktst PeleC shktst gate: |dp|/min(p[±1]) must exceed this
///                       before flattening is applied (PeleC default: 0.33)
void reconstruct_ppm_pele(
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& p,
    const std::vector<double>& T,
    double gamma,
    double R_gas,
    double dt,
    double dx,
    int    face_begin,
    int    face_end,
    FaceStates& fs,
    bool   flatten        = false,
    double flatten_z1     = 0.75,
    double flatten_z2     = 0.85,
    double flatten_shktst = 0.33);

} // namespace splay
