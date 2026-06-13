#include "splay/reconstruction_ppm_pele.hpp"
#include <cmath>
#include <algorithm>

namespace splay {

// ─── Parabola build (CW84 §1, same algorithm as reconstruct_ppm_scalar) ──────
// Builds per-cell parabola edge values aL[i], aR[i].
// Requires at least 2 ghost cells on each side (accesses i-1 and i+2).
static void build_ppm_parabolas(
    const std::vector<double>& q,
    std::vector<double>& aL,
    std::vector<double>& aR)
{
    const int n = static_cast<int>(q.size());

    // Step 1: 4th-order face interpolation + monotone clamp (CW84 eq. 1.6)
    std::vector<double> qi_half(n, 0.0);
    for (int i = 1; i < n - 2; ++i) {
        qi_half[i] = (7.0 * (q[i] + q[i+1]) - (q[i-1] + q[i+2])) / 12.0;
        const double qlo = std::min(q[i], q[i+1]);
        const double qhi = std::max(q[i], q[i+1]);
        qi_half[i] = std::max(qlo, std::min(qhi, qi_half[i]));
    }

    // Step 2: assign parabola edge values per cell
    for (int i = 1; i < n - 1; ++i) {
        aL[i] = qi_half[i-1];
        aR[i] = qi_half[i];
    }

    // Step 3: CW84 monotonicity constraints (eq. 1.10)
    for (int i = 1; i < n - 1; ++i) {
        if ((aR[i] - q[i]) * (q[i] - aL[i]) <= 0.0) {
            aL[i] = q[i];
            aR[i] = q[i];
        } else {
            const double dq = aR[i] - aL[i];
            const double q6 = 6.0 * (q[i] - 0.5 * (aL[i] + aR[i]));
            if (dq * (dq - q6) < 0.0) aR[i] = 3.0 * q[i] - 2.0 * aL[i];
            if (dq * (dq + q6) < 0.0) aL[i] = 3.0 * q[i] - 2.0 * aR[i];
        }
        // Additional clamp to neighbouring cell range (prevents negative
        // density/pressure at strong shocks, same guard as existing PPM).
        const double lo = std::min({q[i-1], q[i], q[i+1]});
        const double hi = std::max({q[i-1], q[i], q[i+1]});
        aL[i] = std::max(lo, std::min(hi, aL[i]));
        aR[i] = std::max(lo, std::min(hi, aR[i]));
    }
}

// ─── Parabola averaging helpers (CW84 eq. 1.10) ──────────────────────────────

// Average of the parabola over the rightmost fraction sigma of the cell.
// sigma = 0 → returns aR (right edge); sigma = 1 → returns q (cell average).
// Used for the LEFT state at the right face of cell i.
static inline double pavg_right(
    double aL, double aR, double q, double sigma)
{
    const double q6 = 6.0 * q - 3.0 * (aL + aR);
    return aR - 0.5 * sigma * ((aR - aL) - (1.0 - 2.0 * sigma / 3.0) * q6);
}

// Average of the parabola over the leftmost fraction sigma of the cell.
// sigma = 0 → returns aL (left edge); sigma = 1 → returns q (cell average).
// Used for the RIGHT state at the left face of cell i+1.
static inline double pavg_left(
    double aL, double aR, double q, double sigma)
{
    const double q6 = 6.0 * q - 3.0 * (aL + aR);
    return aL + 0.5 * sigma * ((aR - aL) + (1.0 - 2.0 * sigma / 3.0) * q6);
}

// ─── CW84/PeleC shock flattening (re-implemented locally) ────────────────────
// Identical algorithm to compute_flattening() in reconstruction.cpp.
static std::vector<double> compute_flattening_pele(
    const std::vector<double>& p,
    const std::vector<double>& u,
    int n, double z1, double z2, double shktst)
{
    std::vector<double> chi(n, 0.0);
    for (int i = 2; i < n - 2; ++i) {
        if (u[i+1] - u[i-1] >= 0.0) continue;
        const double dp_near = std::abs(p[i+1] - p[i-1]);
        const double dp_far  = std::abs(p[i+2] - p[i-2]);
        if (dp_far < 1e-14) continue;
        // PeleC shktst gate: suppress flattening for weak pressure perturbations.
        const double p_min = std::min(p[i+1], p[i-1]);
        if (p_min > 0.0 && dp_near / p_min <= shktst) continue;
        const double zeta = dp_near / dp_far;
        chi[i] = std::max(0.0, std::min(1.0, (zeta - z1) / (z2 - z1)));
    }
    std::vector<double> out(n, 0.0);
    for (int i = 1; i < n - 1; ++i)
        out[i] = std::max({chi[i-1], chi[i], chi[i+1]});
    return out;
}

// ─── Main entry point ─────────────────────────────────────────────────────────
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
    bool   flatten,
    double flatten_z1,
    double flatten_z2,
    double flatten_shktst)
{
    const int n = static_cast<int>(rho.size());
    const double dt_dx = (dx > 0.0) ? dt / dx : 0.0;

    // ── Build per-cell parabola edge values ───────────────────────────────────
    std::vector<double> aL_rho(n, 0.0), aR_rho(n, 0.0);
    std::vector<double> aL_u(n, 0.0),   aR_u(n, 0.0);
    std::vector<double> aL_p(n, 0.0),   aR_p(n, 0.0);

    build_ppm_parabolas(rho, aL_rho, aR_rho);
    build_ppm_parabolas(u,   aL_u,   aR_u);
    build_ppm_parabolas(p,   aL_p,   aR_p);

    // ── Characteristic tracing per face ───────────────────────────────────────
    //
    // PPM_pele approximation:
    //   1D primitive characteristic projection equivalent in spirit to PeleC,
    //   not a direct AMReX/PeleC copy.  Wave speeds evaluated at cell centres
    //   (t^n), not at the characteristic foot — O(dt) approximation.
    //
    // For each face f between cell i (left) and i+1 (right):
    //
    //   LEFT state (from cell i, waves moving rightward toward the face):
    //     u+c  wave: sigma_p = max(0, min(1, (u[i]+c[i])*dt/dx))  always >0 subsonic
    //     u    wave: sigma_0 = max(0, min(1,  u[i]       *dt/dx))  >0 if u>0
    //     u-c  wave: sigma_m = max(0, min(1, (u[i]-c[i])*dt/dx))  0 in subsonic
    //
    //   RIGHT state (from cell i+1, waves moving leftward toward the face):
    //     u-c  wave: sm2 = max(0, min(1,-(u[i+1]-c[i+1])*dt/dx)) always >0 subsonic
    //     u    wave: s02 = max(0, min(1, -u[i+1]         *dt/dx)) >0 if u<0
    //     u+c  wave: sp2 = max(0, min(1,-(u[i+1]+c[i+1])*dt/dx)) 0 in subsonic
    //
    // Characteristic assembly (acoustic Riemann invariants):
    //   beta_m = acoustic amplitude of u-c correction = 0 in subsonic
    //   drho_s = entropy density correction from contact wave
    //   p_L  = p_sp  + beta_m
    //   u_L  = u_sp  - beta_m / (rho*c)
    //   rho_L = rho_sp + drho_s + beta_m / c^2

    for (int f = face_begin; f < face_end; ++f) {
        const int i  = f;       // left cell
        const int ip = f + 1;   // right cell

        // ── LEFT state (from cell i) ──────────────────────────────────────────
        {
            const double ci  = std::sqrt(gamma * p[i] / rho[i]);

            // Characteristic CFL fractions
            const double sp = std::max(0.0, std::min(1.0, (u[i] + ci) * dt_dx));
            const double s0 = std::max(0.0, std::min(1.0,  u[i]        * dt_dx));
            const double sm = std::max(0.0, std::min(1.0, (u[i] - ci) * dt_dx));

            // Parabola averages for u+c wave (always the primary wave for left face)
            const double rho_sp = pavg_right(aL_rho[i], aR_rho[i], rho[i], sp);
            const double u_sp   = pavg_right(aL_u[i],   aR_u[i],   u[i],   sp);
            const double p_sp   = pavg_right(aL_p[i],   aR_p[i],   p[i],   sp);

            // Entropy wave: only density is carried (p unchanged across contact)
            const double rho_s0 = (s0 > 0.0)
                ? pavg_right(aL_rho[i], aR_rho[i], rho[i], s0)
                : aR_rho[i];

            // u-c wave: zero contribution in subsonic flow (sm = 0)
            const double rho_sm = (sm > 0.0)
                ? pavg_right(aL_rho[i], aR_rho[i], rho[i], sm) : rho_sp;
            const double u_sm   = (sm > 0.0)
                ? pavg_right(aL_u[i],   aR_u[i],   u[i],   sm) : u_sp;
            const double p_sm   = (sm > 0.0)
                ? pavg_right(aL_p[i],   aR_p[i],   p[i],   sm) : p_sp;

            // Acoustic amplitude of u-c wave (Riemann invariant dp - rho*c*du)
            const double beta_m = 0.5 * ((p_sm - p_sp) - rho[i] * ci * (u_sm - u_sp));

            // Entropy density correction (contact wave carries density, not pressure)
            const double drho_s = rho_s0 - rho_sp;

            fs.p_L[f]   = p_sp   + beta_m;
            fs.u_L[f]   = u_sp   - beta_m / (rho[i] * ci);
            fs.rho_L[f] = rho_sp + drho_s + beta_m / (ci * ci);

            // Guard: enforce positive density and pressure
            fs.rho_L[f] = std::max(fs.rho_L[f], 1e-14);
            fs.p_L[f]   = std::max(fs.p_L[f],   1e-14);
        }

        // ── RIGHT state (from cell i+1, mirror image) ─────────────────────────
        {
            const double cip = std::sqrt(gamma * p[ip] / rho[ip]);

            // Characteristic CFL fractions (leftward waves, negate speeds)
            const double sm2 = std::max(0.0, std::min(1.0, -(u[ip] - cip) * dt_dx));
            const double s02 = std::max(0.0, std::min(1.0, -u[ip]          * dt_dx));
            const double sp2 = std::max(0.0, std::min(1.0, -(u[ip] + cip) * dt_dx));

            // Parabola averages for u-c wave (primary leftward wave in subsonic)
            const double rho_sm2 = pavg_left(aL_rho[ip], aR_rho[ip], rho[ip], sm2);
            const double u_sm2   = pavg_left(aL_u[ip],   aR_u[ip],   u[ip],   sm2);
            const double p_sm2   = pavg_left(aL_p[ip],   aR_p[ip],   p[ip],   sm2);

            // Entropy wave (only density, leftward if u < 0)
            const double rho_s02 = (s02 > 0.0)
                ? pavg_left(aL_rho[ip], aR_rho[ip], rho[ip], s02)
                : aL_rho[ip];

            // u+c wave: zero in subsonic (sp2 = 0)
            const double rho_sp2 = (sp2 > 0.0)
                ? pavg_left(aL_rho[ip], aR_rho[ip], rho[ip], sp2) : rho_sm2;
            const double u_sp2   = (sp2 > 0.0)
                ? pavg_left(aL_u[ip],   aR_u[ip],   u[ip],   sp2) : u_sm2;
            const double p_sp2   = (sp2 > 0.0)
                ? pavg_left(aL_p[ip],   aR_p[ip],   p[ip],   sp2) : p_sm2;

            // Acoustic amplitude of u+c wave (Riemann invariant dp + rho*c*du)
            const double beta_p = 0.5 * ((p_sp2 - p_sm2) + rho[ip] * cip * (u_sp2 - u_sm2));

            // Entropy density correction
            const double drho_s2 = rho_s02 - rho_sm2;

            fs.p_R[f]   = p_sm2   + beta_p;
            fs.u_R[f]   = u_sm2   + beta_p / (rho[ip] * cip);
            fs.rho_R[f] = rho_sm2 + drho_s2 + beta_p / (cip * cip);

            fs.rho_R[f] = std::max(fs.rho_R[f], 1e-14);
            fs.p_R[f]   = std::max(fs.p_R[f],   1e-14);
        }
    }

    // ── Optional CW84/PeleC flattening ────────────────────────────────────────
    // Applied after tracing: blends traced face states toward cell averages
    // near strong shocks detected by the pressure-ratio criterion.
    if (flatten) {
        const auto chi = compute_flattening_pele(p, u, n, flatten_z1, flatten_z2, flatten_shktst);
        for (int f = face_begin; f < face_end; ++f) {
            const double cf  = chi[f];
            const double cf1 = chi[f + 1];
            fs.rho_L[f] = (1.0 - cf)  * fs.rho_L[f] + cf  * rho[f];
            fs.u_L[f]   = (1.0 - cf)  * fs.u_L[f]   + cf  * u[f];
            fs.p_L[f]   = (1.0 - cf)  * fs.p_L[f]   + cf  * p[f];
            fs.rho_R[f] = (1.0 - cf1) * fs.rho_R[f] + cf1 * rho[f + 1];
            fs.u_R[f]   = (1.0 - cf1) * fs.u_R[f]   + cf1 * u[f + 1];
            fs.p_R[f]   = (1.0 - cf1) * fs.p_R[f]   + cf1 * p[f + 1];
        }
    }

    // ── Derive T from ideal gas; fill conservative face states ────────────────
    for (int f = face_begin; f < face_end; ++f) {
        fs.T_L[f] = fs.p_L[f] / (fs.rho_L[f] * R_gas);
        fs.T_R[f] = fs.p_R[f] / (fs.rho_R[f] * R_gas);

        const double EL = fs.p_L[f] / ((gamma - 1.0) * fs.rho_L[f])
                        + 0.5 * fs.u_L[f] * fs.u_L[f];
        const double ER = fs.p_R[f] / ((gamma - 1.0) * fs.rho_R[f])
                        + 0.5 * fs.u_R[f] * fs.u_R[f];

        fs.rhou_L[f] = fs.rho_L[f] * fs.u_L[f];
        fs.rhou_R[f] = fs.rho_R[f] * fs.u_R[f];
        fs.rhoE_L[f] = fs.rho_L[f] * EL;
        fs.rhoE_R[f] = fs.rho_R[f] * ER;
    }
}

} // namespace splay
