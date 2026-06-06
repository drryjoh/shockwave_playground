#include "splay/reconstruction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace splay {

FaceStates::FaceStates(int n_faces) {
    rho_L.assign(n_faces, 0.0); rho_R.assign(n_faces, 0.0);
    rhou_L.assign(n_faces, 0.0); rhou_R.assign(n_faces, 0.0);
    rhoE_L.assign(n_faces, 0.0); rhoE_R.assign(n_faces, 0.0);
    p_L.assign(n_faces, 0.0);   p_R.assign(n_faces, 0.0);
    u_L.assign(n_faces, 0.0);   u_R.assign(n_faces, 0.0);
    T_L.assign(n_faces, 0.0);   T_R.assign(n_faces, 0.0);
}

// ─── Limiter functions ────────────────────────────────────────────────────────
double limiter_minmod(double r) {
    return std::max(0.0, std::min(1.0, r));
}

double limiter_vanleer(double r) {
    if (r <= 0.0) return 0.0;
    return (r + std::abs(r)) / (1.0 + std::abs(r));
}

double limiter_mc(double r) {
    return std::max(0.0, std::min({2.0 * r, 0.5 * (1.0 + r), 2.0}));
}

static double apply_limiter(Limiter lim, double r) {
    switch (lim) {
        case Limiter::None:    return 1.0;   // unlimited
        case Limiter::Minmod:  return limiter_minmod(r);
        case Limiter::VanLeer: return limiter_vanleer(r);
        case Limiter::MC:      return limiter_mc(r);
        case Limiter::PPM:     return 1.0;   // PPM handles its own limiting
        default: return 1.0;
    }
}

// ─── MUSCL reconstruction ────────────────────────────────────────────────────
static void reconstruct_muscl(
    const std::vector<double>& q,
    Limiter limiter,
    int face_begin, int face_end,
    std::vector<double>& qL,
    std::vector<double>& qR)
{
    // Face i+1/2 is between cell i and i+1.
    // MUSCL: qL[i] = q[i]   + 0.5 * phi(r) * (q[i]   - q[i-1])
    //        qR[i] = q[i+1] - 0.5 * phi(r) * (q[i+2] - q[i+1])
    for (int f = face_begin; f < face_end; ++f) {
        // Left state: cell i = f
        const double dL  = q[f]   - q[f-1];
        const double dR  = q[f+1] - q[f];
        const double r_L = (std::abs(dR) > 1e-14) ? dL / dR : 0.0;
        const double phi_L = apply_limiter(limiter, r_L);
        qL[f] = q[f]   + 0.5 * phi_L * dR;

        // Right state: cell i+1 = f+1
        const double dRR  = q[f+2] - q[f+1];
        const double dRL  = q[f+1] - q[f];
        const double r_R  = (std::abs(dRL) > 1e-14) ? dRR / dRL : 0.0;
        const double phi_R = apply_limiter(limiter, r_R);
        qR[f] = q[f+1] - 0.5 * phi_R * dRL;
    }
}

// ─── PPM reconstruction (Colella & Woodward 1984, simplified) ────────────────
// Parabolic reconstruction with monotonicity constraint.
// Does not implement contact steepening or flattening — kept simple for clarity.
static void reconstruct_ppm_scalar(
    const std::vector<double>& q,
    int face_begin, int face_end,
    std::vector<double>& qL,
    std::vector<double>& qR)
{
    const int n = static_cast<int>(q.size());

    // Step 1: build interface values q_{i+1/2} using 4th-order interpolation.
    // q_{i+1/2} = (7*(q_i + q_{i+1}) - (q_{i-1} + q_{i+2})) / 12
    std::vector<double> qi_half(n, 0.0);
    for (int i = 1; i < n - 2; ++i) {
        qi_half[i] = (7.0 * (q[i] + q[i+1]) - (q[i-1] + q[i+2])) / 12.0;
    }

    // Step 2: PPM monotonicity constraint on cell-average parabola.
    // For each cell i, the parabola has edge values: aL = q_{i-1/2}, aR = q_{i+1/2}
    std::vector<double> aL(n, 0.0), aR(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        aL[i] = qi_half[i-1];
        aR[i] = qi_half[i];
    }

    // Monotonicity: constrain so parabola doesn't overshoot neighbours.
    for (int i = 1; i < n - 1; ++i) {
        if ((aR[i] - q[i]) * (q[i] - aL[i]) <= 0.0) {
            aL[i] = q[i];
            aR[i] = q[i];
        } else {
            // dq = aR - aL
            double dq = aR[i] - aL[i];
            double q6 = 6.0 * (q[i] - 0.5 * (aL[i] + aR[i]));
            if (dq * (dq - q6) < 0.0) aL[i] = 3.0 * q[i] - 2.0 * aR[i];
            if (dq * (dq + q6) < 0.0) aR[i] = 3.0 * q[i] - 2.0 * aL[i];
        }
    }

    // Step 3: left state at face f is aR[f]; right state is aL[f+1].
    for (int f = face_begin; f < face_end; ++f) {
        qL[f] = aR[f];
        qR[f] = aL[f+1];
    }
}

// ─── Central reconstruction ──────────────────────────────────────────────────
static void reconstruct_central(
    const std::vector<double>& q,
    int face_begin, int face_end,
    std::vector<double>& qL,
    std::vector<double>& qR)
{
    // Second-order piecewise-linear unlimited (central) reconstruction.
    for (int f = face_begin; f < face_end; ++f) {
        qL[f] = 0.5 * (q[f]   + q[f+1]);
        qR[f] = 0.5 * (q[f]   + q[f+1]);
    }
}

// ─── Main entry point ─────────────────────────────────────────────────────────
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
    FaceStates&    fs)
{
    switch (scheme) {
        case InviscidScheme::Central:
            reconstruct_central(rho, face_begin, face_end, fs.rho_L, fs.rho_R);
            reconstruct_central(u,   face_begin, face_end, fs.u_L,   fs.u_R);
            reconstruct_central(p,   face_begin, face_end, fs.p_L,   fs.p_R);
            reconstruct_central(T,   face_begin, face_end, fs.T_L,   fs.T_R);
            break;

        case InviscidScheme::MUSCL:
            reconstruct_muscl(rho, limiter, face_begin, face_end, fs.rho_L, fs.rho_R);
            reconstruct_muscl(u,   limiter, face_begin, face_end, fs.u_L,   fs.u_R);
            reconstruct_muscl(p,   limiter, face_begin, face_end, fs.p_L,   fs.p_R);
            reconstruct_muscl(T,   limiter, face_begin, face_end, fs.T_L,   fs.T_R);
            break;

        case InviscidScheme::PPM:
            reconstruct_ppm_scalar(rho, face_begin, face_end, fs.rho_L, fs.rho_R);
            reconstruct_ppm_scalar(u,   face_begin, face_end, fs.u_L,   fs.u_R);
            reconstruct_ppm_scalar(p,   face_begin, face_end, fs.p_L,   fs.p_R);
            reconstruct_ppm_scalar(T,   face_begin, face_end, fs.T_L,   fs.T_R);
            break;

        default:
            throw std::runtime_error("reconstruct: unknown scheme.");
    }

    // Fill conservative face states from primitives (needed by Riemann solver)
    for (int f = face_begin; f < face_end; ++f) {
        // E = p/((gamma-1)*rho) + 0.5*u^2
        const double EL = fs.p_L[f] / ((gamma - 1.0) * fs.rho_L[f]) + 0.5 * fs.u_L[f] * fs.u_L[f];
        const double ER = fs.p_R[f] / ((gamma - 1.0) * fs.rho_R[f]) + 0.5 * fs.u_R[f] * fs.u_R[f];
        fs.rhou_L[f] = fs.rho_L[f] * fs.u_L[f];
        fs.rhou_R[f] = fs.rho_R[f] * fs.u_R[f];
        fs.rhoE_L[f] = fs.rho_L[f] * EL;
        fs.rhoE_R[f] = fs.rho_R[f] * ER;
    }
}

} // namespace splay
