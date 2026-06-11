#include "splay/dg_residual.hpp"
#include "splay/riemann.hpp"
#include "splay/transport.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace splay {

// ─────────────────────────────────────────────────────────────────────────────
// DG Residual Assembly  (weak / GLL-lumped-mass form)
// ─────────────────────────────────────────────────────────────────────────────
//
// For each interior cell i, DOF j:
//
//   (h/2)·w[j]·R_j =
//     + Σ_k w[k]·F_inv(U_k)·D[k][j]                 (inviscid volume)
//     - (F*_inv_R·δ(j,p) - F*_inv_L·δ(j,0))         (inviscid face)
//     - Σ_k w[k]·F_vis(U_k,dU_k)·D[k][j]            (viscous volume)
//     + (F*_vis_R·δ(j,p) - F*_vis_L·δ(j,0))         (viscous face)
//     - Σ_k w[k]·eps_k·(dQ_k)·D[k][j]               (AV volume)
//     + ({eps}·{dQ/dx})·(δ(j,p) - δ(j,0))           (AV face)
//
// => R_j = [2/(h·w[j])] * [above]
//
// GLL face DOFs — no interpolation at faces:
//   left  face of cell i: DOF 0  (xi = -1)
//   right face of cell i: DOF p  (xi = +1)
//
// ── Viscous face flux ─────────────────────────────────────────────────────────
//
//   SIPG: F*_vis = mu·{du/dx} - (eta/h)·mu·[u]   (pen = eta/h, user-supplied)
//
//   BR2 (Bassi-Rebay 2, Johnson & Kercher 2021):
//     Face flux:   F*_vis = mu·{du/dx}            (no face penalty)
//     Volume lift: + G(y+):(⟨y⟩−y+)⊗n · ∇v  (element-boundary integral)
//     where G(y+) = μ(y+) and the lift uses the interior state.
//     The lifting operator alone ensures energy stability without a face penalty.
//     Adding a large penalty (p+1)^2/h is anti-diffusive at strong shocks.
//
// ── AV face flux ──────────────────────────────────────────────────────────────
//   F*_AV = {eps}·{dQ/dx}   (no explicit penalty; eps provides stabilisation)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Inviscid Euler fluxes: F = [rho*u, rho*u^2+p, (rhoE+p)*u]
inline void euler_flux(double rho, double rhou, double rhoE, double pres,
                       double& F0, double& F1, double& F2) {
    const double u = rhou / rho;
    F0 = rhou;
    F1 = rhou * u + pres;
    F2 = (rhoE + pres) * u;
}

} // anonymous namespace

// ── compute_residual_dg ───────────────────────────────────────────────────────

void compute_residual_dg(
    const DGState&        s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    DGResidual&           R,
    ResidualPart          part)
{
    R.zero();

    const int    ib     = m.interior_begin();
    const int    ie     = m.interior_end();
    const int    n      = m.n_total;
    const int    nd     = s.n_dof;
    const int    p      = s.p;
    const double h      = m.dx;
    const double inv_h2 = 2.0 / h;   // reference-element Jacobian: d(xi)/dx

    const bool do_inv = (part != ResidualPart::ViscousOnly);
    const bool do_vis = (part != ResidualPart::InviscidOnly) && cfg.viscous_terms;
    const bool do_av  = (cfg.dg.av_method != AVMethod::None);

    const auto& D = s.basis.D;
    const auto& w = s.basis.w;

    // ── Transport at each DOF ─────────────────────────────────────────────────
    // mu_dof[j][i], kap_dof[j][i]
    std::vector<std::vector<double>> mu_dof (nd, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> kap_dof(nd, std::vector<double>(n, 0.0));
    if (do_vis) {
        for (int j = 0; j < nd; ++j)
            for (int i = 0; i < n; ++i) {
                mu_dof [j][i] = tm.viscosity   (s.T[j][i]);
                kap_dof[j][i] = tm.conductivity(s.T[j][i]);
            }
    }

    // ── Volume inviscid fluxes at each DOF ────────────────────────────────────
    // Fv_rho[j][i] = rho*u,  Fv_rhou[j][i] = rho*u^2+p,  Fv_rhoE[j][i] = (rhoE+p)*u
    std::vector<std::vector<double>> Fv_rho (nd, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Fv_rhou(nd, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Fv_rhoE(nd, std::vector<double>(n, 0.0));
    if (do_inv) {
        for (int j = 0; j < nd; ++j)
            for (int i = ib - 1; i <= ie; ++i)   // include one ghost layer
                euler_flux(s.rho[j][i], s.rhou[j][i], s.rhoE[j][i], s.prim_p[j][i],
                           Fv_rho[j][i], Fv_rhou[j][i], Fv_rhoE[j][i]);
    }

    // ── Viscous penalty coefficient ───────────────────────────────────────────
    // pen = penalty / h;  already has 1/h folded in for the face flux computation.
    double vis_pen = 0.0;
    if (do_vis) {
        if (cfg.dg.visc_scheme == ViscousScheme::SIPG)
            vis_pen = cfg.dg.sipg_penalty / h;
        else  // BR2: lifting operator provides stabilisation; no additional face penalty needed
            vis_pen = 0.0;
    }

    // ── Face loop: numerical fluxes for faces [ib-1 .. ie-1] ─────────────────
    // face f is the interface between cells f and f+1.
    const int face_begin = ib - 1;
    const int face_end   = ie;

    // Per-face arrays (indexed by left-cell index)
    std::vector<double> fi_r(n, 0.0), fi_ru(n, 0.0), fi_rE(n, 0.0);  // inviscid
    std::vector<double> fv_r(n, 0.0), fv_ru(n, 0.0), fv_rE(n, 0.0);  // viscous
    std::vector<double> fa_r(n, 0.0), fa_ru(n, 0.0), fa_rE(n, 0.0);  // AV
    // BR2 lifting: half-jumps [u]/2 and [T]/2 at each face.
    // The lifting term +G(y⁺):(⟨y⟩−y⁺)⊗n·∇v = μ_int·[u]/2·D[face][j]·inv_h2 (positive at both faces).
    std::vector<double> fjump_u(n, 0.0), fjump_T(n, 0.0);

    for (int f = face_begin; f < face_end; ++f) {
        const int iL = f;
        const int iR = f + 1;

        // Left state: right-face DOF of left cell (index p)
        const double rho_L  = s.face_rho_R (iL);
        const double rhou_L = s.face_rhou_R(iL);
        const double rhoE_L = s.face_rhoE_R(iL);
        const double u_L    = s.face_u_R   (iL);
        const double p_L    = s.face_p_R   (iL);

        // Right state: left-face DOF of right cell (index 0)
        const double rho_R  = s.face_rho_L (iR);
        const double rhou_R = s.face_rhou_L(iR);
        const double rhoE_R = s.face_rhoE_L(iR);
        const double u_R    = s.face_u_L   (iR);
        const double p_R    = s.face_p_L   (iR);

        // ── Inviscid: Riemann solver (reuses FVM solvers) ─────────────────────
        if (do_inv) {
            Flux3 fv = compute_inviscid_flux(
                rho_L, u_L, p_L, rhoE_L / rho_L,
                rho_R, u_R, p_R, rhoE_R / rho_R,
                gas.gamma, cfg.riemann_solver);
            fi_r [f] = fv[0];
            fi_ru[f] = fv[1];
            fi_rE[f] = fv[2];
        }

        // ── Viscous: SIPG or BR2 face flux ────────────────────────────────────
        // F*_vis = mu·{du/dx} - pen·mu·[u]
        // where {du/dx} = 0.5*(du/dx_L + du/dx_R)
        //       [u]     = u_R - u_L
        if (do_vis) {
            const double mu_avg  = 0.5 * (mu_dof [p][iL] + mu_dof [0][iR]);
            const double kap_avg = 0.5 * (kap_dof[p][iL] + kap_dof[0][iR]);

            // DG gradients at face: (2/h)*Σ_j U_j·D[face_node][j]
            // Left cell: evaluate at right-endpoint node p
            // Right cell: evaluate at left-endpoint node 0
            double du_L = 0.0, dT_L = 0.0;
            double du_R = 0.0, dT_R = 0.0;
            for (int j = 0; j < nd; ++j) {
                du_L += s.u[j][iL] * D[p][j];
                dT_L += s.T[j][iL] * D[p][j];
                du_R += s.u[j][iR] * D[0][j];
                dT_R += s.T[j][iR] * D[0][j];
            }
            du_L *= inv_h2;  dT_L *= inv_h2;
            du_R *= inv_h2;  dT_R *= inv_h2;

            const double avg_du = 0.5 * (du_L + du_R);
            const double avg_dT = 0.5 * (dT_L + dT_R);
            const double jmp_u  = u_R - u_L;
            const double jmp_T  = s.face_T_L(iR) - s.face_T_R(iL);

            // Shear stress and heat flux at face (F*_vis = [0, tau*, u_avg*tau* - q*])
            const double tau_star = mu_avg  * (avg_du - vis_pen * jmp_u);
            const double q_star   = kap_avg * (avg_dT - vis_pen * jmp_T);
            const double u_avg    = 0.5 * (u_L + u_R);

            fv_r [f] = 0.0;
            fv_ru[f] = tau_star;
            fv_rE[f] = u_avg * tau_star + q_star;

            // BR2 lifting: store half-jumps for use in the per-cell volume integral.
            fjump_u[f] = 0.5 * jmp_u;
            fjump_T[f] = 0.5 * jmp_T;
        }

        // ── AV face flux: {eps}·{dQ/dx} ───────────────────────────────────────
        if (do_av) {
            const double eps_avg = 0.5 * (s.epsilon[iL] + s.epsilon[iR]);
            if (eps_avg > 0.0) {
                double dr_L  = 0.0, dru_L = 0.0, drE_L = 0.0;
                double dr_R  = 0.0, dru_R = 0.0, drE_R = 0.0;
                for (int j = 0; j < nd; ++j) {
                    dr_L  += s.rho [j][iL] * D[p][j];
                    dru_L += s.rhou[j][iL] * D[p][j];
                    drE_L += s.rhoE[j][iL] * D[p][j];
                    dr_R  += s.rho [j][iR] * D[0][j];
                    dru_R += s.rhou[j][iR] * D[0][j];
                    drE_R += s.rhoE[j][iR] * D[0][j];
                }
                // {dQ/dx} = 0.5*inv_h2*(grad_L + grad_R); factor = eps_avg*inv_h2
                const double fac = eps_avg * inv_h2;
                fa_r [f] = fac * 0.5 * (dr_L  + dr_R);
                fa_ru[f] = fac * 0.5 * (dru_L + dru_R);
                fa_rE[f] = fac * 0.5 * (drE_L + drE_R);
            }
        }
    }

    // ── Cell loop: assemble DOF residuals ─────────────────────────────────────
    for (int i = ib; i < ie; ++i) {
        const int fL = i - 1;
        const int fR = i;

        // Local viscous volume fluxes: Fvis[k] = [0, tau_k, u_k*tau_k + q_k]
        double Fvis_ru[3] = {}, Fvis_rE[3] = {};
        if (do_vis) {
            for (int k = 0; k < nd; ++k) {
                double du_k = 0.0, dT_k = 0.0;
                for (int j = 0; j < nd; ++j) {
                    du_k += s.u[j][i] * D[k][j];
                    dT_k += s.T[j][i] * D[k][j];
                }
                du_k *= inv_h2;
                dT_k *= inv_h2;
                const double tau_k = mu_dof [k][i] * du_k;
                const double q_k   = kap_dof[k][i] * dT_k;
                Fvis_ru[k] = tau_k;
                Fvis_rE[k] = s.u[k][i] * tau_k + q_k;
            }
        }

        // Local AV volume fluxes: F_AV = eps_i · dQ/dx at each node
        double Fav_r[3]  = {}, Fav_ru[3] = {}, Fav_rE[3] = {};
        if (do_av && s.epsilon[i] > 0.0) {
            const double eps_inv_h2 = s.epsilon[i] * inv_h2;
            for (int k = 0; k < nd; ++k) {
                double dr = 0.0, dru = 0.0, drE = 0.0;
                for (int j = 0; j < nd; ++j) {
                    dr  += s.rho [j][i] * D[k][j];
                    dru += s.rhou[j][i] * D[k][j];
                    drE += s.rhoE[j][i] * D[k][j];
                }
                Fav_r [k] = eps_inv_h2 * dr;
                Fav_ru[k] = eps_inv_h2 * dru;
                Fav_rE[k] = eps_inv_h2 * drE;
            }
        }

        for (int j = 0; j < nd; ++j) {
            double Rr = 0.0, Rru = 0.0, RrE = 0.0;

            // ── Inviscid volume ───────────────────────────────────────────────
            if (do_inv) {
                for (int k = 0; k < nd; ++k) {
                    const double wDkj = w[k] * D[k][j];
                    Rr  += wDkj * Fv_rho [k][i];
                    Rru += wDkj * Fv_rhou[k][i];
                    RrE += wDkj * Fv_rhoE[k][i];
                }
                // Inviscid face: subtract right, add left (weak form sign)
                if (j == p) { Rr -= fi_r[fR];  Rru -= fi_ru[fR];  RrE -= fi_rE[fR]; }
                if (j == 0) { Rr += fi_r[fL];  Rru += fi_ru[fL];  RrE += fi_rE[fL]; }
            }

            // ── Viscous volume + face ─────────────────────────────────────────
            if (do_vis) {
                for (int k = 0; k < nd; ++k) {
                    const double wDkj = w[k] * D[k][j];
                    // viscous sign: subtract from volume (dFvis/dx on RHS = +; move to LHS = -)
                    Rru -= wDkj * Fvis_ru[k];
                    RrE -= wDkj * Fvis_rE[k];
                }
                // Viscous face: add right, subtract left (opposite to inviscid)
                if (j == p) { Rru += fv_ru[fR];  RrE += fv_rE[fR]; }
                if (j == 0) { Rru -= fv_ru[fL];  RrE -= fv_rE[fL]; }
                // BR2 lifting: η₀·G(y⁺):(⟨y⟩−y⁺)⊗n · ∇φ_j at each face boundary.
                // η₀ = n_faces + 1 = 3 (in 1D each element has 2 faces).
                // This matches the JENRE/Cockburn-Kanschat stability requirement η₀ ≥ n_faces.
                // Uses interior transport (mu/kap at the face node of this cell).
                constexpr double eta0 = 3.0;   // n_faces + 1 = 2 + 1
                const double br2_ru_R = eta0 * mu_dof [p][i] * fjump_u[fR];
                const double br2_rE_R = eta0 * (mu_dof [p][i] * s.u[p][i] * fjump_u[fR]
                                              + kap_dof[p][i] * fjump_T[fR]);
                const double br2_ru_L = eta0 * mu_dof [0][i] * fjump_u[fL];
                const double br2_rE_L = eta0 * (mu_dof [0][i] * s.u[0][i] * fjump_u[fL]
                                              + kap_dof[0][i] * fjump_T[fL]);
                Rru += br2_ru_R * D[p][j] * inv_h2 + br2_ru_L * D[0][j] * inv_h2;
                RrE += br2_rE_R * D[p][j] * inv_h2 + br2_rE_L * D[0][j] * inv_h2;
            }

            // ── AV volume + face ──────────────────────────────────────────────
            if (do_av) {
                for (int k = 0; k < nd; ++k) {
                    const double wDkj = w[k] * D[k][j];
                    Rr  -= wDkj * Fav_r [k];
                    Rru -= wDkj * Fav_ru[k];
                    RrE -= wDkj * Fav_rE[k];
                }
                if (j == p) { Rr += fa_r[fR];  Rru += fa_ru[fR];  RrE += fa_rE[fR]; }
                if (j == 0) { Rr -= fa_r[fL];  Rru -= fa_ru[fL];  RrE -= fa_rE[fL]; }
            }

            // ── Scale by inverse lumped mass: 2/(h·w[j]) ─────────────────────
            const double scale = inv_h2 / w[j];
            R.r_rho [j][i] = scale * Rr;
            R.r_rhou[j][i] = scale * Rru;
            R.r_rhoE[j][i] = scale * RrE;
        }
    }
}

// ── compute_dt_dg ─────────────────────────────────────────────────────────────
//
// DG CFL: dt <= h / ((2p+1) * lambda_max)
// Viscous parabolic: dt <= h^2 / ((2p+1)^2 * nu)

double compute_dt_dg(
    const DGState&        s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    std::string&          active_constraint)
{
    const int    ib      = m.interior_begin();
    const int    ie      = m.interior_end();
    const int    nd      = s.n_dof;
    const int    p       = s.p;
    const double h       = m.dx;
    const double dg_fac  = 1.0 / (2 * p + 1);

    double dt_conv  = std::numeric_limits<double>::max();
    double dt_visc  = std::numeric_limits<double>::max();
    double dt_therm = std::numeric_limits<double>::max();

    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < nd; ++j) {
            const double a   = gas.sound_speed(s.prim_p[j][i], s.rho[j][i]);
            const double lam = std::abs(s.u[j][i]) + a;
            if (lam > 1e-14)
                dt_conv = std::min(dt_conv, h / lam);

            if (cfg.viscous_terms && cfg.viscous_dt) {
                const double mu      = tm.viscosity   (s.T[j][i]);
                const double kap     = tm.conductivity(s.T[j][i]);
                const double nu      = mu  / s.rho[j][i];
                const double alpha_t = kap / (s.rho[j][i] * gas.cv);
                if (nu      > 1e-20) dt_visc  = std::min(dt_visc,  h * h / nu);
                if (alpha_t > 1e-20) dt_therm = std::min(dt_therm, h * h / alpha_t);
            }
        }
    }

    // Apply DG scaling: convective O(1/(2p+1)), parabolic O(1/(2p+1)^2)
    dt_conv  *= dg_fac;
    dt_visc  *= dg_fac * dg_fac;
    dt_therm *= dg_fac * dg_fac;

    double dt_local       = dt_conv;
    active_constraint     = "convective";

    if (cfg.viscous_terms) {
        if (dt_visc < dt_local)  { dt_local = dt_visc;  active_constraint = "viscous"; }
        if (dt_therm < dt_local) { dt_local = dt_therm; active_constraint = "thermal"; }
    }

    return cfg.cfl * dt_local;
}

} // namespace splay
