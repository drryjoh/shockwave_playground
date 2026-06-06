#include "splay/residual.hpp"
#include "splay/reconstruction.hpp"
#include "splay/riemann.hpp"
#include "splay/viscous_flux.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace splay {

Residual::Residual(int n_total) {
    r_rho.assign(n_total, 0.0);
    r_rhou.assign(n_total, 0.0);
    r_rhoE.assign(n_total, 0.0);
}

void Residual::zero() {
    std::fill(r_rho.begin(),  r_rho.end(),  0.0);
    std::fill(r_rhou.begin(), r_rhou.end(), 0.0);
    std::fill(r_rhoE.begin(), r_rhoE.end(), 0.0);
}

void compute_residual(
    const State&          s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    Residual&             R)
{
    R.zero();

    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    const int n  = m.n_total;
    const double dx = m.dx;

    // ── Transport coefficients ────────────────────────────────────────────────
    std::vector<double> mu(n, 0.0), kappa(n, 0.0);
    if (cfg.viscous_terms) {
        compute_transport(s.T, tm, mu, kappa);
    }

    // ── Face reconstruction ───────────────────────────────────────────────────
    // Faces: face f is between cell f and f+1.
    // We need faces from ib-1 to ie-1 (i.e., one face left of interior, one right).
    const int face_begin = ib - 1;
    const int face_end   = ie;         // exclusive; last interior face is ie-1

    FaceStates fs(n);   // allocated for full n_total; only [face_begin..face_end) used.

    reconstruct(s.rho, s.u, s.p, s.T,
                cfg.inviscid_scheme, cfg.limiter, gas.gamma,
                face_begin, face_end, fs);

    // ── Inviscid flux assembly ────────────────────────────────────────────────
    std::vector<Flux3> inv_flux(n);
    for (int f = face_begin; f < face_end; ++f) {
        inv_flux[f] = compute_inviscid_flux(
            fs.rho_L[f], fs.u_L[f], fs.p_L[f], fs.rhoE_L[f] / fs.rho_L[f],
            fs.rho_R[f], fs.u_R[f], fs.p_R[f], fs.rhoE_R[f] / fs.rho_R[f],
            gas.gamma, cfg.riemann_solver);
    }

    // ── Viscous flux assembly ─────────────────────────────────────────────────
    std::vector<Flux3> visc_flux(n, {0.0, 0.0, 0.0});
    if (cfg.viscous_terms) {
        compute_all_viscous_fluxes(mu, kappa, s.u, s.T, dx,
                                   face_begin, face_end, visc_flux);
    }

    // ── Residual: R = -(F_{i+1/2} - F_{i-1/2}) / dx ─────────────────────────
    // Positive R means the cell gains conserved quantity (dU/dt = R).
    for (int i = ib; i < ie; ++i) {
        const int fR = i;       // right face of cell i  (between i and i+1)
        const int fL = i - 1;   // left  face of cell i  (between i-1 and i)

        R.r_rho[i]  = -(inv_flux[fR][0] - inv_flux[fL][0]) / dx
                      + (visc_flux[fR][0] - visc_flux[fL][0]) / dx;
        R.r_rhou[i] = -(inv_flux[fR][1] - inv_flux[fL][1]) / dx
                      + (visc_flux[fR][1] - visc_flux[fL][1]) / dx;
        R.r_rhoE[i] = -(inv_flux[fR][2] - inv_flux[fL][2]) / dx
                      + (visc_flux[fR][2] - visc_flux[fL][2]) / dx;
    }
}

double compute_dt(
    const State&          s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    std::string&          active_constraint)
{
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    const double dx = m.dx;

    double dt_conv  = std::numeric_limits<double>::max();
    double dt_visc  = std::numeric_limits<double>::max();
    double dt_therm = std::numeric_limits<double>::max();

    for (int i = ib; i < ie; ++i) {
        const double a   = gas.sound_speed(s.p[i], s.rho[i]);
        const double lam = std::abs(s.u[i]) + a;
        if (lam > 1e-14)
            dt_conv = std::min(dt_conv, dx / lam);

        if (cfg.viscous_terms) {
            const double mu    = tm.viscosity(s.T[i]);
            const double kappa = tm.conductivity(s.T[i]);
            const double nu    = mu / s.rho[i];
            if (nu > 1e-20)
                dt_visc = std::min(dt_visc, dx * dx / nu);
            const double thermal_diff = kappa / (s.rho[i] * gas.cv);
            if (thermal_diff > 1e-20)
                dt_therm = std::min(dt_therm, dx * dx / thermal_diff);
        }
    }

    double dt_local = dt_conv;
    active_constraint = "convective";

    if (cfg.viscous_terms) {
        if (dt_visc < dt_local) {
            dt_local = dt_visc;
            active_constraint = "viscous";
        }
        if (dt_therm < dt_local) {
            dt_local = dt_therm;
            active_constraint = "thermal";
        }
    }

    return cfg.cfl * dt_local;
}

} // namespace splay
