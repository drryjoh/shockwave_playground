#include "splay/rk.hpp"
#include "splay/residual.hpp"
#include "splay/mpi_decomp.hpp"

namespace splay {

/// U_out[i] = alpha * a[i] + beta * (b[i] + dt * R[i])  for i in [ib, ie)
static void rk_combine(
    State& out, const State& a, const State& b, const Residual& Rb,
    double alpha, double beta, double dt,
    int ib, int ie)
{
    for (int i = ib; i < ie; ++i) {
        out.rho[i]  = alpha * a.rho[i]  + beta * (b.rho[i]  + dt * Rb.r_rho[i]);
        out.rhou[i] = alpha * a.rhou[i] + beta * (b.rhou[i] + dt * Rb.r_rhou[i]);
        out.rhoE[i] = alpha * a.rhoE[i] + beta * (b.rhoE[i] + dt * Rb.r_rhoE[i]);
    }
}

static void refresh(State& s, const Mesh& m, const GasModel& gas,
                    const BCState& bc_left, const BCState& bc_right,
                    const MPIDecomp& decomp)
{
    // Update primitives only for interior cells (ghost cells updated separately).
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    for (int i = ib; i < ie; ++i) s.update_primitives_cell(i, gas);

    apply_boundary_conditions(s, m, bc_left, bc_right, decomp,
                               gas.gamma, gas.cv, gas.R);
    decomp.fill_ghosts(s);
}

void ssprk3_step(
    State&                s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    const BCState&        bc_left,
    const BCState&        bc_right,
    const MPIDecomp&      decomp,
    double                dt)
{
    const int ib = m.interior_begin();
    const int ie = m.interior_end();

    // Pre-allocate stage buffers (no heap alloc inside loop at steady state).
    State    s1(m.n_total), s2(m.n_total);
    Residual R(m.n_total);

    // ─── Stage 1: U^(1) = U^n + dt * R(U^n) ─────────────────────────────────
    compute_residual(s, m, gas, tm, cfg, R);
    rk_combine(s1, s, s, R, 0.0, 1.0, dt, ib, ie);
    refresh(s1, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 2: U^(2) = (3/4)*U^n + (1/4)*(U^(1) + dt*R(U^(1))) ──────────
    compute_residual(s1, m, gas, tm, cfg, R);
    rk_combine(s2, s, s1, R, 0.75, 0.25, dt, ib, ie);
    refresh(s2, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 3: U^{n+1} = (1/3)*U^n + (2/3)*(U^(2) + dt*R(U^(2))) ─────────
    compute_residual(s2, m, gas, tm, cfg, R);
    rk_combine(s, s, s2, R, 1.0 / 3.0, 2.0 / 3.0, dt, ib, ie);
    refresh(s, m, gas, bc_left, bc_right, decomp);
}

void godunov_split_step(
    State&                s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    const BCState&        bc_left,
    const BCState&        bc_right,
    const MPIDecomp&      decomp,
    double                dt)
{
    const int ib = m.interior_begin();
    const int ie = m.interior_end();

    State    s1(m.n_total), s2(m.n_total);
    Residual R(m.n_total);

    // ── Sub-step 1: SSPRK3 with inviscid-only residual (flatten applied) ──────
    compute_residual(s, m, gas, tm, cfg, R, ResidualPart::InviscidOnly);
    rk_combine(s1, s, s, R, 0.0, 1.0, dt, ib, ie);
    refresh(s1, m, gas, bc_left, bc_right, decomp);

    compute_residual(s1, m, gas, tm, cfg, R, ResidualPart::InviscidOnly);
    rk_combine(s2, s, s1, R, 0.75, 0.25, dt, ib, ie);
    refresh(s2, m, gas, bc_left, bc_right, decomp);

    compute_residual(s2, m, gas, tm, cfg, R, ResidualPart::InviscidOnly);
    rk_combine(s, s, s2, R, 1.0 / 3.0, 2.0 / 3.0, dt, ib, ie);
    refresh(s, m, gas, bc_left, bc_right, decomp);

    // ── Sub-step 2: explicit Euler with viscous-only residual ─────────────────
    if (cfg.viscous_terms) {
        compute_residual(s, m, gas, tm, cfg, R, ResidualPart::ViscousOnly);
        for (int i = ib; i < ie; ++i) {
            s.rho[i]  += dt * R.r_rho[i];
            s.rhou[i] += dt * R.r_rhou[i];
            s.rhoE[i] += dt * R.r_rhoE[i];
        }
        refresh(s, m, gas, bc_left, bc_right, decomp);
    }
}

} // namespace splay
