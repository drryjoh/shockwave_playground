#include "splay/rk.hpp"
#include "splay/residual.hpp"
#include "splay/mpi_decomp.hpp"
#include "splay/dg_av.hpp"
#include "splay/dg_residual.hpp"

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
    compute_residual(s, m, gas, tm, cfg, R, ResidualPart::Full, dt);
    rk_combine(s1, s, s, R, 0.0, 1.0, dt, ib, ie);
    refresh(s1, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 2: U^(2) = (3/4)*U^n + (1/4)*(U^(1) + dt*R(U^(1))) ──────────
    compute_residual(s1, m, gas, tm, cfg, R, ResidualPart::Full, dt);
    rk_combine(s2, s, s1, R, 0.75, 0.25, dt, ib, ie);
    refresh(s2, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 3: U^{n+1} = (1/3)*U^n + (2/3)*(U^(2) + dt*R(U^(2))) ─────────
    compute_residual(s2, m, gas, tm, cfg, R, ResidualPart::Full, dt);
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
    compute_residual(s, m, gas, tm, cfg, R, ResidualPart::InviscidOnly, dt);
    rk_combine(s1, s, s, R, 0.0, 1.0, dt, ib, ie);
    refresh(s1, m, gas, bc_left, bc_right, decomp);

    compute_residual(s1, m, gas, tm, cfg, R, ResidualPart::InviscidOnly, dt);
    rk_combine(s2, s, s1, R, 0.75, 0.25, dt, ib, ie);
    refresh(s2, m, gas, bc_left, bc_right, decomp);

    compute_residual(s2, m, gas, tm, cfg, R, ResidualPart::InviscidOnly, dt);
    rk_combine(s, s, s2, R, 1.0 / 3.0, 2.0 / 3.0, dt, ib, ie);
    refresh(s, m, gas, bc_left, bc_right, decomp);

    // ── Sub-step 2: explicit Euler with viscous-only residual ─────────────────
    if (cfg.viscous_terms) {
        compute_residual(s, m, gas, tm, cfg, R, ResidualPart::ViscousOnly, dt);
        for (int i = ib; i < ie; ++i) {
            s.rho[i]  += dt * R.r_rho[i];
            s.rhou[i] += dt * R.r_rhou[i];
            s.rhoE[i] += dt * R.r_rhoE[i];
        }
        refresh(s, m, gas, bc_left, bc_right, decomp);
    }
}

// ─── DG SSPRK3 ───────────────────────────────────────────────────────────────

/// U_out[j][i] = alpha*a[j][i] + beta*(b[j][i] + dt*R[j][i]) for i in [ib,ie)
static void dg_rk_combine(
    DGState& out, const DGState& a, const DGState& b, const DGResidual& Rb,
    double alpha, double beta, double dt,
    int ib, int ie)
{
    const int nd = out.n_dof;
    for (int i = ib; i < ie; ++i) {
        for (int j = 0; j < nd; ++j) {
            out.rho [j][i] = alpha * a.rho [j][i] + beta * (b.rho [j][i] + dt * Rb.r_rho [j][i]);
            out.rhou[j][i] = alpha * a.rhou[j][i] + beta * (b.rhou[j][i] + dt * Rb.r_rhou[j][i]);
            out.rhoE[j][i] = alpha * a.rhoE[j][i] + beta * (b.rhoE[j][i] + dt * Rb.r_rhoE[j][i]);
        }
    }
}

/// Update primitives, apply BCs, exchange MPI ghosts, re-update ghost primitives.
// ── Positivity-preserving limiter ─────────────────────────────────────────────
// For each cell: if any DOF has ρ < ε_rho or T < T_min, rescale all
// conservative DOFs linearly toward the cell GLL-quadrature average so the
// minimum value is just above the floor.  Preserves the cell average exactly.
// Applied to conservative variables (ρ, ρu, ρE) before update_primitives.
static constexpr double PPL_RHO_MIN = 1e-4;   // kg/m³
static constexpr double PPL_T_MIN   = 1.0;    // K  (used via p = ρRT/M)

static void apply_positivity_limiter_dg(DGState& s, const Mesh& m,
                                        const GasModel& gas)
{
    const int    ib   = m.interior_begin();
    const int    ie   = m.interior_end();
    const int    nd   = s.n_dof;
    const auto&  w    = s.basis.w;
    const double w_sum = 2.0;   // GLL weights always sum to 2 on [-1,1]

    for (int i = ib; i < ie; ++i) {
        // ── Cell-average conservative variables (GLL quadrature) ─────────────
        double rho_avg = 0.0, rhou_avg = 0.0, rhoE_avg = 0.0;
        for (int j = 0; j < nd; ++j) {
            rho_avg  += w[j] * s.rho [j][i];
            rhou_avg += w[j] * s.rhou[j][i];
            rhoE_avg += w[j] * s.rhoE[j][i];
        }
        rho_avg  /= w_sum;
        rhou_avg /= w_sum;
        rhoE_avg /= w_sum;

        // ── Density limiter ───────────────────────────────────────────────────
        double rho_min = s.rho[0][i];
        for (int j = 1; j < nd; ++j) rho_min = std::min(rho_min, s.rho[j][i]);

        if (rho_min < PPL_RHO_MIN && rho_avg > PPL_RHO_MIN) {
            const double sf = std::min(1.0, 0.9999 * (rho_avg - PPL_RHO_MIN)
                                                   / (rho_avg - rho_min));
            for (int j = 0; j < nd; ++j) {
                s.rho [j][i] = rho_avg  + sf * (s.rho [j][i] - rho_avg);
                s.rhou[j][i] = rhou_avg + sf * (s.rhou[j][i] - rhou_avg);
                s.rhoE[j][i] = rhoE_avg + sf * (s.rhoE[j][i] - rhoE_avg);
            }
            // Re-read rho_avg after scaling (unchanged, but rhou/rhoE now safe)
        }

        // ── Pressure/temperature limiter ──────────────────────────────────────
        // Estimate internal energy e_j = E_j - 0.5*u_j^2 for each DOF.
        // Use p_min_physical from gas: p = (γ-1)*ρ*e = ρ*R/M * T
        // Simplified: check e_j > 0 (internal energy positivity).
        double e_min = 1e300;
        for (int j = 0; j < nd; ++j) {
            const double rho_j = s.rho [j][i];
            if (rho_j <= 0.0) { e_min = -1.0; break; }
            const double u_j = s.rhou[j][i] / rho_j;
            const double e_j = s.rhoE[j][i] / rho_j - 0.5 * u_j * u_j;
            e_min = std::min(e_min, e_j);
        }

        const double e_avg = (rho_avg > 0.0)
            ? (rhoE_avg / rho_avg - 0.5 * (rhou_avg / rho_avg) * (rhou_avg / rho_avg))
            : 0.0;
        const double e_floor = gas.total_energy(PPL_T_MIN, 0.0);  // e at T_min, u=0

        if (e_min < e_floor && e_avg > e_floor) {
            const double sf = std::min(1.0, 0.9999 * (e_avg - e_floor)
                                                   / (e_avg - e_min));
            for (int j = 0; j < nd; ++j) {
                s.rho [j][i] = rho_avg  + sf * (s.rho [j][i] - rho_avg);
                s.rhou[j][i] = rhou_avg + sf * (s.rhou[j][i] - rhou_avg);
                s.rhoE[j][i] = rhoE_avg + sf * (s.rhoE[j][i] - rhoE_avg);
            }
        }
    }
}

static void dg_refresh(DGState& s, const Mesh& m, const GasModel& gas,
                       const BCState& bc_left, const BCState& bc_right,
                       const MPIDecomp& decomp)
{
    s.update_primitives(m.interior_begin(), m.interior_end(), gas);
    // apply_boundary_conditions_dg fills ghost DOFs and calls update_primitives
    // over the full domain, so ghost primitives are set for the physical BC case.
    apply_boundary_conditions_dg(s, m, bc_left, bc_right, gas);
    // MPI exchange overwrites ghost DOFs for interior-to-interior boundaries.
    fill_ghosts_dg(s, m, decomp.rank, decomp.nranks,
                   decomp.left_neighbor, decomp.right_neighbor);
    // Re-derive primitives for MPI-filled ghost cells (no-op for nranks==1).
    s.update_primitives(0, m.interior_begin(), gas);
    s.update_primitives(m.interior_end(), m.n_total, gas);
}

void ssprk3_step_dg(
    DGState&              s,
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

    DGState    s1(m.n_total, s.p), s2(m.n_total, s.p);
    DGResidual R(m.n_total, s.n_dof);

    const bool ppl = cfg.dg.positivity_limiter;

    // ─── Stage 1: U^(1) = U^n + dt * R(U^n) ─────────────────────────────────
    compute_av_dg(s, m, gas, cfg.dg);
    compute_residual_dg(s, m, gas, tm, cfg, R);
    dg_rk_combine(s1, s, s, R, 0.0, 1.0, dt, ib, ie);
    if (ppl) apply_positivity_limiter_dg(s1, m, gas);
    dg_refresh(s1, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 2: U^(2) = (3/4)U^n + (1/4)(U^(1) + dt*R(U^(1))) ─────────────
    compute_av_dg(s1, m, gas, cfg.dg);
    compute_residual_dg(s1, m, gas, tm, cfg, R);
    dg_rk_combine(s2, s, s1, R, 0.75, 0.25, dt, ib, ie);
    if (ppl) apply_positivity_limiter_dg(s2, m, gas);
    dg_refresh(s2, m, gas, bc_left, bc_right, decomp);

    // ─── Stage 3: U^{n+1} = (1/3)U^n + (2/3)(U^(2) + dt*R(U^(2))) ───────────
    compute_av_dg(s2, m, gas, cfg.dg);
    compute_residual_dg(s2, m, gas, tm, cfg, R);
    dg_rk_combine(s, s, s2, R, 1.0 / 3.0, 2.0 / 3.0, dt, ib, ie);
    if (ppl) apply_positivity_limiter_dg(s, m, gas);
    dg_refresh(s, m, gas, bc_left, bc_right, decomp);
}

} // namespace splay
