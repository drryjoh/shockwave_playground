#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <cmath>

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

#include "splay/config.hpp"
#include "splay/gas.hpp"
#include "splay/mesh.hpp"
#include "splay/state.hpp"
#include "splay/transport.hpp"
#include "splay/residual.hpp"
#include "splay/rk.hpp"
#include "splay/mpi_decomp.hpp"
#include "splay/io.hpp"
#include "splay/diagnostics.hpp"
#include "splay/dg_state.hpp"
#include "splay/dg_av.hpp"
#include "splay/dg_residual.hpp"

namespace fs = std::filesystem;

// ── DG output helper: project GLL DOFs to cell-center averages ───────────────
// Uses GLL-weighted average (sum of w_j = 2, so divide by 2) to produce
// cell-average conserved variables compatible with write_csv / print_step_diag.
static void project_dg_to_fvm(const splay::DGState& dgs, const splay::GasModel& gas,
                               splay::State& fvm)
{
    const auto& w  = dgs.basis.w;
    const int   nd = dgs.n_dof;
    for (int i = 0; i < dgs.n_total; ++i) {
        double r = 0.0, ru = 0.0, rE = 0.0;
        for (int j = 0; j < nd; ++j) {
            r  += w[j] * dgs.rho [j][i];
            ru += w[j] * dgs.rhou[j][i];
            rE += w[j] * dgs.rhoE[j][i];
        }
        fvm.rho [i] = 0.5 * r;   // Σw = 2, so avg = sum/2
        fvm.rhou[i] = 0.5 * ru;
        fvm.rhoE[i] = 0.5 * rE;
        fvm.update_primitives_cell(i, gas);
    }
}

// ── DG tanh init: apply smooth transition profile at each GLL node's x ───────
static void init_tanh_dg(splay::DGState& dgs, const splay::Mesh& m,
                          const splay::GasModel& gas,
                          double p_L, double T_L, double u_L,
                          double p_R, double T_R, double u_R,
                          double x0, double delta)
{
    for (int i = 0; i < m.n_total; ++i) {
        for (int j = 0; j < dgs.n_dof; ++j) {
            const double xi_j = dgs.basis.xi[j];
            const double x    = m.x_cell[i] + 0.5 * xi_j * m.dx;
            const double phi  = 0.5 * (1.0 + std::tanh((x - x0) / delta));
            const double p    = p_L + phi * (p_R - p_L);
            const double T    = T_L + phi * (T_R - T_L);
            const double u    = u_L + phi * (u_R - u_L);
            const double rho  = p / (gas.R * T);
            dgs.rho [j][i] = rho;
            dgs.rhou[j][i] = rho * u;
            dgs.rhoE[j][i] = rho * gas.total_energy(T, u);
        }
    }
    dgs.update_primitives(0, m.n_total, gas);
}

int main(int argc, char** argv) {
    // ─── MPI init ─────────────────────────────────────────────────────────────
    int rank = 0, nranks = 1;
#ifdef SPLAY_ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

    try {
        // ─── Parse arguments ──────────────────────────────────────────────────
        if (argc < 2) {
            if (rank == 0)
                std::cerr << "Usage: splay <input.yml> [--restart <restart_dir>]\n";
#ifdef SPLAY_ENABLE_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        std::string yaml_path    = argv[1];
        std::string restart_path = "";
        for (int a = 2; a < argc - 1; ++a) {
            if (std::string(argv[a]) == "--restart")
                restart_path = argv[a + 1];
        }

        // ─── Configuration ────────────────────────────────────────────────────
        splay::Config cfg = splay::load_config(yaml_path, restart_path);

        // ─── Gas and transport models ─────────────────────────────────────────
        splay::GasModel       gas = splay::GasModel::from_name(cfg.gas);
        splay::TransportModel tm  = splay::TransportModel::from_name(cfg.gas);

        // ─── Mesh ─────────────────────────────────────────────────────────────
        splay::Mesh m = splay::Mesh::build(cfg.grid, rank, nranks);

        // ─── MPI decomposition ────────────────────────────────────────────────
        splay::MPIDecomp decomp = splay::MPIDecomp::build(rank, nranks,
                                                           m.n_ghost, m.n_local);

        // ─── Banner ───────────────────────────────────────────────────────────
        if (rank == 0) splay::print_banner(cfg, nranks);

        // Ensure output directories exist.
        if (rank == 0)
            fs::create_directories(cfg.diagnostics.output_dir + "/" + cfg.case_name);
#ifdef SPLAY_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        const double t_end         = cfg.solver.time_end;
        const int    log_step      = cfg.diagnostics.log_step;
        const int    snapshot_step = cfg.diagnostics.snapshot_step;
        double time = 0.0;
        int    step = 0;

        // ═════════════════════════════════════════════════════════════════════
        // DG path
        // ═════════════════════════════════════════════════════════════════════
        if (cfg.solver.spatial == splay::SpatialDiscretization::DG) {

            if (!cfg.restart_path.empty() && rank == 0)
                std::cerr << "[SPLAY] Warning: DG restart not supported; starting fresh.\n";

            const int p = cfg.solver.dg.poly_order;
            splay::DGState dgs(m.n_total, p);
            splay::State   fvm_proj(m.n_total);  // for output / diagnostics

            // ── IC ─────────────────────────────────────────────────────────
            if (cfg.init.type == "step") {
                splay::init_step_dg(dgs, m, gas,
                    cfg.init.rho_left,  cfg.init.p_left,  cfg.init.u_left,
                    cfg.init.rho_right, cfg.init.p_right, cfg.init.u_right,
                    cfg.init.location);
            } else if (cfg.init.type == "gaussian_perturbation") {
                splay::init_gaussian_perturbation_dg(dgs, m, gas,
                    cfg.init.p_left, cfg.init.T_left, cfg.init.u_left,
                    cfg.init.amplitude, cfg.init.location, cfg.init.sigma);
            } else {
                // tanh: resolve the smooth profile at each GLL node
                init_tanh_dg(dgs, m, gas,
                    cfg.init.p_left,  cfg.init.T_left,  cfg.init.u_left,
                    cfg.init.p_right, cfg.init.T_right, cfg.init.u_right,
                    cfg.init.location, cfg.init.thickness);
            }

            // ── Initial BCs and ghost exchange ─────────────────────────────
            splay::apply_boundary_conditions_dg(dgs, m, cfg.bc_left, cfg.bc_right, gas);
            splay::fill_ghosts_dg(dgs, m, decomp.rank, decomp.nranks,
                                   decomp.left_neighbor, decomp.right_neighbor);

            // ── Initial snapshot ───────────────────────────────────────────
            splay::write_csv_dg(dgs, m, tm, gas, decomp,
                                cfg.diagnostics.output_dir, cfg.case_name, step, time);

            // ── Time loop ──────────────────────────────────────────────────
            while (time < t_end) {
                std::string active_constraint;
                double dt_local = splay::compute_dt_dg(dgs, m, gas, tm, cfg.solver,
                                                        active_constraint);
                double dt = decomp.global_min_dt(dt_local);
                if (time + dt > t_end) dt = t_end - time;
                if (dt <= 0.0) break;

                splay::ssprk3_step_dg(dgs, m, gas, tm, cfg.solver,
                                      cfg.bc_left, cfg.bc_right, decomp, dt);
                time += dt;
                ++step;

                if (snapshot_step > 0 && step % snapshot_step == 0) {
                    splay::write_csv_dg(dgs, m, tm, gas, decomp,
                                        cfg.diagnostics.output_dir, cfg.case_name, step, time);
                }

                if (step % log_step == 0 || time >= t_end) {
                    if (rank == 0)
                        std::cout << "[DG] step=" << step
                                  << "  t=" << time
                                  << "  dt=" << dt
                                  << "  (" << active_constraint << ")\n";
                }
            }

            // ── Final output ───────────────────────────────────────────────
            splay::write_csv_dg(dgs, m, tm, gas, decomp,
                                cfg.diagnostics.output_dir, cfg.case_name, step, time);

            if (rank == 0)
                std::cout << "[SPLAY] Done (DG).  Final time=" << time
                          << " step=" << step << "\n";

        // ═════════════════════════════════════════════════════════════════════
        // FVM path (original)
        // ═════════════════════════════════════════════════════════════════════
        } else {

            splay::State s(m.n_total);

            if (!cfg.restart_path.empty()) {
                splay::read_restart(s, m, decomp, cfg.restart_path, time, step);
                if (rank == 0)
                    std::cout << "[SPLAY] Restarting from step=" << step
                              << " time=" << time << "\n";
            } else {
                if (cfg.init.type == "step") {
                    splay::init_step(s, m, gas,
                        cfg.init.rho_left,  cfg.init.p_left,  cfg.init.u_left,
                        cfg.init.rho_right, cfg.init.p_right, cfg.init.u_right,
                        cfg.init.location);
                } else if (cfg.init.type == "gaussian_perturbation") {
                    splay::init_gaussian_perturbation(s, m, gas,
                        cfg.init.p_left, cfg.init.T_left, cfg.init.u_left,
                        cfg.init.amplitude, cfg.init.location, cfg.init.sigma);
                } else {
                    splay::init_tanh(s, m, gas,
                        cfg.init.p_left,  cfg.init.T_left,  cfg.init.u_left,
                        cfg.init.p_right, cfg.init.T_right, cfg.init.u_right,
                        cfg.init.location, cfg.init.thickness);
                }
            }

            splay::apply_boundary_conditions(s, m, cfg.bc_left, cfg.bc_right,
                                              decomp, gas.gamma, gas.cv, gas.R);
            decomp.fill_ghosts(s);

            splay::write_csv(s, m, tm, gas, decomp,
                             cfg.diagnostics.output_dir, cfg.case_name, step, time);

            while (time < t_end) {
                std::string active_constraint;
                double dt_local = splay::compute_dt(s, m, gas, tm, cfg.solver,
                                                     active_constraint);
                double dt = decomp.global_min_dt(dt_local);
                if (time + dt > t_end) dt = t_end - time;
                if (dt <= 0.0) break;

                if (cfg.solver.operator_splitting)
                    splay::godunov_split_step(s, m, gas, tm, cfg.solver,
                                              cfg.bc_left, cfg.bc_right, decomp, dt);
                else
                    splay::ssprk3_step(s, m, gas, tm, cfg.solver,
                                       cfg.bc_left, cfg.bc_right, decomp, dt);

                time += dt;
                ++step;

                if (snapshot_step > 0 && step % snapshot_step == 0)
                    splay::write_csv(s, m, tm, gas, decomp,
                                     cfg.diagnostics.output_dir, cfg.case_name, step, time);

                if (step % log_step == 0 || time >= t_end) {
                    splay::Residual R(m.n_total);
                    splay::compute_residual(s, m, gas, tm, cfg.solver, R);
                    splay::print_step_diag(step, time, dt, active_constraint,
                                           s, m, gas, decomp,
                                           R.r_rho, R.r_rhou, R.r_rhoE);
                }
            }

            splay::write_csv(s, m, tm, gas, decomp,
                             cfg.diagnostics.output_dir, cfg.case_name, step, time);

            std::string restart_out = cfg.diagnostics.output_dir + "/"
                                    + cfg.case_name + "/restart/step_"
                                    + std::to_string(step);
            splay::write_restart(s, m, cfg, decomp, restart_out, step, time);

            if (rank == 0)
                std::cout << "[SPLAY] Done.  Final time=" << time
                          << " step=" << step << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[SPLAY][rank=" << rank << "] ERROR: " << e.what() << "\n";
#ifdef SPLAY_ENABLE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef SPLAY_ENABLE_MPI
    MPI_Finalize();
#endif
    return 0;
}
