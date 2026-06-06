#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>

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

namespace fs = std::filesystem;

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
            if (std::string(argv[a]) == "--restart") {
                restart_path = argv[a + 1];
            }
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

        // ─── State ────────────────────────────────────────────────────────────
        splay::State s(m.n_total);

        double time = 0.0;
        int    step = 0;

        if (!cfg.restart_path.empty()) {
            splay::read_restart(s, m, decomp, cfg.restart_path, time, step);
            if (rank == 0)
                std::cout << "[SPLAY] Restarting from step=" << step
                          << " time=" << time << "\n";
        } else {
            if (cfg.init.type == "gaussian_perturbation") {
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

        // Apply BCs and fill ghosts before first step.
        splay::apply_boundary_conditions(s, m, cfg.bc_left, cfg.bc_right,
                                          decomp, gas.gamma, gas.cv, gas.R);
        decomp.fill_ghosts(s);

        // ─── Banner ───────────────────────────────────────────────────────────
        if (rank == 0) splay::print_banner(cfg, nranks);

        // Ensure output directories exist.
        if (rank == 0) {
            fs::create_directories(cfg.diagnostics.output_dir + "/" + cfg.case_name);
        }
#ifdef SPLAY_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Write initial snapshot.
        splay::write_csv(s, m, tm, gas, decomp,
                         cfg.diagnostics.output_dir, cfg.case_name, step, time);

        // ─── Time loop ────────────────────────────────────────────────────────
        const double t_end         = cfg.solver.time_end;
        const int    log_step      = cfg.diagnostics.log_step;
        const int    snapshot_step = cfg.diagnostics.snapshot_step;

        while (time < t_end) {
            // Compute stable dt.
            std::string active_constraint;
            double dt_local = splay::compute_dt(s, m, gas, tm, cfg.solver, active_constraint);
            double dt = decomp.global_min_dt(dt_local);

            // Clamp to not overshoot final time.
            if (time + dt > t_end) dt = t_end - time;
            if (dt <= 0.0) break;

            // RK advance.
            if (cfg.solver.operator_splitting)
                splay::godunov_split_step(s, m, gas, tm, cfg.solver,
                                          cfg.bc_left, cfg.bc_right, decomp, dt);
            else
                splay::ssprk3_step(s, m, gas, tm, cfg.solver,
                                   cfg.bc_left, cfg.bc_right, decomp, dt);

            time += dt;
            ++step;

            // Periodic snapshots.
            if (snapshot_step > 0 && step % snapshot_step == 0) {
                splay::write_csv(s, m, tm, gas, decomp,
                                 cfg.diagnostics.output_dir, cfg.case_name, step, time);
            }

            // Diagnostics.
            if (step % log_step == 0 || time >= t_end) {
                splay::Residual R(m.n_total);
                splay::compute_residual(s, m, gas, tm, cfg.solver, R);
                splay::print_step_diag(step, time, dt, active_constraint,
                                       s, m, gas, decomp,
                                       R.r_rho, R.r_rhou, R.r_rhoE);
            }
        }

        // ─── Final output ─────────────────────────────────────────────────────
        splay::write_csv(s, m, tm, gas, decomp,
                         cfg.diagnostics.output_dir, cfg.case_name, step, time);

        // Write restart.
        std::string restart_out = cfg.diagnostics.output_dir + "/"
                                + cfg.case_name + "/restart/step_" + std::to_string(step);
        splay::write_restart(s, m, cfg, decomp, restart_out, step, time);

        if (rank == 0)
            std::cout << "[SPLAY] Done.  Final time=" << time << " step=" << step << "\n";

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
