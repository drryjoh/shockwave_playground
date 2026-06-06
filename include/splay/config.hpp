#pragma once
#include <string>

namespace splay {

// ─── Inviscid scheme ──────────────────────────────────────────────────────────
enum class InviscidScheme { Central, MUSCL, PPM };

// ─── Riemann solver ───────────────────────────────────────────────────────────
enum class RiemannSolver { Central, Rusanov, HLLC };

// ─── Slope limiter ────────────────────────────────────────────────────────────
enum class Limiter { None, Minmod, VanLeer, MC, PPM };

// ─── Boundary condition type ──────────────────────────────────────────────────
enum class BCType { Inflow, Outflow };

// ─── Boundary condition state ─────────────────────────────────────────────────
struct BCState {
    BCType   type        = BCType::Outflow;
    double   pressure    = 0.0;   // Pa
    double   temperature = 0.0;   // K
    double   velocity    = 0.0;   // m/s
};

// ─── Initialization ───────────────────────────────────────────────────────────
struct InitConfig {
    std::string type      = "tanh";
    double      location  = 0.0;   // m (shock centre)
    double      thickness = 1e-8;  // m (tanh half-width delta)
    // Left (post-shock) primitive state
    double p_left  = 0.0;
    double T_left  = 0.0;
    double u_left  = 0.0;
    // Right (pre-shock) primitive state
    double p_right = 0.0;
    double T_right = 0.0;
    double u_right = 0.0;
};

// ─── Grid ─────────────────────────────────────────────────────────────────────
struct GridConfig {
    double x_min      = 0.0;
    double x_max      = 1.0;
    int    n_cells    = 100;
    int    ghost_cells = 3;   // cells on each side
};

// ─── Solver ───────────────────────────────────────────────────────────────────
struct SolverConfig {
    double         cfl            = 0.5;
    double         time_end       = 1.0;
    int            rk_order       = 3;
    bool           viscous_terms  = true;
    InviscidScheme inviscid_scheme = InviscidScheme::Central;
    RiemannSolver  riemann_solver  = RiemannSolver::Central;
    Limiter        limiter         = Limiter::None;
};

// ─── Diagnostics ──────────────────────────────────────────────────────────────
struct DiagConfig {
    int         log_step   = 100;
    std::string units      = "SI";
    std::string output_dir = "output";
};

// ─── Top-level configuration ─────────────────────────────────────────────────
struct Config {
    std::string  case_name = "splay_run";
    std::string  gas       = "argon";

    GridConfig   grid;
    SolverConfig solver;
    DiagConfig   diagnostics;
    BCState      bc_left;
    BCState      bc_right;
    InitConfig   init;

    // Optional restart path (empty = fresh start)
    std::string restart_path = "";
};

/// Parse configuration from a YAML file.
Config load_config(const std::string& yaml_path, const std::string& restart_path = "");

} // namespace splay
