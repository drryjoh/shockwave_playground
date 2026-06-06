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
    double      location  = 0.0;   // m (shock centre / Gaussian centre)
    double      thickness = 1e-8;  // m (tanh half-width delta)
    // Left primitive state (post-shock for tanh; base state for gaussian_perturbation)
    double p_left  = 0.0;
    double T_left  = 0.0;
    double u_left  = 0.0;
    // Right primitive state (pre-shock for tanh; unused for gaussian_perturbation)
    double p_right = 0.0;
    double T_right = 0.0;
    double u_right = 0.0;
    // Gaussian perturbation parameters
    double amplitude = 1e-4;  // fractional amplitude of pressure perturbation
    double sigma     = 1e-2;  // m (Gaussian half-width)
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
    double         cfl              = 0.5;
    double         time_end         = 1.0;
    int            rk_order         = 3;
    bool           viscous_terms    = true;
    // When true (default), the viscous parabolic stability restriction
    // dt <= dx^2/nu is included in the timestep calculation.
    // Set to false to use only the convective CFL even when viscous_terms=true.
    // NOTE: disabling viscous_dt may cause instability if dt >> dx^2/nu.
    bool           viscous_dt       = true;
    InviscidScheme inviscid_scheme   = InviscidScheme::Central;
    RiemannSolver  riemann_solver    = RiemannSolver::Central;
    Limiter        limiter           = Limiter::None;
    // CW84/PeleC-style shock flattening: blends face states toward cell averages
    // near strong shocks (detected by pressure-ratio criterion + converging flow).
    bool   flatten    = false;  // enable flattening
    double flatten_z1 = 0.75;   // lower pressure-ratio threshold (start of ramp)
    double flatten_z2 = 0.85;   // upper pressure-ratio threshold (fully flattened)
    // Godunov operator splitting: advance inviscid+flatten first (SSPRK3), then
    // viscous diffusion separately (explicit Euler).  When false (default) both
    // operators are combined in the same RK residual each stage.
    bool   operator_splitting = false;
};

// ─── Diagnostics ──────────────────────────────────────────────────────────────
struct DiagConfig {
    int         log_step      = 100;  // print to stdout every N steps
    int         snapshot_step = 0;    // write CSV snapshot every N steps (0 = only initial+final)
    std::string units         = "SI";
    std::string output_dir    = "output";
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
