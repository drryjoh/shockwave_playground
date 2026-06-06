#include "splay/config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace splay {

// ─── Unit conversion helpers ─────────────────────────────────────────────────
static double to_si_length(double val, const std::string& units) {
    if (units == "m"  || units == "SI") return val;
    if (units == "cm")                  return val * 1e-2;
    if (units == "mm")                  return val * 1e-3;
    if (units == "um" || units == "micron") return val * 1e-6;
    throw std::runtime_error("Unknown length unit: " + units);
}

// ─── Enum parsers ────────────────────────────────────────────────────────────
static InviscidScheme parse_inviscid_scheme(const std::string& s) {
    if (s == "central") return InviscidScheme::Central;
    if (s == "muscl")   return InviscidScheme::MUSCL;
    if (s == "ppm")     return InviscidScheme::PPM;
    throw std::runtime_error("Unknown inviscid_scheme: " + s);
}

static RiemannSolver parse_riemann_solver(const std::string& s) {
    if (s == "none" || s == "central") return RiemannSolver::Central;
    if (s == "rusanov")                return RiemannSolver::Rusanov;
    if (s == "hllc")                   return RiemannSolver::HLLC;
    throw std::runtime_error("Unknown riemann_solver: " + s);
}

static Limiter parse_limiter(const std::string& s) {
    if (s == "none")    return Limiter::None;
    if (s == "minmod")  return Limiter::Minmod;
    if (s == "vanleer") return Limiter::VanLeer;
    if (s == "mc")      return Limiter::MC;
    if (s == "ppm")     return Limiter::PPM;
    throw std::runtime_error("Unknown limiter: " + s);
}

static BCType parse_bc_type(const std::string& s) {
    if (s == "inflow")  return BCType::Inflow;
    if (s == "outflow") return BCType::Outflow;
    throw std::runtime_error("Unknown boundary condition type: " + s);
}

// ─── YAML node helpers ───────────────────────────────────────────────────────
/// Try to get a nested node by dotted key like "domain.left" or nested map.
static YAML::Node get_node(const YAML::Node& root, const std::string& dotted_key) {
    // First try dotted key directly
    if (root[dotted_key]) return root[dotted_key];
    // Then try nested
    auto dot = dotted_key.find('.');
    if (dot == std::string::npos) return root[dotted_key];
    const std::string head = dotted_key.substr(0, dot);
    const std::string tail = dotted_key.substr(dot + 1);
    if (!root[head]) return YAML::Node(YAML::NodeType::Undefined);
    return get_node(root[head], tail);
}

template<typename T>
static T required(const YAML::Node& n, const std::string& key) {
    auto node = get_node(n, key);
    if (!node) throw std::runtime_error("Required key missing: " + key);
    return node.as<T>();
}

template<typename T>
static T optional(const YAML::Node& n, const std::string& key, T default_val) {
    auto node = get_node(n, key);
    if (!node) return default_val;
    return node.as<T>();
}

// ─── Main parser ─────────────────────────────────────────────────────────────
Config load_config(const std::string& yaml_path, const std::string& restart_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to open YAML file '" + yaml_path + "': " + e.what());
    }

    Config cfg;
    cfg.restart_path = restart_path;

    // Case name
    cfg.case_name = optional<std::string>(root, "case_name", "splay_run");

    // Gas
    cfg.gas = required<std::string>(root, "gas");
    if (cfg.gas != "argon" && cfg.gas != "nitrogen")
        throw std::runtime_error("gas must be 'argon' or 'nitrogen', got: " + cfg.gas);

    // ── Grid ──
    auto grid_node = root["grid"];
    if (!grid_node) throw std::runtime_error("Missing 'grid' section.");
    {
        std::string units = optional<std::string>(grid_node, "units", "m");
        cfg.grid.n_cells  = required<int>(grid_node, "n");
        cfg.grid.ghost_cells = optional<int>(grid_node, "ghost_cells", 3);

        if (grid_node["L"]) {
            double L = to_si_length(grid_node["L"].as<double>(), units);
            cfg.grid.x_min = optional<double>(grid_node, "x_min", 0.0);
            cfg.grid.x_max = cfg.grid.x_min + L;
        } else {
            cfg.grid.x_min = to_si_length(required<double>(grid_node, "x_min"), units);
            cfg.grid.x_max = to_si_length(required<double>(grid_node, "x_max"), units);
        }
        if (cfg.grid.x_max <= cfg.grid.x_min)
            throw std::runtime_error("grid: x_max must be > x_min.");
        if (cfg.grid.n_cells < 2)
            throw std::runtime_error("grid: n must be >= 2.");
    }

    // ── Solver ──
    auto solver_node = root["solver"];
    if (!solver_node) throw std::runtime_error("Missing 'solver' section.");
    {
        cfg.solver.cfl           = optional<double>(solver_node, "cfl", 0.5);
        cfg.solver.time_end      = required<double>(solver_node, "time_end");
        cfg.solver.rk_order      = optional<int>(solver_node, "rk", 3);
        cfg.solver.viscous_terms = optional<bool>(solver_node, "viscous_terms", true);
        cfg.solver.viscous_dt    = optional<bool>(solver_node, "viscous_dt",    true);

        std::string inv = optional<std::string>(solver_node, "inviscid_scheme", "central");
        std::string rs  = optional<std::string>(solver_node, "riemann_solver",  "none");
        std::string lim = optional<std::string>(solver_node, "limiter",          "none");

        cfg.solver.inviscid_scheme = parse_inviscid_scheme(inv);
        cfg.solver.riemann_solver  = parse_riemann_solver(rs);
        cfg.solver.limiter         = parse_limiter(lim);

        cfg.solver.flatten    = optional<bool>  (solver_node, "flatten",    false);
        cfg.solver.flatten_z1 = optional<double>(solver_node, "flatten_z1", 0.75);
        cfg.solver.flatten_z2 = optional<double>(solver_node, "flatten_z2", 0.85);
        if (cfg.solver.flatten_z1 >= cfg.solver.flatten_z2)
            throw std::runtime_error("solver.flatten_z1 must be < flatten_z2.");

        if (cfg.solver.cfl <= 0.0 || cfg.solver.cfl > 1.0)
            throw std::runtime_error("solver.cfl must be in (0, 1].");
        if (cfg.solver.time_end <= 0.0)
            throw std::runtime_error("solver.time_end must be > 0.");
        if (cfg.solver.rk_order != 3)
            std::cerr << "[SPLAY] Warning: only SSPRK3 (rk=3) is implemented; ignoring rk="
                      << cfg.solver.rk_order << "\n";
    }

    // ── Boundary conditions ──
    // Support both dotted keys and nested maps.
    auto bc_left_node  = get_node(root, "domain.left");
    auto bc_right_node = get_node(root, "domain.right");
    if (!bc_left_node)  throw std::runtime_error("Missing 'domain.left' section.");
    if (!bc_right_node) throw std::runtime_error("Missing 'domain.right' section.");
    {
        cfg.bc_left.type = parse_bc_type(required<std::string>(bc_left_node, "type"));
        if (cfg.bc_left.type == BCType::Inflow) {
            cfg.bc_left.pressure    = required<double>(bc_left_node, "pressure");
            cfg.bc_left.temperature = required<double>(bc_left_node, "temperature");
            cfg.bc_left.velocity    = required<double>(bc_left_node, "velocity");
        }
        cfg.bc_right.type = parse_bc_type(required<std::string>(bc_right_node, "type"));
        if (cfg.bc_right.type == BCType::Inflow) {
            cfg.bc_right.pressure    = required<double>(bc_right_node, "pressure");
            cfg.bc_right.temperature = required<double>(bc_right_node, "temperature");
            cfg.bc_right.velocity    = required<double>(bc_right_node, "velocity");
        }
    }

    // ── Initialization ──
    auto init_node = root["initialization"];
    if (!init_node) throw std::runtime_error("Missing 'initialization' section.");
    {
        std::string init_units = optional<std::string>(init_node, "units", "SI");
        cfg.init.type = optional<std::string>(init_node, "type", "tanh");

        if (cfg.init.type == "tanh") {
            cfg.init.location  = required<double>(init_node, "location");   // in init_units
            cfg.init.thickness = required<double>(init_node, "thickness");  // in init_units

            if (init_units != "SI" && init_units != "m") {
                cfg.init.location  = to_si_length(cfg.init.location,  init_units);
                cfg.init.thickness = to_si_length(cfg.init.thickness, init_units);
            }

            auto lft = init_node["left"];
            auto rgt = init_node["right"];
            if (!lft || !rgt)
                throw std::runtime_error("initialization: tanh requires 'left' and 'right' sub-blocks.");

            cfg.init.p_left  = required<double>(lft, "pressure");
            cfg.init.T_left  = required<double>(lft, "temperature");
            cfg.init.u_left  = required<double>(lft, "velocity");
            cfg.init.p_right = required<double>(rgt, "pressure");
            cfg.init.T_right = required<double>(rgt, "temperature");
            cfg.init.u_right = required<double>(rgt, "velocity");

        } else if (cfg.init.type == "gaussian_perturbation") {
            cfg.init.location  = required<double>(init_node, "location");  // centre x0
            cfg.init.sigma     = required<double>(init_node, "sigma");     // Gaussian half-width
            cfg.init.amplitude = optional<double>(init_node, "amplitude", 1e-4);

            if (init_units != "SI" && init_units != "m") {
                cfg.init.location = to_si_length(cfg.init.location, init_units);
                cfg.init.sigma    = to_si_length(cfg.init.sigma,    init_units);
            }

            auto lft = init_node["left"];
            if (!lft)
                throw std::runtime_error("initialization: gaussian_perturbation requires a 'left' sub-block for the base state.");

            cfg.init.p_left = required<double>(lft, "pressure");
            cfg.init.T_left = required<double>(lft, "temperature");
            cfg.init.u_left = required<double>(lft, "velocity");

        } else {
            throw std::runtime_error("initialization: unknown type '" + cfg.init.type +
                                     "'. Supported: tanh, gaussian_perturbation.");
        }
    }

    // ── Diagnostics ──
    auto diag_node = root["diagnostics"];
    {
        // Support dotted key "log.step" or nested "log: step:"
        int log_step = 100, snapshot_step = 0;
        if (diag_node) {
            auto ls_node = get_node(diag_node, "log.step");
            if (ls_node) log_step = ls_node.as<int>();
            auto ss_node = get_node(diag_node, "snapshot.step");
            if (ss_node) snapshot_step = ss_node.as<int>();
            cfg.diagnostics.units = optional<std::string>(diag_node, "units", "SI");
        }
        cfg.diagnostics.log_step      = log_step;
        cfg.diagnostics.snapshot_step = snapshot_step;
        cfg.diagnostics.output_dir    = optional<std::string>(root, "output_dir", "output");
    }

    return cfg;
}

} // namespace splay
