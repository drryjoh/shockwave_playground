#pragma once
#include <string>
#include "splay/config.hpp"
#include "splay/state.hpp"
#include "splay/mesh.hpp"
#include "splay/gas.hpp"
#include "splay/mpi_decomp.hpp"

namespace splay {

/// Print startup banner with all configuration choices.
void print_banner(const Config& cfg, int nranks);

/// Print per-step diagnostics.
void print_step_diag(int step, double time, double dt,
                     const std::string& active_constraint,
                     const State& s, const Mesh& m,
                     const GasModel& gas, const MPIDecomp& decomp,
                     const std::vector<double>& r_rho,
                     const std::vector<double>& r_rhou,
                     const std::vector<double>& r_rhoE);

} // namespace splay
