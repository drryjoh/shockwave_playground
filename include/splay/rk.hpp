#pragma once
#include "splay/state.hpp"
#include "splay/residual.hpp"
#include "splay/mesh.hpp"
#include "splay/gas.hpp"
#include "splay/config.hpp"
#include "splay/transport.hpp"
#include "splay/mpi_decomp.hpp"

namespace splay {

/// Advance one time step using SSPRK3 (Gottlieb-Shu 1998).
///
/// Stages:
///   U^(1) = U^n + dt * R(U^n)
///   U^(2) = (3/4) U^n + (1/4)[ U^(1) + dt * R(U^(1)) ]
///   U^n+1 = (1/3) U^n + (2/3)[ U^(2) + dt * R(U^(2)) ]
///
/// Ghost cells are filled after each stage via fill_ghosts() and apply_boundary_conditions().
void ssprk3_step(
    State&                s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    const BCState&        bc_left,
    const BCState&        bc_right,
    const MPIDecomp&      decomp,
    double                dt);

/// Advance one time step using Godunov operator splitting:
///   1. SSPRK3 with inviscid-only residual (flattening applied here).
///   2. Explicit Euler with viscous-only residual.
///
/// This decouples the inviscid (flatten) and viscous operators so that
/// flattening captures the shock before the viscous step sees the profile.
/// Useful for reproducing the PeleC-like non-convergence behaviour.
void godunov_split_step(
    State&                s,
    const Mesh&           m,
    const GasModel&       gas,
    const TransportModel& tm,
    const SolverConfig&   cfg,
    const BCState&        bc_left,
    const BCState&        bc_right,
    const MPIDecomp&      decomp,
    double                dt);

} // namespace splay
