#pragma once
#include <vector>
#include "splay/state.hpp"
#include "splay/config.hpp"

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

namespace splay {

struct Mesh;

/// Describes the MPI decomposition and provides ghost-cell exchange.
struct MPIDecomp {
    int rank   = 0;
    int nranks = 1;

    int left_neighbor  = -1;   // MPI rank to the left  (-1 = boundary)
    int right_neighbor = -1;   // MPI rank to the right (-1 = boundary)

    int n_ghost;               // ghost cells on each side

    /// Build decomposition.  Call after MPI_Init.
    static MPIDecomp build(int rank, int nranks, int n_ghost,
                           int n_local);

    /// Exchange nearest-neighbour ghost cells.
    /// Sends/receives n_ghost cells from each neighbour.
    void fill_ghosts(State& s) const;

    /// Reduce a local dt to the global minimum.
    double global_min_dt(double local_dt) const;

    /// Reduce a local double to global minimum.
    double global_min(double v) const;

    /// Reduce a local double to global maximum.
    double global_max(double v) const;

    /// Reduce a local L2 norm contribution to global L2 norm.
    double global_l2_norm(double local_sum_sq, int local_count) const;
};

/// Apply boundary conditions to ghost cells (no MPI needed; always local).
void apply_boundary_conditions(State& s, const Mesh& m,
                                const BCState& bc_left, const BCState& bc_right,
                                const MPIDecomp& decomp,
                                double gamma, double cv, double R_gas);

} // namespace splay
