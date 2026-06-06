#pragma once
#include <vector>
#include "splay/config.hpp"

namespace splay {

/// 1D uniform finite-volume mesh.
///
/// Indexing (local rank view):
///   [0 .. n_ghost-1]              left ghost cells
///   [n_ghost .. n_ghost+n_local-1] interior cells owned by this rank
///   [n_ghost+n_local .. n_total-1] right ghost cells
///
/// x_cell[i] is the cell-centre coordinate in SI (metres).
struct Mesh {
    int    n_global;      ///< total interior cells across all ranks
    int    n_local;       ///< interior cells owned by this rank
    int    n_ghost;       ///< ghost cells on each side
    int    n_total;       ///< n_local + 2*n_ghost  (allocated per rank)

    double x_min_global;  ///< global domain start (m)
    double x_max_global;  ///< global domain end   (m)
    double dx;            ///< uniform cell width (m)

    int    global_start;  ///< index of first owned interior cell in global array
    int    global_end;    ///< one-past last owned interior cell

    std::vector<double> x_cell; ///< cell-centre x[0..n_total-1] in SI

    /// Build mesh from config + MPI rank information.
    static Mesh build(const GridConfig& cfg, int rank, int nranks);

    /// Index helpers
    int interior_begin() const { return n_ghost; }
    int interior_end()   const { return n_ghost + n_local; }
};

} // namespace splay
