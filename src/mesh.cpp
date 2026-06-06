#include "splay/mesh.hpp"
#include <stdexcept>
#include <cmath>

namespace splay {

Mesh Mesh::build(const GridConfig& cfg, int rank, int nranks) {
    if (cfg.n_cells < 1)
        throw std::runtime_error("Mesh: n_cells must be >= 1.");
    if (nranks < 1)
        throw std::runtime_error("Mesh: nranks must be >= 1.");
    if (rank < 0 || rank >= nranks)
        throw std::runtime_error("Mesh: rank out of range.");

    Mesh m;
    m.n_global      = cfg.n_cells;
    m.n_ghost       = cfg.ghost_cells;
    m.x_min_global  = cfg.x_min;
    m.x_max_global  = cfg.x_max;
    m.dx            = (cfg.x_max - cfg.x_min) / static_cast<double>(cfg.n_cells);

    // Decompose global cells among ranks: distribute remainder to early ranks.
    const int base    = cfg.n_cells / nranks;
    const int leftover = cfg.n_cells % nranks;

    // global_start/end for each rank
    // Rank r owns cells [start_r, end_r)
    int start = 0;
    for (int r = 0; r < rank; ++r) {
        start += base + (r < leftover ? 1 : 0);
    }
    const int local_count = base + (rank < leftover ? 1 : 0);

    m.global_start = start;
    m.global_end   = start + local_count;
    m.n_local      = local_count;
    m.n_total      = m.n_local + 2 * m.n_ghost;

    // Build cell-centre coordinates for local+ghost cells.
    // Ghost cells extend the local block on each side.
    m.x_cell.resize(m.n_total);
    for (int i = 0; i < m.n_total; ++i) {
        // Global cell index: (global_start - n_ghost) + i
        const int global_i = (m.global_start - m.n_ghost) + i;
        m.x_cell[i] = m.x_min_global + (static_cast<double>(global_i) + 0.5) * m.dx;
    }

    return m;
}

} // namespace splay
