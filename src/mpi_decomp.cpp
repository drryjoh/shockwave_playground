#include "splay/mpi_decomp.hpp"
#include "splay/mesh.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

namespace splay {

MPIDecomp MPIDecomp::build(int rank, int nranks, int n_ghost, int n_local) {
    MPIDecomp d;
    d.rank    = rank;
    d.nranks  = nranks;
    d.n_ghost = n_ghost;
    d.left_neighbor  = (rank > 0)          ? rank - 1 : -1;
    d.right_neighbor = (rank < nranks - 1) ? rank + 1 : -1;
    return d;
}

void MPIDecomp::fill_ghosts(State& s) const {
#ifdef SPLAY_ENABLE_MPI
    const int ng = n_ghost;
    const int nl = static_cast<int>(s.rho.size()) - 2 * ng;  // n_local

    // Buffer sizes
    const int buf = ng;

    auto exchange = [&](std::vector<double>& v) {
        // Send right ghosts to right neighbour; receive into left ghost.
        // Send left ghosts to left neighbour; receive into right ghost.
        MPI_Status st;

        // -- send rightward (my rightmost interior cells -> right neighbour's left ghost)
        if (right_neighbor >= 0) {
            MPI_Send(v.data() + ng + nl - ng, ng, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD);
        }
        if (left_neighbor >= 0) {
            MPI_Recv(v.data(), ng, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, &st);
        }

        // -- send leftward (my leftmost interior cells -> left neighbour's right ghost)
        if (left_neighbor >= 0) {
            MPI_Send(v.data() + ng, ng, MPI_DOUBLE, left_neighbor, 1, MPI_COMM_WORLD);
        }
        if (right_neighbor >= 0) {
            MPI_Recv(v.data() + ng + nl, ng, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &st);
        }
    };

    exchange(s.rho);
    exchange(s.rhou);
    exchange(s.rhoE);
    exchange(s.u);
    exchange(s.p);
    exchange(s.T);
#else
    // Serial: no MPI exchange needed. Ghost cells are handled by BCs only.
    (void)s;
#endif
}

double MPIDecomp::global_min_dt(double local_dt) const {
#ifdef SPLAY_ENABLE_MPI
    double global_dt;
    MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return global_dt;
#else
    return local_dt;
#endif
}

double MPIDecomp::global_min(double v) const {
#ifdef SPLAY_ENABLE_MPI
    double result;
    MPI_Allreduce(&v, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return result;
#else
    return v;
#endif
}

double MPIDecomp::global_max(double v) const {
#ifdef SPLAY_ENABLE_MPI
    double result;
    MPI_Allreduce(&v, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return v;
#endif
}

double MPIDecomp::global_l2_norm(double local_sum_sq, int local_count) const {
#ifdef SPLAY_ENABLE_MPI
    double global_sum_sq;
    int    global_count;
    MPI_Allreduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count,  &global_count,  1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
    return (global_count > 0) ? std::sqrt(global_sum_sq / global_count) : 0.0;
#else
    return (local_count > 0) ? std::sqrt(local_sum_sq / local_count) : 0.0;
#endif
}

// ─── Boundary conditions ──────────────────────────────────────────────────────
void apply_boundary_conditions(State& s, const Mesh& m,
                                const BCState& bc_left, const BCState& bc_right,
                                const MPIDecomp& decomp,
                                double gamma, double cv, double R_gas)
{
    const int ng = m.n_ghost;
    const int nl = m.n_local;

    // ── Left boundary (rank 0 only) ──────────────────────────────────────────
    if (decomp.left_neighbor < 0) {
        if (bc_left.type == BCType::Inflow) {
            // Fixed primitive state from config.
            const double rho_bc = bc_left.pressure / (R_gas * bc_left.temperature);
            const double u_bc   = bc_left.velocity;
            const double T_bc   = bc_left.temperature;
            const double e_bc   = cv * T_bc;
            const double E_bc   = e_bc + 0.5 * u_bc * u_bc;
            for (int g = 0; g < ng; ++g) {
                s.rho[g]  = rho_bc;
                s.rhou[g] = rho_bc * u_bc;
                s.rhoE[g] = rho_bc * E_bc;
                s.u[g]    = u_bc;
                s.p[g]    = bc_left.pressure;
                s.T[g]    = T_bc;
            }
        } else {
            // Outflow: zero-gradient extrapolation from interior.
            const int i0 = ng;  // first interior cell
            for (int g = 0; g < ng; ++g) {
                const int gi = ng - 1 - g;
                s.rho[gi]  = s.rho[i0];
                s.rhou[gi] = s.rhou[i0];
                s.rhoE[gi] = s.rhoE[i0];
                s.u[gi]    = s.u[i0];
                s.p[gi]    = s.p[i0];
                s.T[gi]    = s.T[i0];
            }
        }
    }

    // ── Right boundary (last rank only) ─────────────────────────────────────
    if (decomp.right_neighbor < 0) {
        if (bc_right.type == BCType::Inflow) {
            const double rho_bc = bc_right.pressure / (R_gas * bc_right.temperature);
            const double u_bc   = bc_right.velocity;
            const double T_bc   = bc_right.temperature;
            const double e_bc   = cv * T_bc;
            const double E_bc   = e_bc + 0.5 * u_bc * u_bc;
            for (int g = 0; g < ng; ++g) {
                const int gi = ng + nl + g;
                s.rho[gi]  = rho_bc;
                s.rhou[gi] = rho_bc * u_bc;
                s.rhoE[gi] = rho_bc * E_bc;
                s.u[gi]    = u_bc;
                s.p[gi]    = bc_right.pressure;
                s.T[gi]    = T_bc;
            }
        } else {
            // Outflow: zero-gradient from last interior cell.
            const int iN = ng + nl - 1;  // last interior cell
            for (int g = 0; g < ng; ++g) {
                const int gi = ng + nl + g;
                s.rho[gi]  = s.rho[iN];
                s.rhou[gi] = s.rhou[iN];
                s.rhoE[gi] = s.rhoE[iN];
                s.u[gi]    = s.u[iN];
                s.p[gi]    = s.p[iN];
                s.T[gi]    = s.T[iN];
            }
        }
    }
}

} // namespace splay
