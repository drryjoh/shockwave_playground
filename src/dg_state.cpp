#include "splay/dg_state.hpp"
#include <cmath>
#include <stdexcept>

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

namespace splay {

// ── DGState ──────────────────────────────────────────────────────────────────

DGState::DGState(int n_total_, int poly_order)
    : n_total(n_total_), p(poly_order), n_dof(poly_order + 1),
      basis(make_dg_basis(poly_order))
{
    rho  .assign(n_dof, std::vector<double>(n_total, 0.0));
    rhou .assign(n_dof, std::vector<double>(n_total, 0.0));
    rhoE .assign(n_dof, std::vector<double>(n_total, 0.0));
    u    .assign(n_dof, std::vector<double>(n_total, 0.0));
    prim_p.assign(n_dof, std::vector<double>(n_total, 0.0));
    T    .assign(n_dof, std::vector<double>(n_total, 0.0));
    epsilon.assign(n_total, 0.0);
}

void DGState::update_primitives_cell(int i, const GasModel& gas) {
    for (int j = 0; j < n_dof; ++j) {
        u[j][i]      = rhou[j][i] / rho[j][i];
        prim_p[j][i] = gas.pressure_from_conserved(rho[j][i], rhoE[j][i], rhou[j][i]);
        T[j][i]      = gas.temperature(rho[j][i], prim_p[j][i]);
    }
}

void DGState::update_primitives(int ib, int ie, const GasModel& gas) {
    for (int i = ib; i < ie; ++i)
        update_primitives_cell(i, gas);
}

// ── DGResidual ───────────────────────────────────────────────────────────────

DGResidual::DGResidual(int n_total_, int n_dof_)
    : n_total(n_total_), n_dof(n_dof_)
{
    r_rho .assign(n_dof, std::vector<double>(n_total, 0.0));
    r_rhou.assign(n_dof, std::vector<double>(n_total, 0.0));
    r_rhoE.assign(n_dof, std::vector<double>(n_total, 0.0));
}

void DGResidual::zero() {
    for (int j = 0; j < n_dof; ++j) {
        std::fill(r_rho [j].begin(), r_rho [j].end(), 0.0);
        std::fill(r_rhou[j].begin(), r_rhou[j].end(), 0.0);
        std::fill(r_rhoE[j].begin(), r_rhoE[j].end(), 0.0);
    }
}

// ── Initialisation ───────────────────────────────────────────────────────────

void init_constant_dg(DGState& s, const Mesh& m, const GasModel& gas,
                      double rho0, double p0, double u0) {
    const double T0    = gas.temperature(rho0, p0);
    const double rhoE0 = rho0 * gas.total_energy(T0, u0);
    const double rhou0 = rho0 * u0;
    for (int i = 0; i < m.n_total; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            s.rho [j][i] = rho0;
            s.rhou[j][i] = rhou0;
            s.rhoE[j][i] = rhoE0;
        }
    }
    s.update_primitives(0, m.n_total, gas);
}

void init_step_dg(DGState& s, const Mesh& m, const GasModel& gas,
                  double rho_L, double p_L, double u_L,
                  double rho_R, double p_R, double u_R,
                  double x_diaphragm) {
    for (int i = 0; i < m.n_total; ++i) {
        const bool   left = (m.x_cell[i] <= x_diaphragm);
        const double rho  = left ? rho_L : rho_R;
        const double pres = left ? p_L   : p_R;
        const double vel  = left ? u_L   : u_R;
        const double Tv   = gas.temperature(rho, pres);
        const double rhoE = rho * gas.total_energy(Tv, vel);
        for (int j = 0; j < s.n_dof; ++j) {
            s.rho [j][i] = rho;
            s.rhou[j][i] = rho * vel;
            s.rhoE[j][i] = rhoE;
        }
    }
    s.update_primitives(0, m.n_total, gas);
}

void init_gaussian_perturbation_dg(DGState& s, const Mesh& m, const GasModel& gas,
                                    double p0, double T0, double u0,
                                    double amplitude, double x0, double sigma) {
    const double rho0 = p0 / (gas.R * T0);
    for (int i = 0; i < m.n_total; ++i) {
        for (int j = 0; j < s.n_dof; ++j) {
            // Physical x of this GLL node: x = x_cell[i] + (xi_j/2)*dx
            const double xi_j = s.basis.xi[j];
            const double x    = m.x_cell[i] + 0.5 * xi_j * m.dx;
            const double drel = (x - x0) / sigma;
            const double pres = p0 * (1.0 + amplitude * std::exp(-drel * drel));
            const double Tv   = pres / (gas.R * rho0);
            s.rho [j][i] = rho0;
            s.rhou[j][i] = rho0 * u0;
            s.rhoE[j][i] = rho0 * gas.total_energy(Tv, u0);
        }
    }
    s.update_primitives(0, m.n_total, gas);
}

// ── Boundary conditions ───────────────────────────────────────────────────────

void apply_boundary_conditions_dg(DGState& s, const Mesh& m,
                                    const BCState& bc_left,
                                    const BCState& bc_right,
                                    const GasModel& gas) {
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    const int ng = m.n_ghost;

    // ── Left ghost cells ─────────────────────────────────────────────────────
    for (int g = 0; g < ng; ++g) {
        const int ghost = g;
        if (bc_left.type == BCType::Outflow) {
            // Zero-gradient: mirror first interior cell
            const int src = ib;
            for (int j = 0; j < s.n_dof; ++j) {
                s.rho [j][ghost] = s.rho [j][src];
                s.rhou[j][ghost] = s.rhou[j][src];
                s.rhoE[j][ghost] = s.rhoE[j][src];
            }
        } else {
            // Inflow: fill all DOFs with prescribed state
            const double rho  = bc_left.pressure / (gas.R * bc_left.temperature);
            const double Tv   = bc_left.temperature;
            const double rhoE = rho * gas.total_energy(Tv, bc_left.velocity);
            for (int j = 0; j < s.n_dof; ++j) {
                s.rho [j][ghost] = rho;
                s.rhou[j][ghost] = rho * bc_left.velocity;
                s.rhoE[j][ghost] = rhoE;
            }
        }
    }

    // ── Right ghost cells ────────────────────────────────────────────────────
    for (int g = 0; g < ng; ++g) {
        const int ghost = ie + g;
        if (bc_right.type == BCType::Outflow) {
            const int src = ie - 1;
            for (int j = 0; j < s.n_dof; ++j) {
                s.rho [j][ghost] = s.rho [j][src];
                s.rhou[j][ghost] = s.rhou[j][src];
                s.rhoE[j][ghost] = s.rhoE[j][src];
            }
        } else {
            const double rho  = bc_right.pressure / (gas.R * bc_right.temperature);
            const double Tv   = bc_right.temperature;
            const double rhoE = rho * gas.total_energy(Tv, bc_right.velocity);
            for (int j = 0; j < s.n_dof; ++j) {
                s.rho [j][ghost] = rho;
                s.rhou[j][ghost] = rho * bc_right.velocity;
                s.rhoE[j][ghost] = rhoE;
            }
        }
    }

    s.update_primitives(0, m.n_total, gas);
}

// ── MPI ghost exchange ────────────────────────────────────────────────────────
//
// For GLL nodal DG, only the face DOFs of the boundary interior cells need to
// be sent to the neighbour (dof[p] of the last left-side interior cell goes to
// the right rank's left ghost, and vice-versa).  We exchange ALL n_dof values
// per cell for simplicity — each side still only sends 1 cell × n_dof × 3 vars.

void fill_ghosts_dg(DGState& s, const Mesh& m, int /*rank*/, int nranks,
                    int left_neighbor, int right_neighbor) {
    if (nranks == 1) return; // BCs already filled all ghosts

#ifdef SPLAY_ENABLE_MPI
    const int ib  = m.interior_begin();
    const int ie  = m.interior_end();
    const int nd  = s.n_dof;
    const int tag = 42;

    // Pack send buffers: [rho DOFs | rhou DOFs | rhoE DOFs] for one cell
    const int buf_size = 3 * nd;
    std::vector<double> send_left(buf_size), send_right(buf_size);
    std::vector<double> recv_left(buf_size), recv_right(buf_size);

    // Left boundary: send face DOFs of first interior cell to left neighbour
    for (int j = 0; j < nd; ++j) {
        send_left[j]        = s.rho [j][ib];
        send_left[nd + j]   = s.rhou[j][ib];
        send_left[2*nd + j] = s.rhoE[j][ib];
    }
    // Right boundary: send face DOFs of last interior cell to right neighbour
    for (int j = 0; j < nd; ++j) {
        send_right[j]        = s.rho [j][ie - 1];
        send_right[nd + j]   = s.rhou[j][ie - 1];
        send_right[2*nd + j] = s.rhoE[j][ie - 1];
    }

    MPI_Request reqs[4];
    int n_reqs = 0;

    if (left_neighbor >= 0) {
        MPI_Isend(send_left.data(), buf_size, MPI_DOUBLE, left_neighbor,  tag,     MPI_COMM_WORLD, &reqs[n_reqs++]);
        MPI_Irecv(recv_left.data(), buf_size, MPI_DOUBLE, left_neighbor,  tag + 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
    }
    if (right_neighbor >= 0) {
        MPI_Isend(send_right.data(), buf_size, MPI_DOUBLE, right_neighbor, tag + 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
        MPI_Irecv(recv_right.data(), buf_size, MPI_DOUBLE, right_neighbor, tag,     MPI_COMM_WORLD, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // Unpack into ghost cells — caller (dg_refresh) calls update_primitives
    // over the full range after fill_ghosts_dg returns.
    if (left_neighbor >= 0) {
        const int ghost = ib - 1;
        for (int j = 0; j < nd; ++j) {
            s.rho [j][ghost] = recv_left[j];
            s.rhou[j][ghost] = recv_left[nd + j];
            s.rhoE[j][ghost] = recv_left[2*nd + j];
        }
    }
    if (right_neighbor >= 0) {
        const int ghost = ie;
        for (int j = 0; j < nd; ++j) {
            s.rho [j][ghost] = recv_right[j];
            s.rhou[j][ghost] = recv_right[nd + j];
            s.rhoE[j][ghost] = recv_right[2*nd + j];
        }
    }
#else
    (void)s; (void)m; (void)left_neighbor; (void)right_neighbor;
#endif
}

} // namespace splay
