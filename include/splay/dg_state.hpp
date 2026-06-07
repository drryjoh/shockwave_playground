#pragma once
#include <vector>
#include "splay/mesh.hpp"
#include "splay/gas.hpp"
#include "splay/config.hpp"
#include "splay/dg_basis.hpp"

namespace splay {

// ── DGState: nodal DOF arrays for the DG solution ────────────────────────────
//
// Storage layout: q[j][i]  =  j-th GLL node of cell i,  j = 0..n_dof-1.
//
// GLL endpoint convention (no interpolation needed at faces):
//   q[0][i]   — left-face  value of cell i  (ξ = -1)
//   q[p][i]   — right-face value of cell i  (ξ = +1)
//
// Ghost-cell protocol:
//   Ghost cells [0..n_ghost-1] and [n_ghost+n_local..n_total-1] carry DOF
//   values set by apply_boundary_conditions_dg (physical BCs) or
//   fill_ghosts_dg (MPI exchange).  For BC ghosts all n_dof slots are filled
//   with the BC state; for MPI ghosts only the outward face DOF is strictly
//   required (face_R of left ghost, face_L of right ghost), but all DOFs are
//   exchanged for simplicity.
struct DGState {
    int n_total;   ///< total cells  (interior + 2 × n_ghost)
    int p;         ///< polynomial order (1 or 2)
    int n_dof;     ///< p + 1

    DGBasis basis; ///< precomputed GLL tables

    // ── Conservative DOFs ─────────────────────────────────────────────────────
    std::vector<std::vector<double>> rho;   ///< [j][i]  kg/m³
    std::vector<std::vector<double>> rhou;  ///< [j][i]  kg/(m²·s)
    std::vector<std::vector<double>> rhoE;  ///< [j][i]  J/m³

    // ── Primitive DOFs (cached; updated after each RK stage) ─────────────────
    std::vector<std::vector<double>> u;      ///< [j][i]  m/s
    std::vector<std::vector<double>> prim_p; ///< [j][i]  Pa
    std::vector<std::vector<double>> T;      ///< [j][i]  K

    // ── Artificial viscosity coefficient (one per cell) ───────────────────────
    std::vector<double> epsilon; ///< [i]  m²/s  (set to zero when AV is off)

    explicit DGState(int n_total, int poly_order);

    // ── Inline face accessors (no interpolation: GLL endpoints are DOFs) ──────
    double face_rho_L (int i) const { return rho [0][i]; }
    double face_rho_R (int i) const { return rho [p][i]; }
    double face_rhou_L(int i) const { return rhou[0][i]; }
    double face_rhou_R(int i) const { return rhou[p][i]; }
    double face_rhoE_L(int i) const { return rhoE[0][i]; }
    double face_rhoE_R(int i) const { return rhoE[p][i]; }
    double face_u_L   (int i) const { return u   [0][i]; }
    double face_u_R   (int i) const { return u   [p][i]; }
    double face_p_L   (int i) const { return prim_p[0][i]; }
    double face_p_R   (int i) const { return prim_p[p][i]; }
    double face_T_L   (int i) const { return T   [0][i]; }
    double face_T_R   (int i) const { return T   [p][i]; }

    // ── Primitive updates ─────────────────────────────────────────────────────
    void update_primitives     (int ib, int ie, const GasModel& gas);
    void update_primitives_cell(int i,           const GasModel& gas);
};

// ── Residual arrays (same layout as DGState) ─────────────────────────────────
struct DGResidual {
    int n_total;
    int n_dof;
    std::vector<std::vector<double>> r_rho;
    std::vector<std::vector<double>> r_rhou;
    std::vector<std::vector<double>> r_rhoE;

    explicit DGResidual(int n_total, int n_dof);
    void zero();
};

// ── Initialisation helpers ───────────────────────────────────────────────────

/// Uniform state: all DOFs in every cell set to the same primitive values.
void init_constant_dg(DGState& s, const Mesh& m, const GasModel& gas,
                      double rho0, double p0, double u0);

/// Step ICs (Sod / Riemann problem).  Each cell is assigned the L or R state
/// based on x_cell[i] vs x_diaphragm; all DOFs in a cell share that value.
void init_step_dg(DGState& s, const Mesh& m, const GasModel& gas,
                  double rho_L, double p_L, double u_L,
                  double rho_R, double p_R, double u_R,
                  double x_diaphragm);

/// Gaussian pressure perturbation on a uniform background.
/// Each DOF is placed at its physical position x = x_cell[i] + (xi_j/2)*dx
/// so the perturbation is resolved at sub-cell resolution.
void init_gaussian_perturbation_dg(DGState& s, const Mesh& m, const GasModel& gas,
                                    double p0, double T0, double u0,
                                    double amplitude, double x0, double sigma);

// ── Ghost cell / boundary condition helpers ───────────────────────────────────

/// Apply physical boundary conditions to left and right ghost cells.
/// For outflow: copy first/last interior cell DOFs to ghost cells.
/// For inflow:  fill ghost cell DOFs with the BC primitive state.
void apply_boundary_conditions_dg(DGState& s, const Mesh& m,
                                    const BCState& bc_left,
                                    const BCState& bc_right,
                                    const GasModel& gas);

/// Exchange face DOFs with MPI neighbours (single-rank: no-op).
/// For MPI: sends the outward face DOF of each boundary interior cell
/// and receives the face DOF from the neighbour into the ghost cell.
void fill_ghosts_dg(DGState& s, const Mesh& m, int rank, int nranks,
                    int left_neighbor, int right_neighbor);

} // namespace splay
