#pragma once
#include <array>
#include <stdexcept>

namespace splay {

// ── DG basis tables: GLL nodes, weights, and derivative matrix ───────────────
//
// GLL (Gauss-Lobatto-Legendre) nodes include the endpoints ξ = ±1.
// Therefore face DOFs are exactly dof[0] (left face) and dof[p] (right face) —
// no interpolation is required at faces; this also minimises MPI communication
// to a single face-value exchange per boundary.
//
// All tables are precomputed at compile-time; zero polynomial evaluation occurs
// during the solve.
//
// D[k][j] = dL_j/dξ evaluated at ξ_k
//   where L_j is the Lagrange basis polynomial through the GLL nodes.
//
// Quadrature: mass matrix is diagonal (lumped) with M_jj = (dx/2) * w[j].
//
// DG volume residual for cell i, DOF j:
//   R_j = (2/(dx·w[j])) · [Σ_k w[k] · F(U_k) · D[k][j] ± face terms]
//
// Supported orders: p = 1 (2 DOFs) and p = 2 (3 DOFs).

struct DGBasis {
    int n_dof;                              ///< p+1
    std::array<double, 3> xi;              ///< GLL nodes on [-1, 1]
    std::array<double, 3> w;               ///< GLL weights  (sum = 2)
    std::array<std::array<double, 3>, 3> D; ///< D[k][j] = dL_j/dxi(xi_k)

    // ── Legendre modal coefficients at GLL nodes ─────────────────────────────
    // Used by the Persson-Peraire smoothness sensor to project the nodal
    // solution onto modal space without matrix inversion.
    //
    // modal_proj[j][k] = w[k] · P_j(xi[k])
    //   where P_j is the j-th L2-normalised Legendre polynomial on [-1,1].
    // Modal coefficient: a_j ≈ Σ_k modal_proj[j][k] · u_k
    // The ||P_j||² = 1 by construction (normalised), so:
    //   ||U_e - Π_{p-1}U_e||² ≈ a_p²
    //   ||U_e||²              ≈ Σ_j a_j²
    std::array<std::array<double, 3>, 3> modal_proj; ///< [mode j][node k]
};

// ── p = 1: 2-node GLL ─────────────────────────────────────────────────────────
// Nodes  : ξ = {-1, 1}
// Weights: w = {1, 1}          (sum = 2 ✓)
// Basis  : L_0 = (1-ξ)/2,  L_1 = (1+ξ)/2
// D[k][j]: dL_j/dξ = {-1/2, +1/2} (constant, independent of k)
//
// Normalised Legendre: P_0 = 1/√2,  P_1 = √(3/2)·ξ
// modal_proj[0][k] = w[k] · (1/√2)
// modal_proj[1][k] = w[k] · √(3/2) · xi[k]
inline DGBasis make_dg_basis_p1() {
    DGBasis b;
    b.n_dof = 2;
    b.xi  = { -1.0, 1.0, 0.0 };
    b.w   = {  1.0, 1.0, 0.0 };
    b.D   = {{
        { -0.5,  0.5, 0.0 },   // k=0, ξ=-1
        { -0.5,  0.5, 0.0 },   // k=1, ξ=+1
        {  0.0,  0.0, 0.0 }    // unused slot
    }};
    // modal_proj[mode][node]
    // P_0 = 1/√2 ≈ 0.70711,  P_1(ξ) = √(3/2)·ξ ≈ 1.22474·ξ
    constexpr double inv_sqrt2  = 0.70710678118654752;
    constexpr double sqrt1p5    = 1.22474487139158905;
    b.modal_proj = {{
        { inv_sqrt2 * 1.0, inv_sqrt2 * 1.0, 0.0 },   // mode 0: w·P_0
        { sqrt1p5 * (-1.0), sqrt1p5 * 1.0, 0.0 },    // mode 1: w·P_1(xi)
        { 0.0, 0.0, 0.0 }                              // unused
    }};
    return b;
}

// ── p = 2: 3-node GLL ─────────────────────────────────────────────────────────
// Nodes  : ξ = {-1, 0, 1}
// Weights: w = {1/3, 4/3, 1/3}  (sum = 2 ✓)
// Basis  : L_0 = ξ(ξ-1)/2,  L_1 = (1-ξ²),  L_2 = ξ(ξ+1)/2
// D[k][j]:
//   at ξ=-1: {-3/2,  2, -1/2}
//   at ξ= 0: {-1/2,  0,  1/2}
//   at ξ=+1: { 1/2, -2,  3/2}
//
// Normalised Legendre: P_0 = 1/√2, P_1 = √(3/2)·ξ, P_2 = √(5/2)·(3ξ²-1)/2
// modal_proj[j][k] = w[k] · P_j(xi[k])
inline DGBasis make_dg_basis_p2() {
    DGBasis b;
    b.n_dof = 3;
    b.xi  = { -1.0, 0.0, 1.0 };
    b.w   = { 1.0/3.0, 4.0/3.0, 1.0/3.0 };
    b.D   = {{
        { -3.0/2.0,  2.0, -1.0/2.0 },  // k=0, ξ=-1
        { -1.0/2.0,  0.0,  1.0/2.0 },  // k=1, ξ= 0
        {  1.0/2.0, -2.0,  3.0/2.0 }   // k=2, ξ=+1
    }};
    // P_0(ξ) = 1/√2
    // P_1(ξ) = √(3/2)·ξ
    // P_2(ξ) = √(5/2)·(3ξ²-1)/2
    constexpr double inv_sqrt2  = 0.70710678118654752;
    constexpr double sqrt1p5    = 1.22474487139158905;
    constexpr double sqrt2p5    = 1.58113883008418967;
    // P_2 at nodes: P_2(-1) = √(5/2)·1 = sqrt2p5, P_2(0) = -√(5/2)/2, P_2(1) = sqrt2p5
    constexpr double p2_at_m1   =  sqrt2p5;
    constexpr double p2_at_0    = -sqrt2p5 * 0.5;
    constexpr double p2_at_p1   =  sqrt2p5;
    b.modal_proj = {{
        // mode 0: w[k] · P_0 = w[k] / √2
        { (1.0/3.0)*inv_sqrt2, (4.0/3.0)*inv_sqrt2, (1.0/3.0)*inv_sqrt2 },
        // mode 1: w[k] · P_1(xi[k]) = w[k] · √(3/2) · xi[k]
        { (1.0/3.0)*sqrt1p5*(-1.0), 0.0, (1.0/3.0)*sqrt1p5*(1.0) },
        // mode 2: w[k] · P_2(xi[k])
        { (1.0/3.0)*p2_at_m1, (4.0/3.0)*p2_at_0, (1.0/3.0)*p2_at_p1 }
    }};
    return b;
}

/// Return basis for polynomial order p (1 or 2).
inline DGBasis make_dg_basis(int p) {
    if (p == 1) return make_dg_basis_p1();
    if (p == 2) return make_dg_basis_p2();
    throw std::runtime_error("DG polynomial order must be 1 or 2 (got " +
                              std::to_string(p) + ")");
}

} // namespace splay
