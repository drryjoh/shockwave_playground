# SPLAY Design History and Note of Record

This document records the design rationale behind SPLAY, the Shock Playground code.

## Motivation

The project arose from the need to study one-dimensional shock structure using finite volume methods without automatically treating the shock as an Euler discontinuity. Conventional shock-capturing schemes such as PPM, MUSCL with limiters, and Riemann-solver-based fluxes are designed to stabilize discontinuities. That behavior can interfere with a calculation whose purpose is to resolve the finite-thickness Navier-Stokes shock layer.

## Central flux baseline

The baseline physical calculation uses a central inviscid flux with physical viscosity and heat conduction. The purpose is to let the Navier-Stokes transport terms determine the shock thickness rather than allowing numerical shock dissipation to collapse the profile.

## MUSCL and PPM comparison modes

MUSCL and PPM are included intentionally. They are not the baseline for the resolved viscous shock. They are included to demonstrate how limiter-based and shock-capturing flux reconstructions may alter, sharpen, or distort a shock layer that should be resolved by physical viscosity and thermal conduction.

The code therefore permits MUSCL and PPM both with and without physical diffusion:

- MUSCL/PPM without viscosity: Euler shock-capturing reference.
- MUSCL/PPM with viscosity: test of whether the reconstruction interferes with the resolved Navier-Stokes shock.

## Viscous stress

For strictly 1D compressible Navier-Stokes flow with the Stokes hypothesis, the viscous stress is:

```text
tau_xx = (4/3)*mu*du/dx
```

The heat flux is:

```text
q_x = -k*dT/dx
```

The viscous energy flux is:

```text
u*tau_xx - q_x
```

## CFL and diffusion restriction

The acoustic CFL is based on hyperbolic wave speeds:

```text
|u| + a
```

The diffusion timestep restriction is different. It is a numerical parabolic stability restriction, not a physical characteristic speed. It scales like:

```text
dt ~ dx^2/nu
```

and can dominate on fine meshes.

## Mesh and initialization

The grid is generated at runtime from the YAML configuration. There is no external mesh file or preprocessing step. In MPI mode, every rank reads the same configuration, computes the same global mesh definition, owns a contiguous subset of cells, and initializes its local cells from the analytic initial condition.

The target shock initialization uses a hyperbolic tangent profile between left and right states. The left state is the post-shock state and the right state is the pre-shock state.

## MPI from the beginning

MPI is included from the start because even though the problem is 1D, the target calculations may use fine grids and many comparison cases. The 1D setting permits a simple contiguous decomposition and nearest-neighbor ghost exchange.

## Restart files

Restart files must be human-readable and editable. The initial design favors one restart file per MPI rank plus metadata, avoiding large gather/scatter operations while preserving restart capability.

## Transport model

Transport properties are temperature-dependent. They are generated from Cantera data for pure argon and pure nitrogen using fourth-order polynomial fits. The generated coefficients are compiled into the solver.

## Target tutorial

The primary tutorial is an argon shock case with specified pre-shock and post-shock states. Five simulations are required: central viscous, MUSCL Euler, PPM Euler, MUSCL viscous, and PPM viscous. These cases are designed to expose the differences between physically resolved and numerically captured shocks.

