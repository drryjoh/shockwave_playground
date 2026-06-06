# SPLAY Requirements

SPLAY, short for Shock Playground, is a 1D finite-volume solver for studying Euler and Navier-Stokes shock waves in pure argon and pure nitrogen.

## Primary goal

Resolve a physical Navier-Stokes shock as a continuous viscous structure using physical viscosity and thermal conduction. The baseline method is central inviscid flux plus second-order central viscous terms.

## Comparison goal

Support MUSCL and PPM shock-capturing schemes both with and without physical diffusion so that their effect on the resolved shock layer can be quantified.

## Required physics

- 1D compressible Euler equations.
- 1D compressible Navier-Stokes equations.
- Ideal gas law.
- Calorically perfect gas.
- Constant gamma.
- Pure argon and pure nitrogen only.
- No mixture laws.
- Temperature-dependent viscosity and conductivity from fourth-order Cantera-based fits.

## Required numerics

- 1D finite volume.
- Uniform runtime-generated grid.
- Explicit SSPRK3.
- Central inviscid flux.
- MUSCL reconstruction.
- PPM reconstruction.
- Rusanov and HLLC Riemann solvers.
- Central second-order viscous fluxes.
- Independent control of inviscid scheme and viscous terms.

## Required parallelism

- MPI from the start.
- 1D contiguous domain decomposition.
- Nearest-neighbor ghost exchange.
- Serial mode with one MPI rank.
- Optional TBB or standard shared-memory loops.

## Required configuration

Use YAML. Support nested keys and dotted keys. Convert all units internally to SI.

## Required outputs

- CSV solution output.
- Human-readable restart files.
- Diagnostics logs.
- Python plotting scripts.
- Tutorial comparison plots.

## Required tutorials

The primary tutorial is a target argon shock with five cases:

1. central viscous Navier-Stokes baseline
2. MUSCL Euler
3. PPM Euler
4. MUSCL Navier-Stokes
5. PPM Navier-Stokes

