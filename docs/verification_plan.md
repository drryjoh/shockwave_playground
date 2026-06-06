# SPLAY Verification Plan

## Purpose

Before interpreting the target shock calculations, verify basic numerical and software behavior.

## Tests

### 1. Uniform flow preservation

Run a uniform state with periodic or compatible boundary conditions. Verify that the solution remains uniform in serial and MPI.

### 2. Smooth advection

Advect a sine wave with the Euler equations in a smooth low-Mach setting. Verify second-order convergence for the central or MUSCL unlimited option where appropriate.

### 3. Gaussian diffusion

Run a scalar-like thermal diffusion or viscous diffusion test where an analytic broadening rate is available. Verify second-order central diffusion behavior.

### 4. Normal shock state calculator

Verify ideal-gas Rankine-Hugoniot relations for argon and nitrogen. Check that computed downstream states match expected values within tolerance.

### 5. Sod shock tube

Run an inviscid Euler shock tube using MUSCL and PPM with HLLC. This verifies shock-capturing mode, not the resolved Navier-Stokes shock mode.

### 6. Restart consistency

Run a case continuously for N steps. Then run N/2 steps, write restart, restart, and run to N steps. Compare the final states.

### 7. MPI consistency

Run the same case with one rank and multiple ranks. Compare output to roundoff-level differences where possible.

### 8. Target shock sanity checks

For the target argon shock tutorial, verify:

- density remains positive
- pressure remains positive
- temperature remains positive
- viscosity remains positive
- conductivity remains positive
- central viscous case produces a smooth shock layer
- Euler MUSCL/PPM cases produce sharper numerical shock structures
- viscous MUSCL/PPM cases can be compared against the central viscous baseline

## Shock thickness diagnostic

Estimate shock thickness using a 10%-90% transition width for density, pressure, or temperature. Report the number of cells across this thickness.

