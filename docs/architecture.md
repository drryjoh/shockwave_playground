# SPLAY Architecture

## Project structure

Recommended layout:

```text
SPLAY/
├── CMakeLists.txt
├── README.md
├── docs/
├── include/splay/
├── src/
├── tools/
├── scripts/
├── tutorials/
└── tests/
```

## Main modules

### Config

Reads YAML, supports dotted keys, validates required fields, performs unit conversion, and stores all quantities internally in SI.

### Gas

Provides pure-species ideal gas properties for argon and nitrogen.

### Transport

Evaluates generated fourth-order polynomial fits for viscosity and thermal conductivity.

### Mesh

Generates a uniform 1D finite-volume mesh at runtime. In MPI mode, each rank creates only its local cells plus ghost cells.

### MPI decomposition

Computes contiguous global index ranges, neighbor ranks, ghost-cell width, and exchange schedules.

### Reconstruction

Implements central, MUSCL, and PPM face-state reconstruction.

### Riemann

Implements central, Rusanov, and HLLC fluxes.

### Viscous flux

Computes second-order central viscous fluxes using:

```text
tau_xx = (4/3)*mu*dudx
q_x    = -k*dTdx
F_visc = [0, tau_xx, u*tau_xx - q_x]
```

### Residual

Computes the semi-discrete residual R(U). Keep this module independent of the time integrator so implicit methods can be added later.

### Time integration

Implements SSPRK3 initially.

### I/O

Writes CSV solution output, diagnostics logs, and human-readable restart files.

## Data layout

Prefer structure-of-arrays:

```text
rho[i], rhou[i], rhoE[i]
p[i], T[i], u[i]
mu[i], k[i]
```

Avoid allocations in the time-stepping loop.

## Parallel initialization

All ranks read the same input file. Each rank computes its owned global index range and initializes its owned cells analytically. Ghost cells are filled by boundary conditions and MPI exchange.

## Restart design

Support one file per rank plus metadata for the initial implementation. Files must be human-readable and editable.

