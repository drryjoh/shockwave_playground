# Master Prompt for Offline Coding Agent: SPLAY

You are to create a C++ codebase named **SPLAY**, short for **Shock Playground**. The code is a special-purpose, efficient, MPI-capable, one-dimensional finite-volume solver for studying shock waves in argon and nitrogen. The primary goal is to resolve a viscous Navier-Stokes shock as a continuous physical structure, while also supporting MUSCL and PPM shock-capturing schemes for comparison.

Generate a complete, buildable, documented project. The project should include source code, CMake build files, YAML configuration parsing, MPI decomposition, optional shared-memory parallelism, transport-coefficient generation from Cantera, tutorials, restart capability, and post-processing scripts.

## Core objective

SPLAY must simulate 1D compressible Euler and Navier-Stokes shock problems using finite volume. The central design goal is to compare:

1. A central inviscid flux with physical viscosity and thermal conduction, intended to resolve the Navier-Stokes shock thickness as a continuous structure.
2. MUSCL and PPM shock-capturing schemes with physical diffusion disabled, intended to reproduce conventional Euler shock-capturing behavior.
3. MUSCL and PPM shock-capturing schemes with physical diffusion enabled, intended to demonstrate whether limiter/Riemann reconstruction interferes with the physically resolved viscous shock structure.

Do not make the code general 2D/3D. It is always 1D. Use this specialization to keep the code simple, efficient, and clear.

## Governing equations

Implement the 1D compressible Navier-Stokes equations in conservative form:

```text
U = [rho, rho*u, rho*E]^T
```

where:

```text
E = e + 0.5*u^2
p = rho*R*T
p = (gamma - 1)*rho*e
```

The conservative equation is:

```text
dU/dt + dF_inv/dx = dF_visc/dx
```

or equivalently use the residual form consistently in code.

The inviscid flux is:

```text
F_inv = [rho*u,
         rho*u^2 + p,
         u*(rho*E + p)]^T
```

The viscous flux is:

```text
F_visc = [0,
          tau_xx,
          u*tau_xx - q_x]^T
```

For strictly 1D Newtonian flow with Stokes hypothesis:

```text
tau_xx = (4/3)*mu*dudx
q_x    = -k*dTdx
```

The viscous contribution to the energy flux is therefore:

```text
u*tau_xx - q_x = u*tau_xx + k*dTdx
```

Use clear sign conventions and unit tests to ensure that the implemented residual matches the selected convention.

## Gas model

Support only pure argon and pure nitrogen. Do not implement mixture laws.

Each gas must be selectable from the YAML file:

```yaml
gas: argon
```

or

```yaml
gas: nitrogen
```

Each gas is calorically perfect and thermally perfect with constant:

- molecular weight
- R
- gamma
- cp
- cv

Use ideal gas closure. Make the gas model modular enough that another single species could be added later, but do not add mixture rules.

## Transport model

Use temperature-dependent viscosity and thermal conductivity:

```text
mu = mu(T)
k  = k(T)
```

Create a Python utility:

```text
tools/generate_transport_fits.py
```

This utility should:

1. Use Cantera to sample pure argon and pure nitrogen transport data.
2. Fit fourth-order polynomials for viscosity and conductivity over a user-selected temperature range.
3. Generate a C++ header, for example:

```text
include/splay/transport_coeffs.hpp
```

4. Include coefficients for argon and nitrogen.
5. Guard against negative viscosity or conductivity.
6. Warn or clamp if evaluating outside the fit range.
7. Store fit metadata in comments: gas, T_min, T_max, number of samples, fit date if available.

For the first implementation, polynomial form may be:

```text
mu(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
k(T)  = b0 + b1*T + b2*T^2 + b3*T^3 + b4*T^4
```

Alternatively, log-polynomial fits are acceptable if better conditioned, but document the exact form.

## Spatial discretization

Use a uniform 1D finite-volume mesh generated at runtime from the configuration file.

Baseline method:

1. Cell-centered conservative variables.
2. Cell-centered primitive variables computed from conservative variables.
3. Uniform grid.
4. Second-order finite volume.
5. Central differences for viscous gradients.
6. At least two ghost cells, more if required by PPM.
7. Leave code structure open to higher-order reconstruction later.

### Central inviscid flux

For the resolved Navier-Stokes baseline, implement central inviscid flux:

```text
F_{i+1/2} = 0.5*(F(U_L) + F(U_R))
```

For the pure central second-order mode, use unlimited second-order central reconstruction or piecewise-linear states with no limiter. Also allow first-order face states as a diagnostic if useful.

### MUSCL

Implement MUSCL reconstruction as an optional inviscid scheme.

Required options:

```yaml
inviscid_scheme: muscl
riemann_solver: hllc
limiter: minmod
```

Limiters:

- none
- minmod
- vanleer
- mc

The limiter must be selectable from YAML.

### PPM

Implement PPM reconstruction as an optional inviscid scheme.

Required options:

```yaml
inviscid_scheme: ppm
riemann_solver: hllc
limiter: ppm
```

The implementation can be a clean baseline PPM appropriate for 1D Euler/Navier-Stokes comparison. Document what flattening, monotonicity constraints, contact steepening, or shock detection are implemented. For this project, prefer clarity over excessive PPM variants.

### Riemann solvers

Implement selectable inviscid numerical flux options:

```yaml
riemann_solver: none
riemann_solver: central
riemann_solver: rusanov
riemann_solver: hllc
```

Interpretation:

- `none` or `central`: use central average flux.
- `rusanov`: local Lax-Friedrichs/Rusanov flux.
- `hllc`: HLLC approximate Riemann solver.

This is needed to isolate whether observed shock distortion comes from reconstruction, limiting, or Riemann-solver dissipation.

## Physics options

The inviscid reconstruction and viscous terms must be independent.

Allow all combinations:

```yaml
solver:
  viscous_terms: true | false
  inviscid_scheme: central | muscl | ppm
  riemann_solver: none | central | rusanov | hllc
  limiter: none | minmod | vanleer | mc | ppm
```

Required comparison modes:

1. Central flux + viscous terms on: resolved Navier-Stokes shock baseline.
2. MUSCL + viscous terms on: test whether MUSCL interferes with resolved viscous shock structure.
3. PPM + viscous terms on: test whether PPM interferes with resolved viscous shock structure.
4. MUSCL + viscous terms off: Euler shock-capturing reference.
5. PPM + viscous terms off: Euler shock-capturing reference.
6. Central flux + viscous terms off: diagnostic only; may be unstable near shocks.

Do not silently disable viscosity when MUSCL or PPM is selected.

## Time integration

Implement explicit SSPRK3 as the default:

```yaml
solver:
  rk: 3
```

Use a clean residual function:

```cpp
R(U)
```

or equivalent so that later implicit integrators can be added without rewriting the spatial discretization.

Architect the code so future methods can be added:

- Backward Euler
- SDIRK2
- JFNK

Do not implement implicit methods now unless trivial, but structure the residual cleanly.

## Time-step restriction

When viscous terms are disabled, use only the convective/acoustic CFL:

```text
dt <= CFL * min_i dx/(abs(u_i) + a_i)
```

When viscous terms are enabled, include convection and diffusion restrictions:

```text
dt <= CFL * min_i [
  dx/(abs(u_i) + a_i),
  dx^2/nu_i,
  rho_i*cv_i*dx^2/k_i
]
```

where:

```text
nu_i = mu_i/rho_i
```

Report which constraint is active during the run. The diffusive restriction is a numerical parabolic stability condition, not a physical characteristic speed.

## Mesh generation

Generate the 1D mesh internally at runtime from the YAML configuration. Do not require external mesh files.

The YAML grid block should support:

```yaml
grid:
  units: cm
  L: 7.600403652e-4
  n: 2000
```

Also support optional:

```yaml
grid:
  x_min: 0.0
  x_max: 1.0e-3
  n: 2000
  ghost_cells: 3
```

If only `L` is specified, set `x_min = 0` and `x_max = L` after unit conversion.

For global cell index `i`:

```text
dx  = L/N
x_i = x_min + (i + 0.5)*dx
```

Internally convert all coordinates to SI units.

## MPI decomposition

SPLAY must be MPI-capable from the start.

Parallel behavior:

1. All MPI ranks read the same YAML configuration file.
2. All ranks know the global grid definition.
3. Decompose the global 1D cell range into contiguous local blocks.
4. Each rank owns only its local interior cells plus ghost cells.
5. Each rank computes local cell coordinates from global indices.
6. Each rank initializes its owned cells directly from the analytic initial condition.
7. Fill ghost cells using boundary conditions and MPI neighbor exchange.
8. Exchange nearest-neighbor ghost cells every residual evaluation and RK stage.
9. Support serial execution with one MPI rank.

Prefer structure-of-arrays memory layout:

```text
rho[i], rhou[i], rhoE[i]
```

or equivalent arrays for conservative variables, primitive variables, transport properties, and residuals.

Avoid repeated memory allocation inside the time-stepping loop.

## Shared-memory parallelism

Add optional thread-level parallelism using Intel TBB or standard C++ parallel loops.

The configuration may contain:

```yaml
cmake:
  mpi: on
  smp: tbb
```

This section is primarily informational for tutorials and build options. The build system should expose CMake options such as:

```bash
-D SPLAY_ENABLE_MPI=ON
-D SPLAY_ENABLE_TBB=ON
```

The code should run correctly without TBB.

## YAML configuration

Use YAML input files. The parser must support ordinary nested maps and dotted keys such as:

```yaml
domain.left:
  type: inflow
```

and equivalent nested form:

```yaml
domain:
  left:
    type: inflow
```

Use clear error messages for missing or invalid options.

Support unit conversion. Internally store all quantities in SI. Output diagnostics in SI when requested:

```yaml
diagnostics:
  units: SI
```

Required input sections:

```yaml
cmake:
  mpi: on
  smp: tbb

diagnostics:
  log.step: 10
  units: SI

gas: argon

domain.left:
  type: inflow
  pressure: 1.56880625e7
  temperature: 2632.24759509
  velocity: 1168.9

domain.right:
  type: outflow

grid:
  units: cm
  L: 7.600403652e-4
  n: 2000

initialization:
  type: tanh
  units: SI
  location: 3.0e-6
  thickness: 5.0e-8
  left:
    pressure: 1.56880625e7
    temperature: 2632.24759509
    velocity: 1168.9
  right:
    pressure: 5.0e5
    temperature: 300.0
    velocity: 0.0

solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: false
  inviscid_scheme: muscl
  riemann_solver: hllc
  limiter: minmod
```

Defaults:

```yaml
diagnostics:
  log.step: 100
```

## Initialization

Implement hyperbolic tangent initialization between left and right primitive states.

The left state is the post-shock state and the right state is the pre-shock state for the target tutorial.

Use:

```text
phi(x) = 0.5*(phi_L + phi_R) + 0.5*(phi_R - phi_L)*tanh((x - x_s)/delta_s)
```

Check orientation carefully. For `x << x_s`, this formula gives `phi_L`. For `x >> x_s`, it gives `phi_R`.

Apply this to primitive variables:

- pressure
- temperature
- velocity

Then compute:

```text
rho = p/(R*T)
e   = cv*T
E   = e + 0.5*u^2
rhoE = rho*E
```

Do not use a discontinuous jump as default. If provided, make it an optional diagnostic mode.

If a restart is supplied, skip analytic initialization and read the restart state instead.

## Boundary conditions

Implement:

- fixed inflow
- outflow
- optional fixed left/right state

For the target tutorial:

Left boundary:

```yaml
domain.left:
  type: inflow
  pressure: 1.56880625e7
  temperature: 2632.24759509
  velocity: 1168.9
```

This is a supersonic inflow fixed to the post-shock state.

Right boundary:

```yaml
domain.right:
  type: outflow
```

Use a reasonable zero-gradient extrapolation for outflow. Document limitations.

## Restart files

Implement human-readable and editable restart files.

Restart files should include:

1. grid coordinates
2. conservative variables
3. primitive variables if convenient
4. current time
5. current step
6. gas name
7. selected numerical options
8. enough metadata to verify consistency with the YAML file

Support restart command line:

```bash
mpirun -np 4 ./splay input_argon_shock.yml --restart restart/step_1000
```

For parallel runs, support either:

1. one restart file per MPI rank plus metadata, or
2. one gathered restart file written by rank 0.

For the first implementation, prefer one restart file per rank because it avoids large gather operations and is simple for 1D decomposition.

Restart files must be editable by the user. CSV plus a YAML metadata file is acceptable.

## Diagnostics and output

Output CSV files suitable for Python and ParaView.

At minimum output:

- x
- rho
- u
- p
- T
- Mach
- mu
- k
- residuals
- local dt information if useful

Each tutorial case must write to a separate output directory.

Log at startup:

1. project name and version
2. selected gas
3. grid size and domain length in SI
4. selected inviscid scheme
5. selected Riemann solver
6. selected limiter
7. whether viscous terms are enabled
8. whether the run is Euler or Navier-Stokes
9. CFL
10. final time
11. MPI rank count
12. ghost-cell count

During the run, log at `diagnostics.log.step` interval:

- step
- time
- dt
- active dt constraint
- residual norm
- min/max density
- min/max pressure
- min/max temperature

## Normal shock calculator

Include a utility function to compute ideal-gas normal-shock downstream states from:

- gas
- upstream Mach number
- upstream pressure
- upstream temperature
- upstream velocity or frame convention

This is used for verification and tutorial documentation.

For the supplied tutorial, verify that the stated left and right states are consistent with an approximately Mach 5.03 argon shock in the chosen frame. If there is a frame convention issue, document it clearly instead of silently changing the supplied states.

## Required target tutorial case

Create a tutorial centered on this case:

```text
L = 7.600403652e-4 cm
M = 5.03
right/pre-shock: P = 5e5 Pa, T = 300 K, V = 0 m/s
left/post-shock: P = 15688062.5 Pa, T = 2632.24759509 K, V = 1168.9 m/s
shock location = 3 micrometers
initial tanh thickness = 5e-8 m
left boundary = supersonic inflow fixed to post-shock state
right boundary = outflow
final time = 7.433391691866941e-07 s
CFL = 0.01
N = 2000
```

The grid length is specified in cm, but initialization is specified in SI. Convert internally to SI.

Create tutorial inputs for five cases:

### 1. Central viscous Navier-Stokes baseline

```yaml
case_name: argon_shock_central_navier_stokes
gas: argon
solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: true
  inviscid_scheme: central
  riemann_solver: none
  limiter: none
```

### 2. MUSCL Euler comparison

```yaml
case_name: argon_shock_muscl_euler
gas: argon
solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: false
  inviscid_scheme: muscl
  riemann_solver: hllc
  limiter: minmod
```

### 3. PPM Euler comparison

```yaml
case_name: argon_shock_ppm_euler
gas: argon
solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: false
  inviscid_scheme: ppm
  riemann_solver: hllc
  limiter: ppm
```

### 4. MUSCL Navier-Stokes comparison

```yaml
case_name: argon_shock_muscl_navier_stokes
gas: argon
solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: true
  inviscid_scheme: muscl
  riemann_solver: hllc
  limiter: minmod
```

### 5. PPM Navier-Stokes comparison

```yaml
case_name: argon_shock_ppm_navier_stokes
gas: argon
solver:
  cfl: 0.01
  time_end: 7.433391691866941e-07
  rk: 3
  viscous_terms: true
  inviscid_scheme: ppm
  riemann_solver: hllc
  limiter: ppm
```

All five cases should use the same domain, boundary, grid, and initialization sections.

## Tutorial requirements

Create:

```text
tutorials/argon_target_shock/
```

with:

- five YAML input files
- README.md
- run_serial.sh
- run_mpi.sh
- plot_compare.py
- optional restart example

The tutorial README must explain:

1. how to generate transport fits
2. how to configure CMake
3. how to compile
4. how to run in serial
5. how to run with MPI
6. how to restart
7. how to edit a restart file
8. how to plot results
9. what each case is intended to demonstrate
10. why the central viscous case is the baseline physical Navier-Stokes shock calculation
11. why MUSCL/PPM cases are included

Example commands:

```bash
python tools/generate_transport_fits.py --gas argon --tmin 200 --tmax 5000 --samples 400 --output include/splay/transport_coeffs.hpp

cmake -S . -B build -D SPLAY_ENABLE_MPI=ON -D SPLAY_ENABLE_TBB=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml
mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml

mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml --restart output/argon_shock_central_navier_stokes/restart/step_1000

python tutorials/argon_target_shock/plot_compare.py output/argon_target_shock
```

## Verification tests

Before relying on the shock case, include simple verification tests:

1. Smooth linear advection of a sine wave.
2. Diffusion of a Gaussian.
3. Euler Sod shock tube or equivalent standard 1D shock-capturing test.
4. Normal-shock state calculator check.
5. Uniform flow preservation across MPI decompositions.
6. Restart consistency: run N steps continuously and compare to run N/2 + restart + N/2.
7. MPI consistency: compare serial and MPI runs for the same setup.

For smooth tests, verify at least second-order convergence where applicable.

## Post-processing

Create Python scripts to:

1. plot density, velocity, pressure, temperature, Mach number, viscosity, and conductivity
2. overlay all five target tutorial cases
3. estimate number of cells across shock thickness
4. report min/max values and check for nonphysical states

Shock thickness diagnostic may use a 10%-90% variation definition for density, pressure, or temperature.

## Project layout

Use a clean layout such as:

```text
SPLAY/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   ├── design_history.md
│   ├── tutorials.md
│   └── verification_plan.md
├── include/splay/
│   ├── config.hpp
│   ├── gas.hpp
│   ├── mesh.hpp
│   ├── state.hpp
│   ├── transport.hpp
│   ├── transport_coeffs.hpp
│   ├── reconstruction.hpp
│   ├── riemann.hpp
│   ├── viscous_flux.hpp
│   ├── residual.hpp
│   ├── rk.hpp
│   ├── mpi_decomp.hpp
│   ├── io.hpp
│   └── diagnostics.hpp
├── src/
│   ├── main.cpp
│   ├── config.cpp
│   ├── gas.cpp
│   ├── mesh.cpp
│   ├── transport.cpp
│   ├── reconstruction.cpp
│   ├── riemann.cpp
│   ├── viscous_flux.cpp
│   ├── residual.cpp
│   ├── rk.cpp
│   ├── mpi_decomp.cpp
│   ├── io.cpp
│   └── diagnostics.cpp
├── tools/
│   └── generate_transport_fits.py
├── scripts/
│   └── plot_profile.py
├── tutorials/
│   └── argon_target_shock/
│       ├── README.md
│       ├── argon_shock_central_navier_stokes.yml
│       ├── argon_shock_muscl_euler.yml
│       ├── argon_shock_ppm_euler.yml
│       ├── argon_shock_muscl_navier_stokes.yml
│       ├── argon_shock_ppm_navier_stokes.yml
│       ├── run_serial.sh
│       ├── run_mpi.sh
│       └── plot_compare.py
└── tests/
    ├── test_gas.cpp
    ├── test_transport.cpp
    ├── test_normal_shock.cpp
    ├── test_mpi_decomp.cpp
    └── test_restart.cpp
```

## Documentation files to create

Create these files:

1. `docs/requirements.md`
2. `docs/architecture.md`
3. `docs/design_history.md`
4. `docs/tutorials.md`
5. `docs/verification_plan.md`

The design history should not be a transcript. It should be a note of record summarizing the design decisions from the planning conversation:

- The goal is to resolve a Navier-Stokes shock as a continuous viscous structure.
- PPM and MUSCL can behave like Euler shock-capturing methods and may interfere with the resolved shock layer.
- Central inviscid flux with physical viscosity/conduction is the baseline.
- MUSCL and PPM are included both with and without viscosity to demonstrate their effect.
- Diffusive timestep restrictions are numerical parabolic stability restrictions, not physical characteristic speeds.
- The grid is generated at runtime from YAML.
- MPI is included from the beginning using 1D contiguous decomposition.
- Restart files must be human-readable and editable.
- Transport comes from Cantera-generated fourth-order fits for pure argon and nitrogen.

## Coding standards

Use modern C++17 or C++20.

Prioritize:

1. correctness
2. clarity
3. reproducibility
4. modularity
5. MPI correctness
6. performance without premature complexity

Use assertions and clear runtime checks for:

- positive density
- positive pressure
- positive temperature
- positive viscosity
- positive conductivity
- valid CFL
- valid grid length and cell count

Avoid hiding numerical choices. Print them clearly.

## Final deliverable

Return a complete repository implementing SPLAY. The first working target is the target argon shock tutorial with five cases. The code must compile with CMake, run in serial, run with MPI, write CSV output, support editable restarts, and include plotting scripts and documentation.
