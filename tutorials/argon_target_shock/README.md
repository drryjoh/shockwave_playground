# SPLAY Tutorial: Argon Target Shock (M ≈ 5.03)

This tutorial compares five distinct numerical configurations for a Mach-5.03
argon normal shock in a 1D domain.  The central goal is to contrast a
**physically-resolved Navier-Stokes shock** (continuous viscous layer) against
conventional **Euler shock-capturing** results (MUSCL and PPM) and hybrid modes.

## Case summary

| Case file                              | Scheme  | Limiter | Viscous | Purpose                         |
|----------------------------------------|---------|---------|---------|----------------------------------|
| `argon_shock_central_navier_stokes.yml`| Central | none    | YES     | **Baseline: resolved NS shock** |
| `argon_shock_muscl_euler.yml`          | MUSCL   | minmod  | NO      | Euler shock-capturing ref       |
| `argon_shock_ppm_euler.yml`            | PPM     | ppm     | NO      | Euler shock-capturing ref       |
| `argon_shock_muscl_navier_stokes.yml`  | MUSCL   | minmod  | YES     | Does MUSCL distort the NS layer?|
| `argon_shock_ppm_navier_stokes.yml`    | PPM     | ppm     | YES     | Does PPM  distort the NS layer? |

## Problem parameters

```
Gas        : Argon (gamma=5/3, calorically perfect)
Mach       : ~5.03 (shock-frame reference; see below)
Domain     : L = 7.600403652e-4 cm = 7.600403652e-6 m
Cells      : N = 2000
Ghost cells: 3

Left (post-shock) state  : P=15.69 MPa, T=2632 K, u=1168.9 m/s
Right (pre-shock) state  : P=0.5 MPa,  T=300 K,  u=0 m/s
Shock location           : x_s = 3 µm
Initial tanh thickness   : δ = 50 nm
Final time               : 7.43e-7 s
CFL                      : 0.01
```

### Frame convention note

The right (pre-shock) gas is at rest in the lab frame.  The shock moves
leftward into the quiescent gas.  The left boundary is fixed to the
post-shock (downstream) primitive state, which corresponds to a supersonic
inflow in the shock-attached frame.  In the lab frame the shock propagates at
approximately `u_shock = a1 * M1 ≈ 347 * 5.03 ≈ 1746 m/s` leftward.

## Quick-start guide

### 1 – Generate transport fits

```bash
# With Cantera installed:
python tools/generate_transport_fits.py \
    --gas both --tmin 200 --tmax 5000 --samples 400 \
    --output include/splay/transport_coeffs.hpp

# Without Cantera (uses Sutherland law, adequate for initial testing):
python tools/generate_transport_fits.py --no-cantera \
    --output include/splay/transport_coeffs.hpp
```

### 2 – Configure and compile

```bash
# Serial build (no MPI):
cmake -S . -B build -D SPLAY_ENABLE_MPI=OFF -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# MPI build:
cmake -S . -B build -D SPLAY_ENABLE_MPI=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# MPI + TBB:
cmake -S . -B build \
      -D SPLAY_ENABLE_MPI=ON \
      -D SPLAY_ENABLE_TBB=ON \
      -D CMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3 – Run in serial

```bash
# Single case:
./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml

# All five cases:
bash tutorials/argon_target_shock/run_serial.sh
```

### 4 – Run with MPI

```bash
# Four ranks:
mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml

# All five cases at np=4:
bash tutorials/argon_target_shock/run_mpi.sh 4
```

### 5 – Restart a run

After a successful run a restart directory is written to:

```
output/<case_name>/restart/step_<N>/
```

To restart:

```bash
# Serial:
./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml \
    --restart output/argon_shock_central_navier_stokes/restart/step_1000

# MPI (same nranks as original):
mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml \
    --restart output/argon_shock_central_navier_stokes/restart/step_1000
```

### 6 – Edit a restart file

Each restart directory contains:

- `metadata.yml` — step, time, gas, grid info (human readable, editable).
- `rank<N>_data.csv` — per-rank CSV with columns `x,rho,rhou,rhoE,u,p,T`.

You can edit any row of the CSV to modify the state and restart from the
edited file.  Ensure the conservative variables remain physically consistent
(positive density, pressure, temperature).

### 7 – Plot results

```bash
python tutorials/argon_target_shock/plot_compare.py output/
```

Produces `output/argon_shock_comparison.png` overlaying all five cases for
density, velocity, pressure, temperature, Mach, and viscosity.

## Why these five cases?

### Central viscous NS (baseline)

This is the physically correct calculation.  The central inviscid flux adds no
numerical dissipation beyond what a standard centred scheme provides.  The
physical viscosity and heat conduction from the gas are the only diffusive
mechanisms.  The result should converge to the Navier-Stokes shock profile as
the grid is refined.

### MUSCL/PPM Euler (reference)

With viscous terms off, MUSCL and PPM behave as Euler shock-capturing codes.
The shock is captured in ~2–5 cells by numerical dissipation from the limiter
and Riemann solver.  This is the conventional CFD shock resolution — fast and
robust, but the shock layer has no physical thickness.

### MUSCL/PPM Navier-Stokes (diagnostic)

With viscous terms on, the MUSCL/PPM limiters and HLLC Riemann solver compete
with (and in some cases overwhelm) the physical diffusion.  The shock structure
may be distorted or spuriously diffused compared to the central NS baseline.
These cases show whether limiter/Riemann-solver dissipation interferes with the
resolved viscous shock layer.

## Verification

Run the built-in verification tests:

```bash
cd build && ctest -V
```

Tests include:
- Gas model unit checks
- Transport coefficient evaluation
- Normal-shock Rankine-Hugoniot verification
- MPI decomposition consistency
- Restart round-trip consistency
