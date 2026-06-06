# SPLAY Tutorials

## Target tutorial: argon shock

Directory:

```text
tutorials/argon_target_shock/
```

## Physical setup

```text
L = 7.600403652e-4 cm
N = 2000
M = 5.03
shock location = 3.0e-6 m
initial tanh thickness = 5.0e-8 m
final time = 7.433391691866941e-07 s
CFL = 0.01
```

Right/pre-shock state:

```text
P = 5.0e5 Pa
T = 300.0 K
u = 0.0 m/s
```

Left/post-shock state:

```text
P = 1.56880625e7 Pa
T = 2632.24759509 K
u = 1168.9 m/s
```

Left boundary is fixed supersonic inflow at the post-shock state. Right boundary is outflow.

## Required cases

### Central viscous Navier-Stokes baseline

```yaml
case_name: argon_shock_central_navier_stokes
gas: argon
solver:
  viscous_terms: true
  inviscid_scheme: central
  riemann_solver: none
  limiter: none
```

### MUSCL Euler

```yaml
case_name: argon_shock_muscl_euler
gas: argon
solver:
  viscous_terms: false
  inviscid_scheme: muscl
  riemann_solver: hllc
  limiter: minmod
```

### PPM Euler

```yaml
case_name: argon_shock_ppm_euler
gas: argon
solver:
  viscous_terms: false
  inviscid_scheme: ppm
  riemann_solver: hllc
  limiter: ppm
```

### MUSCL Navier-Stokes

```yaml
case_name: argon_shock_muscl_navier_stokes
gas: argon
solver:
  viscous_terms: true
  inviscid_scheme: muscl
  riemann_solver: hllc
  limiter: minmod
```

### PPM Navier-Stokes

```yaml
case_name: argon_shock_ppm_navier_stokes
gas: argon
solver:
  viscous_terms: true
  inviscid_scheme: ppm
  riemann_solver: hllc
  limiter: ppm
```

## Example commands

Generate transport fits:

```bash
python tools/generate_transport_fits.py --gas argon --tmin 200 --tmax 5000 --samples 400 --output include/splay/transport_coeffs.hpp
```

Configure and build:

```bash
cmake -S . -B build -D SPLAY_ENABLE_MPI=ON -D SPLAY_ENABLE_TBB=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run serial:

```bash
./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml
```

Run MPI:

```bash
mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml
```

Restart:

```bash
mpirun -np 4 ./build/splay tutorials/argon_target_shock/argon_shock_central_navier_stokes.yml --restart output/argon_shock_central_navier_stokes/restart/step_1000
```

Plot comparison:

```bash
python tutorials/argon_target_shock/plot_compare.py output/argon_target_shock
```

