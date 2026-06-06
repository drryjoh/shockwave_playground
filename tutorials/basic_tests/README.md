# SPLAY Tutorial: Basic Test Cases

Three minimal test cases for verifying solver correctness on well-understood problems before running the full argon shock tutorial.

## Cases

| File | Init type | What it tests |
|---|---|---|
| `constant_state.yml` | tanh (L=R) | Uniform state preserved exactly — zero residual |
| `contact_discontinuity.yml` | tanh | Density jump convects at flow speed, pressure stays flat |
| `pressure_perturbation.yml` | gaussian_perturbation | Acoustic pulse splits into two symmetric waves at ±c |

### Constant state

Uniform supersonic argon flow (rho≈1 kg/m³, M=1.5, p=101325 Pa). The residual should be identically zero at every step. Any non-zero residual indicates a bug in the flux computation or boundary conditions.

### Contact discontinuity

rho_L=1 kg/m³ and rho_R=2 kg/m³ separated by a tanh interface, with equal pressure (101325 Pa) and equal velocity (617 m/s) on both sides. The density jump should convect rightward at 617 m/s without generating pressure waves. Checks that the Riemann solver correctly decouples contact waves from acoustic waves.

### Small pressure perturbation

Fluid at rest (u=0) with a narrow Gaussian pressure bump of fractional amplitude 1e-4 centred at x=0.5 m:

```
p(x,0) = p0 * (1 + 1e-4 * exp(-((x - 0.5)/0.05)^2))
```

By linear acoustics the pulse should split into two equal half-amplitude waves travelling at ±c ≈ ±411 m/s. At t=8e-4 s the peaks should appear near x≈0.17 m and x≈0.83 m, each with dp/p0 ≈ 5e-5.

## Running

### One-shot: run all cases and plot

```bash
bash tutorials/basic_tests/run.sh
```

`run.sh` will auto-build the solver if needed, run all three cases, then open the plot. Output CSVs go to `output/<case_name>/snapshots/`.

```bash
bash tutorials/basic_tests/run.sh --no-plot   # skip the plot
```

### Run and plot separately

```bash
# Build (if not already done)
cmake -S . -B build -D SPLAY_ENABLE_MPI=OFF -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run individual cases
./build/splay tutorials/basic_tests/constant_state.yml
./build/splay tutorials/basic_tests/contact_discontinuity.yml
./build/splay tutorials/basic_tests/pressure_perturbation.yml

# Plot (requires matplotlib + numpy)
python3 tutorials/basic_tests/plot.py

# Save to file instead of opening a window
python3 tutorials/basic_tests/plot.py --save basic_tests.png
```

## Expected output

**Constant state** — `|R_rho|` should print as `0.0000e+00` at every log step.

**Contact discontinuity** — `rho` range should stay at `[1.0001, 2.0001]` and `p` range should stay flat at `[1.013e+05, 1.013e+05]`.

**Pressure perturbation** — `rho` range stays near `[1.0000, 1.0001]`; the tiny pressure variation is only visible in the snapshot CSVs, not in the console summary.

## Notes on the `gaussian_perturbation` init type

The `pressure_perturbation.yml` case uses a new initialization type added alongside these tutorials. Config syntax:

```yaml
initialization:
  type: gaussian_perturbation
  location: 0.5       # centre x0 (m)
  sigma: 0.05         # Gaussian half-width (m)
  amplitude: 1.0e-4   # fractional pressure amplitude
  left:               # base uniform state
    pressure: 101325.0
    temperature: 486.8
    velocity: 0.0
```

Density is held constant at `rho0 = p0 / (R * T0)`. Only pressure (and therefore temperature) varies with the Gaussian profile. No `right:` block or `thickness:` field is needed for this type.

## Periodic BCs

Periodic boundary conditions are not currently implemented. A sine-wave advection test would require adding `BCType::Periodic` support.
