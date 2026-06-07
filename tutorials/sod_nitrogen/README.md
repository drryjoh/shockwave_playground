# Nitrogen Sod Shock Tube

A 1D Riemann problem (Sod shock tube) in nitrogen (γ = 1.4, diatomic ideal gas).
Used to verify the Euler inviscid solver against the exact analytic solution.

## Problem setup

| Quantity | Left (x ≤ 0.5 m) | Right (x > 0.5 m) |
|---|---|---|
| Density [kg/m³] | 1.000 | 0.125 |
| Pressure [Pa] | 1×10⁵ | 1×10⁴ |
| Velocity [m/s] | 0 | 0 |
| Temperature [K] | ≈ 337 | ≈ 270 |

Domain: [0, 1] m, diaphragm at x = 0.5 m.
Final time: t ≈ 5.34×10⁻⁴ s  (non-dimensional t·c_L/L ≈ 0.20).
N = 400 cells, MUSCL + Van Leer limiter, HLLC Riemann solver.

## Wave structure (nitrogen, γ = 1.4)

- **Left rarefaction fan**: head at x ≈ 0.30 m, tail at x ≈ 0.46 m
- **Contact discontinuity**: x ≈ 0.66 m  (u* ≈ 293 m/s)
- **Right shock**: x ≈ 0.82 m  (p* ≈ 30.3 kPa)

## Running

```bash
# From repo root
bash tutorials/sod_nitrogen/run.sh
```

Save plot:
```bash
python3 tutorials/sod_nitrogen/plot_sod.py --save sod_nitrogen.png
```

## Regression test

```bash
python3 tests/test_sod_regression.py
```

Expected L1 errors for N=400 MUSCL+HLLC:

| Field | L1 error |
|---|---|
| ρ [kg/m³] | ~1.6×10⁻³ |
| u [m/s] | ~0.94 |
| p [Pa] | ~101 |

## Note on γ

Nitrogen uses γ = 1.4, matching the *classic* Sod problem in non-dimensional
form (p_L/p_R = 10, ρ_L/ρ_R = 8). Argon uses γ = 5/3, which gives a slightly
stronger shock and narrower rarefaction for the same pressure ratio.
