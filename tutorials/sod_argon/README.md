# Argon Sod Shock Tube

A 1D Riemann problem (Sod shock tube) in argon (γ = 5/3, monatomic ideal gas).
Used to verify the Euler inviscid solver against the exact analytic solution.

## Problem setup

| Quantity | Left (x ≤ 0.5 m) | Right (x > 0.5 m) |
|---|---|---|
| Density [kg/m³] | 1.000 | 0.125 |
| Pressure [Pa] | 1×10⁵ | 1×10⁴ |
| Velocity [m/s] | 0 | 0 |
| Temperature [K] | ≈ 480 | ≈ 384 |

Domain: [0, 1] m, diaphragm at x = 0.5 m.
Final time: t ≈ 4.90×10⁻⁴ s  (non-dimensional t·c_L/L ≈ 0.20).
N = 400 cells, MUSCL + Van Leer limiter, HLLC Riemann solver.

## Wave structure (argon, γ = 5/3)

- **Left rarefaction fan**: head at x ≈ 0.30 m, tail at x ≈ 0.47 m
- **Contact discontinuity**: x ≈ 0.63 m  (u* ≈ 266 m/s)
- **Right shock**: x ≈ 0.79 m  (p* ≈ 29.4 kPa)

## Running

```bash
# From repo root
bash tutorials/sod_argon/run.sh
```

Save plot:
```bash
python3 tutorials/sod_argon/plot_sod.py --save sod_argon.png
```

## Regression test

```bash
python3 tests/test_sod_regression.py
```

Expected L1 errors for N=400 MUSCL+HLLC (well within solver tolerance):

| Field | L1 error |
|---|---|
| ρ [kg/m³] | ~1.8×10⁻³ |
| u [m/s] | ~0.88 |
| p [Pa] | ~104 |
