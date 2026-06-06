#!/usr/bin/env bash
# Run all five argon shock tutorial cases with MPI.
# Usage: ./run_mpi.sh [nranks]
set -e

NP=${1:-4}
SPLAY=../../build/splay
TUTORIAL_DIR=$(dirname "$0")

cases=(
    argon_shock_central_navier_stokes
    argon_shock_muscl_euler
    argon_shock_ppm_euler
    argon_shock_muscl_navier_stokes
    argon_shock_ppm_navier_stokes
)

for case in "${cases[@]}"; do
    echo "========================================"
    echo "  Running: $case  (np=$NP)"
    echo "========================================"
    mpirun -np $NP $SPLAY "$TUTORIAL_DIR/${case}.yml"
done

echo ""
echo "All cases complete.  Output in: output/"
