#!/usr/bin/env bash
# Run all fine (N=2000) argon shock cases.
# Each case takes a few minutes.
#
# Usage (from repo root or any directory):
#   bash tutorials/argon_target_shock/fine/run.sh
#   bash tutorials/argon_target_shock/fine/run.sh --no-plot
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FINE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

if [ ! -x "$SPLAY" ]; then
    echo "[fine/run.sh] splay binary not found — building..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DSPLAY_ENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
fi

cd "$REPO_ROOT"

CASES=(
    argon_shock_central_navier_stokes
    argon_shock_ppm_euler
    argon_shock_muscl_euler
    argon_shock_ppm_euler_flatten
    argon_shock_muscl_euler_flatten
    argon_shock_ppm_navier_stokes
    argon_shock_muscl_navier_stokes
    argon_shock_ppm_navier_stokes_flatten
    argon_shock_muscl_navier_stokes_flatten
)

for case in "${CASES[@]}"; do
    echo "========================================"
    echo "  Running: $case"
    echo "========================================"
    "$SPLAY" "$FINE_DIR/${case}.yml"
    echo ""
done

echo "Fine cases complete.  Output: $REPO_ROOT/output/"

if [ "$1" != "--no-plot" ]; then
    TUTORIAL_DIR="$(dirname "$FINE_DIR")"
    echo "Plotting resolved shock comparison..."
    python3 "$TUTORIAL_DIR/plot_resolved.py"
fi
