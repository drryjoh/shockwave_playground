#!/usr/bin/env bash
# Run all coarse (N=200) argon shock cases.
# Completes in a few minutes.
#
# Usage (from repo root or any directory):
#   bash tutorials/argon_target_shock/coarse/run.sh
#   bash tutorials/argon_target_shock/coarse/run.sh --no-plot
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
COARSE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

if [ ! -x "$SPLAY" ]; then
    echo "[coarse/run.sh] splay binary not found — building..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DSPLAY_ENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
fi

cd "$REPO_ROOT"

CASES=(
    argon_shock_ppm_euler_quick
    argon_shock_muscl_euler_quick
    argon_shock_ppm_euler_flatten_quick
    argon_shock_muscl_euler_flatten_quick
    argon_shock_ppm_ns_quick
    argon_shock_muscl_ns_quick
    argon_shock_ppm_ns_flatten_quick
    argon_shock_muscl_ns_flatten_quick
)

for case in "${CASES[@]}"; do
    echo "========================================"
    echo "  Running: $case"
    echo "========================================"
    "$SPLAY" "$COARSE_DIR/${case}.yml"
    echo ""
done

echo "Coarse cases complete.  Output: $REPO_ROOT/output/"

if [ "$1" != "--no-plot" ]; then
    TUTORIAL_DIR="$(dirname "$COARSE_DIR")"
    echo "Plotting flatten comparison..."
    python3 "$TUTORIAL_DIR/plot_flatten.py"
fi
