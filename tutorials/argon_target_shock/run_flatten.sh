#!/usr/bin/env bash
# Run all flattening-comparison quick cases then show the comparison plot.
#
# Usage (from repo root or any directory):
#   bash tutorials/argon_target_shock/run_flatten.sh
#   bash tutorials/argon_target_shock/run_flatten.sh --no-plot
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TUTORIAL_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

# ── Auto-build if binary is missing ─────────────────────────────────────────
if [ ! -x "$SPLAY" ]; then
    echo "[run_flatten.sh] splay binary not found — building..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DSPLAY_ENABLE_MPI=OFF \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
    echo "[run_flatten.sh] Build complete."
    echo ""
fi

cd "$REPO_ROOT"

CASES=(
    # Euler baselines (already exist from run_serial.sh --quick)
    argon_shock_ppm_euler_quick
    argon_shock_muscl_euler_quick
    # Euler + flatten
    argon_shock_ppm_euler_flatten_quick
    argon_shock_muscl_euler_flatten_quick
    # NS baselines
    argon_shock_ppm_ns_quick
    argon_shock_muscl_ns_quick
    # NS + flatten (PeleC-like)
    argon_shock_ppm_ns_flatten_quick
    argon_shock_muscl_ns_flatten_quick
)

for case in "${CASES[@]}"; do
    echo "========================================"
    echo "  Running: $case"
    echo "========================================"
    "$SPLAY" "$TUTORIAL_DIR/coarse/${case}.yml"
    echo ""
done

echo "All cases complete."
echo "Output: $REPO_ROOT/output/"

if [ "$1" != "--no-plot" ]; then
    echo "Plotting..."
    python3 "$TUTORIAL_DIR/plot_flatten.py"
fi
