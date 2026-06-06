#!/usr/bin/env bash
# Run all three basic test cases then open the time-evolution plot.
#
# Usage (from repo root or any directory):
#   bash tutorials/basic_tests/run.sh            # run + plot
#   bash tutorials/basic_tests/run.sh --no-plot  # run only
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TUTORIAL_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

# ── Auto-build if binary is missing ─────────────────────────────────────────
if [ ! -x "$SPLAY" ]; then
    echo "[run.sh] splay binary not found — building..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DSPLAY_ENABLE_MPI=OFF \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
    echo "[run.sh] Build complete."
    echo ""
fi

# Run from repo root so output/ lands at $REPO_ROOT/output/
cd "$REPO_ROOT"

CASES=(
    constant_state
    contact_discontinuity
    pressure_perturbation
)

for case in "${CASES[@]}"; do
    echo "========================================"
    echo "  Running: $case"
    echo "========================================"
    "$SPLAY" "$TUTORIAL_DIR/${case}.yml"
    echo ""
done

echo "All cases complete."
echo "Output: $REPO_ROOT/output/"

if [ "$1" != "--no-plot" ]; then
    echo "Plotting..."
    python3 "$TUTORIAL_DIR/plot.py"
fi
