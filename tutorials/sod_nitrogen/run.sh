#!/usr/bin/env bash
# Run the nitrogen Sod shock tube tutorial and optionally plot results.
#
# Usage (from repo root or any directory):
#   bash tutorials/sod_nitrogen/run.sh
#   bash tutorials/sod_nitrogen/run.sh --no-plot
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TUTORIAL_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

if [ ! -x "$SPLAY" ]; then
    echo "[sod_nitrogen/run.sh] splay binary not found — building..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DSPLAY_ENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
fi

cd "$REPO_ROOT"

echo "========================================"
echo "  Running: nitrogen_sod"
echo "========================================"
"$SPLAY" "$TUTORIAL_DIR/nitrogen_sod.yml"
echo ""
echo "Output: $REPO_ROOT/output/nitrogen_sod/"

if [ "$1" != "--no-plot" ]; then
    echo "Plotting Sod comparison..."
    python3 "$TUTORIAL_DIR/plot_sod.py"
fi
