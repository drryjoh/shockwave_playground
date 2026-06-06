#!/usr/bin/env bash
# Run all five argon shock tutorial cases in serial.
#
# Usage (from this directory):
#   bash run_serial.sh          # full run (viscous cases are very long)
#   bash run_serial.sh --quick  # quick-test with reduced N and time
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TUTORIAL_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SPLAY="$BUILD_DIR/splay"

# ── Auto-build if binary is missing ─────────────────────────────────────────
if [ ! -x "$SPLAY" ]; then
    echo "[run_serial.sh] splay binary not found — building now..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DSPLAY_ENABLE_MPI=OFF \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
    echo "[run_serial.sh] Build complete."
    echo ""
fi

# ── Select cases ─────────────────────────────────────────────────────────────
if [ "$1" = "--quick" ]; then
    echo "=== QUICK-TEST MODE (reduced grid, short time) ==="
    cases=(
        argon_shock_muscl_euler_quick
        argon_shock_ppm_euler_quick
    )
else
    cases=(
        argon_shock_central_navier_stokes
        argon_shock_muscl_euler
        argon_shock_ppm_euler
        argon_shock_muscl_navier_stokes
        argon_shock_ppm_navier_stokes
    )
    echo "NOTE: Viscous cases (central_navier_stokes, muscl_navier_stokes,"
    echo "      ppm_navier_stokes) use an explicit parabolic dt ~3e-17 s."
    echo "      They require ~25 billion steps to reach t_end=7.4e-7 s."
    echo "      Use --quick for fast inviscid cases, or see docs/ for implicit"
    echo "      time integration options."
    echo ""
fi

# ── Run ──────────────────────────────────────────────────────────────────────
for case in "${cases[@]}"; do
    echo "========================================"
    echo "  Running: $case"
    echo "========================================"
    "$SPLAY" "$TUTORIAL_DIR/${case}.yml"
done

echo ""
echo "All cases complete.  Output in: $TUTORIAL_DIR/output/"
echo "Plot:  python $TUTORIAL_DIR/plot_compare.py $TUTORIAL_DIR/output/"
