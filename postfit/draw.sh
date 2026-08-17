#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PLOT_DIR="$PROJECT/scripts/plot"
OUTPUT_DIR="$PROJECT/results/plot"
source "$PROJECT/config/gvv_env.sh"
ROOT_BIN=${ROOT_BIN:-$ROOTSYS/bin/root}

if [[ $# -eq 0 ]]; then
    PROJECTION="$PROJECT/results/projection0.root"
else
    PROJECTION=$1
    if [[ $PROJECTION != /* ]]; then
        PROJECTION="$PWD/$PROJECTION"
    fi
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [projection.root]" >&2
    exit 2
fi
if [[ ! -r $PROJECTION ]]; then
    echo "[GVV draw] Projection is missing or unreadable: $PROJECTION" >&2
    exit 3
fi
if [[ ! -x $ROOT_BIN ]]; then
    echo "[GVV draw] ROOT executable is unavailable: $ROOT_BIN" >&2
    exit 4
fi

mkdir -p "$OUTPUT_DIR"

"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection_2_3.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection_detailed\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection_components.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection_components\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/draw_angular_moments.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/angular_moments\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/draw_angular_moments_odd.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/angular_moments_odd_diagnostic\")"

echo "[GVV draw] Projection and moment figures written under $OUTPUT_DIR/"
