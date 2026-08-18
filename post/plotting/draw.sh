#!/bin/bash

set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PLOT_DIR="$SCRIPT_DIR/macros"
OUTPUT_DIR="$SCRIPT_DIR/results"
source "$PROJECT/config/gvv_env.sh"
ROOT_BIN=${ROOT_BIN:-$ROOTSYS/bin/root}

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 results/projection-<tag>.root" >&2
    exit 2
fi
PROJECTION=$1
if [[ $PROJECTION != /* ]]; then
    PROJECTION="$PWD/$PROJECTION"
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
PROJECTION_NAME=$(basename -- "$PROJECTION")
OUTPUT_TAG=${PROJECTION_NAME#projection-}
OUTPUT_TAG=${OUTPUT_TAG%.root}
cd "$PROJECT"

"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection_2_3.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection-$OUTPUT_TAG\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection_detailed-$OUTPUT_TAG\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/Draw_projection_components.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/projection_components-$OUTPUT_TAG\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/draw_angular_moments.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/angular_moments-$OUTPUT_TAG\")"
"$ROOT_BIN" -l -b -q \
    "$PLOT_DIR/draw_angular_moments_odd.cxx(\"$PROJECTION\",\"$OUTPUT_DIR/angular_moments_odd_diagnostic-$OUTPUT_TAG\")"

echo "[GVV draw] Projection and moment figures written under $OUTPUT_DIR/"
