#!/bin/bash
# CNN end-to-end verification script
# Compiles and runs comprehensive 512-window CNN verification

set -e

echo "=== CNN E2E Verification ==="
echo "Date: $(date)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_DIR="${REPO_ROOT}/sim"
WORK_DIR="${REPO_ROOT}/work"

mkdir -p "$SIM_DIR" "$WORK_DIR"

echo "Compiling CNN RTL and testbench..."
iverilog -g2012 \
  -I"${REPO_ROOT}/rtl" \
  "${REPO_ROOT}/rtl/pkg_accelerator.sv" \
  "${REPO_ROOT}/rtl/mac_unit.sv" \
  "${REPO_ROOT}/rtl/systolic_array.sv" \
  "${REPO_ROOT}/rtl/cnn_bias_requantize.sv" \
  "${REPO_ROOT}/rtl/cnn_systolic_adapter.sv" \
  "${REPO_ROOT}/rtl/cnn_inference_top.sv" \
  "${REPO_ROOT}/tb/tb_cnn_inference_top.sv" \
  -o "${SIM_DIR}/cnn_e2e.vvp" \
  2>&1 | tee "${SIM_DIR}/compile.log"

if [ $? -ne 0 ]; then
  echo "ERROR: Compilation failed"
  cat "${SIM_DIR}/compile.log"
  exit 1
fi

echo "Compilation successful"
echo ""
echo "Running CNN E2E simulation..."
vvp "${SIM_DIR}/cnn_e2e.vvp" 2>&1 | tee "${SIM_DIR}/cnn_e2e.log"

SIM_STATUS=${PIPESTATUS[0]}

if [ $SIM_STATUS -eq 0 ]; then
  echo ""
  echo "=== CNN E2E Test PASSED ==="
else
  echo ""
  echo "=== CNN E2E Test FAILED ==="
  echo "See ${SIM_DIR}/cnn_e2e.log for details"
  exit $SIM_STATUS
fi
