#!/bin/bash
# Run CNN end-to-end verification

set -e

echo "=== CNN E2E Verification ==="
echo "Date: $(date)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"

SIM_DIR="sim"
mkdir -p "$SIM_DIR"

echo ""
echo "Compiling CNN testbench..."
iverilog -g2012 \
  -I rtl \
  rtl/pkg_accelerator.sv \
  rtl/systolic_array.sv \
  rtl/quantize_unit.sv \
  rtl/cnn_bias_requantize.sv \
  rtl/mac_unit.sv \
  rtl/cnn_inference_top.sv \
  tb/tb_cnn_inference_top.sv \
  -o "$SIM_DIR/cnn_e2e.vvp" \
  2>&1 | tee "$SIM_DIR/compile.log"

if [ $? -ne 0 ]; then
  echo "ERROR: Compilation failed"
  exit 1
fi

echo ""
echo "Running CNN simulation..."
vvp "$SIM_DIR/cnn_e2e.vvp" -n 2>&1 | tee "$SIM_DIR/cnn_e2e.log"

if [ $? -ne 0 ]; then
  echo "ERROR: Simulation failed"
  exit 1
fi

echo ""
echo "=== CNN E2E Test Complete ==="
