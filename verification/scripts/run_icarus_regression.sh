#!/usr/bin/env bash
# Generate all directed vectors and run the complete Icarus regression.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
work_dir="${REGRESSION_WORK_DIR:-/tmp/fpga-ai-icarus-regression}"

mkdir -p "$work_dir"
"$python_bin" "$repo_dir/verification/scripts/test_generator.py" \
  --test all --output-dir "$work_dir/vectors"

iverilog -g2012 -s tb_inference_accelerator -o "$work_dir/sim.out" \
  "$repo_dir/rtl/pkg_accelerator.sv" \
  "$repo_dir/rtl/mac_unit.sv" \
  "$repo_dir/rtl/systolic_array.sv" \
  "$repo_dir/rtl/sram_controller.sv" \
  "$repo_dir/rtl/axi_stream_input.sv" \
  "$repo_dir/rtl/axi_stream_output.sv" \
  "$repo_dir/rtl/quantize_unit.sv" \
  "$repo_dir/rtl/layer_ctrl_fsm.sv" \
  "$repo_dir/rtl/inference_accelerator_top.sv" \
  "$repo_dir/tb/tb_inference_accelerator.sv"

for test_dir in "$work_dir"/vectors/test_*; do
  vvp "$work_dir/sim.out" +test_dir="$test_dir"
done
