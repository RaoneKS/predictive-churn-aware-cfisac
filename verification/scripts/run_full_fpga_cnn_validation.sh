#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== FPGA CNN FULL VALIDATION ==="
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo

if [ "$(git branch --show-current)" != "fpga-cnn-integration" ]; then
  echo "ERROR: must run on fpga-cnn-integration"
  exit 2
fi

echo "=== 1. Authoritative reference ==="
python3 verification/scripts/check_cnn_reference.py

echo
echo "=== 2. Generic accelerator regression ==="
./verification/scripts/run_icarus_regression.sh

echo
echo "=== 3. CNN end-to-end RTL regression ==="
bash verification/scripts/run_cnn_e2e.sh

echo
echo "=== 4. DE10 Quartus implementation ==="
bash quartus/de10_standard_hw/run_quartus.sh

echo
echo "=== VALIDATION FLOW COMPLETED ==="
echo "Final commit: $(git rev-parse HEAD)"
