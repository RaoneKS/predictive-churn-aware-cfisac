#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/predictive-churn-aware-cfisac}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$ROOT/src" ]]; then
  echo "ERROR: project root not found: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT/tools"
cp "$HERE/cfisac_runtime_adapter.py" "$ROOT/tools/cfisac_runtime_adapter.py"
cp "$HERE/cfisac_live_dashboard.py" "$ROOT/tools/cfisac_live_dashboard.py"
cp "$HERE/diagnose_runtime.py" "$ROOT/tools/diagnose_runtime.py"

python3 -m pip install streamlit plotly pandas numpy pyyaml
python3 -m py_compile "$ROOT/tools/cfisac_runtime_adapter.py" "$ROOT/tools/cfisac_live_dashboard.py" "$ROOT/tools/diagnose_runtime.py"

echo
echo "Installed into: $ROOT/tools"
echo "Run diagnostics first:"
echo "  cd $ROOT && python3 tools/diagnose_runtime.py"
echo
echo "Then launch dashboard:"
echo "  cd $ROOT && streamlit run tools/cfisac_live_dashboard.py"
