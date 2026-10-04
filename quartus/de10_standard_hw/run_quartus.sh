#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
quartus_sh_bin="${QUARTUS_SH:-/home/raone/intelFPGA/23.1std/quartus/bin/quartus_sh}"
cd "$project_dir"
python3 "$project_dir/../../verification/scripts/prepare_de10_model_data.py"
"$quartus_sh_bin" --flow compile de10_standard_hw
