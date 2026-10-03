#!/usr/bin/env bash
# Reproducible Quartus Prime Standard 23.1 build for the DE10-Standard target.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
quartus_sh_bin="${QUARTUS_SH:-/home/raone/intelFPGA/23.1std/quartus/bin/quartus_sh}"

if [[ ! -x "$quartus_sh_bin" ]]; then
  echo "ERROR: Quartus shell not found: $quartus_sh_bin" >&2
  echo "Set QUARTUS_SH to the Quartus Prime Standard 23.1 quartus_sh executable." >&2
  exit 1
fi

cd "$project_dir"
"$quartus_sh_bin" --flow compile de10_standard
