# DE10-Standard Quartus build

This project targets the Intel DE10-Standard Cyclone V SoC device
`5CSXFC6D6F31C6` using Quartus Prime Standard 23.1. It synthesizes the
accelerator RTL top level directly; a later board-integration phase can add a
pin-assigned wrapper without changing this baseline accelerator project.
Its transaction-level top-level ports are intentionally assigned as virtual
pins, preventing the 577 RTL ports from being mistaken for physical board I/O
while still retaining the core for fitting and timing analysis.

This is a core implementation result, not a board-programmable image: virtual
pins mean I/O delays are not characterized. A physical DE10 wrapper and pin
constraints are required for hardware timing sign-off and programming.

## Build

Run from the repository root:

```bash
./quartus/de10_standard/run_quartus.sh
```

The script defaults to the local Standard 23.1 installation at
`/home/raone/intelFPGA/23.1std/quartus/bin/quartus_sh`. To use another
installation, set `QUARTUS_SH` to that executable. The command runs Quartus
analysis and synthesis, fitter, assembler, and timing analysis in order.

The only timing constraint is a 10.000 ns (100 MHz) clock on `clk`, specified
in `de10_standard.sdc`.

## Regression

Generate vectors, compile the simulation, and run every directed test:

```bash
verification/scripts/run_icarus_regression.sh
```

Set `PYTHON_BIN` when NumPy is installed in a virtual environment, for example
`PYTHON_BIN=/tmp/fpga-ai-venv/bin/python verification/scripts/run_icarus_regression.sh`.

Quartus-generated `db`, `incremental_db`, `output_files`, and log artifacts
are intentionally ignored by Git. Consult the generated `*.rpt` files after
a build for utilization and timing results.
