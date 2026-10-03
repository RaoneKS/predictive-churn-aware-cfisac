# Phase 2 implementation record

## Scope

This record covers the existing `inference_accelerator_top` RTL compiled as a
DE10-Standard core target. No accelerator RTL was changed for Quartus
compatibility. A board wrapper, physical pin locations, and peripheral
interfaces are outside this phase.

## Tool and target

| Item | Value |
| --- | --- |
| Tool | Quartus Prime Standard 23.1std.1, Build 993 |
| FPGA | Cyclone V SoC `5CSXFC6D6F31C6` |
| Top level | `inference_accelerator_top` |
| Clock constraint | `clk`: 10.000 ns (100 MHz) |

## Reproducible implementation command

```bash
./quartus/de10_standard/run_quartus.sh
```

This invokes Quartus analysis/synthesis, fitter, assembler, and timing
analysis. Generated reports go in `output_files/`.

## Completed implementation result

The virtual-pin core baseline completed analysis/synthesis, fitting, assembly,
and timing analysis with zero errors. The assembler reported the expected
evaluation-mode warning that it could not generate programming files.

| Resource | Used | Available | Utilization |
| --- | ---: | ---: | ---: |
| Logic cells after synthesis | 7,542 | — | — |
| RAM blocks after fitting | 14 | 553 | 3% |
| DSP blocks after fitting | 65 | 112 | 58% |

At the slow 1100 mV, 85 C timing corner, the 100 MHz core constraint reported:

| Check | Slack |
| --- | ---: |
| Setup | +0.386 ns |
| Hold | +0.216 ns |

This is a core estimate, not I/O timing sign-off: the transaction/debug ports
are virtual, and Quartus reports expected unconstrained I/O setup/hold paths.

## Build issue encountered and resolution

The accelerator top level has 577 ports, exceeding the device's 288 physical
user I/O locations. An initial fit therefore stopped with:

```
Error (179000): Design requires 577 user-specified I/O pins -- too many to fit
in the 288 user I/O pin locations available in the selected device.
```

The project assigns the accelerator transaction/debug interfaces as virtual
pins. This preserves the RTL while enabling placement, routing, and core timing
analysis. Future DE10 integration must replace this baseline with a compact
physical interface and actual pin assignments.

## Functional regression

```bash
/tmp/fpga-ai-venv/bin/python verification/scripts/test_generator.py \
  --test all --output-dir /tmp/fpga-ai-vectors
iverilog -g2012 -s tb_inference_accelerator -o /tmp/fpga-ai-sim.out \
  rtl/pkg_accelerator.sv rtl/mac_unit.sv rtl/systolic_array.sv \
  rtl/sram_controller.sv rtl/axi_stream_input.sv rtl/axi_stream_output.sv \
  rtl/quantize_unit.sv rtl/layer_ctrl_fsm.sv rtl/inference_accelerator_top.sv \
  tb/tb_inference_accelerator.sv
for test_dir in /tmp/fpga-ai-vectors/test_*; do
  vvp /tmp/fpga-ai-sim.out +test_dir="$test_dir"
done
```

All ten directed tests passed: **784/784 expected output values matched**.

The vector generator requires NumPy. It was installed in the isolated temporary
environment `/tmp/fpga-ai-venv` for this run; it is not a repository change.
