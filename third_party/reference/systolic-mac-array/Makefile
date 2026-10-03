# Systolic MAC Array - simulation makefile (Icarus Verilog)
#
#   make        # compile + run the original self-checking testbench
#   make sv     # compile + run the SystemVerilog constrained-random suite
#   make both   # run both suites back to back
#   make wave   # run, then open the waveform in GTKWave
#   make lint   # lint with Verilator (if installed)
#   make clean

IVERILOG ?= iverilog
VVP      ?= vvp
GTKWAVE  ?= gtkwave
VERILATOR?= verilator

RTL     := rtl/pe.v rtl/systolic_array.v

TB      := tb/tb_systolic.v
OUT     := sim/tb
VCD     := sim/dump.vcd

TB_SV   := tb/tb_systolic_sv.sv
OUT_SV  := sim/tb_sv
VCD_SV  := sim/dump_sv.vcd

.PHONY: all sim sv both wave wave-sv lint clean

all: sim

# ---------------------------------------------------------------- base suite
sim: $(OUT)
	$(VVP) $(OUT)

$(OUT): $(RTL) $(TB)
	@mkdir -p sim
	$(IVERILOG) -g2012 -o $(OUT) $(RTL) $(TB)

# ------------------------------------------- SystemVerilog constrained-random
# 6 directed corner cases + 200 constrained-random matrices across the full
# int8 range, with functional coverage bins. Fails on a mismatch OR on any
# uncovered bin -- a suite that reports PASS while silently never exercising
# a case is worse than no suite at all.
sv: $(OUT_SV)
	$(VVP) $(OUT_SV)

$(OUT_SV): $(RTL) $(TB_SV)
	@mkdir -p sim
	$(IVERILOG) -g2012 -o $(OUT_SV) $(RTL) $(TB_SV)

both: sim sv

# ------------------------------------------------------------------- waves
wave: sim
	$(GTKWAVE) $(VCD) &

wave-sv: sv
	$(GTKWAVE) $(VCD_SV) &

lint:
	$(VERILATOR) --lint-only -Wall $(RTL)

clean:
	rm -f $(OUT) $(OUT_SV) $(VCD) $(VCD_SV)
