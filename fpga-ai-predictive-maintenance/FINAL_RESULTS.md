# Final Results — FPGA-Based Edge-AI Predictive Maintenance

## Functional Verification

### RTL simulation
- Normal input: Class 0 — PASS
- Outer-race input: Class 3 — PASS
- Testbench: BEARING_CNN_DEMO_TEST PASSED
- Final simulation latency: 8,969 cycles

### Physical DE10-Standard
- Normal mode: Class 0 — PASS
- Outer-race mode: Class 3 — PASS
- Heartbeat indicator: PASS
- Mode indicator: PASS
- Completion indicator: PASS

## FPGA Implementation

- Device: Intel Cyclone V SoC 5CSXFC6D6F31C6
- Board clock: 50 MHz
- Post-fit Fmax: 59.01 MHz
- Worst-case setup slack: +3.054 ns
- Worst-case hold slack: +0.215 ns
- DSP blocks: 20 / 112 (18%)
- Registers: 30,766
- Quartus compilation: 0 errors
- FPGA programming: successful

## Accelerator

- INT8 inference
- INT8 × INT8 → INT32 accumulation
- Conv1D
- ReLU
- MaxPool1D
- Global Average Pooling
- Fully connected classifier
- Time-multiplexed multiplier datapaths
- Pipelined Conv1 and Conv2 datapaths
- FPGA-resident CWRU demonstration samples

## Demonstrated Classes

- Class 0: Normal
- Class 3: Outer-race fault

## Hardware Demonstration

SW1 selects the demonstration input and SW0 starts inference.

The current demonstrator uses preloaded CWRU vibration samples stored in FPGA memory. It does not yet include a live physical vibration-sensor interface.

