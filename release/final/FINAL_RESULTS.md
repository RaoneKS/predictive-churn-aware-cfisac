# FPGA-Based Edge-AI Accelerator for Industrial Predictive Maintenance

## Final Verification Status

### Neural-network simulation
- Normal input: class 0 — PASS
- Outer-fault input: class 3 — PASS
- Testbench: BEARING_CNN_DEMO_TEST PASSED
- Final simulated latency: 8,969 clock cycles

### Physical DE10-Standard validation
- Normal mode (SW1=0): class 0 — PASS
- Outer-fault mode (SW1=1): class 3 — PASS
- Heartbeat LED: PASS
- Inference-complete indicator: PASS
- Mode indicator: PASS

### FPGA implementation
- Target device: Intel Cyclone V SoC, 5CSXFC6D6F31C6
- Quartus Prime Lite/Standard 25.1
- Board clock: 50 MHz
- Post-fit Fmax: 59.01 MHz
- Worst-case setup slack: +3.054 ns
- Worst-case hold slack: +0.215 ns
- DSP blocks: 20 / 112 (18%)
- Registers: 30,766
- Full Quartus compilation: PASS
- FPGA programming: PASS

## Accelerator architecture

- INT8 neural-network inference
- Conv1D layers
- INT8 x INT8 -> INT32 accumulation
- ReLU and requantization
- MaxPool1D
- Global average pooling
- Fully connected classifier
- Time-multiplexed multiplier datapaths
- Pipelined Conv1 and Conv2 datapaths
- Cyclone V FPGA implementation
- Board-level switch/LED demonstration

## Demonstrated classes

- Class 0: Normal
- Class 3: Outer-race fault

## Important implementation note

The final design was timing-closed for the 50 MHz board clock. The measured worst-case post-fit Fmax is 59.01 MHz under the reported slow 1100 mV, 85 C timing model.
