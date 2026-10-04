# CNN Integration Progress

**Status:** Integration code prepared; local Icarus verification pending

**Last Updated:** October 5, 2026

**Branch:** `fpga-cnn-integration`

**Commit:** `d54e23ff0b8f03171ed2a02cb931f74339543dc7`

## Architecture Decision

### Conflict Identified
The validated generic accelerator architecture cannot support CNN Conv2 tile accumulation (3 partial K=8 sums accumulated in INT32 before bias/ReLU/shift) without modifying protected modules:
- `systolic_array.sv` - outputs final column sums immediately
- `quantize_unit.sv` - applies ReLU/shift/saturate independently per output
- `layer_ctrl_fsm.sv` - designed for layer-descriptor tiling, not multi-spatial scheduling

### Solution (Option C)
**Create isolated CNN-specific components that instantiate systolic_array directly:**
- `rtl/cnn_systolic_adapter.sv` - thin wrapper around systolic_array for CNN scheduling
- `rtl/cnn_inference_top.sv` - manifest-aligned scalar reference CNN pipeline; systolic adapter is instantiated as the isolated hardware integration point
- New testbench loading authoritative vectors from `verification/cnn_vectors/` and `ml/artifacts/`

**Preservation guarantee:**
- Generic accelerator remains completely untouched
- Generic regression must remain 10/10 PASS, 784/784 outputs
- Systolic array reused at its core PE level

## Implementation

### Files / verification updates

1. **rtl/cnn_systolic_adapter.sv**
   - Direct instantiation of `systolic_array.sv`
   - Pack/unpack interface for CNN im2col scheduling
   - Single-tile MAC execution (no modification of systolic array)

2. **rtl/cnn_inference_top.sv**
   - Complete CNN pipeline: Conv1 → Pool → Conv2 → GAP → Classifier
   - Window-serial processing (one 256-sample window per inference)
   - Proper Conv2 tile accumulation (3 × K=8 partial sums)
   - Testbench-provided weight/bias loading

3. **tb/tb_cnn_inference_top.sv**
   - Instantiates CNN DUT
   - Loads input windows and reference data
   - Compares logits and predictions
   - Reports per-class accuracy

4. **verification/scripts/run_cnn_e2e.sh**
   - Compiles CNN RTL + testbench
   - Runs iverilog simulation
   - Captures results

## Design Specifications

### Conv1
- Input: 256 INT8 samples (scale 0.0625)
- Kernel: K=5, OC=8
- Output: 256 INT8 (scale 0.125)
- ReLU: yes
- Shift: 8

### MaxPool1D
- Kernel: 2, Stride: 2
- Input: 256, Output: 128

### Conv2
- Input: 128 INT8 samples, 8 channels (scale 0.125)
- Kernel: K=3, IC=8, OC=8
- **Critical:** Three K=8 tiles per spatial position accumulate in INT32 before bias/ReLU/shift
- Output: 128 INT8 (scale 0.25)
- ReLU: yes
- Shift: 8

### GlobalAveragePool
- Input: 128 spatial, 8 channels
- Output: 8 channels (scale 0.25)
- Signed truncation toward zero: `avg = (sum >= 0) ? (sum >>> 7) : -((-sum) >>> 7)`

### Classifier
- Input: 8 INT8 (scale 0.25)
- Output: 4 classes (scale 0.0625, int32)
- Bias: included
- Shift: 5
- ReLU: no

## Authoritative Reference

**Expected Accuracy:** 458/512 = 89.453125%

**Per-class breakdown:**
- normal: 128/128 (100%)
- inner:  128/128 (100%)
- ball:    92/128 (71.875%)
- outer:  110/128 (85.9375%)

## Verification Requirements

✓ Load authoritative weights from:
  - `ml/artifacts/conv1_weights_kxoc_int8.npy`
  - `ml/artifacts/conv1_bias_accumulator_int32.npy`
  - `ml/artifacts/conv2_weights_kxoc_int8.npy`
  - `ml/artifacts/conv2_bias_accumulator_int32.npy`
  - `ml/artifacts/classifier_weights_kxoc_int8.npy`
  - `ml/artifacts/classifier_bias_accumulator_int32.npy`

✓ Load test vectors from:
  - `verification/cnn_vectors/input_int8.npy` (512 windows)
  - `verification/cnn_vectors/reference_logits_int8.npy`
  - `verification/cnn_vectors/reference_predictions.npy`
  - `verification/cnn_vectors/reference_labels.npy`

✓ Execute all 512 windows

✓ Compare outputs:
  - All four logits per window
  - Predictions against reference

✓ Intermediate verification (first 32 windows):
  - Conv1 output
  - MaxPool output
  - Conv2 output
  - GAP output
  - Classifier accumulator

✓ Back-to-back tests:
  - Normal window → Outer window without reset
  - Same window twice without reset

✓ Report:
  - Total windows processed
  - Exact logit matches (X/512)
  - Prediction matches (X/512)
  - Per-class accuracy
  - Error indices

## Test Results

(Pending first run)

### Compilation
- Status: Pending
- Log: `sim/compile.log`

### Single Window Test
- Status: Pending
- Cycles: TBD

### Full 512-Window Test
- Status: Pending
- Total matches: TBD / 512
- Predictions: TBD / 512
- Accuracy: TBD%
- Reference: 89.453125%

### Back-to-Back Tests
- Normal→Outer: Pending
- Same window twice: Pending

### Generic Regression
- Status: Pending (must remain 10/10, 784/784)

## Known Issues / Blockers

Local simulation has not been executed from this environment. The scalar reference path is intentionally kept as the independent correctness baseline; the systolic adapter must be proven tile-equivalent before replacing stage execution with it.

## Next Steps

1. Run `./verification/scripts/run_cnn_e2e.sh` locally and capture compile/simulation status
2. If compilation fails: fix Icarus-compatible RTL
3. If simulation fails: debug vector loading/arithmetic
4. Verify exact 2048/2048 logits against authoritative vectors
5. Verify prediction agreement is 458/512
6. Verify first 32 windows and back-to-back cases
7. Run generic regression and document evidence
8. Only then replace scalar stage execution with the proven systolic adapter
9. Re-run CNN and generic regressions, then document final results
