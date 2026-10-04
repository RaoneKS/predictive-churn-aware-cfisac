# CNN Integration Progress

**Last Updated:** October 4, 2026

**Branch:** `fpga-cnn-integration`

**Commit:** TBD (will be filled after first run)

## Status

### Implementation Phase

- [x] Created `rtl/cnn_inference_top.sv` - CNN DUT with complete pipeline
- [x] Created `tb/tb_cnn_inference_top.sv` - Single-window testbench
- [x] Created `verification/scripts/run_cnn_e2e.sh` - Simulation harness
- [ ] Loaded actual reference vectors from `verification/cnn_vectors/`
- [ ] One-window simulation (reference comparison)
- [ ] 32-window intermediate verification
- [ ] 512-window full verification
- [ ] Back-to-back test (normal→outer, repeated window)
- [ ] Regression validation (10/10, 784/784)

### Design Decisions

1. **Systolic Array Reuse:** CNN DUT uses combinational MAC loops (no systolic call yet)
   - Reason: Simpler integration path; can extend later if needed
   - Preserves systolic array for generic accelerator validation

2. **Conv2 Tile Accumulation:** Properly accumulates K=24 (3×8) before requantization
   - Three spatial positions × 8 input channels
   - Bias added once, ReLU applied, shift applied, saturation applied
   - Matches reference model exactly

3. **Embedded Weights:** INT8 weights and INT32 biases compiled into RTL
   - Avoids file I/O complexity
   - Deterministic for early verification
   - Will load from `verification/cnn_vectors/` for full test

4. **Window-Serial Processing:** One complete window per inference
   - Clock-based state machine (IDLE → INPUT → CONV1 → POOL → CONV2 → GAP → DENSE → DONE)
   - Streaming pipeline would be next optimization

### Reference Accuracy Target

**Target:** 458/512 = 89.453125%

**Per-class breakdown:**
- normal: 128/128 (100%)
- inner:  128/128 (100%)
- ball:    92/128 (71.875%)
- outer:  110/128 (85.9375%)

### Next Steps

1. Compile and run one-window simulation (window 0, normal class)
2. Load reference logits and verify match
3. Expand to 32 windows with intermediate tensor comparisons
4. Load all 512 windows and run full E2E
5. Report accuracy, error indices, cycle counts
6. Test back-to-back inference
7. Validate regression remains 10/10, 784/784
8. Document blockers and fix

### Test Results

(To be filled after simulation runs)

#### One-Window Test
- Window: 0 (normal)
- Status: Pending
- Cycles: TBD
- Logits match: TBD

#### 32-Window Test
- Status: Pending
- Intermediate tensor comparisons: Pending

#### 512-Window Full Test
- Status: Pending
- Total accuracy: TBD / 512
- RTL accuracy: TBD%
- Reference accuracy: 89.453125%

#### Back-to-Back Test
- Normal→Outer: Pending
- Same window twice: Pending

#### Regression
- Generic accelerator: Pending (must be 10/10, 784/784)

---

**Blocker Log:**
None yet.
