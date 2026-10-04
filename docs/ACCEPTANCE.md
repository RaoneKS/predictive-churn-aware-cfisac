# FPGA CNN Integration — Acceptance Criteria

## 1. Scope

This document is the single authoritative acceptance specification for the CNN integration work.

Work ONLY on:

    fpga-cnn-integration

Do NOT modify `main`.

The final deliverable is a reproducible simulation + Quartus implementation of the INT8 bearing CNN using the generic accelerator architecture.

Physical DE10 execution is optional and must be reported separately from simulation results.

---

## 2. Baseline — Already Verified

The authoritative generic accelerator baseline is:

    ./verification/scripts/run_icarus_regression.sh

Required baseline result:

    10/10 tests PASS
    784/784 outputs matched

Do NOT replace this with older 792/792 figures.

The baseline must remain passing after CNN integration.

---

## 3. CNN Model

Use the existing CWRU bearing CNN definition and artifacts.

Classes:

    0 = normal
    1 = inner
    2 = ball
    3 = outer

Input:

    CWRU 12 kHz drive-end vibration
    window = 256 samples
    per-window zero mean / unit standard deviation

Network:

    Conv1D(1, 8, kernel=5)
    ReLU
    MaxPool1D(2)
    Conv1D(8, 8, kernel=3)
    ReLU
    GlobalAveragePooling1D
    Dense(8, 4)

Quantized deployment:

    INT8 activations
    INT8 weights
    INT32 accumulation

Scales:

    input       = 0.0625
    Conv1       = 0.125
    Conv2       = 0.25
    GAP         = 0.25
    logits      = 0.0625

Layer parameters:

    Conv1:
        Kphysical = 8
        output channels = 8
        shift = 8
        ReLU = enabled

    Conv2:
        K = 24
        output channels = 8
        shift = 8
        ReLU = enabled

    Classifier:
        K = 8
        logical outputs = 4
        physical outputs = 8
        shift = 5
        ReLU = disabled

GAP signed truncation must match:

    positive: sum >>> 7
    negative: -((-sum) >>> 7)

---

## 4. Bias Requirement

The software reference contains INT32 bias.

The RTL must correctly implement bias if the CNN integration requires it.

Do NOT silently omit bias.

Do NOT claim bias folding unless it is mathematically demonstrated and verified.

For each affected layer:

    INT32 accumulation
    + INT32 bias
    -> requantization
    -> optional ReLU
    -> INT8

---

## 5. Conv2 Tiling Requirement

Conv2 has:

    K = 24

The 8x8 accelerator cannot treat this as a single K=8 operation.

It must use:

    K tile 0 = 8
    K tile 1 = 8
    K tile 2 = 8

The three INT32 partial sums must be combined BEFORE:

    bias
    shift
    ReLU
    INT8 saturation

Do NOT requantize each K tile independently.

---

## 6. Fresh Reference Requirement

Generate the reference vectors from the actual repository code and actual 512-window holdout.

Do not rely solely on stale generated files.

The fresh reference must reproduce:

    458 / 512
    = 89.453125%

INT8 reference accuracy.

If it does not reproduce 462/512:

    STOP RTL equivalence analysis.

Investigate the dataset/windowing/normalization/model/artifact mismatch first.

---

## 7. RTL vs Reference

The RTL must execute all 512 holdout windows.

Do not test only a subset and extrapolate.

Required comparisons:

### First 32 windows

Compare all intermediate tensors:

    Conv1
    MaxPool
    Conv2
    GAP
    Dense/logits

Comparisons must be exact or use explicitly documented integer tolerance where mathematically unavoidable.

### Remaining 480 windows

At minimum compare:

    final logits
    predicted class

All 512 windows must actually execute.

---

## 8. Error Cross-Check

The fresh reference is expected to have:

    54 reference error windows

if it reproduces 458/512.

Cross-check the RTL/reference error indices.

Report:

    reference error indices
    RTL error indices
    common errors
    RTL-only errors
    reference-only errors

Also provide per-class error counts where practical.

Do NOT assume that 50 errors exist without actually calculating them.

---

## 9. Accuracy Definitions

Keep these separate:

### Reference accuracy

Fresh software INT8 model vs true labels.

Expected:

    458/512 = 89.453125%

### RTL simulation accuracy

RTL predicted classes vs true labels.

This must be measured independently.

### RTL/reference equivalence

RTL output vs software reference.

This proves implementation equivalence, not classification quality.

### Physical DE10 accuracy

Actual board execution vs true labels.

This is a separate measurement.

Never call reference accuracy "hardware accuracy."

---

## 10. Back-to-Back Verification

Required:

### Test A

Run different-class windows consecutively without reset.

Example:

    normal -> outer

Verify both outputs.

### Test B

Run the same window twice without reset.

Verify:

    output_1 == output_2

No stale state may contaminate subsequent inference.

---

## 11. Full Simulation

The complete 512-window simulation must actually finish.

For long simulations it is acceptable to run in the background, for example:

    nohup ... > /tmp/cnn_full.log 2>&1 &

But the final report must verify:

    process completed
    exit status
    log exists
    all 512 windows executed
    final comparison completed

Do not claim completion merely because the process was started.

---

## 12. Performance

Do not trust old cycle counts without reproducing them.

Read latency constants from the actual RTL.

Current known generic accelerator timing model:

    ARRAY_LATENCY = 18
    QUANT_LATENCY = 2
    TOTAL_LATENCY = 20

These values must still be verified against the actual checked-out RTL.

Performance counters currently producing `x` are NOT valid performance measurements.

Do not report them as measured values.

If throughput is reported at 50 MHz, explicitly identify 50 MHz as the assumed/verified clock.

---

## 13. Quartus / DE10 Requirement

Target:

    Terasic DE10-Standard
    Cyclone V SoC
    5CSXFC6D6F31C6

Target clock:

    50 MHz
    20 ns period

Verify the actual top-level clock constraint.

Do not verify this merely by reading the SDC file.

Run Quartus STA evidence including:

    report_clocks
    check_timing

The report must establish:

    clock recognized
    clock period = 20 ns
    unconstrained paths status
    relevant timing corner/model
    worst setup slack
    worst hold slack

If the 20 ns clock is not recognized:

    TIMING CONSTRAINT NOT VERIFIED

---

## 14. Quartus Failure Classification

Use exactly these classifications where applicable.

Analysis/synthesis failure:

    NOT MAPPABLE TO DE10 — ANALYSIS/SYNTHESIS FAILED

Fit failure:

    NOT MAPPABLE TO DE10 — FIT FAILED

Programming-file generation failure:

    NOT DEPLOYABLE TO DE10 — PROGRAMMING FILE GENERATION FAILED

Fits but fails 50 MHz timing:

    FITS DE10 BUT DOES NOT MEET 50 MHz TIMING

Missing/unverified valid 20 ns constraint:

    TIMING CONSTRAINT NOT VERIFIED

Fits and meets 50 MHz timing:

    DE10 MAPPABLE AND MEETS 50 MHz TIMING

---

## 15. Holdout Provenance

Inspect the dataset generation / split code.

Determine whether recording-level separation between training and holdout is demonstrably established.

If it can be established, document the evidence.

If it cannot be independently established, state verbatim:

    HOLDOUT PROVENANCE NOT FULLY ESTABLISHED

This is a disclosure, NOT a blocking criterion.

It does NOT prevent COMPLETE, provided the limitation is explicitly documented.

Do not silently assume recording-level separation.

---

## 16. Physical DE10 Measurement

Physical board execution is separate from simulation.

If 512-window physical execution is not performed, state:

    Physical DE10 512-window accuracy: NOT MEASURED

This does NOT block COMPLETE for the simulation + Quartus deliverable.

Do not claim physical 512-window accuracy unless the board actually executed the windows and the results were recorded.

---

## 17. Required Progress File

Maintain:

    docs/cnn_progress.md

Throughout the implementation.

Record:

    date/time
    commit
    action
    command
    result
    evidence
    blockers
    next step

Do not fabricate results.

---

## 18. Git Requirements

Work only on:

    fpga-cnn-integration

Never modify:

    main

Before every push verify:

    git branch --show-current
    git status
    git log -1 --oneline

After every meaningful commit:

    git push origin fpga-cnn-integration

Verify the push succeeded.

Record the final commit SHA.

---

## 19. Required Final Evidence Table

The final report must contain at least:

| Criterion | Result | Evidence |
|---|---|---|
| Generic regression | | |
| Fresh reference windows | | |
| Fresh reference accuracy | | |
| RTL executed windows | | |
| RTL/reference logits | | |
| RTL/true-label accuracy | | |
| Reference error count | | |
| RTL error count | | |
| Error-index cross-check | | |
| Per-class errors | | |
| First 32 Conv1 checks | | |
| First 32 MaxPool checks | | |
| First 32 Conv2 checks | | |
| First 32 GAP checks | | |
| First 32 Dense/logit checks | | |
| Back-to-back different classes | | |
| Same-window repeated inference | | |
| Total cycles | | |
| Clock constraint | | |
| `report_clocks` | | |
| `check_timing` | | |
| Unconstrained paths | | |
| Timing corner/model | | |
| Worst setup slack | | |
| Worst hold slack | | |
| Quartus synthesis | | |
| Quartus fit | | |
| Programming-file generation | | |
| Holdout provenance | | |
| Physical DE10 512-window accuracy | | |
| Final commit SHA | | |
| Push verified | | |

---

## 20. COMPLETE Definition

The deliverable may be marked:

    COMPLETE

when all BLOCKING simulation, RTL-equivalence, and Quartus requirements have actual evidence.

The following explicitly disclosed statuses do NOT block COMPLETE:

    HOLDOUT PROVENANCE NOT FULLY ESTABLISHED

and:

    Physical DE10 512-window accuracy: NOT MEASURED

provided they are clearly reported.

The final report must NOT imply that physical DE10 512-window validation occurred if it did not.

If a blocking requirement lacks evidence, the final status must be:

    NOT COMPLETE

Do not hide missing evidence behind assumptions.
