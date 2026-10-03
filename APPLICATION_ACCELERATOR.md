# Application Accelerator Verification

## Source of truth

The deployed fixed-model CNN arithmetic is defined by:

- `ml/hardware_reference.py`
- `ml/artifacts/accelerator_manifest.json`
- exported INT8 weights/biases under `ml/artifacts/`
- `quartus/de10_standard_hw/model_data/`

The board-level RTL is `quartus/de10_standard_hw/bearing_cnn_demo.sv`.

## CNN datapath

```text
256-sample INT8 window
        |
        v
Conv1: 1 -> 8, K=5, same padding
        |   8 output MAC lanes in parallel
        v
ReLU + >>8 + INT8 saturation
        |
        v
MaxPool1D(2)
        |
        v
Conv2: 8 -> 8, K=3, same padding
        |   K=24 flattened as channel-major × tap
        |   3 blocks × 8 input lanes
        |   64 products per block cycle
        v
ReLU + >>8 + INT8 saturation
        |
        v
GAP over 128 samples
        |   signed truncation toward zero /128
        v
8-element INT8 feature vector
        |
        v
8 x 8 classifier MAC
        |
        v
>>5 + INT8 saturation
        |
        v
Argmax(logits[0:3])
```

## Exact exported-model verification

The exact integer deployment reference was rerun from the exported artifacts over the recording-level test holdout:

| Class | Correct | Total |
|---|---:|---:|
| normal | 128 | 128 |
| inner | 124 | 128 |
| ball | 83 | 128 |
| outer | 127 | 128 |
| **Total** | **462** | **512** |

**Exact exported INT8 reference accuracy: 90.234375%.**

The older `89.453125%` value was produced by the training script's rounded requantization path rather than the exact deployed shift-only reference. The manifest now records the exact deployment-reference value.

## Demo vectors

The exported board demo reference gives:

- normal case: logits `[39, 2, -36, -51]`, class `0`
- outer case: logits `[-33, 3, 9, 15]`, class `3`

## Hardware status

The new board-level RTL uses a 64-MAC parallel datapath rather than the previous one-MAC-per-cycle scalar reference implementation.

This workspace cannot run Quartus/Icarus binaries, so a fresh local compile/simulation is still required before treating the new RTL as bitstream-verified. No new Quartus timing/resource numbers are claimed here.
