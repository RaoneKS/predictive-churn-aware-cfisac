# Phase 3 completion record

This file records the completed CWRU bearing-vibration ML deployment hand-off.

FP32 accuracy: 92.38% (473/512 held-out windows).

Integer-equivalent INT8 accuracy: 89.45% (458/512 held-out windows).

The model is Conv1D(1,8,k=5), ReLU, MaxPool1D(2), Conv1D(8,8,k=3), ReLU, GlobalAveragePool1D, Dense(8,4).

The accelerator mapping is Conv1: 256 one-tile 8x8 operations; Conv2: 128 groups of three 8x8 K-tiles with INT32 partial sums; Dense: one padded 8x8 operation. Required requantization right shifts are 8, 8, and 5.

The artifact accelerator_manifest.json supplies exact weight, bias, activation-scale, and lane-padding specifications. The baseline Phase 2 RTL was not changed.
