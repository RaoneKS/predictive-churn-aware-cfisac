# Phase 3: predictive-maintenance ML pipeline

This directory is intentionally separate from the validated Phase 2 RTL.

## Dataset

The pipeline uses the public Case Western Reserve University (CWRU) 12 kHz
drive-end bearing-vibration recordings. The initial four-class task is normal,
inner-race fault, ball fault, and outer-race fault. Split recordings before
windowing to prevent windows from a single acquisition appearing in both train
and test sets.

## FPGA workload contract

The accelerator accepts INT8 activations and weights packed into 8-element
vectors, with INT32 accumulation and INT8 requantized output. The intended
compact 1D-CNN mapping is:

1. Window and normalize vibration samples offline/on the HPS.
2. Run Conv1D as im2col plus matrix multiplication, padding input and output
   channels to multiples of eight.
3. Run the final dense classifier as a padded matrix multiplication.
4. Apply each layer's exported INT8 scale, zero point, and ReLU setting.

`train_and_export.py` will create an `artifacts/` directory containing model
metrics, padded INT8 weights, bias vectors, and layer quantization metadata.
