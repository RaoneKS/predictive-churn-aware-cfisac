#!/usr/bin/env python3
"""Build the consolidated authoritative CNN vectors used by the Icarus E2E test.

The source of truth is the committed CWRU subset plus committed INT8 model
artifacts. This script deliberately uses the same reference_inference()
implementation as generate_vectors.py so CI and local verification share the
same quantization and padding semantics.
"""
from pathlib import Path
import numpy as np

from generate_vectors import (
    ARTIFACTS_DIR, DATA_DIR, LABELS, MAX_TOTAL_WINDOWS, NUM_WINDOWS_PER_CLASS,
    load_manifest, load_reference_data, load_raw_window, reference_inference,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "verification" / "cnn_vectors"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    ref = load_reference_data()

    inputs, logits, preds, labels = [], [], [], []
    count = 0

    for expected_class, label in enumerate(LABELS):
        paths = sorted(DATA_DIR.glob(f"{label}/*.mat"))
        if not paths:
            raise RuntimeError(f"No committed CWRU recording found for {label}")
        recording = paths[-1]

        for window_idx in range(NUM_WINDOWS_PER_CLASS):
            if count >= MAX_TOTAL_WINDOWS:
                break
            window = load_raw_window(recording, window_idx)
            result = reference_inference(window, manifest, ref)
            inputs.append(window)
            logits.append(result["logits"])
            preds.append(result["prediction"])
            labels.append(expected_class)
            count += 1

    if count != 512:
        raise RuntimeError(f"Expected 512 windows, generated {count}")

    arrays = {
        "input_int8.npy": np.asarray(inputs, dtype=np.int8),
        "reference_logits_int8.npy": np.asarray(logits, dtype=np.int32),
        "reference_predictions.npy": np.asarray(preds, dtype=np.int32),
        "reference_labels.npy": np.asarray(labels, dtype=np.int32),
        "conv1_weights.npy": np.asarray(ref["conv1_w"], dtype=np.int8).reshape(8, 1, 5),
        "conv1_bias.npy": np.asarray(ref["conv1_b"], dtype=np.int32),
        "conv2_weights.npy": np.asarray(ref["conv2_w"], dtype=np.int8).reshape(8, 8, 3),
        "conv2_bias.npy": np.asarray(ref["conv2_b"], dtype=np.int32),
        "classifier_weights.npy": np.asarray(ref["clf_w"], dtype=np.int8).T,
        "classifier_bias.npy": np.asarray(ref["clf_b"], dtype=np.int32),
    }

    for name, arr in arrays.items():
        np.save(OUT / name, arr)
        print(f"Wrote {name}: shape={arr.shape}, dtype={arr.dtype}")

    accuracy = int(np.sum(np.asarray(preds) == np.asarray(labels)))
    print(f"AUTHORITATIVE_REFERENCE_ACCURACY={accuracy}/512")
    if accuracy != 458:
        raise RuntimeError(f"Reference accuracy changed: expected 458/512, got {accuracy}/512")

if __name__ == "__main__":
    main()
