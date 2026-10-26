#!/usr/bin/env python3
"""Check the authoritative 512-window INT8 reference set and report error indices."""

from pathlib import Path
import numpy as np
import json

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/"verification"/"cnn_vectors"
MANIFEST=ROOT/"ml"/"artifacts"/"accelerator_manifest.json"

def main():
    pred=np.load(V/"reference_predictions.npy").astype(np.int32)
    labels=np.load(V/"reference_labels.npy").astype(np.int32)
    logits=np.load(V/"reference_logits_int8.npy").astype(np.int32)

    if pred.shape!=(512,) or labels.shape!=(512,) or logits.shape!=(512,4):
        raise SystemExit(f"Unexpected shapes: predictions={pred.shape}, labels={labels.shape}, logits={logits.shape}")

    correct=(pred==labels)
    errors=np.flatnonzero(~correct).tolist()
    print(f"windows: {len(labels)}")
    print(f"correct: {int(correct.sum())}")
    print(f"accuracy: {100.0*correct.mean():.6f}%")
    print(f"errors: {len(errors)}")
    print(f"error_indices: {errors}")

    for cls in range(4):
        mask=labels==cls
        cc=int(np.sum(correct & mask))
        total=int(np.sum(mask))
        print(f"class_{cls}: {cc}/{total} = {100.0*cc/total:.6f}%")

    # Deterministic tie-aware argmax sanity check.
    argmax=np.argmax(logits,axis=1)
    if not np.array_equal(argmax,pred):
        bad=np.flatnonzero(argmax!=pred).tolist()
        raise SystemExit(f"ERROR: reference_predictions disagrees with logits argmax at {bad}")
    manifest=json.loads(MANIFEST.read_text())
    expected_accuracy=float(manifest["integer_int8_accuracy"])
    expected_correct=int(round(expected_accuracy*len(labels)))
    if int(correct.sum()) != expected_correct:
        raise SystemExit(
            f"ERROR: reference accuracy {int(correct.sum())}/{len(labels)} disagrees "
            f"with manifest expectation {expected_correct}/{len(labels)} "
            f"({expected_accuracy:.6%})"
        )

    print("CNN_REFERENCE_CHECK_PASSED")

if __name__=="__main__":
    main()
