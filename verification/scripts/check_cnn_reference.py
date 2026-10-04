#!/usr/bin/env python3
"""Check the authoritative 512-window INT8 reference set and report error indices."""

from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/"verification"/"cnn_vectors"

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
    if len(errors)!=54 or int(correct.sum())!=458:
        raise SystemExit("ERROR: authoritative reference is not the expected 458/512 result")

    print("CNN_REFERENCE_CHECK_PASSED")

if __name__=="__main__":
    main()
