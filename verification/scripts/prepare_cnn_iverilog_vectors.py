#!/usr/bin/env python3
"""Convert authoritative CNN .npy vectors into Icarus $readmemh files.
Does not regenerate or modify the reference vectors."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "verification" / "cnn_vectors"
OUT = ROOT / "sim" / "cnn_vectors"

def hex_lines(a, bits):
    mask = (1 << bits) - 1
    flat = np.asarray(a).reshape(-1)
    return "\n".join(f"{int(x) & mask:0{bits//4}x}" for x in flat) + "\n"

def emit(name, arr, bits):
    p = OUT / name
    p.write_text(hex_lines(arr, bits))
    print(f"{p}: {np.asarray(arr).shape}, {np.asarray(arr).dtype}")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    emit("input.hex", np.load(V/"input_int8.npy"), 8)
    emit("reference_logits.hex", np.load(V/"reference_logits_int8.npy"), 32)
    emit("reference_predictions.hex", np.load(V/"reference_predictions.npy"), 8)
    emit("reference_labels.hex", np.load(V/"reference_labels.npy"), 8)
    emit("conv1_weights.hex", np.load(V/"conv1_weights.npy"), 8)
    emit("conv1_bias.hex", np.load(V/"conv1_bias.npy"), 32)
    emit("conv2_weights.hex", np.load(V/"conv2_weights.npy"), 8)
    emit("conv2_bias.hex", np.load(V/"conv2_bias.npy"), 32)
    emit("classifier_weights.hex", np.load(V/"classifier_weights.npy"), 8)
    emit("classifier_bias.hex", np.load(V/"classifier_bias.npy"), 32)
    print("Authoritative vectors converted; source .npy files were not modified.")

if __name__ == "__main__":
    main()
