#!/usr/bin/env python3
"""Prepare DE10 board ROM images from canonical KxOC INT8 reference vectors.

Canonical vector layout:
  input_int8:          [window, sample]       (512, 256)
  conv1_weights:       [K, output_channel]     (5, 8)
  conv2_weights:       [K, output_channel]     (24, 8)
  classifier_weights:  [K, output_channel]     (8, 4)

The board demo pads the classifier's four logical outputs to eight physical
lanes. Conversion to ROM order happens here, without mutating reference vectors.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "verification" / "cnn_vectors"
OUT = ROOT / "quartus" / "de10_standard_hw" / "model_data"


def hex_signed8(x):
    return f"{int(np.int8(x)) & 0xff:02x}"


def hex_signed32(x):
    return f"{int(np.int32(x)) & 0xffffffff:08x}"


def write_hex(path, values, formatter):
    path.write_text("\n".join(formatter(v) for v in np.asarray(values).reshape(-1)) + "\n")


def require(path):
    if not path.exists():
        raise SystemExit(
            f"ERROR: missing {path}. Run verification/scripts/build_authoritative_cnn_vectors.py first."
        )


def main():
    required = [
        "input_int8.npy",
        "conv1_weights.npy", "conv1_bias.npy",
        "conv2_weights.npy", "conv2_bias.npy",
        "classifier_weights.npy", "classifier_bias.npy",
    ]
    for name in required:
        require(V / name)

    inp = np.load(V / "input_int8.npy")
    c1w = np.load(V / "conv1_weights.npy")
    c1b = np.load(V / "conv1_bias.npy")
    c2w = np.load(V / "conv2_weights.npy")
    c2b = np.load(V / "conv2_bias.npy")
    fcw = np.load(V / "classifier_weights.npy")
    fcb = np.load(V / "classifier_bias.npy")

    expected = {
        "input_int8": (inp.shape, (512, 256)),
        "conv1_weights": (c1w.shape, (5, 8)),
        "conv1_bias": (c1b.shape, (8,)),
        "conv2_weights": (c2w.shape, (24, 8)),
        "conv2_bias": (c2b.shape, (8,)),
        "classifier_weights": (fcw.shape, (8, 4)),
        "classifier_bias": (fcb.shape, (4,)),
    }
    bad = [f"{name}: got {got}, expected {want}" for name, (got, want) in expected.items() if got != want]
    if bad:
        raise SystemExit("ERROR: canonical model tensor shape mismatch:\n  " + "\n  ".join(bad))

    OUT.mkdir(parents=True, exist_ok=True)

    # The board wrapper selects reference window 0 (normal) or 384 (outer).
    demo = np.concatenate([inp[0], inp[384]]).astype(np.int8)
    write_hex(OUT / "demo_inputs.hex", demo, hex_signed8)

    # The ROM indexes weights as K * 8 + output_channel.
    write_hex(OUT / "conv1_weights.hex", c1w.reshape(-1), hex_signed8)
    write_hex(OUT / "conv1_bias.hex", c1b, hex_signed32)
    write_hex(OUT / "conv2_weights.hex", c2w.reshape(-1), hex_signed8)
    write_hex(OUT / "conv2_bias.hex", c2b, hex_signed32)

    # Physical FC has 8 lanes, but only lanes 0..3 are real classes.
    fc_hw = np.zeros((8, 8), dtype=np.int8)
    fc_hw[:, :4] = fcw
    write_hex(OUT / "classifier_weights.hex", fc_hw.reshape(-1), hex_signed8)
    # RTL declares exactly four classifier biases; do not emit four padded words.
    write_hex(OUT / "classifier_bias.hex", fcb, hex_signed32)

    print("DE10 model ROMs prepared from canonical KxOC vectors.")
    print("Normal window: 0")
    print("Outer window: 384")
    for p in sorted(OUT.glob("*.hex")):
        print(f"{p.name}: {sum(1 for _ in p.open())} words")


if __name__ == "__main__":
    main()
