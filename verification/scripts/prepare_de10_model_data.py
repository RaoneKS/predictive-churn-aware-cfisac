#!/usr/bin/env python3
"""Prepare DE10 board ROM images from the authoritative CNN vectors.

The physical demo uses two representative windows:
  demo_inputs[0:256]   = reference window 0 (normal)
  demo_inputs[256:512] = reference window 384 (outer)
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
    path.write_text("\n".join(formatter(v) for v in values) + "\n")

def require(path):
    if not path.exists():
        raise SystemExit(
            f"ERROR: missing {path}. Run verification/scripts/prepare_cnn_iverilog_vectors.py "
            "or restore the authoritative verification/cnn_vectors/*.npy files first."
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

    if inp.shape[0] < 385 or inp.shape[1:] != (1, 256):
        raise SystemExit(f"ERROR: unexpected input shape {inp.shape}; need at least 385 windows")
    if c1w.shape != (8, 1, 5) or c1b.shape != (8,):
        raise SystemExit(f"ERROR: unexpected Conv1 shapes {c1w.shape}, {c1b.shape}")
    if c2w.shape != (8, 8, 3) or c2b.shape != (8,):
        raise SystemExit(f"ERROR: unexpected Conv2 shapes {c2w.shape}, {c2b.shape}")
    if fcw.shape != (4, 8) or fcb.shape != (4,):
        raise SystemExit(f"ERROR: unexpected classifier shapes {fcw.shape}, {fcb.shape}")

    OUT.mkdir(parents=True, exist_ok=True)

    # Board wrapper selects window 0 for normal and window 384 for outer.
    demo = np.concatenate([inp[0, 0, :], inp[384, 0, :]]).astype(np.int8)
    write_hex(OUT / "demo_inputs.hex", demo, hex_signed8)

    # Hardware wrapper stores Conv1 as [K][OC].
    c1_hw = np.transpose(c1w[:, 0, :], (1, 0)).reshape(-1)
    write_hex(OUT / "conv1_weights.hex", c1_hw, hex_signed8)
    write_hex(OUT / "conv1_bias.hex", c1b, hex_signed32)

    # Hardware wrapper stores Conv2 as [input_channel * kernel + tap][OC].
    c2_hw = np.transpose(c2w, (1, 2, 0)).reshape(-1)
    write_hex(OUT / "conv2_weights.hex", c2_hw, hex_signed8)
    write_hex(OUT / "conv2_bias.hex", c2b, hex_signed32)

    # Hardware wrapper uses [input_feature][physical_output].
    fc_hw = np.zeros((8, 8), dtype=np.int8)
    fc_hw[:, :4] = fcw.T
    write_hex(OUT / "classifier_weights.hex", fc_hw.reshape(-1), hex_signed8)
    fc_bias_hw = np.zeros(8, dtype=np.int32)
    fc_bias_hw[:4] = fcb
    write_hex(OUT / "classifier_bias.hex", fc_bias_hw, hex_signed32)

    print("DE10 model ROMs prepared from authoritative vectors.")
    print(f"Normal window: 0")
    print(f"Outer window: 384")
    for p in sorted(OUT.glob("*.hex")):
        print(f"{p.name}: {sum(1 for _ in p.open())} words")

if __name__ == "__main__":
    main()
