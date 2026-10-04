#!/usr/bin/env python3
"""
Generate CNN verification vectors from actual .npy files.
Validates vector shapes, dtypes, and creates deterministic test cases.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
ML_ROOT = REPO_ROOT / "ml"
ARTIFACTS_DIR = ML_ROOT / "artifacts"
DATA_DIR = ML_ROOT / "data" / "cwru_12k_de"
OUTPUT_DIR = Path(__file__).parent / "generated"

# Constants
LABELS = ["normal", "inner", "ball", "outer"]
WINDOW_SIZE = 256
NUM_WINDOWS_PER_CLASS = 128
MAX_TOTAL_WINDOWS = 512

def load_manifest():
    """Load and validate the accelerator manifest."""
    manifest_path = ARTIFACTS_DIR / "accelerator_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"✓ Loaded manifest: {manifest.get('format', 'unknown')}")
    return manifest

def validate_npy_file(filepath, name, expected_dtype=None, expected_shape_desc=None):
    """Validate a .npy file and report shape/dtype."""
    if not filepath.exists():
        print(f"  ERROR: {name} not found at {filepath}")
        return None
    
    data = np.load(filepath)
    print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
    
    if expected_dtype and data.dtype != expected_dtype:
        print(f"    WARNING: expected dtype {expected_dtype}, got {data.dtype}")
    
    return data

def load_reference_data():
    """Load and validate all reference CNN data."""
    print("\n=== Validating CNN Reference Matrices ===")
    
    # Conv1 weights and bias
    conv1_w = validate_npy_file(
        ARTIFACTS_DIR / "conv1_weights_kxoc_int8.npy",
        "conv1_weights_kxoc_int8.npy",
        np.int8
    )
    conv1_b = validate_npy_file(
        ARTIFACTS_DIR / "conv1_bias_accumulator_int32.npy",
        "conv1_bias_accumulator_int32.npy",
        np.int32
    )
    
    # Conv2 weights and bias
    conv2_w = validate_npy_file(
        ARTIFACTS_DIR / "conv2_weights_kxoc_int8.npy",
        "conv2_weights_kxoc_int8.npy",
        np.int8
    )
    conv2_b = validate_npy_file(
        ARTIFACTS_DIR / "conv2_bias_accumulator_int32.npy",
        "conv2_bias_accumulator_int32.npy",
        np.int32
    )
    
    # Classifier weights and bias
    clf_w = validate_npy_file(
        ARTIFACTS_DIR / "classifier_weights_kxoc_int8.npy",
        "classifier_weights_kxoc_int8.npy",
        np.int8
    )
    clf_b = validate_npy_file(
        ARTIFACTS_DIR / "classifier_bias_accumulator_int32.npy",
        "classifier_bias_accumulator_int32.npy",
        np.int32
    )
    
    if not all([conv1_w is not None, conv1_b is not None,
                conv2_w is not None, conv2_b is not None,
                clf_w is not None, clf_b is not None]):
        print("ERROR: Failed to load all reference matrices")
        sys.exit(1)
    
    return {
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "clf_w": clf_w,
        "clf_b": clf_b
    }

def saturate_int8(val):
    """Saturate value to INT8 range [-128, 127]."""
    return max(-128, min(127, int(val)))

def mac_operation(x, w, b, shift_amount, relu_enable):
    """
    Multiply-Accumulate: INT8 × INT8 → INT32, then requantize to INT8.
    
    Args:
        x: input matrix (rows × input_cols) INT8
        w: weight matrix (input_cols × output_cols) INT8
        b: bias vector (output_cols,) INT32
        shift_amount: arithmetic right shift
        relu_enable: apply ReLU before requantization
    
    Returns:
        output (rows × output_cols) INT8
        accumulators (rows × output_cols) INT32
    """
    rows, input_cols = x.shape
    output_cols = w.shape[1]
    
    accumulators = np.zeros((rows, output_cols), dtype=np.int32)
    output = np.zeros((rows, output_cols), dtype=np.int8)
    
    for row in range(rows):
        for col in range(output_cols):
            # Bias contribution
            acc = int(b[col]) if col < len(b) else 0
            
            # MAC loop
            for k in range(input_cols):
                x_val = int(x[row, k]) if k < x.shape[1] else 0
                w_val = int(w[k, col]) if k < w.shape[0] else 0
                acc += x_val * w_val
            
            # Clamp to INT32 range
            acc = max(-(1 << 31), min((1 << 31) - 1, acc))
            accumulators[row, col] = acc
            
            # Requantize: ReLU (optional) → arithmetic shift → saturate
            if relu_enable:
                acc = max(0, acc)
            
            output[row, col] = saturate_int8(acc >> shift_amount)
    
    return output, accumulators

def im2col_1d(x, kernel_size):
    """
    Convert 1D input to im2col format for 1D convolution.
    
    Args:
        x: (1, channels) for dense, or (spatial, channels) for conv
        kernel_size: kernel width for conv, 1 for dense
    
    Returns:
        im2col_output (spatial_out, kernel_size * channels)
    """
    if x.ndim == 1:
        x = x[:, None]
    
    spatial, channels = x.shape
    
    if kernel_size == 1:
        # Dense operation
        return x
    
    # Conv1d with symmetric padding
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    # Extract sliding windows: each row is [ch0_t0, ch1_t0, ..., ch0_t1, ch1_t1, ...]
    output = []
    for t in range(spatial):
        window = []
        for ch in range(channels):
            window.extend(padded[t:t+kernel_size, ch])
        output.append(window)
    
    return np.array(output, dtype=np.int8)

def load_raw_window(recording_path, window_index):
    """
    Load one 256-sample window from a CWRU recording.
    
    Returns:
        window (256,) INT8, normalized and quantized
    """
    data = loadmat(recording_path)
    key = next(k for k in data if k.endswith("DE_time"))
    signal = data[key].reshape(-1).astype(np.float32)
    
    start = window_index * WINDOW_SIZE
    end = start + WINDOW_SIZE
    if end > len(signal):
        raise ValueError(f"Window {window_index} out of range")
    
    window = signal[start:end]
    
    # Normalize: zero mean, unit variance
    mean = window.mean()
    std = window.std() + 1e-8
    window = (window - mean) / std
    
    # Quantize: scale = 0.0625 (training default)
    window = np.clip(np.round(window / 0.0625), -127, 127).astype(np.int8)
    
    return window

def reference_inference(window, manifest, ref_data):
    """
    Run the exact reference INT8 inference model.
    
    Returns:
        dict with all intermediate tensors and logits
    """
    # Conv1: kernel=5, 8 output channels
    conv1_im2col = im2col_1d(window[None, :], 5)  # (256, 5) 
    conv1_out, conv1_acc = mac_operation(
        conv1_im2col, ref_data["conv1_w"], ref_data["conv1_b"],
        shift_amount=8, relu_enable=True
    )  # (256, 8)
    
    # Max pool: 1d, stride=2
    pool_out = np.maximum(conv1_out[::2, :], conv1_out[1::2, :]).astype(np.int8)  # (128, 8)
    
    # Conv2: kernel=3, 8 output channels
    conv2_im2col = im2col_1d(pool_out, 3)  # (128, 24)
    conv2_out, conv2_acc = mac_operation(
        conv2_im2col, ref_data["conv2_w"], ref_data["conv2_b"],
        shift_amount=8, relu_enable=True
    )  # (128, 8)
    
    # GAP: average over spatial dimension (128)
    gap_out = np.trunc(conv2_out.astype(np.int32).sum(0) / 128).astype(np.int8)[None, :]  # (1, 8)
    
    # Classifier: dense, 4 output channels
    logits, clf_acc = mac_operation(
        gap_out, ref_data["clf_w"], ref_data["clf_b"],
        shift_amount=5, relu_enable=False
    )  # (1, 4)
    
    logits_1d = logits[0, :]  # (4,)
    prediction = int(np.argmax(logits_1d))
    
    return {
        "window": window,
        "conv1_im2col": conv1_im2col,
        "conv1": conv1_out,
        "conv1_acc": conv1_acc,
        "pool": pool_out,
        "conv2_im2col": conv2_im2col,
        "conv2": conv2_out,
        "conv2_acc": conv2_acc,
        "gap": gap_out,
        "classifier_acc": clf_acc,
        "logits": logits_1d,
        "prediction": prediction
    }

def generate_test_vectors(manifest, ref_data):
    """Generate all 512 CNN test vectors."""
    print("\n=== Generating 512 CNN Test Vectors ===")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    test_cases = []
    window_count = 0
    
    for expected_class, label in enumerate(LABELS):
        recording_paths = sorted(DATA_DIR.glob(f"{label}/*.mat"))
        if not recording_paths:
            print(f"WARNING: No recordings found for {label}")
            continue
        
        # Use last recording (holdout set)
        recording = recording_paths[-1]
        
        # Generate windows_per_class for this class
        windows_per_class = NUM_WINDOWS_PER_CLASS // len(LABELS)
        
        for window_idx in range(windows_per_class):
            if window_count >= MAX_TOTAL_WINDOWS:
                break
            
            try:
                raw_window = load_raw_window(recording, window_idx)
            except ValueError as e:
                print(f"  Skipping window {window_idx} of {label}: {e}")
                continue
            
            # Run reference inference
            result = reference_inference(raw_window, manifest, ref_data)
            
            # Create test case directory
            test_name = f"{expected_class:03d}_{label}_{window_idx:03d}"
            test_dir = OUTPUT_DIR / test_name
            test_dir.mkdir(exist_ok=True)
            
            # Save intermediate tensors as .npy for inspection
            for key in ["window", "conv1_im2col", "conv1", "conv1_acc", "pool",
                        "conv2_im2col", "conv2", "conv2_acc", "gap", "classifier_acc"]:
                if key in result:
                    np.save(test_dir / f"{key}.npy", result[key])
            
            # Save reference logits and prediction
            np.save(test_dir / "reference_logits_int8.npy", result["logits"])
            
            case_meta = {
                "test_id": window_count,
                "test_name": test_name,
                "label": label,
                "expected_class": expected_class,
                "reference_logits": result["logits"].astype(int).tolist(),
                "reference_prediction": result["prediction"],
                "window_shape": result["window"].shape,
                "window_dtype": str(result["window"].dtype),
                "conv1_out_shape": result["conv1"].shape,
                "conv2_out_shape": result["conv2"].shape,
                "gap_shape": result["gap"].shape,
                "classifier_logits_shape": result["logits"].shape
            }
            
            with open(test_dir / "meta.json", "w") as f:
                json.dump(case_meta, f, indent=2)
            
            test_cases.append(case_meta)
            window_count += 1
            
            if window_count % 64 == 0:
                print(f"  Generated {window_count} vectors...")
    
    # Save summary
    summary = {
        "format": "cnn-verification-v1",
        "total_windows": window_count,
        "windows_per_class": NUM_WINDOWS_PER_CLASS // len(LABELS),
        "num_classes": len(LABELS),
        "labels": LABELS,
        "test_cases": test_cases,
        "reference_accuracy": sum(1 for tc in test_cases if tc["expected_class"] == tc["reference_prediction"]) / max(1, len(test_cases)) if test_cases else 0
    }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated {window_count} test vectors in {OUTPUT_DIR}")
    print(f"✓ Reference accuracy: {summary['reference_accuracy']:.1%}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Generate CNN verification vectors from actual .npy reference data"
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for test vectors")
    args = parser.parse_args()
    
    print("=== CNN Vector Generator ===\n")
    
    manifest = load_manifest()
    ref_data = load_reference_data()
    summary = generate_test_vectors(manifest, ref_data)
    
    print("\n" + json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
