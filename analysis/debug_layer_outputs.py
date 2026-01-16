"""
Debug script to trace layer-by-layer outputs for both batch_size=1 and batch_size=2.
This will help identify where outputs go to zero.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
from pathlib import Path


def load_mlx_model():
    """Load the MLX ResNet model."""
    from src.resnet_embedding import load_resnet34_embedding

    print(f"[INFO] Loading MLX model...")
    model = load_resnet34_embedding("models/embedding_mlx/weights.npz")
    return model


def load_pytorch_resnet():
    """Load the PyTorch model and extract the ResNet part."""
    from pyannote.audio import Model

    print("[INFO] Loading PyTorch model...")
    full_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    full_model.eval()

    resnet = full_model.resnet
    resnet.eval()

    print(f"[INFO] PyTorch ResNet loaded (eval mode)")
    return resnet


def create_test_input(batch_size, time_frames=100, freq_bins=80):
    """Create identical test input for both models."""
    np.random.seed(42)
    mel_np = np.random.randn(batch_size, time_frames, freq_bins).astype(np.float32)

    # PyTorch: (batch, time, freq) - 3D
    mel_pt = torch.from_numpy(mel_np)

    # MLX: (batch, time, freq, 1) - 4D with channel dim
    mel_mlx = mx.array(np.expand_dims(mel_np, axis=-1))

    return mel_pt, mel_mlx, mel_np


def debug_mlx_forward_pass(model, x, batch_size):
    """Run forward pass with intermediate outputs."""
    print(f"\n{'='*70}")
    print(f"MLX Forward Pass Debug (batch_size={batch_size})")
    print(f"{'='*70}")

    # Ensure 4D input
    if x.ndim == 3:
        x = mx.expand_dims(x, axis=-1)

    print(f"\n[0] Input:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    Has NaN: {bool(mx.any(mx.isnan(x)))}")
    print(f"    Has Inf: {bool(mx.any(mx.isinf(x)))}")

    # Initial convolution
    x = model.conv1(x)
    print(f"\n[1] After conv1:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    Has NaN: {bool(mx.any(mx.isnan(x)))}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    x = model.bn1(x)
    print(f"\n[2] After bn1:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    Has NaN: {bool(mx.any(mx.isnan(x)))}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    x = mx.maximum(x, 0)  # ReLU
    print(f"\n[3] After relu:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    # Layer 1
    x = model.layer1(x)
    print(f"\n[4] After layer1:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    Has NaN: {bool(mx.any(mx.isnan(x)))}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    if mx.all(x == 0):
        print(f"    ⚠️  OUTPUT WENT TO ZERO AFTER LAYER1!")
        return x

    # Layer 2
    x = model.layer2(x)
    print(f"\n[5] After layer2:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    if mx.all(x == 0):
        print(f"    ⚠️  OUTPUT WENT TO ZERO AFTER LAYER2!")
        return x

    # Layer 3
    x = model.layer3(x)
    print(f"\n[6] After layer3:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    if mx.all(x == 0):
        print(f"    ⚠️  OUTPUT WENT TO ZERO AFTER LAYER3!")
        return x

    # Layer 4
    x = model.layer4(x)
    print(f"\n[7] After layer4:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Min: {float(mx.min(x)):.6f}, Max: {float(mx.max(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    if mx.all(x == 0):
        print(f"    ⚠️  OUTPUT WENT TO ZERO AFTER LAYER4!")
        return x

    # Temporal pooling
    x = model.pool(x)
    print(f"\n[8] After pooling:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    # FC + BN
    x = model.fc(x)
    print(f"\n[9] After fc:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    x = model.bn_fc(x)
    print(f"\n[10] After bn_fc:")
    print(f"    Shape: {x.shape}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    # L2 normalization
    x = x / (mx.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
    print(f"\n[11] After L2 normalization:")
    print(f"    Shape: {x.shape}")
    print(f"    Norm: {mx.linalg.norm(x, axis=1)}")
    print(f"    Mean: {float(mx.mean(x)):.6f}, Std: {float(mx.std(x)):.6f}")
    print(f"    Sample values: {x[0, :5]}")
    print(f"    All zeros: {bool(mx.all(x == 0))}")

    return x


def check_batchnorm_stats(model):
    """Check BatchNorm layer statistics."""
    print(f"\n{'='*70}")
    print("BatchNorm Layer Statistics")
    print(f"{'='*70}")

    bn_layers = [
        ('bn1', model.bn1),
        ('layer1.layers.0.bn1', model.layer1.layers[0].bn1),
        ('layer1.layers.0.bn2', model.layer1.layers[0].bn2),
        ('layer2.layers.0.bn1', model.layer2.layers[0].bn1),
        ('layer3.layers.0.bn1', model.layer3.layers[0].bn1),
        ('layer4.layers.0.bn1', model.layer4.layers[0].bn1),
        ('bn_fc', model.bn_fc),
    ]

    for name, bn_layer in bn_layers:
        print(f"\n{name}:")
        # Check weight and bias
        if hasattr(bn_layer, 'weight') and bn_layer.weight is not None:
            weight = bn_layer.weight
            print(f"  Weight shape: {weight.shape}")
            print(f"  Weight mean: {float(mx.mean(weight)):.6f}")
            print(f"  Weight all ones: {bool(mx.allclose(weight, mx.ones_like(weight)))}")
        else:
            print(f"  No weight parameter")

        if hasattr(bn_layer, 'bias') and bn_layer.bias is not None:
            bias = bn_layer.bias
            print(f"  Bias shape: {bias.shape}")
            print(f"  Bias mean: {float(mx.mean(bias)):.6f}")
            print(f"  Bias all zeros: {bool(mx.allclose(bias, mx.zeros_like(bias)))}")
        else:
            print(f"  No bias parameter")


def check_weight_loading(model):
    """Verify which weights were actually loaded."""
    print(f"\n{'='*70}")
    print("Weight Loading Verification")
    print(f"{'='*70}")

    params = dict(nn.utils.tree_flatten(model.parameters()))

    print(f"\nTotal parameters in model: {len(params)}")

    # Check a few key weights
    key_checks = [
        ('conv1.weight', lambda: model.conv1.weight if hasattr(model.conv1, 'weight') else None),
        ('layer1.layers.0.conv1.weight', lambda: model.layer1.layers[0].conv1.weight if hasattr(model.layer1.layers[0].conv1, 'weight') else None),
        ('layer1.layers.0.bn1.weight', lambda: model.layer1.layers[0].bn1.weight if hasattr(model.layer1.layers[0].bn1, 'weight') else None),
        ('layer2.layers.0.conv1.weight', lambda: model.layer2.layers[0].conv1.weight if hasattr(model.layer2.layers[0].conv1, 'weight') else None),
        ('layer2.layers.0.shortcut_conv.weight', lambda: model.layer2.layers[0].shortcut_conv.weight if hasattr(model.layer2.layers[0], 'shortcut_conv') and hasattr(model.layer2.layers[0].shortcut_conv, 'weight') else None),
        ('fc.weight', lambda: model.fc.weight if hasattr(model.fc, 'weight') else None),
    ]

    print(f"\nChecking key weights:")
    for key, getter in key_checks:
        param = getter()
        if param is not None:
            print(f"\n{key}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {float(mx.mean(param)):.6f}")
            print(f"  Std: {float(mx.std(param)):.6f}")
            print(f"  Min: {float(mx.min(param)):.6f}")
            print(f"  Max: {float(mx.max(param)):.6f}")
            print(f"  All zeros: {bool(mx.all(param == 0))}")
        else:
            print(f"\n{key}: NOT FOUND")


def main():
    print("="*70)
    print(" Layer-by-Layer Debug Analysis")
    print("="*70)

    # Load models
    mlx_model = load_mlx_model()

    # Check weight loading
    check_weight_loading(mlx_model)

    # Check BatchNorm stats
    check_batchnorm_stats(mlx_model)

    # Test batch_size=1
    print(f"\n\n{'#'*70}")
    print("# TEST 1: batch_size=1")
    print(f"{'#'*70}")
    mel_pt_1, mel_mlx_1, mel_np_1 = create_test_input(batch_size=1, time_frames=100)
    output_1 = debug_mlx_forward_pass(mlx_model, mel_mlx_1, batch_size=1)

    # Test batch_size=2
    print(f"\n\n{'#'*70}")
    print("# TEST 2: batch_size=2")
    print(f"{'#'*70}")
    mel_pt_2, mel_mlx_2, mel_np_2 = create_test_input(batch_size=2, time_frames=100)
    output_2 = debug_mlx_forward_pass(mlx_model, mel_mlx_2, batch_size=2)

    # Summary
    print(f"\n\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f"\nbatch_size=1: {'All zeros ❌' if mx.all(output_1 == 0) else 'Has values ✓'}")
    print(f"batch_size=2: {'All zeros ❌' if mx.all(output_2 == 0) else 'Has values ✓'}")


if __name__ == "__main__":
    main()
