"""Check actual bias values in weights file"""
import mlx.core as mx
import numpy as np

# Load weights
weights = mx.load("models/segmentation_mlx/weights.npz")

print("Bias values in weights file:")
for key in sorted(weights.keys()):
    if 'bias' in key and 'lstm' in key:
        w = np.array(weights[key])
        print(f"\n{key}:")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean():.6f}")
        print(f"  Std: {w.std():.6f}")
        print(f"  Range: [{w.min():.6f}, {w.max():.6f}]")
        print(f"  First 10 values: {w[:10]}")
