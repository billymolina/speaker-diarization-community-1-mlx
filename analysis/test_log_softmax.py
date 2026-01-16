"""
Test if log_softmax produces identical results in PyTorch vs MLX
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np

print("=" * 80)
print("LOG_SOFTMAX COMPARISON")
print("=" * 80)

# Create identical test logits
test_logits = np.array([
    [-43.66, -40.32, -38.51, -39.14, -45.39, -45.14, -43.61],  # Frame 1
    [-42.1, -38.8, -36.2, -37.5, -43.9, -43.5, -42.0],          # Frame 2
    [-5.32, -10.2, -8.5, -9.1, -12.3, -11.8, -10.5],           # Frame 3
], dtype=np.float32)

print("\n1️⃣ Input logits:")
print(test_logits)

# PyTorch
print("\n2️⃣ PyTorch log_softmax:")
pt_logits = torch.from_numpy(test_logits)
pt_output = torch.log_softmax(pt_logits, dim=-1)
print(pt_output)
print(f"   Range: [{pt_output.min():.6f}, {pt_output.max():.6f}]")
print(f"   Sum per frame (should be ~1 after exp):")
for i in range(3):
    s = torch.exp(pt_output[i]).sum()
    print(f"     Frame {i}: {s:.10f}")

# MLX
print("\n3️⃣ MLX log_softmax:")
mlx_logits = mx.array(test_logits)
mlx_output = nn.log_softmax(mlx_logits, axis=-1)
print(mlx_output)
print(f"   Range: [{float(mx.min(mlx_output)):.6f}, {float(mx.max(mlx_output)):.6f}]")
print(f"   Sum per frame (should be ~1 after exp):")
for i in range(3):
    s = mx.sum(mx.exp(mlx_output[i]))
    print(f"     Frame {i}: {float(s):.10f}")

# Compare
print("\n4️⃣ Comparison:")
pt_np = pt_output.numpy()
mlx_np = np.array(mlx_output.tolist())

diff = np.abs(pt_np - mlx_np)
print(f"   Max difference: {diff.max():.10f}")
print(f"   Mean difference: {diff.mean():.10f}")

if diff.max() < 1e-6:
    print(f"   ✓ log_softmax produces identical results")
else:
    print(f"   ⚠️  log_softmax has differences!")
    print(f"\n   Detailed comparison (first frame):")
    print(f"   {'Class':<8} {'PyTorch':<15} {'MLX':<15} {'Diff':<15}")
    print(f"   " + "-" * 56)
    for c in range(7):
        print(f"   {c:<8} {pt_np[0,c]:<15.10f} {mlx_np[0,c]:<15.10f} {diff[0,c]:<15.10f}")

# Test with actual model logits
print("\n5️⃣ Testing with real model logits:")
# From the debug output
real_pt = np.array([-0.12139168, -6.829805, -2.4110262, -3.813716, -9.02123, -9.597413, -6.6813245])
real_mlx = np.array([-1.8525882, -3.0288858, -0.6530862, -1.4407687, -5.388873, -5.136155, -3.6056566])

print(f"   PyTorch logit: {real_pt}")
print(f"   MLX logit: {real_mlx}")
print(f"   Difference: {np.abs(real_pt - real_mlx)}")
print(f"   Max diff: {np.abs(real_pt - real_mlx).max():.6f}")

# These are AFTER log_softmax - let's see what they were before
print(f"\n   If these are AFTER log_softmax, working backwards:")
pt_probs = np.exp(real_pt)
mlx_probs = np.exp(real_mlx)
print(f"   PyTorch probs sum: {pt_probs.sum():.10f}")
print(f"   MLX probs sum: {mlx_probs.sum():.10f}")

if abs(pt_probs.sum() - 1.0) < 0.01 and abs(mlx_probs.sum() - 1.0) < 0.01:
    print(f"   ✓ Both are valid log probabilities (sum to 1)")
    print(f"   The models produce different logit distributions!")
else:
    print(f"   These might not be after log_softmax")

print("\n" + "=" * 80)
