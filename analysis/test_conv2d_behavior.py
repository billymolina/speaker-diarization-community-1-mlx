"""
Minimal test to compare PyTorch and MLX Conv2d behavior.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn as torch_nn


def main():
    print("="*70)
    print(" Conv2d Behavior Test")
    print("="*70)

    # Create identical simple input
    np.random.seed(42)
    input_np = np.random.randn(1, 4, 4, 1).astype(np.float32)  # Small 4x4 input

    print(f"\nInput shape (numpy): {input_np.shape}")
    print(f"Input:\n{input_np[0, :, :, 0]}")

    # Create identical weights
    weight_np = np.random.randn(2, 3, 3, 1).astype(np.float32)  # 2 output channels, 3x3 kernel

    print(f"\nWeight shape (numpy): {weight_np.shape}")

    # PyTorch Conv2d
    # PyTorch format: (batch, channels, height, width)
    input_pt_np = np.transpose(input_np, (0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
    input_pt = torch.from_numpy(input_pt_np)

    # PyTorch weight format: (out_channels, in_channels, kernel_h, kernel_w)
    weight_pt_np = np.transpose(weight_np, (0, 3, 1, 2))  # (O,K_h,K_w,I) -> (O,I,K_h,K_w)
    weight_pt = torch.from_numpy(weight_pt_np)

    conv_pt = torch_nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv_pt.weight.data = weight_pt

    with torch.no_grad():
        output_pt = conv_pt(input_pt)

    output_pt_np = output_pt.detach().cpu().numpy()
    print(f"\nPyTorch:")
    print(f"  Input shape: {input_pt.shape}")
    print(f"  Weight shape: {weight_pt.shape}")
    print(f"  Output shape: {output_pt_np.shape}")
    print(f"  Output (channel 0):\n{output_pt_np[0, 0, :, :]}")

    # MLX Conv2d
    # MLX format: (batch, height, width, channels)
    input_mlx = mx.array(input_np)

    # MLX weight format: (out_channels, kernel_h, kernel_w, in_channels)
    weight_mlx = mx.array(weight_np)

    conv_mlx = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    conv_mlx.weight = weight_mlx

    output_mlx = conv_mlx(input_mlx)
    output_mlx_np = np.array(output_mlx)

    print(f"\nMLX:")
    print(f"  Input shape: {input_mlx.shape}")
    print(f"  Weight shape: {weight_mlx.shape}")
    print(f"  Output shape: {output_mlx_np.shape}")
    print(f"  Output (channel 0):\n{output_mlx_np[0, :, :, 0]}")

    # Compare
    # PyTorch: (B, C, H, W), MLX: (B, H, W, C)
    # Transpose PyTorch to MLX format
    output_pt_transposed = np.transpose(output_pt_np, (0, 2, 3, 1))

    diff = np.abs(output_pt_transposed - output_mlx_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n{'='*70}")
    print(" Comparison")
    print(f"{'='*70}")
    print(f"Max abs diff: {max_diff:.6e}")
    print(f"Mean abs diff: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print(f"\n✓ Conv2d outputs match!")
    else:
        print(f"\n✗ Conv2d outputs DO NOT match!")
        print(f"\nDifference map (channel 0):\n{diff[0, :, :, 0]}")


if __name__ == "__main__":
    main()
