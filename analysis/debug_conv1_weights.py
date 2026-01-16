"""
Debug conv1 weight loading by comparing PyTorch and MLX weights.
"""

import numpy as np
import mlx.core as mx
import torch


def main():
    print("="*70)
    print(" Conv1 Weight Loading Debug")
    print("="*70)

    # Load PyTorch model
    from pyannote.audio import Model
    pt_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    pt_resnet = pt_model.resnet
    pt_resnet.eval()

    # Load MLX model
    from src.resnet_embedding import load_resnet34_embedding
    mlx_model = load_resnet34_embedding("models/embedding_mlx/weights.npz")

    # Get conv1 weights
    pt_weight = pt_resnet.conv1.weight.detach().cpu().numpy()
    mlx_weight = np.array(mlx_model.conv1.weight)

    print(f"\n[1] Weight Shapes:")
    print(f"    PyTorch: {pt_weight.shape}")
    print(f"    MLX: {mlx_weight.shape}")

    print(f"\n[2] Weight Statistics:")
    print(f"    PT mean: {np.mean(pt_weight):.6f}, std: {np.std(pt_weight):.6f}")
    print(f"    MLX mean: {np.mean(mlx_weight):.6f}, std: {np.std(mlx_weight):.6f}")

    print(f"\n[3] Sample Values (first 5 elements flattened):")
    print(f"    PT: {pt_weight.flatten()[:5]}")
    print(f"    MLX: {mlx_weight.flatten()[:5]}")

    # Transpose PyTorch weight to MLX format
    # PyTorch: (out_channels, in_channels, height, width)
    # MLX: (out_channels, height, width, in_channels)
    print(f"\n[4] Transposing PyTorch weight to MLX format...")
    pt_weight_transposed = np.transpose(pt_weight, (0, 2, 3, 1))
    print(f"    Transposed shape: {pt_weight_transposed.shape}")

    # Compare
    abs_diff = np.abs(pt_weight_transposed - mlx_weight)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)

    print(f"\n[5] Weight Comparison:")
    print(f"    Max abs diff: {max_abs:.6e}")
    print(f"    Mean abs diff: {mean_abs:.6e}")
    print(f"    Shapes match: {pt_weight_transposed.shape == mlx_weight.shape}")

    if max_abs < 1e-6:
        print(f"\n✓ Weights match perfectly!")
    elif max_abs < 1e-3:
        print(f"\n⚠  Weights are very close (minor numerical differences)")
    else:
        print(f"\n✗ Weights DO NOT match!")
        print(f"\nDifferences (first 10 locations with max diff):")
        diff_flat = abs_diff.flatten()
        top_indices = np.argsort(diff_flat)[-10:][::-1]
        for idx in top_indices:
            pt_val = pt_weight_transposed.flatten()[idx]
            mlx_val = mlx_weight.flatten()[idx]
            diff = diff_flat[idx]
            print(f"  Index {idx}: PT={pt_val:.6f}, MLX={mlx_val:.6f}, diff={diff:.6f}")

    # Test with identical input
    print(f"\n{'='*70}")
    print(" Testing Conv1 Forward Pass")
    print(f"{'='*70}")

    np.random.seed(42)
    test_input_np = np.random.randn(1, 1, 80, 100).astype(np.float32)

    # PyTorch
    test_input_pt = torch.from_numpy(test_input_np)
    with torch.no_grad():
        pt_output = pt_resnet.conv1(test_input_pt)
    pt_output_np = pt_output.detach().cpu().numpy()

    # MLX (need to convert input format)
    # PyTorch: (1, 1, 80, 100) = (batch, channels, freq, time)
    # MLX: (batch, time, freq, channels)
    test_input_mlx_np = np.transpose(test_input_np, (0, 3, 2, 1))  # (B,C,F,T) -> (B,T,F,C)
    test_input_mlx = mx.array(test_input_mlx_np)
    mlx_output = mlx_model.conv1(test_input_mlx)
    mlx_output_np = np.array(mlx_output)

    # Convert PyTorch output to MLX format for comparison
    # PyTorch output: (B, C, F, T)
    # MLX output: (B, T, F, C)
    pt_output_transposed = np.transpose(pt_output_np, (0, 3, 2, 1))

    print(f"\nInput:")
    print(f"  PT shape: {test_input_pt.shape}")
    print(f"  MLX shape: {test_input_mlx.shape}")
    print(f"  Mean: {np.mean(test_input_np):.6f}")

    print(f"\nOutput:")
    print(f"  PT shape: {pt_output_np.shape}")
    print(f"  MLX shape: {mlx_output_np.shape}")
    print(f"  PT transposed shape: {pt_output_transposed.shape}")

    output_diff = np.abs(pt_output_transposed - mlx_output_np)
    max_output_diff = np.max(output_diff)
    mean_output_diff = np.mean(output_diff)

    print(f"\nOutput Comparison:")
    print(f"  Max abs diff: {max_output_diff:.6e}")
    print(f"  Mean abs diff: {mean_output_diff:.6e}")
    print(f"  PT sample: {pt_output_transposed.flatten()[:5]}")
    print(f"  MLX sample: {mlx_output_np.flatten()[:5]}")

    if max_output_diff < 1e-5:
        print(f"\n✓ Conv1 outputs match!")
    else:
        print(f"\n✗ Conv1 outputs DO NOT match!")


if __name__ == "__main__":
    main()
