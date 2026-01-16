"""
Layer-by-layer comparison between PyTorch and MLX ResNet34.
This will help identify where numerical divergence starts.
"""

import numpy as np
import mlx.core as mx
import torch
import torch.nn.functional as F

def load_pytorch_resnet():
    """Load the PyTorch model and extract the ResNet part."""
    from pyannote.audio import Model

    print("[INFO] Loading PyTorch model...")
    full_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    full_model.eval()
    resnet = full_model.resnet
    resnet.eval()
    return resnet


def load_mlx_model():
    """Load the MLX ResNet model."""
    from src.resnet_embedding import load_resnet34_embedding

    print(f"[INFO] Loading MLX model...")
    model = load_resnet34_embedding("models/embedding_mlx/weights.npz")
    return model


def compare_tensors(pt_tensor, mlx_array, name=""):
    """Compare PyTorch tensor and MLX array."""
    pt_np = pt_tensor.detach().cpu().numpy()
    mlx_np = np.array(mlx_array)

    # Handle dimension order differences
    # PyTorch: (batch, channels, freq, time) = (B, C, F, T)
    # MLX: (batch, time, freq, channels) = (B, T, F, C)
    if pt_np.ndim == 4 and mlx_np.ndim == 4:
        # Transpose PyTorch to match MLX: (B,C,F,T) -> (B,T,F,C)
        pt_np_transposed = np.transpose(pt_np, (0, 3, 2, 1))
        abs_diff = np.abs(pt_np_transposed - mlx_np)
    else:
        abs_diff = np.abs(pt_np - mlx_np)

    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)

    print(f"{name}:")
    print(f"  PT shape: {pt_np.shape}, MLX shape: {mlx_np.shape}")
    print(f"  PT mean: {np.mean(pt_np):.6f}, MLX mean: {np.mean(mlx_np):.6f}")
    print(f"  PT std: {np.std(pt_np):.6f}, MLX std: {np.std(mlx_np):.6f}")
    print(f"  Max abs diff: {max_abs:.6e}")
    print(f"  Mean abs diff: {mean_abs:.6e}")
    print(f"  Sample PT: {pt_np.flatten()[:3]}")
    print(f"  Sample MLX: {mlx_np.flatten()[:3]}")

    return max_abs, mean_abs


def main():
    print("="*70)
    print(" Layer-by-Layer Comparison")
    print("="*70)

    # Load models
    pt_model = load_pytorch_resnet()
    mlx_model = load_mlx_model()

    # Create identical input
    np.random.seed(42)
    mel_np = np.random.randn(1, 100, 80).astype(np.float32)

    # PyTorch input
    mel_pt = torch.from_numpy(mel_np)
    print(f"\n[INPUT] PyTorch shape: {mel_pt.shape}")

    # MLX input (needs channel dimension)
    mel_mlx = mx.array(np.expand_dims(mel_np, axis=-1))
    print(f"[INPUT] MLX shape: {mel_mlx.shape}")

    print("\n" + "="*70)
    print("Layer-by-Layer Forward Pass")
    print("="*70)

    # PyTorch forward (manual step-by-step)
    with torch.no_grad():
        # Input preparation
        pt_x = mel_pt.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        pt_x = pt_x.unsqueeze_(1)  # Add channel dim
        print(f"\n[0] After input prep:")
        print(f"    PT shape: {pt_x.shape}")

        # Conv1 + BN + ReLU
        pt_x = pt_model.conv1(pt_x)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_model.conv1(mel_mlx), "[1] After conv1")

        pt_x = pt_model.bn1(pt_x)
        mlx_x = mlx_model.bn1(mlx_model.conv1(mel_mlx))
        max_diff, mean_diff = compare_tensors(pt_x, mlx_x, "\n[2] After bn1")

        pt_x = F.relu(pt_x)
        mlx_x = mx.maximum(mlx_x, 0)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_x, "\n[3] After relu")

        # Store for MLX path
        mlx_full = mlx_x

        # Layer 1
        pt_x = pt_model.layer1(pt_x)
        mlx_full = mlx_model.layer1(mlx_full)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_full, "\n[4] After layer1")

        # Layer 2
        pt_x = pt_model.layer2(pt_x)
        mlx_full = mlx_model.layer2(mlx_full)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_full, "\n[5] After layer2")

        # Layer 3
        pt_x = pt_model.layer3(pt_x)
        mlx_full = mlx_model.layer3(mlx_full)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_full, "\n[6] After layer3")

        # Layer 4
        pt_x = pt_model.layer4(pt_x)
        mlx_full = mlx_model.layer4(mlx_full)
        max_diff, mean_diff = compare_tensors(pt_x, mlx_full, "\n[7] After layer4")

        # Pooling
        pt_pooled = pt_model.pool(pt_x)
        mlx_pooled = mlx_model.pool(mlx_full)
        max_diff, mean_diff = compare_tensors(pt_pooled, mlx_pooled, "\n[8] After pooling")

        # FC
        pt_embed = pt_model.seg_1(pt_pooled)
        mlx_embed = mlx_model.fc(mlx_pooled)
        max_diff, mean_diff = compare_tensors(pt_embed, mlx_embed, "\n[9] After FC (final)")

    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"Final max abs diff: {max_diff:.6e}")
    print(f"Final mean abs diff: {mean_diff:.6e}")

    if max_diff < 0.01:
        print("\n✓ Numerical accuracy is excellent!")
        return 0
    elif max_diff < 0.1:
        print("\n⚠  Numerical accuracy is acceptable but not perfect")
        return 0
    else:
        print("\n✗ Numerical differences are too large")
        return 1


if __name__ == "__main__":
    exit(main())
