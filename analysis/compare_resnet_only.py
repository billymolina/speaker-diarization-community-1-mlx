"""
Compare the ResNet34 part of the model only (after feature extraction).

The pyannote model has two parts:
1. Audio preprocessing (waveform → mel spectrogram)
2. ResNet34 embedding network

Our MLX model is just the ResNet34 part, so we need to compare them at the mel spectrogram level.
"""

import numpy as np
import mlx.core as mx
import torch
from pathlib import Path


def load_pytorch_resnet():
    """Load the PyTorch model and extract the ResNet part."""
    from pyannote.audio import Model

    print("[INFO] Loading PyTorch model...")
    full_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    full_model.eval()

    # The resnet part is accessible via full_model.resnet
    resnet = full_model.resnet
    resnet.eval()

    print(f"[INFO] PyTorch ResNet loaded")
    return resnet, full_model


def load_mlx_model():
    """Load the MLX ResNet model."""
    from src.resnet_embedding import load_resnet34_embedding

    print(f"[INFO] Loading MLX model...")
    model = load_resnet34_embedding("models/embedding_mlx/weights.npz")
    return model


def create_mel_spectrogram_input(batch_size=2, time_frames=100, freq_bins=80):
    """
    Create random mel spectrogram input in both PyTorch and MLX formats.

    PyTorch ResNet expects: (batch, channels, time, freq)
    MLX ResNet expects: (batch, time, freq, channels)
    """
    print(f"\n[INFO] Creating mel spectrogram input...")
    print(f"       Shape: batch={batch_size}, time={time_frames}, freq={freq_bins}")

    np.random.seed(42)
    # Create in (batch, time, freq) format
    mel_np = np.random.randn(batch_size, time_frames, freq_bins).astype(np.float32)

    # PyTorch: (batch, time, freq) - 3D
    mel_pt = torch.from_numpy(mel_np)

    # MLX: (batch, time, freq, 1) - add channel dimension
    mel_mlx = mx.array(np.expand_dims(mel_np, axis=-1))

    print(f"[INFO] PyTorch shape: {mel_pt.shape}")
    print(f"[INFO] MLX shape: {mel_mlx.shape}")

    return mel_pt, mel_mlx


def compare_outputs(pt_output, mlx_output, name="Output"):
    """Compare outputs and print statistics."""
    pt_np = pt_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)

    abs_diff = np.abs(pt_np - mlx_np)
    rel_diff = abs_diff / (np.abs(pt_np) + 1e-8)

    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    max_rel = np.max(rel_diff)
    mean_rel = np.mean(rel_diff)

    print(f"\n{name} Comparison:")
    print(f"  Shapes: PT={pt_np.shape}, MLX={mlx_np.shape}")
    print(f"  Max abs diff:  {max_abs:.6e}")
    print(f"  Mean abs diff: {mean_abs:.6e}")
    print(f"  Max rel diff:  {max_rel:.6e}")
    print(f"  Mean rel diff: {mean_rel:.6e}")

    # Sample values
    print(f"  Sample PT:  {pt_np.flatten()[:5]}")
    print(f"  Sample MLX: {mlx_np.flatten()[:5]}")

    # Check tolerance
    tolerance = 0.01
    matches = max_abs < tolerance

    print(f"  Status: {'✓ PASS' if matches else '✗ FAIL'} (tolerance={tolerance})")

    return matches


def main():
    print("=" * 70)
    print(" ResNet34 Comparison (Mel Spectrogram Input)")
    print("=" * 70)

    # Load models
    pt_resnet, pt_full_model = load_pytorch_resnet()
    mlx_model = load_mlx_model()

    # Create test inputs (mel spectrograms)
    mel_pt, mel_mlx = create_mel_spectrogram_input(batch_size=2, time_frames=100, freq_bins=80)

    print(f"\n{'='*70}")
    print("Running Forward Pass...")
    print(f"{'='*70}")

    # PyTorch forward
    print("\n[1/2] PyTorch ResNet...")
    with torch.no_grad():
        try:
            pt_result = pt_resnet(mel_pt)
            # ResNet returns tuple: (loss?, embeddings)
            if isinstance(pt_result, tuple):
                pt_output = pt_result[1]  # Take second element (embeddings)
                print(f"       (Model returned tuple, using embeddings from element [1])")
            else:
                pt_output = pt_result
            print(f"       Output shape: {pt_output.shape}")
            print(f"       Output norm: {torch.linalg.norm(pt_output, dim=1)}")
        except Exception as e:
            print(f"[ERROR] PyTorch failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # MLX forward
    print("\n[2/2] MLX ResNet...")
    try:
        mlx_output = mlx_model(mel_mlx)
        print(f"       Output shape: {mlx_output.shape}")
        print(f"       Output norm: {mx.linalg.norm(mlx_output, axis=1)}")
    except Exception as e:
        print(f"[ERROR] MLX failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare
    print(f"\n{'='*70}")
    match = compare_outputs(pt_output, mlx_output, "Embeddings")
    print(f"{'='*70}")

    # Test with different batch sizes
    print(f"\n{'='*70}")
    print("Testing Different Batch Sizes...")
    print(f"{'='*70}")

    batch_results = []
    for bs in [1, 2, 4]:
        print(f"\n[INFO] Batch size = {bs}")
        mel_pt, mel_mlx = create_mel_spectrogram_input(batch_size=bs, time_frames=150)

        with torch.no_grad():
            pt_result = pt_resnet(mel_pt)
            pt_out = pt_result[1] if isinstance(pt_result, tuple) else pt_result

        mlx_out = mlx_model(mel_mlx)

        result = compare_outputs(pt_out, mlx_out, f"Batch {bs}")
        batch_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f"  Main test: {'✓ PASS' if match else '✗ FAIL'}")
    print(f"  Batch tests: {sum(batch_results)}/{len(batch_results)} passed")

    if match and all(batch_results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
