"""
Deep comparison between PyTorch (original) and MLX (converted) ResNet34 embedding models.

This script performs comprehensive testing to ensure the MLX conversion is accurate.
"""

import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
import sys


def load_pytorch_model():
    """Load the original PyTorch model from pyannote."""
    try:
        from pyannote.audio import Model
        import torch

        print("[INFO] Loading PyTorch model from pyannote.audio...")
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        model.eval()  # Set to evaluation mode

        print(f"[INFO] PyTorch model loaded successfully")
        print(f"       Device: {next(model.parameters()).device}")

        return model, torch

    except ImportError as e:
        print(f"[ERROR] Failed to import required packages: {e}")
        print("       Please install: pip install pyannote.audio torch")
        sys.exit(1)


def load_mlx_model(weights_path: str):
    """Load the converted MLX model."""
    from src.resnet_embedding import load_resnet34_embedding

    print(f"[INFO] Loading MLX model from {weights_path}...")
    model = load_resnet34_embedding(weights_path)
    print(f"[INFO] MLX model loaded successfully")

    return model


def create_test_inputs(torch, batch_size=2, time_frames=500, freq_bins=80):
    """
    Create identical test inputs for both models.

    Note: pyannote models typically expect at least 400-500 frames due to windowing.

    Returns:
        pytorch_input: Tensor in PyTorch format (batch, channels, time, freq)
        mlx_input: Array in MLX format (batch, time, freq, channels)
    """
    print(f"\n[INFO] Creating test inputs...")
    print(f"       Shape: batch={batch_size}, time={time_frames}, freq={freq_bins}")

    # Create random input in numpy
    np.random.seed(42)  # For reproducibility
    input_np = np.random.randn(batch_size, 1, time_frames, freq_bins).astype(np.float32)

    # PyTorch format: (batch, channels, time, freq)
    pytorch_input = torch.from_numpy(input_np)

    # MLX format: (batch, time, freq, channels)
    # Need to transpose from (batch, channels, time, freq) to (batch, time, freq, channels)
    mlx_input_np = np.transpose(input_np, (0, 2, 3, 1))
    mlx_input = mx.array(mlx_input_np)

    print(f"[INFO] PyTorch input shape: {pytorch_input.shape}")
    print(f"[INFO] MLX input shape: {mlx_input.shape}")

    return pytorch_input, mlx_input


def compare_outputs(pytorch_output, mlx_output, name="Output", tolerance=1e-3):
    """
    Compare PyTorch and MLX outputs and report differences.

    Returns:
        bool: True if outputs match within tolerance
    """
    # Convert to numpy for comparison
    if hasattr(pytorch_output, 'detach'):
        pt_np = pytorch_output.detach().cpu().numpy()
    else:
        pt_np = pytorch_output

    mlx_np = np.array(mlx_output)

    # Compute differences
    abs_diff = np.abs(pt_np - mlx_np)
    rel_diff = abs_diff / (np.abs(pt_np) + 1e-8)

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    # Check if within tolerance
    matches = max_abs_diff < tolerance

    # Print results
    print(f"\n{'='*60}")
    print(f"{name} Comparison:")
    print(f"{'='*60}")
    print(f"  Shapes:  PyTorch={pt_np.shape}, MLX={mlx_np.shape}")
    print(f"  Max absolute difference:  {max_abs_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"  Max relative difference:  {max_rel_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")
    print(f"  Within tolerance ({tolerance}): {'✓ YES' if matches else '✗ NO'}")

    # Show some sample values
    print(f"\n  Sample values (first 5):")
    print(f"    PyTorch: {pt_np.flatten()[:5]}")
    print(f"    MLX:     {mlx_np.flatten()[:5]}")
    print(f"    Diff:    {abs_diff.flatten()[:5]}")

    return matches


def test_forward_pass(pytorch_model, mlx_model, pytorch_input, mlx_input, torch):
    """Test complete forward pass and compare outputs."""
    print(f"\n{'='*70}")
    print(f"TEST 1: Forward Pass Comparison")
    print(f"{'='*70}")

    # Run PyTorch model
    print("\n[1/2] Running PyTorch forward pass...")
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input)

    print(f"       PyTorch output shape: {pytorch_output.shape}")
    print(f"       PyTorch output norm: {torch.linalg.norm(pytorch_output, dim=1)}")

    # Run MLX model
    print("\n[2/2] Running MLX forward pass...")
    try:
        mlx_output = mlx_model(mlx_input)
        print(f"       MLX output shape: {mlx_output.shape}")
        print(f"       MLX output norm: {mx.linalg.norm(mlx_output, axis=1)}")
    except Exception as e:
        print(f"[ERROR] MLX forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare outputs
    match = compare_outputs(pytorch_output, mlx_output, "Final Embeddings", tolerance=1e-2)

    return match


def test_embedding_similarity(pytorch_model, mlx_model, torch):
    """Test that embedding similarities are preserved."""
    print(f"\n{'='*70}")
    print(f"TEST 2: Embedding Similarity Preservation")
    print(f"{'='*70}")

    # Create three different inputs (use 500 frames minimum)
    np.random.seed(123)

    # Input 1 and 2 are similar
    input1_np = np.random.randn(1, 1, 500, 80).astype(np.float32)
    input2_np = input1_np + np.random.randn(1, 1, 500, 80).astype(np.float32) * 0.1

    # Input 3 is very different
    input3_np = np.random.randn(1, 1, 500, 80).astype(np.float32)

    # Convert to both formats
    def to_both_formats(input_np):
        pt_input = torch.from_numpy(input_np)
        mlx_input = mx.array(np.transpose(input_np, (0, 2, 3, 1)))
        return pt_input, mlx_input

    pt1, mlx1 = to_both_formats(input1_np)
    pt2, mlx2 = to_both_formats(input2_np)
    pt3, mlx3 = to_both_formats(input3_np)

    # Get embeddings
    print("\n[INFO] Computing embeddings...")
    with torch.no_grad():
        pt_emb1 = pytorch_model(pt1)[0]
        pt_emb2 = pytorch_model(pt2)[0]
        pt_emb3 = pytorch_model(pt3)[0]

    mlx_emb1 = mlx_model(mlx1)[0]
    mlx_emb2 = mlx_model(mlx2)[0]
    mlx_emb3 = mlx_model(mlx3)[0]

    # Compute similarities (cosine similarity for L2-normalized vectors)
    pt_sim_12 = float(torch.sum(pt_emb1 * pt_emb2))
    pt_sim_13 = float(torch.sum(pt_emb1 * pt_emb3))

    mlx_sim_12 = float(mx.sum(mlx_emb1 * mlx_emb2))
    mlx_sim_13 = float(mx.sum(mlx_emb1 * mlx_emb3))

    print(f"\nSimilarity Results:")
    print(f"  Similar inputs (1 vs 2):")
    print(f"    PyTorch:  {pt_sim_12:.4f}")
    print(f"    MLX:      {mlx_sim_12:.4f}")
    print(f"    Diff:     {abs(pt_sim_12 - mlx_sim_12):.4f}")

    print(f"\n  Different inputs (1 vs 3):")
    print(f"    PyTorch:  {pt_sim_13:.4f}")
    print(f"    MLX:      {mlx_sim_13:.4f}")
    print(f"    Diff:     {abs(pt_sim_13 - mlx_sim_13):.4f}")

    # Check that similarity relationships are preserved
    sim_diff_12 = abs(pt_sim_12 - mlx_sim_12)
    sim_diff_13 = abs(pt_sim_13 - mlx_sim_13)

    tolerance = 0.05  # Allow 5% difference in similarity scores
    match = (sim_diff_12 < tolerance) and (sim_diff_13 < tolerance)

    print(f"\n  Similarity preservation (tolerance={tolerance}):")
    print(f"    {'✓ PASS' if match else '✗ FAIL'}")

    return match


def test_batch_processing(pytorch_model, mlx_model, torch):
    """Test that both models handle batching correctly."""
    print(f"\n{'='*70}")
    print(f"TEST 3: Batch Processing")
    print(f"{'='*70}")

    results = []

    for batch_size in [1, 2, 4]:
        print(f"\n[INFO] Testing batch_size={batch_size}...")

        pt_input, mlx_input = create_test_inputs(torch, batch_size=batch_size, time_frames=500)

        # Run both models
        with torch.no_grad():
            pt_output = pytorch_model(pt_input)

        try:
            mlx_output = mlx_model(mlx_input)

            # Compare
            match = compare_outputs(pt_output, mlx_output, f"Batch {batch_size}", tolerance=1e-2)
            results.append(match)

        except Exception as e:
            print(f"[ERROR] Batch size {batch_size} failed: {e}")
            results.append(False)

    all_passed = all(results)
    print(f"\n[INFO] Batch processing: {sum(results)}/{len(results)} tests passed")

    return all_passed


def test_different_lengths(pytorch_model, mlx_model, torch):
    """Test with different input sequence lengths."""
    print(f"\n{'='*70}")
    print(f"TEST 4: Variable Sequence Lengths")
    print(f"{'='*70}")

    results = []

    for time_frames in [500, 600, 800, 1000]:
        print(f"\n[INFO] Testing time_frames={time_frames}...")

        pt_input, mlx_input = create_test_inputs(torch, batch_size=1, time_frames=time_frames)

        # Run both models
        with torch.no_grad():
            pt_output = pytorch_model(pt_input)

        try:
            mlx_output = mlx_model(mlx_input)

            # Just check they produce valid outputs
            valid = (pt_output.shape[-1] == 256) and (mlx_output.shape[-1] == 256)

            if valid:
                match = compare_outputs(pt_output, mlx_output, f"Length {time_frames}", tolerance=1e-2)
                results.append(match)
            else:
                print(f"[ERROR] Invalid output shapes: PT={pt_output.shape}, MLX={mlx_output.shape}")
                results.append(False)

        except Exception as e:
            print(f"[ERROR] Length {time_frames} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    all_passed = all(results)
    print(f"\n[INFO] Variable length: {sum(results)}/{len(results)} tests passed")

    return all_passed


def test_embedding_statistics(pytorch_model, mlx_model, torch, num_samples=100):
    """Test embedding statistics across many samples."""
    print(f"\n{'='*70}")
    print(f"TEST 5: Embedding Statistics ({num_samples} samples)")
    print(f"{'='*70}")

    print(f"\n[INFO] Generating {num_samples} random embeddings...")

    pt_embeddings = []
    mlx_embeddings = []

    for i in range(num_samples):
        if (i + 1) % 20 == 0:
            print(f"       Progress: {i+1}/{num_samples}")

        # Random input (use 500 frames)
        pt_input, mlx_input = create_test_inputs(torch, batch_size=1, time_frames=500)

        # Get embeddings
        with torch.no_grad():
            pt_emb = pytorch_model(pt_input)[0].cpu().numpy()
        mlx_emb = np.array(mlx_model(mlx_input)[0])

        pt_embeddings.append(pt_emb)
        mlx_embeddings.append(mlx_emb)

    pt_embeddings = np.array(pt_embeddings)
    mlx_embeddings = np.array(mlx_embeddings)

    # Compute statistics
    pt_mean = np.mean(pt_embeddings, axis=0)
    mlx_mean = np.mean(mlx_embeddings, axis=0)

    pt_std = np.std(pt_embeddings, axis=0)
    mlx_std = np.std(mlx_embeddings, axis=0)

    # Check norms (should be ~1 for L2 normalized vectors)
    pt_norms = np.linalg.norm(pt_embeddings, axis=1)
    mlx_norms = np.linalg.norm(mlx_embeddings, axis=1)

    print(f"\nEmbedding Statistics:")
    print(f"  Mean values:")
    print(f"    PyTorch mean: {np.mean(pt_mean):.6f} (std: {np.std(pt_mean):.6f})")
    print(f"    MLX mean:     {np.mean(mlx_mean):.6f} (std: {np.std(mlx_mean):.6f})")
    print(f"    Difference:   {np.mean(np.abs(pt_mean - mlx_mean)):.6f}")

    print(f"\n  Standard deviation:")
    print(f"    PyTorch std: {np.mean(pt_std):.6f}")
    print(f"    MLX std:     {np.mean(mlx_std):.6f}")

    print(f"\n  L2 Norms:")
    print(f"    PyTorch: mean={np.mean(pt_norms):.4f}, std={np.std(pt_norms):.4f}")
    print(f"    MLX:     mean={np.mean(mlx_norms):.4f}, std={np.std(mlx_norms):.4f}")

    # Check that norms are close to 1
    norm_ok = (np.abs(np.mean(pt_norms) - 1.0) < 0.01) and (np.abs(np.mean(mlx_norms) - 1.0) < 0.01)

    print(f"\n  L2 normalization: {'✓ OK' if norm_ok else '✗ FAIL'}")

    return norm_ok


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and MLX ResNet34 embedding models"
    )
    parser.add_argument(
        "--mlx-weights",
        type=str,
        default="models/embedding_mlx/weights.npz",
        help="Path to MLX weights file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for statistics test"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" ResNet34 Embedding Model: PyTorch vs MLX Comparison")
    print("=" * 70)

    # Load models
    pytorch_model, torch = load_pytorch_model()
    mlx_model = load_mlx_model(args.mlx_weights)

    # Create test inputs
    pytorch_input, mlx_input = create_test_inputs(torch)

    # Run tests
    results = {}

    try:
        results['forward_pass'] = test_forward_pass(pytorch_model, mlx_model, pytorch_input, mlx_input, torch)
    except Exception as e:
        print(f"\n[ERROR] Forward pass test failed: {e}")
        results['forward_pass'] = False

    try:
        results['similarity'] = test_embedding_similarity(pytorch_model, mlx_model, torch)
    except Exception as e:
        print(f"\n[ERROR] Similarity test failed: {e}")
        results['similarity'] = False

    try:
        results['batch'] = test_batch_processing(pytorch_model, mlx_model, torch)
    except Exception as e:
        print(f"\n[ERROR] Batch test failed: {e}")
        results['batch'] = False

    try:
        results['variable_length'] = test_different_lengths(pytorch_model, mlx_model, torch)
    except Exception as e:
        print(f"\n[ERROR] Variable length test failed: {e}")
        results['variable_length'] = False

    try:
        results['statistics'] = test_embedding_statistics(pytorch_model, mlx_model, torch, args.num_samples)
    except Exception as e:
        print(f"\n[ERROR] Statistics test failed: {e}")
        results['statistics'] = False

    # Print summary
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)

    test_names = {
        'forward_pass': 'Forward Pass Comparison',
        'similarity': 'Similarity Preservation',
        'batch': 'Batch Processing',
        'variable_length': 'Variable Sequence Lengths',
        'statistics': 'Embedding Statistics',
    }

    for key, name in test_names.items():
        status = "✓ PASS" if results.get(key, False) else "✗ FAIL"
        print(f"  {status:8s} {name}")

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! MLX model matches PyTorch model.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the differences above.")
        return 1


if __name__ == "__main__":
    exit(main())
