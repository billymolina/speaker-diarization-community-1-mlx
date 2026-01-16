#!/usr/bin/env python3
"""
Compare raw model outputs (logits) between MLX and PyTorch on same audio chunk.
This will tell us if the difference is in the model or post-processing.
"""

import torch
import mlx.core as mx
import torchaudio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models import load_pyannote_model

try:
    from pyannote.audio import Model
except ImportError:
    print("❌ pyannote.audio not available")
    sys.exit(1)

def compare_raw_outputs():
    """Compare raw logits from both models on same audio."""
    
    print("\n" + "="*70)
    print("Raw Model Output Comparison")
    print("="*70)
    
    # Load both models
    print("\n[1] Loading models...")
    mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
    pytorch_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    pytorch_model.eval()
    print("✓ Both models loaded")
    
    # Load a short audio clip (10 seconds)
    print("\n[2] Loading test audio...")
    audio_path = "/Users/billymolina/Github/3D-Speaker/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
    waveform, sr = torchaudio.load(audio_path)
    
    # Take first 10 seconds
    samples = 160000
    waveform = waveform[:, :samples]
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"✓ Audio loaded: {waveform.shape}")
    
    # Run PyTorch inference
    print("\n[3] Running PyTorch inference...")
    with torch.no_grad():
        pytorch_logits = pytorch_model(waveform.unsqueeze(0))
    
    pytorch_probs = torch.softmax(pytorch_logits, dim=-1)
    pytorch_preds = torch.argmax(pytorch_probs, dim=-1)
    
    print(f"  PyTorch output shape: {pytorch_logits.shape}")
    print(f"  PyTorch logits range: [{pytorch_logits.min():.3f}, {pytorch_logits.max():.3f}]")
    
    # Run MLX inference
    print("\n[4] Running MLX inference...")
    # MLX model expects (batch, samples) just like PyTorch after removing channel dim
    audio_mlx = mx.array(waveform.numpy(), dtype=mx.float32)  # Already (1, samples)
    mlx_logits = mlx_model(audio_mlx)
    
    mlx_probs = mx.softmax(mlx_logits, axis=-1)
    mlx_preds = mx.argmax(mlx_probs, axis=-1)
    
    print(f"  MLX output shape: {mlx_logits.shape}")
    print(f"  MLX logits range: [{float(mx.min(mlx_logits)):.3f}, {float(mx.max(mlx_logits)):.3f}]")
    
    # Compare shapes
    print("\n[5] Shape comparison:")
    pytorch_shape = pytorch_logits.shape
    mlx_shape = mlx_logits.shape
    
    if pytorch_shape[1] != mlx_shape[1]:
        print(f"  ⚠️  Time dimension mismatch!")
        print(f"     PyTorch: {pytorch_shape[1]} frames")
        print(f"     MLX: {mlx_shape[1]} frames")
        print(f"     Difference: {abs(pytorch_shape[1] - mlx_shape[1])} frames")
    else:
        print(f"  ✓ Time dimension matches: {pytorch_shape[1]} frames")
    
    if pytorch_shape[2] != mlx_shape[2]:
        print(f"  ⚠️  Class dimension mismatch!")
    else:
        print(f"  ✓ Class dimension matches: {pytorch_shape[2]} classes")
    
    # Compare actual values frame-by-frame
    print("\n[6] Comparing first 10 frames:")
    print("-" * 70)
    
    pytorch_np = pytorch_logits[0].cpu().numpy()
    mlx_np = np.array(mlx_logits[0])
    
    # Take minimum number of frames
    min_frames = min(pytorch_np.shape[0], mlx_np.shape[0])
    
    print(f"{'Frame':<8} {'PyTorch Pred':<15} {'MLX Pred':<15} {'Match':<8} {'L2 Distance'}")
    print("-" * 70)
    
    matches = 0
    total_distance = 0.0
    
    for i in range(min(10, min_frames)):
        pt_pred = int(torch.argmax(torch.from_numpy(pytorch_np[i])))
        mlx_pred = int(mx.argmax(mx.array(mlx_np[i])))
        
        # Calculate L2 distance between logit vectors
        distance = np.linalg.norm(pytorch_np[i] - mlx_np[i])
        total_distance += distance
        
        match = "✓" if pt_pred == mlx_pred else "✗"
        if pt_pred == mlx_pred:
            matches += 1
        
        print(f"{i:<8} Class {pt_pred:<10} Class {mlx_pred:<10} {match:<8} {distance:.4f}")
    
    print("-" * 70)
    print(f"Agreement: {matches}/10 frames ({matches*10}%)")
    print(f"Average L2 distance: {total_distance/10:.4f}")
    
    # Compare over all frames
    print("\n[7] Full frame comparison:")
    
    all_matches = 0
    for i in range(min_frames):
        pt_pred = int(torch.argmax(torch.from_numpy(pytorch_np[i])))
        mlx_pred = int(mx.argmax(mx.array(mlx_np[i])))
        if pt_pred == mlx_pred:
            all_matches += 1
    
    agreement = (all_matches / min_frames) * 100
    print(f"  Frame agreement: {all_matches}/{min_frames} ({agreement:.1f}%)")
    
    # Statistical comparison
    print("\n[8] Statistical comparison:")
    
    # Compare mean logits
    pt_mean = pytorch_np.mean(axis=0)
    mlx_mean = mlx_np[:min_frames].mean(axis=0)
    
    print(f"  Mean logits per class:")
    print(f"    {'Class':<10} {'PyTorch':<12} {'MLX':<12} {'Diff':<12}")
    for i in range(7):
        diff = abs(pt_mean[i] - mlx_mean[i])
        print(f"    Class {i:<5} {pt_mean[i]:>10.4f}  {mlx_mean[i]:>10.4f}  {diff:>10.4f}")
    
    # Correlation
    flat_pytorch = pytorch_np[:min_frames].flatten()
    flat_mlx = mlx_np[:min_frames].flatten()
    correlation = np.corrcoef(flat_pytorch, flat_mlx)[0, 1]
    
    print(f"\n  Logit correlation: {correlation:.4f}")
    
    if correlation > 0.99:
        print("  ✓ Extremely high correlation - models are virtually identical")
    elif correlation > 0.95:
        print("  ✓ Very high correlation - models are very similar")
    elif correlation > 0.90:
        print("  ⚠️  Good correlation but some differences exist")
    else:
        print("  ⚠️  Low correlation - significant model differences!")
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70 + "\n")
    
    return {
        'frame_agreement': agreement,
        'correlation': correlation,
        'avg_distance': total_distance / 10
    }

if __name__ == "__main__":
    try:
        results = compare_raw_outputs()
        
        print("\nDiagnosis:")
        if results['correlation'] > 0.99 and results['frame_agreement'] > 95:
            print("✅ Models are nearly identical - differences are in post-processing")
        elif results['correlation'] > 0.95:
            print("⚠️  Models are similar but have some differences")
            print("   Possible causes: weight loading issues, numerical precision")
        else:
            print("❌ Models produce significantly different outputs")
            print("   Possible causes: incorrect weight mapping, architecture differences")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
