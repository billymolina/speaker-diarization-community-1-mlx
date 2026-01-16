"""
Exactly replicate what compare_model_outputs.py does to debug the difference
"""
import torch
import mlx.core as mx
import torchaudio
import numpy as np
from src.models import load_pyannote_model
from pyannote.audio import Model

print("=" * 80)
print("EXACT COMPARISON REPLICATION")
print("=" * 80)

# Load models (exactly as in compare_model_outputs.py)
print("\n[1] Loading models...")
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
pytorch_model = Model.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()
print("✓ Both models loaded")

# Load audio (exactly as in compare_model_outputs.py)
print("\n[2] Loading test audio...")
audio_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
waveform, sr = torchaudio.load(audio_path)

# Take first 10 seconds
samples = 160000
waveform = waveform[:, :samples]

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

print(f"✓ Audio loaded: {waveform.shape}")

# Run PyTorch inference (exactly as in compare_model_outputs.py)
print("\n[3] Running PyTorch inference...")
with torch.no_grad():
    pytorch_logits = pytorch_model(waveform.unsqueeze(0))

print(f"  PyTorch output shape: {pytorch_logits.shape}")
print(f"  PyTorch logits range: [{pytorch_logits.min():.3f}, {pytorch_logits.max():.3f}]")

# Run MLX inference (exactly as in compare_model_outputs.py)
print("\n[4] Running MLX inference...")
audio_mlx = mx.array(waveform.numpy(), dtype=mx.float32)
print(f"  MLX input shape: {audio_mlx.shape}")
mlx_logits = mlx_model(audio_mlx)

print(f"  MLX output shape: {mlx_logits.shape}")
print(f"  MLX logits range: [{float(mx.min(mlx_logits)):.3f}, {float(mx.max(mlx_logits)):.3f}]")

# Final comparison
print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

pytorch_logits_np = pytorch_logits[0].numpy()
mlx_logits_np = np.array(mlx_logits[0])

final_diff = np.abs(pytorch_logits_np - mlx_logits_np)
print(f"\nFinal logits difference:")
print(f"  Max: {final_diff.max():.10f}")
print(f"  Mean: {final_diff.mean():.10f}")
print(f"  PyTorch range: [{pytorch_logits_np.min():.6f}, {pytorch_logits_np.max():.6f}]")
print(f"  MLX range: [{mlx_logits_np.min():.6f}, {mlx_logits_np.max():.6f}]")

if final_diff.max() < 1e-4:
    print("\n✓ MODELS MATCH!")
else:
    print(f"\n✗ MODELS DIFFER! (max difference: {final_diff.max():.10f})")
    
    # Check if it's using the real audio file path issue
    print("\n[DEBUG] Checking PyTorch forward pass details...")
    with torch.no_grad():
        # Check what PyTorch actually receives
        print(f"  PyTorch input shape to forward: {waveform.unsqueeze(0).shape}")
        print(f"  PyTorch input type: {waveform.dtype}")
        
        # Check MLX input
        print(f"  MLX input shape to forward: {audio_mlx.shape}")
        print(f"  MLX input dtype: {audio_mlx.dtype}")
