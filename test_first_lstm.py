"""Test just the first LSTM layer"""
import torch
import mlx.core as mx
import torchaudio
import numpy as np
from pyannote.audio.models.segmentation import PyanNet
from src.models import load_pyannote_model

# Load models
print("Loading models...")
pytorch_model = PyanNet().eval()
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")

# Load audio
wav_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform[:, :160000]
print(f"Audio: {waveform.shape}")

# Get SincNet outputs
print("\n[1] SincNet outputs...")
with torch.no_grad():
    sincnet_pt = pytorch_model.sincnet(waveform)
sincnet_pt_transposed = sincnet_pt.permute(0, 2, 1)
print(f"  PyTorch: {sincnet_pt_transposed.shape}")

waveform_mlx = mx.array(waveform.numpy())
sincnet_mlx = mlx_model.sincnet(waveform_mlx)
print(f"  MLX: {sincnet_mlx.shape}")

# Test FIRST LSTM layer only
print("\n[2] First LSTM layer (forward)...")
with torch.no_grad():
    h_pt_full, _ = pytorch_model.lstm(sincnet_pt_transposed)

# For MLX, run just first forward LSTM
h_mlx_fwd, _ = mlx_model.lstm_forward[0](sincnet_mlx)
mx.eval(h_mlx_fwd)

print(f"  PyTorch full LSTM: {h_pt_full.shape}, range=[{h_pt_full.min():.4f}, {h_pt_full.max():.4f}]")
print(f"  MLX first fwd: {h_mlx_fwd.shape}, range=[{h_mlx_fwd.min().item():.4f}, {h_mlx_fwd.max().item():.4f}]")

# Check if MLX LSTM layer is already saturating
if h_mlx_fwd.max().item() > 0.99 or h_mlx_fwd.min().item() < -0.99:
    print("  ⚠️  MLX LSTM output is SATURATED!")
    print("  This indicates tanh output is clipping")
    
print("\n[3] Check single timestep...")
# Take first timestep input
x0_mlx = sincnet_mlx[:, 0:1, :]  # (1, 1, 60)
print(f"  Input shape: {x0_mlx.shape}")
print(f"  Input range: [{x0_mlx.min().item():.4f}, {x0_mlx.max().item():.4f}]")

h0_mlx, _ = mlx_model.lstm_forward[0](x0_mlx)
mx.eval(h0_mlx)
print(f"  Output shape: {h0_mlx.shape}")
print(f"  Output range: [{h0_mlx.min().item():.4f}, {h0_mlx.max().item():.4f}]")
print(f"  Output: {np.array(h0_mlx)[0, 0, :10]}")
