"""Debug LSTM layer outputs"""
import torch
import mlx.core as mx
import mlx.nn as nn
import torchaudio
import numpy as np
from pyannote.audio.models.segmentation import PyanNet
from src.models import load_pyannote_model

def load_audio(duration=10):
    """Load test audio"""
    wav_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
    waveform, sr = torchaudio.load(wav_path)
    
    # Take first N seconds
    samples = int(duration * sr)
    waveform = waveform[:, :samples]
    
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    return waveform

print("="*70)
print("LSTM Layer Debugging")
print("="*70)

# Load models
print("\n[1] Loading models...")
pytorch_model = PyanNet().eval()
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
print("✓ Models loaded")

# Load test audio
print("\n[2] Loading test audio...")
waveform_pt = load_audio()
print(f"✓ Audio: {waveform_pt.shape}")

# Convert to MLX
waveform_mlx = mx.array(waveform_pt.numpy())

# Get SincNet outputs
print("\n[3] Getting SincNet outputs...")
with torch.no_grad():
    sincnet_pt = pytorch_model.sincnet(waveform_pt)
print(f"  PyTorch SincNet: {sincnet_pt.shape}")

sincnet_mlx = mlx_model.sincnet(waveform_mlx)
print(f"  MLX SincNet: {sincnet_mlx.shape}")

# Transpose MLX to match PyTorch format (B, C, T) -> (B, T, C)
sincnet_pt_transposed = sincnet_pt.permute(0, 2, 1)
print(f"  PyTorch transposed: {sincnet_pt_transposed.shape}")

# Get LSTM outputs - run through the full model layers manually
print("\n[4] Comparing LSTM outputs...")
with torch.no_grad():
    lstm_pt, _ = pytorch_model.lstm(sincnet_pt_transposed)
print(f"  PyTorch LSTM: {lstm_pt.shape}")
print(f"  PyTorch range: [{lstm_pt.min():.4f}, {lstm_pt.max():.4f}]")
print(f"  PyTorch mean: {lstm_pt.mean():.4f}, std: {lstm_pt.std():.4f}")

# For MLX, run through the LSTM layers manually
h = sincnet_mlx
for i, (lstm_fwd, lstm_back) in enumerate(zip(mlx_model.lstm_forward, mlx_model.lstm_backward)):
    # Forward direction
    h_fwd, _ = lstm_fwd(h)
    # Backward direction (reverse time axis)
    h_back, _ = lstm_back(h[:, ::-1, :])  # Reverse time axis
    h_back = h_back[:, ::-1, :]  # Reverse back
    # Concatenate
    h = mx.concatenate([h_fwd, h_back], axis=-1)

lstm_mlx = h
mx.eval(lstm_mlx)
print(f"  MLX LSTM: {lstm_mlx.shape}")
print(f"  MLX range: [{lstm_mlx.min().item():.4f}, {lstm_mlx.max().item():.4f}]")
print(f"  MLX mean: {lstm_mlx.mean().item():.4f}, std: {lstm_mlx.std().item():.4f}")

# Convert to numpy for comparison
lstm_pt_np = lstm_pt.numpy()
lstm_mlx_np = np.array(lstm_mlx)

# Compare values
print("\n[5] Value comparison:")
diff = np.abs(lstm_pt_np - lstm_mlx_np)
print(f"  Max absolute difference: {diff.max():.6f}")
print(f"  Mean absolute difference: {diff.mean():.6f}")
print(f"  Median absolute difference: {np.median(diff):.6f}")

# Correlation
flat_pt = lstm_pt_np.flatten()
flat_mlx = lstm_mlx_np.flatten()
corr = np.corrcoef(flat_pt, flat_mlx)[0, 1]
print(f"  Correlation: {corr:.6f}")

# Check specific frames
print("\n[6] First 5 frames comparison:")
for i in range(min(5, lstm_pt_np.shape[1])):
    pt_frame = lstm_pt_np[0, i, :5]
    mlx_frame = lstm_mlx_np[0, i, :5]
    print(f"  Frame {i}:")
    print(f"    PyTorch: [{', '.join([f'{x:.4f}' for x in pt_frame])}...]")
    print(f"    MLX:     [{', '.join([f'{x:.4f}' for x in mlx_frame])}...]")
    diff = np.abs(pt_frame - mlx_frame)
    print(f"    Diff:    [{', '.join([f'{x:.4f}' for x in diff])}...]")

print("\n" + "="*70)
print("Debug Complete")
print("="*70)
