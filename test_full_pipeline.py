"""
Test the full pipeline: SincNet -> LSTM with actual audio input.
This matches what happens in the real model.
"""
import torch
import mlx.core as mx
import numpy as np

print("=" * 80)
print("FULL PIPELINE: SincNet -> LSTM")
print("=" * 80)

# Load test audio
test_file = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"

try:
    import torchaudio
    waveform, sample_rate = torchaudio.load(test_file)
    print(f"Loaded audio: {waveform.shape}, sample_rate={sample_rate}")
    
    # Take first 5 seconds
    audio_np = waveform[:, :sample_rate * 5].numpy()[0]  # (80000,)
    print(f"Using {len(audio_np)/sample_rate:.1f} seconds of audio")
    
except Exception as e:
    print(f"Could not load audio: {e}")
    print("Using synthetic audio instead")
    np.random.seed(42)
    audio_np = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds at 16kHz

# ============================================================================
# GET PYTORCH SINCNET OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH SINCNET")
print("=" * 80)

# Load PyTorch model
from pyannote.audio import Model as PyanoteModel
pytorch_model = PyanoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Get SincNet output from PyTorch
with torch.no_grad():
    audio_pt = torch.from_numpy(audio_np)[None, :]
    sincnet_out_pt = pytorch_model.sincnet(audio_pt)
    print(f"PyTorch SincNet output: {sincnet_out_pt.shape}")
    print(f"PyTorch SincNet range: [{sincnet_out_pt.min():.6f}, {sincnet_out_pt.max():.6f}]")

# Convert to MLX array for testing and transpose to (batch, seq, features)
sincnet_out_np = sincnet_out_pt.numpy().transpose(0, 2, 1)  # (1, 293, 60)
sincnet_out = mx.array(sincnet_out_np)
print(f"\nConverted to MLX array: {sincnet_out.shape}")
print(f"SincNet output range: [{sincnet_out.min():.6f}, {sincnet_out.max():.6f}]")
print(f"SincNet output mean: {sincnet_out.mean():.6f}, std: {sincnet_out.std():.6f}")

# ============================================================================
# MLX LSTM
# ============================================================================
print("\n" + "=" * 80)
print("MLX LSTM")
print("=" * 80)

# Create MLX LSTM and load weights
import mlx.nn as mlx_nn
lstm_mlx = mlx_nn.LSTM(60, 128)
weights = mx.load("models/segmentation_mlx/weights.npz")
lstm_mlx.Wx = weights['lstm.weight_ih_l0']
lstm_mlx.Wh = weights['lstm.weight_hh_l0']
lstm_mlx.bias = weights['lstm.bias_ih_l0'] + weights['lstm.bias_hh_l0']

# Process through first LSTM layer
h_mlx, c_mlx = lstm_mlx(sincnet_out)
print(f"\nFirst LSTM forward output shape: {h_mlx.shape}")
print(f"First LSTM output range: [{h_mlx.min():.6f}, {h_mlx.max():.6f}]")
print(f"First LSTM output mean: {h_mlx.mean():.6f}, std: {h_mlx.std():.6f}")
print(f"First LSTM cell range: [{c_mlx.min():.6f}, {c_mlx.max():.6f}]")

# Check timestep-by-timestep
print(f"\nFirst 10 timesteps:")
for t in range(min(10, h_mlx.shape[1])):
    h_t = h_mlx[0, t, :]
    print(f"  t={t}: [{h_t.min():.6f}, {h_t.max():.6f}], std={h_t.std():.6f}")

# ============================================================================
# PYTORCH COMPARISON (using PyTorch LSTM directly)
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH LSTM (with MLX SincNet output)")
print("=" * 80)

import torch.nn as nn

# Create PyTorch LSTM
lstm_pt = nn.LSTM(60, 128, batch_first=True)

# Load weights
weights = mx.load("models/segmentation_mlx/weights.npz")
w_ih = weights['lstm.weight_ih_l0']
w_hh = weights['lstm.weight_hh_l0']
b_ih = weights['lstm.bias_ih_l0']
b_hh = weights['lstm.bias_hh_l0']

with torch.no_grad():
    lstm_pt.weight_ih_l0.copy_(torch.from_numpy(np.array(w_ih)))
    lstm_pt.weight_hh_l0.copy_(torch.from_numpy(np.array(w_hh)))
    lstm_pt.bias_ih_l0.copy_(torch.from_numpy(np.array(b_ih)))
    lstm_pt.bias_hh_l0.copy_(torch.from_numpy(np.array(b_hh)))

# Use the SAME SincNet output from MLX (transposed to match)
sincnet_out_pt_for_lstm = sincnet_out_pt.transpose(1, 2)  # (1, 293, 60)
print(f"Using PyTorch SincNet output (transposed): {sincnet_out_pt_for_lstm.shape}")

with torch.no_grad():
    h_pt, (h_final_pt, c_final_pt) = lstm_pt(sincnet_out_pt_for_lstm)

print(f"\nPyTorch LSTM output shape: {h_pt.shape}")
print(f"PyTorch LSTM output range: [{h_pt.min():.6f}, {h_pt.max():.6f}]")
print(f"PyTorch LSTM output mean: {h_pt.mean():.6f}, std: {h_pt.std():.6f}")
print(f"PyTorch LSTM cell range: [{c_final_pt.min():.6f}, {c_final_pt.max():.6f}]")

print(f"\nFirst 10 timesteps:")
for t in range(min(10, h_pt.shape[1])):
    h_t = h_pt[0, t, :]
    print(f"  t={t}: [{h_t.min():.6f}, {h_t.max():.6f}], std={h_t.std():.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

h_pt_np = h_pt.numpy()
h_mlx_np = np.array(h_mlx)

diff = np.abs(h_pt_np - h_mlx_np)
print(f"\nDifference:")
print(f"  Max absolute difference: {diff.max():.10f}")
print(f"  Mean absolute difference: {diff.mean():.10f}")
print(f"  Correlation: {np.corrcoef(h_pt_np.flatten(), h_mlx_np.flatten())[0, 1]:.6f}")

print(f"\nFirst 10 timesteps:")
for t in range(min(10, h_mlx.shape[1])):
    diff_t = np.abs(h_pt_np[0, t, :] - h_mlx_np[0, t, :])
    corr = np.corrcoef(h_pt_np[0, t, :], h_mlx_np[0, t, :])[0, 1]
    print(f"  t={t}: max_diff={diff_t.max():.10f}, mean_diff={diff_t.mean():.10f}, corr={corr:.6f}")

if diff.max() < 1e-5:
    print("\n✓ OUTPUTS MATCH!")
else:
    print(f"\n✗ OUTPUTS DIFFER! (max difference: {diff.max():.10f})")
