"""
Compare SincNet outputs between PyTorch and MLX models
"""
import torch
import mlx.core as mx
import numpy as np
from src.models import PyannoteSegmentationModel

print("=" * 80)
print("SINCNET COMPARISON: PyTorch vs MLX")
print("=" * 80)

# Load PyTorch model
from pyannote.audio import Model as PyanoteModel
pytorch_model = PyanoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Load MLX model
mlx_model = PyannoteSegmentationModel()

# Load MLX model weights manually
weights = mx.load("models/segmentation_mlx/weights.npz")

# SincConv1d learnable parameters
mlx_model.sincnet.sinc_conv.low_hz_ = weights['sincnet.conv1d.0.filterbank.low_hz_']
mlx_model.sincnet.sinc_conv.band_hz_ = weights['sincnet.conv1d.0.filterbank.band_hz_']

# Waveform normalization
mlx_model.sincnet.wav_norm.weight = weights['sincnet.wav_norm1d.weight']
mlx_model.sincnet.wav_norm.bias = weights['sincnet.wav_norm1d.bias']

# SincNet layer 1 normalization
mlx_model.sincnet.norm1.weight = weights['sincnet.norm1d.0.weight']
mlx_model.sincnet.norm1.bias = weights['sincnet.norm1d.0.bias']

# SincNet layer 2 (conv + norm)
conv2_weight = weights['sincnet.conv1d.1.weight']
mlx_model.sincnet.conv2.weight = mx.transpose(conv2_weight, (0, 2, 1))
mlx_model.sincnet.conv2.bias = weights['sincnet.conv1d.1.bias']
mlx_model.sincnet.norm2.weight = weights['sincnet.norm1d.1.weight']
mlx_model.sincnet.norm2.bias = weights['sincnet.norm1d.1.bias']

# SincNet layer 3 (conv + norm)
conv3_weight = weights['sincnet.conv1d.2.weight']
mlx_model.sincnet.conv3.weight = mx.transpose(conv3_weight, (0, 2, 1))
mlx_model.sincnet.conv3.bias = weights['sincnet.conv1d.2.bias']
mlx_model.sincnet.norm3.weight = weights['sincnet.norm1d.2.weight']
mlx_model.sincnet.norm3.bias = weights['sincnet.norm1d.2.bias']

print("Loaded SincNet weights")

# Create test audio
np.random.seed(42)
audio_np = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds

print(f"Input audio shape: {audio_np.shape}")

# ============================================================================
# PYTORCH SINCNET
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH SINCNET")
print("=" * 80)

with torch.no_grad():
    audio_pt = torch.from_numpy(audio_np)[None, :]
    sincnet_out_pt = pytorch_model.sincnet(audio_pt)
    print(f"Output shape: {sincnet_out_pt.shape}")
    print(f"Output range: [{sincnet_out_pt.min():.6f}, {sincnet_out_pt.max():.6f}]")
    print(f"Output mean: {sincnet_out_pt.mean():.6f}, std: {sincnet_out_pt.std():.6f}")
    
    # Transpose to (batch, time, features) for comparison
    sincnet_out_pt_transposed = sincnet_out_pt.transpose(1, 2)
    print(f"Transposed shape: {sincnet_out_pt_transposed.shape}")

# ============================================================================
# MLX SINCNET
# ============================================================================
print("\n" + "=" * 80)
print("MLX SINCNET")
print("=" * 80)

audio_mlx = mx.array(audio_np)[None, :]
sincnet_out_mlx = mlx_model.sincnet(audio_mlx)
print(f"Output shape: {sincnet_out_mlx.shape}")
print(f"Output range: [{sincnet_out_mlx.min():.6f}, {sincnet_out_mlx.max():.6f}]")
print(f"Output mean: {sincnet_out_mlx.mean():.6f}, std: {sincnet_out_mlx.std():.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

sincnet_pt_np = sincnet_out_pt_transposed.numpy()
sincnet_mlx_np = np.array(sincnet_out_mlx)

print(f"PyTorch shape: {sincnet_pt_np.shape}")
print(f"MLX shape: {sincnet_mlx_np.shape}")

if sincnet_pt_np.shape != sincnet_mlx_np.shape:
    print("\n✗ SHAPES DON'T MATCH!")
else:
    diff = np.abs(sincnet_pt_np - sincnet_mlx_np)
    corr = np.corrcoef(sincnet_pt_np.flatten(), sincnet_mlx_np.flatten())[0, 1]
    
    print(f"\nDifference:")
    print(f"  Max absolute difference: {diff.max():.10f}")
    print(f"  Mean absolute difference: {diff.mean():.10f}")
    print(f"  Correlation: {corr:.10f}")
    
    if diff.max() < 1e-5:
        print("\n✓ OUTPUTS MATCH!")
    elif corr > 0.999:
        print("\n≈ OUTPUTS NEARLY MATCH (high correlation)")
    else:
        print(f"\n✗ OUTPUTS DIFFER!")
        
        # Sample differences
        print(f"\nFirst timestep:")
        print(f"  PyTorch (first 10 features): {sincnet_pt_np[0, 0, :10]}")
        print(f"  MLX (first 10 features): {sincnet_mlx_np[0, 0, :10]}")
        print(f"  Difference: {diff[0, 0, :10]}")
