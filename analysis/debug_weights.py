#!/usr/bin/env python3
"""
Debug weight loading by comparing individual layer outputs.
"""

import torch
import mlx.core as mx
import torchaudio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import PyannoteSegmentationModel

try:
    from pyannote.audio import Model
except ImportError:
    print("❌ pyannote.audio not available")
    sys.exit(1)

def debug_layer_by_layer():
    """Compare layer outputs step by step."""
    
    print("\n" + "="*70)
    print("Layer-by-Layer Debugging")
    print("="*70)
    
    # Load models
    print("\n[1] Loading models...")
    mlx_model = PyannoteSegmentationModel()
    mlx_model.load_weights("models/segmentation_mlx/weights.npz")
    
    pytorch_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    pytorch_model.eval()
    print("✓ Models loaded")
    
    # Load audio
    print("\n[2] Loading test audio...")
    audio_path = "/Users/billymolina/Github/3D-Speaker/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform[:, :160000]  # 10 seconds
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"✓ Audio: {waveform.shape}")
    
    # Convert to both formats
    audio_torch = waveform.unsqueeze(0)
    audio_mlx = mx.array(waveform.numpy(), dtype=mx.float32)
    
    print("\n[3] Comparing SincNet output...")
    
    # PyTorch SincNet
    with torch.no_grad():
        pytorch_sincnet_out = pytorch_model.sincnet(audio_torch[0])
    
    print(f"  PyTorch SincNet output: {pytorch_sincnet_out.shape}")
    print(f"  PyTorch range: [{pytorch_sincnet_out.min():.4f}, {pytorch_sincnet_out.max():.4f}]")
    print(f"  PyTorch mean: {pytorch_sincnet_out.mean():.4f}, std: {pytorch_sincnet_out.std():.4f}")
    
    # MLX SincNet
    mlx_sincnet_out = mlx_model.sincnet(audio_mlx)
    
    print(f"  MLX SincNet output: {mlx_sincnet_out.shape}")
    print(f"  MLX range: [{float(mx.min(mlx_sincnet_out)):.4f}, {float(mx.max(mlx_sincnet_out)):.4f}]")
    print(f"  MLX mean: {float(mx.mean(mlx_sincnet_out)):.4f}, std: {float(mx.std(mlx_sincnet_out)):.4f}")
    
    # Compare shapes - need to account for different format
    print(f"\n  Format difference:")
    print(f"    PyTorch: (batch, channels, time) = {pytorch_sincnet_out.shape}")
    print(f"    MLX: (batch, time, channels) = {mlx_sincnet_out.shape}")
    
    # Convert MLX to PyTorch format for comparison
    mlx_sincnet_np = np.array(mlx_sincnet_out)
    mlx_sincnet_torch_format = mlx_sincnet_np.transpose(0, 2, 1)  # (batch, time, channels) -> (batch, channels, time)
    
    pytorch_sincnet_np = pytorch_sincnet_out.cpu().numpy()
    
    print(f"\n  After format alignment:")
    print(f"    PyTorch: {pytorch_sincnet_np.shape}")
    print(f"    MLX: {mlx_sincnet_torch_format.shape}")
    
    # Compare values
    if pytorch_sincnet_np.shape == mlx_sincnet_torch_format.shape:
        diff = np.abs(pytorch_sincnet_np - mlx_sincnet_torch_format)
        print(f"\n  Value comparison:")
        print(f"    Max absolute difference: {diff.max():.6f}")
        print(f"    Mean absolute difference: {diff.mean():.6f}")
        print(f"    Median absolute difference: {np.median(diff):.6f}")
        
        correlation = np.corrcoef(pytorch_sincnet_np.flatten(), mlx_sincnet_torch_format.flatten())[0, 1]
        print(f"    Correlation: {correlation:.6f}")
        
        if correlation < 0.99:
            print(f"\n  ⚠️  SincNet outputs differ significantly!")
            print(f"     This suggests issues in:")
            print(f"       - SincConv1d implementation")
            print(f"       - Conv1d layers")
            print(f"       - InstanceNorm")
            print(f"       - Weight loading for SincNet")
    
    # Check specific weights
    print("\n[4] Checking SincNet weights...")
    
    # Check PyTorch SincNet weights
    pytorch_sincnet = pytorch_model.sincnet
    print(f"\n  PyTorch SincNet structure:")
    for name, param in pytorch_sincnet.named_parameters():
        if 'filterbank' in name:
            print(f"    {name}: {param.shape}, range=[{param.min():.4f}, {param.max():.4f}]")
    
    # Check MLX weights
    weights = mx.load("models/segmentation_mlx/weights.npz")
    print(f"\n  MLX loaded weights:")
    for key in sorted(weights.keys()):
        if 'sincnet' in key and 'filterbank' in key:
            w = weights[key]
            print(f"    {key}: {w.shape}, range=[{float(mx.min(w)):.4f}, {float(mx.max(w)):.4f}]")
    
    # Check if filterbank weights match
    print(f"\n[5] Comparing filterbank parameters...")
    pt_low_hz = pytorch_sincnet.conv1d[0].filterbank.low_hz_.data
    pt_band_hz = pytorch_sincnet.conv1d[0].filterbank.band_hz_.data
    
    mlx_low_hz = mlx_model.sincnet.sinc_conv.low_hz_
    mlx_band_hz = mlx_model.sincnet.sinc_conv.band_hz_
    
    print(f"  low_hz shapes: PyTorch={pt_low_hz.shape}, MLX={mlx_low_hz.shape}")
    print(f"  band_hz shapes: PyTorch={pt_band_hz.shape}, MLX={mlx_band_hz.shape}")
    
    # Compare values
    pt_low_np = pt_low_hz.cpu().numpy()
    mlx_low_np = np.array(mlx_low_hz)
    
    if pt_low_np.shape == mlx_low_np.shape:
        low_diff = np.abs(pt_low_np - mlx_low_np)
        print(f"\n  low_hz comparison:")
        print(f"    Max diff: {low_diff.max():.6f}")
        print(f"    Mean diff: {low_diff.mean():.6f}")
        if low_diff.max() > 0.01:
            print(f"    ⚠️  Filterbank parameters don't match!")
    
    print("\n" + "="*70)
    print("Debug Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        debug_layer_by_layer()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
