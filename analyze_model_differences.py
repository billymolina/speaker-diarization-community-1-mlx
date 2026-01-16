"""
Deep analysis of model output differences to improve correlation
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import load_pyannote_model
from pyannote.audio import Model as PyannoteModel

print("=" * 80)
print("DEEP MODEL ANALYSIS - Finding Remaining Differences")
print("=" * 80)

# Load models
print("\n📦 Loading models...")
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")

pytorch_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Test on real audio segment
audio_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
print(f"📊 Loading audio: {Path(audio_path).name}")

import torchaudio
waveform, sr = torchaudio.load(audio_path)
print(f"   Sample rate: {sr}, Duration: {waveform.shape[1]/sr:.1f}s")

# Take a 5-second chunk for detailed analysis
chunk_samples = 5 * sr
waveform_chunk = waveform[:, :chunk_samples]

print(f"\n🔬 Analyzing 5-second chunk ({chunk_samples} samples)")

# Prepare inputs
print("\n1️⃣ Input Preparation:")
pytorch_input = waveform_chunk.unsqueeze(0)  # [1, 1, samples]
mlx_input = mx.array(waveform_chunk.numpy()).reshape(1, chunk_samples)  # [1, samples]

print(f"   PyTorch shape: {pytorch_input.shape}")
print(f"   MLX shape: {mlx_input.shape}")
print(f"   Input mean: PT={pytorch_input.mean():.6f}, MLX={float(mx.mean(mlx_input)):.6f}")
print(f"   Input std: PT={pytorch_input.std():.6f}, MLX={float(mx.std(mlx_input)):.6f}")

# Get outputs
print("\n2️⃣ Model Forward Pass:")
with torch.no_grad():
    pytorch_output = pytorch_model(pytorch_input)
    print(f"   PyTorch output shape: {pytorch_output.shape}")
    print(f"   PyTorch output range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
    print(f"   PyTorch output mean: {pytorch_output.mean():.6f}")
    print(f"   PyTorch output std: {pytorch_output.std():.6f}")

mlx_output = mlx_model(mlx_input)
print(f"   MLX output shape: {mlx_output.shape}")
print(f"   MLX output range: [{float(mx.min(mlx_output)):.3f}, {float(mx.max(mlx_output)):.3f}]")
print(f"   MLX output mean: {float(mx.mean(mlx_output)):.6f}")
print(f"   MLX output std: {float(mx.std(mlx_output)):.6f}")

# Convert to numpy for detailed analysis
pt_np = pytorch_output.squeeze().cpu().numpy()  # [time, classes]
mlx_np = np.array(mlx_output[0])  # [time, classes]

print(f"\n3️⃣ Output Analysis:")
print(f"   Shape: {pt_np.shape}")
print(f"   Time steps: {pt_np.shape[0]}")
print(f"   Classes: {pt_np.shape[1]}")

# Per-class analysis
print("\n4️⃣ Per-Class Analysis:")
num_classes = pt_np.shape[1]
print(f"\n   Class    PT_mean   PT_std    MLX_mean  MLX_std   Diff_mean  Diff_std  Correlation")
print("   " + "-" * 85)
correlations = []
mean_diffs = []
for c in range(num_classes):
    pt_class = pt_np[:, c]
    mlx_class = mlx_np[:, c]
    
    pt_mean = pt_class.mean()
    pt_std = pt_class.std()
    mlx_mean = mlx_class.mean()
    mlx_std = mlx_class.std()
    
    diff_mean = abs(pt_mean - mlx_mean)
    diff_std = abs(pt_std - mlx_std)
    
    corr = np.corrcoef(pt_class, mlx_class)[0, 1]
    correlations.append(corr)
    mean_diffs.append(diff_mean)
    
    print(f"   {c:5d}    {pt_mean:8.4f}  {pt_std:7.4f}   {mlx_mean:8.4f}  {mlx_std:7.4f}   {diff_mean:9.4f}  {diff_std:8.4f}  {corr:11.4f}")

print(f"\n   Overall correlation: {np.mean(correlations):.4f}")
print(f"   Min correlation: {np.min(correlations):.4f} (class {np.argmin(correlations)})")
print(f"   Max correlation: {np.max(correlations):.4f} (class {np.argmax(correlations)})")
print(f"   Mean absolute difference: {np.mean(mean_diffs):.4f}")

# Temporal analysis
print("\n5️⃣ Temporal Analysis:")
time_correlations = []
for t in range(0, pt_np.shape[0], 10):  # Every 10th frame
    if t < pt_np.shape[0]:
        corr = np.corrcoef(pt_np[t], mlx_np[t])[0, 1]
        time_correlations.append(corr)

print(f"   Frames analyzed: {len(time_correlations)}")
print(f"   Mean temporal correlation: {np.mean(time_correlations):.4f}")
print(f"   Min temporal correlation: {np.min(time_correlations):.4f}")
print(f"   Max temporal correlation: {np.max(time_correlations):.4f}")
print(f"   Std temporal correlation: {np.std(time_correlations):.4f}")

# Check if outputs are actually probabilities
print("\n6️⃣ Probability Distribution Check:")
pt_sum = np.exp(pt_np).sum(axis=1)  # Should be ~1 if log softmax
mlx_sum = np.exp(mlx_np).sum(axis=1)
print(f"   PT exp(output) sum per frame:")
print(f"     Mean: {pt_sum.mean():.6f} (should be ~1.0 for log_softmax)")
print(f"     Range: [{pt_sum.min():.6f}, {pt_sum.max():.6f}]")
print(f"   MLX exp(output) sum per frame:")
print(f"     Mean: {mlx_sum.mean():.6f} (should be ~1.0 for log_softmax)")
print(f"     Range: [{mlx_sum.min():.6f}, {mlx_sum.max():.6f}]")

# Check predictions
print("\n7️⃣ Prediction Agreement:")
pt_pred = pt_np.argmax(axis=1)
mlx_pred = mlx_np.argmax(axis=1)
agreement = (pt_pred == mlx_pred).mean()
print(f"   Prediction agreement: {agreement*100:.2f}%")
print(f"   PT most common class: {np.bincount(pt_pred).argmax()} ({np.bincount(pt_pred).max()/len(pt_pred)*100:.1f}%)")
print(f"   MLX most common class: {np.bincount(mlx_pred).argmax()} ({np.bincount(mlx_pred).max()/len(mlx_pred)*100:.1f}%)")

# Error distribution
print("\n8️⃣ Error Distribution:")
errors = np.abs(pt_np - mlx_np)
print(f"   Mean absolute error: {errors.mean():.6f}")
print(f"   Median absolute error: {np.median(errors):.6f}")
print(f"   Max absolute error: {errors.max():.6f}")
print(f"   95th percentile error: {np.percentile(errors, 95):.6f}")
print(f"   99th percentile error: {np.percentile(errors, 99):.6f}")

# Where are the largest errors?
error_per_class = errors.mean(axis=0)
worst_classes = np.argsort(error_per_class)[-5:]
print(f"\n   Classes with highest errors:")
for c in reversed(worst_classes):
    print(f"     Class {c}: MAE = {error_per_class[c]:.6f}, Correlation = {correlations[c]:.4f}")

# Check numerical precision accumulation
print("\n9️⃣ Numerical Precision Analysis:")
# Compare magnitudes
pt_mag = np.abs(pt_np)
mlx_mag = np.abs(mlx_np)
print(f"   PT mean magnitude: {pt_mag.mean():.6f}")
print(f"   MLX mean magnitude: {mlx_mag.mean():.6f}")
print(f"   Magnitude difference: {abs(pt_mag.mean() - mlx_mag.mean()):.6f}")

# Relative error
relative_errors = np.abs((pt_np - mlx_np) / (np.abs(pt_np) + 1e-10))
print(f"   Mean relative error: {relative_errors.mean():.6f}")
print(f"   Median relative error: {np.median(relative_errors):.6f}")

# Check for systematic bias
print("\n🔟 Systematic Bias Check:")
bias = (mlx_np - pt_np).mean(axis=0)
print(f"   Mean bias per class: {bias.mean():.6f}")
print(f"   Bias std: {bias.std():.6f}")
print(f"   Classes with largest bias:")
largest_bias = np.argsort(np.abs(bias))[-5:]
for c in reversed(largest_bias):
    direction = "MLX higher" if bias[c] > 0 else "PT higher"
    print(f"     Class {c}: {bias[c]:+.6f} ({direction})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Summary and recommendations
print("\n📋 Summary and Recommendations:")
print("\n1. Current Correlation: {:.2f}%".format(np.mean(correlations) * 100))

if np.mean(correlations) < 0.95:
    print("\n2. Potential Issues Found:")
    
    if abs(pt_sum.mean() - 1.0) > 0.01 or abs(mlx_sum.mean() - 1.0) > 0.01:
        print("   ⚠️  Probability normalization may be off")
        print("      Check log_softmax implementation")
    
    if errors.mean() > 0.1:
        print("   ⚠️  High mean absolute error")
        print("      Check for missing normalizations or activations")
    
    if bias.std() > 0.5:
        print("   ⚠️  High variance in class-wise bias")
        print("      Check for systematic differences in layer implementations")
    
    if np.std(time_correlations) > 0.1:
        print("   ⚠️  Temporal correlation varies significantly")
        print("      Check LSTM hidden state initialization or reset")
    
    worst_corr = np.min(correlations)
    if worst_corr < 0.8:
        print(f"   ⚠️  Class {np.argmin(correlations)} has very low correlation ({worst_corr:.4f})")
        print("      This class may have different behavior between models")

print("\n3. Next Steps:")
if agreement < 0.70:
    print("   → Prediction agreement is low - check decision boundaries")
if errors.mean() > 0.05:
    print("   → High errors suggest architectural or preprocessing differences")
if bias.std() > 0.3:
    print("   → Systematic bias suggests missing normalization or scaling")
