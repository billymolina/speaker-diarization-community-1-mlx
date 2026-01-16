"""
Debug version of compare_model_outputs.py with layer inspection
"""
import torch
import mlx.core as mx
import torchaudio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models import load_pyannote_model
from pyannote.audio import Model

print("=" * 80)
print("DEBUG COMPARISON")
print("=" * 80)

# Load models
print("\n[1] Loading models...")
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
pytorch_model = Model.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()
print("✓ Both models loaded")

# Load audio
print("\n[2] Loading test audio...")
audio_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
waveform, sr = torchaudio.load(audio_path)

# Take first 10 seconds
samples = 160000
waveform = waveform[:, :samples]

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

print(f"✓ Audio loaded: {waveform.shape}")

# PYTORCH
print("\n[3] PyTorch processing...")
pytorch_input = waveform.unsqueeze(0)
print(f"  Input shape: {pytorch_input.shape}")

with torch.no_grad():
    # SincNet
    sincnet_pt = pytorch_model.sincnet(pytorch_input)
    print(f"  After SincNet: {sincnet_pt.shape}, range [{sincnet_pt.min():.6f}, {sincnet_pt.max():.6f}]")
    
    # Transpose for LSTM
    sincnet_pt_t = sincnet_pt.transpose(1, 2)
    print(f"  After transpose: {sincnet_pt_t.shape}")
    
    # LSTM
    lstm_pt, _ = pytorch_model.lstm(sincnet_pt_t)
    print(f"  After LSTM: {lstm_pt.shape}, range [{lstm_pt.min():.6f}, {lstm_pt.max():.6f}]")
    
    # Full model
    pytorch_logits = pytorch_model(pytorch_input)
    print(f"  Final logits: {pytorch_logits.shape}, range [{pytorch_logits.min():.6f}, {pytorch_logits.max():.6f}]")

# MLX
print("\n[4] MLX processing...")
mlx_input = mx.array(waveform.numpy(), dtype=mx.float32)
print(f"  Input shape: {mlx_input.shape}")

# Get SincNet output
sincnet_mlx = mlx_model.sincnet(mlx_input)
print(f"  After SincNet: {sincnet_mlx.shape}, range [{sincnet_mlx.min():.6f}, {sincnet_mlx.max():.6f}]")

# Compare SincNet outputs
sincnet_diff = np.abs(np.array(sincnet_mlx) - sincnet_pt_t.numpy())
print(f"  SincNet diff: max={sincnet_diff.max():.10f}, mean={sincnet_diff.mean():.10f}")

# Get LSTM output (manually)
h_mlx = sincnet_mlx
for i, (lstm_fwd, lstm_bwd) in enumerate(zip(mlx_model.lstm_forward, mlx_model.lstm_backward)):
    h_fwd, _ = lstm_fwd(h_mlx)
    h_bwd, _ = lstm_bwd(h_mlx[:, ::-1, :])
    h_bwd = h_bwd[:, ::-1, :]
    h_mlx = mx.concatenate([h_fwd, h_bwd], axis=-1)
print(f"  After LSTM: {h_mlx.shape}, range [{h_mlx.min():.6f}, {h_mlx.max():.6f}]")

# Compare LSTM outputs
lstm_diff = np.abs(np.array(h_mlx) - lstm_pt.numpy())
print(f"  LSTM diff: max={lstm_diff.max():.10f}, mean={lstm_diff.mean():.10f}")

# Linear layers
with torch.no_grad():
    linear1_pt = pytorch_model.linear[0](lstm_pt)
    relu1_pt = torch.nn.functional.relu(linear1_pt)
    linear2_pt = pytorch_model.linear[1](relu1_pt)
    relu2_pt = torch.nn.functional.relu(linear2_pt)
    classifier_pt = pytorch_model.classifier(relu2_pt)
    print(f"  After Linear1+ReLU: {relu1_pt.shape}, range [{relu1_pt.min():.6f}, {relu1_pt.max():.6f}]")
    print(f"  After Linear2+ReLU: {relu2_pt.shape}, range [{relu2_pt.min():.6f}, {relu2_pt.max():.6f}]")
    print(f"  After Classifier: {classifier_pt.shape}, range [{classifier_pt.min():.6f}, {classifier_pt.max():.6f}]")

# MLX linear layers
linear1_mlx = mlx_model.linear1(h_mlx)
relu1_mlx = mx.maximum(linear1_mlx, 0)
linear2_mlx = mlx_model.linear2(relu1_mlx)
relu2_mlx = mx.maximum(linear2_mlx, 0)
classifier_mlx = mlx_model.classifier(relu2_mlx)
print(f"  After Linear1+ReLU: {relu1_mlx.shape}, range [{relu1_mlx.min():.6f}, {relu1_mlx.max():.6f}]")
print(f"  After Linear2+ReLU: {relu2_mlx.shape}, range [{relu2_mlx.min():.6f}, {relu2_mlx.max():.6f}]")
print(f"  After Classifier: {classifier_mlx.shape}, range [{classifier_mlx.min():.6f}, {classifier_mlx.max():.6f}]")

# Compare each layer
linear1_diff = np.abs(np.array(relu1_mlx) - relu1_pt.numpy())
print(f"  Linear1+ReLU diff: max={linear1_diff.max():.10f}, mean={linear1_diff.mean():.10f}")

linear2_diff = np.abs(np.array(relu2_mlx) - relu2_pt.numpy())
print(f"  Linear2+ReLU diff: max={linear2_diff.max():.10f}, mean={linear2_diff.mean():.10f}")

classifier_diff = np.abs(np.array(classifier_mlx) - classifier_pt.numpy())
print(f"  Classifier diff: max={classifier_diff.max():.10f}, mean={classifier_diff.mean():.10f}")

# Full model
mlx_logits = mlx_model(mlx_input)
print(f"  Full model logits: {mlx_logits.shape}, range [{mlx_logits.min():.6f}, {mlx_logits.max():.6f}]")

# Compare final outputs
print("\n[5] Final comparison...")
pytorch_np = pytorch_logits[0].numpy()
mlx_np = np.array(mlx_logits[0])

diff = np.abs(pytorch_np - mlx_np)
print(f"  Max difference: {diff.max():.10f}")
print(f"  Mean difference: {diff.mean():.10f}")
print(f"  Correlation: {np.corrcoef(pytorch_np.flatten(), mlx_np.flatten())[0, 1]:.6f}")

if diff.max() < 1e-4:
    print("\n✓ MODELS MATCH!")
else:
    print(f"\n✗ MODELS DIFFER!")
    print(f"\nPyTorch first frame logits:\n{pytorch_np[0]}")
    print(f"\nMLX first frame logits:\n{mlx_np[0]}")
