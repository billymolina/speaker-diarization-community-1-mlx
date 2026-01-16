"""
Check if there's a difference between manual steps and full model call
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models import load_pyannote_model
from pyannote.audio import Model as PyannoteModel

print("=" * 80)
print("MANUAL VS FULL MODEL COMPARISON")
print("=" * 80)

# Load models
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
pt_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0")
pt_model.eval()

# Load audio
audio_path = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
waveform, sr = torchaudio.load(audio_path)
waveform = waveform[:, :16000]  # 1 second

pt_input = waveform.unsqueeze(0)
mlx_input = mx.array(waveform.numpy()).reshape(1, 16000)

print("\n1️⃣ PyTorch - Manual Steps:")
with torch.no_grad():
    sincnet = pt_model.sincnet(pt_input)
    sincnet_t = sincnet.transpose(1, 2)
    lstm_out, _ = pt_model.lstm(sincnet_t)
    lin1 = torch.relu(pt_model.linear[0](lstm_out))
    lin2 = torch.relu(pt_model.linear[1](lin1))
    logits_manual = pt_model.classifier(lin2)
    log_probs_manual = torch.log_softmax(logits_manual, dim=-1)
    
print(f"   Manual logits: {logits_manual[0,0,:]}")
print(f"   Manual log_probs: {log_probs_manual[0,0,:]}")
print(f"   Manual range: [{log_probs_manual.min():.3f}, {log_probs_manual.max():.3f}]")

print("\n2️⃣ PyTorch - Full Model:")
with torch.no_grad():
    output_full = pt_model(pt_input)
    
print(f"   Full model: {output_full[0,0,:]}")
print(f"   Full range: [{output_full.min():.3f}, {output_full.max():.3f}]")

print("\n3️⃣ PyTorch Comparison:")
diff_pt = (log_probs_manual - output_full).abs().max()
print(f"   Max difference manual vs full: {diff_pt:.10f}")
if diff_pt < 1e-5:
    print(f"   ✓ Manual and full model match")
else:
    print(f"   ⚠️  Manual and full model DIFFER!")

print("\n" + "-" * 80)

print("\n4️⃣ MLX - Manual Steps:")
sincnet = mlx_model.sincnet(mlx_input)
h = sincnet
for lstm_fwd, lstm_bwd in zip(mlx_model.lstm_forward, mlx_model.lstm_backward):
    h_fwd, _ = lstm_fwd(h)
    h_bwd, _ = lstm_bwd(h[:, ::-1, :])
    h_bwd = h_bwd[:, ::-1, :]
    h = mx.concatenate([h_fwd, h_bwd], axis=-1)
lin1 = mx.maximum(mlx_model.linear1(h), 0)
lin2 = mx.maximum(mlx_model.linear2(lin1), 0)
logits_manual = mlx_model.classifier(lin2)
log_probs_manual = nn.log_softmax(logits_manual, axis=-1)

print(f"   Manual logits: {logits_manual[0,0,:]}")
print(f"   Manual log_probs: {log_probs_manual[0,0,:]}")
print(f"   Manual range: [{float(mx.min(log_probs_manual)):.3f}, {float(mx.max(log_probs_manual)):.3f}]")

print("\n5️⃣ MLX - Full Model:")
output_full = mlx_model(mlx_input)

print(f"   Full model: {output_full[0,0,:]}")
print(f"   Full range: [{float(mx.min(output_full)):.3f}, {float(mx.max(output_full)):.3f}]")

print("\n6️⃣ MLX Comparison:")
diff_mlx = float(mx.max(mx.abs(log_probs_manual - output_full)))
print(f"   Max difference manual vs full: {diff_mlx:.10f}")
if diff_mlx < 1e-5:
    print(f"   ✓ Manual and full model match")
else:
    print(f"   ⚠️  Manual and full model DIFFER!")

print("\n" + "=" * 80)
print("CROSS-MODEL COMPARISON")
print("=" * 80)

print("\n7️⃣ Comparing first frame outputs (after log_softmax):")
pt_np = output_full[0,0,:].cpu().numpy()
mlx_np = np.array(output_full[0,0,:].tolist())

print(f"\n   {'Class':<8} {'PyTorch':<15} {'MLX':<15} {'Diff':<15}")
print(f"   " + "-" * 56)
for c in range(7):
    diff = abs(pt_np[c] - mlx_np[c])
    print(f"   {c:<8} {pt_np[c]:<15.6f} {mlx_np[c]:<15.6f} {diff:<15.6f}")

corr = np.corrcoef(pt_np, mlx_np)[0,1]
print(f"\n   Correlation (single frame): {corr:.6f}")

print("\n" + "=" * 80)
