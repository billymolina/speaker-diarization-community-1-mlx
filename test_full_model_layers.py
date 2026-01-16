"""
Test layer-by-layer outputs from SincNet -> LSTM -> Linear -> Classifier
"""
import torch
import mlx.core as mx
import numpy as np

print("=" * 80)
print("FULL MODEL LAYER-BY-LAYER COMPARISON")
print("=" * 80)

# Load weights
weights = mx.load("models/segmentation_mlx/weights.npz")

# Load PyTorch model
from pyannote.audio import Model as PyanoteModel
pytorch_model = PyanoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Create test input
np.random.seed(42)
audio_np = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds

print(f"Input audio shape: {audio_np.shape}")

# ============================================================================
# PYTORCH: Full Forward Pass
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH FORWARD PASS")
print("=" * 80)

with torch.no_grad():
    audio_pt = torch.from_numpy(audio_np)[None, :]
    
    # SincNet
    sincnet_out = pytorch_model.sincnet(audio_pt).transpose(1, 2)  # (1, T, 60)
    print(f"1. SincNet output: {sincnet_out.shape}")
    print(f"   Range: [{sincnet_out.min():.6f}, {sincnet_out.max():.6f}]")
    
    # LSTM
    lstm_out, _ = pytorch_model.lstm(sincnet_out)
    print(f"2. LSTM output: {lstm_out.shape}")
    print(f"   Range: [{lstm_out.min():.6f}, {lstm_out.max():.6f}]")
    
    # Linear 1 + ReLU
    linear1_out = pytorch_model.linear[0](lstm_out)
    relu1_out = torch.nn.functional.relu(linear1_out)
    print(f"3. Linear1 output: {linear1_out.shape}")
    print(f"   Range before ReLU: [{linear1_out.min():.6f}, {linear1_out.max():.6f}]")
    print(f"   Range after ReLU: [{relu1_out.min():.6f}, {relu1_out.max():.6f}]")
    
    # Linear 2 + ReLU
    linear2_out = pytorch_model.linear[1](relu1_out)
    relu2_out = torch.nn.functional.relu(linear2_out)
    print(f"4. Linear2 output: {linear2_out.shape}")
    print(f"   Range before ReLU: [{linear2_out.min():.6f}, {linear2_out.max():.6f}]")
    print(f"   Range after ReLU: [{relu2_out.min():.6f}, {relu2_out.max():.6f}]")
    
    # Classifier
    logits_pt = pytorch_model.classifier(relu2_out)
    print(f"5. Classifier output: {logits_pt.shape}")
    print(f"   Range: [{logits_pt.min():.6f}, {logits_pt.max():.6f}]")

# ============================================================================
# MLX: Full Forward Pass
# ============================================================================
print("\n" + "=" * 80)
print("MLX FORWARD PASS")
print("=" * 80)

# Use the same SincNet output
sincnet_out_mlx = mx.array(sincnet_out.numpy())
print(f"1. SincNet output (from PyTorch): {sincnet_out_mlx.shape}")
print(f"   Range: [{sincnet_out_mlx.min():.6f}, {sincnet_out_mlx.max():.6f}]")

# LSTM (4-layer bidirectional manual implementation)
h_mlx = sincnet_out_mlx
for layer in range(4):
    input_size = 60 if layer == 0 else 256
    
    # Create LSTMs
    import mlx.nn as mlx_nn
    lstm_fwd = mlx_nn.LSTM(input_size, 128)
    lstm_bwd = mlx_nn.LSTM(input_size, 128)
    
    # Load weights
    lstm_fwd.Wx = weights[f'lstm.weight_ih_l{layer}']
    lstm_fwd.Wh = weights[f'lstm.weight_hh_l{layer}']
    lstm_fwd.bias = weights[f'lstm.bias_ih_l{layer}'] + weights[f'lstm.bias_hh_l{layer}']
    
    lstm_bwd.Wx = weights[f'lstm.weight_ih_l{layer}_reverse']
    lstm_bwd.Wh = weights[f'lstm.weight_hh_l{layer}_reverse']
    lstm_bwd.bias = weights[f'lstm.bias_ih_l{layer}_reverse'] + weights[f'lstm.bias_hh_l{layer}_reverse']
    
    # Forward and backward
    h_fwd, _ = lstm_fwd(h_mlx)
    h_bwd, _ = lstm_bwd(h_mlx[:, ::-1, :])
    h_bwd = h_bwd[:, ::-1, :]
    h_mlx = mx.concatenate([h_fwd, h_bwd], axis=-1)

print(f"2. LSTM output: {h_mlx.shape}")
print(f"   Range: [{h_mlx.min():.6f}, {h_mlx.max():.6f}]")

# Compare LSTM outputs
diff_lstm = np.abs(np.array(h_mlx) - lstm_out.numpy())
print(f"   LSTM diff: max={diff_lstm.max():.10f}, mean={diff_lstm.mean():.10f}")

# Linear 1 + ReLU
import mlx.nn as mlx_nn
linear1_mlx = mlx_nn.Linear(256, 128)
linear1_mlx.weight = weights['linear.0.weight']
linear1_mlx.bias = weights['linear.0.bias']

linear1_out_mlx = linear1_mlx(h_mlx)
relu1_out_mlx = mx.maximum(linear1_out_mlx, 0)
print(f"3. Linear1 output: {linear1_out_mlx.shape}")
print(f"   Range before ReLU: [{linear1_out_mlx.min():.6f}, {linear1_out_mlx.max():.6f}]")
print(f"   Range after ReLU: [{relu1_out_mlx.min():.6f}, {relu1_out_mlx.max():.6f}]")

# Compare Linear1 outputs
diff_linear1 = np.abs(np.array(linear1_out_mlx) - linear1_out.numpy())
print(f"   Linear1 diff (before ReLU): max={diff_linear1.max():.10f}, mean={diff_linear1.mean():.10f}")

diff_relu1 = np.abs(np.array(relu1_out_mlx) - relu1_out.numpy())
print(f"   ReLU1 diff: max={diff_relu1.max():.10f}, mean={diff_relu1.mean():.10f}")

# Linear 2 + ReLU
linear2_mlx = mlx_nn.Linear(128, 128)
linear2_mlx.weight = weights['linear.1.weight']
linear2_mlx.bias = weights['linear.1.bias']

linear2_out_mlx = linear2_mlx(relu1_out_mlx)
relu2_out_mlx = mx.maximum(linear2_out_mlx, 0)
print(f"4. Linear2 output: {linear2_out_mlx.shape}")
print(f"   Range before ReLU: [{linear2_out_mlx.min():.6f}, {linear2_out_mlx.max():.6f}]")
print(f"   Range after ReLU: [{relu2_out_mlx.min():.6f}, {relu2_out_mlx.max():.6f}]")

# Compare Linear2 outputs
diff_linear2 = np.abs(np.array(linear2_out_mlx) - linear2_out.numpy())
print(f"   Linear2 diff (before ReLU): max={diff_linear2.max():.10f}, mean={diff_linear2.mean():.10f}")

diff_relu2 = np.abs(np.array(relu2_out_mlx) - relu2_out.numpy())
print(f"   ReLU2 diff: max={diff_relu2.max():.10f}, mean={diff_relu2.mean():.10f}")

# Classifier
classifier_mlx = mlx_nn.Linear(128, 7)
classifier_mlx.weight = weights['classifier.weight']
classifier_mlx.bias = weights['classifier.bias']

logits_mlx = classifier_mlx(relu2_out_mlx)
print(f"5. Classifier output: {logits_mlx.shape}")
print(f"   Range: [{logits_mlx.min():.6f}, {logits_mlx.max():.6f}]")

# Compare final outputs
diff_logits = np.abs(np.array(logits_mlx) - logits_pt.numpy())
print(f"   Classifier diff: max={diff_logits.max():.10f}, mean={diff_logits.mean():.10f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nLayer-by-layer differences:")
print(f"  LSTM: max={diff_lstm.max():.10f}")
print(f"  Linear1 (before ReLU): max={diff_linear1.max():.10f}")
print(f"  ReLU1: max={diff_relu1.max():.10f}")
print(f"  Linear2 (before ReLU): max={diff_linear2.max():.10f}")
print(f"  ReLU2: max={diff_relu2.max():.10f}")
print(f"  Classifier: max={diff_logits.max():.10f}")

print(f"\nFinal output:")
print(f"  PyTorch range: [{logits_pt.min():.6f}, {logits_pt.max():.6f}]")
print(f"  MLX range: [{logits_mlx.min():.6f}, {logits_mlx.max():.6f}]")

if diff_logits.max() < 1e-5:
    print("\n✓ MODELS MATCH!")
else:
    print(f"\n✗ MODELS DIFFER! (max difference: {diff_logits.max():.10f})")
