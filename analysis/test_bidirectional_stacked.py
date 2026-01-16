"""
Test the bidirectional stacked LSTM exactly as in the model.
"""
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

print("=" * 80)
print("BIDIRECTIONAL STACKED LSTM: PyTorch vs MLX")
print("=" * 80)

# Load weights
weights = mx.load("models/segmentation_mlx/weights.npz")

# Load PyTorch model for reference
from pyannote.audio import Model as PyanoteModel
pytorch_model = PyanoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Create test input
np.random.seed(42)
audio_np = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds

# Get SincNet output from PyTorch
with torch.no_grad():
    audio_pt = torch.from_numpy(audio_np)[None, :]
    sincnet_out_pt = pytorch_model.sincnet(audio_pt).transpose(1, 2)  # (1, T, 60)
    print(f"SincNet output shape: {sincnet_out_pt.shape}")
    
# ============================================================================
# PYTORCH: 4-layer Bidirectional LSTM
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH: 4-layer Bidirectional LSTM")
print("=" * 80)

lstm_pt = nn.LSTM(60, 128, num_layers=4, bidirectional=True, batch_first=True)

# Load all weights
for layer in range(4):
    w_ih = weights[f'lstm.weight_ih_l{layer}']
    w_hh = weights[f'lstm.weight_hh_l{layer}']
    b_ih = weights[f'lstm.bias_ih_l{layer}']
    b_hh = weights[f'lstm.bias_hh_l{layer}']
    
    w_ih_rev = weights[f'lstm.weight_ih_l{layer}_reverse']
    w_hh_rev = weights[f'lstm.weight_hh_l{layer}_reverse']
    b_ih_rev = weights[f'lstm.bias_ih_l{layer}_reverse']
    b_hh_rev = weights[f'lstm.bias_hh_l{layer}_reverse']
    
    with torch.no_grad():
        getattr(lstm_pt, f'weight_ih_l{layer}').copy_(torch.from_numpy(np.array(w_ih)))
        getattr(lstm_pt, f'weight_hh_l{layer}').copy_(torch.from_numpy(np.array(w_hh)))
        getattr(lstm_pt, f'bias_ih_l{layer}').copy_(torch.from_numpy(np.array(b_ih)))
        getattr(lstm_pt, f'bias_hh_l{layer}').copy_(torch.from_numpy(np.array(b_hh)))
        
        getattr(lstm_pt, f'weight_ih_l{layer}_reverse').copy_(torch.from_numpy(np.array(w_ih_rev)))
        getattr(lstm_pt, f'weight_hh_l{layer}_reverse').copy_(torch.from_numpy(np.array(w_hh_rev)))
        getattr(lstm_pt, f'bias_ih_l{layer}_reverse').copy_(torch.from_numpy(np.array(b_ih_rev)))
        getattr(lstm_pt, f'bias_hh_l{layer}_reverse').copy_(torch.from_numpy(np.array(b_hh_rev)))

with torch.no_grad():
    h_pt, _ = lstm_pt(sincnet_out_pt)

print(f"PyTorch output shape: {h_pt.shape}")
print(f"PyTorch output range: [{h_pt.min():.6f}, {h_pt.max():.6f}]")
print(f"PyTorch output mean: {h_pt.mean():.6f}, std: {h_pt.std():.6f}")

# ============================================================================
# MLX: Manual Bidirectional 4-layer LSTM
# ============================================================================
print("\n" + "=" * 80)
print("MLX: Manual Bidirectional 4-layer LSTM")
print("=" * 80)

# Convert SincNet output to MLX
h_mlx = mx.array(sincnet_out_pt.numpy())
print(f"Initial input shape: {h_mlx.shape}")

for layer in range(4):
    print(f"\n--- Layer {layer} ---")
    print(f"  Input shape: {h_mlx.shape}")
    
    # Create forward and backward LSTMs
    input_size = 60 if layer == 0 else 256
    lstm_fwd = mlx_nn.LSTM(input_size, 128)
    lstm_bwd = mlx_nn.LSTM(input_size, 128)
    
    # Load weights
    lstm_fwd.Wx = weights[f'lstm.weight_ih_l{layer}']
    lstm_fwd.Wh = weights[f'lstm.weight_hh_l{layer}']
    lstm_fwd.bias = weights[f'lstm.bias_ih_l{layer}'] + weights[f'lstm.bias_hh_l{layer}']
    
    lstm_bwd.Wx = weights[f'lstm.weight_ih_l{layer}_reverse']
    lstm_bwd.Wh = weights[f'lstm.weight_hh_l{layer}_reverse']
    lstm_bwd.bias = weights[f'lstm.bias_ih_l{layer}_reverse'] + weights[f'lstm.bias_hh_l{layer}_reverse']
    
    # Forward pass
    h_fwd, c_fwd = lstm_fwd(h_mlx)
    print(f"  Forward output: [{h_fwd.min():.6f}, {h_fwd.max():.6f}]")
    print(f"  Forward cell: [{c_fwd.min():.6f}, {c_fwd.max():.6f}]")
    
    # Backward pass (reverse time dimension)
    h_rev = h_mlx[:, ::-1, :]
    h_bwd, c_bwd = lstm_bwd(h_rev)
    h_bwd = h_bwd[:, ::-1, :]  # Reverse back
    print(f"  Backward output: [{h_bwd.min():.6f}, {h_bwd.max():.6f}]")
    print(f"  Backward cell: [{c_bwd.min():.6f}, {c_bwd.max():.6f}]")
    
    # Concatenate
    h_mlx = mx.concatenate([h_fwd, h_bwd], axis=-1)
    print(f"  Concatenated output: [{h_mlx.min():.6f}, {h_mlx.max():.6f}], shape: {h_mlx.shape}")

print(f"\nFinal MLX output shape: {h_mlx.shape}")
print(f"Final MLX output range: [{h_mlx.min():.6f}, {h_mlx.max():.6f}]")
print(f"Final MLX output mean: {h_mlx.mean():.6f}, std: {h_mlx.std():.6f}")

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

if diff.max() < 1e-5:
    print("\n✓ OUTPUTS MATCH!")
else:
    print(f"\n✗ OUTPUTS DIFFER! (max difference: {diff.max():.10f})")
