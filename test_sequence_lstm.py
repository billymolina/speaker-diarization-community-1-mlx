"""
Compare LSTM processing of a sequence between PyTorch and MLX.
Tests multiple timesteps to see where divergence occurs.
"""
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

print("=" * 80)
print("SEQUENCE LSTM COMPARISON: PyTorch vs MLX")
print("=" * 80)

# Load the actual weights
weights = mx.load("models/segmentation_mlx/weights.npz")
print(f"\nLoaded {len(weights)} weight tensors")

# Get first layer forward LSTM weights
w_ih = weights['lstm.weight_ih_l0']  # (512, 60)
w_hh = weights['lstm.weight_hh_l0']  # (512, 128)
b_ih = weights['lstm.bias_ih_l0']    # (512,)
b_hh = weights['lstm.bias_hh_l0']    # (512,)

# Create test input (short sequence)
batch_size = 1
seq_len = 10  # Short sequence to see progression
input_size = 60
hidden_size = 128

# Create identical input for both frameworks
np.random.seed(42)
x_np = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

print(f"\nInput shape: ({batch_size}, {seq_len}, {input_size})")

# ============================================================================
# PYTORCH LSTM
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH LSTM")
print("=" * 80)

# Create PyTorch LSTM
lstm_pt = nn.LSTM(input_size, hidden_size, batch_first=True)

# Load weights
with torch.no_grad():
    lstm_pt.weight_ih_l0.copy_(torch.from_numpy(np.array(w_ih)))
    lstm_pt.weight_hh_l0.copy_(torch.from_numpy(np.array(w_hh)))
    lstm_pt.bias_ih_l0.copy_(torch.from_numpy(np.array(b_ih)))
    lstm_pt.bias_hh_l0.copy_(torch.from_numpy(np.array(b_hh)))

# Run PyTorch LSTM
x_pt = torch.from_numpy(x_np)
with torch.no_grad():
    h_pt, (h_final_pt, c_final_pt) = lstm_pt(x_pt)

print(f"\nPyTorch output shape: {h_pt.shape}")
print(f"\nTimestep-by-timestep ranges:")
for t in range(seq_len):
    h_t = h_pt[0, t, :]
    print(f"  t={t}: [{h_t.min():.6f}, {h_t.max():.6f}], mean={h_t.mean():.6f}, std={h_t.std():.6f}")

print(f"\nFinal hidden state: [{h_final_pt.min():.6f}, {h_final_pt.max():.6f}]")
print(f"Final cell state: [{c_final_pt.min():.6f}, {c_final_pt.max():.6f}]")

# ============================================================================
# MLX LSTM
# ============================================================================
print("\n" + "=" * 80)
print("MLX LSTM")
print("=" * 80)

# Create MLX LSTM
lstm_mlx = mlx_nn.LSTM(input_size, hidden_size)

# Load weights
lstm_mlx.Wx = w_ih
lstm_mlx.Wh = w_hh
lstm_mlx.bias = b_ih + b_hh

# Run MLX LSTM
x_mlx = mx.array(x_np)
h_mlx, c_mlx = lstm_mlx(x_mlx)

print(f"\nMLX output shape: {h_mlx.shape}")
print(f"\nTimestep-by-timestep ranges:")
for t in range(seq_len):
    h_t = h_mlx[0, t, :]
    print(f"  t={t}: [{h_t.min():.6f}, {h_t.max():.6f}], mean={h_t.mean():.6f}, std={h_t.std():.6f}")

# Get final states
h_final_mlx = h_mlx[:, -1, :]
c_final_mlx = c_mlx[:, -1, :]
print(f"\nFinal hidden state: [{h_final_mlx.min():.6f}, {h_final_mlx.max():.6f}]")
print(f"Final cell state: [{c_final_mlx.min():.6f}, {c_final_mlx.max():.6f}]")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

h_pt_np = h_pt.numpy()
h_mlx_np = np.array(h_mlx)

print(f"\nTimestep-by-timestep differences:")
for t in range(seq_len):
    diff = np.abs(h_pt_np[0, t, :] - h_mlx_np[0, t, :])
    corr = np.corrcoef(h_pt_np[0, t, :], h_mlx_np[0, t, :])[0, 1]
    print(f"  t={t}: max_diff={diff.max():.10f}, mean_diff={diff.mean():.10f}, corr={corr:.6f}")

# Overall comparison
diff_all = np.abs(h_pt_np - h_mlx_np)
print(f"\nOverall difference:")
print(f"  Max absolute difference: {diff_all.max():.10f}")
print(f"  Mean absolute difference: {diff_all.mean():.10f}")
print(f"  Correlation: {np.corrcoef(h_pt_np.flatten(), h_mlx_np.flatten())[0, 1]:.6f}")

if diff_all.max() < 1e-5:
    print("\n✓ OUTPUTS MATCH!")
else:
    print(f"\n✗ OUTPUTS DIFFER! (max difference: {diff_all.max():.10f})")
