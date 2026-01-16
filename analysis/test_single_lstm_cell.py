"""
Compare a single LSTM cell operation between PyTorch and MLX with actual loaded weights.
This will help identify the exact source of divergence.
"""
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

print("=" * 80)
print("SINGLE LSTM CELL COMPARISON: PyTorch vs MLX")
print("=" * 80)

# Load the actual weights
weights = mx.load("models/segmentation_mlx/weights.npz")
print(f"\nLoaded {len(weights)} weight tensors")

# Get first layer forward LSTM weights
w_ih = weights['lstm.weight_ih_l0']  # (512, 60)
w_hh = weights['lstm.weight_hh_l0']  # (512, 128)
b_ih = weights['lstm.bias_ih_l0']    # (512,)
b_hh = weights['lstm.bias_hh_l0']    # (512,)

print(f"\nWeight shapes:")
print(f"  w_ih: {w_ih.shape}")
print(f"  w_hh: {w_hh.shape}")
print(f"  b_ih: {b_ih.shape}")
print(f"  b_hh: {b_hh.shape}")

# Create test input (single timestep)
batch_size = 1
input_size = 60
hidden_size = 128

# Create identical input for both frameworks
np.random.seed(42)
x_np = np.random.randn(batch_size, input_size).astype(np.float32)
h_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
c_np = np.random.randn(batch_size, hidden_size).astype(np.float32)

print(f"\nInput shapes:")
print(f"  x: ({batch_size}, {input_size})")
print(f"  h_prev: ({batch_size}, {hidden_size})")
print(f"  c_prev: ({batch_size}, {hidden_size})")

# ============================================================================
# PYTORCH LSTM CELL
# ============================================================================
print("\n" + "=" * 80)
print("PYTORCH LSTM CELL")
print("=" * 80)

# Create PyTorch tensors
x_pt = torch.from_numpy(x_np)
h_pt = torch.from_numpy(h_np)
c_pt = torch.from_numpy(c_np)

# Create PyTorch LSTM cell
lstm_cell_pt = nn.LSTMCell(input_size, hidden_size)

# Load weights into PyTorch LSTM cell
with torch.no_grad():
    lstm_cell_pt.weight_ih.copy_(torch.from_numpy(np.array(w_ih)))
    lstm_cell_pt.weight_hh.copy_(torch.from_numpy(np.array(w_hh)))
    lstm_cell_pt.bias_ih.copy_(torch.from_numpy(np.array(b_ih)))
    lstm_cell_pt.bias_hh.copy_(torch.from_numpy(np.array(b_hh)))

# Run PyTorch LSTM cell
with torch.no_grad():
    h_next_pt, c_next_pt = lstm_cell_pt(x_pt, (h_pt, c_pt))

print(f"\nPyTorch output:")
print(f"  h_next range: [{h_next_pt.min():.6f}, {h_next_pt.max():.6f}]")
print(f"  h_next mean: {h_next_pt.mean():.6f}, std: {h_next_pt.std():.6f}")
print(f"  c_next range: [{c_next_pt.min():.6f}, {c_next_pt.max():.6f}]")
print(f"  c_next mean: {c_next_pt.mean():.6f}, std: {c_next_pt.std():.6f}")

# ============================================================================
# MLX LSTM CELL (Manual Implementation)
# ============================================================================
print("\n" + "=" * 80)
print("MLX LSTM CELL (Manual Implementation)")
print("=" * 80)

# Create MLX tensors
x_mlx = mx.array(x_np)
h_mlx = mx.array(h_np)
c_mlx = mx.array(c_np)

# Manual LSTM cell computation in MLX
# Formula: ifgo = W_ih @ x + b_ih + W_hh @ h + b_hh
# Then: i, f, g, o = split(sigmoid/tanh(ifgo))

print("\nStep 1: Compute W_ih @ x + b_ih")
# Note: MLX LSTM uses .T in forward pass, so we need to transpose
Wx_T = mx.transpose(w_ih, (1, 0))  # (60, 512)
print(f"  Wx_T shape: {Wx_T.shape}")
x_contribution = x_mlx @ Wx_T  # (1, 60) @ (60, 512) = (1, 512)
print(f"  x_contribution shape: {x_contribution.shape}")
x_contribution = x_contribution + b_ih
print(f"  x_contribution range: [{x_contribution.min():.6f}, {x_contribution.max():.6f}]")

print("\nStep 2: Compute W_hh @ h + b_hh")
Wh_T = mx.transpose(w_hh, (1, 0))  # (128, 512)
print(f"  Wh_T shape: {Wh_T.shape}")
h_contribution = h_mlx @ Wh_T  # (1, 128) @ (128, 512) = (1, 512)
print(f"  h_contribution shape: {h_contribution.shape}")
h_contribution = h_contribution + b_hh
print(f"  h_contribution range: [{h_contribution.min():.6f}, {h_contribution.max():.6f}]")

print("\nStep 3: Combine contributions")
ifgo = x_contribution + h_contribution
print(f"  ifgo shape: {ifgo.shape}")
print(f"  ifgo range: [{ifgo.min():.6f}, {ifgo.max():.6f}]")

print("\nStep 4: Split and apply activations")
i, f, g, o = mx.split(ifgo, 4, axis=-1)
print(f"  i shape: {i.shape}")

i = mx.sigmoid(i)
f = mx.sigmoid(f)
g = mx.tanh(g)
o = mx.sigmoid(o)

print(f"  After activations:")
print(f"    i range: [{i.min():.6f}, {i.max():.6f}]")
print(f"    f range: [{f.min():.6f}, {f.max():.6f}]")
print(f"    g range: [{g.min():.6f}, {g.max():.6f}]")
print(f"    o range: [{o.min():.6f}, {o.max():.6f}]")

print("\nStep 5: Compute cell and hidden states")
c_next_mlx = f * c_mlx + i * g
h_next_mlx = o * mx.tanh(c_next_mlx)

print(f"\nMLX output:")
print(f"  h_next range: [{h_next_mlx.min():.6f}, {h_next_mlx.max():.6f}]")
print(f"  h_next mean: {h_next_mlx.mean():.6f}, std: {h_next_mlx.std():.6f}")
print(f"  c_next range: [{c_next_mlx.min():.6f}, {c_next_mlx.max():.6f}]")
print(f"  c_next mean: {c_next_mlx.mean():.6f}, std: {c_next_mlx.std():.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

h_next_mlx_np = np.array(h_next_mlx)
h_next_pt_np = h_next_pt.numpy()

h_diff = np.abs(h_next_mlx_np - h_next_pt_np)
c_diff = np.abs(np.array(c_next_mlx) - c_next_pt.numpy())

print(f"\nHidden state difference:")
print(f"  Max absolute difference: {h_diff.max():.10f}")
print(f"  Mean absolute difference: {h_diff.mean():.10f}")
print(f"  Correlation: {np.corrcoef(h_next_mlx_np.flatten(), h_next_pt_np.flatten())[0, 1]:.6f}")

print(f"\nCell state difference:")
print(f"  Max absolute difference: {c_diff.max():.10f}")
print(f"  Mean absolute difference: {c_diff.mean():.10f}")

if h_diff.max() < 1e-5:
    print("\n✓ OUTPUTS MATCH! (difference < 1e-5)")
else:
    print(f"\n✗ OUTPUTS DIFFER! (max difference: {h_diff.max():.10f})")
    print("\nDifference matrix (first 10 elements):")
    print(h_diff[0, :10])
    print("\nPyTorch h_next (first 10 elements):")
    print(h_next_pt_np[0, :10])
    print("\nMLX h_next (first 10 elements):")
    print(h_next_mlx_np[0, :10])

# ============================================================================
# TEST WITH MLX LSTM LAYER (nn.LSTM)
# ============================================================================
print("\n" + "=" * 80)
print("MLX nn.LSTM LAYER (Built-in)")
print("=" * 80)

# Create MLX LSTM
lstm_mlx = mlx_nn.LSTM(input_size, hidden_size)

# Load weights - NO TRANSPOSE since MLX applies .T internally
lstm_mlx.Wx = w_ih
lstm_mlx.Wh = w_hh
lstm_mlx.bias = b_ih + b_hh

# Prepare input (needs sequence dimension)
x_seq_mlx = mx.expand_dims(x_mlx, axis=1)  # (1, 1, 60)

# Run MLX LSTM
h_out_mlx, c_out_mlx = lstm_mlx(x_seq_mlx, hidden=h_mlx, cell=c_mlx)

# Extract single timestep output
h_next_mlx_builtin = h_out_mlx[:, 0, :]  # (1, 128)
c_next_mlx_builtin = c_out_mlx[:, 0, :]  # (1, 128)

print(f"\nMLX nn.LSTM output:")
print(f"  h_next range: [{h_next_mlx_builtin.min():.6f}, {h_next_mlx_builtin.max():.6f}]")
print(f"  h_next mean: {h_next_mlx_builtin.mean():.6f}, std: {h_next_mlx_builtin.std():.6f}")

# Compare with manual implementation
h_builtin_np = np.array(h_next_mlx_builtin)
h_diff_builtin = np.abs(h_builtin_np - h_next_mlx_np)

print(f"\nMLX nn.LSTM vs Manual implementation:")
print(f"  Max difference: {h_diff_builtin.max():.10f}")
print(f"  Mean difference: {h_diff_builtin.mean():.10f}")

# Compare with PyTorch
h_diff_builtin_pt = np.abs(h_builtin_np - h_next_pt_np)
print(f"\nMLX nn.LSTM vs PyTorch:")
print(f"  Max difference: {h_diff_builtin_pt.max():.10f}")
print(f"  Mean difference: {h_diff_builtin_pt.mean():.10f}")
print(f"  Correlation: {np.corrcoef(h_builtin_np.flatten(), h_next_pt_np.flatten())[0, 1]:.6f}")

if h_diff_builtin_pt.max() < 1e-5:
    print("\n✓ MLX nn.LSTM matches PyTorch!")
else:
    print(f"\n✗ MLX nn.LSTM differs from PyTorch (max diff: {h_diff_builtin_pt.max():.10f})")
