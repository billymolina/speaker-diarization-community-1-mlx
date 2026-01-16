"""
Verify that pre-adding biases in MLX gives equivalent results to PyTorch's internal addition
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np

print("=" * 80)
print("LSTM BIAS EQUIVALENCE TEST")
print("=" * 80)

# Create identical test scenario
input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2

# Create random input
np.random.seed(42)
input_np = np.random.randn(batch_size, sequence_length, input_size).astype(np.float32)

# Create PyTorch LSTM
print("\n1️⃣ PyTorch LSTM (2 biases):")
pt_lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
pt_lstm.eval()

# Get its biases
bias_ih = pt_lstm.bias_ih_l0.detach().clone()
bias_hh = pt_lstm.bias_hh_l0.detach().clone()
weight_ih = pt_lstm.weight_ih_l0.detach().clone()
weight_hh = pt_lstm.weight_hh_l0.detach().clone()

print(f"   bias_ih mean: {bias_ih.mean():.6f}")
print(f"   bias_hh mean: {bias_hh.mean():.6f}")
print(f"   bias sum mean: {(bias_ih + bias_hh).mean():.6f}")

# Run PyTorch
input_pt = torch.from_numpy(input_np)
with torch.no_grad():
    output_pt, _ = pt_lstm(input_pt)

print(f"   Output shape: {output_pt.shape}")
print(f"   Output mean: {output_pt.mean():.6f}")
print(f"   Output [0,0,:3]: {output_pt[0,0,:3]}")

# Create MLX LSTM with combined bias
print("\n2️⃣ MLX LSTM (1 combined bias):")
mlx_lstm = nn.LSTM(input_size, hidden_size)

# Load weights from PyTorch
mlx_lstm.Wx = mx.array(weight_ih.numpy())
mlx_lstm.Wh = mx.array(weight_hh.numpy())
mlx_lstm.bias = mx.array((bias_ih + bias_hh).numpy())  # ← Combined bias

# Run MLX  
input_mlx = mx.array(input_np)
output_mlx = mlx_lstm(input_mlx)

if isinstance(output_mlx, tuple):
    output_mlx = output_mlx[0]

print(f"   Output shape: {output_mlx.shape}")
print(f"   Output mean: {float(mx.mean(output_mlx)):.6f}")
print(f"   Output [0,0,:3]: {output_mlx[0,0,:3]}")

# Compare
print("\n3️⃣ Comparison:")
output_mlx_np = np.array(output_mlx.tolist())
output_pt_np = output_pt.numpy()

max_diff = np.abs(output_pt_np - output_mlx_np).max()
mean_diff = np.abs(output_pt_np - output_mlx_np).mean()
correlation = np.corrcoef(output_pt_np.flatten(), output_mlx_np.flatten())[0, 1]

print(f"   Max difference: {max_diff:.10f}")
print(f"   Mean difference: {mean_diff:.10f}")
print(f"   Correlation: {correlation:.10f}")

if max_diff < 1e-5:
    print("\n   ✅ BIAS HANDLING IS CORRECT!")
    print("   Pre-adding bias_ih + bias_hh in MLX is mathematically equivalent")
    print("   to PyTorch's internal addition")
else:
    print("\n   ❌ BIAS HANDLING MAY BE INCORRECT!")
    print("   Differences exceed expected numerical precision")

# Test with different bias configurations
print("\n4️⃣ Testing Edge Cases:")

# Case 1: Zero bias_hh
pt_lstm.bias_hh_l0.data.zero_()
with torch.no_grad():
    out_pt_zero_hh, _ = pt_lstm(input_pt)

mlx_lstm.bias = mx.array(pt_lstm.bias_ih_l0.numpy())  # Only ih since hh=0
out_mlx_zero_hh = mlx_lstm(input_mlx)
if isinstance(out_mlx_zero_hh, tuple):
    out_mlx_zero_hh = out_mlx_zero_hh[0]

diff_zero_hh = np.abs(out_pt_zero_hh.numpy() - np.array(out_mlx_zero_hh.tolist())).max()
print(f"   Zero bias_hh - max diff: {diff_zero_hh:.10f}")

# Case 2: Zero bias_ih  
pt_lstm.bias_ih_l0.data = bias_ih.clone()
pt_lstm.bias_hh_l0.data = bias_hh.clone()
pt_lstm.bias_ih_l0.data.zero_()
with torch.no_grad():
    out_pt_zero_ih, _ = pt_lstm(input_pt)

mlx_lstm.bias = mx.array(pt_lstm.bias_hh_l0.numpy())  # Only hh since ih=0
out_mlx_zero_ih = mlx_lstm(input_mlx)
if isinstance(out_mlx_zero_ih, tuple):
    out_mlx_zero_ih = out_mlx_zero_ih[0]

diff_zero_ih = np.abs(out_pt_zero_ih.numpy() - np.array(out_mlx_zero_ih.tolist())).max()
print(f"   Zero bias_ih - max diff: {diff_zero_ih:.10f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if max_diff < 1e-5 and diff_zero_hh < 1e-5 and diff_zero_ih < 1e-5:
    print("\n✅ LSTM bias handling is CORRECT!")
    print("   MLX's single bias = PyTorch's bias_ih + bias_hh")
    print("   ")
    print("   The model differences are NOT caused by bias handling.")
    print("   Must investigate other sources:")
    print("   - InstanceNorm parameters")
    print("   - Numerical precision in deep stacking")
    print("   - Hidden state initialization")
else:
    print("\n❌ LSTM bias handling needs investigation")
    print(f"   Max differences: main={max_diff}, zero_hh={diff_zero_hh}, zero_ih={diff_zero_ih}")
