"""
Detailed investigation of weight formats and transpositions
Check if weights are loaded in the correct format
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pyannote.audio import Model as PyannoteModel

print("=" * 80)
print("WEIGHT FORMAT INVESTIGATION")
print("=" * 80)

# Load PyTorch model
print("\n1️⃣ Loading PyTorch model...")
pt_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0")
pt_model.eval()

print("\n2️⃣ Checking Linear Layer Weight Formats:")
print("\n  PyTorch Linear Layer 0:")
pt_linear0_weight = pt_model.linear[0].weight
print(f"    Shape: {pt_linear0_weight.shape}")  # [out_features, in_features]
print(f"    PyTorch convention: [out_features, in_features]")
print(f"    First row (output neuron 0): {pt_linear0_weight[0, :5]}")
print(f"    First col (input feature 0): {pt_linear0_weight[:5, 0]}")

# Load MLX weights
weights = mx.load("models/segmentation_mlx/weights.npz")
mlx_linear0_weight = weights['linear.0.weight']
print(f"\n  MLX Loaded Weight:")
print(f"    Shape: {mlx_linear0_weight.shape}")
print(f"    First row: {mlx_linear0_weight[0, :5]}")
print(f"    First col: {mlx_linear0_weight[:5, 0]}")

# Check if they match
pt_np = pt_linear0_weight.detach().cpu().numpy()
mlx_np = np.array(mlx_linear0_weight.tolist())

print(f"\n  Direct comparison:")
print(f"    Max difference: {np.abs(pt_np - mlx_np).max():.10f}")
print(f"    Match? {np.allclose(pt_np, mlx_np, atol=1e-6)}")

if pt_np.shape != mlx_np.shape:
    print(f"\n  ⚠️  Shape mismatch: {pt_np.shape} vs {mlx_np.shape}")
else:
    print(f"  ✓ Shapes match and weights are identical")

# Check MLX Linear layer convention
print(f"\n3️⃣ MLX Linear Layer Convention Test:")
test_linear = nn.Linear(10, 5)
print(f"  MLX Linear(10, 5) weight shape: {test_linear.weight.shape}")
print(f"  Expected: [5, 10] = [out_features, in_features]")

# Test with known values
test_input = mx.ones((1, 10)) * 2.0
test_linear.weight = mx.ones((5, 10)) * 0.5
test_linear.bias = mx.zeros((5,))
test_output = test_linear(test_input)
print(f"\n  Test: input=2.0, weight=0.5, bias=0")
print(f"  Expected output: 2.0 * 0.5 * 10 = {2.0 * 0.5 * 10}")
print(f"  Actual output: {test_output[0, 0]}")

if float(test_output[0, 0]) == 10.0:
    print(f"  ✓ MLX uses [out, in] format (same as PyTorch)")
else:
    print(f"  ⚠️  MLX might use different format!")

print("\n4️⃣ LSTM Weight Format Investigation:")
print("\n  PyTorch LSTM weight_ih_l0:")
pt_lstm_ih = pt_model.lstm.weight_ih_l0
print(f"    Shape: {pt_lstm_ih.shape}")  # [hidden_size*4, input_size]
print(f"    [512, 60] = [128*4, 60]")
print(f"    Format: [4*hidden, input]")
print(f"    Gate order: input, forget, cell, output (i,f,g,o)")

print(f"\n  MLX LSTM weight loaded:")
mlx_lstm_ih = weights['lstm.weight_ih_l0']
print(f"    Shape: {mlx_lstm_ih.shape}")
print(f"    First 5x5 block:")
for i in range(5):
    print(f"      {mlx_lstm_ih[i, :5]}")

# Check gate order by examining bias patterns
print(f"\n5️⃣ Checking LSTM Gate Order:")
pt_bias_ih = pt_model.lstm.bias_ih_l0.detach().cpu().numpy()
pt_bias_hh = pt_model.lstm.bias_hh_l0.detach().cpu().numpy()
pt_bias = pt_bias_ih + pt_bias_hh

print(f"  PyTorch combined bias (first 16 values of each gate):")
print(f"    Gate 0 (input):  mean={pt_bias[0:128].mean():.4f}, std={pt_bias[0:128].std():.4f}")
print(f"    Gate 1 (forget): mean={pt_bias[128:256].mean():.4f}, std={pt_bias[128:256].std():.4f}")
print(f"    Gate 2 (cell):   mean={pt_bias[256:384].mean():.4f}, std={pt_bias[256:384].std():.4f}")
print(f"    Gate 3 (output): mean={pt_bias[384:512].mean():.4f}, std={pt_bias[384:512].std():.4f}")

# Check if forget gate has positive bias (common initialization)
if pt_bias[128:256].mean() > pt_bias[0:128].mean():
    print(f"  ✓ Forget gate bias is higher (typical for LSTM)")

mlx_bias = np.array((weights['lstm.bias_ih_l0'] + weights['lstm.bias_hh_l0']).tolist())
print(f"\n  MLX combined bias:")
print(f"    Gate 0: mean={mlx_bias[0:128].mean():.4f}, std={mlx_bias[0:128].std():.4f}")
print(f"    Gate 1: mean={mlx_bias[128:256].mean():.4f}, std={mlx_bias[128:256].std():.4f}")
print(f"    Gate 2: mean={mlx_bias[256:384].mean():.4f}, std={mlx_bias[256:384].std():.4f}")
print(f"    Gate 3: mean={mlx_bias[384:512].mean():.4f}, std={mlx_bias[384:512].std():.4f}")

print(f"\n  Bias match: {np.allclose(pt_bias, mlx_bias, atol=1e-6)}")

# Check LSTM weight matrix structure
print(f"\n6️⃣ LSTM Weight Matrix Structure:")
pt_lstm_ih_np = pt_lstm_ih.detach().cpu().numpy()
mlx_lstm_ih_np = np.array(mlx_lstm_ih.tolist())

print(f"  Direct match: {np.allclose(pt_lstm_ih_np, mlx_lstm_ih_np, atol=1e-6)}")
print(f"  Max diff: {np.abs(pt_lstm_ih_np - mlx_lstm_ih_np).max():.10f}")

# Check if gates are in different order
print(f"\n7️⃣ Testing Different Gate Orders:")
gate_size = 128
gates_pt = [
    pt_lstm_ih_np[0:128, :],      # Gate 0
    pt_lstm_ih_np[128:256, :],    # Gate 1  
    pt_lstm_ih_np[256:384, :],    # Gate 2
    pt_lstm_ih_np[384:512, :]     # Gate 3
]

gates_mlx = [
    mlx_lstm_ih_np[0:128, :],
    mlx_lstm_ih_np[128:256, :],
    mlx_lstm_ih_np[256:384, :],
    mlx_lstm_ih_np[384:512, :]
]

# Try different permutations
from itertools import permutations
orders = {
    'ifgo': [0,1,2,3],  # PyTorch order
    'ifog': [0,1,3,2],
    'igfo': [0,2,1,3],
    'gifo': [2,0,1,3],  # Sometimes used
}

print(f"  Testing gate order permutations:")
for name, order in orders.items():
    reordered = np.concatenate([gates_pt[i] for i in order], axis=0)
    diff = np.abs(reordered - mlx_lstm_ih_np).max()
    print(f"    {name}: max diff = {diff:.10f}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
