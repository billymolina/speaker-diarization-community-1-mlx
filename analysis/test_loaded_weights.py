"""
Test MLX LSTM with loaded PyTorch weights to see if the weights themselves cause saturation.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Load weights
weights = mx.load("models/segmentation_mlx/weights.npz")
print(f"Loaded {len(weights)} weight tensors")

# Check LSTM weight statistics
print("\n" + "=" * 80)
print("LSTM WEIGHT STATISTICS")
print("=" * 80)

for i in range(4):
    print(f"\nLayer {i}:")
    
    # Forward LSTM
    w_ih = weights[f'lstm.weight_ih_l{i}']
    w_hh = weights[f'lstm.weight_hh_l{i}']
    b_ih = weights[f'lstm.bias_ih_l{i}']
    b_hh = weights[f'lstm.bias_hh_l{i}']
    b_combined = b_ih + b_hh
    
    print(f"  Forward:")
    print(f"    weight_ih: shape={w_ih.shape}, range=[{w_ih.min():.6f}, {w_ih.max():.6f}], std={w_ih.std():.6f}")
    print(f"    weight_hh: shape={w_hh.shape}, range=[{w_hh.min():.6f}, {w_hh.max():.6f}], std={w_hh.std():.6f}")
    print(f"    bias_combined: range=[{b_combined.min():.6f}, {b_combined.max():.6f}], std={b_combined.std():.6f}")
    
    # Backward LSTM
    w_ih_rev = weights[f'lstm.weight_ih_l{i}_reverse']
    w_hh_rev = weights[f'lstm.weight_hh_l{i}_reverse']
    b_ih_rev = weights[f'lstm.bias_ih_l{i}_reverse']
    b_hh_rev = weights[f'lstm.bias_hh_l{i}_reverse']
    b_combined_rev = b_ih_rev + b_hh_rev
    
    print(f"  Backward:")
    print(f"    weight_ih_reverse: shape={w_ih_rev.shape}, range=[{w_ih_rev.min():.6f}, {w_ih_rev.max():.6f}], std={w_ih_rev.std():.6f}")
    print(f"    weight_hh_reverse: shape={w_hh_rev.shape}, range=[{w_hh_rev.min():.6f}, {w_hh_rev.max():.6f}], std={w_hh_rev.std():.6f}")
    print(f"    bias_combined_reverse: range=[{b_combined_rev.min():.6f}, {b_combined_rev.max():.6f}], std={b_combined_rev.std():.6f}")

# Create test input (same as what SincNet would output)
print("\n" + "=" * 80)
print("TESTING WITH LOADED WEIGHTS")
print("=" * 80)

# Create random input similar to SincNet output
batch_size = 1
seq_len = 100  # ~1 second at 100Hz
input_size = 60  # SincNet output

x = mx.random.normal((batch_size, seq_len, input_size)) * 0.5  # Scale to reasonable range
print(f"\nInput shape: {x.shape}")
print(f"Input range: [{x.min():.6f}, {x.max():.6f}]")
print(f"Input std: {x.std():.6f}")

# Create 4-layer bidirectional LSTM
num_layers = 4
hidden_size = 128
lstm_fwd_layers = []
lstm_bwd_layers = []

for i in range(num_layers):
    input_dim = input_size if i == 0 else hidden_size * 2
    lstm_fwd = nn.LSTM(input_dim, hidden_size)
    lstm_bwd = nn.LSTM(input_dim, hidden_size)
    
    # Load weights
    lstm_fwd.Wx = weights[f'lstm.weight_ih_l{i}']
    lstm_fwd.Wh = weights[f'lstm.weight_hh_l{i}']
    lstm_fwd.bias = weights[f'lstm.bias_ih_l{i}'] + weights[f'lstm.bias_hh_l{i}']
    
    lstm_bwd.Wx = weights[f'lstm.weight_ih_l{i}_reverse']
    lstm_bwd.Wh = weights[f'lstm.weight_hh_l{i}_reverse']
    lstm_bwd.bias = weights[f'lstm.bias_ih_l{i}_reverse'] + weights[f'lstm.bias_hh_l{i}_reverse']
    
    lstm_fwd_layers.append(lstm_fwd)
    lstm_bwd_layers.append(lstm_bwd)

print("\nLoaded weights into 4-layer bidirectional LSTM")

# Run through layers
h = x
for i, (lstm_fwd, lstm_bwd) in enumerate(zip(lstm_fwd_layers, lstm_bwd_layers)):
    print(f"\nLayer {i+1}:")
    print(f"  Input: shape={h.shape}, range=[{h.min():.6f}, {h.max():.6f}], std={h.std():.6f}")
    
    # Forward
    h_fwd, c_fwd = lstm_fwd(h)
    print(f"  Forward output: range=[{h_fwd.min():.6f}, {h_fwd.max():.6f}], std={h_fwd.std():.6f}")
    
    # Check cell state
    print(f"  Forward cell: range=[{c_fwd.min():.6f}, {c_fwd.max():.6f}], std={c_fwd.std():.6f}")
    
    # Backward
    h_rev = h[:, ::-1, :]
    h_bwd, c_bwd = lstm_bwd(h_rev)
    h_bwd = h_bwd[:, ::-1, :]
    print(f"  Backward output: range=[{h_bwd.min():.6f}, {h_bwd.max():.6f}], std={h_bwd.std():.6f}")
    print(f"  Backward cell: range=[{c_bwd.min():.6f}, {c_bwd.max():.6f}], std={c_bwd.std():.6f}")
    
    # Concatenate
    h = mx.concatenate([h_fwd, h_bwd], axis=-1)
    print(f"  Concatenated: range=[{h.min():.6f}, {h.max():.6f}], std={h.std():.6f}")
    
    # Check for saturation
    if mx.abs(h).max() > 0.9:
        print(f"  ⚠️  WARNING: Output values approaching saturation!")
    if mx.abs(h).max() > 0.99:
        print(f"  🔴 CRITICAL: Output values SATURATED!")

print(f"\n" + "=" * 80)
print(f"FINAL RESULT")
print(f"=" * 80)
print(f"Output shape: {h.shape}")
print(f"Output range: [{h.min():.6f}, {h.max():.6f}]")
print(f"Output std: {h.std():.6f}")
print(f"Max absolute value: {mx.abs(h).max():.6f}")
