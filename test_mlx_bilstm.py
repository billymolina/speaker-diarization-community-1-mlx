"""
Test MLX LSTM to see if it behaves the same as PyTorch when manually implementing bidirectional.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Set random seed for reproducibility
mx.random.seed(42)
np.random.seed(42)

# Create test input
batch_size = 1
seq_len = 5
input_size = 3
hidden_size = 4

# Create test input (same shape as PyTorch: batch, seq, input)
x = mx.random.normal((batch_size, seq_len, input_size))
print(f"Input shape: {x.shape}")
print(f"Input:\n{x[0]}\n")

# Create two separate LSTMs for forward and backward
lstm_fwd = nn.LSTM(input_size, hidden_size)
lstm_bwd = nn.LSTM(input_size, hidden_size)

print("=" * 80)
print("MANUAL BIDIRECTIONAL LSTM IN MLX")
print("=" * 80)

# Forward pass
print("\nForward pass:")
output_fwd, cell_fwd = lstm_fwd(x)
print(f"Forward output shape: {output_fwd.shape}")
print(f"Forward output:\n{output_fwd[0]}\n")
print(f"Forward output range: [{output_fwd.min():.6f}, {output_fwd.max():.6f}]")

# Backward pass (reverse time dimension)
print("\nBackward pass:")
x_reverse = x[:, ::-1, :]
print(f"Reversed input shape: {x_reverse.shape}")
print(f"Reversed input:\n{x_reverse[0]}\n")

output_bwd, cell_bwd = lstm_bwd(x_reverse)
print(f"Backward output shape (before reversing back): {output_bwd.shape}")
print(f"Backward output (on reversed input):\n{output_bwd[0]}\n")

# Reverse backward output back
output_bwd = output_bwd[:, ::-1, :]
print(f"Backward output (after reversing back):\n{output_bwd[0]}\n")
print(f"Backward output range: [{output_bwd.min():.6f}, {output_bwd.max():.6f}]")

# Concatenate
manual_output = mx.concatenate([output_fwd, output_bwd], axis=-1)
print(f"\nManual concatenated output shape: {manual_output.shape}")
print(f"Manual output:\n{manual_output[0]}\n")
print(f"Manual output range: [{manual_output.min():.6f}, {manual_output.max():.6f}]")

# Test with multiple layers
print("\n" + "=" * 80)
print("STACKING 4 LAYERS")
print("=" * 80)

# Create 4-layer bidirectional LSTM
num_layers = 4
lstm_fwd_layers = []
lstm_bwd_layers = []

for i in range(num_layers):
    input_dim = input_size if i == 0 else hidden_size * 2
    lstm_fwd_layers.append(nn.LSTM(input_dim, hidden_size))
    lstm_bwd_layers.append(nn.LSTM(input_dim, hidden_size))

# Run through 4 layers
h = x
for i, (lstm_fwd, lstm_bwd) in enumerate(zip(lstm_fwd_layers, lstm_bwd_layers)):
    print(f"\nLayer {i+1}:")
    print(f"  Input shape: {h.shape}")
    print(f"  Input range: [{h.min():.6f}, {h.max():.6f}]")
    
    # Forward
    h_fwd, _ = lstm_fwd(h)
    print(f"  Forward output range: [{h_fwd.min():.6f}, {h_fwd.max():.6f}]")
    
    # Backward
    h_rev = h[:, ::-1, :]
    h_bwd, _ = lstm_bwd(h_rev)
    h_bwd = h_bwd[:, ::-1, :]
    print(f"  Backward output range: [{h_bwd.min():.6f}, {h_bwd.max():.6f}]")
    
    # Concatenate
    h = mx.concatenate([h_fwd, h_bwd], axis=-1)
    print(f"  Concatenated output range: [{h.min():.6f}, {h.max():.6f}]")
    
    # Check for saturation
    if mx.abs(h).max() > 0.9:
        print(f"  ⚠️  WARNING: Output values approaching saturation!")

print(f"\nFinal output shape: {h.shape}")
print(f"Final output range: [{h.min():.6f}, {h.max():.6f}]")
print(f"\nFinal output:\n{h[0]}")
