"""
Test to understand PyTorch bidirectional LSTM behavior and weight structure.
"""
import torch
import torch.nn as nn
import numpy as np

# Create a simple bidirectional LSTM
batch_size = 1
seq_len = 5
input_size = 3
hidden_size = 4

# Create test input
x = torch.randn(batch_size, seq_len, input_size)
print(f"Input shape: {x.shape}")
print(f"Input:\n{x[0]}\n")

# Create bidirectional LSTM (single layer)
lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

# Run forward pass
output, (h_n, c_n) = lstm(x)
print(f"Output shape: {output.shape}")  # Should be (batch, seq, 2*hidden)
print(f"h_n shape: {h_n.shape}")  # Should be (2, batch, hidden)
print(f"c_n shape: {c_n.shape}\n")

print("Output (concatenated forward and backward):")
print(output[0])
print()

# Now manually implement bidirectional LSTM
print("=" * 80)
print("MANUAL BIDIRECTIONAL LSTM IMPLEMENTATION")
print("=" * 80)

# Create separate forward and backward LSTMs
lstm_fwd = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
lstm_bwd = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

# Copy weights from bidirectional LSTM
# PyTorch stores bidirectional weights as:
# - weight_ih_l0: forward input-hidden
# - weight_hh_l0: forward hidden-hidden
# - weight_ih_l0_reverse: backward input-hidden
# - weight_hh_l0_reverse: backward hidden-hidden

print("\nCopying weights from bidirectional LSTM to separate LSTMs...")
with torch.no_grad():
    lstm_fwd.weight_ih_l0.copy_(lstm.weight_ih_l0)
    lstm_fwd.weight_hh_l0.copy_(lstm.weight_hh_l0)
    lstm_fwd.bias_ih_l0.copy_(lstm.bias_ih_l0)
    lstm_fwd.bias_hh_l0.copy_(lstm.bias_hh_l0)
    
    lstm_bwd.weight_ih_l0.copy_(lstm.weight_ih_l0_reverse)
    lstm_bwd.weight_hh_l0.copy_(lstm.weight_hh_l0_reverse)
    lstm_bwd.bias_ih_l0.copy_(lstm.bias_ih_l0_reverse)
    lstm_bwd.bias_hh_l0.copy_(lstm.bias_hh_l0_reverse)

# Run manual bidirectional pass
print("\nRunning manual bidirectional LSTM:")

# Forward pass
output_fwd, (h_fwd, c_fwd) = lstm_fwd(x)
print(f"Forward output shape: {output_fwd.shape}")
print(f"Forward output:\n{output_fwd[0]}\n")

# Backward pass (reverse input)
x_reverse = torch.flip(x, dims=[1])  # Reverse time dimension
print(f"Reversed input:\n{x_reverse[0]}\n")

output_bwd, (h_bwd, c_bwd) = lstm_bwd(x_reverse)
print(f"Backward output shape (before reversing back): {output_bwd.shape}")
print(f"Backward output (on reversed input):\n{output_bwd[0]}\n")

# Reverse backward output back
output_bwd = torch.flip(output_bwd, dims=[1])
print(f"Backward output (after reversing back):\n{output_bwd[0]}\n")

# Concatenate
manual_output = torch.cat([output_fwd, output_bwd], dim=-1)
print(f"Manual concatenated output shape: {manual_output.shape}")
print(f"Manual output:\n{manual_output[0]}\n")

# Compare with PyTorch bidirectional output
print("=" * 80)
print("COMPARISON")
print("=" * 80)
diff = torch.abs(output - manual_output)
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")
print(f"Match: {torch.allclose(output, manual_output, atol=1e-6)}")

if not torch.allclose(output, manual_output, atol=1e-6):
    print("\nDifference matrix:")
    print(diff[0])
    print("\nPyTorch bidirectional output:")
    print(output[0])
    print("\nManual bidirectional output:")
    print(manual_output[0])
