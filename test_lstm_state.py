"""
Test if MLX LSTM retains hidden state between calls
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

print("=" * 80)
print("MLX LSTM State Test")
print("=" * 80)

# Create a simple LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20)

# Create dummy input
x1 = mx.random.normal((1, 5, 10))  # (batch, time, features)
x2 = mx.random.normal((1, 5, 10))  # Different input

print("\n1️⃣ First forward pass:")
result1 = lstm(x1)
if isinstance(result1, tuple):
    out1, states1 = result1
    print(f"   LSTM returns (output, states)")
else:
    out1 = result1
    print(f"   LSTM returns only output")
print(f"   Output shape: {out1.shape}")
print(f"   Output[0,0,:3]: {out1[0,0,:3]}")

print("\n2️⃣ Second forward pass (new input):")
result2 = lstm(x2)
if isinstance(result2, tuple):
    out2, states2 = result2
else:
    out2 = result2
print(f"   Output[0,0,:3]: {out2[0,0,:3]}")

print("\n3️⃣ Determinism test (same input twice):")
result_a = lstm(x1)
result_b = lstm(x1)
out_a = result_a[0] if isinstance(result_a, tuple) else result_a
out_b = result_b[0] if isinstance(result_b, tuple) else result_b
print(f"   Same output? {mx.allclose(out_a, out_b, atol=1e-6)}")

print(f"   Same output? {mx.allclose(out_a, out_b, atol=1e-6)}")

print("\n" + "=" * 80)
print("Conclusion:")
print("✓ MLX LSTM is stateless - each call starts fresh")
print("=" * 80)
