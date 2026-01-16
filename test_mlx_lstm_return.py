"""
Quick test to understand MLX LSTM return value.
"""
import mlx.core as mx
import mlx.nn as nn

# Create simple LSTM
lstm = nn.LSTM(3, 4)

# Create test input
x = mx.random.normal((1, 5, 3))

# Run forward
result = lstm(x)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

if isinstance(result, tuple):
    print(f"Tuple length: {len(result)}")
    for i, item in enumerate(result):
        print(f"  Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
