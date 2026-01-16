"""Check LSTM weight shapes"""
import mlx.core as mx
import torch
from pyannote.audio.models.segmentation import PyanNet

# Load PyTorch model
pt_model = PyanNet()
print("PyTorch LSTM weights:")
for name, param in pt_model.named_parameters():
    if 'lstm' in name:
        print(f"  {name}: {param.shape}")

print("\nMLX converted weights:")
weights = mx.load("models/segmentation_mlx/weights.npz")
for key in sorted(weights.keys()):
    if 'lstm' in key:
        print(f"  {key}: {weights[key].shape}")

# Check if weights need transposing
print("\n" + "="*70)
print("Checking LSTM weight formats:")
print("="*70)

# PyTorch LSTM weight format:
print("\nPyTorch LSTM l0:")
print(f"  weight_ih_l0: {pt_model.state_dict()['lstm.weight_ih_l0'].shape}")  
print(f"  weight_hh_l0: {pt_model.state_dict()['lstm.weight_hh_l0'].shape}")

# MLX LSTM expects:
# - Wx: (input_size, 4 * hidden_size) for ih
# - Wh: (hidden_size, 4 * hidden_size) for hh
print("\nMLX LSTM expected format:")
print("  Wx: (input_size, 4*hidden_size)")
print("  Wh: (hidden_size, 4*hidden_size)")

print("\nPyTorch format:")
print("  weight_ih: (4*hidden_size, input_size)")
print("  weight_hh: (4*hidden_size, hidden_size)")

print("\n⚠️  NEED TO TRANSPOSE!")
print("  PyTorch weight_ih (4*hidden, input) -> MLX Wx (input, 4*hidden)")
print("  PyTorch weight_hh (4*hidden, hidden) -> MLX Wh (hidden, 4*hidden)")
