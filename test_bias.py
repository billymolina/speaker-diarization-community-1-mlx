"""Test LSTM bias combination"""
import mlx.core as mx
import torch
from pyannote.audio.models.segmentation import PyanNet

# Load PyTorch model
pt_model = PyanNet()

# Check biases
print("Checking bias values:")
print(f"  bias_ih_l0: {pt_model.state_dict()['lstm.bias_ih_l0'][:10]}")
print(f"  bias_hh_l0: {pt_model.state_dict()['lstm.bias_hh_l0'][:10]}")
print(f"  Combined: {(pt_model.state_dict()['lstm.bias_ih_l0'] + pt_model.state_dict()['lstm.bias_hh_l0'])[:10]}")

print("\nBias statistics:")
for name, param in pt_model.named_parameters():
    if 'bias' in name and 'lstm' in name:
        print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")

# Load MLX converted weights
weights = mx.load("models/segmentation_mlx/weights.npz")
print("\nMLX weights:")
for i in range(4):
    bias_ih = weights[f'lstm.bias_ih_l{i}']
    bias_hh = weights[f'lstm.bias_hh_l{i}']
    combined = bias_ih + bias_hh
    print(f"  Layer {i} combined bias: mean={combined.mean().item():.6f}, std={combined.std().item():.6f}, range=[{combined.min().item():.6f}, {combined.max().item():.6f}]")

print("\n" + "="*70)
print("PyTorch LSTM expects separate ih and hh biases")
print("MLX LSTM has a single bias parameter")
print("Question: Should we add them or just use one?")
print("="*70)

# Check PyTorch LSTM code to see how biases are used
print("\nPyTorch LSTM formula:")
print("  i, f, g, o = (input @ W_ih.T + b_ih) + (hidden @ W_hh.T + b_hh)")
print("  Therefore b_ih + b_hh is correct")

print("\nMLX LSTM formula:")
print("  Need to check MLX source code...")
