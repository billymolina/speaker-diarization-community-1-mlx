"""Check PyTorch model bias values directly"""
import torch
from pyannote.audio import Model

# Load model
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

print("PyTorch model LSTM bias values:")
state_dict = model.state_dict()

for key in sorted(state_dict.keys()):
    if 'bias' in key and 'lstm' in key:
        w = state_dict[key].cpu().numpy()
        print(f"\n{key}:")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean():.6f}")
        print(f"  Std: {w.std():.6f}")
        print(f"  Range: [{w.min():.6f}, {w.max():.6f}]")
        print(f"  First 10 values: {w[:10]}")
