"""Compare PyanNet (local) vs HF model"""
import torch
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio import Model

# Load both
local = PyanNet()
hf = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

print("Comparing bias_ih_l0:")
print(f"\nPyanNet (random init):")
local_bias = local.state_dict()['lstm.bias_ih_l0'].numpy()
print(f"  Mean: {local_bias.mean():.6f}, Std: {local_bias.std():.6f}")
print(f"  Range: [{local_bias.min():.6f}, {local_bias.max():.6f}]")
print(f"  First 10: {local_bias[:10]}")

print(f"\nHuggingFace (pretrained):")
hf_bias = hf.state_dict()['lstm.bias_ih_l0'].cpu().numpy()
print(f"  Mean: {hf_bias.mean():.6f}, Std: {hf_bias.std():.6f}")
print(f"  Range: [{hf_bias.min():.6f}, {hf_bias.max():.6f}]")
print(f"  First 10: {hf_bias[:10]}")

print("\nConclusion: The HF model really has these large biases!")
print("This is expected for a trained LSTM model.")
