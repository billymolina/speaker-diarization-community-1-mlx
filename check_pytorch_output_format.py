"""
Check what PyTorch model actually returns - raw logits or log probabilities?
"""
import torch
from pyannote.audio import Model
import torchaudio

# Load model
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
model.eval()

# Check the model structure
print("=" * 80)
print("PyTorch Model Structure Analysis")
print("=" * 80)

print("\n1️⃣ Model Architecture:")
print(model)

print("\n2️⃣ Last layers:")
# Try to access the last few layers
if hasattr(model, 'model'):
    inner_model = model.model
    print("Inner model type:", type(inner_model))
    print(inner_model)
else:
    print("No inner model attribute")

# Test with dummy input
print("\n3️⃣ Test with dummy input:")
dummy_input = torch.randn(1, 1, 16000)  # 1 second of audio
with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Output mean: {output.mean():.3f}")
print(f"Output std: {output.std():.3f}")

# Check if outputs sum to 1 when exponentiated (log softmax check)
exp_sum = torch.exp(output).sum(dim=-1)
print(f"\nexp(output).sum(dim=-1):")
print(f"  Mean: {exp_sum.mean():.6f} (should be ~1.0 for log_softmax)")
print(f"  Min: {exp_sum.min():.6f}")
print(f"  Max: {exp_sum.max():.6f}")

# Check if outputs sum to 1 when softmaxed (raw logits check)
softmax_sum = torch.softmax(output, dim=-1).sum(dim=-1)
print(f"\nsoftmax(output).sum(dim=-1):")
print(f"  Mean: {softmax_sum.mean():.6f} (should be ~1.0)")

# Conclusion
if abs(exp_sum.mean() - 1.0) < 0.01:
    print("\n✓ PyTorch model returns LOG PROBABILITIES (log_softmax)")
elif output.max() > 10:
    print("\n✓ PyTorch model returns RAW LOGITS (no activation)")
else:
    print("\n? PyTorch model output format unclear")

print("\n" + "=" * 80)
