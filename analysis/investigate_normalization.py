"""
Investigate potential scaling/normalization differences
Check if there's a hidden normalization or scaling factor we're missing
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import load_pyannote_model
from pyannote.audio import Model as PyannoteModel

print("=" * 80)
print("INVESTIGATING NORMALIZATION/SCALING ISSUES")
print("=" * 80)

# Load models
print("\n1️⃣ Loading models...")
mlx_model = load_pyannote_model("models/segmentation_mlx/weights.npz")
pytorch_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Check model internals
print("\n2️⃣ Checking PyTorch model for hidden normalizations...")
print("\nPyTorch model structure:")
for name, module in pytorch_model.named_modules():
    # Look for normalization layers or parameters
    if any(x in name.lower() for x in ['norm', 'scale', 'bias', 'dropout']):
        print(f"  {name}: {type(module).__name__}")

# Check LSTM dropout
if hasattr(pytorch_model, 'lstm'):
    print(f"\n3️⃣ LSTM Configuration:")
    print(f"  Dropout: {pytorch_model.lstm.dropout}")
    print(f"  Training mode: {pytorch_model.training}")
    print(f"  Note: Dropout should be OFF during eval()")

# Check for any scaling parameters in linear layers
print("\n4️⃣ Checking Linear Layer Parameters:")
if hasattr(pytorch_model, 'linear'):
    for i, layer in enumerate(pytorch_model.linear):
        print(f"\n  Linear layer {i}:")
        print(f"    Weight shape: {layer.weight.shape}")
        print(f"    Bias shape: {layer.bias.shape}")
        print(f"    Weight mean: {layer.weight.mean():.6f}")
        print(f"    Weight std: {layer.weight.std():.6f}")
        print(f"    Bias mean: {layer.bias.mean():.6f}")
        print(f"    Bias std: {layer.bias.std():.6f}")

# Check classifier
if hasattr(pytorch_model, 'classifier'):
    print(f"\n5️⃣ Classifier Parameters:")
    print(f"  Weight shape: {pytorch_model.classifier.weight.shape}")
    print(f"  Bias shape: {pytorch_model.classifier.bias.shape}")
    print(f"  Weight mean: {pytorch_model.classifier.weight.mean():.6f}")
    print(f"  Weight std: {pytorch_model.classifier.weight.std():.6f}")
    print(f"  Bias mean: {pytorch_model.classifier.bias.mean():.6f}")
    print(f"  Bias std: {pytorch_model.classifier.bias.std():.6f}")

# Compare with MLX model
print(f"\n6️⃣ MLX Model Parameters:")
print(f"  Linear1 weight mean: {float(mx.mean(mlx_model.linear1.weight)):.6f}")
print(f"  Linear1 weight std: {float(mx.std(mlx_model.linear1.weight)):.6f}")
print(f"  Linear1 bias mean: {float(mx.mean(mlx_model.linear1.bias)):.6f}")

print(f"  Linear2 weight mean: {float(mx.mean(mlx_model.linear2.weight)):.6f}")
print(f"  Linear2 weight std: {float(mx.std(mlx_model.linear2.weight)):.6f}")
print(f"  Linear2 bias mean: {float(mx.mean(mlx_model.linear2.bias)):.6f}")

print(f"  Classifier weight mean: {float(mx.mean(mlx_model.classifier.weight)):.6f}")
print(f"  Classifier weight std: {float(mx.std(mlx_model.classifier.weight)):.6f}")
print(f"  Classifier bias mean: {float(mx.mean(mlx_model.classifier.bias)):.6f}")

# Check if weights match
print(f"\n7️⃣ Weight Matching Check:")
pt_linear1_weight = pytorch_model.linear[0].weight.detach().cpu().numpy()
mlx_linear1_weight = mx.array(mlx_model.linear1.weight).tolist()
import numpy as np
mlx_linear1_weight_np = np.array(mlx_linear1_weight)

diff = abs(pt_linear1_weight - mlx_linear1_weight_np).max()
print(f"  Linear1 weight max diff: {diff:.10f}")

pt_classifier_weight = pytorch_model.classifier.weight.detach().cpu().numpy()
mlx_classifier_weight_np = np.array(mx.array(mlx_model.classifier.weight).tolist())

diff = abs(pt_classifier_weight - mlx_classifier_weight_np).max()
print(f"  Classifier weight max diff: {diff:.10f}")

# Test dropout behavior
print(f"\n8️⃣ Dropout Test:")
dummy_input = torch.randn(1, 1, 16000)
pytorch_model.train()
with torch.no_grad():
    out_train1 = pytorch_model(dummy_input)
    out_train2 = pytorch_model(dummy_input)

print(f"  Training mode - same input twice:")
print(f"    Outputs identical? {torch.allclose(out_train1, out_train2)}")
print(f"    (Should be False if dropout is active)")

pytorch_model.eval()
with torch.no_grad():
    out_eval1 = pytorch_model(dummy_input)
    out_eval2 = pytorch_model(dummy_input)

print(f"  Eval mode - same input twice:")
print(f"    Outputs identical? {torch.allclose(out_eval1, out_eval2)}")
print(f"    (Should be True in eval mode)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\n📋 Key Findings:")
print("  - Check if LSTM dropout is properly disabled in eval mode")
print("  - Verify all normalization layers match between models")
print("  - Confirm weight loading is correct")
