"""
Deep investigation of LSTM weight loading and bias handling
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyannote.audio import Model as PyannoteModel

print("=" * 80)
print("LSTM WEIGHT INVESTIGATION")
print("=" * 80)

# Load PyTorch model
print("\n1️⃣ Loading PyTorch model...")
pytorch_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0")
pytorch_model.eval()

# Load weights file
print("\n2️⃣ Loading MLX weights file...")
weights = mx.load("models/segmentation_mlx/weights.npz")
print(f"   Loaded {len(weights)} tensors")

# Examine LSTM weights in detail
print("\n3️⃣ PyTorch LSTM Structure:")
print(f"   Type: {type(pytorch_model.lstm)}")
print(f"   Input size: {pytorch_model.lstm.input_size}")
print(f"   Hidden size: {pytorch_model.lstm.hidden_size}")
print(f"   Num layers: {pytorch_model.lstm.num_layers}")
print(f"   Bidirectional: {pytorch_model.lstm.bidirectional}")
print(f"   Dropout: {pytorch_model.lstm.dropout}")

# Check first layer weights
print("\n4️⃣ PyTorch LSTM Layer 0 Weights:")
weight_ih_l0 = pytorch_model.lstm.weight_ih_l0
weight_hh_l0 = pytorch_model.lstm.weight_hh_l0
bias_ih_l0 = pytorch_model.lstm.bias_ih_l0
bias_hh_l0 = pytorch_model.lstm.bias_hh_l0

print(f"   weight_ih_l0 shape: {weight_ih_l0.shape}")  # [hidden*4, input]
print(f"   weight_hh_l0 shape: {weight_hh_l0.shape}")  # [hidden*4, hidden]
print(f"   bias_ih_l0 shape: {bias_ih_l0.shape}")      # [hidden*4]
print(f"   bias_hh_l0 shape: {bias_hh_l0.shape}")      # [hidden*4]

print(f"\n   bias_ih_l0 mean: {bias_ih_l0.mean():.6f}")
print(f"   bias_hh_l0 mean: {bias_hh_l0.mean():.6f}")
print(f"   bias_ih + bias_hh mean: {(bias_ih_l0 + bias_hh_l0).mean():.6f}")

# Check what MLX loaded
print("\n5️⃣ MLX Loaded Weights for LSTM Layer 0:")
mlx_weight_ih = weights['lstm.weight_ih_l0']
mlx_weight_hh = weights['lstm.weight_hh_l0']
mlx_bias_ih = weights['lstm.bias_ih_l0']
mlx_bias_hh = weights['lstm.bias_hh_l0']

print(f"   weight_ih_l0 shape: {mlx_weight_ih.shape}")
print(f"   weight_hh_l0 shape: {mlx_weight_hh.shape}")
print(f"   bias_ih_l0 shape: {mlx_bias_ih.shape}")
print(f"   bias_hh_l0 shape: {mlx_bias_hh.shape}")

print(f"\n   bias_ih_l0 mean: {float(mx.mean(mlx_bias_ih)):.6f}")
print(f"   bias_hh_l0 mean: {float(mx.mean(mlx_bias_hh)):.6f}")
print(f"   bias_ih + bias_hh mean: {float(mx.mean(mlx_bias_ih + mlx_bias_hh)):.6f}")

# Verify weights match
print("\n6️⃣ Weight Matching:")
pt_bias_sum = (bias_ih_l0 + bias_hh_l0).detach().cpu().numpy()
mlx_bias_sum = np.array((mlx_bias_ih + mlx_bias_hh).tolist())

diff = np.abs(pt_bias_sum - mlx_bias_sum).max()
print(f"   PyTorch bias sum matches MLX sum: max diff = {diff:.10f}")

# Check MLX LSTM structure
print("\n7️⃣ MLX LSTM API Check:")
test_lstm = nn.LSTM(input_size=60, hidden_size=128)
print(f"   MLX LSTM parameters:")
for name in dir(test_lstm):
    if 'weight' in name.lower() or 'bias' in name.lower():
        try:
            param = getattr(test_lstm, name)
            if hasattr(param, 'shape'):
                print(f"     {name}: {param.shape}")
        except:
            pass

# Critical check: How does MLX LSTM handle bias?
print("\n8️⃣ MLX LSTM Bias Handling Test:")
print("   Creating test LSTM with known weights...")
test_lstm = nn.LSTM(input_size=10, hidden_size=20)

# Set known bias values
test_bias_ih = mx.ones((80,)) * 0.5  # 80 = 20 * 4 gates
test_bias_hh = mx.ones((80,)) * 0.3

print(f"   If MLX LSTM has ONE bias:")
print(f"     We should set: bias = bias_ih + bias_hh")
print(f"     Expected bias value: 0.5 + 0.3 = 0.8")

print(f"\n   If MLX LSTM has TWO biases:")
print(f"     We should set: bias_ih = 0.5, bias_hh = 0.3")
print(f"     LSTM will add them internally")

# Check MLX LSTM source or documentation
print("\n9️⃣ Checking current model's bias loading...")
print("   From src/models.py lines 587-607:")
print("   Code: self.lstm_forward[i].bias = bias_ih + bias_hh")
print("   ")
print("   ✓ This assumes MLX LSTM has ONE bias parameter")
print("   ✓ PyTorch adds bias_ih + bias_hh internally")
print("   ✓ So we pre-add them for MLX")

# Verify this is correct by checking MLX LSTM parameter count
print("\n🔟 Parameter Count Verification:")
mlx_lstm = nn.LSTM(input_size=60, hidden_size=128)
print(f"   MLX LSTM parameters:")

# Count parameters
param_count = 0
for name, param in mlx_lstm.parameters().items():
    size = np.prod(param.shape)
    param_count += size
    print(f"     {name}: {param.shape} = {size:,} params")

print(f"\n   Total MLX LSTM params: {param_count:,}")

# Compare with PyTorch
pt_lstm = torch.nn.LSTM(input_size=60, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
pt_param_count = sum(p.numel() for p in pt_lstm.parameters())
print(f"   Total PyTorch LSTM params: {pt_param_count:,}")

if param_count == pt_param_count:
    print("\n   ✓ Parameter counts match - bias handling is likely correct")
elif param_count == pt_param_count - 128:  # Missing one bias
    print("\n   ⚠️  MLX has fewer parameters - may need to keep biases separate!")
else:
    print(f"\n   ⚠️  Parameter count mismatch: {param_count} vs {pt_param_count}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)

print("\n📋 Summary:")
print("   Current approach: Add bias_ih + bias_hh before loading into MLX")
print("   Verification needed: Test if this matches PyTorch behavior exactly")
print("   Next step: Create side-by-side test with identical inputs")
