"""
Check if BatchNorm running statistics exist in the converted weights.
"""

import mlx.core as mx
from pathlib import Path

weights_path = Path("models/embedding_mlx/weights.npz")

if not weights_path.exists():
    print(f"Weights file not found: {weights_path}")
    exit(1)

print(f"Loading weights from {weights_path}")
state_dict = mx.load(str(weights_path))

print(f"\nTotal keys in checkpoint: {len(state_dict)}")

# Filter for BatchNorm related keys
bn_keys = [k for k in state_dict.keys() if 'bn' in k.lower() or 'batchnorm' in k.lower()]
print(f"\nBatchNorm related keys: {len(bn_keys)}")

# Look for running stats
running_mean_keys = [k for k in state_dict.keys() if 'running_mean' in k]
running_var_keys = [k for k in state_dict.keys() if 'running_var' in k]

print(f"\n{'='*70}")
print("Running Statistics:")
print(f"{'='*70}")
print(f"running_mean keys: {len(running_mean_keys)}")
print(f"running_var keys: {len(running_var_keys)}")

if running_mean_keys:
    print("\nrunning_mean keys found:")
    for key in running_mean_keys[:10]:
        print(f"  - {key}")
    if len(running_mean_keys) > 10:
        print(f"  ... and {len(running_mean_keys) - 10} more")

if running_var_keys:
    print("\nrunning_var keys found:")
    for key in running_var_keys[:10]:
        print(f"  - {key}")
    if len(running_var_keys) > 10:
        print(f"  ... and {len(running_var_keys) - 10} more")

# Check a specific bn_fc layer
bn_fc_keys = [k for k in state_dict.keys() if 'seg_1' in k or 'bn_fc' in k or (k.startswith('resnet.') and 'seg_1' in k)]
print(f"\n{'='*70}")
print("bn_fc (final BatchNorm) keys:")
print(f"{'='*70}")
for key in bn_fc_keys:
    value = state_dict[key]
    print(f"{key}")
    print(f"  Shape: {value.shape}")
    print(f"  Mean: {float(mx.mean(value)):.6f}")
    print(f"  Std: {float(mx.std(value)):.6f}")

# Show all keys for reference
print(f"\n{'='*70}")
print("All checkpoint keys:")
print(f"{'='*70}")
for i, key in enumerate(sorted(state_dict.keys())):
    if i < 30:
        print(f"{i+1:3d}. {key}")
    elif i == 30:
        print(f"  ... and {len(state_dict) - 30} more")
        break
