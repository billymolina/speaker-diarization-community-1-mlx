"""
Test if running statistics are being loaded correctly.
"""

from src.resnet_embedding import load_resnet34_embedding
import mlx.core as mx

print("="*70)
print("Testing Running Statistics Loading")
print("="*70)

model = load_resnet34_embedding("models/embedding_mlx/weights.npz")

print("\n" + "="*70)
print("Checking BatchNorm running statistics:")
print("="*70)

bn_layers = [
    ('bn1', model.bn1),
    ('layer1.layers[0].bn1', model.layer1.layers[0].bn1),
    ('layer1.layers[0].bn2', model.layer1.layers[0].bn2),
    ('layer2.layers[0].bn1', model.layer2.layers[0].bn1),
    ('layer3.layers[0].bn1', model.layer3.layers[0].bn1),
    ('layer4.layers[0].bn1', model.layer4.layers[0].bn1),
]

for name, bn in bn_layers:
    print(f"\n{name}:")
    if hasattr(bn, 'running_mean') and bn.running_mean is not None:
        print(f"  running_mean: shape={bn.running_mean.shape}, mean={float(mx.mean(bn.running_mean)):.6f}")
    else:
        print(f"  running_mean: NOT LOADED or None")

    if hasattr(bn, 'running_var') and bn.running_var is not None:
        print(f"  running_var: shape={bn.running_var.shape}, mean={float(mx.mean(bn.running_var)):.6f}")
    else:
        print(f"  running_var: NOT LOADED or None")

print("\n" + "="*70)
print("Testing inference mode:")
print("="*70)

import numpy as np
mel = mx.array(np.random.randn(1, 100, 80, 1).astype(np.float32))

output = model(mel)
print(f"Output shape: {output.shape}")
print(f"Output norm: {float(mx.linalg.norm(output)):.6f}")
print(f"Sample values: {output[0, :5]}")
print(f"All zeros: {bool(mx.all(output == 0))}")
