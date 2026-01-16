"""
Convert pyannote/wespeaker-voxceleb-resnet34-LM embedding model to MLX format.
"""

import argparse
import mlx.core as mx
import numpy as np
from pathlib import Path
import torch
from typing import Dict


def convert_conv2d_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert Conv2d weight from PyTorch to MLX format.

    PyTorch Conv2d: (out_channels, in_channels, height, width)
    MLX Conv2d: (out_channels, height, width, in_channels)
    """
    weight_np = weight.detach().cpu().numpy()
    # Transpose from (out, in, h, w) to (out, h, w, in)
    weight_mlx = np.transpose(weight_np, (0, 2, 3, 1))
    return mx.array(weight_mlx, dtype=mx.float32)


def convert_batchnorm_weight(weight: torch.Tensor) -> mx.array:
    """Convert BatchNorm weight to MLX format."""
    return mx.array(weight.detach().cpu().numpy(), dtype=mx.float32)


def map_pytorch_to_mlx_keys(pytorch_key: str) -> str:
    """
    Map PyTorch model keys to MLX model keys.

    PyTorch WeSpeaker ResNet34 structure:
        - conv1, bn1
        - layer1.0.conv1, layer1.0.bn1, ...
        - layer2.0.conv1, ...
        - layer3.0.conv1, ...
        - layer4.0.conv1, ...
        - fc, bn

    MLX ResNet34Embedding structure matches closely
    """
    # Direct mappings
    key = pytorch_key

    # Handle batch norm running stats (MLX uses different names)
    key = key.replace('running_mean', 'running_mean')
    key = key.replace('running_var', 'running_var')
    key = key.replace('num_batches_tracked', 'num_batches_tracked')

    # Handle shortcut connections
    key = key.replace('downsample.0', 'shortcut_conv')
    key = key.replace('downsample.1', 'shortcut_bn')

    # Handle final layers
    key = key.replace('fc.weight', 'fc.weight')
    key = key.replace('fc.bias', 'fc.bias')
    key = key.replace('fc_bn', 'bn_fc')

    return key


def load_pytorch_model():
    """
    Load the PyTorch model from pyannote Hub.

    Returns:
        PyTorch model or state dict
    """
    try:
        from pyannote.audio import Model

        print("[INFO] Loading model from pyannote.audio...")
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        print("[INFO] Model loaded successfully")

        return model.state_dict()

    except ImportError:
        print("[ERROR] pyannote.audio not installed. Install with:")
        print("  pip install pyannote.audio")
        raise

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("\nTrying alternative method...")

        # Alternative: Load from cached HuggingFace
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin"
        )

        print(f"[INFO] Loading from {model_path}")

        # Load with custom handler to avoid pyannote dependency
        import pickle

        class ModelLoader:
            """Custom loader to handle pyannote models."""

            def __call__(self, *args, **kwargs):
                return None

        # Inject dummy classes
        import sys
        sys.modules['pyannote'] = type('module', (), {})()
        sys.modules['pyannote.audio'] = type('module', (), {})()
        sys.modules['pyannote.audio.models'] = type('module', (), {})()
        sys.modules['pyannote.audio.models.embedding'] = type('module', (), {
            'WeSpeakerResNet34': ModelLoader
        })()

        checkpoint = torch.load(model_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            # Assume the checkpoint is the state_dict
            return checkpoint


def convert_weights(state_dict: Dict) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX format.

    Args:
        state_dict: PyTorch state dictionary

    Returns:
        MLX weights dictionary
    """
    mlx_weights = {}

    print(f"\n[INFO] Converting {len(state_dict)} tensors...")

    for key, value in state_dict.items():
        # Skip non-parameter tensors
        if 'num_batches_tracked' in key:
            continue

        # Map key names
        mlx_key = map_pytorch_to_mlx_keys(key)

        # Convert weights based on layer type
        if ('conv' in key and 'weight' in key) or '.shortcut.0.weight' in key:
            # Conv2d weights need transposition (including 1x1 convs in shortcuts)
            mlx_weights[mlx_key] = convert_conv2d_weight(value)
            print(f"  Conv2d: {key} -> {mlx_key} {list(value.shape)} -> {list(mlx_weights[mlx_key].shape)}")

        elif any(x in key for x in ['weight', 'bias', 'running_mean', 'running_var']):
            # Linear, BatchNorm, and other parameters
            mlx_weights[mlx_key] = convert_batchnorm_weight(value)
            print(f"  Param:  {key} -> {mlx_key} {list(value.shape)}")

        else:
            print(f"  Skip:   {key}")

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert pyannote/wespeaker-voxceleb-resnet34-LM to MLX format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/embedding_mlx",
        help="Output directory for MLX model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float32",
        help="Data type for conversion"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Converting pyannote/wespeaker-voxceleb-resnet34-LM to MLX")
    print("=" * 60)

    # Load PyTorch model
    try:
        state_dict = load_pytorch_model()
        print(f"[INFO] Loaded {len(state_dict)} parameters")

        # Print model structure
        print("\n[INFO] Model structure:")
        for key in sorted(state_dict.keys()):
            print(f"  {key:60s} {list(state_dict[key].shape)}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 1

    # Convert weights
    mlx_weights = convert_weights(state_dict)

    # Convert dtype if requested
    if args.dtype == "float16":
        print(f"\n[INFO] Converting to {args.dtype}...")
        mlx_weights = {k: v.astype(mx.float16) for k, v in mlx_weights.items()}

    # Save weights
    weights_file = output_path / "weights.npz"
    print(f"\n[INFO] Saving weights to {weights_file}...")
    mx.savez(str(weights_file), **mlx_weights)

    # Save config
    config = {
        "model_type": "resnet34_embedding",
        "feat_dim": 80,
        "embed_dim": 256,
        "m_channels": 32,
        "pooling": "TSTP",
        "dtype": args.dtype,
        "source": "pyannote/wespeaker-voxceleb-resnet34-LM"
    }

    import json
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Saved config to {config_file}")

    # Print summary
    total_params = sum(v.size for v in mlx_weights.values())
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Conversion complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Output directory: {output_path}")
    print(f"  Weights file: {weights_file}")
    print(f"  Config file: {config_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
