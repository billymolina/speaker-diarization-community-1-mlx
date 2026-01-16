#!/usr/bin/env python3
"""Convert pyannote models to MLX format."""

from pyannote.audio import Model
import torch
import mlx.core as mx
from pathlib import Path
import sys


def convert_model(repo_id: str, output_dir: str):
    """Convert a pyannote model to MLX format."""
    print(f'[INFO] Loading model from {repo_id}...')
    
    try:
        # Load the pyannote model
        model = Model.from_pretrained(repo_id, use_auth_token=True)
        print('[INFO] Model loaded successfully')
        
        # Extract state dict
        state_dict = model.state_dict()
        print(f'[INFO] Model has {len(state_dict)} layers')
        
        # Convert to MLX format
        print('[INFO] Converting to MLX format...')
        mlx_weights = {}
        for key, value in state_dict.items():
            print(f'  Converting {key}: {list(value.shape)}')
            mlx_weights[key] = mx.array(value.cpu().numpy())
        
        # Save MLX model
        print('[INFO] Saving MLX model...')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mx.savez(str(output_path / 'weights.npz'), **mlx_weights)
        
        print(f'[SUCCESS] Model converted and saved to {output_path}')
        print(f'Total parameters: {sum(v.size for v in mlx_weights.values()):,}')
        
    except Exception as e:
        print(f'[ERROR] {type(e).__name__}: {e}')
        print('\n[INFO] If you see authentication errors, you may need to:')
        print('  1. Accept the pyannote model license at: https://huggingface.co/pyannote/segmentation-3.0')
        print('  2. Authenticate with Hugging Face: huggingface-cli login')
        sys.exit(1)


if __name__ == '__main__':
    # Convert segmentation model
    print('='*60)
    print('Converting pyannote/segmentation-3.0 to MLX')
    print('='*60)
    convert_model('pyannote/segmentation-3.0', 'models/segmentation_mlx')
    
    print('\n' + '='*60)
    print('Conversion complete!')
    print('='*60)
