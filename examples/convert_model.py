"""
Example: Convert PyTorch model to MLX
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from convert import convert_pyannote_model


def main():
    """
    Example conversion of pyannote model to MLX format
    """
    
    # Option 1: Convert from local PyTorch checkpoint
    pytorch_path = "path/to/pytorch/model.pt"
    mlx_path = "models/mlx_model"
    
    # Option 2: Convert from Hugging Face (requires authentication)
    # pytorch_path = "pyannote/segmentation"
    # mlx_path = "models/segmentation_mlx"
    
    print(f"Converting model from {pytorch_path} to {mlx_path}")
    print("This may take a few minutes...")
    
    try:
        convert_pyannote_model(
            pytorch_path=pytorch_path,
            mlx_path=mlx_path,
            model_type="segmentation",  # or "embedding" or "full"
            dtype="float16",
        )
        print(f"\nConversion successful! Model saved to {mlx_path}")
        
    except FileNotFoundError:
        print(f"\nError: Could not find model at {pytorch_path}")
        print("\nTo convert a pyannote model, you need to:")
        print("1. Download the model from Hugging Face (requires authentication)")
        print("2. Or provide a local PyTorch checkpoint path")
        print("\nExample with Hugging Face:")
        print("  from huggingface_hub import snapshot_download")
        print("  model_path = snapshot_download('pyannote/segmentation')")
        
    except Exception as e:
        print(f"\nConversion failed: {e}")


if __name__ == "__main__":
    main()
