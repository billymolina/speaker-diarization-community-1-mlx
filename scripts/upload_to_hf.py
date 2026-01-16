#!/usr/bin/env python3
"""
Upload pyannote/segmentation-3.0 MLX model to Hugging Face mlx-community
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
REPO_ID = "mlx-community/pyannote-segmentation-3.0-mlx"
MODEL_DIR = "models/segmentation_mlx"
SOURCE_REPO = "pyannote/segmentation-3.0"

def upload_model():
    """Upload the MLX model to Hugging Face Hub"""
    
    print("=" * 80)
    print("UPLOADING MLX MODEL TO HUGGING FACE")
    print("=" * 80)
    
    # Check if files exist
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {MODEL_DIR}")
        return
    
    weights_file = model_path / "weights.npz"
    if not weights_file.exists():
        print(f"❌ Error: Weights file not found: {weights_file}")
        return
    
    readme_file = model_path / "README.md"
    if not readme_file.exists():
        print(f"❌ Error: README.md not found: {readme_file}")
        return
    
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"❌ Error: config.json not found: {config_file}")
        return
    
    print(f"\n✅ Found all required files:")
    print(f"   - {weights_file.name} ({weights_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   - {readme_file.name}")
    print(f"   - {config_file.name}")
    
    # Get token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        print(f"\n❌ HF_TOKEN not set in .env file")
        print("\nPlease set your Hugging Face token:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Add to .env file: HF_TOKEN=your_token_here")
        return
    
    # Initialize Hugging Face API
    api = HfApi(token=hf_token)
    
    # Check authentication
    try:
        user_info = api.whoami()
        print(f"\n✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\nPlease check your HF_TOKEN in .env file")
        print("  Get token from: https://huggingface.co/settings/tokens")
        return
    
    # Create repository (if doesn't exist)
    print(f"\n📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return
    
    # Upload files
    print(f"\n⬆️  Uploading files from {MODEL_DIR}...")
    try:
        upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload MLX conversion of {SOURCE_REPO}"
        )
        print(f"\n✅ Upload complete!")
        print(f"\n🎉 Model published at: https://huggingface.co/{REPO_ID}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return
    
    # Print usage instructions
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\nTo use the model:")
    print("\n```python")
    print("from huggingface_hub import hf_hub_download")
    print("import mlx.core as mx")
    print()
    print("# Download weights")
    print(f"weights_path = hf_hub_download(")
    print(f'    repo_id="{REPO_ID}",')
    print(f'    filename="weights.npz"')
    print(f")")
    print()
    print("# Load model")
    print("weights = mx.load(weights_path)")
    print("```")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    upload_model()
