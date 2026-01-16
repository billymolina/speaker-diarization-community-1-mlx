"""
Upload the MLX ResNet34 speaker embedding model to Hugging Face mlx-community.

Prerequisites:
1. Install huggingface_hub: pip install huggingface_hub
2. Login to Hugging Face: huggingface-cli login
3. Have write access to mlx-community (or create under your own account first)
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_model():
    """Upload the model to Hugging Face."""

    # Configuration
    repo_name = "wespeaker-voxceleb-resnet34-LM"  # Model name
    organization = "mlx-community"  # Target organization
    repo_id = f"{organization}/{repo_name}"

    # Local model directory
    model_dir = Path("models/embedding_mlx")

    # Files to upload
    files_to_upload = [
        "weights.npz",           # Model weights
        "config.json",           # Model configuration
        "README.md",             # Model card
        "resnet_embedding.py",   # Model code
        "example_usage.py",      # Usage example
    ]

    print("="*70)
    print(" Uploading MLX ResNet34 to Hugging Face")
    print("="*70)

    # Verify all files exist
    print("\n[1] Verifying files...")
    for filename in files_to_upload:
        filepath = model_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            return
        print(f"  ✓ Found: {filename} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")

    # Initialize Hugging Face API
    print("\n[2] Initializing Hugging Face API...")
    api = HfApi()

    # Check if user is logged in
    try:
        user_info = api.whoami()
        print(f"  ✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"  ✗ Not logged in. Please run: huggingface-cli login")
        return

    # Create repository (if it doesn't exist)
    print(f"\n[3] Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"  ✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Error creating repository: {e}")
        print(f"\n  Note: If you don't have access to {organization}, try uploading to your own account first:")
        print(f"        Change 'organization' in this script to your HuggingFace username")
        return

    # Upload files
    print(f"\n[4] Uploading files to {repo_id}...")
    for filename in files_to_upload:
        filepath = model_dir / filename
        print(f"\n  Uploading {filename}...")

        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"    ✓ Uploaded successfully")
        except Exception as e:
            print(f"    ✗ Upload failed: {e}")
            return

    print("\n" + "="*70)
    print(" ✓ Upload completed successfully!")
    print("="*70)
    print(f"\nModel available at: https://huggingface.co/{repo_id}")
    print("\nUsage:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  weights_path = hf_hub_download(repo_id='{repo_id}', filename='weights.npz')")
    print(f"  model_code = hf_hub_download(repo_id='{repo_id}', filename='resnet_embedding.py')")


def upload_to_personal_account():
    """Alternative: Upload to your personal Hugging Face account first."""

    print("\n" + "="*70)
    print(" Alternative: Upload to Personal Account")
    print("="*70)
    print("\nIf you don't have write access to mlx-community, you can:")
    print("\n1. Edit this script and change:")
    print("   organization = 'mlx-community'")
    print("   to:")
    print("   organization = 'YOUR_USERNAME'")
    print("\n2. Re-run this script")
    print("\n3. Later, request the mlx-community admins to transfer the model")
    print("\n4. Or submit a PR to mlx-community with a link to your model")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        upload_to_personal_account()
    else:
        # Check if huggingface_hub is installed
        try:
            import huggingface_hub
            upload_model()
        except ImportError:
            print("Error: huggingface_hub not installed")
            print("\nPlease install it:")
            print("  pip install huggingface_hub")
            print("\nThen login:")
            print("  huggingface-cli login")
